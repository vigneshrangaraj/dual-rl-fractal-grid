# main.py

import time
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt

from plotter.reward_plotter import RewardPlotter
from plotter.action_state_plotter import ActionStatePlotter
from utils.config import Config
from env.dual_rl_env import DualRLEnv
from agents.tertiary.sac_agent import SACAgent
from agents.secondary.ia3c_agent import IA3CAgent
from utils.helper import Helper as helper
from utils.data_logger import EpisodeDataLogger
import os
from typing import List

# Create directory for models if it doesn't exist
os.makedirs("models", exist_ok=True)


def save_models(tertiary_agent, secondary_agents, episode):
    """Save models at regular intervals"""
    if episode % 10 == 0:  # Save every 10 episodes
        tertiary_agent.save(f"models/tertiary_agent_episode")
        for i, agent in enumerate(secondary_agents):
            agent.save(f"models/secondary_agent_{i}_episode")


def main(tertiary_action=None):
    # Clean up previous episode data file if it exists
    if os.path.exists("episode_data.csv"):
        os.remove("episode_data.csv")
        print("Cleaned up previous episode_data.csv file")
    
    # Load configuration parameters
    global sec_rewards, sec_done
    config = Config()

    # Initialize the dual-level environment
    dual_env = DualRLEnv(config)

    # Reset environment to get an initial tertiary state and flatten it
    state = dual_env.reset()
    tertiary_state = state.get("tertiary", {})
    time_step = tertiary_state.get("timestep", 0)
    flat_state = helper.flatten_tertiary_state(tertiary_state)
    state_dim = flat_state.shape[0]

    num_microgrids = getattr(config, "num_microgrids", 1)
    num_der_total = getattr(config, "num_der_total", 4)
    # New action space: configurable DER actions + 1 BESS action + switching actions per microgrid
    action_dim = ((num_der_total + 1) * num_microgrids) + dual_env.tertiary_env.switches

    # Instantiate the tertiary SAC agent
    tertiary_agent = SACAgent(state_dim, action_dim, config)

    # For secondary agents
    num_secondary = dual_env.tertiary_env.microgrids[0].num_secondary_agents
    secondary_agents = []

    for i in range(num_microgrids):
        for j in range(num_secondary):
            secondary_agent = IA3CAgent(config, agent_id=j, microgrid_id=i, inverter_id=j)
            secondary_agents.append(secondary_agent)

    # Initialize the plotter
    plotter = RewardPlotter(len(secondary_agents))

    actions_plotter = ActionStatePlotter(
        num_secondary_agents=len(secondary_agents),
        num_tertiary_actions=num_der_total + 1,  # DER actions + 1 BESS action
        num_tertiary_states=len(flat_state),
        num_secondary_states=len(state["secondary"][0]),
    )
    
    # Initialize data logger
    data_logger = EpisodeDataLogger("episode_data.csv")

    num_secondary_steps = getattr(config, "num_secondary_steps", 5)
    num_episodes = getattr(config, "num_episodes", 1000)

    for ep in range(num_episodes):
        print(f"=================Starting episode {ep + 1}/{num_episodes}")
        done = False
        episode_reward = 0.0
        time_step = 0
        secondary_episode_rewards = [0.0 for _ in secondary_agents]
        secondary_episode_voltage_violations = [0 for _ in secondary_agents]
        bess_soc_for_day = []

        # Initialize hidden and cell states
        h_states = [torch.zeros(1, 1, 64) for _ in secondary_agents]
        c_states = [torch.zeros(1, 1, 64) for _ in secondary_agents]
        next_h_states = []
        next_c_states = []
        aggregated_ter_state = state.get("tertiary", None)

        while not done:
            context_vectors = []

            if (config.use_lstm):
                for i in range(len(secondary_agents)):
                    q_i = h_states[i].squeeze(0)
                    others = [h_states[j].squeeze(0) for j in range(len(secondary_agents)) if j != i]
                    context = compute_attention_context(q_i, others)
                    context_vectors.append(context)

            ter_state = state.get("tertiary", None)
            micrigrid = ter_state.get("microgrids", None)[0]
            bess_soc_for_day.append(micrigrid.get("bess_soc", None))
            ter_action, ter_log_prob, ter_value = tertiary_agent.select_action(ter_state, dual_env.tertiary_env.switch_set)
            # replace the last value of ter_action with simulate_battery_operation -- for testing only
            # if ter_action is not None:
            #     mg_actions = ter_action.get("microgrids", None)
            #     if mg_actions is not None and len(mg_actions) > 0:
            #         mg_actions[0]["battery_operation"] = dual_env.tertiary_env.simulate_bess_action(time_step)
            next_state, ter_rewards, ter_done, ter_info = dual_env.step(ter_action, time_step)


            # add to aggregated state
            helper.add_to_aggregated_state(aggregated_ter_state, next_state)

            sec_total_reward = 0.0
            sec_state = state.get("secondary", None)
            convergences = []
            steps_secondary_run = 0
            this_secondary_episode_rewards = [0.0 for _ in secondary_agents]
            for _ in range(num_secondary_steps):
                steps_secondary_run += 1
                secondary_actions = []
                secondary_log_probs = []
                new_sec_states = []
                if config.use_lstm:
                    for i, agent in enumerate(secondary_agents):
                        agent_state = sec_state[i]
                        sec_action, sec_log_prob, next_h, next_c = agent.select_action(agent_state,
                                                                                       (h_states[i], c_states[i]),
                                                                                       context_vectors[i].view(1, -1))
                        secondary_actions.append(sec_action)
                        secondary_log_probs.append(sec_log_prob)

                        next_h_states.append(next_h)
                        next_c_states.append(next_c)

                        h_states[i] = next_h
                        c_states[i] = next_c
                else:
                    for i, agent in enumerate(secondary_agents):
                        agent_state = sec_state[i]
                        sec_action, sec_log_prob, sec_value = agent.select_action(agent_state)
                        secondary_actions.append(sec_action)
                        secondary_log_probs.append(sec_log_prob)

                new_sec_state, sec_rewards, sec_done, sec_info = dual_env.secondary_env.step(secondary_actions,
                                                                                             dual_env.tertiary_env,
                                                                                             ter_action.get("tie_lines",
                                                                                                            None))
                sec_total_reward += np.mean(sec_rewards)
                convergences.append(sec_info.get("is_converged", False))

                # Accumulate rewards for each secondary agent
                for i, reward in enumerate(sec_rewards):
                    this_secondary_episode_rewards[i] += reward
                for i, violation in enumerate(sec_info.get("violations", [])):
                    secondary_episode_voltage_violations[i] += violation

                sec_state = new_sec_state

                if config.use_lstm:
                    for i, agent in enumerate(secondary_agents):
                        agent.learn(
                            state=sec_state[i],
                            log_prob=secondary_log_probs[i],
                            reward=sec_rewards[i],
                            next_state=sec_state[i],
                            done=sec_done,
                            hidden=(h_states[i], c_states[i]),
                            next_hidden=(next_h_states[i], next_c_states[i]),
                            context_vector=context_vectors[i]
                        )
                else:
                    for i, agent in enumerate(secondary_agents):
                        agent.learn(
                            state=sec_state[i],
                            log_prob=secondary_log_probs[i],
                            reward=sec_rewards[i],
                            next_state=sec_state[i],
                            done=sec_done
                        )

                if sec_done:
                    dual_env.secondary_env.time_step = 0
                    break

                # done secondary steps

            for micrigrid in dual_env.tertiary_env.microgrids:
                micrigrid.pf_net = sec_info.get("net", None)

            aggregated_sec_reward = sec_total_reward / steps_secondary_run
            # Normalize secondary rewards
            overall_reward = ter_rewards + config.alpha_sec * aggregated_sec_reward

            for i, reward in enumerate(this_secondary_episode_rewards):
                reward = reward / steps_secondary_run
                secondary_episode_rewards[i] += reward

            # Check for how much was borrowed from ext_grid.. if it is negative, it means we are selling
            # from the grid if positive, ext_grid is selling to the grid
            # Use temporal pricing for economic incentives
            p_mw = sec_info.get("new_energy", None)
            if p_mw is not None:
                # Get current temporal prices
                current_prices = config.get_temporal_prices(time_step)
                current_buy_price = current_prices["buy_price"]
                current_sell_price = current_prices["sell_price"]
                
                if p_mw > 0:
                    # We are selling to the grid - reward based on current sell price
                    overall_reward -= current_buy_price * p_mw
                else:
                    # We are borrowing from the grid - penalty based on current buy price
                    overall_reward += current_sell_price * abs(p_mw)

            episode_reward += overall_reward

            tertiary_agent.remember(ter_state, ter_action, ter_log_prob, overall_reward, next_state, ter_done)
            # Update agents
            tertiary_agent.learn()

            # Log step data
            data_logger.log_step_data(
                episode=ep + 1,
                step=time_step,
                hour=time_step,
                tertiary_env=dual_env.tertiary_env,
                secondary_env=dual_env.secondary_env,
                secondary_rewards=sec_rewards,
                overall_reward=overall_reward,
                config=config
            )

            time_step += 1
            state = {
                "tertiary": next_state,
                "secondary": sec_state
            }

            # log progress and actions
            print(f"Episode {ep + 1}, Step {time_step}: Tertiary action: {ter_action}, Tertiary_state: {ter_state}, "
                  f"Secondary actions: {secondary_actions}, Overall reward: {overall_reward:.2f}")
            print(f"Overall step: Episode {ep + 1}, Step {time_step}: Tertiary reward: {ter_rewards:.2f}, "
                  f"Secondary rewards: {sec_rewards}, Overall reward: {overall_reward:.2f}")
            done = ter_done

        # print bess soc
        print(f"BESS SOC for day: {bess_soc_for_day}")

        # Normalize secondary rewards by number of steps
        secondary_episode_rewards = [r / 24 for r in secondary_episode_rewards]

        # log progress and actions for entire day
        print(f"Episode {ep + 1} finished with total reward: {episode_reward:.2f}")
        print(f"Secondary episode rewards: {secondary_episode_rewards}")

        # End episode and export data to CSV
        data_logger.end_episode()
        
        # Get episode summary
        episode_summary = data_logger.get_episode_summary(ep + 1)
        if episode_summary:
            print(f"Episode {ep + 1} Summary:")
            print(f"  Avg BESS SOC: {episode_summary['avg_bess_soc']:.3f}")
            print(f"  BESS SOC Range: {episode_summary['min_bess_soc']:.3f} - {episode_summary['max_bess_soc']:.3f}")
            print(f"  Total DER Generation: {episode_summary['total_der_generation']:.2f} MWh")
            print(f"  Total Load: {episode_summary['total_load']:.2f} MWh")
            print(f"  Avg Grid Power: {episode_summary['avg_grid_power']:.2f} MW")
            print(f"  Total Overall Reward: {episode_summary['total_overall_reward']:.2f}")
        
        # Update plots
        actions_plotter.update(ep + 1, ter_action, secondary_actions, aggregated_ter_state, sec_state)
        plotter.update(ep + 1, episode_reward, secondary_episode_rewards, secondary_episode_voltage_violations)

        # Save models periodically
        save_models(tertiary_agent, secondary_agents, ep + 1)

        # log ter state
        print(f"Tertiary state: {state.get("tertiary")}")
        # log sec state
        print(f"Secondary state: {state.get("secondary")}")

        print(f"Episode {ep + 1} finished with overall reward: {episode_reward:.2f}")

    # Save final models
    tertiary_agent.save("models/tertiary_agent_final")
    for i, agent in enumerate(secondary_agents):
        agent.save(f"models/secondary_agent_{i}_final")

    # Save final plot and reward data
    plotter.save_final_plot()

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the plot window open


def compute_attention_context(q_i, others, scale=True):
    """
    Compute attention-based context from other hidden states
    q_i: (1, d), others: list of (1, d)
    """
    if not others:
        return torch.zeros_like(q_i)

    k = torch.stack(others)  # (N-1, 1, d)
    v = k.clone()  # (N-1, 1, d)
    q = q_i.unsqueeze(0).unsqueeze(0)  # (1, 1, d)

    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1))  # (1, 1, N-1)
    if scale:
        scores = scores / (d_k ** 0.5)
    weights = torch.softmax(scores, dim=-1)  # (1, 1, N-1)
    context = torch.matmul(weights, v).squeeze(0).squeeze(0)  # (d,)
    return context


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.basicConfig(level=logging.ERROR)
        logging.error("An error occurred: %s", e)
        import traceback

        logging.error("Traceback: %s", traceback.format_exc())
        logging.error("Error in main: %s", e)