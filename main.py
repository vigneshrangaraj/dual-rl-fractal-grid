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
    action_dim = getattr(config, "sac_action_dim", 2) * num_microgrids + dual_env.tertiary_env.switches

    # Instantiate the tertiary SAC agent
    tertiary_agent = SACAgent(state_dim, action_dim, config)

    # For secondary agents
    num_secondary = getattr(config, "num_secondary_agents", 5)
    secondary_agents = []

    for i in range(num_microgrids):
        for j in range(num_secondary):
            secondary_agent = IA3CAgent(config, agent_id=j, microgrid_id=i, inverter_id=j)
            secondary_agents.append(secondary_agent)

    # Initialize the plotter
    plotter = RewardPlotter(len(secondary_agents))

    actions_plotter = ActionStatePlotter(
        num_secondary_agents=len(secondary_agents),
        num_tertiary_actions=3,
        num_tertiary_states=len(flat_state),
        num_secondary_states=len(state["secondary"][0]),
        smooth_window=50
    )

    num_secondary_steps = getattr(config, "num_secondary_steps", 5)
    num_episodes = getattr(config, "num_episodes", 1000)

    for ep in range(num_episodes):
        print(f"=================Starting episode {ep + 1}/{num_episodes}")
        done = False
        episode_reward = 0.0
        secondary_episode_rewards = [0.0 for _ in secondary_agents]

        while not done:
            ter_state = state.get("tertiary", None)
            ter_action, ter_log_prob, ter_value = tertiary_agent.select_action(ter_state, dual_env.tertiary_env.switch_set)
            next_state, ter_rewards, ter_done, ter_info = dual_env.step(ter_action, time_step)

            sec_total_reward = 0.0
            sec_state = state.get("secondary", None)
            convergences = []
            for _ in range(num_secondary_steps):
                secondary_actions = []
                secondary_log_probs = []
                new_sec_states = []
                for i, agent in enumerate(secondary_agents):
                    agent_state = sec_state[i]
                    sec_action, sec_log_prob, sec_value = agent.select_action(agent_state)
                    secondary_actions.append(sec_action)
                    secondary_log_probs.append(sec_log_prob)

                new_sec_state, sec_rewards, sec_done, sec_info = dual_env.secondary_env.step(secondary_actions,
                                                                                             dual_env.tertiary_env.microgrids,
                                                                                             ter_action.get("tie_lines", None))
                sec_total_reward += np.mean(sec_rewards)
                convergences.append(sec_info.get("is_converged", False))
                
                # Accumulate rewards for each secondary agent
                for i, reward in enumerate(sec_rewards):
                    secondary_episode_rewards[i] += reward
                    
                sec_state = new_sec_state
                if sec_done:
                    dual_env.secondary_env.time_step = 0
                    break

            for micrigrid in dual_env.tertiary_env.microgrids:
                micrigrid.pf_net = sec_info.get("net", None)

            aggregated_sec_reward = sec_total_reward / num_secondary_steps
            # Normalize secondary rewards
            overall_reward = ter_rewards + config.alpha_sec * aggregated_sec_reward

            # Check for how much was borrwed from ext_grid.. if it is negative, it means we are borrowing
            # from the grid if positive, we are giving back to the grid
            # reward giving back and selling instead of borrowing
            p_mw = sec_info.get("new_energy", None)
            if p_mw is not None:
                overall_reward += config.beta_ext_grid * p_mw

            # Also check for load deficit
            is_deficient, deficit = dual_env.tertiary_env.check_power_deficit(sec_info.get("net", None))
            if is_deficient:
                overall_reward -= config.beta_deficient * ( deficit * 100)
            episode_reward += overall_reward

            # Update agents
            tertiary_agent.learn(
                state=ter_state,
                action=ter_action,
                log_prob=ter_log_prob,
                reward=overall_reward,
                next_state=next_state,
                done=ter_done
            )

            for i, agent in enumerate(secondary_agents):
                agent.learn(
                    state=sec_state[i],
                    log_prob=secondary_log_probs[i],
                    reward=sec_rewards[i],
                    next_state=sec_state[i],
                    done=sec_done
                )

            time_step += 1
            state = {
                "tertiary": next_state,
                "secondary": sec_state
            }

            # log progress and actions
            print(f"Episode {ep + 1}, Step {time_step}: Tertiary action: {ter_action}, "
                         f"Secondary actions: {secondary_actions}, Overall reward: {overall_reward:.2f}")
            print(f"Overall step: Episode {ep + 1}, Step {time_step}: Tertiary reward: {ter_rewards:.2f}, "
                         f"Secondary rewards: {sec_rewards}, Overall reward: {overall_reward:.2f}")
            done = ter_done

        # Normalize secondary rewards by number of steps
        secondary_episode_rewards = [r / num_secondary_steps for r in secondary_episode_rewards]
        
        # Update plots
        plotter.update(ep + 1, episode_reward, secondary_episode_rewards)
        #actions_plotter.update(ep + 1, ter_action, secondary_actions, ter_state, sec_state)
        
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

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.basicConfig(level=logging.ERROR)
        logging.error("An error occurred: %s", e)
        import traceback
        logging.error("Traceback: %s", traceback.format_exc())
        logging.error("Error in main: %s", e)