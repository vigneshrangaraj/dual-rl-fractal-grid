# main.py

import time
import logging
import numpy as np
import torch
from utils.config import Config
from env.dual_rl_env import DualRLEnv
from agents.tertiary.sac_agent import SACAgent
from agents.secondary.ia3c_agent import IA3CAgent
from utils.helper import Helper as helper


def main():
    # Load configuration parameters
    global sec_rewards, sec_done
    config = Config()
    logging.basicConfig(level=logging.INFO)

    # Initialize the dual-level environment
    dual_env = DualRLEnv(config)

    # Reset environment to get an initial tertiary state and flatten it
    init_state = dual_env.reset()  # overall state: {"tertiary": ter_state, "secondary": sec_state}
    tertiary_state = init_state.get("tertiary", {})
    time_step = tertiary_state.get("timestep", 0)
    flat_state = helper.flatten_tertiary_state(tertiary_state)
    state_dim = flat_state.shape[0]

    # For the tertiary action, we assume a vector of dimension 3:
    # [dispatch_power, battery_operation, energy_shared]
    num_microgrids = getattr(config, "num_microgrids", 1)
    action_dim = getattr(config, "sac_action_dim", 2) * num_microgrids + dual_env.tertiary_env.switches

    # Instantiate the tertiary SAC agent with the determined state and action dimensions.
    tertiary_agent = SACAgent(state_dim, action_dim, config)

    # For secondary agents, instantiate one agent for each DER as specified in config
    num_secondary = getattr(config, "num_secondary_agents", 5)

    secondary_agents = []

    for i in range(num_microgrids):
        for j in range(num_secondary):
            secondary_agent = IA3CAgent(config, agent_id=j, microgrid_id=i,inverter_id=j)
            secondary_agents.append(secondary_agent)

    # Number of secondary steps per tertiary step
    num_secondary_steps = getattr(config, "num_secondary_steps", 5)

    num_episodes = getattr(config, "num_episodes", 1000)

    for ep in range(num_episodes):
        logging.info(f"Starting episode {ep + 1}/{num_episodes}")
        state = dual_env.reset()  # state is a dict: {"tertiary": ter_state, "secondary": sec_state}
        done = False
        episode_reward = 0.0

        while not done:
            logging.info("Current time step: %d", time_step)
            # Tertiary agent selects a global action based on aggregated state
            ter_state = state.get("tertiary", None)
            ter_action, ter_log_prob, ter_value = tertiary_agent.select_action(ter_state, dual_env.tertiary_env.switch_set)

            # For secondary updates, perform a loop of secondary steps
            sec_total_reward = 0.0
            sec_state = state.get("secondary", None)
            for _ in range(num_secondary_steps):
                # For each secondary agent, get its local state from the secondary environment.
                # We assume sec_state is a list with one state dict per agent.
                secondary_actions = []
                secondary_log_probs = []
                new_sec_states = []
                for i, agent in enumerate(secondary_agents):
                    # Each secondary agent obtains its action based on its local state.
                    # (State may include local voltage, phase angle, d–q currents, etc.)
                    agent_state = sec_state[i]
                    sec_action, sec_log_prob, sec_value = agent.select_action(agent_state)
                    secondary_actions.append(sec_action)
                    secondary_log_probs.append(sec_log_prob)

                # Update the secondary environment for one step using the selected actions.
                new_sec_state, sec_rewards, sec_done, sec_info = dual_env.secondary_env.step(secondary_actions)
                sec_total_reward += np.mean(sec_rewards)
                sec_state = new_sec_state  # update secondary state for the next secondary step
                if sec_done:
                    break
            # Average the secondary reward over the steps
            aggregated_sec_reward = sec_total_reward / num_secondary_steps

            # Now update the tertiary environment with its action.
            next_state, ter_rewards, ter_done, ter_info = dual_env.step(ter_action, time_step)

            # Compute the overall reward (e.g., as a weighted sum of tertiary and secondary rewards)
            overall_reward = ter_rewards + config.alpha_sec * aggregated_sec_reward
            episode_reward += overall_reward

            # Update the tertiary agent (one-step update; in practice, use a replay buffer)
            tertiary_agent.learn(
                state=ter_state,
                action=ter_action,
                log_prob=ter_log_prob,
                reward=overall_reward,
                next_state=next_state,
                done=ter_done
            )

            # Update secondary agents
            for i, agent in enumerate(secondary_agents):
                # Placeholder: Use each agent's transition from its secondary state.
                # You might need to accumulate trajectories and update with on-policy methods.
                agent.learn(
                    state=state["secondary"][i],
                    action=secondary_actions[i],
                    reward=sec_rewards[i],
                    next_state=sec_state[i],
                    log_prob=secondary_log_probs[i],
                    done=ter_done
                )

            # Update the time step
            time_step += 1
            state = {
                "tertiary": next_state,
                "secondary": sec_state
            }

            done = ter_done or sec_done

        logging.info(f"Episode {ep + 1} finished with overall reward: {episode_reward:.2f}")

    tertiary_agent.save("models/tertiary_agent")
    for i, agent in enumerate(secondary_agents):
        agent.save(f"models/secondary_agent_{i}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Log the error message
        logging.basicConfig(level=logging.ERROR)
        logging.error("An error occurred: %s", e)
        # Optionally, you can also log the traceback
        import traceback
        logging.error("Traceback: %s", traceback.format_exc())
        logging.error("Error in main: %s", e)