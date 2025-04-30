# main.py

import time
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.config import Config
from env.dual_rl_env import DualRLEnv
from agents.tertiary.sac_agent import SACAgent
from agents.secondary.ia3c_agent import IA3CAgent
from utils.helper import Helper as helper
import os
from typing import List

# Create directory for models if it doesn't exist
os.makedirs("models", exist_ok=True)

def moving_average(data: List[float], window_size: int) -> List[float]:
    """Compute moving average using a fixed window size."""
    if len(data) < window_size:
        return [float("nan")] * len(data)
    return [float("nan")] * (window_size - 1) + [
        sum(data[i - window_size + 1:i + 1]) / window_size for i in range(window_size - 1, len(data))
    ]

class RewardPlotter:
    """Class to handle real-time reward plotting (only smoothened)"""
    def __init__(self, num_secondary_agents: int, smooth_window: int = 50):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.tertiary_rewards: List[float] = []
        self.secondary_rewards: List[List[float]] = [[] for _ in range(num_secondary_agents)]
        self.episodes: List[int] = []
        self.smooth_window = smooth_window

        # Create plots directory if it doesn't exist
        os.makedirs("plots", exist_ok=True)

        # Set up tertiary reward plot (only smoothened)
        self.ax1.set_title("Tertiary Agent Reward (Smoothed)")
        self.ax1.set_xlabel("Episode")
        self.ax1.set_ylabel("Reward")
        self.tertiary_smooth_line, = self.ax1.plot([], [], "r-", label=f"Smoothed ({smooth_window})")
        self.ax1.legend()

        # Set up secondary rewards plot (only smoothened)
        self.ax2.set_title("Secondary Agents Rewards (Smoothed)")
        self.ax2.set_xlabel("Episode")
        self.ax2.set_ylabel("Reward")
        self.secondary_smooth_lines = [self.ax2.plot([], [], label=f"Agent {i} Smoothed")[0] for i in range(num_secondary_agents)]
        self.ax2.legend()

        plt.tight_layout()
        plt.ion()  # Turn on interactive mode

    def update(self, episode: int, tertiary_reward: float, secondary_rewards: List[float]):
        self.episodes.append(episode)
        self.tertiary_rewards.append(tertiary_reward)
        for i, reward in enumerate(secondary_rewards):
            self.secondary_rewards[i].append(reward)

        # Update tertiary plot (only smoothened)
        smooth_ter = moving_average(self.tertiary_rewards, self.smooth_window)
        self.tertiary_smooth_line.set_data(self.episodes, smooth_ter)
        self.ax1.relim()
        self.ax1.autoscale_view()

        # Update secondary plots (only smoothened)
        for i, line in enumerate(self.secondary_smooth_lines):
            smooth_sec = moving_average(self.secondary_rewards[i], self.smooth_window)
            line.set_data(self.episodes, smooth_sec)
        self.ax2.relim()
        self.ax2.autoscale_view()

        plt.draw()
        plt.pause(0.01)
        # Save plot every 1 episode (as before)
        if episode % 1 == 0:
            self.save_plot(episode)

    def save_plot(self, episode):
        """Save the current plot to a file"""
        filename = f"plots/rewards_episode.png"
        self.fig.savefig(filename, dpi=300, bbox_inches="tight")
        logging.info(f"Plot saved to {filename}")
        
    def save_final_plot(self):
        """Save the final plot"""
        self.save_plot("final")
        # Also save the reward data for potential future analysis
        np.save("plots/tertiary_rewards.npy", np.array(self.tertiary_rewards))
        np.save("plots/secondary_rewards.npy", np.array(self.secondary_rewards))
        np.save("plots/episodes.npy", np.array(self.episodes))
        logging.info("Final plot and reward data saved to plots/ directory")

def save_models(tertiary_agent, secondary_agents, episode):
    """Save models at regular intervals"""
    if episode % 10 == 0:  # Save every 10 episodes
        tertiary_agent.save(f"models/tertiary_agent_episode")
        for i, agent in enumerate(secondary_agents):
            agent.save(f"models/secondary_agent_{i}_episode")

def main():
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

    num_secondary_steps = getattr(config, "num_secondary_steps", 5)
    num_episodes = getattr(config, "num_episodes", 1000)

    for ep in range(num_episodes):
        logging.info(f"Starting episode {ep + 1}/{num_episodes}")
        done = False
        episode_reward = 0.0
        secondary_episode_rewards = [0.0 for _ in secondary_agents]

        while not done:
            logging.info("Current time step: %d", time_step)
            ter_state = state.get("tertiary", None)
            ter_action, ter_log_prob, ter_value = tertiary_agent.select_action(ter_state, dual_env.tertiary_env.switch_set)
            next_state, ter_rewards, ter_done, ter_info = dual_env.step(ter_action, time_step)

            sec_total_reward = 0.0
            sec_state = state.get("secondary", None)
            for _ in range(num_secondary_steps):
                secondary_actions = []
                secondary_log_probs = []
                new_sec_states = []
                for i, agent in enumerate(secondary_agents):
                    agent_state = sec_state[i]
                    sec_action, sec_log_prob, sec_value = agent.select_action(agent_state)
                    secondary_actions.append(sec_action)
                    secondary_log_probs.append(sec_log_prob)

                new_sec_state, sec_rewards, sec_done, sec_info = dual_env.secondary_env.step(secondary_actions)
                sec_total_reward += np.mean(sec_rewards)
                
                # Accumulate rewards for each secondary agent
                for i, reward in enumerate(sec_rewards):
                    secondary_episode_rewards[i] += reward
                    
                sec_state = new_sec_state
                if sec_done:
                    break

            aggregated_sec_reward = sec_total_reward / num_secondary_steps
            # Normalize secondary rewards
            overall_reward = ter_rewards + config.alpha_sec * aggregated_sec_reward
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
                    state=state["secondary"][i],
                    action=secondary_actions[i],
                    reward=sec_rewards[i],
                    next_state=sec_state[i],
                    log_prob=secondary_log_probs[i],
                    done=ter_done
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