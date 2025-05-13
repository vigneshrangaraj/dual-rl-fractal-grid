import logging
import os
from typing import List

import numpy as np
from matplotlib import pyplot as plt


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
        self.secondary_smooth_lines = [self.ax2.plot([], [], label=f"Agent {i} Smoothed")[0] for i in
                                       range(num_secondary_agents)]
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