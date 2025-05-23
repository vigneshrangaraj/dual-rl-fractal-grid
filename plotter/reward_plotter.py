import logging
import os
from typing import List
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

def moving_average(data: List[float], window_size: int) -> List[float]:
    if len(data) < window_size:
        return [float("nan")] * len(data)
    return [float("nan")] * (window_size - 1) + [
        sum(data[i - window_size + 1:i + 1]) / window_size for i in range(window_size - 1, len(data))
    ]

def moving_std(data: List[float], window_size: int) -> List[float]:
    if len(data) < window_size:
        return [float("nan")] * len(data)
    return [float("nan")] * (window_size - 1) + [
        np.std(data[i - window_size + 1:i + 1]) for i in range(window_size - 1, len(data))
    ]

class RewardPlotter:
    def __init__(self, num_secondary_agents: int, smooth_window: int = 50):
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 8))
        self.tertiary_rewards: List[float] = []
        self.secondary_rewards: List[List[float]] = [[] for _ in range(num_secondary_agents)]
        self.episodes: List[int] = []
        self.smooth_window = smooth_window
        self.secondary_voltage_violations = [[] for _ in range(num_secondary_agents)]

        self.tertiary_fill = None
        self.secondary_fills = [None for _ in range(num_secondary_agents)]
        self.violation_fills = [None for _ in range(num_secondary_agents)]

        os.makedirs("plots", exist_ok=True)

        self.ax1.set_title(r"Tertiary Agent Reward (Smoothed)")
        self.ax1.set_xlabel(r"Episode")
        self.ax1.set_ylabel(r"Reward")
        self.tertiary_smooth_line, = self.ax1.plot([], [], "r-", label=f"Smoothed ({smooth_window})")
        self.ax1.legend()

        self.ax2.set_title(r"Secondary Agents Rewards (Smoothed)")
        self.ax2.set_xlabel(r"Episode")
        self.ax2.set_ylabel(r"Reward")
        self.secondary_smooth_lines = [self.ax2.plot([], [], label=rf"Agent {i} Smoothed")[0] for i in range(num_secondary_agents)]
        self.ax2.legend()

        self.ax3.set_title(r"Secondary Agents Voltage Violations (Smoothed)")
        self.ax3.set_xlabel(r"Episode")
        self.ax3.set_ylabel(r"Violations")
        self.secondary_voltage_violation_lines = [self.ax3.plot([], [], label=rf"Agent {i}")[0] for i in range(num_secondary_agents)]
        self.ax3.legend()

        plt.tight_layout()
        plt.ion()

    def update(self, episode: int, tertiary_reward: float, secondary_rewards: List[float],
               secondary_voltage_violations: List[float]):
        self.episodes.append(episode)
        self.tertiary_rewards.append(tertiary_reward)
        for i, reward in enumerate(secondary_rewards):
            self.secondary_rewards[i].append(reward)
        for i, violation in enumerate(secondary_voltage_violations):
            self.secondary_voltage_violations[i].append(violation)

        smooth_ter = moving_average(self.tertiary_rewards, self.smooth_window)
        std_ter = moving_std(self.tertiary_rewards, self.smooth_window)
        if len(self.episodes) == len(smooth_ter):
            self.tertiary_smooth_line.set_data(self.episodes, smooth_ter)
            if self.tertiary_fill:
                self.tertiary_fill.remove()
            self.tertiary_fill = self.ax1.fill_between(self.episodes,
                                                       np.array(smooth_ter) - np.array(std_ter),
                                                       np.array(smooth_ter) + np.array(std_ter),
                                                       alpha=0.1, color='red')
        self.ax1.relim()
        self.ax1.autoscale_view()

        for i, line in enumerate(self.secondary_smooth_lines):
            smooth_sec = moving_average(self.secondary_rewards[i], self.smooth_window)
            std_sec = moving_std(self.secondary_rewards[i], self.smooth_window)
            if len(self.episodes) == len(smooth_sec):
                line.set_data(self.episodes, smooth_sec)
                if self.secondary_fills[i]:
                    self.secondary_fills[i].remove()
                self.secondary_fills[i] = self.ax2.fill_between(self.episodes,
                                                                np.array(smooth_sec) - np.array(std_sec),
                                                                np.array(smooth_sec) + np.array(std_sec),
                                                                alpha=0.1)
        self.ax2.relim()
        self.ax2.autoscale_view()

        for i, line in enumerate(self.secondary_voltage_violation_lines):
            smooth_viol = moving_average(self.secondary_voltage_violations[i], self.smooth_window)
            std_viol = moving_std(self.secondary_voltage_violations[i], self.smooth_window)
            if len(self.episodes) == len(smooth_viol):
                line.set_data(self.episodes, smooth_viol)
                if self.violation_fills[i]:
                    self.violation_fills[i].remove()
                self.violation_fills[i] = self.ax3.fill_between(self.episodes,
                                                                np.array(smooth_viol) - np.array(std_viol),
                                                                np.array(smooth_viol) + np.array(std_viol),
                                                                alpha=0.1)
        self.ax3.relim()
        self.ax3.autoscale_view()

        plt.draw()
        plt.pause(0.01)
        if episode % 1 == 0:
            self.save_plot(episode)

    def save_plot(self, episode):
        filename = f"plots/rewards_episode.png"
        self.fig.savefig(filename, dpi=300, bbox_inches="tight")
        logging.info(f"Plot saved to {filename}")

    def save_final_plot(self):
        self.save_plot("final")
        np.save("plots/tertiary_rewards.npy", np.array(self.tertiary_rewards))
        np.save("plots/secondary_rewards.npy", np.array(self.secondary_rewards))
        np.save("plots/episodes.npy", np.array(self.episodes))
        logging.info("Final plot and reward data saved to plots/ directory")