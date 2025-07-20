#!/usr/bin/env python3
"""
Centralized Reward Plotter for Centralized SAC Agent.
This module provides plotting functionality specifically for the centralized SAC agent.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional

class CentralizedRewardPlotter:
    """
    Reward plotter specifically designed for centralized SAC agent.
    Tracks and plots rewards, BESS SOC, and other metrics for the centralized agent.
    """
    
    def __init__(self, save_dir: str = "plots"):
        """
        Initialize the centralized reward plotter.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        self.episode_rewards = []
        self.avg_bess_soc = []
        self.final_bess_soc = []
        self.episode_steps = []
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize figure for real-time plotting
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle("Centralized SAC Agent Training Progress", fontsize=16)
        
    def update(self, episode_reward: float, avg_bess_soc: float, final_bess_soc: float, steps: int):
        """
        Update the plotter with new episode data.
        
        Args:
            episode_reward: Total reward for the episode
            avg_bess_soc: Average BESS SOC during the episode
            final_bess_soc: Final BESS SOC at the end of the episode
            steps: Number of steps in the episode
        """
        self.episode_rewards.append(episode_reward)
        self.avg_bess_soc.append(avg_bess_soc)
        self.final_bess_soc.append(final_bess_soc)
        self.episode_steps.append(steps)
        
    def plot_rewards(self, window_size: int = 100):
        """
        Plot the reward progression with moving average.
        
        Args:
            window_size: Size of the moving average window
        """
        if not self.episode_rewards:
            print("No reward data to plot")
            return
            
        episodes = list(range(1, len(self.episode_rewards) + 1))
        
        # Clear the first subplot
        self.axes[0, 0].clear()
        
        # Plot raw rewards
        self.axes[0, 0].plot(episodes, self.episode_rewards, alpha=0.6, color='blue', label='Raw Rewards')
        
        # Plot moving average if enough data
        if len(self.episode_rewards) >= window_size:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window_size)/window_size, mode='valid')
            moving_avg_episodes = list(range(window_size, len(self.episode_rewards) + 1))
            self.axes[0, 0].plot(moving_avg_episodes, moving_avg, color='red', linewidth=2, label=f'Moving Average ({window_size})')
        
        self.axes[0, 0].set_title("Episode Rewards")
        self.axes[0, 0].set_xlabel("Episode")
        self.axes[0, 0].set_ylabel("Total Reward")
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True, alpha=0.3)
        
    def plot_bess_soc(self):
        """Plot BESS SOC progression."""
        if not self.avg_bess_soc:
            print("No BESS SOC data to plot")
            return
            
        episodes = list(range(1, len(self.avg_bess_soc) + 1))
        
        # Clear the second subplot
        self.axes[0, 1].clear()
        
        # Plot average and final BESS SOC
        self.axes[0, 1].plot(episodes, self.avg_bess_soc, color='green', label='Average BESS SOC', linewidth=2)
        self.axes[0, 1].plot(episodes, self.final_bess_soc, color='orange', label='Final BESS SOC', linewidth=2)
        
        # Add horizontal lines for SOC bounds
        self.axes[0, 1].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Min SOC (0.1)')
        self.axes[0, 1].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Max SOC (0.9)')
        
        self.axes[0, 1].set_title("BESS State of Charge")
        self.axes[0, 1].set_xlabel("Episode")
        self.axes[0, 1].set_ylabel("SOC")
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True, alpha=0.3)
        self.axes[0, 1].set_ylim(0, 1)
        
    def plot_episode_steps(self):
        """Plot episode length progression."""
        if not self.episode_steps:
            print("No episode steps data to plot")
            return
            
        episodes = list(range(1, len(self.episode_steps) + 1))
        
        # Clear the third subplot
        self.axes[1, 0].clear()
        
        self.axes[1, 0].plot(episodes, self.episode_steps, color='purple', linewidth=2)
        self.axes[1, 0].set_title("Episode Length")
        self.axes[1, 0].set_xlabel("Episode")
        self.axes[1, 0].set_ylabel("Steps")
        self.axes[1, 0].grid(True, alpha=0.3)
        
    def plot_reward_distribution(self):
        """Plot reward distribution histogram."""
        if not self.episode_rewards:
            print("No reward data to plot")
            return
            
        # Clear the fourth subplot
        self.axes[1, 1].clear()
        
        self.axes[1, 1].hist(self.episode_rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        self.axes[1, 1].set_title("Reward Distribution")
        self.axes[1, 1].set_xlabel("Reward")
        self.axes[1, 1].set_ylabel("Frequency")
        self.axes[1, 1].grid(True, alpha=0.3)
        
        # Add mean line
        mean_reward = np.mean(self.episode_rewards)
        self.axes[1, 1].axvline(mean_reward, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.2f}')
        self.axes[1, 1].legend()
        
    def update_plots(self, window_size: int = 100):
        """
        Update all plots with current data.
        
        Args:
            window_size: Size of the moving average window for rewards
        """
        self.plot_rewards(window_size)
        self.plot_bess_soc()
        self.plot_episode_steps()
        self.plot_reward_distribution()
        
        # Adjust layout and ensure live interactive plotting and saving
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        self.save_plot()
        
    def save_plot(self, filename: str = "centralized_rewards_episode.png"):
        """
        Save the current plot to file.
        
        Args:
            filename: Name of the file to save
        """
        filepath = os.path.join(self.save_dir, filename)
        self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Centralized reward plot saved to: {filepath}")
        
    def get_statistics(self) -> dict:
        """
        Get training statistics.
        
        Returns:
            Dictionary containing training statistics
        """
        if not self.episode_rewards:
            return {}
            
        stats = {
            'total_episodes': len(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'mean_avg_bess_soc': np.mean(self.avg_bess_soc),
            'mean_final_bess_soc': np.mean(self.final_bess_soc),
            'mean_episode_steps': np.mean(self.episode_steps),
            'total_training_steps': np.sum(self.episode_steps)
        }
        
        return stats
        
    def print_statistics(self):
        """Print training statistics to console."""
        stats = self.get_statistics()
        if not stats:
            print("No training data available")
            return
            
        print("\n" + "="*50)
        print("CENTRALIZED SAC TRAINING STATISTICS")
        print("="*50)
        print(f"Total Episodes: {stats['total_episodes']}")
        print(f"Total Training Steps: {stats['total_training_steps']}")
        print(f"Mean Episode Length: {stats['mean_episode_steps']:.2f} steps")
        print(f"Mean Reward: {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f}")
        print(f"Reward Range: [{stats['min_reward']:.4f}, {stats['max_reward']:.4f}]")
        print(f"Mean Average BESS SOC: {stats['mean_avg_bess_soc']:.4f}")
        print(f"Mean Final BESS SOC: {stats['mean_final_bess_soc']:.4f}")
        print("="*50) 

    def save_data(self):
        """
        Save all centralized reward plot data as .npy files in the plots directory for later analysis.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        np.save(os.path.join(self.save_dir, "episodes.npy"), np.arange(1, len(self.episode_rewards) + 1))
        np.save(os.path.join(self.save_dir, "episode_rewards.npy"), np.array(self.episode_rewards))
        np.save(os.path.join(self.save_dir, "avg_bess_soc.npy"), np.array(self.avg_bess_soc))
        np.save(os.path.join(self.save_dir, "final_bess_soc.npy"), np.array(self.final_bess_soc))
        np.save(os.path.join(self.save_dir, "episode_steps.npy"), np.array(self.episode_steps)) 