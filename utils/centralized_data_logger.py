#!/usr/bin/env python3
"""
Centralized Data Logger for Centralized SAC Agent.
This module provides data logging functionality specifically for the centralized SAC agent.
"""

import csv
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

class CentralizedDataLogger:
    """
    Data logger specifically designed for centralized SAC agent.
    Logs episode-level data and provides analysis tools for the centralized agent.
    """
    
    def __init__(self, csv_filename: str = "episode_data_centralized.csv"):
        """
        Initialize the centralized data logger.
        
        Args:
            csv_filename: Name of the CSV file to export data
        """
        self.csv_filename = csv_filename
        self.episode_data = []
        
        # Create CSV file with headers if it doesn't exist
        if not os.path.exists(csv_filename):
            self._create_csv_headers()
        else:
            print(f"File {csv_filename} already exists. Data will be appended.")
            os.remove(csv_filename)

        # Initialize variables
        self.current_episode_data = []
        self.all_keys = set(["Episode", "Step", "Secondary_Reward", "Tertiary_Reward", "New_Energy", "Overall_Reward"])
        self.step_rows = []
        self.step_csv = "centralized_step_data.csv"

    
    def _create_csv_headers(self):
        """Create CSV file with appropriate headers for centralized agent."""
        headers = [
            'Episode', 'Total_Reward', 'Steps', 'Avg_BESS_SOC', 'Final_BESS_SOC',
            'Min_BESS_SOC', 'Max_BESS_SOC', 'Reward_Std', 'Convergence_Steps',
            'Timestamp'
        ]
        
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
    
    def log_episode(self, episode: int, total_reward: float, steps: int, 
                   avg_bess_soc: float, final_bess_soc: float, 
                   min_bess_soc: Optional[float] = None, 
                   max_bess_soc: Optional[float] = None,
                   reward_std: Optional[float] = None,
                   convergence_steps: Optional[int] = None):
        """
        Log episode-level data for centralized agent.
        
        Args:
            episode: Episode number
            total_reward: Total reward for the episode
            steps: Number of steps in the episode
            avg_bess_soc: Average BESS SOC during episode
            final_bess_soc: Final BESS SOC at end of episode
            min_bess_soc: Minimum BESS SOC during episode (optional)
            max_bess_soc: Maximum BESS SOC during episode (optional)
            reward_std: Standard deviation of rewards during episode (optional)
            convergence_steps: Steps to convergence (optional)
        """
        # Use default values if not provided
        min_bess_soc = min_bess_soc if min_bess_soc is not None else avg_bess_soc
        max_bess_soc = max_bess_soc if max_bess_soc is not None else avg_bess_soc
        reward_std = reward_std if reward_std is not None else 0.0
        convergence_steps = convergence_steps if convergence_steps is not None else steps
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create episode data row
        episode_data = {
            'Episode': episode,
            'Total_Reward': total_reward,
            'Steps': steps,
            'Avg_BESS_SOC': avg_bess_soc,
            'Final_BESS_SOC': final_bess_soc,
            'Min_BESS_SOC': min_bess_soc,
            'Max_BESS_SOC': max_bess_soc,
            'Reward_Std': reward_std,
            'Convergence_Steps': convergence_steps,
            'Timestamp': timestamp
        }
        
        # Store in memory
        self.episode_data.append(episode_data)
        
        # Append to CSV file
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                episode, total_reward, steps, avg_bess_soc, final_bess_soc,
                min_bess_soc, max_bess_soc, reward_std, convergence_steps, timestamp
            ])
    
    def log_detailed_episode(self, episode: int, step_data: List[Dict]):
        """
        Log detailed step-by-step data for an episode.
        
        Args:
            episode: Episode number
            step_data: List of dictionaries containing step data
        """
        detailed_filename = f"detailed_episode_{episode}_centralized.csv"
        
        if not step_data:
            return
            
        # Create headers for detailed data
        headers = ['Episode', 'Step', 'Hour', 'Reward', 'BESS_SOC', 'BESS_P_MW', 
                  'Grid_Power_MW', 'DER_Generation_MW', 'Load_MW', 'Voltage_Avg']
        
        with open(detailed_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            for step_idx, step_info in enumerate(step_data):
                row = [
                    episode, step_idx, step_info.get('hour', 0),
                    step_info.get('reward', 0.0), step_info.get('bess_soc', 0.5),
                    step_info.get('bess_p_mw', 0.0), step_info.get('grid_power_mw', 0.0),
                    step_info.get('der_generation_mw', 0.0), step_info.get('load_mw', 0.0),
                    step_info.get('voltage_avg', 1.0)
                ]
                writer.writerow(row)
        
        print(f"Detailed episode {episode} data saved to: {detailed_filename}")
    
    def get_episode_summary(self, episode: int) -> Dict:
        """
        Get summary statistics for a specific episode.
        
        Args:
            episode: Episode number
            
        Returns:
            Dictionary containing episode summary
        """
        # Load data from CSV
        df = pd.read_csv(self.csv_filename)
        episode_data = df[df['Episode'] == episode]
        
        if episode_data.empty:
            return {}
        
        summary = {
            'episode': episode,
            'total_reward': episode_data['Total_Reward'].iloc[0],
            'steps': episode_data['Steps'].iloc[0],
            'avg_bess_soc': episode_data['Avg_BESS_SOC'].iloc[0],
            'final_bess_soc': episode_data['Final_BESS_SOC'].iloc[0],
            'min_bess_soc': episode_data['Min_BESS_SOC'].iloc[0],
            'max_bess_soc': episode_data['Max_BESS_SOC'].iloc[0],
            'reward_std': episode_data['Reward_Std'].iloc[0],
            'convergence_steps': episode_data['Convergence_Steps'].iloc[0]
        }
        
        return summary
    
    def get_training_summary(self) -> Dict:
        """
        Get overall training summary statistics.
        
        Returns:
            Dictionary containing training summary
        """
        if not os.path.exists(self.csv_filename):
            return {}
        
        df = pd.read_csv(self.csv_filename)
        
        if df.empty:
            return {}
        
        summary = {
            'total_episodes': len(df),
            'total_training_steps': df['Steps'].sum(),
            'mean_reward': df['Total_Reward'].mean(),
            'std_reward': df['Total_Reward'].std(),
            'min_reward': df['Total_Reward'].min(),
            'max_reward': df['Total_Reward'].max(),
            'mean_avg_bess_soc': df['Avg_BESS_SOC'].mean(),
            'mean_final_bess_soc': df['Final_BESS_SOC'].mean(),
            'mean_episode_steps': df['Steps'].mean(),
            'mean_convergence_steps': df['Convergence_Steps'].mean(),
            'bess_soc_range': {
                'min': df['Min_BESS_SOC'].min(),
                'max': df['Max_BESS_SOC'].max(),
                'mean_min': df['Min_BESS_SOC'].mean(),
                'mean_max': df['Max_BESS_SOC'].mean()
            }
        }
        
        return summary
    
    def print_training_summary(self):
        """Print training summary statistics to console."""
        summary = self.get_training_summary()
        if not summary:
            print("No training data available")
            return
        
        print("\n" + "="*60)
        print("CENTRALIZED SAC TRAINING SUMMARY")
        print("="*60)
        print(f"Total Episodes: {summary['total_episodes']}")
        print(f"Total Training Steps: {summary['total_training_steps']}")
        print(f"Mean Episode Length: {summary['mean_episode_steps']:.2f} steps")
        print(f"Mean Convergence Steps: {summary['mean_convergence_steps']:.2f} steps")
        print(f"Mean Reward: {summary['mean_reward']:.4f} ± {summary['std_reward']:.4f}")
        print(f"Reward Range: [{summary['min_reward']:.4f}, {summary['max_reward']:.4f}]")
        print(f"Mean Average BESS SOC: {summary['mean_avg_bess_soc']:.4f}")
        print(f"Mean Final BESS SOC: {summary['mean_final_bess_soc']:.4f}")
        print(f"BESS SOC Range: [{summary['bess_soc_range']['min']:.4f}, {summary['bess_soc_range']['max']:.4f}]")
        print(f"Mean BESS SOC Min: {summary['bess_soc_range']['mean_min']:.4f}")
        print(f"Mean BESS SOC Max: {summary['bess_soc_range']['mean_max']:.4f}")
        print("="*60)
    
    def export_to_excel(self, filename: str = "centralized_training_data.xlsx"):
        """
        Export training data to Excel file with multiple sheets.
        
        Args:
            filename: Name of the Excel file
        """
        if not os.path.exists(self.csv_filename):
            print("No data to export")
            return
        
        df = pd.read_csv(self.csv_filename)
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name='Episode_Data', index=False)
            
            # Summary statistics sheet
            summary = self.get_training_summary()
            if summary:
                summary_df = pd.DataFrame([summary])
                summary_df.to_excel(writer, sheet_name='Training_Summary', index=False)
            
            # BESS SOC analysis sheet
            bess_analysis = df[['Episode', 'Avg_BESS_SOC', 'Final_BESS_SOC', 'Min_BESS_SOC', 'Max_BESS_SOC']]
            bess_analysis.to_excel(writer, sheet_name='BESS_Analysis', index=False)
            
            # Reward analysis sheet
            reward_analysis = df[['Episode', 'Total_Reward', 'Reward_Std', 'Steps', 'Convergence_Steps']]
            reward_analysis.to_excel(writer, sheet_name='Reward_Analysis', index=False)
        
        print(f"Training data exported to Excel: {filename}")
    
    def plot_training_progress(self, save_plot: bool = True):
        """
        Create and optionally save training progress plots.
        
        Args:
            save_plot: Whether to save the plot to file
        """
        if not os.path.exists(self.csv_filename):
            print("No data to plot")
            return
        
        df = pd.read_csv(self.csv_filename)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Centralized SAC Training Progress", fontsize=16)
        
        # Reward progression
        axes[0, 0].plot(df['Episode'], df['Total_Reward'], 'b-', alpha=0.7)
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Total Reward")
        axes[0, 0].grid(True, alpha=0.3)
        
        # BESS SOC progression
        axes[0, 1].plot(df['Episode'], df['Avg_BESS_SOC'], 'g-', label='Average SOC', linewidth=2)
        axes[0, 1].plot(df['Episode'], df['Final_BESS_SOC'], 'r-', label='Final SOC', linewidth=2)
        axes[0, 1].set_title("BESS State of Charge")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("SOC")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Episode length
        axes[1, 0].plot(df['Episode'], df['Steps'], 'purple', linewidth=2)
        axes[1, 0].set_title("Episode Length")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Steps")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Convergence steps
        axes[1, 1].plot(df['Episode'], df['Convergence_Steps'], 'orange', linewidth=2)
        axes[1, 1].set_title("Convergence Steps")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Steps to Convergence")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_filename = "centralized_training_progress.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Training progress plot saved to: {plot_filename}")
        
        plt.show() 

    def flatten_dict(self, d, parent_key='', sep='_'):
        """Recursively flattens a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def log_step_data(
        self,
        episode: int,
        step: int,
        secondary_reward: float,
        ter_reward: float,
        new_energy: float,
        ter_state,
        sec_state,
        voltage_violations,
        overall_reward: float
    ):
        """
        Log detailed step data for centralized agent, flattening all dicts and dynamically expanding header.
        """
        # Flatten tertiary state
        ter_flat = self.flatten_dict(ter_state)

        # Flatten secondary state (list of dicts)
        sec_flat = {}
        for idx, agent in enumerate(sec_state):
            agent_flat = self.flatten_dict(agent, parent_key=f"sec_agent{idx}")
            sec_flat.update(agent_flat)

        # Flatten voltage violations (list or dict)
        if isinstance(voltage_violations, dict):
            viol_flat = self.flatten_dict(voltage_violations, parent_key="viol")
        elif isinstance(voltage_violations, list):
            viol_flat = {f"viol_agent{idx}": v for idx, v in enumerate(voltage_violations)}
        else:
            viol_flat = {"viol": voltage_violations}

        # Compose row
        row = {
            "Episode": episode,
            "Step": step,
            "Secondary_Reward": secondary_reward,
            "Tertiary_Reward": ter_reward,
            "New_Energy": new_energy,
            "Overall_Reward": overall_reward,
            **ter_flat,
            **sec_flat,
            **viol_flat
        }
        self.step_rows.append(row)
        new_keys = set(row.keys()) - self.all_keys
        if new_keys:
            self.all_keys.update(new_keys)
            # Rewrite CSV with new header
            with open(self.step_csv, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(self.all_keys))
                writer.writeheader()
                for r in self.step_rows:
                    writer.writerow(r)
        else:
            # Just append the new row
            with open(self.step_csv, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(self.all_keys))
                writer.writerow(row)