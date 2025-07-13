#!/usr/bin/env python3
"""
Data logger for capturing episode data and exporting to CSV.
"""

import csv
import os
import pandas as pd
import numpy as np
from datetime import datetime

class EpisodeDataLogger:
    def __init__(self, csv_filename="episode_data.csv"):
        """
        Initialize the data logger.
        
        Args:
            csv_filename: Name of the CSV file to export data
        """
        self.csv_filename = csv_filename
        self.episode_data = []
        self.current_episode_data = []
        
        # Create CSV file with headers if it doesn't exist
        if not os.path.exists(csv_filename):
            self._create_csv_headers()
    
    def _create_csv_headers(self):
        """Create CSV file with appropriate headers."""
        headers = [
            'Episode', 'Step', 'Hour',
            'DER1_P_MW', 'DER2_P_MW', 'DER3_P_MW', 'DER4_P_MW',
            'BESS_P_MW', 'BESS_SOC',
            'Load_Bus4_P_MW', 'Load_Bus5_P_MW', 'Load_Bus6_P_MW', 'Load_Bus7_P_MW', 'Load_Bus3_P_MW',
            'DER1_Voltage', 'DER2_Voltage', 'DER3_Voltage', 'DER4_Voltage',
            'Grid_Power_MW',
            'Secondary_Rewards',
            'Overall_Reward',
            'Current_Buy_Price', 'Current_Sell_Price'
        ]
        
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
    
    def log_step_data(self, episode, step, hour, tertiary_env, secondary_env, 
                     secondary_rewards, overall_reward, config):
        """
        Log data for a single step.
        
        Args:
            episode: Current episode number
            step: Current step number
            hour: Current hour (0-23)
            tertiary_env: Tertiary environment object
            secondary_env: Secondary environment object
            secondary_rewards: List of secondary agent rewards
            overall_reward: Overall reward for the step
            config: Configuration object
        """
        try:
            # Get DER generation data
            der_powers = []
            der_voltages = []
            
            # Extract DER data from the network
            if tertiary_env.net is not None and hasattr(tertiary_env, 'index_map'):
                for mg_id in range(len(tertiary_env.microgrids)):
                    if mg_id in tertiary_env.index_map:
                        # Get DER generation (solar and wind)
                        solar_gens = list(tertiary_env.index_map[mg_id].get('solar_gen', {}).values())
                        wind_gens = list(tertiary_env.index_map[mg_id].get('wind_gen', {}).values())
                        
                        # Combine solar and wind DERs
                        all_ders = solar_gens + wind_gens
                        
                        for der_idx, der_bus in enumerate(all_ders):
                            if der_idx < 4:  # Limit to 4 DERs
                                try:
                                    # Get DER power
                                    der_power = tertiary_env.net.gen.loc[der_bus, 'p_mw']
                                    der_powers.append(der_power)
                                    
                                    # Get DER voltage
                                    if hasattr(tertiary_env.net, 'res_gen'):
                                        der_voltage = tertiary_env.net.res_gen.vm_pu[der_bus]
                                        der_voltages.append(der_voltage)
                                    else:
                                        der_voltages.append(1.0)
                                except:
                                    der_powers.append(0.0)
                                    der_voltages.append(1.0)
            
            # Pad DER data to 4 DERs
            while len(der_powers) < 4:
                der_powers.append(0.0)
            while len(der_voltages) < 4:
                der_voltages.append(1.0)
            
            # Get BESS data
            bess_p_mw = 0.0
            bess_soc = 0.5
            
            if tertiary_env.net is not None and hasattr(tertiary_env, 'index_map'):
                for mg_id in range(len(tertiary_env.microgrids)):
                    if mg_id in tertiary_env.index_map and 'storage' in tertiary_env.index_map[mg_id]:
                        storage_idx = list(tertiary_env.index_map[mg_id]['storage'].values())[0]
                        try:
                            bess_p_mw = tertiary_env.net.storage.loc[storage_idx, 'p_mw']
                            bess_soc = tertiary_env.microgrids[mg_id].last_soc
                        except:
                            pass
            
            # Get load data
            load_powers = [0.0] * 5  # 5 load buses (3, 4, 5, 6, 7)
            
            if tertiary_env.net is not None:
                try:
                    # Get loads from the network
                    for i, load_idx in enumerate(tertiary_env.net.load.index):
                        if i < 5:  # Limit to 5 loads
                            load_powers[i] = tertiary_env.net.load.loc[load_idx, 'p_mw']
                except:
                    pass
            
            # Get grid power
            grid_power = 0.0
            if tertiary_env.net is not None and hasattr(tertiary_env.net, 'res_ext_grid'):
                try:
                    grid_power = tertiary_env.net.res_ext_grid.p_mw.sum()
                except:
                    pass
            
            # Get current prices
            current_prices = config.get_temporal_prices(hour)
            current_buy_price = current_prices['buy_price']
            current_sell_price = current_prices['sell_price']
            
            # Calculate average secondary voltage
            avg_secondary_voltage = np.mean(der_voltages) if der_voltages else 1.0
            
            # Calculate average secondary reward
            avg_secondary_reward = np.mean(secondary_rewards) if secondary_rewards else 0.0
            
            # Create data row
            row_data = [
                episode, step, hour,
                der_powers[0], der_powers[1], der_powers[2], der_powers[3],
                bess_p_mw, bess_soc,
                load_powers[0], load_powers[1], load_powers[2], load_powers[3], load_powers[4],
                der_voltages[0], der_voltages[1], der_voltages[2], der_voltages[3],
                grid_power,
                avg_secondary_reward,
                overall_reward,
                current_buy_price, current_sell_price
            ]
            
            self.current_episode_data.append(row_data)
            
        except Exception as e:
            print(f"Error logging step data: {e}")
            # Log empty row if there's an error
            empty_row = [episode, step, hour] + [0.0] * 20
            self.current_episode_data.append(empty_row)
    
    def end_episode(self):
        """End the current episode and write data to CSV."""
        try:
            # Append episode data to CSV
            with open(self.csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in self.current_episode_data:
                    writer.writerow(row)
            
            # Clear current episode data
            self.current_episode_data = []
            
            print(f"Episode data exported to {self.csv_filename}")
            
        except Exception as e:
            print(f"Error writing episode data: {e}")
    
    def get_episode_summary(self, episode):
        """Get summary statistics for an episode."""
        if not self.current_episode_data:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.current_episode_data, columns=[
            'Episode', 'Step', 'Hour',
            'DER1_P_MW', 'DER2_P_MW', 'DER3_P_MW', 'DER4_P_MW',
            'BESS_P_MW', 'BESS_SOC',
            'Load_Bus4_P_MW', 'Load_Bus5_P_MW', 'Load_Bus6_P_MW', 'Load_Bus7_P_MW', 'Load_Bus3_P_MW',
            'DER1_Voltage', 'DER2_Voltage', 'DER3_Voltage', 'DER4_Voltage',
            'Grid_Power_MW',
            'Secondary_Rewards',
            'Overall_Reward',
            'Current_Buy_Price', 'Current_Sell_Price'
        ])
        
        summary = {
            'episode': episode,
            'total_steps': len(df),
            'avg_bess_soc': df['BESS_SOC'].mean(),
            'min_bess_soc': df['BESS_SOC'].min(),
            'max_bess_soc': df['BESS_SOC'].max(),
            'total_der_generation': df[['DER1_P_MW', 'DER2_P_MW', 'DER3_P_MW', 'DER4_P_MW']].sum().sum(),
            'total_load': df[['Load_Bus4_P_MW', 'Load_Bus5_P_MW', 'Load_Bus6_P_MW', 'Load_Bus7_P_MW', 'Load_Bus3_P_MW']].sum().sum(),
            'avg_grid_power': df['Grid_Power_MW'].mean(),
            'avg_secondary_reward': df['Secondary_Rewards'].mean(),
            'total_overall_reward': df['Overall_Reward'].sum(),
            'avg_voltage': df[['DER1_Voltage', 'DER2_Voltage', 'DER3_Voltage', 'DER4_Voltage']].mean().mean()
        }
        
        return summary 