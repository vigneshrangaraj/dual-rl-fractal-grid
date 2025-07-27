# main_centralized.py
# Centralized SAC agent for ablation study

import time
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from typing import List

from plotter.centralized_reward_plotter import CentralizedRewardPlotter
from utils.config import Config
from env.dual_rl_env import DualRLEnv
from agents.centralized_sac_agent import CentralizedSACAgent
from agents.centralized_ddpg_agent import CentralizedDDPGAgent
from agents.centralized_ppo_agent import CentralizedPPOAgent
from utils.helper import Helper as helper
from utils.centralized_data_logger import CentralizedDataLogger

# Create directory for models if it doesn't exist
os.makedirs("models", exist_ok=True)


def save_centralized_model(centralized_agent, episode):
    """Save centralized model at regular intervals"""
    if episode % 10 == 0:  # Save every 10 episodes
        centralized_agent.save(f"models/centralized_agent_episode")


def main():
    """Main function for centralized SAC training."""
    # Clean up previous episode data file if it exists
    if os.path.exists("episode_data_centralized.csv"):
        os.remove("episode_data_centralized.csv")
        print("Cleaned up previous episode_data_centralized.csv file")
    if os.path.exists("centralized_step_data.csv"):
        os.remove("centralized_step_data.csv")
        print("Cleaned up previous centralized_step_data.csv file")
    
    # Load configuration parameters
    config = Config()
    
    # Check if centralized SAC is enabled
    if not config.use_centralized_sac and not getattr(config, "use_centralized_ddpg", False) and not getattr(config, "use_centralized_ppo", False):
        print("Centralized agent is disabled. Set config.use_centralized_sac = True, config.use_centralized_ddpg = True, or config.use_centralized_ppo = True to enable.")
        return

    # Initialize the dual-level environment
    dual_env = DualRLEnv(config)

    # Reset environment to get initial states
    state = dual_env.reset()
    tertiary_state = state.get("tertiary", {})
    secondary_states = state.get("secondary", [])
    
    # Calculate combined state dimension
    ter_flat = helper.flatten_tertiary_state(tertiary_state)
    ter_state_dim = ter_flat.shape[0]
    
    # Secondary state dimension (5 features per agent: voltage, reactive_power, i_d, i_q, delta)
    sec_state_dim = len(secondary_states) * 5
    combined_state_dim = ter_state_dim + sec_state_dim
    
    # Calculate combined action dimension
    num_microgrids = getattr(config, "num_microgrids", 1)
    num_der_total = getattr(config, "num_der_total", 4)
    num_bess_total = getattr(config, "num_bess_total", 1)
    num_secondary_agents = len(secondary_states)
    # Tertiary actions: DER actions + BESS actions + tie line actions
    tertiary_action_dim = (num_der_total + num_bess_total) * num_microgrids + dual_env.tertiary_env.switches
    # Secondary actions: reactive power for each agent
    secondary_action_dim = num_secondary_agents
    combined_action_dim = tertiary_action_dim + secondary_action_dim
    print(f"Combined state dimension: {combined_state_dim}")
    print(f"Combined action dimension: {combined_action_dim}")
    print(f"Tertiary state dim: {ter_state_dim}, Secondary state dim: {sec_state_dim}")
    print(f"Tertiary action dim: {tertiary_action_dim}, Secondary action dim: {secondary_action_dim}")
    # Instantiate the centralized agent based on config
    if config.use_centralized_sac:
        centralized_agent = CentralizedSACAgent(combined_state_dim, combined_action_dim, config)
    elif getattr(config, "use_centralized_ddpg", False):
        centralized_agent = CentralizedDDPGAgent(combined_state_dim, combined_action_dim, config)
    elif getattr(config, "use_centralized_ppo", False):
        centralized_agent = CentralizedPPOAgent(combined_state_dim, combined_action_dim, config)
    else:
        raise ValueError("No centralized agent selected: set use_centralized_sac, use_centralized_ddpg, or use_centralized_ppo in config.")

    # Initialize the dedicated plotter for centralized agent
    plotter = CentralizedRewardPlotter()

    # Initialize dedicated data logger for centralized agent
    data_logger = CentralizedDataLogger("centralized_step_data.csv")

    num_episodes = getattr(config, "num_episodes", 1000)

    # Training loop
    for ep in range(num_episodes):
        print(f"=================Starting episode {ep + 1}/{num_episodes}")
        done = False
        episode_reward = 0.0
        time_step = 0
        bess_soc_for_day = []

        while not done:
            # Get current states
            ter_state = state.get("tertiary", {})
            sec_states = state.get("secondary", [])
            
            # Track BESS SOC for analysis
            microgrid = ter_state.get("microgrids", [{}])[0]
            bess_soc_for_day.append(microgrid.get("bess_soc", 0.0))
            
            # Select combined action
            ter_action, sec_actions, log_prob, q_value = centralized_agent.select_action(
                ter_state, sec_states, dual_env.tertiary_env.switch_set
            )
            
            # Execute tertiary action
            next_state, ter_rewards, ter_done, ter_info = dual_env.step(ter_action, time_step)
            
            # Run secondary environment with the selected actions
            new_sec_states, sec_rewards, sec_done, sec_info = dual_env.secondary_env.step(
                sec_actions, dual_env.tertiary_env, ter_action.get("tie_lines", [])
            )
            dual_env.secondary_env.time_step = 0
            next_sec_states = new_sec_states

            # print actions
            print(f"Tertiary actions: {ter_action}")
            print(f"Secondary actions: {sec_actions}")
            print(f"Log prob: {log_prob}")
            
            # Calculate combined reward (weighted sum of tertiary and secondary rewards)
            aggregated_sec_reward = np.mean(sec_rewards)
            overall_reward = np.mean(ter_rewards) + config.alpha_sec * aggregated_sec_reward

            # Use new_energy and temporal pricing if available
            if not sec_info.get("is_converged", False):
                p_mw = 100
            else:
                p_mw = sec_info.get("new_energy", None)
            if p_mw is not None:
                current_prices = config.get_temporal_prices(time_step)
                current_buy_price = current_prices["buy_price"]
                current_sell_price = current_prices["sell_price"]

                if p_mw > 0:
                    # Selling to the grid - reward based on current sell price
                    overall_reward -= current_buy_price * p_mw
                else:
                    # Borrowing from the grid - penalty based on current buy price
                    overall_reward += current_sell_price * abs(p_mw)

            episode_reward += overall_reward

            # Log step data
            data_logger.log_step_data(
                episode=ep + 1,
                step=time_step,
                secondary_reward=aggregated_sec_reward,
                ter_reward=np.mean(ter_rewards),
                new_energy=p_mw,
                ter_state=ter_state,
                sec_state=sec_states,
                voltage_violations=sec_info.get("violations", []),
                overall_reward=overall_reward
            )

            # Store experience in replay buffer
            next_ter_state = next_state
            centralized_agent.remember(
                ter_state, sec_states, ter_action, sec_actions, 
                log_prob, overall_reward, next_ter_state, next_sec_states, ter_done
            )
            
            # Learn from experience
            centralized_agent.learn()
            
            # Update states
            state["tertiary"]= next_state
            state["secondary"] = next_sec_states
            
            time_step += 1
            done = ter_done or time_step >= config.max_steps

        # Calculate episode statistics
        avg_bess_soc = np.mean(bess_soc_for_day) if bess_soc_for_day else 0.0
        final_bess_soc = bess_soc_for_day[-1] if bess_soc_for_day else 0.0
        min_bess_soc = np.min(bess_soc_for_day) if bess_soc_for_day else 0.0
        max_bess_soc = np.max(bess_soc_for_day) if bess_soc_for_day else 0.0
        
        # Log episode data using dedicated centralized logger
        data_logger.log_episode(
            episode=ep + 1,
            total_reward=episode_reward,
            steps=time_step,
            avg_bess_soc=avg_bess_soc,
            final_bess_soc=final_bess_soc,
            min_bess_soc=min_bess_soc,
            max_bess_soc=max_bess_soc
        )
        
        # Update dedicated centralized plotter
        plotter.update(episode_reward, avg_bess_soc, final_bess_soc, time_step)
        plotter.save_data()
        
        # Save models periodically
        save_centralized_model(centralized_agent, ep + 1)

        # Update and save plots
        #plotter.update_plots(50)
        #plotter.save_plot("centralized_rewards_episode.png")
        
        # Print episode summary
        print(f"Episode {ep + 1} completed:")
        print(f"  Total reward: {episode_reward:.4f}")
        print(f"  Steps: {time_step}")
        print("=" * 50)

    # Save final model
    centralized_agent.save("models/centralized_agent_final")
    

    
    # Print training statistics
    plotter.print_statistics()
    data_logger.print_training_summary()
    
    print("\nCentralized SAC training completed!")
    print("Results saved to:")
    print("  - models/centralized_agent_final_*.pt")
    print("  - plots/centralized_rewards_episode.png")
    print("  - episode_data_centralized.csv")
    print("  - Additional analysis available via data_logger.export_to_excel()")


if __name__ == "__main__":
    main() 