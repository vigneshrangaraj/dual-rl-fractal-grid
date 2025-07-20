#!/usr/bin/env python3
"""
Test script for centralized SAC agent.
"""

import numpy as np
import torch
from utils.config import Config
from env.dual_rl_env import DualRLEnv
from agents.centralized_sac_agent import CentralizedSACAgent
from utils.helper import Helper as helper

def test_centralized_agent():
    """Test the centralized SAC agent functionality."""
    print("Testing Centralized SAC Agent...")
    
    # Load configuration
    config = Config()
    config.use_centralized_sac = True  # Enable centralized SAC
    config.num_episodes = 5  # Small number for testing
    
    # Initialize environment
    dual_env = DualRLEnv(config)
    
    # Reset environment to get initial states
    state = dual_env.reset()
    tertiary_state = state.get("tertiary", {})
    secondary_states = state.get("secondary", [])
    
    print(f"Initial tertiary state keys: {list(tertiary_state.keys())}")
    print(f"Number of secondary agents: {len(secondary_states)}")
    print(f"Secondary state keys: {list(secondary_states[0].keys()) if secondary_states else 'None'}")
    
    # Calculate dimensions
    ter_flat = helper.flatten_tertiary_state(tertiary_state)
    ter_state_dim = ter_flat.shape[0]
    sec_state_dim = len(secondary_states) * 5  # 5 features per secondary agent
    combined_state_dim = ter_state_dim + sec_state_dim
    
    num_microgrids = getattr(config, "num_microgrids", 1)
    num_der_total = getattr(config, "num_der_total", 4)
    num_secondary_agents = len(secondary_states)
    
    tertiary_action_dim = (num_der_total + 1) * num_microgrids + dual_env.tertiary_env.switches
    secondary_action_dim = num_secondary_agents
    combined_action_dim = tertiary_action_dim + secondary_action_dim
    
    print(f"Tertiary state dimension: {ter_state_dim}")
    print(f"Secondary state dimension: {sec_state_dim}")
    print(f"Combined state dimension: {combined_state_dim}")
    print(f"Tertiary action dimension: {tertiary_action_dim}")
    print(f"Secondary action dimension: {secondary_action_dim}")
    print(f"Combined action dimension: {combined_action_dim}")
    
    # Initialize centralized agent
    centralized_agent = CentralizedSACAgent(combined_state_dim, combined_action_dim, config)
    print("Centralized SAC agent initialized successfully!")
    
    # Test action selection
    print("\nTesting action selection...")
    ter_action, sec_actions, log_prob, q_value = centralized_agent.select_action(
        tertiary_state, secondary_states, dual_env.tertiary_env.switch_set
    )
    
    print(f"Tertiary action type: {type(ter_action)}")
    print(f"Tertiary action keys: {list(ter_action.keys()) if isinstance(ter_action, dict) else 'Not a dict'}")
    print(f"Secondary actions: {sec_actions}")
    print(f"Log probability: {log_prob}")
    print(f"Q-value: {q_value}")
    
    # Test one step of the environment
    print("\nTesting environment step...")
    next_state, ter_rewards, ter_done, ter_info = dual_env.step(ter_action, 0)
    
    # Run secondary environment
    new_sec_states, sec_rewards, sec_done, sec_info = dual_env.secondary_env.step(
        sec_actions, dual_env.tertiary_env, ter_action.get("tie_lines", [])
    )
    
    print(f"Tertiary rewards: {ter_rewards}")
    print(f"Secondary rewards: {sec_rewards}")
    print(f"Tertiary done: {ter_done}")
    print(f"Secondary done: {sec_done}")
    
    # Test experience storage
    print("\nTesting experience storage...")
    combined_reward = np.mean(ter_rewards) + 0.1 * np.mean(sec_rewards)
    centralized_agent.remember(
        tertiary_state, secondary_states, ter_action, sec_actions,
        log_prob, combined_reward, next_state.get("tertiary", {}), new_sec_states, ter_done
    )
    print(f"Experience stored. Buffer size: {len(centralized_agent.memory)}")
    
    # Test learning (if enough samples)
    if len(centralized_agent.memory) >= centralized_agent.batch_size:
        print("\nTesting learning...")
        centralized_agent.learn()
        print("Learning completed successfully!")
    else:
        print(f"\nNot enough samples for learning. Need {centralized_agent.batch_size}, have {len(centralized_agent.memory)}")
    
    print("\nCentralized SAC agent test completed successfully!")

if __name__ == "__main__":
    test_centralized_agent() 