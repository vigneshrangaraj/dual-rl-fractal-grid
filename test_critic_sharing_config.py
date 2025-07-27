#!/usr/bin/env python3
"""
Test script to demonstrate critic sharing configuration for IA3C agents.
This script shows how to enable/disable critic sharing for ablation studies.
"""

import numpy as np
from utils.config import Config
from agents.secondary.ia3c_agent import IA3CAgent

def test_critic_sharing_config():
    """Test the critic sharing configuration parameter."""
    
    print("=== Testing Critic Sharing Configuration ===\n")
    
    # Create a simple adjacency matrix for testing
    adjacency_matrix = np.array([
        [0, 1, 0, 1],  # Agent 0 connected to agents 1 and 3
        [1, 0, 1, 0],  # Agent 1 connected to agents 0 and 2
        [0, 1, 0, 1],  # Agent 2 connected to agents 1 and 3
        [1, 0, 1, 0]   # Agent 3 connected to agents 0 and 2
    ])
    
    # Test 1: Critic sharing enabled (default)
    print("Test 1: Critic sharing ENABLED (default)")
    config = Config()
    config.enable_critic_sharing = True
    config.spatial_decay_alpha = 0.5
    
    agent = IA3CAgent(config, agent_id=0, adjacency_matrix=adjacency_matrix)
    
    # Simulate a batch of experiences from different agents
    batch = [
        {'state': {'voltage': 1.0, 'i_d': 0.1, 'i_q': 0.2, 'delta': 0.3}, 
         'log_prob': 0.1, 'reward': 1.0, 'next_state': {'voltage': 1.01, 'i_d': 0.11, 'i_q': 0.21, 'delta': 0.31}, 'done': False},
        {'state': {'voltage': 1.02, 'i_d': 0.12, 'i_q': 0.22, 'delta': 0.32}, 
         'log_prob': 0.2, 'reward': 0.8, 'next_state': {'voltage': 1.03, 'i_d': 0.13, 'i_q': 0.23, 'delta': 0.33}, 'done': False},
        {'state': {'voltage': 1.04, 'i_d': 0.14, 'i_q': 0.24, 'delta': 0.34}, 
         'log_prob': 0.3, 'reward': 0.9, 'next_state': {'voltage': 1.05, 'i_d': 0.15, 'i_q': 0.25, 'delta': 0.35}, 'done': False}
    ]
    agent_indices = [0, 1, 2]  # Agent 0's own experience + neighbors
    
    print(f"  - enable_critic_sharing: {config.enable_critic_sharing}")
    print(f"  - spatial_decay_alpha: {config.spatial_decay_alpha}")
    print(f"  - Batch size: {len(batch)} experiences")
    print(f"  - Agent indices: {agent_indices}")
    
    # Test the critic sharing
    try:
        loss = agent.learn_with_critic_sharing(batch, agent_indices)
        print(f"  - Learning successful, loss: {loss:.4f}")
    except Exception as e:
        print(f"  - Error: {e}")
    
    print()
    
    # Test 2: Critic sharing disabled
    print("Test 2: Critic sharing DISABLED")
    config.enable_critic_sharing = False
    
    agent = IA3CAgent(config, agent_id=0, adjacency_matrix=adjacency_matrix)
    
    print(f"  - enable_critic_sharing: {config.enable_critic_sharing}")
    print(f"  - Batch size: {len(batch)} experiences")
    print(f"  - Agent indices: {agent_indices}")
    
    # Test the critic sharing (should fall back to regular learning)
    try:
        loss = agent.learn_with_critic_sharing(batch, agent_indices)
        print(f"  - Learning successful, loss: {loss:.4f}")
        print("  - Note: Only agent's own experience was used (critic sharing disabled)")
    except Exception as e:
        print(f"  - Error: {e}")
    
    print()
    
    # Test 3: Different decay factors
    print("Test 3: Different spatial decay factors")
    decay_factors = [0.1, 0.5, 1.0, 2.0]
    
    for alpha in decay_factors:
        config.enable_critic_sharing = True
        config.spatial_decay_alpha = alpha
        
        agent = IA3CAgent(config, agent_id=0, adjacency_matrix=adjacency_matrix)
        weights = agent.compute_spatial_weights(agent_indices)
        
        print(f"  - spatial_decay_alpha: {alpha}")
        print(f"  - Spatial weights: {weights}")
        print(f"  - Weight distribution: Agent 0: {weights[0]:.3f}, Agent 1: {weights[1]:.3f}, Agent 2: {weights[2]:.3f}")
        print()
    
    print("=== Test Complete ===")
    print("\nUsage for ablation study:")
    print("1. Set config.enable_critic_sharing = False to disable critic sharing")
    print("2. Set config.enable_critic_sharing = True to enable critic sharing")
    print("3. Adjust config.spatial_decay_alpha to control the decay rate")
    print("4. Compare performance between the two configurations")

if __name__ == "__main__":
    test_critic_sharing_config() 