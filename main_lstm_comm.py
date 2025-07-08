
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from plotter.reward_plotter import RewardPlotter
from plotter.action_state_plotter import ActionStatePlotter
from utils.config import Config
from env.dual_rl_env import DualRLEnv
from agents.tertiary.sac_agent import SACAgent
from agents.secondary.ia3c_agent import IA3CAgent
from utils.helper import Helper as helper
import os
from typing import List

# Create directory for models if it doesn't exist
os.makedirs("models", exist_ok=True)

def compute_attention_context(q_i, others, scale=True):
    """
    Compute attention-based context from other hidden states
    q_i: (1, d), others: list of (1, d)
    """
    if not others:
        return torch.zeros_like(q_i)

    k = torch.stack(others)           # (N-1, 1, d)
    v = k.clone()                     # (N-1, 1, d)
    q = q_i.unsqueeze(0).unsqueeze(0)  # (1, 1, d)

    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1))  # (1, 1, N-1)
    if scale:
        scores = scores / (d_k ** 0.5)
    weights = torch.softmax(scores, dim=-1)        # (1, 1, N-1)
    context = torch.matmul(weights, v).squeeze(0).squeeze(0)  # (d,)
    return context

def save_models(tertiary_agent, secondary_agents, episode):
    """Save models at regular intervals"""
    if episode % 10 == 0:
        tertiary_agent.save(f"models/tertiary_agent_episode")
        for i, agent in enumerate(secondary_agents):
            agent.save(f"models/secondary_agent_{i}_episode")

def main(tertiary_action=None):
    config = Config()
    env = DualRLEnv(config=config)

    tertiary_agent = SACAgent(config=config, action_space=env.tertiary_action_space)
    secondary_agents = [IA3CAgent(config=config, agent_id=i) for i in range(env.num_secondary_agents)]

    reward_plotter = RewardPlotter()
    action_state_plotter = ActionStatePlotter(env)

    num_episodes = config.training.num_episodes

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        # Initialize hidden and cell states
        h_states = [torch.zeros(1, 1, 64) for _ in secondary_agents]
        c_states = [torch.zeros(1, 1, 64) for _ in secondary_agents]

        while not done:
            actions = []
            context_vectors = []

            # Compute attention-based context vectors
            for i in range(len(secondary_agents)):
                q_i = h_states[i].squeeze(0)
                others = [h_states[j].squeeze(0) for j in range(len(secondary_agents)) if j != i]
                context = compute_attention_context(q_i, others)
                context_vectors.append(context)

            # Each agent selects an action with LSTM + attention context
            for i, agent in enumerate(secondary_agents):
                obs_tensor = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                context = context_vectors[i].unsqueeze(0)
                action, h_states[i], c_states[i] = agent.select_action(obs_tensor, (h_states[i], c_states[i]), context)
                actions.append(action)

            next_obs, rewards, done, info = env.step(actions)
            total_reward += sum(rewards)
            obs = next_obs

        print(f"Episode {ep + 1} - Total Reward: {total_reward}")

if __name__ == "__main__":
    main()
