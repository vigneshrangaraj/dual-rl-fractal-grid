import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import random
from typing import Any, Tuple, List

from utils.helper import Helper as helper

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output actions in [-1, 1]
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Learnable log standard deviation for action noise
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        action_mean = self.actor(state)
        action_std = torch.exp(self.log_std).expand_as(action_mean)
        return action_mean, action_std

    def get_action_and_value(self, state, action=None):
        action_mean, action_std = self.forward(state)
        dist = Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(state).squeeze(-1)
        
        return action, log_prob, entropy, value

class PPOBuffer:
    def __init__(self, max_size: int = 10000):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.max_size = max_size

    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get_all(self):
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.FloatTensor(np.array(self.actions)),
            torch.FloatTensor(np.array(self.rewards)),
            torch.FloatTensor(np.array(self.values)),
            torch.FloatTensor(np.array(self.log_probs)),
            torch.FloatTensor(np.array(self.dones))
        )

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.states)

class CentralizedPPOAgent:
    def __init__(self, state_dim: int, action_dim: int, config: Any):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=getattr(config, "ppo_lr", 3e-4))
        
        self.buffer = PPOBuffer(max_size=getattr(config, "ppo_buffer_size", 10000))
        
        # PPO hyperparameters
        self.clip_ratio = getattr(config, "ppo_clip_ratio", 0.2)
        self.value_loss_coef = getattr(config, "ppo_value_loss_coef", 0.5)
        self.entropy_coef = getattr(config, "ppo_entropy_coef", 0.01)
        self.max_grad_norm = getattr(config, "ppo_max_grad_norm", 0.5)
        self.target_kl = getattr(config, "ppo_target_kl", 0.01)
        self.update_epochs = getattr(config, "ppo_update_epochs", 4)
        self.batch_size = getattr(config, "ppo_batch_size", 64)
        self.gamma = getattr(config, "ppo_gamma", 0.99)
        self.gae_lambda = getattr(config, "ppo_gae_lambda", 0.95)

    def select_action(self, ter_state, sec_states, switch_set) -> Tuple[np.ndarray, np.ndarray, float, float]:
        state = self._combine_state(ter_state, sec_states)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.actor_critic.get_action_and_value(state_tensor)
        
        action = action.cpu().numpy().flatten()
        log_prob = log_prob.cpu().numpy().item()
        value = value.cpu().numpy().item()
        
        # Calculate action dimensions
        num_microgrids = getattr(self.config, "num_microgrids", 1)
        num_der_total = getattr(self.config, "num_der_total", 4)
        num_secondary_agents = len(sec_states)
        
        # Tertiary actions: DER actions + BESS action + tie line actions
        tertiary_action_dim = (num_der_total + 1) * num_microgrids + len(switch_set)
        
        # Split action into tertiary and secondary
        ter_action_vector = action[:tertiary_action_dim]
        sec_actions = action[tertiary_action_dim:]
        
        # Convert tertiary action vector back to structured format
        ter_action = helper.unpack_tertiary_action(ter_action_vector, switch_set)
        
        return ter_action, sec_actions, log_prob, value

    def remember(self, ter_state, sec_states, ter_action, sec_actions, log_prob, reward, next_ter_state, next_sec_states, done):
        state = self._combine_state(ter_state, sec_states)
        
        # Flatten tertiary action
        ter_action_vector = helper.flatten_tertiary_action(ter_action)
        
        # Combine tertiary and secondary actions
        action = np.concatenate([np.array(ter_action_vector).flatten(), np.array(sec_actions).flatten()])
        
        # Get value for the current state
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, _, value = self.actor_critic.get_action_and_value(state_tensor, action_tensor)
            value = value.cpu().numpy().item()
        
        self.buffer.add(state, action, reward, value, log_prob, float(done))

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return
        
        states, actions, rewards, values, log_probs, dones = self.buffer.get_all()
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        values = values.to(self.device)
        log_probs = log_probs.to(self.device)
        dones = dones.to(self.device)

        # Compute advantages using GAE
        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for epoch in range(self.update_epochs):
            # Create mini-batches
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = log_probs[batch_indices]

                # Get current action distribution
                action_mean, action_std = self.actor_critic(batch_states)
                dist = Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(batch_actions).sum(-1)

                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                new_values = self.actor_critic.critic(batch_states).squeeze(-1)
                value_loss = nn.MSELoss()(new_values, batch_returns)

                # Entropy loss
                entropy = dist.entropy().sum(-1).mean()
                entropy_loss = -entropy

                # Total loss
                total_loss = actor_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # Clear buffer after update
        self.buffer.clear()

    def _compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        last_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        return advantages

    def _combine_state(self, ter_state, sec_states):
        # Flatten tertiary state using helper
        ter_flat = helper.flatten_tertiary_state(ter_state)

        # Flatten secondary states
        sec_features = []
        for sec_state in sec_states:
            # Extract secondary state features
            voltage = sec_state.get("voltage", 0.0)
            reactive_power = sec_state.get("reactive_power", 0.0)
            i_d = sec_state.get("i_d", 0.0)
            i_q = sec_state.get("i_q", 0.0)
            delta = sec_state.get("delta", 0.0)
            sec_features.extend([voltage, reactive_power, i_d, i_q, delta])

        # Combine tertiary and secondary features
        combined_features = torch.cat([ter_flat, torch.tensor(sec_features, dtype=torch.float32)])
        return combined_features.numpy()

    def _flatten_dict(self, d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f'{parent_key}{sep}{k}' if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def save(self, path: str):
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer']) 