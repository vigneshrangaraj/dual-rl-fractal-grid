import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import Any, Tuple
from utils.helper import Helper as helper

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Assumes actions are normalized to [-1, 1]
        )

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class ReplayBuffer:
    def __init__(self, max_size: int = 100000):
        self.buffer = deque(maxlen=max_size)
        self.experience = namedtuple("Experience",
            field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self, batch_size: int):
        experiences = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.FloatTensor(np.array([e.action for e in experiences]))
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.FloatTensor(np.array([e.done for e in experiences])).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class CentralizedDDPGAgent:
    def __init__(self, state_dim: int, action_dim: int, config: Any):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.target_actor = Actor(state_dim, action_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=getattr(config, "policy_lr", 1e-3))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=getattr(config, "q_lr", 1e-3))

        self.replay_buffer = ReplayBuffer(max_size=getattr(config, "replay_size", 100000))
        self.batch_size = getattr(config, "batch_size", 128)
        self.gamma = getattr(config, "gamma", 0.99)
        self.tau = getattr(config, "tau", 0.005)
        self.action_noise = getattr(config, "ddpg_action_noise", 0.1)

    def select_action(self, ter_state, sec_states, switch_set) -> Tuple[np.ndarray, np.ndarray, float, float]:
        # Flatten and concatenate states
        state = self._combine_state(ter_state, sec_states)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()
        self.actor.train()
        # Add exploration noise
        action += self.action_noise * np.random.randn(self.action_dim)
        # Clip to [-1, 1]
        action = np.clip(action, -1, 1)
        # Split action into tertiary and secondary
        ter_action = action[:self.action_dim - len(sec_states)]
        sec_actions = action[self.action_dim - len(sec_states):]
        log_prob = 0.0  # DDPG is deterministic
        q_value = 0.0   # Not used for DDPG action selection
        return helper.unpack_tertiary_action(ter_action, switch_set), sec_actions, log_prob, q_value

    def remember(self, ter_state, sec_states, ter_action, sec_actions, log_prob, reward, next_ter_state, next_sec_states, done):
        state = self._combine_state(ter_state, sec_states)
        next_state = self._combine_state(next_ter_state, next_sec_states)
        ter_action = helper.flatten_tertiary_action(ter_action)
        action = np.concatenate([np.array(ter_action).flatten(), np.array(sec_actions).flatten()])
        self.replay_buffer.add(state, action, reward, next_state, float(done))

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Critic update
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            q_next = self.target_critic(next_states, next_actions)
            q_target = rewards + self.gamma * q_next * (1 - dones)
        q_expected = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_expected, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)

    def _soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def _combine_state(self, ter_state, sec_states):
        # Flatten tertiary state
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
        return combined_features

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
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])