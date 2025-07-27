import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as D
import numpy as np
from collections import deque
import random

from torch.distributions import Normal

from utils import helper as helper_module

# Create helper instance
helper = helper_module.Helper()


class ReplayBuffer:
    """Replay buffer for storing experience tuples for centralized SAC agent."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, log_prob, reward, next_state, done):
        """Add experience tuple to buffer."""
        self.buffer.append((state, action, log_prob, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of experiences from buffer."""
        batch = random.sample(self.buffer, batch_size)
        state, action, log_prob, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, log_prob, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class CentralizedPolicyNetwork(nn.Module):
    """Policy network for centralized SAC agent with larger capacity for combined states."""
    
    def __init__(self, state_dim, action_dim):
        super(CentralizedPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        """Forward pass through policy network."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class CentralizedQNetwork(nn.Module):
    """Q-network for centralized SAC agent with larger capacity."""
    
    def __init__(self, state_dim, action_dim):
        super(CentralizedQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.q = nn.Linear(256, 1)

    def forward(self, state, action):
        """Forward pass through Q-network."""
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.q(x)


class CentralizedSACAgent:
    """Centralized SAC agent that combines tertiary and secondary control."""
    
    def __init__(self, state_dim, action_dim, config):
        """
        Initialize centralized SAC agent.
        
        Args:
            state_dim: Combined state dimension (tertiary + secondary)
            action_dim: Combined action dimension (tertiary + secondary)
            config: Configuration object
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = CentralizedPolicyNetwork(state_dim, action_dim).to(self.device)
        self.q1 = CentralizedQNetwork(state_dim, action_dim).to(self.device)
        self.q2 = CentralizedQNetwork(state_dim, action_dim).to(self.device)
        self.q1_target = CentralizedQNetwork(state_dim, action_dim).to(self.device)
        self.q2_target = CentralizedQNetwork(state_dim, action_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=config.q_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=config.q_lr)

        self.memory = ReplayBuffer(config.replay_size)
        self.gamma = config.sac_gamma
        self.tau = config.sac_tau
        self.alpha = config.sac_alpha
        self.batch_size = config.sac_batch_size
        self.config = config

    def flatten_combined_state(self, tertiary_state, secondary_states):
        """
        Flatten and combine tertiary and secondary states.
        
        Args:
            tertiary_state: Tertiary environment state
            secondary_states: List of secondary agent states
            
        Returns:
            Combined flattened state tensor
        """
        # Flatten tertiary state
        ter_flat = helper.flatten_tertiary_state(tertiary_state)
        
        # Flatten secondary states
        sec_features = []
        for sec_state in secondary_states:
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

    def select_action(self, tertiary_state, secondary_states, switch_set, evaluate=False):
        """
        Select action for combined state.
        
        Args:
            tertiary_state: Tertiary environment state
            secondary_states: List of secondary agent states
            switch_set: Set of switches for tertiary actions
            evaluate: Whether in evaluation mode
            
        Returns:
            Tuple of (tertiary_action, secondary_actions, log_prob, q_value)
        """
        combined_state = self.flatten_combined_state(tertiary_state, secondary_states).to(self.device)
        
        if combined_state.dim() == 1:
            combined_state = combined_state.unsqueeze(0)

        with torch.no_grad():
            action, log_prob = self.policy(combined_state)

            if evaluate:
                # Use deterministic action for evaluation
                action = torch.tanh(self.policy.mean(self.policy.fc3(self.policy.fc2(self.policy.fc1(combined_state)))))
                log_prob = torch.zeros_like(action)

            q1_value = self.q1(combined_state, action)
            q2_value = self.q2(combined_state, action)
            q_value = torch.min(q1_value, q2_value)
        
        # Split action into tertiary and secondary parts
        action_np = action.detach().cpu().numpy()[0]
        tertiary_action, secondary_actions = self.unpack_combined_action(action_np, switch_set, len(secondary_states))
        
        return tertiary_action, secondary_actions, log_prob, q_value

    def unpack_combined_action(self, action_vector, switch_set, num_secondary_agents):
        """
        Unpack combined action vector into tertiary and secondary actions.
        
        Args:
            action_vector: Combined action vector
            switch_set: Set of switches for tertiary actions
            num_secondary_agents: Number of secondary agents
            
        Returns:
            Tuple of (tertiary_action_dict, secondary_actions_list)
        """
        if torch.is_tensor(action_vector):
            action_vector = action_vector.detach().cpu().numpy()

        num_microgrids = getattr(self.config, "num_microgrids", 1)
        num_bess_total = getattr(self.config, "num_bess_total", 4)
        num_der_total = getattr(self.config, "num_der_total", 4)
        
        # Calculate tertiary action dimension
        tertiary_action_dim = ( num_der_total + num_bess_total )* num_microgrids + len(switch_set)
        
        # Split action vector
        tertiary_action_vec = action_vector[:tertiary_action_dim]
        secondary_action_vec = action_vector[tertiary_action_dim:]
        
        # Unpack tertiary action
        tertiary_action = helper.unpack_tertiary_action(tertiary_action_vec, switch_set)
        
        # Unpack secondary actions (each agent has 1 action - reactive power)
        secondary_actions = []
        for i in range(num_secondary_agents):
            if i < len(secondary_action_vec):
                secondary_actions.append(secondary_action_vec[i])
            else:
                secondary_actions.append(0.0)  # Default action
        
        return tertiary_action, secondary_actions

    def remember(self, tertiary_state, secondary_states, tertiary_action, secondary_actions, 
                log_prob, reward, next_tertiary_state, next_secondary_states, done):
        """Store experience in replay buffer."""
        combined_state = self.flatten_combined_state(tertiary_state, secondary_states)
        combined_action = self.flatten_combined_action(tertiary_action, secondary_actions)
        combined_next_state = self.flatten_combined_state(next_tertiary_state, next_secondary_states)
        
        self.memory.push(combined_state, combined_action, log_prob, reward, combined_next_state, done)

    def flatten_combined_action(self, tertiary_action, secondary_actions):
        """Flatten combined action into single vector."""
        # Flatten tertiary action
        ter_flat = helper.flatten_tertiary_action(tertiary_action)
        
        # Flatten secondary actions
        sec_flat = np.array(secondary_actions, dtype=np.float32)
        
        # Combine
        combined_action = np.concatenate([ter_flat, sec_flat])
        return combined_action

    def learn(self):
        """Learn from experience in replay buffer."""
        if len(self.memory) < self.batch_size:
            return

        state, action, log_prob, reward, next_state, done = self.memory.sample(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action, next_log_prob = self.policy(next_state)
            target_q1 = self.q1_target(next_state, next_action)
            target_q2 = self.q2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1 = self.q1(state, action)
        current_q2 = self.q2(state, action)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        new_action, log_prob = self.policy(state)
        q_new_action = torch.min(self.q1(state, new_action), self.q2(state, new_action))
        policy_loss = (self.alpha * log_prob - q_new_action).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        # Update target networks
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filepath):
        """Save model weights."""
        torch.save(self.policy.state_dict(), f"{filepath}_policy.pt")
        torch.save(self.q1.state_dict(), f"{filepath}_q1.pt")
        torch.save(self.q2.state_dict(), f"{filepath}_q2.pt")

    def load(self, filepath):
        """Load model weights."""
        self.policy.load_state_dict(torch.load(f"{filepath}_policy.pt"))
        self.q1.load_state_dict(torch.load(f"{filepath}_q1.pt"))
        self.q2.load_state_dict(torch.load(f"{filepath}_q2.pt")) 