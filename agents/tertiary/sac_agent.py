# agents/tertiary/sac_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
from utils.helper import Helper as helper


# --------------------------
# Network Definitions
# --------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.fc = MLP(state_dim, hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = self.fc(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        std = log_std.exp()
        dist = D.Normal(mean, std)
        z = dist.rsample()  # reparameterization trick
        action = torch.tanh(z)  # ensure bounded output (e.g., between -1 and 1)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(1, keepdim=True)

class SACAgent:
    def __init__(self, state_dim, action_dim, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = getattr(config, "hidden_dim", 256)
        self.gamma = getattr(config, "sac_gamma", 0.99)
        self.tau = getattr(config, "sac_tau", 0.005)
        self.alpha = getattr(config, "sac_alpha", 0.2)
        self.lr = getattr(config, "sac_lr", 3e-4)
        self.batch_size = getattr(config, "sac_batch_size", 64)

        self.policy = GaussianPolicy(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.q1 = MLP(state_dim + action_dim, 1, self.hidden_dim).to(self.device)
        self.q2 = MLP(state_dim + action_dim, 1, self.hidden_dim).to(self.device)
        self.q1_target = MLP(state_dim + action_dim, 1, self.hidden_dim).to(self.device)
        self.q2_target = MLP(state_dim + action_dim, 1, self.hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_opt = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=self.lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=self.lr)

        # Replay buffer and training functions are omitted for brevity.

    def select_action(self, state):
        if isinstance(state, dict):
            state_tensor = helper.flatten_tertiary_state(state).unsqueeze(0).to(self.device)
        elif isinstance(state, (list, np.ndarray)):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state_tensor = state.unsqueeze(0).to(self.device)

        action, log_prob = self.policy(state_tensor)
        # action is of shape [1, action_dim]
        action_dict = helper.unpack_tertiary_action(action[0])
        return action_dict, log_prob.sum(1, keepdim=True), None

    def learn(self, state, action, log_prob, reward, next_state, done):
        state = helper.flatten_tertiary_state(state).to(self.device)
        next_state = helper.flatten_tertiary_state(next_state).to(self.device)
        action = helper.flatten_tertiary_action(action).to(self.device)

        # Compute target Q value
        with torch.no_grad():
            next_action, next_log_prob = self.policy(next_state)
            target_q1 = self.q1_target(torch.cat([next_state, next_action], dim=1))
            target_q2 = self.q2_target(torch.cat([next_state, next_action], dim=1))
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        # Update Q networks
        current_q1 = self.q1(torch.cat([state, action], dim=1))
        current_q2 = self.q2(torch.cat([state, action], dim=1))
        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.q1_opt.zero_grad()
        q_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q_loss.backward()
        self.q2_opt.step()

        # Update policy network
        new_action, new_log_prob = self.policy(state)
        q_new_action = torch.min(self.q1(torch.cat([state, new_action], dim=1)),
                                  self.q2(torch.cat([state, new_action], dim=1)))
        policy_loss = (self.alpha * new_log_prob - q_new_action).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        # Soft update of target networks
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filepath):
        torch.save(self.policy.state_dict(), f"{filepath}_policy.pt")
        torch.save(self.q1.state_dict(), f"{filepath}_q1.pt")
        torch.save(self.q2.state_dict(), f"{filepath}_q2.pt")

    def load(self, filepath):
        self.policy.load_state_dict(torch.load(f"{filepath}_policy.pt"))
        self.q1.load_state_dict(torch.load(f"{filepath}_q1.pt"))
        self.q2.load_state_dict(torch.load(f"{filepath}_q2.pt"))