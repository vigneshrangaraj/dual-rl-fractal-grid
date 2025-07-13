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
from utils import helper as helper_module

# Create helper instance
helper = helper_module.Helper()


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, log_prob, reward, next_state, done):
        self.buffer.append((state, action, log_prob, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action,log_prob,reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, log_prob, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
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

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

class SACAgent:
    def __init__(self, state_dim, action_dim, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.q1 = QNetwork(state_dim, action_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=config.policy_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=config.q_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=config.q_lr)

        self.memory = ReplayBuffer(config.replay_size)
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.sac_alpha
        self.batch_size = config.sac_batch_size

    def select_action(self, state, switch_set, evaluate=False):
        state = helper.flatten_tertiary_state(state).to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            action, log_prob = self.policy(state)

            if evaluate:
                action = torch.tanh(self.policy.mean_linear(self.policy.net(state)))
                log_prob = torch.zeros_like(action)

            q1_value = self.q1(state, action)
            q2_value = self.q2(state, action)
            ter_value = torch.min(q1_value, q2_value)
        return helper.unpack_tertiary_action(action.detach().cpu().numpy()[0], switch_set), log_prob, ter_value

    def remember(self, state, action, log_prob, reward, next_state, done):
        self.memory.push(helper.flatten_tertiary_state(state), helper.flatten_tertiary_action(action), log_prob, reward, helper.flatten_tertiary_state(next_state), done)

    def learn(self):
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
