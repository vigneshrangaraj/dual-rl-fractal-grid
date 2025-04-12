# agents/secondary/ia3c_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D


# Define a simple Actor-Critic network.
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Actor head: outputs mean; we use a learnable log_std parameter.
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head: outputs a scalar value.
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        # Use a fixed standard deviation via a learned log_std.
        std = self.log_std.exp().expand_as(mean)
        value = self.value_head(x)
        return mean, std, value


class IA3CAgent:
    def __init__(self, config, agent_id=0):
        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = getattr(config, "state_dim", 2)  # e.g., [voltage, reactive_power]
        self.action_dim = getattr(config, "action_dim", 1)
        self.hidden_dim = getattr(config, "hidden_dim", 128)
        self.gamma = getattr(config, "gamma", 0.99)
        self.lr = getattr(config, "lr", 1e-3)
        self.entropy_coef = getattr(config, "entropy_coef", 0.01)
        self.value_loss_coef = getattr(config, "value_loss_coef", 0.5)

        self.network = ActorCriticNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

    def select_action(self, state):
        state_vector = torch.tensor([state["voltage"], state["reactive_power"]],
                                    dtype=torch.float32).to(self.device)
        mean, std, value = self.network(state_vector)
        # Create a normal distribution from the actor outputs.
        dist = D.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()  # Sum if action_dim > 1
        return action.item(), log_prob, value

    def learn(self, state, action, log_prob, reward, next_state, done):
        # Convert state and next_state into tensors.
        state_vector = torch.tensor([state["voltage"], state["reactive_power"]],
                                    dtype=torch.float32).to(self.device)
        next_state_vector = torch.tensor([next_state["voltage"], next_state["reactive_power"]],
                                         dtype=torch.float32).to(self.device)

        mean, std, value = self.network(state_vector)
        _, _, next_value = self.network(next_state_vector)

        # Compute the target value.
        target = reward + self.gamma * next_value * (1 - int(done))
        advantage = target - value

        # Actor loss: maximize log_prob * advantage.
        actor_loss = -log_prob * advantage.detach()
        # Critic loss: mean squared error.
        critic_loss = advantage.pow(2)
        # Entropy loss for exploration.
        dist_current = D.Normal(mean, std)
        entropy_loss = -dist_current.entropy().mean()

        total_loss = actor_loss + self.value_loss_coef * critic_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def save(self, filepath):
        torch.save(self.network.state_dict(), filepath)

    def load(self, filepath):
        self.network.load_state_dict(torch.load(filepath))