import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D


class DiscreteActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DiscreteActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Actor head for discrete actions (returns logits over discrete actions)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        # Critic head: outputs a scalar value.
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.actor_head(x)
        value = self.value_head(x)
        return logits, value


class IA3CAgent:
    def __init__(self, config, agent_id=0, microgrid_id=0, inverter_id=0):
        self.agent_id = agent_id
        self.microgrid_id = microgrid_id
        self.inverter_id = inverter_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = getattr(config, "state_dim", 4)
        self.action_dim = getattr(config, "action_dim", 10)  # Discrete levels of V_ref
        self.hidden_dim = getattr(config, "hidden_dim", 128)
        self.gamma = getattr(config, "gamma", 0.99)
        self.lr = getattr(config, "lr", 1e-3)
        self.entropy_coef = getattr(config, "entropy_coef", 0.01)
        self.value_loss_coef = getattr(config, "value_loss_coef", 0.5)
        self.v_min = 1.00
        self.v_max = 1.14

        self.network = DiscreteActorCriticNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

    def select_action(self, state):
        state_vector = torch.tensor([state["voltage"], state["delta"], state["i_q"], state["i_d"]],
                                    dtype=torch.float32).to(self.device)
        logits, value = self.network(state_vector)
        dist = D.Categorical(logits=logits)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)

        # Map discrete action index to actual V_ref in range [1.00, 1.14]
        v_ref = self.v_min + (self.v_max - self.v_min) * action_idx.item() / (self.action_dim - 1)
        return v_ref, log_prob, value

    def learn(self, state, log_prob, reward, next_state, done):
        state_vector = torch.tensor([state["voltage"], state["i_d"], state["i_q"], state["delta"]],
                                    dtype=torch.float32).to(self.device)
        next_state_vector = torch.tensor([next_state["voltage"], next_state["i_d"], next_state["i_q"], next_state["delta"]],
                                         dtype=torch.float32).to(self.device)

        logits, value = self.network(state_vector)
        _, next_value = self.network(next_state_vector)

        target = reward + self.gamma * next_value * (1 - int(done))
        advantage = target - value

        actor_loss = -log_prob * advantage.detach()
        critic_loss = advantage.pow(2)
        entropy_loss = -D.Categorical(logits=logits).entropy().mean()

        total_loss = actor_loss + self.value_loss_coef * critic_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)
