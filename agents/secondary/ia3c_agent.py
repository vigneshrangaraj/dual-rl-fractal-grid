import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D


class DiscreteActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, lstm_hidden_dim=64, use_lstm=True):
        super(DiscreteActorCriticNetwork, self).__init__()
        self.use_lstm = use_lstm
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        if use_lstm:
            self.lstm = nn.LSTM(input_size=state_dim, hidden_size=lstm_hidden_dim, batch_first=True)
            fc_input_dim = lstm_hidden_dim * 2  # includes communication context vector
        else:
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            fc_input_dim = hidden_dim

        # Actor and critic heads
        if use_lstm:
            self.actor_head = nn.Linear(256, action_dim)
            self.value_head = nn.Linear(256, 1)
        else:
            # Actor head for discrete actions (returns logits over discrete actions)
            self.actor_head = nn.Linear(hidden_dim, action_dim)
            # Critic head: outputs a scalar value.
            self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state_seq, hidden=None, context_vector=None):
        """
        Args:
            state_seq: Tensor of shape (batch, seq_len, state_dim) — LSTM input
            hidden: Tuple (h, c) for LSTM hidden state
            context_vector: Tensor of shape (batch, hidden_dim) from attention or mean pooling
        """
        if self.use_lstm:
            """
                    Args:
                        state_seq: Tensor of shape (batch, seq_len, state_dim) — LSTM input
                        hidden: Tuple (h, c) for LSTM hidden state
                        context_vector: Tensor of shape (batch, hidden_dim) from attention or mean pooling

                    """
            if self.use_lstm:
                # Pass through LSTM
                lstm_out, (h_n, c_n) = self.lstm(state_seq, hidden)  # lstm_out: (batch, seq, hidden_dim)
                last_hidden = h_n[-1]  # shape: (batch, hidden_dim)

                # Combine with context vector
                if context_vector is not None:
                    if context_vector.dim() == 2:  # (batch, hidden_dim)
                        x = torch.cat([last_hidden, context_vector], dim=-1)
                    else:
                        raise ValueError(f"context_vector must be 2D, got shape {context_vector.shape}")
                else:
                    x = torch.cat([last_hidden, torch.zeros_like(last_hidden)], dim=-1)
            else:
                # No LSTM path
                x = F.relu(self.fc1(state_seq))  # (batch, hidden)
                x = F.relu(self.fc2(x))  # (batch, hidden)

            logits = self.actor_head(x)  # (batch, action_dim)
            value = self.value_head(x)  # (batch, 1)
            return logits, (h_n, c_n), value
        else:
            x = F.relu(self.fc1(state_seq))
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

        self.network = DiscreteActorCriticNetwork(self.state_dim, self.action_dim, self.hidden_dim, use_lstm=config.use_lstm).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

    def select_action(self, state, hidden=None, context_vector=None):
        if self.network.use_lstm:
            self.network.eval()
            with torch.no_grad():
                # Convert dict to tensor and reshape to 3D for LSTM
                state_vector = torch.tensor(
                    [state["voltage"], state["delta"], state["i_q"], state["i_d"]],
                    dtype=torch.float32
                ).view(1, 1, -1).to(self.device)  # Shape: (1, 1, state_dim)

                logits, (h_n, c_n), value = self.network(state_vector, hidden, context_vector.to(self.device))
                probs = F.softmax(logits, dim=-1)
                dist = D.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            return action.item(), log_prob, h_n, c_n
        else:
            state_vector = torch.tensor([state["voltage"], state["delta"], state["i_q"], state["i_d"]],
                                        dtype=torch.float32).to(self.device)
            logits, value = self.network(state_vector)
            dist = D.Categorical(logits=logits)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)

            # Map discrete action index to actual V_ref in range [1.00, 1.14]
            v_ref = self.v_min + (self.v_max - self.v_min) * action_idx.item() / (self.action_dim - 1)
            return v_ref, log_prob, value

    def learn(self, state, log_prob, reward, next_state, done, hidden=None, next_hidden=None, context_vector=None):
        if self.network.use_lstm:
            # Prepare input tensors (batch=1, seq=1, dim=4)
            state_tensor = torch.tensor(
                [[[state["voltage"], state["i_d"], state["i_q"], state["delta"]]]],
                dtype=torch.float32
            ).to(self.device)  # shape: (1, 1, 4)

            next_state_tensor = torch.tensor(
                [[[next_state["voltage"], next_state["i_d"], next_state["i_q"], next_state["delta"]]]],
                dtype=torch.float32
            ).to(self.device)  # shape: (1, 1, 4)

            # Ensure context is (1, hidden_dim)
            context_vector = context_vector.view(1, -1).to(self.device)

            # Forward pass with context and hidden states
            logits, (h_n, c_n), value = self.network(state_tensor, hidden, context_vector)
            next_logics, (next_h_n, next_c_n), next_value = self.network(next_state_tensor, next_hidden, context_vector)

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
        else:
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

