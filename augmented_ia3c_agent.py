
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.optim as optim


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
        self.actor_head = nn.Linear(fc_input_dim, action_dim)
        self.value_head = nn.Linear(fc_input_dim, 1)

    def forward(self, state_seq, hidden=None, context_vector=None):
        """
        state_seq: shape (batch, seq_len, state_dim) if LSTM, else (batch, state_dim)
        context_vector: shape (batch, lstm_hidden_dim), aggregated from other agents
        """
        if self.use_lstm:
            lstm_out, (h_n, c_n) = self.lstm(state_seq, hidden)
            last_hidden = h_n.squeeze(0)  # shape (batch, lstm_hidden_dim)
            if context_vector is not None:
                x = torch.cat([last_hidden, context_vector], dim=-1)
            else:
                x = torch.cat([last_hidden, torch.zeros_like(last_hidden)], dim=-1)
        else:
            x = F.relu(self.fc1(state_seq))
            x = F.relu(self.fc2(x))

        logits = self.actor_head(x)
        value = self.value_head(x)
        return logits, value

class IA3CAgent:
    def __init__(self, config, agent_id):
        self.config = config
        self.agent_id = agent_id
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.lr = config.lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use LSTM-based critic with hidden state communication
        self.network = DiscreteActorCriticNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            use_lstm=True
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

    def select_action(self, obs_seq, hidden, context_vector):
        """
        obs_seq: shape (1, 1, state_dim)
        hidden: (h, c) tuple of LSTM hidden state
        context_vector: shape (1, lstm_hidden_dim)
        """
        self.network.eval()
        with torch.no_grad():
            logits, _ = self.network(obs_seq.to(self.device), hidden, context_vector.to(self.device))
            probs = F.softmax(logits, dim=-1)
            dist = D.Categorical(probs)
            action = dist.sample().item()
        return action, hidden[0], hidden[1]

    def save(self, path_prefix):
        torch.save(self.network.state_dict(), f"{path_prefix}_network.pth")

    def load(self, path_prefix):
        self.network.load_state_dict(torch.load(f"{path_prefix}_network.pth"))

