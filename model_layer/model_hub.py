import torch
from torch import nn
import torch.nn.functional as F


class RNN(nn.Module):
    name = 'RNN'

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob, fc_size, device):
        super().__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.fc_size = fc_size
        self.device = device

        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, self.fc_size)
        self.fc2 = nn.Linear(self.fc_size, output_size)

    def init_hidden_states(self, state_dim):
        return torch.zeros(state_dim).requires_grad_(True).to(self.device)

    def forward(self, x):
        batch_size = x.shape[0]
        state_dim = (self.num_layers, batch_size, self.hidden_size)
        h0 = self.init_hidden_states(state_dim)
        x, h = self.rnn(x, h0)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x[:, -1, :]


class GRU(nn.Module):
    name = 'GRU'

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob, fc_size, device):
        super().__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.fc2 = nn.Linear(fc_size, output_size)

    def init_hidden_states(self, state_dim):
        return torch.zeros(state_dim).requires_grad_(True).to(self.device)

    def forward(self, x):
        batch_size = x.shape[0]
        state_dim = (self.num_layers, batch_size, self.hidden_size)
        h0 = self.init_hidden_states(state_dim)
        x, hn = self.gru(x, h0)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x[:, -1, :]


class LSTM(nn.Module):
    name = 'LSTM'

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob, device, fc_size):
        super().__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.fc2 = nn.Linear(fc_size, output_size)

    def init_hidden_states(self, state_dim):
        return torch.zeros(state_dim).requires_grad_(True).to(self.device), \
               torch.zeros(state_dim).requires_grad_(True).to(self.device)

    def forward(self, x):
        batch_size = x.shape[0]
        state_dim = (self.num_layers, batch_size, self.hidden_size)
        h0, c0 = self.init_hidden_states(state_dim)

        x, (h, c) = self.lstm(x, (h0, c0))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x[:, -1, :]
