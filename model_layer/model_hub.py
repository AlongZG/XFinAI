import torch
from torch import nn
import torch.nn.functional as F


class RNNBase(nn.Module):

    def __init__(self, params, rnn_cell):
        super().__init__()

        self.input_size = params['input_size']
        self.num_layers = params['num_layers']
        self.hidden_size = params['hidden_size']
        self.fc_size = params['fc_size']
        self.device = params['device']
        self.dropout_prob = params['dropout_prob']
        self.output_size = params["output_size"]
        self.rnn = rnn_cell(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            batch_first=True, dropout=self.dropout_prob)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc1 = nn.Linear(self.hidden_size, self.fc_size)
        self.fc2 = nn.Linear(self.fc_size, self.output_size)

    def init_hidden_states(self, state_dim):
        return torch.zeros(state_dim).requires_grad_(True).to(self.device)

    def forward(self, x):
        batch_size = x.shape[0]
        state_dim = (self.num_layers, batch_size, self.hidden_size)
        hidden = self.init_hidden_states(state_dim)
        x, hidden = self.rnn(x, hidden)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x[:, -1, :]


class RNN(RNNBase):
    name = 'RNN'

    def __init__(self, params):
        super(RNN, self).__init__(params, nn.RNN)


class LSTM(RNNBase):
    name = 'LSTM'

    def __init__(self, params):
        super(LSTM, self).__init__(params, nn.LSTM)

    def init_hidden_states(self, state_dim):
        return torch.zeros(state_dim).requires_grad_(True).to(self.device), \
               torch.zeros(state_dim).requires_grad_(True).to(self.device)


class GRU(RNNBase):
    name = 'GRU'

    def __init__(self, params):
        super(GRU, self).__init__(params, nn.GRU)


class EncoderGRU(nn.Module):
    name = 'EncoderGRU'

    def __init__(self, params):
        super(EncoderGRU, self).__init__()
        self.input_size = params["input_size"]
        self.batch_size = params["batch_size"]
        self.output_size = params["output_size"]
        self.hidden_size = params["hidden_size"]
        self.device = params["device"]
        self.dropout_prob = params["dropout_prob"]
        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True,
                          dropout=self.dropout_prob)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, x, hidden):
        x = x.view(self.batch_size, 1, -1)
        x, hidden = self.gru(x, hidden)
        return x, hidden

    def init_hidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)


class AttnDecoderGRU(nn.Module):
    name = 'AttnDecoderGRU'

    def __init__(self, params):
        super(AttnDecoderGRU, self).__init__()
        self.input_size = params["input_size"]
        self.batch_size = params["batch_size"]
        self.output_size = params["output_size"]
        self.seq_length = params["seq_length"]
        self.hidden_size = params["hidden_size"]
        self.device = params["device"]
        self.dropout_prob = params["dropout_prob"]

        self.attn = nn.Linear(self.hidden_size + self.output_size, self.seq_length)
        self.attn_combine = nn.Linear(self.hidden_size + self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, last_input, hidden, encoder_outputs):
        attn_weights = F.softmax(
            self.attn(torch.cat((last_input, hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.transpose(0, 1)).squeeze(1)
        output = torch.cat((last_input, attn_applied), 1)
        output = self.attn_combine(output).unsqueeze(1)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output[:, 0, :], hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)
