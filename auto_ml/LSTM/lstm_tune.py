import glog
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pandas as pd
import sys
import nni

sys.path.append('../..')
import xfinai_config
from data_layer.base_dataset import FuturesDataset


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob, device, directions=1):
        super(LSTM, self).__init__()

        self.name = 'LSTM'
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.directions = directions
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_size, output_size)

    def init_hidden_states(self, batch_size):
        state_dim = (self.num_layers * self.directions, batch_size, self.hidden_size)
        return torch.zeros(state_dim).to(self.device), torch.zeros(state_dim).to(self.device)

    def forward(self, x, states):
        x, (h, c) = self.lstm(x, states)
        pred = self.linear(x)
        return pred[:, 1, :], (h, c)


def load_data(future_index):
    train_data = pd.read_pickle(f"{xfinai_config.featured_data_path}/{future_index}_train_data.pkl")
    val_data = pd.read_pickle(f"{xfinai_config.featured_data_path}/{future_index}_val_data.pkl")
    test_data = pd.read_pickle(f"{xfinai_config.featured_data_path}/{future_index}_test_data.pkl")
    return train_data, val_data, test_data


def train(train_data_loader, model, criterion, optimizer, params):
    glog.info(f"Start Training Model")

    # Set to train mode
    model.train()
    training_states = model.init_hidden_states(params['batch_size'])
    running_train_loss = 0.0

    # Begin training
    for idx, (x_batch, y_batch) in enumerate(train_data_loader):
        optimizer.zero_grad()

        # Convert to Tensors
        x_batch = x_batch.float().to(model.device)
        y_batch = y_batch.float().to(model.device)

        # Truncated Backpropagation
        training_states = [state.detach() for state in training_states]
        # Make prediction
        y_pred, training_states = model(x_batch, training_states)

        # Calculate loss
        loss = criterion(y_pred, y_batch)
        loss.backward()
        running_train_loss += loss.item()

        optimizer.step()

    glog.info(f"End Training Model")


def test(val_data_loader, model, criterion, params):
    # Set to eval mode
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        val_states = model.init_hidden_states(params['batch_size'])

        for idx, (x_batch, y_batch) in enumerate(val_data_loader):
            # Convert to Tensors
            x_batch = x_batch.float().to(model.device)
            y_batch = y_batch.float().to(model.device)

            val_states = [state.detach() for state in val_states]
            y_pred, val_states = model(x_batch, val_states)

            val_loss = criterion(y_pred, y_batch)
            running_val_loss += val_loss.item()

    val_loss_average = running_val_loss / len(val_data_loader)
    return val_loss_average


def main(params, future_index):
    train_data, val_data, test_data = load_data(future_index)

    # Transfer to accelerator
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create dataset & data loader
    train_dataset = FuturesDataset(data=train_data, label=xfinai_config.label, seq_length=params['seq_length'],
                                   features_list=xfinai_config.features_list)
    val_dataset = FuturesDataset(data=train_data, label=xfinai_config.label, seq_length=params['seq_length'],
                                 features_list=xfinai_config.features_list)
    train_loader = DataLoader(dataset=train_dataset, **xfinai_config.data_loader_config,
                              batch_size=params['batch_size'])
    val_loader = DataLoader(dataset=val_dataset, **xfinai_config.data_loader_config,
                            batch_size=params['batch_size'])

    # create model instance
    model = LSTM(
        input_size=xfinai_config.lstm_model_config['input_size'],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        output_size=xfinai_config.lstm_model_config['output_size'],
        dropout_prob=params['dropout_prob'],
        device=device
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.linear.parameters(),
                            lr=params['learning_rate'],
                            weight_decay=params['weight_decay'])

    epochs = xfinai_config.lstm_model_config['epochs']
    print(model)

    # train the model
    for epoch in range(epochs):
        train(train_data_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, params=params)
        test_acc = test(val_data_loader=val_loader, model=model, criterion=criterion, params=params)

        # report intermediate result
        nni.report_intermediate_result(test_acc)
        # print(test_acc)

    # report final result
    nni.report_final_result(test_acc)
    # print(test_acc)


if __name__ == '__main__':
    future_name = 'ic'
    train_params = nni.get_next_parameter()
    # train_params = {
    #     "batch_size": 128,
    #     "hidden_size": 4,
    #     "weight_decay": 0.0001,
    #     "num_layers": 2,
    #     "learning_rate": 0.1,
    #     "dropout_prob": 0.1
    # }
    main(train_params, future_name)
