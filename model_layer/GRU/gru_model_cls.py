import glog
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
import xfinai_config
from data_layer.base_dataset import FuturesDatasetRecurrent
from utils import path_wrapper, plotter


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob, device):
        super(GRU, self).__init__()

        self.name = 'GRU_Cls'
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def init_hidden_states(self, batch_size):
        state_dim = (self.num_layers, batch_size, self.hidden_size)
        return torch.zeros(state_dim).to(self.device)

    def forward(self, x, states):
        x, h = self.gru(x, states)
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x[:, -1, :], h


def load_data(future_index):
    train_data = pd.read_pickle(f"{xfinai_config.featured_data_path}/{future_index}_train_data.pkl")
    val_data = pd.read_pickle(f"{xfinai_config.featured_data_path}/{future_index}_val_data.pkl")
    test_data = pd.read_pickle(f"{xfinai_config.featured_data_path}/{future_index}_test_data.pkl")
    return train_data, val_data, test_data


def eval_model(model, dataloader, data_set_name, future_name, params):
    with torch.no_grad():
        y_real_list = np.array([])
        y_pred_list = np.array([])
        hidden_state = model.init_hidden_states(params['batch_size'])

        for idx, (x_batch, y_batch) in enumerate(dataloader):
            # Convert to Tensors
            x_batch = x_batch.float().to(model.device)
            y_batch = y_batch.float().to(model.device)

            hidden_state = hidden_state.detach()
            y_pred, hidden_state = model(x_batch, hidden_state)

            y_real_list = np.append(y_real_list, y_batch.squeeze(1).cpu().numpy())
            y_pred_list = np.append(y_pred_list, y_pred.squeeze(1).cpu().numpy())

    plt.figure(figsize=[15, 3], dpi=100)
    plt.plot(y_real_list, label=f'{data_set_name}_real')
    plt.plot(y_pred_list, label=f'{data_set_name}_pred')
    plt.legend()
    plt.title(f"Inference On {data_set_name} Set - {model.name} {future_name}")
    plt.xlabel('Time')
    plt.ylabel('Return')
    plt.subplots_adjust(bottom=0.15)

    result_dir = path_wrapper.wrap_path(f"{xfinai_config.inference_result_path}/{future_name}/{model.name}")
    plt.savefig(f"{result_dir}/{data_set_name}.png")


def save_model(model, future_name):
    dir_path = path_wrapper.wrap_path(f"{xfinai_config.model_save_path}/{future_name}")
    save_path = f"{dir_path}/{model.name}.pth"
    glog.info(f"Starting save model state, save_path: {save_path}")
    torch.save(model.state_dict(), save_path)


def train(train_data_loader, model, criterion, optimizer, params):
    glog.info(f"Start Training Model")

    # Set to train mode
    model.train()
    hidden_state = model.init_hidden_states(params['batch_size'])
    running_train_loss = 0.0

    # Begin training
    for idx, (x_batch, y_batch) in enumerate(train_data_loader):
        optimizer.zero_grad()

        # Convert to Tensors
        x_batch = x_batch.float().to(model.device)
        y_batch = y_batch.float().to(model.device)

        # Truncated Backpropagation
        hidden_state = hidden_state.detach()
        # Make prediction
        y_pred, hidden_state = model(x_batch, hidden_state)

        # Calculate loss
        loss = criterion(y_pred, y_batch)
        loss.backward()
        running_train_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    glog.info(f"End Training Model")

    train_loss_average = running_train_loss / len(train_data_loader)
    return model, train_loss_average


def validate(val_data_loader, model, criterion, params):
    # Set to eval mode
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        hidden_state = model.init_hidden_states(params['batch_size'])

        for idx, (x_batch, y_batch) in enumerate(val_data_loader):
            # Convert to Tensors
            x_batch = x_batch.float().to(model.device)
            y_batch = y_batch.float().to(model.device)

            hidden_state = hidden_state.detach()
            y_pred, hidden_state = model(x_batch, hidden_state)

            val_loss = criterion(y_pred, y_batch)
            running_val_loss += val_loss.item()

    val_loss_average = running_val_loss / len(val_data_loader)
    return val_loss_average


def main(future_name, params):
    # Load Data
    train_data, val_data, test_data = load_data(future_name)

    # Transfer to accelerator
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create dataset & data loader
    train_dataset = FuturesDatasetRecurrent(data=train_data, label=xfinai_config.label, seq_length=params['seq_length'])
    val_dataset = FuturesDatasetRecurrent(data=val_data, label=xfinai_config.label, seq_length=params['seq_length'])
    test_dataset = FuturesDatasetRecurrent(data=test_data, label=xfinai_config.label, seq_length=params['seq_length'])
    train_loader = DataLoader(dataset=train_dataset, **xfinai_config.data_loader_config,
                              batch_size=params['batch_size'])
    val_loader = DataLoader(dataset=val_dataset, **xfinai_config.data_loader_config,
                            batch_size=params['batch_size'])
    test_loader = DataLoader(dataset=test_dataset, **xfinai_config.data_loader_config,
                             batch_size=params['batch_size'])

    # create model instance
    model = GRU(
        input_size=len(train_dataset.features_list),
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        output_size=xfinai_config.model_config['gru']['output_size'],
        dropout_prob=params['dropout_prob'],
        device=device
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(),
                            lr=params['learning_rate'],
                            weight_decay=params['weight_decay'])

    epochs = xfinai_config.model_config['gru']['epochs']

    print(model)
    train_losses = []
    val_losses = []
    # train the model
    for epoch in range(epochs):
        trained_model, train_score = train(train_data_loader=train_loader, model=model, criterion=criterion,
                                           optimizer=optimizer,
                                           params=params)
        val_score = validate(val_data_loader=val_loader, model=trained_model, criterion=criterion, params=params)

        # report intermediate result
        print(f"Epoch :{epoch} train_score {train_score} val_score {val_score}")
        train_losses.append(train_score)
        val_losses.append(val_score)

    # # save the model
    # save_model(trained_model, future_index)

    # plot losses
    plotter.plot_loss(train_losses, epochs, 'Train_Loss', trained_model.name, future_name)
    plotter.plot_loss(val_losses, epochs, 'Val_Loss', trained_model.name, future_name)

    # eval model on 3 datasets
    for dataloader, data_set_name in zip([train_loader, val_loader, test_loader],
                                         ['Train', 'Val', 'Test']):
        eval_model(model=trained_model, dataloader=dataloader, data_set_name=data_set_name,
                   future_name=future_name, params=params)


if __name__ == '__main__':
    future_name = 'ic'
    params = {
        "batch_size": 128,
        "hidden_size": 128,
        "seq_length": 8,
        "weight_decay": 0.004576554841733018,
        "num_layers": 3,
        "learning_rate": 0.09289479785972599,
        "dropout_prob": 0.17641607866906145
    }
    main(future_name, params)
