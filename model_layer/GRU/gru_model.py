import glog
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
import sys

sys.path.append('../')
import xfinai_config
from data_layer.base_dataset import FuturesDatasetRecurrent
from utils import path_wrapper, plotter


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob, device):
        super(GRU, self).__init__()

        self.name = 'GRU'
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_(True).to(self.device)
        x, hn = self.gru(x, h0)
        x = self.fc1(x)
        return x[:, -1, :]


def load_data(future_index):
    train_data = pd.read_pickle(f"{xfinai_config.featured_data_path}/{future_index}_train_data.pkl")
    val_data = pd.read_pickle(f"{xfinai_config.featured_data_path}/{future_index}_val_data.pkl")
    test_data = pd.read_pickle(f"{xfinai_config.featured_data_path}/{future_index}_test_data.pkl")
    return train_data, val_data, test_data


def eval_model(model, dataloader, data_set_name, future_name):
    with torch.no_grad():
        y_real_list = np.array([])
        y_pred_list = np.array([])

        for idx, (x_batch, y_batch) in enumerate(dataloader):
            # Convert to Tensors
            x_batch = x_batch.float().to(model.device)
            y_batch = y_batch.float().to(model.device)

            y_pred = model(x_batch)

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


def train(train_data_loader, model, criterion, optimizer):
    # Set to train mode
    model.train()
    running_train_loss = 0.0

    # Begin training
    for idx, (x_batch, y_batch) in enumerate(train_data_loader):
        optimizer.zero_grad()

        # Convert to Tensors
        x_batch = x_batch.float().to(model.device)
        y_batch = y_batch.float().to(model.device)

        # Make prediction
        y_pred = model(x_batch)

        # Calculate loss
        loss = criterion(y_pred, y_batch)
        loss.backward()
        running_train_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    train_loss_average = running_train_loss / len(train_data_loader)
    return model, train_loss_average


def validate(val_data_loader, model, criterion):
    # Set to eval mode
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for idx, (x_batch, y_batch) in enumerate(val_data_loader):
            # Convert to Tensors
            x_batch = x_batch.float().to(model.device)
            y_batch = y_batch.float().to(model.device)

            y_pred = model(x_batch)

            val_loss = criterion(y_pred, y_batch)
            running_val_loss += val_loss.item()

    val_loss_average = running_val_loss / len(val_data_loader)
    return val_loss_average


def main(future_name, params):
    # seed everything
    seed_everything(xfinai_config.seed)

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

    glog.info(f"Start Training Model")

    # train the model
    for epoch in range(epochs):
        trained_model, train_score = train(train_data_loader=train_loader, model=model, criterion=criterion,
                                           optimizer=optimizer)
        val_score = validate(val_data_loader=val_loader, model=trained_model, criterion=criterion)

        # report intermediate result
        if (epoch + 1) % 10 == 0:
            print(f"Epoch :{epoch} train_score {train_score} val_score {val_score}")
        train_losses.append(train_score)
        val_losses.append(val_score)

    glog.info(f"End Training Model")

    # # save the model
    save_model(trained_model, future_name)

    # plot losses
    plotter.plot_loss(train_losses, epochs, 'Train_Loss', trained_model.name, future_name)
    plotter.plot_loss(val_losses, epochs, 'Val_Loss', trained_model.name, future_name)

    # eval model on 3 datasets
    for dataloader, data_set_name in zip([train_loader, val_loader, test_loader],
                                         ['Train', 'Val', 'Test']):
        eval_model(model=trained_model, dataloader=dataloader, data_set_name=data_set_name,
                   future_name=future_name)


if __name__ == '__main__':
    future_name = 'ic'
    params = {
        "epochs": 100,
        "batch_size": 16,
        "hidden_size": 128,
        "seq_length": 32,
        "weight_decay": 0.00015107010693290502,
        "num_layers": 3,
        "learning_rate": 0.0009239257488546358,
        "dropout_prob": 0.11674166921830238
    }
    main(future_name, params)
