import glog
import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import joblib
import sys

sys.path.append('../')
import xfinai_config
from data_layer.create_data_loader import FuturesDataset
from utils import path_wrapper, plotter


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


def train(train_data_loader, val_data_loader, model,
          criterion, optimizer, epochs, validate_every=1):
    glog.info(f"Start Training Model")
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Set to train mode
        model.train()
        training_states = model.init_hidden_states(xfinai_config.lstm_model_config['batch_size'])
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

        # Average loss across timesteps
        train_losses.append(running_train_loss / len(train_data_loader))

        if epoch % validate_every == 0:
            # Set to eval mode
            model.eval()
            running_val_loss = 0.0
            val_states = model.init_hidden_states(xfinai_config.lstm_model_config['batch_size'])

            for idx, (x_batch, y_batch) in enumerate(val_data_loader):
                # Convert to Tensors
                x_batch = x_batch.float().to(model.device)
                y_batch = y_batch.float().to(model.device)

                val_states = [state.detach() for state in val_states]
                y_pred, val_states = model(x_batch, val_states)

                val_loss = criterion(y_pred, y_batch)
                running_val_loss += val_loss.item()

        val_losses.append(running_val_loss / len(val_data_loader))

        glog.info(f"Epoch:{epoch} train_loss:{running_train_loss} val_loss:{running_val_loss}")

    glog.info(f"End Training Model")
    return train_losses, val_losses, model


def eval_model(model, dataloader, data_set_name, future_name):
    with torch.no_grad():
        y_real_list = np.array([])
        y_pred_list = np.array([])
        states = model.init_hidden_states(xfinai_config.lstm_model_config['batch_size'])

        for idx, (x_batch, y_batch) in enumerate(dataloader):
            # Convert to Tensors
            x_batch = x_batch.float().to(model.device)
            y_batch = y_batch.float().to(model.device)

            states = [state.detach() for state in states]
            y_pred, states = model(x_batch, states)

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


def main(future_name):
    # Load Dataloader
    train_dataloader_list = joblib.load(f'{xfinai_config.data_loader_path}/train_dataloader_list.pkl')
    val_dataloader_list = joblib.load(f'{xfinai_config.data_loader_path}/val_dataloader_list.pkl')
    test_dataloader_list = joblib.load(f'{xfinai_config.data_loader_path}/test_dataloader_list.pkl')

    train_dataloader = train_dataloader_list[xfinai_config.futures_index_map[future_name]]
    val_dataloader = val_dataloader_list[xfinai_config.futures_index_map[future_name]]
    test_dataloader = test_dataloader_list[xfinai_config.futures_index_map[future_name]]

    # Transfer to accelerator
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # create model instance
    model = LSTM(
        input_size=xfinai_config.lstm_model_config['input_size'],
        hidden_size=xfinai_config.lstm_model_config['hidden_size'],
        num_layers=xfinai_config.lstm_model_config['num_layers'],
        output_size=xfinai_config.lstm_model_config['output_size'],
        dropout_prob=xfinai_config.lstm_model_config['dropout_prob'],
        device=device
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.linear.parameters(),
                            lr=xfinai_config.lstm_model_config['learning_rate'],
                            weight_decay=xfinai_config.lstm_model_config['weight_decay'])

    epochs = xfinai_config.lstm_model_config['epochs']

    print(model)

    # train the model
    train_losses, val_losses, trained_model = train(train_dataloader, val_dataloader, model, criterion,
                                                    optimizer, epochs)
    # save the model
    save_model(trained_model, future_name)

    # plot losses
    plotter.plot_loss(train_losses, epochs, 'Train_Loss', trained_model.name, future_name)
    plotter.plot_loss(val_losses, epochs, 'Val_Loss', trained_model.name, future_name)

    # eval model on 3 datasets
    for dataloader, data_set_name in zip([train_dataloader, val_dataloader, test_dataloader],
                                         ['Train', 'Val', 'Test']):
        eval_model(model=trained_model, dataloader=dataloader, data_set_name=data_set_name,
                   future_name=future_name)


if __name__ == '__main__':
    future_name = 'IC'
    main(future_name)
