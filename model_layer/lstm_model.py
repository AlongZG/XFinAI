import torch 
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
import joblib
import sys

sys.path.append('../')
import xfinai_config
from data_layer.create_data_loader import FuturesDataset


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob, device, directions=1):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.directions = directions
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_size, output_size)

    def init_hidden_states(self, batch_size):
        state_dim = (self.num_layers * self.directions, batch_size, self.hidden_size)
        return (torch.zeros(state_dim).to(self.device), torch.zeros(state_dim).to(self.device))

    def forward(self, x, states):
        x, (h, c) = self.lstm(x, states)
        pred = self.linear(x)
        return pred[:,1,:], (h, c)


def train(train_data_loader, val_data_loader, model,
 criterion, optimizer, epochs, validate_every=1):

    train_losses = []
    validation_losses = []


    for epoch in range(epochs):
        # Set to train mode
        model.train()
        training_states = model.init_hidden_states(xfinai_config.lstm_model_config['batch_size'])
        running_train_loss = 0.0

        # Begin training
        for idx, (x_batch, y_batch) in enumerate(train_data_loader):
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
            optimizer.zero_grad()

        # Average loss across timesteps
        train_losses.append(running_train_loss / len(train_data_loader))

        if epoch % validate_every == 0:
            # Set to eval mode
            model.eval()
            running_validation_loss = 0.0
            validation_states = model.init_hidden_states(xfinai_config.lstm_model_config['batch_size'])

            for idx, (x_batch, y_batch) in enumerate(val_data_loader):

                # Convert to Tensors
                x_batch = x_batch.float().to(model.device)
                y_batch = y_batch.float().to(model.device)

                validation_states = [state.detach() for state in validation_states]
                y_pred, validation_states = model(x_batch, validation_states)

                validation_loss = criterion(y_pred, y_batch)
                running_validation_loss += validation_loss.item()

        validation_losses.append(running_validation_loss / len(val_data_loader))
    
        print(f"Epoch:{epoch} train_loss:{running_train_loss} val_loss:{running_validation_loss}")
        
    return train_losses, validation_losses

def plot_loss(losses, epoch_num, loss_name):
    plt.plot(range(epoch_num), losses, 'r--', label=loss_name)
    plt.legend()
    plt.title(f"{loss_name} Per Epoch")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def main():
    # Load Dataloader
    train_dataloader_list = joblib.load('../data_layer/data_loaders/train_dataloader_list.pkl')
    val_dataloader_list = joblib.load('../data_layer/data_loaders/val_dataloader_list.pkl')
    test_dataloader_list = joblib.load('../data_layer/data_loaders/test_dataloader_list.pkl')

    train_dataloader_ic = train_dataloader_list[0]
    val_dataloader_ic = val_dataloader_list[0]
    test_dataloader_ic = test_dataloader_list[0]

    # Transfer to accelerator
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    model = LSTM(
        input_size=xfinai_config.lstm_model_config['input_size'],
        hidden_size=xfinai_config.lstm_model_config['hidden_size'],
        num_layers=xfinai_config.lstm_model_config['num_layers'],
        output_size=xfinai_config.lstm_model_config['output_size'],
        dropout_prob=xfinai_config.lstm_model_config['dropout_prob'],
        device = device
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.linear.parameters(), 
                lr=xfinai_config.lstm_model_config['learning_rate'],
                weight_decay=xfinai_config.lstm_model_config['weight_decay'])
    epochs = xfinai_config.lstm_model_config['epochs']

    print(model)

    train_losses, validation_losses = train(train_dataloader_ic, val_dataloader_ic, model, criterion, optimizer, epochs)
    plot_loss(train_losses, epochs,'Train Loss')   
    plot_loss(validation_losses, epochs,'Validation Loss')    
 
    return train_losses, validation_losses

if __name__ == '__main__':
    train_losses, validation_losses = main()

