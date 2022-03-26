import glog
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
import sys

sys.path.append('../')
import xfinai_config
from data_layer.base_dataset import FuturesDatasetRecurrent
from utils import base_io, plotter
from model_layer.model_hub import GRU


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


def main(future_index):
    # seed everything
    seed_everything(xfinai_config.seed)

    # Load Data

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
        fc_size=params['fc_size'],
        device=device
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(),
                            lr=params['learning_rate'],
                            weight_decay=params['weight_decay'])

    epochs = params['epochs']

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
    base_io.save_model(model=trained_model, future_index=future_index)

    # plot losses
    plotter.plot_loss(train_losses, epochs, 'Train_Loss', trained_model.name, future_index)
    plotter.plot_loss(val_losses, epochs, 'Val_Loss', trained_model.name, future_index)



if __name__ == '__main__':
    future_name = 'IC'
    main(future_name)
