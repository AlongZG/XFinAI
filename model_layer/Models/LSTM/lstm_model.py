import glog
import torch
from torch import nn, optim
from pytorch_lightning import seed_everything
import sys

sys.path.append('../')
import xfinai_config
from utils import path_wrapper, plotter, base_io
from model_layer.model_hub import LSTM


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

    # Load params
    params = base_io.load_best_params(future_index, LSTM.name)

    # Load Data
    train_loader, val_loader, test_loader = base_io.get_data_loader(future_index, params)

    # Transfer to accelerator
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # create model instance
    model = LSTM(
        input_size=len(train_loader.dataset.features_list),
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        fc_size=params['fc_size'],
        output_size=xfinai_config.model_config[LSTM.name]['output_size'],
        dropout_prob=params['dropout_prob'],
        device=device
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(),
                            lr=params['learning_rate'],
                            weight_decay=params['weight_decay'])

    epochs = params['epochs']

    train_losses = []
    val_losses = []

    # train the model
    glog.info(f"Start Training Model")
    for epoch in range(epochs):
        trained_model, train_loss = train(train_data_loader=train_loader, model=model, criterion=criterion,
                                          optimizer=optimizer)
        validation_loss = validate(val_data_loader=val_loader, model=trained_model, criterion=criterion)

        # report intermediate result
        print(f"Epoch :{epoch} train_loss {train_loss} validation_loss {validation_loss}")
        train_losses.append(train_loss)
        val_losses.append(validation_loss)

    glog.info(f"End Training Model")

    # save the model
    base_io.save_model(model=trained_model, future_index=future_index)

    # plot losses
    plotter.plot_loss(train_losses, epochs, '训练集损失函数值', trained_model.name, future_index)
    plotter.plot_loss(val_losses, epochs, '验证集损失函数值', trained_model.name, future_index)


if __name__ == '__main__':
    future_name = 'IC'
    main(future_name)
