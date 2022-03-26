import glog
import torch
from torch import nn, optim
from pytorch_lightning import seed_everything
import sys

sys.path.append('../')
import xfinai_config
from utils import plotter, base_io


class RecurrentModelTrainer:
    def __init__(self, model_class, future_index):
        self.__model = None
        self.__device = None
        self.model_class = model_class
        self.future_index = future_index
        self.model_name = model_class.name
        self.params = base_io.load_best_params(future_index=self.future_index, model_name=self.model_name)
        self.train_loader, self.val_loader, self.test_loader = base_io.get_data_loader(self.future_index,
                                                                                       self.params)
        self.__optimizer = optim.AdamW(self.model.parameters(), lr=self.params['learning_rate'],
                                       weight_decay=self.params['weight_decay'])
        self.__criterion = nn.MSELoss()

    @property
    def device(self):
        if self.__device is None:
            # Transfer to accelerator
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            return device
        else:
            return self.__device

    @property
    def model(self):
        if self.__model is None:
            self.__model = self.model_class(
                input_size=len(self.train_loader.dataset.features_list),
                hidden_size=self.params['hidden_size'],
                num_layers=self.params['num_layers'],
                fc_size=self.params['fc_size'],
                output_size=xfinai_config.model_config[self.model_class.name]['output_size'],
                dropout_prob=self.params['dropout_prob'],
                device=self.device
            ).to(self.device)
            return self.__model
        else:
            return self.__model

    def train(self):
        # Set to train mode
        self.model.train()
        running_train_loss = 0.0

        # Begin training
        for idx, (x_batch, y_batch) in enumerate(self.train_loader):
            self.__optimizer.zero_grad()

            # Convert to Tensors
            x_batch = x_batch.float().to(self.device)
            y_batch = y_batch.float().to(self.device)

            # Make prediction
            y_pred = self.model(x_batch)

            # Calculate loss
            loss = self.__criterion(y_pred, y_batch)
            loss.backward()
            running_train_loss += loss.item()

            self.__optimizer.step()

        train_loss_average = running_train_loss / len(self.train_loader)
        return train_loss_average

    def validate(self):
        # Set to eval mode
        self.model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for idx, (x_batch, y_batch) in enumerate(self.val_loader):
                # Convert to Tensors
                x_batch = x_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)

                y_pred = self.model(x_batch)

                val_loss = self.__criterion(y_pred, y_batch)
                running_val_loss += val_loss.item()

        val_loss_average = running_val_loss / len(self.val_loader)
        return val_loss_average

    def run(self):
        # seed everything
        seed_everything(xfinai_config.seed)

        epochs = self.params['epochs']

        print(self.model)
        train_losses = []
        val_losses = []

        glog.info(f"Start Training Model {self.future_index} {self.model_name}")
        # train the model
        for epoch in range(epochs):
            train_loss = self.train()
            val_loss = self.validate()

            # report intermediate result
            print(f"Epoch :{epoch} train_loss {train_loss} val_loss {val_loss}")
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        glog.info(f"End Training Model {self.future_index} {self.model_name}")

        # save the model
        base_io.save_model(model=self.model, future_index=self.future_index)

        # plot losses
        plotter.plot_loss(train_losses, epochs, '训练集损失函数值', self.model.name, self.future_index)
        plotter.plot_loss(val_losses, epochs, '验证集损失函数值', self.model.name, self.future_index)
