import glog
import torch
from torch import nn, optim
from pytorch_lightning import seed_everything
import sys

sys.path.append('../')
import xfinai_config
from utils import plotter, base_io


class RecurrentModelTrainer:
    def __init__(self, model_class, future_index, params):
        self.__model = None
        self.__device = None
        self.model_class = model_class
        self.future_index = future_index
        self.model_name = model_class.name
        self.__params = params
        self.train_loader, self.val_loader, self.test_loader = base_io.get_data_loader(self.future_index,
                                                                                       self.__params)
        self.__optimizer = optim.AdamW(self.model.parameters(), lr=self.params['learning_rate'],
                                       weight_decay=self.params['weight_decay'])
        self.criterion = nn.MSELoss()

    @property
    def params(self):
        self.__params.update({"input_size": len(self.train_loader.dataset.features_list),
                              "device": self.device,
                              "output_size": xfinai_config.model_config[self.model_name]['output_size']})
        return self.__params

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
            self.__model = self.model_class(self.params).to(self.device)
            return self.__model
        else:
            return self.__model

    @model.setter
    def model(self, __model):
        self.__model = __model

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
            loss = self.criterion(y_pred, y_batch)
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

                val_loss = self.criterion(y_pred, y_batch)
                running_val_loss += val_loss.item()

        val_loss_average = running_val_loss / len(self.val_loader)
        return val_loss_average

    def run(self):
        # seed everything
        seed_everything(xfinai_config.seed, workers=True)

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


class Seq2SeqModelTrainer:
    def __init__(self, future_index, encoder_class, decoder_class, params):
        self.__encoder = None
        self.__decoder = None
        self.__device = None
        self.__params = params
        self.encoder_class = encoder_class
        self.decoder_class = decoder_class
        self.future_index = future_index
        self.model_name = f"{self.encoder_class.name}_{self.decoder_class.name}"
        self.train_loader, self.val_loader, self.test_loader = base_io.get_data_loader(self.future_index,
                                                                                       self.__params)
        self.__encoder_optimizer = optim.AdamW(self.encoder.parameters(), lr=self.params['learning_rate'],
                                               weight_decay=self.params['weight_decay'])
        self.__decoder_optimizer = optim.AdamW(self.decoder.parameters(), lr=self.params['learning_rate'],
                                               weight_decay=self.params['weight_decay'])
        self.criterion = nn.MSELoss()

    @property
    def params(self):
        if "input_size" not in self.__params:
            self.__params.update({"input_size": len(self.train_loader.dataset.features_list),
                                  "device": self.device,
                                  "output_size": xfinai_config.model_config[self.model_name]['output_size']})
            return self.__params
        else:
            return self.__params

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
    def encoder(self):
        if self.__encoder is None:
            self.__encoder = self.encoder_class(self.params).to(self.device)
            return self.__encoder
        else:
            return self.__encoder

    @property
    def decoder(self):
        if self.__decoder is None:
            self.__decoder = self.decoder_class(self.params).to(self.device)
            return self.__decoder
        else:
            return self.__decoder

    @encoder.setter
    def encoder(self, model):
        self.__encoder = model

    @decoder.setter
    def decoder(self, model):
        self.__decoder = model

    def inference(self, x_batch):
        encoder_outputs = torch.zeros(self.params['seq_length'], self.params['batch_size'],
                                      self.params['hidden_size'], device=self.device)

        for time_step in range(self.params['seq_length']):
            x_time_step = x_batch[:, time_step, :]
            encoder_hidden = self.encoder.init_hidden()
            encoder_output, encoder_hidden = self.encoder(x_time_step, encoder_hidden)
            encoder_outputs[time_step] = encoder_output[:, 0, :]

        decoder_input = torch.zeros(self.params['batch_size'], 1, device=self.encoder.device)
        decoder_hidden = encoder_hidden

        for time_step in range(self.params['seq_length']):
            output, hidden, attn_weights = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = output

        return output, attn_weights

    def train(self):

        # Set to train mode
        self.encoder.train()
        self.decoder.train()
        running_train_loss = 0.0

        # Begin training
        for idx, (x_batch, y_batch) in enumerate(self.train_loader):
            self.__encoder_optimizer.zero_grad()
            self.__decoder_optimizer.zero_grad()

            # Convert to Tensors
            x_batch = x_batch.float().to(self.device)
            y_batch = y_batch.float().to(self.device)

            # Make prediction
            y_pred, attn_weights = self.inference(x_batch)

            # Calculate loss
            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            running_train_loss += loss.item()

            self.__encoder_optimizer.step()
            self.__decoder_optimizer.step()

        train_loss_average = running_train_loss / len(self.train_loader)

        return train_loss_average

    def validate(self):
        # Set to eval mode
        self.encoder.eval()
        self.decoder.eval()

        running_val_loss = 0.0
        with torch.no_grad():
            for idx, (x_batch, y_batch) in enumerate(self.val_loader):
                # Convert to Tensors
                x_batch = x_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)

                y_pred, attn_weights = self.inference(x_batch)

                val_loss = self.criterion(y_pred, y_batch)
                running_val_loss += val_loss.item()

        val_loss_average = running_val_loss / len(self.val_loader)
        return val_loss_average

    def run(self):

        epochs = self.params['epochs']

        print(self.encoder)
        print(self.decoder)

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
        base_io.save_model(model=(self.encoder, self.decoder), future_index=self.future_index, seq2seq=True)

        # plot losses
        plotter.plot_loss(train_losses, epochs, '训练集损失函数值', self.model_name, self.future_index)
        plotter.plot_loss(val_losses, epochs, '验证集损失函数值', self.model_name, self.future_index)
