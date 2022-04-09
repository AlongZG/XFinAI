import glog
import nni
import torch
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt
import os
import time
import sys

sys.path.append('../')
import xfinai_config
from utils import path_wrapper
from model_layer.model_evaluator import RecurrentModelEvaluator, Seq2SeqModelEvaluator


class RecurrentModelTuner(RecurrentModelEvaluator):
    def __init__(self, model_class, future_index, target_metric_func, metric_name, params):
        super().__init__(future_index=future_index, model_class=model_class)

        self.__log_dir = None
        self.__writer = None
        self.target_metric_func = target_metric_func
        self.metric_name = metric_name
        self.params = params

    @property
    def log_dir(self):
        if self.__log_dir is None:
            try:
                self.__log_dir = os.environ['NNI_OUTPUT_DIR']
            except KeyError as e:
                glog.info(f"Fetch KeyError {e}, Regarding As Test Tune Mode, Use Tensorboard Default Log Dir")
                self.__log_dir = f"{xfinai_config.tensorboard_log_default_path}/" \
                                 f"{self.future_index}_{self.model_name}_{time.time()}"
                path_wrapper.wrap_path(self.__log_dir)
        return self.__log_dir

    @property
    def writer(self):
        if self.__writer is None:
            self.__writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'tensorboard'))
        return self.__writer

    def validate(self):
        # Set to eval mode
        self.model.eval()
        running_val_loss = 0.0
        running_val_metric = 0.0

        with torch.no_grad():
            for idx, (x_batch, y_batch) in enumerate(self.val_loader):
                # Convert to Tensors
                x_batch = x_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)

                y_pred = self.model(x_batch)

                val_loss = self.criterion(y_pred, y_batch)
                running_val_loss += val_loss.item()

                y_pred_array = y_pred.to('cpu').squeeze().numpy()
                y_batch_array = y_batch.to('cpu').squeeze().numpy()

                running_val_metric += self.target_metric_func(y_batch_array, y_pred_array)

        val_loss_average = running_val_loss / len(self.val_loader)
        val_metric_average = running_val_metric / len(self.val_loader)
        return val_loss_average, val_metric_average

    def plot_result(self, y_real_list, y_pred_list, data_set_name):
        fig = plt.figure(figsize=[15, 3], dpi=100)
        plt.plot(y_real_list, label=f'{data_set_name}_real')
        plt.plot(y_pred_list, label=f'{data_set_name}_pred')
        plt.legend()
        plt.title(f"{self.future_index}{data_set_name} {self.model_name}模型预测结果")
        plt.xlabel('时间点')
        plt.ylabel('收益率')
        plt.subplots_adjust(bottom=0.15)

        self.writer.add_figure(f"{data_set_name}/{self.future_index}", fig)

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
            validation_loss, target_tune_metric = self.validate()

            # report intermediate result
            nni.report_intermediate_result(target_tune_metric)

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/validation', validation_loss, epoch)
            self.writer.add_scalar(f'{self.metric_name}/validation', target_tune_metric, epoch)

            train_losses.append(train_loss)
            val_losses.append(validation_loss)

        self.eval_model()

        glog.info(f"End Training Model {self.future_index} {self.model_name}")
        self.writer.close()

        # report final result
        nni.report_final_result(target_tune_metric)


class Seq2SeqModelTuner(Seq2SeqModelEvaluator):
    def __init__(self, encoder_class, decoder_class, future_index, target_metric_func, metric_name, params):
        super().__init__(future_index=future_index, encoder_class=encoder_class, decoder_class=decoder_class)

        self.__log_dir = None
        self.__writer = None
        self.target_metric_func = target_metric_func
        self.metric_name = metric_name
        self.params = params

    @property
    def log_dir(self):
        if self.__log_dir is None:
            try:
                self.__log_dir = os.environ['NNI_OUTPUT_DIR']
            except KeyError as e:
                glog.info(f"Fetch KeyError {e}, Regarding As Test Tune Mode, Use Tensorboard Default Log Dir")
                self.__log_dir = f"{xfinai_config.tensorboard_log_default_path}/" \
                                 f"{self.future_index}_{self.model_name}_{time.time()}"
                path_wrapper.wrap_path(self.__log_dir)
        return self.__log_dir

    @property
    def writer(self):
        if self.__writer is None:
            self.__writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'tensorboard'))
        return self.__writer

    def validate(self):
        # Set to eval mode
        self.encoder.eval()
        self.decoder.eval()

        running_val_loss = 0.0
        running_val_metric = 0.0

        with torch.no_grad():
            for idx, (x_batch, y_batch) in enumerate(self.val_loader):
                # Convert to Tensors
                x_batch = x_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)

                y_pred, attn_weights = self.inference(x_batch)

                val_loss = self.criterion(y_pred, y_batch)
                running_val_loss += val_loss.item()

                y_pred_array = y_pred.to('cpu').squeeze().numpy()
                y_batch_array = y_batch.to('cpu').squeeze().numpy()

                running_val_metric += self.target_metric_func(y_batch_array, y_pred_array)

        val_loss_average = running_val_loss / len(self.val_loader)
        val_metric_average = running_val_metric / len(self.val_loader)
        return val_loss_average, val_metric_average

    def plot_result(self, y_real_list, y_pred_list, data_set_name):
        fig = plt.figure(figsize=[15, 3], dpi=100)
        plt.plot(y_real_list, label=f'{data_set_name}_real')
        plt.plot(y_pred_list, label=f'{data_set_name}_pred')
        plt.legend()
        plt.title(f"{self.future_index}{data_set_name} {self.model_name}模型预测结果")
        plt.xlabel('时间点')
        plt.ylabel('收益率')
        plt.subplots_adjust(bottom=0.15)

        self.writer.add_figure(f"{data_set_name}/{self.future_index}", fig)

    def run(self):
        # seed everything
        seed_everything(xfinai_config.seed)

        epochs = self.params['epochs']

        print(self.encoder)
        print(self.decoder)

        train_losses = []
        val_losses = []

        glog.info(f"Start Training Model {self.future_index} {self.model_name}")
        # train the model
        for epoch in range(epochs):
            train_loss = self.train()
            validation_loss, target_tune_metric = self.validate()

            # report intermediate result
            nni.report_intermediate_result(target_tune_metric)

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/validation', validation_loss, epoch)
            self.writer.add_scalar(f'{self.metric_name}/validation', target_tune_metric, epoch)

            train_losses.append(train_loss)
            val_losses.append(validation_loss)

        self.eval_model()

        glog.info(f"End Training Model {self.future_index} {self.model_name}")
        self.writer.close()

        # report final result
        nni.report_final_result(target_tune_metric)
