import glog
import pandas as pd
import nni
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt
import os
import time
import sys

sys.path.append('../')
import xfinai_config
from utils import base_io, path_wrapper
from eval_result.model_evaluator import RecurrentModelEvaluator


class RecurrentModelTuner(RecurrentModelEvaluator):
    def __init__(self, model_class, future_index, target_metric_func, metric_name, params):
        self.__model = None
        self.__device = None
        self.__log_dir = None
        self.__writer = None
        self.model_class = model_class
        self.future_index = future_index
        self.target_metric_func = target_metric_func
        self.metric_name = metric_name
        self.model_name = model_class.name
        self.params = params
        self.train_loader, self.val_loader, self.test_loader = base_io.get_data_loader(self.future_index,
                                                                                       self.params)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.params['learning_rate'],
                                     weight_decay=self.params['weight_decay'])
        self.criterion = nn.MSELoss()

        super().__init__(future_index=self.future_index, model_class=self.model_class, train_loader=self.train_loader,
                         val_loader=self.val_loader, test_loader=self.test_loader, params=self.params)

    @property
    def device(self):
        if self.__device is None:
            # Transfer to accelerator
            use_cuda = torch.cuda.is_available()
            __device = torch.device("cuda" if use_cuda else "cpu")
            return __device
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

    def train(self):
        # Set to train mode
        self.model.train()
        running_train_loss = 0.0

        # Begin training
        for idx, (x_batch, y_batch) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            # Convert to Tensors
            x_batch = x_batch.float().to(self.device)
            y_batch = y_batch.float().to(self.device)

            # Make prediction
            y_pred = self.model(x_batch)

            # Calculate loss
            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            running_train_loss += loss.item()

            self.optimizer.step()

        train_loss_average = running_train_loss / len(self.train_loader)
        return train_loss_average

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

    def eval_model(self):
        glog.info(f"Start Eval Model {self.model_name} On {self.future_index}")
        metrics_result_list = {}
        for dataloader, data_set_name in zip([self.train_loader, self.val_loader, self.test_loader],
                                             ['训练集', '验证集', '测试集']):
            y_real_list, y_pred_list = self.make_prediction(dataloader, data_set_name)

            glog.info(f"Plot Result {self.model_name} {self.future_index} {data_set_name}")
            self.plot_result(y_real_list, y_pred_list, data_set_name)

            glog.info(f"Calc Metrics {self.model_name} {self.future_index} {data_set_name}")
            metrics_result = self.calc_metrics(y_real_list, y_pred_list)
            metrics_result_list.update({data_set_name: metrics_result})

        df_metrics_result = pd.DataFrame(metrics_result_list)
        metrics_result_path = f"{self.log_dir}/metrics.csv"
        glog.info(f"Save metrics result to {metrics_result_path}")
        df_metrics_result.to_csv(metrics_result_path)

        glog.info(f"End Eval Model {self.model_name} On {self.future_index}")
        print(df_metrics_result, flush=True)

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

