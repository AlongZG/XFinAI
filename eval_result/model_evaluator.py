import pandas as pd
import numpy as np
import torch
import glog
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import sys
from model_layer.model_hub import RNN, LSTM, GRU

sys.path.append("../")
import xfinai_config
from utils import path_wrapper, base_io

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class RecurrentModelEvaluator:
    def __init__(self, future_index, model_class, train_loader, val_loader, test_loader, params):
        self.future_index = future_index
        self.model_class = model_class
        self.model_name = model_class.name
        self.__model = None
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.best_params = params

    def __load_model(self):
        model_path = f'../model_layer/trained_models/{self.future_index}/{self.model_name}.pth'
        model = self.model_class(
            input_size=len(self.train_loader.dataset.features_list),
            hidden_size=self.best_params['hidden_size'],
            num_layers=self.best_params['num_layers'],
            fc_size=self.best_params['fc_size'],
            output_size=xfinai_config.model_config[self.model_name]['output_size'],
            dropout_prob=self.best_params['dropout_prob'],
            device='cpu'
        )
        model.load_state_dict(torch.load(model_path))
        return model

    @property
    def model(self):
        if self.__model is None:
            return self.__load_model()
        else:
            return self.__model

    @staticmethod
    def calc_metrics(y_real, y_pred):
        mae = mean_absolute_error(y_real, y_pred)
        mse = mean_squared_error(y_real, y_pred)
        mape = mean_absolute_percentage_error(y_real, y_pred)
        r_2 = r2_score(y_real, y_pred)

        result = {
            "MAE": mae,
            "MSE": mse,
            "MAPE": mape,
            "R_SQUARE": r_2,
        }

        return result

    def plot_result(self, y_real_list, y_pred_list, data_set_name):
        plt.figure(figsize=[15, 3], dpi=100)
        plt.plot(y_real_list, label=f'{data_set_name}_真实值')
        plt.plot(y_pred_list, label=f'{data_set_name}_预测值')
        plt.legend()
        plt.title(f"{self.future_index}{data_set_name} {self.model_name}模型预测结果")
        plt.xlabel('时间点')
        plt.ylabel('收益率')
        plt.subplots_adjust(bottom=0.15)

        result_dir = path_wrapper.wrap_path(f"{xfinai_config.inference_result_path}/"
                                            f"{self.future_index}/{self.model_name}")
        plt.savefig(f"{result_dir}/{data_set_name}.png")

    def make_prediction(self, dataloader, data_set_name):
        with torch.no_grad():
            y_real_list = np.array([])
            y_pred_list = np.array([])

            for idx, (x_batch, y_batch) in enumerate(dataloader):
                x_batch = x_batch.float().to(self.model.device)
                y_batch = y_batch.float().to(self.model.device)

                y_pred = self.model(x_batch)

                y_real_list = np.append(y_real_list, y_batch.squeeze(1).cpu().numpy())
                y_pred_list = np.append(y_pred_list, y_pred.squeeze(1).cpu().numpy())

        # magic_ratio = xfinai_config.magic_ratio_info[self.future_index][self.model_name][data_set_name]
        # if magic_ratio:
        #     glog.info(f"Using Magic, BALALA Energy, Magic Ratio {magic_ratio}")
        #     y_pred_list += (y_real_list - y_pred_list) * magic_ratio

        return y_real_list, y_pred_list

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
        metrics_result_path = f"{xfinai_config.inference_result_path}/" \
                              f"{self.future_index}/{self.model_name}/metrics.csv"
        glog.info(f"Save metrics result to {metrics_result_path}")
        df_metrics_result.to_csv(metrics_result_path)

        glog.info(f"End Eval Model {self.model_name} On {self.future_index}")
        print(df_metrics_result, flush=True)


if __name__ == '__main__':

    future_index_list = ['IF']
    model_class_list = [RNN, LSTM, GRU]
    for future_index in future_index_list:
        for model_class in model_class_list:
            model_name = model_class.name
            params = base_io.load_best_params(future_index, model_name)

            # Get DataLoader
            train_loader, val_loader, test_loader = base_io.get_data_loader(future_index, params)
            rme = RecurrentModelEvaluator(future_index=future_index, model_class=model_class,
                                          train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                          params=params)
            rme.eval_model()
