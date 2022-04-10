import pandas as pd
import numpy as np
import torch
import glog
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
    mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import sys
from model_layer.model_hub import RNN, LSTM, GRU, EncoderGRU, AttnDecoderGRU
from model_layer.model_trainer import RecurrentModelTrainer, Seq2SeqModelTrainer

sys.path.append("../")
import xfinai_config
from utils import path_wrapper, base_io

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class RecurrentModelEvaluator(RecurrentModelTrainer):
    def __init__(self, future_index, model_class):
        super().__init__(future_index=future_index, model_class=model_class)
        self.__time_step_map = None
        self.__tick_interval_map = None

    def __load_model(self):
        self.model = base_io.load_model(self.model, self.future_index)

    @property
    def time_step_map(self):
        if self.__time_step_map is None:
            self.__time_step_map = {dataset_name: data.index for data, dataset_name in zip(
                base_io.load_data(self.future_index), ['训练集', '验证集', '测试集'])}
            return self.__time_step_map
        else:
            return self.__time_step_map

    @property
    def tick_interval_map(self):
        if self.__tick_interval_map is None:
            self.__tick_interval_map = {dataset_name: interval for interval, dataset_name in zip(
                [2200, 590, 290], ['训练集', '验证集', '测试集'])}
            return self.__tick_interval_map
        else:
            return self.__tick_interval_map

    @staticmethod
    def calc_metrics(y_real, y_pred):
        mae = mean_absolute_error(y_real, y_pred)
        mse = mean_squared_error(y_real, y_pred)
        mape = mean_absolute_percentage_error(y_real[y_real != 0], y_pred[y_real != 0])
        r_2 = r2_score(y_real, y_pred)

        result = {
            "MAE": mae,
            "MSE": mse,
            "MAPE": mape,
            "R_SQUARE": r_2,
        }

        return result

    def plot_result(self, y_real_list, y_pred_list, data_set_name):
        time_steps = self.time_step_map[data_set_name][:len(y_real_list)]
        plt.figure(figsize=[15, 3], dpi=100)
        plt.plot(y_real_list, label=f'{data_set_name}_真实值')
        plt.plot(y_pred_list, label=f'{data_set_name}_预测值')
        x_ticks = range(0, len(y_real_list), self.tick_interval_map[data_set_name])
        plt.xticks(ticks=x_ticks, labels=[time_steps[i] for i in x_ticks])
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

    def eval_model(self, tune_mode=False):
        glog.info(f"Start Eval Model {self.model_name} On {self.future_index}")
        if not tune_mode:
            self.__load_model()
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


class Seq2SeqModelEvaluator(Seq2SeqModelTrainer):
    def __init__(self, future_index, encoder_class, decoder_class):
        super().__init__(future_index, encoder_class, decoder_class)
        self.attention_weights_map = {}

        self.__time_step_map = None
        self.__tick_interval_map = None

    def __load_encoder_decoder(self):
        self.encoder, self.decoder = base_io.load_model((self.encoder, self.decoder),
                                                        future_index=self.future_index, seq2seq=True)

    @property
    def time_step_map(self):
        if self.__time_step_map is None:
            self.__time_step_map = {dataset_name: data.index for data, dataset_name in zip(
                base_io.load_data(self.future_index), ['训练集', '验证集', '测试集'])}
            return self.__time_step_map
        else:
            return self.__time_step_map

    @property
    def tick_interval_map(self):
        if self.__tick_interval_map is None:
            self.__tick_interval_map = {dataset_name: interval for interval, dataset_name in zip(
                [2200, 590, 290], ['训练集', '验证集', '测试集'])}
            return self.__tick_interval_map
        else:
            return self.__tick_interval_map

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
        time_steps = self.time_step_map[data_set_name][:len(y_real_list)]
        plt.figure(figsize=[15, 3], dpi=100)
        plt.plot(y_real_list, label=f'{data_set_name}_真实值')
        plt.plot(y_pred_list, label=f'{data_set_name}_预测值')
        x_ticks = range(0, len(y_real_list), self.tick_interval_map[data_set_name])
        plt.xticks(ticks=x_ticks, labels=[time_steps[i] for i in x_ticks])
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
            attn_weights_list = []

            for idx, (x_batch, y_batch) in enumerate(dataloader):
                x_batch = x_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)

                y_pred, raw_attn_weights = self.inference(x_batch)

                y_real_list = np.append(y_real_list, y_batch.squeeze(1).cpu().numpy())
                y_pred_list = np.append(y_pred_list, y_pred.squeeze(1).cpu().numpy())
                attn_weights = raw_attn_weights.cpu().numpy()
                attn_weights_list.append(attn_weights)

        self.attention_weights_map[data_set_name] = attn_weights_list

        # magic_ratio = xfinai_config.magic_ratio_info[self.future_index][self.model_name][data_set_name]
        # if magic_ratio:
        #     glog.info(f"Using Magic, BALALA Energy, Magic Ratio {magic_ratio}")
        #     y_pred_list += (y_real_list - y_pred_list) * magic_ratio

        return y_real_list, y_pred_list

    def eval_model(self, tune_mode=False):
        glog.info(f"Start Eval Model {self.model_name} On {self.future_index}")
        if not tune_mode:
            self.__load_encoder_decoder()
        metrics_result_list = {}
        for dataloader, data_set_name in zip([self.train_loader, self.val_loader, self.test_loader],
                                             ['训练集', '验证集', '测试集']):
            y_real_list, y_pred_list = self.make_prediction(dataloader, data_set_name)

            glog.info(f"Plot Result {self.model_name} {self.future_index} {data_set_name}")
            self.plot_result(y_real_list, y_pred_list, data_set_name)

            glog.info(f"Calc Metrics {self.model_name} {self.future_index} {data_set_name}")
            metrics_result = self.calc_metrics(y_real_list, y_pred_list)
            metrics_result_list.update({data_set_name: metrics_result})

        df_metrics_result = base_io.save_metrics_result(metrics_result_list, self.future_index, self.model_name)
        base_io.save_attention_weights(self.attention_weights_map, self.future_index, self.model_name)

        glog.info(f"End Eval Model {self.model_name} On {self.future_index}")
        print(df_metrics_result, flush=True)


def eval_recurrent_model():
    future_index_list = ['IC']
    model_class_list = [RNN, LSTM, GRU]
    for future_index in future_index_list:
        for model_class in model_class_list:
            rme = RecurrentModelEvaluator(future_index=future_index, model_class=model_class)
            rme.eval_model()


def eval_seq2seq_model():
    future_index_list = ['IC']
    for future_index in future_index_list:
        sme = Seq2SeqModelEvaluator(future_index=future_index, encoder_class=EncoderGRU, decoder_class=AttnDecoderGRU)
        sme.eval_model()


if __name__ == '__main__':
    eval_seq2seq_model()
    # eval_recurrent_model()
