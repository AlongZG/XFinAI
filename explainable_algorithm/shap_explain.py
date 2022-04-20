import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import joblib
import sys
import shap

sys.path.append("../")
import xfinai_config
from data_layer.base_dataset import FuturesDatasetRecurrent
from model_layer.model_evaluator import Seq2SeqModelEvaluator
from model_layer.model_hub import *
from utils import base_io


class Seq2SeqSHAPExplainer(Seq2SeqModelEvaluator):
    def __init__(self, future_index, encoder_class, decoder_class, params):
        super(Seq2SeqSHAPExplainer, self).__init__(future_index, encoder_class, decoder_class, params)
        self.load_encoder_decoder()
        self.train_data, self.val_data, self.test_data = base_io.load_data(future_index)
        self.__explainer = None
        self.__explain_data_random_batch = None
        self.__explain_data_full_batch = None
        self.seed = 505
        np.random.seed(self.seed)

    @property
    def explainer(self):
        if self.__explainer is None:
            self.__explainer = shap.Explainer(self.prediction_func, self.explain_data_random_batch)
            return self.__explainer

        else:
            return self.__explainer

    @property
    def explain_data_random_batch(self):
        if self.__explain_data_random_batch is None:
            batch_num = self.test_data.shape[0] // self.params["batch_size"]
            batch_num_sample = np.random.choice(batch_num)
            self.__explain_data_random_batch = self.test_data.iloc[batch_num_sample * self.params["batch_size"]:
                                                                   (batch_num_sample + 1) * self.params["batch_size"]]
            return self.__explain_data_random_batch
        else:
            return self.__explain_data_random_batch

    @property
    def explain_data_full_batch(self):
        if self.__explain_data_full_batch is None:
            batch_num = self.test_data.shape[0] // self.params["batch_size"]
            self.__explain_data_full_batch = self.test_data.iloc[: batch_num * self.params["batch_size"]]
            return self.__explain_data_full_batch
        else:
            return self.__explain_data_full_batch

    def create_data_loader(self, data):
        dataset = FuturesDatasetRecurrent(data=data, label=xfinai_config.label, seq_length=self.params['seq_length'])
        loader = DataLoader(dataset=dataset, **xfinai_config.data_loader_config, batch_size=self.params['batch_size'])
        return loader

    def prediction_func(self, train_data):

        full_batch_num = train_data.shape[0] // np.lcm(self.params['batch_size'], self.params['seq_length'])
        full_batch_num += 1

        padded_num = full_batch_num * np.lcm(self.params['batch_size'], self.params['seq_length']) - \
                     train_data.shape[0] + self.params['seq_length']
        train_data_sub = train_data.iloc[-padded_num:]
        train_data_padded = pd.concat([train_data, train_data_sub])

        train_data_loader = self.create_data_loader(train_data_padded)
        pre_result = np.array([])

        with torch.no_grad():
            for idx, (x_batch, y_batch) in enumerate(train_data_loader):
                # Convert to Tensors
                x_batch = x_batch.float().to(self.device)
                prediction_raw, _ = self.inference(x_batch)
                y_pred = prediction_raw.cpu().detach().numpy()
                pre_result = np.append(pre_result, y_pred)

        pre_result_unpadded = pre_result[:-(padded_num - self.params['seq_length'])]

        return pre_result_unpadded

    def calc_shap_values(self):
        # shap_values = self.explainer(self.explain_data_random_batch)
        shap_values = self.explainer(self.explain_data_full_batch)
        return shap_values

    def save_results(self):
        shap_values = self.calc_shap_values()
        joblib.dump(shap_values, f"{xfinai_config.shap_values_path}/"
                                 f"{self.future_index}_{self.model_name}_shap_values.pkl")
        joblib.dump(self.explainer, f"{xfinai_config.shap_values_path}/"
                                    f"{self.future_index}_{self.model_name}_explainer.pkl")
        joblib.dump(self.explain_data_random_batch,
                    f"{xfinai_config.shap_values_path}/"
                    f"{self.future_index}_{self.model_name}_explain_data_random_batch.pkl")


def main():
    future_index = 'IC'
    encoder_class = EncoderGRU
    decoder_class = AttnDecoderGRU
    model_name = f"{EncoderGRU.name}_{AttnDecoderGRU.name}"
    params = base_io.load_best_params(future_index, model_name)
    explainer = Seq2SeqSHAPExplainer(future_index=future_index, encoder_class=encoder_class,
                                     decoder_class=decoder_class, params=params)
    explainer.save_results()


if __name__ == '__main__':
    main()
