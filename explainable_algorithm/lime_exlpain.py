import torch
from lime.lime_tabular import RecurrentTabularExplainer
import numpy as np

import matplotlib.pyplot as plt
import sys

sys.path.append("../")
import xfinai_config
from eval_result.model_evaluator import Seq2SeqModelEvaluator, RecurrentModelEvaluator
from model_layer.model_hub import *


class Seq2SeqLimeExplainer(Seq2SeqModelEvaluator):
    def __init__(self, future_index, encoder_class, decoder_class):
        super(Seq2SeqLimeExplainer, self).__init__(future_index, encoder_class, decoder_class)

        self.num_samples = 6400
        self.num_feature = 10
        self.train_x = np.array(self.train_loader.dataset.data_x)
        self.test_x = np.array(self.test_loader.dataset.data_x)
        self.train_y = np.array(self.train_loader.dataset.data_y)
        self.test_y = np.array(self.test_loader.dataset.data_y)
        self.__explainer = None

    @property
    def explainer(self):
        if self.__explainer is None:
            self.__explainer = RecurrentTabularExplainer(training_data=self.train_x, training_labels=self.train_y,
                                                         feature_names=self.train_loader.dataset.features_list,
                                                         discretize_continuous=False, mode='regression')
            return self.__explainer
        else:
            return self.__explainer

    def prediction_func(self, batch_x):
        pre_result = np.array([])
        full_batch_num = batch_x.shape[0] // self.params['batch_size']
        full_batch_length = full_batch_num * self.params['batch_size']
        batch_x = batch_x[:full_batch_length]

        for batch_num in range(full_batch_num):
            current_batch = batch_x[batch_num * self.params['batch_size']: (batch_num + 1) * self.params['batch_size']]
            prediction_raw, _ = self.inference(torch.tensor(current_batch).float().detach().to(self.device))
            prediction = prediction_raw.cpu().detach().numpy()
            pre_result = np.append(pre_result, prediction)
        return pre_result

    def explain_instance(self, random=False):
        if random:
            index = np.random.randint(0, len(self.test_x))
        else:
            index = np.argmax(self.test_y)

        explain = self.explainer.explain_instance(self.test_x[index], self.prediction_func,
                                                  num_features=self.num_feature, num_samples=self.num_samples)
        self.save_lime_result(explain)

    @staticmethod
    def save_lime_result(explain):
        explain.save_to_file(f"{xfinai_config.lime_result_path}/lime_explain.html")
        explain_fig = explain.as_pyplot_figure()
        plt.show()


def main():
    future_index = 'IC'
    encoder_class = EncoderGRU
    decoder_class = AttnDecoderGRU
    explainer = Seq2SeqLimeExplainer(future_index=future_index, encoder_class=encoder_class,
                                     decoder_class=decoder_class)
    explainer.explain_instance()


if __name__ == '__main__':
    main()
