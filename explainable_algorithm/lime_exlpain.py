from lime.lime_tabular import RecurrentTabularExplainer
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
import xfinai_config
from model_layer.model_evaluator import Seq2SeqModelEvaluator
from model_layer.model_hub import *
from utils import plotter, base_io


class Seq2SeqLimeExplainer(Seq2SeqModelEvaluator):
    def __init__(self, future_index, encoder_class, decoder_class, params):
        super(Seq2SeqLimeExplainer, self).__init__(future_index, encoder_class, decoder_class, params)
        self.load_encoder_decoder()
        self.num_samples = 6400
        self.num_feature = 10
        self.train_x = np.array(self.train_loader.dataset.data_x)
        self.test_x = np.array(self.test_loader.dataset.data_x)
        self.train_y = np.array(self.train_loader.dataset.data_y)
        self.test_y = np.array(self.test_loader.dataset.data_y)
        self.__explainer = None
        self.__lime_feature_pattern = r'_t-\d.*'
        self.seed = 505
        np.random.seed(self.seed)

    @property
    def explainer(self):
        if self.__explainer is None:
            self.__explainer = RecurrentTabularExplainer(training_data=self.train_x, training_labels=self.train_y,
                                                         feature_names=self.train_loader.dataset.features_list,
                                                         discretize_continuous=False, mode='regression',
                                                         random_state=self.seed)
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

    def explain_instance(self, index, tag):
        explain = self.explainer.explain_instance(self.test_x[index], self.prediction_func,
                                                  num_features=self.num_feature, num_samples=self.num_samples)
        self.save_lime_result(explain, tag=tag)
        return explain

    def run_lime_explain(self):
        random_index = np.random.randint(0, len(self.test_x))
        max_index = np.argmax(self.test_y)
        min_index = np.argmin(self.test_y)

        random_explain = self.explain_instance(random_index, tag='random')
        max_explain = self.explain_instance(max_index, tag='max')
        min_explain = self.explain_instance(min_index, tag='min')

        return random_explain, max_explain, min_explain

    def lime_value_plotter(self, explain, tag):
        explain_result_list = explain.as_list()
        explain_df = pd.DataFrame(explain_result_list)
        explain_df[0] = explain_df[0].apply(lambda x: re.sub(self.__lime_feature_pattern, '', x))

        explain_df.rename(columns={0: "特征名称", 1: "Lime权重值"}, inplace=True)

        plt.figure(figsize=[10, 6], dpi=300)
        sns.set_theme(style="white", context="talk", palette='summer', rc=plotter.rc_params)
        plt.title(f"Lime算法解释结果")
        sns.barplot(x="Lime权重值", y="特征名称", data=explain_df)
        plt.subplots_adjust(bottom=0.25, left=0.25)
        plt.savefig(f"{xfinai_config.lime_result_path}/lime_explain_{tag}.jpg")

    def save_lime_result(self, explain, tag):
        self.lime_value_plotter(explain, tag)
        explain.save_to_file(f"{xfinai_config.lime_result_path}/lime_explain_{tag}.html")


def main():
    future_index = 'IC'
    encoder_class = EncoderGRU
    decoder_class = AttnDecoderGRU
    model_name = f"{EncoderGRU.name}_{AttnDecoderGRU.name}"
    params = base_io.load_best_params(future_index, model_name)
    explainer = Seq2SeqLimeExplainer(future_index=future_index, encoder_class=encoder_class,
                                     decoder_class=decoder_class, params=params)
    explainer.run_lime_explain()


if __name__ == '__main__':
    main()
