import sys

import nni
from sklearn.metrics import r2_score

sys.path.append('../..')
from model_layer.model_hub import GRU
from model_layer.model_tuner import RecurrentModelTuner
from utils import base_io


def main(model_class, future_index, params):
    target_metric_func = r2_score
    metric_name = 'R_Square'
    tune_model = RecurrentModelTuner(model_class=model_class, future_index=future_index,
                                     target_metric_func=target_metric_func, metric_name=metric_name,
                                     params=params)
    tune_model.run()


if __name__ == '__main__':
    future_index = 'IC'
    model_class = GRU
    # params = base_io.load_best_params(future_index, model_class.name)
    params = nni.get_next_parameter()
    main(model_class, future_index, params)
