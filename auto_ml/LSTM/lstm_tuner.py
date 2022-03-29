import sys
from sklearn.metrics import r2_score

sys.path.append('../')
from model_layer.model_hub import LSTM
from model_layer.model_tuner import RecurrentModelTuner


def main(model_class, future_index):
    target_metric_func = r2_score
    metric_name = 'R_Square'
    train_model = RecurrentModelTuner(model_class=model_class, future_index=future_index,
                                      target_metric_func=target_metric_func, metric_name=metric_name)
    train_model.run()


if __name__ == '__main__':
    future_index = 'IC'
    model_class = LSTM
    main(model_class, future_index)
