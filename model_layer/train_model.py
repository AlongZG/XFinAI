import sys

sys.path.append('../')
from model_layer.model_hub import RNN, LSTM, GRU
from model_layer.model_trainer import RecurrentModelTrainer


def main(future_index_list, model_class_list):
    for future_index in future_index_list:
        for model_class in model_class_list:
            train_model = RecurrentModelTrainer(model_class=model_class, future_index=future_index)
            train_model.run()


if __name__ == '__main__':
    future_list = ['IF', 'IH']
    model_list = [RNN, LSTM, GRU]
    main(future_list, model_list)
