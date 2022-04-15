import sys
from pytorch_lightning import seed_everything

sys.path.append('../')
import xfinai_config
from model_layer.model_hub import RNN, LSTM, GRU, EncoderGRU, AttnDecoderGRU
from model_layer.model_trainer import RecurrentModelTrainer, Seq2SeqModelTrainer
from utils import base_io


def train_recurrent_model():
    future_list = ['IC', 'IH', 'IF']
    model_list = [RNN, LSTM, GRU]
    for future_index in future_list:
        for model_class in model_list:
            params = base_io.load_best_params(future_index, model_class.name)
            train_model = RecurrentModelTrainer(model_class=model_class, future_index=future_index, params=params)
            train_model.run()


def train_seq2seq_model():
    future_list = ['IH']
    for future_index in future_list:
        model_name = f"{EncoderGRU.name}_{AttnDecoderGRU.name}"
        params = base_io.load_best_params(future_index, model_name)
        train_model = Seq2SeqModelTrainer(encoder_class=EncoderGRU, decoder_class=AttnDecoderGRU,
                                          future_index=future_index, params=params)
        train_model.run()


if __name__ == '__main__':
    seed_everything(xfinai_config.seed, workers=True)
    # train_recurrent_model()
    train_seq2seq_model()
