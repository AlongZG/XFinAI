import sys

sys.path.append('../')
from model_layer.model_hub import RNN, LSTM, GRU, EncoderGRU, AttnDecoderGRU
from model_layer.model_trainer import RecurrentModelTrainer, Seq2SeqModelTrainer


def train_recurrent_model():
    future_list = ['IC', 'IH', 'IF']
    model_list = [RNN, LSTM, GRU]
    for future_index in future_list:
        for model_class in model_list:
            train_model = RecurrentModelTrainer(model_class=model_class, future_index=future_index)
            train_model.run()


def train_seq2seq_model():
    future_list = ['IC']
    for future_index in future_list:
        train_model = Seq2SeqModelTrainer(encoder_class=EncoderGRU, decoder_class=AttnDecoderGRU,
                                          future_index=future_index)
        train_model.run()


if __name__ == '__main__':
    # train_recurrent_model()
    train_seq2seq_model()
