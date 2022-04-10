import sys

import nni
from sklearn.metrics import r2_score

sys.path.append('../..')
from model_layer.model_hub import EncoderGRU, AttnDecoderGRU
from model_layer.model_tuner import Seq2SeqModelTuner
from utils import base_io


def main(future_index, encoder_class, decoder_class, params):
    target_metric_func = r2_score
    metric_name = 'R_Square'
    tune_model = Seq2SeqModelTuner(encoder_class=encoder_class, decoder_class=decoder_class,
                                   future_index=future_index, target_metric_func=target_metric_func,
                                   metric_name=metric_name, params=params)
    tune_model.run()


if __name__ == '__main__':
    future_index = 'IC'
    encoder_class = EncoderGRU
    decoder_class = AttnDecoderGRU
    model_name = f"{encoder_class.name}_{decoder_class.name}"
    # params = base_io.load_best_params(future_index, model_name)
    # params = nni.get_next_parameter()
    params = {
     "epochs": 2,
     "batch_size": 64,
     "hidden_size": 8,
     "seq_length": 32,
     "weight_decay": 0.0028780633371441426,
     "learning_rate": 0.003468997588562518,
     "dropout_prob": 0.17088626278010194
}
    main(future_index, encoder_class, decoder_class, params)
