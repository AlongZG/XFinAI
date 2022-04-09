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
    params = nni.get_next_parameter()
    main(future_index, encoder_class, decoder_class, params)
