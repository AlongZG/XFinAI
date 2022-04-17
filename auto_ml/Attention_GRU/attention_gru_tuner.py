import sys
from pytorch_lightning import seed_everything

import nni
from sklearn.metrics import r2_score


sys.path.append('../..')
from model_layer.model_hub import EncoderGRU, AttnDecoderGRU
from model_layer.model_tuner import Seq2SeqModelTuner
import xfinai_config


def main(future_index, encoder_class, decoder_class, params):
    target_metric_func = r2_score
    metric_name = 'R_Square'
    tune_model = Seq2SeqModelTuner(encoder_class=encoder_class, decoder_class=decoder_class,
                                   future_index=future_index, target_metric_func=target_metric_func,
                                   metric_name=metric_name, params=params)
    tune_model.run()


if __name__ == '__main__':
    seed_everything(xfinai_config.seed, workers=True)
    future_index = 'IH'
    encoder_class = EncoderGRU
    decoder_class = AttnDecoderGRU
    model_name = f"{encoder_class.name}_{decoder_class.name}"
    params = nni.get_next_parameter()
    # params = {
    #     "epochs": 10,
    #     "batch_size": 32,
    #     "hidden_size": 8,
    #     "seq_length": 64,
    #     "weight_decay": 0.014896126293298352,
    #     "learning_rate": 0.007421382952353398,
    #     "dropout_prob": 0.20520860052969295
    # }
    main(future_index, encoder_class, decoder_class, params)
