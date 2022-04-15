import joblib
import pandas as pd
import json
import sys
import torch
import glog
from torch.utils.data import DataLoader

sys.path.append("../")

import xfinai_config
from utils import path_wrapper
from data_layer.base_dataset import FuturesDatasetRecurrent


def load_data(future_index):
    train_data = pd.read_pickle(f"{xfinai_config.featured_data_path}/{future_index}_train_data.pkl")
    val_data = pd.read_pickle(f"{xfinai_config.featured_data_path}/{future_index}_val_data.pkl")
    test_data = pd.read_pickle(f"{xfinai_config.featured_data_path}/{future_index}_test_data.pkl")
    return train_data, val_data, test_data


def get_dataset(future_index, params):

    train_data, val_data, test_data = load_data(future_index)
    train_dataset = FuturesDatasetRecurrent(data=train_data, label=xfinai_config.label, seq_length=params['seq_length'])
    val_dataset = FuturesDatasetRecurrent(data=val_data, label=xfinai_config.label, seq_length=params['seq_length'])
    test_dataset = FuturesDatasetRecurrent(data=test_data, label=xfinai_config.label, seq_length=params['seq_length'])

    return train_dataset, val_dataset, test_dataset


def get_data_loader(future_index, params):
    # Load Data
    train_dataset, val_dataset, test_dataset = get_dataset(future_index, params)

    train_loader = DataLoader(dataset=train_dataset, **xfinai_config.data_loader_config,
                              batch_size=params['batch_size'])
    val_loader = DataLoader(dataset=val_dataset, **xfinai_config.data_loader_config,
                            batch_size=params['batch_size'])
    test_loader = DataLoader(dataset=test_dataset, **xfinai_config.data_loader_config,
                             batch_size=params['batch_size'])

    return train_loader, val_loader, test_loader


def load_best_params(future_index, model_name):
    params_dir = path_wrapper.wrap_path(f"{xfinai_config.best_params_path}/{future_index}")
    params_path = f"{params_dir}/{model_name}.json"
    with open(params_path, 'r') as f:
        params = json.load(f)
    return params


def save_model(model, future_index, seq2seq=False):
    dir_path = path_wrapper.wrap_path(f"{xfinai_config.model_save_path}/{future_index}")
    if not seq2seq:
        save_path = f"{dir_path}/{model.name}.pth"
        glog.info(f"Starting save model state, save_path: {save_path}")
        torch.save(model.state_dict(), save_path)
    else:
        encoder, decoder = model
        dir_path = path_wrapper.wrap_path(f"{dir_path}/{encoder.name}_{decoder.name}")

        encoder_save_path = f"{dir_path}/{encoder.name}.pth"
        glog.info(f"Starting save encoder state, save_path: {encoder_save_path}")
        torch.save(encoder.state_dict(), encoder_save_path)

        decoder_save_path = f"{dir_path}/{decoder.name}.pth"
        glog.info(f"Starting save decoder state, save_path: {decoder_save_path}")
        torch.save(decoder.state_dict(), decoder_save_path)
        

def load_model(model, future_index, seq2seq=False):
    dir_path = path_wrapper.wrap_path(f"{xfinai_config.model_save_path}/{future_index}")
    if not seq2seq:
        model_path = f"{dir_path}/{model.name}.pth"
        glog.info(f"Loading model state, model_path: {model_path}")
        model.load_state_dict(torch.load(model_path))
        return model
    else:
        encoder, decoder = model
        dir_path = path_wrapper.wrap_path(f"{dir_path}/{encoder.name}_{decoder.name}")

        encoder_path = f"{dir_path}/{encoder.name}.pth"
        glog.info(f"Loading encoder state, model_path: {encoder_path}")
        encoder.load_state_dict(torch.load(encoder_path))

        decoder_path = f"{dir_path}/{decoder.name}.pth"
        glog.info(f"Loading decoder state, model_path: {decoder_path}")
        decoder.load_state_dict(torch.load(decoder_path))
        return encoder, decoder


def save_attention_weights(attention_weights, future_index, model_name):
    attention_weights_dir = path_wrapper.wrap_path(f"{xfinai_config.attention_weights_path}/"
                                                   f"{future_index}/{model_name}")
    attention_weights_path = f"{attention_weights_dir}/attention_weights.pkl"
    glog.info(f"Save attention weights  to {attention_weights_path}")
    joblib.dump(attention_weights, attention_weights_path)


def save_metrics_result(metrics_result_list, future_index, model_name):
    df_metrics_result = pd.DataFrame(metrics_result_list)
    dir_path = path_wrapper.wrap_path(f"{xfinai_config.inference_result_path}/"
                                      f"{future_index}/{model_name}")
    metrics_result_path = f"{dir_path}/metrics.csv"
    glog.info(f"Save metrics result to {metrics_result_path}")
    df_metrics_result.to_csv(metrics_result_path)
    return df_metrics_result
