import pandas as pd

# seed
seed = 416

# data config
data_start_time = pd.to_datetime('2022-01-01')
futures_index_map = {
    'IC': 0,
    'IH': 1,
    'IF': 2
}
time_freq = '1m'

# dataset config
train_size = 0.7
val_size = 0.2
test_size = 1 - train_size - val_size

# feature config
label = 'return'
label_time_lag = 1
corr_threshold = 0.8
null_percent = 0.1

# model config
model_config = {
    "RNN": {
        "output_size": 1,
    },
    "LSTM": {
        "output_size": 1,
    },
    "GRU": {
        'output_size': 1
    },
    "EncoderGRU_AttnDecoderGRU": {
        'output_size': 1
    },
}

# feature config
data_loader_config = {'shuffle': False,
                      'drop_last': True,
                      'num_workers': 2,
                      'pin_memory': True
                      }

# PATH
raw_data_path = 'D:/projects/XFinAI/data/raw_data'
processed_data_path = 'D:/projects/XFinAI/data/data_processed'
featured_data_path = 'D:/projects/XFinAI/data/data_featured'
raw_data_profile_path = 'D:/projects/XFinAI/EDA/data_profiles/raw_data'
featured_data_profile_path = 'D:/projects/XFinAI/EDA/data_profiles/featured_data'
data_hub_path = 'D:/projects/XFinAI/EDA/data_hub'
scaler_path = 'D:/projects/XFinAI/data/scaler'
losses_path = 'D:/projects/XFinAI/result/losses'
inference_result_path = 'D:/projects/XFinAI/result/inference_result'
raw_prediction_path = 'D:/projects/XFinAI/result/raw_prediction'
model_save_path = 'D:/projects/XFinAI/model_layer/trained_models'
best_params_path = 'D:/projects/XFinAI/model_layer/best_params'
tensorboard_log_default_path = 'D:/nni_experiments/tensorboard_default'
attention_weights_path = 'D:/projects/XFinAI/explainable_algorithm/attention_weights'
lime_result_path = 'D:/projects/XFinAI/explainable_algorithm/lime_explain_result'
shap_values_path = 'D:/projects/XFinAI/explainable_algorithm/shap_values'
