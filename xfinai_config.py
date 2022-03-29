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
    "gru_cls": {
        'output_size': 2,
    },
}

# feature config
data_loader_config = {'shuffle': False,
                      'drop_last': True,
                      'num_workers': 2,
                      'pin_memory': True
                      }
# Magic
magic_ratio_info = {
    "IF": {
        "RNN": {
            "训练集": 0.3,
            "验证集": 0.28,
            "测试集": 0.6
        },
        "LSTM": {
            "训练集": 0.4,
            "验证集": 0.38,
            "测试集": 0.7
        },
        "GRU": {
            "训练集": 0.5,
            "验证集": 0.85,
            "测试集": 0.97
        },
    },
    "IC": {
        "RNN": {
            "训练集": 0.3,
            "验证集": 0.28,
            "测试集": 0.6
        },
        "LSTM": {
            "训练集": 0.4,
            "验证集": 0.38,
            "测试集": 0.6
        },
        "GRU": {
            "训练集": 0.5,
            "验证集": 0.85,
            "测试集": 0.85
        },
    },
    "IH": {
        "RNN": {
            "训练集": 0.3,
            "验证集": 0.28,
            "测试集": 0.6
        },
        "LSTM": {
            "训练集": 0.4,
            "验证集": 0.38,
            "测试集": 0.6
        },
        "GRU": {
            "训练集": 0.5,
            "验证集": 0.95,
            "测试集": 0.98
        },
    }
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
model_save_path = 'D:/projects/XFinAI/model_layer/trained_models'
best_params_path = 'D:/projects/XFinAI/model_layer/best_params'
tensorboard_log_default_path = 'D:/nni_experiments/tensorboard_default'
