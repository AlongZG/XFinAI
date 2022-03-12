import pandas as pd

# data config
data_start_time = pd.to_datetime('2021-10-01')
futures_index_map = {
    'ic': 0,
    'ih': 1,
    'if': 2
}

# dataset config
train_size = 0.7
val_size = 0.2
test_size = 1 - train_size - val_size

# feature config
label = 'return'
features_list = ['open', 'high', 'low', 'close', 'volume', 'money', 'open_interest']

# model config
seq_length = 32
batch_size = 64
lstm_model_config = {
    'batch_size': batch_size,
    'input_size': len(features_list),
    'hidden_size': 10,
    'num_layers': 2,
    'output_size': 1,
    'dropout_prob': 0.1,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'epochs': 20
}

# feature config
data_loader_config = {'shuffle': False,
                      'drop_last': True,
                      'num_workers': 1,
                      'pin_memory': True
                      }

# PATH
origin_data_path = 'D:/projects/XFinAI/data/origin_data'
processed_data_path = 'D:/projects/XFinAI/data/data_processed'
featured_data_path = 'D:/projects/XFinAI/data/data_featured'
losses_path = 'D:/projects/XFinAI/result/losses'
inference_result_path = 'D:/projects/XFinAI/result/inference_result'
model_save_path = 'D:/projects/XFinAI/model_layer/trained_models'
