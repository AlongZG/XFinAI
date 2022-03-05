import pandas as pd

# data config
data_start_time = pd.to_datetime('2021-10-01')
futures_index_map = {
    'IC': 0,
    'IH': 1,
    'IF': 2
}

# dataset config
train_size = 0.7
val_size = 0.2
test_size = 1 - train_size - val_size

# feature config
label = 'return'
features_list = ['open', 'high', 'low', 'close', 'volume', 'money']

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
    'epochs': 1
}

# feature config
data_loader_config = {'batch_size': batch_size,
                      'shuffle': False,
                      'drop_last': True,
                      'num_workers': 2,
                      'pin_memory': True
                      }

# PATH
origin_data_path = 'D:/毕业论文/XFinAI/data/origin_data'
processed_data_path = 'D:/毕业论文/XFinAI/data/data_processed'
data_loader_path = 'D:/毕业论文/XFinAI/data_layer/data_loaders'
losses_path = 'D:/毕业论文/XFinAI/result/losses'
inference_result_path = 'D:/毕业论文/XFinAI/result/inference_result'
model_save_path = 'D:/毕业论文/XFinAI/model_layer/trained_models'
