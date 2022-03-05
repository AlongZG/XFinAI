import pandas as pd

# data config
data_start_time = pd.to_datetime('2021-01-01')

# dataset config
train_size = 0.7
val_size = 0.2
test_size = 1 - train_size - val_size

# feature config
label = 'return'
features_list = ['open', 'high', 'low', 'close', 'volume', 'money']

# model config
seq_length = 32
batch_size = 128
lstm_model_config = {
    'batch_size': batch_size,
    'input_size': len(features_list),
    'hidden_size': 10,
    'num_layers': 2,
    'output_size': 1,
    'dropout_prob': 0.1,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'epochs': 100
}

# feature config
data_loader_config = {'batch_size': batch_size,
                      'shuffle': False,
                      'drop_last': True,
                      'num_workers': 4,
                      'pin_memory': True
                      }

# PATH
losses_path = '../result/losses'
inference_result_path = '../result/inference_result'
model_save_path = './trained_models'
