import torch
from torch.utils.data import Dataset
import pandas as pd
import math
import sys
sys.path.append("../..")

import config


# Load Origin Data
df_ic = pd.read_pickle('../data/IC_1m.pkl')


# Split Data Set
data_size = df_ic.shape[0]
train_data = df_ic.iloc[:math.floor(config.train_size*data_size)]
val_data = df_ic.iloc[math.ceil(config.train_size*data_size):math.floor((config.train_size+config.val_size)*data_size)]
test_data = df_ic.iloc[math.floor((config.train_size+config.val_size)*data_size):]


# Create torch dataset
# Define BaseClass
class FuturesDataset(Dataset):

    def __init__(self, data, label, seq_length, features_list):
        self.data = data
        self.label = label
        self.features_list = features_list
        self.seq_length = seq_length

        self.data_set = self.create_xy_pairs()

    def create_xy_pairs(self):
        pairs = []
        for idx in range(self.data_length - self.seq_length):
            x = self.data[idx:idx + self.seq_length][self.features_list].values
            y = self.data[idx + self.seq_length:idx + self.seq_length + 1][self.label].values
            pairs.append((x, y))
        return pairs

    def __len__(self):
        return self.data_set.shape[0]

    def __getitem__(self, idx):
        return self.data_set[idx]


params = {'batch_size': BATCH_SIZE,
          'shuffle': False,
          'drop_last': True, # Disregard last incomplete batch
          'num_workers': 2}

params_test = {'batch_size': 1,
          'shuffle': False,
          'drop_last': False, # Disregard last incomplete batch
          'num_workers': 2}

#  datasets and dataloader for 14 asset
train_data_loader = FuturesDataset(data=train_data,
)
training_ds_list = [CryptoDataset(training_data, label, FEATURES, TARGET) for training_data in training_data_list]
training_dl_list = [DataLoader(training_ds, **params) for training_ds in training_ds_list]

validation_ds_list = [CryptoDataset(validation_data, SEQ_LENGTH, FEATURES, TARGET) for validation_data in validation_data_list]
validation_dl_list = [DataLoader(validation_ds, **params) for validation_ds in validation_ds_list]

