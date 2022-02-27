from torch.utils.data import DataLoader, Dataset
import pandas as pd
import math
import glog
import joblib
import sys
sys.path.append("../")

import xfinai_config
# from base_dataset import FuturesDataset

# Define Dataset BaseClass
class FuturesDataset(Dataset):

    def __init__(self, data, label, seq_length, features_list):
        self.data = data
        self.label = label
        self.features_list = features_list
        self.seq_length = seq_length
        self.data_set = self.create_xy_pairs()
        self.data_length = len(self.data_set)

    def create_xy_pairs(self):
        pairs = []
        for idx in range(self.data.shape[0] - self.seq_length):
            x = self.data[idx:idx + self.seq_length][self.features_list].values
            y = self.data[idx + self.seq_length:idx + self.seq_length + 1][self.label].values
            pairs.append((x, y))
        return pairs

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        return self.data_set[idx]

def data_split(df):
    data_size = df.shape[0]
    train_data = df.iloc[:math.floor(xfinai_config.train_size*data_size)]
    val_data = df.iloc[math.ceil(xfinai_config.train_size*data_size):math.floor((xfinai_config.train_size+xfinai_config.val_size)*data_size)]
    test_data = df.iloc[math.floor((xfinai_config.train_size+xfinai_config.val_size)*data_size):]
    return train_data, val_data, test_data

def main():
    # Load Origin Data
    glog.info("Loading Origin Data")
    df_ic = pd.read_pickle('../data/data_processed/ic_processed.pkl')
    df_ih = pd.read_pickle('../data/data_processed/if_processed.pkl')
    df_if = pd.read_pickle('../data/data_processed/ih_processed.pkl')


    # Split Data 
    glog.info("Spliting Data")


    train_data_ic, val_data_ic, test_data_ic = data_split(df_ic)
    train_data_if, val_data_if, test_data_if = data_split(df_if)
    train_data_ih, val_data_ih, test_data_ih = data_split(df_ih)

    train_data_list = [train_data_ic, train_data_if, train_data_ih]
    val_data_list = [val_data_ic, val_data_if, val_data_ih]
    test_data_list = [test_data_ic, test_data_if, test_data_ih]

    #  datasets and dataloader for 3 futures
    glog.info("Creating dataset & dataloader")

    train_dataset_list = [FuturesDataset(data=train_data, label=xfinai_config.label,seq_length=xfinai_config.seq_length, 
                                        features_list=xfinai_config.features_list) 
                                        for train_data in train_data_list]
    train_dataloader_list = [DataLoader(train_dataset, **xfinai_config.data_loader_config) for train_dataset in train_dataset_list]

    val_dataset_list = [FuturesDataset(data=val_data, label=xfinai_config.label,seq_length=xfinai_config.seq_length, 
                                        features_list=xfinai_config.features_list) 
                                        for val_data in val_data_list]
    val_dataloader_list = [DataLoader(val_dataset, **xfinai_config.data_loader_config) for val_dataset in val_dataset_list]

    test_dataset_list = [FuturesDataset(data=test_data, label=xfinai_config.label,seq_length=xfinai_config.seq_length, 
                                        features_list=xfinai_config.features_list) 
                                        for test_data in test_data_list]
    test_dataloader_list = [DataLoader(test_dataset, **xfinai_config.data_loader_config) for test_dataset in test_dataset_list]

    # Save DataLoader
    glog.info("Saving dataloader")
    joblib.dump(train_dataloader_list, './data_loaders/train_dataloader_list.pkl')
    joblib.dump(val_dataloader_list, './data_loaders/val_dataloader_list.pkl')
    joblib.dump(test_dataloader_list, './data_loaders/test_dataloader_list.pkl')

if __name__ == '__main__':
    main()