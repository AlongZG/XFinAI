from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import math
import glog
import joblib
import sys

sys.path.append("../")

import xfinai_config
from utils import path_wrapper


def load_processed_data(index):
    df = pd.read_pickle(f'{xfinai_config.processed_data_path}/{index}_processed.pkl')
    return df


def data_split(df):
    data_size = df.shape[0]
    train_data = df.iloc[:math.floor(xfinai_config.train_size * data_size)]
    val_data = df.iloc[math.ceil(xfinai_config.train_size * data_size):math.floor(
        (xfinai_config.train_size + xfinai_config.val_size) * data_size)]
    test_data = df.iloc[math.floor((xfinai_config.train_size + xfinai_config.val_size) * data_size):]
    return train_data, val_data, test_data


def fe_pipeline(train_data, val_data, test_data):
    scaler = StandardScaler()
    scaler.fit(train_data[xfinai_config.features_list])

    train_data.loc[:, xfinai_config.features_list] = scaler.transform(train_data[xfinai_config.features_list])
    val_data.loc[:, xfinai_config.features_list] = scaler.transform(val_data[xfinai_config.features_list])
    test_data.loc[:, xfinai_config.features_list] = scaler.transform(test_data[xfinai_config.features_list])
    return train_data, val_data, test_data


def save_data(train_data, val_data, test_data, index):
    feature_data_dir = path_wrapper.wrap_path(xfinai_config.featured_data_path)
    train_data.to_pickle(f"{feature_data_dir}/{index}_train_data.pkl")
    val_data.to_pickle(f"{feature_data_dir}/{index}_val_data.pkl")
    test_data.to_pickle(f"{feature_data_dir}/{index}_test_data.pkl")


def main():
    for future_index in xfinai_config.futures_index_map:
        # Load Origin Data
        glog.info(f"Loading Origin Data future_index: {future_index}")
        df_processed = load_processed_data(future_index)

        # Split Data
        glog.info("Split Data future_index: {future_index}")
        train_data, val_data, test_data = data_split(df_processed)

        # Feature Engineering, Train Scaler
        glog.info("Feature Engineering future_index: {future_index}")
        train_data, val_data, test_data = fe_pipeline(train_data, val_data, test_data)

        glog.info("Saving Data: {future_index}")
        save_data(train_data, val_data, test_data, index=future_index)


if __name__ == '__main__':
    main()
