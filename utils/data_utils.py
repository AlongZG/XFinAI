import glog
import math
import sys

sys.path.append("../")

import xfinai_config


def split_data(df):
    data_size = df.shape[0]
    train_data = df.iloc[:math.floor(xfinai_config.train_size * data_size)]
    val_data = df.iloc[math.ceil(xfinai_config.train_size * data_size):math.floor(
        (xfinai_config.train_size + xfinai_config.val_size) * data_size)]
    test_data = df.iloc[math.floor((xfinai_config.train_size + xfinai_config.val_size) * data_size):]
    return train_data, val_data, test_data


def clean_data(df):
    # delete too many null columns
    null_percent_features = df.isnull().sum() / df.shape[0]
    safe_null_features = null_percent_features[null_percent_features <= xfinai_config.null_percent].index
    if len(safe_null_features) < len(df.columns):
        glog.info(f"Too many null values \n"
                  f"{null_percent_features[null_percent_features > xfinai_config.null_percent]}")
    df = df[safe_null_features]

    # fill remaining nan
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df
