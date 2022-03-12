from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import glog
import sys
from ta_factors import TaFactor
from feature_selector import corr_selector

sys.path.append("../")
import xfinai_config
from utils import path_wrapper, data_utils


def load_processed_data(index):
    df = pd.read_pickle(f'{xfinai_config.processed_data_path}/{index}_processed.pkl')
    return df


def fe_ta(df):
    tf = TaFactor(df)
    return tf.run()


def fe_scale(train_data, val_data, test_data):
    features_list = list(train_data.columns)
    features_list.remove(xfinai_config.label)

    scaler = StandardScaler()
    scaler.fit(train_data[features_list])

    train_data.loc[:, features_list] = scaler.transform(train_data[features_list])
    val_data.loc[:, features_list] = scaler.transform(val_data[features_list])
    test_data.loc[:, features_list] = scaler.transform(test_data[features_list])
    return train_data, val_data, test_data


def feature_select(train_data, val_data, test_data):
    features_list = list(train_data.columns)
    features_list.remove(xfinai_config.label)

    feature_selected_list = list(corr_selector(df_feature=train_data[features_list],
                                         threshold=xfinai_config.corr_threshold).columns)
    feature_selected_list.append(xfinai_config.label)

    train_data = train_data[feature_selected_list]
    val_data = val_data[feature_selected_list]
    test_data = test_data[feature_selected_list]

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

        # generate ta_factors
        df_ta_factor = fe_ta(df_processed)
        df_ta_factor = data_utils.clean_data(df_ta_factor)

        # Split Data
        glog.info(f"Split Data future_index: {future_index}")
        train_data, val_data, test_data = data_utils.split_data(df_ta_factor)

        # Feature Engineering, Train Scaler
        glog.info(f"Feature Scaling future_index: {future_index}")
        train_data, val_data, test_data = fe_scale(train_data, val_data, test_data)

        # select features
        glog.info(f"Feature Selecting future_index: {future_index}")
        train_data, val_data, test_data = feature_select(train_data, val_data, test_data)

        glog.info(f"Saving Data: {future_index}")
        save_data(train_data, val_data, test_data, index=future_index)


if __name__ == '__main__':
    main()
