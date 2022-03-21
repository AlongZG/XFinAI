import pandas as pd
import sys
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport

sys.path.append("../")
import xfinai_config


def load_data_raw(future_index):
    df_raw = pd.read_pickle(f'{xfinai_config.raw_data_path}/{future_index}_{xfinai_config.time_freq}.pkl')
    # Add time restrict
    df_restrict_time = df_raw.loc[xfinai_config.data_start_time:]
    return df_restrict_time


def load_featured_data(future_index):
    df_featured_train = pd.read_pickle(f'{xfinai_config.featured_data_path}/{future_index}_train_data.pkl')
    df_featured_val = pd.read_pickle(f'{xfinai_config.featured_data_path}/{future_index}_val_data.pkl')
    df_featured_test = pd.read_pickle(f'{xfinai_config.featured_data_path}/{future_index}_test_data.pkl')
    df_concated = pd.concat([df_featured_train, df_featured_val, df_featured_test])
    return df_concated


def generate_profile_raw(future_index):
    raw_data = load_data_raw(future_index)
    report_raw_data = ProfileReport(raw_data)
    report_raw_data.to_file(f"{xfinai_config.raw_data_profile_path}/{future_index}.html")


def generate_profile_featured(future_index):
    featured_data = load_featured_data(future_index)
    report_featured_data = ProfileReport(featured_data)
    report_featured_data.to_file(f"{xfinai_config.featured_data_profile_path}/{future_index}.html")


def main():
    for future_index in xfinai_config.futures_index_map:
        generate_profile_raw(future_index)
        generate_profile_featured(future_index)


if __name__ == '__main__':
    main()
