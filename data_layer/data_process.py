import pandas as pd
import sys
from functools import partial
import glog

sys.path.append("../")

import xfinai_config


# Calc Return
def calc_return(df, time_lag=1, price_type='close'):
    return df[price_type].pct_change(-time_lag).shift(-time_lag).fillna(method='ffill')


# labeled return, (up or down)
def label_return(df, time_lag=1, price_type='close'):
    return_series = df[price_type].pct_change(-time_lag).shift(-time_lag).fillna(method='ffill')
    return_labeled = (return_series > 0).astype('int')
    return return_labeled


# Calc Return
def calc_return_direction(df, time_lag=1, price_type='close'):
    ret = df[price_type].pct_change(-time_lag).shift(-time_lag).fillna(method='ffill')
    return (ret > 0).astype('int')


def simple_price(df, time_lag=1, price_type='close'):
    return df[price_type].shift(-time_lag).fillna(method='ffill').values


# Create the label
def generate_label(df, label_func, label):
    df_labeled = df.copy(deep=True)
    df_labeled[label] = label_func(df_labeled)
    return df_labeled


def main():
    for future_index in xfinai_config.futures_index_map:
        # Load Origin Data
        glog.info(f"Load Origin Data future_index: {future_index}")
        df_raw = pd.read_pickle(f'{xfinai_config.raw_data_path}/{future_index}_{xfinai_config.time_freq}.pkl')

        # Add time restrict
        df_restrict_time = df_raw.loc[xfinai_config.data_start_time:]

        # Generate Label
        glog.info(f"Generate Label future_index: {future_index}")
        generate_label_func = partial(calc_return, price_type='close', time_lag=xfinai_config.label_time_lag)
        label_name = xfinai_config.label
        df_labeled = generate_label(df=df_restrict_time, label_func=generate_label_func, label=label_name)

        # Save Data
        glog.info(f"Save future_index: {future_index}")
        df_labeled.to_pickle(f'{xfinai_config.processed_data_path}/{future_index}_processed.pkl')


if __name__ == '__main__':
    main()
