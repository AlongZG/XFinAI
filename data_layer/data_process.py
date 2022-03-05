import pandas as pd
import sys

sys.path.append("../")

import xfinai_config


# Calc Return
def calc_return(df):
    return df['close'].pct_change().fillna(method='bfill')


# Create the label
def generate_label(df, label_func, label):
    df_labeled = df.copy(deep=True)
    df_labeled[label] = label_func(df_labeled)
    return df_labeled


def main():
    # Load Origin Data
    df_ic = pd.read_pickle(f'{xfinai_config.origin_data_path}/IC_1m.pkl')
    df_if = pd.read_pickle(f'{xfinai_config.origin_data_path}/IF_1m.pkl')
    df_ih = pd.read_pickle(f'{xfinai_config.origin_data_path}/IH_1m.pkl')

    # Add time restrict
    df_ic = df_ic.loc[xfinai_config.data_start_time:]
    df_if = df_if.loc[xfinai_config.data_start_time:]
    df_ih = df_ih.loc[xfinai_config.data_start_time:]

    df_ic = generate_label(df_ic, calc_return, 'return')
    df_if = generate_label(df_if, calc_return, 'return')
    df_ih = generate_label(df_ih, calc_return, 'return')

    df_ic.to_pickle(f'{xfinai_config.processed_data_path}/ic_processed.pkl')
    df_if.to_pickle(f'{xfinai_config.processed_data_path}/if_processed.pkl')
    df_ih.to_pickle(f'{xfinai_config.processed_data_path}/ih_processed.pkl')


if __name__ == '__main__':
    main()
