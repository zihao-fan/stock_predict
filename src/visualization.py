# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import data_helper
from data_helper import index_list, days_skipped

from matplotlib import pyplot

bins = [-1., 0.0, 1.]
labels = ['fall', 'rise']

def get_df(stock_code):
    raw_df = data_helper.get_dataframe(stock_code)
    raw_df = raw_df[days_skipped:]
    return raw_df

def histogram_by_label(df, feature_name):
    x = df['涨跌幅'].values
    y = df[feature_name].values
    hist, bin_edges = np.histogram(y, bins='auto')

    rise = df.loc[df['涨跌幅'] > 0.02][feature_name].values
    fall = df.loc[df['涨跌幅'] < -0.02][feature_name].values

    pyplot.hist(rise, bins=bin_edges, normed=True, alpha=0.5, label='rise')
    pyplot.hist(fall, bins=bin_edges, normed=True, alpha=0.5, label='fall')
    pyplot.legend(loc='upper right')
    # pyplot.title(feature_name)
    pyplot.show()

if __name__ == '__main__':
    df = get_df('sh600004')
    histogram_by_label(df, 'MACD_MACD')