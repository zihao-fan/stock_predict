# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot
from highfreq_helper import get_equal_bin_edges

bins = [-1., 0.0, 1.]
labels = ['fall', 'rise']
current_path = os.path.realpath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])

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

def plot_feature_hisogram(df, feature_name):
    x = df[feature_name].values
    bin_edges = get_equal_bin_edges(x, 100)
    print 'bin_edges', bin_edges, 'len', len(bin_edges)
    pyplot.hist(x, bins=bin_edges, normed=True)
    pyplot.show()

if __name__ == '__main__':
    datafile_path = os.path.join(root_path, 'data', '000905_20100101_20170515.data')
    data = pd.read_pickle(datafile_path)
    plot_feature_hisogram(data, 'rate')