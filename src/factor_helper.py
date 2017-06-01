# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np 
import os
import json
from scipy import stats

current_path = os.path.realpath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])

def csv_to_pd(filename):
    full_path = os.path.join(root_path, 'raw_data', filename)
    df = pd.read_csv(full_path)
    df.sort(columns=['secID'], inplace=True)
    df.set_index(keys=['secID'], drop=False, inplace=True)
    ids = df['secID'].unique().tolist()

    df.to_pickle(os.path.join(root_path, 'data', 'factor_pd.pickle'))
    with open (os.path.join(root_path, 'data', 'factor_ids.json'), 'w') as f:
        json.dump(ids, f)

    print 'Data saved.'

def read_price(filename):
    full_path = os.path.join(root_path, 'raw_data', filename)
    df = pd.read_csv(full_path, index_col=0).transpose()
    df.index = df.index.map(int)
    return df


def select_by_id(df, price, secid=''):
    if len(secid) > 0:
        part_df = df.loc[df.secID == secid].copy()
    else:
        part_df = df.copy()
    part_df.sort(columns=['date'], inplace=True)
    part_df = part_df.set_index(keys=['date'])
    part_df = part_df.drop(['secID'], axis=1)
    part_df = part_df.fillna(axis=1, method='ffill')
    part_df['close_price'] = pd.Series(0, index=part_df.index)
    
    current_price = price[secid]
    print current_price.index
    print part_df.index

    for index, row in part_df.iterrows():
        part_df['close_price'].loc[index] = current_price.loc[index]
    print part_df
    return part_df

if __name__ == '__main__':
    # csv_to_pd('select_factor.csv')
    price = read_price('Daliy_ClosePrice.csv')

    ids = []
    with open(os.path.join(root_path, 'data', 'factor_ids.json'), 'r') as f:
        ids = json.load(f)
    df = pd.read_pickle(os.path.join(root_path, 'data', 'factor_pd.pickle'))
    part_df = select_by_id(df, price, ids[0])