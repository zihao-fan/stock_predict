from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

INDEX_DIR = '/data01/zihao/stock_dataset/index'
SH_DIR = '/data01/zihao/stock_dataset/sh'
SZ_DIR = '/data01/zihao/stock_dataset/sz'

def get_filenames_from_dir(my_dir):
    csv_files = [f for f in listdir(my_dir) if isfile(join(my_dir, f))]
    return csv_files

if __name__ == '__main__':
    file_list = get_filenames_from_dir(INDEX_DIR)
    print(file_list)