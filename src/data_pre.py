import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
import os

try:
    train_data = pd.read_csv('./data/raw/train.csv')
    test_data = pd.read_csv('./data/raw/test.csv')


    data_path = os.path.join('data','processed')
    os.makedirs(data_path)

    train_data.to_csv(os.path.join(data_path, 'train_processed.csv'),index=False)
    test_data.to_csv(os.path.join(data_path, 'test_processed.csv'),index=False)
except Exception as e:
    print(e)