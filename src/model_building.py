import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import classification_report
import yaml

try:
    
    train = pd.read_csv('./data/processed/train_processed.csv')
    test = pd.read_csv('./data/processed/test_processed.csv')

    #x_train = train.iloc[:,:-1]
    #y_train = train.iloc[:,-1]
    x_train = train.drop('Machine failure',axis=1)
    y_train = train['Machine failure']

    n_estimators = yaml.safe_load(open('params.yaml'))['model_building']['n_estimators']


    rf = RandomForestClassifier(n_estimators=n_estimators)
    rf.fit(x_train, y_train)


    pickle.dump(rf, open('model.pkl','wb'))
except Exception as e:
    print(e)

