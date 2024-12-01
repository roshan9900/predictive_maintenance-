import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import yaml

try:

    ai4i_2020_predictive_maintenance_dataset = fetch_ucirepo(id=601) 
    X = ai4i_2020_predictive_maintenance_dataset.data.features 
    y = ai4i_2020_predictive_maintenance_dataset.data.targets 


    one = OneHotEncoder()
    dfone = pd.DataFrame(one.fit_transform(pd.DataFrame(X['Type'])).toarray(),columns=['H','L','M'])
    df = pd.concat([X,dfone],axis=1)
    df.drop('Type',axis=1,inplace=True)

    x,y = SMOTE().fit_resample(df,y['Machine failure'])
    x['Machine failure'] = y
    
    #y = y['Machine failure']
    
    #x['Machine failure'] = y

    test_size = yaml.safe_load(open('params.yaml'))['data_collection']['test_size']


    train_data, test_data = train_test_split(x,test_size=test_size, random_state=42)


    data_path = os.path.join('data','raw')

    os.makedirs(data_path)

    train_data.to_csv(os.path.join(data_path,'train.csv'),index=False)
    test_data.to_csv(os.path.join(data_path, 'test.csv'),index=False)
except Exception as e:
    print(e)



