import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import json
import yaml
from dvclive import  Live
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
import dagshub
dagshub.init(repo_owner='roshansalunke91', repo_name='predictive_maintenance-', mlflow=True)

mlflow.set_experiment('gb')
mlflow.set_tracking_uri('https://dagshub.com/roshansalunke91/predictive_maintenance-.mlflow')

try:
    with mlflow.start_run():
        test_size = yaml.safe_load(open('params.yaml'))['data_collection']['test_size']
        n_estimators = yaml.safe_load(open('params.yaml'))['model_building']['n_estimators']
        test_data = pd.read_csv('./data/processed/test_processed.csv')
        x_test = test_data.iloc[:,:-1]
        y_test = test_data.iloc[:,-1]

    
        rf = pickle.load(open('model.pkl','rb'))
        pred = rf.predict(x_test)

        acc = accuracy_score(y_test, pred)
        f1_scor = f1_score(y_test, pred)
        precision_scor = precision_score(y_test, pred)
        recall_scor = recall_score(y_test, pred)

        mlflow.log_metric('accuracy',acc)
        mlflow.log_metric('f1_score',f1_scor)
        mlflow.log_metric('precission',precision_scor)
        mlflow.log_metric('recall',recall_scor)
        
        mlflow.log_param('test_size',test_size)
        mlflow.log_param('n_estimators',n_estimators)
        
        cm = confusion_matrix(y_test, pred)
        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True)
        plt.xlabel('pred')
        plt.ylabel('actual')
        plt.title('cm')
        plt.savefig('con_met.png')
        
        mlflow.log_artifact('con_met.png')
        
        mlflow.sklearn.log_model(rf,'gradient')
        
        mlflow.log_artifact(__file__)
        mlflow.set_tag('author','RS')
        mlflow.set_tag('model','gb')

        
        
        
        
    with Live(save_dvc_exp=True) as live:
        
        
        live.log_metric('accuracy',acc)
        live.log_metric('f1_score',f1_scor)
        live.log_metric('precission',precision_scor)
        live.log_metric('recall',recall_scor)
        
        live.log_param('test_size',test_size)
        live.log_param('n_estimators',n_estimators)
        
        
        
    metrics_dict = {
        'acc':acc, 
        'f1':f1_scor,
        'pre':precision_scor,
        'reca':recall_scor
    }

    with open('metrics.json','w') as file:
        json.dump(metrics_dict, file, indent=4)
        
        
        
    
except Exception as e:
    print(e)