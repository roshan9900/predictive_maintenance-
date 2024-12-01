import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost  import XGBClassifier



l = [GradientBoostingClassifier(random_state=42), RandomForestClassifier(random_state=42),
     AdaBoostClassifier(random_state=42), DecisionTreeClassifier(random_state=42),
     LogisticRegression(random_state=42), SVC(random_state=42), XGBClassifier(random_state=42)]

ai4i_2020_predictive_maintenance_dataset = fetch_ucirepo(id=601) 
X = ai4i_2020_predictive_maintenance_dataset.data.features 
y = ai4i_2020_predictive_maintenance_dataset.data.targets 
y = y['Machine failure']

one = OneHotEncoder()
dfone = pd.DataFrame(one.fit_transform(pd.DataFrame(X['Type'])).toarray(),columns=['H','L','M'])
df = pd.concat([X,dfone],axis=1)
df.drop('Type',axis=1,inplace=True)



x,y = SMOTE().fit_resample(df,y)


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.2, random_state=42)


d = {}

def model_building(model_list):
    for i in model_list:
        i.fit(x_train, y_train)
        pred = i.predict(x_test)
        print('*'*50)
        print(f'classification report of {i}')
        print(classification_report(y_test, pred))
        
        if i not in d:
        
            d[i] = [['acc',accuracy_score(pred, y_test)],['f1',f1_score(y_test, pred)]]
                        
    return d

        
d = model_building(l)
    

