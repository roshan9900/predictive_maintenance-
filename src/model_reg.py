import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import json
import dagshub
from mlflow.models import infer_signature

dagshub.init(repo_owner='roshansalunke91', repo_name='predictive_maintenance-', mlflow=True)

# Set up MLflow
mlflow.set_experiment('model_reg')
mlflow.set_tracking_uri('https://dagshub.com/roshansalunke91/predictive_maintenance-.mlflow')

try:
    
    # Load datasets
        train = pd.read_csv('./data/processed/train_processed.csv')
        test = pd.read_csv('./data/processed/test_processed.csv')

    # Prepare features and target
        x_train = train.drop('Machine failure', axis=1)
        y_train = train['Machine failure']
        
        
        x_test = test.drop('Machine failure', axis=1)
        y_test = test['Machine failure']
    
    # Initialize the model and parameter grid
        rf = RandomForestClassifier(random_state=42)
        param_dict = {
        'n_estimators': [100, 300, 500],
        'max_depth': [None, 5, 10]
        }
    
    # Perform Randomized Search
        random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dict, cv=5, n_jobs=-1, verbose=2)

        with mlflow.start_run(run_name='random forest') as parent_run:
            random_search.fit(x_train, y_train)

            for i in range(len(random_search.cv_results_['params'])):
                with mlflow.start_run(run_name=f'combination of {i+1}',nested=True) as child_run:
                    mlflow.log_params(random_search.cv_results_['params'][i])
                    mlflow.log_metric('mean_test_score', random_search.cv_results_['mean_test_score'][i])
        # Log best parameters
            best_params = random_search.best_params_
            mlflow.log_params(best_params)

            signature = infer_signature(x_test, random_search.best_estimator_.predict(x_test))
        # Save the best model
            mlflow.sklearn.log_model(random_search.best_estimator_, 'best estimator', signature=signature)
            mlflow.log_artifact(__file__)

        # Load test data
            x_test = test.drop('Machine failure', axis=1)
            y_test = test['Machine failure']

        # Predict and evaluate
            pred = random_search.predict(x_test)
            acc = accuracy_score(y_test, pred)
            f1_scor = f1_score(y_test, pred)
            precision_scor = precision_score(y_test, pred)
            recall_scor = recall_score(y_test, pred)

        # Log metrics
            mlflow.log_metric('accuracy', acc)
            mlflow.log_metric('f1_score', f1_scor)
            mlflow.log_metric('precision', precision_scor)
            mlflow.log_metric('recall', recall_scor)

        # Log confusion matrix
            cm = confusion_matrix(y_test, pred)
            plt.figure(figsize=(5, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.savefig('con_met.png')
            mlflow.log_artifact('con_met.png')

        # Log input datasets
            mlflow.log_artifact('./data/processed/train_processed.csv', artifact_path='train_data')
            mlflow.log_artifact('./data/processed/test_processed.csv', artifact_path='test_data')

        # Save metrics to a JSON file
            metrics_dict = {
            'accuracy': acc,
            'f1_score': f1_scor,
            'precision': precision_scor,
            'recall': recall_scor
        }
            with open('metrics.json', 'w') as file:
                json.dump(metrics_dict, file, indent=4)

except Exception as e:
    print(f"An error occurred: {e}")