from fastapi import FastAPI
import pickle
import pandas
from data_model import pm
import pandas as pd

app = FastAPI(
    title='Predictive Metenance',
    description='Finding the failure of the machines'
)

with open('C:/Users/Roshan Salunke/Downloads/Data Science Course/MLOps/predictive_mentenance/model.pkl','rb') as f:
    model = pickle.load(f)
    
@app.get('/')
def index():
    return "welcome to the app"

@app.post('/predict')
def model_predict(pm: pm):
    sample = pd.DataFrame({
        'Air temperature':[pm.Air_temperature],
    'Process temperature':[pm.Process_temperature],
    'Rotational speed':[pm.Rotational_speed],
    'Torque':[pm.Torque],
    'Tool wear': [pm.Tool_wear],
    'H':[pm.H],
    'L':[pm.L],
    'M':[pm.M]
        
    })
    
    
    predicted_value = model.predict(sample)
    
    if predicted_value==1:
        return 'Fail'
    else:
        return 'Not Fail'