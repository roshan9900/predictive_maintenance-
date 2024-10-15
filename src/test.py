import json
import requests
from data_model import pm


url = 'http://127.0.0.1:8000/predict'

x_new = dict(
    Air_temperature= 23,
    Process_temperature= 423,
    Rotational_speed= 32,
    Torque= 42,
    Tool_wear= 12,
    H= 1,
    L= 0,
    M= 1
    )
x_new_json = json.dumps(x_new)
responce = requests.post(url, x_new_json)
print('response ',responce)