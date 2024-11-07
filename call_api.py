import httpx
import numpy as np
import requests
import json

def getData(url, days, sims, beta_epsilon):
    params = {
        "days": days, "sims": sims, "beta": beta_epsilon[0],
        "epsilon": beta_epsilon[1]
    }
    
    response = requests.post(url, headers= {"Content-Type": "application/json", "X-API-Key": "e54d4431-5dab-474e-b71a-0db1fcb9e659"},json=params)
    return response.json()
'''
def main():
    result = getData("http://192.168.1.255:8000/", 5, 2, [0.1, 0.2])
    print(result)
'''
def get_data(days, sims, beta_epsilon):
    mult_params = {
    'days': days,
    'sims': sims,
    'beta_epsilon': beta_epsilon.tolist()
    }
    data_request = requests.post("https://gleam-seir-api-883627921778.us-west1.run.app/multiple",headers= {"Content-Type": "application/json", "X-API-Key": "e54d4431-5dab-474e-b71a-0db1fcb9e659"},json=mult_params)
    return np.array(json.loads(data_request.json())['train_set'])
    
    


