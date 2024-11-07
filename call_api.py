import httpx
import numpy as np
import requests
import json

def getData(url, days, sims, beta_epsilon):
    params = {
        "days": days, "sims": sims, "beta": beta_epsilon[0],
        "epsilon": beta_epsilon[1]
    }
    
    response = requests.post(url, json=params)
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
    data_request = requests.get("http://host.docker.internal:8080/multiple",headers= {"Content-Type": "application/json"},json=mult_params)
    #requests.get("http://host.docker.internal:8000/multiple",headers= {"Content-Type": "application/json"},json=mult_params)
    return np.array(json.loads(data_request.json())['train_set'])
    
    


