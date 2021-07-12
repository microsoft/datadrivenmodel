"""
Fixed policies to test our sim integration with. These are intended to take
Brain states and return Brain actions.
"""

import random
from typing import Dict
import requests
import numpy as np
import pandas as pd

def random_policy(state):
    """
    Ignore the state, move randomly.
    """
    action = {"Vm": random.uniform(-3, 3)}
    return action

def lqr(state):
    """
    Ignore the state, move randomly.
    """
    s = np.array([state['theta'], state['alpha'], state['theta_dot'], state['alpha_dot']])
    action = {"Vm": [np.array([-2.0, 35.0, -1.5, 3.0]).T.dot(s)]}
    return action

def data_policy(state, iteration):
    df = pd.read_csv('./logs/2021-07-12-11-52-05_random_log.csv')
    action = {"Vm": df["action_Vm"].iloc[iteration]}

    return action

def brain_policy(
    state: Dict[str, float], exported_brain_url: str = "http://localhost:5000"
):

    prediction_endpoint = f"{exported_brain_url}/v1/prediction"
    response = requests.get(prediction_endpoint, json=state)

    return response.json()
