"""
Fixed policies to test our sim integration with. These are intended to take
Brain states and return Brain actions.
"""

import random
from typing import Dict
import requests
import numpy as np

def random_policy(state):
    """
    Ignore the state, move randomly.
    """
    action = {"action_Vm": random.uniform(-3, 3)}
    return action


def brain_policy(
    state: Dict[str, float], exported_brain_url: str = "http://localhost:5000"
):

    prediction_endpoint = f"{exported_brain_url}/v1/prediction"
    response = requests.get(prediction_endpoint, json=state)

    return response.json()

def mixed_policy(state):
    p = 0.5
    if np.random.uniform(0, 1) < p:
        K = np.array([-2.0, 35.0, -1.5, 3.0])
        state = np.array([state["state_theta"], state["state_alpha"], state["state_theta_dot"], state["state_alpha_dot"]])
        return {"action_Vm":  K.T.dot(state)}
    else:
        action = {"action_Vm": random.uniform(-3, 3)}
        return action