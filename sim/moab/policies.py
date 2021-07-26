"""
Fixed policies to test our sim integration with. These are intended to take
Brain states and return Brain actions.
"""

import numpy as np
import pickle
from typing import Any, Dict, List

def random_policy(state: Dict = None) -> Dict:
    """
    Ignore the state, move randomly.
    """
    action = {
        'input_roll': np.random.uniform(-1, 1),
        'input_pitch': np.random.uniform(-1,1)
    }
    return action

def coast(state: Dict = None) -> Dict:
    """
    Ignore the state, go straight.
    """
    action = {
        'input_roll': 0,
        'input_pitch': 0
    }
    return action

def small_perturbations(action: Dict = None) -> Dict:
    """
    5% perturbation
    """
    action = {
        'input_roll': list(action.values())[0]+np.random.uniform(-0.05, 0.05),
        'input_pitch': list(action.values())[1]+np.random.uniform(-0.05,0.05)
    }
    
    return action

# 
def benchmark(state: Dict = None) -> Dict:
    """
    TO DO: Add benchmark control policies such as PID, LQR, LQG, MPC
    These benchmark policies can be use-case specific. If benchmark is not accessible,
    you can also replay pre-recorded datasets of actions corresponding states.
    """
    pass

POLICIES = {"random": random_policy,
            "coast": coast,
            "small_perturbations": small_perturbations}