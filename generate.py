import numpy as np
import pandas as pd
import matplotlib as plt
import random
import sys, os
import time
import policies
import pdb
# from simulator import class as Model
from sim.qube_simulator import QubeSimulator as Model

def get_config():
    # TODO: Modify test_config for your problem
    test_config = {
        # Parameters
        "Lp": 0.129,
        "mp": 0.024,
        "Rm": 8.4,
        "kt": 0.042,
        "km": 0.042,
        "mr": 0.095,
        "Lr": 0.085,
        "Dr": 0.00027,
        "Dp": 0.00005,
        "frequency": 80,
        
        # Initial Conditions
        "theta": np.random.uniform(-0.27, 0.27),
        "alpha": np.random.uniform(-0.05, 0.05), # make sure pi if resetting downward
        "theta_dot": np.random.uniform(-0.05, 0.05),
        "alpha_dot": np.random.uniform(-0.05, 0.05),
        
        "max_iterations": 640
    }
    return test_config

def get_state(model):
    sim_state = {
        ## TODO: Add simulator state as dictionary
        "theta": model.state[0],
        "alpha": model.state[1],
        "theta_dot": model.state[2],
        "alpha_dot": model.state[3]
    }
    return sim_state

def get_terminal(state):
    ## TODO: Add terminal conditions to return True or False
    if abs(state["theta"]) >= np.radians(90):
        terminal = True
    elif abs(state["alpha"]) >= np.radians(60):
        terminal = True
    else:
        terminal = False
    return terminal

def generate_data(total_sample=100000):
    # TODO: specify state and action space dimensions
    action_space_dim = 1
    state_space_dim = 4

    model = Model()
    config = get_config()

    n = 0
    episode = 0
    while True:
        # All methods are based on how they are defined in model i.e. the simulator, QubeSimulator
        # reset 
        episode += 1
        config = get_config()
        model.reset(config)
        s = get_state(model)
        iteration = 0
        if n > total_sample-1:
            break
        else:
            pass 
        for t in range(config["max_iterations"]):
            iteration += 1
            if n > total_sample-1:
                break
                print('maximum number of samples recoded from the environment')
            else:
                pass 
            action = policies.random_policy(s)

            # Concatenate states and actions
            sa = s.copy()
            sa.update(action)
            sa.update({"episode": episode})
            sa.update({"iteration": iteration})
            
            # Send to csv
            sample = pd.DataFrame(data=[sa])
            if n == 0:
                sample.to_csv(data_fname, mode='a', index=False)
            else:
                sample.to_csv(data_fname, mode='a', header=False, index=False)

            
            # TODO: Modify action as input to simulator step
            model.step(
                action['Vm']
            )

            # Get updated sim state after step
            s_ = get_state(model)
            terminal = get_terminal(s_)
            
            n += 1
            s = s_
            if terminal:
                print("Episode finished after {} timesteps".format(t+1))
                break

    print('Saving data as csv format, located here: {}'.format(os.getcwd()+'/'+data_fname))

if __name__ == '__main__':
    ## TODO: specify filename for data
    data_fname = 'data.csv'

    try:
        os.remove(data_fname)
    except:
        pass

    generate_data(total_sample=100000)