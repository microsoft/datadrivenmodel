from predictor import ModelPredictor as predictor
from datamodeler import read_env_data
from conf_params_var import STATE_SPACE_DIM, ACTION_SPACE_DIM

from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd 


HORIZON = 5

if __name__ == "__main__":
    x_set, y_set = read_env_data()
    modeltype = 'nn'
    predictor = predictor(modeltype=modeltype,
                                    noise_percentage = 0,
                                    state_space_dim = STATE_SPACE_DIM,
                                    action_space_dim = ACTION_SPACE_DIM,
                                    markovian_order = 0)

    predicted_states = np.empty_like(y_set)
    log_actions = np.empty(shape =(x_set.shape[0], ACTION_SPACE_DIM))
    
    for i in range(0, x_set.shape[0]-HORIZON, HORIZON):
        state = x_set[i, : STATE_SPACE_DIM]
        for j in range(0, HORIZON):
            action = x_set[i+j, STATE_SPACE_DIM: STATE_SPACE_DIM + ACTION_SPACE_DIM]
            log_actions [i+j,:] = action 
            new_state = predictor.predict(state = state, action = action) 
            predicted_states[i+j, :] = new_state
            state = new_state

    state_columns = []
    for i in range(STATE_SPACE_DIM):
        state_columns.append('state'+str(i))
    
    action_columns = []
    for i in range(ACTION_SPACE_DIM):
        action_columns.append('action'+str(i))
    
    df_predict = pd.DataFrame(data=predicted_states, columns=state_columns)
    df_actual = pd.DataFrame(data=y_set, columns= state_columns)
    df_log_actions = pd.DataFrame(data= log_actions, columns= action_columns)

    df_predict.to_csv('state_predicted.csv')
    df_actual.to_csv('state_actual.csv')
    df_log_actions.to_csv('action_actual.csv')

    for i in range(0, STATE_SPACE_DIM):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(y_set[:,i], c ='b', marker = "^", label = 'state'+str(i))
        ax.plot(predicted_states[:,i], c = 'r', marker ='o', label = 'predictions'+str(i))
        plt.legend(loc = 2)
        plt.show()