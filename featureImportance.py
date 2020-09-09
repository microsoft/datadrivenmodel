import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import joblib
from env_data_modeler import env_gb_modeler

import argparse
import pickle
from conf_params_var import STATE_SPACE_DIM, ACTION_SPACE_DIM, FEATURE_NAME


parser = argparse.ArgumentParser()
parser.add_argument("--widthbar", type=float, default=.5,help="choose width of the bars around")
parser.add_argument("--featureNameProvided", type=bool, default=False,help="write xlabel name in conf_params_var.py")

def read_env_data():
    try:
        with open('./env_data/x_set.pickle', 'rb') as f:
            x_set = pickle.load(f)
        with open('./env_data/y_set.pickle', 'rb') as f:
            y_set = pickle.load(f)
    except:
        print('No data was available. Note: x_set.pickle and y_set.pickle should be found in env_data folder')
    return x_set, y_set


def feature_plots(feature_data, total_width=0.5, featureNameProvided = False):
    fig, ax = plt.subplots()

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  
    n_bars = len(feature_data)
    bar_width = total_width / n_bars # width of single bar
    bars = []
    for i, (name, values) in enumerate(feature_data.items()):
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width , color=colors[i % len(colors)])
        bars.append(bar[0])


    ax.legend(bars, feature_data.keys())
    plt.xlabel('Feature Number', fontsize=18)
    plt.ylabel('Feature Importance', fontsize=18)

    if featureNameProvided == True:
        plt.xticks(ticks=range(len(FEATURE_NAME)), labels=FEATURE_NAME)
    plt.show()



if __name__=="__main__":
    args=parser.parse_args()

    state_space_dim=int(STATE_SPACE_DIM)
    action_space_dim=int(ACTION_SPACE_DIM)


    x_set, y_set=read_env_data()
    x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.33, random_state=42)

    ## Computing Feature importance using gradient boosting
    print('computing Feature Importance ....')
    feature_importance_data = {}
    for i in range (0, y_set.shape[1]):
        gb_estimator=env_gb_modeler()
        gb_estimator.create_gb_model()
        gb_model= gb_estimator.train_gb_model(x_train,y_train[:,i])
        feature_importance_data['y' + str(i)] = gb_model.feature_importances_
        print('feature importance for y', str(i), ' :', feature_importance_data['y' + str(i)])
       
    modelname='./models/feature_importance.sav'
    joblib.dump(feature_importance_data, modelname)
        
    feature_plots(feature_importance_data, total_width= args.widthbar, featureNameProvided = args.featureNameProvided)




