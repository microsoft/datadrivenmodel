# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

import joblib
from env_data_modeler import env_gb_modeler
import matplotlib as mpl

import argparse
import pickle
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--thrhld", type=float, default=3,help="choose the threshold to declare outlier (thrhld*sigma)")

def read_env_data():
    try:
        with open('./env_data/x_set.pickle', 'rb') as f:
            x_set = pickle.load(f)
        with open('./env_data/y_set.pickle', 'rb') as f:
            y_set = pickle.load(f)
    except:
        print('No data was available. Note: x_set.pickle and y_set.pickle should be found in env_data folder')
    return x_set, y_set


######################## Functions for Outlier Detection and Ploting  ###################
def plotOutliers(y_set, y_predict_all, outlier_data, config):
    fig = plt.figure()
    numSubPlots = y_set.shape[1]

    outlierData = outlier_data['y' + str(0)]
    dataLabel = []
    for key, value in config['IO']['feature_name'].items():
        if value == 'state':
            dataLabel.append(key)

    ax1 = plt.subplot(numSubPlots, 1, 0+1)
    plt.plot(y_set[:,0], label=dataLabel[0], linewidth=1, color = 'blue' )
    plt.plot(y_predict_all[0], label=dataLabel[0], linewidth=1, color = 'black' )
    plt.scatter(outlierData,y_set[outlierData,0], label='outlier', linewidth=1, marker = '*', color = 'red', s = 50)
    plt.xticks(rotation='horizontal')
    plt.legend(loc='upper right')
    
    for i in range(1,numSubPlots):        
        outlierData = outlier_data['y' + str(i)]

        ax2 = plt.subplot(numSubPlots, 1, i+1, sharex=ax1)
        plt.plot(y_set[:,i], label=dataLabel[i], linewidth=1, color = 'blue' )
        plt.plot(y_predict_all[i], label=dataLabel[i], linewidth=1, color = 'black' )
        plt.scatter(outlierData,y_set[outlierData,i], label='outlier', linewidth=1, marker = '*', color = 'red', s = 50)
        plt.xticks(rotation='horizontal')
        plt.legend(loc='upper right')


    # plt.show()
                                         
def findOutliersAll(x_set,y_set, thrhld=2):
    ## Computing Feature importance using gradient boosting
    print('computing Outliers ....')
    outlier_data = {}
    y_predict_all = []
    for i in range (0, y_set.shape[1]):
        gb_estimator=GradientBoostingRegressor(n_iter_no_change=50, validation_fraction=.2)
        gb_model= gb_estimator.fit(x_set,y_set[:,i])
        y_predict = gb_estimator.predict(x_set)
        outlier_data['y' + str(i)] = findOutlier(y_set[:,i], y_predict, thrhld=thrhld)
        y_predict_all.append(y_predict)
        print('y', str(i), ': ', outlier_data['y' + str(i)])
    return outlier_data, y_predict_all

def findOutlier(y, y_predict, thrhld=2):
    y_std = y.std(axis = 0)
    outL = np.where(np.abs(y-y_predict) > thrhld*y_std) # tuple
    return outL[0]

###################### Lot Inputs #########################
def plotInputs(x_set, y_set, config):
    fig = plt.figure()
    numSubPlots = x_set.shape[1] - y_set.shape[1]  ## Num of inputs
    
    dataLabel = []
    for key, value in config['IO']['feature_name'].items():
        if value == 'action':
            dataLabel.append(key)
    ax1 = plt.subplot(numSubPlots, 1, 1)
    plt.plot(x_set[:,y_set.shape[1]+0], label=dataLabel[0], linewidth=1, color = 'blue' )
    plt.xticks(rotation='horizontal')
    plt.legend(loc='upper right')
    
    for i in range(1,numSubPlots):
        ax2 = plt.subplot(numSubPlots, 1, i+1, sharex=ax1)
        plt.plot(x_set[:,y_set.shape[1]+i], label=dataLabel[i], linewidth=1, color = 'blue' )
        plt.xticks(rotation='horizontal')
        plt.legend(loc='upper right')

    # plt.show()
    return fig

################ Functions for NaN checks
def hasNaN(x_set):
    for i in range (0, x_set.shape[1]):
        print('x', str(i), ' has NaN: ', np.isnan(np.sum(x_set[:,i])))
    return np.isnan(np.sum(x_set[:, i]))

def maxMinMeanStd(x, varName = 'x'):
    return x.min(axis = 0), x.max(axis = 0), x.mean(axis = 0), x.std(axis = 0)



if __name__=="__main__":
    args=parser.parse_args()

    with open('config/config_model.yml') as cmfile:
        config = yaml.full_load(cmfile)

    x_set, y_set = read_env_data()
    mpl.rcParams['agg.path.chunksize'] = max(10000, x_set.shape[1]+100)

    ##################### Outlier Code Usage ###############################################

    ## Adding Outliers as the original data doesn't have
    addOutlierIdx = [100, 10000,10001,10002, 10003, 20000, 30000]
    for i in addOutlierIdx:
        y_set[i,0] = 2.
        x_set[i+1,0] = 2.

        y_set[i,1] = 1.5
        x_set[i+1,1] = 1.5

    ## Find Outlier and Plot them
    outlier_data, y_predict_all = findOutliersAll(x_set, y_set, thrhld=args.thrhld)
    modelname='./models/OutlierData_Y.sav'
    joblib.dump(outlier_data, modelname)
    plotOutliers(y_set, y_predict_all, outlier_data, config)
    plotInputs(x_set,y_set, config)
    plt.show()

    ############################# Detecting NaN ######################
    hasNaN(x_set)