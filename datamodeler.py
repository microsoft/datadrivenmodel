import math
import numpy as np
import time
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import save_model
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
from env_data_modeler import env_nn_modeler
from env_data_modeler import env_gb_modeler
from env_data_modeler import env_lstm_modeler
from env_data_modeler import env_poly_modeler
from env_data_modeler import create_nn_model_wrapper
from env_data_modeler import create_lstm_model_wrapper
import h5py
import argparse
import pickle
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from conf_params_var import STATE_SPACE_DIM, ACTION_SPACE_DIM



parser = argparse.ArgumentParser()
parser.add_argument("--use-gb", type=bool, default=False,help="choose gradient boosting as a model")
parser.add_argument("--grid-search", type=bool, default=False, help="use grid search to tune hyperparameter tuning, if applicable")
parser.add_argument("--genetic-search", type=bool, default=False, help="use genetic algorithn to tune hyperparameter if applicable")
parser.add_argument("--use-lstm", type=bool, default=False, help="using lstm model after hyperparameter tuning")
parser.add_argument("--use-nn", type=bool, default=False, help="choose multilayer perceptron as a model")
parser.add_argument("--use-poly", type=bool, default=False, help="choose polynominal fitting as a model")
#parser.add_argument("tune-ga", type=bool, default=False, help="uses genetic algorithm and TPOT library for hyperparameter tuning")
parser.add_argument("--tune-rs", type=bool, default=False, help="uses random search from scikitlearn for hyperparameter tuning")



def read_env_data():
    try:
        with open('./env_data/x_set.pickle', 'rb') as f:
            x_set = pickle.load(f)
        with open('./env_data/y_set.pickle', 'rb') as f:
            y_set = pickle.load(f)
    except:
        print('No data was available. Note: x_set.pickle and y_set.pickle should be found in env_data folder')
    return x_set, y_set


if __name__=="__main__":

    args=parser.parse_args()

    markovian_order=int(2)

    state_space_dim=int(STATE_SPACE_DIM)
    action_space_dim=int(ACTION_SPACE_DIM)

    polydegree=int(1)

    try: 
        outputfile = open("EvalutationScores.txt", "a")
    except:
        outputfile = open("EvalutationScores.txt", "w")
    

    randomsearch_dist_lstm = {
        "activation": ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
        "dropout_rate": [0,0.1,0.5],
        "num_neurons": np.random.randint(2, 11, size=1),
        "num_hidden_layers": np.random.randint(2,11, size=1),
        "learning_rate": np.random.choice([10**-1,10**-3], size=1),
        "num_lstm_units":np.random.randint(2,101,size=1),
        "decay": np.random.uniform(10**-3,10**-9, size=1),
        "markovian_order":[markovian_order],
        "state_space_dim":[state_space_dim],
        "action_space_dim":[action_space_dim]}

    random_search_nn = {
        "activation": ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
        "dropout_rate": [0,0.1,0.5],
        "num_neurons": np.random.randint(2, 30, size=1),
        "num_hidden_layers": np.random.randint(2,20, size=1),
        "learning_rate": np.random.choice([10**-1,10**-3], size=1),
        "decay": np.random.uniform(10**-3,10**-9, size=1),
        "state_space_dim":[state_space_dim],
        "action_space_dim":[action_space_dim]}

    random_search_gb={
        "loss":["ls", "lad", "huber"],
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
        "min_samples_split": [2,10] ,#np.linspace(0.1, 0.5, 10),
        "min_samples_leaf": [1,2,10] ,#np.linspace(0.1 0.5, 10),
        "max_depth":[3,5],
        "max_features":["log2","sqrt"],
        "criterion": ["friedman_mse",  "mae"],
        "subsample":[0.5, 0.6, 0.8, 0.85, 0.9, 0.95, 1.0,1.0,1.0],
        "n_estimators":[50,100,200]
    }

    x_set, y_set=read_env_data()

    if args.use_nn==True:
        scaler_x_set=preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(x_set)
        scaler_y_set=preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(y_set)
        joblib.dump(scaler_x_set, './models/scaler_x_set.pkl')
        joblib.dump(scaler_y_set, './models/scaler_y_set.pkl')
        x_set=scaler_x_set.transform(x_set)
        y_set=scaler_y_set.transform(y_set)

    if args.use_lstm==True:
        l=x_set.shape[0]
        m=x_set.shape[1]
        n=x_set.shape[2]
        print('reshaping data for normalization ..')
        print('shape of original inputs', x_set.shape, y_set.shape)
        x_set=x_set.reshape(l, m*n)
        scaler_x_set=preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(x_set)
        scaler_y_set=preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(y_set)

        joblib.dump(scaler_x_set, './models/scaler_x_set.pkl')
        joblib.dump(scaler_y_set, './models/scaler_y_set.pkl')

        x_set=scaler_x_set.transform(x_set)
        y_set=scaler_y_set.transform(y_set)

        x_set=x_set.reshape((l,m,n))

    args = parser.parse_args()
    x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.33, random_state=42)


    if args.tune_rs==True:
        if args.use_lstm==True:
            model=KerasRegressor(build_fn=create_lstm_model_wrapper,epochs=10, batch_size=1024, verbose=1)
            random_search = RandomizedSearchCV(estimator=model, param_distributions=randomsearch_dist_lstm, n_iter=50, n_jobs=-1, cv=5)
            result = random_search.fit(x_train, y_train)
            print("Best: %f using %s" % (result.best_score_, result.best_params_))
            filename='./models/lstm_random_search_results_'+str(100*result.best_score_)+'.pkl'
            joblib.dump(result.best_params_, filename)

        elif args.use_nn==True:
            model=KerasRegressor(build_fn=create_nn_model_wrapper,epochs=100, batch_size=1024, verbose=1)
            random_search = RandomizedSearchCV(estimator=model, param_distributions=random_search_nn, n_iter=50, n_jobs=-1, cv=5)
            result = random_search.fit(x_train, y_train)
            print("Best: %f using %s" % (result.best_score_, result.best_params_))
            filename='./models/nn_random_search_results_'+str(100*result.best_score_)+'.pkl'
            joblib.dump(result.best_params_, filename)
            config={"epochs": 1000,
                "batch_size": 512,
                "activation":result.best_params_["activation"],
                "n_layer": result.best_params_["num_hidden_layers"],
                "n_neuron": result.best_params_["num_neurons"],
                "lr": result.best_params_["learning_rate"],
                "decay": result.best_params_["decay"],
                "dropout": result.best_params_["dropout_rate"]
                    }
            nn_estimator=env_nn_modeler(state_space_dim=state_space_dim,action_space_dim=action_space_dim)
            nn_estimator.create_model(config)
            nn_estimator.train_nn_model(x_train,y_train,config["epochs"],config["batch_size"])
            nnmodel=nn_estimator.model
            nn_estimator.evaluate_nn_model(x_test, y_test,config["batch_size"])
            test_score=nn_estimator.score[1]*100
            randomsample=np.random.random_integers(0,10,1)
            x_sample=x_set[randomsample]
            print('random sample:', x_sample)
            predict_sample=nnmodel.predict(x_sample)
            print('estimator prediction: ', predict_sample)
            print('actual value:', y_set[randomsample])
            modelname='./models/nnmodel'+str(int(test_score))+'.h5'
            nnmodel.save(modelname)
            modelname2='./models/nnmodel.h5'
            nnmodel.save(modelname2)
        time.sleep(10)



    if args.use_gb==True and args.tune_rs==True:
        for i in range (0, y_set.shape[1]):
            gb_estimator=env_gb_modeler(state_space_dim=state_space_dim,action_space_dim=action_space_dim)
            gb_estimator.create_gb_model()
            gb_estimator.train_gb_model(x_train,y_train[:,i])
            score=gb_estimator.evaluate_gb_model(x_test, y_test[:,i])
            print('evaluation score for default is: ', score)
            model=GradientBoostingRegressor()
            random_search = RandomizedSearchCV(estimator=model, param_distributions=random_search_gb, n_iter=10, n_jobs=-1, cv=3, verbose=0)
            result = random_search.fit(x_train, y_train[:,i])
            print("Best: %f using %s" % (result.best_score_, result.best_params_))
            filename='./models/gb_random_search_results_'+str(i)+'th'+str(100*result.best_score_)+'.pkl'
            joblib.dump(result.best_params_, filename)
            model_opt=GradientBoostingRegressor(result.best_params_)
            modelname='./models/gbmodel'+str(int(i))+'.sav'
            joblib.dump(model_opt, modelname)

    elif args.use_gb==True and args.tune_rs==False:
        outputfile.write("\n")  
        outputfile.write("\n") 
        outputfile.write("\n") 
        outputfile.write("--------------Evaluation score for gradient boost ----------------")
        outputfile.write("\n")        
        print('using gradient boost regressor ....')
        scaler_x_set=preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(x_set)
        scaler_y_set=preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(y_set)
        joblib.dump(scaler_x_set, './models/scaler_x_set.pkl')
        joblib.dump(scaler_y_set, './models/scaler_y_set.pkl')
        x_set=scaler_x_set.transform(x_set)
        y_set=scaler_y_set.transform(y_set)
        
        for i in range (0, y_set.shape[1]):
            gb_estimator=env_gb_modeler()
            gb_estimator.create_gb_model()
            gb_estimator.train_gb_model(x_train,y_train[:,i])
            score=gb_estimator.evaluate_gb_model(x_test, y_test[:,i])
            print('evaluation score is:', score)
            outputfile.write("Evaluation score for state number " + str(i) + " is: " + str(score))
            outputfile.write("\n")
            modelname='./models/gbmodel'+str(int(i))+'.sav'
            joblib.dump(gb_estimator.model, modelname)

    if args.use_poly==True:
        outputfile.write("\n")  
        outputfile.write("\n") 
        outputfile.write("\n") 
        outputfile.write("--------------Evaluation score for polynomial fit ----------------")
        outputfile.write("\n") 
        print('using polynomial fitting ....')
        for i in range (0, y_set.shape[1]):
            poly_estimator=env_poly_modeler()
            poly_estimator.create_poly_model(degree=polydegree)
            poly_estimator.train_poly_model(x_train,y_train[:,i])
            score=poly_estimator.evaluate_poly_model(x_test, y_test[:,i])
            print('evaluation score is:', score)
            outputfile.write("Evaluation score for state number " + str(i) + " is: " + str(score))
            outputfile.write("\n")
            modelname='./models/polymodel'+str(int(i))+'.sav'
            joblib.dump(poly_estimator.model, modelname)
            joblib.dump(poly_estimator.poly, './models/polydegree.sav')
            randomsample=np.random.random_integers(0,10,1)
            x_sample=x_set[randomsample]
            #print('random sample:', x_sample)
            predict_sample=poly_estimator.predict_poly_model(x_sample)
            print('estimator prediction: ', predict_sample)
            print('actual value:', y_set[randomsample,i])

# use the hyperparamter tuned network

##default neural network without hyperparamter tuning
    if  args.tune_rs==False and args.use_lstm==True:
        outputfile.write("\n")  
        outputfile.write("\n") 
        outputfile.write("\n") 
        outputfile.write("--------------Evaluation score for LSTM ----------------")
        outputfile.write("\n") 
        the_lstm_estimator=env_lstm_modeler()
        config={"epochs": 100,
                "batch_size": 512,
                "activation":'linear',
                "n_hidden_layer": 1,
                "n_neuron": 12,
                "lr": 10**-1,
                "decay": 10**-3,
                "dropout": 0.5,
                "markovian_order":markovian_order, # should be >1 , zero order==>1
                "num_lstm_units": 10}
        the_lstm_estimator.create_model(config)
        the_lstm_estimator.train_nn_model(x_train,y_train,config["epochs"],config["batch_size"])
        lstmmodel=the_lstm_estimator.model
        the_lstm_estimator.evaluate_nn_model(x_test, y_test,config["batch_size"])
        test_score=the_lstm_estimator.score[1]*100
        randomsample=np.random.random_integers(0,10,1)
        x_sample=x_set[randomsample]
        print('random sample:', x_sample)
        predict_sample=lstmmodel.predict(x_sample)
        print('estimator prediction: ', predict_sample)
        print('actual value:', y_set[randomsample])
        outputfile.write("Evaluation score is: " + str(test_score/100))
        outputfile.write("\n")
        modelname='./models/lstmmodel'+str(int(test_score))+'.h5'
        print(modelname)
        lstmmodel.save(modelname)
        modelname2='./models/lstmmodel.h5'
        lstmmodel.save(modelname2)

    if args.tune_rs==False and args.use_nn==True:
        outputfile.write("\n")  
        outputfile.write("\n") 
        outputfile.write("\n") 
        outputfile.write("--------------Evaluation score for MLP neural network ----------------")
        outputfile.write("\n") 
        nn_estimator=env_nn_modeler(state_space_dim=state_space_dim,action_space_dim=action_space_dim)
        config={"epochs": 100,
                "batch_size": 512,
                "activation": 'linear',
                "n_layer": 5,
                "n_neuron": 12,
                "lr": 10**-5,
                "decay": 10**-7,
                "dropout": 0.5}
        nn_estimator.create_model(config)
        nn_estimator.train_nn_model(x_train,y_train,config["epochs"],config["batch_size"])
        nnmodel=nn_estimator.model
        nn_estimator.evaluate_nn_model(x_test, y_test,config["batch_size"])
        test_score=nn_estimator.score[1]*100
        outputfile.write("Evaluation score is: " + str(test_score/100))
        outputfile.write("\n")
        randomsample=np.random.random_integers(0,10,1)
        x_sample=x_set[randomsample]
        print('random sample:', x_sample)
        predict_sample=nnmodel.predict(x_sample)
        print('estimator prediction: ', predict_sample)
        print('actual value:', y_set[randomsample])
        modelname='./models/nnmodel'+str(int(test_score))+'.h5'
        nnmodel.save(modelname)
        modelname2='./models/nnmodel.h5'
        nnmodel.save(modelname2)
    else:
        pass

    outputfile.close()
