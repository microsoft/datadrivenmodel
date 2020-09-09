# Pipeline template for building surrogate models from slow simulation environment and training brain on bonsai platform


### This branch contains minumum number of files for the generation of the surrogate models and their integration with cs project template. It also contains the code for calculating feature importance and detecting outliers in the data


### Main steps to follow:
1) Generate random samples from similation environment and save as numpy array in env_data folder. 
example on how to save numpy array using pickle:
```bash
with open('./env_data/x_set.pickle', 'wb') as f:
    pickle.dump(x_set, f, pickle.HIGHEST_PROTOCOL)
with open('./env_data/y_set.pickle', 'wb') as f:
    pickle.dump(y_set, f, pickle.HIGHEST_PROTOCOL)
```
---------------
2) Second step is to build surrogate model of the environmet using datamodeler.py file. One my use gradient boost, polynomial, and multi-layer perceptron neural network or LSTM to build models. It should be noted that for the case of lstm, the data (x_set.pickle, y_set.picle) should be three dimension data: [sample, timestep, feature]. An example for data preparation can be found in gym_sampler.py file. 
Examples are as follows.
```bash 
>>> python datamodeler.py --use-poly=True
>>> python datamodeler.py --use-nn=True
>>> python datamodeler.py --use-gb=True
```
If it is desirable to perform hyperparamter tuning for MLP, LSTM, and gradient boost models. 
```bash
> python datamodeler.py --use-gb=True --tune-rs=True
```
##### The resulting models are automatically saved into _models_ folder. Evaluation scores from the test data set are printed on the terminal during model training.  
----------------
3) Third step is to modify star.py file for machine teaching and run hub.py file. A sample implementation is shown in star_sample_implementation.py file. One may choose to integrate the model within simulation_integration.py file (from cs project template. It should be noted that all you need to do is to call Modelpredictor class form predictor.py. This class can be used in cs project template or any other custom python code.  

#### Code snippet: 
-- initialize class with model type, state space and action space dimensions and markovian order

```bash
from predictor import ModelPredictor

modeltype="gb"
noise_percentage=0
state_space_dim=4
action_space_dim=1
markovian_order=0	
predictor=ModelPredictor(modeltype=modeltype,noise_percentage=noise_percentage, state_space_dim=state_space_dim,action_space_dim=action_space_dim,markovian_order=markovian_order)
```
-- calculate next state as a function of current state and current action. IMPORTANT: input state and action are arrays. You need to convert brain action, i.e. dictionary, to an array before feeding into the predictor class. 

```bash
next_state=predictor.predict(action=action, state=state)
```

#### LSTM model is not supported, please refer to dev-min branch if you wish to use it.

## Data Cleaning: 
This helps in removing the Outliers and NaNs in the data. Outlier detection algorithm fits y_Set as a function of x_set, and if the prediction by the model is far away from the actual value then we define the point as outlier
### Main steps to follow to detect outliers:
Once the data is generated (in the same way as data generation process for datadriven modeling), we can detect outliers in the following way:
```bash 
>>> python checkDataQuality.py
>>> python checkDataQuality.py --thrhld 4
```
By default, outlier threshold is 3, it can be adjusted based on the data. 

## Feature Importance
This code helps in computing feature importance using gradient boosting trees.
### Main steps to follow to detect outliers:
Once the data is generated (in the same way as data generation process for datadriven modeling), we can compute feature importance in the following way:
```bash 
>>> python featureImportance.py
```

