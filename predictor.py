import joblib
import numpy as np
from keras.models import load_model
from collections import deque
import yaml

class ModelPredictor():
    def __init__(self, modeltype='gb', noise_percentage=0, action_space_dim=None, \
        state_space_dim=None, markovian_order=None):
        self.action_space_dim=action_space_dim
        self.state_space_dim=state_space_dim
        self.markovian_order=markovian_order
        self.noise_percentage=noise_percentage
        self.modeltype=modeltype
        self.brain_actions=np.empty((self.action_space_dim))

        print(modeltype, ' is used as the data driven model to train brain.')
        if modeltype=='gb':	
            for i in range(0,self.state_space_dim):
                filename='./models/gbmodel'+str(i)+'.sav'
                loaded_model=joblib.load(filename)
                setattr(self,'model'+str(i),loaded_model)
        elif modeltype=='poly':
            self.polydegree=joblib.load('./sim/datadrivenmodel/models/polydegree.sav')
            print('poyl degree is :', self.polydegree)
            for i in range(0, self.state_space_dim):
                filename='./models/polymodel'+str(i)+'.sav'
                loaded_model=joblib.load(filename)
                setattr(self,'model'+str(i),loaded_model)
        elif modeltype=='nn':
            self.model=load_model('./models/nnmodel.h5')
            self.scaler_x_set = joblib.load('./models/scaler_x_set.pkl')
            self.scaler_y_set = joblib.load('./models/scaler_y_set.pkl') 
            print(self.model.summary())
        elif modeltype=='lstm':
            self.model=load_model('./models/lstmmodel.h5')
            self.scaler_x_set = joblib.load('./models/scaler_x_set_lstm.pkl')
            self.scaler_y_set = joblib.load('./models/scaler_y_set_lstm.pkl') 
            print(self.model.summary())
            #self.action_history_to_brain=self._generate_automated_actions_name()
        else:
            print('ERROR: you need to specify which data driven is being used!!!')
            time.sleep(600)

        # Obtain model limitations
        with open('config/model_limits.yml') as mlimfile:
            self.model_limits = yaml.full_load(mlimfile)

        # Obtain model config for feature names and order
        with open('config/config_model.yml') as cmfile:
            self.model_config = yaml.full_load(cmfile)
    
    def reset_state_random(self):
        if self.modeltype=='lstm':
            self.state=deque(np.random.uniform(low=-1, high=1, size=(self.markovian_order*self.state_space_dim,)),maxlen=self.markovian_order*self.state_space_dim)
        else:
            self.state = np.random.uniform(low=-1, high=1, size=(self.state_space_dim,))
        return self.state

    def reset_state(self, config):
        if self.modeltype=='lstm':
            self.reset_lstm_action_history_zero()
            self.state = deque(np.zeros(shape=(self.markovian_order*self.state_space_dim,)), maxlen=self.markovian_order*self.state_space_dim)
        else:
            self.state = []
            for key, value in self.model_config['IO']['feature_name'].items():
                if value == 'state':
                    self.state.append(config[key]) # Ensure scenario has same key name
        return np.array(self.state)
    
    def reset_lstm_action_history_zero(self):
        self.action_history=deque(np.zeros(shape=(self.markovian_order*self.action_space_dim,)),maxlen=self.markovian_order*self.action_space_dim)
        return self.action_history

    def predict(self, state, action=None):
        if self.modeltype=='lstm':
            for i in range(self.state_space_dim,0,-1):
                self.state=deque(self.state, maxlen=self.markovian_order*self.state_space_dim)
                self.state.appendleft(state[i-1])

            model_input_state=np.reshape(np.array(self.state)+np.random.uniform(low=-self.noise_percentage/100,high=self.noise_percentage/100,\
                     size=self.markovian_order*self.state_space_dim), newshape=(self.markovian_order, self.state_space_dim))
                     
            for i in range(self.action_space_dim,0,-1):
                self.action_history=deque(self.action_history, maxlen=self.markovian_order*self.action_space_dim)
                self.action_history.appendleft(action[i-1])

            model_input_actions=np.reshape(np.ravel(self.action_history), newshape=(self.markovian_order, self.action_space_dim))
            model_input=np.append(model_input_state, model_input_actions, axis=1)

            # Reshape to transform, then reshape back
            model_input = model_input.reshape(1, self.markovian_order*(self.state_space_dim+self.action_space_dim))
            model_input = self.scaler_x_set.transform(model_input)
            model_input = model_input.reshape(self.markovian_order, self.state_space_dim+self.action_space_dim)
            
            # Predict using transformed
            newstates=np.ravel(self.model.predict(np.array([model_input])))

            # Inverse transform prediction
            newstates = np.ravel(self.scaler_y_set.inverse_transform([newstates]))
            return newstates
        else:
            # k=0
            # for key in action.keys():
            # 	self.brain_actions[k]=(action[key])
            # 	k=k+1
            self.state=state 
            self.brain_actions=action
            model_input=np.append(self.state*(1+np.random.uniform(low=-self.noise_percentage/100,high=self.noise_percentage, size=self.state_space_dim)), self.brain_actions)
        
        if self.modeltype=='gb':
            self.state=[]
            for i in range(0, self.state_space_dim):
                ithmodel=getattr(self,'model'+str(i))
                self.state=np.append(self.state, ithmodel.predict(np.array([model_input])),axis=0)	

        elif self.modeltype=='poly':
            self.state=[]
            #print('shape of input is: ', model_input.shape)
            model_input=self.polydegree.fit_transform([model_input])
            #print('model input after transformation is: ', model_input)
            #print('shape of input is: ', model_input.shape)
            model_input=model_input.reshape(1,-1)
            for i in range(0, self.state_space_dim):
                ithmodel=getattr(self,'model'+str(i))					
                self.state=np.append(self.state, ithmodel.predict(np.array(model_input)),axis=0)

        elif self.modeltype=='nn':
            self.state=[]
            #print('model summary is:', self.model.summary())
            #print('model input after reshaping is: ', model_input)
            #print('reshape of input is: ', model_input.shape)
            model_input=self.scaler_x_set.transform([model_input])
            self.state=self.model.predict(np.array(model_input))
            self.state=self.scaler_y_set.inverse_transform(self.state)
            #print('self.state is .. :', self.state)

        self.state=np.ravel(self.state)
        
        return self.state

    def warn_limitation(self, features):
        i = 0
        num_tripped = 0
        for key, value in self.model_config['IO']['feature_name'].items():
            if features[i] > self.model_limits[key]['max'] or features[i] < self.model_limits[key]['min']:
                print('Sim should not be necessarily trusted since predicting with the feature {} outside of range it was trained on, i.e. extrapolating.'.format(key))
                num_tripped += 1
            i += 1
        return num_tripped