from BaseTrainer import BaseModel

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras import optimizers

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor

from sklearn import linear_model

import xgboost as xgb

class KerasNeuralNetModel(BaseModel):
    def build_model(self, config=None):

        num_hidden_layers = config["n_layer"]
        learning_rate = config["lr"]
        self.model = Sequential()
        self.model.add(
            Dense(
                config["n_neuron"],
                input_dim=self.input_dim,
                activation=config["activation"],
                kernel_initializer="glorot_normal",
            )
        )
        for i in range(0, num_hidden_layers):
            self.model.add(
                Dense(
                    config["n_neuron"],
                    activation="relu",
                    kernel_initializer="glorot_normal",
                )
            )
        self.model.add(
            Dense(
                self.output_dim, activation="tanh", kernel_initializer="glorot_normal"
            )
        )
        opt = optimizers.Adam(lr=learning_rate, decay=config["decay"])
        # opt=optimizers.RMSprop(lr=learning_rate)
        self.model.compile(
            optimizer="adam", loss="mean_squared_error", metrics=["accuracy"]
        )
        self.model.summary()
        return self.model