# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras import optimizers
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


class env_nn_modeler:
    def __init__(self, state_space_dim=None, action_space_dim=None):
        self.input_dim = int(state_space_dim + action_space_dim)
        self.output_dim = int(state_space_dim)

    def create_model(self, config):
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

    def train_nn_model(self, x_train, y_train, epochs=10, batch_size=32):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        return self.model

    def evaluate_nn_model(self, x_test, y_test, batch_size=32):
        self.score = self.model.evaluate(x_test, y_test, batch_size=batch_size)
        return self.score

    def predict_nn_model(self, x_sample):
        self.prediction = self.model.predict(x_sample, batch_size=32)
        return self.prediction


class env_lstm_modeler:
    def __init__(self, state_space_dim=None, action_space_dim=None):
        self.input_dim = int(state_space_dim + action_space_dim)
        self.output_dim = int(state_space_dim)

    def create_model(self, config):
        num_hidden_layers = config["num_hidden_layer"]
        learning_rate = config["lr"]
        self.model = Sequential()
        self.model.add(
            LSTM(
                units=config["num_lstm_units"],
                activation=config["activation"],
                input_shape=(config["markovian_order"], self.input_dim),
            )
        )
        for i in range(0, num_hidden_layers):
            self.model.add(Dense(config["n_neuron"], activation=config["activation"]))
        self.model.add(Dropout(rate=config["dropout"]))
        self.model.add(
            Dense(self.output_dim, activation="tanh", kernel_initializer="normal")
        )
        opt = optimizers.Adam(lr=learning_rate, decay=config["decay"])
        self.model.compile(
            optimizer="adam", loss="mean_squared_error", metrics=["accuracy"]
        )
        self.model.summary()
        return self.model

    def train_nn_model(self, x_train, y_train, epochs=10, batch_size=32):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        return self.model

    def evaluate_nn_model(self, x_test, y_test, batch_size=32):
        self.score = self.model.evaluate(x_test, y_test, batch_size=batch_size)
        return self.score

    def predict_nn_model(self, x_sample):
        self.prediction = self.model.predict(x_sample, batch_size=32)
        return self.prediction


class env_gb_modeler:
    def __init__(self, state_space_dim=None, action_space_dim=None):
        self.input_dim = int(state_space_dim + action_space_dim)
        self.output_dim = int(state_space_dim)

    def create_gb_model(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth
        )
        return self.model

    def train_gb_model(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return self.model

    def evaluate_gb_model(self, x_test, y_test):
        self.score = self.model.score(x_test, y_test)
        return self.score

    def predict_gb_model(self, x_sample):
        self.prediction = self.model.predict(x_sample)
        return self.prediction


class env_poly_modeler:
    def __init__(self, state_space_dim=None, action_space_dim=None):
        self.input_dim = int(state_space_dim + action_space_dim)
        self.output_dim = int(state_space_dim)

    def create_poly_model(self, degree=3):
        self.poly = PolynomialFeatures(degree=degree)
        self.model = linear_model.LinearRegression()
        return self.model

    def train_poly_model(self, x_train, y_train):
        x_train_transform = self.poly.fit_transform(x_train)

        self.model.fit(x_train_transform, y_train)
        return self.model

    def evaluate_poly_model(self, x_test, y_test):
        x_test_transform = self.poly.fit_transform(x_test)
        self.score = self.model.score(x_test_transform, y_test)
        return self.score

    def predict_poly_model(self, x_sample):
        x_sample_transform = self.poly.fit_transform(x_sample)
        print(x_sample_transform)
        print(x_sample_transform.shape)
        self.prediction = self.model.predict(x_sample_transform)
        return self.prediction


def create_nn_model_wrapper(
    activation="relu",
    state_space_dim=None,
    action_space_dim=None,
    dropout_rate=0.1,
    num_hidden_layers=2,
    num_neurons=8,
    learning_rate=0.001,
    decay=10 ** -5,
):

    # input_dim=int(state_space_dim+action_space_dim)
    # output_dim=int(state_space_dim)
    input_dim = state_space_dim + action_space_dim
    output_dim = state_space_dim
    num_hidden_layers = num_hidden_layers
    num_neurons = num_neurons
    learning_rate = learning_rate
    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, activation=activation))
    for i in range(0, num_hidden_layers):
        model.add(Dense(num_neurons, activation=activation))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(output_dim, activation="tanh", kernel_initializer="normal"))
    opt = optimizers.Adam(lr=learning_rate, decay=decay)
    # opt=optimizers.RMSprop(lr=learning_rate)
    # BUG: why accuracy here as the metric? doesn't make sense for MV regression
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    model.summary()
    return model


def create_lstm_model_wrapper(
    activation="relu",
    state_space_dim=None,
    action_space_dim=None,
    dropout_rate=0.1,
    num_hidden_layers=2,
    num_neurons=8,
    learning_rate=0.001,
    num_lstm_units=10,
    markovian_order=2,
    decay=10 ** -9,
):

    input_dim = state_space_dim + action_space_dim
    output_dim = state_space_dim
    print("input and output dimensions are:", input_dim, output_dim)
    num_hidden_layers = num_hidden_layers
    num_neurons = num_neurons
    learning_rate = learning_rate
    model = Sequential()
    model.add(
        LSTM(
            units=num_lstm_units,
            activation=activation,
            input_shape=(markovian_order, input_dim),
        )
    )
    for i in range(0, num_hidden_layers):
        model.add(Dense(num_neurons, activation=activation))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(output_dim, activation="tanh", kernel_initializer="normal"))
    opt = optimizers.Adam(lr=learning_rate, decay=decay)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    model.summary()
    return model
