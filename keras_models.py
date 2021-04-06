from base import BaseModel

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras import optimizers

from sklearn.metrics import mean_squared_error
from tune_sklearn import TuneGridSearchCV, TuneSearchCV


class KerasNeuralNetModel(BaseModel):
    def build_model(self, config=None):

        num_hidden_layers = config["num_layers"]
        optimizer = config["optimizer"]
        learning_rate = config["lr"]
        self.model = Sequential()
        self.model.add(
            Dense(
                config["num_units"],
                input_dim=self.input_dim,
                activation=config["activation"],
                kernel_initializer="glorot_normal",
            )
        )
        for i in range(0, num_hidden_layers):
            self.model.add(
                Dense(
                    config["num_units"],
                    activation="relu",
                    kernel_initializer="glorot_normal",
                )
            )
        self.model.add(
            Dense(
                self.output_dim,
                activation=config["activation"],
                kernel_initializer="glorot_normal",
            )
        )
        # opt=optimizers.RMSprop(lr=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss="mean_squared_error",
            metrics=["mean_squared_error"],
        )
        print(self.model.summary())
        return self.model

    def sweep(self, X, y):

        optimizers = ["rmsprop", "adam"]
        kernel_initializer = ["glorot_uniform", "normal"]
        epochs = [5, 10]
        param_grid = dict(
            optimizer=optimizers, nb_epoch=epochs, kernel_initializer=kernel_initializer
        )
        grid = TuneGridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
        )
        grid_result = grid.fit(X, y)

        return grid_result


if __name__ == "__main__":

    config = {
        "num_layers": 10,
        "num_units": 50,
        "lr": 0.01,
        "activation": "relu",
        "optimizer": "adam",
    }
    keras_model = KerasNeuralNetModel()
    keras_model.build_model(config=config)
    # keras_model.fit(X, y)

    # keras_model.sweep(X, y)
