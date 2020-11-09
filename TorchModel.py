import numpy as np
import os
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetRegressor

from BaseTrainer import BaseModel


def load_data(dataset_path: str) -> Tuple[np.array, np.array]:

    X = np.load(os.path.join(dataset_path, "x_set.npy"))
    y = np.load(os.path.join(dataset_path, "y_set.npy"))
    return X, y


class MVRegressor(nn.Module):
    def __init__(
        self,
        num_units=50,
        input_dim: int = 28,
        output_dim: int = 18,
        n_layers: int = 10,
    ):
        super(MVRegressor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.dense1 = nn.Linear(input_dim, num_units)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(num_units, self.output_dim)

    def forward(self, X, **kwargs):

        for _ in range(self.n_layers):
            X = self.dense1(X)
            X = F.relu(X)
            X = self.dropout(X)
        X = F.relu(X)
        X = self.output(X)
        return F.relu(X)


class PyTorchModel(BaseModel):
    def build_model(self):

        if not all([hasattr(self, "input_dim"), hasattr(self, "output_dim")]):

            raise ValueError("Please load dataset first to obtain proper sizes")

        self.model = NeuralNetRegressor(
            MVRegressor,
            module__input_dim=self.input_dim,
            module__output_dim=self.output_dim,
            max_epochs=10,
            lr=0.1,
            criterion=nn.MSELoss,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
        )

    def fit(self, X, y):

        X, y = torch.tensor(X).float(), torch.tensor(y).float()
        self.model.fit(X, y)

    def predict(self, X):

        X = torch.tensor(X)
        return self.model.predict(X)


if __name__ == "__main__":

    pytorch_model = PyTorchModel()
    X, y = pytorch_model.load_numpy("/home/alizaidi/bonsai/repsol/data/scenario1")

    pytorch_model.build_model()
    pytorch_model.fit(X, y)
    predict_one = pytorch_model.predict(X[0])

    from tune_sklearn import TuneGridSearchCV

    # tune tests
    params = {"lr": [0.01, 0.02], "module__num_units": [10, 50]}
    gs = TuneGridSearchCV(pytorch_model.model, params, scoring="neg_mean_squared_error")
