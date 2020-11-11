import numpy as np
from typing import Tuple, Dict
import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetRegressor

from base import BaseModel
from tune_sklearn import TuneGridSearchCV, TuneSearchCV


class MVRegressor(nn.Module):
    def __init__(
        self,
        num_units=50,
        input_dim: int = 28,
        output_dim: int = 18,
        p_dropout: float = 0.5,
        n_layers: int = 10,
    ):
        super(MVRegressor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.dense0 = nn.Linear(input_dim, num_units)
        self.dropout = nn.Dropout(p_dropout)
        self.dense1 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, self.output_dim)

    def forward(self, X, **kwargs):

        X = self.dense0(X)
        for _ in range(self.n_layers):
            X = F.relu(X)
            X = self.dropout(X)
        X = F.relu(X)
        X = self.output(X)
        return F.relu(X)


class PyTorchModel(BaseModel):
    def build_model(self, network=MVRegressor, device: str = "cpu"):

        if not all([hasattr(self, "input_dim"), hasattr(self, "output_dim")]):

            raise ValueError("Please load dataset first to obtain proper sizes")

        if device == "cpu":
            self.device = device
        else:
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")

        self.model = NeuralNetRegressor(
            network,
            device=self.device,
            module__input_dim=self.input_dim,
            module__output_dim=self.output_dim,
            max_epochs=10,
            lr=0.1,
            criterion=nn.MSELoss,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
        )

    def fit(self, X, y):

        X, y = (
            torch.tensor(X).float().to(device=self.device),
            torch.tensor(y).float().to(device=self.device),
        )
        self.model.fit(X, y)

    def predict(self, X, inverse_transform: bool = False):

        X = torch.tensor(X)
        preds = self.model.predict(X)

        if inverse_transform:
            preds = self.yscalar.inverse_transform(preds)

        return preds

    def sweep(
        self,
        params: Dict,
        X,
        y,
        search_algorithm: str = "bayesian",
        num_trials: int = 3,
    ):

        X, y = (
            torch.tensor(X).float().to(device=self.device),
            torch.tensor(y).float().to(device=self.device),
        )
        tune_search = TuneSearchCV(
            self.model,
            params,
            search_optimization=search_algorithm,
            n_trials=num_trials,
            early_stopping=True,
        )
        tune_search.fit(X, y)

        return tune_search


if __name__ == "__main__":

    pytorch_model = PyTorchModel()
    X, y = pytorch_model.load_numpy("/home/alizaidi/bonsai/repsol/data/scenario1")
    X, y = pytorch_model.scale_data(X, y)

    pytorch_model.build_model()
    # pytorch_model.fit(X, y)
    # predict_one = pytorch_model.predict(X[0])

    # tune tests
    # params = {"lr": [0.01, 0.02], "module__num_units": [10, 50]}
    # gs = TuneGridSearchCV(pytorch_model.model, params, scoring="neg_mean_squared_error")
    # gs.fit(torch.tensor(X).float(), torch.tensor(y).float())

    params = {"lr": [0.01, 0.02], "module__num_units": [10, 50]}
    pytorch_model.sweep(params=params, X=X, y=y, search_algorithm="hyperopt")
