from typing import Dict
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetRegressor
from skorch.callbacks import LRScheduler
from torch.optim.lr_scheduler import CyclicLR

from base import BaseModel


class MVRegressor(nn.Module):
    def __init__(
        self,
        num_units: int = 50,
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
    def build_model(
        self,
        network=MVRegressor,
        device: str = "cpu",
        scale_data: bool = False,
        num_layers: int = 10,
        num_units: int = 50,
        dropout: float = 0.5,
        num_epochs: int = 10,
        batch_size: int = 128,
    ):

        self.scale_data = scale_data
        self.num_layers = num_layers
        self.num_units = num_units
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.batch_size = batch_size

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
            module__n_layers=self.num_layers,
            module__num_units=self.num_units,
            module__p_dropout=self.dropout,
            max_epochs=self.num_epochs,
            criterion=nn.MSELoss,
            batch_size=self.batch_size,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            callbacks=[
                (
                    "lr_scheduler",
                    LRScheduler(
                        policy=CyclicLR, base_lr=0.001, max_lr=0.01, step_every="batch"
                    ),
                ),
            ],
        )

    def fit(self, X, y, **fit_params):

        if self.scale_data:
            X, y = self.scalar(X, y)

        X, y = (
            torch.tensor(X).float().to(device=self.device),
            torch.tensor(y).float().to(device=self.device),
        )
        self.model.fit(X, y, **fit_params)

    def load_model(
        self,
        input_dim: str,
        output_dim: str,
        filename: str,
        scale_data: bool = False,
    ):

        self.scale_data = scale_data
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.build_model(scale_data=scale_data)
        self.model = pickle.load(open(filename, "rb"))

    def predict(self, X):

        if self.scale_data:
            X = self.xscalar.transform(X)
        X = torch.tensor(X).float().to(device=self.device)
        preds = self.model.predict(X)

        if self.scale_data:
            preds = self.yscalar.inverse_transform(preds)

        return preds

    def sweep(
        self,
        params: Dict,
        X,
        y,
        search_algorithm: str = "bayesian",
        num_trials: int = 3,
        scoring_func: str = "r2",
    ):

        from tune_sklearn import TuneGridSearchCV, TuneSearchCV

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
            scoring=scoring_func,
        )
        tune_search.fit(X, y)

        return tune_search


if __name__ == "__main__":

    pytorch_model = PyTorchModel()
    X, y = pytorch_model.load_csv(
        dataset_path="csv_data/cartpole-log.csv",
        max_rows=1000,
        augm_cols=["action_command", "config_length", "config_masspole"],
    )
    pytorch_model.build_model()
    pytorch_model.fit(X, y)
    # tune tests
    # params = {"lr": [0.01, 0.02], "module__num_units": [10, 50]}
    # gs = TuneGridSearchCV(pytorch_model.model, params, scoring="neg_mean_squared_error")
    # gs.fit(torch.tensor(X).float(), torch.tensor(y).float())

    # params = {"lr": [0.01, 0.02], "module__num_units": [10, 50]}
    # pytorch_model.sweep(params=params, X=X, y=y, search_algorithm="hyperopt")
