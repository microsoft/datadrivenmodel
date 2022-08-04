import pytest
from torch_models import PyTorchModel
import numpy as np
import os

torch_model = PyTorchModel()
X, y, _, _ = torch_model.load_csv(
    input_cols="state",
    output_cols="state",
    augm_cols=["action_command", "config_length", "config_masspole"],
    dataset_path="csv_data/cartpole_st1_at.csv",
    max_rows=1000,
    test_perc=0.15,
)


def test_shape():

    assert X.shape[0] == 833 == y.shape[0]
    assert X.shape[1] == torch_model.input_dim
    assert y.shape[1] == torch_model.output_dim


def test_build():

    torch_model.build_model()
    assert torch_model.scale_data == False
    assert torch_model.model is not None


def test_fit():

    torch_model.build_model()
    torch_model.fit(X, y)


def test_sweep():

    torch_model.build_model()
    params = {"lr": [0.01, 0.02], "module__num_units": [10, 50]}
    torch_model.sweep(params=params, X=X, y=y, search_algorithm="hyperopt")

    torch_model.sweep(params=params, X=X, y=y, search_algorithm="bayesian")


def test_predictor():

    torch_model.build_model()
    torch_model.fit(X, y)
    y_hat = torch_model.predict(X)

    assert y_hat.shape[0] == y.shape[0]


def test_save_model():

    torch_model.build_model()
    torch_model.fit(X, y)
    yhat = torch_model.predict(X)

    torch_model.save_model(filename="tmp.pkl")

    loaded_model = PyTorchModel()
    loaded_model.load_model(
        filename="tmp.pkl", input_dim=X.shape[1], output_dim=y.shape[1]
    )

    yhat_loaded = loaded_model.predict(X)
    np.all

    assert np.array_equal(yhat, yhat_loaded)
    os.remove("tmp.pkl")
