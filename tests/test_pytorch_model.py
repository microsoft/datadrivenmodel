import os
import pytest
from torch_models import PyTorchModel

torch_model = PyTorchModel()
X, y = torch_model.load_csv(
    dataset_path="csv_data/cartpole-log.csv",
    max_rows=1000,
    augm_cols=["action_command", "config_length", "config_masspole"],
)


def test_shape():

    assert X.shape[0] == 980 == y.shape[0]
    assert X.shape[1] == torch_model.input_dim
    assert y.shape[1] == torch_model.output_dim


def test_build():

    torch_model.build_model()
    assert torch_model.scale_data == False
    assert torch_model.model is not None


def test_fit():

    torch_model.build_model()
    torch_model.fit(X, y)

