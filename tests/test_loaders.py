import os
import pytest
import pandas as pd
from loaders import CsvReader
from base import BaseModel

data_dir = "csv_data"
df = pd.read_csv(os.path.join(data_dir, "cartpole_st1_at.csv"), nrows=1000)
df2 = pd.read_csv(os.path.join(data_dir, "cartpole_at_st.csv"), nrows=1000)
features = [
    "state_x_position",
    "state_x_velocity",
    "state_angle_position",
    "state_angle_velocity",
    "action_command",
    "config_length",
    "config_masspole",
]

labels = [
    "state_x_position",
    "state_x_velocity",
    "state_angle_position",
    "state_angle_velocity",
]


@pytest.fixture
def csv_reader():
    csv_reader = CsvReader()
    return csv_reader


def test_cartpole_at_st1(csv_reader):

    cp_df = csv_reader.read(
        df, iteration_order=-1, feature_cols=features, label_cols=labels
    )
    assert cp_df.shape[0] == 980
    assert cp_df.shape[1] == 13
    assert (
        cp_df["state_x_position"].values[0] == cp_df["prev_state_x_position"].values[1]
    )


def test_cartpole_at_st(csv_reader):

    cp2_df = csv_reader.read(
        df2, feature_cols=features, label_cols=labels, iteration_order=1
    )

    assert cp2_df.shape[0] == 980
    assert cp2_df.shape[1] == 13
    assert (
        cp2_df["state_x_position"].values[0]
        == cp2_df["prev_state_x_position"].values[1]
    )


def test_base_reader():

    base_model = BaseModel()
    X, y, _, _ = base_model.load_csv(
        dataset_path=os.path.join(data_dir, "cartpole_st1_at.csv"),
        max_rows=1000,
        augm_cols=["action_command", "config_length", "config_masspole"],
        test_perc=0.15,
    )

    assert X.shape[0] == 833 == y.shape[0]
    assert X.shape[1] == 7
    assert y.shape[1] == 4


def test_diff_names():

    base_model = BaseModel()
    X, y, _, _ = base_model.load_csv(
        dataset_path=os.path.join(data_dir, "off_names.csv"),
        input_cols=[
            "x_position",
            "x_velocity",
            "angle_position",
            "angle_velocity",
        ],
        output_cols=["angle_position", "angle_velocity"],
        augm_cols=["command", "length", "masspole"],
        max_rows=1000,
        test_perc=0.15,
    )

    assert X.shape[0] == 833 == y.shape[0]
