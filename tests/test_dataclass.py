import os
import pytest
import pandas as pd
import numpy as np
import pathlib
from dataclass import DataClass

data_dir = "csv_data"
csv_dir = os.path.join(data_dir, "cartpole_st1_at.csv")
df = pd.read_csv(csv_dir, nrows=1000)
df2 = pd.read_csv(csv_dir, nrows=1000)
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
def dataclass_obj():
    dataclass_obj = DataClass()
    return dataclass_obj


def test_cartpole_at_st1(dataclass_obj):

    cp_df = dataclass_obj.read(
        df, iteration_order=-1, feature_cols=features, label_cols=labels
    )
    assert cp_df.shape[0] == 980
    assert cp_df.shape[1] == 13
    assert (
        cp_df["state_x_position"].values[0] == cp_df["prev_state_x_position"].values[1]
    )


def test_cartpole_at_st(dataclass_obj):

    cp2_df = dataclass_obj.read(
        df2, feature_cols=features, label_cols=labels, iteration_order=1
    )

    assert cp2_df.shape[0] == 980
    assert cp2_df.shape[1] == 13

    # TODO: Review. We are matching the current state value to the next value of "PREV".
    #      > This doesn't seem right at first.
    assert (
        cp2_df["state_x_position"].values[0]
        == cp2_df["prev_state_x_position"].values[1]
    )


def test_diff_labels_1(dataclass_obj):
    # testing diff labels:
    # - iteration order = -1

    X, y, _, _ = dataclass_obj.load_csv(
        dataset_path=csv_dir,
        max_rows=1000,
        augm_cols=["action_command", "config_length", "config_masspole"],
        diff_state=True,
        iteration_order=-1,
    )

    X_2, y_2 = dataclass_obj.get_train_set()

    assert (
        X.shape[0] == 833 == y.shape[0]
    ), f"X.shape[0] ({X.shape[0]}) -- y.shape[0] ({y.shape[0]})"
    assert (
        X.shape[1] == 7 == dataclass_obj.input_dim
    ), f"X.shape[1] ({X.shape[1]}) -- input_dim ({dataclass_obj.input_dim})"
    assert (
        y.shape[1] == 4 == dataclass_obj.output_dim
    ), f"y.shape[1] ({y.shape[1]}) -- output_dim ({dataclass_obj.output_dim})"

    assert (
        X_2.shape[0] == 833 == y_2.shape[0]
    ), f"X_2.shape[0] ({X_2.shape[0]}) -- y_2.shape[0] ({y_2.shape[0]})"
    assert (
        X_2.shape[1] == 7 == dataclass_obj.input_dim
    ), f"X_2.shape[1] ({X_2.shape[1]}) -- input_dim ({dataclass_obj.input_dim})"
    assert (
        y_2.shape[1] == 4 == dataclass_obj.output_dim
    ), f"y_2.shape[1] ({y_2.shape[1]}) -- output_dim ({dataclass_obj.output_dim})"

    return


def test_diff_labels_2(dataclass_obj):
    # testing diff labels:
    # - iteration order = 1

    X, y, _, _ = dataclass_obj.load_csv(
        dataset_path=csv_dir,
        max_rows=1000,
        augm_cols=["action_command", "config_length", "config_masspole"],
        diff_state=True,
        iteration_order=1,
    )

    assert (
        X.shape[0] == 833 == y.shape[0]
    ), f"X.shape[0] ({X.shape[0]}) -- y.shape[0] ({y.shape[0]})"
    assert (
        X.shape[1] == 7 == dataclass_obj.input_dim
    ), f"X.shape[1] ({X.shape[1]}) -- input_dim ({dataclass_obj.input_dim})"
    assert (
        y.shape[1] == 4 == dataclass_obj.output_dim
    ), f"y.shape[1] ({y.shape[1]}) -- output_dim ({dataclass_obj.output_dim})"

    return


def test_concatenate_states_1(dataclass_obj):
    # testing concatenation:
    # - 3 concat states
    # - zero padding disabled
    # - iteration order = -1

    X, y, _, _ = dataclass_obj.load_csv(
        dataset_path=csv_dir,
        max_rows=1000,
        augm_cols=["action_command", "config_length", "config_masspole"],
        concatenated_steps=3,
        concatenated_zero_padding=False,
        iteration_order=-1,
    )

    ## 833 training rows - 24 dropped rows --> 799 rows
    assert (
        X.shape[0] == 799 == y.shape[0]
    ), f"X.shape[0] ({X.shape[0]}) -- y.shape[0] ({y.shape[0]})"
    assert (
        X.shape[1] == 21 == dataclass_obj.input_dim
    ), f"X.shape[1] ({X.shape[1]}) -- input_dim ({dataclass_obj.input_dim})"
    assert (
        y.shape[1] == 4 == dataclass_obj.output_dim
    ), f"y.shape[1] ({y.shape[1]}) -- output_dim ({dataclass_obj.output_dim})"

    return


def test_concatenate_states_2(dataclass_obj):
    # testing concatenation:
    # - 4 concat states
    # - zero padding enabled
    # - iteration order = 1

    X, y, _, _ = dataclass_obj.load_csv(
        dataset_path=csv_dir,
        max_rows=1000,
        augm_cols=["action_command", "config_length", "config_masspole"],
        concatenated_steps=4,
        concatenated_zero_padding=True,
        iteration_order=1,
    )

    assert (
        X.shape[0] == 833 == y.shape[0]
    ), f"X.shape[0] ({X.shape[0]}) -- y.shape[0] ({y.shape[0]})"
    assert (
        X.shape[1] == 28 == dataclass_obj.input_dim
    ), f"X.shape[1] ({X.shape[1]}) -- input_dim ({dataclass_obj.input_dim})"
    assert (
        y.shape[1] == 4 == dataclass_obj.output_dim
    ), f"y.shape[1] ({y.shape[1]}) -- output_dim ({dataclass_obj.output_dim})"

    return


def test_concatenate_states_3(dataclass_obj):
    # testing loading 2 csv's consecutively on same DataClass object

    X, y, _, _ = dataclass_obj.load_csv(
        dataset_path=csv_dir,
        max_rows=1000,
        augm_cols=["action_command", "config_length", "config_masspole"],
        concatenated_steps=3,
        concatenated_zero_padding=False,
        iteration_order=-1,
    )

    assert (
        X.shape[0] == 799 == y.shape[0]
    ), f"X.shape[0] ({X.shape[0]}) -- y.shape[0] ({y.shape[0]})"
    assert (
        X.shape[1] == 21 == dataclass_obj.input_dim
    ), f"X.shape[1] ({X.shape[1]}) -- input_dim ({dataclass_obj.input_dim})"
    assert (
        y.shape[1] == 4 == dataclass_obj.output_dim
    ), f"y.shape[1] ({y.shape[1]}) -- output_dim ({dataclass_obj.output_dim})"

    X, y, _, _ = dataclass_obj.load_csv(
        dataset_path=csv_dir,
        max_rows=1000,
        augm_cols=["action_command", "config_length", "config_masspole"],
        concatenated_steps=4,
        concatenated_zero_padding=True,
        iteration_order=1,
    )

    assert (
        X.shape[0] == 833 == y.shape[0]
    ), f"X.shape[0] ({X.shape[0]}) -- y.shape[0] ({y.shape[0]})"
    assert (
        X.shape[1] == 28 == dataclass_obj.input_dim
    ), f"X.shape[1] ({X.shape[1]}) -- input_dim ({dataclass_obj.input_dim})"
    assert (
        y.shape[1] == 4 == dataclass_obj.output_dim
    ), f"y.shape[1] ({y.shape[1]}) -- output_dim ({dataclass_obj.output_dim})"

    return
