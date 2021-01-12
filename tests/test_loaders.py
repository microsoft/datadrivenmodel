import os
import pytest
from loaders import CsvReader
from base import BaseModel

data_dir = "csv_data"


@pytest.fixture
def csv_reader():
    csv_reader = CsvReader()
    return csv_reader


def test_cartpole_at_st1(csv_reader):

    cp_df = csv_reader.read(
        os.path.join(data_dir, "cartpole-log.csv"), max_rows=1000, iteration_order=-1
    )
    assert cp_df.shape[0] == 980
    assert cp_df.shape[1] == 13
    assert (
        cp_df["state_x_position"].values[0] == cp_df["prev_state_x_position"].values[1]
    )


def test_cartpole_at_st(csv_reader):

    cp2_df = csv_reader.read(
        os.path.join(data_dir, "cartpole_at_st.csv"), iteration_order=1, max_rows=1000
    )

    assert cp2_df.shape[0] == 980
    assert cp2_df.shape[1] == 13
    assert (
        cp2_df["state_x_position"].values[1]
        == cp2_df["next_state_x_position"].values[0]
    )


def test_base_reader():

    base_model = BaseModel()
    X, y = base_model.load_csv(
        dataset_path=os.path.join(data_dir, "cartpole-log.csv"),
        max_rows=1000,
        augm_cols=["action_command", "config_length", "config_masspole"],
    )

    assert X.shape[0] == 980 == y.shape[0]
    assert X.shape[1] == 7
    assert y.shape[1] == 4


def test_diff_names():

    base_model = BaseModel()
    X, y = base_model.load_csv(
        dataset_path=os.path.join(data_dir, "off_names.csv"), max_rows=1000
    )

    assert X.shape[0] == 980 == y.shape[0]
