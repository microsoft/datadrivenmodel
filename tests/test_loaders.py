import os
from data_loader import csv_reader, data_dir


def test_cartpole_at_st1():

    cp_df = csv_reader(csv_reader(os.path.join(data_dir, "cartpole-log.csv")))
    assert cp_df.shape[0] == 490000
    assert cp_df.shape[1] == 16

