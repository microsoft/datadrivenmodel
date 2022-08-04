import os
import pytest
import pandas as pd
import numpy as np
import pathlib
from dataclass import DataClass
from ddm_trainer import main

# from yaml import aa
from hydra.experimental import compose, initialize


# Initialize outside, to ensure only happens once (and avoid errors)
initialize(config_path="../conf", job_name="model_validation")


@pytest.mark.skip(reason="Too slow")
def test_sweeping_lightgbm():
    cfg_obj = compose(
        config_name="config",
        overrides=["data=cartpole_st1_at", "model=lightgbm", "simulator.policy=random"],
    )
    main(cfg_obj)


@pytest.mark.skip(reason="Too slow")
def test_sweeping_svr():
    cfg_obj = compose(
        config_name="config",
        overrides=["data=cartpole_st1_at", "model=SVR", "simulator.policy=random"],
    )
    main(cfg_obj)


# def test_sweeping_torch():
#    cfg_obj = compose(config_name="config", overrides=["data=cartpole-updated", "model=torch"])
#    main(cfg_obj)


@pytest.mark.skip(reason="Too slow")
def test_sweeping_xgboost():
    cfg_obj = compose(
        config_name="config",
        overrides=["data=cartpole_st1_at", "model=xgboost", "simulator.policy=random"],
    )
    main(cfg_obj)


if __name__ == "__main__":
    pass
