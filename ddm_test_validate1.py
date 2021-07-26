import logging
import datetime
import pathlib
import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List
import time

from base import BaseModel

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger("datamodeler")

import hydra
from omegaconf import DictConfig

from all_models import available_models

## Add a local simulator in a `sim` folder to validate data-driven model
## Example: Quanser from a Microsoft Bonsai
"""
├───ddm_test_validate.py
├───main.py
├───sim
│   ├───quanser
│   │   ├───sim
│   │   |    ├───qube_simulator.py
"""
# TODO: from main import TemplateSimulatorSession, env_setup

dir_path = os.path.dirname(os.path.realpath(__file__))
env_name = "DDM"
log_path = "logs"


class Simulator(BaseModel):
    def __init__(
        self,
        model,
        states: List[str],
        actions: List[str],
        configs: List[str],
        log_file: str = None,
        diff_state: bool = False,
        sim_orig=None,
    ):
        self.dd_model = model
        self.features = states + actions + configs
        self.labels = states
        self.config_keys = configs
        self.state_keys = states
        self.action_keys = actions
        self.sim_orig = (
            sim_orig()
        )  # include simulator function if comparing to simulator
        self.diff_state = diff_state
        if log_file == "enable":
            current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_file = os.path.join(
                log_path, current_time + "_" + env_name + "_log.csv"
            )
            log_file2 = os.path.join(
                log_path, current_time + "_" + "SIMDATA" + "_log.csv"
            )
            logs_directory = pathlib.Path(log_file).parent.absolute()
            if not pathlib.Path(logs_directory).exists():
                print(
                    "Directory does not exist at {0}, creating now...".format(
                        str(logs_directory)
                    )
                )
                logs_directory.mkdir(parents=True, exist_ok=True)
        else:
            log_file2 = None
        self.log_file = log_file
        self.log_file2 = log_file2

    def episode_start(self, config: Dict[str, Any] = None):
        if config:
            self.config = config
        else:
            # configs randomized here. Need to decide a single place for configs
            # range either in main.py or in simulator configs
            self.config = {j: np.random.uniform(-1, 1) for j in self.config_keys}
        if self.sim_orig:
            # Assign same state as would be used by simulator
            self.sim_orig.episode_start(config)
            _fullstate = self.sim_orig.get_state()
            # idempotent dict comprehension for sims with prefix in configurations
            self.state = {
                j: _fullstate[l]
                for j in self.state_keys
                for l in _fullstate.keys()
                if l in j
            }
        else:
            # randomized state here need to be changed to appropriate ranges of each state
            # see Table of All Data (TOAD) from Discovery Session
            self.state = {j: np.random.uniform(-0.1, 0.1) for j in self.state_keys}

    def episode_step(self, action: Dict[str, int]):

        input_list = [
            list(
                self.state.values()
            ),  # replace by self.sim_orig.state.values() for per iteration
            list(self.config.values()),
            list(action.values()),
        ]

        input_array = [item for subl in input_list for item in subl]
        X = np.array(input_array).reshape(1, -1)
        if self.diff_state:
            preds = np.array(list(self.state.values())) + self.dd_model.predict(
                X
            )  # compensating for output being delta state st+1-st
            # preds = np.array(list(simstate))+self.dd_model.predict(X) # if doing per iteration prediction of delta state st+1-st
        else:
            preds = self.dd_model.predict(X)  # absolute prediction
        self.state = dict(zip(self.features, preds.reshape(preds.shape[1]).tolist()))
        return self.state

    def get_state(self):
        return self.state

    def get_sim_state(self):
        if self.sim_orig:
            _fullstate = self.sim_orig.get_state()
            # idempotent dict comprehension for sims with prefix in configurations
            return {
                j: _fullstate[l]
                for j in self.state_keys
                for l in _fullstate.keys()
                if l in j
            }
        else:
            return self.state

    def get_init_sim_actions(self, _init_actions=None):
        # If simulator exists, get initial actions from the sim
        if self.sim_orig:
            _fullstate = self.sim_orig.get_state()
            _init_actions = {
                j: _fullstate[l]
                for j in self.action_keys
                for l in _fullstate.keys()
                if l in j
            }
        if not _init_actions:
            # If simulator is unavailable or does not have initial actions, assign
            # random actions, with range same as actuator limits. See TOAD from discovery
            _init_actions = {j: np.random.uniform(-1, 1) for j in self.action_keys}
        return _init_actions

    def test_policies(self, policy, action):
        if policy == "random":
            return {j: np.random.uniform(-1, 1) for j in self.action_keys}
        elif policy == "zeros":
            return {j: 0 for j in self.action_keys}
        else:  # coasting is default policy - no change in actions
            return action
        # TO DO: Add benchmark policy or other case specific scenarios

    def halted(self):
        # Recommend implementing, be sure to use self.get_sim_state() not self.get_state()
        pass

    def log_iterations(
        self, state, action, fname: str, episode: int = 0, iteration: int = 1
    ):
        """Log iterations during training to a CSV.

        Parameters
        ----------
        state : Dict
        action : Dict
        episode : int, optional
        iteration : int, optional
        """
        config = self.config
        data = {**state, **action, **config}
        data["episode"] = episode
        data["iteration"] = iteration
        log_df = pd.DataFrame(data, index=[0])

        if os.path.exists(fname):
            log_df.to_csv(path_or_buf=fname, mode="a", header=False, index=False)
        else:
            log_df.to_csv(path_or_buf=fname, mode="w", header=True, index=False)


def test_sim_model(
    num_episodes: int = 100,
    num_iterations: int = 250,
    log_iterations: bool = True,
    sim: Simulator = None,
):
    """Test a policy using random actions over a fixed number of episodes

    Parameters
    ----------
    num_episodes : int, optional
        number of iterations to run, by default 10
    """
    for episode in range(num_episodes):
        iteration = 0
        terminal = False
        """
        TODO: Add episode_start(config) so sim works properly and not initializing
        with unrealistic initial conditions.
        """
        sim.episode_start()
        ddm_state = sim.get_state()
        sim_state = sim.get_sim_state()
        # it is important to know initial actions for evolution of the dynamics
        # action = random_action()

        action = sim.get_init_sim_actions()
        if log_iterations:
            sim.log_iterations(ddm_state, action, sim.log_file, episode, iteration)
            if sim.sim_orig:
                sim.log_iterations(sim_state, action, sim.log_file2, episode, iteration)
        while not terminal:
            action = sim.test_policies("random", action)
            # sim iteration
            sim.episode_step(action)
            ddm_state = sim.get_state()
            if sim.sim_orig:
                sim.sim_orig.episode_step(action)
                sim_state = sim.get_sim_state()
            iteration += 1
            if log_iterations:
                sim.log_iterations(ddm_state, action, sim.log_file, episode, iteration)
                if sim.sim_orig:
                    sim.log_iterations(
                        sim_state, action, sim.log_file2, episode, iteration
                    )
            print(f"Running iteration #{iteration} for episode #{episode}")
            print(f"Observations for Sim: {sim_state}")
            print(f"Observations for Data: {ddm_state}")
            # Add additional terminal conditions if required. Here only time-out is used.
            terminal = iteration >= num_iterations or sim.halted()

    return sim


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    save_path = cfg["model"]["saver"]["filename"]
    if cfg["data"]["full_or_relative"] == "relative":
        save_path = os.path.join(dir_path, save_path)
    model_name = cfg["model"]["name"]
    states = cfg["simulator"]["states"]
    actions = cfg["simulator"]["actions"]
    configs = cfg["simulator"]["configs"]
    policy = cfg["simulator"]["policy"]
    logflag = cfg["simulator"]["logging"]
    scale_data = cfg["model"]["build_params"]["scale_data"]
    diff_state = cfg["data"]["diff_state"]

    logger.info(f"Training with a new {policy} policy")

    input_cols = cfg["data"]["inputs"]
    output_cols = cfg["data"]["outputs"]
    augmented_cols = cfg["data"]["augmented_cols"]

    input_cols = input_cols + augmented_cols

    ddModel = available_models[model_name]
    model = ddModel()

    # model.build_model(**cfg["model"]["build_params"])
    if model_name.lower() == "pytorch":
        model.load_model(
            input_dim=len(input_cols),
            output_dim=len(output_cols),
            filename=save_path,
            scale_data=scale_data,
        )
    else:
        model.load_model(filename=save_path, scale_data=scale_data)

    # Grab standardized way to interact with sim API
    sim = Simulator(model, states, actions, configs, logflag, diff_state)

    test_sim_model(1, 250, logflag, sim)

    return sim


if __name__ == "__main__":

    main()
