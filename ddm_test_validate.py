import logging
import datetime
import pathlib
import os
import time
from distutils.util import strtobool
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from azure.core.exceptions import HttpResponseError
from dotenv import load_dotenv, set_key
from microsoft_bonsai_api.simulator.client import BonsaiClient, BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import (
    SimulatorInterface,
    SimulatorSessionResponse,
    SimulatorState,
)

from base import BaseModel

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger("datamodeler")

import hydra
from omegaconf import DictConfig

from model_loader import available_models

from sim.moab.moab_main import SimulatorSession,  env_setup
from sim.moab.policies import coast, random_policy

dir_path = os.path.dirname(os.path.realpath(__file__))
env_name = "DDM"
log_path = "logs"


class Simulator(BaseModel):
    def __init__(self, model, states=List[str], actions=List[str], configs=List[str], 
        log_file: str = None,):
        self.dd_model = model
        self.features = states + actions + configs
        self.labels = states
        self.config_keys = configs
        self.state_keys = states
        self.action_keys = actions
        self.sim_physics = SimulatorSession()
        if log_file=="enable":
            current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_file = current_time + "_" + env_name + "_log.csv"
            log_file2 = current_time + "_" + "SIM" + "_log.csv"
            logs_directory = pathlib.Path(log_file).parent.absolute()
            if not pathlib.Path(logs_directory).exists():
                print(
                    "Directory does not exist at {0}, creating now...".format(
                        str(logs_directory)
                    )
                )
                logs_directory.mkdir(parents=True, exist_ok=True)
        self.log_file = os.path.join(log_path, log_file)
        self.log_file2 = os.path.join(log_path, log_file2)

    def episode_start(self, config: Dict[str, Any] = None):
        if config:
            self.config = config
        else:
            self.config = {"initial_x":np.random.uniform(-1,1),"initial_y":np.random.uniform(-1,1),"initial_vel_x":np.random.uniform(-1,1), 
                            "initial_vel_y":np.random.uniform(-1,1),"initial_roll": np.random.uniform(-1,1), "initial_pitch": np.random.uniform(-1,1)}
        # Assign same state as would be used by simulator
        self.sim_physics.episode_start(config)
        _fullstate = self.sim_physics.get_state()
        # idempotent dict comprehension for sims with prefix in configurations
        self.state = {j:_fullstate[l] for j in self.state_keys for l in _fullstate.keys() if l in j}
        

    def episode_step(self, action: Dict[str, int]):

        input_list = [
            list(self.state.values()),
            list(self.config.values()),
            list(action.values()),
        ]

        input_array = [item for subl in input_list for item in subl]
        X = np.array(input_array).reshape(1, -1)
        preds = self.dd_model.predict(X)
        self.state = dict(zip(self.features, preds.reshape(preds.shape[1]).tolist()))
        return self.state

    def get_state(self):
        return self.state

    def get_sim_state(self):
        _fullstate = self.sim_physics.get_state()
        # idempotent dict comprehension for sims with prefix in configurations
        return {j:_fullstate[l] for j in self.state_keys for l in _fullstate.keys() if l in j}

    def halted(self):
        pass

    def log_iterations(self, state, action, fname: str, episode: int = 0, iteration: int = 1):
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
        print(fname)


        if os.path.exists(fname):
            log_df.to_csv(
                path_or_buf=fname, mode="a", header=False, index=False
            )
        else:
            log_df.to_csv(path_or_buf=fname, mode="w", header=True, index=False)

def test_random_policy(
    num_episodes: int = 100,
    num_iterations: int = 250,
    log_iterations: bool= True,
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
        sim.episode_start()
        ddm_state = sim.get_state()
        sim_state = sim.get_sim_state()
        # it is important to know initial actions for evolution of the dynamics
        # action = random_action()
        action = {"input_roll":ddm_state["state_roll"],"input_pitch":ddm_state["state_pitch"]}
        if log_iterations:
            sim.log_iterations(ddm_state, action, sim.log_file, episode, iteration)
            sim.log_iterations(sim_state, action, sim.log_file2, episode, iteration)
        while not terminal:
            action = random_policy(ddm_state)
            # sim iteration
            sim.episode_step(action)
            sim.sim_physics.episode_step(action)
            ddm_state = sim.get_state()
            sim_state = sim.get_sim_state()
            iteration += 1
            if log_iterations:
                sim.log_iterations(ddm_state, action, sim.log_file, episode, iteration)
                sim.log_iterations(sim_state, action, sim.log_file2, episode, iteration)
            print(f"Running iteration #{iteration} for episode #{episode}")
            print(f"Observations for Sim: {sim_state}")
            print(f"Observations for Data: {ddm_state}")
            terminal = iteration >= num_iterations

    return sim
    


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    save_path = cfg["model"]["saver"][0]["filename"]
    if cfg["data"]["full_or_relative"] == "relative":
        save_path = os.path.join(dir_path, save_path)
    model_name = cfg["model"]["name"]
    states = cfg["simulator"]["states"]
    actions = cfg["simulator"]["actions"]
    configs = cfg["simulator"]["configs"]
    policy = cfg["simulator"]["policy"]
    logflag = cfg["simulator"]["logging"]
    scale_data = cfg["model"]["build_params"][7]["scale_data"]

    logger.info(f"Training with a new {policy} policy")

    ddModel = available_models[model_name]
    model = ddModel()

    model.build_model(model_type=model_name)
    model.load_model(filename=save_path, scale_data=scale_data)

    # Grab standardized way to interact with sim API
    sim = Simulator(model, states, actions, configs, logflag)
    
    test_random_policy(100, 250,logflag=="enable", sim)
    
    return sim

        

if __name__ == "__main__":

    main()