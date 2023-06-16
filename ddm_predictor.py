import logging
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Union
from omegaconf import ListConfig
from functools import partial

import pandas as pd
from policies import random_policy, brain_policy
from signal_builder import SignalBuilder

import numpy as np

# see reason below for why commented out (UPDATE #comment-out-azure-cli)
# from azure.core.exceptions import HttpResponseError
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
for name in logging.Logger.manager.loggerDict.keys():
    if "azure" in name:
        logging.getLogger(name).setLevel(logging.WARNING)
        logging.propagate = True
logger = logging.getLogger("datamodeler")

import hydra
from omegaconf import DictConfig

dir_path = os.path.dirname(os.path.realpath(__file__))
env_name = "DDM"

from ray.rllib.env import BaseEnv
import gymnasium as gym
from gymnasium.spaces import Box


def type_conversion(obj, type, minimum, maximum):
    if type == "str":
        return str(obj)
    elif type == "int":
        if obj <= minimum:
            return int(minimum)
        elif obj >= maximum:
            return int(maximum)
        else:
            return int(obj)
    elif type == "float":
        if obj <= minimum:
            return float(minimum)
        elif obj >= maximum:
            return float(maximum)
        else:
            return float(obj)
    elif type == "bool":
        return obj


# helper function that return None if element is not present in config
def hydra_read_config_var(cfg: DictConfig, level: str, key_name: str):
    """Reads the config file and returns the config as a dictionary"""

    return cfg[level][key_name] if key_name in cfg[level] else None


class Simulator(BaseModel):
    def __init__(
        self,
        model,
        states: List[str],
        actions: List[str],
        configs: List[str],
        inputs: List[str],
        outputs: Union[List[str], Dict[str, str]],
        episode_inits: Dict[str, float],
        initial_states: Dict[str, float],
        signal_builder: Dict[str, float],
        diff_state: bool = False,
        lagged_inputs: int = 1,
        lagged_padding: bool = False,
        concatenate_var_length: Optional[Dict[str, int]] = None,
        prep_pipeline: Optional[Callable] = None,
        iteration_col: Optional[str] = None,
        exogeneous_variables: Optional[List[str]] = None,
        exogeneous_save_path: Optional[str] = None,
        initial_values_save_path: Optional[str] = None,
    ):
        self.model = model
        # self.features = states + configs + actions
        # self.labels = states
        self.features = inputs
        if type(outputs) == ListConfig:
            outputs = list(outputs)
            self.label_types = None
        elif type(outputs) == DictConfig:
            output_types = outputs
            outputs = list(outputs.keys())
            self.label_types = output_types

        # if you're using exogeneous variables these will be looked up
        # from a saved dataset and appended during episode_step
        if exogeneous_variables and exogeneous_save_path:
            if os.path.dirname(exogeneous_save_path) == "":
                exogeneous_save_path = os.path.join(dir_path, exogeneous_save_path)
            if not os.path.exists(exogeneous_save_path):
                raise ValueError(
                    f"Exogeneous variables not found at {exogeneous_save_path}"
                )
            logger.info(f"Reading exogeneous variables from {exogeneous_save_path}")
            exogeneous_vars_df = pd.read_csv(exogeneous_save_path)
            self.exogeneous_variables = exogeneous_variables
            self.exog_df = exogeneous_vars_df

        if initial_values_save_path:
            if os.path.dirname(initial_values_save_path) == "":
                initial_values_save_path = os.path.join(
                    dir_path, initial_values_save_path
                )
            if not os.path.exists(initial_values_save_path):
                raise ValueError(
                    f"Initial values not found at {initial_values_save_path}"
                )
            logger.info(f"Reading initial values from {initial_values_save_path}")
            initial_values_df = pd.read_csv(initial_values_save_path)
            self.initial_values_df = initial_values_df

        self.labels = outputs
        self.config_keys = configs
        self.episode_inits = episode_inits
        self.state_keys = states
        self.action_keys = actions
        self.signal_builder = signal_builder
        self.diff_state = diff_state
        self.lagged_inputs = lagged_inputs
        self.lagged_padding = lagged_padding
        self.concatenate_var_length = concatenate_var_length
        self.prep_pipeline = prep_pipeline
        self.iteration_col = iteration_col

        if self.concatenate_var_length:
            logger.info(f"Using variable length lags: {self.concatenate_var_length}")
            self.lagged_feature_cols = [
                feat + f"_{i}"
                for feat in list(self.concatenate_var_length.keys())
                for i in range(1, self.concatenate_var_length[feat] + 1)
            ]
            self.non_lagged_feature_cols = list(
                set(self.features) - set(list(self.concatenate_var_length.keys()))
            )
            # need to verify order here
            # this matches dataclass when concatenating inputs
            self.features = self.non_lagged_feature_cols + self.lagged_feature_cols
        elif self.lagged_inputs > 1:
            logger.info(f"Using {self.lagged_inputs} lagged inputs as features")
            self.lagged_feature_cols = [
                feat + f"_{i}"
                for i in range(1, self.lagged_inputs + 1)
                for feat in self.features
            ]
            self.features = self.lagged_feature_cols
        else:
            self.lagged_feature_cols = []

        # create a dictionary containing initial_states
        # with some initial values
        # these should be coming from the simulator.yaml
        # the initial values aren't important
        # these will be updated in self.episode_start

        # create a mapper that maps config values to
        # initial state values
        # these will be used when mapping scenario keys
        # to self.initial_states values during episode_start
        initial_states_mapper = {}
        if type(list(initial_states.values())[0]) == DictConfig:
            self.initial_states = {k: v["min"] for k, v in initial_states.items()}
            for k, v in initial_states.items():
                initial_states_mapper[v["inkling_name"]] = k
        else:
            self.initial_states = initial_states
        self.initial_states_mapper = initial_states_mapper

        logger.info(f"DDM features: {self.features}")
        logger.info(f"DDM outputs: {self.labels}")

    def episode_start(self, config: Optional[Dict[str, Any]] = None):
        """Initialize DDM. This could include initializations of configs
        as well as initial values for states.

        Parameters
        ----------
        config : Dict[str, Any], optional
            episode initializations, by default None
        """

        self.iteration_counter = 0

        # if you are using both initial values and exogeneous variables, then
        # make sure to sample a single episode from each and play it through
        if hasattr(self, "initial_values_df"):
            # if self.initial_values_df is not None:
            initial_values_episode = (
                self.initial_values_df["episode"].sample(1).values[0]
            )
            initial_values_data = self.initial_values_df[
                self.initial_values_df["episode"] == initial_values_episode
            ]
            for i in list(self.initial_states.keys()):
                # terminals are not assumed to be in the lookup dataset
                # however, we will need to terminate the episdoe when we reach
                # the end of the dataset so we need a terminal variable in the MDP
                if i == "terminal":
                    self.initial_states[i] = False
                else:
                    self.initial_states[i] = initial_values_data[i].values[0]

        # if using exogeneous variables
        # sample from exog df and play it through the episode
        if hasattr(self, "exog_df"):
            if hasattr(self, "initial_values_df"):
                logger.info(f"Using sampled episode from initial values dataset")
                exog_episode = initial_values_episode
            else:
                exog_episode = self.exog_df["episode"].sample(1).values[0]
            exog_data = self.exog_df[self.exog_df["episode"] == exog_episode]
            self.exog_ep = exog_data
            for i in self.exogeneous_variables:
                self.initial_states[i] = self.exog_ep[i].values.tolist()[0]

        # initialize states based on simulator.yaml
        # we have defined the initial dict in our
        # constructor
        initial_state = self.initial_states

        # if initial state from config if available (e.g. when brain training)
        # skip if config missing
        # check if any keys from config exit in mapper
        # if so update self.initial_states with config
        # create new config to update self.all_data
        if config:
            new_config = {}
            for k, v in config.items():
                if k in self.initial_states_mapper.keys():
                    initial_state[self.initial_states_mapper[k]] = v
                else:
                    new_config[k] = v
            logger.info(f"Initial states: {initial_state}")
        else:
            new_config = None

        # if config:
        #     initial_state.update(
        #         (k, config[k]) for k in initial_state.keys() & config.keys()
        #     )

        initial_action = {k: random.random() for k in self.action_keys}
        if new_config:
            logger.info(f"Initializing episode with provided config: {new_config}")
            self.config = new_config
        elif not new_config and self.episode_inits:
            logger.info(
                f"No episode initializations provided, using initializations in yaml `episode_inits`"
            )
            logger.info(f"Episode config: {self.episode_inits}")
            self.config = self.episode_inits
        else:
            logger.warn(
                "No config provided, so using random Gaussians. This probably not what you want!"
            )
            # TODO: during ddm_trainer save the ranges of configs (and maybe states too for initial conditions)
            # to a file so we can sample from that range instead of random Gaussians
            # request_continue = input("Are you sure you want to continue with random configs?")
            if self.config_keys:
                self.config = {k: random.random() for k in self.config_keys}
            else:
                self.config = None

        # update state with initial_state values if
        # provided by config
        # otherwise default is used
        self.state = initial_state
        self.action = initial_action

        # Grab signal params pertaining to specific format of key_parameter from Inkling
        self.config_signals = {}
        if new_config and self.signal_builder:
            for k, v in self.signal_builder["signal_params"].items():
                for key, value in new_config.items():
                    if k in key:
                        self.config_signals.update({key: value})

        if self.config_signals:
            # If signal params from Inkling, use those for building signals
            self.signals = {}
            for key, val in self.signal_builder["signal_types"].items():
                self.signals.update(
                    {
                        key: SignalBuilder(
                            val,
                            new_config["horizon"],
                            {
                                k.split("_")[1]: v
                                for k, v in self.config_signals.items()
                                if key in k
                            },
                        )
                    }
                )

            self.current_signals = {}
            for key, val in self.signals.items():
                self.current_signals.update(
                    {key: float(self.signals[key].get_current_signal())}
                )
        elif self.signal_builder:
            # Otherwise use signal builder from simulator/conf
            self.signals = {}
            for key, val in self.signal_builder["signal_types"].items():
                self.signals.update(
                    {
                        key: SignalBuilder(
                            val,
                            self.signal_builder["horizon"],
                            self.signal_builder["signal_params"][key],
                        )
                    }
                )

            self.current_signals = {}
            for key, val in self.signals.items():
                self.current_signals.update(
                    {key: float(self.signals[key].get_current_signal())}
                )
        else:
            print("No signal builder used")

        # capture all data
        # TODO: check if we can pick a subset of data yaml, i.e., what happens if
        # {simulator.state, simulator.action, simulator.config} is a strict subset {data.inputs + data.augmented_cols, self.outputs}
        if self.config:
            self.all_data = {**self.state, **self.action, **self.config}
        else:
            self.all_data = {**self.state, **self.action}
        if self.prep_pipeline:
            from preprocess import pipeline

            self.all_data = pipeline(self.all_data)

        if self.iteration_col:
            self.all_data[self.iteration_col] = self.iteration_counter
            logger.info(
                f"Iteration used as a feature. Iteration #: {self.iteration_counter}"
            )

        ## if you're using lagged_features, we need to initialize them
        ## will initially be set to the same value, which is either 0
        ## or the initial value of the state depending on zero_padding
        ## and gets updated during each episode step
        if self.lagged_inputs > 1 or self.concatenate_var_length:
            self.lagged_all_data = {
                k: self.all_data["_".join(k.split("_")[:-1])]
                if not self.lagged_padding
                else 0
                for k in self.lagged_feature_cols
            }
            self.all_data = {**self.all_data, **self.lagged_all_data}
            self.all_data["terminal"] = False

        # if self.concatenate_var_length:
        #     all_data = {
        #         k: self.all_data[k] for k in self.features
        #         if k not in self.lagged_feature_cols
        #         else
        #     }

    def episode_step(self, action: Dict[str, float]) -> Dict:
        # load design matrix for self.model.predict
        # should match the shape of conf.data.inputs
        # make dict of D={states, actions, configs}
        # ddm_inputs = filter D \ (conf.data.inputs+conf.data.augmented_cols)
        # ddm_outputs = filter D \ conf.data.outputs

        # initialize matrix of all actions
        # set current action to action_1
        # all other actions get pushed back one value to action_{i+1}
        if self.concatenate_var_length:
            # only create lagged action if they were provided in
            # concatenate_var_length
            actions_to_lag = list(
                set(list(self.concatenate_var_length.keys())) & set(list(action.keys()))
            )
            if actions_to_lag:
                lagged_action = {
                    f"{k}_{i}": action[k] if i == 1 else self.all_data[f"{k}_{i-1}"]
                    for k in actions_to_lag
                    for i in range(1, self.concatenate_var_length[k] + 1)
                }
                action = lagged_action
        elif self.lagged_inputs > 1:
            lagged_action = {
                f"{k}_{i}": v if i == 1 else self.all_data[f"{k}_{i-1}"]
                for k, v in action.items()
                for i in range(1, self.lagged_inputs + 1)
            }
            action = lagged_action
        self.all_data.update(action)
        if self.prep_pipeline:
            from preprocess import pipeline

            self.all_data = pipeline(self.all_data)
        self.iteration_counter += 1
        if self.iteration_col:
            logger.info(
                f"Iteration used as a feature. Iteration #: {self.iteration_counter}"
            )

        if hasattr(self, "exogeneous_variables"):
            logger.info(
                f"Updating {self.exogeneous_variables} using next iteration from episode #: {self.exog_ep['episode'].values[0]}"
            )
            next_iteration = self.exog_ep[
                self.exog_ep["iteration"] == self.iteration_counter + 1
            ]
            self.all_data.update(
                next_iteration.reset_index()[self.exogeneous_variables].loc[0].to_dict()
            )
            # set terminal to true if at the last iteration
            if self.iteration_counter == self.exog_ep["iteration"].max() - 1:
                self.all_data["terminal"] = True

        # Use the signal builder's value as input to DDM if specified
        if self.signal_builder:
            for key in self.features:
                if key in self.signals:
                    self.all_data.update({key: self.current_signals[key]})

        # Use the signal builder's value as input to DDM if specified
        # if self.signal_builder:
        #     for key in self.features:
        #         if key in self.signals:
        #             self.all_data.update({key: self.current_signals[key]})

        # MAKE SURE THIS IS SORTED ACCORDING TO THE ORDER USED IN TRAINING
        ddm_input = {k: self.all_data[k] for k in self.features}

        # input_list = [
        #     list(self.state.values()),
        #     list(self.config.values()),
        #     list(action.values()),
        # ]

        # input_array = [item for subl in input_list for item in subl]
        input_array = list(ddm_input.values())
        X = np.array(input_array).reshape(1, -1)
        if self.diff_state:
            preds = np.array(list(self.state.values())) + self.model.predict(
                X
            )  # compensating for output being delta state st+1-st
            # preds = np.array(list(simstate))+self.dd_model.predict(X) # if doing per iteration prediction of delta state st+1-st
        else:
            preds = self.model.predict(X)  # absolute prediction
        ddm_output = dict(zip(self.labels, preds.reshape(preds.shape[1]).tolist()))

        # update lagged values in ddm_output -> which updates self.all_data
        # current predictions become the new t1, everything else is pushed back by 1
        if self.concatenate_var_length:
            lagged_ddm_output = {
                f"{k}_{i}": v if i == 1 else self.all_data[f"{k}_{i-1}"]
                for k, v in ddm_output.items()
                for i in range(1, self.concatenate_var_length[k] + 1)
            }
            ddm_output = lagged_ddm_output
        elif self.lagged_inputs > 1:
            lagged_ddm_output = {
                f"{k}_{i}": v if i == 1 else self.all_data[f"{k}_{i-1}"]
                for k, v in ddm_output.items()
                for i in range(1, self.lagged_inputs + 1)
            }
            ddm_output = lagged_ddm_output
        self.all_data.update(ddm_output)
        if self.iteration_col:
            self.all_data[self.iteration_col] = self.iteration_counter

        # current state is just the first value
        states_lagged = list(
            set(list(self.concatenate_var_length.keys())) & set(self.state_keys)
        )
        if self.lagged_inputs > 1 and not self.concatenate_var_length:
            self.state = {k: self.all_data[f"{k}_1"] for k in self.state_keys}
        elif self.concatenate_var_length:
            self.state = {
                k: self.all_data[f"{k}_1"] if k in states_lagged else self.all_data[k]
                for k in self.state_keys
            }
        else:
            self.state = {k: self.all_data[k] for k in self.state_keys}
        # self.state = dict(zip(self.state_keys, preds.reshape(preds.shape[1]).tolist()))

        if self.signal_builder:
            self.current_signals = {}
            for key, val in self.signals.items():
                self.current_signals.update(
                    {key: float(self.signals[key].get_current_signal())}
                )

        return dict(self.state)

    def get_state(self) -> Dict:
        if hasattr(self, "label_types"):
            for key, val_type in self.label_types.items():
                state_val = self.state[key]
                val_type = val_type.split(" ")
                if len(val_type) < 2:
                    bottom = state_val - 10
                    top = state_val + 10
                    val_type = val_type[0]
                elif len(val_type) == 2:
                    # val_type, val_range = val_type.split(" ")
                    val_range = val_type[1]
                    val_type = val_type[0]
                    val_range = val_range.split(",")
                    bottom = float(val_range[0])
                    top = float(val_range[1])
                else:
                    raise ValueError(f"Invalid label type provided: {type(val_type)}")
                state_val = type_conversion(state_val, val_type, bottom, top)
                self.state[key] = state_val

        if self.signal_builder:
            state_plus_signals = {**self.state, **self.current_signals}
            logger.info(f"Current state with signals: {state_plus_signals}")
            return state_plus_signals
        else:
            logger.info(f"Current state: {self.state}")
            return dict(self.state)

    def halted(self):
        pass


def get_reward(obs, action):
    # fill out reward here

    reward = np.random.random()
    return reward


def get_terminal(obs, action) -> int:
    # fill out terminal here

    terminal = np.random.random()
    return terminal > 0.5


def run_baseenv_aml(simulator: Simulator):
    class DDMBaseEnv(BaseEnv):
        def __init__(self, simulator, config):
            self.simulator = simulator

        def poll(
            self,
        ):
            obs = self.simulator.get_state()
            return obs

        def send_actions(self, action_dict):
            # action_dict is typcially a dictionary of env_ids keys
            # with action values, but we only have a singleton
            self.simulator.episode_step(action_dict)

        def try_reset(self, config):
            self.simulator.episode_start(config)


def flatten_structure(structure):
    flattened_structure = {}
    for key, value in structure.items():
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if isinstance(value, np.ndarray):
            flattened_structure[key] = value.flatten()
        else:
            flattened_structure[key] = value
    return flattened_structure


def run_gym_aml(
    simulator: Simulator, config: Optional[Dict[str, float]] = None, local: bool = False
):
    class GymWrapper(gym.Env):
        def __init__(self, config):
            self.sim = simulator
            self.config = config
            self.observation_space = gym.spaces.Dict(
                {
                    k: Box(low=-float("inf"), high=float("inf"))
                    for k in self.sim.state_keys
                }
            )
            self.action_space = gym.spaces.Dict(
                {k: Box(low=-1, high=1, dtype=np.int32) for k in self.sim.action_keys}
            )
            self.max_iter = 10

        def _get_obs(self):
            """Get the observable state."""

            sim_state = self.sim.state
            sim_state_array = {k: np.array([v]) for k, v in sim_state.items()}
            # sim_state = flatten_structure(sim_state)
            # sim_state = np.array(list(sim_state.items()))

            return sim_state_array

        def _get_info(self):
            """Get additional info not needed by the agent's decision."""
            return {}

        def reset(
            self,
            # config: Optional[Dict[str, Any]] = None,
            seed: int = None,
            options: Optional[Dict[str, Any]] = None,
        ):
            self.sim.episode_start(options)

            return self._get_obs(), self._get_info()

        def step(self, action: Dict[str, float]):
            obs = self.sim.episode_step(action)
            reward = get_reward(obs, action)

            terminated = get_terminal(obs, action)

            info = self._get_info()

            truncated = self.sim.iteration_counter >= self.max_iter

            # return obs, reward, terminal, info
            return (
                self._get_obs(),
                reward,
                terminated,
                truncated,
                info,
            )

        def render(self):
            return NotImplementedError

    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.logger import pretty_print
    from ray.tune.registry import register_env

    # Register the simulation as an RLlib environment.
    register_env("GymWrapper", lambda config: GymWrapper(config))

    def train():
        # Define the algo for training the agent
        algo = (
            PPOConfig()
            # Modify also instance_count in job definition to use more than one worker
            # Setting workers to zero allows using breakpoints in sim for debugging
            .rollouts(num_rollout_workers=1 if not local else 0)
            .resources(num_gpus=0)
            # Set the training batch size to the appropriate number of steps
            .training(train_batch_size=4_000)
            .environment(env="GymWrapper")
            .build()
        )
        # Train for 10 iterations
        for i in range(10):
            result = algo.train()
            print(pretty_print(result))

        # outputs can be found in AML Studio under the "Outputs + Logs" tab of your job
        checkpoint_dir = algo.save(checkpoint_dir="./outputs")
        print(f"Checkpoint saved in directory {checkpoint_dir}")

    if local:
        train()
    else:
        from ray_on_aml.core import Ray_On_AML

        ray_on_aml = Ray_On_AML()
        ray = ray_on_aml.getRay()

        if ray:
            print("head node detected")
            ray.init(address="auto")
            print(ray.cluster_resources())
            train()
        else:
            print("in worker node")


def env_setup():
    """Helper function to setup connection with Project Bonsai

    Returns
    -------
    Tuple
        workspace, and access_key
    """

    load_dotenv(verbose=True)
    workspace = os.getenv("SIM_WORKSPACE")
    access_key = os.getenv("SIM_ACCESS_KEY")

    env_file_path = os.path.join(dir_path, ".env")
    env_file_exists = os.path.exists(env_file_path)
    if not env_file_exists:
        open(env_file_path, "a").close()

    if not all([env_file_exists, workspace]):
        workspace = input("Please enter your workspace id: ")
        set_key(env_file_path, "SIM_WORKSPACE", workspace)
    if not all([env_file_exists, access_key]):
        access_key = input("Please enter your access key: ")
        set_key(env_file_path, "SIM_ACCESS_KEY", access_key)

    load_dotenv(verbose=True, override=True)
    workspace = os.getenv("SIM_WORKSPACE")
    access_key = os.getenv("SIM_ACCESS_KEY")

    return workspace, access_key


def test_policy(
    num_episodes: int = 5,
    num_iterations: int = 5,
    sim: Optional[Simulator] = None,
    config: Optional[Dict[str, float]] = None,
    policy=random_policy,
):
    """Test a policy using random actions over a fixed number of episodes

    Parameters
    ----------
    num_episodes : int, optional
        number of iterations to run, by default 10
    """

    def _config_clean(in_config):
        new_config = {}
        for k, v in in_config.items():
            if type(v) in [DictConfig, dict]:
                v = random.uniform(in_config[k]["min"], in_config[k]["max"])
                k = in_config[k]["inkling_name"]
            new_config[k] = v
        return new_config

    for episode in range(num_episodes):
        iteration = 0
        terminal = False
        if config:
            config = _config_clean(config)
            logger.info(f"Configuration: {new_config}")
            sim.episode_start(new_config)
        else:
            sim.episode_start()
        sim_state = sim.get_state()
        while not terminal:
            action = policy(sim_state)
            sim.episode_step(action)
            sim_state = sim.get_state()
            logger.info(f"Running iteration #{iteration} for episode #{episode}")
            logger.info(f"Action: {action}")
            logger.info(f"Observations: {sim_state}")
            iteration += 1
            terminal = iteration >= num_iterations

    return sim


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    save_path = cfg["model"]["saver"]["filename"]
    save_path = os.path.join(dir_path, save_path)
    model_name = cfg["model"]["name"]
    states = cfg["simulator"]["states"]
    actions = cfg["simulator"]["actions"]
    configs = cfg["simulator"]["configs"]
    initial_states = cfg["simulator"]["initial_states"]
    policy = cfg["simulator"]["policy"]
    # logflag = cfg["simulator"]["logging"]
    # logging not yet implemented

    ts_model = model_name.lower() in ["nhits", "tftmodel", "varima", "ets", "sfarima"]
    if ts_model:
        scale_data = cfg["model"]["scale_data"]
    else:
        scale_data = cfg["model"]["build_params"]["scale_data"]
    # scale_data = cfg["data"]["scale_data"]
    diff_state = hydra_read_config_var(cfg, "data", "diff_state")
    concatenated_steps = hydra_read_config_var(cfg, "data", "concatenated_steps")
    concatenated_zero_padding = hydra_read_config_var(
        cfg, "data", "concatenated_zero_padding"
    )
    concatenate_var_length = hydra_read_config_var(cfg, "data", "concatenate_length")
    exogeneous_variables = hydra_read_config_var(cfg, "data", "exogeneous_variables")
    exogeneous_save_path = hydra_read_config_var(cfg, "data", "exogeneous_save_path")
    initial_values_save_path = hydra_read_config_var(
        cfg, "data", "initial_values_save_path"
    )

    workspace_setup = cfg["simulator"]["workspace_setup"]
    episode_inits = cfg["simulator"]["episode_inits"]

    input_cols = cfg["data"]["inputs"]
    output_cols = cfg["data"]["outputs"]
    augmented_cols = cfg["data"]["augmented_cols"]
    preprocess = hydra_read_config_var(cfg, "data", "preprocess")
    iteration_col = hydra_read_config_var(cfg, "data", "iteration_col")
    iteration_col = iteration_col if iteration_col in input_cols else None
    if type(input_cols) == ListConfig:
        input_cols = list(input_cols)
    if type(output_cols) == ListConfig:
        output_cols = list(output_cols)
    if type(augmented_cols) == ListConfig:
        augmented_cols = list(augmented_cols)

    input_cols = input_cols + augmented_cols

    ts_model = False
    logger.info(f"Using DDM with {policy} policy")
    if model_name.lower() == "pytorch":
        from all_models import available_models
    elif model_name.lower() in ["nhits", "tftmodel", "varima", "ets", "sfarima"]:
        from timeseriesclass import darts_models as available_models

        ts_model = True
    else:
        from model_loader import available_models

    Model = available_models[model_name]
    if not ts_model:
        model = Model()
    else:
        model = Model()
        model.build_model()

    model.load_model(filename=save_path, scale_data=scale_data)
    # model.build_model(**cfg["model"]["build_params"])

    if not initial_states:
        if not initial_values_save_path:
            logger.warn(
                "No initial values provided, using randomly initialized states which is probably NOT what you want"
            )
        initial_states = {k: random.random() for k in states}

    signal_builder = cfg["simulator"]["signal_builder"]

    # Grab standardized way to interact with sim API
    sim = Simulator(
        model,
        states,
        actions,
        configs,
        input_cols,
        output_cols,
        episode_inits,
        initial_states,
        signal_builder,
        diff_state,
        concatenated_steps,
        concatenated_zero_padding,
        concatenate_var_length,
        prep_pipeline=preprocess,
        iteration_col=iteration_col,
        exogeneous_variables=exogeneous_variables,
        exogeneous_save_path=exogeneous_save_path,
        initial_values_save_path=initial_values_save_path,
    )

    if policy == "random":
        random_policy_from_keys = partial(random_policy, action_keys=sim.action_keys)
        test_policy(
            sim=sim,
            config=None,
            policy=random_policy_from_keys,
        )
    elif policy == "ray-local":
        run_gym_aml(sim, config=None, local=True)
    elif policy == "ray-aml":
        run_gym_aml(sim, config=None, local=False)
    elif isinstance(policy, int):
        # If docker PORT provided, set as exported brain PORT
        port = policy
        url = f"http://localhost:{port}"
        print(f"Connecting to exported brain running at {url}...")
        trained_brain_policy = partial(brain_policy, exported_brain_url=url)
        test_policy(
            sim=sim,
            config={**initial_states},
            policy=trained_brain_policy,
        )
    elif policy == "bonsai":
        if workspace_setup:
            logger.info(f"Loading workspace information form .env")
            env_setup()
            load_dotenv(verbose=True, override=True)
        # Configure client to interact with Bonsai service
        config_client = BonsaiClientConfig()
        client = BonsaiClient(config_client)

        # SimulatorInterface needs to be initialized with
        # existin state attribute
        # TODO: see if we can move this into constructor method
        sim.episode_start()

        # Create simulator session and init sequence id
        registration_info = SimulatorInterface(
            name=env_name,
            timeout=60,
            simulator_context=config_client.simulator_context,
        )

        def CreateSession(
            registration_info: SimulatorInterface, config_client: BonsaiClientConfig
        ):
            """Creates a new Simulator Session and returns new session, sequenceId"""

            try:
                print(
                    "config: {}, {}".format(
                        config_client.server, config_client.workspace
                    )
                )
                registered_session: SimulatorSessionResponse = client.session.create(
                    workspace_name=config_client.workspace, body=registration_info
                )
                print("Registered simulator. {}".format(registered_session.session_id))

                return registered_session, 1
            # except HttpResponseError as ex:
            # print(
            #     "HttpResponseError in Registering session: StatusCode: {}, Error: {}, Exception: {}".format(
            #         ex.status_code, ex.error.message, ex
            #     )
            # )
            # raise ex
            except Exception as ex:
                print(
                    "UnExpected error: {}, Most likely, it's some network connectivity issue, make sure you are able to reach bonsai platform from your network.".format(
                        ex
                    )
                )
                raise ex

        registered_session, sequence_id = CreateSession(
            registration_info, config_client
        )
        episode = 0
        iteration = 0

        try:
            while True:
                # Advance by the new state depending on the event type
                sim_state = SimulatorState(
                    sequence_id=sequence_id,
                    state=sim.get_state(),
                    halted=sim.halted(),
                )
                try:
                    event = client.session.advance(
                        workspace_name=config_client.workspace,
                        session_id=registered_session.session_id,
                        body=sim_state,
                    )
                    sequence_id = event.sequence_id
                    print(
                        "[{}] Last Event: {}".format(
                            time.strftime("%H:%M:%S"), event.type
                        )
                    )
                # UPDATE #comment-out-azure-cli:
                # - commented out the HttpResponseError since it relies on azure-cli-core which has
                # - conflicting dependencies with microsoft-bonsai-api
                # - the catch-all exception below should still re-connect on disconnects
                # except HttpResponseError as ex:
                #     print(
                #         "HttpResponseError in Advance: StatusCode: {}, Error: {}, Exception: {}".format(
                #             ex.status_code, ex.error.message, ex
                #         )
                #     )
                #     # This can happen in network connectivity issue, though SDK has retry logic, but even after that request may fail,
                #     # if your network has some issue, or sim session at platform is going away..
                #     # So let's re-register sim-session and get a new session and continue iterating. :-)
                #     registered_session, sequence_id = CreateSession(
                #         registration_info, config_client
                #     )
                #     continue
                except Exception as err:
                    print("Unexpected error in Advance: {}".format(err))
                    # Ideally this shouldn't happen, but for very long-running sims It can happen with various reasons, let's re-register sim & Move on.
                    # If possible try to notify Bonsai team to see, if this is platform issue and can be fixed.
                    registered_session, sequence_id = CreateSession(
                        registration_info, config_client
                    )
                    continue

                # Event loop
                if event.type == "Idle":
                    time.sleep(event.idle.callback_time)
                    print("Idling...")
                elif event.type == "EpisodeStart":
                    print(event.episode_start.config)
                    sim.episode_start(event.episode_start.config)
                    episode += 1
                elif event.type == "EpisodeStep":
                    iteration += 1
                    sim.episode_step(event.episode_step.action)
                elif event.type == "EpisodeFinish":
                    print("Episode Finishing...")
                    iteration = 0
                elif event.type == "Unregister":
                    print(
                        "Simulator Session unregistered by platform because '{}', Registering again!".format(
                            event.unregister.details
                        )
                    )
                    registered_session, sequence_id = CreateSession(
                        registration_info, config_client
                    )
                    continue
                else:
                    pass
        except KeyboardInterrupt:
            # Gracefully unregister with keyboard interrupt
            client.session.delete(
                workspace_name=config_client.workspace,
                session_id=registered_session.session_id,
            )
            print("Unregistered simulator.")
        except Exception as err:
            # Gracefully unregister for any other exceptions
            client.session.delete(
                workspace_name=config_client.workspace,
                session_id=registered_session.session_id,
            )
            print("Unregistered simulator because: {}".format(err))


if __name__ == "__main__":
    main()
