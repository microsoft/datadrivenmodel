import logging
import os
import random
import time
import datetime
import pathlib
import copy
from typing import Any, Dict, List
from collections import deque,OrderedDict

from numpy.lib.function_base import iterable
from omegaconf import ListConfig

import numpy as np
import pandas as pd

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
log_path = "logs"


class Simulator(BaseModel):
    def __init__(
        self,
        model,
        states: List[str],
        actions: List[str],
        configs: List[str],
        inputs: List[str],
        outputs: List[str],
        episode_inits: Dict[str, float],
        initial_states: Dict[str, float],
        diff_state: bool = False,
        concatenated_steps: int=1,
        concatenated_zero_padding: bool=True,
        log_file: str=None
    ):

        self.model = model
        # self.features = states + configs + actions
        # self.labels = states
        self.features = inputs
        self.labels = outputs
        self.config_keys = configs
        self.episode_inits = episode_inits
        self.state_keys = states
        self.action_keys = actions
        self.diff_state = diff_state
        self.concatenated_steps = concatenated_steps
        self.concatenated_zero_padding = concatenated_zero_padding
        self.initial_states = initial_states
        # Add logging
        if log_file == "enable":
            current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_file = os.path.join(
                log_path, current_time + "_" + env_name + "_log.csv"
            )
            logs_directory = pathlib.Path(log_file).parent.absolute()
            if not pathlib.Path(logs_directory).exists():
                print(
                    "Directory does not exist at {0}, creating now...".format(
                        str(logs_directory)
                    )
                )
                logs_directory.mkdir(parents=True, exist_ok=True)
        self.log_file = log_file

        logger.info(f"DDM features: {self.features}")
        logger.info(f"DDM outputs: {self.labels}")

    def episode_start(self, config: Dict[str, Any] = None):
        """Initialize DDM. This could include initializations of configs
        as well as initial values for states.

        Parameters
        ----------
        config : Dict[str, Any], optional
            episode initializations, by default None
        """

        # initialize states based on simulator.yaml
        initial_state = self.initial_states
        initial_action = {k: random.random()*0.1 for k in self.action_keys}
        # define initial state from config if available (e.g. when brain training)
        # skip if config missing
        if config:
            logger.info(f"Initializing episode with provided config: {config}")
            self.config = config
            # Edit the config keys and initial state/action keys for a custom example.
            # TODO: A generic state/action to config mapper or other generic design is 
            # under development and can replace this requirement.
            initial_state = {
                      "theta":self.config["config_initial_theta"],
                      "alpha":self.config["config_initial_alpha"],
                      "theta_dot": self.config["config_initial_theta_dot"],
                      "alpha_dot": self.config["config_initial_alpha_dot"]
                      }
            initial_action = {k: random.random() for k in self.action_keys}
        elif not config and self.episode_inits:
            logger.info(
                f"No episode initializations provided, using initializations in yaml `episode_inits`"
            )
            logger.info(f"Episode config: {self.episode_inits}")
            self.config = self.episode_inits
        else:
            logger.warning(
                "No config provided, so using random Gaussians. This probably not what you want!"
            )
            # TODO: during ddm_trainer save the ranges of configs (and maybe states too for initial conditions)
            # to a file so we can sample from that range instead of random Gaussians
            # request_continue = input("Are you sure you want to continue with random configs?")
            self.config = {k: random.random()*0.1 for k in self.config_keys}
        self.state = dict(initial_state)
        self.action = initial_action
        # capture all data
        # TODO: check if we can pick a subset of data yaml, i.e., what happens if
        # {simulator.state, simulator.action, simulator.config} is a strict subset {data.inputs + data.augmented_cols, self.outputs}
        self.all_data = {**self.state, **self.action, **self.config}
        # repeat states instead of zero padding so that average is similar to the
        # first state
        # features can be subset of all data, history will always be of features only 
        if self.concatenated_zero_padding:
            self.hist = {k+f"_{i}": 0 \
                    for i in range(1,self.concatenated_steps)\
                    for k in self.features\
                    }
        else:
            self.hist = {k+f"_{i}": self.all_data[k] \
                    for i in range(1,self.concatenated_steps)\
                    for k in self.features\
                    }

        self.features_w_hist = {**{k:self.all_data[k] for k in self.features}, **self.hist}
        print(self.features_w_hist)



    def episode_step(self, action: Dict[str, int], iteration: int):

        # load design matrix for self.model.predict
        # should match the shape of conf.data.inputs
        # make dict of D={states, actions, configs}
        # ddm_inputs = filter D \ (conf.data.inputs+conf.data.augmented_cols)
        # ddm_outputs = filter D \ conf.data.outputs      

        self.all_data.update(action)
        self.features_w_hist.update(action)

        ddm_input = self.features_w_hist

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
        self.all_data.update(ddm_output)
        self.state = {k: self.all_data[k] for k in self.state_keys}
        # shift the historical states by 1
        i=self.concatenated_steps-2
        while i>=0:
            for k in self.features:
                if i==0:
                    self.hist[k+f"_{i+1}"]=copy.deepcopy(self.all_data[k])
                else:
                    self.hist[k+f"_{i+1}"]=copy.deepcopy(self.features_w_hist[k+f"_{i}"])

            i-=1
        # update all data with updates and updated historical states
        self.features_w_hist.update({**{k:self.all_data[k] for k in self.features}, **self.hist})
        # self.state = dict(zip(self.state_keys, preds.reshape(preds.shape[1]).tolist()))
        print(self.features_w_hist)
        time.sleep(1)
        return self.state

    def get_state(self):

        return self.state

    def halted(self):

        pass

    def log_iterations(
        self, episode: int = 0, iteration: int = 1
    ):
        """Log iterations during training to a CSV.

        Parameters
        ----------
        state : Dict
        action : Dict
        episode : int, optional
        iteration : int, optional
        """
        data = copy.deepcopy(self.features_w_hist)
        data["episode"] = episode
        data["iteration"] = iteration
        log_df = pd.DataFrame(data, index=[0])

        if os.path.exists(self.log_file):
            log_df.to_csv(path_or_buf=self.log_file, mode="a", header=False, index=False)
        else:
            log_df.to_csv(path_or_buf=self.log_file, mode="w", header=True, index=False)

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


def test_random_policy(
    num_episodes: int = 500,
    num_iterations: int = 250,
    sim: Simulator = None,
):
    """Test a policy using random actions over a fixed number of episodes

    Parameters
    ----------
    num_episodes : int, optional
        number of iterations to run, by default 10
    """

    def random_action():
        return {k: random.random() for k in sim.action_keys}

    for episode in range(num_episodes):
        iteration = 0
        terminal = False
        sim.episode_start()
        sim_state = sim.get_state()
        while not terminal:
            action = random_action()
            sim.episode_step(action, iteration)
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
    if cfg["data"]["full_or_relative"] == "relative":
        save_path = os.path.join(dir_path, save_path)
    model_name = cfg["model"]["name"]
    states = cfg["simulator"]["states"]
    actions = cfg["simulator"]["actions"]
    configs = cfg["simulator"]["configs"]
    policy = cfg["simulator"]["policy"]
    logflag = cfg["simulator"]["logging"]
    # logging not yet implemented
    scale_data = cfg["model"]["build_params"]["scale_data"]
    diff_state = cfg["data"]["diff_state"]
    workspace_setup = cfg["simulator"]["workspace_setup"]
    episode_inits = cfg["simulator"]["episode_inits"]
    initial_states = cfg["simulator"]["initial_states"]
    concatenated_steps = cfg['data']['concatenated_steps']
    concatenated_zero_padding = cfg['data']['concatenated_zero_padding']

    input_cols = cfg["data"]["inputs"]
    output_cols = cfg["data"]["outputs"]
    augmented_cols = cfg["data"]["augmented_cols"]
    if type(input_cols) == ListConfig:
        input_cols = list(input_cols)
    if type(output_cols) == ListConfig:
        output_cols = list(output_cols)
    if type(augmented_cols) == ListConfig:
        augmented_cols = list(augmented_cols)

    input_cols = input_cols + augmented_cols

    logger.info(f"Training with a new {policy} policy")
    if model_name.lower() == "pytorch":
        from all_models import available_models
    else:
        from model_loader import available_models

    Model = available_models[model_name]
    model = Model()

    model.load_model(filename=save_path, scale_data=scale_data)
    # model.build_model(**cfg["model"]["build_params"])

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
        diff_state,
        concatenated_steps,
        concatenated_zero_padding,
        logflag
    )

    # do a random action to get initial state
    sim.episode_start()

    if policy == "random":
        test_random_policy(1000, 250, sim)
    elif policy == "bonsai":
        if workspace_setup:
            logger.info(f"Loading workspace information form .env")
            env_setup()
            load_dotenv(verbose=True, override=True)
        # Configure client to interact with Bonsai service
        config_client = BonsaiClientConfig()
        client = BonsaiClient(config_client)

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

                    if sim.log_file:
                        sim.log_iterations(
                            episode=episode,
                            iteration=iteration
                        )
                elif event.type == "EpisodeStep":
                    iteration += 1
                    sim.episode_step(event.episode_step.action,iteration)
                    
                    if sim.log_file:
                        sim.log_iterations(
                            episode=episode,
                            iteration=iteration
                        )
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
