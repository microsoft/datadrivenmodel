import logging
import os
import random
import time
from distutils.util import strtobool
from typing import Any, Dict, List

import numpy as np
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
from omegaconf import DictConfig, ListConfig, OmegaConf

from model_loader import available_models

dir_path = os.path.dirname(os.path.realpath(__file__))
env_name = "DDM"


class Simulator(BaseModel):
    def __init__(self, model, states=List[str], actions=List[str], configs=List[str]):

        self.model = model
        self.features = states + actions + configs
        self.labels = states
        self.config_keys = configs
        self.state_keys = states
        self.action_keys = actions

    def episode_start(self, config: Dict[str, Any] = None):

        initial_state = {k: random.random() for k in self.state_keys}
        if config:
            self.config = config
        else:
            self.config = {k: random.random() for k in self.config_keys}
        self.state = initial_state

    def episode_step(self, action: Dict[str, int]):

        input_list = [
            list(self.state.values()),
            list(self.config.values()),
            list(action.values()),
        ]

        input_array = [item for subl in input_list for item in subl]
        X = np.array(input_array).reshape(1, -1)
        preds = self.model.predict(X)
        self.state = dict(zip(self.features, preds.reshape(preds.shape[1]).tolist()))
        return self.state

    def get_state(self):

        return self.state

    def halted(self):

        pass


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

    env_file_exists = os.path.exists(".env")
    if not env_file_exists:
        open(".env", "a").close()

    if not all([env_file_exists, workspace]):
        workspace = input("Please enter your workspace id: ")
        set_key(".env", "SIM_WORKSPACE", workspace)
    if not all([env_file_exists, access_key]):
        access_key = input("Please enter your access key: ")
        set_key(".env", "SIM_ACCESS_KEY", access_key)

    load_dotenv(verbose=True, override=True)
    workspace = os.getenv("SIM_WORKSPACE")
    access_key = os.getenv("SIM_ACCESS_KEY")

    return workspace, access_key


def test_random_policy(
    num_episodes: int = 500, num_iterations: int = 250, sim: Simulator = None,
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
            sim.episode_step(action)
            sim_state = sim.get_state()
            print(f"Running iteration #{iteration} for episode #{episode}")
            print(f"Observations: {sim_state}")
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
    scale_data = cfg["model"]["build_params"][7]["scale_data"]

    logger.info(f"Training with a new {policy} policy")

    Model = available_models[model_name]
    model = Model()

    model.build_model(model_type=model_name)
    model.load_model(filename=save_path, scale_data=scale_data)

    # Grab standardized way to interact with sim API
    sim = Simulator(model, states, actions, configs)

    # do a random action to get initial state
    sim.episode_start()

    if policy == "random":
        test_random_policy(1000, 250, sim)
    elif policy == "bonsai":
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
            """Creates a new Simulator Session and returns new session, sequenceId
            """

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
            except HttpResponseError as ex:
                print(
                    "HttpResponseError in Registering session: StatusCode: {}, Error: {}, Exception: {}".format(
                        ex.status_code, ex.error.message, ex
                    )
                )
                raise ex
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
                    sequence_id=sequence_id, state=sim.get_state(), halted=sim.halted(),
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
                except HttpResponseError as ex:
                    print(
                        "HttpResponseError in Advance: StatusCode: {}, Error: {}, Exception: {}".format(
                            ex.status_code, ex.error.message, ex
                        )
                    )
                    # This can happen in network connectivity issue, though SDK has retry logic, but even after that request may fail,
                    # if your network has some issue, or sim session at platform is going away..
                    # So let's re-register sim-session and get a new session and continue iterating. :-)
                    registered_session, sequence_id = CreateSession(
                        registration_info, config_client
                    )
                    continue
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
