import logging
import os
import random
import sys
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
from gboost_models import GBoostModel

# Add stdout handler, with level INFO
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.DEBUG)
formater = logging.Formatter("%(name)-13s: %(levelname)-8s %(message)s")
console.setFormatter(formater)
logging.getLogger(__name__).addHandler(console)

#save_path = os.path.join("models", "gbm_pole")
save_path = './models/xgbm_pole_multi.pkl'
ddm_model = GBoostModel()
#ddm_model.load_model(dir_path=save_path, model_type="lightgbm")
ddm_model.load_model(dir_path=save_path)

feature_cols = [
    "x_position",
    "x_velocity",
    "angle_position",
    "angle_velocity",
    "action_command",
    "config_length",
    "config_masspole",
]
label_cols = [
    "state_x_position",
    "state_x_velocity",
    "state_angle_position",
    "state_angle_velocity",
]


def random_action():

    return {"action_command": random.randint(0, 1)}


initial_state = {
    "state_x_position": 0,
    "state_x_velocity": 0,
    "state_angle_position": 0,
    "state_angle_velocity": 0,
    "action_command": 0,
    "config_length": 0.5,
    "config_masspole": 0.1,
}


class Simulator(BaseModel):
    def __init__(self, feature_cols=List[str], label_cols=List[str]):

        self.ddm = ddm_model
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.last_position = initial_state

    def episode_start(self, config: Dict[str, Any] = initial_state):

        self.last_position.update(config)
        action = random_action()
        self.episode_step(action)

    def episode_step(self, action: Dict[str, int]):

        self.last_position.update(action)
        X = np.array(list(self.last_position.values())).reshape(1, -1)
        preds = self.ddm.predict(X)
        self.state = dict(zip(self.label_cols, preds.reshape(preds.shape[1]).tolist()))
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
    num_episodes: int = 2, num_iterations: int = 1,
):
    """Test a policy using random actions over a fixed number of episodes

    Parameters
    ----------
    num_episodes : int, optional
        number of iterations to run, by default 10
    """

    sim = Simulator(feature_cols=feature_cols, label_cols=label_cols)
    # test_config = {"length": 1.5}
    for episode in range(num_episodes):
        iteration = 0
        terminal = False
        # sim_state = sim.episode_start()
        while not terminal:
            action = random_action()
            sim.episode_step(action)
            sim_state = sim.get_state()
            print(f"Running iteration #{iteration} for episode #{episode}")
            print(f"Observations: {sim_state}")
            iteration += 1
            terminal = iteration >= num_iterations

    return sim


def main(config_setup: bool = False, env_name: str = "ddm-sim-generic"):
    """Main entrypoint for running simulator connections

    Parameters
    ----------
    render : bool, optional
        visualize steps in environment, by default True, by default False
    log_iterations: bool, optional
        log iterations during training to a CSV file
    """

    # workspace environment variables
    if config_setup:
        env_setup()
        load_dotenv(verbose=True, override=True)

    # Grab standardized way to interact with sim API
    # sc1_path = os.path.join(os.getcwd(), "models/sc1-small.pkl")
    sim = Simulator(feature_cols=feature_cols, label_cols=label_cols)

    # do a random action to get initial state
    sim.episode_start()

    # Configure client to interact with Bonsai service
    config_client = BonsaiClientConfig()
    client = BonsaiClient(config_client)

    # Create simulator session and init sequence id
    registration_info = SimulatorInterface(
        name=env_name, timeout=60, simulator_context=config_client.simulator_context,
    )

    def CreateSession(
        registration_info: SimulatorInterface, config_client: BonsaiClientConfig
    ):
        """Creates a new Simulator Session and returns new session, sequenceId
        """

        try:
            print(
                "config: {}, {}".format(config_client.server, config_client.workspace)
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

    registered_session, sequence_id = CreateSession(registration_info, config_client)
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
                    "[{}] Last Event: {}".format(time.strftime("%H:%M:%S"), event.type)
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
                print("Simulator Session unregistered by platform because '{}', Registering again!".format(event.unregister.details))
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

    import argparse

    parser = argparse.ArgumentParser(description="Bonsai and Simulator Integration...")
    parser.add_argument(
        "--log-iterations",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Log iterations during training",
    )
    parser.add_argument(
        "--config-setup",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Use a local environment file to setup access keys and workspace ids",
    )
    parser.add_argument(
        "--test-local",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Run simulator locally without connecting to platform",
    )

    args = parser.parse_args()

    if args.test_local:
        test_random_policy(num_episodes=100, num_iterations=1)
    else:
        main(config_setup=args.config_setup,)

