import logging
import os
import random
import time
from typing import Any, Dict, List
from omegaconf import ListConfig
from functools import partial
from policies import random_policy, brain_policy
from signal_builder import SignalBuilder

import numpy as np
import pdb

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
        signal_builder: Dict[str, float],
        diff_state: bool = False,
        lagged_inputs: int = 1,
        lagged_padding: bool = False,
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
        self.signal_builder = signal_builder
        self.diff_state = diff_state
        self.lagged_inputs = lagged_inputs
        self.lagged_padding = lagged_padding

        if self.lagged_inputs > 1:
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

        # TODO: Add logging

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

        self.iteration_counter = 0

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
            self.config = {k: random.random() for k in self.config_keys}

        # update state with initial_state values if
        # provided by config
        # otherwise default is used
        self.state = initial_state
        self.action = initial_action

        # Grab signal params pertaining to specific format of key_parameter from Inkling
        self.config_signals = {}
        if new_config and self.signal_builder is not None:
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
        self.all_data = {**self.state, **self.action, **self.config}

        ## if you're using lagged_features, compute it now
        if self.lagged_inputs > 1:
            self.lagged_all_data = {
                k: self.all_data["_".join(k.split("_")[:-1])] for k in self.features
            }
        self.all_data = self.lagged_all_data

    def episode_step(self, action: Dict[str, int]) -> Dict:

        # load design matrix for self.model.predict
        # should match the shape of conf.data.inputs
        # make dict of D={states, actions, configs}
        # ddm_inputs = filter D \ (conf.data.inputs+conf.data.augmented_cols)
        # ddm_outputs = filter D \ conf.data.outputs
        # update(ddm_state) =

        if self.lagged_inputs > 1:
            lagged_action = {
                f"{k}_{i}": v if i == 1 else self.all_data[f"{k}_{i-1}"]
                for k, v in action.items()
                for i in range(1, self.lagged_inputs + 1)
            }
            action = lagged_action
        self.all_data.update(action)
        self.iteration_counter += 1

        # Use the signal builder's value as input to DDM if specified
        if self.signal_builder:
            for key in self.features:
                if key in self.signals:
                    self.all_data.update({key: self.current_signals[key]})

        # Use the signal builder's value as input to DDM if specified
        if self.signal_builder:
            for key in self.features:
                if key in self.signals:
                    self.all_data.update({key: self.current_signals[key]})

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
        if self.lagged_inputs > 1:
            lagged_ddm_output = {
                f"{k}_{i}": v if i == 1 else self.all_data[f"{k}_{i-1}"]
                for k, v in ddm_output.items()
                for i in range(1, self.lagged_inputs + 1)
            }
            ddm_output = lagged_ddm_output
        self.all_data.update(ddm_output)

        # current state is just the first value
        if self.lagged_inputs > 1:
            self.state = {k: self.all_data[f"{k}_1"] for k in self.state_keys}
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

        if self.signal_builder:
            state_plus_signals = {**self.state, **self.current_signals}
            logger.info(f"Current state with signals: {state_plus_signals}")
            return state_plus_signals
        else:
            logger.info(f"Current state: {self.state}")
            return dict(self.state)

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
    sim: Simulator = None,
    config: Dict[str, float] = None,
    policy=random_policy,
):
    """Test a policy using random actions over a fixed number of episodes

    Parameters
    ----------
    num_episodes : int, optional
        number of iterations to run, by default 10
    """

    def _config_clean(in_config: Dict):

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
        new_config = _config_clean(config)
        logger.info(f"Configuration: {new_config}")
        sim.episode_start(new_config)
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
    if cfg["data"]["full_or_relative"] == "relative":
        save_path = os.path.join(dir_path, save_path)
    model_name = cfg["model"]["name"]
    states = cfg["simulator"]["states"]
    actions = cfg["simulator"]["actions"]
    configs = cfg["simulator"]["configs"]
    initial_states = cfg["simulator"]["initial_states"]
    policy = cfg["simulator"]["policy"]
    # logflag = cfg["simulator"]["logging"]
    # logging not yet implemented
    scale_data = cfg["model"]["build_params"]["scale_data"]
    diff_state = cfg["data"]["diff_state"]
    concatenated_steps = cfg["data"]["concatenated_steps"]
    concatenated_zero_padding = cfg["data"]["concatenated_zero_padding"]

    workspace_setup = cfg["simulator"]["workspace_setup"]
    episode_inits = cfg["simulator"]["episode_inits"]

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

    logger.info(f"Using DDM with {policy} policy")
    if model_name.lower() == "pytorch":
        from all_models import available_models
    else:
        from model_loader import available_models

    Model = available_models[model_name]
    model = Model()

    model.load_model(filename=save_path, scale_data=scale_data)
    # model.build_model(**cfg["model"]["build_params"])

    if not initial_states:
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
    )

    # do a random action to get initial state
    sim.episode_start()

    if policy == "random":
        random_policy_from_keys = partial(random_policy, action_keys=sim.action_keys)
        test_policy(
            sim=sim,
            config={**initial_states},
            policy=random_policy_from_keys,
        )
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
