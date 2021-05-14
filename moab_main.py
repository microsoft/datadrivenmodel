#!/usr/bin/env python3

"""
MSFT Bonsai SDK3 Template for Simulator Integration using Python
Copyright 2020 Microsoft

Usage:
  For registering simulator with the Bonsai service for training:
    python simulator_integration.py   
    Then connect your registered simulator to a Brain via UI, or using the CLI: `bonsai simulator unmanaged connect -b <brain-name> -a <train-or-assess> -c BalancePole --simulator-name Cartpole
"""

import datetime
import json
import logging
import os
import pathlib
import sys
import time
from typing import Any, Dict, List
import numpy as np
from torchdyn.models import *; from torchdyn.datasets import *
from torchdyn import *
from torch.autograd import grad
from base import BaseModel
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hydra.experimental import initialize, compose
# from model_loader import available_models
import logging
import matplotlib.pyplot as plt
import numpy as np
from rich import print
from rich.logging import RichHandler
import os
import pickle


from dotenv import load_dotenv, set_key
from microsoft_bonsai_api.simulator.client import BonsaiClient, BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import (
    SimulatorInterface,
    SimulatorState,
    SimulatorSessionResponse,
)
from azure.core.exceptions import HttpResponseError
from distutils.util import strtobool

import torch.utils.data as data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from jinja2 import Template
from base import BaseModel

logging.basicConfig()
logging.root.setLevel(logging.INFO)
for name in logging.Logger.manager.loggerDict.keys():
    if "azure" in name:
        logging.getLogger(name).setLevel(logging.WARNING)
        logging.propagate = True
logger = logging.getLogger("datamodeler")

import hydra
from hydra.experimental import initialize, compose
from omegaconf import DictConfig, ListConfig, OmegaConf

dir_path = os.path.dirname(os.path.realpath(__file__))
env_name = "DDM"

log_path = "logs"
DefaultTimeDelta = 0.045
R = 0.1125
max_vel = 2
MaxDistancePerStep = DefaultTimeDelta * max_vel
Dmin = -MaxDistancePerStep-R
Dmax = MaxDistancePerStep + R
Vmin = - max_vel
Vmax = max_vel
default_config = {"initial_x":np.random.uniform(Dmin,Dmax),"initial_y":np.random.uniform(Dmin,Dmax),"initial_vel_x":np.random.uniform(-1,1), 
        "initial_vel_y":np.random.uniform(-1,1),"initial_roll": np.random.uniform(-1,1), "initial_pitch": np.random.uniform(-1,1)}
env_name = "DDM"

nrows = 1
batchsize = 1
ckpt = torch.load("modelDelState.ckpt")


# initialize(config_path="conf", job_name="LNN_workflow")
# cfg = compose(config_name="config", overrides=["data=moab_5Kepi", "model=xgboost"])

class Simulator(BaseModel):
    def __init__(
        self,
        states: List[str],
        actions: List[str],
        configs: List[str],
        diff_state: bool,
        log_data: bool,
        log_file: str = None,
        sim_orig=None,
    ):
        hdim = 128
        inp_dim = 8
        out_dim = 8
        num_actions = 2
        self.L = nn.Sequential(
            nn.Linear(inp_dim, hdim),  ## Input is 8
            nn.Softplus(),
            nn.Linear(hdim, hdim),
            nn.Softplus(),
            nn.Linear(hdim, out_dim)).to(device) ## Output is 8

        # Using forces independent of dynamics q and qd, all interaction is handled in fLLN.
        self.f = nn.Sequential(
            nn.Linear(num_actions, hdim), ## Input is 2 for 2 actions
            nn.Softplus(),
            nn.Linear(hdim, 4)).to(device) ## Output is 4 accelerations
        # self.netf = fLNN(self.L, self.f, Xst[:train_size,-num_actions:]) # last two are actions
        # self.modelf = NeuralDE(func=netf, solver='dopri5').to(device)
        self.features = states + actions + configs
        self.labels = states
        self.config_keys = configs
        self.state_keys = states
        self.action_keys = actions
        self.sim_orig = sim_orig  # include simulator function if comparing to simulator
        self.diff_state = diff_state
        self.log_data = log_data
        if not log_file:
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
        else:
            pass
        self.log_file = log_file
        self.state = {j: np.random.uniform(-0.1, 0.1) for j in self.state_keys}
        

    def episode_start(self, config: Dict[str, Any] = None):
        # if config:
        #     self.config = config
        # else:
        #     # configs randomized here. Need to decide a single place for configs
        #     # range either in main.py or in simulator configs
        #     self.config = {j: np.random.uniform(-1, 1) for j in self.config_keys}
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
            # self.state = {j: np.random.uniform(-0.1, 0.1) for j in self.state_keys}
            # self.state= {"ball_x": float(np.random.uniform(-0.1, 0.1)),
            #             "ball_y": float(np.random.uniform(-0.1, 0.1)),
            #             "ball_vel_x": float(np.random.uniform(-0.1, 0.1)),
            #             "ball_vel_y": float(np.random.uniform(-0.1, 0.1))}
            a_tn = np.array([[config['initial_roll'],config['initial_pitch']]])
            a_t = torch.from_numpy(a_tn.astype(np.float32))
            # s_np = np.array([[np.random.uniform(-1,1) for i in range(8)]])
            s_np = np.array([[config['initial_roll'],config['initial_pitch'],config['initial_x'],config['initial_y'],0.0,0.0,config['initial_vel_x'],config['initial_vel_y']]])
            s_t = torch.from_numpy(s_np.astype(np.float32))
            y_it, ypred1 = iterative_predict(s_t,a_t, self.L, self.f)
            # Assuming linearity in scaling transforms need to revisit:
            self.y_out1 = yscalar.inverse_transform(s_t)+y_it
            self.yinp = s_t+ypred1[0]['y_pred']
            self.state = {"ball_x": min(Dmax, max(Dmin,float(self.y_out1[0][2].item()))),
                        "ball_y": min(Dmax,max(Dmin,float(self.y_out1[0][3].item()))),
                        "ball_vel_x": min(Vmax,max(Vmin,float(self.y_out1[0][6].item()))),
                        "ball_vel_y": min(Vmax,max(Vmin,float(self.y_out1[0][7].item())))}
            self.action = {j: a_tn[0][i] for i in range(len(a_t)) for j in self.action_keys}


    def episode_step(self, action: Dict[str, int]):

        Xact = np.array(list(action.values())).reshape(1, -1)
        X = np.concatenate((self.y_out1,Xact),axis=1)
        Xs = xscalar.transform(X)
        if self.diff_state:
            # preds = np.array(list(self.state.values()))+self.dd_model.predict(X) # compensating for output being delta state st+1-st
            # preds = np.array(list(simstate))+self.dd_model.predict(X) # if doing per iteration prediction of delta state st+1-st
            # _yinp = model.yscalar.transform(X)
            y_it, ypred1 = iterative_predict(self.yinp,
                torch.from_numpy(Xs[:,-len(action):].astype(np.float32)),
                self.L,
                self.f
                )
            self.y_out1 = self.y_out1+y_it
            self.yinp = self.yinp+ypred1[0]['y_pred']
        else:
            
            y_it, ypred1 = iterative_predict(self.yinp,
                torch.from_numpy(Xs[:,-len(action):].astype(np.float32)),
                self.L,
                self.f
                )
            self.y_out1 = y_it
            self.yinp = ypred1[0]['y_pred']
            # preds = self.dd_model.predict(X) # absolute prediction
        simstates = dict(zip(self.features, self.y_out1.reshape(self.y_out1.shape[1]).tolist()))
        self.state= {"ball_x": min(Dmax,max(Dmin, float(simstates['state_ball_x']))),
                    "ball_y": min(Dmax,max(Dmin, float(simstates['state_ball_y']))),
                    "ball_vel_x": min(Vmax,max(Vmin, float(simstates['state_ball_vel_x']))),
                    "ball_vel_y": min(Vmax,max(Vmin, float(simstates['state_ball_vel_y'])))}
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
        pass

    def log_iterations(self, state, action, episode: int = 0, iteration: int = 1):
        """Log iterations during training to a CSV.

        Parameters
        ----------
        state : Dict
        action : Dict
        episode : int, optional
        iteration : int, optional
        """
        # def add_prefixes(d, prefix: str):
        #     return {f"{prefix}_{k}": v for k, v in d.items()}

        # state = add_prefixes(state, "state")
        # action = add_prefixes(action, "action")
        # config = add_prefixes(self.config, "config")
        data = {**state, **action} #, **config}
        data["episode"] = episode
        data["iteration"] = iteration
        log_df = pd.DataFrame(data, index=[0])
        print("DATA")
        print(log_df)

        if os.path.exists(self.log_file):
            log_df.to_csv(
                path_or_buf=self.log_file, mode="a", header=False, index=False
            )
        else:
            log_df.to_csv(path_or_buf=self.log_file, mode="w", header=True, index=False)


def iterative_predict(s_t,a_t,L,f):
    _test = data.TensorDataset(s_t, s_t) #data.TensorDataset(X, Xdd)
    _predict_loaders = data.DataLoader(_test, batch_size=1, shuffle=False)
    _netf = fLNN(L, f, a_t) # Send Action from forcing function
    _modelf = NeuralDE(func=_netf, solver='dopri5').to(device)
    
    _learn = Learner(_modelf)
    
    _learn.load_state_dict(ckpt['state_dict'])
    _learn.eval()
    _trainer = pl.Trainer(logger=False)
    _ypred1=_trainer.predict(_learn, _predict_loaders)
    
    _y_out1 = yscalar.inverse_transform(_ypred1[0]['y_pred'])
    return _y_out1, _ypred1


class fLNN(nn.Module):
    def __init__(self, L, f, u):
        super().__init__()
        self.L, self.f, self.u = L, f, u
    def forward(self, x):
        with torch.set_grad_enabled(True):
            self.n = n = x.shape[1]//2 # Split in position and velocity states, 
            qqd = x.requires_grad_(True) # m x 8
            L = self._lagrangian(qqd).sum()
            J = grad(L, qqd, create_graph=True)[0] ; # m x 8
            DL_q, DL_qd = J[:,:n], J[:,n:] #J[:,:n], J[:,n:] mx4, mx4
            DDL_qd = []
            for i in range(n): # Replace range from n to full dimension. Previously used range(n):
                J_qd_i = DL_qd[:,i][:,None] # m x 1
                H_i = grad(J_qd_i.sum(), qqd, create_graph=True)[0][:,:,None] # m x 4
                DDL_qd.append(H_i) #m x 8 x 2 2 times
            DDL_qd = torch.cat(DDL_qd, 2) # m x 8 x 4
            DDL_qqd, DDL_qdqd = DDL_qd[:,:n,:], DDL_qd[:,n:,:]### m x 4 x 4, m x4 x 4
            T = torch.einsum('ijk, ij -> ik', DDL_qqd, qqd[:,n:]) # m x 4 x 4, m x 4 -> m x 4
            qdd = torch.einsum('ijk, ij -> ik', DDL_qdqd.inverse(), DL_q - T - self.f(self.u[:len(qqd),:]))#self.f(qqd))#[:,n:]) mx4x4,mx 4

        return torch.cat([qqd[:,self.n:], qdd], 1)

    def _lagrangian(self, qqd):
        return self.L(qqd)


class Learner(pl.LightningModule):

    def __init__(self, model:nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.defunc(0, x)

    def loss(self, y_hat, y):
        return ((y - y_hat)**2).mean()#((y - y_hat[:,1])**2).mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model.defunc(0, x)
        loss = self.loss(y_hat, y) 
        #print(loss.detach())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        return {'loss' : loss, 'y_pred' : y_hat, 'y_true' : y}#{'loss': loss}
    
    def test_step(self, batch, batch_idx):
        # Gets "x" and "y" tensors for current batch
        x, y = batch
        # OK to use output of forward for test (as per best practices PL)
        y_test = self.forward(x)
        # Calculate loss for the current batch
        loss = self.loss(y_test, y)
        self.log('y_test', y)

        return  {'test_loss' : loss }

    def predict(self, batch, batch_idx: int , dataloader_idx: int = None):
        x, y = batch
        
        y_hat = self.model.defunc(0, x)
        # print(y_hat.shape)
        loss = self.loss(y_hat, y)
        return {'loss' : loss, 'y_pred' : y_hat, 'y_true' : y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_dataloader(self):
        return trainloader
    
    def test_dataloader(self):
        return testloader

def env_setup():
    """Helper function to setup connection with Project Bonsai

    Returns
    -------
    Tuple
        workspace, and access_key
    """

    load_dotenv(verbose=False)
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

    load_dotenv(verbose=False, override=False)
    workspace = os.getenv("SIM_WORKSPACE")
    access_key = os.getenv("SIM_ACCESS_KEY")

    return workspace, access_key


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main entrypoint for running simulator connections

    Parameters
    ----------
    render : bool, optional
        visualize steps in environment, by default True, by default False
    log_iterations: bool, optional
        log iterations during training to a CSV file
    """


## For loading and predicting directly
    
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

    # ddModel = available_models[model_name]
    # model = ddModel()
    # order_data(config=cfg)
    # model = BaseModel()
    # logger.info(f'Model type: {available_models[cfg["model"]["name"]]}')
    logger.info(f'Model type: {"LNN"}')
    # Extract features from yaml file
    input_cols = cfg['data']['inputs']
    output_cols = cfg['data']['outputs']
    augmented_cols = cfg['data']['augmented_cols']
    dataset_path = cfg['data']['path']
    iteration_order = cfg['data']['iteration_order']
    episode_col = cfg['data']['episode_col']
    iteration_col = cfg['data']['iteration_col']
    max_rows = cfg['data']['max_rows']
    delta_state = cfg["data"]["diff_state"]

    # if cfg["data"]["full_or_relative"] == "relative":
    #     dataset_path = os.path.join(dir_path, dataset_path)

    # save_path = os.path.join(dir_path, save_path + ".pkl")

    if type(input_cols) == ListConfig:
        input_cols = list(input_cols)
    if type(output_cols) == ListConfig:
        output_cols = list(output_cols)
    if type(augmented_cols) == ListConfig:
        augmented_cols = list(augmented_cols)
    # model = BaseModel() # reassign to base class to reuse xscalar and yscalar attributes
    global xscalar, yscalar
    xscalar = pickle.load(
                open(os.path.join(save_path, "xscalar.pkl"), "rb")
            )
    yscalar = pickle.load(
                open(os.path.join(save_path, "yscalar.pkl"), "rb")
            )

    # Grab standardized way to interact with sim API
    sim = Simulator(states, actions, configs, diff_state, log_data=logflag)

    # test_sim_model(2, 250, logflag, sim)
    
    # workspace environment variables
    if policy != "bonsai":
        error("Not a brain policy")
        exit
############################################################################################
    # env_setup()
    # load_dotenv(verbose=True, override=True)

    # Configure client to interact with Bonsai service
    config_client = BonsaiClientConfig()
    client = BonsaiClient(config_client)

    # # Create simulator session and init sequence id
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
            # TODO: it's risky not doing doing `get_state` without first initializing the sim
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
                # print(event.episode_start.config)
                sim.episode_start(event.episode_start.config)
                episode += 1
                # Note: episode iteration starts at 1 for matching Telescope
                if logflag=="enable":
                    sim.log_iterations(
                        episode=episode,
                        iteration=iteration,
                        state=sim.get_state(),
                        action=sim.action,
                    )
            elif event.type == "EpisodeStep":
                iteration += 1
                sim.episode_step(event.episode_step.action)
                if logflag=="enable":
                    sim.log_iterations(
                        episode=episode,
                        iteration=iteration,
                        state=sim.get_state(),
                        action=event.episode_step.action,
                    )
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

@hydra.main(config_path="conf", config_name="config")
def test_sim(cfg: DictConfig):
    """Main entrypoint for running simulator connections

    Parameters
    ----------
    render : bool, optional
        visualize steps in environment, by default True, by default False
    log_iterations: bool, optional
        log iterations during training to a CSV file
    """


## For loading and predicting directly
    
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

    # ddModel = available_models[model_name]
    # model = ddModel()
    # order_data(config=cfg)
    # model = BaseModel()
    # logger.info(f'Model type: {available_models[cfg["model"]["name"]]}')
    logger.info(f'Model type: {"LNN"}')
    # Extract features from yaml file
    input_cols = cfg['data']['inputs']
    output_cols = cfg['data']['outputs']
    augmented_cols = cfg['data']['augmented_cols']
    dataset_path = cfg['data']['path']
    iteration_order = cfg['data']['iteration_order']
    episode_col = cfg['data']['episode_col']
    iteration_col = cfg['data']['iteration_col']
    max_rows = cfg['data']['max_rows']
    delta_state = cfg["data"]["diff_state"]

    # if cfg["data"]["full_or_relative"] == "relative":
    #     dataset_path = os.path.join(dir_path, dataset_path)

    # save_path = os.path.join(dir_path, save_path + ".pkl")

    if type(input_cols) == ListConfig:
        input_cols = list(input_cols)
    if type(output_cols) == ListConfig:
        output_cols = list(output_cols)
    if type(augmented_cols) == ListConfig:
        augmented_cols = list(augmented_cols)
    # model = BaseModel() # reassign to base class to reuse xscalar and yscalar attributes
    global xscalar, yscalar
    xscalar = pickle.load(
                open(os.path.join(save_path, "xscalar.pkl"), "rb")
            )
    yscalar = pickle.load(
                open(os.path.join(save_path, "yscalar.pkl"), "rb")
            )

if __name__ == "__main__":

    main()
    # TEST

    # hdim = 128
    # inp_dim = 8
    # out_dim = 8
    # num_actions = 2
    # L = nn.Sequential(
    #         nn.Linear(inp_dim, hdim),  ## Input is 8
    #         nn.Softplus(),
    #         nn.Linear(hdim, hdim),
    #         nn.Softplus(),
    #         nn.Linear(hdim, out_dim)).to(device) ## Output is 8

    #     # Using forces independent of dynamics q and qd, all interaction is handled in fLLN.
    # f = nn.Sequential(
    #         nn.Linear(num_actions, hdim), ## Input is 2 for 2 actions
    #         nn.Softplus(),
    #         nn.Linear(hdim, 4)).to(device) ## Output is 4 accelerations
    # a_t = torch.from_numpy(np.array([[0.01,0.01]]).astype(np.float32))
    # s_np = np.array([[np.random.uniform(-1,1) for i in range(8)]])
    # s_t = torch.from_numpy(s_np.astype(np.float32))
    # test_sim()
    # y_it, ypred1 = iterative_predict(s_t,a_t, L, f)

