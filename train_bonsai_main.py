#!/usr/bin/env python3

"""
MSFT Bonsai SDK3 Template for Simulator Integration using Python
Copyright 2020 Microsoft
Usage:
  For registering simulator with the Bonsai service for training:
    python main.py --api-host https://api.bons.ai \
           --workspace <workspace_id> \
           --accesskey="<access_key> \
  Then connect your registered simulator to a Brain via UI
  Alternatively, one can set the SIM_ACCESS_KEY and SIM_WORKSPACE as
  environment variables.
"""

import json
import time
from typing import Dict, Any, Optional
from microsoft_bonsai_api.simulator.client import (
    BonsaiClientConfig,
    BonsaiClient
)
from microsoft_bonsai_api.simulator.generated.models import (
    SimulatorState,
    SimulatorInterface,
)
import numpy as np
import yaml
from predictor import ModelPredictor

class TemplateSimulatorSession():
    def __init__(self):
        with open('config/config_model.yml') as cmfile:
            self.model_config = yaml.full_load(cmfile)
        
        # Obtain model limitations
        with open('config/model_limits.yml') as mlimfile:
            self.model_limits = yaml.full_load(mlimfile)
        
        state_space_dim = 0
        action_space_dim = 0
        self.state = []
        self.action = []
        for key, value in self.model_config['IO']['feature_name'].items():
            if value == 'state':
                state_space_dim += 1
                # Setting values to initialize, to be replaced by episode_start
                self.state.append(self.model_limits[key]['mean'])
            elif value == 'action':
                action_space_dim += 1
                # Setting values to initialize, to be replaced by episode_start
                self.action.append(self.model_limits[key]['mean'])
            else:
                print('Please fix config_model.yml to specify either state or action')
                exit()

        self.predictor = ModelPredictor(
            modeltype=self.model_config['MODEL']['type'],
            noise_percentage=0,
            state_space_dim=state_space_dim,
            action_space_dim=action_space_dim,
            markovian_order=self.model_config['LSTM']['markovian_order'],
        )
    
    def get_state(self) -> Dict[str, Any]:
        """Called to retreive the current state of the simulator.
        
        Returns
        -------
        dictionary
            Dictionary of sim_state elements at current iteration
        """
        state_dict = {}
        i = 0
        for key, value in self.model_config['IO']['feature_name'].items():
            if value == 'state':
                state_dict[key] = float(self.state[i])
                i += 1

        return state_dict
    
    def episode_start(self, config: Dict[str, Any]):
        """Method invoked at the start of each episode with a given 
        episode configuration.
        Parameters
        ----------
        config : Dict[str, Any]
        SimConfig parameters for the current episode defined in Inkling
        """
        self.state = self.predictor.reset_state(config)

        try:
            self.predictor.noise_percentage = config['noise_percentage']
        except:
            pass

    def episode_step(self, action: Dict[str, Any]):
        """Called for each step of the episode 
        Parameters
        ----------
        action : Dict[str, Any]
        BrainAction chosen from the Bonsai Service, prediction or exploration
        """
        self.action = []
        for key, value in self.model_config['IO']['feature_name'].items():
            if value == 'action':
                self.action.append(action[key])

        self.state = self.predictor.predict(
            state=self.state,
            action=self.action,
        )

    def halted(self) -> bool:
        """ Should return True if the simulator cannot continue"""
        features = np.concatenate([self.state, self.action])
        num_tripped = self.predictor.warn_limitation(features)
        
        if num_tripped > 0:
            return True

        return False

def main():
    # Grab standardized way to interact with sim API
    sim = TemplateSimulatorSession()

    # Configure client to interact with Bonsai service
    config_client = BonsaiClientConfig()
    client = BonsaiClient(config_client)

    # Create simulator session and init sequence id
    registration_info = SimulatorInterface(
                            name='datadrivenmodel', 
                            timeout=60, 
                            simulator_context=config_client.simulator_context, 
    )

    def CreateSession(registration_info: SimulatorInterface, config_client: BonsaiClientConfig):
        """Creates a new Simulator Session and returns new session, sequenceId
        """
        try:
            print("config: {}, {}".format(config_client.server, config_client.workspace))
            registered_session: SimulatorSessionResponse = client.session.create(
                workspace_name=config_client.workspace, body=registration_info
            )
            print("Registered simulator. {}".format(registered_session.session_id))

            return registered_session, 1
        except HttpResponseError as ex:
            print("HttpResponseError in Registering session: StatusCode: {}, Error: {}, Exception: {}".format(ex.status_code, ex.error.message, ex))
            raise ex
        except Exception as ex:
            print("UnExpected error: {}, Most likely, It's some network connectivity issue, make sure, you are able to reach bonsai platform from your PC.".format(ex))
            raise ex
    
    registered_session, sequence_id = CreateSession(registration_info, config_client)

    sequence_id = 1

    try:
        while True:
            # Advance by the new state depending on the event type
            sim_state = SimulatorState(
                            sequence_id=sequence_id, state=sim.get_state(), 
                            halted=sim.halted()
            )
            try:
                event = client.session.advance(
                            workspace_name=config_client.workspace, 
                            session_id=registered_session.session_id, 
                            body=sim_state
                )
                sequence_id = event.sequence_id
                print("[{}] Last Event: {}".format(time.strftime('%H:%M:%S'), 
                                                event.type))

            except HttpResponseError as ex:
                print("HttpResponseError in Advance: StatusCode: {}, Error: {}, Exception: {}".format(ex.status_code, ex.error.message, ex))
                # This can happen in network connectivity issue, though SDK has retry logic, but even after that request may fail, 
                # if your network has some issue, or sim session at platform is going away..
                # So let's re-register sim-session and get a new session and continue iterating. :-) 
                registered_session, sequence_id = CreateSession(registration_info, config_client)
                continue
            except Exception as err:
                print("Unexpected error in Advance: {}".format(err))
                # Ideally this shouldn't happen, but for very long-running sims It can happen with various reasons, let's re-register sim & Move on.
                # If possible try to notify Bonsai team to see, if this is platform issue and can be fixed.
                registered_session, sequence_id = CreateSession(registration_info, config_client)
                continue

            # Event loop
            if event.type == 'Idle':
                time.sleep(event.idle.callback_time)
                print('Idling...')
            elif event.type == 'EpisodeStart':
                sim.episode_start(event.episode_start.config)
            elif event.type == 'EpisodeStep':
                sim.episode_step(event.episode_step.action)
            elif event.type == 'EpisodeFinish':
                print('Episode Finishing...')
            elif event.type == 'Unregister':
                print("Simulator Session unregistered by platform, Registering again!")
                registered_session, sequence_id = CreateSession(registration_info, config_client)
                continue
            else:
                pass
    except KeyboardInterrupt:
        # Gracefully unregister with keyboard interrupt
        client.session.delete(
            workspace_name=config_client.workspace, 
            session_id=registered_session.session_id
        )
        print("Unregistered simulator.")
    except Exception as err:
        # Gracefully unregister for any other exceptions
        client.session.delete(
            workspace_name=config_client.workspace, 
            session_id=registered_session.session_id
        )
        print("Unregistered simulator because: {}".format(err))

if __name__ == "__main__":
    main()