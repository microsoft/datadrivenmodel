#!/bin/usr/env python
'''
Author: Andres Felipe Alba Hernandez
v-analba@microsoft.com
Load configuration variables on the data driven model
instead of modifying parameters at the env_data_modeler
we can set the values here an import them.
'''

'''
Number of variables in the state an action space eg. (velocity, position) will
be dimension 2.
'''
STATE_SPACE_DIM=4
ACTION_SPACE_DIM=1
FEATURE_NAME = ['S0', 'S1', 'S2', 'S3', 'A0']
OUTPUT_NAME = ['S0', 'S1', 'S2', 'S3']
OUTPUT_NAME = ['INP0']