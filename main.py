# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Importance Weighted Actor-Learner Architectures."""
# Python 2.X and 3.X compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import libraries
import tensorflow as tf
import csv

# import modules
from config import *
from utils import logger
from xvfbwrapper import Xvfb

# Define global variables
FLAGS = tf.app.flags.FLAGS
DIR_LOG = '/home/ubuntu/scalable_agent/logs/agent'

from experiment import *

# Dictionary of parameters to run
params = {'level_name': ['nav_maze_random_goal_01',  'nav_maze_random_goal_02', 'nav_maze_random_goal_03',
'explore_object_locations_small', 'explore_object_locations_large',
'explore_goal_locations_small', 'explore_goal_locations_large']}

ALGO = 'nav'

def write(path):
    # Write configuration to file
    config = FLAGS.flag_values_dict()
    with open(path + '/config.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in config.items():
           writer.writerow([key, value])

def run():
    # Set verbosity
    tf.logging.set_verbosity(tf.logging.INFO)

    # Configure the game and the level
    action_set = environment.DEFAULT_ACTION_SET
    if FLAGS.level_name == 'dmlab30' and FLAGS.mode == 'train':
        level_names = dmlab30.LEVEL_MAPPING.keys()
    elif FLAGS.level_name == 'dmlab30' and FLAGS.mode == 'test':
        level_names = dmlab30.LEVEL_MAPPING.values()
    else:
        level_names = [FLAGS.level_name]

    # Train or test the algorithm
    if FLAGS.mode == 'train':
        train(action_set, level_names)
    else:
        test(action_set, level_names)

def main(_):
    # Configure parameters for each run and perform training/testing
    if params:
        for key, values in params.items():
            for param in values:
                try:
                    # Set parameter
                    setattr(FLAGS, key, param)

                    # Set log directory
                    dir = DIR_LOG + '_' + ALGO + '_' + key + '_' + param + '_' + FLAGS.level_name

                    # Configure loggers
                    logger.configure(dir=dir)
                    setattr(FLAGS, 'logdir', dir)
                    write(FLAGS.logdir)

                    # Run algorithm
                    with Xvfb(width=1400, height=900, colordepth=24) as xvfb:
                        run()
                except Exception as e:
                    print(e)
                    pass
    else:
        run()
if __name__ == '__main__':
    tf.app.run()
