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

"""Environments and environment helper classes."""
# Python 2.X and 3.X compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import libraries
import socket
import sys
import time
import numpy as np
import os
import subprocess
import tempfile
import re
from builtins import str
from builtins import range
from random import randint
from distutils.dir_util import copy_tree, remove_tree
import collections
import tensorflow as tf

# Import mission
from malmo.missions.mission_generator import Classroom
from malmo.missions.mission import MissionEnvironment, MissionStateBuilder

# Setup global variables - configuration
NAME_GAME = "Minecraft"
ENV_DIR = "malmo"
EXEC_FILE = "launchClient"
IP = "localhost"
CURRENT_DIR = os.getcwd()
ENV_DIR = os.path.join(os.path.dirname(CURRENT_DIR), ENV_DIR, NAME_GAME)
DEFAULT_ACTION_SET = ('move 1', 'move -1',  'turn 1', 'turn -1', 'strafe 1', 'strafe -1')

# Import modules
nest = tf.contrib.framework.nest
FLAGS = tf.app.flags.FLAGS


def _port_has_listener(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result == 0


def launch_minecraft_in_background(minecraft_path, seed, ports=None, timeout=360, replaceable=True):
    if ports is None:
        ports = []
    if len(ports) == 0:
        ports = [10000]  # Default
    processes = []
    for port in ports:
        while _port_has_listener(port):
            print('Something is listening on port', port, '- will assume Minecraft is running.')
            port = int(str(11) + str(randint(0, 9)) + str(randint(0, 9)) + str(randint(0, 9)))
        replaceable_arg = "" if not replaceable else " -replaceable "
        print('Nothing is listening on port', port, '- will attempt to launch Minecraft from a new terminal.')
        if os.name == 'nt':
            p = subprocess.Popen([minecraft_path + '/launchClient.bat', '-port', str(port), replaceable_arg.strip()],
                             creationflags=subprocess.CREATE_NEW_CONSOLE, close_fds=True)
        elif sys.platform == 'darwin':
            # Can't pass parameters to launchClient via Terminal.app, so create a small launch
            # script instead.
            # (Launching a process to run the terminal app to run a small launch script to run
            # the launchClient script to run Minecraft... is it possible that this is not the most
            # straightforward way to go about things?)
            launcher_file = "/tmp/launcher_" + str(os.getpid()) + ".sh"
            tmp_file = open(launcher_file, "w")
            tmp_file.write(minecraft_path + '/launchClient.sh -port ' + str(port) + replaceable_arg)
            tmp_file.close()
            os.chmod(launcher_file, 0o700)
            p = subprocess.Popen(['open', '-a', 'Terminal.app', launcher_file])
        else:
            # Create folder if it does not exist
            env_path = minecraft_path + seed
            if not os.path.exists(env_path):
                os.makedirs(env_path)

                # Copy the folder
                copy_tree(ENV_DIR, env_path)
            os.chdir(env_path)

            log_file_cmd = tempfile.NamedTemporaryFile(delete=False)
            args_cmd = ['./launchClient.sh', "-port", str(port)]
            p = subprocess.Popen(args_cmd, bufsize=5000, stdout=log_file_cmd)
            regexp = re.compile(b'DORMANT')
            while True:
                log_file_cmd.seek(0)
                line = log_file_cmd.read()
                if regexp.search(line):
                    time.sleep(5)
                    break
                time.sleep(1)
            # p = subprocess.Popen(minecraft_path + "/launchClient.sh -port " + str(port) + replaceable_arg,
            #                  close_fds=True, shell=True,
            #                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        processes.append(p)
        print('Giving Minecraft some time to launch... ')
        launched = False
        for i in range(timeout // 3):
            time.sleep(3)
            if _port_has_listener(port):
                print('ok')
                launched = True
                break
        if not launched:
            print('Minecraft not yet launched. Giving up.')
            exit(1)
    return processes


class PyProcessDmLab(object):
    """DeepMind Lab wrapper for PyProcess."""

    def __init__(self, level, config, num_action_repeats, seed,
                 runfiles_path=None, level_cache=None):
        # Define initial attributes
        self._num_action_repeats = num_action_repeats
        self._random_state = np.random.RandomState(seed=seed)
        self._width = config.get('width', None)
        self._height = config.get('height', None)
        self._ms_per_tick = 10
        self._nmaps = 10

        # Create specific client and port
        self._default_port = str(10000)
        self._default_port = str(randint(1, 6)) + str(randint(0,4)) + str(randint(0,9)) + str(randint(0,9)) + str(randint(0,9))
        self._port = [int(self._default_port[:len(self._default_port) - len(str(seed))] + str(seed))]

        # Launch process
        self._process = launch_minecraft_in_background(ENV_DIR, str(seed), ports=self._port)
        self._client = (IP, self._port[-1])
        os.chdir(CURRENT_DIR)

        # Define mission and retrieve it from class
        self._mission = Classroom(self._ms_per_tick, self._width, self._height, seed=seed, nmaps=self._nmaps)

        # Build state environment
        self._state_builder = MissionStateBuilder(width=self._width, height=self._height)

        # Setup recording directory
        self._recording_dir = os.path.join(CURRENT_DIR, 'records/{}'.format(self._mission.mission_name))
        if not os.path.exists(self._recording_dir):
            os.makedirs(self._recording_dir)
        self._recording_path = os.path.join(self._recording_dir, '{}.tgz'.format(NAME_GAME + str(seed)))

        # Configure malmo environment
        self._env = MissionEnvironment(self._mission.mission_name, self._mission.mission_xml,
                                       self._state_builder, mission_starts=self._mission.start_spawn, remotes=self._client,
                                       role=seed, force_world_reset=False)

    def _reset(self):
        return self._env.reset(self._random_state.randint(0, self._nmaps))

    def initial(self):
        return self._env.reset(self._random_state.randint(0, self._nmaps))

    def step(self, action):
        observation, reward, done, _ = self._env.step(action)
        if done:
            observation = self._reset()
        reward = np.array(reward, dtype=np.float32)
        return reward, done, observation

    def close(self):
        return None

    @staticmethod
    def _tensor_specs(method_name, unused_kwargs, constructor_kwargs):
        """Returns a nest of `TensorSpec` with the method's output specification."""
        width = constructor_kwargs['config'].get('width', 320)
        height = constructor_kwargs['config'].get('height', 240)
        num_ch = 4 if FLAGS.depth else 3
        observation_spec = tf.contrib.framework.TensorSpec([height, width, num_ch], tf.uint8)

        if method_name == 'initial':
            return observation_spec
        elif method_name == 'step':
            return (
                tf.contrib.framework.TensorSpec([], tf.float32),
                tf.contrib.framework.TensorSpec([], tf.bool),
                observation_spec,
            )

StepOutputInfo = collections.namedtuple('StepOutputInfo', 'episode_return episode_step')
StepOutput = collections.namedtuple('StepOutput', 'reward info done observation')

class FlowEnvironment(object):
    """An environment that returns a new state for every modifying method.

    The environment returns a new environment state for every modifying action and
    forces previous actions to be completed first. Similar to `flow` for
    `TensorArray`.
    """

    def __init__(self, env):
        """Initializes the environment.

        Args:
          env: An environment with `initial()` and `step(action)` methods where
            `initial` returns the initial observations and `step` takes an action
            and returns a tuple of (reward, done, observation). `observation`
            should be the observation after the step is taken. If `done` is
            True, the observation should be the first observation in the next
            episode.
        """
        self._env = env

    def initial(self):
        """Returns the initial output and initial state.

        Returns:
          A tuple of (`StepOutput`, environment state). The environment state should
          be passed in to the next invocation of `step` and should not be used in
          any other way. The reward and transition type in the `StepOutput` is the
          reward/transition type that lead to the observation in `StepOutput`.
        """
        with tf.name_scope('flow_environment_initial'):
            initial_reward = tf.constant(0.)
            initial_info = StepOutputInfo(tf.constant(0.), tf.constant(0))
            initial_done = tf.constant(True)
            initial_observation = self._env.initial()

            initial_output = StepOutput(
                initial_reward,
                initial_info,
                initial_done,
                initial_observation)

            # Control dependency to make sure the next step can't be taken before the
            # initial output has been read from the environment.
            with tf.control_dependencies(nest.flatten(initial_output)):
                initial_flow = tf.constant(0, dtype=tf.int64)
            initial_state = (initial_flow, initial_info)
            return initial_output, initial_state

    def step(self, action, state):
        """Takes a step in the environment.

        Args:
          action: An action tensor suitable for the underlying environment.
          state: The environment state from the last step or initial state.

        Returns:
          A tuple of (`StepOutput`, environment state). The environment state should
          be passed in to the next invocation of `step` and should not be used in
          any other way. On episode end (i.e. `done` is True), the returned reward
          should be included in the sum of rewards for the ending episode and not
          part of the next episode.
        """
        with tf.name_scope('flow_environment_step'):
            flow, info = nest.map_structure(tf.convert_to_tensor, state)

            # Make sure the previous step has been executed before running the next
            # step.
            with tf.control_dependencies([flow]):
                reward, done, observation = self._env.step(action)

            with tf.control_dependencies(nest.flatten(observation)):
                new_flow = tf.add(flow, 1)

            # When done, include the reward in the output info but not in the
            # state for the next step.
            new_info = StepOutputInfo(info.episode_return + reward,
                                      info.episode_step + 1)
            new_state = new_flow, nest.map_structure(
                lambda a, b: tf.where(done, a, b),
                StepOutputInfo(tf.constant(0.), tf.constant(0)),
                new_info)

            output = StepOutput(reward, new_info, done, observation)
            return output, new_state