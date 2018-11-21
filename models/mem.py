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
# TODO Initialization

"""ResNet Model"""
# Python 2.X and 3.X compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import libraries
import collections
import functools
import sonnet as snt
import tensorflow as tf

# Tensorflow objects/functions
nest = tf.contrib.framework.nest

# Structure to be sent from actors to learner.
AgentOutput = collections.namedtuple('AgentOutput', 'action policy_logits baseline encoder_output memory memory_state')

class Agent(snt.RNNCore):
  """Agent with ResNet."""

  def __init__(self, num_actions):
    super(Agent, self).__init__(name='agent')
    self._num_actions = num_actions

    with self._enter_variable_scope():
      self._core = tf.contrib.rnn.LSTMBlockCell(256)

  def initial_state(self, batch_size):
    return self._core.zero_state(batch_size, tf.float32)

  def _torso(self, input_):
    last_action, env_output = input_
    reward, _, _, frame = env_output

    # Convert to floats.
    frame = tf.to_float(frame)

    frame /= 255.0
    with tf.variable_scope('convnet'):
      conv_out = frame
      for i, (nf, rf, stride) in enumerate([(32, 8, 4), (64, 4, 2), (64, 3, 1)]):
        # Downscale.
        conv_out = snt.Conv2D(nf, rf, stride=stride, padding='VALID',
                              initializers={'b': tf.zeros_initializer(),
                                            'w': tf.orthogonal_initializer()})(conv_out)
        conv_out = tf.nn.relu( conv_out)
    conv_out = snt.BatchFlatten()(conv_out)
    conv_out = snt.Linear(256)(conv_out)
    conv_out = tf.nn.relu(conv_out)

    # Append clipped last reward and one hot last action.
    clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
    one_hot_last_action = tf.one_hot(last_action, self._num_actions)
    return tf.concat([conv_out, clipped_reward, one_hot_last_action], axis=1)

  def _head(self, core_output, encoder_output, memory):
    output_mlp = tf.nn.relu(snt.Linear(256, name='mlp')(core_output))
    policy_logits = snt.Linear(self._num_actions, name='policy_logits')(output_mlp)
    baseline = tf.squeeze(snt.Linear(1, name='baseline')(output_mlp), axis=-1)

    # Sample an action from the policy.
    new_action = tf.multinomial(policy_logits, num_samples=1, output_dtype=tf.int32)
    new_action = tf.squeeze(new_action, 1, name='new_action')

    # Take last memory output
    memory_state = core_output[:, 256:]

    return AgentOutput(new_action, policy_logits, baseline, encoder_output, memory, memory_state)

  def _memory(self, memory):
    # Create the variables
    w_key = tf.get_variable("w_key", shape=[1, memory.get_shape()[-1], 256])
    w_val = tf.get_variable("w_val", shape=[1, memory.get_shape()[-1], 256])
    # mem_key = snt.Linear(128, name='mem_key')(memory)
    # mem_val = snt.Linear(128, name='mem_value')(memory)
    mem_key = tf.matmul(memory,  tf.tile(w_key, [memory.get_shape()[0], 1, 1]))
    mem_val = tf.matmul(memory, tf.tile(w_val, [memory.get_shape()[0], 1, 1]))

    return mem_key, mem_val

  def _build(self, input_, core_state, memory=None, memory_state=None):
    # Build graph
    memory = tf.expand_dims(tf.expand_dims(memory, 0), 0)
    action, env_output = input_
    actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0), (action, env_output))
    outputs, core_state = self.unroll(actions, env_outputs, core_state, memory, memory_state)
    return nest.map_structure(lambda t: tf.squeeze(t, 0), outputs), core_state

  def unroll(self, actions, env_outputs, core_state, memory, memory_state):
    _, _, done, _ = env_outputs
    torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs))
    memory_key, memory_val = snt.BatchApply(self._memory)(memory)

    # Note, in this implementation we can't use CuDNN RNN to speed things up due
    # to the state reset. This can be XLA-compiled (LSTMBlockCell needs to be
    # changed to implement snt.LSTMCell).
    initial_core_state = self._core.zero_state(tf.shape(actions)[1], tf.float32)
    initial_memory_state = tf.zeros(tf.shape(memory_state))
    core_output_list = []
    for input_, d, mem_key, mem_val in zip(tf.unstack(torso_outputs), tf.unstack(done),
                                           tf.unstack(memory_key), tf.unstack(memory_val)):

      # If the episode ended, the core state should be reset before the next.
      core_state = nest.map_structure(functools.partial(tf.where, d),
                                      initial_core_state, core_state)
      memory_state = nest.map_structure(functools.partial(tf.where, d),
                                       initial_memory_state, memory_state)
      input_ = tf.concat([input_, memory_state], axis=1)
      core_output, core_state = self._core(input_, core_state)
      att = tf.nn.softmax(tf.squeeze(tf.matmul(mem_key, tf.expand_dims(core_output, axis=-1)), axis=-1))
      memory_state = tf.squeeze(tf.matmul(tf.expand_dims(att, axis=1), mem_val), axis=1)
      core_output_list.append(tf.concat([core_output, memory_state], axis=1))

    return snt.BatchApply(self._head)(tf.stack(core_output_list), torso_outputs, memory), core_state
