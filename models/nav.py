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
from six.moves import range

# Tensorflow objects/functions
nest = tf.contrib.framework.nest
FLAGS = tf.app.flags.FLAGS

# Structure to be sent from actors to learner.
AgentOutput = collections.namedtuple('AgentOutput', 'action policy_logits baseline depth_logits')

class Agent(snt.RNNCore):
  """Agent with ResNet."""

  def __init__(self, num_actions):
    super(Agent, self).__init__(name='agent')
    self._num_actions = num_actions
    with self._enter_variable_scope():
      self._core_first = tf.contrib.rnn.LSTMBlockCell(64)
      self._core_second = tf.contrib.rnn.LSTMBlockCell(256)

  def initial_state(self, batch_size):
    return tf.transpose(tf.concat([self._core_first.zero_state(batch_size, tf.float32),
               self._core_second.zero_state(batch_size, tf.float32)], axis=-1), [1, 0, 2])

  def _torso(self, input_):
    last_action, env_output = input_
    reward, _, _, frame = env_output

    # Convert to floats.
    frame = tf.to_float(frame)
    frame /= 255.0

    # Perform preprocessing
    frame, depth = tf.split(frame, [3, 1], axis=-1)
    with tf.variable_scope('convnet'):
      conv_out = frame
      for i, (nf, rf, stride) in enumerate([(16, 8, 4), (32, 4, 2)]):
        # Downscale.
        conv_out = snt.Conv2D(nf, rf, stride=stride, padding='VALID',
                              initializers={'b': tf.zeros_initializer(),
                                            'w': tf.orthogonal_initializer()})(conv_out)
        conv_out = tf.nn.relu(conv_out)
    conv_out = snt.BatchFlatten()(conv_out)
    conv_out = snt.Linear(256)(conv_out)
    conv_out = tf.nn.relu(conv_out)

    # Append clipped last reward and one hot last action.
    clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
    return tf.concat([conv_out, clipped_reward], axis=1)

  def _head(self, core_output):
    # Policy gradient
    policy_logits = snt.Linear(self._num_actions, name='policy_logits')(core_output)
    baseline = tf.squeeze(snt.Linear(1, name='baseline')(core_output), axis=-1)

    # Depth prediction
    depth = snt.Linear(128, name='depth_logits')(core_output)
    depth_logits = tf.stack([snt.Linear(FLAGS.deep_quantization,
                                        name='depth_logits_{}'.format(i))(depth) for i in range(64)], axis=1)

    # Sample an action from the policy.
    new_action = tf.multinomial(policy_logits, num_samples=1, output_dtype=tf.int32)
    new_action = tf.squeeze(new_action, 1, name='new_action')

    return AgentOutput(new_action, policy_logits, baseline, depth_logits)

  def _build(self, input_, core_state):
    # Build graph
    action, env_output = input_
    actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0), (action, env_output))
    outputs, core_state = self.unroll(actions, env_outputs, core_state)
    return nest.map_structure(lambda t: tf.squeeze(t, 0), outputs), core_state

  @snt.reuse_variables
  def unroll(self, actions, env_outputs, core_state):
    _, _, done, _ = env_outputs
    torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs))

    # Note, in this implementation we can't use CuDNN RNN to speed things up due
    # to the state reset. This can be XLA-compiled (LSTMBlockCell needs to be
    # changed to implement snt.LSTMCell).
    initial_core_state_first = self._core_first.zero_state(tf.shape(actions)[1], tf.float32)
    initial_core_state_second = self._core_second.zero_state(tf.shape(actions)[1], tf.float32)
    core_state_first = tf.contrib.rnn.LSTMStateTuple(c=core_state[:, 0, :64], h=core_state[:, 1, :64])
    core_state_second = tf.contrib.rnn.LSTMStateTuple(c=core_state[:, 0, 64:], h=core_state[:, 1, 64:])
    core_output_list = []
    for input_, d, act in zip(tf.unstack(torso_outputs), tf.unstack(done), tf.unstack(actions)):
      # First layer
      core_state_first = nest.map_structure(functools.partial(tf.where, d),
                                      initial_core_state_first, core_state_first)
      core_output_first, core_state_first = self._core_first(input_, core_state_first)

      # Second layer
      core_state_second = nest.map_structure(functools.partial(tf.where, d),
                                      initial_core_state_second, core_state_second)
      one_hot_last_action = tf.one_hot(act, self._num_actions)
      input_second = tf.concat([input_[:,:-1], core_output_first, one_hot_last_action], axis=1)
      core_output, core_state_second = self._core_second(input_second, core_state_second)
      core_output_list.append(core_output)
    core_state = tf.transpose(tf.concat([core_state_first, core_state_second], axis=-1), [1, 0, 2])
    return snt.BatchApply(self._head)(tf.stack(core_output_list)), core_state