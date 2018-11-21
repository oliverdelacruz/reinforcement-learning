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
from six.moves import range

# Tensorflow objects/functions
nest = tf.contrib.framework.nest

# Structure to be sent from actors to learner.
AgentOutput = collections.namedtuple('AgentOutput', 'action policy_logits baseline')
AgentOutputPC = collections.namedtuple('AgentOutputPC', 'baseline q')

class Agent(snt.RNNCore):
    """Agent with ResNet."""

    def __init__(self, num_actions):
        super(Agent, self).__init__(name='agent')
        self._num_actions = num_actions
        self._num_atoms = 16

        with self._enter_variable_scope():
            self._core = tf.contrib.rnn.LSTMBlockCell(256)

    def initial_state(self, batch_size):
        return self._core.zero_state(batch_size, tf.float32)

    @snt.reuse_variables
    def _torso(self, input_):
        last_action, env_output = input_
        reward, _, _, frame = env_output
        conv_out = self._conv(frame)
        conv_out = snt.Linear(256)(conv_out)
        conv_out = tf.nn.relu(conv_out)

        # Append clipped last reward and one hot last action.
        clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
        one_hot_last_action = tf.one_hot(last_action, self._num_actions)
        return tf.concat([conv_out, clipped_reward, one_hot_last_action], axis=1)

    @snt.reuse_variables
    def _head(self, core_output):
        policy_logits = snt.Linear(self._num_actions, name='policy_logits')(core_output)
        # quantiles = tf.reshape(snt.Linear(self._num_actions * self._num_atoms, name='quantiles')(core_output),
        #                        shape=[-1, self.num_actions, self._num_atoms])
        baseline = snt.Linear(self._num_atoms, name='quantiles')(core_output)

        # Sample an action from the policy.
        new_action = tf.multinomial(policy_logits, num_samples=1, output_dtype=tf.int32)
        new_action = tf.squeeze(new_action, 1, name='new_action')

        # Estimate Q action value function and V value function
        #q_action_values = tf.reduce_sum(quantiles * self.q, axis=-1)
        #baseline = tf.squeeze(tf.gather(q_action_values, new_action, axis=-1))
        #value_function = tf.multiply(policy_logits, q_action_values)

        return AgentOutput(new_action, policy_logits, baseline)

    @snt.reuse_variables
    def _recurrent(self, torso_outputs, actions, done, core_state):
        # Note, in this implementation we can't use CuDNN RNN to speed things up due
        # to the state reset. This can be XLA-compiled (LSTMBlockCell needs to be
        # changed to implement snt.LSTMCell).
        initial_core_state = self._core.zero_state(tf.shape(actions)[1], tf.float32)

        core_output_list = []
        for input_, d in zip(tf.unstack(torso_outputs), tf.unstack(done)):
            # If the episode ended, the core state should be reset before the next.
            core_state = nest.map_structure(functools.partial(tf.where, d),
                                            initial_core_state, core_state)
            core_output, core_state = self._core(input_, core_state)
            core_output_list.append(core_output)
        return core_output_list, core_state

    @snt.reuse_variables
    def _aux_rp(self, conv_out):
        conv_out = snt.Linear(3, name='rp')(conv_out)
        return conv_out

    def _build(self, input_, core_state):
        # Build graph
        action, env_output = input_
        actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0), (action, env_output))
        outputs, core_state = self.unroll(actions, env_outputs, core_state)
        return nest.map_structure(lambda t: tf.squeeze(t, 0), outputs), core_state

    def decov_unroll(self, actions, env_outputs, core_state):
        _, _, done, _ = env_outputs
        torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs))
        core_output_list, core_state = self._recurrent(torso_outputs, actions, done, core_state)
        return snt.BatchApply(self._deconv)(tf.stack(core_output_list))

    def unroll(self, actions, env_outputs, core_state):
        _, _, done, _ = env_outputs
        torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs))
        core_output_list, core_state = self._recurrent(torso_outputs, actions, done, core_state)
        return snt.BatchApply(self._head)(tf.stack(core_output_list)), core_state

    def rp(self, env_outputs):
        _, _, _, frame = env_outputs
        conv_out = tf.transpose(snt.BatchApply(self._conv)(frame), [1, 0, 2])
        return snt.BatchApply(self._aux_rp)(tf.reshape(conv_out, [conv_out.get_shape()[0], 1, -1]))

    @snt.reuse_variables
    def _conv(self, frame):
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
                conv_out = tf.nn.relu(conv_out)
        conv_out = snt.BatchFlatten()(conv_out)
        return conv_out

    @snt.reuse_variables
    def _deconv(self, core_output):
        fc_map = tf.nn.relu(snt.Linear(9 * 9 * 32, name='pc_fc')(core_output))
        w_deconv_v, b_deconv_v = self._conv_variable([4, 4, 1, 32], "pc_deconv_v", deconv=True)
        w_deconv_a, b_deconv_a = self._conv_variable([4, 4, self._num_actions, 32],  "pc_deconv_a", deconv=True)
        h_pc = tf.reshape(fc_map, [-1, 9, 9, 32])

        # Dueling network for V and Advantage
        deconv_v = tf.nn.relu(self._deconv2d(h_pc, w_deconv_v, 9, 9, 2) + b_deconv_v)
        deconv_a = tf.nn.relu(self._deconv2d(h_pc, w_deconv_a, 9, 9, 2) + b_deconv_a)
        deconv_a_mean = tf.reduce_mean(deconv_a, reduction_indices=3, keep_dims=True)

        # Pixel change Q (output)
        q = deconv_v + deconv_a - deconv_a_mean

        # Max Q
        baseline = tf.reduce_max(q, reduction_indices=3, keep_dims=False)
        return AgentOutputPC(baseline, q)

    def _conv_variable(self, weight_shape, name, deconv=False):
        name_w = "W_{0}".format(name)
        name_b = "b_{0}".format(name)
        if deconv:
            output_channels = weight_shape[2]
        else:
            output_channels = weight_shape[3]
        bias_shape = [output_channels]
        weight = tf.get_variable(name_w, weight_shape, initializer=tf.orthogonal_initializer())
        bias = tf.get_variable(name_b, bias_shape, initializer=tf.zeros_initializer())
        return weight, bias

    def _get2d_deconv_output_size(self, input_height, input_width,
                                  filter_height, filter_width,
                                  stride, padding_type):
        if padding_type == 'VALID':
            out_height = (input_height - 1) * stride + filter_height
            out_width = (input_width - 1) * stride + filter_width
        elif padding_type == 'SAME':
            out_height = input_height * stride
            out_width = input_width * stride
        return out_height, out_width

    def _deconv2d(self, x, W, input_width, input_height, stride):
        filter_height = W.get_shape()[0].value
        filter_width = W.get_shape()[1].value
        out_channel = W.get_shape()[2].value
        out_height, out_width = self._get2d_deconv_output_size(input_height,
                                                               input_width,
                                                               filter_height,
                                                               filter_width,
                                                               stride,
                                                               'VALID')
        batch_size = tf.shape(x)[0]
        output_shape = tf.stack([batch_size, out_height, out_width, out_channel])
        return tf.nn.conv2d_transpose(x, W, output_shape,
                                      strides=[1, stride, stride, 1],
                                      padding='VALID')
