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

""" Replay buffer"""
# Python 2.X and 3.X compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import libraries
import tensorflow as tf
nest = tf.contrib.framework.nest

class MemoryBuffer():
    def __init__(self, capacity, template=None):
        self._capacity = capacity
        if template is None:
            template = (capacity+1, 263)
        self._buffer, self._mem = self.create(template)

    def size(self):
        return tf.minimum(self._index, self._capacity)

    def append(self, mem_buffer, mem_state=None):
        op_roll = tf.group([tf.assign(self._buffer, tf.manip.roll(self._buffer, shift=1, axis=0)), 
                            tf.assign(self._mem, mem_state)])
        with tf.control_dependencies([op_roll]):
            return tf.group([tf.scatter_update(self._buffer, tf.range(1), mem_buffer)])

    def sample(self):
        return self._buffer[:-1, ...]

    def sample_mem(self):
        return self._mem

    def reset(self):
        op_assign = tf.group([tf.assign(self._buffer, tf.zeros(tf.shape(self._buffer))),
                              tf.assign(self._mem, tf.zeros(tf.shape(self._mem)))])
        with tf.control_dependencies([op_assign]):
            return self._buffer[:-1, ...]

    def create(self, template):
        with tf.variable_scope(None, default_name='memory', reuse=False):
            return tf.get_local_variable('buffer', shape=template, trainable=False, initializer=tf.zeros_initializer), \
                   tf.get_local_variable('o', shape=[1, 256], trainable=False, initializer=tf.zeros_initializer)

class ReplayBuffer():
    def __init__(self, id_actor, unroll_length, capacity, template_env_output, template_agent_output=None):
        self._unroll_length = unroll_length
        self._capacity = capacity
        self._id_actor = tf.constant(id_actor)
        self.create(template_env_output, template_agent_output)
        with tf.variable_scope(None, default_name='replay', reuse=False):
            self._index = tf.get_local_variable('index', initializer=tf.constant(0, dtype=tf.int32),
                                            dtype=tf.int32, trainable=False,  use_resource=True)
            self._top_index = tf.get_local_variable('top_index', initializer=tf.constant(0, dtype=tf.int32),
                                            dtype=tf.int32, trainable=False,  use_resource=True)
            self._index_pos = tf.get_local_variable('rew_pos_index', initializer=tf.constant(0, dtype=tf.int32),
                                            dtype=tf.int32, trainable=False,  use_resource=True)
            self._index_neg = tf.get_local_variable('rew_neg_index', initializer=tf.constant(0, dtype=tf.int32),
                                            dtype=tf.int32, trainable=False,  use_resource=True)
            self.pos_rew = tf.get_local_variable('pos_rew', initializer=tf.zeros([self._capacity], dtype=tf.int32),
                                            dtype=tf.int32, trainable=False,  use_resource=True)
            self.neg_rew = tf.get_local_variable('neg_rew', initializer=tf.zeros([self._capacity], dtype=tf.int32),
                                            dtype=tf.int32, trainable=False,  use_resource=True)

    def size(self):
        return tf.minimum(self._index.read_value(), self._capacity)

    def is_full(self):
        return tf.greater_equal(self.size(), self._capacity)

    def append_pos_rew(self, index):
        op_roll_pos = tf.assign(self.pos_rew, tf.manip.roll(self.pos_rew, shift=1, axis=0))
        with tf.control_dependencies([op_roll_pos]):
            assign_op_pos = self.pos_rew[0].assign(index)
            with tf.control_dependencies([assign_op_pos]):
                return self._index_pos.assign_add(1)

    def append_neg_rew(self, index):
        op_roll_neg = tf.assign(self.neg_rew, tf.manip.roll(self.neg_rew, shift=1, axis=0))
        with tf.control_dependencies([op_roll_neg]):
            assign_op_neg = self.neg_rew[0].assign(index)
            with tf.control_dependencies([assign_op_neg]):
                return self._index_neg.assign_add(1)

    def append_rew(self, reward):
        index = self._index.read_value()
        was_full = self.is_full()
        with tf.control_dependencies([index, was_full]):
            op_cond = \
                tf.cond(was_full,
                        lambda: tf.cond(tf.greater(self._index_pos, 0),
                                        lambda: tf.cond(tf.greater(
                                            tf.gather(self.neg_rew, self._index_neg - 1),
                                            tf.gather(self.pos_rew, self._index_pos - 1)),
                                            lambda: self._index_pos.assign_add(-1),
                                            lambda: self._index_neg.assign_add(-1)),
                                        lambda: self._index_neg.assign_add(-1)),
                        lambda: tf.constant(0, dtype=tf.int32))
            with tf.control_dependencies([op_cond]):
                op_assign_rew = tf.cond(tf.cast(reward, tf.bool),
                                        lambda: self.append_pos_rew(index),
                                        lambda: self.append_neg_rew(index))
                with tf.control_dependencies([op_assign_rew]):
                    return self._index.assign_add(1)

    def append(self, env_output, agent_output=None):
        was_full = self.is_full()
        env_output = nest.map_structure(lambda v: v[:-1], env_output)
        agent_output = nest.map_structure(lambda v: v[:-1], agent_output)

        # The control dependency ensures that the final agent and environment states
        with tf.control_dependencies([was_full]):
            # Roll out and append reward buffer
            op_assign_rew = tf.map_fn(lambda x: self.append_rew(x), env_output.reward,
                                      parallel_iterations=1, dtype=tf.int32)
            # Roll out buffer
            op_roll_env = nest.flatten(nest.map_structure(
                lambda v: tf.assign(v, tf.manip.roll(v, shift=-self._unroll_length, axis=0)), self.buffer_env))
            op_roll_agent = nest.flatten(nest.map_structure(
                lambda v: tf.assign(v, tf.manip.roll(v, shift=-self._unroll_length, axis=0)), self.buffer_agent))

            # The control dependency ensures that the final agent and environment states
            with tf.control_dependencies([tf.group([op_assign_rew, op_roll_env, op_roll_agent])]):
                op_assign_env = nest.map_structure(lambda v, t: v[-self._unroll_length:, ...].assign(
                    tf.squeeze(t)), self.buffer_env, env_output)
                op_assign_agent = nest.map_structure(lambda v, t: v[-self._unroll_length:, ...].assign(
                    tf.squeeze(t)), self.buffer_agent, agent_output)

                # The control dependency ensures that the final agent and environment states
                with tf.control_dependencies([tf.group([op_assign_env, op_assign_agent])]):
                        return tf.cond(was_full,
                                       lambda: self._top_index.assign_add(self._unroll_length),
                                       lambda: self._top_index.assign_add(0))

    def sample(self, size=1):
        positions = tf.random_uniform((size,), minval=0, maxval=self.size() - 1, dtype=tf.int32)
        env_output = nest.map_structure(lambda v: tf.gather(v.read_value(), positions), self.buffer_env)
        agent_output = nest.map_structure(lambda v: tf.gather(v.read_value(), positions), self.buffer_agent)
        return env_output, agent_output

    def sample_sequence(self, size=1, shift=0):
        start_pos = tf.random_uniform(
            (size,), minval=0, maxval=tf.maximum(self.size() - self._unroll_length - 1 - shift, 1), dtype=tf.int32)
        positions = tf.reshape(start_pos + tf.range(self._unroll_length + 1 + shift), [-1])
        env_output = nest.map_structure(lambda v: tf.gather(v.read_value(), positions), self.buffer_env)
        agent_output = nest.map_structure(lambda v: tf.gather(v.read_value(), positions), self.buffer_agent)
        return env_output, agent_output

    # def sample_sequence(self, size=2, shift=0):
    #     start_pos = tf.tile(
    #         tf.random_uniform((size, 1),  minval=0 + shift, maxval=tf.maximum(self.size() - self._unroll_length - 1, 1),
    #                           dtype=tf.int32), [1, self._unroll_length + 1])
    #     idx = tf.reshape(start_pos +
    #         tf.reshape(tf.tile(tf.range(self._unroll_length + 1), [size]), [size, self._unroll_length + 1]),
    #         [-1])
    #     env_output = nest.map_structure(lambda v: tf.gather(v.read_value(), idx), self.buffer_env)
    #     agent_output = nest.map_structure(lambda v: tf.gather(v.read_value(), idx), self.buffer_agent)
    #     return env_output, agent_output

    def sample_rp_sequence(self):
        ind = tf.cast(tf.squeeze(tf.random_uniform((1,), minval=0, maxval=2, dtype=tf.int32)), tf.bool)
        op_zero = tf.where(tf.cast(self._index_pos, tf.bool), ind , tf.constant(False, dtype=tf.bool))
        end_pos = tf.cond(op_zero,
                          lambda: tf.gather(self.pos_rew, tf.squeeze(
                              tf.random_uniform((1,), minval=0, maxval=self._index_pos, dtype=tf.int32))),
                          lambda: tf.gather(self.neg_rew, tf.squeeze(
                              tf.random_uniform((1,), minval=0, maxval=self._index_neg, dtype=tf.int32))))
        positions = tf.maximum(end_pos - 4 - self._top_index, 0) + tf.range(5)
        env_output = nest.map_structure(lambda v: tf.gather(v.read_value(), positions), self.buffer_env)
        agent_output = nest.map_structure(lambda v: tf.gather(v.read_value(), positions), self.buffer_agent)
        return env_output, agent_output

    def create_state(self, t):
        # Creates a unique variable scope to ensure the variable name is unique.
        with tf.variable_scope(None, default_name='replay', reuse=False):
            return tf.get_local_variable(t.op.name, initializer=t, use_resource=True)

    def create(self, env_output, agent_output=None):
        env_output = nest.map_structure(lambda v: tf.zeros(v.shape, v.dtype), env_output)
        env_output = nest.map_structure(lambda v: tf.expand_dims(v, 0), env_output)
        env_output = nest.map_structure(
            lambda v: tf.tile(v, tf.TensorShape([self._capacity] + [1] * (len(v.get_shape().as_list()) - 1))),
            env_output)
        agent_output = nest.map_structure(
            lambda v: tf.tile(v, tf.TensorShape([self._capacity] + [1] * (len(v.get_shape().as_list()) - 1))),
            agent_output)
        self.buffer_env, self.buffer_agent = nest.map_structure(self.create_state, (env_output, agent_output))