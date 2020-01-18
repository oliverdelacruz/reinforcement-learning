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
import collections
import contextlib
import sys
import six
import numpy as np
import tensorflow as tf
from six.moves import range
from tensorflow.python.ops import math_ops

# Import modules
from utils import dmlab30, py_process, vtrace, replay_buffer
try:
    import dynamic_batching
except tf.errors.NotFoundError:
    tf.logging.warning('Running without dynamic batching.')

# Import classes or variables
from models.a3c import Agent
from environments import malmo as environment

# Tensorflow objects/functions
nest = tf.contrib.framework.nest
FLAGS = tf.app.flags.FLAGS

# Structure to be sent from actors to learner.
structure = 'level_name agent_state env_outputs agent_outputs '
if FLAGS.value_replay:
    structure += 'env_outputs_vr agent_outputs_vr '
if FLAGS.pixel_change:
    structure += 'env_outputs_pc agent_outputs_pc '
if FLAGS.reward_prediction:
    structure += 'env_outputs_rp '
if FLAGS.value_replay or FLAGS.pixel_change or FLAGS.reward_prediction:
    structure += 'buffer_full'
    bool_buffer = True
else:
    bool_buffer = False
ActorOutput = collections.namedtuple('ActorOutput', structure)

def is_single_machine():
    return FLAGS.task == -1

def build_actor(agent, env, level_name, action_set, id_actor):
    """Builds the actor loop."""
    # Initial values.
    initial_env_output, initial_env_state = env.initial()
    initial_agent_state = agent.initial_state(1)
    initial_action = tf.zeros([1], dtype=tf.int32)

    # Run agent
    dummy_agent_output, _ = agent(
        (initial_action,
         nest.map_structure(lambda t: tf.expand_dims(t, 0), initial_env_output)),
        initial_agent_state)
    initial_agent_output = nest.map_structure(
        lambda t: tf.zeros(t.shape, t.dtype), dummy_agent_output)

    # Initialize buffer
    if bool_buffer:
        buffer = replay_buffer.ReplayBuffer(id_actor, FLAGS.unroll_length, FLAGS.buffer_size,
                                            initial_env_output, initial_agent_output)

    # All state that needs to persist across training iterations. This includes
    # the last environment output, agent state and last agent output. These
    # variables should never go on the parameter servers.
    def create_state(t):
        # Creates a unique variable scope to ensure the variable name is unique.
        with tf.variable_scope(None, default_name='state'):
            return tf.get_local_variable(t.op.name, initializer=t, use_resource=True)

    persistent_state = nest.map_structure(
        create_state, (initial_env_state, initial_env_output, initial_agent_state,
                       initial_agent_output))

    def step(input_, unused_i):
        """Steps through the agent and the environment."""
        env_state, env_output, agent_state, agent_output = input_

        # Run agent.
        action = agent_output[0]
        batched_env_output = nest.map_structure(lambda t: tf.expand_dims(t, 0), env_output)

        # Forward one-step
        agent_output, agent_state = agent((action, batched_env_output), agent_state)

        # Convert action index to the native action.
        action = agent_output[0][0]
        raw_action = tf.gather(action_set, action)
        env_output, env_state = env.step(raw_action, env_state)

        return env_state, env_output, agent_state, agent_output

    # Run the unroll. `read_value()` is needed to make sure later usage will
    # return the first values and not a new snapshot of the variables.
    first_values = nest.map_structure(lambda v: v.read_value(), persistent_state)
    _, first_env_output, first_agent_state, first_agent_output = first_values

    # Use scan to apply `step` multiple times, therefore unrolling the agent
    # and environment interaction for `FLAGS.unroll_length`. `tf.scan` forwards
    # the output of each call of `step` as input of the subsequent call of `step`.
    # The unroll sequence is initialized with the agent and environment states
    # and outputs as stored at the end of the previous unroll.
    # `output` stores lists of all states and outputs stacked along the entire
    # unroll. Note that the initial states and outputs (fed through `initializer`)
    # are not in `output` and will need to be added manually later.
    output = tf.scan(step, tf.range(FLAGS.unroll_length), first_values)
    _, env_outputs, _, agent_outputs = output

    # Update persistent state with the last output from the loop.
    assign_ops = nest.map_structure(lambda v, t: v.assign(t[-1]),
                                    persistent_state, output)

    # The control dependency ensures that the final agent and environment states
    # and outputs are stored in `persistent_state` (to initialize next unroll).
    with tf.control_dependencies(nest.flatten(assign_ops)):
        # Remove the batch dimension from the agent state/output.
        first_agent_state = nest.map_structure(lambda t: t[0], first_agent_state)
        first_agent_output = nest.map_structure(lambda t: t[0], first_agent_output)
        agent_outputs = nest.map_structure(lambda t: t[:, 0], agent_outputs)

        # Concatenate first output and the unroll along the time dimension.
        full_agent_outputs, full_env_outputs = nest.map_structure(
            lambda first, rest: tf.concat([[first], rest], 0),
            (first_agent_output, first_env_output), (agent_outputs, env_outputs))

        # Append buffer
        if bool_buffer:
            op_assign_index = buffer.append(full_env_outputs, full_agent_outputs)

            with tf.control_dependencies([op_assign_index]):
                # Sample buffer
                env_outputs_vr, agent_outputs_vr = buffer.sample_sequence()
                env_outputs_pc, agent_outputs_pc = buffer.sample_sequence(shift=1)
                env_outputs_rp, agent_outputs_rp = buffer.sample_rp_sequence()
                is_full = buffer.is_full()

                with tf.control_dependencies([is_full]):
                    output = ActorOutput(
                        level_name=level_name, agent_state=first_agent_state,
                        env_outputs=full_env_outputs, agent_outputs=full_agent_outputs,
                        env_outputs_vr=env_outputs_vr, agent_outputs_vr=agent_outputs_vr,
                        env_outputs_pc=env_outputs_pc, agent_outputs_pc=agent_outputs_pc,
                        env_outputs_rp=env_outputs_rp, buffer_full=is_full)

                    # No backpropagation should be done here.
                    return nest.map_structure(tf.stop_gradient, output)
        else:
            output = ActorOutput(
                level_name=level_name, agent_state=first_agent_state,
                env_outputs=full_env_outputs, agent_outputs=full_agent_outputs)

            # No backpropagation should be done here.
            return nest.map_structure(tf.stop_gradient, output)


def compute_depth_loss(logits, frame):
    # Perform preprocessing
    frame = tf.to_float(frame[:-1, ...])
    frame /= 255.0

    # Convert to floats.
    batch_size = FLAGS.batch_size * FLAGS.unroll_length
    frame, depth = tf.split(frame, [3, 1], axis=-1)
    depth = tf.reshape(depth, [batch_size] + depth.get_shape().as_list()[2:])
    logits = tf.reshape(logits, [batch_size, -1, logits.get_shape().as_list()[-1]])

    # Create a low resolution (4x16) depth map
    crop_size = tf.expand_dims(tf.constant([0.2, 0.05, 0.8, 0.95]), axis=0)
    depth = tf.reshape(tf.image.crop_and_resize(
        depth, tf.matmul(tf.ones([batch_size, 1]), crop_size),
        tf.range(batch_size), crop_size=[4, 16]), tf.shape(logits)[:-1])

    # Bucketize depth and compute loss
    depth = math_ops._bucketize(tf.pow(depth, [10]),
                                [0, 0.05, 0.175, 0.3, 0.425, 0.55, 0.675, 0.8, 1.01]) - 1
    depth_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=depth), axis=-1)

    return tf.reduce_sum(depth_loss)


def compute_baseline_loss(advantages):
    # Loss for the baseline, summed over the time dimension.
    # Multiply by 0.5 to match the standard update rule:
    # d(loss) / d(baseline) = advantage
    return .5 * tf.reduce_sum(tf.square(advantages))


def compute_entropy_loss(logits):
    policy = tf.nn.softmax(logits)
    log_policy = tf.nn.log_softmax(logits)
    entropy_per_timestep = tf.reduce_sum(-policy * log_policy, axis=-1)
    return -tf.reduce_sum(entropy_per_timestep)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=actions, logits=logits)
    advantages = tf.stop_gradient(advantages)
    policy_gradient_loss_per_timestep = cross_entropy * advantages
    return tf.reduce_sum(policy_gradient_loss_per_timestep)


def compute_huber_quantile_loss(x, rho, delta=1.0):
    shape = tf.shape(x)
    huber_loss = tf.where(
        tf.less_equal(tf.abs(x), delta),
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta))
    return tf.reduce_sum(tf.reduce_sum(tf.reduce_mean(
        tf.abs(tf.tile(tf.expand_dims(tf.expand_dims(rho, axis=0), axis=0), [shape[0], shape[1], 1, 1])
               - tf.where(tf.less(x, 0), tf.ones(shape),  tf.zeros(shape))) * huber_loss,
        axis=-1), axis=-1))


def compute_pc(first_env_outputs, env_outputs):
    obs = tf.cast(tf.concat([[first_env_outputs.observation], env_outputs.observation], 0), tf.float32)
    pc = tf.abs(obs[1:, :, 2:-2, 2:-2, :] - obs[0:-1, :, 2:-2, 2:-2, :])
    s = pc.get_shape().as_list()
    pc = tf.reduce_mean(tf.reduce_mean(tf.reshape(tf.reduce_mean(pc, axis=-1),
                                                  [s[0], s[1], s[2] // 4, 4, s[3] // 4, 4]), axis=-1), axis=3)
    pc = tf.concat([tf.expand_dims(tf.zeros_like(pc[0, ...]), axis=0), pc], axis=0)
    return env_outputs._replace(reward=pc[:-1, ...])


def compute_vs(learner_outputs, env_outputs, agent_outputs):
    # Use last baseline value (from the value function) to bootstrap.
    bootstrap_value = learner_outputs.baseline[-1]

    # At this point, the environment outputs at time step `t` are the inputs that
    # lead to the learner_outputs at time step `t`. After the following shifting,
    # the actions in agent_outputs and learner_outputs at time step `t` is what
    # leads to the environment outputs at time step `t`.
    agent_outputs = nest.map_structure(lambda t: t[1:], agent_outputs)
    rewards, infos, done, _ = nest.map_structure(lambda t: t[1:], env_outputs)
    learner_outputs = nest.map_structure(lambda t: t[:-1], learner_outputs)
    discounts = tf.to_float(~done) * FLAGS.discounting
    discounts = tf.reshape(discounts, discounts.get_shape().as_list() + [1] * (len(rewards.get_shape().as_list()) - 2))

    # Append bootstrapped value to get [r1, ..., v_t+1]
    values = learner_outputs.baseline
    values_t_plus_1 = tf.concat([values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
    rewards = rewards + discounts * values_t_plus_1 - values

    # Note that all sequences are reversed, computation starts from the back.
    sequences = (
        tf.reverse(discounts, axis=[0]),
        tf.reverse(rewards, axis=[0]),
    )

    # V-trace vs are calculated through a scan from the back to the beginning
    # of the given trajectory.
    def scanfunc(acc, sequence_item):
      discount_t, reward_t = sequence_item
      return reward_t + discount_t * acc

    initial_values = tf.zeros_like(bootstrap_value)
    rewards = tf.scan(
        fn=scanfunc,
        elems=sequences,
        initializer=initial_values,
        parallel_iterations=1,
        back_prop=False)
    # Reverse the results back to original order.
    vs_minus_v_xs = tf.reverse(rewards, [0])
    vs = tf.add(vs_minus_v_xs, values)
    return vs, learner_outputs, agent_outputs, done, infos


def compute_vtrace(learner_outputs, env_outputs, agent_outputs, aux_rewards=None):
    # Use last baseline value (from the value function) to bootstrap.
    bootstrap_value = learner_outputs.baseline[-1]

    # At this point, the environment outputs at time step `t` are the inputs that
    # lead to the learner_outputs at time step `t`. After the following shifting,
    # the actions in agent_outputs and learner_outputs at time step `t` is what
    # leads to the environment outputs at time step `t`.
    agent_outputs = nest.map_structure(lambda t: t[1:], agent_outputs)
    rewards, infos, done, _ = nest.map_structure(lambda t: t[1:], env_outputs)
    learner_outputs = nest.map_structure(lambda t: t[:-1], learner_outputs)
    if aux_rewards is not None:
        rewards += aux_rewards

    if FLAGS.reward_clipping == 'abs_one':
        clipped_rewards = tf.clip_by_value(rewards, -1, 1)
    elif FLAGS.reward_clipping == 'soft_asymmetric':
        squeezed = tf.tanh(rewards / 5.0)
        # Negative rewards are given less weight than positive rewards.
        clipped_rewards = tf.where(rewards < 0, .3 * squeezed, squeezed) * 5.
    discounts = tf.to_float(~done) * FLAGS.discounting

    # Compute V-trace returns and weights.
    # Note, this is put on the CPU because it's faster than on GPU. It can be
    # improved further with XLA-compilation or with a custom TensorFlow operation.
    with tf.device('/cpu'):
        return vtrace.from_logits(
            behaviour_policy_logits=agent_outputs.policy_logits,
            target_policy_logits=learner_outputs.policy_logits,
            actions=agent_outputs.action,
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs.baseline,
            bootstrap_value=bootstrap_value), learner_outputs, agent_outputs, done, infos


def compute_loss(agent, agent_state, env_outputs, agent_outputs,
                 env_outputs_vr=None, agent_outputs_vr=None,
                 env_outputs_pc=None, agent_outputs_pc=None,
                 env_outputs_rp=None, buffer_full=None):
    # Compute loss as a weighted sum of the baseline loss, the policy gradient
    # loss and an entropy regularization term.
    # AC baseline
    learner_outputs, _ = agent.unroll(agent_outputs.action, env_outputs, agent_state)
    aux_rewards = None

    # ICM module/Curiosity
    if FLAGS.curiosity:
        curiosity_outputs = agent.icm_unroll(agent_outputs.action, env_outputs)
        icm_inverse_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=agent_outputs.action[1:],
            logits=curiosity_outputs.icm_inverse,
            name='sparse_softmax_curiosity'
        )
        aux_rewards = curiosity_outputs.icm_forward
        icm_forward_loss = tf.reduce_sum(aux_rewards)

    vtrace_returns, learner_outputs, agent_outputs, done, infos = \
        compute_vtrace(learner_outputs, env_outputs, agent_outputs, aux_rewards=aux_rewards)

    if FLAGS.qr:
        num_atoms = FLAGS.num_atoms
        total_loss = compute_policy_gradient_loss(
            learner_outputs.policy_logits, agent_outputs.action,
            tf.reduce_sum(vtrace_returns.pg_advantages, axis=-1) / num_atoms)
        v_t_selected = tf.transpose(
            tf.reshape(tf.tile(vtrace_returns.vs, [1, 1, num_atoms]),
                       [-1, FLAGS.batch_size, num_atoms, num_atoms]),
            [0, 1, 2, 3])
        v_t_selected_target = tf.reshape(
            tf.tile(learner_outputs.baseline, [1, 1, num_atoms]),
            [-1, FLAGS.batch_size, num_atoms, num_atoms])

        rho = tf.range(0, num_atoms + 1) / num_atoms
        rho = tf.get_variable('rho', trainable=False, initializer=tf.cast(tf.reshape(tf.tile(
            tf.slice(rho, [0], [num_atoms]) + tf.slice(rho, [1], [num_atoms]) / 2,
            [num_atoms]), [num_atoms, num_atoms]), tf.float32))
        total_loss += FLAGS.baseline_cost * compute_huber_quantile_loss(v_t_selected_target - v_t_selected,
                                                                        rho)
    else:
        total_loss = compute_policy_gradient_loss(
            learner_outputs.policy_logits, agent_outputs.action,
            vtrace_returns.pg_advantages)

        total_loss += FLAGS.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs.baseline)

    total_loss += FLAGS.entropy_cost * compute_entropy_loss(
        learner_outputs.policy_logits)

    # Replay buffer
    if bool_buffer:
        is_full = tf.where(
            tf.equal(tf.reduce_sum(tf.cast(buffer_full, tf.int32)),
                     tf.constant(FLAGS.batch_size, dtype=tf.int32)),
            tf.constant(1.0, dtype=tf.float32), tf.constant(0, dtype=tf.float32), name="is_full")

    # Value replay
    if FLAGS.value_replay:
        learner_outputs_vr, _ = agent.unroll(agent_outputs_vr.action, env_outputs_vr,
                                             agent.initial_state(FLAGS.batch_size))
        vtrace_returns_vr, learner_outputs_vr, _, _, _ = compute_vtrace(learner_outputs_vr, env_outputs_vr,
                                                                        agent_outputs_vr)
        # Value replay loss
        total_loss += is_full * compute_baseline_loss(
            vtrace_returns_vr.vs - learner_outputs_vr.baseline)

    # Pixel change
    if FLAGS.pixel_change:
        first_env_outputs_pc = nest.map_structure(lambda t: t[0], env_outputs_pc)
        agent_outputs_pc = nest.map_structure(lambda t: t[1:], agent_outputs_pc)
        env_outputs_pc = nest.map_structure(lambda t: t[1:], env_outputs_pc)
        learner_outputs_pc = agent.decov_unroll(agent_outputs_pc.action, env_outputs_pc,
                                                agent.initial_state(FLAGS.batch_size))
        env_outputs_pc = compute_pc(first_env_outputs_pc, env_outputs_pc)
        vs_pc, learner_outputs_pc, agent_outputs_pc, _, _ = compute_vs(learner_outputs_pc, env_outputs_pc,
                                                                       agent_outputs_pc)

        # Extract Q for taken action
        q_pc = tf.transpose(learner_outputs_pc.q, [0, 1, 4, 2, 3])
        q_pc = tf.reshape(q_pc, [-1] + q_pc.get_shape().as_list()[2:])
        idx = tf.stack(
            [tf.range(0, q_pc.get_shape().as_list()[0]), tf.reshape(agent_outputs_pc.action, [-1])], axis=1)
        q_pc = tf.reshape(tf.gather_nd(q_pc, idx), [-1, FLAGS.batch_size] + q_pc.get_shape().as_list()[2:])

        # Pixel change loss - TD target for Q
        total_loss += is_full * .05 * compute_baseline_loss(vs_pc - q_pc)

    # Reward prediction
    if FLAGS.reward_prediction:
        labels = tf.sign(tf.cast(env_outputs_rp.reward[-1], tf.int32)) + 1
        env_outputs_rp = nest.map_structure(lambda t: t[:3], env_outputs_rp)
        logits_rp = tf.squeeze(agent.rp(env_outputs_rp))

        # Reward prediction loss
        rp_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits_rp,
            name='sparse_softmax_rp'
        )
        total_loss += is_full * tf.reduce_sum(rp_loss)

    # Depth prediction
    if FLAGS.depth:
        total_loss += compute_depth_loss(learner_outputs.depth_logits, env_outputs.observation)

    # ICM module/Curiosity
    if FLAGS.curiosity:
        total_loss = total_loss * FLAGS.lmbda + (1 - FLAGS.beta) * icm_inverse_loss + FLAGS.beta * icm_forward_loss

    return total_loss, done, infos


def build_learner(agent, agent_state, env_outputs, agent_outputs,
                  env_outputs_vr=None, agent_outputs_vr=None,
                  env_outputs_pc=None, agent_outputs_pc=None,
                  env_outputs_rp=None, buffer_full=None, **kwargs):
    """Builds the learner loop.

    Args:
      agent: A snt.RNNCore module outputting `AgentOutput` named tuples, with an
        `unroll` call for computing the outputs for a whole trajectory.
      agent_state: The initial agent state for each sequence in the batch.
      env_outputs: A `StepOutput` namedtuple where each field is of shape
      agent_outputs: An `AgentOut        [T+1, ...].
put` namedtuple where each field is of shape
        [T+1, ...].

    Returns:
      A tuple of (done, infos, and environment frames) where
      the environment frames tensor causes an update.
    """
    # Estimate loss and retrieve additional information
    total_loss, done, infos = compute_loss(agent, agent_state, env_outputs, agent_outputs,
                                           env_outputs_vr=env_outputs_vr, agent_outputs_vr=agent_outputs_vr,
                                           env_outputs_pc=env_outputs_pc, agent_outputs_pc=agent_outputs_pc,
                                           env_outputs_rp=env_outputs_rp, buffer_full=buffer_full)

    # Optimization
    num_env_frames = tf.train.get_global_step()
    learning_rate = tf.train.polynomial_decay(FLAGS.learning_rate, num_env_frames,
                                              FLAGS.total_environment_frames)
    optimizer = tf.train.RMSPropOptimizer(learning_rate, FLAGS.decay,
                                          FLAGS.momentum, FLAGS.epsilon)
    train_op = optimizer.minimize(total_loss)

    # Merge updating the network and environment frames into a single tensor.
    with tf.control_dependencies([train_op]):
        num_env_frames_and_train = num_env_frames.assign_add(
            FLAGS.batch_size * FLAGS.unroll_length * FLAGS.num_action_repeats)

    # Adding a few summaries.
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('total_loss', total_loss)

    return done, infos, num_env_frames_and_train


def create_environment(level_name, seed, is_test=False):
    """Creates an environment wrapped in a `FlowEnvironment`."""
    if level_name in dmlab30.ALL_LEVELS:
        level_name = 'contributed/dmlab30/' + level_name

    # Note, you may want to use a level cache to speed of compilation of
    # environment maps. See the documentation for the Python interface of DeepMind
    # Lab.
    config = {
        'width': FLAGS.width,
        'height': FLAGS.height,
        'datasetPath': FLAGS.dataset_path,
        'logLevel': 'WARN',
    }
    if is_test:
        config['allowHoldOutLevels'] = 'true'
        # Mixer seed for evalution, see
        # https://github.com/deepmind/lab/blob/master/docs/users/python_api.md
        config['mixerSeed'] = 0x600D5EED
    p = py_process.PyProcess(environment.PyProcessDmLab, level_name, config,
                             FLAGS.num_action_repeats, seed)
    return environment.FlowEnvironment(p.proxy)


@contextlib.contextmanager
def pin_global_variables(device):
    """Pins global variables to the specified device."""
    def getter(getter, *args, **kwargs):
        var_collections = kwargs.get('collections', None)
        if var_collections is None:
            var_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        if tf.GraphKeys.GLOBAL_VARIABLES in var_collections:
            with tf.device(device):
                return getter(*args, **kwargs)
        else:
            return getter(*args, **kwargs)

    with tf.variable_scope('', custom_getter=getter) as vs:
        yield vs


def train(action_set, level_names):
    """Train."""
    if is_single_machine():
        local_job_device = ''
        shared_job_device = ''
        is_actor_fn = lambda i: True
        is_learner = True
        global_variable_device = '/gpu'
        server = tf.train.Server.create_local_server()
        filters = []
    else:
        local_job_device = '/job:%s/task:%d' % (FLAGS.job_name, FLAGS.task)
        shared_job_device = '/job:learner/task:0'
        is_actor_fn = lambda i: FLAGS.job_name == 'actor' and i == FLAGS.task
        is_learner = FLAGS.job_name == 'learner'

        # Placing the variable on CPU, makes it cheaper to send it to all the
        # actors. Continual copying the variables from the GPU is slow.
        global_variable_device = shared_job_device + '/cpu'
        cluster = tf.train.ClusterSpec({
            'actor': ['localhost:%d' % (8001 + i) for i in range(FLAGS.num_actors)],
            'learner': ['localhost:8000']
        })
        server = tf.train.Server(cluster, job_name=FLAGS.job_name,
                                 task_index=FLAGS.task)
        filters = [shared_job_device, local_job_device]

    # Only used to find the actor output structure.
    with tf.Graph().as_default():
        agent = Agent(len(action_set))
        env = create_environment(level_names[0], seed=0)
        structure = build_actor(agent, env, level_names[0], action_set, -1)
        flattened_structure = nest.flatten(structure)
        dtypes = [t.dtype for t in flattened_structure]
        shapes = [t.shape.as_list() for t in flattened_structure]

    with tf.Graph().as_default(), \
         tf.device(local_job_device + '/cpu'), \
         pin_global_variables(global_variable_device):
        tf.set_random_seed(FLAGS.seed)  # Makes initialization deterministic.

        # Create Queue and Agent on the learner.
        with tf.device(shared_job_device):
            queue = tf.FIFOQueue(1, dtypes, shapes, shared_name='buffer')
            agent = Agent(len(action_set))

        if is_single_machine() and 'dynamic_batching' in sys.modules:
            # For single machine training, we use dynamic batching for improved GPU
            # utilization. The semantics of single machine training are slightly
            # different from the distributed setting because within a single unroll
            # of an environment, the actions may be computed using different weights
            # if an update happens within the unroll.
            old_build = agent._build

            @dynamic_batching.batch_fn
            def build(*args):
                with tf.device('/gpu'):
                    return old_build(*args)

            tf.logging.info('Using dynamic batching.')
            agent._build = build

        # Build actors and ops to enqueue their output.
        enqueue_ops = []
        for i in range(FLAGS.num_actors):
            if is_actor_fn(i):
                level_name = level_names[i % len(level_names)]
                tf.logging.info('Creating actor %d with level %s', i, level_name)
                env = create_environment(level_name, seed=i + 1)
                actor_output = build_actor(agent, env, level_name, action_set, i)
                with tf.device(shared_job_device):
                    enqueue_ops.append(queue.enqueue(nest.flatten(actor_output)))

        # If running in a single machine setup, run actors with QueueRunners
        # (separate threads).
        if is_learner and enqueue_ops:
            tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

        # Build learner.
        if is_learner:
            # Create global step, which is the number of environment frames processed.
            tf.get_variable(
                'num_environment_frames',
                initializer=tf.zeros_initializer(),
                shape=[],
                dtype=tf.int64,
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

            # Create batch (time major) and recreate structure.
            dequeued = queue.dequeue_many(FLAGS.batch_size)
            dequeued = nest.pack_sequence_as(structure, dequeued)

            def make_time_major(s):
                return nest.map_structure(
                    lambda t: tf.transpose(t, [1, 0] + list(range(t.shape.ndims))[2:]), s)
            dict_inputs = {
                'env_outputs': make_time_major(dequeued.env_outputs),
                'agent_outputs': make_time_major(dequeued.agent_outputs)
            }
            if FLAGS.value_replay:
                dict_inputs['env_outputs_vr'] = make_time_major(dequeued.env_outputs_vr)
                dict_inputs['agent_outputs_vr'] = make_time_major(dequeued.agent_outputs_vr)
            if FLAGS.pixel_change:
                dict_inputs['env_outputs_pc'] = make_time_major(dequeued.env_outputs_pc)
                dict_inputs['agent_outputs_pc'] = make_time_major(dequeued.agent_outputs_pc)
            if FLAGS.reward_prediction:
                dict_inputs['env_outputs_rp'] = make_time_major(dequeued.env_outputs_rp)
            dequeued = dequeued._replace(**dict_inputs)

            with tf.device('/gpu'):
                # Using StagingArea allows us to prepare the next batch and send it to
                # the GPU while we're performing a training step. This adds up to 1 step
                # policy lag.
                flattened_output = nest.flatten(dequeued)
                area = tf.contrib.staging.StagingArea(
                    [t.dtype for t in flattened_output],
                    [t.shape for t in flattened_output])
                stage_op = area.put(flattened_output)
                data_from_actors = nest.pack_sequence_as(structure, area.get())

                # Unroll agent on sequence, create losses and update ops.
                output = build_learner(agent, **data_from_actors._asdict())

        # Create MonitoredSession (to run the graph, checkpoint and log).
        tf.logging.info('Creating MonitoredSession, is_chief %s', is_learner)
        config = tf.ConfigProto(allow_soft_placement=True, device_filters=filters)
        with tf.train.MonitoredTrainingSession(
                server.target,
                is_chief=is_learner,
                checkpoint_dir=FLAGS.logdir,
                save_checkpoint_secs=900,
                save_summaries_steps=400000,
                log_step_count_steps=400000,
                config=config,
                hooks=[py_process.PyProcessHook()]) as session:

            if is_learner:
                # Logging.
                level_returns = {level_name: [] for level_name in level_names}
                summary_writer = tf.summary.FileWriterCache.get(FLAGS.logdir)

                # Prepare data for first run.
                session.run_step_fn(lambda step_context: step_context.session.run(stage_op))

                # Execute learning and track performance.
                num_env_frames_v = 0
                while num_env_frames_v < FLAGS.total_environment_frames:
                    level_names_v, done_v, infos_v, num_env_frames_v, _ = session.run(
                        (data_from_actors.level_name,) + output + (stage_op,))
                    level_names_v = np.repeat([level_names_v], done_v.shape[0], 0)

                    for level_name, episode_return, episode_step in zip(
                            level_names_v[done_v],
                            infos_v.episode_return[done_v],
                            infos_v.episode_step[done_v]):
                        episode_frames = episode_step * FLAGS.num_action_repeats
                        tf.logging.info('Level: %s Episode return: %f Global step: %d',
                                        level_name, episode_return, num_env_frames_v)

                        summary = tf.summary.Summary()
                        summary.value.add(tag=str(level_name) + '/episode_return',
                                          simple_value=episode_return)
                        summary.value.add(tag=str(level_name) + '/episode_frames',
                                          simple_value=episode_frames)
                        summary_writer.add_summary(summary, num_env_frames_v)

                        if FLAGS.level_name == 'dmlab30':
                            level_returns[level_name].append(episode_return)

                    if (FLAGS.level_name == 'dmlab30' and
                            min(map(len, level_returns.values())) >= 1):
                        no_cap = dmlab30.compute_human_normalized_score(level_returns,
                                                                        per_level_cap=None)
                        cap_100 = dmlab30.compute_human_normalized_score(level_returns,
                                                                         per_level_cap=100)
                        summary = tf.summary.Summary()
                        summary.value.add(tag='dmlab30/training_no_cap', simple_value=no_cap)
                        summary.value.add(tag='dmlab30/training_cap_100', simple_value=cap_100)
                        summary_writer.add_summary(summary, num_env_frames_v)

                        # Clear level scores.
                        level_returns = {level_name: [] for level_name in level_names}

            else:
                # Execute actors (they just need to enqueue their output).
                while True:
                    session.run(enqueue_ops)


def test(action_set, level_names):
    """Test."""

    level_returns = {level_name: [] for level_name in level_names}
    with tf.Graph().as_default():
        agent = Agent(len(action_set))
        outputs = {}
        for level_name in level_names:
            env = create_environment(level_name, seed=1, is_test=True)
            outputs[level_name] = build_actor(agent, env, level_name, action_set, -1)

        with tf.train.SingularMonitoredSession(
                checkpoint_dir=FLAGS.logdir,
                hooks=[py_process.PyProcessHook()]) as session:
            for level_name in level_names:
                tf.logging.info('Testing level: %s', level_name)
                while True:
                    done_v, infos_v = session.run((
                        outputs[level_name].env_outputs.done,
                        outputs[level_name].env_outputs.info
                    ))
                    returns = level_returns[level_name]
                    returns.extend(infos_v.episode_return[1:][done_v[1:]])

                    if len(returns) >= FLAGS.test_num_episodes:
                        tf.logging.info('Mean episode return: %f', np.mean(returns))
                        break

    if FLAGS.level_name == 'dmlab30':
        no_cap = dmlab30.compute_human_normalized_score(level_returns,
                                                        per_level_cap=None)
        cap_100 = dmlab30.compute_human_normalized_score(level_returns,
                                                         per_level_cap=100)
        tf.logging.info('No cap.: %f Cap 100: %f', no_cap, cap_100)