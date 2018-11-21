"""Configuration Parameters"""
# Python 2.X and 3.X compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import libraries
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('logdir', './logs/agent' , 'TensorFlow log directory.')
flags.DEFINE_enum('mode', 'train', ['train', 'test'], 'Training or test mode.')

# Flags used for testing.
flags.DEFINE_integer('test_num_episodes', 10, 'Number of episodes per level.')

# Flags used for distributed training.
flags.DEFINE_integer('task', -1, 'Task id. Use -1 for local training.')
flags.DEFINE_enum('job_name', 'learner', ['learner', 'actor'],
                  'Job name. Ignored when task is set to -1.')

# Training.
flags.DEFINE_integer('total_environment_frames', int(200.0e6),
                     'Total environment frames to train for.')
flags.DEFINE_integer('num_actors', 48, 'Number of actors.')
flags.DEFINE_integer('batch_size', 32, 'Batch size for training.')
flags.DEFINE_integer('unroll_length', 20, 'Unroll length in agent steps.')
flags.DEFINE_integer('num_action_repeats', 4, 'Number of action repeats.')
flags.DEFINE_integer('seed', 1, 'Random seed.')
flags.DEFINE_integer('num_atoms', 16, 'Number of atoms for quantile regression')

# Loss settings.
flags.DEFINE_float('entropy_cost', 0.001, 'Entropy cost/multiplier.')
flags.DEFINE_float('baseline_cost', .5, 'Baseline cost/multiplier.')
flags.DEFINE_float('discounting', .99, 'Discounting factor.')
flags.DEFINE_enum('reward_clipping', 'abs_one', ['abs_one', 'soft_asymmetric'],
                  'Reward clipping.')
flags.DEFINE_float('lmbda', .1, 'Coefficient for the .')
flags.DEFINE_float('beta', .2, 'Coefficient for the tradeoff between the inverse or forward model.')

# Environment settings.
flags.DEFINE_string('level_name', 'explore_object_locations_small',
                    '''Level name or \'dmlab30\' for the full DmLab-30 suite '''
                    '''with levels assigned round robin to the actors.''')
flags.DEFINE_integer('width', 84, 'Width of observation.')
flags.DEFINE_integer('height', 84, 'Height of observation.')

# Optimizer settings.
flags.DEFINE_float('learning_rate', 7e-4, 'Learning rate.')
flags.DEFINE_float('decay', .99, 'RMSProp optimizer decay.')
flags.DEFINE_float('momentum', 0., 'RMSProp momentum.')
flags.DEFINE_float('epsilon', .1, 'RMSProp epsilon.')

# Depth predictions settings
flags.DEFINE_integer('depth_pixels', 64, 'Number of depth pixels for auxiliary supervision.')
flags.DEFINE_integer('deep_quantization', 8, 'Number of bins for depth.')

# Components
flags.DEFINE_integer('buffer_size', 2000, 'buffer size')

# Components
flags.DEFINE_bool('depth', False, 'depth prediction.')
flags.DEFINE_bool('curiosity', True, 'depth prediction.')
flags.DEFINE_bool('qr', False, 'distributional value function with quantile regression.')
flags.DEFINE_bool('pixel_change', False, 'use pixel change.')
flags.DEFINE_bool('value_replay', False, 'use value replay.')
flags.DEFINE_bool('reward_prediction', False, 'use reward prediction.')
