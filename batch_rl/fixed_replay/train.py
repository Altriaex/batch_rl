# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""The entry point for running experiments with fixed replay datasets.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import os
import os.path as osp
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
from absl import app
from absl import flags

from batch_rl.fixed_replay import run_experiment
from batch_rl.fixed_replay.agents import dqn_agent
from batch_rl.fixed_replay.agents import multi_head_dqn_agent
from batch_rl.fixed_replay.agents import quantile_agent
from batch_rl.fixed_replay.agents import rainbow_agent
from batch_rl.fixed_replay.replay_memory import fixed_replay_buffer

import numpy as np
from dopamine.discrete_domains import run_experiment as base_run_experiment

#from dopamine.google import xm_utils

flags.DEFINE_string('agent_name', None, 'Name of the agent.')
flags.DEFINE_string('replay_dir', None, 'Directory from which to load the '
                    'replay data')
flags.DEFINE_string('init_checkpoint_dir', None, 'Directory from which to load '
                    'the initial checkpoint before training starts.')
flags.DEFINE_string('exp_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"third_party/py/dopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')
flags.DEFINE_string('original_log_folder', None, 'To the root of the original logs')
flags.DEFINE_string('reward_model_type', None, 'the type of reward model')
flags.DEFINE_string('preference_model_type', None, 'the type of preference model')
flags.DEFINE_boolean('use_preference_rewards', True, "whether to use preference rewards")
FLAGS = flags.FLAGS


def create_agent(sess, environment, replay_data_dir, summary_writer=None):
  """Creates a DQN agent.

  Args:
    sess: A `tf.Session`object  for running associated ops.
    environment: An Atari 2600 environment.
    replay_data_dir: Directory to which log the replay buffers periodically.
    summary_writer: A Tensorflow summary writer to pass to the agent
      for in-agent training statistics in Tensorboard.

  Returns:
    A DQN agent with metrics.
  """
  if FLAGS.agent_name == 'dqn':
    agent = dqn_agent.FixedReplayDQNAgent
  elif FLAGS.agent_name == 'c51':
    agent = rainbow_agent.FixedReplayRainbowAgent
  elif FLAGS.agent_name == 'quantile':
    agent = quantile_agent.FixedReplayQuantileAgent
  elif FLAGS.agent_name == 'multi_head_dqn':
    agent = multi_head_dqn_agent.FixedReplayMultiHeadDQNAgent
  else:
    raise ValueError('{} is not a valid agent name'.format(FLAGS.agent_name))

  return agent(sess, num_actions=environment.action_space.n,
               replay_data_dir=replay_data_dir, summary_writer=summary_writer,
               init_checkpoint_dir=FLAGS.init_checkpoint_dir)

def create_logs_for_training(FLAGS):
    folder_name = "_".join(["training_logs", FLAGS.agent_name, FLAGS.preference_model_type, FLAGS.reward_model_type])
    training_log_path = ""
    for subfolder in [FLAGS.exp_dir, folder_name]:
        training_log_path = osp.join(training_log_path, subfolder)
    path, split = osp.split(FLAGS.exp_dir)
    path, game = osp.split(path)
    exp_base, exp_id = osp.split(path)
    original_log_path = osp.join(FLAGS.original_log_folder, game, split)
    if osp.exists(training_log_path):
        shutil.rmtree(training_log_path)
    shutil.copytree(original_log_path, training_log_path)
    preference_reward_file = "_".join([exp_id, game, split, FLAGS.preference_model_type, FLAGS.reward_model_type]) + ".zip"
    shutil.copy2(osp.join(exp_base, "preference_rewards", preference_reward_file), training_log_path)
    shutil.unpack_archive(osp.join(training_log_path, preference_reward_file), extract_dir=osp.join(training_log_path, "replay_logs"), format="zip")
    os.remove(osp.join(training_log_path, preference_reward_file))
    return training_log_path

def pack_agents(FLAGS):
    path, split = osp.split(FLAGS.exp_dir)
    path, game = osp.split(path)
    exp_base, exp_id = osp.split(path)
    agent_name = "_".join([FLAGS.agent_name, FLAGS.preference_model_type, FLAGS.reward_model_type])
    archive_name = osp.join(exp_base, "agents", "_".join([exp_id, game, split, agent_name]))
    shutil.make_archive(base_name=archive_name, root_dir=osp.join(FLAGS.exp_dir, agent_name), base_dir=None, format="zip")

def generate_data_from_records(records, frame_shape, stack_size):
    stack_obs = [np.zeros(frame_shape+(1,), dtype=np.uint8) for\
                    _ in range(stack_size-1)]
    for r_id in range(len(records['terminal'])):
        if records["terminal"][r_id] == 1:
            stack_obs = [np.zeros(frame_shape+(1,), dtype=np.uint8) for\
                            _ in range(stack_size-1)]
        stack_obs.append(records['observation'][r_id][:, :, None])
        if len(stack_obs) > stack_size:
            stack_obs.pop(0)
        yield {"states": np.concatenate(stack_obs, 2).flatten(),
               "actions": np.array([records["action"][r_id]]),
               "rewards": np.array([records["reward"][r_id]]),
               "terminals": np.array([records["reward"][r_id]])}


def _single_float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _single_int64_feature(value):
  """Returns an int32_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(example): 
  feature = {
      'states': tf.train.Feature(int64_list=tf.train.Int64List(value=example["states"].flatten())),
      'actions': _single_int64_feature(example["actions"]),
      'rewards': _single_float_feature(example["rewards"]),
      'terminals': _single_int64_feature(example["terminals"])
  }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

def check_buffer(replay_data_dir):
    # parallel write? 
    from dopamine.agents.dqn import dqn_agent as dopamine_dqn
    create_buffer = functools.partial(fixed_replay_buffer.FixedReplayBuffer,
    data_dir=replay_data_dir,observation_shape=dopamine_dqn.NATURE_DQN_OBSERVATION_SHAPE,
        stack_size=dopamine_dqn.NATURE_DQN_STACK_SIZE,
        update_horizon=1,
        gamma=0.99,
        observation_dtype=dopamine_dqn.NATURE_DQN_DTYPE.as_numpy_dtype,
        batch_size=32,
        replay_capacity=1000000)
    buffer = create_buffer(replay_suffix=0)._replay_buffers[0]
    records = buffer._store                    
    
    gen = generate_data_from_records(records, dopamine_dqn.NATURE_DQN_OBSERVATION_SHAPE, stack_size=dopamine_dqn.NATURE_DQN_STACK_SIZE)
    
    tfrecord_dir, _ = osp.split(replay_data_dir)
    tfrecord_dir = osp.join(tfrecord_dir, "tfrecords")
    os.mkdir(tfrecord_dir)
    with tf.python_io.TFRecordWriter(osp.join(tfrecord_dir, "0.tfrecord"), options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)) as writer:
        for example in gen:
            example = serialize_example(example)
            writer.write(example)


def main(unused_argv):
    if FLAGS.use_preference_rewards:
        training_log_path = create_logs_for_training(FLAGS)
        agent_name = "_".join([FLAGS.agent_name, FLAGS.preference_model_type, FLAGS.reward_model_type])
        FLAGS.replay_dir = training_log_path
    else:
        raise NotImplementedError
    tf.logging.set_verbosity(tf.logging.INFO)
    base_run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    replay_data_dir = os.path.join(FLAGS.replay_dir, 'replay_logs')
    check_buffer(replay_data_dir)
    exit()
    create_agent_fn = functools.partial(
        create_agent, replay_data_dir=replay_data_dir)
    runner = run_experiment.FixedReplayRunner(osp.join(FLAGS.exp_dir, agent_name), create_agent_fn)
    runner.run_experiment()
    pack_agents(FLAGS)


if __name__ == '__main__':
    should_be_required = ["agent_name", "original_log_folder", "reward_model_type", "preference_model_type", "exp_dir"]
    for f in should_be_required:
        flags.mark_flag_as_required(f)
    app.run(main)
