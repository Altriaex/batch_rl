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
import gin

from batch_rl.fixed_replay import run_experiment
from batch_rl.fixed_replay.agents import dqn_agent
from batch_rl.fixed_replay.agents import multi_head_dqn_agent
from batch_rl.fixed_replay.agents import quantile_agent
from batch_rl.fixed_replay.agents import rainbow_agent

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
flags.DEFINE_string('query_method', None, 'the type of reward model')
flags.DEFINE_boolean('use_preference_rewards', True, "whether to use preference rewards")
flags.DEFINE_string('batch_id', '4', 'run rewrard trained on data upto this id.')
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
    folder_name = "_".join(["training_logs", FLAGS.agent_name, FLAGS.query_method, "query_selection", FLAGS.batch_id])
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
    preference_reward_file = "_".join([exp_id, game, split, FLAGS.query_method, "query_selection", FLAGS.batch_id]) + ".zip"
    shutil.copy2(osp.join(exp_base, "preference_rewards", preference_reward_file), training_log_path)
    shutil.unpack_archive(osp.join(training_log_path, preference_reward_file), extract_dir=osp.join(training_log_path, "replay_logs"), format="zip")
    os.remove(osp.join(training_log_path, preference_reward_file))
    return training_log_path

def pack_agents(FLAGS):
    path, split = osp.split(FLAGS.exp_dir)
    path, game = osp.split(path)
    exp_base, exp_id = osp.split(path)
    agent_name = "_".join([FLAGS.agent_name, FLAGS.query_method, "query_selection", FLAGS.batch_id])
    archive_name = osp.join(exp_base, "agents", "_".join([exp_id, game, split, agent_name]))
    shutil.make_archive(base_name=archive_name, root_dir=osp.join(FLAGS.exp_dir, agent_name), base_dir=None, format="zip")



def main(unused_argv):
    path, split = osp.split(FLAGS.exp_dir)
    path, game = osp.split(path)
    gin.bind_parameter('atari_lib.create_atari_environment.game_name', game)
    if FLAGS.use_preference_rewards:
        training_log_path = create_logs_for_training(FLAGS)
        agent_name = "_".join([FLAGS.agent_name, FLAGS.query_method, "query_selection", FLAGS.batch_id])
        FLAGS.replay_dir = training_log_path
    else:
        raise NotImplementedError
    tf.logging.set_verbosity(tf.logging.INFO)
    base_run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    replay_data_dir = os.path.join(FLAGS.replay_dir, 'replay_logs')
    create_agent_fn = functools.partial(
        create_agent, replay_data_dir=replay_data_dir)
    runner = run_experiment.FixedReplayRunner(osp.join(FLAGS.exp_dir, agent_name), create_agent_fn)
    runner.run_experiment()
    pack_agents(FLAGS)


if __name__ == '__main__':
    should_be_required = ["agent_name", "original_log_folder", "query_method", "exp_dir", "batch_id"]
    for f in should_be_required:
        flags.mark_flag_as_required(f)
    app.run(main)
