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

"""Quantile Regression agent (QR-DQN) with fixed replay buffer(s)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from batch_rl.fixed_replay.replay_memory import fixed_replay_buffer
from batch_rl.multi_head import quantile_agent
import gin
import tensorflow.compat.v1 as tf


@gin.configurable
class FixedReplayQuantileAgent(quantile_agent.QuantileAgent):
  """An implementation of the DQN agent with fixed replay buffer(s)."""

  def __init__(self, sess, num_actions, replay_data_dir, replay_suffix=None,
               init_checkpoint_dir=None, **kwargs):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: tf.Session, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      replay_data_dir: str, log Directory from which to load the replay buffer.
      replay_suffix: int, If not None, then only load the replay buffer
        corresponding to the specific suffix in data directory.
      init_checkpoint_dir: str, directory from which initial checkpoint before
        training is loaded if there doesn't exist any checkpoint in the current
        agent directory. If None, no initial checkpoint is loaded.
      **kwargs: Arbitrary keyword arguments.
    """
    assert replay_data_dir is not None
    # Set replay_log_dir before calling parent's initializer
    tf.logging.info(
        'Creating FixedReplayAgent with replay directory: %s', replay_data_dir)
    tf.logging.info('\t init_checkpoint_dir: %s', init_checkpoint_dir)
    tf.logging.info('\t replay_suffix %s', replay_suffix)
    self._replay_data_dir = replay_data_dir
    self._replay_suffix = replay_suffix
    if init_checkpoint_dir is not None:
      self._init_checkpoint_dir = os.path.join(
          init_checkpoint_dir, 'checkpoints')
    else:
      self._init_checkpoint_dir = None
    super(FixedReplayQuantileAgent, self).__init__(
        sess, num_actions, **kwargs)

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    self._record_observation(observation)
    self.action = self._select_action()
    return self.action
  
  def _train_step(self):
    """Runs a single training step.
    Runs a training op if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.
    Also, syncs weights from online to target network if training steps is a
    multiple of target update period.
    """
    # Run a train op at the rate of self.update_period if enough training steps
    # have been run. This matches the Nature DQN behaviour.
    self._sess.run(self._train_op)
    if (self.summary_writer is not None and
        self.training_steps > 0 and
        self.training_steps % self.summary_writing_frequency == 0):
      summary = self._sess.run(self._merged_summaries)
      self.summary_writer.add_summary(summary, self.training_steps)

    if self.training_steps % self.target_update_period == 0:
      self._sess.run(self._sync_qt_ops)
    self.training_steps += 1
    
  def end_episode(self, reward):
    assert self.eval_mode, 'Eval mode is not set to be True.'
    super(FixedReplayQuantileAgent, self).end_episode(reward)

  def _build_replay_buffer(self, use_staging):
    """Creates the replay buffer used by the agent."""

    return fixed_replay_buffer.WrappedFixedReplayBuffer(
        data_dir=self._replay_data_dir,
        replay_suffix=self._replay_suffix,
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype.as_numpy_dtype)
  
  def initialize_CNN_from_reward_model(self, ckpt_path):
    #ckpt_path = "G:\\crowd_pbrl\\experiments\\large_short\\Pong\\1\\reward_model_bt_reward1"
    latest = tf.train.latest_checkpoint(ckpt_path)
    tf.logging.info('Loading CNN from {}'.format(latest))
    mapping = {
      'reward_model/Conv/kernel': 'Online/conv2d/kernel:0',
      'reward_model/Conv/bias': 'Online/conv2d/bias:0',
      'reward_model/Conv_1/kernel': 'Online/conv2d_1/kernel:0',
      'reward_model/Conv_1/bias': 'Online/conv2d_1/bias:0',
      'reward_model/Conv_2/kernel': 'Online/conv2d_2/kernel:0',
      'reward_model/Conv_2/bias': 'Online/conv2d_2/bias:0'
    }
    ops = []
    vars = tf.trainable_variables()
    for key, var_name in mapping.items():
      np_array = tf.train.load_variable(latest, key)
      var = [v for v in vars if v.name == var_name][0]
      ops.append(tf.assign(var, np_array))
    self._sess.run(ops)
    self._sess.run(self._sync_qt_ops)
    





