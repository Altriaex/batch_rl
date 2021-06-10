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
import math
import os
from absl import logging
from batch_rl.fixed_replay.replay_memory import fixed_replay_buffer
from batch_rl.multi_head import quantile_agent
import gin
import tensorflow.compat.v1 as tf
from batch_rl.fixed_replay.replay_memory.dataset_replay_buffer import DatasetReplayBuffer
from dopamine.discrete_domains import atari_lib
from dopamine.agents.dqn import dqn_agent
from batch_rl.multi_head import atari_helpers
import numpy as np

class TargetCpHook(tf.estimator.SessionRunHook):
    def __init__(self, copy_op, copy_freq):
        self.step_t = tf.train.get_global_step()
        self.copy_op = copy_op
        self.copy_freq = copy_freq
    
    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self.step_t)

    def after_run(self, run_context, run_values):
        steps = run_values.results
        if steps % self.copy_freq == 0:
            run_context.session.run(self.copy_op)

@gin.configurable
class FixedReplayQuantileAgent(quantile_agent.QuantileAgent):
  """An implementation of the DQN agent with fixed replay buffer(s)."""

  def __init__(self,
               sess,
               num_actions,
               replay_data_dir,
               kappa = 1.0,
               init_checkpoint_dir=None,
               observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=atari_lib.NATURE_DQN_DTYPE,
               stack_size=atari_lib.NATURE_DQN_STACK_SIZE,
               network=atari_helpers.QuantileNetwork,
               num_atoms=200,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.1,
               epsilon_eval=0.05,
               epsilon_decay_period=1000000,
               replay_scheme='uniform',
               eval_mode=False,
               max_tf_checkpoints_to_keep=4,
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.00005, epsilon=0.0003125),
               summary_writer=None,
               summary_writing_frequency=500):
    self._num_atoms = num_atoms
    self.kappa = kappa
    '''logging.info('Creating %s agent with the following parameters:',
                 self.__class__.__name__)
    logging.info('\t gamma: %f', gamma)
    logging.info('\t update_horizon: %f', update_horizon)
    logging.info('\t min_replay_history: %d', min_replay_history)
    logging.info('\t update_period: %d', update_period)
    logging.info('\t target_update_period: %d', target_update_period)
    logging.info('\t epsilon_train: %f', epsilon_train)
    logging.info('\t epsilon_eval: %f', epsilon_eval)
    logging.info('\t epsilon_decay_period: %d', epsilon_decay_period)
    logging.info('\t tf_device: %s', tf_device)
    logging.info('\t optimizer: %s', optimizer)
    logging.info('\t max_tf_checkpoints_to_keep: %d',
                 max_tf_checkpoints_to_keep)'''
    self._replay_scheme = replay_scheme
    self.num_actions = num_actions
    self.observation_shape = tuple(observation_shape)
    self.observation_dtype = observation_dtype
    self.stack_size = stack_size
    self.network = network
    self.gamma = gamma
    self.update_horizon = update_horizon
    self.cumulative_gamma = math.pow(gamma, update_horizon)
    self.min_replay_history = min_replay_history
    self.target_update_period = target_update_period
    self.epsilon_fn = epsilon_fn
    self.epsilon_train = epsilon_train
    self.epsilon_eval = epsilon_eval
    self.epsilon_decay_period = epsilon_decay_period
    self.update_period = update_period
    self.eval_mode = eval_mode
    self.training_steps = 0
    self.optimizer = optimizer
    self.summary_writer = summary_writer
    self.summary_writing_frequency = summary_writing_frequency
    self._replay = DatasetReplayBuffer(replay_data_dir)
    self.max_tf_checkpoints_to_keep = max_tf_checkpoints_to_keep

      # Create a placeholder for the state input to the DQN network.
      # The last axis indicates the number of consecutive frames stacked.
    self.state_shape = (1,) + self.observation_shape + (stack_size,)
    self._replay_data_dir = replay_data_dir
    tf.logging.info('\t init_checkpoint_dir: %s', init_checkpoint_dir)
    if init_checkpoint_dir is not None:
      self._init_checkpoint_dir = os.path.join(
          init_checkpoint_dir, 'checkpoints')
    else:
      self._init_checkpoint_dir = None

  def build(self, sess):
    self._sess = sess
    self.state = np.zeros(self.state_shape)
    self.state_ph = tf.placeholder(
        self.observation_dtype, self.state_shape, name='state_ph')
    self.online_convnet = self._create_network(name='Online')
    self._net_outputs = self.online_convnet(self.state_ph)
    self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]

    if self.summary_writer is not None:
      # All tf.summaries should have been defined prior to running this.
      self._merged_summaries = tf.summary.merge_all()
    var_map = atari_lib.maybe_transform_variable_names(
        tf.global_variables())
    self._saver = tf.train.Saver(
        var_list=var_map, max_to_keep=self.max_tf_checkpoints_to_keep)
    # Variables to be initialized by the agent once it interacts with the
    # environment.
    self._observation = None
    self._last_observation = None
    

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

  def end_episode(self, reward):
    assert self.eval_mode, 'Eval mode is not set to be True.'
    super(FixedReplayQuantileAgent, self).end_episode(reward)
  
  def _train_step(self):
    raise NotImplementedError
  def _store_transition(self, last_observation, action, reward, is_terminal):
    raise NotImplementedError("Not Supported for DatasetReplayBuffer")

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    """Returns a self-contained bundle of the agent's state.
    This is used for checkpointing. It will return a dictionary containing all
    non-TensorFlow objects (to be saved into a file by the caller), and it saves
    all TensorFlow objects into a checkpoint file.
    Args:
      checkpoint_dir: str, directory where TensorFlow objects will be saved.
      iteration_number: int, iteration number to use for naming the checkpoint
        file.
    Returns:
      A dict containing additional Python objects to be checkpointed by the
        experiment. If the checkpoint directory does not exist, returns None.
    """
    if not tf.io.gfile.exists(checkpoint_dir):
      return None
    # Checkpoint the out-of-graph replay buffer.
    bundle_dictionary = {}
    bundle_dictionary['state'] = self.state
    #bundle_dictionary['training_steps'] = self.training_steps
    return bundle_dictionary

  def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
    """Restores the agent from a checkpoint.
    Restores the agent's Python objects to those specified in bundle_dictionary,
    and restores the TensorFlow objects to those specified in the
    checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
      agent's state.
    Args:
      checkpoint_dir: str, path to the checkpoint saved by tf.Save.
      iteration_number: int, checkpoint version, used when restoring the replay
        buffer.
      bundle_dictionary: dict, containing additional Python objects owned by
        the agent.
    Returns:
      bool, True if unbundling was successful.
    """
    if bundle_dictionary is not None:
      for key in self.__dict__:
        if key in bundle_dictionary:
          self.__dict__[key] = bundle_dictionary[key]
    elif not self.allow_partial_reload:
      return False
    else:
      logging.warning("Unable to reload the agent's parameters!")
    # Restore the agent's TensorFlow graph.
    latest_model = tf.train.latest_checkpoint(os.path.join(checkpoint_dir, "tf_ckpt"))
    self._saver.restore(self._sess, latest_model)
    logging.warning("Restored agent from {}".format(latest_model))
    return True

  def _build_train_op(self):
    raise NotImplementedError
  
  def _build_networks(self):
    raise NotImplementedError

  def _build_target_distribution(self, rewards, terminals, replay_next_target_net_outputs):
    batch_size = tf.shape(rewards)[0]
    # size of rewards: batch_size x 1
    rewards = rewards[:, None]
    # size of tiled_support: batch_size x num_atoms

    is_terminal_multiplier = 1. - tf.cast(terminals, tf.float32)
    # Incorporate terminal state to discount factor.
    # size of gamma_with_terminal: batch_size x 1
    gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
    gamma_with_terminal = gamma_with_terminal[:, None]

    # size of next_qt_argmax: 1 x batch_size
    next_qt_argmax = tf.argmax(
        replay_next_target_net_outputs.q_values, axis=1)[:, None]
    batch_indices = tf.range(tf.to_int64(batch_size))[:, None]
    # size of next_qt_argmax: batch_size x 2
    batch_indexed_next_qt_argmax = tf.concat(
        [batch_indices, next_qt_argmax], axis=1)
    # size of next_logits (next quantiles): batch_size x num_atoms
    next_logits = tf.gather_nd(
        replay_next_target_net_outputs.logits,
        batch_indexed_next_qt_argmax)
    return rewards + gamma_with_terminal * next_logits

  def build_input_fn(self, num_buffers):
    def input_fn():
      ele = self._replay._load_replay_buffers(num_buffers=num_buffers)
      return ele, None
    return input_fn

  def build_model_fn(self):
    def model_fn(features, labels, mode):
      states, actions = features["states"], features["actions"]
      rewards, terminals = features["rewards"], features["terminals"]
      next_states = features["next_states"]
      online_convnet = self._create_network(name='Online')
      target_convnet = self._create_network(name='Target')
      replay_net_outputs = online_convnet(states)
      replay_next_target_net_outputs = target_convnet(next_states)
      target_distribution = self._build_target_distribution(rewards, terminals, replay_next_target_net_outputs)
      target_distribution = tf.stop_gradient(target_distribution)

      # size of indices: batch_size x 1.
      indices = tf.range(tf.shape(replay_net_outputs.logits)[0])[:, None]
      # size of reshaped_actions: batch_size x 2.
      reshaped_actions = tf.concat([indices, actions[:, None]], 1)
      # For each element of the batch, fetch the logits for its selected action.
      chosen_action_logits = tf.gather_nd(replay_net_outputs.logits,
                                        reshaped_actions)
      bellman_errors = (target_distribution[:, None, :] -
                      chosen_action_logits[:, :, None])  # Input `u' of Eq. 9.
      huber_loss = (  # Eq. 9 of paper.
        tf.to_float(tf.abs(bellman_errors) <= self.kappa) *
        0.5 * bellman_errors ** 2 +
        tf.to_float(tf.abs(bellman_errors) > self.kappa) *
        self.kappa * (tf.abs(bellman_errors) - 0.5 * self.kappa))

      tau_hat = ((tf.range(self._num_atoms, dtype=tf.float32) + 0.5) /
               self._num_atoms)  # Quantile midpoints.  See Lemma 2 of paper.
      quantile_huber_loss = (  # Eq. 10 of paper.
        tf.abs(tau_hat[None, :, None] - tf.to_float(bellman_errors < 0)) *
        huber_loss)

      # Sum over tau dimension, average over target value dimension.
      loss = tf.reduce_sum(tf.reduce_mean(quantile_huber_loss, 2), 1)
      loss = tf.reduce_mean(loss)
      
      sync_qt_ops = self._build_sync_op()
      cp_hook = TargetCpHook(sync_qt_ops, self.target_update_period/self.update_period)
      step_t = tf.train.get_or_create_global_step()
      if mode ==  tf.estimator.ModeKeys.TRAIN:
        train_step =  self.optimizer.minimize(loss, global_step=step_t, var_list=tf.trainable_variables("Online"))
        estimator_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                    loss=loss,
                                                    train_op=train_step,
                                                    training_hooks=[cp_hook])
      else:
        raise NotImplementedError
      return estimator_spec
    return model_fn
