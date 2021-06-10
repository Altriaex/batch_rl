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

"""Runner for experiments with a fixed replay buffer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as osp
import time
import tensorflow.compat.v1 as tf
from absl import logging
from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment
from dopamine.discrete_domains import atari_lib
import gin
import gc


@gin.configurable
class FixedReplayRunner(run_experiment.Runner):
  """Object that handles running Dopamine experiments with fixed replay buffer."""
  def __init__(self,
               base_dir,
               create_agent_fn,
               create_environment_fn=atari_lib.create_atari_environment,
               checkpoint_file_prefix='ckpt',
               logging_file_prefix='log',
               log_every_n=1,
               num_iterations=200,
               training_steps=250000,
               evaluation_steps=125000,
               max_steps_per_episode=27000,
               clip_rewards=True,
               n_cpu=4,
               gpu_id=0):
    assert base_dir is not None
    tf.disable_v2_behavior()

    self._logging_file_prefix = logging_file_prefix
    self._log_every_n = log_every_n
    self._num_iterations = num_iterations
    self._training_steps = training_steps
    self._evaluation_steps = evaluation_steps
    self._max_steps_per_episode = max_steps_per_episode
    self._base_dir = base_dir
    self._clip_rewards = clip_rewards
    self._create_directories()
    self._summary_writer = tf.summary.FileWriter(self._base_dir)
    self.create_agent_fn = create_agent_fn
    self.checkpoint_file_prefix = checkpoint_file_prefix
    self._environment = create_environment_fn()
    self.sess_config = tf.ConfigProto(inter_op_parallelism_threads=2,
                            intra_op_parallelism_threads=n_cpu)
    # Allocate only subset of the GPU memory as needed which allows for running
    # multiple agents/workers on the same GPU.
    self.sess_config.gpu_options.allow_growth = True
    self.sess_config.gpu_options.visible_device_list = str(gpu_id)
    self._start_iteration = 0

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir,
                                                   checkpoint_file_prefix)
    latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
        self._checkpoint_dir)
    if latest_checkpoint_version >= 0:
      experiment_data = self._checkpointer.load_checkpoint(
          latest_checkpoint_version)
      if self._agent.unbundle(
          self._checkpoint_dir, latest_checkpoint_version, experiment_data):
        if experiment_data is not None:
          assert 'logs' in experiment_data
          assert 'current_iteration' in experiment_data
          self._logger.data = experiment_data['logs']
          self._start_iteration = experiment_data['current_iteration'] + 1
        #logging.info('Reloaded checkpoint and will start from iteration %d',
        #             self._start_iteration)

  def _run_train_phase(self):
    raise NotImplementedError

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    logging.info('Beginning training...')
    if self._num_iterations <= self._start_iteration:
      logging.warning('num_iterations (%d) < start_iteration(%d)',
                      self._num_iterations, self._start_iteration)
      return
    run_config = tf.estimator.RunConfig(
                            log_step_count_steps=10000,
                            keep_checkpoint_max=4,
                            save_checkpoints_steps=250000,
                            session_config=self.sess_config,
                            save_summary_steps=10000)
    self._agent = self.create_agent_fn(None, self._environment,
                                  summary_writer=self._summary_writer)
    estimator = tf.estimator.Estimator(self._agent.build_model_fn(),    
                                       model_dir=osp.join(self._checkpoint_dir,
                                                          'tf_ckpt'),
                                       config=run_config)
    for iteration in range(self._start_iteration, self._num_iterations):
      statistics = iteration_statistics.IterationStatistics()
      logging.info('Starting iteration %d', iteration)
      # The estimator load 4 buffers instead of 5
      # It trains the agent for training_steps/self._agent.update_period
      estimator.train(input_fn=self._agent.build_input_fn(4), steps=self._training_steps/self._agent.update_period)
      tf.reset_default_graph()
      self._sess = tf.Session('', config=self.sess_config)
      self._agent.build(self._sess)
      self._sess.run(tf.global_variables_initializer())
      self._initialize_checkpointer_and_maybe_resume(self.checkpoint_file_prefix)
      num_episodes_eval, average_reward_eval = self._run_eval_phase(statistics)
      self._save_tensorboard_summaries(iteration, num_episodes_eval, average_reward_eval)
      self._log_experiment(iteration, statistics.data_lists)
      self._checkpoint_experiment(iteration)
      self._sess.close()
      tf.reset_default_graph()
    self._summary_writer.flush()
    
  def _checkpoint_experiment(self, iteration):
    """Checkpoint experiment data.
    Args:
      iteration: int, iteration number for checkpointing.
    """
    experiment_data = self._agent.bundle_and_checkpoint(self._checkpoint_dir,
                                                        iteration)
    if experiment_data:
      experiment_data['current_iteration'] = iteration
      experiment_data['logs'] = self._logger.data
      self._checkpointer.save_checkpoint(iteration, experiment_data)

  def _save_tensorboard_summaries(self, iteration,
                                  num_episodes_eval,
                                  average_reward_eval):
    """Save statistics as tensorboard summaries.

    Args:
      iteration: int, The current iteration number.
      num_episodes_eval: int, number of evaluation episodes run.
      average_reward_eval: float, The average evaluation reward.
    """
    summary = tf.Summary(value=[
        tf.Summary.Value(tag='Eval/NumEpisodes',
                         simple_value=num_episodes_eval),
        tf.Summary.Value(tag='Eval/AverageReturns',
                         simple_value=average_reward_eval)
    ])
    self._summary_writer.add_summary(summary, iteration)

