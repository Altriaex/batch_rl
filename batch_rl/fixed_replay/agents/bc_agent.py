'''
Descripttion: 
Author: CoeusZhang
Date: 2022-04-27 10:27:01
LastEditTime: 2022-04-27 14:58:27
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from batch_rl.fixed_replay.replay_memory import fixed_replay_buffer
from batch_rl.multi_head import atari_helpers
import gin
import tensorflow.compat.v1 as tf

from dopamine.agents.dqn import dqn_agent
from dopamine.agents.rainbow import rainbow_agent
import gin
import tensorflow.compat.v1 as tf


@gin.configurable
class BehavioralCloningAgent(rainbow_agent.RainbowAgent):
    """An extension of Rainbow to perform quantile regression."""
    def __init__(self,
               sess,
               num_actions,
               replay_data_dir,
               replay_suffix=None,
               init_checkpoint_dir=None,
               network=atari_helpers.NatureDQNNetwork,
               min_replay_history=50000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.1,
               epsilon_eval=0.05,
               epsilon_decay_period=1000000,
               tf_device='/cpu:0',
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.00005, epsilon=0.0003125),
               summary_writer=None,
               summary_writing_frequency=500):
        assert replay_data_dir is not None
        # Set replay_log_dir before calling parent's initializer
        tf.logging.info(
            'Creating FixedReplay BC Agent with replay directory: %s', replay_data_dir)
        tf.logging.info('\t init_checkpoint_dir: %s', init_checkpoint_dir)
        tf.logging.info('\t replay_suffix %s', replay_suffix)
        self._replay_data_dir = replay_data_dir
        self._replay_suffix = replay_suffix
        if init_checkpoint_dir is not None:
            self._init_checkpoint_dir = os.path.join(
                init_checkpoint_dir, 'checkpoints')
        else:
            self._init_checkpoint_dir = None
        super(BehavioralCloningAgent, self).__init__(
            sess=sess,
            num_actions=num_actions,
            network=network,
            min_replay_history=min_replay_history,
            epsilon_fn=epsilon_fn,
            epsilon_train=epsilon_train,
            epsilon_eval=epsilon_eval,
            epsilon_decay_period=epsilon_decay_period,
            tf_device=tf_device,
            optimizer=optimizer,
            summary_writer=summary_writer,
            summary_writing_frequency=summary_writing_frequency)

    def _create_network(self, name):
        """Builds a Quantile ConvNet.

        Equivalent to Rainbow ConvNet, only now the output logits are interpreted
        as quantiles.

        Args:
        name: str, this name is passed to the tf.keras.Model and used to create
            variable scope under the hood by the tf.keras.Model.

        Returns:
        network: tf.keras.Model, the network instantiated by the Keras model.
        """
        network = self.network(self.num_actions, name=name)
        return network

    def _build_train_op(self):
        """Builds a training op.

        Returns:
        train_op: An op performing one step of training.
        """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self._replay.actions,
                    logits=self._replay_net_outputs.q_values)
        update_priorities_op = tf.no_op()
        with tf.control_dependencies([update_priorities_op]):
            if self.summary_writer is not None:
                with tf.variable_scope('Losses'):
                    tf.summary.scalar('CrossEntropyLoss', tf.reduce_mean(loss))
        return self.optimizer.minimize(tf.reduce_mean(loss)), loss
    def step(self, reward, observation):
        self._record_observation(observation)
        self.action = self._select_action()
        return self.action

    def _train_multiple_steps(self, total_steps, num_buffers=2):
        """Run multiple training steps to speedup training.
        """
        self._replay.memory.reload_buffer(num_buffers=num_buffers)  
        n_steps_sofar = self.training_steps
        while self.training_steps < n_steps_sofar + total_steps:
            for _ in range(self.target_update_period):
                # run multiple training steps without interruption
                self._sess.run(self._train_op)
            self._sess.run(self._sync_qt_ops)
            self.training_steps += self.target_update_period
        if self.summary_writer is not None:
            summary = self._sess.run(self._merged_summaries)
            self.summary_writer.add_summary(summary, self.training_steps)
    def end_episode(self, reward):
        assert self.eval_mode, 'Eval mode is not set to be True.'
        super(BehavioralCloningAgent, self).end_episode(reward)

    def _build_replay_buffer(self, use_staging):
        return fixed_replay_buffer.WrappedFixedReplayBuffer(
            data_dir=self._replay_data_dir,
            replay_suffix=self._replay_suffix,
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            use_staging=use_staging,
            update_horizon=self.update_horizon,
            gamma=self.gamma,
            observation_dtype=self.observation_dtype.as_numpy_dtype)