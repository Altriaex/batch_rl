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

"""End to end tests for FixedReplayRunner."""

import datetime
import os
import shutil



from absl import flags

from batch_rl.fixed_replay import train
import tensorflow.compat.v1 as tf

flags.DEFINE_string('exp_root', None, 'path to root directory for experiments.')
flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"third_party/py/dopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')
FLAGS = flags.FLAGS


class FixedReplayRunnerIntegrationTest(tf.test.TestCase):
  """Tests for Atari environment with various agents.

  """

  def setUp(self):
    super(FixedReplayRunnerIntegrationTest, self).setUp()
    print("***************")
    print(FLAGS.base_dir)
    FLAGS.base_dir = os.path.join(
        FLAGS.exp_root,
        datetime.datetime.utcnow().strftime('run_%Y_%m_%d_%H_%M_%S'))
    self._checkpoint_dir = os.path.join(FLAGS.base_dir, 'checkpoints')
    self._logging_dir = os.path.join(FLAGS.base_dir, 'logs')

  def quickFixedReplayREMFlags(self):
    """Assign flags for a quick run of FixedReplay agent."""
    FLAGS.gin_bindings = [
        "create_runner.schedule='continuous_train_and_eval'",
        'FixedReplayRunner.training_steps=100',
        'FixedReplayRunner.evaluation_steps=10',
        'FixedReplayRunner.num_iterations=1',
        'FixedReplayRunner.max_steps_per_episode=100',
    ]
    FLAGS.alsologtostderr = True
    FLAGS.gin_files = [os.path.join(*['batch_rl', 'fixed_replay','configs',
                                    'rem.gin'])]
    FLAGS.agent_name = 'multi_head_dqn'

  def verifyFilesCreated(self, base_dir):
    """Verify that files have been created."""
    # Check checkpoint files
    self.assertTrue(
        os.path.exists(os.path.join(self._checkpoint_dir, 'ckpt.0')))
    self.assertTrue(
        os.path.exists(os.path.join(self._checkpoint_dir, 'checkpoint')))
    self.assertTrue(
        os.path.exists(
            os.path.join(self._checkpoint_dir,
                         'sentinel_checkpoint_complete.0')))
    # Check log files
    self.assertTrue(os.path.exists(os.path.join(self._logging_dir, 'log_0')))

  def testIntegrationFixedReplayREM(self):
    """Test the FixedReplayMultiHeadDQN agent."""
    assert FLAGS.replay_dir is not None, 'Please provide a replay directory'
    tf.logging.info('####### Training the REM agent #####')
    tf.logging.info('####### REM base_dir: {}'.format(FLAGS.base_dir))
    tf.logging.info('#######  replay_dir: {}'.format(FLAGS.replay_dir))
    self.quickFixedReplayREMFlags()
    train.main([])
    self.verifyFilesCreated(FLAGS.base_dir)
    shutil.rmtree(FLAGS.base_dir)

if __name__ == '__main__':
  flags.mark_flag_as_required('exp_root')
  tf.test.main()
