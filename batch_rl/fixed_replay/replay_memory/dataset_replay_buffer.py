'''
Descripttion: 
Author: CoeusZhang
Date: 2021-06-09 08:33:44
LastEditTime: 2021-06-09 23:47:35
'''
import numpy as np
import tensorflow.compat.v1 as tf
import collections
gfile = tf.gfile
from concurrent import futures
from . import nonverbose_circular_replay_buffer as circular_replay_buffer
from dopamine.agents.dqn import dqn_agent as dopamine_dqn
import gin
STACK_SIZE = dopamine_dqn.NATURE_DQN_STACK_SIZE
OBS_DTYPE = dopamine_dqn.NATURE_DQN_DTYPE.as_numpy_dtype
OBS_SHAPE = dopamine_dqn.NATURE_DQN_OBSERVATION_SHAPE
import gc

@gin.configurable
class DatasetReplayBuffer(object):
    def __init__(self, data_dir, buffer_shuffle_cache, batch_size):
        self._data_dir = data_dir
        self._loaded_buffers = False
        self.add_count = np.array(0)
        self.batch_size = batch_size
        self.buffer_shuffle_cache = buffer_shuffle_cache
        self.datasets = []

    def _load_buffer(self, suffix):
        replay_buffer = circular_replay_buffer.OutOfGraphReplayBuffer(
                            observation_shape=OBS_SHAPE, stack_size=STACK_SIZE, replay_capacity=1000000, batch_size=self.batch_size)
        replay_buffer.load(self._data_dir, suffix)
        tf.logging.info('Loaded replay buffer ckpt {} from {}'.format(
                        suffix, self._data_dir))
        return replay_buffer._store
    
    def _load_replay_buffers(self, num_buffers=1):
        if len(self.datasets) > 0:
            self.datasets = []
            gc.collect()
        ckpts = gfile.ListDirectory(self._data_dir)
        ckpt_counters = collections.Counter(
          [name.split('.')[-2] for name in ckpts])
        ckpt_suffixes = [x for x in ckpt_counters if ckpt_counters[x] in [6, 7]]
        ckpt_suffixes = np.random.choice(ckpt_suffixes, num_buffers, 
                                         replace=False)
        
        with futures.ThreadPoolExecutor(max_workers=2) as thread_pool_executor:
            buffer_futures = [thread_pool_executor.submit(
            self._load_buffer, suffix) for suffix in ckpt_suffixes]
        output_types = {"states": tf.uint8,
                      "next_states": tf.uint8,
                      "actions": tf.int32,
                      "rewards": tf.float32,
                      "terminals": tf.float32}
        output_shapes = {"states": tf.TensorShape(OBS_SHAPE\
                                        + (STACK_SIZE,)),
                       "next_states": tf.TensorShape(OBS_SHAPE\
                                        + (STACK_SIZE,)),
                       "actions": tf.TensorShape(()),
                       "rewards": tf.TensorShape(()),
                       "terminals": tf.TensorShape(())}
        for b in buffer_futures:
            buffer = b.result()
            dataset = tf.data.Dataset.from_generator(
                lambda: self.generate_data_from_records(buffer),
                output_types=output_types, output_shapes=output_shapes)
            dataset =dataset.shuffle(self.buffer_shuffle_cache).repeat()
            self.datasets.append(dataset)
        total_buffer = tf.data.Dataset.zip(tuple(self.datasets))
        total_buffer = total_buffer.shuffle(self.buffer_shuffle_cache).batch(self.batch_size//4).prefetch(1)
        iterator = tf.data.make_one_shot_iterator(total_buffer)
        ele = iterator.get_next()[0]
        return ele

    def generate_data_from_records(self, records):
        stack_obs = [np.zeros(OBS_SHAPE+(1,), dtype=np.uint8) for\
                    _ in range(STACK_SIZE)]
        next_stack_obs = [np.zeros(OBS_SHAPE+(1,), dtype=np.uint8) for\
                    _ in range(STACK_SIZE-1)]
        next_stack_obs.append(records['observation'][0][:, :, None])
        n_records = len(records['terminal'])
        r_id = 0
        while True:
            stack_obs.append(records['observation'][r_id][:, :, None])
            stack_obs.pop(0)
            next_r_id = min(r_id+1, n_records-1)
            next_stack_obs.append(records['observation'][next_r_id][:, :, None])
            next_stack_obs.pop(0)
            states = np.concatenate(stack_obs, 2)
            next_states = np.concatenate(next_stack_obs, 2)
            if records["terminal"][r_id] == 1:
                stack_obs = [np.zeros(OBS_SHAPE+(1,), dtype=np.uint8) for\
                            _ in range(STACK_SIZE)]
                next_stack_obs = [np.zeros(OBS_SHAPE+(1,), dtype=np.uint8) for\
                    _ in range(STACK_SIZE-1)]
                next_r_id = min(r_id+1, n_records-1)
                next_stack_obs.append(records['observation'][next_r_id][:, :, None])
            r_id += 1
            if r_id == n_records - 1:
                r_id = 0
                stack_obs = [np.zeros(OBS_SHAPE+(1,), dtype=np.uint8) for\
                    _ in range(STACK_SIZE)]
                next_stack_obs = [np.zeros(OBS_SHAPE+(1,), dtype=np.uint8) for\
                    _ in range(STACK_SIZE-1)]
                next_stack_obs.append(records['observation'][0][:, :, None])
            yield {"states": states,
                   "next_states": next_states,
                   "actions": records["action"][r_id],
                   "rewards": records["reward"][r_id],
                   "terminals": records["reward"][r_id]}