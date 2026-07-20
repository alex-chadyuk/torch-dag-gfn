"""Replay buffer for off-policy GFlowNet training.

A fixed-capacity circular buffer of transitions. Unlike the original, adjacency
and mask matrices are stored as plain float32 arrays (no bit-packing) -- at the
synthetic scale considered here the memory cost is negligible and the code is
much simpler. Terminated transitions are not stored (their delta-score is 0 and
they carry no learning signal for the detailed-balance loss).
"""
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, num_variables):
        self.capacity = capacity
        self.num_variables = num_variables
        shape = (capacity, num_variables, num_variables)

        self._adjacency = np.zeros(shape, dtype=np.float32)
        self._mask = np.zeros(shape, dtype=np.float32)
        self._next_adjacency = np.zeros(shape, dtype=np.float32)
        self._next_mask = np.zeros(shape, dtype=np.float32)
        self._num_edges = np.zeros((capacity,), dtype=np.int64)
        self._actions = np.zeros((capacity,), dtype=np.int64)
        self._delta_scores = np.zeros((capacity,), dtype=np.float32)

        self._index = 0
        self._is_full = False

    def add(self, observations, actions, next_observations, delta_scores, dones):
        keep = ~dones
        num_samples = int(np.sum(keep))
        if num_samples == 0:
            return

        add_idx = np.arange(self._index, self._index + num_samples) % self.capacity
        self._is_full |= (self._index + num_samples >= self.capacity)
        self._index = (self._index + num_samples) % self.capacity

        self._adjacency[add_idx] = observations['adjacency'][keep]
        self._mask[add_idx] = observations['mask'][keep]
        self._next_adjacency[add_idx] = next_observations['adjacency'][keep]
        self._next_mask[add_idx] = next_observations['mask'][keep]
        self._num_edges[add_idx] = observations['num_edges'][keep]
        self._actions[add_idx] = actions[keep]
        self._delta_scores[add_idx] = delta_scores[keep]

    def sample(self, batch_size, rng):
        indices = rng.choice(len(self), size=batch_size, replace=False)
        return {
            'adjacency': self._adjacency[indices],
            'mask': self._mask[indices],
            'next_adjacency': self._next_adjacency[indices],
            'next_mask': self._next_mask[indices],
            'num_edges': self._num_edges[indices],
            'actions': self._actions[indices],
            'delta_scores': self._delta_scores[indices],
        }

    def __len__(self):
        return self.capacity if self._is_full else self._index
