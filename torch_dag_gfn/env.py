"""Vectorised environment for sequential DAG construction (plain NumPy).

Each of `num_envs` parallel trajectories builds a DAG one edge at a time. The
action space is `N**2 + 1`: action `s * N + t` adds edge s -> t, and the final
action `N**2` terminates the trajectory. Acyclicity is enforced with an
incrementally-maintained transitive closure of the transpose adjacency, exactly
as in the original DAG-GFlowNet. Two differences from the JAX version:

  * a static `type_mask` forbids continuous -> categorical edges (the CLG
    constraint), folded into the action mask each step;
  * local scores are computed inline (no multiprocessing); the scorer memoises,
    so repeated (node, parents) queries are cheap.

`env.step` returns per-transition delta-scores log R(s_{t+1}) - log R(s_t), the
quantity consumed directly by the detailed-balance loss.
"""
import numpy as np

from copy import deepcopy


class DAGEnv:
    def __init__(self, num_envs, scorer, type_mask=None, max_parents=None):
        self.num_envs = num_envs
        self.scorer = scorer
        self.num_variables = scorer.num_variables
        self.max_parents = max_parents if (max_parents is not None) else self.num_variables

        if type_mask is None:
            type_mask = np.ones((self.num_variables, self.num_variables), dtype=np.bool_)
        self.type_mask = type_mask.astype(np.int_)

        self._state = None
        self._closure_T = None

    def reset(self):
        shape = (self.num_envs, self.num_variables, self.num_variables)
        closure_T = np.eye(self.num_variables, dtype=np.bool_)
        self._closure_T = np.tile(closure_T, (self.num_envs, 1, 1))
        self._state = {
            'adjacency': np.zeros(shape, dtype=np.int_),
            'mask': (1 - self._closure_T) * self.type_mask,
            'num_edges': np.zeros((self.num_envs,), dtype=np.int_),
            'score': np.zeros((self.num_envs,), dtype=np.float64),
            'order': np.full(shape, -1, dtype=np.int_),
        }
        return deepcopy(self._state)

    def step(self, actions):
        actions = np.asarray(actions)
        sources, targets = np.divmod(actions, self.num_variables)
        dones = (sources == self.num_variables)

        # Delta-scores must be computed against the current parents (before the
        # adjacency is mutated below).
        delta_scores = self._delta_scores(sources, targets, dones)

        src, tgt = sources[~dones], targets[~dones]
        if not np.all(self._state['mask'][~dones, src, tgt]):
            raise ValueError('Some actions are invalid: the edge is already '
                             'present, would create a cycle, or is forbidden by '
                             'the CLG type constraint.')

        # Update the adjacency matrices.
        self._state['adjacency'][~dones, src, tgt] = 1
        self._state['adjacency'][dones] = 0

        # Update the transitive closure of the transpose (rank-1 outer-product OR).
        source_rows = np.expand_dims(self._closure_T[~dones, src, :], axis=1)
        target_cols = np.expand_dims(self._closure_T[~dones, :, tgt], axis=2)
        self._closure_T[~dones] |= np.logical_and(source_rows, target_cols)
        self._closure_T[dones] = np.eye(self.num_variables, dtype=np.bool_)

        # Recompute the action mask: disallow present edges, cycle-inducing edges,
        # CLG-forbidden edges, and edges exceeding the max in-degree.
        num_parents = np.sum(self._state['adjacency'], axis=1, keepdims=True)
        self._state['mask'] = (
            (1 - (self._state['adjacency'] + self._closure_T))
            * self.type_mask
            * (num_parents < self.max_parents)
        )

        # Update the edge insertion order (used to reconstruct sampled DAGs).
        self._state['order'][~dones, src, tgt] = self._state['num_edges'][~dones]
        self._state['order'][dones] = -1

        # Update the edge count.
        self._state['num_edges'] += 1
        self._state['num_edges'][dones] = 0

        # Accumulate score(G) - score(G_0); reset terminated trajectories.
        self._state['score'] += delta_scores
        self._state['score'][dones] = 0

        return (deepcopy(self._state), delta_scores, dones, {})

    def _delta_scores(self, sources, targets, dones):
        deltas = np.zeros((self.num_envs,), dtype=np.float64)
        for i in range(self.num_envs):
            if dones[i]:
                continue
            target = int(targets[i])
            source = int(sources[i])
            adjacency = self._state['adjacency'][i]
            indices = tuple(np.nonzero(adjacency[:, target])[0].tolist())
            indices_after = tuple(sorted(indices + (source,)))

            before = self.scorer.local_score(target, indices)
            after = self.scorer.local_score(target, indices_after)
            deltas[i] = (after.score + after.prior) - (before.score + before.prior)
        return deltas
