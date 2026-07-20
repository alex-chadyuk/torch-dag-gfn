"""Synthetic mixed-type SCM generator.

Generates a random Erdos-Renyi DAG whose nodes are a mix of Gaussian and
categorical (multinomial) variables, then samples data by ancestral sampling.

We enforce the conditional-Gaussian (CLG) structure: a categorical node may
only have categorical parents. This is the standard well-definedness condition
for mixed conditional-Gaussian networks, and it is what the BIC score in
`scores.py` assumes. Types are assigned in topological order so the constraint
holds by construction (a node with any Gaussian parent is forced to be Gaussian).
"""
import numpy as np

from dataclasses import dataclass


GAUSSIAN = 'gaussian'
CATEGORICAL = 'categorical'


@dataclass(frozen=True)
class VarSpec:
    """Type specification for a single variable.

    kind : 'gaussian' or 'categorical'.
    cardinality : number of states K (categorical only; 0 for Gaussian).
    """
    kind: str
    cardinality: int = 0

    @property
    def is_categorical(self):
        return self.kind == CATEGORICAL


def sample_er_dag(num_variables, num_edges, rng):
    """Sample an Erdos-Renyi DAG. Returns (adjacency, order).

    `adjacency[i, j] == 1` denotes an edge i -> j. `order` is a topological
    order (parents before children): every edge goes from an earlier to a
    later node in `order`, so the graph is acyclic by construction.
    """
    max_edges = num_variables * (num_variables - 1) // 2
    p = 0. if max_edges == 0 else min(num_edges / max_edges, 1.)

    order = rng.permutation(num_variables)
    adjacency = np.zeros((num_variables, num_variables), dtype=np.int_)
    for a in range(num_variables):
        for b in range(a + 1, num_variables):
            if rng.random() < p:
                adjacency[order[a], order[b]] = 1  # edge order[a] -> order[b]
    return adjacency, order


def assign_types(adjacency, order, frac_discrete, num_categories, rng):
    """Assign a VarSpec to each node, enforcing the CLG constraint.

    A node may become categorical only if all of its parents are categorical.
    Nodes are visited in topological order, so a node's parents are already
    typed when it is visited.
    """
    num_variables = adjacency.shape[0]
    specs = [None] * num_variables
    for node in order:
        parents = np.nonzero(adjacency[:, node])[0]
        eligible = all(specs[p].is_categorical for p in parents)
        if eligible and (rng.random() < frac_discrete):
            specs[node] = VarSpec(CATEGORICAL, cardinality=num_categories)
        else:
            specs[node] = VarSpec(GAUSSIAN)
    return specs


def _parent_config_index(codes, cardinalities):
    """Mixed-radix index of joint categorical-parent configurations.

    codes : (n, m) int array of parent state codes.
    cardinalities : list of m parent cardinalities.
    Returns (n,) int array in [0, prod(cardinalities)).
    """
    if codes.shape[1] == 0:
        return np.zeros(codes.shape[0], dtype=np.int_)
    index = np.zeros(codes.shape[0], dtype=np.int_)
    for col, card in enumerate(cardinalities):
        index = index * card + codes[:, col]
    return index


def sample_mixed_scm(
        num_variables,
        num_edges,
        num_samples,
        frac_discrete=0.5,
        num_categories=3,
        obs_noise=0.1,
        edge_scale=1.0,
        logit_scale=2.0,
        rng=None,
    ):
    """Sample a mixed-type synthetic dataset from a random SCM.

    Parameters
    ----------
    num_variables, num_edges, num_samples : int
        Number of variables, target average number of edges, number of rows.
    frac_discrete : float
        Probability a CLG-eligible node is made categorical.
    num_categories : int
        Number of states K for categorical variables.
    obs_noise : float
        Standard deviation of the Gaussian noise.
    edge_scale : float
        Scale of the (Normal) linear coefficients / per-category effects for
        Gaussian nodes.
    logit_scale : float
        Scale of the (Normal) logits defining categorical CPTs.
    rng : np.random.Generator

    Returns
    -------
    adjacency : (N, N) int ndarray -- ground-truth DAG (i -> j).
    data : (num_samples, N) float64 ndarray -- Gaussian columns are real,
        categorical columns hold integer state codes stored as floats.
    var_specs : list of VarSpec.
    """
    if rng is None:
        rng = np.random.default_rng()

    adjacency, order = sample_er_dag(num_variables, num_edges, rng)
    specs = assign_types(adjacency, order, frac_discrete, num_categories, rng)

    data = np.zeros((num_samples, num_variables), dtype=np.float64)
    for node in order:  # topological order: parents already sampled
        parents = np.nonzero(adjacency[:, node])[0]
        spec = specs[node]

        if spec.is_categorical:
            # CLG guarantees all parents are categorical.
            cards = [specs[p].cardinality for p in parents]
            if len(parents) == 0:
                codes = np.zeros((num_samples, 0), dtype=np.int_)
            else:
                codes = data[:, parents].astype(np.int_)
            config = _parent_config_index(codes, cards)
            num_configs = int(np.prod(cards)) if cards else 1

            # One random probability vector per parent configuration.
            logits = rng.normal(0., logit_scale, size=(num_configs, spec.cardinality))
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)

            cdf = np.cumsum(probs[config], axis=1)
            u = rng.random((num_samples, 1))
            data[:, node] = (u > cdf).sum(axis=1)
        else:
            # Gaussian node: linear in continuous parents, per-category effect
            # for categorical parents, plus Gaussian noise (zero intercept).
            mean = np.zeros(num_samples)
            for p in parents:
                if specs[p].is_categorical:
                    effects = rng.normal(0., edge_scale, size=specs[p].cardinality)
                    mean += effects[data[:, p].astype(np.int_)]
                else:
                    weight = rng.normal(0., edge_scale)
                    mean += weight * data[:, p]
            data[:, node] = mean + rng.normal(0., obs_noise, size=num_samples)

    return adjacency, data, specs


def type_mask_from_specs(var_specs):
    """Static (N, N) bool edge-mask enforcing CLG.

    `mask[i, j]` is False iff an edge i -> j would give a categorical node j a
    Gaussian parent i (forbidden under CLG). All other edges are allowed.
    """
    num_variables = len(var_specs)
    mask = np.ones((num_variables, num_variables), dtype=np.bool_)
    for i, spec_i in enumerate(var_specs):
        if not spec_i.is_categorical:  # i is Gaussian
            for j, spec_j in enumerate(var_specs):
                if spec_j.is_categorical:  # j is categorical
                    mask[i, j] = False
    return mask
