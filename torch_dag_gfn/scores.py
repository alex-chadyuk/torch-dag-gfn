"""Decomposable BIC score for mixed multinomial+Gaussian data, plus graph priors.

The score of a DAG G factorises as a sum of per-node local scores
`local(i, parents(i))`, so adding a single edge changes exactly one term -- this
is what lets the environment return delta-scores. Under the conditional-Gaussian
(CLG) assumption, categorical nodes have only categorical parents (enforced by
the environment's edge-mask and the generator), so each local score is a
closed-form BIC:

  * Gaussian node: OLS regression on [intercept, continuous parents,
    one-hot(categorical parents)], scored by the Gaussian log-likelihood at the
    MLE minus the BIC complexity penalty.
  * Categorical node: multinomial log-likelihood from contingency counts over
    the joint categorical-parent configuration, minus the BIC penalty.

The reward used by the GFlowNet is R(G) = exp(sum_i local(i) + log p(G)), so
`local(i).score` (data term) and `local(i).prior` (graph-prior term) are kept
separate, mirroring the original DAG-GFlowNet score interface.
"""
import math
import numpy as np

from collections import namedtuple

from torch_dag_gfn.data import GAUSSIAN, CATEGORICAL


LocalScore = namedtuple('LocalScore', ['key', 'score', 'prior'])


# -- Graph priors log p(G) (ported verbatim from the JAX implementation) --------

class BasePrior:
    """Modular prior over graphs: returns the log p(G) contribution of a node
    given its number of parents."""
    def __init__(self, num_variables=None):
        self._num_variables = num_variables
        self._log_prior = None

    def __call__(self, num_parents):
        return self.log_prior[num_parents]

    @property
    def log_prior(self):
        raise NotImplementedError

    @property
    def num_variables(self):
        if self._num_variables is None:
            raise RuntimeError('The number of variables is not defined.')
        return self._num_variables

    @num_variables.setter
    def num_variables(self, value):
        self._num_variables = value


class UniformPrior(BasePrior):
    @property
    def log_prior(self):
        if self._log_prior is None:
            self._log_prior = np.zeros((self.num_variables,))
        return self._log_prior


class ErdosRenyiPrior(BasePrior):
    def __init__(self, num_variables=None, num_edges_per_node=1.):
        super().__init__(num_variables)
        self.num_edges_per_node = num_edges_per_node

    @property
    def log_prior(self):
        if self._log_prior is None:
            num_edges = self.num_variables * self.num_edges_per_node
            p = num_edges / ((self.num_variables * (self.num_variables - 1)) // 2)
            all_parents = np.arange(self.num_variables)
            self._log_prior = (all_parents * math.log(p)
                + (self.num_variables - all_parents - 1) * math.log1p(-p))
        return self._log_prior


class EdgePrior(BasePrior):
    def __init__(self, num_variables=None, beta=1.):
        super().__init__(num_variables)
        self.beta = beta

    @property
    def log_prior(self):
        if self._log_prior is None:
            self._log_prior = np.arange(self.num_variables) * math.log(self.beta)
        return self._log_prior


class FairPrior(BasePrior):
    @property
    def log_prior(self):
        if self._log_prior is None:
            from scipy.special import gammaln
            all_parents = np.arange(self.num_variables)
            self._log_prior = (
                - gammaln(self.num_variables + 1)
                + gammaln(self.num_variables - all_parents + 1)
                + gammaln(all_parents + 1)
            )
        return self._log_prior


_PRIORS = {
    'uniform': UniformPrior,
    'erdos_renyi': ErdosRenyiPrior,
    'edge': EdgePrior,
    'fair': FairPrior,
}


def get_prior(name, num_variables=None, **kwargs):
    return _PRIORS[name](num_variables=num_variables, **kwargs)


# -- Mixed BIC score ------------------------------------------------------------

class BICScore:
    """Mixed multinomial+Gaussian BIC score.

    Parameters
    ----------
    data : (num_samples, N) ndarray
        Gaussian columns hold real values; categorical columns hold integer
        state codes (stored as floats).
    var_specs : list of VarSpec
        Per-variable type and cardinality.
    prior : BasePrior
    variance_floor : float
        Lower bound on the residual variance of a Gaussian node (numerical
        guard against a perfect fit).
    """
    def __init__(self, data, var_specs, prior, variance_floor=1e-8):
        self.data = np.asarray(data, dtype=np.float64)
        self.var_specs = list(var_specs)
        self.num_variables = self.data.shape[1]
        self.num_samples = self.data.shape[0]
        self.prior = prior
        self.prior.num_variables = self.num_variables
        self.variance_floor = variance_floor
        self._log_n = math.log(self.num_samples)

        # Precompute integer codes for categorical columns.
        self._codes = [None] * self.num_variables
        for i, spec in enumerate(self.var_specs):
            if spec.is_categorical:
                self._codes[i] = self.data[:, i].astype(np.int_)

        self._cache = {}

    # -- environment-facing interface (matches the original scorer) -------------

    def get_local_scores(self, target, indices, indices_after=None):
        all_indices = indices if (indices_after is None) else indices_after
        local_score_after = self.local_score(target, all_indices)
        if indices_after is not None:
            local_score_before = self.local_score(target, indices)
        else:
            local_score_before = None
        return (local_score_before, local_score_after)

    def local_score(self, target, parents):
        key = (target, tuple(parents))
        if key not in self._cache:
            spec = self.var_specs[target]
            if spec.is_categorical:
                score = self._categorical_local_score(target, key[1])
            else:
                score = self._gaussian_local_score(target, key[1])
            self._cache[key] = LocalScore(
                key=key, score=score, prior=self.prior(len(key[1])))
        return self._cache[key]

    # -- per-type local scores --------------------------------------------------

    def _categorical_local_score(self, target, parents):
        spec = self.var_specs[target]
        K = spec.cardinality
        y = self._codes[target]

        cards = [self.var_specs[p].cardinality for p in parents]
        num_configs = int(np.prod(cards)) if cards else 1

        # Joint parent configuration index (mixed radix), CLG => all discrete.
        config = np.zeros(self.num_samples, dtype=np.int_)
        for p, card in zip(parents, cards):
            config = config * card + self._codes[p]

        counts = np.bincount(
            config * K + y, minlength=num_configs * K
        ).reshape(num_configs, K).astype(np.float64)

        conds = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_terms = np.where(counts > 0, np.log(counts) - np.log(conds), 0.)
        loglik = float(np.sum(counts * log_terms))

        num_params = num_configs * (K - 1)
        return loglik - 0.5 * num_params * self._log_n

    def _gaussian_local_score(self, target, parents):
        y = self.data[:, target]
        n = self.num_samples

        columns = [np.ones((n, 1))]  # intercept
        for p in parents:
            spec = self.var_specs[p]
            if spec.is_categorical:
                # One-hot with the first level dropped (kept full-rank vs intercept).
                onehot = np.eye(spec.cardinality)[self._codes[p]][:, 1:]
                columns.append(onehot)
            else:
                columns.append(self.data[:, p:p + 1])
        X = np.concatenate(columns, axis=1)

        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
        sigma2 = max(float(residuals @ residuals) / n, self.variance_floor)

        loglik = -0.5 * n * (math.log(2. * math.pi * sigma2) + 1.)
        num_params = X.shape[1] + 1  # regression coefficients + variance
        return loglik - 0.5 * num_params * self._log_n

    # -- convenience: full-graph score (used by sanity checks) ------------------

    def structure_score(self, adjacency):
        """Total score sum_i (data local score + log prior) for a DAG."""
        total = 0.
        for target in range(self.num_variables):
            parents = tuple(np.nonzero(adjacency[:, target])[0].tolist())
            local = self.local_score(target, parents)
            total += local.score + local.prior
        return total
