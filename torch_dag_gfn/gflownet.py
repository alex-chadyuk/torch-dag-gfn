"""DAG-GFlowNet controller: sampling, detailed-balance loss, and training step.

Ports the policy/loss math from the original `utils/gflownet.py` and `gflownet.py`
to PyTorch. The controller holds an online network and a target network (a DQN-
style stabiliser copied from the online net every `update_target_every` steps).
Gradients flow only through the online net's terms; the target net's stop-log-
probability is detached.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F

from tqdm.auto import trange

from torch_dag_gfn.nets import GFlowNetPolicy


MASKED_VALUE = -1e5


def mask_logits(logits, masks):
    return masks * logits + (1. - masks) * MASKED_VALUE


def log_policy(logits, stop, masks):
    """Normalised log-policy over N**2 + 1 actions.

    P(add edge e) = sigmoid(-stop) * softmax(masked_logits)[e];  P(stop) = sigmoid(stop).
    `logits`, `masks` are (B, N**2); `stop` is (B, 1).
    """
    masked_logits = mask_logits(logits, masks)
    can_continue = masks.any(dim=-1, keepdim=True)

    logp_continue = F.logsigmoid(-stop) + F.log_softmax(masked_logits, dim=-1)
    logp_stop = F.logsigmoid(stop)

    # When no edge can be added, force the stop action.
    logp_continue = torch.where(can_continue, logp_continue,
                                torch.full_like(logp_continue, MASKED_VALUE))
    logp_stop = logp_stop * can_continue.to(logp_stop.dtype)

    return torch.cat((logp_continue, logp_stop), dim=-1)


def uniform_log_policy(masks):
    """Uniform log-policy over the valid edges plus the stop action."""
    num_edges = masks.sum(dim=-1, keepdim=True)
    logp_stop = -torch.log1p(num_edges)
    logp_continue = mask_logits(logp_stop, masks)
    return torch.cat((logp_continue, logp_stop), dim=-1)


def detailed_balance_loss(log_pi_t, log_pi_tp1, actions, delta_scores, num_edges, delta=1.):
    """Detailed-balance loss for the fully-terminable-states case.

    error = delta_scores + log P_B - log P_F
            + log P(s_f | s_t) - stop_grad(log P(s_f | s_{t+1}))
    with a uniform backward policy log P_B = -log(1 + num_edges). Huber loss over
    the batch. `actions`, `delta_scores`, `num_edges` are (B, 1).
    """
    log_pF = torch.gather(log_pi_t, -1, actions)
    log_pB = -torch.log1p(num_edges)

    error = ((delta_scores + log_pB - log_pF).squeeze(-1)
             + log_pi_t[:, -1] - log_pi_tp1[:, -1].detach())
    return F.huber_loss(error, torch.zeros_like(error), delta=delta)


def batch_random_choice(probs, masks, rng):
    """Inverse-CDF categorical sampling, falling back to the stop action if a
    masked-out action is drawn. `probs` is (B, N**2 + 1), `masks` is (B, N**2)."""
    batch_size, num_actions = probs.shape
    u = rng.random((batch_size, 1))
    cum_probs = np.cumsum(probs, axis=1)
    samples = np.sum(cum_probs < u, axis=1, keepdims=True)
    samples = np.minimum(samples, num_actions - 1)

    stop_col = np.ones((batch_size, 1), dtype=masks.dtype)
    masks_full = np.concatenate((masks, stop_col), axis=1)
    is_valid = np.take_along_axis(masks_full, samples, axis=1)
    stop_action = num_actions - 1  # = N**2
    samples = np.where(is_valid > 0, samples, stop_action)
    return np.squeeze(samples, axis=1)


class DAGGFlowNet:
    def __init__(self, num_variables, lr=1e-3, delta=1., update_target_every=1000,
                 device='cpu', **policy_kwargs):
        """`policy_kwargs` are forwarded verbatim to `GFlowNetPolicy` (embed_dim,
        num_heads, key_size, num_backbone, num_head_layers, widening_factor)."""
        self.device = torch.device(device)
        self.online = GFlowNetPolicy(num_variables, **policy_kwargs).to(self.device)
        self.target = copy.deepcopy(self.online)
        for param in self.target.parameters():
            param.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.delta = delta
        self.update_target_every = update_target_every
        self._steps = 0

    def _tensor(self, array, dtype=torch.float32):
        return torch.as_tensor(np.asarray(array), dtype=dtype, device=self.device)

    @torch.no_grad()
    def act(self, observations, epsilon, rng):
        adjacency = self._tensor(observations['adjacency'])
        masks = self._tensor(observations['mask'])
        batch_size = adjacency.shape[0]
        masks_flat = masks.reshape(batch_size, -1)

        logits, stop = self.online(adjacency, masks)
        log_pi = log_policy(logits, stop, masks_flat)
        log_uniform = uniform_log_policy(masks_flat)

        is_exploration = torch.as_tensor(
            rng.random((batch_size, 1)) < (1. - epsilon), device=self.device)
        log_pi = torch.where(is_exploration, log_uniform, log_pi)

        probs = torch.exp(log_pi).cpu().numpy()
        actions = batch_random_choice(probs, masks_flat.cpu().numpy(), rng)
        return actions

    def step(self, batch):
        adjacency = self._tensor(batch['adjacency'])
        mask = self._tensor(batch['mask'])
        next_adjacency = self._tensor(batch['next_adjacency'])
        next_mask = self._tensor(batch['next_mask'])
        actions = self._tensor(batch['actions'], dtype=torch.int64).view(-1, 1)
        delta_scores = self._tensor(batch['delta_scores']).view(-1, 1)
        num_edges = self._tensor(batch['num_edges']).view(-1, 1)
        batch_size = adjacency.shape[0]

        logits_t, stop_t = self.online(adjacency, mask)
        log_pi_t = log_policy(logits_t, stop_t, mask.reshape(batch_size, -1))

        with torch.no_grad():
            logits_tp1, stop_tp1 = self.target(next_adjacency, next_mask)
            log_pi_tp1 = log_policy(logits_tp1, stop_tp1,
                                    next_mask.reshape(batch_size, -1))

        loss = detailed_balance_loss(
            log_pi_t, log_pi_tp1, actions, delta_scores, num_edges, delta=self.delta)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.online.parameters():
            if param.grad is not None:
                torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
        self.optimizer.step()

        self._steps += 1
        if self._steps % self.update_target_every == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()


def posterior_estimate(gflownet, env, num_samples, rng, verbose=True):
    """Sample `num_samples` terminal DAGs from the trained policy (epsilon = 1,
    i.e. pure learned policy). Returns a (num_samples, N, N) array of adjacencies."""
    samples = []
    observations = env.reset()
    with trange(num_samples, disable=not verbose, desc='Posterior') as pbar:
        while len(samples) < num_samples:
            order = observations['order']
            actions = gflownet.act(observations, epsilon=1., rng=rng)
            observations, _, dones, _ = env.step(actions)
            samples.extend([order[i] for i, done in enumerate(dones) if done])
            pbar.update(min(num_samples - pbar.n, int(np.sum(dones))))
    orders = np.stack(samples[:num_samples], axis=0)
    return (orders >= 0).astype(np.int_)
