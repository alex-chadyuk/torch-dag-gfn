"""Train DAG-GFlowNet (PyTorch) on a synthetic mixed multinomial+Gaussian dataset.

Example:
    python train.py --num_variables 5 --num_edges 5 --num_samples 500 \
        --frac_discrete 0.4 --num_categories 3 \
        --num_iterations 5000 --batch_size 128 --prefill 500 --seed 0 \
        --output_folder results-smoke
"""
import json
import numpy as np
import torch

from argparse import ArgumentParser
from pathlib import Path
from tqdm.auto import trange

from torch_dag_gfn.data import sample_mixed_scm, type_mask_from_specs
from torch_dag_gfn.scores import BICScore, get_prior
from torch_dag_gfn.env import DAGEnv
from torch_dag_gfn.buffer import ReplayBuffer
from torch_dag_gfn.gflownet import DAGGFlowNet, posterior_estimate
from torch_dag_gfn.metrics import expected_shd, expected_edges, threshold_metrics


def epsilon_schedule(iteration, prefill, num_iterations, min_exploration):
    """Linear ramp of epsilon (= probability of following the learned policy)
    from 0 at `prefill` up to `1 - min_exploration` at `prefill + num_iterations/2`."""
    if iteration < prefill:
        return 0.
    transition = max(num_iterations // 2, 1)
    frac = min((iteration - prefill) / transition, 1.)
    return frac * (1. - min_exploration)


def main(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Generate the synthetic mixed dataset.
    ground_truth, data, var_specs = sample_mixed_scm(
        num_variables=args.num_variables,
        num_edges=args.num_edges,
        num_samples=args.num_samples,
        frac_discrete=args.frac_discrete,
        num_categories=args.num_categories,
        obs_noise=args.obs_noise,
        rng=rng,
    )

    # Scorer, environment, replay buffer, GFlowNet.
    prior = get_prior(args.prior)
    scorer = BICScore(data, var_specs, prior)
    type_mask = type_mask_from_specs(var_specs)
    env = DAGEnv(args.num_envs, scorer, type_mask=type_mask)
    replay = ReplayBuffer(args.replay_capacity, args.num_variables)
    gflownet = DAGGFlowNet(
        args.num_variables,
        lr=args.lr,
        delta=args.delta,
        update_target_every=args.update_target_every,
        device=args.device,
    )

    # Training loop.
    observations = env.reset()
    with trange(args.prefill + args.num_iterations, desc='Training') as pbar:
        for iteration in pbar:
            epsilon = epsilon_schedule(
                iteration, args.prefill, args.num_iterations, args.min_exploration)
            actions = gflownet.act(observations, epsilon, rng)
            next_observations, delta_scores, dones, _ = env.step(actions)
            replay.add(observations, actions, next_observations, delta_scores, dones)
            observations = next_observations

            if iteration >= args.prefill and len(replay) >= args.batch_size:
                batch = replay.sample(args.batch_size, rng)
                loss = gflownet.step(batch)
                pbar.set_postfix(loss=f'{loss:.3f}', epsilon=f'{epsilon:.2f}')

    # Posterior estimate + metrics.
    posterior = posterior_estimate(
        gflownet, env, args.num_samples_posterior, rng, verbose=not args.quiet)

    num_true_edges = int(ground_truth.sum())
    results = {
        'expected_shd': expected_shd(posterior, ground_truth),
        'expected_edges': expected_edges(posterior),
        'num_true_edges': num_true_edges,
        'empty_graph_shd': float(num_true_edges),  # E-SHD baseline of the empty DAG
        'num_categorical': int(sum(s.is_categorical for s in var_specs)),
        **threshold_metrics(posterior, ground_truth),
    }
    print('\nResults:', json.dumps(results, indent=2))

    # Save artifacts.
    output = Path(args.output_folder)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / 'arguments.json', 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    np.savez(
        output / 'data.npz',
        data=data,
        adjacency=ground_truth,
        kinds=np.array([s.kind for s in var_specs]),
        cardinalities=np.array([s.cardinality for s in var_specs]),
    )
    np.save(output / 'ground_truth.npy', ground_truth)
    np.save(output / 'posterior.npy', posterior)
    torch.save(gflownet.online.state_dict(), output / 'model.pt')
    with open(output / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    parser = ArgumentParser(description='PyTorch DAG-GFlowNet (mixed synthetic data).')

    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--num_variables', type=int, default=5)
    data_group.add_argument('--num_edges', type=int, default=5,
        help='Average number of edges in the ground-truth DAG.')
    data_group.add_argument('--num_samples', type=int, default=500)
    data_group.add_argument('--frac_discrete', type=float, default=0.5,
        help='Probability a CLG-eligible node is made categorical.')
    data_group.add_argument('--num_categories', type=int, default=3,
        help='Number of states for categorical variables.')
    data_group.add_argument('--obs_noise', type=float, default=0.1,
        help='Std of the Gaussian observation noise.')

    opt_group = parser.add_argument_group('Optimization')
    opt_group.add_argument('--lr', type=float, default=1e-3)
    opt_group.add_argument('--delta', type=float, default=1.,
        help='Delta for the Huber loss.')
    opt_group.add_argument('--batch_size', type=int, default=128)
    opt_group.add_argument('--num_iterations', type=int, default=5000)
    opt_group.add_argument('--num_envs', type=int, default=8)

    replay_group = parser.add_argument_group('Replay buffer')
    replay_group.add_argument('--replay_capacity', type=int, default=100_000)
    replay_group.add_argument('--prefill', type=int, default=500)

    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--min_exploration', type=float, default=0.1)
    misc.add_argument('--update_target_every', type=int, default=1000)
    misc.add_argument('--num_samples_posterior', type=int, default=1000)
    misc.add_argument('--prior', type=str, default='uniform',
        choices=['uniform', 'erdos_renyi', 'edge', 'fair'])
    misc.add_argument('--seed', type=int, default=0)
    misc.add_argument('--device', type=str, default='cpu')
    misc.add_argument('--output_folder', type=str, default='output')
    misc.add_argument('--quiet', action='store_true')

    main(parser.parse_args())
