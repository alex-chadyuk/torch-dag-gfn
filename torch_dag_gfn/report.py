"""Post-run report: render a self-contained `README.md` inside a run directory.

The report is built purely from the artifacts `train.py` writes, so it can be
(re)generated for any existing run without retraining:

    python -m torch_dag_gfn.report out-26-07-20-16-22

Required artifacts: `arguments.json`, `results.json`, `ground_truth.npy`,
`posterior.npy`. Optional ones (`data.npz`, `run_meta.json`, `losses.npy`) add
extra sections when present. numpy only — no torch import at report time.
"""
import json
import platform
import sys

from datetime import datetime
from pathlib import Path

import numpy as np


# Argparse groups of `train.py`, mirrored so the config table reads the same way
# as `python train.py --help`. Keys absent from every group land in 'Other'.
ARGUMENT_GROUPS = [
    ('Data', ['num_variables', 'num_edges', 'num_samples', 'frac_discrete',
              'num_categories', 'obs_noise']),
    ('Optimization', ['lr', 'delta', 'batch_size', 'num_iterations', 'num_envs']),
    ('Replay buffer', ['replay_capacity', 'prefill']),
    ('Miscellaneous', ['min_exploration', 'update_target_every',
                       'num_samples_posterior', 'prior', 'seed', 'device',
                       'output_folder', 'quiet']),
]

FILE_DESCRIPTIONS = {
    'arguments.json': 'Full CLI configuration of the run (`vars(args)`).',
    'results.json': 'Metrics computed from the posterior samples.',
    'ground_truth.npy': 'Ground-truth adjacency `(N, N)`; `A[i, j] == 1` is `i -> j`.',
    'posterior.npy': 'Posterior DAG samples `(num_samples_posterior, N, N)`.',
    'data.npz': 'Observations + ground-truth adjacency + variable kinds/cardinalities.',
    'model.pt': '`state_dict` of the online GFlowNet policy network.',
    'losses.npy': 'Detailed-balance loss per gradient step.',
    'run_meta.json': 'Timing, versions and hardware of the run.',
    'inspect.sh': 'Ad-hoc console dump of this run directory (not written by `train.py`).',
}

SPARK_BLOCKS = '▁▂▃▄▅▆▇█'


# --- small formatting helpers ------------------------------------------------

def _table(header, rows, align=None):
    """Render a GitHub-flavoured markdown table."""
    align = align or ['---'] * len(header)
    lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(align) + ' |']
    lines += ['| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in rows]
    return lines


def _matrix_block(array, **kwargs):
    """A numpy matrix inside a fenced code block."""
    return ['```', np.array2string(array, **kwargs), '```']


def _sparkline(values, width=60):
    """Coarse unicode sparkline: `values` averaged into at most `width` buckets."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return ''
    buckets = min(width, values.size)
    means = np.array([chunk.mean() for chunk in np.array_split(values, buckets)])
    low, high = means.min(), means.max()
    if high - low < 1e-12:
        levels = np.zeros(buckets, dtype=int)
    else:
        scaled = (means - low) / (high - low) * (len(SPARK_BLOCKS) - 1)
        levels = np.rint(scaled).astype(int)
    return ''.join(SPARK_BLOCKS[level] for level in levels)


def _human_size(num_bytes):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if num_bytes < 1024 or unit == 'GB':
            return f'{num_bytes:.0f} {unit}' if unit == 'B' else f'{num_bytes:.1f} {unit}'
        num_bytes /= 1024


def _format_duration(seconds):
    seconds = float(seconds)
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f'{hours} h {minutes} min {secs} s'
    if minutes:
        return f'{minutes} min {secs} s'
    return f'{seconds:.1f} s'


def _command_line(arguments):
    """Reconstruct the `python train.py ...` invocation from `arguments.json`."""
    parts = ['python train.py']
    for key, value in arguments.items():
        if isinstance(value, bool):
            if value:
                parts.append(f'--{key}')
        else:
            parts.append(f'--{key} {value}')
    return ' \\\n    '.join(parts)


# --- run metadata (written by train.py, consumed here) -----------------------

def collect_run_meta(started_at=None, finished_at=None, duration_seconds=None):
    """Environment/timing record for the current process. `train.py` saves the
    result as `run_meta.json`; the report renders it if the file is present."""
    meta = {
        'started_at': started_at,
        'finished_at': finished_at,
        'duration_seconds': duration_seconds,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor() or platform.machine(),
    }
    try:  # torch is not needed to render a report, only to produce one
        import torch
        meta['torch'] = torch.__version__
    except ImportError:  # pragma: no cover - torch is a hard dep of train.py
        pass
    return meta


# --- report sections ---------------------------------------------------------

def _section_summary(run, arguments, results, posterior, ground_truth):
    num_true = int(results.get('num_true_edges', ground_truth.sum()))
    baseline = float(results.get('empty_graph_shd', num_true))
    e_shd = float(results['expected_shd'])
    verdict = ('beats' if e_shd < baseline else
               'ties' if e_shd == baseline else 'does not beat')
    lines = [
        f'# Run report — `{run.name}`',
        '',
        f'PyTorch DAG-GFlowNet on a synthetic mixed multinomial + Gaussian SCM: '
        f'{arguments["num_variables"]} variables, {num_true} true edges, '
        f'{arguments["num_samples"]} observations, seed {arguments["seed"]}.',
        '',
        f'Expected SHD **{e_shd:.3f}** against an empty-graph baseline of '
        f'{baseline:.3f} — the posterior **{verdict}** the trivial baseline. '
        f'Edge-marginal ROC-AUC **{float(results["roc_auc"]):.3f}** '
        f'(chance = 0.5) over {posterior.shape[0]} posterior samples.',
        '',
        '*Generated by `torch_dag_gfn/report.py` on '
        f'{datetime.now().strftime("%Y-%m-%d %H:%M")}; regenerate with '
        f'`python -m torch_dag_gfn.report {run}`.*',
    ]
    return lines


def _section_reproduce(arguments):
    return ['## Reproduce', '', '```bash', _command_line(arguments), '```']


def _section_configuration(arguments):
    lines = ['## Configuration']
    grouped = {key for _, keys in ARGUMENT_GROUPS for key in keys}
    groups = ARGUMENT_GROUPS + [('Other', [k for k in arguments if k not in grouped])]
    for title, keys in groups:
        rows = [(f'`{key}`', f'`{arguments[key]}`') for key in keys if key in arguments]
        if rows:
            lines += ['', f'**{title}**', ''] + _table(['Argument', 'Value'], rows)
    return lines


def _section_dataset(run, arguments, results, ground_truth):
    lines = ['## Dataset (synthetic ground truth)', '']
    num_variables = ground_truth.shape[0]

    kinds, cardinalities = None, None
    data_path = run / 'data.npz'
    if data_path.exists():
        with np.load(data_path) as data:
            kinds = [str(kind) for kind in data['kinds']]
            cardinalities = [int(card) for card in data['cardinalities']]
            num_samples, _ = data['data'].shape
        lines += [f'`data.npz` holds {num_samples} observations of '
                  f'{num_variables} variables.', '']

    if kinds is not None:
        parents = [np.flatnonzero(ground_truth[:, j]).tolist() for j in range(num_variables)]
        rows = [(index, kinds[index], cardinalities[index] or '—',
                 ', '.join(map(str, parents[index])) or '—')
                for index in range(num_variables)]
        lines += _table(['Variable', 'Kind', 'Cardinality', 'Parents'], rows) + ['']
        lines += ['Categorical nodes may only have categorical parents '
                  '(conditional-Gaussian constraint), which the environment enforces '
                  'through a static type mask.', '']
    else:
        lines += [f'{num_variables} variables (`data.npz` absent — variable types '
                  'unavailable).', '']

    edges = [(int(i), int(j)) for i, j in zip(*ground_truth.nonzero())]
    edge_list = ', '.join(f'`{i} -> {j}`' for i, j in edges) or '_(empty graph)_'
    lines += [f'**Ground-truth DAG** ({len(edges)} edges): {edge_list}', '']
    lines += ['Adjacency `A[i, j] == 1` means `i -> j`:', '']
    lines += _matrix_block(ground_truth.astype(int))
    if 'num_categorical' in results:
        lines += ['', f'Categorical variables: {results["num_categorical"]} / '
                      f'{num_variables}.']
    return lines


def _section_results(results, posterior, ground_truth):
    num_variables = ground_truth.shape[0]
    num_true = int(results.get('num_true_edges', ground_truth.sum()))
    # threshold_metrics scores all N^2 ordered pairs, diagonal included.
    base_rate = num_true / (num_variables ** 2)

    rows = [
        ('Expected SHD', f'**{float(results["expected_shd"]):.3f}**',
         f'empty-graph SHD = {float(results.get("empty_graph_shd", num_true)):.1f}'),
        ('Expected edges', f'**{float(results["expected_edges"]):.3f}**',
         f'true edges = {num_true}'),
        ('Edge-marginal ROC-AUC', f'**{float(results["roc_auc"]):.3f}**', 'chance = 0.5'),
        ('Edge-marginal PRC-AUC', f'**{float(results["prc_auc"]):.3f}**',
         f'base rate = {base_rate:.3f}'),
        ('Average precision', f'**{float(results["ave_prec"]):.3f}**',
         f'base rate = {base_rate:.3f}'),
    ]
    reported = {'expected_shd', 'expected_edges', 'roc_auc', 'prc_auc', 'ave_prec',
                'num_true_edges', 'empty_graph_shd', 'num_categorical'}
    rows += [(f'`{key}`', f'{value}', '—')
             for key, value in results.items() if key not in reported]

    return (['## Results', '']
            + _table(['Metric (`results.json`)', 'Value', 'Baseline / target'], rows)
            + ['', 'Expected SHD and expected edges are averages over the posterior '
                   'samples; the AUC metrics score the edge marginals against the '
                   'ground-truth adjacency.'])


def _section_posterior(posterior, ground_truth, top_k=10):
    marginals = posterior.mean(axis=0)
    truth = ground_truth.astype(bool)
    lines = ['## Posterior', '',
             f'{posterior.shape[0]} sampled DAGs; mean edge count per sample '
             f'{posterior.sum(axis=(1, 2)).mean():.3f}.', '',
             'Marginal edge probabilities `P(i -> j)`:', '']
    lines += _matrix_block(marginals, precision=3, suppress_small=True)

    order = np.dstack(np.unravel_index(
        np.argsort(marginals, axis=None)[::-1], marginals.shape))[0]
    rows = [(f'`{int(i)} -> {int(j)}`', f'{marginals[i, j]:.3f}',
             '✓' if truth[i, j] else '')
            for i, j in order[:top_k]]
    lines += ['', f'Top {len(rows)} edges by marginal probability '
                  '(✓ = present in the ground truth):', '']
    lines += _table(['Edge', 'P(edge)', 'True'], rows)

    true_marginals = marginals[truth]
    false_marginals = marginals[~truth]
    recovered = int((true_marginals >= 0.5).sum())
    spurious = int((false_marginals >= 0.5).sum())
    lines += ['', 'At a 0.5 marginal threshold: '
              f'**{recovered}/{true_marginals.size} true edges recovered**, '
              f'**{spurious} spurious edges**.']
    if true_marginals.size:
        lines += ['', f'- Weakest true edge: `P = {true_marginals.min():.3f}`',
                  f'- Strongest non-edge: `P = {false_marginals.max():.3f}`']
    return lines


def _section_training(run):
    losses_path = run / 'losses.npy'
    if not losses_path.exists():
        return []
    losses = np.load(losses_path)
    if losses.size == 0:
        return []
    tenth = max(losses.size // 10, 1)
    lines = ['## Training', '',
             f'{losses.size} gradient steps of the detailed-balance loss '
             '(Huber-smoothed).', '',
             '```', _sparkline(losses), '```', '']
    lines += _table(
        ['Window', 'Mean loss'],
        [('first 10 % of steps', f'{losses[:tenth].mean():.4f}'),
         ('last 10 % of steps', f'{losses[-tenth:].mean():.4f}'),
         ('final step', f'{losses[-1]:.4f}')])
    return lines


def _section_environment(run):
    meta_path = run / 'run_meta.json'
    if not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text())
    labels = [('started_at', 'Started'), ('finished_at', 'Finished'),
              ('duration_seconds', 'Wall-clock'), ('python', 'Python'),
              ('torch', 'torch'), ('numpy', 'numpy'),
              ('platform', 'Platform'), ('processor', 'Processor')]
    rows = []
    for key, label in labels:
        value = meta.get(key)
        if value is None:
            continue
        if key == 'duration_seconds':
            value = _format_duration(value)
        rows.append((label, f'`{value}`'))
    rows += [(key, f'`{value}`') for key, value in meta.items()
             if key not in dict(labels) and value is not None]
    return ['## Environment', ''] + _table(['Field', 'Value'], rows)


def _section_files(run):
    rows = []
    for path in sorted(run.iterdir()):
        if path.name == 'README.md' or path.name.startswith('.') or path.is_dir():
            continue
        rows.append((f'`{path.name}`', _human_size(path.stat().st_size),
                     FILE_DESCRIPTIONS.get(path.name, '—')))
    return (['## Files', ''] + _table(['File', 'Size', 'Contents'], rows)
            + ['', 'Load the arrays with `numpy.load`; `model.pt` with '
                   '`torch.load(..., map_location="cpu")`.'])


# --- entry point -------------------------------------------------------------

def write_report(output_folder, filename='README.md'):
    """Render `<output_folder>/README.md` from the run's saved artifacts.

    Returns the path of the written file."""
    run = Path(output_folder)
    arguments = json.loads((run / 'arguments.json').read_text())
    results = json.loads((run / 'results.json').read_text())
    ground_truth = np.load(run / 'ground_truth.npy')
    posterior = np.load(run / 'posterior.npy')

    sections = [
        _section_summary(run, arguments, results, posterior, ground_truth),
        _section_reproduce(arguments),
        _section_dataset(run, arguments, results, ground_truth),
        _section_results(results, posterior, ground_truth),
        _section_posterior(posterior, ground_truth),
        _section_training(run),
        _section_configuration(arguments),
        _section_environment(run),
        _section_files(run),
    ]
    body = '\n\n'.join('\n'.join(section).strip() for section in sections if section)

    path = run / filename
    path.write_text(body.rstrip() + '\n')
    return path


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 0 if argv else 1
    for folder in argv:
        print(f'Wrote {write_report(folder)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
