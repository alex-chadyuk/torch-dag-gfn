# torch-dag-gfn — PyTorch DAG-GFlowNet (mixed multinomial + Gaussian)

A minimal, self-contained **PyTorch** reimplementation of DAG-GFlowNet
(Deleu et al. 2022, *Bayesian Structure Learning with Generative Flow Networks*),
scoped to **synthetic datasets with a mix of multinomial (categorical) and
Gaussian variables**.

## What this is (and isn't)

- **In scope:** random Erdős–Rényi SCMs mixing Gaussian and categorical
  variables; a decomposable **BIC** score; the faithful edge-token **Linear
  Transformer** policy; **detailed-balance** training with a target network and
  replay buffer; posterior sampling and evaluation (E-SHD, expected edges,
  ROC/PRC-AUC).
- **Out of scope (dropped vs. the JAX original):** Sachs / real data,
  interventional data, multiprocessing score workers, bit-packed replay,
  `gym`/`pgmpy` dependencies. The mixed BIC score subsumes the all-Gaussian and
  all-discrete cases, so the original BGe/BDe scores are not ported separately.

## Modelling choices

- **Mixed BIC score** (`torch_dag_gfn/scores.py`). The score of a DAG factorises
  into per-node local scores, so adding one edge changes exactly one term (the
  environment returns those delta-scores directly). A **Gaussian** node is scored
  by an OLS regression on `[intercept, continuous parents, one-hot(categorical
  parents)]` (Gaussian log-likelihood at the MLE minus the BIC penalty); a
  **categorical** node by a multinomial contingency-count log-likelihood minus
  the BIC penalty. The GFlowNet reward is `R(G) = exp(Σ local + log p(G))`.
- **Conditional-Gaussian (CLG) structure.** Categorical nodes may only have
  categorical parents — the standard well-definedness condition for mixed CG
  networks. This is enforced two ways: the generator assigns variable types in
  topological order so the constraint holds by construction, and the environment
  applies a static `type_mask` that forbids continuous → categorical edges.
- **Faithful policy net** (`torch_dag_gfn/nets.py`). The N² candidate edges are
  the tokens; a shared body of Linear-Transformer blocks (Katharopoulos et al.
  `elu(x)+1` attention, re-injecting the raw adjacency at each block) feeds two
  heads producing per-edge "add" logits and a pooled "stop" logit.

## Layout

```
train.py                    CLI: generate data -> train -> evaluate -> save
torch_dag_gfn/
  data.py                   mixed SCM generator + VarSpec + CLG type_mask
  scores.py                 mixed BIC score + graph priors + memoisation
  env.py                    vectorised DAG env (closure-mask acyclicity + CLG mask)
  nets.py                   linear attention, transformer block, edge-token policy
  gflownet.py               controller: act / detailed-balance loss / step + posterior
  buffer.py                 replay buffer
  metrics.py                E-SHD, expected edges, ROC/PRC-AUC
  report.py                 renders the run README.md from the saved artifacts
```

## Environment

```bash
conda create -y -n torch-dag-gfn python=3.11
conda activate torch-dag-gfn
pip install -r requirements.txt   # torch, numpy, scipy, networkx, scikit-learn, tqdm
```
CPU-only; no GPU needed at synthetic scale. numpy>=2 compatible.

## Usage

```bash
python train.py \
    --num_variables 5 --num_edges 5 --num_samples 500 \
    --frac_discrete 0.4 --num_categories 3 \
    --num_iterations 5000 --batch_size 128 --prefill 500 --seed 0 \
    --output_folder results-smoke
```

Key flags: `--frac_discrete` (fraction of CLG-eligible nodes made categorical),
`--num_categories` (states per categorical variable), `--obs_noise`,
`--prior {uniform,erdos_renyi,edge,fair}`, `--lr`, `--num_samples_posterior`,
`--update_target_every`. Run `python train.py --help` for the full list.

Outputs written to `--output_folder`: `arguments.json`, `data.npz`
(data + ground-truth adjacency + variable kinds/cardinalities),
`ground_truth.npy`, `posterior.npy` (`(num_samples, N, N)`), `model.pt`,
`results.json`, `losses.npy` (loss per gradient step), `run_meta.json`
(timing/versions/hardware), and `README.md` — a rendered run report.

## Run reports

After training, `torch_dag_gfn/report.py` writes a `README.md` into the output
folder: headline verdict (E-SHD vs. the empty-graph baseline), the exact command
that reproduces the run, the ground-truth DAG and variable types, the metrics
table with baselines, posterior edge marginals and threshold recovery, the loss
trace, the environment record, and a file manifest.

It reads only the saved artifacts, so any past run can be (re)reported without
retraining — numpy is the only dependency:

```bash
python -m torch_dag_gfn.report out-26-07-20-16-22 [more-run-dirs ...]
```

## Results (canonical smoke run, seed 0)

Wall-clock ≈ 34 min on CPU (Apple Silicon; 5 000 gradient steps + 500 prefill +
1 000 posterior samples). Seed-0 ground truth: 2 edges over 5 nodes (2 categorical,
3 Gaussian); both true edges are categorical → Gaussian.

| Metric (`results.json`) | Value | Baseline / target |
|---|---|---|
| Expected SHD | **0.137** | empty-graph SHD = 2.0 |
| Expected edges | **2.137** | true edges = 2 |
| Edge-marginal ROC-AUC | **1.000** | chance = 0.5 |
| Edge-marginal PRC-AUC | **1.000** | — |
| Average precision | **1.000** | — |

The posterior concentrates on the true DAG: both true edges have marginal
probability 1.0, spurious edges only 0.02–0.04. This demonstrates the pipeline
learns correct mixed-type structure (one seed on one small graph — not a
benchmarked SOTA claim).


