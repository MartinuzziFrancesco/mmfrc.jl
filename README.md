# Storage and recall of multiple chaotic attractors in minimal reservoir computers

Code accompanying:

> F. Martinuzzi, H. Kantz, "Storage and recall of multiple chaotic attractors in minimal reservoir computers", *Chaos* (2026).

This repository contains the Julia code used to run all experiments and generate the figures
in the paper. It does **not** contain any generated data, results, or figures — everything is
regenerated from scratch by the scripts below, starting from the eight benchmark chaotic
systems (Aizawa, Arneodo, Chua, GenesioTesi, Halvorsen, Lorenz, Rössler, SprottS).

## Requirements

- [Julia](https://julialang.org/) 1.12 (the environment was originally run on 1.12.4; verified
  working end-to-end on 1.12.6)
- All dependencies are registered Julia packages (including
  [`ChaoticDynamicalSystemLibrary.jl`](https://juliahub.com) and
  [`ReservoirComputing.jl`](https://github.com/SciML/ReservoirComputing.jl)) and will be
  installed automatically. `ReservoirComputing` is pinned to `0.12.1` in `Project.toml`
  ([compat]): the custom PARC layer in `base/parc_model.jl` hooks into a few of that
  package's internal (non-exported) functions, which changed between `0.12.1` and later
  `0.12.x` releases, so a newer version breaks PARC training. Everything else resolves to
  the latest compatible version.

Scripts must be run with `--project=.` — some `using` statements execute before the
in-script `Pkg.activate`, so invoking a script without `--project=.` (e.g. plain
`julia bt_dry_run.jl`) fails immediately. From the repository root:

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

## The two protocols

The paper studies two ways of training a single minimal echo state network (ESN) to
reproduce **pairs** of chaotic attractors, across 10 minimal deterministic reservoir
topologies (`base/constants.jl`) and all 28 unordered pairs drawn from the 8 benchmark
systems:

- **ICT** (input concatenation training) — the two systems' coordinates are concatenated
  into a single input/output and the ESN is trained to reproduce their joint evolution.
  Code: `base/bt_training.jl`, run with `bt_runs.jl` → writes to `bt_results/`.
- **PARC** (parameter-aware reservoir computing) — a scalar label fed through the bias term
  selects which attractor to recall. Code: `base/parc_model.jl` + `base/parc_training.jl`,
  run with `parc_runs.jl` → writes to `parc_results/`.

There is also a third variant, `obt_runs.jl` (`base/obt_training.jl` → `obt_results/`), which
trains on the two attractors as separate, unlabelled, non-concatenated segments. It is kept
here because it's part of the same codebase, but it is not one of the two protocols reported
in the current manuscript text — treat it as exploratory/development code.

## Reproducing the results

Each of `bt_runs.jl`, `obt_runs.jl`, `parc_runs.jl` is self-contained: it activates the
project, regenerates the chaotic trajectories (`solve_csystems`, cached to
`data/chaotic_systems.json`), sweeps all (topology, system pair) jobs, and writes one JSON
file per job to `<protocol>_results/<topology>/fixed_<PairName>.json`.

1. **Smoke-test first.** `bt_dry_run.jl` / `parc_dry_run.jl` run the same pipeline with a
   drastically reduced grid/IC count (`DRY_RUN = true` at the top of the file) so you can
   check everything runs before committing to the full sweep:

   ```bash
   julia --project=. bt_dry_run.jl
   julia --project=. parc_dry_run.jl
   ```

2. **Full run.** The scripts parallelize over SLURM tasks (`SLURM_PROCID`/`SLURM_NTASKS`),
   striping the (topology × pair) job list round-robin across tasks; each task also uses
   `Threads.nthreads()` threads internally. On a SLURM cluster:

   ```bash
   sbatch bt_launch.sh
   sbatch obt_launch.sh    # optional, see note above
   sbatch parc_launch.sh
   ```

   Adjust `--ntasks`, `--cpus-per-task`, `--mem-per-cpu`, `--time` and `--partition` in the
   `.sh` files for your cluster. The full sweep (10 topologies × 28 pairs × 25-point
   hyperparameter grid search × 10 initial conditions, per protocol) is compute-heavy and is
   intended to be run on a cluster, not a laptop.

   Without SLURM, each script still runs single-task/single-node:

   ```bash
   julia --project=. -t auto bt_runs.jl
   julia --project=. -t auto parc_runs.jl
   ```

3. **Aggregate.** `analysis.jl` reads `bt_results/` and `parc_results/` and writes the
   per-topology summary tables to `results/results_by_init_bt.json` and
   `results/results_by_init_parc.json` used by the figure scripts.

   ```bash
   julia --project=. analysis.jl
   ```

   `add_results.jl` is a standalone utility for adding a new evaluation metric to already
   computed result files without retraining — not needed for the default pipeline.

## Reproducing the figures

All figure scripts read from `bt_results/` / `parc_results/` / `obt_results/` and
`results/`, and write `.png`/`.eps` files to `figures/`, using the shared house theme in
`base/theme.jl`.

```bash
julia --project=. viz.jl          # main results figures
julia --project=. viz_rev.jl      # revision figures
julia --project=. viz_rev_bis.jl  # further revision figures
julia --project=. viz_states.jl   # reservoir-state (neuron time series) figures
```

`viz_states.jl` retrains a small number of ICT reservoirs to expose their internal state
trajectories (not stored in `bt_results/`) and caches the plotted data to
`figures/data/ict_states_*.json`; `viz_states_replot.jl` (which uses the shared plotting
code in `viz_states_plot.jl`) restyles those figures from the cached data without retraining:

```bash
julia --project=. viz_states_replot.jl
```

### Schematic / methods figures

`methods_figures/` is a separate, self-contained Julia project (own `Project.toml`) that
generates the schematic and methods illustrations (attractor overview, PARC/ICT diagrams,
KLD-degradation illustration):

```bash
cd methods_figures
julia --project=. -e 'import Pkg; Pkg.instantiate()'
julia --project=. cs.jl
julia --project=. mf.jl
julia --project=. minimal.jl
julia --project=. kld.jl
```

## Repository layout

```
Project.toml            environment for the main pipeline
base/
  constants.jl           topologies, benchmark systems, experiment sizes
  data.jl                 ODE integration + train/test set construction
  evaluation.jl           KLD / total-variation attractor-comparison metrics
  bt_training.jl           ICT training/grid search
  parc_model.jl            PARC ESN model
  parc_training.jl         PARC training/grid search
  obt_training.jl          third (unlabelled, non-concatenated) training variant
  theme.jl                 CairoMakie house theme
bt_runs.jl / bt_dry_run.jl / bt_launch.sh        ICT experiment sweep
parc_runs.jl / parc_dry_run.jl / parc_launch.sh  PARC experiment sweep
obt_runs.jl / obt_launch.sh                      third-variant experiment sweep
analysis.jl              aggregate results/results_by_init_{bt,parc}.json
add_results.jl           optional: add a metric to existing result files
viz.jl / viz_rev.jl / viz_rev_bis.jl             main + revision result figures
viz_states.jl / viz_states_plot.jl / viz_states_replot.jl   reservoir-state figures
methods_figures/          separate project: schematic/methods figures
```

## License

MIT, see `LICENSE`.
