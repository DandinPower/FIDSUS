# FIDSUS

This repository contains a federated learning (FL) codebase with multiple algorithms and multiple model backbones. The main entrypoint is `main.py`, and `run.sh` provides a simple way to sweep over (algorithm, dataset, client_activity_rate). A companion script, `visualizer.py`, parses stdout logs and generates comparison figures.

## Requirements

- UV (see https://docs.astral.sh/uv/getting-started/installation/)
- OS: Linux/macOS recommended (bash scripts)

Setup:

```bash
uv sync
source .venv/bin/activate
```

## Dataset layout

The code expects per-client `.npz` files under:

```
dataset/<DATASET_NAME>/
  train/
    0.npz 1.npz ... (one file per client)
  test/
    0.npz 1.npz ...
```

Each `.npz` should contain a dict-like object with keys:

* `x`: features (float-compatible)
* `y`: labels (int)

Data is loaded via `simulation/utils/data_utils.py` (`read_client_data_un`).

Important: `-data/--dataset` must match the folder name exactly (case-sensitive on Linux).

## Supported algorithms

Select via `-algo/--algorithm`:

* `FedAvg`
* `FedProx`
* `FedProto`
* `MOON`
* `GPFL`
* `FedGH`
* `FedAvgDBE`
* `FIDSUS`

(See `simulation/flcore/servers/`.)

## Model selection by dataset

Select via `-m/--model`:

* For `NSLKDD` and `UNSW`: use `1dcnn`
* For `FashionMNIST` and `mnist`: use `2dcnn` or `microvit`

Examples:

* Tabular intrusion detection datasets (feature vectors): `-m 1dcnn`
* Image datasets: `-m 2dcnn` or `-m microvit`

## Key runtime knobs

Common flags (see `main.py`):

* `-data` / `--dataset`: dataset name (folder under `dataset/`)
* `-algo` / `--algorithm`: FL algorithm
* `-m` / `--model`: `1dcnn`, `2dcnn`, `microvit`
* `-nc` / `--num_clients`: total clients (must match number of per-client files)
* `-jr` / `--join_ratio`: fraction of clients selected each round
* `-car` / `--client_activity_rate`: fraction of selected clients that are active each round
* `-gr` / `--global_rounds`: number of communication rounds
* `-ls` / `--local_epochs`: local epochs per round
* `-dev` / `--device`: `cuda` or `cpu`
* `-did` / `--device_id`: CUDA device id (string)

Note on `client_activity_rate`: internally it uses `int(client_activity_rate * current_num_join_clients)`. If this becomes 0, training will break. Ensure `client_activity_rate * (num_clients * join_ratio) >= 1`.

## Running sweeps with `run.sh`

`run.sh` is a simple nested loop over:

* `algos`
* `datasets`
* `client_activity_rates`

Edit these arrays to match your desired sweep.

Current example (image datasets + MicroViT):

```bash
algos=("FedAvg" "FIDSUS" "FedProx")
datasets=("mnist" "FashionMNIST")
client_activity_rates=("0.6" "0.8" "1")
model="microvit"
```

## Logging all experiments

To store all stdout outputs into a log file:

```bash
bash run.sh > result.log 2>&1
```

## Generating figures from logs

After you have a `.log`, generate figures with:

```bash
python visualizer.py --logs result.log --out_dir results
```

What you get:

* One figure per (algorithm, dataset)
* Multiple curves within each figure corresponding to different `client_activity_rate` values

Figures will be written to `results/` as:

```
<dataset>_<algorithm>_CAR_compare.png
```