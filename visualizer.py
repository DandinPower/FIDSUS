#!/usr/bin/env python3
"""
Parse federated-learning stdout logs and generate figures, (algorithm, dataset) per figure, with multiple curves for different client_activity_rate.

Expected log patterns (tolerant to typos):
- "Algorithm: FedAvg"
- "Client activity rate: 0.6"
- "-------------Round number: 42-------------"
- "Averaged Test Accurancy: 0.9036"   (also accepts "Accuracy")

Usage examples:
- Single concatenated log:
    python visualizer.py --logs run.log --out_dir figs
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAVE_SEABORN = True
except Exception:
    _HAVE_SEABORN = False


# ----------------------------
# Parsing
# ----------------------------

ALGO_RE = re.compile(r"^\s*Algorithm:\s*(?P<algo>\S+)\s*$", re.IGNORECASE)
CAR_RE = re.compile(r"^\s*Client activity rate:\s*(?P<car>[0-9]*\.?[0-9]+)\s*$", re.IGNORECASE)
DATASET_RE = re.compile(r"^\s*Dataset:\s*(?P<ds>\S+)\s*$", re.IGNORECASE)

ROUND_RE = re.compile(r"^\s*-+\s*Round number:\s*(?P<r>\d+)\s*-+\s*$", re.IGNORECASE)
# Accept common misspelling "Accurancy"
TESTACC_RE = re.compile(
    r"^\s*Averaged Test Acc(?:urancy|uracy):\s*(?P<acc>[0-9]*\.?[0-9]+)\s*$",
    re.IGNORECASE,
)
FILEPATH_RE = re.compile(r"^\s*File path:\s*(?P<path>.+?)\s*$", re.IGNORECASE)

# From ".../mnist_FedAvg__nc50_0.h5" or ".../FashionMNIST_FedProx__nc50_0.h5"
# Dataset = first token, Algo = second token before "__"
FILE_BASENAME_RE = re.compile(
    r"(?P<ds>[^/_\\]+)_(?P<algo>[^/_\\]+)__",
    re.IGNORECASE,
)


@dataclass
class RunBlock:
    algo: Optional[str] = None
    dataset: Optional[str] = None
    car: Optional[float] = None
    rounds: List[int] = field(default_factory=list)
    test_acc: List[float] = field(default_factory=list)

    def is_plottable(self) -> bool:
        return (
            self.algo is not None
            and self.dataset is not None
            and self.car is not None
            and len(self.rounds) > 0
            and len(self.rounds) == len(self.test_acc)
        )


def _finalize_block(block: RunBlock, out: Dict[Tuple[str, str, float], Tuple[List[int], List[float]]]) -> None:
    if not block.is_plottable():
        return

    # Deduplicate by round (keep last), then sort by round
    rr_to_acc: Dict[int, float] = {}
    for r, a in zip(block.rounds, block.test_acc):
        rr_to_acc[int(r)] = float(a)

    rounds_sorted = sorted(rr_to_acc.keys())
    acc_sorted = [rr_to_acc[r] for r in rounds_sorted]

    key = (block.dataset, block.algo, float(block.car))
    out[key] = (rounds_sorted, acc_sorted)


def parse_logs(paths: List[str]) -> Dict[Tuple[str, str, float], Tuple[List[int], List[float]]]:
    """
    Returns dict keyed by (dataset, algo, car) -> (rounds[], test_acc[])
    """
    results: Dict[Tuple[str, str, float], Tuple[List[int], List[float]]] = {}
    block = RunBlock()

    current_round: Optional[int] = None

    def start_new_block_if_needed() -> None:
        # If we already collected some rounds and see a new header, we should finalize.
        nonlocal block, current_round
        if len(block.rounds) > 0:
            _finalize_block(block, results)
            block = RunBlock()
            current_round = None

    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Detect new run header via Algorithm line
                m = ALGO_RE.match(line)
                if m:
                    # If we encounter Algorithm again after collecting metrics, it's a new run.
                    start_new_block_if_needed()
                    block.algo = m.group("algo")
                    continue

                m = CAR_RE.match(line)
                if m:
                    # New run can reuse same algo but different car; finalize if metrics already started.
                    if len(block.rounds) > 0:
                        start_new_block_if_needed()
                    block.car = float(m.group("car"))
                    continue

                m = DATASET_RE.match(line)
                if m:
                    if len(block.rounds) > 0:
                        # dataset line appearing late usually means new run; be conservative
                        start_new_block_if_needed()
                    block.dataset = m.group("ds")
                    continue

                m = FILEPATH_RE.match(line)
                if m:
                    fp = m.group("path").strip()
                    base = os.path.basename(fp)
                    fm = FILE_BASENAME_RE.search(base)
                    if fm:
                        # Use file path inference as a fallback / cross-check.
                        ds_infer = fm.group("ds")
                        algo_infer = fm.group("algo")
                        if block.dataset is None:
                            block.dataset = ds_infer
                        if block.algo is None:
                            block.algo = algo_infer
                    continue

                m = ROUND_RE.match(line)
                if m:
                    current_round = int(m.group("r"))
                    continue

                m = TESTACC_RE.match(line)
                if m and current_round is not None:
                    block.rounds.append(current_round)
                    block.test_acc.append(float(m.group("acc")))
                    continue

    # finalize last block
    _finalize_block(block, results)
    return results


# ----------------------------
# Plotting
# ----------------------------

def set_plot_style() -> None:
    if _HAVE_SEABORN:
        sns.set_theme(style="whitegrid", context="talk")
        sns.despine()
    else:
        plt.style.use("default")
        plt.rcParams.update({
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 12,
        })


def plot_six_figures(
    series: Dict[Tuple[str, str, float], Tuple[List[int], List[float]]],
    out_dir: str
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    datasets = sorted({k[0] for k in series.keys()}, key=lambda s: s.lower())
    algos = sorted({k[1] for k in series.keys()}, key=lambda s: s.lower())
    cars = sorted({k[2] for k in series.keys()})

    set_plot_style()

    for ds in datasets:
        for algo in algos:
            # Collect the 3 curves (or whatever exists) for this (ds, algo)
            keys = [(ds, algo, car) for car in cars if (ds, algo, car) in series]
            if not keys:
                continue

            fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)

            for car in sorted([k[2] for k in keys]):
                rounds, acc = series[(ds, algo, car)]
                acc_plot = acc
                ax.plot(rounds, acc_plot, linewidth=2.0, label=f"CAR = {car:g}")

            ax.set_title(f"{ds} | {algo}")
            ax.set_xlabel("Communication round")
            ax.set_ylabel("Test accuracy")
            ax.set_xlim(left=0)
            ax.set_ylim(0.0, 1.0)

            # Make ticks less cluttered for 0..100
            if len(keys) > 0:
                max_r = max(series[(ds, algo, k[2])][0][-1] for k in keys)
                if max_r >= 50:
                    ax.set_xticks(list(range(0, int(max_r) + 1, 10)))

            leg = ax.legend(title="Client activity rate", frameon=True, loc="lower right")
            if leg is not None:
                leg._legend_box.align = "right"

            fig.tight_layout()

            safe_ds = re.sub(r"[^A-Za-z0-9._-]+", "_", ds)
            safe_algo = re.sub(r"[^A-Za-z0-9._-]+", "_", algo)
            png_path = os.path.join(out_dir, f"{safe_ds}_{safe_algo}_CAR_compare.png")
            fig.savefig(png_path, bbox_inches="tight", dpi=300)

            plt.close(fig)


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="*", default=[], help="One or more log file paths.")
    ap.add_argument("--out_dir", default="figs", help="Output directory for figures.")
    # ap.add_argument("--smooth", type=int, default=1, help="Moving average window (1 = no smoothing).")
    # ap.add_argument("--no_pdf", action="store_true", help="Disable PDF output.")
    args = ap.parse_args()

    paths: List[str] = []
    if args.logs:
        paths.extend(args.logs)

    # De-dup while preserving order
    seen = set()
    paths = [p for p in paths if not (p in seen or seen.add(p))]

    if not paths:
        raise SystemExit("No log files provided. Use --logs")

    series = parse_logs(paths)

    if not series:
        raise SystemExit("Parsed 0 runs. Check that your log contains 'Algorithm:', 'Client activity rate:', "
                         "'Round number', and 'Averaged Test Accurancy/Accuracy' lines.")

    # Basic sanity check: report what we found
    combos = sorted(series.keys(), key=lambda k: (k[0].lower(), k[1].lower(), k[2]))
    print(f"[OK] Parsed {len(combos)} run(s):")
    for ds, algo, car in combos:
        rr, acc = series[(ds, algo, car)]
        print(f"  - dataset={ds}, algo={algo}, car={car:g}, points={len(rr)}, last_round={rr[-1]}, last_acc={acc[-1]:.4f}")

    plot_six_figures(series, out_dir=args.out_dir)
    print(f"[OK] Figures written to: {args.out_dir}")


if __name__ == "__main__":
    main()
