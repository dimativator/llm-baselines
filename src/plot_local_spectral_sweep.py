"""Plot locally persisted losses and effective ranks for a spectral-WD sweep."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def _coefficient(path: Path) -> float:
    match = re.search(r"_cf([0-9.]+)_", path.name)
    if match is None:
        raise ValueError(f"Cannot parse coefficient from {path.name}")
    return float(match.group(1))


def _read_metrics(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open() as metric_file:
        for line in metric_file:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--pattern",
        default="h200_gpu*_llama124m_adamw_slorr_nuc_decoupled_cf*/metrics.jsonl",
    )
    parser.add_argument(
        "--title",
        default="LLaMA 124M Adam + decoupled SLORR-Nuc (local logs)",
    )
    args = parser.parse_args()

    metric_files = sorted(
        args.run_root.glob(args.pattern),
        key=lambda path: _coefficient(path.parent),
    )
    if not metric_files:
        raise SystemExit(f"No metrics.jsonl files under {args.run_root}")

    figure, (loss_axis, rank_axis) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for metric_path in metric_files:
        coefficient = _coefficient(metric_path.parent)
        records = _read_metrics(metric_path)
        train = [record for record in records if record.get("kind") == "train"]
        validation = [record for record in records if record.get("kind") == "val"]
        ranks = [record for record in records if record.get("kind") == "rank"]

        if train:
            loss_axis.plot(
                [record["iter"] for record in train],
                [record["loss"] for record in train],
                alpha=0.35,
                linewidth=0.8,
                label=f"cf={coefficient:g} train",
            )
        if validation:
            loss_axis.plot(
                [record["iter"] for record in validation],
                [record["loss"] for record in validation],
                marker="o",
                markersize=2.5,
                linewidth=1.5,
                label=f"cf={coefficient:g} val",
            )
        if ranks:
            rank_axis.plot(
                [record["iter"] for record in ranks],
                [record["effective_rank/mean_weighted"] for record in ranks],
                marker="o",
                markersize=3,
                label=f"cf={coefficient:g}",
            )

    loss_axis.set_ylabel("Cross-entropy loss")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend(ncol=2, fontsize=8)
    rank_axis.set_xlabel("Optimizer step")
    rank_axis.set_ylabel("Weighted mean effective rank")
    rank_axis.grid(alpha=0.25)
    rank_axis.legend(ncol=3, fontsize=8)
    figure.suptitle(args.title)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
