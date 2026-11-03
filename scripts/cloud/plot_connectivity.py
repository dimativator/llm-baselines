"""Render measured endpoint and mode-connectivity results without filling gaps."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

LABELS = ("l2", "swd05", "swd07", "swd1")
NAMES = dict(l2="AdamW, WD 0.1", swd05="SWD 0.5", swd07="SWD 0.7", swd1="SWD 1.0")
PAIRS = tuple(left + "__" + right for left, right in itertools.combinations(LABELS, 2))
COLORS = ("#3568ab", "#d27022", "#37906b", "#964f9a")
STYLES = dict(straight=("#8b8b8b", "--"), bezier=("#255fa7", "-"))


class PaperLossNormalize(matplotlib.colors.Normalize):
    """Shifted log color scale from the authors' plane_plot.py visualization."""

    def __call__(self, value, clip=None):
        values = np.ma.asarray(value)
        floor = -5.0
        shifted = np.ma.maximum(values - self.vmin, np.exp(floor))
        return 0.9 * (np.ma.log(shifted) - floor) / (np.log(self.vmax - self.vmin) - floor)


def read_json(path: Path, default: object) -> object:
    return json.loads(path.read_text()) if path.is_file() else default


def read_metrics(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            # An actively written final JSONL record is retried on the next run.
    return rows


def pair_name(pair: str) -> str:
    left, right = pair.split("__")
    return NAMES[left] + " ↔ " + NAMES[right]


def save(fig: plt.Figure, output: Path, name: str) -> list[str]:
    paths = []
    for extension in ("png", "pdf"):
        path = output / (name + "." + extension)
        fig.savefig(path, dpi=180, facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def panel_grid(title: str, subtitle: str, smoke: bool = False) -> tuple[plt.Figure, list]:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), layout="constrained")
    prefix = "CPU SMOKE / synthetic tokens\n" if smoke else ""
    fig.suptitle(prefix + title + "\n" + subtitle, fontsize=15)
    for axis, pair in zip(axes.flat, PAIRS):
        axis.set_title(pair_name(pair), fontsize=11)
        axis.grid(alpha=0.18)
    return fig, list(axes.flat)


def missing(axis, message: str = "Awaiting measured results") -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes,
              color="#777777", fontsize=11)


def endpoint_plot(root: Path, output: Path, smoke: bool = False) -> list[str]:
    histories = {label: read_metrics(root / label / "metrics.jsonl") for label in LABELS}
    if not any(histories.values()):
        return []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    fig.suptitle(("CPU SMOKE / synthetic tokens\n" if smoke else "") + "Matched Llama endpoints", fontsize=15)
    for label, color in zip(LABELS, COLORS):
        for axis, metric in zip(axes, ("train_loss", "val_loss")):
            rows = [row for row in histories[label] if metric in row and "step" in row]
            if rows:
                axis.plot([row["step"] for row in rows], [row[metric] for row in rows],
                          color=color, label=NAMES[label], linewidth=1.7)
    for axis, title in zip(axes, ("Training minibatch cross-entropy", "Fixed-subset validation cross-entropy")):
        axis.set(title=title, xlabel="Optimizer updates", ylabel="Cross-entropy (nats / token)")
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(fontsize=9)
        else:
            missing(axis)
    return save(fig, output, "endpoint_training")


def curves_plot(curves: dict, output: Path, split: str, metric: str, smoke: bool = False) -> list[str]:
    title = ("Training" if split == "train" else "Validation") + (
        " cross-entropy along connections" if metric == "loss" else " token error along connections")
    fig, axes = panel_grid(title, "Solid: trained Bézier curve · Dashed: straight segment · Fixed evaluation subsets", smoke)
    for axis, pair in zip(axes, PAIRS):
        rows = curves[pair]
        token_counts = set()
        for kind, (color, style) in STYLES.items():
            subset = sorted([row for row in rows if row["curve"] == kind and split in row], key=lambda row: row["t"])
            if not subset:
                continue
            token_counts.update(row[split]["tokens"] for row in subset)
            axis.plot([row["t"] for row in subset], [row[split][metric] for row in subset],
                      color=color, linestyle=style, linewidth=2, label=kind)
        if token_counts:
            counts = ", ".join(f"{value:,}" for value in sorted(token_counts))
            axis.set_title(pair_name(pair) + f"\n{counts} tokens per point", fontsize=10)
        else:
            missing(axis)
        axis.set(xlabel="Curve parameter t", xlim=(0, 1),
                 ylabel="Cross-entropy (nats / token)" if metric == "loss" else "Next-token error (%)")
    return save(fig, output, split + "_" + metric + "_paths")


def surfaces_plot(surfaces: dict, output: Path, smoke: bool = False) -> list[str]:
    if not any(surfaces.values()):
        return []
    fig, axes = panel_grid("Validation loss in the endpoint–control plane",
                          "Axes are basis coefficients, not orthonormal distances · Each contour uses a smaller fixed subset", smoke)
    for axis, pair in zip(axes, PAIRS):
        rows = surfaces[pair]
        if not rows:
            missing(axis)
            continue
        a_values = sorted({row["a"] for row in rows})
        b_values = sorted({row["b"] for row in rows})
        measured = {(row["a"], row["b"]): row["loss"] for row in rows}
        values = np.array([[measured.get((a, b), np.nan) for a in a_values] for b in b_values])
        if len(a_values) > 1 and len(b_values) > 1 and np.isfinite(values).sum() >= 4:
            low, high = np.nanmin(values), np.nanmax(values)
            if high <= low:
                high = low + 0.01
            # Match the paper's seven log-spaced bands and reversed-jet palette.
            levels = low + np.exp(np.linspace(-5.0, np.log(high - low), 8))
            levels[0], levels[-1] = low, high
            norm = PaperLossNormalize(low - 1e-8, high + 1e-8)
            masked = np.ma.masked_invalid(values)
            contours = axis.contourf(a_values, b_values, masked, levels=levels,
                                     cmap="jet_r", norm=norm, alpha=0.55)
            axis.contour(a_values, b_values, masked, levels=levels,
                         cmap="jet_r", norm=norm, linewidths=1.2)
            fig.colorbar(contours, ax=axis, label="Cross-entropy", shrink=0.85, format="%.3g")
        else:
            points = axis.scatter([row["a"] for row in rows], [row["b"] for row in rows],
                                  c=[row["loss"] for row in rows], cmap="jet_r")
            fig.colorbar(points, ax=axis, label="Cross-entropy", shrink=0.85)
        t = np.linspace(0, 1, 201)
        axis.plot(t, np.zeros_like(t), color="black", linestyle="--", linewidth=2)
        axis.plot(t, 2*t*(1-t), color="black", linewidth=2.5)
        axis.scatter([0, 1], [0, 0], color="black", s=35, zorder=5)
        left, right = pair.split("__")
        for x, label in ((0, NAMES[left]), (1, NAMES[right])):
            axis.annotate(label, (x, 0), xytext=(0, -16), textcoords="offset points",
                          ha="center", fontsize=8, color="black",
                          bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1))
        counts = ", ".join(f"{value:,}" for value in sorted({row["tokens"] for row in rows}))
        axis.set_title(pair_name(pair) + f"\n{counts} validation tokens per point", fontsize=10)
        axis.set(xlabel="a: endpoint direction", ylabel="b: control-bend direction",
                 xlim=(-0.1, 1.1), ylim=(-0.1, 0.65))
        axis.grid(False)
    fig.legend(handles=[Line2D([0], [0], color="black", linestyle="--", label="Straight: b = 0"),
                        Line2D([0], [0], color="black", label="Bézier: b = 2t(1−t)")],
               loc="outside lower center", ncol=2, frameon=False)
    return save(fig, output, "loss_plane_contours")


def rank_plot(finals: dict, output: Path, smoke: bool = False) -> list[str]:
    if not any(result.get("ranks") for result in finals.values()):
        return []
    fig, axes = panel_grid("Effective rank along trained Bézier paths",
                          "Arithmetic mean across transformer matrices · exp(entropy of normalized singular values)", smoke)
    for axis, pair in zip(axes, PAIRS):
        rows = finals[pair].get("ranks", [])
        if rows:
            rows = sorted(rows, key=lambda row: row["t"])
            axis.plot([row["t"] for row in rows], [row["mean_effective_rank"] for row in rows],
                      color=STYLES["bezier"][0], marker="o", linewidth=2)
        else:
            missing(axis)
        axis.set(xlabel="Curve parameter t", ylabel="Mean effective rank", xlim=(0, 1))
    return save(fig, output, "effective_rank_paths")


def summary_csv(curves: dict, output: Path) -> None:
    fields = ("pair", "curve", "split", "grid_points", "tokens_per_point", "complete_endpoints",
              "endpoint0_loss", "endpoint1_loss", "max_loss", "max_loss_t",
              "barrier_above_worse_endpoint", "barrier_above_chord")
    with (output / "path_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pair in PAIRS:
            for kind in STYLES:
                subset = sorted([row for row in curves[pair] if row["curve"] == kind], key=lambda row: row["t"])
                for split in ("train", "val"):
                    rows = [row for row in subset if split in row]
                    if not rows:
                        continue
                    complete = rows[0]["t"] == 0 and rows[-1]["t"] == 1
                    maximum = max(rows, key=lambda row: row[split]["loss"])
                    row = dict(pair=pair, curve=kind, split=split, grid_points=len(rows),
                               tokens_per_point=";".join(str(n) for n in sorted({r[split]["tokens"] for r in rows})),
                               complete_endpoints=complete, max_loss=maximum[split]["loss"], max_loss_t=maximum["t"])
                    if complete:
                        l0, l1 = rows[0][split]["loss"], rows[-1][split]["loss"]
                        row.update(endpoint0_loss=l0, endpoint1_loss=l1,
                                   barrier_above_worse_endpoint=row["max_loss"] - max(l0, l1),
                                   barrier_above_chord=max(r[split]["loss"] - ((1-r["t"])*l0 + r["t"]*l1) for r in rows))
                    writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="Explicitly label synthetic CPU smoke results")
    options = parser.parse_args()
    root, output = options.root, options.output
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "axes.titlepad": 9})
    curves = {pair: read_json(root / pair / "curves.json", []) for pair in PAIRS}
    surfaces = {pair: read_json(root / pair / "surface.json", []) for pair in PAIRS}
    finals = {pair: read_json(root / pair / "final.json", {}) for pair in PAIRS}
    smoke = options.smoke or any(read_json(root / f"LANE_{lane}_STARTED.json", {}).get("smoke", False) for lane in (0, 1))
    smoke = smoke or any(read_json(root / label / "config.json", {}).get("model", {}).get("vocab_size") == 128 for label in LABELS)
    plots = endpoint_plot(root, output, smoke)
    if any(curves.values()):
        for split, metric in (("train", "loss"), ("val", "loss"), ("val", "error_percent")):
            plots += curves_plot(curves, output, split, metric, smoke)
    plots += surfaces_plot(surfaces, output, smoke)
    plots += rank_plot(finals, output, smoke)
    summary_csv(curves, output)
    sources = {}
    for name in LABELS + PAIRS:
        for filename in ("metrics.jsonl", "curves.json", "surface.json", "final.json", "config.json"):
            path = root / name / filename
            if path.is_file():
                sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(root=str(root), outputs=plots, sources_sha256=sources, smoke=smoke,
                    complete_pairs=[pair for pair in PAIRS if finals[pair]],
                    note="Subset estimates, no replicate uncertainty. Missing panels contain no imputed measurements.")
    (output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(figures=len(plots) // 2, complete_pairs=len(manifest["complete_pairs"]), output=str(output))))


if __name__ == "__main__":
    main()
