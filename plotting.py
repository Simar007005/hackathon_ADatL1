from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def savefig(fig, path: Path, dpi: int = 160):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[PLOT] saved {path}")


def plot_data_distributions(data, layout, feature_names=("Et", "eta", "phi"), outpath=None):
    """Histogram each (scaled) input feature over the real objects in an EventData, one row per collection.

    ``layout`` is the collection -> number of slots dict the data was built with (``LAYOUT`` in train.ipynb).
    Panels for features a collection does not have (e.g. eta for MET) are left empty.
    """
    import dataset

    n_features = len(feature_names)
    x = data.x.reshape(len(data.x), data.mask.shape[1], n_features)
    fig, axes = plt.subplots(len(layout), n_features, figsize=(3.5 * n_features, 2.5 * len(layout)), squeeze=False)
    start = 0
    for row, (coll, n_slots) in zip(axes, layout.items()):
        sl = slice(start, start + n_slots)
        start += n_slots
        objs = x[:, sl][data.mask[:, sl]]  # (n_objects, n_features)
        for i, (ax, feat) in enumerate(zip(row, feature_names)):
            if feat not in dataset.KINEMATIC_BRANCHES.get(coll, {}):
                ax.text(0.5, 0.5, f"{coll} has no {feat}", ha="center", va="center", transform=ax.transAxes, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            ax.hist(objs[:, i], bins=100, histtype="step")
            ax.set_yscale("log")
            ax.set_title(f"{coll} ({n_slots} slot{'s' if n_slots > 1 else ''})", fontsize=9)
            ax.set_xlabel(f"{feat} (scaled)")
    fig.tight_layout()
    if outpath:
        savefig(fig, outpath)
    return fig


def plot_loss_curve(history: Dict[str, List[Tuple[int, float]]], outpath=None):
    """history: {"train": [(epoch, loss), ...], "val": [(epoch, loss), ...]}"""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, points in history.items():
        if points:
            epochs, losses = zip(*points)
            ax.plot(epochs, losses, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend()
    if outpath:
        savefig(fig, outpath)
    return fig


def plot_score_distributions(scores: Dict[str, np.ndarray], title: str = "", outpath=None):
    """scores: {sample name: anomaly scores}, including "background"."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    all_scores = np.concatenate(list(scores.values()))
    bins = np.linspace(*np.quantile(all_scores, [0.0, 0.999]), 80)
    for name, s in scores.items():
        ax.hist(s, bins=bins, histtype="step", density=True, label=name,
                linewidth=2 if name == "background" else 1)
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(fontsize=8)
    if outpath:
        savefig(fig, outpath)
    return fig


def plot_roc(rocs: Dict[str, Tuple[np.ndarray, np.ndarray, float]], acceptance: float = None,
             title: str = "", outpath=None):
    """rocs: {signal name: (background acceptance, signal efficiency, auc)}"""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (fpr, tpr, auc) in rocs.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    if acceptance is not None:
        ax.axvline(acceptance, color="gray", linestyle="--", label=f"acceptance = {acceptance:g}")
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1)
    ax.set_xlabel("background (zero-bias) acceptance")
    ax.set_ylabel("signal efficiency")
    ax.set_title(title)
    ax.legend(fontsize=8)
    if outpath:
        savefig(fig, outpath)
    return fig


def plot_signal_efficiency(eff: Dict[str, Dict[str, float]], acceptance: float, outpath=None):
    """eff: {model name: {signal name: efficiency}} as a grouped bar chart."""
    models = list(eff)
    signals = list(next(iter(eff.values())))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(signals)), 4.5))
    for i, m in enumerate(models):
        ax.bar(np.arange(len(signals)) + i * width, [eff[m][s] for s in signals], width, label=m)
    ax.set_xticks(np.arange(len(signals)) + 0.4 - width / 2)
    ax.set_xticklabels(signals, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"signal efficiency at {acceptance:g} acceptance")
    ax.legend()
    if outpath:
        savefig(fig, outpath)
    return fig


def _pareto_front(cost: np.ndarray, metric: np.ndarray) -> np.ndarray:
    """Indices of the trials no other trial beats at a lower (or equal) cost, sorted by cost."""
    order = np.lexsort((-metric, cost))
    front, best = [], -np.inf
    for i in order:
        if metric[i] > best:
            front.append(i)
            best = metric[i]
    return np.array(front, dtype=int)


COST_LABELS = {"n_params": "number of trainable parameters", "bops": "bit operations (BOPs)",
               "n_layers": "number of dense layers"}


def plot_trials(trials, metrics=("auc", "efficiency"), cost: str = "bops", signal: str = None, outpath=None):
    """Performance vs model cost for every saved trial (see utils.save_trial).

    trials: DataFrame from utils.load_trials, one row per (trial, signal).
    metrics: columns to plot, one panel each ("auc" and/or "efficiency").
    cost:   x axis, a column from utils.model_cost: "bops" (FPGA area), "n_params" or "n_layers" (latency).
    signal: plot a single signal; by default each metric is averaged over all signals.

    The dashed line is the Pareto front: the best performance reachable at a given cost.
    On an FPGA, a cheaper model needs fewer resources and has a lower latency, so aim up and to the left.
    """
    df = trials if signal is None else trials[trials["signal"] == signal]
    if df.empty:
        raise ValueError("No trials to plot" + (f" for signal {signal}" if signal else ""))
    per_trial = df.groupby("trial").agg(
        cost=(cost, "first"), acceptance=("acceptance", "first"),
        **{m: (m, "mean") for m in metrics},
    ).reset_index()

    labels = {"auc": "AUC", "efficiency": "signal efficiency"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        x, y = per_trial["cost"].to_numpy(float), per_trial[m].to_numpy(float)
        front = _pareto_front(x, y)
        ax.plot(x[front], y[front], color="gray", linestyle="--", linewidth=1, drawstyle="steps-post", zorder=1,
                label="Pareto front")
        ax.scatter(x, y, s=40, zorder=2)
        for xi, yi, name in zip(x, y, per_trial["trial"]):
            ax.annotate(name, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ylabel = labels.get(m, m)
        if m == "efficiency":
            acc = per_trial["acceptance"].unique()
            ylabel += f" at {acc[0]:g} acceptance" if len(acc) == 1 else " (mixed acceptances!)"
        ax.set_ylabel(("" if signal else "mean ") + ylabel)
        if cost != "n_layers":
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel(COST_LABELS.get(cost, cost))
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(signal if signal else f"averaged over {df['signal'].nunique()} signals")
    fig.tight_layout()
    if outpath:
        savefig(fig, outpath)
    return fig
