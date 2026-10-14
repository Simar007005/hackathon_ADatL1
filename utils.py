import copy
import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import cloudpickle
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from tqdm.auto import tqdm

import dataset
from models.autoencoder import anomaly_score, masked_mse


def get_device() -> torch.device:
    """The GPU if PyTorch can see one, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _bit_width(quant_proxy, default):
    """Bit width of a Brevitas quantizer, or ``default`` if it is disabled (or not Brevitas)."""
    if quant_proxy is None or not getattr(quant_proxy, "is_quant_enabled", False):
        return default
    return float(quant_proxy.bit_width())


@torch.no_grad()
def model_cost(model, input_shape, default_bits: int = 16) -> Dict[str, int]:
    """How big and how slow a model would be on an FPGA, without converting it to firmware.

    Returns, for one event:
      n_params: number of trainable parameters.
      bops:     bit operations of the dense (Linear) layers, the usual estimate of the FPGA area a
                quantized network needs (Hawks et al. 2021, arXiv:2102.11289). For a layer with n inputs,
                m outputs, b_w-bit weights, b_a-bit inputs and a fraction f of weights pruned to 0:
                    m * n * ((1 - f) * b_a * b_w + b_a + b_w + log2(n))
                Layers that are not quantized with Brevitas count as ``default_bits`` (16 bits is
                what hls4ml uses by default).
      n_layers: number of dense layers an event goes through. Each one costs a few clock cycles,
                so this is a rough measure of the latency.

    Quantize with Brevitas (``brevitas.nn.QuantLinear``, ``QuantReLU``, ``QuantIdentity``, ...) and
    prune weights to lower the BOPs.
    """
    bits = default_bits  # bit width of the values flowing into the next layer
    cost = {"n_params": count_parameters(model), "bops": 0.0, "n_layers": 0}

    def hook(module, inputs, output):
        nonlocal bits
        if isinstance(module, nn.Linear):  # also brevitas.nn.QuantLinear
            n, m = module.in_features, module.out_features
            n_vectors = inputs[0].numel() // n  # e.g. one per object for "objects" input
            b_a = _bit_width(getattr(module, "input_quant", None), bits)
            b_w = _bit_width(getattr(module, "weight_quant", None), default_bits)
            pruned = float((module.weight == 0).float().mean())
            cost["bops"] += n_vectors * m * n * ((1 - pruned) * b_a * b_w + b_a + b_w + math.log2(n))
            cost["n_layers"] += 1
            bits = _bit_width(getattr(module, "output_quant", None), default_bits)
        elif hasattr(module, "act_quant"):  # Brevitas activations: QuantReLU, QuantIdentity, ...
            bits = _bit_width(module.act_quant, bits)

    was_training = model.training
    model.eval()
    handles = [m.register_forward_hook(hook) for m in model.modules()]
    try:
        device = next(model.parameters()).device
        model(torch.zeros(1, *input_shape, device=device))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    cost["bops"] = int(round(cost["bops"]))
    return cost


@torch.no_grad()
def summary(model, input_shape):
    """Print the model layer by layer (output shape, parameters), then its cost (see ``model_cost``)."""
    rows = []

    def make_hook(name):
        def hook(module, inputs, output):
            output = getattr(output, "value", output)  # Brevitas may return a QuantTensor
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            rows.append((name, type(module).__name__, tuple(output.shape[1:]), n_params))
        return hook

    layers = list(model.named_children()) or [("model", model)]
    handles = [layer.register_forward_hook(make_hook(name)) for name, layer in layers]
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        model(torch.zeros(1, *input_shape, device=device))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    print(f"{'layer':8s} {'type':16s} {'output shape':14s} {'parameters':>10s}")
    for name, kind, shape, n_params in rows:
        print(f"{name:8s} {kind:16s} {str(shape):14s} {n_params:10d}")
    cost = model_cost(model, input_shape)
    print(f"\nparameters: {cost['n_params']:,}   bit operations (BOPs): {cost['bops']:,}   dense layers: {cost['n_layers']}")


def compute_scores(model, data) -> np.ndarray:
    """Anomaly score of every event in an EventData."""
    return anomaly_score(model, data.x, data.mask)


def fit(model, train, val, n_epochs: int, batch_size: int = 1024, learning_rate: float = 1e-3,
        patience: int = None, device=None) -> Dict[str, list]:
    """Train an autoencoder on ``train`` (an EventData) to minimise the masked reconstruction error.

    patience: early stopping. Stop once the validation loss has not improved for ``patience`` epochs
              in a row, and restore the weights of the best epoch. None = always train for ``n_epochs``.

    Returns the loss history {"train": [(epoch, loss), ...], "val": [...]} for ``plotting.plot_loss_curve``.
    """
    device = device or get_device()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x, mask = torch.as_tensor(train.x, device=device), torch.as_tensor(train.mask, device=device)

    history = {"train": [], "val": []}
    best_loss, best_state, bad_epochs = math.inf, None, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(x), device=device)
        total = 0.0
        for i in tqdm(range(0, len(x), batch_size), desc=f"epoch {epoch}/{n_epochs}", leave=False):
            idx = order[i:i + batch_size]
            x_hat = model(x[idx])
            loss = masked_mse(getattr(x_hat, "value", x_hat), x[idx], mask[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)

        train_loss = total / len(x)
        val_loss = float(compute_scores(model, val).mean())
        history["train"].append((epoch, train_loss))
        history["val"].append((epoch, val_loss))
        print(f"epoch {epoch}/{n_epochs}: loss = {train_loss:.4f}, val_loss = {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        elif patience is not None:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping: no improvement for {patience} epochs")
                break

    if patience is not None and best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored the weights of the best epoch (val_loss = {best_loss:.4f})")
    return history


def save_model(model, build, input_shape, path):
    """Save a model so that ``load_model`` can rebuild it in any notebook.

    The file holds the trained weights, plus the function that built the model (``build``, called
    with ``input_shape``) so that the architecture can be built again. ``build`` is stored with
    cloudpickle, so this also works for models defined in a notebook, and for Brevitas models.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"build": cloudpickle.dumps(build), "input_shape": tuple(int(n) for n in input_shape),
                "state_dict": model.state_dict()}, path)


def load_model(path):
    """Load a model saved by ``save_model`` (train.ipynb) onto the CPU."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = pickle.loads(checkpoint["build"])(checkpoint["input_shape"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def load_inputs(model_path) -> dict:
    """The input choice a model was trained with: {"layout": ..., "features": ..., "input_type": ...}.

    train.ipynb saves it as inputs.yml next to the model.
    """
    path = Path(model_path).parent / "inputs.yml"
    if not path.exists():
        print(f"[WARNING] {path} not found, assuming the default layout and features (Et, eta, phi), flat input")
        inputs = {"layout": dict(dataset.DEFAULT_LAYOUT), "features": ["Et", "eta", "phi"]}
    else:
        with open(path) as f:
            inputs = yaml.safe_load(f)
    inputs.setdefault("input_type", "flat")
    return inputs


# ---------------------------------------------------------------------------
# Metrics. In all of them, background = normal (zero-bias) events, signal = anomalies.
# ---------------------------------------------------------------------------
def _labels_and_scores(bkg_scores, sig_scores):
    y = np.r_[np.zeros(len(bkg_scores)), np.ones(len(sig_scores))]
    return y, np.r_[bkg_scores, sig_scores]


def roc(bkg_scores: np.ndarray, sig_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (background acceptance, signal efficiency, AUC)."""
    y, s = _labels_and_scores(bkg_scores, sig_scores)
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr, float(roc_auc_score(y, s))


def threshold_at_acceptance(bkg_scores: np.ndarray, acceptance: float) -> float:
    """Score threshold that keeps the given fraction of background events."""
    return float(np.quantile(bkg_scores, 1.0 - acceptance))


def efficiency_at_acceptance(bkg_scores: np.ndarray, sig_scores: np.ndarray, acceptance: float) -> float:
    """Fraction of signal events above the threshold that keeps `acceptance` of the background.

    This is how a trigger is judged: the rate of normal events it may keep is fixed
    by the bandwidth budget, and we want to catch as much signal as possible within it.
    """
    thr = threshold_at_acceptance(bkg_scores, acceptance)
    return float(np.mean(sig_scores > thr))


# ---------------------------------------------------------------------------
# Trials: a log of every evaluated model, so you can compare them all at the end.
# ---------------------------------------------------------------------------
TRIALS_PATH = Path("output/trials.csv")


def save_trial(name: str, metrics: Dict[str, Dict[str, float]], cost: Dict[str, int], model_path: str = "",
               config: str = "", acceptance: float = None, notes: str = "", path=TRIALS_PATH) -> pd.DataFrame:
    """Append one trial to the trials file (one row per signal) and return all trials.

    metrics: {signal name: {"auc": ..., "efficiency": ...}}
    cost:    the model's size, from ``model_cost``: {"n_params": ..., "bops": ..., "n_layers": ...}
    Saving a trial under a name that already exists replaces the old entry.
    """
    path = Path(path)
    trials = load_trials(path)
    if not trials.empty:
        trials = trials[trials["trial"] != name]
    timestamp = datetime.now().isoformat(timespec="seconds")
    new = pd.DataFrame([
        {"trial": name, "signal": signal, "auc": m["auc"], "efficiency": m["efficiency"],
         "acceptance": acceptance, **cost, "model_path": str(model_path),
         "config": str(config), "notes": notes, "timestamp": timestamp}
        for signal, m in metrics.items()
    ])
    trials = pd.concat([trials, new], ignore_index=True) if not trials.empty else new
    path.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(path, index=False)
    print(f"[TRIAL] saved '{name}' to {path} ({trials['trial'].nunique()} trials in total)")
    return trials


def load_trials(path=TRIALS_PATH) -> pd.DataFrame:
    """All saved trials, one row per (trial, signal). Empty DataFrame if none were saved yet."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False)


def delete_trial(name: str, path=TRIALS_PATH) -> pd.DataFrame:
    """Remove a trial from the trials file."""
    trials = load_trials(path)
    if not trials.empty:
        trials = trials[trials["trial"] != name]
        trials.to_csv(path, index=False)
    return trials
