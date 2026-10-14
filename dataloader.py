"""Turn a data config (config/*.yml) into arrays ready for training and evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

import dataset

REPO = Path(__file__).resolve().parent


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _root(cfg: dict) -> Path:
    root = Path(cfg.get("root", "."))
    return root if root.is_absolute() else REPO / root


def download_config(config_path: str):
    """Download every [sample, split] that a config refers to."""
    cfg = load_config(config_path)
    entries = cfg["train"] + cfg["val"] + cfg["test"]["background"] + cfg["test"]["signals"]
    for sample, split in entries:
        dataset.download([sample], splits=[split], root=_root(cfg))


def preprocess(x: np.ndarray, mask: np.ndarray, features: Sequence[str], scaling: Optional[dict]) -> np.ndarray:
    """Apply the scaling from the config to real objects; padded slots stay 0.

    ``features`` names the last axis of ``x``; ``scaling["mean"]`` / ``scaling["std"]`` are keyed by feature.
    """
    x = x.copy()
    if scaling is not None:
        if scaling.get("log_et", False) and "Et" in features:
            i = list(features).index("Et")
            x[..., i] = np.log1p(x[..., i])
        mean = np.asarray([scaling["mean"][f] for f in features], dtype=np.float32)
        std = np.asarray([scaling["std"][f] for f in features], dtype=np.float32)
        x = (x - mean) / std
    x[~mask] = 0.0
    return x


def load_arrays(
    entries: List[List[str]], cfg: dict, max_events: Optional[int], layout: Dict[str, int],
    features: Sequence[str], apply_scaling: bool = True, chunk_size: int = 20_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate a list of [sample, split] entries.

    ``layout`` and ``features`` are passed to ``dataset.to_padded``: how many objects of each
    collection to keep (highest Et first) and which features of each object.

    The events are read ``chunk_size`` at a time and written straight into the output arrays,
    so memory use stays close to the size of the result (this matters on small machines like Binder).

    Returns ``x`` (N, n_objects, n_features), ``mask`` (N, n_objects) and the
    integer ``label`` of each event (0 = zero bias, <0 = simulated normal, >0 = signal).
    """
    root = _root(cfg)
    scaling = cfg.get("scaling") if apply_scaling else None

    # Upper bound on the number of events; fewer remain after dropping removed events.
    capacity = 0
    for sample, split in entries:
        n = dataset.num_rows(sample, split, root=root)
        capacity += n if max_events is None else min(n, max_events)
    n_objects = sum(layout.values())
    x = np.empty((capacity, n_objects, len(features)), dtype=np.float32)
    mask = np.empty((capacity, n_objects), dtype=bool)
    label = np.empty(capacity, dtype=np.int64)

    n = 0
    for sample, split in entries:
        for events in dataset.iter_events(
            sample, split, max_events=max_events, chunk_size=chunk_size, physical_units=True, root=root
        ):
            x_chunk, mask_chunk = dataset.to_padded(events, layout, features)
            if apply_scaling:
                x_chunk = preprocess(x_chunk, mask_chunk, features, scaling)
            k = len(x_chunk)
            x[n:n + k], mask[n:n + k] = x_chunk, mask_chunk
            label[n:n + k] = np.asarray(events.label, dtype=np.int64)
            n += k
    return x[:n], mask[:n], label[:n]


def _to_input(x: np.ndarray, input_type: str) -> np.ndarray:
    if input_type == "flat":
        return x.reshape(len(x), -1)
    if input_type == "objects":
        return x
    raise ValueError(f"Unknown input_type: {input_type}")


@dataclass
class EventData:
    """Model inputs for a set of events.

    ``x`` is (N, n_objects * n_features) for "flat" input or (N, n_objects, n_features) for
    "objects" input. ``mask`` is (N, n_objects): True for real objects, False for padding.
    """

    x: np.ndarray
    mask: np.ndarray

    def __len__(self) -> int:
        return len(self.x)


def make_dataset(x, mask, input_type: str = "flat") -> EventData:
    return EventData(_to_input(x, input_type).astype(np.float32, copy=False), mask.astype(bool, copy=False))


def create_datasets(
    config_path: str,
    layout: Dict[str, int],
    features: Sequence[str],
    input_type: str = "flat",  # "flat": (N, n_objects * n_features), "objects": (N, n_objects, n_features)
    apply_scaling: bool = True,
    load_test: bool = False,
    load_train: bool = True,
) -> Tuple[Optional[EventData], Optional[EventData], Optional[Dict[str, EventData]]]:
    """Build the train / val datasets, and optionally one test dataset per sample.

    The samples come from the config; ``layout`` and ``features`` choose what goes into the model
    (see ``dataset.to_padded``), e.g. ``layout={"MET": 1, "jets": 10}``, ``features=["Et", "eta", "phi"]``.

    The test datasets are keyed by sample name: the background first, then each signal.
    With ``load_train=False`` only the test datasets are loaded (train and val are None).
    """
    cfg = load_config(config_path)
    max_events = cfg.get("max_events", {})

    def load(entries, n):
        x, mask, _ = load_arrays(entries, cfg, n, layout, features, apply_scaling)
        return make_dataset(x, mask, input_type)

    train = val = None
    if load_train:
        train = load(cfg["train"], max_events.get("train"))
        val = load(cfg["val"], max_events.get("val"))

    test = None
    if load_test:
        test = {}
        n = max_events.get("test")
        test["background"] = load(cfg["test"]["background"], n)
        for sample, split in cfg["test"]["signals"]:
            test[sample] = load([[sample, split]], n)

    return train, val, test


if __name__ == "__main__":
    # Example usage: python dataloader.py
    train, val, _ = create_datasets("config/baseline.yml", dataset.DEFAULT_LAYOUT, ["Et", "eta", "phi"])

    print(f"Training events:   {len(train):,}  x shape = {train.x.shape}, mask shape = {train.mask.shape}")
    print(f"Validation events: {len(val):,}")
