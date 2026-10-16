"""Score leaderboard submissions on the benchmark test set.

A submission is a folder ``submissions/<team>/`` holding what train.ipynb saves to ``output/<RUN_NAME>/``:
``best_model.pt`` and ``inputs.yml``. Every submission is evaluated the same way, on the test samples of
``config/baseline.yml`` (zero-bias run 396102 and the four benchmark signals), whatever config it was trained with.

    python leaderboard/score.py submissions/my_team         # score one submission and print the result
    python leaderboard/score.py output/my-model             # works on any folder saved by train.ipynb
    python leaderboard/score.py --all --out results.json    # score every submission (what the CI does)

The metrics are:
    recon_error: mean reconstruction error (masked MSE, the anomaly score) of the zero-bias test events.
                 It depends on the inputs and their scaling, so only compare models with the same inputs.
    mean_auc, min_auc, max_auc: AUC of each benchmark signal against zero bias, averaged / worst / best.
    bops:        bit operations, the FPGA cost estimate of utils.model_cost.
"""

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import dataset  # noqa: E402
from dataloader import create_datasets, load_config  # noqa: E402
from utils import compute_scores, load_inputs, load_model, model_cost, roc  # noqa: E402

CONFIG = REPO / "config" / "baseline.yml"
SUBMISSIONS = REPO / "submissions"
TEAM_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$")


def download_test_data():
    """Download the test samples of CONFIG, if they aren't there yet (only the test split, ~220 MB)."""
    cfg = load_config(CONFIG)
    root = Path(cfg.get("root", "."))
    root = root if root.is_absolute() else REPO / root
    for sample, split in cfg["test"]["background"] + cfg["test"]["signals"]:
        try:
            dataset.files(sample, split, root=root)
        except FileNotFoundError:
            dataset.download([sample], splits=[split], root=root)


_test_sets = {}  # test datasets, loaded once for each different input choice


def _test_set(inputs):
    key = json.dumps([inputs["layout"], inputs["features"], inputs["input_type"]])
    if key not in _test_sets:
        _, _, _test_sets[key] = create_datasets(str(CONFIG), layout=inputs["layout"], features=inputs["features"],
                                                input_type=inputs["input_type"], load_test=True, load_train=False)
    return _test_sets[key]


def score(folder) -> dict:
    """All the leaderboard metrics of the model saved in ``folder``."""
    folder = Path(folder)
    model_path = folder / "best_model.pt"
    for path in (model_path, folder / "inputs.yml"):
        if not path.exists():
            raise FileNotFoundError(f"{path.name} is missing: copy best_model.pt and inputs.yml from output/<RUN_NAME>/")

    model = load_model(model_path)
    inputs = load_inputs(model_path)
    test = _test_set(inputs)
    scores = {name: compute_scores(model, data) for name, data in test.items()}
    for name, s in scores.items():
        if not np.all(np.isfinite(s)):
            raise ValueError(f"the model gives NaN or infinite anomaly scores on {name}")

    aucs = {name: roc(scores["background"], s)[2] for name, s in scores.items() if name != "background"}
    cost = model_cost(model, test["background"].x.shape[1:])
    return {
        "recon_error": float(scores["background"].mean()),
        "mean_auc": float(np.mean(list(aucs.values()))),
        "min_auc": min(aucs.values()),
        "max_auc": max(aucs.values()),
        "bops": cost["bops"],
        "n_params": cost["n_params"],
        "aucs": aucs,
        "inputs": {"layout": inputs["layout"], "features": inputs["features"], "input_type": inputs["input_type"]},
    }


def print_result(name, result):
    print(f"\n{name}")
    print(f"  reconstruction error (zero bias): {result['recon_error']:.4f}")
    for signal, auc in result["aucs"].items():
        print(f"  AUC {signal:35s} {auc:.4f}")
    print(f"  mean AUC: {result['mean_auc']:.4f}   lowest: {result['min_auc']:.4f}   highest: {result['max_auc']:.4f}")
    print(f"  BOPs: {result['bops']:,}   parameters: {result['n_params']:,}")


def markdown(results) -> str:
    """The results as a Markdown table, for the summary of a GitHub Actions run."""
    lines = ["| team | mean AUC | lowest AUC | highest AUC | reconstruction error | BOPs |", "|---|---|---|---|---|---|"]
    for name, r in results.items():
        if "error" in r:
            lines.append(f"| {name} | **failed**: {r['error'][:200]} | | | | |".replace("\n", " ").replace("<", "&lt;"))
        else:
            lines.append(f"| {name} | {r['mean_auc']:.4f} | {r['min_auc']:.4f} | {r['max_auc']:.4f} "
                         f"| {r['recon_error']:.4f} | {r['bops']:,} |")
    lines.append("\nAUC of each signal: " + "; ".join(
        f"**{name}**: " + ", ".join(f"{sig} {auc:.4f}" for sig, auc in r["aucs"].items())
        for name, r in results.items() if "error" not in r))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="*", help="submission folders to score")
    parser.add_argument("--all", action="store_true", help="score every folder in submissions/")
    parser.add_argument("--out", help="write the results to this JSON file")
    parser.add_argument("--markdown", help="append the results to this file as a Markdown table")
    args = parser.parse_args()

    folders = [Path(f) for f in args.folders]
    if args.all:
        folders += sorted(p for p in SUBMISSIONS.glob("*") if p.is_dir())
    if not folders:
        parser.error("give a submission folder, or --all")

    download_test_data()
    results, failed = {}, False
    for folder in folders:
        name = folder.resolve().name
        if folder.resolve().parent == SUBMISSIONS and not TEAM_NAME.match(name):
            error = "team names may only use letters, digits, '_', '-' and '.' (at most 40 characters)"
            print(f"\n{name}: FAILED: {error}")
            results[name] = {"error": error}
            failed = True
            continue
        try:
            results[name] = score(folder)
            print_result(name, results[name])
        except Exception as e:
            traceback.print_exc()
            print(f"\n{name}: FAILED: {e}")
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            failed = True

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nWrote {args.out}")
    if args.markdown:
        with open(args.markdown, "a") as f:
            f.write(markdown(results))
    # When scoring every submission, one broken submission must not stop the leaderboard from updating.
    sys.exit(1 if failed and not args.all else 0)


if __name__ == "__main__":
    main()
