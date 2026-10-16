"""Write LEADERBOARD.md from the results of score.py.

    python leaderboard/render.py results.json

Teams are ranked by mean AUC (higher is better); ties go to the cheaper model (fewer BOPs). The submission
time and author of each team are those of the last commit that changed ``submissions/<team>/``. This script
only reads the JSON and git: it never loads a submitted model.
"""

import json
import os
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "LEADERBOARD.md"


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def repo_url() -> str:
    if os.environ.get("GITHUB_REPOSITORY"):
        return f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ['GITHUB_REPOSITORY']}"
    url = git("remote", "get-url", "origin")
    url = re.sub(r"^git@github\.com:", "https://github.com/", url)
    return re.sub(r"\.git$", "", url)


def github_login(sha: str):
    """GitHub username of a commit's author, or None if GitHub doesn't know it (or we can't ask)."""
    if not os.environ.get("GITHUB_REPOSITORY"):
        return None
    request = urllib.request.Request(f"{os.environ.get('GITHUB_API_URL', 'https://api.github.com')}/repos/"
                                     f"{os.environ['GITHUB_REPOSITORY']}/commits/{sha}")
    if os.environ.get("GITHUB_TOKEN"):
        request.add_header("Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return (json.load(response).get("author") or {}).get("login")
    except Exception:
        return None


def submitted(team: str, url: str) -> str:
    """'<commit> <date> by <author>' for the last commit that changed the team's folder."""
    log = git("log", "-1", "--format=%H%x09%cI%x09%an", "--", f"submissions/{team}")
    if not log:
        return ""
    sha, date, author = log.split("\t")
    date = datetime.fromisoformat(date).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
    login = github_login(sha)
    by = f"[{login}](https://github.com/{login})" if login else author
    return f"[`{sha[:7]}`]({url}/commit/{sha}) {date} by {by}"


def pareto(results: dict) -> set:
    """Teams that no other team beats on both mean AUC and BOPs."""
    return {a for a, r in results.items()
            if not any(s["mean_auc"] >= r["mean_auc"] and s["bops"] <= r["bops"]
                       and (s["mean_auc"], s["bops"]) != (r["mean_auc"], r["bops"])
                       for b, s in results.items() if b != a)}


def main():
    results = json.loads(Path(sys.argv[1]).read_text())
    ok = {team: r for team, r in results.items() if "error" not in r}
    failed = {team: r for team, r in results.items() if "error" in r}
    front = pareto(ok)
    url = repo_url()

    lines = [
        "# Leaderboard",
        "",
        "Every submission is evaluated on the test samples of `config/baseline.yml`: zero-bias run 396102 against the",
        "four benchmark signals. Ranked by **mean AUC**; ties go to the model with fewer BOPs. ★ marks the",
        "*Pareto front*: models that no other model beats on both mean AUC and BOPs.",
        "See [How to submit](README.md#how-to-submit) to add your team.",
        "",
        "| # | team | mean AUC | lowest AUC | highest AUC | reconstruction error | BOPs | submitted |",
        "|---|---|---|---|---|---|---|---|",
    ]
    ranked = sorted(ok, key=lambda t: (-ok[t]["mean_auc"], ok[t]["bops"], t.lower()))
    for rank, team in enumerate(ranked, 1):
        r = ok[team]
        star = " ★" if team in front else ""
        lines.append(
            f"| {rank} | [{team}](submissions/{team}){star} | {r['mean_auc']:.4f} | {r['min_auc']:.4f} "
            f"| {r['max_auc']:.4f} | {r['recon_error']:.4f} | {r['bops']:,} | {submitted(team, url)} |"
        )
    if not ok:
        lines.append("| | *no submissions yet* | | | | | | |")

    if failed:
        lines += ["", "## Submissions that could not be scored", "",
                  "Look at the log of the latest *Leaderboard* workflow run for the full error.", "",
                  "| team | error | submitted |", "|---|---|---|"]
        for team in sorted(failed, key=str.lower):
            error = failed[team]["error"][:200].replace("\n", " ").replace("|", "\\|").replace("<", "&lt;")
            lines.append(f"| [{team}](submissions/{team}) | {error} | {submitted(team, url)} |")

    lines += ["", f"*Last updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.*", ""]
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
