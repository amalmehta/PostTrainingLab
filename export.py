"""Aggregate trained seeds into the website's data files.

Reads runs/seed*/ (from train.py) and writes site/data/results.js and
site/data/checkpoints.js. Metrics and curves are kept per seed (the page
draws the mean with a min-max band); the trajectory gallery, difficulty
buckets, live demo and training scrubber use seed 0's checkpoints.

  python export.py            # from runs/
  python export.py --quick    # from runs_quick/
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from posttrain.env import SIZE, expert_path
from posttrain.evals import EvalSuite, rollout
from posttrain.model import Policy, export_weights

ROOT = Path(__file__).resolve().parent
GALLERY_STAGES = ["sft", "dpo", "grpo", "hack"]   # random / scratch fans are all failures
BUCKET_STAGES = ["sft", "dpo", "grpo"]


def level_json(level):
    g, a = level
    return dict(grid="".join(map(str, g.reshape(-1))), agent=[int(a[0]), int(a[1])],
                optimal=len(expert_path(g, a)))


def episode(ro, i):
    k = int(ro["steps"][i]) + 1
    return dict(path=ro["paths"][:k, i].tolist(), success=bool(ro["success"][i]),
                steps=int(ro["steps"][i]))


@torch.no_grad()
def build_gallery(policies, suite, n_levels, n_samples):
    """Test levels of rising difficulty: sampled fans + the greedy path per checkpoint."""
    test, opt = suite.sets["test"], suite.optimal["test"]
    order = np.argsort(opt, kind="stable")
    picks = [int(order[int(f * (len(order) - 1))]) for f in np.linspace(0.35, 0.95, n_levels)]
    gallery = []
    for j, i in enumerate(picks):
        runs, greedy = {}, {}
        for s in [x for x in GALLERY_STAGES if x in policies]:
            ro = rollout(policies[s], [test[i]] * n_samples, record=True,
                         gen=torch.Generator().manual_seed(100 + j))
            runs[s] = [episode(ro, k) for k in range(n_samples)]
            greedy[s] = episode(rollout(policies[s], [test[i]], greedy=True, record=True), 0)
        gallery.append(dict(level=level_json(test[i]), runs=runs, greedy=greedy))
    return gallery


@torch.no_grad()
def by_optimal_length(policy, levels, optimal):
    """Greedy success bucketed by optimal path length."""
    ro = rollout(policy, levels, greedy=True)
    edges = [0, 8, 12, 16, 20, 99]
    return [dict(lo=lo, hi=hi, n=int(m.sum()), success=float(ro["success"][m].mean()))
            for lo, hi in zip(edges[:-1], edges[1:])
            for m in [(optimal >= lo) & (optimal < hi)] if m.any()]


def load(path):
    p = Policy()
    p.load_state_dict(torch.load(path))
    return p.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    runs = ROOT / ("runs_quick" if args.quick else "runs")
    seed_dirs = sorted((d for d in runs.glob("seed*") if (d / "results.json").exists()),
                       key=lambda d: int(d.name[4:]))
    if not seed_dirs:
        raise SystemExit(f"no finished seeds in {runs}/ — run train.py first")
    per_seed = [json.loads((d / "results.json").read_text()) for d in seed_dirs]
    base = per_seed[0]
    cfg = base["config"]
    STAGES, LABELS = base["stages"], base["labels"]  # whatever stages these runs trained
    suite = EvalSuite(cfg["eval_levels"])

    s0 = seed_dirs[0]
    policies = {s: load(s0 / f"{s}.pt") for s in STAGES}
    gallery = build_gallery(policies, suite, cfg["gallery_levels"], cfg["gallery_samples"])
    buckets = {s: by_optimal_length(policies[s], suite.sets["test"], suite.optimal["test"])
               for s in BUCKET_STAGES if s in STAGES}

    results = dict(
        config=cfg, labels=LABELS, stages=STAGES, grid_size=SIZE,
        seeds=[r["seed"] for r in per_seed],
        evals=[r["evals"] for r in per_seed],          # one entry per seed
        history=[r["history"] for r in per_seed],      # one entry per seed
        demo_transitions=[r["demo_transitions"] for r in per_seed],
        demo_success=[r["demo_success"] for r in per_seed],
        dpo_stats=[r["dpo_stats"] for r in per_seed],
        runtime_sec=[r["runtime_sec"] for r in per_seed],
        gallery=gallery, buckets=buckets,
        pools={k: [level_json(lv) for lv in v[:100]] for k, v in suite.sets.items()})

    # Live-demo checkpoints (float32) + training snapshots (float16) from seed 0.
    ckpts = {s: dict(label=LABELS[s], weights=export_weights(policies[s])) for s in STAGES}
    snaps = {}
    for f in sorted((s0 / "snaps").glob("*.pt")):
        run, it = re.fullmatch(r"(\w+?)_(\d+)", f.stem).groups()
        snaps.setdefault(run, {})[int(it)] = export_weights(load(f), dtype="f2")
    snaps = {run: [dict(iter=it, weights=w) for it, w in sorted(d.items())] for run, d in snaps.items()}

    data = ROOT / "site" / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "results.js").write_text("window.RESULTS = " + json.dumps(results) + ";\n")
    (data / "checkpoints.js").write_text(
        "window.CHECKPOINTS = " + json.dumps(ckpts) + ";\nwindow.SNAPSHOTS = " + json.dumps(snaps) + ";\n")
    sizes = {f.name: f"{f.stat().st_size / 1e6:.1f} MB" for f in data.iterdir()}
    print(f"exported {len(per_seed)} seed(s) from {runs}/ -> {sizes}")
    for s in STAGES:
        vals = [r["evals"][s]["test"]["success"] for r in per_seed]
        print(f"{LABELS[s]:16s} test success {np.mean(vals):.3f} (range {min(vals):.3f}–{max(vals):.3f})")


if __name__ == "__main__":
    main()
