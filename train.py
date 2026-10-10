"""Run the post-training pipeline for one seed.

  random init --SFT on habit-prone demos--> SFT --GRPO + KL to SFT---------> GRPO
                                            SFT --offline DPO (+NLL) on SFT samples--> DPO
                                            SFT --the same, without the NLL term--> plain DPO
                                            SFT --GRPO on a hackable proxy----> Reward hacker
  random init --GRPO, no SFT, same budget----------------------------------> RL from scratch

Writes runs/seed<N>/: results.json, <stage>.pt, and snaps/<run>_<iter>.pt
(GRPO and reward-hacker snapshots for the website's training scrubber).
Run several seeds (in parallel if you like), then `python export.py` to build
the website data. `--quick` shrinks everything and writes to runs_quick/.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from posttrain.dpo import make_pairs, train_dpo
from posttrain.evals import EvalSuite, rollout
from posttrain.hacking import closeness_reward
from posttrain.model import Policy
from posttrain.rl import train_grpo
from posttrain.sft import level_hash, make_demos, sample_train_levels, train_sft

ROOT = Path(__file__).resolve().parent
STAGES = ["random", "sft", "grpo", "dpo", "dpo_plain", "hack", "scratch"]
LABELS = {"random": "Random init", "sft": "SFT", "grpo": "SFT → GRPO", "dpo": "SFT → DPO",
          "dpo_plain": "Plain DPO (no NLL)", "hack": "Reward hacker", "scratch": "RL from scratch"}


def config(quick):
    q = quick
    return dict(eval_levels=40 if q else 150, demos=200 if q else 3000, noise=0.0, habit=0.8,
                sft_steps=100 if q else 1500, sft_lr=1.5e-3,
                grpo_iters=6 if q else 120, levels_per_iter=32, group=8,
                grpo_lr=3e-4, beta=0.02, eval_every=3 if q else 10, snap_every=3 if q else 20,
                # DPO draws the same number of episodes as GRPO, all from SFT up front
                dpo_epochs=1 if q else 3, dpo_lr=1e-4, dpo_beta=0.1, dpo_nll=1.0, dpo_eval_every=10 if q else 40,
                gallery_levels=4, gallery_samples=8 if q else 24)


@torch.no_grad()
def proxy_eval(policy, levels):
    """Mean proxy (closeness) reward of greedy rollouts."""
    ro = rollout(policy, levels, greedy=True, record=True)
    return float(closeness_reward(ro, levels).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = config(args.quick)
    out = ROOT / ("runs_quick" if args.quick else "runs") / f"seed{args.seed}"
    (out / "snaps").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log_lines = []

    def log(msg):
        line = f"[seed {args.seed} {time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + 1)
    suite = EvalSuite(cfg["eval_levels"])  # same held-out levels for every seed
    banned = {level_hash(lv) for v in suite.sets.values() for lv in v}

    def sample_levels(n):
        return sample_train_levels(n, rng, banned)

    log(f"making {cfg['demos']} demos (habit {cfg['habit']}, noise {cfg['noise']})")
    demos = make_demos(sample_levels(cfg["demos"]), cfg["noise"], rng,
                       habit=cfg["habit"], successes_only=True)
    val = make_demos(sample_levels(200), 0.0, rng)  # clean expert labels
    log(f"demonstrator solved {demos[2]:.1%} of levels; kept {len(demos[0])} transitions")

    policies, evals, hist = {}, {}, {}
    policies["random"] = Policy()
    init_state = copy.deepcopy(policies["random"].state_dict())
    evals["random"] = suite(policies["random"])

    sft = Policy()
    sft.load_state_dict(init_state)
    hist["sft"] = train_sft(sft, demos, val, suite, steps=cfg["sft_steps"], lr=cfg["sft_lr"],
                            eval_every=max(cfg["sft_steps"] // 5, 1), seed=args.seed, log=log)
    policies["sft"] = sft
    evals["sft"] = hist["sft"][-1].get("eval") or suite(sft)

    def frozen(model):
        ref = copy.deepcopy(model).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        return ref

    def snapper(name):
        torch.save(sft.state_dict(), out / "snaps" / f"{name}_0.pt")

        def snap(it, pol):
            if it % cfg["snap_every"] == 0:
                torch.save(pol.state_dict(), out / "snaps" / f"{name}_{it}.pt")
        return snap

    def grpo_run(name, start, ref, beta, reward_fn=None, snapshot=None):
        log(f"{name}: GRPO" + (" on the proxy reward" if reward_fn else ""))
        pol = copy.deepcopy(start)
        hist[name] = train_grpo(pol, ref, suite, sample_levels, iters=cfg["grpo_iters"],
                                levels_per_iter=cfg["levels_per_iter"], group=cfg["group"],
                                lr=cfg["grpo_lr"], beta=beta, eval_every=cfg["eval_every"],
                                seed=args.seed, log=log, snapshot=snapshot, reward_fn=reward_fn)
        policies[name] = pol
        evals[name] = hist[name][-1].get("eval") or suite(pol)

    grpo_run("grpo", sft, frozen(sft), cfg["beta"], snapshot=snapper("grpo"))

    log("dpo: sampling preference pairs from SFT")
    n_dpo_levels = cfg["grpo_iters"] * cfg["levels_per_iter"]
    pairs = make_pairs(sft, sample_levels(n_dpo_levels), group=cfg["group"], seed=args.seed)
    log(f"dpo: {pairs['stats']}")
    for name, nll in [("dpo", cfg["dpo_nll"]), ("dpo_plain", 0.0)]:
        pol = copy.deepcopy(sft)
        hist[name] = train_dpo(pol, frozen(sft), pairs, suite, epochs=cfg["dpo_epochs"], lr=cfg["dpo_lr"],
                               beta=cfg["dpo_beta"], nll=nll, eval_every=cfg["dpo_eval_every"],
                               seed=args.seed, log=log, tag=name)
        policies[name] = pol
        evals[name] = hist[name][-1]["eval"]

    grpo_run("hack", sft, frozen(sft), cfg["beta"], reward_fn=closeness_reward, snapshot=snapper("hack"))

    scratch = Policy()
    scratch.load_state_dict(init_state)
    grpo_run("scratch", scratch, None, 0.0)

    for s in STAGES:  # how each checkpoint scores on the hackable proxy, for contrast
        evals[s]["test"]["proxy"] = proxy_eval(policies[s], suite.sets["test"])
        torch.save(policies[s].state_dict(), out / f"{s}.pt")

    results = dict(seed=args.seed, config=cfg, labels=LABELS, stages=STAGES, evals=evals,
                   history=hist, demo_transitions=int(len(demos[0])), demo_success=demos[2],
                   dpo_stats=pairs["stats"], runtime_sec=time.time() - t0, log=log_lines)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    log("done")
    for s in STAGES:
        e = evals[s]
        log(f"{LABELS[s]:16s} test {e['test']['success']:.3f} pass@8 {e['test']['pass_at_k']:.3f} "
            f"proxy {e['test']['proxy']:.3f} | hard {e['hard']['success']:.3f}")


if __name__ == "__main__":
    main()
