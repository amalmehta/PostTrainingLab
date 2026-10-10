"""Rollouts and the eval suite.

Metrics, each computed on a fixed set of held-out levels:
  success   greedy decoding solves the level
  pass@k    unbiased estimate that at least one of k sampled attempts solves it
  efficiency  optimal steps / taken steps, over solved episodes (1.0 = optimal)
  reward    mean verifiable reward under greedy decoding
"""
from math import comb

import numpy as np
import torch

from .env import MAX_STEPS, BatchEnv, expert_path, make_level


def make_levels(n, seed, dist="train"):
    rng = np.random.default_rng(seed)
    return [make_level(rng, dist) for _ in range(n)]


def rollout(policy, levels, greedy=False, temperature=1.0, record=False, gen=None):
    """Run one episode per level. With record=True also returns tokens,
    actions and per-step masks (for training) and agent paths (for plots)."""
    env = BatchEnv([g for g, _ in levels], [a for _, a in levels])
    toks, acts, masks, paths = [], [], [], [env.pos.copy()]
    for _ in range(MAX_STEPS):
        live = ~env.done
        if not live.any():
            break
        obs = env.obs()
        a = np.zeros(env.B, dtype=np.int64)  # finished episodes ignore actions
        a[live] = policy.act(obs[live], greedy=greedy, temperature=temperature, gen=gen)
        if record:
            toks.append(obs)
            acts.append(a)
            masks.append(live)
        env.step(a)
        paths.append(env.pos.copy())
    out = dict(success=env.success.copy(), steps=env.steps.copy(),
               reward=env.rewards())
    if record:
        out.update(tokens=np.stack(toks), actions=np.stack(acts),
                   masks=np.stack(masks), paths=np.stack(paths))
    return out


def pass_at_k(n, c, k):
    """Unbiased pass@k from n samples with c successes (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


@torch.no_grad()
def evaluate(policy, levels, optimal, n_samples=8, k=8, seed=0):
    policy.eval()
    g = rollout(policy, levels, greedy=True)
    solved = g["success"]
    eff = (optimal[solved] / g["steps"][solved]).mean() if solved.any() else 0.0
    gen = torch.Generator().manual_seed(seed)
    many = rollout(policy, levels * n_samples, gen=gen)
    wins = many["success"].reshape(n_samples, len(levels)).sum(0)
    return dict(success=float(solved.mean()),
                pass_at_k=float(np.mean([pass_at_k(n_samples, c, k) for c in wins])),
                sample_success=float(many["success"].mean()),
                efficiency=float(eff),
                reward=float(g["reward"].mean()))


class EvalSuite:
    """Fixed held-out level sets: in-distribution ("test") and OOD ("hard")."""

    def __init__(self, n=300):
        self.sets = {"test": make_levels(n, seed=10_001, dist="train"),
                     "hard": make_levels(n, seed=20_002, dist="hard")}
        self.optimal = {k: np.array([len(expert_path(g, a)) for g, a in v])
                        for k, v in self.sets.items()}

    def __call__(self, policy, **kw):
        return {k: evaluate(policy, v, self.optimal[k], **kw) for k, v in self.sets.items()}
