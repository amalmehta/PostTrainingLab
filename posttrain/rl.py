"""Stage 2: GRPO with a verifiable reward.

Each iteration samples `levels_per_iter` fresh training levels and runs
`group` attempts on each. An attempt's advantage is its reward minus the
group mean, divided by the group std (no value network). Each iteration
takes a few full-batch PPO epochs: minibatching the ~5k steps of one
iteration made each Adam step mostly noise. Every action in
the episode shares that advantage. The loss is the PPO-clipped policy
gradient plus beta * KL(policy || reference), where the reference is the
frozen SFT model. KL is exact over the 4 actions, averaged over steps.

A group whose attempts all score the same reward gives zero gradient.
The share of such groups ("dead groups") is logged: it is why RL from a
random init barely moves under a sparse reward.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .evals import rollout


def train_grpo(policy, ref, suite, sample_levels, iters=120, levels_per_iter=16,
               group=8, lr=3e-4, beta=0.02, epochs=4, minibatch=None, clip=0.2,
               eval_every=10, seed=0, log=print, snapshot=None, scale_by_std=True,
               reward_fn=None):
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(seed)
    hist = []
    for it in range(1, iters + 1):
        policy.eval()
        levels = sample_levels(levels_per_iter)
        batch = [lv for lv in levels for _ in range(group)]
        ro = rollout(policy, batch, record=True, gen=gen)

        # reward_fn swaps in a different reward (e.g. a hackable proxy); the
        # true reward is still logged alongside it.
        r_train = reward_fn(ro, batch) if reward_fn is not None else ro["reward"]
        R = r_train.reshape(levels_per_iter, group)
        std = R.std(1, keepdims=True)
        adv = R - R.mean(1, keepdims=True)
        if scale_by_std:
            adv = adv / (std + 1e-6)
        adv = adv.reshape(-1)
        dead = float((std[:, 0] < 1e-8).mean())

        m = ro["masks"]  # [T, B]
        X = torch.as_tensor(ro["tokens"][m])
        A = torch.as_tensor(ro["actions"][m])
        ADV = torch.as_tensor(np.broadcast_to(adv, m.shape)[m], dtype=torch.float32)
        with torch.no_grad():
            old_logp = policy(X).log_softmax(-1).gather(1, A[:, None])[:, 0]
            ref_logp = ref(X).log_softmax(-1) if ref is not None else None

        policy.train()
        n = len(X)
        stats = []
        for _ in range(epochs):
            for idx in torch.randperm(n, generator=gen).split(minibatch or n):
                logp_all = policy(X[idx]).log_softmax(-1)
                logp = logp_all.gather(1, A[idx, None])[:, 0]
                ratio = (logp - old_logp[idx]).exp()
                a = ADV[idx]
                pg = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
                p = logp_all.exp()
                ent = -(p * logp_all).sum(-1).mean()
                kl = ((p * (logp_all - ref_logp[idx])).sum(-1).mean()
                      if ref_logp is not None else torch.zeros(()))
                loss = pg + beta * kl
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                stats.append((kl.item(), ent.item()))

        kl_m, ent_m = np.mean(stats, axis=0)
        row = dict(iter=it, reward=float(ro["reward"].mean()), train_reward=float(r_train.mean()),
                   success=float(ro["success"].mean()),
                   length=float(ro["steps"].mean()), kl=float(kl_m),
                   entropy=float(ent_m), dead_groups=dead, tokens=int(n))
        if it % eval_every == 0 or it == 1:
            row["eval"] = suite(policy)
            log(f"grpo {it:4d} R {row['reward']:.3f} (train {row['train_reward']:.3f}) succ {row['success']:.3f} "
                f"len {row['length']:.1f} kl {row['kl']:.3f} ent {row['entropy']:.2f} dead {dead:.2f} | "
                f"test {row['eval']['test']['success']:.3f} (s {row['eval']['test']['sample_success']:.3f}) hard {row['eval']['hard']['success']:.3f}")
        if snapshot is not None:
            snapshot(it, policy)
        hist.append(row)
    return hist
