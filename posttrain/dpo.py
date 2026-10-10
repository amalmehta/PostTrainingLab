"""Stage 2 (alternative): offline DPO on self-generated preference pairs.

Sample `group` attempts per level from the frozen SFT model, once. In each
group, the chosen trajectory is the shortest successful attempt; the
rejected one is a failed attempt if there is one, otherwise the longest
successful attempt when it is at least `min_gap` steps longer. A trajectory's
log-probability is the sum of its action log-probs, and the loss is the DPO
objective  -log σ(β [(log πθ(τw) - log πref(τw)) - (log πθ(τl) - log πref(τl))]).

Unlike GRPO, no new samples are drawn during training: everything comes from
the SFT model up front.

Plain DPO only cares about the *gap* between chosen and rejected, and here it
pushes both likelihoods down (rejected much faster), which drains probability
from good actions and wrecks greedy decoding. `nll` > 0 adds the negative
log-likelihood of the chosen trajectories (per step), as in RPO, to hold them up.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .evals import rollout


def make_pairs(policy, levels, group=8, min_gap=2, chunk=64, seed=0):
    """Returns dict(tokens [N, L], actions [N], traj [N], pairs [P, 2], stats)."""
    gen = torch.Generator().manual_seed(seed)
    toks, acts, traj, pairs = [], [], [], []
    n_traj = 0
    n_fail_pairs = 0
    for start in range(0, len(levels), chunk):
        lv = levels[start:start + chunk]
        batch = [l for l in lv for _ in range(group)]
        ro = rollout(policy, batch, record=True, gen=gen)
        m = ro["masks"]  # [T, B]
        B = m.shape[1]
        # flatten each episode's steps, episode-major
        for b in range(B):
            k = m[:, b]
            toks.append(ro["tokens"][k, b])
            acts.append(ro["actions"][k, b])
            traj.append(np.full(int(k.sum()), n_traj + b))
        succ = ro["success"].reshape(len(lv), group)
        steps = ro["steps"].reshape(len(lv), group)
        for i in range(len(lv)):
            ids = n_traj + i * group + np.arange(group)
            wins = np.nonzero(succ[i])[0]
            if len(wins) == 0:
                continue
            w = wins[np.argmin(steps[i, wins])]
            fails = np.nonzero(~succ[i])[0]
            if len(fails):
                pairs.append((ids[w], ids[fails[0]]))
                n_fail_pairs += 1
            else:
                l = wins[np.argmax(steps[i, wins])]
                if steps[i, l] - steps[i, w] >= min_gap:
                    pairs.append((ids[w], ids[l]))
        n_traj += B
    return dict(tokens=np.concatenate(toks), actions=np.concatenate(acts),
                traj=np.concatenate(traj), pairs=np.array(pairs, dtype=np.int64),
                stats=dict(episodes=n_traj, pairs=len(pairs), fail_pairs=n_fail_pairs))


def traj_logps(model, tokens, actions, traj, n_traj):
    """Sum of action log-probs per trajectory id (ids are 0..n_traj-1)."""
    lp = model(tokens).log_softmax(-1).gather(1, actions[:, None])[:, 0]
    return torch.zeros(n_traj).index_add(0, traj, lp)


def train_dpo(policy, ref, data, suite, epochs=4, batch_pairs=64, lr=1e-4, beta=0.1, nll=0.0,
              eval_every=50, seed=0, log=print, tag="dpo"):
    T = torch.as_tensor(data["tokens"])
    A = torch.as_tensor(data["actions"])
    tr = torch.as_tensor(data["traj"])
    pairs = torch.as_tensor(data["pairs"])
    # index of each trajectory's steps, so a minibatch can gather just its own
    order = torch.argsort(tr, stable=True)
    counts = torch.bincount(tr)
    starts = torch.cumsum(counts, 0) - counts

    def gather(ids):
        idx = torch.cat([order[starts[i]:starts[i] + counts[i]] for i in ids.tolist()])
        local = torch.repeat_interleave(torch.arange(len(ids)), counts[ids])
        return T[idx], A[idx], local

    with torch.no_grad():  # reference log-probs for every trajectory, once
        ref.eval()
        ref_lp = torch.zeros(len(counts))
        for chunk in torch.arange(len(counts)).split(512):
            t, a, local = gather(chunk)
            ref_lp[chunk] = traj_logps(ref, t, a, local, len(chunk))

    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(seed)
    hist, step = [], 0
    for ep in range(epochs):
        for bidx in torch.randperm(len(pairs), generator=gen).split(batch_pairs):
            policy.train()
            ids = pairs[bidx].reshape(-1)  # chosen, rejected interleaved
            t, a, local = gather(ids)
            lp = traj_logps(policy, t, a, local, len(ids)).view(-1, 2)
            rl = ref_lp[ids].view(-1, 2)
            margin = beta * ((lp[:, 0] - rl[:, 0]) - (lp[:, 1] - rl[:, 1]))
            loss = -F.logsigmoid(margin).mean()
            if nll:
                chosen_steps = counts[ids].view(-1, 2)[:, 0].float()
                loss = loss + nll * (-(lp[:, 0] / chosen_steps).mean())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1
            row = dict(step=step, epoch=ep, loss=loss.item(),
                       acc=(margin > 0).float().mean().item(),
                       chosen=(lp[:, 0] - rl[:, 0]).mean().item(),
                       rejected=(lp[:, 1] - rl[:, 1]).mean().item())
            if step % eval_every == 0 or step == 1:
                row["eval"] = suite(policy)
                log(f"{tag} {step:4d} loss {row['loss']:.3f} acc {row['acc']:.2f} "
                    f"Δchosen {row['chosen']:+.2f} Δrejected {row['rejected']:+.2f} | "
                    f"test {row['eval']['test']['success']:.3f} hard {row['eval']['hard']['success']:.3f}")
            hist.append(row)
    if "eval" not in hist[-1]:
        hist[-1]["eval"] = suite(policy)
    return hist
