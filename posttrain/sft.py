"""Stage 1: supervised fine-tuning (behaviour cloning) on human-style demos.

The demonstrator is a planner (BFS) with a habit: at each step, with
probability `habit`, it follows its "instinct" instead of planning: step
straight toward the current target (the key, then the goal), ignoring walls
and the door. With probability `noise` it presses a random key. Whatever it
did is recorded as the label, so SFT clones the habit along with the skill.
Greedy decoding cannot strip a habit that is the most likely action, which
is what RL later has to fix.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .env import DOOR, GOAL, KEY, MAX_STEPS, MOVES, WALL, BatchEnv, _bfs, make_level


def level_hash(level):
    g, a = level
    return hash((g.tobytes(), tuple(a)))


def sample_train_levels(n, rng, banned):
    """Training levels, skipping any that also appear in an eval set."""
    out = []
    while len(out) < n:
        lv = make_level(rng, "train")
        if level_hash(lv) not in banned:
            out.append(lv)
    return out


def expert_action(grid, pos, has_key):
    target = tuple(np.argwhere(grid == (GOAL if has_key else KEY))[0])
    return _bfs(grid, tuple(pos), target, door_open=has_key)[0]


def instinct_action(grid, pos, has_key, rng):
    """Walk straight at the target: close the larger axis gap first, then the
    other; if both are blocked (or the target is aligned behind a wall), step
    in a random open direction."""
    target = np.argwhere(grid == (GOAL if has_key else KEY))[0]
    dr, dc = target[0] - pos[0], target[1] - pos[1]
    vert = (1 if dr > 0 else 0, abs(dr))     # (down / up, gap)
    horiz = (3 if dc > 0 else 2, abs(dc))    # (right / left, gap)
    order = [vert, horiz] if abs(dr) >= abs(dc) else [horiz, vert]
    prefs = [a for a, gap in order if gap > 0]

    def open_(a):
        cell = grid[pos[0] + MOVES[a][0], pos[1] + MOVES[a][1]]
        return cell != WALL and not (cell == DOOR and not has_key)

    for a in prefs:
        if open_(a):
            return a
    return int(rng.choice([a for a in range(4) if open_(a)]))


def make_demos(levels, noise, rng, habit=0.0, successes_only=False):
    """Roll out the habit-prone demonstrator; return tokens [N, L], labels [N].
    With successes_only, transitions from episodes that timed out are dropped."""
    env = BatchEnv([g for g, _ in levels], [a for _, a in levels])
    X, Y, E = [], [], []
    for _ in range(MAX_STEPS):
        live = np.nonzero(~env.done)[0]
        if len(live) == 0:
            break
        obs = env.obs()
        acts = np.zeros(env.B, dtype=np.int64)
        for i in live:
            u = rng.random()
            if u < noise:
                acts[i] = rng.integers(4)
            elif u < noise + habit:
                acts[i] = instinct_action(env.grid[i], env.pos[i], env.has_key[i], rng)
            else:
                acts[i] = expert_action(env.grid[i], env.pos[i], env.has_key[i])
        X.append(obs[live])
        Y.append(acts[live])
        E.append(live)
        env.step(acts)
    X, Y = np.concatenate(X), np.concatenate(Y)
    keep = np.concatenate([env.success[e] for e in E]) if successes_only else np.ones(len(Y), bool)
    return X[keep], Y[keep], float(env.success.mean())


def train_sft(policy, demos, val, suite, steps=1500, batch=256, lr=1e-3,
              eval_every=250, seed=0, log=print):
    X, Y = (torch.as_tensor(a) for a in demos[:2])
    Xv, Yv = (torch.as_tensor(a) for a in val[:2])
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    gen = torch.Generator().manual_seed(seed)
    hist = []
    for step in range(1, steps + 1):
        policy.train()
        idx = torch.randint(len(X), (batch,), generator=gen)
        loss = F.cross_entropy(policy(X[idx]), Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 25 == 0 or step == 1:
            policy.eval()
            with torch.no_grad():
                lv = policy(Xv)
                row = dict(step=step, train_loss=loss.item(),
                           val_loss=F.cross_entropy(lv, Yv).item(),
                           val_acc=(lv.argmax(-1) == Yv).float().mean().item())
            if step % eval_every == 0 or step == 1:
                row["eval"] = suite(policy)
                log(f"sft {step:5d} loss {row['train_loss']:.3f} val_acc {row['val_acc']:.3f} "
                    f"test {row['eval']['test']['success']:.3f} hard {row['eval']['hard']['success']:.3f}")
            hist.append(row)
    return hist
