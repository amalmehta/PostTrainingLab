"""Key & Door: a tiny gridworld with a verifiable reward.

A 7x7 grid (border walls, 5x5 interior) is split by a wall line with one
locked door. The agent starts on one side with a key; the goal is on the
other side. To win it must pick up the key, walk through the door, and
reach the goal within MAX_STEPS.

Everything here is plain NumPy and vectorised over a batch of episodes,
so rollouts for RL are cheap on CPU. The same rules are re-implemented in
site/app.js; tests/test_parity.py checks the two agree.
"""
from collections import deque

import numpy as np

SIZE = 7
MAX_STEPS = 30
STEP_COST = 0.01

# Cell codes stored in the grid.
EMPTY, WALL, KEY, DOOR, GOAL = 0, 1, 2, 3, 4
# Extra observation tokens (never stored in the grid).
AGENT, CLS_NOKEY, CLS_KEY = 5, 6, 7
VOCAB = 8
INNER = SIZE - 2
SEQ_LEN = 1 + INNER * INNER  # CLS token + one token per interior cell (border is always wall)

# up, down, left, right as (drow, dcol)
MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)
ACTION_NAMES = ["up", "down", "left", "right"]

# Level distributions. "train" is used for SFT demos, RL and the
# in-distribution test set; "hard" is a held-out, out-of-distribution split
# with denser clutter.
DISTS = {
    "train": dict(wall_density=(0.05, 0.20)),
    "hard": dict(wall_density=(0.28, 0.38)),
}


def make_level(rng, dist="train"):
    """Sample one solvable level. Returns (grid[SIZE, SIZE] int8, agent (r, c))."""
    lo, hi = DISTS[dist]["wall_density"]
    while True:
        g = np.zeros((SIZE, SIZE), dtype=np.int8)
        g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = WALL
        vertical = rng.random() < 0.5
        split = int(rng.integers(2, SIZE - 2))  # wall line strictly inside the interior
        door_at = int(rng.integers(1, SIZE - 1))
        if vertical:
            g[1:-1, split] = WALL
            g[door_at, split] = DOOR
        else:
            g[split, 1:-1] = WALL
            g[split, door_at] = DOOR
        density = rng.uniform(lo, hi)
        interior = [(r, c) for r in range(1, SIZE - 1) for c in range(1, SIZE - 1)
                    if g[r, c] == EMPTY]
        for r, c in interior:
            if rng.random() < density:
                g[r, c] = WALL

        def side(r, c):
            return (c < split) if vertical else (r < split)

        start_side = rng.random() < 0.5
        mine = [(r, c) for r in range(SIZE) for c in range(SIZE)
                if g[r, c] == EMPTY and side(r, c) == start_side]
        theirs = [(r, c) for r in range(SIZE) for c in range(SIZE)
                  if g[r, c] == EMPTY and side(r, c) != start_side]
        if len(mine) < 2 or len(theirs) < 1:
            continue
        ia, ik = rng.choice(len(mine), size=2, replace=False)
        agent, key = mine[ia], mine[ik]
        goal = theirs[int(rng.integers(len(theirs)))]
        g[key] = KEY
        g[goal] = GOAL
        if expert_path(g, agent) is not None:
            return g, agent


def _bfs(g, start, target, door_open):
    """Shortest list of actions from start to target, or None."""
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == target:
            acts = []
            while prev[cur] is not None:
                cur, a = prev[cur]
                acts.append(a)
            return acts[::-1]
        for a, (dr, dc) in enumerate(MOVES):
            nxt = (cur[0] + dr, cur[1] + dc)
            cell = g[nxt]
            if nxt in prev or cell == WALL or (cell == DOOR and not door_open):
                continue
            if cell == GOAL and nxt != target:
                continue  # touching the goal ends the episode early
            prev[nxt] = (cur, a)
            q.append(nxt)
    return None


def expert_path(g, agent):
    """Optimal action list: walk to the key, then through the door to the goal."""
    key = tuple(np.argwhere(g == KEY)[0])
    goal = tuple(np.argwhere(g == GOAL)[0])
    a = _bfs(g, tuple(agent), key, door_open=False)
    if a is None:
        return None
    b = _bfs(g, key, goal, door_open=True)
    if b is None or len(a) + len(b) > MAX_STEPS:
        return None
    return a + b


class BatchEnv:
    """Vectorised Key & Door over B independent levels."""

    def __init__(self, grids, agents):
        self.grid = np.array(grids, dtype=np.int8).copy()
        self.pos = np.array(agents, dtype=np.int64).copy()
        self.B = len(self.grid)
        self.has_key = np.zeros(self.B, dtype=bool)
        self.done = np.zeros(self.B, dtype=bool)
        self.success = np.zeros(self.B, dtype=bool)
        self.steps = np.zeros(self.B, dtype=np.int64)

    def obs(self):
        """Token ids [B, SEQ_LEN]: CLS (encodes has_key) then the interior cells."""
        cells = self.grid[:, 1:-1, 1:-1].reshape(self.B, -1).astype(np.int64)
        cells[np.arange(self.B), (self.pos[:, 0] - 1) * INNER + self.pos[:, 1] - 1] = AGENT
        cls = np.where(self.has_key, CLS_KEY, CLS_NOKEY)[:, None]
        return np.concatenate([cls, cells], axis=1)

    def step(self, actions):
        """Apply actions to unfinished episodes. Returns a copy of `done`."""
        live = ~self.done
        idx = np.nonzero(live)[0]
        if len(idx) == 0:
            return self.done.copy()
        nxt = self.pos[idx] + MOVES[np.asarray(actions)[idx]]
        cell = self.grid[idx, nxt[:, 0], nxt[:, 1]]
        blocked = (cell == WALL) | ((cell == DOOR) & ~self.has_key[idx])
        move = ~blocked
        self.pos[idx[move]] = nxt[move]
        # Picking up the key / opening the door turns the cell empty.
        got_key = move & (cell == KEY)
        opened = move & (cell == DOOR)
        clear = got_key | opened
        self.grid[idx[clear], nxt[clear, 0], nxt[clear, 1]] = EMPTY
        self.has_key[idx[got_key]] = True
        self.steps[idx] += 1
        won = move & (cell == GOAL)
        self.success[idx[won]] = True
        self.done[idx] = won | (self.steps[idx] >= MAX_STEPS)
        return self.done.copy()

    def rewards(self):
        """Verifiable episode reward: 1 - 0.01 * steps on success, else 0."""
        return np.where(self.success, 1.0 - STEP_COST * self.steps, 0.0)
