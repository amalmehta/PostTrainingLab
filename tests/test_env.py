import numpy as np

from posttrain.env import DOOR, KEY, MAX_STEPS, WALL, BatchEnv, expert_path, make_level
from posttrain.evals import pass_at_k
from posttrain.sft import level_hash, sample_train_levels


def test_levels_are_well_formed_and_expert_solves_them():
    rng = np.random.default_rng(0)
    for dist in ["train", "hard"]:
        levels = [make_level(rng, dist) for _ in range(50)]
        for g, a in levels:
            assert (g == KEY).sum() == 1 and (g == DOOR).sum() == 1
            assert (g[0] == WALL).all() and (g[:, -1] == WALL).all()
        env = BatchEnv([g for g, _ in levels], [a for _, a in levels])
        paths = [expert_path(g, a) for g, a in levels]
        for t in range(MAX_STEPS):
            env.step(np.array([p[t] if t < len(p) else 0 for p in paths]))
        assert env.success.all()
        assert (env.steps == [len(p) for p in paths]).all()
        assert np.allclose(env.rewards(), 1 - 0.01 * env.steps)


def test_door_blocks_without_key():
    rng = np.random.default_rng(3)
    g, a = make_level(rng)
    door = tuple(np.argwhere(g == DOOR)[0])
    # put the agent next to the door on an open side and walk into it
    for i, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
        nb = (door[0] - dr, door[1] - dc)
        if g[nb] == 0:
            env = BatchEnv([g], [nb])
            env.step(np.array([i]))
            assert tuple(env.pos[0]) == nb  # blocked
            env.has_key[0] = True
            env.step(np.array([i]))
            assert tuple(env.pos[0]) == door and env.grid[0][door] == 0
            return
    raise AssertionError("no open neighbour found")


def test_training_levels_exclude_eval_levels():
    rng = np.random.default_rng(0)
    evals = [make_level(np.random.default_rng(7)) for _ in range(5)]
    banned = {level_hash(l) for l in evals}
    train = sample_train_levels(200, np.random.default_rng(7), banned)
    assert not banned & {level_hash(l) for l in train}


def test_pass_at_k():
    assert pass_at_k(8, 0, 8) == 0.0
    assert pass_at_k(8, 1, 8) == 1.0
    assert abs(pass_at_k(8, 2, 1) - 0.25) < 1e-9
