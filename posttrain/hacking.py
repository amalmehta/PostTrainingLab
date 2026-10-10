"""A deliberately flawed proxy reward, to show reward hacking.

The proxy is a dense shaping reward: every step pays for being close to the
goal, 1 - (Manhattan distance to goal) / DMAX, and the episode's reward is the
sum of those payments scaled by the step limit (so it lies in [0, 1]). It
looks like a helpful "warmer / colder" signal, but it has two bugs:
  * Summing per-step payments rewards staying in the game. Touching the goal
    ends the episode and stops the payments, so hovering near the goal until
    the time limit beats actually finishing.
  * Manhattan distance ignores walls, so the near side of the wall line,
    across from the goal, pays almost as much as being next to it. The key
    and the door are not needed at all.
"""
import numpy as np

from .env import GOAL, MAX_STEPS, SIZE

DMAX = 2 * (SIZE - 3)  # largest Manhattan distance between interior cells


def closeness_reward(ro, batch):
    """Per-episode proxy reward from a recorded rollout over `batch` levels."""
    goals = np.array([np.argwhere(g == GOAL)[0] for g, _ in batch])  # [B, 2]
    paths = ro["paths"][1:]  # positions after each step, [T, B, 2]
    d = np.abs(paths - goals[None]).sum(-1)  # [T, B]
    close = 1.0 - d / DMAX
    taken = np.arange(len(paths))[:, None] < ro["steps"][None]  # steps actually taken
    return (close * taken).sum(0) / MAX_STEPS
