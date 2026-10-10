# Post Training Lab

A tiny transformer learns a key-and-door maze through post-training. SFT copies a planner's bad habit; GRPO (online RL) and DPO (offline preferences) then try to fix it, and a reward-hacking run shows what a subtly wrong reward does. Results cover 3 seeds, with a website that runs every policy live in your browser.

![Live race: four checkpoints on the same maze, with a training scrubber](docs/images/race.png)

![Training curves across seeds: GRPO vs RL from scratch](docs/images/curves.png)

![Reward hacking: the proxy climbs while true success collapses](docs/images/hacking.png)

![Trajectory fans: sampled attempts and greedy paths per checkpoint](docs/images/trajectories.png)

```mermaid
flowchart LR
  R[Random init] -->|SFT on habit-prone demos| S[SFT]
  S -->|GRPO + KL| G[GRPO]
  S -->|offline DPO + NLL| D[DPO]
  S -->|GRPO on a flawed proxy| H[Reward hacker]
  R -->|GRPO, same budget| X[RL from scratch]
  G & D & H & X --> E[Eval suite: test + hard OOD]
```

**Links**
- Website: open [`site/index.html`](site/index.html)
- [Instructions](docs/INSTRUCTIONS.md): setup, training, export, demo controls
- [System design](docs/SYSTEM-DESIGN.md): architecture, flows, decisions, testing, limits
- [File structure](docs/FILE-STRUCTURE.md): what's where
