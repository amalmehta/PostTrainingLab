# Post Training Lab

A tiny transformer learns a key-and-door maze in two post-training stages: SFT on demos from a planner with a bad habit, then GRPO against a verifiable reward. It comes with a held-out eval suite and a website that runs the trained policies live in your browser.

![Live race: four checkpoints on the same maze](docs/images/race.png)

![GRPO vs RL from scratch: training curves](docs/images/curves.png)

![Trajectory fans: sampled attempts per checkpoint](docs/images/trajectories.png)

```mermaid
flowchart LR
  R[Random init] -->|SFT on habit-prone demos| S[SFT]
  S -->|GRPO + KL to SFT| G[SFT → GRPO]
  R -->|GRPO, same budget| X[RL from scratch]
  S & G & X --> E[Eval suite: test + hard OOD]
```

**Links**
- Website: open [`site/index.html`](site/index.html)
- [Instructions](docs/INSTRUCTIONS.md): setup, training, viewing, demo controls
- [System design](docs/SYSTEM-DESIGN.md): architecture, flows, decisions, testing, limits
- [File structure](docs/FILE-STRUCTURE.md): what's where
