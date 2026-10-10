# Instructions

## Setup

Requires Python 3.10+ and, for the parity test only, Node 18+.

```bash
uv venv --python 3.11 .venv
```

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

`requirements.txt` pins `torch==2.2.2` and `numpy<2` because 2.2.2 is the last
PyTorch release with Intel-Mac wheels. On Apple Silicon or Linux you can loosen
both pins.

## Run the tests

```bash
.venv/bin/python -m pytest
```

## Train

Training runs once per seed and writes `runs/seed<N>/`. Run the three seeds in
parallel, with 2 torch threads each (more threads barely help on most CPUs):

```bash
for s in 0 1 2; do .venv/bin/python train.py --seed $s --threads 2 > runs/train_seed$s.log 2>&1 & done
```

Each seed runs SFT, GRPO, DPO (with and without the NLL term), the reward
hacker and RL from scratch. That takes a few hours on a 2019 Intel MacBook Pro,
and much longer if the machine is busy. Follow along with
`tail -f runs/train_seed0.log`.

When the seeds finish, build the website data (a few minutes):

```bash
.venv/bin/python export.py
```

`export.py` uses every finished seed it finds, so you can export after one seed
and again later. For a quick smoke test of the whole path:

```bash
.venv/bin/python train.py --quick --seed 0
```

```bash
.venv/bin/python export.py --quick
```

`export.py --quick` publishes the smoke-test data to `site/data/`, so run the
full `export.py` again afterwards.

## View the website

Open `site/index.html` in a browser. It is fully static and needs no server.
To serve it anyway:

```bash
python3 -m http.server -d site 8000
```

## Use the demo

- **In-distribution / Hard (OOD)** switch between the two held-out level pools.
- **‹ ›** and **Random level** pick a level. The level label shows its index in the pool.
- **Greedy / Sample** sets argmax or temperature-1 decoding. Sampling gives a different run each time.
- Each panel's title is a menu: pick any checkpoint (SFT, DPO, plain DPO, GRPO, reward hacker, RL from scratch, random init).
- **Training scrubber** rewinds panels showing GRPO or the reward hacker to a snapshot from training (every 20 iterations; 0 = SFT).
- **Play**, **Step** and **Reset** drive all four panels in lockstep. **Speed** sets steps per second.
- **Edit walls**: click interior cells to add or remove walls. An edit that would make
  the level unsolvable is refused.
- Under each maze, the bars show the policy's probability for each action in the current state.
