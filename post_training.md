PROJECT NAME: post_training

META-INSTRUCTIONS:

<Read it all before acting. Ask about anything unclear, contradictory or
 underspecified — before starting and mid-build. Ask in the question widget
 (AskUserQuestion): related questions batched, concrete options, your
 recommendation first. Plain text only if the widget isn't available.>

<Don't expand scope. Anything not listed here is a proposal, including changes
 to this file — propose it, don't do it.>

<Prefer doing over describing: run the code, write the files, test it.>

<Always in scope, no proposal needed: when it goes on GitHub, a README that is
 easy to read at a glance — a line on what it is, then clear visuals
 (screenshots, a diagram or a chart), then links. Everything else goes in
 linked files: docs/INSTRUCTIONS.md (setup, run, use),
 docs/SYSTEM-DESIGN.md (see below) and docs/FILE-STRUCTURE.md (what's where). If what you're
 building is an application rather than a script, also a small unobtrusive feedback tab, and a settings
 button or tab (⌘, on the Mac) that gathers its preferences in one place.>

<If what you're building is an application, build it as a Mac app first; the
 website comes after, as its own step.>

<Name things the way a person would say them — "Goal Tracker", not
 goal_tracker — for the app, its windows, titles, files people open, repo
 descriptions and README headings. When you create the GitHub repo, name it
 with no "_" or "-": one word or joined words, e.g. GoalTracker.>

<Always in scope: a system design doc in the codebase, docs/SYSTEM-DESIGN.md,
 kept current as the build changes. Cover the architecture (with a Mermaid
 diagram), each component's job, the main flows, where data lives, the key
 design decisions and their trade-offs, how it's tested, and known limits.>

<Finish by listing every deliverable: path, what it is, how to check it works.>

<Git rules (no Claude attribution, never commit .claude/) are in
 ~/.claude/CLAUDE.md and apply on their own — nothing to repeat here.>

<Keep the changelog at the bottom current.>

CONTEXT:

come up with an interesting project that focuses specifically on post training / evals / sft. show the env in a cool demo plus some of the trajectory plots.

DELIVERABLES:

project website

OPEN QUESTIONS / ASSUMPTIONS:

Asked and answered (2026-10-07):
- Concept: SFT→RL gridworld agent ("Key & Door"): a tiny transformer trained with SFT on
  human-style demos, then GRPO with a verifiable reward; held-out eval suite.
- Scope: static website only. It is treated as script + website, so no Mac app, feedback tab or settings.
- Publishing: local only (no GitHub repo, no hosting).
- Shut down the iOS Simulators that were saturating the CPU (approved).

Decided without asking:
- Name: "Post Training Lab" (site title and headings). The folder keeps its given name.
- Everything runs on CPU (Intel Mac, no GPU), so torch 2.2.2 and numpy<2 are pinned.
- Env: 7×7 grid (5×5 interior; shrunk from 9×9 so RL credit assignment fits a CPU budget), one wall line with a locked door, key on the agent's side, goal across.
  Reward = 1 − 0.01 × steps on success, else 0; 30-step limit. A "hard" OOD split uses denser walls.
- Policy: hand-written 3-layer transformer (~60k params), so the browser runs the identical
  forward pass. A Node parity test enforces this.
- SFT demos come from a BFS planner with a habit: 80% of the time it walks straight at its target
  instead of planning (a "human data" stand-in), and only successful episodes are kept. SFT inherits
  the habit, which leaves RL something systematic to fix. Uniform random label noise was tried first:
  greedy decoding strips it for free, and RL then had nothing consistent to improve.
- Baseline: GRPO from the same random init with the same budget (RL without SFT).
- Eval levels are hashed and excluded from training data (contamination guard).
- The site loads data as .js files so it opens from file:// with no server.
- 2026-10-09 "keep improving": you chose all four proposed features (DPO stage, checkpoint
  scrubber, reward-hacking eval, multi-seed error bands). Decided without asking: DPO is offline
  on SFT samples with the same episode budget as GRPO, plus an NLL term (plain DPO collapsed in
  trials and is kept as a second run). The hack proxy is a summed per-step closeness reward
  (an averaged version barely got hacked). 3 seeds run in parallel; the live demo and gallery use seed 0;
  snapshots every 20 iterations are stored as float16. The v1 single-seed run is archived in runs_v1/.
- README, INSTRUCTIONS, FILE-STRUCTURE are written now, though the project is not on GitHub
  yet, so it is ready if you publish it later.

CHANGELOG:

- 2026-10-07 — created
- 2026-09-15 — added meta-instruction: built-out applications include a small feedback tab
- 2026-09-15 — added meta-instruction: no "Claude" attribution in commits, PRs, or branches
- 2026-09-16 — added meta-instruction: always include a README when adding to GitHub
- 2026-09-16 — changed meta-instruction: ask clarifying questions in the question widget
- 2026-09-17 — added meta-instructions: Claude never a contributor; never commit .claude/
- 2026-09-26 — compressed the meta-instructions and every field prompt; git rules moved to the global instruction file
- 2026-09-27 — added meta-instruction: applications are built as a Mac app first, then a website
- 2026-09-28 — folded inputs, instructions, constraints, deliverables and done criteria into one free-form CONTEXT
- 2026-09-28 — changed meta-instruction: a README on GitHub always includes a visual
- 2026-09-28 — added meta-instruction: name things like a person would, never snake_case
- 2026-09-28 — changed meta-instruction: README leads with visuals; instructions live in a linked guide
- 2026-09-28 — changed meta-instruction: README is visuals and links; details in docs/INSTRUCTIONS.md and docs/FILE-STRUCTURE.md
- 2026-09-29 — changed meta-instruction: GitHub repo names have no "_" or "-"
- 2026-10-02 — added a DELIVERABLES field after CONTEXT
- 2026-10-02 — added meta-instruction: every project has a system design doc at docs/SYSTEM-DESIGN.md
- 2026-10-05 — added meta-instruction: applications include a settings button or tab
- 2026-10-08 — built: Key & Door env, SFT + GRPO pipeline, eval suite, project website, docs
- 2026-10-08 — final run: SFT 71% → GRPO 85% test success (hard OOD 63% → 75%); README screenshots; train.py --from-runs
- 2026-10-09 — added DPO (+NLL and plain), reward-hacking run, training scrubber, 3-seed bands; split train.py / export.py
