// Post Training Lab page: hero tiles, live race + training scrubber, diagram,
// curves (mean + min-max band over seeds), DPO and reward-hacking sections,
// trajectory fans, eval table and notes. Data comes from export.py via
// data/results.js (window.RESULTS) and data/checkpoints.js (window.CHECKPOINTS,
// window.SNAPSHOTS).
(function () {
  const R = window.RESULTS, CK = window.CHECKPOINTS, SN = window.SNAPSHOTS || {}, C = window.Core;
  const STAGES = R.stages;  // random, sft, grpo, dpo, hack, scratch
  const COLOR = { random: "--c-random", sft: "--c-sft", grpo: "--c-grpo", dpo: "--c-dpo",
                  hack: "--c-hack", scratch: "--c-scratch", dpo_plain: "--c-dpo-plain" };
  const LABEL = R.labels;
  const has = s => STAGES.includes(s);
  const NS = R.seeds.length;
  const pct = v => `${Math.round(v * 100)}%`;
  const pct1 = v => `${(v * 100).toFixed(1)}%`;
  const $ = s => document.querySelector(s);
  const css = () => { const s = getComputedStyle(document.documentElement); return n => s.getPropertyValue(n).trim(); };
  const avg = a => a.reduce((x, y) => x + y, 0) / a.length;
  const sd = a => { const m = avg(a); return Math.sqrt(avg(a.map(v => (v - m) ** 2))); };

  // ---------- aggregation over seeds ----------
  const evalVals = (s, set, k) => R.evals.map(e => e[s][set][k]);
  const M = (s, set, k) => avg(evalVals(s, set, k));
  // Curve of f(row) for one stage: mean line + [x, min, max] band over seeds.
  function curve(stage, f, xKey, filter = () => true, smoothW = 1) {
    const perSeed = R.history.map(h => {
      const pts = h[stage].filter(filter).map(r => [r[xKey], f(r)]);
      return smoothW > 1 ? pts.map((p, i) => {
        const w = pts.slice(Math.max(0, i - smoothW + 1), i + 1); return [p[0], avg(w.map(q => q[1]))];
      }) : pts;
    });
    const xs = perSeed[0].map(p => p[0]).filter(x => perSeed.every(ps => ps.some(p => p[0] === x)));
    const at = (ps, x) => ps.find(p => p[0] === x)[1];
    const points = xs.map(x => [x, avg(perSeed.map(ps => at(ps, x)))]);
    const band = NS > 1 ? xs.map(x => { const v = perSeed.map(ps => at(ps, x)); return [x, Math.min(...v), Math.max(...v)]; }) : null;
    return { points, band };
  }
  const series = (stage, c, name) => ({ name: name || LABEL[stage], color: COLOR[stage], points: c.points, band: c.band });

  // ---------- hero tiles ----------
  const rng = (s, set, k) => { const v = evalVals(s, set, k); return NS > 1 ? `range ${pct(Math.min(...v))}–${pct(Math.max(...v))}` : ""; };
  $("#seedCount").textContent = String(NS);
  $("#tiles").innerHTML = [
    { label: "Test success, SFT → GRPO", value: `${pct(M("sft", "test", "success"))} → ${pct(M("grpo", "test", "success"))}`, hero: true,
      delta: `greedy, ${R.config.eval_levels} held-out mazes, mean of ${NS} seed${NS > 1 ? "s" : ""}${NS > 1 ? ` (GRPO ${rng("grpo", "test", "success")})` : ""}` },
    { label: "SFT → DPO (offline)", value: `${pct(M("sft", "test", "success"))} → ${pct(M("dpo", "test", "success"))}`,
      delta: "same sample budget, no new rollouts" },
    { label: "Hard (OOD), SFT → GRPO", value: `${pct(M("sft", "hard", "success"))} → ${pct(M("grpo", "hard", "success"))}`,
      delta: "denser walls, never trained on" },
    { label: "Reward hacker, true success", value: `${pct(M("hack", "test", "success"))}`,
      delta: `proxy reward ${M("sft", "test", "proxy").toFixed(2)} → ${M("hack", "test", "proxy").toFixed(2)} as it games the metric` },
  ].map(t => `<div class="tile${t.hero ? " wide" : ""}"><div class="label">${t.label}</div><div class="value${t.hero ? " hero" : ""}">${t.value}</div><div class="delta">${t.delta}</div></div>`).join("");

  // ---------- race demo ----------
  const models = Object.fromEntries(STAGES.map(s => [s, C.loadModel(CK[s].weights)]));
  const snapModels = Object.fromEntries(Object.entries(SN).map(([run, list]) =>
    [run, list.map(x => ({ iter: x.iter, model: C.loadModel(x.weights) }))]));
  const scrubIters = (SN.grpo || []).map(x => x.iter);
  const st = { pool: "test", idx: 0, decode: "greedy", playing: false, timer: null, editing: false, base: null,
               scrub: Math.max(0, scrubIters.length - 1) };
  const DEFAULT_PANELS = ["sft", "dpo", "grpo", "hack"];
  const panels = DEFAULT_PANELS.map(s => {
    const div = document.createElement("div");
    div.className = "panel";
    div.innerHTML = `<header><span class="dot"></span><select aria-label="Checkpoint">${STAGES.map(x =>
      `<option value="${x}"${x === s ? " selected" : ""}>${LABEL[x]}</option>`).join("")}</select></header>
      <canvas aria-label="maze"></canvas>
      <div class="status"><span class="steps"></span><span class="badge"></span></div>
      <div class="probs">${C.ACTIONS.map(a => `<span>${a}</span><span class="bar"><i></i></span><span class="num"></span>`).join("")}</div>`;
    $("#race").appendChild(div);
    const p = { stage: s, div, canvas: div.querySelector("canvas"), env: null, probs: null, trail: [] };
    div.querySelector("select").addEventListener("change", e => { p.stage = e.target.value; paint(p); reset(); });
    p.canvas.addEventListener("click", ev => onEdit(p, ev));
    paint(p);
    return p;
  });
  function paint(p) {
    p.div.querySelector(".dot").style.background = `var(${COLOR[p.stage]})`;
    p.div.querySelectorAll(".bar i").forEach(i => { i.style.background = `var(${COLOR[p.stage]})`; });
  }
  // The scrubber swaps GRPO / hacker panels to a snapshot; the last position is the final float32 checkpoint.
  function modelFor(stage) {
    const list = snapModels[stage];
    if (!list || st.scrub >= list.length - 1) return models[stage];
    return list[st.scrub].model;
  }
  function scrubText() {
    if (!scrubIters.length) return "no snapshots";
    const it = scrubIters[st.scrub];
    return st.scrub === scrubIters.length - 1 ? `iter ${it} (final)` : `iter ${it}${it === 0 ? " (= SFT)" : ""}`;
  }
  const scrub = $("#scrub");
  scrub.max = Math.max(0, scrubIters.length - 1);
  scrub.value = st.scrub;
  $("#scrubLabel").textContent = scrubText();
  scrub.addEventListener("input", () => { st.scrub = +scrub.value; $("#scrubLabel").textContent = scrubText(); reset(); });

  function loadLevel() {
    const lv = R.pools[st.pool][st.idx];
    st.base = { grid: Array.from(lv.grid, Number), agent: lv.agent };
    $("#levelLabel").textContent = `${st.pool} #${st.idx + 1}/${R.pools[st.pool].length}`;
    reset();
  }
  function reset() {
    stop();
    st.optimal = C.optimalSteps(st.base.grid, st.base.agent);
    for (const p of panels) {
      p.env = new C.Env(st.base.grid, st.base.agent);
      p.trail = [p.env.pos.slice()];
      think(p);
    }
    render();
  }
  function think(p) {
    p.probs = p.env.done ? null : C.softmax(C.forward(modelFor(p.stage), p.env.obs()));
  }
  function stepAll() {
    for (const p of panels) {
      if (p.env.done) continue;
      let a;
      if (st.decode === "greedy") a = p.probs.indexOf(Math.max(...p.probs));
      else { let u = Math.random(); a = 0; while (a < 3 && (u -= p.probs[a]) > 0) a++; }
      p.env.step(a);
      p.trail.push(p.env.pos.slice());
      think(p);
    }
    render();
    if (panels.every(p => p.env.done)) stop();
  }
  function play() {
    if (panels.every(p => p.env.done)) reset();
    st.playing = true; $("#play").textContent = "❚❚ Pause";
    const loop = () => { if (!st.playing) return; stepAll(); st.timer = setTimeout(loop, 1000 / +$("#speed").value); };
    loop();
  }
  function stop() { st.playing = false; clearTimeout(st.timer); $("#play").textContent = "▶ Play"; }

  function render() {
    const v = css();
    for (const p of panels) {
      drawMaze(p.canvas, p.env.grid, p.env.pos, p.env.hasKey, p.trail, v, COLOR[p.stage]);
      const e = p.env, b = p.div.querySelector(".badge");
      p.div.querySelector(".steps").textContent = `step ${e.steps}${st.optimal ? ` · optimal ${st.optimal}` : ""}${e.hasKey && !e.done ? " · 🔑" : ""}`;
      b.className = "badge" + (e.success ? " ok" : e.done ? " bad" : "");
      b.textContent = e.success ? `solved · R ${e.reward().toFixed(2)}` : e.done ? "timed out · R 0" : "running";
      const bars = p.div.querySelectorAll(".bar i"), nums = p.div.querySelectorAll(".num");
      C.ACTIONS.forEach((_, i) => {
        const q = p.probs ? p.probs[i] : 0;
        bars[i].style.width = `${q * 100}%`;
        nums[i].textContent = p.probs ? q.toFixed(2) : "–";
      });
      p.div.classList.toggle("editing", st.editing);
    }
  }

  function drawMaze(canvas, grid, pos, hasKey, trail, v, trailColor) {
    const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth || 240;
    canvas.width = canvas.height = Math.round(w * dpr);
    const g = canvas.getContext("2d"), n = C.SIZE, s = (w * dpr) / n;
    g.clearRect(0, 0, canvas.width, canvas.height);
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      const cell = grid[r * n + c], x = c * s, y = r * s;
      g.fillStyle = (r + c) % 2 ? v("--floor-2") : v("--floor");
      g.fillRect(x, y, s, s);
      if (cell === C.WALL) { g.fillStyle = v("--wall"); roundRect(g, x + 1, y + 1, s - 2, s - 2, s * 0.12); g.fill(); }
      if (cell === C.DOOR) {
        g.fillStyle = v("--door"); roundRect(g, x + s * 0.12, y + s * 0.08, s * 0.76, s * 0.84, s * 0.1); g.fill();
        g.fillStyle = v("--floor"); g.beginPath(); g.arc(x + s / 2, y + s * 0.45, s * 0.09, 0, 7); g.fill();
        g.fillRect(x + s * 0.46, y + s * 0.45, s * 0.08, s * 0.2);
      }
      if (cell === C.KEY) drawKey(g, x + s / 2, y + s / 2, s * 0.32, v("--key"));
      if (cell === C.GOAL) drawStar(g, x + s / 2, y + s / 2, s * 0.36, v("--goal"));
    }
    if (trail.length > 1) {
      g.strokeStyle = v(trailColor); g.globalAlpha = 0.45; g.lineWidth = s * 0.12; g.lineJoin = g.lineCap = "round";
      g.beginPath();
      trail.forEach(([r, c], i) => { const X = (c + 0.5) * s, Y = (r + 0.5) * s; i ? g.lineTo(X, Y) : g.moveTo(X, Y); });
      g.stroke(); g.globalAlpha = 1;
    }
    const ax = (pos[1] + 0.5) * s, ay = (pos[0] + 0.5) * s;
    g.fillStyle = v("--surface"); g.beginPath(); g.arc(ax, ay, s * 0.34, 0, 7); g.fill();
    g.fillStyle = v("--agent"); g.beginPath(); g.arc(ax, ay, s * 0.28, 0, 7); g.fill();
    if (hasKey) drawKey(g, ax + s * 0.26, ay - s * 0.26, s * 0.16, v("--key"));
  }
  function roundRect(g, x, y, w, h, r) { g.beginPath(); g.roundRect ? g.roundRect(x, y, w, h, r) : g.rect(x, y, w, h); }
  function drawKey(g, x, y, r, col) {
    g.strokeStyle = col; g.fillStyle = col; g.lineWidth = r * 0.35; g.lineCap = "round";
    g.beginPath(); g.arc(x - r * 0.45, y, r * 0.42, 0, 7); g.stroke();
    g.beginPath(); g.moveTo(x - r * 0.05, y); g.lineTo(x + r, y); g.moveTo(x + r * 0.7, y); g.lineTo(x + r * 0.7, y + r * 0.4); g.stroke();
  }
  function drawStar(g, x, y, r, col) {
    g.fillStyle = col; g.beginPath();
    for (let i = 0; i < 10; i++) {
      const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.45 : r;
      g.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a));
    }
    g.closePath(); g.fill();
  }

  function onEdit(p, ev) {
    if (!st.editing) return;
    const rect = p.canvas.getBoundingClientRect(), n = C.SIZE;
    const c = Math.floor((ev.clientX - rect.left) / rect.width * n), r = Math.floor((ev.clientY - rect.top) / rect.height * n);
    if (r < 1 || c < 1 || r > n - 2 || c > n - 2) return;
    const i = r * n + c, cell = st.base.grid[i];
    if ((cell !== C.EMPTY && cell !== C.WALL) || (r === st.base.agent[0] && c === st.base.agent[1])) return;
    const next = st.base.grid.slice(); next[i] = cell === C.WALL ? C.EMPTY : C.WALL;
    if (C.optimalSteps(next, st.base.agent) === null) {
      $("#editHint").textContent = "That wall would make the maze unsolvable, so it was not placed.";
      return;
    }
    $("#editHint").textContent = "Click any interior cell in any panel to toggle a wall. Unsolvable mazes are refused.";
    st.base.grid = next; reset();
  }

  function seg(id, key, after) {
    const box = $(id);
    box.addEventListener("click", e => {
      const b = e.target.closest("button"); if (!b) return;
      box.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
      st[key] = b.dataset.v; after();
    });
  }
  seg("#pool", "pool", () => { st.idx = 0; loadLevel(); });
  seg("#decode", "decode", reset);
  $("#prev").onclick = () => { st.idx = (st.idx - 1 + R.pools[st.pool].length) % R.pools[st.pool].length; loadLevel(); };
  $("#next").onclick = () => { st.idx = (st.idx + 1) % R.pools[st.pool].length; loadLevel(); };
  $("#shuffle").onclick = () => { st.idx = Math.floor(Math.random() * R.pools[st.pool].length); loadLevel(); };
  $("#play").onclick = () => (st.playing ? stop() : play());
  $("#step").onclick = () => { stop(); stepAll(); };
  $("#reset").onclick = reset;
  $("#edit").onchange = e => { st.editing = e.target.checked; $("#editHint").hidden = !st.editing; render(); };
  window.addEventListener("resize", render);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
  loadLevel();

  // ---------- pipeline diagram ----------
  const nParams = Object.values(CK.random.weights.tensors).reduce((a, t) => a + t.shape.reduce((x, y) => x * y, 1), 0);
  $("#paramCount").textContent = `${Math.round(nParams / 1000)}k`;
  const cfg = R.config;
  (function diagram() {
    const box = (x, y, w, t, s, col) => `<rect class="bx" x="${x}" y="${y}" width="${w}" height="54" rx="10"/>
        <rect x="${x}" y="${y + 12}" width="4" height="30" rx="2" fill="var(${col})"/>
        <text class="t" x="${x + 16}" y="${y + 23}">${t}</text><text class="s" x="${x + 16}" y="${y + 41}">${s}</text>`;
    $("#diagram").innerHTML = `<svg viewBox="0 0 900 300" role="img" aria-label="Pipeline diagram">
      <defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="var(--muted)"/></marker></defs>
      <style>.bx{fill:var(--surface-2);stroke:var(--line)} .t{fill:var(--text);font-size:14px;font-weight:600}
        .s{fill:var(--text-2);font-size:12px} .ar{stroke:var(--muted);stroke-width:1.5;fill:none;marker-end:url(#ah)}</style>
      ${box(10, 123, 150, "Random init", `${Math.round(nParams / 1000)}k params, 3-layer`, "--c-random")}
      ${box(210, 70, 160, "SFT", `${Math.round(avg(R.demo_transitions)).toLocaleString()} demo steps`, "--c-sft")}
      ${box(470, 10, 190, "SFT → GRPO", `${cfg.grpo_iters} iters · online · KL β=${cfg.beta}`, "--c-grpo")}
      ${box(470, 82, 190, "SFT → DPO", `offline pairs · β=${cfg.dpo_beta}`, "--c-dpo")}
      ${box(470, 154, 190, "Reward hacker", "GRPO on a flawed proxy", "--c-hack")}
      ${box(470, 232, 190, "RL from scratch", `${cfg.grpo_iters} iters · no SFT, no KL`, "--c-scratch")}
      ${box(755, 123, 135, "Eval suite", "test + hard sets", "--c-random")}
      <path class="ar" d="M160,150 C185,150 185,97 208,97"/>
      <path class="ar" d="M370,97 C420,97 420,37 468,37"/>
      <path class="ar" d="M370,97 L468,109"/>
      <path class="ar" d="M370,97 C420,97 420,181 468,181"/>
      <path class="ar" d="M160,150 C300,150 330,259 468,259"/>
      <path class="ar" d="M660,37 C720,37 720,140 753,143"/>
      <path class="ar" d="M660,109 C710,109 720,146 753,148"/>
      <path class="ar" d="M660,181 C710,181 720,156 753,153"/>
      <path class="ar" d="M660,259 C730,259 730,165 753,160"/>
    </svg>`;
    const ds = R.dpo_stats[0];
    $("#configCards").innerHTML = [
      ["Environment", `${C.SIZE}×${C.SIZE} grid (${C.SIZE - 2}×${C.SIZE - 2} interior), one wall line with a locked door. Observation = ${(C.SIZE - 2) ** 2} interior-cell tokens + a CLS token that encodes “holding key”. 4 actions, max ${C.MAX_STEPS} steps.`],
      ["Policy", `Pre-LN transformer, d=48, 4 heads, 3 layers, ReLU MLP. Action logits read off the CLS token. Hand-written so the browser runs the identical forward pass.`],
      ["SFT data", `${cfg.demos.toLocaleString()} levels attempted by a BFS planner with a habit: ${pct(cfg.habit)} of the time it walks straight at the target instead of planning. Only the ${pct(avg(R.demo_success))} of episodes that reached the goal are kept.`],
      ["GRPO", `${cfg.levels_per_iter} fresh levels × ${cfg.group} attempts per iteration. Advantage = group-normalised reward. PPO clip 0.2, 4 full-batch epochs per iteration, exact KL to SFT (β=${cfg.beta}).`],
      ["DPO", `Loss = DPO + ${cfg.dpo_nll ?? 1} × NLL of the chosen trajectory (RPO-style); plain DPO drops the NLL term. The same ${(cfg.grpo_iters * cfg.levels_per_iter * cfg.group).toLocaleString()} episodes GRPO sees, but all sampled from SFT up front. Chosen = shortest solved attempt, rejected = a failed one (else a much longer one): ${ds.pairs.toLocaleString()} pairs in seed ${R.seeds[0]}. ${cfg.dpo_epochs} epochs, β=${cfg.dpo_beta}.`],
      ["Reward hacker", `Identical GRPO recipe, but the reward is a per-step "closeness to the goal" bonus summed over the episode (Manhattan distance, walls ignored). It seems sensible; see below for why it isn't.`],
    ].map(([t, d]) => `<div class="card"><b>${t}</b>${d}</div>`).join("");
  })();

  // ---------- curves ----------
  const isEval = r => !!r.eval;
  const lastOf = (stage, f) => avg(R.history.map(h => f(h[stage][h[stage].length - 1])));
  Charts.lineChart($("#sftCharts"), {
    title: "Loss", sub: "cross-entropy on demo actions", xLabel: "step", endLabels: true,
    series: [series("sft", curve("sft", r => r.train_loss, "step"), "train"),
             { ...series("sft", curve("sft", r => r.val_loss, "step"), "val"), color: "--c-random" }],
  });
  Charts.lineChart($("#sftCharts"), {
    title: "Accuracy and success", sub: "val accuracy vs clean planner · greedy test success", xLabel: "step",
    yMin: 0, yMax: 1, yFmt: pct, endLabels: true,
    series: [{ ...series("sft", curve("sft", r => r.val_acc, "step"), "val acc"), color: "--c-random" },
             { ...series("sft", curve("sft", r => r.eval.test.success, "step", isEval), "test success"), marker: true }],
  });

  function chartGrid(target, legendEl, stages, charts) {
    Charts.legend($(legendEl), stages.map(s => ({ name: LABEL[s], color: COLOR[s] })));
    for (const c of charts) Charts.lineChart($(target), {
      title: c.title, sub: c.sub, xLabel: c.x || "iter", yMin: c.yMin, yMax: c.yMax,
      yFmt: c.yFmt || (v => v.toFixed(2)), refs: c.refs, series: c.series });
  }
  const sftRef = set => ({ y: M("sft", set, "success"), label: `SFT ${pct(M("sft", set, "success"))}` });
  const evalCurve = (s, set) => series(s, curve(s, r => r.eval[set].success, "iter", isEval));
  chartGrid("#rlCharts", "#rlLegend", ["grpo", "scratch"], [
    { title: "Test success (greedy)", sub: "held-out, in-distribution", yMin: 0, yMax: 1, yFmt: pct, refs: [sftRef("test")],
      series: [evalCurve("grpo", "test"), evalCurve("scratch", "test")] },
    { title: "Hard success (greedy)", sub: "out-of-distribution, denser walls", yMin: 0, yMax: 1, yFmt: pct, refs: [sftRef("hard")],
      series: [evalCurve("grpo", "hard"), evalCurve("scratch", "hard")] },
    { title: "Training reward", sub: "mean over sampled attempts, 5-iter moving avg", yMin: 0,
      series: ["grpo", "scratch"].map(s => series(s, curve(s, r => r.reward, "iter", undefined, 5))) },
    { title: "KL to SFT reference", sub: "nats per step (GRPO only; scratch has no reference)", yMin: 0,
      series: [series("grpo", curve("grpo", r => r.kl, "iter", undefined, 3))] },
    { title: "Policy entropy", sub: "nats per step; ln 4 = 1.39 is uniform", yMin: 0, yFmt: v => v.toFixed(1),
      series: ["grpo", "scratch"].map(s => series(s, curve(s, r => r.entropy, "iter", undefined, 3))) },
    { title: "Dead groups", sub: "share of groups where every attempt got the same reward (zero gradient)", yMin: 0, yMax: 1, yFmt: pct,
      series: ["grpo", "scratch"].map(s => series(s, curve(s, r => r.dead_groups, "iter", undefined, 5))) },
  ]);

  // DPO
  const ds0 = R.dpo_stats;
  $("#dpoSub").innerHTML = `Same starting point, same number of sampled episodes as GRPO
    (${(cfg.grpo_iters * cfg.levels_per_iter * cfg.group).toLocaleString()}), but every sample comes from the frozen SFT model
    up front, and none are drawn during training. Each level's 8 attempts give at most one pair: ${Math.round(avg(ds0.map(d => d.pairs))).toLocaleString()}
    pairs on average, ${pct(avg(ds0.map(d => d.fail_pairs / d.pairs)))} of them solved-vs-failed.`;
  const dpoX = "step";
  const DP = ["dpo", "dpo_plain"].filter(has);
  chartGrid("#dpoCharts", "#dpoLegend", DP, [
    { title: "Test success (greedy)", sub: "held-out, in-distribution", x: dpoX, yMin: 0, yMax: 1, yFmt: pct,
      refs: [sftRef("test"), { y: M("grpo", "test", "success"), label: `GRPO ${pct(M("grpo", "test", "success"))}` }],
      series: DP.map(s => series(s, curve(s, r => r.eval.test.success, dpoX, isEval))) },
    { title: "Preference accuracy", sub: "share of pairs where the implicit reward ranks chosen above rejected", x: dpoX, yMin: 0, yMax: 1, yFmt: pct,
      series: DP.map(s => series(s, curve(s, r => r.acc, dpoX, undefined, 10))) },
    { title: "Chosen log-ratio vs reference", sub: "log πθ − log πref of the preferred trajectory; below 0 = less likely than under SFT", x: dpoX,
      series: DP.map(s => series(s, curve(s, r => r.chosen, dpoX, undefined, 10))) },
  ]);

  // Reward hacking
  $("#hackCallout").innerHTML = `<b>The proxy:</b> a dense "warmer / colder" shaping reward. Every step pays
    1 − (Manhattan distance to goal)/${2 * (C.SIZE - 3)}, and the episode reward is the sum, scaled by the ${C.MAX_STEPS}-step limit.
    It looks helpful but has two flaws. Summing per-step payments rewards <em>staying in the game</em>: touching the goal ends the
    episode and stops the payments. And the distance ignores walls, so the near side of the wall line pays almost as much as the goal,
    with no key or door needed. GRPO on this reward takes the proxy from ${M("sft", "test", "proxy").toFixed(2)} to
    ${M("hack", "test", "proxy").toFixed(2)}, while true test success goes from ${pct(M("sft", "test", "success"))} to
    ${pct(M("hack", "test", "success"))}. Watch it in the race above; the scrubber shows the behaviour appearing during training.`;
  chartGrid("#hackCharts", "#hackLegend", ["hack", "grpo"], [
    { title: "What it was trained on", sub: "proxy reward of sampled attempts (hacker) · true reward (GRPO)", yMin: 0, yMax: 1,
      series: [series("hack", curve("hack", r => r.train_reward, "iter", undefined, 3), `${LABEL.hack} (proxy)`),
               series("grpo", curve("grpo", r => r.reward, "iter", undefined, 3), `${LABEL.grpo} (true)`)] },
    { title: "True test success (greedy)", sub: "what we actually wanted", yMin: 0, yMax: 1, yFmt: pct, refs: [sftRef("test")],
      series: [evalCurve("hack", "test"), evalCurve("grpo", "test")] },
    { title: "Episode length", sub: "steps per sampled attempt", yMin: 0, yMax: C.MAX_STEPS, yFmt: v => v.toFixed(0),
      series: ["hack", "grpo"].map(s => series(s, curve(s, r => r.length, "iter", undefined, 3))) },
  ]);

  // ---------- trajectory fans ----------
  const GAL = Object.keys(R.gallery[0].runs);
  const gal = $("#gallery");
  gal.insertAdjacentHTML("beforeend", `<div class="gal-row headrow"><div></div>${GAL.map(s =>
    `<div class="gal-head"><span class="dot" style="background:var(${COLOR[s]})"></span>${LABEL[s]}</div>`).join("")}</div>`);
  R.gallery.forEach((g, gi) => {
    const row = document.createElement("div");
    row.className = "gal-row";
    row.innerHTML = `<div class="meta"><b>Level ${gi + 1}</b>optimal ${g.level.optimal} steps</div>`;
    for (const s of GAL) {
      const runs = g.runs[s], wins = runs.filter(r => r.success), gr = g.greedy[s];
      const mean = wins.length ? (wins.reduce((a, r) => a + r.steps, 0) / wins.length).toFixed(1) : "–";
      const f = document.createElement("div");
      f.className = "fan";
      f.title = LABEL[s];
      f.innerHTML = fanSvg(g.level, runs, gi, gr, COLOR[s]) +
        `<div class="cap">greedy: ${gr.success ? `solved in ${gr.steps}` : "times out"}<br>samples: ${wins.length}/${runs.length} solved · ${mean} steps</div>`;
      row.appendChild(f);
    }
    gal.appendChild(row);
  });
  function fanSvg(level, runs, seed, greedy, color) {
    const n = C.SIZE, s = 20, grid = Array.from(level.grid, Number);
    let rnd = seed * 9301 + 49297;
    const jit = () => { rnd = (rnd * 9301 + 49297) % 233280; return (rnd / 233280 - 0.5) * 7; };
    let out = `<svg viewBox="0 0 ${n * s} ${n * s}" role="img" aria-label="sampled trajectories">`;
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      const v = grid[r * n + c], x = c * s, y = r * s;
      const fill = v === C.WALL ? "var(--wall)" : (r + c) % 2 ? "var(--floor-2)" : "var(--floor)";
      out += `<rect x="${x}" y="${y}" width="${s}" height="${s}" fill="${fill}"/>`;
      if (v === C.KEY) out += `<circle cx="${x + 10}" cy="${y + 10}" r="5" fill="var(--key)"/>`;
      if (v === C.DOOR) out += `<rect x="${x + 3}" y="${y + 2}" width="14" height="16" rx="2" fill="var(--door)"/>`;
      if (v === C.GOAL) out += `<rect x="${x + 4}" y="${y + 4}" width="12" height="12" rx="3" fill="var(--goal)"/>`;
    }
    for (const run of runs) {
      const jx = jit(), jy = jit();
      const pts = run.path.map(([r, c]) => `${(c + 0.5) * s + jx},${(r + 0.5) * s + jy}`).join(" ");
      out += `<polyline points="${pts}" fill="none" stroke="var(${run.success ? "--ok" : "--bad"})" stroke-width="1.6" stroke-opacity="0.45" stroke-linejoin="round" stroke-linecap="round"/>`;
    }
    const gp = greedy.path.map(([r, c]) => `${(c + 0.5) * s},${(r + 0.5) * s}`).join(" ");
    out += `<polyline points="${gp}" fill="none" stroke="var(--surface)" stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/>`;
    out += `<polyline points="${gp}" fill="none" stroke="var(${color})" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>`;
    const [ar, ac] = level.agent;
    out += `<circle cx="${(ac + 0.5) * s}" cy="${(ar + 0.5) * s}" r="6" fill="var(--agent)" stroke="var(--surface)" stroke-width="2"/></svg>`;
    return out;
  }

  // success by difficulty
  const BK = Object.keys(R.buckets);
  Charts.legend($("#bucketLegend"), BK.map(s => ({ name: LABEL[s], color: COLOR[s] })));
  const MIN_N = 5;  // buckets with fewer mazes are too noisy to plot
  const keep = R.buckets[BK[0]].map(b => b.n >= MIN_N);
  const bk = R.buckets[BK[0]].filter((_, i) => keep[i]);
  const bucketName = i => { const b = bk[Math.round(i)]; return b ? (b.hi >= 99 ? `${b.lo}+` : `${b.lo}–${b.hi - 1}`) : ""; };
  Charts.lineChart($("#bucketChart"), {
    title: "Greedy success by optimal path length", sub: `mazes per bucket: ${bk.map(b => b.n).join(" / ")} (buckets under ${MIN_N} mazes hidden)`,
    xLabel: "optimal steps", xFmt: bucketName, xTicks: bk.map((_, i) => i), yMin: 0, yMax: 1, yFmt: pct, height: 200,
    series: BK.map(s => ({ name: LABEL[s], color: COLOR[s], marker: true,
      points: R.buckets[s].filter((_, i) => keep[i]).map((b, i) => [i, b.success]) })),
  });

  // ---------- eval table ----------
  const cols = [
    ["test", "success", "Test success", pct1], ["test", "pass_at_k", "Test pass@8", pct1],
    ["test", "sample_success", "Test sampled", pct1], ["hard", "success", "Hard success", pct1],
    ["hard", "pass_at_k", "Hard pass@8", pct1], ["test", "proxy", "Proxy reward", v => v.toFixed(2)],
  ];
  const ORDER = ["random", "sft", "dpo", "dpo_plain", "grpo", "hack", "scratch"].filter(has);
  // bold the best true-metric score among honest runs; for the proxy column the hacker's win is the point
  const best = cols.map(([set, k]) => Math.max(...ORDER.filter(s => k === "proxy" || s !== "hack").map(s => M(s, set, k))));
  const cell = (s, set, k, f, i) => {
    const v = evalVals(s, set, k), m = avg(v);
    const pm = NS > 1 ? ` <span class="pm">± ${k === "proxy" ? sd(v).toFixed(2) : (sd(v) * 100).toFixed(1)}</span>` : "";
    return `<td class="${Math.abs(m - best[i]) < 1e-9 && m > 0 ? "best" : ""}">${f(m)}${pm}</td>`;
  };
  $("#evalTable").innerHTML = `<thead><tr><th>Checkpoint</th>${cols.map(c => `<th>${c[2]}</th>`).join("")}</tr></thead><tbody>` +
    ORDER.map(s => `<tr><td><span class="stage"><span class="dot" style="background:var(${COLOR[s]})"></span>${LABEL[s]}</span></td>${
      cols.map(([set, k, , f], i) => cell(s, set, k, f, i)).join("")}</tr>`).join("") + "</tbody>";

  // ---------- notes (numbers filled from the runs) ----------
  const deadScratch = avg(R.history.map(h => avg(h.scratch.slice(0, 10).map(r => r.dead_groups))));
  const deadGrpo = avg(R.history.map(h => avg(h.grpo.slice(-10).map(r => r.dead_groups))));
  const sftAcc = lastOf("sft", r => r.val_acc);
  const dpoGain = M("dpo", "test", "success") - M("sft", "test", "success");
  const hackLen = lastOf("hack", r => r.length);
  function plainDpoNote() {
    const ch = lastOf("dpo_plain", r => r.chosen), rj = lastOf("dpo_plain", r => r.rejected);
    const both = ch < 0 && rj < 0;
    return `<b>Why DPO needs the NLL term here.</b> Without it, plain DPO ends at ${pct(M("dpo_plain", "test", "success"))} test success.
      It widens the chosen–rejected gap (final log-ratios: chosen ${ch.toFixed(1)}, rejected ${rj.toFixed(1)})${both
        ? ", but by making <em>both</em> trajectories less likely than under SFT, which drains probability from the good actions too" : ""}.`;
  }
  $("#notesList").innerHTML = [
    `<b>SFT copies the habit along with the skill.</b> The demonstrator walks straight at its target ${pct(cfg.habit)} of
      the time. SFT's actions match the clean planner only ${pct(sftAcc)} of the time, and greedy decoding, which always takes the
      habit, walks into walls and loops. It solves ${pct(M("sft", "test", "success"))} of test mazes.`,
    `<b>GRPO breaks the habit.</b> Greedy test success goes ${pct(M("sft", "test", "success"))} → ${pct(M("grpo", "test", "success"))}
      and sampled success ${pct(M("sft", "test", "sample_success"))} → ${pct(M("grpo", "test", "sample_success"))} (mean of ${NS} seed${NS > 1 ? "s" : ""}).`,
    `<b>Online vs offline.</b> DPO sees the same number of episodes but never samples from its own updated policy.
      Test success goes ${pct(M("sft", "test", "success"))} → ${pct(M("dpo", "test", "success"))}
      (${dpoGain >= 0 ? "+" : ""}${(dpoGain * 100).toFixed(1)} points), compared with GRPO's ${pct(M("grpo", "test", "success"))}.`,
    ...(has("dpo_plain") ? [plainDpoNote()] : []),
    `<b>RL sharpens; it doesn't add much new.</b> SFT's pass@8 is already ${pct(M("sft", "test", "pass_at_k"))}: the planner's behaviour is in its
      samples, just not in its argmax. After GRPO, pass@8 is ${pct(M("grpo", "test", "pass_at_k"))} while greedy rises to ${pct(M("grpo", "test", "success"))}.`,
    `<b>Optimising the wrong number.</b> On the closeness proxy, GRPO moves the proxy ${M("sft", "test", "proxy").toFixed(2)} → ${M("hack", "test", "proxy").toFixed(2)}
      and true success ${pct(M("sft", "test", "success"))} → ${pct(M("hack", "test", "success"))}; sampled episodes end up ${hackLen.toFixed(0)} of ${C.MAX_STEPS} steps long.
      Only the eval on the true metric reveals it.`,
    `<b>Warm starts matter under sparse reward.</b> From random init, ${pct(deadScratch)} of GRPO groups in the first 10 iterations
      had every attempt fail. Identical rewards give zero advantage, hence no gradient. RL from scratch reaches ${pct(M("scratch", "test", "success"))} in the same budget.`,
    `<b>Dead groups come back at the end, for the opposite reason.</b> Late in GRPO, ${pct(deadGrpo)} of groups have every attempt succeed,
      so the learning signal shrinks as the policy improves.`,
    `<b>Generalisation.</b> On the hard split (denser walls, never seen in training): SFT ${pct(M("sft", "hard", "success"))},
      DPO ${pct(M("dpo", "hard", "success"))}, GRPO ${pct(M("grpo", "hard", "success"))}.`,
  ].map(t => `<li>${t}</li>`).join("");
  $("#runtime").textContent = `${NS} seed${NS > 1 ? "s" : ""}, ~${Math.round(avg(R.runtime_sec) / 60)} min each on a busy 2019 Intel MacBook Pro`;
})();
