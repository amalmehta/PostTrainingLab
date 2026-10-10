// Key & Door env + tiny transformer forward pass, ported from posttrain/.
// Runs in the browser (window.Core) and in Node (module.exports) so
// tests/test_parity.py can check it matches PyTorch exactly.
(function (root) {
  const SIZE = 7, INNER = SIZE - 2, MAX_STEPS = 30, STEP_COST = 0.01;
  const EMPTY = 0, WALL = 1, KEY = 2, DOOR = 3, GOAL = 4;
  const AGENT = 5, CLS_NOKEY = 6, CLS_KEY = 7;
  const MOVES = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  const ACTIONS = ["up", "down", "left", "right"];

  class Env {
    // grid: array of SIZE * SIZE cell codes (row-major), agent: [r, c]
    constructor(grid, agent) {
      this.grid = Int8Array.from(grid);
      this.pos = [agent[0], agent[1]];
      this.hasKey = false; this.done = false; this.success = false; this.steps = 0;
    }
    obs() {
      const t = new Int32Array(1 + INNER * INNER);
      t[0] = this.hasKey ? CLS_KEY : CLS_NOKEY;
      for (let r = 1; r <= INNER; r++)
        for (let c = 1; c <= INNER; c++)
          t[1 + (r - 1) * INNER + (c - 1)] = this.grid[r * SIZE + c];
      t[1 + (this.pos[0] - 1) * INNER + (this.pos[1] - 1)] = AGENT;
      return t;
    }
    step(a) {
      if (this.done) return;
      const r = this.pos[0] + MOVES[a][0], c = this.pos[1] + MOVES[a][1];
      const cell = this.grid[r * SIZE + c];
      const blocked = cell === WALL || (cell === DOOR && !this.hasKey);
      if (!blocked) {
        this.pos = [r, c];
        if (cell === KEY) { this.hasKey = true; this.grid[r * SIZE + c] = EMPTY; }
        if (cell === DOOR) this.grid[r * SIZE + c] = EMPTY;
        if (cell === GOAL) this.success = true;
      }
      this.steps++;
      this.done = this.success || this.steps >= MAX_STEPS;
    }
    reward() { return this.success ? 1 - STEP_COST * this.steps : 0; }
  }

  // Shortest path length start -> target (same rules as posttrain/env.py _bfs).
  function bfs(grid, start, target, doorOpen) {
    const dist = new Int32Array(SIZE * SIZE).fill(-1);
    const s = start[0] * SIZE + start[1], goalIdx = target[0] * SIZE + target[1];
    dist[s] = 0; const q = [s];
    for (let h = 0; h < q.length; h++) {
      const cur = q[h];
      if (cur === goalIdx) return dist[cur];
      for (const [dr, dc] of MOVES) {
        const n = cur + dr * SIZE + dc, cell = grid[n];
        if (dist[n] >= 0 || cell === WALL || (cell === DOOR && !doorOpen)) continue;
        if (cell === GOAL && n !== goalIdx) continue;
        dist[n] = dist[cur] + 1; q.push(n);
      }
    }
    return null;
  }
  function find(grid, code) {
    const i = Array.prototype.indexOf.call(grid, code);
    return i < 0 ? null : [Math.floor(i / SIZE), i % SIZE];
  }
  // Optimal number of steps for a fresh level, or null if unsolvable.
  function optimalSteps(grid, agent) {
    const key = find(grid, KEY), goal = find(grid, GOAL);
    if (!key || !goal || !find(grid, DOOR)) return null;
    const a = bfs(grid, agent, key, false); if (a === null) return null;
    const b = bfs(grid, key, goal, true); if (b === null) return null;
    return a + b <= MAX_STEPS ? a + b : null;
  }

  // ---- model ----
  function halfToFloat(h) {  // IEEE 754 binary16 -> number
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    if (e === 0) return s * f * 2 ** -24;
    if (e === 31) return f ? NaN : s * Infinity;
    return s * (1 + f / 1024) * 2 ** (e - 15);
  }
  function decode(b64, dtype) {
    const bin = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    if (dtype !== "f2") return new Float32Array(bytes.buffer);
    const h = new Uint16Array(bytes.buffer), out = new Float32Array(h.length);
    for (let i = 0; i < h.length; i++) out[i] = halfToFloat(h[i]);
    return out;
  }
  function loadModel(blob) {
    const W = {};
    for (const [k, v] of Object.entries(blob.tensors)) W[k] = decode(v.data, blob.dtype);
    return { cfg: blob.config, W };
  }
  // y[n] = b[n] + sum_k x[k] * W[n, k]   (torch Linear layout: W is [out, in])
  function linear(x, W, b, inDim, outDim) {
    const y = new Float32Array(outDim);
    for (let n = 0; n < outDim; n++) {
      let s = b[n]; const o = n * inDim;
      for (let k = 0; k < inDim; k++) s += x[k] * W[o + k];
      y[n] = s;
    }
    return y;
  }
  function layerNorm(x, g, b) {
    const d = x.length; let m = 0, v = 0;
    for (let i = 0; i < d; i++) m += x[i];
    m /= d;
    for (let i = 0; i < d; i++) v += (x[i] - m) ** 2;
    const inv = 1 / Math.sqrt(v / d + 1e-5), y = new Float32Array(d);
    for (let i = 0; i < d; i++) y[i] = (x[i] - m) * inv * g[i] + b[i];
    return y;
  }
  function softmax(z) {
    const m = Math.max(...z); const e = Array.from(z, v => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s);
  }
  // tokens: Int32Array(1 + INNER * INNER) -> logits Float32Array(4)
  function forward(model, tokens) {
    const { d, heads, layers, mlp } = model.cfg, W = model.W, T = tokens.length, hd = d / heads;
    let x = [];
    for (let t = 0; t < T; t++) {
      const v = new Float32Array(d);
      for (let i = 0; i < d; i++) v[i] = W["tok.weight"][tokens[t] * d + i] + W["pos.weight"][t * d + i];
      x.push(v);
    }
    for (let l = 0; l < layers; l++) {
      const p = `blocks.${l}.`;
      const qkv = x.map(v => linear(layerNorm(v, W[p + "ln1.weight"], W[p + "ln1.bias"]),
        W[p + "qkv.weight"], W[p + "qkv.bias"], d, 3 * d));
      const y = x.map(() => new Float32Array(d));
      const scale = 1 / Math.sqrt(hd), sc = new Float32Array(T);
      for (let h = 0; h < heads; h++) {
        const qo = h * hd, ko = d + h * hd, vo = 2 * d + h * hd;
        for (let i = 0; i < T; i++) {
          let mx = -Infinity;
          for (let j = 0; j < T; j++) {
            let s = 0; for (let k = 0; k < hd; k++) s += qkv[i][qo + k] * qkv[j][ko + k];
            sc[j] = s * scale; if (sc[j] > mx) mx = sc[j];
          }
          let sum = 0; for (let j = 0; j < T; j++) { sc[j] = Math.exp(sc[j] - mx); sum += sc[j]; }
          for (let j = 0; j < T; j++) {
            const w = sc[j] / sum;
            for (let k = 0; k < hd; k++) y[i][qo + k] += w * qkv[j][vo + k];
          }
        }
      }
      x = x.map((v, i) => {
        const a = linear(y[i], W[p + "proj.weight"], W[p + "proj.bias"], d, d);
        const r = new Float32Array(d); for (let k = 0; k < d; k++) r[k] = v[k] + a[k];
        const h1 = linear(layerNorm(r, W[p + "ln2.weight"], W[p + "ln2.bias"]), W[p + "fc1.weight"], W[p + "fc1.bias"], d, mlp);
        for (let k = 0; k < mlp; k++) if (h1[k] < 0) h1[k] = 0;
        const h2 = linear(h1, W[p + "fc2.weight"], W[p + "fc2.bias"], mlp, d);
        for (let k = 0; k < d; k++) r[k] += h2[k];
        return r;
      });
    }
    return linear(layerNorm(x[0], W["ln_f.weight"], W["ln_f.bias"]), W["head.weight"], W["head.bias"], d, 4);
  }

  const Core = { SIZE, INNER, MAX_STEPS, EMPTY, WALL, KEY, DOOR, GOAL, MOVES, ACTIONS,
    Env, bfs, optimalSteps, loadModel, forward, softmax };
  if (typeof module !== "undefined") module.exports = Core; else root.Core = Core;
})(this);
