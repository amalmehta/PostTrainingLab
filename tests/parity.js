// Replays Python-recorded episodes through site/js/core.js and compares.
const fs = require("fs");
const Core = require("../site/js/core.js");
const blob = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const model = Core.loadModel(blob.model);
let maxErr = 0;
for (const c of blob.cases) {
  const env = new Core.Env(c.grid, c.agent);
  c.actions.forEach((a, t) => {
    const o = Array.from(env.obs());
    if (o.join() !== c.obs[t].join()) throw new Error(`obs mismatch at step ${t}`);
    const z = Core.forward(model, env.obs());
    z.forEach((v, i) => { maxErr = Math.max(maxErr, Math.abs(v - c.logits[t][i])); });
    env.step(a);
  });
  if (env.steps !== c.steps || env.success !== c.success) throw new Error("env end-state mismatch");
}
if (maxErr > 1e-4) throw new Error(`logit mismatch ${maxErr}`);
console.log(`ok max logit err ${maxErr.toExponential(2)}`);
