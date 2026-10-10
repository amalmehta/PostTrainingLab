"""The browser (site/js/core.js) must agree with PyTorch/NumPy exactly."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from posttrain.env import BatchEnv, make_level
from posttrain.model import Policy, export_weights, load_weights

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("dtype", ["f4", "f2"])  # f2 = the float16 training snapshots
def test_js_matches_python(tmp_path, dtype):
    torch.manual_seed(0)
    blob_w = export_weights(Policy(), dtype=dtype)
    model = load_weights(blob_w)  # what the browser sees, after any rounding
    rng = np.random.default_rng(5)
    cases = []
    for _ in range(4):
        g, a = make_level(rng)
        acts = rng.integers(0, 4, size=20).tolist()
        env = BatchEnv([g], [a])
        obs = []
        for act in acts:
            obs.append(env.obs()[0].tolist())
            env.step(np.array([act]))
        with torch.no_grad():
            logits = model(torch.tensor(obs)).tolist()
        cases.append(dict(grid=g.reshape(-1).tolist(), agent=list(a), actions=acts,
                          obs=obs, logits=logits, steps=int(env.steps[0]),
                          success=bool(env.success[0])))
    blob = dict(model=blob_w, cases=cases)
    (tmp_path / "case.json").write_text(json.dumps(blob))
    out = subprocess.run(["node", str(ROOT / "tests" / "parity.js"), str(tmp_path / "case.json")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "ok" in out.stdout
