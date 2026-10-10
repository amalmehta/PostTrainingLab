// Minimal SVG line charts: recessive grid, 2px lines, end labels,
// crosshair + tooltip on hover. Colors are CSS custom properties.
(function (root) {
  const NS = "http://www.w3.org/2000/svg";
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }
  function niceTicks(lo, hi, n) {
    const span = hi - lo || 1, raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || mag * 10;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  // opts: {title, sub, series:[{name, color, points:[[x,y]], marker}], yMin, yMax,
  //        yFmt, xFmt, xLabel, refs:[{y, label}], endLabels, height}
  function lineChart(container, opts) {
    const W = 360, H = opts.height || 210, m = { l: 38, r: opts.endLabels ? 70 : 12, t: 10, b: 28 };
    const box = document.createElement("div");
    box.className = "chart";
    box.innerHTML = `<div class="title">${opts.title}</div>` + (opts.sub ? `<div class="csub">${opts.sub}</div>` : "");
    container.appendChild(box);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.title }, box);
    const all = opts.series.flatMap(s => s.points);
    const bands = opts.series.flatMap(s => s.band || []);
    const xs = all.map(p => p[0]), ys = all.map(p => p[1]).concat((opts.refs || []).map(r => r.y),
      bands.map(b => b[1]), bands.map(b => b[2]));
    const x0 = opts.xMin ?? Math.min(...xs), x1 = opts.xMax ?? Math.max(...xs);
    let y0 = opts.yMin ?? Math.min(0, ...ys), y1 = opts.yMax ?? Math.max(...ys) * 1.05;
    if (y1 === y0) y1 = y0 + 1;
    const X = v => m.l + ((v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
    const Y = v => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
    const yFmt = opts.yFmt || (v => +v.toFixed(2)), xFmt = opts.xFmt || (v => v);

    for (const t of niceTicks(y0, y1, 4)) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "gridline" }, svg);
      el("text", { x: m.l - 6, y: Y(t) + 3.5, "text-anchor": "end" }, svg).textContent = yFmt(t);
    }
    for (const t of opts.xTicks || niceTicks(x0, x1, 5))
      el("text", { x: X(t), y: H - m.b + 15, "text-anchor": "middle" }, svg).textContent = xFmt(t);
    el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }, svg);
    if (opts.xLabel) el("text", { x: W - m.r, y: H - 2, "text-anchor": "end" }, svg).textContent = opts.xLabel;
    for (const r of opts.refs || []) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(r.y), y2: Y(r.y), class: "ref" }, svg);
      if (r.label) el("text", { x: m.l + 4, y: Y(r.y) - 4 }, svg).textContent = r.label;
    }
    const ends = [];
    for (const s of opts.series) {
      if (!s.points.length) continue;
      if (s.band && s.band.length > 1) {  // [[x, lo, hi]] drawn as a light wash under the line
        const up = s.band.map((b, i) => `${i ? "L" : "M"}${X(b[0]).toFixed(1)},${Y(b[2]).toFixed(1)}`).join("");
        const dn = s.band.slice().reverse().map(b => `L${X(b[0]).toFixed(1)},${Y(b[1]).toFixed(1)}`).join("");
        el("path", { d: up + dn + "Z", class: "band", style: `fill:var(${s.color})` }, svg);
      }
      const d = s.points.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
      el("path", { d, class: "series", style: `stroke:var(${s.color})` }, svg);
      if (s.marker) for (const p of s.points)
        el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4, style: `fill:var(${s.color});stroke:var(--surface);stroke-width:2` }, svg);
      const last = s.points[s.points.length - 1];
      el("circle", { cx: X(last[0]), cy: Y(last[1]), r: 4, style: `fill:var(${s.color});stroke:var(--surface);stroke-width:2` }, svg);
      ends.push({ y: Y(last[1]), x: X(last[0]), s, v: last[1] });
    }
    if (opts.endLabels) {  // only when they don't collide; legend carries identity otherwise
      ends.sort((a, b) => a.y - b.y);
      const clash = ends.some((e, i) => i && e.y - ends[i - 1].y < 12);
      if (!clash) for (const e of ends)
        el("text", { x: e.x + 8, y: e.y + 4, class: "endlabel" }, svg).textContent = `${e.s.name} ${yFmt(e.v)}`;
    }

    // hover layer
    const cross = el("line", { y1: m.t, y2: H - m.b, class: "cross", visibility: "hidden" }, svg);
    const hit = el("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: "transparent" }, svg);
    const tip = document.createElement("div");
    tip.className = "tip"; tip.hidden = true; box.appendChild(tip);
    const allX = [...new Set(xs)].sort((a, b) => a - b);
    hit.addEventListener("mousemove", ev => {
      const r = svg.getBoundingClientRect(), sx = (ev.clientX - r.left) * W / r.width;
      const xv = x0 + (sx - m.l) / (W - m.l - m.r) * (x1 - x0);
      const near = allX.reduce((a, b) => Math.abs(b - xv) < Math.abs(a - xv) ? b : a);
      cross.setAttribute("x1", X(near)); cross.setAttribute("x2", X(near)); cross.setAttribute("visibility", "visible");
      const rows = opts.series.map(s => {
        const p = s.points.reduce((a, b) => Math.abs(b[0] - near) < Math.abs(a[0] - near) ? b : a, s.points[0]);
        return p ? `<div class="row"><span class="dot" style="background:var(${s.color})"></span>${s.name}<b>${yFmt(p[1])}</b></div>` : "";
      }).join("");
      tip.innerHTML = `<div style="color:var(--muted)">${opts.xLabel || "x"} ${xFmt(near)}</div>${rows}`;
      tip.hidden = false;
      const bx = box.getBoundingClientRect(), px = ev.clientX - bx.left, py = ev.clientY - bx.top;
      tip.style.left = Math.min(px + 12, bx.width - tip.offsetWidth - 4) + "px";
      tip.style.top = Math.max(py - tip.offsetHeight - 8, 4) + "px";
    });
    hit.addEventListener("mouseleave", () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
    return box;
  }

  function legend(container, items) {
    container.innerHTML = items.map(i => `<span><i style="background:var(${i.color})"></i>${i.name}</span>`).join("");
  }

  root.Charts = { lineChart, legend, svgEl: el };
})(this);
