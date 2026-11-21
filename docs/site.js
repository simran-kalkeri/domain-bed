(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const SVGNS = "http://www.w3.org/2000/svg";
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const el = (name, attrs, parent, text) => {
    const n = document.createElementNS(SVGNS, name);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  };

  /* ================= loss landscape (conceptual) ================= */
  const SHARP = { c: -1.15, s: 0.13, a: 0.95 };
  const FLAT = { c: 1.0, s: 0.5, a: 0.85 };
  const RHO = 0.25;
  const g = (x, m) => m.a * Math.exp(-((x - m.c) ** 2) / (2 * m.s * m.s));
  const loss = (x) => Math.max(0.02, 1 - g(x, SHARP) - g(x, FLAT));

  const LS = { W: 760, H: 360, l: 52, r: 16, t: 28, b: 40, x0: -2.4, x1: 2.4, y1: 1.1 };
  const narrow = (svg) => svg.getBoundingClientRect().width < 560;
  const lx = (x) => LS.l + ((x - LS.x0) / (LS.x1 - LS.x0)) * (LS.W - LS.l - LS.r);
  const ly = (y) => LS.t + (1 - y / LS.y1) * (LS.H - LS.t - LS.b);
  const pathOf = (f) => {
    let d = "";
    for (let i = 0; i <= 480; i++) {
      const x = LS.x0 + ((LS.x1 - LS.x0) * i) / 480;
      d += `${i ? "L" : "M"}${lx(x).toFixed(1)},${ly(f(x)).toFixed(1)}`;
    }
    return d;
  };

  const ls = $("landscape");
  let shift = 0;

  function drawLandscape() {
    ls.querySelectorAll(":scope > :not(desc)").forEach((n) => n.remove());
    // A smaller drawing box on phones keeps the labels readable.
    LS.W = narrow(ls) ? 460 : 760;
    LS.H = narrow(ls) ? 300 : 360;
    LS.l = narrow(ls) ? 40 : 52;
    ls.setAttribute("viewBox", `0 0 ${LS.W} ${LS.H}`);
    const ink = css("--ink"), muted = css("--ink-2"), accent = css("--accent"), sharpC = css("--sharp");

    for (const y of [0, 0.5, 1]) {
      el("line", { x1: LS.l, x2: LS.W - LS.r, y1: ly(y), y2: ly(y), class: "ls-axis", "stroke-dasharray": y ? "3 6" : "" }, ls);
      el("text", { x: LS.l - 12, y: ly(y) + 4, "text-anchor": "end", class: "ls-tick" }, ls, y.toFixed(1));
    }
    el("text", { x: LS.l, y: 12, class: "ls-tick" }, ls, "Loss");
    el("text", { x: LS.W - LS.r, y: LS.H - 8, "text-anchor": "end", class: "ls-tick" }, ls, "Weights θ");

    // SAM neighbourhoods
    for (const [m, c] of [[SHARP, sharpC], [FLAT, accent]]) {
      el("rect", { x: lx(m.c - RHO), y: LS.t, width: lx(m.c + RHO) - lx(m.c - RHO), height: LS.H - LS.t - LS.b, fill: c, opacity: 0.08, rx: 6 }, ls);
      el("text", { x: lx(m.c), y: LS.H - LS.b + 22, "text-anchor": "middle", class: "ls-tick" }, ls, "±ρ");
    }

    // curves
    el("path", { d: pathOf((x) => loss(x - shift)), fill: "none", stroke: muted, "stroke-width": 2, "stroke-dasharray": "6 6", opacity: shift > 0.005 ? 0.9 : 0 }, ls);
    el("path", { d: pathOf(loss), fill: "none", stroke: ink, "stroke-width": 2.75, "stroke-linejoin": "round" }, ls);

    // solutions and their loss after the shift
    for (const [m, c, label, dx] of [[SHARP, sharpC, "Sharp minimum", 0], [FLAT, accent, "Flat minimum", 0]]) {
      const x = lx(m.c), yTrain = ly(loss(m.c)), yTest = ly(loss(m.c - shift));
      if (shift > 0.005) {
        el("line", { x1: x, x2: x, y1: yTrain, y2: yTest, stroke: c, "stroke-width": 3, "stroke-linecap": "round" }, ls);
        el("circle", { cx: x, cy: yTest, r: 6, fill: css("--surface"), stroke: c, "stroke-width": 2.5 }, ls);
      }
      el("circle", { cx: x, cy: yTrain, r: 7.5, fill: c }, ls);
      // Sharp label sits left of its narrow well, clear of the shifted curve.
      const tx = m === SHARP ? x - 16 : x;
      el("text", { x: tx + dx, y: m === SHARP ? yTrain - (LS.W < 600 ? 12 : -5) : yTrain + 30, "text-anchor": m === SHARP ? "end" : "middle", fill: c, style: "font: 600 14px var(--sans)" }, ls, label);
    }
    el("text", { x: lx(1.75), y: ly(0.62), "text-anchor": "middle", class: "ls-tick", opacity: shift > 0.005 ? 1 : 0 }, ls, "unseen domain");
  }

  function updateReadout() {
    $("shift-out").textContent = shift.toFixed(2);
    $("ro-sharp-train").textContent = loss(SHARP.c).toFixed(2);
    $("ro-sharp-test").textContent = loss(SHARP.c - shift).toFixed(2);
    $("ro-flat-train").textContent = loss(FLAT.c).toFixed(2);
    $("ro-flat-test").textContent = loss(FLAT.c - shift).toFixed(2);
  }

  const slider = $("shift");
  let userMoved = false;
  slider.addEventListener("input", () => { userMoved = true; shift = +slider.value; drawLandscape(); updateReadout(); });
  drawLandscape(); updateReadout();

  // One orchestrated moment: ease the shift in once, unless the visitor prefers less motion.
  if (!reduceMotion) {
    const target = 0.3, dur = 1600;
    setTimeout(() => {
      const t0 = performance.now();
      const tick = (now) => {
        if (userMoved) return;
        const p = Math.min(1, (now - t0) / dur);
        const e = 1 - Math.pow(1 - p, 3);
        shift = target * e;
        slider.value = shift.toFixed(2);
        drawLandscape(); updateReadout();
        if (p < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }, 700);
  } else {
    shift = 0.3; slider.value = "0.3"; drawLandscape(); updateReadout();
  }

  /* ================= results ================= */
  const RESULTS = {
    pacs: {
      name: "PACS",
      domains: ["Art painting", "Cartoon", "Photo", "Sketch"],
      range: [75, 100],
      rows: {
        ERM: [82.34, 78.56, 95.12, 76.89, 83.23],
        "ERM++": [84.12, 79.34, 95.67, 77.32, 84.11],
        "DC-SAM": [85.34, 81.12, 96.23, 79.67, 85.59],
      },
    },
    vlcs: {
      name: "VLCS",
      domains: ["Caltech101", "LabelMe", "SUN09", "VOC2007"],
      range: [60, 100], // VLCS spans 62 to 98
      rows: {
        ERM: [97.23, 62.45, 71.34, 73.12, 76.04],
        "ERM++": [97.56, 63.12, 72.01, 73.89, 76.65],
        "DC-SAM": [98.12, 64.34, 73.45, 74.67, 77.65],
      },
    },
  };
  const ALGOS = [["ERM", "--erm"], ["ERM++", "--ermpp"], ["DC-SAM", "--dcsam"]];
  let bench = "pacs";

  function drawHeadline() {
    const r = RESULTS[bench].rows;
    const avg = (a) => r[a][4];
    const d = (b) => (avg("DC-SAM") - avg(b)).toFixed(2);
    $("headline").innerHTML = `
      <dl class="hn"><dt>ERM, average</dt><dd>${avg("ERM").toFixed(2)}</dd></dl>
      <dl class="hn"><dt>ERM++, average</dt><dd>${avg("ERM++").toFixed(2)}</dd></dl>
      <dl class="hn is-dcsam"><dt>DC-SAM, average</dt><dd>${avg("DC-SAM").toFixed(2)}<small>+${d("ERM")} over ERM, +${d("ERM++")} over ERM++</small></dd></dl>`;
  }

  function drawDots() {
    const svg = $("dotplot");
    const R = RESULTS[bench];
    svg.querySelectorAll(":scope > :not(desc)").forEach((n) => n.remove());
    const small = narrow(svg);
    const W = small ? 440 : 900, rowH = small ? 52 : 58, top = 14, labelW = small ? 104 : 150, right = small ? 56 : 64;
    const labels = [...R.domains, "Average"];
    const H = top + labels.length * rowH + 30 + 14;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const [lo, hi] = R.range;
    const X = (v) => labelW + ((v - lo) / (hi - lo)) * (W - labelW - right);
    const rowY = (i) => top + i * rowH + rowH / 2 + (i === 4 ? 14 : 0);

    for (let v = lo; v <= hi; v += small ? 10 : 5) {
      el("line", { x1: X(v), x2: X(v), y1: top, y2: H - 30, class: "dp-grid", "stroke-dasharray": "2 6" }, svg);
      el("text", { x: X(v), y: H - 8, "text-anchor": "middle", class: "dp-tick" }, svg, v);
    }
    el("line", { x1: 0, x2: W, y1: rowY(4) - rowH / 2 - 4, y2: rowY(4) - rowH / 2 - 4, class: "dp-grid" }, svg);

    labels.forEach((name, i) => {
      const y = rowY(i);
      el("text", { x: 0, y: y + 5, class: "dp-row-label" + (i === 4 ? " avg" : "") }, svg, name);
      const vals = ALGOS.map(([a]) => R.rows[a][i]);
      el("line", { x1: X(Math.min(...vals)), x2: X(Math.max(...vals)), y1: y, y2: y, class: "dp-range" }, svg);
      ALGOS.forEach(([a, c]) => {
        const v = R.rows[a][i];
        const dot = el("circle", { cx: X(v), cy: y, r: a === "DC-SAM" ? 8 : 6.5, fill: css(c), stroke: css("--bg"), "stroke-width": 2 }, svg);
        el("title", {}, dot, `${a}: ${v.toFixed(2)}`);
      });
      el("text", { x: X(R.rows["DC-SAM"][i]) + 14, y: y + 5, class: "dp-val" }, svg, R.rows["DC-SAM"][i].toFixed(2));
    });
  }

  function drawTable() {
    const R = RESULTS[bench];
    const cols = [...R.domains, "Average"];
    const best = cols.map((_, i) => Math.max(...ALGOS.map(([a]) => R.rows[a][i])));
    const head = `<caption class="visually-hidden">${R.name} accuracy by held-out domain</caption><thead><tr><th scope="col">Algorithm</th>${cols.map((c, i) => `<th scope="col"${i === 4 ? ' class="avg"' : ""}>${c}</th>`).join("")}</tr></thead>`;
    const body = ALGOS.map(([a]) => `<tr${a === "DC-SAM" ? ' class="is-dcsam"' : ""}><th scope="row">${a}</th>${R.rows[a].map((v, i) => `<td class="${[v === best[i] ? "best" : "", i === 4 ? "avg" : ""].join(" ").trim()}">${v.toFixed(2)}</td>`).join("")}</tr>`).join("");
    $("results-table").innerHTML = head + `<tbody>${body}</tbody>`;
  }

  const drawResults = () => { drawHeadline(); drawDots(); drawTable(); };
  drawResults();

  // Redraw both charts when crossing the phone breakpoint.
  let wasNarrow = narrow(ls);
  new ResizeObserver(() => {
    const n = narrow(ls);
    if (n !== wasNarrow) { wasNarrow = n; drawLandscape(); drawDots(); }
  }).observe(document.body);

  const tabs = [...document.querySelectorAll('[role="tab"]')];
  function selectTab(t, focus) {
    tabs.forEach((x) => {
      const on = x === t;
      x.setAttribute("aria-selected", String(on));
      x.tabIndex = on ? 0 : -1;
    });
    $("panel-results").setAttribute("aria-labelledby", t.id);
    bench = t.dataset.bench;
    drawResults();
    if (focus) t.focus();
  }
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => selectTab(t));
    t.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        selectTab(tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length], true);
      }
    });
  });

  /* ================= theme ================= */
  const toggle = $("theme-toggle");
  const root = document.documentElement;
  const meta = document.querySelector('meta[name="theme-color"]');
  const sync = () => {
    const dark = root.dataset.theme === "dark";
    toggle.setAttribute("aria-pressed", String(dark));
    meta.content = dark ? "#100F1C" : "#FBFBFD";
  };
  sync();
  toggle.addEventListener("click", () => {
    const dark = root.dataset.theme !== "dark";
    if (dark) root.dataset.theme = "dark"; else delete root.dataset.theme;
    try { localStorage.setItem("theme", dark ? "dark" : "light"); } catch (e) {}
    sync();
    drawLandscape(); drawResults();
  });

  /* ================= copy buttons ================= */
  document.querySelectorAll(".copy").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const code = btn.parentElement.querySelector("code");
      try { await navigator.clipboard.writeText(code.textContent); btn.textContent = "Copied"; }
      catch {
        const r = document.createRange(); r.selectNodeContents(code);
        const s = getSelection(); s.removeAllRanges(); s.addRange(r);
        btn.textContent = "Press Ctrl+C";
      }
      setTimeout(() => (btn.textContent = "Copy"), 1800);
    });
  });
})();
