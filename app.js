// Vibe Coder Bootcamp: course UI, in-browser Python runner (Skulpt), and concept cards.
(function () {
  const $ = id => document.getElementById(id);
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} },
  };

  const PARTS = [
    { name: "Part 1 · Foundations", lessons: FOUNDATION_LESSONS },
    { name: "Part 2 · ML from scratch", lessons: ML_LESSONS },
    { name: "Part 3 · NumPy", lessons: NUMPY_LESSONS },
    { name: "Part 4 · pandas and data prep", lessons: DATA_LESSONS },
    { name: "Part 5 · PyTorch", lessons: PYTORCH_LESSONS },
  ];
  const LESSONS = PARTS.flatMap(p => p.lessons.map(l => Object.assign({ part: p.name }, l)));
  const DEFAULT_OUT = "Write your code, then press Run tests.\nUse print() anywhere to debug.";

  let solved = store.get("vcb.solved", {});
  let cur = store.get("vcb.cur", 0);
  if (cur >= LESSONS.length) cur = 0;
  let hintN = 0, solStep = 0;
  const ed = $("code");

  // ---------- Python runtimes ----------
  // Main runtime: Pyodide (real CPython compiled to WebAssembly) in a Web Worker, so
  // NumPy and pandas work and an infinite loop can be stopped. If Pyodide can't load,
  // plain-Python lessons fall back to Skulpt, a small JS interpreter.
  const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
  const RUN_TIMEOUT = 20000;
  const WORKER_SRC = `
let py = null; const loaded = new Set();
function write(files) {
  for (const [p, c] of Object.entries(files)) {
    let d = "/home/pyodide";
    for (const part of p.split("/").slice(0, -1)) { d += "/" + part; try { py.FS.mkdir(d); } catch (_) {} }
    py.FS.writeFile("/home/pyodide/" + p, c);
  }
}
self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "init") {
      importScripts(m.index + "pyodide.js");
      py = await loadPyodide({ indexURL: m.index });
      write(m.files);
      py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
      postMessage({ id: m.id, ok: true }); return;
    }
    if (m.type === "run") {
      const need = (m.packages || []).filter(p => p !== "torch" && !loaded.has(p));
      if (need.length) { await py.loadPackage(need); need.forEach(p => loaded.add(p)); }
      postMessage({ id: m.id, started: true });
      let out = "";
      py.setStdout({ batched: s => { out += s + "\\n"; } });
      py.setStderr({ batched: s => { out += s + "\\n"; } });
      const g = py.globals.get("dict")();
      try {
        await py.runPythonAsync(m.code, { globals: g });
        let results = null;
        if (g.has("_results")) results = g.get("_results").toJs();
        postMessage({ id: m.id, ok: true, out, results });
      } catch (err) {
        postMessage({ id: m.id, ok: false, out, error: String(err.message || err) });
      } finally { g.destroy(); }
    }
  } catch (err) { postMessage({ id: m.id, ok: false, out: "", error: String(err.message || err) }); }
};`;
  let engine = "loading"; // "pyodide" | "skulpt" | "none" | "loading"
  let worker = null, msgId = 0, pending = {};
  function setStatus(text, cls) { $("status").textContent = text; $("status").className = cls || ""; }
  function startWorker() {
    return new Promise((resolve, reject) => {
      try {
        const url = URL.createObjectURL(new Blob([WORKER_SRC], { type: "text/javascript" }));
        worker = new Worker(url);
      } catch (e) { reject(e); return; }
      worker.onmessage = (e) => { const p = pending[e.data.id]; if (p) p(e.data); };
      worker.onerror = (e) => reject(e);
      const id = ++msgId;
      const t = setTimeout(() => reject(new Error("timeout")), 60000);
      pending[id] = (r) => { clearTimeout(t); delete pending[id]; r.ok ? resolve() : reject(new Error(r.error)); };
      worker.postMessage({ type: "init", id, index: PYODIDE, files: typeof TORCH_FILES !== "undefined" ? TORCH_FILES : {} });
    });
  }
  function boot() {
    engine = "loading";
    setStatus("Loading Python… (about 10 s the first time)", "loading");
    return startWorker().then(() => { engine = "pyodide"; setStatus("Python 3 ready"); })
      .catch(() => {
        try { worker && worker.terminate(); } catch (_) {}
        worker = null;
        engine = typeof Sk !== "undefined" ? "skulpt" : "none";
        setStatus(engine === "skulpt" ? "Basic Python only (NumPy/pandas unavailable here)" : "Python didn't load. Check your connection and reload.", engine === "skulpt" ? "loading" : "err");
        refreshNotice();
      });
  }
  let booting = boot();

  function readf(x) {
    if (Sk.builtinFiles === undefined || Sk.builtinFiles.files[x] === undefined) throw "File not found: '" + x + "'";
    return Sk.builtinFiles.files[x];
  }
  function skulptRun(src, wantResults) {
    let out = "";
    Sk.configure({ output: t => (out += t), read: readf, __future__: Sk.python3, execLimit: 6000, yieldLimit: 100 });
    return Sk.misceval.asyncToPromise(() => Sk.importMainWithBody("<stdin>", false, src, true))
      .then(m => ({ ok: true, out, results: wantResults ? Sk.ffi.remapToJs(m.tp$getattr(new Sk.builtin.str("_results"))) : null }))
      .catch(e => ({ ok: false, out, error: e.toString() }));
  }
  function pyodideRun(src, packages, onStart) {
    return new Promise((resolve) => {
      const id = ++msgId; let timer = null;
      pending[id] = (r) => {
        if (r.started) { if (onStart) onStart(); timer = setTimeout(() => {
          delete pending[id];
          try { worker.terminate(); } catch (_) {}
          booting = boot();
          resolve({ ok: false, out: "", error: "TimeLimitError: your code ran for more than " + RUN_TIMEOUT / 1000 + " seconds and was stopped." });
        }, RUN_TIMEOUT); return; }
        clearTimeout(timer); delete pending[id]; resolve(r);
      };
      worker.postMessage({ type: "run", id, code: src, packages });
    });
  }
  // Runs code with the current lesson's packages. Returns {ok, out, results, error}.
  async function runPython(src, wantResults, onStart) {
    await booting;
    const pk = LESSONS[cur].packages || [];
    if (engine === "pyodide") return pyodideRun(src, pk, onStart);
    if (engine === "skulpt" && !pk.length) return skulptRun(src, wantResults);
    return { ok: false, out: "", error: needsFullMessage() };
  }
  function needsFullMessage() {
    return "This lesson needs NumPy, pandas or PyTorch, which only load when the course runs as a normal website " +
      "(GitHub Pages, or `python3 -m http.server` in the project folder). Plain-Python lessons still work here.";
  }
  function refreshNotice() {
    const pk = LESSONS[cur] && LESSONS[cur].packages;
    $("libnote").hidden = !(pk && pk.length && engine !== "pyodide" && engine !== "loading");
    $("libnote").textContent = needsFullMessage();
  }
  function cleanTrace(s) {
    // Drop Pyodide's internal frames so the traceback shows only the learner's code.
    const lines = s.split("\n");
    const i = lines.findIndex(l => l.includes('File "<exec>"'));
    const body = i >= 0 ? ["Traceback (most recent call last):"].concat(lines.slice(i)) : lines;
    return body.filter(l => !/\/lib\/python3|_pyodide/.test(l)).join("\n").replace(/File "<exec>"/g, "File \"solution.py\"").trim();
  }
  function friendly(err) {
    let s = cleanTrace(String(err));
    if (/NoneType/.test(s)) s += "\n\nTip: something returned None. Did you forget a return statement, or leave 'pass' in?";
    if (/TimeLimitError/.test(s)) s += "\n\nTip: your code ran too long. Look for a loop that never ends.";
    if (/IndentationError/.test(s)) s += "\n\nTip: Python needs consistent indentation. Use 4 spaces per level.";
    const mod = s.match(/No module named '(numpy|pandas)'/);
    if (mod) return "ModuleNotFoundError: " + mod[1] + " didn't download. Check your internet connection and reload the page.";
    s = s.split("\nThe module '")[0];
    if (/NameError/.test(s)) s += "\n\nTip: a name is used before it exists. Check spelling, and that you defined it above.";
    return s;
  }
  function showErr(el, text) { el.innerHTML = ""; const sp = document.createElement("span"); sp.className = "err"; sp.textContent = text; el.appendChild(sp); }

  // ---------- Navigation ----------
  function solvedCount() { return LESSONS.filter(l => solved[l.id]).length; }
  function buildNav() {
    const rail = $("rail"), pick = $("picker");
    rail.innerHTML = ""; pick.innerHTML = "";
    let part = null, st = null, grp = null;
    LESSONS.forEach((l, i) => {
      if (l.part !== part) { part = l.part; const p = document.createElement("div"); p.className = "part"; p.textContent = part; rail.appendChild(p); }
      if (l.stage !== st) {
        st = l.stage;
        const h = document.createElement("div"); h.className = "stage"; h.textContent = st; rail.appendChild(h);
        grp = document.createElement("optgroup"); grp.label = st; pick.appendChild(grp);
      }
      const b = document.createElement("button");
      b.className = "l" + (solved[l.id] ? " done" : "");
      b.setAttribute("aria-current", i === cur ? "true" : "false");
      b.innerHTML = '<span class="dot">' + (solved[l.id] ? "✓" : "") + "</span><span></span>";
      b.lastChild.textContent = l.title;
      b.onclick = () => go(i);
      rail.appendChild(b);
      const o = document.createElement("option");
      o.value = i; o.textContent = (solved[l.id] ? "✓ " : "") + l.title; if (i === cur) o.selected = true;
      grp.appendChild(o);
    });
    pick.onchange = () => go(+pick.value);
    const n = solvedCount();
    $("pbar").style.width = (n / LESSONS.length * 100) + "%";
    $("ptext").textContent = n + " of " + LESSONS.length + " lessons solved";
  }
  function gutter() {
    const n = ed.value.split("\n").length; let s = "";
    for (let i = 1; i <= n; i++) s += i + "\n";
    $("gut").textContent = s;
    ed.style.height = "auto"; ed.style.height = ed.scrollHeight + "px";
  }
  function go(i) {
    saveCode(); cur = i; store.set("vcb.cur", i);
    const l = LESSONS[i]; hintN = 0; solStep = 0;
    $("eyebrow").textContent = l.stage + " · lesson " + (i + 1) + " of " + LESSONS.length;
    $("title").textContent = l.title;
    $("learnbox").hidden = !l.learn;
    if (l.learn) { $("learn").innerHTML = l.learn; $("example").textContent = l.example; $("exout").hidden = true; }
    drawFigure(l);
    $("why").textContent = l.why; $("task").innerHTML = l.task; $("pseudo").textContent = l.pseudo;
    ed.value = store.get("vcb.code." + l.id, l.starter); gutter();
    ["d-pseudo", "d-hints", "d-sol"].forEach(d => ($(d).open = false));
    renderHints(); renderSol();
    $("res").hidden = true; $("banner").innerHTML = ""; $("out").textContent = DEFAULT_OUT;
    refreshNotice();
    buildNav();
  }

  // ---------- Diagrams ----------
  // Place the figure after the explanation when a lesson has one, otherwise under the title.
  function drawFigure(l) {
    const fig = $("fig"), v = typeof VISUALS !== "undefined" && VISUALS[l.id];
    fig.hidden = !v;
    if (!v) return;
    try { $("figsvg").innerHTML = v.draw(); $("figcap").textContent = v.caption; }
    catch (e) { fig.hidden = true; return; }
    if (l.learn) $("learn").after(fig); else $("title").after(fig);
  }

  // ---------- Hints & solution ----------
  function renderHints() {
    const l = LESSONS[cur], ol = $("hints"); ol.innerHTML = "";
    for (let i = 0; i < hintN; i++) { const li = document.createElement("li"); li.textContent = l.hints[i]; ol.appendChild(li); }
    if (hintN === 0) { const li = document.createElement("li"); li.style.listStyle = "none"; li.style.color = "var(--muted)"; li.textContent = "Hints appear one at a time."; ol.appendChild(li); }
    $("hcount").textContent = hintN + "/" + l.hints.length;
    $("morehint").hidden = hintN >= l.hints.length;
  }
  $("morehint").onclick = () => { hintN++; renderHints(); };
  function renderSol() {
    const b = $("solbody"), l = LESSONS[cur]; b.innerHTML = "";
    if (solStep === 0) {
      b.innerHTML = '<div class="warnbox">Struggle for 15 minutes first. Reading a solution feels like learning, but writing it is what sticks.</div>';
      const btn = document.createElement("button"); btn.className = "btn plain"; btn.textContent = "I tried, show me";
      btn.onclick = () => { solStep = 1; renderSol(); }; b.appendChild(btn);
    } else {
      const p = document.createElement("pre"); p.className = "pc"; p.textContent = l.solution; b.appendChild(p);
      const n = document.createElement("p"); n.className = "note"; n.style.marginTop = "10px";
      n.textContent = "Now close this, reset your code, and write it again from memory. Then redo it tomorrow.";
      b.appendChild(n);
    }
  }

  // ---------- Editor ----------
  function saveCode() { if (LESSONS[cur]) store.set("vcb.code." + LESSONS[cur].id, ed.value); }
  ed.addEventListener("input", () => { gutter(); saveCode(); });
  ed.addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); runTests(); return; }
    const s = ed.selectionStart, en = ed.selectionEnd, v = ed.value;
    const set = (val, pos) => { ed.value = val; ed.selectionStart = ed.selectionEnd = pos; gutter(); saveCode(); };
    if (e.key === "Tab") {
      e.preventDefault();
      if (e.shiftKey) {
        const ls = v.lastIndexOf("\n", s - 1) + 1, m = v.slice(ls).match(/^ {1,4}/);
        if (m) set(v.slice(0, ls) + v.slice(ls + m[0].length), Math.max(ls, s - m[0].length));
      } else set(v.slice(0, s) + "    " + v.slice(en), s + 4);
    } else if (e.key === "Enter") {
      const ls = v.lastIndexOf("\n", s - 1) + 1, line = v.slice(ls, s);
      let ind = line.match(/^ */)[0]; if (/:\s*$/.test(line)) ind += "    ";
      e.preventDefault(); set(v.slice(0, s) + "\n" + ind + v.slice(en), s + 1 + ind.length);
    } else if (e.key === "Backspace" && s === en) {
      const ls = v.lastIndexOf("\n", s - 1) + 1, before = v.slice(ls, s);
      if (before.length && /^ +$/.test(before) && before.length % 4 === 0) { e.preventDefault(); set(v.slice(0, s - 4) + v.slice(s), s - 4); }
    }
  });

  // ---------- Running ----------
  let busy = false;
  function loadingText(base) {
    const pk = (LESSONS[cur].packages || []).filter(p => p !== "torch");
    return pk.length ? "Loading " + pk.join(" and ") + " (first time only)…" : base;
  }
  async function guarded(fn) { if (busy) return; busy = true; try { await fn(); } finally { busy = false; } }
  const runExample = () => guarded(async () => {
    const box = $("exout"); box.hidden = false; box.textContent = loadingText("Running…");
    const r = await runPython(LESSONS[cur].example, false, () => (box.textContent = "Running…"));
    if (r.ok) box.textContent = r.out || "(no output)"; else showErr(box, r.out + friendly(r.error));
  });
  const runOnly = () => guarded(async () => {
    $("res").hidden = true; $("banner").innerHTML = ""; $("out").textContent = loadingText("Running…");
    const r = await runPython(ed.value, false, () => ($("out").textContent = "Running…"));
    if (r.ok) $("out").textContent = r.out || "(no output. Add print() to see values.)"; else showErr($("out"), r.out + friendly(r.error));
  });
  const runTests = () => guarded(async () => {
    const l = LESSONS[cur];
    $("out").textContent = loadingText("Running tests…"); $("banner").innerHTML = "";
    const res = $("res"); res.innerHTML = ""; res.hidden = true;
    const rr = await runPython(ed.value + "\n" + HARNESS + "\n" + l.tests, true, () => ($("out").textContent = "Running tests…"));
    if (!rr.ok) { showErr($("out"), rr.out + friendly(rr.error)); return; }
    const r = rr.results || []; const out = rr.out;
    let pass = 0;
    r.forEach(([ok, m]) => {
      const d = document.createElement("div"); d.className = ok ? "p" : "f";
      d.textContent = (ok ? "✓ " : "✗ ") + m; res.appendChild(d); if (ok) pass++;
    });
    res.hidden = false; $("out").textContent = out;
    if (pass === r.length) {
      solved[l.id] = true; store.set("vcb.solved", solved);
      const last = cur === LESSONS.length - 1;
      $("banner").innerHTML = '<div class="banner win">All ' + r.length + " tests pass." +
        (last ? " You finished the course." : ' <button class="btn plain" id="nextb" style="margin-left:8px">Next lesson</button>') + "</div>";
      const nb = $("nextb"); if (nb) nb.onclick = () => go(cur + 1);
      buildNav();
    }
  });
  $("runex").onclick = runExample;
  $("run").onclick = runOnly;
  $("check").onclick = runTests;
  $("reset").onclick = function () {
    if (this.dataset.arm) { ed.value = LESSONS[cur].starter; gutter(); saveCode(); this.textContent = "Reset code"; delete this.dataset.arm; }
    else { this.dataset.arm = 1; this.textContent = "Click again to reset"; setTimeout(() => { this.textContent = "Reset code"; delete this.dataset.arm; }, 3000); }
  };

  // ---------- Timer ----------
  let tLeft = 1200, tId = null;
  function tdraw() {
    const a = Math.abs(tLeft), m = Math.floor(a / 60), s = a % 60;
    $("tclock").textContent = (tLeft < 0 ? "-" : "") + m + ":" + String(s).padStart(2, "0");
    $("tclock").className = tLeft < 0 ? "over" : "";
  }
  $("tstart").onclick = () => {
    if (tId) { clearInterval(tId); tId = null; $("tstart").textContent = "Start"; }
    else { tId = setInterval(() => { tLeft--; tdraw(); }, 1000); $("tstart").textContent = "Pause"; }
  };
  $("treset").onclick = () => { clearInterval(tId); tId = null; tLeft = 1200; tdraw(); $("tstart").textContent = "Start"; };

  // ---------- Concept cards ----------
  let rating = store.get("vcb.cards", {}); // index -> "good" | "again"
  const topics = ["All", ...new Set(CONCEPTS.map(c => c.topic))];
  let topic = "All", deck = [], pos = 0, shown = false;
  function buildDeck() {
    const idx = CONCEPTS.map((c, i) => i).filter(i => topic === "All" || CONCEPTS[i].topic === topic);
    const rank = i => (rating[i] === "again" ? 0 : rating[i] === "good" ? 2 : 1);
    deck = idx.sort((a, b) => rank(a) - rank(b) || a - b); pos = 0; shown = false; drawCard();
  }
  function drawFilters() {
    const f = $("filters"); f.innerHTML = "";
    topics.forEach(t => {
      const b = document.createElement("button"); b.className = "chip"; b.textContent = t;
      b.setAttribute("aria-pressed", t === topic ? "true" : "false");
      b.onclick = () => { topic = t; drawFilters(); buildDeck(); }; f.appendChild(b);
    });
  }
  function drawCard() {
    const i = deck[pos], c = CONCEPTS[i];
    $("ctopic").textContent = c.topic + (rating[i] === "again" ? " · marked for review" : rating[i] === "good" ? " · you knew this" : "");
    $("cpos").textContent = (pos + 1) + " / " + deck.length;
    $("cq").textContent = c.q; $("ca").textContent = c.a; $("ca").hidden = !shown;
    const btns = $("cbtns"); btns.innerHTML = "";
    const mk = (label, cls, fn) => { const b = document.createElement("button"); b.className = "btn " + cls; b.textContent = label; b.onclick = fn; btns.appendChild(b); };
    if (!shown) mk("Reveal answer", "primary", () => { shown = true; drawCard(); });
    else {
      mk("I knew it", "good", () => rate("good"));
      mk("Review again", "again", () => rate("again"));
    }
    if (pos > 0) mk("Back", "plain", () => { pos--; shown = false; drawCard(); });
    const g = Object.values(rating).filter(v => v === "good").length;
    $("ctally").textContent = g + " of " + CONCEPTS.length + " cards marked as known.";
  }
  function rate(v) { rating[deck[pos]] = v; store.set("vcb.cards", rating); pos = (pos + 1) % deck.length; shown = false; drawCard(); }

  // ---------- Tabs ----------
  function tab(which) {
    const course = which === "course";
    $("view-course").hidden = !course; $("view-cards").hidden = course;
    $("tab-course").setAttribute("aria-selected", course); $("tab-cards").setAttribute("aria-selected", !course);
    store.set("vcb.tab", which);
  }
  $("tab-course").onclick = () => tab("course");
  $("tab-cards").onclick = () => tab("cards");

  drawFilters(); buildDeck();
  tab(location.hash === "#cards" ? "cards" : store.get("vcb.tab", "course"));
  go(cur);
})();
