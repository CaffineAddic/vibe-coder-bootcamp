// Vibe Coder Bootcamp: course UI, in-browser Python runner (Skulpt), and concept cards.
(function () {
  const $ = id => document.getElementById(id);
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} },
  };

  const PARTS = [
    { name: "Part 1 · Foundations", lessons: FOUNDATION_LESSONS },
    { name: "Part 2 · ML coding", lessons: ML_LESSONS },
  ];
  const LESSONS = PARTS.flatMap(p => p.lessons.map(l => Object.assign({ part: p.name }, l)));
  const DEFAULT_OUT = "Write your code, then press Run tests.\nUse print() anywhere to debug.";

  let solved = store.get("vcb.solved", {});
  let cur = store.get("vcb.cur", 0);
  if (cur >= LESSONS.length) cur = 0;
  let hintN = 0, solStep = 0;
  const ed = $("code");
  const ready = typeof Sk !== "undefined";
  $("status").textContent = ready ? "Python ready" : "Python didn't load. Check your connection and reload.";
  $("status").className = ready ? "" : "err";

  // ---------- Python runner ----------
  function readf(x) {
    if (Sk.builtinFiles === undefined || Sk.builtinFiles.files[x] === undefined) throw "File not found: '" + x + "'";
    return Sk.builtinFiles.files[x];
  }
  function exec(src, onOut) {
    Sk.configure({ output: onOut, read: readf, __future__: Sk.python3, execLimit: 6000, yieldLimit: 100 });
    return Sk.misceval.asyncToPromise(() => Sk.importMainWithBody("<stdin>", false, src, true));
  }
  function friendly(err) {
    let s = err.toString();
    if (/NoneType/.test(s)) s += "\n\nTip: something returned None. Did you forget a return statement, or leave 'pass' in?";
    if (/TimeLimitError/.test(s)) s += "\n\nTip: your code ran too long. Look for a loop that never ends.";
    if (/IndentationError/.test(s)) s += "\n\nTip: Python needs consistent indentation. Use 4 spaces per level.";
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
    $("why").textContent = l.why; $("task").innerHTML = l.task; $("pseudo").textContent = l.pseudo;
    ed.value = store.get("vcb.code." + l.id, l.starter); gutter();
    ["d-pseudo", "d-hints", "d-sol"].forEach(d => ($(d).open = false));
    renderHints(); renderSol();
    $("res").hidden = true; $("banner").innerHTML = ""; $("out").textContent = DEFAULT_OUT;
    buildNav();
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
  async function runExample() {
    if (!ready) return;
    const box = $("exout"); box.hidden = false; box.textContent = "Running…"; let out = "";
    try { await exec(LESSONS[cur].example, t => (out += t)); box.textContent = out || "(no output)"; }
    catch (e) { showErr(box, out + friendly(e)); }
  }
  async function runOnly() {
    if (!ready) return;
    let out = ""; $("res").hidden = true; $("banner").innerHTML = ""; $("out").textContent = "Running…";
    try { await exec(ed.value, t => (out += t)); $("out").textContent = out || "(no output. Add print() to see values.)"; }
    catch (e) { showErr($("out"), out + friendly(e)); }
  }
  async function runTests() {
    if (!ready) return;
    const l = LESSONS[cur]; let out = "";
    $("out").textContent = "Running tests…"; $("banner").innerHTML = "";
    const res = $("res"); res.innerHTML = ""; res.hidden = true;
    let mod;
    try { mod = await exec(ed.value + "\n" + HARNESS + "\n" + l.tests, t => (out += t)); }
    catch (e) { showErr($("out"), out + friendly(e)); return; }
    const r = Sk.ffi.remapToJs(mod.tp$getattr(new Sk.builtin.str("_results")));
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
  }
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
