// Diagrams for the lessons, drawn as inline SVG. Colors come from CSS classes
// (see .fig rules in style.css), so every diagram follows light and dark mode.
// Each entry: id -> { caption, draw: () => svg string }.
const V = (() => {
  const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const svg = (w, h, body, label) =>
    `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px" role="img" aria-label="${esc(label || "")}">${body}</svg>`;
  const rect = (x, y, w, h, cls = "bx", rx = 6) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" class="${cls}"/>`;
  const text = (x, y, s, cls = "t", anchor = "middle") => `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}" dominant-baseline="middle">${esc(s)}</text>`;
  const line = (x1, y1, x2, y2, cls = "ln") => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="${cls}"/>`;
  const arrow = (x1, y1, x2, y2, cls = "ar") => {
    const a = Math.atan2(y2 - y1, x2 - x1), L = 8, W = 4.5;
    const bx = x2 - L * Math.cos(a), by = y2 - L * Math.sin(a);
    const p = [[x2, y2], [bx + W * Math.sin(a), by - W * Math.cos(a)], [bx - W * Math.sin(a), by + W * Math.cos(a)]];
    return `<line x1="${x1}" y1="${y1}" x2="${bx}" y2="${by}" class="${cls}"/><polygon points="${p.map(q => q.join(",")).join(" ")}" class="${cls}h"/>`;
  };
  const path = (d, cls = "ln") => `<path d="${d}" class="${cls}"/>`;
  const circle = (x, y, r, cls = "dot") => `<circle cx="${x}" cy="${y}" r="${r}" class="${cls}"/>`;
  // grid of cells; values[r][c]; hl(r,c) -> class or null
  function grid(x, y, vals, cw = 40, ch = 32, hl = () => null, fmt = v => v) {
    let s = "";
    vals.forEach((row, r) => row.forEach((v, c) => {
      const cls = hl(r, c) || "bx";
      s += rect(x + c * cw, y + r * ch, cw - 4, ch - 4, cls, 4);
      if (v !== "" && v !== null) s += text(x + c * cw + (cw - 4) / 2, y + r * ch + (ch - 4) / 2, fmt(v), cls === "bx" || cls === "bx2" ? "t" : "ta");
    }));
    return s;
  }
  // plot fn over [x0,x1] into box; returns {s, X, Y}
  function axes(px, py, pw, ph, xr, yr, xt, yt, xl, yl) {
    const X = v => px + (v - xr[0]) / (xr[1] - xr[0]) * pw;
    const Y = v => py + ph - (v - yr[0]) / (yr[1] - yr[0]) * ph;
    let s = "";
    (yt || []).forEach(v => { s += line(px, Y(v), px + pw, Y(v), "gr"); s += text(px - 8, Y(v), v, "tm", "end"); });
    (xt || []).forEach(v => { s += text(X(v), py + ph + 14, v, "tm"); });
    s += line(px, py + ph, px + pw, py + ph, "ax") + line(px, py, px, py + ph, "ax");
    if (xl) s += text(px + pw / 2, py + ph + 32, xl, "tm");
    if (yl) s += `<text x="${px - 40}" y="${py + ph / 2}" class="tm" text-anchor="middle" transform="rotate(-90 ${px - 40} ${py + ph / 2})">${esc(yl)}</text>`;
    return { s, X, Y };
  }
  function curve(fn, X, Y, x0, x1, cls = "cv", n = 120) {
    let d = "";
    for (let i = 0; i <= n; i++) { const x = x0 + (x1 - x0) * i / n; d += (i ? "L" : "M") + X(x).toFixed(1) + "," + Y(fn(x)).toFixed(1); }
    return path(d, cls);
  }
  return { esc, svg, rect, text, line, arrow, path, circle, grid, axes, curve };
})();

const VISUALS = (() => {
  const { svg, rect, text, line, arrow, path, circle, grid, axes, curve } = V;
  const D = {};

  // ================= Part 1 =================
  D.variables = { caption: "A variable is a name that points at a value. Each value has a type.", draw: () => {
    const rows = [["lr", "0.01", "float"], ["epochs", "10", "int"], ["name", "\"model_v1\"", "str"], ["done", "False", "bool"]];
    let s = text(70, 16, "names", "tm") + text(330, 16, "values", "tm") + text(530, 16, "type()", "tm");
    rows.forEach(([n, v, t], i) => {
      const y = 32 + i * 46;
      s += rect(20, y, 100, 34, "bx2") + text(70, y + 17, n, "t");
      s += arrow(124, y + 17, 246, y + 17);
      s += rect(250, y, 160, 34, "acc") + text(330, y + 17, v, "ta");
      s += text(530, y + 17, t, "tm");
    });
    return svg(640, 222, s, "Four names pointing to four values");
  }};

  D.functions = { caption: "Arguments go in, the body runs, and return sends one value back. Printing is not returning.", draw: () => {
    let s = rect(20, 40, 120, 70, "bx2") + text(80, 62, "w = 1.0", "t") + text(80, 88, "grad = 2.0", "t");
    s += arrow(144, 75, 196, 75);
    s += rect(200, 20, 256, 110, "acc") + text(328, 46, "def sgd_update(w, grad, lr=0.1):", "tac") + text(328, 80, "w - lr * grad", "ta") + text(328, 108, "lr defaults to 0.1", "tm");
    s += arrow(460, 75, 512, 75) + text(486, 62, "return", "tm");
    s += rect(516, 50, 104, 50, "ok") + text(568, 75, "0.8", "to");
    return svg(640, 150, s, "A function turns arguments into a returned value");
  }};

  D.conditions = { caption: "Python checks each condition from the top and runs only the first branch that's true.", draw: () => {
    const dia = (cx, cy, w, h, label) => path(`M${cx},${cy - h / 2} L${cx + w / 2},${cy} L${cx},${cy + h / 2} L${cx - w / 2},${cy} Z`, "acc") + text(cx, cy, label, "ta");
    let s = rect(250, 6, 140, 30, "bx2") + text(320, 21, "x", "t") + arrow(320, 36, 320, 54);
    s += dia(320, 84, 170, 56, "x < lo ?") + arrow(405, 84, 470, 84) + text(437, 72, "yes", "tm") + rect(474, 68, 140, 32, "ok") + text(544, 84, "return lo", "to");
    s += arrow(320, 112, 320, 136) + text(334, 124, "no", "tm", "start");
    s += dia(320, 166, 170, 56, "x > hi ?") + arrow(405, 166, 470, 166) + text(437, 154, "yes", "tm") + rect(474, 150, 140, 32, "ok") + text(544, 166, "return hi", "to");
    s += arrow(320, 194, 320, 218) + text(334, 206, "no", "tm", "start");
    s += rect(250, 222, 140, 32, "ok") + text(320, 238, "return x", "to");
    return svg(640, 262, s, "Flowchart of clip: two checks, three outcomes");
  }};

  D.loops = { caption: "The accumulator pattern: start before the loop, update inside it, use it after.", draw: () => {
    const xs = [3, 1, 4, 2];
    let s = text(60, 18, "step", "tm") + text(200, 18, "x", "tm") + text(380, 18, "running", "tm");
    s += text(60, 46, "start", "tm") + rect(330, 32, 100, 28, "bx2") + text(380, 46, "0", "t");
    let run = 0;
    xs.forEach((x, i) => {
      const y = 70 + i * 36; run += x;
      s += text(60, y + 14, "i = " + i, "tm") + rect(170, y, 60, 28, "acc") + text(200, y + 14, x, "ta");
      s += text(290, y + 14, "+=", "tm") + rect(330, y, 100, 28, "bx2") + text(380, y + 14, run, "t");
    });
    s += arrow(436, 70 + 3 * 36 + 14, 500, 70 + 3 * 36 + 14) + rect(504, 70 + 3 * 36, 110, 28, "ok") + text(559, 70 + 3 * 36 + 14, "return 10", "to");
    return svg(640, 216, s, "Trace of summing 3, 1, 4, 2");
  }};

  D.lists = { caption: "Indexes start at 0, negative indexes count from the end, and xs[1:4] stops before index 4.", draw: () => {
    const v = [10, 20, 30, 40, 50, 60];
    let s = "";
    v.forEach((x, i) => {
      const X = 100 + i * 86, inS = i >= 1 && i < 4;
      s += text(X + 40, 18, i, "tm") + rect(X, 30, 80, 44, inS ? "acc" : "bx") + text(X + 40, 52, x, inS ? "ta" : "t") + text(X + 40, 90, i - 6, "tm");
    });
    s += text(84, 18, "index", "tm", "end") + text(84, 90, "negative", "tm", "end");
    s += line(186, 108, 186, 120, "lna") + line(186, 120, 444, 120, "lna") + line(444, 108, 444, 120, "lna") + text(315, 136, "xs[1:4] → [20, 30, 40]", "ta");
    return svg(640, 150, s, "List indexing and slicing");
  }};

  D.strings = { caption: "Each method returns a new string. Chain them to clean text step by step.", draw: () => {
    const steps = [["\"  Hello, World!  \"", "raw"], ["\"  hello, world!  \"", ".lower()"], ["\"  hello world  \"", "remove . , ! ?"], ["[\"hello\", \"world\"]", ".split()"], ["\"hello world\"", "\" \".join(...)"]];
    let s = "";
    steps.forEach(([v, op], i) => {
      const y = 8 + i * 44;
      s += rect(200, y, 300, 32, i === 4 ? "ok" : i === 3 ? "acc" : "bx") + text(350, y + 16, v, i === 4 ? "to" : i === 3 ? "ta" : "t");
      if (i) s += text(190, y - 6, op, "tm", "end") + arrow(350, y - 12, 350, y - 1);
    });
    return svg(640, 228, s, "String cleaning pipeline");
  }};

  D.dicts = { caption: "A dict jumps straight to a key's slot through its hash, so lookup takes the same time however big it gets.", draw: () => {
    let s = rect(20, 70, 90, 36, "bx2") + text(65, 88, "\"cat\"", "t") + arrow(114, 88, 184, 88) + rect(188, 64, 110, 48, "acc") + text(243, 82, "hash()", "ta") + text(243, 100, "→ slot 2", "tm");
    s += arrow(302, 88, 372, 88);
    ["\"dog\" : 1", "", "\"cat\" : 0", "", "\"bird\" : 2"].forEach((v, i) => {
      const y = 12 + i * 32;
      s += text(392, y + 14, i, "tm") + rect(406, y, 190, 28, i === 2 ? "ok" : "bx") + (v ? text(501, y + 14, v, i === 2 ? "to" : "t") : "");
    });
    return svg(640, 176, s, "Dictionary lookup through a hash");
  }};

  D.comprehensions = { caption: "[f(x) for x in xs if cond] maps and filters in one line. Every input gets checked, and only the kept ones are transformed.", draw: () => {
    const xs = [-1, 2, -3, 4];
    let s = text(40, 24, "xs", "tm", "start") + text(40, 112, "relu", "tm", "start");
    xs.forEach((x, i) => {
      const X = 120 + i * 120;
      s += rect(X, 8, 80, 32, "bx2") + text(X + 40, 24, x, "t") + arrow(X + 40, 44, X + 40, 92) + text(X + 52, 68, x > 0 ? "keep" : "→ 0", "tm", "start");
      s += rect(X, 96, 80, 32, x > 0 ? "ok" : "acc") + text(X + 40, 112, Math.max(0, x), x > 0 ? "to" : "ta");
    });
    s += text(320, 150, "[max(0, x) for x in xs]", "tac");
    return svg(640, 164, s, "A comprehension maps each element");
  }};

  D.errors = { caption: "Read a traceback from the bottom: the last line names the error, and the line above it shows where it happened.", draw: () => {
    const L = [["Traceback (most recent call last):", "tm"], ["  File \"solution.py\", line 4, in <module>", "tm"], ["  File \"solution.py\", line 2, in safe_div", "ta"], ["    return a / b", "t"], ["ZeroDivisionError: division by zero", "tb"]];
    let s = rect(10, 6, 470, 176, "code");
    L.forEach(([t, c], i) => { s += `<text x="26" y="${32 + i * 32}" class="${c} mono" dominant-baseline="middle">${V.esc(t)}</text>`; });
    s += arrow(560, 160, 476, 160, "arb") + text(565, 160, "1. what went wrong", "tb", "start");
    s += arrow(560, 96, 476, 96, "ara") + text(565, 96, "2. where", "ta", "start");
    return svg(740, 190, s, "Reading a Python traceback");
  }};

  D.classes = { caption: "A class is a blueprint. Each object built from it keeps its own data on self.", draw: () => {
    let s = rect(220, 8, 200, 92, "acc") + text(320, 30, "class RunningMean", "ta") + line(232, 46, 408, 46, "lna") + text(320, 64, "__init__: total, count", "tm") + text(320, 84, "update(x) · value()", "tm");
    s += arrow(280, 104, 150, 150) + arrow(360, 104, 490, 150) + text(196, 122, "RunningMean()", "tm", "end") + text(444, 122, "RunningMean()", "tm", "start");
    s += rect(40, 154, 220, 74, "bx2") + text(150, 176, "loss_meter", "t") + text(150, 202, "total=6.0  count=3", "tm");
    s += rect(380, 154, 220, 74, "bx2") + text(490, 176, "acc_meter", "t") + text(490, 202, "total=0.0  count=0", "tm");
    return svg(640, 236, s, "One class, two independent objects");
  }};

  D.recursion = { caption: "Each call waits for a smaller one until the base case answers. Then the results flow back up.", draw: () => {
    const calls = ["factorial(4)", "factorial(3)", "factorial(2)", "factorial(1)"], ret = [24, 6, 2, 1];
    let s = "";
    calls.forEach((c, i) => {
      const x = 20 + i * 40, y = 8 + i * 50, last = i === 3;
      s += rect(x, y, 220, 36, last ? "ok" : "bx2") + text(x + 110, y + 18, c + (last ? "  base case" : ""), last ? "to" : "t");
      if (!last) s += arrow(x + 40, y + 40, x + 76, y + 50 - 2);
      s += rect(420, y, 60, 36, "acc") + text(450, y + 18, ret[i], "ta");
      if (i) s += arrow(450, y - 2, 450, y - 12);
      s += text(500, y + 18, last ? "returns 1" : (4 - i) + " × " + ret[i + 1], "tm", "start");
    });
    s += text(150, 216, "calls go down", "tm") + text(450, 216, "answers come back up", "tm");
    return svg(640, 226, s, "Call stack of factorial(4)");
  }};

  D.bigo = { caption: "How the number of steps grows with input size n. Removing a nested loop moves you from the steep curve to a flat one.", draw: () => {
    const { s: a, X, Y } = axes(70, 14, 440, 210, [1, 20], [0, 100], [1, 5, 10, 15, 20], [0, 25, 50, 75, 100], "n (input size)", "steps");
    let s = a;
    const fs = [["O(n²)", x => x * x, "c1"], ["O(n log n)", x => x * Math.log2(x), "c2"], ["O(n)", x => x, "c3"], ["O(log n)", x => Math.log2(x), "c4"], ["O(1)", () => 1, "c5"]];
    fs.forEach(([n, f, c], i) => {
      const xe = n === "O(n²)" ? 10 : n === "O(n log n)" ? 20 : 20;
      s += curve(x => Math.min(f(x), 100), X, Y, 1, xe, "cv " + c);
      s += line(530, 30 + i * 30, 556, 30 + i * 30, "cv " + c) + text(564, 30 + i * 30, n, "t", "start");
    });
    return svg(660, 270, s, "Growth of common complexity classes");
  }};

  D.binsearch = { caption: "Each step checks the middle and throws away half. 8 items take at most 3 checks, and a million take about 20.", draw: () => {
    const v = [1, 3, 5, 7, 9, 11, 13, 15], steps = [[0, 7, 3], [4, 7, 5], [4, 4, 4]];
    let s = "";
    steps.forEach(([lo, hi, mid], r) => {
      const y = 10 + r * 64;
      s += text(30, y + 18, "step " + (r + 1), "tm", "start");
      v.forEach((x, i) => {
        const X = 110 + i * 62, out = i < lo || i > hi;
        s += rect(X, y, 56, 36, i === mid ? (x === 9 ? "ok" : "acc") : out ? "dim" : "bx2") + text(X + 28, y + 18, x, i === mid ? (x === 9 ? "to" : "ta") : out ? "td" : "t");
      });
      if (r < 2) s += text(110 + lo * 62 + 28, y + 48, "lo", "tm") + text(110 + hi * 62 + 28, y + 48, "hi", "tm");
    });
    s += text(110 + 4 * 62 + 28, 10 + 2 * 64 + 50, "found 9", "to");
    return svg(620, 210, s, "Binary search for 9");
  }};

  // ================= Part 2 =================
  D.argmax = { caption: "argmax returns the position of the largest score, which is the predicted class.", draw: () => {
    const v = [0.1, 0.7, 0.2], { s: a, X, Y } = axes(70, 14, 300, 150, [0, 3], [0, 1], [], [0, 0.5, 1]);
    let s = a;
    v.forEach((p, i) => { s += rect(X(i) + 20, Y(p), 60, Y(0) - Y(p), i === 1 ? "accf" : "bx2", 2) + text(X(i) + 50, Y(0) + 14, "class " + i, "tm") + text(X(i) + 50, Y(p) - 10, p, i === 1 ? "ta" : "tm"); });
    s += arrow(420, 90, 386, 90) + text(430, 90, "argmax = 1", "ta", "start");
    return svg(600, 196, s, "Bar chart of class scores");
  }};
  D.word_counts = { caption: "Lowercase, split, then count each token in a dict.", draw: () => {
    const toks = ["the", "cat", "saw", "the", "dog"];
    let s = rect(120, 6, 400, 32, "bx2") + text(320, 22, "\"The cat saw the dog\"", "t") + arrow(320, 42, 320, 58);
    toks.forEach((t, i) => { s += rect(80 + i * 100, 62, 80, 30, t === "the" ? "acc" : "bx") + text(120 + i * 100, 77, t, t === "the" ? "ta" : "t"); });
    s += arrow(320, 96, 320, 112);
    [["the", 2], ["cat", 1], ["saw", 1], ["dog", 1]].forEach(([k, n], i) => { s += rect(110 + i * 110, 116, 100, 30, k === "the" ? "ok" : "bx2") + text(160 + i * 110, 131, k + ": " + n, k === "the" ? "to" : "t"); });
    return svg(640, 154, s, "Counting tokens");
  }};
  D.vocab = { caption: "Each new token gets the next id. Encoding swaps tokens for their ids.", draw: () => {
    let s = text(20, 20, "tokens", "tm", "start");
    ["a", "b", "a", "c"].forEach((t, i) => { s += rect(110 + i * 70, 6, 60, 30, "bx2") + text(140 + i * 70, 21, t, "t"); });
    s += text(20, 74, "vocab", "tm", "start");
    [["a", 0], ["b", 1], ["c", 2]].forEach(([k, v], i) => { s += rect(110 + i * 94, 60, 84, 30, "acc") + text(152 + i * 94, 75, k + " → " + v, "ta"); });
    s += text(20, 128, "encode", "tm", "start") + text(120, 128, "[\"c\", \"a\", \"a\"]", "t", "start") + arrow(290, 128, 340, 128) + rect(346, 113, 120, 30, "ok") + text(406, 128, "[2, 0, 0]", "to");
    return svg(640, 150, s, "Building a vocabulary");
  }};
  D.dot = { caption: "Multiply matching positions, then add everything up. One number comes out.", draw: () => {
    const a = [1, 2, 3], b = [4, 5, 6];
    let s = text(30, 26, "a", "tm") + text(30, 72, "b", "tm") + text(30, 120, "a·b", "tm");
    a.forEach((x, i) => {
      const X = 70 + i * 110;
      s += rect(X, 10, 80, 32, "bx2") + text(X + 40, 26, x, "t") + rect(X, 56, 80, 32, "bx2") + text(X + 40, 72, b[i], "t");
      s += rect(X, 104, 80, 32, "acc") + text(X + 40, 120, x * b[i], "ta");
      if (i < 2) s += text(X + 95, 120, "+", "t");
    });
    s += text(410, 120, "=", "t") + rect(430, 104, 80, 32, "ok") + text(470, 120, "32", "to");
    return svg(560, 146, s, "Dot product of two vectors");
  }};
  D.mean_std = { caption: "The mean is the center. The standard deviation is the typical distance from it.", draw: () => {
    const v = [2, 4, 4, 4, 5, 5, 7, 9], { s: a, X } = axes(40, 36, 520, 60, [0, 11], [0, 1], [0, 2, 4, 6, 8, 10], []);
    let s = a.replace(/<line[^>]*class="ax"\/>$/, "");
    const cnt = {};
    v.forEach(x => { cnt[x] = (cnt[x] || 0) + 1; s += circle(X(x), 96 - 14 * cnt[x], 7, "dotf"); });
    s = rect(X(3), 24, X(7) - X(3), 72, "band", 4) + s + line(X(5), 22, X(5), 96, "lna") + text(X(5), 12, "mean = 5", "ta");
    s += arrow(X(5), 136, X(7), 136, "ara") + arrow(X(5), 136, X(3), 136, "ara") + text(X(7) + 8, 136, "±1 std = 2", "ta", "start");
    return svg(600, 150, s, "Dots on a number line with mean and one standard deviation band");
  }};
  D.zscore = { caption: "Standardizing shifts the data to center 0 and rescales it to spread 1. The shape stays the same.", draw: () => {
    const raw = [10, 20, 30], z = [-1.22, 0, 1.22];
    const A = axes(40, 20, 240, 10, [0, 40], [0, 1], [0, 10, 20, 30, 40], []), B = axes(360, 20, 240, 10, [-2, 2], [0, 1], [-2, -1, 0, 1, 2], []);
    let s = A.s + B.s + text(160, 6, "raw: mean 20, std 8.2", "tm") + text(480, 6, "standardized: mean 0, std 1", "tm");
    raw.forEach(x => { s += circle(A.X(x), 30, 7, "dotf"); });
    z.forEach(x => { s += circle(B.X(x), 30, 7, "dota"); });
    s += arrow(290, 30, 350, 30);
    return svg(640, 64, s, "Before and after standardization");
  }};
  D.transpose = { caption: "Transposing turns rows into columns, so a (2, 3) matrix becomes (3, 2). Element [i][j] moves to [j][i].", draw: () => {
    const M = [[1, 2, 3], [4, 5, 6]], T = [[1, 4], [2, 5], [3, 6]];
    const hl = (r, c) => (r === 0 && c === 2 ? "acc" : null), hlT = (r, c) => (r === 2 && c === 0 ? "acc" : null);
    let s = grid(40, 30, M, 48, 40, hl) + text(110, 14, "M  (2, 3)", "tm") + arrow(200, 70, 290, 70) + text(245, 56, ".T", "tm");
    s += grid(320, 10, T, 48, 40, hlT) + text(368, 140, "Mᵀ  (3, 2)", "tm") + text(470, 50, "M[0][2] = 3", "ta", "start") + text(470, 74, "→ Mᵀ[2][0] = 3", "ta", "start");
    return svg(640, 150, s, "Transpose of a 2 by 3 matrix");
  }};
  D.matvec = { caption: "Each output number is one row of W dotted with x. W is (out, in), so the result has length out.", draw: () => {
    const W = [[1, 2], [3, 4], [5, 6]];
    let s = grid(40, 20, W, 48, 36, r => (r === 1 ? "acc" : null)) + text(84, 10, "W (3, 2)", "tm");
    s += text(150, 72, "@", "t") + grid(170, 38, [[1], [1]], 48, 36, () => "acc") + text(194, 28, "x (2,)", "tm");
    s += text(240, 72, "=", "t") + grid(262, 20, [[3], [7], [11]], 48, 36, r => (r === 1 ? "ok" : null)) + text(286, 10, "y (3,)", "tm");
    s += text(340, 74, "row 1 · x = 3×1 + 4×1 = 7", "ta", "start");
    return svg(640, 134, s, "Matrix times vector");
  }};
  D.matmul = { caption: "Cell C[i][j] is row i of A dotted with column j of B. The inner sizes must match: (n, k) @ (k, m) → (n, m).", draw: () => {
    const A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]], C = [[19, 22], [43, 50]];
    let s = grid(30, 40, A, 48, 40, r => (r === 1 ? "acc" : null)) + text(76, 26, "A (2, 2)", "tm") + text(140, 80, "@", "t");
    s += grid(160, 40, B, 48, 40, (r, c) => (c === 0 ? "acc" : null)) + text(206, 26, "B (2, 2)", "tm") + text(270, 80, "=", "t");
    s += grid(290, 40, C, 48, 40, (r, c) => (r === 1 && c === 0 ? "ok" : null)) + text(336, 26, "C (2, 2)", "tm");
    s += text(400, 70, "C[1][0] = 3×5 + 4×7", "ta", "start") + text(400, 94, "        = 43", "ta", "start");
    s += text(320, 140, "shapes:  (n, k) @ (k, m) → (n, m)", "tm");
    return svg(640, 152, s, "Matrix multiplication highlighting one cell");
  }};
  D.activations = { caption: "ReLU keeps positives and zeros out negatives. Sigmoid squashes any number into (0, 1), which is why it's used for probabilities.", draw: () => {
    const A = axes(50, 20, 220, 140, [-4, 4], [-1, 4], [-4, 0, 4], [0, 2, 4]), B = axes(370, 20, 220, 140, [-6, 6], [0, 1], [-6, 0, 6], [0, 0.5, 1]);
    let s = A.s + B.s + text(160, 8, "ReLU(x) = max(0, x)", "ta") + text(480, 8, "sigmoid(x) = 1 / (1 + e⁻ˣ)", "ta");
    s += curve(x => Math.max(0, x), A.X, A.Y, -4, 4, "cv c1") + curve(x => 1 / (1 + Math.exp(-x)), B.X, B.Y, -6, 6, "cv c1");
    return svg(640, 186, s, "Plots of ReLU and sigmoid");
  }};
  D.softmax = { caption: "Softmax turns any scores into probabilities that sum to 1. Bigger scores get exponentially more of the share.", draw: () => {
    const z = [1, 2, 3], e = z.map(Math.exp), S = e.reduce((a, b) => a + b), p = e.map(x => x / S);
    const A = axes(50, 26, 200, 120, [0, 3], [0, 3], [], [0, 1, 2, 3]), B = axes(390, 26, 200, 120, [0, 3], [0, 1], [], [0, 0.5, 1]);
    let s = A.s + B.s + text(150, 10, "scores (logits)", "tm") + text(490, 10, "probabilities", "tm");
    z.forEach((v, i) => { s += rect(A.X(i) + 12, A.Y(v), 42, A.Y(0) - A.Y(v), "bx2", 2) + text(A.X(i) + 33, A.Y(v) - 9, v, "tm"); });
    p.forEach((v, i) => { s += rect(B.X(i) + 12, B.Y(v), 42, B.Y(0) - B.Y(v), "accf", 2) + text(B.X(i) + 33, B.Y(v) - 9, v.toFixed(2), "ta"); });
    s += arrow(270, 90, 350, 90) + text(310, 76, "exp, ÷ sum", "tm") + text(490, 172, "sum = 1.00", "ta");
    return svg(640, 184, s, "Scores become probabilities");
  }};
  D.cross_entropy = { caption: "The loss is −log(p of the right class). Confidently wrong (p near 0) costs a lot, and near-certain right costs almost nothing.", draw: () => {
    const { s: a, X, Y } = axes(70, 16, 440, 170, [0, 1], [0, 5], [0, 0.25, 0.5, 0.75, 1], [0, 1, 2, 3, 4, 5], "p (probability of the correct class)", "loss");
    let s = a + curve(x => Math.min(5, -Math.log(x)), X, Y, 0.0068, 1, "cv c1");
    s += circle(X(0.7), Y(-Math.log(0.7)), 5, "dota") + text(X(0.7) + 10, Y(-Math.log(0.7)) - 14, "p = 0.7 → 0.36", "ta", "start");
    s += circle(X(0.05), Y(-Math.log(0.05)), 5, "dotb") + text(X(0.05) + 12, Y(-Math.log(0.05)), "p = 0.05 → 3.0", "tb", "start");
    return svg(560, 236, s, "Plot of negative log");
  }};
  D.metrics = { caption: "Every metric is built from the four cells of the confusion matrix.", draw: () => {
    let s = text(250, 12, "predicted", "tm") + text(190, 34, "1", "tm") + text(310, 34, "0", "tm");
    s += `<text x="40" y="110" class="tm" text-anchor="middle" transform="rotate(-90 40 110)">actual</text>` + text(84, 74, "1", "tm") + text(84, 144, "0", "tm");
    s += rect(130, 46, 116, 60, "ok") + text(188, 68, "TP", "to") + text(188, 88, "caught", "tm");
    s += rect(254, 46, 116, 60, "bad") + text(312, 68, "FN", "tb") + text(312, 88, "missed", "tm");
    s += rect(130, 114, 116, 60, "bad") + text(188, 136, "FP", "tb") + text(188, 156, "false alarm", "tm");
    s += rect(254, 114, 116, 60, "bx2") + text(312, 144, "TN", "t");
    s += text(400, 70, "precision = TP / (TP + FP)", "ta", "start") + text(400, 96, "recall    = TP / (TP + FN)", "ta", "start") + text(400, 130, "accuracy  = (TP + TN) / all", "t", "start");
    return svg(660, 184, s, "Confusion matrix and the metrics built from it");
  }};
  D.linear = { caption: "A Linear layer is a matrix multiply plus a bias. Every input connects to every output through one weight.", draw: () => {
    let s = "";
    const ins = [50, 110], outs = [40, 120];
    ins.forEach((y, i) => outs.forEach(y2 => { s += line(130, y, 330, y2, "lnf"); }));
    ins.forEach((y, i) => { s += circle(110, y, 20, "nodeb") + text(110, y, "x" + i, "t"); });
    outs.forEach((y, j) => { s += circle(350, y, 20, "nodea") + text(350, y, "y" + j, "ta"); });
    s += text(230, 20, "W (2, 2)", "tm") + text(420, 60, "y = W x + b", "ta", "start") + text(420, 88, "W: (out, in)", "tm", "start") + text(420, 110, "b: (out,)", "tm", "start");
    return svg(640, 160, s, "A linear layer as a network");
  }};
  D.scaler = { caption: "fit learns the numbers from training data only. transform reuses those same numbers everywhere.", draw: () => {
    let s = rect(20, 20, 150, 50, "bx2") + text(95, 45, "train data", "t") + arrow(174, 45, 244, 45) + text(209, 32, "fit", "tm");
    s += rect(248, 14, 150, 62, "acc") + text(323, 36, "mean = 1.0", "ta") + text(323, 58, "std = 1.0", "ta");
    s += arrow(323, 80, 323, 116) + text(333, 98, "reused", "tm", "start");
    s += rect(20, 120, 150, 50, "bx2") + text(95, 145, "test data", "t") + arrow(174, 145, 244, 145) + text(209, 132, "transform", "tm");
    s += rect(248, 120, 150, 50, "ok") + text(323, 145, "(x − 1.0) / 1.0", "to");
    s += text(430, 70, "never fit on test:", "tb", "start") + text(430, 92, "that's leakage", "tb", "start");
    return svg(640, 180, s, "Fit on train, transform test");
  }};
  D.mse = { caption: "MSE averages the squared gaps between predictions and targets. The gradient points from target to prediction.", draw: () => {
    const { s: a, X, Y } = axes(60, 14, 360, 160, [0, 5], [0, 10], [0, 1, 2, 3, 4, 5], [0, 5, 10]);
    let s = a + line(X(0), Y(1), X(5), Y(9), "cv c1");
    [[1, 4.5], [2, 3.5], [3, 7.5], [4, 8]].forEach(([x, y]) => { const yh = 1 + 1.6 * x; s += line(X(x), Y(y), X(x), Y(yh), "lnb") + circle(X(x), Y(y), 5, "dotf"); });
    s += text(440, 50, "line: predictions ŷ", "ta", "start") + text(440, 76, "dots: targets y", "t", "start") + text(440, 102, "red: errors ŷ − y", "tb", "start") + text(440, 136, "MSE = mean(error²)", "t", "start");
    return svg(640, 200, s, "Regression line with error bars");
  }};
  D.gd = { caption: "Gradient descent steps downhill. Each step is the learning rate times the slope, so steps shrink as the bowl flattens.", draw: () => {
    const { s: a, X, Y } = axes(60, 10, 420, 170, [-3, 3], [0, 9], [-3, -2, -1, 0, 1, 2, 3], [0, 3, 6, 9], "w", "loss");
    let s = a + curve(x => x * x, X, Y, -3, 3, "cv c1");
    let w = -2.8, pts = [];
    for (let i = 0; i < 6; i++) { pts.push(w); w = w - 0.3 * 2 * w; }
    pts.forEach((p, i) => { if (i) s += arrow(X(pts[i - 1]), Y(pts[i - 1] ** 2), X(p), Y(p ** 2), "ara"); s += circle(X(p), Y(p ** 2), 5, "dota"); });
    s += text(500, 40, "w ← w − lr · ∂loss/∂w", "ta", "start") + text(500, 66, "lr = 0.3", "tm", "start");
    return svg(680, 220, s, "Gradient descent on a parabola");
  }};
  D.logreg = { caption: "Logistic regression learns a straight line that splits the classes. Sigmoid turns distance from the line into a probability.", draw: () => {
    const { s: a, X, Y } = axes(40, 10, 300, 200, [-3, 3], [-3, 3], [], []);
    let s = a + path(`M${X(-3)},${Y(3)} L${X(3)},${Y(-3)} L${X(3)},${Y(3)} Z`, "regionA");
    s += line(X(-3), Y(3), X(3), Y(-3), "cv c1");
    [[2, 1], [1, 2], [1.5, 0.3], [2.4, 2.2], [0.6, 1.4]].forEach(([x, y]) => { s += circle(X(x), Y(y), 6, "dota"); });
    [[-1, -2], [-2, -1], [-0.5, -1.8], [-2.3, -0.4], [-1.5, -2.5]].forEach(([x, y]) => { s += circle(X(x), Y(y), 6, "dotf"); });
    s += text(380, 50, "decision boundary: w·x + b = 0", "ta", "start") + text(380, 80, "p = sigmoid(w·x + b)", "t", "start") + text(380, 110, "gradient = (p − y) · x", "t", "start");
    return svg(660, 220, s, "Two classes separated by a line");
  }};
  D.knn = { caption: "kNN finds the k closest training points and takes a majority vote.", draw: () => {
    const { s: a, X, Y } = axes(40, 10, 300, 200, [0, 6], [0, 6], [], []);
    let s = a;
    const A = [[1, 1], [1.5, 2], [2, 1.2], [0.8, 2.6]], B = [[4.5, 4.5], [5, 3.8], [4, 5.2], [3.4, 3.5]], q = [3.6, 3.0];
    s += circle(X(q[0]), Y(q[1]), 70, "ring");
    A.forEach(([x, y]) => { s += circle(X(x), Y(y), 6, "dotf"); });
    B.forEach(([x, y]) => { s += circle(X(x), Y(y), 6, "dota"); });
    s += path(`M${X(q[0]) - 8},${Y(q[1])} L${X(q[0])},${Y(q[1]) - 8} L${X(q[0]) + 8},${Y(q[1])} L${X(q[0])},${Y(q[1]) + 8} Z`, "query");
    s += text(380, 60, "◆ new point", "t", "start") + text(380, 90, "k = 3 nearest: all blue", "t", "start") + text(380, 120, "prediction: blue", "ta", "start");
    return svg(660, 220, s, "k nearest neighbors vote");
  }};
  D.kmeans = { caption: "k-means repeats two steps: assign each point to its nearest center, then move each center to the mean of its points.", draw: () => {
    const A = axes(30, 24, 250, 170, [-2, 14], [-2, 14], [], []), B = axes(380, 24, 250, 170, [-2, 14], [-2, 14], [], []);
    const pts = [[0, 0], [0, 2], [1, 1], [10, 10], [10, 12], [11, 11]];
    let s = A.s + B.s + text(155, 8, "1. assign", "ta") + text(505, 8, "2. move centers", "ta");
    pts.forEach(([x, y], i) => { const c = i < 3 ? "dotf" : "dota"; s += circle(A.X(x), A.Y(y), 6, c) + circle(B.X(x), B.Y(y), 6, c); });
    const cOld = [[4, 6], [7, 4]], cNew = [[0.33, 1], [10.33, 11]];
    cOld.forEach(([x, y], i) => { s += path(`M${A.X(x) - 7},${A.Y(y) - 7} L${A.X(x) + 7},${A.Y(y) + 7} M${A.X(x) + 7},${A.Y(y) - 7} L${A.X(x) - 7},${A.Y(y) + 7}`, "cross"); });
    cNew.forEach(([x, y], i) => { s += arrow(B.X(cOld[i][0]), B.Y(cOld[i][1]), B.X(x), B.Y(y), "arm"); s += path(`M${B.X(x) - 7},${B.Y(y) - 7} L${B.X(x) + 7},${B.Y(y) + 7} M${B.X(x) + 7},${B.Y(y) - 7} L${B.X(x) - 7},${B.Y(y) + 7}`, "cross"); });
    s += arrow(290, 105, 370, 105);
    return svg(640, 208, s, "Two steps of k-means");
  }};
  D.batches = { caption: "Slice the data into fixed-size chunks. The last batch holds whatever is left over.", draw: () => {
    let s = "";
    [1, 2, 3, 4, 5, 6, 7].forEach((x, i) => { const b = Math.floor(i / 3); s += rect(40 + i * 70 + b * 16, 20, 60, 36, ["bx2", "acc", "ok"][b]) + text(70 + i * 70 + b * 16, 38, x, ["t", "ta", "to"][b]); });
    s += text(140, 76, "batch 0", "tm") + text(366, 76, "batch 1", "tm") + text(556, 76, "batch 2 (smaller)", "tm");
    s += text(320, 104, "range(0, len(data), batch_size)  →  0, 3, 6", "tm");
    return svg(660, 116, s, "Seven items in batches of three");
  }};
  const attnPipe = (shapes) => {
    let s = rect(10, 74, 70, 36, "bx2") + text(45, 92, "x", "t") + text(45, 124, shapes.x, "tm");
    [["Q", 20], ["K", 92], ["V", 164]].forEach(([n, y]) => { s += arrow(84, 92, 134, y + 18) + rect(138, y, 60, 36, "acc") + text(168, y + 18, n, "ta"); });
    s += text(168, 214, shapes.qkv, "tm");
    s += arrow(202, 38, 252, 70) + arrow(202, 110, 252, 84) + rect(256, 54, 120, 46, "bx2") + text(316, 70, "Q Kᵀ / √d", "t") + text(316, 88, shapes.s, "tm");
    s += arrow(380, 77, 410, 77) + rect(414, 54, 86, 46, "bx2") + text(457, 70, "softmax", "t") + text(457, 88, "rows sum 1", "tm");
    s += arrow(504, 77, 534, 120) + arrow(202, 182, 534, 140) + rect(538, 110, 124, 46, "ok") + text(600, 126, "weights @ V", "to") + text(600, 144, shapes.out, "tm");
    return s;
  };
  D.attention = { caption: "Every token makes a query, compares it to every key, and takes a weighted average of the values. Follow the shapes.", draw: () =>
    svg(670, 226, attnPipe({ x: "(T, d)", qkv: "each (T, d)", s: "(T, T)", out: "(T, d)" }), "Attention data flow") };
  D.causal = { caption: "The causal mask keeps the lower triangle. Token i can see tokens 0..i, never the future.", draw: () => {
    const T = 5, M = [...Array(T)].map((_, r) => [...Array(T)].map((_, c) => (c <= r ? "✓" : "−∞")));
    let s = grid(80, 30, M, 46, 36, (r, c) => (c <= r ? "acc" : "dim")) + text(195, 14, "keys (positions) →", "tm");
    s += `<text x="60" y="120" class="tm" text-anchor="middle" transform="rotate(-90 60 120)">queries →</text>`;
    s += text(340, 60, "row = the token doing the looking", "t", "start") + text(340, 88, "−∞ before softmax → weight 0", "t", "start") + text(340, 116, "mask = tril(ones(T, T))", "ta", "start");
    return svg(640, 216, s, "Causal mask lower triangle");
  }};

  // ================= Part 3: NumPy =================
  D.np_arrays = { caption: "axis=0 runs down the rows and gives one result per column. axis=1 runs across a row and gives one result per row.", draw: () => {
    const X = [[0, 1, 2], [3, 4, 5]];
    let s = text(250, 12, "X  shape (2, 3)", "tm") + grid(170, 40, X, 56, 40);
    s += arrow(170, 30, 330, 30, "arm") + text(160, 30, "axis=1 →", "tm", "end");
    s += arrow(150, 44, 150, 114, "ara") + text(140, 80, "axis=0 ↓", "ta", "end");
    s += grid(170, 130, [[3, 5, 7]], 56, 40, () => "acc") + text(350, 150, "sum(axis=0) → (3,)", "ta", "start");
    s += arrow(346, 80, 396, 80) + grid(406, 40, [[3], [12]], 56, 40, () => "ok") + text(476, 60, "sum(axis=1)", "to", "start") + text(476, 82, "→ (2,)", "to", "start");
    return svg(640, 180, s, "Summing along each axis");
  }};
  D.np_masks = { caption: "A comparison makes a True/False array. Indexing with it keeps the True positions. Fancy indexing picks one entry per row.", draw: () => {
    const x = [3, -1, 4, -2, 5];
    let s = text(70, 22, "x", "tm", "end") + text(70, 62, "x > 0", "tm", "end") + text(70, 102, "x[x > 0]", "tm", "end");
    x.forEach((v, i) => { s += rect(90 + i * 56, 6, 50, 32, "bx2") + text(115 + i * 56, 22, v, "t") + rect(90 + i * 56, 46, 50, 32, v > 0 ? "acc" : "dim") + text(115 + i * 56, 62, v > 0 ? "T" : "F", v > 0 ? "ta" : "td"); });
    [3, 4, 5].forEach((v, i) => { s += rect(90 + i * 56, 86, 50, 32, "ok") + text(115 + i * 56, 102, v, "to"); });
    const P = [[0.1, 0.9], [0.8, 0.2]];
    s += grid(420, 20, P, 56, 36, (r, c) => ([1, 0][r] === c ? "acc" : null)) + text(476, 104, "probs[[0,1], [1,0]]", "tm") + text(476, 124, "→ [0.9, 0.8]", "ta");
    return svg(640, 134, s, "Boolean masks and fancy indexing");
  }};
  D.np_broadcast = { caption: "Line shapes up from the right. A size-1 dimension (or a missing one) gets stretched to match.", draw: () => {
    const row = (ox, y, label, cells, cls) => { let s = text(ox + 140, y + 15, label, "tm", "end"); cells.forEach((c, i) => { const d = c === "·"; s += rect(ox + 160 + i * 64, y, 56, 30, d ? "dim" : cls) + text(ox + 188 + i * 64, y + 15, c, d ? "td" : cls === "ok" ? "to" : cls === "bad" ? "tb" : "ta"); }); return s; };
    let s = text(170, 12, "works", "to") + row(0, 26, "X", ["32", "10"], "acc") + row(0, 60, "mean(axis=0)", ["·", "10"], "acc") + row(0, 94, "result", ["32", "10"], "ok");
    s += text(520, 12, "fails", "tb") + row(340, 26, "X", ["32", "10"], "acc") + row(340, 60, "mean(axis=1)", ["·", "32"], "bad") + text(596, 110, "10 ≠ 32", "tb");
    s += text(350, 146, "fix: mean(axis=1, keepdims=True) → (32, 1)", "ta");
    return svg(700, 158, s, "Broadcasting compatibility");
  }};
  D.np_softmax = { caption: "Pick the axis, keep its dimension, and every slice along it sums to 1. Subtracting the max changes nothing except preventing overflow.", draw: () => {
    const Z = [[1, 2, 3], [1000, 1001, 1002]], P = [[0.09, 0.24, 0.67], [0.09, 0.24, 0.67]];
    let s = grid(20, 30, Z, 64, 36) + text(116, 14, "Z (2, 3)", "tm") + text(235, 50, "− max", "tm") + arrow(212, 66, 260, 66);
    s += grid(270, 30, [[-2, -1, 0], [-2, -1, 0]], 56, 36) + text(354, 14, "safe to exp", "tm") + arrow(442, 66, 470, 66);
    s += grid(476, 30, P, 56, 36, () => "acc") + text(560, 14, "softmax(axis=-1)", "tm") + text(560, 120, "each row sums to 1", "ta");
    return svg(660, 132, s, "Stable softmax over rows");
  }};
  D.np_onehot = { caption: "np.eye(C)[labels] picks rows of the identity matrix: row i is the one-hot vector for class i.", draw: () => {
    const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    let s = grid(40, 30, I, 42, 34, r => (r === 2 || r === 0 ? "acc" : null)) + text(102, 14, "np.eye(3)", "tm");
    s += text(200, 64, "[2, 0]", "t") + arrow(240, 64, 290, 64);
    s += grid(300, 30, [[0, 0, 1], [1, 0, 0]], 42, 34, () => "ok") + text(362, 14, "one_hot (2, 3)", "tm") + text(450, 64, "label 2, label 0", "tm", "start");
    return svg(640, 140, s, "One-hot from the identity matrix");
  }};
  D.np_pairwise = { caption: "‖a − b‖² = ‖a‖² + ‖b‖² − 2 a·b. A column plus a row broadcasts to the full (n, m) grid in one shot.", draw: () => {
    let s = grid(40, 50, [["‖a₀‖²"], ["‖a₁‖²"]], 70, 36, () => "acc") + text(74, 34, "(n, 1)", "tm") + text(130, 86, "+", "t");
    s += grid(150, 68, [["‖b₀‖²", "‖b₁‖²", "‖b₂‖²"]], 70, 36, () => "acc") + text(254, 52, "(1, m)", "tm") + text(370, 86, "−", "t");
    s += rect(390, 64, 90, 40, "bx2") + text(435, 84, "2 A Bᵀ", "t") + text(435, 120, "(n, m)", "tm") + text(500, 86, "=", "t");
    s += grid(520, 50, [["", "", ""], ["", "", ""]], 36, 36, () => "ok") + text(572, 34, "D² (n, m)", "tm");
    return svg(660, 136, s, "Pairwise distance trick");
  }};
  D.np_linreg = { caption: "With matrices the whole gradient is one expression. Check every shape: Xᵀ (D, N) @ err (N,) → (D,).", draw: () => {
    let s = rect(20, 30, 90, 120, "bx2") + text(65, 90, "X", "t") + text(65, 166, "(N, D)", "tm") + text(126, 90, "@", "t");
    s += rect(140, 60, 30, 60, "acc") + text(155, 90, "w", "ta") + text(155, 136, "(D,)", "tm") + text(196, 90, "−  y  =", "t");
    s += rect(240, 30, 30, 120, "bad") + text(255, 90, "err", "tb") + text(255, 166, "(N,)", "tm");
    s += rect(340, 60, 120, 60, "bx2") + text(400, 90, "Xᵀ", "t") + text(400, 136, "(D, N)", "tm") + text(500, 90, "@ err =", "t");
    s += rect(546, 60, 34, 60, "ok") + text(563, 90, "dw", "to") + text(563, 136, "(D,)", "tm");
    return svg(640, 180, s, "Shapes in vectorized linear regression");
  }};
  D.np_attention = { caption: "@ multiplies the last two dimensions and treats the leading ones as a batch. The (T, T) mask broadcasts over the batch.", draw: () =>
    svg(670, 226, attnPipe({ x: "(B, T, d)", qkv: "each (B, T, d)", s: "(B, T, T)", out: "(B, T, d)" }), "Batched attention shapes") };

  // ================= Part 4: pandas =================
  const table = (x, y, cols, rows, hlRow, cw = 76) => {
    let s = "";
    cols.forEach((c, i) => { s += rect(x + i * cw, y, cw - 4, 26, "hdr", 3) + text(x + i * cw + (cw - 4) / 2, y + 13, c, "th"); });
    rows.forEach((r, j) => r.forEach((v, i) => {
      const cls = hlRow && hlRow(j, i) || "bx";
      s += rect(x + i * cw, y + 30 + j * 28, cw - 4, 24, cls, 3) + text(x + i * cw + (cw - 4) / 2, y + 42 + j * 28, v, cls === "bx" ? "t" : cls === "ok" ? "to" : cls === "bad" ? "tb" : cls === "dim" ? "td" : "ta");
    }));
    return s;
  };
  D.pd_frames = { caption: "A boolean mask keeps matching rows, then sort_values orders them. Each step returns a new DataFrame.", draw: () => {
    const rows = [["a", "2.0", "5", "10"], ["b", "10.0", "3", "30"], ["c", "1.0", "100", "100"], ["d", "50.0", "0", "0"]];
    let s = table(10, 10, ["item", "price", "qty", "revenue"], rows, (j) => (j === 3 ? "dim" : null), 70);
    s += arrow(296, 100, 346, 100) + text(321, 86, "filter", "tm") + text(321, 116, "sort", "tm");
    s += table(356, 24, ["item", "revenue"], [["c", "100"], ["b", "30"], ["a", "10"]], (j) => (j < 2 ? "acc" : null), 80);
    s += text(436, 150, ".head(2) → c, b", "ta");
    return svg(640, 160, s, "Filtering and sorting a table");
  }};
  D.pd_missing = { caption: "Drop rows you can't train on (no label). Fill feature gaps from statistics, never from the label.", draw: () => {
    const rows = [["20", "NY", "1"], ["NaN", "SF", "0"], ["40", "None", "NaN"], ["30", "NY", "1"]];
    let s = table(10, 10, ["age", "city", "label"], rows, (j, i) => (rows[j][i] === "NaN" || rows[j][i] === "None" ? "bad" : null), 70);
    s += arrow(226, 80, 276, 80);
    const out = [["20", "NY", "1"], ["25", "SF", "0"], ["30", "NY", "1"]];
    s += table(290, 10, ["age", "city", "label"], out, (j, i) => (j === 1 && i === 0 ? "acc" : null), 70);
    s += text(510, 50, "row 3 dropped:", "tm", "start") + text(510, 70, "no label", "tm", "start") + text(510, 100, "age NaN → 25", "ta", "start") + text(510, 120, "(median)", "tm", "start");
    return svg(640, 150, s, "Handling missing values");
  }};
  D.pd_groupby = { caption: "groupby splits the rows by key, applies a function to each group, and combines the results.", draw: () => {
    let s = text(70, 10, "split", "tm") + text(330, 10, "apply", "tm") + text(560, 10, "combine", "tm");
    const A = [["a", "10"], ["a", "30"], ["a", "20"]], B = [["b", "5"], ["b", "15"]];
    s += table(10, 22, ["user", "amt"], A, () => "acc", 62) + table(10, 142, ["user", "amt"], B, () => "bx2", 62);
    s += arrow(140, 80, 240, 80) + arrow(140, 186, 240, 186);
    s += rect(246, 62, 170, 36, "acc") + text(331, 80, "size 3 · sum 60", "ta") + rect(246, 168, 170, 36, "bx2") + text(331, 186, "size 2 · sum 20", "t");
    s += arrow(420, 80, 480, 110) + arrow(420, 186, 480, 150);
    s += table(484, 92, ["user", "n", "total"], [["a", "3", "60"], ["b", "2", "20"]], () => "ok", 50);
    return svg(640, 230, s, "Split apply combine");
  }};
  D.pd_merge = { caption: "A left join keeps every left row. Aggregate first so each key appears once and rows don't multiply.", draw: () => {
    let s = text(80, 8, "users", "tm") + table(10, 18, ["user"], [["c"], ["a"], ["b"]], null, 70);
    s += text(205, 8, "totals", "tm") + table(130, 18, ["user", "total"], [["a", "30"], ["b", "5"]], () => "acc", 70);
    s += arrow(290, 70, 340, 70) + text(315, 56, "how=left", "tm");
    s += table(350, 18, ["user", "total_spent"], [["c", "0"], ["a", "30"], ["b", "5"]], (j, i) => (j === 0 && i === 1 ? "ok" : null), 100);
    s += text(560, 60, "c had no orders", "tm", "start") + text(560, 80, "NaN → 0", "ta", "start");
    return svg(680, 130, s, "Left join");
  }};
  D.pd_encode = { caption: "One column per training category. A category that appears only in test data gets all zeros instead of crashing.", draw: () => {
    let s = table(10, 10, ["city"], [["NY"], ["SF"], ["LA"]], (j) => (j === 2 ? "bad" : null), 80);
    s += arrow(96, 70, 146, 70);
    s += table(156, 10, ["city_NY", "city_SF"], [["1", "0"], ["0", "1"], ["0", "0"]], (j) => (j === 2 ? "dim" : "acc"), 90);
    s += text(370, 50, "categories from train: NY, SF", "t", "start") + text(370, 80, "LA unseen → [0, 0]", "tb", "start");
    return svg(640, 120, s, "One-hot encoding with an unseen category");
  }};
  D.pd_leak = { caption: "Split first. Learn every statistic from the training side only, then apply those numbers to both sides.", draw: () => {
    let s = text(160, 8, "wrong", "tb") + rect(30, 20, 260, 34, "bad") + text(160, 37, "scale with mean of ALL rows", "tb") + arrow(160, 58, 160, 78, "arb") + rect(30, 82, 125, 34, "bx2") + text(92, 99, "train", "t") + rect(165, 82, 125, 34, "bx2") + text(227, 99, "test", "t") + text(160, 134, "test rows leaked into the mean", "tm");
    s += text(490, 8, "right", "to") + rect(360, 20, 125, 34, "bx2") + text(422, 37, "train", "t") + rect(495, 20, 125, 34, "bx2") + text(557, 37, "test", "t");
    s += arrow(422, 58, 422, 78) + rect(360, 82, 125, 34, "acc") + text(422, 99, "μ, σ from train", "ta");
    s += arrow(489, 99, 540, 99, "ara") + arrow(422, 120, 422, 140, "ara");
    s += rect(500, 82, 120, 34, "ok") + text(560, 99, "scale test", "to") + rect(360, 142, 125, 30, "ok") + text(422, 157, "scale train", "to");
    return svg(640, 180, s, "Leaky versus correct scaling");
  }};
  D.pd_group_split = { caption: "Split by user, not by row, so no user's data ends up on both sides.", draw: () => {
    let s = text(160, 8, "random rows: user b leaks", "tb");
    const R = ["a", "b", "b", "c", "a", "b"], side = [0, 0, 1, 0, 1, 1];
    R.forEach((u, i) => { s += rect(20 + i * 48, 22, 42, 30, side[i] ? "acc" : "bx2") + text(41 + i * 48, 37, u, side[i] ? "ta" : "t"); });
    s += text(80, 68, "train", "tm") + text(230, 68, "test (mixed)", "tm");
    s += text(490, 8, "group split: clean", "to");
    ["a", "a", "c", "b", "b", "b"].forEach((u, i) => { s += rect(350 + i * 48, 22, 42, 30, i >= 3 ? "acc" : "bx2") + text(371 + i * 48, 37, u, i >= 3 ? "ta" : "t"); });
    s += text(420, 68, "train: a, c", "tm") + text(564, 68, "test: b", "tm");
    return svg(660, 80, s, "Random versus group split");
  }};
  D.pd_time = { caption: "shift(1) inside a group looks one row back for the same user. Each feature only uses the past.", draw: () => {
    let s = table(10, 10, ["user", "ts", "amount", "prev_amount"], [["a", "Jan 4 23:00", "20", "0"], ["a", "Jan 5 08:00", "30", "20"], ["b", "Jan 2 12:00", "5", "0"], ["b", "Jan 3 10:00", "7", "5"]], (j, i) => (i === 3 ? "acc" : null), 116);
    s += arrow(354, 52, 384, 76, "ara") + arrow(354, 108, 384, 132, "ara");
    s += text(490, 52, "first event per user → 0", "tm", "start") + text(490, 80, "previous amount →", "ta", "start");
    return svg(700, 150, s, "Lag feature within each user");
  }};
  D.pd_pipeline = { caption: "Keep the fitted numbers in one object so training and prediction use the exact same transform.", draw: () => {
    const st = [["raw CSV", "bx2"], ["drop no label", "bx2"], ["split", "acc"], ["fit on train", "acc"], ["transform", "bx2"], ["X, y arrays", "ok"]];
    let s = "";
    st.forEach(([n, c], i) => { const x = 8 + i * 126; s += rect(x, 30, 112, 44, c) + text(x + 56, 52, n, c === "ok" ? "to" : c === "acc" ? "ta" : "t"); if (i) s += arrow(x - 12, 52, x - 2, 52); });
    s += text(380, 100, "medians · means · stds · category lists live in Prep", "tm");
    return svg(770, 112, s, "Data preparation pipeline");
  }};

  // ================= Part 5: PyTorch =================
  D.pt_tensors = { caption: "Splitting D into heads: view to (B, T, h, D/h), then transpose so each head gets its own (T, D/h) slab.", draw: () => {
    let s = rect(20, 40, 160, 80, "bx2") + text(100, 80, "(B, T, D)", "t") + text(100, 136, "x", "tm");
    s += arrow(184, 80, 236, 80) + text(210, 66, "view", "tm");
    s += rect(240, 40, 160, 80, "bx2");
    [0, 1, 2, 3].forEach(i => { s += line(240 + 40 * (i + 1), 40, 240 + 40 * (i + 1), 120, "lnf"); });
    s += text(320, 80, "(B, T, h, D/h)", "t");
    s += arrow(404, 80, 486, 80) + text(445, 66, "transpose", "tm");
    [0, 1, 2, 3].forEach(i => { s += rect(496 + i * 10, 30 + i * 10, 130, 70, i === 3 ? "acc" : "bx2"); });
    s += text(591, 95, "(B, h, T, D/h)", "ta") + text(591, 148, "one slab per head", "tm");
    return svg(680, 160, s, "Reshaping into heads");
  }};
  D.pt_autograd = { caption: "The forward pass records a graph. backward() walks it in reverse with the chain rule and fills .grad on the leaves.", draw: () => {
    const node = (x, y, l, c = "nodeb") => circle(x, y, 24, c) + text(x, y, l, c === "nodea" ? "ta" : "t");
    let s = node(60, 50, "w", "nodea") + node(60, 140, "x") + node(200, 95, "×") + node(320, 95, "+") + node(320, 185, "b", "nodea") + node(440, 95, "²") + node(560, 95, "loss");
    s += arrow(84, 58, 176, 88) + arrow(84, 132, 176, 102) + arrow(224, 95, 296, 95) + arrow(320, 161, 320, 119) + arrow(344, 95, 416, 95) + arrow(464, 95, 536, 95);
    s += path("M560,68 C520,-4 120,-4 74,24", "back") + text(320, 40, "backward(): ∂loss/∂w → w.grad", "tb");
    s += text(320, 236, "forward →", "tm");
    return svg(640, 246, s, "A computation graph");
  }};
  D.pt_module = { caption: "Layers assigned in __init__ are registered, so parameters() finds them. forward chains them.", draw: () => {
    const L = [["x", "(N, 4)", "bx2"], ["fc1  Linear(4, 8)", "32 + 8 params", "acc"], ["ReLU", "no params", "bx2"], ["fc2  Linear(8, 3)", "24 + 3 params", "acc"], ["logits", "(N, 3)", "ok"]];
    let s = "";
    L.forEach(([n, d, c], i) => { const x = 8 + i * 152; s += rect(x, 30, 138, 50, c) + text(x + 69, 48, n, c === "acc" ? "ta" : c === "ok" ? "to" : "t") + text(x + 69, 66, d, "tm"); if (i) s += arrow(x - 12, 55, x - 2, 55); });
    s += text(380, 104, "count_params = 40 + 27 = 67", "ta");
    return svg(770, 116, s, "An MLP as a chain of layers");
  }};
  D.pt_train = { caption: "Five lines, in this order, every step. Forgetting zero_grad makes gradients pile up.", draw: () => {
    const st = ["1. logits = model(X)", "2. loss = loss_fn(logits, y)", "3. opt.zero_grad()", "4. loss.backward()", "5. opt.step()"];
    const cx = 380, cy = 122, R = 92;
    let s = circle(cx, cy, R, "ringf") + text(cx, cy - 8, "one", "tm") + text(cx, cy + 10, "training step", "tm");
    st.forEach((t, i) => {
      const a = -Math.PI / 2 + i * 2 * Math.PI / 5, x = cx + R * Math.cos(a), y = cy + R * Math.sin(a);
      s += circle(x, y, 9, i === 2 ? "dotb" : "dota");
      const right = Math.cos(a) > 0.1, left = Math.cos(a) < -0.1;
      s += text(x + (right ? 18 : left ? -18 : 0), y + (Math.abs(Math.cos(a)) < 0.1 ? -22 : 0), t, i === 2 ? "tb" : "t", right ? "start" : left ? "end" : "middle");
    });
    return svg(760, 240, s, "The training loop cycle");
  }};
  D.pt_data = { caption: "Dataset returns one example. DataLoader shuffles indexes, groups them, and stacks the examples into batch tensors.", draw: () => {
    let s = rect(10, 30, 170, 80, "bx2") + text(95, 52, "Dataset", "t") + text(95, 74, "__len__ → 3", "tm") + text(95, 94, "__getitem__(i)", "tm");
    s += arrow(184, 70, 234, 70);
    ["[2, 3, 0, 0]", "[4, 4, 4, 4]"].forEach((v, i) => { s += rect(238, 32 + i * 40, 130, 32, "acc") + text(303, 48 + i * 40, v, "ta"); });
    s += text(303, 122, "items (max_len,)", "tm") + arrow(372, 70, 422, 70) + text(397, 56, "stack", "tm");
    s += rect(426, 34, 200, 72, "ok") + text(526, 58, "xb  (batch, max_len)", "to") + text(526, 84, "yb  (batch,)", "to");
    return svg(640, 134, s, "Dataset to DataLoader batches");
  }};
  D.pt_eval = { caption: "They do different jobs. eval() changes how layers behave, and no_grad() stops recording for backprop. Inference wants both.", draw: () => {
    let s = rect(20, 20, 280, 110, "acc") + text(160, 44, "model.eval()", "ta") + text(160, 72, "Dropout: off", "t") + text(160, 96, "BatchNorm: running stats", "t");
    s += rect(340, 20, 280, 110, "bx2") + text(480, 44, "with torch.no_grad():", "t") + text(480, 72, "no graph recorded", "t") + text(480, 96, "less memory, faster", "t");
    return svg(640, 140, s, "eval versus no_grad");
  }};
  D.pt_selfattn = { caption: "The same pipeline as before, now with learned projections. Q, K, V and out are each nn.Linear(D, D).", draw: () =>
    svg(670, 226, attnPipe({ x: "(B, T, D)", qkv: "Linear(D, D) each", s: "(B, T, T)", out: "→ out Linear" }), "Self-attention module") };
  D.pt_mha = { caption: "Split into h heads, run attention on each in parallel, then merge back and mix with one output Linear.", draw: () => {
    let s = rect(10, 80, 80, 40, "bx2") + text(50, 100, "x", "t") + text(50, 134, "(B,T,D)", "tm");
    for (let i = 0; i < 4; i++) { const y = 10 + i * 46; s += arrow(94, 100, 148, y + 18) + rect(152, y, 160, 36, i ? "bx2" : "acc") + text(232, y + 18, "head " + i + ": attn on D/h", i ? "t" : "ta") + arrow(316, y + 18, 370, 100); }
    s += rect(374, 76, 100, 48, "bx2") + text(424, 94, "merge", "t") + text(424, 112, "(B,T,D)", "tm") + arrow(478, 100, 518, 100) + rect(522, 76, 104, 48, "ok") + text(574, 100, "out Linear", "to");
    return svg(640, 200, s, "Multi-head attention");
  }};
  D.pt_block = { caption: "Pre-norm block: normalize, transform, add back. The residual path (the long arrows) lets gradients skip straight through.", draw: () => {
    let s = "";
    const xs = [20, 170, 320, 470], y = 110;
    s += rect(10, y - 20, 60, 40, "bx2") + text(40, y, "x", "t");
    s += rect(110, 60, 90, 34, "bx2") + text(155, 77, "LayerNorm", "t") + rect(220, 60, 110, 34, "acc") + text(275, 77, "attention", "ta");
    s += circle(370, y, 16, "nodeb") + text(370, y, "+", "t");
    s += rect(410, 60, 90, 34, "bx2") + text(455, 77, "LayerNorm", "t") + rect(520, 60, 80, 34, "acc") + text(560, 77, "MLP", "ta");
    s += circle(620, y, 16, "nodeb") + text(620, y, "+", "t");
    s += arrow(70, y, 354, y, "arm") + arrow(386, y, 604, y, "arm");
    s += arrow(90, y, 110, 77) + arrow(200, 77, 218, 77) + arrow(332, 77, 362, 98) + arrow(386, y, 408, 77) + arrow(500, 77, 518, 77) + arrow(602, 77, 614, 96);
    s += text(212, 150, "x = x + attn(ln1(x))", "tm") + text(500, 150, "x = x + mlp(ln2(x))", "tm");
    return svg(660, 164, s, "Transformer block");
  }};
  return D;
})();
