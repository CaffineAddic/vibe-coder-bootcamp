// Checks every lesson: each reference solution must pass its own tests, and
// each lesson example must run without errors.
//  - Plain-Python lessons run in Skulpt, the browser fallback interpreter.
//  - Every lesson also runs in CPython 3 (python3 on PATH, with numpy and pandas),
//    which matches Pyodide, the main in-browser runtime.
//  - With REAL_TORCH=1, the PyTorch lessons run on the real torch package instead of
//    the course's torchlite (requires `pip install torch`).
// Usage: npm install && npm test        (or: REAL_TORCH=1 npm test)
const fs = require("fs");
const os = require("os");
const path = require("path");
const { execFileSync } = require("child_process");
const { torchFiles, torchDataJs } = require("../tools/torch-files.js");
global.window = global;
require("skulpt/dist/skulpt.min.js");
require("skulpt/dist/skulpt-stdlib.js");

const root = path.join(__dirname, "..");
const load = (f, name) => eval(fs.readFileSync(path.join(root, "course", f), "utf8") + ";" + name);
const all = [
  ...load("foundations.js", "FOUNDATION_LESSONS"), ...load("ml.js", "ML_LESSONS"),
  ...load("numpy.js", "NUMPY_LESSONS"), ...load("data.js", "DATA_LESSONS"), ...load("pytorch.js", "PYTORCH_LESSONS"),
];
const HELP = load("harness.js", "HARNESS");

// Write the torch teaching package to a temp dir for CPython.
const pkgDir = fs.mkdtempSync(path.join(os.tmpdir(), "vcb-"));
for (const [f, c] of Object.entries(torchFiles(root))) {
  fs.mkdirSync(path.dirname(path.join(pkgDir, f)), { recursive: true });
  fs.writeFileSync(path.join(pkgDir, f), c);
}

function skulpt(src) {
  let out = "";
  Sk.configure({
    output: t => (out += t),
    read: f => { if (Sk.builtinFiles && Sk.builtinFiles.files[f] !== undefined) return Sk.builtinFiles.files[f]; throw new Error("File not found: " + f); },
    __future__: Sk.python3, execLimit: 20000,
  });
  return Sk.misceval.asyncToPromise(() => Sk.importMainWithBody("<stdin>", false, src, true));
}

const REAL_TORCH = process.env.REAL_TORCH === "1";
if (REAL_TORCH) fs.rmSync(path.join(pkgDir, "torch"), { recursive: true, force: true });

function cpython(src) {
  const file = path.join(pkgDir, "run.py");
  fs.writeFileSync(file, src + "\nimport json as _j\nprint('__RESULTS__' + _j.dumps(globals().get('_results', [])))\n");
  const out = execFileSync("python3", [file], { env: { ...process.env, PYTHONPATH: pkgDir }, encoding: "utf8", timeout: 60000 });
  return JSON.parse(out.split("__RESULTS__").pop());
}

(async () => {
  let failed = 0;
  if (fs.readFileSync(path.join(root, "course", "torchlite.js"), "utf8") !== torchDataJs(root)) {
    console.log("course/torchlite.js is out of date. Run: node build.js"); failed++;
  }
  const ids = new Set();
  for (const l of all) {
    if (ids.has(l.id)) { console.log("DUPLICATE ID", l.id); failed++; }
    ids.add(l.id);
    const problems = [];
    try {
      const r = cpython(l.solution + "\n" + HELP + "\n" + l.tests);
      r.filter(([ok]) => !ok).forEach(b => problems.push("cpython: " + b[1]));
      if (l.example) cpython(l.example);
    } catch (e) { problems.push("cpython error: " + String(e.stderr || e).trim().split("\n").slice(-3).join(" | ")); }
    if (!l.packages) {
      try {
        const m = await skulpt(l.solution + "\n" + HELP + "\n" + l.tests);
        Sk.ffi.remapToJs(m.tp$getattr(new Sk.builtin.str("_results"))).filter(([ok]) => !ok).forEach(b => problems.push("skulpt: " + b[1]));
        if (l.example) await skulpt(l.example);
      } catch (e) { problems.push("skulpt error: " + String(e)); }
    }
    if (problems.length) { failed++; console.log("FAIL", l.id, problems); } else console.log("ok  ", l.id);
  }
  fs.rmSync(pkgDir, { recursive: true, force: true });
  console.log(failed ? `\n${failed} problem(s)` : `\nAll ${all.length} lessons pass.`);
  process.exit(failed ? 1 : 0);
})();
