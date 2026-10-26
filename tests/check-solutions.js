// Runs every reference solution (and every lesson example) through Skulpt,
// the same Python interpreter the course uses in the browser.
// Usage: npm install && npm test
const fs = require("fs");
const path = require("path");
global.window = global;
require("skulpt/dist/skulpt.min.js");
require("skulpt/dist/skulpt-stdlib.js");

const root = path.join(__dirname, "..");
const load = (f, name) => eval(fs.readFileSync(path.join(root, "course", f), "utf8") + ";" + name);
const all = [...load("foundations.js", "FOUNDATION_LESSONS"), ...load("ml.js", "ML_LESSONS")];
const HELP = load("harness.js", "HARNESS");

function run(src) {
  let out = "";
  Sk.configure({
    output: t => (out += t),
    read: f => { if (Sk.builtinFiles && Sk.builtinFiles.files[f] !== undefined) return Sk.builtinFiles.files[f]; throw new Error("File not found: " + f); },
    __future__: Sk.python3, execLimit: 20000,
  });
  return Sk.misceval.asyncToPromise(() => Sk.importMainWithBody("<stdin>", false, src, true)).then(m => ({ m, out }));
}

(async () => {
  let failed = 0;
  const ids = new Set();
  for (const l of all) {
    if (ids.has(l.id)) { console.log("DUPLICATE ID", l.id); failed++; }
    ids.add(l.id);
    try {
      const { m } = await run(l.solution + "\n" + HELP + "\n" + l.tests);
      const r = Sk.ffi.remapToJs(m.tp$getattr(new Sk.builtin.str("_results")));
      const bad = r.filter(([ok]) => !ok);
      if (bad.length) { failed++; console.log("FAIL", l.id, bad.map(b => b[1])); } else console.log("ok  ", l.id);
      if (l.example) await run(l.example);
    } catch (e) { failed++; console.log("ERR ", l.id, String(e)); }
  }
  console.log(failed ? `\n${failed} problem(s)` : `\nAll ${all.length} solutions pass.`);
  process.exit(failed ? 1 : 0);
})();
