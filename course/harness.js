// Test helpers appended after the learner's code. check() records a result; close() compares floats.
const HARNESS = "\n_results = []\ndef check(cond, msg):\n    _results.append((bool(cond), msg))\ndef close(a, b, tol=1e-5):\n    return abs(a - b) < tol\n";
