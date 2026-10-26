# Vibe Coder Bootcamp

**A programming course for vibe coders who want to write code on their own again.**

If AI autocomplete has been doing your typing and you froze the last time someone asked you to write code on a whiteboard, this is for you. It starts at variables and loops and ends with you implementing scaled dot-product attention with a causal mask, from scratch, with no libraries.

Everything runs in the browser. There's nothing to install, no account, and no AI help. Write code, press **Run tests**, and see exactly which checks fail.

## What's inside

**Part 1 · Foundations (13 lessons).** Each lesson explains one idea, shows a runnable example, then gives you an exercise.
Variables and types · functions and return · if/elif/else · loops · lists and slicing · strings · dicts and sets · comprehensions · errors and exceptions · classes · recursion · Big-O · binary search

**Part 2 · ML coding (23 exercises).** The from-scratch implementations ML interviews ask for, in plain Python lists.
argmax · tokenizing and vocab · dot product · mean/std · standardization · transpose · matvec · matmul · ReLU/sigmoid · stable softmax · cross-entropy · precision/recall · a `Linear` layer class · a fit/transform scaler · MSE and its gradient · gradient descent · logistic regression · kNN · k-means · mini-batches · attention · causal attention

**Concept cards (37 cards).** Interview theory questions on Python, CS basics, ML fundamentals, training, and transformers. Answer out loud, reveal, and rate yourself. Cards you miss come back first.

Every exercise has:
- a short "why this matters in interviews" note
- tests that run your code and tell you what's wrong
- pseudocode, progressive hints, and a solution that's hidden until you say you've tried
- a 20-minute interview timer and a "when you freeze" checklist

Progress and your code are saved in your browser's local storage.

## Why plain Python and no NumPy?

If you can write `matmul` and `softmax` with loops, the NumPy and PyTorch versions are easy. Interviewers often ask for the loop version on purpose, because it shows you understand the math rather than the API.

## Run it

**Online:** enable GitHub Pages on this repo (Settings → Pages → deploy from the `main` branch, root folder) and open the URL.

**Locally:** open `dist/vibe-coder-bootcamp.html` in a browser. It's a single self-contained file. It needs an internet connection the first time to load the Python interpreter from the jsDelivr CDN.

To serve the multi-file version locally:

```sh
python3 -m http.server 8000
# open http://localhost:8000
```

## Suggested two-week plan

| Days | Work |
|---|---|
| 1–3 | Part 1, lessons 1–8 |
| 4–5 | Part 1, lessons 9–13 |
| 6–8 | Part 2 through Core ML functions |
| 9–11 | Part 2 through Attention |
| 12–14 | Redo attention, matmul, softmax and the training loop from a blank file with the timer on. Concept cards every day. |

Rules that make it work: turn off AI autocomplete, struggle for 15 minutes before opening a hint, and when you read a solution, reset and write it again from memory.

## Project layout

```
index.html            page shell
style.css             styles (light and dark)
app.js                course UI, Python runner, concept cards
course/
  foundations.js      Part 1 lessons
  ml.js               Part 2 exercises
  concepts.js         interview concept cards
  harness.js          test helpers (check, close)
tests/
  check-solutions.js  runs every reference solution against its tests
build.js              bundles everything into dist/vibe-coder-bootcamp.html
```

Python runs in the browser through [Skulpt](https://skulpt.org), an open-source Python interpreter written in JavaScript. It supports the core language and `math`, which is all this course needs.

## Development

```sh
npm install
npm test        # every reference solution must pass its own tests
node build.js   # rebuild the single-file version
```

See [CONTRIBUTING.md](CONTRIBUTING.md) to add lessons or cards.

## License

MIT. See [LICENSE](LICENSE).
