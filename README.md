# Vibe Coder Bootcamp

**A programming course for vibe coders who want to write code on their own again.**

If AI autocomplete has been doing your typing and you froze the last time someone asked you to write code on a whiteboard, this is for you. It starts at variables and loops, moves through NumPy, pandas and the data-processing questions ML interviews ask, and ends with you writing multi-head attention and a transformer block in PyTorch.

Everything runs in the browser. There's nothing to install, no account, and no AI help. Write code, press **Run tests**, and see exactly which checks fail.

## What's inside

62 lessons in five parts, plus 53 interview concept cards.

**Part 1 · Foundations (13).** Each lesson explains one idea, shows a runnable example, then gives you an exercise.
Variables and types · functions · if/elif/else · loops · lists and slicing · strings · dicts and sets · comprehensions · errors · classes · recursion · Big-O · binary search

**Part 2 · ML from scratch (23).** The implementations interviews ask for, in plain Python lists.
argmax · tokenizing and vocab · dot product · mean/std · standardization · transpose · matvec · matmul · ReLU/sigmoid · stable softmax · cross-entropy · precision/recall · a `Linear` class · a fit/transform scaler · MSE gradient · gradient descent · logistic regression · kNN · k-means · mini-batches · attention · causal attention

**Part 3 · NumPy (8).** Arrays, shapes and axes · boolean masks and fancy indexing · broadcasting (LayerNorm) · vectorized softmax · one-hot and accuracy · pairwise distances without loops · vectorized gradient descent · batched attention with a causal mask

**Part 4 · pandas and data prep (9).** Select, filter, sort · missing values · groupby and transform · joins without losing or duplicating rows · categorical encoding with unseen categories · split-then-scale to avoid leakage · group and time splits · datetime and lag features · an end-to-end `fit`/`transform` preprocessing class

**Part 5 · PyTorch (9).** Tensors and reshaping into heads · autograd and `no_grad` · `nn.Module` · the training loop · `Dataset` and `DataLoader` · `eval()` vs `no_grad()` · a self-attention module · multi-head attention · a pre-norm transformer block

**Concept cards (53).** Python, CS basics, ML fundamentals, training, NumPy, pandas, data prep, PyTorch and transformers. Answer out loud, reveal, and rate yourself. Cards you miss come back first.

Every exercise has:
- a short "why this matters in interviews" note
- tests that run your code and tell you exactly what's wrong
- pseudocode, progressive hints, and a solution that stays hidden until you say you've tried
- a 20-minute interview timer and a "when you freeze" checklist

Progress and your code are saved in your browser's local storage.

## How the code runs

Your code runs in real Python 3 in the browser through [Pyodide](https://pyodide.org), inside a Web Worker. NumPy and pandas are the real libraries. Code that runs longer than 20 seconds is stopped, so an infinite loop can't freeze the page. The first load takes about 10 seconds.

Real PyTorch can't run in a browser, so Part 5 uses **torchlite** (`course/torchlite.py`): a small NumPy-backed implementation of the PyTorch API with real autograd, covering tensors, `nn.Module`, common layers and losses, `SGD`/`Adam`, and `Dataset`/`DataLoader`. Every Part 5 reference solution is also tested against real PyTorch (`npm run test:real-torch`), so what you write is ordinary PyTorch.

If Pyodide can't load (for example, inside a sandbox that blocks it), Parts 1 and 2 fall back to [Skulpt](https://skulpt.org), and lessons that need libraries show a notice.

## Why plain Python and no NumPy?

If you can write `matmul` and `softmax` with loops, the NumPy and PyTorch versions are easy. Interviewers often ask for the loop version on purpose, because it shows you understand the math rather than the API.

## Run it

**Online:** enable GitHub Pages on this repo (Settings → Pages → deploy from the `main` branch, root folder) and open the URL.

**Locally:** serve the folder and open it. It needs an internet connection to load Python from the jsDelivr CDN.

```sh
python3 -m http.server 8000
# open http://localhost:8000
```

## Suggested two-week plan

| Days | Work |
|---|---|
| 1–2 | Part 1 |
| 3–5 | Part 2 |
| 6–7 | Part 3 (NumPy) |
| 8–9 | Part 4 (pandas and data prep) |
| 10–12 | Part 5 (PyTorch) |
| 13–14 | Redo attention, the training loop and the preprocessing class from a blank file with the timer on |

Do 10 concept cards every day from day 1.

Rules that make it work: turn off AI autocomplete, struggle for 15 minutes before opening a hint, and when you read a solution, reset and write it again from memory.

## Project layout

```
index.html            page shell
style.css             styles (light and dark)
app.js                course UI, Python runner (Pyodide worker + Skulpt fallback), concept cards
course/
  foundations.js      Part 1
  ml.js               Part 2
  numpy.js            Part 3
  data.js             Part 4
  pytorch.js          Part 5
  torchlite.py        the PyTorch-compatible teaching library
  torchlite.js        generated from torchlite.py by build.js
  concepts.js         concept cards
  harness.js          test helpers (check, close)
tools/torch-files.js  packages torchlite.py as an importable `torch` package
tests/
  check-solutions.js  runs every reference solution against its tests
build.js              regenerates torchlite.js and bundles dist/vibe-coder-bootcamp.html
```

Python runs in the browser through [Skulpt](https://skulpt.org), an open-source Python interpreter written in JavaScript. It supports the core language and `math`, which is all this course needs.

## Development

```sh
npm install
npm test                  # every solution passes its tests (needs python3 with numpy and pandas)
npm run test:real-torch   # also checks Part 5 against real PyTorch (needs pip install torch)
npm run build             # regenerate torchlite.js and the single-file build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) to add lessons or cards.

## License

MIT. See [LICENSE](LICENSE).
