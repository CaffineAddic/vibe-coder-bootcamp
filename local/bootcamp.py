#!/usr/bin/env python3
"""
Vibe Coder Bootcamp, local edition.

Serves the course on http://127.0.0.1:<port>, runs your code in this machine's
Python (real NumPy, pandas, and real PyTorch if it's installed), and saves your
progress and code to a file so it survives browser resets.

Only the Python standard library is needed to run the server itself.

    python3 local/bootcamp.py            # start and open the browser
    python3 local/bootcamp.py --no-open  # start without opening a browser
    python3 local/bootcamp.py --port 8765 --torchlite
    python3 local/bootcamp.py --stop     # stop a server started from the app menu

Progress file: $XDG_DATA_HOME/vibe-coder-bootcamp/progress.json
               (usually ~/.local/share/vibe-coder-bootcamp/progress.json)
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "vibe-coder-bootcamp"
PROGRESS = DATA / "progress.json"
PIDFILE = DATA / "server.pid"
MARKER = "__VCB_RESULTS_7f3a__"
RUN_TIMEOUT = 30
MAX_BODY = 2_000_000

# torchlite: the course's PyTorch-compatible teaching library, packaged as `torch`.
NN_NAMES = ["Module", "Linear", "Embedding", "LayerNorm", "Dropout", "ReLU", "Sigmoid", "Tanh", "Softmax",
            "Sequential", "ModuleList", "Parameter", "CrossEntropyLoss", "MSELoss", "BCELoss", "BCEWithLogitsLoss"]
F_NAMES = ["relu", "sigmoid", "tanh", "softmax", "log_softmax", "cross_entropy", "mse_loss",
           "binary_cross_entropy", "binary_cross_entropy_with_logits"]


def build_torchlite(dest: Path) -> Path:
    src = (ROOT / "course" / "torchlite.py").read_text()
    files = {
        "torch/__init__.py": src,
        "torch/nn/__init__.py": "from torch import " + ", ".join(NN_NAMES) + "\nfrom torch.nn import functional\n",
        "torch/nn/functional.py": "from torch import F as _F\n" + "\n".join(f"{n} = _F.{n}" for n in F_NAMES) + "\n",
        "torch/optim.py": "from torch import SGD, Adam, Optimizer\n",
        "torch/utils/__init__.py": "from torch.utils import data\n",
        "torch/utils/data.py": "from torch import Dataset, DataLoader, TensorDataset\n",
    }
    for rel, text in files.items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return dest


def has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


class State:
    python = sys.executable
    torch_mode = "torchlite"  # or "real"
    torchlite_dir: Path = None
    workdir: Path = None
    lock = threading.Lock()


def info() -> dict:
    versions = {}
    for m in ("numpy", "pandas", "torch"):
        if has(m):
            try:
                versions[m] = __import__(m).__version__
            except Exception:
                versions[m] = "installed"
    return {
        "local": True,
        "python": sys.version.split()[0],
        "packages": versions,
        "torch": State.torch_mode,
        "progress_file": str(PROGRESS),
    }


def run_code(code: str) -> dict:
    """Run learner code in a fresh Python process with a time limit."""
    runner = (
        "import json as _vcb_json, sys as _vcb_sys\n"
        "_vcb_src = open(_vcb_sys.argv[1]).read()\n"
        "_vcb_g = {'__name__': '__main__'}\n"
        "try:\n"
        "    exec(compile(_vcb_src, 'solution.py', 'exec'), _vcb_g)\n"
        "finally:\n"
        "    _r = _vcb_g.get('_results')\n"
        "    if _r is not None:\n"
        "        _vcb_sys.stdout.flush()\n"
        f"        print('\\n{MARKER}' + _vcb_json.dumps([[bool(a), str(b)] for a, b in _r]))\n"
    )
    with tempfile.TemporaryDirectory(dir=State.workdir) as tmp:
        sol = Path(tmp) / "solution.py"
        sol.write_text(code)
        drv = Path(tmp) / "_driver.py"
        drv.write_text(runner)
        env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
        if State.torch_mode == "torchlite":
            env["PYTHONPATH"] = str(State.torchlite_dir) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            p = subprocess.run([State.python, "-X", "utf8", str(drv), str(sol)], cwd=tmp, env=env,
                               capture_output=True, text=True, timeout=RUN_TIMEOUT)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            return {"ok": False, "out": out, "error": f"TimeLimitError: your code ran for more than {RUN_TIMEOUT} seconds and was stopped."}
    out, results = p.stdout, None
    if MARKER in out:
        out, _, tail = out.rpartition(MARKER)
        try:
            results = json.loads(tail.strip())
        except ValueError:
            results = None
        out = out[:-1] if out.endswith("\n") else out
    if p.returncode != 0:
        err = p.stderr.strip()
        # hide the driver's own frame so the traceback points at solution.py
        lines = err.splitlines()
        first = next((i for i, l in enumerate(lines) if 'File "solution.py"' in l), None)
        if first is not None:
            lines = ["Traceback (most recent call last):"] + lines[first:]
        return {"ok": False, "out": out, "error": "\n".join(lines), "results": results}
    if p.stderr.strip():
        out += ("\n" if out and not out.endswith("\n") else "") + p.stderr
    return {"ok": True, "out": out, "results": results}


def load_progress() -> dict:
    try:
        return json.loads(PROGRESS.read_text())
    except (OSError, ValueError):
        return {}


def save_progress(data: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    if PROGRESS.exists():
        shutil.copy2(PROGRESS, PROGRESS.with_suffix(".json.bak"))
    tmp.replace(PROGRESS)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT), **k)

    def log_message(self, fmt, *args):
        if os.environ.get("VCB_VERBOSE"):
            super().log_message(fmt, *args)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _same_origin(self) -> bool:
        # Refuse API calls from other websites open in your browser.
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        return origin is None or origin in (f"http://{host}",)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path == "/api/info":
            return self._json(info())
        if self.path == "/api/progress":
            return self._json(load_progress())
        if self.path.startswith("/api/"):
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        return super().do_GET()

    def do_POST(self):
        if not self._same_origin():
            return self._json({"error": "forbidden"}, HTTPStatus.FORBIDDEN)
        try:
            body = self._body()
        except ValueError as e:
            return self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
        if self.path == "/api/run":
            return self._json(run_code(str(body.get("code", ""))))
        if self.path == "/api/progress":
            if not isinstance(body, dict):
                return self._json({"error": "expected an object"}, HTTPStatus.BAD_REQUEST)
            with State.lock:
                save_progress({k: v for k, v in body.items() if isinstance(k, str) and k.startswith("vcb.")})
            return self._json({"ok": True})
        return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def main():
    ap = argparse.ArgumentParser(description="Run Vibe Coder Bootcamp locally.")
    ap.add_argument("--port", type=int, default=int(os.environ.get("VCB_PORT", 8765)))
    ap.add_argument("--no-open", action="store_true", help="don't open a browser tab")
    ap.add_argument("--torchlite", action="store_true", help="use the course's torchlite even if real PyTorch is installed")
    ap.add_argument("--stop", action="store_true", help="stop a running bootcamp server")
    args = ap.parse_args()

    if args.stop:
        try:
            pid = int(PIDFILE.read_text())
            os.kill(pid, 15)
            print(f"Stopped the bootcamp server (pid {pid}).")
        except (OSError, ValueError):
            print("No running bootcamp server found.")
        PIDFILE.unlink(missing_ok=True)
        return

    State.workdir = Path(tempfile.mkdtemp(prefix="vcb-"))
    State.torchlite_dir = build_torchlite(State.workdir / "torchlite")
    State.torch_mode = "real" if (has("torch") and not args.torchlite) else "torchlite"
    missing = [m for m in ("numpy", "pandas") if not has(m)]

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        url = f"http://127.0.0.1:{args.port}/"
        print(f"Port {args.port} is busy. If the bootcamp is already running, it's at {url}")
        if not args.no_open:
            webbrowser.open(url)
        sys.exit(0)

    import signal
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    DATA.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Vibe Coder Bootcamp running at {url}")
    print(f"  Python {sys.version.split()[0]} ({sys.executable})")
    print(f"  PyTorch lessons use: {'real PyTorch' if State.torch_mode == 'real' else 'torchlite'}")
    print(f"  Progress saved to {PROGRESS}")
    if missing:
        print(f"  Missing: {', '.join(missing)}. Lessons that need them will error. Run local/install.sh to set them up.")
    print("Press Ctrl+C to stop.")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Your progress is saved.")
    finally:
        shutil.rmtree(State.workdir, ignore_errors=True)
        PIDFILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
