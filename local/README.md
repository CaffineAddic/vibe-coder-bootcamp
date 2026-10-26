# Local edition

Run the bootcamp on your own machine. It works offline, runs your code in real
Python (real NumPy, pandas and optionally real PyTorch), and saves your progress
and code to a file, so clearing your browser or switching browsers loses nothing.

## Install (Arch Linux, or any Linux)

```sh
sudo pacman -S --needed python     # Arch only, if you don't have Python yet
./local/install.sh                 # NumPy + pandas; PyTorch lessons use torchlite
./local/install.sh --torch         # same, plus real PyTorch (CPU build, ~200 MB)
```

Then start it with `vibe-bootcamp`, or open **Vibe Coder Bootcamp** from your app menu.
It opens at http://127.0.0.1:8765/ and only listens on your own machine.

If `vibe-bootcamp` isn't found, add `~/.local/bin` to your PATH:

```sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc   # or ~/.zshrc
```

## Commands

| Command | What it does |
|---|---|
| `vibe-bootcamp` | start the server and open the browser |
| `vibe-bootcamp --no-open` | start without opening a browser |
| `vibe-bootcamp --stop` | stop a server started from the app menu |
| `vibe-bootcamp --port 9000` | use a different port |
| `vibe-bootcamp --torchlite` | use torchlite even if real PyTorch is installed |
| `./local/install.sh` | update after pulling new lessons (progress is kept) |
| `./local/install.sh --uninstall` | remove everything except your progress |

## Your progress

Saved to `~/.local/share/vibe-coder-bootcamp/progress.json` every time you solve a
lesson, rate a card or type in the editor. The previous version is kept as
`progress.json.bak`. Back it up or copy it to another machine like any other file.

## Without installing

Python's standard library is enough to run the server:

```sh
python3 local/bootcamp.py
```

Lessons that need NumPy or pandas will use whatever your `python3` has installed.

## How it works

`bootcamp.py` serves the course and adds three endpoints on 127.0.0.1:
`/api/info`, `/api/run` (runs code in a fresh Python process with a 30-second limit)
and `/api/progress`. The page detects the local server and switches to it
automatically. The same page on GitHub Pages keeps using Pyodide in the browser.
Requests from other websites are refused.
