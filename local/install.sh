#!/usr/bin/env bash
# Installs the local edition of Vibe Coder Bootcamp for the current user.
# Works on Arch Linux (and most other Linux distros). Nothing is installed system-wide.
#
#   ./local/install.sh            # NumPy + pandas; PyTorch lessons use torchlite
#   ./local/install.sh --torch    # also install real PyTorch (CPU build, ~200 MB)
#   ./local/install.sh --uninstall
#
# What it creates:
#   ~/.local/share/vibe-coder-bootcamp/app/       a copy of the course
#   ~/.local/share/vibe-coder-bootcamp/venv/      Python with numpy and pandas
#   ~/.local/share/vibe-coder-bootcamp/progress.json   your progress (kept on uninstall)
#   ~/.local/bin/vibe-bootcamp                    the command that starts it
#   ~/.local/share/applications/vibe-coder-bootcamp.desktop   app-menu entry
set -euo pipefail

DATA="${XDG_DATA_HOME:-$HOME/.local/share}/vibe-coder-bootcamp"
BIN="$HOME/.local/bin"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WITH_TORCH=0

for arg in "$@"; do
  case "$arg" in
    --torch) WITH_TORCH=1 ;;
    --uninstall)
      rm -rf "$DATA/app" "$DATA/venv"
      rm -f "$BIN/vibe-bootcamp" "$APPS/vibe-coder-bootcamp.desktop" "$ICONS/vibe-coder-bootcamp.svg"
      echo "Uninstalled. Your progress is still in $DATA/progress.json"
      exit 0 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

if ! command -v python3 >/dev/null; then
  echo "python3 not found. On Arch: sudo pacman -S python"
  exit 1
fi

echo "==> Copying the course to $DATA/app"
mkdir -p "$DATA"
rm -rf "$DATA/app"
mkdir -p "$DATA/app"
cp -r "$SRC/index.html" "$SRC/style.css" "$SRC/app.js" "$SRC/favicon.svg" "$SRC/course" "$SRC/local" "$DATA/app/"
[ -f "$SRC/og.png" ] && cp "$SRC/og.png" "$DATA/app/"

echo "==> Creating a Python environment in $DATA/venv"
# A venv avoids touching Arch's system Python (pip refuses to install there, PEP 668).
if [ ! -x "$DATA/venv/bin/python" ]; then
  python3 -m venv "$DATA/venv"
fi
"$DATA/venv/bin/python" -m pip install --quiet --upgrade pip
echo "==> Installing NumPy and pandas"
"$DATA/venv/bin/python" -m pip install --quiet numpy pandas
if [ "$WITH_TORCH" = 1 ]; then
  echo "==> Installing PyTorch (CPU build)"
  "$DATA/venv/bin/python" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "==> Installing the vibe-bootcamp command in $BIN"
mkdir -p "$BIN"
cat > "$BIN/vibe-bootcamp" <<EOF
#!/usr/bin/env bash
exec "$DATA/venv/bin/python" "$DATA/app/local/bootcamp.py" "\$@"
EOF
chmod +x "$BIN/vibe-bootcamp"

echo "==> Adding an app-menu entry"
mkdir -p "$APPS" "$ICONS"
cp "$SRC/favicon.svg" "$ICONS/vibe-coder-bootcamp.svg"
cat > "$APPS/vibe-coder-bootcamp.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Vibe Coder Bootcamp
Comment=Practice ML interview coding with real tests
Exec=$BIN/vibe-bootcamp
Icon=vibe-coder-bootcamp
Terminal=false
Categories=Development;Education;
EOF
command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" 2>/dev/null || true

echo
echo "Done."
echo "  Start it:  vibe-bootcamp     (or find 'Vibe Coder Bootcamp' in your app menu)"
echo "  Progress:  $DATA/progress.json"
case ":$PATH:" in
  *":$BIN:"*) ;;
  *) echo "  Note: $BIN isn't on your PATH. Add this to ~/.bashrc or ~/.zshrc:"
     echo "        export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
esac
