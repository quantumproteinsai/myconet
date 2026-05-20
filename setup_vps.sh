#!/usr/bin/env bash
# ============================================================
# setup_vps.sh
# MycoNet — VPS installation and reproducibility test
# Run as your normal user (not root) on the OVH VPS
#
# Usage:
#   chmod +x setup_vps.sh
#   ./setup_vps.sh
# ============================================================

set -euo pipefail

echo "=================================================="
echo "  MycoNet VPS setup"
echo "=================================================="

# ── 1. System dependencies ────────────────────────────────
echo ""
echo "==> Checking system packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    screen git build-essential \
    libfreetype6-dev pkg-config

# ── 2. Virtual environment ────────────────────────────────
VENV="$HOME/myconet-env"
echo ""
echo "==> Creating virtual environment at $VENV..."
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip --quiet

# ── 3. Install myconet from PyPI ─────────────────────────
echo ""
echo "==> Installing myconet from PyPI..."
pip install myconet --quiet
echo "    myconet $(pip show myconet | grep Version | awk '{print $2}') installed"

# ── 4. Clone repo (for examples/) ────────────────────────
REPO="$HOME/myconet-repo"
echo ""
echo "==> Cloning repository..."
if [ -d "$REPO" ]; then
    echo "    Repo already exists — pulling latest..."
    cd "$REPO" && git pull --quiet
else
    git clone https://github.com/quantumproteinsai/myconet.git "$REPO" --quiet
fi

# ── 5. Quick start test ──────────────────────────────────
echo ""
echo "==> Running quick start test..."
cd "$REPO"
python3 examples/quickstart.py

# ── 6. Smoke test (unit tests) ───────────────────────────
echo ""
echo "==> Running unit tests..."
pip install pytest --quiet
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -15

# ── 7. Figure 1 ──────────────────────────────────────────
echo ""
echo "==> Generating Figure 1..."
python3 examples/drought_stress.py --save fig1.png
echo "    Figure saved: $REPO/fig1.png"

# ── 8. Instructions ───────────────────────────────────────
echo ""
echo "=================================================="
echo "  Setup complete."
echo "=================================================="
echo ""
echo "Figure 1 is at: $REPO/fig1.png"
echo ""
echo "Download it with:"
echo "  scp user@your-vps-ip:$REPO/fig1.png ."
echo ""
echo "To run the full Fokker-Planck simulation (~5 min):"
echo "  screen -S myconet"
echo "  source $VENV/bin/activate"
echo "  cd $REPO"
echo "  python3 examples/make_fig1.py"
echo "  # Ctrl+A then D to detach, screen -r myconet to reattach"
echo ""
