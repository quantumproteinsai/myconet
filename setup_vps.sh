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

# ── 5. Quick smoke test ───────────────────────────────────
echo ""
echo "==> Running smoke test (unit tests)..."
cd "$REPO"
pip install pytest --quiet
python -m pytest tests/test_freiman.py -v --tb=short 2>&1 | tail -15

# ── 6. Instructions ───────────────────────────────────────
echo ""
echo "=================================================="
echo "  Setup complete."
echo "=================================================="
echo ""
echo "To generate Figure 1 (single run, ~3 min):"
echo ""
echo "  source $VENV/bin/activate"
echo "  cd $REPO"
echo "  python examples/drought_stress.py --save fig1.png"
echo ""
echo "To run the full 10-run ensemble in the background:"
echo ""
echo "  screen -S myconet"
echo "  source $VENV/bin/activate"
echo "  cd $REPO"
echo "  python examples/drought_stress.py --ensemble --save fig1.png"
echo "  # Press Ctrl+A then D to detach"
echo "  # Reconnect later with: screen -r myconet"
echo ""
echo "When done, download the figure:"
echo "  scp user@your-vps-ip:$REPO/fig1.png ."
echo ""
