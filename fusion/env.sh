# Source this to run the fusion scripts with the repository's bundled site-packages.
# The .venv interpreter symlink is dangling on this machine, so a system Python 3.10
# is used with the venv's site-packages on PYTHONPATH; a stdlib extension missing
# from that build (_bz2, needed by torchvision) is supplied through a shim dir.
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/lib/python3.10/site-packages:/tmp/claude-1003/-media-bisg-SSD-8TB-github-polesight-polesight-dataset/7895a749-180d-43f2-a3f9-ffe4528ed802/scratchpad/shim"
export PY=/usr/local/bin/python3.10
export WANDB_MODE=disabled
