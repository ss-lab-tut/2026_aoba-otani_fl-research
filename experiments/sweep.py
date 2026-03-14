"""
experiments/sweep.py

Run all config files under configs/ in alphabetical order.

Usage:
    python experiments/sweep.py
"""

# --- Import safety block (prevents import error if executed from subdirectory) ---
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# --- Standard imports ---
import subprocess


def main():
    config_dir = Path("configs")
    configs = sorted(config_dir.glob("*.json"))

    if not configs:
        print("[ERR] No config files found in configs/")
        sys.exit(1)

    print("[INFO] Num configs:", len(configs))
    for cfg in configs:
        print("--------------------------------------------------")
        print("[RUN]", cfg)
        subprocess.run(["python", "experiments/run.py", str(cfg)], check=True)

    print("--------------------------------------------------")
    print("[OK] sweep finished")


if __name__ == "__main__":
    main()
