"""
experiments/sweep_seeds.py

Run all configs under configs/ multiple times with different seeds.

This script does NOT modify the original config files.
It creates a temporary config with overridden seed and runs experiments/run.py.

Usage:
    python experiments/sweep_seeds.py
"""

# --- Import safety block ---
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# --- Standard imports ---
import subprocess
import json
import tempfile


def run_one_config_with_seed(config_path: Path, seed: int) -> None:
    # Load original config
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    # Override seed
    cfg["seed"] = seed

    # Write to temporary config file (so original file stays unchanged)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        tmp_path = f.name

    # Run one experiment
    subprocess.run(["python", "experiments/run.py", tmp_path], check=True)


def main():
    config_dir = Path("configs")
    configs = sorted(config_dir.glob("*.json"))
    if not configs:
        print("[ERR] No config files found in configs/")
        sys.exit(1)

    # Change seeds here (simple and explicit)
    seeds = [0, 1, 2, 3, 4]

    print("[INFO] Seeds:", seeds)
    print("[INFO] Num configs:", len(configs))

    for cfg_path in configs:
        print("==================================================")
        print("[CONFIG]", cfg_path)
        for s in seeds:
            print("--------------------------------------------------")
            print("[RUN] seed =", s)
            run_one_config_with_seed(cfg_path, s)

    print("==================================================")
    print("[OK] sweep_seeds finished")


if __name__ == "__main__":
    main()
