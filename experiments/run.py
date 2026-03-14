"""
experiments/run.py

Run a single experiment using one config file.

Usage:
    python experiments/run.py configs/demo.json
"""

# --- Import safety block (prevents import error if executed from subdirectory) ---
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# --- Standard imports ---
from src.main import load_config, make_run_dir, save_json, set_seed


def run_experiment(config: dict) -> dict:
    """
    Replace this function body with your real experiment.

    It must return a dictionary of metrics.

    Example metrics:
        - accuracy
        - loss
        - reward
        - sla_violation
        - latency
    """

    # ----------------------------
    # Example placeholder experiment
    # ----------------------------

    x = 1 + 1  # Replace with real experiment logic

    metrics = {
        "metric_x": x,
        "metric_y": x * 2,
    }

    return metrics


def main():
    # ------------------------------------------------------------
    # Check arguments
    # ------------------------------------------------------------
    if len(sys.argv) != 2:
        print("Usage: python experiments/run.py configs/<config>.json")
        sys.exit(1)

    config_path = sys.argv[1]

    # ------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------
    config = load_config(config_path)

    seed = int(config.get("seed", 0))
    set_seed(seed)

    experiment_name = config.get("experiment_name", "experiment")

    # ------------------------------------------------------------
    # Create run directory
    # ------------------------------------------------------------
    run_dir = make_run_dir(experiment_name)

    # Save config for reproducibility
    save_json(run_dir / "config_used.json", config)

    # ------------------------------------------------------------
    # Run experiment
    # ------------------------------------------------------------
    metrics = run_experiment(config)

    # ------------------------------------------------------------
    # Standardized result structure
    # ------------------------------------------------------------
    result = {
        "experiment_name": experiment_name,
        "group": config.get("group"),
        "variant": config.get("variant"),
        "seed": seed,
        "metrics": metrics,
        "config_path": config_path,
        "run_dir": str(run_dir),
    }

    save_json(run_dir / "metrics.json", result)

    print("--------------------------------------------------")
    print("[OK] Experiment finished:", experiment_name)
    print("[OK] Output saved to:", run_dir)
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
