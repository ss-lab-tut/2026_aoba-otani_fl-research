from pathlib import Path
from datetime import datetime
import json
import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_run_dir(tag: str) -> Path:
    base = Path("results")
    base.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{timestamp}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
