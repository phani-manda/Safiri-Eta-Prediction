"""One-command reproduction of the whole Safiri pipeline.

Runs, in dependency order, the exact commands the reports reference:

    1. src/data/generator.py           -> data/raw/shipments.csv
    2. src/features/engineering.py     -> data/processed/features_v1.csv
    3. src/models/train.py             -> models/eta_regressor.joblib
                                           models/delay_classifier.joblib
    4. src/explainability/explainer.py -> importance / contributor / narrative output

Pass ``--tests`` to finish with the full pytest suite:

    python scripts/reproduce.py --tests
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("generator", "src/data/generator.py"),
    ("feature engineering", "src/features/engineering.py"),
    ("training (regressor + classifier)", "src/models/train.py"),
    ("explainability", "src/explainability/explainer.py"),
]


def run_step(name: str, rel_path: str) -> None:
    script = REPO_ROOT / rel_path
    print(f"\n=== {name}  ({rel_path}) ===", flush=True)
    completed = subprocess.run([sys.executable, str(script)], cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"step '{name}' failed with exit code {completed.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tests",
        action="store_true",
        help="run the pytest suite after regenerating data, features and models",
    )
    args = parser.parse_args()

    print(f"Safiri pipeline reproduction\nrepo: {REPO_ROOT}")
    for name, rel_path in STEPS:
        run_step(name, rel_path)

    if args.tests:
        print("\n=== test suite (pytest) ===", flush=True)
        completed = subprocess.run([sys.executable, "-m", "pytest"], cwd=REPO_ROOT)
        if completed.returncode != 0:
            raise SystemExit(f"pytest failed with exit code {completed.returncode}")

    print("\nDone. All artifacts regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())