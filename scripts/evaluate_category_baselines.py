"""Evaluate popularity and last-category baselines on prepared validation queries."""

import argparse
from pathlib import Path

from cp4285.experiment.category_baselines import run_baselines

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_baselines(args.sequence_dir, args.output)
