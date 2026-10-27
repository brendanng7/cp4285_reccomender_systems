"""Assign prepared training queries to eight consecutive periods."""

import argparse
from pathlib import Path

from cp4285.experiment.category_periods import run_preparation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_preparation(args.sequence_dir, args.output)
