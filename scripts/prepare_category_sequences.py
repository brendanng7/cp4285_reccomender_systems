"""Construct training histories and isolated next-review validation examples."""

import argparse
from pathlib import Path

from cp4285.experiment.category_sequences import run_preparation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_preparation(args.audit_dir, args.output)
