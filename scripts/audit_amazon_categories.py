"""Run the experiment package's preliminary Electronics category audit."""

import argparse
from pathlib import Path

from cp4285.experiment.category_audit import run_audit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    run_audit(options.reviews, options.label_root, options.output)
