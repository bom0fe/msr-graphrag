#!/usr/bin/env python
"""Generate plots and summary tables from experiment JSON files."""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.visualize_results import make_all_plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--ablation", default=None,
                    help="Optional ablation result directory.")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    make_all_plots(args.results, args.out, ablation_dir=args.ablation)


if __name__ == "__main__":
    main()
