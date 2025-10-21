#!/usr/bin/env python
"""
Diagnostic script to analyze prototype selection during training.

This script monitors selection patterns to help diagnose issues like:
- Histology prototype collapse (0 selected)
- Excessive gene prototype selection
- Imbalanced selection between modalities

Usage:
    python scripts/analyze_selection.py --results_dir results/BLCA_causal_separation_gumbel_k0
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_results(results_dir):
    """Load training results and dumps."""
    results_file = Path(results_dir) / 'summary.json'

    if not results_file.exists():
        print(f"Error: {results_file} not found")
        return None

    with open(results_file, 'r') as f:
        data = json.load(f)

    return data


def analyze_selection_patterns(results_dir):
    """Analyze prototype selection patterns."""

    print("=" * 80)
    print("Prototype Selection Analysis")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print()

    # Load results
    data = load_results(results_dir)
    if data is None:
        return

    # Check if we have selection information
    test_data = data.get('test', {})

    # Print basic metrics
    print("Test Set Performance:")
    print(f"  C-Index (Causal):   {test_data.get('c_index_causal', 'N/A'):.4f}")
    print(f"  C-Index (Confound): {test_data.get('c_index_confound', 'N/A'):.4f}")
    print(f"  C-Index (Full):     {test_data.get('c_index_full', 'N/A'):.4f}")
    print()

    # Check for selection statistics
    if 'histo_causal_count' in test_data:
        print("Prototype Selection Statistics:")
        print(f"  Avg Histology Causal Prototypes: {test_data.get('histo_causal_count', 0):.2f} / 16")
        print(f"  Avg Gene Causal Prototypes:      {test_data.get('gene_causal_count', 0):.2f} / 50")
        print()

        histo_ratio = test_data.get('histo_causal_count', 0) / 16
        gene_ratio = test_data.get('gene_causal_count', 0) / 50

        print(f"  Histology Selection Ratio: {histo_ratio:.2%}")
        print(f"  Gene Selection Ratio:      {gene_ratio:.2%}")
        print()

        # Diagnose issues
        print("Diagnosis:")
        if histo_ratio < 0.1:
            print("  ⚠️  WARNING: Histology prototype collapse detected!")
            print("      - Almost no histology prototypes are selected as causal")
            print("      - This suggests histology features may not be predictive")
            print("      - Or the selection network needs stronger balance constraints")

        if gene_ratio > 0.7:
            print("  ⚠️  WARNING: Excessive gene prototype selection!")
            print("      - Most gene prototypes are selected")
            print("      - Increase sparsity penalty to reduce selection")

        if abs(histo_ratio - gene_ratio) > 0.3:
            print("  ⚠️  WARNING: Large imbalance between modalities!")
            print("      - Consider increasing balance loss weight")

        # Overfitting check
        train_data = data.get('train', {})
        val_data = data.get('val', {})

        if train_data and test_data:
            train_c = train_data.get('c_index_full', 0)
            test_c = test_data.get('c_index_full', 0)
            gap = train_c - test_c

            print()
            print("Overfitting Analysis:")
            print(f"  Train C-Index: {train_c:.4f}")
            print(f"  Test C-Index:  {test_c:.4f}")
            print(f"  Gap:           {gap:.4f}")

            if gap > 0.15:
                print("  ⚠️  SEVERE OVERFITTING DETECTED!")
                print("      - Increase weight decay")
                print("      - Increase regularization (sparsity, balance)")
                print("      - Consider dropout or other techniques")
            elif gap > 0.08:
                print("  ⚠️  Moderate overfitting detected")
                print("      - Consider stronger regularization")

    print()
    print("=" * 80)


def plot_selection_distribution(results_dir):
    """Plot selection distribution if dump data is available."""
    # TODO: Implement visualization of selection masks across samples
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze prototype selection patterns')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to results directory')

    args = parser.parse_args()

    analyze_selection_patterns(args.results_dir)
