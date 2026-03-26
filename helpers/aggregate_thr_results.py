"""
Aggregate per-run threshold sensitivity results into seed-averaged statistics.

Usage:
    python helpers/aggregate_thr_results.py <individual_csv> <aggregated_csv>

individual_csv columns expected: seed, threshold_percentile, mean_return, density_model, ...
aggregated_csv columns produced: threshold_percentile, mean_reward, std_reward, ci95_reward, n_seeds
"""

import sys
import os
import numpy as np
import pandas as pd


def aggregate(individual_path: str, aggregated_path: str) -> None:
    df = pd.read_csv(individual_path)

    if 'mean_return' not in df.columns:
        raise ValueError(f"Expected 'mean_return' column in {individual_path}. Found: {list(df.columns)}")

    if 'threshold_percentile' not in df.columns:
        raise ValueError(f"Expected 'threshold_percentile' column in {individual_path}.")

    grouped = df.groupby('threshold_percentile')['mean_return']

    rows = []
    for thr, group in grouped:
        values = group.dropna().values
        n = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(n) if n > 0 else 0.0
        rows.append({
            'threshold_percentile': thr,
            'mean_reward': mean,
            'std_reward': std,
            'ci95_reward': ci95,
            'n_seeds': n,
        })

    agg_df = pd.DataFrame(rows).sort_values('threshold_percentile').reset_index(drop=True)

    os.makedirs(os.path.dirname(aggregated_path), exist_ok=True)
    agg_df.to_csv(aggregated_path, index=False)
    print(f"Aggregated results saved to: {aggregated_path}")
    print(agg_df.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python helpers/aggregate_thr_results.py <individual_csv> <aggregated_csv>")
        sys.exit(1)
    aggregate(sys.argv[1], sys.argv[2])
