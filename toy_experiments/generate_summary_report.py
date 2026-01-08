"""
Generate comprehensive summary report comparing all density models.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_results(results_dir: str) -> Dict:
    """Load results from JSON file."""
    results_path = Path(results_dir) / "results_summary.json"
    if not results_path.exists():
        return None

    with open(results_path, 'r') as f:
        return json.load(f)


def create_markdown_table(data: List[List[str]], headers: List[str]) -> str:
    """Create a markdown table."""
    # Header
    table = "| " + " | ".join(headers) + " |\n"
    table += "|" + "|".join(["---" for _ in headers]) + "|\n"

    # Rows
    for row in data:
        table += "| " + " | ".join(str(cell) for cell in row) + " |\n"

    return table


def generate_summary_report():
    """Generate comprehensive summary report."""
    report_lines = []

    report_lines.append("# Toy Density Model Comparison - Summary Report")
    report_lines.append("")
    report_lines.append("This report compares the performance of different density models")
    report_lines.append("on synthetic mixture of Gaussians datasets with varying dimensionalities.")
    report_lines.append("")
    report_lines.append("## Models Compared")
    report_lines.append("")
    report_lines.append("1. **RealNVP**: Normalizing flow with coupling layers")
    report_lines.append("2. **VAE**: Variational Autoencoder")
    report_lines.append("3. **KDE**: Kernel Density Estimation")
    report_lines.append("4. **Neural ODE**: Continuous normalizing flow")
    report_lines.append("5. **Diffusion**: DDIM denoising diffusion probabilistic model")
    report_lines.append("")

    # Process each dimension
    dimensions = [2, 4, 8]

    for dim in dimensions:
        results_dir = f"toy_experiments/results/dim{dim}"
        results = load_results(results_dir)

        if results is None:
            report_lines.append(f"\n## {dim}D Results")
            report_lines.append(f"**Status**: Not available (experiments may not have completed)")
            continue

        report_lines.append(f"\n## {dim}D Results")
        report_lines.append("")

        # Loglikelihood statistics
        report_lines.append("### Log-Likelihood Statistics")
        report_lines.append("")

        # Create table for each split
        splits = ['test', 'ood_easy', 'ood_medium', 'ood_hard', 'ood_very_hard']

        for split in splits:
            report_lines.append(f"\n#### {split.replace('_', ' ').title()}")
            report_lines.append("")

            headers = ["Model", "Mean", "Std", "Min", "Max", "Median"]
            data = []

            for model_name, model_results in results.items():
                stats = model_results['score_statistics'].get(split, {})
                if stats:
                    data.append([
                        model_name,
                        f"{stats['mean']:.3f}",
                        f"{stats['std']:.3f}",
                        f"{stats['min']:.3f}",
                        f"{stats['max']:.3f}",
                        f"{stats['median']:.3f}",
                    ])

            if data:
                report_lines.append(create_markdown_table(data, headers))

        # OOD Detection Performance
        report_lines.append("\n### OOD Detection Performance")
        report_lines.append("")

        ood_levels = ['easy', 'medium', 'hard', 'very_hard']

        # AUROC table
        report_lines.append("#### AUROC (Area Under ROC Curve)")
        report_lines.append("")

        headers = ["Model"] + [level.replace('_', ' ').title() for level in ood_levels]
        data = []

        for model_name, model_results in results.items():
            row = [model_name]
            for level in ood_levels:
                auroc = model_results['metrics'].get(f'{level}_auroc', 0)
                row.append(f"{auroc:.4f}")
            data.append(row)

        report_lines.append(create_markdown_table(data, headers))

        # AUPRC table
        report_lines.append("\n#### AUPRC (Area Under Precision-Recall Curve)")
        report_lines.append("")

        data = []
        for model_name, model_results in results.items():
            row = [model_name]
            for level in ood_levels:
                auprc = model_results['metrics'].get(f'{level}_auprc', 0)
                row.append(f"{auprc:.4f}")
            data.append(row)

        report_lines.append(create_markdown_table(data, headers))

        # Correlation with true log-probs
        report_lines.append("\n### Correlation with True Log-Probabilities")
        report_lines.append("")

        headers = ["Model", "Test", "OOD Easy", "OOD Medium", "OOD Hard", "OOD Very Hard"]
        data = []

        for model_name, model_results in results.items():
            row = [model_name]
            for split in splits:
                corr_key = f'{split}_correlation'
                corr = model_results['metrics'].get(corr_key, np.nan)
                if np.isnan(corr):
                    row.append("N/A")
                else:
                    row.append(f"{corr:.4f}")
            data.append(row)

        report_lines.append(create_markdown_table(data, headers))

        # Add visualization references
        report_lines.append(f"\n### Visualizations")
        report_lines.append("")
        report_lines.append(f"See `toy_experiments/results/dim{dim}/` for:")
        report_lines.append(f"- `loglikelihood_comparison.png`")
        report_lines.append(f"- `ood_detection_performance.png`")
        report_lines.append(f"- `score_distributions.png`")
        report_lines.append("")

    # Key findings
    report_lines.append("\n## Key Findings & Interpretation")
    report_lines.append("")
    report_lines.append("### Log-Likelihood Scale Comparison")
    report_lines.append("")
    report_lines.append("The models produce different scales of log-likelihoods:")
    report_lines.append("")
    report_lines.append("- **RealNVP**: Direct log-probability under normalizing flow")
    report_lines.append("  - Scale: Typically negative values (e.g., -5 to -50)")
    report_lines.append("  - Interpretation: Exact log-probability of data under learned distribution")
    report_lines.append("")
    report_lines.append("- **VAE**: Negative reconstruction error")
    report_lines.append("  - Scale: Typically negative values (e.g., -1 to -20)")
    report_lines.append("  - Interpretation: NOT a true likelihood, but reconstruction quality")
    report_lines.append("  - Note: ELBO provides a lower bound on true log-likelihood")
    report_lines.append("")
    report_lines.append("- **KDE**: Kernel density estimation with Gaussian kernels")
    report_lines.append("  - Scale: Log-probability from kernel density estimate")
    report_lines.append("  - Interpretation: Non-parametric density estimation")
    report_lines.append("")
    report_lines.append("- **Neural ODE**: Log-probability via continuous normalizing flow")
    report_lines.append("  - Scale: Similar to RealNVP (negative values)")
    report_lines.append("  - Interpretation: Exact log-probability using ODE integration")
    report_lines.append("  - Note: May be more expensive computationally")
    report_lines.append("")
    report_lines.append("- **Diffusion**: ELBO-based log-likelihood from diffusion model")
    report_lines.append("  - Scale: Typically negative values (e.g., -10 to -100)")
    report_lines.append("  - Interpretation: Evidence Lower Bound on log-likelihood via reverse diffusion")
    report_lines.append("  - Note: Uses DDIM scheduler for faster inference")
    report_lines.append("")
    report_lines.append("### Why Scales Differ")
    report_lines.append("")
    report_lines.append("1. **Different objectives**:")
    report_lines.append("   - RealNVP/Neural ODE maximize log-likelihood directly")
    report_lines.append("   - VAE/Diffusion maximize ELBO (Evidence Lower Bound)")
    report_lines.append("   - KDE uses non-parametric density estimation")
    report_lines.append("")
    report_lines.append("2. **Different architectures**:")
    report_lines.append("   - RealNVP uses invertible coupling transformations")
    report_lines.append("   - VAE has bottleneck latent space with encoder-decoder")
    report_lines.append("   - KDE uses Gaussian kernels around training points")
    report_lines.append("   - Neural ODE uses continuous-time dynamics via ODE integration")
    report_lines.append("   - Diffusion uses iterative denoising process with DDIM scheduler")
    report_lines.append("")
    report_lines.append("3. **Normalization constants**:")
    report_lines.append("   - Each model may have different implicit normalization")
    report_lines.append("   - Absolute values are less important than relative ordering")
    report_lines.append("")
    report_lines.append("### Practical Implications for GORMPO")
    report_lines.append("")
    report_lines.append("When using these models for OOD penalty in GORMPO:")
    report_lines.append("")
    report_lines.append("1. **Normalization is crucial**:")
    report_lines.append("   - Normalize scores to [0, 1] range before applying penalty")
    report_lines.append("   - Use percentile-based thresholds instead of absolute values")
    report_lines.append("")
    report_lines.append("2. **Model selection**:")
    report_lines.append("   - RealNVP: Best for direct likelihood estimation, moderate training time")
    report_lines.append("   - VAE: Fastest training, good for high-dimensional data")
    report_lines.append("   - KDE: No training needed, but memory-intensive for large datasets")
    report_lines.append("   - Neural ODE: Most flexible, but computationally expensive")
    report_lines.append("   - Diffusion: High-quality density estimation, slower inference")
    report_lines.append("")
    report_lines.append("3. **Penalty coefficient tuning**:")
    report_lines.append("   - Different models will require different `reward-penalty-coef` values")
    report_lines.append("   - Start with small values (0.001-0.01) and tune based on performance")
    report_lines.append("   - Monitor the distribution of penalty values during training")
    report_lines.append("")
    report_lines.append("## Recommendations")
    report_lines.append("")
    report_lines.append("Based on these experiments:")
    report_lines.append("")
    report_lines.append("1. **For RL applications**: Use **RealNVP** or **Neural ODE** for true likelihood")
    report_lines.append("2. **For fast prototyping**: Use **VAE** with normalized scores")
    report_lines.append("3. **For baseline comparison**: Use **KDE** (no training required)")
    report_lines.append("4. **For highest quality**: Use **Diffusion** or **Neural ODE** (accept slower inference)")
    report_lines.append("5. **Always normalize**: Convert raw scores to standardized range before penalty")
    report_lines.append("6. **Validate empirically**: Test on actual RL environment to tune penalty scale")
    report_lines.append("")

    # Save report
    report_path = Path("toy_experiments/SUMMARY_REPORT.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Summary report generated: {report_path}")

    # Also print to console
    print("\n" + "="*80)
    for line in report_lines[:50]:  # Print first 50 lines
        print(line)
    print("\n... (see full report in toy_experiments/SUMMARY_REPORT.md)")
    print("="*80)


if __name__ == "__main__":
    generate_summary_report()
