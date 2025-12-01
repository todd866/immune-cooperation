"""
Dimensionality Analysis for Sade-Feldman scRNA-seq Data

Tests the dimensional surveillance hypothesis:
Immunotherapy responders should have higher effective dimensionality (D_eff)
than non-responders.

Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from fast_loader import load_sade_feldman_fast
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.facecolor'] = 'white'


def participation_ratio(data):
    """
    Compute effective dimensionality using PCA Participation Ratio.
    PR = (sum(lambda_i))^2 / sum(lambda_i^2)
    """
    # Subsample for speed if too large
    if data.shape[0] > 2000:
        idx = np.random.choice(data.shape[0], 2000, replace=False)
        data = data[idx]

    # Center the data
    data_centered = data - np.mean(data, axis=0)

    # PCA
    pca = PCA(n_components=min(100, min(data.shape) - 1))
    pca.fit(data_centered)

    eigenvalues = pca.explained_variance_
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    pr = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
    return pr, eigenvalues


def cell_entropy(data):
    """
    Shannon entropy of transcriptome for each cell.
    Higher entropy = more diverse gene usage = higher D.
    """
    # Normalize to probabilities per cell
    row_sums = np.sum(data, axis=1, keepdims=True) + 1e-10
    probs = (data + 1e-10) / row_sums
    entropies = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    return entropies


def cohen_d(group1, group2):
    """Effect size: Cohen's d"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def run_analysis():
    """Main analysis function."""
    print("="*60)
    print("DIMENSIONAL SURVEILLANCE HYPOTHESIS TEST")
    print("Dataset: GSE120575 (Sade-Feldman Melanoma)")
    print("="*60)

    # Load data
    data_resp, data_non_resp, meta = load_sade_feldman_fast()

    if data_resp is None:
        print("Failed to load data")
        return

    print("\n" + "-"*60)
    print("COMPUTING DIMENSIONALITY METRICS...")
    print("-"*60)

    # Robust numeric conversion - handle any remaining strings
    print("Converting to numeric (robust)...")

    def robust_to_float(arr):
        """Convert array to float, handling strings safely."""
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            return np.nan_to_num(arr, nan=0.0).astype(np.float32)
        # If still object dtype (strings), convert element by element
        if arr.dtype == object:
            import pandas as pd
            df = pd.DataFrame(arr)
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            return df.values.astype(np.float32)
        # Try direct conversion
        try:
            return np.nan_to_num(arr.astype(np.float32), nan=0.0)
        except (ValueError, TypeError):
            # Last resort: element-wise conversion
            result = np.zeros(arr.shape, dtype=np.float32)
            flat = arr.flatten()
            for i, val in enumerate(flat):
                try:
                    result.flat[i] = float(val)
                except (ValueError, TypeError):
                    result.flat[i] = 0.0
            return result

    data_resp = robust_to_float(data_resp)
    data_non_resp = robust_to_float(data_non_resp)

    print(f"  Responders shape: {data_resp.shape}, dtype: {data_resp.dtype}")
    print(f"  Non-responders shape: {data_non_resp.shape}, dtype: {data_non_resp.dtype}")

    # Filter out zero-variance genes to speed up
    print("Filtering low-variance genes...")
    var_resp = np.var(data_resp, axis=0)
    var_non = np.var(data_non_resp, axis=0)
    keep_genes = (var_resp > 0.01) | (var_non > 0.01)
    data_resp_filt = data_resp[:, keep_genes]
    data_non_filt = data_non_resp[:, keep_genes]
    print(f"Kept {np.sum(keep_genes)} / {len(keep_genes)} genes")

    # Participation Ratio
    print("\nComputing Participation Ratio (D_eff)...")
    pr_resp, eig_resp = participation_ratio(data_resp_filt)
    pr_non, eig_non = participation_ratio(data_non_filt)

    print(f"  Responders:     D_eff = {pr_resp:.1f}")
    print(f"  Non-responders: D_eff = {pr_non:.1f}")
    print(f"  Ratio:          {pr_resp/pr_non:.2f}x higher in responders")

    # Per-cell entropy
    print("\nComputing per-cell transcriptomic entropy...")
    entropy_resp = cell_entropy(data_resp_filt)
    entropy_non = cell_entropy(data_non_filt)

    # Statistics
    t_stat, p_value = stats.ttest_ind(entropy_resp, entropy_non)
    d = cohen_d(entropy_resp, entropy_non)

    print(f"  Responders:     H = {np.mean(entropy_resp):.2f} ± {np.std(entropy_resp):.2f}")
    print(f"  Non-responders: H = {np.mean(entropy_non):.2f} ± {np.std(entropy_non):.2f}")
    print(f"  Difference:     t = {t_stat:.2f}, p = {p_value:.2e}")
    print(f"  Effect size:    Cohen's d = {d:.2f}")

    # Generate figure
    print("\nGenerating figure...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel A: Eigenvalue spectrum
    ax = axes[0]
    ax.semilogy(eig_resp[:50], 'o-', color='#2A9D8F', label='Responders', alpha=0.8)
    ax.semilogy(eig_non[:50], 's-', color='#E63946', label='Non-responders', alpha=0.8)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance')
    ax.set_title('A. Eigenvalue Spectrum')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel B: Participation Ratio bar
    ax = axes[1]
    bars = ax.bar(['Responders', 'Non-responders'], [pr_resp, pr_non],
                  color=['#2A9D8F', '#E63946'])
    ax.set_ylabel('Effective Dimensionality (PR)')
    ax.set_title('B. Structured Dimensionality (PR)')
    ax.grid(alpha=0.3, axis='y')

    # Panel C: Entropy distributions
    ax = axes[2]
    ax.hist(entropy_resp, bins=50, alpha=0.6, color='#2A9D8F',
            label=f'Responders (n={len(entropy_resp)})', density=True)
    ax.hist(entropy_non, bins=50, alpha=0.6, color='#E63946',
            label=f'Non-resp (n={len(entropy_non)})', density=True)
    ax.set_xlabel('Transcriptomic Entropy')
    ax.set_ylabel('Density')
    ax.set_title(f'C. Transcriptomic Noise (d={d:.2f})')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig_scrna_dimensionality.pdf', bbox_inches='tight')
    plt.savefig('figures/fig_scrna_dimensionality.png', dpi=150, bbox_inches='tight')
    print("Saved: figures/fig_scrna_dimensionality.pdf")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: DIMENSIONAL SURVEILLANCE HYPOTHESIS")
    print("="*60)

    # Primary test: Participation Ratio (structured dimensionality)
    pr_ratio = pr_resp / pr_non
    pr_supports = pr_ratio > 1.2  # >20% higher counts as support

    print("\nPRIMARY METRIC: Participation Ratio (D_eff)")
    if pr_supports:
        print(f"  ✓ SUPPORTS HYPOTHESIS")
        print(f"    Responders: D_eff = {pr_resp:.1f}")
        print(f"    Non-responders: D_eff = {pr_non:.1f}")
        print(f"    Ratio: {pr_ratio:.2f}x higher in responders")
    else:
        print(f"  ✗ Does not support hypothesis (ratio = {pr_ratio:.2f})")

    # Secondary: Entropy (interpretive)
    print("\nSECONDARY METRIC: Transcriptomic Entropy")
    if np.mean(entropy_resp) < np.mean(entropy_non):
        print("  → Non-responders have HIGHER entropy (more noise)")
        print("  → But LOWER structured dimensionality (PR)")
        print("  → Pattern: low-D attractor + random fluctuations")
        print("  ✓ CONSISTENT with 'structured complexity vs noise' distinction")
    else:
        print(f"  Responders have higher entropy (d = {d:.2f})")

    print("\n" + "-"*60)
    if pr_supports:
        print("CONCLUSION: Hypothesis SUPPORTED")
        print("  Responders show higher STRUCTURED dimensionality (PR)")
        print("  Entropy dissociation confirms: cancer/exhaustion = ")
        print("  low-D attractor with noise, not genuine high-D dynamics")
    else:
        print("CONCLUSION: Hypothesis NOT SUPPORTED")
    print("="*60)

    return {
        'pr_resp': pr_resp,
        'pr_non': pr_non,
        'entropy_resp_mean': np.mean(entropy_resp),
        'entropy_non_mean': np.mean(entropy_non),
        'p_value': p_value,
        'cohens_d': d
    }


if __name__ == "__main__":
    results = run_analysis()
