"""
Dimensionality Analysis for scRNA-seq Data

Tests the hypothesis that immunotherapy responders have higher effective
dimensionality (D_eff) than non-responders.

Metrics:
- PCA Participation Ratio: PR = (sum(lambda_i))^2 / sum(lambda_i^2)
- Transcriptomic Entropy: H = -sum(p_i * log(p_i))

For real data, use GSE120575 (Sade-Feldman melanoma checkpoint response)

Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.facecolor'] = 'white'


def calculate_effective_dimensionality(data_matrix):
    """
    Calculates Effective Dimensionality using PCA Participation Ratio.

    PR = (sum(eigenvalues)^2) / sum(eigenvalues^2)

    If variance equally spread across N dims, PR = N.
    If variance all in 1 dim, PR = 1.
    """
    data_centered = data_matrix - np.mean(data_matrix, axis=0)

    pca = PCA()
    pca.fit(data_centered)

    eigenvalues = pca.explained_variance_
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    pr = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
    return pr, eigenvalues


def calculate_cell_entropy(data_matrix):
    """
    Shannon entropy of transcriptome for each cell.
    Higher entropy = more diverse gene usage (High-D).
    """
    row_sums = np.sum(data_matrix, axis=1, keepdims=True) + 1e-10
    data_prob = (data_matrix + 1e-10) / row_sums
    entropies = -np.sum(data_prob * np.log(data_prob), axis=1)
    return entropies


def generate_mock_data(n_cells=500, n_genes=1000):
    """
    Generate synthetic scRNA-seq data for demonstration.

    Responder: High dimensionality (many latent factors)
    Non-responder: Low dimensionality (few latent factors)
    """
    np.random.seed(42)

    # Responder: 80 latent dimensions
    n_latent_high = 80
    latent = np.random.randn(n_cells, n_latent_high)
    mixing = np.random.randn(n_latent_high, n_genes)
    data_responder = np.abs(np.dot(latent, mixing))

    # Non-responder: 5 latent dimensions (collapsed/exhausted)
    n_latent_low = 5
    latent = np.random.randn(n_cells, n_latent_low)
    mixing = np.random.randn(n_latent_low, n_genes)
    data_non_responder = np.abs(np.dot(latent, mixing))

    return data_responder, data_non_responder


def load_real_data(h5ad_path):
    """
    Load real scRNA-seq data from h5ad file.

    Requires: pip install scanpy anndata

    For GSE120575 (Sade-Feldman):
    - Download from GEO or Single Cell Portal
    - Check adata.obs.columns for response metadata column name
    """
    import scanpy as sc

    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)

    # Adjust column names based on actual metadata
    # Common names: 'response', 'Response', 'clinical_response'
    response_col = 'response'  # <-- ADJUST THIS

    responders = adata[adata.obs[response_col] == 'Responder'].X
    non_responders = adata[adata.obs[response_col] == 'Non-responder'].X

    if hasattr(responders, "toarray"):
        responders = responders.toarray()
        non_responders = non_responders.toarray()

    return responders, non_responders


def plot_results(data_resp, data_non_resp, output_path='figures/fig_dimensionality_analysis.pdf'):
    """Generate publication-quality figure."""

    print("Calculating metrics...")
    pr_resp, ev_resp = calculate_effective_dimensionality(data_resp)
    pr_non, ev_non = calculate_effective_dimensionality(data_non_resp)

    ent_resp = calculate_cell_entropy(data_resp)
    ent_non = calculate_cell_entropy(data_non_resp)

    print(f"Responder D_eff:     {pr_resp:.2f}")
    print(f"Non-Responder D_eff: {pr_non:.2f}")
    print(f"Ratio: {pr_resp/pr_non:.1f}x")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # A: Eigenvalue spectrum
    ax = axes[0]
    n_comps = 50
    ax.plot(range(n_comps), ev_resp[:n_comps]/ev_resp.sum(),
            label=f'Responder\n$D_{{eff}}$={pr_resp:.0f}', color='#2A9D8F', lw=2)
    ax.plot(range(n_comps), ev_non[:n_comps]/ev_non.sum(),
            label=f'Non-Responder\n$D_{{eff}}$={pr_non:.0f}', color='#E63946', lw=2)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('A. Eigenvalue Spectrum')
    ax.legend()
    ax.set_xlim(0, 50)

    # B: D_eff comparison
    ax = axes[1]
    bars = ax.bar(['Responder', 'Non-Responder'], [pr_resp, pr_non],
                  color=['#2A9D8F', '#E63946'], alpha=0.8)
    ax.set_ylabel('Effective Dimensionality ($D_{eff}$)')
    ax.set_title('B. Population Complexity')
    ax.text(0.5, 0.95, f'{pr_resp/pr_non:.0f}x difference',
            transform=ax.transAxes, ha='center', va='top', fontsize=11)

    # C: Cell entropy
    ax = axes[2]
    sns.kdeplot(ent_resp, fill=True, color='#2A9D8F', label='Responder', ax=ax, alpha=0.6)
    sns.kdeplot(ent_non, fill=True, color='#E63946', label='Non-Responder', ax=ax, alpha=0.6)
    ax.set_xlabel('Transcriptomic Entropy (per cell)')
    ax.set_ylabel('Density')
    ax.set_title('C. Individual Cell Complexity')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return pr_resp, pr_non


if __name__ == "__main__":
    # Toggle for real vs mock data
    USE_REAL_DATA = False
    REAL_DATA_PATH = "GSE120575.h5ad"

    if USE_REAL_DATA:
        data_resp, data_non_resp = load_real_data(REAL_DATA_PATH)
    else:
        print("Using mock data (set USE_REAL_DATA=True for real analysis)")
        data_resp, data_non_resp = generate_mock_data()

    print(f"Data shapes: Responder {data_resp.shape}, Non-Responder {data_non_resp.shape}")

    plot_results(data_resp, data_non_resp)
