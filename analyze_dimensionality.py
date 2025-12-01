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


def load_real_data(data_dir="sade_feldman_data"):
    """
    Load GSE120575 (Sade-Feldman melanoma checkpoint response) from raw GEO files.

    Requires: pip install scanpy anndata pandas

    Download files first:
        curl -O "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE120nnn/GSE120575/suppl/GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz"
        curl -O "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE120nnn/GSE120575/suppl/GSE120575_patient_ID_single_cells.txt.gz"
        gunzip *.gz
    """
    import pandas as pd
    import anndata
    import os

    print("Loading expression matrix (this may take a minute)...")

    # 1. Load Expression Matrix (Rows=Genes, Cols=Cells)
    matrix_path = os.path.join(data_dir, "GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt")
    df_expr = pd.read_csv(matrix_path, sep='\t', index_col=0)
    print(f"  Expression matrix: {df_expr.shape[0]} genes x {df_expr.shape[1]} cells")

    # 2. Load Metadata
    meta_path = os.path.join(data_dir, "GSE120575_patient_ID_single_cells.txt")
    df_meta = pd.read_csv(meta_path, sep='\t')

    # Clean column names (GEO uses verbose names like 'characteristics: response')
    df_meta.columns = [c.split(': ')[-1] if ': ' in c else c for c in df_meta.columns]
    print(f"  Metadata columns: {list(df_meta.columns)}")

    # 3. Create AnnData (Cells x Genes, so transpose)
    print("Creating AnnData object...")
    adata = anndata.AnnData(X=df_expr.T.values)
    adata.obs_names = df_expr.columns
    adata.var_names = df_expr.index

    # 4. Merge metadata - align by cell names
    df_meta_indexed = df_meta.set_index(df_meta.columns[0])
    adata.obs = df_meta_indexed.loc[adata.obs_names]

    # 5. Find response column and split
    response_col = [c for c in adata.obs.columns if 'response' in c.lower()]
    if not response_col:
        print("  Available columns:", list(adata.obs.columns))
        raise ValueError("No 'response' column found in metadata")
    response_col = response_col[0]
    print(f"  Using response column: '{response_col}'")
    print(f"  Response values: {adata.obs[response_col].value_counts().to_dict()}")

    # Split by response
    resp_mask = adata.obs[response_col].astype(str).str.lower().str.contains('responder') & \
                ~adata.obs[response_col].astype(str).str.lower().str.contains('non')
    non_resp_mask = adata.obs[response_col].astype(str).str.lower().str.contains('non-responder|non responder|nonresponder')

    responders = adata.X[resp_mask]
    non_responders = adata.X[non_resp_mask]

    print(f"  Responders: {responders.shape[0]} cells")
    print(f"  Non-responders: {non_responders.shape[0]} cells")

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
    REAL_DATA_DIR = "sade_feldman_data"  # Directory with downloaded GEO files

    if USE_REAL_DATA:
        data_resp, data_non_resp = load_real_data(REAL_DATA_DIR)
    else:
        print("Using mock data (set USE_REAL_DATA=True for real analysis)")
        data_resp, data_non_resp = generate_mock_data()

    print(f"Data shapes: Responder {data_resp.shape}, Non-Responder {data_non_resp.shape}")

    plot_results(data_resp, data_non_resp)
