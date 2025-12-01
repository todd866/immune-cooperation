"""
Robust Loader for Sade-Feldman scRNA-seq Data (GSE120575)

Handles common GEO text file issues:
1. Auto-detects header location
2. Fixes off-by-one column errors
3. Memory-safe cell alignment

Author: Ian Todd
"""

import pandas as pd
import os
import sys


def load_sade_feldman_robust(data_dir="sade_feldman_data"):
    """Load and align Sade-Feldman scRNA-seq data with robust error handling."""

    print(f"--- Loading Sade-Feldman Data from {data_dir} ---")

    # 1. ROBUST METADATA LOADING
    meta_path = os.path.join(data_dir, "GSE120575_patient_ID_single_cells.txt")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}. Did you run 'gunzip *.gz'?")

    print("Scanning metadata headers...")
    header_row = 0
    with open(meta_path, 'r', encoding='latin-1') as f:
        for i, line in enumerate(f):
            # Look for known column names in this dataset
            if "characteristics: response" in line or "characteristics: patinet ID" in line:
                header_row = i
                break

    # Load with the detected header row
    df_meta = pd.read_csv(meta_path, sep='\t', skiprows=header_row, encoding='latin-1')

    # Clean up column names (remove "characteristics: " prefix)
    df_meta.columns = [c.split(': ')[-1] if ': ' in c else c for c in df_meta.columns]

    print(f"Loaded metadata: {df_meta.shape[0]} rows, {df_meta.shape[1]} columns")
    print(f"Columns: {list(df_meta.columns[:10])}...")

    # 2. ROBUST EXPRESSION LOADING
    expr_path = os.path.join(data_dir, "GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt")

    if not os.path.exists(expr_path):
        raise FileNotFoundError(f"Missing {expr_path}. Did you run 'gunzip *.gz'?")

    print("Loading expression matrix (this uses ~2GB RAM)...")

    # FIX: GEO files have trailing tab on every line
    # Header: gene_col + N cell_ids (but file has trailing tab)
    # Data: gene_name + N values + trailing tab
    # Solution: Read raw without parsing, then fix

    with open(expr_path, 'r') as f:
        header_line = f.readline()
    # Split preserving the trailing empty string from trailing tab
    header_cols = header_line.rstrip('\n').split('\t')
    # Filter out empty strings
    header_cols = [c for c in header_cols if c.strip()]
    n_header = len(header_cols)
    print(f"Header has {n_header} columns (including gene name column)")

    # The data rows have n_header + 1 fields (trailing tab)
    # Create column names: header columns + dummy for trailing
    col_names = header_cols + ['_trailing']

    # Read using python engine which handles inconsistent field counts
    # Use on_bad_lines='warn' to handle extra columns gracefully
    df_expr = pd.read_csv(expr_path, sep='\t', skiprows=1, names=col_names,
                          na_values=[''], engine='python', on_bad_lines='warn')

    # Set the first column as index (gene names)
    df_expr = df_expr.set_index(df_expr.columns[0])

    # Drop the trailing empty column
    if '_trailing' in df_expr.columns:
        df_expr = df_expr.drop('_trailing', axis=1)
        print("Dropped trailing empty column")

    # Also drop any remaining all-NaN columns
    nan_cols = df_expr.columns[df_expr.isna().all()]
    if len(nan_cols) > 0:
        df_expr = df_expr.drop(nan_cols, axis=1)
        print(f"Dropped {len(nan_cols)} additional NaN columns")

    print(f"Expression matrix: {df_expr.shape[0]} genes x {df_expr.shape[1]} cells")

    # 3. ALIGNMENT
    print("Aligning metadata and expression...")

    common_cells = []
    meta_index_col = None

    # Try 'title' first (standard GEO)
    if 'title' in df_meta.columns:
        intersect = set(df_expr.columns).intersection(set(df_meta['title']))
        if len(intersect) > 100:
            print(f"Found match using 'title' column ({len(intersect)} cells).")
            meta_index_col = 'title'
            common_cells = list(intersect)

    # If 'title' failed, try 'geo_accession'
    if not common_cells and 'geo_accession' in df_meta.columns:
        intersect = set(df_expr.columns).intersection(set(df_meta['geo_accession']))
        if len(intersect) > 100:
            print(f"Found match using 'geo_accession' column ({len(intersect)} cells).")
            meta_index_col = 'geo_accession'
            common_cells = list(intersect)

    # Try the first column if nothing else works
    if not common_cells:
        first_col = df_meta.columns[0]
        intersect = set(df_expr.columns).intersection(set(df_meta[first_col].astype(str)))
        if len(intersect) > 100:
            print(f"Found match using '{first_col}' column ({len(intersect)} cells).")
            meta_index_col = first_col
            common_cells = list(intersect)

    if not common_cells:
        print("ERROR: Could not match Cell IDs between files.")
        print("Expression cols sample:", list(df_expr.columns[:5]))
        print("Metadata cols:", list(df_meta.columns))
        return None, None

    # Filter both to the common set
    df_expr = df_expr[common_cells]
    df_meta = df_meta.set_index(meta_index_col).loc[common_cells]

    # 4. SPLIT BY RESPONSE
    print("Splitting Responders vs Non-Responders...")

    # Identify the response column (case insensitive search)
    resp_col = [c for c in df_meta.columns if 'response' in c.lower()]
    if not resp_col:
        print("ERROR: Could not find 'response' column in metadata.")
        print("Available columns:", list(df_meta.columns))
        return None, None
    resp_col = resp_col[0]

    print(f"Response column: '{resp_col}'")
    print(f"Unique values: {df_meta[resp_col].unique()}")

    # Boolean masks - "Non-responder" contains "Responder" so order matters
    is_non_responder = df_meta[resp_col].astype(str).str.contains("Non", case=False, na=False)
    is_responder = (~is_non_responder) & df_meta[resp_col].astype(str).str.contains("Responder", case=False, na=False)

    # Transpose to Cells x Genes for analysis
    data_resp = df_expr.loc[:, is_responder].T
    data_non_resp = df_expr.loc[:, is_non_responder].T

    print(f"SUCCESS: Loaded {len(data_resp)} Responders and {len(data_non_resp)} Non-Responders.")
    print(f"Gene count: {data_resp.shape[1]}")

    return data_resp.values, data_non_resp.values


if __name__ == "__main__":
    try:
        r, nr = load_sade_feldman_robust()
        if r is not None:
            print("\n" + "="*50)
            print("Data loaded successfully!")
            print(f"Responders shape: {r.shape}")
            print(f"Non-responders shape: {nr.shape}")
            print("="*50)
    except Exception as e:
        import traceback
        print(f"Loader failed: {e}")
        traceback.print_exc()
