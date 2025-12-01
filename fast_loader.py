"""
Fast Loader for Sade-Feldman scRNA-seq Data (GSE120575)

Fixes the GEO formatting issue where header length != data length
by manually patching the column names and using C engine.

Author: Ian Todd
"""

import pandas as pd
import numpy as np
import os


def load_sade_feldman_fast(data_dir="sade_feldman_data"):
    """Load expression matrix with C engine by patching header mismatch."""

    print(f"--- Fast Loading Sade-Feldman Data ---")

    # Use preprocessed file if available (normalized column counts)
    expr_path = os.path.join(data_dir, "expression_fixed.txt")
    if not os.path.exists(expr_path):
        expr_path = os.path.join(data_dir, "GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt")
    meta_path = os.path.join(data_dir, "GSE120575_patient_ID_single_cells.txt")

    # 1. SNIFF THE STRUCTURE
    # Note: Lines 1-2 may be clean but line 3+ often have trailing tabs
    print("Sniffing file structure...")
    with open(expr_path, 'r') as f:
        header_line = f.readline().rstrip('\n').split('\t')
        f.readline()  # Skip first data line (might be clean)
        third_line = f.readline().rstrip('\n').split('\t')  # Check line 3

    n_header = len(header_line)
    n_data = len(third_line)  # Use line 3 as reference for data width

    print(f"Header columns: {n_header}")
    print(f"Data columns:   {n_data}")

    # 2. PATCH THE COLUMNS
    # Handle various formatting issues
    col_names = header_line.copy()

    # Add index name if missing
    if n_data == n_header + 1:
        print("Detected missing index name. Patching...")
        col_names = ['Gene_Symbol'] + header_line
    elif n_data > n_header:
        diff = n_data - n_header
        print(f"Detected {diff} extra columns. Patching...")
        col_names = ['Gene_Symbol'] + header_line + [f"DROP_{i}" for i in range(diff - 1)]

    # Deduplicate column names (add suffix to duplicates)
    seen = {}
    deduped = []
    for name in col_names:
        if name == '' or name is None:
            name = 'EMPTY'
        if name in seen:
            seen[name] += 1
            deduped.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            deduped.append(name)
    col_names = deduped
    print(f"Column names prepared: {len(col_names)} (deduplicated)")

    # 3. FAST LOAD (C Engine)
    # Problem: Line 2 has different field count than lines 3+
    # Solution: Skip lines with wrong field count using on_bad_lines
    print("Loading expression matrix with C engine...")
    try:
        df_expr = pd.read_csv(
            expr_path,
            sep='\t',
            names=col_names,
            header=0,      # Skip the file's header row
            index_col=0,   # Use Gene_Symbol as index
            engine='c',    # Force C engine for speed
            na_values=[''],
            low_memory=False,
            on_bad_lines='skip'  # Skip malformed lines
        )
    except TypeError:
        # Older pandas
        df_expr = pd.read_csv(
            expr_path,
            sep='\t',
            names=col_names,
            header=0,
            index_col=0,
            engine='c',
            na_values=[''],
            low_memory=False,
            error_bad_lines=False
        )

    # Drop any dummy columns
    drop_cols = [c for c in df_expr.columns if str(c).startswith('DROP_')]
    if drop_cols:
        print(f"Dropping {len(drop_cols)} trailing/empty columns...")
        df_expr.drop(columns=drop_cols, inplace=True)

    print(f"Expression matrix: {df_expr.shape[0]} genes x {df_expr.shape[1]} cells")

    # 4. LOAD METADATA
    print("Loading metadata...")
    # Find the actual header row (contains "Sample name" and "title")
    header_row = 0
    with open(meta_path, 'r', encoding='latin-1') as f:
        for i, line in enumerate(f):
            if "Sample name" in line and "title" in line:
                header_row = i
                break

    df_meta = pd.read_csv(meta_path, sep='\t', skiprows=header_row, encoding='latin-1')
    # Clean column names - remove "characteristics: " prefix
    df_meta.columns = [c.replace('characteristics: ', '').strip() for c in df_meta.columns]
    print(f"Metadata: {df_meta.shape[0]} rows, {df_meta.shape[1]} columns")
    print(f"Columns: {list(df_meta.columns[:8])}")

    # 5. ALIGN CELLS
    print("Aligning expression and metadata...")

    # Find matching column
    meta_index_col = None
    common_cells = []

    for col in ['title', 'geo_accession', df_meta.columns[0]]:
        if col in df_meta.columns:
            intersect = set(df_expr.columns).intersection(set(df_meta[col].astype(str)))
            if len(intersect) > 100:
                meta_index_col = col
                common_cells = list(intersect)
                print(f"Matched {len(common_cells)} cells using '{col}'")
                break

    if not common_cells:
        print("ERROR: Could not match cells between files")
        return None, None, None

    # Filter to common cells
    df_expr = df_expr[common_cells]
    df_meta = df_meta.set_index(meta_index_col).loc[common_cells]

    # 6. SPLIT BY RESPONSE
    resp_col = [c for c in df_meta.columns if 'response' in c.lower()]
    if not resp_col:
        print("ERROR: No response column found")
        return df_expr, df_meta, None
    resp_col = resp_col[0]

    print(f"Response column: '{resp_col}'")
    print(f"Values: {df_meta[resp_col].value_counts().to_dict()}")

    # Create masks
    is_non_resp = df_meta[resp_col].astype(str).str.contains("Non", case=False, na=False)
    is_resp = (~is_non_resp) & df_meta[resp_col].astype(str).str.contains("Responder", case=False, na=False)

    # Transpose to cells x genes and convert to numeric
    data_resp = df_expr.loc[:, is_resp].T
    data_non_resp = df_expr.loc[:, is_non_resp].T

    # Convert to numeric, coercing errors to NaN
    data_resp = data_resp.apply(pd.to_numeric, errors='coerce').fillna(0)
    data_non_resp = data_non_resp.apply(pd.to_numeric, errors='coerce').fillna(0)

    print(f"\nSUCCESS:")
    print(f"  Responders: {len(data_resp)} cells x {data_resp.shape[1]} genes")
    print(f"  Non-responders: {len(data_non_resp)} cells x {data_non_resp.shape[1]} genes")

    return data_resp.values.astype(np.float32), data_non_resp.values.astype(np.float32), df_meta


if __name__ == "__main__":
    try:
        r, nr, meta = load_sade_feldman_fast()
        if r is not None:
            print("\n" + "="*50)
            print("Data loaded successfully!")
            print(f"Ready for dimensionality analysis.")
            print("="*50)
    except Exception as e:
        import traceback
        print(f"Failed: {e}")
        traceback.print_exc()
