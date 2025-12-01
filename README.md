# Immune Cooperation

**Paper:** *The Physics of Immune Cooperation: Dimensional Surveillance and Attractor Enforcement in Multicellular Systems*

**Author:** Ian Todd, University of Sydney

**Status:** Ready for submission (47 pages, 18 references)

---

## The Argument

The immune system is conventionally understood as pattern recognition: distinguishing "self" from "non-self" via molecular signatures. We propose something more fundamental: **immunity is dimensional surveillance**.

Cells embedded in functional tissue exhibit high effective dimensionality—complex, context-sensitive dynamics. Cells that have "defected" (cancer, viral infection, senescence) exhibit low dimensionality—they've collapsed into simpler attractors.

Immune receptors (CD molecules, TCR-MHC interactions) function as **synchronization probes**: they couple to target cells and measure dynamical complexity. Low-D cells get flagged for elimination.

**Central thesis:** Health is high dimensionality maintained at criticality; disease is dimensional collapse.

---

## Key Theoretical Contributions

| Section | Contribution |
|---------|--------------|
| **Dynamical Friction** | Immune cells expend energy resisting synchronization with low-D targets. Exhaustion time scales as t ~ E₀/k(ΔD)² |
| **Sensor Collapse** | Autoimmunity as hardware failure—exhausted sensors lose resolution, produce aliasing (inflamm-aging) |
| **Hyper-Coupling** | Autism-autoimmunity connection: same K > K_c error in different substrates |
| **Dimensional Bottleneck** | Immune synapse as IFF codeforming—Self/Non-self are generated codes, not pre-existing labels |
| **Regeneration Tradeoff** | Mammals: deep attractors → no regeneration, aggressive cancer. Axolotl: shallow attractors + active guidance → regeneration + cancer resistance |

---

## Simulations

| Script | Output | Description |
|--------|--------|-------------|
| `vdj_dynamics_sim.py` | `fig4_dynamics_discrimination.pdf` | Spectral entropy discriminates high-D from low-D dynamics (Cohen's d = 3.7) |
| `costly_signalling_sim.py` | `fig1_costly_tradeoff.pdf`, `fig3_checkpoint_blockade.pdf` | Metabolic tradeoff between complexity and replication; checkpoint camouflage |
| `costly_signalling_friction.py` | `fig_dynamical_friction.pdf` | T-cell exhaustion as forced dimensional collapse from coupling to low-D targets |
| `run_dimensionality_analysis.py` | `fig_scrna_dimensionality.pdf` | **Empirical validation**: Responders have 2x higher D_eff (26.4 vs 13.0) |

Run any simulation:
```bash
python3 vdj_dynamics_sim.py
python3 costly_signalling_sim.py
python3 costly_signalling_friction.py
```

---

## scRNA-seq Analysis (GSE120575)

We test the hypothesis that immunotherapy responders have higher effective dimensionality than non-responders using the Sade-Feldman melanoma checkpoint response dataset.

### Data Acquisition

```bash
# Download from GEO
curl -O "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE120nnn/GSE120575/suppl/GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz"
curl -O "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE120nnn/GSE120575/suppl/GSE120575_patient_ID_single_cells.txt.gz"

# Decompress
gunzip *.gz

# Move to data folder
mkdir -p sade_feldman_data
mv GSE120575_*.txt sade_feldman_data/
```

### Data Preprocessing

The GEO file has a formatting quirk: header and line 2 have 16,292 columns, but lines 3+ have 16,293 (trailing tab). This breaks standard parsers.

**Fix with awk:**
```bash
cd sade_feldman_data
awk -F'\t' 'NR==1 {OFS="\t"; print $0, ""} NR>1 {if (NF==16292) print $0, ""; else print $0}' \
    GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt > expression_fixed.txt
```

### Verify Data Integrity

Compare your files against `DATA_CHECKSUMS.txt`:
```bash
shasum -a 256 sade_feldman_data/GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt
# Should match: f70438a6a5d6ea89e84dd3a912bf3b188ce34926c70fca21825b22333a3c4d56

shasum -a 256 sade_feldman_data/GSE120575_patient_ID_single_cells.txt
# Should match: e63a093f478ca10bf7a6d428dadfa3a0bad1f7e2fb1e050db68c459a8a1828c6
```

### Run Analysis

```bash
python3 run_dimensionality_analysis.py
```

This computes:
- **Participation Ratio (D_eff)**: PCA-based effective dimensionality per group
- **Transcriptomic Entropy**: Per-cell Shannon entropy of gene expression
- **Statistics**: t-test, Cohen's d effect size

Output: `figures/fig_scrna_dimensionality.pdf`

### Data Loader

`fast_loader.py` handles the GEO formatting issues:
1. Sniffs file structure to detect column count mismatch
2. Patches column names with deduplication
3. Uses C engine for speed (~60s load time for 2GB file)
4. Aligns expression matrix with metadata
5. Splits by treatment response

---

## Project Structure

```
immune_cooperation/
├── immune_cooperation.tex    # Main paper (LaTeX)
├── immune_cooperation.pdf    # Compiled paper
├── README.md                 # This file
├── DATA_CHECKSUMS.txt        # SHA256 hashes for data verification
├── .gitignore
│
├── figures/                  # Generated figures
│   ├── fig1_costly_tradeoff.pdf
│   ├── fig3_checkpoint_blockade.pdf
│   ├── fig4_dynamics_discrimination.pdf
│   └── fig_dynamical_friction.pdf
│
├── vdj_dynamics_sim.py           # Dynamics discrimination simulation
├── costly_signalling_sim.py      # Costly signalling simulation
├── costly_signalling_friction.py # Dynamical friction simulation
├── fast_loader.py                # Robust GEO data loader
├── run_dimensionality_analysis.py # scRNA-seq analysis
│
└── sade_feldman_data/        # (gitignored - download from GEO)
    ├── GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt
    ├── GSE120575_patient_ID_single_cells.txt
    └── expression_fixed.txt  # Preprocessed version
```

---

## Key References

- Cohen et al. (2022). "A complex systems approach to aging biology." *Nature Aging*.
- Levin (2021). "Bioelectric signaling: Reprogrammable circuits underlying embryogenesis, regeneration, and cancer." *Cell*.
- Sade-Feldman et al. (2018). "Defining T Cell States Associated with Response to Checkpoint Immunotherapy in Melanoma." *Cell*.

---

## License

Code: MIT License

Paper: © 2025 Ian Todd. All rights reserved.
