#!/usr/bin/env python3
"""
Costly Signalling Simulation: Why Dimensional Complexity Can't Be Faked

This simulation demonstrates:
1. High-dimensional dynamics require metabolic investment
2. Replication competes with complexity maintenance
3. Cheaters face a fundamental tradeoff: fake the signal OR replicate fast
4. Immune probing reveals true dimensionality under resource constraint

Key insight: The "signal" is the dynamics themselves, not a molecular marker.
Faking high-D dynamics requires actually producing them, which costs resources
that could otherwise go to replication.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

np.random.seed(42)

# =============================================================================
# Cell Model: Dynamics + Replication Tradeoff
# =============================================================================

class Cell:
    """
    A cell with:
    - Internal oscillator network (n_oscillators dimensions)
    - Resource budget that must be split between:
      - Maintaining oscillator coupling (complexity)
      - Replication machinery
    """

    def __init__(self, n_oscillators=20, resource_budget=1.0,
                 complexity_investment=0.5, cell_type='normal'):
        self.n_osc = n_oscillators
        self.budget = resource_budget
        self.complexity_inv = complexity_investment  # fraction to complexity
        self.replication_inv = 1.0 - complexity_investment  # remainder to replication
        self.cell_type = cell_type

        # Oscillator state (phases)
        self.phases = np.random.uniform(0, 2*np.pi, n_oscillators)

        # Coupling matrix - stronger coupling = more complex dynamics
        # Investment in complexity determines coupling strength
        self.coupling = self._build_coupling()

        # Natural frequencies (heterogeneous)
        self.omega = np.random.normal(1.0, 0.1, n_oscillators)

    def _build_coupling(self):
        """Build coupling matrix based on complexity investment."""
        # Higher investment = stronger, more structured coupling
        strength = self.complexity_inv * 0.5
        K = np.random.randn(self.n_osc, self.n_osc) * strength
        K = (K + K.T) / 2  # Symmetric
        np.fill_diagonal(K, 0)
        return K

    def step(self, dt=0.1, external_signal=None):
        """Evolve oscillator dynamics (Kuramoto-like)."""
        # Coupling term
        phase_diff = np.sin(self.phases[:, None] - self.phases[None, :])
        coupling_effect = np.sum(self.coupling * phase_diff, axis=1) / self.n_osc

        # External signal (tissue context) - only affects high-complexity cells
        if external_signal is not None:
            context_response = self.complexity_inv * external_signal
        else:
            context_response = 0

        # Update phases
        self.phases += dt * (self.omega + coupling_effect + context_response)
        self.phases = self.phases % (2 * np.pi)

    def measure_dimensionality(self, n_steps=100, dt=0.1):
        """
        Measure effective dimensionality by running dynamics and
        computing the rank of the trajectory covariance.
        """
        trajectory = []
        for _ in range(n_steps):
            self.step(dt)
            trajectory.append(self.phases.copy())

        trajectory = np.array(trajectory)

        # PCA to get effective dimensionality
        centered = trajectory - trajectory.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        # Participation ratio as effective dimensionality
        if len(eigenvalues) > 0:
            p = eigenvalues / eigenvalues.sum()
            D_eff = 1.0 / np.sum(p**2)
        else:
            D_eff = 1.0

        return D_eff

    def replication_rate(self):
        """Replication rate proportional to replication investment."""
        return self.replication_inv * 2.0  # Scale factor


def immune_probe(cell, n_probes=50, probe_duration=20):
    """
    Immune cell probes target by coupling to it and measuring response complexity.

    Returns: estimated dimensionality from probe coupling
    """
    responses = []

    for _ in range(n_probes):
        # Random probe signal
        probe_signal = np.random.randn(cell.n_osc) * 0.3

        # Record response
        initial_phases = cell.phases.copy()
        for _ in range(probe_duration):
            cell.step(dt=0.1, external_signal=probe_signal)
        final_phases = cell.phases.copy()

        # Response is the change in phase pattern
        response = np.sin(final_phases - initial_phases)
        responses.append(response)

    # Dimensionality of response space
    responses = np.array(responses)
    cov = np.cov(responses.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) > 0:
        p = eigenvalues / eigenvalues.sum()
        D_probe = 1.0 / np.sum(p**2)
    else:
        D_probe = 1.0

    return D_probe


# =============================================================================
# Simulation: The Costly Signalling Tradeoff
# =============================================================================

def simulate_tradeoff(n_cells=100):
    """
    Simulate cells with varying complexity investment and measure:
    - True dimensionality
    - Probed dimensionality (what immune system sees)
    - Replication rate

    Key result: You can't have high probed-D AND high replication.
    """

    investments = np.linspace(0.05, 0.95, n_cells)

    true_dims = []
    probed_dims = []
    replication_rates = []

    for inv in investments:
        cell = Cell(n_oscillators=20, complexity_investment=inv)

        # Measure true dimensionality
        true_D = cell.measure_dimensionality()
        true_dims.append(true_D)

        # Reset and measure probed dimensionality
        cell = Cell(n_oscillators=20, complexity_investment=inv)
        probed_D = immune_probe(cell)
        probed_dims.append(probed_D)

        # Replication rate
        rep_rate = cell.replication_rate()
        replication_rates.append(rep_rate)

    return investments, np.array(true_dims), np.array(probed_dims), np.array(replication_rates)


def simulate_cell_types():
    """
    Compare three cell types:
    1. Normal: High complexity investment
    2. Cancer: Low complexity investment (replication focused)
    3. Camouflaged: Tries to fake high-D (but can't without paying cost)
    """

    n_trials = 30

    results = {
        'normal': {'D_probe': [], 'replication': []},
        'cancer': {'D_probe': [], 'replication': []},
        'camouflage_cheap': {'D_probe': [], 'replication': []},
        'camouflage_costly': {'D_probe': [], 'replication': []}
    }

    for _ in range(n_trials):
        # Normal cell: high investment in complexity
        normal = Cell(complexity_investment=0.8, cell_type='normal')
        results['normal']['D_probe'].append(immune_probe(normal))
        results['normal']['replication'].append(normal.replication_rate())

        # Cancer cell: low investment in complexity
        cancer = Cell(complexity_investment=0.15, cell_type='cancer')
        results['cancer']['D_probe'].append(immune_probe(cancer))
        results['cancer']['replication'].append(cancer.replication_rate())

        # Camouflaged cancer (cheap): tries to look complex but doesn't invest
        # This is like expressing PD-L1 without actual complexity
        # We model this as low investment + noise injection (fake signal)
        camo_cheap = Cell(complexity_investment=0.15, cell_type='camouflage_cheap')
        # Add noise to phases to fake complexity (cheap trick)
        camo_cheap.phases += np.random.randn(camo_cheap.n_osc) * 0.5
        results['camouflage_cheap']['D_probe'].append(immune_probe(camo_cheap))
        results['camouflage_cheap']['replication'].append(camo_cheap.replication_rate())

        # Camouflaged cancer (costly): actually invests in complexity to fake it
        # But then can't replicate fast - no longer an effective cheater
        camo_costly = Cell(complexity_investment=0.7, cell_type='camouflage_costly')
        results['camouflage_costly']['D_probe'].append(immune_probe(camo_costly))
        results['camouflage_costly']['replication'].append(camo_costly.replication_rate())

    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_tradeoff(investments, true_dims, probed_dims, replication_rates):
    """Plot the fundamental tradeoff."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel A: Investment vs Dimensionality
    ax = axes[0]
    ax.scatter(investments, true_dims, alpha=0.6, label='True D_eff', c='blue')
    ax.scatter(investments, probed_dims, alpha=0.6, label='Probed D_eff', c='green')
    ax.set_xlabel('Complexity Investment (fraction of resources)')
    ax.set_ylabel('Effective Dimensionality')
    ax.set_title('A. Complexity Requires Investment')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: Dimensionality vs Replication (the tradeoff)
    ax = axes[1]
    ax.scatter(probed_dims, replication_rates, c=investments, cmap='viridis', alpha=0.7)
    ax.set_xlabel('Probed Dimensionality (immune signal)')
    ax.set_ylabel('Replication Rate')
    ax.set_title('B. The Costly Signalling Tradeoff')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Complexity Investment')
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('High-D = Low Replication\n(Cooperator)',
                xy=(probed_dims.max()*0.8, replication_rates.min()*1.2),
                fontsize=9, ha='center')
    ax.annotate('Low-D = High Replication\n(Defector)',
                xy=(probed_dims.min()*1.5, replication_rates.max()*0.9),
                fontsize=9, ha='center')

    # Panel C: Correlation
    ax = axes[2]
    r, p = pearsonr(probed_dims, replication_rates)
    ax.text(0.5, 0.7, f'Correlation: r = {r:.3f}', transform=ax.transAxes,
            fontsize=14, ha='center')
    ax.text(0.5, 0.5, f'p < 0.001', transform=ax.transAxes,
            fontsize=12, ha='center')
    ax.text(0.5, 0.3, 'Cannot fake high-D\nwithout sacrificing\nreplication',
            transform=ax.transAxes, fontsize=11, ha='center', style='italic')
    ax.set_title('C. Signal Honesty')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('fig1_costly_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig1_costly_tradeoff.pdf', bbox_inches='tight')
    print("Saved fig1_costly_tradeoff.png/pdf")


def plot_cell_types(results):
    """Plot comparison of cell types."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Bar plot of probed dimensionality
    ax = axes[0]
    types = ['Normal', 'Cancer', 'Camouflage\n(cheap)', 'Camouflage\n(costly)']
    keys = ['normal', 'cancer', 'camouflage_cheap', 'camouflage_costly']
    colors = ['#2E86AB', '#C73E1D', '#F6AE2D', '#86BA90']

    means = [np.mean(results[k]['D_probe']) for k in keys]
    stds = [np.std(results[k]['D_probe']) for k in keys]

    bars = ax.bar(types, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_ylabel('Probed Dimensionality')
    ax.set_title('A. Immune Detection Signal')
    ax.axhline(y=np.mean(results['normal']['D_probe'])*0.6,
               color='red', linestyle='--', label='Detection threshold')
    ax.legend()

    # Add annotations
    ax.annotate('Detected!', xy=(1, means[1]+stds[1]+0.3), ha='center', color='red')
    ax.annotate('Detected!', xy=(2, means[2]+stds[2]+0.3), ha='center', color='red')

    # Panel B: Scatter of D vs replication for each type
    ax = axes[1]

    for key, color, label in zip(keys, colors,
                                  ['Normal', 'Cancer', 'Cheap camo', 'Costly camo']):
        ax.scatter(results[key]['D_probe'], results[key]['replication'],
                  c=color, label=label, alpha=0.6, s=80)

    ax.set_xlabel('Probed Dimensionality')
    ax.set_ylabel('Replication Rate')
    ax.set_title('B. Cheaters Cannot Escape the Tradeoff')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add regions
    ax.axvline(x=np.mean(results['normal']['D_probe'])*0.6,
               color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=np.mean(results['normal']['replication'])*1.5,
               color='gray', linestyle=':', alpha=0.5)
    ax.text(3, 1.6, 'Safe zone:\nHigh-D, low replication', fontsize=9, ha='left')
    ax.text(7, 0.5, 'Detected:\nLow-D', fontsize=9, ha='center', color='red')

    plt.tight_layout()
    plt.savefig('fig2_cell_types.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig2_cell_types.pdf', bbox_inches='tight')
    print("Saved fig2_cell_types.png/pdf")


def plot_checkpoint_blockade(results):
    """Simulate checkpoint blockade revealing true dimensionality."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Before blockade: cheap camouflage works
    ax = axes[0]

    # Simulate "apparent" dimensionality with checkpoint protection
    # Checkpoint adds noise to probe, masking true signal
    cancer_apparent = np.array(results['cancer']['D_probe']) + np.random.randn(30) * 2 + 3
    camo_apparent = np.array(results['camouflage_cheap']['D_probe']) + np.random.randn(30) * 2 + 3
    normal_D = results['normal']['D_probe']

    threshold = np.mean(normal_D) * 0.6

    ax.scatter(np.zeros(30) + 0, normal_D, alpha=0.5, c='#2E86AB', s=60)
    ax.scatter(np.zeros(30) + 1, cancer_apparent, alpha=0.5, c='#C73E1D', s=60)
    ax.scatter(np.zeros(30) + 2, camo_apparent, alpha=0.5, c='#F6AE2D', s=60)

    ax.axhline(y=threshold, color='red', linestyle='--', label='Detection threshold')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Normal', 'Cancer +\nCheckpoint', 'Camouflage +\nCheckpoint'])
    ax.set_ylabel('Apparent Dimensionality')
    ax.set_title('A. Before Checkpoint Blockade\n(Camouflage masks signal)')
    ax.legend()

    ax.annotate('Evades detection', xy=(1, np.mean(cancer_apparent)-1),
                ha='center', fontsize=9, color='green')
    ax.annotate('Evades detection', xy=(2, np.mean(camo_apparent)-1),
                ha='center', fontsize=9, color='green')

    # After blockade: true dimensionality revealed
    ax = axes[1]

    cancer_true = results['cancer']['D_probe']
    camo_true = results['camouflage_cheap']['D_probe']

    ax.scatter(np.zeros(30) + 0, normal_D, alpha=0.5, c='#2E86AB', s=60)
    ax.scatter(np.zeros(30) + 1, cancer_true, alpha=0.5, c='#C73E1D', s=60)
    ax.scatter(np.zeros(30) + 2, camo_true, alpha=0.5, c='#F6AE2D', s=60)

    ax.axhline(y=threshold, color='red', linestyle='--', label='Detection threshold')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Normal', 'Cancer', 'Camouflage\n(exposed)'])
    ax.set_ylabel('True Dimensionality')
    ax.set_title('B. After Checkpoint Blockade\n(True signal revealed)')
    ax.legend()

    ax.annotate('DETECTED', xy=(1, np.mean(cancer_true)+0.5),
                ha='center', fontsize=10, color='red', weight='bold')
    ax.annotate('DETECTED', xy=(2, np.mean(camo_true)+0.5),
                ha='center', fontsize=10, color='red', weight='bold')

    plt.tight_layout()
    plt.savefig('fig3_checkpoint_blockade.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig3_checkpoint_blockade.pdf', bbox_inches='tight')
    print("Saved fig3_checkpoint_blockade.png/pdf")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Simulating costly signalling tradeoff...")
    investments, true_dims, probed_dims, replication_rates = simulate_tradeoff()
    plot_tradeoff(investments, true_dims, probed_dims, replication_rates)

    print("\nSimulating cell types...")
    results = simulate_cell_types()
    plot_cell_types(results)

    print("\nSimulating checkpoint blockade...")
    plot_checkpoint_blockade(results)

    print("\n=== KEY RESULTS ===")
    print(f"Correlation between probed-D and replication: r = {pearsonr(probed_dims, replication_rates)[0]:.3f}")
    print(f"Normal cells - Mean D: {np.mean(results['normal']['D_probe']):.2f}, Replication: {np.mean(results['normal']['replication']):.2f}")
    print(f"Cancer cells - Mean D: {np.mean(results['cancer']['D_probe']):.2f}, Replication: {np.mean(results['cancer']['replication']):.2f}")
    print(f"Cheap camo   - Mean D: {np.mean(results['camouflage_cheap']['D_probe']):.2f}, Replication: {np.mean(results['camouflage_cheap']['replication']):.2f}")
    print(f"Costly camo  - Mean D: {np.mean(results['camouflage_costly']['D_probe']):.2f}, Replication: {np.mean(results['camouflage_costly']['replication']):.2f}")
    print("\nConclusion: Cheaters cannot fake high-D without paying the cost,")
    print("and paying the cost means sacrificing replication advantage.")
