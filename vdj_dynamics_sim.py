"""
VDJ Dynamics Simulation

Models immune detection as temporal dynamics, not static vectors.
Cells are coupled oscillators; dimensionality = number of active modes.
The immune system probes synchronization patterns over time.

Key insight: Low-D systems show coherent, predictable dynamics.
High-D systems show complex, unpredictable dynamics.
This is detectable through temporal correlation patterns.

Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.facecolor'] = 'white'

np.random.seed(42)


def generate_cell_dynamics(n_timesteps: int, n_modes: int, dt: float = 0.01) -> np.ndarray:
    """
    Generate cellular dynamics as sum of oscillatory modes.

    More modes = higher effective dimensionality.
    Each mode has random frequency and phase.

    Low n_modes: coherent, predictable signal (pathogen-like)
    High n_modes: complex, noisy signal (self-like)
    """
    t = np.arange(n_timesteps) * dt
    signal = np.zeros(n_timesteps)

    # Generate random frequencies and phases for each mode
    # Frequencies span a biologically plausible range
    base_freqs = np.random.uniform(0.5, 10, n_modes)  # Hz
    phases = np.random.uniform(0, 2*np.pi, n_modes)
    amplitudes = np.random.exponential(1.0, n_modes)
    amplitudes = amplitudes / np.sum(amplitudes)  # Normalize total power

    for i in range(n_modes):
        signal += amplitudes[i] * np.sin(2*np.pi*base_freqs[i]*t + phases[i])

    # Add small noise floor
    signal += np.random.randn(n_timesteps) * 0.05

    return signal, t


def generate_receptor_probe(n_timesteps: int, probe_freq: float, dt: float = 0.01) -> np.ndarray:
    """
    Generate a receptor "probe" signal - an oscillator at a specific frequency.

    The receptor tests whether the target cell responds at this frequency.
    """
    t = np.arange(n_timesteps) * dt
    phase = np.random.uniform(0, 2*np.pi)
    probe = np.sin(2*np.pi*probe_freq*t + phase)
    return probe


def compute_synchronization(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """
    Compute phase synchronization between two signals.

    Uses Hilbert transform to extract instantaneous phase,
    then measures phase locking value (PLV).
    """
    # Get analytic signals
    analytic1 = hilbert(signal1)
    analytic2 = hilbert(signal2)

    # Extract phases
    phase1 = np.angle(analytic1)
    phase2 = np.angle(analytic2)

    # Phase locking value
    phase_diff = phase1 - phase2
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))

    return plv


def compute_spectral_entropy(signal: np.ndarray) -> float:
    """
    Compute spectral entropy - a measure of signal complexity.

    Low entropy = few dominant frequencies (low-D, pathogen-like)
    High entropy = many frequencies (high-D, self-like)
    """
    # Compute power spectrum
    spectrum = np.abs(fft(signal))**2
    spectrum = spectrum[:len(spectrum)//2]  # Take positive frequencies

    # Normalize to probability distribution
    spectrum = spectrum / np.sum(spectrum)
    spectrum = spectrum[spectrum > 1e-10]  # Remove zeros

    # Compute entropy
    entropy = -np.sum(spectrum * np.log(spectrum))

    # Normalize by max possible entropy
    max_entropy = np.log(len(spectrum))
    normalized_entropy = entropy / max_entropy

    return normalized_entropy


def compute_autocorrelation_decay(signal: np.ndarray, max_lag: int = 100) -> float:
    """
    Measure how quickly autocorrelation decays.

    Fast decay = complex, unpredictable (high-D)
    Slow decay = coherent, predictable (low-D)
    """
    n = len(signal)
    signal = signal - np.mean(signal)

    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[n-1:n-1+max_lag]  # Positive lags only
    autocorr = autocorr / autocorr[0]  # Normalize

    # Fit exponential decay
    lags = np.arange(max_lag)
    # Find half-life (where autocorr drops to 0.5)
    half_life_idx = np.where(autocorr < 0.5)[0]
    if len(half_life_idx) > 0:
        half_life = half_life_idx[0]
    else:
        half_life = max_lag

    return half_life


def probe_cell_with_repertoire(cell_signal: np.ndarray, n_probes: int,
                                dt: float = 0.01) -> np.ndarray:
    """
    Probe a cell with multiple receptor frequencies.

    Returns the synchronization pattern across probes.
    """
    n_timesteps = len(cell_signal)
    probe_freqs = np.linspace(0.5, 10, n_probes)  # Range of probe frequencies

    sync_pattern = np.zeros(n_probes)
    for i, freq in enumerate(probe_freqs):
        probe = generate_receptor_probe(n_timesteps, freq, dt)
        sync_pattern[i] = compute_synchronization(cell_signal, probe)

    return sync_pattern


def run_discrimination_experiment():
    """
    Main experiment: can we discriminate low-D from high-D cells
    based on their temporal dynamics?
    """
    n_timesteps = 2000
    dt = 0.01
    n_samples = 200
    n_probes = 50

    # Low-D cells (pathogen-like): 2-5 modes
    low_d_modes = [2, 3, 4, 5]
    # High-D cells (self-like): 20-50 modes
    high_d_modes = [20, 30, 40, 50]

    results = {
        'spectral_entropy': {'low_d': [], 'high_d': []},
        'autocorr_decay': {'low_d': [], 'high_d': []},
        'sync_variance': {'low_d': [], 'high_d': []}
    }

    print("Generating low-D (pathogen-like) dynamics...")
    for _ in range(n_samples):
        n_modes = np.random.choice(low_d_modes)
        signal, t = generate_cell_dynamics(n_timesteps, n_modes, dt)

        results['spectral_entropy']['low_d'].append(compute_spectral_entropy(signal))
        results['autocorr_decay']['low_d'].append(compute_autocorrelation_decay(signal))

        sync_pattern = probe_cell_with_repertoire(signal, n_probes, dt)
        results['sync_variance']['low_d'].append(np.var(sync_pattern))

    print("Generating high-D (self-like) dynamics...")
    for _ in range(n_samples):
        n_modes = np.random.choice(high_d_modes)
        signal, t = generate_cell_dynamics(n_timesteps, n_modes, dt)

        results['spectral_entropy']['high_d'].append(compute_spectral_entropy(signal))
        results['autocorr_decay']['high_d'].append(compute_autocorrelation_decay(signal))

        sync_pattern = probe_cell_with_repertoire(signal, n_probes, dt)
        results['sync_variance']['high_d'].append(np.var(sync_pattern))

    return results


def plot_results(results):
    """Generate figures showing discrimination."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Spectral entropy distributions
    ax = axes[0, 0]
    ax.hist(results['spectral_entropy']['high_d'], bins=25, alpha=0.7,
            color='#2A9D8F', label='High-D (self)', density=True)
    ax.hist(results['spectral_entropy']['low_d'], bins=25, alpha=0.7,
            color='#E63946', label='Low-D (pathogen)', density=True)
    ax.set_xlabel('Spectral Entropy')
    ax.set_ylabel('Density')
    ax.set_title('Low-D Cells Have Lower Spectral Entropy')
    ax.legend()

    # Compute effect size
    low_d = results['spectral_entropy']['low_d']
    high_d = results['spectral_entropy']['high_d']
    pooled_std = np.sqrt((np.var(low_d) + np.var(high_d)) / 2)
    cohens_d = (np.mean(high_d) - np.mean(low_d)) / pooled_std
    ax.text(0.05, 0.95, f"Cohen's d = {cohens_d:.2f}", transform=ax.transAxes,
            va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. Autocorrelation decay
    ax = axes[0, 1]
    ax.hist(results['autocorr_decay']['high_d'], bins=25, alpha=0.7,
            color='#2A9D8F', label='High-D (self)', density=True)
    ax.hist(results['autocorr_decay']['low_d'], bins=25, alpha=0.7,
            color='#E63946', label='Low-D (pathogen)', density=True)
    ax.set_xlabel('Autocorrelation Half-life (timesteps)')
    ax.set_ylabel('Density')
    ax.set_title('Low-D Cells Have Slower Decorrelation')
    ax.legend()

    low_d = results['autocorr_decay']['low_d']
    high_d = results['autocorr_decay']['high_d']
    pooled_std = np.sqrt((np.var(low_d) + np.var(high_d)) / 2)
    cohens_d = (np.mean(low_d) - np.mean(high_d)) / pooled_std
    ax.text(0.95, 0.95, f"Cohen's d = {cohens_d:.2f}", transform=ax.transAxes,
            va='top', ha='right', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 3. Example time series
    ax = axes[1, 0]
    np.random.seed(123)

    # Low-D example
    signal_low, t = generate_cell_dynamics(500, n_modes=3, dt=0.01)
    ax.plot(t[:300], signal_low[:300] + 2, color='#E63946', linewidth=1, label='Low-D (3 modes)')

    # High-D example
    signal_high, t = generate_cell_dynamics(500, n_modes=30, dt=0.01)
    ax.plot(t[:300], signal_high[:300], color='#2A9D8F', linewidth=1, label='High-D (30 modes)')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Signal (offset for clarity)')
    ax.set_title('Example Cellular Dynamics')
    ax.legend()
    ax.set_xlim(0, 3)

    # 4. Synchronization patterns
    ax = axes[1, 1]

    n_probes = 50
    probe_freqs = np.linspace(0.5, 10, n_probes)

    sync_low = probe_cell_with_repertoire(signal_low, n_probes)
    sync_high = probe_cell_with_repertoire(signal_high, n_probes)

    ax.plot(probe_freqs, sync_low, 'o-', color='#E63946', markersize=4,
            linewidth=1, label='Low-D (3 modes)', alpha=0.8)
    ax.plot(probe_freqs, sync_high, 's-', color='#2A9D8F', markersize=4,
            linewidth=1, label='High-D (30 modes)', alpha=0.8)
    ax.set_xlabel('Probe Frequency (Hz)')
    ax.set_ylabel('Phase Synchronization')
    ax.set_title('Repertoire Synchronization Pattern')
    ax.legend()

    # Add annotation
    ax.annotate('Low-D shows\nspiky pattern\n(few resonances)',
                xy=(3, np.max(sync_low)), xytext=(6, 0.6),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, ha='center')

    plt.tight_layout()
    plt.savefig('fig4_dynamics_discrimination.png', dpi=150, bbox_inches='tight')
    plt.savefig('fig4_dynamics_discrimination.pdf', bbox_inches='tight')
    plt.close()

    print("Generated: fig4_dynamics_discrimination.png/pdf")


def compute_summary_stats(results):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("DYNAMICS SIMULATION RESULTS")
    print("="*60)

    for metric in ['spectral_entropy', 'autocorr_decay', 'sync_variance']:
        low_d = np.array(results[metric]['low_d'])
        high_d = np.array(results[metric]['high_d'])

        pooled_std = np.sqrt((np.var(low_d) + np.var(high_d)) / 2)
        if metric == 'autocorr_decay':
            cohens_d = (np.mean(low_d) - np.mean(high_d)) / pooled_std
        else:
            cohens_d = (np.mean(high_d) - np.mean(low_d)) / pooled_std

        # AUC
        from sklearn.metrics import roc_auc_score
        labels = np.concatenate([np.zeros(len(low_d)), np.ones(len(high_d))])
        if metric == 'autocorr_decay':
            scores = np.concatenate([low_d, high_d])
            auc = 1 - roc_auc_score(labels, scores)  # Flip because low_d has higher values
        else:
            scores = np.concatenate([low_d, high_d])
            auc = roc_auc_score(labels, scores)

        print(f"\n{metric}:")
        print(f"  Low-D:  {np.mean(low_d):.3f} ± {np.std(low_d):.3f}")
        print(f"  High-D: {np.mean(high_d):.3f} ± {np.std(high_d):.3f}")
        print(f"  Cohen's d: {cohens_d:.2f}")
        print(f"  AUC: {auc:.3f}")

    print("\n" + "="*60)


def plot_dimensionality_sweep():
    """Show how metrics change continuously with dimensionality."""

    n_timesteps = 2000
    dt = 0.01
    n_samples_per = 50

    mode_range = [2, 3, 5, 8, 12, 20, 30, 50]

    entropy_means = []
    entropy_stds = []
    decay_means = []
    decay_stds = []

    for n_modes in mode_range:
        entropies = []
        decays = []
        for _ in range(n_samples_per):
            signal, t = generate_cell_dynamics(n_timesteps, n_modes, dt)
            entropies.append(compute_spectral_entropy(signal))
            decays.append(compute_autocorrelation_decay(signal))

        entropy_means.append(np.mean(entropies))
        entropy_stds.append(np.std(entropies))
        decay_means.append(np.mean(decays))
        decay_stds.append(np.std(decays))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.errorbar(mode_range, entropy_means, yerr=entropy_stds, fmt='o-',
                capsize=5, color='#264653', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Oscillatory Modes')
    ax.set_ylabel('Spectral Entropy')
    ax.set_title('Complexity Increases with Dimensionality')
    ax.fill_between([0, 8], [0]*2, [1]*2, alpha=0.1, color='#E63946')
    ax.fill_between([8, 55], [0]*2, [1]*2, alpha=0.1, color='#2A9D8F')
    ax.text(4, 0.9, 'Pathogen\nzone', ha='center', fontsize=10, color='#E63946')
    ax.text(30, 0.9, 'Self\nzone', ha='center', fontsize=10, color='#2A9D8F')
    ax.set_xlim(0, 55)

    ax = axes[1]
    ax.errorbar(mode_range, decay_means, yerr=decay_stds, fmt='o-',
                capsize=5, color='#264653', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Oscillatory Modes')
    ax.set_ylabel('Autocorrelation Half-life')
    ax.set_title('Predictability Decreases with Dimensionality')
    ax.fill_between([0, 8], [0]*2, [100]*2, alpha=0.1, color='#E63946')
    ax.fill_between([8, 55], [0]*2, [100]*2, alpha=0.1, color='#2A9D8F')
    ax.set_xlim(0, 55)

    plt.tight_layout()
    plt.savefig('fig5_dimensionality_sweep.png', dpi=150, bbox_inches='tight')
    plt.savefig('fig5_dimensionality_sweep.pdf', bbox_inches='tight')
    plt.close()

    print("Generated: fig5_dimensionality_sweep.png/pdf")


if __name__ == "__main__":
    print("VDJ Dynamics Simulation")
    print("-" * 40)

    results = run_discrimination_experiment()
    plot_results(results)
    compute_summary_stats(results)
    plot_dimensionality_sweep()

    print("\nDone!")
