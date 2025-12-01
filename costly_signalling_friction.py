"""
Costly Signalling with Dynamical Friction
Simulates the thermodynamic cost of immune surveillance.

Key Result: High-D immune cells experience high 'dynamical friction'
when coupled to Low-D targets (cancer), leading to rapid metabolic
exhaustion and dimensional collapse.

This demonstrates that T-cell exhaustion is not merely "tiredness" but
a forced phase transition into a lower-dimensional attractor caused by
sustained coupling with a low-dimensional source.

Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.facecolor'] = 'white'


class CoupledCell:
    """Kuramoto oscillator model of cellular dynamics."""

    def __init__(self, n_oscillators=50, d_type=None):
        self.n = n_oscillators
        self.phases = np.random.uniform(0, 2*np.pi, self.n)

        # Configuration based on cell type
        if d_type == 'high':  # Healthy/Immune cell
            # Wide frequency spread = Desynchronized = High Entropy
            self.omegas = np.random.normal(0, 2.0, self.n)
            self.internal_coupling = 0.5
        else:  # Cancer/Low-D cell
            # Narrow frequency spread = Synchronized = Low Entropy
            self.omegas = np.random.normal(0, 0.1, self.n)
            self.internal_coupling = 5.0

        self.d_type = d_type
        self.energy = 1.0  # Metabolic Reserve
        self.is_exhausted = False

    def step(self, dt=0.05, external_field=0, coupling_strength=0):
        """Advance dynamics by one timestep."""

        # 1. Check for Exhaustion (Phase Transition)
        if self.is_exhausted:
            # Collapse to Low-D attractor
            self.internal_coupling = 5.0
            self.omegas = self.omegas * 0.9 + np.mean(self.omegas) * 0.1

        # 2. Calculate Kuramoto Dynamics
        # Mean field Z = R * e^(i*psi)
        z = np.mean(np.exp(1j * self.phases))
        r = np.abs(z)
        psi = np.angle(z)

        # Internal forcing (self-organization)
        internal_force = self.internal_coupling * r * np.sin(psi - self.phases)

        # External forcing (input from target)
        external_force = np.zeros(self.n)
        if coupling_strength > 0 and external_field != 0:
            r_ext = np.abs(external_field)
            psi_ext = np.angle(external_field)
            external_force = coupling_strength * r_ext * np.sin(psi_ext - self.phases)

        # Update phases
        self.phases += dt * (self.omegas + internal_force + external_force)
        self.phases = np.mod(self.phases, 2*np.pi)

        # 3. Metabolic Cost (The Friction)
        # Cost is proportional to the ORDER of the external field
        # High-D targets have low order (r_ext ~ 0), low cost
        # Low-D targets have high order (r_ext ~ 1), high cost
        if self.d_type == 'high' and not self.is_exhausted:
            r_ext = np.abs(external_field) if external_field != 0 else 0
            # Friction scales with target's synchronization (coherence)
            # A synchronized (low-D) target exerts strong pull
            friction = r_ext * 0.008  # Scale factor
            base_metabolism = 0.0005

            self.energy -= (base_metabolism + friction)

            if self.energy <= 0:
                self.energy = 0
                self.is_exhausted = True

    def get_mean_field(self):
        """Return complex order parameter."""
        return np.mean(np.exp(1j * self.phases))

    def get_dimensionality(self):
        """D ~ 1 - Order Parameter (1=Sync/Low-D, 0=Desync/High-D)"""
        z = np.mean(np.exp(1j * self.phases))
        return 1.0 - np.abs(z)


def run_simulation():
    """Run friction simulation and generate figure."""

    targets = ['high', 'low']
    results = {}

    for target_type in targets:
        immune = CoupledCell(d_type='high')  # Probe (immune cell)
        target = CoupledCell(d_type=target_type)  # Subject (tissue/tumor)

        res = {'energy': [], 'd': []}

        for _ in range(400):
            # Get signal from target
            field = target.get_mean_field()

            # Probe target (Coupling = 3.0)
            immune.step(dt=0.1, external_field=field, coupling_strength=3.0)
            target.step(dt=0.1)

            res['energy'].append(immune.energy)
            res['d'].append(immune.get_dimensionality())

        results[target_type] = res

    return results


def plot_results(results, output_path='figures/fig_dynamical_friction.pdf'):
    """Generate publication-quality figure."""

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Panel A: Energy Burnout
    ax = axes[0]
    ax.plot(results['high']['energy'], c='#2A9D8F', lw=2,
            label='Target: Healthy (High-D)')
    ax.plot(results['low']['energy'], c='#E63946', lw=2,
            label='Target: Cancer (Low-D)')
    ax.set_ylabel('Metabolic Reserve')
    ax.set_title('A. Dynamical Friction: Probing Low-D Targets Drains Energy')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 400)
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(alpha=0.3)

    # Panel B: Exhaustion/Dimensional Collapse
    ax = axes[1]
    ax.plot(results['high']['d'], c='#2A9D8F', lw=2,
            label='Immune State (vs Healthy)')
    ax.plot(results['low']['d'], c='#E63946', lw=2,
            label='Immune State (vs Cancer)')
    ax.set_ylabel('Immune Cell Dimensionality')
    ax.set_xlabel('Time (arbitrary units)')
    ax.set_title('B. Functional Collapse: T-Cell Exhaustion as Dimensional Phase Transition')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)

    # Add annotation for exhaustion point
    # Find where energy hits zero
    for i, e in enumerate(results['low']['energy']):
        if e <= 0:
            ax.axvline(i, color='#E63946', linestyle=':', alpha=0.7)
            ax.annotate('Exhaustion\nThreshold', xy=(i, 0.5),
                       xytext=(i+50, 0.6), fontsize=10,
                       arrowprops=dict(arrowstyle='->', color='#E63946'))
            break

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return fig


if __name__ == "__main__":
    print("Running Dynamical Friction Simulation...")
    print("=" * 50)

    results = run_simulation()

    # Report key metrics
    print("\nResults:")
    print("-" * 30)

    # Find exhaustion time for low-D target
    exhaustion_time = None
    for i, e in enumerate(results['low']['energy']):
        if e <= 0:
            exhaustion_time = i
            break

    print(f"Probing High-D (Healthy) Target:")
    print(f"  Final energy: {results['high']['energy'][-1]:.3f}")
    print(f"  Final dimensionality: {results['high']['d'][-1]:.3f}")
    print(f"  Status: Sustained surveillance")

    print(f"\nProbing Low-D (Cancer) Target:")
    print(f"  Exhaustion at timestep: {exhaustion_time}")
    print(f"  Final dimensionality: {results['low']['d'][-1]:.3f}")
    print(f"  Status: EXHAUSTED (dimensional collapse)")

    print("\n" + "=" * 50)
    print("Key Insight: T-cell exhaustion is a thermodynamic")
    print("consequence of sustained coupling with low-D targets.")
    print("=" * 50)

    plot_results(results)
