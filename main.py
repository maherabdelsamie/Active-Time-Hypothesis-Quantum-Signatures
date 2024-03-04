import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Constants and Time Range
E_0 = 1.0  # Baseline energy
kappa = 0.1  # Proportionality factor
omega_0 = 1.0  # Baseline frequency
phi_0 = 0.0  # Initial phase
t = np.linspace(0, 10, 1000)  # Time array

# Generative function g(t'), simulating periodic generative effects of time
def g(t, active=True):
    return np.sin(t) if active else np.zeros_like(t)

# Function to compute alpha(t') = kappa * g(t')
def alpha(t, active=True):
    return kappa * g(t, active)

# Function to calculate delta E(t') = E_0 * (1 + alpha(t')) - E_0
def delta_E(t, active=True):
    if active:
        return E_0 * (1 + alpha(t, True)) - E_0
    else:
        return np.zeros_like(t)  # Represents no change in energy without generative influence

# Function for Omega(t'') = omega_0 + Theta(t'')
def Omega(t, active=True):
    Theta = 0.5 * np.cos(t) if active else np.zeros_like(t)
    return omega_0 + Theta

# Function to compute phi(t') by integrating Omega(t'') over time
def phi(t, active=True):
    integral, _ = quad(lambda x: Omega(x, active), 0, t)
    return phi_0 + integral

# Generating data for plots
energy_variations_active = delta_E(t, active=True)
energy_variations_inactive = delta_E(t, active=False)
phase_evolution_active = np.array([phi(ti, active=True) for ti in t])
phase_evolution_inactive = np.array([phi(ti, active=False) for ti in t])
quantum_state_evolution_active = np.exp(1j * phase_evolution_active)
quantum_state_evolution_inactive = np.exp(1j * phase_evolution_inactive)

# Plotting
plt.figure(figsize=(14, 6))

# Energy Variations Plot
plt.subplot(1, 2, 1)
plt.plot(t, energy_variations_active, label='With Generative Influence')
plt.plot(t, energy_variations_inactive, label='Without Generative Influence', linestyle='--')
plt.xlabel('Time $t\'$')
plt.ylabel('Energy Variation $\delta E(t\')$')
plt.title('Energy Variations Under Time\'s Influence')
plt.legend()

# Quantum State Evolution Plot
plt.subplot(1, 2, 2)
plt.plot(t, np.real(quantum_state_evolution_active), label='Real Part with Generative Influence')
plt.plot(t, np.real(quantum_state_evolution_inactive), label='Real Part without Generative Influence', linestyle='--')
plt.plot(t, np.imag(quantum_state_evolution_active), label='Imaginary Part with Generative Influence', linestyle=':')
plt.plot(t, np.imag(quantum_state_evolution_inactive), label='Imaginary Part without Generative Influence', linestyle='-.')
plt.xlabel('Time $t\'$')
plt.ylabel('Quantum State Evolution $\psi(t\')$')
plt.title('Quantum State Evolution Under Time\'s Influence')
plt.legend()

plt.tight_layout()
plt.show()

# Quantification of Differences (Optional)
energy_diff = np.abs(energy_variations_active - energy_variations_inactive)
phase_diff = np.abs(phase_evolution_active - phase_evolution_inactive)

print(f"Maximum energy variation difference: {np.max(energy_diff)}")
print(f"Maximum phase evolution difference: {np.max(phase_diff)}")
