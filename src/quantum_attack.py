"""
Quantum Adversarial Attack Utilities for QNN Robustness Evaluation

This module provides quantum adversarial attacks and noise injection methods:
- Depolarizing Noise
- Amplitude Damping Noise
- Random Pauli-X Noise

Author: Harsh Gupta
"""

import pennylane as qml
import numpy as np

# ==============================
#  Depolarizing Noise
# ==============================
def inject_depolarizing_noise(wires, prob=0.05):
    """
    Approximates depolarizing noise with random Pauli rotations for lightning.qubit compatibility
    Args:
        wires (list or range): List of qubit wires
        prob (float): Depolarization probability
    """
    for w in wires:
        if np.random.random() < prob / 3:
            qml.RX(np.pi, wires=w)  # Pauli X
        if np.random.random() < prob / 3:
            qml.RY(np.pi, wires=w)  # Pauli Y
        if np.random.random() < prob / 3:
            qml.RZ(np.pi, wires=w)  # Pauli Z

# ==============================
#  Amplitude Damping Noise
# ==============================
def inject_amplitude_damping(wires, gamma=0.05):
    """
    Approximates amplitude damping with rotations and CNOTs for lightning.qubit compatibility
    Args:
        wires (list or range): List of qubit wires
        gamma (float): Damping rate
    """
    for idx, w in enumerate(wires):
        theta = np.arccos(np.sqrt(1 - gamma))
        qml.RY(theta, wires=w)
        next_wire = (idx + 1) % len(wires)
        qml.CNOT(wires=[w, next_wire])

# ==============================
#  Random Pauli-X Noise
# ==============================
def inject_random_pauli_x_noise(wires, prob=0.05):
    """
    Applies RX(pi) with probability prob to simulate bit flip-like noise
    Args:
        wires (list or range): List of qubit wires
        prob (float): Probability of applying RX(pi)
    """
    for w in wires:
        if np.random.random() < prob:
            qml.RX(np.pi, wires=w)  # RX(pi) = Pauli X