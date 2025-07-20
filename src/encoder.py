import torch
import pennylane as qml
import numpy as np
import logging

# ------------------ Logging Configuration ------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ------------------ Constants ------------------
NUM_QUBITS = 16  # Match qnn_model.py

# ------------------ Quantum Device Initialization ------------------
def initialize_device(n_qubits: int = NUM_QUBITS) -> qml.device:
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)  # Use lightning.qubit
        logger.info("Initialized 'lightning.qubit' device with %d qubits.", n_qubits)
        return dev
    except Exception as e:
        logger.warning("Fallback to 'default.qubit' due to error: %s", e)
        return qml.device("default.qubit", wires=n_qubits)

dev = initialize_device()

# ------------------ Angle Encoding ------------------
def angle_encoding(inputs: torch.Tensor, wires: range) -> None:
    for i in range(NUM_QUBITS):
        qml.RY(np.pi * inputs[i], wires=i)

# ------------------ Quantum Circuit for Expectation ------------------
@qml.qnode(dev, interface="torch", diff_method="adjoint", dtype=torch.float64)
def quantum_feature_map(inputs: torch.Tensor) -> torch.Tensor:
    """
    Applies angle encoding and returns expectation values ⟨Z⟩ for each qubit.
    Supports single input of shape [NUM_QUBITS].
    """
    if inputs.shape[-1] != NUM_QUBITS:
        raise ValueError(f"Expected inputs of size {NUM_QUBITS}, got {inputs.shape[-1]}")
    angle_encoding(inputs, wires=range(NUM_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]

# ------------------ Encoded State (Expectations) ------------------
def get_encoded_state(inputs: torch.Tensor) -> torch.Tensor:
    """
    Encodes inputs by executing the QNode and returns expectation values.
    Inputs: [batch_size, NUM_QUBITS] from CNN preprocessing.
    """
    if inputs.dim() == 1:
        inputs = inputs.unsqueeze(0)  # Ensure batch dimension

    if inputs.shape[1] != NUM_QUBITS:
        raise ValueError(f"Expected input with NUM_QUBITS={NUM_QUBITS}, got {inputs.shape[1]}")

    if inputs.min() < 0 or inputs.max() > 1:
        inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())

    expectations = []
    for i in range(inputs.shape[0]):
        exp_vals = quantum_feature_map(inputs[i])

        # ✳️ Fix: If exp_vals is a list of floats → convert to tensor
        if isinstance(exp_vals, list):
            exp_vals = torch.tensor(exp_vals, dtype=torch.float64)

        expectations.append(exp_vals)

    stacked = torch.stack(expectations)  # [batch_size, NUM_QUBITS]
    logger.info("Computed expectation values for batch of shape %s", stacked.shape)
    return stacked

# ------------------ Visualization ------------------
@qml.qnode(dev, interface="torch")
def visualization_circuit(inputs: torch.Tensor) -> None:
    if inputs.dim() == 1:
        inputs = inputs.unsqueeze(0)
    for b in range(inputs.shape[0]):
        angle_encoding(inputs[b], wires=range(NUM_QUBITS))
    return None

def visualize_circuit(inputs: torch.Tensor) -> None:
    drawer = qml.draw(visualization_circuit)
    print(drawer(inputs))

# ------------------ CLI Test Mode ------------------
if __name__ == "__main__":
    logger.info("Testing encoder module with dummy input...")

    TEST_QUBITS = 16
    test_input = torch.rand(2, TEST_QUBITS).double()  # Simulate CNN output

    try:
        test_dev = initialize_device(TEST_QUBITS)

        @qml.qnode(test_dev, interface="torch", diff_method="adjoint", dtype=torch.float64)
        def single_sample_circuit(inputs: torch.Tensor) -> torch.Tensor:
            angle_encoding(inputs, wires=range(TEST_QUBITS))
            return [qml.expval(qml.PauliZ(i)) for i in range(TEST_QUBITS)]

        def test_qnode(inputs: torch.Tensor) -> torch.Tensor:
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)  # Add batch dimension
            batch_size = inputs.shape[0]
            expectations = []
            for b in range(batch_size):
                exp_vals = torch.tensor(single_sample_circuit(inputs[b]), dtype=torch.float64)
                expectations.append(exp_vals)

            return torch.stack(expectations)  # [batch_size, TEST_QUBITS]

        expectations = test_qnode(test_input)
        logger.info("Test successful. Expectation values shape: %s", expectations.shape)
        logger.info("Sample expectations: %s", expectations[0])
    except Exception as e:
        logger.error("Test run failed: %s", e)
