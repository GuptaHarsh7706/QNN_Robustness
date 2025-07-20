import torch
import pennylane as qml
from pennylane import numpy as np
import logging
from dataclasses import dataclass
import time

# ------------------ Configuration ------------------
@dataclass
class Config:
    NUM_QUBITS: int = 16
    NUM_CLASSES: int = 10
    NUM_LAYERS: int = 6
    DEVICE_NAME: str = "lightning.qubit"  # Use "default.qubit" if GPU fails or "default.mixed" if RAM>64GB
    IMAGE_SIZE: int = 28
    BATCH_SIZE: int = 32
    LR: float = 0.001
    EPOCHS: int = 20
    LIMIT_SAMPLES: int = 1000

cfg = Config()

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ------------------ Quantum Circuit ------------------
DEVICE = qml.device(cfg.DEVICE_NAME, wires=cfg.NUM_QUBITS)

def quantum_circuit(inputs, weights):
    """Quantum circuit with angle embedding and entanglement"""
    qml.AngleEmbedding(inputs, wires=range(cfg.NUM_QUBITS), rotation='Y')
    qml.templates.BasicEntanglerLayers(weights, wires=range(cfg.NUM_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(cfg.NUM_QUBITS)]

@qml.qnode(DEVICE, interface="torch", diff_method="adjoint")#Change to parameter-shift if using default.mixed
def qnode(inputs, weights):
    return quantum_circuit(inputs, weights)

# ------------------ QNN Classifier ------------------
class QNNClassifier(torch.nn.Module):
    def __init__(self, n_qubits: int, n_classes: int, n_layers: int):
        super().__init__()
        self.n_qubits = n_qubits

        # QNode converted into TorchLayer with learnable weights
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # Linear preprocessor to compress 784D → 16D
        self.classical_preprocess = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(cfg.IMAGE_SIZE * cfg.IMAGE_SIZE, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, n_qubits),
            torch.nn.ReLU()
        )

        # Classical classifier head
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_qubits, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, n_classes)
        )

        # Weight initialization
        for m in self.model:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x ∈ [B, 1, 28, 28]
        Returns: logits ∈ [B, 10]
        """
        x = x.float()
        x = self.classical_preprocess(x)               # [B, 784] → [B, 16]
        x = self.qlayer(x.double())                    # Run quantum circuit on CPU
        x = x.float()                                  # Convert to float32
        x = x.to(next(self.model.parameters()).device) # Match model's device
        return self.model(x)

# ------------------ CLI Test Block ------------------
if __name__ == "__main__":
    logger.info("--- Starting QNN Model Test ---")
    logger.info(f"Configuration: {cfg}")
    logger.info(f"Using PennyLane device: {cfg.DEVICE_NAME}")

    model = QNNClassifier(
        n_qubits=cfg.NUM_QUBITS,
        n_classes=cfg.NUM_CLASSES,
        n_layers=cfg.NUM_LAYERS
    )
    model.eval()
    logger.info("✅ Model initialized successfully.")

    dummy_input = torch.rand(2, 1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    with torch.no_grad():
        start_time = time.time()
        logits = model(dummy_input)
        end_time = time.time()
        preds = torch.argmax(logits, dim=1)

    duration = (end_time - start_time) * 1000
    logger.info(f"✅ Forward pass on batch of size {dummy_input.size(0)} took: {duration:.2f} ms")
    logger.info(f"Logits shape: {logits.shape} | Predictions: {preds.tolist()}")
    logger.info("--- QNN Test Completed ---")
