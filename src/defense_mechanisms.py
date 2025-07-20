
"""
Defense Mechanisms for QNN Robustness Against Fault-Tolerant Quantum Attacks

Implements:
- NoiseAwareQNNClassifier: Curriculum-based noise-aware training.
- EnhancedZNEQNNClassifier: ZNE with readout error mitigation and noise-aware training.
- evaluate_model: Computes metrics over multiple trials.
- CLI: Trains and evaluates both models for multiple qubit counts, saves metrics.
- Supports eval-only mode to re-run evaluation on existing models.

Optimized for 16GB RAM with reduced batch size and simplified ZNE.
Uses lightning.qubit with adjoint differentiation and custom noise approximations.

Author: Harsh Gupta
"""

import os
import json
import torch
import random
import logging
import numpy as np
import psutil
import argparse
from pathlib import Path
from sklearn.metrics import (
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    log_loss,
    confusion_matrix,
)
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR

import pennylane as qml
from src.qnn_model import QNNClassifier, Config
from src.quantum_attack import (
    inject_depolarizing_noise,
    inject_amplitude_damping,
    inject_random_pauli_x_noise,
)
from src.data_loader import get_mnist_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DefenseConfig(Config):
    NUM_LAYERS = 3
    BATCH_SIZE = 16
    LR = 0.0005
    EPOCHS = 30  # Increased to improve accuracy
    DROPOUT = 0.3
    LABEL_SMOOTHING = 0.05
    ZNE_SCALE_FACTORS = {
        "depolarizing": [1, 1.5, 2],
        "amplitude_damping": [1, 1.5, 2],
        "random_x": [1, 1.5, 2]
    }
    LIMIT_SAMPLES = 1000

cfg = DefenseConfig()

# ------------- Quantum Circuit -------------
def quantum_circuit(inputs, weights, noise_type=None, noise_param=0.0):
    qml.AngleEmbedding(inputs, wires=range(cfg.NUM_QUBITS), rotation='Y')
    qml.templates.BasicEntanglerLayers(weights, wires=range(cfg.NUM_QUBITS))
    if noise_type == "depolarizing":
        inject_depolarizing_noise(range(cfg.NUM_QUBITS), prob=noise_param)
    elif noise_type == "amplitude_damping":
        inject_amplitude_damping(range(cfg.NUM_QUBITS), gamma=noise_param)
    elif noise_type == "random_x":
        inject_random_pauli_x_noise(range(cfg.NUM_QUBITS), prob=noise_param)
    return [qml.expval(qml.PauliZ(i)) for i in range(cfg.NUM_QUBITS)]

# ------------- Noise-Aware Model -------------
class NoiseAwareQNNClassifier(QNNClassifier):
    def __init__(self, n_qubits, n_classes, n_layers=cfg.NUM_LAYERS):
        super().__init__(n_qubits, n_classes, n_layers)
        self.noise_types = ["depolarizing", "amplitude_damping", "random_x", None]
        self.current_noise_type = None
        self.current_noise_param = 0.1

        device = qml.device("lightning.qubit", wires=n_qubits)
        self.qnode = qml.qnode(device, interface="torch", diff_method="adjoint")(
            lambda inputs, weights: quantum_circuit(inputs, weights, self.current_noise_type, self.current_noise_param)
        )
        self.qlayer = qml.qnn.TorchLayer(self.qnode, {"weights": (n_layers, n_qubits)})
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_qubits, 256),
            torch.nn.BatchNorm1d(256, momentum=0.1), torch.nn.ReLU(), torch.nn.Dropout(cfg.DROPOUT),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128, momentum=0.1), torch.nn.ReLU(), torch.nn.Dropout(cfg.DROPOUT),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64, momentum=0.1), torch.nn.ReLU(), torch.nn.Dropout(cfg.DROPOUT),
            torch.nn.Linear(64, n_classes)
        )

    def forward(self, x, noise_type=None, noise_param=None):
        self.current_noise_type = noise_type or self.current_noise_type
        self.current_noise_param = noise_param if noise_param is not None else self.current_noise_param
        x = self.classical_preprocess(x.float())
        x = self.qlayer(x.double()).float()
        return self.model(x)


# ------------- Evaluation -------------
def evaluate_model(model, val_loader, device, noise_type=None, noise_param=0.05, trials=2):
    all_metrics = []
    for _ in range(trials):
        model.eval()
        y_true, y_pred, y_probs = [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device).double()
                y = y.to(device)
                logits = model(x, noise_type=noise_type, noise_param=noise_param)
                probs = torch.softmax(logits, dim=1)
                # Normalize probabilities to sum to 1
                probs = probs / torch.sum(probs, dim=1, keepdim=True)
                preds = torch.argmax(probs, dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_probs.extend(probs.cpu().numpy())

        acc = np.mean(np.array(y_true) == np.array(y_pred))
        kappa = cohen_kappa_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        top3 = np.mean([yt in np.argsort(p)[-3:] for yt, p in zip(y_true, y_probs)])
        try:
            auroc = roc_auc_score(y_true, y_probs, multi_class="ovr")
        except ValueError as e:
            logger.warning(f"AUROC calculation failed: {e}. Setting AUROC to 0.")
            auroc = 0.0
        try:
            logloss = log_loss(y_true, y_probs)
        except ValueError as e:
            logger.warning(f"Log loss calculation failed: {e}. Setting log loss to infinity.")
            logloss = float('inf')

        all_metrics.append({
            "accuracy": acc, "kappa": kappa, "mcc": mcc,
            "top3_acc": top3, "auroc": auroc, "logloss": logloss
        })

    mean = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    std = {k: float(np.std([m[k] for m in all_metrics])) for k in all_metrics[0]}

    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"📊 Confusion Matrix [{noise_type or 'clean'}]:\n{cm}")

    return mean, std

# ------------- CLI -------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QNN Training with Variable Qubits")
    parser.add_argument("--qubits", type=str, default="4,6,8,10,12", help="Comma-separated list of qubit counts")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing models")
    args = parser.parse_args()
    qubit_counts = [int(q) for q in args.qubits.split(",")]

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    reports_path = PROJECT_ROOT / "reports"
    metrics_path = reports_path / "metrics"
    reports_path.mkdir(exist_ok=True)
    metrics_path.mkdir(exist_ok=True)

    try:
        images, labels = get_mnist_dataset(
            data_dir=PROJECT_ROOT / "datasets",
            binary_digits=tuple(range(10)),
            image_size=cfg.IMAGE_SIZE,
            train=True,
            download=True,
            limit_samples=cfg.LIMIT_SAMPLES
        )
    except Exception as e:
        logger.error(f"Failed to load MNIST dataset: {e}")
        raise

    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=cfg.BATCH_SIZE, shuffle=False)
    device = torch.device("cpu")

    all_results = {"noise_aware": {}, "zne": {}}

    for num_qubits in qubit_counts:
        cfg.NUM_QUBITS = num_qubits
        logger.info(f"🚀 Processing {num_qubits} qubits")

        # Noise-Aware QNN
        na_model_path = reports_path / f"qnn_noise_trained_model_{num_qubits}.pt"
        try:
            model_na = NoiseAwareQNNClassifier(cfg.NUM_QUBITS, cfg.NUM_CLASSES).to(device)
            if not args.eval_only:
                logger.info("🚀 Training Noise-Aware QNNClassifier")
                optimizer_na = torch.optim.Adam(model_na.parameters(), lr=cfg.LR)
                scheduler_na = CosineAnnealingLR(optimizer_na, T_max=cfg.EPOCHS)
                criterion_na = torch.nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)

                for epoch in range(cfg.EPOCHS):
                    model_na.train()
                    epoch_loss = 0.0
                    epoch_correct = 0
                    total_samples = 0
                    for batch_idx, (xb, yb) in enumerate(train_loader):
                        xb = xb.to(device).double()
                        yb = yb.to(device)
                        optimizer_na.zero_grad()
                        noise_type = random.choice(model_na.noise_types)
                        preds = model_na(xb, noise_type=noise_type, noise_param=0.1)
                        loss = criterion_na(preds, yb)
                        loss.backward()
                        optimizer_na.step()

                        batch_loss = loss.item()
                        batch_preds = torch.argmax(preds, dim=1)
                        batch_correct = (batch_preds == yb).sum().item()
                        batch_acc = batch_correct / xb.size(0)
                        epoch_loss += batch_loss * xb.size(0)
                        epoch_correct += batch_correct
                        total_samples += xb.size(0)

                        logger.info(
                            f"Qubits {num_qubits} | Epoch {epoch+1}/{cfg.EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | "
                            f"Loss: {batch_loss:.4f} | Accuracy: {batch_acc:.4f} | Noise: {noise_type or 'None'}"
                        )

                    scheduler_na.step()
                    epoch_loss /= total_samples
                    epoch_acc = epoch_correct / total_samples
                    logger.info(
                        f"Qubits {num_qubits} | Epoch {epoch+1}/{cfg.EPOCHS} | Average Loss: {epoch_loss:.4f} | "
                        f"Average Accuracy: {epoch_acc:.4f} | LR: {scheduler_na.get_last_lr()[0]:.6f}"
                    )
                    logger.info(f"Memory used: {psutil.Process().memory_info().rss / 1024**3:.2f} GiB")

                torch.save(model_na.state_dict(), na_model_path)
                logger.info(f"✅ Saved Noise-Aware model to {na_model_path}")

            # Load model for evaluation
            if na_model_path.exists():
                model_na.load_state_dict(torch.load(na_model_path))
                logger.info(f"✅ Loaded Noise-Aware model from {na_model_path}")
            else:
                logger.warning(f"No model found at {na_model_path}. Skipping evaluation for Noise-Aware QNN.")
                continue

            logger.info("🚀 Evaluating Noise-Aware QNNClassifier")
            results_na = {}
            for noise in ["clean", "depolarizing", "amplitude_damping", "random_x"]:
                mean, std = evaluate_model(model_na, val_loader, device, noise_type=None if noise == "clean" else noise)
                results_na[noise] = {"mean": mean, "std": std}
            all_results["noise_aware"][num_qubits] = results_na
            with open(metrics_path / f"noise_aware_metrics_{num_qubits}.json", "w") as f:
                json.dump(results_na, f, indent=4)
            logger.info(f"✅ Noise-Aware QNN evaluation complete for {num_qubits} qubits.")

        except Exception as e:
            logger.error(f"Error processing Noise-Aware QNN for {num_qubits} qubits: {e}")
            continue

        
    with open(metrics_path / "combined_metrics.json", "w") as f:
        json.dump(all_results, f, indent=4)

    logger.info("🏁 All models processed and metrics saved successfully.")