import os
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.data_loader import get_mnist_dataset
from src.qnn_model import QNNClassifier, Config

# ---------------------- Logging ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------- Train Function ----------------------
def train_model():
    """
    Trains the QNNClassifier on MNIST (10 digits), tracks per-batch/epoch metrics,
    and saves the trained model in reports/qnn_trained_model.pt.
    """

    # === Project Config ===
    try:
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
    except IndexError:
        PROJECT_ROOT = Path.cwd()

    cfg = Config()
    logger.info(f"Configuration: {cfg}")

    # === Load Full MNIST Dataset ===
    logger.info("📦 Loading and preprocessing MNIST dataset...")
    images, labels = get_mnist_dataset(
        data_dir=PROJECT_ROOT / "datasets",
        binary_digits=tuple(range(10)),  # All digits: 0–9
        image_size=cfg.IMAGE_SIZE,
        train=True,
        download=True,
        limit_samples=cfg.LIMIT_SAMPLES
    )

    logger.info(f"✅ Dataset loaded - Images: {images.shape}, Labels: {labels.shape}")

    # === Train/Validation Split ===
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=42
    )
    logger.info(f"🧠 Train size: {X_train.shape}, Validation size: {X_val.shape}")

    # === DataLoaders ===
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # keep 0 for CPU (adjust if GPU or dataloader hangs)
        drop_last=True,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    logger.info(f"📊 Batches - Train: {len(train_loader)}, Val: {len(val_loader)}")

    # === Model, Optimizer, Loss ===
    device = torch.device("cpu")  # Change to "cuda" if GPU is available
    model = QNNClassifier(cfg.NUM_QUBITS, cfg.NUM_CLASSES, cfg.NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    criterion = torch.nn.CrossEntropyLoss()

    # === Training Loop ===
    logger.info(f"🚀 Training on device: {device} for {cfg.EPOCHS} epochs")
    for epoch in range(cfg.EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            try:
                x_batch = x_batch.to(device, non_blocking=True).double()
                y_batch = y_batch.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                logger.info(
                    f"[TRAIN] Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

            except Exception as e:
                logger.exception(f"❌ Error in Epoch {epoch+1}, Batch {batch_idx+1}")
                raise e

        avg_train_loss = epoch_loss / len(train_loader)
        logger.info(f"📈 [TRAIN] Epoch {epoch+1}/{cfg.EPOCHS} - Avg Loss: {avg_train_loss:.4f}")

        # === Validation Phase ===
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
                x_batch = x_batch.to(device).double()
                y_batch = y_batch.to(device)

                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

                logger.info(
                    f"[VAL] Epoch {epoch+1}, Batch {batch_idx+1}/{len(val_loader)} | "
                    f"Batch Loss: {loss.item():.4f} | Batch Accuracy: {(preds == y_batch).float().mean()*100:.2f}%"
                )

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total * 100.0

        logger.info(
            f"✅ [VAL] Epoch {epoch+1}/{cfg.EPOCHS} | Avg Loss: {avg_val_loss:.4f} | "
            f"Overall Accuracy: {val_accuracy:.2f}%"
        )

    # === Save Model ===
    reports_path = PROJECT_ROOT / "reports"
    os.makedirs(reports_path, exist_ok=True)
    model_path = reports_path / "qnn_trained_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"💾 Model saved to: {model_path}")

# ---------------------- CLI ----------------------
if __name__ == "__main__":
    logger.info("=== QNN Training Pipeline Started ===")
    train_model()
    logger.info("=== Training Completed ===")
