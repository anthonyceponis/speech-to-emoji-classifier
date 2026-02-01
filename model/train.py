import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
import logging
import random
import metrax
from pathlib import Path
from typing import Tuple, List

# CHANGE 1: Import the new CNNModel
from model import CNNModel
from dataset import construct_dataset, split_dataset

# --- Setup Logging ---
logger = logging.getLogger("model_training")
logger.setLevel(logging.INFO)

# --- Randomness ---
seed = 42
random.seed(seed)
key = jax.random.PRNGKey(seed)


# --- Data Loader ---
def data_loader(dataset: List, batch_size: int, shuffle: bool = True):
    """Generates batches from the dataset."""
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        batch = [dataset[idx] for idx in batch_indices]

        # Stack samples:
        # X -> (Batch, 98, 40) - 98 time frames, 40 mel bands
        # y -> (Batch, Classes)
        X_batch = jnp.stack([item[0] for item in batch])
        y_batch = jnp.stack([item[1] for item in batch])

        yield X_batch, y_batch


@jax.jit
def train_step(state, batch, dropout_key):
    X, y = batch

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            X,
            training=True,
            rngs={"dropout": dropout_key},
        )
        loss = optax.softmax_cross_entropy(logits, y).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def infer_step(state, batch):
    X, y = batch
    logits = state.apply_fn({"params": state.params}, X, training=False)

    loss = optax.softmax_cross_entropy(logits, y).mean()
    preds = jax.nn.softmax(logits)
    pred_classes = jnp.argmax(preds, axis=-1)
    true_classes = jnp.argmax(y, axis=-1)

    return loss, pred_classes, true_classes


def log_metrics(metrics: dict[str, float]):
    results = [f"{k}: {v:.4f}" for k, v in metrics.items()]
    logger.info(" | ".join(results))


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    dataset = construct_dataset()
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    # 2. Get Output Dim
    with open("label_names.txt", "r") as f:
        output_dim = len(f.readlines())

    # 3. Initialize Model
    key, init_key = jax.random.split(key)

    # CHANGE 2: Instantiate the CNN Model
    model = CNNModel(output_dim)

    # CHANGE 3: Update dummy input to match 2D Spectrogram shape (Batch, Time, Features)
    # 98 is the TARGET_FRAMES we set in dataset.py
    dummy_input = jnp.zeros((1, 98, 40))

    # Initialize params
    params = model.init(init_key, dummy_input, training=False)["params"]

    # 4. Config Optimizer
    lr = 0.001
    opt = optax.adamw(lr, weight_decay=1e-4)

    # 5. Setup State
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt)

    # 6. Training Loop
    BATCH_SIZE = 32
    epochs = 1000
    patience = 10
    best_val_loss = float("inf")
    ckpt_dir = Path("checkpoints").resolve()

    for epoch in range(1, epochs + 1):
        # -- Train --
        epoch_train_loss = 0.0
        train_batches = 0

        for X_batch, y_batch in data_loader(train_dataset, BATCH_SIZE, shuffle=True):
            # Generate dropout key
            key, dropout_key = jax.random.split(key)

            state, loss = train_step(state, (X_batch, y_batch), dropout_key)
            epoch_train_loss += loss
            train_batches += 1

        avg_train_loss = epoch_train_loss / train_batches

        # -- Validation --
        epoch_val_loss = 0.0
        val_batches = 0
        val_acc_tracker = metrax.Accuracy.empty()

        for X_batch, y_batch in data_loader(val_dataset, BATCH_SIZE, shuffle=False):
            loss, pred_classes, true_classes = infer_step(state, (X_batch, y_batch))
            epoch_val_loss += loss
            val_batches += 1

            val_acc_tracker = val_acc_tracker.merge(
                metrax.Accuracy.from_model_output(
                    predictions=pred_classes, labels=true_classes
                )
            )

        avg_val_loss = epoch_val_loss / val_batches
        epoch_val_accuracy = val_acc_tracker.compute()

        # -- Checkpointing --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 10
            checkpoints.save_checkpoint(
                ckpt_dir=ckpt_dir,
                target=state,
                step=epoch,
                overwrite=True,
            )
        else:
            patience -= 1
            if patience == 0:
                logger.info("Early stopping triggered!")
                break

        log_metrics(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_acc": epoch_val_accuracy,
                "patience": patience,
            }
        )

    # 7. Testing
    print("\nTESTING")
    print("-----------")
    test_acc_tracker = metrax.Accuracy.empty()

    state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)

    for X_batch, y_batch in data_loader(test_dataset, BATCH_SIZE, shuffle=False):
        _, pred_classes, true_classes = infer_step(state, (X_batch, y_batch))
        test_acc_tracker = test_acc_tracker.merge(
            metrax.Accuracy.from_model_output(
                predictions=pred_classes, labels=true_classes
            )
        )

    log_metrics({"test_accuracy": test_acc_tracker.compute()})
