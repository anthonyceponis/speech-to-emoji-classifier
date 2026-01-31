import tensorflow as tf
import os
from pathlib import Path
from scipy.io import wavfile
import jax.numpy as jnp
import jax
import optax
from flax.training import train_state
from flax import linen as nn
from tqdm import tqdm
from functools import partial
import logging
import joblib
import random
import metrax


# TODO: set a param for the dimension of output
class SimpleModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=9)(x)


def download_dataset() -> Path:
    current_dir = os.getcwd()

    # downloads a smaller subset to save time
    data_url = "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
    return Path(
        tf.keras.utils.get_file(
            "mini_speech_commands.zip",
            origin=data_url,
            extract=True,
            cache_dir=current_dir,
            cache_subdir="data",
        )
    )


@jax.jit
def run_dft(samples: jax.Array) -> jax.Array:
    """
    Computes discrete fourier transfrom for given audio sample.
    I learnt from a great explanation here:
    https://www.youtube.com/watch?v=nl9TZanwbBk
    """

    N = len(samples)

    k = jnp.arange(N).reshape(N, 1)
    n = jnp.arange(N).reshape(1, N)

    # compute the matrix using broadcasting: (N, 1) * (1, N) -> (N, N)
    exponent = (-2j * jnp.pi * k * n) / N
    dft_matrix = jnp.exp(exponent)

    fourier_coefficients = jnp.abs(dft_matrix @ samples)

    return fourier_coefficients


@partial(jax.jit, static_argnames=("N",))
def fft(x, N):
    if N == 1:
        return x

    even = fft(x[::2], N // 2)
    odd = fft(x[1::2], N // 2)

    factor = jnp.exp(-2j * jnp.pi * jnp.arange(N // 2) / N)
    return jnp.concatenate([even + factor * odd, even - factor * odd])


def run_fft(samples):
    N = len(samples)

    N_padded = 16_384
    if N_padded != N:
        samples = jnp.pad(samples, (0, N_padded - N))

    return jnp.abs(fft(samples, N_padded))


def construct_dataset(dataset_dirpath: Path) -> list[tuple[jax.Array, jax.Array]]:
    """Constructs a dataset, a list of (features, one-hot label) pairs."""

    # check if already constructed
    processed_data_dirpath = Path("processed_data")
    if processed_data_dirpath.exists():
        dataset = []
        for chunk_filepath in processed_data_dirpath.iterdir():
            chunk = joblib.load(chunk_filepath)
            for example in chunk:
                dataset.append(example)
        random.shuffle(dataset)
        return dataset

    os.makedirs(processed_data_dirpath, exist_ok=True)

    dataset_dirpath = dataset_dirpath / "mini_speech_commands"

    N = 16_000
    dataset = []

    label_dirpaths = []

    for label_dirpath in dataset_dirpath.iterdir():
        if not label_dirpath.is_dir():
            continue
        label_dirpaths.append(label_dirpath)

    logging.warning("initial jit of fft takes much longer, be patient!")
    for i, label_dirpath in tqdm(enumerate(sorted(label_dirpaths)), desc="label dirs"):

        logging.info(f"processing {label_dirpath.stem} dir")

        label = jax.nn.one_hot(i, len(label_dirpaths))

        for file_path in tqdm(
            sorted(list(label_dirpath.iterdir())), desc="processing files"
        ):
            if file_path.suffix != ".wav":
                continue

            _, audio = wavfile.read(file_path)

            samples = jnp.array(audio[:N])
            fourier_coefficients = run_fft(
                samples
            )  # swap between fft and dft to feel the difference
            dataset.append((fourier_coefficients, label))

    chunk_size = 1000

    # cache dataset construction
    for i in range(0, len(dataset), chunk_size):
        chunk = dataset[i : min((i + chunk_size), len(dataset))]
        joblib.dump(
            chunk,
            processed_data_dirpath / f"{i}.pkl",
        )

    random.shuffle(dataset)
    return dataset


# TODO: add type signature
@jax.jit
def train_step(state, batch: tuple[jax.Array, float]):
    X, y = batch

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, X)
        loss = optax.softmax_cross_entropy(logits, y).mean()

        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    return state, loss


# TODO: add type signature
@jax.jit
def val_step(state, batch: tuple[float, jax.Array]):
    X, y = batch
    logits = state.apply_fn({"params": state.params}, X)
    loss = optax.softmax_cross_entropy(logits, y).mean()

    preds = jax.nn.softmax(logits)

    return loss, preds


def split_dataset(
    dataset: list[tuple[jax.Array, jax.Array]],
) -> tuple[
    list[tuple[jax.Array, jax.Array]],
    list[tuple[jax.Array, jax.Array]],
    list[tuple[jax.Array, jax.Array]],
]:
    """Splits dataset into train, val and test sets."""
    train_count = int(len(dataset) * 0.8)
    val_count = int(len(dataset) * 0.1)

    return (
        dataset[:train_count],
        dataset[train_count : train_count + val_count],
        dataset[train_count + val_count :],
    )


def log_metrics(metrics: dict[str, float]):
    results = []
    for metric, val in metrics.items():
        results.append(f"{metric}: {val:.4f}")
    logging.info(" | ".join(results))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # fix randomness
    seed = 42
    random.seed(seed)
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)

    # load datasets
    raw_dataset_dirpath = download_dataset()
    dataset = construct_dataset(raw_dataset_dirpath)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    # init model
    model = SimpleModel()
    X_init, _ = dataset[0]
    params = model.init(init_key, X_init)["params"]

    # config optimiser
    lr = 0.001
    opt = optax.adam(lr)

    # setup model state
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt)

    # training params
    epochs = 1000
    max_patience = 10
    patience = max_patience
    best_val_loss = float("inf")
    best_model_state = None

    # main epoch loop
    for epoch in range(1, epochs + 1):

        # track patience
        patience -= 1
        if patience == 0:
            logging.info("ran out of patience, early stopping!")
            break

        # setup metric trackers
        epoch_train_loss = 0
        epoch_val_loss = 0
        epoch_val_accuracy_tracker = metrax.Accuracy.empty()

        # train loop
        for example in train_dataset:
            state, loss = train_step(state, example)
            epoch_train_loss += loss

        # val loop
        for example in val_dataset:
            loss, preds = val_step(state, example)
            epoch_val_loss += loss
            x, y = example
            epoch_val_accuracy_tracker = epoch_val_accuracy_tracker.merge(
                metrax.Accuracy.from_model_output(predictions=preds, labels=y)
            )

        # calc metrics
        epoch_train_loss /= len(train_dataset)
        epoch_val_loss /= len(val_dataset)
        epoch_val_accuracy = epoch_val_accuracy_tracker.compute()

        # save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience = max_patience
            best_model_state = state

        # log metrics
        metrics = {
            "epoch": epoch,
            "patience": patience,
            "train_loss": epoch_train_loss,
            "epoch_val_loss": epoch_val_loss,
            "epoch_val_accuracy": epoch_val_accuracy,
        }
        log_metrics(metrics)

    # test loop
    test_accuracy_tracker = metrax.Accuracy.empty()
    for example in test_dataset:
        _, preds = val_step(state, example)
        x, y = example
        test_accuracy_tracker = test_accuracy_tracker.merge(
            metrax.Accuracy.from_model_output(predictions=preds, labels=y)
        )

    print("TESTING")
    print("-----------")

    test_accuracy = test_accuracy_tracker.compute()
    metrics = {"test_accuracy": test_accuracy}

    log_metrics(metrics)
