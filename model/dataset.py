import os
from scipy.io import wavfile
from pathlib import Path
import tensorflow as tf
import jax
import jax.numpy as jnp
import numpy as np
import random
from tqdm import tqdm
import logging
import joblib
from fourier import run_fft

SAMPLE_RATE = 16000
FRAME_SIZE = 400  # 25ms window
FRAME_STRIDE = 160  # 10ms stride
N_FFT = 512  # closest power of 2 for FFT
NUM_MELS = 40  # number of features to extract


def download_dataset() -> Path:
    current_dir = os.getcwd()
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


def get_mel_filterbank():
    """Creates a Mel Filterbank matrix to convert FFT frequency samples within a frame to Mel bins."""

    # note that these magic numbers are based on emperical studies
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (SAMPLE_RATE / 2) / 700)

    # sample in log freq space
    mel_points = np.linspace(low_freq_mel, high_freq_mel, NUM_MELS + 2)

    # convert back to linear freq space
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    # convert hz to fft indicies within a frame, so we assign each mel point to an existing fft frame index
    # we have to apply it as a proportion because remember that frames have less samples that the range of sampled frequencies.
    bin_points = np.floor((N_FFT + 1) * hz_points / SAMPLE_RATE).astype(int)

    # create the triangular weigted mel features
    fbank = np.zeros((NUM_MELS, int(N_FFT / 2 + 1)))
    for m in range(1, NUM_MELS + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_points[m - 1]) / (
                bin_points[m] - bin_points[m - 1]
            )
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_points[m + 1] - k) / (
                bin_points[m + 1] - bin_points[m]
            )

    return jnp.array(fbank)


@jax.jit
def extract_log_mel_spectrogram(audio, mel_filters):
    """
    Chops up clip into overlapping windows.
    Performs fft on samples in window.
    Applies mel filterbank matrix transformation per window.
    """

    # pad/truncate audio samples
    if len(audio) < SAMPLE_RATE:
        audio = jnp.pad(audio, (0, SAMPLE_RATE - len(audio)))
    else:
        audio = audio[:SAMPLE_RATE]

    num_frames = (len(audio) - FRAME_SIZE) // FRAME_STRIDE + 1

    # create window frames
    indices = (
        jnp.tile(jnp.arange(0, FRAME_SIZE), (num_frames, 1))
        + jnp.tile(
            jnp.arange(0, num_frames * FRAME_STRIDE, FRAME_STRIDE), (FRAME_SIZE, 1)
        ).T
    )
    frames = audio[indices]

    # multiply windows by bell curve weight decay
    window = jnp.hanning(FRAME_SIZE)
    frames = frames * window

    # pad frames
    padding = N_FFT - FRAME_SIZE
    frames_padded = jnp.pad(frames, ((0, 0), (0, padding)))

    # run fft on frames
    run_fft_batched = jax.vmap(run_fft)
    fft_mag = run_fft_batched(frames_padded)

    # slice to keep only the first N_FFT//2 + 1 bins.
    fft_mag = fft_mag[:, : N_FFT // 2 + 1]

    # due to P = V^2/R, power is proportional to the square of the amplitude of the signal
    # we are interested in modelling the power of the signal as opposed to just amplitude
    power_frames = (fft_mag**2) / N_FFT

    # apply mel filterbank matrix transformation
    mel_energy = jnp.dot(power_frames, mel_filters.T)

    # take log of mel binned frequencies to better align with human perception scale
    log_mel_energy = jnp.log(mel_energy + 1e-10)

    # crop/pad to meet target frames
    TARGET_FRAMES = 98
    current_frames = log_mel_energy.shape[0]
    if current_frames < TARGET_FRAMES:
        padding = ((0, TARGET_FRAMES - current_frames), (0, 0))
        log_mel_energy = jnp.pad(log_mel_energy, padding)
    elif current_frames > TARGET_FRAMES:
        log_mel_energy = log_mel_energy[:TARGET_FRAMES, :]

    return log_mel_energy


def construct_dataset() -> list[tuple[jax.Array, jax.Array]]:
    """Downloads raw dataset and constructs a list of (features, one-hot label) pairs."""

    dataset_dirpath = download_dataset()

    # check if already constructed
    processed_data_dirpath = Path("processed_data")  # Renamed to avoid loading old data
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

    dataset = []
    label_dirpaths = []

    # find how many different labels there are
    for label_dirpath in dataset_dirpath.iterdir():
        if label_dirpath.is_file() or label_dirpath.name.startswith("."):
            continue
        label_dirpaths.append(label_dirpath)
    label_dirpaths.sort()

    # save label names to disk
    with open("label_names.txt", "w") as f:
        f.write("\n".join([x.stem for x in label_dirpaths]))

    # precompute mel matrix
    mel_filters = get_mel_filterbank()
    logging.info(f"generated mel filterbank with shape {mel_filters.shape}")

    for i, label_dirpath in tqdm(enumerate(label_dirpaths), desc="label dirs"):

        logging.info(f"processing {label_dirpath.stem} dir")
        label = jax.nn.one_hot(i, len(label_dirpaths))

        for file_path in tqdm(
            sorted(list(label_dirpath.iterdir())), desc="processing files"
        ):
            if file_path.suffix != ".wav":
                continue

            _, audio = wavfile.read(file_path)

            # normalise sample range to [-1,1]
            audio = audio.astype(np.float32) / 32768.0

            # extract features
            features = extract_log_mel_spectrogram(audio, mel_filters)

            dataset.append((features, label))

    # cache dataset construction
    chunk_size = 1000
    for i in range(0, len(dataset), chunk_size):
        chunk = dataset[i : min((i + chunk_size), len(dataset))]
        joblib.dump(
            chunk,
            processed_data_dirpath / f"{i}.pkl",
        )

    random.shuffle(dataset)
    return dataset


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
