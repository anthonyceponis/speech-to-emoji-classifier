from model import CNNModel
import logging
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state, checkpoints
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path

# Update imports to use the new feature extraction
from dataset import extract_log_mel_spectrogram, get_mel_filterbank

logging.basicConfig(level=logging.INFO)


# Define the expected JSON structure
class InferenceRequest(BaseModel):
    samples: list[float]
    sample_rate: int = 16000


# fix randomness
seed = 42
key = jax.random.PRNGKey(seed)
key, init_key = jax.random.split(key)

if __name__ == "__main__":
    # read label names
    with open("label_names.txt", "r") as f:
        label_names = [line.strip() for line in f.readlines()]

    # init model
    model = CNNModel(len(label_names))
    dummy_input = jnp.zeros((1, 98, 40))
    params = model.init(init_key, dummy_input, training=False)["params"]

    # optimiser
    lr = 0.001
    opt = optax.adam(lr)

    # load checkpoint
    init_state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=opt
    )
    state = checkpoints.restore_checkpoint(
        ckpt_dir=Path("checkpoints").resolve(), target=init_state
    )

    # precompute mel filterbank matrix
    mel_filters = get_mel_filterbank()

    # create app
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/infer")
    async def infer(request: InferenceRequest):
        print(f"received {len(request.samples)} audio samples")
        raw_audio = jnp.array(request.samples, dtype=jnp.float32)

        print("extracting features...")
        features = extract_log_mel_spectrogram(raw_audio, mel_filters)
        features_batched = features[None, ...]

        print("predicting...")
        logits = state.apply_fn(
            {"params": state.params}, features_batched, training=False
        )

        prediction_index = jnp.argmax(logits)
        prediction_label = label_names[prediction_index]

        return {"prediction": prediction_label}

    # run server
    uvicorn.run(app, host="0.0.0.0", port=8000)
