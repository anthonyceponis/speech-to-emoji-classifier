import flax.linen as nn


class CNNModel(nn.Module):
    """
    Model suggestion by gemini.
    Not particularly interesting, just a cnn.
    Baisically model sound in 2d as a spectrogram.
    Focus is on signal processing.
    """

    output_dim: int

    @nn.compact
    def __call__(self, x, training: bool):
        # add channel dimension
        x = x[..., None]

        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)

        x = nn.Dense(self.output_dim)(x)
        return x
