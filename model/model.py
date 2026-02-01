import flax.linen as nn


class SimpleModel(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x, training: bool):
        # Input shape is now (Batch, 40)

        # 1. Normalize the MFCCs (crucial for stability)
        x = nn.LayerNorm()(x)

        # 2. First dense layer
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)

        # 3. Second dense layer
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)

        # 4. Output
        x = nn.Dense(self.output_dim)(x)
        return x


class CNNModel(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x, training: bool):
        # Input shape: (Batch, 98, 40)

        # 1. Add "Channel" dimension for the CNN (Batch, 98, 40, 1)
        x = x[..., None]

        # 2. First Convolutional Block
        # We use (3, 3) kernels to scan time and frequency simultaneously
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # 3. Second Convolutional Block
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # 4. Flatten the 2D maps into a 1D vector
        x = x.reshape((x.shape[0], -1))

        # 5. Dense Classification Layers
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)

        x = nn.Dense(self.output_dim)(x)
        return x
