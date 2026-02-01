import jax
import jax.numpy as jnp
import math


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


@jax.jit
def run_fft(x):
    """
    Compute fast fourier transform on given audio sample.

    This might be one of the most sophisticated, possibly most insane, code snippets I have ever written.
    Understanding this properly can take many hours and is worth the investment.
    Covers dynammic programming, recursion, symmetry, complex numbers, roots of unity, clever bit permutations etc.

    I recommend asking ai and watching these two yt videos to understand it fully:

    1. https://www.youtube.com/watch?v=h7apO7q16V0 (polynomial interpretation, I dont quite understand it and prefer the wave correlation interpretation but still has some useful visuals and background.)
    2. https://fr.mathworks.com/videos/understanding-the-discrete-fourier-transform-and-the-fft-1700042348737.html (explains the wave correlation argument but does not explain fft.)
    """

    N = x.shape[0]
    log_n = int(math.log2(N))

    # bottom up bit-reversal permutation
    j = jnp.arange(N)
    k = jnp.zeros_like(j)
    for i in range(log_n):
        k = (k << 1) | (j & 1)
        j = j >> 1
    x = x[k]

    # iterative butterfly operation
    for i in range(log_n):
        m = 1 << (i + 1)
        half_m = m // 2

        k_indices = jnp.arange(half_m)
        factor = jnp.exp(-2j * jnp.pi * k_indices / m)

        x_reshaped = x.reshape((-1, m))

        even = x_reshaped[:, :half_m]
        odd = x_reshaped[:, half_m:]

        odd_factor = factor * odd
        x = jnp.concatenate([even + odd_factor, even - odd_factor], axis=1).flatten()

    return jnp.abs(x)
