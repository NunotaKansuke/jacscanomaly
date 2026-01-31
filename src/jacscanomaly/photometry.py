import jax.numpy as jnp
from jax import jit

@jit
def linear_fit(x, y, w):
    x_mean = jnp.sum(w * x) / jnp.sum(w)
    y_mean = jnp.sum(w * y) / jnp.sum(w)

    x_c = x - x_mean
    y_c = y - y_mean

    wxx_sum = jnp.sum(w * x_c * x_c)
    wxy_sum = jnp.sum(w * x_c * y_c)

    a = wxy_sum / wxx_sum
    b = y_mean - a * x_mean
    return a, b

@jit
def solve_fs_fb(A, flux, ferr):
    w = 1.0 / (ferr ** 2)
    return linear_fit(A, flux, w)

