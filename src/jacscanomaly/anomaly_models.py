from __future__ import annotations
import jax.numpy as jnp
from jax import jit
from .photometry import linear_fit

@jit
def get_chi2_flat(data_time, data_flux, data_ferr):
    bunsi = jnp.sum(data_flux/(data_ferr**2))
    bunbo = jnp.sum(1/(data_ferr**2))
    model_flux = bunsi/bunbo
    chi2s = ((data_flux-model_flux)/data_ferr)**2
    return jnp.sum(chi2s), chi2s

@jit
def calc_A_j_0(t0, teff, t):
    Q = 1 + ((t - t0) / teff) ** 2
    return 1 / jnp.sqrt(Q)

@jit
def calc_A_j_1(t0, teff, t):
    Q = 1 + ((t - t0) / teff) ** 2
    return (Q + 2) / jnp.sqrt(Q * (Q + 4))

@jit
def get_chi2_anom(t0, teff, data_time, data_flux, data_ferr):
    A0 = calc_A_j_0(t0, teff, data_time)
    fs0, fb0 = linear_fit(A0, data_flux, (1 / data_ferr) ** 2)
    m0 = A0 * fs0 + fb0
    chi2s0 = ((data_flux - m0) / data_ferr) ** 2
    chi2_0 = jnp.sum(chi2s0)

    A1 = calc_A_j_1(t0, teff, data_time)
    fs1, fb1 = linear_fit(A1, data_flux, (1 / data_ferr) ** 2)
    m1 = A1 * fs1 + fb1
    chi2s1 = ((data_flux - m1) / data_ferr) ** 2
    chi2_1 = jnp.sum(chi2s1)

    choose0 = chi2_0 < chi2_1
    chi2_best = jnp.where(choose0, chi2_0, chi2_1)
    chi2s_best = jnp.where(choose0, chi2s0, chi2s1)
    return chi2_best, chi2s_best

@jit
def predict_flat_model(data_flux, data_ferr):
    w = 1.0 / (data_ferr ** 2)
    mu = jnp.sum(w * data_flux) / jnp.sum(w)
    return jnp.full_like(data_flux, mu)

@jit
def predict_anom_model(t0, teff, data_time, data_flux, data_ferr):
    A0 = calc_A_j_0(t0, teff, data_time)
    fs0, fb0 = linear_fit(A0, data_flux, (1.0 / data_ferr) ** 2)
    m0 = A0 * fs0 + fb0
    chi2_0 = jnp.sum(((data_flux - m0) / data_ferr) ** 2)

    A1 = calc_A_j_1(t0, teff, data_time)
    fs1, fb1 = linear_fit(A1, data_flux, (1.0 / data_ferr) ** 2)
    m1 = A1 * fs1 + fb1
    chi2_1 = jnp.sum(((data_flux - m1) / data_ferr) ** 2)

    choose0 = chi2_0 < chi2_1
    return jnp.where(choose0, m0, m1), choose0

@jit
def get_flat_plot_model(t_plot, data_flux, data_ferr):
    w = 1.0 / (data_ferr ** 2)
    mu = jnp.sum(w * data_flux) / jnp.sum(w)
    return jnp.full_like(t_plot, mu)

@jit
def get_anom_plot_model(t_plot, t0, teff, data_time, data_flux, data_ferr):
    A0 = calc_A_j_0(t0, teff, data_time)
    fs0, fb0 = linear_fit(A0, data_flux, (1.0 / data_ferr) ** 2)
    m0 = A0 * fs0 + fb0
    chi2_0 = jnp.sum(((data_flux - m0) / data_ferr) ** 2)
    A0_plot = calc_A_j_0(t0, teff, t_plot)
    m0_plot = A0_plot * fs0 + fb0

    A1 = calc_A_j_1(t0, teff, data_time)
    fs1, fb1 = linear_fit(A1, data_flux, (1.0 / data_ferr) ** 2)
    m1 = A1 * fs1 + fb1
    chi2_1 = jnp.sum(((data_flux - m1) / data_ferr) ** 2)
    A1_plot = calc_A_j_0(t0, teff, t_plot)
    m1_plot = A1_plot * fs0 + fb0

    choose0 = chi2_0 < chi2_1
    return jnp.where(choose0, m0_plot, m1_plot), choose0
