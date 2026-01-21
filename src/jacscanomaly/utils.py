import jax
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
def calc_A_pspl(t0,tE,u0,t):
    u = jnp.sqrt(u0**2 + ((t-t0)/tE)**2)
    A = (u**2 + 2)/(u*jnp.sqrt(u**2+4))
    return A

@jit
def calc_chi2(params, data):
    t0, tE, u0 = params
    A = calc_A_pspl(t0, tE, u0, data[:,0])
    fs, fb = linear_fit(A, data[:,1], 1/data[:,2]**2)
    f_model = fs * A + fb
    res = data[:,1] - f_model
    chi2 = jnp.sum((res / data[:,2])**2)
    return chi2

@jit
def calc_res_norm(params, data):
    t0, tE, u0 = params
    A = calc_A_pspl(t0, tE, u0, data[:,0])
    fs, fb = linear_fit(A, data[:,1], 1/data[:,2]**2)
    f_model = fs * A + fb
    res = (data[:,1] - f_model) / data[:,2]
    return res

@jit
def calc_res(params, data):
    t0, tE, u0 = params
    A = calc_A_pspl(t0, tE, u0, data[:,0])
    fs, fb = linear_fit(A, data[:,1], 1/data[:,2]**2)
    f_model = fs * A + fb
    res = (data[:,1] - f_model)
    return res

@jit
def get_chi2_flat(data_time,data_flux,data_ferr):
    bunsi = jnp.sum(data_flux/(data_ferr**2))
    bunbo= jnp.sum(1/(data_ferr**2))
    model_flux = bunsi/bunbo
    return jnp.sum(((data_flux-model_flux)/data_ferr)**2), ((data_flux-model_flux)/data_ferr)**2

@jit
def calc_A_j_0(t0,teff,t):
    Q = 1+((t-t0)/teff)**2
    A = 1/jnp.sqrt(Q)
    return A

@jit
def calc_A_j_1(t0,teff,t):
    Q = 1+((t-t0)/teff)**2
    A = (Q+2)/jnp.sqrt(Q*(Q+4))
    return A

@jit
def get_chi2_anom(t0,teff,data_time,data_flux,data_ferr):
    A_j_0 = calc_A_j_0(t0,teff,data_time)
    fs_j_0, fb_j_0 = linear_fit(A_j_0, data_flux,(1/data_ferr)**2)
    model_flux_j_0= A_j_0*fs_j_0 + fb_j_0
    chi2s_j_0 = ((data_flux-model_flux_j_0)/data_ferr)**2
    chi2_j_0 = jnp.sum(chi2s_j_0)

    A_j_1 = calc_A_j_1(t0,teff,data_time)
    fs_j_1, fb_j_1 = linear_fit(A_j_1, data_flux,(1/data_ferr)**2)
    model_flux_j_1= A_j_1*fs_j_1 + fb_j_1
    chi2s_j_1 = ((data_flux-model_flux_j_1)/data_ferr)**2
    chi2_j_1 = jnp.sum(chi2s_j_1)

    choose_model_0 = chi2_j_0 < chi2_j_1

    chi2_best = jnp.where(choose_model_0, chi2_j_0, chi2_j_1)
    chi2s_best = jnp.where(choose_model_0, chi2s_j_0, chi2s_j_1)

    return chi2_best, chi2s_best

@jit
def predict_flat_model(data_flux, data_ferr):
    w = 1.0 / (data_ferr ** 2)
    mu = jnp.sum(w * data_flux) / jnp.sum(w)
    return jnp.full_like(data_flux, mu)

@jit
def predict_anom_model(t0, teff, data_time, data_flux, data_ferr):
    # j=0
    A0 = calc_A_j_0(t0, teff, data_time)
    fs0, fb0 = linear_fit(A0, data_flux, (1.0 / data_ferr) ** 2)
    m0 = A0 * fs0 + fb0
    chi2_0 = jnp.sum(((data_flux - m0) / data_ferr) ** 2)

    # j=1
    A1 = calc_A_j_1(t0, teff, data_time)
    fs1, fb1 = linear_fit(A1, data_flux, (1.0 / data_ferr) ** 2)
    m1 = A1 * fs1 + fb1
    chi2_1 = jnp.sum(((data_flux - m1) / data_ferr) ** 2)

    choose0 = chi2_0 < chi2_1
    return jnp.where(choose0, m0, m1), choose0  # choose0 も返すとデバッグに便利
