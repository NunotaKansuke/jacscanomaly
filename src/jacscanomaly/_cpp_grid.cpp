#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

PyArrayObject* as_double_array(PyObject* obj) {
    return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)
    );
}

double calc_a0(double t0, double teff, double t) {
    const double u = (t - t0) / teff;
    const double q = 1.0 + u * u;
    return 1.0 / std::sqrt(q);
}

double calc_a1(double t0, double teff, double t) {
    const double u = (t - t0) / teff;
    const double q = 1.0 + u * u;
    return (q + 2.0) / std::sqrt(q * (q + 4.0));
}

struct Window {
    npy_intp start = 0;
    npy_intp end = 0;

    int size() const {
        return static_cast<int>(end - start);
    }
};

struct Fit {
    double a = 0.0;
    double b = 0.0;
    double chi2 = 0.0;
    bool valid = false;
};

double pspl_magnification(double t0, double tE, double u0, double t) {
    const double tau = (t - t0) / tE;
    const double u = std::sqrt(tau * tau + u0 * u0);
    const double u_safe = std::max(u, 1e-12);
    return (u_safe * u_safe + 2.0) / (u_safe * std::sqrt(u_safe * u_safe + 4.0));
}

Fit fit_pspl_fluxes(
    double t0,
    double tE,
    double u0,
    const double* time,
    const double* flux,
    const double* ferr,
    npy_intp n
) {
    double sw = 0.0;
    double sx = 0.0;
    double sy = 0.0;
    for (npy_intp i = 0; i < n; ++i) {
        const double fe = std::max(ferr[i], 1e-12);
        const double w = 1.0 / (fe * fe);
        const double a = pspl_magnification(t0, tE, u0, time[i]);
        sw += w;
        sx += w * a;
        sy += w * flux[i];
    }
    if (!(sw > 0.0)) {
        return {};
    }

    const double x_mean = sx / sw;
    const double y_mean = sy / sw;
    double wxx = 0.0;
    double wxy = 0.0;
    for (npy_intp i = 0; i < n; ++i) {
        const double fe = std::max(ferr[i], 1e-12);
        const double w = 1.0 / (fe * fe);
        const double xc = pspl_magnification(t0, tE, u0, time[i]) - x_mean;
        const double yc = flux[i] - y_mean;
        wxx += w * xc * xc;
        wxy += w * xc * yc;
    }
    if (!(wxx > 0.0)) {
        return {};
    }

    Fit fit;
    fit.a = wxy / wxx;
    fit.b = y_mean - fit.a * x_mean;
    fit.valid = true;
    double chi2 = 0.0;
    for (npy_intp i = 0; i < n; ++i) {
        const double fe = std::max(ferr[i], 1e-12);
        const double model = fit.a * pspl_magnification(t0, tE, u0, time[i]) + fit.b;
        const double r = (flux[i] - model) / fe;
        chi2 += r * r;
    }
    fit.chi2 = chi2;
    return fit;
}

void pspl_residuals(
    const double q[3],
    const double* time,
    const double* flux,
    const double* ferr,
    npy_intp n,
    std::vector<double>& residual,
    Fit* out_fit = nullptr,
    double u0_min = 0.0,
    int min_t0_support_points = 0,
    double t0_support_tE_coeff = 0.0
) {
    const double t0 = q[0];
    const double tE = std::max(std::exp(q[1]), 1e-12);
    const double u0 = q[2];
    if (std::abs(u0) < u0_min) {
        if (out_fit != nullptr) {
            *out_fit = {};
        }
        residual.assign(static_cast<size_t>(n), 1e100);
        return;
    }
    if (min_t0_support_points > 0 && t0_support_tE_coeff > 0.0) {
        int support = 0;
        const double t0_support_window = t0_support_tE_coeff * tE;
        const double lo = t0 - t0_support_window;
        const double hi = t0 + t0_support_window;
        for (npy_intp i = 0; i < n; ++i) {
            if (time[i] >= lo && time[i] <= hi) {
                ++support;
            }
        }
        if (support < min_t0_support_points) {
            if (out_fit != nullptr) {
                *out_fit = {};
            }
            residual.assign(static_cast<size_t>(n), 1e100);
            return;
        }
    }
    const Fit fit = fit_pspl_fluxes(t0, tE, u0, time, flux, ferr, n);
    if (out_fit != nullptr) {
        *out_fit = fit;
    }
    residual.resize(static_cast<size_t>(n));
    if (!fit.valid) {
        std::fill(residual.begin(), residual.end(), 1e100);
        return;
    }
    for (npy_intp i = 0; i < n; ++i) {
        const double fe = std::max(ferr[i], 1e-12);
        const double model = fit.a * pspl_magnification(t0, tE, u0, time[i]) + fit.b;
        residual[static_cast<size_t>(i)] = (flux[i] - model) / fe;
    }
}

double dot_vec(const std::vector<double>& a, const std::vector<double>& b) {
    double out = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        out += a[i] * b[i];
    }
    return out;
}

bool solve_3x3(double a[3][3], double b[3], double x[3]) {
    double m[3][4] = {
        {a[0][0], a[0][1], a[0][2], b[0]},
        {a[1][0], a[1][1], a[1][2], b[1]},
        {a[2][0], a[2][1], a[2][2], b[2]},
    };
    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 3; ++row) {
            if (std::abs(m[row][col]) > std::abs(m[pivot][col])) {
                pivot = row;
            }
        }
        if (std::abs(m[pivot][col]) < 1e-30) {
            return false;
        }
        if (pivot != col) {
            for (int k = col; k < 4; ++k) {
                std::swap(m[col][k], m[pivot][k]);
            }
        }
        const double div = m[col][col];
        for (int k = col; k < 4; ++k) {
            m[col][k] /= div;
        }
        for (int row = 0; row < 3; ++row) {
            if (row == col) {
                continue;
            }
            const double factor = m[row][col];
            for (int k = col; k < 4; ++k) {
                m[row][k] -= factor * m[col][k];
            }
        }
    }
    x[0] = m[0][3];
    x[1] = m[1][3];
    x[2] = m[2][3];
    return true;
}

double sumsq(const std::vector<double>& r) {
    return dot_vec(r, r);
}

template <typename Func>
Fit fit_weighted_line(
    Func basis,
    double t0,
    double teff,
    const double* time,
    const double* flux,
    const double* weight,
    Window window
) {
    double sw = 0.0;
    double sx = 0.0;
    double sy = 0.0;

    for (npy_intp i = window.start; i < window.end; ++i) {
        const double t = time[i];
        const double w = weight[i];
        const double x = basis(t0, teff, t);
        sw += w;
        sx += w * x;
        sy += w * flux[i];
    }
    if (!(sw > 0.0)) {
        return {};
    }

    const double x_mean = sx / sw;
    const double y_mean = sy / sw;
    double wxx = 0.0;
    double wxy = 0.0;
    for (npy_intp i = window.start; i < window.end; ++i) {
        const double t = time[i];
        const double w = weight[i];
        const double xc = basis(t0, teff, t) - x_mean;
        const double yc = flux[i] - y_mean;
        wxx += w * xc * xc;
        wxy += w * xc * yc;
    }
    if (!(wxx > 0.0)) {
        return {};
    }

    Fit fit;
    fit.a = wxy / wxx;
    fit.b = y_mean - fit.a * x_mean;
    fit.valid = true;

    double chi2 = 0.0;
    for (npy_intp i = window.start; i < window.end; ++i) {
        const double t = time[i];
        const double model = fit.a * basis(t0, teff, t) + fit.b;
        const double r = flux[i] - model;
        chi2 += r * r * weight[i];
    }
    fit.chi2 = chi2;
    return fit;
}

// Apply the post-grid one-lobe test without transferring the candidate batch
// through XLA.  This deliberately mirrors PlanetSignalExtractor's former JAX
// implementation: a flat and the two point-lens template shapes are fit in
// each local window, then the per-sample improvement is box-smoothed and its
// number of above-threshold lobes is counted.
PyObject* unimodal_mask(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject *time_obj = nullptr, *residual_obj = nullptr, *ferr_obj = nullptr;
    PyObject *t0_obj = nullptr, *teff_obj = nullptr;
    double teff_coeff = 3.0, min_improvement = 9.0, peak_frac = 0.2;
    int min_pts = 4, smooth_points = 5, max_lobes = 1;
    static const char* kwlist[] = {
        "time", "residual", "ferr", "t0", "teff", "teff_coeff",
        "min_pts", "min_improvement", "peak_frac", "smooth_points",
        "max_lobes", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOO|diddii",
            const_cast<char**>(kwlist), &time_obj, &residual_obj, &ferr_obj,
            &t0_obj, &teff_obj, &teff_coeff, &min_pts, &min_improvement,
            &peak_frac, &smooth_points, &max_lobes)) {
        return nullptr;
    }
    PyArrayObject* time_arr = as_double_array(time_obj);
    PyArrayObject* residual_arr = as_double_array(residual_obj);
    PyArrayObject* ferr_arr = as_double_array(ferr_obj);
    PyArrayObject* t0_arr = as_double_array(t0_obj);
    PyArrayObject* teff_arr = as_double_array(teff_obj);
    if (!time_arr || !residual_arr || !ferr_arr || !t0_arr || !teff_arr) {
        Py_XDECREF(time_arr); Py_XDECREF(residual_arr); Py_XDECREF(ferr_arr);
        Py_XDECREF(t0_arr); Py_XDECREF(teff_arr);
        return nullptr;
    }
    if (PyArray_NDIM(time_arr) != 1 || PyArray_NDIM(residual_arr) != 1 ||
        PyArray_NDIM(ferr_arr) != 1 || PyArray_NDIM(t0_arr) != 1 ||
        PyArray_NDIM(teff_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "all inputs must be one-dimensional arrays");
        Py_DECREF(time_arr); Py_DECREF(residual_arr); Py_DECREF(ferr_arr);
        Py_DECREF(t0_arr); Py_DECREF(teff_arr);
        return nullptr;
    }
    const npy_intp n = PyArray_DIM(time_arr, 0);
    const npy_intp n_grid = PyArray_DIM(t0_arr, 0);
    if (PyArray_DIM(residual_arr, 0) != n || PyArray_DIM(ferr_arr, 0) != n ||
        PyArray_DIM(teff_arr, 0) != n_grid) {
        PyErr_SetString(PyExc_ValueError, "incompatible unimodal-mask input lengths");
        Py_DECREF(time_arr); Py_DECREF(residual_arr); Py_DECREF(ferr_arr);
        Py_DECREF(t0_arr); Py_DECREF(teff_arr);
        return nullptr;
    }
    npy_intp dims[1] = {n_grid};
    PyArrayObject* out = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dims, NPY_BOOL));
    if (!out) {
        Py_DECREF(time_arr); Py_DECREF(residual_arr); Py_DECREF(ferr_arr);
        Py_DECREF(t0_arr); Py_DECREF(teff_arr);
        return nullptr;
    }
    const double* time = static_cast<const double*>(PyArray_DATA(time_arr));
    const double* residual = static_cast<const double*>(PyArray_DATA(residual_arr));
    const double* ferr = static_cast<const double*>(PyArray_DATA(ferr_arr));
    const double* t0s = static_cast<const double*>(PyArray_DATA(t0_arr));
    const double* teffs = static_cast<const double*>(PyArray_DATA(teff_arr));
    auto* accepted = static_cast<npy_bool*>(PyArray_DATA(out));
    int width = std::max(1, smooth_points);
    if ((width % 2) == 0) ++width;
    const int pad = width / 2;

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(dynamic, 8)
    for (npy_intp g = 0; g < n_grid; ++g) {
        accepted[g] = 0;
        const double teff = std::abs(teffs[g]);
        if (!(teff > 0.0) || !std::isfinite(teff)) continue;
        const double lo = t0s[g] - teff_coeff * teff;
        const double hi = t0s[g] + teff_coeff * teff;
        const Window window{
            static_cast<npy_intp>(std::upper_bound(time, time + n, lo) - time),
            static_cast<npy_intp>(std::lower_bound(time, time + n, hi) - time),
        };
        const int count = window.size();
        if (count < min_pts) continue;
        std::vector<double> weight(static_cast<size_t>(count));
        double sw = 0.0, sy = 0.0;
        for (int k = 0; k < count; ++k) {
            const npy_intp i = window.start + k;
            const double fe = std::max(ferr[i], 1e-12);
            weight[static_cast<size_t>(k)] = 1.0 / (fe * fe);
            sw += weight[static_cast<size_t>(k)];
            sy += weight[static_cast<size_t>(k)] * residual[i];
        }
        if (!(sw > 0.0)) continue;
        const double mu = sy / sw;
        // fit_weighted_line expects arrays whose window offsets begin at zero.
        std::vector<double> local_time(static_cast<size_t>(count));
        std::vector<double> local_residual(static_cast<size_t>(count));
        for (int k = 0; k < count; ++k) {
            local_time[static_cast<size_t>(k)] = time[window.start + k];
            local_residual[static_cast<size_t>(k)] = residual[window.start + k];
        }
        const Window local_window{0, static_cast<npy_intp>(count)};
        const Fit fit0 = fit_weighted_line(calc_a0, t0s[g], teff,
            local_time.data(), local_residual.data(), weight.data(), local_window);
        const Fit fit1 = fit_weighted_line(calc_a1, t0s[g], teff,
            local_time.data(), local_residual.data(), weight.data(), local_window);
        const Fit fit = (fit0.valid && (!fit1.valid || fit0.chi2 < fit1.chi2)) ? fit0 : fit1;
        const bool use0 = fit0.valid && (!fit1.valid || fit0.chi2 < fit1.chi2);
        if (!fit.valid) continue;
        std::vector<double> improvement(static_cast<size_t>(count));
        double peak = 0.0;
        for (int k = 0; k < count; ++k) {
            const double rflat = local_residual[static_cast<size_t>(k)] - mu;
            const double a = use0 ? calc_a0(t0s[g], teff, local_time[static_cast<size_t>(k)])
                                  : calc_a1(t0s[g], teff, local_time[static_cast<size_t>(k)]);
            const double ranom = local_residual[static_cast<size_t>(k)] - (fit.a * a + fit.b);
            const double value = std::max((rflat * rflat - ranom * ranom)
                * weight[static_cast<size_t>(k)], 0.0);
            improvement[static_cast<size_t>(k)] = value;
            peak = std::max(peak, value);
        }
        if (!(peak > 0.0) || !std::isfinite(peak)) continue;
        const double threshold = std::max(min_improvement, peak_frac * peak);
        int lobes = 0;
        bool previous = false;
        for (int k = 0; k < count; ++k) {
            double smooth = 0.0;
            for (int j = -pad; j <= pad; ++j) {
                const int at = k + j;
                if (at >= 0 && at < count) {
                    smooth += improvement[static_cast<size_t>(at)];
                } else if (window.start == 0 && at < 0) {
                    smooth += improvement.front();
                } else if (window.end == n && at >= count) {
                    smooth += improvement.back();
                }
            }
            const bool active = smooth / static_cast<double>(width) >= threshold;
            if (active && !previous) ++lobes;
            previous = active;
        }
        accepted[g] = lobes <= max_lobes;
    }
    Py_END_ALLOW_THREADS
    Py_DECREF(time_arr); Py_DECREF(residual_arr); Py_DECREF(ferr_arr);
    Py_DECREF(t0_arr); Py_DECREF(teff_arr);
    return reinterpret_cast<PyObject*>(out);
}

PyObject* run_grid(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* time_obj = nullptr;
    PyObject* flux_obj = nullptr;
    PyObject* weight_obj = nullptr;
    PyObject* t0_obj = nullptr;
    PyObject* teff_obj = nullptr;
    double sigma = 3.0;
    double teff_coeff = 3.0;
    int min_pts = 4;

    static const char* kwlist[] = {
        "time", "flux", "weight", "t0", "teff", "sigma", "teff_coeff", "min_pts", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOOO|ddi",
            const_cast<char**>(kwlist),
            &time_obj,
            &flux_obj,
            &weight_obj,
            &t0_obj,
            &teff_obj,
            &sigma,
            &teff_coeff,
            &min_pts
        )) {
        return nullptr;
    }

    PyArrayObject* time_arr = as_double_array(time_obj);
    PyArrayObject* flux_arr = as_double_array(flux_obj);
    PyArrayObject* weight_arr = as_double_array(weight_obj);
    PyArrayObject* t0_arr = as_double_array(t0_obj);
    PyArrayObject* teff_arr = as_double_array(teff_obj);
    if (!time_arr || !flux_arr || !weight_arr || !t0_arr || !teff_arr) {
        Py_XDECREF(time_arr);
        Py_XDECREF(flux_arr);
        Py_XDECREF(weight_arr);
        Py_XDECREF(t0_arr);
        Py_XDECREF(teff_arr);
        return nullptr;
    }

    if (PyArray_NDIM(time_arr) != 1 || PyArray_NDIM(flux_arr) != 1 || PyArray_NDIM(weight_arr) != 1 ||
        PyArray_NDIM(t0_arr) != 1 || PyArray_NDIM(teff_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "all inputs must be one-dimensional arrays");
        Py_DECREF(time_arr);
        Py_DECREF(flux_arr);
        Py_DECREF(weight_arr);
        Py_DECREF(t0_arr);
        Py_DECREF(teff_arr);
        return nullptr;
    }

    const npy_intp n = PyArray_DIM(time_arr, 0);
    const npy_intp n_grid = PyArray_DIM(t0_arr, 0);
    if (PyArray_DIM(flux_arr, 0) != n || PyArray_DIM(weight_arr, 0) != n) {
        PyErr_SetString(PyExc_ValueError, "time, flux, and weight must have the same length");
        Py_DECREF(time_arr);
        Py_DECREF(flux_arr);
        Py_DECREF(weight_arr);
        Py_DECREF(t0_arr);
        Py_DECREF(teff_arr);
        return nullptr;
    }
    if (PyArray_DIM(teff_arr, 0) != n_grid) {
        PyErr_SetString(PyExc_ValueError, "t0 and teff must have the same length");
        Py_DECREF(time_arr);
        Py_DECREF(flux_arr);
        Py_DECREF(weight_arr);
        Py_DECREF(t0_arr);
        Py_DECREF(teff_arr);
        return nullptr;
    }

    {
        npy_intp dims[1] = {n_grid};
        PyArrayObject* dchi2_arr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dims, NPY_DOUBLE));
        PyArrayObject* nwin_arr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dims, NPY_INT32));
        PyArrayObject* ncontrib_arr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dims, NPY_INT32));
        PyArrayObject* neff_arr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dims, NPY_DOUBLE));
        PyArrayObject* peak_arr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dims, NPY_DOUBLE));
        PyArrayObject* rho1_arr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dims, NPY_DOUBLE));
        PyArrayObject* longest_arr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dims, NPY_INT32));
        if (!dchi2_arr || !nwin_arr || !ncontrib_arr || !neff_arr || !peak_arr || !rho1_arr || !longest_arr) {
            Py_XDECREF(dchi2_arr);
            Py_XDECREF(nwin_arr);
            Py_XDECREF(ncontrib_arr);
            Py_XDECREF(neff_arr);
            Py_XDECREF(peak_arr);
            Py_XDECREF(rho1_arr);
            Py_XDECREF(longest_arr);
            goto fail;
        }

        const double* time = static_cast<const double*>(PyArray_DATA(time_arr));
        const double* flux = static_cast<const double*>(PyArray_DATA(flux_arr));
        const double* weight = static_cast<const double*>(PyArray_DATA(weight_arr));
        const double* t0_grid = static_cast<const double*>(PyArray_DATA(t0_arr));
        const double* teff_grid = static_cast<const double*>(PyArray_DATA(teff_arr));

        double* dchi2 = static_cast<double*>(PyArray_DATA(dchi2_arr));
        std::int32_t* nwin = static_cast<std::int32_t*>(PyArray_DATA(nwin_arr));
        std::int32_t* ncontrib = static_cast<std::int32_t*>(PyArray_DATA(ncontrib_arr));
        double* neff = static_cast<double*>(PyArray_DATA(neff_arr));
        double* peak = static_cast<double*>(PyArray_DATA(peak_arr));
        double* rho1 = static_cast<double*>(PyArray_DATA(rho1_arr));
        std::int32_t* longest = static_cast<std::int32_t*>(PyArray_DATA(longest_arr));
        const double sigma2 = sigma * sigma;

        Py_BEGIN_ALLOW_THREADS
        #pragma omp parallel for schedule(dynamic, 256)
        for (npy_intp g = 0; g < n_grid; ++g) {
            const double t0 = t0_grid[g];
            const double teff = teff_grid[g];
            const double lo = t0 - teff_coeff * teff;
            const double hi = t0 + teff_coeff * teff;
            const double* begin = time;
            const double* end = time + n;
            const Window window{
                static_cast<npy_intp>(std::upper_bound(begin, end, lo) - begin),
                static_cast<npy_intp>(std::lower_bound(begin, end, hi) - begin),
            };

            const int count = window.size();
            double sw = 0.0;
            double sy = 0.0;
            for (npy_intp i = window.start; i < window.end; ++i) {
                sw += weight[i];
                sy += weight[i] * flux[i];
            }

            if (count < min_pts || !(sw > 0.0)) {
                dchi2[g] = 0.0;
                nwin[g] = 0;
                ncontrib[g] = 0;
                neff[g] = 0.0;
                peak[g] = 0.0;
                rho1[g] = 0.0;
                longest[g] = 0;
                continue;
            }

            const double mu = sy / sw;
            double chi2_flat = 0.0;
            for (npy_intp i = window.start; i < window.end; ++i) {
                const double r = flux[i] - mu;
                chi2_flat += r * r * weight[i];
            }

            Fit fit0 = fit_weighted_line(calc_a0, t0, teff, time, flux, weight, window);
            Fit fit1 = fit_weighted_line(calc_a1, t0, teff, time, flux, weight, window);
            const bool choose0 = fit0.valid && (!fit1.valid || fit0.chi2 < fit1.chi2);
            const Fit fit = choose0 ? fit0 : fit1;
            if (!fit.valid) {
                dchi2[g] = 0.0;
                nwin[g] = 0;
                ncontrib[g] = 0;
                neff[g] = 0.0;
                peak[g] = 0.0;
                rho1[g] = 0.0;
                longest[g] = 0;
                continue;
            }

            const double delta = chi2_flat - fit.chi2;
            double sum_u = 0.0;
            double sum_u2 = 0.0;
            double max_u = 0.0;
            double sum_diff = 0.0;
            int contrib_count = 0;
            int current_run = 0;
            int longest_run = 0;

            for (npy_intp i = window.start; i < window.end; ++i) {
                const double t = time[i];
                const double r_flat = flux[i] - mu;
                const double a = choose0 ? calc_a0(t0, teff, t) : calc_a1(t0, teff, t);
                const double r_anom = flux[i] - (fit.a * a + fit.b);
                const double norm_flat = r_flat * r_flat;
                const double norm_anom = r_anom * r_anom;
                const double diff = norm_flat - norm_anom;
                const double improvement = std::max(diff, 0.0);
                sum_diff += diff;
                sum_u += improvement;
                sum_u2 += improvement * improvement;
                max_u = std::max(max_u, improvement);
                const bool is_contrib = improvement > sigma2;
                if (is_contrib) {
                    ++contrib_count;
                    ++current_run;
                    longest_run = std::max(longest_run, current_run);
                } else {
                    current_run = 0;
                }
            }

            const double mean_diff = sum_diff / static_cast<double>(count);
            double var = 0.0;
            double cov1 = 0.0;
            int pairs = 0;
            bool have_prev = false;
            double prev_centered = 0.0;
            for (npy_intp i = window.start; i < window.end; ++i) {
                const double t = time[i];
                const double r_flat = flux[i] - mu;
                const double a = choose0 ? calc_a0(t0, teff, t) : calc_a1(t0, teff, t);
                const double r_anom = flux[i] - (fit.a * a + fit.b);
                const double diff = r_flat * r_flat - r_anom * r_anom;
                const double centered = diff - mean_diff;
                var += centered * centered;
                if (have_prev) {
                    cov1 += prev_centered * centered;
                    ++pairs;
                }
                prev_centered = centered;
                have_prev = true;
            }
            var /= static_cast<double>(count);
            const double rho = (pairs > 0 && var > 0.0) ? (cov1 / static_cast<double>(pairs)) / var : 0.0;

            dchi2[g] = delta;
            nwin[g] = count;
            ncontrib[g] = contrib_count;
            neff[g] = sum_u2 > 0.0 ? (sum_u * sum_u) / sum_u2 : 0.0;
            peak[g] = sum_u > 0.0 ? max_u / sum_u : 0.0;
            rho1[g] = rho;
            longest[g] = longest_run;
        }
        Py_END_ALLOW_THREADS

        Py_DECREF(time_arr);
        Py_DECREF(flux_arr);
        Py_DECREF(weight_arr);
        Py_DECREF(t0_arr);
        Py_DECREF(teff_arr);

        return Py_BuildValue(
            "NNNNNNN",
            dchi2_arr,
            nwin_arr,
            ncontrib_arr,
            neff_arr,
            peak_arr,
            rho1_arr,
            longest_arr
        );
    }

fail:
    Py_XDECREF(time_arr);
    Py_XDECREF(flux_arr);
    Py_XDECREF(weight_arr);
    Py_XDECREF(t0_arr);
    Py_XDECREF(teff_arr);
    return nullptr;
}

PyObject* extract_clusters(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* t0_obj = nullptr;
    PyObject* teff_obj = nullptr;
    PyObject* dchi2_obj = nullptr;
    double sigma_overlap = 3.0;
    int min_points = 3;
    static const char* kwlist[] = {
        "t0", "teff", "dchi2", "sigma_overlap", "min_points", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOO|di", const_cast<char**>(kwlist),
            &t0_obj, &teff_obj, &dchi2_obj, &sigma_overlap, &min_points)) {
        return nullptr;
    }
    PyArrayObject* t0_arr = as_double_array(t0_obj);
    PyArrayObject* teff_arr = as_double_array(teff_obj);
    PyArrayObject* dchi2_arr = as_double_array(dchi2_obj);
    if (!t0_arr || !teff_arr || !dchi2_arr) {
        Py_XDECREF(t0_arr); Py_XDECREF(teff_arr); Py_XDECREF(dchi2_arr);
        return nullptr;
    }
    if (PyArray_NDIM(t0_arr) != 1 || PyArray_NDIM(teff_arr) != 1 || PyArray_NDIM(dchi2_arr) != 1 ||
        PyArray_DIM(t0_arr, 0) != PyArray_DIM(teff_arr, 0) || PyArray_DIM(t0_arr, 0) != PyArray_DIM(dchi2_arr, 0)) {
        PyErr_SetString(PyExc_ValueError, "t0, teff, and dchi2 must be one-dimensional arrays of equal length");
        Py_DECREF(t0_arr); Py_DECREF(teff_arr); Py_DECREF(dchi2_arr);
        return nullptr;
    }
    const npy_intp n = PyArray_DIM(t0_arr, 0);
    const double* t0 = static_cast<const double*>(PyArray_DATA(t0_arr));
    const double* teff = static_cast<const double*>(PyArray_DATA(teff_arr));
    const double* dchi2 = static_cast<const double*>(PyArray_DATA(dchi2_arr));
    std::vector<unsigned char> remaining(static_cast<std::size_t>(n), 1);
    std::vector<double> rows;
    rows.reserve(static_cast<std::size_t>(n) * 3);
    npy_intp remaining_count = n;
    while (remaining_count > 0) {
        npy_intp best = -1;
        double best_value = -std::numeric_limits<double>::infinity();
        for (npy_intp i = 0; i < n; ++i) {
            if (remaining[static_cast<std::size_t>(i)] && dchi2[i] > best_value) {
                best = i;
                best_value = dchi2[i];
            }
        }
        if (best < 0 || !std::isfinite(best_value)) break;
        const double t0_best = t0[best];
        const double teff_best = teff[best];
        for (npy_intp i = 0; i < n; ++i) {
            if (!remaining[static_cast<std::size_t>(i)]) continue;
            if (std::abs(t0[i] - t0_best) < sigma_overlap * (teff[i] + teff_best)) {
                remaining[static_cast<std::size_t>(i)] = 0;
                --remaining_count;
            }
        }
        rows.push_back(t0_best);
        rows.push_back(teff_best);
        rows.push_back(best_value);
        if (remaining_count < min_points) break;
    }
    npy_intp dims[2] = {static_cast<npy_intp>(rows.size() / 3), 3};
    PyArrayObject* out = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(2, dims, NPY_DOUBLE));
    if (out) std::copy(rows.begin(), rows.end(), static_cast<double*>(PyArray_DATA(out)));
    Py_DECREF(t0_arr); Py_DECREF(teff_arr); Py_DECREF(dchi2_arr);
    return reinterpret_cast<PyObject*>(out);
}

PyObject* fit_pspl(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* time_obj = nullptr;
    PyObject* flux_obj = nullptr;
    PyObject* ferr_obj = nullptr;
    PyObject* p0_obj = nullptr;
    int maxiter = 1000;
    double damping_parameter = 1e-6;
    double tol = 1e-3;
    double u0_min = 0.01;
    int min_t0_support_points = 3;
    double t0_support_tE_coeff = 3.0;

    static const char* kwlist[] = {
        "time", "flux", "ferr", "p0", "maxiter", "damping_parameter", "tol",
        "u0_min", "min_t0_support_points", "t0_support_tE_coeff", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOO|idddid",
            const_cast<char**>(kwlist),
            &time_obj,
            &flux_obj,
            &ferr_obj,
            &p0_obj,
            &maxiter,
            &damping_parameter,
            &tol,
            &u0_min,
            &min_t0_support_points,
            &t0_support_tE_coeff
        )) {
        return nullptr;
    }

    PyArrayObject* time_arr = as_double_array(time_obj);
    PyArrayObject* flux_arr = as_double_array(flux_obj);
    PyArrayObject* ferr_arr = as_double_array(ferr_obj);
    PyArrayObject* p0_arr = as_double_array(p0_obj);
    if (!time_arr || !flux_arr || !ferr_arr || !p0_arr) {
        Py_XDECREF(time_arr);
        Py_XDECREF(flux_arr);
        Py_XDECREF(ferr_arr);
        Py_XDECREF(p0_arr);
        return nullptr;
    }
    if (PyArray_NDIM(time_arr) != 1 || PyArray_NDIM(flux_arr) != 1 ||
        PyArray_NDIM(ferr_arr) != 1 || PyArray_NDIM(p0_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "time, flux, ferr, and p0 must be one-dimensional arrays");
        Py_DECREF(time_arr);
        Py_DECREF(flux_arr);
        Py_DECREF(ferr_arr);
        Py_DECREF(p0_arr);
        return nullptr;
    }

    const npy_intp n = PyArray_DIM(time_arr, 0);
    if (PyArray_DIM(flux_arr, 0) != n || PyArray_DIM(ferr_arr, 0) != n || PyArray_DIM(p0_arr, 0) < 3) {
        PyErr_SetString(PyExc_ValueError, "invalid input shapes for fit_pspl");
        Py_DECREF(time_arr);
        Py_DECREF(flux_arr);
        Py_DECREF(ferr_arr);
        Py_DECREF(p0_arr);
        return nullptr;
    }
    if (n < 4) {
        PyErr_SetString(PyExc_ValueError, "Need at least 4 data points for PSPL fit.");
        Py_DECREF(time_arr);
        Py_DECREF(flux_arr);
        Py_DECREF(ferr_arr);
        Py_DECREF(p0_arr);
        return nullptr;
    }

    const double* time = static_cast<const double*>(PyArray_DATA(time_arr));
    const double* flux = static_cast<const double*>(PyArray_DATA(flux_arr));
    const double* ferr = static_cast<const double*>(PyArray_DATA(ferr_arr));
    const double* p0 = static_cast<const double*>(PyArray_DATA(p0_arr));

    double q[3] = {
        p0[0],
        std::log(std::max(std::abs(p0[1]), 1e-12)),
        std::abs(p0[2]) < u0_min ? std::copysign(u0_min, p0[2] == 0.0 ? 1.0 : p0[2]) : p0[2],
    };
    double lambda = std::max(damping_parameter, 1e-12);
    std::vector<double> residual;
    Fit fit;
    pspl_residuals(q, time, flux, ferr, n, residual, &fit, u0_min, min_t0_support_points, t0_support_tE_coeff);
    double chi2 = sumsq(residual);

    for (int iter = 0; iter < maxiter; ++iter) {
        double jac_cols[3][1];  // placeholder to keep the stack small; columns live in vectors below.
        (void)jac_cols;
        std::vector<double> jcol[3];
        for (int k = 0; k < 3; ++k) {
            double qp[3] = {q[0], q[1], q[2]};
            double qm[3] = {q[0], q[1], q[2]};
            const double step = 1e-5 * std::max(std::abs(q[k]), 1.0);
            qp[k] += step;
            qm[k] -= step;
            std::vector<double> rp;
            std::vector<double> rm;
            pspl_residuals(qp, time, flux, ferr, n, rp, nullptr, u0_min, min_t0_support_points, t0_support_tE_coeff);
            pspl_residuals(qm, time, flux, ferr, n, rm, nullptr, u0_min, min_t0_support_points, t0_support_tE_coeff);
            jcol[k].resize(static_cast<size_t>(n));
            const double inv = 1.0 / (2.0 * step);
            for (npy_intp i = 0; i < n; ++i) {
                jcol[k][static_cast<size_t>(i)] = (rp[static_cast<size_t>(i)] - rm[static_cast<size_t>(i)]) * inv;
            }
        }

        double jtj[3][3] = {};
        double rhs[3] = {};
        for (int a = 0; a < 3; ++a) {
            rhs[a] = -dot_vec(jcol[a], residual);
            for (int b = 0; b < 3; ++b) {
                jtj[a][b] = dot_vec(jcol[a], jcol[b]);
            }
        }

        double step_q[3] = {};
        bool accepted = false;
        double best_trial_chi2 = chi2;
        double best_trial_q[3] = {q[0], q[1], q[2]};
        for (int attempt = 0; attempt < 12; ++attempt) {
            double a[3][3] = {
                {jtj[0][0], jtj[0][1], jtj[0][2]},
                {jtj[1][0], jtj[1][1], jtj[1][2]},
                {jtj[2][0], jtj[2][1], jtj[2][2]},
            };
            for (int k = 0; k < 3; ++k) {
                a[k][k] += lambda * std::max(jtj[k][k], 1.0);
            }
            if (!solve_3x3(a, rhs, step_q)) {
                lambda *= 10.0;
                continue;
            }
            double q_trial[3] = {q[0] + step_q[0], q[1] + step_q[1], q[2] + step_q[2]};
            q_trial[1] = std::min(std::max(q_trial[1], std::log(1e-6)), std::log(1e8));
            if (std::abs(q_trial[2]) < u0_min) {
                q_trial[2] = std::copysign(u0_min, q_trial[2] == 0.0 ? q[2] : q_trial[2]);
            }
            std::vector<double> trial_residual;
            pspl_residuals(q_trial, time, flux, ferr, n, trial_residual, nullptr, u0_min, min_t0_support_points, t0_support_tE_coeff);
            const double trial_chi2 = sumsq(trial_residual);
            if (std::isfinite(trial_chi2) && trial_chi2 < best_trial_chi2) {
                best_trial_chi2 = trial_chi2;
                best_trial_q[0] = q_trial[0];
                best_trial_q[1] = q_trial[1];
                best_trial_q[2] = q_trial[2];
                residual.swap(trial_residual);
                accepted = true;
                break;
            }
            lambda *= 10.0;
        }

        if (!accepted) {
            break;
        }
        const double prev_chi2 = chi2;
        q[0] = best_trial_q[0];
        q[1] = best_trial_q[1];
        q[2] = best_trial_q[2];
        chi2 = best_trial_chi2;
        lambda = std::max(lambda * 0.3, 1e-12);
        if (std::abs(prev_chi2 - chi2) < tol) {
            break;
        }
    }

    pspl_residuals(q, time, flux, ferr, n, residual, &fit, u0_min, min_t0_support_points, t0_support_tE_coeff);
    chi2 = sumsq(residual);

    npy_intp param_dims[1] = {3};
    npy_intp data_dims[1] = {n};
    PyArrayObject* params_arr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, param_dims, NPY_DOUBLE));
    PyArrayObject* model_arr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, data_dims, NPY_DOUBLE));
    PyArrayObject* residual_arr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, data_dims, NPY_DOUBLE));
    if (!params_arr || !model_arr || !residual_arr) {
        Py_XDECREF(params_arr);
        Py_XDECREF(model_arr);
        Py_XDECREF(residual_arr);
        Py_DECREF(time_arr);
        Py_DECREF(flux_arr);
        Py_DECREF(ferr_arr);
        Py_DECREF(p0_arr);
        return nullptr;
    }

    const double t0 = q[0];
    const double tE = std::max(std::exp(q[1]), 1e-12);
    const double u0 = q[2];
    double* params = static_cast<double*>(PyArray_DATA(params_arr));
    params[0] = t0;
    params[1] = tE;
    params[2] = u0;
    double* model = static_cast<double*>(PyArray_DATA(model_arr));
    double* residual_out = static_cast<double*>(PyArray_DATA(residual_arr));
    for (npy_intp i = 0; i < n; ++i) {
        model[i] = fit.a * pspl_magnification(t0, tE, u0, time[i]) + fit.b;
        residual_out[i] = flux[i] - model[i];
    }

    Py_DECREF(time_arr);
    Py_DECREF(flux_arr);
    Py_DECREF(ferr_arr);
    Py_DECREF(p0_arr);

    return Py_BuildValue("NdddNN", params_arr, fit.a, fit.b, chi2, model_arr, residual_arr);
}

PyMethodDef methods[] = {
    {
        "run_grid",
        reinterpret_cast<PyCFunction>(run_grid),
        METH_VARARGS | METH_KEYWORDS,
        "Evaluate anomaly grid points with a plain C++ for-loop backend.",
    },
    {
        "unimodal_mask",
        reinterpret_cast<PyCFunction>(unimodal_mask),
        METH_VARARGS | METH_KEYWORDS,
        "Classify template-grid candidates with the native one-lobe test.",
    },
    {
        "fit_pspl",
        reinterpret_cast<PyCFunction>(fit_pspl),
        METH_VARARGS | METH_KEYWORDS,
        "Fit a PSPL single-lens model with a small C++ Levenberg-Marquardt solver.",
    },
    {
        "extract_clusters",
        reinterpret_cast<PyCFunction>(extract_clusters),
        METH_VARARGS | METH_KEYWORDS,
        "Extract non-overlapping anomaly clusters from a grid result.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_cpp_grid",
    "C++ grid backend for jacscanomaly.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__cpp_grid(void) {
    import_array();
    return PyModule_Create(&module);
}
