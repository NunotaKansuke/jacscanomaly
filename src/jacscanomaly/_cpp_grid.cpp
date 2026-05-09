#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

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

struct Fit {
    double a = 0.0;
    double b = 0.0;
    double chi2 = 0.0;
    bool valid = false;
};

template <typename Func>
Fit fit_weighted_line(
    Func basis,
    double t0,
    double teff,
    const double* time,
    const double* flux,
    const double* weight,
    npy_intp n,
    double lo,
    double hi
) {
    double sw = 0.0;
    double sx = 0.0;
    double sy = 0.0;

    for (npy_intp i = 0; i < n; ++i) {
        const double t = time[i];
        if (!(t > lo && t < hi)) {
            continue;
        }
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
    for (npy_intp i = 0; i < n; ++i) {
        const double t = time[i];
        if (!(t > lo && t < hi)) {
            continue;
        }
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
    for (npy_intp i = 0; i < n; ++i) {
        const double t = time[i];
        if (!(t > lo && t < hi)) {
            continue;
        }
        const double model = fit.a * basis(t0, teff, t) + fit.b;
        const double r = flux[i] - model;
        chi2 += r * r * weight[i];
    }
    fit.chi2 = chi2;
    return fit;
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
        for (npy_intp g = 0; g < n_grid; ++g) {
            const double t0 = t0_grid[g];
            const double teff = teff_grid[g];
            const double lo = t0 - teff_coeff * teff;
            const double hi = t0 + teff_coeff * teff;

            int count = 0;
            double sw = 0.0;
            double sy = 0.0;
            for (npy_intp i = 0; i < n; ++i) {
                const double t = time[i];
                if (t > lo && t < hi) {
                    ++count;
                    sw += weight[i];
                    sy += weight[i] * flux[i];
                }
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
            for (npy_intp i = 0; i < n; ++i) {
                const double t = time[i];
                if (t > lo && t < hi) {
                    const double r = flux[i] - mu;
                    chi2_flat += r * r * weight[i];
                }
            }

            Fit fit0 = fit_weighted_line(calc_a0, t0, teff, time, flux, weight, n, lo, hi);
            Fit fit1 = fit_weighted_line(calc_a1, t0, teff, time, flux, weight, n, lo, hi);
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

            for (npy_intp i = 0; i < n; ++i) {
                const double t = time[i];
                if (!(t > lo && t < hi)) {
                    continue;
                }
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
            for (npy_intp i = 0; i < n; ++i) {
                const double t = time[i];
                if (!(t > lo && t < hi)) {
                    have_prev = false;
                    continue;
                }
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

PyMethodDef methods[] = {
    {
        "run_grid",
        reinterpret_cast<PyCFunction>(run_grid),
        METH_VARARGS | METH_KEYWORDS,
        "Evaluate anomaly grid points with a plain C++ for-loop backend.",
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
