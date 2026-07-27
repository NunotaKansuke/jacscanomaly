#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include <numpy/arrayobject.h>

#include "VBMicrolensingLibrary.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kNParam = 6;

PyArrayObject* as_double_array(PyObject* obj) {
    return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
}

struct LinearFit {
    double fs = 0.0;
    double fb = 0.0;
    bool valid = false;
};

double sumsq(const std::vector<double>& x) {
    double out = 0.0;
    for (double v : x) out += v * v;
    return out;
}

LinearFit solve_fluxes(const std::vector<double>& A, const double* flux, const double* ferr) {
    const size_t n = A.size();
    double sw = 0.0, sx = 0.0, sy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double fe = std::max(ferr[i], 1e-12);
        const double w = 1.0 / (fe * fe);
        sw += w;
        sx += w * A[i];
        sy += w * flux[i];
    }
    if (!(sw > 0.0)) return {};
    const double xm = sx / sw, ym = sy / sw;
    double wxx = 0.0, wxy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double fe = std::max(ferr[i], 1e-12);
        const double w = 1.0 / (fe * fe);
        wxx += w * (A[i] - xm) * (A[i] - xm);
        wxy += w * (A[i] - xm) * (flux[i] - ym);
    }
    if (!(wxx > 0.0) || !std::isfinite(wxx)) return {};
    return {wxy / wxx, ym - (wxy / wxx) * xm, true};
}

bool solve_linear(double a[kNParam][kNParam], double b[kNParam], double x[kNParam]) {
    double m[kNParam][kNParam + 1];
    for (int i = 0; i < kNParam; ++i) {
        for (int j = 0; j < kNParam; ++j) m[i][j] = a[i][j];
        m[i][kNParam] = b[i];
    }
    for (int col = 0; col < kNParam; ++col) {
        int pivot = col;
        for (int row = col + 1; row < kNParam; ++row) {
            if (std::abs(m[row][col]) > std::abs(m[pivot][col])) pivot = row;
        }
        if (std::abs(m[pivot][col]) < 1e-30) return false;
        if (pivot != col) {
            for (int j = col; j <= kNParam; ++j) std::swap(m[col][j], m[pivot][j]);
        }
        const double inv = 1.0 / m[col][col];
        for (int j = col; j <= kNParam; ++j) m[col][j] *= inv;
        for (int row = 0; row < kNParam; ++row) {
            if (row == col) continue;
            const double f = m[row][col];
            for (int j = col; j <= kNParam; ++j) m[row][j] -= f * m[col][j];
        }
    }
    for (int i = 0; i < kNParam; ++i) x[i] = m[i][kNParam];
    return true;
}

void clamp_q(double q[kNParam], double max_piE) {
    q[1] = std::clamp(q[1], std::log(1e-6), std::log(1e8));
    // Keep trial sources inside the well-behaved part of VBM's ESPL lookup
    // table.  Larger values are not useful for the point-lens FSPL workflow
    // and make the library emit a diagnostic for every model evaluation.
    q[3] = std::clamp(q[3], -50.0, std::log(10.0));
    if (std::isfinite(max_piE) && max_piE > 0.0) {
        q[4] = std::clamp(q[4], -max_piE, max_piE);
        q[5] = std::clamp(q[5], -max_piE, max_piE);
    }
}

bool residuals(
    VBMicrolensing& vbm,
    const double q[kNParam],
    const double* time, const double* flux, const double* ferr, npy_intp n,
    std::vector<double>& residual, std::vector<double>* A_out = nullptr, LinearFit* fit_out = nullptr
) {
    std::vector<double> A(static_cast<size_t>(n));
    std::vector<double> y1(static_cast<size_t>(n));
    std::vector<double> y2(static_cast<size_t>(n));
    // VBM's bundled Sun table stores time as JD-2450000.  Its table loader
    // applies that offset internally, so its parallax API must receive the
    // same reduced convention even when jacscanomaly's public API uses JD.
    constexpr double kVBMTimeOffset = 2450000.0;
    std::vector<double> vbm_time(static_cast<size_t>(n));
    for (npy_intp i = 0; i < n; ++i) vbm_time[static_cast<size_t>(i)] = time[i] - kVBMTimeOffset;
    // VBM convention: [u0, log(tE), t0, log(rho), piE_1, piE_2].
    double p[kNParam] = {q[2], q[1], q[0] - kVBMTimeOffset, q[3], q[4], q[5]};
    // ESPLMag2, used internally by this VBM call, already changes method away
    // from the finite-source regime when appropriate.  Give VBM every datum
    // and leave that numerical decision to its native implementation.
    vbm.ESPLLightCurveParallax(p, vbm_time.data(), A.data(), y1.data(), y2.data(), static_cast<int>(n));
    for (double a : A) {
        if (!std::isfinite(a) || a <= 0.0) {
            residual.assign(static_cast<size_t>(n), 1e100);
            return false;
        }
    }
    const LinearFit fit = solve_fluxes(A, flux, ferr);
    if (!fit.valid) {
        residual.assign(static_cast<size_t>(n), 1e100);
        return false;
    }
    residual.resize(static_cast<size_t>(n));
    for (npy_intp i = 0; i < n; ++i) {
        const double fe = std::max(ferr[i], 1e-12);
        residual[static_cast<size_t>(i)] = (flux[i] - (fit.fs * A[static_cast<size_t>(i)] + fit.fb)) / fe;
    }
    if (A_out) *A_out = std::move(A);
    if (fit_out) *fit_out = fit;
    return true;
}

PyObject* fit_fspl_parallax(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject *time_obj = nullptr, *flux_obj = nullptr, *ferr_obj = nullptr, *p0_obj = nullptr;
    const char *coordinates = nullptr, *sun_table = nullptr, *espl_table = nullptr;
    int maxiter = 200;
    double damping_parameter = 1e-4, tol = 1e-5, vbm_tol = 1e-4, vbm_reltol = 1e-4;
    double max_piE = 5.0;
    static const char* kwlist[] = {
        "time", "flux", "ferr", "p0", "coordinates", "sun_table", "espl_table",
        "maxiter", "damping_parameter", "tol", "vbm_tol", "vbm_reltol", "max_piE", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOsss|iddddd", const_cast<char**>(kwlist),
            &time_obj, &flux_obj, &ferr_obj, &p0_obj, &coordinates, &sun_table, &espl_table,
            &maxiter, &damping_parameter, &tol, &vbm_tol, &vbm_reltol, &max_piE)) {
        return nullptr;
    }

    PyArrayObject *time_arr = as_double_array(time_obj), *flux_arr = as_double_array(flux_obj);
    PyArrayObject *ferr_arr = as_double_array(ferr_obj), *p0_arr = as_double_array(p0_obj);
    npy_intp n = 0;
    if (!time_arr || !flux_arr || !ferr_arr || !p0_arr) goto fail;
    n = PyArray_DIM(time_arr, 0);
    if (PyArray_NDIM(time_arr) != 1 || PyArray_NDIM(flux_arr) != 1 || PyArray_NDIM(ferr_arr) != 1 ||
        PyArray_NDIM(p0_arr) != 1 || PyArray_DIM(flux_arr, 0) != n || PyArray_DIM(ferr_arr, 0) != n ||
        PyArray_DIM(p0_arr, 0) < kNParam || n < 7) {
        PyErr_SetString(PyExc_ValueError, "fit_fspl_parallax requires equal one-dimensional arrays, p0 with 6 values, and at least 7 data points.");
        goto fail;
    }
    {
        VBMicrolensing vbm;
        vbm.Tol = vbm_tol;
        vbm.RelTol = vbm_reltol;
        vbm.t_in_HJD = true;
        vbm.parallaxsystem = 1;
        std::vector<char> coord(coordinates, coordinates + std::strlen(coordinates) + 1);
        std::vector<char> sun(sun_table, sun_table + std::strlen(sun_table) + 1);
        vbm.SetObjectCoordinates(coord.data());
        if (!vbm.AreCoordinatesSet()) {
            PyErr_SetString(PyExc_ValueError, "coordinates must be 'HH:MM:SS.s +/-DD:MM:SS.s'.");
            goto fail;
        }
        vbm.LoadSunTable(sun.data());
        vbm.LoadESPLTable(espl_table);

        const double* p0 = static_cast<const double*>(PyArray_DATA(p0_arr));
        double q[kNParam] = {p0[0], std::log(std::max(std::abs(p0[1]), 1e-12)), p0[2],
                              std::log(std::max(std::abs(p0[3]), 1e-12)), p0[4], p0[5]};
        clamp_q(q, max_piE);
        const double* time = static_cast<const double*>(PyArray_DATA(time_arr));
        const double* flux = static_cast<const double*>(PyArray_DATA(flux_arr));
        const double* ferr = static_cast<const double*>(PyArray_DATA(ferr_arr));
        std::vector<double> residual;
        if (!residuals(vbm, q, time, flux, ferr, n, residual)) {
            PyErr_SetString(PyExc_RuntimeError, "VBMicrolensing failed to evaluate the starting FSPL-parallax model.");
            goto fail;
        }
        double chi2 = sumsq(residual);
        double lambda = std::max(damping_parameter, 1e-12);
        const double fd[kNParam] = {1e-4, 1e-4, 1e-5, 1e-4, 1e-5, 1e-5};
        for (int iter = 0; iter < maxiter; ++iter) {
            std::array<std::vector<double>, kNParam> jac;
            for (int k = 0; k < kNParam; ++k) {
                double qp[kNParam], qm[kNParam];
                std::copy(q, q + kNParam, qp); std::copy(q, q + kNParam, qm);
                qp[k] += fd[k]; qm[k] -= fd[k]; clamp_q(qp, max_piE); clamp_q(qm, max_piE);
                std::vector<double> rp, rm;
                residuals(vbm, qp, time, flux, ferr, n, rp);
                residuals(vbm, qm, time, flux, ferr, n, rm);
                jac[k].resize(static_cast<size_t>(n));
                const double inv = 1.0 / (qp[k] - qm[k]);
                for (npy_intp i = 0; i < n; ++i) jac[k][static_cast<size_t>(i)] = (rp[static_cast<size_t>(i)] - rm[static_cast<size_t>(i)]) * inv;
            }
            double jtj[kNParam][kNParam] = {}, rhs[kNParam] = {};
            for (int a = 0; a < kNParam; ++a) for (int b = 0; b < kNParam; ++b) {
                double dot = 0.0;
                for (npy_intp i = 0; i < n; ++i) dot += jac[a][static_cast<size_t>(i)] * jac[b][static_cast<size_t>(i)];
                jtj[a][b] = dot;
                if (b == 0) for (npy_intp i = 0; i < n; ++i) rhs[a] -= jac[a][static_cast<size_t>(i)] * residual[static_cast<size_t>(i)];
            }
            bool accepted = false;
            double best_q[kNParam];
            double best_chi2 = chi2;
            for (int attempt = 0; attempt < 12; ++attempt) {
                double mat[kNParam][kNParam], step[kNParam] = {};
                for (int a = 0; a < kNParam; ++a) for (int b = 0; b < kNParam; ++b) {
                    mat[a][b] = jtj[a][b];
                    if (a == b) mat[a][b] += lambda * std::max(jtj[a][a], 1.0);
                }
                if (!solve_linear(mat, rhs, step)) { lambda *= 10.0; continue; }
                // In the PSPL-like regime rho is only weakly identified.  A
                // finite-difference Jacobian can then propose an unphysical
                // multi-decade logrho jump even though the other parameters
                // are well constrained.  Keep this coordinate in a local
                // trust region; accepted LM iterations can still move rho.
                step[3] = std::clamp(step[3], -0.25, 0.25);
                double trial[kNParam];
                for (int k = 0; k < kNParam; ++k) trial[k] = q[k] + step[k];
                clamp_q(trial, max_piE);
                std::vector<double> trial_residual;
                residuals(vbm, trial, time, flux, ferr, n, trial_residual);
                const double trial_chi2 = sumsq(trial_residual);
                if (std::isfinite(trial_chi2) && trial_chi2 < best_chi2) {
                    std::copy(trial, trial + kNParam, best_q); best_chi2 = trial_chi2;
                    residual.swap(trial_residual); accepted = true; break;
                }
                lambda *= 10.0;
            }
            if (!accepted) break;
            const double improvement = chi2 - best_chi2;
            std::copy(best_q, best_q + kNParam, q); chi2 = best_chi2;
            lambda = std::max(lambda * 0.3, 1e-12);
            if (improvement < tol) break;
        }
        std::vector<double> A; LinearFit fit;
        residuals(vbm, q, time, flux, ferr, n, residual, &A, &fit);
        chi2 = sumsq(residual);
        npy_intp pdims[1] = {kNParam}, ddims[1] = {n};
        auto* params = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, pdims, NPY_DOUBLE));
        auto* model = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, ddims, NPY_DOUBLE));
        auto* raw_residual = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, ddims, NPY_DOUBLE));
        if (!params || !model || !raw_residual) { Py_XDECREF(params); Py_XDECREF(model); Py_XDECREF(raw_residual); goto fail; }
        double* po = static_cast<double*>(PyArray_DATA(params));
        po[0] = q[0]; po[1] = std::exp(q[1]); po[2] = q[2]; po[3] = std::exp(q[3]); po[4] = q[4]; po[5] = q[5];
        double* mo = static_cast<double*>(PyArray_DATA(model));
        double* ro = static_cast<double*>(PyArray_DATA(raw_residual));
        for (npy_intp i = 0; i < n; ++i) { mo[i] = fit.fs * A[static_cast<size_t>(i)] + fit.fb; ro[i] = flux[i] - mo[i]; }
        Py_DECREF(time_arr); Py_DECREF(flux_arr); Py_DECREF(ferr_arr); Py_DECREF(p0_arr);
        return Py_BuildValue("NdddNN", params, fit.fs, fit.fb, chi2, model, raw_residual);
    }
fail:
    Py_XDECREF(time_arr); Py_XDECREF(flux_arr); Py_XDECREF(ferr_arr); Py_XDECREF(p0_arr);
    return nullptr;
}

PyMethodDef methods[] = {{"fit_fspl_parallax", reinterpret_cast<PyCFunction>(fit_fspl_parallax), METH_VARARGS | METH_KEYWORDS,
    "Fit annual-parallax FSPL with VBMicrolensing and finite-difference C++ LM."}, {nullptr, nullptr, 0, nullptr}};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_vbm_cpp", "VBMicrolensing C++ backend.", -1, methods};
}  // namespace

PyMODINIT_FUNC PyInit__vbm_cpp(void) { import_array(); return PyModule_Create(&module); }
