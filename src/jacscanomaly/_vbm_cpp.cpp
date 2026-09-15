#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include <numpy/arrayobject.h>

#include "VBMicrolensingLibrary.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {

PyArrayObject* as_double_array(PyObject* object) {
    return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)
    );
}

PyObject* finite_source_magnification(
    PyObject*, PyObject* args, PyObject* kwargs
) {
    PyObject* u_object = nullptr;
    PyObject* table_object = Py_None;
    double rho = 0.0;
    double tol = 1.0e-4;
    double reltol = 1.0e-4;
    static const char* kwlist[] = {
        "u", "rho", "espl_table", "tol", "reltol", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "Od|Odd",
            const_cast<char**>(kwlist),
            &u_object,
            &rho,
            &table_object,
            &tol,
            &reltol
        )) {
        return nullptr;
    }

    PyArrayObject* u_array = as_double_array(u_object);
    if (u_array == nullptr) return nullptr;
    if (
        PyArray_NDIM(u_array) != 1
        || !std::isfinite(rho)
        || !(rho > 0.0)
        || !std::isfinite(tol)
        || !(tol > 0.0)
        || !std::isfinite(reltol)
        || !(reltol > 0.0)
    ) {
        PyErr_SetString(
            PyExc_ValueError,
            "u must be one-dimensional; rho, tol, and reltol must be finite and positive."
        );
        Py_DECREF(u_array);
        return nullptr;
    }

    const char* table = nullptr;
    if (table_object != Py_None) {
        table = PyUnicode_AsUTF8(table_object);
        if (table == nullptr) {
            Py_DECREF(u_array);
            return nullptr;
        }
    }

    const npy_intp n = PyArray_DIM(u_array, 0);
    npy_intp dimensions[1] = {n};
    auto* output = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(1, dimensions, NPY_DOUBLE)
    );
    if (output == nullptr) {
        Py_DECREF(u_array);
        return nullptr;
    }

    try {
        VBMicrolensing magnifier;
        magnifier.Tol = tol;
        magnifier.RelTol = reltol;
        if (table != nullptr && std::strlen(table) > 0) {
            magnifier.LoadESPLTable(table);
        }

        const double* input = static_cast<const double*>(
            PyArray_DATA(u_array)
        );
        double* values = static_cast<double*>(PyArray_DATA(output));
        for (npy_intp index = 0; index < n; ++index) {
            if (!std::isfinite(input[index])) {
                throw std::invalid_argument(
                    "u must contain only finite values."
                );
            }
            values[index] = magnifier.ESPLMag2(
                std::abs(input[index]), rho
            );
            if (!std::isfinite(values[index]) || values[index] <= 0.0) {
                throw std::runtime_error(
                    "compiled finite-source magnification is invalid."
                );
            }
        }
    } catch (const std::exception& exception) {
        Py_DECREF(u_array);
        Py_DECREF(output);
        PyErr_SetString(PyExc_ValueError, exception.what());
        return nullptr;
    }

    Py_DECREF(u_array);
    return reinterpret_cast<PyObject*>(output);
}

PyObject* score_finite_source_seeds(
    PyObject*, PyObject* args, PyObject* kwargs
) {
    PyObject* time_object = nullptr;
    PyObject* flux_object = nullptr;
    PyObject* ferr_object = nullptr;
    PyObject* seeds_object = nullptr;
    PyObject* table_object = Py_None;
    double tol = 1.0e-4;
    double reltol = 1.0e-4;
    static const char* kwlist[] = {
        "time", "flux", "ferr", "seeds", "espl_table", "tol", "reltol", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOO|Odd",
            const_cast<char**>(kwlist),
            &time_object,
            &flux_object,
            &ferr_object,
            &seeds_object,
            &table_object,
            &tol,
            &reltol
        )) {
        return nullptr;
    }

    PyArrayObject* time_array = as_double_array(time_object);
    PyArrayObject* flux_array = as_double_array(flux_object);
    PyArrayObject* ferr_array = as_double_array(ferr_object);
    PyArrayObject* seeds_array = as_double_array(seeds_object);
    if (
        time_array == nullptr
        || flux_array == nullptr
        || ferr_array == nullptr
        || seeds_array == nullptr
    ) {
        Py_XDECREF(time_array);
        Py_XDECREF(flux_array);
        Py_XDECREF(ferr_array);
        Py_XDECREF(seeds_array);
        return nullptr;
    }

    const bool valid_shape = (
        PyArray_NDIM(time_array) == 1
        && PyArray_NDIM(flux_array) == 1
        && PyArray_NDIM(ferr_array) == 1
        && PyArray_NDIM(seeds_array) == 2
        && PyArray_DIM(time_array, 0) == PyArray_DIM(flux_array, 0)
        && PyArray_DIM(time_array, 0) == PyArray_DIM(ferr_array, 0)
        && PyArray_DIM(seeds_array, 1) == 4
        && PyArray_DIM(seeds_array, 0) > 0
        && PyArray_DIM(time_array, 0) > 0
    );
    if (!valid_shape || !std::isfinite(tol) || !(tol > 0.0)
        || !std::isfinite(reltol) || !(reltol > 0.0)) {
        PyErr_SetString(
            PyExc_ValueError,
            "time, flux, and ferr must be equal-length 1-D arrays; "
            "seeds must have shape (n_seeds, 4); tolerances must be positive."
        );
        Py_DECREF(time_array);
        Py_DECREF(flux_array);
        Py_DECREF(ferr_array);
        Py_DECREF(seeds_array);
        return nullptr;
    }

    const char* table = nullptr;
    if (table_object != Py_None) {
        table = PyUnicode_AsUTF8(table_object);
        if (table == nullptr) {
            Py_DECREF(time_array);
            Py_DECREF(flux_array);
            Py_DECREF(ferr_array);
            Py_DECREF(seeds_array);
            return nullptr;
        }
    }

    const npy_intp n_points = PyArray_DIM(time_array, 0);
    const npy_intp n_seeds = PyArray_DIM(seeds_array, 0);
    npy_intp dimensions[1] = {n_seeds};
    auto* output = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(1, dimensions, NPY_DOUBLE)
    );
    if (output == nullptr) {
        Py_DECREF(time_array);
        Py_DECREF(flux_array);
        Py_DECREF(ferr_array);
        Py_DECREF(seeds_array);
        return nullptr;
    }

    const double* time = static_cast<const double*>(PyArray_DATA(time_array));
    const double* flux = static_cast<const double*>(PyArray_DATA(flux_array));
    const double* ferr = static_cast<const double*>(PyArray_DATA(ferr_array));
    const double* seeds = static_cast<const double*>(PyArray_DATA(seeds_array));
    double* scores = static_cast<double*>(PyArray_DATA(output));

    bool invalid_data = false;
    for (npy_intp index = 0; index < n_points; ++index) {
        if (!std::isfinite(time[index]) || !std::isfinite(flux[index])
            || !std::isfinite(ferr[index]) || !(ferr[index] > 0.0)) {
            invalid_data = true;
            break;
        }
    }
    if (invalid_data) {
        PyErr_SetString(PyExc_ValueError, "time, flux, and ferr must be finite with positive ferr.");
        Py_DECREF(time_array);
        Py_DECREF(flux_array);
        Py_DECREF(ferr_array);
        Py_DECREF(seeds_array);
        Py_DECREF(output);
        return nullptr;
    }

    try {
        VBMicrolensing magnifier;
        magnifier.Tol = tol;
        magnifier.RelTol = reltol;
        if (table != nullptr && std::strlen(table) > 0) {
            magnifier.LoadESPLTable(table);
        }

        for (npy_intp seed_index = 0; seed_index < n_seeds; ++seed_index) {
            const double* seed = seeds + seed_index * 4;
            double& score = scores[seed_index];
            score = std::numeric_limits<double>::infinity();
            if (!std::isfinite(seed[0]) || !std::isfinite(seed[1])
                || !std::isfinite(seed[2]) || !std::isfinite(seed[3])
                || seed[1] == 0.0) {
                continue;
            }
            const double rho = std::exp(std::max(-50.0, std::min(10.0, seed[3])));
            const double tE = std::abs(seed[1]);
            double weight_sum = 0.0;
            double weighted_magnification = 0.0;
            double weighted_flux = 0.0;
            bool invalid_seed = false;
            for (npy_intp index = 0; index < n_points; ++index) {
                const double u = std::sqrt(
                    std::pow((time[index] - seed[0]) / tE, 2.0)
                    + seed[2] * seed[2]
                );
                const double magnification = magnifier.ESPLMag2(std::abs(u), rho);
                if (!std::isfinite(magnification) || !(magnification > 0.0)) {
                    invalid_seed = true;
                    break;
                }
                const double error = std::max(ferr[index], 1.0e-12);
                const double weight = 1.0 / (error * error);
                weight_sum += weight;
                weighted_magnification += weight * magnification;
                weighted_flux += weight * flux[index];
            }
            if (invalid_seed || !(weight_sum > 0.0) || !std::isfinite(weight_sum)) {
                continue;
            }
            const double mean_magnification = weighted_magnification / weight_sum;
            const double mean_flux = weighted_flux / weight_sum;
            double denominator = 0.0;
            double numerator = 0.0;
            for (npy_intp index = 0; index < n_points; ++index) {
                const double u = std::sqrt(
                    std::pow((time[index] - seed[0]) / tE, 2.0)
                    + seed[2] * seed[2]
                );
                const double magnification = magnifier.ESPLMag2(std::abs(u), rho);
                const double error = std::max(ferr[index], 1.0e-12);
                const double weight = 1.0 / (error * error);
                const double centered_magnification = magnification - mean_magnification;
                const double centered_flux = flux[index] - mean_flux;
                denominator += weight * centered_magnification * centered_magnification;
                numerator += weight * centered_magnification * centered_flux;
            }
            if (!std::isfinite(denominator) || !(denominator > 0.0)) {
                continue;
            }
            const double source_flux = numerator / denominator;
            const double blend_flux = mean_flux - source_flux * mean_magnification;
            if (!std::isfinite(source_flux) || !std::isfinite(blend_flux)) {
                continue;
            }
            double chi2 = 0.0;
            for (npy_intp index = 0; index < n_points; ++index) {
                const double u = std::sqrt(
                    std::pow((time[index] - seed[0]) / tE, 2.0)
                    + seed[2] * seed[2]
                );
                const double magnification = magnifier.ESPLMag2(std::abs(u), rho);
                const double error = std::max(ferr[index], 1.0e-12);
                const double residual = (
                    flux[index] - (source_flux * magnification + blend_flux)
                ) / error;
                chi2 += residual * residual;
            }
            if (std::isfinite(chi2)) {
                score = chi2;
            }
        }
    } catch (const std::exception& exception) {
        Py_DECREF(time_array);
        Py_DECREF(flux_array);
        Py_DECREF(ferr_array);
        Py_DECREF(seeds_array);
        Py_DECREF(output);
        PyErr_SetString(PyExc_ValueError, exception.what());
        return nullptr;
    }

    Py_DECREF(time_array);
    Py_DECREF(flux_array);
    Py_DECREF(ferr_array);
    Py_DECREF(seeds_array);
    return reinterpret_cast<PyObject*>(output);
}

PyMethodDef methods[] = {
    {
        "fspl_magnification",
        reinterpret_cast<PyCFunction>(finite_source_magnification),
        METH_VARARGS | METH_KEYWORDS,
        "Evaluate finite-source point-lens magnification in the compiled kernel."
    },
    {
        "score_fspl_seeds",
        reinterpret_cast<PyCFunction>(score_finite_source_seeds),
        METH_VARARGS | METH_KEYWORDS,
        "Score a batch of FSPL seeds with profiled source and blend fluxes."
    },
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_vbm_cpp",
    "Compiled finite-source magnification backend for jacscanomaly.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__vbm_cpp(void) {
    import_array();
    return PyModule_Create(&module);
}
