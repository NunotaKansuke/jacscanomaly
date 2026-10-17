#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

PyArrayObject* as_double_array(PyObject* obj) {
    return reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)
    );
}

PyObject* run_dp(PyObject*, PyObject* args) {
    PyObject* time_obj = nullptr;
    PyObject* baseline_obj = nullptr;
    PyObject* anomaly_obj = nullptr;
    double season_gap = 0.0;
    double transition_penalty = 0.0;
    double span_penalty = 0.0;
    if (!PyArg_ParseTuple(
            args, "OOOddd", &time_obj, &baseline_obj, &anomaly_obj,
            &season_gap, &transition_penalty, &span_penalty)) {
        return nullptr;
    }

    PyArrayObject* time_array = as_double_array(time_obj);
    PyArrayObject* baseline_array = as_double_array(baseline_obj);
    PyArrayObject* anomaly_array = as_double_array(anomaly_obj);
    if (time_array == nullptr || baseline_array == nullptr || anomaly_array == nullptr) {
        Py_XDECREF(time_array);
        Py_XDECREF(baseline_array);
        Py_XDECREF(anomaly_array);
        return nullptr;
    }
    if (PyArray_NDIM(time_array) != 1 || PyArray_NDIM(baseline_array) != 1 ||
        PyArray_NDIM(anomaly_array) != 1 ||
        PyArray_SIZE(time_array) != PyArray_SIZE(baseline_array) ||
        PyArray_SIZE(time_array) != PyArray_SIZE(anomaly_array)) {
        PyErr_SetString(PyExc_ValueError, "time and emission arrays must be matching 1D arrays");
        Py_DECREF(time_array);
        Py_DECREF(baseline_array);
        Py_DECREF(anomaly_array);
        return nullptr;
    }

    const npy_intp n = PyArray_SIZE(time_array);
    if (n == 0) {
        npy_intp dims[1] = {0};
        PyObject* state = PyArray_SimpleNew(1, dims, NPY_INT8);
        Py_DECREF(time_array);
        Py_DECREF(baseline_array);
        Py_DECREF(anomaly_array);
        if (state == nullptr) return nullptr;
        PyObject* result = PyTuple_New(2);
        if (result == nullptr) {
            Py_DECREF(state);
            return nullptr;
        }
        PyTuple_SET_ITEM(result, 0, state);
        PyTuple_SET_ITEM(result, 1, PyFloat_FromDouble(0.0));
        return result;
    }

    const double* time = static_cast<const double*>(PyArray_DATA(time_array));
    const double* emission_b = static_cast<const double*>(PyArray_DATA(baseline_array));
    const double* emission_a = static_cast<const double*>(PyArray_DATA(anomaly_array));
    std::vector<double> dp_b(static_cast<size_t>(n));
    std::vector<double> dp_a(static_cast<size_t>(n));
    std::vector<uint8_t> parent_b(static_cast<size_t>(n), 0);
    std::vector<uint8_t> parent_a(static_cast<size_t>(n), 0);
    dp_b[0] = emission_b[0];
    dp_a[0] = emission_a[0];

    for (npy_intp i = 1; i < n; ++i) {
        const double gap = time[i] - time[i - 1];
        const bool connected = gap <= season_gap;
        const double transition = connected ? transition_penalty : 0.0;

        const double switch_to_b = dp_a[static_cast<size_t>(i - 1)] + transition;
        if (dp_b[static_cast<size_t>(i - 1)] <= switch_to_b) {
            dp_b[static_cast<size_t>(i)] = dp_b[static_cast<size_t>(i - 1)];
            parent_b[static_cast<size_t>(i)] = 0;
        } else {
            dp_b[static_cast<size_t>(i)] = switch_to_b;
            parent_b[static_cast<size_t>(i)] = 1;
        }
        dp_b[static_cast<size_t>(i)] += emission_b[i];

        const double switch_to_a = dp_b[static_cast<size_t>(i - 1)] + transition;
        const double span_cost = connected ? span_penalty * std::max(gap, 0.0) : 0.0;
        const double anomaly_increment = emission_a[i] + span_cost;
        if (dp_a[static_cast<size_t>(i - 1)] <= switch_to_a) {
            dp_a[static_cast<size_t>(i)] = dp_a[static_cast<size_t>(i - 1)] + anomaly_increment;
            parent_a[static_cast<size_t>(i)] = 1;
        } else {
            dp_a[static_cast<size_t>(i)] = switch_to_a + anomaly_increment;
            parent_a[static_cast<size_t>(i)] = 0;
        }
    }

    const double last_b = dp_b[static_cast<size_t>(n - 1)];
    const double last_a = dp_a[static_cast<size_t>(n - 1)];
    const double objective = std::min(last_b, last_a);
    npy_intp dims[1] = {n};
    PyArrayObject* state_array = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(1, dims, NPY_INT8)
    );
    if (state_array == nullptr) {
        Py_DECREF(time_array);
        Py_DECREF(baseline_array);
        Py_DECREF(anomaly_array);
        return nullptr;
    }
    int8_t* state = static_cast<int8_t*>(PyArray_DATA(state_array));
    if (!std::isfinite(objective)) {
        std::fill(state, state + n, static_cast<int8_t>(0));
    } else {
        uint8_t current = last_b <= last_a ? 0 : 1;
        state[n - 1] = static_cast<int8_t>(current);
        for (npy_intp i = n - 1; i > 0; --i) {
            current = current == 0
                ? parent_b[static_cast<size_t>(i)]
                : parent_a[static_cast<size_t>(i)];
            state[i - 1] = static_cast<int8_t>(current);
        }
    }

    Py_DECREF(time_array);
    Py_DECREF(baseline_array);
    Py_DECREF(anomaly_array);
    PyObject* result = PyTuple_New(2);
    if (result == nullptr) {
        Py_DECREF(state_array);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, reinterpret_cast<PyObject*>(state_array));
    PyTuple_SET_ITEM(result, 1, PyFloat_FromDouble(objective));
    return result;
}

PyMethodDef methods[] = {
    {"run_dp", run_dp, METH_VARARGS, "Run the two-state anomaly dynamic program."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_contamination_dp",
    "Native exact recurrence for contamination segmentation.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__contamination_dp() {
    import_array();
    return PyModule_Create(&module);
}
