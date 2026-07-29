#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "cpp/ephemeris.hpp"
#include "cpp/parallax_trajectory.hpp"
#include "cpp/vbm_magnification.hpp"

using jacscanomaly::Ephemeris;
using jacscanomaly::ObserverConvention;
using jacscanomaly::ParallaxTrajectory;
using jacscanomaly::TrajectoryComponents;
using jacscanomaly::Vec3;
using jacscanomaly::VbmMagnification;

namespace {

PyArrayObject* as_double_array(PyObject* obj) {
    return reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
}

PyArrayObject* as_int_array(PyObject* obj) {
    return reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(obj, NPY_INT64, NPY_ARRAY_IN_ARRAY));
}

bool is_none(PyObject* obj) { return obj == nullptr || obj == Py_None; }

bool read_ephemeris(PyObject* obj, const char* name, Ephemeris& out) {
    if (is_none(obj)) return false;
    PyObject* seq = PySequence_Fast(obj, "ephemeris must be a (time, position, velocity) tuple");
    if (!seq) return false;
    const Py_ssize_t nitems = PySequence_Fast_GET_SIZE(seq);
    if (nitems < 2 || nitems > 3) {
        Py_DECREF(seq);
        PyErr_Format(PyExc_ValueError, "%s must contain time, position and optional velocity", name);
        return false;
    }
    PyObject** items = PySequence_Fast_ITEMS(seq);
    PyArrayObject* t = as_double_array(items[0]);
    PyArrayObject* p = as_double_array(items[1]);
    PyArrayObject* v = nitems == 3 && !is_none(items[2]) ? as_double_array(items[2]) : nullptr;
    if (!t || !p || (nitems == 3 && !is_none(items[2]) && !v)) {
        PyErr_Format(PyExc_ValueError, "invalid %s arrays", name);
        Py_XDECREF(t); Py_XDECREF(p); Py_XDECREF(v); Py_DECREF(seq);
        return false;
    }
    if (PyArray_NDIM(t) != 1 || PyArray_NDIM(p) != 2 || PyArray_DIM(p, 1) != 3 ||
        PyArray_DIM(t, 0) != PyArray_DIM(p, 0) ||
        (v && (PyArray_NDIM(v) != 2 || PyArray_DIM(v, 0) != PyArray_DIM(t, 0) || PyArray_DIM(v, 1) != 3))) {
        PyErr_Format(PyExc_ValueError, "%s arrays have incompatible shapes", name);
        Py_DECREF(t); Py_DECREF(p); Py_XDECREF(v); Py_DECREF(seq);
        return false;
    }
    const npy_intp n = PyArray_DIM(t, 0);
    out.time.resize(static_cast<size_t>(n));
    out.position.resize(static_cast<size_t>(n));
    if (v) out.velocity.resize(static_cast<size_t>(n));
    const double* td = static_cast<const double*>(PyArray_DATA(t));
    const double* pd = static_cast<const double*>(PyArray_DATA(p));
    const double* vd = v ? static_cast<const double*>(PyArray_DATA(v)) : nullptr;
    for (npy_intp i = 0; i < n; ++i) {
        out.time[static_cast<size_t>(i)] = td[i];
        out.position[static_cast<size_t>(i)] = {pd[3 * i], pd[3 * i + 1], pd[3 * i + 2]};
        if (vd) out.velocity[static_cast<size_t>(i)] = {vd[3 * i], vd[3 * i + 1], vd[3 * i + 2]};
    }
    out.has_velocity = vd != nullptr;
    Py_DECREF(t); Py_DECREF(p); Py_XDECREF(v); Py_DECREF(seq);
    try {
        out.validate(name);
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return false;
    }
    return true;
}

ObserverConvention parse_convention(const char* value) {
    const std::string mode(value ? value : "earth_geocentric_offset");
    if (mode == "earth_geocentric_offset" || mode == "vbm") return ObserverConvention::EarthGeocentricOffset;
    if (mode == "heliocentric_observer") return ObserverConvention::HeliocentricObserver;
    if (mode == "gulls") return ObserverConvention::Gulls;
    throw std::invalid_argument("observer_convention must be earth_geocentric_offset, heliocentric_observer, or gulls");
}

PyObject* make_array(const std::vector<double>& values) {
    npy_intp dims[1] = {static_cast<npy_intp>(values.size())};
    PyArrayObject* out = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dims, NPY_DOUBLE));
    if (!out) return nullptr;
    std::copy(values.begin(), values.end(), static_cast<double*>(PyArray_DATA(out)));
    return reinterpret_cast<PyObject*>(out);
}

PyObject* make_trajectory_dict(const std::vector<TrajectoryComponents>& rows) {
    PyObject* dict = PyDict_New();
    if (!dict) return nullptr;
    const char* names[] = {"tau", "beta", "u", "earth_n", "earth_e", "satellite_n", "satellite_e", "observer_n", "observer_e"};
    for (const char* name : names) {
        std::vector<double> values;
        values.reserve(rows.size());
        for (const auto& row : rows) {
            if (std::string(name) == "tau") values.push_back(row.tau);
            else if (std::string(name) == "beta") values.push_back(row.beta);
            else if (std::string(name) == "u") values.push_back(row.u);
            else if (std::string(name) == "earth_n") values.push_back(row.earth_n);
            else if (std::string(name) == "earth_e") values.push_back(row.earth_e);
            else if (std::string(name) == "satellite_n") values.push_back(row.satellite_n);
            else if (std::string(name) == "satellite_e") values.push_back(row.satellite_e);
            else if (std::string(name) == "observer_n") values.push_back(row.observer_n);
            else values.push_back(row.observer_e);
        }
        PyObject* array = make_array(values);
        if (!array || PyDict_SetItemString(dict, name, array) < 0) {
            Py_XDECREF(array); Py_DECREF(dict); return nullptr;
        }
        Py_DECREF(array);
    }
    return dict;
}

struct EvaluatorObject {
    PyObject_HEAD
    std::unique_ptr<ParallaxTrajectory> trajectory;
    std::unique_ptr<VbmMagnification> magnification;
    std::vector<double> time;
    std::vector<double> flux;
    std::vector<double> ferr;
    std::vector<long long> dataset;
    bool finite_source = false;
    int nparam = 0;
};

bool parse_raw(EvaluatorObject* self, PyObject* obj, std::vector<double>& raw) {
    PyArrayObject* array = as_double_array(obj);
    if (!array) return false;
    if (PyArray_NDIM(array) != 1 || PyArray_DIM(array, 0) != self->nparam) {
        PyErr_Format(PyExc_ValueError, "raw_params must be a one-dimensional array of length %d", self->nparam);
        Py_DECREF(array); return false;
    }
    const double* data = static_cast<const double*>(PyArray_DATA(array));
    raw.assign(data, data + self->nparam);
    Py_DECREF(array);
    for (double value : raw) if (!std::isfinite(value)) { PyErr_SetString(PyExc_ValueError, "raw_params must be finite"); return false; }
    return true;
}

void unpack_raw(EvaluatorObject* self, const std::vector<double>& raw, double& t0, double& tE, double& u0, double& rho, double& piEN, double& piEE) {
    t0 = raw[0];
    tE = std::exp(raw[1]);
    u0 = raw[2];
    int pi_index = 3;
    rho = 0.0;
    if (self->finite_source) { rho = std::exp(raw[3]); pi_index = 4; }
    piEN = raw[pi_index]; piEE = raw[pi_index + 1];
    if (!(tE > 0.0) || (self->finite_source && !(rho > 0.0))) throw std::invalid_argument("log_tE/log_rho produced a non-positive physical parameter");
}

bool active_mask(PyObject* obj, size_t n, std::vector<unsigned char>& mask) {
    mask.assign(n, 1);
    if (is_none(obj)) return true;
    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY));
    if (!array) return false;
    if (PyArray_NDIM(array) != 1 || static_cast<size_t>(PyArray_DIM(array, 0)) != n) {
        PyErr_SetString(PyExc_ValueError, "active_mask must match the data length");
        Py_DECREF(array); return false;
    }
    const npy_bool* data = static_cast<const npy_bool*>(PyArray_DATA(array));
    for (size_t i = 0; i < n; ++i) mask[i] = data[i] ? 1 : 0;
    Py_DECREF(array); return true;
}

bool evaluate_internal(EvaluatorObject* self, const std::vector<double>& raw, std::vector<double>& magnifications, std::vector<double>& residual, PyObject* mask_obj) {
    try {
        double t0, tE, u0, rho, piEN, piEE;
        unpack_raw(self, raw, t0, tE, u0, rho, piEN, piEE);
        const size_t n = self->time.size();
        std::vector<unsigned char> mask;
        if (!active_mask(mask_obj, n, mask)) return false;
        magnifications.resize(n);
        std::vector<double> weights(n);
        for (size_t i = 0; i < n; ++i) {
            const TrajectoryComponents row = self->trajectory->at(self->time[i], t0, tE, u0, piEN, piEE);
            magnifications[i] = (*self->magnification)(row.u, rho);
            weights[i] = mask[i] ? 1.0 / (self->ferr[i] * self->ferr[i]) : 0.0;
        }
        std::vector<double> sums(5, 0.0);
        for (size_t i = 0; i < n; ++i) {
            const double w = weights[i];
            const long long group = self->dataset[i];
            (void)group;
            sums[0] += w;
            sums[1] += w * magnifications[i];
            sums[2] += w * self->flux[i];
        }
        // Profile fs/fb independently for every dataset.  Dataset ids are
        // arbitrary non-negative integers; sparse ids are harmless.
        long long max_id = 0;
        for (long long id : self->dataset) max_id = std::max(max_id, id);
        if (max_id < 0 || max_id > 1000000) throw std::invalid_argument("dataset_id contains an invalid group id");
        std::vector<double> sw(static_cast<size_t>(max_id + 1), 0.0), sx(sw.size(), 0.0), sy(sw.size(), 0.0);
        for (size_t i = 0; i < n; ++i) {
            const size_t g = static_cast<size_t>(self->dataset[i]);
            const double w = weights[i]; sw[g] += w; sx[g] += w * magnifications[i]; sy[g] += w * self->flux[i];
        }
        std::vector<double> fs(sw.size(), 0.0), fb(sw.size(), 0.0);
        for (size_t g = 0; g < sw.size(); ++g) {
            if (!(sw[g] > 0.0)) continue;
            const double xm = sx[g] / sw[g], ym = sy[g] / sw[g];
            double wxx = 0.0, wxy = 0.0;
            for (size_t i = 0; i < n; ++i) if (static_cast<size_t>(self->dataset[i]) == g && mask[i]) {
                const double w = weights[i];
                wxx += w * (magnifications[i] - xm) * (magnifications[i] - xm);
                wxy += w * (magnifications[i] - xm) * (self->flux[i] - ym);
            }
            if (!(wxx > 0.0) || !std::isfinite(wxx)) throw std::invalid_argument("dataset flux profile is singular");
            fs[g] = wxy / wxx; fb[g] = ym - fs[g] * xm;
        }
        residual.resize(n);
        for (size_t i = 0; i < n; ++i) {
            const size_t g = static_cast<size_t>(self->dataset[i]);
            residual[i] = mask[i] ? (self->flux[i] - (fs[g] * magnifications[i] + fb[g])) / self->ferr[i] : 0.0;
            if (!std::isfinite(residual[i])) throw std::invalid_argument("profiled residual is not finite");
        }
        return true;
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
        return false;
    }
}

PyObject* evaluator_new(PyTypeObject* type, PyObject*, PyObject*) {
    return type->tp_alloc(type, 0);
}

int evaluator_init(EvaluatorObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *time_obj = nullptr, *flux_obj = nullptr, *ferr_obj = nullptr, *dataset_obj = nullptr;
    double ra = 0.0, dec = 0.0, tref = 0.0;
    const char* time_kind = "jd";
    const char* convention = "earth_geocentric_offset";
    PyObject *earth_obj = Py_None, *sat_obj = Py_None, *observer_obj = Py_None, *reference_obj = Py_None;
    int finite_source = 0;
    PyObject* table_obj = Py_None;
    double tol = 1e-4, reltol = 1e-4;
    int allow_extrapolation = 0;
    static const char* kwlist[] = {
        "time", "flux", "ferr", "dataset_id", "ra_deg", "dec_deg", "tref", "time_kind", "observer_convention",
        "earth_ephemeris", "satellite_or_observer_ephemeris", "reference_ephemeris", "finite_source", "espl_table_path",
        "vbm_tol", "vbm_reltol", "allow_extrapolation", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOdddss|OOOpOddp", const_cast<char**>(kwlist),
            &time_obj, &flux_obj, &ferr_obj, &dataset_obj, &ra, &dec, &tref, &time_kind, &convention,
            &earth_obj, &sat_obj, &reference_obj, &finite_source, &table_obj, &tol, &reltol, &allow_extrapolation)) return -1;
    PyArrayObject* time_arr = as_double_array(time_obj);
    PyArrayObject* flux_arr = as_double_array(flux_obj);
    PyArrayObject* ferr_arr = as_double_array(ferr_obj);
    PyArrayObject* dataset_arr = is_none(dataset_obj) ? nullptr : as_int_array(dataset_obj);
    if (!time_arr || !flux_arr || !ferr_arr || (!is_none(dataset_obj) && !dataset_arr)) goto fail;
    if (PyArray_NDIM(time_arr) != 1 || PyArray_NDIM(flux_arr) != 1 || PyArray_NDIM(ferr_arr) != 1 ||
        PyArray_DIM(time_arr, 0) != PyArray_DIM(flux_arr, 0) || PyArray_DIM(time_arr, 0) != PyArray_DIM(ferr_arr, 0) ||
        (dataset_arr && (PyArray_NDIM(dataset_arr) != 1 || PyArray_DIM(dataset_arr, 0) != PyArray_DIM(time_arr, 0)))) {
        PyErr_SetString(PyExc_ValueError, "time, flux, ferr, and dataset_id must be one-dimensional and have equal length"); goto fail;
    }
    {
        const npy_intp n = PyArray_DIM(time_arr, 0);
        const double* td = static_cast<const double*>(PyArray_DATA(time_arr));
        const double* fd = static_cast<const double*>(PyArray_DATA(flux_arr));
        const double* ed = static_cast<const double*>(PyArray_DATA(ferr_arr));
        self->time.assign(td, td + n); self->flux.assign(fd, fd + n); self->ferr.assign(ed, ed + n);
        self->dataset.assign(static_cast<size_t>(n), 0);
        if (dataset_arr) {
            const long long* gd = static_cast<const long long*>(PyArray_DATA(dataset_arr));
            for (npy_intp i = 0; i < n; ++i) self->dataset[static_cast<size_t>(i)] = gd[i];
        }
        for (npy_intp i = 0; i < n; ++i) if (!std::isfinite(self->time[i]) || !std::isfinite(self->flux[i]) || !(self->ferr[i] > 0.0) || !std::isfinite(self->ferr[i]) || self->dataset[static_cast<size_t>(i)] < 0) {
            PyErr_SetString(PyExc_ValueError, "time/flux must be finite, ferr must be positive, and dataset_id non-negative"); goto fail;
        }
    }
    try {
        Ephemeris earth, satellite, observer, reference;
        const bool has_earth = read_ephemeris(earth_obj, "earth_ephemeris", earth);
        const bool has_satellite = read_ephemeris(sat_obj, "satellite_or_observer_ephemeris", satellite);
        const bool has_observer = read_ephemeris(observer_obj, "satellite_or_observer_ephemeris", observer);
        const bool has_reference = read_ephemeris(reference_obj, "reference_ephemeris", reference);
        earth.allow_extrapolation = allow_extrapolation != 0;
        satellite.allow_extrapolation = allow_extrapolation != 0;
        observer.allow_extrapolation = allow_extrapolation != 0;
        reference.allow_extrapolation = allow_extrapolation != 0;
        const ObserverConvention mode = parse_convention(convention);
        if (mode == ObserverConvention::EarthGeocentricOffset && !has_earth) throw std::invalid_argument("earth_geocentric_offset requires earth_ephemeris");
        if (mode != ObserverConvention::EarthGeocentricOffset && ((!has_satellite && !has_observer) || !has_reference)) {
            throw std::invalid_argument("heliocentric_observer/gulls require observer and reference ephemerides");
        }
        // For the public API, satellite_or_observer_ephemeris is the observer
        // table in heliocentric/gulls modes and the geocentric offset in the
        // earth mode.
        if (mode != ObserverConvention::EarthGeocentricOffset) {
            observer = has_observer ? observer : satellite;
        }
        std::string table;
        if (!is_none(table_obj)) {
            PyObject* encoded = PyObject_Str(table_obj);
            if (!encoded) goto fail;
            const char* value = PyUnicode_AsUTF8(encoded);
            if (value) table = value;
            Py_DECREF(encoded);
        }
        self->trajectory = std::make_unique<ParallaxTrajectory>(
            std::move(earth), std::move(satellite), has_satellite, std::move(observer), has_observer || has_satellite,
            std::move(reference), has_reference, ra, dec, tref, mode, std::string(time_kind), mode == ObserverConvention::Gulls);
        self->finite_source = finite_source != 0;
        self->nparam = self->finite_source ? 6 : 5;
        self->magnification = std::make_unique<VbmMagnification>(self->finite_source, table, tol, reltol);
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what()); goto fail;
    }
    Py_DECREF(time_arr); Py_DECREF(flux_arr); Py_DECREF(ferr_arr); Py_XDECREF(dataset_arr);
    return 0;
fail:
    Py_XDECREF(time_arr); Py_XDECREF(flux_arr); Py_XDECREF(ferr_arr); Py_XDECREF(dataset_arr);
    return -1;
}

void evaluator_dealloc(EvaluatorObject* self) {
    self->magnification.reset(); self->trajectory.reset(); Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* evaluator_trajectory(EvaluatorObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* raw_obj = nullptr; int components = 0;
    static const char* kwlist[] = {"raw_params", "components", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", const_cast<char**>(kwlist), &raw_obj, &components)) return nullptr;
    std::vector<double> raw; if (!parse_raw(self, raw_obj, raw)) return nullptr;
    try {
        double t0, tE, u0, rho, piEN, piEE; unpack_raw(self, raw, t0, tE, u0, rho, piEN, piEE); (void)rho;
        std::vector<TrajectoryComponents> rows; rows.reserve(self->time.size());
        for (double t : self->time) rows.push_back(self->trajectory->at(t, t0, tE, u0, piEN, piEE));
        if (components) return make_trajectory_dict(rows);
        return Py_BuildValue("OOO", make_array([&] { std::vector<double> v; for (auto& r : rows) v.push_back(r.tau); return v; }()), make_array([&] { std::vector<double> v; for (auto& r : rows) v.push_back(r.beta); return v; }()), make_array([&] { std::vector<double> v; for (auto& r : rows) v.push_back(r.u); return v; }()));
    } catch (const std::exception& exc) { PyErr_SetString(PyExc_ValueError, exc.what()); return nullptr; }
}

PyObject* evaluator_evaluate(EvaluatorObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* raw_obj = nullptr; PyObject* mask_obj = Py_None;
    static const char* kwlist[] = {"raw_params", "active_mask", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", const_cast<char**>(kwlist), &raw_obj, &mask_obj)) return nullptr;
    std::vector<double> raw, mags, residual; if (!parse_raw(self, raw_obj, raw)) return nullptr;
    if (!evaluate_internal(self, raw, mags, residual, mask_obj)) return nullptr;
    // The model is deliberately returned as flux, matching the evaluator
    // contract.  Magnification is available through magnification().
    std::vector<unsigned char> mask; if (!active_mask(mask_obj, self->time.size(), mask)) return nullptr;
    std::vector<double> model(self->time.size());
    // Reprofile via residual relation is not enough for masked points; use the
    // same full-data profile through a second internal call without exposing it.
    // The public model() method is the canonical unmasked output.
    std::vector<double> full_mags, full_residual; if (!evaluate_internal(self, raw, full_mags, full_residual, Py_None)) return nullptr;
    // Solve profile from the full residual equation using a two-pass weighted fit.
    long long max_id = 0; for (long long id : self->dataset) max_id = std::max(max_id, id);
    std::vector<double> sw(static_cast<size_t>(max_id + 1), 0.0), sx(sw.size(), 0.0), sy(sw.size(), 0.0);
    for (size_t i = 0; i < self->time.size(); ++i) { const double w = 1.0/(self->ferr[i]*self->ferr[i]); const size_t g=static_cast<size_t>(self->dataset[i]); sw[g]+=w; sx[g]+=w*full_mags[i]; sy[g]+=w*self->flux[i]; }
    std::vector<double> fs(sw.size(), 0.0), fb(sw.size(), 0.0);
    for (size_t g=0; g<sw.size(); ++g) { const double xm=sx[g]/sw[g], ym=sy[g]/sw[g]; double wxx=0,wxy=0; for(size_t i=0;i<self->time.size();++i) if(static_cast<size_t>(self->dataset[i])==g){const double w=1.0/(self->ferr[i]*self->ferr[i]);wxx+=w*(full_mags[i]-xm)*(full_mags[i]-xm);wxy+=w*(full_mags[i]-xm)*(self->flux[i]-ym);} fs[g]=wxy/wxx;fb[g]=ym-fs[g]*xm; }
    for (size_t i=0;i<model.size();++i){const size_t g=static_cast<size_t>(self->dataset[i]);model[i]=fs[g]*full_mags[i]+fb[g];}
    return make_array(model);
}

PyObject* evaluator_magnification(EvaluatorObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* raw_obj = nullptr;
    static const char* kwlist[] = {"raw_params", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", const_cast<char**>(kwlist), &raw_obj)) return nullptr;
    std::vector<double> raw; if (!parse_raw(self, raw_obj, raw)) return nullptr;
    try { double t0,tE,u0,rho,piEN,piEE; unpack_raw(self,raw,t0,tE,u0,rho,piEN,piEE); std::vector<double> values; values.reserve(self->time.size()); for(double t:self->time) values.push_back((*self->magnification)(self->trajectory->at(t,t0,tE,u0,piEN,piEE).u,rho)); return make_array(values); }
    catch(const std::exception& exc){PyErr_SetString(PyExc_ValueError,exc.what());return nullptr;}
}

PyObject* evaluator_residual(EvaluatorObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* raw_obj=nullptr; PyObject* mask_obj=Py_None; static const char* kwlist[]={"raw_params","active_mask",nullptr};
    if(!PyArg_ParseTupleAndKeywords(args,kwargs,"O|O",const_cast<char**>(kwlist),&raw_obj,&mask_obj))return nullptr;
    std::vector<double> raw,mags,residual; if(!parse_raw(self,raw_obj,raw))return nullptr; if(!evaluate_internal(self,raw,mags,residual,mask_obj))return nullptr; return make_array(residual);
}

PyObject* evaluator_jacobian(EvaluatorObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* raw_obj=nullptr; PyObject* mask_obj=Py_None; double fd_step=1e-5; static const char* kwlist[]={"raw_params","active_mask","fd_step",nullptr};
    if(!PyArg_ParseTupleAndKeywords(args,kwargs,"O|Od",const_cast<char**>(kwlist),&raw_obj,&mask_obj,&fd_step))return nullptr;
    std::vector<double> raw,base; if(!parse_raw(self,raw_obj,raw))return nullptr; if(!evaluate_internal(self,raw,base,base,mask_obj))return nullptr;
    const size_t n=self->time.size(); npy_intp dims[2]={static_cast<npy_intp>(n),static_cast<npy_intp>(self->nparam)}; PyArrayObject* out=reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(2,dims,NPY_DOUBLE,0)); if(!out)return nullptr; double* jd=static_cast<double*>(PyArray_DATA(out));
    for(int k=0;k<self->nparam;++k){ double step=std::max(std::abs(fd_step),1e-8); if(k==0)step=std::max(step,1e-6); if(k==1|| (self->finite_source&&k==3))step=std::max(step,1e-5); std::vector<double> plus=raw,minus=raw,rp,rm; plus[k]+=step;minus[k]-=step; bool okp=evaluate_internal(self,plus,rp,rp,mask_obj); bool okm=evaluate_internal(self,minus,rm,rm,mask_obj); if(okp&&okm){for(size_t i=0;i<n;++i)jd[i*self->nparam+k]=(rp[i]-rm[i])/(2*step);} else if(okp){std::vector<double> r0;evaluate_internal(self,raw,r0,r0,mask_obj);for(size_t i=0;i<n;++i)jd[i*self->nparam+k]=(rp[i]-r0[i])/step;} else if(okm){std::vector<double> r0;evaluate_internal(self,raw,r0,r0,mask_obj);for(size_t i=0;i<n;++i)jd[i*self->nparam+k]=(r0[i]-rm[i])/step;} }
    return reinterpret_cast<PyObject*>(out);
}

PyObject* evaluator_residual_and_jacobian(EvaluatorObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* residual = evaluator_residual(self,args,kwargs); if(!residual)return nullptr; PyObject* jacobian=evaluator_jacobian(self,args,kwargs); if(!jacobian){Py_DECREF(residual);return nullptr;} PyObject* out=PyTuple_Pack(2,residual,jacobian);Py_DECREF(residual);Py_DECREF(jacobian);return out;
}

PyMethodDef evaluator_methods[] = {
    {"trajectory", reinterpret_cast<PyCFunction>(evaluator_trajectory), METH_VARARGS|METH_KEYWORDS, "Evaluate tau/beta/u and optional trajectory components."},
    {"magnification", reinterpret_cast<PyCFunction>(evaluator_magnification), METH_VARARGS|METH_KEYWORDS, "Evaluate VBM magnification."},
    {"evaluate", reinterpret_cast<PyCFunction>(evaluator_evaluate), METH_VARARGS|METH_KEYWORDS, "Evaluate profiled model flux."},
    {"residual", reinterpret_cast<PyCFunction>(evaluator_residual), METH_VARARGS|METH_KEYWORDS, "Evaluate profiled weighted residual."},
    {"jacobian", reinterpret_cast<PyCFunction>(evaluator_jacobian), METH_VARARGS|METH_KEYWORDS, "Evaluate adaptive central finite-difference Jacobian."},
    {"residual_and_jacobian", reinterpret_cast<PyCFunction>(evaluator_residual_and_jacobian), METH_VARARGS|METH_KEYWORDS, "Evaluate residual and Jacobian."},
    {nullptr,nullptr,0,nullptr}
};

PyTypeObject EvaluatorType = {PyVarObject_HEAD_INIT(nullptr, 0)};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "_parallax_cpp", "Native parallax backend", -1, nullptr, nullptr, nullptr, nullptr, nullptr};

}  // namespace

PyMODINIT_FUNC PyInit__parallax_cpp(void) {
    import_array();
    EvaluatorType.tp_name = "jacscanomaly._parallax_cpp.ParallaxEvaluator";
    EvaluatorType.tp_basicsize = sizeof(EvaluatorObject);
    EvaluatorType.tp_dealloc = reinterpret_cast<destructor>(evaluator_dealloc);
    EvaluatorType.tp_flags = Py_TPFLAGS_DEFAULT;
    EvaluatorType.tp_doc = "Native C++ trajectory/VBM evaluator";
    EvaluatorType.tp_methods = evaluator_methods;
    EvaluatorType.tp_new = evaluator_new;
    EvaluatorType.tp_init = reinterpret_cast<initproc>(evaluator_init);
    if (PyType_Ready(&EvaluatorType) < 0) return nullptr;
    PyObject* mod = PyModule_Create(&module); if (!mod) return nullptr;
    Py_INCREF(&EvaluatorType); if (PyModule_AddObject(mod, "ParallaxEvaluator", reinterpret_cast<PyObject*>(&EvaluatorType)) < 0) {Py_DECREF(&EvaluatorType);Py_DECREF(mod);return nullptr;}
    return mod;
}
