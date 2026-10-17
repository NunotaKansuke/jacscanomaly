#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "VBMicrolensingLibrary.h"

namespace jacscanomaly {

class VbmMagnification {
public:
    VbmMagnification(bool finite_source, const std::string& espl_table, double tol, double reltol)
        : finite_source_(finite_source), vbm_(std::make_unique<VBMicrolensing>()) {
        vbm_->Tol = tol;
        vbm_->RelTol = reltol;
        if (finite_source_ && !espl_table.empty()) {
            vbm_->LoadESPLTable(espl_table.c_str());
            load_table_copy(espl_table);
        }
    }

    double operator()(double u, double rho) {
        if (!std::isfinite(u) || u < 0.0) throw std::runtime_error("invalid lens separation");
        const double value = finite_source_ ? vbm_->ESPLMag2(u, std::max(rho, 1e-12)) : vbm_->PSPLMag(u);
        if (!std::isfinite(value) || value <= 0.0) throw std::runtime_error("VBMicrolensing returned an invalid magnification");
        return value;
    }

    // Magnification and its partial derivatives with respect to the lens-source
    // separation u and the source radius rho.  The value is VBM's own, so it is
    // identical to operator().  Derivatives are the analytic derivatives of the
    // branch VBM evaluates: the point-lens closed form, the ESPLMag2
    // quadrupole expansion, or the bilinear ESPL table interpolation (whose
    // derivative is piecewise, exactly like the interpolant).
    void value_and_gradient(double u, double rho, double& magnification, double& d_du, double& d_drho) {
        magnification = (*this)(u, rho);
        gradient(u, rho, d_du, d_drho);
    }

    // Derivatives only, for callers that already hold the magnification.
    void gradient(double u, double rho, double& d_du, double& d_drho) {
        d_drho = 0.0;
        const double u2 = u * u;
        const double u2_4 = u2 + 4.0;
        const double s_u2_4 = std::sqrt(u2_4);
        const double point_du = -8.0 / (u2 * u2_4 * s_u2_4);  // d/du of (u^2+2)/(u sqrt(u^2+4))
        if (!finite_source_) {
            d_du = point_du;
            return;
        }
        const double r = std::max(rho, 1e-12);
        const double quad = 4.0 * (u2 + 1.0) * r * r / (u2 * u * u2_4 * u2_4 * s_u2_4);
        if (quad * 10.0 < vbm_->Tol) {
            d_du = point_du + quad * (2.0 * u / (u2 + 1.0) - 3.0 / u - 5.0 * u / u2_4);
            d_drho = 2.0 * quad / r;
            return;
        }
        constexpr double fr_scale = 10.857362047581296;
        double fr = -fr_scale * std::log(0.01 * r);
        double dfr_drho = -fr_scale / r;
        const double fr_max = __rsize_ESPL - 1.000001;
        if (fr > fr_max) { fr = fr_max; dfr_drho = 0.0; }
        if (fr < 0.0) { fr = 0.0; dfr_drho = 0.0; }
        const int ir = std::min(static_cast<int>(std::floor(fr)), __rsize_ESPL - 2);
        fr -= ir;
        const double z = u / r;
        const bool inside = z < 1.0;
        double zz, dzz_du, dzz_drho;
        if (inside) {
            zz = z * (__zsize_ESPL - 1);
            dzz_du = (__zsize_ESPL - 1) / r;
            dzz_drho = -zz / r;
        } else {
            zz = 0.99999999999999 / z * (__zsize_ESPL - 1);
            dzz_du = -zz / u;
            dzz_drho = zz / r;
        }
        const int iz = std::min(static_cast<int>(std::floor(zz)), __zsize_ESPL - 2);
        const double zf = zz - iz;
        const double cr = 1.0 - fr, cz = 1.0 - zf;
        if (table_in_.empty()) throw std::runtime_error("ESPL table was not loaded");
        const std::vector<double>& table = inside ? table_in_ : table_out_;
        auto at = [&](int r, int z_index) { return table[static_cast<size_t>(r) * __zsize_ESPL + z_index]; };
        const double t00 = at(ir, iz), t10 = at(ir + 1, iz), t01 = at(ir, iz + 1), t11 = at(ir + 1, iz + 1);
        const double corr = t00 * cr * cz + t10 * fr * cz + t01 * cr * zf + t11 * fr * zf;
        const double dcorr_dfr = (t10 - t00) * cz + (t11 - t01) * zf;
        const double dcorr_dzf = (t01 - t00) * cr + (t11 - t10) * fr;
        double pref, dpref_du = 0.0, dpref_drho = 0.0;
        if (inside) {
            pref = std::sqrt(1.0 + 4.0 / (r * r));
            dpref_drho = -4.0 / (r * r * r * pref);
        } else {
            pref = (u2 + 2.0) / (u * s_u2_4);
            dpref_du = point_du;
        }
        d_du = dpref_du * corr + pref * dcorr_dzf * dzz_du;
        d_drho = dpref_drho * corr + pref * (dcorr_dfr * dfr_drho + dcorr_dzf * dzz_drho);
    }

private:
    // VBM keeps its ESPL tables private.  The file holds the "in" and "out"
    // magnification blocks first (151 x 101 doubles each); a private copy lets
    // value_and_gradient differentiate the same bilinear interpolant.
    void load_table_copy(const std::string& path) {
        constexpr size_t block = static_cast<size_t>(__rsize_ESPL) * __zsize_ESPL;
        table_in_.resize(block);
        table_out_.resize(block);
        std::FILE* file = std::fopen(path.c_str(), "rb");
        if (!file) throw std::runtime_error("cannot open ESPL table: " + path);
        const bool ok = std::fread(table_in_.data(), sizeof(double), block, file) == block
            && std::fread(table_out_.data(), sizeof(double), block, file) == block;
        std::fclose(file);
        if (!ok) throw std::runtime_error("ESPL table is truncated: " + path);
    }

    std::vector<double> table_in_, table_out_;
    bool finite_source_ = false;
    std::unique_ptr<VBMicrolensing> vbm_;
};

}  // namespace jacscanomaly
