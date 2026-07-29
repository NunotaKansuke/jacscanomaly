#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "VBMicrolensingLibrary.h"

namespace jacscanomaly {

class VbmMagnification {
public:
    VbmMagnification(bool finite_source, const std::string& espl_table, double tol, double reltol)
        : finite_source_(finite_source), vbm_(std::make_unique<VBMicrolensing>()) {
        vbm_->Tol = tol;
        vbm_->RelTol = reltol;
        if (finite_source_ && !espl_table.empty()) vbm_->LoadESPLTable(espl_table.c_str());
    }

    double operator()(double u, double rho) {
        if (!std::isfinite(u) || u < 0.0) throw std::runtime_error("invalid lens separation");
        const double value = finite_source_ ? vbm_->ESPLMag2(u, std::max(rho, 1e-12)) : vbm_->PSPLMag(u);
        if (!std::isfinite(value) || value <= 0.0) throw std::runtime_error("VBMicrolensing returned an invalid magnification");
        return value;
    }

private:
    bool finite_source_ = false;
    std::unique_ptr<VBMicrolensing> vbm_;
};

}  // namespace jacscanomaly
