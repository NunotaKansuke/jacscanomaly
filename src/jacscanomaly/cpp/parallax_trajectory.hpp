#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

#include "ephemeris.hpp"
#include "sky_projection.hpp"

namespace jacscanomaly {

constexpr double AU_C_DAY = 0.005775518331436995;

enum class ObserverConvention {
    EarthGeocentricOffset,
    HeliocentricObserver,
    Gulls,
};

struct TrajectoryComponents {
    double tau = 0.0;
    double beta = 0.0;
    double u = 0.0;
    double earth_n = 0.0;
    double earth_e = 0.0;
    double satellite_n = 0.0;
    double satellite_e = 0.0;
    double observer_n = 0.0;
    double observer_e = 0.0;
};

inline Vec3 light_time_position(const Ephemeris& ephemeris, double t, const SkyProjection& sky) {
    return ephemeris.interpolate_position(t);
}

class ParallaxTrajectory {
public:
    ParallaxTrajectory(
        Ephemeris earth,
        Ephemeris satellite,
        bool has_satellite,
        Ephemeris observer,
        bool has_observer,
        Ephemeris reference,
        bool has_reference,
        double ra_deg,
        double dec_deg,
        double tref,
        ObserverConvention convention,
        std::string time_kind,
        bool strict_gulls = false
    )
        : earth_(std::move(earth)),
          satellite_(std::move(satellite)),
          has_satellite_(has_satellite),
          observer_(std::move(observer)),
          has_observer_(has_observer),
          reference_(std::move(reference)),
          has_reference_(has_reference),
          sky_(ra_deg, dec_deg),
          tref_(tref),
          convention_(convention),
          time_kind_(std::move(time_kind)),
          strict_gulls_(strict_gulls) {
        if (time_kind_ != "jd" && time_kind_ != "hjd") throw std::invalid_argument("time_kind must be 'jd' or 'hjd'");
        if (convention_ == ObserverConvention::EarthGeocentricOffset) {
            earth_.validate("earth_ephemeris");
            if (has_satellite_) satellite_.validate("satellite_ephemeris");
        } else {
            if (!has_observer_ || !has_reference_) throw std::invalid_argument("heliocentric observer conventions require observer and reference ephemerides");
            observer_.validate("observer_ephemeris");
            reference_.validate("reference_ephemeris");
        }
        if (!std::isfinite(tref_)) throw std::invalid_argument("tref must be finite");
        tref_eval_ = ephemeris_time(
            tref_,
            convention_ == ObserverConvention::EarthGeocentricOffset
                ? earth_
                : reference_
        );
        if (convention_ == ObserverConvention::EarthGeocentricOffset) {
            earth_ref_ = earth_.interpolate_position(tref_eval_);
            earth_vref_ = earth_.local_velocity(tref_eval_);
        } else {
            reference_ref_ = reference_.interpolate_position(tref_eval_);
            reference_vref_ = reference_.local_velocity(tref_eval_);
        }
    }

    double ephemeris_time(double t, const Ephemeris& ephemeris) const {
        if (strict_gulls_ || time_kind_ == "jd") return t;
        double eval = t;
        for (int i = 0; i < 5; ++i) {
            const Vec3 r = ephemeris.interpolate_position(eval);
            eval = t - dot(r, sky_.line_of_sight) * AU_C_DAY;
        }
        return eval;
    }

    double observer_light_time(double t, const Ephemeris& ephemeris) const {
        if (strict_gulls_ || time_kind_ != "jd") return 0.0;
        return dot(ephemeris.interpolate_position(t), sky_.line_of_sight)
            * AU_C_DAY;
    }

    TrajectoryComponents at(double t, double t0, double tE, double u0, double piEN, double piEE) const {
        if (!(tE > 0.0) || !std::isfinite(tE)) throw std::invalid_argument("tE must be positive");
        Vec3 displacement{};
        Vec3 earth_displacement{};
        Vec3 satellite_displacement{};
        const bool use_light_time = !strict_gulls_ && convention_ == ObserverConvention::EarthGeocentricOffset;
        if (convention_ == ObserverConvention::EarthGeocentricOffset) {
            const double eval_t = ephemeris_time(t, earth_);
            const double dt = t - tref_;
            const Vec3 r = earth_.interpolate_position(eval_t);
            earth_displacement = r - earth_ref_ - earth_vref_ * dt;
            displacement = earth_displacement;
            if (has_satellite_) {
                satellite_displacement = satellite_.interpolate_position(t);
                displacement = displacement + satellite_displacement;
            }
        } else {
            const double eval_t = ephemeris_time(t, observer_);
            const double dt = t - tref_;
            const Vec3 r_obs = observer_.interpolate_position(eval_t);
            displacement = r_obs - reference_ref_ - reference_vref_ * dt;
            if (convention_ == ObserverConvention::Gulls) {
                // Strict GULLS uses complete observer and reference heliocentric
                // orbits; no VBM light-time term is added here.
                (void)use_light_time;
            }
        }
        const double d_n = sky_.north_component(displacement);
        const double d_e = sky_.east_component(displacement);
        double source_time = t;
        double source_peak_time = t0;
        if (!strict_gulls_ && time_kind_ == "jd") {
            const Ephemeris& source_ephemeris = convention_ == ObserverConvention::EarthGeocentricOffset ? earth_ : observer_;
            source_time = t + observer_light_time(t, source_ephemeris);
            source_peak_time = t0 + observer_light_time(t0, source_ephemeris);
        }
        const double tau_rect = (source_time - source_peak_time) / tE;
        const double d_tau = -(piEN * d_n + piEE * d_e);
        const double d_beta = piEE * d_n - piEN * d_e;
        TrajectoryComponents out;
        out.tau = tau_rect + d_tau;
        out.beta = u0 + d_beta;
        out.u = std::hypot(out.tau, out.beta);
        out.earth_n = sky_.north_component(earth_displacement);
        out.earth_e = sky_.east_component(earth_displacement);
        out.satellite_n = sky_.north_component(satellite_displacement);
        out.satellite_e = sky_.east_component(satellite_displacement);
        out.observer_n = d_n;
        out.observer_e = d_e;
        return out;
    }

private:
    Ephemeris earth_;
    Ephemeris satellite_;
    bool has_satellite_ = false;
    Ephemeris observer_;
    bool has_observer_ = false;
    Ephemeris reference_;
    bool has_reference_ = false;
    SkyProjection sky_;
    double tref_ = 0.0;
    double tref_eval_ = 0.0;
    ObserverConvention convention_;
    std::string time_kind_;
    bool strict_gulls_ = false;
    Vec3 earth_ref_{};
    Vec3 earth_vref_{};
    Vec3 reference_ref_{};
    Vec3 reference_vref_{};
};

}  // namespace jacscanomaly
