#pragma once

#include <cmath>
#include <stdexcept>

#include "ephemeris.hpp"

namespace jacscanomaly {

struct SkyProjection {
    Vec3 line_of_sight;
    Vec3 north;
    Vec3 east;

    SkyProjection(double ra_deg, double dec_deg) {
        constexpr double pi = 3.141592653589793238462643383279502884;
        const double ra = ra_deg * pi / 180.0;
        const double dec = dec_deg * pi / 180.0;
        const double ca = std::cos(ra), sa = std::sin(ra);
        const double cd = std::cos(dec), sd = std::sin(dec);
        line_of_sight = {cd * ca, cd * sa, sd};
        north = {-sd * ca, -sd * sa, cd};
        east = {-sa, ca, 0.0};
        const double norms[] = {
            dot(line_of_sight, line_of_sight),
            dot(north, north),
            dot(east, east),
        };
        if (std::abs(norms[0] - 1.0) > 1e-10 || std::abs(norms[1] - 1.0) > 1e-10 ||
            std::abs(norms[2] - 1.0) > 1e-10 || std::abs(dot(line_of_sight, north)) > 1e-10 ||
            std::abs(dot(line_of_sight, east)) > 1e-10 || std::abs(dot(north, east)) > 1e-10) {
            throw std::invalid_argument("sky basis is not orthonormal");
        }
    }

    double north_component(const Vec3& value) const { return dot(value, north); }
    double east_component(const Vec3& value) const { return dot(value, east); }
};

}  // namespace jacscanomaly
