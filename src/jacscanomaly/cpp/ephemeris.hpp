#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace jacscanomaly {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator*(const Vec3& a, double b) {
    return {a.x * b, a.y * b, a.z * b};
}

inline Vec3 operator*(double b, const Vec3& a) { return a * b; }

inline double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

struct Ephemeris {
    std::vector<double> time;
    std::vector<Vec3> position;
    std::vector<Vec3> velocity;
    bool has_velocity = false;
    bool allow_extrapolation = false;

    void validate(const std::string& name) const {
        if (time.size() < 2 || position.size() != time.size()) {
            throw std::invalid_argument(name + " requires at least two position rows");
        }
        if (has_velocity && velocity.size() != time.size()) {
            throw std::invalid_argument(name + " velocity must match position length");
        }
        for (std::size_t i = 0; i < time.size(); ++i) {
            if (!std::isfinite(time[i])) throw std::invalid_argument(name + " time contains NaN/inf");
            if (i && !(time[i] > time[i - 1])) {
                throw std::invalid_argument(name + " time must be strictly increasing");
            }
            const Vec3& p = position[i];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                throw std::invalid_argument(name + " position contains NaN/inf");
            }
            if (has_velocity) {
                const Vec3& v = velocity[i];
                if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
                    throw std::invalid_argument(name + " velocity contains NaN/inf");
                }
            }
        }
    }

    std::size_t bracket(double t) const {
        if (t < time.front() || t > time.back()) {
            if (!allow_extrapolation) throw std::out_of_range("ephemeris query is outside its time range");
            return t < time.front() ? 0 : time.size() - 2;
        }
        auto it = std::upper_bound(time.begin(), time.end(), t);
        std::size_t i = static_cast<std::size_t>(std::distance(time.begin(), it));
        if (i == 0) return 0;
        if (i >= time.size()) return time.size() - 2;
        return i - 1;
    }

    Vec3 interpolate_position(double t) const {
        const std::size_t i = bracket(t);
        const double den = time[i + 1] - time[i];
        const double w = (t - time[i]) / den;
        return position[i] * (1.0 - w) + position[i + 1] * w;
    }

    Vec3 local_velocity(double t) const {
        if (has_velocity) {
            const std::size_t i = bracket(t);
            const double den = time[i + 1] - time[i];
            const double w = (t - time[i]) / den;
            return velocity[i] * (1.0 - w) + velocity[i + 1] * w;
        }
        // A local derivative is deliberate: a full-table gradient would make
        // endpoint noise and distant cadence gaps affect the reference speed.
        if (t <= time.front()) return (position[1] - position[0]) * (1.0 / (time[1] - time[0]));
        if (t >= time.back()) {
            const std::size_t n = time.size();
            return (position[n - 1] - position[n - 2]) * (1.0 / (time[n - 1] - time[n - 2]));
        }
        const std::size_t i = bracket(t);
        if (i == 0) return (position[1] - position[0]) * (1.0 / (time[1] - time[0]));
        if (i + 1 >= time.size() - 1) {
            return (position[time.size() - 1] - position[time.size() - 2]) *
                   (1.0 / (time[time.size() - 1] - time[time.size() - 2]));
        }
        const double span = time[i + 1] - time[i - 1];
        return (position[i + 1] - position[i - 1]) * (1.0 / span);
    }
};

}  // namespace jacscanomaly
