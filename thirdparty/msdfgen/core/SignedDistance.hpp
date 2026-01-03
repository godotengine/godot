
#pragma once

#include <cmath>
#include <cfloat>
#include "base.h"

namespace msdfgen {

/// Represents a signed distance and alignment, which together can be compared to uniquely determine the closest edge segment.
class SignedDistance {

public:
    double distance;
    double dot;

    inline SignedDistance() : distance(-DBL_MAX), dot(0) { }
    inline SignedDistance(double dist, double d) : distance(dist), dot(d) { }

};

inline bool operator<(const SignedDistance a, const SignedDistance b) {
    return fabs(a.distance) < fabs(b.distance) || (fabs(a.distance) == fabs(b.distance) && a.dot < b.dot);
}

inline bool operator>(const SignedDistance a, const SignedDistance b) {
    return fabs(a.distance) > fabs(b.distance) || (fabs(a.distance) == fabs(b.distance) && a.dot > b.dot);
}

inline bool operator<=(const SignedDistance a, const SignedDistance b) {
    return fabs(a.distance) < fabs(b.distance) || (fabs(a.distance) == fabs(b.distance) && a.dot <= b.dot);
}

inline bool operator>=(const SignedDistance a, const SignedDistance b) {
    return fabs(a.distance) > fabs(b.distance) || (fabs(a.distance) == fabs(b.distance) && a.dot >= b.dot);
}

}
