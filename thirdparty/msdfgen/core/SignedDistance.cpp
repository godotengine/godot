
#include "SignedDistance.h"

#include <cmath>

namespace msdfgen {

const SignedDistance SignedDistance::INFINITE(-1e240, 1);

SignedDistance::SignedDistance() : distance(-1e240), dot(1) { }

SignedDistance::SignedDistance(double dist, double d) : distance(dist), dot(d) { }

bool operator<(SignedDistance a, SignedDistance b) {
    return fabs(a.distance) < fabs(b.distance) || (fabs(a.distance) == fabs(b.distance) && a.dot < b.dot);
}

bool operator>(SignedDistance a, SignedDistance b) {
    return fabs(a.distance) > fabs(b.distance) || (fabs(a.distance) == fabs(b.distance) && a.dot > b.dot);
}

bool operator<=(SignedDistance a, SignedDistance b) {
    return fabs(a.distance) < fabs(b.distance) || (fabs(a.distance) == fabs(b.distance) && a.dot <= b.dot);
}

bool operator>=(SignedDistance a, SignedDistance b) {
    return fabs(a.distance) > fabs(b.distance) || (fabs(a.distance) == fabs(b.distance) && a.dot >= b.dot);
}

}
