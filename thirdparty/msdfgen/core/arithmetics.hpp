
#pragma once

#include <cmath>
#include "base.h"

namespace msdfgen {

/// Returns the smaller of the arguments.
template <typename T>
inline T min(T a, T b) {
    return b < a ? b : a;
}

/// Returns the larger of the arguments.
template <typename T>
inline T max(T a, T b) {
    return a < b ? b : a;
}

/// Returns the middle out of three values
template <typename T>
inline T median(T a, T b, T c) {
    return max(min(a, b), min(max(a, b), c));
}

/// Returns the weighted average of a and b.
template <typename T, typename S>
inline T mix(T a, T b, S weight) {
    return T((S(1)-weight)*a+weight*b);
}

/// Clamps the number to the interval from 0 to 1.
template <typename T>
inline T clamp(T n) {
    return n >= T(0) && n <= T(1) ? n : T(n > T(0));
}

/// Clamps the number to the interval from 0 to b.
template <typename T>
inline T clamp(T n, T b) {
    return n >= T(0) && n <= b ? n : T(n > T(0))*b;
}

/// Clamps the number to the interval from a to b.
template <typename T>
inline T clamp(T n, T a, T b) {
    return n >= a && n <= b ? n : n < a ? a : b;
}

/// Returns 1 for positive values, -1 for negative values, and 0 for zero.
template <typename T>
inline int sign(T n) {
    return (T(0) < n)-(n < T(0));
}

/// Returns 1 for non-negative values and -1 for negative values.
template <typename T>
inline int nonZeroSign(T n) {
    return 2*(n > T(0))-1;
}

}
