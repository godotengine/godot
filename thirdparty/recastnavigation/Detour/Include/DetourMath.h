/**
@defgroup detour Detour

Members in this module are wrappers around the standard math library
*/

#ifndef DETOURMATH_H
#define DETOURMATH_H

#include <math.h>
// This include is required because libstdc++ has problems with isfinite
// if cmath is included before math.h.
#include <cmath>

inline float dtMathFabsf(float x) { return fabsf(x); }
inline float dtMathSqrtf(float x) { return sqrtf(x); }
inline float dtMathFloorf(float x) { return floorf(x); }
inline float dtMathCeilf(float x) { return ceilf(x); }
inline float dtMathCosf(float x) { return cosf(x); }
inline float dtMathSinf(float x) { return sinf(x); }
inline float dtMathAtan2f(float y, float x) { return atan2f(y, x); }
inline bool dtMathIsfinite(float x) { return std::isfinite(x); }

#endif
