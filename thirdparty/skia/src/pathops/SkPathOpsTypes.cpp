/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#include "include/private/SkFloatBits.h"
#include "src/core/SkArenaAlloc.h"
#include "src/pathops/SkOpCoincidence.h"
#include "src/pathops/SkPathOpsTypes.h"

static bool arguments_denormalized(float a, float b, int epsilon) {
    float denormalizedCheck = FLT_EPSILON * epsilon / 2;
    return fabsf(a) <= denormalizedCheck && fabsf(b) <= denormalizedCheck;
}

// from http://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
// FIXME: move to SkFloatBits.h
static bool equal_ulps(float a, float b, int epsilon, int depsilon) {
    if (arguments_denormalized(a, b, depsilon)) {
        return true;
    }
    int aBits = SkFloatAs2sCompliment(a);
    int bBits = SkFloatAs2sCompliment(b);
    // Find the difference in ULPs.
    return aBits < bBits + epsilon && bBits < aBits + epsilon;
}

static bool equal_ulps_no_normal_check(float a, float b, int epsilon, int depsilon) {
    int aBits = SkFloatAs2sCompliment(a);
    int bBits = SkFloatAs2sCompliment(b);
    // Find the difference in ULPs.
    return aBits < bBits + epsilon && bBits < aBits + epsilon;
}

static bool equal_ulps_pin(float a, float b, int epsilon, int depsilon) {
    if (!SkScalarIsFinite(a) || !SkScalarIsFinite(b)) {
        return false;
    }
    if (arguments_denormalized(a, b, depsilon)) {
        return true;
    }
    int aBits = SkFloatAs2sCompliment(a);
    int bBits = SkFloatAs2sCompliment(b);
    // Find the difference in ULPs.
    return aBits < bBits + epsilon && bBits < aBits + epsilon;
}

static bool d_equal_ulps(float a, float b, int epsilon) {
    int aBits = SkFloatAs2sCompliment(a);
    int bBits = SkFloatAs2sCompliment(b);
    // Find the difference in ULPs.
    return aBits < bBits + epsilon && bBits < aBits + epsilon;
}

static bool not_equal_ulps(float a, float b, int epsilon) {
    if (arguments_denormalized(a, b, epsilon)) {
        return false;
    }
    int aBits = SkFloatAs2sCompliment(a);
    int bBits = SkFloatAs2sCompliment(b);
    // Find the difference in ULPs.
    return aBits >= bBits + epsilon || bBits >= aBits + epsilon;
}

static bool not_equal_ulps_pin(float a, float b, int epsilon) {
    if (!SkScalarIsFinite(a) || !SkScalarIsFinite(b)) {
        return false;
    }
    if (arguments_denormalized(a, b, epsilon)) {
        return false;
    }
    int aBits = SkFloatAs2sCompliment(a);
    int bBits = SkFloatAs2sCompliment(b);
    // Find the difference in ULPs.
    return aBits >= bBits + epsilon || bBits >= aBits + epsilon;
}

static bool d_not_equal_ulps(float a, float b, int epsilon) {
    int aBits = SkFloatAs2sCompliment(a);
    int bBits = SkFloatAs2sCompliment(b);
    // Find the difference in ULPs.
    return aBits >= bBits + epsilon || bBits >= aBits + epsilon;
}

static bool less_ulps(float a, float b, int epsilon) {
    if (arguments_denormalized(a, b, epsilon)) {
        return a <= b - FLT_EPSILON * epsilon;
    }
    int aBits = SkFloatAs2sCompliment(a);
    int bBits = SkFloatAs2sCompliment(b);
    // Find the difference in ULPs.
    return aBits <= bBits - epsilon;
}

static bool less_or_equal_ulps(float a, float b, int epsilon) {
    if (arguments_denormalized(a, b, epsilon)) {
        return a < b + FLT_EPSILON * epsilon;
    }
    int aBits = SkFloatAs2sCompliment(a);
    int bBits = SkFloatAs2sCompliment(b);
    // Find the difference in ULPs.
    return aBits < bBits + epsilon;
}

// equality using the same error term as between
bool AlmostBequalUlps(float a, float b) {
    const int UlpsEpsilon = 2;
    return equal_ulps(a, b, UlpsEpsilon, UlpsEpsilon);
}

bool AlmostPequalUlps(float a, float b) {
    const int UlpsEpsilon = 8;
    return equal_ulps(a, b, UlpsEpsilon, UlpsEpsilon);
}

bool AlmostDequalUlps(float a, float b) {
    const int UlpsEpsilon = 16;
    return d_equal_ulps(a, b, UlpsEpsilon);
}

bool AlmostDequalUlps(double a, double b) {
    if (fabs(a) < SK_ScalarMax && fabs(b) < SK_ScalarMax) {
        return AlmostDequalUlps(SkDoubleToScalar(a), SkDoubleToScalar(b));
    }
    return fabs(a - b) / std::max(fabs(a), fabs(b)) < FLT_EPSILON * 16;
}

bool AlmostEqualUlps(float a, float b) {
    const int UlpsEpsilon = 16;
    return equal_ulps(a, b, UlpsEpsilon, UlpsEpsilon);
}

bool AlmostEqualUlpsNoNormalCheck(float a, float b) {
    const int UlpsEpsilon = 16;
    return equal_ulps_no_normal_check(a, b, UlpsEpsilon, UlpsEpsilon);
}

bool AlmostEqualUlps_Pin(float a, float b) {
    const int UlpsEpsilon = 16;
    return equal_ulps_pin(a, b, UlpsEpsilon, UlpsEpsilon);
}

bool NotAlmostEqualUlps(float a, float b) {
    const int UlpsEpsilon = 16;
    return not_equal_ulps(a, b, UlpsEpsilon);
}

bool NotAlmostEqualUlps_Pin(float a, float b) {
    const int UlpsEpsilon = 16;
    return not_equal_ulps_pin(a, b, UlpsEpsilon);
}

bool NotAlmostDequalUlps(float a, float b) {
    const int UlpsEpsilon = 16;
    return d_not_equal_ulps(a, b, UlpsEpsilon);
}

bool RoughlyEqualUlps(float a, float b) {
    const int UlpsEpsilon = 256;
    const int DUlpsEpsilon = 1024;
    return equal_ulps(a, b, UlpsEpsilon, DUlpsEpsilon);
}

bool AlmostBetweenUlps(float a, float b, float c) {
    const int UlpsEpsilon = 2;
    return a <= c ? less_or_equal_ulps(a, b, UlpsEpsilon) && less_or_equal_ulps(b, c, UlpsEpsilon)
        : less_or_equal_ulps(b, a, UlpsEpsilon) && less_or_equal_ulps(c, b, UlpsEpsilon);
}

bool AlmostLessUlps(float a, float b) {
    const int UlpsEpsilon = 16;
    return less_ulps(a, b, UlpsEpsilon);
}

bool AlmostLessOrEqualUlps(float a, float b) {
    const int UlpsEpsilon = 16;
    return less_or_equal_ulps(a, b, UlpsEpsilon);
}

int UlpsDistance(float a, float b) {
    SkFloatIntUnion floatIntA, floatIntB;
    floatIntA.fFloat = a;
    floatIntB.fFloat = b;
    // Different signs means they do not match.
    if ((floatIntA.fSignBitInt < 0) != (floatIntB.fSignBitInt < 0)) {
        // Check for equality to make sure +0 == -0
        return a == b ? 0 : SK_MaxS32;
    }
    // Find the difference in ULPs.
    return SkTAbs(floatIntA.fSignBitInt - floatIntB.fSignBitInt);
}

// cube root approximation using bit hack for 64-bit float
// adapted from Kahan's cbrt
static double cbrt_5d(double d) {
    const unsigned int B1 = 715094163;
    double t = 0.0;
    unsigned int* pt = (unsigned int*) &t;
    unsigned int* px = (unsigned int*) &d;
    pt[1] = px[1] / 3 + B1;
    return t;
}

// iterative cube root approximation using Halley's method (double)
static double cbrta_halleyd(const double a, const double R) {
    const double a3 = a * a * a;
    const double b = a * (a3 + R + R) / (a3 + a3 + R);
    return b;
}

// cube root approximation using 3 iterations of Halley's method (double)
static double halley_cbrt3d(double d) {
    double a = cbrt_5d(d);
    a = cbrta_halleyd(a, d);
    a = cbrta_halleyd(a, d);
    return cbrta_halleyd(a, d);
}

double SkDCubeRoot(double x) {
    if (approximately_zero_cubed(x)) {
        return 0;
    }
    double result = halley_cbrt3d(fabs(x));
    if (x < 0) {
        result = -result;
    }
    return result;
}

SkOpGlobalState::SkOpGlobalState(SkOpContourHead* head,
                                 SkArenaAlloc* allocator
                                 SkDEBUGPARAMS(bool debugSkipAssert)
                                 SkDEBUGPARAMS(const char* testName))
    : fAllocator(allocator)
    , fCoincidence(nullptr)
    , fContourHead(head)
    , fNested(0)
    , fWindingFailed(false)
    , fPhase(SkOpPhase::kIntersecting)
    SkDEBUGPARAMS(fDebugTestName(testName))
    SkDEBUGPARAMS(fAngleID(0))
    SkDEBUGPARAMS(fCoinID(0))
    SkDEBUGPARAMS(fContourID(0))
    SkDEBUGPARAMS(fPtTID(0))
    SkDEBUGPARAMS(fSegmentID(0))
    SkDEBUGPARAMS(fSpanID(0))
    SkDEBUGPARAMS(fDebugSkipAssert(debugSkipAssert)) {
#if DEBUG_T_SECT_LOOP_COUNT
    debugResetLoopCounts();
#endif
#if DEBUG_COIN
    fPreviousFuncName = nullptr;
#endif
}
