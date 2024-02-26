/*
 * Copyright 2006 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "include/core/SkMatrix.h"
#include "include/core/SkPoint3.h"
#include "include/private/SkTPin.h"
#include "include/private/SkVx.h"
#include "src/core/SkGeometry.h"
#include "src/core/SkPointPriv.h"

#include <algorithm>
#include <tuple>
#include <utility>

namespace {

using float2 = skvx::float2;
using float4 = skvx::float4;

SkVector to_vector(const float2& x) {
    SkVector vector;
    x.store(&vector);
    return vector;
}

////////////////////////////////////////////////////////////////////////

int is_not_monotonic(SkScalar a, SkScalar b, SkScalar c) {
    SkScalar ab = a - b;
    SkScalar bc = b - c;
    if (ab < 0) {
        bc = -bc;
    }
    return ab == 0 || bc < 0;
}

////////////////////////////////////////////////////////////////////////

int valid_unit_divide(SkScalar numer, SkScalar denom, SkScalar* ratio) {
    SkASSERT(ratio);

    if (numer < 0) {
        numer = -numer;
        denom = -denom;
    }

    if (denom == 0 || numer == 0 || numer >= denom) {
        return 0;
    }

    SkScalar r = numer / denom;
    if (SkScalarIsNaN(r)) {
        return 0;
    }
    SkASSERTF(r >= 0 && r < SK_Scalar1, "numer %f, denom %f, r %f", numer, denom, r);
    if (r == 0) { // catch underflow if numer <<<< denom
        return 0;
    }
    *ratio = r;
    return 1;
}

// Just returns its argument, but makes it easy to set a break-point to know when
// SkFindUnitQuadRoots is going to return 0 (an error).
int return_check_zero(int value) {
    if (value == 0) {
        return 0;
    }
    return value;
}

} // namespace

/** From Numerical Recipes in C.

    Q = -1/2 (B + sign(B) sqrt[B*B - 4*A*C])
    x1 = Q / A
    x2 = C / Q
*/
int SkFindUnitQuadRoots(SkScalar A, SkScalar B, SkScalar C, SkScalar roots[2]) {
    SkASSERT(roots);

    if (A == 0) {
        return return_check_zero(valid_unit_divide(-C, B, roots));
    }

    SkScalar* r = roots;

    // use doubles so we don't overflow temporarily trying to compute R
    double dr = (double)B * B - 4 * (double)A * C;
    if (dr < 0) {
        return return_check_zero(0);
    }
    dr = sqrt(dr);
    SkScalar R = SkDoubleToScalar(dr);
    if (!SkScalarIsFinite(R)) {
        return return_check_zero(0);
    }

    SkScalar Q = (B < 0) ? -(B-R)/2 : -(B+R)/2;
    r += valid_unit_divide(Q, A, r);
    r += valid_unit_divide(C, Q, r);
    if (r - roots == 2) {
        if (roots[0] > roots[1]) {
            using std::swap;
            swap(roots[0], roots[1]);
        } else if (roots[0] == roots[1]) { // nearly-equal?
            r -= 1; // skip the double root
        }
    }
    return return_check_zero((int)(r - roots));
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void SkEvalQuadAt(const SkPoint src[3], SkScalar t, SkPoint* pt, SkVector* tangent) {
    SkASSERT(src);
    SkASSERT(t >= 0 && t <= SK_Scalar1);

    if (pt) {
        *pt = SkEvalQuadAt(src, t);
    }
    if (tangent) {
        *tangent = SkEvalQuadTangentAt(src, t);
    }
}

SkPoint SkEvalQuadAt(const SkPoint src[3], SkScalar t) {
    return to_point(SkQuadCoeff(src).eval(t));
}

SkVector SkEvalQuadTangentAt(const SkPoint src[3], SkScalar t) {
    // The derivative equation is 2(b - a +(a - 2b +c)t). This returns a
    // zero tangent vector when t is 0 or 1, and the control point is equal
    // to the end point. In this case, use the quad end points to compute the tangent.
    if ((t == 0 && src[0] == src[1]) || (t == 1 && src[1] == src[2])) {
        return src[2] - src[0];
    }
    SkASSERT(src);
    SkASSERT(t >= 0 && t <= SK_Scalar1);

    float2 P0 = from_point(src[0]);
    float2 P1 = from_point(src[1]);
    float2 P2 = from_point(src[2]);

    float2 B = P1 - P0;
    float2 A = P2 - P1 - B;
    float2 T = A * t + B;

    return to_vector(T + T);
}

static inline float2 interp(const float2& v0,
                            const float2& v1,
                            const float2& t) {
    return v0 + (v1 - v0) * t;
}

void SkChopQuadAt(const SkPoint src[3], SkPoint dst[5], SkScalar t) {
    SkASSERT(t > 0 && t < SK_Scalar1);

    float2 p0 = from_point(src[0]);
    float2 p1 = from_point(src[1]);
    float2 p2 = from_point(src[2]);
    float2 tt(t);

    float2 p01 = interp(p0, p1, tt);
    float2 p12 = interp(p1, p2, tt);

    dst[0] = to_point(p0);
    dst[1] = to_point(p01);
    dst[2] = to_point(interp(p01, p12, tt));
    dst[3] = to_point(p12);
    dst[4] = to_point(p2);
}

void SkChopQuadAtHalf(const SkPoint src[3], SkPoint dst[5]) {
    SkChopQuadAt(src, dst, 0.5f);
}

float SkMeasureAngleBetweenVectors(SkVector a, SkVector b) {
    float cosTheta = sk_ieee_float_divide(a.dot(b), sqrtf(a.dot(a) * b.dot(b)));
    // Pin cosTheta such that if it is NaN (e.g., if a or b was 0), then we return acos(1) = 0.
    cosTheta = std::max(std::min(1.f, cosTheta), -1.f);
    return acosf(cosTheta);
}

SkVector SkFindBisector(SkVector a, SkVector b) {
    std::array<SkVector, 2> v;
    if (a.dot(b) >= 0) {
        // a,b are within +/-90 degrees apart.
        v = {a, b};
    } else if (a.cross(b) >= 0) {
        // a,b are >90 degrees apart. Find the bisector of their interior normals instead. (Above 90
        // degrees, the original vectors start cancelling each other out which eventually becomes
        // unstable.)
        v[0].set(-a.fY, +a.fX);
        v[1].set(+b.fY, -b.fX);
    } else {
        // a,b are <-90 degrees apart. Find the bisector of their interior normals instead. (Below
        // -90 degrees, the original vectors start cancelling each other out which eventually
        // becomes unstable.)
        v[0].set(+a.fY, -a.fX);
        v[1].set(-b.fY, +b.fX);
    }
    // Return "normalize(v[0]) + normalize(v[1])".
    skvx::float2 x0_x1{v[0].fX, v[1].fX};
    skvx::float2 y0_y1{v[0].fY, v[1].fY};
    auto invLengths = 1.0f / sqrt(x0_x1 * x0_x1 + y0_y1 * y0_y1);
    x0_x1 *= invLengths;
    y0_y1 *= invLengths;
    return SkPoint{x0_x1[0] + x0_x1[1], y0_y1[0] + y0_y1[1]};
}

float SkFindQuadMidTangent(const SkPoint src[3]) {
    // Tangents point in the direction of increasing T, so tan0 and -tan1 both point toward the
    // midtangent. The bisector of tan0 and -tan1 is orthogonal to the midtangent:
    //
    //     n dot midtangent = 0
    //
    SkVector tan0 = src[1] - src[0];
    SkVector tan1 = src[2] - src[1];
    SkVector bisector = SkFindBisector(tan0, -tan1);

    // The midtangent can be found where (F' dot bisector) = 0:
    //
    //   0 = (F'(T) dot bisector) = |2*T 1| * |p0 - 2*p1 + p2| * |bisector.x|
    //                                        |-2*p0 + 2*p1  |   |bisector.y|
    //
    //                     = |2*T 1| * |tan1 - tan0| * |nx|
    //                                 |2*tan0     |   |ny|
    //
    //                     = 2*T * ((tan1 - tan0) dot bisector) + (2*tan0 dot bisector)
    //
    //   T = (tan0 dot bisector) / ((tan0 - tan1) dot bisector)
    float T = sk_ieee_float_divide(tan0.dot(bisector), (tan0 - tan1).dot(bisector));
    if (!(T > 0 && T < 1)) {  // Use "!(positive_logic)" so T=nan will take this branch.
        T = .5;  // The quadratic was a line or near-line. Just chop at .5.
    }

    return T;
}

/** Quad'(t) = At + B, where
    A = 2(a - 2b + c)
    B = 2(b - a)
    Solve for t, only if it fits between 0 < t < 1
*/
int SkFindQuadExtrema(SkScalar a, SkScalar b, SkScalar c, SkScalar tValue[1]) {
    /*  At + B == 0
        t = -B / A
    */
    return valid_unit_divide(a - b, a - b - b + c, tValue);
}

static inline void flatten_double_quad_extrema(SkScalar coords[14]) {
    coords[2] = coords[6] = coords[4];
}

/*  Returns 0 for 1 quad, and 1 for two quads, either way the answer is
 stored in dst[]. Guarantees that the 1/2 quads will be monotonic.
 */
int SkChopQuadAtYExtrema(const SkPoint src[3], SkPoint dst[5]) {
    SkASSERT(src);
    SkASSERT(dst);

    SkScalar a = src[0].fY;
    SkScalar b = src[1].fY;
    SkScalar c = src[2].fY;

    if (is_not_monotonic(a, b, c)) {
        SkScalar    tValue;
        if (valid_unit_divide(a - b, a - b - b + c, &tValue)) {
            SkChopQuadAt(src, dst, tValue);
            flatten_double_quad_extrema(&dst[0].fY);
            return 1;
        }
        // if we get here, we need to force dst to be monotonic, even though
        // we couldn't compute a unit_divide value (probably underflow).
        b = SkScalarAbs(a - b) < SkScalarAbs(b - c) ? a : c;
    }
    dst[0].set(src[0].fX, a);
    dst[1].set(src[1].fX, b);
    dst[2].set(src[2].fX, c);
    return 0;
}

/*  Returns 0 for 1 quad, and 1 for two quads, either way the answer is
    stored in dst[]. Guarantees that the 1/2 quads will be monotonic.
 */
int SkChopQuadAtXExtrema(const SkPoint src[3], SkPoint dst[5]) {
    SkASSERT(src);
    SkASSERT(dst);

    SkScalar a = src[0].fX;
    SkScalar b = src[1].fX;
    SkScalar c = src[2].fX;

    if (is_not_monotonic(a, b, c)) {
        SkScalar tValue;
        if (valid_unit_divide(a - b, a - b - b + c, &tValue)) {
            SkChopQuadAt(src, dst, tValue);
            flatten_double_quad_extrema(&dst[0].fX);
            return 1;
        }
        // if we get here, we need to force dst to be monotonic, even though
        // we couldn't compute a unit_divide value (probably underflow).
        b = SkScalarAbs(a - b) < SkScalarAbs(b - c) ? a : c;
    }
    dst[0].set(a, src[0].fY);
    dst[1].set(b, src[1].fY);
    dst[2].set(c, src[2].fY);
    return 0;
}

//  F(t)    = a (1 - t) ^ 2 + 2 b t (1 - t) + c t ^ 2
//  F'(t)   = 2 (b - a) + 2 (a - 2b + c) t
//  F''(t)  = 2 (a - 2b + c)
//
//  A = 2 (b - a)
//  B = 2 (a - 2b + c)
//
//  Maximum curvature for a quadratic means solving
//  Fx' Fx'' + Fy' Fy'' = 0
//
//  t = - (Ax Bx + Ay By) / (Bx ^ 2 + By ^ 2)
//
SkScalar SkFindQuadMaxCurvature(const SkPoint src[3]) {
    SkScalar    Ax = src[1].fX - src[0].fX;
    SkScalar    Ay = src[1].fY - src[0].fY;
    SkScalar    Bx = src[0].fX - src[1].fX - src[1].fX + src[2].fX;
    SkScalar    By = src[0].fY - src[1].fY - src[1].fY + src[2].fY;

    SkScalar numer = -(Ax * Bx + Ay * By);
    SkScalar denom = Bx * Bx + By * By;
    if (denom < 0) {
        numer = -numer;
        denom = -denom;
    }
    if (numer <= 0) {
        return 0;
    }
    if (numer >= denom) {  // Also catches denom=0.
        return 1;
    }
    SkScalar t = numer / denom;
    SkASSERT((0 <= t && t < 1) || SkScalarIsNaN(t));
    return t;
}

int SkChopQuadAtMaxCurvature(const SkPoint src[3], SkPoint dst[5]) {
    SkScalar t = SkFindQuadMaxCurvature(src);
    if (t > 0 && t < 1) {
        SkChopQuadAt(src, dst, t);
        return 2;
    } else {
        memcpy(dst, src, 3 * sizeof(SkPoint));
        return 1;
    }
}

void SkConvertQuadToCubic(const SkPoint src[3], SkPoint dst[4]) {
    float2 scale(SkDoubleToScalar(2.0 / 3.0));
    float2 s0 = from_point(src[0]);
    float2 s1 = from_point(src[1]);
    float2 s2 = from_point(src[2]);

    dst[0] = to_point(s0);
    dst[1] = to_point(s0 + (s1 - s0) * scale);
    dst[2] = to_point(s2 + (s1 - s2) * scale);
    dst[3] = to_point(s2);
}

//////////////////////////////////////////////////////////////////////////////
///// CUBICS // CUBICS // CUBICS // CUBICS // CUBICS // CUBICS // CUBICS /////
//////////////////////////////////////////////////////////////////////////////

static SkVector eval_cubic_derivative(const SkPoint src[4], SkScalar t) {
    SkQuadCoeff coeff;
    float2 P0 = from_point(src[0]);
    float2 P1 = from_point(src[1]);
    float2 P2 = from_point(src[2]);
    float2 P3 = from_point(src[3]);

    coeff.fA = P3 + 3 * (P1 - P2) - P0;
    coeff.fB = times_2(P2 - times_2(P1) + P0);
    coeff.fC = P1 - P0;
    return to_vector(coeff.eval(t));
}

static SkVector eval_cubic_2ndDerivative(const SkPoint src[4], SkScalar t) {
    float2 P0 = from_point(src[0]);
    float2 P1 = from_point(src[1]);
    float2 P2 = from_point(src[2]);
    float2 P3 = from_point(src[3]);
    float2 A = P3 + 3 * (P1 - P2) - P0;
    float2 B = P2 - times_2(P1) + P0;

    return to_vector(A * t + B);
}

void SkEvalCubicAt(const SkPoint src[4], SkScalar t, SkPoint* loc,
                   SkVector* tangent, SkVector* curvature) {
    SkASSERT(src);
    SkASSERT(t >= 0 && t <= SK_Scalar1);

    if (loc) {
        *loc = to_point(SkCubicCoeff(src).eval(t));
    }
    if (tangent) {
        // The derivative equation returns a zero tangent vector when t is 0 or 1, and the
        // adjacent control point is equal to the end point. In this case, use the
        // next control point or the end points to compute the tangent.
        if ((t == 0 && src[0] == src[1]) || (t == 1 && src[2] == src[3])) {
            if (t == 0) {
                *tangent = src[2] - src[0];
            } else {
                *tangent = src[3] - src[1];
            }
            if (!tangent->fX && !tangent->fY) {
                *tangent = src[3] - src[0];
            }
        } else {
            *tangent = eval_cubic_derivative(src, t);
        }
    }
    if (curvature) {
        *curvature = eval_cubic_2ndDerivative(src, t);
    }
}

/** Cubic'(t) = At^2 + Bt + C, where
    A = 3(-a + 3(b - c) + d)
    B = 6(a - 2b + c)
    C = 3(b - a)
    Solve for t, keeping only those that fit betwee 0 < t < 1
*/
int SkFindCubicExtrema(SkScalar a, SkScalar b, SkScalar c, SkScalar d,
                       SkScalar tValues[2]) {
    // we divide A,B,C by 3 to simplify
    SkScalar A = d - a + 3*(b - c);
    SkScalar B = 2*(a - b - b + c);
    SkScalar C = b - a;

    return SkFindUnitQuadRoots(A, B, C, tValues);
}

// This does not return b when t==1, but it otherwise seems to get better precision than
// "a*(1 - t) + b*t" for things like chopping cubics on exact cusp points.
// The responsibility falls on the caller to check that t != 1 before calling.
template<int N, typename T>
inline static skvx::Vec<N,T> unchecked_mix(const skvx::Vec<N,T>& a, const skvx::Vec<N,T>& b,
                                           const skvx::Vec<N,T>& t) {
    return (b - a)*t + a;
}

void SkChopCubicAt(const SkPoint src[4], SkPoint dst[7], SkScalar t) {
    SkASSERT(0 <= t && t <= 1);

    if (t == 1) {
        memcpy(dst, src, sizeof(SkPoint) * 4);
        dst[4] = dst[5] = dst[6] = src[3];
        return;
    }

    float2 p0 = skvx::bit_pun<float2>(src[0]);
    float2 p1 = skvx::bit_pun<float2>(src[1]);
    float2 p2 = skvx::bit_pun<float2>(src[2]);
    float2 p3 = skvx::bit_pun<float2>(src[3]);
    float2 T = t;

    float2 ab = unchecked_mix(p0, p1, T);
    float2 bc = unchecked_mix(p1, p2, T);
    float2 cd = unchecked_mix(p2, p3, T);
    float2 abc = unchecked_mix(ab, bc, T);
    float2 bcd = unchecked_mix(bc, cd, T);
    float2 abcd = unchecked_mix(abc, bcd, T);

    dst[0] = skvx::bit_pun<SkPoint>(p0);
    dst[1] = skvx::bit_pun<SkPoint>(ab);
    dst[2] = skvx::bit_pun<SkPoint>(abc);
    dst[3] = skvx::bit_pun<SkPoint>(abcd);
    dst[4] = skvx::bit_pun<SkPoint>(bcd);
    dst[5] = skvx::bit_pun<SkPoint>(cd);
    dst[6] = skvx::bit_pun<SkPoint>(p3);
}

void SkChopCubicAt(const SkPoint src[4], SkPoint dst[10], float t0, float t1) {
    SkASSERT(0 <= t0 && t0 <= t1 && t1 <= 1);

    if (t1 == 1) {
        SkChopCubicAt(src, dst, t0);
        dst[7] = dst[8] = dst[9] = src[3];
        return;
    }

    // Perform both chops in parallel using 4-lane SIMD.
    float4 p00, p11, p22, p33, T;
    p00.lo = p00.hi = skvx::bit_pun<float2>(src[0]);
    p11.lo = p11.hi = skvx::bit_pun<float2>(src[1]);
    p22.lo = p22.hi = skvx::bit_pun<float2>(src[2]);
    p33.lo = p33.hi = skvx::bit_pun<float2>(src[3]);
    T.lo = t0;
    T.hi = t1;

    float4 ab = unchecked_mix(p00, p11, T);
    float4 bc = unchecked_mix(p11, p22, T);
    float4 cd = unchecked_mix(p22, p33, T);
    float4 abc = unchecked_mix(ab, bc, T);
    float4 bcd = unchecked_mix(bc, cd, T);
    float4 abcd = unchecked_mix(abc, bcd, T);
    float4 middle = unchecked_mix(abc, bcd, skvx::shuffle<2,3,0,1>(T));

    dst[0] = skvx::bit_pun<SkPoint>(p00.lo);
    dst[1] = skvx::bit_pun<SkPoint>(ab.lo);
    dst[2] = skvx::bit_pun<SkPoint>(abc.lo);
    dst[3] = skvx::bit_pun<SkPoint>(abcd.lo);
    middle.store(dst + 4);
    dst[6] = skvx::bit_pun<SkPoint>(abcd.hi);
    dst[7] = skvx::bit_pun<SkPoint>(bcd.hi);
    dst[8] = skvx::bit_pun<SkPoint>(cd.hi);
    dst[9] = skvx::bit_pun<SkPoint>(p33.hi);
}

void SkChopCubicAt(const SkPoint src[4], SkPoint dst[],
                   const SkScalar tValues[], int tCount) {
    SkASSERT(std::all_of(tValues, tValues + tCount, [](SkScalar t) { return t >= 0 && t <= 1; }));
    SkASSERT(std::is_sorted(tValues, tValues + tCount));

    if (dst) {
        if (tCount == 0) { // nothing to chop
            memcpy(dst, src, 4*sizeof(SkPoint));
        } else {
            int i = 0;
            for (; i < tCount - 1; i += 2) {
                // Do two chops at once.
                float2 tt = float2::Load(tValues + i);
                if (i != 0) {
                    float lastT = tValues[i - 1];
                    tt = skvx::pin((tt - lastT) / (1 - lastT), float2(0), float2(1));
                }
                SkChopCubicAt(src, dst, tt[0], tt[1]);
                src = dst = dst + 6;
            }
            if (i < tCount) {
                // Chop the final cubic if there was an odd number of chops.
                SkASSERT(i + 1 == tCount);
                float t = tValues[i];
                if (i != 0) {
                    float lastT = tValues[i - 1];
                    t = SkTPin(sk_ieee_float_divide(t - lastT, 1 - lastT), 0.f, 1.f);
                }
                SkChopCubicAt(src, dst, t);
            }
        }
    }
}

void SkChopCubicAtHalf(const SkPoint src[4], SkPoint dst[7]) {
    SkChopCubicAt(src, dst, 0.5f);
}

float SkMeasureNonInflectCubicRotation(const SkPoint pts[4]) {
    SkVector a = pts[1] - pts[0];
    SkVector b = pts[2] - pts[1];
    SkVector c = pts[3] - pts[2];
    if (a.isZero()) {
        return SkMeasureAngleBetweenVectors(b, c);
    }
    if (b.isZero()) {
        return SkMeasureAngleBetweenVectors(a, c);
    }
    if (c.isZero()) {
        return SkMeasureAngleBetweenVectors(a, b);
    }
    // Postulate: When no points are colocated and there are no inflection points in T=0..1, the
    // rotation is: 360 degrees, minus the angle [p0,p1,p2], minus the angle [p1,p2,p3].
    return 2*SK_ScalarPI - SkMeasureAngleBetweenVectors(a,-b) - SkMeasureAngleBetweenVectors(b,-c);
}

static skvx::float4 fma(const skvx::float4& f, float m, const skvx::float4& a) {
    return skvx::fma(f, skvx::float4(m), a);
}

// Finds the root nearest 0.5. Returns 0.5 if the roots are undefined or outside 0..1.
static float solve_quadratic_equation_for_midtangent(float a, float b, float c, float discr) {
    // Quadratic formula from Numerical Recipes in C:
    float q = -.5f * (b + copysignf(sqrtf(discr), b));
    // The roots are q/a and c/q. Pick the midtangent closer to T=.5.
    float _5qa = -.5f*q*a;
    float T = fabsf(q*q + _5qa) < fabsf(a*c + _5qa) ? sk_ieee_float_divide(q,a)
                                                    : sk_ieee_float_divide(c,q);
    if (!(T > 0 && T < 1)) {  // Use "!(positive_logic)" so T=NaN will take this branch.
        // Either the curve is a flat line with no rotation or FP precision failed us. Chop at .5.
        T = .5;
    }
    return T;
}

static float solve_quadratic_equation_for_midtangent(float a, float b, float c) {
    return solve_quadratic_equation_for_midtangent(a, b, c, b*b - 4*a*c);
}

float SkFindCubicMidTangent(const SkPoint src[4]) {
    // Tangents point in the direction of increasing T, so tan0 and -tan1 both point toward the
    // midtangent. The bisector of tan0 and -tan1 is orthogonal to the midtangent:
    //
    //     bisector dot midtangent == 0
    //
    SkVector tan0 = (src[0] == src[1]) ? src[2] - src[0] : src[1] - src[0];
    SkVector tan1 = (src[2] == src[3]) ? src[3] - src[1] : src[3] - src[2];
    SkVector bisector = SkFindBisector(tan0, -tan1);

    // Find the T value at the midtangent. This is a simple quadratic equation:
    //
    //     midtangent dot bisector == 0, or using a tangent matrix C' in power basis form:
    //
    //                   |C'x  C'y|
    //     |T^2  T  1| * |.    .  | * |bisector.x| == 0
    //                   |.    .  |   |bisector.y|
    //
    // The coeffs for the quadratic equation we need to solve are therefore:  C' * bisector
    static const skvx::float4 kM[4] = {skvx::float4(-1,  2, -1,  0),
                                       skvx::float4( 3, -4,  1,  0),
                                       skvx::float4(-3,  2,  0,  0)};
    auto C_x = fma(kM[0], src[0].fX,
               fma(kM[1], src[1].fX,
               fma(kM[2], src[2].fX, skvx::float4(src[3].fX, 0,0,0))));
    auto C_y = fma(kM[0], src[0].fY,
               fma(kM[1], src[1].fY,
               fma(kM[2], src[2].fY, skvx::float4(src[3].fY, 0,0,0))));
    auto coeffs = C_x * bisector.x() + C_y * bisector.y();

    // Now solve the quadratic for T.
    float T = 0;
    float a=coeffs[0], b=coeffs[1], c=coeffs[2];
    float discr = b*b - 4*a*c;
    if (discr > 0) {  // This will only be false if the curve is a line.
        return solve_quadratic_equation_for_midtangent(a, b, c, discr);
    } else {
        // This is a 0- or 360-degree flat line. It doesn't have single points of midtangent.
        // (tangent == midtangent at every point on the curve except the cusp points.)
        // Chop in between both cusps instead, if any. There can be up to two cusps on a flat line,
        // both where the tangent is perpendicular to the starting tangent:
        //
        //     tangent dot tan0 == 0
        //
        coeffs = C_x * tan0.x() + C_y * tan0.y();
        a = coeffs[0];
        b = coeffs[1];
        if (a != 0) {
            // We want the point in between both cusps. The midpoint of:
            //
            //     (-b +/- sqrt(b^2 - 4*a*c)) / (2*a)
            //
            // Is equal to:
            //
            //     -b / (2*a)
            T = -b / (2*a);
        }
        if (!(T > 0 && T < 1)) {  // Use "!(positive_logic)" so T=NaN will take this branch.
            // Either the curve is a flat line with no rotation or FP precision failed us. Chop at
            // .5.
            T = .5;
        }
        return T;
    }
}

static void flatten_double_cubic_extrema(SkScalar coords[14]) {
    coords[4] = coords[8] = coords[6];
}

/** Given 4 points on a cubic bezier, chop it into 1, 2, 3 beziers such that
    the resulting beziers are monotonic in Y. This is called by the scan
    converter.  Depending on what is returned, dst[] is treated as follows:
    0   dst[0..3] is the original cubic
    1   dst[0..3] and dst[3..6] are the two new cubics
    2   dst[0..3], dst[3..6], dst[6..9] are the three new cubics
    If dst == null, it is ignored and only the count is returned.
*/
int SkChopCubicAtYExtrema(const SkPoint src[4], SkPoint dst[10]) {
    SkScalar    tValues[2];
    int         roots = SkFindCubicExtrema(src[0].fY, src[1].fY, src[2].fY,
                                           src[3].fY, tValues);

    SkChopCubicAt(src, dst, tValues, roots);
    if (dst && roots > 0) {
        // we do some cleanup to ensure our Y extrema are flat
        flatten_double_cubic_extrema(&dst[0].fY);
        if (roots == 2) {
            flatten_double_cubic_extrema(&dst[3].fY);
        }
    }
    return roots;
}

int SkChopCubicAtXExtrema(const SkPoint src[4], SkPoint dst[10]) {
    SkScalar    tValues[2];
    int         roots = SkFindCubicExtrema(src[0].fX, src[1].fX, src[2].fX,
                                           src[3].fX, tValues);

    SkChopCubicAt(src, dst, tValues, roots);
    if (dst && roots > 0) {
        // we do some cleanup to ensure our Y extrema are flat
        flatten_double_cubic_extrema(&dst[0].fX);
        if (roots == 2) {
            flatten_double_cubic_extrema(&dst[3].fX);
        }
    }
    return roots;
}

/** http://www.faculty.idc.ac.il/arik/quality/appendixA.html

    Inflection means that curvature is zero.
    Curvature is [F' x F''] / [F'^3]
    So we solve F'x X F''y - F'y X F''y == 0
    After some canceling of the cubic term, we get
    A = b - a
    B = c - 2b + a
    C = d - 3c + 3b - a
    (BxCy - ByCx)t^2 + (AxCy - AyCx)t + AxBy - AyBx == 0
*/
int SkFindCubicInflections(const SkPoint src[4], SkScalar tValues[]) {
    SkScalar    Ax = src[1].fX - src[0].fX;
    SkScalar    Ay = src[1].fY - src[0].fY;
    SkScalar    Bx = src[2].fX - 2 * src[1].fX + src[0].fX;
    SkScalar    By = src[2].fY - 2 * src[1].fY + src[0].fY;
    SkScalar    Cx = src[3].fX + 3 * (src[1].fX - src[2].fX) - src[0].fX;
    SkScalar    Cy = src[3].fY + 3 * (src[1].fY - src[2].fY) - src[0].fY;

    return SkFindUnitQuadRoots(Bx*Cy - By*Cx,
                               Ax*Cy - Ay*Cx,
                               Ax*By - Ay*Bx,
                               tValues);
}

int SkChopCubicAtInflections(const SkPoint src[], SkPoint dst[10]) {
    SkScalar    tValues[2];
    int         count = SkFindCubicInflections(src, tValues);

    if (dst) {
        if (count == 0) {
            memcpy(dst, src, 4 * sizeof(SkPoint));
        } else {
            SkChopCubicAt(src, dst, tValues, count);
        }
    }
    return count + 1;
}

// Assumes the third component of points is 1.
// Calcs p0 . (p1 x p2)
static double calc_dot_cross_cubic(const SkPoint& p0, const SkPoint& p1, const SkPoint& p2) {
    const double xComp = (double) p0.fX * ((double) p1.fY - (double) p2.fY);
    const double yComp = (double) p0.fY * ((double) p2.fX - (double) p1.fX);
    const double wComp = (double) p1.fX * (double) p2.fY - (double) p1.fY * (double) p2.fX;
    return (xComp + yComp + wComp);
}

// Returns a positive power of 2 that, when multiplied by n, and excepting the two edge cases listed
// below, shifts the exponent of n to yield a magnitude somewhere inside [1..2).
// Returns 2^1023 if abs(n) < 2^-1022 (including 0).
// Returns NaN if n is Inf or NaN.
inline static double previous_inverse_pow2(double n) {
    uint64_t bits;
    memcpy(&bits, &n, sizeof(double));
    bits = ((1023llu*2 << 52) + ((1llu << 52) - 1)) - bits; // exp=-exp
    bits &= (0x7ffllu) << 52; // mantissa=1.0, sign=0
    memcpy(&n, &bits, sizeof(double));
    return n;
}

inline static void write_cubic_inflection_roots(double t0, double s0, double t1, double s1,
                                                double* t, double* s) {
    t[0] = t0;
    s[0] = s0;

    // This copysign/abs business orients the implicit function so positive values are always on the
    // "left" side of the curve.
    t[1] = -copysign(t1, t1 * s1);
    s[1] = -fabs(s1);

    // Ensure t[0]/s[0] <= t[1]/s[1] (s[1] is negative from above).
    if (copysign(s[1], s[0]) * t[0] > -fabs(s[0]) * t[1]) {
        using std::swap;
        swap(t[0], t[1]);
        swap(s[0], s[1]);
    }
}

SkCubicType SkClassifyCubic(const SkPoint P[4], double t[2], double s[2], double d[4]) {
    // Find the cubic's inflection function, I = [T^3  -3T^2  3T  -1] dot D. (D0 will always be 0
    // for integral cubics.)
    //
    // See "Resolution Independent Curve Rendering using Programmable Graphics Hardware",
    // 4.2 Curve Categorization:
    //
    // https://www.microsoft.com/en-us/research/wp-content/uploads/2005/01/p1000-loop.pdf
    double A1 = calc_dot_cross_cubic(P[0], P[3], P[2]);
    double A2 = calc_dot_cross_cubic(P[1], P[0], P[3]);
    double A3 = calc_dot_cross_cubic(P[2], P[1], P[0]);

    double D3 = 3 * A3;
    double D2 = D3 - A2;
    double D1 = D2 - A2 + A1;

    // Shift the exponents in D so the largest magnitude falls somewhere in 1..2. This protects us
    // from overflow down the road while solving for roots and KLM functionals.
    double Dmax = std::max(std::max(fabs(D1), fabs(D2)), fabs(D3));
    double norm = previous_inverse_pow2(Dmax);
    D1 *= norm;
    D2 *= norm;
    D3 *= norm;

    if (d) {
        d[3] = D3;
        d[2] = D2;
        d[1] = D1;
        d[0] = 0;
    }

    // Now use the inflection function to classify the cubic.
    //
    // See "Resolution Independent Curve Rendering using Programmable Graphics Hardware",
    // 4.4 Integral Cubics:
    //
    // https://www.microsoft.com/en-us/research/wp-content/uploads/2005/01/p1000-loop.pdf
    if (0 != D1) {
        double discr = 3*D2*D2 - 4*D1*D3;
        if (discr > 0) { // Serpentine.
            if (t && s) {
                double q = 3*D2 + copysign(sqrt(3*discr), D2);
                write_cubic_inflection_roots(q, 6*D1, 2*D3, q, t, s);
            }
            return SkCubicType::kSerpentine;
        } else if (discr < 0) { // Loop.
            if (t && s) {
                double q = D2 + copysign(sqrt(-discr), D2);
                write_cubic_inflection_roots(q, 2*D1, 2*(D2*D2 - D3*D1), D1*q, t, s);
            }
            return SkCubicType::kLoop;
        } else { // Cusp.
            if (t && s) {
                write_cubic_inflection_roots(D2, 2*D1, D2, 2*D1, t, s);
            }
            return SkCubicType::kLocalCusp;
        }
    } else {
        if (0 != D2) { // Cusp at T=infinity.
            if (t && s) {
                write_cubic_inflection_roots(D3, 3*D2, 1, 0, t, s); // T1=infinity.
            }
            return SkCubicType::kCuspAtInfinity;
        } else { // Degenerate.
            if (t && s) {
                write_cubic_inflection_roots(1, 0, 1, 0, t, s); // T0=T1=infinity.
            }
            return 0 != D3 ? SkCubicType::kQuadratic : SkCubicType::kLineOrPoint;
        }
    }
}

template <typename T> void bubble_sort(T array[], int count) {
    for (int i = count - 1; i > 0; --i)
        for (int j = i; j > 0; --j)
            if (array[j] < array[j-1])
            {
                T   tmp(array[j]);
                array[j] = array[j-1];
                array[j-1] = tmp;
            }
}

/**
 *  Given an array and count, remove all pair-wise duplicates from the array,
 *  keeping the existing sorting, and return the new count
 */
static int collaps_duplicates(SkScalar array[], int count) {
    for (int n = count; n > 1; --n) {
        if (array[0] == array[1]) {
            for (int i = 1; i < n; ++i) {
                array[i - 1] = array[i];
            }
            count -= 1;
        } else {
            array += 1;
        }
    }
    return count;
}

#ifdef SK_DEBUG

#define TEST_COLLAPS_ENTRY(array)   array, SK_ARRAY_COUNT(array)

static void test_collaps_duplicates() {
    static bool gOnce;
    if (gOnce) { return; }
    gOnce = true;
    const SkScalar src0[] = { 0 };
    const SkScalar src1[] = { 0, 0 };
    const SkScalar src2[] = { 0, 1 };
    const SkScalar src3[] = { 0, 0, 0 };
    const SkScalar src4[] = { 0, 0, 1 };
    const SkScalar src5[] = { 0, 1, 1 };
    const SkScalar src6[] = { 0, 1, 2 };
    const struct {
        const SkScalar* fData;
        int fCount;
        int fCollapsedCount;
    } data[] = {
        { TEST_COLLAPS_ENTRY(src0), 1 },
        { TEST_COLLAPS_ENTRY(src1), 1 },
        { TEST_COLLAPS_ENTRY(src2), 2 },
        { TEST_COLLAPS_ENTRY(src3), 1 },
        { TEST_COLLAPS_ENTRY(src4), 2 },
        { TEST_COLLAPS_ENTRY(src5), 2 },
        { TEST_COLLAPS_ENTRY(src6), 3 },
    };
    for (size_t i = 0; i < SK_ARRAY_COUNT(data); ++i) {
        SkScalar dst[3];
        memcpy(dst, data[i].fData, data[i].fCount * sizeof(dst[0]));
        int count = collaps_duplicates(dst, data[i].fCount);
        SkASSERT(data[i].fCollapsedCount == count);
        for (int j = 1; j < count; ++j) {
            SkASSERT(dst[j-1] < dst[j]);
        }
    }
}
#endif

static SkScalar SkScalarCubeRoot(SkScalar x) {
    return SkScalarPow(x, 0.3333333f);
}

/*  Solve coeff(t) == 0, returning the number of roots that
    lie withing 0 < t < 1.
    coeff[0]t^3 + coeff[1]t^2 + coeff[2]t + coeff[3]

    Eliminates repeated roots (so that all tValues are distinct, and are always
    in increasing order.
*/
static int solve_cubic_poly(const SkScalar coeff[4], SkScalar tValues[3]) {
    if (SkScalarNearlyZero(coeff[0])) {  // we're just a quadratic
        return SkFindUnitQuadRoots(coeff[1], coeff[2], coeff[3], tValues);
    }

    SkScalar a, b, c, Q, R;

    {
        SkASSERT(coeff[0] != 0);

        SkScalar inva = SkScalarInvert(coeff[0]);
        a = coeff[1] * inva;
        b = coeff[2] * inva;
        c = coeff[3] * inva;
    }
    Q = (a*a - b*3) / 9;
    R = (2*a*a*a - 9*a*b + 27*c) / 54;

    SkScalar Q3 = Q * Q * Q;
    SkScalar R2MinusQ3 = R * R - Q3;
    SkScalar adiv3 = a / 3;

    if (R2MinusQ3 < 0) { // we have 3 real roots
        // the divide/root can, due to finite precisions, be slightly outside of -1...1
        SkScalar theta = SkScalarACos(SkTPin(R / SkScalarSqrt(Q3), -1.0f, 1.0f));
        SkScalar neg2RootQ = -2 * SkScalarSqrt(Q);

        tValues[0] = SkTPin(neg2RootQ * SkScalarCos(theta/3) - adiv3, 0.0f, 1.0f);
        tValues[1] = SkTPin(neg2RootQ * SkScalarCos((theta + 2*SK_ScalarPI)/3) - adiv3, 0.0f, 1.0f);
        tValues[2] = SkTPin(neg2RootQ * SkScalarCos((theta - 2*SK_ScalarPI)/3) - adiv3, 0.0f, 1.0f);
        SkDEBUGCODE(test_collaps_duplicates();)

        // now sort the roots
        bubble_sort(tValues, 3);
        return collaps_duplicates(tValues, 3);
    } else {              // we have 1 real root
        SkScalar A = SkScalarAbs(R) + SkScalarSqrt(R2MinusQ3);
        A = SkScalarCubeRoot(A);
        if (R > 0) {
            A = -A;
        }
        if (A != 0) {
            A += Q / A;
        }
        tValues[0] = SkTPin(A - adiv3, 0.0f, 1.0f);
        return 1;
    }
}

/*  Looking for F' dot F'' == 0

    A = b - a
    B = c - 2b + a
    C = d - 3c + 3b - a

    F' = 3Ct^2 + 6Bt + 3A
    F'' = 6Ct + 6B

    F' dot F'' -> CCt^3 + 3BCt^2 + (2BB + CA)t + AB
*/
static void formulate_F1DotF2(const SkScalar src[], SkScalar coeff[4]) {
    SkScalar    a = src[2] - src[0];
    SkScalar    b = src[4] - 2 * src[2] + src[0];
    SkScalar    c = src[6] + 3 * (src[2] - src[4]) - src[0];

    coeff[0] = c * c;
    coeff[1] = 3 * b * c;
    coeff[2] = 2 * b * b + c * a;
    coeff[3] = a * b;
}

/*  Looking for F' dot F'' == 0

    A = b - a
    B = c - 2b + a
    C = d - 3c + 3b - a

    F' = 3Ct^2 + 6Bt + 3A
    F'' = 6Ct + 6B

    F' dot F'' -> CCt^3 + 3BCt^2 + (2BB + CA)t + AB
*/
int SkFindCubicMaxCurvature(const SkPoint src[4], SkScalar tValues[3]) {
    SkScalar coeffX[4], coeffY[4];
    int      i;

    formulate_F1DotF2(&src[0].fX, coeffX);
    formulate_F1DotF2(&src[0].fY, coeffY);

    for (i = 0; i < 4; i++) {
        coeffX[i] += coeffY[i];
    }

    int numRoots = solve_cubic_poly(coeffX, tValues);
    // now remove extrema where the curvature is zero (mins)
    // !!!! need a test for this !!!!
    return numRoots;
}

int SkChopCubicAtMaxCurvature(const SkPoint src[4], SkPoint dst[13],
                              SkScalar tValues[3]) {
    SkScalar    t_storage[3];

    if (tValues == nullptr) {
        tValues = t_storage;
    }

    SkScalar roots[3];
    int rootCount = SkFindCubicMaxCurvature(src, roots);

    // Throw out values not inside 0..1.
    int count = 0;
    for (int i = 0; i < rootCount; ++i) {
        if (0 < roots[i] && roots[i] < 1) {
            tValues[count++] = roots[i];
        }
    }

    if (dst) {
        if (count == 0) {
            memcpy(dst, src, 4 * sizeof(SkPoint));
        } else {
            SkChopCubicAt(src, dst, tValues, count);
        }
    }
    return count + 1;
}

// Returns a constant proportional to the dimensions of the cubic.
// Constant found through experimentation -- maybe there's a better way....
static SkScalar calc_cubic_precision(const SkPoint src[4]) {
    return (SkPointPriv::DistanceToSqd(src[1], src[0]) + SkPointPriv::DistanceToSqd(src[2], src[1])
            + SkPointPriv::DistanceToSqd(src[3], src[2])) * 1e-8f;
}

// Returns true if both points src[testIndex], src[testIndex+1] are in the same half plane defined
// by the line segment src[lineIndex], src[lineIndex+1].
static bool on_same_side(const SkPoint src[4], int testIndex, int lineIndex) {
    SkPoint origin = src[lineIndex];
    SkVector line = src[lineIndex + 1] - origin;
    SkScalar crosses[2];
    for (int index = 0; index < 2; ++index) {
        SkVector testLine = src[testIndex + index] - origin;
        crosses[index] = line.cross(testLine);
    }
    return crosses[0] * crosses[1] >= 0;
}

// Return location (in t) of cubic cusp, if there is one.
// Note that classify cubic code does not reliably return all cusp'd cubics, so
// it is not called here.
SkScalar SkFindCubicCusp(const SkPoint src[4]) {
    // When the adjacent control point matches the end point, it behaves as if
    // the cubic has a cusp: there's a point of max curvature where the derivative
    // goes to zero. Ideally, this would be where t is zero or one, but math
    // error makes not so. It is not uncommon to create cubics this way; skip them.
    if (src[0] == src[1]) {
        return -1;
    }
    if (src[2] == src[3]) {
        return -1;
    }
    // Cubics only have a cusp if the line segments formed by the control and end points cross.
    // Detect crossing if line ends are on opposite sides of plane formed by the other line.
    if (on_same_side(src, 0, 2) || on_same_side(src, 2, 0)) {
        return -1;
    }
    // Cubics may have multiple points of maximum curvature, although at most only
    // one is a cusp.
    SkScalar maxCurvature[3];
    int roots = SkFindCubicMaxCurvature(src, maxCurvature);
    for (int index = 0; index < roots; ++index) {
        SkScalar testT = maxCurvature[index];
        if (0 >= testT || testT >= 1) {  // no need to consider max curvature on the end
            continue;
        }
        // A cusp is at the max curvature, and also has a derivative close to zero.
        // Choose the 'close to zero' meaning by comparing the derivative length
        // with the overall cubic size.
        SkVector dPt = eval_cubic_derivative(src, testT);
        SkScalar dPtMagnitude = SkPointPriv::LengthSqd(dPt);
        SkScalar precision = calc_cubic_precision(src);
        if (dPtMagnitude < precision) {
            // All three max curvature t values may be close to the cusp;
            // return the first one.
            return testT;
        }
    }
    return -1;
}

#include "src/pathops/SkPathOpsCubic.h"

typedef int (SkDCubic::*InterceptProc)(double intercept, double roots[3]) const;

static bool cubic_dchop_at_intercept(const SkPoint src[4], SkScalar intercept, SkPoint dst[7],
                                     InterceptProc method) {
    SkDCubic cubic;
    double roots[3];
    int count = (cubic.set(src).*method)(intercept, roots);
    if (count > 0) {
        SkDCubicPair pair = cubic.chopAt(roots[0]);
        for (int i = 0; i < 7; ++i) {
            dst[i] = pair.pts[i].asSkPoint();
        }
        return true;
    }
    return false;
}

bool SkChopMonoCubicAtY(SkPoint src[4], SkScalar y, SkPoint dst[7]) {
    return cubic_dchop_at_intercept(src, y, dst, &SkDCubic::horizontalIntersect);
}

bool SkChopMonoCubicAtX(SkPoint src[4], SkScalar x, SkPoint dst[7]) {
    return cubic_dchop_at_intercept(src, x, dst, &SkDCubic::verticalIntersect);
}

///////////////////////////////////////////////////////////////////////////////
//
// NURB representation for conics.  Helpful explanations at:
//
// http://citeseerx.ist.psu.edu/viewdoc/
//   download?doi=10.1.1.44.5740&rep=rep1&type=ps
// and
// http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/NURBS/RB-conics.html
//
// F = (A (1 - t)^2 + C t^2 + 2 B (1 - t) t w)
//     ------------------------------------------
//         ((1 - t)^2 + t^2 + 2 (1 - t) t w)
//
//   = {t^2 (P0 + P2 - 2 P1 w), t (-2 P0 + 2 P1 w), P0}
//     ------------------------------------------------
//             {t^2 (2 - 2 w), t (-2 + 2 w), 1}
//

// F' = 2 (C t (1 + t (-1 + w)) - A (-1 + t) (t (-1 + w) - w) + B (1 - 2 t) w)
//
//  t^2 : (2 P0 - 2 P2 - 2 P0 w + 2 P2 w)
//  t^1 : (-2 P0 + 2 P2 + 4 P0 w - 4 P1 w)
//  t^0 : -2 P0 w + 2 P1 w
//
//  We disregard magnitude, so we can freely ignore the denominator of F', and
//  divide the numerator by 2
//
//    coeff[0] for t^2
//    coeff[1] for t^1
//    coeff[2] for t^0
//
static void conic_deriv_coeff(const SkScalar src[],
                              SkScalar w,
                              SkScalar coeff[3]) {
    const SkScalar P20 = src[4] - src[0];
    const SkScalar P10 = src[2] - src[0];
    const SkScalar wP10 = w * P10;
    coeff[0] = w * P20 - P20;
    coeff[1] = P20 - 2 * wP10;
    coeff[2] = wP10;
}

static bool conic_find_extrema(const SkScalar src[], SkScalar w, SkScalar* t) {
    SkScalar coeff[3];
    conic_deriv_coeff(src, w, coeff);

    SkScalar tValues[2];
    int roots = SkFindUnitQuadRoots(coeff[0], coeff[1], coeff[2], tValues);
    SkASSERT(0 == roots || 1 == roots);

    if (1 == roots) {
        *t = tValues[0];
        return true;
    }
    return false;
}

// We only interpolate one dimension at a time (the first, at +0, +3, +6).
static void p3d_interp(const SkScalar src[7], SkScalar dst[7], SkScalar t) {
    SkScalar ab = SkScalarInterp(src[0], src[3], t);
    SkScalar bc = SkScalarInterp(src[3], src[6], t);
    dst[0] = ab;
    dst[3] = SkScalarInterp(ab, bc, t);
    dst[6] = bc;
}

static void ratquad_mapTo3D(const SkPoint src[3], SkScalar w, SkPoint3 dst[3]) {
    dst[0].set(src[0].fX * 1, src[0].fY * 1, 1);
    dst[1].set(src[1].fX * w, src[1].fY * w, w);
    dst[2].set(src[2].fX * 1, src[2].fY * 1, 1);
}

static SkPoint project_down(const SkPoint3& src) {
    return {src.fX / src.fZ, src.fY / src.fZ};
}

// return false if infinity or NaN is generated; caller must check
bool SkConic::chopAt(SkScalar t, SkConic dst[2]) const {
    SkPoint3 tmp[3], tmp2[3];

    ratquad_mapTo3D(fPts, fW, tmp);

    p3d_interp(&tmp[0].fX, &tmp2[0].fX, t);
    p3d_interp(&tmp[0].fY, &tmp2[0].fY, t);
    p3d_interp(&tmp[0].fZ, &tmp2[0].fZ, t);

    dst[0].fPts[0] = fPts[0];
    dst[0].fPts[1] = project_down(tmp2[0]);
    dst[0].fPts[2] = project_down(tmp2[1]); dst[1].fPts[0] = dst[0].fPts[2];
    dst[1].fPts[1] = project_down(tmp2[2]);
    dst[1].fPts[2] = fPts[2];

    // to put in "standard form", where w0 and w2 are both 1, we compute the
    // new w1 as sqrt(w1*w1/w0*w2)
    // or
    // w1 /= sqrt(w0*w2)
    //
    // However, in our case, we know that for dst[0]:
    //     w0 == 1, and for dst[1], w2 == 1
    //
    SkScalar root = SkScalarSqrt(tmp2[1].fZ);
    dst[0].fW = tmp2[0].fZ / root;
    dst[1].fW = tmp2[2].fZ / root;
    SkASSERT(sizeof(dst[0]) == sizeof(SkScalar) * 7);
    SkASSERT(0 == offsetof(SkConic, fPts[0].fX));
    return SkScalarsAreFinite(&dst[0].fPts[0].fX, 7 * 2);
}

void SkConic::chopAt(SkScalar t1, SkScalar t2, SkConic* dst) const {
    if (0 == t1 || 1 == t2) {
        if (0 == t1 && 1 == t2) {
            *dst = *this;
            return;
        } else {
            SkConic pair[2];
            if (this->chopAt(t1 ? t1 : t2, pair)) {
                *dst = pair[SkToBool(t1)];
                return;
            }
        }
    }
    SkConicCoeff coeff(*this);
    float2 tt1(t1);
    float2 aXY = coeff.fNumer.eval(tt1);
    float2 aZZ = coeff.fDenom.eval(tt1);
    float2 midTT((t1 + t2) / 2);
    float2 dXY = coeff.fNumer.eval(midTT);
    float2 dZZ = coeff.fDenom.eval(midTT);
    float2 tt2(t2);
    float2 cXY = coeff.fNumer.eval(tt2);
    float2 cZZ = coeff.fDenom.eval(tt2);
    float2 bXY = times_2(dXY) - (aXY + cXY) * 0.5f;
    float2 bZZ = times_2(dZZ) - (aZZ + cZZ) * 0.5f;
    dst->fPts[0] = to_point(aXY / aZZ);
    dst->fPts[1] = to_point(bXY / bZZ);
    dst->fPts[2] = to_point(cXY / cZZ);
    float2 ww = bZZ / sqrt(aZZ * cZZ);
    dst->fW = ww[0];
}

SkPoint SkConic::evalAt(SkScalar t) const {
    return to_point(SkConicCoeff(*this).eval(t));
}

SkVector SkConic::evalTangentAt(SkScalar t) const {
    // The derivative equation returns a zero tangent vector when t is 0 or 1,
    // and the control point is equal to the end point.
    // In this case, use the conic endpoints to compute the tangent.
    if ((t == 0 && fPts[0] == fPts[1]) || (t == 1 && fPts[1] == fPts[2])) {
        return fPts[2] - fPts[0];
    }
    float2 p0 = from_point(fPts[0]);
    float2 p1 = from_point(fPts[1]);
    float2 p2 = from_point(fPts[2]);
    float2 ww(fW);

    float2 p20 = p2 - p0;
    float2 p10 = p1 - p0;

    float2 C = ww * p10;
    float2 A = ww * p20 - p20;
    float2 B = p20 - C - C;

    return to_vector(SkQuadCoeff(A, B, C).eval(t));
}

void SkConic::evalAt(SkScalar t, SkPoint* pt, SkVector* tangent) const {
    SkASSERT(t >= 0 && t <= SK_Scalar1);

    if (pt) {
        *pt = this->evalAt(t);
    }
    if (tangent) {
        *tangent = this->evalTangentAt(t);
    }
}

static SkScalar subdivide_w_value(SkScalar w) {
    return SkScalarSqrt(SK_ScalarHalf + w * SK_ScalarHalf);
}

void SkConic::chop(SkConic * SK_RESTRICT dst) const {
    float2 scale = SkScalarInvert(SK_Scalar1 + fW);
    SkScalar newW = subdivide_w_value(fW);

    float2 p0 = from_point(fPts[0]);
    float2 p1 = from_point(fPts[1]);
    float2 p2 = from_point(fPts[2]);
    float2 ww(fW);

    float2 wp1 = ww * p1;
    float2 m = (p0 + times_2(wp1) + p2) * scale * 0.5f;
    SkPoint mPt = to_point(m);
    if (!mPt.isFinite()) {
        double w_d = fW;
        double w_2 = w_d * 2;
        double scale_half = 1 / (1 + w_d) * 0.5;
        mPt.fX = SkDoubleToScalar((fPts[0].fX + w_2 * fPts[1].fX + fPts[2].fX) * scale_half);
        mPt.fY = SkDoubleToScalar((fPts[0].fY + w_2 * fPts[1].fY + fPts[2].fY) * scale_half);
    }
    dst[0].fPts[0] = fPts[0];
    dst[0].fPts[1] = to_point((p0 + wp1) * scale);
    dst[0].fPts[2] = dst[1].fPts[0] = mPt;
    dst[1].fPts[1] = to_point((wp1 + p2) * scale);
    dst[1].fPts[2] = fPts[2];

    dst[0].fW = dst[1].fW = newW;
}

/*
 *  "High order approximation of conic sections by quadratic splines"
 *      by Michael Floater, 1993
 */
#define AS_QUAD_ERROR_SETUP                                         \
    SkScalar a = fW - 1;                                            \
    SkScalar k = a / (4 * (2 + a));                                 \
    SkScalar x = k * (fPts[0].fX - 2 * fPts[1].fX + fPts[2].fX);    \
    SkScalar y = k * (fPts[0].fY - 2 * fPts[1].fY + fPts[2].fY);

void SkConic::computeAsQuadError(SkVector* err) const {
    AS_QUAD_ERROR_SETUP
    err->set(x, y);
}

bool SkConic::asQuadTol(SkScalar tol) const {
    AS_QUAD_ERROR_SETUP
    return (x * x + y * y) <= tol * tol;
}

// Limit the number of suggested quads to approximate a conic
#define kMaxConicToQuadPOW2     5

int SkConic::computeQuadPOW2(SkScalar tol) const {
    if (tol < 0 || !SkScalarIsFinite(tol) || !SkPointPriv::AreFinite(fPts, 3)) {
        return 0;
    }

    AS_QUAD_ERROR_SETUP

    SkScalar error = SkScalarSqrt(x * x + y * y);
    int pow2;
    for (pow2 = 0; pow2 < kMaxConicToQuadPOW2; ++pow2) {
        if (error <= tol) {
            break;
        }
        error *= 0.25f;
    }
    // float version -- using ceil gives the same results as the above.
    if ((false)) {
        SkScalar err = SkScalarSqrt(x * x + y * y);
        if (err <= tol) {
            return 0;
        }
        SkScalar tol2 = tol * tol;
        if (tol2 == 0) {
            return kMaxConicToQuadPOW2;
        }
        SkScalar fpow2 = SkScalarLog2((x * x + y * y) / tol2) * 0.25f;
        int altPow2 = SkScalarCeilToInt(fpow2);
        if (altPow2 != pow2) {
            SkDebugf("pow2 %d altPow2 %d fbits %g err %g tol %g\n", pow2, altPow2, fpow2, err, tol);
        }
        pow2 = altPow2;
    }
    return pow2;
}

// This was originally developed and tested for pathops: see SkOpTypes.h
// returns true if (a <= b <= c) || (a >= b >= c)
static bool between(SkScalar a, SkScalar b, SkScalar c) {
    return (a - b) * (c - b) <= 0;
}

static SkPoint* subdivide(const SkConic& src, SkPoint pts[], int level) {
    SkASSERT(level >= 0);

    if (0 == level) {
        memcpy(pts, &src.fPts[1], 2 * sizeof(SkPoint));
        return pts + 2;
    } else {
        SkConic dst[2];
        src.chop(dst);
        const SkScalar startY = src.fPts[0].fY;
        SkScalar endY = src.fPts[2].fY;
        if (between(startY, src.fPts[1].fY, endY)) {
            // If the input is monotonic and the output is not, the scan converter hangs.
            // Ensure that the chopped conics maintain their y-order.
            SkScalar midY = dst[0].fPts[2].fY;
            if (!between(startY, midY, endY)) {
                // If the computed midpoint is outside the ends, move it to the closer one.
                SkScalar closerY = SkTAbs(midY - startY) < SkTAbs(midY - endY) ? startY : endY;
                dst[0].fPts[2].fY = dst[1].fPts[0].fY = closerY;
            }
            if (!between(startY, dst[0].fPts[1].fY, dst[0].fPts[2].fY)) {
                // If the 1st control is not between the start and end, put it at the start.
                // This also reduces the quad to a line.
                dst[0].fPts[1].fY = startY;
            }
            if (!between(dst[1].fPts[0].fY, dst[1].fPts[1].fY, endY)) {
                // If the 2nd control is not between the start and end, put it at the end.
                // This also reduces the quad to a line.
                dst[1].fPts[1].fY = endY;
            }
            // Verify that all five points are in order.
            SkASSERT(between(startY, dst[0].fPts[1].fY, dst[0].fPts[2].fY));
            SkASSERT(between(dst[0].fPts[1].fY, dst[0].fPts[2].fY, dst[1].fPts[1].fY));
            SkASSERT(between(dst[0].fPts[2].fY, dst[1].fPts[1].fY, endY));
        }
        --level;
        pts = subdivide(dst[0], pts, level);
        return subdivide(dst[1], pts, level);
    }
}

int SkConic::chopIntoQuadsPOW2(SkPoint pts[], int pow2) const {
    SkASSERT(pow2 >= 0);
    *pts = fPts[0];
    SkDEBUGCODE(SkPoint* endPts);
    if (pow2 == kMaxConicToQuadPOW2) {  // If an extreme weight generates many quads ...
        SkConic dst[2];
        this->chop(dst);
        // check to see if the first chop generates a pair of lines
        if (SkPointPriv::EqualsWithinTolerance(dst[0].fPts[1], dst[0].fPts[2]) &&
                SkPointPriv::EqualsWithinTolerance(dst[1].fPts[0], dst[1].fPts[1])) {
            pts[1] = pts[2] = pts[3] = dst[0].fPts[1];  // set ctrl == end to make lines
            pts[4] = dst[1].fPts[2];
            pow2 = 1;
            SkDEBUGCODE(endPts = &pts[5]);
            goto commonFinitePtCheck;
        }
    }
    SkDEBUGCODE(endPts = ) subdivide(*this, pts + 1, pow2);
commonFinitePtCheck:
    const int quadCount = 1 << pow2;
    const int ptCount = 2 * quadCount + 1;
    SkASSERT(endPts - pts == ptCount);
    if (!SkPointPriv::AreFinite(pts, ptCount)) {
        // if we generated a non-finite, pin ourselves to the middle of the hull,
        // as our first and last are already on the first/last pts of the hull.
        for (int i = 1; i < ptCount - 1; ++i) {
            pts[i] = fPts[1];
        }
    }
    return 1 << pow2;
}

float SkConic::findMidTangent() const {
    // Tangents point in the direction of increasing T, so tan0 and -tan1 both point toward the
    // midtangent. The bisector of tan0 and -tan1 is orthogonal to the midtangent:
    //
    //     bisector dot midtangent = 0
    //
    SkVector tan0 = fPts[1] - fPts[0];
    SkVector tan1 = fPts[2] - fPts[1];
    SkVector bisector = SkFindBisector(tan0, -tan1);

    // Start by finding the tangent function's power basis coefficients. These define a tangent
    // direction (scaled by some uniform value) as:
    //                                                |T^2|
    //     Tangent_Direction(T) = dx,dy = |A  B  C| * |T  |
    //                                    |.  .  .|   |1  |
    //
    // The derivative of a conic has a cumbersome order-4 denominator. However, this isn't necessary
    // if we are only interested in a vector in the same *direction* as a given tangent line. Since
    // the denominator scales dx and dy uniformly, we can throw it out completely after evaluating
    // the derivative with the standard quotient rule. This leaves us with a simpler quadratic
    // function that we use to find a tangent.
    SkVector A = (fPts[2] - fPts[0]) * (fW - 1);
    SkVector B = (fPts[2] - fPts[0]) - (fPts[1] - fPts[0]) * (fW*2);
    SkVector C = (fPts[1] - fPts[0]) * fW;

    // Now solve for "bisector dot midtangent = 0":
    //
    //                            |T^2|
    //     bisector * |A  B  C| * |T  | = 0
    //                |.  .  .|   |1  |
    //
    float a = bisector.dot(A);
    float b = bisector.dot(B);
    float c = bisector.dot(C);
    return solve_quadratic_equation_for_midtangent(a, b, c);
}

bool SkConic::findXExtrema(SkScalar* t) const {
    return conic_find_extrema(&fPts[0].fX, fW, t);
}

bool SkConic::findYExtrema(SkScalar* t) const {
    return conic_find_extrema(&fPts[0].fY, fW, t);
}

bool SkConic::chopAtXExtrema(SkConic dst[2]) const {
    SkScalar t;
    if (this->findXExtrema(&t)) {
        if (!this->chopAt(t, dst)) {
            // if chop can't return finite values, don't chop
            return false;
        }
        // now clean-up the middle, since we know t was meant to be at
        // an X-extrema
        SkScalar value = dst[0].fPts[2].fX;
        dst[0].fPts[1].fX = value;
        dst[1].fPts[0].fX = value;
        dst[1].fPts[1].fX = value;
        return true;
    }
    return false;
}

bool SkConic::chopAtYExtrema(SkConic dst[2]) const {
    SkScalar t;
    if (this->findYExtrema(&t)) {
        if (!this->chopAt(t, dst)) {
            // if chop can't return finite values, don't chop
            return false;
        }
        // now clean-up the middle, since we know t was meant to be at
        // an Y-extrema
        SkScalar value = dst[0].fPts[2].fY;
        dst[0].fPts[1].fY = value;
        dst[1].fPts[0].fY = value;
        dst[1].fPts[1].fY = value;
        return true;
    }
    return false;
}

void SkConic::computeTightBounds(SkRect* bounds) const {
    SkPoint pts[4];
    pts[0] = fPts[0];
    pts[1] = fPts[2];
    int count = 2;

    SkScalar t;
    if (this->findXExtrema(&t)) {
        this->evalAt(t, &pts[count++]);
    }
    if (this->findYExtrema(&t)) {
        this->evalAt(t, &pts[count++]);
    }
    bounds->setBounds(pts, count);
}

void SkConic::computeFastBounds(SkRect* bounds) const {
    bounds->setBounds(fPts, 3);
}

#if 0  // unimplemented
bool SkConic::findMaxCurvature(SkScalar* t) const {
    // TODO: Implement me
    return false;
}
#endif

SkScalar SkConic::TransformW(const SkPoint pts[], SkScalar w, const SkMatrix& matrix) {
    if (!matrix.hasPerspective()) {
        return w;
    }

    SkPoint3 src[3], dst[3];

    ratquad_mapTo3D(pts, w, src);

    matrix.mapHomogeneousPoints(dst, src, 3);

    // w' = sqrt(w1*w1/w0*w2)
    // use doubles temporarily, to handle small numer/denom
    double w0 = dst[0].fZ;
    double w1 = dst[1].fZ;
    double w2 = dst[2].fZ;
    return sk_double_to_float(sqrt(sk_ieee_double_divide(w1 * w1, w0 * w2)));
}

int SkConic::BuildUnitArc(const SkVector& uStart, const SkVector& uStop, SkRotationDirection dir,
                          const SkMatrix* userMatrix, SkConic dst[kMaxConicsForArc]) {
    // rotate by x,y so that uStart is (1.0)
    SkScalar x = SkPoint::DotProduct(uStart, uStop);
    SkScalar y = SkPoint::CrossProduct(uStart, uStop);

    SkScalar absY = SkScalarAbs(y);

    // check for (effectively) coincident vectors
    // this can happen if our angle is nearly 0 or nearly 180 (y == 0)
    // ... we use the dot-prod to distinguish between 0 and 180 (x > 0)
    if (absY <= SK_ScalarNearlyZero && x > 0 && ((y >= 0 && kCW_SkRotationDirection == dir) ||
                                                 (y <= 0 && kCCW_SkRotationDirection == dir))) {
        return 0;
    }

    if (dir == kCCW_SkRotationDirection) {
        y = -y;
    }

    // We decide to use 1-conic per quadrant of a circle. What quadrant does [xy] lie in?
    //      0 == [0  .. 90)
    //      1 == [90 ..180)
    //      2 == [180..270)
    //      3 == [270..360)
    //
    int quadrant = 0;
    if (0 == y) {
        quadrant = 2;        // 180
        SkASSERT(SkScalarAbs(x + SK_Scalar1) <= SK_ScalarNearlyZero);
    } else if (0 == x) {
        SkASSERT(absY - SK_Scalar1 <= SK_ScalarNearlyZero);
        quadrant = y > 0 ? 1 : 3; // 90 : 270
    } else {
        if (y < 0) {
            quadrant += 2;
        }
        if ((x < 0) != (y < 0)) {
            quadrant += 1;
        }
    }

    const SkPoint quadrantPts[] = {
        { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 }, { -1, -1 }, { 0, -1 }, { 1, -1 }
    };
    const SkScalar quadrantWeight = SK_ScalarRoot2Over2;

    int conicCount = quadrant;
    for (int i = 0; i < conicCount; ++i) {
        dst[i].set(&quadrantPts[i * 2], quadrantWeight);
    }

    // Now compute any remaing (sub-90-degree) arc for the last conic
    const SkPoint finalP = { x, y };
    const SkPoint& lastQ = quadrantPts[quadrant * 2];  // will already be a unit-vector
    const SkScalar dot = SkVector::DotProduct(lastQ, finalP);
    SkASSERT(0 <= dot && dot <= SK_Scalar1 + SK_ScalarNearlyZero);

    if (dot < 1) {
        SkVector offCurve = { lastQ.x() + x, lastQ.y() + y };
        // compute the bisector vector, and then rescale to be the off-curve point.
        // we compute its length from cos(theta/2) = length / 1, using half-angle identity we get
        // length = sqrt(2 / (1 + cos(theta)). We already have cos() when to computed the dot.
        // This is nice, since our computed weight is cos(theta/2) as well!
        //
        const SkScalar cosThetaOver2 = SkScalarSqrt((1 + dot) / 2);
        offCurve.setLength(SkScalarInvert(cosThetaOver2));
        if (!SkPointPriv::EqualsWithinTolerance(lastQ, offCurve)) {
            dst[conicCount].set(lastQ, offCurve, finalP, cosThetaOver2);
            conicCount += 1;
        }
    }

    // now handle counter-clockwise and the initial unitStart rotation
    SkMatrix    matrix;
    matrix.setSinCos(uStart.fY, uStart.fX);
    if (dir == kCCW_SkRotationDirection) {
        matrix.preScale(SK_Scalar1, -SK_Scalar1);
    }
    if (userMatrix) {
        matrix.postConcat(*userMatrix);
    }
    for (int i = 0; i < conicCount; ++i) {
        matrix.mapPoints(dst[i].fPts, 3);
    }
    return conicCount;
}
