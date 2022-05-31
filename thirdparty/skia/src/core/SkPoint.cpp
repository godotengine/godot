/*
 * Copyright 2008 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "src/core/SkMathPriv.h"
#include "src/core/SkPointPriv.h"

///////////////////////////////////////////////////////////////////////////////

void SkPoint::scale(SkScalar scale, SkPoint* dst) const {
    SkASSERT(dst);
    dst->set(fX * scale, fY * scale);
}

bool SkPoint::normalize() {
    return this->setLength(fX, fY, SK_Scalar1);
}

bool SkPoint::setNormalize(SkScalar x, SkScalar y) {
    return this->setLength(x, y, SK_Scalar1);
}

bool SkPoint::setLength(SkScalar length) {
    return this->setLength(fX, fY, length);
}

/*
 *  We have to worry about 2 tricky conditions:
 *  1. underflow of mag2 (compared against nearlyzero^2)
 *  2. overflow of mag2 (compared w/ isfinite)
 *
 *  If we underflow, we return false. If we overflow, we compute again using
 *  doubles, which is much slower (3x in a desktop test) but will not overflow.
 */
template <bool use_rsqrt> bool set_point_length(SkPoint* pt, float x, float y, float length,
                                                float* orig_length = nullptr) {
    SkASSERT(!use_rsqrt || (orig_length == nullptr));

    // our mag2 step overflowed to infinity, so use doubles instead.
    // much slower, but needed when x or y are very large, other wise we
    // divide by inf. and return (0,0) vector.
    double xx = x;
    double yy = y;
    double dmag = sqrt(xx * xx + yy * yy);
    double dscale = sk_ieee_double_divide(length, dmag);
    x *= dscale;
    y *= dscale;
    // check if we're not finite, or we're zero-length
    if (!sk_float_isfinite(x) || !sk_float_isfinite(y) || (x == 0 && y == 0)) {
        pt->set(0, 0);
        return false;
    }
    float mag = 0;
    if (orig_length) {
        mag = sk_double_to_float(dmag);
    }
    pt->set(x, y);
    if (orig_length) {
        *orig_length = mag;
    }
    return true;
}

SkScalar SkPoint::Normalize(SkPoint* pt) {
    float mag;
    if (set_point_length<false>(pt, pt->fX, pt->fY, 1.0f, &mag)) {
        return mag;
    }
    return 0;
}

SkScalar SkPoint::Length(SkScalar dx, SkScalar dy) {
    float mag2 = dx * dx + dy * dy;
    if (SkScalarIsFinite(mag2)) {
        return sk_float_sqrt(mag2);
    } else {
        double xx = dx;
        double yy = dy;
        return sk_double_to_float(sqrt(xx * xx + yy * yy));
    }
}

bool SkPoint::setLength(float x, float y, float length) {
    return set_point_length<false>(this, x, y, length);
}

bool SkPointPriv::SetLengthFast(SkPoint* pt, float length) {
    return set_point_length<true>(pt, pt->fX, pt->fY, length);
}


///////////////////////////////////////////////////////////////////////////////

SkScalar SkPointPriv::DistanceToLineBetweenSqd(const SkPoint& pt, const SkPoint& a,
                                               const SkPoint& b,
                                               Side* side) {

    SkVector u = b - a;
    SkVector v = pt - a;

    SkScalar uLengthSqd = LengthSqd(u);
    SkScalar det = u.cross(v);
    if (side) {
        SkASSERT(-1 == kLeft_Side &&
                  0 == kOn_Side &&
                  1 == kRight_Side);
        *side = (Side) SkScalarSignAsInt(det);
    }
    SkScalar temp = sk_ieee_float_divide(det, uLengthSqd);
    temp *= det;
    // It's possible we have a degenerate line vector, or we're so far away it looks degenerate
    // In this case, return squared distance to point A.
    if (!SkScalarIsFinite(temp)) {
        return LengthSqd(v);
    }
    return temp;
}

SkScalar SkPointPriv::DistanceToLineSegmentBetweenSqd(const SkPoint& pt, const SkPoint& a,
                                                      const SkPoint& b) {
    // See comments to distanceToLineBetweenSqd. If the projection of c onto
    // u is between a and b then this returns the same result as that
    // function. Otherwise, it returns the distance to the closer of a and
    // b. Let the projection of v onto u be v'.  There are three cases:
    //    1. v' points opposite to u. c is not between a and b and is closer
    //       to a than b.
    //    2. v' points along u and has magnitude less than y. c is between
    //       a and b and the distance to the segment is the same as distance
    //       to the line ab.
    //    3. v' points along u and has greater magnitude than u. c is not
    //       not between a and b and is closer to b than a.
    // v' = (u dot v) * u / |u|. So if (u dot v)/|u| is less than zero we're
    // in case 1. If (u dot v)/|u| is > |u| we are in case 3. Otherwise
    // we're in case 2. We actually compare (u dot v) to 0 and |u|^2 to
    // avoid a sqrt to compute |u|.

    SkVector u = b - a;
    SkVector v = pt - a;

    SkScalar uLengthSqd = LengthSqd(u);
    SkScalar uDotV = SkPoint::DotProduct(u, v);

    // closest point is point A
    if (uDotV <= 0) {
        return LengthSqd(v);
    // closest point is point B
    } else if (uDotV > uLengthSqd) {
        return DistanceToSqd(b, pt);
    // closest point is inside segment
    } else {
        SkScalar det = u.cross(v);
        SkScalar temp = sk_ieee_float_divide(det, uLengthSqd);
        temp *= det;
        // It's possible we have a degenerate segment, or we're so far away it looks degenerate
        // In this case, return squared distance to point A.
        if (!SkScalarIsFinite(temp)) {
            return LengthSqd(v);
        }
        return temp;
    }
}
