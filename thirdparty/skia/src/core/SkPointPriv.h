/*
 * Copyright 2006 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkPointPriv_DEFINED
#define SkPointPriv_DEFINED

#include "include/core/SkPoint.h"
#include "include/core/SkRect.h"

class SkPointPriv {
public:
    enum Side {
        kLeft_Side  = -1,
        kOn_Side    =  0,
        kRight_Side =  1,
    };

    static bool AreFinite(const SkPoint array[], int count) {
        return SkScalarsAreFinite(&array[0].fX, count << 1);
    }

    static const SkScalar* AsScalars(const SkPoint& pt) { return &pt.fX; }

    static bool CanNormalize(SkScalar dx, SkScalar dy) {
        return SkScalarsAreFinite(dx, dy) && (dx || dy);
    }

    static SkScalar DistanceToLineBetweenSqd(const SkPoint& pt, const SkPoint& a,
                                             const SkPoint& b, Side* side = nullptr);

    static SkScalar DistanceToLineBetween(const SkPoint& pt, const SkPoint& a,
                                          const SkPoint& b, Side* side = nullptr) {
        return SkScalarSqrt(DistanceToLineBetweenSqd(pt, a, b, side));
    }

    static SkScalar DistanceToLineSegmentBetweenSqd(const SkPoint& pt, const SkPoint& a,
                                                   const SkPoint& b);

    static SkScalar DistanceToLineSegmentBetween(const SkPoint& pt, const SkPoint& a,
                                                 const SkPoint& b) {
        return SkScalarSqrt(DistanceToLineSegmentBetweenSqd(pt, a, b));
    }

    static SkScalar DistanceToSqd(const SkPoint& pt, const SkPoint& a) {
        SkScalar dx = pt.fX - a.fX;
        SkScalar dy = pt.fY - a.fY;
        return dx * dx + dy * dy;
    }

    static bool EqualsWithinTolerance(const SkPoint& p1, const SkPoint& p2) {
        return !CanNormalize(p1.fX - p2.fX, p1.fY - p2.fY);
    }

    static bool EqualsWithinTolerance(const SkPoint& pt, const SkPoint& p, SkScalar tol) {
        return SkScalarNearlyZero(pt.fX - p.fX, tol)
               && SkScalarNearlyZero(pt.fY - p.fY, tol);
    }

    static SkScalar LengthSqd(const SkPoint& pt) {
        return SkPoint::DotProduct(pt, pt);
    }

    static void Negate(SkIPoint& pt) {
        pt.fX = -pt.fX;
        pt.fY = -pt.fY;
    }

    static void RotateCCW(const SkPoint& src, SkPoint* dst) {
        // use a tmp in case src == dst
        SkScalar tmp = src.fX;
        dst->fX = src.fY;
        dst->fY = -tmp;
    }

    static void RotateCCW(SkPoint* pt) {
        RotateCCW(*pt, pt);
    }

    static void RotateCW(const SkPoint& src, SkPoint* dst) {
        // use a tmp in case src == dst
        SkScalar tmp = src.fX;
        dst->fX = -src.fY;
        dst->fY = tmp;
    }

    static void RotateCW(SkPoint* pt) {
        RotateCW(*pt, pt);
    }

    static bool SetLengthFast(SkPoint* pt, float length);

    static SkPoint MakeOrthog(const SkPoint& vec, Side side = kLeft_Side) {
        SkASSERT(side == kRight_Side || side == kLeft_Side);
        return (side == kRight_Side) ? SkPoint{-vec.fY, vec.fX} : SkPoint{vec.fY, -vec.fX};
    }

    // counter-clockwise fan
    static void SetRectFan(SkPoint v[], SkScalar l, SkScalar t, SkScalar r, SkScalar b,
            size_t stride) {
        SkASSERT(stride >= sizeof(SkPoint));

        ((SkPoint*)((intptr_t)v + 0 * stride))->set(l, t);
        ((SkPoint*)((intptr_t)v + 1 * stride))->set(l, b);
        ((SkPoint*)((intptr_t)v + 2 * stride))->set(r, b);
        ((SkPoint*)((intptr_t)v + 3 * stride))->set(r, t);
    }

    // tri strip with two counter-clockwise triangles
    static void SetRectTriStrip(SkPoint v[], SkScalar l, SkScalar t, SkScalar r, SkScalar b,
            size_t stride) {
        SkASSERT(stride >= sizeof(SkPoint));

        ((SkPoint*)((intptr_t)v + 0 * stride))->set(l, t);
        ((SkPoint*)((intptr_t)v + 1 * stride))->set(l, b);
        ((SkPoint*)((intptr_t)v + 2 * stride))->set(r, t);
        ((SkPoint*)((intptr_t)v + 3 * stride))->set(r, b);
    }
    static void SetRectTriStrip(SkPoint v[], const SkRect& rect, size_t stride) {
        SetRectTriStrip(v, rect.fLeft, rect.fTop, rect.fRight, rect.fBottom, stride);
    }
};

#endif
