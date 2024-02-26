/*
 * Copyright 2019 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkPathMakers_DEFINED
#define SkPathMakers_DEFINED

#include "include/core/SkPathTypes.h"
#include "include/core/SkPoint.h"
#include "include/core/SkRRect.h"

template <unsigned N> class SkPath_PointIterator {
public:
    SkPath_PointIterator(SkPathDirection dir, unsigned startIndex)
    : fCurrent(startIndex % N)
    , fAdvance(dir == SkPathDirection::kCW ? 1 : N - 1) { }

    const SkPoint& current() const {
        SkASSERT(fCurrent < N);
        return fPts[fCurrent];
    }

    const SkPoint& next() {
        fCurrent = (fCurrent + fAdvance) % N;
        return this->current();
    }

    protected:
    SkPoint fPts[N];

    private:
    unsigned fCurrent;
    unsigned fAdvance;
};

class SkPath_RectPointIterator : public SkPath_PointIterator<4> {
public:
    SkPath_RectPointIterator(const SkRect& rect, SkPathDirection dir, unsigned startIndex)
        : SkPath_PointIterator(dir, startIndex) {

        fPts[0] = SkPoint::Make(rect.fLeft, rect.fTop);
        fPts[1] = SkPoint::Make(rect.fRight, rect.fTop);
        fPts[2] = SkPoint::Make(rect.fRight, rect.fBottom);
        fPts[3] = SkPoint::Make(rect.fLeft, rect.fBottom);
    }
};

class SkPath_OvalPointIterator : public SkPath_PointIterator<4> {
public:
    SkPath_OvalPointIterator(const SkRect& oval, SkPathDirection dir, unsigned startIndex)
        : SkPath_PointIterator(dir, startIndex) {

        const SkScalar cx = oval.centerX();
        const SkScalar cy = oval.centerY();

        fPts[0] = SkPoint::Make(cx, oval.fTop);
        fPts[1] = SkPoint::Make(oval.fRight, cy);
        fPts[2] = SkPoint::Make(cx, oval.fBottom);
        fPts[3] = SkPoint::Make(oval.fLeft, cy);
    }
};

class SkPath_RRectPointIterator : public SkPath_PointIterator<8> {
public:
    SkPath_RRectPointIterator(const SkRRect& rrect, SkPathDirection dir, unsigned startIndex)
        : SkPath_PointIterator(dir, startIndex) {

        const SkRect& bounds = rrect.getBounds();
        const SkScalar L = bounds.fLeft;
        const SkScalar T = bounds.fTop;
        const SkScalar R = bounds.fRight;
        const SkScalar B = bounds.fBottom;

        fPts[0] = SkPoint::Make(L + rrect.radii(SkRRect::kUpperLeft_Corner).fX, T);
        fPts[1] = SkPoint::Make(R - rrect.radii(SkRRect::kUpperRight_Corner).fX, T);
        fPts[2] = SkPoint::Make(R, T + rrect.radii(SkRRect::kUpperRight_Corner).fY);
        fPts[3] = SkPoint::Make(R, B - rrect.radii(SkRRect::kLowerRight_Corner).fY);
        fPts[4] = SkPoint::Make(R - rrect.radii(SkRRect::kLowerRight_Corner).fX, B);
        fPts[5] = SkPoint::Make(L + rrect.radii(SkRRect::kLowerLeft_Corner).fX, B);
        fPts[6] = SkPoint::Make(L, B - rrect.radii(SkRRect::kLowerLeft_Corner).fY);
        fPts[7] = SkPoint::Make(L, T + rrect.radii(SkRRect::kUpperLeft_Corner).fY);
    }
};

#endif
