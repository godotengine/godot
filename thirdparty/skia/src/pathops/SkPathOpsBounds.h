/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#ifndef SkPathOpBounds_DEFINED
#define SkPathOpBounds_DEFINED

#include "include/core/SkRect.h"
#include "src/pathops/SkPathOpsRect.h"

// SkPathOpsBounds, unlike SkRect, does not consider a line to be empty.
struct SkPathOpsBounds : public SkRect {
    static bool Intersects(const SkPathOpsBounds& a, const SkPathOpsBounds& b) {
        return AlmostLessOrEqualUlps(a.fLeft, b.fRight)
                && AlmostLessOrEqualUlps(b.fLeft, a.fRight)
                && AlmostLessOrEqualUlps(a.fTop, b.fBottom)
                && AlmostLessOrEqualUlps(b.fTop, a.fBottom);
    }

   // Note that add(), unlike SkRect::join() or SkRect::growToInclude()
   // does not treat the bounds of horizontal and vertical lines as
   // empty rectangles.
    void add(SkScalar left, SkScalar top, SkScalar right, SkScalar bottom) {
        if (left < fLeft) fLeft = left;
        if (top < fTop) fTop = top;
        if (right > fRight) fRight = right;
        if (bottom > fBottom) fBottom = bottom;
    }

    void add(const SkPathOpsBounds& toAdd) {
        add(toAdd.fLeft, toAdd.fTop, toAdd.fRight, toAdd.fBottom);
    }

    void add(const SkPoint& pt) {
        if (pt.fX < fLeft) fLeft = pt.fX;
        if (pt.fY < fTop) fTop = pt.fY;
        if (pt.fX > fRight) fRight = pt.fX;
        if (pt.fY > fBottom) fBottom = pt.fY;
    }

    void add(const SkDPoint& pt) {
        if (pt.fX < fLeft) fLeft = SkDoubleToScalar(pt.fX);
        if (pt.fY < fTop) fTop = SkDoubleToScalar(pt.fY);
        if (pt.fX > fRight) fRight = SkDoubleToScalar(pt.fX);
        if (pt.fY > fBottom) fBottom = SkDoubleToScalar(pt.fY);
    }

    bool almostContains(const SkPoint& pt) const {
        return AlmostLessOrEqualUlps(fLeft, pt.fX)
                && AlmostLessOrEqualUlps(pt.fX, fRight)
                && AlmostLessOrEqualUlps(fTop, pt.fY)
                && AlmostLessOrEqualUlps(pt.fY, fBottom);
    }

    bool contains(const SkPoint& pt) const {
        return fLeft <= pt.fX && fTop <= pt.fY &&
               fRight >= pt.fX && fBottom >= pt.fY;
    }

    using INHERITED = SkRect;
};

#endif
