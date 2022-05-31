/*
 * Copyright 2018 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkRRectPriv_DEFINED
#define SkRRectPriv_DEFINED

#include "include/core/SkRRect.h"

class SkRBuffer;
class SkWBuffer;

class SkRRectPriv {
public:
    static bool IsCircle(const SkRRect& rr) {
        return rr.isOval() && SkScalarNearlyEqual(rr.fRadii[0].fX, rr.fRadii[0].fY);
    }

    static SkVector GetSimpleRadii(const SkRRect& rr) {
        SkASSERT(!rr.isComplex());
        return rr.fRadii[0];
    }

    static bool IsSimpleCircular(const SkRRect& rr) {
        return rr.isSimple() && SkScalarNearlyEqual(rr.fRadii[0].fX, rr.fRadii[0].fY);
    }

    // Looser version of IsSimpleCircular, where the x & y values of the radii
    // only have to be nearly equal instead of strictly equal.
    static bool IsNearlySimpleCircular(const SkRRect& rr, SkScalar tolerance = SK_ScalarNearlyZero);

    static bool EqualRadii(const SkRRect& rr) {
        return rr.isRect() || SkRRectPriv::IsCircle(rr)  || SkRRectPriv::IsSimpleCircular(rr);
    }

    static const SkVector* GetRadiiArray(const SkRRect& rr) { return rr.fRadii; }

    static bool AllCornersCircular(const SkRRect& rr, SkScalar tolerance = SK_ScalarNearlyZero);

// -- GODOT start --
    //static bool ReadFromBuffer(SkRBuffer* buffer, SkRRect* rr);

    //static void WriteToBuffer(const SkRRect& rr, SkWBuffer* buffer);
// -- GODOT end --

    // Test if a point is in the rrect, if it were a closed set.
    static bool ContainsPoint(const SkRRect& rr, const SkPoint& p) {
        return rr.getBounds().contains(p.fX, p.fY) && rr.checkCornerContainment(p.fX, p.fY);
    }

    // Compute an approximate largest inscribed bounding box of the rounded rect. For empty,
    // rect, oval, and simple types this will be the largest inscribed rectangle. Otherwise it may
    // not be the global maximum, but will be non-empty, touch at least one edge and be contained
    // in the round rect.
    static SkRect InnerBounds(const SkRRect& rr);

    // Attempt to compute the intersection of two round rects. The intersection is not necessarily
    // a round rect. This returns intersections only when the shape is representable as a new
    // round rect (or rect). Empty is returned if 'a' and 'b' do not intersect or if the
    // intersection is too complicated. This is conservative, it may not always detect that an
    // intersection could be represented as a round rect. However, when it does return a round rect
    // that intersection will be exact (i.e. it is NOT just a subset of the actual intersection).
    static SkRRect ConservativeIntersect(const SkRRect& a, const SkRRect& b);
};

#endif
