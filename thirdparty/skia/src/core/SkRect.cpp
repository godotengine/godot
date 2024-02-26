/*
 * Copyright 2006 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "include/core/SkRect.h"

#include "include/private/SkMalloc.h"
#include "src/core/SkRectPriv.h"

bool SkIRect::intersect(const SkIRect& a, const SkIRect& b) {
    SkIRect tmp = {
        std::max(a.fLeft,   b.fLeft),
        std::max(a.fTop,    b.fTop),
        std::min(a.fRight,  b.fRight),
        std::min(a.fBottom, b.fBottom)
    };
    if (tmp.isEmpty()) {
        return false;
    }
    *this = tmp;
    return true;
}

void SkIRect::join(const SkIRect& r) {
    // do nothing if the params are empty
    if (r.fLeft >= r.fRight || r.fTop >= r.fBottom) {
        return;
    }

    // if we are empty, just assign
    if (fLeft >= fRight || fTop >= fBottom) {
        *this = r;
    } else {
        if (r.fLeft < fLeft)     fLeft = r.fLeft;
        if (r.fTop < fTop)       fTop = r.fTop;
        if (r.fRight > fRight)   fRight = r.fRight;
        if (r.fBottom > fBottom) fBottom = r.fBottom;
    }
}

/////////////////////////////////////////////////////////////////////////////

void SkRect::toQuad(SkPoint quad[4]) const {
    SkASSERT(quad);

    quad[0].set(fLeft, fTop);
    quad[1].set(fRight, fTop);
    quad[2].set(fRight, fBottom);
    quad[3].set(fLeft, fBottom);
}

#include "include/private/SkVx.h"

bool SkRect::setBoundsCheck(const SkPoint pts[], int count) {
    SkASSERT((pts && count > 0) || count == 0);

    if (count <= 0) {
        this->setEmpty();
        return true;
    }

    skvx::float4 min, max;
    if (count & 1) {
        min = max = skvx::float2::Load(pts).xyxy();
        pts   += 1;
        count -= 1;
    } else {
        min = max = skvx::float4::Load(pts);
        pts   += 2;
        count -= 2;
    }

    skvx::float4 accum = min * 0;
    while (count) {
        skvx::float4 xy = skvx::float4::Load(pts);
        accum = accum * xy;
        min = skvx::min(min, xy);
        max = skvx::max(max, xy);
        pts   += 2;
        count -= 2;
    }

    const bool all_finite = all(accum * 0 == 0);
    if (all_finite) {
        this->setLTRB(std::min(min[0], min[2]), std::min(min[1], min[3]),
                      std::max(max[0], max[2]), std::max(max[1], max[3]));
    } else {
        this->setEmpty();
    }
    return all_finite;
}

void SkRect::setBoundsNoCheck(const SkPoint pts[], int count) {
    if (!this->setBoundsCheck(pts, count)) {
        this->setLTRB(SK_ScalarNaN, SK_ScalarNaN, SK_ScalarNaN, SK_ScalarNaN);
    }
}

#define CHECK_INTERSECT(al, at, ar, ab, bl, bt, br, bb) \
    SkScalar L = std::max(al, bl);                   \
    SkScalar R = std::min(ar, br);                   \
    SkScalar T = std::max(at, bt);                   \
    SkScalar B = std::min(ab, bb);                   \
    do { if (!(L < R && T < B)) return false; } while (0)
    // do the !(opposite) check so we return false if either arg is NaN

bool SkRect::intersect(const SkRect& r) {
    CHECK_INTERSECT(r.fLeft, r.fTop, r.fRight, r.fBottom, fLeft, fTop, fRight, fBottom);
    this->setLTRB(L, T, R, B);
    return true;
}

bool SkRect::intersect(const SkRect& a, const SkRect& b) {
    CHECK_INTERSECT(a.fLeft, a.fTop, a.fRight, a.fBottom, b.fLeft, b.fTop, b.fRight, b.fBottom);
    this->setLTRB(L, T, R, B);
    return true;
}

void SkRect::join(const SkRect& r) {
    if (r.isEmpty()) {
        return;
    }

    if (this->isEmpty()) {
        *this = r;
    } else {
        fLeft   = std::min(fLeft, r.fLeft);
        fTop    = std::min(fTop, r.fTop);
        fRight  = std::max(fRight, r.fRight);
        fBottom = std::max(fBottom, r.fBottom);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////

// -- GODOT start --
/*
#include "include/core/SkString.h"
#include "src/core/SkStringUtils.h"

static const char* set_scalar(SkString* storage, SkScalar value, SkScalarAsStringType asType) {
    storage->reset();
    SkAppendScalar(storage, value, asType);
    return storage->c_str();
}

void SkRect::dump(bool asHex) const {
    SkScalarAsStringType asType = asHex ? kHex_SkScalarAsStringType : kDec_SkScalarAsStringType;

    SkString line;
    if (asHex) {
        SkString tmp;
        line.printf( "SkRect::MakeLTRB(%s, %f\n", set_scalar(&tmp, fLeft, asType), fLeft);
        line.appendf("                 %s, %f\n", set_scalar(&tmp, fTop, asType), fTop);
        line.appendf("                 %s, %f\n", set_scalar(&tmp, fRight, asType), fRight);
        line.appendf("                 %s  %f);", set_scalar(&tmp, fBottom, asType), fBottom);
    } else {
        SkString strL, strT, strR, strB;
        SkAppendScalarDec(&strL, fLeft);
        SkAppendScalarDec(&strT, fTop);
        SkAppendScalarDec(&strR, fRight);
        SkAppendScalarDec(&strB, fBottom);
        line.printf("SkRect::MakeLTRB(%s, %s, %s, %s);",
                    strL.c_str(), strT.c_str(), strR.c_str(), strB.c_str());
    }
    SkDebugf("%s\n", line.c_str());
}
*/
// -- GODOT end --

////////////////////////////////////////////////////////////////////////////////////////////////

template<typename R>
static bool subtract(const R& a, const R& b, R* out) {
    if (a.isEmpty() || b.isEmpty() || !R::Intersects(a, b)) {
        // Either already empty, or subtracting the empty rect, or there's no intersection, so
        // in all cases the answer is A.
        *out = a;
        return true;
    }

    // 4 rectangles to consider. If the edge in A is contained in B, the resulting difference can
    // be represented exactly as a rectangle. Otherwise the difference is the largest subrectangle
    // that is disjoint from B:
    // 1. Left part of A:   (A.left,  A.top,    B.left,  A.bottom)
    // 2. Right part of A:  (B.right, A.top,    A.right, A.bottom)
    // 3. Top part of A:    (A.left,  A.top,    A.right, B.top)
    // 4. Bottom part of A: (A.left,  B.bottom, A.right, A.bottom)
    //
    // Depending on how B intersects A, there will be 1 to 4 positive areas:
    //  - 4 occur when A contains B
    //  - 3 occur when B intersects a single edge
    //  - 2 occur when B intersects at a corner, or spans two opposing edges
    //  - 1 occurs when B spans two opposing edges and contains a 3rd, resulting in an exact rect
    //  - 0 occurs when B contains A, resulting in the empty rect
    //
    // Compute the relative areas of the 4 rects described above. Since each subrectangle shares
    // either the width or height of A, we only have to divide by the other dimension, which avoids
    // overflow on int32 types, and even if the float relative areas overflow to infinity, the
    // comparisons work out correctly and (one of) the infinitely large subrects will be chosen.
    float aHeight = (float) a.height();
    float aWidth = (float) a.width();
    float leftArea = 0.f, rightArea = 0.f, topArea = 0.f, bottomArea = 0.f;
    int positiveCount = 0;
    if (b.fLeft > a.fLeft) {
        leftArea = (b.fLeft - a.fLeft) / aWidth;
        positiveCount++;
    }
    if (a.fRight > b.fRight) {
        rightArea = (a.fRight - b.fRight) / aWidth;
        positiveCount++;
    }
    if (b.fTop > a.fTop) {
        topArea = (b.fTop - a.fTop) / aHeight;
        positiveCount++;
    }
    if (a.fBottom > b.fBottom) {
        bottomArea = (a.fBottom - b.fBottom) / aHeight;
        positiveCount++;
    }

    if (positiveCount == 0) {
        SkASSERT(b.contains(a));
        *out = R::MakeEmpty();
        return true;
    }

    *out = a;
    if (leftArea > rightArea && leftArea > topArea && leftArea > bottomArea) {
        // Left chunk of A, so the new right edge is B's left edge
        out->fRight = b.fLeft;
    } else if (rightArea > topArea && rightArea > bottomArea) {
        // Right chunk of A, so the new left edge is B's right edge
        out->fLeft = b.fRight;
    } else if (topArea > bottomArea) {
        // Top chunk of A, so the new bottom edge is B's top edge
        out->fBottom = b.fTop;
    } else {
        // Bottom chunk of A, so the new top edge is B's bottom edge
        SkASSERT(bottomArea > 0.f);
        out->fTop = b.fBottom;
    }

    // If we have 1 valid area, the disjoint shape is representable as a rectangle.
    SkASSERT(!R::Intersects(*out, b));
    return positiveCount == 1;
}

bool SkRectPriv::Subtract(const SkRect& a, const SkRect& b, SkRect* out) {
    return subtract<SkRect>(a, b, out);
}

bool SkRectPriv::Subtract(const SkIRect& a, const SkIRect& b, SkIRect* out) {
    return subtract<SkIRect>(a, b, out);
}
