/*
 * Copyright 2018 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkRectPriv_DEFINED
#define SkRectPriv_DEFINED

#include "include/core/SkRect.h"
#include "src/core/SkMathPriv.h"

class SkRectPriv {
public:
    // Returns an irect that is very large, and can be safely round-trip with SkRect and still
    // be considered non-empty (i.e. width/height > 0) even if we round-out the SkRect.
    static SkIRect MakeILarge() {
        // SK_MaxS32 >> 1 seemed better, but it did not survive round-trip with SkRect and rounding.
        // Also, 1 << 29 can be perfectly represented in float, while SK_MaxS32 >> 1 cannot.
        const int32_t large = 1 << 29;
        return { -large, -large, large, large };
    }

    static SkIRect MakeILargestInverted() {
        return { SK_MaxS32, SK_MaxS32, SK_MinS32, SK_MinS32 };
    }

    static SkRect MakeLargeS32() {
        SkRect r;
        r.set(MakeILarge());
        return r;
    }

    static SkRect MakeLargest() {
        return { SK_ScalarMin, SK_ScalarMin, SK_ScalarMax, SK_ScalarMax };
    }

    static constexpr SkRect MakeLargestInverted() {
        return { SK_ScalarMax, SK_ScalarMax, SK_ScalarMin, SK_ScalarMin };
    }

    static void GrowToInclude(SkRect* r, const SkPoint& pt) {
        r->fLeft  =  std::min(pt.fX, r->fLeft);
        r->fRight =  std::max(pt.fX, r->fRight);
        r->fTop    = std::min(pt.fY, r->fTop);
        r->fBottom = std::max(pt.fY, r->fBottom);
    }

    // Conservative check if r can be expressed in fixed-point.
    // Will return false for very large values that might have fit
    static bool FitsInFixed(const SkRect& r) {
        return SkFitsInFixed(r.fLeft) && SkFitsInFixed(r.fTop) &&
               SkFitsInFixed(r.fRight) && SkFitsInFixed(r.fBottom);
    }

    static bool Is16Bit(const SkIRect& r) {
        return  SkTFitsIn<int16_t>(r.fLeft)  && SkTFitsIn<int16_t>(r.fTop) &&
                SkTFitsIn<int16_t>(r.fRight) && SkTFitsIn<int16_t>(r.fBottom);
    }

    // Returns r.width()/2 but divides first to avoid width() overflowing.
    static SkScalar HalfWidth(const SkRect& r) {
        return SkScalarHalf(r.fRight) - SkScalarHalf(r.fLeft);
    }
    // Returns r.height()/2 but divides first to avoid height() overflowing.
    static SkScalar HalfHeight(const SkRect& r) {
        return SkScalarHalf(r.fBottom) - SkScalarHalf(r.fTop);
    }

    // Evaluate A-B. If the difference shape cannot be represented as a rectangle then false is
    // returned and 'out' is set to the largest rectangle contained in said shape. If true is
    // returned then A-B is representable as a rectangle, which is stored in 'out'.
    static bool Subtract(const SkRect& a, const SkRect& b, SkRect* out);
    static bool Subtract(const SkIRect& a, const SkIRect& b, SkIRect* out);

    // Evaluate A-B, and return the largest rectangle contained in that shape (since the difference
    // may not be representable as rectangle). The returned rectangle will not intersect B.
    static SkRect Subtract(const SkRect& a, const SkRect& b) {
        SkRect diff;
        Subtract(a, b, &diff);
        return diff;
    }
    static SkIRect Subtract(const SkIRect& a, const SkIRect& b) {
        SkIRect diff;
        Subtract(a, b, &diff);
        return diff;
    }
};


#endif
