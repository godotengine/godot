/*
 * Copyright 2019 Google LLC.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkPathTypes_DEFINED
#define SkPathTypes_DEFINED

#include "include/core/SkTypes.h"

enum class SkPathFillType {
    /** Specifies that "inside" is computed by a non-zero sum of signed edge crossings */
    kWinding,
    /** Specifies that "inside" is computed by an odd number of edge crossings */
    kEvenOdd,
    /** Same as Winding, but draws outside of the path, rather than inside */
    kInverseWinding,
    /** Same as EvenOdd, but draws outside of the path, rather than inside */
    kInverseEvenOdd
};

static inline bool SkPathFillType_IsEvenOdd(SkPathFillType ft) {
    return (static_cast<int>(ft) & 1) != 0;
}

static inline bool SkPathFillType_IsInverse(SkPathFillType ft) {
    return (static_cast<int>(ft) & 2) != 0;
}

static inline SkPathFillType SkPathFillType_ConvertToNonInverse(SkPathFillType ft) {
    return static_cast<SkPathFillType>(static_cast<int>(ft) & 1);
}

enum class SkPathDirection {
    /** clockwise direction for adding closed contours */
    kCW,
    /** counter-clockwise direction for adding closed contours */
    kCCW,
};

enum SkPathSegmentMask {
    kLine_SkPathSegmentMask   = 1 << 0,
    kQuad_SkPathSegmentMask   = 1 << 1,
    kConic_SkPathSegmentMask  = 1 << 2,
    kCubic_SkPathSegmentMask  = 1 << 3,
};

enum class SkPathVerb {
    kMove,   //!< SkPath::RawIter returns 1 point
    kLine,   //!< SkPath::RawIter returns 2 points
    kQuad,   //!< SkPath::RawIter returns 3 points
    kConic,  //!< SkPath::RawIter returns 3 points + 1 weight
    kCubic,  //!< SkPath::RawIter returns 4 points
    kClose   //!< SkPath::RawIter returns 0 points
};

#endif
