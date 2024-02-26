/*
 * Copyright 2011 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#ifndef SkLineClipper_DEFINED
#define SkLineClipper_DEFINED

#include "include/core/SkPoint.h"
#include "include/core/SkRect.h"

class SkLineClipper {
public:
    enum {
        kMaxPoints = 4,
        kMaxClippedLineSegments = kMaxPoints - 1
    };

    /*  Clip the line pts[0]...pts[1] against clip, ignoring segments that
        lie completely above or below the clip. For portions to the left or
        right, turn those into vertical line segments that are aligned to the
        edge of the clip.

        Return the number of line segments that result, and store the end-points
        of those segments sequentially in lines as follows:
            1st segment: lines[0]..lines[1]
            2nd segment: lines[1]..lines[2]
            3rd segment: lines[2]..lines[3]
     */
    static int ClipLine(const SkPoint pts[2], const SkRect& clip,
                        SkPoint lines[kMaxPoints], bool canCullToTheRight);

    /*  Intersect the line segment against the rect. If there is a non-empty
        resulting segment, return true and set dst[] to that segment. If not,
        return false and ignore dst[].

        ClipLine is specialized for scan-conversion, as it adds vertical
        segments on the sides to show where the line extended beyond the
        left or right sides. IntersectLine does not.
     */
    static bool IntersectLine(const SkPoint src[2], const SkRect& clip, SkPoint dst[2]);
};

#endif
