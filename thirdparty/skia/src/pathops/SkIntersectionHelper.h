/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#ifndef SkIntersectionHelper_DEFINED
#define SkIntersectionHelper_DEFINED

#include "include/core/SkPath.h"
#include "src/pathops/SkOpContour.h"
#include "src/pathops/SkOpSegment.h"

#ifdef SK_DEBUG
#include "src/pathops/SkPathOpsPoint.h"
#endif

class SkIntersectionHelper {
public:
    enum SegmentType {
        kHorizontalLine_Segment = -1,
        kVerticalLine_Segment = 0,
        kLine_Segment = SkPath::kLine_Verb,
        kQuad_Segment = SkPath::kQuad_Verb,
        kConic_Segment = SkPath::kConic_Verb,
        kCubic_Segment = SkPath::kCubic_Verb,
    };

    bool advance() {
        fSegment = fSegment->next();
        return fSegment != nullptr;
    }

    SkScalar bottom() const {
        return bounds().fBottom;
    }

    const SkPathOpsBounds& bounds() const {
        return fSegment->bounds();
    }

    SkOpContour* contour() const {
        return fSegment->contour();
    }

    void init(SkOpContour* contour) {
        fSegment = contour->first();
    }

    SkScalar left() const {
        return bounds().fLeft;
    }

    const SkPoint* pts() const {
        return fSegment->pts();
    }

    SkScalar right() const {
        return bounds().fRight;
    }

    SkOpSegment* segment() const {
        return fSegment;
    }

    SegmentType segmentType() const {
        SegmentType type = (SegmentType) fSegment->verb();
        if (type != kLine_Segment) {
            return type;
        }
        if (fSegment->isHorizontal()) {
            return kHorizontalLine_Segment;
        }
        if (fSegment->isVertical()) {
            return kVerticalLine_Segment;
        }
        return kLine_Segment;
    }

    bool startAfter(const SkIntersectionHelper& after) {
        fSegment = after.fSegment->next();
        return fSegment != nullptr;
    }

    SkScalar top() const {
        return bounds().fTop;
    }

    SkScalar weight() const {
        return fSegment->weight();
    }

    SkScalar x() const {
        return bounds().fLeft;
    }

    bool xFlipped() const {
        return x() != pts()[0].fX;
    }

    SkScalar y() const {
        return bounds().fTop;
    }

    bool yFlipped() const {
        return y() != pts()[0].fY;
    }

private:
    SkOpSegment* fSegment;
};

#endif
