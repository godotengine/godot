/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#ifndef SkPathWriter_DEFINED
#define SkPathWriter_DEFINED

#include "include/core/SkPath.h"
#include "include/private/SkTArray.h"
#include "include/private/SkTDArray.h"

class SkOpPtT;

// Construct the path one contour at a time.
// If the contour is closed, copy it to the final output.
// Otherwise, keep the partial contour for later assembly.

class SkPathWriter {
public:
    SkPathWriter(SkPath& path);
    void assemble();
    void conicTo(const SkPoint& pt1, const SkOpPtT* pt2, SkScalar weight);
    void cubicTo(const SkPoint& pt1, const SkPoint& pt2, const SkOpPtT* pt3);
    bool deferredLine(const SkOpPtT* pt);
    void deferredMove(const SkOpPtT* pt);
    void finishContour();
    bool hasMove() const { return !fFirstPtT; }
    void init();
    bool isClosed() const;
    const SkPath* nativePath() const { return fPathPtr; }
    void quadTo(const SkPoint& pt1, const SkOpPtT* pt2);

private:
    bool changedSlopes(const SkOpPtT* pt) const;
    void close();
    const SkTDArray<const SkOpPtT*>& endPtTs() const { return fEndPtTs; }
    void lineTo();
    bool matchedLast(const SkOpPtT*) const;
    void moveTo();
    const SkTArray<SkPath>& partials() const { return fPartials; }
    bool someAssemblyRequired();
    SkPoint update(const SkOpPtT* pt);

    SkPath fCurrent;  // contour under construction
    SkTArray<SkPath> fPartials;   // contours with mismatched starts and ends
    SkTDArray<const SkOpPtT*> fEndPtTs;  // possible pt values for partial starts and ends
    SkPath* fPathPtr;  // closed contours are written here
    const SkOpPtT* fDefer[2];  // [0] deferred move, [1] deferred line
    const SkOpPtT* fFirstPtT;  // first in current contour
};

#endif /* defined(__PathOps__SkPathWriter__) */
