/*
 * Copyright 2018 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkPathOpsTCurve_DEFINED
#define SkPathOpsTCurve_DEFINED

#include "src/pathops/SkPathOpsPoint.h"

class SkArenaAlloc;
class SkIntersections;

class SkTCurve {
public:
    virtual ~SkTCurve() {}
    virtual const SkDPoint& operator[](int n) const = 0;
    virtual SkDPoint& operator[](int n) = 0;

    virtual bool collapsed() const = 0;
    virtual bool controlsInside() const = 0;
    virtual void debugInit() = 0;
#if DEBUG_T_SECT
    virtual void dumpID(int id) const = 0;
#endif
    virtual SkDVector dxdyAtT(double t) const = 0;
    virtual bool hullIntersects(const SkDQuad& , bool* isLinear) const = 0;
    virtual bool hullIntersects(const SkDConic& , bool* isLinear) const = 0;
    virtual bool hullIntersects(const SkDCubic& , bool* isLinear) const = 0;
    virtual bool hullIntersects(const SkTCurve& , bool* isLinear) const = 0;
    virtual int intersectRay(SkIntersections* i, const SkDLine& line) const = 0;
    virtual bool IsConic() const = 0;
    virtual SkTCurve* make(SkArenaAlloc& ) const = 0;
    virtual int maxIntersections() const = 0;
    virtual void otherPts(int oddMan, const SkDPoint* endPt[2]) const = 0;
    virtual int pointCount() const = 0;
    virtual int pointLast() const = 0;
    virtual SkDPoint ptAtT(double t) const = 0;
    virtual void setBounds(SkDRect* ) const = 0;
    virtual void subDivide(double t1, double t2, SkTCurve* curve) const = 0;
#ifdef SK_DEBUG
    virtual SkOpGlobalState* globalState() const = 0;
#endif
};

#endif
