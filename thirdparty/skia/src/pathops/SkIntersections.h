/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#ifndef SkIntersections_DEFINE
#define SkIntersections_DEFINE

#include "src/pathops/SkPathOpsConic.h"
#include "src/pathops/SkPathOpsCubic.h"
#include "src/pathops/SkPathOpsLine.h"
#include "src/pathops/SkPathOpsPoint.h"
#include "src/pathops/SkPathOpsQuad.h"

class SkIntersections {
public:
    SkIntersections(SkDEBUGCODE(SkOpGlobalState* globalState = nullptr))
        : fSwap(0)
#ifdef SK_DEBUG
        SkDEBUGPARAMS(fDebugGlobalState(globalState))
        , fDepth(0)
#endif
    {
        sk_bzero(fPt, sizeof(fPt));
        sk_bzero(fPt2, sizeof(fPt2));
        sk_bzero(fT, sizeof(fT));
        sk_bzero(fNearlySame, sizeof(fNearlySame));
#if DEBUG_T_SECT_LOOP_COUNT
        sk_bzero(fDebugLoopCount, sizeof(fDebugLoopCount));
#endif
        reset();
        fMax = 0;  // require that the caller set the max
    }

    class TArray {
    public:
        explicit TArray(const double ts[10]) : fTArray(ts) {}
        double operator[](int n) const {
            return fTArray[n];
        }
        const double* fTArray;
    };
    TArray operator[](int n) const { return TArray(fT[n]); }

    void allowNear(bool nearAllowed) {
        fAllowNear = nearAllowed;
    }

    void clearCoincidence(int index) {
        SkASSERT(index >= 0);
        int bit = 1 << index;
        fIsCoincident[0] &= ~bit;
        fIsCoincident[1] &= ~bit;
    }

    int conicHorizontal(const SkPoint a[3], SkScalar weight, SkScalar left, SkScalar right,
                SkScalar y, bool flipped) {
        SkDConic conic;
        conic.set(a, weight);
        fMax = 2;
        return horizontal(conic, left, right, y, flipped);
    }

    int conicVertical(const SkPoint a[3], SkScalar weight, SkScalar top, SkScalar bottom,
            SkScalar x, bool flipped) {
        SkDConic conic;
        conic.set(a, weight);
        fMax = 2;
        return vertical(conic, top, bottom, x, flipped);
    }

    int conicLine(const SkPoint a[3], SkScalar weight, const SkPoint b[2]) {
        SkDConic conic;
        conic.set(a, weight);
        SkDLine line;
        line.set(b);
        fMax = 3; // 2;  permit small coincident segment + non-coincident intersection
        return intersect(conic, line);
    }

    int cubicHorizontal(const SkPoint a[4], SkScalar left, SkScalar right, SkScalar y,
                        bool flipped) {
        SkDCubic cubic;
        cubic.set(a);
        fMax = 3;
        return horizontal(cubic, left, right, y, flipped);
    }

    int cubicVertical(const SkPoint a[4], SkScalar top, SkScalar bottom, SkScalar x, bool flipped) {
        SkDCubic cubic;
        cubic.set(a);
        fMax = 3;
        return vertical(cubic, top, bottom, x, flipped);
    }

    int cubicLine(const SkPoint a[4], const SkPoint b[2]) {
        SkDCubic cubic;
        cubic.set(a);
        SkDLine line;
        line.set(b);
        fMax = 3;
        return intersect(cubic, line);
    }

#ifdef SK_DEBUG
    SkOpGlobalState* globalState() const { return fDebugGlobalState; }
#endif

    bool hasT(double t) const {
        SkASSERT(t == 0 || t == 1);
        return fUsed > 0 && (t == 0 ? fT[0][0] == 0 : fT[0][fUsed - 1] == 1);
    }

    bool hasOppT(double t) const {
        SkASSERT(t == 0 || t == 1);
        return fUsed > 0 && (fT[1][0] == t || fT[1][fUsed - 1] == t);
    }

    int insertSwap(double one, double two, const SkDPoint& pt) {
        if (fSwap) {
            return insert(two, one, pt);
        } else {
            return insert(one, two, pt);
        }
    }

    bool isCoincident(int index) {
        return (fIsCoincident[0] & 1 << index) != 0;
    }

    int lineHorizontal(const SkPoint a[2], SkScalar left, SkScalar right, SkScalar y,
                       bool flipped) {
        SkDLine line;
        line.set(a);
        fMax = 2;
        return horizontal(line, left, right, y, flipped);
    }

    int lineVertical(const SkPoint a[2], SkScalar top, SkScalar bottom, SkScalar x, bool flipped) {
        SkDLine line;
        line.set(a);
        fMax = 2;
        return vertical(line, top, bottom, x, flipped);
    }

    int lineLine(const SkPoint a[2], const SkPoint b[2]) {
        SkDLine aLine, bLine;
        aLine.set(a);
        bLine.set(b);
        fMax = 2;
        return intersect(aLine, bLine);
    }

    bool nearlySame(int index) const {
        SkASSERT(index == 0 || index == 1);
        return fNearlySame[index];
    }

    const SkDPoint& pt(int index) const {
        return fPt[index];
    }

    const SkDPoint& pt2(int index) const {
        return fPt2[index];
    }

    int quadHorizontal(const SkPoint a[3], SkScalar left, SkScalar right, SkScalar y,
                       bool flipped) {
        SkDQuad quad;
        quad.set(a);
        fMax = 2;
        return horizontal(quad, left, right, y, flipped);
    }

    int quadVertical(const SkPoint a[3], SkScalar top, SkScalar bottom, SkScalar x, bool flipped) {
        SkDQuad quad;
        quad.set(a);
        fMax = 2;
        return vertical(quad, top, bottom, x, flipped);
    }

    int quadLine(const SkPoint a[3], const SkPoint b[2]) {
        SkDQuad quad;
        quad.set(a);
        SkDLine line;
        line.set(b);
        return intersect(quad, line);
    }

    // leaves swap, max alone
    void reset() {
        fAllowNear = true;
        fUsed = 0;
        sk_bzero(fIsCoincident, sizeof(fIsCoincident));
    }

    void set(bool swap, int tIndex, double t) {
        fT[(int) swap][tIndex] = t;
    }

    void setMax(int max) {
        SkASSERT(max <= (int) SK_ARRAY_COUNT(fPt));
        fMax = max;
    }

    void swap() {
        fSwap ^= true;
    }

    bool swapped() const {
        return fSwap;
    }

    int used() const {
        return fUsed;
    }

    void downDepth() {
        SkASSERT(--fDepth >= 0);
    }

    bool unBumpT(int index) {
        SkASSERT(fUsed == 1);
        fT[0][index] = fT[0][index] * (1 + BUMP_EPSILON * 2) - BUMP_EPSILON;
        if (!between(0, fT[0][index], 1)) {
            fUsed = 0;
            return false;
        }
        return true;
    }

    void upDepth() {
        SkASSERT(++fDepth < 16);
    }

    void alignQuadPts(const SkPoint a[3], const SkPoint b[3]);
    int cleanUpCoincidence();
    int closestTo(double rangeStart, double rangeEnd, const SkDPoint& testPt, double* dist) const;
    void cubicInsert(double one, double two, const SkDPoint& pt, const SkDCubic& c1,
                     const SkDCubic& c2);
    void flip();
    int horizontal(const SkDLine&, double left, double right, double y, bool flipped);
    int horizontal(const SkDQuad&, double left, double right, double y, bool flipped);
    int horizontal(const SkDQuad&, double left, double right, double y, double tRange[2]);
    int horizontal(const SkDCubic&, double y, double tRange[3]);
    int horizontal(const SkDConic&, double left, double right, double y, bool flipped);
    int horizontal(const SkDCubic&, double left, double right, double y, bool flipped);
    int horizontal(const SkDCubic&, double left, double right, double y, double tRange[3]);
    static double HorizontalIntercept(const SkDLine& line, double y);
    static int HorizontalIntercept(const SkDQuad& quad, SkScalar y, double* roots);
    static int HorizontalIntercept(const SkDConic& conic, SkScalar y, double* roots);
    // FIXME : does not respect swap
    int insert(double one, double two, const SkDPoint& pt);
    void insertNear(double one, double two, const SkDPoint& pt1, const SkDPoint& pt2);
    // start if index == 0 : end if index == 1
    int insertCoincident(double one, double two, const SkDPoint& pt);
    int intersect(const SkDLine&, const SkDLine&);
    int intersect(const SkDQuad&, const SkDLine&);
    int intersect(const SkDQuad&, const SkDQuad&);
    int intersect(const SkDConic&, const SkDLine&);
    int intersect(const SkDConic&, const SkDQuad&);
    int intersect(const SkDConic&, const SkDConic&);
    int intersect(const SkDCubic&, const SkDLine&);
    int intersect(const SkDCubic&, const SkDQuad&);
    int intersect(const SkDCubic&, const SkDConic&);
    int intersect(const SkDCubic&, const SkDCubic&);
    int intersectRay(const SkDLine&, const SkDLine&);
    int intersectRay(const SkDQuad&, const SkDLine&);
    int intersectRay(const SkDConic&, const SkDLine&);
    int intersectRay(const SkDCubic&, const SkDLine&);
    int intersectRay(const SkTCurve& tCurve, const SkDLine& line) {
        return tCurve.intersectRay(this, line);
    }

    void merge(const SkIntersections& , int , const SkIntersections& , int );
    int mostOutside(double rangeStart, double rangeEnd, const SkDPoint& origin) const;
    void removeOne(int index);
    void setCoincident(int index);
    int vertical(const SkDLine&, double top, double bottom, double x, bool flipped);
    int vertical(const SkDQuad&, double top, double bottom, double x, bool flipped);
    int vertical(const SkDConic&, double top, double bottom, double x, bool flipped);
    int vertical(const SkDCubic&, double top, double bottom, double x, bool flipped);
    static double VerticalIntercept(const SkDLine& line, double x);
    static int VerticalIntercept(const SkDQuad& quad, SkScalar x, double* roots);
    static int VerticalIntercept(const SkDConic& conic, SkScalar x, double* roots);

    int depth() const {
#ifdef SK_DEBUG
        return fDepth;
#else
        return 0;
#endif
    }

    enum DebugLoop {
        kIterations_DebugLoop,
        kCoinCheck_DebugLoop,
        kComputePerp_DebugLoop,
    };

    void debugBumpLoopCount(DebugLoop );
    int debugCoincidentUsed() const;
    int debugLoopCount(DebugLoop ) const;
    void debugResetLoopCount();
    void dump() const;  // implemented for testing only

private:
    bool cubicCheckCoincidence(const SkDCubic& c1, const SkDCubic& c2);
    bool cubicExactEnd(const SkDCubic& cubic1, bool start, const SkDCubic& cubic2);
    void cubicNearEnd(const SkDCubic& cubic1, bool start, const SkDCubic& cubic2, const SkDRect& );
    void cleanUpParallelLines(bool parallel);
    void computePoints(const SkDLine& line, int used);

    SkDPoint fPt[13];  // FIXME: since scans store points as SkPoint, this should also
    SkDPoint fPt2[2];  // used by nearly same to store alternate intersection point
    double fT[2][13];
    uint16_t fIsCoincident[2];  // bit set for each curve's coincident T
    bool fNearlySame[2];  // true if end points nearly match
    unsigned char fUsed;
    unsigned char fMax;
    bool fAllowNear;
    bool fSwap;
#ifdef SK_DEBUG
    SkOpGlobalState* fDebugGlobalState;
    int fDepth;
#endif
#if DEBUG_T_SECT_LOOP_COUNT
    int fDebugLoopCount[3];
#endif
};

#endif
