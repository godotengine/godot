/*
 * Copyright 2015 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkPathOpsConic_DEFINED
#define SkPathOpsConic_DEFINED

#include "src/pathops/SkPathOpsQuad.h"

struct SkDConic {
    static const int kPointCount = 3;
    static const int kPointLast = kPointCount - 1;
    static const int kMaxIntersections = 4;

    SkDQuad fPts;
    SkScalar fWeight;

    bool collapsed() const {
        return fPts.collapsed();
    }

    bool controlsInside() const {
        return fPts.controlsInside();
    }

    void debugInit() {
        fPts.debugInit();
        fWeight = 0;
    }

    void debugSet(const SkDPoint* pts, SkScalar weight);

    SkDConic flip() const {
        SkDConic result = {{{fPts[2], fPts[1], fPts[0]}
                SkDEBUGPARAMS(fPts.fDebugGlobalState) }, fWeight};
        return result;
    }

#ifdef SK_DEBUG
    SkOpGlobalState* globalState() const { return fPts.globalState(); }
#endif

    static bool IsConic() { return true; }

    const SkDConic& set(const SkPoint pts[kPointCount], SkScalar weight
            SkDEBUGPARAMS(SkOpGlobalState* state = nullptr)) {
        fPts.set(pts  SkDEBUGPARAMS(state));
        fWeight = weight;
        return *this;
    }

    const SkDPoint& operator[](int n) const { return fPts[n]; }
    SkDPoint& operator[](int n) { return fPts[n]; }

    static int AddValidTs(double s[], int realRoots, double* t) {
        return SkDQuad::AddValidTs(s, realRoots, t);
    }

    void align(int endIndex, SkDPoint* dstPt) const {
        fPts.align(endIndex, dstPt);
    }

    SkDVector dxdyAtT(double t) const;
    static int FindExtrema(const double src[], SkScalar weight, double tValue[1]);

    bool hullIntersects(const SkDQuad& quad, bool* isLinear) const {
        return fPts.hullIntersects(quad, isLinear);
    }

    bool hullIntersects(const SkDConic& conic, bool* isLinear) const {
        return fPts.hullIntersects(conic.fPts, isLinear);
    }

    bool hullIntersects(const SkDCubic& cubic, bool* isLinear) const;

    bool isLinear(int startIndex, int endIndex) const {
        return fPts.isLinear(startIndex, endIndex);
    }

    static int maxIntersections() { return kMaxIntersections; }

    bool monotonicInX() const {
        return fPts.monotonicInX();
    }

    bool monotonicInY() const {
        return fPts.monotonicInY();
    }

    void otherPts(int oddMan, const SkDPoint* endPt[2]) const {
        fPts.otherPts(oddMan, endPt);
    }

    static int pointCount() { return kPointCount; }
    static int pointLast() { return kPointLast; }
    SkDPoint ptAtT(double t) const;

    static int RootsReal(double A, double B, double C, double t[2]) {
        return SkDQuad::RootsReal(A, B, C, t);
    }

    static int RootsValidT(const double A, const double B, const double C, double s[2]) {
        return SkDQuad::RootsValidT(A, B, C, s);
    }

    SkDConic subDivide(double t1, double t2) const;
    void subDivide(double t1, double t2, SkDConic* c) const { *c = this->subDivide(t1, t2); }

    static SkDConic SubDivide(const SkPoint a[kPointCount], SkScalar weight, double t1, double t2) {
        SkDConic conic;
        conic.set(a, weight);
        return conic.subDivide(t1, t2);
    }

    SkDPoint subDivide(const SkDPoint& a, const SkDPoint& c, double t1, double t2,
            SkScalar* weight) const;

    static SkDPoint SubDivide(const SkPoint pts[kPointCount], SkScalar weight,
                              const SkDPoint& a, const SkDPoint& c,
                              double t1, double t2, SkScalar* newWeight) {
        SkDConic conic;
        conic.set(pts, weight);
        return conic.subDivide(a, c, t1, t2, newWeight);
    }

    // utilities callable by the user from the debugger when the implementation code is linked in
    void dump() const;
    void dumpID(int id) const;
    void dumpInner() const;

};

class SkTConic : public SkTCurve {
public:
    SkDConic fConic;

    SkTConic() {}

    SkTConic(const SkDConic& c)
        : fConic(c) {
    }

    ~SkTConic() override {}

    const SkDPoint& operator[](int n) const override { return fConic[n]; }
    SkDPoint& operator[](int n) override { return fConic[n]; }

    bool collapsed() const override { return fConic.collapsed(); }
    bool controlsInside() const override { return fConic.controlsInside(); }
    void debugInit() override { return fConic.debugInit(); }
#if DEBUG_T_SECT
    void dumpID(int id) const override { return fConic.dumpID(id); }
#endif
    SkDVector dxdyAtT(double t) const override { return fConic.dxdyAtT(t); }
#ifdef SK_DEBUG
    SkOpGlobalState* globalState() const override { return fConic.globalState(); }
#endif
    bool hullIntersects(const SkDQuad& quad, bool* isLinear) const override;

    bool hullIntersects(const SkDConic& conic, bool* isLinear) const override {
        return conic.hullIntersects(fConic, isLinear);
    }

    bool hullIntersects(const SkDCubic& cubic, bool* isLinear) const override;

    bool hullIntersects(const SkTCurve& curve, bool* isLinear) const override {
        return curve.hullIntersects(fConic, isLinear);
    }

    int intersectRay(SkIntersections* i, const SkDLine& line) const override;
    bool IsConic() const override { return true; }
    SkTCurve* make(SkArenaAlloc& heap) const override { return heap.make<SkTConic>(); }

    int maxIntersections() const override { return SkDConic::kMaxIntersections; }

    void otherPts(int oddMan, const SkDPoint* endPt[2]) const override {
        fConic.otherPts(oddMan, endPt);
    }

    int pointCount() const override { return SkDConic::kPointCount; }
    int pointLast() const override { return SkDConic::kPointLast; }
    SkDPoint ptAtT(double t) const override { return fConic.ptAtT(t); }
    void setBounds(SkDRect* ) const override;

    void subDivide(double t1, double t2, SkTCurve* curve) const override {
        ((SkTConic*) curve)->fConic = fConic.subDivide(t1, t2);
    }
};

#endif
