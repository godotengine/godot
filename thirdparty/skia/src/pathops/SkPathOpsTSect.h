/*
 * Copyright 2014 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#ifndef SkPathOpsTSect_DEFINED
#define SkPathOpsTSect_DEFINED

#include "include/private/SkMacros.h"
#include "src/core/SkArenaAlloc.h"
#include "src/pathops/SkIntersections.h"
#include "src/pathops/SkPathOpsBounds.h"
#include "src/pathops/SkPathOpsRect.h"
#include "src/pathops/SkPathOpsTCurve.h"

#include <utility>

#ifdef SK_DEBUG
typedef uint8_t SkOpDebugBool;
#else
typedef bool SkOpDebugBool;
#endif

static inline bool SkDoubleIsNaN(double x) {
    return x != x;
}

class SkTCoincident {
public:
    SkTCoincident() {
        this->init();
    }

    void debugInit() {
#ifdef SK_DEBUG
        this->fPerpPt.fX = this->fPerpPt.fY = SK_ScalarNaN;
        this->fPerpT = SK_ScalarNaN;
        this->fMatch = 0xFF;
#endif
    }

    char dumpIsCoincidentStr() const;
    void dump() const;

    bool isMatch() const {
        SkASSERT(!!fMatch == fMatch);
        return SkToBool(fMatch);
    }

    void init() {
        fPerpT = -1;
        fMatch = false;
        fPerpPt.fX = fPerpPt.fY = SK_ScalarNaN;
    }

    void markCoincident() {
        if (!fMatch) {
            fPerpT = -1;
        }
        fMatch = true;
    }

    const SkDPoint& perpPt() const {
        return fPerpPt;
    }

    double perpT() const {
        return fPerpT;
    }

    void setPerp(const SkTCurve& c1, double t, const SkDPoint& cPt, const SkTCurve& );

private:
    SkDPoint fPerpPt;
    double fPerpT;  // perpendicular intersection on opposite curve
    SkOpDebugBool fMatch;
};

class SkTSect;
class SkTSpan;

struct SkTSpanBounded {
    SkTSpan* fBounded;
    SkTSpanBounded* fNext;
};

class SkTSpan {
public:
    SkTSpan(const SkTCurve& curve, SkArenaAlloc& heap) {
        fPart = curve.make(heap);
    }

    void addBounded(SkTSpan* , SkArenaAlloc* );
    double closestBoundedT(const SkDPoint& pt) const;
    bool contains(double t) const;

    void debugInit(const SkTCurve& curve, SkArenaAlloc& heap) {
#ifdef SK_DEBUG
        SkTCurve* fake = curve.make(heap);
        fake->debugInit();
        init(*fake);
        initBounds(*fake);
        fCoinStart.init();
        fCoinEnd.init();
#endif
    }

    const SkTSect* debugOpp() const;

#ifdef SK_DEBUG
    void debugSetGlobalState(SkOpGlobalState* state) {
        fDebugGlobalState = state;
    }

    const SkTSpan* debugSpan(int ) const;
    const SkTSpan* debugT(double t) const;
    bool debugIsBefore(const SkTSpan* span) const;
#endif
    void dump() const;
    void dumpAll() const;
    void dumpBounded(int id) const;
    void dumpBounds() const;
    void dumpCoin() const;

    double endT() const {
        return fEndT;
    }

    SkTSpan* findOppSpan(const SkTSpan* opp) const;

    SkTSpan* findOppT(double t) const {
        SkTSpan* result = oppT(t);
        SkOPASSERT(result);
        return result;
    }

    SkDEBUGCODE(SkOpGlobalState* globalState() const { return fDebugGlobalState; })

    bool hasOppT(double t) const {
        return SkToBool(oppT(t));
    }

    int hullsIntersect(SkTSpan* span, bool* start, bool* oppStart);
    void init(const SkTCurve& );
    bool initBounds(const SkTCurve& );

    bool isBounded() const {
        return fBounded != nullptr;
    }

    bool linearsIntersect(SkTSpan* span);
    double linearT(const SkDPoint& ) const;

    void markCoincident() {
        fCoinStart.markCoincident();
        fCoinEnd.markCoincident();
    }

    const SkTSpan* next() const {
        return fNext;
    }

    bool onlyEndPointsInCommon(const SkTSpan* opp, bool* start,
            bool* oppStart, bool* ptsInCommon);

    const SkTCurve& part() const {
        return *fPart;
    }

    int pointCount() const {
        return fPart->pointCount();
    }

    const SkDPoint& pointFirst() const {
        return (*fPart)[0];
    }

    const SkDPoint& pointLast() const {
        return (*fPart)[fPart->pointLast()];
    }

    bool removeAllBounded();
    bool removeBounded(const SkTSpan* opp);

    void reset() {
        fBounded = nullptr;
    }

    void resetBounds(const SkTCurve& curve) {
        fIsLinear = fIsLine = false;
        initBounds(curve);
    }

    bool split(SkTSpan* work, SkArenaAlloc* heap) {
        return splitAt(work, (work->fStartT + work->fEndT) * 0.5, heap);
    }

    bool splitAt(SkTSpan* work, double t, SkArenaAlloc* heap);

    double startT() const {
        return fStartT;
    }

private:

    // implementation is for testing only
    int debugID() const {
        return PATH_OPS_DEBUG_T_SECT_RELEASE(fID, -1);
    }

    void dumpID() const;

    int hullCheck(const SkTSpan* opp, bool* start, bool* oppStart);
    int linearIntersects(const SkTCurve& ) const;
    SkTSpan* oppT(double t) const;

    void validate() const;
    void validateBounded() const;
    void validatePerpT(double oppT) const;
    void validatePerpPt(double t, const SkDPoint& ) const;

    SkTCurve* fPart;
    SkTCoincident fCoinStart;
    SkTCoincident fCoinEnd;
    SkTSpanBounded* fBounded;
    SkTSpan* fPrev;
    SkTSpan* fNext;
    SkDRect fBounds;
    double fStartT;
    double fEndT;
    double fBoundsMax;
    SkOpDebugBool fCollapsed;
    SkOpDebugBool fHasPerp;
    SkOpDebugBool fIsLinear;
    SkOpDebugBool fIsLine;
    SkOpDebugBool fDeleted;
    SkDEBUGCODE(SkOpGlobalState* fDebugGlobalState);
    SkDEBUGCODE(SkTSect* fDebugSect);
    PATH_OPS_DEBUG_T_SECT_CODE(int fID);
    friend class SkTSect;
};

class SkTSect {
public:
    SkTSect(const SkTCurve& c
                             SkDEBUGPARAMS(SkOpGlobalState* ) PATH_OPS_DEBUG_T_SECT_PARAMS(int id));
    static void BinarySearch(SkTSect* sect1, SkTSect* sect2,
            SkIntersections* intersections);

    SkDEBUGCODE(SkOpGlobalState* globalState() { return fDebugGlobalState; })
    bool hasBounded(const SkTSpan* ) const;

    const SkTSect* debugOpp() const {
        return SkDEBUGRELEASE(fOppSect, nullptr);
    }

#ifdef SK_DEBUG
    const SkTSpan* debugSpan(int id) const;
    const SkTSpan* debugT(double t) const;
#endif
    void dump() const;
    void dumpBoth(SkTSect* ) const;
    void dumpBounded(int id) const;
    void dumpBounds() const;
    void dumpCoin() const;
    void dumpCoinCurves() const;
    void dumpCurves() const;

private:
    enum {
        kZeroS1Set = 1,
        kOneS1Set = 2,
        kZeroS2Set = 4,
        kOneS2Set = 8
    };

    SkTSpan* addFollowing(SkTSpan* prior);
    void addForPerp(SkTSpan* span, double t);
    SkTSpan* addOne();

    SkTSpan* addSplitAt(SkTSpan* span, double t) {
        SkTSpan* result = this->addOne();
        SkDEBUGCODE(result->debugSetGlobalState(this->globalState()));
        result->splitAt(span, t, &fHeap);
        result->initBounds(fCurve);
        span->initBounds(fCurve);
        return result;
    }

    bool binarySearchCoin(SkTSect* , double tStart, double tStep, double* t,
                          double* oppT, SkTSpan** oppFirst);
    SkTSpan* boundsMax();
    bool coincidentCheck(SkTSect* sect2);
    void coincidentForce(SkTSect* sect2, double start1s, double start1e);
    bool coincidentHasT(double t);
    int collapsed() const;
    void computePerpendiculars(SkTSect* sect2, SkTSpan* first,
                               SkTSpan* last);
    int countConsecutiveSpans(SkTSpan* first,
                              SkTSpan** last) const;

    int debugID() const {
        return PATH_OPS_DEBUG_T_SECT_RELEASE(fID, -1);
    }

    bool deleteEmptySpans();
    void dumpCommon(const SkTSpan* ) const;
    void dumpCommonCurves(const SkTSpan* ) const;
    static int EndsEqual(const SkTSect* sect1, const SkTSect* sect2,
                         SkIntersections* );
    bool extractCoincident(SkTSect* sect2, SkTSpan* first,
                           SkTSpan* last, SkTSpan** result);
    SkTSpan* findCoincidentRun(SkTSpan* first, SkTSpan** lastPtr);
    int intersects(SkTSpan* span, SkTSect* opp,
                   SkTSpan* oppSpan, int* oppResult);
    bool isParallel(const SkDLine& thisLine, const SkTSect* opp) const;
    int linesIntersect(SkTSpan* span, SkTSect* opp,
                       SkTSpan* oppSpan, SkIntersections* );
    bool markSpanGone(SkTSpan* span);
    bool matchedDirection(double t, const SkTSect* sect2, double t2) const;
    void matchedDirCheck(double t, const SkTSect* sect2, double t2,
                         bool* calcMatched, bool* oppMatched) const;
    void mergeCoincidence(SkTSect* sect2);

    const SkDPoint& pointLast() const {
        return fCurve[fCurve.pointLast()];
    }

    SkTSpan* prev(SkTSpan* ) const;
    bool removeByPerpendicular(SkTSect* opp);
    void recoverCollapsed();
    bool removeCoincident(SkTSpan* span, bool isBetween);
    void removeAllBut(const SkTSpan* keep, SkTSpan* span,
                      SkTSect* opp);
    bool removeSpan(SkTSpan* span);
    void removeSpanRange(SkTSpan* first, SkTSpan* last);
    bool removeSpans(SkTSpan* span, SkTSect* opp);
    void removedEndCheck(SkTSpan* span);

    void resetRemovedEnds() {
        fRemovedStartT = fRemovedEndT = false;
    }

    SkTSpan* spanAtT(double t, SkTSpan** priorSpan);
    SkTSpan* tail();
    bool trim(SkTSpan* span, SkTSect* opp);
    bool unlinkSpan(SkTSpan* span);
    bool updateBounded(SkTSpan* first, SkTSpan* last,
                       SkTSpan* oppFirst);
    void validate() const;
    void validateBounded() const;

    const SkTCurve& fCurve;
    SkSTArenaAlloc<1024> fHeap;
    SkTSpan* fHead;
    SkTSpan* fCoincident;
    SkTSpan* fDeleted;
    int fActiveCount;
    bool fRemovedStartT;
    bool fRemovedEndT;
    bool fHung;
    SkDEBUGCODE(SkOpGlobalState* fDebugGlobalState);
    SkDEBUGCODE(SkTSect* fOppSect);
    PATH_OPS_DEBUG_T_SECT_CODE(int fID);
    PATH_OPS_DEBUG_T_SECT_CODE(int fDebugCount);
#if DEBUG_T_SECT
    int fDebugAllocatedCount;
#endif
    friend class SkTSpan;
};

#endif
