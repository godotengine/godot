/*
 * Copyright 2015 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

// given a prospective edge, compute its initial winding by projecting a ray
// if the ray hits another edge
    // if the edge doesn't have a winding yet, hop up to that edge and start over
        // concern : check for hops forming a loop
    // if the edge is unsortable, or
    // the intersection is nearly at the ends, or
    // the tangent at the intersection is nearly coincident to the ray,
        // choose a different ray and try again
            // concern : if it is unable to succeed after N tries, try another edge? direction?
// if no edge is hit, compute the winding directly

// given the top span, project the most perpendicular ray and look for intersections
    // let's try up and then down. What the hey

// bestXY is initialized by caller with basePt

#include "src/core/SkTSort.h"
#include "src/pathops/SkOpContour.h"
#include "src/pathops/SkOpSegment.h"
#include "src/pathops/SkPathOpsCurve.h"

#include <utility>

enum class SkOpRayDir {
    kLeft,
    kTop,
    kRight,
    kBottom,
};

#if DEBUG_WINDING
const char* gDebugRayDirName[] = {
    "kLeft",
    "kTop",
    "kRight",
    "kBottom"
};
#endif

static int xy_index(SkOpRayDir dir) {
    return static_cast<int>(dir) & 1;
}

static SkScalar pt_xy(const SkPoint& pt, SkOpRayDir dir) {
    return (&pt.fX)[xy_index(dir)];
}

static SkScalar pt_yx(const SkPoint& pt, SkOpRayDir dir) {
    return (&pt.fX)[!xy_index(dir)];
}

static double pt_dxdy(const SkDVector& v, SkOpRayDir dir) {
    return (&v.fX)[xy_index(dir)];
}

static double pt_dydx(const SkDVector& v, SkOpRayDir dir) {
    return (&v.fX)[!xy_index(dir)];
}

static SkScalar rect_side(const SkRect& r, SkOpRayDir dir) {
    return (&r.fLeft)[static_cast<int>(dir)];
}

static bool sideways_overlap(const SkRect& rect, const SkPoint& pt, SkOpRayDir dir) {
    int i = !xy_index(dir);
    return approximately_between((&rect.fLeft)[i], (&pt.fX)[i], (&rect.fRight)[i]);
}

static bool less_than(SkOpRayDir dir) {
    return static_cast<bool>((static_cast<int>(dir) & 2) == 0);
}

static bool ccw_dxdy(const SkDVector& v, SkOpRayDir dir) {
    bool vPartPos = pt_dydx(v, dir) > 0;
    bool leftBottom = ((static_cast<int>(dir) + 1) & 2) != 0;
    return vPartPos == leftBottom;
}

struct SkOpRayHit {
    SkOpRayDir makeTestBase(SkOpSpan* span, double t) {
        fNext = nullptr;
        fSpan = span;
        fT = span->t() * (1 - t) + span->next()->t() * t;
        SkOpSegment* segment = span->segment();
        fSlope = segment->dSlopeAtT(fT);
        fPt = segment->ptAtT(fT);
        fValid = true;
        return fabs(fSlope.fX) < fabs(fSlope.fY) ? SkOpRayDir::kLeft : SkOpRayDir::kTop;
    }

    SkOpRayHit* fNext;
    SkOpSpan* fSpan;
    SkPoint fPt;
    double fT;
    SkDVector fSlope;
    bool fValid;
};

void SkOpContour::rayCheck(const SkOpRayHit& base, SkOpRayDir dir, SkOpRayHit** hits,
                           SkArenaAlloc* allocator) {
    // if the bounds extreme is outside the best, we're done
    SkScalar baseXY = pt_xy(base.fPt, dir);
    SkScalar boundsXY = rect_side(fBounds, dir);
    bool checkLessThan = less_than(dir);
    if (!approximately_equal(baseXY, boundsXY) && (baseXY < boundsXY) == checkLessThan) {
        return;
    }
    SkOpSegment* testSegment = &fHead;
    do {
        testSegment->rayCheck(base, dir, hits, allocator);
    } while ((testSegment = testSegment->next()));
}

void SkOpSegment::rayCheck(const SkOpRayHit& base, SkOpRayDir dir, SkOpRayHit** hits,
                           SkArenaAlloc* allocator) {
    if (!sideways_overlap(fBounds, base.fPt, dir)) {
        return;
    }
    SkScalar baseXY = pt_xy(base.fPt, dir);
    SkScalar boundsXY = rect_side(fBounds, dir);
    bool checkLessThan = less_than(dir);
    if (!approximately_equal(baseXY, boundsXY) && (baseXY < boundsXY) == checkLessThan) {
        return;
    }
    double tVals[3];
    SkScalar baseYX = pt_yx(base.fPt, dir);
    int roots = (*CurveIntercept[fVerb * 2 + xy_index(dir)])(fPts, fWeight, baseYX, tVals);
    for (int index = 0; index < roots; ++index) {
        double t = tVals[index];
        if (base.fSpan->segment() == this && approximately_equal(base.fT, t)) {
            continue;
        }
        SkDVector slope;
        SkPoint pt;
        SkDEBUGCODE(sk_bzero(&slope, sizeof(slope)));
        bool valid = false;
        if (approximately_zero(t)) {
            pt = fPts[0];
        } else if (approximately_equal(t, 1)) {
            pt = fPts[SkPathOpsVerbToPoints(fVerb)];
        } else {
            SkASSERT(between(0, t, 1));
            pt = this->ptAtT(t);
            if (SkDPoint::ApproximatelyEqual(pt, base.fPt)) {
                if (base.fSpan->segment() == this) {
                    continue;
                }
            } else {
                SkScalar ptXY = pt_xy(pt, dir);
                if (!approximately_equal(baseXY, ptXY) && (baseXY < ptXY) == checkLessThan) {
                    continue;
                }
                slope = this->dSlopeAtT(t);
                if (fVerb == SkPath::kCubic_Verb && base.fSpan->segment() == this
                        && roughly_equal(base.fT, t)
                        && SkDPoint::RoughlyEqual(pt, base.fPt)) {
    #if DEBUG_WINDING
                    SkDebugf("%s (rarely expect this)\n", __FUNCTION__);
    #endif
                    continue;
                }
                if (fabs(pt_dydx(slope, dir) * 10000) > fabs(pt_dxdy(slope, dir))) {
                    valid = true;
                }
            }
        }
        SkOpSpan* span = this->windingSpanAtT(t);
        if (!span) {
            valid = false;
        } else if (!span->windValue() && !span->oppValue()) {
            continue;
        }
        SkOpRayHit* newHit = allocator->make<SkOpRayHit>();
        newHit->fNext = *hits;
        newHit->fPt = pt;
        newHit->fSlope = slope;
        newHit->fSpan = span;
        newHit->fT = t;
        newHit->fValid = valid;
        *hits = newHit;
    }
}

SkOpSpan* SkOpSegment::windingSpanAtT(double tHit) {
    SkOpSpan* span = &fHead;
    SkOpSpanBase* next;
    do {
        next = span->next();
        if (approximately_equal(tHit, next->t())) {
            return nullptr;
        }
        if (tHit < next->t()) {
            return span;
        }
    } while (!next->final() && (span = next->upCast()));
    return nullptr;
}

static bool hit_compare_x(const SkOpRayHit* a, const SkOpRayHit* b) {
    return a->fPt.fX < b->fPt.fX;
}

static bool reverse_hit_compare_x(const SkOpRayHit* a, const SkOpRayHit* b) {
    return b->fPt.fX  < a->fPt.fX;
}

static bool hit_compare_y(const SkOpRayHit* a, const SkOpRayHit* b) {
    return a->fPt.fY < b->fPt.fY;
}

static bool reverse_hit_compare_y(const SkOpRayHit* a, const SkOpRayHit* b) {
    return b->fPt.fY  < a->fPt.fY;
}

static double get_t_guess(int tTry, int* dirOffset) {
    double t = 0.5;
    *dirOffset = tTry & 1;
    int tBase = tTry >> 1;
    int tBits = 0;
    while (tTry >>= 1) {
        t /= 2;
        ++tBits;
    }
    if (tBits) {
        int tIndex = (tBase - 1) & ((1 << tBits) - 1);
        t += t * 2 * tIndex;
    }
    return t;
}

bool SkOpSpan::sortableTop(SkOpContour* contourHead) {
    SkSTArenaAlloc<1024> allocator;
    int dirOffset;
    double t = get_t_guess(fTopTTry++, &dirOffset);
    SkOpRayHit hitBase;
    SkOpRayDir dir = hitBase.makeTestBase(this, t);
    if (hitBase.fSlope.fX == 0 && hitBase.fSlope.fY == 0) {
        return false;
    }
    SkOpRayHit* hitHead = &hitBase;
    dir = static_cast<SkOpRayDir>(static_cast<int>(dir) + dirOffset);
    if (hitBase.fSpan && hitBase.fSpan->segment()->verb() > SkPath::kLine_Verb
            && !pt_dydx(hitBase.fSlope, dir)) {
        return false;
    }
    SkOpContour* contour = contourHead;
    do {
        if (!contour->count()) {
            continue;
        }
        contour->rayCheck(hitBase, dir, &hitHead, &allocator);
    } while ((contour = contour->next()));
    // sort hits
    SkSTArray<1, SkOpRayHit*> sorted;
    SkOpRayHit* hit = hitHead;
    while (hit) {
        sorted.push_back(hit);
        hit = hit->fNext;
    }
    int count = sorted.count();
    SkTQSort(sorted.begin(), sorted.end(),
             xy_index(dir) ? less_than(dir) ? hit_compare_y : reverse_hit_compare_y
                           : less_than(dir) ? hit_compare_x : reverse_hit_compare_x);
    // verify windings
#if DEBUG_WINDING
    SkDebugf("%s dir=%s seg=%d t=%1.9g pt=(%1.9g,%1.9g)\n", __FUNCTION__,
            gDebugRayDirName[static_cast<int>(dir)], hitBase.fSpan->segment()->debugID(),
            hitBase.fT, hitBase.fPt.fX, hitBase.fPt.fY);
    for (int index = 0; index < count; ++index) {
        hit = sorted[index];
        SkOpSpan* span = hit->fSpan;
        SkOpSegment* hitSegment = span ? span->segment() : nullptr;
        bool operand = span ? hitSegment->operand() : false;
        bool ccw = ccw_dxdy(hit->fSlope, dir);
        SkDebugf("%s [%d] valid=%d operand=%d span=%d ccw=%d ", __FUNCTION__, index,
                hit->fValid, operand, span ? span->debugID() : -1, ccw);
        if (span) {
            hitSegment->dumpPtsInner();
        }
        SkDebugf(" t=%1.9g pt=(%1.9g,%1.9g) slope=(%1.9g,%1.9g)\n", hit->fT,
                hit->fPt.fX, hit->fPt.fY, hit->fSlope.fX, hit->fSlope.fY);
    }
#endif
    const SkPoint* last = nullptr;
    int wind = 0;
    int oppWind = 0;
    for (int index = 0; index < count; ++index) {
        hit = sorted[index];
        if (!hit->fValid) {
            return false;
        }
        bool ccw = ccw_dxdy(hit->fSlope, dir);
//        SkASSERT(!approximately_zero(hit->fT) || !hit->fValid);
        SkOpSpan* span = hit->fSpan;
        if (!span) {
            return false;
        }
        SkOpSegment* hitSegment = span->segment();
        if (span->windValue() == 0 && span->oppValue() == 0) {
            continue;
        }
        if (last && SkDPoint::ApproximatelyEqual(*last, hit->fPt)) {
            return false;
        }
        if (index < count - 1) {
            const SkPoint& next = sorted[index + 1]->fPt;
            if (SkDPoint::ApproximatelyEqual(next, hit->fPt)) {
                return false;
            }
        }
        bool operand = hitSegment->operand();
        if (operand) {
            using std::swap;
            swap(wind, oppWind);
        }
        int lastWind = wind;
        int lastOpp = oppWind;
        int windValue = ccw ? -span->windValue() : span->windValue();
        int oppValue = ccw ? -span->oppValue() : span->oppValue();
        wind += windValue;
        oppWind += oppValue;
        bool sumSet = false;
        int spanSum = span->windSum();
        int windSum = SkOpSegment::UseInnerWinding(lastWind, wind) ? wind : lastWind;
        if (spanSum == SK_MinS32) {
            span->setWindSum(windSum);
            sumSet = true;
        } else {
            // the need for this condition suggests that UseInnerWinding is flawed
            // happened when last = 1 wind = -1
#if 0
            SkASSERT((hitSegment->isXor() ? (windSum & 1) == (spanSum & 1) : windSum == spanSum)
                    || (abs(wind) == abs(lastWind)
                    && (windSum ^ wind ^ lastWind) == spanSum));
#endif
        }
        int oSpanSum = span->oppSum();
        int oppSum = SkOpSegment::UseInnerWinding(lastOpp, oppWind) ? oppWind : lastOpp;
        if (oSpanSum == SK_MinS32) {
            span->setOppSum(oppSum);
        } else {
#if 0
            SkASSERT(hitSegment->oppXor() ? (oppSum & 1) == (oSpanSum & 1) : oppSum == oSpanSum
                    || (abs(oppWind) == abs(lastOpp)
                    && (oppSum ^ oppWind ^ lastOpp) == oSpanSum));
#endif
        }
        if (sumSet) {
            if (this->globalState()->phase() == SkOpPhase::kFixWinding) {
                hitSegment->contour()->setCcw(ccw);
            } else {
                (void) hitSegment->markAndChaseWinding(span, span->next(), windSum, oppSum, nullptr);
                (void) hitSegment->markAndChaseWinding(span->next(), span, windSum, oppSum, nullptr);
            }
        }
        if (operand) {
            using std::swap;
            swap(wind, oppWind);
        }
        last = &hit->fPt;
        this->globalState()->bumpNested();
    }
    return true;
}

SkOpSpan* SkOpSegment::findSortableTop(SkOpContour* contourHead) {
    SkOpSpan* span = &fHead;
    SkOpSpanBase* next;
    do {
        next = span->next();
        if (span->done()) {
            continue;
        }
        if (span->windSum() != SK_MinS32) {
            return span;
        }
        if (span->sortableTop(contourHead)) {
            return span;
        }
    } while (!next->final() && (span = next->upCast()));
    return nullptr;
}

SkOpSpan* SkOpContour::findSortableTop(SkOpContour* contourHead) {
    bool allDone = true;
    if (fCount) {
        SkOpSegment* testSegment = &fHead;
        do {
            if (testSegment->done()) {
                continue;
            }
            allDone = false;
            SkOpSpan* result = testSegment->findSortableTop(contourHead);
            if (result) {
                return result;
            }
        } while ((testSegment = testSegment->next()));
    }
    if (allDone) {
      fDone = true;
    }
    return nullptr;
}

SkOpSpan* FindSortableTop(SkOpContourHead* contourHead) {
    for (int index = 0; index < SkOpGlobalState::kMaxWindingTries; ++index) {
        SkOpContour* contour = contourHead;
        do {
            if (contour->done()) {
                continue;
            }
            SkOpSpan* result = contour->findSortableTop(contourHead);
            if (result) {
                return result;
            }
        } while ((contour = contour->next()));
    }
    return nullptr;
}
