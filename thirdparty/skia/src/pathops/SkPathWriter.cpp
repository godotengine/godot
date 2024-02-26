/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#include "src/core/SkTSort.h"
#include "src/pathops/SkOpSegment.h"
#include "src/pathops/SkOpSpan.h"
#include "src/pathops/SkPathOpsPoint.h"
#include "src/pathops/SkPathWriter.h"

// wrap path to keep track of whether the contour is initialized and non-empty
SkPathWriter::SkPathWriter(SkPath& path)
    : fPathPtr(&path)
{
    init();
}

void SkPathWriter::close() {
    if (fCurrent.isEmpty()) {
        return;
    }
    SkASSERT(this->isClosed());
#if DEBUG_PATH_CONSTRUCTION
    SkDebugf("path.close();\n");
#endif
    fCurrent.close();
    fPathPtr->addPath(fCurrent);
    fCurrent.reset();
    init();
}

void SkPathWriter::conicTo(const SkPoint& pt1, const SkOpPtT* pt2, SkScalar weight) {
    SkPoint pt2pt = this->update(pt2);
#if DEBUG_PATH_CONSTRUCTION
    SkDebugf("path.conicTo(%1.9g,%1.9g, %1.9g,%1.9g, %1.9g);\n",
            pt1.fX, pt1.fY, pt2pt.fX, pt2pt.fY, weight);
#endif
    fCurrent.conicTo(pt1, pt2pt, weight);
}

void SkPathWriter::cubicTo(const SkPoint& pt1, const SkPoint& pt2, const SkOpPtT* pt3) {
    SkPoint pt3pt = this->update(pt3);
#if DEBUG_PATH_CONSTRUCTION
    SkDebugf("path.cubicTo(%1.9g,%1.9g, %1.9g,%1.9g, %1.9g,%1.9g);\n",
            pt1.fX, pt1.fY, pt2.fX, pt2.fY, pt3pt.fX, pt3pt.fY);
#endif
    fCurrent.cubicTo(pt1, pt2, pt3pt);
}

bool SkPathWriter::deferredLine(const SkOpPtT* pt) {
    SkASSERT(fFirstPtT);
    SkASSERT(fDefer[0]);
    if (fDefer[0] == pt) {
        // FIXME: why we're adding a degenerate line? Caller should have preflighted this.
        return true;
    }
    if (pt->contains(fDefer[0])) {
        // FIXME: why we're adding a degenerate line?
        return true;
    }
    if (this->matchedLast(pt)) {
        return false;
    }
    if (fDefer[1] && this->changedSlopes(pt)) {
        this->lineTo();
        fDefer[0] = fDefer[1];
    }
    fDefer[1] = pt;
    return true;
}

void SkPathWriter::deferredMove(const SkOpPtT* pt) {
    if (!fDefer[1]) {
        fFirstPtT = fDefer[0] = pt;
        return;
    }
    SkASSERT(fDefer[0]);
    if (!this->matchedLast(pt)) {
        this->finishContour();
        fFirstPtT = fDefer[0] = pt;
    }
}

void SkPathWriter::finishContour() {
    if (!this->matchedLast(fDefer[0])) {
        if (!fDefer[1]) {
          return;
        }
        this->lineTo();
    }
    if (fCurrent.isEmpty()) {
        return;
    }
    if (this->isClosed()) {
        this->close();
    } else {
        SkASSERT(fDefer[1]);
        fEndPtTs.push_back(fFirstPtT);
        fEndPtTs.push_back(fDefer[1]);
        fPartials.push_back(fCurrent);
        this->init();
    }
}

void SkPathWriter::init() {
    fCurrent.reset();
    fFirstPtT = fDefer[0] = fDefer[1] = nullptr;
}

bool SkPathWriter::isClosed() const {
    return this->matchedLast(fFirstPtT);
}

void SkPathWriter::lineTo() {
    if (fCurrent.isEmpty()) {
        this->moveTo();
    }
#if DEBUG_PATH_CONSTRUCTION
    SkDebugf("path.lineTo(%1.9g,%1.9g);\n", fDefer[1]->fPt.fX, fDefer[1]->fPt.fY);
#endif
    fCurrent.lineTo(fDefer[1]->fPt);
}

bool SkPathWriter::matchedLast(const SkOpPtT* test) const {
    if (test == fDefer[1]) {
        return true;
    }
    if (!test) {
        return false;
    }
    if (!fDefer[1]) {
        return false;
    }
    return test->contains(fDefer[1]);
}

void SkPathWriter::moveTo() {
#if DEBUG_PATH_CONSTRUCTION
    SkDebugf("path.moveTo(%1.9g,%1.9g);\n", fFirstPtT->fPt.fX, fFirstPtT->fPt.fY);
#endif
    fCurrent.moveTo(fFirstPtT->fPt);
}

void SkPathWriter::quadTo(const SkPoint& pt1, const SkOpPtT* pt2) {
    SkPoint pt2pt = this->update(pt2);
#if DEBUG_PATH_CONSTRUCTION
    SkDebugf("path.quadTo(%1.9g,%1.9g, %1.9g,%1.9g);\n",
            pt1.fX, pt1.fY, pt2pt.fX, pt2pt.fY);
#endif
    fCurrent.quadTo(pt1, pt2pt);
}

// if last point to be written matches the current path's first point, alter the
// last to avoid writing a degenerate lineTo when the path is closed
SkPoint SkPathWriter::update(const SkOpPtT* pt) {
    if (!fDefer[1]) {
        this->moveTo();
    } else if (!this->matchedLast(fDefer[0])) {
        this->lineTo();
    }
    SkPoint result = pt->fPt;
    if (fFirstPtT && result != fFirstPtT->fPt && fFirstPtT->contains(pt)) {
        result = fFirstPtT->fPt;
    }
    fDefer[0] = fDefer[1] = pt;  // set both to know that there is not a pending deferred line
    return result;
}

bool SkPathWriter::someAssemblyRequired() {
    this->finishContour();
    return fEndPtTs.count() > 0;
}

bool SkPathWriter::changedSlopes(const SkOpPtT* ptT) const {
    if (matchedLast(fDefer[0])) {
        return false;
    }
    SkVector deferDxdy = fDefer[1]->fPt - fDefer[0]->fPt;
    SkVector lineDxdy = ptT->fPt - fDefer[1]->fPt;
    return deferDxdy.fX * lineDxdy.fY != deferDxdy.fY * lineDxdy.fX;
}

class DistanceLessThan {
public:
    DistanceLessThan(double* distances) : fDistances(distances) { }
    double* fDistances;
    bool operator()(const int one, const int two) const {
        return fDistances[one] < fDistances[two];
    }
};

    /*
        check start and end of each contour
        if not the same, record them
        match them up
        connect closest
        reassemble contour pieces into new path
    */
void SkPathWriter::assemble() {
    if (!this->someAssemblyRequired()) {
        return;
    }
#if DEBUG_PATH_CONSTRUCTION
    SkDebugf("%s\n", __FUNCTION__);
#endif
    SkOpPtT const* const* runs = fEndPtTs.begin();  // starts, ends of partial contours
    int endCount = fEndPtTs.count(); // all starts and ends
    SkASSERT(endCount > 0);
    SkASSERT(endCount == fPartials.count() * 2);
#if DEBUG_ASSEMBLE
    for (int index = 0; index < endCount; index += 2) {
        const SkOpPtT* eStart = runs[index];
        const SkOpPtT* eEnd = runs[index + 1];
        SkASSERT(eStart != eEnd);
        SkASSERT(!eStart->contains(eEnd));
        SkDebugf("%s contour start=(%1.9g,%1.9g) end=(%1.9g,%1.9g)\n", __FUNCTION__,
                eStart->fPt.fX, eStart->fPt.fY, eEnd->fPt.fX, eEnd->fPt.fY);
    }
#endif
    // lengthen any partial contour adjacent to a simple segment
    for (int pIndex = 0; pIndex < endCount; pIndex++) {
        SkOpPtT* opPtT = const_cast<SkOpPtT*>(runs[pIndex]);
        SkPath p;
        SkPathWriter partWriter(p);
        do {
            if (!zero_or_one(opPtT->fT)) {
                break;
            }
            SkOpSpanBase* opSpanBase = opPtT->span();
            SkOpSpanBase* start = opPtT->fT ? opSpanBase->prev() : opSpanBase->upCast()->next();
            int step = opPtT->fT ? 1 : -1;
            const SkOpSegment* opSegment = opSpanBase->segment();
            const SkOpSegment* nextSegment = opSegment->isSimple(&start, &step);
            if (!nextSegment) {
                break;
            }
            SkOpSpanBase* opSpanEnd = start->t() ? start->prev() : start->upCast()->next();
            if (start->starter(opSpanEnd)->alreadyAdded()) {
                break;
            }
            nextSegment->addCurveTo(start, opSpanEnd, &partWriter);
            opPtT = opSpanEnd->ptT();
            SkOpPtT** runsPtr = const_cast<SkOpPtT**>(&runs[pIndex]);
            *runsPtr = opPtT;
        } while (true);
        partWriter.finishContour();
        const SkTArray<SkPath>& partPartials = partWriter.partials();
        if (!partPartials.count()) {
            continue;
        }
        // if pIndex is even, reverse and prepend to fPartials; otherwise, append
        SkPath& partial = const_cast<SkPath&>(fPartials[pIndex >> 1]);
        const SkPath& part = partPartials[0];
        if (pIndex & 1) {
            partial.addPath(part, SkPath::kExtend_AddPathMode);
        } else {
            SkPath reverse;
            reverse.reverseAddPath(part);
            reverse.addPath(partial, SkPath::kExtend_AddPathMode);
            partial = reverse;
        }
    }
    SkTDArray<int> sLink, eLink;
    int linkCount = endCount / 2; // number of partial contours
    sLink.append(linkCount);
    eLink.append(linkCount);
    int rIndex, iIndex;
    for (rIndex = 0; rIndex < linkCount; ++rIndex) {
        sLink[rIndex] = eLink[rIndex] = SK_MaxS32;
    }
    const int entries = endCount * (endCount - 1) / 2;  // folded triangle
    SkSTArray<8, double, true> distances(entries);
    SkSTArray<8, int, true> sortedDist(entries);
    SkSTArray<8, int, true> distLookup(entries);
    int rRow = 0;
    int dIndex = 0;
    for (rIndex = 0; rIndex < endCount - 1; ++rIndex) {
        const SkOpPtT* oPtT = runs[rIndex];
        for (iIndex = rIndex + 1; iIndex < endCount; ++iIndex) {
            const SkOpPtT* iPtT = runs[iIndex];
            double dx = iPtT->fPt.fX - oPtT->fPt.fX;
            double dy = iPtT->fPt.fY - oPtT->fPt.fY;
            double dist = dx * dx + dy * dy;
            distLookup.push_back(rRow + iIndex);
            distances.push_back(dist);  // oStart distance from iStart
            sortedDist.push_back(dIndex++);
        }
        rRow += endCount;
    }
    SkASSERT(dIndex == entries);
    SkTQSort<int>(sortedDist.begin(), sortedDist.end(), DistanceLessThan(distances.begin()));
    int remaining = linkCount;  // number of start/end pairs
    for (rIndex = 0; rIndex < entries; ++rIndex) {
        int pair = sortedDist[rIndex];
        pair = distLookup[pair];
        int row = pair / endCount;
        int col = pair - row * endCount;
        int ndxOne = row >> 1;
        bool endOne = row & 1;
        int* linkOne = endOne ? eLink.begin() : sLink.begin();
        if (linkOne[ndxOne] != SK_MaxS32) {
            continue;
        }
        int ndxTwo = col >> 1;
        bool endTwo = col & 1;
        int* linkTwo = endTwo ? eLink.begin() : sLink.begin();
        if (linkTwo[ndxTwo] != SK_MaxS32) {
            continue;
        }
        SkASSERT(&linkOne[ndxOne] != &linkTwo[ndxTwo]);
        bool flip = endOne == endTwo;
        linkOne[ndxOne] = flip ? ~ndxTwo : ndxTwo;
        linkTwo[ndxTwo] = flip ? ~ndxOne : ndxOne;
        if (!--remaining) {
            break;
        }
    }
    SkASSERT(!remaining);
#if DEBUG_ASSEMBLE
    for (rIndex = 0; rIndex < linkCount; ++rIndex) {
        int s = sLink[rIndex];
        int e = eLink[rIndex];
        SkDebugf("%s %c%d <- s%d - e%d -> %c%d\n", __FUNCTION__, s < 0 ? 's' : 'e',
                s < 0 ? ~s : s, rIndex, rIndex, e < 0 ? 'e' : 's', e < 0 ? ~e : e);
    }
#endif
    rIndex = 0;
    do {
        bool forward = true;
        bool first = true;
        int sIndex = sLink[rIndex];
        SkASSERT(sIndex != SK_MaxS32);
        sLink[rIndex] = SK_MaxS32;
        int eIndex;
        if (sIndex < 0) {
            eIndex = sLink[~sIndex];
            sLink[~sIndex] = SK_MaxS32;
        } else {
            eIndex = eLink[sIndex];
            eLink[sIndex] = SK_MaxS32;
        }
        SkASSERT(eIndex != SK_MaxS32);
#if DEBUG_ASSEMBLE
        SkDebugf("%s sIndex=%c%d eIndex=%c%d\n", __FUNCTION__, sIndex < 0 ? 's' : 'e',
                    sIndex < 0 ? ~sIndex : sIndex, eIndex < 0 ? 's' : 'e',
                    eIndex < 0 ? ~eIndex : eIndex);
#endif
        do {
            const SkPath& contour = fPartials[rIndex];
            if (!first) {
                SkPoint prior, next;
                if (!fPathPtr->getLastPt(&prior)) {
                    return;
                }
                if (forward) {
                    next = contour.getPoint(0);
                } else {
                    SkAssertResult(contour.getLastPt(&next));
                }
                if (prior != next) {
                    /* TODO: if there is a gap between open path written so far and path to come,
                       connect by following segments from one to the other, rather than introducing
                       a diagonal to connect the two.
                     */
                }
            }
            if (forward) {
                fPathPtr->addPath(contour,
                        first ? SkPath::kAppend_AddPathMode : SkPath::kExtend_AddPathMode);
            } else {
                SkASSERT(!first);
                fPathPtr->reversePathTo(contour);
            }
            if (first) {
                first = false;
            }
#if DEBUG_ASSEMBLE
            SkDebugf("%s rIndex=%d eIndex=%s%d close=%d\n", __FUNCTION__, rIndex,
                eIndex < 0 ? "~" : "", eIndex < 0 ? ~eIndex : eIndex,
                sIndex == ((rIndex != eIndex) ^ forward ? eIndex : ~eIndex));
#endif
            if (sIndex == ((rIndex != eIndex) ^ forward ? eIndex : ~eIndex)) {
                fPathPtr->close();
                break;
            }
            if (forward) {
                eIndex = eLink[rIndex];
                SkASSERT(eIndex != SK_MaxS32);
                eLink[rIndex] = SK_MaxS32;
                if (eIndex >= 0) {
                    SkASSERT(sLink[eIndex] == rIndex);
                    sLink[eIndex] = SK_MaxS32;
                } else {
                    SkASSERT(eLink[~eIndex] == ~rIndex);
                    eLink[~eIndex] = SK_MaxS32;
                }
            } else {
                eIndex = sLink[rIndex];
                SkASSERT(eIndex != SK_MaxS32);
                sLink[rIndex] = SK_MaxS32;
                if (eIndex >= 0) {
                    SkASSERT(eLink[eIndex] == rIndex);
                    eLink[eIndex] = SK_MaxS32;
                } else {
                    SkASSERT(sLink[~eIndex] == ~rIndex);
                    sLink[~eIndex] = SK_MaxS32;
                }
            }
            rIndex = eIndex;
            if (rIndex < 0) {
                forward ^= 1;
                rIndex = ~rIndex;
            }
        } while (true);
        for (rIndex = 0; rIndex < linkCount; ++rIndex) {
            if (sLink[rIndex] != SK_MaxS32) {
                break;
            }
        }
    } while (rIndex < linkCount);
#if DEBUG_ASSEMBLE
    for (rIndex = 0; rIndex < linkCount; ++rIndex) {
       SkASSERT(sLink[rIndex] == SK_MaxS32);
       SkASSERT(eLink[rIndex] == SK_MaxS32);
    }
#endif
    return;
}
