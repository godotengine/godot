/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "include/private/SkMacros.h"
#include "src/core/SkTSort.h"
#include "src/pathops/SkAddIntersections.h"
#include "src/pathops/SkOpCoincidence.h"
#include "src/pathops/SkOpEdgeBuilder.h"
#include "src/pathops/SkPathOpsCommon.h"
#include "src/pathops/SkPathWriter.h"

const SkOpAngle* AngleWinding(SkOpSpanBase* start, SkOpSpanBase* end, int* windingPtr,
        bool* sortablePtr) {
    // find first angle, initialize winding to computed fWindSum
    SkOpSegment* segment = start->segment();
    const SkOpAngle* angle = segment->spanToAngle(start, end);
    if (!angle) {
        *windingPtr = SK_MinS32;
        return nullptr;
    }
    bool computeWinding = false;
    const SkOpAngle* firstAngle = angle;
    bool loop = false;
    bool unorderable = false;
    int winding = SK_MinS32;
    do {
        angle = angle->next();
        if (!angle) {
            return nullptr;
        }
        unorderable |= angle->unorderable();
        if ((computeWinding = unorderable || (angle == firstAngle && loop))) {
            break;    // if we get here, there's no winding, loop is unorderable
        }
        loop |= angle == firstAngle;
        segment = angle->segment();
        winding = segment->windSum(angle);
    } while (winding == SK_MinS32);
    // if the angle loop contains an unorderable span, the angle order may be useless
    // directly compute the winding in this case for each span
    if (computeWinding) {
        firstAngle = angle;
        winding = SK_MinS32;
        do {
            SkOpSpanBase* startSpan = angle->start();
            SkOpSpanBase* endSpan = angle->end();
            SkOpSpan* lesser = startSpan->starter(endSpan);
            int testWinding = lesser->windSum();
            if (testWinding == SK_MinS32) {
                testWinding = lesser->computeWindSum();
            }
            if (testWinding != SK_MinS32) {
                segment = angle->segment();
                winding = testWinding;
            }
            angle = angle->next();
        } while (angle != firstAngle);
    }
    *sortablePtr = !unorderable;
    *windingPtr = winding;
    return angle;
}

SkOpSpan* FindUndone(SkOpContourHead* contourHead) {
    SkOpContour* contour = contourHead;
    do {
        if (contour->done()) {
            continue;
        }
        SkOpSpan* result = contour->undoneSpan();
        if (result) {
            return result;
        }
    } while ((contour = contour->next()));
    return nullptr;
}

SkOpSegment* FindChase(SkTDArray<SkOpSpanBase*>* chase, SkOpSpanBase** startPtr,
        SkOpSpanBase** endPtr) {
    while (chase->count()) {
        SkOpSpanBase* span;
        chase->pop(&span);
        SkOpSegment* segment = span->segment();
        *startPtr = span->ptT()->next()->span();
        bool done = true;
        *endPtr = nullptr;
        if (SkOpAngle* last = segment->activeAngle(*startPtr, startPtr, endPtr, &done)) {
            *startPtr = last->start();
            *endPtr = last->end();
    #if TRY_ROTATE
            *chase->insert(0) = span;
    #else
            *chase->append() = span;
    #endif
            return last->segment();
        }
        if (done) {
            continue;
        }
        // find first angle, initialize winding to computed wind sum
        int winding;
        bool sortable;
        const SkOpAngle* angle = AngleWinding(*startPtr, *endPtr, &winding, &sortable);
        if (!angle) {
            return nullptr;
        }
        if (winding == SK_MinS32) {
            continue;
        }
        int sumWinding SK_INIT_TO_AVOID_WARNING;
        if (sortable) {
            segment = angle->segment();
            sumWinding = segment->updateWindingReverse(angle);
        }
        SkOpSegment* first = nullptr;
        const SkOpAngle* firstAngle = angle;
        while ((angle = angle->next()) != firstAngle) {
            segment = angle->segment();
            SkOpSpanBase* start = angle->start();
            SkOpSpanBase* end = angle->end();
            int maxWinding SK_INIT_TO_AVOID_WARNING;
            if (sortable) {
                segment->setUpWinding(start, end, &maxWinding, &sumWinding);
            }
            if (!segment->done(angle)) {
                if (!first && (sortable || start->starter(end)->windSum() != SK_MinS32)) {
                    first = segment;
                    *startPtr = start;
                    *endPtr = end;
                }
                // OPTIMIZATION: should this also add to the chase?
                if (sortable) {
                    // TODO: add error handling
                    SkAssertResult(segment->markAngle(maxWinding, sumWinding, angle, nullptr));
                }
            }
        }
        if (first) {
       #if TRY_ROTATE
            *chase->insert(0) = span;
       #else
            *chase->append() = span;
       #endif
            return first;
        }
    }
    return nullptr;
}

bool SortContourList(SkOpContourHead** contourList, bool evenOdd, bool oppEvenOdd) {
    SkTDArray<SkOpContour* > list;
    SkOpContour* contour = *contourList;
    do {
        if (contour->count()) {
            contour->setOppXor(contour->operand() ? evenOdd : oppEvenOdd);
            *list.append() = contour;
        }
    } while ((contour = contour->next()));
    int count = list.count();
    if (!count) {
        return false;
    }
    if (count > 1) {
        SkTQSort<SkOpContour>(list.begin(), list.end());
    }
    contour = list[0];
    SkOpContourHead* contourHead = static_cast<SkOpContourHead*>(contour);
    contour->globalState()->setContourHead(contourHead);
    *contourList = contourHead;
    for (int index = 1; index < count; ++index) {
        SkOpContour* next = list[index];
        contour->setNext(next);
        contour = next;
    }
    contour->setNext(nullptr);
    return true;
}

static void calc_angles(SkOpContourHead* contourList  DEBUG_COIN_DECLARE_PARAMS()) {
    DEBUG_STATIC_SET_PHASE(contourList);
    SkOpContour* contour = contourList;
    do {
        contour->calcAngles();
    } while ((contour = contour->next()));
}

static bool missing_coincidence(SkOpContourHead* contourList  DEBUG_COIN_DECLARE_PARAMS()) {
    DEBUG_STATIC_SET_PHASE(contourList);
    SkOpContour* contour = contourList;
    bool result = false;
    do {
        result |= contour->missingCoincidence();
    } while ((contour = contour->next()));
    return result;
}

static bool move_multiples(SkOpContourHead* contourList  DEBUG_COIN_DECLARE_PARAMS()) {
    DEBUG_STATIC_SET_PHASE(contourList);
    SkOpContour* contour = contourList;
    do {
        if (!contour->moveMultiples()) {
            return false;
        }
    } while ((contour = contour->next()));
    return true;
}

static bool move_nearby(SkOpContourHead* contourList  DEBUG_COIN_DECLARE_PARAMS()) {
    DEBUG_STATIC_SET_PHASE(contourList);
    SkOpContour* contour = contourList;
    do {
        if (!contour->moveNearby()) {
            return false;
        }
    } while ((contour = contour->next()));
    return true;
}

static bool sort_angles(SkOpContourHead* contourList) {
    SkOpContour* contour = contourList;
    do {
        if (!contour->sortAngles()) {
            return false;
        }
    } while ((contour = contour->next()));
    return true;
}

bool HandleCoincidence(SkOpContourHead* contourList, SkOpCoincidence* coincidence) {
    SkOpGlobalState* globalState = contourList->globalState();
    // match up points within the coincident runs
    if (!coincidence->addExpanded(DEBUG_PHASE_ONLY_PARAMS(kIntersecting))) {
        return false;
    }
    // combine t values when multiple intersections occur on some segments but not others
    if (!move_multiples(contourList  DEBUG_PHASE_PARAMS(kWalking))) {
        return false;
    }
    // move t values and points together to eliminate small/tiny gaps
    if (!move_nearby(contourList  DEBUG_COIN_PARAMS())) {
        return false;
    }
    // add coincidence formed by pairing on curve points and endpoints
    coincidence->correctEnds(DEBUG_PHASE_ONLY_PARAMS(kIntersecting));
    if (!coincidence->addEndMovedSpans(DEBUG_COIN_ONLY_PARAMS())) {
        return false;
    }
    const int SAFETY_COUNT = 3;
    int safetyHatch = SAFETY_COUNT;
    // look for coincidence present in A-B and A-C but missing in B-C
    do {
        bool added;
        if (!coincidence->addMissing(&added  DEBUG_ITER_PARAMS(SAFETY_COUNT - safetyHatch))) {
            return false;
        }
        if (!added) {
            break;
        }
        if (!--safetyHatch) {
            SkASSERT(globalState->debugSkipAssert());
            return false;
        }
        move_nearby(contourList  DEBUG_ITER_PARAMS(SAFETY_COUNT - safetyHatch - 1));
    } while (true);
    // check to see if, loosely, coincident ranges may be expanded
    if (coincidence->expand(DEBUG_COIN_ONLY_PARAMS())) {
        bool added;
        if (!coincidence->addMissing(&added  DEBUG_COIN_PARAMS())) {
            return false;
        }
        if (!coincidence->addExpanded(DEBUG_COIN_ONLY_PARAMS())) {
            return false;
        }
        if (!move_multiples(contourList  DEBUG_COIN_PARAMS())) {
            return false;
        }
        move_nearby(contourList  DEBUG_COIN_PARAMS());
    }
    // the expanded ranges may not align -- add the missing spans
    if (!coincidence->addExpanded(DEBUG_PHASE_ONLY_PARAMS(kWalking))) {
        return false;
    }
    // mark spans of coincident segments as coincident
    coincidence->mark(DEBUG_COIN_ONLY_PARAMS());
    // look for coincidence lines and curves undetected by intersection
    if (missing_coincidence(contourList  DEBUG_COIN_PARAMS())) {
        (void) coincidence->expand(DEBUG_PHASE_ONLY_PARAMS(kIntersecting));
        if (!coincidence->addExpanded(DEBUG_COIN_ONLY_PARAMS())) {
            return false;
        }
        if (!coincidence->mark(DEBUG_PHASE_ONLY_PARAMS(kWalking))) {
            return false;
        }
    } else {
        (void) coincidence->expand(DEBUG_COIN_ONLY_PARAMS());
    }
    (void) coincidence->expand(DEBUG_COIN_ONLY_PARAMS());

    SkOpCoincidence overlaps(globalState);
    safetyHatch = SAFETY_COUNT;
    do {
        SkOpCoincidence* pairs = overlaps.isEmpty() ? coincidence : &overlaps;
        // adjust the winding value to account for coincident edges
        if (!pairs->apply(DEBUG_ITER_ONLY_PARAMS(SAFETY_COUNT - safetyHatch))) {
            return false;
        }
        // For each coincident pair that overlaps another, when the receivers (the 1st of the pair)
        // are different, construct a new pair to resolve their mutual span
        if (!pairs->findOverlaps(&overlaps  DEBUG_ITER_PARAMS(SAFETY_COUNT - safetyHatch))) {
            return false;
        }
        if (!--safetyHatch) {
            SkASSERT(globalState->debugSkipAssert());
            return false;
        }
    } while (!overlaps.isEmpty());
    calc_angles(contourList  DEBUG_COIN_PARAMS());
    if (!sort_angles(contourList)) {
        return false;
    }
#if DEBUG_COINCIDENCE_VERBOSE
    coincidence->debugShowCoincidence();
#endif
#if DEBUG_COINCIDENCE
    coincidence->debugValidate();
#endif
    SkPathOpsDebug::ShowActiveSpans(contourList);
    return true;
}
