/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#include "src/pathops/SkAddIntersections.h"
#include "src/pathops/SkOpCoincidence.h"
#include "src/pathops/SkOpEdgeBuilder.h"
#include "src/pathops/SkPathOpsCommon.h"
#include "src/pathops/SkPathWriter.h"

#include <utility>

static bool findChaseOp(SkTDArray<SkOpSpanBase*>& chase, SkOpSpanBase** startPtr,
        SkOpSpanBase** endPtr, SkOpSegment** result) {
    while (chase.count()) {
        SkOpSpanBase* span;
        chase.pop(&span);
        // OPTIMIZE: prev makes this compatible with old code -- but is it necessary?
        *startPtr = span->ptT()->prev()->span();
        SkOpSegment* segment = (*startPtr)->segment();
        bool done = true;
        *endPtr = nullptr;
        if (SkOpAngle* last = segment->activeAngle(*startPtr, startPtr, endPtr, &done)) {
            *startPtr = last->start();
            *endPtr = last->end();
   #if TRY_ROTATE
            *chase.insert(0) = span;
   #else
            *chase.append() = span;
   #endif
            *result = last->segment();
            return true;
        }
        if (done) {
            continue;
        }
        int winding;
        bool sortable;
        const SkOpAngle* angle = AngleWinding(*startPtr, *endPtr, &winding, &sortable);
        if (!angle) {
            *result = nullptr;
            return true;
        }
        if (winding == SK_MinS32) {
            continue;
        }
        int sumMiWinding, sumSuWinding;
        if (sortable) {
            segment = angle->segment();
            sumMiWinding = segment->updateWindingReverse(angle);
            if (sumMiWinding == SK_MinS32) {
                SkASSERT(segment->globalState()->debugSkipAssert());
                *result = nullptr;
                return true;
            }
            sumSuWinding = segment->updateOppWindingReverse(angle);
            if (sumSuWinding == SK_MinS32) {
                SkASSERT(segment->globalState()->debugSkipAssert());
                *result = nullptr;
                return true;
            }
            if (segment->operand()) {
                using std::swap;
                swap(sumMiWinding, sumSuWinding);
            }
        }
        SkOpSegment* first = nullptr;
        const SkOpAngle* firstAngle = angle;
        while ((angle = angle->next()) != firstAngle) {
            segment = angle->segment();
            SkOpSpanBase* start = angle->start();
            SkOpSpanBase* end = angle->end();
            int maxWinding = 0, sumWinding = 0, oppMaxWinding = 0, oppSumWinding = 0;
            if (sortable) {
                segment->setUpWindings(start, end, &sumMiWinding, &sumSuWinding,
                        &maxWinding, &sumWinding, &oppMaxWinding, &oppSumWinding);
            }
            if (!segment->done(angle)) {
                if (!first && (sortable || start->starter(end)->windSum() != SK_MinS32)) {
                    first = segment;
                    *startPtr = start;
                    *endPtr = end;
                }
                // OPTIMIZATION: should this also add to the chase?
                if (sortable) {
                    if (!segment->markAngle(maxWinding, sumWinding, oppMaxWinding,
                            oppSumWinding, angle, nullptr)) {
                        return false;
                    }
                }
            }
        }
        if (first) {
       #if TRY_ROTATE
            *chase.insert(0) = span;
       #else
            *chase.append() = span;
       #endif
            *result = first;
            return true;
        }
    }
    *result = nullptr;
    return true;
}

static bool bridgeOp(SkOpContourHead* contourList, const SkPathOp op,
        const int xorMask, const int xorOpMask, SkPathWriter* writer) {
    bool unsortable = false;
    bool lastSimple = false;
    bool simple = false;
    do {
        SkOpSpan* span = FindSortableTop(contourList);
        if (!span) {
            break;
        }
        SkOpSegment* current = span->segment();
        SkOpSpanBase* start = span->next();
        SkOpSpanBase* end = span;
        SkTDArray<SkOpSpanBase*> chase;
        do {
            if (current->activeOp(start, end, xorMask, xorOpMask, op)) {
                do {
                    if (!unsortable && current->done()) {
                        break;
                    }
                    SkASSERT(unsortable || !current->done());
                    SkOpSpanBase* nextStart = start;
                    SkOpSpanBase* nextEnd = end;
                    lastSimple = simple;
                    SkOpSegment* next = current->findNextOp(&chase, &nextStart, &nextEnd,
                            &unsortable, &simple, op, xorMask, xorOpMask);
                    if (!next) {
                        if (!unsortable && writer->hasMove()
                                && current->verb() != SkPath::kLine_Verb
                                && !writer->isClosed()) {
                            if (!current->addCurveTo(start, end, writer)) {
                                return false;
                            }
                            if (!writer->isClosed()) {
                                SkPathOpsDebug::ShowActiveSpans(contourList);
                            }
                        } else if (lastSimple) {
                            if (!current->addCurveTo(start, end, writer)) {
                                return false;
                            }
                        }
                        break;
                    }
        #if DEBUG_FLOW
                    SkDebugf("%s current id=%d from=(%1.9g,%1.9g) to=(%1.9g,%1.9g)\n", __FUNCTION__,
                            current->debugID(), start->pt().fX, start->pt().fY,
                            end->pt().fX, end->pt().fY);
        #endif
                    if (!current->addCurveTo(start, end, writer)) {
                        return false;
                    }
                    current = next;
                    start = nextStart;
                    end = nextEnd;
                } while (!writer->isClosed() && (!unsortable || !start->starter(end)->done()));
                if (current->activeWinding(start, end) && !writer->isClosed()) {
                    SkOpSpan* spanStart = start->starter(end);
                    if (!spanStart->done()) {
                        if (!current->addCurveTo(start, end, writer)) {
                            return false;
                        }
                        current->markDone(spanStart);
                    }
                }
                writer->finishContour();
            } else {
                SkOpSpanBase* last;
                if (!current->markAndChaseDone(start, end, &last)) {
                    return false;
                }
                if (last && !last->chased()) {
                    last->setChased(true);
                    SkASSERT(!SkPathOpsDebug::ChaseContains(chase, last));
                    *chase.append() = last;
#if DEBUG_WINDING
                    SkDebugf("%s chase.append id=%d", __FUNCTION__, last->segment()->debugID());
                    if (!last->final()) {
                         SkDebugf(" windSum=%d", last->upCast()->windSum());
                    }
                    SkDebugf("\n");
#endif
                }
            }
            if (!findChaseOp(chase, &start, &end, &current)) {
                return false;
            }
            SkPathOpsDebug::ShowActiveSpans(contourList);
            if (!current) {
                break;
            }
        } while (true);
    } while (true);
    return true;
}

// diagram of why this simplifcation is possible is here:
// https://skia.org/dev/present/pathops link at bottom of the page
// https://drive.google.com/file/d/0BwoLUwz9PYkHLWpsaXd0UDdaN00/view?usp=sharing
static const SkPathOp gOpInverse[kReverseDifference_SkPathOp + 1][2][2] = {
//                  inside minuend                               outside minuend
//     inside subtrahend     outside subtrahend      inside subtrahend     outside subtrahend
{{ kDifference_SkPathOp,   kIntersect_SkPathOp }, { kUnion_SkPathOp, kReverseDifference_SkPathOp }},
{{ kIntersect_SkPathOp,   kDifference_SkPathOp }, { kReverseDifference_SkPathOp, kUnion_SkPathOp }},
{{ kUnion_SkPathOp, kReverseDifference_SkPathOp }, { kDifference_SkPathOp,   kIntersect_SkPathOp }},
{{ kXOR_SkPathOp,                 kXOR_SkPathOp }, { kXOR_SkPathOp,                kXOR_SkPathOp }},
{{ kReverseDifference_SkPathOp, kUnion_SkPathOp }, { kIntersect_SkPathOp,   kDifference_SkPathOp }},
};

static const bool gOutInverse[kReverseDifference_SkPathOp + 1][2][2] = {
    {{ false, false }, { true, false }},  // diff
    {{ false, false }, { false, true }},  // sect
    {{ false, true }, { true, true }},    // union
    {{ false, true }, { true, false }},   // xor
    {{ false, true }, { false, false }},  // rev diff
};

#if DEBUG_T_SECT_LOOP_COUNT

#include "include/private/SkMutex.h"

SkOpGlobalState debugWorstState(nullptr, nullptr  SkDEBUGPARAMS(false) SkDEBUGPARAMS(nullptr));

void ReportPathOpsDebugging() {
    debugWorstState.debugLoopReport();
}

extern void (*gVerboseFinalize)();

#endif

bool OpDebug(const SkPath& one, const SkPath& two, SkPathOp op, SkPath* result
        SkDEBUGPARAMS(bool skipAssert) SkDEBUGPARAMS(const char* testName)) {
#if DEBUG_DUMP_VERIFY
#ifndef SK_DEBUG
    const char* testName = "release";
#endif
    if (SkPathOpsDebug::gDumpOp) {
        SkPathOpsDebug::DumpOp(one, two, op, testName);
    }
#endif
    op = gOpInverse[op][one.isInverseFillType()][two.isInverseFillType()];
    bool inverseFill = gOutInverse[op][one.isInverseFillType()][two.isInverseFillType()];
    SkPathFillType fillType = inverseFill ? SkPathFillType::kInverseEvenOdd :
            SkPathFillType::kEvenOdd;
    SkRect rect1, rect2;
    if (kIntersect_SkPathOp == op && one.isRect(&rect1) && two.isRect(&rect2)) {
        result->reset();
        result->setFillType(fillType);
        if (rect1.intersect(rect2)) {
            result->addRect(rect1);
        }
        return true;
    }
    if (one.isEmpty() || two.isEmpty()) {
        SkPath work;
        switch (op) {
            case kIntersect_SkPathOp:
                break;
            case kUnion_SkPathOp:
            case kXOR_SkPathOp:
                work = one.isEmpty() ? two : one;
                break;
            case kDifference_SkPathOp:
                if (!one.isEmpty()) {
                    work = one;
                }
                break;
            case kReverseDifference_SkPathOp:
                if (!two.isEmpty()) {
                    work = two;
                }
                break;
            default:
                SkASSERT(0);  // unhandled case
        }
        if (inverseFill != work.isInverseFillType()) {
            work.toggleInverseFillType();
        }
        return Simplify(work, result);
    }
    SkSTArenaAlloc<4096> allocator;  // FIXME: add a constant expression here, tune
    SkOpContour contour;
    SkOpContourHead* contourList = static_cast<SkOpContourHead*>(&contour);
    SkOpGlobalState globalState(contourList, &allocator
            SkDEBUGPARAMS(skipAssert) SkDEBUGPARAMS(testName));
    SkOpCoincidence coincidence(&globalState);
    const SkPath* minuend = &one;
    const SkPath* subtrahend = &two;
    if (op == kReverseDifference_SkPathOp) {
        using std::swap;
        swap(minuend, subtrahend);
        op = kDifference_SkPathOp;
    }
#if DEBUG_SORT
    SkPathOpsDebug::gSortCount = SkPathOpsDebug::gSortCountDefault;
#endif
    // turn path into list of segments
    SkOpEdgeBuilder builder(*minuend, contourList, &globalState);
    if (builder.unparseable()) {
        return false;
    }
    const int xorMask = builder.xorMask();
    builder.addOperand(*subtrahend);
    if (!builder.finish()) {
        return false;
    }
#if DEBUG_DUMP_SEGMENTS
    contourList->dumpSegments("seg", op);
#endif

    const int xorOpMask = builder.xorMask();
    if (!SortContourList(&contourList, xorMask == kEvenOdd_PathOpsMask,
            xorOpMask == kEvenOdd_PathOpsMask)) {
        result->reset();
        result->setFillType(fillType);
        return true;
    }
    // find all intersections between segments
    SkOpContour* current = contourList;
    do {
        SkOpContour* next = current;
        while (AddIntersectTs(current, next, &coincidence)
                && (next = next->next()))
            ;
    } while ((current = current->next()));
#if DEBUG_VALIDATE
    globalState.setPhase(SkOpPhase::kWalking);
#endif
    bool success = HandleCoincidence(contourList, &coincidence);
#if DEBUG_COIN
    globalState.debugAddToGlobalCoinDicts();
#endif
    if (!success) {
        return false;
    }
#if DEBUG_ALIGNMENT
    contourList->dumpSegments("aligned");
#endif
    // construct closed contours
    SkPath original = *result;
    result->reset();
    result->setFillType(fillType);
    SkPathWriter wrapper(*result);
    if (!bridgeOp(contourList, op, xorMask, xorOpMask, &wrapper)) {
        *result = original;
        return false;
    }
    wrapper.assemble();  // if some edges could not be resolved, assemble remaining
#if DEBUG_T_SECT_LOOP_COUNT
    static SkMutex& debugWorstLoop = *(new SkMutex);
    {
        SkAutoMutexExclusive autoM(debugWorstLoop);
        if (!gVerboseFinalize) {
            gVerboseFinalize = &ReportPathOpsDebugging;
        }
        debugWorstState.debugDoYourWorst(&globalState);
    }
#endif
    return true;
}

bool Op(const SkPath& one, const SkPath& two, SkPathOp op, SkPath* result) {
#if DEBUG_DUMP_VERIFY
    if (SkPathOpsDebug::gVerifyOp) {
        if (!OpDebug(one, two, op, result  SkDEBUGPARAMS(false) SkDEBUGPARAMS(nullptr))) {
            SkPathOpsDebug::ReportOpFail(one, two, op);
            return false;
        }
        SkPathOpsDebug::VerifyOp(one, two, op, *result);
        return true;
    }
#endif
    return OpDebug(one, two, op, result  SkDEBUGPARAMS(true) SkDEBUGPARAMS(nullptr));
}
