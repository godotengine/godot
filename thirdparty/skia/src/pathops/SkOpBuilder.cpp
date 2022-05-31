/*
 * Copyright 2014 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "include/core/SkMatrix.h"
#include "include/pathops/SkPathOps.h"
#include "src/core/SkArenaAlloc.h"
#include "src/core/SkPathPriv.h"
#include "src/pathops/SkOpEdgeBuilder.h"
#include "src/pathops/SkPathOpsCommon.h"

static bool one_contour(const SkPath& path) {
    SkSTArenaAlloc<256> allocator;
    int verbCount = path.countVerbs();
    uint8_t* verbs = (uint8_t*) allocator.makeArrayDefault<uint8_t>(verbCount);
    (void) path.getVerbs(verbs, verbCount);
    for (int index = 1; index < verbCount; ++index) {
        if (verbs[index] == SkPath::kMove_Verb) {
            return false;
        }
    }
    return true;
}

void SkOpBuilder::ReversePath(SkPath* path) {
    SkPath temp;
    SkPoint lastPt;
    SkAssertResult(path->getLastPt(&lastPt));
    temp.moveTo(lastPt);
    temp.reversePathTo(*path);
    temp.close();
    *path = temp;
}

bool SkOpBuilder::FixWinding(SkPath* path) {
    SkPathFillType fillType = path->getFillType();
    if (fillType == SkPathFillType::kInverseEvenOdd) {
        fillType = SkPathFillType::kInverseWinding;
    } else if (fillType == SkPathFillType::kEvenOdd) {
        fillType = SkPathFillType::kWinding;
    }
    if (one_contour(*path)) {
        SkPathFirstDirection dir = SkPathPriv::ComputeFirstDirection(*path);
        if (dir != SkPathFirstDirection::kUnknown) {
            if (dir == SkPathFirstDirection::kCW) {
                ReversePath(path);
            }
            path->setFillType(fillType);
            return true;
        }
    }
    SkSTArenaAlloc<4096> allocator;
    SkOpContourHead contourHead;
    SkOpGlobalState globalState(&contourHead, &allocator  SkDEBUGPARAMS(false)
            SkDEBUGPARAMS(nullptr));
    SkOpEdgeBuilder builder(*path, &contourHead, &globalState);
    if (builder.unparseable() || !builder.finish()) {
        return false;
    }
    if (!contourHead.count()) {
        return true;
    }
    if (!contourHead.next()) {
        return false;
    }
    contourHead.joinAllSegments();
    contourHead.resetReverse();
    bool writePath = false;
    SkOpSpan* topSpan;
    globalState.setPhase(SkOpPhase::kFixWinding);
    while ((topSpan = FindSortableTop(&contourHead))) {
        SkOpSegment* topSegment = topSpan->segment();
        SkOpContour* topContour = topSegment->contour();
        SkASSERT(topContour->isCcw() >= 0);
#if DEBUG_WINDING
        SkDebugf("%s id=%d nested=%d ccw=%d\n",  __FUNCTION__,
                topSegment->debugID(), globalState.nested(), topContour->isCcw());
#endif
        if ((globalState.nested() & 1) != SkToBool(topContour->isCcw())) {
            topContour->setReverse();
            writePath = true;
        }
        topContour->markAllDone();
        globalState.clearNested();
    }
    if (!writePath) {
        path->setFillType(fillType);
        return true;
    }
    SkPath empty;
    SkPathWriter woundPath(empty);
    SkOpContour* test = &contourHead;
    do {
        if (!test->count()) {
            continue;
        }
        if (test->reversed()) {
            test->toReversePath(&woundPath);
        } else {
            test->toPath(&woundPath);
        }
    } while ((test = test->next()));
    *path = *woundPath.nativePath();
    path->setFillType(fillType);
    return true;
}

void SkOpBuilder::add(const SkPath& path, SkPathOp op) {
    if (0 == fOps.count() && op != kUnion_SkPathOp) {
        fPathRefs.push_back() = SkPath();
        *fOps.append() = kUnion_SkPathOp;
    }
    fPathRefs.push_back() = path;
    *fOps.append() = op;
}

void SkOpBuilder::reset() {
    fPathRefs.reset();
    fOps.reset();
}

/* OPTIMIZATION: Union doesn't need to be all-or-nothing. A run of three or more convex
   paths with union ops could be locally resolved and still improve over doing the
   ops one at a time. */
bool SkOpBuilder::resolve(SkPath* result) {
    SkPath original = *result;
    int count = fOps.count();
    bool allUnion = true;
    SkPathFirstDirection firstDir = SkPathFirstDirection::kUnknown;
    for (int index = 0; index < count; ++index) {
        SkPath* test = &fPathRefs[index];
        if (kUnion_SkPathOp != fOps[index] || test->isInverseFillType()) {
            allUnion = false;
            break;
        }
        // If all paths are convex, track direction, reversing as needed.
        if (test->isConvex()) {
            SkPathFirstDirection dir = SkPathPriv::ComputeFirstDirection(*test);
            if (dir == SkPathFirstDirection::kUnknown) {
                allUnion = false;
                break;
            }
            if (firstDir == SkPathFirstDirection::kUnknown) {
                firstDir = dir;
            } else if (firstDir != dir) {
                ReversePath(test);
            }
            continue;
        }
        // If the path is not convex but its bounds do not intersect the others, simplify is enough.
        const SkRect& testBounds = test->getBounds();
        for (int inner = 0; inner < index; ++inner) {
            // OPTIMIZE: check to see if the contour bounds do not intersect other contour bounds?
            if (SkRect::Intersects(fPathRefs[inner].getBounds(), testBounds)) {
                allUnion = false;
                break;
            }
        }
    }
    if (!allUnion) {
        *result = fPathRefs[0];
        for (int index = 1; index < count; ++index) {
            if (!Op(*result, fPathRefs[index], fOps[index], result)) {
                reset();
                *result = original;
                return false;
            }
        }
        reset();
        return true;
    }
    SkPath sum;
    for (int index = 0; index < count; ++index) {
        if (!Simplify(fPathRefs[index], &fPathRefs[index])) {
            reset();
            *result = original;
            return false;
        }
        if (!fPathRefs[index].isEmpty()) {
            // convert the even odd result back to winding form before accumulating it
            if (!FixWinding(&fPathRefs[index])) {
                *result = original;
                return false;
            }
            sum.addPath(fPathRefs[index]);
        }
    }
    reset();
    bool success = Simplify(sum, result);
    if (!success) {
        *result = original;
    }
    return success;
}
