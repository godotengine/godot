/*
* Copyright 2013 Google Inc.
*
* Use of this source code is governed by a BSD-style license that can be
* found in the LICENSE file.
*/
#include "src/pathops/SkOpContour.h"
#include "src/pathops/SkPathWriter.h"
#include "src/pathops/SkReduceOrder.h"

void SkOpContour::toPath(SkPathWriter* path) const {
    if (!this->count()) {
        return;
    }
    const SkOpSegment* segment = &fHead;
    do {
        SkAssertResult(segment->addCurveTo(segment->head(), segment->tail(), path));
    } while ((segment = segment->next()));
    path->finishContour();
    path->assemble();
}

void SkOpContour::toReversePath(SkPathWriter* path) const {
    const SkOpSegment* segment = fTail;
    do {
        SkAssertResult(segment->addCurveTo(segment->tail(), segment->head(), path));
    } while ((segment = segment->prev()));
    path->finishContour();
    path->assemble();
}

SkOpSpan* SkOpContour::undoneSpan() {
    SkOpSegment* testSegment = &fHead;
    do {
        if (testSegment->done()) {
            continue;
        }
        return testSegment->undoneSpan();
    } while ((testSegment = testSegment->next()));
    fDone = true;
    return nullptr;
}

void SkOpContourBuilder::addConic(SkPoint pts[3], SkScalar weight) {
    this->flush();
    fContour->addConic(pts, weight);
}

void SkOpContourBuilder::addCubic(SkPoint pts[4]) {
    this->flush();
    fContour->addCubic(pts);
}

void SkOpContourBuilder::addCurve(SkPath::Verb verb, const SkPoint pts[4], SkScalar weight) {
    if (SkPath::kLine_Verb == verb) {
        this->addLine(pts);
        return;
    }
    SkArenaAlloc* allocator = fContour->globalState()->allocator();
    switch (verb) {
        case SkPath::kQuad_Verb: {
            SkPoint* ptStorage = allocator->makeArrayDefault<SkPoint>(3);
            memcpy(ptStorage, pts, sizeof(SkPoint) * 3);
            this->addQuad(ptStorage);
        } break;
        case SkPath::kConic_Verb: {
            SkPoint* ptStorage = allocator->makeArrayDefault<SkPoint>(3);
            memcpy(ptStorage, pts, sizeof(SkPoint) * 3);
            this->addConic(ptStorage, weight);
        } break;
        case SkPath::kCubic_Verb: {
            SkPoint* ptStorage = allocator->makeArrayDefault<SkPoint>(4);
            memcpy(ptStorage, pts, sizeof(SkPoint) * 4);
            this->addCubic(ptStorage);
        } break;
        default:
            SkASSERT(0);
    }
}

void SkOpContourBuilder::addLine(const SkPoint pts[2]) {
    // if the previous line added is the exact opposite, eliminate both
    if (fLastIsLine) {
        if (fLastLine[0] == pts[1] && fLastLine[1] == pts[0]) {
            fLastIsLine = false;
            return;
        } else {
            flush();
        }
    }
    memcpy(fLastLine, pts, sizeof(fLastLine));
    fLastIsLine = true;
}

void SkOpContourBuilder::addQuad(SkPoint pts[3]) {
    this->flush();
    fContour->addQuad(pts);
}

void SkOpContourBuilder::flush() {
    if (!fLastIsLine)
        return;
    SkArenaAlloc* allocator = fContour->globalState()->allocator();
    SkPoint* ptStorage = allocator->makeArrayDefault<SkPoint>(2);
    memcpy(ptStorage, fLastLine, sizeof(fLastLine));
    (void) fContour->addLine(ptStorage);
    fLastIsLine = false;
}
