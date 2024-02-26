/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#ifndef SkPathOpsCommon_DEFINED
#define SkPathOpsCommon_DEFINED

#include "include/private/SkTDArray.h"
#include "src/pathops/SkOpAngle.h"

class SkOpCoincidence;
class SkOpContour;
class SkPathWriter;

const SkOpAngle* AngleWinding(SkOpSpanBase* start, SkOpSpanBase* end, int* windingPtr,
                              bool* sortable);
SkOpSegment* FindChase(SkTDArray<SkOpSpanBase*>* chase, SkOpSpanBase** startPtr,
                       SkOpSpanBase** endPtr);
SkOpSpan* FindSortableTop(SkOpContourHead* );
SkOpSpan* FindUndone(SkOpContourHead* );
bool FixWinding(SkPath* path);
bool SortContourList(SkOpContourHead** , bool evenOdd, bool oppEvenOdd);
bool HandleCoincidence(SkOpContourHead* , SkOpCoincidence* );
bool OpDebug(const SkPath& one, const SkPath& two, SkPathOp op, SkPath* result
             SkDEBUGPARAMS(bool skipAssert)
             SkDEBUGPARAMS(const char* testName));

#endif
