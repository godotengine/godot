/*
 * Copyright 2013 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "include/core/SkPath.h"
// -- GODOT start --
//#include "include/core/SkString.h"
#include "include/private/SkMutex.h"
//#include "src/core/SkOSFile.h"
// -- GODOT end --
#include "src/core/SkPathPriv.h"
#include "src/pathops/SkOpCoincidence.h"
#include "src/pathops/SkOpContour.h"
#include "src/pathops/SkPathOpsDebug.h"

#include <utility>

#if DEBUG_DUMP_VERIFY
bool SkPathOpsDebug::gDumpOp;  // set to true to write op to file before a crash
bool SkPathOpsDebug::gVerifyOp;  // set to true to compare result against regions
#endif

bool SkPathOpsDebug::gRunFail;  // set to true to check for success on tests known to fail
bool SkPathOpsDebug::gVeryVerbose;  // set to true to run extensive checking tests

#undef FAIL_IF
#define FAIL_IF(cond, coin) \
         do { if (cond) log->record(SkPathOpsDebug::kFail_Glitch, coin); } while (false)

#undef FAIL_WITH_NULL_IF
#define FAIL_WITH_NULL_IF(cond, span) \
         do { if (cond) log->record(SkPathOpsDebug::kFail_Glitch, span); } while (false)

#define RETURN_FALSE_IF(cond, span) \
         do { if (cond) log->record(SkPathOpsDebug::kReturnFalse_Glitch, span); \
         } while (false)

class SkCoincidentSpans;

#if DEBUG_SORT
int SkPathOpsDebug::gSortCountDefault = SK_MaxS32;
int SkPathOpsDebug::gSortCount;
#endif

#if DEBUG_ACTIVE_OP
const char* SkPathOpsDebug::kPathOpStr[] = {"diff", "sect", "union", "xor", "rdiff"};
#endif

#if defined SK_DEBUG || !FORCE_RELEASE

int SkPathOpsDebug::gContourID = 0;
int SkPathOpsDebug::gSegmentID = 0;

bool SkPathOpsDebug::ChaseContains(const SkTDArray<SkOpSpanBase* >& chaseArray,
        const SkOpSpanBase* span) {
    for (int index = 0; index < chaseArray.count(); ++index) {
        const SkOpSpanBase* entry = chaseArray[index];
        if (entry == span) {
            return true;
        }
    }
    return false;
}
#endif

#if DEBUG_ACTIVE_SPANS
SkString SkPathOpsDebug::gActiveSpans;
#endif

#if DEBUG_COIN

SkPathOpsDebug::CoinDict SkPathOpsDebug::gCoinSumChangedDict;
SkPathOpsDebug::CoinDict SkPathOpsDebug::gCoinSumVisitedDict;

static const int kGlitchType_Count = SkPathOpsDebug::kUnalignedTail_Glitch + 1;

struct SpanGlitch {
    const SkOpSpanBase* fBase;
    const SkOpSpanBase* fSuspect;
    const SkOpSegment* fSegment;
    const SkOpSegment* fOppSegment;
    const SkOpPtT* fCoinSpan;
    const SkOpPtT* fEndSpan;
    const SkOpPtT* fOppSpan;
    const SkOpPtT* fOppEndSpan;
    double fStartT;
    double fEndT;
    double fOppStartT;
    double fOppEndT;
    SkPoint fPt;
    SkPathOpsDebug::GlitchType fType;

    void dumpType() const;
};

struct SkPathOpsDebug::GlitchLog {
    void init(const SkOpGlobalState* state) {
        fGlobalState = state;
    }

    SpanGlitch* recordCommon(GlitchType type) {
        SpanGlitch* glitch = fGlitches.push();
        glitch->fBase = nullptr;
        glitch->fSuspect = nullptr;
        glitch->fSegment = nullptr;
        glitch->fOppSegment = nullptr;
        glitch->fCoinSpan = nullptr;
        glitch->fEndSpan = nullptr;
        glitch->fOppSpan = nullptr;
        glitch->fOppEndSpan = nullptr;
        glitch->fStartT = SK_ScalarNaN;
        glitch->fEndT = SK_ScalarNaN;
        glitch->fOppStartT = SK_ScalarNaN;
        glitch->fOppEndT = SK_ScalarNaN;
        glitch->fPt = { SK_ScalarNaN, SK_ScalarNaN };
        glitch->fType = type;
        return glitch;
    }

    void record(GlitchType type, const SkOpSpanBase* base,
            const SkOpSpanBase* suspect = NULL) {
        SpanGlitch* glitch = recordCommon(type);
        glitch->fBase = base;
        glitch->fSuspect = suspect;
    }

    void record(GlitchType type, const SkOpSpanBase* base,
            const SkOpPtT* ptT) {
        SpanGlitch* glitch = recordCommon(type);
        glitch->fBase = base;
        glitch->fCoinSpan = ptT;
    }

    void record(GlitchType type, const SkCoincidentSpans* coin,
            const SkCoincidentSpans* opp = NULL) {
        SpanGlitch* glitch = recordCommon(type);
        glitch->fCoinSpan = coin->coinPtTStart();
        glitch->fEndSpan = coin->coinPtTEnd();
        if (opp) {
            glitch->fOppSpan = opp->coinPtTStart();
            glitch->fOppEndSpan = opp->coinPtTEnd();
        }
    }

    void record(GlitchType type, const SkOpSpanBase* base,
            const SkOpSegment* seg, double t, SkPoint pt) {
        SpanGlitch* glitch = recordCommon(type);
        glitch->fBase = base;
        glitch->fSegment = seg;
        glitch->fStartT = t;
        glitch->fPt = pt;
    }

    void record(GlitchType type, const SkOpSpanBase* base, double t,
            SkPoint pt) {
        SpanGlitch* glitch = recordCommon(type);
        glitch->fBase = base;
        glitch->fStartT = t;
        glitch->fPt = pt;
    }

    void record(GlitchType type, const SkCoincidentSpans* coin,
            const SkOpPtT* coinSpan, const SkOpPtT* endSpan) {
        SpanGlitch* glitch = recordCommon(type);
        glitch->fCoinSpan = coin->coinPtTStart();
        glitch->fEndSpan = coin->coinPtTEnd();
        glitch->fEndSpan = endSpan;
        glitch->fOppSpan = coinSpan;
        glitch->fOppEndSpan = endSpan;
    }

    void record(GlitchType type, const SkCoincidentSpans* coin,
            const SkOpSpanBase* base) {
        SpanGlitch* glitch = recordCommon(type);
        glitch->fBase = base;
        glitch->fCoinSpan = coin->coinPtTStart();
        glitch->fEndSpan = coin->coinPtTEnd();
    }

    void record(GlitchType type, const SkOpPtT* ptTS, const SkOpPtT* ptTE,
            const SkOpPtT* oPtTS, const SkOpPtT* oPtTE) {
        SpanGlitch* glitch = recordCommon(type);
        glitch->fCoinSpan = ptTS;
        glitch->fEndSpan = ptTE;
        glitch->fOppSpan = oPtTS;
        glitch->fOppEndSpan = oPtTE;
    }

    void record(GlitchType type, const SkOpSegment* seg, double startT,
            double endT, const SkOpSegment* oppSeg, double oppStartT, double oppEndT) {
        SpanGlitch* glitch = recordCommon(type);
        glitch->fSegment = seg;
        glitch->fStartT = startT;
        glitch->fEndT = endT;
        glitch->fOppSegment = oppSeg;
        glitch->fOppStartT = oppStartT;
        glitch->fOppEndT = oppEndT;
    }

    void record(GlitchType type, const SkOpSegment* seg,
            const SkOpSpan* span) {
        SpanGlitch* glitch = recordCommon(type);
        glitch->fSegment = seg;
        glitch->fBase = span;
    }

    void record(GlitchType type, double t, const SkOpSpanBase* span) {
        SpanGlitch* glitch = recordCommon(type);
        glitch->fStartT = t;
        glitch->fBase = span;
    }

    void record(GlitchType type, const SkOpSegment* seg) {
        SpanGlitch* glitch = recordCommon(type);
        glitch->fSegment = seg;
    }

    void record(GlitchType type, const SkCoincidentSpans* coin,
            const SkOpPtT* ptT) {
        SpanGlitch* glitch = recordCommon(type);
        glitch->fCoinSpan = coin->coinPtTStart();
        glitch->fEndSpan = ptT;
    }

    SkTDArray<SpanGlitch> fGlitches;
    const SkOpGlobalState* fGlobalState;
};


void SkPathOpsDebug::CoinDict::add(const SkPathOpsDebug::CoinDict& dict) {
    int count = dict.fDict.count();
    for (int index = 0; index < count; ++index) {
        this->add(dict.fDict[index]);
    }
}

void SkPathOpsDebug::CoinDict::add(const CoinDictEntry& key) {
    int count = fDict.count();
    for (int index = 0; index < count; ++index) {
        CoinDictEntry* entry = &fDict[index];
        if (entry->fIteration == key.fIteration && entry->fLineNumber == key.fLineNumber) {
            SkASSERT(!strcmp(entry->fFunctionName, key.fFunctionName));
            if (entry->fGlitchType == kUninitialized_Glitch) {
                entry->fGlitchType = key.fGlitchType;
            }
            return;
        }
    }
    *fDict.append() = key;
}

#endif

#if DEBUG_COIN
static void missing_coincidence(SkPathOpsDebug::GlitchLog* glitches, const SkOpContourHead* contourList) {
    const SkOpContour* contour = contourList;
    // bool result = false;
    do {
        /* result |= */ contour->debugMissingCoincidence(glitches);
    } while ((contour = contour->next()));
    return;
}

static void move_multiples(SkPathOpsDebug::GlitchLog* glitches, const SkOpContourHead* contourList) {
    const SkOpContour* contour = contourList;
    do {
        if (contour->debugMoveMultiples(glitches), false) {
            return;
        }
    } while ((contour = contour->next()));
    return;
}

static void move_nearby(SkPathOpsDebug::GlitchLog* glitches, const SkOpContourHead* contourList) {
    const SkOpContour* contour = contourList;
    do {
        contour->debugMoveNearby(glitches);
    } while ((contour = contour->next()));
}


#endif

#if DEBUG_COIN
void SkOpGlobalState::debugAddToCoinChangedDict() {

#if DEBUG_COINCIDENCE
    SkPathOpsDebug::CheckHealth(fContourHead);
#endif
    // see if next coincident operation makes a change; if so, record it
    SkPathOpsDebug::GlitchLog glitches;
    const char* funcName = fCoinDictEntry.fFunctionName;
    if (!strcmp("calc_angles", funcName)) {
        ;
    } else if (!strcmp("missing_coincidence", funcName)) {
        missing_coincidence(&glitches, fContourHead);
    } else if (!strcmp("move_multiples", funcName)) {
        move_multiples(&glitches, fContourHead);
    } else if (!strcmp("move_nearby", funcName)) {
        move_nearby(&glitches, fContourHead);
    } else if (!strcmp("addExpanded", funcName)) {
        fCoincidence->debugAddExpanded(&glitches);
    } else if (!strcmp("addMissing", funcName)) {
        bool added;
        fCoincidence->debugAddMissing(&glitches, &added);
    } else if (!strcmp("addEndMovedSpans", funcName)) {
        fCoincidence->debugAddEndMovedSpans(&glitches);
    } else if (!strcmp("correctEnds", funcName)) {
        fCoincidence->debugCorrectEnds(&glitches);
    } else if (!strcmp("expand", funcName)) {
        fCoincidence->debugExpand(&glitches);
    } else if (!strcmp("findOverlaps", funcName)) {
        ;
    } else if (!strcmp("mark", funcName)) {
        fCoincidence->debugMark(&glitches);
    } else if (!strcmp("apply", funcName)) {
        ;
    } else {
        SkASSERT(0);   // add missing case
    }
    if (glitches.fGlitches.count()) {
        fCoinDictEntry.fGlitchType = glitches.fGlitches[0].fType;
    }
    fCoinChangedDict.add(fCoinDictEntry);
}
#endif

void SkPathOpsDebug::ShowActiveSpans(SkOpContourHead* contourList) {
#if DEBUG_ACTIVE_SPANS
    SkString str;
    SkOpContour* contour = contourList;
    do {
        contour->debugShowActiveSpans(&str);
    } while ((contour = contour->next()));
    if (!gActiveSpans.equals(str)) {
        const char* s = str.c_str();
        const char* end;
        while ((end = strchr(s, '\n'))) {
            SkDebugf("%.*s", end - s + 1, s);
            s = end + 1;
        }
        gActiveSpans.set(str);
    }
#endif
}

#if DEBUG_COINCIDENCE || DEBUG_COIN
void SkPathOpsDebug::CheckHealth(SkOpContourHead* contourList) {
#if DEBUG_COINCIDENCE
    contourList->globalState()->debugSetCheckHealth(true);
#endif
#if DEBUG_COIN
    GlitchLog glitches;
    const SkOpContour* contour = contourList;
    const SkOpCoincidence* coincidence = contour->globalState()->coincidence();
    coincidence->debugCheckValid(&glitches); // don't call validate; spans may be inconsistent
    do {
        contour->debugCheckHealth(&glitches);
        contour->debugMissingCoincidence(&glitches);
    } while ((contour = contour->next()));
    bool added;
    coincidence->debugAddMissing(&glitches, &added);
    coincidence->debugExpand(&glitches);
    coincidence->debugAddExpanded(&glitches);
    coincidence->debugMark(&glitches);
    unsigned mask = 0;
    for (int index = 0; index < glitches.fGlitches.count(); ++index) {
        const SpanGlitch& glitch = glitches.fGlitches[index];
        mask |= 1 << glitch.fType;
    }
    for (int index = 0; index < kGlitchType_Count; ++index) {
        SkDebugf(mask & (1 << index) ? "x" : "-");
    }
    SkDebugf(" %s\n", contourList->globalState()->debugCoinDictEntry().fFunctionName);
    for (int index = 0; index < glitches.fGlitches.count(); ++index) {
        const SpanGlitch& glitch = glitches.fGlitches[index];
        SkDebugf("%02d: ", index);
        if (glitch.fBase) {
            SkDebugf(" seg/base=%d/%d", glitch.fBase->segment()->debugID(),
                    glitch.fBase->debugID());
        }
        if (glitch.fSuspect) {
            SkDebugf(" seg/base=%d/%d", glitch.fSuspect->segment()->debugID(),
                    glitch.fSuspect->debugID());
        }
        if (glitch.fSegment) {
            SkDebugf(" segment=%d", glitch.fSegment->debugID());
        }
        if (glitch.fCoinSpan) {
            SkDebugf(" coinSeg/Span/PtT=%d/%d/%d", glitch.fCoinSpan->segment()->debugID(),
                    glitch.fCoinSpan->span()->debugID(), glitch.fCoinSpan->debugID());
        }
        if (glitch.fEndSpan) {
            SkDebugf(" endSpan=%d", glitch.fEndSpan->debugID());
        }
        if (glitch.fOppSpan) {
            SkDebugf(" oppSeg/Span/PtT=%d/%d/%d", glitch.fOppSpan->segment()->debugID(),
                    glitch.fOppSpan->span()->debugID(), glitch.fOppSpan->debugID());
        }
        if (glitch.fOppEndSpan) {
            SkDebugf(" oppEndSpan=%d", glitch.fOppEndSpan->debugID());
        }
        if (!SkScalarIsNaN(glitch.fStartT)) {
            SkDebugf(" startT=%g", glitch.fStartT);
        }
        if (!SkScalarIsNaN(glitch.fEndT)) {
            SkDebugf(" endT=%g", glitch.fEndT);
        }
        if (glitch.fOppSegment) {
            SkDebugf(" segment=%d", glitch.fOppSegment->debugID());
        }
        if (!SkScalarIsNaN(glitch.fOppStartT)) {
            SkDebugf(" oppStartT=%g", glitch.fOppStartT);
        }
        if (!SkScalarIsNaN(glitch.fOppEndT)) {
            SkDebugf(" oppEndT=%g", glitch.fOppEndT);
        }
        if (!SkScalarIsNaN(glitch.fPt.fX) || !SkScalarIsNaN(glitch.fPt.fY)) {
            SkDebugf(" pt=%g,%g", glitch.fPt.fX, glitch.fPt.fY);
        }
        DumpGlitchType(glitch.fType);
        SkDebugf("\n");
    }
#if DEBUG_COINCIDENCE
    contourList->globalState()->debugSetCheckHealth(false);
#endif
#if 01 && DEBUG_ACTIVE_SPANS
//    SkDebugf("active after %s:\n", id);
    ShowActiveSpans(contourList);
#endif
#endif
}
#endif

#if DEBUG_COIN
void SkPathOpsDebug::DumpGlitchType(GlitchType glitchType) {
    switch (glitchType) {
        case kAddCorruptCoin_Glitch: SkDebugf(" AddCorruptCoin"); break;
        case kAddExpandedCoin_Glitch: SkDebugf(" AddExpandedCoin"); break;
        case kAddExpandedFail_Glitch: SkDebugf(" AddExpandedFail"); break;
        case kAddIfCollapsed_Glitch: SkDebugf(" AddIfCollapsed"); break;
        case kAddIfMissingCoin_Glitch: SkDebugf(" AddIfMissingCoin"); break;
        case kAddMissingCoin_Glitch: SkDebugf(" AddMissingCoin"); break;
        case kAddMissingExtend_Glitch: SkDebugf(" AddMissingExtend"); break;
        case kAddOrOverlap_Glitch: SkDebugf(" AAddOrOverlap"); break;
        case kCollapsedCoin_Glitch: SkDebugf(" CollapsedCoin"); break;
        case kCollapsedDone_Glitch: SkDebugf(" CollapsedDone"); break;
        case kCollapsedOppValue_Glitch: SkDebugf(" CollapsedOppValue"); break;
        case kCollapsedSpan_Glitch: SkDebugf(" CollapsedSpan"); break;
        case kCollapsedWindValue_Glitch: SkDebugf(" CollapsedWindValue"); break;
        case kCorrectEnd_Glitch: SkDebugf(" CorrectEnd"); break;
        case kDeletedCoin_Glitch: SkDebugf(" DeletedCoin"); break;
        case kExpandCoin_Glitch: SkDebugf(" ExpandCoin"); break;
        case kFail_Glitch: SkDebugf(" Fail"); break;
        case kMarkCoinEnd_Glitch: SkDebugf(" MarkCoinEnd"); break;
        case kMarkCoinInsert_Glitch: SkDebugf(" MarkCoinInsert"); break;
        case kMarkCoinMissing_Glitch: SkDebugf(" MarkCoinMissing"); break;
        case kMarkCoinStart_Glitch: SkDebugf(" MarkCoinStart"); break;
        case kMergeMatches_Glitch: SkDebugf(" MergeMatches"); break;
        case kMissingCoin_Glitch: SkDebugf(" MissingCoin"); break;
        case kMissingDone_Glitch: SkDebugf(" MissingDone"); break;
        case kMissingIntersection_Glitch: SkDebugf(" MissingIntersection"); break;
        case kMoveMultiple_Glitch: SkDebugf(" MoveMultiple"); break;
        case kMoveNearbyClearAll_Glitch: SkDebugf(" MoveNearbyClearAll"); break;
        case kMoveNearbyClearAll2_Glitch: SkDebugf(" MoveNearbyClearAll2"); break;
        case kMoveNearbyMerge_Glitch: SkDebugf(" MoveNearbyMerge"); break;
        case kMoveNearbyMergeFinal_Glitch: SkDebugf(" MoveNearbyMergeFinal"); break;
        case kMoveNearbyRelease_Glitch: SkDebugf(" MoveNearbyRelease"); break;
        case kMoveNearbyReleaseFinal_Glitch: SkDebugf(" MoveNearbyReleaseFinal"); break;
        case kReleasedSpan_Glitch: SkDebugf(" ReleasedSpan"); break;
        case kReturnFalse_Glitch: SkDebugf(" ReturnFalse"); break;
        case kUnaligned_Glitch: SkDebugf(" Unaligned"); break;
        case kUnalignedHead_Glitch: SkDebugf(" UnalignedHead"); break;
        case kUnalignedTail_Glitch: SkDebugf(" UnalignedTail"); break;
        case kUninitialized_Glitch: break;
        default: SkASSERT(0);
    }
}
#endif

#if defined SK_DEBUG || !FORCE_RELEASE
void SkPathOpsDebug::MathematicaIze(char* str, size_t bufferLen) {
    size_t len = strlen(str);
    bool num = false;
    for (size_t idx = 0; idx < len; ++idx) {
        if (num && str[idx] == 'e') {
            if (len + 2 >= bufferLen) {
                return;
            }
            memmove(&str[idx + 2], &str[idx + 1], len - idx);
            str[idx] = '*';
            str[idx + 1] = '^';
            ++len;
        }
        num = str[idx] >= '0' && str[idx] <= '9';
    }
}

bool SkPathOpsDebug::ValidWind(int wind) {
    return wind > SK_MinS32 + 0xFFFF && wind < SK_MaxS32 - 0xFFFF;
}

void SkPathOpsDebug::WindingPrintf(int wind) {
    if (wind == SK_MinS32) {
        SkDebugf("?");
    } else {
        SkDebugf("%d", wind);
    }
}
#endif //  defined SK_DEBUG || !FORCE_RELEASE


// -- GODOT start --
/*
static void show_function_header(const char* functionName) {
    SkDebugf("\nstatic void %s(skiatest::Reporter* reporter, const char* filename) {\n", functionName);
    if (strcmp("skphealth_com76", functionName) == 0) {
        SkDebugf("found it\n");
    }
}

static const char* gOpStrs[] = {
    "kDifference_SkPathOp",
    "kIntersect_SkPathOp",
    "kUnion_SkPathOp",
    "kXOR_PathOp",
    "kReverseDifference_SkPathOp",
};

const char* SkPathOpsDebug::OpStr(SkPathOp op) {
    return gOpStrs[op];
}

static void show_op(SkPathOp op, const char* pathOne, const char* pathTwo) {
    SkDebugf("    testPathOp(reporter, %s, %s, %s, filename);\n", pathOne, pathTwo, gOpStrs[op]);
    SkDebugf("}\n");
}

void SkPathOpsDebug::ShowPath(const SkPath& a, const SkPath& b, SkPathOp shapeOp,
        const char* testName) {
    static SkMutex& mutex = *(new SkMutex);

    SkAutoMutexExclusive ac(mutex);
    show_function_header(testName);
    ShowOnePath(a, "path", true);
    ShowOnePath(b, "pathB", true);
    show_op(shapeOp, "path", "pathB");
}
*/
// -- GODOT end --

#include "src/pathops/SkIntersectionHelper.h"
#include "src/pathops/SkIntersections.h"
#include "src/pathops/SkPathOpsTypes.h"

#if DEBUG_COIN

void SkOpGlobalState::debugAddToGlobalCoinDicts() {
    static SkMutex& mutex = *(new SkMutex);
    SkAutoMutexExclusive ac(mutex);
    SkPathOpsDebug::gCoinSumChangedDict.add(fCoinChangedDict);
    SkPathOpsDebug::gCoinSumVisitedDict.add(fCoinVisitedDict);
}

#endif

#if DEBUG_T_SECT_LOOP_COUNT
void SkOpGlobalState::debugAddLoopCount(SkIntersections* i, const SkIntersectionHelper& wt,
        const SkIntersectionHelper& wn) {
    for (int index = 0; index < (int) SK_ARRAY_COUNT(fDebugLoopCount); ++index) {
        SkIntersections::DebugLoop looper = (SkIntersections::DebugLoop) index;
        if (fDebugLoopCount[index] >= i->debugLoopCount(looper)) {
            continue;
        }
        fDebugLoopCount[index] = i->debugLoopCount(looper);
        fDebugWorstVerb[index * 2] = wt.segment()->verb();
        fDebugWorstVerb[index * 2 + 1] = wn.segment()->verb();
        sk_bzero(&fDebugWorstPts[index * 8], sizeof(SkPoint) * 8);
        memcpy(&fDebugWorstPts[index * 2 * 4], wt.pts(),
                (SkPathOpsVerbToPoints(wt.segment()->verb()) + 1) * sizeof(SkPoint));
        memcpy(&fDebugWorstPts[(index * 2 + 1) * 4], wn.pts(),
                (SkPathOpsVerbToPoints(wn.segment()->verb()) + 1) * sizeof(SkPoint));
        fDebugWorstWeight[index * 2] = wt.weight();
        fDebugWorstWeight[index * 2 + 1] = wn.weight();
    }
    i->debugResetLoopCount();
}

void SkOpGlobalState::debugDoYourWorst(SkOpGlobalState* local) {
    for (int index = 0; index < (int) SK_ARRAY_COUNT(fDebugLoopCount); ++index) {
        if (fDebugLoopCount[index] >= local->fDebugLoopCount[index]) {
            continue;
        }
        fDebugLoopCount[index] = local->fDebugLoopCount[index];
        fDebugWorstVerb[index * 2] = local->fDebugWorstVerb[index * 2];
        fDebugWorstVerb[index * 2 + 1] = local->fDebugWorstVerb[index * 2 + 1];
        memcpy(&fDebugWorstPts[index * 2 * 4], &local->fDebugWorstPts[index * 2 * 4],
                sizeof(SkPoint) * 8);
        fDebugWorstWeight[index * 2] = local->fDebugWorstWeight[index * 2];
        fDebugWorstWeight[index * 2 + 1] = local->fDebugWorstWeight[index * 2 + 1];
    }
    local->debugResetLoopCounts();
}

static void dump_curve(SkPath::Verb verb, const SkPoint& pts, float weight) {
    if (!verb) {
        return;
    }
    const char* verbs[] = { "", "line", "quad", "conic", "cubic" };
    SkDebugf("%s: {{", verbs[verb]);
    int ptCount = SkPathOpsVerbToPoints(verb);
    for (int index = 0; index <= ptCount; ++index) {
        SkDPoint::Dump((&pts)[index]);
        if (index < ptCount - 1) {
            SkDebugf(", ");
        }
    }
    SkDebugf("}");
    if (weight != 1) {
        SkDebugf(", ");
        if (weight == floorf(weight)) {
            SkDebugf("%.0f", weight);
        } else {
            SkDebugf("%1.9gf", weight);
        }
    }
    SkDebugf("}\n");
}

void SkOpGlobalState::debugLoopReport() {
    const char* loops[] = { "iterations", "coinChecks", "perpCalcs" };
    SkDebugf("\n");
    for (int index = 0; index < (int) SK_ARRAY_COUNT(fDebugLoopCount); ++index) {
        SkDebugf("%s: %d\n", loops[index], fDebugLoopCount[index]);
        dump_curve(fDebugWorstVerb[index * 2], fDebugWorstPts[index * 2 * 4],
                fDebugWorstWeight[index * 2]);
        dump_curve(fDebugWorstVerb[index * 2 + 1], fDebugWorstPts[(index * 2 + 1) * 4],
                fDebugWorstWeight[index * 2 + 1]);
    }
}

void SkOpGlobalState::debugResetLoopCounts() {
    sk_bzero(fDebugLoopCount, sizeof(fDebugLoopCount));
    sk_bzero(fDebugWorstVerb, sizeof(fDebugWorstVerb));
    sk_bzero(fDebugWorstPts, sizeof(fDebugWorstPts));
    sk_bzero(fDebugWorstWeight, sizeof(fDebugWorstWeight));
}
#endif

bool SkOpGlobalState::DebugRunFail() {
    return SkPathOpsDebug::gRunFail;
}

// this is const so it can be called by const methods that overwise don't alter state
#if DEBUG_VALIDATE || DEBUG_COIN
void SkOpGlobalState::debugSetPhase(const char* funcName  DEBUG_COIN_DECLARE_PARAMS()) const {
    auto writable = const_cast<SkOpGlobalState*>(this);
#if DEBUG_VALIDATE
    writable->setPhase(phase);
#endif
#if DEBUG_COIN
    SkPathOpsDebug::CoinDictEntry* entry = &writable->fCoinDictEntry;
    writable->fPreviousFuncName = entry->fFunctionName;
    entry->fIteration = iteration;
    entry->fLineNumber = lineNo;
    entry->fGlitchType = SkPathOpsDebug::kUninitialized_Glitch;
    entry->fFunctionName = funcName;
    writable->fCoinVisitedDict.add(*entry);
    writable->debugAddToCoinChangedDict();
#endif
}
#endif

#if DEBUG_T_SECT_LOOP_COUNT
void SkIntersections::debugBumpLoopCount(DebugLoop index) {
    fDebugLoopCount[index]++;
}

int SkIntersections::debugLoopCount(DebugLoop index) const {
    return fDebugLoopCount[index];
}

void SkIntersections::debugResetLoopCount() {
    sk_bzero(fDebugLoopCount, sizeof(fDebugLoopCount));
}
#endif

#include "src/pathops/SkPathOpsConic.h"
#include "src/pathops/SkPathOpsCubic.h"

SkDCubic SkDQuad::debugToCubic() const {
    SkDCubic cubic;
    cubic[0] = fPts[0];
    cubic[2] = fPts[1];
    cubic[3] = fPts[2];
    cubic[1].fX = (cubic[0].fX + cubic[2].fX * 2) / 3;
    cubic[1].fY = (cubic[0].fY + cubic[2].fY * 2) / 3;
    cubic[2].fX = (cubic[3].fX + cubic[2].fX * 2) / 3;
    cubic[2].fY = (cubic[3].fY + cubic[2].fY * 2) / 3;
    return cubic;
}

void SkDQuad::debugSet(const SkDPoint* pts) {
    memcpy(fPts, pts, sizeof(fPts));
    SkDEBUGCODE(fDebugGlobalState = nullptr);
}

void SkDCubic::debugSet(const SkDPoint* pts) {
    memcpy(fPts, pts, sizeof(fPts));
    SkDEBUGCODE(fDebugGlobalState = nullptr);
}

void SkDConic::debugSet(const SkDPoint* pts, SkScalar weight) {
    fPts.debugSet(pts);
    fWeight = weight;
}

void SkDRect::debugInit() {
    fLeft = fTop = fRight = fBottom = SK_ScalarNaN;
}

#include "src/pathops/SkOpAngle.h"
#include "src/pathops/SkOpSegment.h"

#if DEBUG_COIN
// commented-out lines keep this in sync with addT()
 const SkOpPtT* SkOpSegment::debugAddT(double t, SkPathOpsDebug::GlitchLog* log) const {
    debugValidate();
    SkPoint pt = this->ptAtT(t);
    const SkOpSpanBase* span = &fHead;
    do {
        const SkOpPtT* result = span->ptT();
        if (t == result->fT || this->match(result, this, t, pt)) {
//             span->bumpSpanAdds();
             return result;
        }
        if (t < result->fT) {
            const SkOpSpan* prev = result->span()->prev();
            FAIL_WITH_NULL_IF(!prev, span);
            // marks in global state that new op span has been allocated
            this->globalState()->setAllocatedOpSpan();
//             span->init(this, prev, t, pt);
            this->debugValidate();
// #if DEBUG_ADD_T
//             SkDebugf("%s insert t=%1.9g segID=%d spanID=%d\n", __FUNCTION__, t,
//                     span->segment()->debugID(), span->debugID());
// #endif
//             span->bumpSpanAdds();
            return nullptr;
        }
        FAIL_WITH_NULL_IF(span != &fTail, span);
    } while ((span = span->upCast()->next()));
    SkASSERT(0);
    return nullptr;  // we never get here, but need this to satisfy compiler
}
#endif

#if DEBUG_ANGLE
void SkOpSegment::debugCheckAngleCoin() const {
    const SkOpSpanBase* base = &fHead;
    const SkOpSpan* span;
    do {
        const SkOpAngle* angle = base->fromAngle();
        if (angle && angle->debugCheckCoincidence()) {
            angle->debugCheckNearCoincidence();
        }
        if (base->final()) {
             break;
        }
        span = base->upCast();
        angle = span->toAngle();
        if (angle && angle->debugCheckCoincidence()) {
            angle->debugCheckNearCoincidence();
        }
    } while ((base = span->next()));
}
#endif

#if DEBUG_COIN
// this mimics the order of the checks in handle coincidence
void SkOpSegment::debugCheckHealth(SkPathOpsDebug::GlitchLog* glitches) const {
    debugMoveMultiples(glitches);
    debugMoveNearby(glitches);
    debugMissingCoincidence(glitches);
}

// commented-out lines keep this in sync with clearAll()
void SkOpSegment::debugClearAll(SkPathOpsDebug::GlitchLog* glitches) const {
    const SkOpSpan* span = &fHead;
    do {
        this->debugClearOne(span, glitches);
    } while ((span = span->next()->upCastable()));
    this->globalState()->coincidence()->debugRelease(glitches, this);
}

// commented-out lines keep this in sync with clearOne()
void SkOpSegment::debugClearOne(const SkOpSpan* span, SkPathOpsDebug::GlitchLog* glitches) const {
    if (span->windValue()) glitches->record(SkPathOpsDebug::kCollapsedWindValue_Glitch, span);
    if (span->oppValue()) glitches->record(SkPathOpsDebug::kCollapsedOppValue_Glitch, span);
    if (!span->done()) glitches->record(SkPathOpsDebug::kCollapsedDone_Glitch, span);
}
#endif

SkOpAngle* SkOpSegment::debugLastAngle() {
    SkOpAngle* result = nullptr;
    SkOpSpan* span = this->head();
    do {
        if (span->toAngle()) {
            SkASSERT(!result);
            result = span->toAngle();
        }
    } while ((span = span->next()->upCastable()));
    SkASSERT(result);
    return result;
}

#if DEBUG_COIN
// commented-out lines keep this in sync with ClearVisited
void SkOpSegment::DebugClearVisited(const SkOpSpanBase* span) {
    // reset visited flag back to false
    do {
        const SkOpPtT* ptT = span->ptT(), * stopPtT = ptT;
        while ((ptT = ptT->next()) != stopPtT) {
            const SkOpSegment* opp = ptT->segment();
            opp->resetDebugVisited();
        }
    } while (!span->final() && (span = span->upCast()->next()));
}
#endif

#if DEBUG_COIN
// commented-out lines keep this in sync with missingCoincidence()
// look for pairs of undetected coincident curves
// assumes that segments going in have visited flag clear
// Even though pairs of curves correct detect coincident runs, a run may be missed
// if the coincidence is a product of multiple intersections. For instance, given
// curves A, B, and C:
// A-B intersect at a point 1; A-C and B-C intersect at point 2, so near
// the end of C that the intersection is replaced with the end of C.
// Even though A-B correctly do not detect an intersection at point 2,
// the resulting run from point 1 to point 2 is coincident on A and B.
void SkOpSegment::debugMissingCoincidence(SkPathOpsDebug::GlitchLog* log) const {
    if (this->done()) {
        return;
    }
    const SkOpSpan* prior = nullptr;
    const SkOpSpanBase* spanBase = &fHead;
//    bool result = false;
    do {
        const SkOpPtT* ptT = spanBase->ptT(), * spanStopPtT = ptT;
        SkASSERT(ptT->span() == spanBase);
        while ((ptT = ptT->next()) != spanStopPtT) {
            if (ptT->deleted()) {
                continue;
            }
            const SkOpSegment* opp = ptT->span()->segment();
            if (opp->done()) {
                continue;
            }
            // when opp is encounted the 1st time, continue; on 2nd encounter, look for coincidence
            if (!opp->debugVisited()) {
                continue;
            }
            if (spanBase == &fHead) {
                continue;
            }
            if (ptT->segment() == this) {
                continue;
            }
            const SkOpSpan* span = spanBase->upCastable();
            // FIXME?: this assumes that if the opposite segment is coincident then no more
            // coincidence needs to be detected. This may not be true.
            if (span && span->segment() != opp && span->containsCoincidence(opp)) {  // debug has additional condition since it may be called before inner duplicate points have been deleted
                continue;
            }
            if (spanBase->segment() != opp && spanBase->containsCoinEnd(opp)) {  // debug has additional condition since it may be called before inner duplicate points have been deleted
                continue;
            }
            const SkOpPtT* priorPtT = nullptr, * priorStopPtT;
            // find prior span containing opp segment
            const SkOpSegment* priorOpp = nullptr;
            const SkOpSpan* priorTest = spanBase->prev();
            while (!priorOpp && priorTest) {
                priorStopPtT = priorPtT = priorTest->ptT();
                while ((priorPtT = priorPtT->next()) != priorStopPtT) {
                    if (priorPtT->deleted()) {
                        continue;
                    }
                    const SkOpSegment* segment = priorPtT->span()->segment();
                    if (segment == opp) {
                        prior = priorTest;
                        priorOpp = opp;
                        break;
                    }
                }
                priorTest = priorTest->prev();
            }
            if (!priorOpp) {
                continue;
            }
            if (priorPtT == ptT) {
                continue;
            }
            const SkOpPtT* oppStart = prior->ptT();
            const SkOpPtT* oppEnd = spanBase->ptT();
            bool swapped = priorPtT->fT > ptT->fT;
            if (swapped) {
                using std::swap;
                swap(priorPtT, ptT);
                swap(oppStart, oppEnd);
            }
            const SkOpCoincidence* coincidence = this->globalState()->coincidence();
            const SkOpPtT* rootPriorPtT = priorPtT->span()->ptT();
            const SkOpPtT* rootPtT = ptT->span()->ptT();
            const SkOpPtT* rootOppStart = oppStart->span()->ptT();
            const SkOpPtT* rootOppEnd = oppEnd->span()->ptT();
            if (coincidence->contains(rootPriorPtT, rootPtT, rootOppStart, rootOppEnd)) {
                goto swapBack;
            }
            if (testForCoincidence(rootPriorPtT, rootPtT, prior, spanBase, opp)) {
            // mark coincidence
#if DEBUG_COINCIDENCE_VERBOSE
//                 SkDebugf("%s coinSpan=%d endSpan=%d oppSpan=%d oppEndSpan=%d\n", __FUNCTION__,
//                         rootPriorPtT->debugID(), rootPtT->debugID(), rootOppStart->debugID(),
//                         rootOppEnd->debugID());
#endif
                log->record(SkPathOpsDebug::kMissingCoin_Glitch, priorPtT, ptT, oppStart, oppEnd);
                //   coincidences->add(rootPriorPtT, rootPtT, rootOppStart, rootOppEnd);
                // }
#if DEBUG_COINCIDENCE
//                SkASSERT(coincidences->contains(rootPriorPtT, rootPtT, rootOppStart, rootOppEnd);
#endif
                // result = true;
            }
    swapBack:
            if (swapped) {
                using std::swap;
                swap(priorPtT, ptT);
            }
        }
    } while ((spanBase = spanBase->final() ? nullptr : spanBase->upCast()->next()));
    DebugClearVisited(&fHead);
    return;
}

// commented-out lines keep this in sync with moveMultiples()
// if a span has more than one intersection, merge the other segments' span as needed
void SkOpSegment::debugMoveMultiples(SkPathOpsDebug::GlitchLog* glitches) const {
    debugValidate();
    const SkOpSpanBase* test = &fHead;
    do {
        int addCount = test->spanAddsCount();
//        SkASSERT(addCount >= 1);
        if (addCount <= 1) {
            continue;
        }
        const SkOpPtT* startPtT = test->ptT();
        const SkOpPtT* testPtT = startPtT;
        do {  // iterate through all spans associated with start
            const SkOpSpanBase* oppSpan = testPtT->span();
            if (oppSpan->spanAddsCount() == addCount) {
                continue;
            }
            if (oppSpan->deleted()) {
                continue;
            }
            const SkOpSegment* oppSegment = oppSpan->segment();
            if (oppSegment == this) {
                continue;
            }
            // find range of spans to consider merging
            const SkOpSpanBase* oppPrev = oppSpan;
            const SkOpSpanBase* oppFirst = oppSpan;
            while ((oppPrev = oppPrev->prev())) {
                if (!roughly_equal(oppPrev->t(), oppSpan->t())) {
                    break;
                }
                if (oppPrev->spanAddsCount() == addCount) {
                    continue;
                }
                if (oppPrev->deleted()) {
                    continue;
                }
                oppFirst = oppPrev;
            }
            const SkOpSpanBase* oppNext = oppSpan;
            const SkOpSpanBase* oppLast = oppSpan;
            while ((oppNext = oppNext->final() ? nullptr : oppNext->upCast()->next())) {
                if (!roughly_equal(oppNext->t(), oppSpan->t())) {
                    break;
                }
                if (oppNext->spanAddsCount() == addCount) {
                    continue;
                }
                if (oppNext->deleted()) {
                    continue;
                }
                oppLast = oppNext;
            }
            if (oppFirst == oppLast) {
                continue;
            }
            const SkOpSpanBase* oppTest = oppFirst;
            do {
                if (oppTest == oppSpan) {
                    continue;
                }
                // check to see if the candidate meets specific criteria:
                // it contains spans of segments in test's loop but not including 'this'
                const SkOpPtT* oppStartPtT = oppTest->ptT();
                const SkOpPtT* oppPtT = oppStartPtT;
                while ((oppPtT = oppPtT->next()) != oppStartPtT) {
                    const SkOpSegment* oppPtTSegment = oppPtT->segment();
                    if (oppPtTSegment == this) {
                        goto tryNextSpan;
                    }
                    const SkOpPtT* matchPtT = startPtT;
                    do {
                        if (matchPtT->segment() == oppPtTSegment) {
                            goto foundMatch;
                        }
                    } while ((matchPtT = matchPtT->next()) != startPtT);
                    goto tryNextSpan;
            foundMatch:  // merge oppTest and oppSpan
                    oppSegment->debugValidate();
                    oppTest->debugMergeMatches(glitches, oppSpan);
                    oppTest->debugAddOpp(glitches, oppSpan);
                    oppSegment->debugValidate();
                    goto checkNextSpan;
                }
        tryNextSpan:
                ;
            } while (oppTest != oppLast && (oppTest = oppTest->upCast()->next()));
        } while ((testPtT = testPtT->next()) != startPtT);
checkNextSpan:
        ;
    } while ((test = test->final() ? nullptr : test->upCast()->next()));
   debugValidate();
   return;
}

// commented-out lines keep this in sync with moveNearby()
// Move nearby t values and pts so they all hang off the same span. Alignment happens later.
void SkOpSegment::debugMoveNearby(SkPathOpsDebug::GlitchLog* glitches) const {
    debugValidate();
    // release undeleted spans pointing to this seg that are linked to the primary span
    const SkOpSpanBase* spanBase = &fHead;
    do {
        const SkOpPtT* ptT = spanBase->ptT();
        const SkOpPtT* headPtT = ptT;
        while ((ptT = ptT->next()) != headPtT) {
              const SkOpSpanBase* test = ptT->span();
            if (ptT->segment() == this && !ptT->deleted() && test != spanBase
                    && test->ptT() == ptT) {
                if (test->final()) {
                    if (spanBase == &fHead) {
                        glitches->record(SkPathOpsDebug::kMoveNearbyClearAll_Glitch, this);
//                        return;
                    }
                    glitches->record(SkPathOpsDebug::kMoveNearbyReleaseFinal_Glitch, spanBase, ptT);
                } else if (test->prev()) {
                    glitches->record(SkPathOpsDebug::kMoveNearbyRelease_Glitch, test, headPtT);
                }
//                break;
            }
        }
        spanBase = spanBase->upCast()->next();
    } while (!spanBase->final());

    // This loop looks for adjacent spans which are near by
    spanBase = &fHead;
    do {  // iterate through all spans associated with start
        const SkOpSpanBase* test = spanBase->upCast()->next();
        bool found;
        if (!this->spansNearby(spanBase, test, &found)) {
            glitches->record(SkPathOpsDebug::kMoveNearbyMergeFinal_Glitch, test);
        }
        if (found) {
            if (test->final()) {
                if (spanBase->prev()) {
                    glitches->record(SkPathOpsDebug::kMoveNearbyMergeFinal_Glitch, test);
                } else {
                    glitches->record(SkPathOpsDebug::kMoveNearbyClearAll2_Glitch, this);
                    // return
                }
            } else {
                glitches->record(SkPathOpsDebug::kMoveNearbyMerge_Glitch, spanBase);
            }
        }
        spanBase = test;
    } while (!spanBase->final());
    debugValidate();
}
#endif

void SkOpSegment::debugReset() {
    this->init(this->fPts, this->fWeight, this->contour(), this->verb());
}

#if DEBUG_COINCIDENCE_ORDER
void SkOpSegment::debugSetCoinT(int index, SkScalar t) const {
    if (fDebugBaseMax < 0 || fDebugBaseIndex == index) {
        fDebugBaseIndex = index;
        fDebugBaseMin = std::min(t, fDebugBaseMin);
        fDebugBaseMax = std::max(t, fDebugBaseMax);
        return;
    }
    SkASSERT(fDebugBaseMin >= t || t >= fDebugBaseMax);
    if (fDebugLastMax < 0 || fDebugLastIndex == index) {
        fDebugLastIndex = index;
        fDebugLastMin = std::min(t, fDebugLastMin);
        fDebugLastMax = std::max(t, fDebugLastMax);
        return;
    }
    SkASSERT(fDebugLastMin >= t || t >= fDebugLastMax);
    SkASSERT((t - fDebugBaseMin > 0) == (fDebugLastMin - fDebugBaseMin > 0));
}
#endif

#if DEBUG_ACTIVE_SPANS
void SkOpSegment::debugShowActiveSpans(SkString* str) const {
    debugValidate();
    if (done()) {
        return;
    }
    int lastId = -1;
    double lastT = -1;
    const SkOpSpan* span = &fHead;
    do {
        if (span->done()) {
            continue;
        }
        if (lastId == this->debugID() && lastT == span->t()) {
            continue;
        }
        lastId = this->debugID();
        lastT = span->t();
        str->appendf("%s id=%d", __FUNCTION__, this->debugID());
        // since endpoints may have be adjusted, show actual computed curves
        SkDCurve curvePart;
        this->subDivide(span, span->next(), &curvePart);
        const SkDPoint* pts = curvePart.fCubic.fPts;
        str->appendf(" (%1.9g,%1.9g", pts[0].fX, pts[0].fY);
        for (int vIndex = 1; vIndex <= SkPathOpsVerbToPoints(fVerb); ++vIndex) {
            str->appendf(" %1.9g,%1.9g", pts[vIndex].fX, pts[vIndex].fY);
        }
        if (SkPath::kConic_Verb == fVerb) {
            str->appendf(" %1.9gf", curvePart.fConic.fWeight);
        }
        str->appendf(") t=%1.9g tEnd=%1.9g", span->t(), span->next()->t());
        if (span->windSum() == SK_MinS32) {
            str->appendf(" windSum=?");
        } else {
            str->appendf(" windSum=%d", span->windSum());
        }
        if (span->oppValue() && span->oppSum() == SK_MinS32) {
            str->appendf(" oppSum=?");
        } else if (span->oppValue() || span->oppSum() != SK_MinS32) {
            str->appendf(" oppSum=%d", span->oppSum());
        }
        str->appendf(" windValue=%d", span->windValue());
        if (span->oppValue() || span->oppSum() != SK_MinS32) {
            str->appendf(" oppValue=%d", span->oppValue());
        }
        str->appendf("\n");
   } while ((span = span->next()->upCastable()));
}
#endif

#if DEBUG_MARK_DONE
void SkOpSegment::debugShowNewWinding(const char* fun, const SkOpSpan* span, int winding) {
    const SkPoint& pt = span->ptT()->fPt;
    SkDebugf("%s id=%d", fun, this->debugID());
    SkDebugf(" (%1.9g,%1.9g", fPts[0].fX, fPts[0].fY);
    for (int vIndex = 1; vIndex <= SkPathOpsVerbToPoints(fVerb); ++vIndex) {
        SkDebugf(" %1.9g,%1.9g", fPts[vIndex].fX, fPts[vIndex].fY);
    }
    SkDebugf(") t=%1.9g [%d] (%1.9g,%1.9g) tEnd=%1.9g newWindSum=",
            span->t(), span->debugID(), pt.fX, pt.fY, span->next()->t());
    if (winding == SK_MinS32) {
        SkDebugf("?");
    } else {
        SkDebugf("%d", winding);
    }
    SkDebugf(" windSum=");
    if (span->windSum() == SK_MinS32) {
        SkDebugf("?");
    } else {
        SkDebugf("%d", span->windSum());
    }
    SkDebugf(" windValue=%d\n", span->windValue());
}

void SkOpSegment::debugShowNewWinding(const char* fun, const SkOpSpan* span, int winding,
                                      int oppWinding) {
    const SkPoint& pt = span->ptT()->fPt;
    SkDebugf("%s id=%d", fun, this->debugID());
    SkDebugf(" (%1.9g,%1.9g", fPts[0].fX, fPts[0].fY);
    for (int vIndex = 1; vIndex <= SkPathOpsVerbToPoints(fVerb); ++vIndex) {
        SkDebugf(" %1.9g,%1.9g", fPts[vIndex].fX, fPts[vIndex].fY);
    }
    SkDebugf(") t=%1.9g [%d] (%1.9g,%1.9g) tEnd=%1.9g newWindSum=",
            span->t(), span->debugID(), pt.fX, pt.fY, span->next()->t(), winding, oppWinding);
    if (winding == SK_MinS32) {
        SkDebugf("?");
    } else {
        SkDebugf("%d", winding);
    }
    SkDebugf(" newOppSum=");
    if (oppWinding == SK_MinS32) {
        SkDebugf("?");
    } else {
        SkDebugf("%d", oppWinding);
    }
    SkDebugf(" oppSum=");
    if (span->oppSum() == SK_MinS32) {
        SkDebugf("?");
    } else {
        SkDebugf("%d", span->oppSum());
    }
    SkDebugf(" windSum=");
    if (span->windSum() == SK_MinS32) {
        SkDebugf("?");
    } else {
        SkDebugf("%d", span->windSum());
    }
    SkDebugf(" windValue=%d oppValue=%d\n", span->windValue(), span->oppValue());
}

#endif

// loop looking for a pair of angle parts that are too close to be sorted
/* This is called after other more simple intersection and angle sorting tests have been exhausted.
   This should be rarely called -- the test below is thorough and time consuming.
   This checks the distance between start points; the distance between
*/
#if DEBUG_ANGLE
void SkOpAngle::debugCheckNearCoincidence() const {
    const SkOpAngle* test = this;
    do {
        const SkOpSegment* testSegment = test->segment();
        double testStartT = test->start()->t();
        SkDPoint testStartPt = testSegment->dPtAtT(testStartT);
        double testEndT = test->end()->t();
        SkDPoint testEndPt = testSegment->dPtAtT(testEndT);
        double testLenSq = testStartPt.distanceSquared(testEndPt);
        SkDebugf("%s testLenSq=%1.9g id=%d\n", __FUNCTION__, testLenSq, testSegment->debugID());
        double testMidT = (testStartT + testEndT) / 2;
        const SkOpAngle* next = test;
        while ((next = next->fNext) != this) {
            SkOpSegment* nextSegment = next->segment();
            double testMidDistSq = testSegment->distSq(testMidT, next);
            double testEndDistSq = testSegment->distSq(testEndT, next);
            double nextStartT = next->start()->t();
            SkDPoint nextStartPt = nextSegment->dPtAtT(nextStartT);
            double distSq = testStartPt.distanceSquared(nextStartPt);
            double nextEndT = next->end()->t();
            double nextMidT = (nextStartT + nextEndT) / 2;
            double nextMidDistSq = nextSegment->distSq(nextMidT, test);
            double nextEndDistSq = nextSegment->distSq(nextEndT, test);
            SkDebugf("%s distSq=%1.9g testId=%d nextId=%d\n", __FUNCTION__, distSq,
                    testSegment->debugID(), nextSegment->debugID());
            SkDebugf("%s testMidDistSq=%1.9g\n", __FUNCTION__, testMidDistSq);
            SkDebugf("%s testEndDistSq=%1.9g\n", __FUNCTION__, testEndDistSq);
            SkDebugf("%s nextMidDistSq=%1.9g\n", __FUNCTION__, nextMidDistSq);
            SkDebugf("%s nextEndDistSq=%1.9g\n", __FUNCTION__, nextEndDistSq);
            SkDPoint nextEndPt = nextSegment->dPtAtT(nextEndT);
            double nextLenSq = nextStartPt.distanceSquared(nextEndPt);
            SkDebugf("%s nextLenSq=%1.9g\n", __FUNCTION__, nextLenSq);
            SkDebugf("\n");
        }
        test = test->fNext;
    } while (test->fNext != this);
}
#endif

#if DEBUG_ANGLE
SkString SkOpAngle::debugPart() const {
    SkString result;
    switch (this->segment()->verb()) {
        case SkPath::kLine_Verb:
            result.printf(LINE_DEBUG_STR " id=%d", LINE_DEBUG_DATA(fPart.fCurve),
                    this->segment()->debugID());
            break;
        case SkPath::kQuad_Verb:
            result.printf(QUAD_DEBUG_STR " id=%d", QUAD_DEBUG_DATA(fPart.fCurve),
                    this->segment()->debugID());
            break;
        case SkPath::kConic_Verb:
            result.printf(CONIC_DEBUG_STR " id=%d",
                    CONIC_DEBUG_DATA(fPart.fCurve, fPart.fCurve.fConic.fWeight),
                    this->segment()->debugID());
            break;
        case SkPath::kCubic_Verb:
            result.printf(CUBIC_DEBUG_STR " id=%d", CUBIC_DEBUG_DATA(fPart.fCurve),
                    this->segment()->debugID());
            break;
        default:
            SkASSERT(0);
    }
    return result;
}
#endif

#if DEBUG_SORT
void SkOpAngle::debugLoop() const {
    const SkOpAngle* first = this;
    const SkOpAngle* next = this;
    do {
        next->dumpOne(true);
        SkDebugf("\n");
        next = next->fNext;
    } while (next && next != first);
    next = first;
    do {
        next->debugValidate();
        next = next->fNext;
    } while (next && next != first);
}
#endif

void SkOpAngle::debugValidate() const {
#if DEBUG_COINCIDENCE
    if (this->globalState()->debugCheckHealth()) {
        return;
    }
#endif
#if DEBUG_VALIDATE
    const SkOpAngle* first = this;
    const SkOpAngle* next = this;
    int wind = 0;
    int opp = 0;
    int lastXor = -1;
    int lastOppXor = -1;
    do {
        if (next->unorderable()) {
            return;
        }
        const SkOpSpan* minSpan = next->start()->starter(next->end());
        if (minSpan->windValue() == SK_MinS32) {
            return;
        }
        bool op = next->segment()->operand();
        bool isXor = next->segment()->isXor();
        bool oppXor = next->segment()->oppXor();
        SkASSERT(!DEBUG_LIMIT_WIND_SUM || between(0, minSpan->windValue(), DEBUG_LIMIT_WIND_SUM));
        SkASSERT(!DEBUG_LIMIT_WIND_SUM
                || between(-DEBUG_LIMIT_WIND_SUM, minSpan->oppValue(), DEBUG_LIMIT_WIND_SUM));
        bool useXor = op ? oppXor : isXor;
        SkASSERT(lastXor == -1 || lastXor == (int) useXor);
        lastXor = (int) useXor;
        wind += next->debugSign() * (op ? minSpan->oppValue() : minSpan->windValue());
        if (useXor) {
            wind &= 1;
        }
        useXor = op ? isXor : oppXor;
        SkASSERT(lastOppXor == -1 || lastOppXor == (int) useXor);
        lastOppXor = (int) useXor;
        opp += next->debugSign() * (op ? minSpan->windValue() : minSpan->oppValue());
        if (useXor) {
            opp &= 1;
        }
        next = next->fNext;
    } while (next && next != first);
    SkASSERT(wind == 0 || !SkPathOpsDebug::gRunFail);
    SkASSERT(opp == 0 || !SkPathOpsDebug::gRunFail);
#endif
}

void SkOpAngle::debugValidateNext() const {
#if !FORCE_RELEASE
    const SkOpAngle* first = this;
    const SkOpAngle* next = first;
    SkTDArray<const SkOpAngle*>(angles);
    do {
//        SkASSERT_RELEASE(next->fSegment->debugContains(next));
        angles.push_back(next);
        next = next->next();
        if (next == first) {
            break;
        }
        SkASSERT_RELEASE(!angles.contains(next));
        if (!next) {
            return;
        }
    } while (true);
#endif
}

#ifdef SK_DEBUG
void SkCoincidentSpans::debugStartCheck(const SkOpSpanBase* outer, const SkOpSpanBase* over,
        const SkOpGlobalState* debugState) const {
    SkASSERT(coinPtTEnd()->span() == over || !SkOpGlobalState::DebugRunFail());
    SkASSERT(oppPtTEnd()->span() == outer || !SkOpGlobalState::DebugRunFail());
}
#endif

#if DEBUG_COIN
// sets the span's end to the ptT referenced by the previous-next
void SkCoincidentSpans::debugCorrectOneEnd(SkPathOpsDebug::GlitchLog* log,
        const SkOpPtT* (SkCoincidentSpans::* getEnd)() const,
        void (SkCoincidentSpans::*setEnd)(const SkOpPtT* ptT) const ) const {
    const SkOpPtT* origPtT = (this->*getEnd)();
    const SkOpSpanBase* origSpan = origPtT->span();
    const SkOpSpan* prev = origSpan->prev();
    const SkOpPtT* testPtT = prev ? prev->next()->ptT()
            : origSpan->upCast()->next()->prev()->ptT();
    if (origPtT != testPtT) {
        log->record(SkPathOpsDebug::kCorrectEnd_Glitch, this, origPtT, testPtT);
    }
}


/* Commented-out lines keep this in sync with correctEnds */
// FIXME: member pointers have fallen out of favor and can be replaced with
// an alternative approach.
// makes all span ends agree with the segment's spans that define them
void SkCoincidentSpans::debugCorrectEnds(SkPathOpsDebug::GlitchLog* log) const {
    this->debugCorrectOneEnd(log, &SkCoincidentSpans::coinPtTStart, nullptr);
    this->debugCorrectOneEnd(log, &SkCoincidentSpans::coinPtTEnd, nullptr);
    this->debugCorrectOneEnd(log, &SkCoincidentSpans::oppPtTStart, nullptr);
    this->debugCorrectOneEnd(log, &SkCoincidentSpans::oppPtTEnd, nullptr);
}

/* Commented-out lines keep this in sync with expand */
// expand the range by checking adjacent spans for coincidence
bool SkCoincidentSpans::debugExpand(SkPathOpsDebug::GlitchLog* log) const {
    bool expanded = false;
    const SkOpSegment* segment = coinPtTStart()->segment();
    const SkOpSegment* oppSegment = oppPtTStart()->segment();
    do {
        const SkOpSpan* start = coinPtTStart()->span()->upCast();
        const SkOpSpan* prev = start->prev();
        const SkOpPtT* oppPtT;
        if (!prev || !(oppPtT = prev->contains(oppSegment))) {
            break;
        }
        double midT = (prev->t() + start->t()) / 2;
        if (!segment->isClose(midT, oppSegment)) {
            break;
        }
        if (log) log->record(SkPathOpsDebug::kExpandCoin_Glitch, this, prev->ptT(), oppPtT);
        expanded = true;
    } while (false);  // actual continues while expansion is possible
    do {
        const SkOpSpanBase* end = coinPtTEnd()->span();
        SkOpSpanBase* next = end->final() ? nullptr : end->upCast()->next();
        if (next && next->deleted()) {
            break;
        }
        const SkOpPtT* oppPtT;
        if (!next || !(oppPtT = next->contains(oppSegment))) {
            break;
        }
        double midT = (end->t() + next->t()) / 2;
        if (!segment->isClose(midT, oppSegment)) {
            break;
        }
        if (log) log->record(SkPathOpsDebug::kExpandCoin_Glitch, this, next->ptT(), oppPtT);
        expanded = true;
    } while (false);  // actual continues while expansion is possible
    return expanded;
}

// description below
void SkOpCoincidence::debugAddEndMovedSpans(SkPathOpsDebug::GlitchLog* log, const SkOpSpan* base, const SkOpSpanBase* testSpan) const {
    const SkOpPtT* testPtT = testSpan->ptT();
    const SkOpPtT* stopPtT = testPtT;
    const SkOpSegment* baseSeg = base->segment();
    while ((testPtT = testPtT->next()) != stopPtT) {
        const SkOpSegment* testSeg = testPtT->segment();
        if (testPtT->deleted()) {
            continue;
        }
        if (testSeg == baseSeg) {
            continue;
        }
        if (testPtT->span()->ptT() != testPtT) {
            continue;
        }
        if (this->contains(baseSeg, testSeg, testPtT->fT)) {
            continue;
        }
        // intersect perp with base->ptT() with testPtT->segment()
        SkDVector dxdy = baseSeg->dSlopeAtT(base->t());
        const SkPoint& pt = base->pt();
        SkDLine ray = {{{pt.fX, pt.fY}, {pt.fX + dxdy.fY, pt.fY - dxdy.fX}}};
        SkIntersections i;
        (*CurveIntersectRay[testSeg->verb()])(testSeg->pts(), testSeg->weight(), ray, &i);
        for (int index = 0; index < i.used(); ++index) {
            double t = i[0][index];
            if (!between(0, t, 1)) {
                continue;
            }
            SkDPoint oppPt = i.pt(index);
            if (!oppPt.approximatelyEqual(pt)) {
                continue;
            }
            SkOpSegment* writableSeg = const_cast<SkOpSegment*>(testSeg);
            SkOpPtT* oppStart = writableSeg->addT(t);
            if (oppStart == testPtT) {
                continue;
            }
            SkOpSpan* writableBase = const_cast<SkOpSpan*>(base);
            oppStart->span()->addOpp(writableBase);
            if (oppStart->deleted()) {
                continue;
            }
            SkOpSegment* coinSeg = base->segment();
            SkOpSegment* oppSeg = oppStart->segment();
            double coinTs, coinTe, oppTs, oppTe;
            if (Ordered(coinSeg, oppSeg)) {
                coinTs = base->t();
                coinTe = testSpan->t();
                oppTs = oppStart->fT;
                oppTe = testPtT->fT;
            } else {
                using std::swap;
                swap(coinSeg, oppSeg);
                coinTs = oppStart->fT;
                coinTe = testPtT->fT;
                oppTs = base->t();
                oppTe = testSpan->t();
            }
            if (coinTs > coinTe) {
                using std::swap;
                swap(coinTs, coinTe);
                swap(oppTs, oppTe);
            }
            bool added;
            if (this->debugAddOrOverlap(log, coinSeg, oppSeg, coinTs, coinTe, oppTs, oppTe, &added), false) {
                return;
            }
        }
    }
    return;
}

// description below
void SkOpCoincidence::debugAddEndMovedSpans(SkPathOpsDebug::GlitchLog* log, const SkOpPtT* ptT) const {
    FAIL_IF(!ptT->span()->upCastable(), ptT->span());
    const SkOpSpan* base = ptT->span()->upCast();
    const SkOpSpan* prev = base->prev();
    FAIL_IF(!prev, ptT->span());
    if (!prev->isCanceled()) {
        if (this->debugAddEndMovedSpans(log, base, base->prev()), false) {
            return;
        }
    }
    if (!base->isCanceled()) {
        if (this->debugAddEndMovedSpans(log, base, base->next()), false) {
            return;
        }
    }
    return;
}

/*  If A is coincident with B and B includes an endpoint, and A's matching point
    is not the endpoint (i.e., there's an implied line connecting B-end and A)
    then assume that the same implied line may intersect another curve close to B.
    Since we only care about coincidence that was undetected, look at the
    ptT list on B-segment adjacent to the B-end/A ptT loop (not in the loop, but
    next door) and see if the A matching point is close enough to form another
    coincident pair. If so, check for a new coincident span between B-end/A ptT loop
    and the adjacent ptT loop.
*/
void SkOpCoincidence::debugAddEndMovedSpans(SkPathOpsDebug::GlitchLog* log) const {
    const SkCoincidentSpans* span = fHead;
    if (!span) {
        return;
    }
//    fTop = span;
//    fHead = nullptr;
    do {
        if (span->coinPtTStart()->fPt != span->oppPtTStart()->fPt) {
            FAIL_IF(1 == span->coinPtTStart()->fT, span);
            bool onEnd = span->coinPtTStart()->fT == 0;
            bool oOnEnd = zero_or_one(span->oppPtTStart()->fT);
            if (onEnd) {
                if (!oOnEnd) {  // if both are on end, any nearby intersect was already found
                    if (this->debugAddEndMovedSpans(log, span->oppPtTStart()), false) {
                        return;
                    }
                }
            } else if (oOnEnd) {
                if (this->debugAddEndMovedSpans(log, span->coinPtTStart()), false) {
                    return;
                }
            }
        }
        if (span->coinPtTEnd()->fPt != span->oppPtTEnd()->fPt) {
            bool onEnd = span->coinPtTEnd()->fT == 1;
            bool oOnEnd = zero_or_one(span->oppPtTEnd()->fT);
            if (onEnd) {
                if (!oOnEnd) {
                    if (this->debugAddEndMovedSpans(log, span->oppPtTEnd()), false) {
                        return;
                    }
                }
            } else if (oOnEnd) {
                if (this->debugAddEndMovedSpans(log, span->coinPtTEnd()), false) {
                    return;
                }
            }
        }
    } while ((span = span->next()));
//    this->restoreHead();
    return;
}

/* Commented-out lines keep this in sync with addExpanded */
// for each coincident pair, match the spans
// if the spans don't match, add the mssing pt to the segment and loop it in the opposite span
void SkOpCoincidence::debugAddExpanded(SkPathOpsDebug::GlitchLog* log) const {
//    DEBUG_SET_PHASE();
    const SkCoincidentSpans* coin = this->fHead;
    if (!coin) {
        return;
    }
    do {
        const SkOpPtT* startPtT = coin->coinPtTStart();
        const SkOpPtT* oStartPtT = coin->oppPtTStart();
        double priorT = startPtT->fT;
        double oPriorT = oStartPtT->fT;
        FAIL_IF(!startPtT->contains(oStartPtT), coin);
        SkOPASSERT(coin->coinPtTEnd()->contains(coin->oppPtTEnd()));
        const SkOpSpanBase* start = startPtT->span();
        const SkOpSpanBase* oStart = oStartPtT->span();
        const SkOpSpanBase* end = coin->coinPtTEnd()->span();
        const SkOpSpanBase* oEnd = coin->oppPtTEnd()->span();
        FAIL_IF(oEnd->deleted(), coin);
        FAIL_IF(!start->upCastable(), coin);
        const SkOpSpanBase* test = start->upCast()->next();
        FAIL_IF(!coin->flipped() && !oStart->upCastable(), coin);
        const SkOpSpanBase* oTest = coin->flipped() ? oStart->prev() : oStart->upCast()->next();
        FAIL_IF(!oTest, coin);
        const SkOpSegment* seg = start->segment();
        const SkOpSegment* oSeg = oStart->segment();
        while (test != end || oTest != oEnd) {
            const SkOpPtT* containedOpp = test->ptT()->contains(oSeg);
            const SkOpPtT* containedThis = oTest->ptT()->contains(seg);
            if (!containedOpp || !containedThis) {
                // choose the ends, or the first common pt-t list shared by both
                double nextT, oNextT;
                if (containedOpp) {
                    nextT = test->t();
                    oNextT = containedOpp->fT;
                } else if (containedThis) {
                    nextT = containedThis->fT;
                    oNextT = oTest->t();
                } else {
                    // iterate through until a pt-t list found that contains the other
                    const SkOpSpanBase* walk = test;
                    const SkOpPtT* walkOpp;
                    do {
                        FAIL_IF(!walk->upCastable(), coin);
                        walk = walk->upCast()->next();
                    } while (!(walkOpp = walk->ptT()->contains(oSeg))
                            && walk != coin->coinPtTEnd()->span());
                    FAIL_IF(!walkOpp, coin);
                    nextT = walk->t();
                    oNextT = walkOpp->fT;
                }
                // use t ranges to guess which one is missing
                double startRange = nextT - priorT;
                FAIL_IF(!startRange, coin);
                double startPart = (test->t() - priorT) / startRange;
                double oStartRange = oNextT - oPriorT;
                FAIL_IF(!oStartRange, coin);
                double oStartPart = (oTest->t() - oStartPtT->fT) / oStartRange;
                FAIL_IF(startPart == oStartPart, coin);
                bool addToOpp = !containedOpp && !containedThis ? startPart < oStartPart
                        : !!containedThis;
                bool startOver = false;
                addToOpp ? log->record(SkPathOpsDebug::kAddExpandedCoin_Glitch,
                        oPriorT + oStartRange * startPart, test)
                        : log->record(SkPathOpsDebug::kAddExpandedCoin_Glitch,
                        priorT + startRange * oStartPart, oTest);
         //       FAIL_IF(!success, coin);
                if (startOver) {
                    test = start;
                    oTest = oStart;
                }
                end = coin->coinPtTEnd()->span();
                oEnd = coin->oppPtTEnd()->span();
            }
            if (test != end) {
                FAIL_IF(!test->upCastable(), coin);
                priorT = test->t();
                test = test->upCast()->next();
            }
            if (oTest != oEnd) {
                oPriorT = oTest->t();
                oTest = coin->flipped() ? oTest->prev() : oTest->upCast()->next();
                FAIL_IF(!oTest, coin);
            }
        }
    } while ((coin = coin->next()));
    return;
}

/* Commented-out lines keep this in sync addIfMissing() */
// note that over1s, over1e, over2s, over2e are ordered
void SkOpCoincidence::debugAddIfMissing(SkPathOpsDebug::GlitchLog* log, const SkOpPtT* over1s, const SkOpPtT* over2s,
        double tStart, double tEnd, const SkOpSegment* coinSeg, const SkOpSegment* oppSeg, bool* added,
        const SkOpPtT* over1e, const SkOpPtT* over2e) const {
    SkASSERT(tStart < tEnd);
    SkASSERT(over1s->fT < over1e->fT);
    SkASSERT(between(over1s->fT, tStart, over1e->fT));
    SkASSERT(between(over1s->fT, tEnd, over1e->fT));
    SkASSERT(over2s->fT < over2e->fT);
    SkASSERT(between(over2s->fT, tStart, over2e->fT));
    SkASSERT(between(over2s->fT, tEnd, over2e->fT));
    SkASSERT(over1s->segment() == over1e->segment());
    SkASSERT(over2s->segment() == over2e->segment());
    SkASSERT(over1s->segment() == over2s->segment());
    SkASSERT(over1s->segment() != coinSeg);
    SkASSERT(over1s->segment() != oppSeg);
    SkASSERT(coinSeg != oppSeg);
    double coinTs, coinTe, oppTs, oppTe;
    coinTs = TRange(over1s, tStart, coinSeg  SkDEBUGPARAMS(over1e));
    coinTe = TRange(over1s, tEnd, coinSeg  SkDEBUGPARAMS(over1e));
    SkOpSpanBase::Collapsed result = coinSeg->collapsed(coinTs, coinTe);
    if (SkOpSpanBase::Collapsed::kNo != result) {
        return log->record(SkPathOpsDebug::kAddIfCollapsed_Glitch, coinSeg);
    }
    oppTs = TRange(over2s, tStart, oppSeg  SkDEBUGPARAMS(over2e));
    oppTe = TRange(over2s, tEnd, oppSeg  SkDEBUGPARAMS(over2e));
    result = oppSeg->collapsed(oppTs, oppTe);
    if (SkOpSpanBase::Collapsed::kNo != result) {
        return log->record(SkPathOpsDebug::kAddIfCollapsed_Glitch, oppSeg);
    }
    if (coinTs > coinTe) {
        using std::swap;
        swap(coinTs, coinTe);
        swap(oppTs, oppTe);
    }
    this->debugAddOrOverlap(log, coinSeg, oppSeg, coinTs, coinTe, oppTs, oppTe, added);
    return;
}

/* Commented-out lines keep this in sync addOrOverlap() */
// If this is called by addEndMovedSpans(), a returned false propogates out to an abort.
// If this is called by AddIfMissing(), a returned false indicates there was nothing to add
void SkOpCoincidence::debugAddOrOverlap(SkPathOpsDebug::GlitchLog* log,
        const SkOpSegment* coinSeg, const SkOpSegment* oppSeg,
        double coinTs, double coinTe, double oppTs, double oppTe, bool* added) const {
    SkTDArray<SkCoincidentSpans*> overlaps;
    SkOPASSERT(!fTop);   // this is (correctly) reversed in addifMissing()
    if (fTop && !this->checkOverlap(fTop, coinSeg, oppSeg, coinTs, coinTe, oppTs, oppTe,
            &overlaps)) {
        return;
    }
    if (fHead && !this->checkOverlap(fHead, coinSeg, oppSeg, coinTs,
            coinTe, oppTs, oppTe, &overlaps)) {
        return;
    }
    const SkCoincidentSpans* overlap = overlaps.count() ? overlaps[0] : nullptr;
    for (int index = 1; index < overlaps.count(); ++index) { // combine overlaps before continuing
        const SkCoincidentSpans* test = overlaps[index];
        if (overlap->coinPtTStart()->fT > test->coinPtTStart()->fT) {
            log->record(SkPathOpsDebug::kAddOrOverlap_Glitch, overlap, test->coinPtTStart());
        }
        if (overlap->coinPtTEnd()->fT < test->coinPtTEnd()->fT) {
            log->record(SkPathOpsDebug::kAddOrOverlap_Glitch, overlap, test->coinPtTEnd());
        }
        if (overlap->flipped()
                ? overlap->oppPtTStart()->fT < test->oppPtTStart()->fT
                : overlap->oppPtTStart()->fT > test->oppPtTStart()->fT) {
            log->record(SkPathOpsDebug::kAddOrOverlap_Glitch, overlap, test->oppPtTStart());
        }
        if (overlap->flipped()
                ? overlap->oppPtTEnd()->fT > test->oppPtTEnd()->fT
                : overlap->oppPtTEnd()->fT < test->oppPtTEnd()->fT) {
            log->record(SkPathOpsDebug::kAddOrOverlap_Glitch, overlap, test->oppPtTEnd());
        }
        if (!fHead) { this->debugRelease(log, fHead, test);
            this->debugRelease(log, fTop, test);
        }
    }
    const SkOpPtT* cs = coinSeg->existing(coinTs, oppSeg);
    const SkOpPtT* ce = coinSeg->existing(coinTe, oppSeg);
    RETURN_FALSE_IF(overlap && cs && ce && overlap->contains(cs, ce), coinSeg);
    RETURN_FALSE_IF(cs != ce || !cs, coinSeg);
    const SkOpPtT* os = oppSeg->existing(oppTs, coinSeg);
    const SkOpPtT* oe = oppSeg->existing(oppTe, coinSeg);
    RETURN_FALSE_IF(overlap && os && oe && overlap->contains(os, oe), oppSeg);
    SkASSERT(true || !cs || !cs->deleted());
    SkASSERT(true || !os || !os->deleted());
    SkASSERT(true || !ce || !ce->deleted());
    SkASSERT(true || !oe || !oe->deleted());
    const SkOpPtT* csExisting = !cs ? coinSeg->existing(coinTs, nullptr) : nullptr;
    const SkOpPtT* ceExisting = !ce ? coinSeg->existing(coinTe, nullptr) : nullptr;
    RETURN_FALSE_IF(csExisting && csExisting == ceExisting, coinSeg);
    RETURN_FALSE_IF(csExisting && (csExisting == ce ||
            csExisting->contains(ceExisting ? ceExisting : ce)), coinSeg);
    RETURN_FALSE_IF(ceExisting && (ceExisting == cs ||
            ceExisting->contains(csExisting ? csExisting : cs)), coinSeg);
    const SkOpPtT* osExisting = !os ? oppSeg->existing(oppTs, nullptr) : nullptr;
    const SkOpPtT* oeExisting = !oe ? oppSeg->existing(oppTe, nullptr) : nullptr;
    RETURN_FALSE_IF(osExisting && osExisting == oeExisting, oppSeg);
    RETURN_FALSE_IF(osExisting && (osExisting == oe ||
            osExisting->contains(oeExisting ? oeExisting : oe)), oppSeg);
    RETURN_FALSE_IF(oeExisting && (oeExisting == os ||
            oeExisting->contains(osExisting ? osExisting : os)), oppSeg);
    bool csDeleted = false, osDeleted = false, ceDeleted = false,  oeDeleted = false;
    this->debugValidate();
    if (!cs || !os) {
        if (!cs)
            cs = coinSeg->debugAddT(coinTs, log);
        if (!os)
            os = oppSeg->debugAddT(oppTs, log);
//      RETURN_FALSE_IF(callerAborts, !csWritable || !osWritable);
        if (cs && os) cs->span()->debugAddOpp(log, os->span());
//         cs = csWritable;
//         os = osWritable->active();
        RETURN_FALSE_IF((ce && ce->deleted()) || (oe && oe->deleted()), coinSeg);
    }
    if (!ce || !oe) {
        if (!ce)
            ce = coinSeg->debugAddT(coinTe, log);
        if (!oe)
            oe = oppSeg->debugAddT(oppTe, log);
        if (ce && oe) ce->span()->debugAddOpp(log, oe->span());
//         ce = ceWritable;
//         oe = oeWritable;
    }
    this->debugValidate();
    RETURN_FALSE_IF(csDeleted, coinSeg);
    RETURN_FALSE_IF(osDeleted, oppSeg);
    RETURN_FALSE_IF(ceDeleted, coinSeg);
    RETURN_FALSE_IF(oeDeleted, oppSeg);
    RETURN_FALSE_IF(!cs || !ce || cs == ce || cs->contains(ce) || !os || !oe || os == oe || os->contains(oe), coinSeg);
    bool result = true;
    if (overlap) {
        if (overlap->coinPtTStart()->segment() == coinSeg) {
                log->record(SkPathOpsDebug::kAddMissingExtend_Glitch, coinSeg, coinTs, coinTe, oppSeg, oppTs, oppTe);
        } else {
            if (oppTs > oppTe) {
                using std::swap;
                swap(coinTs, coinTe);
                swap(oppTs, oppTe);
            }
            log->record(SkPathOpsDebug::kAddMissingExtend_Glitch, oppSeg, oppTs, oppTe, coinSeg, coinTs, coinTe);
        }
#if 0 && DEBUG_COINCIDENCE_VERBOSE
        if (result) {
             overlap->debugShow();
        }
#endif
    } else {
        log->record(SkPathOpsDebug::kAddMissingCoin_Glitch, coinSeg, coinTs, coinTe, oppSeg, oppTs, oppTe);
#if 0 && DEBUG_COINCIDENCE_VERBOSE
        fHead->debugShow();
#endif
    }
    this->debugValidate();
    return (void) result;
}

// Extra commented-out lines keep this in sync with addMissing()
/* detects overlaps of different coincident runs on same segment */
/* does not detect overlaps for pairs without any segments in common */
// returns true if caller should loop again
void SkOpCoincidence::debugAddMissing(SkPathOpsDebug::GlitchLog* log, bool* added) const {
    const SkCoincidentSpans* outer = fHead;
    *added = false;
    if (!outer) {
        return;
    }
    // fTop = outer;
    // fHead = nullptr;
    do {
    // addifmissing can modify the list that this is walking
    // save head so that walker can iterate over old data unperturbed
    // addifmissing adds to head freely then add saved head in the end
        const SkOpPtT* ocs = outer->coinPtTStart();
        SkASSERT(!ocs->deleted());
        const SkOpSegment* outerCoin = ocs->segment();
        SkASSERT(!outerCoin->done());  // if it's done, should have already been removed from list
        const SkOpPtT* oos = outer->oppPtTStart();
        if (oos->deleted()) {
            return;
        }
        const SkOpSegment* outerOpp = oos->segment();
        SkASSERT(!outerOpp->done());
//        SkOpSegment* outerCoinWritable = const_cast<SkOpSegment*>(outerCoin);
//        SkOpSegment* outerOppWritable = const_cast<SkOpSegment*>(outerOpp);
        const SkCoincidentSpans* inner = outer;
        while ((inner = inner->next())) {
            this->debugValidate();
            double overS, overE;
            const SkOpPtT* ics = inner->coinPtTStart();
            SkASSERT(!ics->deleted());
            const SkOpSegment* innerCoin = ics->segment();
            SkASSERT(!innerCoin->done());
            const SkOpPtT* ios = inner->oppPtTStart();
            SkASSERT(!ios->deleted());
            const SkOpSegment* innerOpp = ios->segment();
            SkASSERT(!innerOpp->done());
//            SkOpSegment* innerCoinWritable = const_cast<SkOpSegment*>(innerCoin);
//            SkOpSegment* innerOppWritable = const_cast<SkOpSegment*>(innerOpp);
            if (outerCoin == innerCoin) {
                const SkOpPtT* oce = outer->coinPtTEnd();
                if (oce->deleted()) {
                    return;
                }
                const SkOpPtT* ice = inner->coinPtTEnd();
                SkASSERT(!ice->deleted());
                if (outerOpp != innerOpp && this->overlap(ocs, oce, ics, ice, &overS, &overE)) {
                    this->debugAddIfMissing(log, ocs->starter(oce), ics->starter(ice),
                            overS, overE, outerOpp, innerOpp, added,
                            ocs->debugEnder(oce),
                            ics->debugEnder(ice));
                }
            } else if (outerCoin == innerOpp) {
                const SkOpPtT* oce = outer->coinPtTEnd();
                SkASSERT(!oce->deleted());
                const SkOpPtT* ioe = inner->oppPtTEnd();
                SkASSERT(!ioe->deleted());
                if (outerOpp != innerCoin && this->overlap(ocs, oce, ios, ioe, &overS, &overE)) {
                    this->debugAddIfMissing(log, ocs->starter(oce), ios->starter(ioe),
                            overS, overE, outerOpp, innerCoin, added,
                            ocs->debugEnder(oce),
                            ios->debugEnder(ioe));
                }
            } else if (outerOpp == innerCoin) {
                const SkOpPtT* ooe = outer->oppPtTEnd();
                SkASSERT(!ooe->deleted());
                const SkOpPtT* ice = inner->coinPtTEnd();
                SkASSERT(!ice->deleted());
                SkASSERT(outerCoin != innerOpp);
                if (this->overlap(oos, ooe, ics, ice, &overS, &overE)) {
                    this->debugAddIfMissing(log, oos->starter(ooe), ics->starter(ice),
                            overS, overE, outerCoin, innerOpp, added,
                            oos->debugEnder(ooe),
                            ics->debugEnder(ice));
                }
            } else if (outerOpp == innerOpp) {
                const SkOpPtT* ooe = outer->oppPtTEnd();
                SkASSERT(!ooe->deleted());
                const SkOpPtT* ioe = inner->oppPtTEnd();
                if (ioe->deleted()) {
                    return;
                }
                SkASSERT(outerCoin != innerCoin);
                if (this->overlap(oos, ooe, ios, ioe, &overS, &overE)) {
                    this->debugAddIfMissing(log, oos->starter(ooe), ios->starter(ioe),
                            overS, overE, outerCoin, innerCoin, added,
                            oos->debugEnder(ooe),
                            ios->debugEnder(ioe));
                }
            }
            this->debugValidate();
        }
    } while ((outer = outer->next()));
    // this->restoreHead();
    return;
}

// Commented-out lines keep this in sync with release()
void SkOpCoincidence::debugRelease(SkPathOpsDebug::GlitchLog* log, const SkCoincidentSpans* coin, const SkCoincidentSpans* remove) const {
    const SkCoincidentSpans* head = coin;
    const SkCoincidentSpans* prev = nullptr;
    const SkCoincidentSpans* next;
    do {
        next = coin->next();
        if (coin == remove) {
            if (prev) {
//                prev->setNext(next);
            } else if (head == fHead) {
//                fHead = next;
            } else {
//                fTop = next;
            }
            log->record(SkPathOpsDebug::kReleasedSpan_Glitch, coin);
        }
        prev = coin;
    } while ((coin = next));
    return;
}

void SkOpCoincidence::debugRelease(SkPathOpsDebug::GlitchLog* log, const SkOpSegment* deleted) const {
    const SkCoincidentSpans* coin = fHead;
    if (!coin) {
        return;
    }
    do {
        if (coin->coinPtTStart()->segment() == deleted
                || coin->coinPtTEnd()->segment() == deleted
                || coin->oppPtTStart()->segment() == deleted
                || coin->oppPtTEnd()->segment() == deleted) {
            log->record(SkPathOpsDebug::kReleasedSpan_Glitch, coin);
        }
    } while ((coin = coin->next()));
}

// Commented-out lines keep this in sync with expand()
// expand the range by checking adjacent spans for coincidence
bool SkOpCoincidence::debugExpand(SkPathOpsDebug::GlitchLog* log) const {
    const SkCoincidentSpans* coin = fHead;
    if (!coin) {
        return false;
    }
    bool expanded = false;
    do {
        if (coin->debugExpand(log)) {
            // check to see if multiple spans expanded so they are now identical
            const SkCoincidentSpans* test = fHead;
            do {
                if (coin == test) {
                    continue;
                }
                if (coin->coinPtTStart() == test->coinPtTStart()
                        && coin->oppPtTStart() == test->oppPtTStart()) {
                    if (log) log->record(SkPathOpsDebug::kExpandCoin_Glitch, fHead, test->coinPtTStart());
                    break;
                }
            } while ((test = test->next()));
            expanded = true;
        }
    } while ((coin = coin->next()));
    return expanded;
}

// Commented-out lines keep this in sync with mark()
/* this sets up the coincidence links in the segments when the coincidence crosses multiple spans */
void SkOpCoincidence::debugMark(SkPathOpsDebug::GlitchLog* log) const {
    const SkCoincidentSpans* coin = fHead;
    if (!coin) {
        return;
    }
    do {
        FAIL_IF(!coin->coinPtTStartWritable()->span()->upCastable(), coin);
        const SkOpSpan* start = coin->coinPtTStartWritable()->span()->upCast();
//         SkASSERT(start->deleted());
        const SkOpSpanBase* end = coin->coinPtTEndWritable()->span();
//         SkASSERT(end->deleted());
        const SkOpSpanBase* oStart = coin->oppPtTStartWritable()->span();
//         SkASSERT(oStart->deleted());
        const SkOpSpanBase* oEnd = coin->oppPtTEndWritable()->span();
//         SkASSERT(oEnd->deleted());
        bool flipped = coin->flipped();
        if (flipped) {
            using std::swap;
            swap(oStart, oEnd);
        }
        /* coin and opp spans may not match up. Mark the ends, and then let the interior
           get marked as many times as the spans allow */
        start->debugInsertCoincidence(log, oStart->upCast());
        end->debugInsertCoinEnd(log, oEnd);
        const SkOpSegment* segment = start->segment();
        const SkOpSegment* oSegment = oStart->segment();
        const SkOpSpanBase* next = start;
        const SkOpSpanBase* oNext = oStart;
        bool ordered;
        FAIL_IF(!coin->ordered(&ordered), coin);
        while ((next = next->upCast()->next()) != end) {
            FAIL_IF(!next->upCastable(), coin);
            if (next->upCast()->debugInsertCoincidence(log, oSegment, flipped, ordered), false) {
                return;
            }
        }
        while ((oNext = oNext->upCast()->next()) != oEnd) {
            FAIL_IF(!oNext->upCastable(), coin);
            if (oNext->upCast()->debugInsertCoincidence(log, segment, flipped, ordered), false) {
                return;
            }
        }
    } while ((coin = coin->next()));
    return;
}
#endif

#if DEBUG_COIN
// Commented-out lines keep this in sync with markCollapsed()
void SkOpCoincidence::debugMarkCollapsed(SkPathOpsDebug::GlitchLog* log, const SkCoincidentSpans* coin, const SkOpPtT* test) const {
    const SkCoincidentSpans* head = coin;
    while (coin) {
        if (coin->collapsed(test)) {
            if (zero_or_one(coin->coinPtTStart()->fT) && zero_or_one(coin->coinPtTEnd()->fT)) {
                log->record(SkPathOpsDebug::kCollapsedCoin_Glitch, coin);
            }
            if (zero_or_one(coin->oppPtTStart()->fT) && zero_or_one(coin->oppPtTEnd()->fT)) {
                log->record(SkPathOpsDebug::kCollapsedCoin_Glitch, coin);
            }
            this->debugRelease(log, head, coin);
        }
        coin = coin->next();
    }
}

// Commented-out lines keep this in sync with markCollapsed()
void SkOpCoincidence::debugMarkCollapsed(SkPathOpsDebug::GlitchLog* log, const SkOpPtT* test) const {
    this->debugMarkCollapsed(log, fHead, test);
    this->debugMarkCollapsed(log, fTop, test);
}
#endif

void SkCoincidentSpans::debugShow() const {
    SkDebugf("coinSpan - id=%d t=%1.9g tEnd=%1.9g\n", coinPtTStart()->segment()->debugID(),
            coinPtTStart()->fT, coinPtTEnd()->fT);
    SkDebugf("coinSpan + id=%d t=%1.9g tEnd=%1.9g\n", oppPtTStart()->segment()->debugID(),
            oppPtTStart()->fT, oppPtTEnd()->fT);
}

void SkOpCoincidence::debugShowCoincidence() const {
#if DEBUG_COINCIDENCE
    const SkCoincidentSpans* span = fHead;
    while (span) {
        span->debugShow();
        span = span->next();
    }
#endif
}

#if DEBUG_COIN
static void DebugCheckBetween(const SkOpSpanBase* next, const SkOpSpanBase* end,
        double oStart, double oEnd, const SkOpSegment* oSegment,
        SkPathOpsDebug::GlitchLog* log) {
    SkASSERT(next != end);
    SkASSERT(!next->contains(end) || log);
    if (next->t() > end->t()) {
        using std::swap;
        swap(next, end);
    }
    do {
        const SkOpPtT* ptT = next->ptT();
        int index = 0;
        bool somethingBetween = false;
        do {
            ++index;
            ptT = ptT->next();
            const SkOpPtT* checkPtT = next->ptT();
            if (ptT == checkPtT) {
                break;
            }
            bool looped = false;
            for (int check = 0; check < index; ++check) {
                if ((looped = checkPtT == ptT)) {
                    break;
                }
                checkPtT = checkPtT->next();
            }
            if (looped) {
                SkASSERT(0);
                break;
            }
            if (ptT->deleted()) {
                continue;
            }
            if (ptT->segment() != oSegment) {
                continue;
            }
            somethingBetween |= between(oStart, ptT->fT, oEnd);
        } while (true);
        SkASSERT(somethingBetween);
    } while (next != end && (next = next->upCast()->next()));
}

static void DebugCheckOverlap(const SkCoincidentSpans* test, const SkCoincidentSpans* list,
        SkPathOpsDebug::GlitchLog* log) {
    if (!list) {
        return;
    }
    const SkOpSegment* coinSeg = test->coinPtTStart()->segment();
    SkASSERT(coinSeg == test->coinPtTEnd()->segment());
    const SkOpSegment* oppSeg = test->oppPtTStart()->segment();
    SkASSERT(oppSeg == test->oppPtTEnd()->segment());
    SkASSERT(coinSeg != test->oppPtTStart()->segment());
    SkDEBUGCODE(double tcs = test->coinPtTStart()->fT);
    SkASSERT(between(0, tcs, 1));
    SkDEBUGCODE(double tce = test->coinPtTEnd()->fT);
    SkASSERT(between(0, tce, 1));
    SkASSERT(tcs < tce);
    double tos = test->oppPtTStart()->fT;
    SkASSERT(between(0, tos, 1));
    double toe = test->oppPtTEnd()->fT;
    SkASSERT(between(0, toe, 1));
    SkASSERT(tos != toe);
    if (tos > toe) {
        using std::swap;
        swap(tos, toe);
    }
    do {
        double lcs, lce, los, loe;
        if (coinSeg == list->coinPtTStart()->segment()) {
            if (oppSeg != list->oppPtTStart()->segment()) {
                continue;
            }
            lcs = list->coinPtTStart()->fT;
            lce = list->coinPtTEnd()->fT;
            los = list->oppPtTStart()->fT;
            loe = list->oppPtTEnd()->fT;
            if (los > loe) {
                using std::swap;
                swap(los, loe);
            }
        } else if (coinSeg == list->oppPtTStart()->segment()) {
            if (oppSeg != list->coinPtTStart()->segment()) {
                continue;
            }
            lcs = list->oppPtTStart()->fT;
            lce = list->oppPtTEnd()->fT;
            if (lcs > lce) {
                using std::swap;
                swap(lcs, lce);
            }
            los = list->coinPtTStart()->fT;
            loe = list->coinPtTEnd()->fT;
        } else {
            continue;
        }
        SkASSERT(tce < lcs || lce < tcs);
        SkASSERT(toe < los || loe < tos);
    } while ((list = list->next()));
}


static void DebugCheckOverlapTop(const SkCoincidentSpans* head, const SkCoincidentSpans* opt,
        SkPathOpsDebug::GlitchLog* log) {
    // check for overlapping coincident spans
    const SkCoincidentSpans* test = head;
    while (test) {
        const SkCoincidentSpans* next = test->next();
        DebugCheckOverlap(test, next, log);
        DebugCheckOverlap(test, opt, log);
        test = next;
    }
}

static void DebugValidate(const SkCoincidentSpans* head, const SkCoincidentSpans* opt,
        SkPathOpsDebug::GlitchLog* log) {
    // look for pts inside coincident spans that are not inside the opposite spans
    const SkCoincidentSpans* coin = head;
    while (coin) {
        SkASSERT(SkOpCoincidence::Ordered(coin->coinPtTStart()->segment(),
                coin->oppPtTStart()->segment()));
        SkASSERT(coin->coinPtTStart()->span()->ptT() == coin->coinPtTStart());
        SkASSERT(coin->coinPtTEnd()->span()->ptT() == coin->coinPtTEnd());
        SkASSERT(coin->oppPtTStart()->span()->ptT() == coin->oppPtTStart());
        SkASSERT(coin->oppPtTEnd()->span()->ptT() == coin->oppPtTEnd());
        coin = coin->next();
    }
    DebugCheckOverlapTop(head, opt, log);
}
#endif

void SkOpCoincidence::debugValidate() const {
#if DEBUG_COINCIDENCE
    DebugValidate(fHead, fTop, nullptr);
    DebugValidate(fTop, nullptr, nullptr);
#endif
}

#if DEBUG_COIN
static void DebugCheckBetween(const SkCoincidentSpans* head, const SkCoincidentSpans* opt,
        SkPathOpsDebug::GlitchLog* log) {
    // look for pts inside coincident spans that are not inside the opposite spans
    const SkCoincidentSpans* coin = head;
    while (coin) {
        DebugCheckBetween(coin->coinPtTStart()->span(), coin->coinPtTEnd()->span(),
                coin->oppPtTStart()->fT, coin->oppPtTEnd()->fT, coin->oppPtTStart()->segment(),
                log);
        DebugCheckBetween(coin->oppPtTStart()->span(), coin->oppPtTEnd()->span(),
                coin->coinPtTStart()->fT, coin->coinPtTEnd()->fT, coin->coinPtTStart()->segment(),
                log);
        coin = coin->next();
    }
    DebugCheckOverlapTop(head, opt, log);
}
#endif

void SkOpCoincidence::debugCheckBetween() const {
#if DEBUG_COINCIDENCE
    if (fGlobalState->debugCheckHealth()) {
        return;
    }
    DebugCheckBetween(fHead, fTop, nullptr);
    DebugCheckBetween(fTop, nullptr, nullptr);
#endif
}

#if DEBUG_COIN
void SkOpContour::debugCheckHealth(SkPathOpsDebug::GlitchLog* log) const {
    const SkOpSegment* segment = &fHead;
    do {
        segment->debugCheckHealth(log);
    } while ((segment = segment->next()));
}

void SkOpCoincidence::debugCheckValid(SkPathOpsDebug::GlitchLog* log) const {
#if DEBUG_VALIDATE
    DebugValidate(fHead, fTop, log);
    DebugValidate(fTop, nullptr, log);
#endif
}

void SkOpCoincidence::debugCorrectEnds(SkPathOpsDebug::GlitchLog* log) const {
    const SkCoincidentSpans* coin = fHead;
    if (!coin) {
        return;
    }
    do {
        coin->debugCorrectEnds(log);
    } while ((coin = coin->next()));
}

// commmented-out lines keep this aligned with missingCoincidence()
void SkOpContour::debugMissingCoincidence(SkPathOpsDebug::GlitchLog* log) const {
//    SkASSERT(fCount > 0);
    const SkOpSegment* segment = &fHead;
//    bool result = false;
    do {
        if (segment->debugMissingCoincidence(log), false) {
//          result = true;
        }
        segment = segment->next();
    } while (segment);
    return;
}

void SkOpContour::debugMoveMultiples(SkPathOpsDebug::GlitchLog* log) const {
    SkASSERT(fCount > 0);
    const SkOpSegment* segment = &fHead;
    do {
        if (segment->debugMoveMultiples(log), false) {
            return;
        }
    } while ((segment = segment->next()));
    return;
}

void SkOpContour::debugMoveNearby(SkPathOpsDebug::GlitchLog* log) const {
    SkASSERT(fCount > 0);
    const SkOpSegment* segment = &fHead;
    do {
        segment->debugMoveNearby(log);
    } while ((segment = segment->next()));
}
#endif

#if DEBUG_COINCIDENCE_ORDER
void SkOpSegment::debugResetCoinT() const {
    fDebugBaseIndex = -1;
    fDebugBaseMin = 1;
    fDebugBaseMax = -1;
    fDebugLastIndex = -1;
    fDebugLastMin = 1;
    fDebugLastMax = -1;
}
#endif

void SkOpSegment::debugValidate() const {
#if DEBUG_COINCIDENCE_ORDER
    {
        const SkOpSpanBase* span = &fHead;
        do {
            span->debugResetCoinT();
        } while (!span->final() && (span = span->upCast()->next()));
        span = &fHead;
        int index = 0;
        do {
            span->debugSetCoinT(index++);
        } while (!span->final() && (span = span->upCast()->next()));
    }
#endif
#if DEBUG_COINCIDENCE
    if (this->globalState()->debugCheckHealth()) {
        return;
    }
#endif
#if DEBUG_VALIDATE
    const SkOpSpanBase* span = &fHead;
    double lastT = -1;
    const SkOpSpanBase* prev = nullptr;
    int count = 0;
    int done = 0;
    do {
        if (!span->final()) {
            ++count;
            done += span->upCast()->done() ? 1 : 0;
        }
        SkASSERT(span->segment() == this);
        SkASSERT(!prev || prev->upCast()->next() == span);
        SkASSERT(!prev || prev == span->prev());
        prev = span;
        double t = span->ptT()->fT;
        SkASSERT(lastT < t);
        lastT = t;
        span->debugValidate();
    } while (!span->final() && (span = span->upCast()->next()));
    SkASSERT(count == fCount);
    SkASSERT(done == fDoneCount);
    SkASSERT(count >= fDoneCount);
    SkASSERT(span->final());
    span->debugValidate();
#endif
}

#if DEBUG_COIN

// Commented-out lines keep this in sync with addOpp()
void SkOpSpanBase::debugAddOpp(SkPathOpsDebug::GlitchLog* log, const SkOpSpanBase* opp) const {
    const SkOpPtT* oppPrev = this->ptT()->oppPrev(opp->ptT());
    if (!oppPrev) {
        return;
    }
    this->debugMergeMatches(log, opp);
    this->ptT()->debugAddOpp(opp->ptT(), oppPrev);
    this->debugCheckForCollapsedCoincidence(log);
}

// Commented-out lines keep this in sync with checkForCollapsedCoincidence()
void SkOpSpanBase::debugCheckForCollapsedCoincidence(SkPathOpsDebug::GlitchLog* log) const {
    const SkOpCoincidence* coins = this->globalState()->coincidence();
    if (coins->isEmpty()) {
        return;
    }
// the insert above may have put both ends of a coincident run in the same span
// for each coincident ptT in loop; see if its opposite in is also in the loop
// this implementation is the motivation for marking that a ptT is referenced by a coincident span
    const SkOpPtT* head = this->ptT();
    const SkOpPtT* test = head;
    do {
        if (!test->coincident()) {
            continue;
        }
        coins->debugMarkCollapsed(log, test);
    } while ((test = test->next()) != head);
}
#endif

bool SkOpSpanBase::debugCoinEndLoopCheck() const {
    int loop = 0;
    const SkOpSpanBase* next = this;
    SkOpSpanBase* nextCoin;
    do {
        nextCoin = next->fCoinEnd;
        SkASSERT(nextCoin == this || nextCoin->fCoinEnd != nextCoin);
        for (int check = 1; check < loop - 1; ++check) {
            const SkOpSpanBase* checkCoin = this->fCoinEnd;
            const SkOpSpanBase* innerCoin = checkCoin;
            for (int inner = check + 1; inner < loop; ++inner) {
                innerCoin = innerCoin->fCoinEnd;
                if (checkCoin == innerCoin) {
                    SkDebugf("*** bad coincident end loop ***\n");
                    return false;
                }
            }
        }
        ++loop;
    } while ((next = nextCoin) && next != this);
    return true;
}

#if DEBUG_COIN
// Commented-out lines keep this in sync with insertCoinEnd()
void SkOpSpanBase::debugInsertCoinEnd(SkPathOpsDebug::GlitchLog* log, const SkOpSpanBase* coin) const {
    if (containsCoinEnd(coin)) {
//         SkASSERT(coin->containsCoinEnd(this));
        return;
    }
    debugValidate();
//     SkASSERT(this != coin);
    log->record(SkPathOpsDebug::kMarkCoinEnd_Glitch, this, coin);
//     coin->fCoinEnd = this->fCoinEnd;
//     this->fCoinEnd = coinNext;
    debugValidate();
}

// Commented-out lines keep this in sync with mergeMatches()
// Look to see if pt-t linked list contains same segment more than once
// if so, and if each pt-t is directly pointed to by spans in that segment,
// merge them
// keep the points, but remove spans so that the segment doesn't have 2 or more
// spans pointing to the same pt-t loop at different loop elements
void SkOpSpanBase::debugMergeMatches(SkPathOpsDebug::GlitchLog* log, const SkOpSpanBase* opp) const {
    const SkOpPtT* test = &fPtT;
    const SkOpPtT* testNext;
    const SkOpPtT* stop = test;
    do {
        testNext = test->next();
        if (test->deleted()) {
            continue;
        }
        const SkOpSpanBase* testBase = test->span();
        SkASSERT(testBase->ptT() == test);
        const SkOpSegment* segment = test->segment();
        if (segment->done()) {
            continue;
        }
        const SkOpPtT* inner = opp->ptT();
        const SkOpPtT* innerStop = inner;
        do {
            if (inner->segment() != segment) {
                continue;
            }
            if (inner->deleted()) {
                continue;
            }
            const SkOpSpanBase* innerBase = inner->span();
            SkASSERT(innerBase->ptT() == inner);
            // when the intersection is first detected, the span base is marked if there are
            // more than one point in the intersection.
//            if (!innerBase->hasMultipleHint() && !testBase->hasMultipleHint()) {
                if (!zero_or_one(inner->fT)) {
                    log->record(SkPathOpsDebug::kMergeMatches_Glitch, innerBase, test);
                } else {
                    SkASSERT(inner->fT != test->fT);
                    if (!zero_or_one(test->fT)) {
                        log->record(SkPathOpsDebug::kMergeMatches_Glitch, testBase, inner);
                    } else {
                        log->record(SkPathOpsDebug::kMergeMatches_Glitch, segment);
//                        SkDEBUGCODE(testBase->debugSetDeleted());
//                        test->setDeleted();
//                        SkDEBUGCODE(innerBase->debugSetDeleted());
//                        inner->setDeleted();
                    }
                }
#ifdef SK_DEBUG   // assert if another undeleted entry points to segment
                const SkOpPtT* debugInner = inner;
                while ((debugInner = debugInner->next()) != innerStop) {
                    if (debugInner->segment() != segment) {
                        continue;
                    }
                    if (debugInner->deleted()) {
                        continue;
                    }
                    SkOPASSERT(0);
                }
#endif
                break;
//            }
            break;
        } while ((inner = inner->next()) != innerStop);
    } while ((test = testNext) != stop);
    this->debugCheckForCollapsedCoincidence(log);
}

#endif

void SkOpSpanBase::debugResetCoinT() const {
#if DEBUG_COINCIDENCE_ORDER
    const SkOpPtT* ptT = &fPtT;
    do {
        ptT->debugResetCoinT();
        ptT = ptT->next();
    } while (ptT != &fPtT);
#endif
}

void SkOpSpanBase::debugSetCoinT(int index) const {
#if DEBUG_COINCIDENCE_ORDER
    const SkOpPtT* ptT = &fPtT;
    do {
        if (!ptT->deleted()) {
            ptT->debugSetCoinT(index);
        }
        ptT = ptT->next();
    } while (ptT != &fPtT);
#endif
}

const SkOpSpan* SkOpSpanBase::debugStarter(SkOpSpanBase const** endPtr) const {
    const SkOpSpanBase* end = *endPtr;
    SkASSERT(this->segment() == end->segment());
    const SkOpSpanBase* result;
    if (t() < end->t()) {
        result = this;
    } else {
        result = end;
        *endPtr = this;
    }
    return result->upCast();
}

void SkOpSpanBase::debugValidate() const {
#if DEBUG_COINCIDENCE
    if (this->globalState()->debugCheckHealth()) {
        return;
    }
#endif
#if DEBUG_VALIDATE
    const SkOpPtT* ptT = &fPtT;
    SkASSERT(ptT->span() == this);
    do {
//        SkASSERT(SkDPoint::RoughlyEqual(fPtT.fPt, ptT->fPt));
        ptT->debugValidate();
        ptT = ptT->next();
    } while (ptT != &fPtT);
    SkASSERT(this->debugCoinEndLoopCheck());
    if (!this->final()) {
        SkASSERT(this->upCast()->debugCoinLoopCheck());
    }
    if (fFromAngle) {
        fFromAngle->debugValidate();
    }
    if (!this->final() && this->upCast()->toAngle()) {
        this->upCast()->toAngle()->debugValidate();
    }
#endif
}

bool SkOpSpan::debugCoinLoopCheck() const {
    int loop = 0;
    const SkOpSpan* next = this;
    SkOpSpan* nextCoin;
    do {
        nextCoin = next->fCoincident;
        SkASSERT(nextCoin == this || nextCoin->fCoincident != nextCoin);
        for (int check = 1; check < loop - 1; ++check) {
            const SkOpSpan* checkCoin = this->fCoincident;
            const SkOpSpan* innerCoin = checkCoin;
            for (int inner = check + 1; inner < loop; ++inner) {
                innerCoin = innerCoin->fCoincident;
                if (checkCoin == innerCoin) {
                    SkDebugf("*** bad coincident loop ***\n");
                    return false;
                }
            }
        }
        ++loop;
    } while ((next = nextCoin) && next != this);
    return true;
}

#if DEBUG_COIN
// Commented-out lines keep this in sync with insertCoincidence() in header
void SkOpSpan::debugInsertCoincidence(SkPathOpsDebug::GlitchLog* log, const SkOpSpan* coin) const {
    if (containsCoincidence(coin)) {
//         SkASSERT(coin->containsCoincidence(this));
        return;
    }
    debugValidate();
//     SkASSERT(this != coin);
    log->record(SkPathOpsDebug::kMarkCoinStart_Glitch, this, coin);
//     coin->fCoincident = this->fCoincident;
//     this->fCoincident = coinNext;
    debugValidate();
}

// Commented-out lines keep this in sync with insertCoincidence()
void SkOpSpan::debugInsertCoincidence(SkPathOpsDebug::GlitchLog* log, const SkOpSegment* segment, bool flipped, bool ordered) const {
    if (this->containsCoincidence(segment)) {
        return;
    }
    const SkOpPtT* next = &fPtT;
    while ((next = next->next()) != &fPtT) {
        if (next->segment() == segment) {
            const SkOpSpan* span;
            const SkOpSpanBase* base = next->span();
            if (!ordered) {
                const SkOpSpanBase* spanEnd = fNext->contains(segment)->span();
                const SkOpPtT* start = base->ptT()->starter(spanEnd->ptT());
                FAIL_IF(!start->span()->upCastable(), this);
                span = const_cast<SkOpSpan*>(start->span()->upCast());
            }
            else if (flipped) {
                span = base->prev();
                FAIL_IF(!span, this);
            }
            else {
                FAIL_IF(!base->upCastable(), this);
                span = base->upCast();
            }
            log->record(SkPathOpsDebug::kMarkCoinInsert_Glitch, span);
            return;
        }
    }
#if DEBUG_COIN
    log->record(SkPathOpsDebug::kMarkCoinMissing_Glitch, segment, this);
#endif
    return;
}
#endif

// called only by test code
int SkIntersections::debugCoincidentUsed() const {
    if (!fIsCoincident[0]) {
        SkASSERT(!fIsCoincident[1]);
        return 0;
    }
    int count = 0;
    SkDEBUGCODE(int count2 = 0;)
    for (int index = 0; index < fUsed; ++index) {
        if (fIsCoincident[0] & (1 << index)) {
            ++count;
        }
#ifdef SK_DEBUG
        if (fIsCoincident[1] & (1 << index)) {
            ++count2;
        }
#endif
    }
    SkASSERT(count == count2);
    return count;
}

#include "src/pathops/SkOpContour.h"

// Commented-out lines keep this in sync with addOpp()
void SkOpPtT::debugAddOpp(const SkOpPtT* opp, const SkOpPtT* oppPrev) const {
    SkDEBUGCODE(const SkOpPtT* oldNext = this->fNext);
    SkASSERT(this != opp);
//    this->fNext = opp;
    SkASSERT(oppPrev != oldNext);
//    oppPrev->fNext = oldNext;
}

bool SkOpPtT::debugContains(const SkOpPtT* check) const {
    SkASSERT(this != check);
    const SkOpPtT* ptT = this;
    int links = 0;
    do {
        ptT = ptT->next();
        if (ptT == check) {
            return true;
        }
        ++links;
        const SkOpPtT* test = this;
        for (int index = 0; index < links; ++index) {
            if (ptT == test) {
                return false;
            }
            test = test->next();
        }
    } while (true);
}

const SkOpPtT* SkOpPtT::debugContains(const SkOpSegment* check) const {
    SkASSERT(this->segment() != check);
    const SkOpPtT* ptT = this;
    int links = 0;
    do {
        ptT = ptT->next();
        if (ptT->segment() == check) {
            return ptT;
        }
        ++links;
        const SkOpPtT* test = this;
        for (int index = 0; index < links; ++index) {
            if (ptT == test) {
                return nullptr;
            }
            test = test->next();
        }
    } while (true);
}

const SkOpPtT* SkOpPtT::debugEnder(const SkOpPtT* end) const {
    return fT < end->fT ? end : this;
}

int SkOpPtT::debugLoopLimit(bool report) const {
    int loop = 0;
    const SkOpPtT* next = this;
    do {
        for (int check = 1; check < loop - 1; ++check) {
            const SkOpPtT* checkPtT = this->fNext;
            const SkOpPtT* innerPtT = checkPtT;
            for (int inner = check + 1; inner < loop; ++inner) {
                innerPtT = innerPtT->fNext;
                if (checkPtT == innerPtT) {
                    if (report) {
                        SkDebugf("*** bad ptT loop ***\n");
                    }
                    return loop;
                }
            }
        }
        // there's nothing wrong with extremely large loop counts -- but this may appear to hang
        // by taking a very long time to figure out that no loop entry is a duplicate
        // -- and it's likely that a large loop count is indicative of a bug somewhere
        if (++loop > 1000) {
            SkDebugf("*** loop count exceeds 1000 ***\n");
            return 1000;
        }
    } while ((next = next->fNext) && next != this);
    return 0;
}

const SkOpPtT* SkOpPtT::debugOppPrev(const SkOpPtT* opp) const {
    return this->oppPrev(const_cast<SkOpPtT*>(opp));
}

void SkOpPtT::debugResetCoinT() const {
#if DEBUG_COINCIDENCE_ORDER
    this->segment()->debugResetCoinT();
#endif
}

void SkOpPtT::debugSetCoinT(int index) const {
#if DEBUG_COINCIDENCE_ORDER
    this->segment()->debugSetCoinT(index, fT);
#endif
}

void SkOpPtT::debugValidate() const {
#if DEBUG_COINCIDENCE
    if (this->globalState()->debugCheckHealth()) {
        return;
    }
#endif
#if DEBUG_VALIDATE
    SkOpPhase phase = contour()->globalState()->phase();
    if (phase == SkOpPhase::kIntersecting || phase == SkOpPhase::kFixWinding) {
        return;
    }
    SkASSERT(fNext);
    SkASSERT(fNext != this);
    SkASSERT(fNext->fNext);
    SkASSERT(debugLoopLimit(false) == 0);
#endif
}

// -- GODOT start --
/*
static void output_scalar(SkScalar num) {
    if (num == (int) num) {
        SkDebugf("%d", (int) num);
    } else {
        SkString str;
        str.printf("%1.9g", num);
        int width = (int) str.size();
        const char* cStr = str.c_str();
        while (cStr[width - 1] == '0') {
            --width;
        }
        str.resize(width);
        SkDebugf("%sf", str.c_str());
    }
}

static void output_points(const SkPoint* pts, int count) {
    for (int index = 0; index < count; ++index) {
        output_scalar(pts[index].fX);
        SkDebugf(", ");
        output_scalar(pts[index].fY);
        if (index + 1 < count) {
            SkDebugf(", ");
        }
    }
}

static void showPathContours(const SkPath& path, const char* pathName) {
    for (auto [verb, pts, w] : SkPathPriv::Iterate(path)) {
        switch (verb) {
            case SkPathVerb::kMove:
                SkDebugf("    %s.moveTo(", pathName);
                output_points(&pts[0], 1);
                SkDebugf(");\n");
                continue;
            case SkPathVerb::kLine:
                SkDebugf("    %s.lineTo(", pathName);
                output_points(&pts[1], 1);
                SkDebugf(");\n");
                break;
            case SkPathVerb::kQuad:
                SkDebugf("    %s.quadTo(", pathName);
                output_points(&pts[1], 2);
                SkDebugf(");\n");
                break;
            case SkPathVerb::kConic:
                SkDebugf("    %s.conicTo(", pathName);
                output_points(&pts[1], 2);
                SkDebugf(", %1.9gf);\n", *w);
                break;
            case SkPathVerb::kCubic:
                SkDebugf("    %s.cubicTo(", pathName);
                output_points(&pts[1], 3);
                SkDebugf(");\n");
                break;
            case SkPathVerb::kClose:
                SkDebugf("    %s.close();\n", pathName);
                break;
            default:
                SkDEBUGFAIL("bad verb");
                return;
        }
    }
}

static const char* gFillTypeStr[] = {
    "kWinding",
    "kEvenOdd",
    "kInverseWinding",
    "kInverseEvenOdd"
};

void SkPathOpsDebug::ShowOnePath(const SkPath& path, const char* name, bool includeDeclaration) {
#define SUPPORT_RECT_CONTOUR_DETECTION 0
#if SUPPORT_RECT_CONTOUR_DETECTION
    int rectCount = path.isRectContours() ? path.rectContours(nullptr, nullptr) : 0;
    if (rectCount > 0) {
        SkTDArray<SkRect> rects;
        SkTDArray<SkPathDirection> directions;
        rects.setCount(rectCount);
        directions.setCount(rectCount);
        path.rectContours(rects.begin(), directions.begin());
        for (int contour = 0; contour < rectCount; ++contour) {
            const SkRect& rect = rects[contour];
            SkDebugf("path.addRect(%1.9g, %1.9g, %1.9g, %1.9g, %s);\n", rect.fLeft, rect.fTop,
                    rect.fRight, rect.fBottom, directions[contour] == SkPathDirection::kCCW
                    ? "SkPathDirection::kCCW" : "SkPathDirection::kCW");
        }
        return;
    }
#endif
    SkPathFillType fillType = path.getFillType();
    SkASSERT(fillType >= SkPathFillType::kWinding && fillType <= SkPathFillType::kInverseEvenOdd);
    if (includeDeclaration) {
        SkDebugf("    SkPath %s;\n", name);
    }
    SkDebugf("    %s.setFillType(SkPath::%s);\n", name, gFillTypeStr[(int)fillType]);
    showPathContours(path, name);
}
*/
// -- GODOT end --

#if DEBUG_DUMP_VERIFY
#include "include/core/SkData.h"
#include "include/core/SkStream.h"

static void dump_path(FILE* file, const SkPath& path, bool force, bool dumpAsHex) {
    SkDynamicMemoryWStream wStream;
    path.dump(&wStream, force, dumpAsHex);
    sk_sp<SkData> data(wStream.detachAsData());
    fprintf(file, "%.*s\n", (int) data->size(), (char*) data->data());
}

static int dumpID = 0;

void SkPathOpsDebug::DumpOp(const SkPath& one, const SkPath& two, SkPathOp op,
        const char* testName) {
    FILE* file = sk_fopen("op_dump.txt", kWrite_SkFILE_Flag);
    DumpOp(file, one, two, op, testName);
}

void SkPathOpsDebug::DumpOp(FILE* file, const SkPath& one, const SkPath& two, SkPathOp op,
        const char* testName) {
    const char* name = testName ? testName : "op";
    fprintf(file,
            "\nstatic void %s_%d(skiatest::Reporter* reporter, const char* filename) {\n",
            name, ++dumpID);
    fprintf(file, "    SkPath path;\n");
    fprintf(file, "    path.setFillType((SkPath::FillType) %d);\n", one.getFillType());
    dump_path(file, one, false, true);
    fprintf(file, "    SkPath path1(path);\n");
    fprintf(file, "    path.reset();\n");
    fprintf(file, "    path.setFillType((SkPath::FillType) %d);\n", two.getFillType());
    dump_path(file, two, false, true);
    fprintf(file, "    SkPath path2(path);\n");
    fprintf(file, "    testPathOp(reporter, path1, path2, (SkPathOp) %d, filename);\n", op);
    fprintf(file, "}\n\n");
    fclose(file);
}

void SkPathOpsDebug::DumpSimplify(const SkPath& path, const char* testName) {
    FILE* file = sk_fopen("simplify_dump.txt", kWrite_SkFILE_Flag);
    DumpSimplify(file, path, testName);
}

void SkPathOpsDebug::DumpSimplify(FILE* file, const SkPath& path, const char* testName) {
    const char* name = testName ? testName : "simplify";
    fprintf(file,
            "\nstatic void %s_%d(skiatest::Reporter* reporter, const char* filename) {\n",
            name, ++dumpID);
    fprintf(file, "    SkPath path;\n");
    fprintf(file, "    path.setFillType((SkPath::FillType) %d);\n", path.getFillType());
    dump_path(file, path, false, true);
    fprintf(file, "    testSimplify(reporter, path, filename);\n");
    fprintf(file, "}\n\n");
    fclose(file);
}

#include "include/core/SkBitmap.h"
#include "include/core/SkCanvas.h"
#include "include/core/SkPaint.h"

const int bitWidth = 64;
const int bitHeight = 64;

static void debug_scale_matrix(const SkPath& one, const SkPath* two, SkMatrix& scale) {
    SkRect larger = one.getBounds();
    if (two) {
        larger.join(two->getBounds());
    }
    SkScalar largerWidth = larger.width();
    if (largerWidth < 4) {
        largerWidth = 4;
    }
    SkScalar largerHeight = larger.height();
    if (largerHeight < 4) {
        largerHeight = 4;
    }
    SkScalar hScale = (bitWidth - 2) / largerWidth;
    SkScalar vScale = (bitHeight - 2) / largerHeight;
    scale.reset();
    scale.preScale(hScale, vScale);
    larger.fLeft *= hScale;
    larger.fRight *= hScale;
    larger.fTop *= vScale;
    larger.fBottom *= vScale;
    SkScalar dx = -16000 > larger.fLeft ? -16000 - larger.fLeft
            : 16000 < larger.fRight ? 16000 - larger.fRight : 0;
    SkScalar dy = -16000 > larger.fTop ? -16000 - larger.fTop
            : 16000 < larger.fBottom ? 16000 - larger.fBottom : 0;
    scale.preTranslate(dx, dy);
}

static int debug_paths_draw_the_same(const SkPath& one, const SkPath& two, SkBitmap& bits) {
    if (bits.width() == 0) {
        bits.allocN32Pixels(bitWidth * 2, bitHeight);
    }
    SkCanvas canvas(bits);
    canvas.drawColor(SK_ColorWHITE);
    SkPaint paint;
    canvas.save();
    const SkRect& bounds1 = one.getBounds();
    canvas.translate(-bounds1.fLeft + 1, -bounds1.fTop + 1);
    canvas.drawPath(one, paint);
    canvas.restore();
    canvas.save();
    canvas.translate(-bounds1.fLeft + 1 + bitWidth, -bounds1.fTop + 1);
    canvas.drawPath(two, paint);
    canvas.restore();
    int errors = 0;
    for (int y = 0; y < bitHeight - 1; ++y) {
        uint32_t* addr1 = bits.getAddr32(0, y);
        uint32_t* addr2 = bits.getAddr32(0, y + 1);
        uint32_t* addr3 = bits.getAddr32(bitWidth, y);
        uint32_t* addr4 = bits.getAddr32(bitWidth, y + 1);
        for (int x = 0; x < bitWidth - 1; ++x) {
            // count 2x2 blocks
            bool err = addr1[x] != addr3[x];
            if (err) {
                errors += addr1[x + 1] != addr3[x + 1]
                        && addr2[x] != addr4[x] && addr2[x + 1] != addr4[x + 1];
            }
        }
    }
    return errors;
}

void SkPathOpsDebug::ReportOpFail(const SkPath& one, const SkPath& two, SkPathOp op) {
    SkDebugf("// Op did not expect failure\n");
    DumpOp(stderr, one, two, op, "opTest");
    fflush(stderr);
}

void SkPathOpsDebug::VerifyOp(const SkPath& one, const SkPath& two, SkPathOp op,
        const SkPath& result) {
    SkPath pathOut, scaledPathOut;
    SkRegion rgnA, rgnB, openClip, rgnOut;
    openClip.setRect(-16000, -16000, 16000, 16000);
    rgnA.setPath(one, openClip);
    rgnB.setPath(two, openClip);
    rgnOut.op(rgnA, rgnB, (SkRegion::Op) op);
    rgnOut.getBoundaryPath(&pathOut);
    SkMatrix scale;
    debug_scale_matrix(one, &two, scale);
    SkRegion scaledRgnA, scaledRgnB, scaledRgnOut;
    SkPath scaledA, scaledB;
    scaledA.addPath(one, scale);
    scaledA.setFillType(one.getFillType());
    scaledB.addPath(two, scale);
    scaledB.setFillType(two.getFillType());
    scaledRgnA.setPath(scaledA, openClip);
    scaledRgnB.setPath(scaledB, openClip);
    scaledRgnOut.op(scaledRgnA, scaledRgnB, (SkRegion::Op) op);
    scaledRgnOut.getBoundaryPath(&scaledPathOut);
    SkBitmap bitmap;
    SkPath scaledOut;
    scaledOut.addPath(result, scale);
    scaledOut.setFillType(result.getFillType());
    int errors = debug_paths_draw_the_same(scaledPathOut, scaledOut, bitmap);
    const int MAX_ERRORS = 9;
    if (errors > MAX_ERRORS) {
        fprintf(stderr, "// Op did not expect errors=%d\n", errors);
        DumpOp(stderr, one, two, op, "opTest");
        fflush(stderr);
    }
}

void SkPathOpsDebug::ReportSimplifyFail(const SkPath& path) {
    SkDebugf("// Simplify did not expect failure\n");
    DumpSimplify(stderr, path, "simplifyTest");
    fflush(stderr);
}

void SkPathOpsDebug::VerifySimplify(const SkPath& path, const SkPath& result) {
    SkPath pathOut, scaledPathOut;
    SkRegion rgnA, openClip, rgnOut;
    openClip.setRect(-16000, -16000, 16000, 16000);
    rgnA.setPath(path, openClip);
    rgnOut.getBoundaryPath(&pathOut);
    SkMatrix scale;
    debug_scale_matrix(path, nullptr, scale);
    SkRegion scaledRgnA;
    SkPath scaledA;
    scaledA.addPath(path, scale);
    scaledA.setFillType(path.getFillType());
    scaledRgnA.setPath(scaledA, openClip);
    scaledRgnA.getBoundaryPath(&scaledPathOut);
    SkBitmap bitmap;
    SkPath scaledOut;
    scaledOut.addPath(result, scale);
    scaledOut.setFillType(result.getFillType());
    int errors = debug_paths_draw_the_same(scaledPathOut, scaledOut, bitmap);
    const int MAX_ERRORS = 9;
    if (errors > MAX_ERRORS) {
        fprintf(stderr, "// Simplify did not expect errors=%d\n", errors);
        DumpSimplify(stderr, path, "simplifyTest");
        fflush(stderr);
    }
}

#endif

// global path dumps for msvs Visual Studio 17 to use from Immediate Window
// -- GODOT start --
/* 
void Dump(const SkPath& path) {
    path.dump();
}

void DumpHex(const SkPath& path) {
    path.dumpHex();
}
*/
// -- GODOT end --
