/*
 * Copyright 2013 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#ifndef SkPathOpsDebug_DEFINED
#define SkPathOpsDebug_DEFINED

// -- GODOT start --
//#include "include/core/SkString.h"
// -- GODOT end --
#include "include/core/SkTypes.h"
#include "include/pathops/SkPathOps.h"

#include <stdlib.h>
#include <stdio.h>

enum class SkOpPhase : char;
struct SkDQuad;
class SkOpAngle;
class SkOpCoincidence;
class SkOpContour;
class SkOpContourHead;
class SkOpPtT;
class SkOpSegment;
class SkOpSpan;
class SkOpSpanBase;
struct SkDPoint;
struct SkDLine;
struct SkDQuad;
struct SkDConic;
struct SkDCubic;
class SkTSect;

// define this when running fuzz
// #define SK_BUILD_FOR_FUZZER

// fake classes to fool msvs Visual Studio 2018 Immediate Window
#define FakeClasses(a, b) \
class SkDebugTCoincident##a##b; \
class SkDebugTSect##a##b; \
class SkDebugTSpan##a##b

FakeClasses(Quad, Quad);
FakeClasses(Conic, Quad);
FakeClasses(Conic, Conic);
FakeClasses(Cubic, Quad);
FakeClasses(Cubic, Conic);
FakeClasses(Cubic, Cubic);

#undef FakeClasses

#ifdef SK_RELEASE
#define FORCE_RELEASE 1
#else
#define FORCE_RELEASE 1  // set force release to 1 for multiple thread -- no debugging
#endif

#define DEBUG_UNDER_DEVELOPMENT 0

#define ONE_OFF_DEBUG 0
#define ONE_OFF_DEBUG_MATHEMATICA 0

#if defined(SK_BUILD_FOR_WIN) || defined(SK_BUILD_FOR_ANDROID)
    #define SK_RAND(seed) rand()
#else
    #define SK_RAND(seed) rand_r(&seed)
#endif
#ifdef SK_BUILD_FOR_WIN
    #define SK_SNPRINTF _snprintf
#else
    #define SK_SNPRINTF snprintf
#endif

#define WIND_AS_STRING(x) char x##Str[12]; \
        if (!SkPathOpsDebug::ValidWind(x)) strcpy(x##Str, "?"); \
        else SK_SNPRINTF(x##Str, sizeof(x##Str), "%d", x)

#if FORCE_RELEASE

#define DEBUG_ACTIVE_OP 0
#define DEBUG_ACTIVE_SPANS 0
#define DEBUG_ADD_INTERSECTING_TS 0
#define DEBUG_ADD_T 0
#define DEBUG_ALIGNMENT 0
#define DEBUG_ANGLE 0
#define DEBUG_ASSEMBLE 0
#define DEBUG_COINCIDENCE 0
#define DEBUG_COINCIDENCE_DUMP 0  // accumulate and dump which algorithms fired
#define DEBUG_COINCIDENCE_ORDER 0  // for well behaved curves, check if pairs match up in t-order
#define DEBUG_COINCIDENCE_VERBOSE 0  // usually whether the next function generates coincidence
#define DEBUG_CUBIC_BINARY_SEARCH 0
#define DEBUG_CUBIC_SPLIT 0
#define DEBUG_DUMP_SEGMENTS 0
#define DEBUG_DUMP_VERIFY 0
#define DEBUG_FLOW 0
#define DEBUG_LIMIT_WIND_SUM 0
#define DEBUG_MARK_DONE 0
#define DEBUG_PATH_CONSTRUCTION 0
#define DEBUG_PERP 0
#define DEBUG_SORT 0
#define DEBUG_T_SECT 0
#define DEBUG_T_SECT_DUMP 0
#define DEBUG_T_SECT_LOOP_COUNT 0
#define DEBUG_VALIDATE 0
#define DEBUG_WINDING 0
#define DEBUG_WINDING_AT_T 0

#else

#define DEBUG_ACTIVE_OP 1
#define DEBUG_ACTIVE_SPANS 1
#define DEBUG_ADD_INTERSECTING_TS 1
#define DEBUG_ADD_T 1
#define DEBUG_ALIGNMENT 0
#define DEBUG_ANGLE 1
#define DEBUG_ASSEMBLE 1
#define DEBUG_COINCIDENCE 1
#define DEBUG_COINCIDENCE_DUMP 0
#define DEBUG_COINCIDENCE_ORDER 0  // tight arc quads may generate out-of-order coincidence spans
#define DEBUG_COINCIDENCE_VERBOSE 1
#define DEBUG_CUBIC_BINARY_SEARCH 0
#define DEBUG_CUBIC_SPLIT 1
#define DEBUG_DUMP_VERIFY 0
#define DEBUG_DUMP_SEGMENTS 1
#define DEBUG_FLOW 1
#define DEBUG_LIMIT_WIND_SUM 15
#define DEBUG_MARK_DONE 1
#define DEBUG_PATH_CONSTRUCTION 1
#define DEBUG_PERP 1
#define DEBUG_SORT 1
#define DEBUG_T_SECT 0        // enabling may trigger validate asserts even though op does not fail
#define DEBUG_T_SECT_DUMP 0  // Use 1 normally. Use 2 to number segments, 3 for script output
#define DEBUG_T_SECT_LOOP_COUNT 0
#define DEBUG_VALIDATE 1
#define DEBUG_WINDING 1
#define DEBUG_WINDING_AT_T 1

#endif

#ifdef SK_RELEASE
    #define SkDEBUGRELEASE(a, b) b
    #define SkDEBUGPARAMS(...)
#else
    #define SkDEBUGRELEASE(a, b) a
    #define SkDEBUGPARAMS(...) , __VA_ARGS__
#endif

#if DEBUG_VALIDATE == 0
    #define PATH_OPS_DEBUG_VALIDATE_PARAMS(...)
#else
    #define PATH_OPS_DEBUG_VALIDATE_PARAMS(...) , __VA_ARGS__
#endif

#if DEBUG_T_SECT == 0
    #define PATH_OPS_DEBUG_T_SECT_RELEASE(a, b) b
    #define PATH_OPS_DEBUG_T_SECT_PARAMS(...)
    #define PATH_OPS_DEBUG_T_SECT_CODE(...)
#else
    #define PATH_OPS_DEBUG_T_SECT_RELEASE(a, b) a
    #define PATH_OPS_DEBUG_T_SECT_PARAMS(...) , __VA_ARGS__
    #define PATH_OPS_DEBUG_T_SECT_CODE(...) __VA_ARGS__
#endif

#if DEBUG_T_SECT_DUMP > 1
    extern int gDumpTSectNum;
#endif

#if DEBUG_COINCIDENCE || DEBUG_COINCIDENCE_DUMP
    #define DEBUG_COIN 1
#else
    #define DEBUG_COIN 0
#endif

#if DEBUG_COIN
    #define DEBUG_COIN_DECLARE_ONLY_PARAMS() \
            int lineNo, SkOpPhase phase, int iteration
    #define DEBUG_COIN_DECLARE_PARAMS() \
            , DEBUG_COIN_DECLARE_ONLY_PARAMS()
    #define DEBUG_COIN_ONLY_PARAMS() \
            __LINE__, SkOpPhase::kNoChange, 0
    #define DEBUG_COIN_PARAMS() \
            , DEBUG_COIN_ONLY_PARAMS()
    #define DEBUG_ITER_ONLY_PARAMS(iteration) \
            __LINE__, SkOpPhase::kNoChange, iteration
    #define DEBUG_ITER_PARAMS(iteration) \
            , DEBUG_ITER_ONLY_PARAMS(iteration)
    #define DEBUG_PHASE_ONLY_PARAMS(phase) \
            __LINE__, SkOpPhase::phase, 0
    #define DEBUG_PHASE_PARAMS(phase) \
            , DEBUG_PHASE_ONLY_PARAMS(phase)
    #define DEBUG_SET_PHASE() \
            this->globalState()->debugSetPhase(__func__, lineNo, phase, iteration)
    #define DEBUG_STATIC_SET_PHASE(obj) \
            obj->globalState()->debugSetPhase(__func__, lineNo, phase, iteration)
#elif DEBUG_VALIDATE
    #define DEBUG_COIN_DECLARE_ONLY_PARAMS() \
            SkOpPhase phase
    #define DEBUG_COIN_DECLARE_PARAMS() \
            , DEBUG_COIN_DECLARE_ONLY_PARAMS()
    #define DEBUG_COIN_ONLY_PARAMS() \
            SkOpPhase::kNoChange
    #define DEBUG_COIN_PARAMS() \
            , DEBUG_COIN_ONLY_PARAMS()
    #define DEBUG_ITER_ONLY_PARAMS(iteration) \
            SkOpPhase::kNoChange
    #define DEBUG_ITER_PARAMS(iteration) \
            , DEBUG_ITER_ONLY_PARAMS(iteration)
    #define DEBUG_PHASE_ONLY_PARAMS(phase) \
            SkOpPhase::phase
    #define DEBUG_PHASE_PARAMS(phase) \
            , DEBUG_PHASE_ONLY_PARAMS(phase)
    #define DEBUG_SET_PHASE() \
            this->globalState()->debugSetPhase(phase)
    #define DEBUG_STATIC_SET_PHASE(obj) \
            obj->globalState()->debugSetPhase(phase)
#else
    #define DEBUG_COIN_DECLARE_ONLY_PARAMS()
    #define DEBUG_COIN_DECLARE_PARAMS()
    #define DEBUG_COIN_ONLY_PARAMS()
    #define DEBUG_COIN_PARAMS()
    #define DEBUG_ITER_ONLY_PARAMS(iteration)
    #define DEBUG_ITER_PARAMS(iteration)
    #define DEBUG_PHASE_ONLY_PARAMS(phase)
    #define DEBUG_PHASE_PARAMS(phase)
    #define DEBUG_SET_PHASE()
    #define DEBUG_STATIC_SET_PHASE(obj)
#endif

#define CUBIC_DEBUG_STR  "{{{%1.9g,%1.9g}, {%1.9g,%1.9g}, {%1.9g,%1.9g}, {%1.9g,%1.9g}}}"
#define CONIC_DEBUG_STR "{{{{%1.9g,%1.9g}, {%1.9g,%1.9g}, {%1.9g,%1.9g}}}, %1.9g}"
#define QUAD_DEBUG_STR   "{{{%1.9g,%1.9g}, {%1.9g,%1.9g}, {%1.9g,%1.9g}}}"
#define LINE_DEBUG_STR   "{{{%1.9g,%1.9g}, {%1.9g,%1.9g}}}"
#define PT_DEBUG_STR "{{%1.9g,%1.9g}}"

#define T_DEBUG_STR(t, n) #t "[" #n "]=%1.9g"
#define TX_DEBUG_STR(t) #t "[%d]=%1.9g"
#define CUBIC_DEBUG_DATA(c) c[0].fX, c[0].fY, c[1].fX, c[1].fY, c[2].fX, c[2].fY, c[3].fX, c[3].fY
#define CONIC_DEBUG_DATA(c, w) c[0].fX, c[0].fY, c[1].fX, c[1].fY, c[2].fX, c[2].fY, w
#define QUAD_DEBUG_DATA(q)  q[0].fX, q[0].fY, q[1].fX, q[1].fY, q[2].fX, q[2].fY
#define LINE_DEBUG_DATA(l)  l[0].fX, l[0].fY, l[1].fX, l[1].fY
#define PT_DEBUG_DATA(i, n) i.pt(n).asSkPoint().fX, i.pt(n).asSkPoint().fY

#ifndef DEBUG_TEST
#define DEBUG_TEST 0
#endif

// Tests with extreme numbers may fail, but all other tests should never fail.
#define FAIL_IF(cond) \
        do { bool fail = (cond); SkOPASSERT(!fail); if (fail) return false; } while (false)

#define FAIL_WITH_NULL_IF(cond) \
        do { bool fail = (cond); SkOPASSERT(!fail); if (fail) return nullptr; } while (false)

class SkPathOpsDebug {
public:
#if DEBUG_COIN
    struct GlitchLog;

    enum GlitchType {
        kUninitialized_Glitch,
        kAddCorruptCoin_Glitch,
        kAddExpandedCoin_Glitch,
        kAddExpandedFail_Glitch,
        kAddIfCollapsed_Glitch,
        kAddIfMissingCoin_Glitch,
        kAddMissingCoin_Glitch,
        kAddMissingExtend_Glitch,
        kAddOrOverlap_Glitch,
        kCollapsedCoin_Glitch,
        kCollapsedDone_Glitch,
        kCollapsedOppValue_Glitch,
        kCollapsedSpan_Glitch,
        kCollapsedWindValue_Glitch,
        kCorrectEnd_Glitch,
        kDeletedCoin_Glitch,
        kExpandCoin_Glitch,
        kFail_Glitch,
        kMarkCoinEnd_Glitch,
        kMarkCoinInsert_Glitch,
        kMarkCoinMissing_Glitch,
        kMarkCoinStart_Glitch,
        kMergeMatches_Glitch,
        kMissingCoin_Glitch,
        kMissingDone_Glitch,
        kMissingIntersection_Glitch,
        kMoveMultiple_Glitch,
        kMoveNearbyClearAll_Glitch,
        kMoveNearbyClearAll2_Glitch,
        kMoveNearbyMerge_Glitch,
        kMoveNearbyMergeFinal_Glitch,
        kMoveNearbyRelease_Glitch,
        kMoveNearbyReleaseFinal_Glitch,
        kReleasedSpan_Glitch,
        kReturnFalse_Glitch,
        kUnaligned_Glitch,
        kUnalignedHead_Glitch,
        kUnalignedTail_Glitch,
    };

    struct CoinDictEntry {
        int fIteration;
        int fLineNumber;
        GlitchType fGlitchType;
        const char* fFunctionName;
    };

    struct CoinDict {
        void add(const CoinDictEntry& key);
        void add(const CoinDict& dict);
        void dump(const char* str, bool visitCheck) const;
        SkTDArray<CoinDictEntry> fDict;
    };

    static CoinDict gCoinSumChangedDict;
    static CoinDict gCoinSumVisitedDict;
    static CoinDict gCoinVistedDict;
#endif

#if defined(SK_DEBUG) || !FORCE_RELEASE
    static int gContourID;
    static int gSegmentID;
#endif

#if DEBUG_SORT
    static int gSortCountDefault;
    static int gSortCount;
#endif

#if DEBUG_ACTIVE_OP
    static const char* kPathOpStr[];
#endif
    static bool gRunFail;
    static bool gVeryVerbose;

#if DEBUG_ACTIVE_SPANS
    static SkString gActiveSpans;
#endif
#if DEBUG_DUMP_VERIFY
    static bool gDumpOp;
    static bool gVerifyOp;
#endif

    static const char* OpStr(SkPathOp );
    static void MathematicaIze(char* str, size_t bufferSize);
    static bool ValidWind(int winding);
    static void WindingPrintf(int winding);

    static void ShowActiveSpans(SkOpContourHead* contourList);
// -- GODOT start --
    //static void ShowOnePath(const SkPath& path, const char* name, bool includeDeclaration);
    //static void ShowPath(const SkPath& one, const SkPath& two, SkPathOp op, const char* name);
// -- GODOT end --

    static bool ChaseContains(const SkTDArray<SkOpSpanBase*>& , const SkOpSpanBase* );

    static void CheckHealth(class SkOpContourHead* contourList);

#if DEBUG_COIN
   static void DumpCoinDict();
   static void DumpGlitchType(GlitchType );
#endif

};

// Visual Studio 2017 does not permit calling member functions from the Immediate Window.
// Global functions work fine, however. Use globals rather than static members inside a class.
const SkOpAngle* AngleAngle(const SkOpAngle*, int id);
SkOpContour* AngleContour(SkOpAngle*, int id);
const SkOpPtT* AnglePtT(const SkOpAngle*, int id);
const SkOpSegment* AngleSegment(const SkOpAngle*, int id);
const SkOpSpanBase* AngleSpan(const SkOpAngle*, int id);

const SkOpAngle* ContourAngle(SkOpContour*, int id);
SkOpContour* ContourContour(SkOpContour*, int id);
const SkOpPtT* ContourPtT(SkOpContour*, int id);
const SkOpSegment* ContourSegment(SkOpContour*, int id);
const SkOpSpanBase* ContourSpan(SkOpContour*, int id);

const SkOpAngle* CoincidenceAngle(SkOpCoincidence*, int id);
SkOpContour* CoincidenceContour(SkOpCoincidence*, int id);
const SkOpPtT* CoincidencePtT(SkOpCoincidence*, int id);
const SkOpSegment* CoincidenceSegment(SkOpCoincidence*, int id);
const SkOpSpanBase* CoincidenceSpan(SkOpCoincidence*, int id);

const SkOpAngle* PtTAngle(const SkOpPtT*, int id);
SkOpContour* PtTContour(SkOpPtT*, int id);
const SkOpPtT* PtTPtT(const SkOpPtT*, int id);
const SkOpSegment* PtTSegment(const SkOpPtT*, int id);
const SkOpSpanBase* PtTSpan(const SkOpPtT*, int id);

const SkOpAngle* SegmentAngle(const SkOpSegment*, int id);
SkOpContour* SegmentContour(SkOpSegment*, int id);
const SkOpPtT* SegmentPtT(const SkOpSegment*, int id);
const SkOpSegment* SegmentSegment(const SkOpSegment*, int id);
const SkOpSpanBase* SegmentSpan(const SkOpSegment*, int id);

const SkOpAngle* SpanAngle(const SkOpSpanBase*, int id);
SkOpContour* SpanContour(SkOpSpanBase*, int id);
const SkOpPtT* SpanPtT(const SkOpSpanBase*, int id);
const SkOpSegment* SpanSegment(const SkOpSpanBase*, int id);
const SkOpSpanBase* SpanSpan(const SkOpSpanBase*, int id);

#if DEBUG_DUMP_VERIFY
void DumpOp(const SkPath& one, const SkPath& two, SkPathOp op,
        const char* testName);
void DumpOp(FILE* file, const SkPath& one, const SkPath& two, SkPathOp op,
        const char* testName);
void DumpSimplify(const SkPath& path, const char* testName);
void DumpSimplify(FILE* file, const SkPath& path, const char* testName);
void ReportOpFail(const SkPath& one, const SkPath& two, SkPathOp op);
void ReportSimplifyFail(const SkPath& path);
void VerifyOp(const SkPath& one, const SkPath& two, SkPathOp op,
    const SkPath& result);
void VerifySimplify(const SkPath& path, const SkPath& result);
#endif

// global path dumps for msvs Visual Studio 17 to use from Immediate Window
void Dump(const SkOpContour& );
void DumpAll(const SkOpContour& );
void DumpAngles(const SkOpContour& );
void DumpContours(const SkOpContour& );
void DumpContoursAll(const SkOpContour& );
void DumpContoursAngles(const SkOpContour& );
void DumpContoursPts(const SkOpContour& );
void DumpContoursPt(const SkOpContour& , int segmentID);
void DumpContoursSegment(const SkOpContour& , int segmentID);
void DumpContoursSpan(const SkOpContour& , int segmentID);
void DumpContoursSpans(const SkOpContour& );
void DumpPt(const SkOpContour& , int );
void DumpPts(const SkOpContour& , const char* prefix = "seg");
void DumpSegment(const SkOpContour& , int );
void DumpSegments(const SkOpContour& , const char* prefix = "seg", SkPathOp op = (SkPathOp) -1);
void DumpSpan(const SkOpContour& , int );
void DumpSpans(const SkOpContour& );

void Dump(const SkOpSegment& );
void DumpAll(const SkOpSegment& );
void DumpAngles(const SkOpSegment& );
void DumpCoin(const SkOpSegment& );
void DumpPts(const SkOpSegment& , const char* prefix = "seg");

void Dump(const SkOpPtT& );
void DumpAll(const SkOpPtT& );

void Dump(const SkOpSpanBase& );
void DumpCoin(const SkOpSpanBase& );
void DumpAll(const SkOpSpanBase& );

void DumpCoin(const SkOpSpan& );
bool DumpSpan(const SkOpSpan& );

void Dump(const SkDConic& );
void DumpID(const SkDConic& , int id);

void Dump(const SkDCubic& );
void DumpID(const SkDCubic& , int id);

void Dump(const SkDLine& );
void DumpID(const SkDLine& , int id);

void Dump(const SkDQuad& );
void DumpID(const SkDQuad& , int id);

void Dump(const SkDPoint& );

void Dump(const SkOpAngle& );

// generates tools/path_sorter.htm and path_visualizer.htm compatible data
void DumpQ(const SkDQuad& quad1, const SkDQuad& quad2, int testNo);
void DumpT(const SkDQuad& quad, double t);

// global path dumps for msvs Visual Studio 17 to use from Immediate Window
void Dump(const SkPath& path);
void DumpHex(const SkPath& path);

#endif
