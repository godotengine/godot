/*
 * Copyright 2015 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkPathPriv_DEFINED
#define SkPathPriv_DEFINED

#include "include/core/SkPathBuilder.h"
#include "include/core/SkRefCnt.h"
#include "include/private/SkIDChangeListener.h"
#include "include/private/SkPathRef.h"

static_assert(0 == static_cast<int>(SkPathFillType::kWinding), "fill_type_mismatch");
static_assert(1 == static_cast<int>(SkPathFillType::kEvenOdd), "fill_type_mismatch");
static_assert(2 == static_cast<int>(SkPathFillType::kInverseWinding), "fill_type_mismatch");
static_assert(3 == static_cast<int>(SkPathFillType::kInverseEvenOdd), "fill_type_mismatch");

class SkPathPriv {
public:
#ifdef SK_BUILD_FOR_ANDROID_FRAMEWORK
    static const int kPathRefGenIDBitCnt = 30; // leave room for the fill type (skbug.com/1762)
#else
    static const int kPathRefGenIDBitCnt = 32;
#endif

    // skbug.com/9906: Not a perfect solution for W plane clipping, but 1/16384 is a
    // reasonable limit (roughly 5e-5)
    inline static constexpr SkScalar kW0PlaneDistance = 1.f / (1 << 14);

    static SkPathFirstDirection AsFirstDirection(SkPathDirection dir) {
        // since we agree numerically for the values in Direction, we can just cast.
        return (SkPathFirstDirection)dir;
    }

    /**
     *  Return the opposite of the specified direction. kUnknown is its own
     *  opposite.
     */
    static SkPathFirstDirection OppositeFirstDirection(SkPathFirstDirection dir) {
        static const SkPathFirstDirection gOppositeDir[] = {
            SkPathFirstDirection::kCCW, SkPathFirstDirection::kCW, SkPathFirstDirection::kUnknown,
        };
        return gOppositeDir[(unsigned)dir];
    }

    /**
     *  Tries to compute the direction of the outer-most non-degenerate
     *  contour. If it can be computed, return that direction. If it cannot be determined,
     *  or the contour is known to be convex, return kUnknown. If the direction was determined,
     *  it is cached to make subsequent calls return quickly.
     */
    static SkPathFirstDirection ComputeFirstDirection(const SkPath&);

    static bool IsClosedSingleContour(const SkPath& path) {
        int verbCount = path.countVerbs();
        if (verbCount == 0)
            return false;
        int moveCount = 0;
        auto verbs = path.fPathRef->verbsBegin();
        for (int i = 0; i < verbCount; i++) {
            switch (verbs[i]) {
                case SkPath::Verb::kMove_Verb:
                    moveCount += 1;
                    if (moveCount > 1) {
                        return false;
                    }
                    break;
                case SkPath::Verb::kClose_Verb:
                    if (i == verbCount - 1) {
                        return true;
                    }
                    return false;
                default: break;
            }
        }
        return false;
    }

    // In some scenarios (e.g. fill or convexity checking all but the last leading move to are
    // irrelevant to behavior). SkPath::injectMoveToIfNeeded should ensure that this is always at
    // least 1.
    static int LeadingMoveToCount(const SkPath& path) {
        int verbCount = path.countVerbs();
        auto verbs = path.fPathRef->verbsBegin();
        for (int i = 0; i < verbCount; i++) {
            if (verbs[i] != SkPath::Verb::kMove_Verb) {
                return i;
            }
        }
        return verbCount; // path is all move verbs
    }

    static void AddGenIDChangeListener(const SkPath& path, sk_sp<SkIDChangeListener> listener) {
        path.fPathRef->addGenIDChangeListener(std::move(listener));
    }

    /**
     * This returns true for a rect that has a move followed by 3 or 4 lines and a close. If
     * 'isSimpleFill' is true, an uncloseed rect will also be accepted as long as it starts and
     * ends at the same corner. This does not permit degenerate line or point rectangles.
     */
    static bool IsSimpleRect(const SkPath& path, bool isSimpleFill, SkRect* rect,
                             SkPathDirection* direction, unsigned* start);

    /**
     * Creates a path from arc params using the semantics of SkCanvas::drawArc. This function
     * assumes empty ovals and zero sweeps have already been filtered out.
     */
    static void CreateDrawArcPath(SkPath* path, const SkRect& oval, SkScalar startAngle,
                                  SkScalar sweepAngle, bool useCenter, bool isFillNoPathEffect);

    /**
     * Determines whether an arc produced by CreateDrawArcPath will be convex. Assumes a non-empty
     * oval.
     */
    static bool DrawArcIsConvex(SkScalar sweepAngle, bool useCenter, bool isFillNoPathEffect);

    static void ShrinkToFit(SkPath* path) {
        path->shrinkToFit();
    }

    /**
     * Returns a C++11-iterable object that traverses a path's verbs in order. e.g:
     *
     *   for (SkPath::Verb verb : SkPathPriv::Verbs(path)) {
     *       ...
     *   }
     */
    struct Verbs {
    public:
        Verbs(const SkPath& path) : fPathRef(path.fPathRef.get()) {}
        struct Iter {
            void operator++() { fVerb++; }
            bool operator!=(const Iter& b) { return fVerb != b.fVerb; }
            SkPath::Verb operator*() { return static_cast<SkPath::Verb>(*fVerb); }
            const uint8_t* fVerb;
        };
        Iter begin() { return Iter{fPathRef->verbsBegin()}; }
        Iter end() { return Iter{fPathRef->verbsEnd()}; }
    private:
        Verbs(const Verbs&) = delete;
        Verbs& operator=(const Verbs&) = delete;
        SkPathRef* fPathRef;
    };

    /**
      * Iterates through a raw range of path verbs, points, and conics. All values are returned
      * unaltered.
      *
      * NOTE: This class's definition will be moved into SkPathPriv once RangeIter is removed.
    */
    using RangeIter = SkPath::RangeIter;

    /**
     * Iterable object for traversing verbs, points, and conic weights in a path:
     *
     *   for (auto [verb, pts, weights] : SkPathPriv::Iterate(skPath)) {
     *       ...
     *   }
     */
    struct Iterate {
    public:
        Iterate(const SkPath& path)
                : Iterate(path.fPathRef->verbsBegin(),
                          // Don't allow iteration through non-finite points.
                          (!path.isFinite()) ? path.fPathRef->verbsBegin()
                                             : path.fPathRef->verbsEnd(),
                          path.fPathRef->points(), path.fPathRef->conicWeights()) {
        }
        Iterate(const uint8_t* verbsBegin, const uint8_t* verbsEnd, const SkPoint* points,
                const SkScalar* weights)
                : fVerbsBegin(verbsBegin), fVerbsEnd(verbsEnd), fPoints(points), fWeights(weights) {
        }
        SkPath::RangeIter begin() { return {fVerbsBegin, fPoints, fWeights}; }
        SkPath::RangeIter end() { return {fVerbsEnd, nullptr, nullptr}; }
    private:
        const uint8_t* fVerbsBegin;
        const uint8_t* fVerbsEnd;
        const SkPoint* fPoints;
        const SkScalar* fWeights;
    };

    /**
     * Returns a pointer to the verb data.
     */
    static const uint8_t* VerbData(const SkPath& path) {
        return path.fPathRef->verbsBegin();
    }

    /** Returns a raw pointer to the path points */
    static const SkPoint* PointData(const SkPath& path) {
        return path.fPathRef->points();
    }

    /** Returns the number of conic weights in the path */
    static int ConicWeightCnt(const SkPath& path) {
        return path.fPathRef->countWeights();
    }

    /** Returns a raw pointer to the path conic weights. */
    static const SkScalar* ConicWeightData(const SkPath& path) {
        return path.fPathRef->conicWeights();
    }

    /** Returns true if the underlying SkPathRef has one single owner. */
    static bool TestingOnly_unique(const SkPath& path) {
        return path.fPathRef->unique();
    }

    // Won't be needed once we can make path's immutable (with their bounds always computed)
    static bool HasComputedBounds(const SkPath& path) {
        return path.hasComputedBounds();
    }

    /** Returns true if constructed by addCircle(), addOval(); and in some cases,
     addRoundRect(), addRRect(). SkPath constructed with conicTo() or rConicTo() will not
     return true though SkPath draws oval.

     rect receives bounds of oval.
     dir receives SkPathDirection of oval: kCW_Direction if clockwise, kCCW_Direction if
     counterclockwise.
     start receives start of oval: 0 for top, 1 for right, 2 for bottom, 3 for left.

     rect, dir, and start are unmodified if oval is not found.

     Triggers performance optimizations on some GPU surface implementations.

     @param rect   storage for bounding SkRect of oval; may be nullptr
     @param dir    storage for SkPathDirection; may be nullptr
     @param start  storage for start of oval; may be nullptr
     @return       true if SkPath was constructed by method that reduces to oval
     */
    static bool IsOval(const SkPath& path, SkRect* rect, SkPathDirection* dir, unsigned* start) {
        bool isCCW = false;
        bool result = path.fPathRef->isOval(rect, &isCCW, start);
        if (dir && result) {
            *dir = isCCW ? SkPathDirection::kCCW : SkPathDirection::kCW;
        }
        return result;
    }

    /** Returns true if constructed by addRoundRect(), addRRect(); and if construction
     is not empty, not SkRect, and not oval. SkPath constructed with other calls
     will not return true though SkPath draws SkRRect.

     rrect receives bounds of SkRRect.
     dir receives SkPathDirection of oval: kCW_Direction if clockwise, kCCW_Direction if
     counterclockwise.
     start receives start of SkRRect: 0 for top, 1 for right, 2 for bottom, 3 for left.

     rrect, dir, and start are unmodified if SkRRect is not found.

     Triggers performance optimizations on some GPU surface implementations.

     @param rrect  storage for bounding SkRect of SkRRect; may be nullptr
     @param dir    storage for SkPathDirection; may be nullptr
     @param start  storage for start of SkRRect; may be nullptr
     @return       true if SkPath contains only SkRRect
     */
    static bool IsRRect(const SkPath& path, SkRRect* rrect, SkPathDirection* dir,
                        unsigned* start) {
        bool isCCW = false;
        bool result = path.fPathRef->isRRect(rrect, &isCCW, start);
        if (dir && result) {
            *dir = isCCW ? SkPathDirection::kCCW : SkPathDirection::kCW;
        }
        return result;
    }

    /**
     *  Sometimes in the drawing pipeline, we have to perform math on path coordinates, even after
     *  the path is in device-coordinates. Tessellation and clipping are two examples. Usually this
     *  is pretty modest, but it can involve subtracting/adding coordinates, or multiplying by
     *  small constants (e.g. 2,3,4). To try to preflight issues where these optionations could turn
     *  finite path values into infinities (or NaNs), we allow the upper drawing code to reject
     *  the path if its bounds (in device coordinates) is too close to max float.
     */
    static bool TooBigForMath(const SkRect& bounds) {
        // This value is just a guess. smaller is safer, but we don't want to reject largish paths
        // that we don't have to.
        constexpr SkScalar scale_down_to_allow_for_small_multiplies = 0.25f;
        constexpr SkScalar max = SK_ScalarMax * scale_down_to_allow_for_small_multiplies;

        // use ! expression so we return true if bounds contains NaN
        return !(bounds.fLeft >= -max && bounds.fTop >= -max &&
                 bounds.fRight <= max && bounds.fBottom <= max);
    }
    static bool TooBigForMath(const SkPath& path) {
        return TooBigForMath(path.getBounds());
    }

    // Returns number of valid points for each SkPath::Iter verb
    static int PtsInIter(unsigned verb) {
        static const uint8_t gPtsInVerb[] = {
            1,  // kMove    pts[0]
            2,  // kLine    pts[0..1]
            3,  // kQuad    pts[0..2]
            3,  // kConic   pts[0..2]
            4,  // kCubic   pts[0..3]
            0,  // kClose
            0   // kDone
        };

        SkASSERT(verb < SK_ARRAY_COUNT(gPtsInVerb));
        return gPtsInVerb[verb];
    }

    // Returns number of valid points for each verb, not including the "starter"
    // point that the Iterator adds for line/quad/conic/cubic
    static int PtsInVerb(unsigned verb) {
        static const uint8_t gPtsInVerb[] = {
            1,  // kMove    pts[0]
            1,  // kLine    pts[0..1]
            2,  // kQuad    pts[0..2]
            2,  // kConic   pts[0..2]
            3,  // kCubic   pts[0..3]
            0,  // kClose
            0   // kDone
        };

        SkASSERT(verb < SK_ARRAY_COUNT(gPtsInVerb));
        return gPtsInVerb[verb];
    }

    static bool IsAxisAligned(const SkPath& path);

    static bool AllPointsEq(const SkPoint pts[], int count) {
        for (int i = 1; i < count; ++i) {
            if (pts[0] != pts[i]) {
                return false;
            }
        }
        return true;
    }

    static int LastMoveToIndex(const SkPath& path) { return path.fLastMoveToIndex; }

    static bool IsRectContour(const SkPath&, bool allowPartial, int* currVerb,
                              const SkPoint** ptsPtr, bool* isClosed, SkPathDirection* direction,
                              SkRect* rect);

    /** Returns true if SkPath is equivalent to nested SkRect pair when filled.
     If false, rect and dirs are unchanged.
     If true, rect and dirs are written to if not nullptr:
     setting rect[0] to outer SkRect, and rect[1] to inner SkRect;
     setting dirs[0] to SkPathDirection of outer SkRect, and dirs[1] to SkPathDirection of
     inner SkRect.

     @param rect  storage for SkRect pair; may be nullptr
     @param dirs  storage for SkPathDirection pair; may be nullptr
     @return      true if SkPath contains nested SkRect pair
     */
    static bool IsNestedFillRects(const SkPath&, SkRect rect[2],
                                  SkPathDirection dirs[2] = nullptr);

    static bool IsInverseFillType(SkPathFillType fill) {
        return (static_cast<int>(fill) & 2) != 0;
    }

    /** Returns equivalent SkPath::FillType representing SkPath fill inside its bounds.
     .

     @param fill  one of: kWinding_FillType, kEvenOdd_FillType,
     kInverseWinding_FillType, kInverseEvenOdd_FillType
     @return      fill, or kWinding_FillType or kEvenOdd_FillType if fill is inverted
     */
    static SkPathFillType ConvertToNonInverseFillType(SkPathFillType fill) {
        return (SkPathFillType)(static_cast<int>(fill) & 1);
    }

    /**
     *  If needed (to not blow-up under a perspective matrix), clip the path, returning the
     *  answer in "result", and return true.
     *
     *  Note result might be empty (if the path was completely clipped out).
     *
     *  If no clipping is needed, returns false and "result" is left unchanged.
     */
    static bool PerspectiveClip(const SkPath& src, const SkMatrix&, SkPath* result);

    /**
     * Gets the number of GenIDChangeListeners. If another thread has access to this path then
     * this may be stale before return and only indicates that the count was the return value
     * at some point during the execution of the function.
     */
    static int GenIDChangeListenersCount(const SkPath&);

    static void UpdatePathPoint(SkPath* path, int index, const SkPoint& pt) {
        SkASSERT(index < path->countPoints());
        SkPathRef::Editor ed(&path->fPathRef);
        ed.writablePoints()[index] = pt;
        path->dirtyAfterEdit();
    }

    static SkPathConvexity GetConvexity(const SkPath& path) {
        return path.getConvexity();
    }
    static SkPathConvexity GetConvexityOrUnknown(const SkPath& path) {
        return path.getConvexityOrUnknown();
    }
    static void SetConvexity(const SkPath& path, SkPathConvexity c) {
        path.setConvexity(c);
    }
    static void ForceComputeConvexity(const SkPath& path) {
        path.setConvexity(SkPathConvexity::kUnknown);
        (void)path.isConvex();
    }

    static void ReverseAddPath(SkPathBuilder* builder, const SkPath& reverseMe) {
        builder->privateReverseAddPath(reverseMe);
    }
};

// Lightweight variant of SkPath::Iter that only returns segments (e.g. lines/conics).
// Does not return kMove or kClose.
// Always "auto-closes" each contour.
// Roughly the same as SkPath::Iter(path, true), but does not return moves or closes
//
class SkPathEdgeIter {
    const uint8_t*  fVerbs;
    const uint8_t*  fVerbsStop;
    const SkPoint*  fPts;
    const SkPoint*  fMoveToPtr;
    const SkScalar* fConicWeights;
    SkPoint         fScratch[2];    // for auto-close lines
    bool            fNeedsCloseLine;
    bool            fNextIsNewContour;
    SkDEBUGCODE(bool fIsConic;)

    enum {
        kIllegalEdgeValue = 99
    };

public:
    SkPathEdgeIter(const SkPath& path);

    SkScalar conicWeight() const {
        SkASSERT(fIsConic);
        return *fConicWeights;
    }

    enum class Edge {
        kLine  = SkPath::kLine_Verb,
        kQuad  = SkPath::kQuad_Verb,
        kConic = SkPath::kConic_Verb,
        kCubic = SkPath::kCubic_Verb,
    };

    static SkPath::Verb EdgeToVerb(Edge e) {
        return SkPath::Verb(e);
    }

    struct Result {
        const SkPoint*  fPts;   // points for the segment, or null if done
        Edge            fEdge;
        bool            fIsNewContour;

        // Returns true when it holds an Edge, false when the path is done.
        explicit operator bool() { return fPts != nullptr; }
    };

    Result next() {
        auto closeline = [&]() {
            fScratch[0] = fPts[-1];
            fScratch[1] = *fMoveToPtr;
            fNeedsCloseLine = false;
            fNextIsNewContour = true;
            return Result{ fScratch, Edge::kLine, false };
        };

        for (;;) {
            SkASSERT(fVerbs <= fVerbsStop);
            if (fVerbs == fVerbsStop) {
                return fNeedsCloseLine
                    ? closeline()
                    : Result{ nullptr, Edge(kIllegalEdgeValue), false };
            }

            SkDEBUGCODE(fIsConic = false;)

            const auto v = *fVerbs++;
            switch (v) {
                case SkPath::kMove_Verb: {
                    if (fNeedsCloseLine) {
                        auto res = closeline();
                        fMoveToPtr = fPts++;
                        return res;
                    }
                    fMoveToPtr = fPts++;
                    fNextIsNewContour = true;
                } break;
                case SkPath::kClose_Verb:
                    if (fNeedsCloseLine) return closeline();
                    break;
                default: {
                    // Actual edge.
                    const int pts_count = (v+2) / 2,
                              cws_count = (v & (v-1)) / 2;
                    SkASSERT(pts_count == SkPathPriv::PtsInIter(v) - 1);

                    fNeedsCloseLine = true;
                    fPts           += pts_count;
                    fConicWeights  += cws_count;

                    SkDEBUGCODE(fIsConic = (v == SkPath::kConic_Verb);)
                    SkASSERT(fIsConic == (cws_count > 0));

                    bool isNewContour = fNextIsNewContour;
                    fNextIsNewContour = false;
                    return { &fPts[-(pts_count + 1)], Edge(v), isNewContour };
                }
            }
        }
    }
};

#endif
