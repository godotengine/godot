/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#ifndef SkPathOps_DEFINED
#define SkPathOps_DEFINED

#include "include/core/SkTypes.h"
#include "include/private/SkTArray.h"
#include "include/private/SkTDArray.h"

class SkPath;
struct SkRect;


// FIXME: move everything below into the SkPath class
/**
  *  The logical operations that can be performed when combining two paths.
  */
enum SkPathOp {
    kDifference_SkPathOp,         //!< subtract the op path from the first path
    kIntersect_SkPathOp,          //!< intersect the two paths
    kUnion_SkPathOp,              //!< union (inclusive-or) the two paths
    kXOR_SkPathOp,                //!< exclusive-or the two paths
    kReverseDifference_SkPathOp,  //!< subtract the first path from the op path
};

/** Set this path to the result of applying the Op to this path and the
    specified path: this = (this op operand).
    The resulting path will be constructed from non-overlapping contours.
    The curve order is reduced where possible so that cubics may be turned
    into quadratics, and quadratics maybe turned into lines.

    Returns true if operation was able to produce a result;
    otherwise, result is unmodified.

    @param one The first operand (for difference, the minuend)
    @param two The second operand (for difference, the subtrahend)
    @param op The operator to apply.
    @param result The product of the operands. The result may be one of the
                  inputs.
    @return True if the operation succeeded.
  */
bool SK_API Op(const SkPath& one, const SkPath& two, SkPathOp op, SkPath* result);

/** Set this path to a set of non-overlapping contours that describe the
    same area as the original path.
    The curve order is reduced where possible so that cubics may
    be turned into quadratics, and quadratics maybe turned into lines.

    Returns true if operation was able to produce a result;
    otherwise, result is unmodified.

    @param path The path to simplify.
    @param result The simplified path. The result may be the input.
    @return True if simplification succeeded.
  */
bool SK_API Simplify(const SkPath& path, SkPath* result);

/** Set the resulting rectangle to the tight bounds of the path.

    @param path The path measured.
    @param result The tight bounds of the path.
    @return True if the bounds could be computed.
  */
bool SK_API TightBounds(const SkPath& path, SkRect* result);

/** Set the result with fill type winding to area equivalent to path.
    Returns true if successful. Does not detect if path contains contours which
    contain self-crossings or cross other contours; in these cases, may return
    true even though result does not fill same area as path.

    Returns true if operation was able to produce a result;
    otherwise, result is unmodified. The result may be the input.

    @param path The path typically with fill type set to even odd.
    @param result The equivalent path with fill type set to winding.
    @return True if winding path was set.
  */
bool SK_API AsWinding(const SkPath& path, SkPath* result);

/** Perform a series of path operations, optimized for unioning many paths together.
  */
class SK_API SkOpBuilder {
public:
    /** Add one or more paths and their operand. The builder is empty before the first
        path is added, so the result of a single add is (emptyPath OP path).

        @param path The second operand.
        @param _operator The operator to apply to the existing and supplied paths.
     */
    void add(const SkPath& path, SkPathOp _operator);

    /** Computes the sum of all paths and operands, and resets the builder to its
        initial state.

        @param result The product of the operands.
        @return True if the operation succeeded.
      */
    bool resolve(SkPath* result);

private:
    SkTArray<SkPath> fPathRefs;
    SkTDArray<SkPathOp> fOps;

    static bool FixWinding(SkPath* path);
    static void ReversePath(SkPath* path);
    void reset();
};

#endif
