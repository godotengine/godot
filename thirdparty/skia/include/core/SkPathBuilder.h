/*
 * Copyright 2015 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkPathBuilder_DEFINED
#define SkPathBuilder_DEFINED

#include "include/core/SkMatrix.h"
#include "include/core/SkPath.h"
#include "include/core/SkPathTypes.h"
#include "include/core/SkRefCnt.h"
#include "include/private/SkTDArray.h"

class SK_API SkPathBuilder {
public:
    SkPathBuilder();
    SkPathBuilder(SkPathFillType);
    SkPathBuilder(const SkPath&);
    SkPathBuilder(const SkPathBuilder&) = default;
    ~SkPathBuilder();

    SkPathBuilder& operator=(const SkPath&);
    SkPathBuilder& operator=(const SkPathBuilder&) = default;

    SkPathFillType fillType() const { return fFillType; }
    SkRect computeBounds() const;

    SkPath snapshot() const;  // the builder is unchanged after returning this path
    SkPath detach();    // the builder is reset to empty after returning this path

    SkPathBuilder& setFillType(SkPathFillType ft) { fFillType = ft; return *this; }
    SkPathBuilder& setIsVolatile(bool isVolatile) { fIsVolatile = isVolatile; return *this; }

    SkPathBuilder& reset();

    SkPathBuilder& moveTo(SkPoint pt);
    SkPathBuilder& moveTo(SkScalar x, SkScalar y) { return this->moveTo(SkPoint::Make(x, y)); }

    SkPathBuilder& lineTo(SkPoint pt);
    SkPathBuilder& lineTo(SkScalar x, SkScalar y) { return this->lineTo(SkPoint::Make(x, y)); }

    SkPathBuilder& quadTo(SkPoint pt1, SkPoint pt2);
    SkPathBuilder& quadTo(SkScalar x1, SkScalar y1, SkScalar x2, SkScalar y2) {
        return this->quadTo(SkPoint::Make(x1, y1), SkPoint::Make(x2, y2));
    }
    SkPathBuilder& quadTo(const SkPoint pts[2]) { return this->quadTo(pts[0], pts[1]); }

    SkPathBuilder& conicTo(SkPoint pt1, SkPoint pt2, SkScalar w);
    SkPathBuilder& conicTo(SkScalar x1, SkScalar y1, SkScalar x2, SkScalar y2, SkScalar w) {
        return this->conicTo(SkPoint::Make(x1, y1), SkPoint::Make(x2, y2), w);
    }
    SkPathBuilder& conicTo(const SkPoint pts[2], SkScalar w) {
        return this->conicTo(pts[0], pts[1], w);
    }

    SkPathBuilder& cubicTo(SkPoint pt1, SkPoint pt2, SkPoint pt3);
    SkPathBuilder& cubicTo(SkScalar x1, SkScalar y1, SkScalar x2, SkScalar y2, SkScalar x3, SkScalar y3) {
        return this->cubicTo(SkPoint::Make(x1, y1), SkPoint::Make(x2, y2), SkPoint::Make(x3, y3));
    }
    SkPathBuilder& cubicTo(const SkPoint pts[3]) {
        return this->cubicTo(pts[0], pts[1], pts[2]);
    }

    SkPathBuilder& close();

    // Append a series of lineTo(...)
    SkPathBuilder& polylineTo(const SkPoint pts[], int count);
    SkPathBuilder& polylineTo(const std::initializer_list<SkPoint>& list) {
        return this->polylineTo(list.begin(), SkToInt(list.size()));
    }

    // Relative versions of segments, relative to the previous position.

    SkPathBuilder& rLineTo(SkPoint pt);
    SkPathBuilder& rLineTo(SkScalar x, SkScalar y) { return this->rLineTo({x, y}); }
    SkPathBuilder& rQuadTo(SkPoint pt1, SkPoint pt2);
    SkPathBuilder& rQuadTo(SkScalar x1, SkScalar y1, SkScalar x2, SkScalar y2) {
        return this->rQuadTo({x1, y1}, {x2, y2});
    }
    SkPathBuilder& rConicTo(SkPoint p1, SkPoint p2, SkScalar w);
    SkPathBuilder& rConicTo(SkScalar x1, SkScalar y1, SkScalar x2, SkScalar y2, SkScalar w) {
        return this->rConicTo({x1, y1}, {x2, y2}, w);
    }
    SkPathBuilder& rCubicTo(SkPoint pt1, SkPoint pt2, SkPoint pt3);
    SkPathBuilder& rCubicTo(SkScalar x1, SkScalar y1, SkScalar x2, SkScalar y2, SkScalar x3, SkScalar y3) {
        return this->rCubicTo({x1, y1}, {x2, y2}, {x3, y3});
    }

    // Arcs

    /** Appends arc to the builder. Arc added is part of ellipse
        bounded by oval, from startAngle through sweepAngle. Both startAngle and
        sweepAngle are measured in degrees, where zero degrees is aligned with the
        positive x-axis, and positive sweeps extends arc clockwise.

        arcTo() adds line connecting the builder's last point to initial arc point if forceMoveTo
        is false and the builder is not empty. Otherwise, added contour begins with first point
        of arc. Angles greater than -360 and less than 360 are treated modulo 360.

        @param oval          bounds of ellipse containing arc
        @param startAngleDeg starting angle of arc in degrees
        @param sweepAngleDeg sweep, in degrees. Positive is clockwise; treated modulo 360
        @param forceMoveTo   true to start a new contour with arc
        @return              reference to the builder
    */
    SkPathBuilder& arcTo(const SkRect& oval, SkScalar startAngleDeg, SkScalar sweepAngleDeg,
                         bool forceMoveTo);

    /** Appends arc to SkPath, after appending line if needed. Arc is implemented by conic
        weighted to describe part of circle. Arc is contained by tangent from
        last SkPath point to p1, and tangent from p1 to p2. Arc
        is part of circle sized to radius, positioned so it touches both tangent lines.

        If last SkPath SkPoint does not start arc, arcTo() appends connecting line to SkPath.
        The length of vector from p1 to p2 does not affect arc.

        Arc sweep is always less than 180 degrees. If radius is zero, or if
        tangents are nearly parallel, arcTo() appends line from last SkPath SkPoint to p1.

        arcTo() appends at most one line and one conic.
        arcTo() implements the functionality of PostScript arct and HTML Canvas arcTo.

        @param p1      SkPoint common to pair of tangents
        @param p2      end of second tangent
        @param radius  distance from arc to circle center
        @return        reference to SkPath
    */
    SkPathBuilder& arcTo(SkPoint p1, SkPoint p2, SkScalar radius);

    enum ArcSize {
        kSmall_ArcSize, //!< smaller of arc pair
        kLarge_ArcSize, //!< larger of arc pair
    };

    /** Appends arc to SkPath. Arc is implemented by one or more conic weighted to describe
        part of oval with radii (r.fX, r.fY) rotated by xAxisRotate degrees. Arc curves
        from last SkPath SkPoint to (xy.fX, xy.fY), choosing one of four possible routes:
        clockwise or counterclockwise,
        and smaller or larger.

        Arc sweep is always less than 360 degrees. arcTo() appends line to xy if either
        radii are zero, or if last SkPath SkPoint equals (xy.fX, xy.fY). arcTo() scales radii r to
        fit last SkPath SkPoint and xy if both are greater than zero but too small to describe
        an arc.

        arcTo() appends up to four conic curves.
        arcTo() implements the functionality of SVG arc, although SVG sweep-flag value is
        opposite the integer value of sweep; SVG sweep-flag uses 1 for clockwise, while
        kCW_Direction cast to int is zero.

        @param r            radii on axes before x-axis rotation
        @param xAxisRotate  x-axis rotation in degrees; positive values are clockwise
        @param largeArc     chooses smaller or larger arc
        @param sweep        chooses clockwise or counterclockwise arc
        @param xy           end of arc
        @return             reference to SkPath
    */
    SkPathBuilder& arcTo(SkPoint r, SkScalar xAxisRotate, ArcSize largeArc, SkPathDirection sweep,
                         SkPoint xy);

    /** Appends arc to the builder, as the start of new contour. Arc added is part of ellipse
        bounded by oval, from startAngle through sweepAngle. Both startAngle and
        sweepAngle are measured in degrees, where zero degrees is aligned with the
        positive x-axis, and positive sweeps extends arc clockwise.

        If sweepAngle <= -360, or sweepAngle >= 360; and startAngle modulo 90 is nearly
        zero, append oval instead of arc. Otherwise, sweepAngle values are treated
        modulo 360, and arc may or may not draw depending on numeric rounding.

        @param oval          bounds of ellipse containing arc
        @param startAngleDeg starting angle of arc in degrees
        @param sweepAngleDeg sweep, in degrees. Positive is clockwise; treated modulo 360
        @return              reference to this builder
    */
    SkPathBuilder& addArc(const SkRect& oval, SkScalar startAngleDeg, SkScalar sweepAngleDeg);

    // Add a new contour

    SkPathBuilder& addRect(const SkRect&, SkPathDirection, unsigned startIndex);
    SkPathBuilder& addOval(const SkRect&, SkPathDirection, unsigned startIndex);
    SkPathBuilder& addRRect(const SkRRect&, SkPathDirection, unsigned startIndex);

    SkPathBuilder& addRect(const SkRect& rect, SkPathDirection dir = SkPathDirection::kCW) {
        return this->addRect(rect, dir, 0);
    }
    SkPathBuilder& addOval(const SkRect& rect, SkPathDirection dir = SkPathDirection::kCW) {
        // legacy start index: 1
        return this->addOval(rect, dir, 1);
    }
    SkPathBuilder& addRRect(const SkRRect& rrect, SkPathDirection dir = SkPathDirection::kCW) {
        // legacy start indices: 6 (CW) and 7 (CCW)
        return this->addRRect(rrect, dir, dir == SkPathDirection::kCW ? 6 : 7);
    }

    SkPathBuilder& addCircle(SkScalar center_x, SkScalar center_y, SkScalar radius,
                             SkPathDirection dir = SkPathDirection::kCW);

    SkPathBuilder& addPolygon(const SkPoint pts[], int count, bool isClosed);
    SkPathBuilder& addPolygon(const std::initializer_list<SkPoint>& list, bool isClosed) {
        return this->addPolygon(list.begin(), SkToInt(list.size()), isClosed);
    }

    SkPathBuilder& addPath(const SkPath&);

    // Performance hint, to reserve extra storage for subsequent calls to lineTo, quadTo, etc.

    void incReserve(int extraPtCount, int extraVerbCount);
    void incReserve(int extraPtCount) {
        this->incReserve(extraPtCount, extraPtCount);
    }

    SkPathBuilder& offset(SkScalar dx, SkScalar dy);

    SkPathBuilder& toggleInverseFillType() {
        fFillType = (SkPathFillType)((unsigned)fFillType ^ 2);
        return *this;
    }

private:
    SkTDArray<SkPoint>  fPts;
    SkTDArray<uint8_t>  fVerbs;
    SkTDArray<SkScalar> fConicWeights;

    SkPathFillType      fFillType;
    bool                fIsVolatile;

    unsigned    fSegmentMask;
    SkPoint     fLastMovePoint;
    int         fLastMoveIndex; // only needed until SkPath is immutable
    bool        fNeedsMoveVerb;

    enum IsA {
        kIsA_JustMoves,     // we only have 0 or more moves
        kIsA_MoreThanMoves, // we have verbs other than just move
        kIsA_Oval,          // we are 0 or more moves followed by an oval
        kIsA_RRect,         // we are 0 or more moves followed by a rrect
    };
    IsA fIsA      = kIsA_JustMoves;
    int fIsAStart = -1;     // tracks direction iff fIsA is not unknown
    bool fIsACCW  = false;  // tracks direction iff fIsA is not unknown

    int countVerbs() const { return fVerbs.count(); }

    // called right before we add a (non-move) verb
    void ensureMove() {
        fIsA = kIsA_MoreThanMoves;
        if (fNeedsMoveVerb) {
            this->moveTo(fLastMovePoint);
        }
    }

    SkPath make(sk_sp<SkPathRef>) const;

    SkPathBuilder& privateReverseAddPath(const SkPath&);

    friend class SkPathPriv;
};

#endif

