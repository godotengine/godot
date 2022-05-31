/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "include/core/SkMatrix.h"
// -- GODOT start --
//#include "include/core/SkString.h"
#include "include/private/SkMalloc.h"
//#include "src/core/SkBuffer.h"
// -- GODOT end --
#include "src/core/SkRRectPriv.h"
#include "src/core/SkRectPriv.h"
#include "src/core/SkScaleToSides.h"

#include <cmath>
#include <utility>

///////////////////////////////////////////////////////////////////////////////

void SkRRect::setOval(const SkRect& oval) {
    if (!this->initializeRect(oval)) {
        return;
    }

    SkScalar xRad = SkRectPriv::HalfWidth(fRect);
    SkScalar yRad = SkRectPriv::HalfHeight(fRect);

    if (xRad == 0.0f || yRad == 0.0f) {
        // All the corners will be square
        memset(fRadii, 0, sizeof(fRadii));
        fType = kRect_Type;
    } else {
        for (int i = 0; i < 4; ++i) {
            fRadii[i].set(xRad, yRad);
        }
        fType = kOval_Type;
    }

    SkASSERT(this->isValid());
}

void SkRRect::setRectXY(const SkRect& rect, SkScalar xRad, SkScalar yRad) {
    if (!this->initializeRect(rect)) {
        return;
    }

    if (!SkScalarsAreFinite(xRad, yRad)) {
        xRad = yRad = 0;    // devolve into a simple rect
    }

    if (fRect.width() < xRad+xRad || fRect.height() < yRad+yRad) {
        // At most one of these two divides will be by zero, and neither numerator is zero.
        SkScalar scale = std::min(sk_ieee_float_divide(fRect. width(), xRad + xRad),
                                     sk_ieee_float_divide(fRect.height(), yRad + yRad));
        SkASSERT(scale < SK_Scalar1);
        xRad *= scale;
        yRad *= scale;
    }

    if (xRad <= 0 || yRad <= 0) {
        // all corners are square in this case
        this->setRect(rect);
        return;
    }

    for (int i = 0; i < 4; ++i) {
        fRadii[i].set(xRad, yRad);
    }
    fType = kSimple_Type;
    if (xRad >= SkScalarHalf(fRect.width()) && yRad >= SkScalarHalf(fRect.height())) {
        fType = kOval_Type;
        // TODO: assert that all the x&y radii are already W/2 & H/2
    }

    SkASSERT(this->isValid());
}

void SkRRect::setNinePatch(const SkRect& rect, SkScalar leftRad, SkScalar topRad,
                           SkScalar rightRad, SkScalar bottomRad) {
    if (!this->initializeRect(rect)) {
        return;
    }

    const SkScalar array[4] = { leftRad, topRad, rightRad, bottomRad };
    if (!SkScalarsAreFinite(array, 4)) {
        this->setRect(rect);    // devolve into a simple rect
        return;
    }

    leftRad = std::max(leftRad, 0.0f);
    topRad = std::max(topRad, 0.0f);
    rightRad = std::max(rightRad, 0.0f);
    bottomRad = std::max(bottomRad, 0.0f);

    SkScalar scale = SK_Scalar1;
    if (leftRad + rightRad > fRect.width()) {
        scale = fRect.width() / (leftRad + rightRad);
    }
    if (topRad + bottomRad > fRect.height()) {
        scale = std::min(scale, fRect.height() / (topRad + bottomRad));
    }

    if (scale < SK_Scalar1) {
        leftRad *= scale;
        topRad *= scale;
        rightRad *= scale;
        bottomRad *= scale;
    }

    if (leftRad == rightRad && topRad == bottomRad) {
        if (leftRad >= SkScalarHalf(fRect.width()) && topRad >= SkScalarHalf(fRect.height())) {
            fType = kOval_Type;
        } else if (0 == leftRad || 0 == topRad) {
            // If the left and (by equality check above) right radii are zero then it is a rect.
            // Same goes for top/bottom.
            fType = kRect_Type;
            leftRad = 0;
            topRad = 0;
            rightRad = 0;
            bottomRad = 0;
        } else {
            fType = kSimple_Type;
        }
    } else {
        fType = kNinePatch_Type;
    }

    fRadii[kUpperLeft_Corner].set(leftRad, topRad);
    fRadii[kUpperRight_Corner].set(rightRad, topRad);
    fRadii[kLowerRight_Corner].set(rightRad, bottomRad);
    fRadii[kLowerLeft_Corner].set(leftRad, bottomRad);

    SkASSERT(this->isValid());
}

// These parameters intentionally double. Apropos crbug.com/463920, if one of the
// radii is huge while the other is small, single precision math can completely
// miss the fact that a scale is required.
static double compute_min_scale(double rad1, double rad2, double limit, double curMin) {
    if ((rad1 + rad2) > limit) {
        return std::min(curMin, limit / (rad1 + rad2));
    }
    return curMin;
}

static bool clamp_to_zero(SkVector radii[4]) {
    bool allCornersSquare = true;

    // Clamp negative radii to zero
    for (int i = 0; i < 4; ++i) {
        if (radii[i].fX <= 0 || radii[i].fY <= 0) {
            // In this case we are being a little fast & loose. Since one of
            // the radii is 0 the corner is square. However, the other radii
            // could still be non-zero and play in the global scale factor
            // computation.
            radii[i].fX = 0;
            radii[i].fY = 0;
        } else {
            allCornersSquare = false;
        }
    }

    return allCornersSquare;
}

void SkRRect::setRectRadii(const SkRect& rect, const SkVector radii[4]) {
    if (!this->initializeRect(rect)) {
        return;
    }

    if (!SkScalarsAreFinite(&radii[0].fX, 8)) {
        this->setRect(rect);    // devolve into a simple rect
        return;
    }

    memcpy(fRadii, radii, sizeof(fRadii));

    if (clamp_to_zero(fRadii)) {
        this->setRect(rect);
        return;
    }

    this->scaleRadii();

    if (!this->isValid()) {
        this->setRect(rect);
        return;
    }
}

bool SkRRect::initializeRect(const SkRect& rect) {
    // Check this before sorting because sorting can hide nans.
    if (!rect.isFinite()) {
        *this = SkRRect();
        return false;
    }
    fRect = rect.makeSorted();
    if (fRect.isEmpty()) {
        memset(fRadii, 0, sizeof(fRadii));
        fType = kEmpty_Type;
        return false;
    }
    return true;
}

// If we can't distinguish one of the radii relative to the other, force it to zero so it
// doesn't confuse us later. See crbug.com/850350
//
static void flush_to_zero(SkScalar& a, SkScalar& b) {
    SkASSERT(a >= 0);
    SkASSERT(b >= 0);
    if (a + b == a) {
        b = 0;
    } else if (a + b == b) {
        a = 0;
    }
}

bool SkRRect::scaleRadii() {
    // Proportionally scale down all radii to fit. Find the minimum ratio
    // of a side and the radii on that side (for all four sides) and use
    // that to scale down _all_ the radii. This algorithm is from the
    // W3 spec (http://www.w3.org/TR/css3-background/) section 5.5 - Overlapping
    // Curves:
    // "Let f = min(Li/Si), where i is one of { top, right, bottom, left },
    //   Si is the sum of the two corresponding radii of the corners on side i,
    //   and Ltop = Lbottom = the width of the box,
    //   and Lleft = Lright = the height of the box.
    // If f < 1, then all corner radii are reduced by multiplying them by f."
    double scale = 1.0;

    // The sides of the rectangle may be larger than a float.
    double width = (double)fRect.fRight - (double)fRect.fLeft;
    double height = (double)fRect.fBottom - (double)fRect.fTop;
    scale = compute_min_scale(fRadii[0].fX, fRadii[1].fX, width,  scale);
    scale = compute_min_scale(fRadii[1].fY, fRadii[2].fY, height, scale);
    scale = compute_min_scale(fRadii[2].fX, fRadii[3].fX, width,  scale);
    scale = compute_min_scale(fRadii[3].fY, fRadii[0].fY, height, scale);

    flush_to_zero(fRadii[0].fX, fRadii[1].fX);
    flush_to_zero(fRadii[1].fY, fRadii[2].fY);
    flush_to_zero(fRadii[2].fX, fRadii[3].fX);
    flush_to_zero(fRadii[3].fY, fRadii[0].fY);

    if (scale < 1.0) {
        SkScaleToSides::AdjustRadii(width,  scale, &fRadii[0].fX, &fRadii[1].fX);
        SkScaleToSides::AdjustRadii(height, scale, &fRadii[1].fY, &fRadii[2].fY);
        SkScaleToSides::AdjustRadii(width,  scale, &fRadii[2].fX, &fRadii[3].fX);
        SkScaleToSides::AdjustRadii(height, scale, &fRadii[3].fY, &fRadii[0].fY);
    }

    // adjust radii may set x or y to zero; set companion to zero as well
    clamp_to_zero(fRadii);

    // May be simple, oval, or complex, or become a rect/empty if the radii adjustment made them 0
    this->computeType();

    // TODO:  Why can't we assert this here?
    //SkASSERT(this->isValid());

    return scale < 1.0;
}

// This method determines if a point known to be inside the RRect's bounds is
// inside all the corners.
bool SkRRect::checkCornerContainment(SkScalar x, SkScalar y) const {
    SkPoint canonicalPt; // (x,y) translated to one of the quadrants
    int index;

    if (kOval_Type == this->type()) {
        canonicalPt.set(x - fRect.centerX(), y - fRect.centerY());
        index = kUpperLeft_Corner;  // any corner will do in this case
    } else {
        if (x < fRect.fLeft + fRadii[kUpperLeft_Corner].fX &&
            y < fRect.fTop + fRadii[kUpperLeft_Corner].fY) {
            // UL corner
            index = kUpperLeft_Corner;
            canonicalPt.set(x - (fRect.fLeft + fRadii[kUpperLeft_Corner].fX),
                            y - (fRect.fTop + fRadii[kUpperLeft_Corner].fY));
            SkASSERT(canonicalPt.fX < 0 && canonicalPt.fY < 0);
        } else if (x < fRect.fLeft + fRadii[kLowerLeft_Corner].fX &&
                   y > fRect.fBottom - fRadii[kLowerLeft_Corner].fY) {
            // LL corner
            index = kLowerLeft_Corner;
            canonicalPt.set(x - (fRect.fLeft + fRadii[kLowerLeft_Corner].fX),
                            y - (fRect.fBottom - fRadii[kLowerLeft_Corner].fY));
            SkASSERT(canonicalPt.fX < 0 && canonicalPt.fY > 0);
        } else if (x > fRect.fRight - fRadii[kUpperRight_Corner].fX &&
                   y < fRect.fTop + fRadii[kUpperRight_Corner].fY) {
            // UR corner
            index = kUpperRight_Corner;
            canonicalPt.set(x - (fRect.fRight - fRadii[kUpperRight_Corner].fX),
                            y - (fRect.fTop + fRadii[kUpperRight_Corner].fY));
            SkASSERT(canonicalPt.fX > 0 && canonicalPt.fY < 0);
        } else if (x > fRect.fRight - fRadii[kLowerRight_Corner].fX &&
                   y > fRect.fBottom - fRadii[kLowerRight_Corner].fY) {
            // LR corner
            index = kLowerRight_Corner;
            canonicalPt.set(x - (fRect.fRight - fRadii[kLowerRight_Corner].fX),
                            y - (fRect.fBottom - fRadii[kLowerRight_Corner].fY));
            SkASSERT(canonicalPt.fX > 0 && canonicalPt.fY > 0);
        } else {
            // not in any of the corners
            return true;
        }
    }

    // A point is in an ellipse (in standard position) if:
    //      x^2     y^2
    //     ----- + ----- <= 1
    //      a^2     b^2
    // or :
    //     b^2*x^2 + a^2*y^2 <= (ab)^2
    SkScalar dist =  SkScalarSquare(canonicalPt.fX) * SkScalarSquare(fRadii[index].fY) +
                     SkScalarSquare(canonicalPt.fY) * SkScalarSquare(fRadii[index].fX);
    return dist <= SkScalarSquare(fRadii[index].fX * fRadii[index].fY);
}

bool SkRRectPriv::IsNearlySimpleCircular(const SkRRect& rr, SkScalar tolerance) {
    SkScalar simpleRadius = rr.fRadii[0].fX;
    return SkScalarNearlyEqual(simpleRadius, rr.fRadii[0].fY, tolerance) &&
           SkScalarNearlyEqual(simpleRadius, rr.fRadii[1].fX, tolerance) &&
           SkScalarNearlyEqual(simpleRadius, rr.fRadii[1].fY, tolerance) &&
           SkScalarNearlyEqual(simpleRadius, rr.fRadii[2].fX, tolerance) &&
           SkScalarNearlyEqual(simpleRadius, rr.fRadii[2].fY, tolerance) &&
           SkScalarNearlyEqual(simpleRadius, rr.fRadii[3].fX, tolerance) &&
           SkScalarNearlyEqual(simpleRadius, rr.fRadii[3].fY, tolerance);
}

bool SkRRectPriv::AllCornersCircular(const SkRRect& rr, SkScalar tolerance) {
    return SkScalarNearlyEqual(rr.fRadii[0].fX, rr.fRadii[0].fY, tolerance) &&
           SkScalarNearlyEqual(rr.fRadii[1].fX, rr.fRadii[1].fY, tolerance) &&
           SkScalarNearlyEqual(rr.fRadii[2].fX, rr.fRadii[2].fY, tolerance) &&
           SkScalarNearlyEqual(rr.fRadii[3].fX, rr.fRadii[3].fY, tolerance);
}

bool SkRRect::contains(const SkRect& rect) const {
    if (!this->getBounds().contains(rect)) {
        // If 'rect' isn't contained by the RR's bounds then the
        // RR definitely doesn't contain it
        return false;
    }

    if (this->isRect()) {
        // the prior test was sufficient
        return true;
    }

    // At this point we know all four corners of 'rect' are inside the
    // bounds of of this RR. Check to make sure all the corners are inside
    // all the curves
    return this->checkCornerContainment(rect.fLeft, rect.fTop) &&
           this->checkCornerContainment(rect.fRight, rect.fTop) &&
           this->checkCornerContainment(rect.fRight, rect.fBottom) &&
           this->checkCornerContainment(rect.fLeft, rect.fBottom);
}

static bool radii_are_nine_patch(const SkVector radii[4]) {
    return radii[SkRRect::kUpperLeft_Corner].fX == radii[SkRRect::kLowerLeft_Corner].fX &&
           radii[SkRRect::kUpperLeft_Corner].fY == radii[SkRRect::kUpperRight_Corner].fY &&
           radii[SkRRect::kUpperRight_Corner].fX == radii[SkRRect::kLowerRight_Corner].fX &&
           radii[SkRRect::kLowerLeft_Corner].fY == radii[SkRRect::kLowerRight_Corner].fY;
}

// There is a simplified version of this method in setRectXY
void SkRRect::computeType() {
    if (fRect.isEmpty()) {
        SkASSERT(fRect.isSorted());
        for (size_t i = 0; i < SK_ARRAY_COUNT(fRadii); ++i) {
            SkASSERT((fRadii[i] == SkVector{0, 0}));
        }
        fType = kEmpty_Type;
        SkASSERT(this->isValid());
        return;
    }

    bool allRadiiEqual = true; // are all x radii equal and all y radii?
    bool allCornersSquare = 0 == fRadii[0].fX || 0 == fRadii[0].fY;

    for (int i = 1; i < 4; ++i) {
        if (0 != fRadii[i].fX && 0 != fRadii[i].fY) {
            // if either radius is zero the corner is square so both have to
            // be non-zero to have a rounded corner
            allCornersSquare = false;
        }
        if (fRadii[i].fX != fRadii[i-1].fX || fRadii[i].fY != fRadii[i-1].fY) {
            allRadiiEqual = false;
        }
    }

    if (allCornersSquare) {
        fType = kRect_Type;
        SkASSERT(this->isValid());
        return;
    }

    if (allRadiiEqual) {
        if (fRadii[0].fX >= SkScalarHalf(fRect.width()) &&
            fRadii[0].fY >= SkScalarHalf(fRect.height())) {
            fType = kOval_Type;
        } else {
            fType = kSimple_Type;
        }
        SkASSERT(this->isValid());
        return;
    }

    if (radii_are_nine_patch(fRadii)) {
        fType = kNinePatch_Type;
    } else {
        fType = kComplex_Type;
    }

    if (!this->isValid()) {
        this->setRect(this->rect());
        SkASSERT(this->isValid());
    }
}

bool SkRRect::transform(const SkMatrix& matrix, SkRRect* dst) const {
    if (nullptr == dst) {
        return false;
    }

    // Assert that the caller is not trying to do this in place, which
    // would violate const-ness. Do not return false though, so that
    // if they know what they're doing and want to violate it they can.
    SkASSERT(dst != this);

    if (matrix.isIdentity()) {
        *dst = *this;
        return true;
    }

    if (!matrix.preservesAxisAlignment()) {
        return false;
    }

    SkRect newRect;
    if (!matrix.mapRect(&newRect, fRect)) {
        return false;
    }

    // The matrix may have scaled us to zero (or due to float madness, we now have collapsed
    // some dimension of the rect, so we need to check for that. Note that matrix must be
    // scale and translate and mapRect() produces a sorted rect. So an empty rect indicates
    // loss of precision.
    if (!newRect.isFinite() || newRect.isEmpty()) {
        return false;
    }

    // At this point, this is guaranteed to succeed, so we can modify dst.
    dst->fRect = newRect;

    // Since the only transforms that were allowed are axis aligned, the type
    // remains unchanged.
    dst->fType = fType;

    if (kRect_Type == fType) {
        SkASSERT(dst->isValid());
        return true;
    }
    if (kOval_Type == fType) {
        for (int i = 0; i < 4; ++i) {
            dst->fRadii[i].fX = SkScalarHalf(newRect.width());
            dst->fRadii[i].fY = SkScalarHalf(newRect.height());
        }
        SkASSERT(dst->isValid());
        return true;
    }

    // Now scale each corner
    SkScalar xScale = matrix.getScaleX();
    SkScalar yScale = matrix.getScaleY();

    // There is a rotation of 90 (Clockwise 90) or 270 (Counter clockwise 90).
    // 180 degrees rotations are simply flipX with a flipY and would come under
    // a scale transform.
    if (!matrix.isScaleTranslate()) {
        const bool isClockwise = matrix.getSkewX() < 0;

        // The matrix location for scale changes if there is a rotation.
        xScale = matrix.getSkewY() * (isClockwise ? 1 : -1);
        yScale = matrix.getSkewX() * (isClockwise ? -1 : 1);

        const int dir = isClockwise ? 3 : 1;
        for (int i = 0; i < 4; ++i) {
            const int src = (i + dir) >= 4 ? (i + dir) % 4 : (i + dir);
            // Swap X and Y axis for the radii.
            dst->fRadii[i].fX = fRadii[src].fY;
            dst->fRadii[i].fY = fRadii[src].fX;
        }
    } else {
        for (int i = 0; i < 4; ++i) {
            dst->fRadii[i].fX = fRadii[i].fX;
            dst->fRadii[i].fY = fRadii[i].fY;
        }
    }

    const bool flipX = xScale < 0;
    if (flipX) {
        xScale = -xScale;
    }

    const bool flipY = yScale < 0;
    if (flipY) {
        yScale = -yScale;
    }

    // Scale the radii without respecting the flip.
    for (int i = 0; i < 4; ++i) {
        dst->fRadii[i].fX *= xScale;
        dst->fRadii[i].fY *= yScale;
    }

    // Now swap as necessary.
    using std::swap;
    if (flipX) {
        if (flipY) {
            // Swap with opposite corners
            swap(dst->fRadii[kUpperLeft_Corner], dst->fRadii[kLowerRight_Corner]);
            swap(dst->fRadii[kUpperRight_Corner], dst->fRadii[kLowerLeft_Corner]);
        } else {
            // Only swap in x
            swap(dst->fRadii[kUpperRight_Corner], dst->fRadii[kUpperLeft_Corner]);
            swap(dst->fRadii[kLowerRight_Corner], dst->fRadii[kLowerLeft_Corner]);
        }
    } else if (flipY) {
        // Only swap in y
        swap(dst->fRadii[kUpperLeft_Corner], dst->fRadii[kLowerLeft_Corner]);
        swap(dst->fRadii[kUpperRight_Corner], dst->fRadii[kLowerRight_Corner]);
    }

    if (!AreRectAndRadiiValid(dst->fRect, dst->fRadii)) {
        return false;
    }

    dst->scaleRadii();
    dst->isValid();  // TODO: is this meant to be SkASSERT(dst->isValid())?

    return true;
}

///////////////////////////////////////////////////////////////////////////////

void SkRRect::inset(SkScalar dx, SkScalar dy, SkRRect* dst) const {
    SkRect r = fRect.makeInset(dx, dy);
    bool degenerate = false;
    if (r.fRight <= r.fLeft) {
        degenerate = true;
        r.fLeft = r.fRight = SkScalarAve(r.fLeft, r.fRight);
    }
    if (r.fBottom <= r.fTop) {
        degenerate = true;
        r.fTop = r.fBottom = SkScalarAve(r.fTop, r.fBottom);
    }
    if (degenerate) {
        dst->fRect = r;
        memset(dst->fRadii, 0, sizeof(dst->fRadii));
        dst->fType = kEmpty_Type;
        return;
    }
    if (!r.isFinite()) {
        *dst = SkRRect();
        return;
    }

    SkVector radii[4];
    memcpy(radii, fRadii, sizeof(radii));
    for (int i = 0; i < 4; ++i) {
        if (radii[i].fX) {
            radii[i].fX -= dx;
        }
        if (radii[i].fY) {
            radii[i].fY -= dy;
        }
    }
    dst->setRectRadii(r, radii);
}

///////////////////////////////////////////////////////////////////////////////

// -- GODOT start --
/*
size_t SkRRect::writeToMemory(void* buffer) const {
    // Serialize only the rect and corners, but not the derived type tag.
    memcpy(buffer, this, kSizeInMemory);
    return kSizeInMemory;
}

void SkRRectPriv::WriteToBuffer(const SkRRect& rr, SkWBuffer* buffer) {
    // Serialize only the rect and corners, but not the derived type tag.
    buffer->write(&rr, SkRRect::kSizeInMemory);
}

size_t SkRRect::readFromMemory(const void* buffer, size_t length) {
    if (length < kSizeInMemory) {
        return 0;
    }

    // The extra (void*) tells GCC not to worry that kSizeInMemory < sizeof(SkRRect).

    SkRRect raw;
    memcpy((void*)&raw, buffer, kSizeInMemory);
    this->setRectRadii(raw.fRect, raw.fRadii);
    return kSizeInMemory;
}

bool SkRRectPriv::ReadFromBuffer(SkRBuffer* buffer, SkRRect* rr) {
    if (buffer->available() < SkRRect::kSizeInMemory) {
        return false;
    }
    SkRRect storage;
    return buffer->read(&storage, SkRRect::kSizeInMemory) &&
           (rr->readFromMemory(&storage, SkRRect::kSizeInMemory) == SkRRect::kSizeInMemory);
}

#include "include/core/SkString.h"
#include "src/core/SkStringUtils.h"

SkString SkRRect::dumpToString(bool asHex) const {
    SkScalarAsStringType asType = asHex ? kHex_SkScalarAsStringType : kDec_SkScalarAsStringType;

    fRect.dump(asHex);
    SkString line("const SkPoint corners[] = {\n");
    for (int i = 0; i < 4; ++i) {
        SkString strX, strY;
        SkAppendScalar(&strX, fRadii[i].x(), asType);
        SkAppendScalar(&strY, fRadii[i].y(), asType);
        line.appendf("    { %s, %s },", strX.c_str(), strY.c_str());
        if (asHex) {
            line.appendf(" %f %f ", fRadii[i].x(), fRadii[i].y());
        }
        line.append("\n");
    }
    line.append("};");
    return line;
}

void SkRRect::dump(bool asHex) const { SkDebugf("%s\n", this->dumpToString(asHex).c_str()); }
*/
// -- GODOT end --

///////////////////////////////////////////////////////////////////////////////

/**
 *  We need all combinations of predicates to be true to have a "safe" radius value.
 */
static bool are_radius_check_predicates_valid(SkScalar rad, SkScalar min, SkScalar max) {
    return (min <= max) && (rad <= max - min) && (min + rad <= max) && (max - rad >= min) &&
           rad >= 0;
}

bool SkRRect::isValid() const {
    if (!AreRectAndRadiiValid(fRect, fRadii)) {
        return false;
    }

    bool allRadiiZero = (0 == fRadii[0].fX && 0 == fRadii[0].fY);
    bool allCornersSquare = (0 == fRadii[0].fX || 0 == fRadii[0].fY);
    bool allRadiiSame = true;

    for (int i = 1; i < 4; ++i) {
        if (0 != fRadii[i].fX || 0 != fRadii[i].fY) {
            allRadiiZero = false;
        }

        if (fRadii[i].fX != fRadii[i-1].fX || fRadii[i].fY != fRadii[i-1].fY) {
            allRadiiSame = false;
        }

        if (0 != fRadii[i].fX && 0 != fRadii[i].fY) {
            allCornersSquare = false;
        }
    }
    bool patchesOfNine = radii_are_nine_patch(fRadii);

    if (fType < 0 || fType > kLastType) {
        return false;
    }

    switch (fType) {
        case kEmpty_Type:
            if (!fRect.isEmpty() || !allRadiiZero || !allRadiiSame || !allCornersSquare) {
                return false;
            }
            break;
        case kRect_Type:
            if (fRect.isEmpty() || !allRadiiZero || !allRadiiSame || !allCornersSquare) {
                return false;
            }
            break;
        case kOval_Type:
            if (fRect.isEmpty() || allRadiiZero || !allRadiiSame || allCornersSquare) {
                return false;
            }

            for (int i = 0; i < 4; ++i) {
                if (!SkScalarNearlyEqual(fRadii[i].fX, SkRectPriv::HalfWidth(fRect)) ||
                    !SkScalarNearlyEqual(fRadii[i].fY, SkRectPriv::HalfHeight(fRect))) {
                    return false;
                }
            }
            break;
        case kSimple_Type:
            if (fRect.isEmpty() || allRadiiZero || !allRadiiSame || allCornersSquare) {
                return false;
            }
            break;
        case kNinePatch_Type:
            if (fRect.isEmpty() || allRadiiZero || allRadiiSame || allCornersSquare ||
                !patchesOfNine) {
                return false;
            }
            break;
        case kComplex_Type:
            if (fRect.isEmpty() || allRadiiZero || allRadiiSame || allCornersSquare ||
                patchesOfNine) {
                return false;
            }
            break;
    }

    return true;
}

bool SkRRect::AreRectAndRadiiValid(const SkRect& rect, const SkVector radii[4]) {
    if (!rect.isFinite() || !rect.isSorted()) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (!are_radius_check_predicates_valid(radii[i].fX, rect.fLeft, rect.fRight) ||
            !are_radius_check_predicates_valid(radii[i].fY, rect.fTop, rect.fBottom)) {
            return false;
        }
    }
    return true;
}
///////////////////////////////////////////////////////////////////////////////

SkRect SkRRectPriv::InnerBounds(const SkRRect& rr) {
    if (rr.isEmpty() || rr.isRect()) {
        return rr.rect();
    }

    // We start with the outer bounds of the round rect and consider three subsets and take the
    // one with maximum area. The first two are the horizontal and vertical rects inset from the
    // corners, the third is the rect inscribed at the corner curves' maximal point. This forms
    // the exact solution when all corners have the same radii (the radii do not have to be
    // circular).
    SkRect innerBounds = rr.getBounds();
    SkVector tl = rr.radii(SkRRect::kUpperLeft_Corner);
    SkVector tr = rr.radii(SkRRect::kUpperRight_Corner);
    SkVector bl = rr.radii(SkRRect::kLowerLeft_Corner);
    SkVector br = rr.radii(SkRRect::kLowerRight_Corner);

    // Select maximum inset per edge, which may move an adjacent corner of the inscribed
    // rectangle off of the rounded-rect path, but that is acceptable given that the general
    // equation for inscribed area is non-trivial to evaluate.
    SkScalar leftShift   = std::max(tl.fX, bl.fX);
    SkScalar topShift    = std::max(tl.fY, tr.fY);
    SkScalar rightShift  = std::max(tr.fX, br.fX);
    SkScalar bottomShift = std::max(bl.fY, br.fY);

    SkScalar dw = leftShift + rightShift;
    SkScalar dh = topShift + bottomShift;

    // Area removed by shifting left/right
    SkScalar horizArea = (innerBounds.width() - dw) * innerBounds.height();
    // And by shifting top/bottom
    SkScalar vertArea = (innerBounds.height() - dh) * innerBounds.width();
    // And by shifting all edges: just considering a corner ellipse, the maximum inscribed rect has
    // a corner at sqrt(2)/2 * (rX, rY), so scale all corner shifts by (1 - sqrt(2)/2) to get the
    // safe shift per edge (since the shifts already are the max radius for that edge).
    // - We actually scale by a value slightly increased to make it so that the shifted corners are
    //   safely inside the curves, otherwise numerical stability can cause it to fail contains().
    static constexpr SkScalar kScale = (1.f - SK_ScalarRoot2Over2) + 1e-5f;
    SkScalar innerArea = (innerBounds.width() - kScale * dw) * (innerBounds.height() - kScale * dh);

    if (horizArea > vertArea && horizArea > innerArea) {
        // Cut off corners by insetting left and right
        innerBounds.fLeft += leftShift;
        innerBounds.fRight -= rightShift;
    } else if (vertArea > innerArea) {
        // Cut off corners by insetting top and bottom
        innerBounds.fTop += topShift;
        innerBounds.fBottom -= bottomShift;
    } else if (innerArea > 0.f) {
        // Inset on all sides, scaled to touch
        innerBounds.fLeft += kScale * leftShift;
        innerBounds.fRight -= kScale * rightShift;
        innerBounds.fTop += kScale * topShift;
        innerBounds.fBottom -= kScale * bottomShift;
    } else {
        // Inner region would collapse to empty
        return SkRect::MakeEmpty();
    }

    SkASSERT(innerBounds.isSorted() && !innerBounds.isEmpty());
    return innerBounds;
}

SkRRect SkRRectPriv::ConservativeIntersect(const SkRRect& a, const SkRRect& b) {
    // Returns the coordinate of the rect matching the corner enum.
    auto getCorner = [](const SkRect& r, SkRRect::Corner corner) -> SkPoint {
        switch(corner) {
            case SkRRect::kUpperLeft_Corner:  return {r.fLeft, r.fTop};
            case SkRRect::kUpperRight_Corner: return {r.fRight, r.fTop};
            case SkRRect::kLowerLeft_Corner:  return {r.fLeft, r.fBottom};
            case SkRRect::kLowerRight_Corner: return {r.fRight, r.fBottom};
            default: SkUNREACHABLE;
        }
    };
    // Returns true if shape A's extreme point is contained within shape B's extreme point, relative
    // to the 'corner' location. If the two shapes' corners have the same ellipse radii, this
    // is sufficient for A's ellipse arc to be contained by B's ellipse arc.
    auto insideCorner = [](SkRRect::Corner corner, const SkPoint& a, const SkPoint& b) {
        switch(corner) {
            case SkRRect::kUpperLeft_Corner:  return a.fX >= b.fX && a.fY >= b.fY;
            case SkRRect::kUpperRight_Corner: return a.fX <= b.fX && a.fY >= b.fY;
            case SkRRect::kLowerRight_Corner: return a.fX <= b.fX && a.fY <= b.fY;
            case SkRRect::kLowerLeft_Corner:  return a.fX >= b.fX && a.fY <= b.fY;
            default:  SkUNREACHABLE;
        }
    };

    auto getIntersectionRadii = [&](const SkRect& r, SkRRect::Corner corner, SkVector* radii) {
        SkPoint test = getCorner(r, corner);
        SkPoint aCorner = getCorner(a.rect(), corner);
        SkPoint bCorner = getCorner(b.rect(), corner);

        if (test == aCorner && test == bCorner) {
            // The round rects share a corner anchor, so pick A or B such that its X and Y radii
            // are both larger than the other rrect's, or return false if neither A or B has the max
            // corner radii (this is more permissive than the single corner tests below).
            SkVector aRadii = a.radii(corner);
            SkVector bRadii = b.radii(corner);
            if (aRadii.fX >= bRadii.fX && aRadii.fY >= bRadii.fY) {
                *radii = aRadii;
                return true;
            } else if (bRadii.fX >= aRadii.fX && bRadii.fY >= aRadii.fY) {
                *radii = bRadii;
                return true;
            } else {
                return false;
            }
        } else if (test == aCorner) {
            // Test that A's ellipse is contained by B. This is a non-trivial function to evaluate
            // so we resrict it to when the corners have the same radii. If not, we use the more
            // conservative test that the extreme point of A's bounding box is contained in B.
            *radii = a.radii(corner);
            if (*radii == b.radii(corner)) {
                return insideCorner(corner, aCorner, bCorner); // A inside B
            } else {
                return b.checkCornerContainment(aCorner.fX, aCorner.fY);
            }
        } else if (test == bCorner) {
            // Mirror of the above
            *radii = b.radii(corner);
            if (*radii == a.radii(corner)) {
                return insideCorner(corner, bCorner, aCorner); // B inside A
            } else {
                return a.checkCornerContainment(bCorner.fX, bCorner.fY);
            }
        } else {
            // This is a corner formed by two straight edges of A and B, so confirm that it is
            // contained in both (if not, then the intersection can't be a round rect).
            *radii = {0.f, 0.f};
            return a.checkCornerContainment(test.fX, test.fY) &&
                   b.checkCornerContainment(test.fX, test.fY);
        }
    };

    // We fill in the SkRRect directly. Since the rect and radii are either 0s or determined by
    // valid existing SkRRects, we know we are finite.
    SkRRect intersection;
    if (!intersection.fRect.intersect(a.rect(), b.rect())) {
        // Definitely no intersection
        return SkRRect::MakeEmpty();
    }

    const SkRRect::Corner corners[] = {
        SkRRect::kUpperLeft_Corner,
        SkRRect::kUpperRight_Corner,
        SkRRect::kLowerRight_Corner,
        SkRRect::kLowerLeft_Corner
    };
    // By definition, edges is contained in the bounds of 'a' and 'b', but now we need to consider
    // the corners. If the bound's corner point is in both rrects, the corner radii will be 0s.
    // If the bound's corner point matches a's edges and is inside 'b', we use a's radii.
    // Same for b's radii. If any corner fails these conditions, we reject the intersection as an
    // rrect. If after determining radii for all 4 corners, they would overlap, we also reject the
    // intersection shape.
    for (auto c : corners) {
        if (!getIntersectionRadii(intersection.fRect, c, &intersection.fRadii[c])) {
            return SkRRect::MakeEmpty(); // Resulting intersection is not a rrect
        }
    }

    // Check for radius overlap along the four edges, since the earlier evaluation was only a
    // one-sided corner check. If they aren't valid, a corner's radii doesn't fit within the rect.
    // If the radii are scaled, the combination of radii from two adjacent corners doesn't fit.
    // Normally for a regularly constructed SkRRect, we want this scaling, but in this case it means
    // the intersection shape is definitively not a round rect.
    if (!SkRRect::AreRectAndRadiiValid(intersection.fRect, intersection.fRadii) ||
        intersection.scaleRadii()) {
        return SkRRect::MakeEmpty();
    }

    // The intersection is an rrect of the given radii. Potentially all 4 corners could have
    // been simplified to (0,0) radii, making the intersection a rectangle.
    intersection.computeType();
    return intersection;
}
