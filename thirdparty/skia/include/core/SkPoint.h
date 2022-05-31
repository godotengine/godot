/*
 * Copyright 2006 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkPoint_DEFINED
#define SkPoint_DEFINED

#include "include/core/SkMath.h"
#include "include/core/SkScalar.h"
#include "include/private/SkSafe32.h"

struct SkIPoint;

/** SkIVector provides an alternative name for SkIPoint. SkIVector and SkIPoint
    can be used interchangeably for all purposes.
*/
typedef SkIPoint SkIVector;

/** \struct SkIPoint
    SkIPoint holds two 32-bit integer coordinates.
*/
struct SkIPoint {
    int32_t fX; //!< x-axis value
    int32_t fY; //!< y-axis value

    /** Sets fX to x, fY to y.

        @param x  integer x-axis value of constructed SkIPoint
        @param y  integer y-axis value of constructed SkIPoint
        @return   SkIPoint (x, y)
    */
    static constexpr SkIPoint Make(int32_t x, int32_t y) {
        return {x, y};
    }

    /** Returns x-axis value of SkIPoint.

        @return  fX
    */
    constexpr int32_t x() const { return fX; }

    /** Returns y-axis value of SkIPoint.

        @return  fY
    */
    constexpr int32_t y() const { return fY; }

    /** Returns true if fX and fY are both zero.

        @return  true if fX is zero and fY is zero
    */
    bool isZero() const { return (fX | fY) == 0; }

    /** Sets fX to x and fY to y.

        @param x  new value for fX
        @param y  new value for fY
    */
    void set(int32_t x, int32_t y) {
        fX = x;
        fY = y;
    }

    /** Returns SkIPoint changing the signs of fX and fY.

        @return  SkIPoint as (-fX, -fY)
    */
    SkIPoint operator-() const {
        return {-fX, -fY};
    }

    /** Offsets SkIPoint by ivector v. Sets SkIPoint to (fX + v.fX, fY + v.fY).

        @param v  ivector to add
    */
    void operator+=(const SkIVector& v) {
        fX = Sk32_sat_add(fX, v.fX);
        fY = Sk32_sat_add(fY, v.fY);
    }

    /** Subtracts ivector v from SkIPoint. Sets SkIPoint to: (fX - v.fX, fY - v.fY).

        @param v  ivector to subtract
    */
    void operator-=(const SkIVector& v) {
        fX = Sk32_sat_sub(fX, v.fX);
        fY = Sk32_sat_sub(fY, v.fY);
    }

    /** Returns true if SkIPoint is equivalent to SkIPoint constructed from (x, y).

        @param x  value compared with fX
        @param y  value compared with fY
        @return   true if SkIPoint equals (x, y)
    */
    bool equals(int32_t x, int32_t y) const {
        return fX == x && fY == y;
    }

    /** Returns true if a is equivalent to b.

        @param a  SkIPoint to compare
        @param b  SkIPoint to compare
        @return   true if a.fX == b.fX and a.fY == b.fY
    */
    friend bool operator==(const SkIPoint& a, const SkIPoint& b) {
        return a.fX == b.fX && a.fY == b.fY;
    }

    /** Returns true if a is not equivalent to b.

        @param a  SkIPoint to compare
        @param b  SkIPoint to compare
        @return   true if a.fX != b.fX or a.fY != b.fY
    */
    friend bool operator!=(const SkIPoint& a, const SkIPoint& b) {
        return a.fX != b.fX || a.fY != b.fY;
    }

    /** Returns ivector from b to a; computed as (a.fX - b.fX, a.fY - b.fY).

        Can also be used to subtract ivector from ivector, returning ivector.

        @param a  SkIPoint or ivector to subtract from
        @param b  ivector to subtract
        @return   ivector from b to a
    */
    friend SkIVector operator-(const SkIPoint& a, const SkIPoint& b) {
        return { Sk32_sat_sub(a.fX, b.fX), Sk32_sat_sub(a.fY, b.fY) };
    }

    /** Returns SkIPoint resulting from SkIPoint a offset by ivector b, computed as:
        (a.fX + b.fX, a.fY + b.fY).

        Can also be used to offset SkIPoint b by ivector a, returning SkIPoint.
        Can also be used to add ivector to ivector, returning ivector.

        @param a  SkIPoint or ivector to add to
        @param b  SkIPoint or ivector to add
        @return   SkIPoint equal to a offset by b
    */
    friend SkIPoint operator+(const SkIPoint& a, const SkIVector& b) {
        return { Sk32_sat_add(a.fX, b.fX), Sk32_sat_add(a.fY, b.fY) };
    }
};

struct SkPoint;

/** SkVector provides an alternative name for SkPoint. SkVector and SkPoint can
    be used interchangeably for all purposes.
*/
typedef SkPoint SkVector;

/** \struct SkPoint
    SkPoint holds two 32-bit floating point coordinates.
*/
struct SK_API SkPoint {
    SkScalar fX; //!< x-axis value
    SkScalar fY; //!< y-axis value

    /** Sets fX to x, fY to y. Used both to set SkPoint and vector.

        @param x  SkScalar x-axis value of constructed SkPoint or vector
        @param y  SkScalar y-axis value of constructed SkPoint or vector
        @return   SkPoint (x, y)
    */
    static constexpr SkPoint Make(SkScalar x, SkScalar y) {
        return {x, y};
    }

    /** Returns x-axis value of SkPoint or vector.

        @return  fX
    */
    constexpr SkScalar x() const { return fX; }

    /** Returns y-axis value of SkPoint or vector.

        @return  fY
    */
    constexpr SkScalar y() const { return fY; }

    /** Returns true if fX and fY are both zero.

        @return  true if fX is zero and fY is zero
    */
    bool isZero() const { return (0 == fX) & (0 == fY); }

    /** Sets fX to x and fY to y.

        @param x  new value for fX
        @param y  new value for fY
    */
    void set(SkScalar x, SkScalar y) {
        fX = x;
        fY = y;
    }

    /** Sets fX to x and fY to y, promoting integers to SkScalar values.

        Assigning a large integer value directly to fX or fY may cause a compiler
        error, triggered by narrowing conversion of int to SkScalar. This safely
        casts x and y to avoid the error.

        @param x  new value for fX
        @param y  new value for fY
    */
    void iset(int32_t x, int32_t y) {
        fX = SkIntToScalar(x);
        fY = SkIntToScalar(y);
    }

    /** Sets fX to p.fX and fY to p.fY, promoting integers to SkScalar values.

        Assigning an SkIPoint containing a large integer value directly to fX or fY may
        cause a compiler error, triggered by narrowing conversion of int to SkScalar.
        This safely casts p.fX and p.fY to avoid the error.

        @param p  SkIPoint members promoted to SkScalar
    */
    void iset(const SkIPoint& p) {
        fX = SkIntToScalar(p.fX);
        fY = SkIntToScalar(p.fY);
    }

    /** Sets fX to absolute value of pt.fX; and fY to absolute value of pt.fY.

        @param pt  members providing magnitude for fX and fY
    */
    void setAbs(const SkPoint& pt) {
        fX = SkScalarAbs(pt.fX);
        fY = SkScalarAbs(pt.fY);
    }

    /** Adds offset to each SkPoint in points array with count entries.

        @param points  SkPoint array
        @param count   entries in array
        @param offset  vector added to points
    */
    static void Offset(SkPoint points[], int count, const SkVector& offset) {
        Offset(points, count, offset.fX, offset.fY);
    }

    /** Adds offset (dx, dy) to each SkPoint in points array of length count.

        @param points  SkPoint array
        @param count   entries in array
        @param dx      added to fX in points
        @param dy      added to fY in points
    */
    static void Offset(SkPoint points[], int count, SkScalar dx, SkScalar dy) {
        for (int i = 0; i < count; ++i) {
            points[i].offset(dx, dy);
        }
    }

    /** Adds offset (dx, dy) to SkPoint.

        @param dx  added to fX
        @param dy  added to fY
    */
    void offset(SkScalar dx, SkScalar dy) {
        fX += dx;
        fY += dy;
    }

    /** Returns the Euclidean distance from origin, computed as:

            sqrt(fX * fX + fY * fY)

        .

        @return  straight-line distance to origin
    */
    SkScalar length() const { return SkPoint::Length(fX, fY); }

    /** Returns the Euclidean distance from origin, computed as:

            sqrt(fX * fX + fY * fY)

        .

        @return  straight-line distance to origin
    */
    SkScalar distanceToOrigin() const { return this->length(); }

    /** Scales (fX, fY) so that length() returns one, while preserving ratio of fX to fY,
        if possible. If prior length is nearly zero, sets vector to (0, 0) and returns
        false; otherwise returns true.

        @return  true if former length is not zero or nearly zero

        example: https://fiddle.skia.org/c/@Point_normalize_2
    */
    bool normalize();

    /** Sets vector to (x, y) scaled so length() returns one, and so that
        (fX, fY) is proportional to (x, y).  If (x, y) length is nearly zero,
        sets vector to (0, 0) and returns false; otherwise returns true.

        @param x  proportional value for fX
        @param y  proportional value for fY
        @return   true if (x, y) length is not zero or nearly zero

        example: https://fiddle.skia.org/c/@Point_setNormalize
    */
    bool setNormalize(SkScalar x, SkScalar y);

    /** Scales vector so that distanceToOrigin() returns length, if possible. If former
        length is nearly zero, sets vector to (0, 0) and return false; otherwise returns
        true.

        @param length  straight-line distance to origin
        @return        true if former length is not zero or nearly zero

        example: https://fiddle.skia.org/c/@Point_setLength
    */
    bool setLength(SkScalar length);

    /** Sets vector to (x, y) scaled to length, if possible. If former
        length is nearly zero, sets vector to (0, 0) and return false; otherwise returns
        true.

        @param x       proportional value for fX
        @param y       proportional value for fY
        @param length  straight-line distance to origin
        @return        true if (x, y) length is not zero or nearly zero

        example: https://fiddle.skia.org/c/@Point_setLength_2
    */
    bool setLength(SkScalar x, SkScalar y, SkScalar length);

    /** Sets dst to SkPoint times scale. dst may be SkPoint to modify SkPoint in place.

        @param scale  factor to multiply SkPoint by
        @param dst    storage for scaled SkPoint

        example: https://fiddle.skia.org/c/@Point_scale
    */
    void scale(SkScalar scale, SkPoint* dst) const;

    /** Scales SkPoint in place by scale.

        @param value  factor to multiply SkPoint by
    */
    void scale(SkScalar value) { this->scale(value, this); }

    /** Changes the sign of fX and fY.
    */
    void negate() {
        fX = -fX;
        fY = -fY;
    }

    /** Returns SkPoint changing the signs of fX and fY.

        @return  SkPoint as (-fX, -fY)
    */
    SkPoint operator-() const {
        return {-fX, -fY};
    }

    /** Adds vector v to SkPoint. Sets SkPoint to: (fX + v.fX, fY + v.fY).

        @param v  vector to add
    */
    void operator+=(const SkVector& v) {
        fX += v.fX;
        fY += v.fY;
    }

    /** Subtracts vector v from SkPoint. Sets SkPoint to: (fX - v.fX, fY - v.fY).

        @param v  vector to subtract
    */
    void operator-=(const SkVector& v) {
        fX -= v.fX;
        fY -= v.fY;
    }

    /** Returns SkPoint multiplied by scale.

        @param scale  scalar to multiply by
        @return       SkPoint as (fX * scale, fY * scale)
    */
    SkPoint operator*(SkScalar scale) const {
        return {fX * scale, fY * scale};
    }

    /** Multiplies SkPoint by scale. Sets SkPoint to: (fX * scale, fY * scale).

        @param scale  scalar to multiply by
        @return       reference to SkPoint
    */
    SkPoint& operator*=(SkScalar scale) {
        fX *= scale;
        fY *= scale;
        return *this;
    }

    /** Returns true if both fX and fY are measurable values.

        @return  true for values other than infinities and NaN
    */
    bool isFinite() const {
        SkScalar accum = 0;
        accum *= fX;
        accum *= fY;

        // accum is either NaN or it is finite (zero).
        SkASSERT(0 == accum || SkScalarIsNaN(accum));

        // value==value will be true iff value is not NaN
        // TODO: is it faster to say !accum or accum==accum?
        return !SkScalarIsNaN(accum);
    }

    /** Returns true if SkPoint is equivalent to SkPoint constructed from (x, y).

        @param x  value compared with fX
        @param y  value compared with fY
        @return   true if SkPoint equals (x, y)
    */
    bool equals(SkScalar x, SkScalar y) const {
        return fX == x && fY == y;
    }

    /** Returns true if a is equivalent to b.

        @param a  SkPoint to compare
        @param b  SkPoint to compare
        @return   true if a.fX == b.fX and a.fY == b.fY
    */
    friend bool operator==(const SkPoint& a, const SkPoint& b) {
        return a.fX == b.fX && a.fY == b.fY;
    }

    /** Returns true if a is not equivalent to b.

        @param a  SkPoint to compare
        @param b  SkPoint to compare
        @return   true if a.fX != b.fX or a.fY != b.fY
    */
    friend bool operator!=(const SkPoint& a, const SkPoint& b) {
        return a.fX != b.fX || a.fY != b.fY;
    }

    /** Returns vector from b to a, computed as (a.fX - b.fX, a.fY - b.fY).

        Can also be used to subtract vector from SkPoint, returning SkPoint.
        Can also be used to subtract vector from vector, returning vector.

        @param a  SkPoint to subtract from
        @param b  SkPoint to subtract
        @return   vector from b to a
    */
    friend SkVector operator-(const SkPoint& a, const SkPoint& b) {
        return {a.fX - b.fX, a.fY - b.fY};
    }

    /** Returns SkPoint resulting from SkPoint a offset by vector b, computed as:
        (a.fX + b.fX, a.fY + b.fY).

        Can also be used to offset SkPoint b by vector a, returning SkPoint.
        Can also be used to add vector to vector, returning vector.

        @param a  SkPoint or vector to add to
        @param b  SkPoint or vector to add
        @return   SkPoint equal to a offset by b
    */
    friend SkPoint operator+(const SkPoint& a, const SkVector& b) {
        return {a.fX + b.fX, a.fY + b.fY};
    }

    /** Returns the Euclidean distance from origin, computed as:

            sqrt(x * x + y * y)

        .

        @param x  component of length
        @param y  component of length
        @return   straight-line distance to origin

        example: https://fiddle.skia.org/c/@Point_Length
    */
    static SkScalar Length(SkScalar x, SkScalar y);

    /** Scales (vec->fX, vec->fY) so that length() returns one, while preserving ratio of vec->fX
        to vec->fY, if possible. If original length is nearly zero, sets vec to (0, 0) and returns
        zero; otherwise, returns length of vec before vec is scaled.

        Returned prior length may be SK_ScalarInfinity if it can not be represented by SkScalar.

        Note that normalize() is faster if prior length is not required.

        @param vec  normalized to unit length
        @return     original vec length

        example: https://fiddle.skia.org/c/@Point_Normalize
    */
    static SkScalar Normalize(SkVector* vec);

    /** Returns the Euclidean distance between a and b.

        @param a  line end point
        @param b  line end point
        @return   straight-line distance from a to b
    */
    static SkScalar Distance(const SkPoint& a, const SkPoint& b) {
        return Length(a.fX - b.fX, a.fY - b.fY);
    }

    /** Returns the dot product of vector a and vector b.

        @param a  left side of dot product
        @param b  right side of dot product
        @return   product of input magnitudes and cosine of the angle between them
    */
    static SkScalar DotProduct(const SkVector& a, const SkVector& b) {
        return a.fX * b.fX + a.fY * b.fY;
    }

    /** Returns the cross product of vector a and vector b.

        a and b form three-dimensional vectors with z-axis value equal to zero. The
        cross product is a three-dimensional vector with x-axis and y-axis values equal
        to zero. The cross product z-axis component is returned.

        @param a  left side of cross product
        @param b  right side of cross product
        @return   area spanned by vectors signed by angle direction
    */
    static SkScalar CrossProduct(const SkVector& a, const SkVector& b) {
        return a.fX * b.fY - a.fY * b.fX;
    }

    /** Returns the cross product of vector and vec.

        Vector and vec form three-dimensional vectors with z-axis value equal to zero.
        The cross product is a three-dimensional vector with x-axis and y-axis values
        equal to zero. The cross product z-axis component is returned.

        @param vec  right side of cross product
        @return     area spanned by vectors signed by angle direction
    */
    SkScalar cross(const SkVector& vec) const {
        return CrossProduct(*this, vec);
    }

    /** Returns the dot product of vector and vector vec.

        @param vec  right side of dot product
        @return     product of input magnitudes and cosine of the angle between them
    */
    SkScalar dot(const SkVector& vec) const {
        return DotProduct(*this, vec);
    }

};

#endif
