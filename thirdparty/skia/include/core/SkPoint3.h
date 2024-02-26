/*
 * Copyright 2015 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkPoint3_DEFINED
#define SkPoint3_DEFINED

#include "include/core/SkPoint.h"

struct SK_API SkPoint3 {
    SkScalar fX, fY, fZ;

    static SkPoint3 Make(SkScalar x, SkScalar y, SkScalar z) {
        SkPoint3 pt;
        pt.set(x, y, z);
        return pt;
    }

    SkScalar x() const { return fX; }
    SkScalar y() const { return fY; }
    SkScalar z() const { return fZ; }

    void set(SkScalar x, SkScalar y, SkScalar z) { fX = x; fY = y; fZ = z; }

    friend bool operator==(const SkPoint3& a, const SkPoint3& b) {
        return a.fX == b.fX && a.fY == b.fY && a.fZ == b.fZ;
    }

    friend bool operator!=(const SkPoint3& a, const SkPoint3& b) {
        return !(a == b);
    }

    /** Returns the Euclidian distance from (0,0,0) to (x,y,z)
    */
    static SkScalar Length(SkScalar x, SkScalar y, SkScalar z);

    /** Return the Euclidian distance from (0,0,0) to the point
    */
    SkScalar length() const { return SkPoint3::Length(fX, fY, fZ); }

    /** Set the point (vector) to be unit-length in the same direction as it
        already points.  If the point has a degenerate length (i.e., nearly 0)
        then set it to (0,0,0) and return false; otherwise return true.
    */
    bool normalize();

    /** Return a new point whose X, Y and Z coordinates are scaled.
    */
    SkPoint3 makeScale(SkScalar scale) const {
        SkPoint3 p;
        p.set(scale * fX, scale * fY, scale * fZ);
        return p;
    }

    /** Scale the point's coordinates by scale.
    */
    void scale(SkScalar value) {
        fX *= value;
        fY *= value;
        fZ *= value;
    }

    /** Return a new point whose X, Y and Z coordinates are the negative of the
        original point's
    */
    SkPoint3 operator-() const {
        SkPoint3 neg;
        neg.fX = -fX;
        neg.fY = -fY;
        neg.fZ = -fZ;
        return neg;
    }

    /** Returns a new point whose coordinates are the difference between
        a and b (i.e., a - b)
    */
    friend SkPoint3 operator-(const SkPoint3& a, const SkPoint3& b) {
        return { a.fX - b.fX, a.fY - b.fY, a.fZ - b.fZ };
    }

    /** Returns a new point whose coordinates are the sum of a and b (a + b)
    */
    friend SkPoint3 operator+(const SkPoint3& a, const SkPoint3& b) {
        return { a.fX + b.fX, a.fY + b.fY, a.fZ + b.fZ };
    }

    /** Add v's coordinates to the point's
    */
    void operator+=(const SkPoint3& v) {
        fX += v.fX;
        fY += v.fY;
        fZ += v.fZ;
    }

    /** Subtract v's coordinates from the point's
    */
    void operator-=(const SkPoint3& v) {
        fX -= v.fX;
        fY -= v.fY;
        fZ -= v.fZ;
    }

    friend SkPoint3 operator*(SkScalar t, SkPoint3 p) {
        return { t * p.fX, t * p.fY, t * p.fZ };
    }

    /** Returns true if fX, fY, and fZ are measurable values.

     @return  true for values other than infinities and NaN
     */
    bool isFinite() const {
        SkScalar accum = 0;
        accum *= fX;
        accum *= fY;
        accum *= fZ;

        // accum is either NaN or it is finite (zero).
        SkASSERT(0 == accum || SkScalarIsNaN(accum));

        // value==value will be true iff value is not NaN
        // TODO: is it faster to say !accum or accum==accum?
        return !SkScalarIsNaN(accum);
    }

    /** Returns the dot product of a and b, treating them as 3D vectors
    */
    static SkScalar DotProduct(const SkPoint3& a, const SkPoint3& b) {
        return a.fX * b.fX + a.fY * b.fY + a.fZ * b.fZ;
    }

    SkScalar dot(const SkPoint3& vec) const {
        return DotProduct(*this, vec);
    }

    /** Returns the cross product of a and b, treating them as 3D vectors
    */
    static SkPoint3 CrossProduct(const SkPoint3& a, const SkPoint3& b) {
        SkPoint3 result;
        result.fX = a.fY*b.fZ - a.fZ*b.fY;
        result.fY = a.fZ*b.fX - a.fX*b.fZ;
        result.fZ = a.fX*b.fY - a.fY*b.fX;

        return result;
    }

    SkPoint3 cross(const SkPoint3& vec) const {
        return CrossProduct(*this, vec);
    }
};

typedef SkPoint3 SkVector3;
typedef SkPoint3 SkColor3f;

#endif
