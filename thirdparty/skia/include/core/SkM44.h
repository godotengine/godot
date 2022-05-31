/*
 * Copyright 2020 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkM44_DEFINED
#define SkM44_DEFINED

#include "include/core/SkMatrix.h"
#include "include/core/SkRect.h"
#include "include/core/SkScalar.h"

struct SK_API SkV2 {
    float x, y;

    bool operator==(const SkV2 v) const { return x == v.x && y == v.y; }
    bool operator!=(const SkV2 v) const { return !(*this == v); }

    static SkScalar   Dot(SkV2 a, SkV2 b) { return a.x * b.x + a.y * b.y; }
    static SkScalar Cross(SkV2 a, SkV2 b) { return a.x * b.y - a.y * b.x; }
    static SkV2 Normalize(SkV2 v) { return v * (1.0f / v.length()); }

    SkV2 operator-() const { return {-x, -y}; }
    SkV2 operator+(SkV2 v) const { return {x+v.x, y+v.y}; }
    SkV2 operator-(SkV2 v) const { return {x-v.x, y-v.y}; }

    SkV2 operator*(SkV2 v) const { return {x*v.x, y*v.y}; }
    friend SkV2 operator*(SkV2 v, SkScalar s) { return {v.x*s, v.y*s}; }
    friend SkV2 operator*(SkScalar s, SkV2 v) { return {v.x*s, v.y*s}; }
    friend SkV2 operator/(SkV2 v, SkScalar s) { return {v.x/s, v.y/s}; }

    void operator+=(SkV2 v) { *this = *this + v; }
    void operator-=(SkV2 v) { *this = *this - v; }
    void operator*=(SkV2 v) { *this = *this * v; }
    void operator*=(SkScalar s) { *this = *this * s; }
    void operator/=(SkScalar s) { *this = *this / s; }

    SkScalar lengthSquared() const { return Dot(*this, *this); }
    SkScalar length() const { return SkScalarSqrt(this->lengthSquared()); }

    SkScalar   dot(SkV2 v) const { return Dot(*this, v); }
    SkScalar cross(SkV2 v) const { return Cross(*this, v); }
    SkV2 normalize()       const { return Normalize(*this); }

    const float* ptr() const { return &x; }
    float* ptr() { return &x; }
};

struct SK_API SkV3 {
    float x, y, z;

    bool operator==(const SkV3& v) const {
        return x == v.x && y == v.y && z == v.z;
    }
    bool operator!=(const SkV3& v) const { return !(*this == v); }

    static SkScalar Dot(const SkV3& a, const SkV3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
    static SkV3   Cross(const SkV3& a, const SkV3& b) {
        return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
    }
    static SkV3 Normalize(const SkV3& v) { return v * (1.0f / v.length()); }

    SkV3 operator-() const { return {-x, -y, -z}; }
    SkV3 operator+(const SkV3& v) const { return { x + v.x, y + v.y, z + v.z }; }
    SkV3 operator-(const SkV3& v) const { return { x - v.x, y - v.y, z - v.z }; }

    SkV3 operator*(const SkV3& v) const {
        return { x*v.x, y*v.y, z*v.z };
    }
    friend SkV3 operator*(const SkV3& v, SkScalar s) {
        return { v.x*s, v.y*s, v.z*s };
    }
    friend SkV3 operator*(SkScalar s, const SkV3& v) { return v*s; }

    void operator+=(SkV3 v) { *this = *this + v; }
    void operator-=(SkV3 v) { *this = *this - v; }
    void operator*=(SkV3 v) { *this = *this * v; }
    void operator*=(SkScalar s) { *this = *this * s; }

    SkScalar lengthSquared() const { return Dot(*this, *this); }
    SkScalar length() const { return SkScalarSqrt(Dot(*this, *this)); }

    SkScalar dot(const SkV3& v) const { return Dot(*this, v); }
    SkV3   cross(const SkV3& v) const { return Cross(*this, v); }
    SkV3 normalize()            const { return Normalize(*this); }

    const float* ptr() const { return &x; }
    float* ptr() { return &x; }
};

struct SK_API SkV4 {
    float x, y, z, w;

    bool operator==(const SkV4& v) const {
        return x == v.x && y == v.y && z == v.z && w == v.w;
    }
    bool operator!=(const SkV4& v) const { return !(*this == v); }

    SkV4 operator-() const { return {-x, -y, -z, -w}; }
    SkV4 operator+(const SkV4& v) const { return { x + v.x, y + v.y, z + v.z, w + v.w }; }
    SkV4 operator-(const SkV4& v) const { return { x - v.x, y - v.y, z - v.z, w - v.w }; }

    SkV4 operator*(const SkV4& v) const {
        return { x*v.x, y*v.y, z*v.z, w*v.w };
    }
    friend SkV4 operator*(const SkV4& v, SkScalar s) {
        return { v.x*s, v.y*s, v.z*s, v.w*s };
    }
    friend SkV4 operator*(SkScalar s, const SkV4& v) { return v*s; }

    const float* ptr() const { return &x; }
    float* ptr() { return &x; }

    float operator[](int i) const {
        SkASSERT(i >= 0 && i < 4);
        return this->ptr()[i];
    }
    float& operator[](int i) {
        SkASSERT(i >= 0 && i < 4);
        return this->ptr()[i];
    }
};

/**
 *  4x4 matrix used by SkCanvas and other parts of Skia.
 *
 *  Skia assumes a right-handed coordinate system:
 *      +X goes to the right
 *      +Y goes down
 *      +Z goes into the screen (away from the viewer)
 */
class SK_API SkM44 {
public:
    SkM44(const SkM44& src) = default;
    SkM44& operator=(const SkM44& src) = default;

    constexpr SkM44()
        : fMat{1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1}
        {}

    SkM44(const SkM44& a, const SkM44& b) {
        this->setConcat(a, b);
    }

    enum Uninitialized_Constructor {
        kUninitialized_Constructor
    };
    SkM44(Uninitialized_Constructor) {}

    enum NaN_Constructor {
        kNaN_Constructor
    };
    constexpr SkM44(NaN_Constructor)
        : fMat{SK_ScalarNaN, SK_ScalarNaN, SK_ScalarNaN, SK_ScalarNaN,
               SK_ScalarNaN, SK_ScalarNaN, SK_ScalarNaN, SK_ScalarNaN,
               SK_ScalarNaN, SK_ScalarNaN, SK_ScalarNaN, SK_ScalarNaN,
               SK_ScalarNaN, SK_ScalarNaN, SK_ScalarNaN, SK_ScalarNaN}
    {}

    /**
     *  The constructor parameters are in row-major order.
     */
    constexpr SkM44(SkScalar m0, SkScalar m4, SkScalar m8,  SkScalar m12,
                    SkScalar m1, SkScalar m5, SkScalar m9,  SkScalar m13,
                    SkScalar m2, SkScalar m6, SkScalar m10, SkScalar m14,
                    SkScalar m3, SkScalar m7, SkScalar m11, SkScalar m15)
        // fMat is column-major order in memory.
        : fMat{m0,  m1,  m2,  m3,
               m4,  m5,  m6,  m7,
               m8,  m9,  m10, m11,
               m12, m13, m14, m15}
    {}

    static SkM44 Rows(const SkV4& r0, const SkV4& r1, const SkV4& r2, const SkV4& r3) {
        SkM44 m(kUninitialized_Constructor);
        m.setRow(0, r0);
        m.setRow(1, r1);
        m.setRow(2, r2);
        m.setRow(3, r3);
        return m;
    }
    static SkM44 Cols(const SkV4& c0, const SkV4& c1, const SkV4& c2, const SkV4& c3) {
        SkM44 m(kUninitialized_Constructor);
        m.setCol(0, c0);
        m.setCol(1, c1);
        m.setCol(2, c2);
        m.setCol(3, c3);
        return m;
    }

    static SkM44 RowMajor(const SkScalar r[16]) {
        return SkM44(r[ 0], r[ 1], r[ 2], r[ 3],
                     r[ 4], r[ 5], r[ 6], r[ 7],
                     r[ 8], r[ 9], r[10], r[11],
                     r[12], r[13], r[14], r[15]);
    }
    static SkM44 ColMajor(const SkScalar c[16]) {
        return SkM44(c[0], c[4], c[ 8], c[12],
                     c[1], c[5], c[ 9], c[13],
                     c[2], c[6], c[10], c[14],
                     c[3], c[7], c[11], c[15]);
    }

    static SkM44 Translate(SkScalar x, SkScalar y, SkScalar z = 0) {
        return SkM44(1, 0, 0, x,
                     0, 1, 0, y,
                     0, 0, 1, z,
                     0, 0, 0, 1);
    }

    static SkM44 Scale(SkScalar x, SkScalar y, SkScalar z = 1) {
        return SkM44(x, 0, 0, 0,
                     0, y, 0, 0,
                     0, 0, z, 0,
                     0, 0, 0, 1);
    }

    static SkM44 Rotate(SkV3 axis, SkScalar radians) {
        SkM44 m(kUninitialized_Constructor);
        m.setRotate(axis, radians);
        return m;
    }

    // Scales and translates 'src' to fill 'dst' exactly.
    static SkM44 RectToRect(const SkRect& src, const SkRect& dst);

    static SkM44 LookAt(const SkV3& eye, const SkV3& center, const SkV3& up);
    static SkM44 Perspective(float near, float far, float angle);

    bool operator==(const SkM44& other) const;
    bool operator!=(const SkM44& other) const {
        return !(other == *this);
    }

    void getColMajor(SkScalar v[]) const {
        memcpy(v, fMat, sizeof(fMat));
    }
    void getRowMajor(SkScalar v[]) const;

    SkScalar rc(int r, int c) const {
        SkASSERT(r >= 0 && r <= 3);
        SkASSERT(c >= 0 && c <= 3);
        return fMat[c*4 + r];
    }
    void setRC(int r, int c, SkScalar value) {
        SkASSERT(r >= 0 && r <= 3);
        SkASSERT(c >= 0 && c <= 3);
        fMat[c*4 + r] = value;
    }

    SkV4 row(int i) const {
        SkASSERT(i >= 0 && i <= 3);
        return {fMat[i + 0], fMat[i + 4], fMat[i + 8], fMat[i + 12]};
    }
    SkV4 col(int i) const {
        SkASSERT(i >= 0 && i <= 3);
        return {fMat[i*4 + 0], fMat[i*4 + 1], fMat[i*4 + 2], fMat[i*4 + 3]};
    }

    void setRow(int i, const SkV4& v) {
        SkASSERT(i >= 0 && i <= 3);
        fMat[i + 0]  = v.x;
        fMat[i + 4]  = v.y;
        fMat[i + 8]  = v.z;
        fMat[i + 12] = v.w;
    }
    void setCol(int i, const SkV4& v) {
        SkASSERT(i >= 0 && i <= 3);
        memcpy(&fMat[i*4], v.ptr(), sizeof(v));
    }

    SkM44& setIdentity() {
        *this = { 1, 0, 0, 0,
                  0, 1, 0, 0,
                  0, 0, 1, 0,
                  0, 0, 0, 1 };
        return *this;
    }

    SkM44& setTranslate(SkScalar x, SkScalar y, SkScalar z = 0) {
        *this = { 1, 0, 0, x,
                  0, 1, 0, y,
                  0, 0, 1, z,
                  0, 0, 0, 1 };
        return *this;
    }

    SkM44& setScale(SkScalar x, SkScalar y, SkScalar z = 1) {
        *this = { x, 0, 0, 0,
                  0, y, 0, 0,
                  0, 0, z, 0,
                  0, 0, 0, 1 };
        return *this;
    }

    /**
     *  Set this matrix to rotate about the specified unit-length axis vector,
     *  by an angle specified by its sin() and cos().
     *
     *  This does not attempt to verify that axis.length() == 1 or that the sin,cos values
     *  are correct.
     */
    SkM44& setRotateUnitSinCos(SkV3 axis, SkScalar sinAngle, SkScalar cosAngle);

    /**
     *  Set this matrix to rotate about the specified unit-length axis vector,
     *  by an angle specified in radians.
     *
     *  This does not attempt to verify that axis.length() == 1.
     */
    SkM44& setRotateUnit(SkV3 axis, SkScalar radians) {
        return this->setRotateUnitSinCos(axis, SkScalarSin(radians), SkScalarCos(radians));
    }

    /**
     *  Set this matrix to rotate about the specified axis vector,
     *  by an angle specified in radians.
     *
     *  Note: axis is not assumed to be unit-length, so it will be normalized internally.
     *        If axis is already unit-length, call setRotateAboutUnitRadians() instead.
     */
    SkM44& setRotate(SkV3 axis, SkScalar radians);

    SkM44& setConcat(const SkM44& a, const SkM44& b);

    friend SkM44 operator*(const SkM44& a, const SkM44& b) {
        return SkM44(a, b);
    }

    SkM44& preConcat(const SkM44& m) {
        return this->setConcat(*this, m);
    }

    SkM44& postConcat(const SkM44& m) {
        return this->setConcat(m, *this);
    }

    /**
     *  A matrix is categorized as 'perspective' if the bottom row is not [0, 0, 0, 1].
     *  For most uses, a bottom row of [0, 0, 0, X] behaves like a non-perspective matrix, though
     *  it will be categorized as perspective. Calling normalizePerspective() will change the
     *  matrix such that, if its bottom row was [0, 0, 0, X], it will be changed to [0, 0, 0, 1]
     *  by scaling the rest of the matrix by 1/X.
     *
     *  | A B C D |    | A/X B/X C/X D/X |
     *  | E F G H | -> | E/X F/X G/X H/X |   for X != 0
     *  | I J K L |    | I/X J/X K/X L/X |
     *  | 0 0 0 X |    |  0   0   0   1  |
     */
    void normalizePerspective();

    /** Returns true if all elements of the matrix are finite. Returns false if any
        element is infinity, or NaN.

        @return  true if matrix has only finite elements
    */
    bool isFinite() const { return SkScalarsAreFinite(fMat, 16); }

    /** If this is invertible, return that in inverse and return true. If it is
     *  not invertible, return false and leave the inverse parameter unchanged.
     */
    bool SK_WARN_UNUSED_RESULT invert(SkM44* inverse) const;

    SkM44 SK_WARN_UNUSED_RESULT transpose() const;

    void dump() const;

    ////////////

    SkV4 map(float x, float y, float z, float w) const;
    SkV4 operator*(const SkV4& v) const {
        return this->map(v.x, v.y, v.z, v.w);
    }
    SkV3 operator*(SkV3 v) const {
        auto v4 = this->map(v.x, v.y, v.z, 0);
        return {v4.x, v4.y, v4.z};
    }
    ////////////////////// Converting to/from SkMatrix

    /* When converting from SkM44 to SkMatrix, the third row and
     * column is dropped.  When converting from SkMatrix to SkM44
     * the third row and column remain as identity:
     * [ a b c ]      [ a b 0 c ]
     * [ d e f ]  ->  [ d e 0 f ]
     * [ g h i ]      [ 0 0 1 0 ]
     *                [ g h 0 i ]
     */
    SkMatrix asM33() const {
        return SkMatrix::MakeAll(fMat[0], fMat[4], fMat[12],
                                 fMat[1], fMat[5], fMat[13],
                                 fMat[3], fMat[7], fMat[15]);
    }

    explicit SkM44(const SkMatrix& src)
    : SkM44(src[SkMatrix::kMScaleX], src[SkMatrix::kMSkewX],  0, src[SkMatrix::kMTransX],
            src[SkMatrix::kMSkewY],  src[SkMatrix::kMScaleY], 0, src[SkMatrix::kMTransY],
            0,                       0,                       1, 0,
            src[SkMatrix::kMPersp0], src[SkMatrix::kMPersp1], 0, src[SkMatrix::kMPersp2])
    {}

    SkM44& preTranslate(SkScalar x, SkScalar y, SkScalar z = 0);
    SkM44& postTranslate(SkScalar x, SkScalar y, SkScalar z = 0);

    SkM44& preScale(SkScalar x, SkScalar y);
    SkM44& preScale(SkScalar x, SkScalar y, SkScalar z);
    SkM44& preConcat(const SkMatrix&);

private:
    /* Stored in column-major.
     *  Indices
     *  0  4  8  12        1 0 0 trans_x
     *  1  5  9  13  e.g.  0 1 0 trans_y
     *  2  6 10  14        0 0 1 trans_z
     *  3  7 11  15        0 0 0 1
     */
    SkScalar fMat[16];

    friend class SkMatrixPriv;
};

#endif
