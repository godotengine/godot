/*
 * Copyright 2006 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkMatrix_DEFINED
#define SkMatrix_DEFINED

#include "include/core/SkRect.h"
#include "include/private/SkMacros.h"
#include "include/private/SkTo.h"

struct SkRSXform;
struct SkPoint3;

// Remove when clients are updated to live without this
#define SK_SUPPORT_LEGACY_MATRIX_RECTTORECT

/**
 *  When we transform points through a matrix containing perspective (the bottom row is something
 *  other than 0,0,1), the bruteforce math can produce confusing results (since we might divide
 *  by 0, or a negative w value). By default, methods that map rects and paths will apply
 *  perspective clipping, but this can be changed by specifying kYes to those methods.
 */
enum class SkApplyPerspectiveClip {
    kNo,    //!< Don't pre-clip the geometry before applying the (perspective) matrix
    kYes,   //!< Do pre-clip the geometry before applying the (perspective) matrix
};

/** \class SkMatrix
    SkMatrix holds a 3x3 matrix for transforming coordinates. This allows mapping
    SkPoint and vectors with translation, scaling, skewing, rotation, and
    perspective.

    SkMatrix elements are in row major order.
    SkMatrix constexpr default constructs to identity.

    SkMatrix includes a hidden variable that classifies the type of matrix to
    improve performance. SkMatrix is not thread safe unless getType() is called first.

    example: https://fiddle.skia.org/c/@Matrix_063
*/
SK_BEGIN_REQUIRE_DENSE
class SK_API SkMatrix {
public:

    /** Creates an identity SkMatrix:

            | 1 0 0 |
            | 0 1 0 |
            | 0 0 1 |
    */
    constexpr SkMatrix() : SkMatrix(1,0,0, 0,1,0, 0,0,1, kIdentity_Mask | kRectStaysRect_Mask) {}

    /** Sets SkMatrix to scale by (sx, sy). Returned matrix is:

            | sx  0  0 |
            |  0 sy  0 |
            |  0  0  1 |

        @param sx  horizontal scale factor
        @param sy  vertical scale factor
        @return    SkMatrix with scale
    */
    static SkMatrix SK_WARN_UNUSED_RESULT Scale(SkScalar sx, SkScalar sy) {
        SkMatrix m;
        m.setScale(sx, sy);
        return m;
    }

    /** Sets SkMatrix to translate by (dx, dy). Returned matrix is:

            | 1 0 dx |
            | 0 1 dy |
            | 0 0  1 |

        @param dx  horizontal translation
        @param dy  vertical translation
        @return    SkMatrix with translation
    */
    static SkMatrix SK_WARN_UNUSED_RESULT Translate(SkScalar dx, SkScalar dy) {
        SkMatrix m;
        m.setTranslate(dx, dy);
        return m;
    }
    static SkMatrix SK_WARN_UNUSED_RESULT Translate(SkVector t) { return Translate(t.x(), t.y()); }
    static SkMatrix SK_WARN_UNUSED_RESULT Translate(SkIVector t) { return Translate(t.x(), t.y()); }

    /** Sets SkMatrix to rotate by |deg| about a pivot point at (0, 0).

        @param deg  rotation angle in degrees (positive rotates clockwise)
        @return     SkMatrix with rotation
    */
    static SkMatrix SK_WARN_UNUSED_RESULT RotateDeg(SkScalar deg) {
        SkMatrix m;
        m.setRotate(deg);
        return m;
    }
    static SkMatrix SK_WARN_UNUSED_RESULT RotateDeg(SkScalar deg, SkPoint pt) {
        SkMatrix m;
        m.setRotate(deg, pt.x(), pt.y());
        return m;
    }
    static SkMatrix SK_WARN_UNUSED_RESULT RotateRad(SkScalar rad) {
        return RotateDeg(SkRadiansToDegrees(rad));
    }

    /** Sets SkMatrix to skew by (kx, ky) about pivot point (0, 0).

        @param kx  horizontal skew factor
        @param ky  vertical skew factor
        @return    SkMatrix with skew
    */
    static SkMatrix SK_WARN_UNUSED_RESULT Skew(SkScalar kx, SkScalar ky) {
        SkMatrix m;
        m.setSkew(kx, ky);
        return m;
    }

    /** \enum SkMatrix::ScaleToFit
        ScaleToFit describes how SkMatrix is constructed to map one SkRect to another.
        ScaleToFit may allow SkMatrix to have unequal horizontal and vertical scaling,
        or may restrict SkMatrix to square scaling. If restricted, ScaleToFit specifies
        how SkMatrix maps to the side or center of the destination SkRect.
    */
    enum ScaleToFit {
        kFill_ScaleToFit,   //!< scales in x and y to fill destination SkRect
        kStart_ScaleToFit,  //!< scales and aligns to left and top
        kCenter_ScaleToFit, //!< scales and aligns to center
        kEnd_ScaleToFit,    //!< scales and aligns to right and bottom
    };

    /** Returns SkMatrix set to scale and translate src to dst. ScaleToFit selects
        whether mapping completely fills dst or preserves the aspect ratio, and how to
        align src within dst. Returns the identity SkMatrix if src is empty. If dst is
        empty, returns SkMatrix set to:

            | 0 0 0 |
            | 0 0 0 |
            | 0 0 1 |

        @param src  SkRect to map from
        @param dst  SkRect to map to
        @param mode How to handle the mapping
        @return     SkMatrix mapping src to dst
    */
    static SkMatrix SK_WARN_UNUSED_RESULT RectToRect(const SkRect& src, const SkRect& dst,
                                                     ScaleToFit mode = kFill_ScaleToFit) {
        return MakeRectToRect(src, dst, mode);
    }

    /** Sets SkMatrix to:

            | scaleX  skewX transX |
            |  skewY scaleY transY |
            |  pers0  pers1  pers2 |

        @param scaleX  horizontal scale factor
        @param skewX   horizontal skew factor
        @param transX  horizontal translation
        @param skewY   vertical skew factor
        @param scaleY  vertical scale factor
        @param transY  vertical translation
        @param pers0   input x-axis perspective factor
        @param pers1   input y-axis perspective factor
        @param pers2   perspective scale factor
        @return        SkMatrix constructed from parameters
    */
    static SkMatrix SK_WARN_UNUSED_RESULT MakeAll(SkScalar scaleX, SkScalar skewX,  SkScalar transX,
                                                  SkScalar skewY,  SkScalar scaleY, SkScalar transY,
                                                  SkScalar pers0, SkScalar pers1, SkScalar pers2) {
        SkMatrix m;
        m.setAll(scaleX, skewX, transX, skewY, scaleY, transY, pers0, pers1, pers2);
        return m;
    }

    /** \enum SkMatrix::TypeMask
        Enum of bit fields for mask returned by getType().
        Used to identify the complexity of SkMatrix, to optimize performance.
    */
    enum TypeMask {
        kIdentity_Mask    = 0,    //!< identity SkMatrix; all bits clear
        kTranslate_Mask   = 0x01, //!< translation SkMatrix
        kScale_Mask       = 0x02, //!< scale SkMatrix
        kAffine_Mask      = 0x04, //!< skew or rotate SkMatrix
        kPerspective_Mask = 0x08, //!< perspective SkMatrix
    };

    /** Returns a bit field describing the transformations the matrix may
        perform. The bit field is computed conservatively, so it may include
        false positives. For example, when kPerspective_Mask is set, all
        other bits are set.

        @return  kIdentity_Mask, or combinations of: kTranslate_Mask, kScale_Mask,
                 kAffine_Mask, kPerspective_Mask
    */
    TypeMask getType() const {
        if (fTypeMask & kUnknown_Mask) {
            fTypeMask = this->computeTypeMask();
        }
        // only return the public masks
        return (TypeMask)(fTypeMask & 0xF);
    }

    /** Returns true if SkMatrix is identity.  Identity matrix is:

            | 1 0 0 |
            | 0 1 0 |
            | 0 0 1 |

        @return  true if SkMatrix has no effect
    */
    bool isIdentity() const {
        return this->getType() == 0;
    }

    /** Returns true if SkMatrix at most scales and translates. SkMatrix may be identity,
        contain only scale elements, only translate elements, or both. SkMatrix form is:

            | scale-x    0    translate-x |
            |    0    scale-y translate-y |
            |    0       0         1      |

        @return  true if SkMatrix is identity; or scales, translates, or both
    */
    bool isScaleTranslate() const {
        return !(this->getType() & ~(kScale_Mask | kTranslate_Mask));
    }

    /** Returns true if SkMatrix is identity, or translates. SkMatrix form is:

            | 1 0 translate-x |
            | 0 1 translate-y |
            | 0 0      1      |

        @return  true if SkMatrix is identity, or translates
    */
    bool isTranslate() const { return !(this->getType() & ~(kTranslate_Mask)); }

    /** Returns true SkMatrix maps SkRect to another SkRect. If true, SkMatrix is identity,
        or scales, or rotates a multiple of 90 degrees, or mirrors on axes. In all
        cases, SkMatrix may also have translation. SkMatrix form is either:

            | scale-x    0    translate-x |
            |    0    scale-y translate-y |
            |    0       0         1      |

        or

            |    0     rotate-x translate-x |
            | rotate-y    0     translate-y |
            |    0        0          1      |

        for non-zero values of scale-x, scale-y, rotate-x, and rotate-y.

        Also called preservesAxisAlignment(); use the one that provides better inline
        documentation.

        @return  true if SkMatrix maps one SkRect into another
    */
    bool rectStaysRect() const {
        if (fTypeMask & kUnknown_Mask) {
            fTypeMask = this->computeTypeMask();
        }
        return (fTypeMask & kRectStaysRect_Mask) != 0;
    }

    /** Returns true SkMatrix maps SkRect to another SkRect. If true, SkMatrix is identity,
        or scales, or rotates a multiple of 90 degrees, or mirrors on axes. In all
        cases, SkMatrix may also have translation. SkMatrix form is either:

            | scale-x    0    translate-x |
            |    0    scale-y translate-y |
            |    0       0         1      |

        or

            |    0     rotate-x translate-x |
            | rotate-y    0     translate-y |
            |    0        0          1      |

        for non-zero values of scale-x, scale-y, rotate-x, and rotate-y.

        Also called rectStaysRect(); use the one that provides better inline
        documentation.

        @return  true if SkMatrix maps one SkRect into another
    */
    bool preservesAxisAlignment() const { return this->rectStaysRect(); }

    /** Returns true if the matrix contains perspective elements. SkMatrix form is:

            |       --            --              --          |
            |       --            --              --          |
            | perspective-x  perspective-y  perspective-scale |

        where perspective-x or perspective-y is non-zero, or perspective-scale is
        not one. All other elements may have any value.

        @return  true if SkMatrix is in most general form
    */
    bool hasPerspective() const {
        return SkToBool(this->getPerspectiveTypeMaskOnly() &
                        kPerspective_Mask);
    }

    /** Returns true if SkMatrix contains only translation, rotation, reflection, and
        uniform scale.
        Returns false if SkMatrix contains different scales, skewing, perspective, or
        degenerate forms that collapse to a line or point.

        Describes that the SkMatrix makes rendering with and without the matrix are
        visually alike; a transformed circle remains a circle. Mathematically, this is
        referred to as similarity of a Euclidean space, or a similarity transformation.

        Preserves right angles, keeping the arms of the angle equal lengths.

        @param tol  to be deprecated
        @return     true if SkMatrix only rotates, uniformly scales, translates

        example: https://fiddle.skia.org/c/@Matrix_isSimilarity
    */
    bool isSimilarity(SkScalar tol = SK_ScalarNearlyZero) const;

    /** Returns true if SkMatrix contains only translation, rotation, reflection, and
        scale. Scale may differ along rotated axes.
        Returns false if SkMatrix skewing, perspective, or degenerate forms that collapse
        to a line or point.

        Preserves right angles, but not requiring that the arms of the angle
        retain equal lengths.

        @param tol  to be deprecated
        @return     true if SkMatrix only rotates, scales, translates

        example: https://fiddle.skia.org/c/@Matrix_preservesRightAngles
    */
    bool preservesRightAngles(SkScalar tol = SK_ScalarNearlyZero) const;

    /** SkMatrix organizes its values in row-major order. These members correspond to
        each value in SkMatrix.
    */
    static constexpr int kMScaleX = 0; //!< horizontal scale factor
    static constexpr int kMSkewX  = 1; //!< horizontal skew factor
    static constexpr int kMTransX = 2; //!< horizontal translation
    static constexpr int kMSkewY  = 3; //!< vertical skew factor
    static constexpr int kMScaleY = 4; //!< vertical scale factor
    static constexpr int kMTransY = 5; //!< vertical translation
    static constexpr int kMPersp0 = 6; //!< input x perspective factor
    static constexpr int kMPersp1 = 7; //!< input y perspective factor
    static constexpr int kMPersp2 = 8; //!< perspective bias

    /** Affine arrays are in column-major order to match the matrix used by
        PDF and XPS.
    */
    static constexpr int kAScaleX = 0; //!< horizontal scale factor
    static constexpr int kASkewY  = 1; //!< vertical skew factor
    static constexpr int kASkewX  = 2; //!< horizontal skew factor
    static constexpr int kAScaleY = 3; //!< vertical scale factor
    static constexpr int kATransX = 4; //!< horizontal translation
    static constexpr int kATransY = 5; //!< vertical translation

    /** Returns one matrix value. Asserts if index is out of range and SK_DEBUG is
        defined.

        @param index  one of: kMScaleX, kMSkewX, kMTransX, kMSkewY, kMScaleY, kMTransY,
                      kMPersp0, kMPersp1, kMPersp2
        @return       value corresponding to index
    */
    SkScalar operator[](int index) const {
        SkASSERT((unsigned)index < 9);
        return fMat[index];
    }

    /** Returns one matrix value. Asserts if index is out of range and SK_DEBUG is
        defined.

        @param index  one of: kMScaleX, kMSkewX, kMTransX, kMSkewY, kMScaleY, kMTransY,
                      kMPersp0, kMPersp1, kMPersp2
        @return       value corresponding to index
    */
    SkScalar get(int index) const {
        SkASSERT((unsigned)index < 9);
        return fMat[index];
    }

    /** Returns one matrix value from a particular row/column. Asserts if index is out
        of range and SK_DEBUG is defined.

        @param r  matrix row to fetch
        @param c  matrix column to fetch
        @return   value at the given matrix position
    */
    SkScalar rc(int r, int c) const {
        SkASSERT(r >= 0 && r <= 2);
        SkASSERT(c >= 0 && c <= 2);
        return fMat[r*3 + c];
    }

    /** Returns scale factor multiplied by x-axis input, contributing to x-axis output.
        With mapPoints(), scales SkPoint along the x-axis.

        @return  horizontal scale factor
    */
    SkScalar getScaleX() const { return fMat[kMScaleX]; }

    /** Returns scale factor multiplied by y-axis input, contributing to y-axis output.
        With mapPoints(), scales SkPoint along the y-axis.

        @return  vertical scale factor
    */
    SkScalar getScaleY() const { return fMat[kMScaleY]; }

    /** Returns scale factor multiplied by x-axis input, contributing to y-axis output.
        With mapPoints(), skews SkPoint along the y-axis.
        Skewing both axes can rotate SkPoint.

        @return  vertical skew factor
    */
    SkScalar getSkewY() const { return fMat[kMSkewY]; }

    /** Returns scale factor multiplied by y-axis input, contributing to x-axis output.
        With mapPoints(), skews SkPoint along the x-axis.
        Skewing both axes can rotate SkPoint.

        @return  horizontal scale factor
    */
    SkScalar getSkewX() const { return fMat[kMSkewX]; }

    /** Returns translation contributing to x-axis output.
        With mapPoints(), moves SkPoint along the x-axis.

        @return  horizontal translation factor
    */
    SkScalar getTranslateX() const { return fMat[kMTransX]; }

    /** Returns translation contributing to y-axis output.
        With mapPoints(), moves SkPoint along the y-axis.

        @return  vertical translation factor
    */
    SkScalar getTranslateY() const { return fMat[kMTransY]; }

    /** Returns factor scaling input x-axis relative to input y-axis.

        @return  input x-axis perspective factor
    */
    SkScalar getPerspX() const { return fMat[kMPersp0]; }

    /** Returns factor scaling input y-axis relative to input x-axis.

        @return  input y-axis perspective factor
    */
    SkScalar getPerspY() const { return fMat[kMPersp1]; }

    /** Returns writable SkMatrix value. Asserts if index is out of range and SK_DEBUG is
        defined. Clears internal cache anticipating that caller will change SkMatrix value.

        Next call to read SkMatrix state may recompute cache; subsequent writes to SkMatrix
        value must be followed by dirtyMatrixTypeCache().

        @param index  one of: kMScaleX, kMSkewX, kMTransX, kMSkewY, kMScaleY, kMTransY,
                      kMPersp0, kMPersp1, kMPersp2
        @return       writable value corresponding to index
    */
    SkScalar& operator[](int index) {
        SkASSERT((unsigned)index < 9);
        this->setTypeMask(kUnknown_Mask);
        return fMat[index];
    }

    /** Sets SkMatrix value. Asserts if index is out of range and SK_DEBUG is
        defined. Safer than operator[]; internal cache is always maintained.

        @param index  one of: kMScaleX, kMSkewX, kMTransX, kMSkewY, kMScaleY, kMTransY,
                      kMPersp0, kMPersp1, kMPersp2
        @param value  scalar to store in SkMatrix
    */
    SkMatrix& set(int index, SkScalar value) {
        SkASSERT((unsigned)index < 9);
        fMat[index] = value;
        this->setTypeMask(kUnknown_Mask);
        return *this;
    }

    /** Sets horizontal scale factor.

        @param v  horizontal scale factor to store
    */
    SkMatrix& setScaleX(SkScalar v) { return this->set(kMScaleX, v); }

    /** Sets vertical scale factor.

        @param v  vertical scale factor to store
    */
    SkMatrix& setScaleY(SkScalar v) { return this->set(kMScaleY, v); }

    /** Sets vertical skew factor.

        @param v  vertical skew factor to store
    */
    SkMatrix& setSkewY(SkScalar v) { return this->set(kMSkewY, v); }

    /** Sets horizontal skew factor.

        @param v  horizontal skew factor to store
    */
    SkMatrix& setSkewX(SkScalar v) { return this->set(kMSkewX, v); }

    /** Sets horizontal translation.

        @param v  horizontal translation to store
    */
    SkMatrix& setTranslateX(SkScalar v) { return this->set(kMTransX, v); }

    /** Sets vertical translation.

        @param v  vertical translation to store
    */
    SkMatrix& setTranslateY(SkScalar v) { return this->set(kMTransY, v); }

    /** Sets input x-axis perspective factor, which causes mapXY() to vary input x-axis values
        inversely proportional to input y-axis values.

        @param v  perspective factor
    */
    SkMatrix& setPerspX(SkScalar v) { return this->set(kMPersp0, v); }

    /** Sets input y-axis perspective factor, which causes mapXY() to vary input y-axis values
        inversely proportional to input x-axis values.

        @param v  perspective factor
    */
    SkMatrix& setPerspY(SkScalar v) { return this->set(kMPersp1, v); }

    /** Sets all values from parameters. Sets matrix to:

            | scaleX  skewX transX |
            |  skewY scaleY transY |
            | persp0 persp1 persp2 |

        @param scaleX  horizontal scale factor to store
        @param skewX   horizontal skew factor to store
        @param transX  horizontal translation to store
        @param skewY   vertical skew factor to store
        @param scaleY  vertical scale factor to store
        @param transY  vertical translation to store
        @param persp0  input x-axis values perspective factor to store
        @param persp1  input y-axis values perspective factor to store
        @param persp2  perspective scale factor to store
    */
    SkMatrix& setAll(SkScalar scaleX, SkScalar skewX,  SkScalar transX,
                     SkScalar skewY,  SkScalar scaleY, SkScalar transY,
                     SkScalar persp0, SkScalar persp1, SkScalar persp2) {
        fMat[kMScaleX] = scaleX;
        fMat[kMSkewX]  = skewX;
        fMat[kMTransX] = transX;
        fMat[kMSkewY]  = skewY;
        fMat[kMScaleY] = scaleY;
        fMat[kMTransY] = transY;
        fMat[kMPersp0] = persp0;
        fMat[kMPersp1] = persp1;
        fMat[kMPersp2] = persp2;
        this->setTypeMask(kUnknown_Mask);
        return *this;
    }

    /** Copies nine scalar values contained by SkMatrix into buffer, in member value
        ascending order: kMScaleX, kMSkewX, kMTransX, kMSkewY, kMScaleY, kMTransY,
        kMPersp0, kMPersp1, kMPersp2.

        @param buffer  storage for nine scalar values
    */
    void get9(SkScalar buffer[9]) const {
        memcpy(buffer, fMat, 9 * sizeof(SkScalar));
    }

    /** Sets SkMatrix to nine scalar values in buffer, in member value ascending order:
        kMScaleX, kMSkewX, kMTransX, kMSkewY, kMScaleY, kMTransY, kMPersp0, kMPersp1,
        kMPersp2.

        Sets matrix to:

            | buffer[0] buffer[1] buffer[2] |
            | buffer[3] buffer[4] buffer[5] |
            | buffer[6] buffer[7] buffer[8] |

        In the future, set9 followed by get9 may not return the same values. Since SkMatrix
        maps non-homogeneous coordinates, scaling all nine values produces an equivalent
        transformation, possibly improving precision.

        @param buffer  nine scalar values
    */
    SkMatrix& set9(const SkScalar buffer[9]);

    /** Sets SkMatrix to identity; which has no effect on mapped SkPoint. Sets SkMatrix to:

            | 1 0 0 |
            | 0 1 0 |
            | 0 0 1 |

        Also called setIdentity(); use the one that provides better inline
        documentation.
    */
    SkMatrix& reset();

    /** Sets SkMatrix to identity; which has no effect on mapped SkPoint. Sets SkMatrix to:

            | 1 0 0 |
            | 0 1 0 |
            | 0 0 1 |

        Also called reset(); use the one that provides better inline
        documentation.
    */
    SkMatrix& setIdentity() { return this->reset(); }

    /** Sets SkMatrix to translate by (dx, dy).

        @param dx  horizontal translation
        @param dy  vertical translation
    */
    SkMatrix& setTranslate(SkScalar dx, SkScalar dy);

    /** Sets SkMatrix to translate by (v.fX, v.fY).

        @param v  vector containing horizontal and vertical translation
    */
    SkMatrix& setTranslate(const SkVector& v) { return this->setTranslate(v.fX, v.fY); }

    /** Sets SkMatrix to scale by sx and sy, about a pivot point at (px, py).
        The pivot point is unchanged when mapped with SkMatrix.

        @param sx  horizontal scale factor
        @param sy  vertical scale factor
        @param px  pivot on x-axis
        @param py  pivot on y-axis
    */
    SkMatrix& setScale(SkScalar sx, SkScalar sy, SkScalar px, SkScalar py);

    /** Sets SkMatrix to scale by sx and sy about at pivot point at (0, 0).

        @param sx  horizontal scale factor
        @param sy  vertical scale factor
    */
    SkMatrix& setScale(SkScalar sx, SkScalar sy);

    /** Sets SkMatrix to rotate by degrees about a pivot point at (px, py).
        The pivot point is unchanged when mapped with SkMatrix.

        Positive degrees rotates clockwise.

        @param degrees  angle of axes relative to upright axes
        @param px       pivot on x-axis
        @param py       pivot on y-axis
    */
    SkMatrix& setRotate(SkScalar degrees, SkScalar px, SkScalar py);

    /** Sets SkMatrix to rotate by degrees about a pivot point at (0, 0).
        Positive degrees rotates clockwise.

        @param degrees  angle of axes relative to upright axes
    */
    SkMatrix& setRotate(SkScalar degrees);

    /** Sets SkMatrix to rotate by sinValue and cosValue, about a pivot point at (px, py).
        The pivot point is unchanged when mapped with SkMatrix.

        Vector (sinValue, cosValue) describes the angle of rotation relative to (0, 1).
        Vector length specifies scale.

        @param sinValue  rotation vector x-axis component
        @param cosValue  rotation vector y-axis component
        @param px        pivot on x-axis
        @param py        pivot on y-axis
    */
    SkMatrix& setSinCos(SkScalar sinValue, SkScalar cosValue,
                   SkScalar px, SkScalar py);

    /** Sets SkMatrix to rotate by sinValue and cosValue, about a pivot point at (0, 0).

        Vector (sinValue, cosValue) describes the angle of rotation relative to (0, 1).
        Vector length specifies scale.

        @param sinValue  rotation vector x-axis component
        @param cosValue  rotation vector y-axis component
    */
    SkMatrix& setSinCos(SkScalar sinValue, SkScalar cosValue);

    /** Sets SkMatrix to rotate, scale, and translate using a compressed matrix form.

        Vector (rsxForm.fSSin, rsxForm.fSCos) describes the angle of rotation relative
        to (0, 1). Vector length specifies scale. Mapped point is rotated and scaled
        by vector, then translated by (rsxForm.fTx, rsxForm.fTy).

        @param rsxForm  compressed SkRSXform matrix
        @return         reference to SkMatrix

        example: https://fiddle.skia.org/c/@Matrix_setRSXform
    */
    SkMatrix& setRSXform(const SkRSXform& rsxForm);

    /** Sets SkMatrix to skew by kx and ky, about a pivot point at (px, py).
        The pivot point is unchanged when mapped with SkMatrix.

        @param kx  horizontal skew factor
        @param ky  vertical skew factor
        @param px  pivot on x-axis
        @param py  pivot on y-axis
    */
    SkMatrix& setSkew(SkScalar kx, SkScalar ky, SkScalar px, SkScalar py);

    /** Sets SkMatrix to skew by kx and ky, about a pivot point at (0, 0).

        @param kx  horizontal skew factor
        @param ky  vertical skew factor
    */
    SkMatrix& setSkew(SkScalar kx, SkScalar ky);

    /** Sets SkMatrix to SkMatrix a multiplied by SkMatrix b. Either a or b may be this.

        Given:

                | A B C |      | J K L |
            a = | D E F |, b = | M N O |
                | G H I |      | P Q R |

        sets SkMatrix to:

                    | A B C |   | J K L |   | AJ+BM+CP AK+BN+CQ AL+BO+CR |
            a * b = | D E F | * | M N O | = | DJ+EM+FP DK+EN+FQ DL+EO+FR |
                    | G H I |   | P Q R |   | GJ+HM+IP GK+HN+IQ GL+HO+IR |

        @param a  SkMatrix on left side of multiply expression
        @param b  SkMatrix on right side of multiply expression
    */
    SkMatrix& setConcat(const SkMatrix& a, const SkMatrix& b);

    /** Sets SkMatrix to SkMatrix multiplied by SkMatrix constructed from translation (dx, dy).
        This can be thought of as moving the point to be mapped before applying SkMatrix.

        Given:

                     | A B C |               | 1 0 dx |
            Matrix = | D E F |,  T(dx, dy) = | 0 1 dy |
                     | G H I |               | 0 0  1 |

        sets SkMatrix to:

                                 | A B C | | 1 0 dx |   | A B A*dx+B*dy+C |
            Matrix * T(dx, dy) = | D E F | | 0 1 dy | = | D E D*dx+E*dy+F |
                                 | G H I | | 0 0  1 |   | G H G*dx+H*dy+I |

        @param dx  x-axis translation before applying SkMatrix
        @param dy  y-axis translation before applying SkMatrix
    */
    SkMatrix& preTranslate(SkScalar dx, SkScalar dy);

    /** Sets SkMatrix to SkMatrix multiplied by SkMatrix constructed from scaling by (sx, sy)
        about pivot point (px, py).
        This can be thought of as scaling about a pivot point before applying SkMatrix.

        Given:

                     | A B C |                       | sx  0 dx |
            Matrix = | D E F |,  S(sx, sy, px, py) = |  0 sy dy |
                     | G H I |                       |  0  0  1 |

        where

            dx = px - sx * px
            dy = py - sy * py

        sets SkMatrix to:

                                         | A B C | | sx  0 dx |   | A*sx B*sy A*dx+B*dy+C |
            Matrix * S(sx, sy, px, py) = | D E F | |  0 sy dy | = | D*sx E*sy D*dx+E*dy+F |
                                         | G H I | |  0  0  1 |   | G*sx H*sy G*dx+H*dy+I |

        @param sx  horizontal scale factor
        @param sy  vertical scale factor
        @param px  pivot on x-axis
        @param py  pivot on y-axis
    */
    SkMatrix& preScale(SkScalar sx, SkScalar sy, SkScalar px, SkScalar py);

    /** Sets SkMatrix to SkMatrix multiplied by SkMatrix constructed from scaling by (sx, sy)
        about pivot point (0, 0).
        This can be thought of as scaling about the origin before applying SkMatrix.

        Given:

                     | A B C |               | sx  0  0 |
            Matrix = | D E F |,  S(sx, sy) = |  0 sy  0 |
                     | G H I |               |  0  0  1 |

        sets SkMatrix to:

                                 | A B C | | sx  0  0 |   | A*sx B*sy C |
            Matrix * S(sx, sy) = | D E F | |  0 sy  0 | = | D*sx E*sy F |
                                 | G H I | |  0  0  1 |   | G*sx H*sy I |

        @param sx  horizontal scale factor
        @param sy  vertical scale factor
    */
    SkMatrix& preScale(SkScalar sx, SkScalar sy);

    /** Sets SkMatrix to SkMatrix multiplied by SkMatrix constructed from rotating by degrees
        about pivot point (px, py).
        This can be thought of as rotating about a pivot point before applying SkMatrix.

        Positive degrees rotates clockwise.

        Given:

                     | A B C |                        | c -s dx |
            Matrix = | D E F |,  R(degrees, px, py) = | s  c dy |
                     | G H I |                        | 0  0  1 |

        where

            c  = cos(degrees)
            s  = sin(degrees)
            dx =  s * py + (1 - c) * px
            dy = -s * px + (1 - c) * py

        sets SkMatrix to:

                                          | A B C | | c -s dx |   | Ac+Bs -As+Bc A*dx+B*dy+C |
            Matrix * R(degrees, px, py) = | D E F | | s  c dy | = | Dc+Es -Ds+Ec D*dx+E*dy+F |
                                          | G H I | | 0  0  1 |   | Gc+Hs -Gs+Hc G*dx+H*dy+I |

        @param degrees  angle of axes relative to upright axes
        @param px       pivot on x-axis
        @param py       pivot on y-axis
    */
    SkMatrix& preRotate(SkScalar degrees, SkScalar px, SkScalar py);

    /** Sets SkMatrix to SkMatrix multiplied by SkMatrix constructed from rotating by degrees
        about pivot point (0, 0).
        This can be thought of as rotating about the origin before applying SkMatrix.

        Positive degrees rotates clockwise.

        Given:

                     | A B C |                        | c -s 0 |
            Matrix = | D E F |,  R(degrees, px, py) = | s  c 0 |
                     | G H I |                        | 0  0 1 |

        where

            c  = cos(degrees)
            s  = sin(degrees)

        sets SkMatrix to:

                                          | A B C | | c -s 0 |   | Ac+Bs -As+Bc C |
            Matrix * R(degrees, px, py) = | D E F | | s  c 0 | = | Dc+Es -Ds+Ec F |
                                          | G H I | | 0  0 1 |   | Gc+Hs -Gs+Hc I |

        @param degrees  angle of axes relative to upright axes
    */
    SkMatrix& preRotate(SkScalar degrees);

    /** Sets SkMatrix to SkMatrix multiplied by SkMatrix constructed from skewing by (kx, ky)
        about pivot point (px, py).
        This can be thought of as skewing about a pivot point before applying SkMatrix.

        Given:

                     | A B C |                       |  1 kx dx |
            Matrix = | D E F |,  K(kx, ky, px, py) = | ky  1 dy |
                     | G H I |                       |  0  0  1 |

        where

            dx = -kx * py
            dy = -ky * px

        sets SkMatrix to:

                                         | A B C | |  1 kx dx |   | A+B*ky A*kx+B A*dx+B*dy+C |
            Matrix * K(kx, ky, px, py) = | D E F | | ky  1 dy | = | D+E*ky D*kx+E D*dx+E*dy+F |
                                         | G H I | |  0  0  1 |   | G+H*ky G*kx+H G*dx+H*dy+I |

        @param kx  horizontal skew factor
        @param ky  vertical skew factor
        @param px  pivot on x-axis
        @param py  pivot on y-axis
    */
    SkMatrix& preSkew(SkScalar kx, SkScalar ky, SkScalar px, SkScalar py);

    /** Sets SkMatrix to SkMatrix multiplied by SkMatrix constructed from skewing by (kx, ky)
        about pivot point (0, 0).
        This can be thought of as skewing about the origin before applying SkMatrix.

        Given:

                     | A B C |               |  1 kx 0 |
            Matrix = | D E F |,  K(kx, ky) = | ky  1 0 |
                     | G H I |               |  0  0 1 |

        sets SkMatrix to:

                                 | A B C | |  1 kx 0 |   | A+B*ky A*kx+B C |
            Matrix * K(kx, ky) = | D E F | | ky  1 0 | = | D+E*ky D*kx+E F |
                                 | G H I | |  0  0 1 |   | G+H*ky G*kx+H I |

        @param kx  horizontal skew factor
        @param ky  vertical skew factor
    */
    SkMatrix& preSkew(SkScalar kx, SkScalar ky);

    /** Sets SkMatrix to SkMatrix multiplied by SkMatrix other.
        This can be thought of mapping by other before applying SkMatrix.

        Given:

                     | A B C |          | J K L |
            Matrix = | D E F |, other = | M N O |
                     | G H I |          | P Q R |

        sets SkMatrix to:

                             | A B C |   | J K L |   | AJ+BM+CP AK+BN+CQ AL+BO+CR |
            Matrix * other = | D E F | * | M N O | = | DJ+EM+FP DK+EN+FQ DL+EO+FR |
                             | G H I |   | P Q R |   | GJ+HM+IP GK+HN+IQ GL+HO+IR |

        @param other  SkMatrix on right side of multiply expression
    */
    SkMatrix& preConcat(const SkMatrix& other);

    /** Sets SkMatrix to SkMatrix constructed from translation (dx, dy) multiplied by SkMatrix.
        This can be thought of as moving the point to be mapped after applying SkMatrix.

        Given:

                     | J K L |               | 1 0 dx |
            Matrix = | M N O |,  T(dx, dy) = | 0 1 dy |
                     | P Q R |               | 0 0  1 |

        sets SkMatrix to:

                                 | 1 0 dx | | J K L |   | J+dx*P K+dx*Q L+dx*R |
            T(dx, dy) * Matrix = | 0 1 dy | | M N O | = | M+dy*P N+dy*Q O+dy*R |
                                 | 0 0  1 | | P Q R |   |      P      Q      R |

        @param dx  x-axis translation after applying SkMatrix
        @param dy  y-axis translation after applying SkMatrix
    */
    SkMatrix& postTranslate(SkScalar dx, SkScalar dy);

    /** Sets SkMatrix to SkMatrix constructed from scaling by (sx, sy) about pivot point
        (px, py), multiplied by SkMatrix.
        This can be thought of as scaling about a pivot point after applying SkMatrix.

        Given:

                     | J K L |                       | sx  0 dx |
            Matrix = | M N O |,  S(sx, sy, px, py) = |  0 sy dy |
                     | P Q R |                       |  0  0  1 |

        where

            dx = px - sx * px
            dy = py - sy * py

        sets SkMatrix to:

                                         | sx  0 dx | | J K L |   | sx*J+dx*P sx*K+dx*Q sx*L+dx+R |
            S(sx, sy, px, py) * Matrix = |  0 sy dy | | M N O | = | sy*M+dy*P sy*N+dy*Q sy*O+dy*R |
                                         |  0  0  1 | | P Q R |   |         P         Q         R |

        @param sx  horizontal scale factor
        @param sy  vertical scale factor
        @param px  pivot on x-axis
        @param py  pivot on y-axis
    */
    SkMatrix& postScale(SkScalar sx, SkScalar sy, SkScalar px, SkScalar py);

    /** Sets SkMatrix to SkMatrix constructed from scaling by (sx, sy) about pivot point
        (0, 0), multiplied by SkMatrix.
        This can be thought of as scaling about the origin after applying SkMatrix.

        Given:

                     | J K L |               | sx  0  0 |
            Matrix = | M N O |,  S(sx, sy) = |  0 sy  0 |
                     | P Q R |               |  0  0  1 |

        sets SkMatrix to:

                                 | sx  0  0 | | J K L |   | sx*J sx*K sx*L |
            S(sx, sy) * Matrix = |  0 sy  0 | | M N O | = | sy*M sy*N sy*O |
                                 |  0  0  1 | | P Q R |   |    P    Q    R |

        @param sx  horizontal scale factor
        @param sy  vertical scale factor
    */
    SkMatrix& postScale(SkScalar sx, SkScalar sy);

    /** Sets SkMatrix to SkMatrix constructed from rotating by degrees about pivot point
        (px, py), multiplied by SkMatrix.
        This can be thought of as rotating about a pivot point after applying SkMatrix.

        Positive degrees rotates clockwise.

        Given:

                     | J K L |                        | c -s dx |
            Matrix = | M N O |,  R(degrees, px, py) = | s  c dy |
                     | P Q R |                        | 0  0  1 |

        where

            c  = cos(degrees)
            s  = sin(degrees)
            dx =  s * py + (1 - c) * px
            dy = -s * px + (1 - c) * py

        sets SkMatrix to:

                                          |c -s dx| |J K L|   |cJ-sM+dx*P cK-sN+dx*Q cL-sO+dx+R|
            R(degrees, px, py) * Matrix = |s  c dy| |M N O| = |sJ+cM+dy*P sK+cN+dy*Q sL+cO+dy*R|
                                          |0  0  1| |P Q R|   |         P          Q          R|

        @param degrees  angle of axes relative to upright axes
        @param px       pivot on x-axis
        @param py       pivot on y-axis
    */
    SkMatrix& postRotate(SkScalar degrees, SkScalar px, SkScalar py);

    /** Sets SkMatrix to SkMatrix constructed from rotating by degrees about pivot point
        (0, 0), multiplied by SkMatrix.
        This can be thought of as rotating about the origin after applying SkMatrix.

        Positive degrees rotates clockwise.

        Given:

                     | J K L |                        | c -s 0 |
            Matrix = | M N O |,  R(degrees, px, py) = | s  c 0 |
                     | P Q R |                        | 0  0 1 |

        where

            c  = cos(degrees)
            s  = sin(degrees)

        sets SkMatrix to:

                                          | c -s dx | | J K L |   | cJ-sM cK-sN cL-sO |
            R(degrees, px, py) * Matrix = | s  c dy | | M N O | = | sJ+cM sK+cN sL+cO |
                                          | 0  0  1 | | P Q R |   |     P     Q     R |

        @param degrees  angle of axes relative to upright axes
    */
    SkMatrix& postRotate(SkScalar degrees);

    /** Sets SkMatrix to SkMatrix constructed from skewing by (kx, ky) about pivot point
        (px, py), multiplied by SkMatrix.
        This can be thought of as skewing about a pivot point after applying SkMatrix.

        Given:

                     | J K L |                       |  1 kx dx |
            Matrix = | M N O |,  K(kx, ky, px, py) = | ky  1 dy |
                     | P Q R |                       |  0  0  1 |

        where

            dx = -kx * py
            dy = -ky * px

        sets SkMatrix to:

                                         | 1 kx dx| |J K L|   |J+kx*M+dx*P K+kx*N+dx*Q L+kx*O+dx+R|
            K(kx, ky, px, py) * Matrix = |ky  1 dy| |M N O| = |ky*J+M+dy*P ky*K+N+dy*Q ky*L+O+dy*R|
                                         | 0  0  1| |P Q R|   |          P           Q           R|

        @param kx  horizontal skew factor
        @param ky  vertical skew factor
        @param px  pivot on x-axis
        @param py  pivot on y-axis
    */
    SkMatrix& postSkew(SkScalar kx, SkScalar ky, SkScalar px, SkScalar py);

    /** Sets SkMatrix to SkMatrix constructed from skewing by (kx, ky) about pivot point
        (0, 0), multiplied by SkMatrix.
        This can be thought of as skewing about the origin after applying SkMatrix.

        Given:

                     | J K L |               |  1 kx 0 |
            Matrix = | M N O |,  K(kx, ky) = | ky  1 0 |
                     | P Q R |               |  0  0 1 |

        sets SkMatrix to:

                                 |  1 kx 0 | | J K L |   | J+kx*M K+kx*N L+kx*O |
            K(kx, ky) * Matrix = | ky  1 0 | | M N O | = | ky*J+M ky*K+N ky*L+O |
                                 |  0  0 1 | | P Q R |   |      P      Q      R |

        @param kx  horizontal skew factor
        @param ky  vertical skew factor
    */
    SkMatrix& postSkew(SkScalar kx, SkScalar ky);

    /** Sets SkMatrix to SkMatrix other multiplied by SkMatrix.
        This can be thought of mapping by other after applying SkMatrix.

        Given:

                     | J K L |           | A B C |
            Matrix = | M N O |,  other = | D E F |
                     | P Q R |           | G H I |

        sets SkMatrix to:

                             | A B C |   | J K L |   | AJ+BM+CP AK+BN+CQ AL+BO+CR |
            other * Matrix = | D E F | * | M N O | = | DJ+EM+FP DK+EN+FQ DL+EO+FR |
                             | G H I |   | P Q R |   | GJ+HM+IP GK+HN+IQ GL+HO+IR |

        @param other  SkMatrix on left side of multiply expression
    */
    SkMatrix& postConcat(const SkMatrix& other);

#ifndef SK_SUPPORT_LEGACY_MATRIX_RECTTORECT
private:
#endif
    /** Sets SkMatrix to scale and translate src SkRect to dst SkRect. stf selects whether
        mapping completely fills dst or preserves the aspect ratio, and how to align
        src within dst. Returns false if src is empty, and sets SkMatrix to identity.
        Returns true if dst is empty, and sets SkMatrix to:

            | 0 0 0 |
            | 0 0 0 |
            | 0 0 1 |

        @param src  SkRect to map from
        @param dst  SkRect to map to
        @return     true if SkMatrix can represent SkRect mapping

        example: https://fiddle.skia.org/c/@Matrix_setRectToRect
    */
    bool setRectToRect(const SkRect& src, const SkRect& dst, ScaleToFit stf);

    /** Returns SkMatrix set to scale and translate src SkRect to dst SkRect. stf selects
        whether mapping completely fills dst or preserves the aspect ratio, and how to
        align src within dst. Returns the identity SkMatrix if src is empty. If dst is
        empty, returns SkMatrix set to:

            | 0 0 0 |
            | 0 0 0 |
            | 0 0 1 |

        @param src  SkRect to map from
        @param dst  SkRect to map to
        @return     SkMatrix mapping src to dst
    */
    static SkMatrix MakeRectToRect(const SkRect& src, const SkRect& dst, ScaleToFit stf) {
        SkMatrix m;
        m.setRectToRect(src, dst, stf);
        return m;
    }
#ifndef SK_SUPPORT_LEGACY_MATRIX_RECTTORECT
public:
#endif

    /** Sets SkMatrix to map src to dst. count must be zero or greater, and four or less.

        If count is zero, sets SkMatrix to identity and returns true.
        If count is one, sets SkMatrix to translate and returns true.
        If count is two or more, sets SkMatrix to map SkPoint if possible; returns false
        if SkMatrix cannot be constructed. If count is four, SkMatrix may include
        perspective.

        @param src    SkPoint to map from
        @param dst    SkPoint to map to
        @param count  number of SkPoint in src and dst
        @return       true if SkMatrix was constructed successfully

        example: https://fiddle.skia.org/c/@Matrix_setPolyToPoly
    */
    bool setPolyToPoly(const SkPoint src[], const SkPoint dst[], int count);

    /** Sets inverse to reciprocal matrix, returning true if SkMatrix can be inverted.
        Geometrically, if SkMatrix maps from source to destination, inverse SkMatrix
        maps from destination to source. If SkMatrix can not be inverted, inverse is
        unchanged.

        @param inverse  storage for inverted SkMatrix; may be nullptr
        @return         true if SkMatrix can be inverted
    */
    bool SK_WARN_UNUSED_RESULT invert(SkMatrix* inverse) const {
        // Allow the trivial case to be inlined.
        if (this->isIdentity()) {
            if (inverse) {
                inverse->reset();
            }
            return true;
        }
        return this->invertNonIdentity(inverse);
    }

    /** Fills affine with identity values in column major order.
        Sets affine to:

            | 1 0 0 |
            | 0 1 0 |

        Affine 3 by 2 matrices in column major order are used by OpenGL and XPS.

        @param affine  storage for 3 by 2 affine matrix

        example: https://fiddle.skia.org/c/@Matrix_SetAffineIdentity
    */
    static void SetAffineIdentity(SkScalar affine[6]);

    /** Fills affine in column major order. Sets affine to:

            | scale-x  skew-x translate-x |
            | skew-y  scale-y translate-y |

        If SkMatrix contains perspective, returns false and leaves affine unchanged.

        @param affine  storage for 3 by 2 affine matrix; may be nullptr
        @return        true if SkMatrix does not contain perspective
    */
    bool SK_WARN_UNUSED_RESULT asAffine(SkScalar affine[6]) const;

    /** Sets SkMatrix to affine values, passed in column major order. Given affine,
        column, then row, as:

            | scale-x  skew-x translate-x |
            |  skew-y scale-y translate-y |

        SkMatrix is set, row, then column, to:

            | scale-x  skew-x translate-x |
            |  skew-y scale-y translate-y |
            |       0       0           1 |

        @param affine  3 by 2 affine matrix
    */
    SkMatrix& setAffine(const SkScalar affine[6]);

    /**
     *  A matrix is categorized as 'perspective' if the bottom row is not [0, 0, 1].
     *  However, for most uses (e.g. mapPoints) a bottom row of [0, 0, X] behaves like a
     *  non-perspective matrix, though it will be categorized as perspective. Calling
     *  normalizePerspective() will change the matrix such that, if its bottom row was [0, 0, X],
     *  it will be changed to [0, 0, 1] by scaling the rest of the matrix by 1/X.
     *
     *  | A B C |    | A/X B/X C/X |
     *  | D E F | -> | D/X E/X F/X |   for X != 0
     *  | 0 0 X |    |  0   0   1  |
     */
    void normalizePerspective() {
        if (fMat[8] != 1) {
            this->doNormalizePerspective();
        }
    }

    /** Maps src SkPoint array of length count to dst SkPoint array of equal or greater
        length. SkPoint are mapped by multiplying each SkPoint by SkMatrix. Given:

                     | A B C |        | x |
            Matrix = | D E F |,  pt = | y |
                     | G H I |        | 1 |

        where

            for (i = 0; i < count; ++i) {
                x = src[i].fX
                y = src[i].fY
            }

        each dst SkPoint is computed as:

                          |A B C| |x|                               Ax+By+C   Dx+Ey+F
            Matrix * pt = |D E F| |y| = |Ax+By+C Dx+Ey+F Gx+Hy+I| = ------- , -------
                          |G H I| |1|                               Gx+Hy+I   Gx+Hy+I

        src and dst may point to the same storage.

        @param dst    storage for mapped SkPoint
        @param src    SkPoint to transform
        @param count  number of SkPoint to transform

        example: https://fiddle.skia.org/c/@Matrix_mapPoints
    */
    void mapPoints(SkPoint dst[], const SkPoint src[], int count) const;

    /** Maps pts SkPoint array of length count in place. SkPoint are mapped by multiplying
        each SkPoint by SkMatrix. Given:

                     | A B C |        | x |
            Matrix = | D E F |,  pt = | y |
                     | G H I |        | 1 |

        where

            for (i = 0; i < count; ++i) {
                x = pts[i].fX
                y = pts[i].fY
            }

        each resulting pts SkPoint is computed as:

                          |A B C| |x|                               Ax+By+C   Dx+Ey+F
            Matrix * pt = |D E F| |y| = |Ax+By+C Dx+Ey+F Gx+Hy+I| = ------- , -------
                          |G H I| |1|                               Gx+Hy+I   Gx+Hy+I

        @param pts    storage for mapped SkPoint
        @param count  number of SkPoint to transform
    */
    void mapPoints(SkPoint pts[], int count) const {
        this->mapPoints(pts, pts, count);
    }

    /** Maps src SkPoint3 array of length count to dst SkPoint3 array, which must of length count or
        greater. SkPoint3 array is mapped by multiplying each SkPoint3 by SkMatrix. Given:

                     | A B C |         | x |
            Matrix = | D E F |,  src = | y |
                     | G H I |         | z |

        each resulting dst SkPoint is computed as:

                           |A B C| |x|
            Matrix * src = |D E F| |y| = |Ax+By+Cz Dx+Ey+Fz Gx+Hy+Iz|
                           |G H I| |z|

        @param dst    storage for mapped SkPoint3 array
        @param src    SkPoint3 array to transform
        @param count  items in SkPoint3 array to transform

        example: https://fiddle.skia.org/c/@Matrix_mapHomogeneousPoints
    */
    void mapHomogeneousPoints(SkPoint3 dst[], const SkPoint3 src[], int count) const;

    /**
     *  Returns homogeneous points, starting with 2D src points (with implied w = 1).
     */
    void mapHomogeneousPoints(SkPoint3 dst[], const SkPoint src[], int count) const;

    /** Returns SkPoint pt multiplied by SkMatrix. Given:

                     | A B C |        | x |
            Matrix = | D E F |,  pt = | y |
                     | G H I |        | 1 |

        result is computed as:

                          |A B C| |x|                               Ax+By+C   Dx+Ey+F
            Matrix * pt = |D E F| |y| = |Ax+By+C Dx+Ey+F Gx+Hy+I| = ------- , -------
                          |G H I| |1|                               Gx+Hy+I   Gx+Hy+I

        @param p  SkPoint to map
        @return   mapped SkPoint
    */
    SkPoint mapPoint(SkPoint pt) const {
        SkPoint result;
        this->mapXY(pt.x(), pt.y(), &result);
        return result;
    }

    /** Maps SkPoint (x, y) to result. SkPoint is mapped by multiplying by SkMatrix. Given:

                     | A B C |        | x |
            Matrix = | D E F |,  pt = | y |
                     | G H I |        | 1 |

        result is computed as:

                          |A B C| |x|                               Ax+By+C   Dx+Ey+F
            Matrix * pt = |D E F| |y| = |Ax+By+C Dx+Ey+F Gx+Hy+I| = ------- , -------
                          |G H I| |1|                               Gx+Hy+I   Gx+Hy+I

        @param x       x-axis value of SkPoint to map
        @param y       y-axis value of SkPoint to map
        @param result  storage for mapped SkPoint

        example: https://fiddle.skia.org/c/@Matrix_mapXY
    */
    void mapXY(SkScalar x, SkScalar y, SkPoint* result) const;

    /** Returns SkPoint (x, y) multiplied by SkMatrix. Given:

                     | A B C |        | x |
            Matrix = | D E F |,  pt = | y |
                     | G H I |        | 1 |

        result is computed as:

                          |A B C| |x|                               Ax+By+C   Dx+Ey+F
            Matrix * pt = |D E F| |y| = |Ax+By+C Dx+Ey+F Gx+Hy+I| = ------- , -------
                          |G H I| |1|                               Gx+Hy+I   Gx+Hy+I

        @param x  x-axis value of SkPoint to map
        @param y  y-axis value of SkPoint to map
        @return   mapped SkPoint
    */
    SkPoint mapXY(SkScalar x, SkScalar y) const {
        SkPoint result;
        this->mapXY(x,y, &result);
        return result;
    }


    /** Returns (0, 0) multiplied by SkMatrix. Given:

                     | A B C |        | 0 |
            Matrix = | D E F |,  pt = | 0 |
                     | G H I |        | 1 |

        result is computed as:

                          |A B C| |0|             C    F
            Matrix * pt = |D E F| |0| = |C F I| = -  , -
                          |G H I| |1|             I    I

        @return   mapped (0, 0)
    */
    SkPoint mapOrigin() const {
        SkScalar x = this->getTranslateX(),
                 y = this->getTranslateY();
        if (this->hasPerspective()) {
            SkScalar w = fMat[kMPersp2];
            if (w) { w = 1 / w; }
            x *= w;
            y *= w;
        }
        return {x, y};
    }

    /** Maps src vector array of length count to vector SkPoint array of equal or greater
        length. Vectors are mapped by multiplying each vector by SkMatrix, treating
        SkMatrix translation as zero. Given:

                     | A B 0 |         | x |
            Matrix = | D E 0 |,  src = | y |
                     | G H I |         | 1 |

        where

            for (i = 0; i < count; ++i) {
                x = src[i].fX
                y = src[i].fY
            }

        each dst vector is computed as:

                           |A B 0| |x|                            Ax+By     Dx+Ey
            Matrix * src = |D E 0| |y| = |Ax+By Dx+Ey Gx+Hy+I| = ------- , -------
                           |G H I| |1|                           Gx+Hy+I   Gx+Hy+I

        src and dst may point to the same storage.

        @param dst    storage for mapped vectors
        @param src    vectors to transform
        @param count  number of vectors to transform

        example: https://fiddle.skia.org/c/@Matrix_mapVectors
    */
    void mapVectors(SkVector dst[], const SkVector src[], int count) const;

    /** Maps vecs vector array of length count in place, multiplying each vector by
        SkMatrix, treating SkMatrix translation as zero. Given:

                     | A B 0 |         | x |
            Matrix = | D E 0 |,  vec = | y |
                     | G H I |         | 1 |

        where

            for (i = 0; i < count; ++i) {
                x = vecs[i].fX
                y = vecs[i].fY
            }

        each result vector is computed as:

                           |A B 0| |x|                            Ax+By     Dx+Ey
            Matrix * vec = |D E 0| |y| = |Ax+By Dx+Ey Gx+Hy+I| = ------- , -------
                           |G H I| |1|                           Gx+Hy+I   Gx+Hy+I

        @param vecs   vectors to transform, and storage for mapped vectors
        @param count  number of vectors to transform
    */
    void mapVectors(SkVector vecs[], int count) const {
        this->mapVectors(vecs, vecs, count);
    }

    /** Maps vector (dx, dy) to result. Vector is mapped by multiplying by SkMatrix,
        treating SkMatrix translation as zero. Given:

                     | A B 0 |         | dx |
            Matrix = | D E 0 |,  vec = | dy |
                     | G H I |         |  1 |

        each result vector is computed as:

                       |A B 0| |dx|                                        A*dx+B*dy     D*dx+E*dy
        Matrix * vec = |D E 0| |dy| = |A*dx+B*dy D*dx+E*dy G*dx+H*dy+I| = ----------- , -----------
                       |G H I| | 1|                                       G*dx+H*dy+I   G*dx+*dHy+I

        @param dx      x-axis value of vector to map
        @param dy      y-axis value of vector to map
        @param result  storage for mapped vector
    */
    void mapVector(SkScalar dx, SkScalar dy, SkVector* result) const {
        SkVector vec = { dx, dy };
        this->mapVectors(result, &vec, 1);
    }

    /** Returns vector (dx, dy) multiplied by SkMatrix, treating SkMatrix translation as zero.
        Given:

                     | A B 0 |         | dx |
            Matrix = | D E 0 |,  vec = | dy |
                     | G H I |         |  1 |

        each result vector is computed as:

                       |A B 0| |dx|                                        A*dx+B*dy     D*dx+E*dy
        Matrix * vec = |D E 0| |dy| = |A*dx+B*dy D*dx+E*dy G*dx+H*dy+I| = ----------- , -----------
                       |G H I| | 1|                                       G*dx+H*dy+I   G*dx+*dHy+I

        @param dx  x-axis value of vector to map
        @param dy  y-axis value of vector to map
        @return    mapped vector
    */
    SkVector mapVector(SkScalar dx, SkScalar dy) const {
        SkVector vec = { dx, dy };
        this->mapVectors(&vec, &vec, 1);
        return vec;
    }

    /** Sets dst to bounds of src corners mapped by SkMatrix.
        Returns true if mapped corners are dst corners.

        Returned value is the same as calling rectStaysRect().

        @param dst  storage for bounds of mapped SkPoint
        @param src  SkRect to map
        @param pc   whether to apply perspective clipping
        @return     true if dst is equivalent to mapped src

        example: https://fiddle.skia.org/c/@Matrix_mapRect
    */
    bool mapRect(SkRect* dst, const SkRect& src,
                 SkApplyPerspectiveClip pc = SkApplyPerspectiveClip::kYes) const;

    /** Sets rect to bounds of rect corners mapped by SkMatrix.
        Returns true if mapped corners are computed rect corners.

        Returned value is the same as calling rectStaysRect().

        @param rect  rectangle to map, and storage for bounds of mapped corners
        @param pc    whether to apply perspective clipping
        @return      true if result is equivalent to mapped rect
    */
    bool mapRect(SkRect* rect, SkApplyPerspectiveClip pc = SkApplyPerspectiveClip::kYes) const {
        return this->mapRect(rect, *rect, pc);
    }

    /** Returns bounds of src corners mapped by SkMatrix.

        @param src  rectangle to map
        @return     mapped bounds
    */
    SkRect mapRect(const SkRect& src,
                   SkApplyPerspectiveClip pc = SkApplyPerspectiveClip::kYes) const {
        SkRect dst;
        (void)this->mapRect(&dst, src, pc);
        return dst;
    }

    /** Maps four corners of rect to dst. SkPoint are mapped by multiplying each
        rect corner by SkMatrix. rect corner is processed in this order:
        (rect.fLeft, rect.fTop), (rect.fRight, rect.fTop), (rect.fRight, rect.fBottom),
        (rect.fLeft, rect.fBottom).

        rect may be empty: rect.fLeft may be greater than or equal to rect.fRight;
        rect.fTop may be greater than or equal to rect.fBottom.

        Given:

                     | A B C |        | x |
            Matrix = | D E F |,  pt = | y |
                     | G H I |        | 1 |

        where pt is initialized from each of (rect.fLeft, rect.fTop),
        (rect.fRight, rect.fTop), (rect.fRight, rect.fBottom), (rect.fLeft, rect.fBottom),
        each dst SkPoint is computed as:

                          |A B C| |x|                               Ax+By+C   Dx+Ey+F
            Matrix * pt = |D E F| |y| = |Ax+By+C Dx+Ey+F Gx+Hy+I| = ------- , -------
                          |G H I| |1|                               Gx+Hy+I   Gx+Hy+I

        @param dst   storage for mapped corner SkPoint
        @param rect  SkRect to map

        Note: this does not perform perspective clipping (as that might result in more than
              4 points, so results are suspect if the matrix contains perspective.
    */
    void mapRectToQuad(SkPoint dst[4], const SkRect& rect) const {
        // This could potentially be faster if we only transformed each x and y of the rect once.
        rect.toQuad(dst);
        this->mapPoints(dst, 4);
    }

    /** Sets dst to bounds of src corners mapped by SkMatrix. If matrix contains
        elements other than scale or translate: asserts if SK_DEBUG is defined;
        otherwise, results are undefined.

        @param dst  storage for bounds of mapped SkPoint
        @param src  SkRect to map

        example: https://fiddle.skia.org/c/@Matrix_mapRectScaleTranslate
    */
    void mapRectScaleTranslate(SkRect* dst, const SkRect& src) const;

    /** Returns geometric mean radius of ellipse formed by constructing circle of
        size radius, and mapping constructed circle with SkMatrix. The result squared is
        equal to the major axis length times the minor axis length.
        Result is not meaningful if SkMatrix contains perspective elements.

        @param radius  circle size to map
        @return        average mapped radius

        example: https://fiddle.skia.org/c/@Matrix_mapRadius
    */
    SkScalar mapRadius(SkScalar radius) const;

    /** Compares a and b; returns true if a and b are numerically equal. Returns true
        even if sign of zero values are different. Returns false if either SkMatrix
        contains NaN, even if the other SkMatrix also contains NaN.

        @param a  SkMatrix to compare
        @param b  SkMatrix to compare
        @return   true if SkMatrix a and SkMatrix b are numerically equal
    */
    friend SK_API bool operator==(const SkMatrix& a, const SkMatrix& b);

    /** Compares a and b; returns true if a and b are not numerically equal. Returns false
        even if sign of zero values are different. Returns true if either SkMatrix
        contains NaN, even if the other SkMatrix also contains NaN.

        @param a  SkMatrix to compare
        @param b  SkMatrix to compare
        @return   true if SkMatrix a and SkMatrix b are numerically not equal
    */
    friend SK_API bool operator!=(const SkMatrix& a, const SkMatrix& b) {
        return !(a == b);
    }

    /** Writes text representation of SkMatrix to standard output. Floating point values
        are written with limited precision; it may not be possible to reconstruct
        original SkMatrix from output.

        example: https://fiddle.skia.org/c/@Matrix_dump
    */
// -- GODOT start --
    //void dump() const;
// -- GODOT end -- 

    /** Returns the minimum scaling factor of SkMatrix by decomposing the scaling and
        skewing elements.
        Returns -1 if scale factor overflows or SkMatrix contains perspective.

        @return  minimum scale factor

        example: https://fiddle.skia.org/c/@Matrix_getMinScale
    */
    SkScalar getMinScale() const;

    /** Returns the maximum scaling factor of SkMatrix by decomposing the scaling and
        skewing elements.
        Returns -1 if scale factor overflows or SkMatrix contains perspective.

        @return  maximum scale factor

        example: https://fiddle.skia.org/c/@Matrix_getMaxScale
    */
    SkScalar getMaxScale() const;

    /** Sets scaleFactors[0] to the minimum scaling factor, and scaleFactors[1] to the
        maximum scaling factor. Scaling factors are computed by decomposing
        the SkMatrix scaling and skewing elements.

        Returns true if scaleFactors are found; otherwise, returns false and sets
        scaleFactors to undefined values.

        @param scaleFactors  storage for minimum and maximum scale factors
        @return              true if scale factors were computed correctly
    */
    bool SK_WARN_UNUSED_RESULT getMinMaxScales(SkScalar scaleFactors[2]) const;

    /** Decomposes SkMatrix into scale components and whatever remains. Returns false if
        SkMatrix could not be decomposed.

        Sets scale to portion of SkMatrix that scale axes. Sets remaining to SkMatrix
        with scaling factored out. remaining may be passed as nullptr
        to determine if SkMatrix can be decomposed without computing remainder.

        Returns true if scale components are found. scale and remaining are
        unchanged if SkMatrix contains perspective; scale factors are not finite, or
        are nearly zero.

        On success: Matrix = Remaining * scale.

        @param scale      axes scaling factors; may be nullptr
        @param remaining  SkMatrix without scaling; may be nullptr
        @return           true if scale can be computed

        example: https://fiddle.skia.org/c/@Matrix_decomposeScale
    */
    bool decomposeScale(SkSize* scale, SkMatrix* remaining = nullptr) const;

    /** Returns reference to const identity SkMatrix. Returned SkMatrix is set to:

            | 1 0 0 |
            | 0 1 0 |
            | 0 0 1 |

        @return  const identity SkMatrix

        example: https://fiddle.skia.org/c/@Matrix_I
    */
    static const SkMatrix& I();

    /** Returns reference to a const SkMatrix with invalid values. Returned SkMatrix is set
        to:

            | SK_ScalarMax SK_ScalarMax SK_ScalarMax |
            | SK_ScalarMax SK_ScalarMax SK_ScalarMax |
            | SK_ScalarMax SK_ScalarMax SK_ScalarMax |

        @return  const invalid SkMatrix

        example: https://fiddle.skia.org/c/@Matrix_InvalidMatrix
    */
    static const SkMatrix& InvalidMatrix();

    /** Returns SkMatrix a multiplied by SkMatrix b.

        Given:

                | A B C |      | J K L |
            a = | D E F |, b = | M N O |
                | G H I |      | P Q R |

        sets SkMatrix to:

                    | A B C |   | J K L |   | AJ+BM+CP AK+BN+CQ AL+BO+CR |
            a * b = | D E F | * | M N O | = | DJ+EM+FP DK+EN+FQ DL+EO+FR |
                    | G H I |   | P Q R |   | GJ+HM+IP GK+HN+IQ GL+HO+IR |

        @param a  SkMatrix on left side of multiply expression
        @param b  SkMatrix on right side of multiply expression
        @return   SkMatrix computed from a times b
    */
    static SkMatrix Concat(const SkMatrix& a, const SkMatrix& b) {
        SkMatrix result;
        result.setConcat(a, b);
        return result;
    }

    friend SkMatrix operator*(const SkMatrix& a, const SkMatrix& b) {
        return Concat(a, b);
    }

    /** Sets internal cache to unknown state. Use to force update after repeated
        modifications to SkMatrix element reference returned by operator[](int index).
    */
    void dirtyMatrixTypeCache() {
        this->setTypeMask(kUnknown_Mask);
    }

    /** Initializes SkMatrix with scale and translate elements.

            | sx  0 tx |
            |  0 sy ty |
            |  0  0  1 |

        @param sx  horizontal scale factor to store
        @param sy  vertical scale factor to store
        @param tx  horizontal translation to store
        @param ty  vertical translation to store
    */
    void setScaleTranslate(SkScalar sx, SkScalar sy, SkScalar tx, SkScalar ty) {
        fMat[kMScaleX] = sx;
        fMat[kMSkewX]  = 0;
        fMat[kMTransX] = tx;

        fMat[kMSkewY]  = 0;
        fMat[kMScaleY] = sy;
        fMat[kMTransY] = ty;

        fMat[kMPersp0] = 0;
        fMat[kMPersp1] = 0;
        fMat[kMPersp2] = 1;

        int mask = 0;
        if (sx != 1 || sy != 1) {
            mask |= kScale_Mask;
        }
        if (tx != 0.0f || ty != 0.0f) {
            mask |= kTranslate_Mask;
        }
        this->setTypeMask(mask | kRectStaysRect_Mask);
    }

    /** Returns true if all elements of the matrix are finite. Returns false if any
        element is infinity, or NaN.

        @return  true if matrix has only finite elements
    */
    bool isFinite() const { return SkScalarsAreFinite(fMat, 9); }

private:
    /** Set if the matrix will map a rectangle to another rectangle. This
        can be true if the matrix is scale-only, or rotates a multiple of
        90 degrees.

        This bit will be set on identity matrices
    */
    static constexpr int kRectStaysRect_Mask = 0x10;

    /** Set if the perspective bit is valid even though the rest of
        the matrix is Unknown.
    */
    static constexpr int kOnlyPerspectiveValid_Mask = 0x40;

    static constexpr int kUnknown_Mask = 0x80;

    static constexpr int kORableMasks = kTranslate_Mask |
                                        kScale_Mask |
                                        kAffine_Mask |
                                        kPerspective_Mask;

    static constexpr int kAllMasks = kTranslate_Mask |
                                     kScale_Mask |
                                     kAffine_Mask |
                                     kPerspective_Mask |
                                     kRectStaysRect_Mask;

    SkScalar        fMat[9];
    mutable int32_t fTypeMask;

    constexpr SkMatrix(SkScalar sx, SkScalar kx, SkScalar tx,
                       SkScalar ky, SkScalar sy, SkScalar ty,
                       SkScalar p0, SkScalar p1, SkScalar p2, int typeMask)
        : fMat{sx, kx, tx,
               ky, sy, ty,
               p0, p1, p2}
        , fTypeMask(typeMask) {}

    static void ComputeInv(SkScalar dst[9], const SkScalar src[9], double invDet, bool isPersp);

    uint8_t computeTypeMask() const;
    uint8_t computePerspectiveTypeMask() const;

    void setTypeMask(int mask) {
        // allow kUnknown or a valid mask
        SkASSERT(kUnknown_Mask == mask || (mask & kAllMasks) == mask ||
                 ((kUnknown_Mask | kOnlyPerspectiveValid_Mask) & mask)
                 == (kUnknown_Mask | kOnlyPerspectiveValid_Mask));
        fTypeMask = mask;
    }

    void orTypeMask(int mask) {
        SkASSERT((mask & kORableMasks) == mask);
        fTypeMask |= mask;
    }

    void clearTypeMask(int mask) {
        // only allow a valid mask
        SkASSERT((mask & kAllMasks) == mask);
        fTypeMask &= ~mask;
    }

    TypeMask getPerspectiveTypeMaskOnly() const {
        if ((fTypeMask & kUnknown_Mask) &&
            !(fTypeMask & kOnlyPerspectiveValid_Mask)) {
            fTypeMask = this->computePerspectiveTypeMask();
        }
        return (TypeMask)(fTypeMask & 0xF);
    }

    /** Returns true if we already know that the matrix is identity;
        false otherwise.
    */
    bool isTriviallyIdentity() const {
        if (fTypeMask & kUnknown_Mask) {
            return false;
        }
        return ((fTypeMask & 0xF) == 0);
    }

    inline void updateTranslateMask() {
        if ((fMat[kMTransX] != 0) | (fMat[kMTransY] != 0)) {
            fTypeMask |= kTranslate_Mask;
        } else {
            fTypeMask &= ~kTranslate_Mask;
        }
    }

    typedef void (*MapXYProc)(const SkMatrix& mat, SkScalar x, SkScalar y,
                                 SkPoint* result);

    static MapXYProc GetMapXYProc(TypeMask mask) {
        SkASSERT((mask & ~kAllMasks) == 0);
        return gMapXYProcs[mask & kAllMasks];
    }

    MapXYProc getMapXYProc() const {
        return GetMapXYProc(this->getType());
    }

    typedef void (*MapPtsProc)(const SkMatrix& mat, SkPoint dst[],
                                  const SkPoint src[], int count);

    static MapPtsProc GetMapPtsProc(TypeMask mask) {
        SkASSERT((mask & ~kAllMasks) == 0);
        return gMapPtsProcs[mask & kAllMasks];
    }

    MapPtsProc getMapPtsProc() const {
        return GetMapPtsProc(this->getType());
    }

    bool SK_WARN_UNUSED_RESULT invertNonIdentity(SkMatrix* inverse) const;

    static bool Poly2Proc(const SkPoint[], SkMatrix*);
    static bool Poly3Proc(const SkPoint[], SkMatrix*);
    static bool Poly4Proc(const SkPoint[], SkMatrix*);

    static void Identity_xy(const SkMatrix&, SkScalar, SkScalar, SkPoint*);
    static void Trans_xy(const SkMatrix&, SkScalar, SkScalar, SkPoint*);
    static void Scale_xy(const SkMatrix&, SkScalar, SkScalar, SkPoint*);
    static void ScaleTrans_xy(const SkMatrix&, SkScalar, SkScalar, SkPoint*);
    static void Rot_xy(const SkMatrix&, SkScalar, SkScalar, SkPoint*);
    static void RotTrans_xy(const SkMatrix&, SkScalar, SkScalar, SkPoint*);
    static void Persp_xy(const SkMatrix&, SkScalar, SkScalar, SkPoint*);

    static const MapXYProc gMapXYProcs[];

    static void Identity_pts(const SkMatrix&, SkPoint[], const SkPoint[], int);
    static void Trans_pts(const SkMatrix&, SkPoint dst[], const SkPoint[], int);
    static void Scale_pts(const SkMatrix&, SkPoint dst[], const SkPoint[], int);
    static void ScaleTrans_pts(const SkMatrix&, SkPoint dst[], const SkPoint[],
                               int count);
    static void Persp_pts(const SkMatrix&, SkPoint dst[], const SkPoint[], int);

    static void Affine_vpts(const SkMatrix&, SkPoint dst[], const SkPoint[], int);

    static const MapPtsProc gMapPtsProcs[];

    // return the number of bytes written, whether or not buffer is null
    size_t writeToMemory(void* buffer) const;
    /**
     * Reads data from the buffer parameter
     *
     * @param buffer Memory to read from
     * @param length Amount of memory available in the buffer
     * @return number of bytes read (must be a multiple of 4) or
     *         0 if there was not enough memory available
     */
    size_t readFromMemory(const void* buffer, size_t length);

    // legacy method -- still needed? why not just postScale(1/divx, ...)?
    bool postIDiv(int divx, int divy);
    void doNormalizePerspective();

    friend class SkPerspIter;
    friend class SkMatrixPriv;
    friend class SerializationTest;
};
SK_END_REQUIRE_DENSE

#endif
