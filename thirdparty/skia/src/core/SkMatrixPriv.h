/*
 * Copyright 2016 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkMatrixPriv_DEFINE
#define SkMatrixPriv_DEFINE

#include "include/core/SkM44.h"
#include "include/core/SkMatrix.h"
#include "include/private/SkVx.h"
#include "src/core/SkPointPriv.h"

class SkMatrixPriv {
public:
    enum {
        // writeTo/readFromMemory will never return a value larger than this
        kMaxFlattenSize = 9 * sizeof(SkScalar) + sizeof(uint32_t),
    };

    static size_t WriteToMemory(const SkMatrix& matrix, void* buffer) {
        return matrix.writeToMemory(buffer);
    }

    static size_t ReadFromMemory(SkMatrix* matrix, const void* buffer, size_t length) {
        return matrix->readFromMemory(buffer, length);
    }

    typedef SkMatrix::MapXYProc MapXYProc;
    typedef SkMatrix::MapPtsProc MapPtsProc;


    static MapPtsProc GetMapPtsProc(const SkMatrix& matrix) {
        return SkMatrix::GetMapPtsProc(matrix.getType());
    }

    static MapXYProc GetMapXYProc(const SkMatrix& matrix) {
        return SkMatrix::GetMapXYProc(matrix.getType());
    }

    /**
     *  Attempt to map the rect through the inverse of the matrix. If it is not invertible,
     *  then this returns false and dst is unchanged.
     */
    static bool SK_WARN_UNUSED_RESULT InverseMapRect(const SkMatrix& mx,
                                                     SkRect* dst, const SkRect& src) {
        if (mx.getType() <= SkMatrix::kTranslate_Mask) {
            SkScalar tx = mx.getTranslateX();
            SkScalar ty = mx.getTranslateY();
            skvx::float4 trans(tx, ty, tx, ty);
            (skvx::float4::Load(&src.fLeft) - trans).store(&dst->fLeft);
            return true;
        }
        // Insert other special-cases here (e.g. scale+translate)

        // general case
        SkMatrix inverse;
        if (mx.invert(&inverse)) {
            inverse.mapRect(dst, src);
            return true;
        }
        return false;
    }

    /** Maps count pts, skipping stride bytes to advance from one SkPoint to the next.
        Points are mapped by multiplying each SkPoint by SkMatrix. Given:

                     | A B C |        | x |
            Matrix = | D E F |,  pt = | y |
                     | G H I |        | 1 |

        each resulting pts SkPoint is computed as:

                          |A B C| |x|                               Ax+By+C   Dx+Ey+F
            Matrix * pt = |D E F| |y| = |Ax+By+C Dx+Ey+F Gx+Hy+I| = ------- , -------
                          |G H I| |1|                               Gx+Hy+I   Gx+Hy+I

        @param mx      matrix used to map the points
        @param pts     storage for mapped points
        @param stride  size of record starting with SkPoint, in bytes
        @param count   number of points to transform
    */
    static void MapPointsWithStride(const SkMatrix& mx, SkPoint pts[], size_t stride, int count) {
        SkASSERT(stride >= sizeof(SkPoint));
        SkASSERT(0 == stride % sizeof(SkScalar));

        SkMatrix::TypeMask tm = mx.getType();

        if (SkMatrix::kIdentity_Mask == tm) {
            return;
        }
        if (SkMatrix::kTranslate_Mask == tm) {
            const SkScalar tx = mx.getTranslateX();
            const SkScalar ty = mx.getTranslateY();
            skvx::float2 trans(tx, ty);
            for (int i = 0; i < count; ++i) {
                (skvx::float2::Load(&pts->fX) + trans).store(&pts->fX);
                pts = (SkPoint*)((intptr_t)pts + stride);
            }
            return;
        }
        // Insert other special-cases here (e.g. scale+translate)

        // general case
        SkMatrix::MapXYProc proc = mx.getMapXYProc();
        for (int i = 0; i < count; ++i) {
            proc(mx, pts->fX, pts->fY, pts);
            pts = (SkPoint*)((intptr_t)pts + stride);
        }
    }

    /** Maps src SkPoint array of length count to dst SkPoint array, skipping stride bytes
        to advance from one SkPoint to the next.
        Points are mapped by multiplying each SkPoint by SkMatrix. Given:

                     | A B C |         | x |
            Matrix = | D E F |,  src = | y |
                     | G H I |         | 1 |

        each resulting dst SkPoint is computed as:

                          |A B C| |x|                               Ax+By+C   Dx+Ey+F
            Matrix * pt = |D E F| |y| = |Ax+By+C Dx+Ey+F Gx+Hy+I| = ------- , -------
                          |G H I| |1|                               Gx+Hy+I   Gx+Hy+I

        @param mx      matrix used to map the points
        @param dst     storage for mapped points
        @param src     points to transform
        @param stride  size of record starting with SkPoint, in bytes
        @param count   number of points to transform
    */
    static void MapPointsWithStride(const SkMatrix& mx, SkPoint dst[], size_t dstStride,
                                    const SkPoint src[], size_t srcStride, int count) {
        SkASSERT(srcStride >= sizeof(SkPoint));
        SkASSERT(dstStride >= sizeof(SkPoint));
        SkASSERT(0 == srcStride % sizeof(SkScalar));
        SkASSERT(0 == dstStride % sizeof(SkScalar));
        for (int i = 0; i < count; ++i) {
            mx.mapPoints(dst, src, 1);
            src = (SkPoint*)((intptr_t)src + srcStride);
            dst = (SkPoint*)((intptr_t)dst + dstStride);
        }
    }

    static void MapHomogeneousPointsWithStride(const SkMatrix& mx, SkPoint3 dst[], size_t dstStride,
                                               const SkPoint3 src[], size_t srcStride, int count);

    static bool PostIDiv(SkMatrix* matrix, int divx, int divy) {
        return matrix->postIDiv(divx, divy);
    }

    static bool CheapEqual(const SkMatrix& a, const SkMatrix& b) {
        return &a == &b || 0 == memcmp(a.fMat, b.fMat, sizeof(a.fMat));
    }

    static const SkScalar* M44ColMajor(const SkM44& m) { return m.fMat; }

    // This is legacy functionality that only checks the 3x3 portion. The matrix could have Z-based
    // shear, or other complex behavior. Only use this if you're planning to use the information
    // to accelerate some purely 2D operation.
    static bool IsScaleTranslateAsM33(const SkM44& m) {
        return m.rc(1,0) == 0 && m.rc(3,0) == 0 &&
               m.rc(0,1) == 0 && m.rc(3,1) == 0 &&
               m.rc(3,3) == 1;

    }

    // Map the four corners of 'r' and return the bounding box of those points. The four corners of
    // 'r' are assumed to have z = 0 and w = 1. If the matrix has perspective, the returned
    // rectangle will be the bounding box of the projected points after being clipped to w > 0.
    static SkRect MapRect(const SkM44& m, const SkRect& r);

    // Returns the differential area scale factor for a local point 'p' that will be transformed
    // by 'm' (which may have perspective). If 'm' does not have perspective, this scale factor is
    // constant regardless of 'p'; when it does have perspective, it is specific to that point.
    //
    // This can be crudely thought of as "device pixel area" / "local pixel area" at 'p'.
    //
    // Returns positive infinity if the transformed homogeneous point has w <= 0.
    static SkScalar DifferentialAreaScale(const SkMatrix& m, const SkPoint& p);
};

#endif
