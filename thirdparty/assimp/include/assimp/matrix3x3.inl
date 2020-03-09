/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team



All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the following
conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
---------------------------------------------------------------------------
*/

/** @file matrix3x3.inl
 *  @brief Inline implementation of the 3x3 matrix operators
 */
#pragma once
#ifndef AI_MATRIX3X3_INL_INC
#define AI_MATRIX3X3_INL_INC

#ifdef __GNUC__
#   pragma GCC system_header
#endif

#ifdef __cplusplus
#include <assimp/matrix3x3.h>
#include <assimp/matrix4x4.h>

#include <algorithm>
#include <cmath>
#include <limits>

// ------------------------------------------------------------------------------------------------
// Construction from a 4x4 matrix. The remaining parts of the matrix are ignored.
template <typename TReal>
AI_FORCE_INLINE
aiMatrix3x3t<TReal>::aiMatrix3x3t( const aiMatrix4x4t<TReal>& pMatrix) {
    a1 = pMatrix.a1; a2 = pMatrix.a2; a3 = pMatrix.a3;
    b1 = pMatrix.b1; b2 = pMatrix.b2; b3 = pMatrix.b3;
    c1 = pMatrix.c1; c2 = pMatrix.c2; c3 = pMatrix.c3;
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
aiMatrix3x3t<TReal>& aiMatrix3x3t<TReal>::operator *= (const aiMatrix3x3t<TReal>& m) {
    *this = aiMatrix3x3t<TReal>(m.a1 * a1 + m.b1 * a2 + m.c1 * a3,
        m.a2 * a1 + m.b2 * a2 + m.c2 * a3,
        m.a3 * a1 + m.b3 * a2 + m.c3 * a3,
        m.a1 * b1 + m.b1 * b2 + m.c1 * b3,
        m.a2 * b1 + m.b2 * b2 + m.c2 * b3,
        m.a3 * b1 + m.b3 * b2 + m.c3 * b3,
        m.a1 * c1 + m.b1 * c2 + m.c1 * c3,
        m.a2 * c1 + m.b2 * c2 + m.c2 * c3,
        m.a3 * c1 + m.b3 * c2 + m.c3 * c3);
    return *this;
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
template <typename TOther>
aiMatrix3x3t<TReal>::operator aiMatrix3x3t<TOther> () const {
    return aiMatrix3x3t<TOther>(static_cast<TOther>(a1),static_cast<TOther>(a2),static_cast<TOther>(a3),
        static_cast<TOther>(b1),static_cast<TOther>(b2),static_cast<TOther>(b3),
        static_cast<TOther>(c1),static_cast<TOther>(c2),static_cast<TOther>(c3));
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
aiMatrix3x3t<TReal> aiMatrix3x3t<TReal>::operator* (const aiMatrix3x3t<TReal>& m) const {
    aiMatrix3x3t<TReal> temp( *this);
    temp *= m;
    return temp;
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
TReal* aiMatrix3x3t<TReal>::operator[] (unsigned int p_iIndex) {
    switch ( p_iIndex ) {
        case 0:
            return &a1;
        case 1:
            return &b1;
        case 2:
            return &c1;
        default:
            break;
    }
    return &a1;
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
const TReal* aiMatrix3x3t<TReal>::operator[] (unsigned int p_iIndex) const {
    switch ( p_iIndex ) {
        case 0:
            return &a1;
        case 1:
            return &b1;
        case 2:
            return &c1;
        default:
            break;
    }
    return &a1;
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
bool aiMatrix3x3t<TReal>::operator== (const aiMatrix4x4t<TReal>& m) const {
    return a1 == m.a1 && a2 == m.a2 && a3 == m.a3 &&
           b1 == m.b1 && b2 == m.b2 && b3 == m.b3 &&
           c1 == m.c1 && c2 == m.c2 && c3 == m.c3;
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
bool aiMatrix3x3t<TReal>::operator!= (const aiMatrix4x4t<TReal>& m) const {
    return !(*this == m);
}

// ---------------------------------------------------------------------------
template<typename TReal>
AI_FORCE_INLINE
bool aiMatrix3x3t<TReal>::Equal(const aiMatrix4x4t<TReal>& m, TReal epsilon) const {
    return
        std::abs(a1 - m.a1) <= epsilon &&
        std::abs(a2 - m.a2) <= epsilon &&
        std::abs(a3 - m.a3) <= epsilon &&
        std::abs(b1 - m.b1) <= epsilon &&
        std::abs(b2 - m.b2) <= epsilon &&
        std::abs(b3 - m.b3) <= epsilon &&
        std::abs(c1 - m.c1) <= epsilon &&
        std::abs(c2 - m.c2) <= epsilon &&
        std::abs(c3 - m.c3) <= epsilon;
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
aiMatrix3x3t<TReal>& aiMatrix3x3t<TReal>::Transpose() {
    // (TReal&) don't remove, GCC complains cause of packed fields
    std::swap( (TReal&)a2, (TReal&)b1);
    std::swap( (TReal&)a3, (TReal&)c1);
    std::swap( (TReal&)b3, (TReal&)c2);
    return *this;
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
TReal aiMatrix3x3t<TReal>::Determinant() const {
    return a1*b2*c3 - a1*b3*c2 + a2*b3*c1 - a2*b1*c3 + a3*b1*c2 - a3*b2*c1;
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
aiMatrix3x3t<TReal>& aiMatrix3x3t<TReal>::Inverse() {
    // Compute the reciprocal determinant
    TReal det = Determinant();
    if(det == static_cast<TReal>(0.0))
    {
        // Matrix not invertible. Setting all elements to nan is not really
        // correct in a mathematical sense; but at least qnans are easy to
        // spot. XXX we might throw an exception instead, which would
        // be even much better to spot :/.
        const TReal nan = std::numeric_limits<TReal>::quiet_NaN();
        *this = aiMatrix3x3t<TReal>( nan,nan,nan,nan,nan,nan,nan,nan,nan);

        return *this;
    }

    TReal invdet = static_cast<TReal>(1.0) / det;

    aiMatrix3x3t<TReal> res;
    res.a1 = invdet  * (b2 * c3 - b3 * c2);
    res.a2 = -invdet * (a2 * c3 - a3 * c2);
    res.a3 = invdet  * (a2 * b3 - a3 * b2);
    res.b1 = -invdet * (b1 * c3 - b3 * c1);
    res.b2 = invdet  * (a1 * c3 - a3 * c1);
    res.b3 = -invdet * (a1 * b3 - a3 * b1);
    res.c1 = invdet  * (b1 * c2 - b2 * c1);
    res.c2 = -invdet * (a1 * c2 - a2 * c1);
    res.c3 = invdet  * (a1 * b2 - a2 * b1);
    *this = res;

    return *this;
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
aiMatrix3x3t<TReal>& aiMatrix3x3t<TReal>::RotationZ(TReal a, aiMatrix3x3t<TReal>& out) {
    out.a1 = out.b2 = std::cos(a);
    out.b1 = std::sin(a);
    out.a2 = - out.b1;

    out.a3 = out.b3 = out.c1 = out.c2 = 0.f;
    out.c3 = 1.f;

    return out;
}

// ------------------------------------------------------------------------------------------------
// Returns a rotation matrix for a rotation around an arbitrary axis.
template <typename TReal>
AI_FORCE_INLINE
aiMatrix3x3t<TReal>& aiMatrix3x3t<TReal>::Rotation( TReal a, const aiVector3t<TReal>& axis, aiMatrix3x3t<TReal>& out) {
  TReal c = std::cos( a), s = std::sin( a), t = 1 - c;
  TReal x = axis.x, y = axis.y, z = axis.z;

  // Many thanks to MathWorld and Wikipedia
  out.a1 = t*x*x + c;   out.a2 = t*x*y - s*z; out.a3 = t*x*z + s*y;
  out.b1 = t*x*y + s*z; out.b2 = t*y*y + c;   out.b3 = t*y*z - s*x;
  out.c1 = t*x*z - s*y; out.c2 = t*y*z + s*x; out.c3 = t*z*z + c;

  return out;
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
aiMatrix3x3t<TReal>& aiMatrix3x3t<TReal>::Translation( const aiVector2t<TReal>& v, aiMatrix3x3t<TReal>& out) {
    out = aiMatrix3x3t<TReal>();
    out.a3 = v.x;
    out.b3 = v.y;
    return out;
}

// ----------------------------------------------------------------------------------------
/** A function for creating a rotation matrix that rotates a vector called
 * "from" into another vector called "to".
 * Input : from[3], to[3] which both must be *normalized* non-zero vectors
 * Output: mtx[3][3] -- a 3x3 matrix in colum-major form
 * Authors: Tomas Möller, John Hughes
 *          "Efficiently Building a Matrix to Rotate One Vector to Another"
 *          Journal of Graphics Tools, 4(4):1-4, 1999
 */
// ----------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE aiMatrix3x3t<TReal>& aiMatrix3x3t<TReal>::FromToMatrix(const aiVector3t<TReal>& from,
        const aiVector3t<TReal>& to, aiMatrix3x3t<TReal>& mtx) {
    const TReal e = from * to;
    const TReal f = (e < 0)? -e:e;

    if (f > static_cast<TReal>(1.0) - static_cast<TReal>(0.00001))     /* "from" and "to"-vector almost parallel */
    {
        aiVector3D u,v;     /* temporary storage vectors */
        aiVector3D x;       /* vector most nearly orthogonal to "from" */

        x.x = (from.x > 0.0)? from.x : -from.x;
        x.y = (from.y > 0.0)? from.y : -from.y;
        x.z = (from.z > 0.0)? from.z : -from.z;

        if (x.x < x.y)
        {
            if (x.x < x.z)
            {
                x.x = static_cast<TReal>(1.0);
                x.y = x.z = static_cast<TReal>(0.0);
            }
            else
            {
                x.z = static_cast<TReal>(1.0);
                x.x = x.y = static_cast<TReal>(0.0);
            }
        }
        else
        {
            if (x.y < x.z)
            {
                x.y = static_cast<TReal>(1.0);
                x.x = x.z = static_cast<TReal>(0.0);
            }
            else
            {
                x.z = static_cast<TReal>(1.0);
                x.x = x.y = static_cast<TReal>(0.0);
            }
        }

        u.x = x.x - from.x; u.y = x.y - from.y; u.z = x.z - from.z;
        v.x = x.x - to.x;   v.y = x.y - to.y;   v.z = x.z - to.z;

        const TReal c1_ = static_cast<TReal>(2.0) / (u * u);
        const TReal c2_ = static_cast<TReal>(2.0) / (v * v);
        const TReal c3_ = c1_ * c2_  * (u * v);

        for (unsigned int i = 0; i < 3; i++)
        {
            for (unsigned int j = 0; j < 3; j++)
            {
                mtx[i][j] =  - c1_ * u[i] * u[j] - c2_ * v[i] * v[j]
                    + c3_ * v[i] * u[j];
            }
            mtx[i][i] += static_cast<TReal>(1.0);
        }
    }
    else  /* the most common case, unless "from"="to", or "from"=-"to" */
    {
        const aiVector3D v = from ^ to;
        /* ... use this hand optimized version (9 mults less) */
        const TReal h = static_cast<TReal>(1.0)/(static_cast<TReal>(1.0) + e);      /* optimization by Gottfried Chen */
        const TReal hvx = h * v.x;
        const TReal hvz = h * v.z;
        const TReal hvxy = hvx * v.y;
        const TReal hvxz = hvx * v.z;
        const TReal hvyz = hvz * v.y;
        mtx[0][0] = e + hvx * v.x;
        mtx[0][1] = hvxy - v.z;
        mtx[0][2] = hvxz + v.y;

        mtx[1][0] = hvxy + v.z;
        mtx[1][1] = e + h * v.y * v.y;
        mtx[1][2] = hvyz - v.x;

        mtx[2][0] = hvxz - v.y;
        mtx[2][1] = hvyz + v.x;
        mtx[2][2] = e + hvz * v.z;
    }
    return mtx;
}

#endif // __cplusplus
#endif // AI_MATRIX3X3_INL_INC
