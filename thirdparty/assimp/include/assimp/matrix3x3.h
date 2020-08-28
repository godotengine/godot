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

/** @file matrix3x3.h
 *  @brief Definition of a 3x3 matrix, including operators when compiling in C++
 */
#pragma once
#ifndef AI_MATRIX3X3_H_INC
#define AI_MATRIX3X3_H_INC

#ifdef __GNUC__
#   pragma GCC system_header
#endif

#include <assimp/defs.h>

#ifdef __cplusplus

template <typename T> class aiMatrix4x4t;
template <typename T> class aiVector2t;

// ---------------------------------------------------------------------------
/** @brief Represents a row-major 3x3 matrix
 *
 *  There's much confusion about matrix layouts (column vs. row order).
 *  This is *always* a row-major matrix. Not even with the
 *  #aiProcess_ConvertToLeftHanded flag, which absolutely does not affect
 *  matrix order - it just affects the handedness of the coordinate system
 *  defined thereby.
 */
template <typename TReal>
class aiMatrix3x3t {
public:
    aiMatrix3x3t() AI_NO_EXCEPT :
        a1(static_cast<TReal>(1.0f)), a2(), a3(),
        b1(), b2(static_cast<TReal>(1.0f)), b3(),
        c1(), c2(), c3(static_cast<TReal>(1.0f)) {}

    aiMatrix3x3t (  TReal _a1, TReal _a2, TReal _a3,
                    TReal _b1, TReal _b2, TReal _b3,
                    TReal _c1, TReal _c2, TReal _c3) :
        a1(_a1), a2(_a2), a3(_a3),
        b1(_b1), b2(_b2), b3(_b3),
        c1(_c1), c2(_c2), c3(_c3)
    {}

    // matrix multiplication.
    aiMatrix3x3t& operator *= (const aiMatrix3x3t& m);
    aiMatrix3x3t  operator  * (const aiMatrix3x3t& m) const;

    // array access operators
    TReal* operator[]       (unsigned int p_iIndex);
    const TReal* operator[] (unsigned int p_iIndex) const;

    // comparison operators
    bool operator== (const aiMatrix4x4t<TReal>& m) const;
    bool operator!= (const aiMatrix4x4t<TReal>& m) const;

    bool Equal(const aiMatrix4x4t<TReal>& m, TReal epsilon = 1e-6) const;

    template <typename TOther>
    operator aiMatrix3x3t<TOther> () const;

    // -------------------------------------------------------------------
    /** @brief Construction from a 4x4 matrix. The remaining parts
     *  of the matrix are ignored.
     */
    explicit aiMatrix3x3t( const aiMatrix4x4t<TReal>& pMatrix);

    // -------------------------------------------------------------------
    /** @brief Transpose the matrix
     */
    aiMatrix3x3t& Transpose();

    // -------------------------------------------------------------------
    /** @brief Invert the matrix.
     *  If the matrix is not invertible all elements are set to qnan.
     *  Beware, use (f != f) to check whether a TReal f is qnan.
     */
    aiMatrix3x3t& Inverse();
    TReal Determinant() const;

    // -------------------------------------------------------------------
    /** @brief Returns a rotation matrix for a rotation around z
     *  @param a Rotation angle, in radians
     *  @param out Receives the output matrix
     *  @return Reference to the output matrix
     */
    static aiMatrix3x3t& RotationZ(TReal a, aiMatrix3x3t& out);

    // -------------------------------------------------------------------
    /** @brief Returns a rotation matrix for a rotation around
     *    an arbitrary axis.
     *
     *  @param a Rotation angle, in radians
     *  @param axis Axis to rotate around
     *  @param out To be filled
     */
    static aiMatrix3x3t& Rotation( TReal a,
        const aiVector3t<TReal>& axis, aiMatrix3x3t& out);

    // -------------------------------------------------------------------
    /** @brief Returns a translation matrix
     *  @param v Translation vector
     *  @param out Receives the output matrix
     *  @return Reference to the output matrix
     */
    static aiMatrix3x3t& Translation( const aiVector2t<TReal>& v, aiMatrix3x3t& out);

    // -------------------------------------------------------------------
    /** @brief A function for creating a rotation matrix that rotates a
     *  vector called "from" into another vector called "to".
     * Input : from[3], to[3] which both must be *normalized* non-zero vectors
     * Output: mtx[3][3] -- a 3x3 matrix in column-major form
     * Authors: Tomas Möller, John Hughes
     *          "Efficiently Building a Matrix to Rotate One Vector to Another"
     *          Journal of Graphics Tools, 4(4):1-4, 1999
     */
    static aiMatrix3x3t& FromToMatrix(const aiVector3t<TReal>& from,
        const aiVector3t<TReal>& to, aiMatrix3x3t& out);

public:
    TReal a1, a2, a3;
    TReal b1, b2, b3;
    TReal c1, c2, c3;
};

typedef aiMatrix3x3t<ai_real> aiMatrix3x3;

#else

struct aiMatrix3x3 {
    ai_real a1, a2, a3;
    ai_real b1, b2, b3;
    ai_real c1, c2, c3;
};

#endif // __cplusplus

#endif // AI_MATRIX3X3_H_INC
