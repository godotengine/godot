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

/** @file matrix4x4.inl
 *  @brief Inline implementation of the 4x4 matrix operators
 */
#pragma once
#ifndef AI_MATRIX4X4_INL_INC
#define AI_MATRIX4X4_INL_INC

#ifdef __cplusplus

#include "matrix4x4.h"
#include "matrix3x3.h"
#include "quaternion.h"

#include <algorithm>
#include <limits>
#include <cmath>

// ----------------------------------------------------------------------------------------
template <typename TReal>
aiMatrix4x4t<TReal>::aiMatrix4x4t() AI_NO_EXCEPT :
    a1(1.0f), a2(), a3(), a4(),
    b1(), b2(1.0f), b3(), b4(),
    c1(), c2(), c3(1.0f), c4(),
    d1(), d2(), d3(), d4(1.0f)
{

}

// ----------------------------------------------------------------------------------------
template <typename TReal>
aiMatrix4x4t<TReal>::aiMatrix4x4t (TReal _a1, TReal _a2, TReal _a3, TReal _a4,
              TReal _b1, TReal _b2, TReal _b3, TReal _b4,
              TReal _c1, TReal _c2, TReal _c3, TReal _c4,
              TReal _d1, TReal _d2, TReal _d3, TReal _d4) :
    a1(_a1), a2(_a2), a3(_a3), a4(_a4),
    b1(_b1), b2(_b2), b3(_b3), b4(_b4),
    c1(_c1), c2(_c2), c3(_c3), c4(_c4),
    d1(_d1), d2(_d2), d3(_d3), d4(_d4)
{

}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
template <typename TOther>
aiMatrix4x4t<TReal>::operator aiMatrix4x4t<TOther> () const
{
    return aiMatrix4x4t<TOther>(static_cast<TOther>(a1),static_cast<TOther>(a2),static_cast<TOther>(a3),static_cast<TOther>(a4),
        static_cast<TOther>(b1),static_cast<TOther>(b2),static_cast<TOther>(b3),static_cast<TOther>(b4),
        static_cast<TOther>(c1),static_cast<TOther>(c2),static_cast<TOther>(c3),static_cast<TOther>(c4),
        static_cast<TOther>(d1),static_cast<TOther>(d2),static_cast<TOther>(d3),static_cast<TOther>(d4));
}


// ----------------------------------------------------------------------------------------
template <typename TReal>
inline aiMatrix4x4t<TReal>::aiMatrix4x4t (const aiMatrix3x3t<TReal>& m)
{
    a1 = m.a1; a2 = m.a2; a3 = m.a3; a4 = static_cast<TReal>(0.0);
    b1 = m.b1; b2 = m.b2; b3 = m.b3; b4 = static_cast<TReal>(0.0);
    c1 = m.c1; c2 = m.c2; c3 = m.c3; c4 = static_cast<TReal>(0.0);
    d1 = static_cast<TReal>(0.0); d2 = static_cast<TReal>(0.0); d3 = static_cast<TReal>(0.0); d4 = static_cast<TReal>(1.0);
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
inline aiMatrix4x4t<TReal>::aiMatrix4x4t (const aiVector3t<TReal>& scaling, const aiQuaterniont<TReal>& rotation, const aiVector3t<TReal>& position)
{
    // build a 3x3 rotation matrix
    aiMatrix3x3t<TReal> m = rotation.GetMatrix();

    a1 = m.a1 * scaling.x;
    a2 = m.a2 * scaling.x;
    a3 = m.a3 * scaling.x;
    a4 = position.x;

    b1 = m.b1 * scaling.y;
    b2 = m.b2 * scaling.y;
    b3 = m.b3 * scaling.y;
    b4 = position.y;

    c1 = m.c1 * scaling.z;
    c2 = m.c2 * scaling.z;
    c3 = m.c3 * scaling.z;
    c4= position.z;

    d1 = static_cast<TReal>(0.0);
    d2 = static_cast<TReal>(0.0);
    d3 = static_cast<TReal>(0.0);
    d4 = static_cast<TReal>(1.0);
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
inline aiMatrix4x4t<TReal>& aiMatrix4x4t<TReal>::operator *= (const aiMatrix4x4t<TReal>& m)
{
    *this = aiMatrix4x4t<TReal>(
        m.a1 * a1 + m.b1 * a2 + m.c1 * a3 + m.d1 * a4,
        m.a2 * a1 + m.b2 * a2 + m.c2 * a3 + m.d2 * a4,
        m.a3 * a1 + m.b3 * a2 + m.c3 * a3 + m.d3 * a4,
        m.a4 * a1 + m.b4 * a2 + m.c4 * a3 + m.d4 * a4,
        m.a1 * b1 + m.b1 * b2 + m.c1 * b3 + m.d1 * b4,
        m.a2 * b1 + m.b2 * b2 + m.c2 * b3 + m.d2 * b4,
        m.a3 * b1 + m.b3 * b2 + m.c3 * b3 + m.d3 * b4,
        m.a4 * b1 + m.b4 * b2 + m.c4 * b3 + m.d4 * b4,
        m.a1 * c1 + m.b1 * c2 + m.c1 * c3 + m.d1 * c4,
        m.a2 * c1 + m.b2 * c2 + m.c2 * c3 + m.d2 * c4,
        m.a3 * c1 + m.b3 * c2 + m.c3 * c3 + m.d3 * c4,
        m.a4 * c1 + m.b4 * c2 + m.c4 * c3 + m.d4 * c4,
        m.a1 * d1 + m.b1 * d2 + m.c1 * d3 + m.d1 * d4,
        m.a2 * d1 + m.b2 * d2 + m.c2 * d3 + m.d2 * d4,
        m.a3 * d1 + m.b3 * d2 + m.c3 * d3 + m.d3 * d4,
        m.a4 * d1 + m.b4 * d2 + m.c4 * d3 + m.d4 * d4);
    return *this;
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
inline aiMatrix4x4t<TReal> aiMatrix4x4t<TReal>::operator* (const TReal& aFloat) const
{
    aiMatrix4x4t<TReal> temp(
        a1 * aFloat,
        a2 * aFloat,
        a3 * aFloat,
        a4 * aFloat,
        b1 * aFloat,
        b2 * aFloat,
        b3 * aFloat,
        b4 * aFloat,
        c1 * aFloat,
        c2 * aFloat,
        c3 * aFloat,
        c4 * aFloat,
        d1 * aFloat,
        d2 * aFloat,
        d3 * aFloat,
        d4 * aFloat);
    return temp;
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
inline aiMatrix4x4t<TReal> aiMatrix4x4t<TReal>::operator+ (const aiMatrix4x4t<TReal>& m) const
{
    aiMatrix4x4t<TReal> temp(
        m.a1 + a1,
        m.a2 + a2,
        m.a3 + a3,
        m.a4 + a4,
        m.b1 + b1,
        m.b2 + b2,
        m.b3 + b3,
        m.b4 + b4,
        m.c1 + c1,
        m.c2 + c2,
        m.c3 + c3,
        m.c4 + c4,
        m.d1 + d1,
        m.d2 + d2,
        m.d3 + d3,
        m.d4 + d4);
    return temp;
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
inline aiMatrix4x4t<TReal> aiMatrix4x4t<TReal>::operator* (const aiMatrix4x4t<TReal>& m) const
{
    aiMatrix4x4t<TReal> temp( *this);
    temp *= m;
    return temp;
}


// ----------------------------------------------------------------------------------------
template <typename TReal>
inline aiMatrix4x4t<TReal>& aiMatrix4x4t<TReal>::Transpose()
{
    // (TReal&) don't remove, GCC complains cause of packed fields
    std::swap( (TReal&)b1, (TReal&)a2);
    std::swap( (TReal&)c1, (TReal&)a3);
    std::swap( (TReal&)c2, (TReal&)b3);
    std::swap( (TReal&)d1, (TReal&)a4);
    std::swap( (TReal&)d2, (TReal&)b4);
    std::swap( (TReal&)d3, (TReal&)c4);
    return *this;
}


// ----------------------------------------------------------------------------------------
template <typename TReal>
inline TReal aiMatrix4x4t<TReal>::Determinant() const
{
    return a1*b2*c3*d4 - a1*b2*c4*d3 + a1*b3*c4*d2 - a1*b3*c2*d4
        + a1*b4*c2*d3 - a1*b4*c3*d2 - a2*b3*c4*d1 + a2*b3*c1*d4
        - a2*b4*c1*d3 + a2*b4*c3*d1 - a2*b1*c3*d4 + a2*b1*c4*d3
        + a3*b4*c1*d2 - a3*b4*c2*d1 + a3*b1*c2*d4 - a3*b1*c4*d2
        + a3*b2*c4*d1 - a3*b2*c1*d4 - a4*b1*c2*d3 + a4*b1*c3*d2
        - a4*b2*c3*d1 + a4*b2*c1*d3 - a4*b3*c1*d2 + a4*b3*c2*d1;
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
inline aiMatrix4x4t<TReal>& aiMatrix4x4t<TReal>::Inverse()
{
    // Compute the reciprocal determinant
    const TReal det = Determinant();
    if(det == static_cast<TReal>(0.0))
    {
        // Matrix not invertible. Setting all elements to nan is not really
        // correct in a mathematical sense but it is easy to debug for the
        // programmer.
        const TReal nan = std::numeric_limits<TReal>::quiet_NaN();
        *this = aiMatrix4x4t<TReal>(
            nan,nan,nan,nan,
            nan,nan,nan,nan,
            nan,nan,nan,nan,
            nan,nan,nan,nan);

        return *this;
    }

    const TReal invdet = static_cast<TReal>(1.0) / det;

    aiMatrix4x4t<TReal> res;
    res.a1 = invdet  * (b2 * (c3 * d4 - c4 * d3) + b3 * (c4 * d2 - c2 * d4) + b4 * (c2 * d3 - c3 * d2));
    res.a2 = -invdet * (a2 * (c3 * d4 - c4 * d3) + a3 * (c4 * d2 - c2 * d4) + a4 * (c2 * d3 - c3 * d2));
    res.a3 = invdet  * (a2 * (b3 * d4 - b4 * d3) + a3 * (b4 * d2 - b2 * d4) + a4 * (b2 * d3 - b3 * d2));
    res.a4 = -invdet * (a2 * (b3 * c4 - b4 * c3) + a3 * (b4 * c2 - b2 * c4) + a4 * (b2 * c3 - b3 * c2));
    res.b1 = -invdet * (b1 * (c3 * d4 - c4 * d3) + b3 * (c4 * d1 - c1 * d4) + b4 * (c1 * d3 - c3 * d1));
    res.b2 = invdet  * (a1 * (c3 * d4 - c4 * d3) + a3 * (c4 * d1 - c1 * d4) + a4 * (c1 * d3 - c3 * d1));
    res.b3 = -invdet * (a1 * (b3 * d4 - b4 * d3) + a3 * (b4 * d1 - b1 * d4) + a4 * (b1 * d3 - b3 * d1));
    res.b4 = invdet  * (a1 * (b3 * c4 - b4 * c3) + a3 * (b4 * c1 - b1 * c4) + a4 * (b1 * c3 - b3 * c1));
    res.c1 = invdet  * (b1 * (c2 * d4 - c4 * d2) + b2 * (c4 * d1 - c1 * d4) + b4 * (c1 * d2 - c2 * d1));
    res.c2 = -invdet * (a1 * (c2 * d4 - c4 * d2) + a2 * (c4 * d1 - c1 * d4) + a4 * (c1 * d2 - c2 * d1));
    res.c3 = invdet  * (a1 * (b2 * d4 - b4 * d2) + a2 * (b4 * d1 - b1 * d4) + a4 * (b1 * d2 - b2 * d1));
    res.c4 = -invdet * (a1 * (b2 * c4 - b4 * c2) + a2 * (b4 * c1 - b1 * c4) + a4 * (b1 * c2 - b2 * c1));
    res.d1 = -invdet * (b1 * (c2 * d3 - c3 * d2) + b2 * (c3 * d1 - c1 * d3) + b3 * (c1 * d2 - c2 * d1));
    res.d2 = invdet  * (a1 * (c2 * d3 - c3 * d2) + a2 * (c3 * d1 - c1 * d3) + a3 * (c1 * d2 - c2 * d1));
    res.d3 = -invdet * (a1 * (b2 * d3 - b3 * d2) + a2 * (b3 * d1 - b1 * d3) + a3 * (b1 * d2 - b2 * d1));
    res.d4 = invdet  * (a1 * (b2 * c3 - b3 * c2) + a2 * (b3 * c1 - b1 * c3) + a3 * (b1 * c2 - b2 * c1));
    *this = res;

    return *this;
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
inline TReal* aiMatrix4x4t<TReal>::operator[](unsigned int p_iIndex) {
    if (p_iIndex > 3) {
        return NULL;
    }
    switch ( p_iIndex ) {
        case 0:
            return &a1;
        case 1:
            return &b1;
        case 2:
            return &c1;
        case 3:
            return &d1;
        default:
            break;
    }
    return &a1;
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
inline const TReal* aiMatrix4x4t<TReal>::operator[](unsigned int p_iIndex) const {
    if (p_iIndex > 3) {
        return NULL;
    }

    switch ( p_iIndex ) {
        case 0:
            return &a1;
        case 1:
            return &b1;
        case 2:
            return &c1;
        case 3:
            return &d1;
        default:
            break;
    }
    return &a1;
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
inline bool aiMatrix4x4t<TReal>::operator== (const aiMatrix4x4t<TReal>& m) const
{
    return (a1 == m.a1 && a2 == m.a2 && a3 == m.a3 && a4 == m.a4 &&
            b1 == m.b1 && b2 == m.b2 && b3 == m.b3 && b4 == m.b4 &&
            c1 == m.c1 && c2 == m.c2 && c3 == m.c3 && c4 == m.c4 &&
            d1 == m.d1 && d2 == m.d2 && d3 == m.d3 && d4 == m.d4);
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
inline bool aiMatrix4x4t<TReal>::operator!= (const aiMatrix4x4t<TReal>& m) const
{
    return !(*this == m);
}

// ---------------------------------------------------------------------------
template<typename TReal>
inline bool aiMatrix4x4t<TReal>::Equal(const aiMatrix4x4t<TReal>& m, TReal epsilon) const {
    return
        std::abs(a1 - m.a1) <= epsilon &&
        std::abs(a2 - m.a2) <= epsilon &&
        std::abs(a3 - m.a3) <= epsilon &&
        std::abs(a4 - m.a4) <= epsilon &&
        std::abs(b1 - m.b1) <= epsilon &&
        std::abs(b2 - m.b2) <= epsilon &&
        std::abs(b3 - m.b3) <= epsilon &&
        std::abs(b4 - m.b4) <= epsilon &&
        std::abs(c1 - m.c1) <= epsilon &&
        std::abs(c2 - m.c2) <= epsilon &&
        std::abs(c3 - m.c3) <= epsilon &&
        std::abs(c4 - m.c4) <= epsilon &&
        std::abs(d1 - m.d1) <= epsilon &&
        std::abs(d2 - m.d2) <= epsilon &&
        std::abs(d3 - m.d3) <= epsilon &&
        std::abs(d4 - m.d4) <= epsilon;
}

// ----------------------------------------------------------------------------------------

#define ASSIMP_MATRIX4_4_DECOMPOSE_PART		\
	const aiMatrix4x4t<TReal>& _this = *this;/* Create alias for conveniance. */ \
	\
	/* extract translation */ \
	pPosition.x = _this[0][3]; \
	pPosition.y = _this[1][3]; \
	pPosition.z = _this[2][3]; \
	\
	/* extract the columns of the matrix. */ \
	aiVector3t<TReal> vCols[3] = { \
		aiVector3t<TReal>(_this[0][0],_this[1][0],_this[2][0]), \
		aiVector3t<TReal>(_this[0][1],_this[1][1],_this[2][1]), \
		aiVector3t<TReal>(_this[0][2],_this[1][2],_this[2][2]) \
	}; \
	\
	/* extract the scaling factors */ \
	pScaling.x = vCols[0].Length(); \
	pScaling.y = vCols[1].Length(); \
	pScaling.z = vCols[2].Length(); \
	\
	/* and the sign of the scaling */ \
	if (Determinant() < 0) pScaling = -pScaling; \
	\
	/* and remove all scaling from the matrix */ \
	if(pScaling.x) vCols[0] /= pScaling.x; \
	if(pScaling.y) vCols[1] /= pScaling.y; \
	if(pScaling.z) vCols[2] /= pScaling.z; \
	\
	do {} while(false)




template <typename TReal>
inline void aiMatrix4x4t<TReal>::Decompose (aiVector3t<TReal>& pScaling, aiQuaterniont<TReal>& pRotation,
    aiVector3t<TReal>& pPosition) const
{
	ASSIMP_MATRIX4_4_DECOMPOSE_PART;

    // build a 3x3 rotation matrix
    aiMatrix3x3t<TReal> m(vCols[0].x,vCols[1].x,vCols[2].x,
        vCols[0].y,vCols[1].y,vCols[2].y,
        vCols[0].z,vCols[1].z,vCols[2].z);

    // and generate the rotation quaternion from it
    pRotation = aiQuaterniont<TReal>(m);
}

template <typename TReal>
inline void aiMatrix4x4t<TReal>::Decompose(aiVector3t<TReal>& pScaling, aiVector3t<TReal>& pRotation, aiVector3t<TReal>& pPosition) const
{
	ASSIMP_MATRIX4_4_DECOMPOSE_PART;

    /*
    assuming a right-handed coordinate system
    and post-multiplication of column vectors,
    the rotation matrix for an euler XYZ rotation is M = Rz * Ry * Rx.
    combining gives:
    
        |  CE  BDE-AF  ADE+BF  0  |
    M = |  CF  BDF+AE  ADF-BE  0  |
        |  -D    CB      AC    0  |
        |   0     0       0    1  |

    where
	A = cos(angle_x), B = sin(angle_x);
	C = cos(angle_y), D = sin(angle_y);
	E = cos(angle_z), F = sin(angle_z);
	*/

	// Use a small epsilon to solve floating-point inaccuracies
    const TReal epsilon = 10e-3f;

	pRotation.y  = std::asin(-vCols[0].z);// D. Angle around oY.

	TReal C = std::cos(pRotation.y);

	if(std::fabs(C) > epsilon)
	{
		// Finding angle around oX.
		TReal tan_x = vCols[2].z / C;// A
		TReal tan_y = vCols[1].z / C;// B

		pRotation.x = std::atan2(tan_y, tan_x);
		// Finding angle around oZ.
		tan_x = vCols[0].x / C;// E
		tan_y = vCols[0].y / C;// F
		pRotation.z = std::atan2(tan_y, tan_x);
	}
	else
	{// oY is fixed.
		pRotation.x = 0;// Set angle around oX to 0. => A == 1, B == 0, C == 0, D == 1.

		// And finding angle around oZ.
		TReal tan_x =  vCols[1].y;// BDF+AE => E
		TReal tan_y = -vCols[1].x;// BDE-AF => F

		pRotation.z = std::atan2(tan_y, tan_x);
	}
}

#undef ASSIMP_MATRIX4_4_DECOMPOSE_PART

template <typename TReal>
inline void aiMatrix4x4t<TReal>::Decompose(aiVector3t<TReal>& pScaling, aiVector3t<TReal>& pRotationAxis, TReal& pRotationAngle,
											aiVector3t<TReal>& pPosition) const
{
aiQuaterniont<TReal> pRotation;

	Decompose(pScaling, pRotation, pPosition);
	pRotation.Normalize();

	TReal angle_cos = pRotation.w;
	TReal angle_sin = std::sqrt(1.0f - angle_cos * angle_cos);

	pRotationAngle = std::acos(angle_cos) * 2;

	// Use a small epsilon to solve floating-point inaccuracies
    const TReal epsilon = 10e-3f;

	if(std::fabs(angle_sin) < epsilon) angle_sin = 1;

	pRotationAxis.x = pRotation.x / angle_sin;
	pRotationAxis.y = pRotation.y / angle_sin;
	pRotationAxis.z = pRotation.z / angle_sin;
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
inline void aiMatrix4x4t<TReal>::DecomposeNoScaling (aiQuaterniont<TReal>& rotation,
    aiVector3t<TReal>& position) const
{
    const aiMatrix4x4t<TReal>& _this = *this;

    // extract translation
    position.x = _this[0][3];
    position.y = _this[1][3];
    position.z = _this[2][3];

    // extract rotation
    rotation = aiQuaterniont<TReal>((aiMatrix3x3t<TReal>)_this);
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
inline aiMatrix4x4t<TReal>& aiMatrix4x4t<TReal>::FromEulerAnglesXYZ(const aiVector3t<TReal>& blubb)
{
    return FromEulerAnglesXYZ(blubb.x,blubb.y,blubb.z);
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
inline aiMatrix4x4t<TReal>& aiMatrix4x4t<TReal>::FromEulerAnglesXYZ(TReal x, TReal y, TReal z)
{
    aiMatrix4x4t<TReal>& _this = *this;

    TReal cx = std::cos(x);
    TReal sx = std::sin(x);
    TReal cy = std::cos(y);
    TReal sy = std::sin(y);
    TReal cz = std::cos(z);
    TReal sz = std::sin(z);

    // mz*my*mx
    _this.a1 = cz * cy;
    _this.a2 = cz * sy * sx - sz * cx;
    _this.a3 = sz * sx + cz * sy * cx;

    _this.b1 = sz * cy;
    _this.b2 = cz * cx + sz * sy * sx;
    _this.b3 = sz * sy * cx - cz * sx;

    _this.c1 = -sy;
    _this.c2 = cy * sx;
    _this.c3 = cy * cx;

    return *this;
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
inline bool aiMatrix4x4t<TReal>::IsIdentity() const
{
    // Use a small epsilon to solve floating-point inaccuracies
    const static TReal epsilon = 10e-3f;

    return (a2 <= epsilon && a2 >= -epsilon &&
            a3 <= epsilon && a3 >= -epsilon &&
            a4 <= epsilon && a4 >= -epsilon &&
            b1 <= epsilon && b1 >= -epsilon &&
            b3 <= epsilon && b3 >= -epsilon &&
            b4 <= epsilon && b4 >= -epsilon &&
            c1 <= epsilon && c1 >= -epsilon &&
            c2 <= epsilon && c2 >= -epsilon &&
            c4 <= epsilon && c4 >= -epsilon &&
            d1 <= epsilon && d1 >= -epsilon &&
            d2 <= epsilon && d2 >= -epsilon &&
            d3 <= epsilon && d3 >= -epsilon &&
            a1 <= 1.f+epsilon && a1 >= 1.f-epsilon &&
            b2 <= 1.f+epsilon && b2 >= 1.f-epsilon &&
            c3 <= 1.f+epsilon && c3 >= 1.f-epsilon &&
            d4 <= 1.f+epsilon && d4 >= 1.f-epsilon);
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
inline aiMatrix4x4t<TReal>& aiMatrix4x4t<TReal>::RotationX(TReal a, aiMatrix4x4t<TReal>& out)
{
    /*
         |  1  0       0       0 |
     M = |  0  cos(A) -sin(A)  0 |
         |  0  sin(A)  cos(A)  0 |
         |  0  0       0       1 |  */
    out = aiMatrix4x4t<TReal>();
    out.b2 = out.c3 = std::cos(a);
    out.b3 = -(out.c2 = std::sin(a));
    return out;
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
inline aiMatrix4x4t<TReal>& aiMatrix4x4t<TReal>::RotationY(TReal a, aiMatrix4x4t<TReal>& out)
{
    /*
         |  cos(A)  0   sin(A)  0 |
     M = |  0       1   0       0 |
         | -sin(A)  0   cos(A)  0 |
         |  0       0   0       1 |
        */
    out = aiMatrix4x4t<TReal>();
    out.a1 = out.c3 = std::cos(a);
    out.c1 = -(out.a3 = std::sin(a));
    return out;
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
inline aiMatrix4x4t<TReal>& aiMatrix4x4t<TReal>::RotationZ(TReal a, aiMatrix4x4t<TReal>& out)
{
    /*
         |  cos(A)  -sin(A)   0   0 |
     M = |  sin(A)   cos(A)   0   0 |
         |  0        0        1   0 |
         |  0        0        0   1 |   */
    out = aiMatrix4x4t<TReal>();
    out.a1 = out.b2 = std::cos(a);
    out.a2 = -(out.b1 = std::sin(a));
    return out;
}

// ----------------------------------------------------------------------------------------
// Returns a rotation matrix for a rotation around an arbitrary axis.
template <typename TReal>
inline aiMatrix4x4t<TReal>& aiMatrix4x4t<TReal>::Rotation( TReal a, const aiVector3t<TReal>& axis, aiMatrix4x4t<TReal>& out)
{
  TReal c = std::cos( a), s = std::sin( a), t = 1 - c;
  TReal x = axis.x, y = axis.y, z = axis.z;

  // Many thanks to MathWorld and Wikipedia
  out.a1 = t*x*x + c;   out.a2 = t*x*y - s*z; out.a3 = t*x*z + s*y;
  out.b1 = t*x*y + s*z; out.b2 = t*y*y + c;   out.b3 = t*y*z - s*x;
  out.c1 = t*x*z - s*y; out.c2 = t*y*z + s*x; out.c3 = t*z*z + c;
  out.a4 = out.b4 = out.c4 = static_cast<TReal>(0.0);
  out.d1 = out.d2 = out.d3 = static_cast<TReal>(0.0);
  out.d4 = static_cast<TReal>(1.0);

  return out;
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
inline aiMatrix4x4t<TReal>& aiMatrix4x4t<TReal>::Translation( const aiVector3t<TReal>& v, aiMatrix4x4t<TReal>& out)
{
    out = aiMatrix4x4t<TReal>();
    out.a4 = v.x;
    out.b4 = v.y;
    out.c4 = v.z;
    return out;
}

// ----------------------------------------------------------------------------------------
template <typename TReal>
inline aiMatrix4x4t<TReal>& aiMatrix4x4t<TReal>::Scaling( const aiVector3t<TReal>& v, aiMatrix4x4t<TReal>& out)
{
    out = aiMatrix4x4t<TReal>();
    out.a1 = v.x;
    out.b2 = v.y;
    out.c3 = v.z;
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
inline aiMatrix4x4t<TReal>& aiMatrix4x4t<TReal>::FromToMatrix(const aiVector3t<TReal>& from,
    const aiVector3t<TReal>& to, aiMatrix4x4t<TReal>& mtx)
{
    aiMatrix3x3t<TReal> m3;
    aiMatrix3x3t<TReal>::FromToMatrix(from,to,m3);
    mtx = aiMatrix4x4t<TReal>(m3);
    return mtx;
}

#endif // __cplusplus
#endif // AI_MATRIX4X4_INL_INC
