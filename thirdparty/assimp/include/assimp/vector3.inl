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

/** @file  vector3.inl
 *  @brief Inline implementation of aiVector3t<TReal> operators
 */
#pragma once
#ifndef AI_VECTOR3D_INL_INC
#define AI_VECTOR3D_INL_INC

#ifdef __cplusplus
#include <assimp/vector3.h>

#include <cmath>

// ------------------------------------------------------------------------------------------------
/** Transformation of a vector by a 3x3 matrix */
template <typename TReal>
AI_FORCE_INLINE
aiVector3t<TReal> operator * (const aiMatrix3x3t<TReal>& pMatrix, const aiVector3t<TReal>& pVector) {
    aiVector3t<TReal> res;
    res.x = pMatrix.a1 * pVector.x + pMatrix.a2 * pVector.y + pMatrix.a3 * pVector.z;
    res.y = pMatrix.b1 * pVector.x + pMatrix.b2 * pVector.y + pMatrix.b3 * pVector.z;
    res.z = pMatrix.c1 * pVector.x + pMatrix.c2 * pVector.y + pMatrix.c3 * pVector.z;
    return res;
}

// ------------------------------------------------------------------------------------------------
/** Transformation of a vector by a 4x4 matrix */
template <typename TReal>
AI_FORCE_INLINE
aiVector3t<TReal> operator * (const aiMatrix4x4t<TReal>& pMatrix, const aiVector3t<TReal>& pVector) {
    aiVector3t<TReal> res;
    res.x = pMatrix.a1 * pVector.x + pMatrix.a2 * pVector.y + pMatrix.a3 * pVector.z + pMatrix.a4;
    res.y = pMatrix.b1 * pVector.x + pMatrix.b2 * pVector.y + pMatrix.b3 * pVector.z + pMatrix.b4;
    res.z = pMatrix.c1 * pVector.x + pMatrix.c2 * pVector.y + pMatrix.c3 * pVector.z + pMatrix.c4;
    return res;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
template <typename TOther>
aiVector3t<TReal>::operator aiVector3t<TOther> () const {
    return aiVector3t<TOther>(static_cast<TOther>(x),static_cast<TOther>(y),static_cast<TOther>(z));
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
void aiVector3t<TReal>::Set( TReal pX, TReal pY, TReal pZ) {
    x = pX;
    y = pY;
    z = pZ;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
TReal aiVector3t<TReal>::SquareLength() const {
    return x*x + y*y + z*z;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
TReal aiVector3t<TReal>::Length() const {
    return std::sqrt( SquareLength());
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
aiVector3t<TReal>& aiVector3t<TReal>::Normalize() {
    *this /= Length();

    return *this;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
aiVector3t<TReal>& aiVector3t<TReal>::NormalizeSafe() {
    TReal len = Length();
    if ( len > static_cast< TReal >( 0 ) ) {
        *this /= len;
    }
    return *this;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
const aiVector3t<TReal>& aiVector3t<TReal>::operator += (const aiVector3t<TReal>& o) {
    x += o.x;
    y += o.y;
    z += o.z;

    return *this;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
const aiVector3t<TReal>& aiVector3t<TReal>::operator -= (const aiVector3t<TReal>& o) {
    x -= o.x;
    y -= o.y;
    z -= o.z;

    return *this;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
const aiVector3t<TReal>& aiVector3t<TReal>::operator *= (TReal f) {
    x *= f;
    y *= f;
    z *= f;

    return *this;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
const aiVector3t<TReal>& aiVector3t<TReal>::operator /= (TReal f) {
    const TReal invF = (TReal) 1.0 / f;
    x *= invF;
    y *= invF;
    z *= invF;

    return *this;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
aiVector3t<TReal>& aiVector3t<TReal>::operator *= (const aiMatrix3x3t<TReal>& mat){
    return (*this =  mat * (*this));
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
aiVector3t<TReal>& aiVector3t<TReal>::operator *= (const aiMatrix4x4t<TReal>& mat){
    return (*this = mat * (*this));
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
TReal aiVector3t<TReal>::operator[](unsigned int i) const {
    switch (i) {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        default:
            break;
    }
    return x;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
TReal& aiVector3t<TReal>::operator[](unsigned int i) {
//    return *(&x + i);
    switch (i) {
        case 0:
            return x;
        case 1:
            return y;
        case 2:
            return z;
        default:
            break;
    }
    return x;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
bool aiVector3t<TReal>::operator== (const aiVector3t<TReal>& other) const {
    return x == other.x && y == other.y && z == other.z;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
bool aiVector3t<TReal>::operator!= (const aiVector3t<TReal>& other) const {
    return x != other.x || y != other.y || z != other.z;
}
// ---------------------------------------------------------------------------
template<typename TReal>
AI_FORCE_INLINE
bool aiVector3t<TReal>::Equal(const aiVector3t<TReal>& other, TReal epsilon) const {
    return
        std::abs(x - other.x) <= epsilon &&
        std::abs(y - other.y) <= epsilon &&
        std::abs(z - other.z) <= epsilon;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
bool aiVector3t<TReal>::operator < (const aiVector3t<TReal>& other) const {
    return x != other.x ? x < other.x : y != other.y ? y < other.y : z < other.z;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE
const aiVector3t<TReal> aiVector3t<TReal>::SymMul(const aiVector3t<TReal>& o) {
    return aiVector3t<TReal>(x*o.x,y*o.y,z*o.z);
}
// ------------------------------------------------------------------------------------------------
// symmetric addition
template <typename TReal>
AI_FORCE_INLINE
aiVector3t<TReal> operator + (const aiVector3t<TReal>& v1, const aiVector3t<TReal>& v2) {
    return aiVector3t<TReal>( v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}
// ------------------------------------------------------------------------------------------------
// symmetric subtraction
template <typename TReal>
AI_FORCE_INLINE
aiVector3t<TReal> operator - (const aiVector3t<TReal>& v1, const aiVector3t<TReal>& v2) {
    return aiVector3t<TReal>( v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}
// ------------------------------------------------------------------------------------------------
// scalar product
template <typename TReal>
AI_FORCE_INLINE
TReal operator * (const aiVector3t<TReal>& v1, const aiVector3t<TReal>& v2) {
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}
// ------------------------------------------------------------------------------------------------
// scalar multiplication
template <typename TReal>
AI_FORCE_INLINE
aiVector3t<TReal> operator * ( TReal f, const aiVector3t<TReal>& v) {
    return aiVector3t<TReal>( f*v.x, f*v.y, f*v.z);
}
// ------------------------------------------------------------------------------------------------
// and the other way around
template <typename TReal>
AI_FORCE_INLINE
aiVector3t<TReal> operator * ( const aiVector3t<TReal>& v, TReal f) {
    return aiVector3t<TReal>( f*v.x, f*v.y, f*v.z);
}
// ------------------------------------------------------------------------------------------------
// scalar division
template <typename TReal>
AI_FORCE_INLINE
aiVector3t<TReal> operator / ( const aiVector3t<TReal>& v, TReal f) {
    return v * (1/f);
}
// ------------------------------------------------------------------------------------------------
// vector division
template <typename TReal>
AI_FORCE_INLINE
aiVector3t<TReal> operator / ( const aiVector3t<TReal>& v, const aiVector3t<TReal>& v2) {
    return aiVector3t<TReal>(v.x / v2.x,v.y / v2.y,v.z / v2.z);
}
// ------------------------------------------------------------------------------------------------
// cross product
template<typename TReal>
AI_FORCE_INLINE
aiVector3t<TReal> operator ^ ( const aiVector3t<TReal>& v1, const aiVector3t<TReal>& v2) {
    return aiVector3t<TReal>( v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x);
}
// ------------------------------------------------------------------------------------------------
// vector negation
template<typename TReal>
AI_FORCE_INLINE
aiVector3t<TReal> operator - ( const aiVector3t<TReal>& v) {
    return aiVector3t<TReal>( -v.x, -v.y, -v.z);
}

// ------------------------------------------------------------------------------------------------

#endif // __cplusplus
#endif // AI_VECTOR3D_INL_INC
