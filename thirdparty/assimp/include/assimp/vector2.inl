/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2020, assimp team



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

/** @file  vector2.inl
 *  @brief Inline implementation of aiVector2t<TReal> operators
 */
#pragma once
#ifndef AI_VECTOR2D_INL_INC
#define AI_VECTOR2D_INL_INC

#ifdef __GNUC__
#   pragma GCC system_header
#endif

#ifdef __cplusplus
#include <assimp/vector2.h>

#include <cmath>

// ------------------------------------------------------------------------------------------------
template <typename TReal>
template <typename TOther>
aiVector2t<TReal>::operator aiVector2t<TOther> () const {
    return aiVector2t<TOther>(static_cast<TOther>(x),static_cast<TOther>(y));
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
inline
void aiVector2t<TReal>::Set( TReal pX, TReal pY) {
    x = pX; y = pY;
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
inline
TReal aiVector2t<TReal>::SquareLength() const {
    return x*x + y*y;
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
inline
TReal aiVector2t<TReal>::Length() const {
    return std::sqrt( SquareLength());
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
inline
aiVector2t<TReal>& aiVector2t<TReal>::Normalize() {
    *this /= Length();
    return *this;
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
inline
const aiVector2t<TReal>& aiVector2t<TReal>::operator += (const aiVector2t& o) {
    x += o.x; y += o.y;
    return *this;
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
inline
const aiVector2t<TReal>& aiVector2t<TReal>::operator -= (const aiVector2t& o) {
    x -= o.x; y -= o.y;
    return *this;
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
inline
const aiVector2t<TReal>& aiVector2t<TReal>::operator *= (TReal f) {
    x *= f; y *= f;
    return *this;
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
inline
const aiVector2t<TReal>& aiVector2t<TReal>::operator /= (TReal f) {
    x /= f; y /= f;
    return *this;
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
inline
TReal aiVector2t<TReal>::operator[](unsigned int i) const {
	switch (i) {
		case 0:
			return x;
		case 1:
			return y;
		default:
			break;

    }
    return x;
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
inline
bool aiVector2t<TReal>::operator== (const aiVector2t& other) const {
    return x == other.x && y == other.y;
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
inline
bool aiVector2t<TReal>::operator!= (const aiVector2t& other) const {
    return x != other.x || y != other.y;
}

// ---------------------------------------------------------------------------
template<typename TReal>
inline
bool aiVector2t<TReal>::Equal(const aiVector2t& other, TReal epsilon) const {
    return
        std::abs(x - other.x) <= epsilon &&
        std::abs(y - other.y) <= epsilon;
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
inline
aiVector2t<TReal>& aiVector2t<TReal>::operator= (TReal f)   {
    x = y = f;
    return *this;
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
inline
const aiVector2t<TReal> aiVector2t<TReal>::SymMul(const aiVector2t& o) {
    return aiVector2t(x*o.x,y*o.y);
}


// ------------------------------------------------------------------------------------------------
// symmetric addition
template <typename TReal>
inline
aiVector2t<TReal> operator + (const aiVector2t<TReal>& v1, const aiVector2t<TReal>& v2) {
    return aiVector2t<TReal>( v1.x + v2.x, v1.y + v2.y);
}

// ------------------------------------------------------------------------------------------------
// symmetric subtraction
template <typename TReal>
inline 
aiVector2t<TReal> operator - (const aiVector2t<TReal>& v1, const aiVector2t<TReal>& v2) {
    return aiVector2t<TReal>( v1.x - v2.x, v1.y - v2.y);
}

// ------------------------------------------------------------------------------------------------
// scalar product
template <typename TReal>
inline 
TReal operator * (const aiVector2t<TReal>& v1, const aiVector2t<TReal>& v2) {
    return v1.x*v2.x + v1.y*v2.y;
}

// ------------------------------------------------------------------------------------------------
// scalar multiplication
template <typename TReal>
inline 
aiVector2t<TReal> operator * ( TReal f, const aiVector2t<TReal>& v) {
    return aiVector2t<TReal>( f*v.x, f*v.y);
}

// ------------------------------------------------------------------------------------------------
// and the other way around
template <typename TReal>
inline 
aiVector2t<TReal> operator * ( const aiVector2t<TReal>& v, TReal f) {
    return aiVector2t<TReal>( f*v.x, f*v.y);
}

// ------------------------------------------------------------------------------------------------
// scalar division
template <typename TReal>
inline 
aiVector2t<TReal> operator / ( const aiVector2t<TReal>& v, TReal f) {
    return v * (1/f);
}

// ------------------------------------------------------------------------------------------------
// vector division
template <typename TReal>
inline 
aiVector2t<TReal> operator / ( const aiVector2t<TReal>& v, const aiVector2t<TReal>& v2) {
    return aiVector2t<TReal>(v.x / v2.x,v.y / v2.y);
}

// ------------------------------------------------------------------------------------------------
// vector negation
template <typename TReal>
inline 
aiVector2t<TReal> operator - ( const aiVector2t<TReal>& v) {
    return aiVector2t<TReal>( -v.x, -v.y);
}

#endif

#endif // AI_VECTOR2D_INL_INC
