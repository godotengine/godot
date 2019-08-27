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

/** @file  color4.inl
 *  @brief Inline implementation of aiColor4t<TReal> operators
 */
#pragma once
#ifndef AI_COLOR4D_INL_INC
#define AI_COLOR4D_INL_INC

#ifdef __cplusplus
#include "color4.h"

// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE const aiColor4t<TReal>& aiColor4t<TReal>::operator += (const aiColor4t<TReal>& o) {
    r += o.r; g += o.g; b += o.b; a += o.a;
    return *this;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE const aiColor4t<TReal>& aiColor4t<TReal>::operator -= (const aiColor4t<TReal>& o) {
    r -= o.r; g -= o.g; b -= o.b; a -= o.a;
    return *this;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE const aiColor4t<TReal>& aiColor4t<TReal>::operator *= (TReal f) {
    r *= f; g *= f; b *= f; a *= f;
    return *this;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE const aiColor4t<TReal>& aiColor4t<TReal>::operator /= (TReal f) {
    r /= f; g /= f; b /= f; a /= f;
    return *this;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE TReal aiColor4t<TReal>::operator[](unsigned int i) const {
    switch ( i ) {
        case 0:
            return r;
        case 1:
            return g;
        case 2:
            return b;
        case 3:
            return a;
        default:
            break;
    }
    return r;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE TReal& aiColor4t<TReal>::operator[](unsigned int i) {
    switch ( i ) {
        case 0:
            return r;
        case 1:
            return g;
        case 2:
            return b;
        case 3:
            return a;
        default:
            break;
    }
    return r;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE bool aiColor4t<TReal>::operator== (const aiColor4t<TReal>& other) const {
    return r == other.r && g == other.g && b == other.b && a == other.a;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE bool aiColor4t<TReal>::operator!= (const aiColor4t<TReal>& other) const {
    return r != other.r || g != other.g || b != other.b || a != other.a;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE bool aiColor4t<TReal>::operator< (const aiColor4t<TReal>& other) const {
    return r < other.r || (
        r == other.r && (
            g < other.g || (
                g == other.g && (
                    b < other.b || (
                        b == other.b && (
                            a < other.a
                        )
                    )
                )
            )
        )
    );
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE aiColor4t<TReal> operator + (const aiColor4t<TReal>& v1, const aiColor4t<TReal>& v2)    {
    return aiColor4t<TReal>( v1.r + v2.r, v1.g + v2.g, v1.b + v2.b, v1.a + v2.a);
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE aiColor4t<TReal> operator - (const aiColor4t<TReal>& v1, const aiColor4t<TReal>& v2)    {
    return aiColor4t<TReal>( v1.r - v2.r, v1.g - v2.g, v1.b - v2.b, v1.a - v2.a);
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE aiColor4t<TReal> operator * (const aiColor4t<TReal>& v1, const aiColor4t<TReal>& v2)    {
    return aiColor4t<TReal>( v1.r * v2.r, v1.g * v2.g, v1.b * v2.b, v1.a * v2.a);
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE aiColor4t<TReal> operator / (const aiColor4t<TReal>& v1, const aiColor4t<TReal>& v2)    {
    return aiColor4t<TReal>( v1.r / v2.r, v1.g / v2.g, v1.b / v2.b, v1.a / v2.a);
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE aiColor4t<TReal> operator * ( TReal f, const aiColor4t<TReal>& v)   {
    return aiColor4t<TReal>( f*v.r, f*v.g, f*v.b, f*v.a);
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE  aiColor4t<TReal> operator * ( const aiColor4t<TReal>& v, TReal f)  {
    return aiColor4t<TReal>( f*v.r, f*v.g, f*v.b, f*v.a);
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE  aiColor4t<TReal> operator / ( const aiColor4t<TReal>& v, TReal f)  {
    return v * (1/f);
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE  aiColor4t<TReal> operator / ( TReal f,const aiColor4t<TReal>& v)   {
    return aiColor4t<TReal>(f,f,f,f)/v;
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE  aiColor4t<TReal> operator + ( const aiColor4t<TReal>& v, TReal f)  {
    return aiColor4t<TReal>( f+v.r, f+v.g, f+v.b, f+v.a);
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE  aiColor4t<TReal> operator - ( const aiColor4t<TReal>& v, TReal f)  {
    return aiColor4t<TReal>( v.r-f, v.g-f, v.b-f, v.a-f);
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE  aiColor4t<TReal> operator + ( TReal f, const aiColor4t<TReal>& v)  {
    return aiColor4t<TReal>( f+v.r, f+v.g, f+v.b, f+v.a);
}
// ------------------------------------------------------------------------------------------------
template <typename TReal>
AI_FORCE_INLINE  aiColor4t<TReal> operator - ( TReal f, const aiColor4t<TReal>& v)  {
    return aiColor4t<TReal>( f-v.r, f-v.g, f-v.b, f-v.a);
}

// ------------------------------------------------------------------------------------------------
template <typename TReal>
inline bool aiColor4t<TReal> :: IsBlack() const {
    // The alpha component doesn't care here. black is black.
    static const TReal epsilon = 10e-3f;
    return std::fabs( r ) < epsilon && std::fabs( g ) < epsilon && std::fabs( b ) < epsilon;
}

#endif // __cplusplus
#endif // AI_VECTOR3D_INL_INC
