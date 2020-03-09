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
/** @file vector2.h
 *  @brief 2D vector structure, including operators when compiling in C++
 */
#pragma once
#ifndef AI_VECTOR2D_H_INC
#define AI_VECTOR2D_H_INC

#ifdef __GNUC__
#   pragma GCC system_header
#endif

#ifdef __cplusplus
#   include <cmath>
#else
#   include <math.h>
#endif

#include "defs.h"

// ----------------------------------------------------------------------------------
/** Represents a two-dimensional vector.
 */

#ifdef __cplusplus
template <typename TReal>
class aiVector2t {
public:
    aiVector2t () : x(), y() {}
    aiVector2t (TReal _x, TReal _y) : x(_x), y(_y) {}
    explicit aiVector2t (TReal _xyz) : x(_xyz), y(_xyz) {}
    aiVector2t (const aiVector2t& o) = default;

    void Set( TReal pX, TReal pY);
    TReal SquareLength() const ;
    TReal Length() const ;
    aiVector2t& Normalize();

    const aiVector2t& operator += (const aiVector2t& o);
    const aiVector2t& operator -= (const aiVector2t& o);
    const aiVector2t& operator *= (TReal f);
    const aiVector2t& operator /= (TReal f);

    TReal operator[](unsigned int i) const;

    bool operator== (const aiVector2t& other) const;
    bool operator!= (const aiVector2t& other) const;

    bool Equal(const aiVector2t& other, TReal epsilon = 1e-6) const;

    aiVector2t& operator= (TReal f);
    const aiVector2t SymMul(const aiVector2t& o);

    template <typename TOther>
    operator aiVector2t<TOther> () const;

    TReal x, y;
};

typedef aiVector2t<ai_real> aiVector2D;

#else

struct aiVector2D {
    ai_real x, y;
};

#endif // __cplusplus

#endif // AI_VECTOR2D_H_INC
