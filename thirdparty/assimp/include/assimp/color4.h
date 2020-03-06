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
/** @file color4.h
 *  @brief RGBA color structure, including operators when compiling in C++
 */
#pragma once
#ifndef AI_COLOR4D_H_INC
#define AI_COLOR4D_H_INC

#ifdef __GNUC__
#   pragma GCC system_header
#endif

#include <assimp/defs.h>

#ifdef __cplusplus

// ----------------------------------------------------------------------------------
/** Represents a color in Red-Green-Blue space including an
*   alpha component. Color values range from 0 to 1. */
// ----------------------------------------------------------------------------------
template <typename TReal>
class aiColor4t {
public:
    aiColor4t() AI_NO_EXCEPT : r(), g(), b(), a() {}
    aiColor4t (TReal _r, TReal _g, TReal _b, TReal _a)
        : r(_r), g(_g), b(_b), a(_a) {}
    explicit aiColor4t (TReal _r) : r(_r), g(_r), b(_r), a(_r) {}
    aiColor4t (const aiColor4t& o) = default;

    // combined operators
    const aiColor4t& operator += (const aiColor4t& o);
    const aiColor4t& operator -= (const aiColor4t& o);
    const aiColor4t& operator *= (TReal f);
    const aiColor4t& operator /= (TReal f);

    // comparison
    bool operator == (const aiColor4t& other) const;
    bool operator != (const aiColor4t& other) const;
    bool operator <  (const aiColor4t& other) const;

    // color tuple access, rgba order
    inline TReal operator[](unsigned int i) const;
    inline TReal& operator[](unsigned int i);

    /** check whether a color is (close to) black */
    inline bool IsBlack() const;

    // Red, green, blue and alpha color values
    TReal r, g, b, a;
};  // !struct aiColor4D

typedef aiColor4t<ai_real> aiColor4D;

#else

struct aiColor4D {
    ai_real r, g, b, a;
};

#endif // __cplusplus

#endif // AI_COLOR4D_H_INC
