/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2020, assimp team


All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the
following conditions are met:

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

----------------------------------------------------------------------
*/
/** @file Defines a helper class to represent an interleaved vertex
  along with arithmetic operations to support vertex operations
  such as subdivision, smoothing etc.

  While the code is kept as general as possible, arithmetic operations
  that are not currently well-defined (and would cause compile errors
  due to missing operators in the math library), are commented.
  */
#pragma once
#ifndef AI_VERTEX_H_INC
#define AI_VERTEX_H_INC

#ifdef __GNUC__
#   pragma GCC system_header
#endif

#include <assimp/vector3.h>
#include <assimp/mesh.h>
#include <assimp/ai_assert.h>

#include <functional>

namespace Assimp    {

    ///////////////////////////////////////////////////////////////////////////
    // std::plus-family operates on operands with identical types - we need to
    // support all the (vectype op float) combinations in vector maths.
    // Providing T(float) would open the way to endless implicit conversions.
    ///////////////////////////////////////////////////////////////////////////
    namespace Intern {
        template <typename T0, typename T1, typename TRES = T0> struct plus {
            TRES operator() (const T0& t0, const T1& t1) const {
                return t0+t1;
            }
        };
        template <typename T0, typename T1, typename TRES = T0> struct minus {
            TRES operator() (const T0& t0, const T1& t1) const {
                return t0-t1;
            }
        };
        template <typename T0, typename T1, typename TRES = T0> struct multiplies {
            TRES operator() (const T0& t0, const T1& t1) const {
                return t0*t1;
            }
        };
        template <typename T0, typename T1, typename TRES = T0> struct divides {
            TRES operator() (const T0& t0, const T1& t1) const {
                return t0/t1;
            }
        };
    }

// ------------------------------------------------------------------------------------------------
/** Intermediate description a vertex with all possible components. Defines a full set of
 *  operators, so you may use such a 'Vertex' in basic arithmetics. All operators are applied
 *  to *all* vertex components equally. This is useful for stuff like interpolation
 *  or subdivision, but won't work if special handling is required for some vertex components. */
// ------------------------------------------------------------------------------------------------
class Vertex {
    friend Vertex operator + (const Vertex&,const Vertex&);
    friend Vertex operator - (const Vertex&,const Vertex&);
    friend Vertex operator * (const Vertex&,ai_real);
    friend Vertex operator / (const Vertex&,ai_real);
    friend Vertex operator * (ai_real, const Vertex&);

public:
    Vertex() {}

    // ----------------------------------------------------------------------------
    /** Extract a particular vertex from a mesh and interleave all components */
    explicit Vertex(const aiMesh* msh, unsigned int idx) {
        ai_assert(idx < msh->mNumVertices);
        position = msh->mVertices[idx];

        if (msh->HasNormals()) {
            normal = msh->mNormals[idx];
        }

        if (msh->HasTangentsAndBitangents()) {
            tangent = msh->mTangents[idx];
            bitangent = msh->mBitangents[idx];
        }

        for (unsigned int i = 0; msh->HasTextureCoords(i); ++i) {
            texcoords[i] = msh->mTextureCoords[i][idx];
        }

        for (unsigned int i = 0; msh->HasVertexColors(i); ++i) {
            colors[i] = msh->mColors[i][idx];
        }
    }

    // ----------------------------------------------------------------------------
    /** Extract a particular vertex from a anim mesh and interleave all components */
    explicit Vertex(const aiAnimMesh* msh, unsigned int idx) {
        ai_assert(idx < msh->mNumVertices);
        position = msh->mVertices[idx];

        if (msh->HasNormals()) {
            normal = msh->mNormals[idx];
        }

        if (msh->HasTangentsAndBitangents()) {
            tangent = msh->mTangents[idx];
            bitangent = msh->mBitangents[idx];
        }

        for (unsigned int i = 0; msh->HasTextureCoords(i); ++i) {
            texcoords[i] = msh->mTextureCoords[i][idx];
        }

        for (unsigned int i = 0; msh->HasVertexColors(i); ++i) {
           colors[i] = msh->mColors[i][idx];
        }
    }

    Vertex& operator += (const Vertex& v) {
        *this = *this+v;
        return *this;
    }

    Vertex& operator -= (const Vertex& v) {
        *this = *this-v;
        return *this;
    }

    Vertex& operator *= (ai_real v) {
        *this = *this*v;
        return *this;
    }

    Vertex& operator /= (ai_real v) {
        *this = *this/v;
        return *this;
    }

    // ----------------------------------------------------------------------------
    /** Convert back to non-interleaved storage */
    void SortBack(aiMesh* out, unsigned int idx) const {
        ai_assert(idx<out->mNumVertices);
        out->mVertices[idx] = position;

        if (out->HasNormals()) {
            out->mNormals[idx] = normal;
        }

        if (out->HasTangentsAndBitangents()) {
            out->mTangents[idx] = tangent;
            out->mBitangents[idx] = bitangent;
        }

        for(unsigned int i = 0; out->HasTextureCoords(i); ++i) {
            out->mTextureCoords[i][idx] = texcoords[i];
        }

        for(unsigned int i = 0; out->HasVertexColors(i); ++i) {
            out->mColors[i][idx] = colors[i];
        }
    }

private:

    // ----------------------------------------------------------------------------
    /** Construct from two operands and a binary operation to combine them */
    template <template <typename t> class op> static Vertex BinaryOp(const Vertex& v0, const Vertex& v1) {
        // this is a heavy task for the compiler to optimize ... *pray*

        Vertex res;
        res.position  = op<aiVector3D>()(v0.position,v1.position);
        res.normal    = op<aiVector3D>()(v0.normal,v1.normal);
        res.tangent   = op<aiVector3D>()(v0.tangent,v1.tangent);
        res.bitangent = op<aiVector3D>()(v0.bitangent,v1.bitangent);

        for (unsigned int i = 0; i < AI_MAX_NUMBER_OF_TEXTURECOORDS; ++i) {
            res.texcoords[i] = op<aiVector3D>()(v0.texcoords[i],v1.texcoords[i]);
        }
        for (unsigned int i = 0; i < AI_MAX_NUMBER_OF_COLOR_SETS; ++i) {
            res.colors[i] = op<aiColor4D>()(v0.colors[i],v1.colors[i]);
        }
        return res;
    }

    // ----------------------------------------------------------------------------
    /** This time binary arithmetics of v0 with a floating-point number */
    template <template <typename, typename, typename> class op> static Vertex BinaryOp(const Vertex& v0, ai_real f) {
        // this is a heavy task for the compiler to optimize ... *pray*

        Vertex res;
        res.position  = op<aiVector3D,ai_real,aiVector3D>()(v0.position,f);
        res.normal    = op<aiVector3D,ai_real,aiVector3D>()(v0.normal,f);
        res.tangent   = op<aiVector3D,ai_real,aiVector3D>()(v0.tangent,f);
        res.bitangent = op<aiVector3D,ai_real,aiVector3D>()(v0.bitangent,f);

        for (unsigned int i = 0; i < AI_MAX_NUMBER_OF_TEXTURECOORDS; ++i) {
            res.texcoords[i] = op<aiVector3D,ai_real,aiVector3D>()(v0.texcoords[i],f);
        }
        for (unsigned int i = 0; i < AI_MAX_NUMBER_OF_COLOR_SETS; ++i) {
            res.colors[i] = op<aiColor4D,ai_real,aiColor4D>()(v0.colors[i],f);
        }
        return res;
    }

    // ----------------------------------------------------------------------------
    /** This time binary arithmetics of v0 with a floating-point number */
    template <template <typename, typename, typename> class op> static Vertex BinaryOp(ai_real f, const Vertex& v0) {
        // this is a heavy task for the compiler to optimize ... *pray*

        Vertex res;
        res.position  = op<ai_real,aiVector3D,aiVector3D>()(f,v0.position);
        res.normal    = op<ai_real,aiVector3D,aiVector3D>()(f,v0.normal);
        res.tangent   = op<ai_real,aiVector3D,aiVector3D>()(f,v0.tangent);
        res.bitangent = op<ai_real,aiVector3D,aiVector3D>()(f,v0.bitangent);

        for (unsigned int i = 0; i < AI_MAX_NUMBER_OF_TEXTURECOORDS; ++i) {
            res.texcoords[i] = op<ai_real,aiVector3D,aiVector3D>()(f,v0.texcoords[i]);
        }
        for (unsigned int i = 0; i < AI_MAX_NUMBER_OF_COLOR_SETS; ++i) {
            res.colors[i] = op<ai_real,aiColor4D,aiColor4D>()(f,v0.colors[i]);
        }
        return res;
    }

public:

    aiVector3D position;
    aiVector3D normal;
    aiVector3D tangent, bitangent;

    aiVector3D texcoords[AI_MAX_NUMBER_OF_TEXTURECOORDS];
    aiColor4D colors[AI_MAX_NUMBER_OF_COLOR_SETS];
};

// ------------------------------------------------------------------------------------------------
AI_FORCE_INLINE Vertex operator + (const Vertex& v0,const Vertex& v1) {
    return Vertex::BinaryOp<std::plus>(v0,v1);
}

AI_FORCE_INLINE Vertex operator - (const Vertex& v0,const Vertex& v1) {
    return Vertex::BinaryOp<std::minus>(v0,v1);
}

AI_FORCE_INLINE Vertex operator * (const Vertex& v0,ai_real f) {
    return Vertex::BinaryOp<Intern::multiplies>(v0,f);
}

AI_FORCE_INLINE Vertex operator / (const Vertex& v0,ai_real f) {
    return Vertex::BinaryOp<Intern::multiplies>(v0,1.f/f);
}

AI_FORCE_INLINE Vertex operator * (ai_real f,const Vertex& v0) {
    return Vertex::BinaryOp<Intern::multiplies>(f,v0);
}

}

#endif // AI_VERTEX_H_INC
