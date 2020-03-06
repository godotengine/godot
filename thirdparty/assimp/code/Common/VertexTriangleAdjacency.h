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

/** @file Defines a helper class to compute a vertex-triangle adjacency map */
#ifndef AI_VTADJACENCY_H_INC
#define AI_VTADJACENCY_H_INC

#include "BaseProcess.h"
#include <assimp/types.h>
#include <assimp/ai_assert.h>

struct aiMesh;
struct aiFace;

namespace Assimp    {

// --------------------------------------------------------------------------------------------
/** @brief The VertexTriangleAdjacency class computes a vertex-triangle
 *  adjacency map from a given index buffer.
 *
 *  @note Although it is called #VertexTriangleAdjacency, the current version does also
 *    support arbitrary polygons. */
// --------------------------------------------------------------------------------------------
class ASSIMP_API VertexTriangleAdjacency {
public:
    // ----------------------------------------------------------------------------
    /** @brief Construction from an existing index buffer
     *  @param pcFaces Index buffer
     *  @param iNumFaces Number of faces in the buffer
     *  @param iNumVertices Number of referenced vertices. This value
     *    is computed automatically if 0 is specified.
     *  @param bComputeNumTriangles If you want the class to compute
     *    a list containing the number of referenced triangles per vertex
     *    per vertex - pass true.  */
    VertexTriangleAdjacency(aiFace* pcFaces,unsigned int iNumFaces,
        unsigned int iNumVertices = 0,
        bool bComputeNumTriangles = true);

    // ----------------------------------------------------------------------------
    /** @brief Destructor */
    ~VertexTriangleAdjacency();

    // ----------------------------------------------------------------------------
    /** @brief Get all triangles adjacent to a vertex
     *  @param iVertIndex Index of the vertex
     *  @return A pointer to the adjacency list. */
    unsigned int* GetAdjacentTriangles(unsigned int iVertIndex) const {
        ai_assert(iVertIndex < mNumVertices);
        return &mAdjacencyTable[ mOffsetTable[iVertIndex]];
    }

    // ----------------------------------------------------------------------------
    /** @brief Get the number of triangles that are referenced by
     *    a vertex. This function returns a reference that can be modified
     *  @param iVertIndex Index of the vertex
     *  @return Number of referenced triangles */
    unsigned int& GetNumTrianglesPtr(unsigned int iVertIndex) {
        ai_assert( iVertIndex < mNumVertices );
        ai_assert( nullptr != mLiveTriangles );
        return mLiveTriangles[iVertIndex];
    }

    //! Offset table
    unsigned int* mOffsetTable;

    //! Adjacency table
    unsigned int* mAdjacencyTable;

    //! Table containing the number of referenced triangles per vertex
    unsigned int* mLiveTriangles;

    //! Debug: Number of referenced vertices
    unsigned int mNumVertices;
};

} //! ns Assimp

#endif // !! AI_VTADJACENCY_H_INC
