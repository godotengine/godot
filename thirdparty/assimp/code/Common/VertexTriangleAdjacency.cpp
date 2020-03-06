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

/** @file Implementation of the VertexTriangleAdjacency helper class
 */

// internal headers
#include "VertexTriangleAdjacency.h"
#include <assimp/mesh.h>

using namespace Assimp;

// ------------------------------------------------------------------------------------------------
VertexTriangleAdjacency::VertexTriangleAdjacency(aiFace *pcFaces,
    unsigned int iNumFaces,
    unsigned int iNumVertices /*= 0*/,
    bool bComputeNumTriangles /*= false*/)
{
    // compute the number of referenced vertices if it wasn't specified by the caller
    const aiFace* const pcFaceEnd = pcFaces + iNumFaces;
    if (0 == iNumVertices)  {
        for (aiFace* pcFace = pcFaces; pcFace != pcFaceEnd; ++pcFace)   {
            ai_assert( nullptr != pcFace );
            ai_assert(3 == pcFace->mNumIndices);
            iNumVertices = std::max(iNumVertices,pcFace->mIndices[0]);
            iNumVertices = std::max(iNumVertices,pcFace->mIndices[1]);
            iNumVertices = std::max(iNumVertices,pcFace->mIndices[2]);
        }
    }

    mNumVertices = iNumVertices + 1;

    unsigned int* pi;

    // allocate storage
    if (bComputeNumTriangles)   {
        pi = mLiveTriangles = new unsigned int[iNumVertices+1];
        ::memset(mLiveTriangles,0,sizeof(unsigned int)*(iNumVertices+1));
        mOffsetTable = new unsigned int[iNumVertices+2]+1;
    } else {
        pi = mOffsetTable = new unsigned int[iNumVertices+2]+1;
        ::memset(mOffsetTable,0,sizeof(unsigned int)*(iNumVertices+1));
        mLiveTriangles = NULL; // important, otherwise the d'tor would crash
    }

    // get a pointer to the end of the buffer
    unsigned int* piEnd = pi+iNumVertices;
    *piEnd++ = 0u;

    // first pass: compute the number of faces referencing each vertex
    for (aiFace* pcFace = pcFaces; pcFace != pcFaceEnd; ++pcFace)
    {
        unsigned nind = pcFace->mNumIndices;
        unsigned * ind = pcFace->mIndices;
        if (nind > 0) pi[ind[0]]++;
        if (nind > 1) pi[ind[1]]++;
        if (nind > 2) pi[ind[2]]++;
    }

    // second pass: compute the final offset table
    unsigned int iSum = 0;
    unsigned int* piCurOut = this->mOffsetTable;
    for (unsigned int* piCur = pi; piCur != piEnd;++piCur,++piCurOut)   {

        unsigned int iLastSum = iSum;
        iSum += *piCur;
        *piCurOut = iLastSum;
    }
    pi = this->mOffsetTable;

    // third pass: compute the final table
    this->mAdjacencyTable = new unsigned int[iSum];
    iSum = 0;
    for (aiFace* pcFace = pcFaces; pcFace != pcFaceEnd; ++pcFace,++iSum)    {
        unsigned nind = pcFace->mNumIndices;
        unsigned * ind = pcFace->mIndices;

        if (nind > 0) mAdjacencyTable[pi[ind[0]]++] = iSum;
        if (nind > 1) mAdjacencyTable[pi[ind[1]]++] = iSum;
        if (nind > 2) mAdjacencyTable[pi[ind[2]]++] = iSum;
    }
    // fourth pass: undo the offset computations made during the third pass
    // We could do this in a separate buffer, but this would be TIMES slower.
    --mOffsetTable;
    *mOffsetTable = 0u;
}
// ------------------------------------------------------------------------------------------------
VertexTriangleAdjacency::~VertexTriangleAdjacency()
{
    // delete allocated storage
    delete[] mOffsetTable;
    delete[] mAdjacencyTable;
    delete[] mLiveTriangles;
}
