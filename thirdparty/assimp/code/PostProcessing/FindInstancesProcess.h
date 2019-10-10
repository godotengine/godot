/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team


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

/** @file  FindInstancesProcess.h
 *  @brief Declares the aiProcess_FindInstances post-process step
 */
#ifndef AI_FINDINSTANCES_H_INC
#define AI_FINDINSTANCES_H_INC

#include "Common/BaseProcess.h"
#include "PostProcessing/ProcessHelper.h"

class FindInstancesProcessTest;
namespace Assimp    {

// -------------------------------------------------------------------------------
/** @brief Get a pseudo(!)-hash representing a mesh.
 *
 *  The hash is built from number of vertices, faces, primitive types,
 *  .... but *not* from the real mesh data. The funcction is not a perfect hash.
 *  @param in Input mesh
 *  @return Hash.
 */
inline
uint64_t GetMeshHash(aiMesh* in) {
    ai_assert(nullptr != in);

    // ... get an unique value representing the vertex format of the mesh
    const unsigned int fhash = GetMeshVFormatUnique(in);

    // and bake it with number of vertices/faces/bones/matidx/ptypes
    return ((uint64_t)fhash << 32u) | ((
        (in->mNumBones << 16u) ^  (in->mNumVertices)       ^
        (in->mNumFaces<<4u)    ^  (in->mMaterialIndex<<15) ^
        (in->mPrimitiveTypes<<28)) & 0xffffffff );
}

// -------------------------------------------------------------------------------
/** @brief Perform a component-wise comparison of two arrays
 *
 *  @param first First array
 *  @param second Second array
 *  @param size Size of both arrays
 *  @param e Epsilon
 *  @return true if the arrays are identical
 */
inline
bool CompareArrays(const aiVector3D* first, const aiVector3D* second,
        unsigned int size, float e) {
    for (const aiVector3D* end = first+size; first != end; ++first,++second) {
        if ( (*first - *second).SquareLength() >= e)
            return false;
    }
    return true;
}

// and the same for colors ...
inline bool CompareArrays(const aiColor4D* first, const aiColor4D* second,
    unsigned int size, float e)
{
    for (const aiColor4D* end = first+size; first != end; ++first,++second) {
        if ( GetColorDifference(*first,*second) >= e)
            return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
/** @brief A post-processing steps to search for instanced meshes
*/
class FindInstancesProcess : public BaseProcess
{
public:

    FindInstancesProcess();
    ~FindInstancesProcess();

public:
    // -------------------------------------------------------------------
    // Check whether step is active in given flags combination
    bool IsActive( unsigned int pFlags) const;

    // -------------------------------------------------------------------
    // Execute step on a given scene
    void Execute( aiScene* pScene);

    // -------------------------------------------------------------------
    // Setup properties prior to executing the process
    void SetupProperties(const Importer* pImp);

private:

    bool configSpeedFlag;

}; // ! end class FindInstancesProcess
}  // ! end namespace Assimp

#endif // !! AI_FINDINSTANCES_H_INC
