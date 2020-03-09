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

/** @file Defines the helper data structures for importing 3DS files.
http://www.jalix.org/ressources/graphics/3DS/_unofficials/3ds-unofficial.txt */

#pragma once
#ifndef AI_SMOOTHINGGROUPS_H_INC
#define AI_SMOOTHINGGROUPS_H_INC

#ifdef __GNUC__
#   pragma GCC system_header
#endif

#include <assimp/vector3.h>

#include <stdint.h>
#include <vector>

// ---------------------------------------------------------------------------
/** Helper structure representing a face with smoothing groups assigned */
struct FaceWithSmoothingGroup {
    FaceWithSmoothingGroup() AI_NO_EXCEPT
    : mIndices()
    , iSmoothGroup(0) {
        // in debug builds set all indices to a common magic value
#ifdef ASSIMP_BUILD_DEBUG
        this->mIndices[0] = 0xffffffff;
        this->mIndices[1] = 0xffffffff;
        this->mIndices[2] = 0xffffffff;
#endif
    }


    //! Indices. .3ds is using uint16. However, after
    //! an unique vertex set has been generated,
    //! individual index values might exceed 2^16
    uint32_t mIndices[3];

    //! specifies to which smoothing group the face belongs to
    uint32_t iSmoothGroup;
};

// ---------------------------------------------------------------------------
/** Helper structure representing a mesh whose faces have smoothing
    groups assigned. This allows us to reuse the code for normal computations
    from smoothings groups for several loaders (3DS, ASE). All of them
    use face structures which inherit from #FaceWithSmoothingGroup,
    but as they add extra members and need to be copied by value we
    need to use a template here.
    */
template <class T>
struct MeshWithSmoothingGroups
{
    //! Vertex positions
    std::vector<aiVector3D> mPositions;

    //! Face lists
    std::vector<T> mFaces;

    //! List of normal vectors
    std::vector<aiVector3D> mNormals;
};

// ---------------------------------------------------------------------------
/** Computes normal vectors for the mesh
 */
template <class T>
void ComputeNormalsWithSmoothingsGroups(MeshWithSmoothingGroups<T>& sMesh);


// include implementations
#include "SmoothingGroups.inl"

#endif // !! AI_SMOOTHINGGROUPS_H_INC
