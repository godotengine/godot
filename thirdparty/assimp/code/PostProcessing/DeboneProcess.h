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

/** Defines a post processing step to limit the number of bones affecting a single vertex. */
#ifndef AI_DEBONEPROCESS_H_INC
#define AI_DEBONEPROCESS_H_INC

#include "Common/BaseProcess.h"

#include <assimp/mesh.h>
#include <assimp/scene.h>

#include <vector>
#include <utility>

#// Forward declarations
class DeboneTest;

namespace Assimp {

#if (!defined AI_DEBONE_THRESHOLD)
#   define AI_DEBONE_THRESHOLD  1.0f
#endif // !! AI_DEBONE_THRESHOLD

// ---------------------------------------------------------------------------
/** This post processing step removes bones nearly losslessly or according to
* a configured threshold. In order to remove the bone, the primitives affected by
* the bone are split from the mesh. The split off (new) mesh is boneless. At any
* point in time, bones without affect upon a given mesh are to be removed.
*/
class DeboneProcess : public BaseProcess {
public:
    DeboneProcess();
    ~DeboneProcess();

    // -------------------------------------------------------------------
    /** Returns whether the processing step is present in the given flag.
    * @param pFlags The processing flags the importer was called with.
    *   A bitwise combination of #aiPostProcessSteps.
    * @return true if the process is present in this flag fields,
    *   false if not.
    */
    bool IsActive( unsigned int pFlags) const;

    // -------------------------------------------------------------------
    /** Called prior to ExecuteOnScene().
    * The function is a request to the process to update its configuration
    * basing on the Importer's configuration property list.
    */
    void SetupProperties(const Importer* pImp);

protected:
    // -------------------------------------------------------------------
    /** Executes the post processing step on the given imported data.
    * At the moment a process is not supposed to fail.
    * @param pScene The imported data to work at.
    */
    void Execute( aiScene* pScene);

    // -------------------------------------------------------------------
    /** Counts bones total/removable in a given mesh.
    * @param pMesh The mesh to process.
    */
    bool ConsiderMesh( const aiMesh* pMesh);

    /// Splits the given mesh by bone count.
    /// @param pMesh the Mesh to split. Is not changed at all, but might be superfluous in case it was split.
    /// @param poNewMeshes Array of submeshes created in the process. Empty if splitting was not necessary.
    void SplitMesh(const aiMesh* pMesh, std::vector< std::pair< aiMesh*,const aiBone* > >& poNewMeshes) const;

    /// Recursively updates the node's mesh list to account for the changed mesh list
    void UpdateNode(aiNode* pNode) const;

    // -------------------------------------------------------------------
    // Apply transformation to a mesh
    void ApplyTransform(aiMesh* mesh, const aiMatrix4x4& mat)const;

public:
    /** Number of bones present in the scene. */
    unsigned int mNumBones;
    unsigned int mNumBonesCanDoWithout;

    float mThreshold;
    bool mAllOrNone;

    /// Per mesh index: Array of indices of the new submeshes.
    std::vector< std::vector< std::pair< unsigned int,aiNode* > > > mSubMeshIndices;
};

} // end of namespace Assimp

#endif // AI_DEBONEPROCESS_H_INC
