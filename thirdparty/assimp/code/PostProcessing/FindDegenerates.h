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

/** @file Defines a post processing step to search all meshes for
  degenerated faces */
#ifndef AI_FINDDEGENERATESPROCESS_H_INC
#define AI_FINDDEGENERATESPROCESS_H_INC

#include "Common/BaseProcess.h"

#include <assimp/mesh.h>

class FindDegeneratesProcessTest;
namespace Assimp    {


// ---------------------------------------------------------------------------
/** FindDegeneratesProcess: Searches a mesh for degenerated triangles.
*/
class ASSIMP_API FindDegeneratesProcess : public BaseProcess {
public:
    FindDegeneratesProcess();
    ~FindDegeneratesProcess();

    // -------------------------------------------------------------------
    // Check whether step is active
    bool IsActive( unsigned int pFlags) const;

    // -------------------------------------------------------------------
    // Execute step on a given scene
    void Execute( aiScene* pScene);

    // -------------------------------------------------------------------
    // Setup import settings
    void SetupProperties(const Importer* pImp);

    // -------------------------------------------------------------------
    // Execute step on a given mesh
    ///@returns true if the current mesh should be deleted, false otherwise
    bool ExecuteOnMesh( aiMesh* mesh);

    // -------------------------------------------------------------------
    /// @brief Enable the instant removal of degenerated primitives
    /// @param enabled  true for enabled.
    void EnableInstantRemoval(bool enabled);

    // -------------------------------------------------------------------
    /// @brief Check whether instant removal is currently enabled
    /// @return The instant removal state.
    bool IsInstantRemoval() const;

    // -------------------------------------------------------------------
    /// @brief Enable the area check for triangles.
    /// @param enabled  true for enabled.
    void EnableAreaCheck( bool enabled );

    // -------------------------------------------------------------------
    /// @brief Check whether the area check is enabled.
    /// @return The area check state.
    bool isAreaCheckEnabled() const;

private:
    //! Configuration option: remove degenerates faces immediately
    bool mConfigRemoveDegenerates;
    //! Configuration option: check for area
    bool mConfigCheckAreaOfTriangle;
};

inline
void FindDegeneratesProcess::EnableInstantRemoval(bool enabled) {
    mConfigRemoveDegenerates = enabled;
}

inline
bool FindDegeneratesProcess::IsInstantRemoval() const {
    return mConfigRemoveDegenerates;
}

inline
void FindDegeneratesProcess::EnableAreaCheck( bool enabled ) {
    mConfigCheckAreaOfTriangle = enabled;
}

inline
bool FindDegeneratesProcess::isAreaCheckEnabled() const {
    return mConfigCheckAreaOfTriangle;
}

} // Namespace Assimp

#endif // !! AI_FINDDEGENERATESPROCESS_H_INC
