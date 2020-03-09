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

/** @file Defines a post processing step to compute vertex normals
    for all loaded vertizes */
#ifndef AI_GENVERTEXNORMALPROCESS_H_INC
#define AI_GENVERTEXNORMALPROCESS_H_INC

#include "Common/assbin_chunks.h"
#include "Common/BaseProcess.h"

#include <assimp/mesh.h>

// Forward declarations
class GenNormalsTest;

namespace Assimp {

// ---------------------------------------------------------------------------
/** The GenFaceNormalsProcess computes vertex normals for all vertices
*/
class ASSIMP_API GenVertexNormalsProcess : public BaseProcess {
public:
    GenVertexNormalsProcess();
    ~GenVertexNormalsProcess();

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

    // -------------------------------------------------------------------
    /** Executes the post processing step on the given imported data.
    * At the moment a process is not supposed to fail.
    * @param pScene The imported data to work at.
    */
    void Execute( aiScene* pScene);


    // setter for configMaxAngle
    inline void SetMaxSmoothAngle(ai_real f) {
        configMaxAngle =f;
    }

    // -------------------------------------------------------------------
    /** Computes normals for a specific mesh
    *  @param pcMesh Mesh
    *  @param meshIndex Index of the mesh
    *  @return true if vertex normals have been computed
    */
    bool GenMeshVertexNormals (aiMesh* pcMesh, unsigned int meshIndex);

private:
    /** Configuration option: maximum smoothing angle, in radians*/
    ai_real configMaxAngle;
    mutable bool force_ = false;
};

} // end of namespace Assimp

#endif // !!AI_GENVERTEXNORMALPROCESS_H_INC
