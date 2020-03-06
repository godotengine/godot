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

/** @file Defines a post processing step to reorder faces for
 better cache locality*/
#ifndef AI_IMPROVECACHELOCALITY_H_INC
#define AI_IMPROVECACHELOCALITY_H_INC

#include "Common/BaseProcess.h"

#include <assimp/types.h>

struct aiMesh;

namespace Assimp
{

// ---------------------------------------------------------------------------
/** The ImproveCacheLocalityProcess reorders all faces for improved vertex
 *  cache locality. It tries to arrange all faces to fans and to render
 *  faces which share vertices directly one after the other.
 *
 *  @note This step expects triagulated input data.
 */
class ImproveCacheLocalityProcess : public BaseProcess
{
public:

    ImproveCacheLocalityProcess();
    ~ImproveCacheLocalityProcess();

public:

    // -------------------------------------------------------------------
    // Check whether the pp step is active
    bool IsActive( unsigned int pFlags) const;

    // -------------------------------------------------------------------
    // Executes the pp step on a given scene
    void Execute( aiScene* pScene);

    // -------------------------------------------------------------------
    // Configures the pp step
    void SetupProperties(const Importer* pImp);

protected:
    // -------------------------------------------------------------------
    /** Executes the postprocessing step on the given mesh
     * @param pMesh The mesh to process.
     * @param meshNum Index of the mesh to process
     */
    ai_real ProcessMesh( aiMesh* pMesh, unsigned int meshNum);

private:
    //! Configuration parameter: specifies the size of the cache to
    //! optimize the vertex data for.
    unsigned int mConfigCacheDepth;
};

} // end of namespace Assimp

#endif // AI_IMPROVECACHELOCALITY_H_INC
