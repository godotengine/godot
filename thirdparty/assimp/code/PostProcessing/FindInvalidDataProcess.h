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

/** @file Defines a post processing step to search an importer's output
 *   for data that is obviously invalid
 */
#ifndef AI_FINDINVALIDDATA_H_INC
#define AI_FINDINVALIDDATA_H_INC

#include "Common/BaseProcess.h"

#include <assimp/types.h>
#include <assimp/anim.h>

struct aiMesh;

class FindInvalidDataProcessTest;

namespace Assimp    {

// ---------------------------------------------------------------------------
/** The FindInvalidData post-processing step. It searches the mesh data
 *  for parts that are obviously invalid and removes them.
 *
 *  Originally this was a workaround for some models written by Blender
 *  which have zero normal vectors. */
class ASSIMP_API FindInvalidDataProcess : public BaseProcess {
public:
    FindInvalidDataProcess();
    ~FindInvalidDataProcess();

    // -------------------------------------------------------------------
    //
    bool IsActive( unsigned int pFlags) const;

    // -------------------------------------------------------------------
    // Setup import settings
    void SetupProperties(const Importer* pImp);

    // -------------------------------------------------------------------
    // Run the step
    void Execute( aiScene* pScene);

    // -------------------------------------------------------------------
    /** Executes the post-processing step on the given mesh
     * @param pMesh The mesh to process.
     * @return 0 - nothing, 1 - removed sth, 2 - please delete me  */
    int ProcessMesh( aiMesh* pMesh);

    // -------------------------------------------------------------------
    /** Executes the post-processing step on the given animation
     * @param anim The animation to process.  */
    void ProcessAnimation (aiAnimation* anim);

    // -------------------------------------------------------------------
    /** Executes the post-processing step on the given anim channel
     * @param anim The animation channel to process.*/
    void ProcessAnimationChannel (aiNodeAnim* anim);

private:
    ai_real configEpsilon;
    bool mIgnoreTexCoods;
};

} // end of namespace Assimp

#endif // AI_AI_FINDINVALIDDATA_H_INC
