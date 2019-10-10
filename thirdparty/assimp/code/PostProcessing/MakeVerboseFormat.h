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

/** @file Defines a post processing step to bring a given scene
 into the verbose format that is expected by most postprocess steps.
 This is the inverse of the "JoinIdenticalVertices" step. */
#ifndef AI_MAKEVERBOSEFORMAT_H_INC
#define AI_MAKEVERBOSEFORMAT_H_INC

#include "Common/BaseProcess.h"

struct aiMesh;

namespace Assimp    {

// ---------------------------------------------------------------------------
/** MakeVerboseFormatProcess: Class to convert an asset to the verbose
 *  format which is expected by most postprocess steps.
 *
 * This is the inverse of what the "JoinIdenticalVertices" step is doing.
 * This step has no official flag (since it wouldn't make sense to run it
 * during import). It is intended for applications intending to modify the
 * returned aiScene. After this step has been executed, they can execute
 * other postprocess steps on the data. The code might also be useful to
 * quickly adapt code that doesn't result in a verbose representation of
 * the scene data.
 * The step has been added because it was required by the viewer, however
 * it has been moved to the main library since others might find it
 * useful, too. */
class ASSIMP_API_WINONLY MakeVerboseFormatProcess : public BaseProcess
{
public:


    MakeVerboseFormatProcess();
    ~MakeVerboseFormatProcess();

public:

    // -------------------------------------------------------------------
    /** Returns whether the processing step is present in the given flag field.
    * @param pFlags The processing flags the importer was called with. A bitwise
    *   combination of #aiPostProcessSteps.
    * @return true if the process is present in this flag fields, false if not */
    bool IsActive( unsigned int /*pFlags*/ ) const
    {
        // NOTE: There is no direct flag that corresponds to
        // this postprocess step.
        return false;
    }

    // -------------------------------------------------------------------
    /** Executes the post processing step on the given imported data.
    * At the moment a process is not supposed to fail.
    * @param pScene The imported data to work at. */
    void Execute( aiScene* pScene);


private:

    //! Apply the postprocess step to a given submesh
    bool MakeVerboseFormat (aiMesh* pcMesh);
};

} // end of namespace Assimp

#endif // !!AI_KILLNORMALPROCESS_H_INC
