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

/** @file Defines a post processing step to triangulate all faces
          with more than three vertices.
 */
#ifndef AI_TRIANGULATEPROCESS_H_INC
#define AI_TRIANGULATEPROCESS_H_INC

#include "BaseProcess.h"

struct aiMesh;

class TriangulateProcessTest;

namespace Assimp {

// ---------------------------------------------------------------------------
/** The TriangulateProcess splits up all faces with more than three indices
 * into triangles. You usually want this to happen because the graphics cards
 * need their data as triangles.
 */
class ASSIMP_API TriangulateProcess : public BaseProcess
{
public:

    TriangulateProcess();
    ~TriangulateProcess();

public:
    // -------------------------------------------------------------------
    /** Returns whether the processing step is present in the given flag field.
     * @param pFlags The processing flags the importer was called with. A bitwise
     *   combination of #aiPostProcessSteps.
     * @return true if the process is present in this flag fields, false if not.
    */
    bool IsActive( unsigned int pFlags) const;

    // -------------------------------------------------------------------
    /** Executes the post processing step on the given imported data.
    * At the moment a process is not supposed to fail.
    * @param pScene The imported data to work at.
    */
    void Execute( aiScene* pScene);

public:
    // -------------------------------------------------------------------
    /** Triangulates the given mesh.
     * @param pMesh The mesh to triangulate.
     */
    bool TriangulateMesh( aiMesh* pMesh);
};

} // end of namespace Assimp

#endif // AI_TRIANGULATEPROCESS_H_INC
