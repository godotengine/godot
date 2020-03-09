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

/** @file Defines a post processing step to sort meshes by the types
    of primitives they contain */
#ifndef AI_SORTBYPTYPEPROCESS_H_INC
#define AI_SORTBYPTYPEPROCESS_H_INC

#include "Common/BaseProcess.h"
#include <assimp/mesh.h>

class SortByPTypeProcessTest;

namespace Assimp    {


// ---------------------------------------------------------------------------
/** SortByPTypeProcess: Sorts meshes by the types of primitives they contain.
 *  A mesh with 5 lines, 3 points and 145 triangles would be split in 3
 * submeshes.
*/
class ASSIMP_API SortByPTypeProcess : public BaseProcess {
public:
    SortByPTypeProcess();
    ~SortByPTypeProcess();

    // -------------------------------------------------------------------
    bool IsActive( unsigned int pFlags) const;

    // -------------------------------------------------------------------
    void Execute( aiScene* pScene);

    // -------------------------------------------------------------------
    void SetupProperties(const Importer* pImp);

private:
    int mConfigRemoveMeshes;
};


} // end of namespace Assimp

#endif // !!AI_SORTBYPTYPEPROCESS_H_INC
