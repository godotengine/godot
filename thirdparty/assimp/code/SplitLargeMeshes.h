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

/** @file Defines a post processing step to split large meshes into submeshes
 */
#ifndef AI_SPLITLARGEMESHES_H_INC
#define AI_SPLITLARGEMESHES_H_INC

#include <vector>
#include "BaseProcess.h"

#include <assimp/mesh.h>
#include <assimp/scene.h>

class SplitLargeMeshesTest;

namespace Assimp
{

class SplitLargeMeshesProcess_Triangle;
class SplitLargeMeshesProcess_Vertex;

// NOTE: If you change these limits, don't forget to change the
// corresponding values in all Assimp ports

// **********************************************************
// Java: ConfigProperty.java,
//  ConfigProperty.DEFAULT_VERTEX_SPLIT_LIMIT
//  ConfigProperty.DEFAULT_TRIANGLE_SPLIT_LIMIT
// **********************************************************

// default limit for vertices
#if (!defined AI_SLM_DEFAULT_MAX_VERTICES)
#   define AI_SLM_DEFAULT_MAX_VERTICES      1000000
#endif

// default limit for triangles
#if (!defined AI_SLM_DEFAULT_MAX_TRIANGLES)
#   define AI_SLM_DEFAULT_MAX_TRIANGLES     1000000
#endif

// ---------------------------------------------------------------------------
/** Post-processing filter to split large meshes into sub-meshes
 *
 * Applied BEFORE the JoinVertices-Step occurs.
 * Returns NON-UNIQUE vertices, splits by triangle number.
*/
class ASSIMP_API SplitLargeMeshesProcess_Triangle : public BaseProcess
{
    friend class SplitLargeMeshesProcess_Vertex;

public:

    SplitLargeMeshesProcess_Triangle();
    ~SplitLargeMeshesProcess_Triangle();

public:
    // -------------------------------------------------------------------
    /** Returns whether the processing step is present in the given flag.
    * @param pFlags The processing flags the importer was called with. A
    *   bitwise combination of #aiPostProcessSteps.
    * @return true if the process is present in this flag fields,
    *   false if not.
    */
    bool IsActive( unsigned int pFlags) const;


    // -------------------------------------------------------------------
    /** Called prior to ExecuteOnScene().
    * The function is a request to the process to update its configuration
    * basing on the Importer's configuration property list.
    */
    virtual void SetupProperties(const Importer* pImp);


    //! Set the split limit - needed for unit testing
    inline void SetLimit(unsigned int l)
        {LIMIT = l;}

    //! Get the split limit
    inline unsigned int GetLimit() const
        {return LIMIT;}

public:

    // -------------------------------------------------------------------
    /** Executes the post processing step on the given imported data.
    * At the moment a process is not supposed to fail.
    * @param pScene The imported data to work at.
    */
    void Execute( aiScene* pScene);

    // -------------------------------------------------------------------
    //! Apply the algorithm to a given mesh
    void SplitMesh (unsigned int a, aiMesh* pcMesh,
        std::vector<std::pair<aiMesh*, unsigned int> >& avList);

    // -------------------------------------------------------------------
    //! Update a node in the asset after a few of its meshes
    //! have been split
    static void UpdateNode(aiNode* pcNode,
        const std::vector<std::pair<aiMesh*, unsigned int> >& avList);

public:
    //! Triangle limit
    unsigned int LIMIT;
};


// ---------------------------------------------------------------------------
/** Post-processing filter to split large meshes into sub-meshes
 *
 * Applied AFTER the JoinVertices-Step occurs.
 * Returns UNIQUE vertices, splits by vertex number.
*/
class ASSIMP_API SplitLargeMeshesProcess_Vertex : public BaseProcess
{
public:

    SplitLargeMeshesProcess_Vertex();
    ~SplitLargeMeshesProcess_Vertex();

public:
    // -------------------------------------------------------------------
    /** Returns whether the processing step is present in the given flag field.
    * @param pFlags The processing flags the importer was called with. A bitwise
    *   combination of #aiPostProcessSteps.
    * @return true if the process is present in this flag fields, false if not.
    */
    bool IsActive( unsigned int pFlags) const;

    // -------------------------------------------------------------------
    /** Called prior to ExecuteOnScene().
    * The function is a request to the process to update its configuration
    * basing on the Importer's configuration property list.
    */
    virtual void SetupProperties(const Importer* pImp);


    //! Set the split limit - needed for unit testing
    inline void SetLimit(unsigned int l)
        {LIMIT = l;}

    //! Get the split limit
    inline unsigned int GetLimit() const
        {return LIMIT;}

public:

    // -------------------------------------------------------------------
    /** Executes the post processing step on the given imported data.
    * At the moment a process is not supposed to fail.
    * @param pScene The imported data to work at.
    */
    void Execute( aiScene* pScene);

    // -------------------------------------------------------------------
    //! Apply the algorithm to a given mesh
    void SplitMesh (unsigned int a, aiMesh* pcMesh,
        std::vector<std::pair<aiMesh*, unsigned int> >& avList);

    // NOTE: Reuse SplitLargeMeshesProcess_Triangle::UpdateNode()

public:
    //! Triangle limit
    unsigned int LIMIT;
};

} // end of namespace Assimp

#endif // !!AI_SPLITLARGEMESHES_H_INC
