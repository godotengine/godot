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

/** @file  OptimizeMeshes.h
 *  @brief Declares a post processing step to join meshes, if possible
 */
#ifndef AI_OPTIMIZEMESHESPROCESS_H_INC
#define AI_OPTIMIZEMESHESPROCESS_H_INC

#include "BaseProcess.h"
#include <assimp/types.h>
#include <vector>

struct aiMesh;
struct aiNode;
class OptimizeMeshesProcessTest;

namespace Assimp    {

// ---------------------------------------------------------------------------
/** @brief Postprocessing step to optimize mesh usage
 *
 *  The implementation looks for meshes that could be joined and joins them.
 *  Usually this will reduce the number of drawcalls.
 *
 *  @note Instanced meshes are currently not processed.
 */
class OptimizeMeshesProcess : public BaseProcess
{
public:
    /// @brief  The class constructor.
    OptimizeMeshesProcess();

    /// @brief  The class destcructor,
    ~OptimizeMeshesProcess();


    /** @brief Internal utility to store additional mesh info
     */
    struct MeshInfo {
        MeshInfo() AI_NO_EXCEPT
        : instance_cnt(0)
        , vertex_format(0)
        , output_id(0xffffffff) {
            // empty
        }

        //! Number of times this mesh is referenced
        unsigned int instance_cnt;

        //! Vertex format id
        unsigned int vertex_format;

        //! Output ID
        unsigned int output_id;
    };

public:
    // -------------------------------------------------------------------
    bool IsActive( unsigned int pFlags) const;

    // -------------------------------------------------------------------
    void Execute( aiScene* pScene);

    // -------------------------------------------------------------------
    void SetupProperties(const Importer* pImp);


    // -------------------------------------------------------------------
    /** @brief Specify whether you want meshes with different
     *   primitive types to be merged as well.
     *
     *  IsActive() sets this property automatically to true if the
     *  aiProcess_SortByPType flag is found.
     */
    void EnablePrimitiveTypeSorting(bool enable) {
        pts = enable;
    }

    // Getter
    bool IsPrimitiveTypeSortingEnabled () const {
        return pts;
    }


    // -------------------------------------------------------------------
    /** @brief Specify a maximum size of a single output mesh.
     *
     *  If a single input mesh already exceeds this limit, it won't
     *  be split.
     *  @param verts Maximum number of vertices per mesh
     *  @param faces Maximum number of faces per mesh
     */
    void SetPreferredMeshSizeLimit (unsigned int verts, unsigned int faces)
    {
        max_verts = verts;
        max_faces = faces;
    }


protected:

    // -------------------------------------------------------------------
    /** @brief Do the actual optimization on all meshes of this node
     *  @param pNode Node we're working with
     */
    void ProcessNode( aiNode* pNode);

    // -------------------------------------------------------------------
    /** @brief Returns true if b can be joined with a
     *
     *  @param verts Number of output verts up to now
     *  @param faces Number of output faces up to now
     */
    bool CanJoin ( unsigned int a, unsigned int b,
        unsigned int verts, unsigned int faces );

    // -------------------------------------------------------------------
    /** @brief Find instanced meshes, for the moment we're excluding
     *   them from all optimizations
     */
    void FindInstancedMeshes (aiNode* pNode);

private:

    //! Scene we're working with
    aiScene* mScene;

    //! Per mesh info
    std::vector<MeshInfo> meshes;

    //! Output meshes
    std::vector<aiMesh*> output;

    //! @see EnablePrimitiveTypeSorting
    mutable bool pts;

    //! @see SetPreferredMeshSizeLimit
    mutable unsigned int max_verts,max_faces;

    //! Temporary storage
    std::vector<aiMesh*> merge_list;
};

} // end of namespace Assimp

#endif // AI_CALCTANGENTSPROCESS_H_INC
