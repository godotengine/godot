/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team



All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the following
conditions are met:

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
---------------------------------------------------------------------------
*/

/** @file  OptimizeMeshes.cpp
 *  @brief Implementation of the aiProcess_OptimizeMeshes step
 */


#ifndef ASSIMP_BUILD_NO_OPTIMIZEMESHES_PROCESS


#include "OptimizeMeshes.h"
#include "ProcessHelper.h"
#include <assimp/SceneCombiner.h>
#include <assimp/Exceptional.h>

using namespace Assimp;

static const unsigned int NotSet   = 0xffffffff;
static const unsigned int DeadBeef = 0xdeadbeef;

// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
OptimizeMeshesProcess::OptimizeMeshesProcess()
    : mScene()
    , pts(false)
    , max_verts( NotSet )
    , max_faces( NotSet ) {
    // empty
}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
OptimizeMeshesProcess::~OptimizeMeshesProcess() {
    // empty
}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool OptimizeMeshesProcess::IsActive( unsigned int pFlags) const
{
    // Our behaviour needs to be different if the SortByPType or SplitLargeMeshes
    // steps are active. Thus we need to query their flags here and store the
    // information, although we're breaking const-correctness.
    // That's a serious design flaw, consider redesign.
    if( 0 != (pFlags & aiProcess_OptimizeMeshes) ) {
        pts = (0 != (pFlags & aiProcess_SortByPType));
        max_verts = ( 0 != ( pFlags & aiProcess_SplitLargeMeshes ) ) ? DeadBeef : max_verts;
        return true;
    }
    return false;
}

// ------------------------------------------------------------------------------------------------
// Setup properties for the post-processing step
void OptimizeMeshesProcess::SetupProperties(const Importer* pImp)
{
    if( max_verts == DeadBeef /* magic hack */ ) {
        max_faces = pImp->GetPropertyInteger(AI_CONFIG_PP_SLM_TRIANGLE_LIMIT,AI_SLM_DEFAULT_MAX_TRIANGLES);
        max_verts = pImp->GetPropertyInteger(AI_CONFIG_PP_SLM_VERTEX_LIMIT,AI_SLM_DEFAULT_MAX_VERTICES);
    }
}

// ------------------------------------------------------------------------------------------------
// Execute step
void OptimizeMeshesProcess::Execute( aiScene* pScene)
{
    const unsigned int num_old = pScene->mNumMeshes;
    if (num_old <= 1) {
        ASSIMP_LOG_DEBUG("Skipping OptimizeMeshesProcess");
        return;
    }

    ASSIMP_LOG_DEBUG("OptimizeMeshesProcess begin");
    mScene = pScene;

    // need to clear persistent members from previous runs
    merge_list.resize( 0 );
    output.resize( 0 );

    // ensure we have the right sizes
    merge_list.reserve(pScene->mNumMeshes);
    output.reserve(pScene->mNumMeshes);

    // Prepare lookup tables
    meshes.resize(pScene->mNumMeshes);
    FindInstancedMeshes(pScene->mRootNode);
    if( max_verts == DeadBeef ) /* undo the magic hack */
        max_verts = NotSet;

    // ... instanced meshes are immediately processed and added to the output list
    for (unsigned int i = 0, n = 0; i < pScene->mNumMeshes;++i) {
        meshes[i].vertex_format = GetMeshVFormatUnique(pScene->mMeshes[i]);

        if (meshes[i].instance_cnt > 1 && meshes[i].output_id == NotSet ) {
            meshes[i].output_id = n++;
            output.push_back(mScene->mMeshes[i]);
        }
    }

    // and process all nodes in the scenegraph recursively
    ProcessNode(pScene->mRootNode);
    if (!output.size()) {
        throw DeadlyImportError("OptimizeMeshes: No meshes remaining; there's definitely something wrong");
    }

    meshes.resize( 0 );
    ai_assert(output.size() <= num_old);

    mScene->mNumMeshes = static_cast<unsigned int>(output.size());
    std::copy(output.begin(),output.end(),mScene->mMeshes);

    if (output.size() != num_old) {
        ASSIMP_LOG_DEBUG_F("OptimizeMeshesProcess finished. Input meshes: ", num_old, ", Output meshes: ", pScene->mNumMeshes);
    } else {
        ASSIMP_LOG_DEBUG( "OptimizeMeshesProcess finished" );
    }
}

// ------------------------------------------------------------------------------------------------
// Process meshes for a single node
void OptimizeMeshesProcess::ProcessNode( aiNode* pNode)
{
    for (unsigned int i = 0; i < pNode->mNumMeshes;++i) {
        unsigned int& im = pNode->mMeshes[i];

        if (meshes[im].instance_cnt > 1) {
            im = meshes[im].output_id;
        }
        else  {
            merge_list.resize( 0 );
            unsigned int verts = 0, faces = 0;

            // Find meshes to merge with us
            for (unsigned int a = i+1; a < pNode->mNumMeshes;++a) {
                unsigned int am = pNode->mMeshes[a];
                if (meshes[am].instance_cnt == 1 && CanJoin(im,am,verts,faces)) {

                    merge_list.push_back(mScene->mMeshes[am]);
                    verts += mScene->mMeshes[am]->mNumVertices;
                    faces += mScene->mMeshes[am]->mNumFaces;

                    pNode->mMeshes[a] = pNode->mMeshes[pNode->mNumMeshes - 1];
                    --pNode->mNumMeshes;
                    --a;
                }
            }

            // and merge all meshes which we found, replace the old ones
            if (!merge_list.empty()) {
                merge_list.push_back(mScene->mMeshes[im]);

                aiMesh* out;
                SceneCombiner::MergeMeshes(&out,0,merge_list.begin(),merge_list.end());
                output.push_back(out);
            } else {
                output.push_back(mScene->mMeshes[im]);
            }
            im = static_cast<unsigned int>(output.size()-1);
        }
    }


    for( unsigned int i = 0; i < pNode->mNumChildren; ++i ) {
        ProcessNode( pNode->mChildren[ i ] );
    }
}

// ------------------------------------------------------------------------------------------------
// Check whether two meshes can be joined
bool OptimizeMeshesProcess::CanJoin ( unsigned int a, unsigned int b, unsigned int verts, unsigned int faces )
{
    if (meshes[a].vertex_format != meshes[b].vertex_format)
        return false;

    aiMesh* ma = mScene->mMeshes[a], *mb = mScene->mMeshes[b];

    if ((NotSet != max_verts && verts+mb->mNumVertices > max_verts) ||
        (NotSet != max_faces && faces+mb->mNumFaces    > max_faces)) {
        return false;
    }

    // Never merge unskinned meshes with skinned meshes
    if (ma->mMaterialIndex != mb->mMaterialIndex || ma->HasBones() != mb->HasBones())
        return false;

    // Never merge meshes with different kinds of primitives if SortByPType did already
    // do its work. We would destroy everything again ...
    if (pts && ma->mPrimitiveTypes != mb->mPrimitiveTypes)
        return false;

    // If both meshes are skinned, check whether we have many bones defined in both meshes.
    // If yes, we can join them.
    if (ma->HasBones()) {
        // TODO
        return false;
    }
    return true;
}

// ------------------------------------------------------------------------------------------------
// Build a LUT of all instanced meshes
void OptimizeMeshesProcess::FindInstancedMeshes (aiNode* pNode)
{
    for( unsigned int i = 0; i < pNode->mNumMeshes; ++i ) {
        ++meshes[ pNode->mMeshes[ i ] ].instance_cnt;
    }

    for( unsigned int i = 0; i < pNode->mNumChildren; ++i ) {
        FindInstancedMeshes( pNode->mChildren[ i ] );
    }
}

// ------------------------------------------------------------------------------------------------

#endif // !! ASSIMP_BUILD_NO_OPTIMIZEMESHES_PROCESS
