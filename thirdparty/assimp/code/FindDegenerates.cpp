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

/** @file  FindDegenerates.cpp
 *  @brief Implementation of the FindDegenerates post-process step.
*/



// internal headers
#include "ProcessHelper.h"
#include "FindDegenerates.h"
#include <assimp/Exceptional.h>

using namespace Assimp;

//remove mesh at position 'index' from the scene
static void removeMesh(aiScene* pScene, unsigned const index);
//correct node indices to meshes and remove references to deleted mesh
static void updateSceneGraph(aiNode* pNode, unsigned const index);

// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
FindDegeneratesProcess::FindDegeneratesProcess()
: mConfigRemoveDegenerates( false )
, mConfigCheckAreaOfTriangle( false ){
    // empty
}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
FindDegeneratesProcess::~FindDegeneratesProcess() {
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool FindDegeneratesProcess::IsActive( unsigned int pFlags) const {
    return 0 != (pFlags & aiProcess_FindDegenerates);
}

// ------------------------------------------------------------------------------------------------
// Setup import configuration
void FindDegeneratesProcess::SetupProperties(const Importer* pImp) {
    // Get the current value of AI_CONFIG_PP_FD_REMOVE
    mConfigRemoveDegenerates = (0 != pImp->GetPropertyInteger(AI_CONFIG_PP_FD_REMOVE,0));
    mConfigCheckAreaOfTriangle = ( 0 != pImp->GetPropertyInteger(AI_CONFIG_PP_FD_CHECKAREA) );
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void FindDegeneratesProcess::Execute( aiScene* pScene) {
    ASSIMP_LOG_DEBUG("FindDegeneratesProcess begin");
    for (unsigned int i = 0; i < pScene->mNumMeshes;++i)
    {
        //Do not process point cloud, ExecuteOnMesh works only with faces data
        if ((pScene->mMeshes[i]->mPrimitiveTypes != aiPrimitiveType::aiPrimitiveType_POINT) && ExecuteOnMesh(pScene->mMeshes[i])) {
            removeMesh(pScene, i);
            --i; //the current i is removed, do not skip the next one
        }
    }
    ASSIMP_LOG_DEBUG("FindDegeneratesProcess finished");
}

static void removeMesh(aiScene* pScene, unsigned const index) {
    //we start at index and copy the pointers one position forward
    //save the mesh pointer to delete it later
    auto delete_me = pScene->mMeshes[index];
    for (unsigned i = index; i < pScene->mNumMeshes - 1; ++i) {
        pScene->mMeshes[i] = pScene->mMeshes[i+1];
    }
    pScene->mMeshes[pScene->mNumMeshes - 1] = nullptr;
    --(pScene->mNumMeshes);
    delete delete_me;

    //removing a mesh also requires updating all references to it in the scene graph
    updateSceneGraph(pScene->mRootNode, index);
}

static void updateSceneGraph(aiNode* pNode, unsigned const index) {
    for (unsigned i = 0; i < pNode->mNumMeshes; ++i) {
        if (pNode->mMeshes[i] > index) {
            --(pNode->mMeshes[i]);
            continue;
        }
        if (pNode->mMeshes[i] == index) {
            for (unsigned j = i; j < pNode->mNumMeshes -1; ++j) {
                pNode->mMeshes[j] = pNode->mMeshes[j+1];
            }
            --(pNode->mNumMeshes);
            --i;
            continue;
        }
    }
    //recurse to all children
    for (unsigned i = 0; i < pNode->mNumChildren; ++i) {
        updateSceneGraph(pNode->mChildren[i], index);
    }
}

static ai_real heron( ai_real a, ai_real b, ai_real c ) {
    ai_real s = (a + b + c) / 2;
    ai_real area = pow((s * ( s - a ) * ( s - b ) * ( s - c ) ), (ai_real)0.5 );
    return area;
}

static ai_real distance3D( const aiVector3D &vA, aiVector3D &vB ) {
    const ai_real lx = ( vB.x - vA.x );
    const ai_real ly = ( vB.y - vA.y );
    const ai_real lz = ( vB.z - vA.z );
    ai_real a = lx*lx + ly*ly + lz*lz;
    ai_real d = pow( a, (ai_real)0.5 );

    return d;
}

static ai_real calculateAreaOfTriangle( const aiFace& face, aiMesh* mesh ) {
    ai_real area = 0;

    aiVector3D vA( mesh->mVertices[ face.mIndices[ 0 ] ] );
    aiVector3D vB( mesh->mVertices[ face.mIndices[ 1 ] ] );
    aiVector3D vC( mesh->mVertices[ face.mIndices[ 2 ] ] );

    ai_real a( distance3D( vA, vB ) );
    ai_real b( distance3D( vB, vC ) );
    ai_real c( distance3D( vC, vA ) );
    area = heron( a, b, c );

    return area;
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported mesh
bool FindDegeneratesProcess::ExecuteOnMesh( aiMesh* mesh) {
    mesh->mPrimitiveTypes = 0;

    std::vector<bool> remove_me;
    if (mConfigRemoveDegenerates) {
        remove_me.resize( mesh->mNumFaces, false );
    }

    unsigned int deg = 0, limit;
    for ( unsigned int a = 0; a < mesh->mNumFaces; ++a ) {
        aiFace& face = mesh->mFaces[a];
        bool first = true;

        // check whether the face contains degenerated entries
        for (unsigned int i = 0; i < face.mNumIndices; ++i) {
            // Polygons with more than 4 points are allowed to have double points, that is
            // simulating polygons with holes just with concave polygons. However,
            // double points may not come directly after another.
            limit = face.mNumIndices;
            if (face.mNumIndices > 4) {
                limit = std::min( limit, i+2 );
            }

            for (unsigned int t = i+1; t < limit; ++t) {
                if (mesh->mVertices[face.mIndices[ i ] ] == mesh->mVertices[ face.mIndices[ t ] ]) {
                    // we have found a matching vertex position
                    // remove the corresponding index from the array
                    --face.mNumIndices;
                    --limit;
                    for (unsigned int m = t; m < face.mNumIndices; ++m) {
                        face.mIndices[ m ] = face.mIndices[ m+1 ];
                    }
                    --t;

                    // NOTE: we set the removed vertex index to an unique value
                    // to make sure the developer gets notified when his
                    // application attempts to access this data.
                    face.mIndices[ face.mNumIndices ] = 0xdeadbeef;

                    if(first) {
                        ++deg;
                        first = false;
                    }

                    if ( mConfigRemoveDegenerates ) {
                        remove_me[ a ] = true;
                        goto evil_jump_outside; // hrhrhrh ... yeah, this rocks baby!
                    }
                }
            }

            if ( mConfigCheckAreaOfTriangle ) {
                if ( face.mNumIndices == 3 ) {
                    ai_real area = calculateAreaOfTriangle( face, mesh );
                    if ( area < 1e-6 ) {
                        if ( mConfigRemoveDegenerates ) {
                            remove_me[ a ] = true;
                            goto evil_jump_outside;
                        }

                        // todo: check for index which is corrupt.
                    }
                }
            }
        }

        // We need to update the primitive flags array of the mesh.
        switch (face.mNumIndices)
        {
        case 1u:
            mesh->mPrimitiveTypes |= aiPrimitiveType_POINT;
            break;
        case 2u:
            mesh->mPrimitiveTypes |= aiPrimitiveType_LINE;
            break;
        case 3u:
            mesh->mPrimitiveTypes |= aiPrimitiveType_TRIANGLE;
            break;
        default:
            mesh->mPrimitiveTypes |= aiPrimitiveType_POLYGON;
            break;
        };
evil_jump_outside:
        continue;
    }

    // If AI_CONFIG_PP_FD_REMOVE is true, remove degenerated faces from the import
    if (mConfigRemoveDegenerates && deg) {
        unsigned int n = 0;
        for (unsigned int a = 0; a < mesh->mNumFaces; ++a)
        {
            aiFace& face_src = mesh->mFaces[a];
            if (!remove_me[a]) {
                aiFace& face_dest = mesh->mFaces[n++];

                // Do a manual copy, keep the index array
                face_dest.mNumIndices = face_src.mNumIndices;
                face_dest.mIndices    = face_src.mIndices;

                if (&face_src != &face_dest) {
                    // clear source
                    face_src.mNumIndices = 0;
                    face_src.mIndices = nullptr;
                }
            }
            else {
                // Otherwise delete it if we don't need this face
                delete[] face_src.mIndices;
                face_src.mIndices = nullptr;
                face_src.mNumIndices = 0;
            }
        }
        // Just leave the rest of the array unreferenced, we don't care for now
        mesh->mNumFaces = n;
        if (!mesh->mNumFaces) {
            //The whole mesh consists of degenerated faces
            //signal upward, that this mesh should be deleted.
            ASSIMP_LOG_DEBUG("FindDegeneratesProcess removed a mesh full of degenerated primitives");
            return true;
        }
    }

    if (deg && !DefaultLogger::isNullLogger()) {
        ASSIMP_LOG_WARN_F( "Found ", deg, " degenerated primitives");
    }
    return false;
}
