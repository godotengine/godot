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

/** @file  OptimizeGraph.cpp
 *  @brief Implementation of the aiProcess_OptimizGraph step
 */


#ifndef ASSIMP_BUILD_NO_OPTIMIZEGRAPH_PROCESS

#include "OptimizeGraph.h"
#include "ProcessHelper.h"
#include <assimp/SceneCombiner.h>
#include <assimp/Exceptional.h>
#include <stdio.h>

using namespace Assimp;

#define AI_RESERVED_NODE_NAME "$Reserved_And_Evil"

/* AI_OG_USE_HASHING enables the use of hashing to speed-up std::set lookups.
 * The unhashed variant should be faster, except for *very* large data sets
 */
#ifdef AI_OG_USE_HASHING
    // Use our standard hashing function to compute the hash
#   define AI_OG_GETKEY(str) SuperFastHash(str.data,str.length)
#else
    // Otherwise hope that std::string will utilize a static buffer
    // for shorter node names. This would avoid endless heap copying.
#   define AI_OG_GETKEY(str) std::string(str.data)
#endif

// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
OptimizeGraphProcess::OptimizeGraphProcess()
: mScene()
, nodes_in()
, nodes_out()
, count_merged() {
    // empty
}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
OptimizeGraphProcess::~OptimizeGraphProcess() {
    // empty
}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool OptimizeGraphProcess::IsActive( unsigned int pFlags) const {
    return (0 != (pFlags & aiProcess_OptimizeGraph));
}

// ------------------------------------------------------------------------------------------------
// Setup properties for the post-processing step
void OptimizeGraphProcess::SetupProperties(const Importer* pImp) {
    // Get value of AI_CONFIG_PP_OG_EXCLUDE_LIST
    std::string tmp = pImp->GetPropertyString(AI_CONFIG_PP_OG_EXCLUDE_LIST,"");
    AddLockedNodeList(tmp);
}

// ------------------------------------------------------------------------------------------------
// Collect new children
void OptimizeGraphProcess::CollectNewChildren(aiNode* nd, std::list<aiNode*>& nodes) {
    nodes_in += nd->mNumChildren;

    // Process children
    std::list<aiNode*> child_nodes;
    for (unsigned int i = 0; i < nd->mNumChildren; ++i) {
        CollectNewChildren(nd->mChildren[i],child_nodes);
        nd->mChildren[i] = nullptr;
    }

    // Check whether we need this node; if not we can replace it by our own children (warn, danger of incest).
    if (locked.find(AI_OG_GETKEY(nd->mName)) == locked.end() ) {
        for (std::list<aiNode*>::iterator it = child_nodes.begin(); it != child_nodes.end();) {

            if (locked.find(AI_OG_GETKEY((*it)->mName)) == locked.end()) {
                (*it)->mTransformation = nd->mTransformation * (*it)->mTransformation;
                nodes.push_back(*it);

                it = child_nodes.erase(it);
                continue;
            }
            ++it;
        }

        if (nd->mNumMeshes || !child_nodes.empty()) {
            nodes.push_back(nd);
        } else {
            delete nd; /* bye, node */
            return;
        }
    } else {

        // Retain our current position in the hierarchy
        nodes.push_back(nd);

        // Now check for possible optimizations in our list of child nodes. join as many as possible
        aiNode* join_master = NULL;
        aiMatrix4x4 inv;

        const LockedSetType::const_iterator end = locked.end();

        std::list<aiNode*> join;
        for (std::list<aiNode*>::iterator it = child_nodes.begin(); it != child_nodes.end();)   {
            aiNode* child = *it;
            if (child->mNumChildren == 0 && locked.find(AI_OG_GETKEY(child->mName)) == end) {

                // There may be no instanced meshes
                unsigned int n = 0;
                for (; n < child->mNumMeshes;++n) {
                    if (meshes[child->mMeshes[n]] > 1) {
                        break;
                    }
                }
                if (n == child->mNumMeshes) {
                    if (!join_master) {
                        join_master = child;
                        inv = join_master->mTransformation;
                        inv.Inverse();
                    } else {
                        child->mTransformation = inv * child->mTransformation ;

                        join.push_back(child);
                        it = child_nodes.erase(it);
                        continue;
                    }
                }
            }
            ++it;
        }
        if (join_master && !join.empty()) {
            join_master->mName.length = ::ai_snprintf(join_master->mName.data, MAXLEN, "$MergedNode_%i",count_merged++);

            unsigned int out_meshes = 0;
            for (std::list<aiNode*>::iterator it = join.begin(); it != join.end(); ++it) {
                out_meshes += (*it)->mNumMeshes;
            }

            // copy all mesh references in one array
            if (out_meshes) {
                unsigned int* meshes = new unsigned int[out_meshes+join_master->mNumMeshes], *tmp = meshes;
                for (unsigned int n = 0; n < join_master->mNumMeshes;++n) {
                    *tmp++ = join_master->mMeshes[n];
                }

                for (std::list<aiNode*>::iterator it = join.begin(); it != join.end(); ++it) {
                    for (unsigned int n = 0; n < (*it)->mNumMeshes; ++n) {

                        *tmp = (*it)->mMeshes[n];
                        aiMesh* mesh = mScene->mMeshes[*tmp++];

                        // manually move the mesh into the right coordinate system
                        const aiMatrix3x3 IT = aiMatrix3x3( (*it)->mTransformation ).Inverse().Transpose();
                        for (unsigned int a = 0; a < mesh->mNumVertices; ++a) {

                            mesh->mVertices[a] *= (*it)->mTransformation;

                            if (mesh->HasNormals())
                                mesh->mNormals[a] *= IT;

                            if (mesh->HasTangentsAndBitangents()) {
                                mesh->mTangents[a] *= IT;
                                mesh->mBitangents[a] *= IT;
                            }
                        }
                    }
                    delete *it; // bye, node
                }
                delete[] join_master->mMeshes;
                join_master->mMeshes = meshes;
                join_master->mNumMeshes += out_meshes;
            }
        }
    }
    // reassign children if something changed
    if (child_nodes.empty() || child_nodes.size() > nd->mNumChildren) {

        delete[] nd->mChildren;

        if (!child_nodes.empty()) {
            nd->mChildren = new aiNode*[child_nodes.size()];
        }
        else nd->mChildren = nullptr;
    }

    nd->mNumChildren = static_cast<unsigned int>(child_nodes.size());

    if (nd->mChildren) {
        aiNode** tmp = nd->mChildren;
        for (std::list<aiNode*>::iterator it = child_nodes.begin(); it != child_nodes.end(); ++it) {
            aiNode* node = *tmp++ = *it;
            node->mParent = nd;
        }
    }

    nodes_out += static_cast<unsigned int>(child_nodes.size());
}

// ------------------------------------------------------------------------------------------------
// Execute the post-processing step on the given scene
void OptimizeGraphProcess::Execute( aiScene* pScene) {
    ASSIMP_LOG_DEBUG("OptimizeGraphProcess begin");
    nodes_in = nodes_out = count_merged = 0;
    mScene = pScene;

    meshes.resize(pScene->mNumMeshes,0);
    FindInstancedMeshes(pScene->mRootNode);

    // build a blacklist of identifiers. If the name of a node matches one of these, we won't touch it
    locked.clear();
    for (std::list<std::string>::const_iterator it = locked_nodes.begin(); it != locked_nodes.end(); ++it) {
#ifdef AI_OG_USE_HASHING
        locked.insert(SuperFastHash((*it).c_str()));
#else
        locked.insert(*it);
#endif
    }

    for (unsigned int i = 0; i < pScene->mNumAnimations; ++i) {
        for (unsigned int a = 0; a < pScene->mAnimations[i]->mNumChannels; ++a) {
            aiNodeAnim* anim = pScene->mAnimations[i]->mChannels[a];
            locked.insert(AI_OG_GETKEY(anim->mNodeName));
        }
    }

    for (unsigned int i = 0; i < pScene->mNumMeshes; ++i) {
        for (unsigned int a = 0; a < pScene->mMeshes[i]->mNumBones; ++a) {

            aiBone* bone = pScene->mMeshes[i]->mBones[a];
            locked.insert(AI_OG_GETKEY(bone->mName));

            // HACK: Meshes referencing bones may not be transformed; we need to look them.
            // The easiest way to do this is to increase their reference counters ...
            meshes[i] += 2;
        }
    }

    for (unsigned int i = 0; i < pScene->mNumCameras; ++i) {
        aiCamera* cam = pScene->mCameras[i];
        locked.insert(AI_OG_GETKEY(cam->mName));
    }

    for (unsigned int i = 0; i < pScene->mNumLights; ++i) {
        aiLight* lgh = pScene->mLights[i];
        locked.insert(AI_OG_GETKEY(lgh->mName));
    }

    // Insert a dummy master node and make it read-only
    aiNode* dummy_root = new aiNode(AI_RESERVED_NODE_NAME);
    locked.insert(AI_OG_GETKEY(dummy_root->mName));

    const aiString prev = pScene->mRootNode->mName;
    pScene->mRootNode->mParent = dummy_root;

    dummy_root->mChildren = new aiNode*[dummy_root->mNumChildren = 1];
    dummy_root->mChildren[0] = pScene->mRootNode;

    // Do our recursive processing of scenegraph nodes. For each node collect
    // a fully new list of children and allow their children to place themselves
    // on the same hierarchy layer as their parents.
    std::list<aiNode*> nodes;
    CollectNewChildren (dummy_root,nodes);

    ai_assert(nodes.size() == 1);

    if (dummy_root->mNumChildren == 0) {
        pScene->mRootNode = NULL;
        throw DeadlyImportError("After optimizing the scene graph, no data remains");
    }

    if (dummy_root->mNumChildren > 1) {
        pScene->mRootNode = dummy_root;

        // Keep the dummy node but assign the name of the old root node to it
        pScene->mRootNode->mName = prev;
    }
    else {

        // Remove the dummy root node again.
        pScene->mRootNode = dummy_root->mChildren[0];

        dummy_root->mChildren[0] = NULL;
        delete dummy_root;
    }

    pScene->mRootNode->mParent = NULL;
    if (!DefaultLogger::isNullLogger()) {
        if ( nodes_in != nodes_out) {
            ASSIMP_LOG_INFO_F("OptimizeGraphProcess finished; Input nodes: ", nodes_in, ", Output nodes: ", nodes_out);
        } else {
            ASSIMP_LOG_DEBUG("OptimizeGraphProcess finished");
        }
    }
    meshes.clear();
    locked.clear();
}

// ------------------------------------------------------------------------------------------------
// Build a LUT of all instanced meshes
void OptimizeGraphProcess::FindInstancedMeshes (aiNode* pNode)
{
    for (unsigned int i = 0; i < pNode->mNumMeshes;++i) {
        ++meshes[pNode->mMeshes[i]];
    }

    for (unsigned int i = 0; i < pNode->mNumChildren; ++i)
        FindInstancedMeshes(pNode->mChildren[i]);
}

#endif // !! ASSIMP_BUILD_NO_OPTIMIZEGRAPH_PROCESS
