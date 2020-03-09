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

/** @file  FindInstancesProcess.cpp
 *  @brief Implementation of the aiProcess_FindInstances postprocessing step
*/


#include "FindInstancesProcess.h"
#include <memory>
#include <stdio.h>

using namespace Assimp;

// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
FindInstancesProcess::FindInstancesProcess()
:   configSpeedFlag (false)
{}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
FindInstancesProcess::~FindInstancesProcess()
{}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool FindInstancesProcess::IsActive( unsigned int pFlags) const
{
    // FindInstances makes absolutely no sense together with PreTransformVertices
    // fixme: spawn error message somewhere else?
    return 0 != (pFlags & aiProcess_FindInstances) && 0 == (pFlags & aiProcess_PreTransformVertices);
}

// ------------------------------------------------------------------------------------------------
// Setup properties for the step
void FindInstancesProcess::SetupProperties(const Importer* pImp)
{
    // AI_CONFIG_FAVOUR_SPEED
    configSpeedFlag = (0 != pImp->GetPropertyInteger(AI_CONFIG_FAVOUR_SPEED,0));
}

// ------------------------------------------------------------------------------------------------
// Compare the bones of two meshes
bool CompareBones(const aiMesh* orig, const aiMesh* inst)
{
    for (unsigned int i = 0; i < orig->mNumBones;++i) {
        aiBone* aha = orig->mBones[i];
        aiBone* oha = inst->mBones[i];

        if (aha->mNumWeights   != oha->mNumWeights   ||
            aha->mOffsetMatrix != oha->mOffsetMatrix) {
            return false;
        }

        // compare weight per weight ---
        for (unsigned int n = 0; n < aha->mNumWeights;++n) {
            if  (aha->mWeights[n].mVertexId != oha->mWeights[n].mVertexId ||
                (aha->mWeights[n].mWeight - oha->mWeights[n].mWeight) < 10e-3f) {
                return false;
            }
        }
    }
    return true;
}

// ------------------------------------------------------------------------------------------------
// Update mesh indices in the node graph
void UpdateMeshIndices(aiNode* node, unsigned int* lookup)
{
    for (unsigned int n = 0; n < node->mNumMeshes;++n)
        node->mMeshes[n] = lookup[node->mMeshes[n]];

    for (unsigned int n = 0; n < node->mNumChildren;++n)
        UpdateMeshIndices(node->mChildren[n],lookup);
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void FindInstancesProcess::Execute( aiScene* pScene)
{
    ASSIMP_LOG_DEBUG("FindInstancesProcess begin");
    if (pScene->mNumMeshes) {

        // use a pseudo hash for all meshes in the scene to quickly find
        // the ones which are possibly equal. This step is executed early
        // in the pipeline, so we could, depending on the file format,
        // have several thousand small meshes. That's too much for a brute
        // everyone-against-everyone check involving up to 10 comparisons
        // each.
        std::unique_ptr<uint64_t[]> hashes (new uint64_t[pScene->mNumMeshes]);
        std::unique_ptr<unsigned int[]> remapping (new unsigned int[pScene->mNumMeshes]);

        unsigned int numMeshesOut = 0;
        for (unsigned int i = 0; i < pScene->mNumMeshes; ++i) {

            aiMesh* inst = pScene->mMeshes[i];
            hashes[i] = GetMeshHash(inst);

            // Find an appropriate epsilon 
            // to compare position differences against
            float epsilon = ComputePositionEpsilon(inst);
            epsilon *= epsilon;

            for (int a = i-1; a >= 0; --a) {
                if (hashes[i] == hashes[a])
                {
                    aiMesh* orig = pScene->mMeshes[a];
                    if (!orig)
                        continue;

                    // check for hash collision .. we needn't check
                    // the vertex format, it *must* match due to the
                    // (brilliant) construction of the hash
                    if (orig->mNumBones       != inst->mNumBones      ||
                        orig->mNumFaces       != inst->mNumFaces      ||
                        orig->mNumVertices    != inst->mNumVertices   ||
                        orig->mMaterialIndex  != inst->mMaterialIndex ||
                        orig->mPrimitiveTypes != inst->mPrimitiveTypes)
                        continue;

                    // up to now the meshes are equal. Now compare vertex positions, normals,
                    // tangents and bitangents using this epsilon.
                    if (orig->HasPositions()) {
                        if(!CompareArrays(orig->mVertices,inst->mVertices,orig->mNumVertices,epsilon))
                            continue;
                    }
                    if (orig->HasNormals()) {
                        if(!CompareArrays(orig->mNormals,inst->mNormals,orig->mNumVertices,epsilon))
                            continue;
                    }
                    if (orig->HasTangentsAndBitangents()) {
                        if (!CompareArrays(orig->mTangents,inst->mTangents,orig->mNumVertices,epsilon) ||
                            !CompareArrays(orig->mBitangents,inst->mBitangents,orig->mNumVertices,epsilon))
                            continue;
                    }

                    // use a constant epsilon for colors and UV coordinates
                    static const float uvEpsilon = 10e-4f;
                    {
                        unsigned int j, end = orig->GetNumUVChannels();
                        for(j = 0; j < end; ++j) {
                            if (!orig->mTextureCoords[j]) {
                                continue;
                            }
                            if(!CompareArrays(orig->mTextureCoords[j],inst->mTextureCoords[j],orig->mNumVertices,uvEpsilon)) {
                                break;
                            }
                        }
                        if (j != end) {
                            continue;
                        }
                    }
                    {
                        unsigned int j, end = orig->GetNumColorChannels();
                        for(j = 0; j < end; ++j) {
                            if (!orig->mColors[j]) {
                                continue;
                            }
                            if(!CompareArrays(orig->mColors[j],inst->mColors[j],orig->mNumVertices,uvEpsilon)) {
                                break;
                            }
                        }
                        if (j != end) {
                            continue;
                        }
                    }

                    // These two checks are actually quite expensive and almost *never* required.
                    // Almost. That's why they're still here. But there's no reason to do them
                    // in speed-targeted imports.
                    if (!configSpeedFlag) {

                        // It seems to be strange, but we really need to check whether the
                        // bones are identical too. Although it's extremely unprobable
                        // that they're not if control reaches here, we need to deal
                        // with unprobable cases, too. It could still be that there are
                        // equal shapes which are deformed differently.
                        if (!CompareBones(orig,inst))
                            continue;

                        // For completeness ... compare even the index buffers for equality
                        // face order & winding order doesn't care. Input data is in verbose format.
                        std::unique_ptr<unsigned int[]> ftbl_orig(new unsigned int[orig->mNumVertices]);
                        std::unique_ptr<unsigned int[]> ftbl_inst(new unsigned int[orig->mNumVertices]);

                        for (unsigned int tt = 0; tt < orig->mNumFaces;++tt) {
                            aiFace& f = orig->mFaces[tt];
                            for (unsigned int nn = 0; nn < f.mNumIndices;++nn)
                                ftbl_orig[f.mIndices[nn]] = tt;

                            aiFace& f2 = inst->mFaces[tt];
                            for (unsigned int nn = 0; nn < f2.mNumIndices;++nn)
                                ftbl_inst[f2.mIndices[nn]] = tt;
                        }
                        if (0 != ::memcmp(ftbl_inst.get(),ftbl_orig.get(),orig->mNumVertices*sizeof(unsigned int)))
                            continue;
                    }

                    // We're still here. Or in other words: 'inst' is an instance of 'orig'.
                    // Place a marker in our list that we can easily update mesh indices.
                    remapping[i] = remapping[a];

                    // Delete the instanced mesh, we don't need it anymore
                    delete inst;
                    pScene->mMeshes[i] = NULL;
                    break;
                }
            }

            // If we didn't find a match for the current mesh: keep it
            if (pScene->mMeshes[i]) {
                remapping[i] = numMeshesOut++;
            }
        }
        ai_assert(0 != numMeshesOut);
        if (numMeshesOut != pScene->mNumMeshes) {

            // Collapse the meshes array by removing all NULL entries
            for (unsigned int real = 0, i = 0; real < numMeshesOut; ++i) {
                if (pScene->mMeshes[i])
                    pScene->mMeshes[real++] = pScene->mMeshes[i];
            }

            // And update the node graph with our nice lookup table
            UpdateMeshIndices(pScene->mRootNode,remapping.get());

            // write to log
            if (!DefaultLogger::isNullLogger()) {
                ASSIMP_LOG_INFO_F( "FindInstancesProcess finished. Found ", (pScene->mNumMeshes - numMeshesOut), " instances" );
            }
            pScene->mNumMeshes = numMeshesOut;
        } else {
            ASSIMP_LOG_DEBUG("FindInstancesProcess finished. No instanced meshes found");
        }
    }
}
