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
/** @file RemoveRedundantMaterials.cpp
 *  @brief Implementation of the "RemoveRedundantMaterials" post processing step
*/

// internal headers

#include "RemoveRedundantMaterials.h"
#include <assimp/ParsingUtils.h>
#include "ProcessHelper.h"
#include "Material/MaterialSystem.h"
#include <stdio.h>

using namespace Assimp;

// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
RemoveRedundantMatsProcess::RemoveRedundantMatsProcess()
: mConfigFixedMaterials() {
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
RemoveRedundantMatsProcess::~RemoveRedundantMatsProcess()
{
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool RemoveRedundantMatsProcess::IsActive( unsigned int pFlags) const
{
    return (pFlags & aiProcess_RemoveRedundantMaterials) != 0;
}

// ------------------------------------------------------------------------------------------------
// Setup import properties
void RemoveRedundantMatsProcess::SetupProperties(const Importer* pImp)
{
    // Get value of AI_CONFIG_PP_RRM_EXCLUDE_LIST
    mConfigFixedMaterials = pImp->GetPropertyString(AI_CONFIG_PP_RRM_EXCLUDE_LIST,"");
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void RemoveRedundantMatsProcess::Execute( aiScene* pScene)
{
    ASSIMP_LOG_DEBUG("RemoveRedundantMatsProcess begin");

    unsigned int redundantRemoved = 0, unreferencedRemoved = 0;
    if (pScene->mNumMaterials)
    {
        // Find out which materials are referenced by meshes
        std::vector<bool> abReferenced(pScene->mNumMaterials,false);
        for (unsigned int i = 0;i < pScene->mNumMeshes;++i)
            abReferenced[pScene->mMeshes[i]->mMaterialIndex] = true;

        // If a list of materials to be excluded was given, match the list with
        // our imported materials and 'salt' all positive matches to ensure that
        // we get unique hashes later.
        if (mConfigFixedMaterials.length()) {

            std::list<std::string> strings;
            ConvertListToStrings(mConfigFixedMaterials,strings);

            for (unsigned int i = 0; i < pScene->mNumMaterials;++i) {
                aiMaterial* mat = pScene->mMaterials[i];

                aiString name;
                mat->Get(AI_MATKEY_NAME,name);

                if (name.length) {
                    std::list<std::string>::const_iterator it = std::find(strings.begin(), strings.end(), name.data);
                    if (it != strings.end()) {

                        // Our brilliant 'salt': A single material property with ~ as first
                        // character to mark it as internal and temporary.
                        const int dummy = 1;
                        ((aiMaterial*)mat)->AddProperty(&dummy,1,"~RRM.UniqueMaterial",0,0);

                        // Keep this material even if no mesh references it
                        abReferenced[i] = true;
                        ASSIMP_LOG_DEBUG_F( "Found positive match in exclusion list: \'", name.data, "\'");
                    }
                }
            }
        }

        // TODO: re-implement this algorithm to work in-place
        unsigned int *aiMappingTable = new unsigned int[pScene->mNumMaterials];
        for ( unsigned int i=0; i<pScene->mNumMaterials; i++ ) {
            aiMappingTable[ i ] = 0;
        }
        unsigned int iNewNum = 0;

        // Iterate through all materials and calculate a hash for them
        // store all hashes in a list and so a quick search whether
        // we do already have a specific hash. This allows us to
        // determine which materials are identical.
        uint32_t *aiHashes = new uint32_t[ pScene->mNumMaterials ];;
        for (unsigned int i = 0; i < pScene->mNumMaterials;++i)
        {
            // No mesh is referencing this material, remove it.
            if (!abReferenced[i]) {
                ++unreferencedRemoved;
                delete pScene->mMaterials[i];
                pScene->mMaterials[i] = nullptr;
                continue;
            }

            // Check all previously mapped materials for a matching hash.
            // On a match we can delete this material and just make it ref to the same index.
            uint32_t me = aiHashes[i] = ComputeMaterialHash(pScene->mMaterials[i]);
            for (unsigned int a = 0; a < i;++a)
            {
                if (abReferenced[a] && me == aiHashes[a]) {
                    ++redundantRemoved;
                    me = 0;
                    aiMappingTable[i] = aiMappingTable[a];
                    delete pScene->mMaterials[i];
                    pScene->mMaterials[i] = nullptr;
                    break;
                }
            }
            // This is a new material that is referenced, add to the map.
            if (me) {
                aiMappingTable[i] = iNewNum++;
            }
        }
        // If the new material count differs from the original,
        // we need to rebuild the material list and remap mesh material indexes.
        if (iNewNum != pScene->mNumMaterials) {
            ai_assert(iNewNum > 0);
            aiMaterial** ppcMaterials = new aiMaterial*[iNewNum];
            ::memset(ppcMaterials,0,sizeof(void*)*iNewNum);
            for (unsigned int p = 0; p < pScene->mNumMaterials;++p)
            {
                // if the material is not referenced ... remove it
                if (!abReferenced[p]) {
                    continue;
                }

                // generate new names for modified materials that had no names
                const unsigned int idx = aiMappingTable[p];
                if (ppcMaterials[idx]) {
                    aiString sz;
                    if( ppcMaterials[idx]->Get(AI_MATKEY_NAME, sz) != AI_SUCCESS ) {
                        sz.length = ::ai_snprintf(sz.data,MAXLEN,"JoinedMaterial_#%u",p);
                        ((aiMaterial*)ppcMaterials[idx])->AddProperty(&sz,AI_MATKEY_NAME);
                    }
                } else {
                    ppcMaterials[idx] = pScene->mMaterials[p];
                }
            }
            // update all material indices
            for (unsigned int p = 0; p < pScene->mNumMeshes;++p) {
                aiMesh* mesh = pScene->mMeshes[p];
                ai_assert( NULL!=mesh );
                mesh->mMaterialIndex = aiMappingTable[mesh->mMaterialIndex];
            }
            // delete the old material list
            delete[] pScene->mMaterials;
            pScene->mMaterials = ppcMaterials;
            pScene->mNumMaterials = iNewNum;
        }
        // delete temporary storage
        delete[] aiHashes;
        delete[] aiMappingTable;
    }
    if (redundantRemoved == 0 && unreferencedRemoved == 0)
    {
        ASSIMP_LOG_DEBUG("RemoveRedundantMatsProcess finished ");
    }
    else
    {
        ASSIMP_LOG_INFO_F("RemoveRedundantMatsProcess finished. Removed ", redundantRemoved, " redundant and ", 
            unreferencedRemoved, " unused materials.");
    }
}
