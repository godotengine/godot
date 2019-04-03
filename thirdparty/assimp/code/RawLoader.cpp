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

/** @file  RawLoader.cpp
 *  @brief Implementation of the RAW importer class
 */


#ifndef ASSIMP_BUILD_NO_RAW_IMPORTER

// internal headers
#include "RawLoader.h"
#include <assimp/ParsingUtils.h>
#include <assimp/fast_atof.h>
#include <memory>
#include <assimp/IOSystem.hpp>
#include <assimp/DefaultLogger.hpp>
#include <assimp/scene.h>
#include <assimp/importerdesc.h>

using namespace Assimp;

static const aiImporterDesc desc = {
    "Raw Importer",
    "",
    "",
    "",
    aiImporterFlags_SupportTextFlavour,
    0,
    0,
    0,
    0,
    "raw"
};

// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
RAWImporter::RAWImporter()
{}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
RAWImporter::~RAWImporter()
{}

// ------------------------------------------------------------------------------------------------
// Returns whether the class can handle the format of the given file.
bool RAWImporter::CanRead( const std::string& pFile, IOSystem* /*pIOHandler*/, bool /*checkSig*/) const
{
    return SimpleExtensionCheck(pFile,"raw");
}

// ------------------------------------------------------------------------------------------------
const aiImporterDesc* RAWImporter::GetInfo () const
{
    return &desc;
}

// ------------------------------------------------------------------------------------------------
// Imports the given file into the given scene structure.
void RAWImporter::InternReadFile( const std::string& pFile,
    aiScene* pScene, IOSystem* pIOHandler)
{
    std::unique_ptr<IOStream> file( pIOHandler->Open( pFile, "rb"));

    // Check whether we can read from the file
    if( file.get() == NULL) {
        throw DeadlyImportError( "Failed to open RAW file " + pFile + ".");
    }

    // allocate storage and copy the contents of the file to a memory buffer
    // (terminate it with zero)
    std::vector<char> mBuffer2;
    TextFileToBuffer(file.get(),mBuffer2);
    const char* buffer = &mBuffer2[0];

    // list of groups loaded from the file
    std::vector< GroupInformation > outGroups(1,GroupInformation("<default>"));
    std::vector< GroupInformation >::iterator curGroup = outGroups.begin();

    // now read all lines
    char line[4096];
    while (GetNextLine(buffer,line))
    {
        // if the line starts with a non-numeric identifier, it marks
        // the beginning of a new group
        const char* sz = line;SkipSpaces(&sz);
        if (IsLineEnd(*sz))continue;
        if (!IsNumeric(*sz))
        {
            const char* sz2 = sz;
            while (!IsSpaceOrNewLine(*sz2))++sz2;
            const unsigned int length = (unsigned int)(sz2-sz);

            // find an existing group with this name
            for (std::vector< GroupInformation >::iterator it = outGroups.begin(), end = outGroups.end();
                it != end;++it)
            {
                if (length == (*it).name.length() && !::strcmp(sz,(*it).name.c_str()))
                {
                    curGroup = it;sz2 = NULL;
                    break;
                }
            }
            if (sz2)
            {
                outGroups.push_back(GroupInformation(std::string(sz,length)));
                curGroup = outGroups.end()-1;
            }
        }
        else
        {
            // there can be maximally 12 floats plus an extra texture file name
            float data[12];
            unsigned int num;
            for (num = 0; num < 12;++num)
            {
                if(!SkipSpaces(&sz) || !IsNumeric(*sz))break;
                sz = fast_atoreal_move<float>(sz,data[num]);
            }
            if (num != 12 && num != 9)
            {
                ASSIMP_LOG_ERROR("A line may have either 9 or 12 floats and an optional texture");
                continue;
            }

            MeshInformation* output = NULL;

            const char* sz2 = sz;
            unsigned int length;
            if (!IsLineEnd(*sz))
            {
                while (!IsSpaceOrNewLine(*sz2))++sz2;
                length = (unsigned int)(sz2-sz);
            }
            else if (9 == num)
            {
                sz = "%default%";
                length = 9;
            }
            else
            {
                sz = "";
                length = 0;
            }

            // search in the list of meshes whether we have one with this texture
            for (auto &mesh : (*curGroup).meshes)
            {
                if (length == mesh.name.length() && (length ? !::strcmp(sz, mesh.name.c_str()) : true))
                {
                    output = &mesh;
                    break;
                }
            }
            // if we don't have the mesh, create it
            if (!output)
            {
                (*curGroup).meshes.push_back(MeshInformation(std::string(sz,length)));
                output = &((*curGroup).meshes.back());
            }
            if (12 == num)
            {
                aiColor4D v(data[0],data[1],data[2],1.0f);
                output->colors.push_back(v);
                output->colors.push_back(v);
                output->colors.push_back(v);

                output->vertices.push_back(aiVector3D(data[3],data[4],data[5]));
                output->vertices.push_back(aiVector3D(data[6],data[7],data[8]));
                output->vertices.push_back(aiVector3D(data[9],data[10],data[11]));
            }
            else
            {
                output->vertices.push_back(aiVector3D(data[0],data[1],data[2]));
                output->vertices.push_back(aiVector3D(data[3],data[4],data[5]));
                output->vertices.push_back(aiVector3D(data[6],data[7],data[8]));
            }
        }
    }

    pScene->mRootNode = new aiNode();
    pScene->mRootNode->mName.Set("<RawRoot>");

    // count the number of valid groups
    // (meshes can't be empty)
    for (auto & outGroup : outGroups)
    {
        if (!outGroup.meshes.empty())
        {
            ++pScene->mRootNode->mNumChildren;
            pScene->mNumMeshes += (unsigned int) outGroup.meshes.size();
        }
    }

    if (!pScene->mNumMeshes)
    {
        throw DeadlyImportError("RAW: No meshes loaded. The file seems to be corrupt or empty.");
    }

    pScene->mMeshes = new aiMesh*[pScene->mNumMeshes];
    aiNode** cc;
    if (1 == pScene->mRootNode->mNumChildren)
    {
        cc = &pScene->mRootNode;
        pScene->mRootNode->mNumChildren = 0;
    } else {
        cc = new aiNode*[pScene->mRootNode->mNumChildren];
        memset(cc, 0, sizeof(aiNode*) * pScene->mRootNode->mNumChildren);
        pScene->mRootNode->mChildren = cc;
    }

    pScene->mNumMaterials = pScene->mNumMeshes;
    aiMaterial** mats = pScene->mMaterials = new aiMaterial*[pScene->mNumMaterials];

    unsigned int meshIdx = 0;
    for (auto & outGroup : outGroups)
    {
        if (outGroup.meshes.empty())continue;

        aiNode* node;
        if (pScene->mRootNode->mNumChildren)
        {
            node = *cc = new aiNode();
            node->mParent = pScene->mRootNode;
        }
        else node = *cc;
        node->mName.Set(outGroup.name);

        // add all meshes
        node->mNumMeshes = (unsigned int) outGroup.meshes.size();
        unsigned int* pi = node->mMeshes = new unsigned int[ node->mNumMeshes ];
        for (std::vector< MeshInformation >::iterator it2 = outGroup.meshes.begin(),
            end2 = outGroup.meshes.end(); it2 != end2; ++it2)
        {
            ai_assert(!(*it2).vertices.empty());

            // allocate the mesh
            *pi++ = meshIdx;
            aiMesh* mesh = pScene->mMeshes[meshIdx] = new aiMesh();
            mesh->mMaterialIndex = meshIdx++;

            mesh->mPrimitiveTypes = aiPrimitiveType_TRIANGLE;

            // allocate storage for the vertex components and copy them
            mesh->mNumVertices = (unsigned int)(*it2).vertices.size();
            mesh->mVertices = new aiVector3D[ mesh->mNumVertices ];
            ::memcpy(mesh->mVertices,&(*it2).vertices[0],sizeof(aiVector3D)*mesh->mNumVertices);

            if ((*it2).colors.size())
            {
                ai_assert((*it2).colors.size() == mesh->mNumVertices);

                mesh->mColors[0] = new aiColor4D[ mesh->mNumVertices ];
                ::memcpy(mesh->mColors[0],&(*it2).colors[0],sizeof(aiColor4D)*mesh->mNumVertices);
            }

            // generate triangles
            ai_assert(0 == mesh->mNumVertices % 3);
            aiFace* fc = mesh->mFaces = new aiFace[ mesh->mNumFaces = mesh->mNumVertices/3 ];
            aiFace* const fcEnd = fc + mesh->mNumFaces;
            unsigned int n = 0;
            while (fc != fcEnd)
            {
                aiFace& f = *fc++;
                f.mIndices = new unsigned int[f.mNumIndices = 3];
                for (unsigned int m = 0; m < 3;++m)
                    f.mIndices[m] = n++;
            }

            // generate a material for the mesh
            aiMaterial* mat = new aiMaterial();

            aiColor4D clr(1.0f,1.0f,1.0f,1.0f);
            if ("%default%" == (*it2).name) // a gray default material
            {
                clr.r = clr.g = clr.b = 0.6f;
            }
            else if ((*it2).name.length() > 0) // a texture
            {
                aiString s;
                s.Set((*it2).name);
                mat->AddProperty(&s,AI_MATKEY_TEXTURE_DIFFUSE(0));
            }
            mat->AddProperty<aiColor4D>(&clr,1,AI_MATKEY_COLOR_DIFFUSE);
            *mats++ = mat;
        }
    }
}

#endif // !! ASSIMP_BUILD_NO_RAW_IMPORTER
