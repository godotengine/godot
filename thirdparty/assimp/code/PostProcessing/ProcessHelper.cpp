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

/// @file ProcessHelper.cpp
/** Implement shared utility functions for postprocessing steps */


#include "ProcessHelper.h"


#include <limits>

namespace Assimp {

// -------------------------------------------------------------------------------
void ConvertListToStrings(const std::string& in, std::list<std::string>& out)
{
    const char* s = in.c_str();
    while (*s) {
        SkipSpacesAndLineEnd(&s);
        if (*s == '\'') {
            const char* base = ++s;
            while (*s != '\'') {
                ++s;
                if (*s == '\0') {
                    ASSIMP_LOG_ERROR("ConvertListToString: String list is ill-formatted");
                    return;
                }
            }
            out.push_back(std::string(base,(size_t)(s-base)));
            ++s;
        }
        else {
            out.push_back(GetNextToken(s));
        }
    }
}

// -------------------------------------------------------------------------------
void FindAABBTransformed (const aiMesh* mesh, aiVector3D& min, aiVector3D& max,
    const aiMatrix4x4& m)
{
    min = aiVector3D ( ai_real( 10e10 ), ai_real( 10e10 ), ai_real( 10e10 ) );
    max = aiVector3D ( ai_real( -10e10 ), ai_real( -10e10 ), ai_real( -10e10 ) );
    for (unsigned int i = 0;i < mesh->mNumVertices;++i)
    {
        const aiVector3D v = m * mesh->mVertices[i];
        min = std::min(v,min);
        max = std::max(v,max);
    }
}

// -------------------------------------------------------------------------------
void FindMeshCenter (aiMesh* mesh, aiVector3D& out, aiVector3D& min, aiVector3D& max)
{
    ArrayBounds(mesh->mVertices,mesh->mNumVertices, min,max);
    out = min + (max-min)*(ai_real)0.5;
}

// -------------------------------------------------------------------------------
void FindSceneCenter (aiScene* scene, aiVector3D& out, aiVector3D& min, aiVector3D& max) {
    if ( NULL == scene ) {
        return;
    }

    if ( 0 == scene->mNumMeshes ) {
        return;
    }
    FindMeshCenter(scene->mMeshes[0], out, min, max);
    for (unsigned int i = 1; i < scene->mNumMeshes; ++i) {
        aiVector3D tout, tmin, tmax;
        FindMeshCenter(scene->mMeshes[i], tout, tmin, tmax);
        if (min[0] > tmin[0]) min[0] = tmin[0];
        if (min[1] > tmin[1]) min[1] = tmin[1];
        if (min[2] > tmin[2]) min[2] = tmin[2];
        if (max[0] < tmax[0]) max[0] = tmax[0];
        if (max[1] < tmax[1]) max[1] = tmax[1];
        if (max[2] < tmax[2]) max[2] = tmax[2];
    }
    out = min + (max-min)*(ai_real)0.5;
}


// -------------------------------------------------------------------------------
void FindMeshCenterTransformed (aiMesh* mesh, aiVector3D& out, aiVector3D& min,
    aiVector3D& max, const aiMatrix4x4& m)
{
    FindAABBTransformed(mesh,min,max,m);
    out = min + (max-min)*(ai_real)0.5;
}

// -------------------------------------------------------------------------------
void FindMeshCenter (aiMesh* mesh, aiVector3D& out)
{
    aiVector3D min,max;
    FindMeshCenter(mesh,out,min,max);
}

// -------------------------------------------------------------------------------
void FindMeshCenterTransformed (aiMesh* mesh, aiVector3D& out,
    const aiMatrix4x4& m)
{
    aiVector3D min,max;
    FindMeshCenterTransformed(mesh,out,min,max,m);
}

// -------------------------------------------------------------------------------
ai_real ComputePositionEpsilon(const aiMesh* pMesh)
{
    const ai_real epsilon = ai_real( 1e-4 );

    // calculate the position bounds so we have a reliable epsilon to check position differences against
    aiVector3D minVec, maxVec;
    ArrayBounds(pMesh->mVertices,pMesh->mNumVertices,minVec,maxVec);
    return (maxVec - minVec).Length() * epsilon;
}

// -------------------------------------------------------------------------------
ai_real ComputePositionEpsilon(const aiMesh* const* pMeshes, size_t num)
{
    ai_assert( NULL != pMeshes );

    const ai_real epsilon = ai_real( 1e-4 );

    // calculate the position bounds so we have a reliable epsilon to check position differences against
    aiVector3D minVec, maxVec, mi, ma;
    MinMaxChooser<aiVector3D>()(minVec,maxVec);

    for (size_t a = 0; a < num; ++a) {
        const aiMesh* pMesh = pMeshes[a];
        ArrayBounds(pMesh->mVertices,pMesh->mNumVertices,mi,ma);

        minVec = std::min(minVec,mi);
        maxVec = std::max(maxVec,ma);
    }
    return (maxVec - minVec).Length() * epsilon;
}


// -------------------------------------------------------------------------------
unsigned int GetMeshVFormatUnique(const aiMesh* pcMesh)
{
    ai_assert(NULL != pcMesh);

    // FIX: the hash may never be 0. Otherwise a comparison against
    // nullptr could be successful
    unsigned int iRet = 1;

    // normals
    if (pcMesh->HasNormals())iRet |= 0x2;
    // tangents and bitangents
    if (pcMesh->HasTangentsAndBitangents())iRet |= 0x4;

#ifdef BOOST_STATIC_ASSERT
    BOOST_STATIC_ASSERT(8 >= AI_MAX_NUMBER_OF_COLOR_SETS);
    BOOST_STATIC_ASSERT(8 >= AI_MAX_NUMBER_OF_TEXTURECOORDS);
#endif

    // texture coordinates
    unsigned int p = 0;
    while (pcMesh->HasTextureCoords(p))
    {
        iRet |= (0x100 << p);
        if (3 == pcMesh->mNumUVComponents[p])
            iRet |= (0x10000 << p);

        ++p;
    }
    // vertex colors
    p = 0;
    while (pcMesh->HasVertexColors(p))iRet |= (0x1000000 << p++);
    return iRet;
}

// -------------------------------------------------------------------------------
VertexWeightTable* ComputeVertexBoneWeightTable(const aiMesh* pMesh)
{
    if (!pMesh || !pMesh->mNumVertices || !pMesh->mNumBones) {
        return NULL;
    }

    VertexWeightTable* avPerVertexWeights = new VertexWeightTable[pMesh->mNumVertices];
    for (unsigned int i = 0; i < pMesh->mNumBones;++i)  {

        aiBone* bone = pMesh->mBones[i];
        for (unsigned int a = 0; a < bone->mNumWeights;++a) {
            const aiVertexWeight& weight = bone->mWeights[a];
            avPerVertexWeights[weight.mVertexId].push_back( std::pair<unsigned int,float>(i,weight.mWeight) );
        }
    }
    return avPerVertexWeights;
}


// -------------------------------------------------------------------------------
const char* TextureTypeToString(aiTextureType in)
{
    switch (in)
    {
    case aiTextureType_NONE:
        return "n/a";
    case aiTextureType_DIFFUSE:
        return "Diffuse";
    case aiTextureType_SPECULAR:
        return "Specular";
    case aiTextureType_AMBIENT:
        return "Ambient";
    case aiTextureType_EMISSIVE:
        return "Emissive";
    case aiTextureType_OPACITY:
        return "Opacity";
    case aiTextureType_NORMALS:
        return "Normals";
    case aiTextureType_HEIGHT:
        return "Height";
    case aiTextureType_SHININESS:
        return "Shininess";
    case aiTextureType_DISPLACEMENT:
        return "Displacement";
    case aiTextureType_LIGHTMAP:
        return "Lightmap";
    case aiTextureType_REFLECTION:
        return "Reflection";
    case aiTextureType_UNKNOWN:
        return "Unknown";
    default:
        break;
    }

    ai_assert(false);
    return  "BUG";
}

// -------------------------------------------------------------------------------
const char* MappingTypeToString(aiTextureMapping in)
{
    switch (in)
    {
    case aiTextureMapping_UV:
        return "UV";
    case aiTextureMapping_BOX:
        return "Box";
    case aiTextureMapping_SPHERE:
        return "Sphere";
    case aiTextureMapping_CYLINDER:
        return "Cylinder";
    case aiTextureMapping_PLANE:
        return "Plane";
    case aiTextureMapping_OTHER:
        return "Other";
    default:
        break;
    }

    ai_assert(false);
    return  "BUG";
}


// -------------------------------------------------------------------------------
aiMesh* MakeSubmesh(const aiMesh *pMesh, const std::vector<unsigned int> &subMeshFaces, unsigned int subFlags)
{
    aiMesh *oMesh = new aiMesh();
    std::vector<unsigned int> vMap(pMesh->mNumVertices,UINT_MAX);

    size_t numSubVerts = 0;
    size_t numSubFaces = subMeshFaces.size();

    for(unsigned int i=0;i<numSubFaces;i++) {
        const aiFace &f = pMesh->mFaces[subMeshFaces[i]];

        for(unsigned int j=0;j<f.mNumIndices;j++)   {
            if(vMap[f.mIndices[j]]==UINT_MAX)   {
                vMap[f.mIndices[j]] = static_cast<unsigned int>(numSubVerts++);
            }
        }
    }

    oMesh->mName = pMesh->mName;

    oMesh->mMaterialIndex = pMesh->mMaterialIndex;
    oMesh->mPrimitiveTypes = pMesh->mPrimitiveTypes;

    // create all the arrays for this mesh if the old mesh contained them

    oMesh->mNumFaces = static_cast<unsigned int>(subMeshFaces.size());
    oMesh->mNumVertices = static_cast<unsigned int>(numSubVerts);
    oMesh->mVertices = new aiVector3D[numSubVerts];
    if( pMesh->HasNormals() ) {
        oMesh->mNormals = new aiVector3D[numSubVerts];
    }

    if( pMesh->HasTangentsAndBitangents() ) {
        oMesh->mTangents = new aiVector3D[numSubVerts];
        oMesh->mBitangents = new aiVector3D[numSubVerts];
    }

    for( size_t a = 0;  pMesh->HasTextureCoords(static_cast<unsigned int>(a)) ; ++a ) {
        oMesh->mTextureCoords[a] = new aiVector3D[numSubVerts];
        oMesh->mNumUVComponents[a] = pMesh->mNumUVComponents[a];
    }

    for( size_t a = 0; pMesh->HasVertexColors( static_cast<unsigned int>(a)); ++a )    {
        oMesh->mColors[a] = new aiColor4D[numSubVerts];
    }

    // and copy over the data, generating faces with linear indices along the way
    oMesh->mFaces = new aiFace[numSubFaces];

    for(unsigned int a = 0; a < numSubFaces; ++a )  {

        const aiFace& srcFace = pMesh->mFaces[subMeshFaces[a]];
        aiFace& dstFace = oMesh->mFaces[a];
        dstFace.mNumIndices = srcFace.mNumIndices;
        dstFace.mIndices = new unsigned int[dstFace.mNumIndices];

        // accumulate linearly all the vertices of the source face
        for( size_t b = 0; b < dstFace.mNumIndices; ++b )   {
            dstFace.mIndices[b] = vMap[srcFace.mIndices[b]];
        }
    }

    for(unsigned int srcIndex = 0; srcIndex < pMesh->mNumVertices; ++srcIndex ) {
        unsigned int nvi = vMap[srcIndex];
        if(nvi==UINT_MAX) {
            continue;
        }

        oMesh->mVertices[nvi] = pMesh->mVertices[srcIndex];
        if( pMesh->HasNormals() ) {
            oMesh->mNormals[nvi] = pMesh->mNormals[srcIndex];
        }

        if( pMesh->HasTangentsAndBitangents() ) {
            oMesh->mTangents[nvi] = pMesh->mTangents[srcIndex];
            oMesh->mBitangents[nvi] = pMesh->mBitangents[srcIndex];
        }
        for( size_t c = 0, cc = pMesh->GetNumUVChannels(); c < cc; ++c )    {
                oMesh->mTextureCoords[c][nvi] = pMesh->mTextureCoords[c][srcIndex];
        }
        for( size_t c = 0, cc = pMesh->GetNumColorChannels(); c < cc; ++c ) {
            oMesh->mColors[c][nvi] = pMesh->mColors[c][srcIndex];
        }
    }

    if(~subFlags&AI_SUBMESH_FLAGS_SANS_BONES)   {
        std::vector<unsigned int> subBones(pMesh->mNumBones,0);

        for(unsigned int a=0;a<pMesh->mNumBones;++a)    {
            const aiBone* bone = pMesh->mBones[a];

            for(unsigned int b=0;b<bone->mNumWeights;b++)   {
                unsigned int v = vMap[bone->mWeights[b].mVertexId];

                if(v!=UINT_MAX) {
                    subBones[a]++;
                }
            }
        }

        for(unsigned int a=0;a<pMesh->mNumBones;++a)    {
            if(subBones[a]>0) {
                oMesh->mNumBones++;
            }
        }

        if(oMesh->mNumBones) {
            oMesh->mBones = new aiBone*[oMesh->mNumBones]();
            unsigned int nbParanoia = oMesh->mNumBones;

            oMesh->mNumBones = 0; //rewind

            for(unsigned int a=0;a<pMesh->mNumBones;++a)    {
                if(subBones[a]==0) {
                    continue;
                }
                aiBone *newBone = new aiBone;
                oMesh->mBones[oMesh->mNumBones++] = newBone;

                const aiBone* bone = pMesh->mBones[a];

                newBone->mName = bone->mName;
                newBone->mOffsetMatrix = bone->mOffsetMatrix;
                newBone->mWeights = new aiVertexWeight[subBones[a]];

                for(unsigned int b=0;b<bone->mNumWeights;b++)   {
                    const unsigned int v = vMap[bone->mWeights[b].mVertexId];

                    if(v!=UINT_MAX) {
                        aiVertexWeight w(v,bone->mWeights[b].mWeight);
                        newBone->mWeights[newBone->mNumWeights++] = w;
                    }
                }
            }

            ai_assert(nbParanoia==oMesh->mNumBones);
            (void)nbParanoia; // remove compiler warning on release build
        }
    }

    return oMesh;
}

} // namespace Assimp
