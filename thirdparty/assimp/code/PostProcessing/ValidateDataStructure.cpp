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

/** @file  ValidateDataStructure.cpp
 *  @brief Implementation of the post processing step to validate
 *    the data structure returned by Assimp.
 */

// internal headers
#include "ValidateDataStructure.h"
#include <assimp/BaseImporter.h>
#include <assimp/fast_atof.h>
#include "ProcessHelper.h"
#include <memory>

// CRT headers
#include <stdarg.h>

using namespace Assimp;

// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
ValidateDSProcess::ValidateDSProcess() :
    mScene()
{}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
ValidateDSProcess::~ValidateDSProcess()
{}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool ValidateDSProcess::IsActive( unsigned int pFlags) const
{
    return (pFlags & aiProcess_ValidateDataStructure) != 0;
}
// ------------------------------------------------------------------------------------------------
AI_WONT_RETURN void ValidateDSProcess::ReportError(const char* msg,...)
{
    ai_assert(NULL != msg);

    va_list args;
    va_start(args,msg);

    char szBuffer[3000];
    const int iLen = vsprintf(szBuffer,msg,args);
    ai_assert(iLen > 0);

    va_end(args);

    throw DeadlyImportError("Validation failed: " + std::string(szBuffer,iLen));
}
// ------------------------------------------------------------------------------------------------
void ValidateDSProcess::ReportWarning(const char* msg,...)
{
    ai_assert(NULL != msg);

    va_list args;
    va_start(args,msg);

    char szBuffer[3000];
    const int iLen = vsprintf(szBuffer,msg,args);
    ai_assert(iLen > 0);

    va_end(args);
    ASSIMP_LOG_WARN("Validation warning: " + std::string(szBuffer,iLen));
}

// ------------------------------------------------------------------------------------------------
inline
int HasNameMatch(const aiString& in, aiNode* node) {
    int result = (node->mName == in ? 1 : 0 );
    for (unsigned int i = 0; i < node->mNumChildren;++i)    {
        result += HasNameMatch(in,node->mChildren[i]);
    }
    return result;
}

// ------------------------------------------------------------------------------------------------
template <typename T>
inline
void ValidateDSProcess::DoValidation(T** parray, unsigned int size, const char* firstName, const char* secondName) {
    // validate all entries
    if (size)
    {
        if (!parray)
        {
            ReportError("aiScene::%s is NULL (aiScene::%s is %i)",
                firstName, secondName, size);
        }
        for (unsigned int i = 0; i < size;++i)
        {
            if (!parray[i])
            {
                ReportError("aiScene::%s[%i] is NULL (aiScene::%s is %i)",
                    firstName,i,secondName,size);
            }
            Validate(parray[i]);
        }
    }
}

// ------------------------------------------------------------------------------------------------
template <typename T>
inline void ValidateDSProcess::DoValidationEx(T** parray, unsigned int size,
    const char* firstName, const char* secondName)
{
    // validate all entries
    if (size)
    {
        if (!parray)    {
            ReportError("aiScene::%s is NULL (aiScene::%s is %i)",
                firstName, secondName, size);
        }
        for (unsigned int i = 0; i < size;++i)
        {
            if (!parray[i])
            {
                ReportError("aiScene::%s[%u] is NULL (aiScene::%s is %u)",
                    firstName,i,secondName,size);
            }
            Validate(parray[i]);

            // check whether there are duplicate names
            for (unsigned int a = i+1; a < size;++a)
            {
                if (parray[i]->mName == parray[a]->mName)
                {
                	ReportError("aiScene::%s[%u] has the same name as "
                        "aiScene::%s[%u]",firstName, i,secondName, a);
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
template <typename T>
inline
void ValidateDSProcess::DoValidationWithNameCheck(T** array, unsigned int size, const char* firstName,
        const char* secondName) {
    // validate all entries
    DoValidationEx(array,size,firstName,secondName);

    for (unsigned int i = 0; i < size;++i) {
        int res = HasNameMatch(array[i]->mName,mScene->mRootNode);
        if (0 == res)   {
            const std::string name = static_cast<char*>(array[i]->mName.data);
            ReportError("aiScene::%s[%i] has no corresponding node in the scene graph (%s)",
                firstName,i, name.c_str());
        } else if (1 != res)  {
            const std::string name = static_cast<char*>(array[i]->mName.data);
            ReportError("aiScene::%s[%i]: there are more than one nodes with %s as name",
                firstName,i, name.c_str());
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void ValidateDSProcess::Execute( aiScene* pScene) {
    mScene = pScene;
    ASSIMP_LOG_DEBUG("ValidateDataStructureProcess begin");

    // validate the node graph of the scene
    Validate(pScene->mRootNode);

    // validate all meshes
    if (pScene->mNumMeshes) {
        DoValidation(pScene->mMeshes,pScene->mNumMeshes,"mMeshes","mNumMeshes");
    }
    else if (!(mScene->mFlags & AI_SCENE_FLAGS_INCOMPLETE)) {
        ReportError("aiScene::mNumMeshes is 0. At least one mesh must be there");
    }
    else if (pScene->mMeshes)   {
        ReportError("aiScene::mMeshes is non-null although there are no meshes");
    }

    // validate all animations
    if (pScene->mNumAnimations) {
        DoValidation(pScene->mAnimations,pScene->mNumAnimations,
            "mAnimations","mNumAnimations");
    }
    else if (pScene->mAnimations)   {
        ReportError("aiScene::mAnimations is non-null although there are no animations");
    }

    // validate all cameras
    if (pScene->mNumCameras) {
        DoValidationWithNameCheck(pScene->mCameras,pScene->mNumCameras,
            "mCameras","mNumCameras");
    }
    else if (pScene->mCameras)  {
        ReportError("aiScene::mCameras is non-null although there are no cameras");
    }

    // validate all lights
    if (pScene->mNumLights) {
        DoValidationWithNameCheck(pScene->mLights,pScene->mNumLights,
            "mLights","mNumLights");
    }
    else if (pScene->mLights)   {
        ReportError("aiScene::mLights is non-null although there are no lights");
    }

    // validate all textures
    if (pScene->mNumTextures) {
        DoValidation(pScene->mTextures,pScene->mNumTextures,
            "mTextures","mNumTextures");
    }
    else if (pScene->mTextures) {
        ReportError("aiScene::mTextures is non-null although there are no textures");
    }

    // validate all materials
    if (pScene->mNumMaterials) {
        DoValidation(pScene->mMaterials,pScene->mNumMaterials,"mMaterials","mNumMaterials");
    }
#if 0
    // NOTE: ScenePreprocessor generates a default material if none is there
    else if (!(mScene->mFlags & AI_SCENE_FLAGS_INCOMPLETE)) {
        ReportError("aiScene::mNumMaterials is 0. At least one material must be there");
    }
#endif
    else if (pScene->mMaterials)    {
        ReportError("aiScene::mMaterials is non-null although there are no materials");
    }

//  if (!has)ReportError("The aiScene data structure is empty");
    ASSIMP_LOG_DEBUG("ValidateDataStructureProcess end");
}

// ------------------------------------------------------------------------------------------------
void ValidateDSProcess::Validate( const aiLight* pLight)
{
    if (pLight->mType == aiLightSource_UNDEFINED)
        ReportWarning("aiLight::mType is aiLightSource_UNDEFINED");

    if (!pLight->mAttenuationConstant &&
        !pLight->mAttenuationLinear   &&
        !pLight->mAttenuationQuadratic) {
        ReportWarning("aiLight::mAttenuationXXX - all are zero");
    }

    if (pLight->mAngleInnerCone > pLight->mAngleOuterCone)
        ReportError("aiLight::mAngleInnerCone is larger than aiLight::mAngleOuterCone");

    if (pLight->mColorDiffuse.IsBlack() && pLight->mColorAmbient.IsBlack()
        && pLight->mColorSpecular.IsBlack())
    {
        ReportWarning("aiLight::mColorXXX - all are black and won't have any influence");
    }
}

// ------------------------------------------------------------------------------------------------
void ValidateDSProcess::Validate( const aiCamera* pCamera)
{
    if (pCamera->mClipPlaneFar <= pCamera->mClipPlaneNear)
        ReportError("aiCamera::mClipPlaneFar must be >= aiCamera::mClipPlaneNear");

    // FIX: there are many 3ds files with invalid FOVs. No reason to
    // reject them at all ... a warning is appropriate.
    if (!pCamera->mHorizontalFOV || pCamera->mHorizontalFOV >= (float)AI_MATH_PI)
        ReportWarning("%f is not a valid value for aiCamera::mHorizontalFOV",pCamera->mHorizontalFOV);
}

// ------------------------------------------------------------------------------------------------
void ValidateDSProcess::Validate( const aiMesh* pMesh)
{
    // validate the material index of the mesh
    if (mScene->mNumMaterials && pMesh->mMaterialIndex >= mScene->mNumMaterials)
    {
        ReportError("aiMesh::mMaterialIndex is invalid (value: %i maximum: %i)",
            pMesh->mMaterialIndex,mScene->mNumMaterials-1);
    }

    Validate(&pMesh->mName);

    for (unsigned int i = 0; i < pMesh->mNumFaces; ++i)
    {
        aiFace& face = pMesh->mFaces[i];

        if (pMesh->mPrimitiveTypes)
        {
            switch (face.mNumIndices)
            {
            case 0:
                ReportError("aiMesh::mFaces[%i].mNumIndices is 0",i);
                break;
            case 1:
                if (0 == (pMesh->mPrimitiveTypes & aiPrimitiveType_POINT))
                {
                    ReportError("aiMesh::mFaces[%i] is a POINT but aiMesh::mPrimitiveTypes "
                        "does not report the POINT flag",i);
                }
                break;
            case 2:
                if (0 == (pMesh->mPrimitiveTypes & aiPrimitiveType_LINE))
                {
                    ReportError("aiMesh::mFaces[%i] is a LINE but aiMesh::mPrimitiveTypes "
                        "does not report the LINE flag",i);
                }
                break;
            case 3:
                if (0 == (pMesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE))
                {
                    ReportError("aiMesh::mFaces[%i] is a TRIANGLE but aiMesh::mPrimitiveTypes "
                        "does not report the TRIANGLE flag",i);
                }
                break;
            default:
                if (0 == (pMesh->mPrimitiveTypes & aiPrimitiveType_POLYGON))
                {
                    this->ReportError("aiMesh::mFaces[%i] is a POLYGON but aiMesh::mPrimitiveTypes "
                        "does not report the POLYGON flag",i);
                }
                break;
            };
        }

        if (!face.mIndices)
            ReportError("aiMesh::mFaces[%i].mIndices is NULL",i);
    }

    // positions must always be there ...
    if (!pMesh->mNumVertices || (!pMesh->mVertices && !mScene->mFlags)) {
        ReportError("The mesh %s contains no vertices", pMesh->mName.C_Str());
    }

    if (pMesh->mNumVertices > AI_MAX_VERTICES) {
        ReportError("Mesh has too many vertices: %u, but the limit is %u",pMesh->mNumVertices,AI_MAX_VERTICES);
    }
    if (pMesh->mNumFaces > AI_MAX_FACES) {
        ReportError("Mesh has too many faces: %u, but the limit is %u",pMesh->mNumFaces,AI_MAX_FACES);
    }

    // if tangents are there there must also be bitangent vectors ...
    if ((pMesh->mTangents != NULL) != (pMesh->mBitangents != NULL)) {
        ReportError("If there are tangents, bitangent vectors must be present as well");
    }

    // faces, too
    if (!pMesh->mNumFaces || (!pMesh->mFaces && !mScene->mFlags))   {
        ReportError("Mesh %s contains no faces", pMesh->mName.C_Str());
    }

    // now check whether the face indexing layout is correct:
    // unique vertices, pseudo-indexed.
    std::vector<bool> abRefList;
    abRefList.resize(pMesh->mNumVertices,false);
    for (unsigned int i = 0; i < pMesh->mNumFaces;++i)
    {
        aiFace& face = pMesh->mFaces[i];
        if (face.mNumIndices > AI_MAX_FACE_INDICES) {
            ReportError("Face %u has too many faces: %u, but the limit is %u",i,face.mNumIndices,AI_MAX_FACE_INDICES);
        }

        for (unsigned int a = 0; a < face.mNumIndices;++a)
        {
            if (face.mIndices[a] >= pMesh->mNumVertices)    {
                ReportError("aiMesh::mFaces[%i]::mIndices[%i] is out of range",i,a);
            }
            // the MSB flag is temporarily used by the extra verbose
            // mode to tell us that the JoinVerticesProcess might have
            // been executed already.
			/*if ( !(this->mScene->mFlags & AI_SCENE_FLAGS_NON_VERBOSE_FORMAT ) && !(this->mScene->mFlags & AI_SCENE_FLAGS_ALLOW_SHARED) &&
				abRefList[face.mIndices[a]])
            {
                ReportError("aiMesh::mVertices[%i] is referenced twice - second "
                    "time by aiMesh::mFaces[%i]::mIndices[%i]",face.mIndices[a],i,a);
            }*/
            abRefList[face.mIndices[a]] = true;
        }
    }

    // check whether there are vertices that aren't referenced by a face
    bool b = false;
    for (unsigned int i = 0; i < pMesh->mNumVertices;++i)   {
        if (!abRefList[i])b = true;
    }
    abRefList.clear();
    if (b) {
    	ReportWarning("There are unreferenced vertices");
    }

    // texture channel 2 may not be set if channel 1 is zero ...
    {
        unsigned int i = 0;
        for (;i < AI_MAX_NUMBER_OF_TEXTURECOORDS;++i)
        {
            if (!pMesh->HasTextureCoords(i))break;
        }
        for (;i < AI_MAX_NUMBER_OF_TEXTURECOORDS;++i)
            if (pMesh->HasTextureCoords(i))
            {
                ReportError("Texture coordinate channel %i exists "
                    "although the previous channel was NULL.",i);
            }
    }
    // the same for the vertex colors
    {
        unsigned int i = 0;
        for (;i < AI_MAX_NUMBER_OF_COLOR_SETS;++i)
        {
            if (!pMesh->HasVertexColors(i))break;
        }
        for (;i < AI_MAX_NUMBER_OF_COLOR_SETS;++i)
            if (pMesh->HasVertexColors(i))
            {
                ReportError("Vertex color channel %i is exists "
                    "although the previous channel was NULL.",i);
            }
    }


    // now validate all bones
    if (pMesh->mNumBones)
    {
        if (!pMesh->mBones)
        {
            ReportError("aiMesh::mBones is NULL (aiMesh::mNumBones is %i)",
                pMesh->mNumBones);
        }
        std::unique_ptr<float[]> afSum(nullptr);
        if (pMesh->mNumVertices)
        {
            afSum.reset(new float[pMesh->mNumVertices]);
            for (unsigned int i = 0; i < pMesh->mNumVertices;++i)
                afSum[i] = 0.0f;
        }

        // check whether there are duplicate bone names
        for (unsigned int i = 0; i < pMesh->mNumBones;++i)
        {
            const aiBone* bone = pMesh->mBones[i];
            if (bone->mNumWeights > AI_MAX_BONE_WEIGHTS) {
                ReportError("Bone %u has too many weights: %u, but the limit is %u",i,bone->mNumWeights,AI_MAX_BONE_WEIGHTS);
            }

            if (!pMesh->mBones[i])
            {
                ReportError("aiMesh::mBones[%i] is NULL (aiMesh::mNumBones is %i)",
                    i,pMesh->mNumBones);
            }
            Validate(pMesh,pMesh->mBones[i],afSum.get());

            for (unsigned int a = i+1; a < pMesh->mNumBones;++a)
            {
                if (pMesh->mBones[i]->mName == pMesh->mBones[a]->mName)
                {
                    const char *name = "unknown";
                    if (nullptr != pMesh->mBones[ i ]->mName.C_Str()) {
                        name = pMesh->mBones[ i ]->mName.C_Str();
                    }
                    ReportError("aiMesh::mBones[%i], name = \"%s\" has the same name as "
                        "aiMesh::mBones[%i]", i, name, a );
                }
            }
        }
        // check whether all bone weights for a vertex sum to 1.0 ...
        for (unsigned int i = 0; i < pMesh->mNumVertices;++i)
        {
            if (afSum[i] && (afSum[i] <= 0.94 || afSum[i] >= 1.05)) {
                ReportWarning("aiMesh::mVertices[%i]: bone weight sum != 1.0 (sum is %f)",i,afSum[i]);
            }
        }
    }
    else if (pMesh->mBones)
    {
        ReportError("aiMesh::mBones is non-null although there are no bones");
    }
}

// ------------------------------------------------------------------------------------------------
void ValidateDSProcess::Validate( const aiMesh* pMesh, const aiBone* pBone,float* afSum) {
    this->Validate(&pBone->mName);

    if (!pBone->mNumWeights)    {
        //ReportError("aiBone::mNumWeights is zero");
    }

    // check whether all vertices affected by this bone are valid
    for (unsigned int i = 0; i < pBone->mNumWeights;++i)
    {
        if (pBone->mWeights[i].mVertexId >= pMesh->mNumVertices)    {
            ReportError("aiBone::mWeights[%i].mVertexId is out of range",i);
        }
        else if (!pBone->mWeights[i].mWeight || pBone->mWeights[i].mWeight > 1.0f)  {
            ReportWarning("aiBone::mWeights[%i].mWeight has an invalid value",i);
        }
        afSum[pBone->mWeights[i].mVertexId] += pBone->mWeights[i].mWeight;
    }
}

// ------------------------------------------------------------------------------------------------
void ValidateDSProcess::Validate( const aiAnimation* pAnimation)
{
    Validate(&pAnimation->mName);

    // validate all materials
    if (pAnimation->mNumChannels)
    {
        if (!pAnimation->mChannels) {
            ReportError("aiAnimation::mChannels is NULL (aiAnimation::mNumChannels is %i)",
                pAnimation->mNumChannels);
        }
        for (unsigned int i = 0; i < pAnimation->mNumChannels;++i)
        {
            if (!pAnimation->mChannels[i])
            {
                ReportError("aiAnimation::mChannels[%i] is NULL (aiAnimation::mNumChannels is %i)",
                    i, pAnimation->mNumChannels);
            }
            Validate(pAnimation, pAnimation->mChannels[i]);
        }
    }
    else {
    	ReportError("aiAnimation::mNumChannels is 0. At least one node animation channel must be there.");
    }
}

// ------------------------------------------------------------------------------------------------
void ValidateDSProcess::SearchForInvalidTextures(const aiMaterial* pMaterial,
    aiTextureType type)
{
    const char* szType = TextureTypeToString(type);

    // ****************************************************************************
    // Search all keys of the material ...
    // textures must be specified with ascending indices
    // (e.g. diffuse #2 may not be specified if diffuse #1 is not there ...)
    // ****************************************************************************

    int iNumIndices = 0;
    int iIndex = -1;
    for (unsigned int i = 0; i < pMaterial->mNumProperties;++i) {
        aiMaterialProperty* prop = pMaterial->mProperties[ i ];
        ai_assert(nullptr != prop);
        if ( !::strcmp(prop->mKey.data,"$tex.file") && prop->mSemantic == static_cast<unsigned int>(type))  {
            iIndex = std::max(iIndex, (int) prop->mIndex);
            ++iNumIndices;

            if (aiPTI_String != prop->mType) {
                ReportError("Material property %s is expected to be a string", prop->mKey.data);
            }
        }
    }
    if (iIndex +1 != iNumIndices)   {
        ReportError("%s #%i is set, but there are only %i %s textures",
            szType,iIndex,iNumIndices,szType);
    }
    if (!iNumIndices)return;
    std::vector<aiTextureMapping> mappings(iNumIndices);

    // Now check whether all UV indices are valid ...
    bool bNoSpecified = true;
    for (unsigned int i = 0; i < pMaterial->mNumProperties;++i)
    {
        aiMaterialProperty* prop = pMaterial->mProperties[i];
        if (prop->mSemantic != type)continue;

        if ((int)prop->mIndex >= iNumIndices)
        {
            ReportError("Found texture property with index %i, although there "
                "are only %i textures of type %s",
                prop->mIndex, iNumIndices, szType);
        }

        if (!::strcmp(prop->mKey.data,"$tex.mapping"))  {
            if (aiPTI_Integer != prop->mType || prop->mDataLength < sizeof(aiTextureMapping))
            {
                ReportError("Material property %s%i is expected to be an integer (size is %i)",
                    prop->mKey.data,prop->mIndex,prop->mDataLength);
            }
            mappings[prop->mIndex] = *((aiTextureMapping*)prop->mData);
        }
        else if (!::strcmp(prop->mKey.data,"$tex.uvtrafo")) {
            if (aiPTI_Float != prop->mType || prop->mDataLength < sizeof(aiUVTransform))
            {
                ReportError("Material property %s%i is expected to be 5 floats large (size is %i)",
                    prop->mKey.data,prop->mIndex, prop->mDataLength);
            }
            mappings[prop->mIndex] = *((aiTextureMapping*)prop->mData);
        }
        else if (!::strcmp(prop->mKey.data,"$tex.uvwsrc")) {
            if (aiPTI_Integer != prop->mType || sizeof(int) > prop->mDataLength)
            {
                ReportError("Material property %s%i is expected to be an integer (size is %i)",
                    prop->mKey.data,prop->mIndex,prop->mDataLength);
            }
            bNoSpecified = false;

            // Ignore UV indices for texture channels that are not there ...

            // Get the value
            iIndex = *((unsigned int*)prop->mData);

            // Check whether there is a mesh using this material
            // which has not enough UV channels ...
            for (unsigned int a = 0; a < mScene->mNumMeshes;++a)
            {
                aiMesh* mesh = this->mScene->mMeshes[a];
                if(mesh->mMaterialIndex == (unsigned int)i)
                {
                    int iChannels = 0;
                    while (mesh->HasTextureCoords(iChannels))++iChannels;
                    if (iIndex >= iChannels)
                    {
                        ReportWarning("Invalid UV index: %i (key %s). Mesh %i has only %i UV channels",
                            iIndex,prop->mKey.data,a,iChannels);
                    }
                }
            }
        }
    }
    if (bNoSpecified)
    {
        // Assume that all textures are using the first UV channel
        for (unsigned int a = 0; a < mScene->mNumMeshes;++a)
        {
            aiMesh* mesh = mScene->mMeshes[a];
            if(mesh->mMaterialIndex == (unsigned int)iIndex && mappings[0] == aiTextureMapping_UV)
            {
                if (!mesh->mTextureCoords[0])
                {
                    // This is a special case ... it could be that the
                    // original mesh format intended the use of a special
                    // mapping here.
                    ReportWarning("UV-mapped texture, but there are no UV coords");
                }
            }
        }
    }
}
// ------------------------------------------------------------------------------------------------
void ValidateDSProcess::Validate( const aiMaterial* pMaterial)
{
    // check whether there are material keys that are obviously not legal
    for (unsigned int i = 0; i < pMaterial->mNumProperties;++i)
    {
        const aiMaterialProperty* prop = pMaterial->mProperties[i];
        if (!prop)  {
            ReportError("aiMaterial::mProperties[%i] is NULL (aiMaterial::mNumProperties is %i)",
                i,pMaterial->mNumProperties);
        }
        if (!prop->mDataLength || !prop->mData) {
            ReportError("aiMaterial::mProperties[%i].mDataLength or "
                "aiMaterial::mProperties[%i].mData is 0",i,i);
        }
        // check all predefined types
        if (aiPTI_String == prop->mType)    {
            // FIX: strings are now stored in a less expensive way, but we can't use the
            // validation routine for 'normal' aiStrings
            if (prop->mDataLength < 5 || prop->mDataLength < 4 + (*reinterpret_cast<uint32_t*>(prop->mData)) + 1)   {
                ReportError("aiMaterial::mProperties[%i].mDataLength is "
                    "too small to contain a string (%i, needed: %i)",
                    i,prop->mDataLength,static_cast<int>(sizeof(aiString)));
            }
            if(prop->mData[prop->mDataLength-1]) {
                ReportError("Missing null-terminator in string material property");
            }
        //  Validate((const aiString*)prop->mData);
        }
        else if (aiPTI_Float == prop->mType)    {
            if (prop->mDataLength < sizeof(float))  {
                ReportError("aiMaterial::mProperties[%i].mDataLength is "
                    "too small to contain a float (%i, needed: %i)",
                    i,prop->mDataLength, static_cast<int>(sizeof(float)));
            }
        }
        else if (aiPTI_Integer == prop->mType)  {
            if (prop->mDataLength < sizeof(int))    {
                ReportError("aiMaterial::mProperties[%i].mDataLength is "
                    "too small to contain an integer (%i, needed: %i)",
                    i,prop->mDataLength, static_cast<int>(sizeof(int)));
            }
        }
        // TODO: check whether there is a key with an unknown name ...
    }

    // make some more specific tests
    ai_real fTemp;
    int iShading;
    if (AI_SUCCESS == aiGetMaterialInteger( pMaterial,AI_MATKEY_SHADING_MODEL,&iShading))   {
        switch ((aiShadingMode)iShading)
        {
        case aiShadingMode_Blinn:
        case aiShadingMode_CookTorrance:
        case aiShadingMode_Phong:

            if (AI_SUCCESS != aiGetMaterialFloat(pMaterial,AI_MATKEY_SHININESS,&fTemp)) {
                ReportWarning("A specular shading model is specified but there is no "
                    "AI_MATKEY_SHININESS key");
            }
            if (AI_SUCCESS == aiGetMaterialFloat(pMaterial,AI_MATKEY_SHININESS_STRENGTH,&fTemp) && !fTemp)  {
                ReportWarning("A specular shading model is specified but the value of the "
                    "AI_MATKEY_SHININESS_STRENGTH key is 0.0");
            }
            break;
        default:
            break;
        }
    }

    if (AI_SUCCESS == aiGetMaterialFloat( pMaterial,AI_MATKEY_OPACITY,&fTemp) && (!fTemp || fTemp > 1.01)) {
        ReportWarning("Invalid opacity value (must be 0 < opacity < 1.0)");
    }

    // Check whether there are invalid texture keys
    // TODO: that's a relict of the past, where texture type and index were baked
    // into the material string ... we could do that in one single pass.
    SearchForInvalidTextures(pMaterial,aiTextureType_DIFFUSE);
    SearchForInvalidTextures(pMaterial,aiTextureType_SPECULAR);
    SearchForInvalidTextures(pMaterial,aiTextureType_AMBIENT);
    SearchForInvalidTextures(pMaterial,aiTextureType_EMISSIVE);
    SearchForInvalidTextures(pMaterial,aiTextureType_OPACITY);
    SearchForInvalidTextures(pMaterial,aiTextureType_SHININESS);
    SearchForInvalidTextures(pMaterial,aiTextureType_HEIGHT);
    SearchForInvalidTextures(pMaterial,aiTextureType_NORMALS);
    SearchForInvalidTextures(pMaterial,aiTextureType_DISPLACEMENT);
    SearchForInvalidTextures(pMaterial,aiTextureType_LIGHTMAP);
    SearchForInvalidTextures(pMaterial,aiTextureType_REFLECTION);
}

// ------------------------------------------------------------------------------------------------
void ValidateDSProcess::Validate( const aiTexture* pTexture)
{
    // the data section may NEVER be NULL
    if (!pTexture->pcData)  {
        ReportError("aiTexture::pcData is NULL");
    }
    if (pTexture->mHeight)
    {
        if (!pTexture->mWidth){
        	ReportError("aiTexture::mWidth is zero (aiTexture::mHeight is %i, uncompressed texture)",
        			pTexture->mHeight);
        }
    }
    else
    {
        if (!pTexture->mWidth) {
            ReportError("aiTexture::mWidth is zero (compressed texture)");
        }
        if ('\0' != pTexture->achFormatHint[3]) {
            ReportWarning("aiTexture::achFormatHint must be zero-terminated");
        }
        else if ('.'  == pTexture->achFormatHint[0])    {
            ReportWarning("aiTexture::achFormatHint should contain a file extension "
                "without a leading dot (format hint: %s).",pTexture->achFormatHint);
        }
    }

    const char* sz = pTexture->achFormatHint;
    if ((sz[0] >= 'A' && sz[0] <= 'Z') ||
        (sz[1] >= 'A' && sz[1] <= 'Z') ||
        (sz[2] >= 'A' && sz[2] <= 'Z') ||
        (sz[3] >= 'A' && sz[3] <= 'Z')) {
        ReportError("aiTexture::achFormatHint contains non-lowercase letters");
    }
}

// ------------------------------------------------------------------------------------------------
void ValidateDSProcess::Validate( const aiAnimation* pAnimation,
     const aiNodeAnim* pNodeAnim)
{
    Validate(&pNodeAnim->mNodeName);

    if (!pNodeAnim->mNumPositionKeys && !pNodeAnim->mScalingKeys && !pNodeAnim->mNumRotationKeys) {
        ReportError("Empty node animation channel");
    }
    // otherwise check whether one of the keys exceeds the total duration of the animation
    if (pNodeAnim->mNumPositionKeys)
    {
        if (!pNodeAnim->mPositionKeys)
        {
        	ReportError("aiNodeAnim::mPositionKeys is NULL (aiNodeAnim::mNumPositionKeys is %i)",
                pNodeAnim->mNumPositionKeys);
        }
        double dLast = -10e10;
        for (unsigned int i = 0; i < pNodeAnim->mNumPositionKeys;++i)
        {
            // ScenePreprocessor will compute the duration if still the default value
            // (Aramis) Add small epsilon, comparison tended to fail if max_time == duration,
            //  seems to be due the compilers register usage/width.
            if (pAnimation->mDuration > 0. && pNodeAnim->mPositionKeys[i].mTime > pAnimation->mDuration+0.001)
            {
                ReportError("aiNodeAnim::mPositionKeys[%i].mTime (%.5f) is larger "
                    "than aiAnimation::mDuration (which is %.5f)",i,
                    (float)pNodeAnim->mPositionKeys[i].mTime,
                    (float)pAnimation->mDuration);
            }
            if (i && pNodeAnim->mPositionKeys[i].mTime <= dLast)
            {
                ReportWarning("aiNodeAnim::mPositionKeys[%i].mTime (%.5f) is smaller "
                    "than aiAnimation::mPositionKeys[%i] (which is %.5f)",i,
                    (float)pNodeAnim->mPositionKeys[i].mTime,
                    i-1, (float)dLast);
            }
            dLast = pNodeAnim->mPositionKeys[i].mTime;
        }
    }
    // rotation keys
    if (pNodeAnim->mNumRotationKeys)
    {
        if (!pNodeAnim->mRotationKeys)
        {
            ReportError("aiNodeAnim::mRotationKeys is NULL (aiNodeAnim::mNumRotationKeys is %i)",
                pNodeAnim->mNumRotationKeys);
        }
        double dLast = -10e10;
        for (unsigned int i = 0; i < pNodeAnim->mNumRotationKeys;++i)
        {
            if (pAnimation->mDuration > 0. && pNodeAnim->mRotationKeys[i].mTime > pAnimation->mDuration+0.001)
            {
                ReportError("aiNodeAnim::mRotationKeys[%i].mTime (%.5f) is larger "
                    "than aiAnimation::mDuration (which is %.5f)",i,
                    (float)pNodeAnim->mRotationKeys[i].mTime,
                    (float)pAnimation->mDuration);
            }
            if (i && pNodeAnim->mRotationKeys[i].mTime <= dLast)
            {
                ReportWarning("aiNodeAnim::mRotationKeys[%i].mTime (%.5f) is smaller "
                    "than aiAnimation::mRotationKeys[%i] (which is %.5f)",i,
                    (float)pNodeAnim->mRotationKeys[i].mTime,
                    i-1, (float)dLast);
            }
            dLast = pNodeAnim->mRotationKeys[i].mTime;
        }
    }
    // scaling keys
    if (pNodeAnim->mNumScalingKeys)
    {
        if (!pNodeAnim->mScalingKeys)   {
            ReportError("aiNodeAnim::mScalingKeys is NULL (aiNodeAnim::mNumScalingKeys is %i)",
                pNodeAnim->mNumScalingKeys);
        }
        double dLast = -10e10;
        for (unsigned int i = 0; i < pNodeAnim->mNumScalingKeys;++i)
        {
            if (pAnimation->mDuration > 0. && pNodeAnim->mScalingKeys[i].mTime > pAnimation->mDuration+0.001)
            {
                ReportError("aiNodeAnim::mScalingKeys[%i].mTime (%.5f) is larger "
                    "than aiAnimation::mDuration (which is %.5f)",i,
                    (float)pNodeAnim->mScalingKeys[i].mTime,
                    (float)pAnimation->mDuration);
            }
            if (i && pNodeAnim->mScalingKeys[i].mTime <= dLast)
            {
                ReportWarning("aiNodeAnim::mScalingKeys[%i].mTime (%.5f) is smaller "
                    "than aiAnimation::mScalingKeys[%i] (which is %.5f)",i,
                    (float)pNodeAnim->mScalingKeys[i].mTime,
                    i-1, (float)dLast);
            }
            dLast = pNodeAnim->mScalingKeys[i].mTime;
        }
    }

    if (!pNodeAnim->mNumScalingKeys && !pNodeAnim->mNumRotationKeys &&
        !pNodeAnim->mNumPositionKeys)
    {
        ReportError("A node animation channel must have at least one subtrack");
    }
}

// ------------------------------------------------------------------------------------------------
void ValidateDSProcess::Validate( const aiNode* pNode)
{
    if (!pNode) {
    	ReportError("A node of the scenegraph is NULL");
    }
    // Validate node name string first so that it's safe to use in below expressions
    this->Validate(&pNode->mName);
    const char* nodeName = (&pNode->mName)->C_Str();
    if (pNode != mScene->mRootNode && !pNode->mParent){
        ReportError("Non-root node %s lacks a valid parent (aiNode::mParent is NULL) ", nodeName);
    }

    // validate all meshes
    if (pNode->mNumMeshes)
    {
        if (!pNode->mMeshes)
        {
            ReportError("aiNode::mMeshes is NULL for node %s (aiNode::mNumMeshes is %i)",
            		  nodeName, pNode->mNumMeshes);
        }
        std::vector<bool> abHadMesh;
        abHadMesh.resize(mScene->mNumMeshes,false);
        for (unsigned int i = 0; i < pNode->mNumMeshes;++i)
        {
            if (pNode->mMeshes[i] >= mScene->mNumMeshes)
            {
                ReportError("aiNode::mMeshes[%i] is out of range for node %s (maximum is %i)",
                    pNode->mMeshes[i], nodeName, mScene->mNumMeshes-1);
            }
            if (abHadMesh[pNode->mMeshes[i]])
            {
                ReportError("aiNode::mMeshes[%i] is already referenced by this node %s (value: %i)",
                    i, nodeName, pNode->mMeshes[i]);
            }
            abHadMesh[pNode->mMeshes[i]] = true;
        }
    }
    if (pNode->mNumChildren)
    {
        if (!pNode->mChildren)  {
            ReportError("aiNode::mChildren is NULL for node %s (aiNode::mNumChildren is %i)",
            		nodeName, pNode->mNumChildren);
        }
        for (unsigned int i = 0; i < pNode->mNumChildren;++i)   {
            Validate(pNode->mChildren[i]);
        }
    }
}

// ------------------------------------------------------------------------------------------------
void ValidateDSProcess::Validate( const aiString* pString)
{
    if (pString->length > MAXLEN)
    {
        ReportError("aiString::length is too large (%lu, maximum is %lu)",
            pString->length,MAXLEN);
    }
    const char* sz = pString->data;
    while (true)
    {
        if ('\0' == *sz)
        {
            if (pString->length != (unsigned int)(sz-pString->data)) {
                ReportError("aiString::data is invalid: the terminal zero is at a wrong offset");
            }
            break;
        }
        else if (sz >= &pString->data[MAXLEN]) {
            ReportError("aiString::data is invalid. There is no terminal character");
        }
        ++sz;
    }
}
