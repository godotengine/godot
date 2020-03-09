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

/** @file A helper class that processes texture transformations */



#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/scene.h>

#include "TextureTransform.h"
#include <assimp/StringUtils.h>

using namespace Assimp;

// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
TextureTransformStep::TextureTransformStep() :
    configFlags()
{
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
TextureTransformStep::~TextureTransformStep()
{
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool TextureTransformStep::IsActive( unsigned int pFlags) const
{
    return  (pFlags & aiProcess_TransformUVCoords) != 0;
}

// ------------------------------------------------------------------------------------------------
// Setup properties
void TextureTransformStep::SetupProperties(const Importer* pImp)
{
    configFlags = pImp->GetPropertyInteger(AI_CONFIG_PP_TUV_EVALUATE,AI_UVTRAFO_ALL);
}

// ------------------------------------------------------------------------------------------------
void TextureTransformStep::PreProcessUVTransform(STransformVecInfo& info)
{
    /*  This function tries to simplify the input UV transformation.
     *  That's very important as it allows us to reduce the number
     *  of output UV channels. The order in which the transformations
     *  are applied is - as always - scaling, rotation, translation.
     */

    char szTemp[512];
    int rounded = 0;


    /* Optimize the rotation angle. That's slightly difficult as
     * we have an inprecise floating-point number (when comparing
     * UV transformations we'll take that into account by using
     * an epsilon of 5 degrees). If there is a rotation value, we can't
     * perform any further optimizations.
     */
    if (info.mRotation)
    {
        float out = info.mRotation;
        if ((rounded = static_cast<int>((info.mRotation / static_cast<float>(AI_MATH_TWO_PI)))))
        {
            out -= rounded * static_cast<float>(AI_MATH_PI);
            ASSIMP_LOG_INFO_F("Texture coordinate rotation ", info.mRotation, " can be simplified to ", out);
        }

        // Next step - convert negative rotation angles to positives
        if (out < 0.f)
            out = (float)AI_MATH_TWO_PI * 2 + out;

        info.mRotation = out;
        return;
    }


    /* Optimize UV translation in the U direction. To determine whether
     * or not we can optimize we need to look at the requested mapping
     * type (e.g. if mirroring is active there IS a difference between
     * offset 2 and 3)
     */
    if ((rounded  = (int)info.mTranslation.x))  {
        float out = 0.0f;
        szTemp[0] = 0;
        if (aiTextureMapMode_Wrap == info.mapU) {
            // Wrap - simple take the fraction of the field
            out = info.mTranslation.x-(float)rounded;
			ai_snprintf(szTemp, 512, "[w] UV U offset %f can be simplified to %f", info.mTranslation.x, out);
        }
        else if (aiTextureMapMode_Mirror == info.mapU && 1 != rounded)  {
            // Mirror
            if (rounded % 2)
                rounded--;
            out = info.mTranslation.x-(float)rounded;

            ai_snprintf(szTemp,512,"[m/d] UV U offset %f can be simplified to %f",info.mTranslation.x,out);
        }
        else if (aiTextureMapMode_Clamp == info.mapU || aiTextureMapMode_Decal == info.mapU)    {
            // Clamp - translations beyond 1,1 are senseless
            ai_snprintf(szTemp,512,"[c] UV U offset %f can be clamped to 1.0f",info.mTranslation.x);

            out = 1.f;
        }
        if (szTemp[0])      {
            ASSIMP_LOG_INFO(szTemp);
            info.mTranslation.x = out;
        }
    }

    /* Optimize UV translation in the V direction. To determine whether
     * or not we can optimize we need to look at the requested mapping
     * type (e.g. if mirroring is active there IS a difference between
     * offset 2 and 3)
     */
    if ((rounded  = (int)info.mTranslation.y))  {
        float out = 0.0f;
        szTemp[0] = 0;
        if (aiTextureMapMode_Wrap == info.mapV) {
            // Wrap - simple take the fraction of the field
            out = info.mTranslation.y-(float)rounded;
            ::ai_snprintf(szTemp,512,"[w] UV V offset %f can be simplified to %f",info.mTranslation.y,out);
        }
        else if (aiTextureMapMode_Mirror == info.mapV  && 1 != rounded) {
            // Mirror
            if (rounded % 2)
                rounded--;
            out = info.mTranslation.x-(float)rounded;

            ::ai_snprintf(szTemp,512,"[m/d] UV V offset %f can be simplified to %f",info.mTranslation.y,out);
        }
        else if (aiTextureMapMode_Clamp == info.mapV || aiTextureMapMode_Decal == info.mapV)    {
            // Clamp - translations beyond 1,1 are senseless
            ::ai_snprintf(szTemp,512,"[c] UV V offset %f canbe clamped to 1.0f",info.mTranslation.y);

            out = 1.f;
        }
        if (szTemp[0])  {
            ASSIMP_LOG_INFO(szTemp);
            info.mTranslation.y = out;
        }
    }
    return;
}

// ------------------------------------------------------------------------------------------------
void UpdateUVIndex(const std::list<TTUpdateInfo>& l, unsigned int n)
{
    // Don't set if == 0 && wasn't set before
    for (std::list<TTUpdateInfo>::const_iterator it = l.begin();it != l.end(); ++it) {
        const TTUpdateInfo& info = *it;

        if (info.directShortcut)
            *info.directShortcut = n;
        else if (!n)
        {
            info.mat->AddProperty<int>((int*)&n,1,AI_MATKEY_UVWSRC(info.semantic,info.index));
        }
    }
}

// ------------------------------------------------------------------------------------------------
inline const char* MappingModeToChar(aiTextureMapMode map)
{
    if (aiTextureMapMode_Wrap == map)
        return "-w";

    if (aiTextureMapMode_Mirror == map)
        return "-m";

    return "-c";
}

// ------------------------------------------------------------------------------------------------
void TextureTransformStep::Execute( aiScene* pScene)
{
    ASSIMP_LOG_DEBUG("TransformUVCoordsProcess begin");


    /*  We build a per-mesh list of texture transformations we'll need
     *  to apply. To achieve this, we iterate through all materials,
     *  find all textures and get their transformations and UV indices.
     *  Then we search for all meshes using this material.
     */
    typedef std::list<STransformVecInfo> MeshTrafoList;
    std::vector<MeshTrafoList> meshLists(pScene->mNumMeshes);

    for (unsigned int i = 0; i < pScene->mNumMaterials;++i) {

        aiMaterial* mat = pScene->mMaterials[i];
        for (unsigned int a = 0; a < mat->mNumProperties;++a)   {

            aiMaterialProperty* prop = mat->mProperties[a];
            if (!::strcmp( prop->mKey.data, "$tex.file"))   {
                STransformVecInfo info;

                // Setup a shortcut structure to allow for a fast updating
                // of the UV index later
                TTUpdateInfo update;
                update.mat = (aiMaterial*) mat;
                update.semantic = prop->mSemantic;
                update.index = prop->mIndex;

                // Get textured properties and transform
                for (unsigned int a2 = 0; a2 < mat->mNumProperties;++a2)  {
                    aiMaterialProperty* prop2 = mat->mProperties[a2];
                    if (prop2->mSemantic != prop->mSemantic || prop2->mIndex != prop->mIndex) {
                        continue;
                    }

                    if ( !::strcmp( prop2->mKey.data, "$tex.uvwsrc")) {
                        info.uvIndex = *((int*)prop2->mData);

                        // Store a direct pointer for later use
                        update.directShortcut = (unsigned int*) prop2->mData;
                    }

                    else if ( !::strcmp( prop2->mKey.data, "$tex.mapmodeu")) {
                        info.mapU = *((aiTextureMapMode*)prop2->mData);
                    }
                    else if ( !::strcmp( prop2->mKey.data, "$tex.mapmodev")) {
                        info.mapV = *((aiTextureMapMode*)prop2->mData);
                    }
                    else if ( !::strcmp( prop2->mKey.data, "$tex.uvtrafo"))  {
                        // ValidateDS should check this
                        ai_assert(prop2->mDataLength >= 20);
                        ::memcpy(&info.mTranslation.x,prop2->mData,sizeof(float)*5);

                        // Directly remove this property from the list
                        mat->mNumProperties--;
                        for (unsigned int a3 = a2; a3 < mat->mNumProperties;++a3) {
                            mat->mProperties[a3] = mat->mProperties[a3+1];
                        }

                        delete prop2;

                        // Warn: could be an underflow, but this does not invoke undefined behaviour
                        --a2;
                    }
                }

                // Find out which transformations are to be evaluated
                if (!(configFlags & AI_UVTRAFO_ROTATION)) {
                    info.mRotation = 0.f;
                }
                if (!(configFlags & AI_UVTRAFO_SCALING)) {
                    info.mScaling = aiVector2D(1.f,1.f);
                }
                if (!(configFlags & AI_UVTRAFO_TRANSLATION)) {
                    info.mTranslation = aiVector2D(0.f,0.f);
                }

                // Do some preprocessing
                PreProcessUVTransform(info);
                info.uvIndex = std::min(info.uvIndex,AI_MAX_NUMBER_OF_TEXTURECOORDS -1u);

                // Find out whether this material is used by more than
                // one mesh. This will make our task much, much more difficult!
                unsigned int cnt = 0;
                for (unsigned int n = 0; n < pScene->mNumMeshes;++n)    {
                    if (pScene->mMeshes[n]->mMaterialIndex == i)
                        ++cnt;
                }

                if (!cnt)
                    continue;
                else if (1 != cnt)  {
                    // This material is referenced by more than one mesh!
                    // So we need to make sure the UV index for the texture
                    // is identical for each of it ...
                    info.lockedPos = AI_TT_UV_IDX_LOCK_TBD;
                }

                // Get all corresponding meshes
                for (unsigned int n = 0; n < pScene->mNumMeshes;++n)    {
                    aiMesh* mesh = pScene->mMeshes[n];
                    if (mesh->mMaterialIndex != i || !mesh->mTextureCoords[0])
                        continue;

                    unsigned int uv = info.uvIndex;
                    if (!mesh->mTextureCoords[uv])  {
                        // If the requested UV index is not available, take the first one instead.
                        uv = 0;
                    }

                    if (mesh->mNumUVComponents[info.uvIndex] >= 3){
                        ASSIMP_LOG_WARN("UV transformations on 3D mapping channels are not supported");
                        continue;
                    }

                    MeshTrafoList::iterator it;

                    // Check whether we have this transform setup already
                    for (it = meshLists[n].begin();it != meshLists[n].end(); ++it)  {

                        if ((*it) == info && (*it).uvIndex == uv)   {
                            (*it).updateList.push_back(update);
                            break;
                        }
                    }

                    if (it == meshLists[n].end())   {
                        meshLists[n].push_back(info);
                        meshLists[n].back().uvIndex = uv;
                        meshLists[n].back().updateList.push_back(update);
                    }
                }
            }
        }
    }

    char buffer[1024]; // should be sufficiently large
    unsigned int outChannels = 0, inChannels = 0, transformedChannels = 0;

    // Now process all meshes. Important: we don't remove unreferenced UV channels.
    // This is a job for the RemoveUnreferencedData-Step.
    for (unsigned int q = 0; q < pScene->mNumMeshes;++q)    {

        aiMesh* mesh = pScene->mMeshes[q];
        MeshTrafoList& trafo =  meshLists[q];

        inChannels += mesh->GetNumUVChannels();

        if (!mesh->mTextureCoords[0] || trafo.empty() ||  (trafo.size() == 1 && trafo.begin()->IsUntransformed())) {
            outChannels += mesh->GetNumUVChannels();
            continue;
        }

        // Move untransformed UV channels to the first position in the list ....
        // except if we need a new locked index which should be as small as possible
        bool veto = false, need = false;
        unsigned int cnt = 0;
        unsigned int untransformed = 0;

        MeshTrafoList::iterator it,it2;
        for (it = trafo.begin();it != trafo.end(); ++it,++cnt)  {

            if (!(*it).IsUntransformed()) {
                need = true;
            }

            if ((*it).lockedPos == AI_TT_UV_IDX_LOCK_TBD)   {
                // Lock this index and make sure it won't be changed
                (*it).lockedPos = cnt;
                veto = true;
                continue;
            }

            if (!veto && it != trafo.begin() && (*it).IsUntransformed())    {
                for (it2 = trafo.begin();it2 != it; ++it2) {
                    if (!(*it2).IsUntransformed())
                        break;
                }
                trafo.insert(it2,*it);
                trafo.erase(it);
                break;
            }
        }
        if (!need)
            continue;

        // Find all that are not at their 'locked' position and move them to it.
        // Conflicts are possible but quite unlikely.
        cnt = 0;
        for (it = trafo.begin();it != trafo.end(); ++it,++cnt) {
            if ((*it).lockedPos != AI_TT_UV_IDX_LOCK_NONE && (*it).lockedPos != cnt) {
                it2 = trafo.begin();unsigned int t = 0;
                while (t != (*it).lockedPos)
                    ++it2;

                if ((*it2).lockedPos != AI_TT_UV_IDX_LOCK_NONE) {
                    ASSIMP_LOG_ERROR("Channel mismatch, can't compute all transformations properly [design bug]");
                    continue;
                }

                std::swap(*it2,*it);
                if ((*it).lockedPos == untransformed)
                    untransformed = cnt;
            }
        }

        // ... and add dummies for all unreferenced channels
        // at the end of the list
        bool ref[AI_MAX_NUMBER_OF_TEXTURECOORDS];
        for (unsigned int n = 0; n < AI_MAX_NUMBER_OF_TEXTURECOORDS;++n)
            ref[n] = (!mesh->mTextureCoords[n] ? true : false);

        for (it = trafo.begin();it != trafo.end(); ++it)
            ref[(*it).uvIndex] = true;

        for (unsigned int n = 0; n < AI_MAX_NUMBER_OF_TEXTURECOORDS;++n) {
            if (ref[n])
                continue;
            trafo.push_back(STransformVecInfo());
            trafo.back().uvIndex = n;
        }

        // Then check whether this list breaks the channel limit.
        // The unimportant ones are at the end of the list, so
        // it shouldn't be too worse if we remove them.
        unsigned int size = (unsigned int)trafo.size();
        if (size > AI_MAX_NUMBER_OF_TEXTURECOORDS) {

            if (!DefaultLogger::isNullLogger()) {
                ASSIMP_LOG_ERROR_F(static_cast<unsigned int>(trafo.size()), " UV channels required but just ", 
                    AI_MAX_NUMBER_OF_TEXTURECOORDS, " available");
            }
            size = AI_MAX_NUMBER_OF_TEXTURECOORDS;
        }


        aiVector3D* old[AI_MAX_NUMBER_OF_TEXTURECOORDS];
        for (unsigned int n = 0; n < AI_MAX_NUMBER_OF_TEXTURECOORDS;++n)
            old[n] = mesh->mTextureCoords[n];

        // Now continue and generate the output channels. Channels
        // that we're not going to need later can be overridden.
        it = trafo.begin();
        for (unsigned int n = 0; n < trafo.size();++n,++it) {

            if (n >= size)  {
                // Try to use an untransformed channel for all channels we threw over board
                UpdateUVIndex((*it).updateList,untransformed);
                continue;
            }

            outChannels++;

            // Write to the log
            if (!DefaultLogger::isNullLogger()) {
                ::ai_snprintf(buffer,1024,"Mesh %u, channel %u: t(%.3f,%.3f), s(%.3f,%.3f), r(%.3f), %s%s",
                    q,n,
                    (*it).mTranslation.x,
                    (*it).mTranslation.y,
                    (*it).mScaling.x,
                    (*it).mScaling.y,
                    AI_RAD_TO_DEG( (*it).mRotation),
                    MappingModeToChar ((*it).mapU),
                    MappingModeToChar ((*it).mapV));

                ASSIMP_LOG_INFO(buffer);
            }

            // Check whether we need a new buffer here
            if (mesh->mTextureCoords[n])    {

                it2 = it;++it2;
                for (unsigned int m = n+1; m < size;++m, ++it2) {

                    if ((*it2).uvIndex == n){
                        it2 = trafo.begin();
                        break;
                    }
                }
                if (it2 == trafo.begin()){
                    mesh->mTextureCoords[n] = new aiVector3D[mesh->mNumVertices];
                }
            }
            else mesh->mTextureCoords[n] = new aiVector3D[mesh->mNumVertices];

            aiVector3D* src = old[(*it).uvIndex];
            aiVector3D* dest, *end;
            dest = mesh->mTextureCoords[n];

            ai_assert(NULL != src);

            // Copy the data to the destination array
            if (dest != src)
                ::memcpy(dest,src,sizeof(aiVector3D)*mesh->mNumVertices);

            end = dest + mesh->mNumVertices;

            // Build a transformation matrix and transform all UV coords with it
            if (!(*it).IsUntransformed()) {
                const aiVector2D& trl = (*it).mTranslation;
                const aiVector2D& scl = (*it).mScaling;

                // fixme: simplify ..
                ++transformedChannels;
                aiMatrix3x3 matrix;

                aiMatrix3x3 m2,m3,m4,m5;

                m4.a1 = scl.x;
                m4.b2 = scl.y;

                m2.a3 = m2.b3 = 0.5f;
                m3.a3 = m3.b3 = -0.5f;

                if ((*it).mRotation > AI_TT_ROTATION_EPSILON )
                    aiMatrix3x3::RotationZ((*it).mRotation,matrix);

                m5.a3 += trl.x; m5.b3 += trl.y;
                matrix = m2 * m4 * matrix * m3 * m5;

                for (src = dest; src != end; ++src) { /* manual homogenious divide */
                    src->z = 1.f;
                    *src = matrix * *src;
                    src->x /= src->z;
                    src->y /= src->z;
                    src->z = 0.f;
                }
            }

            // Update all UV indices
            UpdateUVIndex((*it).updateList,n);
        }
    }

    // Print some detailed statistics into the log
    if (!DefaultLogger::isNullLogger()) {

        if (transformedChannels)    {
            ASSIMP_LOG_INFO_F("TransformUVCoordsProcess end: ", outChannels, " output channels (in: ", inChannels, ", modified: ", transformedChannels,")");
        } else {
            ASSIMP_LOG_DEBUG("TransformUVCoordsProcess finished");
        }
    }
}


