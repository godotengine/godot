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

/** @file PretransformVertices.cpp
 *  @brief Implementation of the "PretransformVertices" post processing step
*/


#include "PretransformVertices.h"
#include "ProcessHelper.h"
#include <assimp/SceneCombiner.h>
#include <assimp/Exceptional.h>

using namespace Assimp;

// some array offsets
#define AI_PTVS_VERTEX 0x0
#define AI_PTVS_FACE 0x1

// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
PretransformVertices::PretransformVertices()
: configKeepHierarchy (false)
, configNormalize(false)
, configTransform(false)
, configTransformation()
, mConfigPointCloud( false ) {
    // empty
}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
PretransformVertices::~PretransformVertices() {
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool PretransformVertices::IsActive( unsigned int pFlags) const
{
    return  (pFlags & aiProcess_PreTransformVertices) != 0;
}

// ------------------------------------------------------------------------------------------------
// Setup import configuration
void PretransformVertices::SetupProperties(const Importer* pImp)
{
    // Get the current value of AI_CONFIG_PP_PTV_KEEP_HIERARCHY, AI_CONFIG_PP_PTV_NORMALIZE,
    // AI_CONFIG_PP_PTV_ADD_ROOT_TRANSFORMATION and AI_CONFIG_PP_PTV_ROOT_TRANSFORMATION
    configKeepHierarchy = (0 != pImp->GetPropertyInteger(AI_CONFIG_PP_PTV_KEEP_HIERARCHY,0));
    configNormalize = (0 != pImp->GetPropertyInteger(AI_CONFIG_PP_PTV_NORMALIZE,0));
    configTransform = (0 != pImp->GetPropertyInteger(AI_CONFIG_PP_PTV_ADD_ROOT_TRANSFORMATION,0));

    configTransformation = pImp->GetPropertyMatrix(AI_CONFIG_PP_PTV_ROOT_TRANSFORMATION, aiMatrix4x4());

    mConfigPointCloud = pImp->GetPropertyBool(AI_CONFIG_EXPORT_POINT_CLOUDS);
}

// ------------------------------------------------------------------------------------------------
// Count the number of nodes
unsigned int PretransformVertices::CountNodes( aiNode* pcNode )
{
    unsigned int iRet = 1;
    for (unsigned int i = 0;i < pcNode->mNumChildren;++i)
    {
        iRet += CountNodes(pcNode->mChildren[i]);
    }
    return iRet;
}

// ------------------------------------------------------------------------------------------------
// Get a bitwise combination identifying the vertex format of a mesh
unsigned int PretransformVertices::GetMeshVFormat( aiMesh* pcMesh )
{
    // the vertex format is stored in aiMesh::mBones for later retrieval.
    // there isn't a good reason to compute it a few hundred times
    // from scratch. The pointer is unused as animations are lost
    // during PretransformVertices.
    if (pcMesh->mBones)
        return (unsigned int)(uint64_t)pcMesh->mBones;


    const unsigned int iRet = GetMeshVFormatUnique(pcMesh);

    // store the value for later use
    pcMesh->mBones = (aiBone**)(uint64_t)iRet;
    return iRet;
}

// ------------------------------------------------------------------------------------------------
// Count the number of vertices in the whole scene and a given
// material index
void PretransformVertices::CountVerticesAndFaces( aiScene* pcScene, aiNode* pcNode, unsigned int iMat,
    unsigned int iVFormat, unsigned int* piFaces, unsigned int* piVertices)
{
    for (unsigned int i = 0; i < pcNode->mNumMeshes;++i)
    {
        aiMesh* pcMesh = pcScene->mMeshes[ pcNode->mMeshes[i] ];
        if (iMat == pcMesh->mMaterialIndex && iVFormat == GetMeshVFormat(pcMesh))
        {
            *piVertices += pcMesh->mNumVertices;
            *piFaces += pcMesh->mNumFaces;
        }
    }
    for (unsigned int i = 0;i < pcNode->mNumChildren;++i)
    {
        CountVerticesAndFaces(pcScene,pcNode->mChildren[i],iMat,
            iVFormat,piFaces,piVertices);
    }
}

// ------------------------------------------------------------------------------------------------
// Collect vertex/face data
void PretransformVertices::CollectData( aiScene* pcScene, aiNode* pcNode, unsigned int iMat,
    unsigned int iVFormat, aiMesh* pcMeshOut,
    unsigned int aiCurrent[2], unsigned int* num_refs)
{
    // No need to multiply if there's no transformation
    const bool identity = pcNode->mTransformation.IsIdentity();
    for (unsigned int i = 0; i < pcNode->mNumMeshes;++i)
    {
        aiMesh* pcMesh = pcScene->mMeshes[ pcNode->mMeshes[i] ];
        if (iMat == pcMesh->mMaterialIndex && iVFormat == GetMeshVFormat(pcMesh))
        {
            // Decrement mesh reference counter
            unsigned int& num_ref = num_refs[pcNode->mMeshes[i]];
            ai_assert(0 != num_ref);
            --num_ref;
            // Save the name of the last mesh
            if (num_ref==0)
            {
                pcMeshOut->mName = pcMesh->mName;
            }

            if (identity)   {
                // copy positions without modifying them
                ::memcpy(pcMeshOut->mVertices + aiCurrent[AI_PTVS_VERTEX],
                    pcMesh->mVertices,
                    pcMesh->mNumVertices * sizeof(aiVector3D));

                if (iVFormat & 0x2) {
                    // copy normals without modifying them
                    ::memcpy(pcMeshOut->mNormals + aiCurrent[AI_PTVS_VERTEX],
                        pcMesh->mNormals,
                        pcMesh->mNumVertices * sizeof(aiVector3D));
                }
                if (iVFormat & 0x4)
                {
                    // copy tangents without modifying them
                    ::memcpy(pcMeshOut->mTangents + aiCurrent[AI_PTVS_VERTEX],
                        pcMesh->mTangents,
                        pcMesh->mNumVertices * sizeof(aiVector3D));
                    // copy bitangents without modifying them
                    ::memcpy(pcMeshOut->mBitangents + aiCurrent[AI_PTVS_VERTEX],
                        pcMesh->mBitangents,
                        pcMesh->mNumVertices * sizeof(aiVector3D));
                }
            }
            else
            {
                // copy positions, transform them to worldspace
                for (unsigned int n = 0; n < pcMesh->mNumVertices;++n)  {
                    pcMeshOut->mVertices[aiCurrent[AI_PTVS_VERTEX]+n] = pcNode->mTransformation * pcMesh->mVertices[n];
                }
                aiMatrix4x4 mWorldIT = pcNode->mTransformation;
                mWorldIT.Inverse().Transpose();

                // TODO: implement Inverse() for aiMatrix3x3
                aiMatrix3x3 m = aiMatrix3x3(mWorldIT);

                if (iVFormat & 0x2)
                {
                    // copy normals, transform them to worldspace
                    for (unsigned int n = 0; n < pcMesh->mNumVertices;++n)  {
                        pcMeshOut->mNormals[aiCurrent[AI_PTVS_VERTEX]+n] =
                            (m * pcMesh->mNormals[n]).Normalize();
                    }
                }
                if (iVFormat & 0x4)
                {
                    // copy tangents and bitangents, transform them to worldspace
                    for (unsigned int n = 0; n < pcMesh->mNumVertices;++n)  {
                        pcMeshOut->mTangents  [aiCurrent[AI_PTVS_VERTEX]+n] = (m * pcMesh->mTangents[n]).Normalize();
                        pcMeshOut->mBitangents[aiCurrent[AI_PTVS_VERTEX]+n] = (m * pcMesh->mBitangents[n]).Normalize();
                    }
                }
            }
            unsigned int p = 0;
            while (iVFormat & (0x100 << p))
            {
                // copy texture coordinates
                memcpy(pcMeshOut->mTextureCoords[p] + aiCurrent[AI_PTVS_VERTEX],
                    pcMesh->mTextureCoords[p],
                    pcMesh->mNumVertices * sizeof(aiVector3D));
                ++p;
            }
            p = 0;
            while (iVFormat & (0x1000000 << p))
            {
                // copy vertex colors
                memcpy(pcMeshOut->mColors[p] + aiCurrent[AI_PTVS_VERTEX],
                    pcMesh->mColors[p],
                    pcMesh->mNumVertices * sizeof(aiColor4D));
                ++p;
            }
            // now we need to copy all faces. since we will delete the source mesh afterwards,
            // we don't need to reallocate the array of indices except if this mesh is
            // referenced multiple times.
            for (unsigned int planck = 0;planck < pcMesh->mNumFaces;++planck)
            {
                aiFace& f_src = pcMesh->mFaces[planck];
                aiFace& f_dst = pcMeshOut->mFaces[aiCurrent[AI_PTVS_FACE]+planck];

                const unsigned int num_idx = f_src.mNumIndices;

                f_dst.mNumIndices = num_idx;

                unsigned int* pi;
                if (!num_ref) { /* if last time the mesh is referenced -> no reallocation */
                    pi = f_dst.mIndices = f_src.mIndices;

                    // offset all vertex indices
                    for (unsigned int hahn = 0; hahn < num_idx;++hahn){
                        pi[hahn] += aiCurrent[AI_PTVS_VERTEX];
                    }
                }
                else {
                    pi = f_dst.mIndices = new unsigned int[num_idx];

                    // copy and offset all vertex indices
                    for (unsigned int hahn = 0; hahn < num_idx;++hahn){
                        pi[hahn] = f_src.mIndices[hahn] + aiCurrent[AI_PTVS_VERTEX];
                    }
                }

                // Update the mPrimitiveTypes member of the mesh
                switch (pcMesh->mFaces[planck].mNumIndices)
                {
                case 0x1:
                    pcMeshOut->mPrimitiveTypes |= aiPrimitiveType_POINT;
                    break;
                case 0x2:
                    pcMeshOut->mPrimitiveTypes |= aiPrimitiveType_LINE;
                    break;
                case 0x3:
                    pcMeshOut->mPrimitiveTypes |= aiPrimitiveType_TRIANGLE;
                    break;
                default:
                    pcMeshOut->mPrimitiveTypes |= aiPrimitiveType_POLYGON;
                    break;
                };
            }
            aiCurrent[AI_PTVS_VERTEX] += pcMesh->mNumVertices;
            aiCurrent[AI_PTVS_FACE]   += pcMesh->mNumFaces;
        }
    }

    // append all children of us
    for (unsigned int i = 0;i < pcNode->mNumChildren;++i) {
        CollectData(pcScene,pcNode->mChildren[i],iMat,
            iVFormat,pcMeshOut,aiCurrent,num_refs);
    }
}

// ------------------------------------------------------------------------------------------------
// Get a list of all vertex formats that occur for a given material index
// The output list contains duplicate elements
void PretransformVertices::GetVFormatList( aiScene* pcScene, unsigned int iMat,
    std::list<unsigned int>& aiOut)
{
    for (unsigned int i = 0; i < pcScene->mNumMeshes;++i)
    {
        aiMesh* pcMesh = pcScene->mMeshes[ i ];
        if (iMat == pcMesh->mMaterialIndex) {
            aiOut.push_back(GetMeshVFormat(pcMesh));
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Compute the absolute transformation matrices of each node
void PretransformVertices::ComputeAbsoluteTransform( aiNode* pcNode )
{
    if (pcNode->mParent)    {
        pcNode->mTransformation = pcNode->mParent->mTransformation*pcNode->mTransformation;
    }

    for (unsigned int i = 0;i < pcNode->mNumChildren;++i)   {
        ComputeAbsoluteTransform(pcNode->mChildren[i]);
    }
}

// ------------------------------------------------------------------------------------------------
// Apply the node transformation to a mesh
void PretransformVertices::ApplyTransform(aiMesh* mesh, const aiMatrix4x4& mat)
{
    // Check whether we need to transform the coordinates at all
    if (!mat.IsIdentity()) {

        if (mesh->HasPositions()) {
            for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
                mesh->mVertices[i] = mat * mesh->mVertices[i];
            }
        }
        if (mesh->HasNormals() || mesh->HasTangentsAndBitangents()) {
            aiMatrix4x4 mWorldIT = mat;
            mWorldIT.Inverse().Transpose();

            // TODO: implement Inverse() for aiMatrix3x3
            aiMatrix3x3 m = aiMatrix3x3(mWorldIT);

            if (mesh->HasNormals()) {
                for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
                    mesh->mNormals[i] = (m * mesh->mNormals[i]).Normalize();
                }
            }
            if (mesh->HasTangentsAndBitangents()) {
                for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
                    mesh->mTangents[i]   = (m * mesh->mTangents[i]).Normalize();
                    mesh->mBitangents[i] = (m * mesh->mBitangents[i]).Normalize();
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Simple routine to build meshes in worldspace, no further optimization
void PretransformVertices::BuildWCSMeshes(std::vector<aiMesh*>& out, aiMesh** in,
    unsigned int numIn, aiNode* node)
{
    // NOTE:
    //  aiMesh::mNumBones store original source mesh, or UINT_MAX if not a copy
    //  aiMesh::mBones store reference to abs. transform we multiplied with

    // process meshes
    for (unsigned int i = 0; i < node->mNumMeshes;++i) {
        aiMesh* mesh = in[node->mMeshes[i]];

        // check whether we can operate on this mesh
        if (!mesh->mBones || *reinterpret_cast<aiMatrix4x4*>(mesh->mBones) == node->mTransformation) {
            // yes, we can.
            mesh->mBones = reinterpret_cast<aiBone**> (&node->mTransformation);
            mesh->mNumBones = UINT_MAX;
        }
        else {

            // try to find us in the list of newly created meshes
            for (unsigned int n = 0; n < out.size(); ++n) {
                aiMesh* ctz = out[n];
                if (ctz->mNumBones == node->mMeshes[i] && *reinterpret_cast<aiMatrix4x4*>(ctz->mBones) ==  node->mTransformation) {

                    // ok, use this one. Update node mesh index
                    node->mMeshes[i] = numIn + n;
                }
            }
            if (node->mMeshes[i] < numIn) {
                // Worst case. Need to operate on a full copy of the mesh
                ASSIMP_LOG_INFO("PretransformVertices: Copying mesh due to mismatching transforms");
                aiMesh* ntz;

                const unsigned int tmp = mesh->mNumBones; //
                mesh->mNumBones = 0;
                SceneCombiner::Copy(&ntz,mesh);
                mesh->mNumBones = tmp;

                ntz->mNumBones = node->mMeshes[i];
                ntz->mBones = reinterpret_cast<aiBone**> (&node->mTransformation);

                out.push_back(ntz);

                node->mMeshes[i] = static_cast<unsigned int>(numIn + out.size() - 1);
            }
        }
    }

    // call children
    for (unsigned int i = 0; i < node->mNumChildren;++i)
        BuildWCSMeshes(out,in,numIn,node->mChildren[i]);
}

// ------------------------------------------------------------------------------------------------
// Reset transformation matrices to identity
void PretransformVertices::MakeIdentityTransform(aiNode* nd)
{
    nd->mTransformation = aiMatrix4x4();

    // call children
    for (unsigned int i = 0; i < nd->mNumChildren;++i)
        MakeIdentityTransform(nd->mChildren[i]);
}

// ------------------------------------------------------------------------------------------------
// Build reference counters for all meshes
void PretransformVertices::BuildMeshRefCountArray(aiNode* nd, unsigned int * refs)
{
    for (unsigned int i = 0; i< nd->mNumMeshes;++i)
        refs[nd->mMeshes[i]]++;

    // call children
    for (unsigned int i = 0; i < nd->mNumChildren;++i)
        BuildMeshRefCountArray(nd->mChildren[i],refs);
}

// ------------------------------------------------------------------------------------------------
// Executes the post processing step on the given imported data.
void PretransformVertices::Execute( aiScene* pScene)
{
    ASSIMP_LOG_DEBUG("PretransformVerticesProcess begin");

    // Return immediately if we have no meshes
    if (!pScene->mNumMeshes)
        return;

    const unsigned int iOldMeshes = pScene->mNumMeshes;
    const unsigned int iOldAnimationChannels = pScene->mNumAnimations;
    const unsigned int iOldNodes = CountNodes(pScene->mRootNode);

    if(configTransform) {
        pScene->mRootNode->mTransformation = configTransformation;
    }

    // first compute absolute transformation matrices for all nodes
    ComputeAbsoluteTransform(pScene->mRootNode);

    // Delete aiMesh::mBones for all meshes. The bones are
    // removed during this step and we need the pointer as
    // temporary storage
    for (unsigned int i = 0; i < pScene->mNumMeshes;++i)    {
        aiMesh* mesh = pScene->mMeshes[i];

        for (unsigned int a = 0; a < mesh->mNumBones;++a)
            delete mesh->mBones[a];

        delete[] mesh->mBones;
        mesh->mBones = NULL;
    }

    // now build a list of output meshes
    std::vector<aiMesh*> apcOutMeshes;

    // Keep scene hierarchy? It's an easy job in this case ...
    // we go on and transform all meshes, if one is referenced by nodes
    // with different absolute transformations a depth copy of the mesh
    // is required.
    if( configKeepHierarchy ) {

        // Hack: store the matrix we're transforming a mesh with in aiMesh::mBones
        BuildWCSMeshes(apcOutMeshes,pScene->mMeshes,pScene->mNumMeshes, pScene->mRootNode);

        // ... if new meshes have been generated, append them to the end of the scene
        if (apcOutMeshes.size() > 0) {
            aiMesh** npp = new aiMesh*[pScene->mNumMeshes + apcOutMeshes.size()];

            memcpy(npp,pScene->mMeshes,sizeof(aiMesh*)*pScene->mNumMeshes);
            memcpy(npp+pScene->mNumMeshes,&apcOutMeshes[0],sizeof(aiMesh*)*apcOutMeshes.size());

            pScene->mNumMeshes  += static_cast<unsigned int>(apcOutMeshes.size());
            delete[] pScene->mMeshes; pScene->mMeshes = npp;
        }

        // now iterate through all meshes and transform them to worldspace
        for (unsigned int i = 0; i < pScene->mNumMeshes; ++i) {
            ApplyTransform(pScene->mMeshes[i],*reinterpret_cast<aiMatrix4x4*>( pScene->mMeshes[i]->mBones ));

            // prevent improper destruction
            pScene->mMeshes[i]->mBones    = NULL;
            pScene->mMeshes[i]->mNumBones = 0;
        }
    } else {
        apcOutMeshes.reserve(pScene->mNumMaterials<<1u);
        std::list<unsigned int> aiVFormats;

        std::vector<unsigned int> s(pScene->mNumMeshes,0);
        BuildMeshRefCountArray(pScene->mRootNode,&s[0]);

        for (unsigned int i = 0; i < pScene->mNumMaterials;++i)     {
            // get the list of all vertex formats for this material
            aiVFormats.clear();
            GetVFormatList(pScene,i,aiVFormats);
            aiVFormats.sort();
            aiVFormats.unique();
            for (std::list<unsigned int>::const_iterator j =  aiVFormats.begin();j != aiVFormats.end();++j) {
                unsigned int iVertices = 0;
                unsigned int iFaces = 0;
                CountVerticesAndFaces(pScene,pScene->mRootNode,i,*j,&iFaces,&iVertices);
                if (0 != iFaces && 0 != iVertices)
                {
                    apcOutMeshes.push_back(new aiMesh());
                    aiMesh* pcMesh = apcOutMeshes.back();
                    pcMesh->mNumFaces = iFaces;
                    pcMesh->mNumVertices = iVertices;
                    pcMesh->mFaces = new aiFace[iFaces];
                    pcMesh->mVertices = new aiVector3D[iVertices];
                    pcMesh->mMaterialIndex = i;
                    if ((*j) & 0x2)pcMesh->mNormals = new aiVector3D[iVertices];
                    if ((*j) & 0x4)
                    {
                        pcMesh->mTangents    = new aiVector3D[iVertices];
                        pcMesh->mBitangents  = new aiVector3D[iVertices];
                    }
                    iFaces = 0;
                    while ((*j) & (0x100 << iFaces))
                    {
                        pcMesh->mTextureCoords[iFaces] = new aiVector3D[iVertices];
                        if ((*j) & (0x10000 << iFaces))pcMesh->mNumUVComponents[iFaces] = 3;
                        else pcMesh->mNumUVComponents[iFaces] = 2;
                        iFaces++;
                    }
                    iFaces = 0;
                    while ((*j) & (0x1000000 << iFaces))
                        pcMesh->mColors[iFaces++] = new aiColor4D[iVertices];

                    // fill the mesh ...
                    unsigned int aiTemp[2] = {0,0};
                    CollectData(pScene,pScene->mRootNode,i,*j,pcMesh,aiTemp,&s[0]);
                }
            }
        }

        // If no meshes are referenced in the node graph it is possible that we get no output meshes.
        if (apcOutMeshes.empty()) {
            
            throw DeadlyImportError("No output meshes: all meshes are orphaned and are not referenced by any nodes");
        }
        else
        {
            // now delete all meshes in the scene and build a new mesh list
            for (unsigned int i = 0; i < pScene->mNumMeshes;++i)
            {
                aiMesh* mesh = pScene->mMeshes[i];
                mesh->mNumBones = 0;
                mesh->mBones    = NULL;

                // we're reusing the face index arrays. avoid destruction
                for (unsigned int a = 0; a < mesh->mNumFaces; ++a) {
                    mesh->mFaces[a].mNumIndices = 0;
                    mesh->mFaces[a].mIndices = NULL;
                }

                delete mesh;

                // Invalidate the contents of the old mesh array. We will most
                // likely have less output meshes now, so the last entries of
                // the mesh array are not overridden. We set them to NULL to
                // make sure the developer gets notified when his application
                // attempts to access these fields ...
                mesh = NULL;
            }

            // It is impossible that we have more output meshes than
            // input meshes, so we can easily reuse the old mesh array
            pScene->mNumMeshes = (unsigned int)apcOutMeshes.size();
            for (unsigned int i = 0; i < pScene->mNumMeshes;++i) {
                pScene->mMeshes[i] = apcOutMeshes[i];
            }
        }
    }

    // remove all animations from the scene
    for (unsigned int i = 0; i < pScene->mNumAnimations;++i)
        delete pScene->mAnimations[i];
    delete[] pScene->mAnimations;

    pScene->mAnimations    = NULL;
    pScene->mNumAnimations = 0;

    // --- we need to keep all cameras and lights
    for (unsigned int i = 0; i < pScene->mNumCameras;++i)
    {
        aiCamera* cam = pScene->mCameras[i];
        const aiNode* nd = pScene->mRootNode->FindNode(cam->mName);
        ai_assert(NULL != nd);

        // multiply all properties of the camera with the absolute
        // transformation of the corresponding node
        cam->mPosition = nd->mTransformation * cam->mPosition;
        cam->mLookAt   = aiMatrix3x3( nd->mTransformation ) * cam->mLookAt;
        cam->mUp       = aiMatrix3x3( nd->mTransformation ) * cam->mUp;
    }

    for (unsigned int i = 0; i < pScene->mNumLights;++i)
    {
        aiLight* l = pScene->mLights[i];
        const aiNode* nd = pScene->mRootNode->FindNode(l->mName);
        ai_assert(NULL != nd);

        // multiply all properties of the camera with the absolute
        // transformation of the corresponding node
        l->mPosition   = nd->mTransformation * l->mPosition;
        l->mDirection  = aiMatrix3x3( nd->mTransformation ) * l->mDirection;
        l->mUp         = aiMatrix3x3( nd->mTransformation ) * l->mUp;
    }

    if( !configKeepHierarchy ) {

        // now delete all nodes in the scene and build a new
        // flat node graph with a root node and some level 1 children
        aiNode* newRoot = new aiNode();
        newRoot->mName = pScene->mRootNode->mName;
        delete pScene->mRootNode;
        pScene->mRootNode = newRoot;

        if (1 == pScene->mNumMeshes && !pScene->mNumLights && !pScene->mNumCameras)
        {
            pScene->mRootNode->mNumMeshes = 1;
            pScene->mRootNode->mMeshes = new unsigned int[1];
            pScene->mRootNode->mMeshes[0] = 0;
        }
        else
        {
            pScene->mRootNode->mNumChildren = pScene->mNumMeshes+pScene->mNumLights+pScene->mNumCameras;
            aiNode** nodes = pScene->mRootNode->mChildren = new aiNode*[pScene->mRootNode->mNumChildren];

            // generate mesh nodes
            for (unsigned int i = 0; i < pScene->mNumMeshes;++i,++nodes)
            {
                aiNode* pcNode = new aiNode();
                *nodes = pcNode;
                pcNode->mParent = pScene->mRootNode;
                pcNode->mName = pScene->mMeshes[i]->mName;

                // setup mesh indices
                pcNode->mNumMeshes = 1;
                pcNode->mMeshes = new unsigned int[1];
                pcNode->mMeshes[0] = i;
            }
            // generate light nodes
            for (unsigned int i = 0; i < pScene->mNumLights;++i,++nodes)
            {
                aiNode* pcNode = new aiNode();
                *nodes = pcNode;
                pcNode->mParent = pScene->mRootNode;
                pcNode->mName.length = ai_snprintf(pcNode->mName.data, MAXLEN, "light_%u",i);
                pScene->mLights[i]->mName = pcNode->mName;
            }
            // generate camera nodes
            for (unsigned int i = 0; i < pScene->mNumCameras;++i,++nodes)
            {
                aiNode* pcNode = new aiNode();
                *nodes = pcNode;
                pcNode->mParent = pScene->mRootNode;
                pcNode->mName.length = ::ai_snprintf(pcNode->mName.data,MAXLEN,"cam_%u",i);
                pScene->mCameras[i]->mName = pcNode->mName;
            }
        }
    }
    else {
        // ... and finally set the transformation matrix of all nodes to identity
        MakeIdentityTransform(pScene->mRootNode);
    }

    if (configNormalize) {
        // compute the boundary of all meshes
        aiVector3D min,max;
        MinMaxChooser<aiVector3D> ()(min,max);

        for (unsigned int a = 0; a <  pScene->mNumMeshes; ++a) {
            aiMesh* m = pScene->mMeshes[a];
            for (unsigned int i = 0; i < m->mNumVertices;++i) {
                min = std::min(m->mVertices[i],min);
                max = std::max(m->mVertices[i],max);
            }
        }

        // find the dominant axis
        aiVector3D d = max-min;
        const ai_real div = std::max(d.x,std::max(d.y,d.z))*ai_real( 0.5);

        d = min + d * (ai_real)0.5;
        for (unsigned int a = 0; a <  pScene->mNumMeshes; ++a) {
            aiMesh* m = pScene->mMeshes[a];
            for (unsigned int i = 0; i < m->mNumVertices;++i) {
                m->mVertices[i] = (m->mVertices[i]-d)/div;
            }
        }
    }

    // print statistics
    if (!DefaultLogger::isNullLogger()) {
        ASSIMP_LOG_DEBUG("PretransformVerticesProcess finished");

        ASSIMP_LOG_INFO_F("Removed ", iOldNodes, " nodes and ", iOldAnimationChannels, " animation channels (", 
            CountNodes(pScene->mRootNode) ," output nodes)" );
        ASSIMP_LOG_INFO_F("Kept ", pScene->mNumLights, " lights and ", pScene->mNumCameras, " cameras." );
        ASSIMP_LOG_INFO_F("Moved ", iOldMeshes, " meshes to WCS (number of output meshes: ", pScene->mNumMeshes, ")");
    }
}
