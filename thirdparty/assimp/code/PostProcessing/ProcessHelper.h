/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2020, assimp team


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

#ifndef AI_PROCESS_HELPER_H_INCLUDED
#define AI_PROCESS_HELPER_H_INCLUDED

#include <assimp/postprocess.h>
#include <assimp/anim.h>
#include <assimp/mesh.h>
#include <assimp/material.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/scene.h>

#include <assimp/SpatialSort.h>
#include "Common/BaseProcess.h"
#include <assimp/ParsingUtils.h>

#include <list>

// -------------------------------------------------------------------------------
// Some extensions to std namespace. Mainly std::min and std::max for all
// flat data types in the aiScene. They're used to quickly determine the
// min/max bounds of data arrays.
#ifdef __cplusplus
namespace std {

    // std::min for aiVector3D
    template <typename TReal>
    inline ::aiVector3t<TReal> min (const ::aiVector3t<TReal>& a, const ::aiVector3t<TReal>& b) {
        return ::aiVector3t<TReal> (min(a.x,b.x),min(a.y,b.y),min(a.z,b.z));
    }

    // std::max for aiVector3t<TReal>
    template <typename TReal>
    inline ::aiVector3t<TReal> max (const ::aiVector3t<TReal>& a, const ::aiVector3t<TReal>& b) {
        return ::aiVector3t<TReal> (max(a.x,b.x),max(a.y,b.y),max(a.z,b.z));
    }

    // std::min for aiVector2t<TReal>
    template <typename TReal>
    inline ::aiVector2t<TReal> min (const ::aiVector2t<TReal>& a, const ::aiVector2t<TReal>& b) {
        return ::aiVector2t<TReal> (min(a.x,b.x),min(a.y,b.y));
    }

    // std::max for aiVector2t<TReal>
    template <typename TReal>
    inline ::aiVector2t<TReal> max (const ::aiVector2t<TReal>& a, const ::aiVector2t<TReal>& b) {
        return ::aiVector2t<TReal> (max(a.x,b.x),max(a.y,b.y));
    }

    // std::min for aiColor4D
    template <typename TReal>
    inline ::aiColor4t<TReal> min (const ::aiColor4t<TReal>& a, const ::aiColor4t<TReal>& b)    {
        return ::aiColor4t<TReal> (min(a.r,b.r),min(a.g,b.g),min(a.b,b.b),min(a.a,b.a));
    }

    // std::max for aiColor4D
    template <typename TReal>
    inline ::aiColor4t<TReal> max (const ::aiColor4t<TReal>& a, const ::aiColor4t<TReal>& b)    {
        return ::aiColor4t<TReal> (max(a.r,b.r),max(a.g,b.g),max(a.b,b.b),max(a.a,b.a));
    }


    // std::min for aiQuaterniont<TReal>
    template <typename TReal>
    inline ::aiQuaterniont<TReal> min (const ::aiQuaterniont<TReal>& a, const ::aiQuaterniont<TReal>& b)    {
        return ::aiQuaterniont<TReal> (min(a.w,b.w),min(a.x,b.x),min(a.y,b.y),min(a.z,b.z));
    }

    // std::max for aiQuaterniont<TReal>
    template <typename TReal>
    inline ::aiQuaterniont<TReal> max (const ::aiQuaterniont<TReal>& a, const ::aiQuaterniont<TReal>& b)    {
        return ::aiQuaterniont<TReal> (max(a.w,b.w),max(a.x,b.x),max(a.y,b.y),max(a.z,b.z));
    }



    // std::min for aiVectorKey
    inline ::aiVectorKey min (const ::aiVectorKey& a, const ::aiVectorKey& b)   {
        return ::aiVectorKey (min(a.mTime,b.mTime),min(a.mValue,b.mValue));
    }

    // std::max for aiVectorKey
    inline ::aiVectorKey max (const ::aiVectorKey& a, const ::aiVectorKey& b)   {
        return ::aiVectorKey (max(a.mTime,b.mTime),max(a.mValue,b.mValue));
    }

    // std::min for aiQuatKey
    inline ::aiQuatKey min (const ::aiQuatKey& a, const ::aiQuatKey& b) {
        return ::aiQuatKey (min(a.mTime,b.mTime),min(a.mValue,b.mValue));
    }

    // std::max for aiQuatKey
    inline ::aiQuatKey max (const ::aiQuatKey& a, const ::aiQuatKey& b) {
        return ::aiQuatKey (max(a.mTime,b.mTime),max(a.mValue,b.mValue));
    }

    // std::min for aiVertexWeight
    inline ::aiVertexWeight min (const ::aiVertexWeight& a, const ::aiVertexWeight& b)  {
        return ::aiVertexWeight (min(a.mVertexId,b.mVertexId),min(a.mWeight,b.mWeight));
    }

    // std::max for aiVertexWeight
    inline ::aiVertexWeight max (const ::aiVertexWeight& a, const ::aiVertexWeight& b)  {
        return ::aiVertexWeight (max(a.mVertexId,b.mVertexId),max(a.mWeight,b.mWeight));
    }

} // end namespace std
#endif // !! C++

namespace Assimp {

// -------------------------------------------------------------------------------
// Start points for ArrayBounds<T> for all supported Ts
template <typename T>
struct MinMaxChooser;

template <> struct MinMaxChooser<float> {
    void operator ()(float& min,float& max) {
        max = -1e10f;
        min =  1e10f;
}};
template <> struct MinMaxChooser<double> {
    void operator ()(double& min,double& max) {
        max = -1e10;
        min =  1e10;
}};
template <> struct MinMaxChooser<unsigned int> {
    void operator ()(unsigned int& min,unsigned int& max) {
        max = 0;
        min = (1u<<(sizeof(unsigned int)*8-1));
}};

template <typename T> struct MinMaxChooser< aiVector3t<T> > {
    void operator ()(aiVector3t<T>& min,aiVector3t<T>& max) {
        max = aiVector3t<T>(-1e10f,-1e10f,-1e10f);
        min = aiVector3t<T>( 1e10f, 1e10f, 1e10f);
}};
template <typename T> struct MinMaxChooser< aiVector2t<T> > {
    void operator ()(aiVector2t<T>& min,aiVector2t<T>& max) {
        max = aiVector2t<T>(-1e10f,-1e10f);
        min = aiVector2t<T>( 1e10f, 1e10f);
    }};
template <typename T> struct MinMaxChooser< aiColor4t<T> > {
    void operator ()(aiColor4t<T>& min,aiColor4t<T>& max) {
        max = aiColor4t<T>(-1e10f,-1e10f,-1e10f,-1e10f);
        min = aiColor4t<T>( 1e10f, 1e10f, 1e10f, 1e10f);
}};

template <typename T> struct MinMaxChooser< aiQuaterniont<T> > {
    void operator ()(aiQuaterniont<T>& min,aiQuaterniont<T>& max) {
        max = aiQuaterniont<T>(-1e10f,-1e10f,-1e10f,-1e10f);
        min = aiQuaterniont<T>( 1e10f, 1e10f, 1e10f, 1e10f);
}};

template <> struct MinMaxChooser<aiVectorKey> {
    void operator ()(aiVectorKey& min,aiVectorKey& max) {
        MinMaxChooser<double>()(min.mTime,max.mTime);
        MinMaxChooser<aiVector3D>()(min.mValue,max.mValue);
}};
template <> struct MinMaxChooser<aiQuatKey> {
    void operator ()(aiQuatKey& min,aiQuatKey& max) {
        MinMaxChooser<double>()(min.mTime,max.mTime);
        MinMaxChooser<aiQuaternion>()(min.mValue,max.mValue);
}};

template <> struct MinMaxChooser<aiVertexWeight> {
    void operator ()(aiVertexWeight& min,aiVertexWeight& max) {
        MinMaxChooser<unsigned int>()(min.mVertexId,max.mVertexId);
        MinMaxChooser<float>()(min.mWeight,max.mWeight);
}};

// -------------------------------------------------------------------------------
/** @brief Find the min/max values of an array of Ts
 *  @param in Input array
 *  @param size Number of elements to process
 *  @param[out] min minimum value
 *  @param[out] max maximum value
 */
template <typename T>
inline void ArrayBounds(const T* in, unsigned int size, T& min, T& max)
{
    MinMaxChooser<T> ()(min,max);
    for (unsigned int i = 0; i < size;++i) {
        min = std::min(in[i],min);
        max = std::max(in[i],max);
    }
}


// -------------------------------------------------------------------------------
/** Little helper function to calculate the quadratic difference
 * of two colours.
 * @param pColor1 First color
 * @param pColor2 second color
 * @return Quadratic color difference */
inline ai_real GetColorDifference( const aiColor4D& pColor1, const aiColor4D& pColor2)
{
    const aiColor4D c (pColor1.r - pColor2.r, pColor1.g - pColor2.g, pColor1.b - pColor2.b, pColor1.a - pColor2.a);
    return c.r*c.r + c.g*c.g + c.b*c.b + c.a*c.a;
}


// -------------------------------------------------------------------------------
/** @brief Extract single strings from a list of identifiers
 *  @param in Input string list.
 *  @param out Receives a list of clean output strings
 *  @sdee #AI_CONFIG_PP_OG_EXCLUDE_LIST */
void ConvertListToStrings(const std::string& in, std::list<std::string>& out);


// -------------------------------------------------------------------------------
/** @brief Compute the AABB of a mesh after applying a given transform
 *  @param mesh Input mesh
 *  @param[out] min Receives minimum transformed vertex
 *  @param[out] max Receives maximum transformed vertex
 *  @param m Transformation matrix to be applied */
void FindAABBTransformed (const aiMesh* mesh, aiVector3D& min, aiVector3D& max, const aiMatrix4x4& m);


// -------------------------------------------------------------------------------
/** @brief Helper function to determine the 'real' center of a mesh
 *
 *  That is the center of its axis-aligned bounding box.
 *  @param mesh Input mesh
 *  @param[out] min Minimum vertex of the mesh
 *  @param[out] max maximum vertex of the mesh
 *  @param[out] out Center point */
void FindMeshCenter (aiMesh* mesh, aiVector3D& out, aiVector3D& min, aiVector3D& max);

// -------------------------------------------------------------------------------
/** @brief Helper function to determine the 'real' center of a scene
 *
 *  That is the center of its axis-aligned bounding box.
 *  @param scene Input scene
 *  @param[out] min Minimum vertex of the scene
 *  @param[out] max maximum vertex of the scene
 *  @param[out] out Center point */
void FindSceneCenter (aiScene* scene, aiVector3D& out, aiVector3D& min, aiVector3D& max);


// -------------------------------------------------------------------------------
// Helper function to determine the 'real' center of a mesh after applying a given transform
void FindMeshCenterTransformed (aiMesh* mesh, aiVector3D& out, aiVector3D& min,aiVector3D& max, const aiMatrix4x4& m);


// -------------------------------------------------------------------------------
// Helper function to determine the 'real' center of a mesh
void FindMeshCenter (aiMesh* mesh, aiVector3D& out);


// -------------------------------------------------------------------------------
// Helper function to determine the 'real' center of a mesh after applying a given transform
void FindMeshCenterTransformed (aiMesh* mesh, aiVector3D& out,const aiMatrix4x4& m);


// -------------------------------------------------------------------------------
// Compute a good epsilon value for position comparisons on a mesh
ai_real ComputePositionEpsilon(const aiMesh* pMesh);


// -------------------------------------------------------------------------------
// Compute a good epsilon value for position comparisons on a array of meshes
ai_real ComputePositionEpsilon(const aiMesh* const* pMeshes, size_t num);


// -------------------------------------------------------------------------------
// Compute an unique value for the vertex format of a mesh
unsigned int GetMeshVFormatUnique(const aiMesh* pcMesh);


// defs for ComputeVertexBoneWeightTable()
typedef std::pair <unsigned int,float> PerVertexWeight;
typedef std::vector <PerVertexWeight> VertexWeightTable;

// -------------------------------------------------------------------------------
// Compute a per-vertex bone weight table
VertexWeightTable* ComputeVertexBoneWeightTable(const aiMesh* pMesh);

// -------------------------------------------------------------------------------
// Get a string for a given aiTextureMapping
const char* MappingTypeToString(aiTextureMapping in);


// flags for MakeSubmesh()
#define AI_SUBMESH_FLAGS_SANS_BONES 0x1

// -------------------------------------------------------------------------------
// Split a mesh given a list of faces to be contained in the sub mesh
aiMesh* MakeSubmesh(const aiMesh *superMesh, const std::vector<unsigned int> &subMeshFaces, unsigned int subFlags);

// -------------------------------------------------------------------------------
// Utility postprocess step to share the spatial sort tree between
// all steps which use it to speedup its computations.
class ComputeSpatialSortProcess : public BaseProcess
{
    bool IsActive( unsigned int pFlags) const
    {
        return NULL != shared && 0 != (pFlags & (aiProcess_CalcTangentSpace |
            aiProcess_GenNormals | aiProcess_JoinIdenticalVertices));
    }

    void Execute( aiScene* pScene)
    {
        typedef std::pair<SpatialSort, ai_real> _Type;
        ASSIMP_LOG_DEBUG("Generate spatially-sorted vertex cache");

        std::vector<_Type>* p = new std::vector<_Type>(pScene->mNumMeshes);
        std::vector<_Type>::iterator it = p->begin();

        for (unsigned int i = 0; i < pScene->mNumMeshes; ++i, ++it) {
            aiMesh* mesh = pScene->mMeshes[i];
            _Type& blubb = *it;
            blubb.first.Fill(mesh->mVertices,mesh->mNumVertices,sizeof(aiVector3D));
            blubb.second = ComputePositionEpsilon(mesh);
        }

        shared->AddProperty(AI_SPP_SPATIAL_SORT,p);
    }
};

// -------------------------------------------------------------------------------
// ... and the same again to cleanup the whole stuff
class DestroySpatialSortProcess : public BaseProcess
{
    bool IsActive( unsigned int pFlags) const
    {
        return NULL != shared && 0 != (pFlags & (aiProcess_CalcTangentSpace |
            aiProcess_GenNormals | aiProcess_JoinIdenticalVertices));
    }

    void Execute( aiScene* /*pScene*/)
    {
        shared->RemoveProperty(AI_SPP_SPATIAL_SORT);
    }
};



} // ! namespace Assimp
#endif // !! AI_PROCESS_HELPER_H_INCLUDED
