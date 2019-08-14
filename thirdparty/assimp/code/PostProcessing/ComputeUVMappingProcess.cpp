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

/** @file GenUVCoords step */


#include "ComputeUVMappingProcess.h"
#include "ProcessHelper.h"
#include <assimp/Exceptional.h>

using namespace Assimp;

namespace {

    const static aiVector3D base_axis_y(0.0,1.0,0.0);
    const static aiVector3D base_axis_x(1.0,0.0,0.0);
    const static aiVector3D base_axis_z(0.0,0.0,1.0);
    const static ai_real angle_epsilon = ai_real( 0.95 );
}

// ------------------------------------------------------------------------------------------------
// Constructor to be privately used by Importer
ComputeUVMappingProcess::ComputeUVMappingProcess()
{
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Destructor, private as well
ComputeUVMappingProcess::~ComputeUVMappingProcess()
{
    // nothing to do here
}

// ------------------------------------------------------------------------------------------------
// Returns whether the processing step is present in the given flag field.
bool ComputeUVMappingProcess::IsActive( unsigned int pFlags) const
{
    return  (pFlags & aiProcess_GenUVCoords) != 0;
}

// ------------------------------------------------------------------------------------------------
// Check whether a ray intersects a plane and find the intersection point
inline bool PlaneIntersect(const aiRay& ray, const aiVector3D& planePos,
    const aiVector3D& planeNormal, aiVector3D& pos)
{
    const ai_real b = planeNormal * (planePos - ray.pos);
    ai_real h = ray.dir * planeNormal;
    if ((h < 10e-5 && h > -10e-5) || (h = b/h) < 0)
        return false;

    pos = ray.pos + (ray.dir * h);
    return true;
}

// ------------------------------------------------------------------------------------------------
// Find the first empty UV channel in a mesh
inline unsigned int FindEmptyUVChannel (aiMesh* mesh)
{
    for (unsigned int m = 0; m < AI_MAX_NUMBER_OF_TEXTURECOORDS;++m)
        if (!mesh->mTextureCoords[m])return m;

    ASSIMP_LOG_ERROR("Unable to compute UV coordinates, no free UV slot found");
    return UINT_MAX;
}

// ------------------------------------------------------------------------------------------------
// Try to remove UV seams
void RemoveUVSeams (aiMesh* mesh, aiVector3D* out)
{
    // TODO: just a very rough algorithm. I think it could be done
    // much easier, but I don't know how and am currently too tired to
    // to think about a better solution.

    const static ai_real LOWER_LIMIT = ai_real( 0.1 );
    const static ai_real UPPER_LIMIT = ai_real( 0.9 );

    const static ai_real LOWER_EPSILON = ai_real( 10e-3 );
    const static ai_real UPPER_EPSILON = ai_real( 1.0-10e-3 );

    for (unsigned int fidx = 0; fidx < mesh->mNumFaces;++fidx)
    {
        const aiFace& face = mesh->mFaces[fidx];
        if (face.mNumIndices < 3) continue; // triangles and polygons only, please

        unsigned int small = face.mNumIndices, large = small;
        bool zero = false, one = false, round_to_zero = false;

        // Check whether this face lies on a UV seam. We can just guess,
        // but the assumption that a face with at least one very small
        // on the one side and one very large U coord on the other side
        // lies on a UV seam should work for most cases.
        for (unsigned int n = 0; n < face.mNumIndices;++n)
        {
            if (out[face.mIndices[n]].x < LOWER_LIMIT)
            {
                small = n;

                // If we have a U value very close to 0 we can't
                // round the others to 0, too.
                if (out[face.mIndices[n]].x <= LOWER_EPSILON)
                    zero = true;
                else round_to_zero = true;
            }
            if (out[face.mIndices[n]].x > UPPER_LIMIT)
            {
                large = n;

                // If we have a U value very close to 1 we can't
                // round the others to 1, too.
                if (out[face.mIndices[n]].x >= UPPER_EPSILON)
                    one = true;
            }
        }
        if (small != face.mNumIndices && large != face.mNumIndices)
        {
            for (unsigned int n = 0; n < face.mNumIndices;++n)
            {
                // If the u value is over the upper limit and no other u
                // value of that face is 0, round it to 0
                if (out[face.mIndices[n]].x > UPPER_LIMIT && !zero)
                    out[face.mIndices[n]].x = 0.0;

                // If the u value is below the lower limit and no other u
                // value of that face is 1, round it to 1
                else if (out[face.mIndices[n]].x < LOWER_LIMIT && !one)
                    out[face.mIndices[n]].x = 1.0;

                // The face contains both 0 and 1 as UV coords. This can occur
                // for faces which have an edge that lies directly on the seam.
                // Due to numerical inaccuracies one U coord becomes 0, the
                // other 1. But we do still have a third UV coord to determine
                // to which side we must round to.
                else if (one && zero)
                {
                    if (round_to_zero && out[face.mIndices[n]].x >=  UPPER_EPSILON)
                        out[face.mIndices[n]].x = 0.0;
                    else if (!round_to_zero && out[face.mIndices[n]].x <= LOWER_EPSILON)
                        out[face.mIndices[n]].x = 1.0;
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
void ComputeUVMappingProcess::ComputeSphereMapping(aiMesh* mesh,const aiVector3D& axis, aiVector3D* out)
{
    aiVector3D center, min, max;
    FindMeshCenter(mesh, center, min, max);

    // If the axis is one of x,y,z run a faster code path. It's worth the extra effort ...
    // currently the mapping axis will always be one of x,y,z, except if the
    // PretransformVertices step is used (it transforms the meshes into worldspace,
    // thus changing the mapping axis)
    if (axis * base_axis_x >= angle_epsilon)    {

        // For each point get a normalized projection vector in the sphere,
        // get its longitude and latitude and map them to their respective
        // UV axes. Problems occur around the poles ... unsolvable.
        //
        // The spherical coordinate system looks like this:
        // x = cos(lon)*cos(lat)
        // y = sin(lon)*cos(lat)
        // z = sin(lat)
        //
        // Thus we can derive:
        // lat  = arcsin (z)
        // lon  = arctan (y/x)
        for (unsigned int pnt = 0; pnt < mesh->mNumVertices;++pnt)  {
            const aiVector3D diff = (mesh->mVertices[pnt]-center).Normalize();
            out[pnt] = aiVector3D((std::atan2(diff.z, diff.y) + AI_MATH_PI_F ) / AI_MATH_TWO_PI_F,
                (std::asin  (diff.x) + AI_MATH_HALF_PI_F) / AI_MATH_PI_F, 0.0);
        }
    }
    else if (axis * base_axis_y >= angle_epsilon)   {
        // ... just the same again
        for (unsigned int pnt = 0; pnt < mesh->mNumVertices;++pnt)  {
            const aiVector3D diff = (mesh->mVertices[pnt]-center).Normalize();
            out[pnt] = aiVector3D((std::atan2(diff.x, diff.z) + AI_MATH_PI_F ) / AI_MATH_TWO_PI_F,
                (std::asin  (diff.y) + AI_MATH_HALF_PI_F) / AI_MATH_PI_F, 0.0);
        }
    }
    else if (axis * base_axis_z >= angle_epsilon)   {
        // ... just the same again
        for (unsigned int pnt = 0; pnt < mesh->mNumVertices;++pnt)  {
            const aiVector3D diff = (mesh->mVertices[pnt]-center).Normalize();
            out[pnt] = aiVector3D((std::atan2(diff.y, diff.x) + AI_MATH_PI_F ) / AI_MATH_TWO_PI_F,
                (std::asin  (diff.z) + AI_MATH_HALF_PI_F) / AI_MATH_PI_F, 0.0);
        }
    }
    // slower code path in case the mapping axis is not one of the coordinate system axes
    else    {
        aiMatrix4x4 mTrafo;
        aiMatrix4x4::FromToMatrix(axis,base_axis_y,mTrafo);

        // again the same, except we're applying a transformation now
        for (unsigned int pnt = 0; pnt < mesh->mNumVertices;++pnt)  {
            const aiVector3D diff = ((mTrafo*mesh->mVertices[pnt])-center).Normalize();
            out[pnt] = aiVector3D((std::atan2(diff.y, diff.x) + AI_MATH_PI_F ) / AI_MATH_TWO_PI_F,
                (std::asin(diff.z) + AI_MATH_HALF_PI_F) / AI_MATH_PI_F, 0.0);
        }
    }


    // Now find and remove UV seams. A seam occurs if a face has a tcoord
    // close to zero on the one side, and a tcoord close to one on the
    // other side.
    RemoveUVSeams(mesh,out);
}

// ------------------------------------------------------------------------------------------------
void ComputeUVMappingProcess::ComputeCylinderMapping(aiMesh* mesh,const aiVector3D& axis, aiVector3D* out)
{
    aiVector3D center, min, max;

    // If the axis is one of x,y,z run a faster code path. It's worth the extra effort ...
    // currently the mapping axis will always be one of x,y,z, except if the
    // PretransformVertices step is used (it transforms the meshes into worldspace,
    // thus changing the mapping axis)
    if (axis * base_axis_x >= angle_epsilon)    {
        FindMeshCenter(mesh, center, min, max);
        const ai_real diff = max.x - min.x;

        // If the main axis is 'z', the z coordinate of a point 'p' is mapped
        // directly to the texture V axis. The other axis is derived from
        // the angle between ( p.x - c.x, p.y - c.y ) and (1,0), where
        // 'c' is the center point of the mesh.
        for (unsigned int pnt = 0; pnt < mesh->mNumVertices;++pnt)  {
            const aiVector3D& pos = mesh->mVertices[pnt];
            aiVector3D& uv  = out[pnt];

            uv.y = (pos.x - min.x) / diff;
            uv.x = (std::atan2( pos.z - center.z, pos.y - center.y) +(ai_real)AI_MATH_PI ) / (ai_real)AI_MATH_TWO_PI;
        }
    }
    else if (axis * base_axis_y >= angle_epsilon)   {
        FindMeshCenter(mesh, center, min, max);
        const ai_real diff = max.y - min.y;

        // just the same ...
        for (unsigned int pnt = 0; pnt < mesh->mNumVertices;++pnt)  {
            const aiVector3D& pos = mesh->mVertices[pnt];
            aiVector3D& uv  = out[pnt];

            uv.y = (pos.y - min.y) / diff;
            uv.x = (std::atan2( pos.x - center.x, pos.z - center.z) +(ai_real)AI_MATH_PI ) / (ai_real)AI_MATH_TWO_PI;
        }
    }
    else if (axis * base_axis_z >= angle_epsilon)   {
        FindMeshCenter(mesh, center, min, max);
        const ai_real diff = max.z - min.z;

        // just the same ...
        for (unsigned int pnt = 0; pnt < mesh->mNumVertices;++pnt)  {
            const aiVector3D& pos = mesh->mVertices[pnt];
            aiVector3D& uv  = out[pnt];

            uv.y = (pos.z - min.z) / diff;
            uv.x = (std::atan2( pos.y - center.y, pos.x - center.x) +(ai_real)AI_MATH_PI ) / (ai_real)AI_MATH_TWO_PI;
        }
    }
    // slower code path in case the mapping axis is not one of the coordinate system axes
    else {
        aiMatrix4x4 mTrafo;
        aiMatrix4x4::FromToMatrix(axis,base_axis_y,mTrafo);
        FindMeshCenterTransformed(mesh, center, min, max,mTrafo);
        const ai_real diff = max.y - min.y;

        // again the same, except we're applying a transformation now
        for (unsigned int pnt = 0; pnt < mesh->mNumVertices;++pnt){
            const aiVector3D pos = mTrafo* mesh->mVertices[pnt];
            aiVector3D& uv  = out[pnt];

            uv.y = (pos.y - min.y) / diff;
            uv.x = (std::atan2( pos.x - center.x, pos.z - center.z) +(ai_real)AI_MATH_PI ) / (ai_real)AI_MATH_TWO_PI;
        }
    }

    // Now find and remove UV seams. A seam occurs if a face has a tcoord
    // close to zero on the one side, and a tcoord close to one on the
    // other side.
    RemoveUVSeams(mesh,out);
}

// ------------------------------------------------------------------------------------------------
void ComputeUVMappingProcess::ComputePlaneMapping(aiMesh* mesh,const aiVector3D& axis, aiVector3D* out)
{
    ai_real diffu,diffv;
    aiVector3D center, min, max;

    // If the axis is one of x,y,z run a faster code path. It's worth the extra effort ...
    // currently the mapping axis will always be one of x,y,z, except if the
    // PretransformVertices step is used (it transforms the meshes into worldspace,
    // thus changing the mapping axis)
    if (axis * base_axis_x >= angle_epsilon)    {
        FindMeshCenter(mesh, center, min, max);
        diffu = max.z - min.z;
        diffv = max.y - min.y;

        for (unsigned int pnt = 0; pnt < mesh->mNumVertices;++pnt)  {
            const aiVector3D& pos = mesh->mVertices[pnt];
            out[pnt].Set((pos.z - min.z) / diffu,(pos.y - min.y) / diffv,0.0);
        }
    }
    else if (axis * base_axis_y >= angle_epsilon)   {
        FindMeshCenter(mesh, center, min, max);
        diffu = max.x - min.x;
        diffv = max.z - min.z;

        for (unsigned int pnt = 0; pnt < mesh->mNumVertices;++pnt)  {
            const aiVector3D& pos = mesh->mVertices[pnt];
            out[pnt].Set((pos.x - min.x) / diffu,(pos.z - min.z) / diffv,0.0);
        }
    }
    else if (axis * base_axis_z >= angle_epsilon)   {
        FindMeshCenter(mesh, center, min, max);
        diffu = max.y - min.y;
        diffv = max.z - min.z;

        for (unsigned int pnt = 0; pnt < mesh->mNumVertices;++pnt)  {
            const aiVector3D& pos = mesh->mVertices[pnt];
            out[pnt].Set((pos.y - min.y) / diffu,(pos.x - min.x) / diffv,0.0);
        }
    }
    // slower code path in case the mapping axis is not one of the coordinate system axes
    else
    {
        aiMatrix4x4 mTrafo;
        aiMatrix4x4::FromToMatrix(axis,base_axis_y,mTrafo);
        FindMeshCenterTransformed(mesh, center, min, max,mTrafo);
        diffu = max.x - min.x;
        diffv = max.z - min.z;

        // again the same, except we're applying a transformation now
        for (unsigned int pnt = 0; pnt < mesh->mNumVertices;++pnt)  {
            const aiVector3D pos = mTrafo * mesh->mVertices[pnt];
            out[pnt].Set((pos.x - min.x) / diffu,(pos.z - min.z) / diffv,0.0);
        }
    }

    // shouldn't be necessary to remove UV seams ...
}

// ------------------------------------------------------------------------------------------------
void ComputeUVMappingProcess::ComputeBoxMapping( aiMesh*, aiVector3D* )
{
    ASSIMP_LOG_ERROR("Mapping type currently not implemented");
}

// ------------------------------------------------------------------------------------------------
void ComputeUVMappingProcess::Execute( aiScene* pScene)
{
    ASSIMP_LOG_DEBUG("GenUVCoordsProcess begin");
    char buffer[1024];

    if (pScene->mFlags & AI_SCENE_FLAGS_NON_VERBOSE_FORMAT)
        throw DeadlyImportError("Post-processing order mismatch: expecting pseudo-indexed (\"verbose\") vertices here");

    std::list<MappingInfo> mappingStack;

    /*  Iterate through all materials and search for non-UV mapped textures
     */
    for (unsigned int i = 0; i < pScene->mNumMaterials;++i)
    {
        mappingStack.clear();
        aiMaterial* mat = pScene->mMaterials[i];
        for (unsigned int a = 0; a < mat->mNumProperties;++a)
        {
            aiMaterialProperty* prop = mat->mProperties[a];
            if (!::strcmp( prop->mKey.data, "$tex.mapping"))
            {
                aiTextureMapping& mapping = *((aiTextureMapping*)prop->mData);
                if (aiTextureMapping_UV != mapping)
                {
                    if (!DefaultLogger::isNullLogger())
                    {
                        ai_snprintf(buffer, 1024, "Found non-UV mapped texture (%s,%u). Mapping type: %s",
                            TextureTypeToString((aiTextureType)prop->mSemantic),prop->mIndex,
                            MappingTypeToString(mapping));

                        ASSIMP_LOG_INFO(buffer);
                    }

                    if (aiTextureMapping_OTHER == mapping)
                        continue;

                    MappingInfo info (mapping);

                    // Get further properties - currently only the major axis
                    for (unsigned int a2 = 0; a2 < mat->mNumProperties;++a2)
                    {
                        aiMaterialProperty* prop2 = mat->mProperties[a2];
                        if (prop2->mSemantic != prop->mSemantic || prop2->mIndex != prop->mIndex)
                            continue;

                        if ( !::strcmp( prop2->mKey.data, "$tex.mapaxis"))  {
                            info.axis = *((aiVector3D*)prop2->mData);
                            break;
                        }
                    }

                    unsigned int idx( 99999999 );

                    // Check whether we have this mapping mode already
                    std::list<MappingInfo>::iterator it = std::find (mappingStack.begin(),mappingStack.end(), info);
                    if (mappingStack.end() != it)
                    {
                        idx = (*it).uv;
                    }
                    else
                    {
                        /*  We have found a non-UV mapped texture. Now
                        *   we need to find all meshes using this material
                        *   that we can compute UV channels for them.
                        */
                        for (unsigned int m = 0; m < pScene->mNumMeshes;++m)
                        {
                            aiMesh* mesh = pScene->mMeshes[m];
                            unsigned int outIdx = 0;
                            if ( mesh->mMaterialIndex != i || ( outIdx = FindEmptyUVChannel(mesh) ) == UINT_MAX ||
                                !mesh->mNumVertices)
                            {
                                continue;
                            }

                            // Allocate output storage
                            aiVector3D* p = mesh->mTextureCoords[outIdx] = new aiVector3D[mesh->mNumVertices];

                            switch (mapping)
                            {
                            case aiTextureMapping_SPHERE:
                                ComputeSphereMapping(mesh,info.axis,p);
                                break;
                            case aiTextureMapping_CYLINDER:
                                ComputeCylinderMapping(mesh,info.axis,p);
                                break;
                            case aiTextureMapping_PLANE:
                                ComputePlaneMapping(mesh,info.axis,p);
                                break;
                            case aiTextureMapping_BOX:
                                ComputeBoxMapping(mesh,p);
                                break;
                            default:
                                ai_assert(false);
                            }
                            if (m && idx != outIdx)
                            {
                                ASSIMP_LOG_WARN("UV index mismatch. Not all meshes assigned to "
                                    "this material have equal numbers of UV channels. The UV index stored in  "
                                    "the material structure does therefore not apply for all meshes. ");
                            }
                            idx = outIdx;
                        }
                        info.uv = idx;
                        mappingStack.push_back(info);
                    }

                    // Update the material property list
                    mapping = aiTextureMapping_UV;
                    ((aiMaterial*)mat)->AddProperty(&idx,1,AI_MATKEY_UVWSRC(prop->mSemantic,prop->mIndex));
                }
            }
        }
    }
    ASSIMP_LOG_DEBUG("GenUVCoordsProcess finished");
}
