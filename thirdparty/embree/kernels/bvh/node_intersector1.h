// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "node_intersector.h"

namespace embree
{
  namespace isa
  {
    //////////////////////////////////////////////////////////////////////////////////////
    // Ray structure used in single-ray traversal
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, int Nx, bool robust>
      struct TravRayBase;
      
    /* Base (without tnear and tfar) */
    template<int N, int Nx>
      struct TravRayBase<N,Nx,false>
    {
      __forceinline TravRayBase() {}

      __forceinline TravRayBase(const Vec3fa& ray_org, const Vec3fa& ray_dir)
        : org_xyz(ray_org), dir_xyz(ray_dir) 
      {
        const Vec3fa ray_rdir = rcp_safe(ray_dir);
        org = Vec3vf<N>(ray_org.x,ray_org.y,ray_org.z);
        dir = Vec3vf<N>(ray_dir.x,ray_dir.y,ray_dir.z);
        rdir = Vec3vf<N>(ray_rdir.x,ray_rdir.y,ray_rdir.z);
#if defined(__AVX2__)
        const Vec3fa ray_org_rdir = ray_org*ray_rdir;
        org_rdir = Vec3vf<N>(ray_org_rdir.x,ray_org_rdir.y,ray_org_rdir.z);
#endif
        nearX = ray_rdir.x >= 0.0f ? 0*sizeof(vfloat<N>) : 1*sizeof(vfloat<N>);
        nearY = ray_rdir.y >= 0.0f ? 2*sizeof(vfloat<N>) : 3*sizeof(vfloat<N>);
        nearZ = ray_rdir.z >= 0.0f ? 4*sizeof(vfloat<N>) : 5*sizeof(vfloat<N>);
        farX  = nearX ^ sizeof(vfloat<N>);
        farY  = nearY ^ sizeof(vfloat<N>);
        farZ  = nearZ ^ sizeof(vfloat<N>);

#if defined(__AVX512ER__) // KNL+
        /* optimization works only for 8-wide BVHs with 16-wide SIMD */
        const vint<16> id(step);
        const vint<16> id2 = align_shift_right<16/2>(id, id);
        permX = select(vfloat<16>(dir.x) >= 0.0f, id, id2);
        permY = select(vfloat<16>(dir.y) >= 0.0f, id, id2);
        permZ = select(vfloat<16>(dir.z) >= 0.0f, id, id2);
#endif

      }

      template<int K>
      __forceinline TravRayBase(size_t k, const Vec3vf<K>& ray_org, const Vec3vf<K>& ray_dir,
                                const Vec3vf<K>& ray_rdir, const Vec3vi<K>& nearXYZ,
                                size_t flip = sizeof(vfloat<N>))
      {
        org  = Vec3vf<Nx>(ray_org.x[k], ray_org.y[k], ray_org.z[k]);
        dir  = Vec3vf<Nx>(ray_dir.x[k], ray_dir.y[k], ray_dir.z[k]);
        rdir = Vec3vf<Nx>(ray_rdir.x[k], ray_rdir.y[k], ray_rdir.z[k]);
#if defined(__AVX2__)
	org_rdir = org*rdir;
#endif
	nearX = nearXYZ.x[k];
	nearY = nearXYZ.y[k];
	nearZ = nearXYZ.z[k];
        farX  = nearX ^ flip;
        farY  = nearY ^ flip;
        farZ  = nearZ ^ flip;

#if defined(__AVX512ER__) // KNL+
        /* optimization works only for 8-wide BVHs with 16-wide SIMD */
        const vint<16> id(step);
        const vint<16> id2 = align_shift_right<16/2>(id, id);
        permX = select(vfloat<16>(dir.x) >= 0.0f, id, id2);
        permY = select(vfloat<16>(dir.y) >= 0.0f, id, id2);
        permZ = select(vfloat<16>(dir.z) >= 0.0f, id, id2);
#endif
      }

      Vec3fa org_xyz, dir_xyz;
      Vec3vf<Nx> org, dir, rdir;
#if defined(__AVX2__)
      Vec3vf<Nx> org_rdir;
#endif
#if defined(__AVX512ER__) // KNL+
      vint16 permX, permY, permZ;
#endif

      size_t nearX, nearY, nearZ;
      size_t farX, farY, farZ;
    };

    /* Base (without tnear and tfar) */
    template<int N, int Nx>
      struct TravRayBase<N,Nx,true>
    {
      __forceinline TravRayBase() {}

      __forceinline TravRayBase(const Vec3fa& ray_org, const Vec3fa& ray_dir)
        : org_xyz(ray_org), dir_xyz(ray_dir) 
      {
        const float ulp3 = 1.0f+3.0f*float(ulp);
        const Vec3fa ray_rdir_near = 1.0f/zero_fix(ray_dir);
        const Vec3fa ray_rdir_far  = ray_rdir_near*ulp3;
        org = Vec3vf<N>(ray_org.x,ray_org.y,ray_org.z);
        dir = Vec3vf<N>(ray_dir.x,ray_dir.y,ray_dir.z);
        rdir_near = Vec3vf<N>(ray_rdir_near.x,ray_rdir_near.y,ray_rdir_near.z);
        rdir_far  = Vec3vf<N>(ray_rdir_far .x,ray_rdir_far .y,ray_rdir_far .z);

        nearX = ray_rdir_near.x >= 0.0f ? 0*sizeof(vfloat<N>) : 1*sizeof(vfloat<N>);
        nearY = ray_rdir_near.y >= 0.0f ? 2*sizeof(vfloat<N>) : 3*sizeof(vfloat<N>);
        nearZ = ray_rdir_near.z >= 0.0f ? 4*sizeof(vfloat<N>) : 5*sizeof(vfloat<N>);
        farX  = nearX ^ sizeof(vfloat<N>);
        farY  = nearY ^ sizeof(vfloat<N>);
        farZ  = nearZ ^ sizeof(vfloat<N>);

#if defined(__AVX512ER__) // KNL+
        /* optimization works only for 8-wide BVHs with 16-wide SIMD */
        const vint<16> id(step);
        const vint<16> id2 = align_shift_right<16/2>(id, id);
        permX = select(vfloat<16>(dir.x) >= 0.0f, id, id2);
        permY = select(vfloat<16>(dir.y) >= 0.0f, id, id2);
        permZ = select(vfloat<16>(dir.z) >= 0.0f, id, id2);
#endif
      }

      template<int K>
      __forceinline TravRayBase(size_t k, const Vec3vf<K>& ray_org, const Vec3vf<K>& ray_dir,
                                const Vec3vf<K>& ray_rdir, const Vec3vi<K>& nearXYZ,
                                size_t flip = sizeof(vfloat<N>))
      {
        const float ulp3 = 1.0f+3.0f*float(ulp);
        org  = Vec3vf<Nx>(ray_org.x[k], ray_org.y[k], ray_org.z[k]);
        dir  = Vec3vf<Nx>(ray_dir.x[k], ray_dir.y[k], ray_dir.z[k]);
        rdir_near = Vec3vf<Nx>(ray_rdir.x[k], ray_rdir.y[k], ray_rdir.z[k]);
        rdir_far  = Vec3vf<Nx>(ray_rdir.x[k]*ulp3, ray_rdir.y[k]*ulp3, ray_rdir.z[k]*ulp3);

	nearX = nearXYZ.x[k];
	nearY = nearXYZ.y[k];
	nearZ = nearXYZ.z[k];
        farX  = nearX ^ flip;
        farY  = nearY ^ flip;
        farZ  = nearZ ^ flip;

#if defined(__AVX512ER__) // KNL+
        /* optimization works only for 8-wide BVHs with 16-wide SIMD */
        const vint<16> id(step);
        const vint<16> id2 = align_shift_right<16/2>(id, id);
        permX = select(vfloat<16>(dir.x) >= 0.0f, id, id2);
        permY = select(vfloat<16>(dir.y) >= 0.0f, id, id2);
        permZ = select(vfloat<16>(dir.z) >= 0.0f, id, id2);
#endif
      }

      Vec3fa org_xyz, dir_xyz;
      Vec3vf<Nx> org, dir, rdir_near, rdir_far;
#if defined(__AVX512ER__) // KNL+
      vint16 permX, permY, permZ;
#endif

      size_t nearX, nearY, nearZ;
      size_t farX, farY, farZ;
    };

    /* Full (with tnear and tfar) */
    template<int N, int Nx, bool robust>
      struct TravRay : TravRayBase<N,Nx,robust>
    {
      __forceinline TravRay() {}

      __forceinline TravRay(const Vec3fa& ray_org, const Vec3fa& ray_dir, float ray_tnear, float ray_tfar)
        : TravRayBase<N,Nx,robust>(ray_org, ray_dir),
          tnear(ray_tnear), tfar(ray_tfar) {}

      template<int K>
      __forceinline TravRay(size_t k, const Vec3vf<K>& ray_org, const Vec3vf<K>& ray_dir,
                            const Vec3vf<K>& ray_rdir, const Vec3vi<K>& nearXYZ,
                            float ray_tnear, float ray_tfar,
                            size_t flip = sizeof(vfloat<N>))
        : TravRayBase<N,Nx,robust>(k, ray_org, ray_dir, ray_rdir, nearXYZ, flip),
          tnear(ray_tnear), tfar(ray_tfar) {}

      vfloat<Nx> tnear;
      vfloat<Nx> tfar;
    };

    //////////////////////////////////////////////////////////////////////////////////////
    // Fast AlignedNode intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, int Nx, bool robust>
      __forceinline size_t intersectNode(const typename BVHN<N>::AlignedNode* node, const TravRay<N,Nx,robust>& ray, vfloat<Nx>& dist);

    template<>
      __forceinline size_t intersectNode<4,4>(const typename BVH4::AlignedNode* node, const TravRay<4,4,false>& ray, vfloat4& dist)
    {
#if defined(__AVX2__)
      const vfloat4 tNearX = msub(vfloat4::load((float*)((const char*)&node->lower_x+ray.nearX)), ray.rdir.x, ray.org_rdir.x);
      const vfloat4 tNearY = msub(vfloat4::load((float*)((const char*)&node->lower_x+ray.nearY)), ray.rdir.y, ray.org_rdir.y);
      const vfloat4 tNearZ = msub(vfloat4::load((float*)((const char*)&node->lower_x+ray.nearZ)), ray.rdir.z, ray.org_rdir.z);
      const vfloat4 tFarX  = msub(vfloat4::load((float*)((const char*)&node->lower_x+ray.farX )), ray.rdir.x, ray.org_rdir.x);
      const vfloat4 tFarY  = msub(vfloat4::load((float*)((const char*)&node->lower_x+ray.farY )), ray.rdir.y, ray.org_rdir.y);
      const vfloat4 tFarZ  = msub(vfloat4::load((float*)((const char*)&node->lower_x+ray.farZ )), ray.rdir.z, ray.org_rdir.z);
#else
      const vfloat4 tNearX = (vfloat4::load((float*)((const char*)&node->lower_x+ray.nearX)) - ray.org.x) * ray.rdir.x;
      const vfloat4 tNearY = (vfloat4::load((float*)((const char*)&node->lower_x+ray.nearY)) - ray.org.y) * ray.rdir.y;
      const vfloat4 tNearZ = (vfloat4::load((float*)((const char*)&node->lower_x+ray.nearZ)) - ray.org.z) * ray.rdir.z;
      const vfloat4 tFarX  = (vfloat4::load((float*)((const char*)&node->lower_x+ray.farX )) - ray.org.x) * ray.rdir.x;
      const vfloat4 tFarY  = (vfloat4::load((float*)((const char*)&node->lower_x+ray.farY )) - ray.org.y) * ray.rdir.y;
      const vfloat4 tFarZ  = (vfloat4::load((float*)((const char*)&node->lower_x+ray.farZ )) - ray.org.z) * ray.rdir.z;
#endif
      
#if defined(__SSE4_1__) && !defined(__AVX512F__) // up to HSW
      const vfloat4 tNear = maxi(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat4 tFar  = mini(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool4 vmask = asInt(tNear) > asInt(tFar);
      const size_t mask = movemask(vmask) ^ ((1<<4)-1);
#elif defined(__AVX512F__) && !defined(__AVX512ER__) // SKX
      const vfloat4 tNear = maxi(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat4 tFar  = mini(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool4 vmask = asInt(tNear) <= asInt(tFar);
      const size_t mask = movemask(vmask);
#else
      const vfloat4 tNear = max(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat4 tFar  = min(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool4 vmask = tNear <= tFar;
      const size_t mask = movemask(vmask);
#endif
      dist = tNear;
      return mask;
    }

#if defined(__AVX__)

    template<>
      __forceinline size_t intersectNode<8,8>(const typename BVH8::AlignedNode* node, const TravRay<8,8,false>& ray, vfloat8& dist)
    {
#if defined(__AVX2__)
      const vfloat8 tNearX = msub(vfloat8::load((float*)((const char*)&node->lower_x+ray.nearX)), ray.rdir.x, ray.org_rdir.x);
      const vfloat8 tNearY = msub(vfloat8::load((float*)((const char*)&node->lower_x+ray.nearY)), ray.rdir.y, ray.org_rdir.y);
      const vfloat8 tNearZ = msub(vfloat8::load((float*)((const char*)&node->lower_x+ray.nearZ)), ray.rdir.z, ray.org_rdir.z);
      const vfloat8 tFarX  = msub(vfloat8::load((float*)((const char*)&node->lower_x+ray.farX )), ray.rdir.x, ray.org_rdir.x);
      const vfloat8 tFarY  = msub(vfloat8::load((float*)((const char*)&node->lower_x+ray.farY )), ray.rdir.y, ray.org_rdir.y);
      const vfloat8 tFarZ  = msub(vfloat8::load((float*)((const char*)&node->lower_x+ray.farZ )), ray.rdir.z, ray.org_rdir.z);
#else
      const vfloat8 tNearX = (vfloat8::load((float*)((const char*)&node->lower_x+ray.nearX)) - ray.org.x) * ray.rdir.x;
      const vfloat8 tNearY = (vfloat8::load((float*)((const char*)&node->lower_x+ray.nearY)) - ray.org.y) * ray.rdir.y;
      const vfloat8 tNearZ = (vfloat8::load((float*)((const char*)&node->lower_x+ray.nearZ)) - ray.org.z) * ray.rdir.z;
      const vfloat8 tFarX  = (vfloat8::load((float*)((const char*)&node->lower_x+ray.farX )) - ray.org.x) * ray.rdir.x;
      const vfloat8 tFarY  = (vfloat8::load((float*)((const char*)&node->lower_x+ray.farY )) - ray.org.y) * ray.rdir.y;
      const vfloat8 tFarZ  = (vfloat8::load((float*)((const char*)&node->lower_x+ray.farZ )) - ray.org.z) * ray.rdir.z;
#endif
      
#if defined(__AVX2__) && !defined(__AVX512F__) // HSW
      const vfloat8 tNear = maxi(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat8 tFar  = mini(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool8 vmask = asInt(tNear) > asInt(tFar);
      const size_t mask = movemask(vmask) ^ ((1<<8)-1);
#elif defined(__AVX512F__) && !defined(__AVX512ER__) // SKX
      const vfloat8 tNear = maxi(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat8 tFar  = mini(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool8 vmask = asInt(tNear) <= asInt(tFar);
      const size_t mask = movemask(vmask);
#else
      const vfloat8 tNear = max(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat8 tFar  = min(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool8 vmask = tNear <= tFar;
      const size_t mask = movemask(vmask);
#endif
      dist = tNear;
      return mask;
    }

#endif

#if defined(__AVX512F__) && !defined(__AVX512VL__) // KNL

    template<>
      __forceinline size_t intersectNode<4,16>(const typename BVH4::AlignedNode* node, const TravRay<4,16,false>& ray, vfloat16& dist)
    {
      const vfloat16 tNearX = msub(vfloat16(*(vfloat4*)((const char*)&node->lower_x+ray.nearX)), ray.rdir.x, ray.org_rdir.x);
      const vfloat16 tNearY = msub(vfloat16(*(vfloat4*)((const char*)&node->lower_x+ray.nearY)), ray.rdir.y, ray.org_rdir.y);
      const vfloat16 tNearZ = msub(vfloat16(*(vfloat4*)((const char*)&node->lower_x+ray.nearZ)), ray.rdir.z, ray.org_rdir.z);
      const vfloat16 tFarX  = msub(vfloat16(*(vfloat4*)((const char*)&node->lower_x+ray.farX )), ray.rdir.x, ray.org_rdir.x);
      const vfloat16 tFarY  = msub(vfloat16(*(vfloat4*)((const char*)&node->lower_x+ray.farY )), ray.rdir.y, ray.org_rdir.y);
      const vfloat16 tFarZ  = msub(vfloat16(*(vfloat4*)((const char*)&node->lower_x+ray.farZ )), ray.rdir.z, ray.org_rdir.z);      
      const vfloat16 tNear  = max(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat16 tFar   = min(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool16 vmask   = le(vbool16(0xf),tNear,tFar);
      const size_t mask     = movemask(vmask);
      dist = tNear;
      return mask;
    }

    template<>
      __forceinline size_t intersectNode<8,16>(const typename BVH8::AlignedNode* node, const TravRay<8,16,false>& ray, vfloat16& dist)
    {
      const vllong8 invalid((size_t)BVH8::emptyNode);
      const vboold8 m_valid(invalid != vllong8::loadu(node->children));
      const vfloat16 bminmaxX  = permute(vfloat16::load((const float*)&node->lower_x), ray.permX);
      const vfloat16 bminmaxY  = permute(vfloat16::load((const float*)&node->lower_y), ray.permY);
      const vfloat16 bminmaxZ  = permute(vfloat16::load((const float*)&node->lower_z), ray.permZ);
      const vfloat16 tNearFarX = msub(bminmaxX, ray.rdir.x, ray.org_rdir.x);
      const vfloat16 tNearFarY = msub(bminmaxY, ray.rdir.y, ray.org_rdir.y);
      const vfloat16 tNearFarZ = msub(bminmaxZ, ray.rdir.z, ray.org_rdir.z);
      const vfloat16 tNear     = max(tNearFarX, tNearFarY, tNearFarZ, ray.tnear);
      const vfloat16 tFar      = min(tNearFarX, tNearFarY, tNearFarZ, ray.tfar);
      const vbool16 vmask      = le(vboolf16(m_valid),tNear,align_shift_right<8>(tFar, tFar));
      const size_t mask        = movemask(vmask);
      dist = tNear;
      return mask;
    }
    
#endif

    //////////////////////////////////////////////////////////////////////////////////////
    // Robust AlignedNode intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, int Nx>
      __forceinline size_t intersectNodeRobust(const typename BVHN<N>::AlignedNode* node, const TravRay<N,Nx,true>& ray, vfloat<Nx>& dist)
    {      
      const vfloat<N> tNearX = (vfloat<N>::load((float*)((const char*)&node->lower_x+ray.nearX)) - ray.org.x) * ray.rdir_near.x;
      const vfloat<N> tNearY = (vfloat<N>::load((float*)((const char*)&node->lower_x+ray.nearY)) - ray.org.y) * ray.rdir_near.y;
      const vfloat<N> tNearZ = (vfloat<N>::load((float*)((const char*)&node->lower_x+ray.nearZ)) - ray.org.z) * ray.rdir_near.z;
      const vfloat<N> tFarX  = (vfloat<N>::load((float*)((const char*)&node->lower_x+ray.farX )) - ray.org.x) * ray.rdir_far.x;
      const vfloat<N> tFarY  = (vfloat<N>::load((float*)((const char*)&node->lower_x+ray.farY )) - ray.org.y) * ray.rdir_far.y;
      const vfloat<N> tFarZ  = (vfloat<N>::load((float*)((const char*)&node->lower_x+ray.farZ )) - ray.org.z) * ray.rdir_far.z;
      const vfloat<N> tNear = max(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat<N> tFar  = min(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool<N> vmask = tNear <= tFar;
      const size_t mask = movemask(vmask);
      dist = tNear;
      return mask;
    }

#if defined(__AVX512F__) && !defined(__AVX512VL__) // KNL

    template<>
      __forceinline size_t intersectNodeRobust<4,16>(const typename BVHN<4>::AlignedNode* node, const TravRay<4,16,true>& ray, vfloat<16>& dist)
    {      
      const vfloat16 tNearX = (vfloat16(*(vfloat<4>*)((const char*)&node->lower_x+ray.nearX)) - ray.org.x) * ray.rdir_near.x;
      const vfloat16 tNearY = (vfloat16(*(vfloat<4>*)((const char*)&node->lower_x+ray.nearY)) - ray.org.y) * ray.rdir_near.y;
      const vfloat16 tNearZ = (vfloat16(*(vfloat<4>*)((const char*)&node->lower_x+ray.nearZ)) - ray.org.z) * ray.rdir_near.z;
      const vfloat16 tFarX  = (vfloat16(*(vfloat<4>*)((const char*)&node->lower_x+ray.farX )) - ray.org.x) * ray.rdir_far.x;
      const vfloat16 tFarY  = (vfloat16(*(vfloat<4>*)((const char*)&node->lower_x+ray.farY )) - ray.org.y) * ray.rdir_far.y;
      const vfloat16 tFarZ  = (vfloat16(*(vfloat<4>*)((const char*)&node->lower_x+ray.farZ )) - ray.org.z) * ray.rdir_far.z;
      const vfloat16 tNear = max(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat16 tFar  = min(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool16 vmask = le((1 << 4)-1,tNear,tFar);
      const size_t mask = movemask(vmask);
      dist = tNear;
      return mask;
    }

    template<>
      __forceinline size_t intersectNodeRobust<8,16>(const typename BVHN<8>::AlignedNode* node, const TravRay<8,16,true>& ray, vfloat<16>& dist)
    {      
      const vfloat16 tNearX = (vfloat16(*(vfloat<8>*)((const char*)&node->lower_x+ray.nearX)) - ray.org.x) * ray.rdir_near.x;
      const vfloat16 tNearY = (vfloat16(*(vfloat<8>*)((const char*)&node->lower_x+ray.nearY)) - ray.org.y) * ray.rdir_near.y;
      const vfloat16 tNearZ = (vfloat16(*(vfloat<8>*)((const char*)&node->lower_x+ray.nearZ)) - ray.org.z) * ray.rdir_near.z;
      const vfloat16 tFarX  = (vfloat16(*(vfloat<8>*)((const char*)&node->lower_x+ray.farX )) - ray.org.x) * ray.rdir_far.x;
      const vfloat16 tFarY  = (vfloat16(*(vfloat<8>*)((const char*)&node->lower_x+ray.farY )) - ray.org.y) * ray.rdir_far.y;
      const vfloat16 tFarZ  = (vfloat16(*(vfloat<8>*)((const char*)&node->lower_x+ray.farZ )) - ray.org.z) * ray.rdir_far.z;
      const vfloat16 tNear = max(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat16 tFar  = min(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool16 vmask = le((1 << 8)-1,tNear,tFar);
      const size_t mask = movemask(vmask);
      dist = tNear;
      return mask;
    }

#endif

    //////////////////////////////////////////////////////////////////////////////////////
    // Fast AlignedNodeMB intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N>
      __forceinline size_t intersectNode(const typename BVHN<N>::AlignedNodeMB* node, const TravRay<N,N,false>& ray, const float time, vfloat<N>& dist)
    {
      const vfloat<N>* pNearX = (const vfloat<N>*)((const char*)&node->lower_x+ray.nearX);
      const vfloat<N>* pNearY = (const vfloat<N>*)((const char*)&node->lower_x+ray.nearY);
      const vfloat<N>* pNearZ = (const vfloat<N>*)((const char*)&node->lower_x+ray.nearZ);
      const vfloat<N>* pFarX  = (const vfloat<N>*)((const char*)&node->lower_x+ray.farX);
      const vfloat<N>* pFarY  = (const vfloat<N>*)((const char*)&node->lower_x+ray.farY);
      const vfloat<N>* pFarZ  = (const vfloat<N>*)((const char*)&node->lower_x+ray.farZ);
#if defined(__AVX2__)
      const vfloat<N> tNearX = msub(madd(time,pNearX[6],vfloat<N>(pNearX[0])), ray.rdir.x, ray.org_rdir.x);
      const vfloat<N> tNearY = msub(madd(time,pNearY[6],vfloat<N>(pNearY[0])), ray.rdir.y, ray.org_rdir.y);
      const vfloat<N> tNearZ = msub(madd(time,pNearZ[6],vfloat<N>(pNearZ[0])), ray.rdir.z, ray.org_rdir.z);
      const vfloat<N> tFarX  = msub(madd(time,pFarX [6],vfloat<N>(pFarX [0])), ray.rdir.x, ray.org_rdir.x);
      const vfloat<N> tFarY  = msub(madd(time,pFarY [6],vfloat<N>(pFarY [0])), ray.rdir.y, ray.org_rdir.y);
      const vfloat<N> tFarZ  = msub(madd(time,pFarZ [6],vfloat<N>(pFarZ [0])), ray.rdir.z, ray.org_rdir.z);
#else
      const vfloat<N> tNearX = (madd(time,pNearX[6],vfloat<N>(pNearX[0])) - ray.org.x) * ray.rdir.x;
      const vfloat<N> tNearY = (madd(time,pNearY[6],vfloat<N>(pNearY[0])) - ray.org.y) * ray.rdir.y;
      const vfloat<N> tNearZ = (madd(time,pNearZ[6],vfloat<N>(pNearZ[0])) - ray.org.z) * ray.rdir.z;
      const vfloat<N> tFarX  = (madd(time,pFarX [6],vfloat<N>(pFarX [0])) - ray.org.x) * ray.rdir.x;
      const vfloat<N> tFarY  = (madd(time,pFarY [6],vfloat<N>(pFarY [0])) - ray.org.y) * ray.rdir.y;
      const vfloat<N> tFarZ  = (madd(time,pFarZ [6],vfloat<N>(pFarZ [0])) - ray.org.z) * ray.rdir.z;
#endif
#if defined(__AVX2__) && !defined(__AVX512F__) // HSW
      const vfloat<N> tNear = maxi(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat<N> tFar  = mini(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool<N> vmask = asInt(tNear) > asInt(tFar);
      const size_t mask = movemask(vmask) ^ ((1<<N)-1);
#elif defined(__AVX512F__) && !defined(__AVX512ER__) // SKX
      const vfloat<N> tNear = maxi(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat<N> tFar  = mini(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool<N> vmask = asInt(tNear) <= asInt(tFar);
      const size_t mask = movemask(vmask);
#else
      const vfloat<N> tNear = max(ray.tnear,tNearX,tNearY,tNearZ);
      const vfloat<N> tFar  = min(ray.tfar, tFarX ,tFarY ,tFarZ );
      const vbool<N> vmask = tNear <= tFar;
      const size_t mask = movemask(vmask);
#endif
      dist = tNear;
      return mask;
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Robust AlignedNodeMB intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N>
      __forceinline size_t intersectNodeRobust(const typename BVHN<N>::AlignedNodeMB* node, const TravRay<N,N,true>& ray, const float time, vfloat<N>& dist)
    {
      const vfloat<N>* pNearX = (const vfloat<N>*)((const char*)&node->lower_x+ray.nearX);
      const vfloat<N>* pNearY = (const vfloat<N>*)((const char*)&node->lower_x+ray.nearY);
      const vfloat<N>* pNearZ = (const vfloat<N>*)((const char*)&node->lower_x+ray.nearZ);
      const vfloat<N> tNearX = (madd(time,pNearX[6],vfloat<N>(pNearX[0])) - ray.org.x) * ray.rdir_near.x;
      const vfloat<N> tNearY = (madd(time,pNearY[6],vfloat<N>(pNearY[0])) - ray.org.y) * ray.rdir_near.y;
      const vfloat<N> tNearZ = (madd(time,pNearZ[6],vfloat<N>(pNearZ[0])) - ray.org.z) * ray.rdir_near.z;
      const vfloat<N> tNear = max(ray.tnear,tNearX,tNearY,tNearZ);
      const vfloat<N>* pFarX = (const vfloat<N>*)((const char*)&node->lower_x+ray.farX);
      const vfloat<N>* pFarY = (const vfloat<N>*)((const char*)&node->lower_x+ray.farY);
      const vfloat<N>* pFarZ = (const vfloat<N>*)((const char*)&node->lower_x+ray.farZ);
      const vfloat<N> tFarX = (madd(time,pFarX[6],vfloat<N>(pFarX[0])) - ray.org.x) * ray.rdir_far.x;
      const vfloat<N> tFarY = (madd(time,pFarY[6],vfloat<N>(pFarY[0])) - ray.org.y) * ray.rdir_far.y;
      const vfloat<N> tFarZ = (madd(time,pFarZ[6],vfloat<N>(pFarZ[0])) - ray.org.z) * ray.rdir_far.z;
      const vfloat<N> tFar = min(ray.tfar,tFarX,tFarY,tFarZ);
      const size_t mask = movemask(tNear <= tFar);
      dist = tNear;
      return mask;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Fast AlignedNodeMB4D intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N>
      __forceinline size_t intersectNodeMB4D(const typename BVHN<N>::NodeRef ref, const TravRay<N,N,false>& ray, const float time, vfloat<N>& dist)
    {
      const typename BVHN<N>::AlignedNodeMB* node = ref.alignedNodeMB();
        
      const vfloat<N>* pNearX = (const vfloat<N>*)((const char*)&node->lower_x+ray.nearX);
      const vfloat<N>* pNearY = (const vfloat<N>*)((const char*)&node->lower_x+ray.nearY);
      const vfloat<N>* pNearZ = (const vfloat<N>*)((const char*)&node->lower_x+ray.nearZ);
      const vfloat<N>* pFarX  = (const vfloat<N>*)((const char*)&node->lower_x+ray.farX);
      const vfloat<N>* pFarY  = (const vfloat<N>*)((const char*)&node->lower_x+ray.farY);
      const vfloat<N>* pFarZ  = (const vfloat<N>*)((const char*)&node->lower_x+ray.farZ);
#if defined (__AVX2__)
      const vfloat<N> tNearX = msub(madd(time,pNearX[6],vfloat<N>(pNearX[0])), ray.rdir.x, ray.org_rdir.x);
      const vfloat<N> tNearY = msub(madd(time,pNearY[6],vfloat<N>(pNearY[0])), ray.rdir.y, ray.org_rdir.y);
      const vfloat<N> tNearZ = msub(madd(time,pNearZ[6],vfloat<N>(pNearZ[0])), ray.rdir.z, ray.org_rdir.z);
      const vfloat<N> tFarX  = msub(madd(time,pFarX [6],vfloat<N>(pFarX [0])), ray.rdir.x, ray.org_rdir.x);
      const vfloat<N> tFarY  = msub(madd(time,pFarY [6],vfloat<N>(pFarY [0])), ray.rdir.y, ray.org_rdir.y);
      const vfloat<N> tFarZ  = msub(madd(time,pFarZ [6],vfloat<N>(pFarZ [0])), ray.rdir.z, ray.org_rdir.z);
#else
      const vfloat<N> tNearX = (madd(time,pNearX[6],vfloat<N>(pNearX[0])) - ray.org.x) * ray.rdir.x;
      const vfloat<N> tNearY = (madd(time,pNearY[6],vfloat<N>(pNearY[0])) - ray.org.y) * ray.rdir.y;
      const vfloat<N> tNearZ = (madd(time,pNearZ[6],vfloat<N>(pNearZ[0])) - ray.org.z) * ray.rdir.z;
      const vfloat<N> tFarX  = (madd(time,pFarX [6],vfloat<N>(pFarX [0])) - ray.org.x) * ray.rdir.x;
      const vfloat<N> tFarY  = (madd(time,pFarY [6],vfloat<N>(pFarY [0])) - ray.org.y) * ray.rdir.y;
      const vfloat<N> tFarZ  = (madd(time,pFarZ [6],vfloat<N>(pFarZ [0])) - ray.org.z) * ray.rdir.z;
#endif
#if defined(__AVX2__) && !defined(__AVX512F__)
      const vfloat<N> tNear = maxi(maxi(tNearX,tNearY),maxi(tNearZ,ray.tnear));
      const vfloat<N> tFar  = mini(mini(tFarX ,tFarY ),mini(tFarZ ,ray.tfar ));
#else
      const vfloat<N> tNear = max(ray.tnear,tNearX,tNearY,tNearZ);
      const vfloat<N> tFar  = min(ray.tfar, tFarX ,tFarY ,tFarZ );
#endif
      vbool<N> vmask = tNear <= tFar;
      if (unlikely(ref.isAlignedNodeMB4D())) {
        const typename BVHN<N>::AlignedNodeMB4D* node1 = (const typename BVHN<N>::AlignedNodeMB4D*) node;
        vmask &= (node1->lower_t <= time) & (time < node1->upper_t);
      }
      const size_t mask = movemask(vmask);
      dist = tNear;
      return mask;
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Robust AlignedNodeMB4D intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N>
      __forceinline size_t intersectNodeMB4DRobust(const typename BVHN<N>::NodeRef ref, const TravRay<N,N,true>& ray, const float time, vfloat<N>& dist)
    {
      const typename BVHN<N>::AlignedNodeMB* node = ref.alignedNodeMB();

      const vfloat<N>* pNearX = (const vfloat<N>*)((const char*)&node->lower_x+ray.nearX);
      const vfloat<N>* pNearY = (const vfloat<N>*)((const char*)&node->lower_x+ray.nearY);
      const vfloat<N>* pNearZ = (const vfloat<N>*)((const char*)&node->lower_x+ray.nearZ);
      const vfloat<N> tNearX = (madd(time,pNearX[6],vfloat<N>(pNearX[0])) - ray.org.x) * ray.rdir_near.x;
      const vfloat<N> tNearY = (madd(time,pNearY[6],vfloat<N>(pNearY[0])) - ray.org.y) * ray.rdir_near.y;
      const vfloat<N> tNearZ = (madd(time,pNearZ[6],vfloat<N>(pNearZ[0])) - ray.org.z) * ray.rdir_near.z;
      const vfloat<N> tNear = max(ray.tnear,tNearX,tNearY,tNearZ);
      const vfloat<N>* pFarX = (const vfloat<N>*)((const char*)&node->lower_x+ray.farX);
      const vfloat<N>* pFarY = (const vfloat<N>*)((const char*)&node->lower_x+ray.farY);
      const vfloat<N>* pFarZ = (const vfloat<N>*)((const char*)&node->lower_x+ray.farZ);
      const vfloat<N> tFarX = (madd(time,pFarX[6],vfloat<N>(pFarX[0])) - ray.org.x) * ray.rdir_far.x;
      const vfloat<N> tFarY = (madd(time,pFarY[6],vfloat<N>(pFarY[0])) - ray.org.y) * ray.rdir_far.y;
      const vfloat<N> tFarZ = (madd(time,pFarZ[6],vfloat<N>(pFarZ[0])) - ray.org.z) * ray.rdir_far.z;
      const vfloat<N> tFar = min(ray.tfar,tFarX,tFarY,tFarZ);
      vbool<N> vmask = tNear <= tFar;
      if (unlikely(ref.isAlignedNodeMB4D())) {
        const typename BVHN<N>::AlignedNodeMB4D* node1 = (const typename BVHN<N>::AlignedNodeMB4D*) node;
        vmask &= (node1->lower_t <= time) & (time < node1->upper_t);
      }
      const size_t mask = movemask(vmask);
      dist = tNear;
      return mask;
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Fast QuantizedBaseNode intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, int Nx, bool robust>
      __forceinline size_t intersectNode(const typename BVHN<N>::QuantizedBaseNode* node, const TravRay<N,Nx,robust>& ray, vfloat<Nx>& dist);

    template<>
      __forceinline size_t intersectNode<4,4>(const typename BVH4::QuantizedBaseNode* node, const TravRay<4,4,false>& ray, vfloat4& dist)
    {
      const vfloat4 start_x(node->start.x);
      const vfloat4 scale_x(node->scale.x);
      const vfloat4 lower_x = madd(node->dequantize<4>(ray.nearX >> 2),scale_x,start_x);
      const vfloat4 upper_x = madd(node->dequantize<4>(ray.farX  >> 2),scale_x,start_x);
      const vfloat4 start_y(node->start.y);
      const vfloat4 scale_y(node->scale.y);
      const vfloat4 lower_y = madd(node->dequantize<4>(ray.nearY >> 2),scale_y,start_y);
      const vfloat4 upper_y = madd(node->dequantize<4>(ray.farY  >> 2),scale_y,start_y);
      const vfloat4 start_z(node->start.z);
      const vfloat4 scale_z(node->scale.z);
      const vfloat4 lower_z = madd(node->dequantize<4>(ray.nearZ >> 2),scale_z,start_z);
      const vfloat4 upper_z = madd(node->dequantize<4>(ray.farZ  >> 2),scale_z,start_z);

#if defined(__AVX2__)
      const vfloat4 tNearX = msub(lower_x, ray.rdir.x, ray.org_rdir.x);
      const vfloat4 tNearY = msub(lower_y, ray.rdir.y, ray.org_rdir.y);
      const vfloat4 tNearZ = msub(lower_z, ray.rdir.z, ray.org_rdir.z);
      const vfloat4 tFarX  = msub(upper_x, ray.rdir.x, ray.org_rdir.x);
      const vfloat4 tFarY  = msub(upper_y, ray.rdir.y, ray.org_rdir.y);
      const vfloat4 tFarZ  = msub(upper_z, ray.rdir.z, ray.org_rdir.z);
#else
      const vfloat4 tNearX = (lower_x - ray.org.x) * ray.rdir.x;
      const vfloat4 tNearY = (lower_y - ray.org.y) * ray.rdir.y;
      const vfloat4 tNearZ = (lower_z - ray.org.z) * ray.rdir.z;
      const vfloat4 tFarX  = (upper_x - ray.org.x) * ray.rdir.x;
      const vfloat4 tFarY  = (upper_y - ray.org.y) * ray.rdir.y;
      const vfloat4 tFarZ  = (upper_z - ray.org.z) * ray.rdir.z;
#endif
      
#if defined(__SSE4_1__) && !defined(__AVX512F__) // up to HSW
      const vfloat4 tNear = maxi(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat4 tFar  = mini(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool4 vmask = asInt(tNear) > asInt(tFar);
      const size_t mask = movemask(vmask) ^ ((1<<4)-1);
#elif defined(__AVX512F__) && !defined(__AVX512ER__) // SKX
      const vfloat4 tNear = maxi(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat4 tFar  = mini(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool4 vmask = asInt(tNear) <= asInt(tFar);
      const size_t mask = movemask(vmask);
#else
      const vfloat4 tNear = max(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat4 tFar  = min(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool4 vmask = tNear <= tFar;
      const size_t mask = movemask(vmask);
#endif
      dist = tNear;
      return mask;
    }

    template<>
      __forceinline size_t intersectNode<4,4>(const typename BVH4::QuantizedBaseNode* node, const TravRay<4,4,true>& ray, vfloat4& dist)
    {
      const vfloat4 start_x(node->start.x);
      const vfloat4 scale_x(node->scale.x);
      const vfloat4 lower_x = madd(node->dequantize<4>(ray.nearX >> 2),scale_x,start_x);
      const vfloat4 upper_x = madd(node->dequantize<4>(ray.farX  >> 2),scale_x,start_x);
      const vfloat4 start_y(node->start.y);
      const vfloat4 scale_y(node->scale.y);
      const vfloat4 lower_y = madd(node->dequantize<4>(ray.nearY >> 2),scale_y,start_y);
      const vfloat4 upper_y = madd(node->dequantize<4>(ray.farY  >> 2),scale_y,start_y);
      const vfloat4 start_z(node->start.z);
      const vfloat4 scale_z(node->scale.z);
      const vfloat4 lower_z = madd(node->dequantize<4>(ray.nearZ >> 2),scale_z,start_z);
      const vfloat4 upper_z = madd(node->dequantize<4>(ray.farZ  >> 2),scale_z,start_z);

      const vfloat4 tNearX = (lower_x - ray.org.x) * ray.rdir_far.x;
      const vfloat4 tNearY = (lower_y - ray.org.y) * ray.rdir_far.y;
      const vfloat4 tNearZ = (lower_z - ray.org.z) * ray.rdir_far.z;
      const vfloat4 tFarX  = (upper_x - ray.org.x) * ray.rdir_far.x;
      const vfloat4 tFarY  = (upper_y - ray.org.y) * ray.rdir_far.y;
      const vfloat4 tFarZ  = (upper_z - ray.org.z) * ray.rdir_far.z;
      
      const vfloat4 tNear = max(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat4 tFar  = min(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool4 vmask = tNear <= tFar;
      const size_t mask = movemask(vmask);
      dist = tNear;
      return mask;
    }


#if defined(__AVX__)

    template<>
      __forceinline size_t intersectNode<8,8>(const typename BVH8::QuantizedBaseNode* node, const TravRay<8,8,false>& ray, vfloat8& dist)
    {
      const vfloat8 start_x(node->start.x);
      const vfloat8 scale_x(node->scale.x);
      const vfloat8 lower_x = madd(node->dequantize<8>(ray.nearX >> 2),scale_x,start_x);
      const vfloat8 upper_x = madd(node->dequantize<8>(ray.farX  >> 2),scale_x,start_x);
      const vfloat8 start_y(node->start.y);
      const vfloat8 scale_y(node->scale.y);
      const vfloat8 lower_y = madd(node->dequantize<8>(ray.nearY >> 2),scale_y,start_y);
      const vfloat8 upper_y = madd(node->dequantize<8>(ray.farY  >> 2),scale_y,start_y);
      const vfloat8 start_z(node->start.z);
      const vfloat8 scale_z(node->scale.z);
      const vfloat8 lower_z = madd(node->dequantize<8>(ray.nearZ >> 2),scale_z,start_z);
      const vfloat8 upper_z = madd(node->dequantize<8>(ray.farZ  >> 2),scale_z,start_z);

#if defined(__AVX2__)
      const vfloat8 tNearX = msub(lower_x, ray.rdir.x, ray.org_rdir.x);
      const vfloat8 tNearY = msub(lower_y, ray.rdir.y, ray.org_rdir.y);
      const vfloat8 tNearZ = msub(lower_z, ray.rdir.z, ray.org_rdir.z);
      const vfloat8 tFarX  = msub(upper_x, ray.rdir.x, ray.org_rdir.x);
      const vfloat8 tFarY  = msub(upper_y, ray.rdir.y, ray.org_rdir.y);
      const vfloat8 tFarZ  = msub(upper_z, ray.rdir.z, ray.org_rdir.z);
#else
      const vfloat8 tNearX = (lower_x - ray.org.x) * ray.rdir.x;
      const vfloat8 tNearY = (lower_y - ray.org.y) * ray.rdir.y;
      const vfloat8 tNearZ = (lower_z - ray.org.z) * ray.rdir.z;
      const vfloat8 tFarX  = (upper_x - ray.org.x) * ray.rdir.x;
      const vfloat8 tFarY  = (upper_y - ray.org.y) * ray.rdir.y;
      const vfloat8 tFarZ  = (upper_z - ray.org.z) * ray.rdir.z;
#endif
      
#if defined(__AVX2__) && !defined(__AVX512F__) // HSW
      const vfloat8 tNear = maxi(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat8 tFar  = mini(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool8 vmask = asInt(tNear) > asInt(tFar);
      const size_t mask = movemask(vmask) ^ ((1<<8)-1);
#elif defined(__AVX512F__) && !defined(__AVX512ER__) // SKX
      const vfloat8 tNear = maxi(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat8 tFar  = mini(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool8 vmask = asInt(tNear) <= asInt(tFar);
      const size_t mask = movemask(vmask);
#else
      const vfloat8 tNear = max(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat8 tFar  = min(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool8 vmask = tNear <= tFar;
      const size_t mask = movemask(vmask);
#endif
      dist = tNear;
      return mask;
    }

    template<>
      __forceinline size_t intersectNode<8,8>(const typename BVH8::QuantizedBaseNode* node, const TravRay<8,8,true>& ray, vfloat8& dist)
    {
      const vfloat8 start_x(node->start.x);
      const vfloat8 scale_x(node->scale.x);
      const vfloat8 lower_x = madd(node->dequantize<8>(ray.nearX >> 2),scale_x,start_x);
      const vfloat8 upper_x = madd(node->dequantize<8>(ray.farX  >> 2),scale_x,start_x);
      const vfloat8 start_y(node->start.y);
      const vfloat8 scale_y(node->scale.y);
      const vfloat8 lower_y = madd(node->dequantize<8>(ray.nearY >> 2),scale_y,start_y);
      const vfloat8 upper_y = madd(node->dequantize<8>(ray.farY  >> 2),scale_y,start_y);
      const vfloat8 start_z(node->start.z);
      const vfloat8 scale_z(node->scale.z);
      const vfloat8 lower_z = madd(node->dequantize<8>(ray.nearZ >> 2),scale_z,start_z);
      const vfloat8 upper_z = madd(node->dequantize<8>(ray.farZ  >> 2),scale_z,start_z);

      const vfloat8 tNearX = (lower_x - ray.org.x) * ray.rdir_far.x;
      const vfloat8 tNearY = (lower_y - ray.org.y) * ray.rdir_far.y;
      const vfloat8 tNearZ = (lower_z - ray.org.z) * ray.rdir_far.z;
      const vfloat8 tFarX  = (upper_x - ray.org.x) * ray.rdir_far.x;
      const vfloat8 tFarY  = (upper_y - ray.org.y) * ray.rdir_far.y;
      const vfloat8 tFarZ  = (upper_z - ray.org.z) * ray.rdir_far.z;
      
      const vfloat8 tNear = max(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat8 tFar  = min(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool8 vmask = tNear <= tFar;
      const size_t mask = movemask(vmask);

      dist = tNear;
      return mask;
    }


#endif

#if defined(__AVX512F__) && !defined(__AVX512VL__) // KNL

    template<>
      __forceinline size_t intersectNode<4,16>(const typename BVH4::QuantizedBaseNode* node, const TravRay<4,16,false>& ray, vfloat16& dist)
    {
      const vfloat16 start_x(node->start.x);
      const vfloat16 scale_x(node->scale.x);
      const vfloat16 lower_x = madd(vfloat16(node->dequantize<4>(ray.nearX >> 2)),scale_x,start_x);
      const vfloat16 upper_x = madd(vfloat16(node->dequantize<4>(ray.farX  >> 2)),scale_x,start_x);
      const vfloat16 start_y(node->start.y);
      const vfloat16 scale_y(node->scale.y);
      const vfloat16 lower_y = madd(vfloat16(node->dequantize<4>(ray.nearY >> 2)),scale_y,start_y);
      const vfloat16 upper_y = madd(vfloat16(node->dequantize<4>(ray.farY  >> 2)),scale_y,start_y);
      const vfloat16 start_z(node->start.z);
      const vfloat16 scale_z(node->scale.z);
      const vfloat16 lower_z = madd(vfloat16(node->dequantize<4>(ray.nearZ >> 2)),scale_z,start_z);
      const vfloat16 upper_z = madd(vfloat16(node->dequantize<4>(ray.farZ  >> 2)),scale_z,start_z);

      const vfloat16 tNearX = msub(lower_x, ray.rdir.x, ray.org_rdir.x);
      const vfloat16 tNearY = msub(lower_y, ray.rdir.y, ray.org_rdir.y);
      const vfloat16 tNearZ = msub(lower_z, ray.rdir.z, ray.org_rdir.z);
      const vfloat16 tFarX  = msub(upper_x, ray.rdir.x, ray.org_rdir.x);
      const vfloat16 tFarY  = msub(upper_y, ray.rdir.y, ray.org_rdir.y);
      const vfloat16 tFarZ  = msub(upper_z, ray.rdir.z, ray.org_rdir.z);      
      const vfloat16 tNear  = max(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat16 tFar   = min(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool16 vmask   = le(vbool16(0xf),tNear,tFar);
      const size_t mask     = movemask(vmask);
      dist = tNear;
      return mask;
    }

    template<>
      __forceinline size_t intersectNode<4,16>(const typename BVH4::QuantizedBaseNode* node, const TravRay<4,16,true>& ray, vfloat16& dist)
    {
      const vfloat16 start_x(node->start.x);
      const vfloat16 scale_x(node->scale.x);
      const vfloat16 lower_x = madd(vfloat16(node->dequantize<4>(ray.nearX >> 2)),scale_x,start_x);
      const vfloat16 upper_x = madd(vfloat16(node->dequantize<4>(ray.farX  >> 2)),scale_x,start_x);
      const vfloat16 start_y(node->start.y);
      const vfloat16 scale_y(node->scale.y);
      const vfloat16 lower_y = madd(vfloat16(node->dequantize<4>(ray.nearY >> 2)),scale_y,start_y);
      const vfloat16 upper_y = madd(vfloat16(node->dequantize<4>(ray.farY  >> 2)),scale_y,start_y);
      const vfloat16 start_z(node->start.z);
      const vfloat16 scale_z(node->scale.z);
      const vfloat16 lower_z = madd(vfloat16(node->dequantize<4>(ray.nearZ >> 2)),scale_z,start_z);
      const vfloat16 upper_z = madd(vfloat16(node->dequantize<4>(ray.farZ  >> 2)),scale_z,start_z);

      const vfloat16 tNearX = (lower_x - ray.org.x) * ray.rdir_far.x;
      const vfloat16 tNearY = (lower_y - ray.org.y) * ray.rdir_far.y;
      const vfloat16 tNearZ = (lower_z - ray.org.z) * ray.rdir_far.z;
      const vfloat16 tFarX  = (upper_x - ray.org.x) * ray.rdir_far.x;
      const vfloat16 tFarY  = (upper_y - ray.org.y) * ray.rdir_far.y;
      const vfloat16 tFarZ  = (upper_z - ray.org.z) * ray.rdir_far.z;

      const vfloat16 tNear  = max(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat16 tFar   = min(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool16 vmask   = le(vbool16(0xf),tNear,tFar);
      const size_t mask     = movemask(vmask);
      dist = tNear;
      return mask;
    }

    template<>
      __forceinline size_t intersectNode<8,16>(const typename BVH8::QuantizedBaseNode* node, const TravRay<8,16,false>& ray, vfloat16& dist)
    {
      const vbool16 m_valid(node->validMask16());
      const vfloat16 bminmaxX  = node->dequantizeLowerUpperX(ray.permX);
      const vfloat16 bminmaxY  = node->dequantizeLowerUpperY(ray.permY);
      const vfloat16 bminmaxZ  = node->dequantizeLowerUpperZ(ray.permZ);
      const vfloat16 tNearFarX = msub(bminmaxX, ray.rdir.x, ray.org_rdir.x);
      const vfloat16 tNearFarY = msub(bminmaxY, ray.rdir.y, ray.org_rdir.y);
      const vfloat16 tNearFarZ = msub(bminmaxZ, ray.rdir.z, ray.org_rdir.z);
      const vfloat16 tNear     = max(tNearFarX, tNearFarY, tNearFarZ, ray.tnear);
      const vfloat16 tFar      = min(tNearFarX, tNearFarY, tNearFarZ, ray.tfar);
      const vbool16 vmask      = le(m_valid,tNear,align_shift_right<8>(tFar, tFar));
      const size_t mask        = movemask(vmask);
      dist = tNear;
      return mask;
    }

    template<>
      __forceinline size_t intersectNode<8,16>(const typename BVH8::QuantizedBaseNode* node, const TravRay<8,16,true>& ray, vfloat16& dist)
    {
      const vbool16 m_valid(node->validMask16());
      const vfloat16 bminmaxX  = node->dequantizeLowerUpperX(ray.permX);
      const vfloat16 bminmaxY  = node->dequantizeLowerUpperY(ray.permY);
      const vfloat16 bminmaxZ  = node->dequantizeLowerUpperZ(ray.permZ);
      const vfloat16 tNearFarX = (bminmaxX - ray.org.x) * ray.rdir_far.x;
      const vfloat16 tNearFarY = (bminmaxY - ray.org.y) * ray.rdir_far.y;
      const vfloat16 tNearFarZ = (bminmaxZ - ray.org.z) * ray.rdir_far.z;
      const vfloat16 tNear     = max(tNearFarX, tNearFarY, tNearFarZ, ray.tnear);
      const vfloat16 tFar      = min(tNearFarX, tNearFarY, tNearFarZ, ray.tfar);
      const vbool16 vmask      = le(m_valid,tNear,align_shift_right<8>(tFar, tFar));
      const size_t mask        = movemask(vmask);
      dist = tNear;
      return mask;
    }

    
#endif


    template<int N, int Nx>
      __forceinline size_t intersectNode(const typename BVHN<N>::QuantizedBaseNodeMB* node, const TravRay<N,Nx,false>& ray, const float time, vfloat<N>& dist)
    {
      const vboolf<N> mvalid    = node->validMask();
      const vfloat<N> lower_x   = node->dequantizeLowerX(time);
      const vfloat<N> upper_x   = node->dequantizeUpperX(time);
      const vfloat<N> lower_y   = node->dequantizeLowerY(time);
      const vfloat<N> upper_y   = node->dequantizeUpperY(time);
      const vfloat<N> lower_z   = node->dequantizeLowerZ(time);
      const vfloat<N> upper_z   = node->dequantizeUpperZ(time);     
#if defined(__AVX2__)
      const vfloat<N> tNearX = msub(lower_x, ray.rdir.x, ray.org_rdir.x);
      const vfloat<N> tNearY = msub(lower_y, ray.rdir.y, ray.org_rdir.y);
      const vfloat<N> tNearZ = msub(lower_z, ray.rdir.z, ray.org_rdir.z);
      const vfloat<N> tFarX  = msub(upper_x, ray.rdir.x, ray.org_rdir.x);
      const vfloat<N> tFarY  = msub(upper_y, ray.rdir.y, ray.org_rdir.y);
      const vfloat<N> tFarZ  = msub(upper_z, ray.rdir.z, ray.org_rdir.z);
#else
      const vfloat<N> tNearX = (lower_x - ray.org.x) * ray.rdir.x;
      const vfloat<N> tNearY = (lower_y - ray.org.y) * ray.rdir.y;
      const vfloat<N> tNearZ = (lower_z - ray.org.z) * ray.rdir.z;
      const vfloat<N> tFarX  = (upper_x - ray.org.x) * ray.rdir.x;
      const vfloat<N> tFarY  = (upper_y - ray.org.y) * ray.rdir.y;
      const vfloat<N> tFarZ  = (upper_z - ray.org.z) * ray.rdir.z;
#endif      

      const vfloat<N> tminX = mini(tNearX,tFarX);
      const vfloat<N> tmaxX = maxi(tNearX,tFarX);
      const vfloat<N> tminY = mini(tNearY,tFarY);
      const vfloat<N> tmaxY = maxi(tNearY,tFarY);
      const vfloat<N> tminZ = mini(tNearZ,tFarZ);
      const vfloat<N> tmaxZ = maxi(tNearZ,tFarZ);
      const vfloat<N> tNear = maxi(tminX,tminY,tminZ,ray.tnear);
      const vfloat<N> tFar  = mini(tmaxX,tmaxY,tmaxZ,ray.tfar);
#if defined(__AVX512F__) && !defined(__AVX512ER__) // SKX
      const vbool<N> vmask =  le(mvalid,asInt(tNear),asInt(tFar));
#else
      const vbool<N> vmask = (asInt(tNear) <= asInt(tFar)) & mvalid;
#endif
      const size_t mask = movemask(vmask);
      dist = tNear;
      return mask;      
    }

    template<int N, int Nx>
      __forceinline size_t intersectNode(const typename BVHN<N>::QuantizedBaseNodeMB* node, const TravRay<N,Nx,true>& ray, const float time, vfloat<N>& dist)
    {
      const vboolf<N> mvalid    = node->validMask();
      const vfloat<N> lower_x   = node->dequantizeLowerX(time);
      const vfloat<N> upper_x   = node->dequantizeUpperX(time);
      const vfloat<N> lower_y   = node->dequantizeLowerY(time);
      const vfloat<N> upper_y   = node->dequantizeUpperY(time);
      const vfloat<N> lower_z   = node->dequantizeLowerZ(time);
      const vfloat<N> upper_z   = node->dequantizeUpperZ(time);     
      const vfloat<N> tNearX = (lower_x - ray.org.x) * ray.rdir_far.x;
      const vfloat<N> tNearY = (lower_y - ray.org.y) * ray.rdir_far.y;
      const vfloat<N> tNearZ = (lower_z - ray.org.z) * ray.rdir_far.z;
      const vfloat<N> tFarX  = (upper_x - ray.org.x) * ray.rdir_far.x;
      const vfloat<N> tFarY  = (upper_y - ray.org.y) * ray.rdir_far.y;
      const vfloat<N> tFarZ  = (upper_z - ray.org.z) * ray.rdir_far.z;

      const vfloat<N> tminX = mini(tNearX,tFarX);
      const vfloat<N> tmaxX = maxi(tNearX,tFarX);
      const vfloat<N> tminY = mini(tNearY,tFarY);
      const vfloat<N> tmaxY = maxi(tNearY,tFarY);
      const vfloat<N> tminZ = mini(tNearZ,tFarZ);
      const vfloat<N> tmaxZ = maxi(tNearZ,tFarZ);
      const vfloat<N> tNear = maxi(tminX,tminY,tminZ,ray.tnear);
      const vfloat<N> tFar  = mini(tmaxX,tmaxY,tmaxZ,ray.tfar);
#if defined(__AVX512F__) && !defined(__AVX512ER__) // SKX
      const vbool<N> vmask =  le(mvalid,asInt(tNear),asInt(tFar));
#else
      const vbool<N> vmask = (asInt(tNear) <= asInt(tFar)) & mvalid;
#endif
      const size_t mask = movemask(vmask);
      dist = tNear;
      return mask;      
    }


#if defined(__AVX512ER__)
    // for KNL
    template<>
      __forceinline size_t intersectNode<4,16>(const typename BVHN<4>::QuantizedBaseNodeMB* node, const TravRay<4,16,false>& ray, const float time, vfloat<4>& dist)
    {
      const size_t  mvalid    = movemask(node->validMask());
      const vfloat16 lower_x  = node->dequantizeLowerX(time);
      const vfloat16 upper_x  = node->dequantizeUpperX(time);
      const vfloat16 lower_y  = node->dequantizeLowerY(time);
      const vfloat16 upper_y  = node->dequantizeUpperY(time);
      const vfloat16 lower_z  = node->dequantizeLowerZ(time);
      const vfloat16 upper_z  = node->dequantizeUpperZ(time);     

      const vfloat16 tNearX = msub(lower_x, ray.rdir.x, ray.org_rdir.x);
      const vfloat16 tNearY = msub(lower_y, ray.rdir.y, ray.org_rdir.y);
      const vfloat16 tNearZ = msub(lower_z, ray.rdir.z, ray.org_rdir.z);
      const vfloat16 tFarX  = msub(upper_x, ray.rdir.x, ray.org_rdir.x);
      const vfloat16 tFarY  = msub(upper_y, ray.rdir.y, ray.org_rdir.y);
      const vfloat16 tFarZ  = msub(upper_z, ray.rdir.z, ray.org_rdir.z);

      const vfloat16 tminX = min(tNearX,tFarX);
      const vfloat16 tmaxX = max(tNearX,tFarX);
      const vfloat16 tminY = min(tNearY,tFarY);
      const vfloat16 tmaxY = max(tNearY,tFarY);
      const vfloat16 tminZ = min(tNearZ,tFarZ);
      const vfloat16 tmaxZ = max(tNearZ,tFarZ);
      const vfloat16 tNear = max(tminX,tminY,tminZ,ray.tnear);
      const vfloat16 tFar  = min(tmaxX,tmaxY,tmaxZ,ray.tfar );
      const vbool16 vmask =  tNear <= tFar;
      const size_t mask = movemask(vmask) & mvalid;
      dist = extractN<4,0>(tNear);
      return mask;      
    }


    // for KNL
    template<>
      __forceinline size_t intersectNode<4,16>(const typename BVHN<4>::QuantizedBaseNodeMB* node, const TravRay<4,16,true>& ray, const float time, vfloat<4>& dist)
    {
      const size_t  mvalid    = movemask(node->validMask());
      const vfloat16 lower_x  = node->dequantizeLowerX(time);
      const vfloat16 upper_x  = node->dequantizeUpperX(time);
      const vfloat16 lower_y  = node->dequantizeLowerY(time);
      const vfloat16 upper_y  = node->dequantizeUpperY(time);
      const vfloat16 lower_z  = node->dequantizeLowerZ(time);
      const vfloat16 upper_z  = node->dequantizeUpperZ(time);     

      const vfloat16 tNearX = (lower_x - ray.org.x) * ray.rdir_far.x;
      const vfloat16 tNearY = (lower_y - ray.org.y) * ray.rdir_far.y;
      const vfloat16 tNearZ = (lower_z - ray.org.z) * ray.rdir_far.z;
      const vfloat16 tFarX  = (upper_x - ray.org.x) * ray.rdir_far.x;
      const vfloat16 tFarY  = (upper_y - ray.org.y) * ray.rdir_far.y;
      const vfloat16 tFarZ  = (upper_z - ray.org.z) * ray.rdir_far.z;

      const vfloat16 tminX = min(tNearX,tFarX);
      const vfloat16 tmaxX = max(tNearX,tFarX);
      const vfloat16 tminY = min(tNearY,tFarY);
      const vfloat16 tmaxY = max(tNearY,tFarY);
      const vfloat16 tminZ = min(tNearZ,tFarZ);
      const vfloat16 tmaxZ = max(tNearZ,tFarZ);
      const vfloat16 tNear = max(tminX,tminY,tminZ,ray.tnear);
      const vfloat16 tFar  = min(tmaxX,tmaxY,tmaxZ,ray.tfar );
      const vbool16 vmask =  tNear <= tFar;
      const size_t mask = movemask(vmask) & mvalid;
      dist = extractN<4,0>(tNear);
      return mask;      
    }

#endif

    //////////////////////////////////////////////////////////////////////////////////////
    // Fast UnalignedNode intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, bool robust>
      __forceinline size_t intersectNode(const typename BVHN<N>::UnalignedNode* node, const TravRay<N,N,robust>& ray, vfloat<N>& dist)
    {
      const Vec3vf<N> dir = xfmVector(node->naabb,ray.dir);
      //const Vec3vf<N> nrdir = Vec3vf<N>(vfloat<N>(-1.0f))/dir;
      const Vec3vf<N> nrdir = Vec3vf<N>(vfloat<N>(-1.0f))*rcp_safe(dir);
      const Vec3vf<N> org = xfmPoint(node->naabb,ray.org);
      const Vec3vf<N> tLowerXYZ = org * nrdir;       // (Vec3fa(zero) - org) * rdir;
      const Vec3vf<N> tUpperXYZ = tLowerXYZ - nrdir; // (Vec3fa(one ) - org) * rdir;

      const vfloat<N> tNearX = mini(tLowerXYZ.x,tUpperXYZ.x);
      const vfloat<N> tNearY = mini(tLowerXYZ.y,tUpperXYZ.y);
      const vfloat<N> tNearZ = mini(tLowerXYZ.z,tUpperXYZ.z);
      const vfloat<N> tFarX  = maxi(tLowerXYZ.x,tUpperXYZ.x);
      const vfloat<N> tFarY  = maxi(tLowerXYZ.y,tUpperXYZ.y);
      const vfloat<N> tFarZ  = maxi(tLowerXYZ.z,tUpperXYZ.z);
      const vfloat<N> tNear  = max(ray.tnear, tNearX,tNearY,tNearZ);
      const vfloat<N> tFar   = min(ray.tfar,  tFarX ,tFarY ,tFarZ );
      const vbool<N> vmask = tNear <= tFar;
      dist = tNear;
      return movemask(vmask);
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Fast UnalignedNodeMB intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, bool robust>
      __forceinline size_t intersectNode(const typename BVHN<N>::UnalignedNodeMB* node, const TravRay<N,N,robust>& ray, const float time, vfloat<N>& dist)
    {
      const AffineSpace3vf<N> xfm = node->space0;
      const Vec3vf<N> b0_lower = zero;
      const Vec3vf<N> b0_upper = one;
      const Vec3vf<N> lower = lerp(b0_lower,node->b1.lower,vfloat<N>(time));
      const Vec3vf<N> upper = lerp(b0_upper,node->b1.upper,vfloat<N>(time));

      const BBox3vf<N> bounds(lower,upper);
      const Vec3vf<N> dir = xfmVector(xfm,ray.dir);
      const Vec3vf<N> rdir = rcp_safe(dir);
      const Vec3vf<N> org = xfmPoint(xfm,ray.org);

      const Vec3vf<N> tLowerXYZ = (bounds.lower - org) * rdir;
      const Vec3vf<N> tUpperXYZ = (bounds.upper - org) * rdir;

      const vfloat<N> tNearX = mini(tLowerXYZ.x,tUpperXYZ.x);
      const vfloat<N> tNearY = mini(tLowerXYZ.y,tUpperXYZ.y);
      const vfloat<N> tNearZ = mini(tLowerXYZ.z,tUpperXYZ.z);
      const vfloat<N> tFarX  = maxi(tLowerXYZ.x,tUpperXYZ.x);
      const vfloat<N> tFarY  = maxi(tLowerXYZ.y,tUpperXYZ.y);
      const vfloat<N> tFarZ  = maxi(tLowerXYZ.z,tUpperXYZ.z);
      const vfloat<N> tNear  = max(ray.tnear, tNearX,tNearY,tNearZ);
      const vfloat<N> tFar   = min(ray.tfar,  tFarX ,tFarY ,tFarZ );
      const vbool<N> vmask = tNear <= tFar;
      dist = tNear;
      return movemask(vmask);
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Node intersectors used in ray traversal
    //////////////////////////////////////////////////////////////////////////////////////

    /*! Intersects N nodes with 1 ray */
    template<int N, int Nx, int types, bool robust>
    struct BVHNNodeIntersector1;

    template<int N, int Nx>
    struct BVHNNodeIntersector1<N, Nx, BVH_AN1, false>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,Nx,false>& ray, float time, vfloat<Nx>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = intersectNode(node.alignedNode(), ray, dist);
        return true;
      }
    };

    template<int N, int Nx>
    struct BVHNNodeIntersector1<N, Nx, BVH_AN1, true>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,Nx,true>& ray, float time, vfloat<Nx>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = intersectNodeRobust(node.alignedNode(), ray, dist);
        return true;
      }
    };

    template<int N, int Nx>
    struct BVHNNodeIntersector1<N, Nx, BVH_AN2, false>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,Nx,false>& ray, float time, vfloat<Nx>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = intersectNode(node.alignedNodeMB(), ray, time, dist);
        return true;
      }
    };

    template<int N, int Nx>
    struct BVHNNodeIntersector1<N, Nx, BVH_AN2, true>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,Nx,true>& ray, float time, vfloat<Nx>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = intersectNodeRobust(node.alignedNodeMB(), ray, time, dist);
        return true;
      }
    };

    template<int N, int Nx>
    struct BVHNNodeIntersector1<N, Nx, BVH_AN2_AN4D, false>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,Nx,false>& ray, float time, vfloat<Nx>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = intersectNodeMB4D<N>(node, ray, time, dist);
        return true;
      }
    };

    template<int N, int Nx>
    struct BVHNNodeIntersector1<N, Nx, BVH_AN2_AN4D, true>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,Nx,true>& ray, float time, vfloat<Nx>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = intersectNodeMB4DRobust<N>(node, ray, time, dist);
        return true;
      }
    };

    template<int N, int Nx>
    struct BVHNNodeIntersector1<N, Nx, BVH_AN1_UN1, false>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,Nx,false>& ray, float time, vfloat<Nx>& dist, size_t& mask)
      {
        if (likely(node.isAlignedNode()))          mask = intersectNode(node.alignedNode(), ray, dist);
        else if (unlikely(node.isUnalignedNode())) mask = intersectNode(node.unalignedNode(), ray, dist);
        else return false;
        return true;
      }
    };

    template<int N, int Nx>
    struct BVHNNodeIntersector1<N, Nx, BVH_AN2_UN2, false>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,Nx,false>& ray, float time, vfloat<Nx>& dist, size_t& mask)
      {
        if (likely(node.isAlignedNodeMB()))           mask = intersectNode(node.alignedNodeMB(), ray, time, dist);
        else if (unlikely(node.isUnalignedNodeMB()))  mask = intersectNode(node.unalignedNodeMB(), ray, time, dist);
        else return false;
        return true;
      }
    };

    template<int N, int Nx>
    struct BVHNNodeIntersector1<N, Nx, BVH_AN2_AN4D_UN2, false>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,Nx,false>& ray, float time, vfloat<Nx>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        if (unlikely(node.isUnalignedNodeMB())) mask = intersectNode(node.unalignedNodeMB(), ray, time, dist);
        else                                    mask = intersectNodeMB4D(node, ray, time, dist);
        return true;
      }
    };

    template<int N, int Nx>
    struct BVHNNodeIntersector1<N, Nx, BVH_QN1, false>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,Nx,false>& ray, float time, vfloat<Nx>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = intersectNode((const typename BVHN<N>::QuantizedNode*)node.quantizedNode(), ray, dist);
        return true;
      }
    };

    template<int N, int Nx>
    struct BVHNNodeIntersector1<N, Nx, BVH_QN1, true>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,Nx,true>& ray, float time, vfloat<Nx>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = intersectNodeRobust((const typename BVHN<N>::QuantizedNode*)node.quantizedNode(), ray, dist);
        return true;
      }
    };

    /*! Intersects N nodes with K rays */
    template<int N, int Nx, bool robust>
      struct BVHNQuantizedBaseNodeIntersector1;

    template<int N, int Nx>
      struct BVHNQuantizedBaseNodeIntersector1<N, Nx, false>
    {
      static __forceinline size_t intersect(const typename BVHN<N>::QuantizedBaseNode* node, const TravRay<N,Nx,false>& ray, vfloat<Nx>& dist)
      {
        return intersectNode(node,ray,dist);
      }

      static __forceinline size_t intersect(const typename BVHN<N>::QuantizedBaseNodeMB* node, const TravRay<N,Nx,false>& ray, const float time, vfloat<N>& dist)
      {
        return intersectNode(node,ray,time,dist);
      }

    };

    template<int N, int Nx>
      struct BVHNQuantizedBaseNodeIntersector1<N, Nx, true>
    {
      static __forceinline size_t intersect(const typename BVHN<N>::QuantizedBaseNode* node, const TravRay<N,Nx,true>& ray, vfloat<Nx>& dist)
      {
        return intersectNode(node,ray,dist); 
      }

      static __forceinline size_t intersect(const typename BVHN<N>::QuantizedBaseNodeMB* node, const TravRay<N,Nx,true>& ray, const float time, vfloat<N>& dist)
      {
        return intersectNode(node,ray,time,dist);
      }

    };


  }
}
