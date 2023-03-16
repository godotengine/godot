// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node_intersector.h"

#if defined(__AVX2__)
#define __FMA_X4__
#endif

#if defined(__aarch64__)
#define __FMA_X4__
#endif


namespace embree
{
  namespace isa
  {
    //////////////////////////////////////////////////////////////////////////////////////
    // Ray structure used in single-ray traversal
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, bool robust>
      struct TravRayBase;
      
    /* Base (without tnear and tfar) */
    template<int N>
      struct TravRayBase<N,false>
    {
      __forceinline TravRayBase() {}

      __forceinline TravRayBase(const Vec3fa& ray_org, const Vec3fa& ray_dir)
        : org_xyz(ray_org), dir_xyz(ray_dir) 
      {
        const Vec3fa ray_rdir = rcp_safe(ray_dir);
        org = Vec3vf<N>(ray_org.x,ray_org.y,ray_org.z);
        dir = Vec3vf<N>(ray_dir.x,ray_dir.y,ray_dir.z);
        rdir = Vec3vf<N>(ray_rdir.x,ray_rdir.y,ray_rdir.z);
#if defined(__FMA_X4__)
        const Vec3fa ray_org_rdir = ray_org*ray_rdir;
#if !defined(__aarch64__)
        org_rdir = Vec3vf<N>(ray_org_rdir.x,ray_org_rdir.y,ray_org_rdir.z);
#else
          //for aarch64, we do not have msub equal instruction, so we negeate orig and use madd
          //x86 will use msub
        neg_org_rdir = Vec3vf<N>(-ray_org_rdir.x,-ray_org_rdir.y,-ray_org_rdir.z);
#endif
#endif
        nearX = ray_rdir.x >= 0.0f ? 0*sizeof(vfloat<N>) : 1*sizeof(vfloat<N>);
        nearY = ray_rdir.y >= 0.0f ? 2*sizeof(vfloat<N>) : 3*sizeof(vfloat<N>);
        nearZ = ray_rdir.z >= 0.0f ? 4*sizeof(vfloat<N>) : 5*sizeof(vfloat<N>);
        farX  = nearX ^ sizeof(vfloat<N>);
        farY  = nearY ^ sizeof(vfloat<N>);
        farZ  = nearZ ^ sizeof(vfloat<N>);
      }

      template<int K>
      __forceinline void init(size_t k, const Vec3vf<K>& ray_org, const Vec3vf<K>& ray_dir,
                              const Vec3vf<K>& ray_rdir, const Vec3vi<K>& nearXYZ,
                              size_t flip = sizeof(vfloat<N>))
      {
        org  = Vec3vf<N>(ray_org.x[k], ray_org.y[k], ray_org.z[k]);
        dir  = Vec3vf<N>(ray_dir.x[k], ray_dir.y[k], ray_dir.z[k]);
        rdir = Vec3vf<N>(ray_rdir.x[k], ray_rdir.y[k], ray_rdir.z[k]);
#if defined(__FMA_X4__)
#if !defined(__aarch64__)
        org_rdir = org*rdir;
#else
        neg_org_rdir = -(org*rdir);
#endif
#endif
	nearX = nearXYZ.x[k];
	nearY = nearXYZ.y[k];
	nearZ = nearXYZ.z[k];
        farX  = nearX ^ flip;
        farY  = nearY ^ flip;
        farZ  = nearZ ^ flip;
      }

      Vec3fa org_xyz, dir_xyz;
      Vec3vf<N> org, dir, rdir;
#if defined(__FMA_X4__)
#if !defined(__aarch64__)
      Vec3vf<N> org_rdir;
#else
        //aarch64 version are keeping negation of the org_rdir and use madd
        //x86 uses msub
      Vec3vf<N> neg_org_rdir;
#endif
#endif
      size_t nearX, nearY, nearZ;
      size_t farX, farY, farZ;
    };

    /* Base (without tnear and tfar) */
    template<int N>
      struct TravRayBase<N,true>
    {
      __forceinline TravRayBase() {}

      __forceinline TravRayBase(const Vec3fa& ray_org, const Vec3fa& ray_dir)
        : org_xyz(ray_org), dir_xyz(ray_dir) 
      {
        const float round_down = 1.0f-3.0f*float(ulp);
        const float round_up   = 1.0f+3.0f*float(ulp);
        const Vec3fa ray_rdir = 1.0f/zero_fix(ray_dir);
        const Vec3fa ray_rdir_near = round_down*ray_rdir;
        const Vec3fa ray_rdir_far  = round_up  *ray_rdir;
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
      }

      template<int K>
      __forceinline void init(size_t k, const Vec3vf<K>& ray_org, const Vec3vf<K>& ray_dir,
                              const Vec3vf<K>& ray_rdir, const Vec3vi<K>& nearXYZ,
                              size_t flip = sizeof(vfloat<N>))
      {
        const vfloat<N> round_down = 1.0f-3.0f*float(ulp);
        const vfloat<N> round_up   = 1.0f+3.0f*float(ulp);
        org  = Vec3vf<N>(ray_org.x[k], ray_org.y[k], ray_org.z[k]);
        dir  = Vec3vf<N>(ray_dir.x[k], ray_dir.y[k], ray_dir.z[k]);
        rdir_near = round_down*Vec3vf<N>(ray_rdir.x[k], ray_rdir.y[k], ray_rdir.z[k]);
        rdir_far  = round_up  *Vec3vf<N>(ray_rdir.x[k], ray_rdir.y[k], ray_rdir.z[k]);

	nearX = nearXYZ.x[k];
	nearY = nearXYZ.y[k];
	nearZ = nearXYZ.z[k];
        farX  = nearX ^ flip;
        farY  = nearY ^ flip;
        farZ  = nearZ ^ flip;
      }

      Vec3fa org_xyz, dir_xyz;
      Vec3vf<N> org, dir, rdir_near, rdir_far;
      size_t nearX, nearY, nearZ;
      size_t farX, farY, farZ;
    };

    /* Full (with tnear and tfar) */
    template<int N, bool robust>
      struct TravRay : TravRayBase<N,robust>
    {
      __forceinline TravRay() {}

      __forceinline TravRay(const Vec3fa& ray_org, const Vec3fa& ray_dir, float ray_tnear, float ray_tfar)
        : TravRayBase<N,robust>(ray_org, ray_dir),
          tnear(ray_tnear), tfar(ray_tfar) {}

      template<int K>
      __forceinline void init(size_t k, const Vec3vf<K>& ray_org, const Vec3vf<K>& ray_dir,
                              const Vec3vf<K>& ray_rdir, const Vec3vi<K>& nearXYZ,
                              float ray_tnear, float ray_tfar,
                              size_t flip = sizeof(vfloat<N>))
      {
        TravRayBase<N,robust>::template init<K>(k, ray_org, ray_dir, ray_rdir, nearXYZ, flip);
        tnear = ray_tnear; tfar = ray_tfar;
      }

      vfloat<N> tnear;
      vfloat<N> tfar;
    };
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Point Query structure used in single-ray traversal
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N>
    struct TravPointQuery
    {
      __forceinline TravPointQuery() {}

      __forceinline TravPointQuery(const Vec3fa& query_org, const Vec3fa& query_rad)
      {
        org = Vec3vf<N>(query_org.x, query_org.y, query_org.z);
        rad = Vec3vf<N>(query_rad.x, query_rad.y, query_rad.z);
      }

      __forceinline vfloat<N> const& tfar() const {
        return rad.x;
      }

      Vec3vf<N> org, rad;
    };
    
    //////////////////////////////////////////////////////////////////////////////////////
    // point query
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N>
    __forceinline size_t pointQuerySphereDistAndMask(
      const TravPointQuery<N>& query, vfloat<N>& dist, vfloat<N> const& minX, vfloat<N> const& maxX, 
      vfloat<N> const& minY, vfloat<N> const& maxY, vfloat<N> const& minZ, vfloat<N> const& maxZ)
    {
      const vfloat<N> vX = min(max(query.org.x, minX), maxX) - query.org.x;
      const vfloat<N> vY = min(max(query.org.y, minY), maxY) - query.org.y;
      const vfloat<N> vZ = min(max(query.org.z, minZ), maxZ) - query.org.z;
      dist = vX * vX + vY * vY + vZ * vZ;
      const vbool<N> vmask = dist <= query.tfar()*query.tfar();
      const vbool<N> valid = minX <= maxX;
      return movemask(vmask) & movemask(valid);
    }

    template<int N>
    __forceinline size_t pointQueryNodeSphere(const typename BVHN<N>::AABBNode* node, const TravPointQuery<N>& query, vfloat<N>& dist)
    {
      const vfloat<N> minX = vfloat<N>::load((float*)((const char*)&node->lower_x));
      const vfloat<N> minY = vfloat<N>::load((float*)((const char*)&node->lower_y));
      const vfloat<N> minZ = vfloat<N>::load((float*)((const char*)&node->lower_z));
      const vfloat<N> maxX = vfloat<N>::load((float*)((const char*)&node->upper_x));
      const vfloat<N> maxY = vfloat<N>::load((float*)((const char*)&node->upper_y));
      const vfloat<N> maxZ = vfloat<N>::load((float*)((const char*)&node->upper_z));
      return pointQuerySphereDistAndMask(query, dist, minX, maxX, minY, maxY, minZ, maxZ);
    }
    
    template<int N>
    __forceinline size_t pointQueryNodeSphere(const typename BVHN<N>::AABBNodeMB* node, const TravPointQuery<N>& query, const float time, vfloat<N>& dist)
    {
      const vfloat<N>* pMinX = (const vfloat<N>*)((const char*)&node->lower_x);
      const vfloat<N>* pMinY = (const vfloat<N>*)((const char*)&node->lower_y);
      const vfloat<N>* pMinZ = (const vfloat<N>*)((const char*)&node->lower_z);
      const vfloat<N>* pMaxX = (const vfloat<N>*)((const char*)&node->upper_x);
      const vfloat<N>* pMaxY = (const vfloat<N>*)((const char*)&node->upper_y);
      const vfloat<N>* pMaxZ = (const vfloat<N>*)((const char*)&node->upper_z);
      const vfloat<N> minX = madd(time,pMinX[6],vfloat<N>(pMinX[0]));
      const vfloat<N> minY = madd(time,pMinY[6],vfloat<N>(pMinY[0]));
      const vfloat<N> minZ = madd(time,pMinZ[6],vfloat<N>(pMinZ[0]));
      const vfloat<N> maxX = madd(time,pMaxX[6],vfloat<N>(pMaxX[0]));
      const vfloat<N> maxY = madd(time,pMaxY[6],vfloat<N>(pMaxY[0]));
      const vfloat<N> maxZ = madd(time,pMaxZ[6],vfloat<N>(pMaxZ[0]));
      return pointQuerySphereDistAndMask(query, dist, minX, maxX, minY, maxY, minZ, maxZ);
    }
    
    template<int N>
      __forceinline size_t pointQueryNodeSphereMB4D(const typename BVHN<N>::NodeRef ref, const TravPointQuery<N>& query, const float time, vfloat<N>& dist)
    {
      const typename BVHN<N>::AABBNodeMB* node = ref.getAABBNodeMB();
      size_t mask = pointQueryNodeSphere(node, query, time, dist);

      if (unlikely(ref.isAABBNodeMB4D())) {
        const typename BVHN<N>::AABBNodeMB4D* node1 = (const typename BVHN<N>::AABBNodeMB4D*) node;
        const vbool<N> vmask = (node1->lower_t <= time) & (time < node1->upper_t);
        mask &= movemask(vmask);
      }

      return mask;
    }
    
    template<int N>
    __forceinline size_t pointQueryNodeSphere(const typename BVHN<N>::QuantizedBaseNode* node, const TravPointQuery<N>& query, vfloat<N>& dist)
    {
      const vfloat<N> start_x(node->start.x);
      const vfloat<N> scale_x(node->scale.x);
      const vfloat<N> minX = madd(node->template dequantize<N>((0*sizeof(vfloat<N>)) >> 2),scale_x,start_x);
      const vfloat<N> maxX = madd(node->template dequantize<N>((1*sizeof(vfloat<N>)) >> 2),scale_x,start_x);
      const vfloat<N> start_y(node->start.y);
      const vfloat<N> scale_y(node->scale.y);
      const vfloat<N> minY = madd(node->template dequantize<N>((2*sizeof(vfloat<N>)) >> 2),scale_y,start_y);
      const vfloat<N> maxY = madd(node->template dequantize<N>((3*sizeof(vfloat<N>)) >> 2),scale_y,start_y);
      const vfloat<N> start_z(node->start.z);
      const vfloat<N> scale_z(node->scale.z);
      const vfloat<N> minZ = madd(node->template dequantize<N>((4*sizeof(vfloat<N>)) >> 2),scale_z,start_z);
      const vfloat<N> maxZ = madd(node->template dequantize<N>((5*sizeof(vfloat<N>)) >> 2),scale_z,start_z);
      return pointQuerySphereDistAndMask(query, dist, minX, maxX, minY, maxY, minZ, maxZ) & movemask(node->validMask());
    }
    
    template<int N>
    __forceinline size_t pointQueryNodeSphere(const typename BVHN<N>::QuantizedBaseNodeMB* node, const TravPointQuery<N>& query, const float time, vfloat<N>& dist)
    {
      const vfloat<N> minX = node->dequantizeLowerX(time);
      const vfloat<N> maxX = node->dequantizeUpperX(time);
      const vfloat<N> minY = node->dequantizeLowerY(time);
      const vfloat<N> maxY = node->dequantizeUpperY(time);
      const vfloat<N> minZ = node->dequantizeLowerZ(time);
      const vfloat<N> maxZ = node->dequantizeUpperZ(time);     
      return pointQuerySphereDistAndMask(query, dist, minX, maxX, minY, maxY, minZ, maxZ) & movemask(node->validMask());
    }
    
    template<int N>
    __forceinline size_t pointQueryNodeSphere(const typename BVHN<N>::OBBNode* node, const TravPointQuery<N>& query, vfloat<N>& dist)
    {
      // TODO: point query - implement
      const vbool<N> vmask = vbool<N>(true);
      const size_t mask = movemask(vmask) & ((1<<N)-1);
      dist = vfloat<N>(0.0f);
      return mask;
    }
    
    template<int N>
    __forceinline size_t pointQueryNodeSphere(const typename BVHN<N>::OBBNodeMB* node, const TravPointQuery<N>& query, const float time, vfloat<N>& dist)
    {
      // TODO: point query - implement
      const vbool<N> vmask = vbool<N>(true);
      const size_t mask = movemask(vmask) & ((1<<N)-1);
      dist = vfloat<N>(0.0f);
      return mask;
    }

    template<int N>
    __forceinline size_t pointQueryAABBDistAndMask(
      const TravPointQuery<N>& query, vfloat<N>& dist, vfloat<N> const& minX, vfloat<N> const& maxX, 
      vfloat<N> const& minY, vfloat<N> const& maxY, vfloat<N> const& minZ, vfloat<N> const& maxZ)
    {
      const vfloat<N> vX = min(max(query.org.x, minX), maxX) - query.org.x;
      const vfloat<N> vY = min(max(query.org.y, minY), maxY) - query.org.y;
      const vfloat<N> vZ = min(max(query.org.z, minZ), maxZ) - query.org.z;
      dist = vX * vX + vY * vY + vZ * vZ;
      const vbool<N> valid = minX <= maxX;
      const vbool<N> vmask = !((maxX < query.org.x - query.rad.x) | (minX > query.org.x + query.rad.x) |
                               (maxY < query.org.y - query.rad.y) | (minY > query.org.y + query.rad.y) |
                               (maxZ < query.org.z - query.rad.z) | (minZ > query.org.z + query.rad.z));
      return movemask(vmask) & movemask(valid);
    }

    template<int N>
    __forceinline size_t pointQueryNodeAABB(const typename BVHN<N>::AABBNode* node, const TravPointQuery<N>& query, vfloat<N>& dist)
    {
      const vfloat<N> minX = vfloat<N>::load((float*)((const char*)&node->lower_x));
      const vfloat<N> minY = vfloat<N>::load((float*)((const char*)&node->lower_y));
      const vfloat<N> minZ = vfloat<N>::load((float*)((const char*)&node->lower_z));
      const vfloat<N> maxX = vfloat<N>::load((float*)((const char*)&node->upper_x));
      const vfloat<N> maxY = vfloat<N>::load((float*)((const char*)&node->upper_y));
      const vfloat<N> maxZ = vfloat<N>::load((float*)((const char*)&node->upper_z));
      return pointQueryAABBDistAndMask(query, dist, minX, maxX, minY, maxY, minZ, maxZ);
    }
    
    template<int N>
    __forceinline size_t pointQueryNodeAABB(const typename BVHN<N>::AABBNodeMB* node, const TravPointQuery<N>& query, const float time, vfloat<N>& dist)
    {
      const vfloat<N>* pMinX = (const vfloat<N>*)((const char*)&node->lower_x);
      const vfloat<N>* pMinY = (const vfloat<N>*)((const char*)&node->lower_y);
      const vfloat<N>* pMinZ = (const vfloat<N>*)((const char*)&node->lower_z);
      const vfloat<N>* pMaxX = (const vfloat<N>*)((const char*)&node->upper_x);
      const vfloat<N>* pMaxY = (const vfloat<N>*)((const char*)&node->upper_y);
      const vfloat<N>* pMaxZ = (const vfloat<N>*)((const char*)&node->upper_z);
      const vfloat<N> minX = madd(time,pMinX[6],vfloat<N>(pMinX[0]));
      const vfloat<N> minY = madd(time,pMinY[6],vfloat<N>(pMinY[0]));
      const vfloat<N> minZ = madd(time,pMinZ[6],vfloat<N>(pMinZ[0]));
      const vfloat<N> maxX = madd(time,pMaxX[6],vfloat<N>(pMaxX[0]));
      const vfloat<N> maxY = madd(time,pMaxY[6],vfloat<N>(pMaxY[0]));
      const vfloat<N> maxZ = madd(time,pMaxZ[6],vfloat<N>(pMaxZ[0]));
      return pointQueryAABBDistAndMask(query, dist, minX, maxX, minY, maxY, minZ, maxZ);
    }
    
    template<int N>
      __forceinline size_t pointQueryNodeAABBMB4D(const typename BVHN<N>::NodeRef ref, const TravPointQuery<N>& query, const float time, vfloat<N>& dist)
    {
      const typename BVHN<N>::AABBNodeMB* node = ref.getAABBNodeMB();
      size_t mask = pointQueryNodeAABB(node, query, time, dist);

      if (unlikely(ref.isAABBNodeMB4D())) {
        const typename BVHN<N>::AABBNodeMB4D* node1 = (const typename BVHN<N>::AABBNodeMB4D*) node;
        const vbool<N> vmask = (node1->lower_t <= time) & (time < node1->upper_t);
        mask &= movemask(vmask);
      }

      return mask;
    }
    
    template<int N>
    __forceinline size_t pointQueryNodeAABB(const typename BVHN<N>::QuantizedBaseNode* node, const TravPointQuery<N>& query, vfloat<N>& dist)
    {
      const size_t mvalid  = movemask(node->validMask());
      const vfloat<N> start_x(node->start.x);
      const vfloat<N> scale_x(node->scale.x);
      const vfloat<N> minX = madd(node->template dequantize<N>((0*sizeof(vfloat<N>)) >> 2),scale_x,start_x);
      const vfloat<N> maxX = madd(node->template dequantize<N>((1*sizeof(vfloat<N>)) >> 2),scale_x,start_x);
      const vfloat<N> start_y(node->start.y);
      const vfloat<N> scale_y(node->scale.y);
      const vfloat<N> minY = madd(node->template dequantize<N>((2*sizeof(vfloat<N>)) >> 2),scale_y,start_y);
      const vfloat<N> maxY = madd(node->template dequantize<N>((3*sizeof(vfloat<N>)) >> 2),scale_y,start_y);
      const vfloat<N> start_z(node->start.z);
      const vfloat<N> scale_z(node->scale.z);
      const vfloat<N> minZ = madd(node->template dequantize<N>((4*sizeof(vfloat<N>)) >> 2),scale_z,start_z);
      const vfloat<N> maxZ = madd(node->template dequantize<N>((5*sizeof(vfloat<N>)) >> 2),scale_z,start_z);
      return pointQueryAABBDistAndMask(query, dist, minX, maxX, minY, maxY, minZ, maxZ) & mvalid;
    }
    
    template<int N>
    __forceinline size_t pointQueryNodeAABB(const typename BVHN<N>::QuantizedBaseNodeMB* node, const TravPointQuery<N>& query, const float time, vfloat<N>& dist)
    {
      const size_t mvalid  = movemask(node->validMask());
      const vfloat<N> minX = node->dequantizeLowerX(time);
      const vfloat<N> maxX = node->dequantizeUpperX(time);
      const vfloat<N> minY = node->dequantizeLowerY(time);
      const vfloat<N> maxY = node->dequantizeUpperY(time);
      const vfloat<N> minZ = node->dequantizeLowerZ(time);
      const vfloat<N> maxZ = node->dequantizeUpperZ(time);     
      return pointQueryAABBDistAndMask(query, dist, minX, maxX, minY, maxY, minZ, maxZ) & mvalid;
    }
    
    template<int N>
    __forceinline size_t pointQueryNodeAABB(const typename BVHN<N>::OBBNode* node, const TravPointQuery<N>& query, vfloat<N>& dist)
    {
      // TODO: point query - implement
      const vbool<N> vmask = vbool<N>(true);
      const size_t mask = movemask(vmask) & ((1<<N)-1);
      dist = vfloat<N>(0.0f);
      return mask;
    }
    
    template<int N>
    __forceinline size_t pointQueryNodeAABB(const typename BVHN<N>::OBBNodeMB* node, const TravPointQuery<N>& query, const float time, vfloat<N>& dist)
    {
      // TODO: point query - implement
      const vbool<N> vmask = vbool<N>(true);
      const size_t mask = movemask(vmask) & ((1<<N)-1);
      dist = vfloat<N>(0.0f);
      return mask;
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Fast AABBNode intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, bool robust>
      __forceinline size_t intersectNode(const typename BVHN<N>::AABBNode* node, const TravRay<N,robust>& ray, vfloat<N>& dist);

    template<>
      __forceinline size_t intersectNode<4>(const typename BVH4::AABBNode* node, const TravRay<4,false>& ray, vfloat4& dist)
    {
#if defined(__FMA_X4__)
#if defined(__aarch64__)
      const vfloat4 tNearX = madd(vfloat4::load((float*)((const char*)&node->lower_x+ray.nearX)), ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat4 tNearY = madd(vfloat4::load((float*)((const char*)&node->lower_x+ray.nearY)), ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat4 tNearZ = madd(vfloat4::load((float*)((const char*)&node->lower_x+ray.nearZ)), ray.rdir.z, ray.neg_org_rdir.z);
      const vfloat4 tFarX  = madd(vfloat4::load((float*)((const char*)&node->lower_x+ray.farX )), ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat4 tFarY  = madd(vfloat4::load((float*)((const char*)&node->lower_x+ray.farY )), ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat4 tFarZ  = madd(vfloat4::load((float*)((const char*)&node->lower_x+ray.farZ )), ray.rdir.z, ray.neg_org_rdir.z);
#else
      const vfloat4 tNearX = msub(vfloat4::load((float*)((const char*)&node->lower_x+ray.nearX)), ray.rdir.x, ray.org_rdir.x);
      const vfloat4 tNearY = msub(vfloat4::load((float*)((const char*)&node->lower_x+ray.nearY)), ray.rdir.y, ray.org_rdir.y);
      const vfloat4 tNearZ = msub(vfloat4::load((float*)((const char*)&node->lower_x+ray.nearZ)), ray.rdir.z, ray.org_rdir.z);
      const vfloat4 tFarX  = msub(vfloat4::load((float*)((const char*)&node->lower_x+ray.farX )), ray.rdir.x, ray.org_rdir.x);
      const vfloat4 tFarY  = msub(vfloat4::load((float*)((const char*)&node->lower_x+ray.farY )), ray.rdir.y, ray.org_rdir.y);
      const vfloat4 tFarZ  = msub(vfloat4::load((float*)((const char*)&node->lower_x+ray.farZ )), ray.rdir.z, ray.org_rdir.z);
#endif
#else
      const vfloat4 tNearX = (vfloat4::load((float*)((const char*)&node->lower_x+ray.nearX)) - ray.org.x) * ray.rdir.x;
      const vfloat4 tNearY = (vfloat4::load((float*)((const char*)&node->lower_x+ray.nearY)) - ray.org.y) * ray.rdir.y;
      const vfloat4 tNearZ = (vfloat4::load((float*)((const char*)&node->lower_x+ray.nearZ)) - ray.org.z) * ray.rdir.z;
      const vfloat4 tFarX  = (vfloat4::load((float*)((const char*)&node->lower_x+ray.farX )) - ray.org.x) * ray.rdir.x;
      const vfloat4 tFarY  = (vfloat4::load((float*)((const char*)&node->lower_x+ray.farY )) - ray.org.y) * ray.rdir.y;
      const vfloat4 tFarZ  = (vfloat4::load((float*)((const char*)&node->lower_x+ray.farZ )) - ray.org.z) * ray.rdir.z;
#endif

#if defined(__aarch64__)
      const vfloat4 tNear = maxi(tNearX, tNearY, tNearZ, ray.tnear);
      const vfloat4 tFar = mini(tFarX, tFarY, tFarZ, ray.tfar);
      const vbool4 vmask = asInt(tNear) <= asInt(tFar);
      const size_t mask = movemask(vmask);
#elif defined(__SSE4_1__) && !defined(__AVX512F__) // up to HSW
      const vfloat4 tNear = maxi(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat4 tFar  = mini(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool4 vmask = asInt(tNear) > asInt(tFar);
      const size_t mask = movemask(vmask) ^ ((1<<4)-1);
#elif defined(__AVX512F__) // SKX
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
      __forceinline size_t intersectNode<8>(const typename BVH8::AABBNode* node, const TravRay<8,false>& ray, vfloat8& dist)
    {
#if defined(__AVX2__)
#if defined(__aarch64__)
      const vfloat8 tNearX = madd(vfloat8::load((float*)((const char*)&node->lower_x+ray.nearX)), ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat8 tNearY = madd(vfloat8::load((float*)((const char*)&node->lower_x+ray.nearY)), ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat8 tNearZ = madd(vfloat8::load((float*)((const char*)&node->lower_x+ray.nearZ)), ray.rdir.z, ray.neg_org_rdir.z);
      const vfloat8 tFarX  = madd(vfloat8::load((float*)((const char*)&node->lower_x+ray.farX )), ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat8 tFarY  = madd(vfloat8::load((float*)((const char*)&node->lower_x+ray.farY )), ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat8 tFarZ  = madd(vfloat8::load((float*)((const char*)&node->lower_x+ray.farZ )), ray.rdir.z, ray.neg_org_rdir.z);
#else
      const vfloat8 tNearX = msub(vfloat8::load((float*)((const char*)&node->lower_x+ray.nearX)), ray.rdir.x, ray.org_rdir.x);
      const vfloat8 tNearY = msub(vfloat8::load((float*)((const char*)&node->lower_x+ray.nearY)), ray.rdir.y, ray.org_rdir.y);
      const vfloat8 tNearZ = msub(vfloat8::load((float*)((const char*)&node->lower_x+ray.nearZ)), ray.rdir.z, ray.org_rdir.z);
      const vfloat8 tFarX  = msub(vfloat8::load((float*)((const char*)&node->lower_x+ray.farX )), ray.rdir.x, ray.org_rdir.x);
      const vfloat8 tFarY  = msub(vfloat8::load((float*)((const char*)&node->lower_x+ray.farY )), ray.rdir.y, ray.org_rdir.y);
      const vfloat8 tFarZ  = msub(vfloat8::load((float*)((const char*)&node->lower_x+ray.farZ )), ray.rdir.z, ray.org_rdir.z);
#endif

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
#elif defined(__AVX512F__) // SKX
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

    //////////////////////////////////////////////////////////////////////////////////////
    // Robust AABBNode intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N>
      __forceinline size_t intersectNodeRobust(const typename BVHN<N>::AABBNode* node, const TravRay<N,true>& ray, vfloat<N>& dist)
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

    //////////////////////////////////////////////////////////////////////////////////////
    // Fast AABBNodeMB intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N>
      __forceinline size_t intersectNode(const typename BVHN<N>::AABBNodeMB* node, const TravRay<N,false>& ray, const float time, vfloat<N>& dist)
    {
      const vfloat<N>* pNearX = (const vfloat<N>*)((const char*)&node->lower_x+ray.nearX);
      const vfloat<N>* pNearY = (const vfloat<N>*)((const char*)&node->lower_x+ray.nearY);
      const vfloat<N>* pNearZ = (const vfloat<N>*)((const char*)&node->lower_x+ray.nearZ);
      const vfloat<N>* pFarX  = (const vfloat<N>*)((const char*)&node->lower_x+ray.farX);
      const vfloat<N>* pFarY  = (const vfloat<N>*)((const char*)&node->lower_x+ray.farY);
      const vfloat<N>* pFarZ  = (const vfloat<N>*)((const char*)&node->lower_x+ray.farZ);
#if defined(__FMA_X4__)
#if defined(__aarch64__)
      const vfloat<N> tNearX = madd(madd(time,pNearX[6],vfloat<N>(pNearX[0])), ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat<N> tNearY = madd(madd(time,pNearY[6],vfloat<N>(pNearY[0])), ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat<N> tNearZ = madd(madd(time,pNearZ[6],vfloat<N>(pNearZ[0])), ray.rdir.z, ray.neg_org_rdir.z);
      const vfloat<N> tFarX  = madd(madd(time,pFarX [6],vfloat<N>(pFarX [0])), ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat<N> tFarY  = madd(madd(time,pFarY [6],vfloat<N>(pFarY [0])), ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat<N> tFarZ  = madd(madd(time,pFarZ [6],vfloat<N>(pFarZ [0])), ray.rdir.z, ray.neg_org_rdir.z);
#else
      const vfloat<N> tNearX = msub(madd(time,pNearX[6],vfloat<N>(pNearX[0])), ray.rdir.x, ray.org_rdir.x);
      const vfloat<N> tNearY = msub(madd(time,pNearY[6],vfloat<N>(pNearY[0])), ray.rdir.y, ray.org_rdir.y);
      const vfloat<N> tNearZ = msub(madd(time,pNearZ[6],vfloat<N>(pNearZ[0])), ray.rdir.z, ray.org_rdir.z);
      const vfloat<N> tFarX  = msub(madd(time,pFarX [6],vfloat<N>(pFarX [0])), ray.rdir.x, ray.org_rdir.x);
      const vfloat<N> tFarY  = msub(madd(time,pFarY [6],vfloat<N>(pFarY [0])), ray.rdir.y, ray.org_rdir.y);
      const vfloat<N> tFarZ  = msub(madd(time,pFarZ [6],vfloat<N>(pFarZ [0])), ray.rdir.z, ray.org_rdir.z);
#endif
#else
      const vfloat<N> tNearX = (madd(time,pNearX[6],vfloat<N>(pNearX[0])) - ray.org.x) * ray.rdir.x;
      const vfloat<N> tNearY = (madd(time,pNearY[6],vfloat<N>(pNearY[0])) - ray.org.y) * ray.rdir.y;
      const vfloat<N> tNearZ = (madd(time,pNearZ[6],vfloat<N>(pNearZ[0])) - ray.org.z) * ray.rdir.z;
      const vfloat<N> tFarX  = (madd(time,pFarX [6],vfloat<N>(pFarX [0])) - ray.org.x) * ray.rdir.x;
      const vfloat<N> tFarY  = (madd(time,pFarY [6],vfloat<N>(pFarY [0])) - ray.org.y) * ray.rdir.y;
      const vfloat<N> tFarZ  = (madd(time,pFarZ [6],vfloat<N>(pFarZ [0])) - ray.org.z) * ray.rdir.z;
#endif
#if defined(__FMA_X4__) && !defined(__AVX512F__) // HSW
      const vfloat<N> tNear = maxi(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat<N> tFar  = mini(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool<N> vmask = asInt(tNear) > asInt(tFar);
      const size_t mask = movemask(vmask) ^ ((1<<N)-1);
#elif defined(__AVX512F__) // SKX
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
    // Robust AABBNodeMB intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N>
      __forceinline size_t intersectNodeRobust(const typename BVHN<N>::AABBNodeMB* node, const TravRay<N,true>& ray, const float time, vfloat<N>& dist)
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
    // Fast AABBNodeMB4D intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N>
      __forceinline size_t intersectNodeMB4D(const typename BVHN<N>::NodeRef ref, const TravRay<N,false>& ray, const float time, vfloat<N>& dist)
    {
      const typename BVHN<N>::AABBNodeMB* node = ref.getAABBNodeMB();
        
      const vfloat<N>* pNearX = (const vfloat<N>*)((const char*)&node->lower_x+ray.nearX);
      const vfloat<N>* pNearY = (const vfloat<N>*)((const char*)&node->lower_x+ray.nearY);
      const vfloat<N>* pNearZ = (const vfloat<N>*)((const char*)&node->lower_x+ray.nearZ);
      const vfloat<N>* pFarX  = (const vfloat<N>*)((const char*)&node->lower_x+ray.farX);
      const vfloat<N>* pFarY  = (const vfloat<N>*)((const char*)&node->lower_x+ray.farY);
      const vfloat<N>* pFarZ  = (const vfloat<N>*)((const char*)&node->lower_x+ray.farZ);
#if defined (__FMA_X4__)
#if defined(__aarch64__)
      const vfloat<N> tNearX = madd(madd(time,pNearX[6],vfloat<N>(pNearX[0])), ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat<N> tNearY = madd(madd(time,pNearY[6],vfloat<N>(pNearY[0])), ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat<N> tNearZ = madd(madd(time,pNearZ[6],vfloat<N>(pNearZ[0])), ray.rdir.z, ray.neg_org_rdir.z);
      const vfloat<N> tFarX  = madd(madd(time,pFarX [6],vfloat<N>(pFarX [0])), ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat<N> tFarY  = madd(madd(time,pFarY [6],vfloat<N>(pFarY [0])), ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat<N> tFarZ  = madd(madd(time,pFarZ [6],vfloat<N>(pFarZ [0])), ray.rdir.z, ray.neg_org_rdir.z);
#else
      const vfloat<N> tNearX = msub(madd(time,pNearX[6],vfloat<N>(pNearX[0])), ray.rdir.x, ray.org_rdir.x);
      const vfloat<N> tNearY = msub(madd(time,pNearY[6],vfloat<N>(pNearY[0])), ray.rdir.y, ray.org_rdir.y);
      const vfloat<N> tNearZ = msub(madd(time,pNearZ[6],vfloat<N>(pNearZ[0])), ray.rdir.z, ray.org_rdir.z);
      const vfloat<N> tFarX  = msub(madd(time,pFarX [6],vfloat<N>(pFarX [0])), ray.rdir.x, ray.org_rdir.x);
      const vfloat<N> tFarY  = msub(madd(time,pFarY [6],vfloat<N>(pFarY [0])), ray.rdir.y, ray.org_rdir.y);
      const vfloat<N> tFarZ  = msub(madd(time,pFarZ [6],vfloat<N>(pFarZ [0])), ray.rdir.z, ray.org_rdir.z);
#endif
#else
      const vfloat<N> tNearX = (madd(time,pNearX[6],vfloat<N>(pNearX[0])) - ray.org.x) * ray.rdir.x;
      const vfloat<N> tNearY = (madd(time,pNearY[6],vfloat<N>(pNearY[0])) - ray.org.y) * ray.rdir.y;
      const vfloat<N> tNearZ = (madd(time,pNearZ[6],vfloat<N>(pNearZ[0])) - ray.org.z) * ray.rdir.z;
      const vfloat<N> tFarX  = (madd(time,pFarX [6],vfloat<N>(pFarX [0])) - ray.org.x) * ray.rdir.x;
      const vfloat<N> tFarY  = (madd(time,pFarY [6],vfloat<N>(pFarY [0])) - ray.org.y) * ray.rdir.y;
      const vfloat<N> tFarZ  = (madd(time,pFarZ [6],vfloat<N>(pFarZ [0])) - ray.org.z) * ray.rdir.z;
#endif
#if defined(__FMA_X4__) && !defined(__AVX512F__)
      const vfloat<N> tNear = maxi(maxi(tNearX,tNearY),maxi(tNearZ,ray.tnear));
      const vfloat<N> tFar  = mini(mini(tFarX ,tFarY ),mini(tFarZ ,ray.tfar ));
#else
      const vfloat<N> tNear = max(ray.tnear,tNearX,tNearY,tNearZ);
      const vfloat<N> tFar  = min(ray.tfar, tFarX ,tFarY ,tFarZ );
#endif
      vbool<N> vmask = tNear <= tFar;
      if (unlikely(ref.isAABBNodeMB4D())) {
        const typename BVHN<N>::AABBNodeMB4D* node1 = (const typename BVHN<N>::AABBNodeMB4D*) node;
        vmask &= (node1->lower_t <= time) & (time < node1->upper_t);
      }
      const size_t mask = movemask(vmask);
      dist = tNear;
      return mask;
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Robust AABBNodeMB4D intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N>
      __forceinline size_t intersectNodeMB4DRobust(const typename BVHN<N>::NodeRef ref, const TravRay<N,true>& ray, const float time, vfloat<N>& dist)
    {
      const typename BVHN<N>::AABBNodeMB* node = ref.getAABBNodeMB();

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
      if (unlikely(ref.isAABBNodeMB4D())) {
        const typename BVHN<N>::AABBNodeMB4D* node1 = (const typename BVHN<N>::AABBNodeMB4D*) node;
        vmask &= (node1->lower_t <= time) & (time < node1->upper_t);
      }
      const size_t mask = movemask(vmask);
      dist = tNear;
      return mask;
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Fast QuantizedBaseNode intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, bool robust>
      __forceinline size_t intersectNode(const typename BVHN<N>::QuantizedBaseNode* node, const TravRay<N,robust>& ray, vfloat<N>& dist);

    template<>
      __forceinline size_t intersectNode<4>(const typename BVH4::QuantizedBaseNode* node, const TravRay<4,false>& ray, vfloat4& dist)
    {
      const size_t mvalid  = movemask(node->validMask());
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

#if defined(__FMA_X4__)
#if defined(__aarch64__)
      const vfloat4 tNearX = madd(lower_x, ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat4 tNearY = madd(lower_y, ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat4 tNearZ = madd(lower_z, ray.rdir.z, ray.neg_org_rdir.z);
      const vfloat4 tFarX  = madd(upper_x, ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat4 tFarY  = madd(upper_y, ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat4 tFarZ  = madd(upper_z, ray.rdir.z, ray.neg_org_rdir.z);
#else
      const vfloat4 tNearX = msub(lower_x, ray.rdir.x, ray.org_rdir.x);
      const vfloat4 tNearY = msub(lower_y, ray.rdir.y, ray.org_rdir.y);
      const vfloat4 tNearZ = msub(lower_z, ray.rdir.z, ray.org_rdir.z);
      const vfloat4 tFarX  = msub(upper_x, ray.rdir.x, ray.org_rdir.x);
      const vfloat4 tFarY  = msub(upper_y, ray.rdir.y, ray.org_rdir.y);
      const vfloat4 tFarZ  = msub(upper_z, ray.rdir.z, ray.org_rdir.z);
#endif
#else
      const vfloat4 tNearX = (lower_x - ray.org.x) * ray.rdir.x;
      const vfloat4 tNearY = (lower_y - ray.org.y) * ray.rdir.y;
      const vfloat4 tNearZ = (lower_z - ray.org.z) * ray.rdir.z;
      const vfloat4 tFarX  = (upper_x - ray.org.x) * ray.rdir.x;
      const vfloat4 tFarY  = (upper_y - ray.org.y) * ray.rdir.y;
      const vfloat4 tFarZ  = (upper_z - ray.org.z) * ray.rdir.z;
#endif
      
#if defined(__aarch64__) || defined(__SSE4_1__) && !defined(__AVX512F__) // up to HSW
      const vfloat4 tNear = maxi(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat4 tFar  = mini(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool4 vmask = asInt(tNear) > asInt(tFar);
      const size_t mask = movemask(vmask) ^ ((1<<4)-1);
#elif defined(__AVX512F__) // SKX
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
      return mask & mvalid;
    }

    template<>
      __forceinline size_t intersectNode<4>(const typename BVH4::QuantizedBaseNode* node, const TravRay<4,true>& ray, vfloat4& dist)
    {
      const size_t mvalid  = movemask(node->validMask());
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

      const vfloat4 tNearX = (lower_x - ray.org.x) * ray.rdir_near.x;
      const vfloat4 tNearY = (lower_y - ray.org.y) * ray.rdir_near.y;
      const vfloat4 tNearZ = (lower_z - ray.org.z) * ray.rdir_near.z;
      const vfloat4 tFarX  = (upper_x - ray.org.x) * ray.rdir_far.x;
      const vfloat4 tFarY  = (upper_y - ray.org.y) * ray.rdir_far.y;
      const vfloat4 tFarZ  = (upper_z - ray.org.z) * ray.rdir_far.z;
      
      const vfloat4 tNear = max(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat4 tFar  = min(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool4 vmask = tNear <= tFar;
      const size_t mask = movemask(vmask);
      dist = tNear;
      return mask & mvalid;
    }


#if defined(__AVX__)

    template<>
      __forceinline size_t intersectNode<8>(const typename BVH8::QuantizedBaseNode* node, const TravRay<8,false>& ray, vfloat8& dist)
    {
      const size_t mvalid  = movemask(node->validMask());
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
#if defined(__aarch64__)
      const vfloat8 tNearX = madd(lower_x, ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat8 tNearY = madd(lower_y, ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat8 tNearZ = madd(lower_z, ray.rdir.z, ray.neg_org_rdir.z);
      const vfloat8 tFarX  = madd(upper_x, ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat8 tFarY  = madd(upper_y, ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat8 tFarZ  = madd(upper_z, ray.rdir.z, ray.neg_org_rdir.z);
#else
      const vfloat8 tNearX = msub(lower_x, ray.rdir.x, ray.org_rdir.x);
      const vfloat8 tNearY = msub(lower_y, ray.rdir.y, ray.org_rdir.y);
      const vfloat8 tNearZ = msub(lower_z, ray.rdir.z, ray.org_rdir.z);
      const vfloat8 tFarX  = msub(upper_x, ray.rdir.x, ray.org_rdir.x);
      const vfloat8 tFarY  = msub(upper_y, ray.rdir.y, ray.org_rdir.y);
      const vfloat8 tFarZ  = msub(upper_z, ray.rdir.z, ray.org_rdir.z);
#endif
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
#elif defined(__AVX512F__) // SKX
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
      return mask & mvalid;
    }

    template<>
      __forceinline size_t intersectNode<8>(const typename BVH8::QuantizedBaseNode* node, const TravRay<8,true>& ray, vfloat8& dist)
    {
      const size_t mvalid  = movemask(node->validMask());
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

      const vfloat8 tNearX = (lower_x - ray.org.x) * ray.rdir_near.x;
      const vfloat8 tNearY = (lower_y - ray.org.y) * ray.rdir_near.y;
      const vfloat8 tNearZ = (lower_z - ray.org.z) * ray.rdir_near.z;
      const vfloat8 tFarX  = (upper_x - ray.org.x) * ray.rdir_far.x;
      const vfloat8 tFarY  = (upper_y - ray.org.y) * ray.rdir_far.y;
      const vfloat8 tFarZ  = (upper_z - ray.org.z) * ray.rdir_far.z;
      
      const vfloat8 tNear = max(tNearX,tNearY,tNearZ,ray.tnear);
      const vfloat8 tFar  = min(tFarX ,tFarY ,tFarZ ,ray.tfar);
      const vbool8 vmask = tNear <= tFar;
      const size_t mask = movemask(vmask);

      dist = tNear;
      return mask & mvalid;
    }


#endif

    template<int N>
      __forceinline size_t intersectNode(const typename BVHN<N>::QuantizedBaseNodeMB* node, const TravRay<N,false>& ray, const float time, vfloat<N>& dist)
    {
      const vboolf<N> mvalid    = node->validMask();
      const vfloat<N> lower_x   = node->dequantizeLowerX(time);
      const vfloat<N> upper_x   = node->dequantizeUpperX(time);
      const vfloat<N> lower_y   = node->dequantizeLowerY(time);
      const vfloat<N> upper_y   = node->dequantizeUpperY(time);
      const vfloat<N> lower_z   = node->dequantizeLowerZ(time);
      const vfloat<N> upper_z   = node->dequantizeUpperZ(time);     
#if defined(__FMA_X4__)
#if defined(__aarch64__)
      const vfloat<N> tNearX = madd(lower_x, ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat<N> tNearY = madd(lower_y, ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat<N> tNearZ = madd(lower_z, ray.rdir.z, ray.neg_org_rdir.z);
      const vfloat<N> tFarX  = madd(upper_x, ray.rdir.x, ray.neg_org_rdir.x);
      const vfloat<N> tFarY  = madd(upper_y, ray.rdir.y, ray.neg_org_rdir.y);
      const vfloat<N> tFarZ  = madd(upper_z, ray.rdir.z, ray.neg_org_rdir.z);
#else
      const vfloat<N> tNearX = msub(lower_x, ray.rdir.x, ray.org_rdir.x);
      const vfloat<N> tNearY = msub(lower_y, ray.rdir.y, ray.org_rdir.y);
      const vfloat<N> tNearZ = msub(lower_z, ray.rdir.z, ray.org_rdir.z);
      const vfloat<N> tFarX  = msub(upper_x, ray.rdir.x, ray.org_rdir.x);
      const vfloat<N> tFarY  = msub(upper_y, ray.rdir.y, ray.org_rdir.y);
      const vfloat<N> tFarZ  = msub(upper_z, ray.rdir.z, ray.org_rdir.z);
#endif
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
#if defined(__AVX512F__) // SKX
      const vbool<N> vmask =  le(mvalid,asInt(tNear),asInt(tFar));
#else
      const vbool<N> vmask = (asInt(tNear) <= asInt(tFar)) & mvalid;
#endif
      const size_t mask = movemask(vmask);
      dist = tNear;
      return mask;      
    }

    template<int N>
      __forceinline size_t intersectNode(const typename BVHN<N>::QuantizedBaseNodeMB* node, const TravRay<N,true>& ray, const float time, vfloat<N>& dist)
    {
      const vboolf<N> mvalid    = node->validMask();
      const vfloat<N> lower_x   = node->dequantizeLowerX(time);
      const vfloat<N> upper_x   = node->dequantizeUpperX(time);
      const vfloat<N> lower_y   = node->dequantizeLowerY(time);
      const vfloat<N> upper_y   = node->dequantizeUpperY(time);
      const vfloat<N> lower_z   = node->dequantizeLowerZ(time);
      const vfloat<N> upper_z   = node->dequantizeUpperZ(time);     
      const vfloat<N> tNearX = (lower_x - ray.org.x) * ray.rdir_near.x;
      const vfloat<N> tNearY = (lower_y - ray.org.y) * ray.rdir_near.y;
      const vfloat<N> tNearZ = (lower_z - ray.org.z) * ray.rdir_near.z;
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
#if defined(__AVX512F__) // SKX
      const vbool<N> vmask =  le(mvalid,asInt(tNear),asInt(tFar));
#else
      const vbool<N> vmask = (asInt(tNear) <= asInt(tFar)) & mvalid;
#endif
      const size_t mask = movemask(vmask);
      dist = tNear;
      return mask;      
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Fast OBBNode intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, bool robust>
      __forceinline size_t intersectNode(const typename BVHN<N>::OBBNode* node, const TravRay<N,robust>& ray, vfloat<N>& dist)
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
      vfloat<N> tNear  = max(ray.tnear, tNearX,tNearY,tNearZ);
      vfloat<N> tFar   = min(ray.tfar,  tFarX ,tFarY ,tFarZ );
      if (robust) {
        tNear = tNear*vfloat<N>(1.0f-3.0f*float(ulp));
        tFar  = tFar *vfloat<N>(1.0f+3.0f*float(ulp));
      }
      const vbool<N> vmask = tNear <= tFar;
      dist = tNear;
      return movemask(vmask);
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // Fast OBBNodeMB intersection
    //////////////////////////////////////////////////////////////////////////////////////

    template<int N, bool robust>
      __forceinline size_t intersectNode(const typename BVHN<N>::OBBNodeMB* node, const TravRay<N,robust>& ray, const float time, vfloat<N>& dist)
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
      vfloat<N> tNear  = max(ray.tnear, tNearX,tNearY,tNearZ);
      vfloat<N> tFar   = min(ray.tfar,  tFarX ,tFarY ,tFarZ );
      if (robust) {
        tNear = tNear*vfloat<N>(1.0f-3.0f*float(ulp));
        tFar  = tFar *vfloat<N>(1.0f+3.0f*float(ulp));
      }
      const vbool<N> vmask = tNear <= tFar;
      dist = tNear;
      return movemask(vmask);
    }
    
    //////////////////////////////////////////////////////////////////////////////////////
    // Node intersectors used in point query raversal
    //////////////////////////////////////////////////////////////////////////////////////
    
    /*! Computes traversal information for N nodes with 1 point query */
    template<int N, int types>
    struct BVHNNodePointQuerySphere1;

    template<int N>
    struct BVHNNodePointQuerySphere1<N, BVH_AN1>
    {
      static __forceinline bool pointQuery(const typename BVHN<N>::NodeRef& node, const TravPointQuery<N>& query, float time, vfloat<N>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = pointQueryNodeSphere(node.getAABBNode(), query, dist);
        return true;
      }
    };

    template<int N>
    struct BVHNNodePointQuerySphere1<N, BVH_AN2>
    {
      static __forceinline bool pointQuery(const typename BVHN<N>::NodeRef& node, const TravPointQuery<N>& query, float time, vfloat<N>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = pointQueryNodeSphere(node.getAABBNodeMB(), query, time, dist);
        return true;
      }
    };

    template<int N>
    struct BVHNNodePointQuerySphere1<N, BVH_AN2_AN4D>
    {
      static __forceinline bool pointQuery(const typename BVHN<N>::NodeRef& node, const TravPointQuery<N>& query, float time, vfloat<N>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = pointQueryNodeSphereMB4D<N>(node, query, time, dist);
        return true;
      }
    };

    template<int N>
    struct BVHNNodePointQuerySphere1<N, BVH_AN1_UN1>
    {
      static __forceinline bool pointQuery(const typename BVHN<N>::NodeRef& node, const TravPointQuery<N>& query, float time, vfloat<N>& dist, size_t& mask)
      {
        if (likely(node.isAABBNode()))          mask = pointQueryNodeSphere(node.getAABBNode(), query, dist);
        else if (unlikely(node.isOBBNode())) mask = pointQueryNodeSphere(node.ungetAABBNode(), query, dist);
        else return false;
        return true;
      }
    };
    
    template<int N>
    struct BVHNNodePointQuerySphere1<N, BVH_AN2_UN2>
    {
      static __forceinline bool pointQuery(const typename BVHN<N>::NodeRef& node, const TravPointQuery<N>& query, float time, vfloat<N>& dist, size_t& mask)
      {
        if (likely(node.isAABBNodeMB()))           mask = pointQueryNodeSphere(node.getAABBNodeMB(), query, time, dist);
        else if (unlikely(node.isOBBNodeMB()))  mask = pointQueryNodeSphere(node.ungetAABBNodeMB(), query, time, dist);
        else return false;
        return true;
      }
    };

    template<int N>
    struct BVHNNodePointQuerySphere1<N, BVH_AN2_AN4D_UN2>
    {
      static __forceinline bool pointQuery(const typename BVHN<N>::NodeRef& node, const TravPointQuery<N>& query, float time, vfloat<N>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        if (unlikely(node.isOBBNodeMB())) mask = pointQueryNodeSphere(node.ungetAABBNodeMB(), query, time, dist);
        else                                    mask = pointQueryNodeSphereMB4D(node, query, time, dist);
        return true;
      }
    };

    template<int N>
    struct BVHNNodePointQuerySphere1<N, BVH_QN1>
    {
      static __forceinline bool pointQuery(const typename BVHN<N>::NodeRef& node, const TravPointQuery<N>& query, float time, vfloat<N>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = pointQueryNodeSphere((const typename BVHN<N>::QuantizedNode*)node.quantizedNode(), query, dist);
        return true;
      }
    };
    
    template<int N>
    struct BVHNQuantizedBaseNodePointQuerySphere1
    {
      static __forceinline size_t pointQuery(const typename BVHN<N>::QuantizedBaseNode* node, const TravPointQuery<N>& query, vfloat<N>& dist)
      {
        return pointQueryNodeSphere(node,query,dist);
      }

      static __forceinline size_t pointQuery(const typename BVHN<N>::QuantizedBaseNodeMB* node, const TravPointQuery<N>& query, const float time, vfloat<N>& dist)
      {
        return pointQueryNodeSphere(node,query,time,dist);
      }
    };

    /*! Computes traversal information for N nodes with 1 point query */
    template<int N, int types>
    struct BVHNNodePointQueryAABB1;

    template<int N>
    struct BVHNNodePointQueryAABB1<N, BVH_AN1>
    {
      static __forceinline bool pointQuery(const typename BVHN<N>::NodeRef& node, const TravPointQuery<N>& query, float time, vfloat<N>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = pointQueryNodeAABB(node.getAABBNode(), query, dist);
        return true;
      }
    };

    template<int N>
    struct BVHNNodePointQueryAABB1<N, BVH_AN2>
    {
      static __forceinline bool pointQuery(const typename BVHN<N>::NodeRef& node, const TravPointQuery<N>& query, float time, vfloat<N>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = pointQueryNodeAABB(node.getAABBNodeMB(), query, time, dist);
        return true;
      }
    };

    template<int N>
    struct BVHNNodePointQueryAABB1<N, BVH_AN2_AN4D>
    {
      static __forceinline bool pointQuery(const typename BVHN<N>::NodeRef& node, const TravPointQuery<N>& query, float time, vfloat<N>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = pointQueryNodeAABBMB4D<N>(node, query, time, dist);
        return true;
      }
    };

    template<int N>
    struct BVHNNodePointQueryAABB1<N, BVH_AN1_UN1>
    {
      static __forceinline bool pointQuery(const typename BVHN<N>::NodeRef& node, const TravPointQuery<N>& query, float time, vfloat<N>& dist, size_t& mask)
      {
        if (likely(node.isAABBNode()))          mask = pointQueryNodeAABB(node.getAABBNode(), query, dist);
        else if (unlikely(node.isOBBNode())) mask = pointQueryNodeAABB(node.ungetAABBNode(), query, dist);
        else return false;
        return true;
      }
    };
    
    template<int N>
    struct BVHNNodePointQueryAABB1<N, BVH_AN2_UN2>
    {
      static __forceinline bool pointQuery(const typename BVHN<N>::NodeRef& node, const TravPointQuery<N>& query, float time, vfloat<N>& dist, size_t& mask)
      {
        if (likely(node.isAABBNodeMB()))           mask = pointQueryNodeAABB(node.getAABBNodeMB(), query, time, dist);
        else if (unlikely(node.isOBBNodeMB()))  mask = pointQueryNodeAABB(node.ungetAABBNodeMB(), query, time, dist);
        else return false;
        return true;
      }
    };

    template<int N>
    struct BVHNNodePointQueryAABB1<N, BVH_AN2_AN4D_UN2>
    {
      static __forceinline bool pointQuery(const typename BVHN<N>::NodeRef& node, const TravPointQuery<N>& query, float time, vfloat<N>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        if (unlikely(node.isOBBNodeMB())) mask = pointQueryNodeAABB(node.ungetAABBNodeMB(), query, time, dist);
        else                                    mask = pointQueryNodeAABBMB4D(node, query, time, dist);
        return true;
      }
    };

    template<int N>
    struct BVHNNodePointQueryAABB1<N, BVH_QN1>
    {
      static __forceinline bool pointQuery(const typename BVHN<N>::NodeRef& node, const TravPointQuery<N>& query, float time, vfloat<N>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = pointQueryNodeAABB((const typename BVHN<N>::QuantizedNode*)node.quantizedNode(), query, dist);
        return true;
      }
    };
    
    template<int N>
    struct BVHNQuantizedBaseNodePointQueryAABB1
    {
      static __forceinline size_t pointQuery(const typename BVHN<N>::QuantizedBaseNode* node, const TravPointQuery<N>& query, vfloat<N>& dist)
      {
        return pointQueryNodeAABB(node,query,dist);
      }

      static __forceinline size_t pointQuery(const typename BVHN<N>::QuantizedBaseNodeMB* node, const TravPointQuery<N>& query, const float time, vfloat<N>& dist)
      {
        return pointQueryNodeAABB(node,query,time,dist);
      }
    };

    
    //////////////////////////////////////////////////////////////////////////////////////
    // Node intersectors used in ray traversal
    //////////////////////////////////////////////////////////////////////////////////////

    /*! Intersects N nodes with 1 ray */
    template<int N, int types, bool robust>
    struct BVHNNodeIntersector1;

    template<int N>
    struct BVHNNodeIntersector1<N, BVH_AN1, false>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,false>& ray, float time, vfloat<N>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = intersectNode(node.getAABBNode(), ray, dist);
        return true;
      }
    };

    template<int N>
    struct BVHNNodeIntersector1<N, BVH_AN1, true>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,true>& ray, float time, vfloat<N>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = intersectNodeRobust(node.getAABBNode(), ray, dist);
        return true;
      }
    };

    template<int N>
    struct BVHNNodeIntersector1<N, BVH_AN2, false>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,false>& ray, float time, vfloat<N>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = intersectNode(node.getAABBNodeMB(), ray, time, dist);
        return true;
      }
    };

    template<int N>
    struct BVHNNodeIntersector1<N, BVH_AN2, true>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,true>& ray, float time, vfloat<N>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = intersectNodeRobust(node.getAABBNodeMB(), ray, time, dist);
        return true;
      }
    };

    template<int N>
    struct BVHNNodeIntersector1<N, BVH_AN2_AN4D, false>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,false>& ray, float time, vfloat<N>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = intersectNodeMB4D<N>(node, ray, time, dist);
        return true;
      }
    };

    template<int N>
    struct BVHNNodeIntersector1<N, BVH_AN2_AN4D, true>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,true>& ray, float time, vfloat<N>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = intersectNodeMB4DRobust<N>(node, ray, time, dist);
        return true;
      }
    };

    template<int N>
    struct BVHNNodeIntersector1<N, BVH_AN1_UN1, false>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,false>& ray, float time, vfloat<N>& dist, size_t& mask)
      {
        if (likely(node.isAABBNode()))          mask = intersectNode(node.getAABBNode(), ray, dist);
        else if (unlikely(node.isOBBNode())) mask = intersectNode(node.ungetAABBNode(), ray, dist);
        else return false;
        return true;
      }
    };

    template<int N>
    struct BVHNNodeIntersector1<N, BVH_AN1_UN1, true>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,true>& ray, float time, vfloat<N>& dist, size_t& mask)
      {
        if (likely(node.isAABBNode()))          mask = intersectNodeRobust(node.getAABBNode(), ray, dist);
        else if (unlikely(node.isOBBNode())) mask = intersectNode(node.ungetAABBNode(), ray, dist);
        else return false;
        return true;
      }
    };

    template<int N>
    struct BVHNNodeIntersector1<N, BVH_AN2_UN2, false>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,false>& ray, float time, vfloat<N>& dist, size_t& mask)
      {
        if (likely(node.isAABBNodeMB()))           mask = intersectNode(node.getAABBNodeMB(), ray, time, dist);
        else if (unlikely(node.isOBBNodeMB()))  mask = intersectNode(node.ungetAABBNodeMB(), ray, time, dist);
        else return false;
        return true;
      }
    };

    template<int N>
    struct BVHNNodeIntersector1<N, BVH_AN2_UN2, true>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,true>& ray, float time, vfloat<N>& dist, size_t& mask)
      {
        if (likely(node.isAABBNodeMB()))           mask = intersectNodeRobust(node.getAABBNodeMB(), ray, time, dist);
        else if (unlikely(node.isOBBNodeMB()))  mask = intersectNode(node.ungetAABBNodeMB(), ray, time, dist);
        else return false;
        return true;
      }
    };

    template<int N>
    struct BVHNNodeIntersector1<N, BVH_AN2_AN4D_UN2, false>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,false>& ray, float time, vfloat<N>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        if (unlikely(node.isOBBNodeMB())) mask = intersectNode(node.ungetAABBNodeMB(), ray, time, dist);
        else                                    mask = intersectNodeMB4D(node, ray, time, dist);
        return true;
      }
    };

    template<int N>
    struct BVHNNodeIntersector1<N, BVH_AN2_AN4D_UN2, true>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,true>& ray, float time, vfloat<N>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        if (unlikely(node.isOBBNodeMB())) mask = intersectNode(node.ungetAABBNodeMB(), ray, time, dist);
        else                                    mask = intersectNodeMB4DRobust(node, ray, time, dist);
        return true;
      }
    };

    template<int N>
    struct BVHNNodeIntersector1<N, BVH_QN1, false>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,false>& ray, float time, vfloat<N>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = intersectNode((const typename BVHN<N>::QuantizedNode*)node.quantizedNode(), ray, dist);
        return true;
      }
    };

    template<int N>
    struct BVHNNodeIntersector1<N, BVH_QN1, true>
    {
      static __forceinline bool intersect(const typename BVHN<N>::NodeRef& node, const TravRay<N,true>& ray, float time, vfloat<N>& dist, size_t& mask)
      {
        if (unlikely(node.isLeaf())) return false;
        mask = intersectNodeRobust((const typename BVHN<N>::QuantizedNode*)node.quantizedNode(), ray, dist);
        return true;
      }
    };

    /*! Intersects N nodes with K rays */
    template<int N, bool robust>
      struct BVHNQuantizedBaseNodeIntersector1;

    template<int N>
      struct BVHNQuantizedBaseNodeIntersector1<N, false>
    {
      static __forceinline size_t intersect(const typename BVHN<N>::QuantizedBaseNode* node, const TravRay<N,false>& ray, vfloat<N>& dist)
      {
        return intersectNode(node,ray,dist);
      }

      static __forceinline size_t intersect(const typename BVHN<N>::QuantizedBaseNodeMB* node, const TravRay<N,false>& ray, const float time, vfloat<N>& dist)
      {
        return intersectNode(node,ray,time,dist);
      }

    };

    template<int N>
      struct BVHNQuantizedBaseNodeIntersector1<N, true>
    {
      static __forceinline size_t intersect(const typename BVHN<N>::QuantizedBaseNode* node, const TravRay<N,true>& ray, vfloat<N>& dist)
      {
        return intersectNode(node,ray,dist); 
      }

      static __forceinline size_t intersect(const typename BVHN<N>::QuantizedBaseNodeMB* node, const TravRay<N,true>& ray, const float time, vfloat<N>& dist)
      {
        return intersectNode(node,ray,time,dist);
      }

    };


  }
}
