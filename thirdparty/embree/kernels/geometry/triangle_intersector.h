// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "triangle.h"
#include "triangle_intersector_moeller.h"

namespace embree
{
  namespace isa
  {
    /*! Intersects M triangles with 1 ray */
    template<int M, bool filter>
    struct TriangleMIntersector1Moeller
    {
      typedef TriangleM<M> Primitive;
      typedef MoellerTrumboreIntersector1<M> Precalculations;

      /*! Intersect a ray with the M triangles and updates the hit. */
      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, IntersectContext* context, const TriangleM<M>& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        pre.intersectEdge(ray,tri.v0,tri.e1,tri.e2,UVIdentity<M>(),Intersect1EpilogM<M,filter>(ray,context,tri.geomID(),tri.primID()));
      }

      /*! Test if the ray is occluded by one of M triangles. */
      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, IntersectContext* context, const TriangleM<M>& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        return pre.intersectEdge(ray,tri.v0,tri.e1,tri.e2,UVIdentity<M>(),Occluded1EpilogM<M,filter>(ray,context,tri.geomID(),tri.primID()));
      }
      
      static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const Primitive& tri)
      {
        return PrimitivePointQuery1<Primitive>::pointQuery(query, context, tri);
      }
      
    };

    /*! Intersects M triangles with K rays. */
    template<int M, int K, bool filter>
    struct TriangleMIntersectorKMoeller
    {
      typedef TriangleM<M> Primitive;
      typedef MoellerTrumboreIntersectorK<M,K> Precalculations;

      /*! Intersects K rays with M triangles. */
      static __forceinline void intersect(const vbool<K>& valid_i, Precalculations& pre, RayHitK<K>& ray, IntersectContext* context, const TriangleM<M>& tri)
      {
        STAT_USER(0,TriangleM<M>::max_size());
        for (size_t i=0; i<TriangleM<M>::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(normal.trav_prims,1,popcnt(valid_i),K);
          const Vec3vf<K> p0 = broadcast<vfloat<K>>(tri.v0,i);
          const Vec3vf<K> e1 = broadcast<vfloat<K>>(tri.e1,i);
          const Vec3vf<K> e2 = broadcast<vfloat<K>>(tri.e2,i);
          pre.intersectEdgeK(valid_i,ray,p0,e1,e2,UVIdentity<K>(),IntersectKEpilogM<M,K,filter>(ray,context,tri.geomID(),tri.primID(),i));
        }
      }

      /*! Test for K rays if they are occluded by any of the M triangles. */
      static __forceinline vbool<K> occluded(const vbool<K>& valid_i, Precalculations& pre, RayK<K>& ray, IntersectContext* context, const TriangleM<M>& tri)
      {
        vbool<K> valid0 = valid_i;

        for (size_t i=0; i<TriangleM<M>::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(shadow.trav_prims,1,popcnt(valid0),K);
          const Vec3vf<K> p0 = broadcast<vfloat<K>>(tri.v0,i);
          const Vec3vf<K> e1 = broadcast<vfloat<K>>(tri.e1,i);
          const Vec3vf<K> e2 = broadcast<vfloat<K>>(tri.e2,i);
          pre.intersectEdgeK(valid0,ray,p0,e1,e2,UVIdentity<K>(),OccludedKEpilogM<M,K,filter>(valid0,ray,context,tri.geomID(),tri.primID(),i));
          if (none(valid0)) break;
        }
        return !valid0;
      }
      
      /*! Intersect a ray with M triangles and updates the hit. */
      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const TriangleM<M>& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        pre.intersectEdge(ray,k,tri.v0,tri.e1,tri.e2,UVIdentity<M>(),Intersect1KEpilogM<M,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }

      /*! Test if the ray is occluded by one of the M triangles. */
      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const TriangleM<M>& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        return pre.intersectEdge(ray,k,tri.v0,tri.e1,tri.e2,UVIdentity<M>(),Occluded1KEpilogM<M,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }
    };
  }
}
