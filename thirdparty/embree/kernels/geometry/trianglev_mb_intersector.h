// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "triangle.h"
#include "intersector_epilog.h"

namespace embree
{
  namespace isa
  {
    /*! Intersects M motion blur triangles with 1 ray */
    template<int M, bool filter>
    struct TriangleMvMBIntersector1Moeller
    {
      typedef TriangleMvMB<M> Primitive;
      typedef MoellerTrumboreIntersector1<M> Precalculations;

      /*! Intersect a ray with the M triangles and updates the hit. */
      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, RayQueryContext* context, const TriangleMvMB<M>& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        const Vec3vf<M> time(ray.time());
        const Vec3vf<M> v0 = madd(time,Vec3vf<M>(tri.dv0),Vec3vf<M>(tri.v0));
        const Vec3vf<M> v1 = madd(time,Vec3vf<M>(tri.dv1),Vec3vf<M>(tri.v1));
        const Vec3vf<M> v2 = madd(time,Vec3vf<M>(tri.dv2),Vec3vf<M>(tri.v2));
        pre.intersect(ray,v0,v1,v2,Intersect1EpilogM<M,filter>(ray,context,tri.geomID(),tri.primID()));
      }

      /*! Test if the ray is occluded by one of M triangles. */
      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, RayQueryContext* context, const TriangleMvMB<M>& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const Vec3vf<M> time(ray.time());
        const Vec3vf<M> v0 = madd(time,Vec3vf<M>(tri.dv0),Vec3vf<M>(tri.v0));
        const Vec3vf<M> v1 = madd(time,Vec3vf<M>(tri.dv1),Vec3vf<M>(tri.v1));
        const Vec3vf<M> v2 = madd(time,Vec3vf<M>(tri.dv2),Vec3vf<M>(tri.v2));
        return pre.intersect(ray,v0,v1,v2,Occluded1EpilogM<M,filter>(ray,context,tri.geomID(),tri.primID()));
      }
      
      static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const Primitive& tri)
      {
        return PrimitivePointQuery1<Primitive>::pointQuery(query, context, tri);
      }
    };
    
    /*! Intersects M motion blur triangles with K rays. */
    template<int M, int K, bool filter>
    struct TriangleMvMBIntersectorKMoeller
    {
      typedef TriangleMvMB<M> Primitive;
      typedef MoellerTrumboreIntersectorK<M,K> Precalculations;

      /*! Intersects K rays with M triangles. */
      static __forceinline void intersect(const vbool<K>& valid_i, Precalculations& pre, RayHitK<K>& ray, RayQueryContext* context, const TriangleMvMB<M>& tri)
      {
        for (size_t i=0; i<TriangleMvMB<M>::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(normal.trav_prims,1,popcnt(valid_i),K);
          const Vec3vf<K> time(ray.time());
          const Vec3vf<K> v0 = madd(time,broadcast<vfloat<K>>(tri.dv0,i),broadcast<vfloat<K>>(tri.v0,i));
          const Vec3vf<K> v1 = madd(time,broadcast<vfloat<K>>(tri.dv1,i),broadcast<vfloat<K>>(tri.v1,i));
          const Vec3vf<K> v2 = madd(time,broadcast<vfloat<K>>(tri.dv2,i),broadcast<vfloat<K>>(tri.v2,i));
          pre.intersectK(valid_i,ray,v0,v1,v2,IntersectKEpilogM<M,K,filter>(ray,context,tri.geomID(),tri.primID(),i));
        }
      }

      /*! Test for K rays if they are occluded by any of the M triangles. */
      static __forceinline vbool<K> occluded(const vbool<K>& valid_i, Precalculations& pre, RayK<K>& ray, RayQueryContext* context, const TriangleMvMB<M>& tri)
      {
        vbool<K> valid0 = valid_i;

        for (size_t i=0; i<TriangleMvMB<M>::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(shadow.trav_prims,1,popcnt(valid0),K);
          const Vec3vf<K> time(ray.time());
          const Vec3vf<K> v0 = madd(time,broadcast<vfloat<K>>(tri.dv0,i),broadcast<vfloat<K>>(tri.v0,i));
          const Vec3vf<K> v1 = madd(time,broadcast<vfloat<K>>(tri.dv1,i),broadcast<vfloat<K>>(tri.v1,i));
          const Vec3vf<K> v2 = madd(time,broadcast<vfloat<K>>(tri.dv2,i),broadcast<vfloat<K>>(tri.v2,i));
          pre.intersectK(valid0,ray,v0,v1,v2,OccludedKEpilogM<M,K,filter>(valid0,ray,context,tri.geomID(),tri.primID(),i));
          if (none(valid0)) break;
        }
        return !valid0;
      }
      
      /*! Intersect a ray with M triangles and updates the hit. */
      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, RayQueryContext* context, const TriangleMvMB<M>& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        const Vec3vf<M> time(ray.time()[k]);
        const Vec3vf<M> v0 = madd(time,Vec3vf<M>(tri.dv0),Vec3vf<M>(tri.v0));
        const Vec3vf<M> v1 = madd(time,Vec3vf<M>(tri.dv1),Vec3vf<M>(tri.v1));
        const Vec3vf<M> v2 = madd(time,Vec3vf<M>(tri.dv2),Vec3vf<M>(tri.v2));
        pre.intersect(ray,k,v0,v1,v2,Intersect1KEpilogM<M,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }

      /*! Test if the ray is occluded by one of the M triangles. */
      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, RayQueryContext* context, const TriangleMvMB<M>& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const Vec3vf<M> time(ray.time()[k]);
        const Vec3vf<M> v0 = madd(time,Vec3vf<M>(tri.dv0),Vec3vf<M>(tri.v0));
        const Vec3vf<M> v1 = madd(time,Vec3vf<M>(tri.dv1),Vec3vf<M>(tri.v1));
        const Vec3vf<M> v2 = madd(time,Vec3vf<M>(tri.dv2),Vec3vf<M>(tri.v2));
        return pre.intersect(ray,k,v0,v1,v2,Occluded1KEpilogM<M,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }
    };

    /*! Intersects M motion blur triangles with 1 ray */
    template<int M, bool filter>
    struct TriangleMvMBIntersector1Pluecker
    {
      typedef TriangleMvMB<M> Primitive;
      typedef PlueckerIntersector1<M> Precalculations;

      /*! Intersect a ray with the M triangles and updates the hit. */
      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, RayQueryContext* context, const TriangleMvMB<M>& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        const Vec3vf<M> time(ray.time());
        const Vec3vf<M> v0 = madd(time,Vec3vf<M>(tri.dv0),Vec3vf<M>(tri.v0));
        const Vec3vf<M> v1 = madd(time,Vec3vf<M>(tri.dv1),Vec3vf<M>(tri.v1));
        const Vec3vf<M> v2 = madd(time,Vec3vf<M>(tri.dv2),Vec3vf<M>(tri.v2));
        pre.intersect(ray,v0,v1,v2,UVIdentity<M>(),Intersect1EpilogM<M,filter>(ray,context,tri.geomID(),tri.primID()));
      }

      /*! Test if the ray is occluded by one of M triangles. */
      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, RayQueryContext* context, const TriangleMvMB<M>& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const Vec3vf<M> time(ray.time());
        const Vec3vf<M> v0 = madd(time,Vec3vf<M>(tri.dv0),Vec3vf<M>(tri.v0));
        const Vec3vf<M> v1 = madd(time,Vec3vf<M>(tri.dv1),Vec3vf<M>(tri.v1));
        const Vec3vf<M> v2 = madd(time,Vec3vf<M>(tri.dv2),Vec3vf<M>(tri.v2));
        return pre.intersect(ray,v0,v1,v2,UVIdentity<M>(),Occluded1EpilogM<M,filter>(ray,context,tri.geomID(),tri.primID()));
      }
      
      static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const Primitive& tri)
      {
        return PrimitivePointQuery1<Primitive>::pointQuery(query, context, tri);
      }
    };
    
    /*! Intersects M motion blur triangles with K rays. */
    template<int M, int K, bool filter>
    struct TriangleMvMBIntersectorKPluecker
    {
      typedef TriangleMvMB<M> Primitive;
      typedef PlueckerIntersectorK<M,K> Precalculations;

      /*! Intersects K rays with M triangles. */
      static __forceinline void intersect(const vbool<K>& valid_i, Precalculations& pre, RayHitK<K>& ray, RayQueryContext* context, const TriangleMvMB<M>& tri)
      {
        for (size_t i=0; i<TriangleMvMB<M>::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(normal.trav_prims,1,popcnt(valid_i),K);
          const Vec3vf<K> time(ray.time());
          const Vec3vf<K> v0 = madd(time,broadcast<vfloat<K>>(tri.dv0,i),broadcast<vfloat<K>>(tri.v0,i));
          const Vec3vf<K> v1 = madd(time,broadcast<vfloat<K>>(tri.dv1,i),broadcast<vfloat<K>>(tri.v1,i));
          const Vec3vf<K> v2 = madd(time,broadcast<vfloat<K>>(tri.dv2,i),broadcast<vfloat<K>>(tri.v2,i));
          pre.intersectK(valid_i,ray,v0,v1,v2,UVIdentity<K>(),IntersectKEpilogM<M,K,filter>(ray,context,tri.geomID(),tri.primID(),i));
        }
      }

      /*! Test for K rays if they are occluded by any of the M triangles. */
      static __forceinline vbool<K> occluded(const vbool<K>& valid_i, Precalculations& pre, RayK<K>& ray, RayQueryContext* context, const TriangleMvMB<M>& tri)
      {
        vbool<K> valid0 = valid_i;

        for (size_t i=0; i<TriangleMvMB<M>::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(shadow.trav_prims,1,popcnt(valid0),K);
          const Vec3vf<K> time(ray.time());
          const Vec3vf<K> v0 = madd(time,broadcast<vfloat<K>>(tri.dv0,i),broadcast<vfloat<K>>(tri.v0,i));
          const Vec3vf<K> v1 = madd(time,broadcast<vfloat<K>>(tri.dv1,i),broadcast<vfloat<K>>(tri.v1,i));
          const Vec3vf<K> v2 = madd(time,broadcast<vfloat<K>>(tri.dv2,i),broadcast<vfloat<K>>(tri.v2,i));
          pre.intersectK(valid0,ray,v0,v1,v2,UVIdentity<K>(),OccludedKEpilogM<M,K,filter>(valid0,ray,context,tri.geomID(),tri.primID(),i));
          if (none(valid0)) break;
        }
        return !valid0;
      }

      /*! Intersect a ray with M triangles and updates the hit. */
      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, RayQueryContext* context, const TriangleMvMB<M>& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        const Vec3vf<M> time(ray.time()[k]);
        const Vec3vf<M> v0 = madd(time,Vec3vf<M>(tri.dv0),Vec3vf<M>(tri.v0));
        const Vec3vf<M> v1 = madd(time,Vec3vf<M>(tri.dv1),Vec3vf<M>(tri.v1));
        const Vec3vf<M> v2 = madd(time,Vec3vf<M>(tri.dv2),Vec3vf<M>(tri.v2));
        pre.intersect(ray,k,v0,v1,v2,UVIdentity<M>(),Intersect1KEpilogM<M,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }

      /*! Test if the ray is occluded by one of the M triangles. */
      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, RayQueryContext* context, const TriangleMvMB<M>& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const Vec3vf<M> time(ray.time()[k]);
        const Vec3vf<M> v0 = madd(time,Vec3vf<M>(tri.dv0),Vec3vf<M>(tri.v0));
        const Vec3vf<M> v1 = madd(time,Vec3vf<M>(tri.dv1),Vec3vf<M>(tri.v1));
        const Vec3vf<M> v2 = madd(time,Vec3vf<M>(tri.dv2),Vec3vf<M>(tri.v2));
        return pre.intersect(ray,k,v0,v1,v2,UVIdentity<M>(),Occluded1KEpilogM<M,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }
    };
  }
}
