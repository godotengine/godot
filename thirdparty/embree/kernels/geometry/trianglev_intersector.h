// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "triangle.h"
#include "triangle_intersector_pluecker.h"
#include "triangle_intersector_moeller.h"
#include "triangle_intersector_woop.h"

namespace embree
{
  namespace isa
  {
    /*! Intersects M triangles with 1 ray */
    template<int M, bool filter>
    struct TriangleMvIntersector1Moeller
    {
      typedef TriangleMv<M> Primitive;
      typedef MoellerTrumboreIntersector1<M> Precalculations;

      /*! Intersect a ray with M triangles and updates the hit. */
      static __forceinline void intersect(Precalculations& pre, RayHit& ray, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        pre.intersect(ray,tri.v0,tri.v1,tri.v2,/*UVIdentity<M>(),*/Intersect1EpilogM<M,filter>(ray,context,tri.geomID(),tri.primID()));
      }

      /*! Test if the ray is occluded by one of the M triangles. */
      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        return pre.intersect(ray,tri.v0,tri.v1,tri.v2,/*UVIdentity<M>(),*/Occluded1EpilogM<M,filter>(ray,context,tri.geomID(),tri.primID()));
      }
      
      static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const Primitive& tri)
      {
        return PrimitivePointQuery1<Primitive>::pointQuery(query, context, tri);
      }
    };


    template<int M, bool filter>
    struct TriangleMvIntersector1Woop
    {
      typedef TriangleMv<M> Primitive;
      typedef WoopIntersector1<M> intersec;
      typedef WoopPrecalculations1<M> Precalculations;

      /*! Intersect a ray with M triangles and updates the hit. */
      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        intersec::intersect(ray,pre,tri.v0,tri.v1,tri.v2,Intersect1EpilogM<M,filter>(ray,context,tri.geomID(),tri.primID()));
      }

      /*! Test if the ray is occluded by one of the M triangles. */
      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        return intersec::intersect(ray,pre,tri.v0,tri.v1,tri.v2,Occluded1EpilogM<M,filter>(ray,context,tri.geomID(),tri.primID()));
      }
      
      static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const Primitive& tri)
      {
        return PrimitivePointQuery1<Primitive>::pointQuery(query, context, tri);
      }
    };


    /*! Intersects M triangles with K rays */
    template<int M, int K, bool filter>
    struct TriangleMvIntersectorKMoeller
    {
      typedef TriangleMv<M> Primitive;
      typedef MoellerTrumboreIntersectorK<M,K> Precalculations;

      /*! Intersects K rays with M triangles. */
      static __forceinline void intersect(const vbool<K>& valid_i, Precalculations& pre, RayHitK<K>& ray, RayQueryContext* context, const Primitive& tri)
      {
        for (size_t i=0; i<M; i++)
        {
          if (!tri.valid(i)) break;
          STAT3(normal.trav_prims,1,popcnt(valid_i),K);
          const Vec3vf<K> v0 = broadcast<vfloat<K>>(tri.v0,i);
          const Vec3vf<K> v1 = broadcast<vfloat<K>>(tri.v1,i);
          const Vec3vf<K> v2 = broadcast<vfloat<K>>(tri.v2,i);
          pre.intersectK(valid_i,ray,v0,v1,v2,/*UVIdentity<K>(),*/IntersectKEpilogM<M,K,filter>(ray,context,tri.geomID(),tri.primID(),i));
        }
      }

      /*! Test for K rays if they are occluded by any of the M triangles. */
      static __forceinline vbool<K> occluded(const vbool<K>& valid_i, Precalculations& pre, RayK<K>& ray, RayQueryContext* context, const Primitive& tri)
      {
        vbool<K> valid0 = valid_i;

        for (size_t i=0; i<M; i++)
        {
          if (!tri.valid(i)) break;
          STAT3(shadow.trav_prims,1,popcnt(valid_i),K);
          const Vec3vf<K> v0 = broadcast<vfloat<K>>(tri.v0,i);
          const Vec3vf<K> v1 = broadcast<vfloat<K>>(tri.v1,i);
          const Vec3vf<K> v2 = broadcast<vfloat<K>>(tri.v2,i);
          pre.intersectK(valid0,ray,v0,v1,v2,/*UVIdentity<K>(),*/OccludedKEpilogM<M,K,filter>(valid0,ray,context,tri.geomID(),tri.primID(),i));
          if (none(valid0)) break;
        }
        return !valid0;
      }
      
      /*! Intersect a ray with M triangles and updates the hit. */
      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        pre.intersect(ray,k,tri.v0,tri.v1,tri.v2,/*UVIdentity<M>(),*/Intersect1KEpilogM<M,K,filter>(ray,k,context,tri.geomID(),tri.primID())); //FIXME: M
      }

      /*! Test if the ray is occluded by one of the M triangles. */
      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        return pre.intersect(ray,k,tri.v0,tri.v1,tri.v2,/*UVIdentity<M>(),*/Occluded1KEpilogM<M,K,filter>(ray,k,context,tri.geomID(),tri.primID())); //FIXME: M
      }
    };

    /*! Intersects M triangles with 1 ray */
    template<int M, bool filter>
    struct TriangleMvIntersector1Pluecker
    {
      typedef TriangleMv<M> Primitive;
      typedef PlueckerIntersector1<M> Precalculations;

      /*! Intersect a ray with M triangles and updates the hit. */
      static __forceinline void intersect(Precalculations& pre, RayHit& ray, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        pre.intersect(ray,tri.v0,tri.v1,tri.v2,UVIdentity<M>(),Intersect1EpilogM<M,filter>(ray,context,tri.geomID(),tri.primID()));
      }

      /*! Test if the ray is occluded by one of the M triangles. */
      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        return pre.intersect(ray,tri.v0,tri.v1,tri.v2,UVIdentity<M>(),Occluded1EpilogM<M,filter>(ray,context,tri.geomID(),tri.primID()));
      }
      
      static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const Primitive& tri)
      {
        return PrimitivePointQuery1<Primitive>::pointQuery(query, context, tri);
      }
    };

    /*! Intersects M triangles with K rays */
    template<int M, int K, bool filter>
    struct TriangleMvIntersectorKPluecker
    {
      typedef TriangleMv<M> Primitive;
      typedef PlueckerIntersectorK<M,K> Precalculations;

      /*! Intersects K rays with M triangles. */
      static __forceinline void intersect(const vbool<K>& valid_i, Precalculations& pre, RayHitK<K>& ray, RayQueryContext* context, const Primitive& tri)
      {
        for (size_t i=0; i<M; i++)
        {
          if (!tri.valid(i)) break;
          STAT3(normal.trav_prims,1,popcnt(valid_i),K);
          const Vec3vf<K> v0 = broadcast<vfloat<K>>(tri.v0,i);
          const Vec3vf<K> v1 = broadcast<vfloat<K>>(tri.v1,i);
          const Vec3vf<K> v2 = broadcast<vfloat<K>>(tri.v2,i);
          pre.intersectK(valid_i,ray,v0,v1,v2,UVIdentity<K>(),IntersectKEpilogM<M,K,filter>(ray,context,tri.geomID(),tri.primID(),i));
        }
      }

      /*! Test for K rays if they are occluded by any of the M triangles. */
      static __forceinline vbool<K> occluded(const vbool<K>& valid_i, Precalculations& pre, RayK<K>& ray, RayQueryContext* context, const Primitive& tri)
      {
        vbool<K> valid0 = valid_i;

        for (size_t i=0; i<M; i++)
        {
          if (!tri.valid(i)) break;
          STAT3(shadow.trav_prims,1,popcnt(valid_i),K);
          const Vec3vf<K> v0 = broadcast<vfloat<K>>(tri.v0,i);
          const Vec3vf<K> v1 = broadcast<vfloat<K>>(tri.v1,i);
          const Vec3vf<K> v2 = broadcast<vfloat<K>>(tri.v2,i);
          pre.intersectK(valid0,ray,v0,v1,v2,UVIdentity<K>(),OccludedKEpilogM<M,K,filter>(valid0,ray,context,tri.geomID(),tri.primID(),i));
          if (none(valid0)) break;
        }
        return !valid0;
      }
      
      /*! Intersect a ray with M triangles and updates the hit. */
      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        pre.intersect(ray,k,tri.v0,tri.v1,tri.v2,UVIdentity<M>(),Intersect1KEpilogM<M,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }

      /*! Test if the ray is occluded by one of the M triangles. */
      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        return pre.intersect(ray,k,tri.v0,tri.v1,tri.v2,UVIdentity<M>(),Occluded1KEpilogM<M,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }
    };
  }
}
