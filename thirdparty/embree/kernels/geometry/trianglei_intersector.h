// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "trianglei.h"
#include "triangle_intersector_moeller.h"
#include "triangle_intersector_pluecker.h"

namespace embree
{
  namespace isa
  {
    /*! Intersects M triangles with 1 ray */
    template<int M, bool filter>
    struct TriangleMiIntersector1Moeller
    {
      typedef TriangleMi<M> Primitive;
      typedef MoellerTrumboreIntersector1<M> Precalculations;

      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        Vec3vf<M> v0, v1, v2; tri.gather(v0,v1,v2,context->scene);
        pre.intersect(ray,v0,v1,v2,Intersect1EpilogM<M,filter>(ray,context,tri.geomID(),tri.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        Vec3vf<M> v0, v1, v2; tri.gather(v0,v1,v2,context->scene);
        return pre.intersect(ray,v0,v1,v2,Occluded1EpilogM<M,filter>(ray,context,tri.geomID(),tri.primID()));
      }
      
      static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const Primitive& tri)
      {
        return PrimitivePointQuery1<Primitive>::pointQuery(query, context, tri);
      }
    };

    /*! Intersects M triangles with K rays */
    template<int M, int K, bool filter>
    struct TriangleMiIntersectorKMoeller
    {
      typedef TriangleMi<M> Primitive;
      typedef MoellerTrumboreIntersectorK<M,K> Precalculations;

      static __forceinline void intersect(const vbool<K>& valid_i, Precalculations& pre, RayHitK<K>& ray, RayQueryContext* context, const Primitive& tri)
      {
        const Scene* scene = context->scene;
        for (size_t i=0; i<Primitive::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(normal.trav_prims,1,popcnt(valid_i),RayHitK<K>::size());
          const Vec3vf<K> v0 = tri.template getVertex<0>(i,scene);
          const Vec3vf<K> v1 = tri.template getVertex<1>(i,scene);
          const Vec3vf<K> v2 = tri.template getVertex<2>(i,scene);
          pre.intersectK(valid_i,ray,v0,v1,v2,IntersectKEpilogM<M,K,filter>(ray,context,tri.geomID(),tri.primID(),i));
        }
      }

      static __forceinline vbool<K> occluded(const vbool<K>& valid_i, Precalculations& pre, RayK<K>& ray, RayQueryContext* context, const Primitive& tri)
      {
        vbool<K> valid0 = valid_i;
        const Scene* scene = context->scene;

        for (size_t i=0; i<Primitive::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(shadow.trav_prims,1,popcnt(valid_i),RayHitK<K>::size());
          const Vec3vf<K> v0 = tri.template getVertex<0>(i,scene);
          const Vec3vf<K> v1 = tri.template getVertex<1>(i,scene);
          const Vec3vf<K> v2 = tri.template getVertex<2>(i,scene);
          pre.intersectK(valid0,ray,v0,v1,v2,OccludedKEpilogM<M,K,filter>(valid0,ray,context,tri.geomID(),tri.primID(),i));
          if (none(valid0)) break;
        }
        return !valid0;
      }
      
      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        Vec3vf<M> v0, v1, v2; tri.gather(v0,v1,v2,context->scene);
        pre.intersect(ray,k,v0,v1,v2,Intersect1KEpilogM<M,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }

      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        Vec3vf<M> v0, v1, v2; tri.gather(v0,v1,v2,context->scene);
        return pre.intersect(ray,k,v0,v1,v2,Occluded1KEpilogM<M,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }
    };

    /*! Intersects M triangles with 1 ray */
    template<int M, bool filter>
    struct TriangleMiIntersector1Pluecker
    {
      typedef TriangleMi<M> Primitive;
      typedef PlueckerIntersector1<M> Precalculations;

      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        Vec3vf<M> v0, v1, v2; tri.gather(v0,v1,v2,context->scene);
        pre.intersect(ray,v0,v1,v2,Intersect1EpilogM<M,filter>(ray,context,tri.geomID(),tri.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        Vec3vf<M> v0, v1, v2; tri.gather(v0,v1,v2,context->scene);
        return pre.intersect(ray,v0,v1,v2,Occluded1EpilogM<M,filter>(ray,context,tri.geomID(),tri.primID()));
      }
      
      static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const Primitive& tri)
      {
        return PrimitivePointQuery1<Primitive>::pointQuery(query, context, tri);
      }
    };

    /*! Intersects M triangles with K rays */
    template<int M, int K, bool filter>
    struct TriangleMiIntersectorKPluecker
    {
      typedef TriangleMi<M> Primitive;
      typedef PlueckerIntersectorK<M,K> Precalculations;

      static __forceinline void intersect(const vbool<K>& valid_i, Precalculations& pre, RayHitK<K>& ray, RayQueryContext* context, const Primitive& tri)
      {
        const Scene* scene = context->scene;
        for (size_t i=0; i<Primitive::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(normal.trav_prims,1,popcnt(valid_i),RayHitK<K>::size());
          const Vec3vf<K> v0 = tri.template getVertex<0>(i,scene);
          const Vec3vf<K> v1 = tri.template getVertex<1>(i,scene);
          const Vec3vf<K> v2 = tri.template getVertex<2>(i,scene);
          pre.intersectK(valid_i,ray,v0,v1,v2,IntersectKEpilogM<M,K,filter>(ray,context,tri.geomID(),tri.primID(),i));
        }
      }

      static __forceinline vbool<K> occluded(const vbool<K>& valid_i, Precalculations& pre, RayK<K>& ray, RayQueryContext* context, const Primitive& tri)
      {
        vbool<K> valid0 = valid_i;
        const Scene* scene = context->scene;

        for (size_t i=0; i<Primitive::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(shadow.trav_prims,1,popcnt(valid_i),RayHitK<K>::size());
          const Vec3vf<K> v0 = tri.template getVertex<0>(i,scene);
          const Vec3vf<K> v1 = tri.template getVertex<1>(i,scene);
          const Vec3vf<K> v2 = tri.template getVertex<2>(i,scene);
          pre.intersectK(valid0,ray,v0,v1,v2,OccludedKEpilogM<M,K,filter>(valid0,ray,context,tri.geomID(),tri.primID(),i));
          if (none(valid0)) break;
        }
        return !valid0;
      }

      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        Vec3vf<M> v0, v1, v2; tri.gather(v0,v1,v2,context->scene);
        pre.intersect(ray,k,v0,v1,v2,Intersect1KEpilogM<M,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }

      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        Vec3vf<M> v0, v1, v2; tri.gather(v0,v1,v2,context->scene);
        return pre.intersect(ray,k,v0,v1,v2,Occluded1KEpilogM<M,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }
    };

    /*! Intersects M motion blur triangles with 1 ray */
    template<int M, bool filter>
    struct TriangleMiMBIntersector1Moeller
    {
      typedef TriangleMi<M> Primitive;
      typedef MoellerTrumboreIntersector1<M> Precalculations;

      /*! Intersect a ray with the M triangles and updates the hit. */
      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        Vec3vf<M> v0,v1,v2; tri.gather(v0,v1,v2,context->scene,ray.time());
        pre.intersect(ray,v0,v1,v2,Intersect1EpilogM<M,filter>(ray,context,tri.geomID(),tri.primID()));
      }

      /*! Test if the ray is occluded by one of M triangles. */
      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        Vec3vf<M> v0,v1,v2; tri.gather(v0,v1,v2,context->scene,ray.time());
        return pre.intersect(ray,v0,v1,v2,Occluded1EpilogM<M,filter>(ray,context,tri.geomID(),tri.primID()));
      }
      
      static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const Primitive& tri)
      {
        return PrimitivePointQuery1<Primitive>::pointQuery(query, context, tri);
      }
    };

    /*! Intersects M motion blur triangles with K rays. */
    template<int M, int K, bool filter>
    struct TriangleMiMBIntersectorKMoeller
    {
      typedef TriangleMi<M> Primitive;
      typedef MoellerTrumboreIntersectorK<M,K> Precalculations;

      /*! Intersects K rays with M triangles. */
      static __forceinline void intersect(const vbool<K>& valid_i, Precalculations& pre, RayHitK<K>& ray, RayQueryContext* context, const TriangleMi<M>& tri)
      {
        for (size_t i=0; i<TriangleMi<M>::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(normal.trav_prims,1,popcnt(valid_i),K);
          Vec3vf<K> v0,v1,v2; tri.template gather<K>(valid_i,v0,v1,v2,i,context->scene,ray.time());
          pre.intersectK(valid_i,ray,v0,v1,v2,IntersectKEpilogM<M,K,filter>(ray,context,tri.geomID(),tri.primID(),i));
        }
      }

      /*! Test for K rays if they are occluded by any of the M triangles. */
      static __forceinline vbool<K> occluded(const vbool<K>& valid_i, Precalculations& pre, RayK<K>& ray, RayQueryContext* context, const TriangleMi<M>& tri)
      {
        vbool<K> valid0 = valid_i;
        for (size_t i=0; i<TriangleMi<M>::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(shadow.trav_prims,1,popcnt(valid0),K);
          Vec3vf<K> v0,v1,v2; tri.template gather<K>(valid_i,v0,v1,v2,i,context->scene,ray.time());
          pre.intersectK(valid0,ray,v0,v1,v2,OccludedKEpilogM<M,K,filter>(valid0,ray,context,tri.geomID(),tri.primID(),i));
          if (none(valid0)) break;
        }
        return !valid0;
      }

      /*! Intersect a ray with M triangles and updates the hit. */
      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, RayQueryContext* context, const TriangleMi<M>& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        Vec3vf<M> v0,v1,v2; tri.gather(v0,v1,v2,context->scene,ray.time()[k]);
        pre.intersect(ray,k,v0,v1,v2,Intersect1KEpilogM<M,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }

      /*! Test if the ray is occluded by one of the M triangles. */
      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, RayQueryContext* context, const TriangleMi<M>& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        Vec3vf<M> v0,v1,v2; tri.gather(v0,v1,v2,context->scene,ray.time()[k]);
        return pre.intersect(ray,k,v0,v1,v2,Occluded1KEpilogM<M,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }
    };

    /*! Intersects M motion blur triangles with 1 ray */
    template<int M, bool filter>
    struct TriangleMiMBIntersector1Pluecker
    {
      typedef TriangleMi<M> Primitive;
      typedef PlueckerIntersector1<M> Precalculations;

      /*! Intersect a ray with the M triangles and updates the hit. */
      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        Vec3vf<M> v0,v1,v2; tri.gather(v0,v1,v2,context->scene,ray.time());
        pre.intersect(ray,v0,v1,v2,Intersect1EpilogM<M,filter>(ray,context,tri.geomID(),tri.primID()));
      }

      /*! Test if the ray is occluded by one of M triangles. */
      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, RayQueryContext* context, const Primitive& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        Vec3vf<M> v0,v1,v2; tri.gather(v0,v1,v2,context->scene,ray.time());
        return pre.intersect(ray,v0,v1,v2,Occluded1EpilogM<M,filter>(ray,context,tri.geomID(),tri.primID()));
      }
      
      static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const Primitive& tri)
      {
        return PrimitivePointQuery1<Primitive>::pointQuery(query, context, tri);
      }
    };

    /*! Intersects M motion blur triangles with K rays. */
    template<int M, int K, bool filter>
    struct TriangleMiMBIntersectorKPluecker
    {
      typedef TriangleMi<M> Primitive;
      typedef PlueckerIntersectorK<M,K> Precalculations;

      /*! Intersects K rays with M triangles. */
      static __forceinline void intersect(const vbool<K>& valid_i, Precalculations& pre, RayHitK<K>& ray, RayQueryContext* context, const TriangleMi<M>& tri)
      {
        for (size_t i=0; i<TriangleMi<M>::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(normal.trav_prims,1,popcnt(valid_i),K);
          Vec3vf<K> v0,v1,v2; tri.template gather<K>(valid_i,v0,v1,v2,i,context->scene,ray.time());
          pre.intersectK(valid_i,ray,v0,v1,v2,IntersectKEpilogM<M,K,filter>(ray,context,tri.geomID(),tri.primID(),i));
        }
      }

      /*! Test for K rays if they are occluded by any of the M triangles. */
      static __forceinline vbool<K> occluded(const vbool<K>& valid_i, Precalculations& pre, RayK<K>& ray, RayQueryContext* context, const TriangleMi<M>& tri)
      {
        vbool<K> valid0 = valid_i;
        for (size_t i=0; i<TriangleMi<M>::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(shadow.trav_prims,1,popcnt(valid0),K);
          Vec3vf<K> v0,v1,v2; tri.template gather<K>(valid_i,v0,v1,v2,i,context->scene,ray.time());
          pre.intersectK(valid0,ray,v0,v1,v2,OccludedKEpilogM<M,K,filter>(valid0,ray,context,tri.geomID(),tri.primID(),i));
          if (none(valid0)) break;
        }
        return !valid0;
      }
      
      /*! Intersect a ray with M triangles and updates the hit. */
      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, RayQueryContext* context, const TriangleMi<M>& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        Vec3vf<M> v0,v1,v2; tri.gather(v0,v1,v2,context->scene,ray.time()[k]);
        pre.intersect(ray,k,v0,v1,v2,Intersect1KEpilogM<M,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }

      /*! Test if the ray is occluded by one of the M triangles. */
      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, RayQueryContext* context, const TriangleMi<M>& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        Vec3vf<M> v0,v1,v2; tri.gather(v0,v1,v2,context->scene,ray.time()[k]);
        return pre.intersect(ray,k,v0,v1,v2,Occluded1KEpilogM<M,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }
    };
  }
}
