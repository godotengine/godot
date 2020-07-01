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

#include "trianglei.h"
#include "triangle_intersector_moeller.h"
#include "triangle_intersector_pluecker.h"

namespace embree
{
  namespace isa
  {
    /*! Intersects M triangles with 1 ray */
    template<int M, int Mx, bool filter>
    struct TriangleMiIntersector1Moeller
    {
      typedef TriangleMi<M> Primitive;
      typedef MoellerTrumboreIntersector1<Mx> Precalculations;

      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        Vec3vf<M> v0, v1, v2; tri.gather(v0,v1,v2,context->scene);
        pre.intersect(ray,v0,v1,v2,/*UVIdentity<Mx>(),*/Intersect1EpilogM<M,Mx,filter>(ray,context,tri.geomID(),tri.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        Vec3vf<M> v0, v1, v2; tri.gather(v0,v1,v2,context->scene);
        return pre.intersect(ray,v0,v1,v2,/*UVIdentity<Mx>(),*/Occluded1EpilogM<M,Mx,filter>(ray,context,tri.geomID(),tri.primID()));
      }
    };

    /*! Intersects M triangles with K rays */
    template<int M, int Mx, int K, bool filter>
    struct TriangleMiIntersectorKMoeller
    {
      typedef TriangleMi<M> Primitive;
      typedef MoellerTrumboreIntersectorK<Mx,K> Precalculations;

      static __forceinline void intersect(const vbool<K>& valid_i, Precalculations& pre, RayHitK<K>& ray, IntersectContext* context, const Primitive& tri)
      {
        const Scene* scene = context->scene;
        for (size_t i=0; i<Primitive::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(normal.trav_prims,1,popcnt(valid_i),RayHitK<K>::size());
          const Vec3vf<K> v0 = tri.getVertex(tri.v0,i,scene);
          const Vec3vf<K> v1 = tri.getVertex(tri.v1,i,scene);
          const Vec3vf<K> v2 = tri.getVertex(tri.v2,i,scene);
          pre.intersectK(valid_i,ray,v0,v1,v2,/*UVIdentity<K>(),*/IntersectKEpilogM<M,K,filter>(ray,context,tri.geomID(),tri.primID(),i));
        }
      }

      static __forceinline vbool<K> occluded(const vbool<K>& valid_i, Precalculations& pre, RayK<K>& ray, IntersectContext* context, const Primitive& tri)
      {
        vbool<K> valid0 = valid_i;
        const Scene* scene = context->scene;

        for (size_t i=0; i<Primitive::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(shadow.trav_prims,1,popcnt(valid_i),RayHitK<K>::size());
          const Vec3vf<K> v0 = tri.getVertex(tri.v0,i,scene);
          const Vec3vf<K> v1 = tri.getVertex(tri.v1,i,scene);
          const Vec3vf<K> v2 = tri.getVertex(tri.v2,i,scene);
          pre.intersectK(valid0,ray,v0,v1,v2,/*UVIdentity<K>(),*/OccludedKEpilogM<M,K,filter>(valid0,ray,context,tri.geomID(),tri.primID(),i));
          if (none(valid0)) break;
        }
        return !valid0;
      }

      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        Vec3vf<M> v0, v1, v2; tri.gather(v0,v1,v2,context->scene);
        pre.intersect(ray,k,v0,v1,v2,/*UVIdentity<Mx>(),*/Intersect1KEpilogM<M,Mx,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }

      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        Vec3vf<M> v0, v1, v2; tri.gather(v0,v1,v2,context->scene);
        return pre.intersect(ray,k,v0,v1,v2,/*UVIdentity<Mx>(),*/Occluded1KEpilogM<M,Mx,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }
    };

    /*! Intersects M triangles with 1 ray */
    template<int M, int Mx, bool filter>
    struct TriangleMiIntersector1Pluecker
    {
      typedef TriangleMi<M> Primitive;
      typedef PlueckerIntersector1<Mx> Precalculations;

      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        Vec3vf<M> v0, v1, v2; tri.gather(v0,v1,v2,context->scene);
        pre.intersect(ray,v0,v1,v2,UVIdentity<Mx>(),Intersect1EpilogM<M,Mx,filter>(ray,context,tri.geomID(),tri.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        Vec3vf<M> v0, v1, v2; tri.gather(v0,v1,v2,context->scene);
        return pre.intersect(ray,v0,v1,v2,UVIdentity<Mx>(),Occluded1EpilogM<M,Mx,filter>(ray,context,tri.geomID(),tri.primID()));
      }
    };

    /*! Intersects M triangles with K rays */
    template<int M, int Mx, int K, bool filter>
    struct TriangleMiIntersectorKPluecker
    {
      typedef TriangleMi<M> Primitive;
      typedef PlueckerIntersectorK<Mx,K> Precalculations;

      static __forceinline void intersect(const vbool<K>& valid_i, Precalculations& pre, RayHitK<K>& ray, IntersectContext* context, const Primitive& tri)
      {
        const Scene* scene = context->scene;
        for (size_t i=0; i<Primitive::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(normal.trav_prims,1,popcnt(valid_i),RayHitK<K>::size());
          const Vec3vf<K> v0 = tri.getVertex(tri.v0,i,scene);
          const Vec3vf<K> v1 = tri.getVertex(tri.v1,i,scene);
          const Vec3vf<K> v2 = tri.getVertex(tri.v2,i,scene);
          pre.intersectK(valid_i,ray,v0,v1,v2,UVIdentity<K>(),IntersectKEpilogM<M,K,filter>(ray,context,tri.geomID(),tri.primID(),i));
        }
      }

      static __forceinline vbool<K> occluded(const vbool<K>& valid_i, Precalculations& pre, RayK<K>& ray, IntersectContext* context, const Primitive& tri)
      {
        vbool<K> valid0 = valid_i;
        const Scene* scene = context->scene;

        for (size_t i=0; i<Primitive::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(shadow.trav_prims,1,popcnt(valid_i),RayHitK<K>::size());
          const Vec3vf<K> v0 = tri.getVertex(tri.v0,i,scene);
          const Vec3vf<K> v1 = tri.getVertex(tri.v1,i,scene);
          const Vec3vf<K> v2 = tri.getVertex(tri.v2,i,scene);
          pre.intersectK(valid0,ray,v0,v1,v2,UVIdentity<K>(),OccludedKEpilogM<M,K,filter>(valid0,ray,context,tri.geomID(),tri.primID(),i));
          if (none(valid0)) break;
        }
        return !valid0;
      }

      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        Vec3vf<M> v0, v1, v2; tri.gather(v0,v1,v2,context->scene);
        pre.intersect(ray,k,v0,v1,v2,UVIdentity<Mx>(),Intersect1KEpilogM<M,Mx,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }

      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        Vec3vf<M> v0, v1, v2; tri.gather(v0,v1,v2,context->scene);
        return pre.intersect(ray,k,v0,v1,v2,UVIdentity<Mx>(),Occluded1KEpilogM<M,Mx,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }
    };

    /*! Intersects M motion blur triangles with 1 ray */
    template<int M, int Mx, bool filter>
    struct TriangleMiMBIntersector1Moeller
    {
      typedef TriangleMi<M> Primitive;
      typedef MoellerTrumboreIntersector1<Mx> Precalculations;

      /*! Intersect a ray with the M triangles and updates the hit. */
      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        Vec3vf<M> v0,v1,v2; tri.gather(v0,v1,v2,context->scene,ray.time());
        pre.intersect(ray,v0,v1,v2,/*UVIdentity<Mx>(),*/Intersect1EpilogM<M,Mx,filter>(ray,context,tri.geomID(),tri.primID()));
      }

      /*! Test if the ray is occluded by one of M triangles. */
      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        Vec3vf<M> v0,v1,v2; tri.gather(v0,v1,v2,context->scene,ray.time());
        return pre.intersect(ray,v0,v1,v2,/*UVIdentity<Mx>(),*/Occluded1EpilogM<M,Mx,filter>(ray,context,tri.geomID(),tri.primID()));
      }
    };

    /*! Intersects M motion blur triangles with K rays. */
    template<int M, int Mx, int K, bool filter>
    struct TriangleMiMBIntersectorKMoeller
    {
      typedef TriangleMi<M> Primitive;
      typedef MoellerTrumboreIntersectorK<Mx,K> Precalculations;

      /*! Intersects K rays with M triangles. */
      static __forceinline void intersect(const vbool<K>& valid_i, Precalculations& pre, RayHitK<K>& ray, IntersectContext* context, const TriangleMi<M>& tri)
      {
        for (size_t i=0; i<TriangleMi<M>::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(normal.trav_prims,1,popcnt(valid_i),K);
          Vec3vf<K> v0,v1,v2; tri.gather(valid_i,v0,v1,v2,i,context->scene,ray.time());
          pre.intersectK(valid_i,ray,v0,v1,v2,/*UVIdentity<K>(),*/IntersectKEpilogM<M,K,filter>(ray,context,tri.geomID(),tri.primID(),i));
        }
      }

      /*! Test for K rays if they are occluded by any of the M triangles. */
      static __forceinline vbool<K> occluded(const vbool<K>& valid_i, Precalculations& pre, RayK<K>& ray, IntersectContext* context, const TriangleMi<M>& tri)
      {
        vbool<K> valid0 = valid_i;
        for (size_t i=0; i<TriangleMi<M>::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(shadow.trav_prims,1,popcnt(valid0),K);
          Vec3vf<K> v0,v1,v2; tri.gather(valid_i,v0,v1,v2,i,context->scene,ray.time());
          pre.intersectK(valid0,ray,v0,v1,v2,/*UVIdentity<K>(),*/OccludedKEpilogM<M,K,filter>(valid0,ray,context,tri.geomID(),tri.primID(),i));
          if (none(valid0)) break;
        }
        return !valid0;
      }

      /*! Intersect a ray with M triangles and updates the hit. */
      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const TriangleMi<M>& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        Vec3vf<M> v0,v1,v2; tri.gather(v0,v1,v2,context->scene,ray.time()[k]);
        pre.intersect(ray,k,v0,v1,v2,/*UVIdentity<Mx>(),*/Intersect1KEpilogM<M,Mx,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }

      /*! Test if the ray is occluded by one of the M triangles. */
      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const TriangleMi<M>& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        Vec3vf<M> v0,v1,v2; tri.gather(v0,v1,v2,context->scene,ray.time()[k]);
        return pre.intersect(ray,k,v0,v1,v2,/*UVIdentity<Mx>(),*/Occluded1KEpilogM<M,Mx,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }
    };

    /*! Intersects M motion blur triangles with 1 ray */
    template<int M, int Mx, bool filter>
    struct TriangleMiMBIntersector1Pluecker
    {
      typedef TriangleMi<M> Primitive;
      typedef PlueckerIntersector1<Mx> Precalculations;

      /*! Intersect a ray with the M triangles and updates the hit. */
      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        Vec3vf<M> v0,v1,v2; tri.gather(v0,v1,v2,context->scene,ray.time());
        pre.intersect(ray,v0,v1,v2,UVIdentity<Mx>(),Intersect1EpilogM<M,Mx,filter>(ray,context,tri.geomID(),tri.primID()));
      }

      /*! Test if the ray is occluded by one of M triangles. */
      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        Vec3vf<M> v0,v1,v2; tri.gather(v0,v1,v2,context->scene,ray.time());
        return pre.intersect(ray,v0,v1,v2,UVIdentity<Mx>(),Occluded1EpilogM<M,Mx,filter>(ray,context,tri.geomID(),tri.primID()));
      }
    };

    /*! Intersects M motion blur triangles with K rays. */
    template<int M, int Mx, int K, bool filter>
    struct TriangleMiMBIntersectorKPluecker
    {
      typedef TriangleMi<M> Primitive;
      typedef PlueckerIntersectorK<Mx,K> Precalculations;

      /*! Intersects K rays with M triangles. */
      static __forceinline void intersect(const vbool<K>& valid_i, Precalculations& pre, RayHitK<K>& ray, IntersectContext* context, const TriangleMi<M>& tri)
      {
        for (size_t i=0; i<TriangleMi<M>::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(normal.trav_prims,1,popcnt(valid_i),K);
          Vec3vf<K> v0,v1,v2; tri.gather(valid_i,v0,v1,v2,i,context->scene,ray.time());
          pre.intersectK(valid_i,ray,v0,v1,v2,UVIdentity<K>(),IntersectKEpilogM<M,K,filter>(ray,context,tri.geomID(),tri.primID(),i));
        }
      }

      /*! Test for K rays if they are occluded by any of the M triangles. */
      static __forceinline vbool<K> occluded(const vbool<K>& valid_i, Precalculations& pre, RayK<K>& ray, IntersectContext* context, const TriangleMi<M>& tri)
      {
        vbool<K> valid0 = valid_i;
        for (size_t i=0; i<TriangleMi<M>::max_size(); i++)
        {
          if (!tri.valid(i)) break;
          STAT3(shadow.trav_prims,1,popcnt(valid0),K);
          Vec3vf<K> v0,v1,v2; tri.gather(valid_i,v0,v1,v2,i,context->scene,ray.time());
          pre.intersectK(valid0,ray,v0,v1,v2,UVIdentity<K>(),OccludedKEpilogM<M,K,filter>(valid0,ray,context,tri.geomID(),tri.primID(),i));
          if (none(valid0)) break;
        }
        return !valid0;
      }

      /*! Intersect a ray with M triangles and updates the hit. */
      static __forceinline void intersect(Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const TriangleMi<M>& tri)
      {
        STAT3(normal.trav_prims,1,1,1);
        Vec3vf<M> v0,v1,v2; tri.gather(v0,v1,v2,context->scene,ray.time()[k]);
        pre.intersect(ray,k,v0,v1,v2,UVIdentity<Mx>(),Intersect1KEpilogM<M,Mx,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }

      /*! Test if the ray is occluded by one of the M triangles. */
      static __forceinline bool occluded(Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const TriangleMi<M>& tri)
      {
        STAT3(shadow.trav_prims,1,1,1);
        Vec3vf<M> v0,v1,v2; tri.gather(v0,v1,v2,context->scene,ray.time()[k]);
        return pre.intersect(ray,k,v0,v1,v2,UVIdentity<Mx>(),Occluded1KEpilogM<M,Mx,K,filter>(ray,k,context,tri.geomID(),tri.primID()));
      }
    };
  }
}
