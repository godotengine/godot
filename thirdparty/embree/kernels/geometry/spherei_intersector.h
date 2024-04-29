// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "intersector_epilog.h"
#include "pointi.h"
#include "sphere_intersector.h"

namespace embree
{
  namespace isa
  {
    template<int M, bool filter>
    struct SphereMiIntersector1
    {
      typedef PointMi<M> Primitive;
      typedef CurvePrecalculations1 Precalculations;

      static __forceinline void intersect(const Precalculations& pre,
                                          RayHit& ray,
                                          IntersectContext* context,
                                          const Primitive& sphere)
      {
        STAT3(normal.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(sphere.geomID());
        Vec4vf<M> v0; sphere.gather(v0, geom);
        const vbool<M> valid = sphere.valid();
        SphereIntersector1<M>::intersect(
          valid, ray, context, geom, pre, v0, Intersect1EpilogM<M, filter>(ray, context, sphere.geomID(), sphere.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre,
                                         Ray& ray,
                                         IntersectContext* context,
                                         const Primitive& sphere)
      {
        STAT3(shadow.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(sphere.geomID());
        Vec4vf<M> v0; sphere.gather(v0, geom);
        const vbool<M> valid = sphere.valid();
        return SphereIntersector1<M>::intersect(
          valid, ray, context, geom, pre, v0, Occluded1EpilogM<M, filter>(ray, context, sphere.geomID(), sphere.primID()));
      }
      
      static __forceinline bool pointQuery(PointQuery* query,
                                           PointQueryContext* context,
                                           const Primitive& sphere)
      {
        return PrimitivePointQuery1<Primitive>::pointQuery(query, context, sphere);
      }
    };

    template<int M, bool filter>
    struct SphereMiMBIntersector1
    {
      typedef PointMi<M> Primitive;
      typedef CurvePrecalculations1 Precalculations;

      static __forceinline void intersect(const Precalculations& pre,
                                          RayHit& ray,
                                          IntersectContext* context,
                                          const Primitive& sphere)
      {
        STAT3(normal.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(sphere.geomID());
        Vec4vf<M> v0; sphere.gather(v0, geom, ray.time());
        const vbool<M> valid = sphere.valid();
        SphereIntersector1<M>::intersect(
          valid, ray, context, geom, pre, v0, Intersect1EpilogM<M, filter>(ray, context, sphere.geomID(), sphere.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre,
                                         Ray& ray,
                                         IntersectContext* context,
                                         const Primitive& sphere)
      {
        STAT3(shadow.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(sphere.geomID());
        Vec4vf<M> v0; sphere.gather(v0, geom, ray.time());
        const vbool<M> valid = sphere.valid();
        return SphereIntersector1<M>::intersect(
          valid, ray, context, geom, pre, v0, Occluded1EpilogM<M, filter>(ray, context, sphere.geomID(), sphere.primID()));
      }

      static __forceinline bool pointQuery(PointQuery* query,
                                           PointQueryContext* context,
                                           const Primitive& sphere)
      {
        return PrimitivePointQuery1<Primitive>::pointQuery(query, context, sphere);
      }
    };

    template<int M, int K, bool filter>
    struct SphereMiIntersectorK
    {
      typedef PointMi<M> Primitive;
      typedef CurvePrecalculationsK<K> Precalculations;

      static __forceinline void intersect(
          const Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive& sphere)
      {
        STAT3(normal.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(sphere.geomID());
        Vec4vf<M> v0; sphere.gather(v0, geom);
        const vbool<M> valid = sphere.valid();
        SphereIntersectorK<M, K>::intersect(
          valid, ray, k, context, geom, pre, v0,
          Intersect1KEpilogM<M, K, filter>(ray, k, context, sphere.geomID(), sphere.primID()));
      }

      static __forceinline bool occluded(
          const Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive& sphere)
      {
        STAT3(shadow.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(sphere.geomID());
        Vec4vf<M> v0; sphere.gather(v0, geom);
        const vbool<M> valid = sphere.valid();
        return SphereIntersectorK<M, K>::intersect(
          valid, ray, k, context, geom, pre, v0,
          Occluded1KEpilogM<M, K, filter>(ray, k, context, sphere.geomID(), sphere.primID()));
      }
    };

    template<int M, int K, bool filter>
    struct SphereMiMBIntersectorK
    {
      typedef PointMi<M> Primitive;
      typedef CurvePrecalculationsK<K> Precalculations;

      static __forceinline void intersect(
          const Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive& sphere)
      {
        STAT3(normal.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(sphere.geomID());
        Vec4vf<M> v0; sphere.gather(v0, geom, ray.time()[k]);
        const vbool<M> valid = sphere.valid();
        SphereIntersectorK<M, K>::intersect(
          valid, ray, k, context, geom, pre, v0,
          Intersect1KEpilogM<M, K, filter>(ray, k, context, sphere.geomID(), sphere.primID()));
      }

      static __forceinline bool occluded(
          const Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive& sphere)
      {
        STAT3(shadow.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(sphere.geomID());
        Vec4vf<M> v0; sphere.gather(v0, geom, ray.time()[k]);
        const vbool<M> valid = sphere.valid();
        return SphereIntersectorK<M, K>::intersect(
          valid, ray, k, context, geom, pre, v0,
          Occluded1KEpilogM<M, K, filter>(ray, k, context, sphere.geomID(), sphere.primID()));
      }
    };
  }  // namespace isa
}  // namespace embree
