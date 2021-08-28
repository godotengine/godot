// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "disc_intersector.h"
#include "intersector_epilog.h"
#include "pointi.h"

namespace embree
{
  namespace isa
  {
    template<int M, bool filter>
    struct DiscMiIntersector1
    {
      typedef PointMi<M> Primitive;
      typedef CurvePrecalculations1 Precalculations;

      static __forceinline void intersect(const Precalculations& pre,
                                          RayHit& ray,
                                          IntersectContext* context,
                                          const Primitive& Disc)
      {
        STAT3(normal.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(Disc.geomID());
        Vec4vf<M> v0; Disc.gather(v0, geom);
        const vbool<M> valid = Disc.valid();
        DiscIntersector1<M>::intersect(
          valid, ray, context, geom, pre, v0, Intersect1EpilogM<M, filter>(ray, context, Disc.geomID(), Disc.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre,
                                         Ray& ray,
                                         IntersectContext* context,
                                         const Primitive& Disc)
      {
        STAT3(shadow.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(Disc.geomID());
        Vec4vf<M> v0; Disc.gather(v0, geom);
        const vbool<M> valid = Disc.valid();
        return DiscIntersector1<M>::intersect(
          valid, ray, context, geom, pre, v0, Occluded1EpilogM<M, filter>(ray, context, Disc.geomID(), Disc.primID()));
      }
    };

    template<int M, bool filter>
    struct DiscMiMBIntersector1
    {
      typedef PointMi<M> Primitive;
      typedef CurvePrecalculations1 Precalculations;

      static __forceinline void intersect(const Precalculations& pre,
                                          RayHit& ray,
                                          IntersectContext* context,
                                          const Primitive& Disc)
      {
        STAT3(normal.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(Disc.geomID());
        Vec4vf<M> v0; Disc.gather(v0, geom, ray.time());
        const vbool<M> valid = Disc.valid();
        DiscIntersector1<M>::intersect(
          valid, ray, context, geom, pre, v0, Intersect1EpilogM<M, filter>(ray, context, Disc.geomID(), Disc.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre,
                                         Ray& ray,
                                         IntersectContext* context,
                                         const Primitive& Disc)
      {
        STAT3(shadow.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(Disc.geomID());
        Vec4vf<M> v0; Disc.gather(v0, geom, ray.time());
        const vbool<M> valid = Disc.valid();
        return DiscIntersector1<M>::intersect(
          valid, ray, context, geom, pre, v0, Occluded1EpilogM<M, filter>(ray, context, Disc.geomID(), Disc.primID()));
      }
    };

    template<int M, int K, bool filter>
    struct DiscMiIntersectorK
    {
      typedef PointMi<M> Primitive;
      typedef CurvePrecalculationsK<K> Precalculations;

      static __forceinline void intersect(
          const Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive& Disc)
      {
        STAT3(normal.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(Disc.geomID());
        Vec4vf<M> v0; Disc.gather(v0, geom);
        const vbool<M> valid = Disc.valid();
        DiscIntersectorK<M, K>::intersect(
            valid, ray, k, context, geom, pre, v0,
            Intersect1KEpilogM<M, K, filter>(ray, k, context, Disc.geomID(), Disc.primID()));
      }

      static __forceinline bool occluded(
          const Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive& Disc)
      {
        STAT3(shadow.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(Disc.geomID());
        Vec4vf<M> v0; Disc.gather(v0, geom);
        const vbool<M> valid = Disc.valid();
        return DiscIntersectorK<M, K>::intersect(
          valid, ray, k, context, geom, pre, v0,
          Occluded1KEpilogM<M, K, filter>(ray, k, context, Disc.geomID(), Disc.primID()));
      }
    };

    template<int M, int K, bool filter>
    struct DiscMiMBIntersectorK
    {
      typedef PointMi<M> Primitive;
      typedef CurvePrecalculationsK<K> Precalculations;

      static __forceinline void intersect(
          const Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive& Disc)
      {
        STAT3(normal.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(Disc.geomID());
        Vec4vf<M> v0; Disc.gather(v0, geom, ray.time()[k]);
        const vbool<M> valid = Disc.valid();
        DiscIntersectorK<M, K>::intersect(
          valid, ray, k, context, geom, pre, v0,
          Intersect1KEpilogM<M, K, filter>(ray, k, context, Disc.geomID(), Disc.primID()));
      }

      static __forceinline bool occluded(
          const Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive& Disc)
      {
        STAT3(shadow.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(Disc.geomID());
        Vec4vf<M> v0; Disc.gather(v0, geom, ray.time()[k]);
        const vbool<M> valid = Disc.valid();
        return DiscIntersectorK<M, K>::intersect(
          valid, ray, k, context, geom, pre, v0, Occluded1KEpilogM<M, K, filter>(ray, k, context, Disc.geomID(), Disc.primID()));
      }
    };

    template<int M, bool filter>
    struct OrientedDiscMiIntersector1
    {
      typedef PointMi<M> Primitive;
      typedef CurvePrecalculations1 Precalculations;

      static __forceinline void intersect(const Precalculations& pre,
                                          RayHit& ray,
                                          IntersectContext* context,
                                          const Primitive& Disc)
      {
        STAT3(normal.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(Disc.geomID());
        Vec4vf<M> v0; Vec3vf<M> n0;
        Disc.gather(v0, n0, geom);
        const vbool<M> valid = Disc.valid();
        DiscIntersector1<M>::intersect(
          valid, ray, context, geom, pre, v0, n0, Intersect1EpilogM<M, filter>(ray, context, Disc.geomID(), Disc.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre,
                                         Ray& ray,
                                         IntersectContext* context,
                                         const Primitive& Disc)
      {
        STAT3(shadow.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(Disc.geomID());
        Vec4vf<M> v0; Vec3vf<M> n0;
        Disc.gather(v0, n0, geom);
        const vbool<M> valid = Disc.valid();
        return DiscIntersector1<M>::intersect(
          valid, ray, context, geom, pre, v0, n0, Occluded1EpilogM<M, filter>(ray, context, Disc.geomID(), Disc.primID()));
      }
    };

    template<int M, bool filter>
    struct OrientedDiscMiMBIntersector1
    {
      typedef PointMi<M> Primitive;
      typedef CurvePrecalculations1 Precalculations;

      static __forceinline void intersect(const Precalculations& pre,
                                          RayHit& ray,
                                          IntersectContext* context,
                                          const Primitive& Disc)
      {
        STAT3(normal.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(Disc.geomID());
        Vec4vf<M> v0; Vec3vf<M> n0;
        Disc.gather(v0, n0, geom, ray.time());
        const vbool<M> valid = Disc.valid();
        DiscIntersector1<M>::intersect(
          valid, ray, context, geom, pre, v0, n0, Intersect1EpilogM<M, filter>(ray, context, Disc.geomID(), Disc.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre,
                                         Ray& ray,
                                         IntersectContext* context,
                                         const Primitive& Disc)
      {
        STAT3(shadow.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(Disc.geomID());
        Vec4vf<M> v0; Vec3vf<M> n0;
        Disc.gather(v0, n0, geom, ray.time());
        const vbool<M> valid = Disc.valid();
        return DiscIntersector1<M>::intersect(
          valid, ray, context, geom, pre, v0, n0, Occluded1EpilogM<M, filter>(ray, context, Disc.geomID(), Disc.primID()));
      }
    };

    template<int M, int K, bool filter>
    struct OrientedDiscMiIntersectorK
    {
      typedef PointMi<M> Primitive;
      typedef CurvePrecalculationsK<K> Precalculations;

      static __forceinline void intersect(
          const Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive& Disc)
      {
        STAT3(normal.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(Disc.geomID());
        Vec4vf<M> v0; Vec3vf<M> n0;
        Disc.gather(v0, n0, geom);
        const vbool<M> valid = Disc.valid();
        DiscIntersectorK<M, K>::intersect(
            valid, ray, k, context, geom, pre, v0, n0,
            Intersect1KEpilogM<M, K, filter>(ray, k, context, Disc.geomID(), Disc.primID()));
      }

      static __forceinline bool occluded(
          const Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive& Disc)
      {
        STAT3(shadow.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(Disc.geomID());
        Vec4vf<M> v0; Vec3vf<M> n0;
        Disc.gather(v0, n0, geom);
        const vbool<M> valid = Disc.valid();
        return DiscIntersectorK<M, K>::intersect(
            valid, ray, k, context, geom, pre, v0, n0,
            Occluded1KEpilogM<M, K, filter>(ray, k, context, Disc.geomID(), Disc.primID()));
      }
    };

    template<int M, int K, bool filter>
    struct OrientedDiscMiMBIntersectorK
    {
      typedef PointMi<M> Primitive;
      typedef CurvePrecalculationsK<K> Precalculations;

      static __forceinline void intersect(
          const Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive& Disc)
      {
        STAT3(normal.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(Disc.geomID());
        Vec4vf<M> v0; Vec3vf<M> n0;
        Disc.gather(v0, n0, geom, ray.time()[k]);
        const vbool<M> valid = Disc.valid();
        DiscIntersectorK<M, K>::intersect(
            valid, ray, k, context, geom, pre, v0, n0,
            Intersect1KEpilogM<M, K, filter>(ray, k, context, Disc.geomID(), Disc.primID()));
      }

      static __forceinline bool occluded(
          const Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive& Disc)
      {
        STAT3(shadow.trav_prims, 1, 1, 1);
        const Points* geom = context->scene->get<Points>(Disc.geomID());
        Vec4vf<M> v0; Vec3vf<M> n0;
        Disc.gather(v0, n0, geom, ray.time()[k]);
        const vbool<M> valid = Disc.valid();
        return DiscIntersectorK<M, K>::intersect(
            valid, ray, k, context, geom, pre, v0, n0,
            Occluded1KEpilogM<M, K, filter>(ray, k, context, Disc.geomID(), Disc.primID()));
      }
    };
  }  // namespace isa
}  // namespace embree
