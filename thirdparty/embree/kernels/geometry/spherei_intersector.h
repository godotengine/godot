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

#include "intersector_epilog.h"
#include "pointi.h"
#include "sphere_intersector.h"

namespace embree
{
  namespace isa
  {
    template<int M, int Mx, bool filter>
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
        Vec4vf<M> v0;
        sphere.gather(v0, context->scene);
        const vbool<Mx> valid = sphere.template valid<Mx>();
        SphereIntersector1<Mx>::intersect(
            valid, ray, pre, v0, Intersect1EpilogM<M, Mx, filter>(ray, context, sphere.geomID(), sphere.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre,
                                         Ray& ray,
                                         IntersectContext* context,
                                         const Primitive& sphere)
      {
        STAT3(shadow.trav_prims, 1, 1, 1);
        Vec4vf<M> v0;
        sphere.gather(v0, context->scene);
        const vbool<Mx> valid = sphere.template valid<Mx>();
        return SphereIntersector1<Mx>::intersect(
            valid, ray, pre, v0, Occluded1EpilogM<M, Mx, filter>(ray, context, sphere.geomID(), sphere.primID()));
      }
    };

    template<int M, int Mx, bool filter>
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
        Vec4vf<M> v0;
        sphere.gather(v0, context->scene, ray.time());
        const vbool<Mx> valid = sphere.template valid<Mx>();
        SphereIntersector1<Mx>::intersect(
            valid, ray, pre, v0, Intersect1EpilogM<M, Mx, filter>(ray, context, sphere.geomID(), sphere.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre,
                                         Ray& ray,
                                         IntersectContext* context,
                                         const Primitive& sphere)
      {
        STAT3(shadow.trav_prims, 1, 1, 1);
        Vec4vf<M> v0;
        sphere.gather(v0, context->scene, ray.time());
        const vbool<Mx> valid = sphere.template valid<Mx>();
        return SphereIntersector1<Mx>::intersect(
            valid, ray, pre, v0, Occluded1EpilogM<M, Mx, filter>(ray, context, sphere.geomID(), sphere.primID()));
      }
    };

    template<int M, int Mx, int K, bool filter>
    struct SphereMiIntersectorK
    {
      typedef PointMi<M> Primitive;
      typedef CurvePrecalculationsK<K> Precalculations;

      static __forceinline void intersect(
          const Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive& sphere)
      {
        STAT3(normal.trav_prims, 1, 1, 1);
        Vec4vf<M> v0;
        sphere.gather(v0, context->scene);
        const vbool<Mx> valid = sphere.template valid<Mx>();
        SphereIntersectorK<Mx, K>::intersect(
            valid,
            ray,
            k,
            pre,
            v0,
            Intersect1KEpilogM<M, Mx, K, filter>(ray, k, context, sphere.geomID(), sphere.primID()));
      }

      static __forceinline bool occluded(
          const Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive& sphere)
      {
        STAT3(shadow.trav_prims, 1, 1, 1);
        Vec4vf<M> v0;
        sphere.gather(v0, context->scene);
        const vbool<Mx> valid = sphere.template valid<Mx>();
        return SphereIntersectorK<Mx, K>::intersect(
            valid,
            ray,
            k,
            pre,
            v0,
            Occluded1KEpilogM<M, Mx, K, filter>(ray, k, context, sphere.geomID(), sphere.primID()));
      }
    };

    template<int M, int Mx, int K, bool filter>
    struct SphereMiMBIntersectorK
    {
      typedef PointMi<M> Primitive;
      typedef CurvePrecalculationsK<K> Precalculations;

      static __forceinline void intersect(
          const Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive& sphere)
      {
        STAT3(normal.trav_prims, 1, 1, 1);
        Vec4vf<M> v0;
        sphere.gather(v0, context->scene, ray.time()[k]);
        const vbool<Mx> valid = sphere.template valid<Mx>();
        SphereIntersectorK<Mx, K>::intersect(
            valid,
            ray,
            k,
            pre,
            v0,
            Intersect1KEpilogM<M, Mx, K, filter>(ray, k, context, sphere.geomID(), sphere.primID()));
      }

      static __forceinline bool occluded(
          const Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive& sphere)
      {
        STAT3(shadow.trav_prims, 1, 1, 1);
        Vec4vf<M> v0;
        sphere.gather(v0, context->scene, ray.time()[k]);
        const vbool<Mx> valid = sphere.template valid<Mx>();
        return SphereIntersectorK<Mx, K>::intersect(
            valid,
            ray,
            k,
            pre,
            v0,
            Occluded1KEpilogM<M, Mx, K, filter>(ray, k, context, sphere.geomID(), sphere.primID()));
      }
    };
  }  // namespace isa
}  // namespace embree
