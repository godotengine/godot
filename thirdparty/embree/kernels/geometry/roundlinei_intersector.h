// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "roundline_intersector.h"
#include "intersector_epilog.h"

namespace embree
{
  namespace isa
  {
    template<int M, bool filter>
    struct RoundLinearCurveMiIntersector1
    {
      typedef LineMi<M> Primitive;
      typedef CurvePrecalculations1 Precalculations;

      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive& line)
      {
        STAT3(normal.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1,vL,vR; line.gather(v0,v1,vL,vR,geom);
        const vbool<M> valid = line.valid();
        RoundLinearCurveIntersector1<M>::intersect(valid,ray,context,geom,pre,v0,v1,vL,vR,Intersect1EpilogM<M,filter>(ray,context,line.geomID(),line.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive& line)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1,vL,vR; line.gather(v0,v1,vL,vR,geom);
        const vbool<M> valid = line.valid();
        return RoundLinearCurveIntersector1<M>::intersect(valid,ray,context,geom,pre,v0,v1,vL,vR,Occluded1EpilogM<M,filter>(ray,context,line.geomID(),line.primID()));
      }
      
      static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const Primitive& line)
      {
        return PrimitivePointQuery1<Primitive>::pointQuery(query, context, line);
      }
    };

    template<int M, bool filter>
    struct RoundLinearCurveMiMBIntersector1
    {
      typedef LineMi<M> Primitive;
      typedef CurvePrecalculations1 Precalculations;

      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive& line)
      {
        STAT3(normal.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1,vL,vR; line.gather(v0,v1,vL,vR,geom,ray.time());
        const vbool<M> valid = line.valid();
        RoundLinearCurveIntersector1<M>::intersect(valid,ray,context,geom,pre,v0,v1,vL,vR,Intersect1EpilogM<M,filter>(ray,context,line.geomID(),line.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive& line)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1,vL,vR; line.gather(v0,v1,vL,vR,geom,ray.time());
        const vbool<M> valid = line.valid();
        return RoundLinearCurveIntersector1<M>::intersect(valid,ray,context,geom,pre,v0,v1,vL,vR,Occluded1EpilogM<M,filter>(ray,context,line.geomID(),line.primID()));
      }
      
      static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const Primitive& line)
      {
        return PrimitivePointQuery1<Primitive>::pointQuery(query, context, line);
      }
    };

    template<int M, int K, bool filter>
    struct RoundLinearCurveMiIntersectorK
    {
      typedef LineMi<M> Primitive;
      typedef CurvePrecalculationsK<K> Precalculations;

      static __forceinline void intersect(const Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive& line)
      {
        STAT3(normal.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1,vL,vR; line.gather(v0,v1,vL,vR,geom);
        const vbool<M> valid = line.valid();
        RoundLinearCurveIntersectorK<M,K>::intersect(valid,ray,k,context,geom,pre,v0,v1,vL,vR,Intersect1KEpilogM<M,K,filter>(ray,k,context,line.geomID(),line.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive& line)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1,vL,vR; line.gather(v0,v1,vL,vR,geom);
        const vbool<M> valid = line.valid();
        return RoundLinearCurveIntersectorK<M,K>::intersect(valid,ray,k,context,geom,pre,v0,v1,vL,vR,Occluded1KEpilogM<M,K,filter>(ray,k,context,line.geomID(),line.primID()));
      }
    };

    template<int M, int K, bool filter>
    struct RoundLinearCurveMiMBIntersectorK
    {
      typedef LineMi<M> Primitive;
      typedef CurvePrecalculationsK<K> Precalculations;

      static __forceinline void intersect(const Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context,  const Primitive& line)
      {
        STAT3(normal.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1,vL,vR; line.gather(v0,v1,vL,vR,geom,ray.time()[k]);
        const vbool<M> valid = line.valid();
        RoundLinearCurveIntersectorK<M,K>::intersect(valid,ray,k,context,geom,pre,v0,v1,vL,vR,Intersect1KEpilogM<M,K,filter>(ray,k,context,line.geomID(),line.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive& line)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1,vL,vR; line.gather(v0,v1,vL,vR,geom,ray.time()[k]);
        const vbool<M> valid = line.valid();
        return RoundLinearCurveIntersectorK<M,K>::intersect(valid,ray,k,context,geom,pre,v0,v1,vL,vR,Occluded1KEpilogM<M,K,filter>(ray,k,context,line.geomID(),line.primID()));
      }
    };
  }
}
