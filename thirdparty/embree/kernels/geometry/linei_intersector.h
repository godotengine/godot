// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "linei.h"
#include "line_intersector.h"
#include "intersector_epilog.h"

namespace embree
{
  namespace isa
  {
    template<int M, int Mx, bool filter>
    struct FlatLinearCurveMiIntersector1
    {
      typedef LineMi<M> Primitive;
      typedef CurvePrecalculations1 Precalculations;

      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive& line)
      {
        STAT3(normal.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1; line.gather(v0,v1,geom);
        const vbool<Mx> valid = line.template valid<Mx>();
        FlatLinearCurveIntersector1<Mx>::intersect(valid,ray,context,geom,pre,v0,v1,Intersect1EpilogM<M,Mx,filter>(ray,context,line.geomID(),line.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive& line)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1; line.gather(v0,v1,geom);
        const vbool<Mx> valid = line.template valid<Mx>();
        return FlatLinearCurveIntersector1<Mx>::intersect(valid,ray,context,geom,pre,v0,v1,Occluded1EpilogM<M,Mx,filter>(ray,context,line.geomID(),line.primID()));
      }
      
      static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const Primitive& line)
      {
        return PrimitivePointQuery1<Primitive>::pointQuery(query, context, line);
      }
    };

    template<int M, int Mx, bool filter>
    struct FlatLinearCurveMiMBIntersector1
    {
      typedef LineMi<M> Primitive;
      typedef CurvePrecalculations1 Precalculations;

      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive& line)
      {
        STAT3(normal.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1; line.gather(v0,v1,geom,ray.time());
        const vbool<Mx> valid = line.template valid<Mx>();
        FlatLinearCurveIntersector1<Mx>::intersect(valid,ray,context,geom,pre,v0,v1,Intersect1EpilogM<M,Mx,filter>(ray,context,line.geomID(),line.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive& line)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1; line.gather(v0,v1,geom,ray.time());
        const vbool<Mx> valid = line.template valid<Mx>();
        return FlatLinearCurveIntersector1<Mx>::intersect(valid,ray,context,geom,pre,v0,v1,Occluded1EpilogM<M,Mx,filter>(ray,context,line.geomID(),line.primID()));
      }
      
      static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const Primitive& line)
      {
        return PrimitivePointQuery1<Primitive>::pointQuery(query, context, line);
      }
    };

    template<int M, int Mx, int K, bool filter>
    struct FlatLinearCurveMiIntersectorK
    {
      typedef LineMi<M> Primitive;
      typedef CurvePrecalculationsK<K> Precalculations;

      static __forceinline void intersect(const Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive& line)
      {
        STAT3(normal.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1; line.gather(v0,v1,geom);
        const vbool<Mx> valid = line.template valid<Mx>();
        FlatLinearCurveIntersectorK<Mx,K>::intersect(valid,ray,k,context,geom,pre,v0,v1,Intersect1KEpilogM<M,Mx,K,filter>(ray,k,context,line.geomID(),line.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive& line)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1; line.gather(v0,v1,geom);
        const vbool<Mx> valid = line.template valid<Mx>();
        return FlatLinearCurveIntersectorK<Mx,K>::intersect(valid,ray,k,context,geom,pre,v0,v1,Occluded1KEpilogM<M,Mx,K,filter>(ray,k,context,line.geomID(),line.primID()));
      }
    };

    template<int M, int Mx, int K, bool filter>
    struct FlatLinearCurveMiMBIntersectorK
    {
      typedef LineMi<M> Primitive;
      typedef CurvePrecalculationsK<K> Precalculations;

      static __forceinline void intersect(const Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context,  const Primitive& line)
      {
        STAT3(normal.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1; line.gather(v0,v1,geom,ray.time()[k]);
        const vbool<Mx> valid = line.template valid<Mx>();
        FlatLinearCurveIntersectorK<Mx,K>::intersect(valid,ray,k,context,geom,pre,v0,v1,Intersect1KEpilogM<M,Mx,K,filter>(ray,k,context,line.geomID(),line.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive& line)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1; line.gather(v0,v1,geom,ray.time()[k]);
        const vbool<Mx> valid = line.template valid<Mx>();
        return FlatLinearCurveIntersectorK<Mx,K>::intersect(valid,ray,k,context,geom,pre,v0,v1,Occluded1KEpilogM<M,Mx,K,filter>(ray,k,context,line.geomID(),line.primID()));
      }
    };
  }
}
