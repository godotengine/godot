// ======================================================================== //
// Copyright 2009-2020 Intel Corporation                                    //
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

#include "roundline_intersector.h"
#include "intersector_epilog.h"

namespace embree
{
  namespace isa
  {
    template<int M, int Mx, bool filter>
    struct RoundLinearCurveMiIntersector1
    {
      typedef LineMi<M> Primitive;
      typedef CurvePrecalculations1 Precalculations;

      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive& line)
      {
        STAT3(normal.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1,vL,vR; line.gather(v0,v1,vL,vR,geom);
        const vbool<Mx> valid = line.template valid<Mx>();
        RoundLinearCurveIntersector1<Mx>::intersect(valid,ray,context,geom,pre,v0,v1,vL,vR,Intersect1EpilogM<M,Mx,filter>(ray,context,line.geomID(),line.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive& line)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1,vL,vR; line.gather(v0,v1,vL,vR,geom);
        const vbool<Mx> valid = line.template valid<Mx>();
        return RoundLinearCurveIntersector1<Mx>::intersect(valid,ray,context,geom,pre,v0,v1,vL,vR,Occluded1EpilogM<M,Mx,filter>(ray,context,line.geomID(),line.primID()));
      }
      
      static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const Primitive& line)
      {
        return PrimitivePointQuery1<Primitive>::pointQuery(query, context, line);
      }
    };

    template<int M, int Mx, bool filter>
    struct RoundLinearCurveMiMBIntersector1
    {
      typedef LineMi<M> Primitive;
      typedef CurvePrecalculations1 Precalculations;

      static __forceinline void intersect(const Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive& line)
      {
        STAT3(normal.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1,vL,vR; line.gather(v0,v1,vL,vR,geom,ray.time());
        const vbool<Mx> valid = line.template valid<Mx>();
        RoundLinearCurveIntersector1<Mx>::intersect(valid,ray,context,geom,pre,v0,v1,vL,vR,Intersect1EpilogM<M,Mx,filter>(ray,context,line.geomID(),line.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive& line)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1,vL,vR; line.gather(v0,v1,vL,vR,geom,ray.time());
        const vbool<Mx> valid = line.template valid<Mx>();
        return RoundLinearCurveIntersector1<Mx>::intersect(valid,ray,context,geom,pre,v0,v1,vL,vR,Occluded1EpilogM<M,Mx,filter>(ray,context,line.geomID(),line.primID()));
      }
      
      static __forceinline bool pointQuery(PointQuery* query, PointQueryContext* context, const Primitive& line)
      {
        return PrimitivePointQuery1<Primitive>::pointQuery(query, context, line);
      }
    };

    template<int M, int Mx, int K, bool filter>
    struct RoundLinearCurveMiIntersectorK
    {
      typedef LineMi<M> Primitive;
      typedef CurvePrecalculationsK<K> Precalculations;

      static __forceinline void intersect(const Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive& line)
      {
        STAT3(normal.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1,vL,vR; line.gather(v0,v1,vL,vR,geom);
        const vbool<Mx> valid = line.template valid<Mx>();
        RoundLinearCurveIntersectorK<Mx,K>::intersect(valid,ray,k,context,geom,pre,v0,v1,vL,vR,Intersect1KEpilogM<M,Mx,K,filter>(ray,k,context,line.geomID(),line.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive& line)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1,vL,vR; line.gather(v0,v1,vL,vR,geom);
        const vbool<Mx> valid = line.template valid<Mx>();
        return RoundLinearCurveIntersectorK<Mx,K>::intersect(valid,ray,k,context,geom,pre,v0,v1,vL,vR,Occluded1KEpilogM<M,Mx,K,filter>(ray,k,context,line.geomID(),line.primID()));
      }
    };

    template<int M, int Mx, int K, bool filter>
    struct RoundLinearCurveMiMBIntersectorK
    {
      typedef LineMi<M> Primitive;
      typedef CurvePrecalculationsK<K> Precalculations;

      static __forceinline void intersect(const Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context,  const Primitive& line)
      {
        STAT3(normal.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1,vL,vR; line.gather(v0,v1,vL,vR,geom,ray.time()[k]);
        const vbool<Mx> valid = line.template valid<Mx>();
        RoundLinearCurveIntersectorK<Mx,K>::intersect(valid,ray,k,context,geom,pre,v0,v1,vL,vR,Intersect1KEpilogM<M,Mx,K,filter>(ray,k,context,line.geomID(),line.primID()));
      }

      static __forceinline bool occluded(const Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive& line)
      {
        STAT3(shadow.trav_prims,1,1,1);
        const LineSegments* geom = context->scene->get<LineSegments>(line.geomID());
        Vec4vf<M> v0,v1,vL,vR; line.gather(v0,v1,vL,vR,geom,ray.time()[k]);
        const vbool<Mx> valid = line.template valid<Mx>();
        return RoundLinearCurveIntersectorK<Mx,K>::intersect(valid,ray,k,context,geom,pre,v0,v1,vL,vR,Occluded1KEpilogM<M,Mx,K,filter>(ray,k,context,line.geomID(),line.primID()));
      }
    };
  }
}
