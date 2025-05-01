// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "primitive.h"
#include "../subdiv/bezier_curve.h"
#include "../common/primref.h"
#include "bezier_hair_intersector.h"
#include "bezier_ribbon_intersector.h"
#include "bezier_curve_intersector.h"
#include "oriented_curve_intersector.h"
#include "../bvh/node_intersector1.h"

// FIXME: this file seems replicate of curve_intersector_virtual.h

namespace embree
{
  namespace isa
  {
    struct VirtualCurveIntersector1
    {
      typedef unsigned char Primitive;
      typedef CurvePrecalculations1 Precalculations;
      
      template<int N, bool robust>
        static __forceinline void intersect(const Accel::Intersectors* This, Precalculations& pre, RayHit& ray, RayQueryContext* context, const Primitive* prim, size_t num, const TravRay<N,robust> &tray, size_t& lazy_node)
      {
        assert(num == 1);
        RTCGeometryType ty = (RTCGeometryType)(*prim);
        assert(This->leafIntersector);
        VirtualCurvePrimitive::Intersectors& leafIntersector = ((VirtualCurvePrimitive*) This->leafIntersector)->vtbl[ty];
        leafIntersector.intersect<1>(&pre,&ray,context,prim);
      }
      
      template<int N, bool robust>        
        static __forceinline bool occluded(const Accel::Intersectors* This, Precalculations& pre, Ray& ray, RayQueryContext* context, const Primitive* prim, size_t num, const TravRay<N,robust> &tray, size_t& lazy_node)
      {
        assert(num == 1);
        RTCGeometryType ty = (RTCGeometryType)(*prim);
        assert(This->leafIntersector);
        VirtualCurvePrimitive::Intersectors& leafIntersector = ((VirtualCurvePrimitive*) This->leafIntersector)->vtbl[ty];
        return leafIntersector.occluded<1>(&pre,&ray,context,prim);
      }
    };

    template<int K>
      struct VirtualCurveIntersectorK 
      {
        typedef unsigned char Primitive;
        typedef CurvePrecalculationsK<K> Precalculations;
        
        static __forceinline void intersect(const vbool<K>& valid_i, const Accel::Intersectors* This, Precalculations& pre, RayHitK<K>& ray, RayQueryContext* context, const Primitive* prim, size_t num, size_t& lazy_node)
        {
          assert(num == 1);
          RTCGeometryType ty = (RTCGeometryType)(*prim);
          assert(This->leafIntersector);
          VirtualCurvePrimitive::Intersectors& leafIntersector = ((VirtualCurvePrimitive*) This->leafIntersector)->vtbl[ty];
          size_t mask = movemask(valid_i);
          while (mask) leafIntersector.intersect<K>(&pre,&ray,bscf(mask),context,prim);
        }
        
        static __forceinline vbool<K> occluded(const vbool<K>& valid_i, const Accel::Intersectors* This, Precalculations& pre, RayK<K>& ray, RayQueryContext* context, const Primitive* prim, size_t num, size_t& lazy_node)
        {
          assert(num == 1);
          RTCGeometryType ty = (RTCGeometryType)(*prim);
          assert(This->leafIntersector);
          VirtualCurvePrimitive::Intersectors& leafIntersector = ((VirtualCurvePrimitive*) This->leafIntersector)->vtbl[ty];
          vbool<K> valid_o = false;
          size_t mask = movemask(valid_i);
          while (mask) {
            size_t k = bscf(mask);
            if (leafIntersector.occluded<K>(&pre,&ray,k,context,prim))
              set(valid_o, k);
          }
          return valid_o;
        }
        
        static __forceinline void intersect(const Accel::Intersectors* This, Precalculations& pre, RayHitK<K>& ray, size_t k, RayQueryContext* context, const Primitive* prim, size_t num, size_t& lazy_node)
        {
          assert(num == 1);
          RTCGeometryType ty = (RTCGeometryType)(*prim);
          assert(This->leafIntersector);
          VirtualCurvePrimitive::Intersectors& leafIntersector = ((VirtualCurvePrimitive*) This->leafIntersector)->vtbl[ty];
          leafIntersector.intersect<K>(&pre,&ray,k,context,prim);
        }
        
        static __forceinline bool occluded(const Accel::Intersectors* This, Precalculations& pre, RayK<K>& ray, size_t k, RayQueryContext* context, const Primitive* prim, size_t num, size_t& lazy_node)
        {
          assert(num == 1);
          RTCGeometryType ty = (RTCGeometryType)(*prim);
          assert(This->leafIntersector);
          VirtualCurvePrimitive::Intersectors& leafIntersector = ((VirtualCurvePrimitive*) This->leafIntersector)->vtbl[ty];
          return leafIntersector.occluded<K>(&pre,&ray,k,context,prim);
        }
      };
  }
}
