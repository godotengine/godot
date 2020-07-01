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

#include "primitive.h"
#include "../subdiv/bezier_curve.h"
#include "../common/primref.h"
#include "curve_intersector_precalculations.h"
#include "../bvh/node_intersector1.h"
#include "../bvh/node_intersector_packet.h"


namespace embree
{
  struct VirtualCurveIntersector
  {
    typedef void (*Intersect1Ty)(void* pre, void* ray, IntersectContext* context, const void* primitive);
    typedef bool (*Occluded1Ty )(void* pre, void* ray, IntersectContext* context, const void* primitive);
    
    typedef void (*Intersect4Ty)(void* pre, void* ray, size_t k, IntersectContext* context, const void* primitive);
    typedef bool (*Occluded4Ty) (void* pre, void* ray, size_t k, IntersectContext* context, const void* primitive);
    
    typedef void (*Intersect8Ty)(void* pre, void* ray, size_t k, IntersectContext* context, const void* primitive);
    typedef bool (*Occluded8Ty) (void* pre, void* ray, size_t k, IntersectContext* context, const void* primitive);
    
    typedef void (*Intersect16Ty)(void* pre, void* ray, size_t k, IntersectContext* context, const void* primitive);
    typedef bool (*Occluded16Ty) (void* pre, void* ray, size_t k, IntersectContext* context, const void* primitive);
    
  public:
    struct Intersectors
    {
      Intersectors() {} // WARNING: Do not zero initialize this, as we otherwise get problems with thread unsafe local static variable initialization (e.g. on VS2013) in curve_intersector_virtual.cpp.
      
      template<int K> void intersect(void* pre, void* ray, IntersectContext* context, const void* primitive);
      template<int K> bool occluded (void* pre, void* ray, IntersectContext* context, const void* primitive);

      template<int K> void intersect(void* pre, void* ray, size_t k, IntersectContext* context, const void* primitive);
      template<int K> bool occluded (void* pre, void* ray, size_t k, IntersectContext* context, const void* primitive);

    public:
      Intersect1Ty intersect1;
      Occluded1Ty  occluded1;
      Intersect4Ty intersect4;
      Occluded4Ty  occluded4;
      Intersect8Ty intersect8;
      Occluded8Ty  occluded8;
      Intersect16Ty intersect16;
      Occluded16Ty  occluded16;
    };
    
    Intersectors vtbl[Geometry::GTY_END];
  };

  template<> __forceinline void VirtualCurveIntersector::Intersectors::intersect<1>(void* pre, void* ray, IntersectContext* context, const void* primitive) { assert(intersect1); intersect1(pre,ray,context,primitive); }
  template<> __forceinline bool VirtualCurveIntersector::Intersectors::occluded<1> (void* pre, void* ray, IntersectContext* context, const void* primitive) { assert(occluded1); return occluded1(pre,ray,context,primitive); }
      
  template<> __forceinline void VirtualCurveIntersector::Intersectors::intersect<4>(void* pre, void* ray, size_t k, IntersectContext* context, const void* primitive) { assert(intersect4); intersect4(pre,ray,k,context,primitive); }
  template<> __forceinline bool VirtualCurveIntersector::Intersectors::occluded<4> (void* pre, void* ray, size_t k, IntersectContext* context, const void* primitive) { assert(occluded4); return occluded4(pre,ray,k,context,primitive); }
      
#if defined(__AVX__)
  template<> __forceinline void VirtualCurveIntersector::Intersectors::intersect<8>(void* pre, void* ray, size_t k, IntersectContext* context, const void* primitive) { assert(intersect8); intersect8(pre,ray,k,context,primitive); }
  template<> __forceinline bool VirtualCurveIntersector::Intersectors::occluded<8> (void* pre, void* ray, size_t k, IntersectContext* context, const void* primitive) { assert(occluded8); return occluded8(pre,ray,k,context,primitive); }
#endif
  
#if defined(__AVX512F__)
  template<> __forceinline void VirtualCurveIntersector::Intersectors::intersect<16>(void* pre, void* ray, size_t k, IntersectContext* context, const void* primitive) { assert(intersect16); intersect16(pre,ray,k,context,primitive); }
  template<> __forceinline bool VirtualCurveIntersector::Intersectors::occluded<16> (void* pre, void* ray, size_t k, IntersectContext* context, const void* primitive) { assert(occluded16); return occluded16(pre,ray,k,context,primitive); }
#endif
  
  namespace isa
  {
    struct VirtualCurveIntersector1
    {
      typedef unsigned char Primitive;
      typedef CurvePrecalculations1 Precalculations;
      
      template<int N, int Nx, bool robust>
        static __forceinline void intersect(const Accel::Intersectors* This, Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive* prim, size_t num, const TravRay<N,Nx,robust> &tray, size_t& lazy_node)
      {
        assert(num == 1);
        RTCGeometryType ty = (RTCGeometryType)(*prim);
        assert(This->leafIntersector);
        VirtualCurveIntersector::Intersectors& leafIntersector = ((VirtualCurveIntersector*) This->leafIntersector)->vtbl[ty];
        leafIntersector.intersect<1>(&pre,&ray,context,prim);
      }

      template<int N, int Nx, bool robust>      
        static __forceinline bool occluded(const Accel::Intersectors* This, Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive* prim, size_t num, const TravRay<N,Nx,robust> &tray, size_t& lazy_node)
      {
        assert(num == 1);
        RTCGeometryType ty = (RTCGeometryType)(*prim);
        assert(This->leafIntersector);
        VirtualCurveIntersector::Intersectors& leafIntersector = ((VirtualCurveIntersector*) This->leafIntersector)->vtbl[ty];
        return leafIntersector.occluded<1>(&pre,&ray,context,prim);
      }
    };

    template<int K>
      struct VirtualCurveIntersectorK 
      {
        typedef unsigned char Primitive;
        typedef CurvePrecalculationsK<K> Precalculations;
        
        template<bool robust>        
        static __forceinline void intersect(const vbool<K>& valid_i, const Accel::Intersectors* This, Precalculations& pre, RayHitK<K>& ray, IntersectContext* context, const Primitive* prim, size_t num, const TravRayK<K, robust> &tray, size_t& lazy_node)
        {
          assert(num == 1);
          RTCGeometryType ty = (RTCGeometryType)(*prim);
          assert(This->leafIntersector);
          VirtualCurveIntersector::Intersectors& leafIntersector = ((VirtualCurveIntersector*) This->leafIntersector)->vtbl[ty];
          size_t mask = movemask(valid_i);
          while (mask) leafIntersector.intersect<K>(&pre,&ray,bscf(mask),context,prim);
        }
        
        template<bool robust>        
        static __forceinline vbool<K> occluded(const vbool<K>& valid_i, const Accel::Intersectors* This, Precalculations& pre, RayK<K>& ray, IntersectContext* context, const Primitive* prim, size_t num, const TravRayK<K, robust> &tray, size_t& lazy_node)
        {
          assert(num == 1);
          RTCGeometryType ty = (RTCGeometryType)(*prim);
          assert(This->leafIntersector);
          VirtualCurveIntersector::Intersectors& leafIntersector = ((VirtualCurveIntersector*) This->leafIntersector)->vtbl[ty];
          vbool<K> valid_o = false;
          size_t mask = movemask(valid_i);
          while (mask) {
            size_t k = bscf(mask);
            if (leafIntersector.occluded<K>(&pre,&ray,k,context,prim))
              set(valid_o, k);
          }
          return valid_o;
        }

        template<int N, int Nx, bool robust>              
        static __forceinline void intersect(const Accel::Intersectors* This, Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive* prim, size_t num, const TravRay<N,Nx,robust> &tray, size_t& lazy_node)
        {
          assert(num == 1);
          RTCGeometryType ty = (RTCGeometryType)(*prim);
          assert(This->leafIntersector);
          VirtualCurveIntersector::Intersectors& leafIntersector = ((VirtualCurveIntersector*) This->leafIntersector)->vtbl[ty];
          leafIntersector.intersect<K>(&pre,&ray,k,context,prim);
        }
        
        template<int N, int Nx, bool robust>      
        static __forceinline bool occluded(const Accel::Intersectors* This, Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive* prim, size_t num, const TravRay<N,Nx,robust> &tray, size_t& lazy_node)
        {
          assert(num == 1);
          RTCGeometryType ty = (RTCGeometryType)(*prim);
          assert(This->leafIntersector);
          VirtualCurveIntersector::Intersectors& leafIntersector = ((VirtualCurveIntersector*) This->leafIntersector)->vtbl[ty];
          return leafIntersector.occluded<K>(&pre,&ray,k,context,prim);
        }
      };
  }
}
