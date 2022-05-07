// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "primitive.h"
#include "../subdiv/bezier_curve.h"
#include "../common/primref.h"
#include "curve_intersector_precalculations.h"
#include "../bvh/node_intersector1.h"
#include "../bvh/node_intersector_packet.h"

#include "intersector_epilog.h"

#include "../subdiv/bezier_curve.h"
#include "../subdiv/bspline_curve.h"
#include "../subdiv/hermite_curve.h"
#include "../subdiv/catmullrom_curve.h"

#include "spherei_intersector.h"
#include "disci_intersector.h"

#include "linei_intersector.h"
#include "roundlinei_intersector.h"
#include "conelinei_intersector.h"

#include "curveNi_intersector.h"
#include "curveNv_intersector.h"
#include "curveNi_mb_intersector.h"

#include "curve_intersector_distance.h"
#include "curve_intersector_ribbon.h"
#include "curve_intersector_oriented.h"
#include "curve_intersector_sweep.h"

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

  template<> __forceinline void VirtualCurveIntersector::Intersectors::intersect<1> (void* pre, void* ray, IntersectContext* context, const void* primitive) { assert(intersect1); intersect1(pre,ray,context,primitive); }
  template<> __forceinline bool VirtualCurveIntersector::Intersectors::occluded<1>  (void* pre, void* ray, IntersectContext* context, const void* primitive) { assert(occluded1); return occluded1(pre,ray,context,primitive); }
      
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
      
      template<int N, bool robust>
        static __forceinline void intersect(const Accel::Intersectors* This, Precalculations& pre, RayHit& ray, IntersectContext* context, const Primitive* prim, size_t num, const TravRay<N,robust> &tray, size_t& lazy_node)
      {
        assert(num == 1);
        RTCGeometryType ty = (RTCGeometryType)(*prim);
        assert(This->leafIntersector);
        VirtualCurveIntersector::Intersectors& leafIntersector = ((VirtualCurveIntersector*) This->leafIntersector)->vtbl[ty];
        leafIntersector.intersect<1>(&pre,&ray,context,prim);
      }

      template<int N, bool robust>      
        static __forceinline bool occluded(const Accel::Intersectors* This, Precalculations& pre, Ray& ray, IntersectContext* context, const Primitive* prim, size_t num, const TravRay<N,robust> &tray, size_t& lazy_node)
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
        
        template<int N, bool robust>              
        static __forceinline void intersect(const Accel::Intersectors* This, Precalculations& pre, RayHitK<K>& ray, size_t k, IntersectContext* context, const Primitive* prim, size_t num, const TravRay<N,robust> &tray, size_t& lazy_node)
        {
          assert(num == 1);
          RTCGeometryType ty = (RTCGeometryType)(*prim);
          assert(This->leafIntersector);
          VirtualCurveIntersector::Intersectors& leafIntersector = ((VirtualCurveIntersector*) This->leafIntersector)->vtbl[ty];
          leafIntersector.intersect<K>(&pre,&ray,k,context,prim);
        }
        
        template<int N, bool robust>      
        static __forceinline bool occluded(const Accel::Intersectors* This, Precalculations& pre, RayK<K>& ray, size_t k, IntersectContext* context, const Primitive* prim, size_t num, const TravRay<N,robust> &tray, size_t& lazy_node)
        {
          assert(num == 1);
          RTCGeometryType ty = (RTCGeometryType)(*prim);
          assert(This->leafIntersector);
          VirtualCurveIntersector::Intersectors& leafIntersector = ((VirtualCurveIntersector*) This->leafIntersector)->vtbl[ty];
          return leafIntersector.occluded<K>(&pre,&ray,k,context,prim);
        }
      };

    template<int N>
    static VirtualCurveIntersector::Intersectors LinearRoundConeNiIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &RoundLinearCurveMiIntersector1<N,true>::intersect;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &RoundLinearCurveMiIntersector1<N,true>::occluded;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty) &RoundLinearCurveMiIntersectorK<N,4,true>::intersect;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty)  &RoundLinearCurveMiIntersectorK<N,4,true>::occluded;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&RoundLinearCurveMiIntersectorK<N,8,true>::intersect;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &RoundLinearCurveMiIntersectorK<N,8,true>::occluded;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&RoundLinearCurveMiIntersectorK<N,16,true>::intersect;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &RoundLinearCurveMiIntersectorK<N,16,true>::occluded;
#endif
      return intersectors;
    }

    template<int N>
    static VirtualCurveIntersector::Intersectors LinearConeNiIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &ConeCurveMiIntersector1<N,true>::intersect;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &ConeCurveMiIntersector1<N,true>::occluded;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty) &ConeCurveMiIntersectorK<N,4,true>::intersect;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty)  &ConeCurveMiIntersectorK<N,4,true>::occluded;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&ConeCurveMiIntersectorK<N,8,true>::intersect;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &ConeCurveMiIntersectorK<N,8,true>::occluded;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&ConeCurveMiIntersectorK<N,16,true>::intersect;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &ConeCurveMiIntersectorK<N,16,true>::occluded;
#endif
      return intersectors;
    }

    template<int N>
    static VirtualCurveIntersector::Intersectors LinearRoundConeNiMBIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &RoundLinearCurveMiMBIntersector1<N,true>::intersect;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &RoundLinearCurveMiMBIntersector1<N,true>::occluded;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty) &RoundLinearCurveMiMBIntersectorK<N,4,true>::intersect;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty)  &RoundLinearCurveMiMBIntersectorK<N,4,true>::occluded;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&RoundLinearCurveMiMBIntersectorK<N,8,true>::intersect;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &RoundLinearCurveMiMBIntersectorK<N,8,true>::occluded;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&RoundLinearCurveMiMBIntersectorK<N,16,true>::intersect;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &RoundLinearCurveMiMBIntersectorK<N,16,true>::occluded;
#endif
      return intersectors;
    }

    template<int N>
    static VirtualCurveIntersector::Intersectors LinearConeNiMBIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &ConeCurveMiMBIntersector1<N,true>::intersect;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &ConeCurveMiMBIntersector1<N,true>::occluded;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty) &ConeCurveMiMBIntersectorK<N,4,true>::intersect;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty)  &ConeCurveMiMBIntersectorK<N,4,true>::occluded;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&ConeCurveMiMBIntersectorK<N,8,true>::intersect;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &ConeCurveMiMBIntersectorK<N,8,true>::occluded;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&ConeCurveMiMBIntersectorK<N,16,true>::intersect;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &ConeCurveMiMBIntersectorK<N,16,true>::occluded;
#endif
      return intersectors;
    }


    template<int N>
      static VirtualCurveIntersector::Intersectors LinearRibbonNiIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &FlatLinearCurveMiIntersector1<N,true>::intersect;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &FlatLinearCurveMiIntersector1<N,true>::occluded;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty) &FlatLinearCurveMiIntersectorK<N,4,true>::intersect;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty)  &FlatLinearCurveMiIntersectorK<N,4,true>::occluded;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&FlatLinearCurveMiIntersectorK<N,8,true>::intersect;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &FlatLinearCurveMiIntersectorK<N,8,true>::occluded;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&FlatLinearCurveMiIntersectorK<N,16,true>::intersect;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &FlatLinearCurveMiIntersectorK<N,16,true>::occluded;
#endif
      return intersectors;
    }
    
    template<int N>
      static VirtualCurveIntersector::Intersectors LinearRibbonNiMBIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &FlatLinearCurveMiMBIntersector1<N,true>::intersect;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &FlatLinearCurveMiMBIntersector1<N,true>::occluded;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty) &FlatLinearCurveMiMBIntersectorK<N,4,true>::intersect;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty)  &FlatLinearCurveMiMBIntersectorK<N,4,true>::occluded;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&FlatLinearCurveMiMBIntersectorK<N,8,true>::intersect;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &FlatLinearCurveMiMBIntersectorK<N,8,true>::occluded;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&FlatLinearCurveMiMBIntersectorK<N,16,true>::intersect;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &FlatLinearCurveMiMBIntersectorK<N,16,true>::occluded;
#endif
      return intersectors;
    }
    
    template<int N>
      static VirtualCurveIntersector::Intersectors SphereNiIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &SphereMiIntersector1<N,true>::intersect;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &SphereMiIntersector1<N,true>::occluded;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty) &SphereMiIntersectorK<N,4,true>::intersect;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty)  &SphereMiIntersectorK<N,4,true>::occluded;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&SphereMiIntersectorK<N,8,true>::intersect;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &SphereMiIntersectorK<N,8,true>::occluded;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&SphereMiIntersectorK<N,16,true>::intersect;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &SphereMiIntersectorK<N,16,true>::occluded;
#endif
      return intersectors;
    }
    
    template<int N>
      static VirtualCurveIntersector::Intersectors SphereNiMBIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &SphereMiMBIntersector1<N,true>::intersect;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &SphereMiMBIntersector1<N,true>::occluded;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty) &SphereMiMBIntersectorK<N,4,true>::intersect;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty)  &SphereMiMBIntersectorK<N,4,true>::occluded;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&SphereMiMBIntersectorK<N,8,true>::intersect;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &SphereMiMBIntersectorK<N,8,true>::occluded;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&SphereMiMBIntersectorK<N,16,true>::intersect;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &SphereMiMBIntersectorK<N,16,true>::occluded;
#endif
      return intersectors;
    }
    
    template<int N>
      static VirtualCurveIntersector::Intersectors DiscNiIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &DiscMiIntersector1<N,true>::intersect;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &DiscMiIntersector1<N,true>::occluded;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty) &DiscMiIntersectorK<N,4,true>::intersect;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty)  &DiscMiIntersectorK<N,4,true>::occluded;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&DiscMiIntersectorK<N,8,true>::intersect;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &DiscMiIntersectorK<N,8,true>::occluded;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&DiscMiIntersectorK<N,16,true>::intersect;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &DiscMiIntersectorK<N,16,true>::occluded;
#endif
      return intersectors;
    }
    
    template<int N>
      static VirtualCurveIntersector::Intersectors DiscNiMBIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &DiscMiMBIntersector1<N,true>::intersect;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &DiscMiMBIntersector1<N,true>::occluded;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty) &DiscMiMBIntersectorK<N,4,true>::intersect;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty)  &DiscMiMBIntersectorK<N,4,true>::occluded;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&DiscMiMBIntersectorK<N,8,true>::intersect;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &DiscMiMBIntersectorK<N,8,true>::occluded;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&DiscMiMBIntersectorK<N,16,true>::intersect;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &DiscMiMBIntersectorK<N,16,true>::occluded;
#endif
      return intersectors;
    }
    
    template<int N>
      static VirtualCurveIntersector::Intersectors OrientedDiscNiIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &OrientedDiscMiIntersector1<N,true>::intersect;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &OrientedDiscMiIntersector1<N,true>::occluded;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty) &OrientedDiscMiIntersectorK<N,4,true>::intersect;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty)  &OrientedDiscMiIntersectorK<N,4,true>::occluded;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&OrientedDiscMiIntersectorK<N,8,true>::intersect;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &OrientedDiscMiIntersectorK<N,8,true>::occluded;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&OrientedDiscMiIntersectorK<N,16,true>::intersect;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &OrientedDiscMiIntersectorK<N,16,true>::occluded;
#endif
      return intersectors;
    }
    
    template<int N>
      static VirtualCurveIntersector::Intersectors OrientedDiscNiMBIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &OrientedDiscMiMBIntersector1<N,true>::intersect;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &OrientedDiscMiMBIntersector1<N,true>::occluded;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty) &OrientedDiscMiMBIntersectorK<N,4,true>::intersect;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty)  &OrientedDiscMiMBIntersectorK<N,4,true>::occluded;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&OrientedDiscMiMBIntersectorK<N,8,true>::intersect;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &OrientedDiscMiMBIntersectorK<N,8,true>::occluded;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&OrientedDiscMiMBIntersectorK<N,16,true>::intersect;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &OrientedDiscMiMBIntersectorK<N,16,true>::occluded;
#endif
      return intersectors;
    }
    
    template<template<typename Ty> class Curve, int N>
      static VirtualCurveIntersector::Intersectors RibbonNiIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &CurveNiIntersector1<N>::template intersect_t<RibbonCurve1Intersector1<Curve>, Intersect1EpilogMU<VSIZEX,true> >;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &CurveNiIntersector1<N>::template occluded_t <RibbonCurve1Intersector1<Curve>, Occluded1EpilogMU<VSIZEX,true> >;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty) &CurveNiIntersectorK<N,4>::template intersect_t<RibbonCurve1IntersectorK<Curve,4>, Intersect1KEpilogMU<VSIZEX,4,true> >;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty)  &CurveNiIntersectorK<N,4>::template occluded_t <RibbonCurve1IntersectorK<Curve,4>, Occluded1KEpilogMU<VSIZEX,4,true> >;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&CurveNiIntersectorK<N,8>::template intersect_t<RibbonCurve1IntersectorK<Curve,8>, Intersect1KEpilogMU<VSIZEX,8,true> >;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &CurveNiIntersectorK<N,8>::template occluded_t <RibbonCurve1IntersectorK<Curve,8>, Occluded1KEpilogMU<VSIZEX,8,true> >;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&CurveNiIntersectorK<N,16>::template intersect_t<RibbonCurve1IntersectorK<Curve,16>, Intersect1KEpilogMU<VSIZEX,16,true> >;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &CurveNiIntersectorK<N,16>::template occluded_t <RibbonCurve1IntersectorK<Curve,16>, Occluded1KEpilogMU<VSIZEX,16,true> >;
#endif
      return intersectors;
    }
    
    template<template<typename Ty> class Curve, int N>
      static VirtualCurveIntersector::Intersectors RibbonNvIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &CurveNvIntersector1<N>::template intersect_t<RibbonCurve1Intersector1<Curve>, Intersect1EpilogMU<VSIZEX,true> >;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &CurveNvIntersector1<N>::template occluded_t <RibbonCurve1Intersector1<Curve>, Occluded1EpilogMU<VSIZEX,true> >;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty) &CurveNvIntersectorK<N,4>::template intersect_t<RibbonCurve1IntersectorK<Curve,4>, Intersect1KEpilogMU<VSIZEX,4,true> >;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty)  &CurveNvIntersectorK<N,4>::template occluded_t <RibbonCurve1IntersectorK<Curve,4>, Occluded1KEpilogMU<VSIZEX,4,true> >;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&CurveNvIntersectorK<N,8>::template intersect_t<RibbonCurve1IntersectorK<Curve,8>, Intersect1KEpilogMU<VSIZEX,8,true> >;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &CurveNvIntersectorK<N,8>::template occluded_t <RibbonCurve1IntersectorK<Curve,8>, Occluded1KEpilogMU<VSIZEX,8,true> >;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&CurveNvIntersectorK<N,16>::template intersect_t<RibbonCurve1IntersectorK<Curve,16>, Intersect1KEpilogMU<VSIZEX,16,true> >;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &CurveNvIntersectorK<N,16>::template occluded_t <RibbonCurve1IntersectorK<Curve,16>, Occluded1KEpilogMU<VSIZEX,16,true> >;
#endif
      return intersectors;
    }
    
    template<template<typename Ty> class Curve, int N>
      static VirtualCurveIntersector::Intersectors RibbonNiMBIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &CurveNiMBIntersector1<N>::template intersect_t<RibbonCurve1Intersector1<Curve>, Intersect1EpilogMU<VSIZEX,true> >;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &CurveNiMBIntersector1<N>::template occluded_t <RibbonCurve1Intersector1<Curve>, Occluded1EpilogMU<VSIZEX,true> >;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty) &CurveNiMBIntersectorK<N,4>::template intersect_t<RibbonCurve1IntersectorK<Curve,4>, Intersect1KEpilogMU<VSIZEX,4,true> >;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty)  &CurveNiMBIntersectorK<N,4>::template occluded_t <RibbonCurve1IntersectorK<Curve,4>, Occluded1KEpilogMU<VSIZEX,4,true> >;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&CurveNiMBIntersectorK<N,8>::template intersect_t<RibbonCurve1IntersectorK<Curve,8>, Intersect1KEpilogMU<VSIZEX,8,true> >;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &CurveNiMBIntersectorK<N,8>::template occluded_t <RibbonCurve1IntersectorK<Curve,8>, Occluded1KEpilogMU<VSIZEX,8,true> >;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&CurveNiMBIntersectorK<N,16>::template intersect_t<RibbonCurve1IntersectorK<Curve,16>, Intersect1KEpilogMU<VSIZEX,16,true> >;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &CurveNiMBIntersectorK<N,16>::template occluded_t <RibbonCurve1IntersectorK<Curve,16>, Occluded1KEpilogMU<VSIZEX,16,true> >;
#endif
      return intersectors;
    }
    
    template<template<typename Ty> class Curve, int N>
      static VirtualCurveIntersector::Intersectors CurveNiIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &CurveNiIntersector1<N>::template intersect_t<SweepCurve1Intersector1<Curve>, Intersect1Epilog1<true> >;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &CurveNiIntersector1<N>::template occluded_t <SweepCurve1Intersector1<Curve>, Occluded1Epilog1<true> >;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty)&CurveNiIntersectorK<N,4>::template intersect_t<SweepCurve1IntersectorK<Curve,4>, Intersect1KEpilog1<4,true> >;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty) &CurveNiIntersectorK<N,4>::template occluded_t <SweepCurve1IntersectorK<Curve,4>, Occluded1KEpilog1<4,true> >;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&CurveNiIntersectorK<N,8>::template intersect_t<SweepCurve1IntersectorK<Curve,8>, Intersect1KEpilog1<8,true> >;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &CurveNiIntersectorK<N,8>::template occluded_t <SweepCurve1IntersectorK<Curve,8>, Occluded1KEpilog1<8,true> >;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&CurveNiIntersectorK<N,16>::template intersect_t<SweepCurve1IntersectorK<Curve,16>, Intersect1KEpilog1<16,true> >;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &CurveNiIntersectorK<N,16>::template occluded_t <SweepCurve1IntersectorK<Curve,16>, Occluded1KEpilog1<16,true> >;
#endif
      return intersectors;
    }
    
    template<template<typename Ty> class Curve, int N>
      static VirtualCurveIntersector::Intersectors CurveNvIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &CurveNvIntersector1<N>::template intersect_t<SweepCurve1Intersector1<Curve>, Intersect1Epilog1<true> >;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &CurveNvIntersector1<N>::template occluded_t <SweepCurve1Intersector1<Curve>, Occluded1Epilog1<true> >;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty)&CurveNvIntersectorK<N,4>::template intersect_t<SweepCurve1IntersectorK<Curve,4>, Intersect1KEpilog1<4,true> >;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty) &CurveNvIntersectorK<N,4>::template occluded_t <SweepCurve1IntersectorK<Curve,4>, Occluded1KEpilog1<4,true> >;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&CurveNvIntersectorK<N,8>::template intersect_t<SweepCurve1IntersectorK<Curve,8>, Intersect1KEpilog1<8,true> >;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &CurveNvIntersectorK<N,8>::template occluded_t <SweepCurve1IntersectorK<Curve,8>, Occluded1KEpilog1<8,true> >;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&CurveNvIntersectorK<N,16>::template intersect_t<SweepCurve1IntersectorK<Curve,16>, Intersect1KEpilog1<16,true> >;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &CurveNvIntersectorK<N,16>::template occluded_t <SweepCurve1IntersectorK<Curve,16>, Occluded1KEpilog1<16,true> >;
#endif
      return intersectors;
    }
    
    template<template<typename Ty> class Curve, int N>
      static VirtualCurveIntersector::Intersectors CurveNiMBIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &CurveNiMBIntersector1<N>::template intersect_t<SweepCurve1Intersector1<Curve>, Intersect1Epilog1<true> >;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &CurveNiMBIntersector1<N>::template occluded_t <SweepCurve1Intersector1<Curve>, Occluded1Epilog1<true> >;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty)&CurveNiMBIntersectorK<N,4>::template intersect_t<SweepCurve1IntersectorK<Curve,4>, Intersect1KEpilog1<4,true> >;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty) &CurveNiMBIntersectorK<N,4>::template occluded_t <SweepCurve1IntersectorK<Curve,4>, Occluded1KEpilog1<4,true> >;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&CurveNiMBIntersectorK<N,8>::template intersect_t<SweepCurve1IntersectorK<Curve,8>, Intersect1KEpilog1<8,true> >;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &CurveNiMBIntersectorK<N,8>::template occluded_t <SweepCurve1IntersectorK<Curve,8>, Occluded1KEpilog1<8,true> >;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&CurveNiMBIntersectorK<N,16>::template intersect_t<SweepCurve1IntersectorK<Curve,16>, Intersect1KEpilog1<16,true> >;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &CurveNiMBIntersectorK<N,16>::template occluded_t <SweepCurve1IntersectorK<Curve,16>, Occluded1KEpilog1<16,true> >;
#endif
      return intersectors;
    }
    
    template<template<typename Ty> class Curve, int N>
      static VirtualCurveIntersector::Intersectors OrientedCurveNiIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &CurveNiIntersector1<N>::template intersect_n<OrientedCurve1Intersector1<Curve>, Intersect1Epilog1<true> >;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &CurveNiIntersector1<N>::template occluded_n <OrientedCurve1Intersector1<Curve>, Occluded1Epilog1<true> >;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty)&CurveNiIntersectorK<N,4>::template intersect_n<OrientedCurve1IntersectorK<Curve,4>, Intersect1KEpilog1<4,true> >;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty) &CurveNiIntersectorK<N,4>::template occluded_n <OrientedCurve1IntersectorK<Curve,4>, Occluded1KEpilog1<4,true> >;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&CurveNiIntersectorK<N,8>::template intersect_n<OrientedCurve1IntersectorK<Curve,8>, Intersect1KEpilog1<8,true> >;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &CurveNiIntersectorK<N,8>::template occluded_n <OrientedCurve1IntersectorK<Curve,8>, Occluded1KEpilog1<8,true> >;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&CurveNiIntersectorK<N,16>::template intersect_n<OrientedCurve1IntersectorK<Curve,16>, Intersect1KEpilog1<16,true> >;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &CurveNiIntersectorK<N,16>::template occluded_n <OrientedCurve1IntersectorK<Curve,16>, Occluded1KEpilog1<16,true> >;
#endif
      return intersectors;
    }
    
    template<template<typename Ty> class Curve, int N>
      static VirtualCurveIntersector::Intersectors OrientedCurveNiMBIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &CurveNiMBIntersector1<N>::template intersect_n<OrientedCurve1Intersector1<Curve>, Intersect1Epilog1<true> >;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &CurveNiMBIntersector1<N>::template occluded_n <OrientedCurve1Intersector1<Curve>, Occluded1Epilog1<true> >;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty)&CurveNiMBIntersectorK<N,4>::template intersect_n<OrientedCurve1IntersectorK<Curve,4>, Intersect1KEpilog1<4,true> >;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty) &CurveNiMBIntersectorK<N,4>::template occluded_n <OrientedCurve1IntersectorK<Curve,4>, Occluded1KEpilog1<4,true> >;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&CurveNiMBIntersectorK<N,8>::template intersect_n<OrientedCurve1IntersectorK<Curve,8>, Intersect1KEpilog1<8,true> >;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &CurveNiMBIntersectorK<N,8>::template occluded_n <OrientedCurve1IntersectorK<Curve,8>, Occluded1KEpilog1<8,true> >;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&CurveNiMBIntersectorK<N,16>::template intersect_n<OrientedCurve1IntersectorK<Curve,16>, Intersect1KEpilog1<16,true> >;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &CurveNiMBIntersectorK<N,16>::template occluded_n <OrientedCurve1IntersectorK<Curve,16>, Occluded1KEpilog1<16,true> >;
#endif
      return intersectors;
    }
    
    template<template<typename Ty> class Curve, int N>
      static VirtualCurveIntersector::Intersectors HermiteRibbonNiIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &CurveNiIntersector1<N>::template intersect_h<RibbonCurve1Intersector1<Curve>, Intersect1EpilogMU<VSIZEX,true> >;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &CurveNiIntersector1<N>::template occluded_h <RibbonCurve1Intersector1<Curve>, Occluded1EpilogMU<VSIZEX,true> >;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty)&CurveNiIntersectorK<N,4>::template intersect_h<RibbonCurve1IntersectorK<Curve,4>, Intersect1KEpilogMU<VSIZEX,4,true> >;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty) &CurveNiIntersectorK<N,4>::template occluded_h <RibbonCurve1IntersectorK<Curve,4>, Occluded1KEpilogMU<VSIZEX,4,true> >;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&CurveNiIntersectorK<N,8>::template intersect_h<RibbonCurve1IntersectorK<Curve,8>, Intersect1KEpilogMU<VSIZEX,8,true> >;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &CurveNiIntersectorK<N,8>::template occluded_h <RibbonCurve1IntersectorK<Curve,8>, Occluded1KEpilogMU<VSIZEX,8,true> >;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&CurveNiIntersectorK<N,16>::template intersect_h<RibbonCurve1IntersectorK<Curve,16>, Intersect1KEpilogMU<VSIZEX,16,true> >;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &CurveNiIntersectorK<N,16>::template occluded_h <RibbonCurve1IntersectorK<Curve,16>, Occluded1KEpilogMU<VSIZEX,16,true> >;
#endif
      return intersectors;
    }
    
    template<template<typename Ty> class Curve, int N>
      static VirtualCurveIntersector::Intersectors HermiteRibbonNiMBIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &CurveNiMBIntersector1<N>::template intersect_h<RibbonCurve1Intersector1<Curve>, Intersect1EpilogMU<VSIZEX,true> >;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &CurveNiMBIntersector1<N>::template occluded_h <RibbonCurve1Intersector1<Curve>, Occluded1EpilogMU<VSIZEX,true> >;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty)&CurveNiMBIntersectorK<N,4>::template intersect_h<RibbonCurve1IntersectorK<Curve,4>, Intersect1KEpilogMU<VSIZEX,4,true> >;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty) &CurveNiMBIntersectorK<N,4>::template occluded_h <RibbonCurve1IntersectorK<Curve,4>, Occluded1KEpilogMU<VSIZEX,4,true> >;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&CurveNiMBIntersectorK<N,8>::template intersect_h<RibbonCurve1IntersectorK<Curve,8>, Intersect1KEpilogMU<VSIZEX,8,true> >;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &CurveNiMBIntersectorK<N,8>::template occluded_h <RibbonCurve1IntersectorK<Curve,8>, Occluded1KEpilogMU<VSIZEX,8,true> >;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&CurveNiMBIntersectorK<N,16>::template intersect_h<RibbonCurve1IntersectorK<Curve,16>, Intersect1KEpilogMU<VSIZEX,16,true> >;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &CurveNiMBIntersectorK<N,16>::template occluded_h <RibbonCurve1IntersectorK<Curve,16>, Occluded1KEpilogMU<VSIZEX,16,true> >;
#endif
      return intersectors;
    }
    
    template<template<typename Ty> class Curve, int N>
      static VirtualCurveIntersector::Intersectors HermiteCurveNiIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &CurveNiIntersector1<N>::template intersect_h<SweepCurve1Intersector1<Curve>, Intersect1Epilog1<true> >;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &CurveNiIntersector1<N>::template occluded_h <SweepCurve1Intersector1<Curve>, Occluded1Epilog1<true> >;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty)&CurveNiIntersectorK<N,4>::template intersect_h<SweepCurve1IntersectorK<Curve,4>, Intersect1KEpilog1<4,true> >;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty) &CurveNiIntersectorK<N,4>::template occluded_h <SweepCurve1IntersectorK<Curve,4>, Occluded1KEpilog1<4,true> >;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&CurveNiIntersectorK<N,8>::template intersect_h<SweepCurve1IntersectorK<Curve,8>, Intersect1KEpilog1<8,true> >;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &CurveNiIntersectorK<N,8>::template occluded_h <SweepCurve1IntersectorK<Curve,8>, Occluded1KEpilog1<8,true> >;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&CurveNiIntersectorK<N,16>::template intersect_h<SweepCurve1IntersectorK<Curve,16>, Intersect1KEpilog1<16,true> >;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &CurveNiIntersectorK<N,16>::template occluded_h <SweepCurve1IntersectorK<Curve,16>, Occluded1KEpilog1<16,true> >;
#endif
      return intersectors;
    }
    
    template<template<typename Ty> class Curve, int N>
      static VirtualCurveIntersector::Intersectors HermiteCurveNiMBIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &CurveNiMBIntersector1<N>::template intersect_h<SweepCurve1Intersector1<Curve>, Intersect1Epilog1<true> >;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &CurveNiMBIntersector1<N>::template occluded_h <SweepCurve1Intersector1<Curve>, Occluded1Epilog1<true> >;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty)&CurveNiMBIntersectorK<N,4>::template intersect_h<SweepCurve1IntersectorK<Curve,4>, Intersect1KEpilog1<4,true> >;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty) &CurveNiMBIntersectorK<N,4>::template occluded_h <SweepCurve1IntersectorK<Curve,4>, Occluded1KEpilog1<4,true> >;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&CurveNiMBIntersectorK<N,8>::template intersect_h<SweepCurve1IntersectorK<Curve,8>, Intersect1KEpilog1<8,true> >;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &CurveNiMBIntersectorK<N,8>::template occluded_h <SweepCurve1IntersectorK<Curve,8>, Occluded1KEpilog1<8,true> >;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&CurveNiMBIntersectorK<N,16>::template intersect_h<SweepCurve1IntersectorK<Curve,16>, Intersect1KEpilog1<16,true> >;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &CurveNiMBIntersectorK<N,16>::template occluded_h <SweepCurve1IntersectorK<Curve,16>, Occluded1KEpilog1<16,true> >;
#endif
      return intersectors;
    }
    
    template<template<typename Ty> class Curve, int N>
      static VirtualCurveIntersector::Intersectors HermiteOrientedCurveNiIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &CurveNiIntersector1<N>::template intersect_hn<OrientedCurve1Intersector1<Curve>, Intersect1Epilog1<true> >;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &CurveNiIntersector1<N>::template occluded_hn <OrientedCurve1Intersector1<Curve>, Occluded1Epilog1<true> >;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty)&CurveNiIntersectorK<N,4>::template intersect_hn<OrientedCurve1IntersectorK<Curve,4>, Intersect1KEpilog1<4,true> >;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty) &CurveNiIntersectorK<N,4>::template occluded_hn <OrientedCurve1IntersectorK<Curve,4>, Occluded1KEpilog1<4,true> >;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&CurveNiIntersectorK<N,8>::template intersect_hn<OrientedCurve1IntersectorK<Curve,8>, Intersect1KEpilog1<8,true> >;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &CurveNiIntersectorK<N,8>::template occluded_hn <OrientedCurve1IntersectorK<Curve,8>, Occluded1KEpilog1<8,true> >;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&CurveNiIntersectorK<N,16>::template intersect_hn<OrientedCurve1IntersectorK<Curve,16>, Intersect1KEpilog1<16,true> >;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &CurveNiIntersectorK<N,16>::template occluded_hn <OrientedCurve1IntersectorK<Curve,16>, Occluded1KEpilog1<16,true> >;
#endif
      return intersectors;
    }
    
    template<template<typename Ty> class Curve, int N>
      static VirtualCurveIntersector::Intersectors HermiteOrientedCurveNiMBIntersectors()
    {
      VirtualCurveIntersector::Intersectors intersectors;
      intersectors.intersect1 = (VirtualCurveIntersector::Intersect1Ty) &CurveNiMBIntersector1<N>::template intersect_hn<OrientedCurve1Intersector1<Curve>, Intersect1Epilog1<true> >;
      intersectors.occluded1  = (VirtualCurveIntersector::Occluded1Ty)  &CurveNiMBIntersector1<N>::template occluded_hn <OrientedCurve1Intersector1<Curve>, Occluded1Epilog1<true> >;
      intersectors.intersect4 = (VirtualCurveIntersector::Intersect4Ty)&CurveNiMBIntersectorK<N,4>::template intersect_hn<OrientedCurve1IntersectorK<Curve,4>, Intersect1KEpilog1<4,true> >;
      intersectors.occluded4  = (VirtualCurveIntersector::Occluded4Ty) &CurveNiMBIntersectorK<N,4>::template occluded_hn <OrientedCurve1IntersectorK<Curve,4>, Occluded1KEpilog1<4,true> >;
#if defined(__AVX__)
      intersectors.intersect8 = (VirtualCurveIntersector::Intersect8Ty)&CurveNiMBIntersectorK<N,8>::template intersect_hn<OrientedCurve1IntersectorK<Curve,8>, Intersect1KEpilog1<8,true> >;
      intersectors.occluded8  = (VirtualCurveIntersector::Occluded8Ty) &CurveNiMBIntersectorK<N,8>::template occluded_hn <OrientedCurve1IntersectorK<Curve,8>, Occluded1KEpilog1<8,true> >;
#endif
#if defined(__AVX512F__)
      intersectors.intersect16 = (VirtualCurveIntersector::Intersect16Ty)&CurveNiMBIntersectorK<N,16>::template intersect_hn<OrientedCurve1IntersectorK<Curve,16>, Intersect1KEpilog1<16,true> >;
      intersectors.occluded16  = (VirtualCurveIntersector::Occluded16Ty) &CurveNiMBIntersectorK<N,16>::template occluded_hn <OrientedCurve1IntersectorK<Curve,16>, Occluded1KEpilog1<16,true> >;
#endif
      return intersectors;
    }
  }
}
