// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../common/geometry.h"
#include "../common/ray.h"
#include "../common/hit.h"
#include "../common/context.h"

namespace embree
{
  namespace isa
  {
    __forceinline bool runIntersectionFilter1Helper(RTCFilterFunctionNArguments* args, const Geometry* const geometry, RayQueryContext* context)
    {
      if (geometry->intersectionFilterN)
      {
        geometry->intersectionFilterN(args);

        if (args->valid[0] == 0)
          return false;
      }

      if (context->getFilter())
      {
        if (context->enforceArgumentFilterFunction() || geometry->hasArgumentFilterFunctions())
          context->getFilter()(args);

        if (args->valid[0] == 0)
          return false;
      }
      
      copyHitToRay(*(RayHit*)args->ray,*(Hit*)args->hit);
      return true;
    }
    
    __forceinline bool runIntersectionFilter1(const Geometry* const geometry, RayHit& ray, RayQueryContext* context, Hit& hit)
    {
      RTCFilterFunctionNArguments args;
      int mask = -1;
      args.valid = &mask;
      args.geometryUserPtr = geometry->userPtr;
      args.context = context->user;
      args.ray = (RTCRayN*)&ray;
      args.hit = (RTCHitN*)&hit;
      args.N = 1;
      return runIntersectionFilter1Helper(&args,geometry,context);
    }

    __forceinline bool runOcclusionFilter1Helper(RTCFilterFunctionNArguments* args, const Geometry* const geometry, RayQueryContext* context)
    {
      if (geometry->occlusionFilterN)
      {
        geometry->occlusionFilterN(args);

        if (args->valid[0] == 0)
          return false;
      }

      if (context->getFilter())
      {
        if (context->enforceArgumentFilterFunction() || geometry->hasArgumentFilterFunctions())
          context->getFilter()(args);

        if (args->valid[0] == 0)
          return false;
      }

      return true;
    }

    __forceinline bool runOcclusionFilter1(const Geometry* const geometry, Ray& ray, RayQueryContext* context, Hit& hit)
    {
      RTCFilterFunctionNArguments args;
      int mask = -1;
      args.valid = &mask;
      args.geometryUserPtr = geometry->userPtr;
      args.context = context->user;
      args.ray = (RTCRayN*)&ray;
      args.hit = (RTCHitN*)&hit;
      args.N = 1;
      return runOcclusionFilter1Helper(&args,geometry,context);
    }

    template<int K>
      __forceinline vbool<K> runIntersectionFilterHelper(RTCFilterFunctionNArguments* args, const Geometry* const geometry, RayQueryContext* context)
    {
      vint<K>* mask = (vint<K>*) args->valid;
      if (geometry->intersectionFilterN)
        geometry->intersectionFilterN(args);
      
      vbool<K> valid_o = *mask != vint<K>(zero);
      if (none(valid_o)) return valid_o;

      if (context->getFilter()) {
        if (context->enforceArgumentFilterFunction() || geometry->hasArgumentFilterFunctions())
          context->getFilter()(args);
      }

      valid_o = *mask != vint<K>(zero);
      if (none(valid_o)) return valid_o;
      
      copyHitToRay(valid_o,*(RayHitK<K>*)args->ray,*(HitK<K>*)args->hit);
      return valid_o;
    }
    
    template<int K>
    __forceinline vbool<K> runIntersectionFilter(const vbool<K>& valid, const Geometry* const geometry, RayHitK<K>& ray, RayQueryContext* context, HitK<K>& hit)
    {
      RTCFilterFunctionNArguments args;
      vint<K> mask = valid.mask32();
      args.valid = (int*)&mask;
      args.geometryUserPtr = geometry->userPtr;
      args.context = context->user;
      args.ray = (RTCRayN*)&ray;
      args.hit = (RTCHitN*)&hit;
      args.N = K;
      return runIntersectionFilterHelper<K>(&args,geometry,context);
    }

    template<int K>
      __forceinline vbool<K> runOcclusionFilterHelper(RTCFilterFunctionNArguments* args, const Geometry* const geometry, RayQueryContext* context)
    {
      vint<K>* mask = (vint<K>*) args->valid;
      if (geometry->occlusionFilterN)
        geometry->occlusionFilterN(args);
      
      vbool<K> valid_o = *mask != vint<K>(zero);
      if (none(valid_o)) return valid_o;

      if (context->getFilter()) {
        if (context->enforceArgumentFilterFunction() || geometry->hasArgumentFilterFunctions())
          context->getFilter()(args);
      }
      valid_o = *mask != vint<K>(zero);

      RayK<K>* ray = (RayK<K>*) args->ray;
      ray->tfar = select(valid_o, vfloat<K>(neg_inf), ray->tfar);
      return valid_o;
    }

    template<int K>
      __forceinline vbool<K> runOcclusionFilter(const vbool<K>& valid, const Geometry* const geometry, RayK<K>& ray, RayQueryContext* context, HitK<K>& hit)
    {
      RTCFilterFunctionNArguments args;
      vint<K> mask = valid.mask32();
      args.valid = (int*)&mask;
      args.geometryUserPtr = geometry->userPtr;
      args.context = context->user;
      args.ray = (RTCRayN*)&ray;
      args.hit = (RTCHitN*)&hit;
      args.N = K;
      return runOcclusionFilterHelper<K>(&args,geometry,context);
    }
  }
}
