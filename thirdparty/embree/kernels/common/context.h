// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "default.h"
#include "rtcore.h"
#include "point_query.h"

namespace embree
{
  class Scene;

  struct RayQueryContext
  {
  public:

    __forceinline RayQueryContext(Scene* scene, RTCRayQueryContext* user_context, RTCIntersectArguments* args)
      : scene(scene), user(user_context), args(args) {}

    __forceinline RayQueryContext(Scene* scene, RTCRayQueryContext* user_context, RTCOccludedArguments* args)
      : scene(scene), user(user_context), args((RTCIntersectArguments*)args) {}

    __forceinline bool hasContextFilter() const {
      return args->filter != nullptr;
    }

    RTCFilterFunctionN getFilter() const {
      return args->filter;
    }

    RTCIntersectFunctionN getIntersectFunction() const {
      return args->intersect;
    }
    
    RTCOccludedFunctionN getOccludedFunction() const {
      return (RTCOccludedFunctionN) args->intersect;
    }

    __forceinline bool isCoherent() const {
      return embree::isCoherent(args->flags);
    }

    __forceinline bool isIncoherent() const {
      return embree::isIncoherent(args->flags);
    }

    __forceinline bool enforceArgumentFilterFunction() const {
      return args->flags & RTC_RAY_QUERY_FLAG_INVOKE_ARGUMENT_FILTER;
    }

#if RTC_MIN_WIDTH
    __forceinline float getMinWidthDistanceFactor() const {
      return args->minWidthDistanceFactor;
    }
#endif

  public:
    Scene* scene = nullptr;
    RTCRayQueryContext* user = nullptr;
    RTCIntersectArguments* args = nullptr;
  };

  template<int M, typename Geometry>
      __forceinline Vec4vf<M> enlargeRadiusToMinWidth(const RayQueryContext* context, const Geometry* geom, const Vec3vf<M>& ray_org, const Vec4vf<M>& v)
    {
#if RTC_MIN_WIDTH
      const vfloat<M> d = length(Vec3vf<M>(v) - ray_org);
      const vfloat<M> r = clamp(context->getMinWidthDistanceFactor()*d, v.w, geom->maxRadiusScale*v.w);
      return Vec4vf<M>(v.x,v.y,v.z,r);
#else
      return v;
#endif
    }

    template<typename Geometry>
    __forceinline Vec3ff enlargeRadiusToMinWidth(const RayQueryContext* context, const Geometry* geom, const Vec3fa& ray_org, const Vec3ff& v)
  {
#if RTC_MIN_WIDTH
    const float d = length(Vec3fa(v) - ray_org);
    const float r = clamp(context->getMinWidthDistanceFactor()*d, v.w, geom->maxRadiusScale*v.w);
    return Vec3ff(v.x,v.y,v.z,r);
#else
    return v;
#endif
  }

  template<typename Geometry>
    __forceinline Vec3ff enlargeRadiusToMinWidth(const RayQueryContext* context, const Geometry* geom, const Vec3fa& ray_org, const Vec4f& v) {
    return enlargeRadiusToMinWidth(context,geom,ray_org,Vec3ff(v.x,v.y,v.z,v.w));
  }
  
  enum PointQueryType
  {
    POINT_QUERY_TYPE_UNDEFINED = 0,
    POINT_QUERY_TYPE_SPHERE = 1,
    POINT_QUERY_TYPE_AABB = 2,
  };

  typedef bool (*PointQueryFunction)(struct RTCPointQueryFunctionArguments* args);

  struct PointQueryContext
  {
  public:
    __forceinline PointQueryContext(Scene* scene, 
                                    PointQuery* query_ws, 
                                    PointQueryType query_type,
                                    PointQueryFunction func, 
                                    RTCPointQueryContext* userContext,
                                    float similarityScale,
                                    void* userPtr)
      : scene(scene)
      , tstate(nullptr)
      , query_ws(query_ws)
      , query_type(query_type)
      , func(func)
      , userContext(userContext)
      , similarityScale(similarityScale)
      , userPtr(userPtr) 
      , primID(RTC_INVALID_GEOMETRY_ID)
      , geomID(RTC_INVALID_GEOMETRY_ID)
      , query_radius(query_ws->radius)
    { 
      update();
    }

  public:
    __forceinline void update()
    {
      if (query_type == POINT_QUERY_TYPE_AABB) {
        assert(similarityScale == 0.f);
        updateAABB();
      }
      else{
        query_radius = Vec3fa(query_ws->radius * similarityScale);
      }
      if (userContext->instStackSize == 0) {
        assert(similarityScale == 1.f);
      }
    }

    __forceinline void updateAABB() 
    {
      if (likely(query_ws->radius == (float)inf || userContext->instStackSize == 0)) {
        query_radius = Vec3fa(query_ws->radius);
        return;
      }

      const AffineSpace3fa m = AffineSpace3fa_load_unaligned((AffineSpace3fa*)userContext->world2inst[userContext->instStackSize-1]);
      BBox3fa bbox(Vec3fa(-query_ws->radius), Vec3fa(query_ws->radius));
      bbox = xfmBounds(m, bbox);
      query_radius = 0.5f * (bbox.upper - bbox.lower);
    }

public:
    Scene* scene;
    void* tstate;

    PointQuery* query_ws; // the original world space point query 
    PointQueryType query_type;
    PointQueryFunction func;
    RTCPointQueryContext* userContext;
    float similarityScale;

    void* userPtr;

    unsigned int primID;
    unsigned int geomID;

    Vec3fa query_radius;  // used if the query is converted to an AABB internally
  };
}

