// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "default.h"
#include "rtcore.h"
#include "point_query.h"

namespace embree
{
  class Scene;

  struct IntersectContext
  {
  public:
    __forceinline IntersectContext(Scene* scene, RTCIntersectContext* user_context)
      : scene(scene), user(user_context) {}

    __forceinline bool hasContextFilter() const {
      return user->filter != nullptr;
    }

    __forceinline bool isCoherent() const {
      return embree::isCoherent(user->flags);
    }

    __forceinline bool isIncoherent() const {
      return embree::isIncoherent(user->flags);
    }
    
  public:
    Scene* scene;
    RTCIntersectContext* user;
  };

  template<int M, typename Geometry>
      __forceinline Vec4vf<M> enlargeRadiusToMinWidth(const IntersectContext* context, const Geometry* geom, const Vec3vf<M>& ray_org, const Vec4vf<M>& v)
    {
#if RTC_MIN_WIDTH
      const vfloat<M> d = length(Vec3vf<M>(v) - ray_org);
      const vfloat<M> r = clamp(context->user->minWidthDistanceFactor*d, v.w, geom->maxRadiusScale*v.w);
      return Vec4vf<M>(v.x,v.y,v.z,r);
#else
      return v;
#endif
    }

    template<typename Geometry>
    __forceinline Vec3ff enlargeRadiusToMinWidth(const IntersectContext* context, const Geometry* geom, const Vec3fa& ray_org, const Vec3ff& v)
  {
#if RTC_MIN_WIDTH
    const float d = length(Vec3fa(v) - ray_org);
    const float r = clamp(context->user->minWidthDistanceFactor*d, v.w, geom->maxRadiusScale*v.w);
    return Vec3ff(v.x,v.y,v.z,r);
#else
    return v;
#endif
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
      if (query_type == POINT_QUERY_TYPE_AABB) {
        assert(similarityScale == 0.f);
        updateAABB();
      }
      if (userContext->instStackSize == 0) {
        assert(similarityScale == 1.f);
      }
    }

  public:
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

    PointQuery* query_ws; // the original world space point query 
    PointQueryType query_type;
    PointQueryFunction func;
    RTCPointQueryContext* userContext;
    const float similarityScale;

    void* userPtr;

    unsigned int primID;
    unsigned int geomID;

    Vec3fa query_radius;  // used if the query is converted to an AABB internally
  };
}

