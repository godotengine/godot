// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "default.h"
#include "builder.h"
#include "geometry.h"
#include "ray.h"
#include "hit.h"

namespace embree
{
  struct IntersectFunctionNArguments;
  struct OccludedFunctionNArguments;
  
  struct IntersectFunctionNArguments : public RTCIntersectFunctionNArguments
  {
    Geometry* geometry;
    RTCScene forward_scene;
    RTCIntersectArguments* args;
  };

  struct OccludedFunctionNArguments : public RTCOccludedFunctionNArguments
  {
    Geometry* geometry;
    RTCScene forward_scene;
    RTCIntersectArguments* args;
  };

  /*! Base class for set of acceleration structures. */
  class AccelSet : public Geometry
  {
  public:
    typedef RTCIntersectFunctionN IntersectFuncN;  
    typedef RTCOccludedFunctionN OccludedFuncN;
    typedef void (*ErrorFunc) ();

      struct IntersectorN
      {
        IntersectorN (ErrorFunc error = nullptr) ;
        IntersectorN (IntersectFuncN intersect, OccludedFuncN occluded, const char* name);
        
        operator bool() const { return name; }
        
      public:
        static const char* type;
        IntersectFuncN intersect;
        OccludedFuncN occluded; 
        const char* name;
      };
      
    public:
      
      /*! construction */
      AccelSet (Device* device, Geometry::GType gtype, size_t items, size_t numTimeSteps);
      
      /*! makes the acceleration structure immutable */
      virtual void immutable () {}
      
      /*! build accel */
      virtual void build () = 0;

      /*! check if the i'th primitive is valid between the specified time range */
      __forceinline bool valid(size_t i, const range<size_t>& itime_range) const
      {
        for (size_t itime = itime_range.begin(); itime <= itime_range.end(); itime++)
          if (!isvalid_non_empty(bounds(i,itime))) return false;
        
        return true;
      }

      /*! Calculates the bounds of an item */
      __forceinline BBox3fa bounds(size_t i, size_t itime = 0) const
      {
        BBox3fa box;
        assert(i < size());
        RTCBoundsFunctionArguments args;
        args.geometryUserPtr = userPtr;
        args.primID = (unsigned int)i;
        args.timeStep = (unsigned int)itime;
        args.bounds_o = (RTCBounds*)&box;
        boundsFunc(&args);
        return box;
      }

      /*! calculates the linear bounds of the i'th item at the itime'th time segment */
      __forceinline LBBox3fa linearBounds(size_t i, size_t itime) const
      {
        BBox3fa box[2];
        assert(i < size());
        RTCBoundsFunctionArguments args;
        args.geometryUserPtr = userPtr;
        args.primID = (unsigned int)i;
        args.timeStep = (unsigned int)(itime+0);
        args.bounds_o = (RTCBounds*)&box[0];
        boundsFunc(&args);
        args.timeStep = (unsigned int)(itime+1);
        args.bounds_o = (RTCBounds*)&box[1];
        boundsFunc(&args);
        return LBBox3fa(box[0],box[1]);
      }

      /*! calculates the build bounds of the i'th item, if it's valid */
      __forceinline bool buildBounds(size_t i, BBox3fa* bbox = nullptr) const
      {
        const BBox3fa b = bounds(i);
        if (bbox) *bbox = b;
        return isvalid_non_empty(b);
      }

      /*! calculates the build bounds of the i'th item at the itime'th time segment, if it's valid */
      __forceinline bool buildBounds(size_t i, size_t itime, BBox3fa& bbox) const
      {
        const LBBox3fa bounds = linearBounds(i,itime);
        bbox = bounds.bounds0; // use bounding box of first timestep to build BVH
        return isvalid_non_empty(bounds);
      }

      /*! calculates the linear bounds of the i'th primitive for the specified time range */
      __forceinline LBBox3fa linearBounds(size_t primID, const BBox1f& dt) const {
        return LBBox3fa([&] (size_t itime) { return bounds(primID, itime); }, dt, time_range, fnumTimeSegments);
      }
      
      /*! calculates the linear bounds of the i'th primitive for the specified time range */
      __forceinline bool linearBounds(size_t i, const BBox1f& time_range, LBBox3fa& bbox) const  {
        if (!valid(i, timeSegmentRange(time_range))) return false;
        bbox = linearBounds(i, time_range);
        return true;
      }

      /* gets version info of topology */
      unsigned int getTopologyVersion() const {
        return numPrimitives;
      }
    
      /* returns true if topology changed */
      bool topologyChanged(unsigned int otherVersion) const {
        return numPrimitives != otherVersion;
      }

  public:

      /*! Intersects a single ray with the scene. */
      __forceinline bool intersect (RayHit& ray, unsigned int geomID, unsigned int primID, RayQueryContext* context) 
      {
        assert(primID < size());
        
        int mask = -1;
        IntersectFunctionNArguments args;
        args.valid = &mask;
        args.geometryUserPtr = userPtr;
        args.context = context->user;
        args.rayhit = (RTCRayHitN*)&ray;
        args.N = 1;
        args.geomID = geomID;
        args.primID = primID;
        args.geometry = this;
        args.forward_scene = nullptr;
        args.args = context->args;

        IntersectFuncN intersectFunc = nullptr;
        intersectFunc = intersectorN.intersect;
        
        if (context->getIntersectFunction())
          intersectFunc = context->getIntersectFunction();

        assert(intersectFunc);
        intersectFunc(&args);

        return mask != 0;
      }

      /*! Tests if single ray is occluded by the scene. */
      __forceinline bool occluded (Ray& ray, unsigned int geomID, unsigned int primID, RayQueryContext* context)
      {
        assert(primID < size());

        int mask = -1;
        OccludedFunctionNArguments args;
        args.valid = &mask;
        args.geometryUserPtr = userPtr;
        args.context = context->user;
        args.ray = (RTCRayN*)&ray;
        args.N = 1;
        args.geomID = geomID;
        args.primID = primID;
        args.geometry = this;
        args.forward_scene = nullptr;
        args.args = context->args;

        OccludedFuncN occludedFunc = nullptr;
        occludedFunc = intersectorN.occluded;

        if (context->getOccludedFunction())
          occludedFunc = context->getOccludedFunction();

        assert(occludedFunc);
        occludedFunc(&args);

        return mask != 0;
      }

      /*! Intersects a single ray with the scene. */
    __forceinline bool intersect (RayHit& ray, unsigned int geomID, unsigned int primID, RayQueryContext* context, RTCScene& forward_scene) 
    {
        assert(primID < size());
        
        int mask = -1;
        IntersectFunctionNArguments args;
        args.valid = &mask;
        args.geometryUserPtr = userPtr;
        args.context = context->user;
        args.rayhit = (RTCRayHitN*)&ray;
        args.N = 1;
        args.geomID = geomID;
        args.primID = primID;
        args.geometry = this;
        args.forward_scene = nullptr;
        args.args = nullptr;

        typedef void (*RTCIntersectFunctionSYCL)(const void* args);
        RTCIntersectFunctionSYCL intersectFunc = nullptr;
        
#if EMBREE_SYCL_GEOMETRY_CALLBACK
        if (context->args->feature_mask & RTC_FEATURE_FLAG_USER_GEOMETRY_CALLBACK_IN_GEOMETRY)
          intersectFunc = (RTCIntersectFunctionSYCL) intersectorN.intersect;
#endif
        
        if (context->args->feature_mask & RTC_FEATURE_FLAG_USER_GEOMETRY_CALLBACK_IN_ARGUMENTS)
          if (context->getIntersectFunction())
            intersectFunc = (RTCIntersectFunctionSYCL) context->getIntersectFunction();

        if (intersectFunc)
          intersectFunc(&args);
        
        forward_scene = args.forward_scene;
        return mask != 0;
      }

      /*! Tests if single ray is occluded by the scene. */
    __forceinline bool occluded (Ray& ray, unsigned int geomID, unsigned int primID, RayQueryContext* context, RTCScene& forward_scene)
      {
        assert(primID < size());

        int mask = -1;
        OccludedFunctionNArguments args;
        args.valid = &mask;
        args.geometryUserPtr = userPtr;
        args.context = context->user;
        args.ray = (RTCRayN*)&ray;
        args.N = 1;
        args.geomID = geomID;
        args.primID = primID;
        args.geometry = this;
        args.forward_scene = nullptr;
        args.args = nullptr;

        typedef void (*RTCOccludedFunctionSYCL)(const void* args);
        RTCOccludedFunctionSYCL occludedFunc = nullptr;

#if EMBREE_SYCL_GEOMETRY_CALLBACK
        if (context->args->feature_mask & RTC_FEATURE_FLAG_USER_GEOMETRY_CALLBACK_IN_GEOMETRY)
          occludedFunc = (RTCOccludedFunctionSYCL) intersectorN.occluded;
#endif
        
        if (context->args->feature_mask & RTC_FEATURE_FLAG_USER_GEOMETRY_CALLBACK_IN_ARGUMENTS)
          if (context->getOccludedFunction())
            occludedFunc = (RTCOccludedFunctionSYCL) context->getOccludedFunction();

        if (occludedFunc)
          occludedFunc(&args);
        
        forward_scene = args.forward_scene;
        return mask != 0;
      }

      /*! Intersects a packet of K rays with the scene. */
      template<int K>
        __forceinline void intersect (const vbool<K>& valid, RayHitK<K>& ray, unsigned int geomID, unsigned int primID, RayQueryContext* context) 
      {
        assert(primID < size());
        
        vint<K> mask = valid.mask32();
        IntersectFunctionNArguments args;
        args.valid = (int*)&mask;
        args.geometryUserPtr = userPtr;
        args.context = context->user;
        args.rayhit = (RTCRayHitN*)&ray;
        args.N = K;
        args.geomID = geomID;
        args.primID = primID;
        args.geometry = this;
        args.forward_scene = nullptr;
        args.args = context->args;

        IntersectFuncN intersectFunc = nullptr;
        intersectFunc = intersectorN.intersect;
        
        if (context->getIntersectFunction())
          intersectFunc = context->getIntersectFunction();

        assert(intersectFunc);
        intersectFunc(&args);
      }

      /*! Tests if a packet of K rays is occluded by the scene. */
      template<int K>
        __forceinline void occluded (const vbool<K>& valid, RayK<K>& ray, unsigned int geomID, unsigned int primID, RayQueryContext* context)
      {
        assert(primID < size());
        
        vint<K> mask = valid.mask32();
        OccludedFunctionNArguments args;
        args.valid = (int*)&mask;
        args.geometryUserPtr = userPtr;
        args.context = context->user;
        args.ray = (RTCRayN*)&ray;
        args.N = K;
        args.geomID = geomID;
        args.primID = primID;
        args.geometry = this;
        args.forward_scene = nullptr;
        args.args = context->args;

        OccludedFuncN occludedFunc = nullptr;
        occludedFunc = intersectorN.occluded;
        
        if (context->getOccludedFunction())
          occludedFunc = context->getOccludedFunction();

        assert(occludedFunc);
        occludedFunc(&args);
      }

    public:
      RTCBoundsFunction boundsFunc;
      IntersectorN intersectorN;
  };
  
#define DEFINE_SET_INTERSECTORN(symbol,intersector)                     \
  AccelSet::IntersectorN symbol() {                                     \
    return AccelSet::IntersectorN(intersector::intersect, \
                                  intersector::occluded, \
                                  TOSTRING(isa) "::" TOSTRING(symbol)); \
  }
}
