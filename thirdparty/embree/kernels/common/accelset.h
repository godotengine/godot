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
  
  typedef void (*ReportIntersectionFunc) (IntersectFunctionNArguments* args, const RTCFilterFunctionNArguments* filter_args);
  typedef void (*ReportOcclusionFunc) (OccludedFunctionNArguments* args, const RTCFilterFunctionNArguments* filter_args);
  
  struct IntersectFunctionNArguments : public RTCIntersectFunctionNArguments
  {
    IntersectContext* internal_context;
    Geometry* geometry;
    ReportIntersectionFunc report;
  };

  struct OccludedFunctionNArguments : public RTCOccludedFunctionNArguments
  {
    IntersectContext* internal_context;
    Geometry* geometry;
    ReportOcclusionFunc report;
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
      __forceinline void intersect (RayHit& ray, unsigned int geomID, unsigned int primID, IntersectContext* context, ReportIntersectionFunc report) 
      {
        assert(primID < size());
        assert(intersectorN.intersect);
        
        int mask = -1;
        IntersectFunctionNArguments args;
        args.valid = &mask;
        args.geometryUserPtr = userPtr;
        args.context = context->user;
        args.rayhit = (RTCRayHitN*)&ray;
        args.N = 1;
        args.geomID = geomID;
        args.primID = primID;
        args.internal_context = context;
        args.geometry = this;
        args.report = report;
        
        intersectorN.intersect(&args);
      }

      /*! Tests if single ray is occluded by the scene. */
      __forceinline void occluded (Ray& ray, unsigned int geomID, unsigned int primID, IntersectContext* context, ReportOcclusionFunc report)
      {
        assert(primID < size());
        assert(intersectorN.occluded);
        
        int mask = -1;
        OccludedFunctionNArguments args;
        args.valid = &mask;
        args.geometryUserPtr = userPtr;
        args.context = context->user;
        args.ray = (RTCRayN*)&ray;
        args.N = 1;
        args.geomID = geomID;
        args.primID = primID;
        args.internal_context = context;
        args.geometry = this;
        args.report = report;
        
        intersectorN.occluded(&args);
      }
   
      /*! Intersects a packet of K rays with the scene. */
      template<int K>
        __forceinline void intersect (const vbool<K>& valid, RayHitK<K>& ray, unsigned int geomID, unsigned int primID, IntersectContext* context, ReportIntersectionFunc report) 
      {
        assert(primID < size());
        assert(intersectorN.intersect);
        
        vint<K> mask = valid.mask32();
        IntersectFunctionNArguments args;
        args.valid = (int*)&mask;
        args.geometryUserPtr = userPtr;
        args.context = context->user;
        args.rayhit = (RTCRayHitN*)&ray;
        args.N = K;
        args.geomID = geomID;
        args.primID = primID;
        args.internal_context = context;
        args.geometry = this;
        args.report = report;
         
        intersectorN.intersect(&args);
      }

      /*! Tests if a packet of K rays is occluded by the scene. */
      template<int K>
        __forceinline void occluded (const vbool<K>& valid, RayK<K>& ray, unsigned int geomID, unsigned int primID, IntersectContext* context, ReportOcclusionFunc report)
      {
        assert(primID < size());
        assert(intersectorN.occluded);
        
        vint<K> mask = valid.mask32();
        OccludedFunctionNArguments args;
        args.valid = (int*)&mask;
        args.geometryUserPtr = userPtr;
        args.context = context->user;
        args.ray = (RTCRayN*)&ray;
        args.N = K;
        args.geomID = geomID;
        args.primID = primID;
        args.internal_context = context;
        args.geometry = this;
        args.report = report;
        
        intersectorN.occluded(&args);
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
