// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "geometry.h"
#include "scene.h"

namespace embree
{
  const char* Geometry::gtype_names[Geometry::GTY_END] =
  {
    "flat_linear_curve",
    "round_linear_curve",
    "oriented_linear_curve",
    "",
    "flat_bezier_curve",
    "round_bezier_curve",
    "oriented_bezier_curve",
    "",
    "flat_bspline_curve",
    "round_bspline_curve",
    "oriented_bspline_curve",
    "",
    "flat_hermite_curve",
    "round_hermite_curve",
    "oriented_hermite_curve",
    "",
    "flat_catmull_rom_curve",
    "round_catmull_rom_curve",
    "oriented_catmull_rom_curve",
    "",    
    "triangles",
    "quads",
    "grid",
    "subdivs",
    "",
    "sphere",
    "disc",
    "oriented_disc",
    "",
    "usergeom",
    "instance_cheap",
    "instance_expensive",
  };
     
  Geometry::Geometry (Device* device, GType gtype, unsigned int numPrimitives, unsigned int numTimeSteps) 
    : device(device), userPtr(nullptr),
      numPrimitives(numPrimitives), numTimeSteps(unsigned(numTimeSteps)), fnumTimeSegments(float(numTimeSteps-1)), time_range(0.0f,1.0f),
      mask(1),
      gtype(gtype),
      gsubtype(GTY_SUBTYPE_DEFAULT),
      quality(RTC_BUILD_QUALITY_MEDIUM),
      state((unsigned)State::MODIFIED),
      enabled(true),
      argumentFilterEnabled(false),
      intersectionFilterN(nullptr), occlusionFilterN(nullptr), pointQueryFunc(nullptr)
  {
    device->refInc();
  }

  Geometry::~Geometry()
  {
    device->refDec();
  }

  void Geometry::setNumPrimitives(unsigned int numPrimitives_in)
  {      
    if (numPrimitives_in == numPrimitives) return;
    
    numPrimitives = numPrimitives_in;
    
    Geometry::update();
  }

  void Geometry::setNumTimeSteps (unsigned int numTimeSteps_in)
  {
    if (numTimeSteps_in == numTimeSteps) {
      return;
    }
    
    numTimeSteps = numTimeSteps_in;
    fnumTimeSegments = float(numTimeSteps_in-1);
    
    Geometry::update();
  }

  void Geometry::setTimeRange (const BBox1f range)
  {
    time_range = range;
    Geometry::update();
  }
  
  BBox1f Geometry::getTimeRange () const
  {
    return time_range;
  }

  void Geometry::update()
  {
    ++modCounter_; // FIXME: required?
    state = (unsigned)State::MODIFIED;
  }
  
  void Geometry::commit() 
  {
    ++modCounter_;
    state = (unsigned)State::COMMITTED;
  }

  void Geometry::preCommit()
  {
    if (State::MODIFIED == (State)state)
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"geometry not committed");
  }

  void Geometry::postCommit()
  {
  }

  void Geometry::enable () 
  {
    if (isEnabled()) 
      return;

    enabled = true;
    ++modCounter_;
  }

  void Geometry::disable () 
  {
    if (isDisabled()) 
      return;
    
    enabled = false;
    ++modCounter_;
  }

  void Geometry::setUserData (void* ptr)
  {
    userPtr = ptr;
  }
  
  void Geometry::setIntersectionFilterFunctionN (RTCFilterFunctionN filter) 
  {
    if (!(getTypeMask() & (MTY_TRIANGLE_MESH | MTY_QUAD_MESH | MTY_CURVES | MTY_SUBDIV_MESH | MTY_USER_GEOMETRY | MTY_GRID_MESH)))
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"filter functions not supported for this geometry"); 

    intersectionFilterN = filter;
  }

  void Geometry::setOcclusionFilterFunctionN (RTCFilterFunctionN filter) 
  {
    if (!(getTypeMask() & (MTY_TRIANGLE_MESH | MTY_QUAD_MESH | MTY_CURVES | MTY_SUBDIV_MESH | MTY_USER_GEOMETRY | MTY_GRID_MESH)))
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"filter functions not supported for this geometry"); 

    occlusionFilterN = filter;
  }
  
  void Geometry::setPointQueryFunction (RTCPointQueryFunction func) 
  {
    pointQueryFunc = func;
  }

  void Geometry::interpolateN(const RTCInterpolateNArguments* const args)
  {
    const void* valid_i = args->valid;
    const unsigned* primIDs = args->primIDs;
    const float* u = args->u;
    const float* v = args->v;
    unsigned int N = args->N;
    RTCBufferType bufferType = args->bufferType;
    unsigned int bufferSlot = args->bufferSlot;
    float* P = args->P;
    float* dPdu = args->dPdu;
    float* dPdv = args->dPdv;
    float* ddPdudu = args->ddPdudu;
    float* ddPdvdv = args->ddPdvdv;
    float* ddPdudv = args->ddPdudv;
    unsigned int valueCount = args->valueCount;

    if (valueCount > 256) throw_RTCError(RTC_ERROR_INVALID_OPERATION,"maximally 256 floating point values can be interpolated per vertex");
    const int* valid = (const int*) valid_i;
 
    __aligned(64) float P_tmp[256];
    __aligned(64) float dPdu_tmp[256];
    __aligned(64) float dPdv_tmp[256];
    __aligned(64) float ddPdudu_tmp[256];
    __aligned(64) float ddPdvdv_tmp[256];
    __aligned(64) float ddPdudv_tmp[256];

    float* Pt = P ? P_tmp : nullptr;
    float* dPdut = nullptr, *dPdvt = nullptr;
    if (dPdu) { dPdut = dPdu_tmp; dPdvt = dPdv_tmp; }
    float* ddPdudut = nullptr, *ddPdvdvt = nullptr, *ddPdudvt = nullptr;
    if (ddPdudu) { ddPdudut = ddPdudu_tmp; ddPdvdvt = ddPdvdv_tmp; ddPdudvt = ddPdudv_tmp; }
    
    for (unsigned int i=0; i<N; i++)
    {
      if (valid && !valid[i]) continue;

      RTCInterpolateArguments iargs;
      iargs.primID = primIDs[i];
      iargs.u = u[i];
      iargs.v = v[i];
      iargs.bufferType = bufferType;
      iargs.bufferSlot = bufferSlot;
      iargs.P = Pt;
      iargs.dPdu = dPdut;
      iargs.dPdv = dPdvt;
      iargs.ddPdudu = ddPdudut;
      iargs.ddPdvdv = ddPdvdvt;
      iargs.ddPdudv = ddPdudvt;
      iargs.valueCount = valueCount;
      interpolate(&iargs);
      
      if (likely(P)) {
        for (unsigned int j=0; j<valueCount; j++) 
          P[j*N+i] = Pt[j];
      }
      if (likely(dPdu)) 
      {
        for (unsigned int j=0; j<valueCount; j++) {
          dPdu[j*N+i] = dPdut[j];
          dPdv[j*N+i] = dPdvt[j];
        }
      }
      if (likely(ddPdudu)) 
      {
        for (unsigned int j=0; j<valueCount; j++) {
          ddPdudu[j*N+i] = ddPdudut[j];
          ddPdvdv[j*N+i] = ddPdvdvt[j];
          ddPdudv[j*N+i] = ddPdudvt[j];
        }
      }
    }
  }

  bool Geometry::pointQuery(PointQuery* query, PointQueryContext* context)
  {
    assert(context->primID < size());

    RTCPointQueryFunctionArguments args;
    args.query           = (RTCPointQuery*)context->query_ws;
    args.userPtr         = context->userPtr;
    args.primID          = context->primID;
    args.geomID          = context->geomID;
    args.context         = context->userContext;
    args.similarityScale = context->similarityScale;

    bool update = false;
    if(context->func)  update |= context->func(&args);
    if(pointQueryFunc) update |= pointQueryFunc(&args);

    if (update && context->userContext->instStackSize > 0)
    {
      // update point query
      if (context->query_type == POINT_QUERY_TYPE_AABB) {
        context->updateAABB();
      } else {
        assert(context->similarityScale > 0.f);
        query->radius = context->query_ws->radius * context->similarityScale;
      }
    }
    return update;
  }
}
