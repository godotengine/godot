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
    "instance",
  };
     
  Geometry::Geometry (Device* device, GType gtype, unsigned int numPrimitives, unsigned int numTimeSteps) 
    : device(device), scene(nullptr), userPtr(nullptr),
      geomID(0), numPrimitives(numPrimitives), numTimeSteps(unsigned(numTimeSteps)), fnumTimeSegments(float(numTimeSteps-1)), time_range(0.0f,1.0f),
      mask(-1),
      gtype(gtype),
      quality(RTC_BUILD_QUALITY_MEDIUM),
      state(MODIFIED),
      numPrimitivesChanged(false),
      enabled(true),
      intersectionFilterN(nullptr), occlusionFilterN(nullptr)
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
    
    if (isEnabled() && scene) disabling();
    numPrimitives = numPrimitives_in;
    numPrimitivesChanged = true;
    if (isEnabled() && scene) enabling();
    
    Geometry::update();
  }

  void Geometry::setNumTimeSteps (unsigned int numTimeSteps_in)
  {
    if (numTimeSteps_in == numTimeSteps)
      return;
    
    if (isEnabled() && scene) disabling();
    numTimeSteps = numTimeSteps_in;
    fnumTimeSegments = float(numTimeSteps_in-1);
    if (isEnabled() && scene) enabling();
    
    Geometry::update();
  }

  void Geometry::setTimeRange (const BBox1f range)
  {
    time_range = range;
    Geometry::update();
  }
  
  void Geometry::update() 
  {
    if (scene)
      scene->setModified();

    state = MODIFIED;
  }
  
  void Geometry::commit() 
  {
    if (scene)
      scene->setModified();

    state = COMMITTED;
  }

  void Geometry::preCommit()
  {
    if (state == MODIFIED)
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"geometry got not committed");
  }

  void Geometry::postCommit()
  {
    numPrimitivesChanged = false;
    
    /* set state to build */
    if (isEnabled())
      state = BUILD;
  }

  void Geometry::updateIntersectionFilters(bool enable)
  {
    const size_t numN  = (intersectionFilterN  != nullptr) + (occlusionFilterN  != nullptr);

    if (enable) {
      scene->numIntersectionFiltersN += numN;
    } else {
      scene->numIntersectionFiltersN -= numN;
    }
  }

  Geometry* Geometry::attach(Scene* scene, unsigned int geomID)
  {
    assert(scene);
    this->scene = scene;
    this->geomID = geomID;
    if (isEnabled()) {
      scene->setModified();
      updateIntersectionFilters(true);
      enabling();
    }
    return this;
  }

  void Geometry::detach()
  {
    if (isEnabled()) {
      scene->setModified();
      updateIntersectionFilters(false);
      disabling();
    }
    this->scene = nullptr;
    this->geomID = -1;
  }
  
  void Geometry::enable () 
  {
    if (isEnabled()) 
      return;

    if (scene) {
      updateIntersectionFilters(true);
      scene->setModified();
      enabling();
    }

    enabled = true;
  }

  void Geometry::disable () 
  {
    if (isDisabled()) 
      return;

    if (scene) {
      updateIntersectionFilters(false);
      scene->setModified();
      disabling();
    }
    
    enabled = false;
  }

  void Geometry::setUserData (void* ptr)
  {
    userPtr = ptr;
  }
  
  void Geometry::setIntersectionFilterFunctionN (RTCFilterFunctionN filter) 
  {
    if (!(getTypeMask() & (MTY_TRIANGLE_MESH | MTY_QUAD_MESH | MTY_CURVES | MTY_SUBDIV_MESH | MTY_USER_GEOMETRY | MTY_GRID_MESH)))
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"filter functions not supported for this geometry"); 

    if (scene && isEnabled()) {
      scene->numIntersectionFiltersN -= intersectionFilterN != nullptr;
      scene->numIntersectionFiltersN += filter != nullptr;
    }
    intersectionFilterN = filter;
  }

  void Geometry::setOcclusionFilterFunctionN (RTCFilterFunctionN filter) 
  {
    if (!(getTypeMask() & (MTY_TRIANGLE_MESH | MTY_QUAD_MESH | MTY_CURVES | MTY_SUBDIV_MESH | MTY_USER_GEOMETRY | MTY_GRID_MESH)))
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"filter functions not supported for this geometry"); 

    if (scene && isEnabled()) {
      scene->numIntersectionFiltersN -= occlusionFilterN != nullptr;
      scene->numIntersectionFiltersN += filter != nullptr;
    }
    occlusionFilterN = filter;
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
}
