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

#include "default.h"
#include "device.h"
#include "buffer.h"
#include "../builders/priminfo.h"

namespace embree
{
  class Scene;

  /*! Base class all geometries are derived from */
  class Geometry : public RefCount
  {
    friend class Scene;
  public:

    /*! type of geometry */
    enum GType
    {
      GTY_FLAT_LINEAR_CURVE = 0,
      GTY_ROUND_LINEAR_CURVE = 1,
      GTY_ORIENTED_LINEAR_CURVE = 2,
      
      GTY_FLAT_BEZIER_CURVE = 4,
      GTY_ROUND_BEZIER_CURVE = 5,
      GTY_ORIENTED_BEZIER_CURVE = 6,
      
      GTY_FLAT_BSPLINE_CURVE = 8,
      GTY_ROUND_BSPLINE_CURVE = 9,
      GTY_ORIENTED_BSPLINE_CURVE = 10,

      GTY_FLAT_HERMITE_CURVE = 12,
      GTY_ROUND_HERMITE_CURVE = 13,
      GTY_ORIENTED_HERMITE_CURVE = 14,

      GTY_TRIANGLE_MESH = 16,
      GTY_QUAD_MESH = 17,
      GTY_GRID_MESH = 18,
      GTY_SUBDIV_MESH = 19,

      GTY_SPHERE_POINT = 21,
      GTY_DISC_POINT = 22,
      GTY_ORIENTED_DISC_POINT = 23,
      
      GTY_USER_GEOMETRY = 25,
      GTY_INSTANCE = 26,
      GTY_END = 27,

      GTY_BASIS_LINEAR = 0,
      GTY_BASIS_BEZIER = 4,
      GTY_BASIS_BSPLINE = 8,
      GTY_BASIS_HERMITE = 12,
      GTY_BASIS_MASK = 12,
      
      GTY_SUBTYPE_FLAT_CURVE = 0,
      GTY_SUBTYPE_ROUND_CURVE = 1,
      GTY_SUBTYPE_ORIENTED_CURVE = 2,
      GTY_SUBTYPE_MASK = 3,
    };

    enum GTypeMask
    {
      MTY_FLAT_LINEAR_CURVE = 1 << GTY_FLAT_LINEAR_CURVE,
      MTY_ROUND_LINEAR_CURVE = 1 << GTY_ROUND_LINEAR_CURVE,
      MTY_ORIENTED_LINEAR_CURVE = 1 << GTY_ORIENTED_LINEAR_CURVE,
      
      MTY_FLAT_BEZIER_CURVE = 1 << GTY_FLAT_BEZIER_CURVE,
      MTY_ROUND_BEZIER_CURVE = 1 << GTY_ROUND_BEZIER_CURVE,
      MTY_ORIENTED_BEZIER_CURVE = 1 << GTY_ORIENTED_BEZIER_CURVE,
      
      MTY_FLAT_BSPLINE_CURVE = 1 << GTY_FLAT_BSPLINE_CURVE,
      MTY_ROUND_BSPLINE_CURVE = 1 << GTY_ROUND_BSPLINE_CURVE,
      MTY_ORIENTED_BSPLINE_CURVE = 1 << GTY_ORIENTED_BSPLINE_CURVE,

      MTY_FLAT_HERMITE_CURVE = 1 << GTY_FLAT_HERMITE_CURVE,
      MTY_ROUND_HERMITE_CURVE = 1 << GTY_ROUND_HERMITE_CURVE,
      MTY_ORIENTED_HERMITE_CURVE = 1 << GTY_ORIENTED_HERMITE_CURVE,

      MTY_CURVE2 = MTY_FLAT_LINEAR_CURVE | MTY_ROUND_LINEAR_CURVE | MTY_ORIENTED_LINEAR_CURVE,
      
      MTY_CURVE4 = MTY_FLAT_BEZIER_CURVE | MTY_ROUND_BEZIER_CURVE | MTY_ORIENTED_BEZIER_CURVE |
                   MTY_FLAT_BSPLINE_CURVE | MTY_ROUND_BSPLINE_CURVE | MTY_ORIENTED_BSPLINE_CURVE |
                   MTY_FLAT_HERMITE_CURVE | MTY_ROUND_HERMITE_CURVE | MTY_ORIENTED_HERMITE_CURVE,

      MTY_SPHERE_POINT = 1 << GTY_SPHERE_POINT,
      MTY_DISC_POINT = 1 << GTY_DISC_POINT,
      MTY_ORIENTED_DISC_POINT = 1 << GTY_ORIENTED_DISC_POINT,

      MTY_POINTS = MTY_SPHERE_POINT | MTY_DISC_POINT | MTY_ORIENTED_DISC_POINT,

      MTY_CURVES = MTY_CURVE2 | MTY_CURVE4 | MTY_POINTS,

      MTY_TRIANGLE_MESH = 1 << GTY_TRIANGLE_MESH,
      MTY_QUAD_MESH = 1 << GTY_QUAD_MESH,
      MTY_GRID_MESH = 1 << GTY_GRID_MESH,
      MTY_SUBDIV_MESH = 1 << GTY_SUBDIV_MESH,
      MTY_USER_GEOMETRY = 1 << GTY_USER_GEOMETRY,
      MTY_INSTANCE = 1 << GTY_INSTANCE,
    };

    static const char* gtype_names[GTY_END];

    enum State {
      MODIFIED = 0,
      COMMITTED = 1,
      BUILD = 2
    };

  public:
    
    /*! Geometry constructor */
    Geometry (Device* device, GType gtype, unsigned int numPrimitives, unsigned int numTimeSteps);

    /*! Geometry destructor */
    virtual ~Geometry();

    /*! updates intersection filter function counts in scene */
    void updateIntersectionFilters(bool enable);

  public:

    /*! tests if geometry is enabled */
    __forceinline bool isEnabled() const { return enabled; }

    /*! tests if geometry is disabled */
    __forceinline bool isDisabled() const { return !isEnabled(); }

    /*! tests if geometry is modified */
    __forceinline bool isModified() const { return state != BUILD; }

    /*! marks geometry modified */
    __forceinline void setModified() {
      if (state == BUILD) state = COMMITTED;
    }

    /*! returns geometry type */
    __forceinline GType getType() const { return gtype; }

    /*! returns curve type */
    __forceinline GType getCurveType() const { return (GType)(gtype & GTY_SUBTYPE_MASK); }

    /*! returns curve basis */
    __forceinline GType getCurveBasis() const { return (GType)(gtype & GTY_BASIS_MASK); }

    /*! returns geometry type mask */
    __forceinline GTypeMask getTypeMask() const { return (GTypeMask)(1 << gtype); }

    /*! returns number of primitives */
    __forceinline size_t size() const { return numPrimitives; }

    /*! sets the number of primitives */
    virtual void setNumPrimitives(unsigned int numPrimitives_in);

    /*! sets number of time steps */
    virtual void setNumTimeSteps (unsigned int numTimeSteps_in);

    /*! sets motion blur time range */
    void setTimeRange (const BBox1f range);

    /*! sets number of vertex attributes */
    virtual void setVertexAttributeCount (unsigned int N) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }

    /*! sets number of topologies */
    virtual void setTopologyCount (unsigned int N) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }

    /*! sets the build quality */
    void setBuildQuality(RTCBuildQuality quality_in)
    {
      this->quality = quality_in;
      Geometry::update();
    }

    /* calculate time segment itime and fractional time ftime */
    __forceinline int timeSegment(float time, float& ftime) const {
      return getTimeSegment(time,time_range.lower,time_range.upper,fnumTimeSegments,ftime);
    }

    template<int N>
      __forceinline vint<N> timeSegment(const vfloat<N>& time, vfloat<N>& ftime) const {
      return getTimeSegment(time,vfloat<N>(time_range.lower),vfloat<N>(time_range.upper),vfloat<N>(fnumTimeSegments),ftime);
    }
    
    /* calculate overlapping time segment range */
    __forceinline range<int> timeSegmentRange(const BBox1f& range) const {
      return getTimeSegmentRange(range,time_range,fnumTimeSegments);
    }

    /* returns time that corresponds to time step */
    __forceinline float timeStep(const int i) const {
      assert(i>=0 && i<(int)numTimeSteps);
      return time_range.lower + time_range.size()*float(i)/fnumTimeSegments;
    }
    
    /*! for all geometries */
  public:

    Geometry* attach(Scene* scene, unsigned int geomID);
    void detach();

    /*! Enable geometry. */
    virtual void enable();

    /*! Update geometry. */
    void update();
    
    /*! commit of geometry */
    virtual void commit();

    /*! Update geometry buffer. */
    virtual void updateBuffer(RTCBufferType type, unsigned int slot) {
      update(); // update everything for geometries not supporting this call
    }
    
    /*! Disable geometry. */
    virtual void disable();

    /*! Verify the geometry */
    virtual bool verify() { return true; }

    /*! called if geometry is switching from disabled to enabled state */
    virtual void enabling() = 0;

    /*! called if geometry is switching from enabled to disabled state */
    virtual void disabling() = 0;

    /*! called before every build */
    virtual void preCommit();
  
    /*! called after every build */
    virtual void postCommit();

    /*! sets constant tessellation rate for the geometry */
    virtual void setTessellationRate(float N) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }

    /*! Set user data pointer. */
    virtual void setUserData(void* ptr);
      
    /*! Get user data pointer. */
    __forceinline void* getUserData() const {
      return userPtr;
    }

    /*! interpolates user data to the specified u/v location */
    virtual void interpolate(const RTCInterpolateArguments* const args) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }

    /*! interpolates user data to the specified u/v locations */
    virtual void interpolateN(const RTCInterpolateNArguments* const args);

    /*! for subdivision surfaces only */
  public:
    virtual void setSubdivisionMode (unsigned topologyID, RTCSubdivisionMode mode) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }

    virtual void setVertexAttributeTopology(unsigned int vertexBufferSlot, unsigned int indexBufferSlot) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }

    /*! Set displacement function. */
    virtual void setDisplacementFunction (RTCDisplacementFunctionN filter) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }

    virtual unsigned int getFirstHalfEdge(unsigned int faceID) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }

    virtual unsigned int getFace(unsigned int edgeID) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }
    
    virtual unsigned int getNextHalfEdge(unsigned int edgeID) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }

    virtual unsigned int getPreviousHalfEdge(unsigned int edgeID) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }

    virtual unsigned int getOppositeHalfEdge(unsigned int topologyID, unsigned int edgeID) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }

    /*! for triangle meshes and bezier curves only */
  public:


    /*! Sets ray mask. */
    virtual void setMask(unsigned mask) { 
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }
    
    /*! Sets specified buffer. */
    virtual void setBuffer(RTCBufferType type, unsigned int slot, RTCFormat format, const Ref<Buffer>& buffer, size_t offset, size_t stride, unsigned int num) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }

    /*! Gets specified buffer. */
    virtual void* getBuffer(RTCBufferType type, unsigned int slot) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry");
    }

    /*! Set intersection filter function for ray packets of size N. */
    virtual void setIntersectionFilterFunctionN (RTCFilterFunctionN filterN);

    /*! Set occlusion filter function for ray packets of size N. */
    virtual void setOcclusionFilterFunctionN (RTCFilterFunctionN filterN);

    /*! for instances only */
  public:

    /*! Sets the instanced scene */
    virtual void setInstancedScene(const Ref<Scene>& scene) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }
    
    /*! Sets transformation of the instance */
    virtual void setTransform(const AffineSpace3fa& transform, unsigned int timeStep) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }

    /*! Returns the transformation of the instance */
    virtual AffineSpace3fa getTransform(float time) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }

    /*! for user geometries only */
  public:

    /*! Set bounds function. */
    virtual void setBoundsFunction (RTCBoundsFunction bounds, void* userPtr) { 
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }

    /*! Set intersect function for ray packets of size N. */
    virtual void setIntersectFunctionN (RTCIntersectFunctionN intersect) { 
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }
    
    /*! Set occlusion function for ray packets of size N. */
    virtual void setOccludedFunctionN (RTCOccludedFunctionN occluded) { 
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }

    /*! returns number of time segments */
    __forceinline unsigned numTimeSegments () const {
      return numTimeSteps-1;
    }

  public:

    virtual PrimInfo createPrimRefArray(mvector<PrimRef>& prims, const range<size_t>& r, size_t k) const {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"createPrimRefArray not implemented for this geometry"); 
    }

    virtual PrimInfo createPrimRefArrayMB(mvector<PrimRef>& prims, size_t itime, const range<size_t>& r, size_t k) const {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"createPrimRefMBArray not implemented for this geometry"); 
    }

    virtual PrimInfoMB createPrimRefMBArray(mvector<PrimRefMB>& prims, const BBox1f& t0t1, const range<size_t>& r, size_t k) const {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"createPrimRefMBArray not implemented for this geometry"); 
    }

    virtual LinearSpace3fa computeAlignedSpace(const size_t primID) const {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"computeAlignedSpace not implemented for this geometry"); 
    }

    virtual LinearSpace3fa computeAlignedSpaceMB(const size_t primID, const BBox1f time_range) const {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"computeAlignedSpace not implemented for this geometry"); 
    }
    
    virtual Vec3fa computeDirection(unsigned int primID) const {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"computeDirection not implemented for this geometry"); 
    }

    virtual Vec3fa computeDirection(unsigned int primID, size_t time) const {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"computeDirection not implemented for this geometry"); 
    }

    virtual BBox3fa vbounds(size_t primID) const {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"vbounds not implemented for this geometry"); 
    }
    
    virtual BBox3fa vbounds(const LinearSpace3fa& space, size_t primID) const {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"vbounds not implemented for this geometry"); 
    }

    virtual BBox3fa vbounds(const Vec3fa& ofs, const float scale, const float r_scale0, const LinearSpace3fa& space, size_t i, size_t itime = 0) const {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"vbounds not implemented for this geometry"); 
    }

    virtual LBBox3fa vlinearBounds(size_t primID, const BBox1f& time_range) const {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"vlinearBounds not implemented for this geometry"); 
    }
    
    virtual LBBox3fa vlinearBounds(const LinearSpace3fa& space, size_t primID, const BBox1f& time_range) const {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"vlinearBounds not implemented for this geometry"); 
    }

    virtual LBBox3fa vlinearBounds(const Vec3fa& ofs, const float scale, const float r_scale0, const LinearSpace3fa& space, size_t primID, const BBox1f& time_range) const {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"vlinearBounds not implemented for this geometry"); 
    }
    
  public:
    __forceinline bool hasIntersectionFilter() const { return intersectionFilterN != nullptr; }
    __forceinline bool hasOcclusionFilter() const { return occlusionFilterN != nullptr; }

  public:
    Device* device;             //!< device this geometry belongs to
    Scene* scene;               //!< pointer to scene this mesh belongs to

    void* userPtr;              //!< user pointer
    unsigned int geomID;        //!< internal geometry ID
    unsigned int numPrimitives; //!< number of primitives of this geometry
    
    unsigned int numTimeSteps;  //!< number of time steps
    float fnumTimeSegments;     //!< number of time segments (precalculation)
    BBox1f time_range;          //!< motion blur time range
    
    unsigned int mask;             //!< for masking out geometry
    struct {
      GType gtype : 6;                //!< geometry type
      RTCBuildQuality quality : 3;    //!< build quality for geometry
      State state : 2;
      bool numPrimitivesChanged : 1; //!< true if number of primitives changed
      bool enabled : 1;              //!< true if geometry is enabled
    };
       
    RTCFilterFunctionN intersectionFilterN;
    RTCFilterFunctionN occlusionFilterN;
  };
}
