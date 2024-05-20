// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "default.h"
#include "device.h"
#include "buffer.h"
#include "../common/point_query.h"
#include "../builders/priminfo.h"
#include "../builders/priminfo_mb.h"

namespace embree
{
  class Scene;
  class Geometry;

  struct GeometryCounts 
  {
    __forceinline GeometryCounts()
      : numFilterFunctions(0),
        numTriangles(0), numMBTriangles(0), 
        numQuads(0), numMBQuads(0), 
        numBezierCurves(0), numMBBezierCurves(0), 
        numLineSegments(0), numMBLineSegments(0), 
        numSubdivPatches(0), numMBSubdivPatches(0), 
        numUserGeometries(0), numMBUserGeometries(0), 
        numInstancesCheap(0), numMBInstancesCheap(0), 
        numInstancesExpensive(0), numMBInstancesExpensive(0), 
        numInstanceArrays(0), numMBInstanceArrays(0),
        numGrids(0), numMBGrids(0),
        numSubGrids(0), numMBSubGrids(0), 
        numPoints(0), numMBPoints(0) {}

    __forceinline size_t size() const {
      return    numTriangles + numQuads + numBezierCurves + numLineSegments + numSubdivPatches + numUserGeometries + numInstancesCheap + numInstancesExpensive + numInstanceArrays + numGrids + numPoints
              + numMBTriangles + numMBQuads + numMBBezierCurves + numMBLineSegments + numMBSubdivPatches + numMBUserGeometries + numMBInstancesCheap + numMBInstancesExpensive + numMBInstanceArrays + numMBGrids + numMBPoints;
    }

    __forceinline unsigned int enabledGeometryTypesMask() const
    {
      unsigned int mask = 0;
      if (numTriangles) mask |= 1 << 0;
      if (numQuads) mask |= 1 << 1;
      if (numBezierCurves+numLineSegments) mask |= 1 << 2;
      if (numSubdivPatches) mask |= 1 << 3;
      if (numUserGeometries) mask |= 1 << 4;
      if (numInstancesCheap) mask |= 1 << 5;
      if (numInstancesExpensive) mask |= 1 << 6;
      if (numInstanceArrays) mask |= 1 << 7;
      if (numGrids) mask |= 1 << 8;
      if (numPoints) mask |= 1 << 9;

      unsigned int maskMB = 0;
      if (numMBTriangles) maskMB |= 1 << 0;
      if (numMBQuads) maskMB |= 1 << 1;
      if (numMBBezierCurves+numMBLineSegments) maskMB |= 1 << 2;
      if (numMBSubdivPatches) maskMB |= 1 << 3;
      if (numMBUserGeometries) maskMB |= 1 << 4;
      if (numMBInstancesCheap) maskMB |= 1 << 5;
      if (numMBInstancesExpensive) maskMB |= 1 << 6;
      if (numMBInstanceArrays) maskMB |= 1 << 7;
      if (numMBGrids) maskMB |= 1 << 8;
      if (numMBPoints) maskMB |= 1 << 9;
      
      return (mask<<8) + maskMB;
    }

    __forceinline GeometryCounts operator+ (GeometryCounts const & rhs) const
    {
      GeometryCounts ret;
      ret.numFilterFunctions = numFilterFunctions + rhs.numFilterFunctions;
      ret.numTriangles = numTriangles + rhs.numTriangles;
      ret.numMBTriangles = numMBTriangles + rhs.numMBTriangles;
      ret.numQuads = numQuads + rhs.numQuads;
      ret.numMBQuads = numMBQuads + rhs.numMBQuads;
      ret.numBezierCurves = numBezierCurves + rhs.numBezierCurves;
      ret.numMBBezierCurves = numMBBezierCurves + rhs.numMBBezierCurves;
      ret.numLineSegments = numLineSegments + rhs.numLineSegments;
      ret.numMBLineSegments = numMBLineSegments + rhs.numMBLineSegments;
      ret.numSubdivPatches = numSubdivPatches + rhs.numSubdivPatches;
      ret.numMBSubdivPatches = numMBSubdivPatches + rhs.numMBSubdivPatches;
      ret.numUserGeometries = numUserGeometries + rhs.numUserGeometries;
      ret.numMBUserGeometries = numMBUserGeometries + rhs.numMBUserGeometries;
      ret.numInstancesCheap = numInstancesCheap + rhs.numInstancesCheap;
      ret.numMBInstancesCheap = numMBInstancesCheap + rhs.numMBInstancesCheap;
      ret.numInstancesExpensive = numInstancesExpensive + rhs.numInstancesExpensive;
      ret.numMBInstancesExpensive = numMBInstancesExpensive + rhs.numMBInstancesExpensive;
      ret.numInstanceArrays = numInstanceArrays + rhs.numInstanceArrays;
      ret.numMBInstanceArrays = numMBInstanceArrays + rhs.numMBInstanceArrays;
      ret.numGrids = numGrids + rhs.numGrids;
      ret.numMBGrids = numMBGrids + rhs.numMBGrids;
      ret.numSubGrids = numSubGrids + rhs.numSubGrids;
      ret.numMBSubGrids = numMBSubGrids + rhs.numMBSubGrids;
      ret.numPoints = numPoints + rhs.numPoints;
      ret.numMBPoints = numMBPoints + rhs.numMBPoints;

      return ret;
    }

    size_t numFilterFunctions;       //!< number of geometries with filter functions enabled
    size_t numTriangles;             //!< number of enabled triangles
    size_t numMBTriangles;           //!< number of enabled motion blurred triangles
    size_t numQuads;                 //!< number of enabled quads
    size_t numMBQuads;               //!< number of enabled motion blurred quads
    size_t numBezierCurves;          //!< number of enabled curves
    size_t numMBBezierCurves;        //!< number of enabled motion blurred curves
    size_t numLineSegments;          //!< number of enabled line segments
    size_t numMBLineSegments;        //!< number of enabled line motion blurred segments
    size_t numSubdivPatches;         //!< number of enabled subdivision patches
    size_t numMBSubdivPatches;       //!< number of enabled motion blurred subdivision patches
    size_t numUserGeometries;        //!< number of enabled user geometries
    size_t numMBUserGeometries;      //!< number of enabled motion blurred user geometries
    size_t numInstancesCheap;        //!< number of enabled cheap instances
    size_t numMBInstancesCheap;      //!< number of enabled motion blurred cheap instances
    size_t numInstancesExpensive;    //!< number of enabled expensive instances
    size_t numMBInstancesExpensive;  //!< number of enabled motion blurred expensive instances
    size_t numInstanceArrays;        //!< number of enabled instance arrays
    size_t numMBInstanceArrays;      //!< number of enabled motion blurred instance arrays
    size_t numGrids;                 //!< number of enabled grid geometries
    size_t numMBGrids;               //!< number of enabled motion blurred grid geometries
    size_t numSubGrids;              //!< number of enabled grid geometries
    size_t numMBSubGrids;            //!< number of enabled motion blurred grid geometries
    size_t numPoints;                //!< number of enabled points
    size_t numMBPoints;              //!< number of enabled motion blurred points
  };

  /*! Base class all geometries are derived from */
  class Geometry : public RefCount
  {
    ALIGNED_CLASS_USM_(16);
    
    friend class Scene;
  public:

    /*! type of geometry */
    enum GType
    {
      GTY_FLAT_LINEAR_CURVE = 0,
      GTY_ROUND_LINEAR_CURVE = 1,
      GTY_ORIENTED_LINEAR_CURVE = 2,
      GTY_CONE_LINEAR_CURVE = 3,
      
      GTY_FLAT_BEZIER_CURVE = 4,
      GTY_ROUND_BEZIER_CURVE = 5,
      GTY_ORIENTED_BEZIER_CURVE = 6,
      
      GTY_FLAT_BSPLINE_CURVE = 8,
      GTY_ROUND_BSPLINE_CURVE = 9,
      GTY_ORIENTED_BSPLINE_CURVE = 10,

      GTY_FLAT_HERMITE_CURVE = 12,
      GTY_ROUND_HERMITE_CURVE = 13,
      GTY_ORIENTED_HERMITE_CURVE = 14,
      
      GTY_FLAT_CATMULL_ROM_CURVE = 16,
      GTY_ROUND_CATMULL_ROM_CURVE = 17,
      GTY_ORIENTED_CATMULL_ROM_CURVE = 18,      

      GTY_TRIANGLE_MESH = 20,
      GTY_QUAD_MESH = 21,
      GTY_GRID_MESH = 22,
      GTY_SUBDIV_MESH = 23,

      GTY_SPHERE_POINT = 25,
      GTY_DISC_POINT = 26,
      GTY_ORIENTED_DISC_POINT = 27,
      
      GTY_USER_GEOMETRY = 29,
      GTY_INSTANCE_CHEAP = 30,
      GTY_INSTANCE_EXPENSIVE = 31,
      GTY_INSTANCE_ARRAY = 24,
      GTY_END = 32,

      GTY_BASIS_LINEAR = 0,
      GTY_BASIS_BEZIER = 4,
      GTY_BASIS_BSPLINE = 8,
      GTY_BASIS_HERMITE = 12,
      GTY_BASIS_CATMULL_ROM = 16,
      GTY_BASIS_MASK = 28,

      GTY_SUBTYPE_FLAT_CURVE = 0,
      GTY_SUBTYPE_ROUND_CURVE = 1,
      GTY_SUBTYPE_ORIENTED_CURVE = 2,
      GTY_SUBTYPE_MASK = 3,
    };

    enum GSubType
    {
      GTY_SUBTYPE_DEFAULT= 0,
      GTY_SUBTYPE_INSTANCE_LINEAR = 0,
      GTY_SUBTYPE_INSTANCE_QUATERNION = 1
    };

    enum GTypeMask
    {
      MTY_FLAT_LINEAR_CURVE = 1ul << GTY_FLAT_LINEAR_CURVE,
      MTY_ROUND_LINEAR_CURVE = 1ul << GTY_ROUND_LINEAR_CURVE,
      MTY_CONE_LINEAR_CURVE = 1ul << GTY_CONE_LINEAR_CURVE,
      MTY_ORIENTED_LINEAR_CURVE = 1ul << GTY_ORIENTED_LINEAR_CURVE,
      
      MTY_FLAT_BEZIER_CURVE = 1ul << GTY_FLAT_BEZIER_CURVE,
      MTY_ROUND_BEZIER_CURVE = 1ul << GTY_ROUND_BEZIER_CURVE,
      MTY_ORIENTED_BEZIER_CURVE = 1ul << GTY_ORIENTED_BEZIER_CURVE,
      
      MTY_FLAT_BSPLINE_CURVE = 1ul << GTY_FLAT_BSPLINE_CURVE,
      MTY_ROUND_BSPLINE_CURVE = 1ul << GTY_ROUND_BSPLINE_CURVE,
      MTY_ORIENTED_BSPLINE_CURVE = 1ul << GTY_ORIENTED_BSPLINE_CURVE,

      MTY_FLAT_HERMITE_CURVE = 1ul << GTY_FLAT_HERMITE_CURVE,
      MTY_ROUND_HERMITE_CURVE = 1ul << GTY_ROUND_HERMITE_CURVE,
      MTY_ORIENTED_HERMITE_CURVE = 1ul << GTY_ORIENTED_HERMITE_CURVE,

      MTY_FLAT_CATMULL_ROM_CURVE = 1ul << GTY_FLAT_CATMULL_ROM_CURVE,
      MTY_ROUND_CATMULL_ROM_CURVE = 1ul << GTY_ROUND_CATMULL_ROM_CURVE,
      MTY_ORIENTED_CATMULL_ROM_CURVE = 1ul << GTY_ORIENTED_CATMULL_ROM_CURVE,

      MTY_CURVE2 = MTY_FLAT_LINEAR_CURVE | MTY_ROUND_LINEAR_CURVE | MTY_CONE_LINEAR_CURVE | MTY_ORIENTED_LINEAR_CURVE,
      
      MTY_CURVE4 = MTY_FLAT_BEZIER_CURVE | MTY_ROUND_BEZIER_CURVE | MTY_ORIENTED_BEZIER_CURVE |
                   MTY_FLAT_BSPLINE_CURVE | MTY_ROUND_BSPLINE_CURVE | MTY_ORIENTED_BSPLINE_CURVE |
                   MTY_FLAT_HERMITE_CURVE | MTY_ROUND_HERMITE_CURVE | MTY_ORIENTED_HERMITE_CURVE |
                   MTY_FLAT_CATMULL_ROM_CURVE | MTY_ROUND_CATMULL_ROM_CURVE | MTY_ORIENTED_CATMULL_ROM_CURVE,

      MTY_SPHERE_POINT = 1ul << GTY_SPHERE_POINT,
      MTY_DISC_POINT = 1ul << GTY_DISC_POINT,
      MTY_ORIENTED_DISC_POINT = 1ul << GTY_ORIENTED_DISC_POINT,

      MTY_POINTS = MTY_SPHERE_POINT | MTY_DISC_POINT | MTY_ORIENTED_DISC_POINT,

      MTY_CURVES = MTY_CURVE2 | MTY_CURVE4 | MTY_POINTS,

      MTY_TRIANGLE_MESH = 1ul << GTY_TRIANGLE_MESH,
      MTY_QUAD_MESH = 1ul << GTY_QUAD_MESH,
      MTY_GRID_MESH = 1ul << GTY_GRID_MESH,
      MTY_SUBDIV_MESH = 1ul << GTY_SUBDIV_MESH,
      MTY_USER_GEOMETRY = 1ul << GTY_USER_GEOMETRY,

      MTY_INSTANCE_CHEAP = 1ul << GTY_INSTANCE_CHEAP,
      MTY_INSTANCE_EXPENSIVE = 1ul << GTY_INSTANCE_EXPENSIVE,
      MTY_INSTANCE = MTY_INSTANCE_CHEAP | MTY_INSTANCE_EXPENSIVE,
      MTY_INSTANCE_ARRAY = 1ul << GTY_INSTANCE_ARRAY,

      MTY_ALL = -1
    };

    static const char* gtype_names[GTY_END];

    enum class State : unsigned {
      MODIFIED = 0,
      COMMITTED = 1,
    };

  public:
    
    /*! Geometry constructor */
    Geometry (Device* device, GType gtype, unsigned int numPrimitives, unsigned int numTimeSteps);

    /*! Geometry destructor */
    virtual ~Geometry();

  public:

    /*! tests if geometry is enabled */
    __forceinline bool isEnabled() const { return enabled; }

    /*! tests if geometry is disabled */
    __forceinline bool isDisabled() const { return !isEnabled(); }

    /* checks if argument version of filter functions are enabled */
    __forceinline bool hasArgumentFilterFunctions() const {
      return argumentFilterEnabled;
    }
    
    /*! tests if that geometry has some filter function set */
    __forceinline bool hasGeometryFilterFunctions () const {
      return (intersectionFilterN  != nullptr) || (occlusionFilterN  != nullptr);
    }

    /*! returns geometry type */
    __forceinline GType getType() const { return gtype; }

    /*! returns curve type */
    __forceinline GType getCurveType() const { return (GType)(gtype & GTY_SUBTYPE_MASK); }

    /*! returns curve basis */
    __forceinline GType getCurveBasis() const { return (GType)(gtype & GTY_BASIS_MASK); }

    /*! returns geometry type mask */
    __forceinline GTypeMask getTypeMask() const { return (GTypeMask)(1 << gtype); }

    /*! returns true of geometry contains motion blur */
    __forceinline bool hasMotionBlur () const {
      return numTimeSteps > 1;
    }

    /*! returns number of primitives */
    __forceinline size_t size() const { return numPrimitives; }

    /*! sets the number of primitives */
    virtual void setNumPrimitives(unsigned int numPrimitives_in);

    /*! sets number of time steps */
    virtual void setNumTimeSteps (unsigned int numTimeSteps_in);

    /*! sets motion blur time range */
    void setTimeRange (const BBox1f range);

    /*! gets motion blur time range */
    BBox1f getTimeRange () const;

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
      return getTimeSegment<N>(time,vfloat<N>(time_range.lower),vfloat<N>(time_range.upper),vfloat<N>(fnumTimeSegments),ftime);
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

    /*! called before every build */
    virtual void preCommit();
  
    /*! called after every build */
    virtual void postCommit();

    virtual void addElementsToCount (GeometryCounts & counts) const {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    };

    /*! sets constant tessellation rate for the geometry */
    virtual void setTessellationRate(float N) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }

    /*! Sets the maximal curve radius scale allowed by min-width feature. */
    virtual void setMaxRadiusScale(float s) {
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

    /* point query api */
    bool pointQuery(PointQuery* query, PointQueryContext* context);

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

    /*! get fast access to first vertex buffer if applicable */
    virtual float * getCompactVertexArray () const {
      return nullptr;
    }

    /*! Returns the modified counter - how many times the geo has been modified */
    __forceinline unsigned int getModCounter () const {
      return modCounter_;
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

    /* Enables argument version of intersection or occlusion filter function. */
    virtual void enableFilterFunctionFromArguments (bool enable) {
      argumentFilterEnabled = enable;
    }

    /*! for instances only */
  public:

    /*! Sets the instanced scene */
    virtual void setInstancedScene(const Ref<Scene>& scene) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry");
    }

    /*! Sets the instanced scenes */
    virtual void setInstancedScenes(const RTCScene* scenes, size_t numScenes) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry");
    }

    /*! Sets transformation of the instance */
    virtual void setTransform(const AffineSpace3fa& transform, unsigned int timeStep) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }

    /*! Sets transformation of the instance */
    virtual void setQuaternionDecomposition(const AffineSpace3ff& qd, unsigned int timeStep) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry"); 
    }

    /*! Returns the transformation of the instance */
    virtual AffineSpace3fa getTransform(float time) {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"operation not supported for this geometry");
    }

    /*! Returns the transformation of the instance */
    virtual AffineSpace3fa getTransform(size_t instance, float time) {
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
    
    /*! Set point query function. */
    void setPointQueryFunction(RTCPointQueryFunction func);

    /*! returns number of time segments */
    __forceinline unsigned numTimeSegments () const {
      return numTimeSteps-1;
    }

  public:

    virtual PrimInfo createPrimRefArray(PrimRef* prims, const range<size_t>& r, size_t k, unsigned int geomID) const {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"createPrimRefArray not implemented for this geometry"); 
    }

    PrimInfo createPrimRefArray(mvector<PrimRef>& prims, const range<size_t>& r, size_t k, unsigned int geomID) const {
      return createPrimRefArray(prims.data(),r,k,geomID);
    }

    PrimInfo createPrimRefArray(avector<PrimRef>& prims, const range<size_t>& r, size_t k, unsigned int geomID) const {
      return createPrimRefArray(prims.data(),r,k,geomID);
    }

    virtual PrimInfo createPrimRefArray(mvector<PrimRef>& prims, mvector<SubGridBuildData>& sgrids, const range<size_t>& r, size_t k, unsigned int geomID) const {
      return createPrimRefArray(prims,r,k,geomID);
    }

    virtual PrimInfo createPrimRefArrayMB(mvector<PrimRef>& prims, size_t itime, const range<size_t>& r, size_t k, unsigned int geomID) const {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"createPrimRefMBArray not implemented for this geometry"); 
    }

    /*! Calculates the PrimRef over the complete time interval */
    virtual PrimInfo createPrimRefArrayMB(PrimRef* prims, const BBox1f& t0t1, const range<size_t>& r, size_t k, unsigned int geomID) const {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"createPrimRefMBArray not implemented for this geometry");
    }

    PrimInfo createPrimRefArrayMB(mvector<PrimRef>& prims, const BBox1f& t0t1, const range<size_t>& r, size_t k, unsigned int geomID) const {
      return createPrimRefArrayMB(prims.data(),t0t1,r,k,geomID);
    }

    PrimInfo createPrimRefArrayMB(avector<PrimRef>& prims, const BBox1f& t0t1, const range<size_t>& r, size_t k, unsigned int geomID) const {
      return createPrimRefArrayMB(prims.data(),t0t1,r,k,geomID);
    }
    
    virtual PrimInfoMB createPrimRefMBArray(mvector<PrimRefMB>& prims, const BBox1f& t0t1, const range<size_t>& r, size_t k, unsigned int geomID) const {
      throw_RTCError(RTC_ERROR_INVALID_OPERATION,"createPrimRefMBArray not implemented for this geometry"); 
    }

    virtual PrimInfoMB createPrimRefMBArray(mvector<PrimRefMB>& prims, mvector<SubGridBuildData>& sgrids, const BBox1f& t0t1, const range<size_t>& r, size_t k, unsigned int geomID) const {
      return createPrimRefMBArray(prims,t0t1,r,k,geomID);
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

    virtual LBBox3fa vlinearBounds(size_t primID, const BBox1f& time_range, const SubGridBuildData * const sgrids) const {
      return vlinearBounds(primID,time_range);
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

    void* userPtr;              //!< user pointer
    unsigned int numPrimitives; //!< number of primitives of this geometry
    
    unsigned int numTimeSteps;  //!< number of time steps
    float fnumTimeSegments;     //!< number of time segments (precalculation)
    BBox1f time_range;          //!< motion blur time range
    
    unsigned int mask;             //!< for masking out geometry
    unsigned int modCounter_ = 1; //!< counter for every modification - used to rebuild scenes when geo is modified

    struct {
      GType gtype : 8;                //!< geometry type
      GSubType gsubtype : 8;          //!< geometry subtype
      RTCBuildQuality quality : 3;    //!< build quality for geometry
      unsigned state : 2;
      bool enabled : 1;               //!< true if geometry is enabled
      bool argumentFilterEnabled : 1; //!< true if argument filter functions are enabled for this geometry
    };
       
    RTCFilterFunctionN intersectionFilterN;
    RTCFilterFunctionN occlusionFilterN;
    RTCPointQueryFunction pointQueryFunc;
  };
}
