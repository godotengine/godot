// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "default.h"
#include "geometry.h"
#include "buffer.h"

namespace embree
{
  /*! represents an array of line segments */
  struct LineSegments : public Geometry
  {
    /*! type of this geometry */
    static const Geometry::GTypeMask geom_type = Geometry::MTY_CURVE2;

  public:

    /*! line segments construction */
    LineSegments (Device* device, Geometry::GType gtype);

  public:
    void setMask (unsigned mask);
    void setNumTimeSteps (unsigned int numTimeSteps);
    void setVertexAttributeCount (unsigned int N);
    void setBuffer(RTCBufferType type, unsigned int slot, RTCFormat format, const Ref<Buffer>& buffer, size_t offset, size_t stride, unsigned int num);
    void* getBufferData(RTCBufferType type, unsigned int slot, BufferDataPointerType pointerType);
    void updateBuffer(RTCBufferType type, unsigned int slot);
    void commit();
    bool verify ();
    void interpolate(const RTCInterpolateArguments* const args);
    void setTessellationRate(float N);
    void setMaxRadiusScale(float s);
    void addElementsToCount (GeometryCounts & counts) const;
    size_t getGeometryDataDeviceByteSize() const;
    void convertToDeviceRepresentation(size_t offset, char* data_host, char* data_device) const;

    template<int N>
    void interpolate_impl(const RTCInterpolateArguments* const args)
    {
      unsigned int primID = args->primID;
      float u = args->u;
      RTCBufferType bufferType = args->bufferType;
      unsigned int bufferSlot = args->bufferSlot;
      float* P = args->P;
      float* dPdu = args->dPdu;
      float* ddPdudu = args->ddPdudu;
      unsigned int valueCount = args->valueCount;
      
      /* calculate base pointer and stride */
      assert((bufferType == RTC_BUFFER_TYPE_VERTEX && bufferSlot < numTimeSteps) ||
             (bufferType == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE && bufferSlot <= vertexAttribs.size()));
      const char* src = nullptr;
      size_t stride = 0;
      if (bufferType == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE) {
        src    = vertexAttribs[bufferSlot].getPtr();
        stride = vertexAttribs[bufferSlot].getStride();
      } else {
        src    = vertices[bufferSlot].getPtr();
        stride = vertices[bufferSlot].getStride();
      }
      
      for (unsigned int i=0; i<valueCount; i+=N)
      {
        const size_t ofs = i*sizeof(float);
        const size_t segment = segments[primID];
        const vbool<N> valid = vint<N>((int)i)+vint<N>(step) < vint<N>(int(valueCount));
        const vfloat<N> p0 = mem<vfloat<N>>::loadu(valid,(float*)&src[(segment+0)*stride+ofs]);
        const vfloat<N> p1 = mem<vfloat<N>>::loadu(valid,(float*)&src[(segment+1)*stride+ofs]);
        if (P      ) mem<vfloat<N>>::storeu(valid,P+i,lerp(p0,p1,u));
        if (dPdu   ) mem<vfloat<N>>::storeu(valid,dPdu+i,p1-p0);
        if (ddPdudu) mem<vfloat<N>>::storeu(valid,dPdu+i,vfloat<N>(zero));
      }
    }
    
  public:

    /*! returns the number of vertices */
    __forceinline size_t numVertices() const {
      return vertices[0].size();
    }

    /*! returns the i'th segment */
    __forceinline const unsigned int& segment(size_t i) const {
      return segments[i];
    }

#if defined(EMBREE_SYCL_SUPPORT) && defined(__SYCL_DEVICE_ONLY__)
    /*! returns the i'th segment */
    template<int M>
    __forceinline const vuint<M> vsegment(const vuint<M>& i) const {
      return segments[i.v];
    }
#endif

    /*! returns the segment to the left of the i'th segment */
    __forceinline bool segmentLeftExists(size_t i) const {
      assert (flags);
      return (flags[i] & RTC_CURVE_FLAG_NEIGHBOR_LEFT) != 0;
    }

    /*! returns the segment to the right of the i'th segment */
    __forceinline bool segmentRightExists(size_t i) const {
      assert (flags);
      return (flags[i] & RTC_CURVE_FLAG_NEIGHBOR_RIGHT) != 0;
    }

     /*! returns i'th vertex of the first time step */
    __forceinline Vec3ff vertex(size_t i) const {
      return vertices0[i];
    }

    /*! returns i'th vertex of the first time step */
    __forceinline const char* vertexPtr(size_t i) const {
      return vertices0.getPtr(i);
    }

    /*! returns i'th normal of the first time step */
    __forceinline Vec3fa normal(size_t i) const {
      return normals0[i];
    }

    /*! returns i'th radius of the first time step */
    __forceinline float radius(size_t i) const {
      return vertices0[i].w;
    }

    /*! returns i'th vertex of itime'th timestep */
    __forceinline Vec3ff vertex(size_t i, size_t itime) const {
      return vertices[itime][i];
    }

    /*! returns i'th vertex of itime'th timestep */
    __forceinline const char* vertexPtr(size_t i, size_t itime) const {
      return vertices[itime].getPtr(i);
    }

    /*! returns i'th normal of itime'th timestep */
    __forceinline Vec3fa normal(size_t i, size_t itime) const {
      return normals[itime][i];
    }

    /*! returns i'th radius of itime'th timestep */
    __forceinline float radius(size_t i, size_t itime) const {
      return vertices[itime][i].w;
    }

    /*! gathers the curve starting with i'th vertex */
    __forceinline void gather(Vec3ff& p0, Vec3ff& p1, unsigned int vid) const
    {
      p0 = vertex(vid+0);
      p1 = vertex(vid+1);
    }

#if defined(EMBREE_SYCL_SUPPORT) && defined(__SYCL_DEVICE_ONLY__)
    template<int M>
    __forceinline void vgather(Vec4vf<M>& p0, Vec4vf<M>& p1, const vuint<M>& vid) const
    {
      p0 = vertex(vid.v+0);
      p1 = vertex(vid.v+1);
    }
#endif

    /*! gathers the curve starting with i'th vertex of itime'th timestep */
    __forceinline void gather(Vec3ff& p0, Vec3ff& p1, unsigned int vid, size_t itime) const
    {
      p0 = vertex(vid+0,itime);
      p1 = vertex(vid+1,itime);
    }

#if defined(EMBREE_SYCL_SUPPORT) && defined(__SYCL_DEVICE_ONLY__)
    template<int M>
    __forceinline void vgather(Vec4vf<M>& p0, Vec4vf<M>& p1, const vuint<M>& vid, const vint<M>& itime) const
    {
      p0 = vertex(vid.v+0,itime.v);
      p1 = vertex(vid.v+1,itime.v);
    }
#endif

     /*! loads curve vertices for specified time */
    __forceinline void gather(Vec3ff& p0, Vec3ff& p1, unsigned int vid, float time) const
    {
      float ftime;
      const size_t itime = timeSegment(time, ftime);

      const float t0 = 1.0f - ftime;
      const float t1 = ftime;
      Vec3ff a0,a1; gather(a0,a1,vid,itime);
      Vec3ff b0,b1; gather(b0,b1,vid,itime+1);
      p0 = madd(Vec3ff(t0),a0,t1*b0);
      p1 = madd(Vec3ff(t0),a1,t1*b1);
    }

    /*! loads curve vertices for specified time for mblur and non-mblur case */
    __forceinline void gather_safe(Vec3ff& p0, Vec3ff& p1, unsigned int vid, float time) const
    {
      if (hasMotionBlur()) gather(p0,p1,vid,time);
      else                 gather(p0,p1,vid);
    }

#if defined(EMBREE_SYCL_SUPPORT) && defined(__SYCL_DEVICE_ONLY__)
    template<int M>
    __forceinline void vgather(Vec4vf<M>& p0, Vec4vf<M>& p1, const vuint<M>& vid, const vfloat<M>& time) const
    {
      vfloat<M> ftime;
      const vint<M> itime = timeSegment<M>(time, ftime);

      const vfloat<M> t0 = 1.0f - ftime;
      const vfloat<M> t1 = ftime;
      Vec4vf<M> a0,a1; vgather<M>(a0,a1,vid,itime);
      Vec4vf<M> b0,b1; vgather<M>(b0,b1,vid,itime+1);
      p0 = madd(Vec4vf<M>(t0),a0,t1*b0);
      p1 = madd(Vec4vf<M>(t0),a1,t1*b1);
    }
#endif
    
    /*! gathers the cone curve starting with i'th vertex */
    __forceinline void gather(Vec3ff& p0, Vec3ff& p1, bool& cL, bool& cR, unsigned int primID, unsigned int vid) const
    {
      gather(p0,p1,vid);
      cL = !segmentLeftExists (primID);
      cR = !segmentRightExists(primID);
    }

#if defined(EMBREE_SYCL_SUPPORT) && defined(__SYCL_DEVICE_ONLY__)
    template<int M>
    __forceinline void vgather(Vec4vf<M>& p0, Vec4vf<M>& p1, vbool<M>& cL, vbool<M>& cR, const vuint<M>& primID, const vuint<M>& vid) const
    {
      vgather<M>(p0,p1,vid);
      cL = !segmentLeftExists (primID.v);
      cR = !segmentRightExists(primID.v);
    }
#endif

    /*! gathers the cone curve starting with i'th vertex of itime'th timestep */
    __forceinline void gather(Vec3ff& p0, Vec3ff& p1, bool& cL, bool& cR, unsigned int primID, size_t vid, size_t itime) const
    {
      gather(p0,p1,vid,itime);
      cL = !segmentLeftExists (primID);
      cR = !segmentRightExists(primID);
    }

     /*! loads cone curve vertices for specified time */
    __forceinline void gather(Vec3ff& p0, Vec3ff& p1, bool& cL, bool& cR, unsigned int primID, size_t vid, float time) const
    {
      gather(p0,p1,vid,time);
      cL = !segmentLeftExists (primID);
      cR = !segmentRightExists(primID);
    }

    /*! loads cone curve vertices for specified time for mblur and non-mblur geometry */
    __forceinline void gather_safe(Vec3ff& p0, Vec3ff& p1, bool& cL, bool& cR, unsigned int primID, size_t vid, float time) const
    {
      if (hasMotionBlur()) gather(p0,p1,cL,cR,primID,vid,time);
      else                 gather(p0,p1,cL,cR,primID,vid);
    }

#if defined(EMBREE_SYCL_SUPPORT) && defined(__SYCL_DEVICE_ONLY__)
    template<int M>
    __forceinline void vgather(Vec4vf<M>& p0, Vec4vf<M>& p1, vbool<M>& cL, vbool<M>& cR, const vuint<M>& primID, const vuint<M>& vid, const vfloat<M>& time) const
    {
      vgather<M>(p0,p1,vid,time);
      cL = !segmentLeftExists (primID.v);
      cR = !segmentRightExists(primID.v);
    }
#endif

    /*! gathers the curve starting with i'th vertex */
    __forceinline void gather(Vec3ff& p0, Vec3ff& p1, Vec3ff& p2, Vec3ff& p3, unsigned int primID, size_t vid) const
    {
      p0 = vertex(vid+0);
      p1 = vertex(vid+1);
      p2 = segmentLeftExists (primID) ? vertex(vid-1) : Vec3ff(inf);
      p3 = segmentRightExists(primID) ? vertex(vid+2) : Vec3ff(inf);
    }

#if defined(EMBREE_SYCL_SUPPORT) && defined(__SYCL_DEVICE_ONLY__)
    template<int M>
    __forceinline void vgather(Vec4vf<M>& p0, Vec4vf<M>& p1, Vec4vf<M>& p2, Vec4vf<M>& p3, const vuint<M>& primID, const vuint<M>& vid) const
    {
      p0 = vertex(vid.v+0);
      p1 = vertex(vid.v+1);
      vbool<M> left  = segmentLeftExists (primID.v);
      vbool<M> right = segmentRightExists(primID.v);
      vuint<M> i2 = select(left, vid-1,vid+0);
      vuint<M> i3 = select(right,vid+2,vid+1);
      p2 = vertex(i2.v);
      p3 = vertex(i3.v);
      p2 =  select(left, p2,Vec4vf<M>(inf));
      p3 =  select(right,p3,Vec4vf<M>(inf));
    }
#endif

     /*! gathers the curve starting with i'th vertex of itime'th timestep */
    __forceinline void gather(Vec3ff& p0, Vec3ff& p1, Vec3ff& p2, Vec3ff& p3, unsigned int primID, size_t vid, size_t itime) const
    {
      p0 = vertex(vid+0,itime);
      p1 = vertex(vid+1,itime);
      p2 = segmentLeftExists (primID) ? vertex(vid-1,itime) : Vec3ff(inf);
      p3 = segmentRightExists(primID) ? vertex(vid+2,itime) : Vec3ff(inf);
    }

#if defined(EMBREE_SYCL_SUPPORT) && defined(__SYCL_DEVICE_ONLY__)
    template<int M>
    __forceinline void vgather(Vec4vf<M>& p0, Vec4vf<M>& p1, Vec4vf<M>& p2, Vec4vf<M>& p3, const vuint<M>& primID, const vuint<M>& vid, const vint<M>& itime) const
    {
      p0 = vertex(vid.v+0, itime.v);
      p1 = vertex(vid.v+1, itime.v);
      vbool<M> left  = segmentLeftExists (primID.v);
      vbool<M> right = segmentRightExists(primID.v);
      vuint<M> i2 = select(left, vid-1,vid+0);
      vuint<M> i3 = select(right,vid+2,vid+1);
      p2 = vertex(i2.v, itime.v);
      p3 = vertex(i3.v, itime.v);
      p2 =  select(left, p2,Vec4vf<M>(inf));
      p3 =  select(right,p3,Vec4vf<M>(inf));
    }
#endif
    
     /*! loads curve vertices for specified time */
    __forceinline void gather(Vec3ff& p0, Vec3ff& p1, Vec3ff& p2, Vec3ff& p3, unsigned int primID, size_t vid, float time) const
    {
      float ftime;
      const size_t itime = timeSegment(time, ftime);

      const float t0 = 1.0f - ftime;
      const float t1 = ftime;
      Vec3ff a0,a1,a2,a3; gather(a0,a1,a2,a3,primID,vid,itime);
      Vec3ff b0,b1,b2,b3; gather(b0,b1,b2,b3,primID,vid,itime+1);
      p0 = madd(Vec3ff(t0),a0,t1*b0);
      p1 = madd(Vec3ff(t0),a1,t1*b1);
      p2 = madd(Vec3ff(t0),a2,t1*b2);
      p3 = madd(Vec3ff(t0),a3,t1*b3);
    }

    /*! loads curve vertices for specified time for mblur and non-mblur geometry */
    __forceinline void gather_safe(Vec3ff& p0, Vec3ff& p1, Vec3ff& p2, Vec3ff& p3, unsigned int primID, size_t vid, float time) const
    {
      if (hasMotionBlur()) gather(p0,p1,p2,p3,primID,vid,time);
      else                 gather(p0,p1,p2,p3,primID,vid);
    }

#if defined(EMBREE_SYCL_SUPPORT) && defined(__SYCL_DEVICE_ONLY__)
    template<int M>
    __forceinline void vgather(Vec4vf<M>& p0, Vec4vf<M>& p1, Vec4vf<M>& p2, Vec4vf<M>& p3, const vuint<M>& primID, const vuint<M>& vid, const vfloat<M>& time) const
    {
      vfloat<M> ftime;
      const vint<M> itime = timeSegment<M>(time, ftime);

      const vfloat<M> t0 = 1.0f - ftime;
      const vfloat<M> t1 = ftime;
      Vec4vf<M> a0,a1,a2,a3; vgather<M>(a0,a1,a2,a3,primID,vid,itime);
      Vec4vf<M> b0,b1,b2,b3; vgather<M>(b0,b1,b2,b3,primID,vid,itime+1);
      p0 = madd(Vec4vf<M>(t0),a0,t1*b0);
      p1 = madd(Vec4vf<M>(t0),a1,t1*b1);
      p2 = madd(Vec4vf<M>(t0),a2,t1*b2);
      p3 = madd(Vec4vf<M>(t0),a3,t1*b3);
    }
#endif
    
    /*! calculates bounding box of i'th line segment */
    __forceinline BBox3fa bounds(const Vec3ff& v0, const Vec3ff& v1) const
    {
      const BBox3ff b = merge(BBox3ff(v0),BBox3ff(v1));
      return enlarge((BBox3fa)b,maxRadiusScale*Vec3fa(max(v0.w,v1.w)));
    }

    /*! calculates bounding box of i'th line segment */
    __forceinline BBox3fa bounds(size_t i) const
    {
      const unsigned int index = segment(i);
      const Vec3ff v0 = vertex(index+0);
      const Vec3ff v1 = vertex(index+1);
      return bounds(v0,v1);
    }

    /*! calculates bounding box of i'th line segment for the itime'th time step */
    __forceinline BBox3fa bounds(size_t i, size_t itime) const
    {
      const unsigned int index = segment(i);
      const Vec3ff v0 = vertex(index+0,itime);
      const Vec3ff v1 = vertex(index+1,itime);
      return bounds(v0,v1);
    }

    /*! calculates bounding box of i'th line segment */
    __forceinline BBox3fa bounds(const LinearSpace3fa& space, size_t i) const
    {
      const unsigned int index = segment(i);
      const Vec3ff v0 = vertex(index+0);
      const Vec3ff v1 = vertex(index+1);
      const Vec3ff w0(xfmVector(space,(Vec3fa)v0),v0.w);
      const Vec3ff w1(xfmVector(space,(Vec3fa)v1),v1.w);
      return bounds(w0,w1);
    }

    /*! calculates bounding box of i'th line segment for the itime'th time step */
    __forceinline BBox3fa bounds(const LinearSpace3fa& space, size_t i, size_t itime) const
    {
      const unsigned int index = segment(i);
      const Vec3ff v0 = vertex(index+0,itime);
      const Vec3ff v1 = vertex(index+1,itime);
      const Vec3ff w0(xfmVector(space,(Vec3fa)v0),v0.w);
      const Vec3ff w1(xfmVector(space,(Vec3fa)v1),v1.w);
      return bounds(w0,w1);
    }

    /*! calculates bounding box of i'th segment */
    __forceinline BBox3fa bounds(const Vec3fa& ofs, const float scale, const float r_scale0, const LinearSpace3fa& space, size_t i, size_t itime = 0) const
    {
      const float r_scale = r_scale0*scale;
      const unsigned int index = segment(i);
      const Vec3ff v0 = vertex(index+0,itime);
      const Vec3ff v1 = vertex(index+1,itime);
      const Vec3ff w0(xfmVector(space,(v0-ofs)*Vec3fa(scale)),maxRadiusScale*v0.w*r_scale);
      const Vec3ff w1(xfmVector(space,(v1-ofs)*Vec3fa(scale)),maxRadiusScale*v1.w*r_scale);
      return bounds(w0,w1);
    }     

    /*! check if the i'th primitive is valid at the itime'th timestep */
    __forceinline bool valid(size_t i, size_t itime) const {
      return valid(i, make_range(itime, itime));
    }

    /*! check if the i'th primitive is valid between the specified time range */
    __forceinline bool valid(size_t i, const range<size_t>& itime_range) const
    {
      const unsigned int index = segment(i);
      if (index+1 >= numVertices()) return false;

#if !defined(__SYCL_DEVICE_ONLY__)

      for (size_t itime = itime_range.begin(); itime <= itime_range.end(); itime++)
      {
        const Vec3ff v0 = vertex(index+0,itime); if (unlikely(!isvalid4(v0))) return false;
        const Vec3ff v1 = vertex(index+1,itime); if (unlikely(!isvalid4(v1))) return false;
        if (min(v0.w,v1.w) < 0.0f) return false;
      }
#endif
      
      return true;
    }

    /*! calculates the linear bounds of the i'th primitive at the itimeGlobal'th time segment */
    __forceinline LBBox3fa linearBounds(size_t i, size_t itime) const {
      return LBBox3fa(bounds(i,itime+0),bounds(i,itime+1));
    }

    /*! calculates the build bounds of the i'th primitive, if it's valid */
    __forceinline bool buildBounds(size_t i, BBox3fa* bbox) const
    {
      if (!valid(i,0)) return false;
      *bbox = bounds(i); 
      return true;
    }

    /*! calculates the build bounds of the i'th primitive at the itime'th time segment, if it's valid */
    __forceinline bool buildBounds(size_t i, size_t itime, BBox3fa& bbox) const
    {
      if (!valid(i,itime+0) || !valid(i,itime+1)) return false;
      bbox = bounds(i,itime);  // use bounds of first time step in builder
      return true;
    }

    /*! calculates the linear bounds of the i'th primitive for the specified time range */
    __forceinline LBBox3fa linearBounds(size_t primID, const BBox1f& dt) const {
      return LBBox3fa([&] (size_t itime) { return bounds(primID, itime); }, dt, time_range, fnumTimeSegments);
    }

    /*! calculates the linear bounds of the i'th primitive for the specified time range */
    __forceinline LBBox3fa linearBounds(const LinearSpace3fa& space, size_t primID, const BBox1f& dt) const {
      return LBBox3fa([&] (size_t itime) { return bounds(space, primID, itime); }, dt, time_range, fnumTimeSegments);
    }

    /*! calculates the linear bounds of the i'th primitive for the specified time range */
    __forceinline LBBox3fa linearBounds(const Vec3fa& ofs, const float scale, const float r_scale0, const LinearSpace3fa& space, size_t primID, const BBox1f& dt) const {
      return LBBox3fa([&] (size_t itime) { return bounds(ofs, scale, r_scale0, space, primID, itime); }, dt, this->time_range, fnumTimeSegments);
    }
    
    /*! calculates the linear bounds of the i'th primitive for the specified time range */
    __forceinline bool linearBounds(size_t i, const BBox1f& time_range, LBBox3fa& bbox) const
    {
      if (!valid(i, timeSegmentRange(time_range))) return false;
      bbox = linearBounds(i, time_range);
      return true;
    }

    /*! get fast access to first vertex buffer */
    __forceinline float * getCompactVertexArray () const {
      return (float*) vertices0.getPtr();
    }

  public:
    BufferView<unsigned int> segments;      //!< array of line segment indices
    BufferView<Vec3ff> vertices0;           //!< fast access to first vertex buffer
    BufferView<Vec3fa> normals0;            //!< fast access to first normal buffer
    BufferView<char> flags;                 //!< start, end flag per segment
    Device::vector<BufferView<Vec3ff>> vertices = device;    //!< vertex array for each timestep
    Device::vector<BufferView<Vec3fa>> normals = device;     //!< normal array for each timestep
    Device::vector<BufferView<char>> vertexAttribs = device; //!< user buffers
    int tessellationRate;                   //!< tessellation rate for bezier curve
    float maxRadiusScale = 1.0;             //!< maximal min-width scaling of curve radii
  };

  namespace isa
  {
    struct LineSegmentsISA : public LineSegments
    {
      LineSegmentsISA (Device* device, Geometry::GType gtype)
        : LineSegments(device,gtype) {}

      LinearSpace3fa computeAlignedSpace(const size_t primID) const
      {
        const Vec3fa dir = normalize(computeDirection(primID));
        if (is_finite(dir)) return frame(dir);
        else return LinearSpace3fa(one);
      }

      LinearSpace3fa computeAlignedSpaceMB(const size_t primID, const BBox1f time_range) const
      {
        Vec3fa axisz(0,0,1);
        Vec3fa axisy(0,1,0);

        const range<int> tbounds = this->timeSegmentRange(time_range);
        if (tbounds.size() == 0) return frame(axisz);
        
        const size_t itime = (tbounds.begin()+tbounds.end())/2;

        const Vec3fa dir = normalize(computeDirection(primID,itime));
        if (is_finite(dir)) return frame(dir);
        else return LinearSpace3fa(one);
      }     

      Vec3fa computeDirection(unsigned int primID) const
      {
        const unsigned vtxID = segment(primID);
        const Vec3fa v0 = vertex(vtxID+0);
        const Vec3fa v1 = vertex(vtxID+1);
        return v1-v0;
      }

      Vec3fa computeDirection(unsigned int primID, size_t time) const
      {
        const unsigned vtxID = segment(primID);
        const Vec3fa v0 = vertex(vtxID+0,time);
        const Vec3fa v1 = vertex(vtxID+1,time);
        return v1-v0;
      }

      PrimInfo createPrimRefArray(PrimRef* prims, const range<size_t>& r, size_t k, unsigned int geomID) const
      {
        PrimInfo pinfo(empty);
        for (size_t j=r.begin(); j<r.end(); j++)
        {
          BBox3fa bounds = empty;
          if (!buildBounds(j,&bounds)) continue;
          const PrimRef prim(bounds,geomID,unsigned(j));
          pinfo.add_center2(prim);
          prims[k++] = prim;
        }
        return pinfo;
      }

      PrimInfo createPrimRefArrayMB(mvector<PrimRef>& prims, size_t itime, const range<size_t>& r, size_t k, unsigned int geomID) const
      {
        PrimInfo pinfo(empty);
        for (size_t j=r.begin(); j<r.end(); j++)
        {
          BBox3fa bounds = empty;
          if (!buildBounds(j,itime,bounds)) continue;
          const PrimRef prim(bounds,geomID,unsigned(j));
          pinfo.add_center2(prim);
          prims[k++] = prim;
        }
        return pinfo;
      }

      PrimInfo createPrimRefArrayMB(PrimRef* prims, const BBox1f& time_range, const range<size_t>& r, size_t k, unsigned int geomID) const
      {
        PrimInfo pinfo(empty);
        const BBox1f t0t1 = BBox1f::intersect(getTimeRange(), time_range);
        if (t0t1.empty()) return pinfo;
        
        for (size_t j = r.begin(); j < r.end(); j++) {
          LBBox3fa lbounds = empty;
          if (!linearBounds(j, t0t1, lbounds))
            continue;
          const PrimRef prim(lbounds.bounds(), geomID, unsigned(j));
          pinfo.add_center2(prim);
          prims[k++] = prim;
        }
        return pinfo;
      }

      PrimInfoMB createPrimRefMBArray(mvector<PrimRefMB>& prims, const BBox1f& t0t1, const range<size_t>& r, size_t k, unsigned int geomID) const
      {
        PrimInfoMB pinfo(empty);
        for (size_t j=r.begin(); j<r.end(); j++)
        {
          if (!valid(j, timeSegmentRange(t0t1))) continue;
          const PrimRefMB prim(linearBounds(j,t0t1),this->numTimeSegments(),this->time_range,this->numTimeSegments(),geomID,unsigned(j));
          pinfo.add_primref(prim);
          prims[k++] = prim;
        }
        return pinfo;
      }

      BBox3fa vbounds(size_t i) const {
        return bounds(i);
      }
      
      BBox3fa vbounds(const LinearSpace3fa& space, size_t i) const {
        return bounds(space,i);
      }

       BBox3fa vbounds(const Vec3fa& ofs, const float scale, const float r_scale0, const LinearSpace3fa& space, size_t i, size_t itime = 0) const {
        return bounds(ofs,scale,r_scale0,space,i,itime);
      }

      LBBox3fa vlinearBounds(size_t primID, const BBox1f& time_range) const {
        return linearBounds(primID,time_range);
      }
      
      LBBox3fa vlinearBounds(const LinearSpace3fa& space, size_t primID, const BBox1f& time_range) const {
        return linearBounds(space,primID,time_range);
      }

       LBBox3fa vlinearBounds(const Vec3fa& ofs, const float scale, const float r_scale0, const LinearSpace3fa& space, size_t primID, const BBox1f& time_range) const {
        return linearBounds(ofs,scale,r_scale0,space,primID,time_range);
      }
    };
  }

  DECLARE_ISA_FUNCTION(LineSegments*, createLineSegments, Device* COMMA Geometry::GType);
}
