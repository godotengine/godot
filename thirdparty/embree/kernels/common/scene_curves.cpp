// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "scene_curves.h"
#include "scene.h"

namespace embree
{
#if defined(EMBREE_LOWEST_ISA)

  void CurveGeometry::resizeBuffers(unsigned int numSteps)
  {
     vertices.resize(numSteps);

    if (getCurveType() == GTY_SUBTYPE_ORIENTED_CURVE)
    {
      normals.resize(numSteps);

      if (getCurveBasis() == GTY_BASIS_HERMITE)
        dnormals.resize(numSteps);
    }
    if (getCurveBasis() == GTY_BASIS_HERMITE)
      tangents.resize(numSteps);
  }

  CurveGeometry::CurveGeometry (Device* device, GType gtype)
    : Geometry(device,gtype,0,1), tessellationRate(4)
  {
    resizeBuffers(numTimeSteps);
  }
  
  void CurveGeometry::setMask (unsigned mask) 
  {
    this->mask = mask; 
    Geometry::update();
  }

  void CurveGeometry::setNumTimeSteps (unsigned int numTimeSteps)
  {
    resizeBuffers(numTimeSteps);
    Geometry::setNumTimeSteps(numTimeSteps);
  }
  
  void CurveGeometry::setVertexAttributeCount (unsigned int N)
  {
    vertexAttribs.resize(N);
    Geometry::update();
  }

  void CurveGeometry::setBuffer(RTCBufferType type, unsigned int slot, RTCFormat format, const Ref<Buffer>& buffer, size_t offset, size_t stride, unsigned int num)
  { 
    /* verify that all accesses are 4 bytes aligned */
    if ((type != RTC_BUFFER_TYPE_FLAGS) && (((size_t(buffer->getPtr()) + offset) & 0x3) || (stride & 0x3)))
      throw_RTCError(RTC_ERROR_INVALID_OPERATION, "data must be 4 bytes aligned");

    if (type == RTC_BUFFER_TYPE_VERTEX)
    {
      if (format != RTC_FORMAT_FLOAT4)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid vertex buffer format");

      if (slot >= vertices.size())
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid vertex buffer slot");
      
      vertices[slot].set(buffer, offset, stride, num, format);
      vertices[slot].checkPadding16();
    }
    else if (type == RTC_BUFFER_TYPE_NORMAL)
    {
      if (getCurveType() != GTY_SUBTYPE_ORIENTED_CURVE)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "unknown buffer type");
        
      if (format != RTC_FORMAT_FLOAT3)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid normal buffer format");

      if (slot >= normals.size())
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid normal buffer slot");
      
      normals[slot].set(buffer, offset, stride, num, format);
      normals[slot].checkPadding16();
    }
    else if (type == RTC_BUFFER_TYPE_TANGENT)
    {
      if (getCurveBasis() != GTY_BASIS_HERMITE)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "unknown buffer type");
        
      if (format != RTC_FORMAT_FLOAT4)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid tangent buffer format");

      if (slot >= tangents.size())
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid tangent buffer slot");
      
      tangents[slot].set(buffer, offset, stride, num, format);
      tangents[slot].checkPadding16();
    }
    else if (type == RTC_BUFFER_TYPE_NORMAL_DERIVATIVE)
    {
      if (getCurveType() != GTY_SUBTYPE_ORIENTED_CURVE)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "unknown buffer type");
        
      if (format != RTC_FORMAT_FLOAT3)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid normal derivative buffer format");

      if (slot >= dnormals.size())
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid normal derivative buffer slot");
      
      dnormals[slot].set(buffer, offset, stride, num, format);
      dnormals[slot].checkPadding16();
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE)
    {
      if (format < RTC_FORMAT_FLOAT || format > RTC_FORMAT_FLOAT16)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid vertex attribute buffer format");

      if (slot >= vertexAttribs.size())
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid vertex attribute buffer slot");
      
      vertexAttribs[slot].set(buffer, offset, stride, num, format);
      vertexAttribs[slot].checkPadding16();
    }
    else if (type == RTC_BUFFER_TYPE_INDEX)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      if (format != RTC_FORMAT_UINT)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid index buffer format");

      curves.set(buffer, offset, stride, num, format);
      setNumPrimitives(num);
    }
    else if (type == RTC_BUFFER_TYPE_FLAGS)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      if (format != RTC_FORMAT_UCHAR)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid flag buffer format");

      flags.set(buffer, offset, stride, num, format);
    }
    else 
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "unknown buffer type");
  }

  void* CurveGeometry::getBuffer(RTCBufferType type, unsigned int slot)
  {
    if (type == RTC_BUFFER_TYPE_INDEX)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return curves.getPtr();
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX)
    {
      if (slot >= vertices.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return vertices[slot].getPtr();
    }
    else if (type == RTC_BUFFER_TYPE_NORMAL)
    {
      if (slot >= normals.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return normals[slot].getPtr();
    }
    else if (type == RTC_BUFFER_TYPE_TANGENT)
    {
      if (slot >= tangents.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return tangents[slot].getPtr();
    }
    else if (type == RTC_BUFFER_TYPE_NORMAL_DERIVATIVE)
    {
      if (slot >= dnormals.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return dnormals[slot].getPtr();
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE)
    {
      if (slot >= vertexAttribs.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return vertexAttribs[slot].getPtr();
    }
    else if (type == RTC_BUFFER_TYPE_FLAGS)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return flags.getPtr();
    }
    else
    {
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "unknown buffer type");
      return nullptr;
    }
  }

  void CurveGeometry::updateBuffer(RTCBufferType type, unsigned int slot)
  {
    if (type == RTC_BUFFER_TYPE_INDEX)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      curves.setModified();
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX)
    {
      if (slot >= vertices.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      vertices[slot].setModified();
    }
    else if (type == RTC_BUFFER_TYPE_NORMAL)
    {
      if (slot >= normals.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      normals[slot].setModified();
    }
    else if (type == RTC_BUFFER_TYPE_TANGENT)
    {
      if (slot >= tangents.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      tangents[slot].setModified();
    }
    else if (type == RTC_BUFFER_TYPE_NORMAL_DERIVATIVE)
    {
      if (slot >= dnormals.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      dnormals[slot].setModified();
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE)
    {
      if (slot >= vertexAttribs.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      vertexAttribs[slot].setModified();
    }
    else if (type == RTC_BUFFER_TYPE_FLAGS) 
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      flags.setModified();
    }
    else
    {
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "unknown buffer type");
    }

    Geometry::update();
  }

  void CurveGeometry::setTessellationRate(float N) {
    tessellationRate = clamp((int)N,1,16);
  }

  void CurveGeometry::setMaxRadiusScale(float s) {
    maxRadiusScale = s;
  }

  void CurveGeometry::addElementsToCount (GeometryCounts & counts) const 
  {
    if (numTimeSteps == 1) counts.numBezierCurves += numPrimitives; 
    else                   counts.numMBBezierCurves += numPrimitives;
  }

  bool CurveGeometry::verify () 
  {
    /*! verify consistent size of vertex arrays */
    if (vertices.size() == 0)
      return false;
    
    for (const auto& buffer : vertices)
      if (vertices[0].size() != buffer.size())
        return false;

    if (getCurveType() == GTY_SUBTYPE_ORIENTED_CURVE)
    {
      if (!normals.size())
        return false;
        
      for (const auto& buffer : normals)
        if (vertices[0].size() != buffer.size())
          return false;

      if (getCurveBasis() == GTY_BASIS_HERMITE)
      {
        if (!dnormals.size())
          return false;
        
        for (const auto& buffer : dnormals)
          if (vertices[0].size() != buffer.size())
            return false;
      }
      else
      {
        if (dnormals.size())
          return false;
      }
    }
    else
    {
      if (normals.size())
        return false;
    }

    if (getCurveBasis() == GTY_BASIS_HERMITE)
    {
      if (!tangents.size())
        return false;
      
      for (const auto& buffer : tangents)
        if (vertices[0].size() != buffer.size())
          return false;
    }
    else
    {
      if (tangents.size())
        return false;
    }
    
    /*! verify indices */
    if (getCurveBasis() == GTY_BASIS_HERMITE)
    {
      for (unsigned int i=0; i<size(); i++) {
        if (curves[i]+1 >= numVertices()) return false;
      }
    }
    else
    {
      for (unsigned int i=0; i<numPrimitives; i++) {
        if (curves[i]+3 >= numVertices()) return false;
      }
    }
    
    /*! verify vertices */
    for (const auto& buffer : vertices) {
      for (size_t i=0; i<buffer.size(); i++) {
	if (!isvalid(buffer[i].x)) return false;
        if (!isvalid(buffer[i].y)) return false;
        if (!isvalid(buffer[i].z)) return false;
        if (!isvalid(buffer[i].w)) return false;
      }
    }
    return true;
  }

  void CurveGeometry::commit()
  {
    /* verify that stride of all time steps are identical */
    for (const auto& buffer : vertices)
      if (buffer.getStride() != vertices[0].getStride())
        throw_RTCError(RTC_ERROR_INVALID_OPERATION,"stride of vertex buffers have to be identical for each time step");

    for (const auto& buffer : normals)
      if (buffer.getStride() != normals[0].getStride())
        throw_RTCError(RTC_ERROR_INVALID_OPERATION,"stride of normal buffers have to be identical for each time step");

    for (const auto& buffer : tangents)
      if (buffer.getStride() != tangents[0].getStride())
        throw_RTCError(RTC_ERROR_INVALID_OPERATION,"stride of tangent buffers have to be identical for each time step");

    for (const auto& buffer : dnormals)
      if (buffer.getStride() != dnormals[0].getStride())
        throw_RTCError(RTC_ERROR_INVALID_OPERATION,"stride of normal derivative buffers have to be identical for each time step");
    
    vertices0 = vertices[0];
    if (getCurveType() == GTY_SUBTYPE_ORIENTED_CURVE)
    {
      normals0 = normals[0];
      if (getCurveBasis() == GTY_BASIS_HERMITE)
        dnormals0 = dnormals[0];
    }
    if (getCurveBasis() == GTY_BASIS_HERMITE)
      tangents0 = tangents[0];

    Geometry::commit();
  }

#endif

  namespace isa
  {
    __forceinline BBox3fa enlarge_bounds(const BBox3fa& bounds)
    {
      const float size = reduce_max(max(abs(bounds.lower),abs(bounds.upper)));
      return enlarge(bounds,Vec3fa(4.0f*float(ulp)*size));
    }   
    
    template<Geometry::GType ctype, template<template<typename Ty> class Curve> class CurveInterfaceT, template<typename Ty> class Curve>
      struct CurveGeometryISA : public CurveInterfaceT<Curve>
    {
      typedef Curve<Vec3ff> Curve3ff;
      typedef Curve<Vec3fa> Curve3fa;
      
      using CurveInterfaceT<Curve>::getCurveScaledRadius;
      using CurveInterfaceT<Curve>::getOrientedCurveScaledRadius;
      using CurveInterfaceT<Curve>::numTimeSteps;
      using CurveInterfaceT<Curve>::fnumTimeSegments;
      using CurveInterfaceT<Curve>::numTimeSegments;
      using CurveInterfaceT<Curve>::tessellationRate;

      using CurveInterfaceT<Curve>::valid;
      using CurveInterfaceT<Curve>::numVertices;
      using CurveInterfaceT<Curve>::vertexAttribs;
      using CurveInterfaceT<Curve>::vertices;
      using CurveInterfaceT<Curve>::curves;
      using CurveInterfaceT<Curve>::curve;
      using CurveInterfaceT<Curve>::radius;
      using CurveInterfaceT<Curve>::vertex;
      using CurveInterfaceT<Curve>::normal;
      
      CurveGeometryISA (Device* device, Geometry::GType gtype)
        : CurveInterfaceT<Curve>(device,gtype) {}

      LinearSpace3fa computeAlignedSpace(const size_t primID) const
      {
        Vec3fa axisz(0,0,1);
        Vec3fa axisy(0,1,0);
        
        const Curve3ff curve = getCurveScaledRadius(primID);
        const Vec3fa p0 = curve.begin();
        const Vec3fa p3 = curve.end();
        const Vec3fa d0 = curve.eval_du(0.0f);
        //const Vec3fa d1 = curve.eval_du(1.0f);
        const Vec3fa axisz_ = normalize(p3 - p0);
        const Vec3fa axisy_ = cross(axisz_,d0);
        if (sqr_length(p3-p0) > 1E-18f) {
          axisz = axisz_;
          axisy = axisy_;
        }
        
        if (sqr_length(axisy) > 1E-18) {
          axisy = normalize(axisy);
          Vec3fa axisx = normalize(cross(axisy,axisz));
          return LinearSpace3fa(axisx,axisy,axisz);
        }
        return frame(axisz);
      }

      LinearSpace3fa computeAlignedSpaceMB(const size_t primID, const BBox1f time_range) const
      {
        Vec3fa axisz(0,0,1);
        Vec3fa axisy(0,1,0);

        const range<int> tbounds = this->timeSegmentRange(time_range);
        if (tbounds.size() == 0) return frame(axisz);
        
        const size_t t = (tbounds.begin()+tbounds.end())/2;
        const Curve3ff curve = getCurveScaledRadius(primID,t);
        const Vec3fa p0 = curve.begin();
        const Vec3fa p3 = curve.end();
        const Vec3fa d0 = curve.eval_du(0.0f);
        //const Vec3fa d1 = curve.eval_du(1.0f);
        const Vec3fa axisz_ = normalize(p3 - p0);
        const Vec3fa axisy_ = cross(axisz_,d0);
        if (sqr_length(p3-p0) > 1E-18f) {
          axisz = axisz_;
          axisy = axisy_;
        }
        
        if (sqr_length(axisy) > 1E-18) {
          axisy = normalize(axisy);
          Vec3fa axisx = normalize(cross(axisy,axisz));
          return LinearSpace3fa(axisx,axisy,axisz);
        }
        return frame(axisz);
      }
      
      Vec3fa computeDirection(unsigned int primID) const
      {
        const Curve3ff c = getCurveScaledRadius(primID);
        const Vec3fa p0 = c.begin();
        const Vec3fa p3 = c.end();
        const Vec3fa axis1 = p3 - p0;
        return axis1;
      }

      Vec3fa computeDirection(unsigned int primID, size_t time) const
      {
        const Curve3ff c = getCurveScaledRadius(primID,time);
        const Vec3fa p0 = c.begin();
        const Vec3fa p3 = c.end();
        const Vec3fa axis1 = p3 - p0;
        return axis1;
      }

      /*! calculates bounding box of i'th bezier curve */
      __forceinline BBox3fa bounds(size_t i, size_t itime = 0) const
      {
        switch (ctype) {
        case Geometry::GTY_SUBTYPE_FLAT_CURVE: return enlarge_bounds(getCurveScaledRadius(i,itime).accurateFlatBounds(tessellationRate));
        case Geometry::GTY_SUBTYPE_ROUND_CURVE: return enlarge_bounds(getCurveScaledRadius(i,itime).accurateRoundBounds());
        case Geometry::GTY_SUBTYPE_ORIENTED_CURVE: return enlarge_bounds(getOrientedCurveScaledRadius(i,itime).accurateBounds());
        default: return empty;
        }
      }
      
      /*! calculates bounding box of i'th bezier curve */
      __forceinline BBox3fa bounds(const LinearSpace3fa& space, size_t i, size_t itime = 0) const
      {
        switch (ctype) {
        case Geometry::GTY_SUBTYPE_FLAT_CURVE: return enlarge_bounds(getCurveScaledRadius(space,i,itime).accurateFlatBounds(tessellationRate));
        case Geometry::GTY_SUBTYPE_ROUND_CURVE: return enlarge_bounds(getCurveScaledRadius(space,i,itime).accurateRoundBounds());
        case Geometry::GTY_SUBTYPE_ORIENTED_CURVE: return enlarge_bounds(getOrientedCurveScaledRadius(space,i,itime).accurateBounds());
        default: return empty;
        }
      }
      
      /*! calculates bounding box of i'th bezier curve */
      __forceinline BBox3fa bounds(const Vec3fa& ofs, const float scale, const float r_scale0, const LinearSpace3fa& space, size_t i, size_t itime = 0) const
      {
        switch (ctype) {
        case Geometry::GTY_SUBTYPE_FLAT_CURVE: return enlarge_bounds(getCurveScaledRadius(ofs,scale,r_scale0,space,i,itime).accurateFlatBounds(tessellationRate));
        case Geometry::GTY_SUBTYPE_ROUND_CURVE: return enlarge_bounds(getCurveScaledRadius(ofs,scale,r_scale0,space,i,itime).accurateRoundBounds());
        case Geometry::GTY_SUBTYPE_ORIENTED_CURVE: return enlarge_bounds(getOrientedCurveScaledRadius(ofs,scale,space,i,itime).accurateBounds());
        default: return empty;
        }
      }

      /*! calculates the linear bounds of the i'th primitive for the specified time range */
      __forceinline LBBox3fa linearBounds(size_t primID, const BBox1f& dt) const {
        return LBBox3fa([&] (size_t itime) { return bounds(primID, itime); }, dt, this->time_range, fnumTimeSegments);
      }
      
      /*! calculates the linear bounds of the i'th primitive for the specified time range */
      __forceinline LBBox3fa linearBounds(const LinearSpace3fa& space, size_t primID, const BBox1f& dt) const {
        return LBBox3fa([&] (size_t itime) { return bounds(space, primID, itime); }, dt, this->time_range, fnumTimeSegments);
      }
      
      /*! calculates the linear bounds of the i'th primitive for the specified time range */
      __forceinline LBBox3fa linearBounds(const Vec3fa& ofs, const float scale, const float r_scale0, const LinearSpace3fa& space, size_t primID, const BBox1f& dt) const {
        return LBBox3fa([&] (size_t itime) { return bounds(ofs, scale, r_scale0, space, primID, itime); }, dt, this->time_range, fnumTimeSegments);
      }
      
      PrimInfo createPrimRefArray(PrimRef* prims, const range<size_t>& r, size_t k, unsigned int geomID) const
      {
        PrimInfo pinfo(empty);
        for (size_t j=r.begin(); j<r.end(); j++)
        {
          if (!valid(ctype, j, make_range<size_t>(0, numTimeSegments()))) continue;
          const BBox3fa box = bounds(j);
          const PrimRef prim(box,geomID,unsigned(j));
          pinfo.add_center2(prim);
          prims[k++] = prim;
        }
        return pinfo;
      }

      PrimInfo createPrimRefArrayMB(PrimRef* prims, const BBox1f& time_range, const range<size_t>& r, size_t k, unsigned int geomID) const
      {
        PrimInfo pinfo(empty);
        const BBox1f t0t1 = BBox1f::intersect(this->time_range, time_range);
        if (t0t1.empty()) return pinfo;
        
        for (size_t j=r.begin(); j<r.end(); j++)
        {
          if (!valid(ctype, j, this->timeSegmentRange(t0t1))) continue;
          const LBBox3fa lbounds = linearBounds(j,t0t1);
          if (lbounds.bounds0.empty() || lbounds.bounds1.empty()) continue; // checks oriented curves with invalid normals which cause NaNs here
          const PrimRef prim(lbounds.bounds(),geomID,unsigned(j));
          pinfo.add_primref(prim);
          prims[k++] = prim;
        }
        return pinfo;
      }

      PrimInfoMB createPrimRefMBArray(mvector<PrimRefMB>& prims, const BBox1f& t0t1, const range<size_t>& r, size_t k, unsigned int geomID) const
      {
        PrimInfoMB pinfo(empty);
        for (size_t j=r.begin(); j<r.end(); j++)
        {
          if (!valid(ctype, j, this->timeSegmentRange(t0t1))) continue;
          const LBBox3fa lbox = linearBounds(j,t0t1);
          const PrimRefMB prim(lbox,this->numTimeSegments(),this->time_range,this->numTimeSegments(),geomID,unsigned(j));
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

    CurveGeometry* createCurves(Device* device, Geometry::GType gtype)
    {
      switch (gtype) {
      case Geometry::GTY_ROUND_BEZIER_CURVE: return new CurveGeometryISA<Geometry::GTY_SUBTYPE_ROUND_CURVE,CurveGeometryInterface,BezierCurveT>(device,gtype);
      case Geometry::GTY_FLAT_BEZIER_CURVE : return new CurveGeometryISA<Geometry::GTY_SUBTYPE_FLAT_CURVE,CurveGeometryInterface,BezierCurveT>(device,gtype);
      case Geometry::GTY_ORIENTED_BEZIER_CURVE : return new CurveGeometryISA<Geometry::GTY_SUBTYPE_ORIENTED_CURVE,CurveGeometryInterface,BezierCurveT>(device,gtype);
        
      case Geometry::GTY_ROUND_BSPLINE_CURVE: return new CurveGeometryISA<Geometry::GTY_SUBTYPE_ROUND_CURVE,CurveGeometryInterface,BSplineCurveT>(device,gtype);
      case Geometry::GTY_FLAT_BSPLINE_CURVE : return new CurveGeometryISA<Geometry::GTY_SUBTYPE_FLAT_CURVE,CurveGeometryInterface,BSplineCurveT>(device,gtype);
      case Geometry::GTY_ORIENTED_BSPLINE_CURVE : return new CurveGeometryISA<Geometry::GTY_SUBTYPE_ORIENTED_CURVE,CurveGeometryInterface,BSplineCurveT>(device,gtype);

      case Geometry::GTY_ROUND_HERMITE_CURVE: return new CurveGeometryISA<Geometry::GTY_SUBTYPE_ROUND_CURVE,HermiteCurveGeometryInterface,HermiteCurveT>(device,gtype);
      case Geometry::GTY_FLAT_HERMITE_CURVE : return new CurveGeometryISA<Geometry::GTY_SUBTYPE_FLAT_CURVE,HermiteCurveGeometryInterface,HermiteCurveT>(device,gtype);
      case Geometry::GTY_ORIENTED_HERMITE_CURVE : return new CurveGeometryISA<Geometry::GTY_SUBTYPE_ORIENTED_CURVE,HermiteCurveGeometryInterface,HermiteCurveT>(device,gtype);

      case Geometry::GTY_ROUND_CATMULL_ROM_CURVE: return new CurveGeometryISA<Geometry::GTY_SUBTYPE_ROUND_CURVE,CurveGeometryInterface,CatmullRomCurveT>(device,gtype);
      case Geometry::GTY_FLAT_CATMULL_ROM_CURVE : return new CurveGeometryISA<Geometry::GTY_SUBTYPE_FLAT_CURVE,CurveGeometryInterface,CatmullRomCurveT>(device,gtype);
      case Geometry::GTY_ORIENTED_CATMULL_ROM_CURVE : return new CurveGeometryISA<Geometry::GTY_SUBTYPE_ORIENTED_CURVE,CurveGeometryInterface,CatmullRomCurveT>(device,gtype);
     
      default: throw_RTCError(RTC_ERROR_INVALID_OPERATION,"invalid geometry type");
      }
    }
  }
}
