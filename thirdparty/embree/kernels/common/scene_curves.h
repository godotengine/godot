// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "default.h"
#include "geometry.h"
#include "buffer.h"

#include "../subdiv/bezier_curve.h"
#include "../subdiv/hermite_curve.h"
#include "../subdiv/bspline_curve.h"
#include "../subdiv/catmullrom_curve.h"
#include "../subdiv/linear_bezier_patch.h"

namespace embree
{
  /*! represents an array of bicubic bezier curves */
  struct CurveGeometry : public Geometry
  {
    /*! type of this geometry */
    static const Geometry::GTypeMask geom_type = Geometry::MTY_CURVE4;

  public:
    
    /*! bezier curve construction */
    CurveGeometry (Device* device, Geometry::GType gtype);
    
  public:
    void setMask(unsigned mask);
    void setNumTimeSteps (unsigned int numTimeSteps);
    void setVertexAttributeCount (unsigned int N);
    void setBuffer(RTCBufferType type, unsigned int slot, RTCFormat format, const Ref<Buffer>& buffer, size_t offset, size_t stride, unsigned int num);
    void* getBufferData(RTCBufferType type, unsigned int slot, BufferDataPointerType pointerType);
    void updateBuffer(RTCBufferType type, unsigned int slot);
    void commit();
    bool verify();
    void setTessellationRate(float N);
    void setMaxRadiusScale(float s);
    void addElementsToCount (GeometryCounts & counts) const;
    size_t getGeometryDataDeviceByteSize() const;
    void convertToDeviceRepresentation(size_t offset, char* data_host, char* data_device) const;

  public:
    
    /*! returns the number of vertices */
    __forceinline size_t numVertices() const {
      return vertices[0].size();
    }

    /*! returns the i'th curve */
    __forceinline const unsigned int& curve(size_t i) const {
      return curves[i];
    }

    /*! returns i'th vertex of the first time step */
    __forceinline Vec3ff vertex(size_t i) const {
      return vertices0[i];
    }

    /*! returns i'th normal of the first time step */
    __forceinline Vec3fa normal(size_t i) const {
      return normals0[i];
    }

    /*! returns i'th tangent of the first time step */
    __forceinline Vec3ff tangent(size_t i) const {
      return tangents0[i];
    }

    /*! returns i'th normal derivative of the first time step */
    __forceinline Vec3fa dnormal(size_t i) const {
      return dnormals0[i];
    }

    /*! returns i'th radius of the first time step */
    __forceinline float radius(size_t i) const {
      return vertices0[i].w;
    }

    /*! returns i'th vertex of itime'th timestep */
    __forceinline Vec3ff vertex(size_t i, size_t itime) const {
      return vertices[itime][i];
    }

    /*! returns i'th normal of itime'th timestep */
    __forceinline Vec3fa normal(size_t i, size_t itime) const {
      return normals[itime][i];
    }

    /*! returns i'th tangent of itime'th timestep */
    __forceinline Vec3ff tangent(size_t i, size_t itime) const {
      return tangents[itime][i];
    }

    /*! returns i'th normal derivative of itime'th timestep */
    __forceinline Vec3fa dnormal(size_t i, size_t itime) const {
      return dnormals[itime][i];
    }

    /*! returns i'th radius of itime'th timestep */
    __forceinline float radius(size_t i, size_t itime) const {
      return vertices[itime][i].w;
    }

    /*! gathers the curve starting with i'th vertex */
    __forceinline void gather(Vec3ff& p0, Vec3ff& p1, Vec3ff& p2, Vec3ff& p3, size_t i) const
    {
      p0 = vertex(i+0);
      p1 = vertex(i+1);
      p2 = vertex(i+2);
      p3 = vertex(i+3);
    }

    /*! gathers the curve starting with i'th vertex of itime'th timestep */
    __forceinline void gather(Vec3ff& p0, Vec3ff& p1, Vec3ff& p2, Vec3ff& p3, size_t i, size_t itime) const
    {
      p0 = vertex(i+0,itime);
      p1 = vertex(i+1,itime);
      p2 = vertex(i+2,itime);
      p3 = vertex(i+3,itime);
    }

    /*! gathers the curve normals starting with i'th vertex */
    __forceinline void gather_normals(Vec3fa& n0, Vec3fa& n1, Vec3fa& n2, Vec3fa& n3, size_t i) const
    {
      n0 = normal(i+0);
      n1 = normal(i+1);
      n2 = normal(i+2);
      n3 = normal(i+3);
    }

    /*! gathers the curve starting with i'th vertex */
    __forceinline void gather(Vec3ff& p0, Vec3ff& p1, Vec3ff& p2, Vec3ff& p3, Vec3fa& n0, Vec3fa& n1, Vec3fa& n2, Vec3fa& n3, size_t i) const
    {
      p0 = vertex(i+0);
      p1 = vertex(i+1);
      p2 = vertex(i+2);
      p3 = vertex(i+3);
      n0 = normal(i+0);
      n1 = normal(i+1);
      n2 = normal(i+2);
      n3 = normal(i+3);
    }

    /*! gathers the curve starting with i'th vertex of itime'th timestep */
    __forceinline void gather(Vec3ff& p0, Vec3ff& p1, Vec3ff& p2, Vec3ff& p3, Vec3fa& n0, Vec3fa& n1, Vec3fa& n2, Vec3fa& n3, size_t i, size_t itime) const
    {
      p0 = vertex(i+0,itime);
      p1 = vertex(i+1,itime);
      p2 = vertex(i+2,itime);
      p3 = vertex(i+3,itime);
      n0 = normal(i+0,itime);
      n1 = normal(i+1,itime);
      n2 = normal(i+2,itime);
      n3 = normal(i+3,itime);
    }

    /*! prefetches the curve starting with i'th vertex of itime'th timestep */
    __forceinline void prefetchL1_vertices(size_t i) const
    {
      prefetchL1(vertices0.getPtr(i)+0);
      prefetchL1(vertices0.getPtr(i)+64);
    }

    /*! prefetches the curve starting with i'th vertex of itime'th timestep */
    __forceinline void prefetchL2_vertices(size_t i) const
    {
      prefetchL2(vertices0.getPtr(i)+0);
      prefetchL2(vertices0.getPtr(i)+64);
    }  

    /*! loads curve vertices for specified time */
    __forceinline void gather(Vec3ff& p0, Vec3ff& p1, Vec3ff& p2, Vec3ff& p3, size_t i, float time) const
    {
      float ftime;
      const size_t itime = timeSegment(time, ftime);

      const float t0 = 1.0f - ftime;
      const float t1 = ftime;
      Vec3ff a0,a1,a2,a3;
      gather(a0,a1,a2,a3,i,itime);
      Vec3ff b0,b1,b2,b3;
      gather(b0,b1,b2,b3,i,itime+1);
      p0 = madd(Vec3ff(t0),a0,t1*b0);
      p1 = madd(Vec3ff(t0),a1,t1*b1);
      p2 = madd(Vec3ff(t0),a2,t1*b2);
      p3 = madd(Vec3ff(t0),a3,t1*b3);
    }

    /*! loads curve vertices for specified time */
    __forceinline void gather_safe(Vec3ff& p0, Vec3ff& p1, Vec3ff& p2, Vec3ff& p3, size_t i, float time) const
    {
      if (hasMotionBlur()) gather(p0,p1,p2,p3,i,time);
      else                 gather(p0,p1,p2,p3,i);
    }
    
    /*! loads curve vertices for specified time */
    __forceinline void gather(Vec3ff& p0, Vec3ff& p1, Vec3ff& p2, Vec3ff& p3, Vec3fa& n0, Vec3fa& n1, Vec3fa& n2, Vec3fa& n3, size_t i, float time) const
    {
      float ftime;
      const size_t itime = timeSegment(time, ftime);

      const float t0 = 1.0f - ftime;
      const float t1 = ftime;
      Vec3ff a0,a1,a2,a3; Vec3fa an0,an1,an2,an3;
      gather(a0,a1,a2,a3,an0,an1,an2,an3,i,itime);
      Vec3ff b0,b1,b2,b3; Vec3fa bn0,bn1,bn2,bn3;
      gather(b0,b1,b2,b3,bn0,bn1,bn2,bn3,i,itime+1);
      p0 = madd(Vec3ff(t0),a0,t1*b0);
      p1 = madd(Vec3ff(t0),a1,t1*b1);
      p2 = madd(Vec3ff(t0),a2,t1*b2);
      p3 = madd(Vec3ff(t0),a3,t1*b3);
      n0 = madd(Vec3ff(t0),an0,t1*bn0);
      n1 = madd(Vec3ff(t0),an1,t1*bn1);
      n2 = madd(Vec3ff(t0),an2,t1*bn2);
      n3 = madd(Vec3ff(t0),an3,t1*bn3);
    }

    /*! loads curve vertices for specified time for mblur and non-mblur case */
    __forceinline void gather_safe(Vec3ff& p0, Vec3ff& p1, Vec3ff& p2, Vec3ff& p3, Vec3fa& n0, Vec3fa& n1, Vec3fa& n2, Vec3fa& n3, size_t i, float time) const
    {
      if (hasMotionBlur()) gather(p0,p1,p2,p3,n0,n1,n2,n3,i,time);
      else                 gather(p0,p1,p2,p3,n0,n1,n2,n3,i);
    }

    template<typename SourceCurve3ff, typename SourceCurve3fa, typename TensorLinearCubicBezierSurface3fa>
    __forceinline TensorLinearCubicBezierSurface3fa getNormalOrientedCurve(RayQueryContext* context, const Vec3fa& ray_org, const unsigned int primID, const size_t itime) const
    {
      Vec3ff v0,v1,v2,v3; Vec3fa n0,n1,n2,n3;
      unsigned int vertexID = curve(primID);
      gather(v0,v1,v2,v3,n0,n1,n2,n3,vertexID,itime);
      SourceCurve3ff ccurve(v0,v1,v2,v3);
      SourceCurve3fa ncurve(n0,n1,n2,n3);
      ccurve = enlargeRadiusToMinWidth(context,this,ray_org,ccurve);
      return TensorLinearCubicBezierSurface3fa::fromCenterAndNormalCurve(ccurve,ncurve);
    }

    template<typename SourceCurve3ff, typename SourceCurve3fa, typename TensorLinearCubicBezierSurface3fa>
    __forceinline TensorLinearCubicBezierSurface3fa getNormalOrientedCurve(RayQueryContext* context, const Vec3fa& ray_org, const unsigned int primID, const float time) const
    {
      float ftime;
      const size_t itime = timeSegment(time, ftime);
      const TensorLinearCubicBezierSurface3fa curve0 = getNormalOrientedCurve<SourceCurve3ff, SourceCurve3fa, TensorLinearCubicBezierSurface3fa>(context,ray_org,primID,itime+0);
      const TensorLinearCubicBezierSurface3fa curve1 = getNormalOrientedCurve<SourceCurve3ff, SourceCurve3fa, TensorLinearCubicBezierSurface3fa>(context,ray_org,primID,itime+1);
      return clerp(curve0,curve1,ftime);
    }

    template<typename SourceCurve3ff, typename SourceCurve3fa, typename TensorLinearCubicBezierSurface3fa>
    __forceinline TensorLinearCubicBezierSurface3fa getNormalOrientedCurveSafe(RayQueryContext* context, const Vec3fa& ray_org, const unsigned int primID, const float time) const
    {
      float ftime = 0.0f;
      const size_t itime = hasMotionBlur() ? timeSegment(time, ftime) : 0;
      const TensorLinearCubicBezierSurface3fa curve0 = getNormalOrientedCurve<SourceCurve3ff, SourceCurve3fa, TensorLinearCubicBezierSurface3fa>(context,ray_org,primID,itime+0);
      if (hasMotionBlur()) {
        const TensorLinearCubicBezierSurface3fa curve1 = getNormalOrientedCurve<SourceCurve3ff, SourceCurve3fa, TensorLinearCubicBezierSurface3fa>(context,ray_org,primID,itime+1);
        return clerp(curve0,curve1,ftime);
      }
      return curve0;
    }

    /*! gathers the hermite curve starting with i'th vertex */
    __forceinline void gather_hermite(Vec3ff& p0, Vec3ff& t0, Vec3ff& p1, Vec3ff& t1, size_t i) const
    {
      p0 = vertex (i+0);
      p1 = vertex (i+1);
      t0 = tangent(i+0);
      t1 = tangent(i+1);
    }

    /*! gathers the hermite curve starting with i'th vertex of itime'th timestep */
    __forceinline void gather_hermite(Vec3ff& p0, Vec3ff& t0, Vec3ff& p1, Vec3ff& t1, size_t i, size_t itime) const
    {
      p0 = vertex (i+0,itime);
      p1 = vertex (i+1,itime);
      t0 = tangent(i+0,itime);
      t1 = tangent(i+1,itime);
    }

    /*! loads curve vertices for specified time */
    __forceinline void gather_hermite(Vec3ff& p0, Vec3ff& t0, Vec3ff& p1, Vec3ff& t1, size_t i, float time) const
    {
      float ftime;
      const size_t itime = timeSegment(time, ftime);
      const float f0 = 1.0f - ftime, f1 = ftime;
      Vec3ff ap0,at0,ap1,at1;
      gather_hermite(ap0,at0,ap1,at1,i,itime);
      Vec3ff bp0,bt0,bp1,bt1;
      gather_hermite(bp0,bt0,bp1,bt1,i,itime+1);
      p0 = madd(Vec3ff(f0),ap0,f1*bp0);
      t0 = madd(Vec3ff(f0),at0,f1*bt0);
      p1 = madd(Vec3ff(f0),ap1,f1*bp1);
      t1 = madd(Vec3ff(f0),at1,f1*bt1);
    }

    /*! loads curve vertices for specified time for mblur and non-mblur geometry */
    __forceinline void gather_hermite_safe(Vec3ff& p0, Vec3ff& t0, Vec3ff& p1, Vec3ff& t1, size_t i, float time) const
    {
      if (hasMotionBlur()) gather_hermite(p0,t0,p1,t1,i,time);
      else                 gather_hermite(p0,t0,p1,t1,i);
    }

    /*! gathers the hermite curve starting with i'th vertex */
    __forceinline void gather_hermite(Vec3ff& p0, Vec3ff& t0, Vec3fa& n0, Vec3fa& dn0, Vec3ff& p1, Vec3ff& t1, Vec3fa& n1, Vec3fa& dn1, size_t i) const
    {
      p0 = vertex (i+0);
      p1 = vertex (i+1);
      t0 = tangent(i+0);
      t1 = tangent(i+1);
      n0 = normal(i+0);
      n1 = normal(i+1);
      dn0 = dnormal(i+0);
      dn1 = dnormal(i+1);
    }

    /*! gathers the hermite curve starting with i'th vertex of itime'th timestep */
    __forceinline void gather_hermite(Vec3ff& p0, Vec3ff& t0, Vec3fa& n0, Vec3fa& dn0, Vec3ff& p1, Vec3ff& t1, Vec3fa& n1, Vec3fa& dn1, size_t i, size_t itime) const
    {
      p0 = vertex (i+0,itime);
      p1 = vertex (i+1,itime);
      t0 = tangent(i+0,itime);
      t1 = tangent(i+1,itime);
      n0 = normal(i+0,itime);
      n1 = normal(i+1,itime);
      dn0 = dnormal(i+0,itime);
      dn1 = dnormal(i+1,itime);
    }

    /*! loads curve vertices for specified time */
    __forceinline void gather_hermite(Vec3ff& p0, Vec3ff& t0, Vec3fa& n0, Vec3fa& dn0, Vec3ff& p1, Vec3ff& t1, Vec3fa& n1, Vec3fa& dn1, size_t i, float time) const
    {
      float ftime;
      const size_t itime = timeSegment(time, ftime);
      const float f0 = 1.0f - ftime, f1 = ftime;
      Vec3ff ap0,at0,ap1,at1; Vec3fa an0,adn0,an1,adn1;
      gather_hermite(ap0,at0,an0,adn0,ap1,at1,an1,adn1,i,itime);
      Vec3ff bp0,bt0,bp1,bt1; Vec3fa bn0,bdn0,bn1,bdn1;
      gather_hermite(bp0,bt0,bn0,bdn0,bp1,bt1,bn1,bdn1,i,itime+1);
      p0 = madd(Vec3ff(f0),ap0,f1*bp0);
      t0 = madd(Vec3ff(f0),at0,f1*bt0);
      n0 = madd(Vec3ff(f0),an0,f1*bn0);
      dn0= madd(Vec3ff(f0),adn0,f1*bdn0);
      p1 = madd(Vec3ff(f0),ap1,f1*bp1);
      t1 = madd(Vec3ff(f0),at1,f1*bt1);
      n1 = madd(Vec3ff(f0),an1,f1*bn1);
      dn1= madd(Vec3ff(f0),adn1,f1*bdn1);
    }

    /*! loads curve vertices for specified time */
    __forceinline void gather_hermite_safe(Vec3ff& p0, Vec3ff& t0, Vec3fa& n0, Vec3fa& dn0, Vec3ff& p1, Vec3ff& t1, Vec3fa& n1, Vec3fa& dn1, size_t i, float time) const
    {
      if (hasMotionBlur()) gather_hermite(p0,t0,n0,dn0,p1,t1,n1,dn1,i,time);
      else                 gather_hermite(p0,t0,n0,dn0,p1,t1,n1,dn1,i);
    }

    template<typename SourceCurve3ff, typename SourceCurve3fa, typename TensorLinearCubicBezierSurface3fa>
      __forceinline TensorLinearCubicBezierSurface3fa getNormalOrientedHermiteCurve(RayQueryContext* context, const Vec3fa& ray_org, const unsigned int primID, const size_t itime) const
    {
      Vec3ff v0,t0,v1,t1; Vec3fa n0,dn0,n1,dn1;
      unsigned int vertexID = curve(primID);
      gather_hermite(v0,t0,n0,dn0,v1,t1,n1,dn1,vertexID,itime);

      SourceCurve3ff ccurve(v0,t0,v1,t1);
      SourceCurve3fa ncurve(n0,dn0,n1,dn1);
      ccurve = enlargeRadiusToMinWidth(context,this,ray_org,ccurve);
      return TensorLinearCubicBezierSurface3fa::fromCenterAndNormalCurve(ccurve,ncurve);
    }

    template<typename SourceCurve3ff, typename SourceCurve3fa, typename TensorLinearCubicBezierSurface3fa>
    __forceinline TensorLinearCubicBezierSurface3fa getNormalOrientedHermiteCurve(RayQueryContext* context, const Vec3fa& ray_org, const unsigned int primID, const float time) const
    {
      float ftime;
      const size_t itime = timeSegment(time, ftime);
      const TensorLinearCubicBezierSurface3fa curve0 = getNormalOrientedHermiteCurve<SourceCurve3ff, SourceCurve3fa, TensorLinearCubicBezierSurface3fa>(context, ray_org, primID,itime+0);
      const TensorLinearCubicBezierSurface3fa curve1 = getNormalOrientedHermiteCurve<SourceCurve3ff, SourceCurve3fa, TensorLinearCubicBezierSurface3fa>(context, ray_org, primID,itime+1);
      return clerp(curve0,curve1,ftime);
    }

    template<typename SourceCurve3ff, typename SourceCurve3fa, typename TensorLinearCubicBezierSurface3fa>
    __forceinline TensorLinearCubicBezierSurface3fa getNormalOrientedHermiteCurveSafe(RayQueryContext* context, const Vec3fa& ray_org, const unsigned int primID, const float time) const
    {
      float ftime = 0.0f;
      const size_t itime = hasMotionBlur() ? timeSegment(time, ftime) : 0;
      const TensorLinearCubicBezierSurface3fa curve0 = getNormalOrientedHermiteCurve<SourceCurve3ff, SourceCurve3fa, TensorLinearCubicBezierSurface3fa>(context, ray_org, primID,itime+0);
      if (hasMotionBlur()) {
        const TensorLinearCubicBezierSurface3fa curve1 = getNormalOrientedHermiteCurve<SourceCurve3ff, SourceCurve3fa, TensorLinearCubicBezierSurface3fa>(context, ray_org, primID,itime+1);
        return clerp(curve0,curve1,ftime);
      }
      return curve0;
    }

    /* returns the projected area */
    __forceinline float projectedPrimitiveArea(const size_t i) const {
      return 1.0f;
    }
  
  private:
    void resizeBuffers(unsigned int numSteps);

  public:
    BufferView<unsigned int> curves;        //!< array of curve indices
    BufferView<Vec3ff> vertices0;           //!< fast access to first vertex buffer
    BufferView<Vec3fa> normals0;            //!< fast access to first normal buffer
    BufferView<Vec3ff> tangents0;           //!< fast access to first tangent buffer
    BufferView<Vec3fa> dnormals0;           //!< fast access to first normal derivative buffer
    Device::vector<BufferView<Vec3ff>> vertices = device;    //!< vertex array for each timestep
    Device::vector<BufferView<Vec3fa>> normals = device;     //!< normal array for each timestep
    Device::vector<BufferView<Vec3ff>> tangents = device;    //!< tangent array for each timestep
    Device::vector<BufferView<Vec3fa>> dnormals = device;    //!< normal derivative array for each timestep
    BufferView<char> flags;                 //!< start, end flag per segment
    Device::vector<BufferView<char>> vertexAttribs = device; //!< user buffers
    int tessellationRate;                   //!< tessellation rate for flat curve
    float maxRadiusScale = 1.0;             //!< maximal min-width scaling of curve radii
  };

  namespace isa
  {
    
  template<template<typename Ty> class Curve>
  struct CurveGeometryInterface : public CurveGeometry
  {
    typedef Curve<Vec3ff> Curve3ff;
    typedef Curve<Vec3fa> Curve3fa;
    
    CurveGeometryInterface (Device* device, Geometry::GType gtype)
      : CurveGeometry(device,gtype) {}
    
    __forceinline const Curve3ff getCurveScaledRadius(size_t i, size_t itime = 0) const 
    {
      const unsigned int index = curve(i);
      Vec3ff v0 = vertex(index+0,itime);
      Vec3ff v1 = vertex(index+1,itime);
      Vec3ff v2 = vertex(index+2,itime);
      Vec3ff v3 = vertex(index+3,itime);
      v0.w *= maxRadiusScale;
      v1.w *= maxRadiusScale;
      v2.w *= maxRadiusScale;
      v3.w *= maxRadiusScale;
      return Curve3ff (v0,v1,v2,v3);
    }
    
    __forceinline const Curve3ff getCurveScaledRadius(const LinearSpace3fa& space, size_t i, size_t itime = 0) const 
    {
      const unsigned int index = curve(i);
      const Vec3ff v0 = vertex(index+0,itime);
      const Vec3ff v1 = vertex(index+1,itime);
      const Vec3ff v2 = vertex(index+2,itime);
      const Vec3ff v3 = vertex(index+3,itime);
      const Vec3ff w0(xfmPoint(space,(Vec3fa)v0), maxRadiusScale*v0.w);
      const Vec3ff w1(xfmPoint(space,(Vec3fa)v1), maxRadiusScale*v1.w);
      const Vec3ff w2(xfmPoint(space,(Vec3fa)v2), maxRadiusScale*v2.w);
      const Vec3ff w3(xfmPoint(space,(Vec3fa)v3), maxRadiusScale*v3.w);
      return Curve3ff(w0,w1,w2,w3);
    }
    
    __forceinline const Curve3ff getCurveScaledRadius(const Vec3fa& ofs, const float scale, const float r_scale0, const LinearSpace3fa& space, size_t i, size_t itime = 0) const 
    {
      const float r_scale = r_scale0*scale;
      const unsigned int index = curve(i);
      const Vec3ff v0 = vertex(index+0,itime);
      const Vec3ff v1 = vertex(index+1,itime);
      const Vec3ff v2 = vertex(index+2,itime);
      const Vec3ff v3 = vertex(index+3,itime);
      const Vec3ff w0(xfmPoint(space,((Vec3fa)v0-ofs)*Vec3fa(scale)), maxRadiusScale*v0.w*r_scale);
      const Vec3ff w1(xfmPoint(space,((Vec3fa)v1-ofs)*Vec3fa(scale)), maxRadiusScale*v1.w*r_scale);
      const Vec3ff w2(xfmPoint(space,((Vec3fa)v2-ofs)*Vec3fa(scale)), maxRadiusScale*v2.w*r_scale);
      const Vec3ff w3(xfmPoint(space,((Vec3fa)v3-ofs)*Vec3fa(scale)), maxRadiusScale*v3.w*r_scale);
      return Curve3ff(w0,w1,w2,w3);
    }
    
    __forceinline const Curve3fa getNormalCurve(size_t i, size_t itime = 0) const 
    {
      const unsigned int index = curve(i);
      const Vec3fa n0 = normal(index+0,itime);
      const Vec3fa n1 = normal(index+1,itime);
      const Vec3fa n2 = normal(index+2,itime);
      const Vec3fa n3 = normal(index+3,itime);
      return Curve3fa (n0,n1,n2,n3);
    }
    
    __forceinline const TensorLinearCubicBezierSurface3fa getOrientedCurveScaledRadius(size_t i, size_t itime = 0) const 
    {
      const Curve3ff center = getCurveScaledRadius(i,itime);
      const Curve3fa normal = getNormalCurve(i,itime);
      const TensorLinearCubicBezierSurface3fa ocurve = TensorLinearCubicBezierSurface3fa::fromCenterAndNormalCurve(center,normal);
      return ocurve;
    }
    
    __forceinline const TensorLinearCubicBezierSurface3fa getOrientedCurveScaledRadius(const LinearSpace3fa& space, size_t i, size_t itime = 0) const {
      return getOrientedCurveScaledRadius(i,itime).xfm(space);
    }
    
    __forceinline const TensorLinearCubicBezierSurface3fa getOrientedCurveScaledRadius(const Vec3fa& ofs, const float scale, const LinearSpace3fa& space, size_t i, size_t itime = 0) const {
      return getOrientedCurveScaledRadius(i,itime).xfm(space,ofs,scale);
    }
    
    /*! check if the i'th primitive is valid at the itime'th time step */
    __forceinline bool valid(Geometry::GType ctype, size_t i, const range<size_t>& itime_range) const
    {
      const unsigned int index = curve(i);
      if (index+3 >= numVertices()) return false;
      
      for (size_t itime = itime_range.begin(); itime <= itime_range.end(); itime++)
      {
        const float r0 = radius(index+0,itime);
        const float r1 = radius(index+1,itime);
        const float r2 = radius(index+2,itime);
        const float r3 = radius(index+3,itime);
        if (!isvalid(r0) || !isvalid(r1) || !isvalid(r2) || !isvalid(r3))
          return false;
        
        const Vec3fa v0 = vertex(index+0,itime);
        const Vec3fa v1 = vertex(index+1,itime);
        const Vec3fa v2 = vertex(index+2,itime);
        const Vec3fa v3 = vertex(index+3,itime);
        if (!isvalid(v0) || !isvalid(v1) || !isvalid(v2) || !isvalid(v3))
          return false;
        
        if (ctype == Geometry::GTY_SUBTYPE_ORIENTED_CURVE)
        {
          const Vec3fa n0 = normal(index+0,itime);
          const Vec3fa n1 = normal(index+1,itime);
          if (!isvalid(n0) || !isvalid(n1))
            return false;

	  const BBox3fa b = getOrientedCurveScaledRadius(i,itime).accurateBounds();
	  if (!isvalid(b))
	    return false;
        }
      }
      
      return true;
    }

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
        size_t ofs = i*sizeof(float);
        const size_t index = curves[primID];
        const vbool<N> valid = vint<N>((int)i)+vint<N>(step) < vint<N>((int)valueCount);
        const vfloat<N> p0 = mem<vfloat<N>>::loadu(valid,(float*)&src[(index+0)*stride+ofs]);
        const vfloat<N> p1 = mem<vfloat<N>>::loadu(valid,(float*)&src[(index+1)*stride+ofs]);
        const vfloat<N> p2 = mem<vfloat<N>>::loadu(valid,(float*)&src[(index+2)*stride+ofs]);
        const vfloat<N> p3 = mem<vfloat<N>>::loadu(valid,(float*)&src[(index+3)*stride+ofs]);
        
        const Curve<vfloat<N>> curve(p0,p1,p2,p3);
        if (P      ) mem<vfloat<N>>::storeu(valid,P+i,      curve.eval(u));
        if (dPdu   ) mem<vfloat<N>>::storeu(valid,dPdu+i,   curve.eval_du(u));
        if (ddPdudu) mem<vfloat<N>>::storeu(valid,ddPdudu+i,curve.eval_dudu(u));
      }
    }

    void interpolate(const RTCInterpolateArguments* const args) {
      interpolate_impl<4>(args);
    }
  };
  
  template<template<typename Ty> class Curve>
  struct HermiteCurveGeometryInterface : public CurveGeometry
  {
    typedef Curve<Vec3ff> HermiteCurve3ff;
    typedef Curve<Vec3fa> HermiteCurve3fa;
    
    HermiteCurveGeometryInterface (Device* device, Geometry::GType gtype)
      : CurveGeometry(device,gtype) {}
    
    __forceinline const HermiteCurve3ff getCurveScaledRadius(size_t i, size_t itime = 0) const 
    {
      const unsigned int index = curve(i);
      Vec3ff v0 = vertex(index+0,itime);
      Vec3ff v1 = vertex(index+1,itime);
      Vec3ff t0 = tangent(index+0,itime);
      Vec3ff t1 = tangent(index+1,itime);
      v0.w *= maxRadiusScale;
      v1.w *= maxRadiusScale;
      t0.w *= maxRadiusScale;
      t1.w *= maxRadiusScale;
      return HermiteCurve3ff (v0,t0,v1,t1);
    }
    
    __forceinline const HermiteCurve3ff getCurveScaledRadius(const LinearSpace3fa& space, size_t i, size_t itime = 0) const 
    {
      const unsigned int index = curve(i);
      const Vec3ff v0 = vertex(index+0,itime);
      const Vec3ff v1 = vertex(index+1,itime);
      const Vec3ff t0 = tangent(index+0,itime);
      const Vec3ff t1 = tangent(index+1,itime);
      const Vec3ff V0(xfmPoint(space,(Vec3fa)v0),maxRadiusScale*v0.w);
      const Vec3ff V1(xfmPoint(space,(Vec3fa)v1),maxRadiusScale*v1.w);
      const Vec3ff T0(xfmVector(space,(Vec3fa)t0),maxRadiusScale*t0.w);
      const Vec3ff T1(xfmVector(space,(Vec3fa)t1),maxRadiusScale*t1.w);
      return HermiteCurve3ff(V0,T0,V1,T1);
    }
    
    __forceinline const HermiteCurve3ff getCurveScaledRadius(const Vec3fa& ofs, const float scale, const float r_scale0, const LinearSpace3fa& space, size_t i, size_t itime = 0) const 
    {
      const float r_scale = r_scale0*scale;
      const unsigned int index = curve(i);
      const Vec3ff v0 = vertex(index+0,itime);
      const Vec3ff v1 = vertex(index+1,itime);
      const Vec3ff t0 = tangent(index+0,itime);
      const Vec3ff t1 = tangent(index+1,itime);
      const Vec3ff V0(xfmPoint(space,(v0-ofs)*Vec3fa(scale)), maxRadiusScale*v0.w*r_scale);
      const Vec3ff V1(xfmPoint(space,(v1-ofs)*Vec3fa(scale)), maxRadiusScale*v1.w*r_scale);
      const Vec3ff T0(xfmVector(space,t0*Vec3fa(scale)), maxRadiusScale*t0.w*r_scale);
      const Vec3ff T1(xfmVector(space,t1*Vec3fa(scale)), maxRadiusScale*t1.w*r_scale);
      return HermiteCurve3ff(V0,T0,V1,T1);
    }
    
    __forceinline const HermiteCurve3fa getNormalCurve(size_t i, size_t itime = 0) const 
    {
      const unsigned int index = curve(i);
      const Vec3fa n0 = normal(index+0,itime);
      const Vec3fa n1 = normal(index+1,itime);
      const Vec3fa dn0 = dnormal(index+0,itime);
      const Vec3fa dn1 = dnormal(index+1,itime);
      return HermiteCurve3fa (n0,dn0,n1,dn1);
    }
    
    __forceinline const TensorLinearCubicBezierSurface3fa getOrientedCurveScaledRadius(size_t i, size_t itime = 0) const 
    {
      const HermiteCurve3ff center = getCurveScaledRadius(i,itime);
      const HermiteCurve3fa normal = getNormalCurve(i,itime);
      const TensorLinearCubicBezierSurface3fa ocurve = TensorLinearCubicBezierSurface3fa::fromCenterAndNormalCurve(center,normal);
      return ocurve;
    }
    
    __forceinline const TensorLinearCubicBezierSurface3fa getOrientedCurveScaledRadius(const LinearSpace3fa& space, size_t i, size_t itime = 0) const {
      return getOrientedCurveScaledRadius(i,itime).xfm(space);
    }
    
    __forceinline const TensorLinearCubicBezierSurface3fa getOrientedCurveScaledRadius(const Vec3fa& ofs, const float scale, const LinearSpace3fa& space, size_t i, size_t itime = 0) const {
      return getOrientedCurveScaledRadius(i,itime).xfm(space,ofs,scale);
    }
    
    /*! check if the i'th primitive is valid at the itime'th time step */
    __forceinline bool valid(Geometry::GType ctype, size_t i, const range<size_t>& itime_range) const
    {
      const unsigned int index = curve(i);
      if (index+1 >= numVertices()) return false;
      
      for (size_t itime = itime_range.begin(); itime <= itime_range.end(); itime++)
      {
        const Vec3ff v0 = vertex(index+0,itime);
        const Vec3ff v1 = vertex(index+1,itime);
        if (!isvalid4(v0) || !isvalid4(v1))
          return false;
        
        const Vec3ff t0 = tangent(index+0,itime);
        const Vec3ff t1 = tangent(index+1,itime);
        if (!isvalid4(t0) || !isvalid4(t1))
          return false;
        
        if (ctype == Geometry::GTY_SUBTYPE_ORIENTED_CURVE)
        {
          const Vec3fa n0 = normal(index+0,itime);
          const Vec3fa n1 = normal(index+1,itime);
          if (!isvalid(n0) || !isvalid(n1))
            return false;
          
          const Vec3fa dn0 = dnormal(index+0,itime);
          const Vec3fa dn1 = dnormal(index+1,itime);
          if (!isvalid(dn0) || !isvalid(dn1))
            return false;

	  const BBox3fa b = getOrientedCurveScaledRadius(i,itime).accurateBounds();
	  if (!isvalid(b))
	    return false;
        }
      }
      
      return true;
    }

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
      
      /* we interpolate vertex attributes linearly for hermite basis */
      if (bufferType == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE)
      {
        assert(bufferSlot <= vertexAttribs.size());
        const char* vsrc = vertexAttribs[bufferSlot].getPtr();
        const size_t vstride = vertexAttribs[bufferSlot].getStride();
        
        for (unsigned int i=0; i<valueCount; i+=N)
        {
          const size_t ofs = i*sizeof(float);
          const size_t index = curves[primID];
          const vbool<N> valid = vint<N>((int)i)+vint<N>(step) < vint<N>((int)valueCount);
          const vfloat<N> p0 = mem<vfloat<N>>::loadu(valid,(float*)&vsrc[(index+0)*vstride+ofs]);
          const vfloat<N> p1 = mem<vfloat<N>>::loadu(valid,(float*)&vsrc[(index+1)*vstride+ofs]);
          
          if (P      ) mem<vfloat<N>>::storeu(valid,P+i,      madd(1.0f-u,p0,u*p1));
          if (dPdu   ) mem<vfloat<N>>::storeu(valid,dPdu+i,   p1-p0);
          if (ddPdudu) mem<vfloat<N>>::storeu(valid,ddPdudu+i,vfloat<N>(zero));
        }
      }
      
      /* interpolation for vertex buffers */
      else
      {
        assert(bufferSlot < numTimeSteps);
        const char* vsrc = vertices[bufferSlot].getPtr();
        const char* tsrc = tangents[bufferSlot].getPtr();
        const size_t vstride = vertices[bufferSlot].getStride();
        const size_t tstride = vertices[bufferSlot].getStride();
        
        for (unsigned int i=0; i<valueCount; i+=N)
        {
          const size_t ofs = i*sizeof(float);
          const size_t index = curves[primID];
          const vbool<N> valid = vint<N>((int)i)+vint<N>(step) < vint<N>((int)valueCount);
          const vfloat<N> p0 = mem<vfloat<N>>::loadu(valid,(float*)&vsrc[(index+0)*vstride+ofs]);
          const vfloat<N> p1 = mem<vfloat<N>>::loadu(valid,(float*)&vsrc[(index+1)*vstride+ofs]);
          const vfloat<N> t0 = mem<vfloat<N>>::loadu(valid,(float*)&tsrc[(index+0)*tstride+ofs]);
          const vfloat<N> t1 = mem<vfloat<N>>::loadu(valid,(float*)&tsrc[(index+1)*tstride+ofs]);
          
          const HermiteCurveT<vfloat<N>> curve(p0,t0,p1,t1);
          if (P      ) mem<vfloat<N>>::storeu(valid,P+i,      curve.eval(u));
          if (dPdu   ) mem<vfloat<N>>::storeu(valid,dPdu+i,   curve.eval_du(u));
          if (ddPdudu) mem<vfloat<N>>::storeu(valid,ddPdudu+i,curve.eval_dudu(u));
        }
      }
    }

    void interpolate(const RTCInterpolateArguments* const args) {
      interpolate_impl<4>(args);
    }
  };
  }
  
  DECLARE_ISA_FUNCTION(CurveGeometry*, createCurves, Device* COMMA Geometry::GType);
}
