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

#include "scene_curves.h"
#include "scene.h"

#include "../subdiv/bezier_curve.h"
#include "../subdiv/hermite_curve.h"
#include "../subdiv/bspline_curve.h"
#include "../subdiv/linear_bezier_patch.h"

namespace embree
{
#if defined(EMBREE_LOWEST_ISA)

  CurveGeometry::CurveGeometry (Device* device, GType gtype)
    : Geometry(device,gtype,0,1), tessellationRate(4)
  {
    vertices.resize(numTimeSteps);
  }

  void CurveGeometry::enabling() 
  {
    if (numTimeSteps == 1) scene->world.numBezierCurves += numPrimitives; 
    else                   scene->worldMB.numBezierCurves += numPrimitives; 
  }
  
  void CurveGeometry::disabling() 
  {
    if (numTimeSteps == 1) scene->world.numBezierCurves -= numPrimitives; 
    else                   scene->worldMB.numBezierCurves -= numPrimitives;
  }
  
  void CurveGeometry::setMask (unsigned mask) 
  {
    this->mask = mask; 
    Geometry::update();
  }

  void CurveGeometry::setNumTimeSteps (unsigned int numTimeSteps)
  {
    vertices.resize(numTimeSteps);
    
    if (getCurveType() == GTY_SUBTYPE_ORIENTED_CURVE)
    {
      normals.resize(numTimeSteps);
      
      if (getCurveBasis() == GTY_BASIS_HERMITE)
        dnormals.resize(numTimeSteps);
    }
    if (getCurveBasis() == GTY_BASIS_HERMITE)
      tangents.resize(numTimeSteps);
    
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
      curves.setModified(true);
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX)
    {
      if (slot >= vertices.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      vertices[slot].setModified(true);
    }
    else if (type == RTC_BUFFER_TYPE_NORMAL)
    {
      if (slot >= normals.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      normals[slot].setModified(true);
    }
    else if (type == RTC_BUFFER_TYPE_TANGENT)
    {
      if (slot >= tangents.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      tangents[slot].setModified(true);
    }
    else if (type == RTC_BUFFER_TYPE_NORMAL_DERIVATIVE)
    {
      if (slot >= dnormals.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      dnormals[slot].setModified(true);
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE)
    {
      if (slot >= vertexAttribs.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      vertexAttribs[slot].setModified(true);
    }
    else if (type == RTC_BUFFER_TYPE_FLAGS) 
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      flags.setModified(true);
    }
    else
    {
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "unknown buffer type");
    }

    Geometry::update();
  }

  void CurveGeometry::setTessellationRate(float N)
  {
    tessellationRate = clamp((int)N,1,16);
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

  void CurveGeometry::preCommit()
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

    Geometry::preCommit();
  }

  void CurveGeometry::postCommit() 
  {
    curves.setModified(false);
    for (auto& buf : vertices) buf.setModified(false);
    for (auto& buf : normals)  buf.setModified(false);
    for (auto& buf : tangents) buf.setModified(false);
    for (auto& buf : dnormals)  buf.setModified(false);
    for (auto& attrib : vertexAttribs) attrib.setModified(false);
    flags.setModified(false);

    Geometry::postCommit();
  }

#endif

  namespace isa
  {
    template<typename Curve3fa, typename Curve4f>
    struct CurveGeometryInterface : public CurveGeometry
    {
      CurveGeometryInterface (Device* device, Geometry::GType gtype)
        : CurveGeometry(device,gtype) {}
      
      __forceinline const Curve3fa getCurve(size_t i, size_t itime = 0) const 
      {
        const unsigned int index = curve(i);
        const Vec3fa v0 = vertex(index+0,itime);
        const Vec3fa v1 = vertex(index+1,itime);
        const Vec3fa v2 = vertex(index+2,itime);
        const Vec3fa v3 = vertex(index+3,itime);
        return Curve3fa (v0,v1,v2,v3);
      }

      __forceinline const Curve3fa getCurve(const LinearSpace3fa& space, size_t i, size_t itime = 0) const 
      {
        const unsigned int index = curve(i);
        const Vec3fa v0 = vertex(index+0,itime);
        const Vec3fa v1 = vertex(index+1,itime);
        const Vec3fa v2 = vertex(index+2,itime);
        const Vec3fa v3 = vertex(index+3,itime);
        Vec3fa w0 = xfmPoint(space,v0); w0.w = v0.w;
        Vec3fa w1 = xfmPoint(space,v1); w1.w = v1.w;
        Vec3fa w2 = xfmPoint(space,v2); w2.w = v2.w;
        Vec3fa w3 = xfmPoint(space,v3); w3.w = v3.w;
        return Curve3fa(w0,w1,w2,w3);
      }

       __forceinline const Curve3fa getCurve(const Vec3fa& ofs, const float scale, const float r_scale0, const LinearSpace3fa& space, size_t i, size_t itime = 0) const 
      {
        const float r_scale = r_scale0*scale;
        const unsigned int index = curve(i);
        const Vec3fa v0 = vertex(index+0,itime);
        const Vec3fa v1 = vertex(index+1,itime);
        const Vec3fa v2 = vertex(index+2,itime);
        const Vec3fa v3 = vertex(index+3,itime);
        Vec3fa w0 = xfmPoint(space,(v0-ofs)*Vec3fa(scale)); w0.w = v0.w*r_scale;
        Vec3fa w1 = xfmPoint(space,(v1-ofs)*Vec3fa(scale)); w1.w = v1.w*r_scale;
        Vec3fa w2 = xfmPoint(space,(v2-ofs)*Vec3fa(scale)); w2.w = v2.w*r_scale;
        Vec3fa w3 = xfmPoint(space,(v3-ofs)*Vec3fa(scale)); w3.w = v3.w*r_scale;
        return Curve3fa(w0,w1,w2,w3);
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

      __forceinline const TensorLinearCubicBezierSurface3fa getOrientedCurve(size_t i, size_t itime = 0) const 
      {
        const Curve3fa center = getCurve(i,itime);
        const Curve3fa normal = getNormalCurve(i,itime);
        const TensorLinearCubicBezierSurface3fa ocurve = TensorLinearCubicBezierSurface3fa::fromCenterAndNormalCurve(center,normal);
        return ocurve;
      }

      __forceinline const TensorLinearCubicBezierSurface3fa getOrientedCurve(const LinearSpace3fa& space, size_t i, size_t itime = 0) const {
        return getOrientedCurve(i,itime).xfm(space);
      }

      __forceinline const TensorLinearCubicBezierSurface3fa getOrientedCurve(const Vec3fa& ofs, const float scale, const LinearSpace3fa& space, size_t i, size_t itime = 0) const {
        return getOrientedCurve(i,itime).xfm(space,ofs,scale);
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
          }
        }
        
        return true;
      }

      void interpolate(const RTCInterpolateArguments* const args)
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
        
        for (unsigned int i=0; i<valueCount; i+=4)
        {
          size_t ofs = i*sizeof(float);
          const size_t index = curves[primID];
          const vbool4 valid = vint4((int)i)+vint4(step) < vint4((int)valueCount);
          const vfloat4 p0 = vfloat4::loadu(valid,(float*)&src[(index+0)*stride+ofs]);
          const vfloat4 p1 = vfloat4::loadu(valid,(float*)&src[(index+1)*stride+ofs]);
          const vfloat4 p2 = vfloat4::loadu(valid,(float*)&src[(index+2)*stride+ofs]);
          const vfloat4 p3 = vfloat4::loadu(valid,(float*)&src[(index+3)*stride+ofs]);
          
          const Curve4f curve(p0,p1,p2,p3);
          if (P      ) vfloat4::storeu(valid,P+i,      curve.eval(u));
          if (dPdu   ) vfloat4::storeu(valid,dPdu+i,   curve.eval_du(u));
          if (ddPdudu) vfloat4::storeu(valid,ddPdudu+i,curve.eval_dudu(u));
        }
      }
    };

    struct HermiteCurveGeometryInterface : public CurveGeometry
    {
      HermiteCurveGeometryInterface (Device* device, Geometry::GType gtype)
        : CurveGeometry(device,gtype) {}
      
      __forceinline const HermiteCurve3fa getCurve(size_t i, size_t itime = 0) const 
      {
        const unsigned int index = curve(i);
        const Vec3fa v0 = vertex(index+0,itime);
        const Vec3fa v1 = vertex(index+1,itime);
        const Vec3fa t0 = tangent(index+0,itime);
        const Vec3fa t1 = tangent(index+1,itime);
        return HermiteCurve3fa (v0,t0,v1,t1);
      }

      __forceinline const HermiteCurve3fa getCurve(const LinearSpace3fa& space, size_t i, size_t itime = 0) const 
      {
        const unsigned int index = curve(i);
        const Vec3fa v0 = vertex(index+0,itime);
        const Vec3fa v1 = vertex(index+1,itime);
        const Vec3fa t0 = tangent(index+0,itime);
        const Vec3fa t1 = tangent(index+1,itime);
        Vec3fa V0 = xfmPoint(space,v0); V0.w = v0.w;
        Vec3fa V1 = xfmPoint(space,v1); V1.w = v1.w;
        Vec3fa T0 = xfmVector(space,t0); T0.w = t0.w;
        Vec3fa T1 = xfmVector(space,t1); T1.w = t1.w;
        return HermiteCurve3fa(V0,T0,V1,T1);
      }

      __forceinline const HermiteCurve3fa getCurve(const Vec3fa& ofs, const float scale, const float r_scale0, const LinearSpace3fa& space, size_t i, size_t itime = 0) const 
      {
        const float r_scale = r_scale0*scale;
        const unsigned int index = curve(i);
        const Vec3fa v0 = vertex(index+0,itime);
        const Vec3fa v1 = vertex(index+1,itime);
        const Vec3fa t0 = tangent(index+0,itime);
        const Vec3fa t1 = tangent(index+1,itime);
        Vec3fa V0 = xfmPoint(space,(v0-ofs)*Vec3fa(scale)); V0.w = v0.w*r_scale;
        Vec3fa V1 = xfmPoint(space,(v1-ofs)*Vec3fa(scale)); V1.w = v1.w*r_scale;
        Vec3fa T0 = xfmVector(space,t0*Vec3fa(scale)); T0.w = t0.w*r_scale;
        Vec3fa T1 = xfmVector(space,t1*Vec3fa(scale)); T1.w = t1.w*r_scale;
        return HermiteCurve3fa(V0,T0,V1,T1);
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

      __forceinline const TensorLinearCubicBezierSurface3fa getOrientedCurve(size_t i, size_t itime = 0) const 
      {
        const HermiteCurve3fa center = getCurve(i,itime);
        const HermiteCurve3fa normal = getNormalCurve(i,itime);
        const TensorLinearCubicBezierSurface3fa ocurve = TensorLinearCubicBezierSurface3fa::fromCenterAndNormalCurve(center,normal);
        return ocurve;
      }

      __forceinline const TensorLinearCubicBezierSurface3fa getOrientedCurve(const LinearSpace3fa& space, size_t i, size_t itime = 0) const {
        return getOrientedCurve(i,itime).xfm(space);
      }

      __forceinline const TensorLinearCubicBezierSurface3fa getOrientedCurve(const Vec3fa& ofs, const float scale, const LinearSpace3fa& space, size_t i, size_t itime = 0) const {
        return getOrientedCurve(i,itime).xfm(space,ofs,scale);
      }

      /*! check if the i'th primitive is valid at the itime'th time step */
      __forceinline bool valid(Geometry::GType ctype, size_t i, const range<size_t>& itime_range) const
      {
        const unsigned int index = curve(i);
        if (index+1 >= numVertices()) return false;
        
        for (size_t itime = itime_range.begin(); itime <= itime_range.end(); itime++)
        {
          const vfloat4 v0(vertex(index+0,itime));
          const vfloat4 v1(vertex(index+1,itime));
          if (!isvalid(v0) || !isvalid(v1))
            return false;

          const vfloat4 t0(tangent(index+0,itime));
          const vfloat4 t1(tangent(index+1,itime));
          if (!isvalid(t0) || !isvalid(t1))
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
          }
        }
        
        return true;
      }

      void interpolate(const RTCInterpolateArguments* const args)
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
          
          for (unsigned int i=0; i<valueCount; i+=4)
          {
            const size_t ofs = i*sizeof(float);
            const size_t index = curves[primID];
            const vbool4 valid = vint4((int)i)+vint4(step) < vint4((int)valueCount);
            const vfloat4 p0 = vfloat4::loadu(valid,(float*)&vsrc[(index+0)*vstride+ofs]);
            const vfloat4 p1 = vfloat4::loadu(valid,(float*)&vsrc[(index+1)*vstride+ofs]);
            
            if (P      ) vfloat4::storeu(valid,P+i,      madd(1.0f-u,p0,u*p1));
            if (dPdu   ) vfloat4::storeu(valid,dPdu+i,   p1-p0);
            if (ddPdudu) vfloat4::storeu(valid,ddPdudu+i,vfloat4(zero));
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
          
          for (unsigned int i=0; i<valueCount; i+=4)
          {
            const size_t ofs = i*sizeof(float);
            const size_t index = curves[primID];
            const vbool4 valid = vint4((int)i)+vint4(step) < vint4((int)valueCount);
            const vfloat4 p0 = vfloat4::loadu(valid,(float*)&vsrc[(index+0)*vstride+ofs]);
            const vfloat4 p1 = vfloat4::loadu(valid,(float*)&vsrc[(index+1)*vstride+ofs]);
            const vfloat4 t0 = vfloat4::loadu(valid,(float*)&tsrc[(index+0)*tstride+ofs]);
            const vfloat4 t1 = vfloat4::loadu(valid,(float*)&tsrc[(index+1)*tstride+ofs]);
            
            const HermiteCurveT<vfloat4> curve(p0,t0,p1,t1);
            if (P      ) vfloat4::storeu(valid,P+i,      curve.eval(u));
            if (dPdu   ) vfloat4::storeu(valid,dPdu+i,   curve.eval_du(u));
            if (ddPdudu) vfloat4::storeu(valid,ddPdudu+i,curve.eval_dudu(u));
          }
        }
      }
    };
    
    template<Geometry::GType ctype, typename CurveInterface, typename Curve3fa, typename Curve4f>
    struct CurveGeometryISA : public CurveInterface
    {
      using CurveInterface::getCurve;
      using CurveInterface::getOrientedCurve;
      using CurveInterface::numTimeSteps;
      using CurveInterface::fnumTimeSegments;
      using CurveInterface::numTimeSegments;
      using CurveInterface::tessellationRate;

      using CurveInterface::valid;
      using CurveInterface::numVertices;
      using CurveInterface::vertexAttribs;
      using CurveInterface::vertices;
      using CurveInterface::curves;
      using CurveInterface::curve;
      using CurveInterface::radius;
      using CurveInterface::vertex;
      using CurveInterface::normal;
      
      CurveGeometryISA (Device* device, Geometry::GType gtype)
        : CurveInterface(device,gtype) {}

      LinearSpace3fa computeAlignedSpace(const size_t primID) const
      {
        Vec3fa axisz(0,0,1);
        Vec3fa axisy(0,1,0);
        
        const Curve3fa curve = getCurve(primID);
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
        const Curve3fa curve = getCurve(primID,t);
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
        const Curve3fa c = getCurve(primID);
        const Vec3fa p0 = c.begin();
        const Vec3fa p3 = c.end();
        const Vec3fa axis1 = p3 - p0;
        return axis1;
      }

      Vec3fa computeDirection(unsigned int primID, size_t time) const
      {
        const Curve3fa c = getCurve(primID,time);
        const Vec3fa p0 = c.begin();
        const Vec3fa p3 = c.end();
        const Vec3fa axis1 = p3 - p0;
        return axis1;
      }

      /*! calculates bounding box of i'th bezier curve */
      __forceinline BBox3fa bounds(size_t i, size_t itime = 0) const
      {
        switch (ctype) {
        case Geometry::GTY_SUBTYPE_FLAT_CURVE: return getCurve(i,itime).accurateFlatBounds(tessellationRate);
        case Geometry::GTY_SUBTYPE_ROUND_CURVE: return getCurve(i,itime).accurateRoundBounds();
        case Geometry::GTY_SUBTYPE_ORIENTED_CURVE: return getOrientedCurve(i,itime).accurateBounds();
        default: return empty;
        }
      }
      
      /*! calculates bounding box of i'th bezier curve */
      __forceinline BBox3fa bounds(const LinearSpace3fa& space, size_t i, size_t itime = 0) const
      {
        switch (ctype) {
        case Geometry::GTY_SUBTYPE_FLAT_CURVE: return getCurve(space,i,itime).accurateFlatBounds(tessellationRate);
        case Geometry::GTY_SUBTYPE_ROUND_CURVE: return getCurve(space,i,itime).accurateRoundBounds();
        case Geometry::GTY_SUBTYPE_ORIENTED_CURVE: return getOrientedCurve(space,i,itime).accurateBounds();
        default: return empty;
        }
      }
      
      /*! calculates bounding box of i'th bezier curve */
      __forceinline BBox3fa bounds(const Vec3fa& ofs, const float scale, const float r_scale0, const LinearSpace3fa& space, size_t i, size_t itime = 0) const
      {
        switch (ctype) {
        case Geometry::GTY_SUBTYPE_FLAT_CURVE: return getCurve(ofs,scale,r_scale0,space,i,itime).accurateFlatBounds(tessellationRate);
        case Geometry::GTY_SUBTYPE_ROUND_CURVE: return getCurve(ofs,scale,r_scale0,space,i,itime).accurateRoundBounds();
        case Geometry::GTY_SUBTYPE_ORIENTED_CURVE: return getOrientedCurve(ofs,scale,space,i,itime).accurateBounds();
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
      
      PrimInfo createPrimRefArray(mvector<PrimRef>& prims, const range<size_t>& r, size_t k) const
      {
        PrimInfo pinfo(empty);
        for (size_t j=r.begin(); j<r.end(); j++)
        {
          if (!valid(ctype, j, make_range<size_t>(0, numTimeSegments()))) continue;
          const BBox3fa box = bounds(j);
          if (box.empty()) continue; // checks oriented curves with invalid normals which cause NaNs here
          const PrimRef prim(box,this->geomID,unsigned(j));
          pinfo.add_center2(prim);
          prims[k++] = prim;
        }
        return pinfo;
      }

      PrimInfoMB createPrimRefMBArray(mvector<PrimRefMB>& prims, const BBox1f& t0t1, const range<size_t>& r, size_t k) const
      {
        PrimInfoMB pinfo(empty);
        for (size_t j=r.begin(); j<r.end(); j++)
        {
          if (!valid(ctype, j, this->timeSegmentRange(t0t1))) continue;
          const LBBox3fa lbox = linearBounds(j,t0t1);
          if (lbox.bounds0.empty() || lbox.bounds1.empty()) continue; // checks oriented curves with invalid normals which cause NaNs here
          const PrimRefMB prim(lbox,this->numTimeSegments(),this->time_range,this->numTimeSegments(),this->geomID,unsigned(j));
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
      case Geometry::GTY_ROUND_BEZIER_CURVE: return new CurveGeometryISA<Geometry::GTY_SUBTYPE_ROUND_CURVE,CurveGeometryInterface<BezierCurve3fa,BezierCurveT<vfloat4>>,BezierCurve3fa,BezierCurveT<vfloat4>>(device,gtype);
      case Geometry::GTY_FLAT_BEZIER_CURVE : return new CurveGeometryISA<Geometry::GTY_SUBTYPE_FLAT_CURVE,CurveGeometryInterface<BezierCurve3fa,BezierCurveT<vfloat4>>,BezierCurve3fa,BezierCurveT<vfloat4>>(device,gtype);
      case Geometry::GTY_ORIENTED_BEZIER_CURVE : return new CurveGeometryISA<Geometry::GTY_SUBTYPE_ORIENTED_CURVE,CurveGeometryInterface<BezierCurve3fa,BezierCurveT<vfloat4>>,BezierCurve3fa,BezierCurveT<vfloat4>>(device,gtype);
        
      case Geometry::GTY_ROUND_BSPLINE_CURVE: return new CurveGeometryISA<Geometry::GTY_SUBTYPE_ROUND_CURVE,CurveGeometryInterface<BSplineCurve3fa,BSplineCurveT<vfloat4>>,BSplineCurve3fa,BSplineCurveT<vfloat4>>(device,gtype);
      case Geometry::GTY_FLAT_BSPLINE_CURVE : return new CurveGeometryISA<Geometry::GTY_SUBTYPE_FLAT_CURVE,CurveGeometryInterface<BSplineCurve3fa,BSplineCurveT<vfloat4>>,BSplineCurve3fa,BSplineCurveT<vfloat4>>(device,gtype);
      case Geometry::GTY_ORIENTED_BSPLINE_CURVE : return new CurveGeometryISA<Geometry::GTY_SUBTYPE_ORIENTED_CURVE,CurveGeometryInterface<BSplineCurve3fa,BSplineCurveT<vfloat4>>,BSplineCurve3fa,BSplineCurveT<vfloat4>>(device,gtype);

      case Geometry::GTY_ROUND_HERMITE_CURVE: return new CurveGeometryISA<Geometry::GTY_SUBTYPE_ROUND_CURVE,HermiteCurveGeometryInterface,HermiteCurve3fa,HermiteCurveT<vfloat4>>(device,gtype);
      case Geometry::GTY_FLAT_HERMITE_CURVE : return new CurveGeometryISA<Geometry::GTY_SUBTYPE_FLAT_CURVE,HermiteCurveGeometryInterface,HermiteCurve3fa,HermiteCurveT<vfloat4>>(device,gtype);
      case Geometry::GTY_ORIENTED_HERMITE_CURVE : return new CurveGeometryISA<Geometry::GTY_SUBTYPE_ORIENTED_CURVE,HermiteCurveGeometryInterface,HermiteCurve3fa,HermiteCurveT<vfloat4>>(device,gtype);
     
      default: throw_RTCError(RTC_ERROR_INVALID_OPERATION,"invalid geometry type");
      }
    }
  }
}
