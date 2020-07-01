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

#include "scene_line_segments.h"
#include "scene.h"

namespace embree
{
#if defined(EMBREE_LOWEST_ISA)

  LineSegments::LineSegments (Device* device, Geometry::GType gtype)
    : Geometry(device,gtype,0,1), tessellationRate(4)
  {
    vertices.resize(numTimeSteps);
  }

  void LineSegments::enabling()
  {
    if (numTimeSteps == 1) scene->world.numLineSegments += numPrimitives;
    else                   scene->worldMB.numLineSegments += numPrimitives;
  }

  void LineSegments::disabling()
  {
    if (numTimeSteps == 1) scene->world.numLineSegments -= numPrimitives;
    else                   scene->worldMB.numLineSegments -= numPrimitives;
  }

  void LineSegments::setMask (unsigned mask)
  {
    this->mask = mask;
    Geometry::update();
  }

  void LineSegments::setNumTimeSteps (unsigned int numTimeSteps)
  {
    vertices.resize(numTimeSteps);
    if (getCurveType() == GTY_SUBTYPE_ORIENTED_CURVE)
      normals.resize(numTimeSteps);
    Geometry::setNumTimeSteps(numTimeSteps);
  }

  void LineSegments::setVertexAttributeCount (unsigned int N)
  {
    vertexAttribs.resize(N);
    Geometry::update();
  }
  
  void LineSegments::setBuffer(RTCBufferType type, unsigned int slot, RTCFormat format, const Ref<Buffer>& buffer, size_t offset, size_t stride, unsigned int num)
  {
    /* verify that all accesses are 4 bytes aligned */
    if ((type != RTC_BUFFER_TYPE_FLAGS) && (((size_t(buffer->getPtr()) + offset) & 0x3) || (stride & 0x3)))
      throw_RTCError(RTC_ERROR_INVALID_OPERATION, "data must be 4 bytes aligned");

    if (type == RTC_BUFFER_TYPE_VERTEX)
    {
      if (format != RTC_FORMAT_FLOAT4)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid vertex buffer format");

      if (slot >= vertices.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid vertex buffer slot");
      
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

      segments.set(buffer, offset, stride, num, format);
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

  void* LineSegments::getBuffer(RTCBufferType type, unsigned int slot)
  {
    if (type == RTC_BUFFER_TYPE_INDEX)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return segments.getPtr();
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

  void LineSegments::updateBuffer(RTCBufferType type, unsigned int slot)
  {
    if (type == RTC_BUFFER_TYPE_INDEX)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      segments.setModified(true);
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

  void LineSegments::setTessellationRate(float N)
  {
    tessellationRate = clamp((int)N,1,16);
  }

  void LineSegments::preCommit() 
  {
    /* verify that stride of all time steps are identical */
    for (unsigned int t=0; t<numTimeSteps; t++)
      if (vertices[t].getStride() != vertices[0].getStride())
        throw_RTCError(RTC_ERROR_INVALID_OPERATION,"stride of vertex buffers have to be identical for each time step");

    for (const auto& buffer : normals)
      if (buffer.getStride() != normals[0].getStride())
        throw_RTCError(RTC_ERROR_INVALID_OPERATION,"stride of normal buffers have to be identical for each time step");

    vertices0 = vertices[0];
    if (getCurveType() == GTY_SUBTYPE_ORIENTED_CURVE)
      normals0 = normals[0];
        
    Geometry::preCommit();
  }

  void LineSegments::postCommit() 
  {
    scene->vertices[geomID] = (float*) vertices0.getPtr();

    segments.setModified(false);
    for (auto& buf : vertices) buf.setModified(false);
    for (auto& buf : normals)  buf.setModified(false);
    for (auto& attrib : vertexAttribs) attrib.setModified(false);
    flags.setModified(false);

    Geometry::postCommit();
  }

  bool LineSegments::verify ()
  { 
    /*! verify consistent size of vertex arrays */
    if (vertices.size() == 0)
      return false;
    
    for (const auto& buffer : vertices)
      if (buffer.size() != numVertices())
        return false;

    for (const auto& buffer : normals)
      if (vertices[0].size() != buffer.size())
        return false;

    /*! verify segment indices */
    for (unsigned int i=0; i<size(); i++) {
      if (segments[i]+1 >= numVertices()) return false;
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

  void LineSegments::interpolate(const RTCInterpolateArguments* const args)
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
      const size_t ofs = i*sizeof(float);
      const size_t segment = segments[primID];
      const vbool4 valid = vint4((int)i)+vint4(step) < vint4(int(valueCount));
      const vfloat4 p0 = vfloat4::loadu(valid,(float*)&src[(segment+0)*stride+ofs]);
      const vfloat4 p1 = vfloat4::loadu(valid,(float*)&src[(segment+1)*stride+ofs]);
      if (P      ) vfloat4::storeu(valid,P+i,lerp(p0,p1,u));
      if (dPdu   ) vfloat4::storeu(valid,dPdu+i,p1-p0);
      if (ddPdudu) vfloat4::storeu(valid,dPdu+i,vfloat4(zero));
    }
  }
#endif

  namespace isa
  {
    LineSegments* createLineSegments(Device* device, Geometry::GType gtype) {
      return new LineSegmentsISA(device,gtype);
    }
  }
}
