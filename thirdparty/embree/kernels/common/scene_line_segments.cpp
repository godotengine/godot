// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
      flags.userData = 1; // to encode that app manages this buffer
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
      segments.setModified();
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

  void LineSegments::setTessellationRate(float N) {
    tessellationRate = clamp((int)N,1,16);
  }

  void LineSegments::setMaxRadiusScale(float s) {
    maxRadiusScale = s;
  }

  void LineSegments::commit() 
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

    /* if no flags buffer is specified we manage and calculate the flags buffer */
    if (!flags.buffer) flags.userData = 0; // to encode that we manage this buffer
    bool recompute_flags_buffer = segments.isLocalModified();

    /* resize flags buffer if number of primitives changed */
    if (!flags.userData && (!flags.buffer || flags.size() != numPrimitives))
    {
      Ref<Buffer> buffer = new Buffer(device, numPrimitives*sizeof(char));
      flags.set(buffer, 0, sizeof(char), numPrimitives, RTC_FORMAT_UCHAR);
      recompute_flags_buffer = true;
    }

    /* recalculate the flags buffer if index buffer got modified */
    if (!flags.userData && recompute_flags_buffer)
    {
      bool hasLeft = false;
      for (size_t i=0; i<numPrimitives; i++) {
        bool hasRight = (i==numPrimitives-1) ? false : segment(i+1) == segment(i)+1;
        flags[i]  = hasLeft  * RTC_CURVE_FLAG_NEIGHBOR_LEFT;
        flags[i] |= hasRight * RTC_CURVE_FLAG_NEIGHBOR_RIGHT;
        hasLeft = hasRight;
      }
    }
    segments.clearLocalModified();

    Geometry::commit();
  }
  
  void LineSegments::addElementsToCount (GeometryCounts & counts) const 
  {
    if (numTimeSteps == 1) counts.numLineSegments += numPrimitives;
    else                   counts.numMBLineSegments += numPrimitives;
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

  void LineSegments::interpolate(const RTCInterpolateArguments* const args) {
    interpolate_impl<4>(args);
  }
#endif

  namespace isa
  {
    LineSegments* createLineSegments(Device* device, Geometry::GType gtype) {
      return new LineSegmentsISA(device,gtype);
    }
  }
}
