// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "scene_points.h"
#include "scene.h"

namespace embree
{
#if defined(EMBREE_LOWEST_ISA)

  Points::Points(Device* device, Geometry::GType gtype) : Geometry(device, gtype, 0, 1)
  {
    vertices.resize(numTimeSteps);
    if (getType() == GTY_ORIENTED_DISC_POINT)
      normals.resize(numTimeSteps);
  }

  void Points::setMask(unsigned mask)
  {
    this->mask = mask;
    Geometry::update();
  }

  void Points::setNumTimeSteps(unsigned int numTimeSteps)
  {
    vertices.resize(numTimeSteps);
    if (getType() == GTY_ORIENTED_DISC_POINT)
      normals.resize(numTimeSteps);
    Geometry::setNumTimeSteps(numTimeSteps);
  }

  void Points::setVertexAttributeCount(unsigned int N)
  {
    vertexAttribs.resize(N);
    Geometry::update();
  }

  void Points::setBuffer(RTCBufferType type,
                         unsigned int slot,
                         RTCFormat format,
                         const Ref<Buffer>& buffer,
                         size_t offset,
                         size_t stride,
                         unsigned int num)
  {
    /* verify that all accesses are 4 bytes aligned */
    if ((type != RTC_BUFFER_TYPE_FLAGS) && (((size_t(buffer->getPtr()) + offset) & 0x3) || (stride & 0x3)))
      throw_RTCError(RTC_ERROR_INVALID_OPERATION, "data must be 4 bytes aligned");

    if (type == RTC_BUFFER_TYPE_VERTEX) {
      if (format != RTC_FORMAT_FLOAT4)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid vertex buffer format");

      if (slot >= vertices.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid vertex buffer slot");

      vertices[slot].set(buffer, offset, stride, num, format);
      vertices[slot].checkPadding16();
      setNumPrimitives(num);
    } else if (type == RTC_BUFFER_TYPE_NORMAL) {
      if (getType() != GTY_ORIENTED_DISC_POINT)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "unknown buffer type");

      if (format != RTC_FORMAT_FLOAT3)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid normal buffer format");

      if (slot >= normals.size())
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid normal buffer slot");

      normals[slot].set(buffer, offset, stride, num, format);
      normals[slot].checkPadding16();
    } else if (type == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE) {
      if (format < RTC_FORMAT_FLOAT || format > RTC_FORMAT_FLOAT16)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid vertex attribute buffer format");

      if (slot >= vertexAttribs.size())
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid vertex attribute buffer slot");

      vertexAttribs[slot].set(buffer, offset, stride, num, format);
      vertexAttribs[slot].checkPadding16();
    } else
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "unknown buffer type");
  }

  void* Points::getBuffer(RTCBufferType type, unsigned int slot)
  {
    if (type == RTC_BUFFER_TYPE_VERTEX) {
      if (slot >= vertices.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return vertices[slot].getPtr();
    } else if (type == RTC_BUFFER_TYPE_NORMAL) {
      if (slot >= normals.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return normals[slot].getPtr();
    } else if (type == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE) {
      if (slot >= vertexAttribs.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return vertexAttribs[slot].getPtr();
    } else {
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "unknown buffer type");
      return nullptr;
    }
  }

  void Points::updateBuffer(RTCBufferType type, unsigned int slot)
  {
    if (type == RTC_BUFFER_TYPE_VERTEX) {
      if (slot >= vertices.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      vertices[slot].setModified();
    } else if (type == RTC_BUFFER_TYPE_NORMAL) {
      if (slot >= normals.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      normals[slot].setModified();
    } else if (type == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE) {
      if (slot >= vertexAttribs.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      vertexAttribs[slot].setModified();
    } else {
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "unknown buffer type");
    }

    Geometry::update();
  }

  void Points::setMaxRadiusScale(float s) {
    maxRadiusScale = s;
  }

  void Points::commit()
  {
    /* verify that stride of all time steps are identical */
    for (unsigned int t = 0; t < numTimeSteps; t++)
      if (vertices[t].getStride() != vertices[0].getStride())
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "stride of vertex buffers have to be identical for each time step");

    for (const auto& buffer : normals)
      if (buffer.getStride() != normals[0].getStride())
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "stride of normal buffers have to be identical for each time step");

    vertices0 = vertices[0];
    if (getType() == GTY_ORIENTED_DISC_POINT)
      normals0 = normals[0];

    Geometry::commit();
  }

  void Points::addElementsToCount (GeometryCounts & counts) const 
  {
    if (numTimeSteps == 1)
      counts.numPoints += numPrimitives;
    else
      counts.numMBPoints += numPrimitives;
  }

  bool Points::verify()
  {
    /*! verify consistent size of vertex arrays */
    if (vertices.size() == 0)
      return false;

    for (const auto& buffer : vertices)
      if (buffer.size() != numVertices())
        return false;

    if (getType() == GTY_ORIENTED_DISC_POINT) {
      if (normals.size() == 0)
        return false;

      for (const auto& buffer : normals)
        if (vertices[0].size() != buffer.size())
          return false;
    } else {
      if (normals.size())
        return false;
    }

    /*! verify vertices */
    for (const auto& buffer : vertices) {
      for (size_t i = 0; i < buffer.size(); i++) {
        if (!isvalid(buffer[i].x))
          return false;
        if (!isvalid(buffer[i].y))
          return false;
        if (!isvalid(buffer[i].z))
          return false;
        if (!isvalid(buffer[i].w))
          return false;
      }
    }
    return true;
  }
#endif

  namespace isa
  {
    Points* createPoints(Device* device, Geometry::GType gtype)
    {
      return new PointsISA(device, gtype);
    }
  }  // namespace isa
}  // namespace embree
