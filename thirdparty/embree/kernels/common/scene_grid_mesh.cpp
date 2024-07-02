// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "scene_grid_mesh.h"
#include "scene.h"

namespace embree
{
#if defined(EMBREE_LOWEST_ISA)

  GridMesh::GridMesh (Device* device)
    : Geometry(device,GTY_GRID_MESH,0,1)
  {
    vertices.resize(numTimeSteps);
  }

  void GridMesh::setMask (unsigned mask) 
  {
    this->mask = mask; 
    Geometry::update();
  }

  void GridMesh::setNumTimeSteps (unsigned int numTimeSteps)
  {
    vertices.resize(numTimeSteps);
    Geometry::setNumTimeSteps(numTimeSteps);
  }

  void GridMesh::setVertexAttributeCount (unsigned int N)
  {
    vertexAttribs.resize(N);
    Geometry::update();
  }
  
  void GridMesh::setBuffer(RTCBufferType type, unsigned int slot, RTCFormat format, const Ref<Buffer>& buffer, size_t offset, size_t stride, unsigned int num)
  {
    /* verify that all accesses are 4 bytes aligned */
    if (((size_t(buffer->getPtr()) + offset) & 0x3) || (stride & 0x3)) 
      throw_RTCError(RTC_ERROR_INVALID_OPERATION, "data must be 4 bytes aligned");

    if (type == RTC_BUFFER_TYPE_VERTEX)
    {
      if (format != RTC_FORMAT_FLOAT3)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid vertex buffer format");

      /* if buffer is larger than 16GB the premultiplied index optimization does not work */
      if (stride*num > 16ll*1024ll*1024ll*1024ll)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "vertex buffer can be at most 16GB large");

      if (slot >= vertices.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid vertex buffer slot");

      vertices[slot].set(buffer, offset, stride, num, format);
      vertices[slot].checkPadding16();
      vertices0 = vertices[0];
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
	else if (type == RTC_BUFFER_TYPE_GRID)
	{
		if (slot != 0)
			throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
		if (format != RTC_FORMAT_GRID)
			throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid index buffer format");

		grids.set(buffer, offset, stride, num, format);
		setNumPrimitives(num);
	}
    else 
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "unknown buffer type");
  }

  void* GridMesh::getBuffer(RTCBufferType type, unsigned int slot)
  {
    if (type == RTC_BUFFER_TYPE_GRID)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return grids.getPtr();
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX)
    {
      if (slot >= vertices.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return vertices[slot].getPtr();
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE)
    {
      if (slot >= vertexAttribs.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return vertexAttribs[slot].getPtr();
    }
    else
    {
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "unknown buffer type");
      return nullptr;
    }
  }

  void GridMesh::updateBuffer(RTCBufferType type, unsigned int slot)
  {
    if (type == RTC_BUFFER_TYPE_GRID)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      grids.setModified();
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX)
    {
      if (slot >= vertices.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      vertices[slot].setModified();
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE)
    {
      if (slot >= vertexAttribs.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      vertexAttribs[slot].setModified();
    }
    else
    {
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "unknown buffer type");
    }

    Geometry::update();
  }

  void GridMesh::commit()
  {
    /* verify that stride of all time steps are identical */
    for (unsigned int t=0; t<numTimeSteps; t++)
      if (vertices[t].getStride() != vertices[0].getStride())
        throw_RTCError(RTC_ERROR_INVALID_OPERATION,"stride of vertex buffers have to be identical for each time step");

#if defined(EMBREE_SYCL_SUPPORT)
    
    /* build quadID_to_primID_xy mapping when hardware ray tracing is supported */
    DeviceGPU* gpu_device = dynamic_cast<DeviceGPU*>(device);
    if (gpu_device)
    {
      const size_t numQuads = getNumTotalQuads();
      quadID_to_primID_xy.resize(numQuads);
      
      for (uint32_t primID=0, quadID=0; primID<size(); primID++)
      {
        const Grid& g = grid(primID);
        for (ssize_t y=0; y<ssize_t(g.resY)-1; y++)
          for (ssize_t x=0; x<ssize_t(g.resX)-1; x++)
            quadID_to_primID_xy[quadID++] = { primID, (uint16_t) x, (uint16_t) y };
      }
    }

#endif
    
    Geometry::commit();
  }
  
  void GridMesh::addElementsToCount (GeometryCounts& counts) const 
  {
    if (numTimeSteps == 1) {
      counts.numGrids += numPrimitives;
      for (size_t primID=0; primID<numPrimitives; primID++)
        counts.numSubGrids += getNumSubGrids(primID);
    }
    else {
      counts.numMBGrids += numPrimitives;
      for (size_t primID=0; primID<numPrimitives; primID++)
        counts.numMBSubGrids += getNumSubGrids(primID);
    }
  }

  bool GridMesh::verify() 
  {
    /*! verify size of vertex arrays */
    if (vertices.size() == 0) return false;
    for (const auto& buffer : vertices)
      if (buffer.size() != numVertices())
        return false;

    /*! verify size of user vertex arrays */
    for (const auto& buffer : vertexAttribs)
      if (buffer.size() != numVertices())
        return false;

    /*! verify vertices */
    for (const auto& buffer : vertices)
      for (size_t i=0; i<buffer.size(); i++)
	if (!isvalid(buffer[i])) 
	  return false;

    return true;
  }
  
  void GridMesh::interpolate(const RTCInterpolateArguments* const args) {
    interpolate_impl<4>(args);
  }
  
#endif

  namespace isa
  {
    GridMesh* createGridMesh(Device* device) {
      return new GridMeshISA(device);
    }
  }
}
