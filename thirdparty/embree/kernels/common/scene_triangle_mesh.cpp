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

#include "scene_triangle_mesh.h"
#include "scene.h"

namespace embree
{
#if defined(EMBREE_LOWEST_ISA)

  TriangleMesh::TriangleMesh (Device* device)
    : Geometry(device,GTY_TRIANGLE_MESH,0,1)
  {
    vertices.resize(numTimeSteps);
  }

  void TriangleMesh::enabling() 
  { 
    if (numTimeSteps == 1) scene->world.numTriangles += numPrimitives;
    else                   scene->worldMB.numTriangles += numPrimitives;
  }
  
  void TriangleMesh::disabling() 
  { 
    if (numTimeSteps == 1) scene->world.numTriangles -= numPrimitives;
    else                   scene->worldMB.numTriangles -= numPrimitives;
  }

  void TriangleMesh::setMask (unsigned mask) 
  {
    this->mask = mask; 
    Geometry::update();
  }

  void TriangleMesh::setNumTimeSteps (unsigned int numTimeSteps)
  {
    vertices.resize(numTimeSteps);
    Geometry::setNumTimeSteps(numTimeSteps);
  }

  void TriangleMesh::setVertexAttributeCount (unsigned int N)
  {
    vertexAttribs.resize(N);
    Geometry::update();
  }
  
  void TriangleMesh::setBuffer(RTCBufferType type, unsigned int slot, RTCFormat format, const Ref<Buffer>& buffer, size_t offset, size_t stride, unsigned int num)
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
    else if (type == RTC_BUFFER_TYPE_INDEX)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      if (format != RTC_FORMAT_UINT3)
        throw_RTCError(RTC_ERROR_INVALID_OPERATION, "invalid index buffer format");

      triangles.set(buffer, offset, stride, num, format);
      setNumPrimitives(num);
    }
    else 
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "unknown buffer type");
  }

  void* TriangleMesh::getBuffer(RTCBufferType type, unsigned int slot)
  {
    if (type == RTC_BUFFER_TYPE_INDEX)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      return triangles.getPtr();
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

  void TriangleMesh::updateBuffer(RTCBufferType type, unsigned int slot)
  {
    if (type == RTC_BUFFER_TYPE_INDEX)
    {
      if (slot != 0)
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      triangles.setModified(true);
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX)
    {
      if (slot >= vertices.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      vertices[slot].setModified(true);
    }
    else if (type == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE)
    {
      if (slot >= vertexAttribs.size())
        throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "invalid buffer slot");
      vertexAttribs[slot].setModified(true);
    }
    else
    {
      throw_RTCError(RTC_ERROR_INVALID_ARGUMENT, "unknown buffer type");
    }

    Geometry::update();
  }

  void TriangleMesh::preCommit() 
  {
    /* verify that stride of all time steps are identical */
    for (unsigned int t=0; t<numTimeSteps; t++)
      if (vertices[t].getStride() != vertices[0].getStride())
        throw_RTCError(RTC_ERROR_INVALID_OPERATION,"stride of vertex buffers have to be identical for each time step");

    Geometry::preCommit();
  }

  void TriangleMesh::postCommit() 
  {
    scene->vertices[geomID] = (float*) vertices0.getPtr();

    triangles.setModified(false);
    for (auto& buf : vertices)
      buf.setModified(false);
    for (auto& attrib : vertexAttribs)
      attrib.setModified(false);
    
    Geometry::postCommit();
  }

  bool TriangleMesh::verify() 
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

    /*! verify triangle indices */
    for (size_t i=0; i<size(); i++) {     
      if (triangles[i].v[0] >= numVertices()) return false; 
      if (triangles[i].v[1] >= numVertices()) return false; 
      if (triangles[i].v[2] >= numVertices()) return false; 
    }

    /*! verify vertices */
    for (const auto& buffer : vertices)
      for (size_t i=0; i<buffer.size(); i++)
	if (!isvalid(buffer[i])) 
	  return false;

    return true;
  }
  
  void TriangleMesh::interpolate(const RTCInterpolateArguments* const args)
  {
    unsigned int primID = args->primID;
    float u = args->u;
    float v = args->v;
    RTCBufferType bufferType = args->bufferType;
    unsigned int bufferSlot = args->bufferSlot;
    float* P = args->P;
    float* dPdu = args->dPdu;
    float* dPdv = args->dPdv;
    float* ddPdudu = args->ddPdudu;
    float* ddPdvdv = args->ddPdvdv;
    float* ddPdudv = args->ddPdudv;
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
      const float w = 1.0f-u-v;
      const Triangle& tri = triangle(primID);
      const vbool4 valid = vint4((int)i)+vint4(step) < vint4(int(valueCount));
      const vfloat4 p0 = vfloat4::loadu(valid,(float*)&src[tri.v[0]*stride+ofs]);
      const vfloat4 p1 = vfloat4::loadu(valid,(float*)&src[tri.v[1]*stride+ofs]);
      const vfloat4 p2 = vfloat4::loadu(valid,(float*)&src[tri.v[2]*stride+ofs]);
      
      if (P) {
        vfloat4::storeu(valid,P+i,madd(w,p0,madd(u,p1,v*p2)));
      }
      if (dPdu) {
        assert(dPdu); vfloat4::storeu(valid,dPdu+i,p1-p0);
        assert(dPdv); vfloat4::storeu(valid,dPdv+i,p2-p0);
      }
      if (ddPdudu) {
        assert(ddPdudu); vfloat4::storeu(valid,ddPdudu+i,vfloat4(zero));
        assert(ddPdvdv); vfloat4::storeu(valid,ddPdvdv+i,vfloat4(zero));
        assert(ddPdudv); vfloat4::storeu(valid,ddPdudv+i,vfloat4(zero));
      }
    }
  }
  
#endif
  
  namespace isa
  {
    TriangleMesh* createTriangleMesh(Device* device) {
      return new TriangleMeshISA(device);
    }
  }
}
