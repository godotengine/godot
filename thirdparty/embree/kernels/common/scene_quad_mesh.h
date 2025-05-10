// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "geometry.h"
#include "buffer.h"

namespace embree
{
  /*! Quad Mesh */
  struct QuadMesh : public Geometry
  {
    /*! type of this geometry */
    static const Geometry::GTypeMask geom_type = Geometry::MTY_QUAD_MESH;
    
    /*! triangle indices */
    struct Quad
    {
      Quad() {}

      Quad (uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
        v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
      }

      /*! outputs triangle indices */
      __forceinline friend embree_ostream operator<<(embree_ostream cout, const Quad& q) {
        return cout << "Quad {" << q.v[0] << ", " << q.v[1] << ", " << q.v[2] << ", " << q.v[3] << " }";
      }

      uint32_t v[4];
    };

  public:

    /*! quad mesh construction */
    QuadMesh (Device* device); 
  
    /* geometry interface */
  public:
    void setMask(unsigned mask);
    void setNumTimeSteps (unsigned int numTimeSteps);
    void setVertexAttributeCount (unsigned int N);
    void setBuffer(RTCBufferType type, unsigned int slot, RTCFormat format, const Ref<Buffer>& buffer, size_t offset, size_t stride, unsigned int num);
    void* getBufferData(RTCBufferType type, unsigned int slot, BufferDataPointerType pointerType);
    void updateBuffer(RTCBufferType type, unsigned int slot);
    void commit();
    bool verify();
    void interpolate(const RTCInterpolateArguments* const args);
    void addElementsToCount (GeometryCounts & counts) const;
    size_t getGeometryDataDeviceByteSize() const;
    void convertToDeviceRepresentation(size_t offset, char* data_host, char* data_device) const;

    template<int N>
      void interpolate_impl(const RTCInterpolateArguments* const args)
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
      
      for (unsigned int i=0; i<valueCount; i+=N)
      {
        const vbool<N> valid = vint<N>((int)i)+vint<N>(step) < vint<N>(int(valueCount));
        const size_t ofs = i*sizeof(float);
        const Quad& tri = quad(primID);
        const vfloat<N> p0 = mem<vfloat<N>>::loadu(valid,(float*)&src[tri.v[0]*stride+ofs]);
        const vfloat<N> p1 = mem<vfloat<N>>::loadu(valid,(float*)&src[tri.v[1]*stride+ofs]);
        const vfloat<N> p2 = mem<vfloat<N>>::loadu(valid,(float*)&src[tri.v[2]*stride+ofs]);
        const vfloat<N> p3 = mem<vfloat<N>>::loadu(valid,(float*)&src[tri.v[3]*stride+ofs]);      
        const vbool<N> left = u+v <= 1.0f;
        const vfloat<N> Q0 = select(left,p0,p2);
        const vfloat<N> Q1 = select(left,p1,p3);
        const vfloat<N> Q2 = select(left,p3,p1);
        const vfloat<N> U  = select(left,u,vfloat<N>(1.0f)-u);
        const vfloat<N> V  = select(left,v,vfloat<N>(1.0f)-v);
        const vfloat<N> W  = 1.0f-U-V;
        if (P) {
          mem<vfloat<N>>::storeu(valid,P+i,madd(W,Q0,madd(U,Q1,V*Q2)));
        }
        if (dPdu) { 
          assert(dPdu); mem<vfloat<N>>::storeu(valid,dPdu+i,select(left,Q1-Q0,Q0-Q1));
          assert(dPdv); mem<vfloat<N>>::storeu(valid,dPdv+i,select(left,Q2-Q0,Q0-Q2));
        }
        if (ddPdudu) { 
          assert(ddPdudu); mem<vfloat<N>>::storeu(valid,ddPdudu+i,vfloat<N>(zero));
          assert(ddPdvdv); mem<vfloat<N>>::storeu(valid,ddPdvdv+i,vfloat<N>(zero));
          assert(ddPdudv); mem<vfloat<N>>::storeu(valid,ddPdudv+i,vfloat<N>(zero));
        }
      }
    }
        
  public:

    /*! returns number of vertices */
    __forceinline size_t numVertices() const {
      return vertices[0].size();
    }
    
    /*! returns i'th quad */
    __forceinline const Quad& quad(size_t i) const {
      return quads[i];
    }

    /*! returns i'th vertex of itime'th timestep */
    __forceinline const Vec3fa vertex(size_t i) const {
      return vertices0[i];
    }

    /*! returns i'th vertex of itime'th timestep */
    __forceinline const char* vertexPtr(size_t i) const {
      return vertices0.getPtr(i);
    }

    /*! returns i'th vertex of itime'th timestep */
    __forceinline const Vec3fa vertex(size_t i, size_t itime) const {
      return vertices[itime][i];
    }

    /*! returns i'th vertex of itime'th timestep */
    __forceinline const char* vertexPtr(size_t i, size_t itime) const {
      return vertices[itime].getPtr(i);
    }

    /*! returns i'th vertex of for specified time */
    __forceinline Vec3fa vertex(size_t i, float time) const
    {
      float ftime;
      const size_t itime = timeSegment(time, ftime);
      const float t0 = 1.0f - ftime;
      const float t1 = ftime;
      Vec3fa v0 = vertex(i, itime+0);
      Vec3fa v1 = vertex(i, itime+1);
      return madd(Vec3fa(t0),v0,t1*v1);
    }

    /*! calculates the bounds of the i'th quad */
    __forceinline BBox3fa bounds(size_t i) const 
    {
      const Quad& q = quad(i);
      const Vec3fa v0 = vertex(q.v[0]);
      const Vec3fa v1 = vertex(q.v[1]);
      const Vec3fa v2 = vertex(q.v[2]);
      const Vec3fa v3 = vertex(q.v[3]);
      return BBox3fa(min(v0,v1,v2,v3),max(v0,v1,v2,v3));
    }

    /*! calculates the bounds of the i'th quad at the itime'th timestep */
    __forceinline BBox3fa bounds(size_t i, size_t itime) const
    {
      const Quad& q = quad(i);
      const Vec3fa v0 = vertex(q.v[0],itime);
      const Vec3fa v1 = vertex(q.v[1],itime);
      const Vec3fa v2 = vertex(q.v[2],itime);
      const Vec3fa v3 = vertex(q.v[3],itime);
      return BBox3fa(min(v0,v1,v2,v3),max(v0,v1,v2,v3));
    }

    /*! check if the i'th primitive is valid at the itime'th timestep */
    __forceinline bool valid(size_t i, size_t itime) const {
      return valid(i, make_range(itime, itime));
    }

    /*! check if the i'th primitive is valid between the specified time range */
    __forceinline bool valid(size_t i, const range<size_t>& itime_range) const
    {
      const Quad& q = quad(i);
      if (unlikely(q.v[0] >= numVertices())) return false;
      if (unlikely(q.v[1] >= numVertices())) return false;
      if (unlikely(q.v[2] >= numVertices())) return false;
      if (unlikely(q.v[3] >= numVertices())) return false;

      for (size_t itime = itime_range.begin(); itime <= itime_range.end(); itime++)
      {
        if (!isvalid(vertex(q.v[0],itime))) return false;
        if (!isvalid(vertex(q.v[1],itime))) return false;
        if (!isvalid(vertex(q.v[2],itime))) return false;
        if (!isvalid(vertex(q.v[3],itime))) return false;
      }

      return true;
    }

    /*! calculates the linear bounds of the i'th quad at the itimeGlobal'th time segment */
    __forceinline LBBox3fa linearBounds(size_t i, size_t itime) const {
      return LBBox3fa(bounds(i,itime+0),bounds(i,itime+1));
    }

    /*! calculates the build bounds of the i'th primitive, if it's valid */
    __forceinline bool buildBounds(size_t i, BBox3fa* bbox = nullptr) const
    {
      const Quad& q = quad(i);
      if (q.v[0] >= numVertices()) return false;
      if (q.v[1] >= numVertices()) return false;
      if (q.v[2] >= numVertices()) return false;
      if (q.v[3] >= numVertices()) return false;

      for (size_t t=0; t<numTimeSteps; t++)
      {
        const Vec3fa v0 = vertex(q.v[0],t);
        const Vec3fa v1 = vertex(q.v[1],t);
        const Vec3fa v2 = vertex(q.v[2],t);
        const Vec3fa v3 = vertex(q.v[3],t);

        if (unlikely(!isvalid(v0) || !isvalid(v1) || !isvalid(v2) || !isvalid(v3)))
          return false;
      }

      if (bbox) 
        *bbox = bounds(i);

      return true;
    }

    /*! calculates the build bounds of the i'th primitive at the itime'th time segment, if it's valid */
    __forceinline bool buildBounds(size_t i, size_t itime, BBox3fa& bbox) const
    {
      const Quad& q = quad(i);
      if (unlikely(q.v[0] >= numVertices())) return false;
      if (unlikely(q.v[1] >= numVertices())) return false;
      if (unlikely(q.v[2] >= numVertices())) return false;
      if (unlikely(q.v[3] >= numVertices())) return false;

      assert(itime+1 < numTimeSteps);
      const Vec3fa a0 = vertex(q.v[0],itime+0); if (unlikely(!isvalid(a0))) return false;
      const Vec3fa a1 = vertex(q.v[1],itime+0); if (unlikely(!isvalid(a1))) return false;
      const Vec3fa a2 = vertex(q.v[2],itime+0); if (unlikely(!isvalid(a2))) return false;
      const Vec3fa a3 = vertex(q.v[3],itime+0); if (unlikely(!isvalid(a3))) return false;
      const Vec3fa b0 = vertex(q.v[0],itime+1); if (unlikely(!isvalid(b0))) return false;
      const Vec3fa b1 = vertex(q.v[1],itime+1); if (unlikely(!isvalid(b1))) return false;
      const Vec3fa b2 = vertex(q.v[2],itime+1); if (unlikely(!isvalid(b2))) return false;
      const Vec3fa b3 = vertex(q.v[3],itime+1); if (unlikely(!isvalid(b3))) return false;
      
      /* use bounds of first time step in builder */
      bbox = BBox3fa(min(a0,a1,a2,a3),max(a0,a1,a2,a3));
      return true;
    }

    /*! calculates the linear bounds of the i'th primitive for the specified time range */
    __forceinline LBBox3fa linearBounds(size_t primID, const BBox1f& dt) const {
      return LBBox3fa([&] (size_t itime) { return bounds(primID, itime); }, dt, time_range, fnumTimeSegments);
    }

    /*! calculates the linear bounds of the i'th primitive for the specified time range */
    __forceinline bool linearBounds(size_t i, const BBox1f& dt, LBBox3fa& bbox) const
    {
      if (!valid(i, timeSegmentRange(dt))) return false;
      bbox = linearBounds(i, dt);
      return true;
    }

    /*! get fast access to first vertex buffer */
    __forceinline float * getCompactVertexArray () const {
      return (float*) vertices0.getPtr();
    }

    /* gets version info of topology */
    unsigned int getTopologyVersion() const {
      return quads.modCounter;
    }
    
    /* returns true if topology changed */
    bool topologyChanged(unsigned int otherVersion) const {
      return quads.isModified(otherVersion); // || numPrimitivesChanged;
    }

    /* returns the projected area */
    __forceinline float projectedPrimitiveArea(const size_t i) const {
      const Quad& q = quad(i);
      const Vec3fa v0 = vertex(q.v[0]);
      const Vec3fa v1 = vertex(q.v[1]);
      const Vec3fa v2 = vertex(q.v[2]);
      const Vec3fa v3 = vertex(q.v[3]);
      return areaProjectedTriangle(v0,v1,v3) +
	areaProjectedTriangle(v1,v2,v3);
    }

  public:
    BufferView<Quad> quads;                 //!< array of quads
    BufferView<Vec3fa> vertices0;           //!< fast access to first vertex buffer
    Device::vector<BufferView<Vec3fa>> vertices = device; //!< vertex array for each timestep
    Device::vector<RawBufferView> vertexAttribs = device; //!< vertex attribute buffers
  };

  namespace isa
  {
    struct QuadMeshISA : public QuadMesh
    {
      QuadMeshISA (Device* device)
        : QuadMesh(device) {}

      LBBox3fa vlinearBounds(size_t primID, const BBox1f& time_range) const {
        return linearBounds(primID,time_range);
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
    };
  }

  DECLARE_ISA_FUNCTION(QuadMesh*, createQuadMesh, Device*);
}
