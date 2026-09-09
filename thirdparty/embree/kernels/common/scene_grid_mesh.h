// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "geometry.h"
#include "buffer.h"

namespace embree
{
  /*! Grid Mesh */
  struct GridMesh : public Geometry
  {
    /*! type of this geometry */
    static const Geometry::GTypeMask geom_type = Geometry::MTY_GRID_MESH;

    /*! grid */
    struct Grid 
    {
      unsigned int startVtxID;
      unsigned int lineVtxOffset;
      unsigned short resX,resY;

      /* border flags due to 3x3 vertex pattern */
      __forceinline unsigned int get3x3FlagsX(const unsigned int x) const
      {
        return (x + 2 >= (unsigned int)resX) ? (1<<15) : 0;
      }

      /* border flags due to 3x3 vertex pattern */
      __forceinline unsigned int get3x3FlagsY(const unsigned int y) const
      {
        return (y + 2 >= (unsigned int)resY) ? (1<<15) : 0;
      }

      /*! outputs grid structure */
      __forceinline friend embree_ostream operator<<(embree_ostream cout, const Grid& t) {
        return cout << "Grid { startVtxID " << t.startVtxID << ", lineVtxOffset " << t.lineVtxOffset << ", resX " << t.resX << ", resY " << t.resY << " }";
      }
    };

  public:

    /*! grid mesh construction */
    GridMesh (Device* device); 

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

#if defined(EMBREE_SYCL_SUPPORT)

    size_t getGeometryDataDeviceByteSize() const;
    void convertToDeviceRepresentation(size_t offset, char* data_host, char* data_device) const;

#endif

    template<int N>
    void interpolate_impl(const RTCInterpolateArguments* const args)
    {
      unsigned int primID = args->primID;
      float U = args->u;
      float V = args->v;
      
      /* clamp input u,v to [0;1] range */
      U = max(min(U,1.0f),0.0f);
      V = max(min(V,1.0f),0.0f);
      
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
      
      const Grid& grid = grids[primID];
      const int grid_width  = grid.resX-1;
      const int grid_height = grid.resY-1;
      const float rcp_grid_width = rcp(float(grid_width));
      const float rcp_grid_height = rcp(float(grid_height));
      const int iu = min((int)floor(U*grid_width ),grid_width);
      const int iv = min((int)floor(V*grid_height),grid_height);
      const float u = U*grid_width-float(iu);
      const float v = V*grid_height-float(iv);
      
      for (unsigned int i=0; i<valueCount; i+=N)
      {
        const size_t ofs = i*sizeof(float);
        const unsigned int idx0 = grid.startVtxID + (iv+0)*grid.lineVtxOffset + iu;
        const unsigned int idx1 = grid.startVtxID + (iv+1)*grid.lineVtxOffset + iu;
        
        const vbool<N> valid = vint<N>((int)i)+vint<N>(step) < vint<N>(int(valueCount));
        const vfloat<N> p0 = mem<vfloat<N>>::loadu(valid,(float*)&src[(idx0+0)*stride+ofs]);
        const vfloat<N> p1 = mem<vfloat<N>>::loadu(valid,(float*)&src[(idx0+1)*stride+ofs]);
        const vfloat<N> p2 = mem<vfloat<N>>::loadu(valid,(float*)&src[(idx1+1)*stride+ofs]);
        const vfloat<N> p3 = mem<vfloat<N>>::loadu(valid,(float*)&src[(idx1+0)*stride+ofs]);
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
          assert(dPdu); mem<vfloat<N>>::storeu(valid,dPdu+i,select(left,Q1-Q0,Q0-Q1)*rcp_grid_width);
          assert(dPdv); mem<vfloat<N>>::storeu(valid,dPdv+i,select(left,Q2-Q0,Q0-Q2)*rcp_grid_height);
        }
        if (ddPdudu) { 
          assert(ddPdudu); mem<vfloat<N>>::storeu(valid,ddPdudu+i,vfloat<N>(zero));
          assert(ddPdvdv); mem<vfloat<N>>::storeu(valid,ddPdvdv+i,vfloat<N>(zero));
          assert(ddPdudv); mem<vfloat<N>>::storeu(valid,ddPdudv+i,vfloat<N>(zero));
        }
      }
    }

    void addElementsToCount (GeometryCounts & counts) const;
    
    __forceinline unsigned int getNumTotalQuads() const
    {
      size_t quads = 0;
      for (size_t primID=0; primID<numPrimitives; primID++)
        quads += getNumQuads(primID);
      return quads;
    }

    __forceinline unsigned int getNumQuads(const size_t gridID) const
    {
      const Grid& g = grid(gridID);
      return (unsigned int) max((int)1,((int)g.resX-1) * ((int)g.resY-1));
    }
    
    __forceinline unsigned int getNumSubGrids(const size_t gridID) const
    {
      const Grid& g = grid(gridID);
      return max((unsigned int)1,((unsigned int)g.resX >> 1) * ((unsigned int)g.resY >> 1));
    }

    /*! get fast access to first vertex buffer */
    __forceinline float * getCompactVertexArray () const {
      return (float*) vertices0.getPtr();
    }

  public:

    /*! returns number of vertices */
    __forceinline size_t numVertices() const {
      return vertices[0].size();
    }
    
    /*! returns i'th grid*/
    __forceinline const Grid& grid(size_t i) const {
      return grids[i];
    }

    /*! returns i'th vertex of the first time step  */
    __forceinline const Vec3fa vertex(size_t i) const { // FIXME: check if this does a unaligned load
      return vertices0[i];
    }

    /*! returns i'th vertex of the first time step */
    __forceinline const char* vertexPtr(size_t i) const {
      return vertices0.getPtr(i);
    }

    /*! returns i'th vertex of itime'th timestep */
    __forceinline const Vec3fa vertex(size_t i, size_t itime) const {
      return vertices[itime][i];
    }

    /*! returns i'th vertex of for specified time */
    __forceinline const Vec3fa vertex(size_t i, float time) const
    {
      float ftime;
      const size_t itime = timeSegment(time, ftime);
      const float t0 = 1.0f - ftime;
      const float t1 = ftime;
      Vec3fa v0 = vertex(i, itime+0);
      Vec3fa v1 = vertex(i, itime+1);
      return madd(Vec3fa(t0),v0,t1*v1);
    }

    /*! returns i'th vertex of itime'th timestep */
    __forceinline const char* vertexPtr(size_t i, size_t itime) const {
      return vertices[itime].getPtr(i);
    }

    /*! returns i'th vertex of the first timestep */
    __forceinline size_t grid_vertex_index(const Grid& g, size_t x, size_t y) const {
      assert(x < (size_t)g.resX);
      assert(y < (size_t)g.resY);
      return g.startVtxID + x + y * g.lineVtxOffset;
    }
    
    /*! returns i'th vertex of the first timestep */
    __forceinline const Vec3fa grid_vertex(const Grid& g, size_t x, size_t y) const {
      const size_t index = grid_vertex_index(g,x,y);
      return vertex(index);
    }

    /*! returns i'th vertex of the itime'th timestep */
    __forceinline const Vec3fa grid_vertex(const Grid& g, size_t x, size_t y, size_t itime) const {
      const size_t index = grid_vertex_index(g,x,y);
      return vertex(index,itime);
    }

    /*! returns i'th vertex of the itime'th timestep */
    __forceinline const Vec3fa grid_vertex(const Grid& g, size_t x, size_t y, float time) const {
      const size_t index = grid_vertex_index(g,x,y);
      return vertex(index,time);
    }
    
    /*! gathers quad vertices */
    __forceinline void gather_quad_vertices(Vec3fa& v0, Vec3fa& v1, Vec3fa& v2, Vec3fa& v3, const Grid& g, size_t x, size_t y) const
    {
      v0 = grid_vertex(g,x+0,y+0);
      v1 = grid_vertex(g,x+1,y+0);
      v2 = grid_vertex(g,x+1,y+1);
      v3 = grid_vertex(g,x+0,y+1);
    }
    
    /*! gathers quad vertices for specified time */
    __forceinline void gather_quad_vertices(Vec3fa& v0, Vec3fa& v1, Vec3fa& v2, Vec3fa& v3, const Grid& g, size_t x, size_t y, float time) const
    {
      v0 = grid_vertex(g,x+0,y+0,time);
      v1 = grid_vertex(g,x+1,y+0,time);
      v2 = grid_vertex(g,x+1,y+1,time);
      v3 = grid_vertex(g,x+0,y+1,time);
    }

    /*! gathers quad vertices for mblur and non-mblur meshes */
    __forceinline void gather_quad_vertices_safe(Vec3fa& v0, Vec3fa& v1, Vec3fa& v2, Vec3fa& v3, const Grid& g, size_t x, size_t y, float time) const
    {
      if (hasMotionBlur()) gather_quad_vertices(v0,v1,v2,v3,g,x,y,time);
      else                 gather_quad_vertices(v0,v1,v2,v3,g,x,y);
    }

    /*! calculates the build bounds of the i'th quad, if it's valid */
    __forceinline bool buildBoundsQuad(const Grid& g, size_t sx, size_t sy, BBox3fa& bbox) const
    {
      BBox3fa b(empty);
      for (size_t t=0; t<numTimeSteps; t++)
      {
        for (size_t y=sy;y<sy+2;y++)
          for (size_t x=sx;x<sx+2;x++)
          {
            const Vec3fa v = grid_vertex(g,x,y,t);
            if (unlikely(!isvalid(v))) return false;
            b.extend(v);
          }
      }

      bbox = b;
      return true;
    }
    
    /*! calculates the build bounds of the i'th primitive, if it's valid */
    __forceinline bool buildBounds(const Grid& g, size_t sx, size_t sy, BBox3fa& bbox) const
    {
      BBox3fa b(empty);
      for (size_t t=0; t<numTimeSteps; t++)
      {
        for (size_t y=sy;y<min(sy+3,(size_t)g.resY);y++)
          for (size_t x=sx;x<min(sx+3,(size_t)g.resX);x++)
          {
            const Vec3fa v = grid_vertex(g,x,y,t);
            if (unlikely(!isvalid(v))) return false;
            b.extend(v);
          }
      }

      bbox = b;
      return true;
    }

    /*! calculates the build bounds of the i'th primitive at the itime'th time segment, if it's valid */
    __forceinline bool buildBounds(const Grid& g, size_t sx, size_t sy, size_t itime, BBox3fa& bbox) const
    {
      assert(itime < numTimeSteps);
      BBox3fa b0(empty);
      for (size_t y=sy;y<min(sy+3,(size_t)g.resY);y++)
        for (size_t x=sx;x<min(sx+3,(size_t)g.resX);x++)
        {
          const Vec3fa v = grid_vertex(g,x,y,itime);
          if (unlikely(!isvalid(v))) return false;
          b0.extend(v);
        }

      /* use bounds of first time step in builder */
      bbox = b0;
      return true;
    }

    __forceinline bool valid(size_t gridID, size_t itime=0) const {
      return valid(gridID, make_range(itime, itime));
    }

    /*! check if the i'th primitive is valid between the specified time range */
    __forceinline bool valid(size_t gridID, const range<size_t>& itime_range) const
    {
      if (unlikely(gridID >= grids.size())) return false;
      const Grid &g = grid(gridID);
      if (unlikely(g.startVtxID + 0                                     >= vertices0.size())) return false;
      if (unlikely(g.startVtxID + (g.resY-1)*g.lineVtxOffset + g.resX-1 >= vertices0.size())) return false;

      for (size_t y=0;y<g.resY;y++)
        for (size_t x=0;x<g.resX;x++)
          for (size_t itime = itime_range.begin(); itime <= itime_range.end(); itime++)
            if (!isvalid(grid_vertex(g,x,y,itime))) return false;
      return true;
    }

    __forceinline BBox3fa bounds(const Grid& g, size_t sx, size_t sy, size_t itime) const
    {
      BBox3fa box(empty);
      buildBounds(g,sx,sy,itime,box);
      return box;
    }

    __forceinline LBBox3fa linearBounds(const Grid& g, size_t sx, size_t sy, size_t itime) const {
      BBox3fa bounds0, bounds1;
      buildBounds(g,sx,sy,itime+0,bounds0);
      buildBounds(g,sx,sy,itime+1,bounds1);
      return LBBox3fa(bounds0,bounds1);
    }

    /*! calculates the linear bounds of the i'th primitive for the specified time range */
    __forceinline LBBox3fa linearBounds(const Grid& g, size_t sx, size_t sy, const BBox1f& dt) const {
      return LBBox3fa([&] (size_t itime) { return bounds(g,sx,sy,itime); }, dt, time_range, fnumTimeSegments);
    }

    __forceinline float projectedPrimitiveArea(const size_t i) const {
      return pos_inf;
    }

  public:
    BufferView<Grid> grids;      //!< array of triangles
    BufferView<Vec3fa> vertices0;        //!< fast access to first vertex buffer
    Device::vector<BufferView<Vec3fa>> vertices = device; //!< vertex array for each timestep
    Device::vector<RawBufferView> vertexAttribs = device; //!< vertex attributes

#if defined(EMBREE_SYCL_SUPPORT)
    
  public:
    struct PrimID_XY { uint32_t primID; uint16_t x,y; };
    Device::vector<PrimID_XY> quadID_to_primID_xy = device;  //!< maps a quad to the primitive ID and grid coordinates
#endif
  };

  namespace isa
  {
    struct GridMeshISA : public GridMesh
    {
      GridMeshISA (Device* device)
        : GridMesh(device) {}

      LBBox3fa vlinearBounds(size_t buildID, const BBox1f& time_range, const SubGridBuildData * const sgrids) const override {
        const SubGridBuildData &subgrid = sgrids[buildID];                      
        const unsigned int primID = subgrid.primID;
        const size_t x = subgrid.x();
        const size_t y = subgrid.y();
        return linearBounds(grid(primID),x,y,time_range);
      }

#if defined(EMBREE_SYCL_SUPPORT)
      PrimInfo createPrimRefArray(PrimRef* prims, const range<size_t>& r, size_t k, unsigned int geomID) const override
      {
        PrimInfo pinfo(empty);
        for (size_t j=r.begin(); j<r.end(); j++)
        {
          BBox3fa bounds = empty;
          const PrimID_XY& quad = quadID_to_primID_xy[j];
          if (!buildBoundsQuad(grids[quad.primID],quad.x,quad.y,bounds)) continue;
          const PrimRef prim(bounds,geomID,unsigned(j));
          pinfo.add_center2(prim);
          prims[k++] = prim;
        }
        return pinfo;
      }
#endif
      
      PrimInfo createPrimRefArray(mvector<PrimRef>& prims, mvector<SubGridBuildData>& sgrids, const range<size_t>& r, size_t k, unsigned int geomID) const override 
      {
        PrimInfo pinfo(empty);
        for (size_t j=r.begin(); j<r.end(); j++)
        {
          if (!valid(j)) continue;
          const GridMesh::Grid &g = grid(j);
          
          for (unsigned int y=0; y<g.resY-1u; y+=2)
          {
            for (unsigned int x=0; x<g.resX-1u; x+=2)
            {
              BBox3fa bounds = empty;
              if (!buildBounds(g,x,y,bounds)) continue; // get bounds of subgrid
              const PrimRef prim(bounds,(unsigned)geomID,(unsigned)k);
              pinfo.add_center2(prim);
              sgrids[k] = SubGridBuildData(x | g.get3x3FlagsX(x), y | g.get3x3FlagsY(y), unsigned(j));
              prims[k++] = prim;                
            }
          }
        }
        return pinfo;
      }

#if defined(EMBREE_SYCL_SUPPORT)
      PrimInfo createPrimRefArrayMB(PrimRef* prims, const BBox1f& time_range, const range<size_t>& r, size_t k, unsigned int geomID) const override
      {
        const BBox1f t0t1 = BBox1f::intersect(getTimeRange(), time_range);
        PrimInfo pinfo(empty);
        for (size_t j=r.begin(); j<r.end(); j++)
        {
          const PrimID_XY& quad = quadID_to_primID_xy[j];
          const LBBox3fa lbounds = linearBounds(grids[quad.primID],quad.x,quad.y,t0t1);
          const PrimRef prim(lbounds.bounds(), unsigned(geomID), unsigned(j));
          pinfo.add_center2(prim);
          prims[k++] = prim;
        }
        return pinfo;
      }
#endif

      PrimInfoMB createPrimRefMBArray(mvector<PrimRefMB>& prims, mvector<SubGridBuildData>& sgrids, const BBox1f& t0t1, const range<size_t>& r, size_t k, unsigned int geomID) const override
      {
        PrimInfoMB pinfoMB(empty);
        for (size_t j=r.begin(); j<r.end(); j++)
        {
          if (!valid(j, timeSegmentRange(t0t1))) continue;
          const GridMesh::Grid &g = grid(j);
          
          for (unsigned int y=0; y<g.resY-1u; y+=2)
          {
            for (unsigned int x=0; x<g.resX-1u; x+=2)
            {
              const PrimRefMB prim(linearBounds(g,x,y,t0t1),numTimeSegments(),time_range,numTimeSegments(),unsigned(geomID),unsigned(k));
              pinfoMB.add_primref(prim);
              sgrids[k] = SubGridBuildData(x | g.get3x3FlagsX(x), y | g.get3x3FlagsY(y), unsigned(j));
              prims[k++] = prim;
            }
          }
        }
        return pinfoMB;
      }
    };
  }

  DECLARE_ISA_FUNCTION(GridMesh*, createGridMesh, Device*);
}
