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
      __forceinline friend std::ostream &operator<<(std::ostream& cout, const Grid& t) {
        return cout << "Grid { startVtxID " << t.startVtxID << ", lineVtxOffset " << t.lineVtxOffset << ", resX " << t.resX << ", resY " << t.resY << " }";
      }
    };

  public:

    /*! grid mesh construction */
    GridMesh (Device* device); 

    /* geometry interface */
  public:
    void enabling();
    void disabling();
    void setMask(unsigned mask);
    void setNumTimeSteps (unsigned int numTimeSteps);
    void setVertexAttributeCount (unsigned int N);
    void setBuffer(RTCBufferType type, unsigned int slot, RTCFormat format, const Ref<Buffer>& buffer, size_t offset, size_t stride, unsigned int num);
    void* getBuffer(RTCBufferType type, unsigned int slot);
    void updateBuffer(RTCBufferType type, unsigned int slot);
    void preCommit();
    void postCommit();
    bool verify();
    void interpolate(const RTCInterpolateArguments* const args);

    __forceinline unsigned int getNumSubGrids(const size_t gridID)
    {
      const Grid &g = grid(gridID);
      return max((unsigned int)1,((unsigned int)g.resX >> 1) * ((unsigned int)g.resY >> 1));
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

    /* returns true if topology changed */
    bool topologyChanged() const {
      return grids.isModified() || numPrimitivesChanged;
    }

  public:
    BufferView<Grid> grids;      //!< array of triangles
    BufferView<Vec3fa> vertices0;        //!< fast access to first vertex buffer
    vector<BufferView<Vec3fa>> vertices; //!< vertex array for each timestep
    vector<RawBufferView> vertexAttribs; //!< vertex attributes
  };

  namespace isa
  {
    struct GridMeshISA : public GridMesh
    {
      GridMeshISA (Device* device)
        : GridMesh(device) {}
    };
  }

  DECLARE_ISA_FUNCTION(GridMesh*, createGridMesh, Device*);
}
