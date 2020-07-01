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
  /*! Triangle Mesh */
  struct TriangleMesh : public Geometry
  {
    /*! type of this geometry */
    static const Geometry::GTypeMask geom_type = Geometry::MTY_TRIANGLE_MESH;

    /*! triangle indices */
    struct Triangle 
    {
      uint32_t v[3];

      /*! outputs triangle indices */
      __forceinline friend std::ostream &operator<<(std::ostream& cout, const Triangle& t) {
        return cout << "Triangle { " << t.v[0] << ", " << t.v[1] << ", " << t.v[2] << " }";
      }
    };

  public:

    /*! triangle mesh construction */
    TriangleMesh (Device* device); 

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

  public:

    /*! returns number of vertices */
    __forceinline size_t numVertices() const {
      return vertices[0].size();
    }
    
    /*! returns i'th triangle*/
    __forceinline const Triangle& triangle(size_t i) const {
      return triangles[i];
    }

    /*! returns i'th vertex of the first time step  */
    __forceinline const Vec3fa vertex(size_t i) const {
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

    /*! calculates the bounds of the i'th triangle */
    __forceinline BBox3fa bounds(size_t i) const 
    {
      const Triangle& tri = triangle(i);
      const Vec3fa v0 = vertex(tri.v[0]);
      const Vec3fa v1 = vertex(tri.v[1]);
      const Vec3fa v2 = vertex(tri.v[2]);
      return BBox3fa(min(v0,v1,v2),max(v0,v1,v2));
    }

    /*! calculates the bounds of the i'th triangle at the itime'th timestep */
    __forceinline BBox3fa bounds(size_t i, size_t itime) const
    {
      const Triangle& tri = triangle(i);
      const Vec3fa v0 = vertex(tri.v[0],itime);
      const Vec3fa v1 = vertex(tri.v[1],itime);
      const Vec3fa v2 = vertex(tri.v[2],itime);
      return BBox3fa(min(v0,v1,v2),max(v0,v1,v2));
    }

    /*! calculates the interpolated bounds of the i'th triangle at the specified time */
    __forceinline BBox3fa bounds(size_t i, float time) const
    {
      float ftime; size_t itime = getTimeSegment(time, fnumTimeSegments, ftime);
      const BBox3fa b0 = bounds(i, itime+0);
      const BBox3fa b1 = bounds(i, itime+1);
      return lerp(b0, b1, ftime);
    }

    /*! check if the i'th primitive is valid at the itime'th timestep */
    __forceinline bool valid(size_t i, size_t itime) const {
      return valid(i, make_range(itime, itime));
    }

    /*! check if the i'th primitive is valid between the specified time range */
    __forceinline bool valid(size_t i, const range<size_t>& itime_range) const
    {
      const Triangle& tri = triangle(i);
      if (unlikely(tri.v[0] >= numVertices())) return false;
      if (unlikely(tri.v[1] >= numVertices())) return false;
      if (unlikely(tri.v[2] >= numVertices())) return false;

      for (size_t itime = itime_range.begin(); itime <= itime_range.end(); itime++)
      {
        if (!isvalid(vertex(tri.v[0],itime))) return false;
        if (!isvalid(vertex(tri.v[1],itime))) return false;
        if (!isvalid(vertex(tri.v[2],itime))) return false;
      }

      return true;
    }

    /*! calculates the linear bounds of the i'th primitive at the itimeGlobal'th time segment */
    __forceinline LBBox3fa linearBounds(size_t i, size_t itime) const {
      return LBBox3fa(bounds(i,itime+0),bounds(i,itime+1));
    }

    /*! calculates the build bounds of the i'th primitive, if it's valid */
    __forceinline bool buildBounds(size_t i, BBox3fa* bbox = nullptr) const
    {
      const Triangle& tri = triangle(i);
      if (unlikely(tri.v[0] >= numVertices())) return false;
      if (unlikely(tri.v[1] >= numVertices())) return false;
      if (unlikely(tri.v[2] >= numVertices())) return false;

      for (size_t t=0; t<numTimeSteps; t++)
      {
        const Vec3fa v0 = vertex(tri.v[0],t);
        const Vec3fa v1 = vertex(tri.v[1],t);
        const Vec3fa v2 = vertex(tri.v[2],t);
        if (unlikely(!isvalid(v0) || !isvalid(v1) || !isvalid(v2)))
          return false;
      }

      if (likely(bbox)) 
        *bbox = bounds(i);

      return true;
    }

    /*! calculates the build bounds of the i'th primitive at the itime'th time segment, if it's valid */
    __forceinline bool buildBounds(size_t i, size_t itime, BBox3fa& bbox) const
    {
      const Triangle& tri = triangle(i);
      if (unlikely(tri.v[0] >= numVertices())) return false;
      if (unlikely(tri.v[1] >= numVertices())) return false;
      if (unlikely(tri.v[2] >= numVertices())) return false;

      assert(itime+1 < numTimeSteps);
      const Vec3fa a0 = vertex(tri.v[0],itime+0); if (unlikely(!isvalid(a0))) return false;
      const Vec3fa a1 = vertex(tri.v[1],itime+0); if (unlikely(!isvalid(a1))) return false;
      const Vec3fa a2 = vertex(tri.v[2],itime+0); if (unlikely(!isvalid(a2))) return false;
      const Vec3fa b0 = vertex(tri.v[0],itime+1); if (unlikely(!isvalid(b0))) return false;
      const Vec3fa b1 = vertex(tri.v[1],itime+1); if (unlikely(!isvalid(b1))) return false;
      const Vec3fa b2 = vertex(tri.v[2],itime+1); if (unlikely(!isvalid(b2))) return false;
      
      /* use bounds of first time step in builder */
      bbox = BBox3fa(min(a0,a1,a2),max(a0,a1,a2));
      return true;
    }

    /*! calculates the linear bounds of the i'th primitive for the specified time range */
    __forceinline LBBox3fa linearBounds(size_t primID, const BBox1f& dt) const {
      return LBBox3fa([&] (size_t itime) { return bounds(primID, itime); }, dt, time_range, fnumTimeSegments);
    }

    /*! calculates the linear bounds of the i'th primitive for the specified time range */
    __forceinline bool linearBounds(size_t i, const BBox1f& dt, LBBox3fa& bbox) const  {
      if (!valid(i, timeSegmentRange(dt))) return false;
      bbox = linearBounds(i, dt);
      return true;
    }

    /* returns true if topology changed */
    bool topologyChanged() const {
      return triangles.isModified() || numPrimitivesChanged;
    }

  public:
    BufferView<Triangle> triangles;      //!< array of triangles
    BufferView<Vec3fa> vertices0;        //!< fast access to first vertex buffer
    vector<BufferView<Vec3fa>> vertices; //!< vertex array for each timestep
    vector<RawBufferView> vertexAttribs; //!< vertex attributes
  };

  namespace isa
  {
    struct TriangleMeshISA : public TriangleMesh
    {
      TriangleMeshISA (Device* device)
        : TriangleMesh(device) {}

      PrimInfo createPrimRefArray(mvector<PrimRef>& prims, const range<size_t>& r, size_t k) const
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

      PrimInfo createPrimRefArrayMB(mvector<PrimRef>& prims, size_t itime, const range<size_t>& r, size_t k) const
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
      
      PrimInfoMB createPrimRefMBArray(mvector<PrimRefMB>& prims, const BBox1f& t0t1, const range<size_t>& r, size_t k) const
      {
        PrimInfoMB pinfo(empty);
        for (size_t j=r.begin(); j<r.end(); j++)
        {
          if (!valid(j, timeSegmentRange(t0t1))) continue;
          const PrimRefMB prim(linearBounds(j,t0t1),this->numTimeSegments(),this->time_range,this->numTimeSegments(),this->geomID,unsigned(j));
          pinfo.add_primref(prim);
          prims[k++] = prim;
        }
        return pinfo;
      }
    };
  }

  DECLARE_ISA_FUNCTION(TriangleMesh*, createTriangleMesh, Device*);
}
