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

#include "buffer.h"
#include "default.h"
#include "geometry.h"

namespace embree
{
  /*! represents an array of points */
  struct Points : public Geometry
  {
    /*! type of this geometry */
    static const Geometry::GTypeMask geom_type = Geometry::MTY_POINTS;

   public:
    /*! line segments construction */
    Points(Device* device, Geometry::GType gtype);

   public:
    void enabling();
    void disabling();
    void setMask(unsigned mask);
    void setNumTimeSteps(unsigned int numTimeSteps);
    void setVertexAttributeCount(unsigned int N);
    void setBuffer(RTCBufferType type,
                   unsigned int slot,
                   RTCFormat format,
                   const Ref<Buffer>& buffer,
                   size_t offset,
                   size_t stride,
                   unsigned int num);
    void* getBuffer(RTCBufferType type, unsigned int slot);
    void updateBuffer(RTCBufferType type, unsigned int slot);
    void preCommit();
    void postCommit();
    bool verify();

   public:
    /*! returns the number of vertices */
    __forceinline size_t numVertices() const
    {
      return vertices[0].size();
    }

    /*! returns i'th vertex of the first time step */
    __forceinline Vec3fa vertex(size_t i) const
    {
      return vertices0[i];
    }

    /*! returns i'th vertex of the first time step */
    __forceinline const char* vertexPtr(size_t i) const
    {
      return vertices0.getPtr(i);
    }

    /*! returns i'th normal of the first time step */
    __forceinline Vec3fa normal(size_t i) const
    {
      return normals0[i];
    }

    /*! returns i'th radius of the first time step */
    __forceinline float radius(size_t i) const
    {
      return vertices0[i].w;
    }

    /*! returns i'th vertex of itime'th timestep */
    __forceinline Vec3fa vertex(size_t i, size_t itime) const
    {
      return vertices[itime][i];
    }

    /*! returns i'th vertex of itime'th timestep */
    __forceinline const char* vertexPtr(size_t i, size_t itime) const
    {
      return vertices[itime].getPtr(i);
    }

    /*! returns i'th normal of itime'th timestep */
    __forceinline Vec3fa normal(size_t i, size_t itime) const
    {
      return normals[itime][i];
    }

    /*! returns i'th radius of itime'th timestep */
    __forceinline float radius(size_t i, size_t itime) const
    {
      return vertices[itime][i].w;
    }

    /*! calculates bounding box of i'th line segment */
    __forceinline BBox3fa bounds(const Vec3fa& v0) const
    {
      return enlarge(BBox3fa(v0), Vec3fa(v0.w));
    }

    /*! calculates bounding box of i'th line segment */
    __forceinline BBox3fa bounds(size_t i) const
    {
      const Vec3fa v0 = vertex(i);
      return bounds(v0);
    }

    /*! calculates bounding box of i'th line segment for the itime'th time step */
    __forceinline BBox3fa bounds(size_t i, size_t itime) const
    {
      const Vec3fa v0 = vertex(i, itime);
      return bounds(v0);
    }

    /*! calculates bounding box of i'th line segment */
    __forceinline BBox3fa bounds(const LinearSpace3fa& space, size_t i) const
    {
      const Vec3fa v0 = vertex(i);
      const Vec3fa w0(xfmVector(space, v0), v0.w);
      return bounds(w0);
    }

    /*! calculates bounding box of i'th line segment for the itime'th time step */
    __forceinline BBox3fa bounds(const LinearSpace3fa& space, size_t i, size_t itime) const
    {
      const Vec3fa v0 = vertex(i, itime);
      const Vec3fa w0(xfmVector(space, v0), v0.w);
      return bounds(w0);
    }

    /*! check if the i'th primitive is valid at the itime'th timestep */
    __forceinline bool valid(size_t i, size_t itime) const
    {
      return valid(i, make_range(itime, itime));
    }

    /*! check if the i'th primitive is valid between the specified time range */
    __forceinline bool valid(size_t i, const range<size_t>& itime_range) const
    {
      const unsigned int index = (unsigned int)i;
      if (index >= numVertices())
        return false;

      for (size_t itime = itime_range.begin(); itime <= itime_range.end(); itime++) {
        const Vec3fa v0 = vertex(index + 0, itime);
        if (unlikely(!isvalid((vfloat4)v0)))
          return false;
        if (v0.w < 0.0f)
          return false;
      }
      return true;
    }

    /*! calculates the linear bounds of the i'th primitive at the itimeGlobal'th time segment */
    __forceinline LBBox3fa linearBounds(size_t i, size_t itime) const
    {
      return LBBox3fa(bounds(i, itime + 0), bounds(i, itime + 1));
    }

    /*! calculates the build bounds of the i'th primitive, if it's valid */
    __forceinline bool buildBounds(size_t i, BBox3fa* bbox) const
    {
      if (!valid(i, 0))
        return false;
      *bbox = bounds(i);
      return true;
    }

    /*! calculates the build bounds of the i'th primitive at the itime'th time segment, if it's valid */
    __forceinline bool buildBounds(size_t i, size_t itime, BBox3fa& bbox) const
    {
      if (!valid(i, itime + 0) || !valid(i, itime + 1))
        return false;
      bbox = bounds(i, itime);  // use bounds of first time step in builder
      return true;
    }

    /*! calculates the linear bounds of the i'th primitive for the specified time range */
    __forceinline LBBox3fa linearBounds(size_t primID, const BBox1f& time_range) const
    {
      return LBBox3fa([&](size_t itime) { return bounds(primID, itime); }, time_range, fnumTimeSegments);
    }

    /*! calculates the linear bounds of the i'th primitive for the specified time range */
    __forceinline LBBox3fa linearBounds(const LinearSpace3fa& space, size_t primID, const BBox1f& time_range) const
    {
      return LBBox3fa([&](size_t itime) { return bounds(space, primID, itime); }, time_range, fnumTimeSegments);
    }

    /*! calculates the linear bounds of the i'th primitive for the specified time range */
    __forceinline bool linearBounds(size_t i, const BBox1f& time_range, LBBox3fa& bbox) const
    {
      if (!valid(i, getTimeSegmentRange(time_range, fnumTimeSegments)))
        return false;
      bbox = linearBounds(i, time_range);
      return true;
    }

    /* returns true if topology changed */
    bool topologyChanged() const
    {
      return numPrimitivesChanged;
    }

   public:
    BufferView<Vec3fa> vertices0;            //!< fast access to first vertex buffer
    BufferView<Vec3fa> normals0;             //!< fast access to first normal buffer
    vector<BufferView<Vec3fa>> vertices;     //!< vertex array for each timestep
    vector<BufferView<Vec3fa>> normals;      //!< normal array for each timestep
    vector<BufferView<char>> vertexAttribs;  //!< user buffers
  };

  namespace isa
  {
    struct PointsISA : public Points
    {
      PointsISA(Device* device, Geometry::GType gtype) : Points(device, gtype) {}

      Vec3fa computeDirection(unsigned int primID) const
      {
        return Vec3fa(1, 0, 0);
      }

      Vec3fa computeDirection(unsigned int primID, size_t time) const
      {
        return Vec3fa(1, 0, 0);
      }

      PrimInfo createPrimRefArray(mvector<PrimRef>& prims, const range<size_t>& r, size_t k) const
      {
        PrimInfo pinfo(empty);
        for (size_t j = r.begin(); j < r.end(); j++) {
          BBox3fa bounds = empty;
          if (!buildBounds(j, &bounds))
            continue;
          const PrimRef prim(bounds, geomID, unsigned(j));
          pinfo.add_center2(prim);
          prims[k++] = prim;
        }
        return pinfo;
      }

      PrimInfo createPrimRefArrayMB(mvector<PrimRef>& prims, size_t itime, const range<size_t>& r, size_t k) const
      {
        PrimInfo pinfo(empty);
        for (size_t j = r.begin(); j < r.end(); j++) {
          BBox3fa bounds = empty;
          if (!buildBounds(j, itime, bounds))
            continue;
          const PrimRef prim(bounds, geomID, unsigned(j));
          pinfo.add_center2(prim);
          prims[k++] = prim;
        }
        return pinfo;
      }

      PrimInfoMB createPrimRefMBArray(mvector<PrimRefMB>& prims,
                                      const BBox1f& t0t1,
                                      const range<size_t>& r,
                                      size_t k) const
      {
        PrimInfoMB pinfo(empty);
        for (size_t j = r.begin(); j < r.end(); j++) {
          if (!valid(j, getTimeSegmentRange(t0t1, fnumTimeSegments)))
            continue;
          const PrimRefMB prim(linearBounds(j, t0t1),
                               this->numTimeSegments(),
                               this->time_range,
                               this->numTimeSegments(),
                               this->geomID,
                               unsigned(j));
          pinfo.add_primref(prim);
          prims[k++] = prim;
        }
        return pinfo;
      }

      BBox3fa vbounds(size_t i) const
      {
        return bounds(i);
      }

      BBox3fa vbounds(const LinearSpace3fa& space, size_t i) const
      {
        return bounds(space, i);
      }

      LBBox3fa vlinearBounds(size_t primID, const BBox1f& time_range) const
      {
        return linearBounds(primID, time_range);
      }

      LBBox3fa vlinearBounds(const LinearSpace3fa& space, size_t primID, const BBox1f& time_range) const
      {
        return linearBounds(space, primID, time_range);
      }
    };
  }  // namespace isa

  DECLARE_ISA_FUNCTION(Points*, createPoints, Device* COMMA Geometry::GType);
}  // namespace embree
