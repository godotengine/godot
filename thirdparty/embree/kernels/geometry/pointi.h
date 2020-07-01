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

#include "primitive.h"

namespace embree
{
  template<int M>
  struct PointMi
  {
    /* Virtual interface to query information about the line segment type */
    struct Type : public PrimitiveType
    {
      const char* name() const;
      size_t sizeActive(const char* This) const;
      size_t sizeTotal(const char* This) const;
      size_t getBytes(const char* This) const;
    };
    static Type type;

   public:
    /* primitive supports multiple time segments */
    static const bool singleTimeSegment = false;

    /* Returns maximum number of stored line segments */
    static __forceinline size_t max_size()
    {
      return M;
    }

    /* Returns required number of primitive blocks for N line segments */
    static __forceinline size_t blocks(size_t N)
    {
      return (N + max_size() - 1) / max_size();
    }

    /* Returns required number of bytes for N line segments */
    static __forceinline size_t bytes(size_t N)
    {
      return blocks(N) * sizeof(PointMi);
    }

   public:
    /* Default constructor */
    __forceinline PointMi() {}

    /* Construction from vertices and IDs */
    __forceinline PointMi(const vuint<M>& geomIDs, const vuint<M>& primIDs, Geometry::GType gtype, uint32_t numPrimitives)
        : gtype((unsigned char)gtype),
          numPrimitives(numPrimitives),
          sharedGeomID(geomIDs[0]),
          primIDs(primIDs)
    {
      assert(all(vuint<M>(geomID()) == geomIDs));
    }

    /* Returns a mask that tells which line segments are valid */
    __forceinline vbool<M> valid() const
    {
      //return primIDs != vuint<M>(-1);
      return vint<M>(step) < vint<M>(numPrimitives);
    }

    /* Returns a mask that tells which line segments are valid */
    template<int Mx>
    __forceinline vbool<Mx> valid() const
    {
      //return vuint<Mx>(primIDs) != vuint<Mx>(-1);
      return vint<Mx>(step) < vint<Mx>(numPrimitives);
    }

    /* Returns if the specified line segment is valid */
    __forceinline bool valid(const size_t i) const
    {
      assert(i < M);
      //return primIDs[i] != -1;
      return i < numPrimitives;
    }

    /* Returns the number of stored line segments */
    __forceinline size_t size() const
    {
      //return bsf(~movemask(valid()));
      return numPrimitives;
    }

    /* Returns the geometry IDs */
    // template<class T>
    // static __forceinline T unmask(T &index) { return index & 0x3fffffff; }

    __forceinline unsigned int geomID(unsigned int i = 0) const
    {
      return sharedGeomID;
    }
    //__forceinline       vuint<M> geomID()       { return unmask(geomIDs); }
    //__forceinline const vuint<M> geomID() const { return unmask(geomIDs); }
    //__forceinline unsigned int geomID(const size_t i) const { assert(i<M); return unmask(geomIDs[i]); }

    /* Returns the primitive IDs */
    __forceinline vuint<M>& primID()
    {
      return primIDs;
    }
    __forceinline const vuint<M>& primID() const
    {
      return primIDs;
    }
    __forceinline unsigned int primID(const size_t i) const
    {
      assert(i < M);
      return primIDs[i];
    }

    /* gather the line segments */
    __forceinline void gather(Vec4vf<M>& p0, const Scene* scene) const;
    __forceinline void gather(Vec4vf<M>& p0, Vec3vf<M>& n0, const Scene* scene) const;

    __forceinline void gather(Vec4vf<M>& p0, const Points* geom, const vint<M>& itime) const;
    __forceinline void gather(Vec4vf<M>& p0, Vec3vf<M>& n0, const Points* geom, const vint<M>& itime) const;

    __forceinline void gather(Vec4vf<M>& p0, const Scene* scene, float time) const;

    __forceinline void gather(Vec4vf<M>& p0, Vec3vf<M>& n0, const Scene* scene, float time) const;

    /* Calculate the bounds of the line segments */
    __forceinline const BBox3fa bounds(const Scene* scene, size_t itime = 0) const
    {
      BBox3fa bounds = empty;
      for (size_t i = 0; i < M && valid(i); i++) {
        const Points* geom = scene->get<Points>(geomID(i));
        const Vec3fa& p0   = geom->vertex(primID(i), itime);
        BBox3fa b          = enlarge(BBox3fa(p0), Vec3fa(p0.w));
        bounds.extend(b);
      }
      return bounds;
    }

    /* Calculate the linear bounds of the primitive */
    __forceinline LBBox3fa linearBounds(const Scene* scene, size_t itime)
    {
      return LBBox3fa(bounds(scene, itime + 0), bounds(scene, itime + 1));
    }

    __forceinline LBBox3fa linearBounds(const Scene* const scene, size_t itime, size_t numTimeSteps)
    {
      LBBox3fa allBounds = empty;
      for (size_t i = 0; i < M && valid(i); i++) {
        const Points* geom = scene->get<Points>(geomID(i));
        allBounds.extend(geom->linearBounds(primID(i), itime, numTimeSteps));
      }
      return allBounds;
    }

    __forceinline LBBox3fa linearBounds(const Scene* const scene, const BBox1f time_range)
    {
      LBBox3fa allBounds = empty;
      for (size_t i = 0; i < M && valid(i); i++) {
        const Points* geom = scene->get<Points>(geomID((unsigned int)i));
        allBounds.extend(geom->linearBounds(primID(i), time_range));
      }
      return allBounds;
    }

    /* Fill line segment from line segment list */
    template<typename PrimRefT>
    __forceinline void fill(const PrimRefT* prims, size_t& begin, size_t end, Scene* scene)
    {
      Geometry::GType gty = scene->get(prims[begin].geomID())->getType();
      vuint<M> geomID, primID;
      vuint<M> v0;
      const PrimRefT* prim = &prims[begin];

      int numPrimitives = 0;
      for (size_t i = 0; i < M; i++) {
        if (begin < end) {
          geomID[i] = prim->geomID();
          primID[i] = prim->primID();
          begin++;
          numPrimitives++;
        } else {
          assert(i);
          if (i > 0) {
            geomID[i] = geomID[i - 1];
            primID[i] = primID[i - 1];
          }
        }
        if (begin < end)
          prim = &prims[begin];  // FIXME: remove this line
      }
      new (this) PointMi(geomID, primID, gty, numPrimitives);  // FIXME: use non temporal store
    }

    template<typename BVH, typename Allocator>
    __forceinline static typename BVH::NodeRef createLeaf(BVH* bvh,
                                                          const PrimRef* prims,
                                                          const range<size_t>& set,
                                                          const Allocator& alloc)
    {
      size_t start    = set.begin();
      size_t items    = PointMi::blocks(set.size());
      size_t numbytes = PointMi::bytes(set.size());
      PointMi* accel  = (PointMi*)alloc.malloc1(numbytes, M * sizeof(float));
      for (size_t i = 0; i < items; i++) {
        accel[i].fill(prims, start, set.end(), bvh->scene);
      }
      return bvh->encodeLeaf((char*)accel, items);
    };

    __forceinline LBBox3fa fillMB(const PrimRef* prims, size_t& begin, size_t end, Scene* scene, size_t itime)
    {
      fill(prims, begin, end, scene);
      return linearBounds(scene, itime);
    }

    __forceinline LBBox3fa fillMB(
        const PrimRefMB* prims, size_t& begin, size_t end, Scene* scene, const BBox1f time_range)
    {
      fill(prims, begin, end, scene);
      return linearBounds(scene, time_range);
    }

    template<typename BVH, typename SetMB, typename Allocator>
    __forceinline static typename BVH::NodeRecordMB4D createLeafMB(BVH* bvh, const SetMB& prims, const Allocator& alloc)
    {
      size_t start                     = prims.object_range.begin();
      size_t end                       = prims.object_range.end();
      size_t items                     = PointMi::blocks(prims.object_range.size());
      size_t numbytes                  = PointMi::bytes(prims.object_range.size());
      PointMi* accel                   = (PointMi*)alloc.malloc1(numbytes, M * sizeof(float));
      const typename BVH::NodeRef node = bvh->encodeLeaf((char*)accel, items);

      LBBox3fa bounds = empty;
      for (size_t i = 0; i < items; i++)
        bounds.extend(accel[i].fillMB(prims.prims->data(), start, end, bvh->scene, prims.time_range));

      return typename BVH::NodeRecordMB4D(node, bounds, prims.time_range);
    };

    /* Updates the primitive */
    __forceinline BBox3fa update(Points* geom)
    {
      BBox3fa bounds = empty;
      for (size_t i = 0; i < M && valid(i); i++) {
        const Vec3fa& p0 = geom->vertex(primID(i));
        BBox3fa b        = enlarge(BBox3fa(p0), Vec3fa(p0.w));
        bounds.extend(b);
      }
      return bounds;
    }

    /*! output operator */
    friend __forceinline std::ostream& operator<<(std::ostream& cout, const PointMi& line)
    {
      return cout << "Line" << M << "i {" << line.v0 << ", " << line.geomID() << ", " << line.primID() << "}";
    }

   public:
    unsigned char gtype;
    unsigned char numPrimitives;
    unsigned int sharedGeomID;

   private:
    vuint<M> primIDs;  // primitive ID
  };

  template<>
  __forceinline void PointMi<4>::gather(Vec4vf4& p0, const Scene* scene) const
  {
    const Points* geom = scene->get<Points>(geomID());
    const vfloat4 a0   = vfloat4::loadu(geom->vertexPtr(primID(0)));
    const vfloat4 a1   = vfloat4::loadu(geom->vertexPtr(primID(1)));
    const vfloat4 a2   = vfloat4::loadu(geom->vertexPtr(primID(2)));
    const vfloat4 a3   = vfloat4::loadu(geom->vertexPtr(primID(3)));
    transpose(a0, a1, a2, a3, p0.x, p0.y, p0.z, p0.w);
  }

  template<>
  __forceinline void PointMi<4>::gather(Vec4vf4& p0, Vec3vf4& n0, const Scene* scene) const
  {
    const Points* geom = scene->get<Points>(geomID());
    const vfloat4 a0   = vfloat4::loadu(geom->vertexPtr(primID(0)));
    const vfloat4 a1   = vfloat4::loadu(geom->vertexPtr(primID(1)));
    const vfloat4 a2   = vfloat4::loadu(geom->vertexPtr(primID(2)));
    const vfloat4 a3   = vfloat4::loadu(geom->vertexPtr(primID(3)));
    transpose(a0, a1, a2, a3, p0.x, p0.y, p0.z, p0.w);
    const vfloat4 b0 = vfloat4(geom->normal(primID(0)));
    const vfloat4 b1 = vfloat4(geom->normal(primID(1)));
    const vfloat4 b2 = vfloat4(geom->normal(primID(2)));
    const vfloat4 b3 = vfloat4(geom->normal(primID(3)));
    transpose(b0, b1, b2, b3, n0.x, n0.y, n0.z);
  }

  template<>
  __forceinline void PointMi<4>::gather(Vec4vf4& p0, const Points* geom, const vint4& itime) const
  {
    const vfloat4 a0 = vfloat4::loadu(geom->vertexPtr(primID(0), itime[0]));
    const vfloat4 a1 = vfloat4::loadu(geom->vertexPtr(primID(1), itime[1]));
    const vfloat4 a2 = vfloat4::loadu(geom->vertexPtr(primID(2), itime[2]));
    const vfloat4 a3 = vfloat4::loadu(geom->vertexPtr(primID(3), itime[3]));
    transpose(a0, a1, a2, a3, p0.x, p0.y, p0.z, p0.w);
  }

  template<>
  __forceinline void PointMi<4>::gather(Vec4vf4& p0, Vec3vf4& n0, const Points* geom, const vint4& itime) const
  {
    const vfloat4 a0 = vfloat4::loadu(geom->vertexPtr(primID(0), itime[0]));
    const vfloat4 a1 = vfloat4::loadu(geom->vertexPtr(primID(1), itime[1]));
    const vfloat4 a2 = vfloat4::loadu(geom->vertexPtr(primID(2), itime[2]));
    const vfloat4 a3 = vfloat4::loadu(geom->vertexPtr(primID(3), itime[3]));
    transpose(a0, a1, a2, a3, p0.x, p0.y, p0.z, p0.w);
    const vfloat4 b0 = vfloat4(geom->normal(primID(0), itime[0]));
    const vfloat4 b1 = vfloat4(geom->normal(primID(1), itime[1]));
    const vfloat4 b2 = vfloat4(geom->normal(primID(2), itime[2]));
    const vfloat4 b3 = vfloat4(geom->normal(primID(3), itime[3]));
    transpose(b0, b1, b2, b3, n0.x, n0.y, n0.z);
  }

  template<>
  __forceinline void PointMi<4>::gather(Vec4vf4& p0, const Scene* scene, float time) const
  {
    const Points* geom = scene->get<Points>(geomID());
    const vfloat4 numTimeSegments(geom->fnumTimeSegments);
    vfloat4 ftime;
    const vint4 itime = getTimeSegment(vfloat4(time), numTimeSegments, ftime);

    Vec4vf4 a0;
    gather(a0, geom, itime);
    Vec4vf4 b0;
    gather(b0, geom, itime + 1);
    p0 = lerp(a0, b0, ftime);
  }

  template<>
  __forceinline void PointMi<4>::gather(Vec4vf4& p0, Vec3vf4& n0, const Scene* scene, float time) const
  {
    const Points* geom = scene->get<Points>(geomID());
    const vfloat4 numTimeSegments(geom->fnumTimeSegments);
    vfloat4 ftime;
    const vint4 itime = getTimeSegment(vfloat4(time), numTimeSegments, ftime);

    Vec4vf4 a0, b0;
    Vec3vf4 norm0, norm1;
    gather(a0, norm0, geom, itime);
    gather(b0, norm1, geom, itime + 1);
    p0 = lerp(a0, b0, ftime);
    n0 = lerp(norm0, norm1, ftime);
  }

#if defined(__AVX__)

  template<>
  __forceinline void PointMi<8>::gather(Vec4vf8& p0, const Scene* scene) const
  {
    const Points* geom = scene->get<Points>(geomID());

    const vfloat4 a0 = vfloat4::loadu(geom->vertexPtr(primID(0)));
    const vfloat4 a1 = vfloat4::loadu(geom->vertexPtr(primID(1)));
    const vfloat4 a2 = vfloat4::loadu(geom->vertexPtr(primID(2)));
    const vfloat4 a3 = vfloat4::loadu(geom->vertexPtr(primID(3)));
    const vfloat4 a4 = vfloat4::loadu(geom->vertexPtr(primID(4)));
    const vfloat4 a5 = vfloat4::loadu(geom->vertexPtr(primID(5)));
    const vfloat4 a6 = vfloat4::loadu(geom->vertexPtr(primID(6)));
    const vfloat4 a7 = vfloat4::loadu(geom->vertexPtr(primID(7)));
    transpose(a0, a1, a2, a3, a4, a5, a6, a7, p0.x, p0.y, p0.z, p0.w);
  }

  template<>
  __forceinline void PointMi<8>::gather(Vec4vf8& p0, Vec3vf8& n0, const Scene* scene) const
  {
    const Points* geom = scene->get<Points>(geomID());

    const vfloat4 a0 = vfloat4::loadu(geom->vertexPtr(primID(0)));
    const vfloat4 a1 = vfloat4::loadu(geom->vertexPtr(primID(1)));
    const vfloat4 a2 = vfloat4::loadu(geom->vertexPtr(primID(2)));
    const vfloat4 a3 = vfloat4::loadu(geom->vertexPtr(primID(3)));
    const vfloat4 a4 = vfloat4::loadu(geom->vertexPtr(primID(4)));
    const vfloat4 a5 = vfloat4::loadu(geom->vertexPtr(primID(5)));
    const vfloat4 a6 = vfloat4::loadu(geom->vertexPtr(primID(6)));
    const vfloat4 a7 = vfloat4::loadu(geom->vertexPtr(primID(7)));
    transpose(a0, a1, a2, a3, a4, a5, a6, a7, p0.x, p0.y, p0.z, p0.w);
    const vfloat4 b0 = vfloat4(geom->normal(primID(0)));
    const vfloat4 b1 = vfloat4(geom->normal(primID(1)));
    const vfloat4 b2 = vfloat4(geom->normal(primID(2)));
    const vfloat4 b3 = vfloat4(geom->normal(primID(3)));
    const vfloat4 b4 = vfloat4(geom->normal(primID(4)));
    const vfloat4 b5 = vfloat4(geom->normal(primID(5)));
    const vfloat4 b6 = vfloat4(geom->normal(primID(6)));
    const vfloat4 b7 = vfloat4(geom->normal(primID(7)));
    transpose(b0, b1, b2, b3, b4, b5, b6, b7, n0.x, n0.y, n0.z);
  }

  template<>
  __forceinline void PointMi<8>::gather(Vec4vf8& p0, const Points* geom, const vint8& itime) const
  {
    const vfloat4 a0 = vfloat4::loadu(geom->vertexPtr(primID(0), itime[0]));
    const vfloat4 a1 = vfloat4::loadu(geom->vertexPtr(primID(1), itime[1]));
    const vfloat4 a2 = vfloat4::loadu(geom->vertexPtr(primID(2), itime[2]));
    const vfloat4 a3 = vfloat4::loadu(geom->vertexPtr(primID(3), itime[3]));
    const vfloat4 a4 = vfloat4::loadu(geom->vertexPtr(primID(4), itime[4]));
    const vfloat4 a5 = vfloat4::loadu(geom->vertexPtr(primID(5), itime[5]));
    const vfloat4 a6 = vfloat4::loadu(geom->vertexPtr(primID(6), itime[6]));
    const vfloat4 a7 = vfloat4::loadu(geom->vertexPtr(primID(7), itime[7]));
    transpose(a0, a1, a2, a3, a4, a5, a6, a7, p0.x, p0.y, p0.z, p0.w);
  }

  template<>
  __forceinline void PointMi<8>::gather(Vec4vf8& p0, Vec3vf8& n0, const Points* geom, const vint8& itime) const
  {
    const vfloat4 a0 = vfloat4::loadu(geom->vertexPtr(primID(0), itime[0]));
    const vfloat4 a1 = vfloat4::loadu(geom->vertexPtr(primID(1), itime[1]));
    const vfloat4 a2 = vfloat4::loadu(geom->vertexPtr(primID(2), itime[2]));
    const vfloat4 a3 = vfloat4::loadu(geom->vertexPtr(primID(3), itime[3]));
    const vfloat4 a4 = vfloat4::loadu(geom->vertexPtr(primID(4), itime[4]));
    const vfloat4 a5 = vfloat4::loadu(geom->vertexPtr(primID(5), itime[5]));
    const vfloat4 a6 = vfloat4::loadu(geom->vertexPtr(primID(6), itime[6]));
    const vfloat4 a7 = vfloat4::loadu(geom->vertexPtr(primID(7), itime[7]));
    transpose(a0, a1, a2, a3, a4, a5, a6, a7, p0.x, p0.y, p0.z, p0.w);
    const vfloat4 b0 = vfloat4(geom->normal(primID(0), itime[0]));
    const vfloat4 b1 = vfloat4(geom->normal(primID(1), itime[1]));
    const vfloat4 b2 = vfloat4(geom->normal(primID(2), itime[2]));
    const vfloat4 b3 = vfloat4(geom->normal(primID(3), itime[3]));
    const vfloat4 b4 = vfloat4(geom->normal(primID(4), itime[4]));
    const vfloat4 b5 = vfloat4(geom->normal(primID(5), itime[5]));
    const vfloat4 b6 = vfloat4(geom->normal(primID(6), itime[6]));
    const vfloat4 b7 = vfloat4(geom->normal(primID(7), itime[7]));
    transpose(b0, b1, b2, b3, b4, b5, b6, b7, n0.x, n0.y, n0.z);
  }

  template<>
  __forceinline void PointMi<8>::gather(Vec4vf8& p0, const Scene* scene, float time) const
  {
    const Points* geom = scene->get<Points>(geomID());
    const vfloat8 numTimeSegments(geom->fnumTimeSegments);
    vfloat8 ftime;
    const vint8 itime = getTimeSegment(vfloat8(time), numTimeSegments, ftime);

    Vec4vf8 a0;
    gather(a0, geom, itime);
    Vec4vf8 b0;
    gather(b0, geom, itime + 1);
    p0 = lerp(a0, b0, ftime);
  }

  template<>
  __forceinline void PointMi<8>::gather(Vec4vf8& p0, Vec3vf8& n0, const Scene* scene, float time) const
  {
    const Points* geom = scene->get<Points>(geomID());
    const vfloat8 numTimeSegments(geom->fnumTimeSegments);
    vfloat8 ftime;
    const vint8 itime = getTimeSegment(vfloat8(time), numTimeSegments, ftime);

    Vec4vf8 a0, b0;
    Vec3vf8 norm0, norm1;
    gather(a0, norm0, geom, itime);
    gather(b0, norm1, geom, itime + 1);
    p0 = lerp(a0, b0, ftime);
    n0 = lerp(norm0, norm1, ftime);
  }
#endif

  template<int M>
  typename PointMi<M>::Type PointMi<M>::type;

  typedef PointMi<4> Point4i;
  typedef PointMi<8> Point8i;
}  // namespace embree
