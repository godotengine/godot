// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "primitive.h"
#include "../common/scene.h"

namespace embree
{
  /* Stores M triangles from an indexed face set */
  template <int M>
  struct TriangleMi
  {
    /* Virtual interface to query information about the triangle type */
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

    /* Returns maximum number of stored triangles */
    static __forceinline size_t max_size() { return M; }

    /* Returns required number of primitive blocks for N primitives */
    static __forceinline size_t blocks(size_t N) { return (N+max_size()-1)/max_size(); }

  public:

    /* Default constructor */
    __forceinline TriangleMi() {  }

    /* Construction from vertices and IDs */
    __forceinline TriangleMi(const vuint<M>& v0,
                             const vuint<M>& v1,
                             const vuint<M>& v2,
                             const vuint<M>& geomIDs,
                             const vuint<M>& primIDs)
#if defined(EMBREE_COMPACT_POLYS)
      : geomIDs(geomIDs), primIDs(primIDs) {}
#else
    : v0_(v0), v1_(v1), v2_(v2), geomIDs(geomIDs), primIDs(primIDs) {}
#endif

    /* Returns a mask that tells which triangles are valid */
    __forceinline vbool<M> valid() const { return primIDs != vuint<M>(-1); }

    /* Returns if the specified triangle is valid */
    __forceinline bool valid(const size_t i) const { assert(i<M); return primIDs[i] != -1; }

    /* Returns the number of stored triangles */
    __forceinline size_t size() const { return bsf(~movemask(valid())); }

    /* Returns the geometry IDs */
    __forceinline vuint<M> geomID() const { return geomIDs; }
    __forceinline unsigned int geomID(const size_t i) const { assert(i<M); return geomIDs[i]; }

    /* Returns the primitive IDs */
    __forceinline vuint<M> primID() const { return primIDs; }
    __forceinline unsigned int primID(const size_t i) const { assert(i<M); return primIDs[i]; }

    /* Calculate the bounds of the triangles */
    __forceinline const BBox3fa bounds(const Scene *const scene, const size_t itime=0) const
    {
      BBox3fa bounds = empty;
      for (size_t i=0; i<M && valid(i); i++) {
        const TriangleMesh* mesh = scene->get<TriangleMesh>(geomID(i));
        bounds.extend(mesh->bounds(primID(i),itime));
      }
      return bounds;
    }

    /* Calculate the linear bounds of the primitive */
    __forceinline LBBox3fa linearBounds(const Scene *const scene, size_t itime) {
      return LBBox3fa(bounds(scene,itime+0),bounds(scene,itime+1));
    }

    __forceinline LBBox3fa linearBounds(const Scene *const scene, size_t itime, size_t numTimeSteps)
    {
      LBBox3fa allBounds = empty;
      for (size_t i=0; i<M && valid(i); i++)
      {
        const TriangleMesh* mesh = scene->get<TriangleMesh>(geomID(i));
        allBounds.extend(mesh->linearBounds(primID(i), itime, numTimeSteps));
      }
      return allBounds;
    }

    __forceinline LBBox3fa linearBounds(const Scene *const scene, const BBox1f time_range)
    {
      LBBox3fa allBounds = empty;
      for (size_t i=0; i<M && valid(i); i++)
      {
        const TriangleMesh* mesh = scene->get<TriangleMesh>(geomID(i));
        allBounds.extend(mesh->linearBounds(primID(i), time_range));
      }
      return allBounds;
    }
    
    /* Non-temporal store */
    __forceinline static void store_nt(TriangleMi* dst, const TriangleMi& src)
    {
#if !defined(EMBREE_COMPACT_POLYS)
      vuint<M>::store_nt(&dst->v0_,src.v0_);
      vuint<M>::store_nt(&dst->v1_,src.v1_);
      vuint<M>::store_nt(&dst->v2_,src.v2_);
#endif
      vuint<M>::store_nt(&dst->geomIDs,src.geomIDs);
      vuint<M>::store_nt(&dst->primIDs,src.primIDs);
    }

    /* Fill triangle from triangle list */
    template<typename PrimRefT>
    __forceinline void fill(const PrimRefT* prims, size_t& begin, size_t end, Scene* scene)
    {
      vuint<M> v0 = zero, v1 = zero, v2 = zero;
      vuint<M> geomID = -1, primID = -1;
      const PrimRefT* prim = &prims[begin];

      for (size_t i=0; i<M; i++)
      {
        if (begin<end) {
          geomID[i] = prim->geomID();
          primID[i] = prim->primID();
#if !defined(EMBREE_COMPACT_POLYS)
          const TriangleMesh* mesh = scene->get<TriangleMesh>(prim->geomID());
          const TriangleMesh::Triangle& tri = mesh->triangle(prim->primID());
          unsigned int int_stride = mesh->vertices0.getStride()/4;
          v0[i] = tri.v[0] * int_stride;
          v1[i] = tri.v[1] * int_stride;
          v2[i] = tri.v[2] * int_stride;
#endif
          begin++;
        } else {
          assert(i);
          if (likely(i > 0)) {
            geomID[i] = geomID[0];
            primID[i] = -1;
            v0[i] = v0[0];
            v1[i] = v0[0];
            v2[i] = v0[0];
          }
        }
        if (begin<end) prim = &prims[begin];
      }
      new (this) TriangleMi(v0,v1,v2,geomID,primID); // FIXME: use non temporal store
    }

    __forceinline LBBox3fa fillMB(const PrimRef* prims, size_t& begin, size_t end, Scene* scene, size_t itime)
    {
      fill(prims, begin, end, scene);
      return linearBounds(scene, itime);
    }

    __forceinline LBBox3fa fillMB(const PrimRefMB* prims, size_t& begin, size_t end, Scene* scene, const BBox1f time_range)
    {
      fill(prims, begin, end, scene);
      return linearBounds(scene, time_range);
    }

    /* Updates the primitive */
    __forceinline BBox3fa update(TriangleMesh* mesh)
    {
      BBox3fa bounds = empty;
      for (size_t i=0; i<M; i++)
      {
        if (primID(i) == -1) break;
        const unsigned int primId = primID(i);
        const TriangleMesh::Triangle& tri = mesh->triangle(primId);
        const Vec3fa p0 = mesh->vertex(tri.v[0]);
        const Vec3fa p1 = mesh->vertex(tri.v[1]);
        const Vec3fa p2 = mesh->vertex(tri.v[2]);
        bounds.extend(merge(BBox3fa(p0),BBox3fa(p1),BBox3fa(p2)));
      }
      return bounds;
    }

  protected:
#if !defined(EMBREE_COMPACT_POLYS)
    vuint<M> v0_;         // 4 byte offset of 1st vertex
    vuint<M> v1_;         // 4 byte offset of 2nd vertex
    vuint<M> v2_;         // 4 byte offset of 3rd vertex
#endif
    vuint<M> geomIDs;    // geometry ID of mesh
    vuint<M> primIDs;    // primitive ID of primitive inside mesh
  };

  namespace isa
  {
    
  template<int M>
    struct TriangleMi : public embree::TriangleMi<M>
  {
#if !defined(EMBREE_COMPACT_POLYS)
    using embree::TriangleMi<M>::v0_;
    using embree::TriangleMi<M>::v1_;
    using embree::TriangleMi<M>::v2_;
#endif
    using embree::TriangleMi<M>::geomIDs;
    using embree::TriangleMi<M>::primIDs;
    using embree::TriangleMi<M>::geomID;
    using embree::TriangleMi<M>::primID;
    using embree::TriangleMi<M>::valid;
        
    /* loads a single vertex */
    template<int vid>
    __forceinline Vec3f getVertex(const size_t index, const Scene *const scene) const
    {
#if defined(EMBREE_COMPACT_POLYS)
      const TriangleMesh* mesh = scene->get<TriangleMesh>(geomID(index));
      const TriangleMesh::Triangle& tri = mesh->triangle(primID(index));
      return (Vec3f) mesh->vertices[0][tri.v[vid]];
#else
      const vuint<M>& v = getVertexOffset<vid>();
      const float* vertices = scene->vertices[geomID(index)];
      return (Vec3f&) vertices[v[index]];
#endif
    }

    template<int vid, typename T>
    __forceinline Vec3<T> getVertex(const size_t index, const Scene *const scene, const size_t itime, const T& ftime) const
    {
#if defined(EMBREE_COMPACT_POLYS)
      const TriangleMesh* mesh = scene->get<TriangleMesh>(geomID(index));
      const TriangleMesh::Triangle& tri = mesh->triangle(primID(index));
      const Vec3fa v0 = mesh->vertices[itime+0][tri.v[vid]];
      const Vec3fa v1 = mesh->vertices[itime+1][tri.v[vid]];
#else
      const vuint<M>& v = getVertexOffset<vid>();
      const TriangleMesh* mesh = scene->get<TriangleMesh>(geomID(index));
      const float* vertices0 = (const float*) mesh->vertexPtr(0,itime+0);
      const float* vertices1 = (const float*) mesh->vertexPtr(0,itime+1);
      const Vec3fa v0 = Vec3fa::loadu(vertices0+v[index]);
      const Vec3fa v1 = Vec3fa::loadu(vertices1+v[index]);
#endif
      const Vec3<T> p0(v0.x,v0.y,v0.z);
      const Vec3<T> p1(v1.x,v1.y,v1.z);
      return lerp(p0,p1,ftime);
    }

    template<int vid, int K, typename T>
    __forceinline Vec3<T> getVertex(const vbool<K>& valid, const size_t index, const Scene *const scene, const vint<K>& itime, const T& ftime) const
    {
      Vec3<T> p0, p1;
      const TriangleMesh* mesh = scene->get<TriangleMesh>(geomID(index));
      
      for (size_t mask=movemask(valid), i=bsf(mask); mask; mask=btc(mask,i), i=bsf(mask))
      {
#if defined(EMBREE_COMPACT_POLYS)
        const TriangleMesh::Triangle& tri = mesh->triangle(primID(index));
        const Vec3fa v0 = mesh->vertices[itime[i]+0][tri.v[vid]];
        const Vec3fa v1 = mesh->vertices[itime[i]+1][tri.v[vid]];
#else
        const vuint<M>& v = getVertexOffset<vid>();
        const float* vertices0 = (const float*) mesh->vertexPtr(0,itime[i]+0);
        const float* vertices1 = (const float*) mesh->vertexPtr(0,itime[i]+1);
        const Vec3fa v0 = Vec3fa::loadu(vertices0+v[index]);
        const Vec3fa v1 = Vec3fa::loadu(vertices1+v[index]);
#endif
        p0.x[i] = v0.x; p0.y[i] = v0.y; p0.z[i] = v0.z;
        p1.x[i] = v1.x; p1.y[i] = v1.y; p1.z[i] = v1.z;
      }
      return (T(one)-ftime)*p0 + ftime*p1;
    }

    struct Triangle {
      vfloat4 v0,v1,v2;
    };
    
#if defined(EMBREE_COMPACT_POLYS)
    
    __forceinline Triangle loadTriangle(const int i, const Scene* const scene) const 
    {
      const unsigned int geomID = geomIDs[i];
      const unsigned int primID = primIDs[i];
      if (unlikely(primID == -1)) return { zero, zero, zero };
      const TriangleMesh* mesh = scene->get<TriangleMesh>(geomID);
      const TriangleMesh::Triangle& tri = mesh->triangle(primID);
      const vfloat4 v0 = (vfloat4) mesh->vertices0[tri.v[0]];
      const vfloat4 v1 = (vfloat4) mesh->vertices0[tri.v[1]];
      const vfloat4 v2 = (vfloat4) mesh->vertices0[tri.v[2]];
      return { v0, v1, v2 };
    }

    __forceinline Triangle loadTriangle(const int i, const int itime, const TriangleMesh* const mesh) const 
    {
      const unsigned int primID = primIDs[i];
      if (unlikely(primID == -1)) return { zero, zero, zero };
      const TriangleMesh::Triangle& tri = mesh->triangle(primID);
      const vfloat4 v0 = (vfloat4) mesh->vertices[itime][tri.v[0]];
      const vfloat4 v1 = (vfloat4) mesh->vertices[itime][tri.v[1]];
      const vfloat4 v2 = (vfloat4) mesh->vertices[itime][tri.v[2]];
      return { v0, v1, v2 };
    }
    
#else

    __forceinline Triangle loadTriangle(const int i, const Scene* const scene) const 
    {
      const float* vertices = scene->vertices[geomID(i)];
      const vfloat4 v0 = vfloat4::loadu(vertices + v0_[i]);
      const vfloat4 v1 = vfloat4::loadu(vertices + v1_[i]);
      const vfloat4 v2 = vfloat4::loadu(vertices + v2_[i]);
      return { v0, v1, v2 };
    }

    __forceinline Triangle loadTriangle(const int i, const int itime, const TriangleMesh* const mesh) const 
    {
      const float* vertices = (const float*) mesh->vertexPtr(0,itime);
      const vfloat4 v0 = vfloat4::loadu(vertices + v0_[i]);
      const vfloat4 v1 = vfloat4::loadu(vertices + v1_[i]);
      const vfloat4 v2 = vfloat4::loadu(vertices + v2_[i]);
      return { v0, v1, v2 };
    }
    
#endif

    /* Gather the triangles */
    __forceinline void gather(Vec3vf<M>& p0, Vec3vf<M>& p1, Vec3vf<M>& p2, const Scene* const scene) const;

    template<int K>
#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER < 2000) // workaround for compiler bug in ICC 2019
    __noinline
#else
    __forceinline
#endif
    void gather(const vbool<K>& valid,
                Vec3vf<K>& p0,
                Vec3vf<K>& p1,
                Vec3vf<K>& p2,
                const size_t index,
                const Scene* const scene,
                const vfloat<K>& time) const
    {
      const TriangleMesh* mesh = scene->get<TriangleMesh>(geomID(index));

      vfloat<K> ftime;
      const vint<K> itime = mesh->timeSegment<K>(time, ftime);

      const size_t first = bsf(movemask(valid));
      if (likely(all(valid,itime[first] == itime)))
      {
        p0 = getVertex<0>(index, scene, itime[first], ftime);
        p1 = getVertex<1>(index, scene, itime[first], ftime);
        p2 = getVertex<2>(index, scene, itime[first], ftime);
      } else {
        p0 = getVertex<0,K>(valid, index, scene, itime, ftime);
        p1 = getVertex<1,K>(valid, index, scene, itime, ftime);
        p2 = getVertex<2,K>(valid, index, scene, itime, ftime);
      }
    }

    __forceinline void gather(Vec3vf<M>& p0,
                              Vec3vf<M>& p1,
                              Vec3vf<M>& p2,
                              const TriangleMesh* mesh,
                              const Scene *const scene,
                              const int itime) const;

    __forceinline void gather(Vec3vf<M>& p0,
                              Vec3vf<M>& p1,
                              Vec3vf<M>& p2,
                              const Scene *const scene,
                              const float time) const;


#if !defined(EMBREE_COMPACT_POLYS)
    template<int N> const vuint<M>& getVertexOffset() const;
#endif
  };

#if !defined(EMBREE_COMPACT_POLYS)
  template<> template<> __forceinline const vuint<4>& TriangleMi<4>::getVertexOffset<0>() const { return v0_; }
  template<> template<> __forceinline const vuint<4>& TriangleMi<4>::getVertexOffset<1>() const { return v1_; }
  template<> template<> __forceinline const vuint<4>& TriangleMi<4>::getVertexOffset<2>() const { return v2_; }
#endif
  
  template<>
  __forceinline void TriangleMi<4>::gather(Vec3vf4& p0,
                                           Vec3vf4& p1,
                                           Vec3vf4& p2,
                                           const Scene* const scene) const
  {
    const Triangle tri0 = loadTriangle(0,scene);
    const Triangle tri1 = loadTriangle(1,scene);
    const Triangle tri2 = loadTriangle(2,scene);
    const Triangle tri3 = loadTriangle(3,scene);
    transpose(tri0.v0,tri1.v0,tri2.v0,tri3.v0,p0.x,p0.y,p0.z);
    transpose(tri0.v1,tri1.v1,tri2.v1,tri3.v1,p1.x,p1.y,p1.z);
    transpose(tri0.v2,tri1.v2,tri2.v2,tri3.v2,p2.x,p2.y,p2.z);
  }

  template<>
  __forceinline void TriangleMi<4>::gather(Vec3vf4& p0,
                                           Vec3vf4& p1,
                                           Vec3vf4& p2,
                                           const TriangleMesh* mesh,
                                           const Scene *const scene,
                                           const int itime) const
  {
    const Triangle tri0 = loadTriangle(0,itime,mesh);
    const Triangle tri1 = loadTriangle(1,itime,mesh);
    const Triangle tri2 = loadTriangle(2,itime,mesh);
    const Triangle tri3 = loadTriangle(3,itime,mesh);
    transpose(tri0.v0,tri1.v0,tri2.v0,tri3.v0,p0.x,p0.y,p0.z);
    transpose(tri0.v1,tri1.v1,tri2.v1,tri3.v1,p1.x,p1.y,p1.z);
    transpose(tri0.v2,tri1.v2,tri2.v2,tri3.v2,p2.x,p2.y,p2.z);
  }

  template<>
  __forceinline void TriangleMi<4>::gather(Vec3vf4& p0,
                                           Vec3vf4& p1,
                                           Vec3vf4& p2,
                                           const Scene *const scene,
                                           const float time) const
  {
    const TriangleMesh* mesh = scene->get<TriangleMesh>(geomID(0)); // in mblur mode all geometries are identical

    float ftime;
    const int itime = mesh->timeSegment(time, ftime);

    Vec3vf4 a0,a1,a2; gather(a0,a1,a2,mesh,scene,itime);
    Vec3vf4 b0,b1,b2; gather(b0,b1,b2,mesh,scene,itime+1);
    p0 = lerp(a0,b0,vfloat4(ftime));
    p1 = lerp(a1,b1,vfloat4(ftime));
    p2 = lerp(a2,b2,vfloat4(ftime));
  }
  }

  template<int M>
  typename TriangleMi<M>::Type TriangleMi<M>::type;

  typedef TriangleMi<4> Triangle4i;
}
