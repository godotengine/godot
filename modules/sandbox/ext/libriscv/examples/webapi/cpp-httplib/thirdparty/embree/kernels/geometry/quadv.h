// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "primitive.h"

namespace embree
{
  /* Stores the vertices of M quads in struct of array layout */
  template <int M>
  struct QuadMv
  { 
  public:
    struct Type : public PrimitiveType 
    {
      const char* name() const;
      size_t sizeActive(const char* This) const;
      size_t sizeTotal(const char* This) const;
      size_t getBytes(const char* This) const;
    };
    static Type type;

  public:

    /* Returns maximum number of stored quads */
    static __forceinline size_t max_size() { return M; }
    
    /* Returns required number of primitive blocks for N primitives */
    static __forceinline size_t blocks(size_t N) { return (N+max_size()-1)/max_size(); }
   
  public:

    /* Default constructor */
    __forceinline QuadMv() {}

    /* Construction from vertices and IDs */
    __forceinline QuadMv(const Vec3vf<M>& v0, const Vec3vf<M>& v1, const Vec3vf<M>& v2, const Vec3vf<M>& v3, const vuint<M>& geomIDs, const vuint<M>& primIDs)
      : v0(v0), v1(v1), v2(v2), v3(v3), geomIDs(geomIDs), primIDs(primIDs) {}
    
    /* Returns a mask that tells which quads are valid */
    __forceinline vbool<M> valid() const { return geomIDs != vuint<M>(-1); }

    /* Returns true if the specified quad is valid */
    __forceinline bool valid(const size_t i) const { assert(i<M); return geomIDs[i] != -1; }

    /* Returns the number of stored quads */
    __forceinline size_t size() const { return bsf(~movemask(valid())); }

    /* Returns the geometry IDs */
    __forceinline       vuint<M>& geomID()       { return geomIDs; }
    __forceinline const vuint<M>& geomID() const { return geomIDs; }
    __forceinline unsigned int geomID(const size_t i) const { assert(i<M); return geomIDs[i]; }

    /* Returns the primitive IDs */
    __forceinline       vuint<M> primID()       { return primIDs; }
    __forceinline const vuint<M> primID() const { return primIDs; }
    __forceinline unsigned int primID(const size_t i) const { assert(i<M); return primIDs[i]; }

    /* Calculate the bounds of the quads */
    __forceinline BBox3fa bounds() const 
    {
      Vec3vf<M> lower = min(v0,v1,v2,v3);
      Vec3vf<M> upper = max(v0,v1,v2,v3);
      vbool<M> mask = valid();
      lower.x = select(mask,lower.x,vfloat<M>(pos_inf));
      lower.y = select(mask,lower.y,vfloat<M>(pos_inf));
      lower.z = select(mask,lower.z,vfloat<M>(pos_inf));
      upper.x = select(mask,upper.x,vfloat<M>(neg_inf));
      upper.y = select(mask,upper.y,vfloat<M>(neg_inf));
      upper.z = select(mask,upper.z,vfloat<M>(neg_inf));
      return BBox3fa(Vec3fa(reduce_min(lower.x),reduce_min(lower.y),reduce_min(lower.z)),
                     Vec3fa(reduce_max(upper.x),reduce_max(upper.y),reduce_max(upper.z)));
    }
    
    /* Non temporal store */
    __forceinline static void store_nt(QuadMv* dst, const QuadMv& src)
    {
      vfloat<M>::store_nt(&dst->v0.x,src.v0.x);
      vfloat<M>::store_nt(&dst->v0.y,src.v0.y);
      vfloat<M>::store_nt(&dst->v0.z,src.v0.z);
      vfloat<M>::store_nt(&dst->v1.x,src.v1.x);
      vfloat<M>::store_nt(&dst->v1.y,src.v1.y);
      vfloat<M>::store_nt(&dst->v1.z,src.v1.z);
      vfloat<M>::store_nt(&dst->v2.x,src.v2.x);
      vfloat<M>::store_nt(&dst->v2.y,src.v2.y);
      vfloat<M>::store_nt(&dst->v2.z,src.v2.z);
      vfloat<M>::store_nt(&dst->v3.x,src.v3.x);
      vfloat<M>::store_nt(&dst->v3.y,src.v3.y);
      vfloat<M>::store_nt(&dst->v3.z,src.v3.z);
      vuint<M>::store_nt(&dst->geomIDs,src.geomIDs);
      vuint<M>::store_nt(&dst->primIDs,src.primIDs);
    }

    /* Fill quad from quad list */
    __forceinline void fill(const PrimRef* prims, size_t& begin, size_t end, Scene* scene)
    {
      vuint<M> vgeomID = -1, vprimID = -1;
      Vec3vf<M> v0 = zero, v1 = zero, v2 = zero, v3 = zero;
      
      for (size_t i=0; i<M && begin<end; i++, begin++)
      {
	const PrimRef& prim = prims[begin];
        const unsigned geomID = prim.geomID();
        const unsigned primID = prim.primID();
        const QuadMesh* __restrict__ const mesh = scene->get<QuadMesh>(geomID);
        const QuadMesh::Quad& quad = mesh->quad(primID);
        const Vec3fa& p0 = mesh->vertex(quad.v[0]);
        const Vec3fa& p1 = mesh->vertex(quad.v[1]);
        const Vec3fa& p2 = mesh->vertex(quad.v[2]);
        const Vec3fa& p3 = mesh->vertex(quad.v[3]);
        vgeomID [i] = geomID;
        vprimID [i] = primID;
        v0.x[i] = p0.x; v0.y[i] = p0.y; v0.z[i] = p0.z;
        v1.x[i] = p1.x; v1.y[i] = p1.y; v1.z[i] = p1.z;
        v2.x[i] = p2.x; v2.y[i] = p2.y; v2.z[i] = p2.z;
        v3.x[i] = p3.x; v3.y[i] = p3.y; v3.z[i] = p3.z;
      }
      QuadMv::store_nt(this,QuadMv(v0,v1,v2,v3,vgeomID,vprimID));
    }

    /* Updates the primitive */
    __forceinline BBox3fa update(QuadMesh* mesh)
    {
      BBox3fa bounds = empty;
      vuint<M> vgeomID = -1, vprimID = -1;
      Vec3vf<M> v0 = zero, v1 = zero, v2 = zero;
	
      for (size_t i=0; i<M; i++)
      {
        if (primID(i) == -1) break;
        const unsigned geomId = geomID(i);
        const unsigned primId = primID(i);
        const QuadMesh::Quad& quad = mesh->quad(primId);
        const Vec3fa p0 = mesh->vertex(quad.v[0]);
        const Vec3fa p1 = mesh->vertex(quad.v[1]);
        const Vec3fa p2 = mesh->vertex(quad.v[2]);
        const Vec3fa p3 = mesh->vertex(quad.v[3]);
        bounds.extend(merge(BBox3fa(p0),BBox3fa(p1),BBox3fa(p2),BBox3fa(p3)));
        vgeomID [i] = geomId;
        vprimID [i] = primId;
        v0.x[i] = p0.x; v0.y[i] = p0.y; v0.z[i] = p0.z;
        v1.x[i] = p1.x; v1.y[i] = p1.y; v1.z[i] = p1.z;
        v2.x[i] = p2.x; v2.y[i] = p2.y; v2.z[i] = p2.z;
        v3.x[i] = p3.x; v3.y[i] = p3.y; v3.z[i] = p3.z;
      }
      new (this) QuadMv(v0,v1,v2,v3,vgeomID,vprimID);
      return bounds;
    }
   
  public:
    Vec3vf<M> v0;      // 1st vertex of the quads
    Vec3vf<M> v1;      // 2nd vertex of the quads
    Vec3vf<M> v2;      // 3rd vertex of the quads
    Vec3vf<M> v3;      // 4th vertex of the quads
  private:
    vuint<M> geomIDs; // geometry ID
    vuint<M> primIDs; // primitive ID
  };

  template<int M>
  typename QuadMv<M>::Type QuadMv<M>::type;

  typedef QuadMv<4> Quad4v;
}
