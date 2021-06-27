// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "primitive.h"
#include "curve_intersector_precalculations.h"

namespace embree
{
  template<int M>
    struct CurveNi
  {
    struct Type : public PrimitiveType {
      const char* name() const;
      size_t sizeActive(const char* This) const;
      size_t sizeTotal(const char* This) const;
      size_t getBytes(const char* This) const;
    };
    static Type type;

  public:

    /* Returns maximum number of stored primitives */
    static __forceinline size_t max_size() { return M; }

    /* Returns required number of primitive blocks for N primitives */
    static __forceinline size_t blocks(size_t N) { return (N+M-1)/M; }

    static __forceinline size_t bytes(size_t N)
    {
      const size_t f = N/M, r = N%M;
      static_assert(sizeof(CurveNi) == 22+25*M, "internal data layout issue");
      return f*sizeof(CurveNi) + (r!=0)*(22 + 25*r);
    }

  public:

    /*! Default constructor. */
    __forceinline CurveNi () {}

    /*! fill curve from curve list */
    __forceinline void fill(const PrimRef* prims, size_t& begin, size_t _end, Scene* scene)
    {  
      size_t end = min(begin+M,_end);
      N = (unsigned char)(end-begin);
      const unsigned int geomID0 = prims[begin].geomID();
      this->geomID(N) = geomID0;
      ty = (unsigned char) scene->get(geomID0)->getType();

      /* encode all primitives */
      BBox3fa bounds = empty;
      for (size_t i=0; i<N; i++)
      {
        const PrimRef& prim = prims[begin+i];
        const unsigned int geomID = prim.geomID(); assert(geomID == geomID0);
        const unsigned int primID = prim.primID();
        bounds.extend(scene->get(geomID)->vbounds(primID));
      }

      /* calculate offset and scale */
      Vec3fa loffset = bounds.lower;
      float lscale = reduce_min(256.0f/(bounds.size()*sqrt(3.0f)));
      if (bounds.size() == Vec3fa(zero)) lscale = 0.0f;
      *this->offset(N) = loffset;
      *this->scale(N) = lscale;
      
      /* encode all primitives */
      for (size_t i=0; i<M && begin<end; i++, begin++)
      {
        const PrimRef& prim = prims[begin];
        const unsigned int geomID = prim.geomID();
        const unsigned int primID = prim.primID();
        const LinearSpace3fa space2 = scene->get(geomID)->computeAlignedSpace(primID);
        
        const LinearSpace3fa space3(trunc(126.0f*space2.vx),trunc(126.0f*space2.vy),trunc(126.0f*space2.vz));
        const BBox3fa bounds = scene->get(geomID)->vbounds(loffset,lscale,max(length(space3.vx),length(space3.vy),length(space3.vz)),space3.transposed(),primID);
        
        bounds_vx_x(N)[i] = (char) space3.vx.x;
        bounds_vx_y(N)[i] = (char) space3.vx.y;
        bounds_vx_z(N)[i] = (char) space3.vx.z;
        bounds_vx_lower(N)[i] = (short) clamp(floor(bounds.lower.x),-32767.0f,32767.0f);
        bounds_vx_upper(N)[i] = (short) clamp(ceil (bounds.upper.x),-32767.0f,32767.0f);
        assert(-32767.0f <= floor(bounds.lower.x) && floor(bounds.lower.x) <= 32767.0f);
        assert(-32767.0f <= ceil (bounds.upper.x) && ceil (bounds.upper.x) <= 32767.0f);

        bounds_vy_x(N)[i] = (char) space3.vy.x;
        bounds_vy_y(N)[i] = (char) space3.vy.y;
        bounds_vy_z(N)[i] = (char) space3.vy.z;
        bounds_vy_lower(N)[i] = (short) clamp(floor(bounds.lower.y),-32767.0f,32767.0f);
        bounds_vy_upper(N)[i] = (short) clamp(ceil (bounds.upper.y),-32767.0f,32767.0f);
        assert(-32767.0f <= floor(bounds.lower.y) && floor(bounds.lower.y) <= 32767.0f);
        assert(-32767.0f <= ceil (bounds.upper.y) && ceil (bounds.upper.y) <= 32767.0f);

        bounds_vz_x(N)[i] = (char) space3.vz.x;
        bounds_vz_y(N)[i] = (char) space3.vz.y;
        bounds_vz_z(N)[i] = (char) space3.vz.z;
        bounds_vz_lower(N)[i] = (short) clamp(floor(bounds.lower.z),-32767.0f,32767.0f);
        bounds_vz_upper(N)[i] = (short) clamp(ceil (bounds.upper.z),-32767.0f,32767.0f);
        assert(-32767.0f <= floor(bounds.lower.z) && floor(bounds.lower.z) <= 32767.0f);
        assert(-32767.0f <= ceil (bounds.upper.z) && ceil (bounds.upper.z) <= 32767.0f);
               
        this->primID(N)[i] = primID;
      }
    }

    template<typename BVH, typename Allocator>
      __forceinline static typename BVH::NodeRef createLeaf (BVH* bvh, const PrimRef* prims, const range<size_t>& set, const Allocator& alloc)
    {
      size_t start = set.begin();
      size_t items = CurveNi::blocks(set.size());
      size_t numbytes = CurveNi::bytes(set.size());
      CurveNi* accel = (CurveNi*) alloc.malloc1(numbytes,BVH::byteAlignment);
      for (size_t i=0; i<items; i++) {
        accel[i].fill(prims,start,set.end(),bvh->scene);
      }
      return bvh->encodeLeaf((char*)accel,items);
    };
    
  public:
    
    // 27.6 - 46 bytes per primitive
    unsigned char ty;
    unsigned char N;
    unsigned char data[4+25*M+16];

    /*
    struct Layout
    {
      unsigned int geomID;
      unsigned int primID[N];
      
      char bounds_vx_x[N];
      char bounds_vx_y[N];
      char bounds_vx_z[N];
      short bounds_vx_lower[N];
      short bounds_vx_upper[N];
      
      char bounds_vy_x[N];
      char bounds_vy_y[N];
      char bounds_vy_z[N];
      short bounds_vy_lower[N];
      short bounds_vy_upper[N];
      
      char bounds_vz_x[N];
      char bounds_vz_y[N];
      char bounds_vz_z[N];
      short bounds_vz_lower[N];
      short bounds_vz_upper[N];
      
      Vec3f offset;
      float scale;
    };
    */
    
    __forceinline       unsigned int& geomID(size_t N)       { return *(unsigned int*)((char*)this+2); }
    __forceinline const unsigned int& geomID(size_t N) const { return *(unsigned int*)((char*)this+2); }
    
    __forceinline       unsigned int* primID(size_t N)       { return (unsigned int*)((char*)this+6); }
    __forceinline const unsigned int* primID(size_t N) const { return (unsigned int*)((char*)this+6); }
    
    __forceinline       char* bounds_vx_x(size_t N)       { return (char*)((char*)this+6+4*N); }
    __forceinline const char* bounds_vx_x(size_t N) const { return (char*)((char*)this+6+4*N); }
    
    __forceinline       char* bounds_vx_y(size_t N)       { return (char*)((char*)this+6+5*N); }
    __forceinline const char* bounds_vx_y(size_t N) const { return (char*)((char*)this+6+5*N); }
    
    __forceinline       char* bounds_vx_z(size_t N)       { return (char*)((char*)this+6+6*N); }
    __forceinline const char* bounds_vx_z(size_t N) const { return (char*)((char*)this+6+6*N); }
    
    __forceinline       short* bounds_vx_lower(size_t N)       { return (short*)((char*)this+6+7*N); }
    __forceinline const short* bounds_vx_lower(size_t N) const { return (short*)((char*)this+6+7*N); }
    
    __forceinline       short* bounds_vx_upper(size_t N)       { return (short*)((char*)this+6+9*N); }
    __forceinline const short* bounds_vx_upper(size_t N) const { return (short*)((char*)this+6+9*N); }
    
    __forceinline       char* bounds_vy_x(size_t N)       { return (char*)((char*)this+6+11*N); }
    __forceinline const char* bounds_vy_x(size_t N) const { return (char*)((char*)this+6+11*N); }
    
    __forceinline       char* bounds_vy_y(size_t N)       { return (char*)((char*)this+6+12*N); }
    __forceinline const char* bounds_vy_y(size_t N) const { return (char*)((char*)this+6+12*N); }
    
    __forceinline       char* bounds_vy_z(size_t N)       { return (char*)((char*)this+6+13*N); }
    __forceinline const char* bounds_vy_z(size_t N) const { return (char*)((char*)this+6+13*N); }
    
    __forceinline       short* bounds_vy_lower(size_t N)       { return (short*)((char*)this+6+14*N); }
    __forceinline const short* bounds_vy_lower(size_t N) const { return (short*)((char*)this+6+14*N); }
    
    __forceinline       short* bounds_vy_upper(size_t N)       { return (short*)((char*)this+6+16*N); }
    __forceinline const short* bounds_vy_upper(size_t N) const { return (short*)((char*)this+6+16*N); }
    
    __forceinline       char* bounds_vz_x(size_t N)       { return (char*)((char*)this+6+18*N); }
    __forceinline const char* bounds_vz_x(size_t N) const { return (char*)((char*)this+6+18*N); }
    
    __forceinline       char* bounds_vz_y(size_t N)       { return (char*)((char*)this+6+19*N); }
    __forceinline const char* bounds_vz_y(size_t N) const { return (char*)((char*)this+6+19*N); }
    
    __forceinline       char* bounds_vz_z(size_t N)       { return (char*)((char*)this+6+20*N); }
    __forceinline const char* bounds_vz_z(size_t N) const { return (char*)((char*)this+6+20*N); }
    
    __forceinline       short* bounds_vz_lower(size_t N)       { return (short*)((char*)this+6+21*N); }
    __forceinline const short* bounds_vz_lower(size_t N) const { return (short*)((char*)this+6+21*N); }
    
    __forceinline       short* bounds_vz_upper(size_t N)       { return (short*)((char*)this+6+23*N); }
    __forceinline const short* bounds_vz_upper(size_t N) const { return (short*)((char*)this+6+23*N); }
    
    __forceinline       Vec3f* offset(size_t N)       { return (Vec3f*)((char*)this+6+25*N); }
    __forceinline const Vec3f* offset(size_t N) const { return (Vec3f*)((char*)this+6+25*N); }
    
    __forceinline       float* scale(size_t N)       { return (float*)((char*)this+6+25*N+12); }
    __forceinline const float* scale(size_t N) const { return (float*)((char*)this+6+25*N+12); }

    __forceinline       char* end(size_t N)       { return (char*)this+6+25*N+16; }
    __forceinline const char* end(size_t N) const { return (char*)this+6+25*N+16; }
  };

  template<int M>
    typename CurveNi<M>::Type CurveNi<M>::type;

  typedef CurveNi<4> Curve4i;
  typedef CurveNi<8> Curve8i;
}
