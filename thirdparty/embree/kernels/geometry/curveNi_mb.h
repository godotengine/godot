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
#include "curve_intersector_precalculations.h"

namespace embree
{
  template<int M>
    struct CurveNiMB
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
      static_assert(sizeof(CurveNiMB) == 6+37*M+24, "internal data layout issue");
      return f*sizeof(CurveNiMB) + (r!=0)*(6+37*r+24);
    }

  public:

    /*! Default constructor. */
    __forceinline CurveNiMB () {}

    /*! fill curve from curve list */
    __forceinline LBBox3fa fillMB(const PrimRefMB* prims, size_t& begin, size_t _end, Scene* scene, const BBox1f time_range)
    {
      size_t end = min(begin+M,_end);
      N = (unsigned char)(end-begin);
      const unsigned int geomID0 = prims[begin].geomID();
      this->geomID(N) = geomID0;
      ty = (unsigned char) scene->get(geomID0)->getType();

      /* encode all primitives */
      LBBox3fa lbounds = empty;
      for (size_t i=0; i<N; i++)
      {
        const PrimRefMB& prim = prims[begin+i];
        const unsigned int geomID = prim.geomID(); assert(geomID == geomID0);
        const unsigned int primID = prim.primID();
        lbounds.extend(scene->get(geomID)->vlinearBounds(primID,time_range));
      }
      BBox3fa bounds = lbounds.bounds();

      /* calculate offset and scale */
      Vec3fa loffset = bounds.lower;
      float lscale = reduce_min(256.0f/(bounds.size()*sqrt(3.0f)));
      if (bounds.size() == Vec3fa(zero)) lscale = 0.0f;
      *this->offset(N) = loffset;
      *this->scale(N) = lscale;
      this->time_offset(N) = time_range.lower;
      this->time_scale(N) = 1.0f/time_range.size();
      
      /* encode all primitives */
      for (size_t i=0; i<M && begin<end; i++, begin++)
      {
        const PrimRefMB& prim = prims[begin];
        const unsigned int geomID = prim.geomID();
        const unsigned int primID = prim.primID();
        const LinearSpace3fa space2 = scene->get(geomID)->computeAlignedSpaceMB(primID,time_range);
        
        const LinearSpace3fa space3(trunc(126.0f*space2.vx),trunc(126.0f*space2.vy),trunc(126.0f*space2.vz));
        const LBBox3fa bounds = scene->get(geomID)->vlinearBounds(loffset,lscale,max(length(space3.vx),length(space3.vy),length(space3.vz)),space3.transposed(),primID,time_range);
        
        // NOTE: this weird (char) (short) cast works around VS2015 Win32 compiler bug
        bounds_vx_x(N)[i] = (char) (short) space3.vx.x;
        bounds_vx_y(N)[i] = (char) (short) space3.vx.y;
        bounds_vx_z(N)[i] = (char) (short) space3.vx.z;
        bounds_vx_lower0(N)[i] = (short) clamp(floor(bounds.bounds0.lower.x),-32767.0f,32767.0f);
        bounds_vx_upper0(N)[i] = (short) clamp(ceil (bounds.bounds0.upper.x),-32767.0f,32767.0f);
        bounds_vx_lower1(N)[i] = (short) clamp(floor(bounds.bounds1.lower.x),-32767.0f,32767.0f);
        bounds_vx_upper1(N)[i] = (short) clamp(ceil (bounds.bounds1.upper.x),-32767.0f,32767.0f);
        assert(-32767.0f <= floor(bounds.bounds0.lower.x) && floor(bounds.bounds0.lower.x) <= 32767.0f);
        assert(-32767.0f <= ceil (bounds.bounds0.upper.x) && ceil (bounds.bounds0.upper.x) <= 32767.0f);
        assert(-32767.0f <= floor(bounds.bounds1.lower.x) && floor(bounds.bounds1.lower.x) <= 32767.0f);
        assert(-32767.0f <= ceil (bounds.bounds1.upper.x) && ceil (bounds.bounds1.upper.x) <= 32767.0f);
        
        bounds_vy_x(N)[i] = (char) (short) space3.vy.x;
        bounds_vy_y(N)[i] = (char) (short) space3.vy.y;
        bounds_vy_z(N)[i] = (char) (short) space3.vy.z;
        bounds_vy_lower0(N)[i] = (short) clamp(floor(bounds.bounds0.lower.y),-32767.0f,32767.0f);
        bounds_vy_upper0(N)[i] = (short) clamp(ceil (bounds.bounds0.upper.y),-32767.0f,32767.0f);
        bounds_vy_lower1(N)[i] = (short) clamp(floor(bounds.bounds1.lower.y),-32767.0f,32767.0f);
        bounds_vy_upper1(N)[i] = (short) clamp(ceil (bounds.bounds1.upper.y),-32767.0f,32767.0f);
        assert(-32767.0f <= floor(bounds.bounds0.lower.y) && floor(bounds.bounds0.lower.y) <= 32767.0f);
        assert(-32767.0f <= ceil (bounds.bounds0.upper.y) && ceil (bounds.bounds0.upper.y) <= 32767.0f);
        assert(-32767.0f <= floor(bounds.bounds1.lower.y) && floor(bounds.bounds1.lower.y) <= 32767.0f);
        assert(-32767.0f <= ceil (bounds.bounds1.upper.y) && ceil (bounds.bounds1.upper.y) <= 32767.0f);

        bounds_vz_x(N)[i] = (char) (short) space3.vz.x;
        bounds_vz_y(N)[i] = (char) (short) space3.vz.y;
        bounds_vz_z(N)[i] = (char) (short) space3.vz.z;
        bounds_vz_lower0(N)[i] = (short) clamp(floor(bounds.bounds0.lower.z),-32767.0f,32767.0f);
        bounds_vz_upper0(N)[i] = (short) clamp(ceil (bounds.bounds0.upper.z),-32767.0f,32767.0f);
        bounds_vz_lower1(N)[i] = (short) clamp(floor(bounds.bounds1.lower.z),-32767.0f,32767.0f);
        bounds_vz_upper1(N)[i] = (short) clamp(ceil (bounds.bounds1.upper.z),-32767.0f,32767.0f);
        assert(-32767.0f <= floor(bounds.bounds0.lower.z) && floor(bounds.bounds0.lower.z) <= 32767.0f);
        assert(-32767.0f <= ceil (bounds.bounds0.upper.z) && ceil (bounds.bounds0.upper.z) <= 32767.0f);
        assert(-32767.0f <= floor(bounds.bounds1.lower.z) && floor(bounds.bounds1.lower.z) <= 32767.0f);
        assert(-32767.0f <= ceil (bounds.bounds1.upper.z) && ceil (bounds.bounds1.upper.z) <= 32767.0f);
               
        this->primID(N)[i] = primID;
      }
      
      return lbounds;
    }

    template<typename BVH, typename SetMB, typename Allocator>
    __forceinline static typename BVH::NodeRecordMB4D createLeafMB(BVH* bvh, const SetMB& prims, const Allocator& alloc)
    {
      size_t start = prims.begin();
      size_t end   = prims.end();
      size_t items = CurveNiMB::blocks(prims.size());
      size_t numbytes = CurveNiMB::bytes(prims.size());
      CurveNiMB* accel = (CurveNiMB*) alloc.malloc1(numbytes,BVH::byteAlignment);
      const typename BVH::NodeRef node = bvh->encodeLeaf((char*)accel,items);
      
      LBBox3fa bounds = empty;
      for (size_t i=0; i<items; i++)
        bounds.extend(accel[i].fillMB(prims.prims->data(),start,end,bvh->scene,prims.time_range));
      
      return typename BVH::NodeRecordMB4D(node,bounds,prims.time_range);
    };

    
  public:
    
    // 27.6 - 46 bytes per primitive
    unsigned char ty;
    unsigned char N;
    unsigned char data[4+37*M+24];

    /*
    struct Layout
    {
      unsigned int geomID;
      unsigned int primID[N];
      
      char bounds_vx_x[N];
      char bounds_vx_y[N];
      char bounds_vx_z[N];
      short bounds_vx_lower0[N];
      short bounds_vx_upper0[N];
      short bounds_vx_lower1[N];
      short bounds_vx_upper1[N];
      
      char bounds_vy_x[N];
      char bounds_vy_y[N];
      char bounds_vy_z[N];
      short bounds_vy_lower0[N];
      short bounds_vy_upper0[N];
      short bounds_vy_lower1[N];
      short bounds_vy_upper1[N];
      
      char bounds_vz_x[N];
      char bounds_vz_y[N];
      char bounds_vz_z[N];
      short bounds_vz_lower0[N];
      short bounds_vz_upper0[N];
      short bounds_vz_lower1[N];
      short bounds_vz_upper1[N];
      
      Vec3f offset;
      float scale;

      float time_offset;
      float time_scale;
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
    
    __forceinline       short* bounds_vx_lower0(size_t N)       { return (short*)((char*)this+6+7*N); }
    __forceinline const short* bounds_vx_lower0(size_t N) const { return (short*)((char*)this+6+7*N); }
    
    __forceinline       short* bounds_vx_upper0(size_t N)       { return (short*)((char*)this+6+9*N); }
    __forceinline const short* bounds_vx_upper0(size_t N) const { return (short*)((char*)this+6+9*N); }

    __forceinline       short* bounds_vx_lower1(size_t N)       { return (short*)((char*)this+6+11*N); }
    __forceinline const short* bounds_vx_lower1(size_t N) const { return (short*)((char*)this+6+11*N); }
    
    __forceinline       short* bounds_vx_upper1(size_t N)       { return (short*)((char*)this+6+13*N); }
    __forceinline const short* bounds_vx_upper1(size_t N) const { return (short*)((char*)this+6+13*N); }

    __forceinline       char* bounds_vy_x(size_t N)       { return (char*)((char*)this+6+15*N); }
    __forceinline const char* bounds_vy_x(size_t N) const { return (char*)((char*)this+6+15*N); }
    
    __forceinline       char* bounds_vy_y(size_t N)       { return (char*)((char*)this+6+16*N); }
    __forceinline const char* bounds_vy_y(size_t N) const { return (char*)((char*)this+6+16*N); }
    
    __forceinline       char* bounds_vy_z(size_t N)       { return (char*)((char*)this+6+17*N); }
    __forceinline const char* bounds_vy_z(size_t N) const { return (char*)((char*)this+6+17*N); }
    
    __forceinline       short* bounds_vy_lower0(size_t N)       { return (short*)((char*)this+6+18*N); }
    __forceinline const short* bounds_vy_lower0(size_t N) const { return (short*)((char*)this+6+18*N); }
    
    __forceinline       short* bounds_vy_upper0(size_t N)       { return (short*)((char*)this+6+20*N); }
    __forceinline const short* bounds_vy_upper0(size_t N) const { return (short*)((char*)this+6+20*N); }

    __forceinline       short* bounds_vy_lower1(size_t N)       { return (short*)((char*)this+6+22*N); }
    __forceinline const short* bounds_vy_lower1(size_t N) const { return (short*)((char*)this+6+22*N); }
    
    __forceinline       short* bounds_vy_upper1(size_t N)       { return (short*)((char*)this+6+24*N); }
    __forceinline const short* bounds_vy_upper1(size_t N) const { return (short*)((char*)this+6+24*N); }
    
    __forceinline       char* bounds_vz_x(size_t N)       { return (char*)((char*)this+6+26*N); }
    __forceinline const char* bounds_vz_x(size_t N) const { return (char*)((char*)this+6+26*N); }
    
    __forceinline       char* bounds_vz_y(size_t N)       { return (char*)((char*)this+6+27*N); }
    __forceinline const char* bounds_vz_y(size_t N) const { return (char*)((char*)this+6+27*N); }
    
    __forceinline       char* bounds_vz_z(size_t N)       { return (char*)((char*)this+6+28*N); }
    __forceinline const char* bounds_vz_z(size_t N) const { return (char*)((char*)this+6+28*N); }
    
    __forceinline       short* bounds_vz_lower0(size_t N)       { return (short*)((char*)this+6+29*N); }
    __forceinline const short* bounds_vz_lower0(size_t N) const { return (short*)((char*)this+6+29*N); }
    
    __forceinline       short* bounds_vz_upper0(size_t N)       { return (short*)((char*)this+6+31*N); }
    __forceinline const short* bounds_vz_upper0(size_t N) const { return (short*)((char*)this+6+31*N); }

    __forceinline       short* bounds_vz_lower1(size_t N)       { return (short*)((char*)this+6+33*N); }
    __forceinline const short* bounds_vz_lower1(size_t N) const { return (short*)((char*)this+6+33*N); }
    
    __forceinline       short* bounds_vz_upper1(size_t N)       { return (short*)((char*)this+6+35*N); }
    __forceinline const short* bounds_vz_upper1(size_t N) const { return (short*)((char*)this+6+35*N); }

    __forceinline       Vec3f* offset(size_t N)       { return (Vec3f*)((char*)this+6+37*N); }
    __forceinline const Vec3f* offset(size_t N) const { return (Vec3f*)((char*)this+6+37*N); }
    
    __forceinline       float* scale(size_t N)       { return (float*)((char*)this+6+37*N+12); }
    __forceinline const float* scale(size_t N) const { return (float*)((char*)this+6+37*N+12); }

    __forceinline       float& time_offset(size_t N)       { return *(float*)((char*)this+6+37*N+16); }
    __forceinline const float& time_offset(size_t N) const { return *(float*)((char*)this+6+37*N+16); }
    
    __forceinline       float& time_scale(size_t N)       { return *(float*)((char*)this+6+37*N+20); }
    __forceinline const float& time_scale(size_t N) const { return *(float*)((char*)this+6+37*N+20); }

    __forceinline       char* end(size_t N)       { return (char*)this+6+37*N+24; }
    __forceinline const char* end(size_t N) const { return (char*)this+6+37*N+24; }
  };

  template<int M>
    typename CurveNiMB<M>::Type CurveNiMB<M>::type;

  typedef CurveNiMB<4> Curve4iMB;
  typedef CurveNiMB<8> Curve8iMB;
}
