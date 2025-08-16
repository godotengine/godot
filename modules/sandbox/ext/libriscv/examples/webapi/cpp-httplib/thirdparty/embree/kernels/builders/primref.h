// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../common/default.h"

namespace embree
{
  /*! A primitive reference stores the bounds of the primitive and its ID. */
  struct __aligned(32) PrimRef 
  {
    __forceinline PrimRef () {}

#if defined(__AVX__)
    __forceinline PrimRef(const PrimRef& v) { 
      vfloat8::store((float*)this,vfloat8::load((float*)&v));
    }
    __forceinline PrimRef& operator=(const PrimRef& v) { 
      vfloat8::store((float*)this,vfloat8::load((float*)&v)); return *this;
    }
#endif

    __forceinline PrimRef (const BBox3fa& bounds, unsigned int geomID, unsigned int primID) 
    {
      lower = Vec3fx(bounds.lower, geomID);
      upper = Vec3fx(bounds.upper, primID);
    }

    __forceinline PrimRef (const BBox3fa& bounds, size_t id) 
    {
#if defined(__64BIT__)
      lower = Vec3fx(bounds.lower, (unsigned)(id & 0xFFFFFFFF));
      upper = Vec3fx(bounds.upper, (unsigned)((id >> 32) & 0xFFFFFFFF));
#else
      lower = Vec3fx(bounds.lower, (unsigned)id);
      upper = Vec3fx(bounds.upper, (unsigned)0);
#endif
    }

    /*! calculates twice the center of the primitive */
    __forceinline const Vec3fa center2() const {
      return lower+upper;
    }
    
    /*! return the bounding box of the primitive */
    __forceinline const BBox3fa bounds() const {
      return BBox3fa(lower,upper);
    }

    /*! size for bin heuristic is 1 */
    __forceinline unsigned size() const { 
      return 1;
    }

    /*! returns bounds and centroid used for binning */
    __forceinline void binBoundsAndCenter(BBox3fa& bounds_o, Vec3fa& center_o) const 
    {
      bounds_o = bounds();
      center_o = embree::center2(bounds_o);
    }

    __forceinline unsigned& geomIDref() {  // FIXME: remove !!!!!!!
      return lower.u;
    }
    __forceinline unsigned& primIDref() {  // FIXME: remove !!!!!!!
      return upper.u;
    }
    
    /*! returns the geometry ID */
    __forceinline unsigned geomID() const { 
      return lower.a;
    }

    /*! returns the primitive ID */
    __forceinline unsigned primID() const { 
      return upper.a;
    }

    /*! returns an size_t sized ID */
    __forceinline size_t ID() const { 
#if defined(__64BIT__)
      return size_t(lower.u) + (size_t(upper.u) << 32);
#else
      return size_t(lower.u);
#endif
    }

    /*! special function for operator< */
    __forceinline uint64_t ID64() const {
      return (((uint64_t)primID()) << 32) + (uint64_t)geomID();
    }
    
    /*! allows sorting the primrefs by ID */
    friend __forceinline bool operator<(const PrimRef& p0, const PrimRef& p1) {
      return p0.ID64() < p1.ID64();
    }

    /*! Outputs primitive reference to a stream. */
    friend __forceinline embree_ostream operator<<(embree_ostream cout, const PrimRef& ref) {
      return cout << "{ lower = " << ref.lower << ", upper = " << ref.upper << ", geomID = " << ref.geomID() << ", primID = " << ref.primID() << " }";
    }

  public:
    Vec3fx lower;     //!< lower bounds and geomID
    Vec3fx upper;     //!< upper bounds and primID
  };

  /*! fast exchange for PrimRefs */
  __forceinline void xchg(PrimRef& a, PrimRef& b)
  {
#if defined(__AVX__)
    const vfloat8 aa = vfloat8::load((float*)&a);
    const vfloat8 bb = vfloat8::load((float*)&b);
    vfloat8::store((float*)&a,bb);
    vfloat8::store((float*)&b,aa);
#else
    std::swap(a,b);
#endif
  }
  
  
  /************************************************************************************/
  /************************************************************************************/
  /************************************************************************************/
  /************************************************************************************/
  
  struct SubGridBuildData {
    unsigned short sx,sy;
    unsigned int primID;
    
    __forceinline SubGridBuildData() {};
    __forceinline SubGridBuildData(const unsigned int sx, const unsigned int sy, const unsigned int primID) : sx(sx), sy(sy), primID(primID) {};
    
    __forceinline size_t x() const { return (size_t)sx & 0x7fff; }
    __forceinline size_t y() const { return (size_t)sy & 0x7fff; }
    
  };
}
