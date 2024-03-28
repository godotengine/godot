// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "primref.h"

namespace embree
{
  // FIXME: maybe there's a better place for this util fct
  __forceinline float areaProjectedTriangle(const Vec3fa& v0, const Vec3fa& v1, const Vec3fa& v2)
  {
    const Vec3fa e0 = v1-v0;
    const Vec3fa e1 = v2-v0;
    const Vec3fa d = cross(e0,e1);
    return fabs(d.x) + fabs(d.y) + fabs(d.z);
  }

  //namespace isa
  //{
    template<typename BBox>
      class CentGeom
    {
    public:
      __forceinline CentGeom () {}

      __forceinline CentGeom (EmptyTy) 
	: geomBounds(empty), centBounds(empty) {}
      
      __forceinline CentGeom (const BBox& geomBounds, const BBox3fa& centBounds) 
	: geomBounds(geomBounds), centBounds(centBounds) {}
      
      template<typename PrimRef> 
        __forceinline void extend_primref(const PrimRef& prim) 
      {
        BBox bounds; Vec3fa center;
        prim.binBoundsAndCenter(bounds,center);
        geomBounds.extend(bounds);
        centBounds.extend(center);
      }

      static void extend_ref (CentGeom& pinfo, const PrimRef& ref) {
        pinfo.extend_primref(ref);
      };
      
       template<typename PrimRef> 
         __forceinline void extend_center2(const PrimRef& prim) 
       {
         BBox3fa bounds = prim.bounds();
         geomBounds.extend(bounds);
         centBounds.extend(bounds.center2());
       }
       
      __forceinline void extend(const BBox& geomBounds_) {
	geomBounds.extend(geomBounds_);
	centBounds.extend(center2(geomBounds_));
      }

      __forceinline void merge(const CentGeom& other) 
      {
	geomBounds.extend(other.geomBounds);
	centBounds.extend(other.centBounds);
      }

      static __forceinline const CentGeom merge2(const CentGeom& a, const CentGeom& b) {
        CentGeom r = a; r.merge(b); return r;
      }

    public:
      BBox geomBounds;   //!< geometry bounds of primitives
      BBox3fa centBounds;   //!< centroid bounds of primitives
    };

    typedef CentGeom<BBox3fa> CentGeomBBox3fa;

    /*! stores bounding information for a set of primitives */
    template<typename BBox>
      class PrimInfoT : public CentGeom<BBox>
    {
    public:
      using CentGeom<BBox>::geomBounds;
      using CentGeom<BBox>::centBounds;

      __forceinline PrimInfoT () {}

      __forceinline PrimInfoT (EmptyTy) 
	: CentGeom<BBox>(empty), begin(0), end(0) {}

      __forceinline PrimInfoT (size_t N) 
	: CentGeom<BBox>(empty), begin(0), end(N) {}

      __forceinline PrimInfoT (size_t begin, size_t end, const CentGeomBBox3fa& centGeomBounds) 
        : CentGeom<BBox>(centGeomBounds), begin(begin), end(end) {}

      template<typename PrimRef> 
        __forceinline void add_primref(const PrimRef& prim) 
      {
        CentGeom<BBox>::extend_primref(prim);
        end++;
      }

       template<typename PrimRef> 
         __forceinline void add_center2(const PrimRef& prim) {
         CentGeom<BBox>::extend_center2(prim);
         end++;
       }

        template<typename PrimRef> 
          __forceinline void add_center2(const PrimRef& prim, const size_t i) {
          CentGeom<BBox>::extend_center2(prim);
          end+=i;
        }

      /*__forceinline void add(const BBox& geomBounds_) {
	CentGeom<BBox>::extend(geomBounds_);
	end++;
      }

      __forceinline void add(const BBox& geomBounds_, const size_t i) {
	CentGeom<BBox>::extend(geomBounds_);
	end+=i;
        }*/

      __forceinline void merge(const PrimInfoT& other) 
      {
	CentGeom<BBox>::merge(other);
        begin += other.begin;
	end += other.end;
      }

      static __forceinline const PrimInfoT merge(const PrimInfoT& a, const PrimInfoT& b) {
        PrimInfoT r = a; r.merge(b); return r;
      }
      
      /*! returns the number of primitives */
      __forceinline size_t size() const { 
	return end-begin; 
      }

      __forceinline float halfArea() {
        return expectedApproxHalfArea(geomBounds);
      }

      __forceinline float leafSAH() const { 
	return expectedApproxHalfArea(geomBounds)*float(size()); 
	//return halfArea(geomBounds)*blocks(num); 
      }
      
      __forceinline float leafSAH(size_t block_shift) const { 
	return expectedApproxHalfArea(geomBounds)*float((size()+(size_t(1)<<block_shift)-1) >> block_shift);
	//return halfArea(geomBounds)*float((num+3) >> 2);
	//return halfArea(geomBounds)*blocks(num); 
      }
      
      /*! stream output */
      friend embree_ostream operator<<(embree_ostream cout, const PrimInfoT& pinfo) {
	return cout << "PrimInfo { begin = " << pinfo.begin << ", end = " << pinfo.end << ", geomBounds = " << pinfo.geomBounds << ", centBounds = " << pinfo.centBounds << "}";
      }
      
    public:
      size_t begin,end;          //!< number of primitives
    };

    typedef PrimInfoT<BBox3fa> PrimInfo;
    //typedef PrimInfoT<LBBox3fa> PrimInfoMB;
//}
}
