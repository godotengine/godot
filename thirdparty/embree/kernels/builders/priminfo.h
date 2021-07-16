// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../common/default.h"
#include "../common/primref.h"
#include "../common/primref_mb.h"

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

    /*! stores bounding information for a set of primitives */
    template<typename BBox>
      class PrimInfoMBT : public CentGeom<BBox>
    {
    public:
      using CentGeom<BBox>::geomBounds;
      using CentGeom<BBox>::centBounds;

      __forceinline PrimInfoMBT () {
      } 

      __forceinline PrimInfoMBT (EmptyTy)
        : CentGeom<BBox>(empty), object_range(0,0), num_time_segments(0), max_num_time_segments(0), max_time_range(0.0f,1.0f), time_range(1.0f,0.0f) {}

      __forceinline PrimInfoMBT (size_t begin, size_t end)
        : CentGeom<BBox>(empty), object_range(begin,end), num_time_segments(0), max_num_time_segments(0), max_time_range(0.0f,1.0f), time_range(1.0f,0.0f) {}

      template<typename PrimRef> 
        __forceinline void add_primref(const PrimRef& prim) 
      {
        CentGeom<BBox>::extend_primref(prim);
        time_range.extend(prim.time_range);
        object_range._end++;
        num_time_segments += prim.size();
        if (max_num_time_segments < prim.totalTimeSegments()) {
          max_num_time_segments = prim.totalTimeSegments();
          max_time_range = prim.time_range;
        }
      }

      __forceinline void merge(const PrimInfoMBT& other)
      {
        CentGeom<BBox>::merge(other);
        time_range.extend(other.time_range);
        object_range._begin += other.object_range.begin();
        object_range._end += other.object_range.end();
        num_time_segments += other.num_time_segments;
        if (max_num_time_segments < other.max_num_time_segments) {
          max_num_time_segments = other.max_num_time_segments;
          max_time_range = other.max_time_range;
        }
      }

      static __forceinline const PrimInfoMBT merge2(const PrimInfoMBT& a, const PrimInfoMBT& b) {
        PrimInfoMBT r = a; r.merge(b); return r;
      }

      __forceinline size_t begin() const {
        return object_range.begin();
      }

      __forceinline size_t end() const {
        return object_range.end();
      }
      
      /*! returns the number of primitives */
      __forceinline size_t size() const { 
	return object_range.size(); 
      }

      __forceinline float halfArea() const {
        return time_range.size()*expectedApproxHalfArea(geomBounds);
      }

      __forceinline float leafSAH() const { 
	return time_range.size()*expectedApproxHalfArea(geomBounds)*float(num_time_segments); 
      }
      
      __forceinline float leafSAH(size_t block_shift) const { 
	return time_range.size()*expectedApproxHalfArea(geomBounds)*float((num_time_segments+(size_t(1)<<block_shift)-1) >> block_shift);
      }

      __forceinline float align_time(float ct) const
      {
        //return roundf(ct * float(numTimeSegments)) / float(numTimeSegments);
        float t0 = (ct-max_time_range.lower)/max_time_range.size();
        float t1 = roundf(t0 * float(max_num_time_segments)) / float(max_num_time_segments);
        return t1*max_time_range.size()+max_time_range.lower;
      }
      
      /*! stream output */
      friend embree_ostream operator<<(embree_ostream cout, const PrimInfoMBT& pinfo) 
      {
	return cout << "PrimInfo { " << 
          "object_range = " << pinfo.object_range << 
          ", time_range = " << pinfo.time_range << 
          ", time_segments = " << pinfo.num_time_segments << 
          ", geomBounds = " << pinfo.geomBounds << 
          ", centBounds = " << pinfo.centBounds << 
          "}";
      }
      
    public:
      range<size_t> object_range; //!< primitive range
      size_t num_time_segments;  //!< total number of time segments of all added primrefs
      size_t max_num_time_segments; //!< maximum number of time segments of a primitive
      BBox1f max_time_range; //!< time range of primitive with max_num_time_segments
      BBox1f time_range; //!< merged time range of primitives when merging prims, or additionally clipped with build time range when used in SetMB
    };

    typedef PrimInfoMBT<typename PrimRefMB::BBox> PrimInfoMB;

    struct SetMB : public PrimInfoMB
    {
      static const size_t PARALLEL_THRESHOLD = 3 * 1024;
      static const size_t PARALLEL_FIND_BLOCK_SIZE = 1024;
      static const size_t PARALLEL_PARTITION_BLOCK_SIZE = 128;

      typedef mvector<PrimRefMB>* PrimRefVector;

      __forceinline SetMB() {}

       __forceinline SetMB(const PrimInfoMB& pinfo_i, PrimRefVector prims)
         : PrimInfoMB(pinfo_i), prims(prims) {}

      __forceinline SetMB(const PrimInfoMB& pinfo_i, PrimRefVector prims, range<size_t> object_range_in, BBox1f time_range_in)
        : PrimInfoMB(pinfo_i), prims(prims)
      {
        object_range = object_range_in;
        time_range = intersect(time_range,time_range_in);
      }
      
      __forceinline SetMB(const PrimInfoMB& pinfo_i, PrimRefVector prims, BBox1f time_range_in)
        : PrimInfoMB(pinfo_i), prims(prims)
      {
        time_range = intersect(time_range,time_range_in);
      }

      void deterministic_order() const 
      {
        /* required as parallel partition destroys original primitive order */
        PrimRefMB* prim = prims->data();
        std::sort(&prim[object_range.begin()],&prim[object_range.end()]);
      }

      template<typename RecalculatePrimRef>
      __forceinline LBBox3fa linearBounds(const RecalculatePrimRef& recalculatePrimRef) const
      {
        auto reduce = [&](const range<size_t>& r) -> LBBox3fa
        {
          LBBox3fa cbounds(empty);
          for (size_t j = r.begin(); j < r.end(); j++)
          {
            PrimRefMB& ref = (*prims)[j];
            const LBBox3fa bn = recalculatePrimRef.linearBounds(ref, time_range);
            cbounds.extend(bn);
          };
          return cbounds;
        };
        
        return parallel_reduce(object_range.begin(), object_range.end(), PARALLEL_FIND_BLOCK_SIZE, PARALLEL_THRESHOLD, LBBox3fa(empty),
                               reduce,
                               [&](const LBBox3fa& b0, const LBBox3fa& b1) -> LBBox3fa { return embree::merge(b0, b1); });
      }

      template<typename RecalculatePrimRef>
        __forceinline LBBox3fa linearBounds(const RecalculatePrimRef& recalculatePrimRef, const LinearSpace3fa& space) const
      {
        auto reduce = [&](const range<size_t>& r) -> LBBox3fa
        {
          LBBox3fa cbounds(empty);
          for (size_t j = r.begin(); j < r.end(); j++)
          {
            PrimRefMB& ref = (*prims)[j];
            const LBBox3fa bn = recalculatePrimRef.linearBounds(ref, time_range, space);
            cbounds.extend(bn);
          };
          return cbounds;
        };
        
        return parallel_reduce(object_range.begin(), object_range.end(), PARALLEL_FIND_BLOCK_SIZE, PARALLEL_THRESHOLD, LBBox3fa(empty),
                               reduce,
                               [&](const LBBox3fa& b0, const LBBox3fa& b1) -> LBBox3fa { return embree::merge(b0, b1); });
      }

      template<typename RecalculatePrimRef>
        const SetMB primInfo(const RecalculatePrimRef& recalculatePrimRef, const LinearSpace3fa& space) const
      {
        auto computePrimInfo = [&](const range<size_t>& r) -> PrimInfoMB
        {
          PrimInfoMB pinfo(empty);
          for (size_t j=r.begin(); j<r.end(); j++)
          {
            PrimRefMB& ref = (*prims)[j];
            PrimRefMB ref1 = recalculatePrimRef(ref,time_range,space);
            pinfo.add_primref(ref1);
          };
          return pinfo;
        };
        
        const PrimInfoMB pinfo = parallel_reduce(object_range.begin(), object_range.end(), PARALLEL_FIND_BLOCK_SIZE, PARALLEL_THRESHOLD, 
                                                 PrimInfoMB(empty), computePrimInfo, PrimInfoMB::merge2);

        return SetMB(pinfo,prims,object_range,time_range);
      }
      
    public:
      PrimRefVector prims;
    };
//}
}
