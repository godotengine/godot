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

#include "heuristic_binning.h"

namespace embree
{
  namespace isa
  {
    struct PrimInfoRange : public CentGeomBBox3fa, public range<size_t>
    {
      __forceinline PrimInfoRange () {
      }

      __forceinline PrimInfoRange(const PrimInfo& pinfo)
        : CentGeomBBox3fa(pinfo), range<size_t>(pinfo.begin,pinfo.end) {}

      __forceinline PrimInfoRange(EmptyTy)
        : CentGeomBBox3fa(EmptyTy()), range<size_t>(0,0) {}

      __forceinline PrimInfoRange (size_t begin, size_t end, const CentGeomBBox3fa& centGeomBounds)
        : CentGeomBBox3fa(centGeomBounds), range<size_t>(begin,end) {}
      
      __forceinline float leafSAH() const { 
	return expectedApproxHalfArea(geomBounds)*float(size()); 
      }
      
      __forceinline float leafSAH(size_t block_shift) const { 
	return expectedApproxHalfArea(geomBounds)*float((size()+(size_t(1)<<block_shift)-1) >> block_shift);
      }
    };
    
    /*! Performs standard object binning */
    template<typename PrimRef, size_t BINS>
      struct HeuristicArrayBinningSAH
      {
        typedef BinSplit<BINS> Split;
        typedef BinInfoT<BINS,PrimRef,BBox3fa> Binner;
        typedef range<size_t> Set;

#if defined(__AVX512ER__) // KNL
        static const size_t PARALLEL_THRESHOLD = 4*768; 
        static const size_t PARALLEL_FIND_BLOCK_SIZE = 768;
        static const size_t PARALLEL_PARTITION_BLOCK_SIZE = 768;
#else
        static const size_t PARALLEL_THRESHOLD = 3 * 1024;
        static const size_t PARALLEL_FIND_BLOCK_SIZE = 1024;
        static const size_t PARALLEL_PARTITION_BLOCK_SIZE = 128;
#endif
        __forceinline HeuristicArrayBinningSAH ()
          : prims(nullptr) {}

        /*! remember prim array */
        __forceinline HeuristicArrayBinningSAH (PrimRef* prims)
          : prims(prims) {}

        /*! finds the best split */
        __noinline const Split find(const PrimInfoRange& pinfo, const size_t logBlockSize)
        {
          if (likely(pinfo.size() < PARALLEL_THRESHOLD))
            return find_template<false>(pinfo,logBlockSize);
          else
            return find_template<true>(pinfo,logBlockSize);
        }

        template<bool parallel>
        __forceinline const Split find_template(const PrimInfoRange& pinfo, const size_t logBlockSize)
        {
          Binner binner(empty);
          const BinMapping<BINS> mapping(pinfo);
          bin_serial_or_parallel<parallel>(binner,prims,pinfo.begin(),pinfo.end(),PARALLEL_FIND_BLOCK_SIZE,mapping);
          return binner.best(mapping,logBlockSize);
        }

        /*! array partitioning */
        __forceinline void split(const Split& split, const PrimInfoRange& pinfo, PrimInfoRange& linfo, PrimInfoRange& rinfo)
        {
          if (likely(pinfo.size() < PARALLEL_THRESHOLD))
            split_template<false>(split,pinfo,linfo,rinfo);
          else
            split_template<true>(split,pinfo,linfo,rinfo);
        }

        template<bool parallel>
        __forceinline void split_template(const Split& split, const PrimInfoRange& set, PrimInfoRange& lset, PrimInfoRange& rset)
        {
          if (!split.valid()) {
            deterministic_order(set);
            return splitFallback(set,lset,rset);
          }
          
          const size_t begin = set.begin();
          const size_t end   = set.end();
          CentGeomBBox3fa local_left(empty);
          CentGeomBBox3fa local_right(empty);
          const unsigned int splitPos = split.pos;
          const unsigned int splitDim = split.dim;
          const unsigned int splitDimMask = (unsigned int)1 << splitDim;

          const typename Binner::vint vSplitPos(splitPos);
          const typename Binner::vbool vSplitMask(splitDimMask);
          auto isLeft = [&] (const PrimRef &ref) { return split.mapping.bin_unsafe(ref,vSplitPos,vSplitMask); };

          size_t center = 0;
          if (!parallel)
            center = serial_partitioning(prims,begin,end,local_left,local_right,isLeft,
                                         [] (CentGeomBBox3fa& pinfo,const PrimRef& ref) { pinfo.extend_center2(ref); });
          else
            center = parallel_partitioning(
              prims,begin,end,EmptyTy(),local_left,local_right,isLeft,
              [] (CentGeomBBox3fa& pinfo,const PrimRef& ref) { pinfo.extend_center2(ref); },
              [] (CentGeomBBox3fa& pinfo0,const CentGeomBBox3fa& pinfo1) { pinfo0.merge(pinfo1); },
              PARALLEL_PARTITION_BLOCK_SIZE);
          
          new (&lset) PrimInfoRange(begin,center,local_left);
          new (&rset) PrimInfoRange(center,end,local_right);
          assert(area(lset.geomBounds) >= 0.0f);
          assert(area(rset.geomBounds) >= 0.0f);
        }

        void deterministic_order(const PrimInfoRange& pinfo)
        {
          /* required as parallel partition destroys original primitive order */
          std::sort(&prims[pinfo.begin()],&prims[pinfo.end()]);
        }

        void splitFallback(const PrimInfoRange& pinfo, PrimInfoRange& linfo, PrimInfoRange& rinfo)
        {
          const size_t begin = pinfo.begin();
          const size_t end   = pinfo.end();
          const size_t center = (begin + end)/2;

          CentGeomBBox3fa left(empty);
          for (size_t i=begin; i<center; i++)
            left.extend_center2(prims[i]);
          new (&linfo) PrimInfoRange(begin,center,left);

          CentGeomBBox3fa right(empty);
          for (size_t i=center; i<end; i++)
            right.extend_center2(prims[i]);
          new (&rinfo) PrimInfoRange(center,end,right);
        }

        void splitByGeometry(const range<size_t>& range, PrimInfoRange& linfo, PrimInfoRange& rinfo)
        {
          assert(range.size() > 1);
          CentGeomBBox3fa left(empty);
          CentGeomBBox3fa right(empty);
          unsigned int geomID = prims[range.begin()].geomID();
          size_t center = serial_partitioning(prims,range.begin(),range.end(),left,right,
                                              [&] ( const PrimRef& prim ) { return prim.geomID() == geomID; },
                                              [ ] ( CentGeomBBox3fa& a, const PrimRef& ref ) { a.extend_center2(ref); });

          new (&linfo) PrimInfoRange(range.begin(),center,left);
          new (&rinfo) PrimInfoRange(center,range.end(),right);
        }

      private:
        PrimRef* const prims;
      };

    /*! Performs standard object binning */
    template<typename PrimRefMB, size_t BINS>
      struct HeuristicArrayBinningMB
      {
        typedef BinSplit<BINS> Split;
        typedef typename PrimRefMB::BBox BBox;
        typedef BinInfoT<BINS,PrimRefMB,BBox> ObjectBinner;
        static const size_t PARALLEL_THRESHOLD = 3 * 1024;
        static const size_t PARALLEL_FIND_BLOCK_SIZE = 1024;
        static const size_t PARALLEL_PARTITION_BLOCK_SIZE = 128;

        /*! finds the best split */
        const Split find(const SetMB& set, const size_t logBlockSize)
        {
          ObjectBinner binner(empty);
          const BinMapping<BINS> mapping(set.size(),set.centBounds);
          bin_parallel(binner,set.prims->data(),set.begin(),set.end(),PARALLEL_FIND_BLOCK_SIZE,PARALLEL_THRESHOLD,mapping);
          Split osplit = binner.best(mapping,logBlockSize);
          osplit.sah *= set.time_range.size();
          if (!osplit.valid()) osplit.data = Split::SPLIT_FALLBACK; // use fallback split
          return osplit;
        }
        
        /*! array partitioning */
        __forceinline void split(const Split& split, const SetMB& set, SetMB& lset, SetMB& rset)
        {
          const size_t begin = set.begin();
          const size_t end   = set.end();
          PrimInfoMB left = empty;
          PrimInfoMB right = empty;
          const vint4 vSplitPos(split.pos);
          const vbool4 vSplitMask(1 << split.dim);
          auto isLeft = [&] (const PrimRefMB &ref) { return any(((vint4)split.mapping.bin_unsafe(ref) < vSplitPos) & vSplitMask); };
          auto reduction = [] (PrimInfoMB& pinfo, const PrimRefMB& ref) { pinfo.add_primref(ref); };
          auto reduction2 = [] (PrimInfoMB& pinfo0,const PrimInfoMB& pinfo1) { pinfo0.merge(pinfo1); };
          size_t center = parallel_partitioning(set.prims->data(),begin,end,EmptyTy(),left,right,isLeft,reduction,reduction2,PARALLEL_PARTITION_BLOCK_SIZE,PARALLEL_THRESHOLD);
          new (&lset) SetMB(left, set.prims,range<size_t>(begin,center),set.time_range);
          new (&rset) SetMB(right,set.prims,range<size_t>(center,end  ),set.time_range);
        }
      };
  }
}
