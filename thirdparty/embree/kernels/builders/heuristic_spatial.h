// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../common/scene.h"
#include "priminfo.h"

namespace embree
{
  static const unsigned int RESERVED_NUM_SPATIAL_SPLITS_GEOMID_BITS = 5;

  namespace isa
  {

    /*! mapping into bins */
    template<size_t BINS>
      struct SpatialBinMapping
      {
      public:
        __forceinline SpatialBinMapping() {}
        
        /*! calculates the mapping */
        __forceinline SpatialBinMapping(const CentGeomBBox3fa& pinfo)
        {
          const vfloat4 lower = (vfloat4) pinfo.geomBounds.lower;
          const vfloat4 upper = (vfloat4) pinfo.geomBounds.upper;
          const vfloat4 eps = 128.0f*vfloat4(ulp)*max(abs(lower),abs(upper));
          const vfloat4 diag = max(eps,(vfloat4) pinfo.geomBounds.size());
          scale = select(upper-lower <= eps,vfloat4(0.0f),vfloat4(BINS)/diag);
          ofs  = (vfloat4) pinfo.geomBounds.lower;
          inv_scale = 1.0f / scale; 
        }

        /*! slower but safe binning */
        __forceinline vint4 bin(const Vec3fa& p) const
        {
          const vint4 i = floori((vfloat4(p)-ofs)*scale);
          return clamp(i,vint4(0),vint4(BINS-1));
        }

        __forceinline std::pair<vint4,vint4> bin(const BBox3fa& b) const
        {
#if defined(__AVX__)
          const vfloat8 ofs8(ofs);
          const vfloat8 scale8(scale);
          const vint8 lu   = floori((vfloat8::loadu(&b)-ofs8)*scale8);
          const vint8 c_lu = clamp(lu,vint8(zero),vint8(BINS-1));
          return std::pair<vint4,vint4>(extract4<0>(c_lu),extract4<1>(c_lu));
#else
          const vint4 lower = floori((vfloat4(b.lower)-ofs)*scale);
          const vint4 upper = floori((vfloat4(b.upper)-ofs)*scale);
          const vint4 c_lower = clamp(lower,vint4(0),vint4(BINS-1));
          const vint4 c_upper = clamp(upper,vint4(0),vint4(BINS-1));
          return std::pair<vint4,vint4>(c_lower,c_upper);
#endif
        }

        
        /*! calculates left spatial position of bin */
        __forceinline float pos(const size_t bin, const size_t dim) const {
          return madd(float(bin),inv_scale[dim],ofs[dim]);
        }

        /*! calculates left spatial position of bin */
        template<size_t N>
        __forceinline vfloat<N> posN(const vfloat<N> bin, const size_t dim) const {
          return madd(bin,vfloat<N>(inv_scale[dim]),vfloat<N>(ofs[dim]));
        }
        
        /*! returns true if the mapping is invalid in some dimension */
        __forceinline bool invalid(const size_t dim) const {
          return scale[dim] == 0.0f;
        }
        
      public:
        vfloat4 ofs,scale,inv_scale;  //!< linear function that maps to bin ID
      };

    /*! stores all information required to perform some split */
    template<size_t BINS>
      struct SpatialBinSplit
      {
        /*! construct an invalid split by default */
        __forceinline SpatialBinSplit() 
          : sah(inf), dim(-1), pos(0), left(-1), right(-1), factor(1.0f) {}
        
        /*! constructs specified split */
        __forceinline SpatialBinSplit(float sah, int dim, int pos, const SpatialBinMapping<BINS>& mapping)
          : sah(sah), dim(dim), pos(pos), left(-1), right(-1), factor(1.0f), mapping(mapping) {}

        /*! constructs specified split */
        __forceinline SpatialBinSplit(float sah, int dim, int pos, int left, int right, float factor, const SpatialBinMapping<BINS>& mapping)
          : sah(sah), dim(dim), pos(pos), left(left), right(right), factor(factor), mapping(mapping) {}
        
        /*! tests if this split is valid */
        __forceinline bool valid() const { return dim != -1; }
        
        /*! calculates surface area heuristic for performing the split */
        __forceinline float splitSAH() const { return sah; }
        
        /*! stream output */
        friend embree_ostream operator<<(embree_ostream cout, const SpatialBinSplit& split) {
          return cout << "SpatialBinSplit { sah = " << split.sah << ", dim = " << split.dim << ", pos = " << split.pos << ", left = " << split.left << ", right = " << split.right << ", factor = " << split.factor << "}";
        }
        
      public:
        float sah;                 //!< SAH cost of the split
        int   dim;                 //!< split dimension
        int   pos;                 //!< split position
        int   left;                //!< number of elements on the left side
        int   right;               //!< number of elements on the right side
        float factor;              //!< factor splitting the extended range
        SpatialBinMapping<BINS> mapping; //!< mapping into bins
      };    
    
    /*! stores all binning information */
    template<size_t BINS, typename PrimRef>
      struct __aligned(64) SpatialBinInfo
    {
      SpatialBinInfo() {
      }

      __forceinline SpatialBinInfo(EmptyTy) {
	clear();
      }

      /*! clears the bin info */
      __forceinline void clear() 
      {
        for (size_t i=0; i<BINS; i++) { 
          bounds[i][0] = bounds[i][1] = bounds[i][2] = empty;
          numBegin[i] = numEnd[i] = 0;
        }
      }
      
      /*! adds binning data */
      __forceinline void add(const size_t dim,
                             const size_t beginID, 
                             const size_t endID, 
                             const size_t binID, 
                             const BBox3fa &b,
                             const size_t n = 1) 
      {
        assert(beginID < BINS);
        assert(endID < BINS);
        assert(binID < BINS);

        numBegin[beginID][dim]+=(unsigned int)n;
        numEnd  [endID][dim]+=(unsigned int)n;
        bounds  [binID][dim].extend(b);        
      }

      /*! extends binning bounds */
      __forceinline void extend(const size_t dim,
                                const size_t binID, 
                                const BBox3fa &b) 
      {
        assert(binID < BINS);
        bounds  [binID][dim].extend(b);        
      }

      /*! bins an array of primitives */
      template<typename PrimitiveSplitterFactory>
        __forceinline void bin2(const PrimitiveSplitterFactory& splitterFactory, const PrimRef* source, size_t begin, size_t end, const SpatialBinMapping<BINS>& mapping)
      {
        for (size_t i=begin; i<end; i++)
        {
          const PrimRef& prim = source[i];
          unsigned splits = prim.geomID() >> (32-RESERVED_NUM_SPATIAL_SPLITS_GEOMID_BITS);
          
          if (unlikely(splits <= 1))
          {
            const vint4 bin = mapping.bin(center(prim.bounds()));
            for (size_t dim=0; dim<3; dim++) 
            {
              assert(bin[dim] >= (int)0 && bin[dim] < (int)BINS);
              add(dim,bin[dim],bin[dim],bin[dim],prim.bounds());
            }
          }
          else
          {
            const vint4 bin0 = mapping.bin(prim.bounds().lower);
            const vint4 bin1 = mapping.bin(prim.bounds().upper);
            
            for (size_t dim=0; dim<3; dim++) 
            {
              if (unlikely(mapping.invalid(dim))) 
                continue;
              
              size_t bin;
              size_t l = bin0[dim];
              size_t r = bin1[dim];
              
              // same bin optimization
              if (likely(l == r)) 
              {
                add(dim,l,l,l,prim.bounds());
                continue;
              }
              size_t bin_start = bin0[dim];
              size_t bin_end   = bin1[dim];
              BBox3fa rest = prim.bounds();
              
              /* assure that split position always overlaps the primitive bounds */
              while (bin_start < bin_end && mapping.pos(bin_start+1,dim) <= rest.lower[dim]) bin_start++;
              while (bin_start < bin_end && mapping.pos(bin_end    ,dim) >= rest.upper[dim]) bin_end--;
              
              const auto splitter = splitterFactory(prim);
              for (bin=bin_start; bin<bin_end; bin++) 
              {
                const float pos = mapping.pos(bin+1,dim);
                BBox3fa left,right;
                splitter(rest,dim,pos,left,right);
                
                if (unlikely(left.empty())) l++;                
                extend(dim,bin,left);
                rest = right;
              }
              if (unlikely(rest.empty())) r--;
              add(dim,l,r,bin,rest);
            }
          }              
        }
      }


      /*! bins an array of primitives */
      __forceinline void binSubTreeRefs(const PrimRef* source, size_t begin, size_t end, const SpatialBinMapping<BINS>& mapping)
      {
        for (size_t i=begin; i<end; i++)
        {
          const PrimRef &prim = source[i];
          const vint4 bin0 = mapping.bin(prim.bounds().lower);
          const vint4 bin1 = mapping.bin(prim.bounds().upper);
          
          for (size_t dim=0; dim<3; dim++) 
          {
            if (unlikely(mapping.invalid(dim))) 
              continue;
            
            const size_t l = bin0[dim];
            const size_t r = bin1[dim];

            const unsigned int n  = prim.primID();
            
            // same bin optimization
            if (likely(l == r)) 
            {
              add(dim,l,l,l,prim.bounds(),n);
              continue;
            }
            const size_t bin_start = bin0[dim];
            const size_t bin_end   = bin1[dim];
            for (size_t bin=bin_start; bin<bin_end; bin++) 
              add(dim,l,r,bin,prim.bounds(),n);
          }
        }              
      }
      
      /*! merges in other binning information */
      void merge (const SpatialBinInfo& other)
      {
        for (size_t i=0; i<BINS; i++) 
        {
          numBegin[i] += other.numBegin[i];
          numEnd  [i] += other.numEnd  [i];
          bounds[i][0].extend(other.bounds[i][0]);
          bounds[i][1].extend(other.bounds[i][1]);
          bounds[i][2].extend(other.bounds[i][2]);
        }
      }

      /*! merges in other binning information */
      static __forceinline const SpatialBinInfo reduce (const SpatialBinInfo& a, const SpatialBinInfo& b)
      {
        SpatialBinInfo c(empty);
        for (size_t i=0; i<BINS; i++) 
        {
          c.numBegin[i] += a.numBegin[i]+b.numBegin[i];
          c.numEnd  [i] += a.numEnd  [i]+b.numEnd  [i];
          c.bounds[i][0] = embree::merge(a.bounds[i][0],b.bounds[i][0]);
          c.bounds[i][1] = embree::merge(a.bounds[i][1],b.bounds[i][1]);
          c.bounds[i][2] = embree::merge(a.bounds[i][2],b.bounds[i][2]);
        }
        return c;
      }
      
      /*! finds the best split by scanning binning information */
      SpatialBinSplit<BINS> best(const SpatialBinMapping<BINS>& mapping, const size_t blocks_shift) const 
      {
        /* sweep from right to left and compute parallel prefix of merged bounds */
        vfloat4 rAreas[BINS];
        vuint4 rCounts[BINS];
        vuint4 count = 0; BBox3fa bx = empty; BBox3fa by = empty; BBox3fa bz = empty;
        for (size_t i=BINS-1; i>0; i--)
        {
          count += numEnd[i];
          rCounts[i] = count;
          bx.extend(bounds[i][0]); rAreas[i][0] = halfArea(bx);
          by.extend(bounds[i][1]); rAreas[i][1] = halfArea(by);
          bz.extend(bounds[i][2]); rAreas[i][2] = halfArea(bz);
          rAreas[i][3] = 0.0f;
        }
        
        /* sweep from left to right and compute SAH */
        vuint4 blocks_add = (1 << blocks_shift)-1;
        vuint4 ii = 1; vfloat4 vbestSAH = pos_inf; vuint4 vbestPos = 0; vuint4 vbestlCount = 0; vuint4 vbestrCount = 0;
        count = 0; bx = empty; by = empty; bz = empty;
        for (size_t i=1; i<BINS; i++, ii+=1)
        {
          count += numBegin[i-1];
          bx.extend(bounds[i-1][0]); float Ax = halfArea(bx);
          by.extend(bounds[i-1][1]); float Ay = halfArea(by);
          bz.extend(bounds[i-1][2]); float Az = halfArea(bz);
          const vfloat4 lArea = vfloat4(Ax,Ay,Az,Az);
          const vfloat4 rArea = rAreas[i];
          const vuint4 lCount = (count     +blocks_add) >> (unsigned int)(blocks_shift);
          const vuint4 rCount = (rCounts[i]+blocks_add) >> (unsigned int)(blocks_shift);
          const vfloat4 sah = madd(lArea,vfloat4(lCount),rArea*vfloat4(rCount));
          // const vfloat4 sah = madd(lArea,vfloat4(vint4(lCount)),rArea*vfloat4(vint4(rCount)));
          const vbool4 mask = sah < vbestSAH;
          vbestPos      = select(mask,ii ,vbestPos);
          vbestSAH      = select(mask,sah,vbestSAH);
          vbestlCount   = select(mask,count,vbestlCount);
          vbestrCount   = select(mask,rCounts[i],vbestrCount);
        }
        
        /* find best dimension */
        float bestSAH = inf;
        int   bestDim = -1;
        int   bestPos = 0;
        unsigned int   bestlCount = 0;
        unsigned int   bestrCount = 0;
        for (int dim=0; dim<3; dim++) 
        {
          /* ignore zero sized dimensions */
          if (unlikely(mapping.invalid(dim)))
            continue;
          
          /* test if this is a better dimension */
          if (vbestSAH[dim] < bestSAH && vbestPos[dim] != 0) {
            bestDim = dim;
            bestPos = vbestPos[dim];
            bestSAH = vbestSAH[dim];
            bestlCount = vbestlCount[dim];
            bestrCount = vbestrCount[dim];
          }
        }
        assert(bestSAH >= 0.0f);
        
        /* return invalid split if no split found */
        if (bestDim == -1) 
          return SpatialBinSplit<BINS>(inf,-1,0,mapping);
        
        /* return best found split */
        return SpatialBinSplit<BINS>(bestSAH,bestDim,bestPos,bestlCount,bestrCount,1.0f,mapping);
      }
      
    private:
      BBox3fa bounds[BINS][3];  //!< geometry bounds for each bin in each dimension
      vuint4    numBegin[BINS];   //!< number of primitives starting in bin
      vuint4    numEnd[BINS];     //!< number of primitives ending in bin
    };
  }
}

