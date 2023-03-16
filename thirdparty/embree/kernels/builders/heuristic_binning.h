// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "priminfo.h"
#include "../../common/algorithms/parallel_reduce.h"
#include "../../common/algorithms/parallel_partition.h"

namespace embree
{
  namespace isa
  { 
    /*! mapping into bins */
    template<size_t BINS>
      struct BinMapping
      {
      public:
        __forceinline BinMapping() {}
        
        /*! calculates the mapping */
        __forceinline BinMapping(size_t N, const BBox3fa& centBounds) 
        {
          num = min(BINS,size_t(4.0f + 0.05f*N));
          assert(num >= 1);
          const vfloat4 eps = 1E-34f;
          const vfloat4 diag = max(eps, (vfloat4) centBounds.size());
          scale = select(diag > eps,vfloat4(0.99f*num)/diag,vfloat4(0.0f));
          ofs  = (vfloat4) centBounds.lower;
        }

        /*! calculates the mapping */
        __forceinline BinMapping(const BBox3fa& centBounds) 
        {
          num = BINS;
          const vfloat4 eps = 1E-34f;
          const vfloat4 diag = max(eps, (vfloat4) centBounds.size());
          scale = select(diag > eps,vfloat4(0.99f*num)/diag,vfloat4(0.0f));
          ofs  = (vfloat4) centBounds.lower;
        }

        /*! calculates the mapping */
        template<typename PrimInfo>
        __forceinline BinMapping(const PrimInfo& pinfo) 
        {
          const vfloat4 eps = 1E-34f;
          num = min(BINS,size_t(4.0f + 0.05f*pinfo.size()));
          const vfloat4 diag = max(eps,(vfloat4) pinfo.centBounds.size());
          scale = select(diag > eps,vfloat4(0.99f*num)/diag,vfloat4(0.0f));
          ofs  = (vfloat4) pinfo.centBounds.lower;
        }

        /*! returns number of bins */
        __forceinline size_t size() const { return num; }
        
        /*! slower but safe binning */
        __forceinline Vec3ia bin(const Vec3fa& p) const 
        {
          const vint4 i = floori((vfloat4(p)-ofs)*scale);
          assert(i[0] >= 0 && (size_t)i[0] < num); 
          assert(i[1] >= 0 && (size_t)i[1] < num);
          assert(i[2] >= 0 && (size_t)i[2] < num);
          
          // we clamp to handle corner cases that could calculate out of bounds bin
          return Vec3ia(clamp(i,vint4(0),vint4(num-1)));
        }

        /*! faster but unsafe binning */
        __forceinline Vec3ia bin_unsafe(const Vec3fa& p) const {
          return Vec3ia(floori((vfloat4(p)-ofs)*scale));
        }

        /*! faster but unsafe binning */
        template<typename PrimRef>
        __forceinline Vec3ia bin_unsafe(const PrimRef& p) const {
          return bin_unsafe(p.binCenter());
        }

        /*! faster but unsafe binning */
        template<typename PrimRef, typename BinBoundsAndCenter>
        __forceinline Vec3ia bin_unsafe(const PrimRef& p, const BinBoundsAndCenter& binBoundsAndCenter) const {
          return bin_unsafe(binBoundsAndCenter.binCenter(p));
        }

        template<typename PrimRef>
        __forceinline bool bin_unsafe(const PrimRef& ref,
                                      const vint4&   vSplitPos,
                                      const vbool4&  splitDimMask) const // FIXME: rename to isLeft
        {
          return any(((vint4)bin_unsafe(center2(ref.bounds())) < vSplitPos) & splitDimMask);
        }
        /*! calculates left spatial position of bin */
        __forceinline float pos(const size_t bin, const size_t dim) const {
          return madd(float(bin),1.0f / scale[dim],ofs[dim]);
        }

        /*! returns true if the mapping is invalid in some dimension */
        __forceinline bool invalid(const size_t dim) const {
          return scale[dim] == 0.0f;
        }
        
        /*! stream output */
        friend embree_ostream operator<<(embree_ostream cout, const BinMapping& mapping) {
          return cout << "BinMapping { num = " << mapping.num << ", ofs = " << mapping.ofs << ", scale = " << mapping.scale << "}";
        }
        
      public:
        size_t num;
        vfloat4 ofs,scale;        //!< linear function that maps to bin ID
      };
    
    /*! stores all information to perform some split */
    template<size_t BINS>
      struct BinSplit
      {
        enum
        {
          SPLIT_OBJECT   = 0,
          SPLIT_FALLBACK = 1,
          SPLIT_ENFORCE  = 2, // splits with larger ID are enforced in createLargeLeaf even if we could create a leaf already
          SPLIT_TEMPORAL = 2,
          SPLIT_GEOMID   = 3,
        };

        /*! construct an invalid split by default */
        __forceinline BinSplit()
          : sah(inf), dim(-1), pos(0), data(0) {}

        __forceinline BinSplit(float sah, unsigned data, int dim = 0, float fpos = 0)
          : sah(sah), dim(dim), fpos(fpos), data(data) {}
        
        /*! constructs specified split */
        __forceinline BinSplit(float sah, int dim, int pos, const BinMapping<BINS>& mapping)
          : sah(sah), dim(dim), pos(pos), data(0), mapping(mapping) {}
        
        /*! tests if this split is valid */
        __forceinline bool valid() const { return dim != -1; }
        
        /*! calculates surface area heuristic for performing the split */
        __forceinline float splitSAH() const { return sah; }
        
        /*! stream output */
        friend embree_ostream operator<<(embree_ostream cout, const BinSplit& split) {
          return cout << "BinSplit { sah = " << split.sah << ", dim = " << split.dim << ", pos = " << split.pos << "}";
        }
        
      public:
        float sah;                //!< SAH cost of the split
        int dim;                  //!< split dimension
        union { int pos; float fpos; };                  //!< bin index for splitting
        unsigned int data;        //!< extra optional split data
        BinMapping<BINS> mapping; //!< mapping into bins
      };
    
    /*! stores extended information about the split */
    template<typename BBox>
      struct SplitInfoT
    {

      __forceinline SplitInfoT () {}
      
      __forceinline SplitInfoT (size_t leftCount, const BBox& leftBounds, size_t rightCount, const BBox& rightBounds)
	: leftCount(leftCount), rightCount(rightCount), leftBounds(leftBounds), rightBounds(rightBounds) {}
      
    public:
      size_t leftCount,rightCount;
      BBox leftBounds,rightBounds;
    };

    typedef SplitInfoT<BBox3fa> SplitInfo;
    typedef SplitInfoT<LBBox3fa> SplitInfo2;
    
    /*! stores all binning information */
    template<size_t BINS, typename PrimRef, typename BBox>
      struct __aligned(64) BinInfoT
    {		  
      typedef BinSplit<BINS> Split;
      typedef vbool4 vbool;
      typedef vint4 vint;
      typedef vfloat4 vfloat;
      
      __forceinline BinInfoT() {
      }
      
      __forceinline BinInfoT(EmptyTy) {
	clear();
      }

      /*! bin access function */
      __forceinline BBox &bounds(const size_t binID, const size_t dimID)             { return _bounds[binID][dimID]; }
      __forceinline const BBox &bounds(const size_t binID, const size_t dimID) const { return _bounds[binID][dimID]; }

      __forceinline unsigned int &counts(const size_t binID, const size_t dimID)             { return _counts[binID][dimID]; }
      __forceinline const unsigned int &counts(const size_t binID, const size_t dimID) const { return _counts[binID][dimID]; }

      __forceinline vuint4 &counts(const size_t binID)             { return _counts[binID]; }
      __forceinline const vuint4 &counts(const size_t binID) const { return _counts[binID]; }

      /*! clears the bin info */
      __forceinline void clear() 
      {
	for (size_t i=0; i<BINS; i++) {
	  bounds(i,0) = bounds(i,1) = bounds(i,2) = empty;
	  counts(i) = vuint4(zero);
	}
      }
      
      /*! bins an array of primitives */
      __forceinline void bin (const PrimRef* prims, size_t N, const BinMapping<BINS>& mapping)
      {
	if (unlikely(N == 0)) return;
	size_t i; 
	for (i=0; i<N-1; i+=2)
        {
          /*! map even and odd primitive to bin */
          BBox prim0; Vec3fa center0;
          prims[i+0].binBoundsAndCenter(prim0,center0); 
          const vint4 bin0 = (vint4)mapping.bin(center0); 
          
          BBox prim1; Vec3fa center1;
          prims[i+1].binBoundsAndCenter(prim1,center1); 
          const vint4 bin1 = (vint4)mapping.bin(center1); 
          
          /*! increase bounds for bins for even primitive */
          const unsigned int b00 = extract<0>(bin0); bounds(b00,0).extend(prim0); 
          const unsigned int b01 = extract<1>(bin0); bounds(b01,1).extend(prim0); 
          const unsigned int b02 = extract<2>(bin0); bounds(b02,2).extend(prim0); 
          const unsigned int s0 = (unsigned int)prims[i+0].size();
          counts(b00,0)+=s0;
          counts(b01,1)+=s0;
          counts(b02,2)+=s0;

          /*! increase bounds of bins for odd primitive */
          const unsigned int b10 = extract<0>(bin1);  bounds(b10,0).extend(prim1); 
          const unsigned int b11 = extract<1>(bin1);  bounds(b11,1).extend(prim1); 
          const unsigned int b12 = extract<2>(bin1);  bounds(b12,2).extend(prim1); 
          const unsigned int s1 = (unsigned int)prims[i+1].size();
          counts(b10,0)+=s1;
          counts(b11,1)+=s1;
          counts(b12,2)+=s1;
        }
	/*! for uneven number of primitives */
	if (i < N)
        {
          /*! map primitive to bin */
          BBox prim0; Vec3fa center0;
          prims[i].binBoundsAndCenter(prim0,center0); 
          const vint4 bin0 = (vint4)mapping.bin(center0); 
          
          /*! increase bounds of bins */
          const unsigned int s0 = (unsigned int)prims[i].size();
          const int b00 = extract<0>(bin0); counts(b00,0)+=s0; bounds(b00,0).extend(prim0);
          const int b01 = extract<1>(bin0); counts(b01,1)+=s0; bounds(b01,1).extend(prim0);
          const int b02 = extract<2>(bin0); counts(b02,2)+=s0; bounds(b02,2).extend(prim0);
        }
      }

      /*! bins an array of primitives */
      template<typename BinBoundsAndCenter>
        __forceinline void bin (const PrimRef* prims, size_t N, const BinMapping<BINS>& mapping, const BinBoundsAndCenter& binBoundsAndCenter)
      {
	if (N == 0) return;
        
	size_t i; 
	for (i=0; i<N-1; i+=2)
        {
          /*! map even and odd primitive to bin */
          BBox prim0; Vec3fa center0; binBoundsAndCenter.binBoundsAndCenter(prims[i+0],prim0,center0); 
          const vint4 bin0 = (vint4)mapping.bin(center0); 
          BBox prim1; Vec3fa center1; binBoundsAndCenter.binBoundsAndCenter(prims[i+1],prim1,center1); 
          const vint4 bin1 = (vint4)mapping.bin(center1); 
          
          /*! increase bounds for bins for even primitive */
          const unsigned int s0 = prims[i+0].size();
          const int b00 = extract<0>(bin0); counts(b00,0)+=s0; bounds(b00,0).extend(prim0);
          const int b01 = extract<1>(bin0); counts(b01,1)+=s0; bounds(b01,1).extend(prim0);
          const int b02 = extract<2>(bin0); counts(b02,2)+=s0; bounds(b02,2).extend(prim0);
          
          /*! increase bounds of bins for odd primitive */
          const unsigned int s1 = prims[i+1].size();
          const int b10 = extract<0>(bin1); counts(b10,0)+=s1; bounds(b10,0).extend(prim1);
          const int b11 = extract<1>(bin1); counts(b11,1)+=s1; bounds(b11,1).extend(prim1);
          const int b12 = extract<2>(bin1); counts(b12,2)+=s1; bounds(b12,2).extend(prim1);
        }
	
	/*! for uneven number of primitives */
	if (i < N)
        {
          /*! map primitive to bin */
          BBox prim0; Vec3fa center0; binBoundsAndCenter.binBoundsAndCenter(prims[i+0],prim0,center0); 
          const vint4 bin0 = (vint4)mapping.bin(center0); 
          
          /*! increase bounds of bins */
          const unsigned int s0 = prims[i+0].size();
          const int b00 = extract<0>(bin0); counts(b00,0)+=s0; bounds(b00,0).extend(prim0);
          const int b01 = extract<1>(bin0); counts(b01,1)+=s0; bounds(b01,1).extend(prim0);
          const int b02 = extract<2>(bin0); counts(b02,2)+=s0; bounds(b02,2).extend(prim0);
        }
      }
      
      __forceinline void bin(const PrimRef* prims, size_t begin, size_t end, const BinMapping<BINS>& mapping) {
	bin(prims+begin,end-begin,mapping);
      }

      template<typename BinBoundsAndCenter>
        __forceinline void bin(const PrimRef* prims, size_t begin, size_t end, const BinMapping<BINS>& mapping, const BinBoundsAndCenter& binBoundsAndCenter) {
	bin<BinBoundsAndCenter>(prims+begin,end-begin,mapping,binBoundsAndCenter);
      }

      /*! merges in other binning information */
      __forceinline void merge (const BinInfoT& other, size_t numBins)
      {
		
	for (size_t i=0; i<numBins; i++) 
        {
          counts(i) += other.counts(i);
          bounds(i,0).extend(other.bounds(i,0));
          bounds(i,1).extend(other.bounds(i,1));
          bounds(i,2).extend(other.bounds(i,2));
        }
      }

      /*! reduces binning information */
      static __forceinline const BinInfoT reduce (const BinInfoT& a, const BinInfoT& b, const size_t numBins = BINS)
      {
        BinInfoT c;
	for (size_t i=0; i<numBins; i++) 
        {
          c.counts(i) = a.counts(i)+b.counts(i);
          c.bounds(i,0) = embree::merge(a.bounds(i,0),b.bounds(i,0));
          c.bounds(i,1) = embree::merge(a.bounds(i,1),b.bounds(i,1));
          c.bounds(i,2) = embree::merge(a.bounds(i,2),b.bounds(i,2));
        }
        return c;
      }
      
      /*! finds the best split by scanning binning information */
      __forceinline Split best(const BinMapping<BINS>& mapping, const size_t blocks_shift) const
      {
	/* sweep from right to left and compute parallel prefix of merged bounds */
	vfloat4 rAreas[BINS];
	vuint4 rCounts[BINS];
	vuint4 count = 0; BBox bx = empty; BBox by = empty; BBox bz = empty;
	for (size_t i=mapping.size()-1; i>0; i--)
        {
          count += counts(i);
          rCounts[i] = count;
          bx.extend(bounds(i,0)); rAreas[i][0] = expectedApproxHalfArea(bx);
          by.extend(bounds(i,1)); rAreas[i][1] = expectedApproxHalfArea(by);
          bz.extend(bounds(i,2)); rAreas[i][2] = expectedApproxHalfArea(bz);
          rAreas[i][3] = 0.0f;
        }
	/* sweep from left to right and compute SAH */
	vuint4 blocks_add = (1 << blocks_shift)-1;
	vuint4 ii = 1; vfloat4 vbestSAH = pos_inf; vuint4 vbestPos = 0; 
	count = 0; bx = empty; by = empty; bz = empty;
	for (size_t i=1; i<mapping.size(); i++, ii+=1)
        {
          count += counts(i-1);
          bx.extend(bounds(i-1,0)); float Ax = expectedApproxHalfArea(bx);
          by.extend(bounds(i-1,1)); float Ay = expectedApproxHalfArea(by);
          bz.extend(bounds(i-1,2)); float Az = expectedApproxHalfArea(bz);
          const vfloat4 lArea = vfloat4(Ax,Ay,Az,Az);
          const vfloat4 rArea = rAreas[i];
          const vuint4 lCount = (count     +blocks_add) >> (unsigned int)(blocks_shift); // if blocks_shift >=1 then lCount < 4B and could be represented with an vint4, which would allow for faster vfloat4 conversions.
          const vuint4 rCount = (rCounts[i]+blocks_add) >> (unsigned int)(blocks_shift);
          const vfloat4 sah = madd(lArea,vfloat4(lCount),rArea*vfloat4(rCount));
          //const vfloat4 sah = madd(lArea,vfloat4(vint4(lCount)),rArea*vfloat4(vint4(rCount)));

          vbestPos = select(sah < vbestSAH,ii ,vbestPos);
          vbestSAH = select(sah < vbestSAH,sah,vbestSAH);
        }
	
	/* find best dimension */
	float bestSAH = inf;
	int   bestDim = -1;
	int   bestPos = 0;
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
          }
        }
	return Split(bestSAH,bestDim,bestPos,mapping);
      }
      
      /*! calculates extended split information */
      __forceinline void getSplitInfo(const BinMapping<BINS>& mapping, const Split& split, SplitInfoT<BBox>& info) const 
      {
	if (split.dim == -1) {
	  new (&info) SplitInfoT<BBox>(0,empty,0,empty);
	  return;
	}
	
	size_t leftCount = 0;
	BBox leftBounds = empty;
	for (size_t i=0; i<(size_t)split.pos; i++) {
	  leftCount += counts(i,split.dim);
	  leftBounds.extend(bounds(i,split.dim));
	}
	size_t rightCount = 0;
	BBox rightBounds = empty;
	for (size_t i=split.pos; i<mapping.size(); i++) {
	  rightCount += counts(i,split.dim);
	  rightBounds.extend(bounds(i,split.dim));
	}
	new (&info) SplitInfoT<BBox>(leftCount,leftBounds,rightCount,rightBounds);
      }

      /*! gets the number of primitives left of the split */
      __forceinline size_t getLeftCount(const BinMapping<BINS>& mapping, const Split& split) const
      {
        if (unlikely(split.dim == -1)) return -1;

        size_t leftCount = 0;
        for (size_t i = 0; i < (size_t)split.pos; i++) {
          leftCount += counts(i, split.dim);
        }
        return leftCount;
      }

      /*! gets the number of primitives right of the split */
      __forceinline size_t getRightCount(const BinMapping<BINS>& mapping, const Split& split) const
      {
        if (unlikely(split.dim == -1)) return -1;

        size_t rightCount = 0;
        for (size_t i = (size_t)split.pos; i<mapping.size(); i++) {
          rightCount += counts(i, split.dim);
        }
        return rightCount;
      }

    private:
      BBox _bounds[BINS][3]; //!< geometry bounds for each bin in each dimension
      vuint4   _counts[BINS];    //!< counts number of primitives that map into the bins
    };
  }

  template<typename BinInfoT, typename BinMapping, typename PrimRef>
  __forceinline void bin_parallel(BinInfoT& binner, const PrimRef* prims, size_t begin, size_t end, size_t blockSize, size_t parallelThreshold, const BinMapping& mapping)
  {
    if (likely(end-begin < parallelThreshold)) {
      binner.bin(prims,begin,end,mapping);
    } else {
      binner = parallel_reduce(begin,end,blockSize,binner,
                              [&](const range<size_t>& r) -> BinInfoT { BinInfoT binner(empty); binner.bin(prims + r.begin(), r.size(), mapping); return binner; },
                              [&](const BinInfoT& b0, const BinInfoT& b1) -> BinInfoT { BinInfoT r = b0; r.merge(b1, mapping.size()); return r; });
    }
  }

  template<typename BinBoundsAndCenter, typename BinInfoT, typename BinMapping, typename PrimRef>
  __forceinline void bin_parallel(BinInfoT& binner, const PrimRef* prims, size_t begin, size_t end, size_t blockSize, size_t parallelThreshold, const BinMapping& mapping, const BinBoundsAndCenter& binBoundsAndCenter)
  {
    if (likely(end-begin < parallelThreshold)) {
      binner.bin(prims,begin,end,mapping,binBoundsAndCenter);
    } else {
      binner = parallel_reduce(begin,end,blockSize,binner,
                              [&](const range<size_t>& r) -> BinInfoT { BinInfoT binner(empty); binner.bin(prims + r.begin(), r.size(), mapping, binBoundsAndCenter); return binner; },
                              [&](const BinInfoT& b0, const BinInfoT& b1) -> BinInfoT { BinInfoT r = b0; r.merge(b1, mapping.size()); return r; });
    }
  }

  template<bool parallel, typename BinInfoT, typename BinMapping, typename PrimRef>
  __forceinline void bin_serial_or_parallel(BinInfoT& binner, const PrimRef* prims, size_t begin, size_t end, size_t blockSize, const BinMapping& mapping)
  {
    if (!parallel) {
      binner.bin(prims,begin,end,mapping);
    } else {
      binner = parallel_reduce(begin,end,blockSize,binner,
                              [&](const range<size_t>& r) -> BinInfoT { BinInfoT binner(empty); binner.bin(prims + r.begin(), r.size(), mapping); return binner; },
                              [&](const BinInfoT& b0, const BinInfoT& b1) -> BinInfoT { BinInfoT r = b0; r.merge(b1, mapping.size()); return r; });
    }
  }

  template<bool parallel, typename BinBoundsAndCenter, typename BinInfoT, typename BinMapping, typename PrimRef>
  __forceinline void bin_serial_or_parallel(BinInfoT& binner, const PrimRef* prims, size_t begin, size_t end, size_t blockSize, const BinMapping& mapping, const BinBoundsAndCenter& binBoundsAndCenter)
  {
    if (!parallel) {
      binner.bin(prims,begin,end,mapping,binBoundsAndCenter);
    } else {
      binner = parallel_reduce(begin,end,blockSize,binner,
                              [&](const range<size_t>& r) -> BinInfoT { BinInfoT binner(empty); binner.bin(prims + r.begin(), r.size(), mapping, binBoundsAndCenter); return binner; },
                              [&](const BinInfoT& b0, const BinInfoT& b1) -> BinInfoT { BinInfoT r = b0; r.merge(b1, mapping.size()); return r; });
    }
  }
}
