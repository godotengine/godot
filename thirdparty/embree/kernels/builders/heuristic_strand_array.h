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
    /*! Performs standard object binning */
    struct HeuristicStrandSplit
    {
      typedef range<size_t> Set;
  
      static const size_t PARALLEL_THRESHOLD = 10000;
      static const size_t PARALLEL_FIND_BLOCK_SIZE = 4096;
      static const size_t PARALLEL_PARTITION_BLOCK_SIZE = 64;

      /*! stores all information to perform some split */
      struct Split
      {    
	/*! construct an invalid split by default */
	__forceinline Split()
	  : sah(inf), axis0(zero), axis1(zero) {}
	
	/*! constructs specified split */
	__forceinline Split(const float sah, const Vec3fa& axis0, const Vec3fa& axis1)
	  : sah(sah), axis0(axis0), axis1(axis1) {}
	
	/*! calculates standard surface area heuristic for the split */
	__forceinline float splitSAH() const { return sah; }

        /*! test if this split is valid */
        __forceinline bool valid() const { return sah != float(inf); }
		
      public:
	float sah;             //!< SAH cost of the split
	Vec3fa axis0, axis1;   //!< axis the two strands are aligned into
      };

      __forceinline HeuristicStrandSplit () // FIXME: required?
        : scene(nullptr), prims(nullptr) {}
      
      /*! remember prim array */
      __forceinline HeuristicStrandSplit (Scene* scene, PrimRef* prims)
        : scene(scene), prims(prims) {}
      
      __forceinline const Vec3fa direction(const PrimRef& prim) {
        return scene->get(prim.geomID())->computeDirection(prim.primID());
      }
      
      __forceinline const BBox3fa bounds(const PrimRef& prim) {
        return scene->get(prim.geomID())->vbounds(prim.primID());
      }

      __forceinline const BBox3fa bounds(const LinearSpace3fa& space, const PrimRef& prim) {
        return scene->get(prim.geomID())->vbounds(space,prim.primID());
      }

      /*! finds the best split */
      const Split find(const range<size_t>& set, size_t logBlockSize)
      {
        Vec3fa axis0(0,0,1);
        uint64_t bestGeomPrimID = -1;

        /* curve with minimum ID determines first axis */
        for (size_t i=set.begin(); i<set.end(); i++)
        {
          const uint64_t geomprimID = prims[i].ID64();
          if (geomprimID >= bestGeomPrimID) continue;
          const Vec3fa axis = direction(prims[i]);
          if (sqr_length(axis) > 1E-18f) {
            axis0 = normalize(axis);
            bestGeomPrimID = geomprimID;
          }
        }
      
        /* find 2nd axis that is most misaligned with first axis and has minimum ID */
        float bestCos = 1.0f;
        Vec3fa axis1 = axis0;
        bestGeomPrimID = -1;
        for (size_t i=set.begin(); i<set.end(); i++) 
        {
          const uint64_t geomprimID = prims[i].ID64();
          Vec3fa axisi = direction(prims[i]);
          float leni = length(axisi);
          if (leni == 0.0f) continue;
          axisi /= leni;
          float cos = abs(dot(axisi,axis0));
          if ((cos == bestCos && (geomprimID < bestGeomPrimID)) || cos < bestCos) {
            bestCos = cos; axis1 = axisi;
            bestGeomPrimID = geomprimID;
          }
        }
      
        /* partition the two strands */
        size_t lnum = 0, rnum = 0;
        BBox3fa lbounds = empty, rbounds = empty;
        const LinearSpace3fa space0 = frame(axis0).transposed();
        const LinearSpace3fa space1 = frame(axis1).transposed();
        
        for (size_t i=set.begin(); i<set.end(); i++)
        {
          PrimRef& prim = prims[i];
          const Vec3fa axisi = normalize(direction(prim));
          const float cos0 = abs(dot(axisi,axis0));
          const float cos1 = abs(dot(axisi,axis1));
          
          if (cos0 > cos1) { lnum++; lbounds.extend(bounds(space0,prim)); }
          else             { rnum++; rbounds.extend(bounds(space1,prim)); }
        }
      
        /*! return an invalid split if we do not partition */
        if (lnum == 0 || rnum == 0) 
          return Split(inf,axis0,axis1);
      
        /*! calculate sah for the split */
        const size_t lblocks = (lnum+(1ull<<logBlockSize)-1ull) >> logBlockSize;
        const size_t rblocks = (rnum+(1ull<<logBlockSize)-1ull) >> logBlockSize;
        const float sah = madd(float(lblocks),halfArea(lbounds),float(rblocks)*halfArea(rbounds));
        return Split(sah,axis0,axis1);
      }

      /*! array partitioning */
      void split(const Split& split, const PrimInfoRange& set, PrimInfoRange& lset, PrimInfoRange& rset) 
      {
        if (!split.valid()) {
          deterministic_order(set);
          return splitFallback(set,lset,rset);
        }
        
        const size_t begin = set.begin();
        const size_t end   = set.end();
        CentGeomBBox3fa local_left(empty);
        CentGeomBBox3fa local_right(empty);

        auto primOnLeftSide = [&] (const PrimRef& prim) -> bool { 
          const Vec3fa axisi = normalize(direction(prim));
          const float cos0 = abs(dot(axisi,split.axis0));
          const float cos1 = abs(dot(axisi,split.axis1));
          return cos0 > cos1;
        };

        auto mergePrimBounds = [this] (CentGeomBBox3fa& pinfo,const PrimRef& ref) { 
          pinfo.extend(bounds(ref)); 
        };
        
        size_t center = serial_partitioning(prims,begin,end,local_left,local_right,primOnLeftSide,mergePrimBounds);
        
        new (&lset) PrimInfoRange(begin,center,local_left);
        new (&rset) PrimInfoRange(center,end,local_right);
        assert(area(lset.geomBounds) >= 0.0f);
        assert(area(rset.geomBounds) >= 0.0f);
      }

      void deterministic_order(const Set& set) 
      {
        /* required as parallel partition destroys original primitive order */
        std::sort(&prims[set.begin()],&prims[set.end()]);
      }
      
      void splitFallback(const Set& set, PrimInfoRange& lset, PrimInfoRange& rset)
      {
        const size_t begin = set.begin();
        const size_t end   = set.end();
        const size_t center = (begin + end)/2;
        
        CentGeomBBox3fa left(empty);
        for (size_t i=begin; i<center; i++)
          left.extend(bounds(prims[i]));
        new (&lset) PrimInfoRange(begin,center,left);
        
        CentGeomBBox3fa right(empty);
        for (size_t i=center; i<end; i++)
          right.extend(bounds(prims[i]));	
        new (&rset) PrimInfoRange(center,end,right);
      }
      
    private:
      Scene* const scene;
      PrimRef* const prims;
    };
  }
}
