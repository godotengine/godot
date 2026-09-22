// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../common/algorithms/parallel_reduce.h"
#include "../../common/algorithms/parallel_sort.h"
#include "../builders/heuristic_spatial.h"
#include "../builders/splitter.h"

#include "../../common/algorithms/parallel_partition.h"
#include "../../common/algorithms/parallel_for_for.h"
#include "../../common/algorithms/parallel_for_for_prefix_sum.h"

#define DBG_PRESPLIT(x)   
#define CHECK_PRESPLIT(x) 

#define GRID_SIZE 1024
//#define MAX_PRESPLITS_PER_PRIMITIVE_LOG 6
#define MAX_PRESPLITS_PER_PRIMITIVE_LOG 5
#define MAX_PRESPLITS_PER_PRIMITIVE (1<<MAX_PRESPLITS_PER_PRIMITIVE_LOG)
//#define PRIORITY_CUTOFF_THRESHOLD 2.0f
#define PRIORITY_SPLIT_POS_WEIGHT 1.5f

namespace embree
{  
  namespace isa
  {
    struct SplittingGrid
    {
      __forceinline SplittingGrid(const BBox3fa& bounds)
      {
        base = bounds.lower;
        const Vec3fa diag = bounds.size();
        extend = max(diag.x,max(diag.y,diag.z));		
        scale = extend == 0.0f ? 0.0f : GRID_SIZE / extend;
      }

      __forceinline bool split_pos(const PrimRef& prim, unsigned int& dim_o, float& fsplit_o) const
      {
        /* compute morton code */
        const Vec3fa lower = prim.lower;
        const Vec3fa upper = prim.upper;
        const Vec3fa glower = (lower-base)*Vec3fa(scale)+Vec3fa(0.2f);
        const Vec3fa gupper = (upper-base)*Vec3fa(scale)-Vec3fa(0.2f);
        Vec3ia ilower(floor(glower));
        Vec3ia iupper(floor(gupper));
        
        /* this ignores dimensions that are empty */
        iupper = (Vec3ia)select(vint4(glower) >= vint4(gupper),vint4(ilower),vint4(iupper));
        
        /* compute a morton code for the lower and upper grid coordinates. */
        const unsigned int lower_code = bitInterleave(ilower.x,ilower.y,ilower.z);
        const unsigned int upper_code = bitInterleave(iupper.x,iupper.y,iupper.z);

        /* if all bits are equal then we cannot split */
        if (unlikely(lower_code == upper_code))
          return false;
		    
        /* compute octree level and dimension to perform the split in */
        const unsigned int diff = 31 - lzcnt(lower_code^upper_code);
        const unsigned int level = diff / 3;
        const unsigned int dim   = diff % 3;
      
        /* now we compute the grid position of the split */
        const unsigned int isplit = iupper[dim] & ~((1<<level)-1);
			    
        /* compute world space position of split */
        const float inv_grid_size = 1.0f / GRID_SIZE;
        const float fsplit = base[dim] + isplit * inv_grid_size * extend;
        assert(prim.lower[dim] <= fsplit && prim.upper[dim] >= fsplit);

        dim_o = dim;
        fsplit_o = fsplit;
        return true;
      }

      __forceinline Vec2i computeMC(const PrimRef& ref) const
      {
        const Vec3fa lower = ref.lower;
        const Vec3fa upper = ref.upper;
        const Vec3fa glower = (lower-base)*Vec3fa(scale)+Vec3fa(0.2f);
        const Vec3fa gupper = (upper-base)*Vec3fa(scale)-Vec3fa(0.2f);
        Vec3ia ilower(floor(glower));
        Vec3ia iupper(floor(gupper));
        
        /* this ignores dimensions that are empty */
        iupper = (Vec3ia)select(vint4(glower) >= vint4(gupper),vint4(ilower),vint4(iupper));
        
        /* compute a morton code for the lower and upper grid coordinates. */
        const unsigned int lower_code = bitInterleave(ilower.x,ilower.y,ilower.z);
        const unsigned int upper_code = bitInterleave(iupper.x,iupper.y,iupper.z);
        return Vec2i(lower_code,upper_code);
      }
      
      Vec3fa base;
      float scale;
      float extend;
    };

    struct PresplitItem
    {
      union {
        float priority;    
        unsigned int data;
      };
      unsigned int index;
      
      __forceinline operator unsigned() const {
	return data;
      }

      template<typename ProjectedPrimitiveAreaFunc>
      __forceinline static float compute_priority(const ProjectedPrimitiveAreaFunc& primitiveArea, const PrimRef &ref, const Vec2i &mc)
      {
	const float area_aabb  = area(ref.bounds());
	const float area_prim  = primitiveArea(ref);
        if (area_prim == 0.0f) return 0.0f;
        const unsigned int diff = 31 - lzcnt(mc.x^mc.y);
        //assert(area_prim <= area_aabb); // may trigger due to numerical issues 
        const float area_diff = max(0.0f, area_aabb - area_prim);
        //const float priority = powf(area_diff * powf(PRIORITY_SPLIT_POS_WEIGHT,(float)diff),1.0f/4.0f);   
        const float priority = sqrtf(sqrtf( area_diff * powf(PRIORITY_SPLIT_POS_WEIGHT,(float)diff) ));
        //const float priority = sqrtf(sqrtf( area_diff ) );
        //const float priority = sqrtfarea_diff;
        //const float priority = area_diff; // 104 fps !!!!!!!!!!
        //const float priority = 0.2f*area_aabb + 0.8f*area_diff; // 104 fps
        //const float priority = area_aabb * max(area_aabb/area_prim,32.0f); 
        //const float priority = area_prim;
        assert(priority >= 0.0f && priority < FLT_LARGE);
	return priority;      
      }
    
    };

    inline std::ostream &operator<<(std::ostream &cout, const PresplitItem& item) {
      return cout << "index " << item.index << " priority " << item.priority;    
    };

#if 1
    
    template<typename Splitter>    
      void splitPrimitive(const Splitter& splitter,
                          const PrimRef& prim,
                          const unsigned int splitprims,
                          const SplittingGrid& grid,
                          PrimRef subPrims[MAX_PRESPLITS_PER_PRIMITIVE],
                          unsigned int& numSubPrims)
    {
      assert(splitprims > 0 && splitprims <= MAX_PRESPLITS_PER_PRIMITIVE);
      
      if (splitprims == 1)
      {
        assert(numSubPrims < MAX_PRESPLITS_PER_PRIMITIVE);
        subPrims[numSubPrims++] = prim;
      }
      else
      {
        unsigned int dim; float fsplit;
        if (!grid.split_pos(prim, dim, fsplit))
        {
          assert(numSubPrims < MAX_PRESPLITS_PER_PRIMITIVE);
          subPrims[numSubPrims++] = prim;
          return;
        }
          
        /* split primitive */
        PrimRef left,right;
        splitter(prim,dim,fsplit,left,right);
        assert(!left.bounds().empty());
        assert(!right.bounds().empty());

        const unsigned int splitprims_left = splitprims/2;
        const unsigned int splitprims_right = splitprims - splitprims_left;
        splitPrimitive(splitter,left,splitprims_left,grid,subPrims,numSubPrims);
        splitPrimitive(splitter,right,splitprims_right,grid,subPrims,numSubPrims);
      }
    }

#else
    
    template<typename Splitter>    
      void splitPrimitive(const Splitter& splitter,
                          const PrimRef& prim,
                          const unsigned int targetSubPrims,
                          const SplittingGrid& grid,
                          PrimRef subPrims[MAX_PRESPLITS_PER_PRIMITIVE],
                          unsigned int& numSubPrims)
    {
      assert(targetSubPrims > 0 && targetSubPrims <= MAX_PRESPLITS_PER_PRIMITIVE);
      
      auto compare = [] ( const PrimRef& a, const PrimRef& b ) {
        return area(a.bounds()) < area(b.bounds());
      };
      
      subPrims[numSubPrims++] = prim;

      while (numSubPrims < targetSubPrims)
      {
        /* get top heap element */
        std::pop_heap(subPrims+0,subPrims+numSubPrims, compare);
        PrimRef top = subPrims[--numSubPrims];

        unsigned int dim; float fsplit;
        if (!grid.split_pos(top, dim, fsplit))
        {
          assert(numSubPrims < MAX_PRESPLITS_PER_PRIMITIVE);
          subPrims[numSubPrims++] = top;
          return;
        }
          
        /* split primitive */
        PrimRef left,right;
        splitter(top,dim,fsplit,left,right);
        assert(!left.bounds().empty());
        assert(!right.bounds().empty());

        subPrims[numSubPrims++] = left;
        std::push_heap(subPrims+0, subPrims+numSubPrims, compare);

        subPrims[numSubPrims++] = right;
        std::push_heap(subPrims+0, subPrims+numSubPrims, compare);
      }
    }
    
#endif

#if !defined(RTHWIF_STANDALONE)

    template<typename Mesh, typename SplitterFactory>    
      PrimInfo createPrimRefArray_presplit(Geometry* geometry, unsigned int geomID, size_t numPrimRefs, mvector<PrimRef>& prims, BuildProgressMonitor& progressMonitor)
    {
      ParallelPrefixSumState<PrimInfo> pstate;
      
      /* first try */
      progressMonitor(0);
      PrimInfo pinfo = parallel_prefix_sum( pstate, size_t(0), geometry->size(), size_t(1024), PrimInfo(empty), [&](const range<size_t>& r, const PrimInfo& base) -> PrimInfo {
	  return geometry->createPrimRefArray(prims,r,r.begin(),geomID);
	}, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });

      /* if we need to filter out geometry, run again */
      if (pinfo.size() != numPrimRefs)
	{
	  progressMonitor(0);
	  pinfo = parallel_prefix_sum( pstate, size_t(0), geometry->size(), size_t(1024), PrimInfo(empty), [&](const range<size_t>& r, const PrimInfo& base) -> PrimInfo {
	      return geometry->createPrimRefArray(prims,r,base.size(),geomID);
	    }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });
	}
      return pinfo;	
    }
#endif
    
    template<typename SplitPrimitiveFunc, typename ProjectedPrimitiveAreaFunc, typename PrimVector>
    PrimInfo createPrimRefArray_presplit(size_t numPrimRefs,
                                         PrimVector& prims,
                                         const PrimInfo& pinfo,
                                         const SplitPrimitiveFunc& splitPrimitive,
                                         const ProjectedPrimitiveAreaFunc& primitiveArea)
    {
      static const size_t MIN_STEP_SIZE = 128;

      /* use correct number of primitives */
      size_t numPrimitives = pinfo.size();
      const size_t numPrimitivesExt = prims.size(); 
      const size_t numSplitPrimitivesBudget = numPrimitivesExt - numPrimitives;

      /* allocate double buffer presplit items */
      avector<PresplitItem> preSplitItem0(numPrimitivesExt);
      avector<PresplitItem> preSplitItem1(numPrimitivesExt);

      /* compute grid */
      SplittingGrid grid(pinfo.geomBounds);
      
      /* init presplit items and get total sum */
      const float psum = parallel_reduce( size_t(0), numPrimitives, size_t(MIN_STEP_SIZE), 0.0f, [&](const range<size_t>& r) -> float {
          float sum = 0.0f;
          for (size_t i=r.begin(); i<r.end(); i++)
          {		
            preSplitItem0[i].index = (unsigned int)i;
            const Vec2i mc = grid.computeMC(prims[i]);
            /* if all bits are equal then we cannot split */
            preSplitItem0[i].priority = (mc.x != mc.y) ? PresplitItem::compute_priority(primitiveArea,prims[i],mc) : 0.0f;    
            /* FIXME: sum undeterministic */
            sum += preSplitItem0[i].priority;
          }
          return sum;
        },[](const float& a, const float& b) -> float { return a+b; });

      /* compute number of splits per primitive */
      const float inv_psum = 1.0f / psum;
      parallel_for( size_t(0), numPrimitives, size_t(MIN_STEP_SIZE), [&](const range<size_t>& r) -> void {
          for (size_t i=r.begin(); i<r.end(); i++)
          {
            if (preSplitItem0[i].priority <= 0.0f) {
              preSplitItem0[i].data = 1;
              continue;
            }
              
            const float rel_p = (float)numSplitPrimitivesBudget * preSplitItem0[i].priority * inv_psum;
            if (rel_p < 1) {
              preSplitItem0[i].data = 1;
              continue;
            }
            
            //preSplitItem0[i].data = max(min(ceilf(rel_p),(float)MAX_PRESPLITS_PER_PRIMITIVE),1.0f);
            preSplitItem0[i].data = max(min(ceilf(logf(rel_p)/logf(2.0f)),(float)MAX_PRESPLITS_PER_PRIMITIVE_LOG),1.0f);
            preSplitItem0[i].data = 1 << preSplitItem0[i].data;
            assert(preSplitItem0[i].data <= MAX_PRESPLITS_PER_PRIMITIVE);
          }
        });

      auto isLeft = [&] (const PresplitItem &ref) { return ref.data <= 1; };        
      size_t center = parallel_partitioning(preSplitItem0.data(),0,numPrimitives,isLeft,1024);
      assert(center <= numPrimitives);

      /* anything to split ? */
      if (center >= numPrimitives)
        return pinfo;
            
      size_t numPrimitivesToSplit = numPrimitives - center;
      assert(preSplitItem0[center].data >= 1.0f);
      
      /* sort presplit items in ascending order */
      radix_sort_u32(preSplitItem0.data() + center,preSplitItem1.data() + center,numPrimitivesToSplit,1024);
      
      CHECK_PRESPLIT(
        parallel_for( size_t(center+1), numPrimitives, size_t(MIN_STEP_SIZE), [&](const range<size_t>& r) -> void {
          for (size_t i=r.begin(); i<r.end(); i++)
            assert(preSplitItem0[i-1].data <= preSplitItem0[i].data);
          });
      );
      
      unsigned int* primOffset0 = (unsigned int*)preSplitItem1.data();
      unsigned int* primOffset1 = (unsigned int*)preSplitItem1.data() + numPrimitivesToSplit;
      
      /* compute actual number of sub-primitives generated within the [center;numPrimitives-1] range */
      const size_t totalNumSubPrims = parallel_reduce( size_t(center), numPrimitives, size_t(MIN_STEP_SIZE), size_t(0), [&](const range<size_t>& t) -> size_t {
        size_t sum = 0;
        for (size_t i=t.begin(); i<t.end(); i++)
        {	
          const unsigned int primrefID  = preSplitItem0[i].index;	
          const unsigned int splitprims = preSplitItem0[i].data;
          assert(splitprims >= 1 && splitprims <= MAX_PRESPLITS_PER_PRIMITIVE);
          
          unsigned int numSubPrims = 0;
          PrimRef subPrims[MAX_PRESPLITS_PER_PRIMITIVE];	
          splitPrimitive(prims[primrefID],splitprims,grid,subPrims,numSubPrims);
          assert(numSubPrims);
          
          numSubPrims--; // can reuse slot 
          sum+=numSubPrims;
          preSplitItem0[i].data = (numSubPrims << 16) | splitprims;
          
          primOffset0[i-center] = numSubPrims;
        }
        return sum;
      },[](const size_t& a, const size_t& b) -> size_t { return a+b; });

      /* if we are over budget, need to shrink the range */
      if (totalNumSubPrims > numSplitPrimitivesBudget) 
      {
        size_t new_center = numPrimitives-1;
        size_t sum = 0;
        for (;new_center>=center;new_center--)
        {
          const unsigned int numSubPrims = preSplitItem0[new_center].data >> 16;
          if (unlikely(sum + numSubPrims >= numSplitPrimitivesBudget)) break;
          sum += numSubPrims;
        }
        new_center++;
        
        primOffset0 += new_center - center;
        numPrimitivesToSplit -= new_center - center;
        center = new_center;
        assert(numPrimitivesToSplit == (numPrimitives - center));
      }
      
      /* parallel prefix sum to compute offsets for storing sub-primitives */
      const unsigned int offset = parallel_prefix_sum(primOffset0,primOffset1,numPrimitivesToSplit,(unsigned int)0,std::plus<unsigned int>());
      assert(numPrimitives+offset <= numPrimitivesExt);
      
      /* iterate over range, and split primitives into sub primitives and append them to prims array */		    
      parallel_for( size_t(center), numPrimitives, size_t(MIN_STEP_SIZE), [&](const range<size_t>& rn) -> void {
        for (size_t j=rn.begin(); j<rn.end(); j++)		    
        {
          const unsigned int primrefID = preSplitItem0[j].index;	
          const unsigned int splitprims = preSplitItem0[j].data & 0xFFFF;
          assert(splitprims >= 1 && splitprims <= MAX_PRESPLITS_PER_PRIMITIVE);
          
          unsigned int numSubPrims = 0;
          PrimRef subPrims[MAX_PRESPLITS_PER_PRIMITIVE];
          splitPrimitive(prims[primrefID],splitprims,grid,subPrims,numSubPrims);

          const unsigned int numSubPrimsExpected MAYBE_UNUSED = preSplitItem0[j].data >> 16;
          assert(numSubPrims-1 == numSubPrimsExpected);
          
          const size_t newID = numPrimitives + primOffset1[j-center];
          assert(newID+numSubPrims-1 <= numPrimitivesExt);
          
          prims[primrefID] = subPrims[0];
          for (size_t i=1;i<numSubPrims;i++)
            prims[newID+i-1] = subPrims[i];
        }
      });

      numPrimitives += offset;
                
      /* recompute centroid bounding boxes */
      const PrimInfo pinfo1 = parallel_reduce(size_t(0),numPrimitives,size_t(MIN_STEP_SIZE),PrimInfo(empty),[&] (const range<size_t>& r) -> PrimInfo {
          PrimInfo p(empty);
          for (size_t j=r.begin(); j<r.end(); j++)
            p.add_center2(prims[j]);
          return p;
        }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });
  
      assert(pinfo1.size() == numPrimitives);
      
      return pinfo1;	
    }

#if !defined(RTHWIF_STANDALONE)
    
     template<typename Mesh, typename SplitterFactory>    
      PrimInfo createPrimRefArray_presplit(Scene* scene, Geometry::GTypeMask types, bool mblur, size_t numPrimRefs, mvector<PrimRef>& prims, BuildProgressMonitor& progressMonitor)
    {
      ParallelForForPrefixSumState<PrimInfo> pstate;
      Scene::Iterator2 iter(scene,types,mblur);

      /* first try */
      progressMonitor(0);
      pstate.init(iter,size_t(1024));
      PrimInfo pinfo = parallel_for_for_prefix_sum0( pstate, iter, PrimInfo(empty), [&](Geometry* mesh, const range<size_t>& r, size_t k, size_t geomID) -> PrimInfo {
	  return mesh->createPrimRefArray(prims,r,k,(unsigned)geomID);
	}, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });
      
      /* if we need to filter out geometry, run again */
      if (pinfo.size() != numPrimRefs)
	{
	  progressMonitor(0);
	  pinfo = parallel_for_for_prefix_sum1( pstate, iter, PrimInfo(empty), [&](Geometry* mesh, const range<size_t>& r, size_t k, size_t geomID, const PrimInfo& base) -> PrimInfo {
	      return mesh->createPrimRefArray(prims,r,base.size(),(unsigned)geomID);
	    }, [](const PrimInfo& a, const PrimInfo& b) -> PrimInfo { return PrimInfo::merge(a,b); });
	}


      SplitterFactory Splitter(scene);
        
      auto split_primitive = [&] (const PrimRef &prim,
                                  const unsigned int splitprims,
                                  const SplittingGrid& grid,
                                  PrimRef subPrims[MAX_PRESPLITS_PER_PRIMITIVE],
                                  unsigned int& numSubPrims)
      {
         const auto splitter = Splitter(prim);
         splitPrimitive(splitter,prim,splitprims,grid,subPrims,numSubPrims);
      };
      
      auto primitiveArea = [&] (const PrimRef &ref) {
        const unsigned int geomID = ref.geomID();
        const unsigned int primID = ref.primID();
        return ((Mesh*)scene->get(geomID))->projectedPrimitiveArea(primID);
      };
      
      return createPrimRefArray_presplit(numPrimRefs,prims,pinfo,split_primitive,primitiveArea);
    }
#endif 
  }
}
