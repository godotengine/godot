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
    /*! Performs standard object binning */
    template<typename PrimRef, size_t BINS>
      struct UnalignedHeuristicArrayBinningSAH
      {
        typedef BinSplit<BINS> Split;
        typedef BinInfoT<BINS,PrimRef,BBox3fa> Binner;
        typedef range<size_t> Set;

        __forceinline UnalignedHeuristicArrayBinningSAH () // FIXME: required?
          : scene(nullptr), prims(nullptr) {}
        
        /*! remember prim array */
        __forceinline UnalignedHeuristicArrayBinningSAH (Scene* scene, PrimRef* prims)
          : scene(scene), prims(prims) {}

        const LinearSpace3fa computeAlignedSpace(const range<size_t>& set)
        {
          Vec3fa axis(0,0,1);
          uint64_t bestGeomPrimID = -1;

          /*! find curve with minimum ID that defines valid direction */
          for (size_t i=set.begin(); i<set.end(); i++)
          {
            const unsigned int geomID = prims[i].geomID();
            const unsigned int primID = prims[i].primID();
            const uint64_t geomprimID = prims[i].ID64();
            if (geomprimID >= bestGeomPrimID) continue;
            const Vec3fa axis1 = scene->get(geomID)->computeDirection(primID);
            if (sqr_length(axis1) > 1E-18f) {
              axis = normalize(axis1);
              bestGeomPrimID = geomprimID;
            }
          }
          return frame(axis).transposed();
        }
        
        const PrimInfo computePrimInfo(const range<size_t>& set, const LinearSpace3fa& space)
        {
          auto computeBounds = [&](const range<size_t>& r) -> CentGeomBBox3fa
            {
              CentGeomBBox3fa bounds(empty);
              for (size_t i=r.begin(); i<r.end(); i++) {
                Geometry* mesh = scene->get(prims[i].geomID());
                bounds.extend(mesh->vbounds(space,prims[i].primID()));
              }
              return bounds;
            };
          
          const CentGeomBBox3fa bounds = parallel_reduce(set.begin(), set.end(), size_t(1024), size_t(4096), 
                                                         CentGeomBBox3fa(empty), computeBounds, CentGeomBBox3fa::merge2);

          return PrimInfo(set.begin(),set.end(),bounds);
        }

        struct BinBoundsAndCenter
        {
          __forceinline BinBoundsAndCenter(Scene* scene, const LinearSpace3fa& space)
            : scene(scene), space(space) {}
          
            /*! returns center for binning */
          __forceinline Vec3fa binCenter(const PrimRef& ref) const
          {
            Geometry* mesh = (Geometry*) scene->get(ref.geomID());
            BBox3fa bounds = mesh->vbounds(space,ref.primID());
            return embree::center2(bounds);
          }
          
          /*! returns bounds and centroid used for binning */
          __forceinline void binBoundsAndCenter(const PrimRef& ref, BBox3fa& bounds_o, Vec3fa& center_o) const
          {
            Geometry* mesh = (Geometry*) scene->get(ref.geomID());
            BBox3fa bounds = mesh->vbounds(space,ref.primID());
            bounds_o = bounds;
            center_o = embree::center2(bounds);
          }

        private:
          Scene* scene;
          const LinearSpace3fa space;
        };
        
        /*! finds the best split */
        __forceinline const Split find(const PrimInfoRange& pinfo, const size_t logBlockSize, const LinearSpace3fa& space)
        {
          if (likely(pinfo.size() < 10000))
            return find_template<false>(pinfo,logBlockSize,space);
          else
            return find_template<true>(pinfo,logBlockSize,space);
        }

        /*! finds the best split */
        template<bool parallel>
        const Split find_template(const PrimInfoRange& set, const size_t logBlockSize, const LinearSpace3fa& space)
        {
          Binner binner(empty);
          const BinMapping<BINS> mapping(set);
          BinBoundsAndCenter binBoundsAndCenter(scene,space);
          bin_serial_or_parallel<parallel>(binner,prims,set.begin(),set.end(),size_t(4096),mapping,binBoundsAndCenter);
          return binner.best(mapping,logBlockSize);
        }
        
        /*! array partitioning */
        __forceinline void split(const Split& split, const LinearSpace3fa& space, const Set& set, PrimInfoRange& lset, PrimInfoRange& rset)
        {
          if (likely(set.size() < 10000))
            split_template<false>(split,space,set,lset,rset);
          else
            split_template<true>(split,space,set,lset,rset);
        }

        /*! array partitioning */
        template<bool parallel>
        __forceinline void split_template(const Split& split, const LinearSpace3fa& space, const Set& set, PrimInfoRange& lset, PrimInfoRange& rset)
        {
          if (!split.valid()) {
            deterministic_order(set);
            return splitFallback(set,lset,rset);
          }
          
          const size_t begin = set.begin();
          const size_t end   = set.end();
          CentGeomBBox3fa local_left(empty);
          CentGeomBBox3fa local_right(empty);
          const int splitPos = split.pos;
          const int splitDim = split.dim;
          BinBoundsAndCenter binBoundsAndCenter(scene,space);

          size_t center = 0;
          if (likely(set.size() < 10000))
            center = serial_partitioning(prims,begin,end,local_left,local_right,
                                         [&] (const PrimRef& ref) { return split.mapping.bin_unsafe(ref,binBoundsAndCenter)[splitDim] < splitPos; },
                                         [] (CentGeomBBox3fa& pinfo,const PrimRef& ref) { pinfo.extend_center2(ref); });
          else
            center = parallel_partitioning(prims,begin,end,EmptyTy(),local_left,local_right,
                                           [&] (const PrimRef& ref) { return split.mapping.bin_unsafe(ref,binBoundsAndCenter)[splitDim] < splitPos; },
                                           [] (CentGeomBBox3fa& pinfo,const PrimRef& ref) { pinfo.extend_center2(ref); },
                                           [] (CentGeomBBox3fa& pinfo0,const CentGeomBBox3fa& pinfo1) { pinfo0.merge(pinfo1); },
                                           128);
          
          new (&lset) PrimInfoRange(begin,center,local_left);
          new (&rset) PrimInfoRange(center,end,local_right);
          assert(area(lset.geomBounds) >= 0.0f);
          assert(area(rset.geomBounds) >= 0.0f);
        }
        
        void deterministic_order(const range<size_t>& set) 
        {
          /* required as parallel partition destroys original primitive order */
          std::sort(&prims[set.begin()],&prims[set.end()]);
        }
        
        void splitFallback(const range<size_t>& set, PrimInfoRange& lset, PrimInfoRange& rset)
        {
          const size_t begin = set.begin();
          const size_t end   = set.end();
          const size_t center = (begin + end)/2;
          
          CentGeomBBox3fa left(empty);
          for (size_t i=begin; i<center; i++)
            left.extend_center2(prims[i]);
          new (&lset) PrimInfoRange(begin,center,left);
          
          CentGeomBBox3fa right(empty);
          for (size_t i=center; i<end; i++)
            right.extend_center2(prims[i]);
          new (&rset) PrimInfoRange(center,end,right);
        }
        
      private:
        Scene* const scene;
        PrimRef* const prims;
      };

    /*! Performs standard object binning */
    template<typename PrimRefMB, size_t BINS>
      struct UnalignedHeuristicArrayBinningMB
      {
        typedef BinSplit<BINS> Split;
        typedef typename PrimRefMB::BBox BBox;
        typedef BinInfoT<BINS,PrimRefMB,BBox> ObjectBinner;
        
        static const size_t PARALLEL_THRESHOLD = 3 * 1024;
        static const size_t PARALLEL_FIND_BLOCK_SIZE = 1024;
        static const size_t PARALLEL_PARTITION_BLOCK_SIZE = 128;

        UnalignedHeuristicArrayBinningMB(Scene* scene)
        : scene(scene) {}

        const LinearSpace3fa computeAlignedSpaceMB(Scene* scene, const SetMB& set)
        {
          Vec3fa axis0(0,0,1);
          uint64_t bestGeomPrimID = -1;

          /*! find curve with minimum ID that defines valid direction */
          for (size_t i=set.begin(); i<set.end(); i++)
          {
            const PrimRefMB& prim = (*set.prims)[i];
            const unsigned int geomID = prim.geomID();
            const unsigned int primID = prim.primID();
            const uint64_t geomprimID = prim.ID64();
            if (geomprimID >= bestGeomPrimID) continue;
            
            const Geometry* mesh = scene->get(geomID);
            const range<int> tbounds = mesh->timeSegmentRange(set.time_range);
            if (tbounds.size() == 0) continue;

            const size_t t = (tbounds.begin()+tbounds.end())/2;
            const Vec3fa axis1 = mesh->computeDirection(primID,t);
            if (sqr_length(axis1) > 1E-18f) {
              axis0 = normalize(axis1);
              bestGeomPrimID = geomprimID;
            }
          }

          return frame(axis0).transposed();
        }

        struct BinBoundsAndCenter
        {
          __forceinline BinBoundsAndCenter(Scene* scene, BBox1f time_range, const LinearSpace3fa& space)
            : scene(scene), time_range(time_range), space(space) {}
          
          /*! returns center for binning */
          template<typename PrimRef>
          __forceinline Vec3fa binCenter(const PrimRef& ref) const
          {
            Geometry* mesh = scene->get(ref.geomID());
            LBBox3fa lbounds = mesh->vlinearBounds(space,ref.primID(),time_range);
            return center2(lbounds.interpolate(0.5f));
          }

          /*! returns bounds and centroid used for binning */
          __noinline void binBoundsAndCenter (const PrimRefMB& ref, BBox3fa& bounds_o, Vec3fa& center_o) const // __noinline is workaround for ICC16 bug under MacOSX
          {
            Geometry* mesh = scene->get(ref.geomID());
            LBBox3fa lbounds = mesh->vlinearBounds(space,ref.primID(),time_range);
            bounds_o = lbounds.interpolate(0.5f);
            center_o = center2(bounds_o);
          }

          /*! returns bounds and centroid used for binning */
          __noinline void binBoundsAndCenter (const PrimRefMB& ref, LBBox3fa& bounds_o, Vec3fa& center_o) const // __noinline is workaround for ICC16 bug under MacOSX
          {
            Geometry* mesh = scene->get(ref.geomID());
            LBBox3fa lbounds = mesh->vlinearBounds(space,ref.primID(),time_range);
            bounds_o = lbounds;
            center_o = center2(lbounds.interpolate(0.5f));
          }
          
        private:
          Scene* scene;
          BBox1f time_range;
          const LinearSpace3fa space;
        };

        /*! finds the best split */
        const Split find(const SetMB& set, const size_t logBlockSize, const LinearSpace3fa& space)
        {
          BinBoundsAndCenter binBoundsAndCenter(scene,set.time_range,space);
          ObjectBinner binner(empty);
          const BinMapping<BINS> mapping(set.size(),set.centBounds);
          bin_parallel(binner,set.prims->data(),set.begin(),set.end(),PARALLEL_FIND_BLOCK_SIZE,PARALLEL_THRESHOLD,mapping,binBoundsAndCenter);
          Split osplit = binner.best(mapping,logBlockSize);
          osplit.sah *= set.time_range.size();
          if (!osplit.valid()) osplit.data = Split::SPLIT_FALLBACK; // use fallback split
          return osplit;
        }
        
        /*! array partitioning */
        __forceinline void split(const Split& split, const LinearSpace3fa& space, const SetMB& set, SetMB& lset, SetMB& rset)
        {
          BinBoundsAndCenter binBoundsAndCenter(scene,set.time_range,space);
          const size_t begin = set.begin();
          const size_t end   = set.end();
          PrimInfoMB left = empty;
          PrimInfoMB right = empty;
          const vint4 vSplitPos(split.pos);
          const vbool4 vSplitMask(1 << split.dim);
          auto isLeft = [&] (const PrimRefMB &ref) { return any(((vint4)split.mapping.bin_unsafe(ref,binBoundsAndCenter) < vSplitPos) & vSplitMask); };
          auto reduction = [] (PrimInfoMB& pinfo, const PrimRefMB& ref) { pinfo.add_primref(ref); };
          auto reduction2 = [] (PrimInfoMB& pinfo0,const PrimInfoMB& pinfo1) { pinfo0.merge(pinfo1); };
          size_t center = parallel_partitioning(set.prims->data(),begin,end,EmptyTy(),left,right,isLeft,reduction,reduction2,PARALLEL_PARTITION_BLOCK_SIZE,PARALLEL_THRESHOLD);
          new (&lset) SetMB(left,set.prims,range<size_t>(begin,center),set.time_range);
          new (&rset) SetMB(right,set.prims,range<size_t>(center,end ),set.time_range);
        }

      private:
        Scene* scene;
      };
  }
}
