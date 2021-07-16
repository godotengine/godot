// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bvh.h"
#include "../builders/bvh_builder_sah.h"
#include "../builders/bvh_builder_msmblur.h"

namespace embree
{
  namespace isa
  {
    /************************************************************************************/
    /************************************************************************************/
    /************************************************************************************/
    /************************************************************************************/

    template<int N>
      struct BVHNBuilderVirtual
      {
        typedef BVHN<N> BVH;
        typedef typename BVH::NodeRef NodeRef;
        typedef FastAllocator::CachedAllocator Allocator;
      
        struct BVHNBuilderV {
          NodeRef build(FastAllocator* allocator, BuildProgressMonitor& progress, PrimRef* prims, const PrimInfo& pinfo, GeneralBVHBuilder::Settings settings);
          virtual NodeRef createLeaf (const PrimRef* prims, const range<size_t>& set, const Allocator& alloc) = 0;
        };

        template<typename CreateLeafFunc>
        struct BVHNBuilderT : public BVHNBuilderV
        {
          BVHNBuilderT (CreateLeafFunc createLeafFunc)
            : createLeafFunc(createLeafFunc) {}

          NodeRef createLeaf (const PrimRef* prims, const range<size_t>& set, const Allocator& alloc) {
            return createLeafFunc(prims,set,alloc);
          }

        private:
          CreateLeafFunc createLeafFunc;
        };

        template<typename CreateLeafFunc>
        static NodeRef build(FastAllocator* allocator, CreateLeafFunc createLeaf, BuildProgressMonitor& progress, PrimRef* prims, const PrimInfo& pinfo, GeneralBVHBuilder::Settings settings) {
          return BVHNBuilderT<CreateLeafFunc>(createLeaf).build(allocator,progress,prims,pinfo,settings);
        }
      };

    template<int N>
      struct BVHNBuilderQuantizedVirtual
      {
        typedef BVHN<N> BVH;
        typedef typename BVH::NodeRef NodeRef;
        typedef FastAllocator::CachedAllocator Allocator;
      
        struct BVHNBuilderV {
          NodeRef build(FastAllocator* allocator, BuildProgressMonitor& progress, PrimRef* prims, const PrimInfo& pinfo, GeneralBVHBuilder::Settings settings);
          virtual NodeRef createLeaf (const PrimRef* prims, const range<size_t>& set, const Allocator& alloc) = 0;
        };

        template<typename CreateLeafFunc>
        struct BVHNBuilderT : public BVHNBuilderV
        {
          BVHNBuilderT (CreateLeafFunc createLeafFunc)
            : createLeafFunc(createLeafFunc) {}

          NodeRef createLeaf (const PrimRef* prims, const range<size_t>& set, const Allocator& alloc) {
            return createLeafFunc(prims,set,alloc);
          }

        private:
          CreateLeafFunc createLeafFunc;
        };

        template<typename CreateLeafFunc>
        static NodeRef build(FastAllocator* allocator, CreateLeafFunc createLeaf, BuildProgressMonitor& progress, PrimRef* prims, const PrimInfo& pinfo, GeneralBVHBuilder::Settings settings) {
          return BVHNBuilderT<CreateLeafFunc>(createLeaf).build(allocator,progress,prims,pinfo,settings);
        }
      };

    template<int N>
      struct BVHNBuilderMblurVirtual
      {
        typedef BVHN<N> BVH;
        typedef typename BVH::AABBNodeMB AABBNodeMB;
        typedef typename BVH::NodeRef NodeRef;
        typedef typename BVH::NodeRecordMB NodeRecordMB;
        typedef FastAllocator::CachedAllocator Allocator;
      
        struct BVHNBuilderV {
          NodeRecordMB build(FastAllocator* allocator, BuildProgressMonitor& progress, PrimRef* prims, const PrimInfo& pinfo, GeneralBVHBuilder::Settings settings, const BBox1f& timeRange);
          virtual NodeRecordMB createLeaf (const PrimRef* prims, const range<size_t>& set, const Allocator& alloc) = 0;
        };

        template<typename CreateLeafFunc>
        struct BVHNBuilderT : public BVHNBuilderV
        {
          BVHNBuilderT (CreateLeafFunc createLeafFunc)
            : createLeafFunc(createLeafFunc) {}

          NodeRecordMB createLeaf (const PrimRef* prims, const range<size_t>& set, const Allocator& alloc) {
            return createLeafFunc(prims,set,alloc);
          }

        private:
          CreateLeafFunc createLeafFunc;
        };

        template<typename CreateLeafFunc>
        static NodeRecordMB build(FastAllocator* allocator, CreateLeafFunc createLeaf, BuildProgressMonitor& progress, PrimRef* prims, const PrimInfo& pinfo, GeneralBVHBuilder::Settings settings, const BBox1f& timeRange) {
          return BVHNBuilderT<CreateLeafFunc>(createLeaf).build(allocator,progress,prims,pinfo,settings,timeRange);
        }
      };
  }
}
