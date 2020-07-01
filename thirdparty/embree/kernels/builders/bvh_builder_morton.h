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

#include "../common/builder.h"
#include "../../common/algorithms/parallel_reduce.h"

namespace embree
{
  namespace isa
  {
    struct BVHBuilderMorton
    {
      static const size_t MAX_BRANCHING_FACTOR = 8;          //!< maximum supported BVH branching factor
      static const size_t MIN_LARGE_LEAF_LEVELS = 8;         //!< create balanced tree of we are that many levels before the maximum tree depth

      /*! settings for morton builder */
      struct Settings
      {
        /*! default settings */
        Settings ()
        : branchingFactor(2), maxDepth(32), minLeafSize(1), maxLeafSize(8), singleThreadThreshold(1024) {}

        /*! initialize settings from API settings */
        Settings (const RTCBuildArguments& settings)
        : branchingFactor(2), maxDepth(32), minLeafSize(1), maxLeafSize(8), singleThreadThreshold(1024)
        {
          if (RTC_BUILD_ARGUMENTS_HAS(settings,maxBranchingFactor)) branchingFactor = settings.maxBranchingFactor;
          if (RTC_BUILD_ARGUMENTS_HAS(settings,maxDepth          )) maxDepth        = settings.maxDepth;
          if (RTC_BUILD_ARGUMENTS_HAS(settings,minLeafSize       )) minLeafSize     = settings.minLeafSize;
          if (RTC_BUILD_ARGUMENTS_HAS(settings,maxLeafSize       )) maxLeafSize     = settings.maxLeafSize;
        }

        Settings (size_t branchingFactor, size_t maxDepth, size_t minLeafSize, size_t maxLeafSize, size_t singleThreadThreshold)
        : branchingFactor(branchingFactor), maxDepth(maxDepth), minLeafSize(minLeafSize), maxLeafSize(maxLeafSize), singleThreadThreshold(singleThreadThreshold) {}

      public:
        size_t branchingFactor;  //!< branching factor of BVH to build
        size_t maxDepth;         //!< maximum depth of BVH to build
        size_t minLeafSize;      //!< minimum size of a leaf
        size_t maxLeafSize;      //!< maximum size of a leaf
        size_t singleThreadThreshold; //!< threshold when we switch to single threaded build
      };

      /*! Build primitive consisting of morton code and primitive ID. */
      struct __aligned(8) BuildPrim
      {
        union {
          struct {
            unsigned int code;     //!< morton code
            unsigned int index;    //!< i'th primitive
          };
          uint64_t t;
        };

        /*! interface for radix sort */
        __forceinline operator unsigned() const { return code; }

        /*! interface for standard sort */
        __forceinline bool operator<(const BuildPrim &m) const { return code < m.code; }
      };

      /*! maps bounding box to morton code */
      struct MortonCodeMapping
      {
        static const size_t LATTICE_BITS_PER_DIM = 10;
        static const size_t LATTICE_SIZE_PER_DIM = size_t(1) << LATTICE_BITS_PER_DIM;

        vfloat4 base;
        vfloat4 scale;

        __forceinline MortonCodeMapping(const BBox3fa& bounds)
        {
          base  = (vfloat4)bounds.lower;
          const vfloat4 diag  = (vfloat4)bounds.upper - (vfloat4)bounds.lower;
          scale = select(diag > vfloat4(1E-19f), rcp(diag) * vfloat4(LATTICE_SIZE_PER_DIM * 0.99f),vfloat4(0.0f));
        }

        __forceinline const vint4 bin (const BBox3fa& box) const
        {
          const vfloat4 lower = (vfloat4)box.lower;
          const vfloat4 upper = (vfloat4)box.upper;
          const vfloat4 centroid = lower+upper;
          return vint4((centroid-base)*scale);
        }

        __forceinline unsigned int code (const BBox3fa& box) const
        {
          const vint4 binID = bin(box);
          const unsigned int x = extract<0>(binID);
          const unsigned int y = extract<1>(binID);
          const unsigned int z = extract<2>(binID);
          const unsigned int xyz = bitInterleave(x,y,z);
          return xyz;
        }
      };

#if defined (__AVX2__)

      /*! for AVX2 there is a fast scalar bitInterleave */
      struct MortonCodeGenerator
      {
        __forceinline MortonCodeGenerator(const MortonCodeMapping& mapping, BuildPrim* dest)
          : mapping(mapping), dest(dest) {}

        __forceinline void operator() (const BBox3fa& b, const unsigned index)
        {
          dest->index = index;
          dest->code = mapping.code(b);
          dest++;
        }

      public:
        const MortonCodeMapping mapping;
        BuildPrim* dest;
        size_t currentID;
      };

#else

      /*! before AVX2 is it better to use the SSE version of bitInterleave */
      struct MortonCodeGenerator
      {
        __forceinline MortonCodeGenerator(const MortonCodeMapping& mapping, BuildPrim* dest)
          : mapping(mapping), dest(dest), currentID(0), slots(0), ax(0), ay(0), az(0), ai(0) {}

        __forceinline ~MortonCodeGenerator()
        {
          if (slots != 0)
          {
            const vint4 code = bitInterleave(ax,ay,az);
            for (size_t i=0; i<slots; i++) {
              dest[currentID-slots+i].index = ai[i];
              dest[currentID-slots+i].code = code[i];
            }
          }
        }

        __forceinline void operator() (const BBox3fa& b, const unsigned index)
        {
          const vint4 binID = mapping.bin(b);
          ax[slots] = extract<0>(binID);
          ay[slots] = extract<1>(binID);
          az[slots] = extract<2>(binID);
          ai[slots] = index;
          slots++;
          currentID++;

          if (slots == 4)
          {
            const vint4 code = bitInterleave(ax,ay,az);
            vint4::storeu(&dest[currentID-4],unpacklo(code,ai));
            vint4::storeu(&dest[currentID-2],unpackhi(code,ai));
            slots = 0;
          }
        }

      public:
        const MortonCodeMapping mapping;
        BuildPrim* dest;
        size_t currentID;
        size_t slots;
        vint4 ax, ay, az, ai;
      };

#endif

      template<
        typename ReductionTy,
        typename Allocator,
        typename CreateAllocator,
        typename CreateNodeFunc,
        typename SetNodeBoundsFunc,
        typename CreateLeafFunc,
        typename CalculateBounds,
        typename ProgressMonitor>

        class BuilderT : private Settings
      {
        ALIGNED_CLASS_(16);

      public:

        BuilderT (CreateAllocator& createAllocator,
                  CreateNodeFunc& createNode,
                  SetNodeBoundsFunc& setBounds,
                  CreateLeafFunc& createLeaf,
                  CalculateBounds& calculateBounds,
                  ProgressMonitor& progressMonitor,
                  const Settings& settings)

          : Settings(settings),
          createAllocator(createAllocator),
          createNode(createNode),
          setBounds(setBounds),
          createLeaf(createLeaf),
          calculateBounds(calculateBounds),
          progressMonitor(progressMonitor),
          morton(nullptr) {}

        ReductionTy createLargeLeaf(size_t depth, const range<unsigned>& current, Allocator alloc)
        {
          /* this should never occur but is a fatal error */
          if (depth > maxDepth)
            throw_RTCError(RTC_ERROR_UNKNOWN,"depth limit reached");

          /* create leaf for few primitives */
          if (current.size() <= maxLeafSize)
            return createLeaf(current,alloc);

          /* fill all children by always splitting the largest one */
          range<unsigned> children[MAX_BRANCHING_FACTOR];
          size_t numChildren = 1;
          children[0] = current;

          do {

            /* find best child with largest number of primitives */
            size_t bestChild = -1;
            size_t bestSize = 0;
            for (size_t i=0; i<numChildren; i++)
            {
              /* ignore leaves as they cannot get split */
              if (children[i].size() <= maxLeafSize)
                continue;

              /* remember child with largest size */
              if (children[i].size() > bestSize) {
                bestSize = children[i].size();
                bestChild = i;
              }
            }
            if (bestChild == size_t(-1)) break;

            /*! split best child into left and right child */
            auto split = children[bestChild].split();

            /* add new children left and right */
            children[bestChild] = children[numChildren-1];
            children[numChildren-1] = split.first;
            children[numChildren+0] = split.second;
            numChildren++;

          } while (numChildren < branchingFactor);

          /* create node */
          auto node = createNode(alloc,numChildren);

          /* recurse into each child */
          ReductionTy bounds[MAX_BRANCHING_FACTOR];
          for (size_t i=0; i<numChildren; i++)
            bounds[i] = createLargeLeaf(depth+1,children[i],alloc);

          return setBounds(node,bounds,numChildren);
        }

        /*! recreates morton codes when reaching a region where all codes are identical */
        __noinline void recreateMortonCodes(const range<unsigned>& current) const
        {
          /* fast path for small ranges */
          if (likely(current.size() < 1024))
          {
            /*! recalculate centroid bounds */
            BBox3fa centBounds(empty);
            for (size_t i=current.begin(); i<current.end(); i++)
              centBounds.extend(center2(calculateBounds(morton[i])));

            /* recalculate morton codes */
            MortonCodeMapping mapping(centBounds);
            for (size_t i=current.begin(); i<current.end(); i++)
              morton[i].code = mapping.code(calculateBounds(morton[i]));

            /* sort morton codes */
            std::sort(morton+current.begin(),morton+current.end());
          }
          else
          {
            /*! recalculate centroid bounds */
            auto calculateCentBounds = [&] ( const range<unsigned>& r ) {
              BBox3fa centBounds = empty;
              for (size_t i=r.begin(); i<r.end(); i++)
                centBounds.extend(center2(calculateBounds(morton[i])));
              return centBounds;
            };
            const BBox3fa centBounds = parallel_reduce(current.begin(), current.end(), unsigned(1024),
                                                       BBox3fa(empty), calculateCentBounds, BBox3fa::merge);

            /* recalculate morton codes */
            MortonCodeMapping mapping(centBounds);
            parallel_for(current.begin(), current.end(), unsigned(1024), [&] ( const range<unsigned>& r ) {
                for (size_t i=r.begin(); i<r.end(); i++) {
                  morton[i].code = mapping.code(calculateBounds(morton[i]));
                }
              });

            /*! sort morton codes */
#if defined(TASKING_TBB)
            tbb::parallel_sort(morton+current.begin(),morton+current.end());
#else
            radixsort32(morton+current.begin(),current.size());
#endif
          }
        }

        __forceinline void split(const range<unsigned>& current, range<unsigned>& left, range<unsigned>& right) const
        {
          const unsigned int code_start = morton[current.begin()].code;
          const unsigned int code_end   = morton[current.end()-1].code;
          unsigned int bitpos = lzcnt(code_start^code_end);

          /* if all items mapped to same morton code, then re-create new morton codes for the items */
          if (unlikely(bitpos == 32))
          {
            recreateMortonCodes(current);
            const unsigned int code_start = morton[current.begin()].code;
            const unsigned int code_end   = morton[current.end()-1].code;
            bitpos = lzcnt(code_start^code_end);

            /* if the morton code is still the same, goto fall back split */
            if (unlikely(bitpos == 32)) {
              current.split(left,right);
              return;
            }
          }

          /* split the items at the topmost different morton code bit */
          const unsigned int bitpos_diff = 31-bitpos;
          const unsigned int bitmask = 1 << bitpos_diff;

          /* find location where bit differs using binary search */
          unsigned begin = current.begin();
          unsigned end   = current.end();
          while (begin + 1 != end) {
            const unsigned mid = (begin+end)/2;
            const unsigned bit = morton[mid].code & bitmask;
            if (bit == 0) begin = mid; else end = mid;
          }
          unsigned center = end;
#if defined(DEBUG)
          for (unsigned int i=begin;  i<center; i++) assert((morton[i].code & bitmask) == 0);
          for (unsigned int i=center; i<end;    i++) assert((morton[i].code & bitmask) == bitmask);
#endif

          left = make_range(current.begin(),center);
          right = make_range(center,current.end());
        }

        ReductionTy recurse(size_t depth, const range<unsigned>& current, Allocator alloc, bool toplevel)
        {
          /* get thread local allocator */
          if (!alloc)
            alloc = createAllocator();

          /* call memory monitor function to signal progress */
          if (toplevel && current.size() <= singleThreadThreshold)
            progressMonitor(current.size());

          /* create leaf node */
          if (unlikely(depth+MIN_LARGE_LEAF_LEVELS >= maxDepth || current.size() <= minLeafSize))
            return createLargeLeaf(depth,current,alloc);

          /* fill all children by always splitting the one with the largest surface area */
          range<unsigned> children[MAX_BRANCHING_FACTOR];
          split(current,children[0],children[1]);
          size_t numChildren = 2;

          while (numChildren < branchingFactor)
          {
            /* find best child with largest number of primitives */
            int bestChild = -1;
            unsigned bestItems = 0;
            for (unsigned int i=0; i<numChildren; i++)
            {
              /* ignore leaves as they cannot get split */
              if (children[i].size() <= minLeafSize)
                continue;

              /* remember child with largest area */
              if (children[i].size() > bestItems) {
                bestItems = children[i].size();
                bestChild = i;
              }
            }
            if (bestChild == -1) break;

            /*! split best child into left and right child */
            range<unsigned> left, right;
            split(children[bestChild],left,right);

            /* add new children left and right */
            children[bestChild] = children[numChildren-1];
            children[numChildren-1] = left;
            children[numChildren+0] = right;
            numChildren++;
          }

          /* create leaf node if no split is possible */
          if (unlikely(numChildren == 1))
            return createLeaf(current,alloc);

          /* allocate node */
          auto node = createNode(alloc,numChildren);

          /* process top parts of tree parallel */
          ReductionTy bounds[MAX_BRANCHING_FACTOR];
          if (current.size() > singleThreadThreshold)
          {
            /*! parallel_for is faster than spawing sub-tasks */
            parallel_for(size_t(0), numChildren, [&] (const range<size_t>& r) {
                for (size_t i=r.begin(); i<r.end(); i++) {
                  bounds[i] = recurse(depth+1,children[i],nullptr,true);
                  _mm_mfence(); // to allow non-temporal stores during build
                }
              });
          }

          /* finish tree sequentially */
          else
          {
            for (size_t i=0; i<numChildren; i++)
              bounds[i] = recurse(depth+1,children[i],alloc,false);
          }

          return setBounds(node,bounds,numChildren);
        }

        /* build function */
        ReductionTy build(BuildPrim* src, BuildPrim* tmp, size_t numPrimitives)
        {
          /* sort morton codes */
          morton = src;
          radix_sort_u32(src,tmp,numPrimitives,singleThreadThreshold);

          /* build BVH */
          const ReductionTy root = recurse(1, range<unsigned>(0,(unsigned)numPrimitives), nullptr, true);
          _mm_mfence(); // to allow non-temporal stores during build
          return root;
        }

      public:
        CreateAllocator& createAllocator;
        CreateNodeFunc& createNode;
        SetNodeBoundsFunc& setBounds;
        CreateLeafFunc& createLeaf;
        CalculateBounds& calculateBounds;
        ProgressMonitor& progressMonitor;

      public:
        BuildPrim* morton;
      };


      template<
      typename ReductionTy,
        typename CreateAllocFunc,
        typename CreateNodeFunc,
        typename SetBoundsFunc,
        typename CreateLeafFunc,
        typename CalculateBoundsFunc,
        typename ProgressMonitor>

        static ReductionTy build(CreateAllocFunc createAllocator,
                                 CreateNodeFunc createNode,
                                 SetBoundsFunc setBounds,
                                 CreateLeafFunc createLeaf,
                                 CalculateBoundsFunc calculateBounds,
                                 ProgressMonitor progressMonitor,
                                 BuildPrim* src,
                                 BuildPrim* tmp,
                                 size_t numPrimitives,
                                 const Settings& settings)
        {
          typedef BuilderT<
            ReductionTy,
            decltype(createAllocator()),
            CreateAllocFunc,
            CreateNodeFunc,
            SetBoundsFunc,
            CreateLeafFunc,
            CalculateBoundsFunc,
            ProgressMonitor> Builder;

          Builder builder(createAllocator,
                          createNode,
                          setBounds,
                          createLeaf,
                          calculateBounds,
                          progressMonitor,
                          settings);

          return builder.build(src,tmp,numPrimitives);
        }
    };
  }
}
