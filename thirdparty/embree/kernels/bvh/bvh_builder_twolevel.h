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

#include "bvh.h"
#include "../common/primref.h"
#include "../builders/priminfo.h"

namespace embree
{
  namespace isa
  {
    template<int N, typename Mesh>
    class BVHNBuilderTwoLevel : public Builder
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::AlignedNode AlignedNode;
      typedef typename BVH::NodeRef NodeRef;

    public:

      typedef void (*createMeshAccelTy)(Mesh* mesh, AccelData*& accel, Builder*& builder);

      struct BuildRef : public PrimRef
      {
      public:
        __forceinline BuildRef () {}

        __forceinline BuildRef (const BBox3fa& bounds, NodeRef node)
          : PrimRef(bounds,(size_t)node), node(node)
        {
          if (node.isLeaf())
            bounds_area = 0.0f;
          else
            bounds_area = area(this->bounds());
        }

        /* used by the open/merge bvh builder */
        __forceinline BuildRef (const BBox3fa& bounds, NodeRef node, const unsigned int geomID, const unsigned int numPrimitives)
          : PrimRef(bounds,geomID,numPrimitives), node(node)
        {
          /* important for relative buildref ordering */
          if (node.isLeaf())
            bounds_area = 0.0f;
          else
            bounds_area = area(this->bounds());
        }

        __forceinline size_t size() const {
          return primID();
        }

        friend bool operator< (const BuildRef& a, const BuildRef& b) {
          return a.bounds_area < b.bounds_area;
        }

        friend __forceinline std::ostream& operator<<(std::ostream& cout, const BuildRef& ref) {
          return cout << "{ lower = " << ref.lower << ", upper = " << ref.upper << ", center2 = " << ref.center2() << ", geomID = " << ref.geomID() << ", numPrimitives = " << ref.numPrimitives() << ", bounds_area = " << ref.bounds_area << " }";
        }

        __forceinline unsigned int numPrimitives() const { return primID(); }

      public:
        NodeRef node;
        float bounds_area;
      };


      __forceinline size_t openBuildRef(BuildRef &bref, BuildRef *const refs) {
        if (bref.node.isLeaf())
        {
          refs[0] = bref;
          return 1;
        }
        NodeRef ref = bref.node;
        unsigned int geomID   = bref.geomID();
        unsigned int numPrims = max((unsigned int)bref.numPrimitives() / N,(unsigned int)1);
        AlignedNode* node = ref.alignedNode();
        size_t n = 0;
        for (size_t i=0; i<N; i++) {
          if (node->child(i) == BVH::emptyNode) continue;
          refs[i] = BuildRef(node->bounds(i),node->child(i),geomID,numPrims);
          n++;
        }
        assert(n > 1);
        return n;        
      }
      
      /*! Constructor. */
      BVHNBuilderTwoLevel (BVH* bvh, Scene* scene, const createMeshAccelTy createMeshAcce, const size_t singleThreadThreshold = DEFAULT_SINGLE_THREAD_THRESHOLD);
      
      /*! Destructor */
      ~BVHNBuilderTwoLevel ();
      
      /*! builder entry point */
      void build();
      void deleteGeometry(size_t geomID);
      void clear();

      void open_sequential(const size_t extSize);

    public:
      
      struct BuilderState
      {
        BuilderState ()
        : builder(nullptr), quality(RTC_BUILD_QUALITY_LOW) {}

        BuilderState (const Ref<Builder>& builder, RTCBuildQuality quality)
        : builder(builder), quality(quality) {}
        
        void clear() {
          builder = nullptr;
          quality = RTC_BUILD_QUALITY_LOW;
        }
        
        Ref<Builder> builder;
        RTCBuildQuality quality;
      };
      
    public:
      BVH* bvh;
      std::vector<BVH*>& objects;
      std::vector<BuilderState> builders;
      
    public:
      Scene* scene;
      createMeshAccelTy createMeshAccel;
      
      mvector<BuildRef> refs;
      mvector<PrimRef> prims;
      std::atomic<int> nextRef;
      const size_t singleThreadThreshold;

      typedef mvector<BuildRef> bvector;

    };
  }
}
