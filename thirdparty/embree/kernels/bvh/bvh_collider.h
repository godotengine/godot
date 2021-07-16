// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bvh.h"
#include "../geometry/trianglev.h"
#include "../geometry/object.h"

namespace embree
{
  namespace isa
  {
    template<int N>
      class BVHNCollider
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::AABBNode AABBNode;

      struct CollideJob
      {
        CollideJob () {}
        
        CollideJob (NodeRef ref0, const BBox3fa& bounds0, size_t depth0,
                    NodeRef ref1, const BBox3fa& bounds1, size_t depth1)
        : ref0(ref0), bounds0(bounds0), depth0(depth0), ref1(ref1), bounds1(bounds1), depth1(depth1) {}
        
        NodeRef ref0;
        BBox3fa bounds0;
        size_t depth0;
        NodeRef ref1;
        BBox3fa bounds1;
        size_t depth1;
      };

      typedef vector_t<CollideJob, aligned_allocator<CollideJob,16>> jobvector;

      void split(const CollideJob& job, jobvector& jobs);
      
    public:
      __forceinline BVHNCollider (Scene* scene0, Scene* scene1, RTCCollideFunc callback, void* userPtr)
        : scene0(scene0), scene1(scene1), callback(callback), userPtr(userPtr) {}

    public:
      virtual void processLeaf(NodeRef leaf0, NodeRef leaf1) = 0;
      void collide_recurse(NodeRef node0, const BBox3fa& bounds0, NodeRef node1, const BBox3fa& bounds1, size_t depth0, size_t depth1);
      void collide_recurse_entry(NodeRef node0, const BBox3fa& bounds0, NodeRef node1, const BBox3fa& bounds1);
    
    protected:
      Scene* scene0;
      Scene* scene1;
      RTCCollideFunc callback;
      void* userPtr;
    };

    template<int N>
      class BVHNColliderUserGeom : public BVHNCollider<N>
    {
      typedef BVHN<N> BVH;
      typedef typename BVH::NodeRef NodeRef;
      typedef typename BVH::AABBNode AABBNode;

      __forceinline BVHNColliderUserGeom (Scene* scene0, Scene* scene1, RTCCollideFunc callback, void* userPtr)
        : BVHNCollider<N>(scene0,scene1,callback,userPtr) {}

      virtual void processLeaf(NodeRef leaf0, NodeRef leaf1);
    public:
      static void collide(BVH* __restrict__ bvh0, BVH* __restrict__ bvh1, RTCCollideFunc callback, void* userPtr);
    };
  }
}
