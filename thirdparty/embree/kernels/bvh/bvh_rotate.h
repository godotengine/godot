// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bvh.h"

namespace embree
{
  namespace isa 
  { 
    template<int N>
    class BVHNRotate
    {
      typedef typename BVHN<N>::NodeRef NodeRef;

    public:
      static const bool enabled = false;

      static __forceinline size_t rotate(NodeRef parentRef, size_t depth = 1) { return 0; }
      static __forceinline void restructure(NodeRef ref, size_t depth = 1) {}
    };

    /* BVH4 tree rotations */
    template<>
    class BVHNRotate<4>
    {
      typedef BVH4::AABBNode AABBNode;
      typedef BVH4::NodeRef NodeRef;
      
    public:
      static const bool enabled = true;

      static size_t rotate(NodeRef parentRef, size_t depth = 1);
    };
  }
}
