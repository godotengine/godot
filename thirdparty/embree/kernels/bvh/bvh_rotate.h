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
      typedef BVH4::AlignedNode AlignedNode;
      typedef BVH4::NodeRef NodeRef;
      
    public:
      static const bool enabled = true;

      static size_t rotate(NodeRef parentRef, size_t depth = 1);
    };
  }
}
