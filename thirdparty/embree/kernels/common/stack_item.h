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

#include "default.h"

namespace embree
{
  /*! An item on the stack holds the node ID and distance of that node. */
  template<typename T>
  struct __aligned(16) StackItemT
  {
    /*! assert that the xchg function works */
    static_assert(sizeof(T) <= 12, "sizeof(T) <= 12 failed");

    __forceinline StackItemT() {}

    __forceinline StackItemT(T &ptr, unsigned &dist) : ptr(ptr), dist(dist) {}

    /*! use SSE instructions to swap stack items */
    __forceinline static void xchg(StackItemT& a, StackItemT& b) 
    { 
      const vfloat4 sse_a = vfloat4::load((float*)&a); 
      const vfloat4 sse_b = vfloat4::load((float*)&b);
      vfloat4::store(&a,sse_b);
      vfloat4::store(&b,sse_a);
    }

    /*! Sort 2 stack items. */
    __forceinline friend void sort(StackItemT& s1, StackItemT& s2) {
      if (s2.dist < s1.dist) xchg(s2,s1);
    }
    
    /*! Sort 3 stack items. */
    __forceinline friend void sort(StackItemT& s1, StackItemT& s2, StackItemT& s3)
    {
      if (s2.dist < s1.dist) xchg(s2,s1);
      if (s3.dist < s2.dist) xchg(s3,s2);
      if (s2.dist < s1.dist) xchg(s2,s1);
    }
    
    /*! Sort 4 stack items. */
    __forceinline friend void sort(StackItemT& s1, StackItemT& s2, StackItemT& s3, StackItemT& s4)
    {
      if (s2.dist < s1.dist) xchg(s2,s1);
      if (s4.dist < s3.dist) xchg(s4,s3);
      if (s3.dist < s1.dist) xchg(s3,s1);
      if (s4.dist < s2.dist) xchg(s4,s2);
      if (s3.dist < s2.dist) xchg(s3,s2);
    }

    /*! use SSE instructions to swap stack items */
    __forceinline static void cmp_xchg(vint4& a, vint4& b) 
    { 
#if defined(__AVX512VL__)
      const vboolf4 mask(shuffle<2,2,2,2>(b) < shuffle<2,2,2,2>(a));
#else
      const vboolf4 mask0(b < a);
      const vboolf4 mask(shuffle<2,2,2,2>(mask0));
#endif
      const vint4 c = select(mask,b,a);
      const vint4 d = select(mask,a,b);
      a = c;
      b = d;
    }
    
    /*! Sort 3 stack items. */
    __forceinline static void sort3(vint4& s1, vint4& s2, vint4& s3)
    {
      cmp_xchg(s2,s1);
      cmp_xchg(s3,s2);
      cmp_xchg(s2,s1);
    }
    
    /*! Sort 4 stack items. */
    __forceinline static void sort4(vint4& s1, vint4& s2, vint4& s3, vint4& s4)
    {
      cmp_xchg(s2,s1);
      cmp_xchg(s4,s3);
      cmp_xchg(s3,s1);
      cmp_xchg(s4,s2);
      cmp_xchg(s3,s2);
    }


    /*! Sort N stack items. */
    __forceinline friend void sort(StackItemT* begin, StackItemT* end)
    {
      for (StackItemT* i = begin+1; i != end; ++i)
      {
        const vfloat4 item = vfloat4::load((float*)i);
        const unsigned dist = i->dist;
        StackItemT* j = i;

        while ((j != begin) && ((j-1)->dist < dist))
        {
          vfloat4::store(j, vfloat4::load((float*)(j-1)));
          --j;
        }

        vfloat4::store(j, item);
      }
    }
    
  public:
    T ptr;
    unsigned dist;
  };

  /*! An item on the stack holds the node ID and active ray mask. */
  template<typename T>
  struct __aligned(8) StackItemMaskT
  {
    T ptr;
    size_t mask;
  };

  struct __aligned(8) StackItemMaskCoherent
  {
    size_t mask;
    size_t parent;
    size_t child;
  };
}
