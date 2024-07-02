// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(ZE_RAYTRACING)
#include "sys/sysinfo.h"
#include "sys/vector.h"
#include "math/vec2.h"
#include "math/vec3.h"
#include "math/bbox.h"
#include "math/affinespace.h"
#else
#include "../../common/default.h"
#endif

namespace embree
{
  enum QuadifierType : uint16_t
  {
    QUADIFIER_PAIRED = 0xFFFF,   // indicates that triangle is paired with a previous triangle
    QUADIFIER_TRIANGLE = 0,      // indicates that this triangle cannot get paired
    QUADIFIER_QUAD = 1,          // all values > 0 and != 0xFFFF indicate offset to paired triangle
    QUADIFIER_MAX_DISTANCE = 31,
  };

  template<typename Ty, size_t N>
  struct static_deque
  {
    __forceinline Ty pop_front() {
      assert(size());
      return operator[](begin++);
    }

    __forceinline void push_back(const Ty& v) {
      assert(size() < N);
      operator[](end++) = v;
    }
    
    __forceinline size_t size() const {
      assert(end >= begin);
      return end-begin;
    }

    __forceinline bool full() const {
      return size() == N;
    }

    __forceinline void erase( size_t j )
    {
      assert(j >= begin && j < end);

      /* fast path as we mostly just merge with the subsequent triangle */
      if (likely(j == begin))
        begin++;

      /* fastest when left side is small */
      else if (j-begin < end-j-1) {
        for (size_t i=j; i>=begin+1; i--) operator[](i) = operator[](i-1);
        begin++;
      }

      /* fastest if right side is small */
      else {
        for (size_t i=j+1; i<end; i++) operator[](i-1) = operator[](i);
        end--;
      }
    }
    
    __forceinline       Ty& operator[] ( const size_t i )       { return array[i%N]; }
    __forceinline const Ty& operator[] ( const size_t i ) const { return array[i%N]; }
    
    Ty array[N];
    size_t begin = 0;
    size_t end = 0;
  };
            
  __forceinline bool pair_triangles(Vec3<uint32_t> a, Vec3<uint32_t> b, uint8_t& lb0, uint8_t& lb1, uint8_t& lb2)
  {
    const vuint<4> va(a.x,a.y,a.z,0);
    const vboolf<4> mb0 = vboolf<4>(0x8) | vuint<4>(b.x) == va;
    const vboolf<4> mb1 = vboolf<4>(0x8) | vuint<4>(b.y) == va;
    const vboolf<4> mb2 = vboolf<4>(0x8) | vuint<4>(b.z) == va;
    lb0 = bsf(movemask(mb0));
    lb1 = bsf(movemask(mb1));
    lb2 = bsf(movemask(mb2));
    return (lb0 == 3) + (lb1 == 3) + (lb2 == 3) <= 1;
  }

  template<typename GetTriangleFunc>
  __forceinline void merge_triangle_window( uint32_t geomID, static_deque<uint32_t,32>& triangleWindow, QuadifierType* quads_o, const GetTriangleFunc& getTriangle )
  {
    uint32_t primID0 = triangleWindow.pop_front();
    
    /* load first triangle */
    Vec3<uint32_t> tri0 = getTriangle(geomID, primID0);
    
    /* find a second triangle in triangle window to pair with */
    for ( size_t slot = triangleWindow.begin; slot != triangleWindow.end; ++slot )
    {
      /* load second triangle */
      uint32_t primID1 = triangleWindow[slot];
      Vec3<uint32_t> tri1 = getTriangle(geomID, primID1);
      
      /* try to pair triangles */
      uint8_t lb0,lb1,lb2;
      bool pair = pair_triangles(tri0,tri1,lb0,lb1,lb2);

      /* the offset between the triangles cannot be too large as hardware limits bits for offset encode */
      uint32_t prim_offset = primID1 - primID0;
      pair &= prim_offset <= QUADIFIER_MAX_DISTANCE;

      /* store pairing if successful */
      if (pair)
      {
        assert(prim_offset > 0 && prim_offset < QUADIFIER_PAIRED);
        quads_o[primID0] = (QuadifierType) prim_offset;
        quads_o[primID1] = QUADIFIER_PAIRED;
        triangleWindow.erase(slot);
        return;
      }
    }
    
    /* make a triangle if we fail to find a candiate to pair with */
    quads_o[primID0] = QUADIFIER_TRIANGLE;
  }
  
  template<typename GetTriangleFunc>
  inline size_t pair_triangles( uint32_t geomID, QuadifierType* quads_o, uint32_t primID0, uint32_t primID1, const GetTriangleFunc& getTriangle ) 
  {
    static_deque<uint32_t, 32> triangleWindow;

    size_t numTrianglePairs = 0;
    for (uint32_t primID=primID0; primID<primID1; primID++)
    {
      triangleWindow.push_back(primID);
      
      if (triangleWindow.full()) {
        merge_triangle_window(geomID, triangleWindow,quads_o,getTriangle);
        numTrianglePairs++;
      }
    }
    
    while (triangleWindow.size()) {
      merge_triangle_window(geomID, triangleWindow,quads_o,getTriangle);
      numTrianglePairs++;
    }

    return numTrianglePairs;
  }
}
