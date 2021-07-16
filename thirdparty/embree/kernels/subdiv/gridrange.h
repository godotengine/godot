// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../common/default.h"

namespace embree
{
  struct __aligned(16) GridRange
  {
    unsigned int u_start;
    unsigned int u_end;
    unsigned int v_start;
    unsigned int v_end;

    __forceinline GridRange() {}

    __forceinline GridRange(unsigned int u_start, unsigned int u_end, unsigned int v_start, unsigned int v_end) 
      : u_start(u_start), u_end(u_end), v_start(v_start), v_end(v_end) {}

    __forceinline unsigned int width() const {
      return u_end-u_start+1;
    }

    __forceinline unsigned int height() const {
      return v_end-v_start+1;
    }

    __forceinline bool hasLeafSize() const
    {
      const unsigned int u_size = u_end-u_start+1;
      const unsigned int v_size = v_end-v_start+1;
      assert(u_size >= 1);
      assert(v_size >= 1);
      return u_size <= 3 && v_size <= 3;
    }

    static __forceinline unsigned int split(unsigned int start,unsigned int end)
    {
      const unsigned int center = (start+end)/2;
      assert (center > start);
      assert (center < end);
      return center;
    }

    __forceinline void split(GridRange& r0, GridRange& r1) const
    {
      assert( hasLeafSize() == false );
      const unsigned int u_size = u_end-u_start+1;
      const unsigned int v_size = v_end-v_start+1;
      r0 = *this;
      r1 = *this;

      if (u_size >= v_size)
      {
        const unsigned int u_mid = split(u_start,u_end);
        r0.u_end   = u_mid;
        r1.u_start = u_mid;
      }
      else
      {
        const unsigned int v_mid = split(v_start,v_end);
        r0.v_end   = v_mid;
        r1.v_start = v_mid;
      }
    }

    __forceinline unsigned int splitIntoSubRanges(GridRange r[4]) const
    {
      assert( !hasLeafSize() );
      unsigned int children = 0;
      GridRange first,second;
      split(first,second);

      if (first.hasLeafSize()) {
        r[0] = first;
        children++;
      } 
      else {
        first.split(r[0],r[1]);
        children += 2;
      }

      if (second.hasLeafSize())	{
        r[children] = second;
        children++;
      }
      else {
        second.split(r[children+0],r[children+1]);
        children += 2;
      }
      return children;      
    }
  };
}
