// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../geometry/primitive.h"
#include "../subdiv/subdivpatch1base.h"

namespace embree
{

  struct __aligned(64) SubdivPatch1 : public SubdivPatch1Base
  {
    struct Type : public PrimitiveType 
    {
      const char* name() const;
      size_t sizeActive(const char* This) const;
      size_t sizeTotal(const char* This) const;
      size_t getBytes(const char* This) const;
    };
    
    static Type type;

  public:

    /*! constructor for cached subdiv patch */
    SubdivPatch1 (const unsigned int gID,
                        const unsigned int pID,
                        const unsigned int subPatch,
                        const SubdivMesh *const mesh,
                        const size_t time,
                        const Vec2f uv[4],
                        const float edge_level[4],
                        const int subdiv[4],
                        const int simd_width) 
      : SubdivPatch1Base(gID,pID,subPatch,mesh,time,uv,edge_level,subdiv,simd_width) {}
  };
}
