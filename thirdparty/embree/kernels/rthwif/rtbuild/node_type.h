// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <iostream>

namespace embree
{
  /* The type of a node. */
  enum NodeType : uint8_t
  {
    NODE_TYPE_MIXED = 0x0,        // identifies a mixed internal node where each child can have a different type
    NODE_TYPE_INTERNAL = 0x0,     // internal BVH node with 6 children
    NODE_TYPE_INSTANCE = 0x1,     // instance leaf
    NODE_TYPE_PROCEDURAL = 0x3,   // procedural leaf
    NODE_TYPE_QUAD = 0x4,         // quad leaf
    NODE_TYPE_INVALID = 0x7       // indicates invalid node
  };

  /* output operator for NodeType */
  inline std::ostream& operator<<(std::ostream& _cout, const NodeType& _type)
  {
#if !defined(__RTRT_GSIM)
    switch (_type)
    {
    case NODE_TYPE_INTERNAL: _cout << "INTERNAL"; break;
    case NODE_TYPE_INSTANCE: _cout << "INSTANCE"; break;
    case NODE_TYPE_PROCEDURAL: _cout << "PROCEDURAL"; break;
    case NODE_TYPE_QUAD: _cout << "QUAD"; break;
    case NODE_TYPE_INVALID: _cout << "INVALID"; break;
    default: _cout << "INVALID NODE TYPE"; break;
    }
#endif
    return _cout;
  };

  /* 
     Sub-type definition for each NodeType
  */

  enum SubType : uint8_t
  {
    SUB_TYPE_NONE = 0,
    
    /* sub-type for NODE_TYPE_INTERNAL */
    SUB_TYPE_INTERNAL6 = 0x00,        // Xe+: internal node with 6 children

    /* Sub-type for NODE_TYPE_QUAD */
    SUB_TYPE_QUAD = 0,                // Xe+: standard quad leaf (64 bytes)

    /* Sub-type for NODE_TYPE_PROCEDURAL */
    SUB_TYPE_PROCEDURAL = 0,          // Xe+: standard procedural leaf
  };
}
