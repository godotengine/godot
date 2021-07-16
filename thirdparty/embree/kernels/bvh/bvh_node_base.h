// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bvh_node_ref.h"

namespace embree
{
  
  /*! BVHN Base Node */
  template<typename NodeRef, int N>
    struct BaseNode_t
  {
    /*! Clears the node. */
    __forceinline void clear()
    {
      for (size_t i=0; i<N; i++)
        children[i] = NodeRef::emptyNode;
    }
    
    /*! Returns reference to specified child */
    __forceinline       NodeRef& child(size_t i)       { assert(i<N); return children[i]; }
    __forceinline const NodeRef& child(size_t i) const { assert(i<N); return children[i]; }
    
    /*! verifies the node */
    __forceinline bool verify() const
    {
      for (size_t i=0; i<N; i++) {
        if (child(i) == NodeRef::emptyNode) {
          for (; i<N; i++) {
            if (child(i) != NodeRef::emptyNode)
              return false;
          }
          break;
        }
      }
      return true;
    }
    
    NodeRef children[N];    //!< Pointer to the N children (can be a node or leaf)
  };
}
