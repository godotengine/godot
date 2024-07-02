// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(ZE_RAYTRACING)
#include "sys/platform.h"
#else
#include "../../../common/sys/platform.h"
#endif

namespace embree
{
  struct BVHStatistics
  {
    struct NodeStat
    {
      NodeStat ( double nodeSAH = 0,
                 size_t numNodes = 0, 
                 size_t numChildrenUsed = 0,
                 size_t numChildrenTotal = 0,
                 size_t numBytes = 0)
        : nodeSAH(nodeSAH),
        numNodes(numNodes), 
        numChildrenUsed(numChildrenUsed),
        numChildrenTotal(numChildrenTotal),
        numBytes(numBytes) {}
      
      double sah()   const { return nodeSAH; }
      size_t bytes() const { return numBytes; }
      size_t size()  const { return numNodes; }
      
      double fillRateNom () const { return double(numChildrenUsed);  }
      double fillRateDen () const { return double(numChildrenTotal);  }
      double fillRate    () const { return fillRateDen() ? fillRateNom()/fillRateDen() : 0.0; }

      friend NodeStat operator+ ( const NodeStat& a, const NodeStat& b)
      {
        return NodeStat(a.nodeSAH + b.nodeSAH,
                        a.numNodes+b.numNodes,
                        a.numChildrenUsed+b.numChildrenUsed,
                        a.numChildrenTotal+b.numChildrenTotal,
                        a.numBytes+b.numBytes);
      }
            
      void print(std::ostream& cout, double totalSAH, size_t totalBytes, size_t numPrimitives) const;
      
    public:
      double nodeSAH;
      size_t numNodes;
      size_t numChildrenUsed;
      size_t numChildrenTotal;
      size_t numBytes;
    };
    
    struct LeafStat
    {
      LeafStat(double leafSAH = 0.0f,
        size_t numLeaves = 0,
        size_t numBlocks = 0,
        size_t numPrimsUsed = 0,
        size_t numPrimsTotal = 0,
        size_t numBytesUsed = 0,
        size_t numBytesTotal = 0)
        : leafSAH(leafSAH),
        numLeaves(numLeaves),
        numBlocks(numBlocks),
        numPrimsUsed(numPrimsUsed),
        numPrimsTotal(numPrimsTotal),
        numBytesUsed(numBytesUsed),
        numBytesTotal(numBytesTotal) {}
      
      double sah()   const { return leafSAH; }
      size_t bytes() const { return numBytesTotal; }
      size_t size()  const { return numLeaves; }
      
      double fillRateNom () const { return double(numPrimsUsed);  }
      double fillRateDen () const { return double(numPrimsTotal);  }
      double fillRate    () const { return fillRateDen() ? fillRateNom()/fillRateDen() : 0.0; }

      friend LeafStat operator+ ( const LeafStat& a, const LeafStat& b)
      {
        return LeafStat(a.leafSAH + b.leafSAH,
                        a.numLeaves+b.numLeaves,
                        a.numBlocks+b.numBlocks,
                        a.numPrimsUsed+b.numPrimsUsed,
                        a.numPrimsTotal+b.numPrimsTotal,
                        a.numBytesUsed+b.numBytesUsed,
                        a.numBytesTotal+b.numBytesTotal);
      }
      
      void print(std::ostream& cout, double totalSAH, size_t totalBytes, size_t numPrimitives, bool blocks = false) const;

    public:
      double leafSAH;                    //!< SAH of the leaves only
      size_t numLeaves;                  //!< Number of leaf nodes.
      size_t numBlocks;                  //!< Number of blocks referenced
      size_t numPrimsUsed;               //!< Number of active primitives
      size_t numPrimsTotal;              //!< Number of active and inactive primitives
      size_t numBytesUsed;               //!< Number of used bytes
      size_t numBytesTotal;              //!< Number of total bytes of leaves.
    };

    BVHStatistics ()
    : numScenePrimitives(0), numBuildPrimitives(0), numBuildPrimitivesPostSplit(0) {}
        
    void print    (std::ostream& cout) const;
    void print_raw(std::ostream& cout) const;

    size_t numScenePrimitives;
    size_t numBuildPrimitives;
    size_t numBuildPrimitivesPostSplit;
    NodeStat internalNode;
    LeafStat quadLeaf;
    LeafStat proceduralLeaf;
    LeafStat instanceLeaf;
  };
}
