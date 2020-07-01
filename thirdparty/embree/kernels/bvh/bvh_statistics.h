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
#include <sstream>

namespace embree
{
  template<int N>
  class BVHNStatistics
  {
    typedef BVHN<N> BVH;
    typedef typename BVH::AlignedNode AlignedNode;
    typedef typename BVH::UnalignedNode UnalignedNode;
    typedef typename BVH::AlignedNodeMB AlignedNodeMB;
    typedef typename BVH::AlignedNodeMB4D AlignedNodeMB4D;
    typedef typename BVH::UnalignedNodeMB UnalignedNodeMB;
    typedef typename BVH::QuantizedNode QuantizedNode;

    typedef typename BVH::NodeRef NodeRef;

    struct Statistics 
    {
      template<typename Node>
        struct NodeStat
      {
        NodeStat ( double nodeSAH = 0,
                   size_t numNodes = 0, 
                   size_t numChildren = 0)
        : nodeSAH(nodeSAH),
          numNodes(numNodes), 
          numChildren(numChildren) {}
        
        double sah(BVH* bvh) const {
          return nodeSAH/bvh->getLinearBounds().expectedHalfArea();
        }

        size_t bytes() const {
          return numNodes*sizeof(Node);
        }

        size_t size() const {
          return numNodes;
        }

        double fillRateNom () const { return double(numChildren);  }
        double fillRateDen () const { return double(numNodes*N);  }
        double fillRate    () const { return fillRateNom()/fillRateDen(); }

        __forceinline friend NodeStat operator+ ( const NodeStat& a, const NodeStat& b)
        {
          return NodeStat(a.nodeSAH + b.nodeSAH,
                          a.numNodes+b.numNodes,
                          a.numChildren+b.numChildren);
        }

        std::string toString(BVH* bvh, double sahTotal, size_t bytesTotal) const
        {
          std::ostringstream stream;
          stream.setf(std::ios::fixed, std::ios::floatfield);
          stream << "sah = " << std::setw(7) << std::setprecision(3) << sah(bvh);
          stream << " (" << std::setw(6) << std::setprecision(2) << 100.0*sah(bvh)/sahTotal << "%), ";          
          stream << "#bytes = " << std::setw(7) << std::setprecision(2) << bytes()/1E6  << " MB ";
          stream << "(" << std::setw(6) << std::setprecision(2) << 100.0*double(bytes())/double(bytesTotal) << "%), ";
          stream << "#nodes = " << std::setw(7) << numNodes << " (" << std::setw(6) << std::setprecision(2) << 100.0*fillRate() << "% filled), ";
          stream << "#bytes/prim = " << std::setw(6) << std::setprecision(2) << double(bytes())/double(bvh->numPrimitives);
          return stream.str();
        }

      public:
        double nodeSAH;
        size_t numNodes;
        size_t numChildren;
      };

      struct LeafStat
      {
        static const int NHIST = 8;

        LeafStat ( double leafSAH = 0.0f, 
                   size_t numLeaves = 0,
                   size_t numPrimsActive = 0,
                   size_t numPrimsTotal = 0,
                   size_t numPrimBlocks = 0,
                   size_t numBytes = 0)
        : leafSAH(leafSAH),
          numLeaves(numLeaves),
          numPrimsActive(numPrimsActive),
          numPrimsTotal(numPrimsTotal),
          numPrimBlocks(numPrimBlocks),
          numBytes(numBytes)
        {
          for (size_t i=0; i<NHIST; i++)
            numPrimBlocksHistogram[i] = 0;
        }

        double sah(BVH* bvh) const {
          return leafSAH/bvh->getLinearBounds().expectedHalfArea();
        }

        size_t bytes(BVH* bvh) const {
          return numBytes;
        }

        size_t size() const {
          return numLeaves;
        }

        double fillRateNom (BVH* bvh) const { return double(numPrimsActive);  }
        double fillRateDen (BVH* bvh) const { return double(numPrimsTotal);  }
        double fillRate    (BVH* bvh) const { return fillRateNom(bvh)/fillRateDen(bvh); }

        __forceinline friend LeafStat operator+ ( const LeafStat& a, const LeafStat& b)
        {
          LeafStat stat(a.leafSAH + b.leafSAH,
                        a.numLeaves+b.numLeaves,
                        a.numPrimsActive+b.numPrimsActive,
                        a.numPrimsTotal+b.numPrimsTotal,
                        a.numPrimBlocks+b.numPrimBlocks,
                        a.numBytes+b.numBytes);
          for (size_t i=0; i<NHIST; i++) {
            stat.numPrimBlocksHistogram[i] += a.numPrimBlocksHistogram[i];
            stat.numPrimBlocksHistogram[i] += b.numPrimBlocksHistogram[i];
          }
          return stat;
        }

        std::string toString(BVH* bvh, double sahTotal, size_t bytesTotal) const
        {
          std::ostringstream stream;
          stream.setf(std::ios::fixed, std::ios::floatfield);
          stream << "sah = " << std::setw(7) << std::setprecision(3) << sah(bvh);
          stream << " (" << std::setw(6) << std::setprecision(2) << 100.0*sah(bvh)/sahTotal << "%), ";
          stream << "#bytes = " << std::setw(7) << std::setprecision(2) << double(bytes(bvh))/1E6  << " MB ";
          stream << "(" << std::setw(6) << std::setprecision(2) << 100.0*double(bytes(bvh))/double(bytesTotal) << "%), ";
          stream << "#nodes = " << std::setw(7) << numLeaves << " (" << std::setw(6) << std::setprecision(2) << 100.0*fillRate(bvh) << "% filled), ";
          stream << "#bytes/prim = " << std::setw(6) << std::setprecision(2) << double(bytes(bvh))/double(bvh->numPrimitives);
          return stream.str();
        }

        std::string histToString() const
        {
          std::ostringstream stream;
          stream.setf(std::ios::fixed, std::ios::floatfield);
          for (size_t i=0; i<NHIST; i++)
            stream << std::setw(6) << std::setprecision(2) << 100.0f*float(numPrimBlocksHistogram[i])/float(numLeaves) << "% ";
          return stream.str();
        }
     
      public:
        double leafSAH;                    //!< SAH of the leaves only
        size_t numLeaves;                  //!< Number of leaf nodes.
        size_t numPrimsActive;             //!< Number of active primitives (
        size_t numPrimsTotal;              //!< Number of active and inactive primitives
        size_t numPrimBlocks;              //!< Number of primitive blocks.
        size_t numBytes;                   //!< Number of bytes of leaves.
        size_t numPrimBlocksHistogram[8];
      };

    public:
      Statistics (size_t depth = 0,
                  LeafStat statLeaf = LeafStat(),
                  NodeStat<AlignedNode> statAlignedNodes = NodeStat<AlignedNode>(),
                  NodeStat<UnalignedNode> statUnalignedNodes = NodeStat<UnalignedNode>(),
                  NodeStat<AlignedNodeMB> statAlignedNodesMB = NodeStat<AlignedNodeMB>(),
                  NodeStat<AlignedNodeMB4D> statAlignedNodesMB4D = NodeStat<AlignedNodeMB4D>(),
                  NodeStat<UnalignedNodeMB> statUnalignedNodesMB = NodeStat<UnalignedNodeMB>(),
                  NodeStat<QuantizedNode> statQuantizedNodes = NodeStat<QuantizedNode>())

      : depth(depth), 
        statLeaf(statLeaf),
        statAlignedNodes(statAlignedNodes),
        statUnalignedNodes(statUnalignedNodes),
        statAlignedNodesMB(statAlignedNodesMB),
        statAlignedNodesMB4D(statAlignedNodesMB4D),
        statUnalignedNodesMB(statUnalignedNodesMB),
        statQuantizedNodes(statQuantizedNodes) {}

      double sah(BVH* bvh) const 
      {
        return statLeaf.sah(bvh) +
          statAlignedNodes.sah(bvh) + 
          statUnalignedNodes.sah(bvh) + 
          statAlignedNodesMB.sah(bvh) + 
          statAlignedNodesMB4D.sah(bvh) + 
          statUnalignedNodesMB.sah(bvh) + 
          statQuantizedNodes.sah(bvh);
      }
      
      size_t bytes(BVH* bvh) const {
        return statLeaf.bytes(bvh) +
          statAlignedNodes.bytes() + 
          statUnalignedNodes.bytes() + 
          statAlignedNodesMB.bytes() + 
          statAlignedNodesMB4D.bytes() + 
          statUnalignedNodesMB.bytes() + 
          statQuantizedNodes.bytes();
      }

      size_t size() const 
      {
        return statLeaf.size() +
          statAlignedNodes.size() + 
          statUnalignedNodes.size() + 
          statAlignedNodesMB.size() + 
          statAlignedNodesMB4D.size() + 
          statUnalignedNodesMB.size() + 
          statQuantizedNodes.size();
      }

      double fillRate (BVH* bvh) const 
      {
        double nom = statLeaf.fillRateNom(bvh) +
          statAlignedNodes.fillRateNom() + 
          statUnalignedNodes.fillRateNom() + 
          statAlignedNodesMB.fillRateNom() + 
          statAlignedNodesMB4D.fillRateNom() + 
          statUnalignedNodesMB.fillRateNom() + 
          statQuantizedNodes.fillRateNom();
        double den = statLeaf.fillRateDen(bvh) +
          statAlignedNodes.fillRateDen() + 
          statUnalignedNodes.fillRateDen() + 
          statAlignedNodesMB.fillRateDen() + 
          statAlignedNodesMB4D.fillRateDen() + 
          statUnalignedNodesMB.fillRateDen() + 
          statQuantizedNodes.fillRateDen();
        return nom/den;
      }

      friend Statistics operator+ ( const Statistics& a, const Statistics& b )
      {
        return Statistics(max(a.depth,b.depth),
                          a.statLeaf + b.statLeaf,
                          a.statAlignedNodes + b.statAlignedNodes,
                          a.statUnalignedNodes + b.statUnalignedNodes,
                          a.statAlignedNodesMB + b.statAlignedNodesMB,
                          a.statAlignedNodesMB4D + b.statAlignedNodesMB4D,
                          a.statUnalignedNodesMB + b.statUnalignedNodesMB,
                          a.statQuantizedNodes + b.statQuantizedNodes);
      }

      static Statistics add ( const Statistics& a, const Statistics& b ) {
        return a+b;
      }

    public:
      size_t depth;
      LeafStat statLeaf;
      NodeStat<AlignedNode> statAlignedNodes;
      NodeStat<UnalignedNode> statUnalignedNodes;
      NodeStat<AlignedNodeMB> statAlignedNodesMB;
      NodeStat<AlignedNodeMB4D> statAlignedNodesMB4D;
      NodeStat<UnalignedNodeMB> statUnalignedNodesMB;
      NodeStat<QuantizedNode> statQuantizedNodes;
    };

  public:

    /* Constructor gathers statistics. */
    BVHNStatistics (BVH* bvh);

    /*! Convert statistics into a string */
    std::string str();

    double sah() const { 
      return stat.sah(bvh); 
    }

    size_t bytesUsed() const {
      return stat.bytes(bvh);
    }

  private:
    Statistics statistics(NodeRef node, const double A, const BBox1f dt);

  private:
    BVH* bvh;
    Statistics stat;
  };

  typedef BVHNStatistics<4> BVH4Statistics;
  typedef BVHNStatistics<8> BVH8Statistics;
}
