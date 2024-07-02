// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qbvh6.h"

namespace embree
{
  template<typename InternalNode>
  void computeInternalNodeStatistics(BVHStatistics& stats, QBVH6::Node node, const BBox1f time_range, const float node_bounds_area, const float root_bounds_area)
  {
    InternalNode* inner = node.innerNode<InternalNode>();

    size_t size = 0;
    for (uint32_t i = 0; i < InternalNode::NUM_CHILDREN; i++)
    {
      if (inner->valid(i))
      {
        size++;
        computeStatistics(stats, inner->child(i), time_range, area(inner->bounds(i)), root_bounds_area, InternalNode::NUM_CHILDREN);
      }
    }

    /* update BVH statistics */
    stats.internalNode.numNodes++;
    stats.internalNode.numChildrenUsed += size;
    stats.internalNode.numChildrenTotal += InternalNode::NUM_CHILDREN;
    stats.internalNode.nodeSAH += time_range.size() * node_bounds_area / root_bounds_area;
    stats.internalNode.numBytes += sizeof(InternalNode);
  }

  void computeStatistics(BVHStatistics& stats, QBVH6::Node node, const BBox1f time_range, const float node_bounds_area, const float root_bounds_area, uint32_t numChildren)
  {
    switch (node.type)
    {
    case NODE_TYPE_INSTANCE:
    {
      stats.instanceLeaf.numLeaves++;
      stats.instanceLeaf.numPrimsUsed++;
      stats.instanceLeaf.numPrimsTotal++;
      stats.instanceLeaf.leafSAH += time_range.size() * node_bounds_area / root_bounds_area;
      stats.instanceLeaf.numBytesUsed += sizeof(InstanceLeaf);
      stats.instanceLeaf.numBytesTotal += sizeof(InstanceLeaf);
      break;
    }
    case NODE_TYPE_QUAD:
    {
      bool last = false;
      stats.quadLeaf.numLeaves++;

      do
      {
        QuadLeaf* quad = node.leafNodeQuad();
        node.node += sizeof(QuadLeaf);
        last = quad->isLast();

        stats.quadLeaf.numPrimsUsed += quad->size();
        stats.quadLeaf.numPrimsTotal += 2;
        stats.quadLeaf.numBytesUsed += quad->usedBytes();
        stats.quadLeaf.numBytesTotal += sizeof(QuadLeaf);
        stats.quadLeaf.leafSAH += quad->size() * time_range.size() * node_bounds_area / root_bounds_area;
        
      } while (!last);
      
      break;
    }
    case NODE_TYPE_PROCEDURAL:
    {
      /*if (node.leafNodeProcedural()->leafDesc.isProceduralInstance()) // FIXME: for some reason we always to into this case!?
      {
        stats.proceduralLeaf.numLeaves++;
        stats.proceduralLeaf.numPrimsUsed += 1;
        stats.proceduralLeaf.numPrimsTotal += 1;
        stats.proceduralLeaf.leafSAH += time_range.size() * node_bounds_area / root_bounds_area;
        stats.proceduralLeaf.numBytesUsed += sizeof(InstanceLeaf);
        stats.proceduralLeaf.numBytesTotal += sizeof(InstanceLeaf);
      }
      else*/
      {
        bool last = false;
        uint32_t currPrim = node.cur_prim;
        stats.proceduralLeaf.numLeaves++;
        
        do
        {
          ProceduralLeaf* leaf = node.leafNodeProcedural();     
          last = leaf->isLast(currPrim);

          if (currPrim == 0) {
            stats.proceduralLeaf.numBlocks++;
            stats.proceduralLeaf.numBytesUsed += leaf->usedBytes();
            stats.proceduralLeaf.numBytesTotal += sizeof(ProceduralLeaf);
          }
          
          uint32_t primsInBlock = leaf->size();
          
          stats.proceduralLeaf.numPrimsUsed++;
          stats.proceduralLeaf.numPrimsTotal++;
          stats.proceduralLeaf.leafSAH += time_range.size() * node_bounds_area / root_bounds_area;
          
          if (++currPrim >= primsInBlock) {
            currPrim = 0;
            node.node += sizeof(ProceduralLeaf);
          }
          
        } while (!last);
      }
      break;
    }
    case NODE_TYPE_INTERNAL:
    {
      computeInternalNodeStatistics<QBVH6::InternalNode6>(stats, node, time_range, node_bounds_area, root_bounds_area);
      break;
    }
    default:
      assert(false);
    }
  }

  BVHStatistics QBVH6::computeStatistics() const
  {
    BVHStatistics stats;
    if (empty()) return stats;
    embree::computeStatistics(stats,root(),BBox1f(0,1),area(bounds),area(bounds),6);
    return stats;
  }

  template<typename QInternalNode>
  void QBVH6::printInternalNodeStatistics(std::ostream& cout, QBVH6::Node node, uint32_t depth, uint32_t numChildren)
  {
    QInternalNode* inner = node.innerNode<QInternalNode>();
    inner->print(cout, depth, false);
    std::cout << std::endl;

    for (uint32_t i = 0; i < QInternalNode::NUM_CHILDREN; i++)
    {
      if (inner->valid(i))
        print(cout, inner->child(i), depth + 1, QInternalNode::NUM_CHILDREN);
    }

    cout << tab(depth) << "}" << std::endl;
  }

  void QBVH6::print( std::ostream& cout, QBVH6::Node node, uint32_t depth, uint32_t numChildren)
  {
    switch (node.type)
    {
    case NODE_TYPE_INSTANCE: {
      node.leafNodeInstance()->print(cout,depth);
      cout << std::endl;
      break;
    }
    case NODE_TYPE_QUAD:
    {
      std::cout << tab(depth) << "List {" << std::endl;
      
      bool last = false;
      
      do
      {
        QuadLeaf* quad = node.leafNodeQuad();
        node.node += sizeof(QuadLeaf);
        last = quad->isLast();

        quad->print(cout,depth+1);
        std::cout << std::endl;

      } while (!last);

      std::cout << tab(depth) << "}" << std::endl;
      break;
    }
    case NODE_TYPE_PROCEDURAL:
    {
      /*if (!node.leafNodeProcedural()->leafDesc.opaqueCullingEnabled())
      {
        InstanceLeaf* leaf = (InstanceLeaf*) node.node;
        leaf->print(cout,depth+1);
        std::cout << std::endl;
      }
      else*/
      {
        std::cout << tab(depth) << "List {" << std::endl;
      
        bool last = false;
        uint32_t currPrim = node.cur_prim;
        
        do
        {
          ProceduralLeaf* leaf = node.leafNodeProcedural();     
          last = leaf->isLast(currPrim);
          
          uint32_t primsInBlock = leaf->size();

          leaf->print(cout,currPrim,depth+1);
          std::cout << std::endl;
          
          if (++currPrim >= primsInBlock) {
            currPrim = 0;
            node.node += sizeof(ProceduralLeaf);
          }
          
        } while (!last);

        std::cout << tab(depth) << "}" << std::endl;
      }
      break;
    }
    case NODE_TYPE_INTERNAL:
    {
      printInternalNodeStatistics<QBVH6::InternalNode6>(cout, node, depth, numChildren);
      break;
    }
    default:
      std::cout << "{ INVALID_NODE }" << std::endl;
      //assert(false);
    }
  }

  unsigned* getBackPointersData(const QBVH6* base) { // FIXME: should be member function
    return (unsigned*)(((const char*)base) + 64 * base->backPointerDataStart);
  }

  unsigned getNumBackpointers(const QBVH6* base) { // FIXME: should be member function
    return ((base->backPointerDataEnd - base->backPointerDataStart) * 64) / sizeof(unsigned);
  }

  uint64_t getBackpointerChildOffset(const QBVH6* base, unsigned idx) { // FIXME: should be member function
    return 64 * uint64_t(base->nodeDataStart + idx);
  }

  uint64_t getParentFromBackpointerOffset(const QBVH6* base, unsigned idx) { // FIXME: should be member function
    return 64 * uint64_t(base->nodeDataStart + (getBackPointersData(base)[idx] >> 6));
  }

  void QBVH6::print ( std::ostream& cout ) const
  {
    
    cout << "QBVH @ "<< this <<" header: {\n";
    cout << "  rootNodeOffset = " << rootNodeOffset << std::endl;
    cout << "  bounds = " << bounds << std::endl;
    cout << "  nodeDataStart = " << nodeDataStart << std::endl;
    cout << "  nodeDataCur = " << nodeDataCur << std::endl;
    cout << "  leafDataStart = " << leafDataCur << std::endl;
    cout << "  leafDataCur = " << leafDataCur << std::endl;
    cout << "  proceduralDataStart = " << proceduralDataStart << std::endl;
    cout << "  proceduralDataCur = " << proceduralDataCur << std::endl;
    cout << "  backPointerDataStart = " << backPointerDataStart << std::endl;
    cout << "  backPointerDataEnd = " << backPointerDataEnd << std::endl;
    cout << "  numPrims = " << numPrims << std::endl;
    cout << "}" << std::endl;

    if (empty()) return;
    
    print(cout,root(),0,6);
    
    if (hasBackPointers())
    {
      cout << "backpointers: {\n";
      for (unsigned bp = 0; bp < getNumBackpointers(this); ++bp) {
        cout << " node @ offset " << (void*)getBackpointerChildOffset(this, bp) << " parent = " << (void*)getParentFromBackpointerOffset(this, bp) << ", num children = " << ((getBackPointersData(this)[bp] >> 3) & 0x7) << "\n";
      }
      cout << "}\n";
    }
  }
}
