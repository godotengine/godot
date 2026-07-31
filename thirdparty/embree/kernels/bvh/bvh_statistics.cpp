// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bvh_statistics.h"
#include "../../common/algorithms/parallel_reduce.h"

namespace embree
{
  template<int N>
  BVHNStatistics<N>::BVHNStatistics (BVH* bvh) : bvh(bvh)
  {
    double A = max(0.0f,bvh->getLinearBounds().expectedHalfArea());
    stat = statistics(bvh->root,A,BBox1f(0.0f,1.0f));
  }
  
  template<int N>
  std::string BVHNStatistics<N>::str()
  {
    std::ostringstream stream;
    stream.setf(std::ios::fixed, std::ios::floatfield);
    stream << "  primitives = " << bvh->numPrimitives << ", vertices = " << bvh->numVertices << ", depth = " << stat.depth << std::endl;
    size_t totalBytes = stat.bytes(bvh);
    double totalSAH = stat.sah(bvh);
    stream << "  total            : sah = "  << std::setw(7) << std::setprecision(3) << totalSAH << " (100.00%), ";
    stream << "#bytes = " << std::setw(7) << std::setprecision(2) << totalBytes/1E6 << " MB (100.00%), ";
    stream << "#nodes = " << std::setw(7) << stat.size() << " (" << std::setw(6) << std::setprecision(2) << 100.0*stat.fillRate(bvh) << "% filled), ";
    stream << "#bytes/prim = " << std::setw(6) << std::setprecision(2) << double(totalBytes)/double(bvh->numPrimitives) << std::endl;
    if (stat.statAABBNodes.numNodes    ) stream << "  getAABBNodes     : "  << stat.statAABBNodes.toString(bvh,totalSAH,totalBytes) << std::endl;
    if (stat.statOBBNodes.numNodes  ) stream << "  ungetAABBNodes   : "  << stat.statOBBNodes.toString(bvh,totalSAH,totalBytes) << std::endl;
    if (stat.statAABBNodesMB.numNodes  ) stream << "  getAABBNodesMB   : "  << stat.statAABBNodesMB.toString(bvh,totalSAH,totalBytes) << std::endl;
    if (stat.statAABBNodesMB4D.numNodes) stream << "  getAABBNodesMB4D : "  << stat.statAABBNodesMB4D.toString(bvh,totalSAH,totalBytes) << std::endl;
    if (stat.statOBBNodesMB.numNodes) stream << "  ungetAABBNodesMB : "  << stat.statOBBNodesMB.toString(bvh,totalSAH,totalBytes) << std::endl;
    if (stat.statQuantizedNodes.numNodes  ) stream << "  quantizedNodes   : "  << stat.statQuantizedNodes.toString(bvh,totalSAH,totalBytes) << std::endl;
    if (true)                               stream << "  leaves           : "  << stat.statLeaf.toString(bvh,totalSAH,totalBytes) << std::endl;
    if (true)                               stream << "    histogram      : "  << stat.statLeaf.histToString() << std::endl;
    return stream.str();
  }
  
  template<int N>
  typename BVHNStatistics<N>::Statistics BVHNStatistics<N>::statistics(NodeRef node, const double A, const BBox1f t0t1)
  {
    Statistics s;
    assert(t0t1.size() > 0.0f);
    double dt = max(0.0f,t0t1.size());
    if (node.isAABBNode())
    {
      AABBNode* n = node.getAABBNode();
      s = s + parallel_reduce(0,N,Statistics(),[&] ( const int i ) {
          if (n->child(i) == BVH::emptyNode) return Statistics();
          const double Ai = max(0.0f,halfArea(n->extend(i)));
          Statistics s = statistics(n->child(i),Ai,t0t1); 
          s.statAABBNodes.numChildren++;
          return s;
        }, Statistics::add);
      s.statAABBNodes.numNodes++;
      s.statAABBNodes.nodeSAH += dt*A;
      s.depth++;
    }
    else if (node.isOBBNode())
    {
      OBBNode* n = node.ungetAABBNode();
      s = s + parallel_reduce(0,N,Statistics(),[&] ( const int i ) {
          if (n->child(i) == BVH::emptyNode) return Statistics();
          const double Ai = max(0.0f,halfArea(n->extent(i)));
          Statistics s = statistics(n->child(i),Ai,t0t1); 
          s.statOBBNodes.numChildren++;
          return s;
        }, Statistics::add);
      s.statOBBNodes.numNodes++;
      s.statOBBNodes.nodeSAH += dt*A;
      s.depth++;
    }
    else if (node.isAABBNodeMB())
    {
      AABBNodeMB* n = node.getAABBNodeMB();
      s = s + parallel_reduce(0,N,Statistics(),[&] ( const int i ) {
          if (n->child(i) == BVH::emptyNode) return Statistics();
          const double Ai = max(0.0f,n->expectedHalfArea(i,t0t1));
          Statistics s = statistics(n->child(i),Ai,t0t1);
          s.statAABBNodesMB.numChildren++;
          return s;
        }, Statistics::add);
      s.statAABBNodesMB.numNodes++;
      s.statAABBNodesMB.nodeSAH += dt*A;
      s.depth++;
    }
    else if (node.isAABBNodeMB4D())
    {
      AABBNodeMB4D* n = node.getAABBNodeMB4D();
      s = s + parallel_reduce(0,N,Statistics(),[&] ( const int i ) {
          if (n->child(i) == BVH::emptyNode) return Statistics();
          const BBox1f t0t1i = intersect(t0t1,n->timeRange(i));
          assert(!t0t1i.empty());
          const double Ai = n->AABBNodeMB::expectedHalfArea(i,t0t1i);
          Statistics s =  statistics(n->child(i),Ai,t0t1i);
          s.statAABBNodesMB4D.numChildren++;
          return s;
        }, Statistics::add);
      s.statAABBNodesMB4D.numNodes++;
      s.statAABBNodesMB4D.nodeSAH += dt*A;
      s.depth++;
    }
    else if (node.isOBBNodeMB())
    {
      OBBNodeMB* n = node.ungetAABBNodeMB();
      s = s + parallel_reduce(0,N,Statistics(),[&] ( const int i ) {
          if (n->child(i) == BVH::emptyNode) return Statistics();
          const double Ai = max(0.0f,halfArea(n->extent0(i)));
          Statistics s = statistics(n->child(i),Ai,t0t1); 
          s.statOBBNodesMB.numChildren++;
          return s;
        }, Statistics::add);
      s.statOBBNodesMB.numNodes++;
      s.statOBBNodesMB.nodeSAH += dt*A;
      s.depth++;
    }
    else if (node.isQuantizedNode())
    {
      QuantizedNode* n = node.quantizedNode();
      s = s + parallel_reduce(0,N,Statistics(),[&] ( const int i ) {
          if (n->child(i) == BVH::emptyNode) return Statistics();
          const double Ai = max(0.0f,halfArea(n->extent(i)));
          Statistics s = statistics(n->child(i),Ai,t0t1); 
          s.statQuantizedNodes.numChildren++;
          return s;
        }, Statistics::add);
      s.statQuantizedNodes.numNodes++;
      s.statQuantizedNodes.nodeSAH += dt*A;
      s.depth++;
    }
    else if (node.isLeaf())
    {
      size_t num; const char* tri = node.leaf(num);
      if (num)
      {
        for (size_t i=0; i<num; i++)
        {
          const size_t bytes = bvh->primTy->getBytes(tri);
          s.statLeaf.numPrimsActive += bvh->primTy->sizeActive(tri);
          s.statLeaf.numPrimsTotal += bvh->primTy->sizeTotal(tri);
          s.statLeaf.numBytes += bytes;
          tri+=bytes;
        }
        s.statLeaf.numLeaves++;
        s.statLeaf.numPrimBlocks += num;
        s.statLeaf.leafSAH += dt*A*num;
        if (num-1 < Statistics::LeafStat::NHIST) {
          s.statLeaf.numPrimBlocksHistogram[num-1]++;
        }
      }
    }
    else {
      abort(); //throw std::runtime_error("not supported node type in bvh_statistics");
    }
    return s;
  } 

#if defined(__AVX__)
  template class BVHNStatistics<8>;
#endif

#if !defined(__AVX__) || (!defined(EMBREE_TARGET_SSE2) && !defined(EMBREE_TARGET_SSE42)) || defined(__aarch64__)
  template class BVHNStatistics<4>;
#endif
}
