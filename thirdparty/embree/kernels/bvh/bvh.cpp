// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bvh.h"
#include "bvh_statistics.h"

namespace embree
{
  template<int N>
  BVHN<N>::BVHN (const PrimitiveType& primTy, Scene* scene)
    : AccelData((N==4) ? AccelData::TY_BVH4 : (N==8) ? AccelData::TY_BVH8 : AccelData::TY_UNKNOWN),
      primTy(&primTy), device(scene->device), scene(scene),
      root(emptyNode), alloc(scene->device,scene->isStaticAccel()), numPrimitives(0), numVertices(0)
  {
  }

  template<int N>
  BVHN<N>::~BVHN ()
  {
    for (size_t i=0; i<objects.size(); i++) 
      delete objects[i];
  }

  template<int N>
  void BVHN<N>::clear()
  {
    set(BVHN::emptyNode,empty,0);
    alloc.clear();
  }

  template<int N>
  void BVHN<N>::set (NodeRef root, const LBBox3fa& bounds, size_t numPrimitives)
  {
    this->root = root;
    this->bounds = bounds;
    this->numPrimitives = numPrimitives;
  }	

  template<int N>
  void BVHN<N>::clearBarrier(NodeRef& node)
  {
    if (node.isBarrier())
      node.clearBarrier();
    else if (!node.isLeaf()) {
      BaseNode* n = node.baseNode(); // FIXME: flags should be stored in BVH
      for (size_t c=0; c<N; c++)
        clearBarrier(n->child(c));
    }
  }

  template<int N>
  void BVHN<N>::layoutLargeNodes(size_t num)
  {
#if defined(__64BIT__) // do not use tree rotations on 32 bit platforms, barrier bit in NodeRef will cause issues
    struct NodeArea 
    {
      __forceinline NodeArea() {}

      __forceinline NodeArea(NodeRef& node, const BBox3fa& bounds)
        : node(&node), A(node.isLeaf() ? float(neg_inf) : area(bounds)) {}

      __forceinline bool operator< (const NodeArea& other) const {
        return this->A < other.A;
      }

      NodeRef* node;
      float A;
    };
    std::vector<NodeArea> lst;
    lst.reserve(num);
    lst.push_back(NodeArea(root,empty));

    while (lst.size() < num)
    {
      std::pop_heap(lst.begin(), lst.end());
      NodeArea n = lst.back(); lst.pop_back();
      if (!n.node->isAABBNode()) break;
      AABBNode* node = n.node->getAABBNode();
      for (size_t i=0; i<N; i++) {
        if (node->child(i) == BVHN::emptyNode) continue;
        lst.push_back(NodeArea(node->child(i),node->bounds(i)));
        std::push_heap(lst.begin(), lst.end());
      }
    }

    for (size_t i=0; i<lst.size(); i++)
      lst[i].node->setBarrier();
      
    root = layoutLargeNodesRecursion(root,alloc.getCachedAllocator());
#endif
  }
  
  template<int N>
  typename BVHN<N>::NodeRef BVHN<N>::layoutLargeNodesRecursion(NodeRef& node, const FastAllocator::CachedAllocator& allocator)
  {
    if (node.isBarrier()) {
      node.clearBarrier();
      return node;
    }
    else if (node.isAABBNode()) 
    {
      AABBNode* oldnode = node.getAABBNode();
      AABBNode* newnode = (BVHN::AABBNode*) allocator.malloc0(sizeof(BVHN::AABBNode),byteNodeAlignment);
      *newnode = *oldnode;
      for (size_t c=0; c<N; c++)
        newnode->child(c) = layoutLargeNodesRecursion(oldnode->child(c),allocator);
      return encodeNode(newnode);
    }
    else return node;
  }

  template<int N>
  double BVHN<N>::preBuild(const std::string& builderName)
  {
    if (builderName == "") 
      return inf;

    if (device->verbosity(2))
    {
      Lock<MutexSys> lock(g_printMutex);
      std::cout << "building BVH" << N << (builderName.find("MBlur") != std::string::npos ? "MB" : "") << "<" << primTy->name() << "> using " << builderName << " ..." << std::endl << std::flush;
    }

    double t0 = 0.0;
    if (device->benchmark || device->verbosity(2)) t0 = getSeconds();
    return t0;
  }

  template<int N>
  void BVHN<N>::postBuild(double t0)
  {
    if (t0 == double(inf))
      return;
    
    double dt = 0.0;
    if (device->benchmark || device->verbosity(2)) 
      dt = getSeconds()-t0;

    std::unique_ptr<BVHNStatistics<N>> stat;

    /* print statistics */
    if (device->verbosity(2))
    {
      if (!stat) stat.reset(new BVHNStatistics<N>(this));
      const size_t usedBytes = alloc.getUsedBytes();
      Lock<MutexSys> lock(g_printMutex);
      std::cout << "finished BVH" << N << "<" << primTy->name() << "> : " << 1000.0f*dt << "ms, " << 1E-6*double(numPrimitives)/dt << " Mprim/s, " << 1E-9*double(usedBytes)/dt << " GB/s" << std::endl;
    
      if (device->verbosity(2))
        std::cout << stat->str();

      if (device->verbosity(2))
      {
        FastAllocator::AllStatistics stat(&alloc);
        for (size_t i=0; i<objects.size(); i++)
          if (objects[i])
            stat = stat + FastAllocator::AllStatistics(&objects[i]->alloc);

        stat.print(numPrimitives);
      }

      if (device->verbosity(3))
      {
        alloc.print_blocks();
        for (size_t i=0; i<objects.size(); i++)
          if (objects[i]) 
            objects[i]->alloc.print_blocks();
      }

      std::cout << std::flush;
    }

    /* benchmark mode */
    if (device->benchmark)
    {
      if (!stat) stat.reset(new BVHNStatistics<N>(this));
      Lock<MutexSys> lock(g_printMutex);
      std::cout << "BENCHMARK_BUILD " << dt << " " << double(numPrimitives)/dt << " " << stat->sah() << " " << stat->bytesUsed() << " BVH" << N << "<" << primTy->name() << ">" << std::endl << std::flush;
    }
  }

#if defined(__AVX__)
  template class BVHN<8>;
#endif

#if !defined(__AVX__) || !defined(EMBREE_TARGET_SSE2) && !defined(EMBREE_TARGET_SSE42)
  template class BVHN<4>;
#endif
}

