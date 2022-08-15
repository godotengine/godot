//===- GenericDomTreeConstruction.h - Dominator Calculation ------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Generic dominator tree construction - This file provides routines to
/// construct immediate dominator information for a flow-graph based on the
/// algorithm described in this document:
///
///   A Fast Algorithm for Finding Dominators in a Flowgraph
///   T. Lengauer & R. Tarjan, ACM TOPLAS July 1979, pgs 121-141.
///
/// This implements the O(n*log(n)) versions of EVAL and LINK, because it turns
/// out that the theoretically slower O(n*log(n)) implementation is actually
/// faster than the almost-linear O(n*alpha(n)) version, even for large CFGs.
///
//===----------------------------------------------------------------------===//


#ifndef LLVM_SUPPORT_GENERICDOMTREECONSTRUCTION_H
#define LLVM_SUPPORT_GENERICDOMTREECONSTRUCTION_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/GenericDomTree.h"

namespace llvm {

template<class GraphT>
unsigned DFSPass(DominatorTreeBase<typename GraphT::NodeType>& DT,
                 typename GraphT::NodeType* V, unsigned N) {
  // This is more understandable as a recursive algorithm, but we can't use the
  // recursive algorithm due to stack depth issues.  Keep it here for
  // documentation purposes.
#if 0
  InfoRec &VInfo = DT.Info[DT.Roots[i]];
  VInfo.DFSNum = VInfo.Semi = ++N;
  VInfo.Label = V;

  Vertex.push_back(V);        // Vertex[n] = V;

  for (succ_iterator SI = succ_begin(V), E = succ_end(V); SI != E; ++SI) {
    InfoRec &SuccVInfo = DT.Info[*SI];
    if (SuccVInfo.Semi == 0) {
      SuccVInfo.Parent = V;
      N = DTDFSPass(DT, *SI, N);
    }
  }
#else
  bool IsChildOfArtificialExit = (N != 0);

  SmallVector<std::pair<typename GraphT::NodeType*,
                        typename GraphT::ChildIteratorType>, 32> Worklist;
  Worklist.push_back(std::make_pair(V, GraphT::child_begin(V)));
  while (!Worklist.empty()) {
    typename GraphT::NodeType* BB = Worklist.back().first;
    typename GraphT::ChildIteratorType NextSucc = Worklist.back().second;

    typename DominatorTreeBase<typename GraphT::NodeType>::InfoRec &BBInfo =
                                                                    DT.Info[BB];

    // First time we visited this BB?
    if (NextSucc == GraphT::child_begin(BB)) {
      BBInfo.DFSNum = BBInfo.Semi = ++N;
      BBInfo.Label = BB;

      DT.Vertex.push_back(BB);       // Vertex[n] = V;

      if (IsChildOfArtificialExit)
        BBInfo.Parent = 1;

      IsChildOfArtificialExit = false;
    }

    // store the DFS number of the current BB - the reference to BBInfo might
    // get invalidated when processing the successors.
    unsigned BBDFSNum = BBInfo.DFSNum;

    // If we are done with this block, remove it from the worklist.
    if (NextSucc == GraphT::child_end(BB)) {
      Worklist.pop_back();
      continue;
    }

    // Increment the successor number for the next time we get to it.
    ++Worklist.back().second;
    
    // Visit the successor next, if it isn't already visited.
    typename GraphT::NodeType* Succ = *NextSucc;

    typename DominatorTreeBase<typename GraphT::NodeType>::InfoRec &SuccVInfo =
                                                                  DT.Info[Succ];
    if (SuccVInfo.Semi == 0) {
      SuccVInfo.Parent = BBDFSNum;
      Worklist.push_back(std::make_pair(Succ, GraphT::child_begin(Succ)));
    }
  }
#endif
    return N;
}

template<class GraphT>
typename GraphT::NodeType* 
Eval(DominatorTreeBase<typename GraphT::NodeType>& DT,
     typename GraphT::NodeType *VIn, unsigned LastLinked) {
  typename DominatorTreeBase<typename GraphT::NodeType>::InfoRec &VInInfo =
                                                                  DT.Info[VIn];
  if (VInInfo.DFSNum < LastLinked)
    return VIn;

  SmallVector<typename GraphT::NodeType*, 32> Work;
  SmallPtrSet<typename GraphT::NodeType*, 32> Visited;

  if (VInInfo.Parent >= LastLinked)
    Work.push_back(VIn);
  
  while (!Work.empty()) {
    typename GraphT::NodeType* V = Work.back();
    typename DominatorTreeBase<typename GraphT::NodeType>::InfoRec &VInfo =
                                                                     DT.Info[V];
    typename GraphT::NodeType* VAncestor = DT.Vertex[VInfo.Parent];

    // Process Ancestor first
    if (Visited.insert(VAncestor).second && VInfo.Parent >= LastLinked) {
      Work.push_back(VAncestor);
      continue;
    } 
    Work.pop_back(); 

    // Update VInfo based on Ancestor info
    if (VInfo.Parent < LastLinked)
      continue;

    typename DominatorTreeBase<typename GraphT::NodeType>::InfoRec &VAInfo =
                                                             DT.Info[VAncestor];
    typename GraphT::NodeType* VAncestorLabel = VAInfo.Label;
    typename GraphT::NodeType* VLabel = VInfo.Label;
    if (DT.Info[VAncestorLabel].Semi < DT.Info[VLabel].Semi)
      VInfo.Label = VAncestorLabel;
    VInfo.Parent = VAInfo.Parent;
  }

  return VInInfo.Label;
}

template<class FuncT, class NodeT>
void Calculate(DominatorTreeBase<typename GraphTraits<NodeT>::NodeType>& DT,
               FuncT& F) {
  typedef GraphTraits<NodeT> GraphT;

  unsigned N = 0;
  bool MultipleRoots = (DT.Roots.size() > 1);
  if (MultipleRoots) {
    typename DominatorTreeBase<typename GraphT::NodeType>::InfoRec &BBInfo =
        DT.Info[nullptr];
    BBInfo.DFSNum = BBInfo.Semi = ++N;
    BBInfo.Label = nullptr;

    DT.Vertex.push_back(nullptr);       // Vertex[n] = V;
  }

  // Step #1: Number blocks in depth-first order and initialize variables used
  // in later stages of the algorithm.
  for (unsigned i = 0, e = static_cast<unsigned>(DT.Roots.size());
       i != e; ++i)
    N = DFSPass<GraphT>(DT, DT.Roots[i], N);

  // it might be that some blocks did not get a DFS number (e.g., blocks of 
  // infinite loops). In these cases an artificial exit node is required.
  MultipleRoots |= (DT.isPostDominator() && N != GraphTraits<FuncT*>::size(&F));

  // When naively implemented, the Lengauer-Tarjan algorithm requires a separate
  // bucket for each vertex. However, this is unnecessary, because each vertex
  // is only placed into a single bucket (that of its semidominator), and each
  // vertex's bucket is processed before it is added to any bucket itself.
  //
  // Instead of using a bucket per vertex, we use a single array Buckets that
  // has two purposes. Before the vertex V with preorder number i is processed,
  // Buckets[i] stores the index of the first element in V's bucket. After V's
  // bucket is processed, Buckets[i] stores the index of the next element in the
  // bucket containing V, if any.
  SmallVector<unsigned, 32> Buckets;
  Buckets.resize(N + 1);
  for (unsigned i = 1; i <= N; ++i)
    Buckets[i] = i;

  for (unsigned i = N; i >= 2; --i) {
    typename GraphT::NodeType* W = DT.Vertex[i];
    typename DominatorTreeBase<typename GraphT::NodeType>::InfoRec &WInfo =
                                                                     DT.Info[W];

    // Step #2: Implicitly define the immediate dominator of vertices
    for (unsigned j = i; Buckets[j] != i; j = Buckets[j]) {
      typename GraphT::NodeType* V = DT.Vertex[Buckets[j]];
      typename GraphT::NodeType* U = Eval<GraphT>(DT, V, i + 1);
      DT.IDoms[V] = DT.Info[U].Semi < i ? U : W;
    }

    // Step #3: Calculate the semidominators of all vertices

    // initialize the semi dominator to point to the parent node
    WInfo.Semi = WInfo.Parent;
    typedef GraphTraits<Inverse<NodeT> > InvTraits;
    for (typename InvTraits::ChildIteratorType CI =
         InvTraits::child_begin(W),
         E = InvTraits::child_end(W); CI != E; ++CI) {
      typename InvTraits::NodeType *N = *CI;
      if (DT.Info.count(N)) {  // Only if this predecessor is reachable!
        unsigned SemiU = DT.Info[Eval<GraphT>(DT, N, i + 1)].Semi;
        if (SemiU < WInfo.Semi)
          WInfo.Semi = SemiU;
      }
    }

    // If V is a non-root vertex and sdom(V) = parent(V), then idom(V) is
    // necessarily parent(V). In this case, set idom(V) here and avoid placing
    // V into a bucket.
    if (WInfo.Semi == WInfo.Parent) {
      DT.IDoms[W] = DT.Vertex[WInfo.Parent];
    } else {
      Buckets[i] = Buckets[WInfo.Semi];
      Buckets[WInfo.Semi] = i;
    }
  }

  if (N >= 1) {
    typename GraphT::NodeType* Root = DT.Vertex[1];
    for (unsigned j = 1; Buckets[j] != 1; j = Buckets[j]) {
      typename GraphT::NodeType* V = DT.Vertex[Buckets[j]];
      DT.IDoms[V] = Root;
    }
  }

  // Step #4: Explicitly define the immediate dominator of each vertex
  for (unsigned i = 2; i <= N; ++i) {
    typename GraphT::NodeType* W = DT.Vertex[i];
    typename GraphT::NodeType*& WIDom = DT.IDoms[W];
    if (WIDom != DT.Vertex[DT.Info[W].Semi])
      WIDom = DT.IDoms[WIDom];
  }

  if (DT.Roots.empty()) return;

  // Add a node for the root.  This node might be the actual root, if there is
  // one exit block, or it may be the virtual exit (denoted by (BasicBlock *)0)
  // which postdominates all real exits if there are multiple exit blocks, or
  // an infinite loop.
  typename GraphT::NodeType* Root = !MultipleRoots ? DT.Roots[0] : nullptr;

  DT.RootNode =
      (DT.DomTreeNodes[Root] =
           llvm::make_unique<DomTreeNodeBase<typename GraphT::NodeType>>(
               Root, nullptr)).get();

  // Loop over all of the reachable blocks in the function...
  for (unsigned i = 2; i <= N; ++i) {
    typename GraphT::NodeType* W = DT.Vertex[i];

    // Don't replace this with 'count', the insertion side effect is important
    if (DT.DomTreeNodes[W])
      continue; // Haven't calculated this node yet?

    typename GraphT::NodeType* ImmDom = DT.getIDom(W);

    assert(ImmDom || DT.DomTreeNodes[nullptr]);

    // Get or calculate the node for the immediate dominator
    DomTreeNodeBase<typename GraphT::NodeType> *IDomNode =
                                                     DT.getNodeForBlock(ImmDom);

    // Add a new tree node for this BasicBlock, and link it as a child of
    // IDomNode
    DT.DomTreeNodes[W] = IDomNode->addChild(
        llvm::make_unique<DomTreeNodeBase<typename GraphT::NodeType>>(
            W, IDomNode));
  }

  // Free temporary memory used to construct idom's
  DT.IDoms.clear();
  DT.Info.clear();
  DT.Vertex.clear();
  DT.Vertex.shrink_to_fit();

  DT.updateDFSNumbers();
}

}

#endif
