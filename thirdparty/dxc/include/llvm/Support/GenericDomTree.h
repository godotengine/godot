//===- GenericDomTree.h - Generic dominator trees for graphs ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines a set of templates that efficiently compute a dominator
/// tree over a generic graph. This is used typically in LLVM for fast
/// dominance queries on the CFG, but is fully generic w.r.t. the underlying
/// graph types.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_GENERICDOMTREE_H
#define LLVM_SUPPORT_GENERICDOMTREE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace llvm {

/// \brief Base class that other, more interesting dominator analyses
/// inherit from.
template <class NodeT> class DominatorBase {
protected:
  std::vector<NodeT *> Roots;
  bool IsPostDominators;
  explicit DominatorBase(bool isPostDom)
      : Roots(), IsPostDominators(isPostDom) {}
  DominatorBase(DominatorBase &&Arg)
      : Roots(std::move(Arg.Roots)),
        IsPostDominators(std::move(Arg.IsPostDominators)) {
    Arg.Roots.clear();
  }
  DominatorBase &operator=(DominatorBase &&RHS) {
    Roots = std::move(RHS.Roots);
    IsPostDominators = std::move(RHS.IsPostDominators);
    RHS.Roots.clear();
    return *this;
  }

public:
  /// getRoots - Return the root blocks of the current CFG.  This may include
  /// multiple blocks if we are computing post dominators.  For forward
  /// dominators, this will always be a single block (the entry node).
  ///
  const std::vector<NodeT *> &getRoots() const { return Roots; }

  /// isPostDominator - Returns true if analysis based of postdoms
  ///
  bool isPostDominator() const { return IsPostDominators; }
};

template <class NodeT> class DominatorTreeBase;
struct PostDominatorTree;

/// \brief Base class for the actual dominator tree node.
template <class NodeT> class DomTreeNodeBase {
  NodeT *TheBB;
  DomTreeNodeBase<NodeT> *IDom;
  std::vector<DomTreeNodeBase<NodeT> *> Children;
  mutable int DFSNumIn, DFSNumOut;

  template <class N> friend class DominatorTreeBase;
  friend struct PostDominatorTree;

public:
  typedef typename std::vector<DomTreeNodeBase<NodeT> *>::iterator iterator;
  typedef typename std::vector<DomTreeNodeBase<NodeT> *>::const_iterator
      const_iterator;

  iterator begin() { return Children.begin(); }
  iterator end() { return Children.end(); }
  const_iterator begin() const { return Children.begin(); }
  const_iterator end() const { return Children.end(); }

  NodeT *getBlock() const { return TheBB; }
  DomTreeNodeBase<NodeT> *getIDom() const { return IDom; }
  const std::vector<DomTreeNodeBase<NodeT> *> &getChildren() const {
    return Children;
  }

  DomTreeNodeBase(NodeT *BB, DomTreeNodeBase<NodeT> *iDom)
      : TheBB(BB), IDom(iDom), DFSNumIn(-1), DFSNumOut(-1) {}

  std::unique_ptr<DomTreeNodeBase<NodeT>>
  addChild(std::unique_ptr<DomTreeNodeBase<NodeT>> C) {
    Children.push_back(C.get());
    return C;
  }

  size_t getNumChildren() const { return Children.size(); }

  void clearAllChildren() { Children.clear(); }

  bool compare(const DomTreeNodeBase<NodeT> *Other) const {
    if (getNumChildren() != Other->getNumChildren())
      return true;

    SmallPtrSet<const NodeT *, 4> OtherChildren;
    for (const_iterator I = Other->begin(), E = Other->end(); I != E; ++I) {
      const NodeT *Nd = (*I)->getBlock();
      OtherChildren.insert(Nd);
    }

    for (const_iterator I = begin(), E = end(); I != E; ++I) {
      const NodeT *N = (*I)->getBlock();
      if (OtherChildren.count(N) == 0)
        return true;
    }
    return false;
  }

  void setIDom(DomTreeNodeBase<NodeT> *NewIDom) {
    assert(IDom && "No immediate dominator?");
    if (IDom != NewIDom) {
      typename std::vector<DomTreeNodeBase<NodeT> *>::iterator I =
          std::find(IDom->Children.begin(), IDom->Children.end(), this);
      assert(I != IDom->Children.end() &&
             "Not in immediate dominator children set!");
      // I am no longer your child...
      IDom->Children.erase(I);

      // Switch to new dominator
      IDom = NewIDom;
      IDom->Children.push_back(this);
    }
  }

  /// getDFSNumIn/getDFSNumOut - These are an internal implementation detail, do
  /// not call them.
  unsigned getDFSNumIn() const { return DFSNumIn; }
  unsigned getDFSNumOut() const { return DFSNumOut; }

private:
  // Return true if this node is dominated by other. Use this only if DFS info
  // is valid.
  bool DominatedBy(const DomTreeNodeBase<NodeT> *other) const {
    return this->DFSNumIn >= other->DFSNumIn &&
           this->DFSNumOut <= other->DFSNumOut;
  }
};

template <class NodeT>
raw_ostream &operator<<(raw_ostream &o, const DomTreeNodeBase<NodeT> *Node) {
  if (Node->getBlock())
    Node->getBlock()->printAsOperand(o, false);
  else
    o << " <<exit node>>";

  o << " {" << Node->getDFSNumIn() << "," << Node->getDFSNumOut() << "}";

  return o << "\n";
}

template <class NodeT>
void PrintDomTree(const DomTreeNodeBase<NodeT> *N, raw_ostream &o,
                  unsigned Lev) {
  o.indent(2 * Lev) << "[" << Lev << "] " << N;
  for (typename DomTreeNodeBase<NodeT>::const_iterator I = N->begin(),
                                                       E = N->end();
       I != E; ++I)
    PrintDomTree<NodeT>(*I, o, Lev + 1);
}

// The calculate routine is provided in a separate header but referenced here.
template <class FuncT, class N>
void Calculate(DominatorTreeBase<typename GraphTraits<N>::NodeType> &DT,
               FuncT &F);

/// \brief Core dominator tree base class.
///
/// This class is a generic template over graph nodes. It is instantiated for
/// various graphs in the LLVM IR or in the code generator.
template <class NodeT> class DominatorTreeBase : public DominatorBase<NodeT> {
  DominatorTreeBase(const DominatorTreeBase &) = delete;
  DominatorTreeBase &operator=(const DominatorTreeBase &) = delete;

  bool dominatedBySlowTreeWalk(const DomTreeNodeBase<NodeT> *A,
                               const DomTreeNodeBase<NodeT> *B) const {
    assert(A != B);
    assert(isReachableFromEntry(B));
    assert(isReachableFromEntry(A));

    const DomTreeNodeBase<NodeT> *IDom;
    while ((IDom = B->getIDom()) != nullptr && IDom != A && IDom != B)
      B = IDom; // Walk up the tree
    return IDom != nullptr;
  }

  /// \brief Wipe this tree's state without releasing any resources.
  ///
  /// This is essentially a post-move helper only. It leaves the object in an
  /// assignable and destroyable state, but otherwise invalid.
  void wipe() {
    DomTreeNodes.clear();
    IDoms.clear();
    Vertex.clear();
    Info.clear();
    RootNode = nullptr;
  }

protected:
  typedef DenseMap<NodeT *, std::unique_ptr<DomTreeNodeBase<NodeT>>>
      DomTreeNodeMapType;
  DomTreeNodeMapType DomTreeNodes;
  DomTreeNodeBase<NodeT> *RootNode;

  mutable bool DFSInfoValid;
  mutable unsigned int SlowQueries;
  // Information record used during immediate dominators computation.
  struct InfoRec {
    unsigned DFSNum;
    unsigned Parent;
    unsigned Semi;
    NodeT *Label;

    InfoRec() : DFSNum(0), Parent(0), Semi(0), Label(nullptr) {}
  };

  DenseMap<NodeT *, NodeT *> IDoms;

  // Vertex - Map the DFS number to the NodeT*
  std::vector<NodeT *> Vertex;

  // Info - Collection of information used during the computation of idoms.
  DenseMap<NodeT *, InfoRec> Info;

  void reset() {
    DomTreeNodes.clear();
    IDoms.clear();
    this->Roots.clear();
    Vertex.clear();
    RootNode = nullptr;
    DFSInfoValid = false;
    SlowQueries = 0;
  }

  // NewBB is split and now it has one successor. Update dominator tree to
  // reflect this change.
  template <class N, class GraphT>
  void Split(DominatorTreeBase<typename GraphT::NodeType> &DT,
             typename GraphT::NodeType *NewBB) {
    assert(std::distance(GraphT::child_begin(NewBB),
                         GraphT::child_end(NewBB)) == 1 &&
           "NewBB should have a single successor!");
    typename GraphT::NodeType *NewBBSucc = *GraphT::child_begin(NewBB);

    std::vector<typename GraphT::NodeType *> PredBlocks;
    typedef GraphTraits<Inverse<N>> InvTraits;
    for (typename InvTraits::ChildIteratorType
             PI = InvTraits::child_begin(NewBB),
             PE = InvTraits::child_end(NewBB);
         PI != PE; ++PI)
      PredBlocks.push_back(*PI);

    assert(!PredBlocks.empty() && "No predblocks?");

    bool NewBBDominatesNewBBSucc = true;
    for (typename InvTraits::ChildIteratorType
             PI = InvTraits::child_begin(NewBBSucc),
             E = InvTraits::child_end(NewBBSucc);
         PI != E; ++PI) {
      typename InvTraits::NodeType *ND = *PI;
      if (ND != NewBB && !DT.dominates(NewBBSucc, ND) &&
          DT.isReachableFromEntry(ND)) {
        NewBBDominatesNewBBSucc = false;
        break;
      }
    }

    // Find NewBB's immediate dominator and create new dominator tree node for
    // NewBB.
    NodeT *NewBBIDom = nullptr;
    unsigned i = 0;
    for (i = 0; i < PredBlocks.size(); ++i)
      if (DT.isReachableFromEntry(PredBlocks[i])) {
        NewBBIDom = PredBlocks[i];
        break;
      }

    // It's possible that none of the predecessors of NewBB are reachable;
    // in that case, NewBB itself is unreachable, so nothing needs to be
    // changed.
    if (!NewBBIDom)
      return;

    for (i = i + 1; i < PredBlocks.size(); ++i) {
      if (DT.isReachableFromEntry(PredBlocks[i]))
        NewBBIDom = DT.findNearestCommonDominator(NewBBIDom, PredBlocks[i]);
    }

    // Create the new dominator tree node... and set the idom of NewBB.
    DomTreeNodeBase<NodeT> *NewBBNode = DT.addNewBlock(NewBB, NewBBIDom);

    // If NewBB strictly dominates other blocks, then it is now the immediate
    // dominator of NewBBSucc.  Update the dominator tree as appropriate.
    if (NewBBDominatesNewBBSucc) {
      DomTreeNodeBase<NodeT> *NewBBSuccNode = DT.getNode(NewBBSucc);
      DT.changeImmediateDominator(NewBBSuccNode, NewBBNode);
    }
  }

public:
  explicit DominatorTreeBase(bool isPostDom)
      : DominatorBase<NodeT>(isPostDom), DFSInfoValid(false), SlowQueries(0) {}

  DominatorTreeBase(DominatorTreeBase &&Arg)
      : DominatorBase<NodeT>(
            std::move(static_cast<DominatorBase<NodeT> &>(Arg))),
        DomTreeNodes(std::move(Arg.DomTreeNodes)),
        RootNode(std::move(Arg.RootNode)),
        DFSInfoValid(std::move(Arg.DFSInfoValid)),
        SlowQueries(std::move(Arg.SlowQueries)), IDoms(std::move(Arg.IDoms)),
        Vertex(std::move(Arg.Vertex)), Info(std::move(Arg.Info)) {
    Arg.wipe();
  }
  DominatorTreeBase &operator=(DominatorTreeBase &&RHS) {
    DominatorBase<NodeT>::operator=(
        std::move(static_cast<DominatorBase<NodeT> &>(RHS)));
    DomTreeNodes = std::move(RHS.DomTreeNodes);
    RootNode = std::move(RHS.RootNode);
    DFSInfoValid = std::move(RHS.DFSInfoValid);
    SlowQueries = std::move(RHS.SlowQueries);
    IDoms = std::move(RHS.IDoms);
    Vertex = std::move(RHS.Vertex);
    Info = std::move(RHS.Info);
    RHS.wipe();
    return *this;
  }

  /// compare - Return false if the other dominator tree base matches this
  /// dominator tree base. Otherwise return true.
  bool compare(const DominatorTreeBase &Other) const {

    const DomTreeNodeMapType &OtherDomTreeNodes = Other.DomTreeNodes;
    if (DomTreeNodes.size() != OtherDomTreeNodes.size())
      return true;

    for (typename DomTreeNodeMapType::const_iterator
             I = this->DomTreeNodes.begin(),
             E = this->DomTreeNodes.end();
         I != E; ++I) {
      NodeT *BB = I->first;
      typename DomTreeNodeMapType::const_iterator OI =
          OtherDomTreeNodes.find(BB);
      if (OI == OtherDomTreeNodes.end())
        return true;

      DomTreeNodeBase<NodeT> &MyNd = *I->second;
      DomTreeNodeBase<NodeT> &OtherNd = *OI->second;

      if (MyNd.compare(&OtherNd))
        return true;
    }

    return false;
  }

  void releaseMemory() { reset(); }

  /// getNode - return the (Post)DominatorTree node for the specified basic
  /// block.  This is the same as using operator[] on this class.
  ///
  DomTreeNodeBase<NodeT> *getNode(NodeT *BB) const {
    auto I = DomTreeNodes.find(BB);
    if (I != DomTreeNodes.end())
      return I->second.get();
    return nullptr;
  }

  DomTreeNodeBase<NodeT> *operator[](NodeT *BB) const { return getNode(BB); }

  /// getRootNode - This returns the entry node for the CFG of the function.  If
  /// this tree represents the post-dominance relations for a function, however,
  /// this root may be a node with the block == NULL.  This is the case when
  /// there are multiple exit nodes from a particular function.  Consumers of
  /// post-dominance information must be capable of dealing with this
  /// possibility.
  ///
  DomTreeNodeBase<NodeT> *getRootNode() { return RootNode; }
  const DomTreeNodeBase<NodeT> *getRootNode() const { return RootNode; }

  /// Get all nodes dominated by R, including R itself.
  void getDescendants(NodeT *R, SmallVectorImpl<NodeT *> &Result) const {
    Result.clear();
    const DomTreeNodeBase<NodeT> *RN = getNode(R);
    if (!RN)
      return; // If R is unreachable, it will not be present in the DOM tree.
    SmallVector<const DomTreeNodeBase<NodeT> *, 8> WL;
    WL.push_back(RN);

    while (!WL.empty()) {
      const DomTreeNodeBase<NodeT> *N = WL.pop_back_val();
      Result.push_back(N->getBlock());
      WL.append(N->begin(), N->end());
    }
  }

  /// properlyDominates - Returns true iff A dominates B and A != B.
  /// Note that this is not a constant time operation!
  ///
  bool properlyDominates(const DomTreeNodeBase<NodeT> *A,
                         const DomTreeNodeBase<NodeT> *B) const {
    if (!A || !B)
      return false;
    if (A == B)
      return false;
    return dominates(A, B);
  }

  bool properlyDominates(const NodeT *A, const NodeT *B) const;

  /// isReachableFromEntry - Return true if A is dominated by the entry
  /// block of the function containing it.
  bool isReachableFromEntry(const NodeT *A) const {
    assert(!this->isPostDominator() &&
           "This is not implemented for post dominators");
    return isReachableFromEntry(getNode(const_cast<NodeT *>(A)));
  }

  bool isReachableFromEntry(const DomTreeNodeBase<NodeT> *A) const { return A; }

  /// dominates - Returns true iff A dominates B.  Note that this is not a
  /// constant time operation!
  ///
  bool dominates(const DomTreeNodeBase<NodeT> *A,
                 const DomTreeNodeBase<NodeT> *B) const {
    // A node trivially dominates itself.
    if (B == A)
      return true;

    // An unreachable node is dominated by anything.
    if (!isReachableFromEntry(B))
      return true;

    // And dominates nothing.
    if (!isReachableFromEntry(A))
      return false;

    // Compare the result of the tree walk and the dfs numbers, if expensive
    // checks are enabled.
#ifdef XDEBUG
    assert((!DFSInfoValid ||
            (dominatedBySlowTreeWalk(A, B) == B->DominatedBy(A))) &&
           "Tree walk disagrees with dfs numbers!");
#endif

    if (DFSInfoValid)
      return B->DominatedBy(A);

    // If we end up with too many slow queries, just update the
    // DFS numbers on the theory that we are going to keep querying.
    SlowQueries++;
    if (SlowQueries > 32) {
      updateDFSNumbers();
      return B->DominatedBy(A);
    }

    return dominatedBySlowTreeWalk(A, B);
  }

  bool dominates(const NodeT *A, const NodeT *B) const;

  NodeT *getRoot() const {
    assert(this->Roots.size() == 1 && "Should always have entry node!");
    return this->Roots[0];
  }

  /// findNearestCommonDominator - Find nearest common dominator basic block
  /// for basic block A and B. If there is no such block then return NULL.
  NodeT *findNearestCommonDominator(NodeT *A, NodeT *B) {
    assert(A->getParent() == B->getParent() &&
           "Two blocks are not in same function");

    // If either A or B is a entry block then it is nearest common dominator
    // (for forward-dominators).
    if (!this->isPostDominator()) {
      NodeT &Entry = A->getParent()->front();
      if (A == &Entry || B == &Entry)
        return &Entry;
    }

    // If B dominates A then B is nearest common dominator.
    if (dominates(B, A))
      return B;

    // If A dominates B then A is nearest common dominator.
    if (dominates(A, B))
      return A;

    DomTreeNodeBase<NodeT> *NodeA = getNode(A);
    DomTreeNodeBase<NodeT> *NodeB = getNode(B);

    // If we have DFS info, then we can avoid all allocations by just querying
    // it from each IDom. Note that because we call 'dominates' twice above, we
    // expect to call through this code at most 16 times in a row without
    // building valid DFS information. This is important as below is a *very*
    // slow tree walk.
    if (DFSInfoValid) {
      DomTreeNodeBase<NodeT> *IDomA = NodeA->getIDom();
      while (IDomA) {
        if (NodeB->DominatedBy(IDomA))
          return IDomA->getBlock();
        IDomA = IDomA->getIDom();
      }
      return nullptr;
    }

    // Collect NodeA dominators set.
    SmallPtrSet<DomTreeNodeBase<NodeT> *, 16> NodeADoms;
    NodeADoms.insert(NodeA);
    DomTreeNodeBase<NodeT> *IDomA = NodeA->getIDom();
    while (IDomA) {
      NodeADoms.insert(IDomA);
      IDomA = IDomA->getIDom();
    }

    // Walk NodeB immediate dominators chain and find common dominator node.
    DomTreeNodeBase<NodeT> *IDomB = NodeB->getIDom();
    while (IDomB) {
      if (NodeADoms.count(IDomB) != 0)
        return IDomB->getBlock();

      IDomB = IDomB->getIDom();
    }

    return nullptr;
  }

  const NodeT *findNearestCommonDominator(const NodeT *A, const NodeT *B) {
    // Cast away the const qualifiers here. This is ok since
    // const is re-introduced on the return type.
    return findNearestCommonDominator(const_cast<NodeT *>(A),
                                      const_cast<NodeT *>(B));
  }

  //===--------------------------------------------------------------------===//
  // API to update (Post)DominatorTree information based on modifications to
  // the CFG...

  /// addNewBlock - Add a new node to the dominator tree information.  This
  /// creates a new node as a child of DomBB dominator node,linking it into
  /// the children list of the immediate dominator.
  DomTreeNodeBase<NodeT> *addNewBlock(NodeT *BB, NodeT *DomBB) {
    assert(getNode(BB) == nullptr && "Block already in dominator tree!");
    DomTreeNodeBase<NodeT> *IDomNode = getNode(DomBB);
    assert(IDomNode && "Not immediate dominator specified for block!");
    DFSInfoValid = false;
    return (DomTreeNodes[BB] = IDomNode->addChild(
                llvm::make_unique<DomTreeNodeBase<NodeT>>(BB, IDomNode))).get();
  }

  /// changeImmediateDominator - This method is used to update the dominator
  /// tree information when a node's immediate dominator changes.
  ///
  void changeImmediateDominator(DomTreeNodeBase<NodeT> *N,
                                DomTreeNodeBase<NodeT> *NewIDom) {
    assert(N && NewIDom && "Cannot change null node pointers!");
    DFSInfoValid = false;
    N->setIDom(NewIDom);
  }

  void changeImmediateDominator(NodeT *BB, NodeT *NewBB) {
    changeImmediateDominator(getNode(BB), getNode(NewBB));
  }

  /// eraseNode - Removes a node from the dominator tree. Block must not
  /// dominate any other blocks. Removes node from its immediate dominator's
  /// children list. Deletes dominator node associated with basic block BB.
  void eraseNode(NodeT *BB) {
    DomTreeNodeBase<NodeT> *Node = getNode(BB);
    assert(Node && "Removing node that isn't in dominator tree.");
    assert(Node->getChildren().empty() && "Node is not a leaf node.");

    // Remove node from immediate dominator's children list.
    DomTreeNodeBase<NodeT> *IDom = Node->getIDom();
    if (IDom) {
      typename std::vector<DomTreeNodeBase<NodeT> *>::iterator I =
          std::find(IDom->Children.begin(), IDom->Children.end(), Node);
      assert(I != IDom->Children.end() &&
             "Not in immediate dominator children set!");
      // I am no longer your child...
      IDom->Children.erase(I);
    }

    DomTreeNodes.erase(BB);
  }

  /// splitBlock - BB is split and now it has one successor. Update dominator
  /// tree to reflect this change.
  void splitBlock(NodeT *NewBB) {
    if (this->IsPostDominators)
      this->Split<Inverse<NodeT *>, GraphTraits<Inverse<NodeT *>>>(*this,
                                                                   NewBB);
    else
      this->Split<NodeT *, GraphTraits<NodeT *>>(*this, NewBB);
  }

  /// print - Convert to human readable form
  ///
  void print(raw_ostream &o) const {
    o << "=============================--------------------------------\n";
    if (this->isPostDominator())
      o << "Inorder PostDominator Tree: ";
    else
      o << "Inorder Dominator Tree: ";
    if (!this->DFSInfoValid)
      o << "DFSNumbers invalid: " << SlowQueries << " slow queries.";
    o << "\n";

    // The postdom tree can have a null root if there are no returns.
    if (getRootNode())
      PrintDomTree<NodeT>(getRootNode(), o, 1);
  }

protected:
  template <class GraphT>
  friend typename GraphT::NodeType *
  Eval(DominatorTreeBase<typename GraphT::NodeType> &DT,
       typename GraphT::NodeType *V, unsigned LastLinked);

  template <class GraphT>
  friend unsigned DFSPass(DominatorTreeBase<typename GraphT::NodeType> &DT,
                          typename GraphT::NodeType *V, unsigned N);

  template <class FuncT, class N>
  friend void
  Calculate(DominatorTreeBase<typename GraphTraits<N>::NodeType> &DT, FuncT &F);


  DomTreeNodeBase<NodeT> *getNodeForBlock(NodeT *BB) {
    if (DomTreeNodeBase<NodeT> *Node = getNode(BB))
      return Node;

    // Haven't calculated this node yet?  Get or calculate the node for the
    // immediate dominator.
    NodeT *IDom = getIDom(BB);

    assert(IDom || this->DomTreeNodes[nullptr]);
    DomTreeNodeBase<NodeT> *IDomNode = getNodeForBlock(IDom);

    // Add a new tree node for this NodeT, and link it as a child of
    // IDomNode
    return (this->DomTreeNodes[BB] = IDomNode->addChild(
                llvm::make_unique<DomTreeNodeBase<NodeT>>(BB, IDomNode))).get();
  }

  NodeT *getIDom(NodeT *BB) const { return IDoms.lookup(BB); }

  void addRoot(NodeT *BB) { this->Roots.push_back(BB); }

public:
  /// updateDFSNumbers - Assign In and Out numbers to the nodes while walking
  /// dominator tree in dfs order.
  void updateDFSNumbers() const {

    if (DFSInfoValid) {
      SlowQueries = 0;
      return;
    }

    unsigned DFSNum = 0;

    SmallVector<std::pair<const DomTreeNodeBase<NodeT> *,
                          typename DomTreeNodeBase<NodeT>::const_iterator>,
                32> WorkStack;

    const DomTreeNodeBase<NodeT> *ThisRoot = getRootNode();

    if (!ThisRoot)
      return;

    // Even in the case of multiple exits that form the post dominator root
    // nodes, do not iterate over all exits, but start from the virtual root
    // node. Otherwise bbs, that are not post dominated by any exit but by the
    // virtual root node, will never be assigned a DFS number.
    WorkStack.push_back(std::make_pair(ThisRoot, ThisRoot->begin()));
    ThisRoot->DFSNumIn = DFSNum++;

    while (!WorkStack.empty()) {
      const DomTreeNodeBase<NodeT> *Node = WorkStack.back().first;
      typename DomTreeNodeBase<NodeT>::const_iterator ChildIt =
          WorkStack.back().second;

      // If we visited all of the children of this node, "recurse" back up the
      // stack setting the DFOutNum.
      if (ChildIt == Node->end()) {
        Node->DFSNumOut = DFSNum++;
        WorkStack.pop_back();
      } else {
        // Otherwise, recursively visit this child.
        const DomTreeNodeBase<NodeT> *Child = *ChildIt;
        ++WorkStack.back().second;

        WorkStack.push_back(std::make_pair(Child, Child->begin()));
        Child->DFSNumIn = DFSNum++;
      }
    }

    SlowQueries = 0;
    DFSInfoValid = true;
  }

  /// recalculate - compute a dominator tree for the given function
  template <class FT> void recalculate(FT &F) {
    typedef GraphTraits<FT *> TraitsTy;
    reset();
    this->Vertex.push_back(nullptr);

    if (!this->IsPostDominators) {
      // Initialize root
      NodeT *entry = TraitsTy::getEntryNode(&F);
      this->Roots.push_back(entry);
      this->IDoms[entry] = nullptr;
      this->DomTreeNodes[entry] = nullptr;

      Calculate<FT, NodeT *>(*this, F);
    } else {
      // Initialize the roots list
      for (typename TraitsTy::nodes_iterator I = TraitsTy::nodes_begin(&F),
                                             E = TraitsTy::nodes_end(&F);
           I != E; ++I) {
        if (TraitsTy::child_begin(I) == TraitsTy::child_end(I))
          addRoot(I);

        // Prepopulate maps so that we don't get iterator invalidation issues
        // later.
        this->IDoms[I] = nullptr;
        this->DomTreeNodes[I] = nullptr;
      }

      Calculate<FT, Inverse<NodeT *>>(*this, F);
    }
  }
};

// These two functions are declared out of line as a workaround for building
// with old (< r147295) versions of clang because of pr11642.
template <class NodeT>
bool DominatorTreeBase<NodeT>::dominates(const NodeT *A, const NodeT *B) const {
  if (A == B)
    return true;

  // Cast away the const qualifiers here. This is ok since
  // this function doesn't actually return the values returned
  // from getNode.
  return dominates(getNode(const_cast<NodeT *>(A)),
                   getNode(const_cast<NodeT *>(B)));
}
template <class NodeT>
bool DominatorTreeBase<NodeT>::properlyDominates(const NodeT *A,
                                                 const NodeT *B) const {
  if (A == B)
    return false;

  // Cast away the const qualifiers here. This is ok since
  // this function doesn't actually return the values returned
  // from getNode.
  return dominates(getNode(const_cast<NodeT *>(A)),
                   getNode(const_cast<NodeT *>(B)));
}

}

#endif
