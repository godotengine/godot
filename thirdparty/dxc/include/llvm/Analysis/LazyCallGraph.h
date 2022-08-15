//===- LazyCallGraph.h - Analysis of a Module's call graph ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Implements a lazy call graph analysis and related passes for the new pass
/// manager.
///
/// NB: This is *not* a traditional call graph! It is a graph which models both
/// the current calls and potential calls. As a consequence there are many
/// edges in this call graph that do not correspond to a 'call' or 'invoke'
/// instruction.
///
/// The primary use cases of this graph analysis is to facilitate iterating
/// across the functions of a module in ways that ensure all callees are
/// visited prior to a caller (given any SCC constraints), or vice versa. As
/// such is it particularly well suited to organizing CGSCC optimizations such
/// as inlining, outlining, argument promotion, etc. That is its primary use
/// case and motivates the design. It may not be appropriate for other
/// purposes. The use graph of functions or some other conservative analysis of
/// call instructions may be interesting for optimizations and subsequent
/// analyses which don't work in the context of an overly specified
/// potential-call-edge graph.
///
/// To understand the specific rules and nature of this call graph analysis,
/// see the documentation of the \c LazyCallGraph below.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LAZYCALLGRAPH_H
#define LLVM_ANALYSIS_LAZYCALLGRAPH_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Allocator.h"
#include <iterator>

namespace llvm {
class PreservedAnalyses;
class raw_ostream;

/// \brief A lazily constructed view of the call graph of a module.
///
/// With the edges of this graph, the motivating constraint that we are
/// attempting to maintain is that function-local optimization, CGSCC-local
/// optimizations, and optimizations transforming a pair of functions connected
/// by an edge in the graph, do not invalidate a bottom-up traversal of the SCC
/// DAG. That is, no optimizations will delete, remove, or add an edge such
/// that functions already visited in a bottom-up order of the SCC DAG are no
/// longer valid to have visited, or such that functions not yet visited in
/// a bottom-up order of the SCC DAG are not required to have already been
/// visited.
///
/// Within this constraint, the desire is to minimize the merge points of the
/// SCC DAG. The greater the fanout of the SCC DAG and the fewer merge points
/// in the SCC DAG, the more independence there is in optimizing within it.
/// There is a strong desire to enable parallelization of optimizations over
/// the call graph, and both limited fanout and merge points will (artificially
/// in some cases) limit the scaling of such an effort.
///
/// To this end, graph represents both direct and any potential resolution to
/// an indirect call edge. Another way to think about it is that it represents
/// both the direct call edges and any direct call edges that might be formed
/// through static optimizations. Specifically, it considers taking the address
/// of a function to be an edge in the call graph because this might be
/// forwarded to become a direct call by some subsequent function-local
/// optimization. The result is that the graph closely follows the use-def
/// edges for functions. Walking "up" the graph can be done by looking at all
/// of the uses of a function.
///
/// The roots of the call graph are the external functions and functions
/// escaped into global variables. Those functions can be called from outside
/// of the module or via unknowable means in the IR -- we may not be able to
/// form even a potential call edge from a function body which may dynamically
/// load the function and call it.
///
/// This analysis still requires updates to remain valid after optimizations
/// which could potentially change the set of potential callees. The
/// constraints it operates under only make the traversal order remain valid.
///
/// The entire analysis must be re-computed if full interprocedural
/// optimizations run at any point. For example, globalopt completely
/// invalidates the information in this analysis.
///
/// FIXME: This class is named LazyCallGraph in a lame attempt to distinguish
/// it from the existing CallGraph. At some point, it is expected that this
/// will be the only call graph and it will be renamed accordingly.
class LazyCallGraph {
public:
  class Node;
  class SCC;
  typedef SmallVector<PointerUnion<Function *, Node *>, 4> NodeVectorT;
  typedef SmallVectorImpl<PointerUnion<Function *, Node *>> NodeVectorImplT;

  /// \brief A lazy iterator used for both the entry nodes and child nodes.
  ///
  /// When this iterator is dereferenced, if not yet available, a function will
  /// be scanned for "calls" or uses of functions and its child information
  /// will be constructed. All of these results are accumulated and cached in
  /// the graph.
  class iterator
      : public iterator_adaptor_base<iterator, NodeVectorImplT::iterator,
                                     std::forward_iterator_tag, Node> {
    friend class LazyCallGraph;
    friend class LazyCallGraph::Node;

    LazyCallGraph *G;
    NodeVectorImplT::iterator E;

    // Build the iterator for a specific position in a node list.
    iterator(LazyCallGraph &G, NodeVectorImplT::iterator NI,
             NodeVectorImplT::iterator E)
        : iterator_adaptor_base(NI), G(&G), E(E) {
      while (I != E && I->isNull())
        ++I;
    }

  public:
    iterator() {}

    using iterator_adaptor_base::operator++;
    iterator &operator++() {
      do {
        ++I;
      } while (I != E && I->isNull());
      return *this;
    }

    reference operator*() const {
      if (I->is<Node *>())
        return *I->get<Node *>();

      Function *F = I->get<Function *>();
      Node &ChildN = G->get(*F);
      *I = &ChildN;
      return ChildN;
    }
  };

  /// \brief A node in the call graph.
  ///
  /// This represents a single node. It's primary roles are to cache the list of
  /// callees, de-duplicate and provide fast testing of whether a function is
  /// a callee, and facilitate iteration of child nodes in the graph.
  class Node {
    friend class LazyCallGraph;
    friend class LazyCallGraph::SCC;

    LazyCallGraph *G;
    Function &F;

    // We provide for the DFS numbering and Tarjan walk lowlink numbers to be
    // stored directly within the node.
    int DFSNumber;
    int LowLink;

    mutable NodeVectorT Callees;
    DenseMap<Function *, size_t> CalleeIndexMap;

    /// \brief Basic constructor implements the scanning of F into Callees and
    /// CalleeIndexMap.
    Node(LazyCallGraph &G, Function &F);

    /// \brief Internal helper to insert a callee.
    void insertEdgeInternal(Function &Callee);

    /// \brief Internal helper to insert a callee.
    void insertEdgeInternal(Node &CalleeN);

    /// \brief Internal helper to remove a callee from this node.
    void removeEdgeInternal(Function &Callee);

  public:
    typedef LazyCallGraph::iterator iterator;

    Function &getFunction() const {
      return F;
    };

    iterator begin() const {
      return iterator(*G, Callees.begin(), Callees.end());
    }
    iterator end() const { return iterator(*G, Callees.end(), Callees.end()); }

    /// Equality is defined as address equality.
    bool operator==(const Node &N) const { return this == &N; }
    bool operator!=(const Node &N) const { return !operator==(N); }
  };

  /// \brief An SCC of the call graph.
  ///
  /// This represents a Strongly Connected Component of the call graph as
  /// a collection of call graph nodes. While the order of nodes in the SCC is
  /// stable, it is not any particular order.
  class SCC {
    friend class LazyCallGraph;
    friend class LazyCallGraph::Node;

    LazyCallGraph *G;
    SmallPtrSet<SCC *, 1> ParentSCCs;
    SmallVector<Node *, 1> Nodes;

    SCC(LazyCallGraph &G) : G(&G) {}

    void insert(Node &N);

    void
    internalDFS(SmallVectorImpl<std::pair<Node *, Node::iterator>> &DFSStack,
                SmallVectorImpl<Node *> &PendingSCCStack, Node *N,
                SmallVectorImpl<SCC *> &ResultSCCs);

  public:
    typedef SmallVectorImpl<Node *>::const_iterator iterator;
    typedef pointee_iterator<SmallPtrSet<SCC *, 1>::const_iterator> parent_iterator;

    iterator begin() const { return Nodes.begin(); }
    iterator end() const { return Nodes.end(); }

    parent_iterator parent_begin() const { return ParentSCCs.begin(); }
    parent_iterator parent_end() const { return ParentSCCs.end(); }

    iterator_range<parent_iterator> parents() const {
      return iterator_range<parent_iterator>(parent_begin(), parent_end());
    }

    /// \brief Test if this SCC is a parent of \a C.
    bool isParentOf(const SCC &C) const { return C.isChildOf(*this); }

    /// \brief Test if this SCC is an ancestor of \a C.
    bool isAncestorOf(const SCC &C) const { return C.isDescendantOf(*this); }

    /// \brief Test if this SCC is a child of \a C.
    bool isChildOf(const SCC &C) const {
      return ParentSCCs.count(const_cast<SCC *>(&C));
    }

    /// \brief Test if this SCC is a descendant of \a C.
    bool isDescendantOf(const SCC &C) const;

    /// \brief Short name useful for debugging or logging.
    ///
    /// We use the name of the first function in the SCC to name the SCC for
    /// the purposes of debugging and logging.
    StringRef getName() const { return (*begin())->getFunction().getName(); }

    ///@{
    /// \name Mutation API
    ///
    /// These methods provide the core API for updating the call graph in the
    /// presence of a (potentially still in-flight) DFS-found SCCs.
    ///
    /// Note that these methods sometimes have complex runtimes, so be careful
    /// how you call them.

    /// \brief Insert an edge from one node in this SCC to another in this SCC.
    ///
    /// By the definition of an SCC, this does not change the nature or make-up
    /// of any SCCs.
    void insertIntraSCCEdge(Node &CallerN, Node &CalleeN);

    /// \brief Insert an edge whose tail is in this SCC and head is in some
    /// child SCC.
    ///
    /// There must be an existing path from the caller to the callee. This
    /// operation is inexpensive and does not change the set of SCCs in the
    /// graph.
    void insertOutgoingEdge(Node &CallerN, Node &CalleeN);

    /// \brief Insert an edge whose tail is in a descendant SCC and head is in
    /// this SCC.
    ///
    /// There must be an existing path from the callee to the caller in this
    /// case. NB! This is has the potential to be a very expensive function. It
    /// inherently forms a cycle in the prior SCC DAG and we have to merge SCCs
    /// to resolve that cycle. But finding all of the SCCs which participate in
    /// the cycle can in the worst case require traversing every SCC in the
    /// graph. Every attempt is made to avoid that, but passes must still
    /// exercise caution calling this routine repeatedly.
    ///
    /// FIXME: We could possibly optimize this quite a bit for cases where the
    /// caller and callee are very nearby in the graph. See comments in the
    /// implementation for details, but that use case might impact users.
    SmallVector<SCC *, 1> insertIncomingEdge(Node &CallerN, Node &CalleeN);

    /// \brief Remove an edge whose source is in this SCC and target is *not*.
    ///
    /// This removes an inter-SCC edge. All inter-SCC edges originating from
    /// this SCC have been fully explored by any in-flight DFS SCC formation,
    /// so this is always safe to call once you have the source SCC.
    ///
    /// This operation does not change the set of SCCs or the members of the
    /// SCCs and so is very inexpensive. It may change the connectivity graph
    /// of the SCCs though, so be careful calling this while iterating over
    /// them.
    void removeInterSCCEdge(Node &CallerN, Node &CalleeN);

    /// \brief Remove an edge which is entirely within this SCC.
    ///
    /// Both the \a Caller and the \a Callee must be within this SCC. Removing
    /// such an edge make break cycles that form this SCC and thus this
    /// operation may change the SCC graph significantly. In particular, this
    /// operation will re-form new SCCs based on the remaining connectivity of
    /// the graph. The following invariants are guaranteed to hold after
    /// calling this method:
    ///
    /// 1) This SCC is still an SCC in the graph.
    /// 2) This SCC will be the parent of any new SCCs. Thus, this SCC is
    ///    preserved as the root of any new SCC directed graph formed.
    /// 3) No SCC other than this SCC has its member set changed (this is
    ///    inherent in the definition of removing such an edge).
    /// 4) All of the parent links of the SCC graph will be updated to reflect
    ///    the new SCC structure.
    /// 5) All SCCs formed out of this SCC, excluding this SCC, will be
    ///    returned in a vector.
    /// 6) The order of the SCCs in the vector will be a valid postorder
    ///    traversal of the new SCCs.
    ///
    /// These invariants are very important to ensure that we can build
    /// optimization pipeliens on top of the CGSCC pass manager which
    /// intelligently update the SCC graph without invalidating other parts of
    /// the SCC graph.
    ///
    /// The runtime complexity of this method is, in the worst case, O(V+E)
    /// where V is the number of nodes in this SCC and E is the number of edges
    /// leaving the nodes in this SCC. Note that E includes both edges within
    /// this SCC and edges from this SCC to child SCCs. Some effort has been
    /// made to minimize the overhead of common cases such as self-edges and
    /// edge removals which result in a spanning tree with no more cycles.
    SmallVector<SCC *, 1> removeIntraSCCEdge(Node &CallerN, Node &CalleeN);

    ///@}
  };

  /// \brief A post-order depth-first SCC iterator over the call graph.
  ///
  /// This iterator triggers the Tarjan DFS-based formation of the SCC DAG for
  /// the call graph, walking it lazily in depth-first post-order. That is, it
  /// always visits SCCs for a callee prior to visiting the SCC for a caller
  /// (when they are in different SCCs).
  class postorder_scc_iterator
      : public iterator_facade_base<postorder_scc_iterator,
                                    std::forward_iterator_tag, SCC> {
    friend class LazyCallGraph;
    friend class LazyCallGraph::Node;

    /// \brief Nonce type to select the constructor for the end iterator.
    struct IsAtEndT {};

    LazyCallGraph *G;
    SCC *C;

    // Build the begin iterator for a node.
    postorder_scc_iterator(LazyCallGraph &G) : G(&G) {
      C = G.getNextSCCInPostOrder();
    }

    // Build the end iterator for a node. This is selected purely by overload.
    postorder_scc_iterator(LazyCallGraph &G, IsAtEndT /*Nonce*/)
        : G(&G), C(nullptr) {}

  public:
    bool operator==(const postorder_scc_iterator &Arg) const {
      return G == Arg.G && C == Arg.C;
    }

    reference operator*() const { return *C; }

    using iterator_facade_base::operator++;
    postorder_scc_iterator &operator++() {
      C = G->getNextSCCInPostOrder();
      return *this;
    }
  };

  /// \brief Construct a graph for the given module.
  ///
  /// This sets up the graph and computes all of the entry points of the graph.
  /// No function definitions are scanned until their nodes in the graph are
  /// requested during traversal.
  LazyCallGraph(Module &M);

  LazyCallGraph(LazyCallGraph &&G);
  LazyCallGraph &operator=(LazyCallGraph &&RHS);

  iterator begin() {
    return iterator(*this, EntryNodes.begin(), EntryNodes.end());
  }
  iterator end() { return iterator(*this, EntryNodes.end(), EntryNodes.end()); }

  postorder_scc_iterator postorder_scc_begin() {
    return postorder_scc_iterator(*this);
  }
  postorder_scc_iterator postorder_scc_end() {
    return postorder_scc_iterator(*this, postorder_scc_iterator::IsAtEndT());
  }

  iterator_range<postorder_scc_iterator> postorder_sccs() {
    return iterator_range<postorder_scc_iterator>(postorder_scc_begin(),
                                                  postorder_scc_end());
  }

  /// \brief Lookup a function in the graph which has already been scanned and
  /// added.
  Node *lookup(const Function &F) const { return NodeMap.lookup(&F); }

  /// \brief Lookup a function's SCC in the graph.
  ///
  /// \returns null if the function hasn't been assigned an SCC via the SCC
  /// iterator walk.
  SCC *lookupSCC(Node &N) const { return SCCMap.lookup(&N); }

  /// \brief Get a graph node for a given function, scanning it to populate the
  /// graph data as necessary.
  Node &get(Function &F) {
    Node *&N = NodeMap[&F];
    if (N)
      return *N;

    return insertInto(F, N);
  }

  ///@{
  /// \name Pre-SCC Mutation API
  ///
  /// These methods are only valid to call prior to forming any SCCs for this
  /// call graph. They can be used to update the core node-graph during
  /// a node-based inorder traversal that precedes any SCC-based traversal.
  ///
  /// Once you begin manipulating a call graph's SCCs, you must perform all
  /// mutation of the graph via the SCC methods.

  /// \brief Update the call graph after inserting a new edge.
  void insertEdge(Node &Caller, Function &Callee);

  /// \brief Update the call graph after inserting a new edge.
  void insertEdge(Function &Caller, Function &Callee) {
    return insertEdge(get(Caller), Callee);
  }

  /// \brief Update the call graph after deleting an edge.
  void removeEdge(Node &Caller, Function &Callee);

  /// \brief Update the call graph after deleting an edge.
  void removeEdge(Function &Caller, Function &Callee) {
    return removeEdge(get(Caller), Callee);
  }

  ///@}

private:
  /// \brief Allocator that holds all the call graph nodes.
  SpecificBumpPtrAllocator<Node> BPA;

  /// \brief Maps function->node for fast lookup.
  DenseMap<const Function *, Node *> NodeMap;

  /// \brief The entry nodes to the graph.
  ///
  /// These nodes are reachable through "external" means. Put another way, they
  /// escape at the module scope.
  NodeVectorT EntryNodes;

  /// \brief Map of the entry nodes in the graph to their indices in
  /// \c EntryNodes.
  DenseMap<Function *, size_t> EntryIndexMap;

  /// \brief Allocator that holds all the call graph SCCs.
  SpecificBumpPtrAllocator<SCC> SCCBPA;

  /// \brief Maps Function -> SCC for fast lookup.
  DenseMap<Node *, SCC *> SCCMap;

  /// \brief The leaf SCCs of the graph.
  ///
  /// These are all of the SCCs which have no children.
  SmallVector<SCC *, 4> LeafSCCs;

  /// \brief Stack of nodes in the DFS walk.
  SmallVector<std::pair<Node *, iterator>, 4> DFSStack;

  /// \brief Set of entry nodes not-yet-processed into SCCs.
  SmallVector<Function *, 4> SCCEntryNodes;

  /// \brief Stack of nodes the DFS has walked but not yet put into a SCC.
  SmallVector<Node *, 4> PendingSCCStack;

  /// \brief Counter for the next DFS number to assign.
  int NextDFSNumber;

  /// \brief Helper to insert a new function, with an already looked-up entry in
  /// the NodeMap.
  Node &insertInto(Function &F, Node *&MappedN);

  /// \brief Helper to update pointers back to the graph object during moves.
  void updateGraphPtrs();

  /// \brief Helper to form a new SCC out of the top of a DFSStack-like
  /// structure.
  SCC *formSCC(Node *RootN, SmallVectorImpl<Node *> &NodeStack);

  /// \brief Retrieve the next node in the post-order SCC walk of the call graph.
  SCC *getNextSCCInPostOrder();
};

// Provide GraphTraits specializations for call graphs.
template <> struct GraphTraits<LazyCallGraph::Node *> {
  typedef LazyCallGraph::Node NodeType;
  typedef LazyCallGraph::iterator ChildIteratorType;

  static NodeType *getEntryNode(NodeType *N) { return N; }
  static ChildIteratorType child_begin(NodeType *N) { return N->begin(); }
  static ChildIteratorType child_end(NodeType *N) { return N->end(); }
};
template <> struct GraphTraits<LazyCallGraph *> {
  typedef LazyCallGraph::Node NodeType;
  typedef LazyCallGraph::iterator ChildIteratorType;

  static NodeType *getEntryNode(NodeType *N) { return N; }
  static ChildIteratorType child_begin(NodeType *N) { return N->begin(); }
  static ChildIteratorType child_end(NodeType *N) { return N->end(); }
};

/// \brief An analysis pass which computes the call graph for a module.
class LazyCallGraphAnalysis {
public:
  /// \brief Inform generic clients of the result type.
  typedef LazyCallGraph Result;

  static void *ID() { return (void *)&PassID; }

  static StringRef name() { return "Lazy CallGraph Analysis"; }

  /// \brief Compute the \c LazyCallGraph for the module \c M.
  ///
  /// This just builds the set of entry points to the call graph. The rest is
  /// built lazily as it is walked.
  LazyCallGraph run(Module &M) { return LazyCallGraph(M); }

private:
  static char PassID;
};

/// \brief A pass which prints the call graph to a \c raw_ostream.
///
/// This is primarily useful for testing the analysis.
class LazyCallGraphPrinterPass {
  raw_ostream &OS;

public:
  explicit LazyCallGraphPrinterPass(raw_ostream &OS);

  PreservedAnalyses run(Module &M, ModuleAnalysisManager *AM);

  static StringRef name() { return "LazyCallGraphPrinterPass"; }
};

}

#endif
