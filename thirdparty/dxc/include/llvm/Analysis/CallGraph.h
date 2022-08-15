//===- CallGraph.h - Build a Module's call graph ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file provides interfaces used to build and manipulate a call graph,
/// which is a very useful tool for interprocedural optimization.
///
/// Every function in a module is represented as a node in the call graph.  The
/// callgraph node keeps track of which functions are called by the function
/// corresponding to the node.
///
/// A call graph may contain nodes where the function that they correspond to
/// is null.  These 'external' nodes are used to represent control flow that is
/// not represented (or analyzable) in the module.  In particular, this
/// analysis builds one external node such that:
///   1. All functions in the module without internal linkage will have edges
///      from this external node, indicating that they could be called by
///      functions outside of the module.
///   2. All functions whose address is used for something more than a direct
///      call, for example being stored into a memory location will also have
///      an edge from this external node.  Since they may be called by an
///      unknown caller later, they must be tracked as such.
///
/// There is a second external node added for calls that leave this module.
/// Functions have a call edge to the external node iff:
///   1. The function is external, reflecting the fact that they could call
///      anything without internal linkage or that has its address taken.
///   2. The function contains an indirect function call.
///
/// As an extension in the future, there may be multiple nodes with a null
/// function.  These will be used when we can prove (through pointer analysis)
/// that an indirect call site can call only a specific set of functions.
///
/// Because of these properties, the CallGraph captures a conservative superset
/// of all of the caller-callee relationships, which is useful for
/// transformations.
///
/// The CallGraph class also attempts to figure out what the root of the
/// CallGraph is, which it currently does by looking for a function named
/// 'main'. If no function named 'main' is found, the external node is used as
/// the entry node, reflecting the fact that any function without internal
/// linkage could be called into (which is common for libraries).
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CALLGRAPH_H
#define LLVM_ANALYSIS_CALLGRAPH_H

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include <map>

namespace llvm {

class Function;
class Module;
class CallGraphNode;

/// \brief The basic data container for the call graph of a \c Module of IR.
///
/// This class exposes both the interface to the call graph for a module of IR.
///
/// The core call graph itself can also be updated to reflect changes to the IR.
class CallGraph {
  Module &M;

  typedef std::map<const Function *, std::unique_ptr<CallGraphNode>> FunctionMapTy;

  /// \brief A map from \c Function* to \c CallGraphNode*.
  FunctionMapTy FunctionMap;

  /// \brief Root is root of the call graph, or the external node if a 'main'
  /// function couldn't be found.
  CallGraphNode *Root;

  /// \brief This node has edges to all external functions and those internal
  /// functions that have their address taken.
  CallGraphNode *ExternalCallingNode;

  /// \brief This node has edges to it from all functions making indirect calls
  /// or calling an external function.
  std::unique_ptr<CallGraphNode> CallsExternalNode;

  /// \brief Replace the function represented by this node by another.
  ///
  /// This does not rescan the body of the function, so it is suitable when
  /// splicing the body of one function to another while also updating all
  /// callers from the old function to the new.
  void spliceFunction(const Function *From, const Function *To);

  /// \brief Add a function to the call graph, and link the node to all of the
  /// functions that it calls.
  void addToCallGraph(Function *F);

  void reset(); // HLSL Change
public:
  explicit CallGraph(Module &M);
  CallGraph(CallGraph &&Arg);
  ~CallGraph();

  void print(raw_ostream &OS) const;
  void dump() const;

  typedef FunctionMapTy::iterator iterator;
  typedef FunctionMapTy::const_iterator const_iterator;

  /// \brief Returns the module the call graph corresponds to.
  Module &getModule() const { return M; }

  inline iterator begin() { return FunctionMap.begin(); }
  inline iterator end() { return FunctionMap.end(); }
  inline const_iterator begin() const { return FunctionMap.begin(); }
  inline const_iterator end() const { return FunctionMap.end(); }

  /// \brief Returns the call graph node for the provided function.
  inline const CallGraphNode *operator[](const Function *F) const {
    const_iterator I = FunctionMap.find(F);
    assert(I != FunctionMap.end() && "Function not in callgraph!");
    return I->second.get();
  }

  /// \brief Returns the call graph node for the provided function.
  inline CallGraphNode *operator[](const Function *F) {
    const_iterator I = FunctionMap.find(F);
    assert(I != FunctionMap.end() && "Function not in callgraph!");
    return I->second.get();
  }

  /// \brief Returns the \c CallGraphNode which is used to represent
  /// undetermined calls into the callgraph.
  CallGraphNode *getExternalCallingNode() const { return ExternalCallingNode; }

  CallGraphNode *getCallsExternalNode() const { return CallsExternalNode.get(); }

  //===---------------------------------------------------------------------
  // Functions to keep a call graph up to date with a function that has been
  // modified.
  //

  /// \brief Unlink the function from this module, returning it.
  ///
  /// Because this removes the function from the module, the call graph node is
  /// destroyed.  This is only valid if the function does not call any other
  /// functions (ie, there are no edges in it's CGN).  The easiest way to do
  /// this is to dropAllReferences before calling this.
  Function *removeFunctionFromModule(CallGraphNode *CGN);

  /// \brief Similar to operator[], but this will insert a new CallGraphNode for
  /// \c F if one does not already exist.
  CallGraphNode *getOrInsertFunction(const Function *F);
};

/// \brief A node in the call graph for a module.
///
/// Typically represents a function in the call graph. There are also special
/// "null" nodes used to represent theoretical entries in the call graph.
class CallGraphNode {
public:
  /// \brief A pair of the calling instruction (a call or invoke)
  /// and the call graph node being called.
  typedef std::pair<WeakVH, CallGraphNode *> CallRecord;

public:
  typedef std::vector<CallRecord> CalledFunctionsVector;

  /// \brief Creates a node for the specified function.
  inline CallGraphNode(Function *F) : F(F), NumReferences(0) {}

  ~CallGraphNode() {
    assert(NumReferences == 0 && "Node deleted while references remain");
  }

  typedef std::vector<CallRecord>::iterator iterator;
  typedef std::vector<CallRecord>::const_iterator const_iterator;

  /// \brief Returns the function that this call graph node represents.
  Function *getFunction() const { return F; }

  inline iterator begin() { return CalledFunctions.begin(); }
  inline iterator end() { return CalledFunctions.end(); }
  inline const_iterator begin() const { return CalledFunctions.begin(); }
  inline const_iterator end() const { return CalledFunctions.end(); }
  inline bool empty() const { return CalledFunctions.empty(); }
  inline unsigned size() const { return (unsigned)CalledFunctions.size(); }

  /// \brief Returns the number of other CallGraphNodes in this CallGraph that
  /// reference this node in their callee list.
  unsigned getNumReferences() const { return NumReferences; }

  /// \brief Returns the i'th called function.
  CallGraphNode *operator[](unsigned i) const {
    assert(i < CalledFunctions.size() && "Invalid index");
    return CalledFunctions[i].second;
  }

  /// \brief Print out this call graph node.
  void dump() const;
  void print(raw_ostream &OS) const;

  //===---------------------------------------------------------------------
  // Methods to keep a call graph up to date with a function that has been
  // modified
  //

  /// \brief Removes all edges from this CallGraphNode to any functions it
  /// calls.
  void removeAllCalledFunctions() {
    while (!CalledFunctions.empty()) {
      CalledFunctions.back().second->DropRef();
      CalledFunctions.pop_back();
    }
  }

  /// \brief Moves all the callee information from N to this node.
  void stealCalledFunctionsFrom(CallGraphNode *N) {
    assert(CalledFunctions.empty() &&
           "Cannot steal callsite information if I already have some");
    std::swap(CalledFunctions, N->CalledFunctions);
  }

  /// \brief Adds a function to the list of functions called by this one.
  void addCalledFunction(CallSite CS, CallGraphNode *M) {
    assert(!CS.getInstruction() || !CS.getCalledFunction() ||
           !CS.getCalledFunction()->isIntrinsic() ||
           !Intrinsic::isLeaf(CS.getCalledFunction()->getIntrinsicID()));
    CalledFunctions.emplace_back(CS.getInstruction(), M);
    M->AddRef();
  }

  void removeCallEdge(iterator I) {
    I->second->DropRef();
    *I = CalledFunctions.back();
    CalledFunctions.pop_back();
  }

  /// \brief Removes the edge in the node for the specified call site.
  ///
  /// Note that this method takes linear time, so it should be used sparingly.
  void removeCallEdgeFor(CallSite CS);

  /// \brief Removes all call edges from this node to the specified callee
  /// function.
  ///
  /// This takes more time to execute than removeCallEdgeTo, so it should not
  /// be used unless necessary.
  void removeAnyCallEdgeTo(CallGraphNode *Callee);

  /// \brief Removes one edge associated with a null callsite from this node to
  /// the specified callee function.
  void removeOneAbstractEdgeTo(CallGraphNode *Callee);

  /// \brief Replaces the edge in the node for the specified call site with a
  /// new one.
  ///
  /// Note that this method takes linear time, so it should be used sparingly.
  void replaceCallEdge(CallSite CS, CallSite NewCS, CallGraphNode *NewNode);

private:
  friend class CallGraph;

  AssertingVH<Function> F;

  std::vector<CallRecord> CalledFunctions;

  /// \brief The number of times that this CallGraphNode occurs in the
  /// CalledFunctions array of this or other CallGraphNodes.
  unsigned NumReferences;

  CallGraphNode(const CallGraphNode &) = delete;
  void operator=(const CallGraphNode &) = delete;

  void DropRef() { --NumReferences; }
  void AddRef() { ++NumReferences; }

  /// \brief A special function that should only be used by the CallGraph class.
  void allReferencesDropped() { NumReferences = 0; }
};

/// \brief An analysis pass to compute the \c CallGraph for a \c Module.
///
/// This class implements the concept of an analysis pass used by the \c
/// ModuleAnalysisManager to run an analysis over a module and cache the
/// resulting data.
class CallGraphAnalysis {
public:
  /// \brief A formulaic typedef to inform clients of the result type.
  typedef CallGraph Result;

  static void *ID() { return (void *)&PassID; }

  /// \brief Compute the \c CallGraph for the module \c M.
  ///
  /// The real work here is done in the \c CallGraph constructor.
  CallGraph run(Module *M) { return CallGraph(*M); }

private:
  static char PassID;
};

/// \brief The \c ModulePass which wraps up a \c CallGraph and the logic to
/// build it.
///
/// This class exposes both the interface to the call graph container and the
/// module pass which runs over a module of IR and produces the call graph. The
/// call graph interface is entirelly a wrapper around a \c CallGraph object
/// which is stored internally for each module.
class CallGraphWrapperPass : public ModulePass {
  std::unique_ptr<CallGraph> G;

public:
  static char ID; // Class identification, replacement for typeinfo

  CallGraphWrapperPass();
  ~CallGraphWrapperPass() override;

  /// \brief The internal \c CallGraph around which the rest of this interface
  /// is wrapped.
  const CallGraph &getCallGraph() const { return *G; }
  CallGraph &getCallGraph() { return *G; }

  typedef CallGraph::iterator iterator;
  typedef CallGraph::const_iterator const_iterator;

  /// \brief Returns the module the call graph corresponds to.
  Module &getModule() const { return G->getModule(); }

  inline iterator begin() { return G->begin(); }
  inline iterator end() { return G->end(); }
  inline const_iterator begin() const { return G->begin(); }
  inline const_iterator end() const { return G->end(); }

  /// \brief Returns the call graph node for the provided function.
  inline const CallGraphNode *operator[](const Function *F) const {
    return (*G)[F];
  }

  /// \brief Returns the call graph node for the provided function.
  inline CallGraphNode *operator[](const Function *F) { return (*G)[F]; }

  /// \brief Returns the \c CallGraphNode which is used to represent
  /// undetermined calls into the callgraph.
  CallGraphNode *getExternalCallingNode() const {
    return G->getExternalCallingNode();
  }

  CallGraphNode *getCallsExternalNode() const {
    return G->getCallsExternalNode();
  }

  //===---------------------------------------------------------------------
  // Functions to keep a call graph up to date with a function that has been
  // modified.
  //

  /// \brief Unlink the function from this module, returning it.
  ///
  /// Because this removes the function from the module, the call graph node is
  /// destroyed.  This is only valid if the function does not call any other
  /// functions (ie, there are no edges in it's CGN).  The easiest way to do
  /// this is to dropAllReferences before calling this.
  Function *removeFunctionFromModule(CallGraphNode *CGN) {
    return G->removeFunctionFromModule(CGN);
  }

  /// \brief Similar to operator[], but this will insert a new CallGraphNode for
  /// \c F if one does not already exist.
  CallGraphNode *getOrInsertFunction(const Function *F) {
    return G->getOrInsertFunction(F);
  }

  //===---------------------------------------------------------------------
  // Implementation of the ModulePass interface needed here.
  //

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnModule(Module &M) override;
  void releaseMemory() override;

  void print(raw_ostream &o, const Module *) const override;
  void dump() const;
};
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
// GraphTraits specializations for call graphs so that they can be treated as
// graphs by the generic graph algorithms.
//

// Provide graph traits for tranversing call graphs using standard graph
// traversals.
template <> struct GraphTraits<CallGraphNode *> {
  typedef CallGraphNode NodeType;

  typedef CallGraphNode::CallRecord CGNPairTy;

  static NodeType *getEntryNode(CallGraphNode *CGN) { return CGN; }

  static CallGraphNode *CGNGetValue(CGNPairTy P) { return P.second; }
  typedef mapped_iterator<NodeType::iterator, decltype(&CGNGetValue)> ChildIteratorType;

  static inline ChildIteratorType child_begin(NodeType *N) {
    return ChildIteratorType(N->begin(), &CGNGetValue);
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return ChildIteratorType(N->end(), &CGNGetValue);
  }
};

template <> struct GraphTraits<const CallGraphNode *> {
  typedef const CallGraphNode NodeType;

  typedef CallGraphNode::CallRecord CGNPairTy;

  static NodeType *getEntryNode(const CallGraphNode *CGN) { return CGN; }

  static const CallGraphNode *CGNGetValue(CGNPairTy P) { return P.second; }

  typedef mapped_iterator<NodeType::const_iterator, decltype(&CGNGetValue)>
      ChildIteratorType;

  static inline ChildIteratorType child_begin(NodeType *N) {
    return ChildIteratorType(N->begin(), &CGNGetValue);
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return ChildIteratorType(N->end(), &CGNGetValue);
  }

};

template <>
struct GraphTraits<CallGraph *> : public GraphTraits<CallGraphNode *> {
  static NodeType *getEntryNode(CallGraph *CGN) {
    return CGN->getExternalCallingNode(); // Start at the external node!
  }
  typedef std::pair<const Function *const, std::unique_ptr<CallGraphNode>>
      PairTy;

  static CallGraphNode *CGGetValuePtr(const PairTy &P) { 
    return P.second.get();
  }

  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  typedef mapped_iterator<CallGraph::iterator, decltype(&CGGetValuePtr)> nodes_iterator;
  static nodes_iterator nodes_begin(CallGraph *CG) {
    return nodes_iterator(CG->begin(), &CGGetValuePtr);
  }
  static nodes_iterator nodes_end(CallGraph *CG) {
    return nodes_iterator(CG->end(), &CGGetValuePtr);
  }
};

template <>
struct GraphTraits<const CallGraph *> : public GraphTraits<
                                            const CallGraphNode *> {
  static NodeType *getEntryNode(const CallGraph *CGN) {
    return CGN->getExternalCallingNode(); // Start at the external node!
  }
  typedef std::pair<const Function *const, std::unique_ptr<CallGraphNode>>
       PairTy;

  static const CallGraphNode *CGGetValuePtr(const PairTy &P) {
    return P.second.get();
  }

  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  typedef mapped_iterator<CallGraph::const_iterator, decltype(&CGGetValuePtr)>
    nodes_iterator;
  static nodes_iterator nodes_begin(const CallGraph *CG) {
    return nodes_iterator(CG->begin(), &CGGetValuePtr);
  }
  static nodes_iterator nodes_end(const CallGraph *CG) {
    return nodes_iterator(CG->end(), &CGGetValuePtr);
  }
};

} // End llvm namespace

#endif
