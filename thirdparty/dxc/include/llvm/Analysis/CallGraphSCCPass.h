//===- CallGraphSCCPass.h - Pass that operates BU on call graph -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the CallGraphSCCPass class, which is used for passes which
// are implemented as bottom-up traversals on the call graph.  Because there may
// be cycles in the call graph, passes of this type operate on the call-graph in
// SCC order: that is, they process function bottom-up, except for recursive
// functions, which they process all at once.
//
// These passes are inherently interprocedural, and are required to keep the
// call graph up-to-date if they do anything which could modify it.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CALLGRAPHSCCPASS_H
#define LLVM_ANALYSIS_CALLGRAPHSCCPASS_H

#include "llvm/Analysis/CallGraph.h"
#include "llvm/Pass.h"

namespace llvm {

class CallGraphNode;
class CallGraph;
class PMStack;
class CallGraphSCC;
  
class CallGraphSCCPass : public Pass {
public:
  explicit CallGraphSCCPass(char &pid) : Pass(PT_CallGraphSCC, pid) {}

  /// createPrinterPass - Get a pass that prints the Module
  /// corresponding to a CallGraph.
  Pass *createPrinterPass(raw_ostream &O,
                          const std::string &Banner) const override;

  using llvm::Pass::doInitialization;
  using llvm::Pass::doFinalization;

  /// doInitialization - This method is called before the SCC's of the program
  /// has been processed, allowing the pass to do initialization as necessary.
  virtual bool doInitialization(CallGraph &CG) {
    return false;
  }

  /// runOnSCC - This method should be implemented by the subclass to perform
  /// whatever action is necessary for the specified SCC.  Note that
  /// non-recursive (or only self-recursive) functions will have an SCC size of
  /// 1, where recursive portions of the call graph will have SCC size > 1.
  ///
  /// SCC passes that add or delete functions to the SCC are required to update
  /// the SCC list, otherwise stale pointers may be dereferenced.
  ///
  virtual bool runOnSCC(CallGraphSCC &SCC) = 0;

  /// doFinalization - This method is called after the SCC's of the program has
  /// been processed, allowing the pass to do final cleanup as necessary.
  virtual bool doFinalization(CallGraph &CG) {
    return false;
  }

  /// Assign pass manager to manager this pass
  void assignPassManager(PMStack &PMS, PassManagerType PMT) override;

  ///  Return what kind of Pass Manager can manage this pass.
  PassManagerType getPotentialPassManagerType() const override {
    return PMT_CallGraphPassManager;
  }

  /// getAnalysisUsage - For this class, we declare that we require and preserve
  /// the call graph.  If the derived class implements this method, it should
  /// always explicitly call the implementation here.
  void getAnalysisUsage(AnalysisUsage &Info) const override;
};

/// CallGraphSCC - This is a single SCC that a CallGraphSCCPass is run on. 
class CallGraphSCC {
  void *Context; // The CGPassManager object that is vending this.
  std::vector<CallGraphNode*> Nodes;
public:
  CallGraphSCC(void *context) : Context(context) {}
  
  void initialize(CallGraphNode*const*I, CallGraphNode*const*E) {
    Nodes.assign(I, E);
  }
  
  bool isSingular() const { return Nodes.size() == 1; }
  unsigned size() const { return Nodes.size(); }
  
  /// ReplaceNode - This informs the SCC and the pass manager that the specified
  /// Old node has been deleted, and New is to be used in its place.
  void ReplaceNode(CallGraphNode *Old, CallGraphNode *New);
  
  typedef std::vector<CallGraphNode*>::const_iterator iterator;
  iterator begin() const { return Nodes.begin(); }
  iterator end() const { return Nodes.end(); }
};

} // End llvm namespace

#endif
