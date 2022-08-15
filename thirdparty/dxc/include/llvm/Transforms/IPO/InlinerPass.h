//===- InlinerPass.h - Code common to all inliners --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a simple policy-based bottom-up inliner.  This file
// implements all of the boring mechanics of the bottom-up inlining, while the
// subclass determines WHAT to inline, which is the much more interesting
// component.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_INLINERPASS_H
#define LLVM_TRANSFORMS_IPO_INLINERPASS_H

#include "llvm/Analysis/CallGraphSCCPass.h"

namespace llvm {
  class CallSite;
  class DataLayout;
  class InlineCost;
  template<class PtrType, unsigned SmallSize>
  class SmallPtrSet;

/// Inliner - This class contains all of the helper code which is used to
/// perform the inlining operations that do not depend on the policy.
///
struct Inliner : public CallGraphSCCPass {
  explicit Inliner(char &ID);
  explicit Inliner(char &ID, int Threshold, bool InsertLifetime);

  /// getAnalysisUsage - For this class, we declare that we require and preserve
  /// the call graph.  If the derived class implements this method, it should
  /// always explicitly call the implementation here.
  void getAnalysisUsage(AnalysisUsage &Info) const override;

  // Main run interface method, this implements the interface required by the
  // Pass class.
  bool runOnSCC(CallGraphSCC &SCC) override;

  using llvm::Pass::doFinalization;
  // doFinalization - Remove now-dead linkonce functions at the end of
  // processing to avoid breaking the SCC traversal.
  bool doFinalization(CallGraph &CG) override;

  /// This method returns the value specified by the -inline-threshold value,
  /// specified on the command line.  This is typically not directly needed.
  ///
  unsigned getInlineThreshold() const { return InlineThreshold; }

  /// Calculate the inline threshold for given Caller. This threshold is lower
  /// if the caller is marked with OptimizeForSize and -inline-threshold is not
  /// given on the comand line. It is higher if the callee is marked with the
  /// inlinehint attribute.
  ///
  unsigned getInlineThreshold(CallSite CS) const;

  /// getInlineCost - This method must be implemented by the subclass to
  /// determine the cost of inlining the specified call site.  If the cost
  /// returned is greater than the current inline threshold, the call site is
  /// not inlined.
  ///
  virtual InlineCost getInlineCost(CallSite CS) = 0;

  /// removeDeadFunctions - Remove dead functions.
  ///
  /// This also includes a hack in the form of the 'AlwaysInlineOnly' flag
  /// which restricts it to deleting functions with an 'AlwaysInline'
  /// attribute. This is useful for the InlineAlways pass that only wants to
  /// deal with that subset of the functions.
  bool removeDeadFunctions(CallGraph &CG, bool AlwaysInlineOnly = false);

  // HLSL Change Starts
  void applyOptions(PassOptions O) override;
  void dumpConfig(raw_ostream &OS) override;
  // HLSL Change Ends

private:
  // InlineThreshold - Cache the value here for easy access.
  unsigned InlineThreshold;

  // InsertLifetime - Insert @llvm.lifetime intrinsics.
  bool InsertLifetime;

  /// shouldInline - Return true if the inliner should attempt to
  /// inline at the given CallSite.
  bool shouldInline(CallSite CS);
};

} // End llvm namespace

#endif
