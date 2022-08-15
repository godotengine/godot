//===- ScopedNoAliasAA.cpp - Scoped No-Alias Alias Analysis ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ScopedNoAlias alias-analysis pass, which implements
// metadata-based scoped no-alias support.
//
// Alias-analysis scopes are defined by an id (which can be a string or some
// other metadata node), a domain node, and an optional descriptive string.
// A domain is defined by an id (which can be a string or some other metadata
// node), and an optional descriptive string.
//
// !dom0 =   metadata !{ metadata !"domain of foo()" }
// !scope1 = metadata !{ metadata !scope1, metadata !dom0, metadata !"scope 1" }
// !scope2 = metadata !{ metadata !scope2, metadata !dom0, metadata !"scope 2" }
//
// Loads and stores can be tagged with an alias-analysis scope, and also, with
// a noalias tag for a specific scope:
//
// ... = load %ptr1, !alias.scope !{ !scope1 }
// ... = load %ptr2, !alias.scope !{ !scope1, !scope2 }, !noalias !{ !scope1 }
//
// When evaluating an aliasing query, if one of the instructions is associated
// has a set of noalias scopes in some domain that is superset of the alias
// scopes in that domain of some other instruction, then the two memory
// accesses are assumed not to alias.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

// A handy option for disabling scoped no-alias functionality. The same effect
// can also be achieved by stripping the associated metadata tags from IR, but
// this option is sometimes more convenient.
#if 0 // HLSL Change Starts - option pending
static cl::opt<bool>
EnableScopedNoAlias("enable-scoped-noalias", cl::init(true));
#else
static const bool EnableScopedNoAlias = true;
#endif // HLSL Change Ends

namespace {
/// AliasScopeNode - This is a simple wrapper around an MDNode which provides
/// a higher-level interface by hiding the details of how alias analysis
/// information is encoded in its operands.
class AliasScopeNode {
  const MDNode *Node;

public:
  AliasScopeNode() : Node(0) {}
  explicit AliasScopeNode(const MDNode *N) : Node(N) {}

  /// getNode - Get the MDNode for this AliasScopeNode.
  const MDNode *getNode() const { return Node; }

  /// getDomain - Get the MDNode for this AliasScopeNode's domain.
  const MDNode *getDomain() const {
    if (Node->getNumOperands() < 2)
      return nullptr;
    return dyn_cast_or_null<MDNode>(Node->getOperand(1));
  }
};

/// ScopedNoAliasAA - This is a simple alias analysis
/// implementation that uses scoped-noalias metadata to answer queries.
class ScopedNoAliasAA : public ImmutablePass, public AliasAnalysis {
public:
  static char ID; // Class identification, replacement for typeinfo
  ScopedNoAliasAA() : ImmutablePass(ID) {
    initializeScopedNoAliasAAPass(*PassRegistry::getPassRegistry());
  }

  bool doInitialization(Module &M) override;

  /// getAdjustedAnalysisPointer - This method is used when a pass implements
  /// an analysis interface through multiple inheritance.  If needed, it
  /// should override this to adjust the this pointer as needed for the
  /// specified pass info.
  void *getAdjustedAnalysisPointer(const void *PI) override {
    if (PI == &AliasAnalysis::ID)
      return (AliasAnalysis*)this;
    return this;
  }

protected:
  bool mayAliasInScopes(const MDNode *Scopes, const MDNode *NoAlias) const;
  void collectMDInDomain(const MDNode *List, const MDNode *Domain,
                         SmallPtrSetImpl<const MDNode *> &Nodes) const;

private:
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  AliasResult alias(const MemoryLocation &LocA,
                    const MemoryLocation &LocB) override;
  bool pointsToConstantMemory(const MemoryLocation &Loc, bool OrLocal) override;
  ModRefBehavior getModRefBehavior(ImmutableCallSite CS) override;
  ModRefBehavior getModRefBehavior(const Function *F) override;
  ModRefResult getModRefInfo(ImmutableCallSite CS,
                             const MemoryLocation &Loc) override;
  ModRefResult getModRefInfo(ImmutableCallSite CS1,
                             ImmutableCallSite CS2) override;
};
}  // End of anonymous namespace

// Register this pass...
char ScopedNoAliasAA::ID = 0;
INITIALIZE_AG_PASS(ScopedNoAliasAA, AliasAnalysis, "scoped-noalias",
                   "Scoped NoAlias Alias Analysis", false, true, false)

ImmutablePass *llvm::createScopedNoAliasAAPass() {
  return new ScopedNoAliasAA();
}

bool ScopedNoAliasAA::doInitialization(Module &M) {
  InitializeAliasAnalysis(this, &M.getDataLayout());
  return true;
}

void
ScopedNoAliasAA::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AliasAnalysis::getAnalysisUsage(AU);
}

void
ScopedNoAliasAA::collectMDInDomain(const MDNode *List, const MDNode *Domain,
                   SmallPtrSetImpl<const MDNode *> &Nodes) const {
  for (unsigned i = 0, ie = List->getNumOperands(); i != ie; ++i)
    if (const MDNode *MD = dyn_cast<MDNode>(List->getOperand(i)))
      if (AliasScopeNode(MD).getDomain() == Domain)
        Nodes.insert(MD);
}

bool
ScopedNoAliasAA::mayAliasInScopes(const MDNode *Scopes,
                                  const MDNode *NoAlias) const {
  if (!Scopes || !NoAlias)
    return true;

  // Collect the set of scope domains relevant to the noalias scopes.
  SmallPtrSet<const MDNode *, 16> Domains;
  for (unsigned i = 0, ie = NoAlias->getNumOperands(); i != ie; ++i)
    if (const MDNode *NAMD = dyn_cast<MDNode>(NoAlias->getOperand(i)))
      if (const MDNode *Domain = AliasScopeNode(NAMD).getDomain())
        Domains.insert(Domain);

  // We alias unless, for some domain, the set of noalias scopes in that domain
  // is a superset of the set of alias scopes in that domain.
  for (const MDNode *Domain : Domains) {
    SmallPtrSet<const MDNode *, 16> NANodes, ScopeNodes;
    collectMDInDomain(NoAlias, Domain, NANodes);
    collectMDInDomain(Scopes, Domain, ScopeNodes);
    if (!ScopeNodes.size())
      continue;

    // To not alias, all of the nodes in ScopeNodes must be in NANodes.
    bool FoundAll = true;
    for (const MDNode *SMD : ScopeNodes)
      if (!NANodes.count(SMD)) {
        FoundAll = false;
        break;
      }

    if (FoundAll)
      return false;
  }

  return true;
}

AliasResult ScopedNoAliasAA::alias(const MemoryLocation &LocA,
                                   const MemoryLocation &LocB) {
  if (!EnableScopedNoAlias)
    return AliasAnalysis::alias(LocA, LocB);

  // Get the attached MDNodes.
  const MDNode *AScopes = LocA.AATags.Scope,
               *BScopes = LocB.AATags.Scope;

  const MDNode *ANoAlias = LocA.AATags.NoAlias,
               *BNoAlias = LocB.AATags.NoAlias;

  if (!mayAliasInScopes(AScopes, BNoAlias))
    return NoAlias;

  if (!mayAliasInScopes(BScopes, ANoAlias))
    return NoAlias;

  // If they may alias, chain to the next AliasAnalysis.
  return AliasAnalysis::alias(LocA, LocB);
}

bool ScopedNoAliasAA::pointsToConstantMemory(const MemoryLocation &Loc,
                                             bool OrLocal) {
  return AliasAnalysis::pointsToConstantMemory(Loc, OrLocal);
}

AliasAnalysis::ModRefBehavior
ScopedNoAliasAA::getModRefBehavior(ImmutableCallSite CS) {
  return AliasAnalysis::getModRefBehavior(CS);
}

AliasAnalysis::ModRefBehavior
ScopedNoAliasAA::getModRefBehavior(const Function *F) {
  return AliasAnalysis::getModRefBehavior(F);
}

AliasAnalysis::ModRefResult
ScopedNoAliasAA::getModRefInfo(ImmutableCallSite CS,
                               const MemoryLocation &Loc) {
  if (!EnableScopedNoAlias)
    return AliasAnalysis::getModRefInfo(CS, Loc);

  if (!mayAliasInScopes(Loc.AATags.Scope, CS.getInstruction()->getMetadata(
                                              LLVMContext::MD_noalias)))
    return NoModRef;

  if (!mayAliasInScopes(
          CS.getInstruction()->getMetadata(LLVMContext::MD_alias_scope),
          Loc.AATags.NoAlias))
    return NoModRef;

  return AliasAnalysis::getModRefInfo(CS, Loc);
}

AliasAnalysis::ModRefResult
ScopedNoAliasAA::getModRefInfo(ImmutableCallSite CS1, ImmutableCallSite CS2) {
  if (!EnableScopedNoAlias)
    return AliasAnalysis::getModRefInfo(CS1, CS2);

  if (!mayAliasInScopes(
          CS1.getInstruction()->getMetadata(LLVMContext::MD_alias_scope),
          CS2.getInstruction()->getMetadata(LLVMContext::MD_noalias)))
    return NoModRef;

  if (!mayAliasInScopes(
          CS2.getInstruction()->getMetadata(LLVMContext::MD_alias_scope),
          CS1.getInstruction()->getMetadata(LLVMContext::MD_noalias)))
    return NoModRef;

  return AliasAnalysis::getModRefInfo(CS1, CS2);
}

