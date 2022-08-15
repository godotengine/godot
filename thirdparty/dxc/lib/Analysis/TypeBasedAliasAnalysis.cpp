//===- TypeBasedAliasAnalysis.cpp - Type-Based Alias Analysis -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TypeBasedAliasAnalysis pass, which implements
// metadata-based TBAA.
//
// In LLVM IR, memory does not have types, so LLVM's own type system is not
// suitable for doing TBAA. Instead, metadata is added to the IR to describe
// a type system of a higher level language. This can be used to implement
// typical C/C++ TBAA, but it can also be used to implement custom alias
// analysis behavior for other languages.
//
// We now support two types of metadata format: scalar TBAA and struct-path
// aware TBAA. After all testing cases are upgraded to use struct-path aware
// TBAA and we can auto-upgrade existing bc files, the support for scalar TBAA
// can be dropped.
//
// The scalar TBAA metadata format is very simple. TBAA MDNodes have up to
// three fields, e.g.:
//   !0 = metadata !{ metadata !"an example type tree" }
//   !1 = metadata !{ metadata !"int", metadata !0 }
//   !2 = metadata !{ metadata !"float", metadata !0 }
//   !3 = metadata !{ metadata !"const float", metadata !2, i64 1 }
//
// The first field is an identity field. It can be any value, usually
// an MDString, which uniquely identifies the type. The most important
// name in the tree is the name of the root node. Two trees with
// different root node names are entirely disjoint, even if they
// have leaves with common names.
//
// The second field identifies the type's parent node in the tree, or
// is null or omitted for a root node. A type is considered to alias
// all of its descendants and all of its ancestors in the tree. Also,
// a type is considered to alias all types in other trees, so that
// bitcode produced from multiple front-ends is handled conservatively.
//
// If the third field is present, it's an integer which if equal to 1
// indicates that the type is "constant" (meaning pointsToConstantMemory
// should return true; see
// http://llvm.org/docs/AliasAnalysis.html#OtherItfs).
//
// With struct-path aware TBAA, the MDNodes attached to an instruction using
// "!tbaa" are called path tag nodes.
//
// The path tag node has 4 fields with the last field being optional.
//
// The first field is the base type node, it can be a struct type node
// or a scalar type node. The second field is the access type node, it
// must be a scalar type node. The third field is the offset into the base type.
// The last field has the same meaning as the last field of our scalar TBAA:
// it's an integer which if equal to 1 indicates that the access is "constant".
//
// The struct type node has a name and a list of pairs, one pair for each member
// of the struct. The first element of each pair is a type node (a struct type
// node or a sclar type node), specifying the type of the member, the second
// element of each pair is the offset of the member.
//
// Given an example
// typedef struct {
//   short s;
// } A;
// typedef struct {
//   uint16_t s;
//   A a;
// } B;
//
// For an acess to B.a.s, we attach !5 (a path tag node) to the load/store
// instruction. The base type is !4 (struct B), the access type is !2 (scalar
// type short) and the offset is 4.
//
// !0 = metadata !{metadata !"Simple C/C++ TBAA"}
// !1 = metadata !{metadata !"omnipotent char", metadata !0} // Scalar type node
// !2 = metadata !{metadata !"short", metadata !1}           // Scalar type node
// !3 = metadata !{metadata !"A", metadata !2, i64 0}        // Struct type node
// !4 = metadata !{metadata !"B", metadata !2, i64 0, metadata !3, i64 4}
//                                                           // Struct type node
// !5 = metadata !{metadata !4, metadata !2, i64 4}          // Path tag node
//
// The struct type nodes and the scalar type nodes form a type DAG.
//         Root (!0)
//         char (!1)  -- edge to Root
//         short (!2) -- edge to char
//         A (!3) -- edge with offset 0 to short
//         B (!4) -- edge with offset 0 to short and edge with offset 4 to A
//
// To check if two tags (tagX and tagY) can alias, we start from the base type
// of tagX, follow the edge with the correct offset in the type DAG and adjust
// the offset until we reach the base type of tagY or until we reach the Root
// node.
// If we reach the base type of tagY, compare the adjusted offset with
// offset of tagY, return Alias if the offsets are the same, return NoAlias
// otherwise.
// If we reach the Root node, perform the above starting from base type of tagY
// to see if we reach base type of tagX.
//
// If they have different roots, they're part of different potentially
// unrelated type systems, so we return Alias to be conservative.
// If neither node is an ancestor of the other and they have the same root,
// then we say NoAlias.
//
// TODO: The current metadata format doesn't support struct
// fields. For example:
//   struct X {
//     double d;
//     int i;
//   };
//   void foo(struct X *x, struct X *y, double *p) {
//     *x = *y;
//     *p = 0.0;
//   }
// Struct X has a double member, so the store to *x can alias the store to *p.
// Currently it's not possible to precisely describe all the things struct X
// aliases, so struct assignments must use conservative TBAA nodes. There's
// no scheme for attaching metadata to @llvm.memcpy yet either.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SetVector.h"
using namespace llvm;

// A handy option for disabling TBAA functionality. The same effect can also be
// achieved by stripping the !tbaa tags from IR, but this option is sometimes
// more convenient.
#if 0 // HLSL Change Starts - option pending
static cl::opt<bool> EnableTBAA("enable-tbaa", cl::init(true));
#else
static const bool EnableTBAA = true;
#endif // HLSL Change Ends

namespace {
  /// TBAANode - This is a simple wrapper around an MDNode which provides a
  /// higher-level interface by hiding the details of how alias analysis
  /// information is encoded in its operands.
  class TBAANode {
    const MDNode *Node;

  public:
    TBAANode() : Node(nullptr) {}
    explicit TBAANode(const MDNode *N) : Node(N) {}

    /// getNode - Get the MDNode for this TBAANode.
    const MDNode *getNode() const { return Node; }

    /// getParent - Get this TBAANode's Alias tree parent.
    TBAANode getParent() const {
      if (Node->getNumOperands() < 2)
        return TBAANode();
      MDNode *P = dyn_cast_or_null<MDNode>(Node->getOperand(1));
      if (!P)
        return TBAANode();
      // Ok, this node has a valid parent. Return it.
      return TBAANode(P);
    }

    /// TypeIsImmutable - Test if this TBAANode represents a type for objects
    /// which are not modified (by any means) in the context where this
    /// AliasAnalysis is relevant.
    bool TypeIsImmutable() const {
      if (Node->getNumOperands() < 3)
        return false;
      ConstantInt *CI = mdconst::dyn_extract<ConstantInt>(Node->getOperand(2));
      if (!CI)
        return false;
      return CI->getValue()[0];
    }
  };

  /// This is a simple wrapper around an MDNode which provides a
  /// higher-level interface by hiding the details of how alias analysis
  /// information is encoded in its operands.
  class TBAAStructTagNode {
    /// This node should be created with createTBAAStructTagNode.
    const MDNode *Node;

  public:
    explicit TBAAStructTagNode(const MDNode *N) : Node(N) {}

    /// Get the MDNode for this TBAAStructTagNode.
    const MDNode *getNode() const { return Node; }

    const MDNode *getBaseType() const {
      return dyn_cast_or_null<MDNode>(Node->getOperand(0));
    }
    const MDNode *getAccessType() const {
      return dyn_cast_or_null<MDNode>(Node->getOperand(1));
    }
    uint64_t getOffset() const {
      return mdconst::extract<ConstantInt>(Node->getOperand(2))->getZExtValue();
    }
    /// TypeIsImmutable - Test if this TBAAStructTagNode represents a type for
    /// objects which are not modified (by any means) in the context where this
    /// AliasAnalysis is relevant.
    bool TypeIsImmutable() const {
      if (Node->getNumOperands() < 4)
        return false;
      ConstantInt *CI = mdconst::dyn_extract<ConstantInt>(Node->getOperand(3));
      if (!CI)
        return false;
      return CI->getValue()[0];
    }
  };

  /// This is a simple wrapper around an MDNode which provides a
  /// higher-level interface by hiding the details of how alias analysis
  /// information is encoded in its operands.
  class TBAAStructTypeNode {
    /// This node should be created with createTBAAStructTypeNode.
    const MDNode *Node;

  public:
    TBAAStructTypeNode() : Node(nullptr) {}
    explicit TBAAStructTypeNode(const MDNode *N) : Node(N) {}

    /// Get the MDNode for this TBAAStructTypeNode.
    const MDNode *getNode() const { return Node; }

    /// Get this TBAAStructTypeNode's field in the type DAG with
    /// given offset. Update the offset to be relative to the field type.
    TBAAStructTypeNode getParent(uint64_t &Offset) const {
      // Parent can be omitted for the root node.
      if (Node->getNumOperands() < 2)
        return TBAAStructTypeNode();

      // Fast path for a scalar type node and a struct type node with a single
      // field.
      if (Node->getNumOperands() <= 3) {
        uint64_t Cur = Node->getNumOperands() == 2
                           ? 0
                           : mdconst::extract<ConstantInt>(Node->getOperand(2))
                                 ->getZExtValue();
        Offset -= Cur;
        MDNode *P = dyn_cast_or_null<MDNode>(Node->getOperand(1));
        if (!P)
          return TBAAStructTypeNode();
        return TBAAStructTypeNode(P);
      }

      // Assume the offsets are in order. We return the previous field if
      // the current offset is bigger than the given offset.
      unsigned TheIdx = 0;
      for (unsigned Idx = 1; Idx < Node->getNumOperands(); Idx += 2) {
        uint64_t Cur = mdconst::extract<ConstantInt>(Node->getOperand(Idx + 1))
                           ->getZExtValue();
        if (Cur > Offset) {
          assert(Idx >= 3 &&
                 "TBAAStructTypeNode::getParent should have an offset match!");
          TheIdx = Idx - 2;
          break;
        }
      }
      // Move along the last field.
      if (TheIdx == 0)
        TheIdx = Node->getNumOperands() - 2;
      uint64_t Cur = mdconst::extract<ConstantInt>(Node->getOperand(TheIdx + 1))
                         ->getZExtValue();
      Offset -= Cur;
      MDNode *P = dyn_cast_or_null<MDNode>(Node->getOperand(TheIdx));
      if (!P)
        return TBAAStructTypeNode();
      return TBAAStructTypeNode(P);
    }
  };
}

namespace {
  /// TypeBasedAliasAnalysis - This is a simple alias analysis
  /// implementation that uses TypeBased to answer queries.
  class TypeBasedAliasAnalysis : public ImmutablePass,
                                 public AliasAnalysis {
  public:
    static char ID; // Class identification, replacement for typeinfo
    TypeBasedAliasAnalysis() : ImmutablePass(ID) {
      initializeTypeBasedAliasAnalysisPass(*PassRegistry::getPassRegistry());
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

    bool Aliases(const MDNode *A, const MDNode *B) const;
    bool PathAliases(const MDNode *A, const MDNode *B) const;

  private:
    void getAnalysisUsage(AnalysisUsage &AU) const override;
    AliasResult alias(const MemoryLocation &LocA,
                      const MemoryLocation &LocB) override;
    bool pointsToConstantMemory(const MemoryLocation &Loc,
                                bool OrLocal) override;
    ModRefBehavior getModRefBehavior(ImmutableCallSite CS) override;
    ModRefBehavior getModRefBehavior(const Function *F) override;
    ModRefResult getModRefInfo(ImmutableCallSite CS,
                               const MemoryLocation &Loc) override;
    ModRefResult getModRefInfo(ImmutableCallSite CS1,
                               ImmutableCallSite CS2) override;
  };
}  // End of anonymous namespace

// Register this pass...
char TypeBasedAliasAnalysis::ID = 0;
INITIALIZE_AG_PASS(TypeBasedAliasAnalysis, AliasAnalysis, "tbaa",
                   "Type-Based Alias Analysis", false, true, false)

ImmutablePass *llvm::createTypeBasedAliasAnalysisPass() {
  return new TypeBasedAliasAnalysis();
}

bool TypeBasedAliasAnalysis::doInitialization(Module &M) {
  InitializeAliasAnalysis(this, &M.getDataLayout());
  return true;
}

void
TypeBasedAliasAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AliasAnalysis::getAnalysisUsage(AU);
}

/// Check the first operand of the tbaa tag node, if it is a MDNode, we treat
/// it as struct-path aware TBAA format, otherwise, we treat it as scalar TBAA
/// format.
static bool isStructPathTBAA(const MDNode *MD) {
  // Anonymous TBAA root starts with a MDNode and dragonegg uses it as
  // a TBAA tag.
  return isa<MDNode>(MD->getOperand(0)) && MD->getNumOperands() >= 3;
}

/// Aliases - Test whether the type represented by A may alias the
/// type represented by B.
bool
TypeBasedAliasAnalysis::Aliases(const MDNode *A,
                                const MDNode *B) const {
  // Make sure that both MDNodes are struct-path aware.
  if (isStructPathTBAA(A) && isStructPathTBAA(B))
    return PathAliases(A, B);

  // Keep track of the root node for A and B.
  TBAANode RootA, RootB;

  // Climb the tree from A to see if we reach B.
  for (TBAANode T(A); ; ) {
    if (T.getNode() == B)
      // B is an ancestor of A.
      return true;

    RootA = T;
    T = T.getParent();
    if (!T.getNode())
      break;
  }

  // Climb the tree from B to see if we reach A.
  for (TBAANode T(B); ; ) {
    if (T.getNode() == A)
      // A is an ancestor of B.
      return true;

    RootB = T;
    T = T.getParent();
    if (!T.getNode())
      break;
  }

  // Neither node is an ancestor of the other.
  
  // If they have different roots, they're part of different potentially
  // unrelated type systems, so we must be conservative.
  if (RootA.getNode() != RootB.getNode())
    return true;

  // If they have the same root, then we've proved there's no alias.
  return false;
}

/// Test whether the struct-path tag represented by A may alias the
/// struct-path tag represented by B.
bool
TypeBasedAliasAnalysis::PathAliases(const MDNode *A,
                                    const MDNode *B) const {
  // Verify that both input nodes are struct-path aware.
  assert(isStructPathTBAA(A) && "MDNode A is not struct-path aware.");
  assert(isStructPathTBAA(B) && "MDNode B is not struct-path aware.");

  // Keep track of the root node for A and B.
  TBAAStructTypeNode RootA, RootB;
  TBAAStructTagNode TagA(A), TagB(B);

  // TODO: We need to check if AccessType of TagA encloses AccessType of
  // TagB to support aggregate AccessType. If yes, return true.

  // Start from the base type of A, follow the edge with the correct offset in
  // the type DAG and adjust the offset until we reach the base type of B or
  // until we reach the Root node.
  // Compare the adjusted offset once we have the same base.

  // Climb the type DAG from base type of A to see if we reach base type of B.
  const MDNode *BaseA = TagA.getBaseType();
  const MDNode *BaseB = TagB.getBaseType();
  uint64_t OffsetA = TagA.getOffset(), OffsetB = TagB.getOffset();
  for (TBAAStructTypeNode T(BaseA); ; ) {
    if (T.getNode() == BaseB)
      // Base type of A encloses base type of B, check if the offsets match.
      return OffsetA == OffsetB;

    RootA = T;
    // Follow the edge with the correct offset, OffsetA will be adjusted to
    // be relative to the field type.
    T = T.getParent(OffsetA);
    if (!T.getNode())
      break;
  }

  // Reset OffsetA and climb the type DAG from base type of B to see if we reach
  // base type of A.
  OffsetA = TagA.getOffset();
  for (TBAAStructTypeNode T(BaseB); ; ) {
    if (T.getNode() == BaseA)
      // Base type of B encloses base type of A, check if the offsets match.
      return OffsetA == OffsetB;

    RootB = T;
    // Follow the edge with the correct offset, OffsetB will be adjusted to
    // be relative to the field type.
    T = T.getParent(OffsetB);
    if (!T.getNode())
      break;
  }

  // Neither node is an ancestor of the other.

  // If they have different roots, they're part of different potentially
  // unrelated type systems, so we must be conservative.
  if (RootA.getNode() != RootB.getNode())
    return true;

  // If they have the same root, then we've proved there's no alias.
  return false;
}

AliasResult TypeBasedAliasAnalysis::alias(const MemoryLocation &LocA,
                                          const MemoryLocation &LocB) {
  if (!EnableTBAA)
    return AliasAnalysis::alias(LocA, LocB);

  // Get the attached MDNodes. If either value lacks a tbaa MDNode, we must
  // be conservative.
  const MDNode *AM = LocA.AATags.TBAA;
  if (!AM) return AliasAnalysis::alias(LocA, LocB);
  const MDNode *BM = LocB.AATags.TBAA;
  if (!BM) return AliasAnalysis::alias(LocA, LocB);

  // If they may alias, chain to the next AliasAnalysis.
  if (Aliases(AM, BM))
    return AliasAnalysis::alias(LocA, LocB);

  // Otherwise return a definitive result.
  return NoAlias;
}

bool TypeBasedAliasAnalysis::pointsToConstantMemory(const MemoryLocation &Loc,
                                                    bool OrLocal) {
  if (!EnableTBAA)
    return AliasAnalysis::pointsToConstantMemory(Loc, OrLocal);

  const MDNode *M = Loc.AATags.TBAA;
  if (!M) return AliasAnalysis::pointsToConstantMemory(Loc, OrLocal);

  // If this is an "immutable" type, we can assume the pointer is pointing
  // to constant memory.
  if ((!isStructPathTBAA(M) && TBAANode(M).TypeIsImmutable()) ||
      (isStructPathTBAA(M) && TBAAStructTagNode(M).TypeIsImmutable()))
    return true;

  return AliasAnalysis::pointsToConstantMemory(Loc, OrLocal);
}

AliasAnalysis::ModRefBehavior
TypeBasedAliasAnalysis::getModRefBehavior(ImmutableCallSite CS) {
  if (!EnableTBAA)
    return AliasAnalysis::getModRefBehavior(CS);

  ModRefBehavior Min = UnknownModRefBehavior;

  // If this is an "immutable" type, we can assume the call doesn't write
  // to memory.
  if (const MDNode *M = CS.getInstruction()->getMetadata(LLVMContext::MD_tbaa))
    if ((!isStructPathTBAA(M) && TBAANode(M).TypeIsImmutable()) ||
        (isStructPathTBAA(M) && TBAAStructTagNode(M).TypeIsImmutable()))
      Min = OnlyReadsMemory;

  return ModRefBehavior(AliasAnalysis::getModRefBehavior(CS) & Min);
}

AliasAnalysis::ModRefBehavior
TypeBasedAliasAnalysis::getModRefBehavior(const Function *F) {
  // Functions don't have metadata. Just chain to the next implementation.
  return AliasAnalysis::getModRefBehavior(F);
}

AliasAnalysis::ModRefResult
TypeBasedAliasAnalysis::getModRefInfo(ImmutableCallSite CS,
                                      const MemoryLocation &Loc) {
  if (!EnableTBAA)
    return AliasAnalysis::getModRefInfo(CS, Loc);

  if (const MDNode *L = Loc.AATags.TBAA)
    if (const MDNode *M =
            CS.getInstruction()->getMetadata(LLVMContext::MD_tbaa))
      if (!Aliases(L, M))
        return NoModRef;

  return AliasAnalysis::getModRefInfo(CS, Loc);
}

AliasAnalysis::ModRefResult
TypeBasedAliasAnalysis::getModRefInfo(ImmutableCallSite CS1,
                                      ImmutableCallSite CS2) {
  if (!EnableTBAA)
    return AliasAnalysis::getModRefInfo(CS1, CS2);

  if (const MDNode *M1 =
          CS1.getInstruction()->getMetadata(LLVMContext::MD_tbaa))
    if (const MDNode *M2 =
            CS2.getInstruction()->getMetadata(LLVMContext::MD_tbaa))
      if (!Aliases(M1, M2))
        return NoModRef;

  return AliasAnalysis::getModRefInfo(CS1, CS2);
}

bool MDNode::isTBAAVtableAccess() const {
  if (!isStructPathTBAA(this)) {
    if (getNumOperands() < 1) return false;
    if (MDString *Tag1 = dyn_cast<MDString>(getOperand(0))) {
      if (Tag1->getString() == "vtable pointer") return true;
    }
    return false;
  }

  // For struct-path aware TBAA, we use the access type of the tag.
  if (getNumOperands() < 2) return false;
  MDNode *Tag = cast_or_null<MDNode>(getOperand(1));
  if (!Tag) return false;
  if (MDString *Tag1 = dyn_cast<MDString>(Tag->getOperand(0))) {
    if (Tag1->getString() == "vtable pointer") return true;
  }
  return false;  
}

MDNode *MDNode::getMostGenericTBAA(MDNode *A, MDNode *B) {
  if (!A || !B)
    return nullptr;

  if (A == B)
    return A;

  // For struct-path aware TBAA, we use the access type of the tag.
  bool StructPath = isStructPathTBAA(A) && isStructPathTBAA(B);
  if (StructPath) {
    A = cast_or_null<MDNode>(A->getOperand(1));
    if (!A) return nullptr;
    B = cast_or_null<MDNode>(B->getOperand(1));
    if (!B) return nullptr;
  }

  SmallSetVector<MDNode *, 4> PathA;
  MDNode *T = A;
  while (T) {
    if (PathA.count(T))
      report_fatal_error("Cycle found in TBAA metadata.");
    PathA.insert(T);
    T = T->getNumOperands() >= 2 ? cast_or_null<MDNode>(T->getOperand(1))
                                 : nullptr;
  }

  SmallSetVector<MDNode *, 4> PathB;
  T = B;
  while (T) {
    if (PathB.count(T))
      report_fatal_error("Cycle found in TBAA metadata.");
    PathB.insert(T);
    T = T->getNumOperands() >= 2 ? cast_or_null<MDNode>(T->getOperand(1))
                                 : nullptr;
  }

  int IA = PathA.size() - 1;
  int IB = PathB.size() - 1;

  MDNode *Ret = nullptr;
  while (IA >= 0 && IB >=0) {
    if (PathA[IA] == PathB[IB])
      Ret = PathA[IA];
    else
      break;
    --IA;
    --IB;
  }
  if (!StructPath)
    return Ret;

  if (!Ret)
    return nullptr;
  // We need to convert from a type node to a tag node.
  Type *Int64 = IntegerType::get(A->getContext(), 64);
  Metadata *Ops[3] = {Ret, Ret,
                      ConstantAsMetadata::get(ConstantInt::get(Int64, 0))};
  return MDNode::get(A->getContext(), Ops);
}

void Instruction::getAAMetadata(AAMDNodes &N, bool Merge) const {
  if (Merge)
    N.TBAA =
        MDNode::getMostGenericTBAA(N.TBAA, getMetadata(LLVMContext::MD_tbaa));
  else
    N.TBAA = getMetadata(LLVMContext::MD_tbaa);

  if (Merge)
    N.Scope = MDNode::getMostGenericAliasScope(
        N.Scope, getMetadata(LLVMContext::MD_alias_scope));
  else
    N.Scope = getMetadata(LLVMContext::MD_alias_scope);

  if (Merge)
    N.NoAlias =
        MDNode::intersect(N.NoAlias, getMetadata(LLVMContext::MD_noalias));
  else
    N.NoAlias = getMetadata(LLVMContext::MD_noalias);
}

