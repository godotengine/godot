//===-- LLVMContextImpl.cpp - Implement LLVMContextImpl -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the opaque LLVMContextImpl.
//
//===----------------------------------------------------------------------===//

#include "LLVMContextImpl.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Module.h"
#include <algorithm>
using namespace llvm;

LLVMContextImpl::LLVMContextImpl(LLVMContext &C)
  : TheTrueVal(nullptr), TheFalseVal(nullptr),
    VoidTy(C, Type::VoidTyID),
    LabelTy(C, Type::LabelTyID),
    HalfTy(C, Type::HalfTyID),
    FloatTy(C, Type::FloatTyID),
    DoubleTy(C, Type::DoubleTyID),
    MetadataTy(C, Type::MetadataTyID),
    X86_FP80Ty(C, Type::X86_FP80TyID),
    FP128Ty(C, Type::FP128TyID),
    PPC_FP128Ty(C, Type::PPC_FP128TyID),
    X86_MMXTy(C, Type::X86_MMXTyID),
    Int1Ty(C, 1),
    Int8Ty(C, 8),
    Int16Ty(C, 16),
    Int32Ty(C, 32),
    Int64Ty(C, 64),
    Int128Ty(C, 128) {
  InlineAsmDiagHandler = nullptr;
  InlineAsmDiagContext = nullptr;
  DiagnosticHandler = nullptr;
  DiagnosticContext = nullptr;
  RespectDiagnosticFilters = false;
  YieldCallback = nullptr;
  YieldOpaqueHandle = nullptr;
  NamedStructTypesUniqueID = 0;
}

namespace {
struct DropReferences {
  // Takes the value_type of a ConstantUniqueMap's internal map, whose 'second'
  // is a Constant*.
  template <typename PairT> void operator()(const PairT &P) {
    P.second->dropAllReferences();
  }
};

// Temporary - drops pair.first instead of second.
struct DropFirst {
  // Takes the value_type of a ConstantUniqueMap's internal map, whose 'second'
  // is a Constant*.
  template<typename PairT>
  void operator()(const PairT &P) {
    P.first->dropAllReferences();
  }
};
}

LLVMContextImpl::~LLVMContextImpl() {
  // NOTE: We need to delete the contents of OwnedModules, but Module's dtor
  // will call LLVMContextImpl::removeModule, thus invalidating iterators into
  // the container. Avoid iterators during this operation:
  while (!OwnedModules.empty())
    delete *OwnedModules.begin();

  // Drop references for MDNodes.  Do this before Values get deleted to avoid
  // unnecessary RAUW when nodes are still unresolved.
  for (auto *I : DistinctMDNodes)
    I->dropAllReferences();
#define HANDLE_MDNODE_LEAF(CLASS)                                              \
  for (auto *I : CLASS##s)                                                     \
    I->dropAllReferences();
#include "llvm/IR/Metadata.def"

  // Also drop references that come from the Value bridges.
  for (auto &Pair : ValuesAsMetadata)
    if (Pair.second) Pair.second->dropUsers(); // HLSL Change - if alloc failed, entry might not be populated
  for (auto &Pair : MetadataAsValues)
    if (Pair.second) Pair.second->dropUse(); // HLSL Change - if alloc failed, entry might not be populated

  // Destroy MDNodes.
  for (MDNode *I : DistinctMDNodes)
    I->deleteAsSubclass();
#define HANDLE_MDNODE_LEAF(CLASS)                                              \
  for (CLASS *I : CLASS##s)                                                    \
    delete I;
#include "llvm/IR/Metadata.def"

  // Free the constants.
  std::for_each(ExprConstants.map_begin(), ExprConstants.map_end(),
                DropFirst());
  std::for_each(ArrayConstants.map_begin(), ArrayConstants.map_end(),
                DropFirst());
  std::for_each(StructConstants.map_begin(), StructConstants.map_end(),
                DropFirst());
  std::for_each(VectorConstants.map_begin(), VectorConstants.map_end(),
                DropFirst());
  ExprConstants.freeConstants();
  ArrayConstants.freeConstants();
  StructConstants.freeConstants();
  VectorConstants.freeConstants();
  DeleteContainerSeconds(CAZConstants);
  DeleteContainerSeconds(CPNConstants);
  DeleteContainerSeconds(UVConstants);
  InlineAsms.freeConstants();
  DeleteContainerSeconds(IntConstants);
  DeleteContainerSeconds(FPConstants);
  
  for (StringMap<ConstantDataSequential*>::iterator I = CDSConstants.begin(),
       E = CDSConstants.end(); I != E; ++I)
    delete I->second;
  CDSConstants.clear();

  // Destroy attributes.
  for (FoldingSetIterator<AttributeImpl> I = AttrsSet.begin(),
         E = AttrsSet.end(); I != E; ) {
    FoldingSetIterator<AttributeImpl> Elem = I++;
    delete &*Elem;
  }

  // Destroy attribute lists.
  for (FoldingSetIterator<AttributeSetImpl> I = AttrsLists.begin(),
         E = AttrsLists.end(); I != E; ) {
    FoldingSetIterator<AttributeSetImpl> Elem = I++;
    delete &*Elem;
  }

  // Destroy attribute node lists.
  for (FoldingSetIterator<AttributeSetNode> I = AttrsSetNodes.begin(),
         E = AttrsSetNodes.end(); I != E; ) {
    FoldingSetIterator<AttributeSetNode> Elem = I++;
    delete &*Elem;
  }

  // Destroy MetadataAsValues.
  {
    SmallVector<MetadataAsValue *, 8> MDVs;
    MDVs.reserve(MetadataAsValues.size());
    for (auto &Pair : MetadataAsValues)
      MDVs.push_back(Pair.second);
    MetadataAsValues.clear();
    for (auto *V : MDVs)
      delete V;
  }

  // Destroy ValuesAsMetadata.
  for (auto &Pair : ValuesAsMetadata)
    delete Pair.second;

  // Destroy MDStrings.
  MDStringCache.clear();
}

void LLVMContextImpl::dropTriviallyDeadConstantArrays() {
  bool Changed;
  do {
    Changed = false;

    for (auto I = ArrayConstants.map_begin(), E = ArrayConstants.map_end();
         I != E; ) {
      auto *C = I->first;
      I++;
      if (C->use_empty()) {
        Changed = true;
        C->destroyConstant();
      }
    }

  } while (Changed);
}

void Module::dropTriviallyDeadConstantArrays() {
  Context.pImpl->dropTriviallyDeadConstantArrays();
}

namespace llvm {
/// \brief Make MDOperand transparent for hashing.
///
/// This overload of an implementation detail of the hashing library makes
/// MDOperand hash to the same value as a \a Metadata pointer.
///
/// Note that overloading \a hash_value() as follows:
///
/// \code
///     size_t hash_value(const MDOperand &X) { return hash_value(X.get()); }
/// \endcode
///
/// does not cause MDOperand to be transparent.  In particular, a bare pointer
/// doesn't get hashed before it's combined, whereas \a MDOperand would.
static const Metadata *get_hashable_data(const MDOperand &X) { return X.get(); }
}

unsigned MDNodeOpsKey::calculateHash(MDNode *N, unsigned Offset) {
  unsigned Hash = hash_combine_range(N->op_begin() + Offset, N->op_end());
#ifndef NDEBUG
  {
    SmallVector<Metadata *, 8> MDs(N->op_begin() + Offset, N->op_end());
    unsigned RawHash = calculateHash(MDs);
    assert(Hash == RawHash &&
           "Expected hash of MDOperand to equal hash of Metadata*");
  }
#endif
  return Hash;
}

unsigned MDNodeOpsKey::calculateHash(ArrayRef<Metadata *> Ops) {
  return hash_combine_range(Ops.begin(), Ops.end());
}

// ConstantsContext anchors
void UnaryConstantExpr::anchor() { }

void BinaryConstantExpr::anchor() { }

void SelectConstantExpr::anchor() { }

void ExtractElementConstantExpr::anchor() { }

void InsertElementConstantExpr::anchor() { }

void ShuffleVectorConstantExpr::anchor() { }

void ExtractValueConstantExpr::anchor() { }

void InsertValueConstantExpr::anchor() { }

void GetElementPtrConstantExpr::anchor() { }

void CompareConstantExpr::anchor() { }

