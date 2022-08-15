///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDbgValueToDbgDeclare.cpp                                              //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Converts calls to llvm.dbg.value to llvm.dbg.declare + alloca + stores.   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <memory>
#include <map>
#include <unordered_map>
#include <utility>

#include "dxc/DXIL/DxilConstants.h"
#include "dxc/DXIL/DxilResourceBase.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DxilPIXPasses/DxilPIXPasses.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

#include "PixPassHelpers.h"
using namespace PIXPassHelpers;

using namespace llvm;

//#define VALUE_TO_DECLARE_LOGGING

#ifdef VALUE_TO_DECLARE_LOGGING
#ifndef PIX_DEBUG_DUMP_HELPER
#error Turn on PIX_DEBUG_DUMP_HELPER in PixPassHelpers.h
#endif
#define VALUE_TO_DECLARE_LOG Log
#else
#define VALUE_TO_DECLARE_LOG(...)
#endif

#define DEBUG_TYPE "dxil-dbg-value-to-dbg-declare"

namespace {
using OffsetInBits = unsigned;
using SizeInBits = unsigned;

// OffsetManager is used to map between "packed" and aligned offsets.
//
// For example, the aligned offsets for a struct [float, half, int, double]
// will be {0, 32, 64, 128} (assuming 32 bit alignments for ints, and 64
// bit for doubles), while the packed offsets will be {0, 32, 48, 80}.
//
// This mapping makes it easier to deal with llvm.dbg.values whose value
// operand does not match exactly the Variable operand's type.
class OffsetManager
{
  unsigned DescendTypeToGetAlignMask(llvm::DIType *Ty) {
    unsigned AlignMask = Ty->getAlignInBits();

    auto *DerivedTy = llvm::dyn_cast<llvm::DIDerivedType>(Ty);
    if (DerivedTy != nullptr) {
      // Working around a bug where byte size is stored instead of bit size
      if (AlignMask == 4 && Ty->getSizeInBits() == 32) {
        AlignMask = 32;
      }
      if (AlignMask == 0) {
        const llvm::DITypeIdentifierMap EmptyMap;
        switch (DerivedTy->getTag()) {
        case llvm::dwarf::DW_TAG_restrict_type:
        case llvm::dwarf::DW_TAG_reference_type:
        case llvm::dwarf::DW_TAG_const_type:
        case llvm::dwarf::DW_TAG_typedef: {
          llvm::DIType *baseType = DerivedTy->getBaseType().resolve(EmptyMap);
          if (baseType != nullptr) {
            if (baseType->getAlignInBits() == 0) {
              (void)baseType->getAlignInBits();
            }
            return DescendTypeToGetAlignMask(baseType);
          }
        }
        }
      }
    }
    return AlignMask;
  }

public:
  OffsetManager() = default;

  // AlignTo aligns the current aligned offset to Ty's natural alignment.
  void AlignTo(
      llvm::DIType *Ty
  )
  {
    unsigned AlignMask = DescendTypeToGetAlignMask(Ty);
    if (AlignMask) {
      VALUE_TO_DECLARE_LOG("Aligning to %d", AlignMask);
      // This is some magic arithmetic. Here's an example:
      //
      // Assume the natural alignment for Ty is 16 bits. Then
      //
      //     AlignMask = 0x0000000f(15)
      //
      // If the current aligned offset is 
      //
      //     CurrentAlignedOffset = 0x00000048(72)
      //
      // Then
      //
      //     T = CurrentAlignOffset + AlignMask = 0x00000057(87)
      //
      // Which mean
      //
      //     T & ~CurrentOffset = 0x00000050(80)
      //
      // is the aligned offset where Ty should be placed.
      AlignMask = AlignMask - 1;
      m_CurrentAlignedOffset =
          (m_CurrentAlignedOffset + AlignMask) & ~AlignMask;
    }
    else {
      VALUE_TO_DECLARE_LOG("Failed to find alignment");
    }
  }

  // Add is used to "add" an aggregate element (struct field, array element)
  // at the current aligned/packed offsets, bumping them by Ty's size.
  OffsetInBits Add(
      llvm::DIBasicType *Ty
  )
  {
    VALUE_TO_DECLARE_LOG("Adding known type at aligned %d / packed %d, size %d",
        m_CurrentAlignedOffset, m_CurrentPackedOffset, Ty->getSizeInBits());

    m_PackedOffsetToAlignedOffset[m_CurrentPackedOffset] = m_CurrentAlignedOffset;
    m_AlignedOffsetToPackedOffset[m_CurrentAlignedOffset] = m_CurrentPackedOffset;

    const OffsetInBits Ret = m_CurrentAlignedOffset;
    m_CurrentPackedOffset += Ty->getSizeInBits();
    m_CurrentAlignedOffset += Ty->getSizeInBits();

    return Ret;
  }

  // AlignToAndAddUnhandledType is used for error handling when Ty
  // could not be handled by the transformation. This is a best-effort
  // way to continue the pass by ignoring the current type and hoping
  // that adding Ty as a blob other fields/elements added will land
  // in the proper offset.
  void AlignToAndAddUnhandledType(
      llvm::DIType *Ty
  )
  {
      VALUE_TO_DECLARE_LOG("Adding unhandled type at aligned %d / packed %d, size %d",
        m_CurrentAlignedOffset, m_CurrentPackedOffset, Ty->getSizeInBits());
    AlignTo(Ty);
    m_CurrentPackedOffset += Ty->getSizeInBits();
    m_CurrentAlignedOffset += Ty->getSizeInBits();
  }

  void AddResourceType(llvm::DIType *Ty)
  {
    VALUE_TO_DECLARE_LOG("Adding resource type at aligned %d / packed %d, size %d", m_CurrentAlignedOffset, m_CurrentPackedOffset, Ty->getSizeInBits());
    m_PackedOffsetToAlignedOffset[m_CurrentPackedOffset] =
        m_CurrentAlignedOffset;
    m_AlignedOffsetToPackedOffset[m_CurrentAlignedOffset] =
        m_CurrentPackedOffset;

    m_CurrentPackedOffset += Ty->getSizeInBits();
    m_CurrentAlignedOffset += Ty->getSizeInBits();
  }

  bool GetAlignedOffsetFromPackedOffset(
      OffsetInBits PackedOffset,
      OffsetInBits *AlignedOffset
  ) const
  {
    return GetOffsetWithMap(
        m_PackedOffsetToAlignedOffset,
        PackedOffset,
        AlignedOffset);
  }

  bool GetPackedOffsetFromAlignedOffset(
      OffsetInBits AlignedOffset,
      OffsetInBits *PackedOffset
  ) const
  {
    return GetOffsetWithMap(
        m_AlignedOffsetToPackedOffset,
        AlignedOffset,
        PackedOffset);
  }

  OffsetInBits GetCurrentPackedOffset() const
  {
    return m_CurrentPackedOffset;
  }

  OffsetInBits GetCurrentAlignedOffset() const
  {
      return m_CurrentAlignedOffset;
  }

private:
  OffsetInBits m_CurrentPackedOffset = 0;
  OffsetInBits m_CurrentAlignedOffset = 0;

  using OffsetMap = std::unordered_map<OffsetInBits, OffsetInBits>;

  OffsetMap m_PackedOffsetToAlignedOffset;
  OffsetMap m_AlignedOffsetToPackedOffset;

  static bool GetOffsetWithMap(
      const OffsetMap &Map,
      OffsetInBits SrcOffset,
      OffsetInBits *DstOffset
  )
  {
    auto it = Map.find(SrcOffset);
    if (it == Map.end())
    {
      return false;
    }

    *DstOffset = it->second;
    return true;
  }
};

// VariableRegisters contains the logic for traversing a DIType T and
// creating AllocaInsts that map back to a specific offset within T.
class VariableRegisters
{
public:
  VariableRegisters(
      llvm::DbgValueInst* DbgValue,
      llvm::DIVariable *Variable,
      llvm::DIType* Ty,
      llvm::Module *M
  );

  llvm::AllocaInst *GetRegisterForAlignedOffset(
      OffsetInBits AlignedOffset
  ) const;

  const OffsetManager& GetOffsetManager() const
  {
    return m_Offsets;
  }

  static SizeInBits GetVariableSizeInbits(DIVariable *Var);

private:
  void PopulateAllocaMap(
      llvm::DIType *Ty
  );

  void PopulateAllocaMap_BasicType(llvm::DIBasicType *Ty
  );

  void PopulateAllocaMap_ArrayType(llvm::DICompositeType *Ty
  );

  void PopulateAllocaMap_StructType(
      llvm::DICompositeType *Ty
  );

  llvm::DILocation *GetVariableLocation() const;
  llvm::Value *GetMetadataAsValue(
      llvm::Metadata *M
  ) const;
  llvm::DIExpression *GetDIExpression(
      llvm::DIType *Ty,
      OffsetInBits Offset,
      SizeInBits ParentSize
  ) const;

  llvm::DebugLoc const &m_dbgLoc;
  llvm::DIVariable *m_Variable = nullptr;
  llvm::IRBuilder<> m_B;
  llvm::Function *m_DbgDeclareFn = nullptr;

  OffsetManager m_Offsets;
  std::unordered_map<OffsetInBits, llvm::AllocaInst *> m_AlignedOffsetToAlloca;
};

class DxilDbgValueToDbgDeclare : public llvm::ModulePass {
public:
    static char ID;
    DxilDbgValueToDbgDeclare() : llvm::ModulePass(ID)
    {
    }
  bool runOnModule(
      llvm::Module &M
  ) override;

private:
  void handleDbgValue(
      llvm::Module &M,
      llvm::DbgValueInst *DbgValue);

  std::unordered_map<llvm::DIVariable *, std::unique_ptr<VariableRegisters>> m_Registers;
};
}  // namespace

char DxilDbgValueToDbgDeclare::ID = 0;

struct ValueAndOffset
{
    llvm::Value *m_V;
    OffsetInBits m_PackedOffset;
};

// SplitValue splits an llvm::Value into possibly multiple
// scalar Values. Those scalar values will later be "stored"
// into their corresponding register.
static OffsetInBits SplitValue(
    llvm::Value *V,
    OffsetInBits CurrentOffset,
    std::vector<ValueAndOffset> *Values,
    llvm::IRBuilder<>& B
)
{
  auto *VTy = V->getType();
  if (auto *ArrTy = llvm::dyn_cast<llvm::ArrayType>(VTy))
  {
    for (unsigned i = 0; i < ArrTy->getNumElements(); ++i)
    {
      CurrentOffset = SplitValue(
        B.CreateExtractValue(V, {i}),
        CurrentOffset,
        Values,
        B);
    }
  }
  else if (auto *StTy = llvm::dyn_cast<llvm::StructType>(VTy))
  {
    for (unsigned i = 0; i < StTy->getNumElements(); ++i)
    {
      CurrentOffset = SplitValue(
          B.CreateExtractValue(V, {i}),
          CurrentOffset,
          Values,
          B);
    }
  }
  else if (auto *VecTy = llvm::dyn_cast<llvm::VectorType>(VTy))
  {
    for (unsigned i = 0; i < VecTy->getNumElements(); ++i)
    {
      CurrentOffset = SplitValue(
          B.CreateExtractElement(V, i),
          CurrentOffset,
          Values,
          B);
    }
  }
  else
  {
    assert(VTy->isFloatTy() || VTy->isDoubleTy() || VTy->isHalfTy() ||
           VTy->isIntegerTy(32) || VTy->isIntegerTy(64) || VTy->isIntegerTy(16));
    Values->emplace_back(ValueAndOffset{V, CurrentOffset});
    CurrentOffset += VTy->getScalarSizeInBits();
  }

  return CurrentOffset;
}

// A more convenient version of SplitValue.
static std::vector<ValueAndOffset> SplitValue(
    llvm::Value* V,
    OffsetInBits CurrentOffset,
    llvm::IRBuilder<>& B
)
{
    std::vector<ValueAndOffset> Ret;
    SplitValue(V, CurrentOffset, &Ret, B);
    return Ret;
}

// Convenient helper for parsing a DIExpression's offset.
static OffsetInBits GetAlignedOffsetFromDIExpression(
    llvm::DIExpression *Exp
)
{
  if (!Exp->isBitPiece())
  {
    return 0;
  }

  return Exp->getBitPieceOffset();
}

bool DxilDbgValueToDbgDeclare::runOnModule(
    llvm::Module &M
)
{
  auto *DbgValueFn =
      llvm::Intrinsic::getDeclaration(&M, llvm::Intrinsic::dbg_value);

  bool Changed = false;
  for (auto it = DbgValueFn->user_begin(); it != DbgValueFn->user_end();)
  {
    llvm::User *User = *it++;

    if (auto *DbgValue = llvm::dyn_cast<llvm::DbgValueInst>(User))
    {
      llvm::Value *V = DbgValue->getValue();
      if (PIXPassHelpers::IsAllocateRayQueryInstruction(V)) {
          continue;
      }
      Changed = true;
      handleDbgValue(M, DbgValue);
      DbgValue->eraseFromParent();
    }
  }

  return Changed;
}

static llvm::DIType* FindStructMemberTypeAtOffset(
    llvm::DICompositeType* Ty,
    uint64_t Offset,
    uint64_t Size
);

static llvm::DIType* FindMemberTypeAtOffset(
    llvm::DIType * Ty,
    uint64_t Offset,
    uint64_t Size
) {
  VALUE_TO_DECLARE_LOG("PopulateAllocaMap for type tag %d", Ty->getTag());
  const llvm::DITypeIdentifierMap EmptyMap;
  if (auto *DerivedTy = llvm::dyn_cast<llvm::DIDerivedType>(Ty)) {
    switch (DerivedTy->getTag()) {
    default:
      assert(!"Unhandled DIDerivedType");
      return nullptr;
    case llvm::dwarf::DW_TAG_arg_variable: // "this" pointer
    case llvm::dwarf::DW_TAG_pointer_type: // "this" pointer
        //what to do here?
      return nullptr;
    case llvm::dwarf::DW_TAG_restrict_type:
    case llvm::dwarf::DW_TAG_reference_type:
    case llvm::dwarf::DW_TAG_const_type:
    case llvm::dwarf::DW_TAG_typedef:
      return FindMemberTypeAtOffset(
          DerivedTy->getBaseType().resolve(EmptyMap),
          Offset,
          Size);
    case llvm::dwarf::DW_TAG_member:
      return FindMemberTypeAtOffset(
          DerivedTy->getBaseType().resolve(EmptyMap),
          Offset,
          Size);
    case llvm::dwarf::DW_TAG_subroutine_type:
      // ignore member functions.
      return nullptr;
    }
  } else if (auto *CompositeTy = llvm::dyn_cast<llvm::DICompositeType>(Ty)) {
    switch (CompositeTy->getTag()) {
    default:
      assert(!"Unhandled DICompositeType");
      return nullptr;
    case llvm::dwarf::DW_TAG_array_type:
      return nullptr;
    case llvm::dwarf::DW_TAG_structure_type:
    case llvm::dwarf::DW_TAG_class_type:
        return FindStructMemberTypeAtOffset(
          CompositeTy, Offset, Size);
    case llvm::dwarf::DW_TAG_enumeration_type:
      return nullptr;
    }
  }
  else if (auto* BasicTy = llvm::dyn_cast<llvm::DIBasicType>(Ty)) {
      if (Offset == 0 && Ty->getSizeInBits() == Size) {
          return BasicTy;
      }
  }

  assert(!"Unhandled DIType");
  return nullptr;
}

// SortMembers traverses all of Ty's members and returns them sorted
// by their offset from Ty's start. Returns true if the function succeeds
// and false otherwise.
static bool SortMembers(
    llvm::DICompositeType* Ty,
    std::map<OffsetInBits, llvm::DIDerivedType*>* SortedMembers
)
{
    auto Elements = Ty->getElements();
    if (Elements.begin() == Elements.end())
    {
        return false;
    }
    for (auto* Element : Elements)
    {
        switch (Element->getTag())
        {
        case llvm::dwarf::DW_TAG_member: {
            if (auto* Member = llvm::dyn_cast<llvm::DIDerivedType>(Element))
            {
                if (Member->getSizeInBits()) {
                    auto it = SortedMembers->emplace(std::make_pair(Member->getOffsetInBits(), Member));
                    (void)it;
                    assert(it.second &&
                        "Invalid DIStructType"
                        " - members with the same offset -- are unions possible?");
                }
                break;
            }
            assert(!"member is not a Member");
            return false;
        }
        case llvm::dwarf::DW_TAG_subprogram: {
            if (auto* SubProgram = llvm::dyn_cast<llvm::DISubprogram>(Element)) {
                continue;
            }
            assert(!"DISubprogram not understood");
            return false;
        }
        case llvm::dwarf::DW_TAG_inheritance: {
            if (auto* Member = llvm::dyn_cast<llvm::DIDerivedType>(Element))
            {
                auto it = SortedMembers->emplace(
                    std::make_pair(Member->getOffsetInBits(), Member));
                (void)it;
                assert(it.second &&
                    "Invalid DIStructType"
                    " - members with the same offset -- are unions possible?");
            }
            continue;
        }
        default:
            assert(!"Unhandled field type in DIStructType");
            return false;
        }
    }
    return true;
}

static bool IsResourceObject(llvm::DIDerivedType *DT) {
  const llvm::DITypeIdentifierMap EmptyMap;
  auto *BT = DT->getBaseType().resolve(EmptyMap);
  if (auto *CompositeTy = llvm::dyn_cast<llvm::DICompositeType>(BT)) {
    // Resource variables (e.g. TextureCube) are composite types but have no
    // elements:
    if (CompositeTy->getElements().begin() ==
        CompositeTy->getElements().end()) {
      auto name = CompositeTy->getName();
      auto openTemplateListMarker = name.find_first_of('<');
      if (openTemplateListMarker != llvm::StringRef::npos) {
        auto hlslType = name.substr(0, openTemplateListMarker);
        for (int i = static_cast<int>(hlsl::DXIL::ResourceKind::Invalid) + 1;
             i < static_cast<int>(hlsl::DXIL::ResourceKind::NumEntries); ++i) {
          if (hlslType == hlsl::GetResourceKindName(
                              static_cast<hlsl::DXIL::ResourceKind>(i))) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

static llvm::DIType* FindStructMemberTypeAtOffset(llvm::DICompositeType* Ty,
    uint64_t Offset,
    uint64_t Size) {
    std::map<OffsetInBits, llvm::DIDerivedType*> SortedMembers;
    if (!SortMembers(Ty, &SortedMembers)) {
        return Ty;
    }

    const llvm::DITypeIdentifierMap EmptyMap;

    for (auto & member : SortedMembers) {
        // "Inheritance" is a member of a composite type, but has size of zero.
        // Therefore, we must descend the hierarchy once to find an actual type.
        llvm::DIType * memberType = member.second;
        if (memberType->getTag() == llvm::dwarf::DW_TAG_inheritance) {
          memberType = member.second->getBaseType().resolve(EmptyMap);
        }
        if (Offset >= member.first &&
            Offset < member.first + memberType->getSizeInBits()) {
            uint64_t OffsetIntoThisType = Offset - member.first;
            return FindMemberTypeAtOffset(memberType, OffsetIntoThisType, Size);
        }
    }

    // Structure resources are expected to fail this (they have no real meaning in
    // storage)
    if (SortedMembers.size() == 1) {
        switch (SortedMembers.begin()->second->getTag()) {
        case llvm::dwarf::DW_TAG_structure_type:
        case llvm::dwarf::DW_TAG_class_type:
            if (IsResourceObject(SortedMembers.begin()->second)) {
                return nullptr;
            }
        }
    }
#ifdef VALUE_TO_DECLARE_LOGGING
    VALUE_TO_DECLARE_LOG("Didn't find a member that straddles the sought type. Container:");
    {
        ScopedIndenter indent;
        Ty->dump();
        DumpFullType(Ty);
    }
    VALUE_TO_DECLARE_LOG("Sought type is at offset %d size %d. Members and offsets:", Offset, Size);
    {
        ScopedIndenter indent;
        for (auto const& member : SortedMembers) {
            member.second->dump();
            LogPartialLine("Offset %d (size %d): ", member.first, member.second->getSizeInBits());
            DumpFullType(member.second);
        }
    }
#endif
    assert(!"Didn't find a member that straddles the sought type");
    return nullptr;
}

static bool IsDITypePointer(DIType *DTy, const llvm::DITypeIdentifierMap &EmptyMap) {
  DIDerivedType *DerivedTy = dyn_cast<DIDerivedType>(DTy);
  if (!DerivedTy) return false;
  switch (DerivedTy->getTag()) {
  case llvm::dwarf::DW_TAG_pointer_type:
    return true;
  case llvm::dwarf::DW_TAG_typedef:
  case llvm::dwarf::DW_TAG_const_type:
  case llvm::dwarf::DW_TAG_restrict_type:
  case llvm::dwarf::DW_TAG_reference_type:
    return IsDITypePointer(DerivedTy->getBaseType().resolve(EmptyMap), EmptyMap);
  }
  return false;
}

void DxilDbgValueToDbgDeclare::handleDbgValue(
    llvm::Module& M,
    llvm::DbgValueInst* DbgValue)
{
  VALUE_TO_DECLARE_LOG("DbgValue named %s", DbgValue->getName().str().c_str());

  llvm::DIVariable *Variable = DbgValue->getVariable();
  if (Variable != nullptr) {
    VALUE_TO_DECLARE_LOG("... DbgValue referred to variable named %s",
        Variable->getName().str().c_str());
  } else {
    VALUE_TO_DECLARE_LOG("... variable was null too");
  }

  llvm::Value *V = DbgValue->getValue();
  if (V == nullptr) {
    // The metadata contained a null Value, so we ignore it. This
    // seems to be a dxcompiler bug.
    VALUE_TO_DECLARE_LOG("...Null value!");
    return;
  }
  
  const llvm::DITypeIdentifierMap EmptyMap;
  llvm::DIType *Ty = Variable->getType().resolve(EmptyMap);
  if (Ty == nullptr) {
    return;
  }

  if (llvm::isa<llvm::PointerType>(V->getType())) {
    // Safeguard: If the type is not a pointer type, then this is
    // dbg.value directly pointing to a memory location instead of
    // a value.
    if (!IsDITypePointer(Ty, EmptyMap)) {
      // We only know how to handle AllocaInsts for now
      if (!isa<AllocaInst>(V)) {
        VALUE_TO_DECLARE_LOG("... variable had pointer type, but is not an alloca.");
        return;
      }

      IRBuilder<> B(DbgValue->getNextNode());
      V = B.CreateLoad(V);
    }
  }

  // Members' "base type" is actually the containing aggregate's type.
  // To find the actual type of the variable, we must descend the container's 
  // type hierarchy to find the type at the expected offset/size.
  if (auto* DerivedTy = llvm::dyn_cast<llvm::DIDerivedType>(Ty)) {
      const llvm::DITypeIdentifierMap EmptyMap;
      switch (DerivedTy->getTag()) {
      case llvm::dwarf::DW_TAG_member: {
          Ty = FindMemberTypeAtOffset(
              DerivedTy->getBaseType().resolve(EmptyMap), 
              DerivedTy->getOffsetInBits(), 
              DerivedTy->getSizeInBits());
        if (Ty == nullptr) {
            return;
        }
      }
      break;
      }
  }

  auto &Register = m_Registers[Variable];
  if (Register == nullptr)
  {
    Register.reset(
        new VariableRegisters(DbgValue, Variable, Ty, &M));
  }

  // Convert the offset from DbgValue's expression to a packed
  // offset, which we'll need in order to determine the (packed)
  // offset of each scalar Value in DbgValue.
  llvm::DIExpression* expression = DbgValue->getExpression();
  const OffsetInBits AlignedOffsetFromVar =
      GetAlignedOffsetFromDIExpression(expression);
  OffsetInBits PackedOffsetFromVar;
  const OffsetManager& Offsets = Register->GetOffsetManager();
  if (!Offsets.GetPackedOffsetFromAlignedOffset(AlignedOffsetFromVar,
                                                &PackedOffsetFromVar))
  {
    // todo: output geometry for GS
    return;
  }

  const OffsetInBits InitialOffset = PackedOffsetFromVar;
  auto* insertPt = llvm::dyn_cast<llvm::Instruction>(V);
  if (insertPt != nullptr && !llvm::isa<TerminatorInst>(insertPt)) {
    insertPt = insertPt->getNextNode();
    // Drivers may crash if phi nodes aren't always at the top of a block,
    // so we must skip over them before inserting instructions.
    while (llvm::isa<llvm::PHINode>(insertPt)) {
      insertPt = insertPt->getNextNode();
    }

    if (insertPt != nullptr) {
      llvm::IRBuilder<> B(insertPt);
      B.SetCurrentDebugLocation(llvm::DebugLoc());

      auto *Zero = B.getInt32(0);
      
      // Now traverse a list of pairs {Scalar Value, InitialOffset + Offset}.
      // InitialOffset is the offset from DbgValue's expression (i.e., the
      // offset from the Variable's start), and Offset is the Scalar Value's
      // packed offset from DbgValue's value.
      for (const ValueAndOffset &VO : SplitValue(V, InitialOffset, B)) {

        OffsetInBits AlignedOffset;
        if (!Offsets.GetAlignedOffsetFromPackedOffset(VO.m_PackedOffset,
                                                      &AlignedOffset)) {
          continue;
        }

        auto *AllocaInst = Register->GetRegisterForAlignedOffset(AlignedOffset);
        if (AllocaInst == nullptr) {
          assert(!"Failed to find alloca for var[offset]");
          continue;
        }

        if (AllocaInst->getAllocatedType()->getArrayElementType() ==
            VO.m_V->getType()) {
          auto *GEP = B.CreateGEP(AllocaInst, {Zero, Zero});
          B.CreateStore(VO.m_V, GEP);
        }
      }
    }
  }
}

SizeInBits VariableRegisters::GetVariableSizeInbits(DIVariable *Var) {
  const llvm::DITypeIdentifierMap EmptyMap;
  DIType *Ty = Var->getType().resolve(EmptyMap);
  DIDerivedType *DerivedTy = nullptr;
  while (Ty && (Ty->getSizeInBits() == 0 && (DerivedTy = dyn_cast<DIDerivedType>(Ty)))) {
    Ty = DerivedTy->getBaseType().resolve(EmptyMap);
  }

  if (!Ty) {
    assert(false && "Unexpected inability to resolve base type with a real size.");
    return 0;
  }
  return Ty->getSizeInBits();
}

llvm::AllocaInst *VariableRegisters::GetRegisterForAlignedOffset(
    OffsetInBits Offset
) const
{
  auto it = m_AlignedOffsetToAlloca.find(Offset);
  if (it == m_AlignedOffsetToAlloca.end())
  {
    return nullptr;
  }
  return it->second;
}

#ifndef NDEBUG
// DITypePeelTypeAlias peels const, typedef, and other alias types off of Ty,
// returning the unalised type.
static llvm::DIType *DITypePeelTypeAlias(
    llvm::DIType* Ty
)
{
  if (auto *DerivedTy = llvm::dyn_cast<llvm::DIDerivedType>(Ty))
  {
    const llvm::DITypeIdentifierMap EmptyMap;
    switch (DerivedTy->getTag())
    {
    case llvm::dwarf::DW_TAG_restrict_type:
    case llvm::dwarf::DW_TAG_reference_type:
    case llvm::dwarf::DW_TAG_const_type:
    case llvm::dwarf::DW_TAG_typedef:
    case llvm::dwarf::DW_TAG_pointer_type:
      return DITypePeelTypeAlias(
          DerivedTy->getBaseType().resolve(EmptyMap));
    case llvm::dwarf::DW_TAG_member:
      return DITypePeelTypeAlias(
          DerivedTy->getBaseType().resolve(EmptyMap));
    }
  }

  return Ty;
}
#endif // NDEBUG

VariableRegisters::VariableRegisters(
    llvm::DbgValueInst *DbgValue,
    llvm::DIVariable *Variable,
    llvm::DIType* Ty,
    llvm::Module *M)
  : m_dbgLoc(DbgValue->getDebugLoc())
  , m_Variable(Variable)
  , m_B(DbgValue->getParent()->getParent()->getEntryBlock().begin())
  , m_DbgDeclareFn(llvm::Intrinsic::getDeclaration(
      M, llvm::Intrinsic::dbg_declare))
{
  PopulateAllocaMap(Ty);
  m_Offsets.AlignTo(Ty); // For padding.
  assert(m_Offsets.GetCurrentAlignedOffset() ==
         DITypePeelTypeAlias(Ty)->getSizeInBits());
}

void VariableRegisters::PopulateAllocaMap(
    llvm::DIType *Ty
)
{
  VALUE_TO_DECLARE_LOG("PopulateAllocaMap for type tag %d", Ty->getTag());
  const llvm::DITypeIdentifierMap EmptyMap;
  if (auto *DerivedTy = llvm::dyn_cast<llvm::DIDerivedType>(Ty))
  {
    switch (DerivedTy->getTag())
    {
    default:
      assert(!"Unhandled DIDerivedType");
      m_Offsets.AlignToAndAddUnhandledType(DerivedTy);
      return;
    case llvm::dwarf::DW_TAG_arg_variable: // "this" pointer
    case llvm::dwarf::DW_TAG_pointer_type: // "this" pointer
    case llvm::dwarf::DW_TAG_restrict_type:
    case llvm::dwarf::DW_TAG_reference_type:
    case llvm::dwarf::DW_TAG_const_type:
    case llvm::dwarf::DW_TAG_typedef:
      PopulateAllocaMap(
          DerivedTy->getBaseType().resolve(EmptyMap));
      return;
    case llvm::dwarf::DW_TAG_member:
      PopulateAllocaMap(
          DerivedTy->getBaseType().resolve(EmptyMap));
      return;
    case llvm::dwarf::DW_TAG_subroutine_type:
        //ignore member functions.
      return;
    }
  }
  else if (auto *CompositeTy = llvm::dyn_cast<llvm::DICompositeType>(Ty))
  {
    switch (CompositeTy->getTag())
    {
    default:
      assert(!"Unhandled DICompositeType");
      m_Offsets.AlignToAndAddUnhandledType(CompositeTy);
      return;
    case llvm::dwarf::DW_TAG_array_type:
      PopulateAllocaMap_ArrayType(CompositeTy);
      return;
    case llvm::dwarf::DW_TAG_structure_type:
    case llvm::dwarf::DW_TAG_class_type:
      PopulateAllocaMap_StructType(CompositeTy);
      return;
    case llvm::dwarf::DW_TAG_enumeration_type: {
      auto * baseType = CompositeTy->getBaseType().resolve(EmptyMap);
      if (baseType != nullptr) {
        PopulateAllocaMap(baseType);
      } else {
        m_Offsets.AlignToAndAddUnhandledType(CompositeTy);
      }
    }
      return;
    }
  }
  else if (auto *BasicTy = llvm::dyn_cast<llvm::DIBasicType>(Ty))
  {
    PopulateAllocaMap_BasicType(BasicTy);
    return;
  }

  assert(!"Unhandled DIType");
  m_Offsets.AlignToAndAddUnhandledType(Ty);
}

static llvm::Type* GetLLVMTypeFromDIBasicType(
    llvm::IRBuilder<> &B,
    llvm::DIBasicType* Ty
)
{
  const SizeInBits Size = Ty->getSizeInBits();

  switch (Ty->getEncoding())
  {
  default:
    break;

  case llvm::dwarf::DW_ATE_boolean:
  case llvm::dwarf::DW_ATE_signed:
  case llvm::dwarf::DW_ATE_unsigned:
    switch(Size)
    {
    case 16:
      return B.getInt16Ty();
    case 32:
      return B.getInt32Ty();
    case 64:
      return B.getInt64Ty();
    }
    break;
  case llvm::dwarf::DW_ATE_float:
    switch(Size)
    {
    case 16:
      return B.getHalfTy();
    case 32:
      return B.getFloatTy();
    case 64:
      return B.getDoubleTy();
    }
    break;
  }

  return nullptr;
}

void VariableRegisters::PopulateAllocaMap_BasicType(
    llvm::DIBasicType *Ty
)
{
  llvm::Type* AllocaElementTy = GetLLVMTypeFromDIBasicType(m_B, Ty);
  assert(AllocaElementTy != nullptr);
  if (AllocaElementTy == nullptr)
  {
      return;
  }

  const OffsetInBits AlignedOffset = m_Offsets.Add(Ty);

  llvm::Type *AllocaTy = llvm::ArrayType::get(AllocaElementTy, 1);
  llvm::AllocaInst *&Alloca = m_AlignedOffsetToAlloca[AlignedOffset];
  Alloca = m_B.CreateAlloca(AllocaTy, m_B.getInt32(0));
  Alloca->setDebugLoc(llvm::DebugLoc());

  auto *Storage = GetMetadataAsValue(llvm::ValueAsMetadata::get(Alloca));
  auto *Variable = GetMetadataAsValue(m_Variable);
  auto *Expression = GetMetadataAsValue(GetDIExpression(Ty, AlignedOffset, GetVariableSizeInbits(m_Variable)));
  auto *DbgDeclare = m_B.CreateCall(
      m_DbgDeclareFn,
      {Storage, Variable, Expression});
  DbgDeclare->setDebugLoc(m_dbgLoc);
}

static unsigned NumArrayElements(
    llvm::DICompositeType *Array
)
{
  if (Array->getElements().size() == 0)
  {
    return 0;
  }

  unsigned NumElements = 1;
  for (llvm::DINode *N : Array->getElements())
  {
    if (auto* Subrange = llvm::dyn_cast<llvm::DISubrange>(N))
    {
      NumElements *= Subrange->getCount();
    }
    else
    {
      assert(!"Unhandled array element");
      return 0;
    }
  }
  return NumElements;
}

void VariableRegisters::PopulateAllocaMap_ArrayType(
    llvm::DICompositeType* Ty
)
{
  unsigned NumElements = NumArrayElements(Ty);
  if (NumElements == 0)
  {
    m_Offsets.AlignToAndAddUnhandledType(Ty);
    return;
  }

  const SizeInBits ArraySizeInBits = Ty->getSizeInBits();
  (void)ArraySizeInBits;

  const llvm::DITypeIdentifierMap EmptyMap;
  llvm::DIType *ElementTy = Ty->getBaseType().resolve(EmptyMap);
  assert(ArraySizeInBits % NumElements == 0 &&
         " invalid DIArrayType"
         " - Size is not a multiple of NumElements");

  // After aligning the current aligned offset to ElementTy's natural
  // alignment, the current aligned offset must match Ty's offset
  // in bits.
  m_Offsets.AlignTo(ElementTy);

  for (unsigned i = 0; i < NumElements; ++i)
  {
    // This is only needed if ElementTy's size is not a multiple of
    // its natural alignment.
    m_Offsets.AlignTo(ElementTy);
    PopulateAllocaMap(
        ElementTy);
  }
}

void VariableRegisters::PopulateAllocaMap_StructType(
    llvm::DICompositeType *Ty
)
{
  VALUE_TO_DECLARE_LOG("Struct type : %s, size %d", Ty->getName().str().c_str(), Ty->getSizeInBits());
  std::map<OffsetInBits, llvm::DIDerivedType *> SortedMembers;
  if (!SortMembers(Ty, &SortedMembers))
  {
      m_Offsets.AlignToAndAddUnhandledType(Ty);
      return;
  }

  m_Offsets.AlignTo(Ty);
  const OffsetInBits StructStart = m_Offsets.GetCurrentAlignedOffset();
  (void)StructStart;
  const llvm::DITypeIdentifierMap EmptyMap;

  for (auto OffsetAndMember : SortedMembers)
  {
    VALUE_TO_DECLARE_LOG("Member: %s at aligned offset %d", OffsetAndMember.second->getName().str().c_str(), OffsetAndMember.first);
    // Align the offsets to the member's type natural alignment. This
    // should always result in the current aligned offset being the
    // same as the member's offset.
    m_Offsets.AlignTo(OffsetAndMember.second);
    assert(m_Offsets.GetCurrentAlignedOffset() ==
        StructStart + OffsetAndMember.first &&
        "Offset mismatch in DIStructType");
    if (IsResourceObject(OffsetAndMember.second)) {
      m_Offsets.AddResourceType(OffsetAndMember.second);
    } else {
      PopulateAllocaMap(
          OffsetAndMember.second->getBaseType().resolve(EmptyMap));
    }
  }
}

llvm::DILocation *VariableRegisters::GetVariableLocation() const
{
  const unsigned DefaultColumn = 1;
  return llvm::DILocation::get(
      m_B.getContext(),
      m_Variable->getLine(),
      DefaultColumn,
      m_Variable->getScope());
}

llvm::Value *VariableRegisters::GetMetadataAsValue(
    llvm::Metadata *M
) const
{
  return llvm::MetadataAsValue::get(m_B.getContext(), M);
}

llvm::DIExpression *VariableRegisters::GetDIExpression(
    llvm::DIType *Ty,
    OffsetInBits Offset,
    SizeInBits ParentSize
) const
{
  llvm::SmallVector<uint64_t, 3> ExpElements;
  if (Offset != 0 || Ty->getSizeInBits() != ParentSize)
  {
    ExpElements.emplace_back(llvm::dwarf::DW_OP_bit_piece);
    ExpElements.emplace_back(Offset);
    ExpElements.emplace_back(Ty->getSizeInBits());
  }
  return llvm::DIExpression::get(m_B.getContext(), ExpElements);
}

using namespace llvm;

INITIALIZE_PASS(DxilDbgValueToDbgDeclare, DEBUG_TYPE,
                "Converts calls to dbg.value to dbg.declare + stores to new virtual registers",
                false, false)

ModulePass *llvm::createDxilDbgValueToDbgDeclarePass() {
  return new DxilDbgValueToDbgDeclare();
}
