///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxcPixLiveVariables_FragmentIterator.cpp                                  //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Implements the FragmentIterator.                                          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/Support/WinIncludes.h"

#include "DxcPixLiveVariables_FragmentIterator.h"

#include "dxc/DXIL/DxilMetadataHelper.h"
#include "dxc/Support/exception.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include <vector>

///////////////////////////////////////////////////////////////////////////////
class FragmentIteratorBase : public dxil_debug_info::MemberIterator
{
public:
  virtual ~FragmentIteratorBase() {}
  virtual unsigned SizeInBits(unsigned ) const override;
  virtual bool Next(unsigned *FragmentIndex) override;

protected:
  FragmentIteratorBase(unsigned NumFragments, unsigned FragmentSizeInBits,
    unsigned InitialOffsetInBits);

  unsigned m_CurrFragment = 0;
  unsigned m_NumFragments = 0;
  unsigned m_FragmentSizeInBits = 0;
  unsigned m_InitialOffsetInBits = 0;
};

unsigned FragmentIteratorBase::SizeInBits(unsigned ) const
{
  return m_FragmentSizeInBits;
}

bool FragmentIteratorBase::Next(unsigned *FragmentIndex) {
  if (m_CurrFragment >= m_NumFragments) {
    return false;
  }

  *FragmentIndex = m_CurrFragment++;
  return true;
}

FragmentIteratorBase::FragmentIteratorBase(
    unsigned NumFragments, unsigned FragmentSizeInBits,
    unsigned InitialOffsetInBits)
    : m_NumFragments(NumFragments), m_FragmentSizeInBits(FragmentSizeInBits),
      m_InitialOffsetInBits(InitialOffsetInBits) {}


///////////////////////////////////////////////////////////////////////////////
static unsigned NumAllocaElements(llvm::AllocaInst *Alloca) {
  llvm::Type *FragmentTy = Alloca->getAllocatedType();
  if (auto *ArrayTy = llvm::dyn_cast<llvm::ArrayType>(FragmentTy)) {
    return ArrayTy->getNumElements();
  }

  const unsigned NumElements = 1;
  return NumElements;
}

static unsigned FragmentSizeInBitsFromAlloca(const llvm::DataLayout &DataLayout,
                                             llvm::AllocaInst *Alloca) {
  llvm::Type *FragmentTy = Alloca->getAllocatedType();
  if (auto *ArrayTy = llvm::dyn_cast<llvm::ArrayType>(FragmentTy)) {
    FragmentTy = ArrayTy->getElementType();
  }

  const unsigned FragmentSizeInBits =
      DataLayout.getTypeAllocSizeInBits(FragmentTy);

  return FragmentSizeInBits;
}

static unsigned
InitialOffsetInBitsFromDIExpression(const llvm::DataLayout &DataLayout,
                                    llvm::AllocaInst *Alloca,
                                    llvm::DIExpression *Expression) {
  unsigned FragmentOffsetInBits = 0;
  if (Expression->getNumElements() > 0) {
    if (Expression->getNumElements() == 1 &&
      Expression->expr_op_begin()->getOp() == llvm::dwarf::DW_OP_deref) {
      return 0;
    }
    else if (!Expression->isBitPiece()) {
      assert(!"Unhandled DIExpression");
      throw hlsl::Exception(E_FAIL, "Unhandled DIExpression");
    }

    FragmentOffsetInBits = Expression->getBitPieceOffset();
    assert(Expression->getBitPieceSize() ==
           DataLayout.getTypeAllocSizeInBits(Alloca->getAllocatedType()));
  }

  return FragmentOffsetInBits;
}

///////////////////////////////////////////////////////////////////////////////
class DILayoutFragmentIterator : public FragmentIteratorBase {
public:
  DILayoutFragmentIterator(const llvm::DataLayout &DataLayout,
                           llvm::AllocaInst *Alloca,
                           llvm::DIExpression *Expression);

  virtual unsigned OffsetInBits(unsigned Index) override;
};

DILayoutFragmentIterator::DILayoutFragmentIterator(
    const llvm::DataLayout &DataLayout, llvm::AllocaInst *Alloca,
    llvm::DIExpression *Expression)
    : FragmentIteratorBase(NumAllocaElements(Alloca),
                       FragmentSizeInBitsFromAlloca(DataLayout, Alloca),
                       InitialOffsetInBitsFromDIExpression(DataLayout, Alloca,
                                                           Expression)) {}

unsigned DILayoutFragmentIterator::OffsetInBits(unsigned Index) {
  return m_InitialOffsetInBits + Index * m_FragmentSizeInBits;
}

///////////////////////////////////////////////////////////////////////////////
static unsigned
NumFragmentsFromArrayDims(const std::vector<hlsl::DxilDIArrayDim> &ArrayDims) {
  unsigned TotalNumFragments = 1;
  for (const hlsl::DxilDIArrayDim &ArrayDim : ArrayDims) {
    TotalNumFragments *= ArrayDim.NumElements;
  }
  return TotalNumFragments;
}

static unsigned FragmentSizeInBitsFrom(const llvm::DataLayout &DataLayout,
                                       llvm::AllocaInst *Alloca,
                                       unsigned TotalNumFragments) {
  const unsigned TotalSizeInBits =
      DataLayout.getTypeAllocSizeInBits(Alloca->getAllocatedType());

  if (TotalNumFragments == 0 || TotalSizeInBits % TotalNumFragments != 0) {
    assert(!"Malformed variable debug layout metadata.");
    throw hlsl::Exception(E_FAIL, "Malformed variable debug layout metadata.");
  }
  const unsigned FragmentSizeInBits = TotalSizeInBits / TotalNumFragments;
  return FragmentSizeInBits;
}

///////////////////////////////////////////////////////////////////////////////
class DebugLayoutFragmentIterator : public FragmentIteratorBase {
public:
  DebugLayoutFragmentIterator(
      const llvm::DataLayout &DataLayout, llvm::AllocaInst *Alloca,
      unsigned InitialOffsetInBits,
      const std::vector<hlsl::DxilDIArrayDim> &ArrayDims);

  virtual unsigned OffsetInBits(unsigned Index) override;

private:
  std::vector<hlsl::DxilDIArrayDim> m_ArrayDims;
};

DebugLayoutFragmentIterator::DebugLayoutFragmentIterator(
    const llvm::DataLayout &DataLayout, llvm::AllocaInst *Alloca,
    unsigned InitialOffsetInBits,
    const std::vector<hlsl::DxilDIArrayDim> &ArrayDims)
    : FragmentIteratorBase(
          NumFragmentsFromArrayDims(ArrayDims),
          FragmentSizeInBitsFrom(DataLayout, Alloca,
                                 NumFragmentsFromArrayDims(ArrayDims)),
          InitialOffsetInBits),
      m_ArrayDims(ArrayDims) {}

unsigned DebugLayoutFragmentIterator::OffsetInBits(unsigned Index) {
  // Figure out the offset of this fragment in the original
  unsigned FragmentOffsetInBits = m_InitialOffsetInBits;
  unsigned RemainingIndex = Index;
  for (const hlsl::DxilDIArrayDim &ArrayDim : m_ArrayDims) {
    FragmentOffsetInBits +=
        (RemainingIndex % ArrayDim.NumElements) * ArrayDim.StrideInBits;
    RemainingIndex /= ArrayDim.NumElements;
  }
  assert(RemainingIndex == 0);
  return FragmentOffsetInBits;
}


///////////////////////////////////////////////////////////////////////////////
class CompositeTypeFragmentIterator : public dxil_debug_info::MemberIterator {
public:
  CompositeTypeFragmentIterator(llvm::DICompositeType* CT);

  virtual unsigned SizeInBits(unsigned Index) const override;
  virtual unsigned OffsetInBits(unsigned Index) override;
  virtual bool Next(unsigned* FragmentIndex) override;

private:
  struct FragmentSizeAndOffset
  {
    unsigned Size;
    unsigned Offset;
  };
  std::vector<FragmentSizeAndOffset> m_fragmentLocations;
  unsigned m_currentFragment = 0;
  void CompositeTypeFragmentIterator::DetermineStructMemberSizesAndOffsets(
    llvm::DIType const*, uint64_t BaseOffset);
};

unsigned SizeIfBaseType(llvm::DIType const* diType)
{
  if (auto* BT = llvm::dyn_cast<llvm::DIBasicType>(diType))
  {
    return BT->getSizeInBits();
  }
  if (auto* DT = llvm::dyn_cast<llvm::DIDerivedType>(diType))
  {
    const llvm::DITypeIdentifierMap EmptyMap;
    return SizeIfBaseType(DT->getBaseType().resolve(EmptyMap));
  }
  return 0;
}

void CompositeTypeFragmentIterator::DetermineStructMemberSizesAndOffsets(llvm::DIType const*diType, uint64_t BaseOffset)
{
  assert(diType->getTag() != llvm::dwarf::DW_TAG_subroutine_type);
  if (diType->getTag() == llvm::dwarf::DW_TAG_member)
  {
    BaseOffset += diType->getOffsetInBits();
  }
  unsigned MemberSize = SizeIfBaseType(diType);
  if (MemberSize != 0)
  {
    m_fragmentLocations.push_back({ MemberSize, static_cast<unsigned>(BaseOffset) });
  }
  else
  {
    if (auto* CT = llvm::dyn_cast<llvm::DICompositeType>(diType))
    {

      switch (diType->getTag())
      {
      case llvm::dwarf::DW_TAG_array_type:
      {
        llvm::DINodeArray elements = CT->getElements();
        unsigned arraySize = 1;
        for (auto const& node : elements)
        {
          if (llvm::DISubrange* SR = llvm::dyn_cast<llvm::DISubrange>(node))
          {
            arraySize *= SR->getCount();
          }
        }
        if (arraySize != 0)
        {
          const llvm::DITypeIdentifierMap EmptyMap;
          llvm::DIType* BT = CT->getBaseType().resolve(EmptyMap);
          unsigned elementSize = static_cast<unsigned>(CT->getSizeInBits()) / arraySize;
          for (unsigned i = 0; i < arraySize; ++i) {
            DetermineStructMemberSizesAndOffsets(BT, BaseOffset + i * elementSize);
          }
        }
      }
      break;
      case llvm::dwarf::DW_TAG_class_type:
      case llvm::dwarf::DW_TAG_structure_type:
        for (auto const& node : CT->getElements())
        {
          if (llvm::DIType* subType = llvm::dyn_cast<llvm::DIType>(node))
          {
            DetermineStructMemberSizesAndOffsets(subType, BaseOffset /*TODO: plus member offset*/);
          }
        }
        break;
      default:
        diType->dump();
        break;
      }
    }
    else if (auto* DT = llvm::dyn_cast<llvm::DIDerivedType>(diType))
    {
      const llvm::DITypeIdentifierMap EmptyMap;
      llvm::DIType* BT = DT->getBaseType().resolve(EmptyMap);
      DetermineStructMemberSizesAndOffsets(BT, BaseOffset);
    }
  }
}

CompositeTypeFragmentIterator::CompositeTypeFragmentIterator(
  llvm::DICompositeType* CT)
{
  DetermineStructMemberSizesAndOffsets(CT, 0);
}

unsigned CompositeTypeFragmentIterator::SizeInBits(unsigned Index) const
{
  return m_fragmentLocations[Index].Size;
}

unsigned CompositeTypeFragmentIterator::OffsetInBits(unsigned Index)
{
  return m_fragmentLocations[Index].Offset;
}

bool CompositeTypeFragmentIterator::Next(
    unsigned *FragmentIndex) 
{
  *FragmentIndex = m_currentFragment;
  m_currentFragment++;
  return m_currentFragment <= static_cast<unsigned>(m_fragmentLocations.size());
}

///////////////////////////////////////////////////////////////////////////////
std::unique_ptr<dxil_debug_info::MemberIterator>
dxil_debug_info::CreateMemberIterator(llvm::DbgDeclareInst *DbgDeclare,
                                          const llvm::DataLayout &DataLayout,
                                          llvm::AllocaInst *Alloca,
                                          llvm::DIExpression *Expression) {
  bool HasVariableDebugLayout = false;
  unsigned FirstFragmentOffsetInBits;
  std::vector<hlsl::DxilDIArrayDim> ArrayDims;

  std::unique_ptr<dxil_debug_info::MemberIterator> Iter;

  try {
    HasVariableDebugLayout = hlsl::DxilMDHelper::GetVariableDebugLayout(
        DbgDeclare, FirstFragmentOffsetInBits, ArrayDims);

    if (HasVariableDebugLayout) {
      Iter.reset(new DebugLayoutFragmentIterator(
          DataLayout, Alloca, FirstFragmentOffsetInBits, ArrayDims));
    } else {
      llvm::DICompositeType *CT = llvm::dyn_cast<llvm::DICompositeType>(
          DbgDeclare->getVariable()->getType());
      if (CT != nullptr && Expression->getNumElements() == 0 ) {
          if (CT->getTag() != llvm::dwarf::DW_TAG_subroutine_type) {
              Iter.reset(new CompositeTypeFragmentIterator(CT));
          }
      } else {
        Iter.reset(
            new DILayoutFragmentIterator(DataLayout, Alloca, Expression));
      }
    }
  } catch (const hlsl::Exception &) {
    return nullptr;
  }

  return Iter;
}
