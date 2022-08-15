
#include "llvm/IR/Operator.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"

#include "ConstantsContext.h"

namespace llvm {
Type *GEPOperator::getSourceElementType() const {
  if (auto *I = dyn_cast<GetElementPtrInst>(this))
    return I->getSourceElementType();
  return cast<GetElementPtrConstantExpr>(this)->getSourceElementType();
}

bool GEPOperator::accumulateConstantOffset(const DataLayout &DL,
                                           APInt &Offset) const {
  assert(Offset.getBitWidth() ==
             DL.getPointerSizeInBits(getPointerAddressSpace()) &&
         "The offset must have exactly as many bits as our pointer.");

  for (gep_type_iterator GTI = gep_type_begin(this), GTE = gep_type_end(this);
       GTI != GTE; ++GTI) {
    ConstantInt *OpC = dyn_cast<ConstantInt>(GTI.getOperand());
    if (!OpC)
      return false;
    if (OpC->isZero())
      continue;

    // Handle a struct index, which adds its field offset to the pointer.
    if (StructType *STy = dyn_cast<StructType>(*GTI)) {
      unsigned ElementIdx = OpC->getZExtValue();
      const StructLayout *SL = DL.getStructLayout(STy);
      Offset += APInt(Offset.getBitWidth(), SL->getElementOffset(ElementIdx));
      continue;
    }

    // For array or vector indices, scale the index by the size of the type.
    APInt Index = OpC->getValue().sextOrTrunc(Offset.getBitWidth());
    Offset += Index * APInt(Offset.getBitWidth(),
                            DL.getTypeAllocSize(GTI.getIndexedType()));
  }
  return true;
}
}
