//===- TargetTransformInfoImpl.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides helpers for the implementation of
/// a TargetTransformInfo-conforming class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_TARGETTRANSFORMINFOIMPL_H
#define LLVM_ANALYSIS_TARGETTRANSFORMINFOIMPL_H

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"

namespace llvm {

/// \brief Base class for use as a mix-in that aids implementing
/// a TargetTransformInfo-compatible class.
class TargetTransformInfoImplBase {
protected:
  typedef TargetTransformInfo TTI;

  const DataLayout &DL;

  explicit TargetTransformInfoImplBase(const DataLayout &DL) : DL(DL) {}

public:
  // Provide value semantics. MSVC requires that we spell all of these out.
  TargetTransformInfoImplBase(const TargetTransformInfoImplBase &Arg)
      : DL(Arg.DL) {}
  TargetTransformInfoImplBase(TargetTransformInfoImplBase &&Arg) : DL(Arg.DL) {}

  const DataLayout &getDataLayout() const { return DL; }

  unsigned getOperationCost(unsigned Opcode, Type *Ty, Type *OpTy) {
    switch (Opcode) {
    default:
      // By default, just classify everything as 'basic'.
      return TTI::TCC_Basic;

    case Instruction::GetElementPtr:
      llvm_unreachable("Use getGEPCost for GEP operations!");

    case Instruction::BitCast:
      assert(OpTy && "Cast instructions must provide the operand type");
      if (Ty == OpTy || (Ty->isPointerTy() && OpTy->isPointerTy()))
        // Identity and pointer-to-pointer casts are free.
        return TTI::TCC_Free;

      // Otherwise, the default basic cost is used.
      return TTI::TCC_Basic;

    case Instruction::IntToPtr: {
      // An inttoptr cast is free so long as the input is a legal integer type
      // which doesn't contain values outside the range of a pointer.
      unsigned OpSize = OpTy->getScalarSizeInBits();
      if (DL.isLegalInteger(OpSize) &&
          OpSize <= DL.getPointerTypeSizeInBits(Ty))
        return TTI::TCC_Free;

      // Otherwise it's not a no-op.
      return TTI::TCC_Basic;
    }
    case Instruction::PtrToInt: {
      // A ptrtoint cast is free so long as the result is large enough to store
      // the pointer, and a legal integer type.
      unsigned DestSize = Ty->getScalarSizeInBits();
      if (DL.isLegalInteger(DestSize) &&
          DestSize >= DL.getPointerTypeSizeInBits(OpTy))
        return TTI::TCC_Free;

      // Otherwise it's not a no-op.
      return TTI::TCC_Basic;
    }
    case Instruction::Trunc:
      // trunc to a native type is free (assuming the target has compare and
      // shift-right of the same width).
      if (DL.isLegalInteger(DL.getTypeSizeInBits(Ty)))
        return TTI::TCC_Free;

      return TTI::TCC_Basic;
    }
  }

  unsigned getGEPCost(const Value *Ptr, ArrayRef<const Value *> Operands) {
    // In the basic model, we just assume that all-constant GEPs will be folded
    // into their uses via addressing modes.
    for (unsigned Idx = 0, Size = Operands.size(); Idx != Size; ++Idx)
      if (!isa<Constant>(Operands[Idx]))
        return TTI::TCC_Basic;

    return TTI::TCC_Free;
  }

  unsigned getCallCost(FunctionType *FTy, int NumArgs) {
    assert(FTy && "FunctionType must be provided to this routine.");

    // The target-independent implementation just measures the size of the
    // function by approximating that each argument will take on average one
    // instruction to prepare.

    if (NumArgs < 0)
      // Set the argument number to the number of explicit arguments in the
      // function.
      NumArgs = FTy->getNumParams();

    return TTI::TCC_Basic * (NumArgs + 1);
  }

  unsigned getIntrinsicCost(Intrinsic::ID IID, Type *RetTy,
                            ArrayRef<Type *> ParamTys) {
    switch (IID) {
    default:
      // Intrinsics rarely (if ever) have normal argument setup constraints.
      // Model them as having a basic instruction cost.
      // FIXME: This is wrong for libc intrinsics.
      return TTI::TCC_Basic;

    case Intrinsic::annotation:
    case Intrinsic::assume:
    case Intrinsic::dbg_declare:
    case Intrinsic::dbg_value:
    case Intrinsic::invariant_start:
    case Intrinsic::invariant_end:
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
    case Intrinsic::objectsize:
    case Intrinsic::ptr_annotation:
    case Intrinsic::var_annotation:
    case Intrinsic::experimental_gc_result_int:
    case Intrinsic::experimental_gc_result_float:
    case Intrinsic::experimental_gc_result_ptr:
    case Intrinsic::experimental_gc_result:
    case Intrinsic::experimental_gc_relocate:
      // These intrinsics don't actually represent code after lowering.
      return TTI::TCC_Free;
    }
  }

  bool hasBranchDivergence() { return false; }

  bool isSourceOfDivergence(const Value *V) { return false; }

  bool isLoweredToCall(const Function *F) {
    // FIXME: These should almost certainly not be handled here, and instead
    // handled with the help of TLI or the target itself. This was largely
    // ported from existing analysis heuristics here so that such refactorings
    // can take place in the future.

    if (F->isIntrinsic())
      return false;

    if (F->hasLocalLinkage() || !F->hasName())
      return true;

    StringRef Name = F->getName();

    // These will all likely lower to a single selection DAG node.
    if (Name == "copysign" || Name == "copysignf" || Name == "copysignl" ||
        Name == "fabs" || Name == "fabsf" || Name == "fabsl" || Name == "sin" ||
        Name == "fmin" || Name == "fminf" || Name == "fminl" ||
        Name == "fmax" || Name == "fmaxf" || Name == "fmaxl" ||
        Name == "sinf" || Name == "sinl" || Name == "cos" || Name == "cosf" ||
        Name == "cosl" || Name == "sqrt" || Name == "sqrtf" || Name == "sqrtl")
      return false;

    // These are all likely to be optimized into something smaller.
    if (Name == "pow" || Name == "powf" || Name == "powl" || Name == "exp2" ||
        Name == "exp2l" || Name == "exp2f" || Name == "floor" ||
        Name == "floorf" || Name == "ceil" || Name == "round" ||
        Name == "ffs" || Name == "ffsl" || Name == "abs" || Name == "labs" ||
        Name == "llabs")
      return false;

    return true;
  }

  void getUnrollingPreferences(Loop *, TTI::UnrollingPreferences &) {}

  bool isLegalAddImmediate(int64_t Imm) { return false; }

  bool isLegalICmpImmediate(int64_t Imm) { return false; }

  bool isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV, int64_t BaseOffset,
                             bool HasBaseReg, int64_t Scale,
                             unsigned AddrSpace) {
    // Guess that only reg and reg+reg addressing is allowed. This heuristic is
    // taken from the implementation of LSR.
    return !BaseGV && BaseOffset == 0 && (Scale == 0 || Scale == 1);
  }

  bool isLegalMaskedStore(Type *DataType, int Consecutive) { return false; }

  bool isLegalMaskedLoad(Type *DataType, int Consecutive) { return false; }

  int getScalingFactorCost(Type *Ty, GlobalValue *BaseGV, int64_t BaseOffset,
                           bool HasBaseReg, int64_t Scale, unsigned AddrSpace) {
    // Guess that all legal addressing mode are free.
    if (isLegalAddressingMode(Ty, BaseGV, BaseOffset, HasBaseReg,
                              Scale, AddrSpace))
      return 0;
    return -1;
  }

  bool isTruncateFree(Type *Ty1, Type *Ty2) { return false; }

  bool isProfitableToHoist(Instruction *I) { return true; }

  bool isTypeLegal(Type *Ty) { return false; }

  unsigned getJumpBufAlignment() { return 0; }

  unsigned getJumpBufSize() { return 0; }

  bool shouldBuildLookupTables() { return true; }

  bool enableAggressiveInterleaving(bool LoopHasReductions) { return false; }

  TTI::PopcntSupportKind getPopcntSupport(unsigned IntTyWidthInBit) {
    return TTI::PSK_Software;
  }

  bool haveFastSqrt(Type *Ty) { return false; }

  unsigned getFPOpCost(Type *Ty) { return TargetTransformInfo::TCC_Basic; }

  unsigned getIntImmCost(const APInt &Imm, Type *Ty) { return TTI::TCC_Basic; }

  unsigned getIntImmCost(unsigned Opcode, unsigned Idx, const APInt &Imm,
                         Type *Ty) {
    return TTI::TCC_Free;
  }

  unsigned getIntImmCost(Intrinsic::ID IID, unsigned Idx, const APInt &Imm,
                         Type *Ty) {
    return TTI::TCC_Free;
  }

  unsigned getNumberOfRegisters(bool Vector) { return 8; }

  unsigned getRegisterBitWidth(bool Vector) { return 32; }

  unsigned getMaxInterleaveFactor(unsigned VF) { return 1; }

  unsigned getArithmeticInstrCost(unsigned Opcode, Type *Ty,
                                  TTI::OperandValueKind Opd1Info,
                                  TTI::OperandValueKind Opd2Info,
                                  TTI::OperandValueProperties Opd1PropInfo,
                                  TTI::OperandValueProperties Opd2PropInfo) {
    return 1;
  }

  unsigned getShuffleCost(TTI::ShuffleKind Kind, Type *Ty, int Index,
                          Type *SubTp) {
    return 1;
  }

  unsigned getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src) { return 1; }

  unsigned getCFInstrCost(unsigned Opcode) { return 1; }

  unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy) {
    return 1;
  }

  unsigned getVectorInstrCost(unsigned Opcode, Type *Val, unsigned Index) {
    return 1;
  }

  unsigned getMemoryOpCost(unsigned Opcode, Type *Src, unsigned Alignment,
                           unsigned AddressSpace) {
    return 1;
  }

  unsigned getMaskedMemoryOpCost(unsigned Opcode, Type *Src, unsigned Alignment,
                                 unsigned AddressSpace) {
    return 1;
  }

  unsigned getInterleavedMemoryOpCost(unsigned Opcode, Type *VecTy,
                                      unsigned Factor,
                                      ArrayRef<unsigned> Indices,
                                      unsigned Alignment,
                                      unsigned AddressSpace) {
    return 1;
  }

  unsigned getIntrinsicInstrCost(Intrinsic::ID ID, Type *RetTy,
                                 ArrayRef<Type *> Tys) {
    return 1;
  }

  unsigned getCallInstrCost(Function *F, Type *RetTy, ArrayRef<Type *> Tys) {
    return 1;
  }

  unsigned getNumberOfParts(Type *Tp) { return 0; }

  unsigned getAddressComputationCost(Type *Tp, bool) { return 0; }

  unsigned getReductionCost(unsigned, Type *, bool) { return 1; }

  unsigned getCostOfKeepingLiveOverCall(ArrayRef<Type *> Tys) { return 0; }

  bool getTgtMemIntrinsic(IntrinsicInst *Inst, MemIntrinsicInfo &Info) {
    return false;
  }

  Value *getOrCreateResultFromMemIntrinsic(IntrinsicInst *Inst,
                                           Type *ExpectedType) {
    return nullptr;
  }

  bool hasCompatibleFunctionAttributes(const Function *Caller,
                                       const Function *Callee) const {
    return (Caller->getFnAttribute("target-cpu") ==
            Callee->getFnAttribute("target-cpu")) &&
           (Caller->getFnAttribute("target-features") ==
            Callee->getFnAttribute("target-features"));
  }
};

/// \brief CRTP base class for use as a mix-in that aids implementing
/// a TargetTransformInfo-compatible class.
template <typename T>
class TargetTransformInfoImplCRTPBase : public TargetTransformInfoImplBase {
private:
  typedef TargetTransformInfoImplBase BaseT;

protected:
  explicit TargetTransformInfoImplCRTPBase(const DataLayout &DL) : BaseT(DL) {}

public:
  // Provide value semantics. MSVC requires that we spell all of these out.
  TargetTransformInfoImplCRTPBase(const TargetTransformInfoImplCRTPBase &Arg)
      : BaseT(static_cast<const BaseT &>(Arg)) {}
  TargetTransformInfoImplCRTPBase(TargetTransformInfoImplCRTPBase &&Arg)
      : BaseT(std::move(static_cast<BaseT &>(Arg))) {}

  using BaseT::getCallCost;

  unsigned getCallCost(const Function *F, int NumArgs) {
    assert(F && "A concrete function must be provided to this routine.");

    if (NumArgs < 0)
      // Set the argument number to the number of explicit arguments in the
      // function.
      NumArgs = F->arg_size();

    if (Intrinsic::ID IID = F->getIntrinsicID()) {
      FunctionType *FTy = F->getFunctionType();
      SmallVector<Type *, 8> ParamTys(FTy->param_begin(), FTy->param_end());
      return static_cast<T *>(this)
          ->getIntrinsicCost(IID, FTy->getReturnType(), ParamTys);
    }

    if (!static_cast<T *>(this)->isLoweredToCall(F))
      return TTI::TCC_Basic; // Give a basic cost if it will be lowered
                             // directly.

    return static_cast<T *>(this)->getCallCost(F->getFunctionType(), NumArgs);
  }

  unsigned getCallCost(const Function *F, ArrayRef<const Value *> Arguments) {
    // Simply delegate to generic handling of the call.
    // FIXME: We should use instsimplify or something else to catch calls which
    // will constant fold with these arguments.
    return static_cast<T *>(this)->getCallCost(F, Arguments.size());
  }

  using BaseT::getIntrinsicCost;

  unsigned getIntrinsicCost(Intrinsic::ID IID, Type *RetTy,
                            ArrayRef<const Value *> Arguments) {
    // Delegate to the generic intrinsic handling code. This mostly provides an
    // opportunity for targets to (for example) special case the cost of
    // certain intrinsics based on constants used as arguments.
    SmallVector<Type *, 8> ParamTys;
    ParamTys.reserve(Arguments.size());
    for (unsigned Idx = 0, Size = Arguments.size(); Idx != Size; ++Idx)
      ParamTys.push_back(Arguments[Idx]->getType());
    return static_cast<T *>(this)->getIntrinsicCost(IID, RetTy, ParamTys);
  }

  unsigned getUserCost(const User *U) {
    if (isa<PHINode>(U))
      return TTI::TCC_Free; // Model all PHI nodes as free.

    if (const GEPOperator *GEP = dyn_cast<GEPOperator>(U)) {
      SmallVector<const Value *, 4> Indices(GEP->idx_begin(), GEP->idx_end());
      return static_cast<T *>(this)
          ->getGEPCost(GEP->getPointerOperand(), Indices);
    }

    if (auto CS = ImmutableCallSite(U)) {
      const Function *F = CS.getCalledFunction();
      if (!F) {
        // Just use the called value type.
        Type *FTy = CS.getCalledValue()->getType()->getPointerElementType();
        return static_cast<T *>(this)
            ->getCallCost(cast<FunctionType>(FTy), CS.arg_size());
      }

      SmallVector<const Value *, 8> Arguments(CS.arg_begin(), CS.arg_end());
      return static_cast<T *>(this)->getCallCost(F, Arguments);
    }

    if (const CastInst *CI = dyn_cast<CastInst>(U)) {
      // Result of a cmp instruction is often extended (to be used by other
      // cmp instructions, logical or return instructions). These are usually
      // nop on most sane targets.
      if (isa<CmpInst>(CI->getOperand(0)))
        return TTI::TCC_Free;
    }

    return static_cast<T *>(this)->getOperationCost(
        Operator::getOpcode(U), U->getType(),
        U->getNumOperands() == 1 ? U->getOperand(0)->getType() : nullptr);
  }
};
}

#endif
