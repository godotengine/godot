//===- BasicTTIImpl.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides a helper that implements much of the TTI interface in
/// terms of the target-independent code generator and TargetLowering
/// interfaces.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_BASICTTIIMPL_H
#define LLVM_CODEGEN_BASICTTIIMPL_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfoImpl.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"

namespace llvm {

extern cl::opt<unsigned> PartialUnrollingThreshold;

/// \brief Base class which can be used to help build a TTI implementation.
///
/// This class provides as much implementation of the TTI interface as is
/// possible using the target independent parts of the code generator.
///
/// In order to subclass it, your class must implement a getST() method to
/// return the subtarget, and a getTLI() method to return the target lowering.
/// We need these methods implemented in the derived class so that this class
/// doesn't have to duplicate storage for them.
template <typename T>
class BasicTTIImplBase : public TargetTransformInfoImplCRTPBase<T> {
private:
  typedef TargetTransformInfoImplCRTPBase<T> BaseT;
  typedef TargetTransformInfo TTI;

  /// Estimate the overhead of scalarizing an instruction. Insert and Extract
  /// are set if the result needs to be inserted and/or extracted from vectors.
  unsigned getScalarizationOverhead(Type *Ty, bool Insert, bool Extract) {
    assert(Ty->isVectorTy() && "Can only scalarize vectors");
    unsigned Cost = 0;

    for (int i = 0, e = Ty->getVectorNumElements(); i < e; ++i) {
      if (Insert)
        Cost += static_cast<T *>(this)
                    ->getVectorInstrCost(Instruction::InsertElement, Ty, i);
      if (Extract)
        Cost += static_cast<T *>(this)
                    ->getVectorInstrCost(Instruction::ExtractElement, Ty, i);
    }

    return Cost;
  }

  /// Estimate the cost overhead of SK_Alternate shuffle.
  unsigned getAltShuffleOverhead(Type *Ty) {
    assert(Ty->isVectorTy() && "Can only shuffle vectors");
    unsigned Cost = 0;
    // Shuffle cost is equal to the cost of extracting element from its argument
    // plus the cost of inserting them onto the result vector.

    // e.g. <4 x float> has a mask of <0,5,2,7> i.e we need to extract from
    // index 0 of first vector, index 1 of second vector,index 2 of first
    // vector and finally index 3 of second vector and insert them at index
    // <0,1,2,3> of result vector.
    for (int i = 0, e = Ty->getVectorNumElements(); i < e; ++i) {
      Cost += static_cast<T *>(this)
                  ->getVectorInstrCost(Instruction::InsertElement, Ty, i);
      Cost += static_cast<T *>(this)
                  ->getVectorInstrCost(Instruction::ExtractElement, Ty, i);
    }
    return Cost;
  }

  /// \brief Local query method delegates up to T which *must* implement this!
  const TargetSubtargetInfo *getST() const {
    return static_cast<const T *>(this)->getST();
  }

  /// \brief Local query method delegates up to T which *must* implement this!
  const TargetLoweringBase *getTLI() const {
    return static_cast<const T *>(this)->getTLI();
  }

protected:
  explicit BasicTTIImplBase(const TargetMachine *TM, const DataLayout &DL)
      : BaseT(DL) {}

  using TargetTransformInfoImplBase::DL;

public:
  // Provide value semantics. MSVC requires that we spell all of these out.
  BasicTTIImplBase(const BasicTTIImplBase &Arg)
      : BaseT(static_cast<const BaseT &>(Arg)) {}
  BasicTTIImplBase(BasicTTIImplBase &&Arg)
      : BaseT(std::move(static_cast<BaseT &>(Arg))) {}

  /// \name Scalar TTI Implementations
  /// @{

  bool hasBranchDivergence() { return false; }

  bool isSourceOfDivergence(const Value *V) { return false; }

  bool isLegalAddImmediate(int64_t imm) {
    return getTLI()->isLegalAddImmediate(imm);
  }

  bool isLegalICmpImmediate(int64_t imm) {
    return getTLI()->isLegalICmpImmediate(imm);
  }

  bool isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV, int64_t BaseOffset,
                             bool HasBaseReg, int64_t Scale,
                             unsigned AddrSpace) {
    TargetLoweringBase::AddrMode AM;
    AM.BaseGV = BaseGV;
    AM.BaseOffs = BaseOffset;
    AM.HasBaseReg = HasBaseReg;
    AM.Scale = Scale;
    return getTLI()->isLegalAddressingMode(DL, AM, Ty, AddrSpace);
  }

  int getScalingFactorCost(Type *Ty, GlobalValue *BaseGV, int64_t BaseOffset,
                           bool HasBaseReg, int64_t Scale, unsigned AddrSpace) {
    TargetLoweringBase::AddrMode AM;
    AM.BaseGV = BaseGV;
    AM.BaseOffs = BaseOffset;
    AM.HasBaseReg = HasBaseReg;
    AM.Scale = Scale;
    return getTLI()->getScalingFactorCost(DL, AM, Ty, AddrSpace);
  }

  bool isTruncateFree(Type *Ty1, Type *Ty2) {
    return getTLI()->isTruncateFree(Ty1, Ty2);
  }

  bool isProfitableToHoist(Instruction *I) {
    return getTLI()->isProfitableToHoist(I);
  }

  bool isTypeLegal(Type *Ty) {
    EVT VT = getTLI()->getValueType(DL, Ty);
    return getTLI()->isTypeLegal(VT);
  }

  unsigned getIntrinsicCost(Intrinsic::ID IID, Type *RetTy,
                            ArrayRef<const Value *> Arguments) {
    return BaseT::getIntrinsicCost(IID, RetTy, Arguments);
  }

  unsigned getIntrinsicCost(Intrinsic::ID IID, Type *RetTy,
                            ArrayRef<Type *> ParamTys) {
    if (IID == Intrinsic::cttz) {
      if (getTLI()->isCheapToSpeculateCttz())
        return TargetTransformInfo::TCC_Basic;
      return TargetTransformInfo::TCC_Expensive;
    }

    if (IID == Intrinsic::ctlz) {
       if (getTLI()->isCheapToSpeculateCtlz())
        return TargetTransformInfo::TCC_Basic;
      return TargetTransformInfo::TCC_Expensive;
    }

    return BaseT::getIntrinsicCost(IID, RetTy, ParamTys);
  }

  unsigned getJumpBufAlignment() { return getTLI()->getJumpBufAlignment(); }

  unsigned getJumpBufSize() { return getTLI()->getJumpBufSize(); }

  bool shouldBuildLookupTables() {
    const TargetLoweringBase *TLI = getTLI();
    return TLI->isOperationLegalOrCustom(ISD::BR_JT, MVT::Other) ||
           TLI->isOperationLegalOrCustom(ISD::BRIND, MVT::Other);
  }

  bool haveFastSqrt(Type *Ty) {
    const TargetLoweringBase *TLI = getTLI();
    EVT VT = TLI->getValueType(DL, Ty);
    return TLI->isTypeLegal(VT) &&
           TLI->isOperationLegalOrCustom(ISD::FSQRT, VT);
  }

  unsigned getFPOpCost(Type *Ty) {
    // By default, FP instructions are no more expensive since they are
    // implemented in HW.  Target specific TTI can override this.
    return TargetTransformInfo::TCC_Basic;
  }

  unsigned getOperationCost(unsigned Opcode, Type *Ty, Type *OpTy) {
    const TargetLoweringBase *TLI = getTLI();
    switch (Opcode) {
    default: break;
    case Instruction::Trunc: {
      if (TLI->isTruncateFree(OpTy, Ty))
        return TargetTransformInfo::TCC_Free;
      return TargetTransformInfo::TCC_Basic;
    }
    case Instruction::ZExt: {
      if (TLI->isZExtFree(OpTy, Ty))
        return TargetTransformInfo::TCC_Free;
      return TargetTransformInfo::TCC_Basic;
    }
    }

    return BaseT::getOperationCost(Opcode, Ty, OpTy);
  }

  void getUnrollingPreferences(Loop *L, TTI::UnrollingPreferences &UP) {
    // This unrolling functionality is target independent, but to provide some
    // motivation for its intended use, for x86:

    // According to the Intel 64 and IA-32 Architectures Optimization Reference
    // Manual, Intel Core models and later have a loop stream detector (and
    // associated uop queue) that can benefit from partial unrolling.
    // The relevant requirements are:
    //  - The loop must have no more than 4 (8 for Nehalem and later) branches
    //    taken, and none of them may be calls.
    //  - The loop can have no more than 18 (28 for Nehalem and later) uops.

    // According to the Software Optimization Guide for AMD Family 15h
    // Processors, models 30h-4fh (Steamroller and later) have a loop predictor
    // and loop buffer which can benefit from partial unrolling.
    // The relevant requirements are:
    //  - The loop must have fewer than 16 branches
    //  - The loop must have less than 40 uops in all executed loop branches

    // The number of taken branches in a loop is hard to estimate here, and
    // benchmarking has revealed that it is better not to be conservative when
    // estimating the branch count. As a result, we'll ignore the branch limits
    // until someone finds a case where it matters in practice.

    unsigned MaxOps;
    const TargetSubtargetInfo *ST = getST();
    if (PartialUnrollingThreshold.getNumOccurrences() > 0)
      MaxOps = PartialUnrollingThreshold;
    else if (ST->getSchedModel().LoopMicroOpBufferSize > 0)
      MaxOps = ST->getSchedModel().LoopMicroOpBufferSize;
    else
      return;

    // Scan the loop: don't unroll loops with calls.
    for (Loop::block_iterator I = L->block_begin(), E = L->block_end(); I != E;
         ++I) {
      BasicBlock *BB = *I;

      for (BasicBlock::iterator J = BB->begin(), JE = BB->end(); J != JE; ++J)
        if (isa<CallInst>(J) || isa<InvokeInst>(J)) {
          ImmutableCallSite CS(J);
          if (const Function *F = CS.getCalledFunction()) {
            if (!static_cast<T *>(this)->isLoweredToCall(F))
              continue;
          }

          return;
        }
    }

    // Enable runtime and partial unrolling up to the specified size.
    UP.Partial = UP.Runtime = true;
    UP.PartialThreshold = UP.PartialOptSizeThreshold = MaxOps;
  }

  /// @}

  /// \name Vector TTI Implementations
  /// @{

  unsigned getNumberOfRegisters(bool Vector) { return Vector ? 0 : 1; }

  unsigned getRegisterBitWidth(bool Vector) { return 32; }

  unsigned getMaxInterleaveFactor(unsigned VF) { return 1; }

  unsigned getArithmeticInstrCost(
      unsigned Opcode, Type *Ty,
      TTI::OperandValueKind Opd1Info = TTI::OK_AnyValue,
      TTI::OperandValueKind Opd2Info = TTI::OK_AnyValue,
      TTI::OperandValueProperties Opd1PropInfo = TTI::OP_None,
      TTI::OperandValueProperties Opd2PropInfo = TTI::OP_None) {
    // Check if any of the operands are vector operands.
    const TargetLoweringBase *TLI = getTLI();
    int ISD = TLI->InstructionOpcodeToISD(Opcode);
    assert(ISD && "Invalid opcode");

    std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(DL, Ty);

    bool IsFloat = Ty->getScalarType()->isFloatingPointTy();
    // Assume that floating point arithmetic operations cost twice as much as
    // integer operations.
    unsigned OpCost = (IsFloat ? 2 : 1);

    if (TLI->isOperationLegalOrPromote(ISD, LT.second)) {
      // The operation is legal. Assume it costs 1.
      // If the type is split to multiple registers, assume that there is some
      // overhead to this.
      // TODO: Once we have extract/insert subvector cost we need to use them.
      if (LT.first > 1)
        return LT.first * 2 * OpCost;
      return LT.first * 1 * OpCost;
    }

    if (!TLI->isOperationExpand(ISD, LT.second)) {
      // If the operation is custom lowered then assume
      // thare the code is twice as expensive.
      return LT.first * 2 * OpCost;
    }

    // Else, assume that we need to scalarize this op.
    if (Ty->isVectorTy()) {
      unsigned Num = Ty->getVectorNumElements();
      unsigned Cost = static_cast<T *>(this)
                          ->getArithmeticInstrCost(Opcode, Ty->getScalarType());
      // return the cost of multiple scalar invocation plus the cost of
      // inserting
      // and extracting the values.
      return getScalarizationOverhead(Ty, true, true) + Num * Cost;
    }

    // We don't know anything about this scalar instruction.
    return OpCost;
  }

  unsigned getShuffleCost(TTI::ShuffleKind Kind, Type *Tp, int Index,
                          Type *SubTp) {
    if (Kind == TTI::SK_Alternate) {
      return getAltShuffleOverhead(Tp);
    }
    return 1;
  }

  unsigned getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src) {
    const TargetLoweringBase *TLI = getTLI();
    int ISD = TLI->InstructionOpcodeToISD(Opcode);
    assert(ISD && "Invalid opcode");
    std::pair<unsigned, MVT> SrcLT = TLI->getTypeLegalizationCost(DL, Src);
    std::pair<unsigned, MVT> DstLT = TLI->getTypeLegalizationCost(DL, Dst);

    // Check for NOOP conversions.
    if (SrcLT.first == DstLT.first &&
        SrcLT.second.getSizeInBits() == DstLT.second.getSizeInBits()) {

      // Bitcast between types that are legalized to the same type are free.
      if (Opcode == Instruction::BitCast || Opcode == Instruction::Trunc)
        return 0;
    }

    if (Opcode == Instruction::Trunc &&
        TLI->isTruncateFree(SrcLT.second, DstLT.second))
      return 0;

    if (Opcode == Instruction::ZExt &&
        TLI->isZExtFree(SrcLT.second, DstLT.second))
      return 0;

    // If the cast is marked as legal (or promote) then assume low cost.
    if (SrcLT.first == DstLT.first &&
        TLI->isOperationLegalOrPromote(ISD, DstLT.second))
      return 1;

    // Handle scalar conversions.
    if (!Src->isVectorTy() && !Dst->isVectorTy()) {

      // Scalar bitcasts are usually free.
      if (Opcode == Instruction::BitCast)
        return 0;

      // Just check the op cost. If the operation is legal then assume it costs
      // 1.
      if (!TLI->isOperationExpand(ISD, DstLT.second))
        return 1;

      // Assume that illegal scalar instruction are expensive.
      return 4;
    }

    // Check vector-to-vector casts.
    if (Dst->isVectorTy() && Src->isVectorTy()) {

      // If the cast is between same-sized registers, then the check is simple.
      if (SrcLT.first == DstLT.first &&
          SrcLT.second.getSizeInBits() == DstLT.second.getSizeInBits()) {

        // Assume that Zext is done using AND.
        if (Opcode == Instruction::ZExt)
          return 1;

        // Assume that sext is done using SHL and SRA.
        if (Opcode == Instruction::SExt)
          return 2;

        // Just check the op cost. If the operation is legal then assume it
        // costs
        // 1 and multiply by the type-legalization overhead.
        if (!TLI->isOperationExpand(ISD, DstLT.second))
          return SrcLT.first * 1;
      }

      // If we are converting vectors and the operation is illegal, or
      // if the vectors are legalized to different types, estimate the
      // scalarization costs.
      unsigned Num = Dst->getVectorNumElements();
      unsigned Cost = static_cast<T *>(this)->getCastInstrCost(
          Opcode, Dst->getScalarType(), Src->getScalarType());

      // Return the cost of multiple scalar invocation plus the cost of
      // inserting and extracting the values.
      return getScalarizationOverhead(Dst, true, true) + Num * Cost;
    }

    // We already handled vector-to-vector and scalar-to-scalar conversions.
    // This
    // is where we handle bitcast between vectors and scalars. We need to assume
    //  that the conversion is scalarized in one way or another.
    if (Opcode == Instruction::BitCast)
      // Illegal bitcasts are done by storing and loading from a stack slot.
      return (Src->isVectorTy() ? getScalarizationOverhead(Src, false, true)
                                : 0) +
             (Dst->isVectorTy() ? getScalarizationOverhead(Dst, true, false)
                                : 0);

    llvm_unreachable("Unhandled cast");
  }

  unsigned getCFInstrCost(unsigned Opcode) {
    // Branches are assumed to be predicted.
    return 0;
  }

  unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy, Type *CondTy) {
    const TargetLoweringBase *TLI = getTLI();
    int ISD = TLI->InstructionOpcodeToISD(Opcode);
    assert(ISD && "Invalid opcode");

    // Selects on vectors are actually vector selects.
    if (ISD == ISD::SELECT) {
      assert(CondTy && "CondTy must exist");
      if (CondTy->isVectorTy())
        ISD = ISD::VSELECT;
    }
    std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(DL, ValTy);

    if (!(ValTy->isVectorTy() && !LT.second.isVector()) &&
        !TLI->isOperationExpand(ISD, LT.second)) {
      // The operation is legal. Assume it costs 1. Multiply
      // by the type-legalization overhead.
      return LT.first * 1;
    }

    // Otherwise, assume that the cast is scalarized.
    if (ValTy->isVectorTy()) {
      unsigned Num = ValTy->getVectorNumElements();
      if (CondTy)
        CondTy = CondTy->getScalarType();
      unsigned Cost = static_cast<T *>(this)->getCmpSelInstrCost(
          Opcode, ValTy->getScalarType(), CondTy);

      // Return the cost of multiple scalar invocation plus the cost of
      // inserting
      // and extracting the values.
      return getScalarizationOverhead(ValTy, true, false) + Num * Cost;
    }

    // Unknown scalar opcode.
    return 1;
  }

  unsigned getVectorInstrCost(unsigned Opcode, Type *Val, unsigned Index) {
    std::pair<unsigned, MVT> LT =
        getTLI()->getTypeLegalizationCost(DL, Val->getScalarType());

    return LT.first;
  }

  unsigned getMemoryOpCost(unsigned Opcode, Type *Src, unsigned Alignment,
                           unsigned AddressSpace) {
    assert(!Src->isVoidTy() && "Invalid type");
    std::pair<unsigned, MVT> LT = getTLI()->getTypeLegalizationCost(DL, Src);

    // Assuming that all loads of legal types cost 1.
    unsigned Cost = LT.first;

    if (Src->isVectorTy() &&
        Src->getPrimitiveSizeInBits() < LT.second.getSizeInBits()) {
      // This is a vector load that legalizes to a larger type than the vector
      // itself. Unless the corresponding extending load or truncating store is
      // legal, then this will scalarize.
      TargetLowering::LegalizeAction LA = TargetLowering::Expand;
      EVT MemVT = getTLI()->getValueType(DL, Src, true);
      if (MemVT.isSimple() && MemVT != MVT::Other) {
        if (Opcode == Instruction::Store)
          LA = getTLI()->getTruncStoreAction(LT.second, MemVT.getSimpleVT());
        else
          LA = getTLI()->getLoadExtAction(ISD::EXTLOAD, LT.second, MemVT);
      }

      if (LA != TargetLowering::Legal && LA != TargetLowering::Custom) {
        // This is a vector load/store for some illegal type that is scalarized.
        // We must account for the cost of building or decomposing the vector.
        Cost += getScalarizationOverhead(Src, Opcode != Instruction::Store,
                                         Opcode == Instruction::Store);
      }
    }

    return Cost;
  }

  unsigned getInterleavedMemoryOpCost(unsigned Opcode, Type *VecTy,
                                      unsigned Factor,
                                      ArrayRef<unsigned> Indices,
                                      unsigned Alignment,
                                      unsigned AddressSpace) {
    VectorType *VT = dyn_cast<VectorType>(VecTy);
    assert(VT && "Expect a vector type for interleaved memory op");

    unsigned NumElts = VT->getNumElements();
    assert(Factor > 1 && NumElts % Factor == 0 && "Invalid interleave factor");

    unsigned NumSubElts = NumElts / Factor;
    VectorType *SubVT = VectorType::get(VT->getElementType(), NumSubElts);

    // Firstly, the cost of load/store operation.
    unsigned Cost = getMemoryOpCost(Opcode, VecTy, Alignment, AddressSpace);

    // Then plus the cost of interleave operation.
    if (Opcode == Instruction::Load) {
      // The interleave cost is similar to extract sub vectors' elements
      // from the wide vector, and insert them into sub vectors.
      //
      // E.g. An interleaved load of factor 2 (with one member of index 0):
      //      %vec = load <8 x i32>, <8 x i32>* %ptr
      //      %v0 = shuffle %vec, undef, <0, 2, 4, 6>         ; Index 0
      // The cost is estimated as extract elements at 0, 2, 4, 6 from the
      // <8 x i32> vector and insert them into a <4 x i32> vector.

      assert(Indices.size() <= Factor &&
             "Interleaved memory op has too many members");
      for (unsigned Index : Indices) {
        assert(Index < Factor && "Invalid index for interleaved memory op");

        // Extract elements from loaded vector for each sub vector.
        for (unsigned i = 0; i < NumSubElts; i++)
          Cost += getVectorInstrCost(Instruction::ExtractElement, VT,
                                     Index + i * Factor);
      }

      unsigned InsSubCost = 0;
      for (unsigned i = 0; i < NumSubElts; i++)
        InsSubCost += getVectorInstrCost(Instruction::InsertElement, SubVT, i);

      Cost += Indices.size() * InsSubCost;
    } else {
      // The interleave cost is extract all elements from sub vectors, and
      // insert them into the wide vector.
      //
      // E.g. An interleaved store of factor 2:
      //      %v0_v1 = shuffle %v0, %v1, <0, 4, 1, 5, 2, 6, 3, 7>
      //      store <8 x i32> %interleaved.vec, <8 x i32>* %ptr
      // The cost is estimated as extract all elements from both <4 x i32>
      // vectors and insert into the <8 x i32> vector.

      unsigned ExtSubCost = 0;
      for (unsigned i = 0; i < NumSubElts; i++)
        ExtSubCost += getVectorInstrCost(Instruction::ExtractElement, SubVT, i);

      Cost += Factor * ExtSubCost;

      for (unsigned i = 0; i < NumElts; i++)
        Cost += getVectorInstrCost(Instruction::InsertElement, VT, i);
    }

    return Cost;
  }

  unsigned getIntrinsicInstrCost(Intrinsic::ID IID, Type *RetTy,
                                 ArrayRef<Type *> Tys) {
    unsigned ISD = 0;
    switch (IID) {
    default: {
      // Assume that we need to scalarize this intrinsic.
      unsigned ScalarizationCost = 0;
      unsigned ScalarCalls = 1;
      Type *ScalarRetTy = RetTy;
      if (RetTy->isVectorTy()) {
        ScalarizationCost = getScalarizationOverhead(RetTy, true, false);
        ScalarCalls = std::max(ScalarCalls, RetTy->getVectorNumElements());
        ScalarRetTy = RetTy->getScalarType();
      }
      SmallVector<Type *, 4> ScalarTys;
      for (unsigned i = 0, ie = Tys.size(); i != ie; ++i) {
        Type *Ty = Tys[i];
        if (Ty->isVectorTy()) {
          ScalarizationCost += getScalarizationOverhead(Ty, false, true);
          ScalarCalls = std::max(ScalarCalls, Ty->getVectorNumElements());
          Ty = Ty->getScalarType();
        }
        ScalarTys.push_back(Ty);
      }
      if (ScalarCalls == 1)
        return 1; // Return cost of a scalar intrinsic. Assume it to be cheap.

      unsigned ScalarCost = static_cast<T *>(this)->getIntrinsicInstrCost(
          IID, ScalarRetTy, ScalarTys);

      return ScalarCalls * ScalarCost + ScalarizationCost;
    }
    // Look for intrinsics that can be lowered directly or turned into a scalar
    // intrinsic call.
    case Intrinsic::sqrt:
      ISD = ISD::FSQRT;
      break;
    case Intrinsic::sin:
      ISD = ISD::FSIN;
      break;
    case Intrinsic::cos:
      ISD = ISD::FCOS;
      break;
    case Intrinsic::exp:
      ISD = ISD::FEXP;
      break;
    case Intrinsic::exp2:
      ISD = ISD::FEXP2;
      break;
    case Intrinsic::log:
      ISD = ISD::FLOG;
      break;
    case Intrinsic::log10:
      ISD = ISD::FLOG10;
      break;
    case Intrinsic::log2:
      ISD = ISD::FLOG2;
      break;
    case Intrinsic::fabs:
      ISD = ISD::FABS;
      break;
    case Intrinsic::minnum:
      ISD = ISD::FMINNUM;
      break;
    case Intrinsic::maxnum:
      ISD = ISD::FMAXNUM;
      break;
    case Intrinsic::copysign:
      ISD = ISD::FCOPYSIGN;
      break;
    case Intrinsic::floor:
      ISD = ISD::FFLOOR;
      break;
    case Intrinsic::ceil:
      ISD = ISD::FCEIL;
      break;
    case Intrinsic::trunc:
      ISD = ISD::FTRUNC;
      break;
    case Intrinsic::nearbyint:
      ISD = ISD::FNEARBYINT;
      break;
    case Intrinsic::rint:
      ISD = ISD::FRINT;
      break;
    case Intrinsic::round:
      ISD = ISD::FROUND;
      break;
    case Intrinsic::pow:
      ISD = ISD::FPOW;
      break;
    case Intrinsic::fma:
      ISD = ISD::FMA;
      break;
    case Intrinsic::fmuladd:
      ISD = ISD::FMA;
      break;
    // FIXME: We should return 0 whenever getIntrinsicCost == TCC_Free.
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
      return 0;
    case Intrinsic::masked_store:
      return static_cast<T *>(this)
          ->getMaskedMemoryOpCost(Instruction::Store, Tys[0], 0, 0);
    case Intrinsic::masked_load:
      return static_cast<T *>(this)
          ->getMaskedMemoryOpCost(Instruction::Load, RetTy, 0, 0);
    }

    const TargetLoweringBase *TLI = getTLI();
    std::pair<unsigned, MVT> LT = TLI->getTypeLegalizationCost(DL, RetTy);

    if (TLI->isOperationLegalOrPromote(ISD, LT.second)) {
      // The operation is legal. Assume it costs 1.
      // If the type is split to multiple registers, assume that there is some
      // overhead to this.
      // TODO: Once we have extract/insert subvector cost we need to use them.
      if (LT.first > 1)
        return LT.first * 2;
      return LT.first * 1;
    }

    if (!TLI->isOperationExpand(ISD, LT.second)) {
      // If the operation is custom lowered then assume
      // thare the code is twice as expensive.
      return LT.first * 2;
    }

    // If we can't lower fmuladd into an FMA estimate the cost as a floating
    // point mul followed by an add.
    if (IID == Intrinsic::fmuladd)
      return static_cast<T *>(this)
                 ->getArithmeticInstrCost(BinaryOperator::FMul, RetTy) +
             static_cast<T *>(this)
                 ->getArithmeticInstrCost(BinaryOperator::FAdd, RetTy);

    // Else, assume that we need to scalarize this intrinsic. For math builtins
    // this will emit a costly libcall, adding call overhead and spills. Make it
    // very expensive.
    if (RetTy->isVectorTy()) {
      unsigned ScalarizationCost = getScalarizationOverhead(RetTy, true, false);
      unsigned ScalarCalls = RetTy->getVectorNumElements();
      SmallVector<Type *, 4> ScalarTys;
      for (unsigned i = 0, ie = Tys.size(); i != ie; ++i) {
        Type *Ty = Tys[i];
        if (Ty->isVectorTy())
          Ty = Ty->getScalarType();
        ScalarTys.push_back(Ty);
      }
      unsigned ScalarCost = static_cast<T *>(this)->getIntrinsicInstrCost(
          IID, RetTy->getScalarType(), ScalarTys);
      for (unsigned i = 0, ie = Tys.size(); i != ie; ++i) {
        if (Tys[i]->isVectorTy()) {
          ScalarizationCost += getScalarizationOverhead(Tys[i], false, true);
          ScalarCalls = std::max(ScalarCalls, Tys[i]->getVectorNumElements());
        }
      }

      return ScalarCalls * ScalarCost + ScalarizationCost;
    }

    // This is going to be turned into a library call, make it expensive.
    return 10;
  }

  /// \brief Compute a cost of the given call instruction.
  ///
  /// Compute the cost of calling function F with return type RetTy and
  /// argument types Tys. F might be nullptr, in this case the cost of an
  /// arbitrary call with the specified signature will be returned.
  /// This is used, for instance,  when we estimate call of a vector
  /// counterpart of the given function.
  /// \param F Called function, might be nullptr.
  /// \param RetTy Return value types.
  /// \param Tys Argument types.
  /// \returns The cost of Call instruction.
  unsigned getCallInstrCost(Function *F, Type *RetTy, ArrayRef<Type *> Tys) {
    return 10;
  }

  unsigned getNumberOfParts(Type *Tp) {
    std::pair<unsigned, MVT> LT = getTLI()->getTypeLegalizationCost(DL, Tp);
    return LT.first;
  }

  unsigned getAddressComputationCost(Type *Ty, bool IsComplex) { return 0; }

  unsigned getReductionCost(unsigned Opcode, Type *Ty, bool IsPairwise) {
    assert(Ty->isVectorTy() && "Expect a vector type");
    unsigned NumVecElts = Ty->getVectorNumElements();
    unsigned NumReduxLevels = Log2_32(NumVecElts);
    unsigned ArithCost =
        NumReduxLevels *
        static_cast<T *>(this)->getArithmeticInstrCost(Opcode, Ty);
    // Assume the pairwise shuffles add a cost.
    unsigned ShuffleCost =
        NumReduxLevels * (IsPairwise + 1) *
        static_cast<T *>(this)
            ->getShuffleCost(TTI::SK_ExtractSubvector, Ty, NumVecElts / 2, Ty);
    return ShuffleCost + ArithCost + getScalarizationOverhead(Ty, false, true);
  }

  /// @}
};

/// \brief Concrete BasicTTIImpl that can be used if no further customization
/// is needed.
class BasicTTIImpl : public BasicTTIImplBase<BasicTTIImpl> {
  typedef BasicTTIImplBase<BasicTTIImpl> BaseT;
  friend class BasicTTIImplBase<BasicTTIImpl>;

  const TargetSubtargetInfo *ST;
  const TargetLoweringBase *TLI;

  const TargetSubtargetInfo *getST() const { return ST; }
  const TargetLoweringBase *getTLI() const { return TLI; }

public:
  explicit BasicTTIImpl(const TargetMachine *ST, Function &F);

  // Provide value semantics. MSVC requires that we spell all of these out.
  BasicTTIImpl(const BasicTTIImpl &Arg)
      : BaseT(static_cast<const BaseT &>(Arg)), ST(Arg.ST), TLI(Arg.TLI) {}
  BasicTTIImpl(BasicTTIImpl &&Arg)
      : BaseT(std::move(static_cast<BaseT &>(Arg))), ST(std::move(Arg.ST)),
        TLI(std::move(Arg.TLI)) {}
};

}

#endif
