//===- llvm/Analysis/TargetTransformInfo.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TargetTransformInfoImpl.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "tti"

namespace {
/// \brief No-op implementation of the TTI interface using the utility base
/// classes.
///
/// This is used when no target specific information is available.
struct NoTTIImpl : TargetTransformInfoImplCRTPBase<NoTTIImpl> {
  explicit NoTTIImpl(const DataLayout &DL)
      : TargetTransformInfoImplCRTPBase<NoTTIImpl>(DL) {}
};
}

TargetTransformInfo::TargetTransformInfo(const DataLayout &DL)
    : TTIImpl(new Model<NoTTIImpl>(NoTTIImpl(DL))) {}

TargetTransformInfo::~TargetTransformInfo() {}

TargetTransformInfo::TargetTransformInfo(TargetTransformInfo &&Arg)
    : TTIImpl(std::move(Arg.TTIImpl)) {}

TargetTransformInfo &TargetTransformInfo::operator=(TargetTransformInfo &&RHS) {
  TTIImpl = std::move(RHS.TTIImpl);
  return *this;
}

unsigned TargetTransformInfo::getOperationCost(unsigned Opcode, Type *Ty,
                                               Type *OpTy) const {
  return TTIImpl->getOperationCost(Opcode, Ty, OpTy);
}

unsigned TargetTransformInfo::getCallCost(FunctionType *FTy,
                                          int NumArgs) const {
  return TTIImpl->getCallCost(FTy, NumArgs);
}

unsigned
TargetTransformInfo::getCallCost(const Function *F,
                                 ArrayRef<const Value *> Arguments) const {
  return TTIImpl->getCallCost(F, Arguments);
}

unsigned
TargetTransformInfo::getIntrinsicCost(Intrinsic::ID IID, Type *RetTy,
                                      ArrayRef<const Value *> Arguments) const {
  return TTIImpl->getIntrinsicCost(IID, RetTy, Arguments);
}

unsigned TargetTransformInfo::getUserCost(const User *U) const {
  return TTIImpl->getUserCost(U);
}

bool TargetTransformInfo::hasBranchDivergence() const {
  return TTIImpl->hasBranchDivergence();
}

bool TargetTransformInfo::isSourceOfDivergence(const Value *V) const {
  return TTIImpl->isSourceOfDivergence(V);
}

bool TargetTransformInfo::isLoweredToCall(const Function *F) const {
  return TTIImpl->isLoweredToCall(F);
}

void TargetTransformInfo::getUnrollingPreferences(
    Loop *L, UnrollingPreferences &UP) const {
  return TTIImpl->getUnrollingPreferences(L, UP);
}

bool TargetTransformInfo::isLegalAddImmediate(int64_t Imm) const {
  return TTIImpl->isLegalAddImmediate(Imm);
}

bool TargetTransformInfo::isLegalICmpImmediate(int64_t Imm) const {
  return TTIImpl->isLegalICmpImmediate(Imm);
}

bool TargetTransformInfo::isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV,
                                                int64_t BaseOffset,
                                                bool HasBaseReg,
                                                int64_t Scale,
                                                unsigned AddrSpace) const {
  return TTIImpl->isLegalAddressingMode(Ty, BaseGV, BaseOffset, HasBaseReg,
                                        Scale, AddrSpace);
}

bool TargetTransformInfo::isLegalMaskedStore(Type *DataType,
                                             int Consecutive) const {
  return TTIImpl->isLegalMaskedStore(DataType, Consecutive);
}

bool TargetTransformInfo::isLegalMaskedLoad(Type *DataType,
                                            int Consecutive) const {
  return TTIImpl->isLegalMaskedLoad(DataType, Consecutive);
}

int TargetTransformInfo::getScalingFactorCost(Type *Ty, GlobalValue *BaseGV,
                                              int64_t BaseOffset,
                                              bool HasBaseReg,
                                              int64_t Scale,
                                              unsigned AddrSpace) const {
  return TTIImpl->getScalingFactorCost(Ty, BaseGV, BaseOffset, HasBaseReg,
                                       Scale, AddrSpace);
}

bool TargetTransformInfo::isTruncateFree(Type *Ty1, Type *Ty2) const {
  return TTIImpl->isTruncateFree(Ty1, Ty2);
}

bool TargetTransformInfo::isProfitableToHoist(Instruction *I) const {
  return TTIImpl->isProfitableToHoist(I);
}

bool TargetTransformInfo::isTypeLegal(Type *Ty) const {
  return TTIImpl->isTypeLegal(Ty);
}

unsigned TargetTransformInfo::getJumpBufAlignment() const {
  return TTIImpl->getJumpBufAlignment();
}

unsigned TargetTransformInfo::getJumpBufSize() const {
  return TTIImpl->getJumpBufSize();
}

bool TargetTransformInfo::shouldBuildLookupTables() const {
  return TTIImpl->shouldBuildLookupTables();
}

bool TargetTransformInfo::enableAggressiveInterleaving(bool LoopHasReductions) const {
  return TTIImpl->enableAggressiveInterleaving(LoopHasReductions);
}

TargetTransformInfo::PopcntSupportKind
TargetTransformInfo::getPopcntSupport(unsigned IntTyWidthInBit) const {
  return TTIImpl->getPopcntSupport(IntTyWidthInBit);
}

bool TargetTransformInfo::haveFastSqrt(Type *Ty) const {
  return TTIImpl->haveFastSqrt(Ty);
}

unsigned TargetTransformInfo::getFPOpCost(Type *Ty) const {
  return TTIImpl->getFPOpCost(Ty);
}

unsigned TargetTransformInfo::getIntImmCost(const APInt &Imm, Type *Ty) const {
  return TTIImpl->getIntImmCost(Imm, Ty);
}

unsigned TargetTransformInfo::getIntImmCost(unsigned Opcode, unsigned Idx,
                                            const APInt &Imm, Type *Ty) const {
  return TTIImpl->getIntImmCost(Opcode, Idx, Imm, Ty);
}

unsigned TargetTransformInfo::getIntImmCost(Intrinsic::ID IID, unsigned Idx,
                                            const APInt &Imm, Type *Ty) const {
  return TTIImpl->getIntImmCost(IID, Idx, Imm, Ty);
}

unsigned TargetTransformInfo::getNumberOfRegisters(bool Vector) const {
  return TTIImpl->getNumberOfRegisters(Vector);
}

unsigned TargetTransformInfo::getRegisterBitWidth(bool Vector) const {
  return TTIImpl->getRegisterBitWidth(Vector);
}

unsigned TargetTransformInfo::getMaxInterleaveFactor(unsigned VF) const {
  return TTIImpl->getMaxInterleaveFactor(VF);
}

unsigned TargetTransformInfo::getArithmeticInstrCost(
    unsigned Opcode, Type *Ty, OperandValueKind Opd1Info,
    OperandValueKind Opd2Info, OperandValueProperties Opd1PropInfo,
    OperandValueProperties Opd2PropInfo) const {
  return TTIImpl->getArithmeticInstrCost(Opcode, Ty, Opd1Info, Opd2Info,
                                         Opd1PropInfo, Opd2PropInfo);
}

unsigned TargetTransformInfo::getShuffleCost(ShuffleKind Kind, Type *Ty,
                                             int Index, Type *SubTp) const {
  return TTIImpl->getShuffleCost(Kind, Ty, Index, SubTp);
}

unsigned TargetTransformInfo::getCastInstrCost(unsigned Opcode, Type *Dst,
                                               Type *Src) const {
  return TTIImpl->getCastInstrCost(Opcode, Dst, Src);
}

unsigned TargetTransformInfo::getCFInstrCost(unsigned Opcode) const {
  return TTIImpl->getCFInstrCost(Opcode);
}

unsigned TargetTransformInfo::getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                                 Type *CondTy) const {
  return TTIImpl->getCmpSelInstrCost(Opcode, ValTy, CondTy);
}

unsigned TargetTransformInfo::getVectorInstrCost(unsigned Opcode, Type *Val,
                                                 unsigned Index) const {
  return TTIImpl->getVectorInstrCost(Opcode, Val, Index);
}

unsigned TargetTransformInfo::getMemoryOpCost(unsigned Opcode, Type *Src,
                                              unsigned Alignment,
                                              unsigned AddressSpace) const {
  return TTIImpl->getMemoryOpCost(Opcode, Src, Alignment, AddressSpace);
}

unsigned
TargetTransformInfo::getMaskedMemoryOpCost(unsigned Opcode, Type *Src,
                                           unsigned Alignment,
                                           unsigned AddressSpace) const {
  return TTIImpl->getMaskedMemoryOpCost(Opcode, Src, Alignment, AddressSpace);
}

unsigned TargetTransformInfo::getInterleavedMemoryOpCost(
    unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
    unsigned Alignment, unsigned AddressSpace) const {
  return TTIImpl->getInterleavedMemoryOpCost(Opcode, VecTy, Factor, Indices,
                                             Alignment, AddressSpace);
}

unsigned
TargetTransformInfo::getIntrinsicInstrCost(Intrinsic::ID ID, Type *RetTy,
                                           ArrayRef<Type *> Tys) const {
  return TTIImpl->getIntrinsicInstrCost(ID, RetTy, Tys);
}

unsigned TargetTransformInfo::getCallInstrCost(Function *F, Type *RetTy,
                                               ArrayRef<Type *> Tys) const {
  return TTIImpl->getCallInstrCost(F, RetTy, Tys);
}

unsigned TargetTransformInfo::getNumberOfParts(Type *Tp) const {
  return TTIImpl->getNumberOfParts(Tp);
}

unsigned TargetTransformInfo::getAddressComputationCost(Type *Tp,
                                                        bool IsComplex) const {
  return TTIImpl->getAddressComputationCost(Tp, IsComplex);
}

unsigned TargetTransformInfo::getReductionCost(unsigned Opcode, Type *Ty,
                                               bool IsPairwiseForm) const {
  return TTIImpl->getReductionCost(Opcode, Ty, IsPairwiseForm);
}

unsigned
TargetTransformInfo::getCostOfKeepingLiveOverCall(ArrayRef<Type *> Tys) const {
  return TTIImpl->getCostOfKeepingLiveOverCall(Tys);
}

bool TargetTransformInfo::getTgtMemIntrinsic(IntrinsicInst *Inst,
                                             MemIntrinsicInfo &Info) const {
  return TTIImpl->getTgtMemIntrinsic(Inst, Info);
}

Value *TargetTransformInfo::getOrCreateResultFromMemIntrinsic(
    IntrinsicInst *Inst, Type *ExpectedType) const {
  return TTIImpl->getOrCreateResultFromMemIntrinsic(Inst, ExpectedType);
}

bool TargetTransformInfo::hasCompatibleFunctionAttributes(
    const Function *Caller, const Function *Callee) const {
  return TTIImpl->hasCompatibleFunctionAttributes(Caller, Callee);
}

TargetTransformInfo::Concept::~Concept() {}

TargetIRAnalysis::TargetIRAnalysis() : TTICallback(&getDefaultTTI) {}

TargetIRAnalysis::TargetIRAnalysis(
    std::function<Result(Function &)> TTICallback)
    : TTICallback(TTICallback) {}

TargetIRAnalysis::Result TargetIRAnalysis::run(Function &F) {
  return TTICallback(F);
}

char TargetIRAnalysis::PassID;

TargetIRAnalysis::Result TargetIRAnalysis::getDefaultTTI(Function &F) {
  return Result(F.getParent()->getDataLayout());
}

// Register the basic pass.
INITIALIZE_PASS(TargetTransformInfoWrapperPass, "tti",
                "Target Transform Information", false, true)
char TargetTransformInfoWrapperPass::ID = 0;

void TargetTransformInfoWrapperPass::anchor() {}

TargetTransformInfoWrapperPass::TargetTransformInfoWrapperPass()
    : ImmutablePass(ID) {
  initializeTargetTransformInfoWrapperPassPass(
      *PassRegistry::getPassRegistry());
}

TargetTransformInfoWrapperPass::TargetTransformInfoWrapperPass(
    TargetIRAnalysis TIRA)
    : ImmutablePass(ID), TIRA(std::move(TIRA)) {
  initializeTargetTransformInfoWrapperPassPass(
      *PassRegistry::getPassRegistry());
}

TargetTransformInfo &TargetTransformInfoWrapperPass::getTTI(Function &F) {
  TTI = TIRA.run(F);
  return *TTI;
}

ImmutablePass *
llvm::createTargetTransformInfoWrapperPass(TargetIRAnalysis TIRA) {
  return new TargetTransformInfoWrapperPass(std::move(TIRA));
}
