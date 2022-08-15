//===- CostModel.cpp ------ Cost Model Analysis ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the cost model analysis. It provides a very basic cost
// estimation for LLVM-IR. This analysis uses the services of the codegen
// to approximate the cost of any IR instruction when lowered to machine
// instructions. The cost results are unit-less and the cost number represents
// the throughput of the machine assuming that all loads hit the cache, all
// branches are predicted, etc. The cost numbers can be added in order to
// compare two or more transformation alternatives.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define CM_NAME "cost-model"
#define DEBUG_TYPE CM_NAME

static cl::opt<bool> EnableReduxCost("costmodel-reduxcost", cl::init(false),
                                     cl::Hidden,
                                     cl::desc("Recognize reduction patterns."));

namespace {
  class CostModelAnalysis : public FunctionPass {

  public:
    static char ID; // Class identification, replacement for typeinfo
    CostModelAnalysis() : FunctionPass(ID), F(nullptr), TTI(nullptr) {
      initializeCostModelAnalysisPass(
        *PassRegistry::getPassRegistry());
    }

    /// Returns the expected cost of the instruction.
    /// Returns -1 if the cost is unknown.
    /// Note, this method does not cache the cost calculation and it
    /// can be expensive in some cases.
    unsigned getInstructionCost(const Instruction *I) const;

  private:
    void getAnalysisUsage(AnalysisUsage &AU) const override;
    bool runOnFunction(Function &F) override;
    void print(raw_ostream &OS, const Module*) const override;

    /// The function that we analyze.
    Function *F;
    /// Target information.
    const TargetTransformInfo *TTI;
  };
}  // End of anonymous namespace

// Register this pass.
char CostModelAnalysis::ID = 0;
static const char cm_name[] = "Cost Model Analysis";
INITIALIZE_PASS_BEGIN(CostModelAnalysis, CM_NAME, cm_name, false, true)
INITIALIZE_PASS_END  (CostModelAnalysis, CM_NAME, cm_name, false, true)

FunctionPass *llvm::createCostModelAnalysisPass() {
  return new CostModelAnalysis();
}

void
CostModelAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

bool
CostModelAnalysis::runOnFunction(Function &F) {
 this->F = &F;
 auto *TTIWP = getAnalysisIfAvailable<TargetTransformInfoWrapperPass>();
 TTI = TTIWP ? &TTIWP->getTTI(F) : nullptr;

 return false;
}

static bool isReverseVectorMask(SmallVectorImpl<int> &Mask) {
  for (unsigned i = 0, MaskSize = Mask.size(); i < MaskSize; ++i)
    if (Mask[i] > 0 && Mask[i] != (int)(MaskSize - 1 - i))
      return false;
  return true;
}

static bool isAlternateVectorMask(SmallVectorImpl<int> &Mask) {
  bool isAlternate = true;
  unsigned MaskSize = Mask.size();

  // Example: shufflevector A, B, <0,5,2,7>
  for (unsigned i = 0; i < MaskSize && isAlternate; ++i) {
    if (Mask[i] < 0)
      continue;
    isAlternate = Mask[i] == (int)((i & 1) ? MaskSize + i : i);
  }

  if (isAlternate)
    return true;

  isAlternate = true;
  // Example: shufflevector A, B, <4,1,6,3>
  for (unsigned i = 0; i < MaskSize && isAlternate; ++i) {
    if (Mask[i] < 0)
      continue;
    isAlternate = Mask[i] == (int)((i & 1) ? i : MaskSize + i);
  }

  return isAlternate;
}

static TargetTransformInfo::OperandValueKind getOperandInfo(Value *V) {
  TargetTransformInfo::OperandValueKind OpInfo =
    TargetTransformInfo::OK_AnyValue;

  // Check for a splat of a constant or for a non uniform vector of constants.
  if (isa<ConstantVector>(V) || isa<ConstantDataVector>(V)) {
    OpInfo = TargetTransformInfo::OK_NonUniformConstantValue;
    if (cast<Constant>(V)->getSplatValue() != nullptr)
      OpInfo = TargetTransformInfo::OK_UniformConstantValue;
  }

  return OpInfo;
}

static bool matchPairwiseShuffleMask(ShuffleVectorInst *SI, bool IsLeft,
                                     unsigned Level) {
  // We don't need a shuffle if we just want to have element 0 in position 0 of
  // the vector.
  if (!SI && Level == 0 && IsLeft)
    return true;
  else if (!SI)
    return false;

  SmallVector<int, 32> Mask(SI->getType()->getVectorNumElements(), -1);

  // Build a mask of 0, 2, ... (left) or 1, 3, ... (right) depending on whether
  // we look at the left or right side.
  for (unsigned i = 0, e = (1 << Level), val = !IsLeft; i != e; ++i, val += 2)
    Mask[i] = val;

  SmallVector<int, 16> ActualMask = SI->getShuffleMask();
  if (Mask != ActualMask)
    return false;

  return true;
}

static bool matchPairwiseReductionAtLevel(const BinaryOperator *BinOp,
                                          unsigned Level, unsigned NumLevels) {
  // Match one level of pairwise operations.
  // %rdx.shuf.0.0 = shufflevector <4 x float> %rdx, <4 x float> undef,
  //       <4 x i32> <i32 0, i32 2 , i32 undef, i32 undef>
  // %rdx.shuf.0.1 = shufflevector <4 x float> %rdx, <4 x float> undef,
  //       <4 x i32> <i32 1, i32 3, i32 undef, i32 undef>
  // %bin.rdx.0 = fadd <4 x float> %rdx.shuf.0.0, %rdx.shuf.0.1
  if (BinOp == nullptr)
    return false;

  assert(BinOp->getType()->isVectorTy() && "Expecting a vector type");

  unsigned Opcode = BinOp->getOpcode();
  Value *L = BinOp->getOperand(0);
  Value *R = BinOp->getOperand(1);

  ShuffleVectorInst *LS = dyn_cast<ShuffleVectorInst>(L);
  if (!LS && Level)
    return false;
  ShuffleVectorInst *RS = dyn_cast<ShuffleVectorInst>(R);
  if (!RS && Level)
    return false;

  // On level 0 we can omit one shufflevector instruction.
  if (!Level && !RS && !LS)
    return false;

  // Shuffle inputs must match.
  Value *NextLevelOpL = LS ? LS->getOperand(0) : nullptr;
  Value *NextLevelOpR = RS ? RS->getOperand(0) : nullptr;
  Value *NextLevelOp = nullptr;
  if (NextLevelOpR && NextLevelOpL) {
    // If we have two shuffles their operands must match.
    if (NextLevelOpL != NextLevelOpR)
      return false;

    NextLevelOp = NextLevelOpL;
  } else if (Level == 0 && (NextLevelOpR || NextLevelOpL)) {
    // On the first level we can omit the shufflevector <0, undef,...>. So the
    // input to the other shufflevector <1, undef> must match with one of the
    // inputs to the current binary operation.
    // Example:
    //  %NextLevelOpL = shufflevector %R, <1, undef ...>
    //  %BinOp        = fadd          %NextLevelOpL, %R
    if (NextLevelOpL && NextLevelOpL != R)
      return false;
    else if (NextLevelOpR && NextLevelOpR != L)
      return false;

    NextLevelOp = NextLevelOpL ? R : L;
  } else
    return false;

  // Check that the next levels binary operation exists and matches with the
  // current one.
  BinaryOperator *NextLevelBinOp = nullptr;
  if (Level + 1 != NumLevels) {
    if (!(NextLevelBinOp = dyn_cast<BinaryOperator>(NextLevelOp)))
      return false;
    else if (NextLevelBinOp->getOpcode() != Opcode)
      return false;
  }

  // Shuffle mask for pairwise operation must match.
  if (matchPairwiseShuffleMask(LS, true, Level)) {
    if (!matchPairwiseShuffleMask(RS, false, Level))
      return false;
  } else if (matchPairwiseShuffleMask(RS, true, Level)) {
    if (!matchPairwiseShuffleMask(LS, false, Level))
      return false;
  } else
    return false;

  if (++Level == NumLevels)
    return true;

  // Match next level.
  return matchPairwiseReductionAtLevel(NextLevelBinOp, Level, NumLevels);
}

static bool matchPairwiseReduction(const ExtractElementInst *ReduxRoot,
                                   unsigned &Opcode, Type *&Ty) {
  if (!EnableReduxCost)
    return false;

  // Need to extract the first element.
  ConstantInt *CI = dyn_cast<ConstantInt>(ReduxRoot->getOperand(1));
  unsigned Idx = ~0u;
  if (CI)
    Idx = CI->getZExtValue();
  if (Idx != 0)
    return false;

  BinaryOperator *RdxStart = dyn_cast<BinaryOperator>(ReduxRoot->getOperand(0));
  if (!RdxStart)
    return false;

  Type *VecTy = ReduxRoot->getOperand(0)->getType();
  unsigned NumVecElems = VecTy->getVectorNumElements();
  if (!isPowerOf2_32(NumVecElems))
    return false;

  // We look for a sequence of shuffle,shuffle,add triples like the following
  // that builds a pairwise reduction tree.
  //
  //  (X0, X1, X2, X3)
  //   (X0 + X1, X2 + X3, undef, undef)
  //    ((X0 + X1) + (X2 + X3), undef, undef, undef)
  //
  // %rdx.shuf.0.0 = shufflevector <4 x float> %rdx, <4 x float> undef,
  //       <4 x i32> <i32 0, i32 2 , i32 undef, i32 undef>
  // %rdx.shuf.0.1 = shufflevector <4 x float> %rdx, <4 x float> undef,
  //       <4 x i32> <i32 1, i32 3, i32 undef, i32 undef>
  // %bin.rdx.0 = fadd <4 x float> %rdx.shuf.0.0, %rdx.shuf.0.1
  // %rdx.shuf.1.0 = shufflevector <4 x float> %bin.rdx.0, <4 x float> undef,
  //       <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  // %rdx.shuf.1.1 = shufflevector <4 x float> %bin.rdx.0, <4 x float> undef,
  //       <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  // %bin.rdx8 = fadd <4 x float> %rdx.shuf.1.0, %rdx.shuf.1.1
  // %r = extractelement <4 x float> %bin.rdx8, i32 0
  if (!matchPairwiseReductionAtLevel(RdxStart, 0,  Log2_32(NumVecElems)))
    return false;

  Opcode = RdxStart->getOpcode();
  Ty = VecTy;

  return true;
}

static std::pair<Value *, ShuffleVectorInst *>
getShuffleAndOtherOprd(BinaryOperator *B) {

  Value *L = B->getOperand(0);
  Value *R = B->getOperand(1);
  ShuffleVectorInst *S = nullptr;

  if ((S = dyn_cast<ShuffleVectorInst>(L)))
    return std::make_pair(R, S);

  S = dyn_cast<ShuffleVectorInst>(R);
  return std::make_pair(L, S);
}

static bool matchVectorSplittingReduction(const ExtractElementInst *ReduxRoot,
                                          unsigned &Opcode, Type *&Ty) {
  if (!EnableReduxCost)
    return false;

  // Need to extract the first element.
  ConstantInt *CI = dyn_cast<ConstantInt>(ReduxRoot->getOperand(1));
  unsigned Idx = ~0u;
  if (CI)
    Idx = CI->getZExtValue();
  if (Idx != 0)
    return false;

  BinaryOperator *RdxStart = dyn_cast<BinaryOperator>(ReduxRoot->getOperand(0));
  if (!RdxStart)
    return false;
  unsigned RdxOpcode = RdxStart->getOpcode();

  Type *VecTy = ReduxRoot->getOperand(0)->getType();
  unsigned NumVecElems = VecTy->getVectorNumElements();
  if (!isPowerOf2_32(NumVecElems))
    return false;

  // We look for a sequence of shuffles and adds like the following matching one
  // fadd, shuffle vector pair at a time.
  //
  // %rdx.shuf = shufflevector <4 x float> %rdx, <4 x float> undef,
  //                           <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  // %bin.rdx = fadd <4 x float> %rdx, %rdx.shuf
  // %rdx.shuf7 = shufflevector <4 x float> %bin.rdx, <4 x float> undef,
  //                          <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  // %bin.rdx8 = fadd <4 x float> %bin.rdx, %rdx.shuf7
  // %r = extractelement <4 x float> %bin.rdx8, i32 0

  unsigned MaskStart = 1;
  Value *RdxOp = RdxStart;
  SmallVector<int, 32> ShuffleMask(NumVecElems, 0);
  unsigned NumVecElemsRemain = NumVecElems;
  while (NumVecElemsRemain - 1) {
    // Check for the right reduction operation.
    BinaryOperator *BinOp;
    if (!(BinOp = dyn_cast<BinaryOperator>(RdxOp)))
      return false;
    if (BinOp->getOpcode() != RdxOpcode)
      return false;

    Value *NextRdxOp;
    ShuffleVectorInst *Shuffle;
    std::tie(NextRdxOp, Shuffle) = getShuffleAndOtherOprd(BinOp);

    // Check the current reduction operation and the shuffle use the same value.
    if (Shuffle == nullptr)
      return false;
    if (Shuffle->getOperand(0) != NextRdxOp)
      return false;

    // Check that shuffle masks matches.
    for (unsigned j = 0; j != MaskStart; ++j)
      ShuffleMask[j] = MaskStart + j;
    // Fill the rest of the mask with -1 for undef.
    std::fill(&ShuffleMask[MaskStart], ShuffleMask.end(), -1);

    SmallVector<int, 16> Mask = Shuffle->getShuffleMask();
    if (ShuffleMask != Mask)
      return false;

    RdxOp = NextRdxOp;
    NumVecElemsRemain /= 2;
    MaskStart *= 2;
  }

  Opcode = RdxOpcode;
  Ty = VecTy;
  return true;
}

unsigned CostModelAnalysis::getInstructionCost(const Instruction *I) const {
  if (!TTI)
    return -1;

  switch (I->getOpcode()) {
  case Instruction::GetElementPtr:{
    Type *ValTy = I->getOperand(0)->getType()->getPointerElementType();
    return TTI->getAddressComputationCost(ValTy);
  }

  case Instruction::Ret:
  case Instruction::PHI:
  case Instruction::Br: {
    return TTI->getCFInstrCost(I->getOpcode());
  }
  case Instruction::Add:
  case Instruction::FAdd:
  case Instruction::Sub:
  case Instruction::FSub:
  case Instruction::Mul:
  case Instruction::FMul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::FRem:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor: {
    TargetTransformInfo::OperandValueKind Op1VK =
      getOperandInfo(I->getOperand(0));
    TargetTransformInfo::OperandValueKind Op2VK =
      getOperandInfo(I->getOperand(1));
    return TTI->getArithmeticInstrCost(I->getOpcode(), I->getType(), Op1VK,
                                       Op2VK);
  }
  case Instruction::Select: {
    const SelectInst *SI = cast<SelectInst>(I);
    Type *CondTy = SI->getCondition()->getType();
    return TTI->getCmpSelInstrCost(I->getOpcode(), I->getType(), CondTy);
  }
  case Instruction::ICmp:
  case Instruction::FCmp: {
    Type *ValTy = I->getOperand(0)->getType();
    return TTI->getCmpSelInstrCost(I->getOpcode(), ValTy);
  }
  case Instruction::Store: {
    const StoreInst *SI = cast<StoreInst>(I);
    Type *ValTy = SI->getValueOperand()->getType();
    return TTI->getMemoryOpCost(I->getOpcode(), ValTy,
                                 SI->getAlignment(),
                                 SI->getPointerAddressSpace());
  }
  case Instruction::Load: {
    const LoadInst *LI = cast<LoadInst>(I);
    return TTI->getMemoryOpCost(I->getOpcode(), I->getType(),
                                 LI->getAlignment(),
                                 LI->getPointerAddressSpace());
  }
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::FPExt:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
  case Instruction::Trunc:
  case Instruction::FPTrunc:
  case Instruction::BitCast:
  case Instruction::AddrSpaceCast: {
    Type *SrcTy = I->getOperand(0)->getType();
    return TTI->getCastInstrCost(I->getOpcode(), I->getType(), SrcTy);
  }
  case Instruction::ExtractElement: {
    const ExtractElementInst * EEI = cast<ExtractElementInst>(I);
    ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(1));
    unsigned Idx = -1;
    if (CI)
      Idx = CI->getZExtValue();

    // Try to match a reduction sequence (series of shufflevector and vector
    // adds followed by a extractelement).
    unsigned ReduxOpCode;
    Type *ReduxType;

    if (matchVectorSplittingReduction(EEI, ReduxOpCode, ReduxType))
      return TTI->getReductionCost(ReduxOpCode, ReduxType, false);
    else if (matchPairwiseReduction(EEI, ReduxOpCode, ReduxType))
      return TTI->getReductionCost(ReduxOpCode, ReduxType, true);

    return TTI->getVectorInstrCost(I->getOpcode(),
                                   EEI->getOperand(0)->getType(), Idx);
  }
  case Instruction::InsertElement: {
    const InsertElementInst * IE = cast<InsertElementInst>(I);
    ConstantInt *CI = dyn_cast<ConstantInt>(IE->getOperand(2));
    unsigned Idx = -1;
    if (CI)
      Idx = CI->getZExtValue();
    return TTI->getVectorInstrCost(I->getOpcode(),
                                   IE->getType(), Idx);
  }
  case Instruction::ShuffleVector: {
    const ShuffleVectorInst *Shuffle = cast<ShuffleVectorInst>(I);
    Type *VecTypOp0 = Shuffle->getOperand(0)->getType();
    unsigned NumVecElems = VecTypOp0->getVectorNumElements();
    SmallVector<int, 16> Mask = Shuffle->getShuffleMask();

    if (NumVecElems == Mask.size()) {
      if (isReverseVectorMask(Mask))
        return TTI->getShuffleCost(TargetTransformInfo::SK_Reverse, VecTypOp0,
                                   0, nullptr);
      if (isAlternateVectorMask(Mask))
        return TTI->getShuffleCost(TargetTransformInfo::SK_Alternate,
                                   VecTypOp0, 0, nullptr);
    }

    return -1;
  }
  case Instruction::Call:
    if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
      SmallVector<Type*, 4> Tys;
      for (unsigned J = 0, JE = II->getNumArgOperands(); J != JE; ++J)
        Tys.push_back(II->getArgOperand(J)->getType());

      return TTI->getIntrinsicInstrCost(II->getIntrinsicID(), II->getType(),
                                        Tys);
    }
    return -1;
  default:
    // We don't have any information on this instruction.
    return -1;
  }
}

void CostModelAnalysis::print(raw_ostream &OS, const Module*) const {
  if (!F)
    return;

  for (Function::iterator B = F->begin(), BE = F->end(); B != BE; ++B) {
    for (BasicBlock::iterator it = B->begin(), e = B->end(); it != e; ++it) {
      Instruction *Inst = it;
      unsigned Cost = getInstructionCost(Inst);
      if (Cost != (unsigned)-1)
        OS << "Cost Model: Found an estimated cost of " << Cost;
      else
        OS << "Cost Model: Unknown cost";

      OS << " for instruction: "<< *Inst << "\n";
    }
  }
}
