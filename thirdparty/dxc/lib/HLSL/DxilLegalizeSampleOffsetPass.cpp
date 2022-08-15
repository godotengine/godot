///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilSignature.cpp                                                         //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DxilLegalizeSampleOffsetPass implementation.                              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/DxilGenerationPass.h"
#include "llvm/Analysis/DxilValueCache.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilUtil.h"

#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"

#include <unordered_set>

using std::vector;
using std::unique_ptr;
using namespace llvm;
using namespace hlsl;

///////////////////////////////////////////////////////////////////////////////
// Legalize Sample offset.

namespace {

// record of the offset value and the call that uses it
// Used mainly for error detection and reporting
struct Offset {
  Value *offset;
  CallInst *call;
};

// When optimizations are disabled, try to legalize sample offset.
class DxilLegalizeSampleOffsetPass : public FunctionPass {

  LoopInfo LI;

public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilLegalizeSampleOffsetPass() : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "DXIL legalize sample offset";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DxilValueCache>();
    AU.setPreservesAll();
  }

  bool runOnFunction(Function &F) override {
    DxilModule &DM = F.getParent()->GetOrCreateDxilModule();
    hlsl::OP *hlslOP = DM.GetOP();

    std::vector<Offset> illegalOffsets;

    CollectIllegalOffsets(illegalOffsets, F, hlslOP);

    if (illegalOffsets.empty())
      return false;

    // Loop unroll if has offset inside loop.
    TryUnrollLoop(illegalOffsets, F);

    // Collect offset again after mem2reg.
    std::vector<Offset> ssaIllegalOffsets;
    CollectIllegalOffsets(ssaIllegalOffsets, F, hlslOP);

    // Run simple optimization to legalize offsets.
    LegalizeOffsets(ssaIllegalOffsets);

    // If 6.7 or more, permit remaining "illegal" offsets
    if (DM.GetShaderModel()->IsSM67Plus())
      return true;

    FinalCheck(F, hlslOP);

    return true;
  }

private:
  void TryUnrollLoop(std::vector<Offset> &illegalOffsets, Function &F);
  void CollectIllegalOffsets(std::vector<Offset> &illegalOffsets,
                             Function &F, hlsl::OP *hlslOP);
  void CollectIllegalOffsets(std::vector<Offset> &illegalOffsets,
                             Function &F, DXIL::OpCode opcode,
                             hlsl::OP *hlslOP);
  void LegalizeOffsets(const std::vector<Offset> &illegalOffsets);
  void FinalCheck(Function &F, hlsl::OP *hlslOP);
};

char DxilLegalizeSampleOffsetPass::ID = 0;

bool HasIllegalOffsetInLoop(std::vector<Offset> &illegalOffsets, LoopInfo &LI,
                            Function &F) {
  DominatorTreeAnalysis DTA;
  DominatorTree DT = DTA.run(F);
  LI.Analyze(DT);

  bool findOffset = false;

  for (auto it : illegalOffsets) {
    if (const Instruction *I = dyn_cast<Instruction>(it.offset)) {
      const BasicBlock *BB = I->getParent();
      // TODO: determine whether values are actually loop dependent, not just in a loop
      if (LI.getLoopFor(BB)) {
        findOffset = true;
        break;
      }
    }
  }
  return findOffset;
}

void GetOffsetRange(DXIL::OpCode opcode, unsigned &offsetStart, unsigned &offsetEnd)
{
  if (DXIL::OpCode::TextureLoad == opcode) {
    offsetStart = DXIL::OperandIndex::kTextureLoadOffset0OpIdx;
    offsetEnd = DXIL::OperandIndex::kTextureLoadOffset2OpIdx;
  } else {
    // assume samples
    offsetStart = DXIL::OperandIndex::kTextureSampleOffset0OpIdx;
    offsetEnd = DXIL::OperandIndex::kTextureSampleOffset2OpIdx;
  }
}

void CollectIllegalOffset(CallInst *CI, DXIL::OpCode opcode,
                          std::vector<Offset> &illegalOffsets) {

  unsigned offsetStart = 0, offsetEnd = 0;

  GetOffsetRange(opcode, offsetStart, offsetEnd);

  Value *offset0 =
      CI->getArgOperand(offsetStart);
  // No offsets
  if (isa<UndefValue>(offset0))
    return;

  for (unsigned i = offsetStart; i <= offsetEnd; i++) {
    Value *offset = CI->getArgOperand(i);
    if (Instruction *I = dyn_cast<Instruction>(offset)) {
      Offset offset = {I, CI};
      illegalOffsets.emplace_back(offset);
    }
    else if(ConstantInt *cOffset = dyn_cast<ConstantInt>(offset)) {
      int64_t val = cOffset->getValue().getSExtValue();
      if (val > 7 || val < -8) {
        Offset offset = {cOffset, CI};
        illegalOffsets.emplace_back(offset);
      }
    }
  }
}
}

// Return true if the call instruction in pair a and b are the same
bool InstEq(const Offset &a, const Offset &b) {
  return a.call == b.call;
}

// Return true if the call instruction in pair a is before that in pair b
bool InstLT(const Offset &a, const Offset &b) {
  DebugLoc aLoc = a.call->getDebugLoc();
  DebugLoc bLoc = b.call->getDebugLoc();

  if (aLoc && bLoc) {
    DIScope *aScope = cast<DIScope>(aLoc->getRawScope());
    DIScope *bScope = cast<DIScope>(bLoc->getRawScope());
    std::string aFile = aScope->getFilename();
    std::string bFile = bScope->getFilename();
    return aFile < bFile || (aFile == bFile && aLoc.getLine() < bLoc.getLine());
  }
  // No line numbers, just compare pointers so that matching instructions will be adjacent
  return a.call < b.call;
}

void DxilLegalizeSampleOffsetPass::FinalCheck(Function &F, hlsl::OP *hlslOP) {
  // Collect offset to make sure no illegal offsets.
  std::vector<Offset> finalIllegalOffsets;
  CollectIllegalOffsets(finalIllegalOffsets, F, hlslOP);

  if (!finalIllegalOffsets.empty()) {
    std::string errorMsg = "Offsets to texture access operations must be immediate values. ";

    auto offsetBegin = finalIllegalOffsets.begin();
    auto offsetEnd = finalIllegalOffsets.end();

    std::sort(offsetBegin, offsetEnd, InstLT);
    offsetEnd = std::unique(offsetBegin, offsetEnd, InstEq);

    for (auto it = offsetBegin; it != offsetEnd; it++) {
      CallInst *CI = it->call;
      if (Instruction *offset = dyn_cast<Instruction>(it->offset)) {
        if (LI.getLoopFor(offset->getParent()))
          dxilutil::EmitErrorOnInstruction(CI, errorMsg + "Unrolling the loop containing the offset value"
                                           " manually and using -O3 may help in some cases.\n");
        else
          dxilutil::EmitErrorOnInstruction(CI, errorMsg);
      } else {
        dxilutil::EmitErrorOnInstruction(CI, "Offsets to texture access operations must be between -8 and 7. ");
      }
    }
  }
}

void DxilLegalizeSampleOffsetPass::TryUnrollLoop(
    std::vector<Offset> &illegalOffsets, Function &F) {
  legacy::FunctionPassManager PM(F.getParent());
  // Scalarize aggregates as mem2reg only applies on scalars.
  PM.add(createSROAPass());
  // Always need mem2reg for simplify illegal offsets.
  PM.add(createPromoteMemoryToRegisterPass());

  bool UnrollLoop = HasIllegalOffsetInLoop(illegalOffsets, LI, F);
  if (UnrollLoop) {
    PM.add(createCFGSimplificationPass());
    PM.add(createLCSSAPass());
    PM.add(createLoopSimplifyPass());
    PM.add(createLoopRotatePass());
    PM.add(createLoopUnrollPass(-2, -1, 0, 0));
  }
  PM.run(F);

  if (UnrollLoop) {
    DxilValueCache *DVC = &getAnalysis<DxilValueCache>();
    DVC->ResetUnknowns();
  }
}

void DxilLegalizeSampleOffsetPass::CollectIllegalOffsets(
    std::vector<Offset> &illegalOffsets, Function &CurF,
    hlsl::OP *hlslOP) {
  CollectIllegalOffsets(illegalOffsets, CurF, DXIL::OpCode::Sample, hlslOP);
  CollectIllegalOffsets(illegalOffsets, CurF, DXIL::OpCode::SampleBias, hlslOP);
  CollectIllegalOffsets(illegalOffsets, CurF, DXIL::OpCode::SampleCmp, hlslOP);
  CollectIllegalOffsets(illegalOffsets, CurF, DXIL::OpCode::SampleCmpLevelZero,
                        hlslOP);
  CollectIllegalOffsets(illegalOffsets, CurF, DXIL::OpCode::SampleGrad, hlslOP);
  CollectIllegalOffsets(illegalOffsets, CurF, DXIL::OpCode::SampleLevel,
                        hlslOP);
  CollectIllegalOffsets(illegalOffsets, CurF, DXIL::OpCode::TextureLoad, hlslOP);
}

void DxilLegalizeSampleOffsetPass::CollectIllegalOffsets(
    std::vector<Offset> &illegalOffsets, Function &CurF,
    DXIL::OpCode opcode, hlsl::OP *hlslOP) {
  auto &intrFuncList = hlslOP->GetOpFuncList(opcode);
  for (auto it : intrFuncList) {
    Function *intrFunc = it.second;
    if (!intrFunc)
      continue;
    for (User *U : intrFunc->users()) {
      CallInst *CI = cast<CallInst>(U);
      // Skip inst not in current function.
      if (CI->getParent()->getParent() != &CurF)
        continue;

      CollectIllegalOffset(CI, opcode, illegalOffsets);
    }
  }
}

void DxilLegalizeSampleOffsetPass::LegalizeOffsets(
    const std::vector<Offset> &illegalOffsets) {
  if (!illegalOffsets.empty()) {
    DxilValueCache *DVC = &getAnalysis<DxilValueCache>();
    for (auto it : illegalOffsets) {
      if (Instruction *I = dyn_cast<Instruction>(it.offset))
        if (Value *V = DVC->GetValue(I))
          I->replaceAllUsesWith(V);
    }
  }
}

FunctionPass *llvm::createDxilLegalizeSampleOffsetPass() {
  return new DxilLegalizeSampleOffsetPass();
}

INITIALIZE_PASS_BEGIN(DxilLegalizeSampleOffsetPass, "dxil-legalize-sample-offset",
                "DXIL legalize sample offset", false, false)
INITIALIZE_PASS_DEPENDENCY(DxilValueCache)
INITIALIZE_PASS_END(DxilLegalizeSampleOffsetPass, "dxil-legalize-sample-offset",
                "DXIL legalize sample offset", false, false)
