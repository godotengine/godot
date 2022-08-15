///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilEliminateOutputDynamicIndexing.cpp                                    //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Eliminate dynamic indexing on output.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/DxilGenerationPass.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilSignatureElement.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilUtil.h"
#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilInstructions.h"

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/ADT/MapVector.h"

using namespace llvm;
using namespace hlsl;

namespace {
class DxilEliminateOutputDynamicIndexing : public ModulePass {
private:

public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilEliminateOutputDynamicIndexing() : ModulePass(ID) {}

  StringRef getPassName() const override {
    return "DXIL eliminate output dynamic indexing";
  }

  bool runOnModule(Module &M) override {
    DxilModule &DM = M.GetOrCreateDxilModule();
    bool bUpdated = false;
    if (DM.GetShaderModel()->IsHS()) {
      // HS write outputs into share memory, dynamic indexing is OK.
      return bUpdated;
    }

    // Skip pass thru entry.
    if (!DM.GetEntryFunction())
      return bUpdated;

    hlsl::OP *hlslOP = DM.GetOP();

    bUpdated |=
        EliminateDynamicOutput(hlslOP, DXIL::OpCode::StoreOutput,
                               DM.GetOutputSignature(), DM.GetEntryFunction());

    return bUpdated;
  }

private:
  bool EliminateDynamicOutput(hlsl::OP *hlslOP, DXIL::OpCode opcode, DxilSignature &outputSig, Function *Entry);
  void ReplaceDynamicOutput(ArrayRef<Value *> tmpSigElts, Value * sigID, Value *zero, Function *F);
  void StoreTmpSigToOutput(ArrayRef<Value *> tmpSigElts, unsigned row,
                           Value *opcode, Value *sigID, Function *StoreOutput,
                           Function *Entry);
};

// Wrapper for StoreOutput and StorePachConstant which has same signature.
// void (opcode, sigId, rowIndex, colIndex, value);
class DxilOutputStore {
public:
  const llvm::CallInst *Instr;
  // Construction and identification
  DxilOutputStore(llvm::CallInst *pInstr) : Instr(pInstr) {}
  // Validation support
  bool isAllowed() const { return true; }
  bool isArgumentListValid() const {
    if (5 != llvm::dyn_cast<llvm::CallInst>(Instr)->getNumArgOperands())
      return false;
    return true;
  }
  // Accessors
  llvm::Value *get_outputSigId() const {
    return Instr->getOperand(DXIL::OperandIndex::kStoreOutputIDOpIdx);
  }
  llvm::Value *get_rowIndex() const {
    return Instr->getOperand(DXIL::OperandIndex::kStoreOutputRowOpIdx);
  }
  uint64_t get_colIndex() const {
    Value *col = Instr->getOperand(DXIL::OperandIndex::kStoreOutputColOpIdx);
    return cast<ConstantInt>(col)->getLimitedValue();
  }
  llvm::Value *get_value() const {
    return Instr->getOperand(DXIL::OperandIndex::kStoreOutputValOpIdx);
  }
};

bool DxilEliminateOutputDynamicIndexing::EliminateDynamicOutput(
    hlsl::OP *hlslOP, DXIL::OpCode opcode, DxilSignature &outputSig,
    Function *Entry) {
  auto &storeOutputs =
      hlslOP->GetOpFuncList(opcode);

  MapVector<Value *, Type *> dynamicSigSet;
  for (auto it : storeOutputs) {
    Function *F = it.second;
    // Skip overload not used.
    if (!F)
      continue;
    for (User *U : F->users()) {
      CallInst *CI = cast<CallInst>(U);
      DxilOutputStore store(CI);
      // Save dynamic indeed sigID.
      if (!isa<ConstantInt>(store.get_rowIndex())) {
        Value *sigID = store.get_outputSigId();
        dynamicSigSet[sigID] = store.get_value()->getType();
      }
    }
  }

  if (dynamicSigSet.empty())
    return false;

  IRBuilder<> AllocaBuilder(dxilutil::FindAllocaInsertionPt(Entry));

  Value *opcodeV = AllocaBuilder.getInt32(static_cast<unsigned>(opcode));
  Value *zero = AllocaBuilder.getInt32(0);

  for (auto sig : dynamicSigSet) {
    Value *sigID = sig.first;
    Type *EltTy = sig.second;
    unsigned ID = cast<ConstantInt>(sigID)->getLimitedValue();
    DxilSignatureElement &sigElt = outputSig.GetElement(ID);
    unsigned row = sigElt.GetRows();
    unsigned col = sigElt.GetCols();
    Type *AT = ArrayType::get(EltTy, row);

    std::vector<Value *> tmpSigElts(col);
    for (unsigned c = 0; c < col; c++) {
      Value *newCol = AllocaBuilder.CreateAlloca(AT);
      tmpSigElts[c] = newCol;
    }

    Function *F = hlslOP->GetOpFunc(opcode, EltTy);
    // Change store output to store tmpSigElts.
    ReplaceDynamicOutput(tmpSigElts, sigID, zero, F);
    // Store tmpSigElts to Output before return.
    StoreTmpSigToOutput(tmpSigElts, row, opcodeV, sigID, F, Entry);
  }
  return true;
}

void DxilEliminateOutputDynamicIndexing::ReplaceDynamicOutput(
    ArrayRef<Value *> tmpSigElts, Value *sigID, Value *zero, Function *F) {
  for (auto it = F->user_begin(); it != F->user_end();) {
    CallInst *CI = cast<CallInst>(*(it++));
    DxilOutputStore store(CI);
    if (sigID == store.get_outputSigId()) {
      uint64_t col = store.get_colIndex();
      Value *tmpSigElt = tmpSigElts[col];
      IRBuilder<> Builder(CI);
      Value *r = store.get_rowIndex();
      // Store to tmpSigElt.
      Value *GEP = Builder.CreateInBoundsGEP(tmpSigElt, {zero, r});
      Builder.CreateStore(store.get_value(), GEP);
      // Remove store output.
      CI->eraseFromParent();
    }
  }
}

void DxilEliminateOutputDynamicIndexing::StoreTmpSigToOutput(
    ArrayRef<Value *> tmpSigElts, unsigned row, Value *opcode, Value *sigID,
    Function *StoreOutput, Function *Entry) {
  Value *args[] = {opcode, sigID, /*row*/ nullptr, /*col*/ nullptr,
                   /*val*/ nullptr};
  // Store the tmpSigElts to Output before every return.
  for (auto &BB : Entry->getBasicBlockList()) {
    if (ReturnInst *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
      IRBuilder<> Builder(RI);
      Value *zero = Builder.getInt32(0);
      for (unsigned c = 0; c<tmpSigElts.size(); c++) {
        Value *col = tmpSigElts[c];
        args[DXIL::OperandIndex::kStoreOutputColOpIdx] = Builder.getInt8(c);
        for (unsigned r = 0; r < row; r++) {
          Value *GEP =
              Builder.CreateInBoundsGEP(col, {zero, Builder.getInt32(r)});
          Value *V = Builder.CreateLoad(GEP);
          args[DXIL::OperandIndex::kStoreOutputRowOpIdx] = Builder.getInt32(r);
          args[DXIL::OperandIndex::kStoreOutputValOpIdx] = V;
          Builder.CreateCall(StoreOutput, args);
        }
      }
    }
  }
}

}

char DxilEliminateOutputDynamicIndexing::ID = 0;

ModulePass *llvm::createDxilEliminateOutputDynamicIndexingPass() {
  return new DxilEliminateOutputDynamicIndexing();
}

INITIALIZE_PASS(DxilEliminateOutputDynamicIndexing,
                "hlsl-dxil-eliminate-output-dynamic",
                "DXIL eliminate output dynamic indexing", false, false)
