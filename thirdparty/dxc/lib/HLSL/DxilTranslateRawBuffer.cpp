///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilTranslateRawBuffer.cpp                                                //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilUtil.h"
#include "dxc/HLSL/DxilGenerationPass.h"
#include "dxc/Support/Global.h"
#include "llvm/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include <vector>

using namespace llvm;
using namespace hlsl;

// Translate RawBufferLoad/RawBufferStore
// This pass is to make sure that we generate correct buffer load for DXIL
// For DXIL < 1.2, rawBufferLoad will be translated to BufferLoad instruction
// without mask.

namespace {

class DxilTranslateRawBuffer : public ModulePass {
public:
  static char ID;
  explicit DxilTranslateRawBuffer() : ModulePass(ID) {}
  bool runOnModule(Module &M) {
    unsigned major, minor;
    DxilModule &DM = M.GetDxilModule();
    DM.GetDxilVersion(major, minor);
    OP *hlslOP = DM.GetOP();
    // Split 64bit for shader model less than 6.3.
    if (major == 1 && minor <= 2) {
      for (auto F = M.functions().begin(); F != M.functions().end();) {
        Function *func = &*(F++);
        DXIL::OpCodeClass opClass;
        if (hlslOP->GetOpCodeClass(func, opClass)) {
          if (opClass == DXIL::OpCodeClass::RawBufferLoad) {
            Type *ETy =
                OP::GetOverloadType(DXIL::OpCode::RawBufferLoad, func);

            bool is64 =
                ETy->isDoubleTy() || ETy == Type::getInt64Ty(ETy->getContext());
            if (is64) {
              ReplaceRawBufferLoad64Bit(func, ETy, M);
              func->eraseFromParent();
            }
          } else if (opClass == DXIL::OpCodeClass::RawBufferStore) {
            Type *ETy =
                OP::GetOverloadType(DXIL::OpCode::RawBufferStore, func);

            bool is64 =
                ETy->isDoubleTy() || ETy == Type::getInt64Ty(ETy->getContext());
            if (is64) {
              ReplaceRawBufferStore64Bit(func, ETy, M);
              func->eraseFromParent();
            }
          }
        }
      }
    }
    if (major == 1 && minor < 2) {
      for (auto F = M.functions().begin(), E = M.functions().end(); F != E;) {
        Function *func = &*(F++);
        if (func->hasName()) {
          if (func->getName().startswith("dx.op.rawBufferLoad")) {
            ReplaceRawBufferLoad(func, M);
            func->eraseFromParent();
          } else if (func->getName().startswith("dx.op.rawBufferStore")) {
            ReplaceRawBufferStore(func, M);
            func->eraseFromParent();
          }
        }
      }
    }
    return true;
  }

private:
  // Replace RawBufferLoad/Store to BufferLoad/Store for DXIL < 1.2
  void ReplaceRawBufferLoad(Function *F, Module &M);
  void ReplaceRawBufferStore(Function *F, Module &M);
  void ReplaceRawBufferLoad64Bit(Function *F, Type *EltTy, Module &M);
  void ReplaceRawBufferStore64Bit(Function *F, Type *EltTy, Module &M);
};
} // namespace

void DxilTranslateRawBuffer::ReplaceRawBufferLoad(Function *F,
                                                  Module &M) {
  dxilutil::ReplaceRawBufferLoadWithBufferLoad(F, M.GetDxilModule().GetOP());
}

void DxilTranslateRawBuffer::ReplaceRawBufferLoad64Bit(Function *F, Type *EltTy, Module &M) {
  dxilutil::ReplaceRawBufferLoad64Bit(F, EltTy, M.GetDxilModule().GetOP());
}

void DxilTranslateRawBuffer::ReplaceRawBufferStore(Function *F,
  Module &M) {
  dxilutil::ReplaceRawBufferStoreWithBufferStore(F, M.GetDxilModule().GetOP());
}

void DxilTranslateRawBuffer::ReplaceRawBufferStore64Bit(Function *F, Type *ETy,
                                                        Module &M) {
  dxilutil::ReplaceRawBufferStore64Bit(F, ETy, M.GetDxilModule().GetOP());
}

char DxilTranslateRawBuffer::ID = 0;
ModulePass *llvm::createDxilTranslateRawBuffer() {
  return new DxilTranslateRawBuffer();
}

INITIALIZE_PASS(DxilTranslateRawBuffer, "hlsl-translate-dxil-raw-buffer",
                "Translate raw buffer load", false, false)
