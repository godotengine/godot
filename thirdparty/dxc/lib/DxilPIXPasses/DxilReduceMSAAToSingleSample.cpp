///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilReduceMSAAToSingleSample.cpp                                          //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides a pass to reduce all MSAA writes to single-sample writes         //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilOperations.h"

#include "dxc/DXIL/DxilInstructions.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DxilPIXPasses/DxilPIXPasses.h"
#include "dxc/HLSL/DxilGenerationPass.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"

using namespace llvm;
using namespace hlsl;

class DxilReduceMSAAToSingleSample : public ModulePass {

public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilReduceMSAAToSingleSample() : ModulePass(ID) {}
  StringRef getPassName() const override {
    return "HLSL DXIL Reduce all MSAA reads to single-sample reads";
  }
  bool runOnModule(Module &M) override;
};

bool DxilReduceMSAAToSingleSample::runOnModule(Module &M) {
  DxilModule &DM = M.GetOrCreateDxilModule();

  LLVMContext &Ctx = M.getContext();
  OP *HlslOP = DM.GetOP();

  // FP16 type doesn't have its own identity, and is covered by float type...

  auto TextureLoadOverloads = std::vector<Type *>{
      Type::getFloatTy(Ctx), Type::getInt16Ty(Ctx), Type::getInt32Ty(Ctx)};

  bool Modified = false;

  for (const auto &Overload : TextureLoadOverloads) {

    Function *TexLoadFunction =
        HlslOP->GetOpFunc(DXIL::OpCode::TextureLoad, Overload);
    auto TexLoadFunctionUses = TexLoadFunction->uses();

    for (auto FI = TexLoadFunctionUses.begin();
         FI != TexLoadFunctionUses.end();) {
      auto &FunctionUse = *FI++;
      auto FunctionUser = FunctionUse.getUser();
      auto instruction = cast<Instruction>(FunctionUser);
      DxilInst_TextureLoad LoadInstruction(instruction);
      auto TextureHandle = LoadInstruction.get_srv();
      auto TextureHandleInst = cast<CallInst>(TextureHandle);
      DxilInst_CreateHandle createHandle(TextureHandleInst);
      // Dynamic rangeId is not supported
      if (isa<ConstantInt>(createHandle.get_rangeId())) {
        unsigned rangeId =
            cast<ConstantInt>(createHandle.get_rangeId())->getLimitedValue();
        if (static_cast<DXIL::ResourceClass>(
                createHandle.get_resourceClass_val()) ==
            DXIL::ResourceClass::SRV) {
          auto Resource = DM.GetSRV(rangeId);
          if (Resource.GetKind() == DXIL::ResourceKind::Texture2DMS ||
              Resource.GetKind() == DXIL::ResourceKind::Texture2DMSArray) {
            // "2" is the mip-level/sample-index operand index:
            // https://github.com/Microsoft/DirectXShaderCompiler/blob/master/docs/DXIL.rst#textureload
            instruction->setOperand(2, HlslOP->GetI32Const(0));
            Modified = true;
          }
        }
      }
    }
  }

  return Modified;
}

char DxilReduceMSAAToSingleSample::ID = 0;

ModulePass *llvm::createDxilReduceMSAAToSingleSamplePass() {
  return new DxilReduceMSAAToSingleSample();
}

INITIALIZE_PASS(DxilReduceMSAAToSingleSample, "hlsl-dxil-reduce-msaa-to-single",
                "HLSL DXIL Reduce all MSAA writes to single-sample writes",
                false, false)
