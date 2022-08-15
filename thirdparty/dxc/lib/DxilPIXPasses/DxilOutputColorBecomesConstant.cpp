///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilOutputColorBecomesConstant.cpp                                        //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides a pass to stomp a pixel shader's output color to a given         //
// constant value                                                            //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DxilPIXPasses/DxilPIXPasses.h"
#include "dxc/HLSL/DxilGenerationPass.h"
#include "dxc/HLSL/DxilSpanAllocator.h"

#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Utils/Local.h"
#include <array>

#include "PixPassHelpers.h"

using namespace llvm;
using namespace hlsl;

class DxilOutputColorBecomesConstant : public ModulePass {

  enum VisualizerInstrumentationMode {
    FromLiteralConstant,
    FromConstantBuffer
  };

  float Red = 1.f;
  float Green = 1.f;
  float Blue = 1.f;
  float Alpha = 1.f;
  VisualizerInstrumentationMode Mode = FromLiteralConstant;

  void visitOutputInstructionCallers(Function *OutputFunction,
                                     const hlsl::DxilSignature &OutputSignature,
                                     OP *HlslOP,
                                     std::function<void(CallInst *)> Visitor);

public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilOutputColorBecomesConstant() : ModulePass(ID) {}
  StringRef getPassName() const override { return "DXIL Constant Color Mod"; }
  void applyOptions(PassOptions O) override;
  bool runOnModule(Module &M) override;
};

void DxilOutputColorBecomesConstant::applyOptions(PassOptions O) {
  GetPassOptionFloat(O, "constant-red", &Red, 1.f);
  GetPassOptionFloat(O, "constant-green", &Green, 1.f);
  GetPassOptionFloat(O, "constant-blue", &Blue, 1.f);
  GetPassOptionFloat(O, "constant-alpha", &Alpha, 1.f);

  int mode = 0;
  GetPassOptionInt(O, "mod-mode", &mode, 0);
  Mode = static_cast<VisualizerInstrumentationMode>(mode);
}

void DxilOutputColorBecomesConstant::visitOutputInstructionCallers(
    Function *OutputFunction, const hlsl::DxilSignature &OutputSignature,
    OP *HlslOP, std::function<void(CallInst *)> Visitor) {

  auto OutputFunctionUses = OutputFunction->uses();

  for (Use &FunctionUse : OutputFunctionUses) {
    iterator_range<Value::user_iterator> FunctionUsers = FunctionUse->users();
    for (User *FunctionUser : FunctionUsers) {
      if (isa<Instruction>(FunctionUser)) {
        auto CallInstruction = cast<CallInst>(FunctionUser);

        // Check if the instruction writes to a render target (as opposed to a
        // system-value, such as RenderTargetArrayIndex)
        Value *OutputID = CallInstruction->getArgOperand(
            DXIL::OperandIndex::kStoreOutputIDOpIdx);
        unsigned SignatureElementIndex =
            cast<ConstantInt>(OutputID)->getLimitedValue();
        const DxilSignatureElement &SignatureElement =
            OutputSignature.GetElement(SignatureElementIndex);

        // We only modify the output color for RTV0
        if (SignatureElement.GetSemantic()->GetKind() ==
                DXIL::SemanticKind::Target &&
            SignatureElement.GetSemanticStartIndex() == 0) {

          // Replace the source operand with the appropriate constant value
          Visitor(CallInstruction);
        }
      }
    }
  }
}

bool DxilOutputColorBecomesConstant::runOnModule(Module &M) {
  // This pass finds all users of the "StoreOutput" function, and replaces their
  // source operands with a constant value.

  DxilModule &DM = M.GetOrCreateDxilModule();

  LLVMContext &Ctx = M.getContext();

  OP *HlslOP = DM.GetOP();

  const hlsl::DxilSignature &OutputSignature = DM.GetOutputSignature();

  Function *FloatOutputFunction =
      HlslOP->GetOpFunc(DXIL::OpCode::StoreOutput, Type::getFloatTy(Ctx));
  Function *IntOutputFunction =
      HlslOP->GetOpFunc(DXIL::OpCode::StoreOutput, Type::getInt32Ty(Ctx));

  bool hasFloatOutputs = false;
  bool hasIntOutputs = false;

  visitOutputInstructionCallers(
      FloatOutputFunction, OutputSignature, HlslOP,
      [&hasFloatOutputs](CallInst *) { hasFloatOutputs = true; });

  visitOutputInstructionCallers(
      IntOutputFunction, OutputSignature, HlslOP,
      [&hasIntOutputs](CallInst *) { hasIntOutputs = true; });

  if (!hasFloatOutputs && !hasIntOutputs) {
    return false;
  }

  // Otherwise, we assume the shader outputs only one or the other (because the
  // 0th RTV can't have a mixed type)
  DXASSERT(!hasFloatOutputs || !hasIntOutputs,
           "Only one or the other type of output: float or int");

  std::array<llvm::Value *, 4> ReplacementColors;

  switch (Mode) {
  case FromLiteralConstant: {
    if (hasFloatOutputs) {
      ReplacementColors[0] = HlslOP->GetFloatConst(Red);
      ReplacementColors[1] = HlslOP->GetFloatConst(Green);
      ReplacementColors[2] = HlslOP->GetFloatConst(Blue);
      ReplacementColors[3] = HlslOP->GetFloatConst(Alpha);
    }
    if (hasIntOutputs) {
      ReplacementColors[0] = HlslOP->GetI32Const(static_cast<int>(Red));
      ReplacementColors[1] = HlslOP->GetI32Const(static_cast<int>(Green));
      ReplacementColors[2] = HlslOP->GetI32Const(static_cast<int>(Blue));
      ReplacementColors[3] = HlslOP->GetI32Const(static_cast<int>(Alpha));
    }
  } break;
  case FromConstantBuffer: {

    // Setup a constant buffer with a single float4 in it:
    SmallVector<llvm::Type *, 4> Elements{
        Type::getFloatTy(Ctx), Type::getFloatTy(Ctx), Type::getFloatTy(Ctx),
        Type::getFloatTy(Ctx)};
    llvm::StructType *CBStructTy =
        llvm::StructType::create(Elements, "PIX_ConstantColorCB_Type");
    std::unique_ptr<DxilCBuffer> pCBuf = llvm::make_unique<DxilCBuffer>();
    pCBuf->SetGlobalName("PIX_ConstantColorCBName");
    pCBuf->SetGlobalSymbol(UndefValue::get(CBStructTy));
    pCBuf->SetID(0);
    pCBuf->SetSpaceID(
        (unsigned int)-2); // This is the reserved-for-tools register space
    pCBuf->SetLowerBound(0);
    pCBuf->SetRangeSize(1);
    pCBuf->SetSize(4);

    Instruction *entryPointInstruction =
        &*(PIXPassHelpers::GetEntryFunction(DM)->begin()->begin());
    IRBuilder<> Builder(entryPointInstruction);

    // Create handle for the newly-added constant buffer (which is achieved via
    // a function call)
    auto ConstantBufferName = "PIX_Constant_Color_CB_Handle";

    CallInst* callCreateHandle = PIXPassHelpers::CreateHandleForResource(DM, Builder, pCBuf.get(), ConstantBufferName);

    DM.AddCBuffer(std::move(pCBuf));

    DM.ReEmitDxilResources();

#define PIX_CONSTANT_VALUE "PIX_Constant_Color_Value"

    // Insert the Buffer load instruction:
    Function *CBLoad = HlslOP->GetOpFunc(
        OP::OpCode::CBufferLoadLegacy,
        hasFloatOutputs ? Type::getFloatTy(Ctx) : Type::getInt32Ty(Ctx));
    Constant *OpArg =
        HlslOP->GetU32Const((unsigned)OP::OpCode::CBufferLoadLegacy);
    Value *ResourceHandle = callCreateHandle;
    Constant *RowIndex = HlslOP->GetU32Const(0);
    CallInst *loadLegacy = Builder.CreateCall(
        CBLoad, {OpArg, ResourceHandle, RowIndex}, PIX_CONSTANT_VALUE);

    // Now extract four color values:
    ReplacementColors[0] =
        Builder.CreateExtractValue(loadLegacy, 0, PIX_CONSTANT_VALUE "0");
    ReplacementColors[1] =
        Builder.CreateExtractValue(loadLegacy, 1, PIX_CONSTANT_VALUE "1");
    ReplacementColors[2] =
        Builder.CreateExtractValue(loadLegacy, 2, PIX_CONSTANT_VALUE "2");
    ReplacementColors[3] =
        Builder.CreateExtractValue(loadLegacy, 3, PIX_CONSTANT_VALUE "3");
  } break;
  default:
    assert(false);
    return 0;
  }

  bool Modified = false;

  // The StoreOutput function can store either a float or an integer, depending
  // on the intended output render-target resource view.
  if (hasFloatOutputs) {
    visitOutputInstructionCallers(
        FloatOutputFunction, OutputSignature, HlslOP,
        [&ReplacementColors, &Modified](CallInst *CallInstruction) {
          Modified = true;
          // The output column is the channel (red, green, blue or alpha) within
          // the output pixel
          Value *OutputColumnOperand = CallInstruction->getOperand(
              hlsl::DXIL::OperandIndex::kStoreOutputColOpIdx);
          ConstantInt *OutputColumnConstant =
              cast<ConstantInt>(OutputColumnOperand);
          APInt OutputColumn = OutputColumnConstant->getValue();
          CallInstruction->setOperand(
              hlsl::DXIL::OperandIndex::kStoreOutputValOpIdx,
              ReplacementColors[*OutputColumn.getRawData()]);
        });
  }

  if (hasIntOutputs) {
    visitOutputInstructionCallers(
        IntOutputFunction, OutputSignature, HlslOP,
        [&ReplacementColors, &Modified](CallInst *CallInstruction) {
          Modified = true;
          // The output column is the channel (red, green, blue or alpha) within
          // the output pixel
          Value *OutputColumnOperand = CallInstruction->getOperand(
              hlsl::DXIL::OperandIndex::kStoreOutputColOpIdx);
          ConstantInt *OutputColumnConstant =
              cast<ConstantInt>(OutputColumnOperand);
          APInt OutputColumn = OutputColumnConstant->getValue();
          CallInstruction->setOperand(
              hlsl::DXIL::OperandIndex::kStoreOutputValOpIdx,
              ReplacementColors[*OutputColumn.getRawData()]);
        });
  }

  return Modified;
}

char DxilOutputColorBecomesConstant::ID = 0;

ModulePass *llvm::createDxilOutputColorBecomesConstantPass() {
  return new DxilOutputColorBecomesConstant();
}

INITIALIZE_PASS(DxilOutputColorBecomesConstant, "hlsl-dxil-constantColor",
                "DXIL Constant Color Mod", false, false)
