///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilAddPixelHitInstrumentation.cpp                                        //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides a pass to add instrumentation to determine pixel hit count and   //
// cost. Used by PIX.                                                        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilOperations.h"

#include "dxc/DXIL/DxilInstructions.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilUtil.h"
#include "dxc/DxilPIXPasses/DxilPIXPasses.h"
#include "dxc/HLSL/DxilGenerationPass.h"

#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Utils/Local.h"

#include "PixPassHelpers.h"

using namespace llvm;
using namespace hlsl;

class DxilAddPixelHitInstrumentation : public ModulePass {

  bool ForceEarlyZ = false;
  bool AddPixelCost = false;
  int RTWidth = 1024;
  int NumPixels = 128;
  int SVPositionIndex = -1;

public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilAddPixelHitInstrumentation() : ModulePass(ID) {}
  StringRef getPassName() const override { return "DXIL Constant Color Mod"; }
  void applyOptions(PassOptions O) override;
  bool runOnModule(Module &M) override;
};

void DxilAddPixelHitInstrumentation::applyOptions(PassOptions O) {
  GetPassOptionBool(O, "force-early-z", &ForceEarlyZ, false);
  GetPassOptionBool(O, "add-pixel-cost", &AddPixelCost, false);
  GetPassOptionInt(O, "rt-width", &RTWidth, 0);
  GetPassOptionInt(O, "num-pixels", &NumPixels, 0);
  GetPassOptionInt(O, "sv-position-index", &SVPositionIndex, 0);
}

bool DxilAddPixelHitInstrumentation::runOnModule(Module &M) {
  // This pass adds instrumentation for pixel hit counting and pixel cost.

  DxilModule &DM = M.GetOrCreateDxilModule();
  LLVMContext &Ctx = M.getContext();
  OP *HlslOP = DM.GetOP();

  // ForceEarlyZ is incompatible with the discard function (the Z has to be
  // tested/written, and may be written before the shader even runs)
  if (ForceEarlyZ) {
    DM.m_ShaderFlags.SetForceEarlyDepthStencil(true);
  }

  hlsl::DxilSignature &InputSignature = DM.GetInputSignature();

  auto &InputElements = InputSignature.GetElements();

  unsigned SV_Position_ID;

  auto SV_Position =
      std::find_if(InputElements.begin(), InputElements.end(),
                   [](const std::unique_ptr<DxilSignatureElement> &Element) {
                     return Element->GetSemantic()->GetKind() ==
                            hlsl::DXIL::SemanticKind::Position;
                   });

  // SV_Position, if present, has to have full mask, so we needn't worry
  // about the shader having selected components that don't include x or y.
  // If not present, we add it.
  if (SV_Position == InputElements.end()) {
    auto SVPosition =
        llvm::make_unique<DxilSignatureElement>(DXIL::SigPointKind::PSIn);
    SVPosition->Initialize("Position", hlsl::CompType::getF32(),
                           hlsl::DXIL::InterpolationMode::Linear, 1, 4,
                           SVPositionIndex == -1 ? 0 : SVPositionIndex, 0);
    SVPosition->AppendSemanticIndex(0);
    SVPosition->SetSigPointKind(DXIL::SigPointKind::PSIn);
    SVPosition->SetKind(hlsl::DXIL::SemanticKind::Position);

    auto index = InputSignature.AppendElement(std::move(SVPosition));
    SV_Position_ID = InputElements[index]->GetID();
  } else {
    SV_Position_ID = SV_Position->get()->GetID();
  }

  auto EntryPointFunction = PIXPassHelpers::GetEntryFunction(DM);

  auto &EntryBlock = EntryPointFunction->getEntryBlock();

  CallInst *HandleForUAV;
  {
    IRBuilder<> Builder(
        dxilutil::FirstNonAllocaInsertionPt(PIXPassHelpers::GetEntryFunction(DM)));

    HandleForUAV = PIXPassHelpers::CreateUAV(DM, Builder, 0, "PIX_CountUAV_Handle");

    DM.ReEmitDxilResources();
  }
  // todo: is it a reasonable assumption that there will be a "Ret" in the entry
  // block, and that these are the only points from which the shader can exit
  // (except for a pixel-kill?)
  auto &Instructions = EntryBlock.getInstList();
  auto It = Instructions.begin();
  while (It != Instructions.end()) {
    auto ThisInstruction = It++;
    LlvmInst_Ret Ret(ThisInstruction);
    if (Ret) {
      // Check that there is at least one instruction preceding the Ret (no need
      // to instrument it if there isn't)
      if (ThisInstruction->getPrevNode() != nullptr) {

        // Start adding instructions right before the Ret:
        IRBuilder<> Builder(ThisInstruction);

        // ------------------------------------------------------------------------------------------------------------
        // Generate instructions to increment (by one) a UAV value corresponding
        // to the pixel currently being rendered
        // ------------------------------------------------------------------------------------------------------------

        // Useful constants
        Constant *Zero32Arg = HlslOP->GetU32Const(0);
        Constant *Zero8Arg = HlslOP->GetI8Const(0);
        Constant *One32Arg = HlslOP->GetU32Const(1);
        Constant *One8Arg = HlslOP->GetI8Const(1);
        UndefValue *UndefArg = UndefValue::get(Type::getInt32Ty(Ctx));
        Constant *NumPixelsByteOffsetArg = HlslOP->GetU32Const(NumPixels * 4);

        // Step 1: Convert SV_POSITION to UINT
        Value *XAsInt;
        Value *YAsInt;
        {
          auto LoadInputOpFunc =
              HlslOP->GetOpFunc(DXIL::OpCode::LoadInput, Type::getFloatTy(Ctx));
          Constant *LoadInputOpcode =
              HlslOP->GetU32Const((unsigned)DXIL::OpCode::LoadInput);
          Constant *SV_Pos_ID = HlslOP->GetU32Const(SV_Position_ID);
          auto XPos =
              Builder.CreateCall(LoadInputOpFunc,
                                 {LoadInputOpcode, SV_Pos_ID, Zero32Arg /*row*/,
                                  Zero8Arg /*column*/, UndefArg},
                                 "XPos");
          auto YPos =
              Builder.CreateCall(LoadInputOpFunc,
                                 {LoadInputOpcode, SV_Pos_ID, Zero32Arg /*row*/,
                                  One8Arg /*column*/, UndefArg},
                                 "YPos");

          XAsInt = Builder.CreateCast(Instruction::CastOps::FPToUI, XPos,
                                      Type::getInt32Ty(Ctx), "XIndex");
          YAsInt = Builder.CreateCast(Instruction::CastOps::FPToUI, YPos,
                                      Type::getInt32Ty(Ctx), "YIndex");
        }

        // Step 2: Calculate pixel index
        Value *Index;
        {
          Constant *RTWidthArg = HlslOP->GetI32Const(RTWidth);
          auto YOffset = Builder.CreateMul(YAsInt, RTWidthArg, "YOffset");
          auto Elementoffset =
              Builder.CreateAdd(XAsInt, YOffset, "ElementOffset");
          Index = Builder.CreateMul(Elementoffset, HlslOP->GetU32Const(4),
                                    "ByteIndex");
        }

        // Insert the UAV increment instruction:
        Function *AtomicOpFunc =
            HlslOP->GetOpFunc(OP::OpCode::AtomicBinOp, Type::getInt32Ty(Ctx));
        Constant *AtomicBinOpcode =
            HlslOP->GetU32Const((unsigned)OP::OpCode::AtomicBinOp);
        Constant *AtomicAdd =
            HlslOP->GetU32Const((unsigned)DXIL::AtomicBinOpCode::Add);
        {
          (void)Builder.CreateCall(
              AtomicOpFunc,
              {
                  AtomicBinOpcode, // i32, ; opcode
                  HandleForUAV,    // %dx.types.Handle, ; resource handle
                  AtomicAdd, // i32, ; binary operation code : EXCHANGE, IADD,
                             // AND, OR, XOR, IMIN, IMAX, UMIN, UMAX
                  Index,     // i32, ; coordinate c0: byte offset
                  UndefArg,  // i32, ; coordinate c1 (unused)
                  UndefArg,  // i32, ; coordinate c2 (unused)
                  One32Arg   // i32); increment value
              },
              "UAVIncResult");
        }

        if (AddPixelCost) {
          // ------------------------------------------------------------------------------------------------------------
          // Generate instructions to increment a value corresponding to the
          // current pixel in the second half of the UAV, by an amount
          // proportional to the estimated average cost of each pixel in the
          // current draw call.
          // ------------------------------------------------------------------------------------------------------------

          // Step 1: Retrieve weight value from UAV; it will be placed after the
          // range we're writing to
          Value *Weight;
          {
            Function *LoadWeight = HlslOP->GetOpFunc(OP::OpCode::BufferLoad,
                                                     Type::getInt32Ty(Ctx));
            Constant *LoadWeightOpcode =
                HlslOP->GetU32Const((unsigned)DXIL::OpCode::BufferLoad);
            Constant *OffsetIntoUAV = HlslOP->GetU32Const(NumPixels * 2 * 4);
            auto WeightStruct = Builder.CreateCall(
                LoadWeight,
                {
                    LoadWeightOpcode, // i32 opcode
                    HandleForUAV,     // %dx.types.Handle, ; resource handle
                    OffsetIntoUAV,    // i32 c0: byte offset
                    UndefArg          // i32 c1: unused
                },
                "WeightStruct");
            Weight = Builder.CreateExtractValue(
                WeightStruct, static_cast<uint64_t>(0LL), "Weight");
          }

          // Step 2: Update write position ("Index") to second half of the UAV
          auto OffsetIndex = Builder.CreateAdd(Index, NumPixelsByteOffsetArg,
                                               "OffsetByteIndex");

          // Step 3: Increment UAV value by the weight
          (void)Builder.CreateCall(
              AtomicOpFunc,
              {
                  AtomicBinOpcode, // i32, ; opcode
                  HandleForUAV,    // %dx.types.Handle, ; resource handle
                  AtomicAdd,   // i32, ; binary operation code : EXCHANGE, IADD,
                               // AND, OR, XOR, IMIN, IMAX, UMIN, UMAX
                  OffsetIndex, // i32, ; coordinate c0: byte offset
                  UndefArg,    // i32, ; coordinate c1 (unused)
                  UndefArg,    // i32, ; coordinate c2 (unused)
                  Weight       // i32); increment value
              },
              "UAVIncResult2");
        }
      }
    }
  }

  bool Modified = false;

  return Modified;
}

char DxilAddPixelHitInstrumentation::ID = 0;

ModulePass *llvm::createDxilAddPixelHitInstrumentationPass() {
  return new DxilAddPixelHitInstrumentation();
}

INITIALIZE_PASS(DxilAddPixelHitInstrumentation,
                "hlsl-dxil-add-pixel-hit-instrmentation",
                "DXIL Count completed PS invocations and costs", false, false)
