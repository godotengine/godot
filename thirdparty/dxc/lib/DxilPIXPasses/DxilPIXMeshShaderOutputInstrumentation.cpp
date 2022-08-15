///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilAddPixelHitInstrumentation.cpp                                        //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides a pass to add instrumentation to retrieve mesh shader output.    //
// Used by PIX.                                                              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilUtil.h"

#include "dxc/DXIL/DxilInstructions.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DxilPIXPasses/DxilPIXPasses.h"
#include "dxc/HLSL/DxilGenerationPass.h"
#include "dxc/HLSL/DxilSpanAllocator.h"

#include "llvm/IR/InstIterator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Transforms/Utils/Local.h"
#include <deque>

#ifdef _WIN32
#include <winerror.h>
#endif

#include "PixPassHelpers.h"

// Keep these in sync with the same-named value in the debugger application's
// WinPixShaderUtils.h

constexpr uint64_t DebugBufferDumpingGroundSize = 64 * 1024;
// The actual max size per record is much smaller than this, but it never
// hurts to be generous.
constexpr size_t CounterOffsetBeyondUsefulData = DebugBufferDumpingGroundSize / 2;

// Keep these in sync with the same-named values in PIX's MeshShaderOutput.cpp
constexpr uint32_t triangleIndexIndicator = 0x1;
constexpr uint32_t int32ValueIndicator = 0x2;
constexpr uint32_t floatValueIndicator = 0x3;
constexpr uint32_t int16ValueIndicator = 0x4;
constexpr uint32_t float16ValueIndicator = 0x5;

using namespace llvm;
using namespace hlsl;
using namespace PIXPassHelpers;

class DxilPIXMeshShaderOutputInstrumentation : public ModulePass 
{
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilPIXMeshShaderOutputInstrumentation() : ModulePass(ID) {}
  StringRef getPassName() const override {
    return "DXIL mesh shader output instrumentation";
  }
  void applyOptions(PassOptions O) override;
  bool runOnModule(Module &M) override;

private:
  CallInst *m_OutputUAV = nullptr;
  int m_RemainingReservedSpaceInBytes = 0;
  Constant *m_OffsetMask = nullptr;
  SmallVector<Value*,2> m_threadUniquifier;

  uint64_t m_UAVSize = 1024 * 1024;
  bool m_ExpandPayload = false;

  struct BuilderContext {
    Module &M;
    DxilModule &DM;
    LLVMContext &Ctx;
    OP *HlslOP;
    IRBuilder<> &Builder;
  };

  SmallVector<Value*, 2> insertInstructionsToCreateDisambiguationValue(OP* HlslOP, LLVMContext& Ctx, StructType * originalPayloadStructType, Instruction * firstGetPayload);
  Value *reserveDebugEntrySpace(BuilderContext &BC, uint32_t SpaceInBytes);
  uint32_t UAVDumpingGroundOffset();
  Value *writeDwordAndReturnNewOffset(BuilderContext &BC, Value *TheOffset,
                                      Value *TheValue);
  template <typename... T> void Instrument(BuilderContext &BC, T... values);
};

void DxilPIXMeshShaderOutputInstrumentation::applyOptions(PassOptions O) 
{
  GetPassOptionUInt64(O, "UAVSize", &m_UAVSize, 1024 * 1024);
  GetPassOptionBool(O, "expand-payload", &m_ExpandPayload, 0);
}

uint32_t DxilPIXMeshShaderOutputInstrumentation::UAVDumpingGroundOffset() 
{
  return static_cast<uint32_t>(m_UAVSize - DebugBufferDumpingGroundSize);
}

Value *DxilPIXMeshShaderOutputInstrumentation::reserveDebugEntrySpace(
    BuilderContext &BC, uint32_t SpaceInBytes) 
{
  // Check the previous caller didn't reserve too much space:
  assert(m_RemainingReservedSpaceInBytes == 0);
  
  // Check that the caller didn't ask for so much memory that it will 
  // overwrite the offset counter:
  assert(m_RemainingReservedSpaceInBytes < (int)CounterOffsetBeyondUsefulData);

  m_RemainingReservedSpaceInBytes = SpaceInBytes;

  // Insert the UAV increment instruction:
  Function *AtomicOpFunc =
      BC.HlslOP->GetOpFunc(OP::OpCode::AtomicBinOp, Type::getInt32Ty(BC.Ctx));
  Constant *AtomicBinOpcode =
      BC.HlslOP->GetU32Const((unsigned)OP::OpCode::AtomicBinOp);
  Constant *AtomicAdd =
      BC.HlslOP->GetU32Const((unsigned)DXIL::AtomicBinOpCode::Add);
  Constant *OffsetArg =
      BC.HlslOP->GetU32Const(UAVDumpingGroundOffset() + CounterOffsetBeyondUsefulData);
  UndefValue *UndefArg = UndefValue::get(Type::getInt32Ty(BC.Ctx));

  Constant *Increment = BC.HlslOP->GetU32Const(SpaceInBytes);

  auto *PreviousValue = BC.Builder.CreateCall(
      AtomicOpFunc,
      {
          AtomicBinOpcode, // i32, ; opcode
          m_OutputUAV,     // %dx.types.Handle, ; resource handle
          AtomicAdd, // i32, ; binary operation code : EXCHANGE, IADD, AND, OR,
                     // XOR, IMIN, IMAX, UMIN, UMAX
          OffsetArg, // i32, ; coordinate c0: index in bytes
          UndefArg,  // i32, ; coordinate c1 (unused)
          UndefArg,  // i32, ; coordinate c2 (unused)
          Increment, // i32); increment value
      },
      "UAVIncResult");

  return BC.Builder.CreateAnd(PreviousValue, m_OffsetMask, "MaskedForUAVLimit");
}

Value *DxilPIXMeshShaderOutputInstrumentation::writeDwordAndReturnNewOffset(
    BuilderContext &BC, Value *TheOffset, Value *TheValue) 
{

  Function *StoreValue =
      BC.HlslOP->GetOpFunc(OP::OpCode::BufferStore, Type::getInt32Ty(BC.Ctx));
  Constant *StoreValueOpcode =
      BC.HlslOP->GetU32Const((unsigned)DXIL::OpCode::BufferStore);
  UndefValue *Undef32Arg = UndefValue::get(Type::getInt32Ty(BC.Ctx));
  Constant *WriteMask_X = BC.HlslOP->GetI8Const(1);

  (void)BC.Builder.CreateCall(
      StoreValue,
      {StoreValueOpcode, // i32 opcode
       m_OutputUAV,      // %dx.types.Handle, ; resource handle
       TheOffset,        // i32 c0: index in bytes into UAV
       Undef32Arg,       // i32 c1: unused
       TheValue,
       Undef32Arg, // unused values
       Undef32Arg, // unused values
       Undef32Arg, // unused values
       WriteMask_X});

  m_RemainingReservedSpaceInBytes -= sizeof(uint32_t);
  assert(m_RemainingReservedSpaceInBytes >=
         0); // or else the caller didn't reserve enough space

  return BC.Builder.CreateAdd(
      TheOffset,
      BC.HlslOP->GetU32Const(static_cast<unsigned int>(sizeof(uint32_t))));
}

template <typename... T>
void DxilPIXMeshShaderOutputInstrumentation::Instrument(BuilderContext &BC,
                                                        T... values)
{
  llvm::SmallVector<llvm::Value *, 10> Values(
      {static_cast<llvm::Value *>(values)...});
  const uint32_t DwordCount = Values.size();
  llvm::Value *byteOffset =
      reserveDebugEntrySpace(BC, DwordCount * sizeof(uint32_t));
  for (llvm::Value *V : Values)
  {
    byteOffset = writeDwordAndReturnNewOffset(BC, byteOffset, V);
  }
}

Value* GetValueFromExpandedPayload(IRBuilder<> &Builder, StructType* originalPayloadStructType, Instruction* firstGetPayload, unsigned int offset, const char * name) {
  auto *DerefPointer = Builder.getInt32(0);
  auto *OffsetToExpandedData = Builder.getInt32(offset);
  auto *GEP = Builder.CreateGEP(
      cast<PointerType>(firstGetPayload->getType()->getScalarType())
          ->getElementType(),
      firstGetPayload, {DerefPointer, OffsetToExpandedData});
  return Builder.CreateLoad(GEP, name);
}

SmallVector<Value*, 2> DxilPIXMeshShaderOutputInstrumentation::
    insertInstructionsToCreateDisambiguationValue(OP* HlslOP, LLVMContext& Ctx, StructType* originalPayloadStructType, Instruction* firstGetPayload) {

    // When a mesh shader is called from an amplification shader, all of the
    // thread id values are relative to the DispatchMesh call made by
    // that amplification shader. Data about what thread counts were passed
    // by the CPU to *CommandList::DispatchMesh are not available, but we
    // will have added that value to the AS->MS payload...

    IRBuilder<> Builder(firstGetPayload->getNextNode());

    auto * ASThreadId = GetValueFromExpandedPayload(Builder, originalPayloadStructType, firstGetPayload, originalPayloadStructType->getStructNumElements(), "ASThreadId");
    auto * ASDispatchMeshYCount = GetValueFromExpandedPayload(Builder, originalPayloadStructType, firstGetPayload, originalPayloadStructType->getStructNumElements() + 1, "ASDispatchMeshYCount");
    auto * ASDispatchMeshZCount = GetValueFromExpandedPayload(Builder, originalPayloadStructType, firstGetPayload, originalPayloadStructType->getStructNumElements() + 2, "ASDispatchMeshZCount");

    Constant *Zero32Arg = HlslOP->GetU32Const(0);
    Constant *One32Arg = HlslOP->GetU32Const(1);
    Constant *Two32Arg = HlslOP->GetU32Const(2);

    auto GroupIdFunc =
        HlslOP->GetOpFunc(DXIL::OpCode::GroupId, Type::getInt32Ty(Ctx));
    Constant *Opcode = HlslOP->GetU32Const((unsigned)DXIL::OpCode::GroupId);
    auto * GroupIdX =
        Builder.CreateCall(GroupIdFunc, {Opcode, Zero32Arg}, "GroupIdX");
    auto * GroupIdY =
        Builder.CreateCall(GroupIdFunc, {Opcode, One32Arg}, "GroupIdY");
    auto * GroupIdZ =
        Builder.CreateCall(GroupIdFunc, {Opcode, Two32Arg}, "GroupIdZ");

    auto *XxY =
      Builder.CreateMul(GroupIdX, ASDispatchMeshYCount);
    auto *XplusY = Builder.CreateAdd(GroupIdY, XxY);
    auto *XYxZ = Builder.CreateMul(XplusY, ASDispatchMeshZCount);
    auto *XYZ = Builder.CreateAdd(GroupIdZ, XYxZ);

    SmallVector<Value *, 2> ret;
    ret.push_back(ASThreadId);
    ret.push_back(XYZ);

    return ret;
}

bool DxilPIXMeshShaderOutputInstrumentation::runOnModule(Module &M)
{
  DxilModule &DM = M.GetOrCreateDxilModule();
  LLVMContext &Ctx = M.getContext();
  OP *HlslOP = DM.GetOP();

  Type *OriginalPayloadStructType = nullptr;
  ExpandedStruct expanded = {};
  Instruction* FirstNewStructGetMeshPayload = nullptr;
  if (m_ExpandPayload) {
    Instruction * getMeshPayloadInstructions = nullptr;
    llvm::Function *entryFunction = PIXPassHelpers::GetEntryFunction(DM);
    for (inst_iterator I = inst_begin(entryFunction),
        E = inst_end(entryFunction);
        I != E; ++I) {
        if (auto* Instr = llvm::cast<Instruction>(&*I)) {
            if (hlsl::OP::IsDxilOpFuncCallInst(Instr,
              hlsl::OP::OpCode::GetMeshPayload)) {
              getMeshPayloadInstructions = Instr;
              Type *OriginalPayloadStructPointerType = Instr->getType();
              OriginalPayloadStructType = OriginalPayloadStructPointerType->getPointerElementType();
              // The validator assures that there is only one call to GetMeshPayload...
              break;
            }
        }
    }
    
    if (OriginalPayloadStructType == nullptr) {
        // If the application used no payload, then we won't attempt to add one.
        // TODO: Is there a credible use case with no AS->MS payload?
        // PIX bug #35288335
        return false;
    }

    if (expanded.ExpandedPayloadStructPtrType == nullptr) {
      expanded = ExpandStructType(Ctx, OriginalPayloadStructType);
    }

    if (getMeshPayloadInstructions != nullptr) {

        Function* DxilFunc = HlslOP->GetOpFunc(OP::OpCode::GetMeshPayload, expanded.ExpandedPayloadStructPtrType);
        Constant* opArg = HlslOP->GetU32Const((unsigned)OP::OpCode::GetMeshPayload);
        IRBuilder<> Builder(getMeshPayloadInstructions);
        Value* args[] = { opArg };
        Instruction* payload = Builder.CreateCall(DxilFunc, args);

        if (FirstNewStructGetMeshPayload == nullptr) {
            FirstNewStructGetMeshPayload = payload;
        }

        ReplaceAllUsesOfInstructionWithNewValueAndDeleteInstruction(getMeshPayloadInstructions, payload, expanded.ExpandedPayloadStructType);
    }
  }
  
  Instruction *firstInsertionPt =
      dxilutil::FirstNonAllocaInsertionPt(GetEntryFunction(DM));
  IRBuilder<> Builder(firstInsertionPt);

  BuilderContext BC{M, DM, Ctx, HlslOP, Builder};

  m_OffsetMask = BC.HlslOP->GetU32Const(UAVDumpingGroundOffset() - 1);

  m_OutputUAV = CreateUAV(DM, Builder, 0, "PIX_DebugUAV_Handle");

  if (FirstNewStructGetMeshPayload == nullptr) {
    m_threadUniquifier.push_back(BC.HlslOP->GetU32Const(0));
    m_threadUniquifier.push_back(BC.HlslOP->GetU32Const(0));
  }
  else {
    m_threadUniquifier = insertInstructionsToCreateDisambiguationValue(HlslOP, Ctx, cast<StructType>(OriginalPayloadStructType), FirstNewStructGetMeshPayload);
  }

  auto F = HlslOP->GetOpFunc(DXIL::OpCode::EmitIndices, Type::getVoidTy(Ctx));
  auto FunctionUses = F->uses();
  for (auto FI = FunctionUses.begin(); FI != FunctionUses.end();)
  {
    auto &FunctionUse = *FI++;
    auto FunctionUser = FunctionUse.getUser();

    auto Call = cast<CallInst>(FunctionUser);

    IRBuilder<> Builder2(Call);
    BuilderContext BC2{M, DM, Ctx, HlslOP, Builder2};

    Instrument(BC2, BC2.HlslOP->GetI32Const(triangleIndexIndicator),
               m_threadUniquifier[0], m_threadUniquifier[1], Call->getOperand(1),
               Call->getOperand(2), Call->getOperand(3), Call->getOperand(4));
  }

  struct OutputType
  {
    Type *type;
    uint32_t tag;
  };
  SmallVector<OutputType, 4> StoreVertexOutputOverloads
  {
    {Type::getInt32Ty(Ctx), int32ValueIndicator},
    {Type::getInt16Ty(Ctx), int16ValueIndicator}, 
    {Type::getFloatTy(Ctx), floatValueIndicator},
    {Type::getHalfTy(Ctx), float16ValueIndicator}
  };

  for (auto const &Overload : StoreVertexOutputOverloads)
  {
    F = HlslOP->GetOpFunc(DXIL::OpCode::StoreVertexOutput, Overload.type);
    FunctionUses = F->uses();
    for (auto FI = FunctionUses.begin(); FI != FunctionUses.end();)
    {
      auto &FunctionUse = *FI++;
      auto FunctionUser = FunctionUse.getUser();

      auto Call = cast<CallInst>(FunctionUser);

      IRBuilder<> Builder2(Call);
      BuilderContext BC2{M, DM, Ctx, HlslOP, Builder2};

      // Expand column index to 32 bits:
      auto ColumnIndex = BC2.Builder.CreateCast(
       Instruction::ZExt, 
        Call->getOperand(3), 
        Type::getInt32Ty(Ctx));

      // Coerce actual value to int32 
      Value *CoercedValue = Call->getOperand(4);

      if (Overload.tag == floatValueIndicator) 
      {
        CoercedValue = BC2.Builder.CreateCast(
          Instruction::BitCast,
          CoercedValue, 
          Type::getInt32Ty(Ctx));
      }
      else if (Overload.tag == float16ValueIndicator) 
      {
        auto * HalfInt = BC2.Builder.CreateCast(
          Instruction::BitCast, 
          CoercedValue, 
          Type::getInt16Ty(Ctx));

        CoercedValue = BC2.Builder.CreateCast(
          Instruction::ZExt, 
          HalfInt, 
          Type::getInt32Ty(Ctx));
      }
      else if (Overload.tag == int16ValueIndicator) 
      {
        CoercedValue = BC2.Builder.CreateCast(
          Instruction::ZExt,
          CoercedValue,
          Type::getInt32Ty(Ctx));
      }

      Instrument(
        BC2, 
        BC2.HlslOP->GetI32Const(Overload.tag),
        m_threadUniquifier[0], m_threadUniquifier[1], 
        Call->getOperand(1),
        Call->getOperand(2),
        ColumnIndex,
        CoercedValue,
        Call->getOperand(5));
    }
  }

  DM.ReEmitDxilResources();

  return true;
}

char DxilPIXMeshShaderOutputInstrumentation::ID = 0;

ModulePass *llvm::createDxilDxilPIXMeshShaderOutputInstrumentation()
{
  return new DxilPIXMeshShaderOutputInstrumentation();
}

INITIALIZE_PASS(DxilPIXMeshShaderOutputInstrumentation,
                "hlsl-dxil-pix-meshshader-output-instrumentation",
                "DXIL mesh shader output instrumentation for PIX", false, false)
