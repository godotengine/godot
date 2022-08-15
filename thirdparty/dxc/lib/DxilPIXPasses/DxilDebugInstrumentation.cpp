///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDebugInstrumentation.cpp                                              //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Adds instrumentation that enables shader debugging in PIX                 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <vector>

#include "dxc/DXIL/DxilFunctionProps.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilUtil.h"
#include "dxc/DxilPIXPasses/DxilPIXPasses.h"
#include "dxc/DxilPIXPasses/DxilPIXVirtualRegisters.h"
#include "dxc/HLSL/DxilGenerationPass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"

#include "PixPassHelpers.h"

using namespace llvm;
using namespace hlsl;

// Overview of instrumentation:
//
// In summary, instructions are added that cause a "trace" of the execution of
// the shader to be written out to a UAV. This trace is then used by a debugger
// application to provide a post-mortem debugging experience that reconstructs
// the execution history of the shader.
//
// The trace is only required for a particular shader instance of interest, and
// a branchless mechanism is used to write the trace either to an incrementing
// location within the UAV, or to a "dumping ground" area at the top of the UAV
// if the instance is not of interest.
//
// The following modifications are made:
//
// First, instructions are added to the top of the entry point function that
// implement the following:
// -  Examine the input variables that define the instance of the shader that is
// running. This will
//    be SV_Position for pixel shaders, SV_Vertex+SV_Instance for vertex
//    shaders, thread id for compute shaders etc. If these system values need to
//    be added to the shader, then they are also added to the input signature,
//    if appropriate.
// -  Compare the above variables with the instance of interest defined by the
// invoker of this pass.
//    Deduce two values: a multiplicand and an addend that together allow a
//    branchless calculation of the offset into the UAV at which to write via
//    "offset = offset * multiplicand + addend." If the instance is NOT of
//    interest, the multiplicand is zero and the addend is sizeof(UAV)-(a little
//    bit), causing writes for uninteresting invocations to end up at the top of
//    the UAV. Otherwise the multiplicand is 1 and the addend is 0.
// -  Calculate an "instance identifier". Even with the above instance
// identification, several invocations may
//    end up matching the selection criteria. Specifically, this happens during
//    a draw call in which many triangles overlap the pixel of interest. More on
//    this below.
//
// During execution, the instrumentation for most instructions cause data to be
// emitted to the UAV. The index at which data is written is identified by
// treating the first uint32 of the UAV as an index which is atomically
// incremented by the instrumentation. The very first value of this counter that
// is encountered by each invocation is used as the "instance identifier"
// mentioned above. That instance identifier is written out with each packet,
// since many pixel shaders executing in parallel will emit interleaved packets,
// and the debugger application uses the identifiers to group packets from each
// separate invocation together.
//
// If an instruction has a non-void and primitive return type, i.e. isn't a
// struct, then the instrumentation will write that value out to the UAV as well
// as part of the "step" data packet.
//
// The limiting size of the UAV is enforced in a branchless way by ANDing the
// offset with a precomputed value that is sizeof(UAV)-64. The actual size of
// the UAV allocated by the caller is required to be a power of two plus 64 for
// this reason. The caller detects UAV overrun by examining a canary value close
// to the end of the power-of-two size of the UAV. If this value has been
// overwritten, the debug session is deemed to have overflowed the UAV. The
// caller will than allocate a UAV that is twice the size and try again, up to a
// predefined maximum.

// Keep these in sync with the same-named value in the debugger application's
// WinPixShaderUtils.h

constexpr uint64_t DebugBufferDumpingGroundSize = 64 * 1024;
// The actual max size per record is much smaller than this, but it never
// hurts to be generous.
constexpr size_t CounterOffsetBeyondUsefulData = DebugBufferDumpingGroundSize / 2;

// These definitions echo those in the debugger application's
// debugshaderrecord.h file
enum DebugShaderModifierRecordType {
  DebugShaderModifierRecordTypeInvocationStartMarker,
  DebugShaderModifierRecordTypeStep,
  DebugShaderModifierRecordTypeEvent,
  DebugShaderModifierRecordTypeInputRegister,
  DebugShaderModifierRecordTypeReadRegister,
  DebugShaderModifierRecordTypeWrittenRegister,
  DebugShaderModifierRecordTypeRegisterRelativeIndex0,
  DebugShaderModifierRecordTypeRegisterRelativeIndex1,
  DebugShaderModifierRecordTypeRegisterRelativeIndex2,
  DebugShaderModifierRecordTypeDXILStepVoid = 251,
  DebugShaderModifierRecordTypeDXILStepFloat = 252,
  DebugShaderModifierRecordTypeDXILStepUint32 = 253,
  DebugShaderModifierRecordTypeDXILStepUint64 = 254,
  DebugShaderModifierRecordTypeDXILStepDouble = 255,
};

// These structs echo those in the debugger application's debugshaderrecord.h
// file, but are recapitulated here because the originals use unnamed unions
// which are disallowed by DXCompiler's build.
//
#pragma pack(push, 4)
struct DebugShaderModifierRecordHeader {
  union {
    struct {
      uint32_t SizeDwords : 4;
      uint32_t Flags : 4;
      uint32_t Type : 8;
      uint32_t HeaderPayload : 16;
    } Details;
    uint32_t u32Header;
  } Header;
  uint32_t UID;
};

struct DebugShaderModifierRecordDXILStepBase {
  union {
    struct {
      uint32_t SizeDwords : 4;
      uint32_t Flags : 4;
      uint32_t Type : 8;
      uint32_t Opcode : 16;
    } Details;
    uint32_t u32Header;
  } Header;
  uint32_t UID;
  uint32_t InstructionOffset;
};

template <typename ReturnType>
struct DebugShaderModifierRecordDXILStep
    : public DebugShaderModifierRecordDXILStepBase {
  ReturnType ReturnValue;
  union {
    struct {
      uint32_t ValueOrdinalBase : 16;
      uint32_t ValueOrdinalIndex : 16;
    } Details;
    uint32_t u32ValueOrdinal;
  } ValueOrdinal;
};

template <>
struct DebugShaderModifierRecordDXILStep<void>
    : public DebugShaderModifierRecordDXILStepBase {};
#pragma pack(pop)

uint32_t
DebugShaderModifierRecordPayloadSizeDwords(size_t recordTotalSizeBytes) {
  return ((recordTotalSizeBytes - sizeof(DebugShaderModifierRecordHeader)) /
          sizeof(uint32_t));
}

class DxilDebugInstrumentation : public ModulePass {

private:
  union ParametersAllTogether {
    unsigned Parameters[3];
    struct PixelShaderParameters {
      unsigned X;
      unsigned Y;
    } PixelShader;
    struct VertexShaderParameters {
      unsigned VertexId;
      unsigned InstanceId;
    } VertexShader;
    struct ComputeShaderParameters {
      unsigned ThreadIdX;
      unsigned ThreadIdY;
      unsigned ThreadIdZ;
    } ComputeShader;
    struct GeometryShaderParameters {
      unsigned PrimitiveId;
      unsigned InstanceId;
    } GeometryShader;
    struct HullShaderParameters {
      unsigned PrimitiveId;
      unsigned ControlPointId;
    } HullShader;
    struct DomainShaderParameters {
      unsigned PrimitiveId;
    } DomainShader;
  } m_Parameters = {{0, 0, 0}};

  union SystemValueIndices {
    struct PixelShaderParameters {
      unsigned Position;
    } PixelShader;
    struct VertexShaderParameters {
      unsigned VertexId;
      unsigned InstanceId;
    } VertexShader;
  };

  uint64_t m_UAVSize = 1024 * 1024;
  Value *m_SelectionCriterion = nullptr;
  CallInst *m_HandleForUAV = nullptr;
  Value *m_InvocationId = nullptr;

  // Together these two values allow branchless writing to the UAV. An
  // invocation of the shader is either of interest or not (e.g. it writes to
  // the pixel the user selected for debugging or it doesn't). If not of
  // interest, debugging output will still occur, but it will be relegated to
  // the very top few bytes of the UAV. Invocations of interest, by contrast,
  // will be written to the UAV at sequentially increasing offsets.

  // This value will either be one or zero (one if the invocation is of
  // interest, zero otherwise)
  Value *m_OffsetMultiplicand = nullptr;
  // This will either be zero (if the invocation is of interest) or
  // (UAVSize)-(SmallValue) if not.
  Value *m_OffsetAddend = nullptr;

  Constant *m_OffsetMask = nullptr;

  Constant *m_CounterOffset = nullptr;

  struct BuilderContext {
    Module &M;
    DxilModule &DM;
    LLVMContext &Ctx;
    OP *HlslOP;
    IRBuilder<> &Builder;
  };

  uint32_t m_RemainingReservedSpaceInBytes = 0;
  Value *m_CurrentIndex = nullptr;

public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilDebugInstrumentation() : ModulePass(ID) {}
  StringRef getPassName() const override {
    return "Add PIX debug instrumentation";
  }
  void applyOptions(PassOptions O) override;
  bool runOnModule(Module &M) override;

  bool RunOnFunction(Module& M, DxilModule& DM, 
    llvm::Function *function);

private:
  SystemValueIndices addRequiredSystemValues(BuilderContext &BC, DXIL::ShaderKind shaderKind);
  void addInvocationSelectionProlog(BuilderContext &BC,
                                    SystemValueIndices SVIndices,
                                    DXIL::ShaderKind shaderKind);
  Value *addPixelShaderProlog(BuilderContext &BC, SystemValueIndices SVIndices);
  Value *addGeometryShaderProlog(BuilderContext &BC);
  Value *addDispatchedShaderProlog(BuilderContext &BC);
  Value* addRaygenShaderProlog(BuilderContext& BC);
  Value* addVertexShaderProlog(BuilderContext& BC,
                               SystemValueIndices SVIndices);
  Value *addHullhaderProlog(BuilderContext &BC);
  Value *addComparePrimitiveIdProlog(BuilderContext &BC,
                               unsigned SVIndices);
  void addDebugEntryValue(BuilderContext &BC, Value *TheValue);
  void addInvocationStartMarker(BuilderContext &BC);
  void reserveDebugEntrySpace(BuilderContext &BC, uint32_t SpaceInDwords);
  void addStoreStepDebugEntry(BuilderContext &BC, StoreInst *Inst);
  void addStepDebugEntry(BuilderContext& BC, Instruction* Inst);
  void addStepDebugEntryValue(BuilderContext &BC, std::uint32_t InstNum,
                              Value *V, std::uint32_t ValueOrdinal,
                              Value *ValueOrdinalIndex);
  uint32_t UAVDumpingGroundOffset();
  template <typename ReturnType>
  void addStepEntryForType(DebugShaderModifierRecordType RecordType,
                           BuilderContext &BC, std::uint32_t InstNum, Value *V,
                           std::uint32_t ValueOrdinal,
                           Value *ValueOrdinalIndex);
};

void DxilDebugInstrumentation::applyOptions(PassOptions O) {
  GetPassOptionUnsigned(O, "parameter0", &m_Parameters.Parameters[0], 0);
  GetPassOptionUnsigned(O, "parameter1", &m_Parameters.Parameters[1], 0);
  GetPassOptionUnsigned(O, "parameter2", &m_Parameters.Parameters[2], 0);
  GetPassOptionUInt64(O, "UAVSize", &m_UAVSize, 1024 * 1024);
}

uint32_t DxilDebugInstrumentation::UAVDumpingGroundOffset() {
  return static_cast<uint32_t>(m_UAVSize - DebugBufferDumpingGroundSize);
}

static unsigned
FindOrAddInputSignatureElement(hlsl::DxilSignature &InputSignature,
    const char * name,
    DXIL::SigPointKind sigPointKind,
                               hlsl::DXIL::SemanticKind semanticKind) {

  auto &InputElements = InputSignature.GetElements();

  auto ExistingElement =
      std::find_if(InputElements.begin(), InputElements.end(),
                   [&](const std::unique_ptr<DxilSignatureElement> &Element) {
                     return Element->GetSemantic()->GetKind() == semanticKind;
                   });

  if (ExistingElement == InputElements.end()) {
    auto AddedElement =
        llvm::make_unique<DxilSignatureElement>(sigPointKind);
    AddedElement->Initialize(name, hlsl::CompType::getF32(),
                                  hlsl::DXIL::InterpolationMode::Undefined, 1,
                                  1);
    AddedElement->AppendSemanticIndex(0);
    AddedElement->SetSigPointKind(sigPointKind);
    AddedElement->SetKind(semanticKind);

    auto index = InputSignature.AppendElement(std::move(AddedElement));
    return InputElements[index]->GetID();
  } else {
    return ExistingElement->get()->GetID();
  }
}

DxilDebugInstrumentation::SystemValueIndices
  DxilDebugInstrumentation::addRequiredSystemValues(
      BuilderContext & BC, DXIL::ShaderKind shaderKind) {
  SystemValueIndices SVIndices{};

  switch (shaderKind) {
  case DXIL::ShaderKind::Amplification:
  case DXIL::ShaderKind::Mesh:
  case DXIL::ShaderKind::Compute:
  case DXIL::ShaderKind::RayGeneration:
  //case DXIL::ShaderKind::Intersection:
  //case DXIL::ShaderKind::AnyHit:
  //case DXIL::ShaderKind::ClosestHit:
  //case DXIL::ShaderKind::Miss:
    // Dispatch* thread Id is not in the input signature
    break;
  case DXIL::ShaderKind::Vertex: {
      hlsl::DxilSignature& InputSignature = BC.DM.GetInputSignature();
      SVIndices.VertexShader.VertexId = FindOrAddInputSignatureElement(
        InputSignature, "VertexId", DXIL::SigPointKind::VSIn,
        hlsl::DXIL::SemanticKind::VertexID);
    SVIndices.VertexShader.InstanceId = FindOrAddInputSignatureElement(
        InputSignature, "InstanceId", DXIL::SigPointKind::VSIn,
        hlsl::DXIL::SemanticKind::InstanceID);
  } break;
  case DXIL::ShaderKind::Geometry:
  case DXIL::ShaderKind::Hull:
  case DXIL::ShaderKind::Domain:
    // GS, HS, DS Primitive id, HS control point id, and GS Instance id are not in the input signature
    break;
  case DXIL::ShaderKind::Pixel: {
    hlsl::DxilSignature &InputSignature = BC.DM.GetInputSignature();
    auto &InputElements = InputSignature.GetElements();

    auto Existing_SV_Position =
        std::find_if(InputElements.begin(), InputElements.end(),
                     [](const std::unique_ptr<DxilSignatureElement> &Element) {
                       return Element->GetSemantic()->GetKind() ==
                              hlsl::DXIL::SemanticKind::Position;
                     });

    // SV_Position, if present, has to have full mask, so we needn't worry
    // about the shader having selected components that don't include x or y.
    // If not present, we add it.
    if (Existing_SV_Position == InputElements.end()) {
      auto Added_SV_Position =
          llvm::make_unique<DxilSignatureElement>(DXIL::SigPointKind::PSIn);
      Added_SV_Position->Initialize("Position", hlsl::CompType::getF32(),
                                    hlsl::DXIL::InterpolationMode::Linear, 1,
                                    4);
      Added_SV_Position->AppendSemanticIndex(0);
      Added_SV_Position->SetSigPointKind(DXIL::SigPointKind::PSIn);
      Added_SV_Position->SetKind(hlsl::DXIL::SemanticKind::Position);

      auto index = InputSignature.AppendElement(std::move(Added_SV_Position));
      SVIndices.PixelShader.Position = InputElements[index]->GetID();
    } else {
      SVIndices.PixelShader.Position = Existing_SV_Position->get()->GetID();
    }
  } break;
  default:
    assert(false); // guaranteed by runOnModule
  }

  return SVIndices;
}

Value *DxilDebugInstrumentation::addDispatchedShaderProlog(BuilderContext &BC) {
  Constant *Zero32Arg = BC.HlslOP->GetU32Const(0);
  Constant *One32Arg = BC.HlslOP->GetU32Const(1);
  Constant *Two32Arg = BC.HlslOP->GetU32Const(2);

  auto ThreadIdFunc =
      BC.HlslOP->GetOpFunc(DXIL::OpCode::ThreadId, Type::getInt32Ty(BC.Ctx));
  Constant *Opcode = BC.HlslOP->GetU32Const((unsigned)DXIL::OpCode::ThreadId);
  auto ThreadIdX =
      BC.Builder.CreateCall(ThreadIdFunc, {Opcode, Zero32Arg}, "ThreadIdX");
  auto ThreadIdY =
      BC.Builder.CreateCall(ThreadIdFunc, {Opcode, One32Arg}, "ThreadIdY");
  auto ThreadIdZ =
      BC.Builder.CreateCall(ThreadIdFunc, {Opcode, Two32Arg}, "ThreadIdZ");

  // Compare to expected thread ID
  auto CompareToX = BC.Builder.CreateICmpEQ(
      ThreadIdX, BC.HlslOP->GetU32Const(m_Parameters.ComputeShader.ThreadIdX),
      "CompareToThreadIdX");
  auto CompareToY = BC.Builder.CreateICmpEQ(
      ThreadIdY, BC.HlslOP->GetU32Const(m_Parameters.ComputeShader.ThreadIdY),
      "CompareToThreadIdY");
  auto CompareToZ = BC.Builder.CreateICmpEQ(
      ThreadIdZ, BC.HlslOP->GetU32Const(m_Parameters.ComputeShader.ThreadIdZ),
      "CompareToThreadIdZ");

  auto CompareXAndY =
      BC.Builder.CreateAnd(CompareToX, CompareToY, "CompareXAndY");

  auto CompareAll =
      BC.Builder.CreateAnd(CompareXAndY, CompareToZ, "CompareAll");

  return CompareAll;
}

Value *DxilDebugInstrumentation::addRaygenShaderProlog(BuilderContext &BC) {
  auto DispatchRaysIndexOpFunc =
      BC.HlslOP->GetOpFunc(DXIL::OpCode::DispatchRaysIndex, Type::getInt32Ty(BC.Ctx));
  Constant *DispatchRaysIndexOpcode =
      BC.HlslOP->GetU32Const((unsigned)DXIL::OpCode::DispatchRaysIndex);
  auto RayX =
      BC.Builder.CreateCall(DispatchRaysIndexOpFunc, {DispatchRaysIndexOpcode, BC.HlslOP->GetI8Const(0) }, "RayX");
  auto RayY =
      BC.Builder.CreateCall(DispatchRaysIndexOpFunc, {DispatchRaysIndexOpcode, BC.HlslOP->GetI8Const(1) }, "RayY");
  auto RayZ =
      BC.Builder.CreateCall(DispatchRaysIndexOpFunc, {DispatchRaysIndexOpcode, BC.HlslOP->GetI8Const(2) }, "RayZ");

  auto CompareToX = BC.Builder.CreateICmpEQ(
      RayX, BC.HlslOP->GetU32Const(m_Parameters.ComputeShader.ThreadIdX),
      "CompareToThreadIdX");
  auto CompareToY = BC.Builder.CreateICmpEQ(
      RayY, BC.HlslOP->GetU32Const(m_Parameters.ComputeShader.ThreadIdY),
      "CompareToThreadIdY");
  auto CompareToZ = BC.Builder.CreateICmpEQ(
      RayZ, BC.HlslOP->GetU32Const(m_Parameters.ComputeShader.ThreadIdZ),
      "CompareToThreadIdZ");

  auto CompareXAndY =
      BC.Builder.CreateAnd(CompareToX, CompareToY, "CompareXAndY");

  auto CompareAll =
      BC.Builder.CreateAnd(CompareXAndY, CompareToZ, "CompareAll");
  return CompareAll;
}

Value *
DxilDebugInstrumentation::addVertexShaderProlog(BuilderContext &BC,
                                                SystemValueIndices SVIndices) {
  Constant *Zero32Arg = BC.HlslOP->GetU32Const(0);
  Constant *Zero8Arg = BC.HlslOP->GetI8Const(0);
  UndefValue *UndefArg = UndefValue::get(Type::getInt32Ty(BC.Ctx));

  auto LoadInputOpFunc =
      BC.HlslOP->GetOpFunc(DXIL::OpCode::LoadInput, Type::getInt32Ty(BC.Ctx));
  Constant *LoadInputOpcode =
      BC.HlslOP->GetU32Const((unsigned)DXIL::OpCode::LoadInput);
  Constant *SV_Vert_ID =
      BC.HlslOP->GetU32Const(SVIndices.VertexShader.VertexId);
  auto VertId =
      BC.Builder.CreateCall(LoadInputOpFunc,
                            {LoadInputOpcode, SV_Vert_ID, Zero32Arg /*row*/,
                             Zero8Arg /*column*/, UndefArg},
                            "VertId");

  Constant *SV_Instance_ID =
      BC.HlslOP->GetU32Const(SVIndices.VertexShader.InstanceId);
  auto InstanceId =
      BC.Builder.CreateCall(LoadInputOpFunc,
                            {LoadInputOpcode, SV_Instance_ID, Zero32Arg /*row*/,
                             Zero8Arg /*column*/, UndefArg},
                            "InstanceId");

  // Compare to expected vertex ID and instance ID
  auto CompareToVert = BC.Builder.CreateICmpEQ(
      VertId, BC.HlslOP->GetU32Const(m_Parameters.VertexShader.VertexId),
      "CompareToVertId");
  auto CompareToInstance = BC.Builder.CreateICmpEQ(
      InstanceId, BC.HlslOP->GetU32Const(m_Parameters.VertexShader.InstanceId),
      "CompareToInstanceId");
  auto CompareBoth =
      BC.Builder.CreateAnd(CompareToVert, CompareToInstance, "CompareBoth");

  return CompareBoth;
}

Value * DxilDebugInstrumentation::addHullhaderProlog(BuilderContext &BC) {
  auto LoadControlPointFunction = BC.HlslOP->GetOpFunc(
      DXIL::OpCode::OutputControlPointID, Type::getInt32Ty(BC.Ctx));
  Constant *LoadControlPointOpcode =
      BC.HlslOP->GetU32Const((unsigned)DXIL::OpCode::OutputControlPointID);
  auto ControlPointId =
      BC.Builder.CreateCall(LoadControlPointFunction,
                            {LoadControlPointOpcode},
                            "ControlPointId");

  auto *CompareToPrimId =
      addComparePrimitiveIdProlog(BC, m_Parameters.HullShader.PrimitiveId);

  auto CompareToControlPoint = BC.Builder.CreateICmpEQ(
      ControlPointId,
      BC.HlslOP->GetU32Const(m_Parameters.HullShader.ControlPointId),
      "CompareToControlPointId");

  auto CompareBoth =
      BC.Builder.CreateAnd(CompareToControlPoint, CompareToPrimId, "CompareBoth");

  return CompareBoth;
}

Value * DxilDebugInstrumentation::addComparePrimitiveIdProlog(BuilderContext &BC,
                                                unsigned primId) {
  auto PrimitiveIdFunction =
      BC.HlslOP->GetOpFunc(DXIL::OpCode::PrimitiveID, Type::getInt32Ty(BC.Ctx));
  Constant *PrimitiveIdOpcode =
      BC.HlslOP->GetU32Const((unsigned)DXIL::OpCode::PrimitiveID);
  auto PrimId =
      BC.Builder.CreateCall(PrimitiveIdFunction,
                            {PrimitiveIdOpcode},
                            "PrimId");

  return BC.Builder.CreateICmpEQ(PrimId, BC.HlslOP->GetU32Const(primId),
                                 "CompareToPrimId");
}

Value * DxilDebugInstrumentation::addGeometryShaderProlog(
    BuilderContext &BC) {
  auto CompareToPrim =
      addComparePrimitiveIdProlog(BC, m_Parameters.GeometryShader.PrimitiveId);

  if (BC.DM.GetGSInstanceCount() <= 1) {
    return CompareToPrim;
  }

  auto GSInstanceIdOpFunc = BC.HlslOP->GetOpFunc(DXIL::OpCode::GSInstanceID,
                                                 Type::getInt32Ty(BC.Ctx));
  Constant *GSInstanceIdOpcode =
      BC.HlslOP->GetU32Const((unsigned)DXIL::OpCode::GSInstanceID);
  auto GSInstanceId = BC.Builder.CreateCall(
      GSInstanceIdOpFunc, {GSInstanceIdOpcode}, "GSInstanceId");

  // Compare to expected vertex ID and instance ID
  auto CompareToInstance = BC.Builder.CreateICmpEQ(
      GSInstanceId,
      BC.HlslOP->GetU32Const(m_Parameters.GeometryShader.InstanceId),
      "CompareToInstanceId");
  auto CompareBoth =
      BC.Builder.CreateAnd(CompareToPrim, CompareToInstance, "CompareBoth");

  return CompareBoth;
}

Value *
DxilDebugInstrumentation::addPixelShaderProlog(BuilderContext &BC,
                                               SystemValueIndices SVIndices) {
  Constant *Zero32Arg = BC.HlslOP->GetU32Const(0);
  Constant *Zero8Arg = BC.HlslOP->GetI8Const(0);
  Constant *One8Arg = BC.HlslOP->GetI8Const(1);
  UndefValue *UndefArg = UndefValue::get(Type::getInt32Ty(BC.Ctx));

  // Convert SV_POSITION to UINT
  Value *XAsInt;
  Value *YAsInt;
  {
    auto LoadInputOpFunc =
        BC.HlslOP->GetOpFunc(DXIL::OpCode::LoadInput, Type::getFloatTy(BC.Ctx));
    Constant *LoadInputOpcode =
        BC.HlslOP->GetU32Const((unsigned)DXIL::OpCode::LoadInput);
    Constant *SV_Pos_ID =
        BC.HlslOP->GetU32Const(SVIndices.PixelShader.Position);
    auto XPos =
        BC.Builder.CreateCall(LoadInputOpFunc,
                              {LoadInputOpcode, SV_Pos_ID, Zero32Arg /*row*/,
                               Zero8Arg /*column*/, UndefArg},
                              "XPos");
    auto YPos =
        BC.Builder.CreateCall(LoadInputOpFunc,
                              {LoadInputOpcode, SV_Pos_ID, Zero32Arg /*row*/,
                               One8Arg /*column*/, UndefArg},
                              "YPos");

    XAsInt = BC.Builder.CreateCast(Instruction::CastOps::FPToUI, XPos,
                                   Type::getInt32Ty(BC.Ctx), "XIndex");
    YAsInt = BC.Builder.CreateCast(Instruction::CastOps::FPToUI, YPos,
                                   Type::getInt32Ty(BC.Ctx), "YIndex");
  }

  // Compare to expected pixel position and primitive ID
  auto CompareToX = BC.Builder.CreateICmpEQ(
      XAsInt, BC.HlslOP->GetU32Const(m_Parameters.PixelShader.X), "CompareToX");
  auto CompareToY = BC.Builder.CreateICmpEQ(
      YAsInt, BC.HlslOP->GetU32Const(m_Parameters.PixelShader.Y), "CompareToY");
  auto ComparePos = BC.Builder.CreateAnd(CompareToX, CompareToY, "ComparePos");

  return ComparePos;
}

void DxilDebugInstrumentation::addInvocationSelectionProlog(
    BuilderContext &BC, SystemValueIndices SVIndices, DXIL::ShaderKind shaderKind) {
  Value *ParameterTestResult = nullptr;
  switch (shaderKind) {
  case DXIL::ShaderKind::RayGeneration:
    ParameterTestResult = addRaygenShaderProlog(BC);
    break;
  case DXIL::ShaderKind::Compute:
  case DXIL::ShaderKind::Amplification:
  case DXIL::ShaderKind::Mesh:
  //case DXIL::ShaderKind::Intersection:
  //case DXIL::ShaderKind::AnyHit:
  //case DXIL::ShaderKind::ClosestHit:
  //case DXIL::ShaderKind::Miss:
    ParameterTestResult = addDispatchedShaderProlog(BC);
    break;
  case DXIL::ShaderKind::Geometry:
    ParameterTestResult = addGeometryShaderProlog(BC);
    break;
  case DXIL::ShaderKind::Vertex:
    ParameterTestResult = addVertexShaderProlog(BC, SVIndices);
    break;
  case DXIL::ShaderKind::Hull:
    ParameterTestResult = addHullhaderProlog(BC);
    break;
  case DXIL::ShaderKind::Domain:
    ParameterTestResult = addComparePrimitiveIdProlog(BC, m_Parameters.DomainShader.PrimitiveId);
    break;
  case DXIL::ShaderKind::Pixel:
    ParameterTestResult = addPixelShaderProlog(BC, SVIndices);
    break;
  default:
    assert(false); // guaranteed by runOnModule
  }

  // This is a convenient place to calculate the values that modify the UAV
  // offset for invocations of interest and for UAV size.
  m_OffsetMultiplicand =
      BC.Builder.CreateCast(Instruction::CastOps::ZExt, ParameterTestResult,
                            Type::getInt32Ty(BC.Ctx), "OffsetMultiplicand");
  auto InverseOffsetMultiplicand =
      BC.Builder.CreateSub(BC.HlslOP->GetU32Const(1), m_OffsetMultiplicand,
                           "ComplementOfMultiplicand");
  m_OffsetAddend =
      BC.Builder.CreateMul(BC.HlslOP->GetU32Const(UAVDumpingGroundOffset()),
                           InverseOffsetMultiplicand, "OffsetAddend");
  m_OffsetMask = BC.HlslOP->GetU32Const(UAVDumpingGroundOffset() - 1);

  m_CounterOffset = BC.HlslOP->GetU32Const(UAVDumpingGroundOffset() + CounterOffsetBeyondUsefulData);

  m_SelectionCriterion = ParameterTestResult;
}

void DxilDebugInstrumentation::reserveDebugEntrySpace(BuilderContext &BC,
                                                      uint32_t SpaceInBytes) {
  assert(m_CurrentIndex == nullptr);
  assert(m_RemainingReservedSpaceInBytes == 0);

  m_RemainingReservedSpaceInBytes = SpaceInBytes;

  // Insert the UAV increment instruction:
  Function *AtomicOpFunc =
      BC.HlslOP->GetOpFunc(OP::OpCode::AtomicBinOp, Type::getInt32Ty(BC.Ctx));
  Constant *AtomicBinOpcode =
      BC.HlslOP->GetU32Const((unsigned)OP::OpCode::AtomicBinOp);
  Constant *AtomicAdd =
      BC.HlslOP->GetU32Const((unsigned)DXIL::AtomicBinOpCode::Add);
  UndefValue *UndefArg = UndefValue::get(Type::getInt32Ty(BC.Ctx));

  // so inc will be zero for uninteresting invocations:
  Constant *Increment = BC.HlslOP->GetU32Const(SpaceInBytes);
  Value *IncrementForThisInvocation = BC.Builder.CreateMul(
      Increment, m_OffsetMultiplicand, "IncrementForThisInvocation");

  auto PreviousValue = BC.Builder.CreateCall(
      AtomicOpFunc,
      {
          AtomicBinOpcode,  // i32, ; opcode
          m_HandleForUAV,   // %dx.types.Handle, ; resource handle
          AtomicAdd,        // i32, ; binary operation code : EXCHANGE, IADD, AND, OR,
                            // XOR, IMIN, IMAX, UMIN, UMAX
          m_CounterOffset,  // i32, ; coordinate c0: index in bytes
          UndefArg,         // i32, ; coordinate c1 (unused)
          UndefArg,         // i32, ; coordinate c2 (unused)
          IncrementForThisInvocation, // i32); increment value
      },
      "UAVIncResult");

  if (m_InvocationId == nullptr) {
    m_InvocationId = PreviousValue;
  }

  auto MaskedForLimit =
      BC.Builder.CreateAnd(PreviousValue, m_OffsetMask, "MaskedForUAVLimit");
  // The return value will either end up being itself (multiplied by one and
  // added with zero) or the "dump uninteresting things here" value of (UAVSize
  // - a bit).
  auto MultipliedForInterest = BC.Builder.CreateMul(
      MaskedForLimit, m_OffsetMultiplicand, "MultipliedForInterest");
  auto AddedForInterest = BC.Builder.CreateAdd(
      MultipliedForInterest, m_OffsetAddend, "AddedForInterest");
  m_CurrentIndex = AddedForInterest;
}

void DxilDebugInstrumentation::addDebugEntryValue(BuilderContext &BC,
                                                  Value *TheValue) {
  assert(m_RemainingReservedSpaceInBytes > 0);

  auto TheValueTypeID = TheValue->getType()->getTypeID();
  if (TheValueTypeID == Type::TypeID::DoubleTyID) {
    Function *SplitDouble =
        BC.HlslOP->GetOpFunc(OP::OpCode::SplitDouble, TheValue->getType());
    Constant *SplitDoubleOpcode =
        BC.HlslOP->GetU32Const((unsigned)DXIL::OpCode::SplitDouble);
    auto SplitDoubleIntruction = BC.Builder.CreateCall(
        SplitDouble, {SplitDoubleOpcode, TheValue}, "SplitDouble");
    auto LowBits =
        BC.Builder.CreateExtractValue(SplitDoubleIntruction, 0, "LowBits");
    auto HighBits =
        BC.Builder.CreateExtractValue(SplitDoubleIntruction, 1, "HighBits");
    // addDebugEntryValue(BC, BC.HlslOP->GetU32Const(0)); // padding
    addDebugEntryValue(BC, LowBits);
    addDebugEntryValue(BC, HighBits);
  } else if (TheValueTypeID == Type::TypeID::IntegerTyID &&
             TheValue->getType()->getIntegerBitWidth() == 64) {
    auto LowBits =
        BC.Builder.CreateTrunc(TheValue, Type::getInt32Ty(BC.Ctx), "LowBits");
    auto ShiftedBits = BC.Builder.CreateLShr(TheValue, 32, "ShiftedBits");
    auto HighBits = BC.Builder.CreateTrunc(
        ShiftedBits, Type::getInt32Ty(BC.Ctx), "HighBits");
    // addDebugEntryValue(BC, BC.HlslOP->GetU32Const(0)); // padding
    addDebugEntryValue(BC, LowBits);
    addDebugEntryValue(BC, HighBits);
  } else if (TheValueTypeID == Type::TypeID::IntegerTyID &&
             (TheValue->getType()->getIntegerBitWidth() == 16 ||
              TheValue->getType()->getIntegerBitWidth() == 1)) {
    auto As32 =
        BC.Builder.CreateZExt(TheValue, Type::getInt32Ty(BC.Ctx), "As32");
    addDebugEntryValue(BC, As32);
  } else if (TheValueTypeID == Type::TypeID::HalfTyID) {
    auto AsFloat =
        BC.Builder.CreateFPCast(TheValue, Type::getFloatTy(BC.Ctx), "AsFloat");
    addDebugEntryValue(BC, AsFloat);
  } else {
    Function *StoreValue =
        BC.HlslOP->GetOpFunc(OP::OpCode::BufferStore,
                             TheValue->getType()); // Type::getInt32Ty(BC.Ctx));
    Constant *StoreValueOpcode =
        BC.HlslOP->GetU32Const((unsigned)DXIL::OpCode::BufferStore);
    UndefValue *Undef32Arg = UndefValue::get(Type::getInt32Ty(BC.Ctx));
    UndefValue *UndefArg = nullptr;
    if (TheValueTypeID == Type::TypeID::IntegerTyID) {
      UndefArg = UndefValue::get(Type::getInt32Ty(BC.Ctx));
    } else if (TheValueTypeID == Type::TypeID::FloatTyID) {
      UndefArg = UndefValue::get(Type::getFloatTy(BC.Ctx));
    } else {
      // The above are the only two valid types for a UAV store
      assert(false);
    }
    Constant *WriteMask_X = BC.HlslOP->GetI8Const(1);
    (void)BC.Builder.CreateCall(
        StoreValue, {StoreValueOpcode, // i32 opcode
                     m_HandleForUAV,   // %dx.types.Handle, ; resource handle
                     m_CurrentIndex,   // i32 c0: index in bytes into UAV
                     Undef32Arg,       // i32 c1: unused
                     TheValue,
                     UndefArg, // unused values
                     UndefArg, // unused values
                     UndefArg, // unused values
                     WriteMask_X});

    m_RemainingReservedSpaceInBytes -= 4;
    assert(m_RemainingReservedSpaceInBytes < 1024); // check for underflow

    if (m_RemainingReservedSpaceInBytes != 0) {
      m_CurrentIndex =
          BC.Builder.CreateAdd(m_CurrentIndex, BC.HlslOP->GetU32Const(4));
    } else {
      m_CurrentIndex = nullptr;
    }
  }
}

void DxilDebugInstrumentation::addInvocationStartMarker(BuilderContext &BC) {
  DebugShaderModifierRecordHeader marker{{{0, 0, 0, 0}}, 0};
  reserveDebugEntrySpace(BC, sizeof(marker));

  marker.Header.Details.SizeDwords =
      DebugShaderModifierRecordPayloadSizeDwords(sizeof(marker));
  ;
  marker.Header.Details.Flags = 0;
  marker.Header.Details.Type =
      DebugShaderModifierRecordTypeInvocationStartMarker;
  addDebugEntryValue(BC, BC.HlslOP->GetU32Const(marker.Header.u32Header));
  addDebugEntryValue(BC, m_InvocationId);
}

template <typename ReturnType>
void DxilDebugInstrumentation::addStepEntryForType(
    DebugShaderModifierRecordType RecordType, BuilderContext &BC,
    std::uint32_t InstNum, Value *V, std::uint32_t ValueOrdinal,
    Value *ValueOrdinalIndex) {
  DebugShaderModifierRecordDXILStep<ReturnType> step = {};
  reserveDebugEntrySpace(BC, sizeof(step));

  step.Header.Details.SizeDwords =
      DebugShaderModifierRecordPayloadSizeDwords(sizeof(step));
  step.Header.Details.Type = static_cast<uint8_t>(RecordType);
  addDebugEntryValue(BC, BC.HlslOP->GetU32Const(step.Header.u32Header));
  addDebugEntryValue(BC, m_InvocationId);
  addDebugEntryValue(BC, BC.HlslOP->GetU32Const(InstNum));

  if (RecordType != DebugShaderModifierRecordTypeDXILStepVoid) {
    addDebugEntryValue(BC, V);

    IRBuilder<> &B = BC.Builder;

    Value *VO = BC.HlslOP->GetU32Const(ValueOrdinal << 16);
    Value *VOI = B.CreateAnd(ValueOrdinalIndex, BC.HlslOP->GetU32Const(0xFFFF),
                             "ValueOrdinalIndex");
    Value *EncodedValueOrdinalAndIndex =
        BC.Builder.CreateOr(VO, VOI, "ValueOrdinal");
    addDebugEntryValue(BC, EncodedValueOrdinalAndIndex);
  }
}

void DxilDebugInstrumentation::addStoreStepDebugEntry(BuilderContext& BC,
    StoreInst* Inst) {
    std::uint32_t ValueOrdinalBase;
    std::uint32_t UnusedValueOrdinalSize;
    llvm::Value* ValueOrdinalIndex;
    if (!pix_dxil::PixAllocaRegWrite::FromInst(Inst, &ValueOrdinalBase,
        &UnusedValueOrdinalSize,
        &ValueOrdinalIndex)) {
        return;
    }

    std::uint32_t InstNum;
    if (!pix_dxil::PixDxilInstNum::FromInst(Inst, &InstNum)) {
        return;
    }

    if (PIXPassHelpers::IsAllocateRayQueryInstruction(Inst->getValueOperand())) {
        return;
    }

    addStepDebugEntryValue(BC, InstNum, Inst->getValueOperand(), ValueOrdinalBase,
        ValueOrdinalIndex);
}

void DxilDebugInstrumentation::addStepDebugEntry(BuilderContext &BC,
                                                 Instruction *Inst) {
  if (Inst->getOpcode() == Instruction::OtherOps::PHI) {
    return;
  }
  if (PIXPassHelpers::IsAllocateRayQueryInstruction(Inst)) {
      return;
  }

  if (auto *St = llvm::dyn_cast<llvm::StoreInst>(Inst)) {
    addStoreStepDebugEntry(BC, St);
    return;
  }

  std::uint32_t RegNum;
  if (!pix_dxil::PixDxilReg::FromInst(Inst, &RegNum)) {
    return;
  }

  std::uint32_t InstNum;
  if (!pix_dxil::PixDxilInstNum::FromInst(Inst, &InstNum)) {
    return;
  }

  addStepDebugEntryValue(BC, InstNum, Inst, RegNum, BC.Builder.getInt32(0));
}

void DxilDebugInstrumentation::addStepDebugEntryValue(
    BuilderContext &BC, std::uint32_t InstNum, Value *V,
    std::uint32_t ValueOrdinal, Value *ValueOrdinalIndex) {
  const Type::TypeID ID = V->getType()->getTypeID();

  switch (ID) {
  case Type::TypeID::StructTyID:
  case Type::TypeID::VoidTyID:
    addStepEntryForType<void>(DebugShaderModifierRecordTypeDXILStepVoid, BC,
                              InstNum, V, ValueOrdinal, ValueOrdinalIndex);
    break;
  case Type::TypeID::FloatTyID:
    addStepEntryForType<float>(DebugShaderModifierRecordTypeDXILStepFloat, BC,
                               InstNum, V, ValueOrdinal, ValueOrdinalIndex);
    break;
  case Type::TypeID::IntegerTyID:
    if (V->getType()->getIntegerBitWidth() == 64) {
      addStepEntryForType<uint64_t>(DebugShaderModifierRecordTypeDXILStepUint64,
                                    BC, InstNum, V, ValueOrdinal,
                                    ValueOrdinalIndex);
    } else {
      addStepEntryForType<uint32_t>(DebugShaderModifierRecordTypeDXILStepUint32,
                                    BC, InstNum, V, ValueOrdinal,
                                    ValueOrdinalIndex);
    }
    break;
  case Type::TypeID::DoubleTyID:
    addStepEntryForType<double>(DebugShaderModifierRecordTypeDXILStepDouble, BC,
                                InstNum, V, ValueOrdinal, ValueOrdinalIndex);
    break;
  case Type::TypeID::HalfTyID:
    addStepEntryForType<float>(DebugShaderModifierRecordTypeDXILStepFloat, BC,
                               InstNum, V, ValueOrdinal, ValueOrdinalIndex);
    break;
  case Type::TypeID::PointerTyID:
    // Skip pointer calculation instructions. They aren't particularly
    // meaningful to the user (being a mere implementation detail for lookup
    // tables, etc.), and their type is problematic from a UI point of view. The
    // subsequent instructions that dereference the pointer will be properly
    // instrumented and show the (meaningful) retrieved value.
    break;
  case Type::TypeID::VectorTyID:
    // Shows up in "insertelement" in raygen shader?
    break;
  case Type::TypeID::FP128TyID:
  case Type::TypeID::LabelTyID:
  case Type::TypeID::MetadataTyID:
  case Type::TypeID::FunctionTyID:
  case Type::TypeID::ArrayTyID:
  case Type::TypeID::X86_FP80TyID:
  case Type::TypeID::X86_MMXTyID:
  case Type::TypeID::PPC_FP128TyID:
    assert(false);
  }
}

bool DxilDebugInstrumentation::runOnModule(Module &M) {
  DxilModule &DM = M.GetOrCreateDxilModule();

  auto ShaderModel = DM.GetShaderModel();
  auto shaderKind = ShaderModel->GetKind();

  bool modified = false;
  if (shaderKind == DXIL::ShaderKind::Library) {
    for (llvm::Function& F : M.functions()) {
      modified = modified | RunOnFunction(M, DM, &F); 
      return modified;
    }
  }
  else {
    llvm::Function *entryFunction = PIXPassHelpers::GetEntryFunction(DM);
    modified = RunOnFunction(M, DM, entryFunction);
  }
  return modified;
}

bool DxilDebugInstrumentation::RunOnFunction(
  Module &M, 
  DxilModule &DM,
  llvm::Function * entryFunction) 
{
  DXIL::ShaderKind shaderKind = DXIL::ShaderKind::Invalid;
  if (!DM.HasDxilFunctionProps(entryFunction)) {
    auto ShaderModel = DM.GetShaderModel();
    shaderKind = ShaderModel->GetKind();
  } else {
    hlsl::DxilFunctionProps const &props =
        DM.GetDxilFunctionProps(entryFunction);
    shaderKind = props.shaderKind;
  }

  switch (shaderKind) {
  case DXIL::ShaderKind::Amplification:
  case DXIL::ShaderKind::Mesh:
  case DXIL::ShaderKind::Vertex:
  case DXIL::ShaderKind::Geometry:
  case DXIL::ShaderKind::Pixel:
  case DXIL::ShaderKind::Compute:
  case DXIL::ShaderKind::RayGeneration:
  case DXIL::ShaderKind::Hull:
  case DXIL::ShaderKind::Domain:
    break;
    //todo:
  case DXIL::ShaderKind::Intersection:
  case DXIL::ShaderKind::AnyHit:
  case DXIL::ShaderKind::ClosestHit:
  case DXIL::ShaderKind::Miss:
  default:
    return false;
  }

  // First record pointers to all instructions in the function:
  std::vector<Instruction *> AllInstructions;
  for (inst_iterator I = inst_begin(entryFunction),
                     E = inst_end(entryFunction);
       I != E; ++I) {
    AllInstructions.push_back(&*I);
  }

  // Branchless instrumentation requires taking care of a few things:
  // -Each invocation of the shader will be either of interest or not of
  // interest
  //    -If of interest, the offset into the output UAV will be as expected
  //    -If not, the offset is forced to (UAVsize) - (Small Amount), and that
  //    output is ignored by the CPU-side code.
  // -The invocation of interest may overflow the UAV. This is handled by taking
  // the modulus of the
  //  output index. Overflow is then detected on the CPU side by checking for
  //  the presence of a canary value at (UAVSize) - (Small Amount) * 2 (which is
  //  actually a conservative definition of overflow).
  //

  Instruction *firstInsertionPt =
      dxilutil::FirstNonAllocaInsertionPt(entryFunction);
  IRBuilder<> Builder(firstInsertionPt);

  LLVMContext &Ctx = M.getContext();
  OP *HlslOP = DM.GetOP();

  BuilderContext BC{M, DM, Ctx, HlslOP, Builder};

  m_HandleForUAV =
      PIXPassHelpers::CreateUAV(BC.DM, BC.Builder, 0, "PIX_DebugUAV_Handle");

  auto SystemValues = addRequiredSystemValues(BC, shaderKind);
  addInvocationSelectionProlog(BC, SystemValues, shaderKind);
  addInvocationStartMarker(BC);

  // Explicitly name new blocks in order to provide stable names for testing
  // purposes
  int NewBlockCounter = 0;

  auto &Blocks = entryFunction->getBasicBlockList();
  for (auto &CurrentBlock : Blocks) {
    struct ValueAndPhi {
      Value *Val;
      PHINode *Phi;
      unsigned Index;
    };

    std::map<BasicBlock *, std::vector<ValueAndPhi>> InsertableEdges;
    auto &Is = CurrentBlock.getInstList();
    for (auto &Inst : Is) {
      if (Inst.getOpcode() != Instruction::OtherOps::PHI) {
        break;
      }
      PHINode &PN = llvm::cast<PHINode>(Inst);
      for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i) {
        BasicBlock *PhiBB = PN.getIncomingBlock(i);
        Value *PhiVal = PN.getIncomingValue(i);
        InsertableEdges[PhiBB].push_back({PhiVal, &PN, i});
      }
    }

    for (auto &InsertableEdge : InsertableEdges) {
      auto *NewBlock = BasicBlock::Create(
          Ctx, "PIXDebug" + std::to_string(NewBlockCounter++),
          InsertableEdge.first->getParent());
      IRBuilder<> Builder(NewBlock);

      auto *PreviousBlock = InsertableEdge.first;

      // Modify all successor operands of the terminator in the previous block
      // that match the current block to point to the new block:
      TerminatorInst *terminator = PreviousBlock->getTerminator();
      unsigned NumSuccessors = terminator->getNumSuccessors();
      for (unsigned SuccessorIndex = 0; SuccessorIndex < NumSuccessors;
           ++SuccessorIndex) {
        auto *CurrentSuccessor = terminator->getSuccessor(SuccessorIndex);
        if (CurrentSuccessor == &CurrentBlock) {
          terminator->setSuccessor(SuccessorIndex, NewBlock);
        }
      }

      // Modify the Phis and add debug instrumentation
      for (auto &ValueNPhi : InsertableEdge.second) {
        // Modify the phi to refer to the new block:
        ValueNPhi.Phi->setIncomingBlock(ValueNPhi.Index, NewBlock);

        // Add instrumentation to the new block
        std::uint32_t RegNum;
        if (!pix_dxil::PixDxilReg::FromInst(ValueNPhi.Phi, &RegNum)) {
          continue;
        }

        std::uint32_t InstNum;
        if (!pix_dxil::PixDxilInstNum::FromInst(ValueNPhi.Phi, &InstNum)) {
          continue;
        }

        BuilderContext BC{M, DM, Ctx, HlslOP, Builder};
        addStepDebugEntryValue(BC, InstNum, ValueNPhi.Val, RegNum,
                               BC.Builder.getInt32(0));
      }

      // Add a branch to the new block to point to the current block
      Builder.CreateBr(&CurrentBlock);
    }
  }

  // Instrument original instructions:
  for (auto &Inst : AllInstructions) {
    // Instrumentation goes after the instruction if it is not a terminator.
    // Otherwise, Instrumentation goes prior to the instruction.
    if (!Inst->isTerminator()) {
      IRBuilder<> Builder(Inst->getNextNode());
      BuilderContext BC2{BC.M, BC.DM, BC.Ctx, BC.HlslOP, Builder};
      addStepDebugEntry(BC2, Inst);
    } else {
      // Insert before this instruction
      IRBuilder<> Builder(Inst);
      BuilderContext BC2{BC.M, BC.DM, BC.Ctx, BC.HlslOP, Builder};
      addStepDebugEntry(BC2, Inst);
    }
  }

  DM.ReEmitDxilResources();

  return true;
}

char DxilDebugInstrumentation::ID = 0;

ModulePass *llvm::createDxilDebugInstrumentationPass() {
  return new DxilDebugInstrumentation();
}

INITIALIZE_PASS(DxilDebugInstrumentation, "hlsl-dxil-debug-instrumentation",
                "HLSL DXIL debug instrumentation for PIX", false, false)
