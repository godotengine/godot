///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLSignatureLower.cpp                                                      //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Lower signatures of entry function to DXIL LoadInput/StoreOutput.         //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "HLSignatureLower.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilSignatureElement.h"
#include "dxc/DXIL/DxilSigPoint.h"
#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilTypeSystem.h"
#include "dxc/DXIL/DxilSemantic.h"
#include "dxc/HLSL/HLModule.h"
#include "dxc/HLSL/HLMatrixLowerHelper.h"
#include "dxc/HLSL/HLMatrixType.h"
#include "dxc/HlslIntrinsicOp.h"
#include "dxc/DXIL/DxilUtil.h"
#include "dxc/HLSL/DxilPackSignatureElement.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace hlsl;

namespace {
// Decompose semantic name (eg FOO1=>FOO,1), change interp mode for SV_Position.
// Return semantic index.
unsigned UpdateSemanticAndInterpMode(StringRef &semName,
                                    DXIL::InterpolationMode &mode,
                                    DXIL::SigPointKind kind,
                                    LLVMContext &Context) {
  llvm::StringRef baseSemName; // The 'FOO' in 'FOO1'.
  uint32_t semIndex;           // The '1' in 'FOO1'

  // Split semName and index.
  Semantic::DecomposeNameAndIndex(semName, &baseSemName, &semIndex);
  semName = baseSemName;
  const Semantic *semantic = Semantic::GetByName(semName, kind);
  if (semantic && semantic->GetKind() == Semantic::Kind::Position) {
    // Update interp mode to no_perspective version for SV_Position.
    switch (mode) {
    case InterpolationMode::Kind::LinearCentroid:
      mode = InterpolationMode::Kind::LinearNoperspectiveCentroid;
      break;
    case InterpolationMode::Kind::LinearSample:
      mode = InterpolationMode::Kind::LinearNoperspectiveSample;
      break;
    case InterpolationMode::Kind::Linear:
      mode = InterpolationMode::Kind::LinearNoperspective;
      break;
    case InterpolationMode::Kind::Constant:
    case InterpolationMode::Kind::Undefined:
    case InterpolationMode::Kind::Invalid: {
      Context.emitError("invalid interpolation mode for SV_Position");
    } break;
    case InterpolationMode::Kind::LinearNoperspective:
    case InterpolationMode::Kind::LinearNoperspectiveCentroid:
    case InterpolationMode::Kind::LinearNoperspectiveSample:
      // Already Noperspective modes.
      break;
    }
  }
  return semIndex;
}

DxilSignatureElement *FindArgInSignature(Argument &arg,
                                         llvm::StringRef semantic,
                                         DXIL::InterpolationMode interpMode,
                                         DXIL::SigPointKind kind,
                                         DxilSignature &sig) {
  // Match output ID.
  unsigned semIndex =
      UpdateSemanticAndInterpMode(semantic, interpMode, kind, arg.getContext());

  for (uint32_t i = 0; i < sig.GetElements().size(); i++) {
    DxilSignatureElement &SE = sig.GetElement(i);
    bool semNameMatch = semantic.equals_lower(SE.GetName());
    bool semIndexMatch = semIndex == SE.GetSemanticIndexVec()[0];

    if (semNameMatch && semIndexMatch) {
      // Find a match.
      return &SE;
    }
  }
  return nullptr;
}
} // namespace

namespace {
void replaceInputOutputWithIntrinsic(DXIL::SemanticKind semKind, Value *GV,
                                     OP *hlslOP, IRBuilder<> &Builder) {
  Type *Ty = GV->getType();
  if (Ty->isPointerTy())
    Ty = Ty->getPointerElementType();

  OP::OpCode opcode;
  switch (semKind) {
  case Semantic::Kind::DomainLocation:
    opcode = OP::OpCode::DomainLocation;
    break;
  case Semantic::Kind::OutputControlPointID:
    opcode = OP::OpCode::OutputControlPointID;
    break;
  case Semantic::Kind::GSInstanceID:
    opcode = OP::OpCode::GSInstanceID;
    break;
  case Semantic::Kind::PrimitiveID:
    opcode = OP::OpCode::PrimitiveID;
    break;
  case Semantic::Kind::SampleIndex:
    opcode = OP::OpCode::SampleIndex;
    break;
  case Semantic::Kind::Coverage:
    opcode = OP::OpCode::Coverage;
    break;
  case Semantic::Kind::InnerCoverage:
    opcode = OP::OpCode::InnerCoverage;
    break;
  case Semantic::Kind::ViewID:
    opcode = OP::OpCode::ViewID;
    break;
  case Semantic::Kind::GroupThreadID:
    opcode = OP::OpCode::ThreadIdInGroup;
    break;
  case Semantic::Kind::GroupID:
    opcode = OP::OpCode::GroupId;
    break;
  case Semantic::Kind::DispatchThreadID:
    opcode = OP::OpCode::ThreadId;
    break;
  case Semantic::Kind::GroupIndex:
    opcode = OP::OpCode::FlattenedThreadIdInGroup;
    break;
  case Semantic::Kind::CullPrimitive: {
    GV->replaceAllUsesWith(ConstantInt::get(Ty, (uint64_t)0));
    return;
  } break;
  default:
    DXASSERT(0, "invalid semantic");
    return;
  }

  Function *dxilFunc = hlslOP->GetOpFunc(opcode, Ty->getScalarType());
  Constant *OpArg = hlslOP->GetU32Const((unsigned)opcode);

  Value *newArg = nullptr;
  if (semKind == Semantic::Kind::DomainLocation ||
      semKind == Semantic::Kind::GroupThreadID ||
      semKind == Semantic::Kind::GroupID ||
      semKind == Semantic::Kind::DispatchThreadID) {
    unsigned vecSize = 1;
    if (FixedVectorType *VT = dyn_cast<FixedVectorType>(Ty))
      vecSize = VT->getNumElements();

    newArg = Builder.CreateCall(dxilFunc, { OpArg,
      semKind == Semantic::Kind::DomainLocation ? hlslOP->GetU8Const(0) : hlslOP->GetU32Const(0) });
    if (vecSize > 1) {
      Value *result = UndefValue::get(Ty);
      result = Builder.CreateInsertElement(result, newArg, (uint64_t)0);

      for (unsigned i = 1; i < vecSize; i++) {
        Value *newElt =
            Builder.CreateCall(dxilFunc, { OpArg,
              semKind == Semantic::Kind::DomainLocation ? hlslOP->GetU8Const(i)
                                                        : hlslOP->GetU32Const(i) });
        result = Builder.CreateInsertElement(result, newElt, i);
      }
      newArg = result;
    }
  } else {
    newArg = Builder.CreateCall(dxilFunc, {OpArg});
  }

  if (newArg->getType() != GV->getType()) {
    DXASSERT_NOMSG(GV->getType()->isPointerTy());
    for (User *U : GV->users()) {
      if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
        LI->replaceAllUsesWith(newArg);
      }
    }
  } else {
    GV->replaceAllUsesWith(newArg);
  }
}
} // namespace

void HLSignatureLower::ProcessArgument(Function *func,
                                     DxilFunctionAnnotation *funcAnnotation,
                                     Argument &arg, DxilFunctionProps &props,
                                     const ShaderModel *pSM,
                                     bool isPatchConstantFunction,
                                     bool forceOut, bool &hasClipPlane) {
  Type *Ty = arg.getType();
  DxilParameterAnnotation &paramAnnotation =
      funcAnnotation->GetParameterAnnotation(arg.getArgNo());
  hlsl::DxilParamInputQual qual =
      forceOut ? DxilParamInputQual::Out : paramAnnotation.GetParamInputQual();
  bool isInout = qual == DxilParamInputQual::Inout;

  // If this was an inout param, do the output side first
  if (isInout) {
    DXASSERT(!isPatchConstantFunction,
             "Patch Constant function should not have inout param");
    m_inoutArgSet.insert(&arg);
    const bool bForceOutTrue = true;
    ProcessArgument(func, funcAnnotation, arg, props, pSM,
                    isPatchConstantFunction, bForceOutTrue, hasClipPlane);
    qual = DxilParamInputQual::In;
  }

  // Get stream index
  unsigned streamIdx = 0;
  switch (qual) {
  case DxilParamInputQual::OutStream1:
    streamIdx = 1;
    break;
  case DxilParamInputQual::OutStream2:
    streamIdx = 2;
    break;
  case DxilParamInputQual::OutStream3:
    streamIdx = 3;
    break;
  default:
    // Use streamIdx = 0 by default.
    break;
  }

  const SigPoint *sigPoint = SigPoint::GetSigPoint(
      SigPointFromInputQual(qual, props.shaderKind, isPatchConstantFunction));

  unsigned rows, cols;
  HLModule::GetParameterRowsAndCols(Ty, rows, cols, paramAnnotation);
  CompType EltTy = paramAnnotation.GetCompType();
  DXIL::InterpolationMode interpMode =
      paramAnnotation.GetInterpolationMode().GetKind();

  // Set undefined interpMode.
  if (sigPoint->GetKind() == DXIL::SigPointKind::MSPOut) {
    if (interpMode != InterpolationMode::Kind::Undefined &&
        interpMode != InterpolationMode::Kind::Constant) {
      dxilutil::EmitErrorOnFunction(HLM.GetModule()->getContext(), func,
        "Mesh shader's primitive outputs' interpolation mode must be constant or undefined.");
    }
    interpMode = InterpolationMode::Kind::Constant;
  }
  else if (!sigPoint->NeedsInterpMode())
    interpMode = InterpolationMode::Kind::Undefined;
  else if (interpMode == InterpolationMode::Kind::Undefined) {
    // Type-based default: linear for floats, constant for others.
    if (EltTy.IsFloatTy())
      interpMode = InterpolationMode::Kind::Linear;
    else
      interpMode = InterpolationMode::Kind::Constant;
  }

  //  back-compat mode - remap obsolete semantics
  if (HLM.GetHLOptions().bDX9CompatMode && paramAnnotation.HasSemanticString()) {
    hlsl::RemapObsoleteSemantic(paramAnnotation, sigPoint->GetKind(), HLM.GetCtx());
  }

  llvm::StringRef semanticStr = paramAnnotation.GetSemanticString();
  if (semanticStr.empty()) {
    dxilutil::EmitErrorOnFunction(HLM.GetModule()->getContext(), func,
        "Semantic must be defined for all parameters of an entry function or "
        "patch constant function");
    return;
  }
  UpdateSemanticAndInterpMode(semanticStr, interpMode, sigPoint->GetKind(),
                             arg.getContext());

  // Get Semantic interpretation, skipping if not in signature
  const Semantic *pSemantic = Semantic::GetByName(semanticStr);
  DXIL::SemanticInterpretationKind interpretation =
      SigPoint::GetInterpretation(pSemantic->GetKind(), sigPoint->GetKind(),
                                  pSM->GetMajor(), pSM->GetMinor());

  // Verify system value semantics do not overlap.
  // Note: Arbitrary are always in the signature and will be verified with a
  // different mechanism. For patch constant function, only validate patch
  // constant elements (others already validated on hull function)
  if (pSemantic->GetKind() != DXIL::SemanticKind::Arbitrary &&
      (!isPatchConstantFunction ||
       (!sigPoint->IsInput() && !sigPoint->IsOutput()))) {
    auto &SemanticUseMap =
        sigPoint->IsInput()
            ? m_InputSemanticsUsed
            : (sigPoint->IsOutput()
                   ? m_OutputSemanticsUsed[streamIdx]
                   : (sigPoint->IsPatchConstOrPrim() ? m_PatchConstantSemanticsUsed
                                                  : m_OtherSemanticsUsed));
    if (SemanticUseMap.count((unsigned)pSemantic->GetKind()) > 0) {
      auto &SemanticIndexSet = SemanticUseMap[(unsigned)pSemantic->GetKind()];
      for (unsigned idx : paramAnnotation.GetSemanticIndexVec()) {
        if (SemanticIndexSet.count(idx) > 0) {
          dxilutil::EmitErrorOnFunction(HLM.GetModule()->getContext(), func, "Parameter with semantic " + semanticStr +
            " has overlapping semantic index at " + std::to_string(idx) + ".");
          return;
        }
      }
    }
    auto &SemanticIndexSet = SemanticUseMap[(unsigned)pSemantic->GetKind()];
    for (unsigned idx : paramAnnotation.GetSemanticIndexVec()) {
      SemanticIndexSet.emplace(idx);
    }
    // Enforce Coverage and InnerCoverage input mutual exclusivity
    if (sigPoint->IsInput()) {
      if ((pSemantic->GetKind() == DXIL::SemanticKind::Coverage &&
           SemanticUseMap.count((unsigned)DXIL::SemanticKind::InnerCoverage) >
               0) ||
          (pSemantic->GetKind() == DXIL::SemanticKind::InnerCoverage &&
           SemanticUseMap.count((unsigned)DXIL::SemanticKind::Coverage) > 0)) {
        dxilutil::EmitErrorOnFunction(HLM.GetModule()->getContext(), func,
            "Pixel shader inputs SV_Coverage and SV_InnerCoverage are mutually "
            "exclusive.");
        return;
      }
    }
  }

  // Validate interpretation and replace argument usage with load/store
  // intrinsics
  {
    switch (interpretation) {
    case DXIL::SemanticInterpretationKind::NA: {
      dxilutil::EmitErrorOnFunction(HLM.GetModule()->getContext(), func, Twine("Semantic ") + semanticStr +
                                    Twine(" is invalid for shader model: ") +
                                    ShaderModel::GetKindName(props.shaderKind));

      return;
    }
    case DXIL::SemanticInterpretationKind::NotInSig:
    case DXIL::SemanticInterpretationKind::Shadow: {
      IRBuilder<> funcBuilder(func->getEntryBlock().getFirstInsertionPt());
      if (DbgDeclareInst *DDI = llvm::FindAllocaDbgDeclare(&arg)) {
        funcBuilder.SetCurrentDebugLocation(DDI->getDebugLoc());
      }
      replaceInputOutputWithIntrinsic(pSemantic->GetKind(), &arg, HLM.GetOP(),
                                      funcBuilder);
      if (interpretation == DXIL::SemanticInterpretationKind::NotInSig)
        return; // This argument should not be included in the signature
      break;
    }
    case DXIL::SemanticInterpretationKind::SV:
    case DXIL::SemanticInterpretationKind::SGV:
    case DXIL::SemanticInterpretationKind::Arb:
    case DXIL::SemanticInterpretationKind::Target:
    case DXIL::SemanticInterpretationKind::TessFactor:
    case DXIL::SemanticInterpretationKind::NotPacked:
    case DXIL::SemanticInterpretationKind::ClipCull:
      // Will be replaced with load/store intrinsics in
      // GenerateDxilInputsOutputs
      break;
    default:
      DXASSERT(false, "Unexpected SemanticInterpretationKind");
      return;
    }
  }

  // Determine signature this argument belongs in, if any
  DxilSignature *pSig = nullptr;
  DXIL::SignatureKind sigKind = sigPoint->GetSignatureKindWithFallback();
  switch (sigKind) {
  case DXIL::SignatureKind::Input:
    pSig = &EntrySig.InputSignature;
    break;
  case DXIL::SignatureKind::Output:
    pSig = &EntrySig.OutputSignature;
    break;
  case DXIL::SignatureKind::PatchConstOrPrim:
    pSig = &EntrySig.PatchConstOrPrimSignature;
    break;
  default:
    DXASSERT(false, "Expected real signature kind at this point");
    return; // No corresponding signature
  }

  // Create and add element to signature
  DxilSignatureElement *pSE = nullptr;
  {
    // Add signature element to appropriate maps
    if (isPatchConstantFunction &&
        sigKind != DXIL::SignatureKind::PatchConstOrPrim) {
      pSE = FindArgInSignature(arg, paramAnnotation.GetSemanticString(),
                               interpMode, sigPoint->GetKind(), *pSig);
      if (!pSE) {
        dxilutil::EmitErrorOnFunction(HLM.GetModule()->getContext(), func, Twine("Signature element ") + semanticStr +
                                      Twine(", referred to by patch constant function, is not found in "
                                            "corresponding hull shader ") +
                                      (sigKind == DXIL::SignatureKind::Input ? "input." : "output."));
        return;
      }
      m_patchConstantInputsSigMap[arg.getArgNo()] = pSE;
    } else {
      std::unique_ptr<DxilSignatureElement> SE = pSig->CreateElement();
      pSE = SE.get();
      pSig->AppendElement(std::move(SE));
      pSE->SetSigPointKind(sigPoint->GetKind());
      pSE->Initialize(semanticStr, EltTy, interpMode, rows, cols,
                      Semantic::kUndefinedRow, Semantic::kUndefinedCol,
                      pSE->GetID(), paramAnnotation.GetSemanticIndexVec());
      m_sigValueMap[pSE] = &arg;
    }
  }

  if (paramAnnotation.IsPrecise())
    m_preciseSigSet.insert(pSE);
  if (sigKind == DXIL::SignatureKind::Output &&
      pSemantic->GetKind() == Semantic::Kind::Position && hasClipPlane) {
    GenerateClipPlanesForVS(&arg);
    hasClipPlane = false;
  }

  // Set Output Stream.
  if (streamIdx > 0)
    pSE->SetOutputStream(streamIdx);
}

void HLSignatureLower::CreateDxilSignatures() {
  DxilFunctionProps &props = HLM.GetDxilFunctionProps(Entry);
  const ShaderModel *pSM = HLM.GetShaderModel();

  DXASSERT(Entry->getReturnType()->isVoidTy(),
           "Should changed in SROA_Parameter_HLSL");

  DxilFunctionAnnotation *EntryAnnotation = HLM.GetFunctionAnnotation(Entry);
  DXASSERT(EntryAnnotation, "must have function annotation for entry function");
  bool bHasClipPlane =
      props.shaderKind == DXIL::ShaderKind::Vertex ? HasClipPlanes() : false;
  const bool isPatchConstantFunctionFalse = false;
  const bool bForOutFasle = false;
  for (Argument &arg : Entry->getArgumentList()) {
    Type *Ty = arg.getType();
    // Skip streamout obj.
    if (HLModule::IsStreamOutputPtrType(Ty))
      continue;

    // Skip OutIndices and InPayload
    DxilParameterAnnotation &paramAnnotation =
      EntryAnnotation->GetParameterAnnotation(arg.getArgNo());
    hlsl::DxilParamInputQual qual = paramAnnotation.GetParamInputQual();
    if (qual == hlsl::DxilParamInputQual::OutIndices ||
        qual == hlsl::DxilParamInputQual::InPayload)
      continue;

    ProcessArgument(Entry, EntryAnnotation, arg, props, pSM,
                    isPatchConstantFunctionFalse, bForOutFasle, bHasClipPlane);
  }

  if (bHasClipPlane) {
    dxilutil::EmitErrorOnFunction(HLM.GetModule()->getContext(), Entry, "Cannot use clipplanes attribute without "
                                  "specifying a 4-component SV_Position "
                                  "output");
  }

  m_OtherSemanticsUsed.clear();

  if (props.shaderKind == DXIL::ShaderKind::Hull) {
    Function *patchConstantFunc = props.ShaderProps.HS.patchConstantFunc;
    if (patchConstantFunc == nullptr) {
      dxilutil::EmitErrorOnFunction(HLM.GetModule()->getContext(), Entry,
          "Patch constant function is not specified.");
    }

    DxilFunctionAnnotation *patchFuncAnnotation =
        HLM.GetFunctionAnnotation(patchConstantFunc);
    DXASSERT(patchFuncAnnotation,
             "must have function annotation for patch constant function");
    const bool isPatchConstantFunctionTrue = true;
    for (Argument &arg : patchConstantFunc->getArgumentList()) {
      ProcessArgument(patchConstantFunc, patchFuncAnnotation, arg, props, pSM,
                      isPatchConstantFunctionTrue, bForOutFasle, bHasClipPlane);
    }
  }
}

// Allocate input/output slots
void HLSignatureLower::AllocateDxilInputOutputs() {
  DxilFunctionProps &props = HLM.GetDxilFunctionProps(Entry);
  const ShaderModel *pSM = HLM.GetShaderModel();
  const HLOptions &opts = HLM.GetHLOptions();
  DXASSERT_NOMSG(opts.PackingStrategy <
                 (unsigned)DXIL::PackingStrategy::Invalid);
  DXIL::PackingStrategy packing = (DXIL::PackingStrategy)opts.PackingStrategy;
  if (packing == DXIL::PackingStrategy::Default)
    packing = pSM->GetDefaultPackingStrategy();

  hlsl::PackDxilSignature(EntrySig.InputSignature, packing);
  if (!EntrySig.InputSignature.IsFullyAllocated()) {
    dxilutil::EmitErrorOnFunction(HLM.GetModule()->getContext(), Entry,
        "Failed to allocate all input signature elements in available space.");
  }

  if (props.shaderKind != DXIL::ShaderKind::Amplification) {
    hlsl::PackDxilSignature(EntrySig.OutputSignature, packing);
    if (!EntrySig.OutputSignature.IsFullyAllocated()) {
      dxilutil::EmitErrorOnFunction(HLM.GetModule()->getContext(), Entry,
          "Failed to allocate all output signature elements in available space.");
    }
  }

  if (props.shaderKind == DXIL::ShaderKind::Hull ||
      props.shaderKind == DXIL::ShaderKind::Domain ||
      props.shaderKind == DXIL::ShaderKind::Mesh) {
    hlsl::PackDxilSignature(EntrySig.PatchConstOrPrimSignature, packing);
    if (!EntrySig.PatchConstOrPrimSignature.IsFullyAllocated()) {
      dxilutil::EmitErrorOnFunction(HLM.GetModule()->getContext(), Entry,
                             "Failed to allocate all patch constant signature "
                             "elements in available space.");
    }
  }
}

namespace {
// Helper functions and class for lower signature.
void GenerateStOutput(Function *stOutput, MutableArrayRef<Value *> args,
                      IRBuilder<> &Builder, bool cast) {
  if (cast) {
    Value *value = args[DXIL::OperandIndex::kStoreOutputValOpIdx];
    args[DXIL::OperandIndex::kStoreOutputValOpIdx] =
        Builder.CreateZExt(value, Builder.getInt32Ty());
  }

  Builder.CreateCall(stOutput, args);
}

void replaceStWithStOutput(Function *stOutput, StoreInst *stInst,
                           Constant *OpArg, Constant *outputID, Value *idx,
                           unsigned cols, Value *vertexOrPrimID, bool bI1Cast) {
  IRBuilder<> Builder(stInst);
  Value *val = stInst->getValueOperand();

  if (VectorType *VT = dyn_cast<VectorType>(val->getType())) {
    DXASSERT_LOCALVAR(VT, cols == VT->getNumElements(), "vec size must match");
    for (unsigned col = 0; col < cols; col++) {
      Value *subVal = Builder.CreateExtractElement(val, col);
      Value *colIdx = Builder.getInt8(col);
      SmallVector<Value *, 4> args = {OpArg, outputID, idx, colIdx, subVal};
      if (vertexOrPrimID)
        args.emplace_back(vertexOrPrimID);
      GenerateStOutput(stOutput, args, Builder, bI1Cast);
    }
    // remove stInst
    stInst->eraseFromParent();
  } else if (!val->getType()->isArrayTy()) {
    // TODO: support case cols not 1
    DXASSERT(cols == 1, "only support scalar here");
    Value *colIdx = Builder.getInt8(0);
    SmallVector<Value *, 4> args = {OpArg, outputID, idx, colIdx, val};
    if (vertexOrPrimID)
      args.emplace_back(vertexOrPrimID);
    GenerateStOutput(stOutput, args, Builder, bI1Cast);
    // remove stInst
    stInst->eraseFromParent();
  } else {
    DXASSERT(0, "not support array yet");
    // TODO: support array.
    Value *colIdx = Builder.getInt8(0);
    ArrayType *AT = cast<ArrayType>(val->getType());
    Value *args[] = {OpArg, outputID, idx, colIdx, /*val*/ nullptr};
    (void)args;
    (void)AT;
  }
}

Value *GenerateLdInput(Function *loadInput, ArrayRef<Value *> args,
                       IRBuilder<> &Builder, Value *zero, bool bCast,
                       Type *Ty) {
  Value *input = Builder.CreateCall(loadInput, args);
  if (!bCast)
    return input;
  else {
    Value *bVal = Builder.CreateICmpNE(input, zero);
    IntegerType *IT = cast<IntegerType>(Ty);
    if (IT->getBitWidth() == 1)
      return bVal;
    else
      return Builder.CreateZExt(bVal, Ty);
  }
}


Value *replaceLdWithLdInput(Function *loadInput, LoadInst *ldInst,
                            unsigned cols, MutableArrayRef<Value *> args,
                            bool bCast) {
  IRBuilder<> Builder(ldInst);
  IRBuilder<> AllocaBuilder(dxilutil::FindAllocaInsertionPt(ldInst));
  Type *Ty = ldInst->getType();
  Type *EltTy = Ty->getScalarType();
  // Change i1 to i32 for load input.
  Value *zero = Builder.getInt32(0);

  if (VectorType *VT = dyn_cast<VectorType>(Ty)) {
    Value *newVec = llvm::UndefValue::get(VT);
    DXASSERT(cols == VT->getNumElements(), "vec size must match");
    for (unsigned col = 0; col < cols; col++) {
      Value *colIdx = Builder.getInt8(col);
      args[DXIL::OperandIndex::kLoadInputColOpIdx] = colIdx;
      Value *input =
          GenerateLdInput(loadInput, args, Builder, zero, bCast, EltTy);
      newVec = Builder.CreateInsertElement(newVec, input, col);
    }
    ldInst->replaceAllUsesWith(newVec);
    ldInst->eraseFromParent();
    return newVec;
  } else {
    Value *colIdx = args[DXIL::OperandIndex::kLoadInputColOpIdx];
    if (colIdx == nullptr) {
      DXASSERT(cols == 1, "only support scalar here");
      colIdx = Builder.getInt8(0);
    } else {
      if (colIdx->getType() == Builder.getInt32Ty()) {
        colIdx = Builder.CreateTrunc(colIdx, Builder.getInt8Ty());
      }
    }

    if (isa<ConstantInt>(colIdx)) {
      args[DXIL::OperandIndex::kLoadInputColOpIdx] = colIdx;
      Value *input =
          GenerateLdInput(loadInput, args, Builder, zero, bCast, EltTy);
      ldInst->replaceAllUsesWith(input);
      ldInst->eraseFromParent();
      return input;
    } else {
      // Vector indexing.
      // Load to array.
      ArrayType *AT = ArrayType::get(ldInst->getType(), cols);
      Value *arrayVec = AllocaBuilder.CreateAlloca(AT);
      Value *zeroIdx = Builder.getInt32(0);

      for (unsigned col = 0; col < cols; col++) {
        Value *colIdx = Builder.getInt8(col);
        args[DXIL::OperandIndex::kLoadInputColOpIdx] = colIdx;
        Value *input =
            GenerateLdInput(loadInput, args, Builder, zero, bCast, EltTy);
        Value *GEP = Builder.CreateInBoundsGEP(arrayVec, {zeroIdx, colIdx});
        Builder.CreateStore(input, GEP);
      }
      Value *vecIndexingPtr =
          Builder.CreateInBoundsGEP(arrayVec, {zeroIdx, colIdx});
      Value *input = Builder.CreateLoad(vecIndexingPtr);
      ldInst->replaceAllUsesWith(input);
      ldInst->eraseFromParent();
      return input;
    }
  }
}


void replaceMatStWithStOutputs(CallInst *CI, HLMatLoadStoreOpcode matOp,
                               Function *ldStFunc, Constant *OpArg, Constant *ID,
                               Constant *columnConsts[],Value *vertexOrPrimID,
                               Value *idxVal) {
  IRBuilder<> LocalBuilder(CI);
  Value *Val = CI->getArgOperand(HLOperandIndex::kMatStoreValOpIdx);
  HLMatrixType MatTy = HLMatrixType::cast(
                                          CI->getArgOperand(HLOperandIndex::kMatStoreDstPtrOpIdx)
                                          ->getType()->getPointerElementType());

  Val = MatTy.emitLoweredRegToMem(Val, LocalBuilder);

  if (matOp == HLMatLoadStoreOpcode::ColMatStore) {
    for (unsigned c = 0; c < MatTy.getNumColumns(); c++) {
      Constant *constColIdx = LocalBuilder.getInt32(c);
      Value *colIdx = LocalBuilder.CreateAdd(idxVal, constColIdx);

      for (unsigned r = 0; r < MatTy.getNumRows(); r++) {
        unsigned matIdx = MatTy.getColumnMajorIndex(r, c);
        Value *Elt = LocalBuilder.CreateExtractElement(Val, matIdx);

        SmallVector<Value*, 6> argList = {OpArg, ID, colIdx, columnConsts[r], Elt};
        if (vertexOrPrimID)
          argList.emplace_back(vertexOrPrimID);
        LocalBuilder.CreateCall(ldStFunc, argList);
      }
    }
  } else {
    for (unsigned r = 0; r < MatTy.getNumRows(); r++) {
      Constant *constRowIdx = LocalBuilder.getInt32(r);
      Value *rowIdx = LocalBuilder.CreateAdd(idxVal, constRowIdx);
      for (unsigned c = 0; c < MatTy.getNumColumns(); c++) {
        unsigned matIdx = MatTy.getRowMajorIndex(r, c);
        Value *Elt = LocalBuilder.CreateExtractElement(Val, matIdx);

        SmallVector<Value*, 6> argList = {OpArg, ID, rowIdx, columnConsts[c], Elt};
        if (vertexOrPrimID)
          argList.emplace_back(vertexOrPrimID);
        LocalBuilder.CreateCall(ldStFunc, argList);
      }
    }
  }
  CI->eraseFromParent();
}


void replaceMatLdWithLdInputs(CallInst *CI, HLMatLoadStoreOpcode matOp,
                              Function *ldStFunc, Constant *OpArg, Constant *ID,
                              Constant *columnConsts[],Value *vertexOrPrimID,
                              Value *idxVal) {
  IRBuilder<> LocalBuilder(CI);
  HLMatrixType MatTy = HLMatrixType::cast(
                                          CI->getArgOperand(HLOperandIndex::kMatLoadPtrOpIdx)
                                          ->getType()->getPointerElementType());
  std::vector<Value *> matElts(MatTy.getNumElements());

  if (matOp == HLMatLoadStoreOpcode::ColMatLoad) {
    for (unsigned c = 0; c < MatTy.getNumColumns(); c++) {
      Constant *constRowIdx = LocalBuilder.getInt32(c);
      Value *rowIdx = LocalBuilder.CreateAdd(idxVal, constRowIdx);
      for (unsigned r = 0; r < MatTy.getNumRows(); r++) {
        SmallVector<Value *, 4> args = { OpArg, ID, rowIdx, columnConsts[r] };
        if (vertexOrPrimID)
          args.emplace_back(vertexOrPrimID);

        Value *input = LocalBuilder.CreateCall(ldStFunc, args);
        unsigned matIdx = MatTy.getColumnMajorIndex(r, c);
        matElts[matIdx] = input;
      }
    }
  } else {
    for (unsigned r = 0; r < MatTy.getNumRows(); r++) {
      Constant *constRowIdx = LocalBuilder.getInt32(r);
      Value *rowIdx = LocalBuilder.CreateAdd(idxVal, constRowIdx);
      for (unsigned c = 0; c < MatTy.getNumColumns(); c++) {
        SmallVector<Value *, 4> args = { OpArg, ID, rowIdx, columnConsts[c] };
        if (vertexOrPrimID)
          args.emplace_back(vertexOrPrimID);

        Value *input = LocalBuilder.CreateCall(ldStFunc, args);
        unsigned matIdx = MatTy.getRowMajorIndex(r, c);
        matElts[matIdx] = input;
      }
    }
  }

  Value *newVec =
    HLMatrixLower::BuildVector(matElts[0]->getType(), matElts, LocalBuilder);
  newVec = MatTy.emitLoweredMemToReg(newVec, LocalBuilder);

  CI->replaceAllUsesWith(newVec);
  CI->eraseFromParent();
}


void replaceDirectInputParameter(Value *param, Function *loadInput,
                                 unsigned cols, MutableArrayRef<Value *> args,
                                 bool bCast, OP *hlslOP, IRBuilder<> &Builder) {
  Value *zero = hlslOP->GetU32Const(0);
  Type *Ty = param->getType();
  Type *EltTy = Ty->getScalarType();

  if (VectorType *VT = dyn_cast<VectorType>(Ty)) {
    Value *newVec = llvm::UndefValue::get(VT);
    DXASSERT(cols == VT->getNumElements(), "vec size must match");

    for (unsigned col = 0; col < cols; col++) {
      Value *colIdx = hlslOP->GetU8Const(col);
      args[DXIL::OperandIndex::kLoadInputColOpIdx] = colIdx;
      Value *input =
          GenerateLdInput(loadInput, args, Builder, zero, bCast, EltTy);
      newVec = Builder.CreateInsertElement(newVec, input, col);
    }

    param->replaceAllUsesWith(newVec);

    // THe individual loadInputs are the authoritative source of values for the vector.
    dxilutil::TryScatterDebugValueToVectorElements(newVec);
  } else if (!Ty->isArrayTy() && !HLMatrixType::isa(Ty)) {
    DXASSERT(cols == 1, "only support scalar here");
    Value *colIdx = hlslOP->GetU8Const(0);
    args[DXIL::OperandIndex::kLoadInputColOpIdx] = colIdx;
    Value *input =
        GenerateLdInput(loadInput, args, Builder, zero, bCast, EltTy);
    param->replaceAllUsesWith(input); // Will properly relocate any DbgValueInst
  } else if (HLMatrixType::isa(Ty)) {
    if (param->use_empty()) return;
    DXASSERT(param->hasOneUse(),
             "matrix arg should only has one use as matrix to vec");
    CallInst *CI = cast<CallInst>(param->user_back());
    HLOpcodeGroup group = GetHLOpcodeGroupByName(CI->getCalledFunction());
    DXASSERT_LOCALVAR(group, group == HLOpcodeGroup::HLCast,
                      "must be hlcast here");
    unsigned opcode = GetHLOpcode(CI);
    HLCastOpcode matOp = static_cast<HLCastOpcode>(opcode);
    switch (matOp) {
    case HLCastOpcode::ColMatrixToVecCast: {
      IRBuilder<> LocalBuilder(CI);
      HLMatrixType MatTy = HLMatrixType::cast(
          CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx)->getType());
      Type *EltTy = MatTy.getElementTypeForReg();
      std::vector<Value *> matElts(MatTy.getNumElements());
      for (unsigned c = 0; c < MatTy.getNumColumns(); c++) {
        Value *rowIdx = hlslOP->GetI32Const(c);
        args[DXIL::OperandIndex::kLoadInputRowOpIdx] = rowIdx;
        for (unsigned r = 0; r < MatTy.getNumRows(); r++) {
          Value *colIdx = hlslOP->GetU8Const(r);
          args[DXIL::OperandIndex::kLoadInputColOpIdx] = colIdx;
          Value *input =
              GenerateLdInput(loadInput, args, Builder, zero, bCast, EltTy);
          matElts[MatTy.getColumnMajorIndex(r, c)] = input;
        }
      }
      Value *newVec =
          HLMatrixLower::BuildVector(EltTy, matElts, LocalBuilder);
      CI->replaceAllUsesWith(newVec);
      CI->eraseFromParent();
    } break;
    case HLCastOpcode::RowMatrixToVecCast: {
      IRBuilder<> LocalBuilder(CI);
      HLMatrixType MatTy = HLMatrixType::cast(
          CI->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx)->getType());
      Type *EltTy = MatTy.getElementTypeForReg();
      std::vector<Value *> matElts(MatTy.getNumElements());
      for (unsigned r = 0; r < MatTy.getNumRows(); r++) {
        Value *rowIdx = hlslOP->GetI32Const(r);
        args[DXIL::OperandIndex::kLoadInputRowOpIdx] = rowIdx;
        for (unsigned c = 0; c < MatTy.getNumColumns(); c++) {
          Value *colIdx = hlslOP->GetU8Const(c);
          args[DXIL::OperandIndex::kLoadInputColOpIdx] = colIdx;
          Value *input =
              GenerateLdInput(loadInput, args, Builder, zero, bCast, EltTy);
          matElts[MatTy.getRowMajorIndex(r, c)] = input;
        }
      }
      Value *newVec =
          HLMatrixLower::BuildVector(EltTy, matElts, LocalBuilder);
      CI->replaceAllUsesWith(newVec);
      CI->eraseFromParent();
    } break;
    default:
      // Only matrix to vector casts are valid.
      break;
    }
  } else {
    DXASSERT(0, "invalid type for direct input");
  }
}

struct InputOutputAccessInfo {
  // For input output which has only 1 row, idx is 0.
  Value *idx;
  // VertexID for HS/DS/GS input, MS vertex output. PrimitiveID for MS primitive output
  Value *vertexOrPrimID;
  // Vector index.
  Value *vectorIdx;
  // Load/Store/LoadMat/StoreMat on input/output.
  Instruction *user;
  InputOutputAccessInfo(Value *index, Instruction *I)
      : idx(index), vertexOrPrimID(nullptr), vectorIdx(nullptr), user(I) {}
  InputOutputAccessInfo(Value *index, Instruction *I, Value *ID, Value *vecIdx)
      : idx(index), vertexOrPrimID(ID), vectorIdx(vecIdx), user(I) {}
};

void collectInputOutputAccessInfo(
    Value *GV, Constant *constZero,
    std::vector<InputOutputAccessInfo> &accessInfoList, bool hasVertexOrPrimID,
    bool bInput, bool bRowMajor, bool isMS) {
  // merge GEP use for input output.
  dxilutil::MergeGepUse(GV);
  for (auto User = GV->user_begin(); User != GV->user_end();) {
    Value *I = *(User++);
    if (LoadInst *ldInst = dyn_cast<LoadInst>(I)) {
      if (bInput) {
        InputOutputAccessInfo info = {constZero, ldInst};
        accessInfoList.push_back(info);
      }
    } else if (StoreInst *stInst = dyn_cast<StoreInst>(I)) {
      if (!bInput) {
        InputOutputAccessInfo info = {constZero, stInst};
        accessInfoList.push_back(info);
      }
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
      // Vector indexing may has more indices.
      // Vector indexing changed to array indexing in SROA_HLSL.
      auto idx = GEP->idx_begin();
      DXASSERT_LOCALVAR(idx, idx->get() == constZero,
                        "only support 0 offset for input pointer");

      Value *vertexOrPrimID = nullptr;
      Value *vectorIdx = nullptr;
      gep_type_iterator GEPIt = gep_type_begin(GEP), E = gep_type_end(GEP);

      // Skip first pointer idx which must be 0.
      GEPIt++;
      if (hasVertexOrPrimID) {
        // Save vertexOrPrimID.
        vertexOrPrimID = GEPIt.getOperand();
        GEPIt++;
      }
      // Start from first index.
      Value *rowIdx = GEPIt.getOperand();
      if (GEPIt != E) {
        if ((*GEPIt)->isVectorTy()) {
          // Vector indexing.
          rowIdx = constZero;
          vectorIdx = GEPIt.getOperand();
          DXASSERT_NOMSG((++GEPIt) == E);
        } else {
          // Array which may have vector indexing.
          // Highest dim index is saved in rowIdx,
          //  array size for highest dim not affect index.
          GEPIt++;
          IRBuilder<> Builder(GEP);
          Type *idxTy = rowIdx->getType();
          for (; GEPIt != E; ++GEPIt) {
            DXASSERT(!GEPIt->isStructTy(),
                     "Struct should be flattened SROA_Parameter_HLSL");
            DXASSERT(!GEPIt->isPointerTy(),
                     "not support pointer type in middle of GEP");
            if (GEPIt->isArrayTy()) {
              Constant *arraySize =
                  ConstantInt::get(idxTy, GEPIt->getArrayNumElements());
              rowIdx = Builder.CreateMul(rowIdx, arraySize);
              rowIdx = Builder.CreateAdd(rowIdx, GEPIt.getOperand());
            } else {
              Type *Ty = *GEPIt;
              DXASSERT_LOCALVAR(Ty, Ty->isVectorTy(),
                                "must be vector type here to index");
              // Save vector idx.
              vectorIdx = GEPIt.getOperand();
            }
          }
          if (HLMatrixType MatTy = HLMatrixType::dyn_cast(*GEPIt)) {
            Constant *arraySize = ConstantInt::get(idxTy, MatTy.getNumColumns());
            if (bRowMajor) {
              arraySize = ConstantInt::get(idxTy, MatTy.getNumRows());
            }
            rowIdx = Builder.CreateMul(rowIdx, arraySize);
          }
        }
      } else
        rowIdx = constZero;

      auto GepUser = GEP->user_begin();
      auto GepUserE = GEP->user_end();
      Value *idxVal = rowIdx;

      for (; GepUser != GepUserE;) {
        auto GepUserIt = GepUser++;
        if (LoadInst *ldInst = dyn_cast<LoadInst>(*GepUserIt)) {
          if (bInput) {
            InputOutputAccessInfo info = {idxVal, ldInst, vertexOrPrimID, vectorIdx};
            accessInfoList.push_back(info);
          }
        } else if (StoreInst *stInst = dyn_cast<StoreInst>(*GepUserIt)) {
          if (!bInput) {
            InputOutputAccessInfo info = {idxVal, stInst, vertexOrPrimID, vectorIdx};
            accessInfoList.push_back(info);
          }
        } else if (CallInst *CI = dyn_cast<CallInst>(*GepUserIt)) {
          HLOpcodeGroup group = GetHLOpcodeGroupByName(CI->getCalledFunction());
          DXASSERT_LOCALVAR(group, group == HLOpcodeGroup::HLMatLoadStore,
                            "input/output should only used by ld/st");
          HLMatLoadStoreOpcode opcode = (HLMatLoadStoreOpcode)GetHLOpcode(CI);
          if ((opcode == HLMatLoadStoreOpcode::ColMatLoad ||
               opcode == HLMatLoadStoreOpcode::RowMatLoad)
                  ? bInput
                  : !bInput) {
            InputOutputAccessInfo info = {idxVal, CI, vertexOrPrimID, vectorIdx};
            accessInfoList.push_back(info);
          }
        } else {
          DXASSERT(0, "input output should only used by ld/st");
        }
      }
    } else if (CallInst *CI = dyn_cast<CallInst>(I)) {
      InputOutputAccessInfo info = {constZero, CI};
      accessInfoList.push_back(info);
    } else {
      DXASSERT(0, "input output should only used by ld/st");
    }
  }
}

void GenerateInputOutputUserCall(InputOutputAccessInfo &info, Value *undefVertexIdx,
    Function *ldStFunc, Constant *OpArg, Constant *ID, unsigned cols, bool bI1Cast,
    Constant *columnConsts[],
    bool bNeedVertexOrPrimID, bool isArrayTy, bool bInput, bool bIsInout) {
  Value *idxVal = info.idx;
  Value *vertexOrPrimID = undefVertexIdx;
  if (bNeedVertexOrPrimID && isArrayTy) {
    vertexOrPrimID = info.vertexOrPrimID;
  }

  if (LoadInst *ldInst = dyn_cast<LoadInst>(info.user)) {
    SmallVector<Value *, 4> args = {OpArg, ID, idxVal, info.vectorIdx};
    if (vertexOrPrimID)
      args.emplace_back(vertexOrPrimID);

    replaceLdWithLdInput(ldStFunc, ldInst, cols, args, bI1Cast);
  } else if (StoreInst *stInst = dyn_cast<StoreInst>(info.user)) {
    if (bInput) {
      DXASSERT_LOCALVAR(bIsInout, bIsInout, "input should not have store use.");
    } else {
      if (!info.vectorIdx) {
        replaceStWithStOutput(ldStFunc, stInst, OpArg, ID, idxVal, cols,
                              vertexOrPrimID, bI1Cast);
      } else {
        Value *V = stInst->getValueOperand();
        Type *Ty = V->getType();
        DXASSERT_LOCALVAR(Ty == Ty->getScalarType() && !Ty->isAggregateType(),
                          Ty, "only support scalar here");

        if (ConstantInt *ColIdx = dyn_cast<ConstantInt>(info.vectorIdx)) {
          IRBuilder<> Builder(stInst);
          if (ColIdx->getType()->getBitWidth() != 8) {
            ColIdx = Builder.getInt8(ColIdx->getValue().getLimitedValue());
          }
          SmallVector<Value *, 6> args = {OpArg, ID, idxVal, ColIdx, V};
          if (vertexOrPrimID)
            args.emplace_back(vertexOrPrimID);
          GenerateStOutput(ldStFunc, args, Builder, bI1Cast);
        } else {
          BasicBlock *BB = stInst->getParent();
          BasicBlock *EndBB = BB->splitBasicBlock(stInst);

          TerminatorInst *TI = BB->getTerminator();
          IRBuilder<> SwitchBuilder(TI);
          LLVMContext &Ctx = stInst->getContext();
          SwitchInst *Switch =
              SwitchBuilder.CreateSwitch(info.vectorIdx, EndBB, cols);
          TI->eraseFromParent();

          Function *F = EndBB->getParent();
          for (unsigned i = 0; i < cols; i++) {
            BasicBlock *CaseBB = BasicBlock::Create(Ctx, "case", F, EndBB);
            Switch->addCase(SwitchBuilder.getInt32(i), CaseBB);
            IRBuilder<> CaseBuilder(CaseBB);

            ConstantInt *CaseIdx = SwitchBuilder.getInt8(i);

            SmallVector<Value *, 6> args = {OpArg, ID, idxVal, CaseIdx, V};
            if (vertexOrPrimID)
              args.emplace_back(vertexOrPrimID);
            GenerateStOutput(ldStFunc, args, CaseBuilder, bI1Cast);

            CaseBuilder.CreateBr(EndBB);
          }
        }
        // remove stInst
        stInst->eraseFromParent();
      }
    }
  } else if (CallInst *CI = dyn_cast<CallInst>(info.user)) {
    HLOpcodeGroup group = GetHLOpcodeGroupByName(CI->getCalledFunction());
    // Intrinsic will be translated later.
    if (group == HLOpcodeGroup::HLIntrinsic || group == HLOpcodeGroup::NotHL)
      return;
    unsigned opcode = GetHLOpcode(CI);
    DXASSERT_NOMSG(group == HLOpcodeGroup::HLMatLoadStore);
    HLMatLoadStoreOpcode matOp = static_cast<HLMatLoadStoreOpcode>(opcode);
    switch (matOp) {
    case HLMatLoadStoreOpcode::ColMatLoad:
    case HLMatLoadStoreOpcode::RowMatLoad: {
      replaceMatLdWithLdInputs(CI, matOp, ldStFunc, OpArg, ID, columnConsts, vertexOrPrimID, idxVal);
    } break;
    case HLMatLoadStoreOpcode::ColMatStore:
    case HLMatLoadStoreOpcode::RowMatStore: {
      replaceMatStWithStOutputs(CI, matOp, ldStFunc, OpArg, ID, columnConsts, vertexOrPrimID, idxVal);
    } break;
    }
  } else {
    DXASSERT(0, "invalid operation on input output");
  }
}

} // namespace

void HLSignatureLower::GenerateDxilInputs() {
  GenerateDxilInputsOutputs(DXIL::SignatureKind::Input);
}

void HLSignatureLower::GenerateDxilOutputs() {
  GenerateDxilInputsOutputs(DXIL::SignatureKind::Output);
}

void HLSignatureLower::GenerateDxilPrimOutputs() {
  GenerateDxilInputsOutputs(DXIL::SignatureKind::PatchConstOrPrim);
}

void HLSignatureLower::GenerateDxilInputsOutputs(DXIL::SignatureKind SK) {
  OP *hlslOP = HLM.GetOP();
  DxilFunctionProps &props = HLM.GetDxilFunctionProps(Entry);
  Module &M = *(HLM.GetModule());

  OP::OpCode opcode = (OP::OpCode)-1;
  switch (SK) {
  case DXIL::SignatureKind::Input:
    opcode = OP::OpCode::LoadInput;
    break;
  case DXIL::SignatureKind::Output:
    opcode = props.IsMS() ? OP::OpCode::StoreVertexOutput : OP::OpCode::StoreOutput;
    break;
  case DXIL::SignatureKind::PatchConstOrPrim:
    opcode = OP::OpCode::StorePrimitiveOutput;
    break;
  default:
    DXASSERT_NOMSG(0);
  }
  bool bInput = SK == DXIL::SignatureKind::Input;
  bool bNeedVertexOrPrimID = bInput && (props.IsGS() || props.IsDS() || props.IsHS());
  bNeedVertexOrPrimID |= !bInput && props.IsMS();

  Constant *OpArg = hlslOP->GetU32Const((unsigned)opcode);

  Constant *columnConsts[] = {
      hlslOP->GetU8Const(0),  hlslOP->GetU8Const(1),  hlslOP->GetU8Const(2),
      hlslOP->GetU8Const(3),  hlslOP->GetU8Const(4),  hlslOP->GetU8Const(5),
      hlslOP->GetU8Const(6),  hlslOP->GetU8Const(7),  hlslOP->GetU8Const(8),
      hlslOP->GetU8Const(9),  hlslOP->GetU8Const(10), hlslOP->GetU8Const(11),
      hlslOP->GetU8Const(12), hlslOP->GetU8Const(13), hlslOP->GetU8Const(14),
      hlslOP->GetU8Const(15)};

  Constant *constZero = hlslOP->GetU32Const(0);

  Value *undefVertexIdx = props.IsMS() || !bInput ? nullptr : UndefValue::get(Type::getInt32Ty(HLM.GetCtx()));

  DxilSignature &Sig =
      bInput ? EntrySig.InputSignature :
      SK == DXIL::SignatureKind::Output ? EntrySig.OutputSignature :
      EntrySig.PatchConstOrPrimSignature;

  DxilTypeSystem &typeSys = HLM.GetTypeSystem();
  DxilFunctionAnnotation *pFuncAnnot = typeSys.GetFunctionAnnotation(Entry);

  Type *i1Ty = Type::getInt1Ty(constZero->getContext());
  Type *i32Ty = constZero->getType();

  llvm::SmallVector<unsigned, 8> removeIndices;
  for (unsigned i = 0; i < Sig.GetElements().size(); i++) {
    DxilSignatureElement *SE = &Sig.GetElement(i);
    llvm::Type *Ty = SE->GetCompType().GetLLVMType(HLM.GetCtx());
    // Cast i1 to i32 for load input.
    bool bI1Cast = false;
    if (Ty == i1Ty) {
      bI1Cast = true;
      Ty = i32Ty;
    }
    if (!hlslOP->IsOverloadLegal(opcode, Ty)) {
      std::string O;
      raw_string_ostream OSS(O);
      Ty->print(OSS);
      OSS << "(type for " << SE->GetName() << ")";
      OSS << " cannot be used as shader inputs or outputs.";
      OSS.flush();
      dxilutil::EmitErrorOnFunction(M.getContext(), Entry, O);
      continue;
    }
    Function *dxilFunc = hlslOP->GetOpFunc(opcode, Ty);
    Constant *ID = hlslOP->GetU32Const(i);
    unsigned cols = SE->GetCols();
    Value *GV = m_sigValueMap[SE];

    bool bIsInout = m_inoutArgSet.count(GV) > 0;

    IRBuilder<> EntryBuilder(Entry->getEntryBlock().getFirstInsertionPt());

    if (DbgDeclareInst *DDI = llvm::FindAllocaDbgDeclare(GV)) {
      EntryBuilder.SetCurrentDebugLocation(DDI->getDebugLoc());
    }

    DXIL::SemanticInterpretationKind SI = SE->GetInterpretation();
    DXASSERT_NOMSG(SI < DXIL::SemanticInterpretationKind::Invalid);
    DXASSERT_NOMSG(SI != DXIL::SemanticInterpretationKind::NA);
    DXASSERT_NOMSG(SI != DXIL::SemanticInterpretationKind::NotInSig);
    if (SI == DXIL::SemanticInterpretationKind::Shadow)
      continue; // Handled in ProcessArgument

    if (!GV->getType()->isPointerTy()) {
      DXASSERT(bInput, "direct parameter must be input");
      Value *vertexOrPrimID = undefVertexIdx;
      Value *args[] = {OpArg, ID, /*rowIdx*/ constZero, /*colIdx*/ nullptr,
                       vertexOrPrimID};
      replaceDirectInputParameter(GV, dxilFunc, cols, args, bI1Cast, hlslOP,
                                  EntryBuilder);
      continue;
    }

    bool bIsArrayTy = GV->getType()->getPointerElementType()->isArrayTy();
    bool bIsPrecise = m_preciseSigSet.count(SE);
    if (bIsPrecise)
      HLModule::MarkPreciseAttributeOnPtrWithFunctionCall(GV, M);
    bool bRowMajor = false;
    if (Argument *Arg = dyn_cast<Argument>(GV)) {
      if (pFuncAnnot) {
        auto &paramAnnot = pFuncAnnot->GetParameterAnnotation(Arg->getArgNo());
        if (paramAnnot.HasMatrixAnnotation())
          bRowMajor = paramAnnot.GetMatrixAnnotation().Orientation ==
                      MatrixOrientation::RowMajor;
      }
    }
    std::vector<InputOutputAccessInfo> accessInfoList;
    collectInputOutputAccessInfo(GV, constZero, accessInfoList,
                                 bNeedVertexOrPrimID && bIsArrayTy, bInput, bRowMajor, props.IsMS());

    for (InputOutputAccessInfo &info : accessInfoList) {
      GenerateInputOutputUserCall(info, undefVertexIdx, dxilFunc, OpArg, ID,
                                  cols, bI1Cast, columnConsts, bNeedVertexOrPrimID,
                                  bIsArrayTy, bInput, bIsInout);
    }
  }
}

void HLSignatureLower::GenerateDxilCSInputs() {
  OP *hlslOP = HLM.GetOP();

  DxilFunctionAnnotation *funcAnnotation = HLM.GetFunctionAnnotation(Entry);
  DXASSERT(funcAnnotation, "must find annotation for entry function");
  IRBuilder<> Builder(Entry->getEntryBlock().getFirstInsertionPt());

  for (Argument &arg : Entry->args()) {
    DxilParameterAnnotation &paramAnnotation =
        funcAnnotation->GetParameterAnnotation(arg.getArgNo());

    llvm::StringRef semanticStr = paramAnnotation.GetSemanticString();
    if (semanticStr.empty()) {
      dxilutil::EmitErrorOnFunction(HLM.GetModule()->getContext(), Entry, "Semantic must be defined for all "
                                    "parameters of an entry function or patch "
                                    "constant function.");
      return;
    }

    const Semantic *semantic =
        Semantic::GetByName(semanticStr, DXIL::SigPointKind::CSIn);
    OP::OpCode opcode;
    switch (semantic->GetKind()) {
    case Semantic::Kind::GroupThreadID:
      opcode = OP::OpCode::ThreadIdInGroup;
      break;
    case Semantic::Kind::GroupID:
      opcode = OP::OpCode::GroupId;
      break;
    case Semantic::Kind::DispatchThreadID:
      opcode = OP::OpCode::ThreadId;
      break;
    case Semantic::Kind::GroupIndex:
      opcode = OP::OpCode::FlattenedThreadIdInGroup;
      break;
    default:
      DXASSERT(semantic->IsInvalid(),
               "else compute shader semantics out-of-date");
      dxilutil::EmitErrorOnFunction(HLM.GetModule()->getContext(), Entry, "invalid semantic found in CS");
      return;
    }

    Constant *OpArg = hlslOP->GetU32Const((unsigned)opcode);
    Type *NumTy = arg.getType();
    DXASSERT(!NumTy->isPointerTy(), "Unexpected byref value for CS SV_***ID semantic.");
    DXASSERT(NumTy->getScalarType()->isIntegerTy(), "Unexpected non-integer value for CS SV_***ID semantic.");

    // Always use the i32 overload of those intrinsics, and then cast as needed
    Function *dxilFunc = hlslOP->GetOpFunc(opcode, Builder.getInt32Ty());
    Value *newArg = nullptr;
    if (opcode == OP::OpCode::FlattenedThreadIdInGroup) {
      newArg = Builder.CreateCall(dxilFunc, {OpArg});
    } else {
      unsigned vecSize = 1;
      if (FixedVectorType *VT = dyn_cast<FixedVectorType>(NumTy))
        vecSize = VT->getNumElements();

      newArg = Builder.CreateCall(dxilFunc, {OpArg, hlslOP->GetU32Const(0)});
      if (vecSize > 1) {
        Value *result = UndefValue::get(VectorType::get(Builder.getInt32Ty(), vecSize));
        result = Builder.CreateInsertElement(result, newArg, (uint64_t)0);

        for (unsigned i = 1; i < vecSize; i++) {
          Value *newElt =
              Builder.CreateCall(dxilFunc, {OpArg, hlslOP->GetU32Const(i)});
          result = Builder.CreateInsertElement(result, newElt, i);
        }
        newArg = result;
      }
    }

    // If the argument is of non-i32 type, convert here
    if (newArg->getType() != NumTy)
      newArg = Builder.CreateZExtOrTrunc(newArg, NumTy);

    if (newArg->getType() != arg.getType()) {
      DXASSERT_NOMSG(arg.getType()->isPointerTy());
      for (User *U : arg.users()) {
        LoadInst *LI = cast<LoadInst>(U);
        LI->replaceAllUsesWith(newArg);
      }
    } else {
      arg.replaceAllUsesWith(newArg);
    }
  }
}

void HLSignatureLower::GenerateDxilPatchConstantLdSt() {
  OP *hlslOP = HLM.GetOP();
  DxilFunctionProps &props = HLM.GetDxilFunctionProps(Entry);
  Module &M = *(HLM.GetModule());
  Constant *constZero = hlslOP->GetU32Const(0);
  DxilSignature &Sig = EntrySig.PatchConstOrPrimSignature;
  DxilTypeSystem &typeSys = HLM.GetTypeSystem();
  DxilFunctionAnnotation *pFuncAnnot = typeSys.GetFunctionAnnotation(Entry);
  auto InsertPt = Entry->getEntryBlock().getFirstInsertionPt();
  const bool bIsHs = props.IsHS();
  const bool bIsInput = !bIsHs;
  const bool bIsInout = false;
  const bool bNeedVertexOrPrimID = false;
  if (bIsHs) {
    DxilFunctionProps &EntryQual = HLM.GetDxilFunctionProps(Entry);
    Function *patchConstantFunc = EntryQual.ShaderProps.HS.patchConstantFunc;
    InsertPt = patchConstantFunc->getEntryBlock().getFirstInsertionPt();
    pFuncAnnot = typeSys.GetFunctionAnnotation(patchConstantFunc);
  }
  IRBuilder<> Builder(InsertPt);
  Type *i1Ty = Builder.getInt1Ty();
  Type *i32Ty = Builder.getInt32Ty();
  // LoadPatchConst don't have vertexIdx operand.
  Value *undefVertexIdx = nullptr;

  Constant *columnConsts[] = {
      hlslOP->GetU8Const(0),  hlslOP->GetU8Const(1),  hlslOP->GetU8Const(2),
      hlslOP->GetU8Const(3),  hlslOP->GetU8Const(4),  hlslOP->GetU8Const(5),
      hlslOP->GetU8Const(6),  hlslOP->GetU8Const(7),  hlslOP->GetU8Const(8),
      hlslOP->GetU8Const(9),  hlslOP->GetU8Const(10), hlslOP->GetU8Const(11),
      hlslOP->GetU8Const(12), hlslOP->GetU8Const(13), hlslOP->GetU8Const(14),
      hlslOP->GetU8Const(15)};

  OP::OpCode opcode =
      bIsInput ? OP::OpCode::LoadPatchConstant : OP::OpCode::StorePatchConstant;
  Constant *OpArg = hlslOP->GetU32Const((unsigned)opcode);

  for (unsigned i = 0; i < Sig.GetElements().size(); i++) {
    DxilSignatureElement *SE = &Sig.GetElement(i);
    Value *GV = m_sigValueMap[SE];

    DXIL::SemanticInterpretationKind SI = SE->GetInterpretation();
    DXASSERT_NOMSG(SI < DXIL::SemanticInterpretationKind::Invalid);
    DXASSERT_NOMSG(SI != DXIL::SemanticInterpretationKind::NA);
    DXASSERT_NOMSG(SI != DXIL::SemanticInterpretationKind::NotInSig);
    if (SI == DXIL::SemanticInterpretationKind::Shadow)
      continue; // Handled in ProcessArgument

    Constant *ID = hlslOP->GetU32Const(i);
    // Generate LoadPatchConstant.
    Type *Ty = SE->GetCompType().GetLLVMType(HLM.GetCtx());
    // Cast i1 to i32 for load input.
    bool bI1Cast = false;
    if (Ty == i1Ty) {
      bI1Cast = true;
      Ty = i32Ty;
    }

    unsigned cols = SE->GetCols();

    Function *dxilFunc = hlslOP->GetOpFunc(opcode, Ty);

    if (!GV->getType()->isPointerTy()) {
      DXASSERT(bIsInput, "Must be DS input.");
      Constant *OpArg = hlslOP->GetU32Const(
          static_cast<unsigned>(OP::OpCode::LoadPatchConstant));
      Value *args[] = {OpArg, ID, /*rowIdx*/ constZero, /*colIdx*/ nullptr};
      replaceDirectInputParameter(GV, dxilFunc, cols, args, bI1Cast, hlslOP,
                                  Builder);
      continue;
    }
    bool bRowMajor = false;
    if (Argument *Arg = dyn_cast<Argument>(GV)) {
      if (pFuncAnnot) {
        auto &paramAnnot = pFuncAnnot->GetParameterAnnotation(Arg->getArgNo());
        if (paramAnnot.HasMatrixAnnotation())
          bRowMajor = paramAnnot.GetMatrixAnnotation().Orientation ==
                      MatrixOrientation::RowMajor;
      }
    }
    std::vector<InputOutputAccessInfo> accessInfoList;
    collectInputOutputAccessInfo(GV, constZero, accessInfoList, bNeedVertexOrPrimID,
                                 bIsInput, bRowMajor, false);

    bool bIsArrayTy = GV->getType()->getPointerElementType()->isArrayTy();
    bool isPrecise = m_preciseSigSet.count(SE);
    if (isPrecise)
      HLModule::MarkPreciseAttributeOnPtrWithFunctionCall(GV, M);

    for (InputOutputAccessInfo &info : accessInfoList) {
      GenerateInputOutputUserCall(info, undefVertexIdx, dxilFunc, OpArg, ID,
                                  cols, bI1Cast, columnConsts, bNeedVertexOrPrimID,
                                  bIsArrayTy, bIsInput, bIsInout);
    }
  }
}

void HLSignatureLower::GenerateDxilPatchConstantFunctionInputs() {
  // Map input patch, to input sig
  // LoadOutputControlPoint for output patch .
  OP *hlslOP = HLM.GetOP();
  Constant *constZero = hlslOP->GetU32Const(0);

  DxilFunctionProps &EntryQual = HLM.GetDxilFunctionProps(Entry);

  Function *patchConstantFunc = EntryQual.ShaderProps.HS.patchConstantFunc;
  DxilFunctionAnnotation *patchFuncAnnotation =
      HLM.GetFunctionAnnotation(patchConstantFunc);
  DXASSERT(patchFuncAnnotation,
           "must find annotation for patch constant function");
  Type *i1Ty = Type::getInt1Ty(constZero->getContext());
  Type *i32Ty = constZero->getType();

  Constant *columnConsts[] = {
      hlslOP->GetU8Const(0),  hlslOP->GetU8Const(1),  hlslOP->GetU8Const(2),
      hlslOP->GetU8Const(3),  hlslOP->GetU8Const(4),  hlslOP->GetU8Const(5),
      hlslOP->GetU8Const(6),  hlslOP->GetU8Const(7),  hlslOP->GetU8Const(8),
      hlslOP->GetU8Const(9),  hlslOP->GetU8Const(10), hlslOP->GetU8Const(11),
      hlslOP->GetU8Const(12), hlslOP->GetU8Const(13), hlslOP->GetU8Const(14),
      hlslOP->GetU8Const(15)};

  for (Argument &arg : patchConstantFunc->args()) {
    DxilParameterAnnotation &paramAnnotation =
        patchFuncAnnotation->GetParameterAnnotation(arg.getArgNo());
    DxilParamInputQual inputQual = paramAnnotation.GetParamInputQual();
    if (inputQual == DxilParamInputQual::InputPatch ||
        inputQual == DxilParamInputQual::OutputPatch) {
      DxilSignatureElement *SE = m_patchConstantInputsSigMap[arg.getArgNo()];
      if (!SE) // Error should have been reported at an earlier stage.
        continue;

      Constant *inputID = hlslOP->GetU32Const(SE->GetID());
      unsigned cols = SE->GetCols();
      Type *Ty = SE->GetCompType().GetLLVMType(HLM.GetCtx());
      // Cast i1 to i32 for load input.
      bool bI1Cast = false;
      if (Ty == i1Ty) {
        bI1Cast = true;
        Ty = i32Ty;
      }
      OP::OpCode opcode = inputQual == DxilParamInputQual::InputPatch
                              ? OP::OpCode::LoadInput
                              : OP::OpCode::LoadOutputControlPoint;
      Function *dxilLdFunc = hlslOP->GetOpFunc(opcode, Ty);
      bool bRowMajor = false;
      if (Argument *Arg = dyn_cast<Argument>(&arg)) {
        if (patchFuncAnnotation) {
          auto &paramAnnot = patchFuncAnnotation->GetParameterAnnotation(Arg->getArgNo());
          if (paramAnnot.HasMatrixAnnotation())
            bRowMajor = paramAnnot.GetMatrixAnnotation().Orientation ==
                      MatrixOrientation::RowMajor;
        }
      }
      std::vector<InputOutputAccessInfo> accessInfoList;
      collectInputOutputAccessInfo(&arg, constZero, accessInfoList,
                                   /*hasVertexOrPrimID*/ true, true, bRowMajor, false);
      for (InputOutputAccessInfo &info : accessInfoList) {
        Constant *OpArg = hlslOP->GetU32Const((unsigned)opcode);
        if (LoadInst *ldInst = dyn_cast<LoadInst>(info.user)) {
          Value *args[] = {OpArg, inputID, info.idx, info.vectorIdx,
                           info.vertexOrPrimID};
          replaceLdWithLdInput(dxilLdFunc, ldInst, cols, args, bI1Cast);
        } else if (CallInst *CI = dyn_cast<CallInst>(info.user)) {
          HLOpcodeGroup group = GetHLOpcodeGroupByName(CI->getCalledFunction());
          // Intrinsic will be translated later.
          if (group == HLOpcodeGroup::HLIntrinsic || group == HLOpcodeGroup::NotHL)
            return;
          unsigned opcode = GetHLOpcode(CI);
          DXASSERT_NOMSG(group == HLOpcodeGroup::HLMatLoadStore);
          HLMatLoadStoreOpcode matOp = static_cast<HLMatLoadStoreOpcode>(opcode);
          if (matOp == HLMatLoadStoreOpcode::ColMatLoad || matOp == HLMatLoadStoreOpcode::RowMatLoad)
            replaceMatLdWithLdInputs(CI, matOp, dxilLdFunc, OpArg, inputID, columnConsts, info.vertexOrPrimID, info.idx);
        } else {
          DXASSERT(0, "input should only be ld");
        }
      }
    }
  }
}

bool HLSignatureLower::HasClipPlanes() {
  if (!HLM.HasDxilFunctionProps(Entry))
    return false;

  DxilFunctionProps &EntryQual = HLM.GetDxilFunctionProps(Entry);
  auto &VS = EntryQual.ShaderProps.VS;
  unsigned numClipPlanes = 0;

  for (unsigned i = 0; i < DXIL::kNumClipPlanes; i++) {
    if (!VS.clipPlanes[i])
      break;
    numClipPlanes++;
  }

  return numClipPlanes != 0;
}

void HLSignatureLower::GenerateClipPlanesForVS(Value *outPosition) {
  DxilFunctionProps &EntryQual = HLM.GetDxilFunctionProps(Entry);
  auto &VS = EntryQual.ShaderProps.VS;
  unsigned numClipPlanes = 0;

  for (unsigned i = 0; i < DXIL::kNumClipPlanes; i++) {
    if (!VS.clipPlanes[i])
      break;
    numClipPlanes++;
  }

  if (!numClipPlanes)
    return;

  LLVMContext &Ctx = HLM.GetCtx();

  Function *dp4 =
      HLM.GetOP()->GetOpFunc(DXIL::OpCode::Dot4, Type::getFloatTy(Ctx));
  Value *dp4Args[] = {
      ConstantInt::get(Type::getInt32Ty(Ctx),
                       static_cast<unsigned>(DXIL::OpCode::Dot4)),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
  };

  // out SV_Position should only have StoreInst use.
  // Done by LegalizeDxilInputOutputs in ScalarReplAggregatesHLSL.cpp
  for (User *U : outPosition->users()) {
    StoreInst *ST = cast<StoreInst>(U);
    Value *posVal = ST->getValueOperand();
    DXASSERT(posVal->getType()->isVectorTy(), "SV_Position must be a vector");
    IRBuilder<> Builder(ST);
    // Put position to args.
    for (unsigned i = 0; i < 4; i++)
      dp4Args[i + 1] = Builder.CreateExtractElement(posVal, i);

    // For each clip plane.
    // clipDistance = dp4 position, clipPlane.
    auto argIt = Entry->getArgumentList().rbegin();

    for (int clipIdx = numClipPlanes - 1; clipIdx >= 0; clipIdx--) {
      Constant *GV = VS.clipPlanes[clipIdx];
      DXASSERT_NOMSG(GV->hasOneUse());
      StoreInst *ST = cast<StoreInst>(GV->user_back());
      Value *clipPlane = ST->getValueOperand();
      ST->eraseFromParent();

      Argument &arg = *(argIt++);

      // Put clipPlane to args.
      for (unsigned i = 0; i < 4; i++)
        dp4Args[i + 5] = Builder.CreateExtractElement(clipPlane, i);

      Value *clipDistance = Builder.CreateCall(dp4, dp4Args);
      Builder.CreateStore(clipDistance, &arg);
    }
  }
}

namespace {
Value *TranslateStreamAppend(CallInst *CI, unsigned ID, hlsl::OP *OP) {
  Function *DxilFunc = OP->GetOpFunc(OP::OpCode::EmitStream, CI->getType());
  // TODO: generate a emit which has the data being emited as its argment.
  // Value *data = CI->getArgOperand(HLOperandIndex::kStreamAppendDataOpIndex);
  Constant *opArg = OP->GetU32Const((unsigned)OP::OpCode::EmitStream);
  IRBuilder<> Builder(CI);

  Constant *streamID = OP->GetU8Const(ID);
  Value *args[] = {opArg, streamID};
  return Builder.CreateCall(DxilFunc, args);
}

Value *TranslateStreamCut(CallInst *CI, unsigned ID, hlsl::OP *OP) {
  Function *DxilFunc = OP->GetOpFunc(OP::OpCode::CutStream, CI->getType());
  // TODO: generate a emit which has the data being emited as its argment.
  // Value *data = CI->getArgOperand(HLOperandIndex::kStreamAppendDataOpIndex);
  Constant *opArg = OP->GetU32Const((unsigned)OP::OpCode::CutStream);
  IRBuilder<> Builder(CI);

  Constant *streamID = OP->GetU8Const(ID);
  Value *args[] = {opArg, streamID};
  return Builder.CreateCall(DxilFunc, args);
}

} // namespace

// Generate DXIL stream output operation.
void HLSignatureLower::GenerateStreamOutputOperation(Value *streamVal, unsigned ID) {
  OP * hlslOP = HLM.GetOP();

  for (auto U = streamVal->user_begin(); U != streamVal->user_end();) {
    Value *user = *(U++);
    // Should only used by append, restartStrip .
    CallInst *CI = cast<CallInst>(user);
    HLOpcodeGroup group = GetHLOpcodeGroupByName(CI->getCalledFunction());
    // Ignore user functions.
    if (group == HLOpcodeGroup::NotHL)
      continue;
    unsigned opcode = GetHLOpcode(CI);
    DXASSERT_LOCALVAR(group, group == HLOpcodeGroup::HLIntrinsic, "Must be HLIntrinsic here");
    IntrinsicOp IOP = static_cast<IntrinsicOp>(opcode);
    switch (IOP) {
    case IntrinsicOp::MOP_Append:
      TranslateStreamAppend(CI, ID, hlslOP);
    break;
    case IntrinsicOp::MOP_RestartStrip:
      TranslateStreamCut(CI, ID, hlslOP);
      break;
    default:
      DXASSERT(0, "invalid operation on stream");
    }
    CI->eraseFromParent();
  }
}
// Generate DXIL stream output operations.
void HLSignatureLower::GenerateStreamOutputOperations() {
  DxilFunctionAnnotation *EntryAnnotation = HLM.GetFunctionAnnotation(Entry);
  DXASSERT(EntryAnnotation, "must find annotation for entry function");

  for (Argument &arg : Entry->getArgumentList()) {
    if (HLModule::IsStreamOutputPtrType(arg.getType())) {
      unsigned streamID = 0;
      DxilParameterAnnotation &paramAnnotation =
          EntryAnnotation->GetParameterAnnotation(arg.getArgNo());
      DxilParamInputQual inputQual = paramAnnotation.GetParamInputQual();
      switch (inputQual) {
      case DxilParamInputQual::OutStream0:
        streamID = 0;
        break;
      case DxilParamInputQual::OutStream1:
        streamID = 1;
        break;
      case DxilParamInputQual::OutStream2:
        streamID = 2;
        break;
      case DxilParamInputQual::OutStream3:
      default:
        DXASSERT(inputQual == DxilParamInputQual::OutStream3,
                 "invalid input qual.");
        streamID = 3;
        break;
      }
      GenerateStreamOutputOperation(&arg, streamID);
    }
  }
}
// Generate DXIL EmitIndices operation.
void HLSignatureLower::GenerateEmitIndicesOperation(Value *indicesOutput) {
  OP * hlslOP = HLM.GetOP();
  Function *DxilFunc = hlslOP->GetOpFunc(OP::OpCode::EmitIndices, Type::getVoidTy(indicesOutput->getContext()));
  Constant *opArg = hlslOP->GetU32Const((unsigned)OP::OpCode::EmitIndices);

  for (auto U = indicesOutput->user_begin(); U != indicesOutput->user_end();) {
    Value *user = *(U++);
    GetElementPtrInst *GEP = cast<GetElementPtrInst>(user);
    auto idx = GEP->idx_begin();
    DXASSERT_LOCALVAR(idx, idx->get() == hlslOP->GetU32Const(0),
                      "only support 0 offset for input pointer");
    gep_type_iterator GEPIt = gep_type_begin(GEP), E = gep_type_end(GEP);

    // Skip first pointer idx which must be 0.
    GEPIt++;
    Value *primIdx = GEPIt.getOperand();
    DXASSERT(++GEPIt == E, "invalid GEP here"); (void)E;

    auto GepUser = GEP->user_begin();
    auto GepUserE = GEP->user_end();
    for (; GepUser != GepUserE;) {
      auto GepUserIt = GepUser++;
      StoreInst *stInst = cast<StoreInst>(*GepUserIt);
      Value *stVal = stInst->getValueOperand();
      VectorType *VT = cast<VectorType>(stVal->getType());
      unsigned eleCount = VT->getNumElements();
      IRBuilder<> Builder(stInst);
      Value *subVal0 = Builder.CreateExtractElement(stVal, hlslOP->GetU32Const(0));
      Value *subVal1 = Builder.CreateExtractElement(stVal, hlslOP->GetU32Const(1));
      Value *subVal2 = eleCount == 3 ?
        Builder.CreateExtractElement(stVal, hlslOP->GetU32Const(2)) : hlslOP->GetU32Const(0);
      Value *args[] = { opArg, primIdx, subVal0, subVal1, subVal2 };
      Builder.CreateCall(DxilFunc, args);
      stInst->eraseFromParent();
    }
    GEP->eraseFromParent();
  }
}
// Generate DXIL EmitIndices operations.
void HLSignatureLower::GenerateEmitIndicesOperations() {
  DxilFunctionAnnotation *EntryAnnotation = HLM.GetFunctionAnnotation(Entry);
  DXASSERT(EntryAnnotation, "must find annotation for entry function");

  for (Argument &arg : Entry->getArgumentList()) {
    DxilParameterAnnotation &paramAnnotation =
      EntryAnnotation->GetParameterAnnotation(arg.getArgNo());
    DxilParamInputQual inputQual = paramAnnotation.GetParamInputQual();
    if (inputQual == DxilParamInputQual::OutIndices) {
      GenerateEmitIndicesOperation(&arg);
    }
  }
}
// Generate DXIL GetMeshPayload operation.
void HLSignatureLower::GenerateGetMeshPayloadOperation() {
  DxilFunctionAnnotation *EntryAnnotation = HLM.GetFunctionAnnotation(Entry);
  DXASSERT(EntryAnnotation, "must find annotation for entry function");

  for (Argument &arg : Entry->getArgumentList()) {
    DxilParameterAnnotation &paramAnnotation =
      EntryAnnotation->GetParameterAnnotation(arg.getArgNo());
    DxilParamInputQual inputQual = paramAnnotation.GetParamInputQual();
    if (inputQual == DxilParamInputQual::InPayload) {
      OP * hlslOP = HLM.GetOP();
      Function *DxilFunc = hlslOP->GetOpFunc(OP::OpCode::GetMeshPayload, arg.getType());
      Constant *opArg = hlslOP->GetU32Const((unsigned)OP::OpCode::GetMeshPayload);
      IRBuilder<> Builder(arg.getParent()->getEntryBlock().getFirstInsertionPt());
      Value *args[] = { opArg };
      Value *payload = Builder.CreateCall(DxilFunc, args);
      arg.replaceAllUsesWith(payload);
    }
  }
}
// Lower signatures.
void HLSignatureLower::Run() {
  DxilFunctionProps &props = HLM.GetDxilFunctionProps(Entry);
  if (props.IsGraphics()) {
    if (props.IsMS()) {
      GenerateEmitIndicesOperations();
      GenerateGetMeshPayloadOperation();
    }
    CreateDxilSignatures();

    // Allocate input output.
    AllocateDxilInputOutputs();

    GenerateDxilInputs();
    GenerateDxilOutputs();
    if (props.IsMS()) {
      GenerateDxilPrimOutputs();
    }
  } else if (props.IsCS()) {
    GenerateDxilCSInputs();
  }

  if (props.IsDS() || props.IsHS())
    GenerateDxilPatchConstantLdSt();
  if (props.IsHS())
    GenerateDxilPatchConstantFunctionInputs();
  if (props.IsGS())
    GenerateStreamOutputOperations();
}
