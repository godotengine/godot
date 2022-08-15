///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilShaderFlags.cpp                                                       //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilShaderFlags.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilResource.h"
#include "dxc/DXIL/DxilResourceBinding.h"
#include "dxc/Support/Global.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/Casting.h"
#include "dxc/DXIL/DxilEntryProps.h"
#include "dxc/DXIL/DxilInstructions.h"
#include "dxc/DXIL/DxilResourceProperties.h"
#include "dxc/DXIL/DxilUtil.h"

using namespace hlsl;
using namespace llvm;

ShaderFlags::ShaderFlags():
  m_bDisableOptimizations(false)
, m_bDisableMathRefactoring(false)
, m_bEnableDoublePrecision(false)
, m_bForceEarlyDepthStencil(false)
, m_bEnableRawAndStructuredBuffers(false)
, m_bLowPrecisionPresent(false)
, m_bEnableDoubleExtensions(false)
, m_bEnableMSAD(false)
, m_bAllResourcesBound(false)
, m_bViewportAndRTArrayIndex(false)
, m_bInnerCoverage(false)
, m_bStencilRef(false)
, m_bTiledResources(false)
, m_bUAVLoadAdditionalFormats(false)
, m_bLevel9ComparisonFiltering(false)
, m_b64UAVs(false)
, m_UAVsAtEveryStage(false)
, m_bCSRawAndStructuredViaShader4X(false)
, m_bROVS(false)
, m_bWaveOps(false)
, m_bInt64Ops(false)
, m_bViewID(false)
, m_bBarycentrics(false)
, m_bUseNativeLowPrecision(false)
, m_bShadingRate(false)
, m_bRaytracingTier1_1(false)
, m_bSamplerFeedback(false)
, m_bAtomicInt64OnTypedResource(false)
, m_bAtomicInt64OnGroupShared(false)
, m_bDerivativesInMeshAndAmpShaders(false)
, m_bResourceDescriptorHeapIndexing(false)
, m_bSamplerDescriptorHeapIndexing(false)
, m_bAtomicInt64OnHeapResource(false)
, m_bResMayNotAlias(false)
, m_bAdvancedTextureOps(false)
, m_bWriteableMSAATextures(false)
, m_align1(0)
{
  // Silence unused field warnings
  (void)m_align1;
}

uint64_t ShaderFlags::GetFeatureInfo() const {
  uint64_t Flags = 0;
  Flags |= m_bEnableDoublePrecision ? hlsl::DXIL::ShaderFeatureInfo_Doubles : 0;
  Flags |= m_bLowPrecisionPresent && !m_bUseNativeLowPrecision
               ? hlsl::DXIL::ShaderFeatureInfo_MinimumPrecision
               : 0;
  Flags |= m_bLowPrecisionPresent && m_bUseNativeLowPrecision
               ? hlsl::DXIL::ShaderFeatureInfo_NativeLowPrecision
               : 0;
  Flags |= m_bEnableDoubleExtensions
               ? hlsl::DXIL::ShaderFeatureInfo_11_1_DoubleExtensions
               : 0;
  Flags |= m_bWaveOps ? hlsl::DXIL::ShaderFeatureInfo_WaveOps : 0;
  Flags |= m_bInt64Ops ? hlsl::DXIL::ShaderFeatureInfo_Int64Ops : 0;
  Flags |= m_bROVS ? hlsl::DXIL::ShaderFeatureInfo_ROVs : 0;
  Flags |=
      m_bViewportAndRTArrayIndex
          ? hlsl::DXIL::
                ShaderFeatureInfo_ViewportAndRTArrayIndexFromAnyShaderFeedingRasterizer
          : 0;
  Flags |= m_bInnerCoverage ? hlsl::DXIL::ShaderFeatureInfo_InnerCoverage : 0;
  Flags |= m_bStencilRef ? hlsl::DXIL::ShaderFeatureInfo_StencilRef : 0;
  Flags |= m_bTiledResources ? hlsl::DXIL::ShaderFeatureInfo_TiledResources : 0;
  Flags |=
      m_bEnableMSAD ? hlsl::DXIL::ShaderFeatureInfo_11_1_ShaderExtensions : 0;
  Flags |=
      m_bCSRawAndStructuredViaShader4X
          ? hlsl::DXIL::
                ShaderFeatureInfo_ComputeShadersPlusRawAndStructuredBuffersViaShader4X
          : 0;
  Flags |=
      m_UAVsAtEveryStage ? hlsl::DXIL::ShaderFeatureInfo_UAVsAtEveryStage : 0;
  Flags |= m_b64UAVs ? hlsl::DXIL::ShaderFeatureInfo_64UAVs : 0;
  Flags |= m_bLevel9ComparisonFiltering
               ? hlsl::DXIL::ShaderFeatureInfo_LEVEL9ComparisonFiltering
               : 0;
  Flags |= m_bUAVLoadAdditionalFormats
               ? hlsl::DXIL::ShaderFeatureInfo_TypedUAVLoadAdditionalFormats
               : 0;
  Flags |= m_bViewID ? hlsl::DXIL::ShaderFeatureInfo_ViewID : 0;
  Flags |= m_bBarycentrics ? hlsl::DXIL::ShaderFeatureInfo_Barycentrics : 0;
  Flags |= m_bShadingRate ? hlsl::DXIL::ShaderFeatureInfo_ShadingRate : 0;
  Flags |= m_bRaytracingTier1_1 ? hlsl::DXIL::ShaderFeatureInfo_Raytracing_Tier_1_1 : 0;
  Flags |= m_bSamplerFeedback ? hlsl::DXIL::ShaderFeatureInfo_SamplerFeedback : 0;
  Flags |= m_bAtomicInt64OnTypedResource ? hlsl::DXIL::ShaderFeatureInfo_AtomicInt64OnTypedResource : 0;
  Flags |= m_bAtomicInt64OnGroupShared ? hlsl::DXIL::ShaderFeatureInfo_AtomicInt64OnGroupShared : 0;
  Flags |= m_bDerivativesInMeshAndAmpShaders ? hlsl::DXIL::ShaderFeatureInfo_DerivativesInMeshAndAmpShaders : 0;
  Flags |= m_bResourceDescriptorHeapIndexing ? hlsl::DXIL::ShaderFeatureInfo_ResourceDescriptorHeapIndexing : 0;
  Flags |= m_bSamplerDescriptorHeapIndexing ? hlsl::DXIL::ShaderFeatureInfo_SamplerDescriptorHeapIndexing : 0;
  Flags |= m_bAtomicInt64OnHeapResource ? hlsl::DXIL::ShaderFeatureInfo_AtomicInt64OnHeapResource : 0;

  Flags |= m_bAdvancedTextureOps ? hlsl::DXIL::ShaderFeatureInfo_AdvancedTextureOps : 0;
  Flags |= m_bWriteableMSAATextures ? hlsl::DXIL::ShaderFeatureInfo_WriteableMSAATextures : 0;

  return Flags;
}

uint64_t ShaderFlags::GetShaderFlagsRaw() const {
  union Cast {
    Cast(const ShaderFlags &flags) {
      shaderFlags = flags;
    }
    ShaderFlags shaderFlags;
    uint64_t  rawData;
  };
  static_assert(sizeof(uint64_t) == sizeof(ShaderFlags),
                "size must match to make sure no undefined bits when cast");
  Cast rawCast(*this);
  return rawCast.rawData;
}

void ShaderFlags::SetShaderFlagsRaw(uint64_t data) {
  union Cast {
    Cast(uint64_t data) {
      rawData = data;
    }
    ShaderFlags shaderFlags;
    uint64_t  rawData;
  };

  Cast rawCast(data);
  *this = rawCast.shaderFlags;
}

uint64_t ShaderFlags::GetShaderFlagsRawForCollection() {
  // This should be all the flags that can be set by DxilModule::CollectShaderFlags.
  ShaderFlags Flags;
  Flags.SetEnableDoublePrecision(true);
  Flags.SetInt64Ops(true);
  Flags.SetLowPrecisionPresent(true);
  Flags.SetEnableDoubleExtensions(true);
  Flags.SetWaveOps(true);
  Flags.SetTiledResources(true);
  Flags.SetEnableMSAD(true);
  Flags.SetUAVLoadAdditionalFormats(true);
  Flags.SetStencilRef(true);
  Flags.SetInnerCoverage(true);
  Flags.SetViewportAndRTArrayIndex(true);
  Flags.Set64UAVs(true);
  Flags.SetUAVsAtEveryStage(true);
  Flags.SetEnableRawAndStructuredBuffers(true);
  Flags.SetCSRawAndStructuredViaShader4X(true);
  Flags.SetViewID(true);
  Flags.SetBarycentrics(true);
  Flags.SetShadingRate(true);
  Flags.SetRaytracingTier1_1(true);
  Flags.SetSamplerFeedback(true);
  Flags.SetAtomicInt64OnTypedResource(true);
  Flags.SetAtomicInt64OnGroupShared(true);
  Flags.SetDerivativesInMeshAndAmpShaders(true);
  Flags.SetResourceDescriptorHeapIndexing(true);
  Flags.SetSamplerDescriptorHeapIndexing(true);
  Flags.SetAtomicInt64OnHeapResource(true);
  Flags.SetResMayNotAlias(true);
  Flags.SetAdvancedTextureOps(true);
  Flags.SetWriteableMSAATextures(true);
  return Flags.GetShaderFlagsRaw();
}

unsigned ShaderFlags::GetGlobalFlags() const {
  unsigned Flags = 0;
  Flags |= m_bDisableOptimizations ? DXIL::kDisableOptimizations : 0;
  Flags |= m_bDisableMathRefactoring ? DXIL::kDisableMathRefactoring : 0;
  Flags |= m_bEnableDoublePrecision ? DXIL::kEnableDoublePrecision : 0;
  Flags |= m_bForceEarlyDepthStencil ? DXIL::kForceEarlyDepthStencil : 0;
  Flags |= m_bEnableRawAndStructuredBuffers ? DXIL::kEnableRawAndStructuredBuffers : 0;
  Flags |= m_bLowPrecisionPresent && !m_bUseNativeLowPrecision? DXIL::kEnableMinPrecision : 0;
  Flags |= m_bEnableDoubleExtensions ? DXIL::kEnableDoubleExtensions : 0;
  Flags |= m_bEnableMSAD ? DXIL::kEnableMSAD : 0;
  Flags |= m_bAllResourcesBound ? DXIL::kAllResourcesBound : 0;
  return Flags;
}

// Given a CreateHandle call, returns arbitrary ConstantInt rangeID
// Note: HLSL is currently assuming that rangeID is a constant value, but this code is assuming
// that it can be either constant, phi node, or select instruction
static ConstantInt *GetArbitraryConstantRangeID(CallInst *handleCall) {
  Value *rangeID =
      handleCall->getArgOperand(DXIL::OperandIndex::kCreateHandleResIDOpIdx);
  ConstantInt *ConstantRangeID = dyn_cast<ConstantInt>(rangeID);
  while (ConstantRangeID == nullptr) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(rangeID)) {
      ConstantRangeID = CI;
    } else if (PHINode *PN = dyn_cast<PHINode>(rangeID)) {
      rangeID = PN->getIncomingValue(0);
    } else if (SelectInst *SI = dyn_cast<SelectInst>(rangeID)) {
      rangeID = SI->getTrueValue();
    } else {
      return nullptr;
    }
  }
  return ConstantRangeID;
}

// Given a handle type, find an arbitrary call instructions to create handle
static CallInst *FindCallToCreateHandle(Value *handleType) {
  Value *curVal = handleType;
  CallInst *CI = dyn_cast<CallInst>(handleType);
  while (CI == nullptr) {
    if (PHINode *PN = dyn_cast<PHINode>(curVal)) {
      curVal = PN->getIncomingValue(0);
    }
    else if (SelectInst *SI = dyn_cast<SelectInst>(curVal)) {
      curVal = SI->getTrueValue();
    }
    else {
      return nullptr;
    }
    CI = dyn_cast<CallInst>(curVal);
  }
  return CI;
}

DxilResourceProperties GetResourcePropertyFromHandleCall(const hlsl::DxilModule *M, CallInst *handleCall) {

  DxilResourceProperties RP;

  ConstantInt *HandleOpCodeConst = cast<ConstantInt>(
      handleCall->getArgOperand(DXIL::OperandIndex::kOpcodeIdx));
  DXIL::OpCode handleOp = static_cast<DXIL::OpCode>(HandleOpCodeConst->getLimitedValue());
  if (handleOp == DXIL::OpCode::CreateHandle) {
    if (ConstantInt *resClassArg =
      dyn_cast<ConstantInt>(handleCall->getArgOperand(
        DXIL::OperandIndex::kCreateHandleResClassOpIdx))) {
      DXIL::ResourceClass resClass = static_cast<DXIL::ResourceClass>(
        resClassArg->getLimitedValue());
      ConstantInt *rangeID = GetArbitraryConstantRangeID(handleCall);
      if (rangeID) {
        DxilResource resource;
        if (resClass == DXIL::ResourceClass::UAV)
          resource = M->GetUAV(rangeID->getLimitedValue());
        else if (resClass == DXIL::ResourceClass::SRV)
          resource = M->GetSRV(rangeID->getLimitedValue());
        RP = resource_helper::loadPropsFromResourceBase(&resource);
      }
    }
  }
  else if (handleOp == DXIL::OpCode::CreateHandleForLib) {
    // If library handle, find DxilResource by checking the name
    if (LoadInst *LI = dyn_cast<LoadInst>(handleCall->getArgOperand(
            DXIL::OperandIndex::kCreateHandleForLibResOpIdx))) {
      Value *resType = LI->getOperand(0);
      for (auto &&res : M->GetUAVs()) {
        if (res->GetGlobalSymbol() == resType) {
          RP = resource_helper::loadPropsFromResourceBase(res.get());
        }
      }
    }
  } else if (handleOp == DXIL::OpCode::AnnotateHandle) {
    DxilInst_AnnotateHandle annotateHandle(cast<Instruction>(handleCall));

    RP = resource_helper::loadPropsFromAnnotateHandle(annotateHandle, *M->GetShaderModel());
  }

  return RP;
}

struct ResourceKey {
  uint8_t Class;
  uint32_t Space;
  uint32_t LowerBound;
  uint32_t UpperBound;
};

struct ResKeyEq {
   bool operator()(const ResourceKey& k1, const ResourceKey& k2) const {
     return k1.Class == k2.Class && k1.Space == k2.Space &&
       k1.LowerBound == k2.LowerBound && k1.UpperBound == k2.UpperBound;
   }
};

struct ResKeyHash {
   std::size_t operator()(const ResourceKey& k) const {
     return std::hash<uint32_t>()(k.LowerBound) ^ (std::hash<uint32_t>()(k.UpperBound)<<1) ^
       (std::hash<uint32_t>()(k.Space)<<2) ^ (std::hash<uint8_t>()(k.Class)<<3);
   }
};

// Limited to retrieving handles created by CreateHandleFromBinding and CreateHandleForLib. returns null otherwise
// map should contain resources indexed by space, class, lower, and upper bounds
DxilResource *GetResourceFromAnnotateHandle(const hlsl::DxilModule *M, CallInst *handleCall,
                                     std::unordered_map<ResourceKey, DxilResource *, ResKeyHash, ResKeyEq> resMap) {
  DxilResource *resource = nullptr;

  ConstantInt *HandleOpCodeConst = cast<ConstantInt>(
      handleCall->getArgOperand(DXIL::OperandIndex::kOpcodeIdx));
  DXIL::OpCode handleOp = static_cast<DXIL::OpCode>(HandleOpCodeConst->getLimitedValue());
  if (handleOp == DXIL::OpCode::AnnotateHandle) {
    DxilInst_AnnotateHandle annotateHandle(cast<Instruction>(handleCall));
    CallInst *createCall = cast<CallInst>(annotateHandle.get_res());
    ConstantInt *HandleOpCodeConst = cast<ConstantInt>(
            createCall->getArgOperand(DXIL::OperandIndex::kOpcodeIdx));
    DXIL::OpCode handleOp = static_cast<DXIL::OpCode>(HandleOpCodeConst->getLimitedValue());
    if (handleOp == DXIL::OpCode::CreateHandleFromBinding) {
      DxilInst_CreateHandleFromBinding fromBind(createCall);
      DxilResourceBinding B = resource_helper::loadBindingFromConstant(*cast<Constant>(fromBind.get_bind()));
      ResourceKey key = {B.resourceClass, B.spaceID, B.rangeLowerBound, B.rangeUpperBound};
      resource = resMap[key];
    } else if (handleOp == DXIL::OpCode::CreateHandleForLib) {
      // If library handle, find DxilResource by checking the name
      if (LoadInst *LI = dyn_cast<LoadInst>(createCall->getArgOperand(
                                              DXIL::OperandIndex::kCreateHandleForLibResOpIdx))) {
        Value *resType = LI->getOperand(0);
        for (auto &&res : M->GetUAVs()) {
          if (res->GetGlobalSymbol() == resType) {
            return resource = res.get();
          }
        }
      }
    }
  }

  return resource;
}


ShaderFlags ShaderFlags::CollectShaderFlags(const Function *F,
                                           const hlsl::DxilModule *M) {
  ShaderFlags flag;
  // Module level options
  flag.SetUseNativeLowPrecision(!M->GetUseMinPrecision());
  flag.SetDisableOptimizations(M->GetDisableOptimization());
  flag.SetAllResourcesBound(M->GetAllResourcesBound());

  bool hasDouble = false;
  // ddiv dfma drcp d2i d2u i2d u2d.
  // fma has dxil op. Others should check IR instruction div/cast.
  bool hasDoubleExtension = false;
  bool has64Int = false;
  bool has16 = false;
  bool hasWaveOps = false;
  bool hasCheckAccessFully = false;
  bool hasMSAD = false;
  bool hasStencilRef = false;
  bool hasInnerCoverage = false;
  bool hasViewID = false;
  bool hasMulticomponentUAVLoads = false;
  bool hasViewportOrRTArrayIndex = false;
  bool hasShadingRate = false;
  bool hasBarycentrics = false;
  bool hasSamplerFeedback = false;
  bool hasRaytracingTier1_1 = false;
  bool hasAtomicInt64OnTypedResource = false;
  bool hasAtomicInt64OnGroupShared = false;
  bool hasDerivativesInMeshAndAmpShaders = false;
  bool hasResourceDescriptorHeapIndexing = false;
  bool hasSamplerDescriptorHeapIndexing = false;
  bool hasAtomicInt64OnHeapResource = false;

  // Used to determine whether to set ResMayNotAlias flag.
  bool hasUAVs = M->GetUAVs().size() > 0;

  bool hasAdvancedTextureOps = false;
  bool hasWriteableMSAATextures = false;

  // Try to maintain compatibility with a v1.0 validator if that's what we have.
  uint32_t valMajor, valMinor;
  M->GetValidatorVersion(valMajor, valMinor);
  bool hasMulticomponentUAVLoadsBackCompat = valMajor == 1 && valMinor == 0;
  bool hasViewportOrRTArrayIndexBackCombat = valMajor == 1 && valMinor < 4;
  bool hasBarycentricsBackCompat = valMajor == 1 && valMinor < 6;

  // Setting additional flag for downlevel shader model may cause some driver to
  // fail shader create due to an unrecognized flag.
  uint32_t dxilMajor, dxilMinor;
  M->GetDxilVersion(dxilMajor, dxilMinor);
  bool canSetResMayNotAlias = DXIL::CompareVersions(dxilMajor, dxilMinor, 1, 7) >= 0;

  Type *int16Ty = Type::getInt16Ty(F->getContext());
  Type *int64Ty = Type::getInt64Ty(F->getContext());


  // Set up resource to binding handle map for 64-bit atomics usage
  std::unordered_map<ResourceKey, DxilResource *, ResKeyHash, ResKeyEq> resMap;
  for (auto &res : M->GetUAVs()) {
    ResourceKey key = {(uint8_t)res->GetClass(), res->GetSpaceID(),
                       res->GetLowerBound(), res->GetUpperBound()};
    resMap.insert({key, res.get()});
    if (res->GetKind() == DXIL::ResourceKind::Texture2DMS ||
        res->GetKind() == DXIL::ResourceKind::Texture2DMSArray)
      hasWriteableMSAATextures = true;
  }

  for (const BasicBlock &BB : F->getBasicBlockList()) {
    for (const Instruction &I : BB.getInstList()) {
      // Skip none dxil function call.
      if (const CallInst *CI = dyn_cast<CallInst>(&I)) {
        if (!OP::IsDxilOpFunc(CI->getCalledFunction()))
          continue;
      }
      Type *Ty = I.getType();
      bool isDouble = Ty->isDoubleTy();
      bool isHalf = Ty->isHalfTy();
      bool isInt16 = Ty == int16Ty;
      bool isInt64 = Ty == int64Ty;
      if (isa<ExtractElementInst>(&I) ||
        isa<InsertElementInst>(&I))
        continue;
      for (Value *operand : I.operands()) {
        Type *Ty = operand->getType();
        isDouble |= Ty->isDoubleTy();
        isHalf |= Ty->isHalfTy();
        isInt16 |= Ty == int16Ty;
        isInt64 |= Ty == int64Ty;
      }
      if (isDouble) {
        hasDouble = true;
        switch (I.getOpcode()) {
        case Instruction::FDiv:
        case Instruction::UIToFP:
        case Instruction::SIToFP:
        case Instruction::FPToUI:
        case Instruction::FPToSI:
          hasDoubleExtension = true;
          break;
        }
      }
      if (isInt64) {
        has64Int = true;
        switch (I.getOpcode()) {
        case Instruction::AtomicCmpXchg:
        case Instruction::AtomicRMW:
          hasAtomicInt64OnGroupShared = true;
          break;
        }
      }

      has16 |= isHalf;
      has16 |= isInt16;
      if (const CallInst *CI = dyn_cast<CallInst>(&I)) {
        if (!OP::IsDxilOpFunc(CI->getCalledFunction()))
          continue;
        Value *opcodeArg = CI->getArgOperand(DXIL::OperandIndex::kOpcodeIdx);
        ConstantInt *opcodeConst = dyn_cast<ConstantInt>(opcodeArg);
        DXASSERT(opcodeConst, "DXIL opcode arg must be immediate");
        unsigned opcode = opcodeConst->getLimitedValue();
        DXASSERT(opcode < static_cast<unsigned>(DXIL::OpCode::NumOpCodes),
          "invalid DXIL opcode");
        DXIL::OpCode dxilOp = static_cast<DXIL::OpCode>(opcode);
        if (hlsl::OP::IsDxilOpWave(dxilOp))
          hasWaveOps = true;
        switch (dxilOp) {
        case DXIL::OpCode::CheckAccessFullyMapped:
          hasCheckAccessFully = true;
          break;
        case DXIL::OpCode::Msad:
          hasMSAD = true;
          break;
        case DXIL::OpCode::TextureLoad:
          if (!isa<Constant>(CI->getArgOperand(DXIL::OperandIndex::kTextureLoadOffset0OpIdx)) ||
              !isa<Constant>(CI->getArgOperand(DXIL::OperandIndex::kTextureLoadOffset1OpIdx)) ||
              !isa<Constant>(CI->getArgOperand(DXIL::OperandIndex::kTextureLoadOffset2OpIdx)))
            hasAdvancedTextureOps = true;
          __fallthrough;
        case DXIL::OpCode::BufferLoad: {
          if (hasMulticomponentUAVLoads) continue;
          // This is the old-style computation (overestimating requirements).
          Value *resHandle = CI->getArgOperand(DXIL::OperandIndex::kBufferLoadHandleOpIdx);
          CallInst *handleCall = FindCallToCreateHandle(resHandle);
          // Check if this is a library handle or general create handle
          if (handleCall) {
            DxilResourceProperties RP = GetResourcePropertyFromHandleCall(M, handleCall);
            if (RP.isUAV()) {
              // Validator 1.0 assumes that all uav load is multi component load.
              if (hasMulticomponentUAVLoadsBackCompat) {
                hasMulticomponentUAVLoads = true;
                continue;
              } else {
                if (DXIL::IsTyped(RP.getResourceKind()) &&
                    RP.Typed.CompCount > 1)
                  hasMulticomponentUAVLoads = true;
              }
            }
          }
       } break;
        case DXIL::OpCode::Fma:
          hasDoubleExtension |= isDouble;
          break;
        case DXIL::OpCode::InnerCoverage:
          hasInnerCoverage = true;
          break;
        case DXIL::OpCode::ViewID:
          hasViewID = true;
          break;
        case DXIL::OpCode::AllocateRayQuery:
        case DXIL::OpCode::GeometryIndex:
          hasRaytracingTier1_1 = true;
          break;
        case DXIL::OpCode::AttributeAtVertex:
          hasBarycentrics = true;
        break;
        case DXIL::OpCode::AtomicBinOp:
        case DXIL::OpCode::AtomicCompareExchange:
          if (isInt64) {
            Value *resHandle = CI->getArgOperand(DXIL::OperandIndex::kAtomicBinOpHandleOpIdx);
            CallInst *handleCall = FindCallToCreateHandle(resHandle);
            DxilResourceProperties RP = GetResourcePropertyFromHandleCall(M, handleCall);
            if (DXIL::IsTyped(RP.getResourceKind()))
                hasAtomicInt64OnTypedResource = true;
            // set uses 64-bit flag if relevant
            if (DxilResource *res = GetResourceFromAnnotateHandle(M, handleCall, resMap)) {
              res->SetHasAtomic64Use(true);
            } else {
              // Assuming CreateHandleFromHeap, which indicates a descriptor
              hasAtomicInt64OnHeapResource = true;
            }
          }
          break;
        case DXIL::OpCode::SampleGrad:
        case DXIL::OpCode::SampleLevel:
        case DXIL::OpCode::SampleCmpLevelZero:
          if (!isa<Constant>(CI->getArgOperand(DXIL::OperandIndex::kTextureSampleOffset0OpIdx)) ||
              !isa<Constant>(CI->getArgOperand(DXIL::OperandIndex::kTextureSampleOffset1OpIdx)) ||
              !isa<Constant>(CI->getArgOperand(DXIL::OperandIndex::kTextureSampleOffset2OpIdx)))
            hasAdvancedTextureOps = true;
          break;
        case DXIL::OpCode::Sample:
        case DXIL::OpCode::SampleBias:
        case DXIL::OpCode::SampleCmp:
          if (!isa<Constant>(CI->getArgOperand(DXIL::OperandIndex::kTextureSampleOffset0OpIdx)) ||
              !isa<Constant>(CI->getArgOperand(DXIL::OperandIndex::kTextureSampleOffset1OpIdx)) ||
              !isa<Constant>(CI->getArgOperand(DXIL::OperandIndex::kTextureSampleOffset2OpIdx)))
            hasAdvancedTextureOps = true;
          __fallthrough;
        case DXIL::OpCode::DerivFineX:
        case DXIL::OpCode::DerivFineY:
        case DXIL::OpCode::DerivCoarseX:
        case DXIL::OpCode::DerivCoarseY:
        case DXIL::OpCode::CalculateLOD: {
          const ShaderModel *pSM = M->GetShaderModel();
          if (pSM->IsAS() || pSM->IsMS())
            hasDerivativesInMeshAndAmpShaders = true;
        } break;
        case DXIL::OpCode::CreateHandleFromHeap: {
          ConstantInt *isSamplerVal = dyn_cast<ConstantInt>(
                        CI->getArgOperand(DXIL::OperandIndex::kCreateHandleFromHeapSamplerHeapOpIdx));
          if (isSamplerVal->getLimitedValue()) {
            hasSamplerDescriptorHeapIndexing = true;
          } else {
            hasResourceDescriptorHeapIndexing = true;
            if (!hasUAVs) {
              // If not already marked, check if UAV.
              DxilResourceProperties RP = GetResourcePropertyFromHandleCall(
                  M, const_cast<CallInst *>(CI));
              if (RP.isUAV())
                hasUAVs = true;
            }
          }
        } break;
        case DXIL::OpCode::TextureStoreSample:
          hasWriteableMSAATextures = true;
          __fallthrough;
        case DXIL::OpCode::SampleCmpLevel:
        case DXIL::OpCode::TextureGatherRaw:
          hasAdvancedTextureOps = true;
          break;
        default:
          // Normal opcodes.
          break;
        }
      }
    }
  }

  // If this function is a shader, add flags based on signatures
  if (M->HasDxilEntryProps(F)) {
    const DxilEntryProps &entryProps = M->GetDxilEntryProps(F);

    // Val ver < 1.4 has a bug where input case was always clobbered by the
    // output check.  The only case where it made a difference such that an
    // incorrect flag would be set was for the HS and DS input cases.
    // It was also checking PS input and output, but PS output could not have
    // the semantic, and since it was clobbering the result, it would always
    // clear it.  Since this flag should not be set for PS at all,
    // it produced the correct result for PS by accident.
    bool checkInputRTArrayIndex = entryProps.props.IsGS();
    if (!hasViewportOrRTArrayIndexBackCombat)
      checkInputRTArrayIndex |= entryProps.props.IsDS() ||
                                entryProps.props.IsHS();
    bool checkOutputRTArrayIndex =
      entryProps.props.IsVS() || entryProps.props.IsDS() ||
      entryProps.props.IsHS();

    for (auto &&E : entryProps.sig.InputSignature.GetElements()) {
      switch (E->GetKind()) {
      case Semantic::Kind::ViewPortArrayIndex:
      case Semantic::Kind::RenderTargetArrayIndex:
        if (checkInputRTArrayIndex)
          hasViewportOrRTArrayIndex = true;
        break;
      case Semantic::Kind::ShadingRate:
        hasShadingRate = true;
        break;
      case Semantic::Kind::Barycentrics:
        hasBarycentrics = true;
        break;
      default:
        break;
      }
    }

    for (auto &&E : entryProps.sig.OutputSignature.GetElements()) {
      switch (E->GetKind()) {
      case Semantic::Kind::ViewPortArrayIndex:
      case Semantic::Kind::RenderTargetArrayIndex:
        if (checkOutputRTArrayIndex)
          hasViewportOrRTArrayIndex = true;
        break;
      case Semantic::Kind::StencilRef:
        if (entryProps.props.IsPS())
          hasStencilRef = true;
        break;
      case Semantic::Kind::InnerCoverage:
        if (entryProps.props.IsPS())
          hasInnerCoverage = true;
        break;
      case Semantic::Kind::ShadingRate:
        hasShadingRate = true;
        break;
      default:
        break;
      }
    }
  }

  if (!hasRaytracingTier1_1) {
    if (const DxilSubobjects *pSubobjects = M->GetSubobjects()) {
      for (const auto &it : pSubobjects->GetSubobjects()) {
        switch (it.second->GetKind()) {
        case DXIL::SubobjectKind::RaytracingPipelineConfig1:
          hasRaytracingTier1_1 = true;
          break;
        case DXIL::SubobjectKind::StateObjectConfig: {
          uint32_t Flags;
          if (it.second->GetStateObjectConfig(Flags) &&
              ((Flags & ~(unsigned)DXIL::StateObjectFlags::ValidMask_1_4) != 0))
            hasRaytracingTier1_1 = true;
        } break;
        default:
          break;
        }
        if (hasRaytracingTier1_1)
          break;
      }
    }
  }

  flag.SetEnableDoublePrecision(hasDouble);
  flag.SetStencilRef(hasStencilRef);
  flag.SetInnerCoverage(hasInnerCoverage);
  flag.SetInt64Ops(has64Int);
  flag.SetLowPrecisionPresent(has16);
  flag.SetEnableDoubleExtensions(hasDoubleExtension);
  flag.SetWaveOps(hasWaveOps);
  flag.SetTiledResources(hasCheckAccessFully);
  flag.SetEnableMSAD(hasMSAD);
  flag.SetUAVLoadAdditionalFormats(hasMulticomponentUAVLoads);
  flag.SetViewID(hasViewID);
  flag.SetViewportAndRTArrayIndex(hasViewportOrRTArrayIndex);
  flag.SetShadingRate(hasShadingRate);
  flag.SetBarycentrics(hasBarycentricsBackCompat ? false : hasBarycentrics);
  flag.SetSamplerFeedback(hasSamplerFeedback);
  flag.SetRaytracingTier1_1(hasRaytracingTier1_1);
  flag.SetAtomicInt64OnTypedResource(hasAtomicInt64OnTypedResource);
  flag.SetAtomicInt64OnGroupShared(hasAtomicInt64OnGroupShared);
  flag.SetDerivativesInMeshAndAmpShaders(hasDerivativesInMeshAndAmpShaders);
  flag.SetResourceDescriptorHeapIndexing(hasResourceDescriptorHeapIndexing);
  flag.SetSamplerDescriptorHeapIndexing(hasSamplerDescriptorHeapIndexing);
  flag.SetAtomicInt64OnHeapResource(hasAtomicInt64OnHeapResource);
  flag.SetAdvancedTextureOps(hasAdvancedTextureOps);
  flag.SetWriteableMSAATextures(hasWriteableMSAATextures);

  // Only bother setting the flag when there are UAVs.
  flag.SetResMayNotAlias(canSetResMayNotAlias && hasUAVs &&
                         !M->GetResMayAlias());

  return flag;
}

void ShaderFlags::CombineShaderFlags(const ShaderFlags &other) {
  SetShaderFlagsRaw(GetShaderFlagsRaw() | other.GetShaderFlagsRaw());
}
