///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilPatchShaderRecordBindings.cpp                                        //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides a pass used by the RayTracing Fallback Lyaer to add modify       //
// bindings to pull local root signature parameters from a global            //
// "shader table" buffer instead                                             //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/DxilGenerationPass.h"
#include "dxc/HLSL/DxilFallbackLayerPass.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilSignatureElement.h"
#include "dxc/DXIL/DxilFunctionProps.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/Support/Global.h"

#include "dxc/Support/Unicode.h"
#include "dxc/DXIL/DxilTypeSystem.h"
#include "dxc/DXIL/DxilConstants.h"
#include "dxc/DXIL/DxilInstructions.h"
#include "dxc/HLSL/DxilSpanAllocator.h"
#include "dxc/DxilRootSignature/DxilRootSignature.h"
#include "dxc/DXIL/DxilUtil.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Scalar.h"
#include <memory>
#include <unordered_set>
#include <functional>
#include <unordered_map>
#include <array>

struct D3D12_VERSIONED_ROOT_SIGNATURE_DESC;
#include "DxilPatchShaderRecordBindingsShared.h"


using namespace llvm;
using namespace hlsl;


bool operator==(const ViewKey &a, const ViewKey &b) {
  return memcmp(&a, &b, sizeof(a)) == 0;
}

const size_t SizeofD3D12GpuVA = sizeof(uint64_t);
const size_t SizeofD3D12GpuDescriptorHandle = sizeof(uint64_t);

Function *CloneFunction(Function *Orig,
    const llvm::Twine &Name,
    llvm::Module *llvmModule) {

    Function *F = Function::Create(Orig->getFunctionType(),
        GlobalValue::LinkageTypes::ExternalLinkage,
        Name, llvmModule);

    SmallVector<ReturnInst *, 2> Returns;
    ValueToValueMapTy vmap;
    // Map params.
    auto entryParamIt = F->arg_begin();
    for (Argument &param : Orig->args()) {
        vmap[&param] = (entryParamIt++);
    }

    DxilModule &DM = llvmModule->GetOrCreateDxilModule();

    llvm::CloneFunctionInto(F, Orig, vmap, /*ModuleLevelChagnes*/ false, Returns);
    DM.GetTypeSystem().CopyFunctionAnnotation(F, Orig, DM.GetTypeSystem());

    if (DM.HasDxilFunctionProps(F)) {
        DM.CloneDxilEntryProps(Orig, F);
    }
    return F;
}


struct ShaderRecordEntry {
  DxilRootParameterType ParameterType;
  unsigned int RecordOffsetInBytes;
  unsigned int OffsetInDescriptors; // Only valid for descriptor tables

  static ShaderRecordEntry InvalidEntry() { return { (DxilRootParameterType)-1, (unsigned int)-1, 0 }; }
  bool IsInvalid() { return (unsigned int)ParameterType == (unsigned int)-1; }
};

struct D3D12_VERSIONED_ROOT_SIGNATURE_DESC;
class DxilPatchShaderRecordBindings : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilPatchShaderRecordBindings() : ModulePass(ID) {}
  StringRef getPassName() const override { return "DXIL Patch Shader Record Binding"; }
  void applyOptions(PassOptions O) override;
  bool runOnModule(Module &M) override;

private:
  void ValidateParameters();
  void AddInputBinding(Module &M);
  void PatchShaderBindings(Module &M);
  void InitializeViewTable();

  unsigned int AddSRVRawBuffer(Module &M, unsigned int registerIndex, unsigned int registerSpace, const std::string &bufferName);
  unsigned int AddHandle(Module &M, unsigned int baseRegisterIndex, unsigned int rangeSize, unsigned int registerSpace, DXIL::ResourceClass resClass, DXIL::ResourceKind resKind, const std::string &bufferName, llvm::Type *type = nullptr, unsigned int constantBufferSize = 0);
  unsigned int AddAliasedHandle(Module &M, unsigned int baseRegisterIndex, unsigned int registerSpace, DXIL::ResourceClass resClass, DXIL::ResourceKind resKind, const std::string &bufferName, llvm::Type *type);
  unsigned int AddCBufferAliasedHandle(Module &M, unsigned int baseRegisterIndex, unsigned int registerSpace, const std::string &bufferName);

  llvm::Value *CreateOffsetToShaderRecord(Module &M, IRBuilder<> &Builder, unsigned int RecordOffsetInBytes, llvm::Value *CbufferOffsetInBytes);
  llvm::Value *CreateShaderRecordBufferLoad(Module &M, IRBuilder<> &Builder, llvm::Value *ShaderRecordOffsetInBytes, llvm::Type* type);
  llvm::Value *CreateCBufferLoadOffsetInBytes(Module &M, IRBuilder<> &Builder, llvm::Instruction *instruction);
  llvm::Value *CreateCBufferLoadLegacy(Module &M, IRBuilder<> &Builder, llvm::Value *ResourceHandle, unsigned int RowToLoad = 0);

  llvm::Value *LoadShaderRecordData(Module &M, IRBuilder<> &Builder,
                                    llvm::Value *offsetToShaderRecord,
                                    unsigned int dataOffsetInShaderRecord);

  void PatchCreateHandleToUseDescriptorIndex(
      _In_ Module &M,
      _In_ IRBuilder<> &Builder,
      _In_ DXIL::ResourceKind &resourceKind,
      _In_ DXIL::ResourceClass &resourceClass,
      _In_ llvm::Type *resourceType,
      _In_ llvm::Value *descriptorIndex,
      _Inout_ DxilInst_CreateHandleForLib &createHandleInstr);


  bool GetHandleInfo(
    Module &M, 
    DxilInst_CreateHandleForLib &createHandleStructForLib, 
    _Out_ unsigned int &shaderRegister, 
    _Out_ unsigned int &registerSpace, 
    _Out_ DXIL::ResourceKind &kind, 
    _Out_ DXIL::ResourceClass &resClass,
    _Out_ llvm::Type *&resType);

  llvm::Value * GetAliasedDescriptorHeapHandle(Module &M, llvm::Type *, DXIL::ResourceClass resClass, DXIL::ResourceKind resKind);

  unsigned int GetConstantBufferOffsetToShaderRecord();

  bool IsCBufferLoad(llvm::Instruction *instruction);

  // Unlike the LLVM version of this function, this does not requires the InstructionToReplace and the ValueToReplaceWith to be the same instruction type
  static void ReplaceUsesOfWith(llvm::Instruction *InstructionToReplace, llvm::Value *ValueToReplaceWith);

  static ShaderRecordEntry FindRootSignatureDescriptor(const DxilVersionedRootSignatureDesc &rootSignatureDescriptor, unsigned int ShaderRecordIdentifierSizeInBytes, DXIL::ResourceClass resourceClass, unsigned int baseRegisterIndex, unsigned int registerSpace);

  // TODO: I would like to see these prefixed with m_
  llvm::Value *ShaderTableHandle = nullptr;
  llvm::Value *DispatchRaysConstantsHandle = nullptr;
  llvm::Value *BaseShaderRecordOffset = nullptr;

  static const unsigned int NumViewTypes = 4;
  struct ViewKeyHasher
  {
  public:
      std::size_t operator()(const ViewKey &x) const {
        return std::hash<unsigned int>()((unsigned int)x.ViewType) ^ 
            std::hash<unsigned int>()((unsigned int)x.StructuredStride);
      }
  };


  std::unordered_map<ViewKey, llvm::Value *, ViewKeyHasher>
      TypeToAliasedDescriptorHeap[NumViewTypes];

  llvm::Function *EntryPointFunction;

  ShaderInfo *pInputShaderInfo;
  const DxilVersionedRootSignatureDesc *pRootSignatureDesc;
  DXIL::ShaderKind ShaderKind;
};

char DxilPatchShaderRecordBindings::ID = 0;

// TODO: Find the right thing to do on failure
void ThrowFailure() {
  throw std::exception();
}

// TODO: Stolen from Brandon's code, merge
// Remove ELF mangling
static inline std::string GetUnmangledName(StringRef name) {
  if (!name.startswith("\x1?"))
      return name;

  size_t pos = name.find("@@");
  if (pos == name.npos)
    return name;


  return name.substr(2, pos - 2);
}

static Function* getFunctionFromName(Module &M, const std::wstring& exportName) {
  for (auto F = M.begin(), E = M.end(); F != E; ++F) {
    std::wstring functionName = Unicode::UTF8ToWideStringOrThrow(GetUnmangledName(F->getName()).c_str());
    if (exportName == functionName) {
      return F;
    }
  }
  return nullptr;
}

ModulePass *llvm::createDxilPatchShaderRecordBindingsPass() {
  return new DxilPatchShaderRecordBindings();
}

INITIALIZE_PASS(DxilPatchShaderRecordBindings, "hlsl-dxil-patch-shader-record-bindings", "Patch shader record bindings to instead pull from the fallback provided bindings", false, false)

void DxilPatchShaderRecordBindings::applyOptions(PassOptions O) {
  for (const auto & option : O) {
    if (0 == option.first.compare("root-signature")) {
      unsigned int cHexRadix = 16;
      pInputShaderInfo = (ShaderInfo*)strtoull(option.second.data(), nullptr, cHexRadix);
      pRootSignatureDesc = (const DxilVersionedRootSignatureDesc*)pInputShaderInfo->pRootSignatureDesc;
    }
  }
}

void AddAnnoationsIfNeeded(DxilModule &DM, llvm::StructType *StructTy, const std::string &FieldName, unsigned int numFields = 1)
{
    auto pAnnotation = DM.GetTypeSystem().GetStructAnnotation(StructTy);
    if (pAnnotation == nullptr)
    {
        pAnnotation = DM.GetTypeSystem().AddStructAnnotation(StructTy);
        pAnnotation->SetCBufferSize(sizeof(uint32_t) * numFields);
        for (unsigned int i = 0; i < numFields; i++)
        {
            pAnnotation->GetFieldAnnotation(i).SetCBufferOffset(sizeof(uint32_t) * i);
            pAnnotation->GetFieldAnnotation(i).SetCompType(hlsl::DXIL::ComponentType::I32);
            pAnnotation->GetFieldAnnotation(i).SetFieldName(FieldName + std::to_string(i));
        }
    }
}

unsigned int DxilPatchShaderRecordBindings::AddHandle(Module &M, unsigned int baseRegisterIndex, unsigned int rangeSize, unsigned int registerSpace, DXIL::ResourceClass resClass, DXIL::ResourceKind resKind, const std::string &bufferName, llvm::Type *type, unsigned int constantBufferSize) {
  LLVMContext & Ctx = M.getContext();
  DxilModule &DM = M.GetOrCreateDxilModule();

  // Set up a SRV with byte address buffer
  unsigned int resourceHandle;
  std::unique_ptr<DxilResource> pHandle;
  std::unique_ptr<DxilCBuffer> pCBuf;
  std::unique_ptr<DxilSampler> pSampler;
  DxilResourceBase *pBaseHandle;
  switch (resClass) {
  case DXIL::ResourceClass::SRV:
    resourceHandle = static_cast<unsigned int>(DM.GetSRVs().size());
    pHandle = llvm::make_unique<DxilResource>();
    pHandle->SetRW(false);
    pBaseHandle = pHandle.get();
    break;
  case DXIL::ResourceClass::UAV:
    resourceHandle = static_cast<unsigned int>(DM.GetUAVs().size());
    pHandle = llvm::make_unique<DxilResource>();
    pHandle->SetRW(true);
    pBaseHandle = pHandle.get();
    break;
  case DXIL::ResourceClass::CBuffer:
    resourceHandle = static_cast<unsigned int>(DM.GetCBuffers().size());
    pCBuf = llvm::make_unique<DxilCBuffer>();
    pCBuf->SetSize(constantBufferSize);
    pBaseHandle = pCBuf.get();
    break;
  case DXIL::ResourceClass::Sampler:
    resourceHandle = static_cast<unsigned int>(DM.GetSamplers().size());
    pSampler = llvm::make_unique<DxilSampler>();
    // TODO: Is this okay? What if one of the samplers in the table is a comparison sampler?
    pSampler->SetSamplerKind(DxilSampler::SamplerKind::Default);
    pBaseHandle = pSampler.get();
    break;
  }

  if (!type) {
    SmallVector<llvm::Type*, 1> Elements{ Type::getInt32Ty(Ctx) };
    std::string ByteAddressBufferName = "struct.ByteAddressBuffer";
    type = M.getTypeByName(ByteAddressBufferName);
    if (!type)
    {
        StructType *StructTy;
        type = StructTy = StructType::create(Elements, ByteAddressBufferName);
  
        AddAnnoationsIfNeeded(DM, StructTy, ByteAddressBufferName);
    }
  }

  GlobalVariable *GV = M.getGlobalVariable(bufferName);
  if (!GV) {
    GV = cast<GlobalVariable>(M.getOrInsertGlobal(bufferName, type));
  }

  pBaseHandle->SetGlobalName(bufferName.c_str());
  pBaseHandle->SetGlobalSymbol(GV);
  pBaseHandle->SetID(resourceHandle);
  pBaseHandle->SetSpaceID(registerSpace);
  pBaseHandle->SetLowerBound(baseRegisterIndex);
  pBaseHandle->SetRangeSize(rangeSize);
  pBaseHandle->SetKind(resKind);

  if (pHandle) {
    pHandle->SetGloballyCoherent(false);
    pHandle->SetHasCounter(false);
    pHandle->SetCompType(CompType::getF32()); // TODO: Need to handle all types
  }

  unsigned int ID;
  switch (resClass) {
  case DXIL::ResourceClass::SRV:
    ID = DM.AddSRV(std::move(pHandle));
    break;
  case DXIL::ResourceClass::UAV:
    ID = DM.AddUAV(std::move(pHandle));
    break;
  case DXIL::ResourceClass::CBuffer:
    ID = DM.AddCBuffer(std::move(pCBuf));
    break;
  case DXIL::ResourceClass::Sampler:
    ID = DM.AddSampler(std::move(pSampler));
    break;
  }

  assert(ID == resourceHandle);
  return ID;
}

unsigned int DxilPatchShaderRecordBindings::GetConstantBufferOffsetToShaderRecord()
{
    switch (ShaderKind)
    {
    case DXIL::ShaderKind::ClosestHit:
    case DXIL::ShaderKind::AnyHit:
    case DXIL::ShaderKind::Intersection:
        return offsetof(DispatchRaysConstants, HitGroupShaderRecordStride);
    case DXIL::ShaderKind::Miss:
        return offsetof(DispatchRaysConstants, MissShaderRecordStride);
    default:
        ThrowFailure();
        return -1;
    }
}


unsigned int DxilPatchShaderRecordBindings::AddSRVRawBuffer(Module &M, unsigned int registerIndex, unsigned int registerSpace, const std::string &bufferName) {
  return AddHandle(M, registerIndex, 1, registerSpace, DXIL::ResourceClass::SRV, DXIL::ResourceKind::RawBuffer, bufferName);
}

llvm::Constant *GetArraySymbol(Module &M, const std::string &bufferName) {
  LLVMContext & Ctx = M.getContext();

  SmallVector<llvm::Type*, 1> Elements{ Type::getInt32Ty(Ctx) };
  llvm::StructType *StructTy = llvm::StructType::create(Elements, bufferName);
  llvm::ArrayType *ArrayTy = ArrayType::get(StructTy, -1);

  return UndefValue::get(ArrayTy->getPointerTo());
}

unsigned int DxilPatchShaderRecordBindings::AddCBufferAliasedHandle(Module &M, unsigned int baseRegisterIndex, unsigned int registerSpace, const std::string &bufferName) {
  const unsigned int maxConstantBufferSize = 4096 * 16;
  return AddHandle(M, baseRegisterIndex, UINT_MAX, registerSpace, DXIL::ResourceClass::CBuffer, DXIL::ResourceKind::CBuffer, bufferName, GetArraySymbol(M, bufferName)->getType(), maxConstantBufferSize);
}

unsigned int DxilPatchShaderRecordBindings::AddAliasedHandle(Module &M, unsigned int baseRegisterIndex, unsigned int registerSpace, DXIL::ResourceClass resClass, DXIL::ResourceKind resKind, const std::string &bufferName, llvm::Type *type) {
  return AddHandle(M, baseRegisterIndex, UINT_MAX, registerSpace, resClass, resKind, bufferName, type);
}

// TODO: Stolen from Brandon's code
DXIL::ShaderKind GetRayShaderKindCopy(Function* F)
{
    if (F->hasFnAttribute("exp-shader"))
        return DXIL::ShaderKind::RayGeneration;

    DxilModule& DM = F->getParent()->GetDxilModule();
    if (DM.HasDxilFunctionProps(F) && DM.GetDxilFunctionProps(F).IsRay())
        return DM.GetDxilFunctionProps(F).shaderKind;

    return DXIL::ShaderKind::Invalid;
}

bool DxilPatchShaderRecordBindings::runOnModule(Module &M) {
  DxilModule &DM = M.GetOrCreateDxilModule();
  EntryPointFunction = pInputShaderInfo->ExportName ? getFunctionFromName(M, pInputShaderInfo->ExportName) : DM.GetEntryFunction();
  ShaderKind = GetRayShaderKindCopy(EntryPointFunction);

  ValidateParameters();
  InitializeViewTable();

  PatchShaderBindings(M);
  DM.ReEmitDxilResources();
  return true;
}

void DxilPatchShaderRecordBindings::ValidateParameters() {
  if (!pInputShaderInfo || !pInputShaderInfo->pRootSignatureDesc) {
    throw std::exception();
  }
}

DxilResourceBase &GetResourceFromID(DxilModule &DM, DXIL::ResourceClass resClass, unsigned int id)
{
    switch (resClass)
    {
    case DXIL::ResourceClass::CBuffer:
        return DM.GetCBuffer(id);
        break;
    case DXIL::ResourceClass::SRV:
        return DM.GetSRV(id);
        break;
    case DXIL::ResourceClass::UAV:
        return DM.GetUAV(id);
        break;
    case DXIL::ResourceClass::Sampler:
        return DM.GetSampler(id);
        break;
    default:
        ThrowFailure();
        llvm_unreachable("invalid resource class");
    }
}

unsigned int FindOrInsertViewIntoList(const ViewKey &key, ViewKey *pViewList, unsigned int &numViews, unsigned int maxViews)
{
    unsigned int viewIndex = 0;
    for (; viewIndex < numViews; viewIndex++)
    {
        if (pViewList[viewIndex] == key)
        {
            break;
        }
    }

    if (viewIndex == numViews)
    {
        if (viewIndex >= maxViews) {
            ThrowFailure();
        }

        pViewList[viewIndex] = key;
        numViews++;
    }
    return viewIndex;
}

llvm::Value *DxilPatchShaderRecordBindings::GetAliasedDescriptorHeapHandle(Module &M, llvm::Type *type, DXIL::ResourceClass resClass, DXIL::ResourceKind resKind)
{
    DxilModule &DM = M.GetOrCreateDxilModule();
    unsigned int resClassIndex = (unsigned int)resClass;
    
    ViewKey key = {};
    key.ViewType = (unsigned int)resKind;
    if (DXIL::IsStructuredBuffer(resKind))
    {
      key.StructuredStride = type->getPrimitiveSizeInBits();
    } else if (resKind != DXIL::ResourceKind::RawBuffer)
    {
      auto containedType = type->getContainedType(0);
      // If it's a vector, get the type of just a single element
      if (containedType->getNumContainedTypes() > 0)
      {
        assert(containedType->getNumContainedTypes() <= 4);
        containedType = containedType->getContainedType(0);
      }
      key.SRVComponentType = (unsigned int)CompType::GetCompType(containedType).GetKind();
    }
    auto aliasedDescriptorHeapHandle = TypeToAliasedDescriptorHeap[resClassIndex].find(key);
    if (aliasedDescriptorHeapHandle == TypeToAliasedDescriptorHeap[resClassIndex].end())
    {
        unsigned int registerSpaceOffset = 0;
        std::string HandleName;

        if (resClass == DXIL::ResourceClass::SRV)
        {
          registerSpaceOffset = FindOrInsertViewIntoList(
              key, 
              pInputShaderInfo->pSRVRegisterSpaceArray, 
              *pInputShaderInfo->pNumSRVSpaces, 
              FallbackLayerNumDescriptorHeapSpacesPerView);

          HandleName = std::string("SRVDescriptorHeapTable") +
                       std::to_string(registerSpaceOffset);
        }
        else if (resClass == DXIL::ResourceClass::UAV)
        {
          registerSpaceOffset = FindOrInsertViewIntoList(
              key,
              pInputShaderInfo->pUAVRegisterSpaceArray,
              *pInputShaderInfo->pNumUAVSpaces,
              FallbackLayerNumDescriptorHeapSpacesPerView);

          if (registerSpaceOffset == 0)
          {
              // Using the descriptor heap declared by the fallback for handling emulated pointers,
              // make sure the name is an exact match
              assert(key.ViewType == (unsigned int)hlsl::DXIL::ResourceKind::RawBuffer);
              HandleName = "\01?DescriptorHeapBufferTable@@3PAURWByteAddressBuffer@@A";
          }
          else
          {
              HandleName = std::string("UAVDescriptorHeapTable") +
                  std::to_string(registerSpaceOffset);
          }
        }
        else if (resClass == DXIL::ResourceClass::CBuffer)
        {
          HandleName = std::string("CBVDescriptorHeapTable");

        } else {
          HandleName = std::string("SamplerDescriptorHeapTable");
        }


        llvm::ArrayType *descriptorHeapType = ArrayType::get(type, 0);
        unsigned int id = AddAliasedHandle(M, FallbackLayerDescriptorHeapTable, FallbackLayerRegisterSpace + FallbackLayerDescriptorHeapSpaceOffset + registerSpaceOffset, resClass, resKind, HandleName, descriptorHeapType);
        
        TypeToAliasedDescriptorHeap[resClassIndex][key] = GetResourceFromID(DM, resClass, id).GetGlobalSymbol();
    }
    return TypeToAliasedDescriptorHeap[resClassIndex][key];
}

void DxilPatchShaderRecordBindings::AddInputBinding(Module &M) {
  DxilModule &DM = M.GetOrCreateDxilModule();
  auto & EntryBlock = EntryPointFunction->getEntryBlock();
  auto & Instructions = EntryBlock.getInstList();

  std::string bufferName;
  unsigned int bufferRegister;

  switch (ShaderKind) {
  case DXIL::ShaderKind::AnyHit:
  case DXIL::ShaderKind::ClosestHit:
  case DXIL::ShaderKind::Intersection:
    bufferRegister = FallbackLayerHitGroupRecordByteAddressBufferRegister;
    bufferName = "\01?HitGroupShaderTable@@3UByteAddressBuffer@@A";
    break;
  case DXIL::ShaderKind::Miss:
    bufferRegister = FallbackLayerMissShaderRecordByteAddressBufferRegister;
    bufferName = "\01?MissShaderTable@@3UByteAddressBuffer@@A";
    break;
  case DXIL::ShaderKind::RayGeneration:
    bufferRegister = FallbackLayerRayGenShaderRecordByteAddressBufferRegister;
    bufferName = "\01?RayGenShaderTable@@3UByteAddressBuffer@@A";
    break;
  case DXIL::ShaderKind::Callable:
    bufferRegister = FallbackLayerCallableShaderRecordByteAddressBufferRegister;
    bufferName = "\01?CallableShaderTable@@3UByteAddressBuffer@@A";
    break;
  }
  unsigned int ShaderRecordID = AddSRVRawBuffer(M, bufferRegister, FallbackLayerRegisterSpace, bufferName);

  auto It = Instructions.begin();
  OP *HlslOP = DM.GetOP();
  LLVMContext & Ctx = M.getContext();

  IRBuilder<> Builder(It);
  {
    auto ShaderTableName = "ShaderTableHandle";
    llvm::Value *Symbol = DM.GetSRV(ShaderRecordID).GetGlobalSymbol();
    llvm::Value *Load = Builder.CreateLoad(Symbol, "LoadShaderTableHandle");

    Function *CreateHandleForLib = HlslOP->GetOpFunc(DXIL::OpCode::CreateHandleForLib, Load->getType());
    Constant *CreateHandleOpcodeArg = HlslOP->GetU32Const((unsigned)DXIL::OpCode::CreateHandleForLib);
    ShaderTableHandle = Builder.CreateCall(CreateHandleForLib, { CreateHandleOpcodeArg, Load }, ShaderTableName);
  }

  {
    auto CbufferName = "Constants";
    const unsigned int sizeOfConstantsInBytes = sizeof(DispatchRaysConstants);
    llvm::StructType *StructTy= M.getTypeByName(CbufferName);
    if (!StructTy)
    {
        const unsigned int numUintsInConstants = sizeOfConstantsInBytes / sizeof(unsigned int);
        SmallVector<llvm::Type*, numUintsInConstants> Elements(numUintsInConstants);
        for (unsigned int i = 0; i < numUintsInConstants; i++)
        {
            Elements[i] = Type::getInt32Ty(Ctx);
        }
        StructTy = llvm::StructType::create(Elements, CbufferName);
        AddAnnoationsIfNeeded(DM, StructTy, std::string(CbufferName), numUintsInConstants);
    }

    unsigned int handle = AddHandle(M, FallbackLayerDispatchConstantsRegister, 1, FallbackLayerRegisterSpace, DXIL::ResourceClass::CBuffer, DXIL::ResourceKind::CBuffer, CbufferName, StructTy, sizeOfConstantsInBytes);

    llvm::Value *Symbol = DM.GetCBuffer(handle).GetGlobalSymbol();
    llvm::Value *Load = Builder.CreateLoad(Symbol, "DispatchRaysConstants");

    Function *CreateHandleForLib = HlslOP->GetOpFunc(DXIL::OpCode::CreateHandleForLib, Load->getType());
    Constant *CreateHandleOpcodeArg = HlslOP->GetU32Const((unsigned)DXIL::OpCode::CreateHandleForLib);
    DispatchRaysConstantsHandle = Builder.CreateCall(CreateHandleForLib, { CreateHandleOpcodeArg, Load }, CbufferName);
  }
  
  // Raygen always reads from the start so no offset calculations needed
  if (ShaderKind != DXIL::ShaderKind::RayGeneration)
  {
      std::string ShaderRecordOffsetFuncName = "\x1?Fallback_ShaderRecordOffset@@YAIXZ";
      Function *ShaderRecordOffsetFunc = M.getFunction(ShaderRecordOffsetFuncName);
      if (!ShaderRecordOffsetFunc)
      {
          FunctionType *ShaderRecordOffsetFuncType = FunctionType::get(llvm::Type::getInt32Ty(Ctx), {}, false);
          ShaderRecordOffsetFunc = Function::Create(ShaderRecordOffsetFuncType, GlobalValue::LinkageTypes::ExternalLinkage, ShaderRecordOffsetFuncName, &M);
      }
      BaseShaderRecordOffset = Builder.CreateCall(ShaderRecordOffsetFunc, {}, "shaderRecordOffset");
  }
  else
  {
      BaseShaderRecordOffset = HlslOP->GetU32Const(0);
  }
}

llvm::Value *DxilPatchShaderRecordBindings::CreateOffsetToShaderRecord(Module &M, IRBuilder<> &Builder, unsigned int RecordOffsetInBytes, llvm::Value *CbufferOffsetInBytes) {
  DxilModule &DM = M.GetOrCreateDxilModule();
  OP *HlslOP = DM.GetOP();

  // Create handle for the newly-added constant buffer (which is achieved via a function call)
  auto AdddName = "ShaderRecordOffsetInBytes";
  Constant *ShaderRecordOffsetInBytes = HlslOP->GetU32Const(RecordOffsetInBytes); // Offset of constants in shader record buffer
  return Builder.CreateAdd(CbufferOffsetInBytes, ShaderRecordOffsetInBytes, AdddName);
}

llvm::Value *DxilPatchShaderRecordBindings::CreateCBufferLoadLegacy(Module &M, IRBuilder<> &Builder, llvm::Value *ResourceHandle, unsigned int RowToLoad) {
  DxilModule &DM = M.GetOrCreateDxilModule();
  OP *HlslOP = DM.GetOP();
  LLVMContext & Ctx = M.getContext();

  auto BufferLoadName = "ConstantBuffer";
  Function *BufferLoad = HlslOP->GetOpFunc(DXIL::OpCode::CBufferLoadLegacy, Type::getInt32Ty(Ctx));
  Constant *CBufferLoadOpcodeArg = HlslOP->GetU32Const((unsigned)DXIL::OpCode::CBufferLoadLegacy);
  Constant *RowToLoadConst = HlslOP->GetU32Const(RowToLoad);
  return Builder.CreateCall(BufferLoad, { CBufferLoadOpcodeArg, ResourceHandle, RowToLoadConst }, BufferLoadName);
}

llvm::Value *DxilPatchShaderRecordBindings::CreateShaderRecordBufferLoad(Module &M, IRBuilder<> &Builder, llvm::Value *ShaderRecordOffsetInBytes, llvm::Type* type) {
  DxilModule &DM = M.GetOrCreateDxilModule();
  OP *HlslOP = DM.GetOP();
  LLVMContext & Ctx = M.getContext();

  // Create handle for the newly-added constant buffer (which is achieved via a function call)
  auto BufferLoadName = "ShaderRecordBuffer";
  if (type->getNumContainedTypes() > 1)
  {
      // TODO: Buffer loads aren't legal with container types, check if this is the right wait to handle this
      type = type->getContainedType(0);
  }

  // TODO Do I need to check the result? Hopefully not
  Function *BufferLoad = HlslOP->GetOpFunc(DXIL::OpCode::BufferLoad, type);
  Constant *BufferLoadOpcodeArg = HlslOP->GetU32Const((unsigned)DXIL::OpCode::BufferLoad);
  Constant *Unused = UndefValue::get(llvm::Type::getInt32Ty(Ctx));
  return Builder.CreateCall(BufferLoad, { BufferLoadOpcodeArg, ShaderTableHandle, ShaderRecordOffsetInBytes, Unused }, BufferLoadName);
}

void DxilPatchShaderRecordBindings::ReplaceUsesOfWith(llvm::Instruction *InstructionToReplace, llvm::Value *ValueToReplaceWith) {
  for (auto UserIter = InstructionToReplace->user_begin(); UserIter != InstructionToReplace->user_end();) {
    // Increment the iterator before the replace since the replace alters the uses list
    auto userInstr = UserIter++;
    userInstr->replaceUsesOfWith(InstructionToReplace, ValueToReplaceWith);
  }
  InstructionToReplace->eraseFromParent();
}

llvm::Value *DxilPatchShaderRecordBindings::CreateCBufferLoadOffsetInBytes(Module &M, IRBuilder<> &Builder, llvm::Instruction *instruction) {
  DxilModule &DM = M.GetOrCreateDxilModule();
  OP *HlslOP = DM.GetOP();

  DxilInst_CBufferLoad cbufferLoad(instruction);
  DxilInst_CBufferLoadLegacy cbufferLoadLegacy(instruction);
  if (cbufferLoad) {
    return cbufferLoad.get_byteOffset();
  } else if (cbufferLoadLegacy) {
    Constant *LegacyMultiplier = HlslOP->GetU32Const(16);
    return Builder.CreateMul(cbufferLoadLegacy.get_regIndex(), LegacyMultiplier);
  } else {
    ThrowFailure();
    return nullptr;
  }
}

bool DxilPatchShaderRecordBindings::IsCBufferLoad(llvm::Instruction *instruction) {
  DxilInst_CBufferLoad cbufferLoad(instruction);
  DxilInst_CBufferLoadLegacy cbufferLoadLegacy(instruction);
  return cbufferLoad || cbufferLoadLegacy;
}

unsigned int GetResolvedRangeID(DXIL::ResourceClass resClass, Value *rangeIdVal)
{
  if (auto CI = dyn_cast<ConstantInt>(rangeIdVal))
  {
    return CI->getZExtValue();
  }
  else
  {
    assert(false);
    return 0;
  }
}

// TODO: This code is quite inefficient
bool DxilPatchShaderRecordBindings::GetHandleInfo(
  Module &M,
  DxilInst_CreateHandleForLib &createHandleStructForLib,
  _Out_ unsigned int &shaderRegister,
  _Out_ unsigned int &registerSpace,
  _Out_ DXIL::ResourceKind &kind,
  _Out_ DXIL::ResourceClass &resClass,
  _Out_ llvm::Type *&resType)
{
  DxilModule &DM = M.GetOrCreateDxilModule();
  LoadInst *loadRangeId = cast<LoadInst>(createHandleStructForLib.get_Resource());
  Value *ResourceSymbol = loadRangeId->getPointerOperand();

  DXIL::ResourceClass resourceClasses[] = {
    DXIL::ResourceClass::CBuffer,
    DXIL::ResourceClass::SRV,
    DXIL::ResourceClass::UAV,
    DXIL::ResourceClass::Sampler
  };

  hlsl::DxilResourceBase *Resource = nullptr;
  for (auto &resourceClass : resourceClasses) {
    
    switch (resourceClass)
    {
    case DXIL::ResourceClass::CBuffer:
    {
      auto &cbuffers = DM.GetCBuffers();
      for (auto &cbuffer : cbuffers)
      {
        if (cbuffer->GetGlobalSymbol() == ResourceSymbol)
        {
          Resource = cbuffer.get();
          break;
        }
      }
      break;
    }
    case DXIL::ResourceClass::SRV:
    case DXIL::ResourceClass::UAV:
    {
      auto &viewList = resourceClass == DXIL::ResourceClass::SRV ? DM.GetSRVs() : DM.GetUAVs();
      for (auto &view : viewList)
      {
        if (view->GetGlobalSymbol() == ResourceSymbol)
        {
          Resource = view.get();
          break;
        }
      }
      break;
    }
    case DXIL::ResourceClass::Sampler:
    {
      auto &samplers = DM.GetSamplers();
      for (auto &sampler : samplers)
      {
        if (sampler->GetGlobalSymbol() == ResourceSymbol)
        {
          Resource = sampler.get();
          break;
        }
      }
      break;
    }
    }
  }

  if (Resource)
  {
    registerSpace = Resource->GetSpaceID();
    shaderRegister = Resource->GetLowerBound();
    kind = Resource->GetKind();
    resClass = Resource->GetClass();
    resType = Resource->GetHLSLType()->getPointerElementType();
  }
  return Resource != nullptr;
}

llvm::Value *DxilPatchShaderRecordBindings::LoadShaderRecordData(
    Module &M, 
    IRBuilder<> &Builder,
    llvm::Value *offsetToShaderRecord,
    unsigned int dataOffsetInShaderRecord)
{
  DxilModule &DM = M.GetOrCreateDxilModule();
  LLVMContext &Ctx = M.getContext();
  OP *HlslOP = DM.GetOP();

  Constant *dataOffset =
      HlslOP->GetU32Const(dataOffsetInShaderRecord);
  Value *shaderTableOffsetToData = Builder.CreateAdd(dataOffset, offsetToShaderRecord);
  return CreateShaderRecordBufferLoad(M, Builder, shaderTableOffsetToData,
      llvm::Type::getInt32Ty(Ctx));
}

void DxilPatchShaderRecordBindings::PatchCreateHandleToUseDescriptorIndex(
    _In_ Module &M,
    _In_ IRBuilder<> &Builder,
    _In_ DXIL::ResourceKind &resourceKind,
    _In_ DXIL::ResourceClass &resourceClass,
    _In_ llvm::Type *resourceType,
    _In_ llvm::Value *descriptorIndex,
    _Inout_ DxilInst_CreateHandleForLib &createHandleInstr)
{
    DxilModule &DM = M.GetOrCreateDxilModule();
    OP *HlslOP = DM.GetOP();

    llvm::Value *descriptorHeapSymbol = GetAliasedDescriptorHeapHandle(M, resourceType, resourceClass, resourceKind);
    llvm::Value *viewSymbol = Builder.CreateGEP(descriptorHeapSymbol, { HlslOP->GetU32Const(0), descriptorIndex }, "IndexIntoDH");
    DxilMDHelper::MarkNonUniform(cast<Instruction>(viewSymbol));
    llvm::Value *handle = Builder.CreateLoad(viewSymbol);

    auto callInst = cast<CallInst>(createHandleInstr.Instr);
    callInst->setCalledFunction(HlslOP->GetOpFunc(
        DXIL::OpCode::CreateHandleForLib,
        handle->getType()));
    createHandleInstr.set_Resource(handle);
}

void DxilPatchShaderRecordBindings::InitializeViewTable() {
    // The Fallback Layer declares a bindless raw buffer that spans the entire descriptor heap,
    // manually add it to the list of UAV register spaces used
    if (*pInputShaderInfo->pNumUAVSpaces == 0)
    {
        ViewKey key = { (unsigned int)hlsl::DXIL::ResourceKind::RawBuffer, {0} };
        unsigned int index = FindOrInsertViewIntoList(
          key, 
          pInputShaderInfo->pUAVRegisterSpaceArray, 
          *pInputShaderInfo->pNumUAVSpaces, 
          FallbackLayerNumDescriptorHeapSpacesPerView);
        (void)index;
        assert(index == 0);
    }
}


void DxilPatchShaderRecordBindings::PatchShaderBindings(Module &M) {
  DxilModule &DM = M.GetOrCreateDxilModule();
  OP *HlslOP = DM.GetOP();

  // Don't erase instructions until the very end because it throws off the iterator
  std::vector<llvm::Instruction *> instructionsToRemove;
  for (BasicBlock &block : EntryPointFunction->getBasicBlockList()) {
    auto & Instructions = block.getInstList();

    for (auto &instr : Instructions) {
      DxilInst_CreateHandleForLib createHandleForLib(&instr);
      if (createHandleForLib) {
        DXIL::ResourceClass resourceClass;
        unsigned int registerSpace;
        unsigned int registerIndex;
        DXIL::ResourceKind kind;
        llvm::Type *resType;
        bool resourceIsResolved = true;
        resourceIsResolved = GetHandleInfo(M, createHandleForLib, registerIndex, registerSpace, kind, resourceClass, resType);

        if (!resourceIsResolved) continue; // TODO: This shouldn't actually be happening?

        ShaderRecordEntry shaderRecord = FindRootSignatureDescriptor(
          *pRootSignatureDesc,
          pInputShaderInfo->ShaderRecordIdentifierSizeInBytes,
          resourceClass,
          registerIndex,
          registerSpace);

        const bool IsBindingSpecifiedInLocalRootSignature = !shaderRecord.IsInvalid();
        if (IsBindingSpecifiedInLocalRootSignature) {
          if (!DispatchRaysConstantsHandle) {
            AddInputBinding(M);
          }

          switch (shaderRecord.ParameterType) {
          case DxilRootParameterType::Constants32Bit:
          {
            for (User *U : instr.users()) {
              llvm::Instruction *instruction = cast<CallInst>(U);
              if (IsCBufferLoad(instruction)) {
                llvm::Instruction *cbufferLoadInstr = instruction;
                IRBuilder<> Builder(cbufferLoadInstr);

                llvm::Value * cbufferOffsetInBytes = CreateCBufferLoadOffsetInBytes(M, Builder, cbufferLoadInstr);
                llvm::Value *LocalOffsetToRootConstant = CreateOffsetToShaderRecord(M, Builder, shaderRecord.RecordOffsetInBytes, cbufferOffsetInBytes);
                llvm::Value *GlobalOffsetToRootConstant = Builder.CreateAdd(LocalOffsetToRootConstant, BaseShaderRecordOffset);
                llvm::Value *srvBufferLoad = CreateShaderRecordBufferLoad(M, Builder, GlobalOffsetToRootConstant, cbufferLoadInstr->getType());
                ReplaceUsesOfWith(cbufferLoadInstr, srvBufferLoad);
              } else {
                ThrowFailure();
              }
            }
            instructionsToRemove.push_back(&instr);
            break;
          }
          case DxilRootParameterType::DescriptorTable:
          {
            IRBuilder<> Builder(&instr);
            llvm::Value *srvBufferLoad = LoadShaderRecordData(
             M, 
             Builder, 
             BaseShaderRecordOffset,
             shaderRecord.RecordOffsetInBytes);

            llvm::Value *DescriptorTableEntryLo = Builder.CreateExtractValue(srvBufferLoad, 0, "DescriptorTableHandleLo");

            unsigned int offsetToLoadInUints = offsetof(DispatchRaysConstants, SrvCbvUavDescriptorHeapStart) / sizeof(uint32_t);
            unsigned int uintsPerRow = 4;
            unsigned int rowToLoad = offsetToLoadInUints / uintsPerRow;
            unsigned int extractValueOffset = offsetToLoadInUints % uintsPerRow;
            llvm::Value *DescHeapConstants = CreateCBufferLoadLegacy(M, Builder, DispatchRaysConstantsHandle, rowToLoad);
            llvm::Value *DescriptorHeapStartAddressLo = Builder.CreateExtractValue(DescHeapConstants, extractValueOffset, "DescriptorHeapStartHandleLo");

            // TODO: The hi bits can only be ignored if the difference is guaranteed to be < 32 bytes. This is an unsafe assumption, particularly given 
            // large descriptor sizes
            llvm::Value *DescriptorTableOffsetInBytes = Builder.CreateSub(DescriptorTableEntryLo, DescriptorHeapStartAddressLo, "TableOffsetInBytes");

            Constant *DescriptorSizeInBytes = HlslOP->GetU32Const(pInputShaderInfo->SrvCbvUavDescriptorSizeInBytes);
            llvm::Value * DescriptorTableStartIndex = Builder.CreateExactUDiv(DescriptorTableOffsetInBytes, DescriptorSizeInBytes, "TableStartIndex");

            Constant *RecordOffset = HlslOP->GetU32Const(shaderRecord.OffsetInDescriptors);
            llvm::Value * BaseDescriptorIndex = Builder.CreateAdd(DescriptorTableStartIndex, RecordOffset, "BaseDescriptorIndex");

            // TODO: Not supporting dynamic indexing yet, should be pulled from CreateHandleForLib
            // If dynamic indexing is being used, add the apps index on top of the calculated index
            llvm::Value * DynamicIndex = HlslOP->GetU32Const(0);

            llvm::Value * DescriptorIndex = Builder.CreateAdd(BaseDescriptorIndex, DynamicIndex, "DescriptorIndex");
            PatchCreateHandleToUseDescriptorIndex(
                M, 
                Builder, 
                kind, 
                resourceClass, 
                resType, 
                DescriptorIndex, 
                createHandleForLib);
            break;
          }
          case DxilRootParameterType::CBV:
          case DxilRootParameterType::SRV:
          case DxilRootParameterType::UAV: {
            IRBuilder<> Builder(&instr);
            llvm::Value *srvBufferLoad = LoadShaderRecordData(
             M, 
             Builder, 
             BaseShaderRecordOffset,
             shaderRecord.RecordOffsetInBytes);

            llvm::Value *DescriptorIndex = Builder.CreateExtractValue(
                srvBufferLoad, 1, "DescriptorHeapIndex");

            // TODO: Handle offset in bytes
            // llvm::Value *OffsetInBytes = Builder.CreateExtractValue(
            //     srvBufferLoad, 0, "OffsetInBytes");

            PatchCreateHandleToUseDescriptorIndex(
                M,
                Builder,
                kind,
                resourceClass,
                resType,
                DescriptorIndex,
                createHandleForLib);

            break;
          }
          default:
            ThrowFailure();
            break;
          }
        }
      }
    }
  }

  for (auto instruction : instructionsToRemove) {
    instruction->eraseFromParent();
  }

}

bool IsParameterTypeCompatibleWithResourceClass(
  DXIL::ResourceClass resourceClass,
  DxilRootParameterType parameterType) {
  switch (parameterType) {
  case DxilRootParameterType::DescriptorTable:
    return true;
  case DxilRootParameterType::Constants32Bit:
  case DxilRootParameterType::CBV:
    return resourceClass == DXIL::ResourceClass::CBuffer;
  case DxilRootParameterType::SRV:
    return resourceClass == DXIL::ResourceClass::SRV;
  case DxilRootParameterType::UAV:
    return resourceClass == DXIL::ResourceClass::UAV;
  default:
    ThrowFailure();
    return false;
  }
}

DxilRootParameterType ConvertD3D12ParameterTypeToDxil(DxilRootParameterType parameter) {
  switch (parameter) {
  case DxilRootParameterType::Constants32Bit:
    return DxilRootParameterType::Constants32Bit;
  case DxilRootParameterType::DescriptorTable:
    return DxilRootParameterType::DescriptorTable;
  case DxilRootParameterType::CBV:
    return DxilRootParameterType::CBV;
  case DxilRootParameterType::SRV:
    return DxilRootParameterType::SRV;
  case DxilRootParameterType::UAV:
    return DxilRootParameterType::UAV;
  }

  assert(false);
  return (DxilRootParameterType)-1;
}

DXIL::ResourceClass ConvertD3D12RangeTypeToDxil(DxilDescriptorRangeType rangeType) {
  switch (rangeType) {
  case DxilDescriptorRangeType::SRV:
    return DXIL::ResourceClass::SRV;
  case DxilDescriptorRangeType::UAV:
    return DXIL::ResourceClass::UAV;
  case DxilDescriptorRangeType::CBV:
    return DXIL::ResourceClass::CBuffer;
  case DxilDescriptorRangeType::Sampler:
    return DXIL::ResourceClass::Sampler;
  }
  assert(false);
  return (DXIL::ResourceClass) - 1;
}

unsigned int GetParameterTypeAlignment(DxilRootParameterType parameterType) {
  switch (parameterType) {
  case DxilRootParameterType::DescriptorTable:
    return SizeofD3D12GpuDescriptorHandle;
  case DxilRootParameterType::Constants32Bit:
    return sizeof(uint32_t);
  case DxilRootParameterType::CBV: // fallthrough
  case DxilRootParameterType::SRV: // fallthrough
  case DxilRootParameterType::UAV:
    return SizeofD3D12GpuVA;
  default:
    return UINT_MAX;
  }
}

template <typename TD3D12_ROOT_SIGNATURE_DESC>
ShaderRecordEntry FindRootSignatureDescriptorHelper(
    const TD3D12_ROOT_SIGNATURE_DESC &rootSignatureDescriptor,
    unsigned int ShaderRecordIdentifierSizeInBytes,
    DXIL::ResourceClass resourceClass, unsigned int baseRegisterIndex,
    unsigned int registerSpace) {
  // Automatically fail if it's looking for a fallback binding as these never
  // need to be patched
  if (registerSpace != FallbackLayerRegisterSpace) {
    unsigned int recordOffset = ShaderRecordIdentifierSizeInBytes;
    for (unsigned int rootParamIndex = 0;
         rootParamIndex < rootSignatureDescriptor.NumParameters;
         rootParamIndex++) {
      auto &rootParam = rootSignatureDescriptor.pParameters[rootParamIndex];
      auto dxilParamType =
          ConvertD3D12ParameterTypeToDxil(rootParam.ParameterType);

#define ALIGN(alignment, num) (((num + alignment - 1) / alignment) * alignment)
      recordOffset = ALIGN(GetParameterTypeAlignment(rootParam.ParameterType),
                           recordOffset);

      switch (rootParam.ParameterType) {
      case DxilRootParameterType::Constants32Bit:
        if (IsParameterTypeCompatibleWithResourceClass(resourceClass,
                                                       dxilParamType) &&
            baseRegisterIndex == rootParam.Constants.ShaderRegister &&
            registerSpace == rootParam.Constants.RegisterSpace) {
          return {dxilParamType, recordOffset, 0};
        }
        recordOffset += rootParam.Constants.Num32BitValues * sizeof(uint32_t);
        break;
      case DxilRootParameterType::DescriptorTable: {
        auto &descriptorTable = rootParam.DescriptorTable;

        unsigned int rangeOffsetInDescriptors = 0;
        for (unsigned int rangeIndex = 0;
             rangeIndex < descriptorTable.NumDescriptorRanges; rangeIndex++) {
          auto &range = descriptorTable.pDescriptorRanges[rangeIndex];
          if (range.OffsetInDescriptorsFromTableStart != (unsigned)-1) {
            rangeOffsetInDescriptors = range.OffsetInDescriptorsFromTableStart;
          }

          if (ConvertD3D12RangeTypeToDxil(range.RangeType) == resourceClass &&
              range.RegisterSpace == registerSpace &&
              range.BaseShaderRegister <= baseRegisterIndex &&
              range.BaseShaderRegister + range.NumDescriptors >
                  baseRegisterIndex) {
            rangeOffsetInDescriptors +=
                baseRegisterIndex - range.BaseShaderRegister;
            return {dxilParamType, recordOffset, rangeOffsetInDescriptors};
          }

          rangeOffsetInDescriptors += range.NumDescriptors;
        }

        recordOffset += SizeofD3D12GpuDescriptorHandle;
        break;
      }
      case DxilRootParameterType::CBV:
      case DxilRootParameterType::SRV:
      case DxilRootParameterType::UAV:
        if (IsParameterTypeCompatibleWithResourceClass(resourceClass,
                                                       dxilParamType) &&
            baseRegisterIndex == rootParam.Descriptor.ShaderRegister &&
            registerSpace == rootParam.Descriptor.RegisterSpace) {
          return {dxilParamType, recordOffset, 0};
        }

        recordOffset += SizeofD3D12GpuVA;
        break;
      }
    }
  }
  return ShaderRecordEntry::InvalidEntry();
}

// TODO: Consider pre-calculating this into a map
ShaderRecordEntry DxilPatchShaderRecordBindings::FindRootSignatureDescriptor(
  const DxilVersionedRootSignatureDesc &rootSignatureDescriptor,
  unsigned int ShaderRecordIdentifierSizeInBytes,
  DXIL::ResourceClass resourceClass,
  unsigned int baseRegisterIndex,
  unsigned int registerSpace) {
  switch (rootSignatureDescriptor.Version) {
  case DxilRootSignatureVersion::Version_1_0:
    return FindRootSignatureDescriptorHelper(rootSignatureDescriptor.Desc_1_0, ShaderRecordIdentifierSizeInBytes, resourceClass, baseRegisterIndex, registerSpace);
  case DxilRootSignatureVersion::Version_1_1:
    return FindRootSignatureDescriptorHelper(rootSignatureDescriptor.Desc_1_1, ShaderRecordIdentifierSizeInBytes, resourceClass, baseRegisterIndex, registerSpace);
  default:
    ThrowFailure();
    return ShaderRecordEntry::InvalidEntry();
  }
}




