///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilShaderAccessTracking.cpp                                        //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides a pass to add instrumentation to determine pixel hit count and   //
// cost. Used by PIX.                                                        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilOperations.h"

#include "dxc/DXIL/DxilConstants.h"
#include "dxc/DXIL/DxilInstructions.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilResourceBinding.h"
#include "dxc/DXIL/DxilResourceProperties.h"
#include "dxc/DxilPIXPasses/DxilPIXPasses.h"
#include "dxc/DxilPIXPasses/DxilPIXVirtualRegisters.h"
#include "dxc/HLSL/DxilGenerationPass.h"
#include "dxc/HLSL/DxilSpanAllocator.h"

#include "llvm/IR/PassManager.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Transforms/Utils/Local.h"

#include <deque>

#include "PixPassHelpers.h"

#ifdef _WIN32
#include <winerror.h>
#endif

using namespace llvm;
using namespace hlsl;
using namespace hlsl::DXIL::OperandIndex;

void ThrowIf(bool a) {
  if (a) {
    throw ::hlsl::Exception(E_INVALIDARG);
  }
}

//---------------------------------------------------------------------------------------------------------------------------------
// These types are taken from PIX's ShaderAccessHelpers.h

enum class ShaderAccessFlags : uint32_t {
  None = 0,
  Read = 1 << 0,
  Write = 1 << 1,

  // "Counter" access is only applicable to UAVs; it means the counter buffer
  // attached to the UAV was accessed, but not necessarily the UAV resource.
  Counter = 1 << 2,

  Sampler = 1 << 3,

  // Descriptor-only read (if any), but not the resource contents (if any).
  // Used for GetDimensions, samplers, and secondary texture for sampler
  // feedback.
  // TODO: Make this a unique value if supported in PIX, then enable
  // GetDimensions
  DescriptorRead = 1 << 0,
};

// Bits in encoded dword:
// 33222222222211111111110000000000
// 10987654321098765432109876543210
// kkkkisssrrrrrrrrrrrrrrrrrrrrrrrr
//
// k: four bits ShaderKind
// i: one bit InstructionOrdinalndicator
// r: 24 bits if i = 0 (resource index) else (instruction ordinal)

constexpr uint32_t InstructionOrdinalndicator = 0x0800'0000;

// (end shared types)
//---------------------------------------------------------------------------------------------------------------------------------

static uint32_t EncodeShaderModel(DXIL::ShaderKind kind) {
  DXASSERT_NOMSG(static_cast<int>(DXIL::ShaderKind::Invalid) <= 16);
  return static_cast<uint32_t>(kind) << 28;
}

enum class ResourceAccessStyle {
  None,
  Sampler,
  UAVRead,
  UAVWrite,
  CBVRead,
  SRVRead,
  EndOfEnum
};

static uint32_t EncodeAccess(ResourceAccessStyle access) {
  DXASSERT_NOMSG(static_cast<int>(ResourceAccessStyle::EndOfEnum) <= 8);
  uint32_t encoded = static_cast<uint32_t>(access);
  return encoded << 24;
}

constexpr uint32_t DWORDsPerResource = 3;
constexpr uint32_t BytesPerDWORD = 4;

static uint32_t OffsetFromAccess(ShaderAccessFlags access) {
  switch (access) {
  case ShaderAccessFlags::Read:
    return 0;
  case ShaderAccessFlags::Write:
    return 1;
  case ShaderAccessFlags::Counter:
    return 2;
  default:
    throw ::hlsl::Exception(E_INVALIDARG);
  }
}

// This enum doesn't have to match PIX's version, because the values are
// received from PIX encoded in ASCII. However, for ease of comparing this code
// with PIX, and to be less confusing to future maintainers, this enum does
// indeed match the same-named enum in PIX.
enum class RegisterType {
  CBV,
  SRV,
  UAV,
  RTV, // not used.
  DSV, // not used.
  Sampler,
  SOV, // not used.
  Invalid,
  Terminator
};

RegisterType RegisterTypeFromResourceClass(DXIL::ResourceClass c) {
  switch (c) {
  case DXIL::ResourceClass::SRV:
    return RegisterType::SRV;
    break;
  case DXIL::ResourceClass::UAV:
    return RegisterType::UAV;
    break;
  case DXIL::ResourceClass::CBuffer:
    return RegisterType::CBV;
    break;
  case DXIL::ResourceClass::Sampler:
    return RegisterType::Sampler;
    break;
  case DXIL::ResourceClass::Invalid:
    return RegisterType::Invalid;
    break;
  default:
    ThrowIf(true);
    return RegisterType::Invalid;
  }
}

struct RegisterTypeAndSpace {
  bool operator<(const RegisterTypeAndSpace &o) const {
    return static_cast<int>(Type) < static_cast<int>(o.Type) ||
           (static_cast<int>(Type) == static_cast<int>(o.Type) &&
            Space < o.Space);
  }
  RegisterType Type;
  unsigned Space;
};

// Identifies a bind point as defined by the root signature
struct RSRegisterIdentifier {
  RegisterType Type;
  unsigned Space;
  unsigned Index;

  bool operator<(const RSRegisterIdentifier &o) const {
    return static_cast<unsigned>(Type) < static_cast<unsigned>(o.Type) &&
           Space < o.Space && Index < o.Index;
  }
};

struct SlotRange {
  unsigned startSlot;
  unsigned numSlots;

  // Number of slots needed if no descriptors from unbounded ranges are included
  unsigned numInvariableSlots;
};

enum class AccessStyle { None, FromRootSig, ResourceFromDescriptorHeap, SamplerFromDescriptorHeap };
struct DxilResourceAndClass {
  AccessStyle accessStyle;
  RegisterType registerType;
  int RegisterSpace;
  unsigned RegisterID;
  Value *index;
  Value *dynamicallyBoundIndex;
};

//---------------------------------------------------------------------------------------------------------------------------------

class DxilShaderAccessTracking : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilShaderAccessTracking() : ModulePass(ID) {}
  StringRef getPassName() const override {
    return "DXIL shader access tracking";
  }
  bool runOnModule(Module &M) override;
  void applyOptions(PassOptions O) override;

private:
  void EmitAccess(LLVMContext &Ctx, OP *HlslOP, IRBuilder<> &, Value *slot,
                  ShaderAccessFlags access);
  bool EmitResourceAccess(DxilModule & DM, DxilResourceAndClass &res, Instruction *instruction,
                          OP *HlslOP, LLVMContext &Ctx,
                          ShaderAccessFlags readWrite);
  DxilResourceAndClass GetResourceFromHandle(Value* resHandle, DxilModule& DM);

private:
  struct DynamicResourceBinding {
    int HeapIndex;
    bool HeapIsSampler; // else resource
    std::string Name;
  };

  std::vector<DynamicResourceBinding> m_dynamicResourceBindings;
  bool m_CheckForDynamicIndexing = false;
  int m_DynamicResourceDataOffset = -1;
  int m_DynamicSamplerDataOffset = -1;
  int m_OutputBufferSize = -1;
  std::map<RegisterTypeAndSpace, SlotRange> m_slotAssignments;
  std::map<llvm::Function *, CallInst *> m_FunctionToUAVHandle;
  std::map<llvm::Function *, std::map<ResourceAccessStyle, Constant *>> m_FunctionToEncodedAccess;
  std::set<RSRegisterIdentifier> m_DynamicallyIndexedBindPoints;
};

static unsigned DeserializeInt(std::deque<char> &q) {
  unsigned i = 0;

  while (!q.empty() && isdigit(q.front())) {
    i *= 10;
    i += q.front() - '0';
    q.pop_front();
  }
  return i;
}

static char DequeFront(std::deque<char> &q) {
  ThrowIf(q.empty());
  auto c = q.front();
  q.pop_front();
  return c;
}

static RegisterType ParseRegisterType(std::deque<char> &q) {
  switch (DequeFront(q)) {
  case 'C':
    return RegisterType::CBV;
  case 'S':
    return RegisterType::SRV;
  case 'U':
    return RegisterType::UAV;
  case 'M':
    return RegisterType::Sampler;
  case 'I':
    return RegisterType::Invalid;
  default:
    return RegisterType::Terminator;
  }
}

static char EncodeRegisterType(RegisterType r) {
  switch (r) {
  case RegisterType::CBV:
    return 'C';
  case RegisterType::SRV:
    return 'S';
  case RegisterType::UAV:
    return 'U';
  case RegisterType::Sampler:
    return 'M';
  case RegisterType::Invalid:
    return 'I';
  }
  return '.';
}

static void ValidateDelimiter(std::deque<char> &q, char d) {
  ThrowIf(q.front() != d);
  q.pop_front();
}

void DxilShaderAccessTracking::applyOptions(PassOptions O) {
  int checkForDynamic;
  GetPassOptionInt(O, "checkForDynamicIndexing", &checkForDynamic, 0);
  m_CheckForDynamicIndexing = checkForDynamic != 0;

  StringRef configOption;
  if (GetPassOption(O, "config", &configOption)) {
    std::deque<char> config;
    config.assign(configOption.begin(), configOption.end());

    // Parse slot assignments. Compare with PIX's ShaderAccessHelpers.cpp
    // (TrackingConfiguration::SerializedRepresentation)
    RegisterType rt = ParseRegisterType(config);
    while (rt != RegisterType::Terminator) {

      RegisterTypeAndSpace rst;
      rst.Type = rt;

      rst.Space = DeserializeInt(config);
      ValidateDelimiter(config, ':');

      SlotRange sr;
      sr.startSlot = DeserializeInt(config);
      ValidateDelimiter(config, ':');

      sr.numSlots = DeserializeInt(config);
      ValidateDelimiter(config, 'i');

      sr.numInvariableSlots = DeserializeInt(config);
      ValidateDelimiter(config, ';');

      m_slotAssignments[rst] = sr;

      rt = ParseRegisterType(config);
    }
    m_DynamicResourceDataOffset = DeserializeInt(config);
    ValidateDelimiter(config, ';');
    m_DynamicSamplerDataOffset = DeserializeInt(config);
    ValidateDelimiter(config, ';');
    m_OutputBufferSize = DeserializeInt(config);
  }
}

void DxilShaderAccessTracking::EmitAccess(LLVMContext &Ctx, OP *HlslOP,
                                          IRBuilder<> &Builder,
                                          Value *ByteIndex,
                                          ShaderAccessFlags access) {

  unsigned OffsetForAccessType =
      static_cast<unsigned>(OffsetFromAccess(access) * BytesPerDWORD);
  auto OffsetByteIndex = Builder.CreateAdd(
      ByteIndex, HlslOP->GetU32Const(OffsetForAccessType), "OffsetByteIndex");

  UndefValue *UndefIntArg = UndefValue::get(Type::getInt32Ty(Ctx));
  Constant *LiteralOne = HlslOP->GetU32Const(1);
  Constant *ElementMask = HlslOP->GetI8Const(1);

  Function *StoreFunc =
      HlslOP->GetOpFunc(OP::OpCode::BufferStore, Type::getInt32Ty(Ctx));
  Constant *StoreOpcode =
      HlslOP->GetU32Const((unsigned)OP::OpCode::BufferStore);
  (void)Builder.CreateCall(
      StoreFunc,
      {
          StoreOpcode, // i32, ; opcode
          m_FunctionToUAVHandle.at(
              Builder.GetInsertBlock()
                  ->getParent()), // %dx.types.Handle, ; resource handle
          OffsetByteIndex,        // i32, ; coordinate c0: byte offset
          UndefIntArg,            // i32, ; coordinate c1 (unused)
          LiteralOne,             // i32, ; value v0
          UndefIntArg,            // i32, ; value v1
          UndefIntArg,            // i32, ; value v2
          UndefIntArg,            // i32, ; value v3
          ElementMask             // i8 ; just the first value is used
      });
}

static ResourceAccessStyle AccessStyleFromAccessAndType(
    AccessStyle accessStyle, 
    RegisterType registerType,
    ShaderAccessFlags readWrite)
{
    switch (accessStyle)
    {
    case AccessStyle::ResourceFromDescriptorHeap:
        switch (registerType)
        {
        case RegisterType::CBV:
          return ResourceAccessStyle::CBVRead;
        case RegisterType::SRV:
          return ResourceAccessStyle::SRVRead;
        case RegisterType::UAV:
            return readWrite == ShaderAccessFlags::Read ?
                ResourceAccessStyle::UAVRead :
                ResourceAccessStyle::UAVWrite;
        default:
          return ResourceAccessStyle::None;
        }
    case AccessStyle::SamplerFromDescriptorHeap:
        return ResourceAccessStyle::Sampler;
    default:
        return ResourceAccessStyle::None;
    }
}

bool DxilShaderAccessTracking::EmitResourceAccess(DxilModule &DM, 
                                                  DxilResourceAndClass &res,
                                                  Instruction *instruction,
                                                  OP *HlslOP, LLVMContext &Ctx,
                                                  ShaderAccessFlags readWrite) {
  IRBuilder<> Builder(instruction);
  
  if (res.accessStyle == AccessStyle::FromRootSig) {
    RegisterTypeAndSpace typeAndSpace{
        res.registerType, 
        static_cast<unsigned>(res.RegisterSpace) // reserved spaces are -ve, but user spaces can only be +ve
    };

    auto slot = m_slotAssignments.find(typeAndSpace);
    // If the assignment isn't found, we assume it's not accessed
    if (slot != m_slotAssignments.end()) {

        Value *slotIndex;
    
      if (isa<ConstantInt>(res.index)) {
        unsigned index = cast<ConstantInt>(res.index)->getLimitedValue();
        if (index > slot->second.numSlots) {
          // out-of-range accesses are written to slot zero:
          slotIndex = HlslOP->GetU32Const(0);
        } else {
          slotIndex = HlslOP->GetU32Const((slot->second.startSlot + index) *
                                          DWORDsPerResource * BytesPerDWORD);
        }
      } else {
        RSRegisterIdentifier id{typeAndSpace.Type, typeAndSpace.Space,
                                res.RegisterID};
        m_DynamicallyIndexedBindPoints.emplace(std::move(id));
    
        // CompareWithSlotLimit will contain 1 if the access is out-of-bounds
        // (both over- and and under-flow via the unsigned >= with slot count)
        auto CompareWithSlotLimit = Builder.CreateICmpUGE(
            res.index, HlslOP->GetU32Const(slot->second.numSlots),
            "CompareWithSlotLimit");
        auto CompareWithSlotLimitAsUint = Builder.CreateCast(
            Instruction::CastOps::ZExt, CompareWithSlotLimit,
            Type::getInt32Ty(Ctx), "CompareWithSlotLimitAsUint");
    
        // IsInBounds will therefore contain 0 if the access is out-of-bounds, and
        // 1 otherwise.
        auto IsInBounds = Builder.CreateSub(
            HlslOP->GetU32Const(1), CompareWithSlotLimitAsUint, "IsInBounds");
    
        auto SlotDwordOffset = Builder.CreateAdd(
            res.index, HlslOP->GetU32Const(slot->second.startSlot),
            "SlotDwordOffset");
        auto SlotByteOffset = Builder.CreateMul(
            SlotDwordOffset,
            HlslOP->GetU32Const(DWORDsPerResource * BytesPerDWORD),
            "SlotByteOffset");
    
        // This will drive an out-of-bounds access slot down to 0
        slotIndex = Builder.CreateMul(SlotByteOffset, IsInBounds, "slotIndex");
      }
    
      EmitAccess(Ctx, HlslOP, Builder, slotIndex, readWrite);
    
      return true; // did modify
    }
  }
  else if (m_DynamicResourceDataOffset != -1) {
      if (res.accessStyle == AccessStyle::ResourceFromDescriptorHeap ||
          res.accessStyle == AccessStyle::SamplerFromDescriptorHeap)
      {
          Constant* BaseOfRecordsForType;
          int LimitForType;
          if (res.accessStyle == AccessStyle::ResourceFromDescriptorHeap) {
              LimitForType = m_DynamicSamplerDataOffset - m_DynamicResourceDataOffset;
              BaseOfRecordsForType =
                  HlslOP->GetU32Const(m_DynamicResourceDataOffset);
          } else {
              LimitForType = m_OutputBufferSize - m_DynamicSamplerDataOffset;
              BaseOfRecordsForType =
                HlslOP->GetU32Const(m_DynamicSamplerDataOffset);
          }

          // Branchless limit: compare offset to size of data reserved for that type,
          // resulting in a value of 0 or 1.
          // Extend that 0/1 to an integer, and multiply the offset by that value.
          // Result: expected offset, or 0 if too large.

          // Add 1 to the index in order to skip over the zeroth entry: that's 
          // reserved for "out of bounds" writes.
          auto *IndexToWrite =
              Builder.CreateAdd(res.dynamicallyBoundIndex, HlslOP->GetU32Const(1));

          // Each record is two dwords:
          // the first dword is for write access, the second for read.
          Constant *SizeofRecord =
              HlslOP->GetU32Const(2 * static_cast<unsigned int>(sizeof(uint32_t)));
          auto *BaseOfRecord =
              Builder.CreateMul(IndexToWrite, SizeofRecord);
          Value* OffsetToWrite;
          if (readWrite == ShaderAccessFlags::Write) {
            OffsetToWrite = BaseOfRecord;
          }
          else {
            OffsetToWrite = Builder.CreateAdd(BaseOfRecord, 
                HlslOP->GetU32Const(static_cast<unsigned int>(sizeof(uint32_t))));
          }

          // Generate the 0 (out of bounds) or 1 (in-bounds) multiplier:
          Constant *BufferLimit = HlslOP->GetU32Const(LimitForType);
          auto *LimitBoolean =
              Builder.CreateICmpULT(OffsetToWrite, BufferLimit);
          
          auto * ZeroIfOutOfBounds = Builder.CreateCast(
              Instruction::CastOps::ZExt, LimitBoolean,
              Type::getInt32Ty(Ctx));
          
          // Limit the offset to the out-of-bounds record if the above generated 0,
          // or leave it as-is if the above generated 1:
          auto *LimitedOffset = Builder.CreateMul(OffsetToWrite, ZeroIfOutOfBounds);
          
          // Offset into the range of records for this type of access (resource or sampler)
          auto* Offset = Builder.CreateAdd(BaseOfRecordsForType, LimitedOffset);

          ResourceAccessStyle accessStyle = AccessStyleFromAccessAndType(
              res.accessStyle, 
              res.registerType,
              readWrite);

          Constant* EncodedFlags = m_FunctionToEncodedAccess
                                .at(Builder.GetInsertBlock()->getParent())
                                .at(accessStyle);

          // Now: if we're out-of-bounds, we'll actually write the offending instruction number instead,
          // again using the mul-by-one-or-zero trick
          auto* OneIfOutOfBounds = Builder.CreateSub(HlslOP->GetU32Const(1), ZeroIfOutOfBounds);
          auto* MultipliedEncodedFlags = Builder.CreateMul(ZeroIfOutOfBounds, EncodedFlags);
          uint32_t InstructionNumber = 0;
          (void)pix_dxil::PixDxilInstNum::FromInst(instruction, &InstructionNumber);
          auto const *shaderModel = DM.GetShaderModel();
          auto shaderKind = shaderModel->GetKind();
          uint32_t EncodedInstructionNumber =
            InstructionNumber | InstructionOrdinalndicator | EncodeShaderModel(shaderKind);
          auto* MultipliedOutOfBoundsValue = Builder.CreateMul(OneIfOutOfBounds, HlslOP->GetU32Const(EncodedInstructionNumber));
          auto* CombinedFlagOrInstructionValue = Builder.CreateAdd(MultipliedEncodedFlags, MultipliedOutOfBoundsValue);

          // If we failed to find an instruction value, just return the access flags:
          if (InstructionNumber == 0) {
            CombinedFlagOrInstructionValue = EncodedFlags;
          }
          Constant *ElementMask = HlslOP->GetI8Const(1);
          Function *StoreFunc =
              HlslOP->GetOpFunc(OP::OpCode::BufferStore, Type::getInt32Ty(Ctx));
          Constant *StoreOpcode =
              HlslOP->GetU32Const((unsigned)OP::OpCode::BufferStore);
          UndefValue *UndefArg = UndefValue::get(Type::getInt32Ty(Ctx));
          (void)Builder.CreateCall(
              StoreFunc,
              {
                  StoreOpcode,                  // i32, ; opcode
                  m_FunctionToUAVHandle.at(
                      Builder.GetInsertBlock()
                          ->getParent()),       // %dx.types.Handle, ; resource handle
                  Offset,                // i32, ; coordinate c0: byte offset
                  UndefArg,                     // i32, ; coordinate c1 (unused)
                  CombinedFlagOrInstructionValue, // i32, ; value v0
                  UndefArg,                     // i32, ; value v1
                  UndefArg,                     // i32, ; value v2
                  UndefArg,                     // i32, ; value v3
                  ElementMask                   // i8 ; just the first value is used
              });
          return true; // did modify
      }
  }

  return false; // did not modify
}

DxilResourceAndClass
DxilShaderAccessTracking::GetResourceFromHandle(Value *resHandle,
                                                DxilModule &DM) {

  DxilResourceAndClass ret{
      AccessStyle::None, 
      RegisterType::Terminator,
      0,
      0,
      nullptr,
      nullptr};

  Constant *C = dyn_cast<Constant>(resHandle);
  if (C && C->isZeroValue()) {
    return ret;
  }

  CallInst *handle = cast<CallInst>(resHandle);

  unsigned rangeId = -1;

  if (hlsl::OP::IsDxilOpFuncCallInst(handle, hlsl::OP::OpCode::CreateHandle))
  {
    DxilInst_CreateHandle createHandle(handle);

    // Dynamic rangeId is not supported - skip and let validation report the
    // error.
    if (isa<ConstantInt>(createHandle.get_rangeId())) {
        rangeId = cast<ConstantInt>(createHandle.get_rangeId())->getLimitedValue();

        auto resClass = static_cast<DXIL::ResourceClass>(createHandle.get_resourceClass_val());

        DxilResourceBase* resource = nullptr;
        RegisterType registerType = RegisterType::Invalid;
        switch (resClass) {
        case DXIL::ResourceClass::SRV:
            resource = &DM.GetSRV(rangeId);
            registerType = RegisterType::SRV;
            break;
        case DXIL::ResourceClass::UAV:
            resource = &DM.GetUAV(rangeId);
          registerType = RegisterType::UAV;
          break;
        case DXIL::ResourceClass::CBuffer:
            resource = &DM.GetCBuffer(rangeId);
            registerType = RegisterType::CBV;
            break;
        case DXIL::ResourceClass::Sampler:
            resource = &DM.GetSampler(rangeId);
            registerType = RegisterType::Sampler;
            break;
        }
        if (resource != nullptr) {
            ret.index = createHandle.get_index();
            ret.registerType = registerType;
            ret.accessStyle = AccessStyle::FromRootSig;
            ret.RegisterID = resource->GetID();
            ret.RegisterSpace = resource->GetSpaceID();
        }
    }
  } else if (hlsl::OP::IsDxilOpFuncCallInst(handle, hlsl::OP::OpCode::AnnotateHandle)) {
      DxilInst_AnnotateHandle annotateHandle(handle);
      auto properties = hlsl::resource_helper::loadPropsFromAnnotateHandle(
          annotateHandle, *DM.GetShaderModel());

      auto* handleCreation = dyn_cast<CallInst>(annotateHandle.get_res());
      if (handleCreation != nullptr) {
        if (hlsl::OP::IsDxilOpFuncCallInst(handleCreation, hlsl::OP::OpCode::CreateHandleFromBinding)) {
            DxilInst_CreateHandleFromBinding createHandleFromBinding(handleCreation);
            Constant* B = cast<Constant>(createHandleFromBinding.get_bind());
            auto binding = hlsl::resource_helper::loadBindingFromConstant(*B);
            ret.accessStyle = AccessStyle::FromRootSig;
            ret.index = createHandleFromBinding.get_index();
            ret.registerType = RegisterTypeFromResourceClass(
                static_cast<hlsl::DXIL::ResourceClass>(binding.resourceClass));
            ret.RegisterSpace = binding.spaceID;
        } else if (hlsl::OP::IsDxilOpFuncCallInst(handleCreation, hlsl::OP::OpCode::CreateHandleFromHeap)) {
            DxilInst_CreateHandleFromHeap createHandleFromHeap(handleCreation);
            ret.accessStyle = createHandleFromHeap.get_samplerHeap_val()
                ? AccessStyle::SamplerFromDescriptorHeap : AccessStyle::ResourceFromDescriptorHeap;
            ret.dynamicallyBoundIndex = createHandleFromHeap.get_index();
  
            ret.registerType = RegisterTypeFromResourceClass(properties.getResourceClass());
  
            DynamicResourceBinding drb{};
            drb.HeapIsSampler = createHandleFromHeap.get_samplerHeap_val();
            drb.HeapIndex = -1;
            drb.Name = "ShaderNameTodo";
            if (auto * constInt = dyn_cast<ConstantInt>(createHandleFromHeap.get_index()))
            {
                drb.HeapIndex = constInt->getLimitedValue();
            }
            m_dynamicResourceBindings.emplace_back(std::move(drb));
  
            return ret;
        } else {
            DXASSERT_NOMSG(false);
        }
      }
  }

  return ret;
}

bool DxilShaderAccessTracking::runOnModule(Module &M) {
  // This pass adds instrumentation for shader access to resources

  DxilModule &DM = M.GetOrCreateDxilModule();
  LLVMContext &Ctx = M.getContext();
  OP *HlslOP = DM.GetOP();

  bool Modified = false;

  if (m_CheckForDynamicIndexing) {

    bool FoundDynamicIndexing = false;

    auto CreateHandleFn =
        HlslOP->GetOpFunc(DXIL::OpCode::CreateHandle, Type::getVoidTy(Ctx));
    for (auto FI = CreateHandleFn->user_begin(); FI != CreateHandleFn->user_end();) {
      auto *FunctionUser = *FI++;
      auto instruction = cast<Instruction>(FunctionUser);
      Value *index = instruction->getOperand(kCreateHandleResIndexOpIdx);
      if (!isa<Constant>(index)) {
        FoundDynamicIndexing = true;
        break;
      }
    }

    auto CreateHandleFromBindingFn =
        HlslOP->GetOpFunc(DXIL::OpCode::CreateHandleFromBinding, Type::getVoidTy(Ctx));
    for (auto FI = CreateHandleFromBindingFn->user_begin(); FI != CreateHandleFromBindingFn->user_end();) {
      auto * FunctionUser = *FI++;
      auto instruction = cast<Instruction>(FunctionUser);
      Value *index = instruction->getOperand(kCreateHandleFromBindingResIndexOpIdx);
      if (!isa<Constant>(index)) {
        FoundDynamicIndexing = true;
        break;
      }
    }

    auto CreateHandleFromHeapFn = HlslOP->GetOpFunc(
        DXIL::OpCode::CreateHandleFromHeap, Type::getVoidTy(Ctx));
    for (auto FI = CreateHandleFromHeapFn->user_begin();
         FI != CreateHandleFromHeapFn->user_end();) {
      auto *FunctionUser = *FI++;
      auto instruction = cast<Instruction>(FunctionUser);
      Value *index = instruction->getOperand(kCreateHandleFromHeapHeapIndexOpIdx);
      if (!isa<Constant>(index)) {
        FoundDynamicIndexing = true;
        break;
      }
    }

    if (FoundDynamicIndexing) {
      if (OSOverride != nullptr) {
        formatted_raw_ostream FOS(*OSOverride);
        FOS << "FoundDynamicIndexing";
      }
    }
  } else {
    {
      if (DM.m_ShaderFlags.GetForceEarlyDepthStencil()) {
        if (OSOverride != nullptr) {
          formatted_raw_ostream FOS(*OSOverride);
          FOS << "ShouldAssumeDsvAccess";
        }
      }
      int uavRegId = 0;
      for (llvm::Function &F : M.functions()) {
        if (!F.getBasicBlockList().empty()) {
          IRBuilder<> Builder(F.getEntryBlock().getFirstInsertionPt());

          m_FunctionToUAVHandle[&F] = PIXPassHelpers::CreateUAV(DM, Builder, uavRegId++, "PIX_CountUAV_Handle");
          auto const* shaderModel = DM.GetShaderModel();
          auto shaderKind = shaderModel->GetKind();
          OP *HlslOP = DM.GetOP();
          for (int accessStyle = static_cast<int>(ResourceAccessStyle::None);
              accessStyle < static_cast<int>(ResourceAccessStyle::EndOfEnum);
              ++accessStyle)
          {
              ResourceAccessStyle style = static_cast<ResourceAccessStyle>(accessStyle);
              m_FunctionToEncodedAccess[&F][style] =
                  HlslOP->GetU32Const(EncodeShaderModel(shaderKind) |
                      EncodeAccess(style));
          }
        }
      }
      DM.ReEmitDxilResources();
    }

    for (llvm::Function &F : M.functions()) {
      // Only used DXIL intrinsics:
      if (!F.isDeclaration() || F.isIntrinsic() || F.use_empty() ||
          !OP::IsDxilOpFunc(&F))
        continue;

      // Gather handle parameter indices, if any
      FunctionType *fnTy =
          cast<FunctionType>(F.getType()->getPointerElementType());
      SmallVector<unsigned, 4> handleParams;
      for (unsigned iParam = 1; iParam < fnTy->getFunctionNumParams();
           ++iParam) {
        if (fnTy->getParamType(iParam) == HlslOP->GetHandleType())
          handleParams.push_back(iParam);
      }
      if (handleParams.empty())
        continue;

      auto FunctionUses = F.uses();
      for (auto FI = FunctionUses.begin(); FI != FunctionUses.end();) {
        auto &FunctionUse = *FI++;
        auto FunctionUser = FunctionUse.getUser();
        auto Call = cast<CallInst>(FunctionUser);
        auto opCode = OP::GetDxilOpFuncCallInst(Call);

        // Base Read/Write on function attribute - should match for all normal
        // resource operations
        ShaderAccessFlags readWrite = ShaderAccessFlags::Write;
        if (OP::GetMemAccessAttr(opCode) == llvm::Attribute::AttrKind::ReadOnly)
          readWrite = ShaderAccessFlags::Read;

        // Special cases
        switch (opCode) {
        case DXIL::OpCode::GetDimensions:
          // readWrite = ShaderAccessFlags::DescriptorRead;  // TODO: Support
          // GetDimensions
          continue;
        case DXIL::OpCode::BufferUpdateCounter:
          readWrite = ShaderAccessFlags::Counter;
          break;
        case DXIL::OpCode::TraceRay:
          // Read of AccelerationStructure; doesn't match function attribute
          // readWrite = ShaderAccessFlags::Read;  // TODO: Support
          continue;
        case DXIL::OpCode::RayQuery_TraceRayInline: {
          // Read of AccelerationStructure; doesn't match function attribute
          auto res = GetResourceFromHandle(Call->getArgOperand(2), DM);
          if (EmitResourceAccess(
            DM,
            res, 
            Call, 
            HlslOP, 
            Ctx,
            ShaderAccessFlags::Read)) 
          {
            Modified = true;
          }
        }
          continue;
        default:
          break;
        }

        for (unsigned iParam : handleParams) {
          auto res = GetResourceFromHandle(Call->getArgOperand(iParam), DM);
          if (res.accessStyle == AccessStyle::None) {
            continue;
          }
          // Don't instrument the accesses to the UAV that we just added
          if (res.RegisterSpace  == -2) {
            break;
          }
          if (EmitResourceAccess(DM, res, Call, HlslOP, Ctx, readWrite)) {
            Modified = true;
          }
          // Remaining resources are DescriptorRead.
          readWrite = ShaderAccessFlags::DescriptorRead;
        }
      }
    }

    if (OSOverride != nullptr) {
      formatted_raw_ostream FOS(*OSOverride);
      FOS << "DynamicallyIndexedBindPoints=";
      for (auto const &bp : m_DynamicallyIndexedBindPoints) {
        FOS << EncodeRegisterType(bp.Type) << bp.Space << ':' << bp.Index
            << ';';
      }
      FOS << ".";

      // todo: this will reflect dynamic resource names when the metadata exists
      FOS << "DynamicallyBoundResources=";
      for (auto const &drb : m_dynamicResourceBindings) {
        FOS << (drb.HeapIsSampler ? 'S' : 'R') << drb.HeapIndex << ';';
      }
      FOS << ".";
    }
  }

  return Modified;
}

char DxilShaderAccessTracking::ID = 0;

ModulePass *llvm::createDxilShaderAccessTrackingPass() {
  return new DxilShaderAccessTracking();
}

INITIALIZE_PASS(DxilShaderAccessTracking,
                "hlsl-dxil-pix-shader-access-instrumentation",
                "HLSL DXIL shader access tracking for PIX", false, false)
