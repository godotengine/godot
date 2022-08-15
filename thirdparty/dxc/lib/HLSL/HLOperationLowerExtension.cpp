///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLOperationLowerExtension.cpp                                             //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/HLOperationLowerExtension.h"

#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/HLSL/HLModule.h"
#include "dxc/HLSL/HLOperationLower.h"
#include "dxc/HLSL/HLOperations.h"
#include "dxc/HlslIntrinsicOp.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/ADT/SmallString.h"

using namespace llvm;
using namespace hlsl;

LLVM_ATTRIBUTE_NORETURN static void ThrowExtensionError(StringRef Details)
{
    std::string Msg = (Twine("Error in dxc extension api: ") + Details).str();
    throw hlsl::Exception(DXC_E_EXTENSION_ERROR, Msg);
}

// The lowering strategy format is a string that matches the following regex:
//
//      [a-z](:(?P<ExtraStrategyInfo>.+))?$
//
// The first character indicates the strategy with an optional : followed by
// additional lowering information specific to that strategy.
//
ExtensionLowering::Strategy ExtensionLowering::GetStrategy(StringRef strategy) {
  if (strategy.size() < 1)
    return Strategy::Unknown;

  switch (strategy[0]) {
    case 'n': return Strategy::NoTranslation;
    case 'r': return Strategy::Replicate;
    case 'p': return Strategy::Pack;
    case 'm': return Strategy::Resource;
    case 'd': return Strategy::Dxil;
    default: break;
  }
  return Strategy::Unknown;
}

llvm::StringRef ExtensionLowering::GetStrategyName(Strategy strategy) {
  switch (strategy) {
    case Strategy::NoTranslation: return "n";
    case Strategy::Replicate:     return "r";
    case Strategy::Pack:          return "p";
    case Strategy::Resource:      return "m"; // m for resource method
    case Strategy::Dxil:          return "d";
    default: break;
  }
  return "?";
}

static std::string ParseExtraStrategyInfo(StringRef strategy)
{
    std::pair<StringRef, StringRef> SplitInfo = strategy.split(":");
    return SplitInfo.second;
}

ExtensionLowering::ExtensionLowering(Strategy strategy, HLSLExtensionsCodegenHelper *helper, OP& hlslOp,  HLResourceLookup &hlResourceLookup)
  : m_strategy(strategy), m_helper(helper), m_hlslOp(hlslOp), m_hlResourceLookup(hlResourceLookup)
  {}

ExtensionLowering::ExtensionLowering(StringRef strategy, HLSLExtensionsCodegenHelper *helper, OP& hlslOp, HLResourceLookup &hlResourceLookup)
  : ExtensionLowering(GetStrategy(strategy), helper, hlslOp, hlResourceLookup)
  {
    m_extraStrategyInfo = ParseExtraStrategyInfo(strategy);
  }

llvm::Value *ExtensionLowering::Translate(llvm::CallInst *CI) {
  switch (m_strategy) {
  case Strategy::NoTranslation: return NoTranslation(CI);
  case Strategy::Replicate:     return Replicate(CI);
  case Strategy::Pack:          return Pack(CI);
  case Strategy::Resource:      return Resource(CI);
  case Strategy::Dxil:          return Dxil(CI);
  default: break;
  }
  return Unknown(CI);
}

llvm::Value *ExtensionLowering::Unknown(CallInst *CI) {
  assert(false && "unknown translation strategy");
  return nullptr;
}

// Interface to describe how to translate types from HL-dxil to dxil.
class FunctionTypeTranslator {
public:
  // Arguments can be exploded into multiple copies of the same type.
  // For example a <2 x i32> could become { i32, 2 } if the vector
  // is expanded in place or { i32, 1 } if the call is replicated.
  struct ArgumentType {
    Type *type;
    int  count;

    ArgumentType(Type *ty, int cnt = 1) : type(ty), count(cnt) {}
  };

  virtual ~FunctionTypeTranslator() {}

  virtual Type *TranslateReturnType(CallInst *CI) = 0;
  virtual ArgumentType TranslateArgumentType(Value *OrigArg) = 0;
};

// Class to create the new function with the translated types for low-level dxil.
class FunctionTranslator {
public:
  template <typename TypeTranslator>
  static Function *GetLoweredFunction(CallInst *CI, ExtensionLowering &lower) {
    TypeTranslator typeTranslator;
    return GetLoweredFunction(typeTranslator, CI, lower);
  }
  
  static Function *GetLoweredFunction(FunctionTypeTranslator &typeTranslator, CallInst *CI, ExtensionLowering &lower) {
    FunctionTranslator translator(typeTranslator, lower);
    return translator.GetLoweredFunction(CI);
  }

  virtual ~FunctionTranslator() {}

protected:
  FunctionTypeTranslator &m_typeTranslator;
  ExtensionLowering &m_lower;

  FunctionTranslator(FunctionTypeTranslator &typeTranslator, ExtensionLowering &lower)
    : m_typeTranslator(typeTranslator)
    , m_lower(lower)
  {}

  Function *GetLoweredFunction(CallInst *CI) {
    // Ge the return type of replicated function.
    Type *RetTy = m_typeTranslator.TranslateReturnType(CI);
    if (!RetTy)
      return nullptr;

    // Get the Function type for replicated function.
    FunctionType *FTy = GetFunctionType(CI, RetTy);
    if (!FTy)
      return nullptr;

    // Create a new function that will be the replicated call.
    AttributeSet attributes = GetAttributeSet(CI);
    std::string name = m_lower.GetExtensionName(CI);
    return cast<Function>(CI->getModule()->getOrInsertFunction(name, FTy, attributes));
  }

  virtual FunctionType *GetFunctionType(CallInst *CI, Type *RetTy) {
    // Create a new function type with the translated argument.
    SmallVector<Type *, 10> ParamTypes;
    ParamTypes.reserve(CI->getNumArgOperands());
    for (unsigned i = 0; i < CI->getNumArgOperands(); ++i) {
      Value *OrigArg = CI->getArgOperand(i);
      FunctionTypeTranslator::ArgumentType newArgType = m_typeTranslator.TranslateArgumentType(OrigArg);
      for (int i = 0; i < newArgType.count; ++i) {
        ParamTypes.push_back(newArgType.type);
      }
    }

    const bool IsVarArg = false;
    return FunctionType::get(RetTy, ParamTypes, IsVarArg);
  }

  AttributeSet GetAttributeSet(CallInst *CI) {
    Function *F = CI->getCalledFunction();
    AttributeSet attributes;
    auto copyAttribute = [=, &attributes](Attribute::AttrKind a) {
      if (F->hasFnAttribute(a)) {
        attributes = attributes.addAttribute(CI->getContext(), AttributeSet::FunctionIndex, a);
      }
    };
    copyAttribute(Attribute::AttrKind::ReadOnly);
    copyAttribute(Attribute::AttrKind::ReadNone);
    copyAttribute(Attribute::AttrKind::ArgMemOnly);
    copyAttribute(Attribute::AttrKind::NoUnwind);

    return attributes;
  }
};

///////////////////////////////////////////////////////////////////////////////
// NoTranslation Lowering.
class NoTranslationTypeTranslator : public FunctionTypeTranslator {
  virtual Type *TranslateReturnType(CallInst *CI) override {
    return CI->getType();
  }
  virtual ArgumentType TranslateArgumentType(Value *OrigArg) override {
    return ArgumentType(OrigArg->getType());
  }
};

llvm::Value *ExtensionLowering::NoTranslation(CallInst *CI) {
  Function *NoTranslationFunction = FunctionTranslator::GetLoweredFunction<NoTranslationTypeTranslator>(CI, *this);
  if (!NoTranslationFunction)
    return nullptr;

  IRBuilder<> builder(CI);
  SmallVector<Value *, 8> args(CI->arg_operands().begin(), CI->arg_operands().end());
  return builder.CreateCall(NoTranslationFunction, args);
}

///////////////////////////////////////////////////////////////////////////////
// Replicated Lowering.
enum {
  NO_COMMON_VECTOR_SIZE = 0x0,
};
// Find the vector size that will be used for replication.
// The function call will be replicated once for each element of the vector
// size.
static unsigned GetReplicatedVectorSize(llvm::CallInst *CI) {
  unsigned commonVectorSize = NO_COMMON_VECTOR_SIZE;
  Type *RetTy = CI->getType();
  if (RetTy->isVectorTy())
    commonVectorSize = RetTy->getVectorNumElements();
  for (unsigned i = 0; i < CI->getNumArgOperands(); ++i) {
    Type *Ty = CI->getArgOperand(i)->getType();
    if (Ty->isVectorTy()) {
      unsigned vectorSize = Ty->getVectorNumElements();
      if (commonVectorSize != NO_COMMON_VECTOR_SIZE && commonVectorSize != vectorSize) {
        // Inconsistent vector sizes; need a different strategy.
        return NO_COMMON_VECTOR_SIZE;
      }
      commonVectorSize = vectorSize;
    }
  }

  return commonVectorSize;
}

class ReplicatedFunctionTypeTranslator : public FunctionTypeTranslator {
  virtual Type *TranslateReturnType(CallInst *CI) override {
    unsigned commonVectorSize = GetReplicatedVectorSize(CI);
    if (commonVectorSize == NO_COMMON_VECTOR_SIZE)
      return nullptr;

    // Result should be vector or void.
    Type *RetTy = CI->getType();
    if (!RetTy->isVoidTy() && !RetTy->isVectorTy())
      return nullptr;

    if (RetTy->isVectorTy()) {
      RetTy = RetTy->getVectorElementType();
    }

    return RetTy;
  }

  virtual ArgumentType TranslateArgumentType(Value *OrigArg) override {
    Type *Ty = OrigArg->getType();
    if (Ty->isVectorTy()) {
      Ty = Ty->getVectorElementType();
    }

    return ArgumentType(Ty);
  }

};

class ReplicateCall {
public:
  ReplicateCall(CallInst *CI, Function &ReplicatedFunction)
    : m_CI(CI)
    , m_ReplicatedFunction(ReplicatedFunction)
    , m_numReplicatedCalls(GetReplicatedVectorSize(CI))
    , m_ScalarizeArgIdx()
    , m_Args(CI->getNumArgOperands())
    , m_ReplicatedCalls(m_numReplicatedCalls)
    , m_Builder(CI)
  {
    assert(m_numReplicatedCalls != NO_COMMON_VECTOR_SIZE);
  }

  Value *Generate() {
    CollectReplicatedArguments();
    CreateReplicatedCalls();
    Value *retVal = GetReturnValue();
    return retVal;
  }

private:
  CallInst *m_CI;
  Function &m_ReplicatedFunction;
  unsigned m_numReplicatedCalls;
  SmallVector<unsigned, 10> m_ScalarizeArgIdx;
  SmallVector<Value *, 10> m_Args;
  SmallVector<Value *, 10> m_ReplicatedCalls;
  IRBuilder<> m_Builder;

  // Collect replicated arguments.
  // For non-vector arguments we can add them to the args list directly.
  // These args will be shared by each replicated call. For the vector
  // arguments we remember the position it will go in the argument list.
  // We will fill in the vector args below when we replicate the call
  // (once for each vector lane).
  void CollectReplicatedArguments() {
    for (unsigned i = 0; i < m_CI->getNumArgOperands(); ++i) {
      Type *Ty = m_CI->getArgOperand(i)->getType();
      if (Ty->isVectorTy()) {
        m_ScalarizeArgIdx.push_back(i);
      }
      else {
        m_Args[i] = m_CI->getArgOperand(i);
      }
    }
  }

  // Create replicated calls.
  // Replicate the call once for each element of the replicated vector size.
  void CreateReplicatedCalls() {
    for (unsigned vecIdx = 0; vecIdx < m_numReplicatedCalls; vecIdx++) {
      for (unsigned i = 0, e = m_ScalarizeArgIdx.size(); i < e; ++i) {
        unsigned argIdx = m_ScalarizeArgIdx[i];
        Value *arg = m_CI->getArgOperand(argIdx);
        m_Args[argIdx] = m_Builder.CreateExtractElement(arg, vecIdx);
      }
      Value *EltOP = m_Builder.CreateCall(&m_ReplicatedFunction, m_Args);
      m_ReplicatedCalls[vecIdx] = EltOP;
    }
  }

  // Get the final replicated value.
  // If the function is a void type then return (arbitrarily) the first call.
  // We do not return nullptr because that indicates a failure to replicate.
  // If the function is a vector type then aggregate all of the replicated
  // call values into a new vector.
  Value *GetReturnValue() {
    if (m_CI->getType()->isVoidTy())
      return m_ReplicatedCalls.back();

    Value *retVal = llvm::UndefValue::get(m_CI->getType());
    for (unsigned i = 0; i < m_ReplicatedCalls.size(); ++i)
      retVal = m_Builder.CreateInsertElement(retVal, m_ReplicatedCalls[i], i);

    return retVal;
  }
};

// Translate the HL call by replicating the call for each vector element.
//
// For example,
//
//    <2xi32> %r = call @ext.foo(i32 %op, <2xi32> %v)
//    ==>
//    %r.1 = call @ext.foo.s(i32 %op, i32 %v.1)
//    %r.2 = call @ext.foo.s(i32 %op, i32 %v.2)
//    <2xi32> %r.v.1 = insertelement %r.1, 0, <2xi32> undef
//    <2xi32> %r.v.2 = insertelement %r.2, 1, %r.v.1
//
// You can then RAWU %r with %r.v.2. The RAWU is not done by the translate function.
Value *ExtensionLowering::Replicate(CallInst *CI) {
  Function *ReplicatedFunction = FunctionTranslator::GetLoweredFunction<ReplicatedFunctionTypeTranslator>(CI, *this);
  if (!ReplicatedFunction)
    return NoTranslation(CI);

  ReplicateCall replicate(CI, *ReplicatedFunction);
  return replicate.Generate();
}

///////////////////////////////////////////////////////////////////////////////
// Packed Lowering.
class PackCall {
public:
  PackCall(CallInst *CI, Function &PackedFunction)
    : m_CI(CI)
    , m_packedFunction(PackedFunction)
    , m_builder(CI)
  {}

  Value *Generate() {
    SmallVector<Value *, 10> args;
    PackArgs(args);
    Value *result = CreateCall(args);
    return UnpackResult(result);
  }
  
  static StructType *ConvertVectorTypeToStructType(Type *vecTy) {
    assert(vecTy->isVectorTy());
    Type *elementTy = vecTy->getVectorElementType();
    unsigned numElements = vecTy->getVectorNumElements();
    SmallVector<Type *, 4> elements;
    for (unsigned i = 0; i < numElements; ++i)
      elements.push_back(elementTy);

    return StructType::get(vecTy->getContext(), elements);
  }

private:
  CallInst *m_CI;
  Function &m_packedFunction;
  IRBuilder<> m_builder;

  void PackArgs(SmallVectorImpl<Value*> &args) {
    args.clear();
    for (Value *arg : m_CI->arg_operands()) {
      if (arg->getType()->isVectorTy())
        arg = PackVectorIntoStruct(m_builder, arg);
      args.push_back(arg);
    }
  }

  Value *CreateCall(const SmallVectorImpl<Value*> &args) {
    return m_builder.CreateCall(&m_packedFunction, args);
  }

  Value *UnpackResult(Value *result) {
    if (result->getType()->isStructTy()) {
      result = PackStructIntoVector(m_builder, result);
    }
    return result;
  }

  static VectorType *ConvertStructTypeToVectorType(Type *structTy) {
    assert(structTy->isStructTy());
    return VectorType::get(structTy->getStructElementType(0), structTy->getStructNumElements());
  }

  static Value *PackVectorIntoStruct(IRBuilder<> &builder, Value *vec) {
    StructType *structTy = ConvertVectorTypeToStructType(vec->getType());
    Value *packed = UndefValue::get(structTy);

    unsigned numElements = structTy->getStructNumElements();
    for (unsigned i = 0; i < numElements; ++i) {
      Value *element = builder.CreateExtractElement(vec, i);
      packed = builder.CreateInsertValue(packed, element, { i });
    }

    return packed;
  }

  static Value *PackStructIntoVector(IRBuilder<> &builder, Value *strukt) {
    Type *vecTy = ConvertStructTypeToVectorType(strukt->getType());
    Value *packed = UndefValue::get(vecTy);

    unsigned numElements = vecTy->getVectorNumElements();
    for (unsigned i = 0; i < numElements; ++i) {
      Value *element = builder.CreateExtractValue(strukt, i);
      packed = builder.CreateInsertElement(packed, element, i);
    }

    return packed;
  }
};

class PackedFunctionTypeTranslator : public FunctionTypeTranslator {
  virtual Type *TranslateReturnType(CallInst *CI) override {
    return TranslateIfVector(CI->getType());
  }
  virtual ArgumentType TranslateArgumentType(Value *OrigArg) override {
    return ArgumentType(TranslateIfVector(OrigArg->getType()));
  }

  Type *TranslateIfVector(Type *ty) {
    if (ty->isVectorTy())
      ty = PackCall::ConvertVectorTypeToStructType(ty);
    return ty;
  }
};

Value *ExtensionLowering::Pack(CallInst *CI) {
  Function *PackedFunction = FunctionTranslator::GetLoweredFunction<PackedFunctionTypeTranslator>(CI, *this);
  if (!PackedFunction)
    return NoTranslation(CI);

  PackCall pack(CI, *PackedFunction);
  Value *result = pack.Generate();
  return result;
}

///////////////////////////////////////////////////////////////////////////////
// Resource Lowering.

// Modify a call to a resouce method. Makes the following transformation:
//
// 1. Convert non-void return value to dx.types.ResRet.
// 2. Expand vectors in place as separate arguments.
//
// Example
// -----------------------------------------------------------------------------
//
//  %0 = call <2 x float> MyBufferOp(i32 138, %class.Buffer %3, <2 x i32> <1 , 2> )
//  %r = call %dx.types.ResRet.f32 MyBufferOp(i32 138, %dx.types.Handle %buf, i32 1, i32 2 )
//  %x = extractvalue %r, 0
//  %y = extractvalue %r, 1
//  %v = <2 x float> undef
//  %v.1 = insertelement %v,   %x, 0
//  %v.2 = insertelement %v.1, %y, 1
class ResourceMethodCall {
public:
  ResourceMethodCall(CallInst *CI)
    : m_CI(CI)
    , m_builder(CI)
  { }

  virtual ~ResourceMethodCall() {}

  virtual Value *Generate(Function *explodedFunction) {
    SmallVector<Value *, 16> args;
    ExplodeArgs(args);
    Value *result = CreateCall(explodedFunction, args);
    result = ConvertResult(result);
    return result;
  }
  
protected:
  CallInst *m_CI;
  IRBuilder<> m_builder;

  void ExplodeArgs(SmallVectorImpl<Value*> &args) {
    for (Value *arg : m_CI->arg_operands()) {
      // vector arg: <N x ty> -> ty, ty, ..., ty (N times)
      if (arg->getType()->isVectorTy()) {
        for (unsigned i = 0; i < arg->getType()->getVectorNumElements(); i++) {
          Value *xarg = m_builder.CreateExtractElement(arg, i);
          args.push_back(xarg);
        }
      }
      // any other value: arg -> arg
      else {
        args.push_back(arg);
      }
    }
  }

  Value *CreateCall(Function *explodedFunction, ArrayRef<Value*> args) {
    return m_builder.CreateCall(explodedFunction, args);
  }

  Value *ConvertResult(Value *result) {
    Type *origRetTy = m_CI->getType();
    if (origRetTy->isVoidTy())
      return ConvertVoidResult(result);
    else if (origRetTy->isVectorTy())
      return ConvertVectorResult(origRetTy, result);
    else
      return ConvertScalarResult(origRetTy, result);
  }

  // Void result does not need any conversion.
  Value *ConvertVoidResult(Value *result) {
    return result;
  }

  // Vector result will be populated with the elements from the resource return.
  Value *ConvertVectorResult(Type *origRetTy, Value *result) {
    Type *resourceRetTy = result->getType();
    assert(origRetTy->isVectorTy());
    assert(resourceRetTy->isStructTy() && "expected resource return type to be a struct");
    
    const unsigned vectorSize = origRetTy->getVectorNumElements();
    const unsigned structSize = resourceRetTy->getStructNumElements();
    const unsigned size = std::min(vectorSize, structSize);
    assert(vectorSize < structSize);
    
    // Copy resource struct elements to vector.
    Value *vector = UndefValue::get(origRetTy);
    for (unsigned i = 0; i < size; ++i) {
      Value *element = m_builder.CreateExtractValue(result, { i });
      vector = m_builder.CreateInsertElement(vector, element, i);
    }

    return vector;
  }

  // Scalar result will be populated with the first element of the resource return.
  Value *ConvertScalarResult(Type *origRetTy, Value *result) {
    assert(origRetTy->isSingleValueType());
    return m_builder.CreateExtractValue(result, { 0 });
  }

};

// Translate function return and argument types for resource method lowering.
class ResourceFunctionTypeTranslator : public FunctionTypeTranslator {
public:
  ResourceFunctionTypeTranslator(OP &hlslOp) : m_hlslOp(hlslOp) {}

  // Translate return type as follows:
  //
  // void     -> void
  // <N x ty> -> dx.types.ResRet.ty
  //  ty      -> dx.types.ResRet.ty
  virtual Type *TranslateReturnType(CallInst *CI) override {
    Type *RetTy = CI->getType();
    if (RetTy->isVoidTy())
      return RetTy;
    else if (RetTy->isVectorTy())
      RetTy = RetTy->getVectorElementType();

    return m_hlslOp.GetResRetType(RetTy);
  }
  
  // Translate argument type as follows:
  //
  // resource -> dx.types.Handle
  // <N x ty> -> { ty, N }
  //  ty      -> { ty, 1 }
  virtual ArgumentType TranslateArgumentType(Value *OrigArg) override {
    int count = 1;
    Type *ty = OrigArg->getType();

    if (ty->isVectorTy()) {
      count = ty->getVectorNumElements();
      ty = ty->getVectorElementType();
    }

    return ArgumentType(ty, count);
  }

private:
  OP& m_hlslOp;
};

Value *ExtensionLowering::Resource(CallInst *CI) {
  // Extra strategy info overrides the default lowering for resource methods.
  if (!m_extraStrategyInfo.empty())
  {
    return CustomResource(CI);
  }

  ResourceFunctionTypeTranslator resourceTypeTranslator(m_hlslOp);
  Function *resourceFunction = FunctionTranslator::GetLoweredFunction(resourceTypeTranslator, CI, *this);
  if (!resourceFunction)
    return NoTranslation(CI);

  ResourceMethodCall explode(CI);
  Value *result = explode.Generate(resourceFunction);
  return result;
}

// This class handles the core logic for custom lowering of resource
// method intrinsics. The goal is to allow resource extension intrinsics
// to be handled the same way as the core hlsl resource intrinsics.
//
// Specifically, we want to support:
//
//  1. Multiple hlsl overloads map to a single dxil intrinsic
//  2. The hlsl overloads can take different parameters for a given resource type
//  3. The hlsl overloads are not consistent across different resource types 
//
// To achieve these goals we need a more complex mechanism for describing how
// to translate the high-level arguments to arguments for a dxil function.
// The custom lowering info describes this lowering using the following format.
//
// [Custom Lowering Info Format]
// A json string encoding a map where each key is either a specific resource type or
// the keyword "default" to be used for any other resource. The value is a
// a custom-format string encoding how high-level arguments are mapped to
// dxil intrinsic arguments.
//
// [Argument Translation Format]
// A comma separated string where the number of fields is exactly equal to the number
// of parameters in the target dxil intrinsic. Each field describes how to generate
// the argument for that dxil intrinsic parameter. It has the following format where
// the hl_arg_index is mandatory, but the other two parts are optional.
//
//      <hl_arg_index>.<vector_index>:<optional_type_info>
//
// The format is precisely described by the following regular expression:
//
//      (?P<hl_arg_index>[-0-9]+)(.(?P<vector_index>[-0-9]+))?(:(?P<optional_type_info>\?i32|\?i16|\?i8|\?float|\?half))?$
//
// Example
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Say we want to define the MyTextureOp extension with the following overloads:
//
// Texture1D
//  MyTextureOp(uint addr, uint offset)
//  MyTextureOp(uint addr, uint offset, uint val)
//
// Texture2D
//  MyTextureOp(uint2 addr, uint2 val)
//  
// And a dxil intrinsic defined as follows
//  @MyTextureOp(i32 opcode,  %dx.types.Handle handle, i32 addr0, i32 addr1, i32 offset, i32 val0, i32 val1)
//
// Then we would define the lowering info json as follows
//
//  {
//      "default"   : "0, 1, 2.0, 2.1,  3     , 4.0:?i32, 4.1:?i32"
//      "Texture2D" : "0, 1, 2.0, 2.1, -1:?i32, 3.0     , 3.1\"
//  }
//
//
//  This would produce the following lowerings (assuming the MyTextureOp opcode is 17)
//
//  hlsl: Texture1D.MyTextureOp(a, b)
//  hl:   @MyTextureOp(17, handle, a, b)
//  dxil: @MyTextureOp(17, handle, a, undef, b, undef, undef)
//
//  hlsl: Texture1D.MyTextureOp(a, b, c)
//  hl:   @MyTextureOp(17, handle, a, b, c)
//  dxil: @MyTextureOp(17, handle, a, undef, b, c, undef)
//
//  hlsl: Texture2D.MyTextureOp(a, c)
//  hl:   @MyTextureOp(17, handle, a, c)
//  dxil: @MyTextureOp(17, handle, a.x, a.y, undef, c.x, c.y)
//
// 
class CustomResourceLowering
{
public:
    CustomResourceLowering(StringRef LoweringInfo, CallInst *CI, HLResourceLookup &ResourceLookup)
    {
        // Parse lowering info json format.
        std::map<ResourceKindName, std::vector<DxilArgInfo>> LoweringInfoMap =
            ParseLoweringInfo(LoweringInfo, CI->getContext());

        // Lookup resource kind based on handle (first arg after hl opcode)
        enum {RESOURCE_HANDLE_ARG=1};
        const char *pName = nullptr;
        if (!ResourceLookup.GetResourceKindName(CI->getArgOperand(RESOURCE_HANDLE_ARG), &pName))
        {
            ThrowExtensionError("Failed to find resource from handle");
        }
        std::string Name(pName);

        // Select lowering info to use based on resource kind.
        const char *DefaultInfoName = "default";
        std::vector<DxilArgInfo> *pArgInfo = nullptr;
        if (LoweringInfoMap.count(Name))
        {
            pArgInfo = &LoweringInfoMap.at(Name);
        }
        else if (LoweringInfoMap.count(DefaultInfoName))
        {
            pArgInfo = &LoweringInfoMap.at(DefaultInfoName);
        }
        else
        {
            ThrowExtensionError("Unable to find lowering info for resource");
        }
        GenerateLoweredArgs(CI, *pArgInfo);
    }

    const std::vector<Value *> &GetLoweredArgs() const
    {
        return m_LoweredArgs;
    }

private:
    struct OptionalTypeSpec
    {
        const char* TypeName;
        Type *LLVMType;
    };

    // These are the supported optional types for generating dxil parameters
    // that have no matching argument in the high-level intrinsic overload.
    // See [Argument Translation Format] for details.
    void InitOptionalTypes(LLVMContext &Ctx)
    {
        // Table of supported optional types.
        // Keep in sync with m_OptionalTypes small vector size to avoid
        // dynamic allocation.
        OptionalTypeSpec OptionalTypes[] = {
            {"?i32",   Type::getInt32Ty(Ctx)},
            {"?float", Type::getFloatTy(Ctx)},
            {"?half",  Type::getHalfTy(Ctx)},
            {"?i8",    Type::getInt8Ty(Ctx)},
            {"?i16",   Type::getInt16Ty(Ctx)},
        };
        DXASSERT(m_OptionalTypes.empty(), "Init should only be called once");
        m_OptionalTypes.clear();
        m_OptionalTypes.reserve(_countof(OptionalTypes));

        for (const OptionalTypeSpec &T : OptionalTypes)
        {
            m_OptionalTypes.push_back(T);
        }
    }

    Type *ParseOptionalType(StringRef OptionalTypeInfo)
    {
        if (OptionalTypeInfo.empty())
        {
            return nullptr;
        }

        for (OptionalTypeSpec &O : m_OptionalTypes)
        {
            if (OptionalTypeInfo == O.TypeName)
            {
                return O.LLVMType;
            }
        }
            
        ThrowExtensionError("Failed to parse optional type");
    }
    
    // Mapping from high level function arg to dxil function arg.
    //
    // The `HighLevelArgIndex` is the index of the function argument to
    // which this dxil argument maps.
    //
    // If `HasVectorIndex` is true then the `VectorIndex` contains the
    // index of the element in the vector pointed to by HighLevelArgIndex.
    //
    // The `OptionalType` is used to specify types for arguments that are not
    // present in all overloads of the high level function. This lets us
    // map multiple high level functions to a single dxil extension intrinsic.
    //
    struct DxilArgInfo
    {
        unsigned HighLevelArgIndex = 0;
        unsigned VectorIndex = 0;
        bool HasVectorIndex = false;
        Type *OptionalType = nullptr;
    };
    typedef std::string ResourceKindName;

    // Convert the lowering info to a machine-friendly format.
    // Note that we use the YAML parser to parse the JSON since JSON
    // is a subset of YAML (and this llvm has no JSON parser).
    //
    // See [Custom Lowering Info Format] for details.
    std::map<ResourceKindName, std::vector<DxilArgInfo>> ParseLoweringInfo(StringRef LoweringInfo, LLVMContext &Ctx)
    {
        InitOptionalTypes(Ctx);
        std::map<ResourceKindName, std::vector<DxilArgInfo>> LoweringInfoMap;

        SourceMgr SM;
        yaml::Stream YAMLStream(LoweringInfo, SM);

        // Make sure we have a valid json input.
        llvm::yaml::document_iterator I = YAMLStream.begin();
        if (I == YAMLStream.end()) {
            ThrowExtensionError("Found empty resource lowering JSON.");
        }
        llvm::yaml::Node *Root = I->getRoot();
        if (!Root) {
            ThrowExtensionError("Error parsing resource lowering JSON.");
        }

        // Parse the top level map object.
        llvm::yaml::MappingNode *Object = dyn_cast<llvm::yaml::MappingNode>(Root);
        if (!Object) {
            ThrowExtensionError("Expected map in top level of resource lowering JSON.");
        }

        // Parse all key/value pairs from the map.
        for (llvm::yaml::MappingNode::iterator KVI = Object->begin(),
            KVE = Object->end();
            KVI != KVE; ++KVI) 
        {
            // Parse key.
            llvm::yaml::ScalarNode *KeyString =
                dyn_cast_or_null<llvm::yaml::ScalarNode>((*KVI).getKey());
            if (!KeyString) {
                ThrowExtensionError("Expected string as key in resource lowering info JSON map.");
            }
            SmallString<32> KeyStorage;
            StringRef Key = KeyString->getValue(KeyStorage);

            // Parse value.
            llvm::yaml::ScalarNode *ValueString =
                dyn_cast_or_null<llvm::yaml::ScalarNode>((*KVI).getValue());
            if (!ValueString) {
                ThrowExtensionError("Expected string as value in resource lowering info JSON map.");
            }
            SmallString<128> ValueStorage;
            StringRef Value = ValueString->getValue(ValueStorage);

            // Parse dxil arg info from value.
            LoweringInfoMap[Key] = ParseDxilArgInfo(Value, Ctx);
        }

        return LoweringInfoMap;
    }


    // Parse the dxail argument translation info.
    // See [Argument Translation Format] for details.
    std::vector<DxilArgInfo> ParseDxilArgInfo(StringRef ArgSpec, LLVMContext &Ctx)
    {
        std::vector<DxilArgInfo> Args;

        SmallVector<StringRef, 14> Splits;
        ArgSpec.split(Splits, ",");

        for (const StringRef &Split : Splits)
        {
            StringRef Field = Split.trim();
            StringRef HighLevelArgInfo;
            StringRef OptionalTypeInfo;
            std::tie(HighLevelArgInfo, OptionalTypeInfo) = Field.split(":");

            Type *OptionalType = ParseOptionalType(OptionalTypeInfo);

            StringRef HighLevelArgIndex;
            StringRef VectorIndex;
            std::tie(HighLevelArgIndex, VectorIndex) = HighLevelArgInfo.split(".");

            // Parse the arg and vector index.
            // Parse the values as signed integers, but store them as unsigned values to
            // allows using -1 as a shorthand for the max value.
            DxilArgInfo ArgInfo;
            ArgInfo.HighLevelArgIndex = static_cast<unsigned>(std::stoi(HighLevelArgIndex));
            if (!VectorIndex.empty())
            {
                ArgInfo.HasVectorIndex = true;
                ArgInfo.VectorIndex = static_cast<unsigned>(std::stoi(VectorIndex));
            }
            ArgInfo.OptionalType = OptionalType;

            Args.push_back(ArgInfo);
        }

        return Args;
    }

    // Create the dxil args based on custom lowering info.
    void GenerateLoweredArgs(CallInst *CI, const std::vector<DxilArgInfo> &ArgInfoRecords)
    {
        IRBuilder<> builder(CI);
        for (const DxilArgInfo &ArgInfo : ArgInfoRecords)
        {
            // Check to see if we have the corresponding high-level arg in the overload for this call.
            if (ArgInfo.HighLevelArgIndex < CI->getNumArgOperands())
            {
                Value *Arg = CI->getArgOperand(ArgInfo.HighLevelArgIndex);
                if (ArgInfo.HasVectorIndex)
                {
                    // We expect a vector type here, but we handle one special case if not.
                    if (Arg->getType()->isVectorTy())
                    {
                        // We allow multiple high-level overloads to map to a single dxil extension function.
                        // If the vector index is invalid for this specific overload then use an undef
                        // value as a replacement.
                        if (ArgInfo.VectorIndex < Arg->getType()->getVectorNumElements())
                        {
                            Arg = builder.CreateExtractElement(Arg, ArgInfo.VectorIndex);
                        }
                        else
                        {
                            Arg = UndefValue::get(Arg->getType()->getVectorElementType());
                        }
                    }
                    else
                    {
                        // If it is a non-vector type then we replace non-zero vector index with
                        // undef. This is to handle hlsl intrinsic overloading rules that allow
                        // scalars in place of single-element vectors. We assume here that a non-vector
                        // means that a single element vector was already scalarized.
                        // 
                        if (ArgInfo.VectorIndex > 0)
                        {
                            Arg = UndefValue::get(Arg->getType());
                        }
                    }
                }

                m_LoweredArgs.push_back(Arg);
            }
            else if (ArgInfo.OptionalType)
            {
                // If there was no matching high-level arg then we look for the optional
                // arg type specified by the lowering info.
                m_LoweredArgs.push_back(UndefValue::get(ArgInfo.OptionalType));
            }
            else
            { 
                // No way to know how to generate the correc type for this dxil arg.
                ThrowExtensionError("Unable to map high-level arg to dxil arg");
            }
        }
    }
    
    std::vector<Value *> m_LoweredArgs;
    SmallVector<OptionalTypeSpec, 5> m_OptionalTypes;
};

// Boilerplate to reuse exising logic as much as possible.
// We just want to overload GetFunctionType here.
class CustomResourceFunctionTranslator : public FunctionTranslator {
public:
  static Function *GetLoweredFunction(
        const CustomResourceLowering &CustomLowering,
        ResourceFunctionTypeTranslator &typeTranslator,
        CallInst *CI,
        ExtensionLowering &lower
    )
  {
      CustomResourceFunctionTranslator T(CustomLowering, typeTranslator, lower);
      return T.FunctionTranslator::GetLoweredFunction(CI);
  }

private:
    CustomResourceFunctionTranslator(
        const CustomResourceLowering &CustomLowering,
        ResourceFunctionTypeTranslator &typeTranslator,
        ExtensionLowering &lower
    )
        : FunctionTranslator(typeTranslator, lower)
        , m_CustomLowering(CustomLowering)
    {
    }

    virtual FunctionType *GetFunctionType(CallInst *CI, Type *RetTy) override {
        SmallVector<Type *, 16> ParamTypes;
        for (Value *V : m_CustomLowering.GetLoweredArgs())
        {
            ParamTypes.push_back(V->getType());
        }
        const bool IsVarArg = false;
        return FunctionType::get(RetTy, ParamTypes, IsVarArg);
    }

private:
    const CustomResourceLowering &m_CustomLowering;
};

// Boilerplate to reuse exising logic as much as possible.
// We just want to overload Generate here.
class CustomResourceMethodCall : public ResourceMethodCall
{
public:
    CustomResourceMethodCall(CallInst *CI, const CustomResourceLowering &CustomLowering)
        : ResourceMethodCall(CI)
        , m_CustomLowering(CustomLowering)
    {}

    virtual Value *Generate(Function *loweredFunction) override {
        Value *result = CreateCall(loweredFunction, m_CustomLowering.GetLoweredArgs());
        result = ConvertResult(result);
        return result;
    }

private:
    const CustomResourceLowering &m_CustomLowering;
};

// Support custom lowering logic for resource functions.
Value *ExtensionLowering::CustomResource(CallInst *CI) {
    CustomResourceLowering CustomLowering(m_extraStrategyInfo, CI, m_hlResourceLookup);
    ResourceFunctionTypeTranslator ResourceTypeTranslator(m_hlslOp);
    Function *ResourceFunction = CustomResourceFunctionTranslator::GetLoweredFunction(
        CustomLowering,
        ResourceTypeTranslator,
        CI,
        *this
    );
    if (!ResourceFunction)
        return NoTranslation(CI);

    CustomResourceMethodCall custom(CI, CustomLowering);
    Value *Result = custom.Generate(ResourceFunction);
    return Result;
}

///////////////////////////////////////////////////////////////////////////////
// Dxil Lowering.

Value *ExtensionLowering::Dxil(CallInst *CI) {
  // Map the extension opcode to the corresponding dxil opcode.
  unsigned extOpcode = GetHLOpcode(CI);
  OP::OpCode dxilOpcode;
  if (!m_helper->GetDxilOpcode(extOpcode, dxilOpcode))
    return nullptr;

  // Find the dxil function based on the overload type.
  Type *overloadTy = OP::GetOverloadType(dxilOpcode, CI->getCalledFunction());
  Function *F = m_hlslOp.GetOpFunc(dxilOpcode, overloadTy->getScalarType());

  // Update the opcode in the original call so we can just copy it below.
  // We are about to delete this call anyway.
  CI->setOperand(0, m_hlslOp.GetI32Const(static_cast<unsigned>(dxilOpcode)));

  // Create the new call.
  Value *result = nullptr;
  if (overloadTy->isVectorTy()) {
    ReplicateCall replicate(CI, *F);
    result = replicate.Generate();
  }
  else {
    IRBuilder<> builder(CI);
    SmallVector<Value *, 8> args(CI->arg_operands().begin(), CI->arg_operands().end());
    result = builder.CreateCall(F, args);
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
// Computing Extension Names.

// Compute the name to use for the intrinsic function call once it is lowered to dxil.
// First checks to see if we have a custom name from the codegen helper and if not
// chooses a default name based on the lowergin strategy.
class ExtensionName {
public:
  ExtensionName(CallInst *CI, ExtensionLowering::Strategy strategy, HLSLExtensionsCodegenHelper *helper)
    : m_CI(CI)
    , m_strategy(strategy)
    , m_helper(helper)
  {}

  std::string Get() {
    std::string name;
    if (m_helper)
      name = GetCustomExtensionName(m_CI, *m_helper);

    if (!HasCustomExtensionName(name))
      name = GetDefaultCustomExtensionName(m_CI, ExtensionLowering::GetStrategyName(m_strategy));

    return name;
  }

private:
  CallInst *m_CI;
  ExtensionLowering::Strategy m_strategy;
  HLSLExtensionsCodegenHelper *m_helper;

  static std::string GetCustomExtensionName(CallInst *CI, HLSLExtensionsCodegenHelper &helper) {
    unsigned opcode = GetHLOpcode(CI);
    std::string name = helper.GetIntrinsicName(opcode);
    ReplaceOverloadMarkerWithTypeName(name, CI);

    return name;
  }

  static std::string GetDefaultCustomExtensionName(CallInst *CI, StringRef strategyName) {
    return (Twine(CI->getCalledFunction()->getName()) + "." + Twine(strategyName)).str();
  }

  static bool HasCustomExtensionName(const std::string name) {
    return name.size() > 0;
  }

  typedef unsigned OverloadArgIndex;
  static constexpr OverloadArgIndex DefaultOverloadIndex = std::numeric_limits<OverloadArgIndex>::max();

  // Choose the (return value or argument) type that determines the overload type
  // for the intrinsic call.
  // If the overload arg index was explicitly specified (see ParseOverloadArgIndex)
  // then we use that arg to pick the overload name. Otherwise we pick a default
  // where we take the return type as the overload. If the return is void we
  // take the first (non-opcode) argument as the overload type.
  static Type *SelectOverloadSlot(CallInst *CI, OverloadArgIndex ArgIndex) {
   if (ArgIndex != DefaultOverloadIndex)
    {
      return CI->getArgOperand(ArgIndex)->getType();
    }

    Type *ty = CI->getType();
    if (ty->isVoidTy()) {
      if (CI->getNumArgOperands() > 1)
        ty = CI->getArgOperand(1)->getType(); // First non-opcode argument.
    }

    return ty;
  }

  static Type *GetOverloadType(CallInst *CI, OverloadArgIndex ArgIndex) {
    Type *ty = SelectOverloadSlot(CI, ArgIndex);
    if (ty->isVectorTy())
      ty = ty->getVectorElementType();

    return ty;
  }

  static std::string GetTypeName(Type *ty) {
      std::string typeName;
      llvm::raw_string_ostream os(typeName);
      ty->print(os);
      os.flush();
      return typeName;
  }

  static std::string GetOverloadTypeName(CallInst *CI, OverloadArgIndex ArgIndex) {
    Type *ty = GetOverloadType(CI, ArgIndex);
    return GetTypeName(ty);
  }

  // Parse the arg index out of the overload marker (if any).
  //
  // The function names use a $o to indicate that the function is overloaded
  // and we should replace $o with the overload type. The extension name can
  // explicitly set which arg to use for the overload type by adding a colon
  // and a number after the $o (e.g. $o:3 would say the overload type is
  // determined by parameter 3).
  //
  // If we find an arg index after the overload marker we update the size
  // of the marker to include the full parsed string size so that it can
  // be replaced with the selected overload type.
  //
  static OverloadArgIndex ParseOverloadArgIndex(
      const std::string& functionName,
      size_t OverloadMarkerStartIndex,
      size_t *pOverloadMarkerSize)
  {
      assert(OverloadMarkerStartIndex != std::string::npos);
      size_t StartIndex = OverloadMarkerStartIndex + *pOverloadMarkerSize;

      // Check if we have anything after the overload marker to parse.
      if (StartIndex >= functionName.size())
      {
          return DefaultOverloadIndex;
      }

      // Does it start with a ':' ?
      if (functionName[StartIndex] != ':')
      {
          return DefaultOverloadIndex;
      }

      // Skip past the :
      ++StartIndex;

      // Collect all the digits.
      std::string Digits;
      Digits.reserve(functionName.size() - StartIndex);
      for (size_t i = StartIndex; i < functionName.size(); ++i)
      {
          char c = functionName[i];
          if (!isdigit(c))
          {
              break;
          }
          Digits.push_back(c);
      }

      if (Digits.empty())
      {
          return DefaultOverloadIndex;
      }

      *pOverloadMarkerSize = *pOverloadMarkerSize + std::strlen(":") + Digits.size();
      return std::stoi(Digits);
  }

  // Find the occurence of the overload marker $o and replace it the the overload type name.
  static void ReplaceOverloadMarkerWithTypeName(std::string &functionName, CallInst *CI) {
    const char *OverloadMarker = "$o";
    size_t OverloadMarkerLength = 2;

    size_t pos = functionName.find(OverloadMarker);
    if (pos != std::string::npos) {
      OverloadArgIndex ArgIndex = ParseOverloadArgIndex(functionName, pos, &OverloadMarkerLength);
      std::string typeName = GetOverloadTypeName(CI, ArgIndex);
      functionName.replace(pos, OverloadMarkerLength, typeName);
    }
  }
};

std::string ExtensionLowering::GetExtensionName(llvm::CallInst *CI) {
  ExtensionName name(CI, m_strategy, m_helper);
  return name.Get();
}
