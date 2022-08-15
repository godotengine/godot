///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilOperations.h                                                          //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Implementation of DXIL operation tables.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

namespace llvm {
class LLVMContext;
class Module;
class Type;
class StructType;
class Function;
class Constant;
class Value;
class Instruction;
class CallInst;
}
#include "llvm/IR/Attributes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/MapVector.h"

#include "DxilConstants.h"
#include <unordered_map>

namespace hlsl {

/// Use this utility class to interact with DXIL operations.
class OP {
public:
  using OpCode = DXIL::OpCode;
  using OpCodeClass = DXIL::OpCodeClass;

public:
  OP() = delete;
  OP(llvm::LLVMContext &Ctx, llvm::Module *pModule);

  void RefreshCache();
  void FixOverloadNames();

  llvm::Function *GetOpFunc(OpCode OpCode, llvm::Type *pOverloadType);
  const llvm::SmallMapVector<llvm::Type *, llvm::Function *, 8> &GetOpFuncList(OpCode OpCode) const;
  bool IsDxilOpUsed(OpCode opcode) const;
  void RemoveFunction(llvm::Function *F);
  llvm::LLVMContext &GetCtx() { return m_Ctx; }
  llvm::Type *GetHandleType() const;
  llvm::Type *GetResourcePropertiesType() const;
  llvm::Type *GetResourceBindingType() const;
  llvm::Type *GetDimensionsType() const;
  llvm::Type *GetSamplePosType() const;
  llvm::Type *GetBinaryWithCarryType() const;
  llvm::Type *GetBinaryWithTwoOutputsType() const;
  llvm::Type *GetSplitDoubleType() const;
  llvm::Type *GetFourI32Type() const;
  llvm::Type *GetFourI16Type() const;

  llvm::Type *GetResRetType(llvm::Type *pOverloadType);
  llvm::Type *GetCBufferRetType(llvm::Type *pOverloadType);
  llvm::Type *GetVectorType(unsigned numElements, llvm::Type *pOverloadType);
  bool IsResRetType(llvm::Type *Ty);

  // Try to get the opcode class for a function.
  // Return true and set `opClass` if the given function is a dxil function.
  // Return false if the given function is not a dxil function.
  bool GetOpCodeClass(const llvm::Function *F, OpCodeClass &opClass);

  // To check if operation uses strict precision types
  bool UseMinPrecision();
  // Set if operation uses strict precision types or not.
  void SetMinPrecision(bool bMinPrecision);

  // Get the size of the type for a given layout
  uint64_t GetAllocSizeForType(llvm::Type *Ty);

  // LLVM helpers. Perhaps, move to a separate utility class.
  llvm::Constant *GetI1Const(bool v);
  llvm::Constant *GetI8Const(char v);
  llvm::Constant *GetU8Const(unsigned char v);
  llvm::Constant *GetI16Const(int v);
  llvm::Constant *GetU16Const(unsigned v);
  llvm::Constant *GetI32Const(int v);
  llvm::Constant *GetU32Const(unsigned v);
  llvm::Constant *GetU64Const(unsigned long long v);
  llvm::Constant *GetFloatConst(float v);
  llvm::Constant *GetDoubleConst(double v);

  static llvm::Type *GetOverloadType(OpCode OpCode, llvm::Function *F);
  static OpCode GetDxilOpFuncCallInst(const llvm::Instruction *I);
  static const char *GetOpCodeName(OpCode OpCode);
  static const char *GetAtomicOpName(DXIL::AtomicBinOpCode OpCode);
  static OpCodeClass GetOpCodeClass(OpCode OpCode);
  static const char *GetOpCodeClassName(OpCode OpCode);
  static llvm::Attribute::AttrKind GetMemAccessAttr(OpCode opCode);
  static bool IsOverloadLegal(OpCode OpCode, llvm::Type *pType);
  static bool CheckOpCodeTable();
  static bool IsDxilOpFuncName(llvm::StringRef name);
  static bool IsDxilOpFunc(const llvm::Function *F);
  static bool IsDxilOpFuncCallInst(const llvm::Instruction *I);
  static bool IsDxilOpFuncCallInst(const llvm::Instruction *I, OpCode opcode);
  static bool IsDxilOpWave(OpCode C);
  static bool IsDxilOpGradient(OpCode C);
  static bool IsDxilOpFeedback(OpCode C);
  static bool IsDxilOpTypeName(llvm::StringRef name);
  static bool IsDxilOpType(llvm::StructType *ST);
  static bool IsDupDxilOpType(llvm::StructType *ST);
  static llvm::StructType *GetOriginalDxilOpType(llvm::StructType *ST,
                                                 llvm::Module &M);
  static void GetMinShaderModelAndMask(OpCode C, bool bWithTranslation,
                                       unsigned &major, unsigned &minor,
                                       unsigned &mask);
  static void GetMinShaderModelAndMask(const llvm::CallInst *CI, bool bWithTranslation,
                                       unsigned valMajor, unsigned valMinor,
                                       unsigned &major, unsigned &minor,
                                       unsigned &mask);

private:
  // Per-module properties.
  llvm::LLVMContext &m_Ctx;
  llvm::Module *m_pModule;

  llvm::Type *m_pHandleType;
  llvm::Type *m_pResourcePropertiesType;
  llvm::Type *m_pResourceBindingType;
  llvm::Type *m_pDimensionsType;
  llvm::Type *m_pSamplePosType;
  llvm::Type *m_pBinaryWithCarryType;
  llvm::Type *m_pBinaryWithTwoOutputsType;
  llvm::Type *m_pSplitDoubleType;
  llvm::Type *m_pFourI32Type;
  llvm::Type *m_pFourI16Type;

  DXIL::LowPrecisionMode m_LowPrecisionMode;

  static const unsigned kUserDefineTypeSlot = 9;
  static const unsigned kObjectTypeSlot = 10;
  static const unsigned kNumTypeOverloads = 11; // void, h,f,d, i1, i8,i16,i32,i64, udt, obj

  llvm::Type *m_pResRetType[kNumTypeOverloads];
  llvm::Type *m_pCBufferRetType[kNumTypeOverloads];

  struct OpCodeCacheItem {
    llvm::SmallMapVector<llvm::Type *, llvm::Function *, 8> pOverloads;
  };
  OpCodeCacheItem m_OpCodeClassCache[(unsigned)OpCodeClass::NumOpClasses];
  std::unordered_map<const llvm::Function *, OpCodeClass> m_FunctionToOpClass;
  void UpdateCache(OpCodeClass opClass, llvm::Type * Ty, llvm::Function *F);
private:
  // Static properties.
  struct OpCodeProperty {
    OpCode opCode;
    const char *pOpCodeName;
    OpCodeClass opCodeClass;
    const char *pOpCodeClassName;
    bool bAllowOverload[kNumTypeOverloads];   // void, h,f,d, i1, i8,i16,i32,i64, udt
    llvm::Attribute::AttrKind FuncAttr;
  };
  static const OpCodeProperty m_OpCodeProps[(unsigned)OpCode::NumOpCodes];

  static const char *m_OverloadTypeName[kNumTypeOverloads];
  static const char *m_NamePrefix;
  static const char *m_TypePrefix;
  static const char *m_MatrixTypePrefix;
  static unsigned GetTypeSlot(llvm::Type *pType);
  static const char *GetOverloadTypeName(unsigned TypeSlot);
  static llvm::StringRef GetTypeName(llvm::Type *Ty, std::string &str);
  static llvm::StringRef ConstructOverloadName(llvm::Type *Ty, DXIL::OpCode opCode,
                                               std::string &funcNameStorage);
};

} // namespace hlsl
