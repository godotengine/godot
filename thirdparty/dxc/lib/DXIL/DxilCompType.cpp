///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilCompType.cpp                                                          //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/DXIL/DxilCompType.h"
#include "dxc/Support/Global.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/DerivedTypes.h"

using namespace llvm;


namespace hlsl {

//------------------------------------------------------------------------------
//
// CompType class methods.
//
CompType::CompType() 
: m_Kind(Kind::Invalid) {
}

CompType::CompType(Kind K)
: m_Kind(K) {
  DXASSERT(m_Kind >= Kind::Invalid && m_Kind < Kind::LastEntry, "otherwise the caller passed out-of-range value");
}

CompType::CompType(unsigned int K) : CompType((Kind)K) {}

bool CompType::operator==(const CompType &o) const {
  return m_Kind == o.m_Kind;
}

CompType::Kind CompType::GetKind() const {
  return m_Kind;
}

uint8_t CompType::GetSizeInBits() const {
  switch (m_Kind) {
  case Kind::Invalid:
    return 0;
  case Kind::I1:
    return 1;
  case Kind::SNormF16:
  case Kind::UNormF16:
  case Kind::I16:
  case Kind::F16:
  case Kind::U16:
    return 16;
  case Kind::SNormF32:
  case Kind::UNormF32:
  case Kind::I32:
  case Kind::U32:
  case Kind::F32:
  case Kind::PackedS8x32:
  case Kind::PackedU8x32:
    return 32;
  case Kind::I64:
  case Kind::U64:
  case Kind::SNormF64:
  case Kind::UNormF64:
  case Kind::F64:
    return 64;
  default:
    DXASSERT(false, "invalid type kind");
  }
  return 0;
}

CompType CompType::getInvalid() {
  return CompType();
}

CompType CompType::getF16() {
  return CompType(Kind::F16);
}

CompType CompType::getF32() {
  return CompType(Kind::F32);
}

CompType CompType::getF64() {
  return CompType(Kind::F64);
}

CompType CompType::getI16() {
  return CompType(Kind::I16);
}

CompType CompType::getI32() {
  return CompType(Kind::I32);
}

CompType CompType::getI64() {
  return CompType(Kind::I64);
}

CompType CompType::getU16() {
  return CompType(Kind::U16);
}

CompType CompType::getU32() {
  return CompType(Kind::U32);
}

CompType CompType::getU64() {
  return CompType(Kind::U64);
}

CompType CompType::getI1() {
  return CompType(Kind::I1);
}

CompType CompType::getSNormF16() {
  return CompType(Kind::SNormF16);
}

CompType CompType::getUNormF16() {
  return CompType(Kind::UNormF16);
}

CompType CompType::getSNormF32() {
  return CompType(Kind::SNormF32);
}

CompType CompType::getUNormF32() {
  return CompType(Kind::UNormF32);
}

CompType CompType::getSNormF64() {
  return CompType(Kind::SNormF64);
}

CompType CompType::getUNormF64() {
  return CompType(Kind::UNormF64);
}

bool CompType::IsInvalid() const {
  return m_Kind == Kind::Invalid;
}

bool CompType::IsFloatTy() const {
  return m_Kind == Kind::F16 || m_Kind == Kind::F32 || m_Kind == Kind::F64;
}

bool CompType::IsIntTy() const {
  return IsSIntTy() || IsUIntTy();
}

bool CompType::IsSIntTy() const {
  return m_Kind == Kind::I16 || m_Kind == Kind::I32 || m_Kind == Kind::I64;
}

bool CompType::IsUIntTy() const {
  return m_Kind == Kind::U16 || m_Kind == Kind::U32 || m_Kind == Kind::U64 ||
    m_Kind == Kind::PackedS8x32 || m_Kind == Kind::PackedU8x32;
}

bool CompType::IsBoolTy() const {
  return m_Kind == Kind::I1;
}

bool CompType::IsSNorm() const {
  return m_Kind == Kind::SNormF16 || m_Kind == Kind::SNormF32 || m_Kind == Kind::SNormF64;
}

bool CompType::IsUNorm() const {
  return m_Kind == Kind::UNormF16 || m_Kind == Kind::UNormF32 || m_Kind == Kind::UNormF64;
}

bool CompType::Is64Bit() const {
  switch (m_Kind) {
  case DXIL::ComponentType::F64:
  case DXIL::ComponentType::SNormF64:
  case DXIL::ComponentType::UNormF64:
  case DXIL::ComponentType::I64:
  case DXIL::ComponentType::U64:
    return true;
  default:
    return false;
  }
}

bool CompType::Is16Bit() const {
  switch (m_Kind) {
  case DXIL::ComponentType::F16:
  case DXIL::ComponentType::I16:
  case DXIL::ComponentType::SNormF16:
  case DXIL::ComponentType::UNormF16:
  case DXIL::ComponentType::U16:
    return true;
  default:
    return false;
  }
}

CompType CompType::GetBaseCompType() const {
  switch (m_Kind) {
  case Kind::I1:        return CompType(Kind::I1);
  case Kind::I16:       __fallthrough;
  case Kind::PackedS8x32: __fallthrough;
  case Kind::PackedU8x32: __fallthrough;
  case Kind::I32:       return CompType(Kind::I32);
  case Kind::I64:       return CompType(Kind::I64);
  case Kind::U16:       __fallthrough;
  case Kind::U32:       return CompType(Kind::U32);
  case Kind::U64:       return CompType(Kind::U64);
  case Kind::SNormF16:  __fallthrough;
  case Kind::UNormF16:  __fallthrough;
  case Kind::F16:       __fallthrough;
  case Kind::SNormF32:  __fallthrough;
  case Kind::UNormF32:  __fallthrough;
  case Kind::F32:       return CompType(Kind::F32);
  case Kind::SNormF64:  __fallthrough;
  case Kind::UNormF64:  __fallthrough;
  case Kind::F64:       return CompType(Kind::F64);
  default:
    DXASSERT(false, "invalid type kind");
  }
  return CompType();
}

bool CompType::HasMinPrec() const {
  switch (m_Kind) {
  case Kind::I16:
  case Kind::U16:
  case Kind::F16:
  case Kind::SNormF16:
  case Kind::UNormF16:
    return true;
  case Kind::I1:
  case Kind::PackedS8x32:
  case Kind::PackedU8x32:
  case Kind::I32:
  case Kind::U32:
  case Kind::I64:
  case Kind::U64:
  case Kind::F32:
  case Kind::F64:
  case Kind::SNormF32:
  case Kind::UNormF32:
  case Kind::SNormF64:
  case Kind::UNormF64:
    break;
  default:
    DXASSERT(false, "invalid comp type");
  }
  return false;
}

Type *CompType::GetLLVMType(LLVMContext &Ctx) const {
  switch (m_Kind) {
  case Kind::I1:        return (Type*)Type::getInt1Ty(Ctx);
  case Kind::I16:
  case Kind::U16:       return (Type*)Type::getInt16Ty(Ctx);
  case Kind::PackedS8x32:
  case Kind::PackedU8x32:
  case Kind::I32:
  case Kind::U32:       return (Type*)Type::getInt32Ty(Ctx);
  case Kind::I64:
  case Kind::U64:       return (Type*)Type::getInt64Ty(Ctx);
  case Kind::SNormF16:
  case Kind::UNormF16:
  case Kind::F16:       return Type::getHalfTy(Ctx);
  case Kind::SNormF32:
  case Kind::UNormF32:
  case Kind::F32:       return Type::getFloatTy(Ctx);
  case Kind::SNormF64:
  case Kind::UNormF64:
  case Kind::F64:       return Type::getDoubleTy(Ctx);
  default:
    DXASSERT(false, "invalid type kind");
  }
  return nullptr;
}

PointerType *CompType::GetLLVMPtrType(LLVMContext &Ctx, const unsigned AddrSpace) const {
  switch (m_Kind) {
  case Kind::I1:        return Type::getInt1PtrTy  (Ctx, AddrSpace);
  case Kind::I16:
  case Kind::U16:       return Type::getInt16PtrTy (Ctx, AddrSpace);
  case Kind::PackedS8x32:
  case Kind::PackedU8x32:
  case Kind::I32:
  case Kind::U32:       return Type::getInt32PtrTy (Ctx, AddrSpace);
  case Kind::I64:
  case Kind::U64:       return Type::getInt64PtrTy (Ctx, AddrSpace);
  case Kind::SNormF16:
  case Kind::UNormF16:
  case Kind::F16:       return Type::getHalfPtrTy  (Ctx, AddrSpace);
  case Kind::SNormF32:
  case Kind::UNormF32:
  case Kind::F32:       return Type::getFloatPtrTy (Ctx, AddrSpace);
  case Kind::SNormF64:
  case Kind::UNormF64:
  case Kind::F64:       return Type::getDoublePtrTy(Ctx, AddrSpace);
  default:
    DXASSERT(false, "invalid type kind");
  }
  return nullptr;
}

Type *CompType::GetLLVMBaseType(llvm::LLVMContext &Ctx) const {
  return GetBaseCompType().GetLLVMType(Ctx);
}

CompType CompType::GetCompType(Type *type) {
  LLVMContext &Ctx = type->getContext();
  if (type == Type::getInt1Ty(Ctx))   return CompType(Kind::I1);
  if (type == Type::getInt16Ty(Ctx))  return CompType(Kind::I16);
  if (type == Type::getInt32Ty(Ctx))  return CompType(Kind::I32);
  if (type == Type::getInt64Ty(Ctx))  return CompType(Kind::I64);
  if (type == Type::getHalfTy(Ctx))   return CompType(Kind::F16);
  if (type == Type::getFloatTy(Ctx))  return CompType(Kind::F32);
  if (type == Type::getDoubleTy(Ctx)) return CompType(Kind::F64);

  DXASSERT(false, "invalid type kind");
  return CompType();
}

static const char *s_TypeKindNames[(unsigned)CompType::Kind::LastEntry] = {
  "invalid",
  "i1", "i16", "u16", "i32", "u32", "i64", "u64",
  "f16", "f32", "f64",
  "snorm_f16", "unorm_f16", "snorm_f32", "unorm_f32", "snorm_f64", "unorm_f64",
  "p32i8", "p32u8",
};

const char *CompType::GetName() const {
  return s_TypeKindNames[(unsigned)m_Kind];
}

static const char *s_TypeKindHLSLNames[(unsigned)CompType::Kind::LastEntry] = {
  "unknown",
  "bool", "int16_t", "uint16_t", "int", "uint", "int64_t", "uint64_t",
  "half", "float", "double",
  "snorm_half", "unorm_half", "snorm_float", "unorm_float", "snorm_double", "unorm_double",
  "int8_t_packed", "uint8_t_packed",
};

static const char *s_TypeKindHLSLNamesMinPrecision[(unsigned)CompType::Kind::LastEntry] = {
  "unknown",
  "bool", "min16i", "min16ui", "int", "uint", "int64_t", "uint64_t",
  "min16float", "float", "double",
  "snorm_min16f", "unorm_min16f", "snorm_float", "unorm_float", "snorm_double", "unorm_double",
  "int8_t_packed", "uint8_t_packed",
};

const char *CompType::GetHLSLName(bool MinPrecision) const {
  return MinPrecision ? s_TypeKindHLSLNamesMinPrecision[(unsigned)m_Kind] : s_TypeKindHLSLNames[(unsigned)m_Kind];
}

} // namespace hlsl
