//===- PDBExtras.cpp - helper functions and classes for PDBs -----*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/PDBExtras.h"

#include "llvm/ADT/ArrayRef.h"

using namespace llvm;

#define CASE_OUTPUT_ENUM_CLASS_STR(Class, Value, Str, Stream)                  \
  case Class::Value:                                                           \
    Stream << Str;                                                             \
    break;

#define CASE_OUTPUT_ENUM_CLASS_NAME(Class, Value, Stream)                      \
  CASE_OUTPUT_ENUM_CLASS_STR(Class, Value, #Value, Stream)

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDB_VariantType &Type) {
  switch (Type) {
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_VariantType, Bool, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_VariantType, Single, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_VariantType, Double, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_VariantType, Int8, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_VariantType, Int16, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_VariantType, Int32, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_VariantType, Int64, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_VariantType, UInt8, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_VariantType, UInt16, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_VariantType, UInt32, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_VariantType, UInt64, OS)
    default:
      OS << "Unknown";
  }
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDB_CallingConv &Conv) {
  OS << "__";
  switch (Conv) {
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, NearCdecl, "cdecl", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, FarCdecl, "cdecl", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, NearPascal, "pascal", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, FarPascal, "pascal", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, NearFastcall, "fastcall", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, FarFastcall, "fastcall", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, Skipped, "skippedcall", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, NearStdcall, "stdcall", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, FarStdcall, "stdcall", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, NearSyscall, "syscall", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, FarSyscall, "syscall", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, Thiscall, "thiscall", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, MipsCall, "mipscall", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, Generic, "genericcall", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, Alphacall, "alphacall", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, Ppccall, "ppccall", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, SuperHCall, "superhcall", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, Armcall, "armcall", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, AM33call, "am33call", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, Tricall, "tricall", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, Sh5call, "sh5call", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, M32R, "m32rcall", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, Clrcall, "clrcall", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, Inline, "inlinecall", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_CallingConv, NearVectorcall, "vectorcall",
                               OS)
  default:
    OS << "unknowncall";
  }
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDB_DataKind &Data) {
  switch (Data) {
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_DataKind, Unknown, "unknown", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_DataKind, Local, "local", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_DataKind, StaticLocal, "static local", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_DataKind, Param, "param", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_DataKind, ObjectPtr, "this ptr", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_DataKind, FileStatic, "static global", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_DataKind, Global, "global", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_DataKind, Member, "member", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_DataKind, StaticMember, "static member", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_DataKind, Constant, "const", OS)
  }
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDB_RegisterId &Reg) {
  switch (Reg) {
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, AL, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, CL, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, DL, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, BL, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, AH, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, CH, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, DH, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, BH, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, AX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, CX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, DX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, BX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, SP, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, BP, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, SI, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, DI, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, EAX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, ECX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, EDX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, EBX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, ESP, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, EBP, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, ESI, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, EDI, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, ES, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, CS, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, SS, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, DS, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, FS, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, GS, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, IP, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, RAX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, RBX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, RCX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, RDX, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, RSI, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, RDI, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, RBP, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, RSP, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, R8, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, R9, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, R10, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, R11, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, R12, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, R13, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, R14, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_RegisterId, R15, OS)
  default:
    OS << static_cast<int>(Reg);
  }
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDB_LocType &Loc) {
  switch (Loc) {
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_LocType, Static, "static", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_LocType, TLS, "tls", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_LocType, RegRel, "regrel", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_LocType, ThisRel, "thisrel", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_LocType, Enregistered, "register", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_LocType, BitField, "bitfield", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_LocType, Slot, "slot", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_LocType, IlRel, "IL rel", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_LocType, MetaData, "metadata", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_LocType, Constant, "constant", OS)
  default:
    OS << "Unknown";
  }
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDB_ThunkOrdinal &Thunk) {
  switch (Thunk) {
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_ThunkOrdinal, BranchIsland, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_ThunkOrdinal, Pcode, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_ThunkOrdinal, Standard, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_ThunkOrdinal, ThisAdjustor, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_ThunkOrdinal, TrampIncremental, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_ThunkOrdinal, UnknownLoad, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_ThunkOrdinal, Vcall, OS)
  }
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDB_Checksum &Checksum) {
  switch (Checksum) {
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Checksum, None, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Checksum, MD5, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Checksum, SHA1, OS)
  }
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDB_Lang &Lang) {
  switch (Lang) {
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, C, OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_Lang, Cpp, "C++", OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, Fortran, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, Masm, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, Pascal, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, Basic, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, Cobol, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, Link, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, Cvtres, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, Cvtpgd, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, CSharp, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, VB, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, ILAsm, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, Java, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, JScript, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, MSIL, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_Lang, HLSL, OS)
  }
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDB_SymType &Tag) {
  switch (Tag) {
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Exe, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Compiland, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, CompilandDetails, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, CompilandEnv, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Function, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Block, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Data, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Annotation, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Label, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, PublicSymbol, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, UDT, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Enum, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, FunctionSig, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, PointerType, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, ArrayType, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, BuiltinType, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Typedef, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, BaseClass, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Friend, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, FunctionArg, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, FuncDebugStart, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, FuncDebugEnd, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, UsingNamespace, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, VTableShape, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, VTable, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Custom, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Thunk, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, CustomType, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, ManagedType, OS)
    CASE_OUTPUT_ENUM_CLASS_NAME(PDB_SymType, Dimension, OS)
  default:
    OS << "Unknown";
  }
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDB_MemberAccess &Access) {
  switch (Access) {
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_MemberAccess, Public, "public", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_MemberAccess, Protected, "protected", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_MemberAccess, Private, "private", OS)
  }
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDB_UdtType &Type) {
  switch (Type) {
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_UdtType, Class, "class", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_UdtType, Struct, "struct", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_UdtType, Interface, "interface", OS)
    CASE_OUTPUT_ENUM_CLASS_STR(PDB_UdtType, Union, "union", OS)
  }
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const PDB_UniqueId &Id) {
  static const char *Lookup = "0123456789ABCDEF";

  static_assert(sizeof(PDB_UniqueId) == 16, "Expected 16-byte GUID");
  ArrayRef<uint8_t> GuidBytes(reinterpret_cast<const uint8_t*>(&Id), 16);
  OS << "{";
  for (int i=0; i < 16;) {
    uint8_t Byte = GuidBytes[i];
    uint8_t HighNibble = (Byte >> 4) & 0xF;
    uint8_t LowNibble = Byte & 0xF;
    OS << Lookup[HighNibble] << Lookup[LowNibble];
    ++i;
    if (i>=4 && i<=10 && i%2==0)
      OS << "-";
  }
  OS << "}";
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const Variant &Value) {
  switch (Value.Type) {
    case PDB_VariantType::Bool:
      OS << (Value.Bool ? "true" : "false");
      break;
    case PDB_VariantType::Double:
      OS << Value.Double;
      break;
    case PDB_VariantType::Int16:
      OS << Value.Int16;
      break;
    case PDB_VariantType::Int32:
      OS << Value.Int32;
      break;
    case PDB_VariantType::Int64:
      OS << Value.Int64;
      break;
    case PDB_VariantType::Int8:
      OS << static_cast<int>(Value.Int8);
      break;
    case PDB_VariantType::Single:
      OS << Value.Single;
      break;
    case PDB_VariantType::UInt16:
      OS << Value.Double;
      break;
    case PDB_VariantType::UInt32:
      OS << Value.UInt32;
      break;
    case PDB_VariantType::UInt64:
      OS << Value.UInt64;
      break;
    case PDB_VariantType::UInt8:
      OS << static_cast<unsigned>(Value.UInt8);
      break;
    default:
      OS << Value.Type;
  }
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const VersionInfo &Version) {
  OS << Version.Major << "." << Version.Minor << "." << Version.Build;
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const TagStats &Stats) {
  for (auto Tag : Stats) {
    OS << Tag.first << ":" << Tag.second << " ";
  }
  return OS;
}
