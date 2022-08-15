//===- PDBTypes.h - Defines enums for various fields contained in PDB ---*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBTYPES_H
#define LLVM_DEBUGINFO_PDB_PDBTYPES_H

#include "llvm/Config/llvm-config.h"
#include <functional>
#include <stdint.h>

namespace llvm {

class PDBSymDumper;
class PDBSymbol;

class IPDBDataStream;
template <class T> class IPDBEnumChildren;
class IPDBLineNumber;
class IPDBRawSymbol;
class IPDBSession;
class IPDBSourceFile;

typedef IPDBEnumChildren<PDBSymbol> IPDBEnumSymbols;
typedef IPDBEnumChildren<IPDBSourceFile> IPDBEnumSourceFiles;
typedef IPDBEnumChildren<IPDBDataStream> IPDBEnumDataStreams;
typedef IPDBEnumChildren<IPDBLineNumber> IPDBEnumLineNumbers;

class PDBSymbolExe;
class PDBSymbolCompiland;
class PDBSymbolCompilandDetails;
class PDBSymbolCompilandEnv;
class PDBSymbolFunc;
class PDBSymbolBlock;
class PDBSymbolData;
class PDBSymbolAnnotation;
class PDBSymbolLabel;
class PDBSymbolPublicSymbol;
class PDBSymbolTypeUDT;
class PDBSymbolTypeEnum;
class PDBSymbolTypeFunctionSig;
class PDBSymbolTypePointer;
class PDBSymbolTypeArray;
class PDBSymbolTypeBuiltin;
class PDBSymbolTypeTypedef;
class PDBSymbolTypeBaseClass;
class PDBSymbolTypeFriend;
class PDBSymbolTypeFunctionArg;
class PDBSymbolFuncDebugStart;
class PDBSymbolFuncDebugEnd;
class PDBSymbolUsingNamespace;
class PDBSymbolTypeVTableShape;
class PDBSymbolTypeVTable;
class PDBSymbolCustom;
class PDBSymbolThunk;
class PDBSymbolTypeCustom;
class PDBSymbolTypeManaged;
class PDBSymbolTypeDimension;
class PDBSymbolUnknown;

/// Specifies which PDB reader implementation is to be used.  Only a value
/// of PDB_ReaderType::DIA is supported.
enum class PDB_ReaderType {
  DIA = 0,
};

/// Defines a 128-bit unique identifier.  This maps to a GUID on Windows, but
/// is abstracted here for the purposes of non-Windows platforms that don't have
/// the GUID structure defined.
struct PDB_UniqueId {
  uint64_t HighPart;
  uint64_t LowPart;
};

/// An enumeration indicating the type of data contained in this table.
enum class PDB_TableType {
  Symbols,
  SourceFiles,
  LineNumbers,
  SectionContribs,
  Segments,
  InjectedSources,
  FrameData
};

/// Defines flags used for enumerating child symbols.  This corresponds to the
/// NameSearchOptions enumeration which is documented here:
/// https://msdn.microsoft.com/en-us/library/yat28ads.aspx
enum PDB_NameSearchFlags {
  NS_Default = 0x0,
  NS_CaseSensitive = 0x1,
  NS_CaseInsensitive = 0x2,
  NS_FileNameExtMatch = 0x4,
  NS_Regex = 0x8,
  NS_UndecoratedName = 0x10
};

/// Specifies the hash algorithm that a source file from a PDB was hashed with.
/// This corresponds to the CV_SourceChksum_t enumeration and are documented
/// here: https://msdn.microsoft.com/en-us/library/e96az21x.aspx
enum class PDB_Checksum { None = 0, MD5 = 1, SHA1 = 2 };

/// These values correspond to the CV_CPU_TYPE_e enumeration, and are documented
/// here: https://msdn.microsoft.com/en-us/library/b2fc64ek.aspx
enum class PDB_Cpu {
  Intel8080 = 0x0,
  Intel8086 = 0x1,
  Intel80286 = 0x2,
  Intel80386 = 0x3,
  Intel80486 = 0x4,
  Pentium = 0x5,
  PentiumPro = 0x6,
  Pentium3 = 0x7,
  MIPS = 0x10,
  MIPS16 = 0x11,
  MIPS32 = 0x12,
  MIPS64 = 0x13,
  MIPSI = 0x14,
  MIPSII = 0x15,
  MIPSIII = 0x16,
  MIPSIV = 0x17,
  MIPSV = 0x18,
  M68000 = 0x20,
  M68010 = 0x21,
  M68020 = 0x22,
  M68030 = 0x23,
  M68040 = 0x24,
  Alpha = 0x30,
  Alpha21164 = 0x31,
  Alpha21164A = 0x32,
  Alpha21264 = 0x33,
  Alpha21364 = 0x34,
  PPC601 = 0x40,
  PPC603 = 0x41,
  PPC604 = 0x42,
  PPC620 = 0x43,
  PPCFP = 0x44,
  PPCBE = 0x45,
  SH3 = 0x50,
  SH3E = 0x51,
  SH3DSP = 0x52,
  SH4 = 0x53,
  SHMedia = 0x54,
  ARM3 = 0x60,
  ARM4 = 0x61,
  ARM4T = 0x62,
  ARM5 = 0x63,
  ARM5T = 0x64,
  ARM6 = 0x65,
  ARM_XMAC = 0x66,
  ARM_WMMX = 0x67,
  ARM7 = 0x68,
  Omni = 0x70,
  Ia64 = 0x80,
  Ia64_2 = 0x81,
  CEE = 0x90,
  AM33 = 0xa0,
  M32R = 0xb0,
  TriCore = 0xc0,
  X64 = 0xd0,
  EBC = 0xe0,
  Thumb = 0xf0,
  ARMNT = 0xf4,
  D3D11_Shader = 0x100,
};

enum class PDB_Machine {
  Invalid = 0xffff,
  Unknown = 0x0,
  Am33 = 0x13,
  Amd64 = 0x8664,
  Arm = 0x1C0,
  ArmNT = 0x1C4,
  Ebc = 0xEBC,
  x86 = 0x14C,
  Ia64 = 0x200,
  M32R = 0x9041,
  Mips16 = 0x266,
  MipsFpu = 0x366,
  MipsFpu16 = 0x466,
  PowerPC = 0x1F0,
  PowerPCFP = 0x1F1,
  R4000 = 0x166,
  SH3 = 0x1A2,
  SH3DSP = 0x1A3,
  SH4 = 0x1A6,
  SH5 = 0x1A8,
  Thumb = 0x1C2,
  WceMipsV2 = 0x169
};

/// These values correspond to the CV_call_e enumeration, and are documented
/// at the following locations:
///   https://msdn.microsoft.com/en-us/library/b2fc64ek.aspx
///   https://msdn.microsoft.com/en-us/library/windows/desktop/ms680207(v=vs.85).aspx
///
enum class PDB_CallingConv {
  NearCdecl = 0x00,
  FarCdecl = 0x01,
  NearPascal = 0x02,
  FarPascal = 0x03,
  NearFastcall = 0x04,
  FarFastcall = 0x05,
  Skipped = 0x06,
  NearStdcall = 0x07,
  FarStdcall = 0x08,
  NearSyscall = 0x09,
  FarSyscall = 0x0a,
  Thiscall = 0x0b,
  MipsCall = 0x0c,
  Generic = 0x0d,
  Alphacall = 0x0e,
  Ppccall = 0x0f,
  SuperHCall = 0x10,
  Armcall = 0x11,
  AM33call = 0x12,
  Tricall = 0x13,
  Sh5call = 0x14,
  M32R = 0x15,
  Clrcall = 0x16,
  Inline = 0x17,
  NearVectorcall = 0x18,
  Reserved = 0x19,
};

/// These values correspond to the CV_CFL_LANG enumeration, and are documented
/// here: https://msdn.microsoft.com/en-us/library/bw3aekw6.aspx
enum class PDB_Lang {
  C = 0x00,
  Cpp = 0x01,
  Fortran = 0x02,
  Masm = 0x03,
  Pascal = 0x04,
  Basic = 0x05,
  Cobol = 0x06,
  Link = 0x07,
  Cvtres = 0x08,
  Cvtpgd = 0x09,
  CSharp = 0x0a,
  VB = 0x0b,
  ILAsm = 0x0c,
  Java = 0x0d,
  JScript = 0x0e,
  MSIL = 0x0f,
  HLSL = 0x10
};

/// These values correspond to the DataKind enumeration, and are documented
/// here: https://msdn.microsoft.com/en-us/library/b2x2t313.aspx
enum class PDB_DataKind {
  Unknown,
  Local,
  StaticLocal,
  Param,
  ObjectPtr,
  FileStatic,
  Global,
  Member,
  StaticMember,
  Constant
};

/// These values correspond to the SymTagEnum enumeration, and are documented
/// here: https://msdn.microsoft.com/en-us/library/bkedss5f.aspx
enum class PDB_SymType {
  None,
  Exe,
  Compiland,
  CompilandDetails,
  CompilandEnv,
  Function,
  Block,
  Data,
  Annotation,
  Label,
  PublicSymbol,
  UDT,
  Enum,
  FunctionSig,
  PointerType,
  ArrayType,
  BuiltinType,
  Typedef,
  BaseClass,
  Friend,
  FunctionArg,
  FuncDebugStart,
  FuncDebugEnd,
  UsingNamespace,
  VTableShape,
  VTable,
  Custom,
  Thunk,
  CustomType,
  ManagedType,
  Dimension,
  Max
};

/// These values correspond to the LocationType enumeration, and are documented
/// here: https://msdn.microsoft.com/en-us/library/f57kaez3.aspx
enum class PDB_LocType {
  Null,
  Static,
  TLS,
  RegRel,
  ThisRel,
  Enregistered,
  BitField,
  Slot,
  IlRel,
  MetaData,
  Constant,
  Max
};

/// These values correspond to the THUNK_ORDINAL enumeration, and are documented
/// here: https://msdn.microsoft.com/en-us/library/dh0k8hft.aspx
enum class PDB_ThunkOrdinal {
  Standard,
  ThisAdjustor,
  Vcall,
  Pcode,
  UnknownLoad,
  TrampIncremental,
  BranchIsland
};

/// These values correspond to the UdtKind enumeration, and are documented
/// here: https://msdn.microsoft.com/en-us/library/wcstk66t.aspx
enum class PDB_UdtType { Struct, Class, Union, Interface };

// HLSL Change Starts - undef FPO if defined
#ifdef FPO
#undef FPO
#endif
// HLSL Change Ends

/// These values correspond to the StackFrameTypeEnum enumeration, and are
/// documented here: https://msdn.microsoft.com/en-us/library/bc5207xw.aspx.
enum class PDB_StackFrameType { FPO, KernelTrap, KernelTSS, EBP, FrameData };

/// These values correspond to the StackFrameTypeEnum enumeration, and are
/// documented here: https://msdn.microsoft.com/en-us/library/bc5207xw.aspx.
enum class PDB_MemoryType { Code, Data, Stack, HeapCode };

/// These values correspond to the Basictype enumeration, and are documented
/// here: https://msdn.microsoft.com/en-us/library/4szdtzc3.aspx
enum class PDB_BuiltinType {
  None = 0,
  Void = 1,
  Char = 2,
  WCharT = 3,
  Int = 6,
  UInt = 7,
  Float = 8,
  BCD = 9,
  Bool = 10,
  Long = 13,
  ULong = 14,
  Currency = 25,
  Date = 26,
  Variant = 27,
  Complex = 28,
  Bitfield = 29,
  BSTR = 30,
  HResult = 31
};

enum class PDB_RegisterId {
  Unknown = 0,
  VFrame = 30006,
  AL = 1,
  CL = 2,
  DL = 3,
  BL = 4,
  AH = 5,
  CH = 6,
  DH = 7,
  BH = 8,
  AX = 9,
  CX = 10,
  DX = 11,
  BX = 12,
  SP = 13,
  BP = 14,
  SI = 15,
  DI = 16,
  EAX = 17,
  ECX = 18,
  EDX = 19,
  EBX = 20,
  ESP = 21,
  EBP = 22,
  ESI = 23,
  EDI = 24,
  ES = 25,
  CS = 26,
  SS = 27,
  DS = 28,
  FS = 29,
  GS = 30,
  IP = 31,
  RAX = 328,
  RBX = 329,
  RCX = 330,
  RDX = 331,
  RSI = 332,
  RDI = 333,
  RBP = 334,
  RSP = 335,
  R8 = 336,
  R9 = 337,
  R10 = 338,
  R11 = 339,
  R12 = 340,
  R13 = 341,
  R14 = 342,
  R15 = 343,
};

enum class PDB_MemberAccess { Private = 1, Protected = 2, Public = 3 };

enum class PDB_ErrorCode {
  Success,
  NoPdbImpl,
  InvalidPath,
  InvalidFileFormat,
  InvalidParameter,
  AlreadyLoaded,
  UnknownError,
  NoMemory,
  DebugInfoMismatch
};

struct VersionInfo {
  uint32_t Major;
  uint32_t Minor;
  uint32_t Build;
  uint32_t QFE;
};

enum PDB_VariantType {
  Empty,
  Unknown,
  Int8,
  Int16,
  Int32,
  Int64,
  Single,
  Double,
  UInt8,
  UInt16,
  UInt32,
  UInt64,
  Bool,
};

struct Variant {
  Variant()
    : Type(PDB_VariantType::Empty) {
  }

  PDB_VariantType Type;
  union {
    bool Bool;
    int8_t Int8;
    int16_t Int16;
    int32_t Int32;
    int64_t Int64;
    float Single;
    double Double;
    uint8_t UInt8;
    uint16_t UInt16;
    uint32_t UInt32;
    uint64_t UInt64;
  };
#define VARIANT_EQUAL_CASE(Enum)                                               \
  case PDB_VariantType::Enum:                                                  \
    return Enum == Other.Enum;
  bool operator==(const Variant &Other) const {
    if (Type != Other.Type)
      return false;
    switch (Type) {
      VARIANT_EQUAL_CASE(Bool)
      VARIANT_EQUAL_CASE(Int8)
      VARIANT_EQUAL_CASE(Int16)
      VARIANT_EQUAL_CASE(Int32)
      VARIANT_EQUAL_CASE(Int64)
      VARIANT_EQUAL_CASE(Single)
      VARIANT_EQUAL_CASE(Double)
      VARIANT_EQUAL_CASE(UInt8)
      VARIANT_EQUAL_CASE(UInt16)
      VARIANT_EQUAL_CASE(UInt32)
      VARIANT_EQUAL_CASE(UInt64)
    default:
      return true;
    }
  }
#undef VARIANT_EQUAL_CASE
  bool operator!=(const Variant &Other) const { return !(*this == Other); }
};

} // namespace llvm

namespace std {
template <> struct hash<llvm::PDB_SymType> {
  typedef llvm::PDB_SymType argument_type;
  typedef std::size_t result_type;

  result_type operator()(const argument_type &Arg) const {
    return std::hash<int>()(static_cast<int>(Arg));
  }
};
}

#endif
