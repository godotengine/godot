//===- DIARawSymbol.cpp - DIA implementation of IPDBRawSymbol ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/PDB/DIA/DIAEnumSymbols.h"
#include "llvm/DebugInfo/PDB/DIA/DIARawSymbol.h"
#include "llvm/DebugInfo/PDB/DIA/DIASession.h"
#include "llvm/DebugInfo/PDB/PDBExtras.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
Variant VariantFromVARIANT(const VARIANT &V) {
  Variant Result;
  switch (V.vt) {
  case VT_I1:
    Result.Int8 = V.cVal;
    Result.Type = PDB_VariantType::Int8;
    break;
  case VT_I2:
    Result.Int16 = V.iVal;
    Result.Type = PDB_VariantType::Int16;
    break;
  case VT_I4:
    Result.Int32 = V.intVal;
    Result.Type = PDB_VariantType::Int32;
    break;
  case VT_I8:
    Result.Int64 = V.llVal;
    Result.Type = PDB_VariantType::Int64;
    break;
  case VT_UI1:
    Result.UInt8 = V.bVal;
    Result.Type = PDB_VariantType::UInt8;
    break;
  case VT_UI2:
    Result.UInt16 = V.uiVal;
    Result.Type = PDB_VariantType::UInt16;
    break;
  case VT_UI4:
    Result.UInt32 = V.uintVal;
    Result.Type = PDB_VariantType::UInt32;
    break;
  case VT_UI8:
    Result.UInt64 = V.ullVal;
    Result.Type = PDB_VariantType::UInt64;
    break;
  case VT_BOOL:
    Result.Bool = (V.boolVal == VARIANT_TRUE) ? true : false;
    Result.Type = PDB_VariantType::Bool;
    break;
  case VT_R4:
    Result.Single = V.fltVal;
    Result.Type = PDB_VariantType::Single;
    break;
  case VT_R8:
    Result.Double = V.dblVal;
    Result.Type = PDB_VariantType::Double;
    break;
  default:
    Result.Type = PDB_VariantType::Unknown;
    break;
  }
  return Result;
}

template <typename ArgType>
ArgType PrivateGetDIAValue(IDiaSymbol *Symbol,
                           HRESULT (__stdcall IDiaSymbol::*Method)(ArgType *)) {
  ArgType Value;
  if (S_OK == (Symbol->*Method)(&Value))
    return static_cast<ArgType>(Value);

  return ArgType();
}

template <typename ArgType, typename RetType>
RetType PrivateGetDIAValue(IDiaSymbol *Symbol,
                           HRESULT (__stdcall IDiaSymbol::*Method)(ArgType *)) {
  ArgType Value;
  if (S_OK == (Symbol->*Method)(&Value))
    return static_cast<RetType>(Value);

  return RetType();
}

std::string
PrivateGetDIAValue(IDiaSymbol *Symbol,
                   HRESULT (__stdcall IDiaSymbol::*Method)(BSTR *)) {
  CComBSTR Result16;
  if (S_OK != (Symbol->*Method)(&Result16))
    return std::string();

  const char *SrcBytes = reinterpret_cast<const char *>(Result16.m_str);
  llvm::ArrayRef<char> SrcByteArray(SrcBytes, Result16.ByteLength());
  std::string Result8;
  if (!llvm::convertUTF16ToUTF8String(SrcByteArray, Result8))
    return std::string();
  return Result8;
}

PDB_UniqueId
PrivateGetDIAValue(IDiaSymbol *Symbol,
                   HRESULT (__stdcall IDiaSymbol::*Method)(GUID *)) {
  GUID Result;
  if (S_OK != (Symbol->*Method)(&Result))
    return PDB_UniqueId();

  static_assert(sizeof(PDB_UniqueId) == sizeof(GUID),
                "PDB_UniqueId is the wrong size!");
  PDB_UniqueId IdResult;
  ::memcpy(&IdResult, &Result, sizeof(GUID));
  return IdResult;
}

template <typename ArgType>
void DumpDIAValue(llvm::raw_ostream &OS, int Indent, StringRef Name,
                  IDiaSymbol *Symbol,
                  HRESULT (__stdcall IDiaSymbol::*Method)(ArgType *)) {
  ArgType Value;
  if (S_OK == (Symbol->*Method)(&Value)) {
    OS << "\n";
    OS.indent(Indent);
    OS << Name << ": " << Value;
  }
}

void DumpDIAValue(llvm::raw_ostream &OS, int Indent, StringRef Name,
                  IDiaSymbol *Symbol,
                  HRESULT (__stdcall IDiaSymbol::*Method)(BSTR *)) {
  BSTR Value = nullptr;
  if (S_OK != (Symbol->*Method)(&Value))
    return;
  const char *Bytes = reinterpret_cast<const char *>(Value);
  ArrayRef<char> ByteArray(Bytes, ::SysStringByteLen(Value));
  std::string Result;
  if (llvm::convertUTF16ToUTF8String(ByteArray, Result)) {
    OS << "\n";
    OS.indent(Indent);
    OS << Name << ": " << Result;
  }
  ::SysFreeString(Value);
}

void DumpDIAValue(llvm::raw_ostream &OS, int Indent, StringRef Name,
                  IDiaSymbol *Symbol,
                  HRESULT (__stdcall IDiaSymbol::*Method)(VARIANT *)) {
  VARIANT Value;
  Value.vt = VT_EMPTY;
  if (S_OK != (Symbol->*Method)(&Value))
    return;
  OS << "\n";
  OS.indent(Indent);
  Variant V = VariantFromVARIANT(Value);
  OS << V;
}
}

namespace llvm {
raw_ostream &operator<<(raw_ostream &OS, const GUID &Guid) {
  const PDB_UniqueId *Id = reinterpret_cast<const PDB_UniqueId *>(&Guid);
  OS << *Id;
  return OS;
}
}

DIARawSymbol::DIARawSymbol(const DIASession &PDBSession,
                           CComPtr<IDiaSymbol> DiaSymbol)
    : Session(PDBSession), Symbol(DiaSymbol) {}

#define RAW_METHOD_DUMP(Stream, Method)                                        \
  DumpDIAValue(Stream, Indent, StringRef(#Method), Symbol, &IDiaSymbol::Method);

void DIARawSymbol::dump(raw_ostream &OS, int Indent) const {
  RAW_METHOD_DUMP(OS, get_access)
  RAW_METHOD_DUMP(OS, get_addressOffset)
  RAW_METHOD_DUMP(OS, get_addressSection)
  RAW_METHOD_DUMP(OS, get_age)
  RAW_METHOD_DUMP(OS, get_arrayIndexTypeId)
  RAW_METHOD_DUMP(OS, get_backEndMajor)
  RAW_METHOD_DUMP(OS, get_backEndMinor)
  RAW_METHOD_DUMP(OS, get_backEndBuild)
  RAW_METHOD_DUMP(OS, get_backEndQFE)
  RAW_METHOD_DUMP(OS, get_baseDataOffset)
  RAW_METHOD_DUMP(OS, get_baseDataSlot)
  RAW_METHOD_DUMP(OS, get_baseSymbolId)
  RAW_METHOD_DUMP(OS, get_baseType)
  RAW_METHOD_DUMP(OS, get_bitPosition)
  RAW_METHOD_DUMP(OS, get_callingConvention)
  RAW_METHOD_DUMP(OS, get_classParentId)
  RAW_METHOD_DUMP(OS, get_compilerName)
  RAW_METHOD_DUMP(OS, get_count)
  RAW_METHOD_DUMP(OS, get_countLiveRanges)
  RAW_METHOD_DUMP(OS, get_frontEndMajor)
  RAW_METHOD_DUMP(OS, get_frontEndMinor)
  RAW_METHOD_DUMP(OS, get_frontEndBuild)
  RAW_METHOD_DUMP(OS, get_frontEndQFE)
  RAW_METHOD_DUMP(OS, get_lexicalParentId)
  RAW_METHOD_DUMP(OS, get_libraryName)
  RAW_METHOD_DUMP(OS, get_liveRangeStartAddressOffset)
  RAW_METHOD_DUMP(OS, get_liveRangeStartAddressSection)
  RAW_METHOD_DUMP(OS, get_liveRangeStartRelativeVirtualAddress)
  RAW_METHOD_DUMP(OS, get_localBasePointerRegisterId)
  RAW_METHOD_DUMP(OS, get_lowerBoundId)
  RAW_METHOD_DUMP(OS, get_memorySpaceKind)
  RAW_METHOD_DUMP(OS, get_name)
  RAW_METHOD_DUMP(OS, get_numberOfAcceleratorPointerTags)
  RAW_METHOD_DUMP(OS, get_numberOfColumns)
  RAW_METHOD_DUMP(OS, get_numberOfModifiers)
  RAW_METHOD_DUMP(OS, get_numberOfRegisterIndices)
  RAW_METHOD_DUMP(OS, get_numberOfRows)
  RAW_METHOD_DUMP(OS, get_objectFileName)
  RAW_METHOD_DUMP(OS, get_oemId)
  RAW_METHOD_DUMP(OS, get_oemSymbolId)
  RAW_METHOD_DUMP(OS, get_offsetInUdt)
  RAW_METHOD_DUMP(OS, get_platform)
  RAW_METHOD_DUMP(OS, get_rank)
  RAW_METHOD_DUMP(OS, get_registerId)
  RAW_METHOD_DUMP(OS, get_registerType)
  RAW_METHOD_DUMP(OS, get_relativeVirtualAddress)
  RAW_METHOD_DUMP(OS, get_samplerSlot)
  RAW_METHOD_DUMP(OS, get_signature)
  RAW_METHOD_DUMP(OS, get_sizeInUdt)
  RAW_METHOD_DUMP(OS, get_slot)
  RAW_METHOD_DUMP(OS, get_sourceFileName)
  RAW_METHOD_DUMP(OS, get_stride)
  RAW_METHOD_DUMP(OS, get_subTypeId)
  RAW_METHOD_DUMP(OS, get_symbolsFileName)
  RAW_METHOD_DUMP(OS, get_symIndexId)
  RAW_METHOD_DUMP(OS, get_targetOffset)
  RAW_METHOD_DUMP(OS, get_targetRelativeVirtualAddress)
  RAW_METHOD_DUMP(OS, get_targetVirtualAddress)
  RAW_METHOD_DUMP(OS, get_targetSection)
  RAW_METHOD_DUMP(OS, get_textureSlot)
  RAW_METHOD_DUMP(OS, get_timeStamp)
  RAW_METHOD_DUMP(OS, get_token)
  RAW_METHOD_DUMP(OS, get_typeId)
  RAW_METHOD_DUMP(OS, get_uavSlot)
  RAW_METHOD_DUMP(OS, get_undecoratedName)
  RAW_METHOD_DUMP(OS, get_unmodifiedTypeId)
  RAW_METHOD_DUMP(OS, get_upperBoundId)
  RAW_METHOD_DUMP(OS, get_virtualBaseDispIndex)
  RAW_METHOD_DUMP(OS, get_virtualBaseOffset)
  RAW_METHOD_DUMP(OS, get_virtualTableShapeId)
  RAW_METHOD_DUMP(OS, get_dataKind)
  RAW_METHOD_DUMP(OS, get_symTag)
  RAW_METHOD_DUMP(OS, get_guid)
  RAW_METHOD_DUMP(OS, get_offset)
  RAW_METHOD_DUMP(OS, get_thisAdjust)
  RAW_METHOD_DUMP(OS, get_virtualBasePointerOffset)
  RAW_METHOD_DUMP(OS, get_locationType)
  RAW_METHOD_DUMP(OS, get_machineType)
  RAW_METHOD_DUMP(OS, get_thunkOrdinal)
  RAW_METHOD_DUMP(OS, get_length)
  RAW_METHOD_DUMP(OS, get_liveRangeLength)
  RAW_METHOD_DUMP(OS, get_virtualAddress)
  RAW_METHOD_DUMP(OS, get_udtKind)
  RAW_METHOD_DUMP(OS, get_constructor)
  RAW_METHOD_DUMP(OS, get_customCallingConvention)
  RAW_METHOD_DUMP(OS, get_farReturn)
  RAW_METHOD_DUMP(OS, get_code)
  RAW_METHOD_DUMP(OS, get_compilerGenerated)
  RAW_METHOD_DUMP(OS, get_constType)
  RAW_METHOD_DUMP(OS, get_editAndContinueEnabled)
  RAW_METHOD_DUMP(OS, get_function)
  RAW_METHOD_DUMP(OS, get_stride)
  RAW_METHOD_DUMP(OS, get_noStackOrdering)
  RAW_METHOD_DUMP(OS, get_hasAlloca)
  RAW_METHOD_DUMP(OS, get_hasAssignmentOperator)
  RAW_METHOD_DUMP(OS, get_isCTypes)
  RAW_METHOD_DUMP(OS, get_hasCastOperator)
  RAW_METHOD_DUMP(OS, get_hasDebugInfo)
  RAW_METHOD_DUMP(OS, get_hasEH)
  RAW_METHOD_DUMP(OS, get_hasEHa)
  RAW_METHOD_DUMP(OS, get_hasInlAsm)
  RAW_METHOD_DUMP(OS, get_framePointerPresent)
  RAW_METHOD_DUMP(OS, get_inlSpec)
  RAW_METHOD_DUMP(OS, get_interruptReturn)
  RAW_METHOD_DUMP(OS, get_hasLongJump)
  RAW_METHOD_DUMP(OS, get_hasManagedCode)
  RAW_METHOD_DUMP(OS, get_hasNestedTypes)
  RAW_METHOD_DUMP(OS, get_noInline)
  RAW_METHOD_DUMP(OS, get_noReturn)
  RAW_METHOD_DUMP(OS, get_optimizedCodeDebugInfo)
  RAW_METHOD_DUMP(OS, get_overloadedOperator)
  RAW_METHOD_DUMP(OS, get_hasSEH)
  RAW_METHOD_DUMP(OS, get_hasSecurityChecks)
  RAW_METHOD_DUMP(OS, get_hasSetJump)
  RAW_METHOD_DUMP(OS, get_strictGSCheck)
  RAW_METHOD_DUMP(OS, get_isAcceleratorGroupSharedLocal)
  RAW_METHOD_DUMP(OS, get_isAcceleratorPointerTagLiveRange)
  RAW_METHOD_DUMP(OS, get_isAcceleratorStubFunction)
  RAW_METHOD_DUMP(OS, get_isAggregated)
  RAW_METHOD_DUMP(OS, get_intro)
  RAW_METHOD_DUMP(OS, get_isCVTCIL)
  RAW_METHOD_DUMP(OS, get_isConstructorVirtualBase)
  RAW_METHOD_DUMP(OS, get_isCxxReturnUdt)
  RAW_METHOD_DUMP(OS, get_isDataAligned)
  RAW_METHOD_DUMP(OS, get_isHLSLData)
  RAW_METHOD_DUMP(OS, get_isHotpatchable)
  RAW_METHOD_DUMP(OS, get_indirectVirtualBaseClass)
  RAW_METHOD_DUMP(OS, get_isInterfaceUdt)
  RAW_METHOD_DUMP(OS, get_intrinsic)
  RAW_METHOD_DUMP(OS, get_isLTCG)
  RAW_METHOD_DUMP(OS, get_isLocationControlFlowDependent)
  RAW_METHOD_DUMP(OS, get_isMSILNetmodule)
  RAW_METHOD_DUMP(OS, get_isMatrixRowMajor)
  RAW_METHOD_DUMP(OS, get_managed)
  RAW_METHOD_DUMP(OS, get_msil)
  RAW_METHOD_DUMP(OS, get_isMultipleInheritance)
  RAW_METHOD_DUMP(OS, get_isNaked)
  RAW_METHOD_DUMP(OS, get_nested)
  RAW_METHOD_DUMP(OS, get_isOptimizedAway)
  RAW_METHOD_DUMP(OS, get_packed)
  RAW_METHOD_DUMP(OS, get_isPointerBasedOnSymbolValue)
  RAW_METHOD_DUMP(OS, get_isPointerToDataMember)
  RAW_METHOD_DUMP(OS, get_isPointerToMemberFunction)
  RAW_METHOD_DUMP(OS, get_pure)
  RAW_METHOD_DUMP(OS, get_RValueReference)
  RAW_METHOD_DUMP(OS, get_isRefUdt)
  RAW_METHOD_DUMP(OS, get_reference)
  RAW_METHOD_DUMP(OS, get_restrictedType)
  RAW_METHOD_DUMP(OS, get_isReturnValue)
  RAW_METHOD_DUMP(OS, get_isSafeBuffers)
  RAW_METHOD_DUMP(OS, get_scoped)
  RAW_METHOD_DUMP(OS, get_isSdl)
  RAW_METHOD_DUMP(OS, get_isSingleInheritance)
  RAW_METHOD_DUMP(OS, get_isSplitted)
  RAW_METHOD_DUMP(OS, get_isStatic)
  RAW_METHOD_DUMP(OS, get_isStripped)
  RAW_METHOD_DUMP(OS, get_unalignedType)
  RAW_METHOD_DUMP(OS, get_notReached)
  RAW_METHOD_DUMP(OS, get_isValueUdt)
  RAW_METHOD_DUMP(OS, get_virtual)
  RAW_METHOD_DUMP(OS, get_virtualBaseClass)
  RAW_METHOD_DUMP(OS, get_isVirtualInheritance)
  RAW_METHOD_DUMP(OS, get_volatileType)
  RAW_METHOD_DUMP(OS, get_wasInlined)
  RAW_METHOD_DUMP(OS, get_unused)
  RAW_METHOD_DUMP(OS, get_value)
}

std::unique_ptr<IPDBEnumSymbols>
DIARawSymbol::findChildren(PDB_SymType Type) const {
  enum SymTagEnum EnumVal = static_cast<enum SymTagEnum>(Type);

  CComPtr<IDiaEnumSymbols> DiaEnumerator;
  if (S_OK != Symbol->findChildrenEx(EnumVal, nullptr, nsNone, &DiaEnumerator))
    return nullptr;

  return llvm::make_unique<DIAEnumSymbols>(Session, DiaEnumerator);
}

std::unique_ptr<IPDBEnumSymbols>
DIARawSymbol::findChildren(PDB_SymType Type, StringRef Name,
                           PDB_NameSearchFlags Flags) const {
  llvm::SmallVector<UTF16, 32> Name16;
  llvm::convertUTF8ToUTF16String(Name, Name16);

  enum SymTagEnum EnumVal = static_cast<enum SymTagEnum>(Type);
  DWORD CompareFlags = static_cast<DWORD>(Flags);
  wchar_t *Name16Str = reinterpret_cast<wchar_t *>(Name16.data());

  CComPtr<IDiaEnumSymbols> DiaEnumerator;
  if (S_OK !=
      Symbol->findChildrenEx(EnumVal, Name16Str, CompareFlags, &DiaEnumerator))
    return nullptr;

  return llvm::make_unique<DIAEnumSymbols>(Session, DiaEnumerator);
}

std::unique_ptr<IPDBEnumSymbols>
DIARawSymbol::findChildrenByRVA(PDB_SymType Type, StringRef Name,
                                PDB_NameSearchFlags Flags, uint32_t RVA) const {
  llvm::SmallVector<UTF16, 32> Name16;
  llvm::convertUTF8ToUTF16String(Name, Name16);

  enum SymTagEnum EnumVal = static_cast<enum SymTagEnum>(Type);
  DWORD CompareFlags = static_cast<DWORD>(Flags);
  wchar_t *Name16Str = reinterpret_cast<wchar_t *>(Name16.data());

  CComPtr<IDiaEnumSymbols> DiaEnumerator;
  if (S_OK !=
      Symbol->findChildrenExByRVA(EnumVal, Name16Str, CompareFlags, RVA,
                                  &DiaEnumerator))
    return nullptr;

  return llvm::make_unique<DIAEnumSymbols>(Session, DiaEnumerator);
}

std::unique_ptr<IPDBEnumSymbols>
DIARawSymbol::findInlineFramesByRVA(uint32_t RVA) const {
  CComPtr<IDiaEnumSymbols> DiaEnumerator;
  if (S_OK != Symbol->findInlineFramesByRVA(RVA, &DiaEnumerator))
    return nullptr;

  return llvm::make_unique<DIAEnumSymbols>(Session, DiaEnumerator);
}

void DIARawSymbol::getDataBytes(llvm::SmallVector<uint8_t, 32> &bytes) const {
  bytes.clear();

  DWORD DataSize = 0;
  Symbol->get_dataBytes(0, &DataSize, nullptr);
  if (DataSize == 0)
    return;

  bytes.resize(DataSize);
  Symbol->get_dataBytes(DataSize, &DataSize, bytes.data());
}

PDB_MemberAccess DIARawSymbol::getAccess() const {
  return PrivateGetDIAValue<DWORD, PDB_MemberAccess>(Symbol,
                                                     &IDiaSymbol::get_access);
}

uint32_t DIARawSymbol::getAddressOffset() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_addressOffset);
}

uint32_t DIARawSymbol::getAddressSection() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_addressSection);
}

uint32_t DIARawSymbol::getAge() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_age);
}

uint32_t DIARawSymbol::getArrayIndexTypeId() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_arrayIndexTypeId);
}

void DIARawSymbol::getBackEndVersion(VersionInfo &Version) const {
  Version.Major = PrivateGetDIAValue(Symbol, &IDiaSymbol::get_backEndMajor);
  Version.Minor = PrivateGetDIAValue(Symbol, &IDiaSymbol::get_backEndMinor);
  Version.Build = PrivateGetDIAValue(Symbol, &IDiaSymbol::get_backEndBuild);
  Version.QFE = PrivateGetDIAValue(Symbol, &IDiaSymbol::get_backEndQFE);
}

uint32_t DIARawSymbol::getBaseDataOffset() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_baseDataOffset);
}

uint32_t DIARawSymbol::getBaseDataSlot() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_baseDataSlot);
}

uint32_t DIARawSymbol::getBaseSymbolId() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_baseSymbolId);
}

PDB_BuiltinType DIARawSymbol::getBuiltinType() const {
  return PrivateGetDIAValue<DWORD, PDB_BuiltinType>(Symbol,
                                                    &IDiaSymbol::get_baseType);
}

uint32_t DIARawSymbol::getBitPosition() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_bitPosition);
}

PDB_CallingConv DIARawSymbol::getCallingConvention() const {
  return PrivateGetDIAValue<DWORD, PDB_CallingConv>(
      Symbol, &IDiaSymbol::get_callingConvention);
}

uint32_t DIARawSymbol::getClassParentId() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_classParentId);
}

std::string DIARawSymbol::getCompilerName() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_compilerName);
}

uint32_t DIARawSymbol::getCount() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_count);
}

uint32_t DIARawSymbol::getCountLiveRanges() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_countLiveRanges);
}

void DIARawSymbol::getFrontEndVersion(VersionInfo &Version) const {
  Version.Major = PrivateGetDIAValue(Symbol, &IDiaSymbol::get_frontEndMajor);
  Version.Minor = PrivateGetDIAValue(Symbol, &IDiaSymbol::get_frontEndMinor);
  Version.Build = PrivateGetDIAValue(Symbol, &IDiaSymbol::get_frontEndBuild);
  Version.QFE = PrivateGetDIAValue(Symbol, &IDiaSymbol::get_frontEndQFE);
}

PDB_Lang DIARawSymbol::getLanguage() const {
  return PrivateGetDIAValue<DWORD, PDB_Lang>(Symbol, &IDiaSymbol::get_language);
}

uint32_t DIARawSymbol::getLexicalParentId() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_lexicalParentId);
}

std::string DIARawSymbol::getLibraryName() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_libraryName);
}

uint32_t DIARawSymbol::getLiveRangeStartAddressOffset() const {
  return PrivateGetDIAValue(Symbol,
                            &IDiaSymbol::get_liveRangeStartAddressOffset);
}

uint32_t DIARawSymbol::getLiveRangeStartAddressSection() const {
  return PrivateGetDIAValue(Symbol,
                            &IDiaSymbol::get_liveRangeStartAddressSection);
}

uint32_t DIARawSymbol::getLiveRangeStartRelativeVirtualAddress() const {
  return PrivateGetDIAValue(
      Symbol, &IDiaSymbol::get_liveRangeStartRelativeVirtualAddress);
}

PDB_RegisterId DIARawSymbol::getLocalBasePointerRegisterId() const {
  return PrivateGetDIAValue<DWORD, PDB_RegisterId>(
      Symbol, &IDiaSymbol::get_localBasePointerRegisterId);
}

uint32_t DIARawSymbol::getLowerBoundId() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_lowerBoundId);
}

uint32_t DIARawSymbol::getMemorySpaceKind() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_memorySpaceKind);
}

std::string DIARawSymbol::getName() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_name);
}

uint32_t DIARawSymbol::getNumberOfAcceleratorPointerTags() const {
  return PrivateGetDIAValue(Symbol,
                            &IDiaSymbol::get_numberOfAcceleratorPointerTags);
}

uint32_t DIARawSymbol::getNumberOfColumns() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_numberOfColumns);
}

uint32_t DIARawSymbol::getNumberOfModifiers() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_numberOfModifiers);
}

uint32_t DIARawSymbol::getNumberOfRegisterIndices() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_numberOfRegisterIndices);
}

uint32_t DIARawSymbol::getNumberOfRows() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_numberOfRows);
}

std::string DIARawSymbol::getObjectFileName() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_objectFileName);
}

uint32_t DIARawSymbol::getOemId() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_oemId);
}

uint32_t DIARawSymbol::getOemSymbolId() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_oemSymbolId);
}

uint32_t DIARawSymbol::getOffsetInUdt() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_offsetInUdt);
}

PDB_Cpu DIARawSymbol::getPlatform() const {
  return PrivateGetDIAValue<DWORD, PDB_Cpu>(Symbol, &IDiaSymbol::get_platform);
}

uint32_t DIARawSymbol::getRank() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_rank);
}

PDB_RegisterId DIARawSymbol::getRegisterId() const {
  return PrivateGetDIAValue<DWORD, PDB_RegisterId>(Symbol,
                                                   &IDiaSymbol::get_registerId);
}

uint32_t DIARawSymbol::getRegisterType() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_registerType);
}

uint32_t DIARawSymbol::getRelativeVirtualAddress() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_relativeVirtualAddress);
}

uint32_t DIARawSymbol::getSamplerSlot() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_samplerSlot);
}

uint32_t DIARawSymbol::getSignature() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_signature);
}

uint32_t DIARawSymbol::getSizeInUdt() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_sizeInUdt);
}

uint32_t DIARawSymbol::getSlot() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_slot);
}

std::string DIARawSymbol::getSourceFileName() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_sourceFileName);
}

uint32_t DIARawSymbol::getStride() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_stride);
}

uint32_t DIARawSymbol::getSubTypeId() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_subTypeId);
}

std::string DIARawSymbol::getSymbolsFileName() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_symbolsFileName);
}

uint32_t DIARawSymbol::getSymIndexId() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_symIndexId);
}

uint32_t DIARawSymbol::getTargetOffset() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_targetOffset);
}

uint32_t DIARawSymbol::getTargetRelativeVirtualAddress() const {
  return PrivateGetDIAValue(Symbol,
                            &IDiaSymbol::get_targetRelativeVirtualAddress);
}

uint64_t DIARawSymbol::getTargetVirtualAddress() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_targetVirtualAddress);
}

uint32_t DIARawSymbol::getTargetSection() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_targetSection);
}

uint32_t DIARawSymbol::getTextureSlot() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_textureSlot);
}

uint32_t DIARawSymbol::getTimeStamp() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_timeStamp);
}

uint32_t DIARawSymbol::getToken() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_token);
}

uint32_t DIARawSymbol::getTypeId() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_typeId);
}

uint32_t DIARawSymbol::getUavSlot() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_uavSlot);
}

std::string DIARawSymbol::getUndecoratedName() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_undecoratedName);
}

uint32_t DIARawSymbol::getUnmodifiedTypeId() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_unmodifiedTypeId);
}

uint32_t DIARawSymbol::getUpperBoundId() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_upperBoundId);
}

Variant DIARawSymbol::getValue() const {
  VARIANT Value;
  Value.vt = VT_EMPTY;
  if (S_OK != Symbol->get_value(&Value))
    return Variant();

  return VariantFromVARIANT(Value);
}

uint32_t DIARawSymbol::getVirtualBaseDispIndex() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_virtualBaseDispIndex);
}

uint32_t DIARawSymbol::getVirtualBaseOffset() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_virtualBaseOffset);
}

uint32_t DIARawSymbol::getVirtualTableShapeId() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_virtualTableShapeId);
}

PDB_DataKind DIARawSymbol::getDataKind() const {
  return PrivateGetDIAValue<DWORD, PDB_DataKind>(Symbol,
                                                 &IDiaSymbol::get_dataKind);
}

PDB_SymType DIARawSymbol::getSymTag() const {
  return PrivateGetDIAValue<DWORD, PDB_SymType>(Symbol,
                                                &IDiaSymbol::get_symTag);
}

PDB_UniqueId DIARawSymbol::getGuid() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_guid);
}

int32_t DIARawSymbol::getOffset() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_offset);
}

int32_t DIARawSymbol::getThisAdjust() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_thisAdjust);
}

int32_t DIARawSymbol::getVirtualBasePointerOffset() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_virtualBasePointerOffset);
}

PDB_LocType DIARawSymbol::getLocationType() const {
  return PrivateGetDIAValue<DWORD, PDB_LocType>(Symbol,
                                                &IDiaSymbol::get_locationType);
}

PDB_Machine DIARawSymbol::getMachineType() const {
  return PrivateGetDIAValue<DWORD, PDB_Machine>(Symbol,
                                                &IDiaSymbol::get_machineType);
}

PDB_ThunkOrdinal DIARawSymbol::getThunkOrdinal() const {
  return PrivateGetDIAValue<DWORD, PDB_ThunkOrdinal>(
      Symbol, &IDiaSymbol::get_thunkOrdinal);
}

uint64_t DIARawSymbol::getLength() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_length);
}

uint64_t DIARawSymbol::getLiveRangeLength() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_liveRangeLength);
}

uint64_t DIARawSymbol::getVirtualAddress() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_virtualAddress);
}

PDB_UdtType DIARawSymbol::getUdtKind() const {
  return PrivateGetDIAValue<DWORD, PDB_UdtType>(Symbol,
                                                &IDiaSymbol::get_udtKind);
}

bool DIARawSymbol::hasConstructor() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_constructor);
}

bool DIARawSymbol::hasCustomCallingConvention() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_customCallingConvention);
}

bool DIARawSymbol::hasFarReturn() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_farReturn);
}

bool DIARawSymbol::isCode() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_code);
}

bool DIARawSymbol::isCompilerGenerated() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_compilerGenerated);
}

bool DIARawSymbol::isConstType() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_constType);
}

bool DIARawSymbol::isEditAndContinueEnabled() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_editAndContinueEnabled);
}

bool DIARawSymbol::isFunction() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_function);
}

bool DIARawSymbol::getAddressTaken() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_addressTaken);
}

bool DIARawSymbol::getNoStackOrdering() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_noStackOrdering);
}

bool DIARawSymbol::hasAlloca() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_hasAlloca);
}

bool DIARawSymbol::hasAssignmentOperator() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_hasAssignmentOperator);
}

bool DIARawSymbol::hasCTypes() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isCTypes);
}

bool DIARawSymbol::hasCastOperator() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_hasCastOperator);
}

bool DIARawSymbol::hasDebugInfo() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_hasDebugInfo);
}

bool DIARawSymbol::hasEH() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_hasEH);
}

bool DIARawSymbol::hasEHa() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_hasEHa);
}

bool DIARawSymbol::hasInlAsm() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_hasInlAsm);
}

bool DIARawSymbol::hasInlineAttribute() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_inlSpec);
}

bool DIARawSymbol::hasInterruptReturn() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_interruptReturn);
}

bool DIARawSymbol::hasFramePointer() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_framePointerPresent);
}

bool DIARawSymbol::hasLongJump() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_hasLongJump);
}

bool DIARawSymbol::hasManagedCode() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_hasManagedCode);
}

bool DIARawSymbol::hasNestedTypes() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_hasNestedTypes);
}

bool DIARawSymbol::hasNoInlineAttribute() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_noInline);
}

bool DIARawSymbol::hasNoReturnAttribute() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_noReturn);
}

bool DIARawSymbol::hasOptimizedCodeDebugInfo() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_optimizedCodeDebugInfo);
}

bool DIARawSymbol::hasOverloadedOperator() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_overloadedOperator);
}

bool DIARawSymbol::hasSEH() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_hasSEH);
}

bool DIARawSymbol::hasSecurityChecks() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_hasSecurityChecks);
}

bool DIARawSymbol::hasSetJump() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_hasSetJump);
}

bool DIARawSymbol::hasStrictGSCheck() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_strictGSCheck);
}

bool DIARawSymbol::isAcceleratorGroupSharedLocal() const {
  return PrivateGetDIAValue(Symbol,
                            &IDiaSymbol::get_isAcceleratorGroupSharedLocal);
}

bool DIARawSymbol::isAcceleratorPointerTagLiveRange() const {
  return PrivateGetDIAValue(Symbol,
                            &IDiaSymbol::get_isAcceleratorPointerTagLiveRange);
}

bool DIARawSymbol::isAcceleratorStubFunction() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isAcceleratorStubFunction);
}

bool DIARawSymbol::isAggregated() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isAggregated);
}

bool DIARawSymbol::isIntroVirtualFunction() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_intro);
}

bool DIARawSymbol::isCVTCIL() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isCVTCIL);
}

bool DIARawSymbol::isConstructorVirtualBase() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isConstructorVirtualBase);
}

bool DIARawSymbol::isCxxReturnUdt() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isCxxReturnUdt);
}

bool DIARawSymbol::isDataAligned() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isDataAligned);
}

bool DIARawSymbol::isHLSLData() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isHLSLData);
}

bool DIARawSymbol::isHotpatchable() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isHotpatchable);
}

bool DIARawSymbol::isIndirectVirtualBaseClass() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_indirectVirtualBaseClass);
}

bool DIARawSymbol::isInterfaceUdt() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isInterfaceUdt);
}

bool DIARawSymbol::isIntrinsic() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_intrinsic);
}

bool DIARawSymbol::isLTCG() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isLTCG);
}

bool DIARawSymbol::isLocationControlFlowDependent() const {
  return PrivateGetDIAValue(Symbol,
                            &IDiaSymbol::get_isLocationControlFlowDependent);
}

bool DIARawSymbol::isMSILNetmodule() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isMSILNetmodule);
}

bool DIARawSymbol::isMatrixRowMajor() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isMatrixRowMajor);
}

bool DIARawSymbol::isManagedCode() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_managed);
}

bool DIARawSymbol::isMSILCode() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_msil);
}

bool DIARawSymbol::isMultipleInheritance() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isMultipleInheritance);
}

bool DIARawSymbol::isNaked() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isNaked);
}

bool DIARawSymbol::isNested() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_nested);
}

bool DIARawSymbol::isOptimizedAway() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isOptimizedAway);
}

bool DIARawSymbol::isPacked() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_packed);
}

bool DIARawSymbol::isPointerBasedOnSymbolValue() const {
  return PrivateGetDIAValue(Symbol,
                            &IDiaSymbol::get_isPointerBasedOnSymbolValue);
}

bool DIARawSymbol::isPointerToDataMember() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isPointerToDataMember);
}

bool DIARawSymbol::isPointerToMemberFunction() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isPointerToMemberFunction);
}

bool DIARawSymbol::isPureVirtual() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_pure);
}

bool DIARawSymbol::isRValueReference() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_RValueReference);
}

bool DIARawSymbol::isRefUdt() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isRefUdt);
}

bool DIARawSymbol::isReference() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_reference);
}

bool DIARawSymbol::isRestrictedType() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_restrictedType);
}

bool DIARawSymbol::isReturnValue() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isReturnValue);
}

bool DIARawSymbol::isSafeBuffers() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isSafeBuffers);
}

bool DIARawSymbol::isScoped() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_scoped);
}

bool DIARawSymbol::isSdl() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isSdl);
}

bool DIARawSymbol::isSingleInheritance() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isSingleInheritance);
}

bool DIARawSymbol::isSplitted() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isSplitted);
}

bool DIARawSymbol::isStatic() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isStatic);
}

bool DIARawSymbol::hasPrivateSymbols() const {
  // hasPrivateSymbols is the opposite of isStripped, but we expose
  // hasPrivateSymbols as a more intuitive interface.
  return !PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isStripped);
}

bool DIARawSymbol::isUnalignedType() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_unalignedType);
}

bool DIARawSymbol::isUnreached() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_notReached);
}

bool DIARawSymbol::isValueUdt() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isValueUdt);
}

bool DIARawSymbol::isVirtual() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_virtual);
}

bool DIARawSymbol::isVirtualBaseClass() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_virtualBaseClass);
}

bool DIARawSymbol::isVirtualInheritance() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_isVirtualInheritance);
}

bool DIARawSymbol::isVolatileType() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_volatileType);
}

bool DIARawSymbol::wasInlined() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_wasInlined);
}

std::string DIARawSymbol::getUnused() const {
  return PrivateGetDIAValue(Symbol, &IDiaSymbol::get_unused);
}
