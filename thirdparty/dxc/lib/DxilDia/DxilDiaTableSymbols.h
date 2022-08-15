///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaTableSymbols.h                                                     //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/Support/WinIncludes.h"

#include <vector>

#include "dia2.h"

#include "dxc/Support/Global.h"
#include "dxc/Support/microcom.h"

#include "DxilDia.h"
#include "DxilDiaTable.h"

namespace dxil_dia {
class Session;

class Symbol : public IDiaSymbol {
  DXC_MICROCOM_TM_REF_FIELDS()

protected:
  DXC_MICROCOM_TM_CTOR_ONLY(Symbol)

    CComPtr<Session> m_pSession;

private:
  DWORD m_ID;
  DWORD m_symTag;
  bool m_hasLexicalParent = false;
  DWORD m_lexicalParent = 0;
  bool m_hasIsHLSLData = false;
  bool m_isHLSLData = false;
  bool m_hasDataKind = false;
  DWORD m_dataKind = 0;
  bool m_hasSourceFileName = false;
  CComBSTR m_sourceFileName;
  bool m_hasName = false;
  CComBSTR m_name;
  bool m_hasValue = false;
  CComVariant m_value;

public:
  virtual ~Symbol();
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) final {
    return DoBasicQueryInterface<IDiaSymbol>(this, iid, ppvObject);
  }

  void Init(Session *pSession, DWORD index, DWORD symTag);

  DWORD GetID() const { return m_ID; }

  void SetDataKind(DWORD value) { m_hasDataKind = true; m_dataKind = value; }
  void SetLexicalParent(DWORD value) { m_hasLexicalParent = true; m_lexicalParent = value; }
  bool HasName() const { return m_hasName; }
  void SetName(LPCWSTR value) { m_hasName = true; m_name = value; }
  void SetValue(LPCSTR value) { m_hasValue = true; m_value = value; }
  void SetValue(VARIANT *pValue) { m_hasValue = true; m_value.Copy(pValue); }
  void SetValue(unsigned value) { m_hasValue = true; m_value = value; }
  void SetSourceFileName(BSTR value) { m_hasSourceFileName = true; m_sourceFileName = value; }
  void SetIsHLSLData(bool value) { m_hasIsHLSLData = true; m_isHLSLData = value; }

  virtual HRESULT GetChildren(std::vector<CComPtr<Symbol>> *children) = 0;

#pragma region IDiaSymbol implementation.
  STDMETHODIMP get_symIndexId(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_symTag(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_name(
    /* [retval][out] */ BSTR *pRetVal) override;

  STDMETHODIMP get_lexicalParent(
    /* [retval][out] */ IDiaSymbol **pRetVal) override;

  STDMETHODIMP get_classParent(
    /* [retval][out] */ IDiaSymbol **pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_type(
    /* [retval][out] */ IDiaSymbol **pRetVal) override;

  STDMETHODIMP get_dataKind(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_locationType(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_addressSection(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_addressOffset(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_relativeVirtualAddress(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_virtualAddress(
    /* [retval][out] */ ULONGLONG *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_registerId(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_offset(
    /* [retval][out] */ LONG *pRetVal) override;

  STDMETHODIMP get_length(
    /* [retval][out] */ ULONGLONG *pRetVal) override;

  STDMETHODIMP get_slot(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_volatileType(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_constType(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_unalignedType(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_access(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_libraryName(
    /* [retval][out] */ BSTR *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_platform(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_language(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_editAndContinueEnabled(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_frontEndMajor(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_frontEndMinor(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_frontEndBuild(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_backEndMajor(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_backEndMinor(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_backEndBuild(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_sourceFileName(
    /* [retval][out] */ BSTR *pRetVal) override;

  STDMETHODIMP get_unused(
    /* [retval][out] */ BSTR *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_thunkOrdinal(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_thisAdjust(
    /* [retval][out] */ LONG *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_virtualBaseOffset(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_virtual(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_intro(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_pure(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_callingConvention(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_value(
    /* [retval][out] */ VARIANT *pRetVal) override;

  STDMETHODIMP get_baseType(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_token(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_timeStamp(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_guid(
    /* [retval][out] */ GUID *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_symbolsFileName(
    /* [retval][out] */ BSTR *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_reference(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_count(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_bitPosition(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_arrayIndexType(
    /* [retval][out] */ IDiaSymbol **pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_packed(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_constructor(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_overloadedOperator(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_nested(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_hasNestedTypes(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_hasAssignmentOperator(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_hasCastOperator(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_scoped(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_virtualBaseClass(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_indirectVirtualBaseClass(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_virtualBasePointerOffset(
    /* [retval][out] */ LONG *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_virtualTableShape(
    /* [retval][out] */ IDiaSymbol **pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_lexicalParentId(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_classParentId(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_typeId(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_arrayIndexTypeId(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_virtualTableShapeId(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_code(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_function(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_managed(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_msil(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_virtualBaseDispIndex(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_undecoratedName(
    /* [retval][out] */ BSTR *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_age(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_signature(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_compilerGenerated(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_addressTaken(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_rank(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_lowerBound(
    /* [retval][out] */ IDiaSymbol **pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_upperBound(
    /* [retval][out] */ IDiaSymbol **pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_lowerBoundId(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_upperBoundId(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_dataBytes(
    /* [in] */ DWORD cbData,
    /* [out] */ DWORD *pcbData,
    /* [size_is][out] */ BYTE *pbData) override { return ENotImpl(); }

  STDMETHODIMP findChildren(
    /* [in] */ enum SymTagEnum symtag,
    /* [in] */ LPCOLESTR name,
    /* [in] */ DWORD compareFlags,
    /* [out] */ IDiaEnumSymbols **ppResult) override;

  STDMETHODIMP findChildrenEx(
    /* [in] */ enum SymTagEnum symtag,
    /* [in] */ LPCOLESTR name,
    /* [in] */ DWORD compareFlags,
    /* [out] */ IDiaEnumSymbols **ppResult) override;

  STDMETHODIMP findChildrenExByAddr(
    /* [in] */ enum SymTagEnum symtag,
    /* [in] */ LPCOLESTR name,
    /* [in] */ DWORD compareFlags,
    /* [in] */ DWORD isect,
    /* [in] */ DWORD offset,
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findChildrenExByVA(
    /* [in] */ enum SymTagEnum symtag,
    /* [in] */ LPCOLESTR name,
    /* [in] */ DWORD compareFlags,
    /* [in] */ ULONGLONG va,
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findChildrenExByRVA(
    /* [in] */ enum SymTagEnum symtag,
    /* [in] */ LPCOLESTR name,
    /* [in] */ DWORD compareFlags,
    /* [in] */ DWORD rva,
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP get_targetSection(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_targetOffset(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_targetRelativeVirtualAddress(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_targetVirtualAddress(
    /* [retval][out] */ ULONGLONG *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_machineType(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_oemId(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_oemSymbolId(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_types(
    /* [in] */ DWORD cTypes,
    /* [out] */ DWORD *pcTypes,
    /* [size_is][size_is][out] */ IDiaSymbol **pTypes) override { return ENotImpl(); }

  STDMETHODIMP get_typeIds(
    /* [in] */ DWORD cTypeIds,
    /* [out] */ DWORD *pcTypeIds,
    /* [size_is][out] */ DWORD *pdwTypeIds) override { return ENotImpl(); }

  STDMETHODIMP get_objectPointerType(
    /* [retval][out] */ IDiaSymbol **pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_udtKind(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_undecoratedNameEx(
    /* [in] */ DWORD undecorateOptions,
    /* [out] */ BSTR *name) override { return ENotImpl(); }

  STDMETHODIMP get_noReturn(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_customCallingConvention(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_noInline(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_optimizedCodeDebugInfo(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_notReached(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_interruptReturn(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_farReturn(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isStatic(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_hasDebugInfo(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isLTCG(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isDataAligned(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_hasSecurityChecks(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_compilerName(
    /* [retval][out] */ BSTR *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_hasAlloca(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_hasSetJump(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_hasLongJump(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_hasInlAsm(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_hasEH(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_hasSEH(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_hasEHa(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isNaked(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isAggregated(
    /* [retval][out] */ BOOL *pRetVal) override;

  STDMETHODIMP get_isSplitted(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_container(
    /* [retval][out] */ IDiaSymbol **pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_inlSpec(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_noStackOrdering(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_virtualBaseTableType(
    /* [retval][out] */ IDiaSymbol **pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_hasManagedCode(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isHotpatchable(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isCVTCIL(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isMSILNetmodule(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isCTypes(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isStripped(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_frontEndQFE(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_backEndQFE(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_wasInlined(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_strictGSCheck(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isCxxReturnUdt(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isConstructorVirtualBase(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_RValueReference(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_unmodifiedType(
    /* [retval][out] */ IDiaSymbol **pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_framePointerPresent(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isSafeBuffers(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_intrinsic(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_sealed(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_hfaFloat(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_hfaDouble(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_liveRangeStartAddressSection(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_liveRangeStartAddressOffset(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_liveRangeStartRelativeVirtualAddress(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_countLiveRanges(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_liveRangeLength(
    /* [retval][out] */ ULONGLONG *pRetVal) override;

  STDMETHODIMP get_offsetInUdt(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_paramBasePointerRegisterId(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_localBasePointerRegisterId(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isLocationControlFlowDependent(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_stride(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_numberOfRows(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_numberOfColumns(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isMatrixRowMajor(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_numericProperties(
    /* [in] */ DWORD cnt,
    /* [out] */ DWORD *pcnt,
    /* [size_is][out] */ DWORD *pProperties) override;

  STDMETHODIMP get_modifierValues(
    /* [in] */ DWORD cnt,
    /* [out] */ DWORD *pcnt,
    /* [size_is][out] */ WORD *pModifiers) override { return ENotImpl(); }

  STDMETHODIMP get_isReturnValue(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isOptimizedAway(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_builtInKind(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_registerType(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_baseDataSlot(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_baseDataOffset(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_textureSlot(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_samplerSlot(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_uavSlot(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_sizeInUdt(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_memorySpaceKind(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_unmodifiedTypeId(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_subTypeId(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_subType(
    /* [retval][out] */ IDiaSymbol **pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_numberOfModifiers(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_numberOfRegisterIndices(
    /* [retval][out] */ DWORD *pRetVal) override;

  STDMETHODIMP get_isHLSLData(
    /* [retval][out] */ BOOL *pRetVal) override;

  STDMETHODIMP get_isPointerToDataMember(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isPointerToMemberFunction(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isSingleInheritance(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isMultipleInheritance(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isVirtualInheritance(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_restrictedType(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isPointerBasedOnSymbolValue(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_baseSymbol(
    /* [retval][out] */ IDiaSymbol **pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_baseSymbolId(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_objectFileName(
    /* [retval][out] */ BSTR *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isAcceleratorGroupSharedLocal(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isAcceleratorPointerTagLiveRange(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isAcceleratorStubFunction(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_numberOfAcceleratorPointerTags(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isSdl(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isWinRTPointer(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isRefUdt(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isValueUdt(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isInterfaceUdt(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP findInlineFramesByAddr(
    /* [in] */ DWORD isect,
    /* [in] */ DWORD offset,
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findInlineFramesByRVA(
    /* [in] */ DWORD rva,
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findInlineFramesByVA(
    /* [in] */ ULONGLONG va,
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findInlineeLines(
    /* [out] */ IDiaEnumLineNumbers **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findInlineeLinesByAddr(
    /* [in] */ DWORD isect,
    /* [in] */ DWORD offset,
    /* [in] */ DWORD length,
    /* [out] */ IDiaEnumLineNumbers **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findInlineeLinesByRVA(
    /* [in] */ DWORD rva,
    /* [in] */ DWORD length,
    /* [out] */ IDiaEnumLineNumbers **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findInlineeLinesByVA(
    /* [in] */ ULONGLONG va,
    /* [in] */ DWORD length,
    /* [out] */ IDiaEnumLineNumbers **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findSymbolsForAcceleratorPointerTag(
    /* [in] */ DWORD tagValue,
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findSymbolsByRVAForAcceleratorPointerTag(
    /* [in] */ DWORD tagValue,
    /* [in] */ DWORD rva,
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP get_acceleratorPointerTags(
    /* [in] */ DWORD cnt,
    /* [out] */ DWORD *pcnt,
    /* [size_is][out] */ DWORD *pPointerTags) override { return ENotImpl(); }

  STDMETHODIMP getSrcLineOnTypeDefn(
    /* [out] */ IDiaLineNumber **ppResult) override { return ENotImpl(); }

  STDMETHODIMP get_isPGO(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_hasValidPGOCounts(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_isOptimizedForSpeed(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_PGOEntryCount(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_PGOEdgeCount(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_PGODynamicInstructionCount(
    /* [retval][out] */ ULONGLONG *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_staticSize(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_finalLiveStaticSize(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_phaseName(
    /* [retval][out] */ BSTR *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_hasControlFlowCheck(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_constantExport(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_dataExport(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_privateExport(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_noNameExport(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_exportHasExplicitlyAssignedOrdinal(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_exportIsForwarder(
    /* [retval][out] */ BOOL *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_ordinal(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_frameSize(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_exceptionHandlerAddressSection(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_exceptionHandlerAddressOffset(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_exceptionHandlerRelativeVirtualAddress(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_exceptionHandlerVirtualAddress(
    /* [retval][out] */ ULONGLONG *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP findInputAssemblyFile(
    /* [out] */ IDiaInputAssemblyFile **ppResult) override { return ENotImpl(); }

  STDMETHODIMP get_characteristics(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_coffGroup(
    /* [retval][out] */ IDiaSymbol **pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_bindID(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_bindSpace(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

  STDMETHODIMP get_bindSlot(
    /* [retval][out] */ DWORD *pRetVal) override { return ENotImpl(); }

#pragma endregion
};

class SymbolChildrenEnumerator : public IDiaEnumSymbols {
public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  DXC_MICROCOM_TM_CTOR(SymbolChildrenEnumerator)
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) {
    return DoBasicQueryInterface<IDiaEnumSymbols>(this, iid, ppvObject);
  }

  void Init(std::vector<CComPtr<Symbol>> &&syms);

#pragma region IDiaEnumSymbols implementation
  /* [id][helpstring][propget] */ HRESULT STDMETHODCALLTYPE get__NewEnum(
    /* [retval][out] */ IUnknown **pRetVal) override { return ENotImpl(); }

  /* [id][helpstring][propget] */ HRESULT STDMETHODCALLTYPE get_Count(
    /* [retval][out] */ LONG *pRetVal) override;

  /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Item(
    /* [in] */ DWORD index,
    /* [retval][out] */ IDiaSymbol **symbol) override;

  HRESULT STDMETHODCALLTYPE Next(
    /* [in] */ ULONG celt,
    /* [out] */ IDiaSymbol **rgelt,
    /* [out] */ ULONG *pceltFetched) override;

  HRESULT STDMETHODCALLTYPE Skip(
    /* [in] */ ULONG celt) override { return ENotImpl(); }

  HRESULT STDMETHODCALLTYPE Reset(void) override;

  HRESULT STDMETHODCALLTYPE Clone(
    /* [out] */ IDiaEnumSymbols **ppenum) override { return ENotImpl(); }
#pragma endregion

private:
  DXC_MICROCOM_TM_REF_FIELDS()
  std::vector<CComPtr<Symbol>> m_symbols;
  std::vector<CComPtr<Symbol>>::iterator m_pos;
};

class SymbolsTable : public impl::TableBase<IDiaEnumSymbols, IDiaSymbol> {
public:
  SymbolsTable(IMalloc *pMalloc, Session *pSession);

  HRESULT GetItem(DWORD index, IDiaSymbol **ppItem) override;
};

}  // namespace dxil_dia
