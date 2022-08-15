///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaSession.h                                                          //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/Support/WinIncludes.h"

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "dia2.h"

#include "dxc/dxcpix.h"
#include "dxc/DXIL/DxilModule.h"

#include "dxc/Support/Global.h"
#include "dxc/Support/microcom.h"

#include "DxilDia.h"
#include "DxilDiaSymbolManager.h"

namespace dxil_dia {
class Session : public IDiaSession, public IDxcPixDxilDebugInfoFactory {
public:
  using RVA = unsigned;
  using RVAMap = std::map<RVA, const llvm::Instruction *>;

  struct LineInfo {
    LineInfo(std::uint32_t start_col, RVA first, RVA last)
      : StartCol(start_col),
        First(first),
        Last(last) {}

    std::uint32_t StartCol = 0;
    RVA First = 0;
    RVA Last = 0;
  };
  using LineToInfoMap = std::unordered_map<std::uint32_t, LineInfo>;

  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  DXC_MICROCOM_TM_CTOR(Session)

  IMalloc *GetMallocNoRef() { return m_pMalloc.p; }

  void Init(std::shared_ptr<llvm::LLVMContext> context,
            std::shared_ptr<llvm::Module> mod,
            std::shared_ptr<llvm::DebugInfoFinder> finder);

  llvm::NamedMDNode *Contents() { return m_contents; }
  llvm::NamedMDNode *Defines() { return m_defines; }
  llvm::NamedMDNode *MainFileName() { return m_mainFileName; }
  llvm::NamedMDNode *Arguments() { return m_arguments; }
  hlsl::DxilModule &DxilModuleRef() { return *m_dxilModule.get(); }
  llvm::Module &ModuleRef() { return *m_module.get(); }
  llvm::DebugInfoFinder &InfoRef() { return *m_finder.get(); }
  const SymbolManager &SymMgr() const { return m_symsMgr; }
  const RVAMap &InstructionsRef() const { return m_instructions; }
  const std::vector<const llvm::Instruction *> &InstructionLinesRef() const { return m_instructionLines; }
  const std::unordered_map<const llvm::Instruction *, RVA> &RvaMapRef() const { return m_rvaMap; }
  const LineToInfoMap &LineToColumnStartMapRef() const { return m_lineToInfoMap; }

  HRESULT getSourceFileIdByName(llvm::StringRef fileName, DWORD *pRetVal);

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) {
    return DoBasicQueryInterface<IDiaSession, IDxcPixDxilDebugInfoFactory>(this, iid, ppvObject);
  }

  STDMETHODIMP get_loadAddress(
    /* [retval][out] */ ULONGLONG *pRetVal) override;

  STDMETHODIMP put_loadAddress(
    /* [in] */ ULONGLONG NewVal) override { return ENotImpl(); }

  STDMETHODIMP get_globalScope(
    /* [retval][out] */ IDiaSymbol **pRetVal) override;

  STDMETHODIMP getEnumTables(
    _COM_Outptr_ IDiaEnumTables **ppEnumTables) override;

  STDMETHODIMP getSymbolsByAddr(
    /* [out] */ IDiaEnumSymbolsByAddr **ppEnumbyAddr) override { return ENotImpl(); }

  STDMETHODIMP findChildren(
    /* [in] */ IDiaSymbol *parent,
    /* [in] */ enum SymTagEnum symtag,
    /* [in] */ LPCOLESTR name,
    /* [in] */ DWORD compareFlags,
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findChildrenEx(
    /* [in] */ IDiaSymbol *parent,
    /* [in] */ enum SymTagEnum symtag,
    /* [in] */ LPCOLESTR name,
    /* [in] */ DWORD compareFlags,
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findChildrenExByAddr(
    /* [in] */ IDiaSymbol *parent,
    /* [in] */ enum SymTagEnum symtag,
    /* [in] */ LPCOLESTR name,
    /* [in] */ DWORD compareFlags,
    /* [in] */ DWORD isect,
    /* [in] */ DWORD offset,
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findChildrenExByVA(
    /* [in] */ IDiaSymbol *parent,
    /* [in] */ enum SymTagEnum symtag,
    /* [in] */ LPCOLESTR name,
    /* [in] */ DWORD compareFlags,
    /* [in] */ ULONGLONG va,
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findChildrenExByRVA(
    /* [in] */ IDiaSymbol *parent,
    /* [in] */ enum SymTagEnum symtag,
    /* [in] */ LPCOLESTR name,
    /* [in] */ DWORD compareFlags,
    /* [in] */ DWORD rva,
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findSymbolByAddr(
    /* [in] */ DWORD isect,
    /* [in] */ DWORD offset,
    /* [in] */ enum SymTagEnum symtag,
    /* [out] */ IDiaSymbol **ppSymbol) override { return ENotImpl(); }

  STDMETHODIMP findSymbolByRVA(
    /* [in] */ DWORD rva,
  /* [in] */ enum SymTagEnum symtag,
    /* [out] */ IDiaSymbol **ppSymbol) override { return ENotImpl(); }

  STDMETHODIMP findSymbolByVA(
    /* [in] */ ULONGLONG va,
  /* [in] */ enum SymTagEnum symtag,
    /* [out] */ IDiaSymbol **ppSymbol) override { return ENotImpl(); }

  STDMETHODIMP findSymbolByToken(
    /* [in] */ ULONG token,
  /* [in] */ enum SymTagEnum symtag,
    /* [out] */ IDiaSymbol **ppSymbol) override { return ENotImpl(); }

  STDMETHODIMP symsAreEquiv(
    /* [in] */ IDiaSymbol *symbolA,
    /* [in] */ IDiaSymbol *symbolB) override { return ENotImpl(); }

  STDMETHODIMP symbolById(
    /* [in] */ DWORD id,
    /* [out] */ IDiaSymbol **ppSymbol) override { return ENotImpl(); }

  STDMETHODIMP findSymbolByRVAEx(
    /* [in] */ DWORD rva,
  /* [in] */ enum SymTagEnum symtag,
    /* [out] */ IDiaSymbol **ppSymbol,
    /* [out] */ long *displacement) override { return ENotImpl(); }

  STDMETHODIMP findSymbolByVAEx(
    /* [in] */ ULONGLONG va,
  /* [in] */ enum SymTagEnum symtag,
    /* [out] */ IDiaSymbol **ppSymbol,
    /* [out] */ long *displacement) override { return ENotImpl(); }

  STDMETHODIMP findFile(
    /* [in] */ IDiaSymbol *pCompiland,
    /* [in] */ LPCOLESTR name,
    /* [in] */ DWORD compareFlags,
    /* [out] */ IDiaEnumSourceFiles **ppResult) override;

  STDMETHODIMP findFileById(
    /* [in] */ DWORD uniqueId,
    /* [out] */ IDiaSourceFile **ppResult) override;

  STDMETHODIMP findLines(
    /* [in] */ IDiaSymbol *compiland,
    /* [in] */ IDiaSourceFile *file,
    /* [out] */ IDiaEnumLineNumbers **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findLinesByAddr(
    /* [in] */ DWORD seg,
    /* [in] */ DWORD offset,
    /* [in] */ DWORD length,
    /* [out] */ IDiaEnumLineNumbers **ppResult) override;

  STDMETHODIMP findLinesByRVA(
    /* [in] */ DWORD rva,
    /* [in] */ DWORD length,
    /* [out] */ IDiaEnumLineNumbers **ppResult) override;

  STDMETHODIMP findLinesByVA(
    /* [in] */ ULONGLONG va,
    /* [in] */ DWORD length,
    /* [out] */ IDiaEnumLineNumbers **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findLinesByLinenum(
    /* [in] */ IDiaSymbol *compiland,
    /* [in] */ IDiaSourceFile *file,
    /* [in] */ DWORD linenum,
    /* [in] */ DWORD column,
    /* [out] */ IDiaEnumLineNumbers **ppResult) override;

  STDMETHODIMP findInjectedSource(
      /* [in] */ LPCOLESTR srcFile,
      /* [out] */ IDiaEnumInjectedSources **ppResult) override;

  STDMETHODIMP getEnumDebugStreams(
    /* [out] */ IDiaEnumDebugStreams **ppEnumDebugStreams) override { return ENotImpl(); }

  STDMETHODIMP findInlineFramesByAddr(
    /* [in] */ IDiaSymbol *parent,
    /* [in] */ DWORD isect,
    /* [in] */ DWORD offset,
    /* [out] */ IDiaEnumSymbols **ppResult) override;

  STDMETHODIMP findInlineFramesByRVA(
    /* [in] */ IDiaSymbol *parent,
    /* [in] */ DWORD rva,
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findInlineFramesByVA(
    /* [in] */ IDiaSymbol *parent,
    /* [in] */ ULONGLONG va,
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findInlineeLines(
    /* [in] */ IDiaSymbol *parent,
    /* [out] */ IDiaEnumLineNumbers **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findInlineeLinesByAddr(
    /* [in] */ IDiaSymbol *parent,
    /* [in] */ DWORD isect,
    /* [in] */ DWORD offset,
    /* [in] */ DWORD length,
    /* [out] */ IDiaEnumLineNumbers **ppResult) override;

  STDMETHODIMP findInlineeLinesByRVA(
    /* [in] */ IDiaSymbol *parent,
    /* [in] */ DWORD rva,
    /* [in] */ DWORD length,
    /* [out] */ IDiaEnumLineNumbers **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findInlineeLinesByVA(
    /* [in] */ IDiaSymbol *parent,
    /* [in] */ ULONGLONG va,
    /* [in] */ DWORD length,
    /* [out] */ IDiaEnumLineNumbers **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findInlineeLinesByLinenum(
    /* [in] */ IDiaSymbol *compiland,
    /* [in] */ IDiaSourceFile *file,
    /* [in] */ DWORD linenum,
    /* [in] */ DWORD column,
    /* [out] */ IDiaEnumLineNumbers **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findInlineesByName(
    /* [in] */ LPCOLESTR name,
    /* [in] */ DWORD option,
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findAcceleratorInlineeLinesByLinenum(
    /* [in] */ IDiaSymbol *parent,
    /* [in] */ IDiaSourceFile *file,
    /* [in] */ DWORD linenum,
    /* [in] */ DWORD column,
    /* [out] */ IDiaEnumLineNumbers **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findSymbolsForAcceleratorPointerTag(
    /* [in] */ IDiaSymbol *parent,
    /* [in] */ DWORD tagValue,
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findSymbolsByRVAForAcceleratorPointerTag(
    /* [in] */ IDiaSymbol *parent,
    /* [in] */ DWORD tagValue,
    /* [in] */ DWORD rva,
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findAcceleratorInlineesByName(
    /* [in] */ LPCOLESTR name,
    /* [in] */ DWORD option,
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP addressForVA(
    /* [in] */ ULONGLONG va,
    /* [out] */ DWORD *pISect,
    /* [out] */ DWORD *pOffset) override { return ENotImpl(); }

  STDMETHODIMP addressForRVA(
    /* [in] */ DWORD rva,
    /* [out] */ DWORD *pISect,
    /* [out] */ DWORD *pOffset) override { return ENotImpl(); }

  STDMETHODIMP findILOffsetsByAddr(
    /* [in] */ DWORD isect,
    /* [in] */ DWORD offset,
    /* [in] */ DWORD length,
    /* [out] */ IDiaEnumLineNumbers **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findILOffsetsByRVA(
    /* [in] */ DWORD rva,
    /* [in] */ DWORD length,
    /* [out] */ IDiaEnumLineNumbers **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findILOffsetsByVA(
    /* [in] */ ULONGLONG va,
    /* [in] */ DWORD length,
    /* [out] */ IDiaEnumLineNumbers **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findInputAssemblyFiles(
    /* [out] */ IDiaEnumInputAssemblyFiles **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findInputAssembly(
    /* [in] */ DWORD index,
    /* [out] */ IDiaInputAssemblyFile **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findInputAssemblyById(
    /* [in] */ DWORD uniqueId,
    /* [out] */ IDiaInputAssemblyFile **ppResult) override { return ENotImpl(); }

  STDMETHODIMP getFuncMDTokenMapSize(
    /* [out] */ DWORD *pcb) override { return ENotImpl(); }

  STDMETHODIMP getFuncMDTokenMap(
    /* [in] */ DWORD cb,
    /* [out] */ DWORD *pcb,
    /* [size_is][out] */ BYTE *pb) override { return ENotImpl(); }

  STDMETHODIMP getTypeMDTokenMapSize(
    /* [out] */ DWORD *pcb) override { return ENotImpl(); }

  STDMETHODIMP getTypeMDTokenMap(
    /* [in] */ DWORD cb,
    /* [out] */ DWORD *pcb,
    /* [size_is][out] */ BYTE *pb) override { return ENotImpl(); }

  STDMETHODIMP getNumberOfFunctionFragments_VA(
    /* [in] */ ULONGLONG vaFunc,
    /* [in] */ DWORD cbFunc,
    /* [out] */ DWORD *pNumFragments) override { return ENotImpl(); }

  STDMETHODIMP getNumberOfFunctionFragments_RVA(
    /* [in] */ DWORD rvaFunc,
    /* [in] */ DWORD cbFunc,
    /* [out] */ DWORD *pNumFragments) override { return ENotImpl(); }

  STDMETHODIMP getFunctionFragments_VA(
    /* [in] */ ULONGLONG vaFunc,
    /* [in] */ DWORD cbFunc,
    /* [in] */ DWORD cFragments,
    /* [size_is][out] */ ULONGLONG *pVaFragment,
    /* [size_is][out] */ DWORD *pLenFragment) override { return ENotImpl(); }

  STDMETHODIMP getFunctionFragments_RVA(
    /* [in] */ DWORD rvaFunc,
    /* [in] */ DWORD cbFunc,
    /* [in] */ DWORD cFragments,
    /* [size_is][out] */ DWORD *pRvaFragment,
    /* [size_is][out] */ DWORD *pLenFragment) override { return ENotImpl(); }

  STDMETHODIMP getExports(
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP getHeapAllocationSites(
    /* [out] */ IDiaEnumSymbols **ppResult) override { return ENotImpl(); }

  STDMETHODIMP findInputAssemblyFile(
    /* [in] */ IDiaSymbol *pSymbol,
    /* [out] */ IDiaInputAssemblyFile **ppResult) override { return ENotImpl(); }

  STDMETHODIMP NewDxcPixDxilDebugInfo(
      _COM_Outptr_ IDxcPixDxilDebugInfo** ppDxilDebugInfo) override;

  STDMETHODIMP NewDxcPixCompilationInfo(
      _COM_Outptr_ IDxcPixCompilationInfo **ppCompilationInfo) override;

private:
  DXC_MICROCOM_TM_REF_FIELDS()
  std::shared_ptr<llvm::LLVMContext> m_context;
  std::shared_ptr<llvm::Module> m_module;
  std::shared_ptr<llvm::DebugInfoFinder> m_finder;
  std::unique_ptr<hlsl::DxilModule> m_dxilModule;
  llvm::NamedMDNode *m_contents;
  llvm::NamedMDNode *m_defines;
  llvm::NamedMDNode *m_mainFileName;
  llvm::NamedMDNode *m_arguments;
  RVAMap m_instructions;
  std::vector<const llvm::Instruction *> m_instructionLines; // Instructions with line info.
  std::unordered_map<const llvm::Instruction *, RVA> m_rvaMap; // Map instruction to its RVA.
  LineToInfoMap m_lineToInfoMap;
  SymbolManager m_symsMgr;

private:
  CComPtr<IDiaEnumTables> m_pEnumTables;
};
}  // namespace dxil_dia
