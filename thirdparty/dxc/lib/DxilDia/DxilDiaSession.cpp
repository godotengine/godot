///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilDiaSession.cpp                                                        //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// DIA API implementation for DXIL modules.                                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "DxilDiaSession.h"

#include "dxc/DxilPIXPasses/DxilPIXPasses.h"
#include "dxc/DxilPIXPasses/DxilPIXVirtualRegisters.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/PassRegistry.h"

#include "DxilDia.h"
#include "DxilDiaEnumTables.h"
#include "DxilDiaTable.h"
#include "DxilDiaTableInjectedSources.h"
#include "DxilDiaTableLineNumbers.h"
#include "DxilDiaTableSourceFiles.h"
#include "DxilDiaTableSymbols.h"

void dxil_dia::Session::Init(
    std::shared_ptr<llvm::LLVMContext> context,
    std::shared_ptr<llvm::Module> mod,
    std::shared_ptr<llvm::DebugInfoFinder> finder) {
  m_pEnumTables = nullptr;
  m_module = mod;
  m_context = context;
  m_finder = finder;
  m_dxilModule = llvm::make_unique<hlsl::DxilModule>(mod.get());

  llvm::legacy::PassManager PM;
  llvm::initializeDxilDbgValueToDbgDeclarePass(*llvm::PassRegistry::getPassRegistry());
  llvm::initializeDxilAnnotateWithVirtualRegisterPass(*llvm::PassRegistry::getPassRegistry());
  PM.add(llvm::createDxilDbgValueToDbgDeclarePass());
  PM.add(llvm::createDxilAnnotateWithVirtualRegisterPass());
  PM.run(*m_module);

  // Extract HLSL metadata.
  m_dxilModule->LoadDxilMetadata();

  // Get file contents.
  m_contents =
    m_module->getNamedMetadata(hlsl::DxilMDHelper::kDxilSourceContentsMDName);
  if (!m_contents)
    m_contents = m_module->getNamedMetadata("llvm.dbg.contents");

  m_defines =
    m_module->getNamedMetadata(hlsl::DxilMDHelper::kDxilSourceDefinesMDName);
  if (!m_defines)
    m_defines = m_module->getNamedMetadata("llvm.dbg.defines");

  m_mainFileName =
    m_module->getNamedMetadata(hlsl::DxilMDHelper::kDxilSourceMainFileNameMDName);
  if (!m_mainFileName)
    m_mainFileName = m_module->getNamedMetadata("llvm.dbg.mainFileName");

  m_arguments =
    m_module->getNamedMetadata(hlsl::DxilMDHelper::kDxilSourceArgsMDName);
  if (!m_arguments)
    m_arguments = m_module->getNamedMetadata("llvm.dbg.args");

  // Build up a linear list of instructions. The index will be used as the
  // RVA.
  for (llvm::Function &fn : m_module->functions()) {
    for (llvm::inst_iterator it = inst_begin(fn), end = inst_end(fn); it != end; ++it) {
      llvm::Instruction &i = *it;
      RVA rva;
      if (!pix_dxil::PixDxilInstNum::FromInst(&i, &rva)) {
        continue;
      }
      m_rvaMap.insert({ &i, rva });
      m_instructions.insert({ rva, &i});
      if (llvm::DebugLoc DL = i.getDebugLoc()) {
        auto result = m_lineToInfoMap.emplace(DL.getLine(), LineInfo(DL.getCol(), rva, rva + 1));
        if (!result.second) {
          result.first->second.StartCol = std::min(result.first->second.StartCol, DL.getCol());
          result.first->second.Last = rva + 1;
        }
        m_instructionLines.push_back(&i);
      }
    }
  }

  // Sanity check to make sure rva map is same as instruction index.
  for (auto It = m_instructions.begin(); It != m_instructions.end(); ++It) {
    DXASSERT(m_rvaMap.find(It->second) != m_rvaMap.end(), "instruction not mapped to rva");
    DXASSERT(m_rvaMap[It->second] == It->first, "instruction mapped to wrong rva");
  }

  // Initialize symbols
  try {
      m_symsMgr.Init(this);
  } catch (const hlsl::Exception &) {
      m_symsMgr = std::move(dxil_dia::SymbolManager());
  }
}

HRESULT dxil_dia::Session::getSourceFileIdByName(
    llvm::StringRef fileName,
    DWORD *pRetVal) {
  if (Contents() != nullptr) {
    for (unsigned i = 0; i < Contents()->getNumOperands(); ++i) {
      llvm::StringRef fn =
        llvm::dyn_cast<llvm::MDString>(Contents()->getOperand(i)->getOperand(0))
        ->getString();
      if (fn.equals(fileName)) {
        *pRetVal = i;
        return S_OK;
      }
    }
  }
  *pRetVal = 0;
  return S_FALSE;
}

STDMETHODIMP dxil_dia::Session::get_loadAddress(
    /* [retval][out] */ ULONGLONG *pRetVal) {
  *pRetVal = 0;
  return S_OK;
}

STDMETHODIMP dxil_dia::Session::get_globalScope(
  /* [retval][out] */ IDiaSymbol **pRetVal) {
  DxcThreadMalloc TM(m_pMalloc);

  if (pRetVal == nullptr) {
    return E_INVALIDARG;
  }
  *pRetVal = nullptr;

  Symbol *ret;
  IFR(m_symsMgr.GetGlobalScope(&ret));
  *pRetVal = ret;
  return S_OK;
}

STDMETHODIMP dxil_dia::Session::getEnumTables(
    /* [out] */ _COM_Outptr_ IDiaEnumTables **ppEnumTables) {
  if (!m_pEnumTables) {
    DxcThreadMalloc TM(m_pMalloc);
    IFR(EnumTables::Create(this, &m_pEnumTables));
  }
  m_pEnumTables.p->AddRef();
  *ppEnumTables = m_pEnumTables;
  return S_OK;
}

STDMETHODIMP dxil_dia::Session::findFileById(
    /* [in] */ DWORD uniqueId,
    /* [out] */ IDiaSourceFile **ppResult) {
  if (!m_pEnumTables) {
    return E_INVALIDARG;
  }
  CComPtr<IDiaTable> pTable;
  VARIANT vtIndex;
  vtIndex.vt = VT_UI4;
  vtIndex.uintVal = (int)Table::Kind::SourceFiles;
  IFR(m_pEnumTables->Item(vtIndex, &pTable));
  CComPtr<IUnknown> pElt;
  IFR(pTable->Item(uniqueId, &pElt));
  return pElt->QueryInterface(ppResult);
}

STDMETHODIMP dxil_dia::Session::findFile(
    /* [in] */ IDiaSymbol *pCompiland,
    /* [in] */ LPCOLESTR name,
    /* [in] */ DWORD compareFlags,
    /* [out] */ IDiaEnumSourceFiles **ppResult) {
    if (!m_pEnumTables) {
        return E_INVALIDARG;
    }
    
    // TODO: properly support compareFlags.
    auto namecmp = &_wcsicmp;
    if (compareFlags & nsCaseSensitive) {
        namecmp = &wcscmp;
    }

    DxcThreadMalloc TM(m_pMalloc);
    CComPtr<IDiaTable> pTable;
    VARIANT vtIndex;
    vtIndex.vt = VT_UI4;
    vtIndex.uintVal = (int)Table::Kind::SourceFiles;
    IFR(m_pEnumTables->Item(vtIndex, &pTable));

    CComPtr<IDiaEnumSourceFiles> pSourceTable;
    IFR(pTable->QueryInterface(&pSourceTable));
    HRESULT hr;
    CComPtr<IDiaSourceFile> src;
    ULONG cnt;
    std::vector<CComPtr<IDiaSourceFile>> sources;

    pSourceTable->Reset();
    while (SUCCEEDED(hr = pSourceTable->Next(1, &src, &cnt)) && hr == S_OK && cnt == 1) {
        CComBSTR currName;
        IFR(src->get_fileName(&currName));
        if (namecmp(name, currName) == 0) {
            sources.emplace_back(src);
        }
        src.Release();
    }

    *ppResult = CreateOnMalloc<SourceFilesTable>(
        GetMallocNoRef(),
        this,
        std::move(sources));

    if (*ppResult == nullptr) {
        return E_OUTOFMEMORY;
    }
    (*ppResult)->AddRef();
    return S_OK;
}

namespace dxil_dia {
static HRESULT DxcDiaFindLineNumbersByRVA(
  Session *pSession,
  DWORD rva,
  DWORD length,
  IDiaEnumLineNumbers **ppResult)
{
  if (!ppResult)
    return E_POINTER;

  std::vector<const llvm::Instruction*> instructions;
  auto &allInstructions = pSession->InstructionsRef();

  // Gather the list of insructions that map to the given rva range.
  for (DWORD i = rva; i < rva + length; ++i) {
    auto It = allInstructions.find(i);
    if (It == allInstructions.end())
      return E_INVALIDARG;

    // Only include the instruction if it has debug info for line mappings.
    const llvm::Instruction *inst = It->second;
    if (inst->getDebugLoc())
      instructions.push_back(inst);
  }

  // Create line number table from explicit instruction list.
  IMalloc *pMalloc = pSession->GetMallocNoRef();
  *ppResult = CreateOnMalloc<LineNumbersTable>(pMalloc, pSession, std::move(instructions));
  if (*ppResult == nullptr)
    return E_OUTOFMEMORY;
  (*ppResult)->AddRef();
  return S_OK;
}
}  // namespace dxil_dia

STDMETHODIMP dxil_dia::Session::findLinesByAddr(
  /* [in] */ DWORD seg,
  /* [in] */ DWORD offset,
  /* [in] */ DWORD length,
  /* [out] */ IDiaEnumLineNumbers **ppResult) {
  DxcThreadMalloc TM(m_pMalloc);
  return DxcDiaFindLineNumbersByRVA(this, offset, length, ppResult);
}

STDMETHODIMP dxil_dia::Session::findLinesByRVA(
  /* [in] */ DWORD rva,
  /* [in] */ DWORD length,
  /* [out] */ IDiaEnumLineNumbers **ppResult) {
  DxcThreadMalloc TM(m_pMalloc);
  return DxcDiaFindLineNumbersByRVA(this, rva, length, ppResult);
}

STDMETHODIMP dxil_dia::Session::findInlineeLinesByAddr(
  /* [in] */ IDiaSymbol *parent,
  /* [in] */ DWORD isect,
  /* [in] */ DWORD offset,
  /* [in] */ DWORD length,
  /* [out] */ IDiaEnumLineNumbers **ppResult) {
  DxcThreadMalloc TM(m_pMalloc);
  return DxcDiaFindLineNumbersByRVA(this, offset, length, ppResult);
}

STDMETHODIMP dxil_dia::Session::findLinesByLinenum(
  /* [in] */ IDiaSymbol *compiland,
  /* [in] */ IDiaSourceFile *file,
  /* [in] */ DWORD linenum,
  /* [in] */ DWORD column,
  /* [out] */ IDiaEnumLineNumbers **ppResult) {
    if (!m_pEnumTables) {
        return E_INVALIDARG;
    }
    *ppResult = nullptr;

    DxcThreadMalloc TM(m_pMalloc);
    CComPtr<IDiaTable> pTable;
    VARIANT vtIndex;
    vtIndex.vt = VT_UI4;
    vtIndex.uintVal = (int)Table::Kind::LineNumbers;
    IFR(m_pEnumTables->Item(vtIndex, &pTable));

    CComPtr<IDiaEnumLineNumbers> pLineTable;
    IFR(pTable->QueryInterface(&pLineTable));
    HRESULT hr;
    CComPtr<IDiaLineNumber> line;
    ULONG cnt;
    std::vector<const llvm::Instruction *> lines;

    std::function<bool(DWORD, DWORD)>column_matches = [column](DWORD colStart, DWORD colEnd) -> bool {
        return true;
    };

    if (column != 0) {
        column_matches = [column](DWORD colStart, DWORD colEnd) -> bool {
            return colStart < column && column < colEnd;
        };
    }

    pLineTable->Reset();
    while (SUCCEEDED(hr = pLineTable->Next(1, &line, &cnt)) && hr == S_OK && cnt == 1) {
        CComPtr<IDiaSourceFile> f;
        DWORD ln, lnEnd, cn, cnEnd;
        IFR(line->get_lineNumber(&ln));
        IFR(line->get_lineNumberEnd(&lnEnd));
        IFR(line->get_columnNumber(&cn));
        IFR(line->get_columnNumberEnd(&cnEnd));
        IFR(line->get_sourceFile(&f));
        
        if (file == f && (ln <= linenum && linenum <= lnEnd) && column_matches(cn, cnEnd)) {
            lines.emplace_back(reinterpret_cast<LineNumber*>(line.p)->Inst());
        }
        line.Release();
    }

    HRESULT result = lines.empty() ? S_FALSE : S_OK;
    *ppResult = CreateOnMalloc<LineNumbersTable>(
        GetMallocNoRef(),
        this,
        std::move(lines));

    if (*ppResult == nullptr) {
        return E_OUTOFMEMORY;
    }
    (*ppResult)->AddRef();
    return result;
}

STDMETHODIMP dxil_dia::Session::findInjectedSource(
  /* [in] */ LPCOLESTR srcFile,
  /* [out] */ IDiaEnumInjectedSources **ppResult) {
  if (Contents() != nullptr) {
    CW2A pUtf8FileName(srcFile, CP_UTF8);
    DxcThreadMalloc TM(m_pMalloc);
    IDiaTable *pTable;
    IFT(Table::Create(this, Table::Kind::InjectedSource, &pTable));
    auto *pInjectedSource =
      reinterpret_cast<InjectedSourcesTable *>(pTable);
    pInjectedSource->Init(pUtf8FileName.m_psz);
    *ppResult = pInjectedSource;
    return S_OK;
  }
  return S_FALSE;
}

static constexpr DWORD kD3DCodeSection = 1;
STDMETHODIMP dxil_dia::Session::findInlineFramesByAddr(
  /* [in] */ IDiaSymbol *parent,
  /* [in] */ DWORD isect,
  /* [in] */ DWORD offset,
  /* [out] */ IDiaEnumSymbols **ppResult) {
  if (parent != nullptr || isect != kD3DCodeSection || ppResult == nullptr) {
    return E_INVALIDARG;
  }
  *ppResult = nullptr;

  DxcThreadMalloc TM(m_pMalloc);
  auto &allInstructions = InstructionsRef();
  auto It = allInstructions.find(offset);
  if (It == allInstructions.end()) {
    return E_INVALIDARG;
  }

  HRESULT hr;
  SymbolChildrenEnumerator *ChildrenEnum;
  IFR(hr = m_symsMgr.DbgScopeOf(It->second, &ChildrenEnum));

  *ppResult = ChildrenEnum;
  return hr;
}
