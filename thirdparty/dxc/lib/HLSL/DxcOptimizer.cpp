///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxcOptimizer.cpp                                                          //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides an IDxcOptimizer implementation.                                 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/Support/WinIncludes.h"
#include "dxc/Support/Global.h"
#include "dxc/Support/Unicode.h"
#include "dxc/Support/microcom.h"
#include "dxc/DxilContainer/DxilContainer.h"
#include "dxc/Support/FileIOHelper.h"
#include "dxc/DXIL/DxilModule.h"
#include "llvm/Analysis/ReducibilityAnalysis.h"
#include "dxc/HLSL/HLMatrixLowerPass.h"
#include "dxc/HLSL/DxilGenerationPass.h"
#include "dxc/HLSL/ComputeViewIdState.h"
#include "llvm/Analysis/DxilValueCache.h"
#include "dxc/DXIL/DxilUtil.h"
#include "dxc/Support/dxcapi.impl.h"

#include "llvm/Pass.h"
#include "llvm/PassInfo.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include <algorithm>
#include <list>   // should change this for string_table
#include <vector>

#include "llvm/PassPrinters/PassPrinters.h"

using namespace llvm;
using namespace hlsl;

inline static bool wcseq(LPCWSTR a, LPCWSTR b) {
  return 0 == wcscmp(a, b);
}
inline static bool wcsstartswith(LPCWSTR value, LPCWSTR prefix) {
  while (*value && *prefix && *value == *prefix) {
    ++value;
    ++prefix;
  }
  return *prefix == L'\0';
}

#include "DxcOptimizer.inc"

static void FatalErrorHandlerStreamWrite(void *user_data, const std::string& reason, bool gen_crash_diag) {
  raw_ostream *OS = (raw_ostream *)user_data;
  *OS << reason;
  throw std::exception();
}

static HRESULT Utf8ToWideCoTaskMalloc(LPCSTR pValue, LPWSTR *ppResult) {
  if (ppResult == nullptr)
    return E_POINTER;
  int count = MultiByteToWideChar(CP_UTF8, 0, pValue, -1, nullptr, 0);
  *ppResult = (wchar_t*)CoTaskMemAlloc(sizeof(wchar_t) * count);
  if (*ppResult == nullptr)
    return E_OUTOFMEMORY;
  MultiByteToWideChar(CP_UTF8, 0, pValue, -1, *ppResult, count);
  return S_OK;
}

class DxcOptimizerPass : public IDxcOptimizerPass {
private:
  DXC_MICROCOM_TM_REF_FIELDS()
  LPCSTR m_pOptionName;
  LPCSTR m_pDescription;
  ArrayRef<LPCSTR> m_pArgNames;
  ArrayRef<LPCSTR> m_pArgDescriptions;
public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  DXC_MICROCOM_TM_CTOR(DxcOptimizerPass)

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) override {
    return DoBasicQueryInterface<IDxcOptimizerPass>(this, iid, ppvObject);
  }

  HRESULT Initialize(LPCSTR pOptionName, LPCSTR pDescription, ArrayRef<LPCSTR> pArgNames, ArrayRef<LPCSTR> pArgDescriptions) {
    DXASSERT(pArgNames.size() == pArgDescriptions.size(), "else lookup tables are out of alignment");
    m_pOptionName = pOptionName;
    m_pDescription = pDescription;
    m_pArgNames = pArgNames;
    m_pArgDescriptions = pArgDescriptions;
    return S_OK;
  }
  static HRESULT Create(IMalloc *pMalloc, LPCSTR pOptionName, LPCSTR pDescription, ArrayRef<LPCSTR> pArgNames, ArrayRef<LPCSTR> pArgDescriptions, IDxcOptimizerPass **ppResult) {
    CComPtr<DxcOptimizerPass> result;
    *ppResult = nullptr;
    result = DxcOptimizerPass::Alloc(pMalloc);
    IFROOM(result);
    IFR(result->Initialize(pOptionName, pDescription, pArgNames, pArgDescriptions));
    *ppResult = result.Detach();
    return S_OK;
  }

  HRESULT STDMETHODCALLTYPE GetOptionName(_COM_Outptr_ LPWSTR *ppResult) override {
    return Utf8ToWideCoTaskMalloc(m_pOptionName, ppResult);
  }
  HRESULT STDMETHODCALLTYPE GetDescription(_COM_Outptr_ LPWSTR *ppResult) override {
    return Utf8ToWideCoTaskMalloc(m_pDescription, ppResult);
  }

  HRESULT STDMETHODCALLTYPE GetOptionArgCount(_Out_ UINT32 *pCount) override {
    if (!pCount) return E_INVALIDARG;
    *pCount = m_pArgDescriptions.size();
    return S_OK;
  }

  HRESULT STDMETHODCALLTYPE GetOptionArgName(UINT32 argIndex, LPWSTR *ppResult) override {
    if (!ppResult) return E_INVALIDARG;
    if (argIndex >= m_pArgNames.size()) return E_INVALIDARG;
    return Utf8ToWideCoTaskMalloc(m_pArgNames[argIndex], ppResult);
  }
  HRESULT STDMETHODCALLTYPE GetOptionArgDescription(UINT32 argIndex, LPWSTR *ppResult) override {
    if (!ppResult) return E_INVALIDARG;
    if (argIndex >= m_pArgDescriptions.size()) return E_INVALIDARG;
    return Utf8ToWideCoTaskMalloc(m_pArgDescriptions[argIndex], ppResult);
  }
};

class DxcOptimizer : public IDxcOptimizer {
private:
  DXC_MICROCOM_TM_REF_FIELDS()
  PassRegistry *m_registry;
  std::vector<const PassInfo *> m_passes;
public:
  DXC_MICROCOM_TM_ADDREF_RELEASE_IMPL()
  DXC_MICROCOM_TM_CTOR(DxcOptimizer)

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **ppvObject) override {
    return DoBasicQueryInterface<IDxcOptimizer>(this, iid, ppvObject);
  }

  HRESULT Initialize();
  const PassInfo *getPassByID(llvm::AnalysisID PassID);
  const PassInfo *getPassByName(const char *pName);
  HRESULT STDMETHODCALLTYPE GetAvailablePassCount(_Out_ UINT32 *pCount) override {
    return AssignToOut<UINT32>(m_passes.size(), pCount);
  }
  HRESULT STDMETHODCALLTYPE GetAvailablePass(UINT32 index, _COM_Outptr_ IDxcOptimizerPass** ppResult) override;
  HRESULT STDMETHODCALLTYPE RunOptimizer(IDxcBlob *pBlob,
    _In_count_(optionCount) LPCWSTR *ppOptions, UINT32 optionCount,
    _COM_Outptr_ IDxcBlob **ppOutputModule,
    _COM_Outptr_opt_ IDxcBlobEncoding **ppOutputText) override;
};

class CapturePassManager : public llvm::legacy::PassManagerBase {
private:
  SmallVector<Pass *, 64> Passes;
public:
  ~CapturePassManager() {
    for (auto P : Passes) delete P;
  }

  void add(Pass *P) override {
    Passes.push_back(P);
  }

  size_t size() const { return Passes.size(); }
  StringRef getPassNameAt(size_t index) const {
    return Passes[index]->getPassName();
  }
  llvm::AnalysisID getPassIDAt(size_t index) const {
    return Passes[index]->getPassID();
  }
};

HRESULT DxcOptimizer::Initialize() {
  try {
    m_registry = PassRegistry::getPassRegistry();

    struct PRL : public PassRegistrationListener {
      std::vector<const PassInfo *> *Passes;
      void passEnumerate(const PassInfo * PI) override {
        DXASSERT(nullptr != PI->getNormalCtor(), "else cannot construct");
        Passes->push_back(PI);
      }
    };
    PRL prl;
    prl.Passes = &this->m_passes;
    m_registry->enumerateWith(&prl);
  }
  CATCH_CPP_RETURN_HRESULT();
  return S_OK;
}

const PassInfo *DxcOptimizer::getPassByID(llvm::AnalysisID PassID) {
  return m_registry->getPassInfo(PassID);
}

const PassInfo *DxcOptimizer::getPassByName(const char *pName) {
  return m_registry->getPassInfo(StringRef(pName));
}

HRESULT STDMETHODCALLTYPE DxcOptimizer::GetAvailablePass(
    UINT32 index, _COM_Outptr_ IDxcOptimizerPass **ppResult) {
  IFR(AssignToOut(nullptr, ppResult));
  if (index >= m_passes.size())
    return E_INVALIDARG;
  return DxcOptimizerPass::Create(
      m_pMalloc, m_passes[index]->getPassArgument(),
      m_passes[index]->getPassName().data(),
      GetPassArgNames(m_passes[index]->getPassArgument()),
      GetPassArgDescriptions(m_passes[index]->getPassArgument()), ppResult);
}

HRESULT STDMETHODCALLTYPE DxcOptimizer::RunOptimizer(
    IDxcBlob *pBlob, _In_count_(optionCount) LPCWSTR *ppOptions,
    UINT32 optionCount, _COM_Outptr_ IDxcBlob **ppOutputModule,
    _COM_Outptr_opt_ IDxcBlobEncoding **ppOutputText) {
  AssignToOutOpt(nullptr, ppOutputModule);
  AssignToOutOpt(nullptr, ppOutputText);
  if (pBlob == nullptr)
    return E_POINTER;
  if (optionCount > 0 && ppOptions == nullptr)
    return E_POINTER;

  DxcThreadMalloc TM(m_pMalloc);

  // Setup input buffer.
  //
  // The ir parsing requires the buffer to be null terminated. We deal with
  // both source and bitcode input, so the input buffer may not be null
  // terminated; we create a new membuf that copies and appends for this.
  //
  // If we have the beginning of a DXIL program header, skip to the bitcode.
  //
  LLVMContext Context;
  SMDiagnostic Err;
  std::unique_ptr<MemoryBuffer> memBuf;
  std::unique_ptr<Module> M;
  const char * pBlobContent = reinterpret_cast<const char *>(pBlob->GetBufferPointer());
  unsigned blobSize = pBlob->GetBufferSize();
  const DxilProgramHeader *pProgramHeader =
    reinterpret_cast<const DxilProgramHeader *>(pBlobContent);
  if (IsValidDxilProgramHeader(pProgramHeader, blobSize)) {
    std::string DiagStr;
    GetDxilProgramBitcode(pProgramHeader, &pBlobContent, &blobSize);
    M = hlsl::dxilutil::LoadModuleFromBitcode(
      llvm::StringRef(pBlobContent, blobSize), Context, DiagStr);
  }
  else {
    StringRef bufStrRef(pBlobContent, blobSize);
    memBuf = MemoryBuffer::getMemBufferCopy(bufStrRef);
    M = parseIR(memBuf->getMemBufferRef(), Err, Context);
  }

  if (M == nullptr) {
    return DXC_E_IR_VERIFICATION_FAILED;
  }

  legacy::PassManager ModulePasses;
  legacy::FunctionPassManager FunctionPasses(M.get());
  legacy::PassManagerBase *pPassManager = &ModulePasses;

  try {
    CComPtr<AbstractMemoryStream> pOutputStream;
    CComPtr<IDxcBlob> pOutputBlob;

    IFT(CreateMemoryStream(m_pMalloc, &pOutputStream));
    IFT(pOutputStream.QueryInterface(&pOutputBlob));

    raw_stream_ostream outStream(pOutputStream.p);

    //
    // Consider some differences from opt.exe:
    //
    // Create a new optimization pass for each one specified on the command line
    // as in StandardLinkOpts, OptLevelO1, etc.
    // No target machine, and so no passes get their target machine ctor called.
    // No print-after-each-pass option.
    // No printing of the pass options.
    // No StripDebug support.
    // No verifyModule before starting.
    // Use of PassPipeline for new manager.
    // No TargetInfo.
    // No DataLayout.
    //
    bool OutputAssembly = false;
    bool AnalyzeOnly = false;

    // First gather flags, wherever they may be.
    SmallVector<UINT32, 2> handled;
    for (UINT32 i = 0; i < optionCount; ++i) {
      if (wcseq(L"-S", ppOptions[i])) {
        OutputAssembly = true;
        handled.push_back(i);
        continue;
      }
      if (wcseq(L"-analyze", ppOptions[i])) {
        AnalyzeOnly = true;
        handled.push_back(i);
        continue;
      }
    }

    // TODO: should really use string_table for this once that's available
    std::list<std::string> optionsAnsi;
    SmallVector<PassOption, 2> options;
    for (UINT32 i = 0; i < optionCount; ++i) {
      if (std::find(handled.begin(), handled.end(), i) != handled.end()) {
        continue;
      }

      // Handle some special cases where we can inject a redirected output stream.
      if (wcsstartswith(ppOptions[i], L"-print-module")) {
        LPCWSTR pName = ppOptions[i] + _countof(L"-print-module") - 1;
        std::string Banner;
        if (*pName) {
          IFTARG(*pName != L':' || *pName != L'=');
          ++pName;
          CW2A name8(pName);
          Banner = "MODULE-PRINT ";
          Banner += name8.m_psz;
          Banner += "\n";
        }
        if (pPassManager == &ModulePasses)
          pPassManager->add(llvm::createPrintModulePass(outStream, Banner));
        continue;
      }

      // Handle special switches to toggle per-function prepasses vs. module passes.
      if (wcseq(ppOptions[i], L"-opt-fn-passes")) {
        pPassManager = &FunctionPasses;
        continue;
      }
      if (wcseq(ppOptions[i], L"-opt-mod-passes")) {
        pPassManager = &ModulePasses;
        continue;
      }

      CW2A optName(ppOptions[i], CP_UTF8);
      // The option syntax is
      const char ArgDelim = ',';
      // '-' OPTION_NAME (',' ARG_NAME ('=' ARG_VALUE)?)*
      char *pCursor = optName.m_psz;
      const char *pEnd = optName.m_psz + strlen(optName.m_psz);
      if (*pCursor != '-' && *pCursor != '/') {
        return E_INVALIDARG;
      }
      ++pCursor;
      const char *pOptionNameStart = pCursor;
      while (*pCursor && *pCursor != ArgDelim) {
        ++pCursor;
      }
      *pCursor = '\0';
      const llvm::PassInfo *PassInf = getPassByName(pOptionNameStart);
      if (!PassInf) {
        return E_INVALIDARG;
      }
      while (pCursor < pEnd) {
        // *pCursor is '\0' when we overwrite ',' to get a null-terminated string
        if (*pCursor && *pCursor != ArgDelim) {
          return E_INVALIDARG;
        }
        ++pCursor;
        const char *pArgStart = pCursor;
        while (*pCursor && *pCursor != ArgDelim) {
          ++pCursor;
        }
        StringRef argString = StringRef(pArgStart, pCursor - pArgStart);
        std::pair<StringRef, StringRef> nameValue = argString.split('=');
        if (!IsPassOptionName(nameValue.first)) {
          return E_INVALIDARG;
        }

        PassOption *OptionPos = std::lower_bound(options.begin(), options.end(), nameValue, PassOptionsCompare());
        // If empty, remove if available; otherwise upsert.
        if (nameValue.second.empty()) {
          if (OptionPos != options.end() && OptionPos->first == nameValue.first) {
            options.erase(OptionPos);
          }
        }
        else {
          if (OptionPos != options.end() && OptionPos->first == nameValue.first) {
            OptionPos->second = nameValue.second;
          }
          else {
            options.insert(OptionPos, nameValue);
          }
        }
      }

      DXASSERT(PassInf->getNormalCtor(), "else pass with no default .ctor was added");
      Pass *pass = PassInf->getNormalCtor()();
      pass->setOSOverride(&outStream);
      pass->applyOptions(options);
      options.clear();
      pPassManager->add(pass);
      if (AnalyzeOnly) {
        const bool Quiet = false;
        PassKind Kind = pass->getPassKind();
        switch (Kind) {
        case PT_BasicBlock:
          pPassManager->add(createBasicBlockPassPrinter(PassInf, outStream, Quiet));
          break;
        case PT_Region:
          pPassManager->add(createRegionPassPrinter(PassInf, outStream, Quiet));
          break;
        case PT_Loop:
          pPassManager->add(createLoopPassPrinter(PassInf, outStream, Quiet));
          break;
        case PT_Function:
          pPassManager->add(createFunctionPassPrinter(PassInf, outStream, Quiet));
          break;
        case PT_CallGraphSCC:
          pPassManager->add(createCallGraphPassPrinter(PassInf, outStream, Quiet));
          break;
        default:
          pPassManager->add(createModulePassPrinter(PassInf, outStream, Quiet));
          break;
        }
      }
    }

    ModulePasses.add(createVerifierPass());

    if (OutputAssembly) {
      ModulePasses.add(llvm::createPrintModulePass(outStream));
    }

    // Now that we have all of the passes ready, run them.
    {
      raw_ostream *err_ostream = &outStream;
      ScopedFatalErrorHandler errHandler(FatalErrorHandlerStreamWrite, err_ostream);

      FunctionPasses.doInitialization();
      for (Function &F : *M.get())
        if (!F.isDeclaration())
          FunctionPasses.run(F);
      FunctionPasses.doFinalization();
      ModulePasses.run(*M.get());
    }

    outStream.flush();
    if (ppOutputText != nullptr) {
      IFT(DxcCreateBlobWithEncodingSet(pOutputBlob, CP_UTF8, ppOutputText));
    }
    if (ppOutputModule != nullptr) {
      CComPtr<AbstractMemoryStream> pProgramStream;
      IFT(CreateMemoryStream(m_pMalloc, &pProgramStream));
      {
        raw_stream_ostream outStream(pProgramStream.p);
        WriteBitcodeToFile(M.get(), outStream, true);
      }
      IFT(pProgramStream.QueryInterface(ppOutputModule));
    }
  }
  CATCH_CPP_RETURN_HRESULT();

  return S_OK;
}

HRESULT CreateDxcOptimizer(_In_ REFIID riid, _Out_ LPVOID *ppv) {
  CComPtr<DxcOptimizer> result = DxcOptimizer::Alloc(DxcGetThreadMallocNoRef());
  if (result == nullptr) {
    *ppv = nullptr;
    return E_OUTOFMEMORY;
  }
  IFR(result->Initialize());

  return result.p->QueryInterface(riid, ppv);
}
