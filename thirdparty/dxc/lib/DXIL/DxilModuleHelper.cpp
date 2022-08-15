///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilModuleHelper.cpp                                                      //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilConstants.h"
#include "dxc/DXIL/DxilShaderModel.h"
#include "dxc/DXIL/DxilSignatureElement.h"
#include "dxc/DXIL/DxilFunctionProps.h"
#include "dxc/Support/WinAdapter.h"
#include "dxc/DXIL/DxilEntryProps.h"
#include "dxc/DXIL/DxilSubobject.h"
#include "dxc/DXIL/DxilInstructions.h"
#include "dxc/DXIL/DxilCounters.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include <unordered_set>

#ifndef _WIN32
using llvm::make_unique;
#else
using std::make_unique;
#endif


using namespace llvm;
using std::string;
using std::vector;
using std::unique_ptr;


namespace {
class DxilErrorDiagnosticInfo : public DiagnosticInfo {
private:
  const char *m_message;
public:
  DxilErrorDiagnosticInfo(const char *str)
    : DiagnosticInfo(DK_FirstPluginKind, DiagnosticSeverity::DS_Error),
    m_message(str) { }

  void print(DiagnosticPrinter &DP) const override {
    DP << m_message;
  }
};
} // anon namespace

namespace hlsl {

// Avoid dependency on DxilModule from llvm::Module using this:
void DxilModule_RemoveGlobal(llvm::Module* M, llvm::GlobalObject* G) {
  if (M && G && M->HasDxilModule()) {
    if (llvm::Function *F = dyn_cast<llvm::Function>(G))
      M->GetDxilModule().RemoveFunction(F);
  }
}
void DxilModule_ResetModule(llvm::Module* M) {
  if (M && M->HasDxilModule())
    delete &M->GetDxilModule();
  M->SetDxilModule(nullptr);
}

void SetDxilHook(Module &M) {
  M.pfnRemoveGlobal = &DxilModule_RemoveGlobal;
  M.pfnResetDxilModule = &DxilModule_ResetModule;
}

void ClearDxilHook(Module &M) {
  if (M.pfnRemoveGlobal == &DxilModule_RemoveGlobal)
    M.pfnRemoveGlobal = nullptr;
}

hlsl::DxilModule *hlsl::DxilModule::TryGetDxilModule(llvm::Module *pModule) {
  LLVMContext &Ctx = pModule->getContext();
  std::string diagStr;
  raw_string_ostream diagStream(diagStr);

  hlsl::DxilModule *pDxilModule = nullptr;
  // TODO: add detail error in DxilMDHelper.
  try {
    pDxilModule = &pModule->GetOrCreateDxilModule();
  } catch (const ::hlsl::Exception &hlslException) {
    diagStream << "load dxil metadata failed -";
    try {
      const char *msg = hlslException.what();
      if (msg == nullptr || *msg == '\0')
        diagStream << " error code " << hlslException.hr << "\n";
      else
        diagStream << msg;
    } catch (...) {
      diagStream << " unable to retrieve error message.\n";
    }
    Ctx.diagnose(DxilErrorDiagnosticInfo(diagStream.str().c_str()));
  } catch (...) {
    Ctx.diagnose(DxilErrorDiagnosticInfo("load dxil metadata failed - unknown error.\n"));
  }
  return pDxilModule;
}

} // namespace hlsl

namespace llvm {
hlsl::DxilModule &Module::GetOrCreateDxilModule(bool skipInit) {
  std::unique_ptr<hlsl::DxilModule> M;
  if (!HasDxilModule()) {
    M = make_unique<hlsl::DxilModule>(this);
    if (!skipInit) {
      M->LoadDxilMetadata();
    }
    SetDxilModule(M.release());
  }
  return GetDxilModule();
}

}
