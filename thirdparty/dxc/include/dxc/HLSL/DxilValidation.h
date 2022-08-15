///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilValidation.h                                                          //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// This file provides support for validating DXIL shaders.                   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>
#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilConstants.h"
#include "dxc/Support/WinAdapter.h"

namespace llvm {
class Module;
class LLVMContext;
class raw_ostream;
class DiagnosticPrinter;
class DiagnosticInfo;
}

namespace hlsl {

#include "dxc/HLSL/DxilValidation.inc"

const char *GetValidationRuleText(ValidationRule value);
void GetValidationVersion(_Out_ unsigned *pMajor, _Out_ unsigned *pMinor);
HRESULT ValidateDxilModule(_In_ llvm::Module *pModule,
                           _In_opt_ llvm::Module *pDebugModule);

// DXIL Container Verification Functions (return false on failure)

bool VerifySignatureMatches(_In_ llvm::Module *pModule,
                            hlsl::DXIL::SignatureKind SigKind,
                            _In_reads_bytes_(SigSize) const void *pSigData,
                            _In_ uint32_t SigSize);

// PSV = data for Pipeline State Validation
bool VerifyPSVMatches(_In_ llvm::Module *pModule,
                      _In_reads_bytes_(PSVSize) const void *pPSVData,
                      _In_ uint32_t PSVSize);

// PSV = data for Pipeline State Validation
bool VerifyRDATMatches(_In_ llvm::Module *pModule,
                       _In_reads_bytes_(RDATSize) const void *pRDATData,
                       _In_ uint32_t RDATSize);

bool VerifyFeatureInfoMatches(_In_ llvm::Module *pModule,
                              _In_reads_bytes_(FeatureInfoSize) const void *pFeatureInfoData,
                              _In_ uint32_t FeatureInfoSize);

// Validate the container parts, assuming supplied module is valid, loaded from the container provided
struct DxilContainerHeader;
HRESULT ValidateDxilContainerParts(_In_ llvm::Module *pModule,
                                   _In_opt_ llvm::Module *pDebugModule,
                                   _In_reads_bytes_(ContainerSize) const DxilContainerHeader *pContainer,
                                   _In_ uint32_t ContainerSize);

// Loads module, validating load, but not module.
HRESULT ValidateLoadModule(_In_reads_bytes_(ILLength) const char *pIL,
                           _In_ uint32_t ILLength,
                           _In_ std::unique_ptr<llvm::Module> &pModule,
                           _In_ llvm::LLVMContext &Ctx,
                           _In_ llvm::raw_ostream &DiagStream,
                           _In_ unsigned bLazyLoad);

// Loads module from container, validating load, but not module.
HRESULT ValidateLoadModuleFromContainer(
    _In_reads_bytes_(ContainerSize) const void *pContainer,
    _In_ uint32_t ContainerSize, _In_ std::unique_ptr<llvm::Module> &pModule,
    _In_ std::unique_ptr<llvm::Module> &pDebugModule,
    _In_ llvm::LLVMContext &Ctx, llvm::LLVMContext &DbgCtx,
    _In_ llvm::raw_ostream &DiagStream);
// Lazy loads module from container, validating load, but not module.
HRESULT ValidateLoadModuleFromContainerLazy(
    _In_reads_bytes_(ContainerSize) const void *pContainer,
    _In_ uint32_t ContainerSize, _In_ std::unique_ptr<llvm::Module> &pModule,
    _In_ std::unique_ptr<llvm::Module> &pDebugModule,
    _In_ llvm::LLVMContext &Ctx, llvm::LLVMContext &DbgCtx,
    _In_ llvm::raw_ostream &DiagStream);

// Load and validate Dxil module from bitcode.
HRESULT ValidateDxilBitcode(_In_reads_bytes_(ILLength) const char *pIL,
                            _In_ uint32_t ILLength,
                            _In_ llvm::raw_ostream &DiagStream);

// Full container validation, including ValidateDxilModule
HRESULT ValidateDxilContainer(_In_reads_bytes_(ContainerSize) const void *pContainer,
                              _In_ uint32_t ContainerSize,
                              _In_ llvm::raw_ostream &DiagStream);

// Full container validation, including ValidateDxilModule, with debug module
HRESULT ValidateDxilContainer(_In_reads_bytes_(ContainerSize) const void *pContainer,
                              _In_ uint32_t ContainerSize,
                              const void *pOptDebugBitcode,
                              uint32_t OptDebugBitcodeSize,
                              _In_ llvm::raw_ostream &DiagStream);

class PrintDiagnosticContext {
private:
  llvm::DiagnosticPrinter &m_Printer;
  bool m_errorsFound;
  bool m_warningsFound;

public:
  PrintDiagnosticContext(llvm::DiagnosticPrinter &printer);

  bool HasErrors() const;
  bool HasWarnings() const;
  void Handle(const llvm::DiagnosticInfo &DI);

  static void PrintDiagnosticHandler(const llvm::DiagnosticInfo &DI,
                                     void *Context);
};
}
