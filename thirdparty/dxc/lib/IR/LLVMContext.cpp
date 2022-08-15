//===-- LLVMContext.cpp - Implement LLVMContext ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements LLVMContext, as a wrapper around the opaque
//  class LLVMContextImpl.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LLVMContext.h"
#include "LLVMContextImpl.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/SourceMgr.h"
#include <cctype>
using namespace llvm;

static ManagedStatic<LLVMContext> GlobalContext;

LLVMContext& llvm::getGlobalContext() {
  return *GlobalContext;
}

LLVMContext::LLVMContext() : pImpl(new LLVMContextImpl(*this)) {
  std::unique_ptr<LLVMContextImpl> implPtrGuard(pImpl); // HLSL Change: Don't leak if constructor throws.
  // Create the fixed metadata kinds. This is done in the same order as the
  // MD_* enum values so that they correspond.

  // Create the 'dbg' metadata kind.
  unsigned DbgID = getMDKindID("dbg");
  assert(DbgID == MD_dbg && "dbg kind id drifted"); (void)DbgID;

  // Create the 'tbaa' metadata kind.
  unsigned TBAAID = getMDKindID("tbaa");
  assert(TBAAID == MD_tbaa && "tbaa kind id drifted"); (void)TBAAID;

  // Create the 'prof' metadata kind.
  unsigned ProfID = getMDKindID("prof");
  assert(ProfID == MD_prof && "prof kind id drifted"); (void)ProfID;

  // Create the 'fpmath' metadata kind.
  unsigned FPAccuracyID = getMDKindID("fpmath");
  assert(FPAccuracyID == MD_fpmath && "fpmath kind id drifted");
  (void)FPAccuracyID;

  // Create the 'range' metadata kind.
  unsigned RangeID = getMDKindID("range");
  assert(RangeID == MD_range && "range kind id drifted");
  (void)RangeID;

  // Create the 'tbaa.struct' metadata kind.
  unsigned TBAAStructID = getMDKindID("tbaa.struct");
  assert(TBAAStructID == MD_tbaa_struct && "tbaa.struct kind id drifted");
  (void)TBAAStructID;

  // Create the 'invariant.load' metadata kind.
  unsigned InvariantLdId = getMDKindID("invariant.load");
  assert(InvariantLdId == MD_invariant_load && "invariant.load kind id drifted");
  (void)InvariantLdId;

  // Create the 'alias.scope' metadata kind.
  unsigned AliasScopeID = getMDKindID("alias.scope");
  assert(AliasScopeID == MD_alias_scope && "alias.scope kind id drifted");
  (void)AliasScopeID;

  // Create the 'noalias' metadata kind.
  unsigned NoAliasID = getMDKindID("noalias");
  assert(NoAliasID == MD_noalias && "noalias kind id drifted");
  (void)NoAliasID;

  // Create the 'nontemporal' metadata kind.
  unsigned NonTemporalID = getMDKindID("nontemporal");
  assert(NonTemporalID == MD_nontemporal && "nontemporal kind id drifted");
  (void)NonTemporalID;

  // Create the 'llvm.mem.parallel_loop_access' metadata kind.
  unsigned MemParallelLoopAccessID = getMDKindID("llvm.mem.parallel_loop_access");
  assert(MemParallelLoopAccessID == MD_mem_parallel_loop_access &&
         "mem_parallel_loop_access kind id drifted");
  (void)MemParallelLoopAccessID;

  // Create the 'nonnull' metadata kind.
  unsigned NonNullID = getMDKindID("nonnull");
  assert(NonNullID == MD_nonnull && "nonnull kind id drifted");
  (void)NonNullID;
  
  // Create the 'dereferenceable' metadata kind.
  unsigned DereferenceableID = getMDKindID("dereferenceable");
  assert(DereferenceableID == MD_dereferenceable && 
         "dereferenceable kind id drifted");
  (void)DereferenceableID;
  
  // Create the 'dereferenceable_or_null' metadata kind.
  unsigned DereferenceableOrNullID = getMDKindID("dereferenceable_or_null");
  assert(DereferenceableOrNullID == MD_dereferenceable_or_null && 
         "dereferenceable_or_null kind id drifted");
  (void)DereferenceableOrNullID;

  implPtrGuard.release(); // HLSL Change: Destructor now on the hook for destruction
}
LLVMContext::~LLVMContext() { delete pImpl; }

void LLVMContext::addModule(Module *M) {
  pImpl->OwnedModules.insert(M);
}

void LLVMContext::removeModule(Module *M) {
  pImpl->OwnedModules.erase(M);
}

//===----------------------------------------------------------------------===//
// Recoverable Backend Errors
//===----------------------------------------------------------------------===//

void LLVMContext::
setInlineAsmDiagnosticHandler(InlineAsmDiagHandlerTy DiagHandler,
                              void *DiagContext) {
  pImpl->InlineAsmDiagHandler = DiagHandler;
  pImpl->InlineAsmDiagContext = DiagContext;
}

/// getInlineAsmDiagnosticHandler - Return the diagnostic handler set by
/// setInlineAsmDiagnosticHandler.
LLVMContext::InlineAsmDiagHandlerTy
LLVMContext::getInlineAsmDiagnosticHandler() const {
  return pImpl->InlineAsmDiagHandler;
}

/// getInlineAsmDiagnosticContext - Return the diagnostic context set by
/// setInlineAsmDiagnosticHandler.
void *LLVMContext::getInlineAsmDiagnosticContext() const {
  return pImpl->InlineAsmDiagContext;
}

void LLVMContext::setDiagnosticHandler(DiagnosticHandlerTy DiagnosticHandler,
                                       void *DiagnosticContext,
                                       bool RespectFilters) {
  pImpl->DiagnosticHandler = DiagnosticHandler;
  pImpl->DiagnosticContext = DiagnosticContext;
  pImpl->RespectDiagnosticFilters = RespectFilters;
}

LLVMContext::DiagnosticHandlerTy LLVMContext::getDiagnosticHandler() const {
  return pImpl->DiagnosticHandler;
}

void *LLVMContext::getDiagnosticContext() const {
  return pImpl->DiagnosticContext;
}

void LLVMContext::setYieldCallback(YieldCallbackTy Callback, void *OpaqueHandle)
{
  pImpl->YieldCallback = Callback;
  pImpl->YieldOpaqueHandle = OpaqueHandle;
}

void LLVMContext::yield() {
  if (pImpl->YieldCallback)
    pImpl->YieldCallback(this, pImpl->YieldOpaqueHandle);
}

void LLVMContext::emitError(const Twine &ErrorStr) {
  diagnose(DiagnosticInfoInlineAsm(ErrorStr));
}

// HLSL Change Start
void LLVMContext::emitWarning(const Twine &WarningStr) {
  diagnose(DiagnosticInfoInlineAsm(WarningStr, DiagnosticSeverity::DS_Warning));
}
// HLSL Change End


void LLVMContext::emitError(const Instruction *I, const Twine &ErrorStr) {
  assert (I && "Invalid instruction");
  diagnose(DiagnosticInfoInlineAsm(*I, ErrorStr));
}

static bool isDiagnosticEnabled(const DiagnosticInfo &DI) {
  // Optimization remarks are selective. They need to check whether the regexp
  // pattern, passed via one of the -pass-remarks* flags, matches the name of
  // the pass that is emitting the diagnostic. If there is no match, ignore the
  // diagnostic and return.
  switch (DI.getKind()) {
  case llvm::DK_OptimizationRemark:
    if (!cast<DiagnosticInfoOptimizationRemark>(DI).isEnabled())
      return false;
    break;
  case llvm::DK_OptimizationRemarkMissed:
    if (!cast<DiagnosticInfoOptimizationRemarkMissed>(DI).isEnabled())
      return false;
    break;
  case llvm::DK_OptimizationRemarkAnalysis:
    if (!cast<DiagnosticInfoOptimizationRemarkAnalysis>(DI).isEnabled())
      return false;
    break;
  default:
    break;
  }
  return true;
}

static const char *getDiagnosticMessagePrefix(DiagnosticSeverity Severity) {
  switch (Severity) {
  case DS_Error:
    return "error";
  case DS_Warning:
    return "warning";
  case DS_Remark:
    return "remark";
  case DS_Note:
    return "note";
  }
  llvm_unreachable("Unknown DiagnosticSeverity");
}

void LLVMContext::diagnose(const DiagnosticInfo &DI) {
  // If there is a report handler, use it.
  if (pImpl->DiagnosticHandler) {
    if (!pImpl->RespectDiagnosticFilters || isDiagnosticEnabled(DI))
      pImpl->DiagnosticHandler(DI, pImpl->DiagnosticContext);
    return;
  }

  if (!isDiagnosticEnabled(DI))
    return;

  // Otherwise, print the message with a prefix based on the severity.
  DiagnosticPrinterRawOStream DP(errs());
  errs() << getDiagnosticMessagePrefix(DI.getSeverity()) << ": ";
  DI.print(DP);
  errs() << "\n";
  if (DI.getSeverity() == DS_Error)
    // exit(1); // HLSL Change - unwind if necessary, but don't terminate the process
    throw std::exception();
}

void LLVMContext::emitError(unsigned LocCookie, const Twine &ErrorStr) {
  diagnose(DiagnosticInfoInlineAsm(LocCookie, ErrorStr));
}

//===----------------------------------------------------------------------===//
// Metadata Kind Uniquing
//===----------------------------------------------------------------------===//

// HLSL Change - Begin
/// Return a unique non-zero ID for the specified metadata kind if it exists.
bool LLVMContext::findMDKindID(StringRef Name, unsigned *ID) const {
  auto it = pImpl->CustomMDKindNames.find(Name);
  if (it != pImpl->CustomMDKindNames.end()) {
    *ID = it->second;
    return true;
  }
  return false;
}
// HLSL Change - End

/// Return a unique non-zero ID for the specified metadata kind.
unsigned LLVMContext::getMDKindID(StringRef Name) const {
  // If this is new, assign it its ID.
  return pImpl->CustomMDKindNames.insert(
                                     std::make_pair(
                                         Name, pImpl->CustomMDKindNames.size()))
      .first->second;
}

/// getHandlerNames - Populate client supplied smallvector using custome
/// metadata name and ID.
void LLVMContext::getMDKindNames(SmallVectorImpl<StringRef> &Names) const {
  Names.resize(pImpl->CustomMDKindNames.size());
  for (StringMap<unsigned>::const_iterator I = pImpl->CustomMDKindNames.begin(),
       E = pImpl->CustomMDKindNames.end(); I != E; ++I)
    Names[I->second] = I->first();
}
