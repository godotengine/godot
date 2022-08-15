//===- llvm/Support/DiagnosticInfo.cpp - Diagnostic Definitions -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the different classes involved in low level diagnostics.
//
// Diagnostics reporting is still done as part of the LLVMContext.
//===----------------------------------------------------------------------===//

#include "LLVMContextImpl.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Regex.h"
#include <atomic>
#include <string>

using namespace llvm;

namespace {

/// \brief Regular expression corresponding to the value given in one of the
/// -pass-remarks* command line flags. Passes whose name matches this regexp
/// will emit a diagnostic when calling the associated diagnostic function
/// (emitOptimizationRemark, emitOptimizationRemarkMissed or
/// emitOptimizationRemarkAnalysis).
struct PassRemarksOpt {
  std::shared_ptr<Regex> Pattern;

  void operator=(const std::string &Val) {
    // Create a regexp object to match pass names for emitOptimizationRemark.
    if (!Val.empty()) {
      Pattern = std::make_shared<Regex>(Val);
      std::string RegexError;
      if (!Pattern->isValid(RegexError))
        report_fatal_error("Invalid regular expression '" + Val +
                               "' in -pass-remarks: " + RegexError,
                           false);
    }
  };
};

#if 0
// These should all be specific to a pipline, not global to the process.
static PassRemarksOpt PassRemarksOptLoc;
static PassRemarksOpt PassRemarksMissedOptLoc;
static PassRemarksOpt PassRemarksAnalysisOptLoc;

// -pass-remarks
//    Command line flag to enable emitOptimizationRemark()
static cl::opt<PassRemarksOpt, true, cl::parser<std::string>>
PassRemarks("pass-remarks", cl::value_desc("pattern"),
            cl::desc("Enable optimization remarks from passes whose name match "
                     "the given regular expression"),
            cl::Hidden, cl::location(PassRemarksOptLoc), cl::ValueRequired,
            cl::ZeroOrMore);

// -pass-remarks-missed
//    Command line flag to enable emitOptimizationRemarkMissed()
static cl::opt<PassRemarksOpt, true, cl::parser<std::string>> PassRemarksMissed(
    "pass-remarks-missed", cl::value_desc("pattern"),
    cl::desc("Enable missed optimization remarks from passes whose name match "
             "the given regular expression"),
    cl::Hidden, cl::location(PassRemarksMissedOptLoc), cl::ValueRequired,
    cl::ZeroOrMore);

// -pass-remarks-analysis
//    Command line flag to enable emitOptimizationRemarkAnalysis()
static cl::opt<PassRemarksOpt, true, cl::parser<std::string>>
PassRemarksAnalysis(
    "pass-remarks-analysis", cl::value_desc("pattern"),
    cl::desc(
        "Enable optimization analysis remarks from passes whose name match "
        "the given regular expression"),
    cl::Hidden, cl::location(PassRemarksAnalysisOptLoc), cl::ValueRequired,
    cl::ZeroOrMore);
}
#else
struct PassRemarksOptNull {
  Regex *Pattern = nullptr;
  void operator=(const std::string &Val) {
  }
};
static PassRemarksOptNull PassRemarksOptLoc;
static PassRemarksOptNull PassRemarksMissedOptLoc;
static PassRemarksOptNull PassRemarksAnalysisOptLoc;

// static PassRemarksOptNull PassRemarks; // HLSL Change
// static PassRemarksOptNull PassRemarksMissed; // HLSL Change
// static PassRemarksOptNull PassRemarksAnalysis; // HLSL Change
}
#endif

int llvm::getNextAvailablePluginDiagnosticKind() {
  static std::atomic<int> PluginKindID(DK_FirstPluginKind);
  return ++PluginKindID;
}

DiagnosticInfoInlineAsm::DiagnosticInfoInlineAsm(const Instruction &I,
                                                 const Twine &MsgStr,
                                                 DiagnosticSeverity Severity)
    : DiagnosticInfo(DK_InlineAsm, Severity), LocCookie(0), MsgStr(MsgStr),
      Instr(&I) {
  if (const MDNode *SrcLoc = I.getMetadata("srcloc")) {
    if (SrcLoc->getNumOperands() != 0)
      if (const auto *CI =
              mdconst::dyn_extract<ConstantInt>(SrcLoc->getOperand(0)))
        LocCookie = CI->getZExtValue();
  }
}

void DiagnosticInfoInlineAsm::print(DiagnosticPrinter &DP) const {
  DP << getMsgStr();
  if (getLocCookie())
    DP << " at line " << getLocCookie();
}

void DiagnosticInfoStackSize::print(DiagnosticPrinter &DP) const {
  DP << "stack size limit exceeded (" << getStackSize() << ") in "
     << getFunction();
}

void DiagnosticInfoDebugMetadataVersion::print(DiagnosticPrinter &DP) const {
  DP << "ignoring debug info with an invalid version (" << getMetadataVersion()
     << ") in " << getModule();
}

void DiagnosticInfoSampleProfile::print(DiagnosticPrinter &DP) const {
  if (getFileName() && getLineNum() > 0)
    DP << getFileName() << ":" << getLineNum() << ": ";
  else if (getFileName())
    DP << getFileName() << ": ";
  DP << getMsg();
}

bool DiagnosticInfoOptimizationBase::isLocationAvailable() const {
  return getDebugLoc();
}

void DiagnosticInfoOptimizationBase::getLocation(StringRef *Filename,
                                                 unsigned *Line,
                                                 unsigned *Column) const {
  DILocation *L = getDebugLoc();
  assert(L != nullptr && "debug location is invalid");
  *Filename = L->getFilename();
  *Line = L->getLine();
  *Column = L->getColumn();
}

const std::string DiagnosticInfoOptimizationBase::getLocationStr() const {
  StringRef Filename("<unknown>");
  unsigned Line = 0;
  unsigned Column = 0;
  if (isLocationAvailable())
    getLocation(&Filename, &Line, &Column);
  return (Filename + ":" + Twine(Line) + ":" + Twine(Column)).str();
}

void DiagnosticInfoOptimizationBase::print(DiagnosticPrinter &DP) const {
  DP << getLocationStr() << ": " << getMsg();
}

bool DiagnosticInfoOptimizationRemark::isEnabled() const {
  return PassRemarksOptLoc.Pattern &&
         PassRemarksOptLoc.Pattern->match(getPassName());
}

bool DiagnosticInfoOptimizationRemarkMissed::isEnabled() const {
  return PassRemarksMissedOptLoc.Pattern &&
         PassRemarksMissedOptLoc.Pattern->match(getPassName());
}

bool DiagnosticInfoOptimizationRemarkAnalysis::isEnabled() const {
  return PassRemarksAnalysisOptLoc.Pattern &&
         PassRemarksAnalysisOptLoc.Pattern->match(getPassName());
}

void DiagnosticInfoMIRParser::print(DiagnosticPrinter &DP) const {
  DP << Diagnostic;
}

void llvm::emitOptimizationRemark(LLVMContext &Ctx, const char *PassName,
                                  const Function &Fn, const DebugLoc &DLoc,
                                  const Twine &Msg) {
  Ctx.diagnose(DiagnosticInfoOptimizationRemark(PassName, Fn, DLoc, Msg));
}

void llvm::emitOptimizationRemarkMissed(LLVMContext &Ctx, const char *PassName,
                                        const Function &Fn,
                                        const DebugLoc &DLoc,
                                        const Twine &Msg) {
  Ctx.diagnose(DiagnosticInfoOptimizationRemarkMissed(PassName, Fn, DLoc, Msg));
}

void llvm::emitOptimizationRemarkAnalysis(LLVMContext &Ctx,
                                          const char *PassName,
                                          const Function &Fn,
                                          const DebugLoc &DLoc,
                                          const Twine &Msg) {
  Ctx.diagnose(
      DiagnosticInfoOptimizationRemarkAnalysis(PassName, Fn, DLoc, Msg));
}

bool DiagnosticInfoOptimizationFailure::isEnabled() const {
  // Only print warnings.
  return getSeverity() == DS_Warning;
}

void llvm::emitLoopVectorizeWarning(LLVMContext &Ctx, const Function &Fn,
                                    const DebugLoc &DLoc, const Twine &Msg) {
  Ctx.diagnose(DiagnosticInfoOptimizationFailure(
      Fn, DLoc, Twine("loop not vectorized: " + Msg)));
}

void llvm::emitLoopInterleaveWarning(LLVMContext &Ctx, const Function &Fn,
                                     const DebugLoc &DLoc, const Twine &Msg) {
  Ctx.diagnose(DiagnosticInfoOptimizationFailure(
      Fn, DLoc, Twine("loop not interleaved: " + Msg)));
}

// HLSL Change start - Dxil Diagnostic Info reporter
DiagnosticInfoDxil::DiagnosticInfoDxil(const Function *F, const DILocation *Loc, const Twine &MsgStr,
                                       DiagnosticSeverity Severity)
  : DiagnosticInfoDxil(F, MsgStr, Severity)
{
  if (Loc) {
    HasLocation = true;
    FileName = Loc->getFilename();
    Line     = Loc->getLine();
    Column   = Loc->getColumn();
  }
}

DiagnosticInfoDxil::DiagnosticInfoDxil(const Function *F, const DIGlobalVariable *DGV, const Twine &MsgStr,
                   DiagnosticSeverity Severity)
  : DiagnosticInfoDxil(F, MsgStr, Severity)
{
  if (DGV) {
    HasLocation = true;
    FileName = DGV->getFilename();
    Line     = DGV->getLine();
    Column   = 0;
  }
}

// Slapdash printing of diagnostic information as a last resort
// Used by validation and linker errors. Doesn't include source snippets.
void DiagnosticInfoDxil::print(DiagnosticPrinter &DP) const {
  if (HasLocation) {
    DP << FileName << ":" << Line << ":";
    if (Column > 0)
      DP << Column << ":";
    DP << " ";
  } else if (Func) {
    DP << "Function: " << Func->getName() << ": ";
  }

  switch (getSeverity()) {
  case DiagnosticSeverity::DS_Note:    DP << "note: "; break;
  case DiagnosticSeverity::DS_Remark:  DP << "remark: "; break;
  case DiagnosticSeverity::DS_Warning: DP << "warning: "; break;
  case DiagnosticSeverity::DS_Error:   DP << "error: "; break;
  }
  DP << getMsgStr();
}
// HLSL Change end - Dxil Diagnostic Info reporter
