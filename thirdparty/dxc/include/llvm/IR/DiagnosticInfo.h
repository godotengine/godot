//===- llvm/Support/DiagnosticInfo.h - Diagnostic Declaration ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the different classes involved in low level diagnostics.
//
// Diagnostics reporting is still done as part of the LLVMContext.
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DIAGNOSTICINFO_H
#define LLVM_IR_DIAGNOSTICINFO_H

#include "llvm-c/Core.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include <functional>

namespace llvm {

// Forward declarations.
class DiagnosticPrinter;
class Function;
class Instruction;
class LLVMContextImpl;
class Twine;
class Value;
class DebugLoc;
class SMDiagnostic;

/// \brief Defines the different supported severity of a diagnostic.
enum DiagnosticSeverity {
  DS_Error,
  DS_Warning,
  DS_Remark,
  // A note attaches additional information to one of the previous diagnostic
  // types.
  DS_Note
};

/// \brief Defines the different supported kind of a diagnostic.
/// This enum should be extended with a new ID for each added concrete subclass.
enum DiagnosticKind {
  DK_Bitcode,
  DK_InlineAsm,
  DK_StackSize,
  DK_Linker,
  DK_DebugMetadataVersion,
  DK_SampleProfile,
  DK_OptimizationRemark,
  DK_OptimizationRemarkMissed,
  DK_OptimizationRemarkAnalysis,
  DK_OptimizationFailure,
  DK_MIRParser,
  DK_DXIL, // HLSL Change
  DK_FirstPluginKind
};

/// \brief Get the next available kind ID for a plugin diagnostic.
/// Each time this function is called, it returns a different number.
/// Therefore, a plugin that wants to "identify" its own classes
/// with a dynamic identifier, just have to use this method to get a new ID
/// and assign it to each of its classes.
/// The returned ID will be greater than or equal to DK_FirstPluginKind.
/// Thus, the plugin identifiers will not conflict with the
/// DiagnosticKind values.
int getNextAvailablePluginDiagnosticKind();

/// \brief This is the base abstract class for diagnostic reporting in
/// the backend.
/// The print method must be overloaded by the subclasses to print a
/// user-friendly message in the client of the backend (let us call it a
/// frontend).
class DiagnosticInfo {
private:
  /// Kind defines the kind of report this is about.
  const /* DiagnosticKind */ int Kind;
  /// Severity gives the severity of the diagnostic.
  const DiagnosticSeverity Severity;

public:
  DiagnosticInfo(/* DiagnosticKind */ int Kind, DiagnosticSeverity Severity)
      : Kind(Kind), Severity(Severity) {}

  virtual ~DiagnosticInfo() {}

  /* DiagnosticKind */ int getKind() const { return Kind; }
  DiagnosticSeverity getSeverity() const { return Severity; }

  /// Print using the given \p DP a user-friendly message.
  /// This is the default message that will be printed to the user.
  /// It is used when the frontend does not directly take advantage
  /// of the information contained in fields of the subclasses.
  /// The printed message must not end with '.' nor start with a severity
  /// keyword.
  virtual void print(DiagnosticPrinter &DP) const = 0;
};

typedef std::function<void(const DiagnosticInfo &)> DiagnosticHandlerFunction;

/// Diagnostic information for inline asm reporting.
/// This is basically a message and an optional location.
class DiagnosticInfoInlineAsm : public DiagnosticInfo {
private:
  /// Optional line information. 0 if not set.
  unsigned LocCookie;
  /// Message to be reported.
  const Twine &MsgStr;
  /// Optional origin of the problem.
  const Instruction *Instr;

public:
  /// \p MsgStr is the message to be reported to the frontend.
  /// This class does not copy \p MsgStr, therefore the reference must be valid
  /// for the whole life time of the Diagnostic.
  DiagnosticInfoInlineAsm(const Twine &MsgStr,
                          DiagnosticSeverity Severity = DS_Error)
      : DiagnosticInfo(DK_InlineAsm, Severity), LocCookie(0), MsgStr(MsgStr),
        Instr(nullptr) {}

  /// \p LocCookie if non-zero gives the line number for this report.
  /// \p MsgStr gives the message.
  /// This class does not copy \p MsgStr, therefore the reference must be valid
  /// for the whole life time of the Diagnostic.
  DiagnosticInfoInlineAsm(unsigned LocCookie, const Twine &MsgStr,
                          DiagnosticSeverity Severity = DS_Error)
      : DiagnosticInfo(DK_InlineAsm, Severity), LocCookie(LocCookie),
        MsgStr(MsgStr), Instr(nullptr) {}

  /// \p Instr gives the original instruction that triggered the diagnostic.
  /// \p MsgStr gives the message.
  /// This class does not copy \p MsgStr, therefore the reference must be valid
  /// for the whole life time of the Diagnostic.
  /// Same for \p I.
  DiagnosticInfoInlineAsm(const Instruction &I, const Twine &MsgStr,
                          DiagnosticSeverity Severity = DS_Error);

  unsigned getLocCookie() const { return LocCookie; }
  const Twine &getMsgStr() const { return MsgStr; }
  const Instruction *getInstruction() const { return Instr; }

  /// \see DiagnosticInfo::print.
  void print(DiagnosticPrinter &DP) const override;

  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == DK_InlineAsm;
  }
};

/// Diagnostic information for stack size reporting.
/// This is basically a function and a size.
class DiagnosticInfoStackSize : public DiagnosticInfo {
private:
  /// The function that is concerned by this stack size diagnostic.
  const Function &Fn;
  /// The computed stack size.
  unsigned StackSize;

public:
  /// \p The function that is concerned by this stack size diagnostic.
  /// \p The computed stack size.
  DiagnosticInfoStackSize(const Function &Fn, unsigned StackSize,
                          DiagnosticSeverity Severity = DS_Warning)
      : DiagnosticInfo(DK_StackSize, Severity), Fn(Fn), StackSize(StackSize) {}

  const Function &getFunction() const { return Fn; }
  unsigned getStackSize() const { return StackSize; }

  /// \see DiagnosticInfo::print.
  void print(DiagnosticPrinter &DP) const override;

  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == DK_StackSize;
  }
};

/// Diagnostic information for debug metadata version reporting.
/// This is basically a module and a version.
class DiagnosticInfoDebugMetadataVersion : public DiagnosticInfo {
private:
  /// The module that is concerned by this debug metadata version diagnostic.
  const Module &M;
  /// The actual metadata version.
  unsigned MetadataVersion;

public:
  /// \p The module that is concerned by this debug metadata version diagnostic.
  /// \p The actual metadata version.
  DiagnosticInfoDebugMetadataVersion(const Module &M, unsigned MetadataVersion,
                          DiagnosticSeverity Severity = DS_Warning)
      : DiagnosticInfo(DK_DebugMetadataVersion, Severity), M(M),
        MetadataVersion(MetadataVersion) {}

  const Module &getModule() const { return M; }
  unsigned getMetadataVersion() const { return MetadataVersion; }

  /// \see DiagnosticInfo::print.
  void print(DiagnosticPrinter &DP) const override;

  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == DK_DebugMetadataVersion;
  }
};

/// Diagnostic information for the sample profiler.
class DiagnosticInfoSampleProfile : public DiagnosticInfo {
public:
  DiagnosticInfoSampleProfile(const char *FileName, unsigned LineNum,
                              const Twine &Msg,
                              DiagnosticSeverity Severity = DS_Error)
      : DiagnosticInfo(DK_SampleProfile, Severity), FileName(FileName),
        LineNum(LineNum), Msg(Msg) {}
  DiagnosticInfoSampleProfile(const char *FileName, const Twine &Msg,
                              DiagnosticSeverity Severity = DS_Error)
      : DiagnosticInfo(DK_SampleProfile, Severity), FileName(FileName),
        LineNum(0), Msg(Msg) {}
  DiagnosticInfoSampleProfile(const Twine &Msg,
                              DiagnosticSeverity Severity = DS_Error)
      : DiagnosticInfo(DK_SampleProfile, Severity), FileName(nullptr),
        LineNum(0), Msg(Msg) {}

  /// \see DiagnosticInfo::print.
  void print(DiagnosticPrinter &DP) const override;

  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == DK_SampleProfile;
  }

  const char *getFileName() const { return FileName; }
  unsigned getLineNum() const { return LineNum; }
  const Twine &getMsg() const { return Msg; }

private:
  /// Name of the input file associated with this diagnostic.
  const char *FileName;

  /// Line number where the diagnostic occurred. If 0, no line number will
  /// be emitted in the message.
  unsigned LineNum;

  /// Message to report.
  const Twine &Msg;
};

/// Common features for diagnostics dealing with optimization remarks.
class DiagnosticInfoOptimizationBase : public DiagnosticInfo {
public:
  /// \p PassName is the name of the pass emitting this diagnostic.
  /// \p Fn is the function where the diagnostic is being emitted. \p DLoc is
  /// the location information to use in the diagnostic. If line table
  /// information is available, the diagnostic will include the source code
  /// location. \p Msg is the message to show. Note that this class does not
  /// copy this message, so this reference must be valid for the whole life time
  /// of the diagnostic.
  DiagnosticInfoOptimizationBase(enum DiagnosticKind Kind,
                                 enum DiagnosticSeverity Severity,
                                 const char *PassName, const Function &Fn,
                                 const DebugLoc &DLoc, const Twine &Msg)
      : DiagnosticInfo(Kind, Severity), PassName(PassName), Fn(Fn), DLoc(DLoc),
        Msg(Msg) {}

  /// \see DiagnosticInfo::print.
  void print(DiagnosticPrinter &DP) const override;

  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == DK_OptimizationRemark;
  }

  /// Return true if this optimization remark is enabled by one of
  /// of the LLVM command line flags (-pass-remarks, -pass-remarks-missed,
  /// or -pass-remarks-analysis). Note that this only handles the LLVM
  /// flags. We cannot access Clang flags from here (they are handled
  /// in BackendConsumer::OptimizationRemarkHandler).
  virtual bool isEnabled() const = 0;

  /// Return true if location information is available for this diagnostic.
  bool isLocationAvailable() const;

  /// Return a string with the location information for this diagnostic
  /// in the format "file:line:col". If location information is not available,
  /// it returns "<unknown>:0:0".
  const std::string getLocationStr() const;

  /// Return location information for this diagnostic in three parts:
  /// the source file name, line number and column.
  void getLocation(StringRef *Filename, unsigned *Line, unsigned *Column) const;

  StringRef getPassName() const { return PassName; }
  const Function &getFunction() const { return Fn; }
  const DebugLoc &getDebugLoc() const { return DLoc; }
  const Twine &getMsg() const { return Msg; }

private:
  /// Name of the pass that triggers this report. If this matches the
  /// regular expression given in -Rpass=regexp, then the remark will
  /// be emitted.
  const char *PassName;

  /// Function where this diagnostic is triggered.
  const Function &Fn;

  /// Debug location where this diagnostic is triggered.
  DebugLoc DLoc;

  /// Message to report.
  const Twine &Msg;
};

/// Diagnostic information for applied optimization remarks.
class DiagnosticInfoOptimizationRemark : public DiagnosticInfoOptimizationBase {
public:
  /// \p PassName is the name of the pass emitting this diagnostic. If
  /// this name matches the regular expression given in -Rpass=, then the
  /// diagnostic will be emitted. \p Fn is the function where the diagnostic
  /// is being emitted. \p DLoc is the location information to use in the
  /// diagnostic. If line table information is available, the diagnostic
  /// will include the source code location. \p Msg is the message to show.
  /// Note that this class does not copy this message, so this reference
  /// must be valid for the whole life time of the diagnostic.
  DiagnosticInfoOptimizationRemark(const char *PassName, const Function &Fn,
                                   const DebugLoc &DLoc, const Twine &Msg)
      : DiagnosticInfoOptimizationBase(DK_OptimizationRemark, DS_Remark,
                                       PassName, Fn, DLoc, Msg) {}

  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == DK_OptimizationRemark;
  }

  /// \see DiagnosticInfoOptimizationBase::isEnabled.
  bool isEnabled() const override;
};

/// Diagnostic information for missed-optimization remarks.
class DiagnosticInfoOptimizationRemarkMissed
    : public DiagnosticInfoOptimizationBase {
public:
  /// \p PassName is the name of the pass emitting this diagnostic. If
  /// this name matches the regular expression given in -Rpass-missed=, then the
  /// diagnostic will be emitted. \p Fn is the function where the diagnostic
  /// is being emitted. \p DLoc is the location information to use in the
  /// diagnostic. If line table information is available, the diagnostic
  /// will include the source code location. \p Msg is the message to show.
  /// Note that this class does not copy this message, so this reference
  /// must be valid for the whole life time of the diagnostic.
  DiagnosticInfoOptimizationRemarkMissed(const char *PassName,
                                         const Function &Fn,
                                         const DebugLoc &DLoc, const Twine &Msg)
      : DiagnosticInfoOptimizationBase(DK_OptimizationRemarkMissed, DS_Remark,
                                       PassName, Fn, DLoc, Msg) {}

  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == DK_OptimizationRemarkMissed;
  }

  /// \see DiagnosticInfoOptimizationBase::isEnabled.
  bool isEnabled() const override;
};

/// Diagnostic information for optimization analysis remarks.
class DiagnosticInfoOptimizationRemarkAnalysis
    : public DiagnosticInfoOptimizationBase {
public:
  /// \p PassName is the name of the pass emitting this diagnostic. If
  /// this name matches the regular expression given in -Rpass-analysis=, then
  /// the diagnostic will be emitted. \p Fn is the function where the diagnostic
  /// is being emitted. \p DLoc is the location information to use in the
  /// diagnostic. If line table information is available, the diagnostic will
  /// include the source code location. \p Msg is the message to show. Note that
  /// this class does not copy this message, so this reference must be valid for
  /// the whole life time of the diagnostic.
  DiagnosticInfoOptimizationRemarkAnalysis(const char *PassName,
                                           const Function &Fn,
                                           const DebugLoc &DLoc,
                                           const Twine &Msg)
      : DiagnosticInfoOptimizationBase(DK_OptimizationRemarkAnalysis, DS_Remark,
                                       PassName, Fn, DLoc, Msg) {}

  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == DK_OptimizationRemarkAnalysis;
  }

  /// \see DiagnosticInfoOptimizationBase::isEnabled.
  bool isEnabled() const override;
};

/// Diagnostic information for machine IR parser.
class DiagnosticInfoMIRParser : public DiagnosticInfo {
  const SMDiagnostic &Diagnostic;

public:
  DiagnosticInfoMIRParser(DiagnosticSeverity Severity,
                          const SMDiagnostic &Diagnostic)
      : DiagnosticInfo(DK_MIRParser, Severity), Diagnostic(Diagnostic) {}

  const SMDiagnostic &getDiagnostic() const { return Diagnostic; }

  void print(DiagnosticPrinter &DP) const override;

  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == DK_MIRParser;
  }
};

// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(DiagnosticInfo, LLVMDiagnosticInfoRef)

/// Emit an optimization-applied message. \p PassName is the name of the pass
/// emitting the message. If -Rpass= is given and \p PassName matches the
/// regular expression in -Rpass, then the remark will be emitted. \p Fn is
/// the function triggering the remark, \p DLoc is the debug location where
/// the diagnostic is generated. \p Msg is the message string to use.
void emitOptimizationRemark(LLVMContext &Ctx, const char *PassName,
                            const Function &Fn, const DebugLoc &DLoc,
                            const Twine &Msg);

/// Emit an optimization-missed message. \p PassName is the name of the
/// pass emitting the message. If -Rpass-missed= is given and \p PassName
/// matches the regular expression in -Rpass, then the remark will be
/// emitted. \p Fn is the function triggering the remark, \p DLoc is the
/// debug location where the diagnostic is generated. \p Msg is the
/// message string to use.
void emitOptimizationRemarkMissed(LLVMContext &Ctx, const char *PassName,
                                  const Function &Fn, const DebugLoc &DLoc,
                                  const Twine &Msg);

/// Emit an optimization analysis remark message. \p PassName is the name of
/// the pass emitting the message. If -Rpass-analysis= is given and \p
/// PassName matches the regular expression in -Rpass, then the remark will be
/// emitted. \p Fn is the function triggering the remark, \p DLoc is the debug
/// location where the diagnostic is generated. \p Msg is the message string
/// to use.
void emitOptimizationRemarkAnalysis(LLVMContext &Ctx, const char *PassName,
                                    const Function &Fn, const DebugLoc &DLoc,
                                    const Twine &Msg);

/// Diagnostic information for optimization failures.
class DiagnosticInfoOptimizationFailure
    : public DiagnosticInfoOptimizationBase {
public:
  /// \p Fn is the function where the diagnostic is being emitted. \p DLoc is
  /// the location information to use in the diagnostic. If line table
  /// information is available, the diagnostic will include the source code
  /// location. \p Msg is the message to show. Note that this class does not
  /// copy this message, so this reference must be valid for the whole life time
  /// of the diagnostic.
  DiagnosticInfoOptimizationFailure(const Function &Fn, const DebugLoc &DLoc,
                                    const Twine &Msg)
      : DiagnosticInfoOptimizationBase(DK_OptimizationFailure, DS_Warning,
                                       nullptr, Fn, DLoc, Msg) {}

  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == DK_OptimizationFailure;
  }

  /// \see DiagnosticInfoOptimizationBase::isEnabled.
  bool isEnabled() const override;
};

/// Emit a warning when loop vectorization is specified but fails. \p Fn is the
/// function triggering the warning, \p DLoc is the debug location where the
/// diagnostic is generated. \p Msg is the message string to use.
void emitLoopVectorizeWarning(LLVMContext &Ctx, const Function &Fn,
                              const DebugLoc &DLoc, const Twine &Msg);

/// Emit a warning when loop interleaving is specified but fails. \p Fn is the
/// function triggering the warning, \p DLoc is the debug location where the
/// diagnostic is generated. \p Msg is the message string to use.
void emitLoopInterleaveWarning(LLVMContext &Ctx, const Function &Fn,
                               const DebugLoc &DLoc, const Twine &Msg);

// HLSL Change start - Dxil Diagnostic Info reporter

/// Diagnostic information for Dxil errors
/// Intended for use in post-codegen passes
/// where location information is stored in metadata
class DiagnosticInfoDxil : public DiagnosticInfo {
private:
  // Function
  const Function *Func;

  bool HasLocation = false;
  unsigned Line = 0;
  unsigned Column = 0;
  StringRef FileName;

  /// Message to be reported.
  const Twine &MsgStr;


public:
  /// This class does not copy \p MsgStr, therefore the reference must be valid
  /// for the whole life time of the Diagnostic.
  /// 
  DiagnosticInfoDxil(const Function *F, const Twine &MsgStr, DiagnosticSeverity Severity) :
    DiagnosticInfo(DK_DXIL, Severity), Func(F), MsgStr(MsgStr)
  {}
  DiagnosticInfoDxil(const Function *F, const DILocation *Loc, const Twine &MsgStr,
                      DiagnosticSeverity Severity = DS_Error);
  DiagnosticInfoDxil(const Function *F, const DIGlobalVariable *DGV, const Twine &MsgStr,
                      DiagnosticSeverity Severity = DS_Error);

  const Function *getFunction() const { return Func; }
  const Twine &getMsgStr() const { return MsgStr; }
  bool hasLocation() const { return HasLocation; }
  unsigned getLine() const { return Line; }
  unsigned getColumn() const { return Column; }
  StringRef getFileName() const { return FileName; }

  /// \see DiagnosticInfo::print.
  void print(DiagnosticPrinter &DP) const override;

  static bool classof(const DiagnosticInfo *DI) {
    return DI->getKind() == DK_DXIL;
  }
};
// HLSL Change end - Dxil Diagnostic Info reporter

} // End namespace llvm

#endif
