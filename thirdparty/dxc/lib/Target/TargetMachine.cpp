//===-- TargetMachine.cpp - General Target Information ---------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the general parts of a Target machine.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/SectionKind.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

//---------------------------------------------------------------------------
// TargetMachine Class
//

TargetMachine::TargetMachine(const Target &T, StringRef DataLayoutString,
                             const Triple &TT, StringRef CPU, StringRef FS,
                             const TargetOptions &Options)
    : TheTarget(T), DL(DataLayoutString), TargetTriple(TT), TargetCPU(CPU),
      TargetFS(FS), CodeGenInfo(nullptr), AsmInfo(nullptr), MRI(nullptr),
      MII(nullptr), STI(nullptr), RequireStructuredCFG(false),
      Options(Options) {}

TargetMachine::~TargetMachine() {
  delete CodeGenInfo;
  delete AsmInfo;
  delete MRI;
  delete MII;
  delete STI;
}

/// \brief Reset the target options based on the function's attributes.
// FIXME: This function needs to go away for a number of reasons:
// a) global state on the TargetMachine is terrible in general,
// b) there's no default state here to keep,
// c) these target options should be passed only on the function
//    and not on the TargetMachine (via TargetOptions) at all.
void TargetMachine::resetTargetOptions(const Function &F) const {
#define RESET_OPTION(X, Y)                                                     \
  do {                                                                         \
    if (F.hasFnAttribute(Y))                                                   \
      Options.X = (F.getFnAttribute(Y).getValueAsString() == "true");          \
  } while (0)

  RESET_OPTION(LessPreciseFPMADOption, "less-precise-fpmad");
  RESET_OPTION(UnsafeFPMath, "unsafe-fp-math");
  RESET_OPTION(NoInfsFPMath, "no-infs-fp-math");
  RESET_OPTION(NoNaNsFPMath, "no-nans-fp-math");
}

/// getRelocationModel - Returns the code generation relocation model. The
/// choices are static, PIC, and dynamic-no-pic, and target default.
Reloc::Model TargetMachine::getRelocationModel() const {
  if (!CodeGenInfo)
    return Reloc::Default;
  return CodeGenInfo->getRelocationModel();
}

/// getCodeModel - Returns the code model. The choices are small, kernel,
/// medium, large, and target default.
CodeModel::Model TargetMachine::getCodeModel() const {
  if (!CodeGenInfo)
    return CodeModel::Default;
  return CodeGenInfo->getCodeModel();
}

/// Get the IR-specified TLS model for Var.
static TLSModel::Model getSelectedTLSModel(const GlobalValue *GV) {
  switch (GV->getThreadLocalMode()) {
  case GlobalVariable::NotThreadLocal:
    llvm_unreachable("getSelectedTLSModel for non-TLS variable");
    break;
  case GlobalVariable::GeneralDynamicTLSModel:
    return TLSModel::GeneralDynamic;
  case GlobalVariable::LocalDynamicTLSModel:
    return TLSModel::LocalDynamic;
  case GlobalVariable::InitialExecTLSModel:
    return TLSModel::InitialExec;
  case GlobalVariable::LocalExecTLSModel:
    return TLSModel::LocalExec;
  }
  llvm_unreachable("invalid TLS model");
}

TLSModel::Model TargetMachine::getTLSModel(const GlobalValue *GV) const {
  bool isLocal = GV->hasLocalLinkage();
  bool isDeclaration = GV->isDeclaration();
  bool isPIC = getRelocationModel() == Reloc::PIC_;
  bool isPIE = Options.PositionIndependentExecutable;
  // FIXME: what should we do for protected and internal visibility?
  // For variables, is internal different from hidden?
  bool isHidden = GV->hasHiddenVisibility();

  TLSModel::Model Model;
  if (isPIC && !isPIE) {
    if (isLocal || isHidden)
      Model = TLSModel::LocalDynamic;
    else
      Model = TLSModel::GeneralDynamic;
  } else {
    if (!isDeclaration || isHidden)
      Model = TLSModel::LocalExec;
    else
      Model = TLSModel::InitialExec;
  }

  // If the user specified a more specific model, use that.
  TLSModel::Model SelectedModel = getSelectedTLSModel(GV);
  if (SelectedModel > Model)
    return SelectedModel;

  return Model;
}

/// getOptLevel - Returns the optimization level: None, Less,
/// Default, or Aggressive.
CodeGenOpt::Level TargetMachine::getOptLevel() const {
  if (!CodeGenInfo)
    return CodeGenOpt::Default;
  return CodeGenInfo->getOptLevel();
}

void TargetMachine::setOptLevel(CodeGenOpt::Level Level) const {
  if (CodeGenInfo)
    CodeGenInfo->setOptLevel(Level);
}

TargetIRAnalysis TargetMachine::getTargetIRAnalysis() {
  return TargetIRAnalysis([](Function &F) {
    return TargetTransformInfo(F.getParent()->getDataLayout());
  });
}

static bool canUsePrivateLabel(const MCAsmInfo &AsmInfo,
                               const MCSection &Section) {
  if (!AsmInfo.isSectionAtomizableBySymbols(Section))
    return true;

  // If it is not dead stripped, it is safe to use private labels.
  const MCSectionMachO &SMO = cast<MCSectionMachO>(Section);
  if (SMO.hasAttribute(MachO::S_ATTR_NO_DEAD_STRIP))
    return true;

  return false;
}

void TargetMachine::getNameWithPrefix(SmallVectorImpl<char> &Name,
                                      const GlobalValue *GV, Mangler &Mang,
                                      bool MayAlwaysUsePrivate) const {
  if (MayAlwaysUsePrivate || !GV->hasPrivateLinkage()) {
    // Simple case: If GV is not private, it is not important to find out if
    // private labels are legal in this case or not.
    Mang.getNameWithPrefix(Name, GV, false);
    return;
  }
  SectionKind GVKind = TargetLoweringObjectFile::getKindForGlobal(GV, *this);
  const TargetLoweringObjectFile *TLOF = getObjFileLowering();
  const MCSection *TheSection = TLOF->SectionForGlobal(GV, GVKind, Mang, *this);
  bool CannotUsePrivateLabel = !canUsePrivateLabel(*AsmInfo, *TheSection);
  TLOF->getNameWithPrefix(Name, GV, CannotUsePrivateLabel, Mang, *this);
}

MCSymbol *TargetMachine::getSymbol(const GlobalValue *GV, Mangler &Mang) const {
  SmallString<128> NameStr;
  getNameWithPrefix(NameStr, GV, Mang);
  const TargetLoweringObjectFile *TLOF = getObjFileLowering();
  return TLOF->getContext().getOrCreateSymbol(NameStr);
}
