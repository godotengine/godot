//===-- LLVMTargetMachine.cpp - Implement the LLVMTargetMachine class -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVMTargetMachine class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetMachine.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

// Enable or disable FastISel. Both options are needed, because
// FastISel is enabled by default with -fast, and we wish to be
// able to enable or disable fast-isel independently from -O0.
static cl::opt<cl::boolOrDefault>
EnableFastISelOption("fast-isel", cl::Hidden,
  cl::desc("Enable the \"fast\" instruction selector"));

void LLVMTargetMachine::initAsmInfo() {
  MRI = TheTarget.createMCRegInfo(getTargetTriple().str());
  MII = TheTarget.createMCInstrInfo();
  // FIXME: Having an MCSubtargetInfo on the target machine is a hack due
  // to some backends having subtarget feature dependent module level
  // code generation. This is similar to the hack in the AsmPrinter for
  // module level assembly etc.
  STI = TheTarget.createMCSubtargetInfo(getTargetTriple().str(), getTargetCPU(),
                                        getTargetFeatureString());

  MCAsmInfo *TmpAsmInfo =
      TheTarget.createMCAsmInfo(*MRI, getTargetTriple().str());
  // TargetSelect.h moved to a different directory between LLVM 2.9 and 3.0,
  // and if the old one gets included then MCAsmInfo will be NULL and
  // we'll crash later.
  // Provide the user with a useful error message about what's wrong.
  assert(TmpAsmInfo && "MCAsmInfo not initialized. "
         "Make sure you include the correct TargetSelect.h"
         "and that InitializeAllTargetMCs() is being invoked!");

  if (Options.DisableIntegratedAS)
    TmpAsmInfo->setUseIntegratedAssembler(false);

  if (Options.CompressDebugSections)
    TmpAsmInfo->setCompressDebugSections(true);

  AsmInfo = TmpAsmInfo;
}

LLVMTargetMachine::LLVMTargetMachine(const Target &T,
                                     StringRef DataLayoutString,
                                     const Triple &TT, StringRef CPU,
                                     StringRef FS, TargetOptions Options,
                                     Reloc::Model RM, CodeModel::Model CM,
                                     CodeGenOpt::Level OL)
    : TargetMachine(T, DataLayoutString, TT, CPU, FS, Options) {
  CodeGenInfo = T.createMCCodeGenInfo(TT.str(), RM, CM, OL);
}

TargetIRAnalysis LLVMTargetMachine::getTargetIRAnalysis() {
  return TargetIRAnalysis([this](Function &F) {
    return TargetTransformInfo(BasicTTIImpl(this, F));
  });
}

/// addPassesToX helper drives creation and initialization of TargetPassConfig.
static MCContext *
addPassesToGenerateCode(LLVMTargetMachine *TM, PassManagerBase &PM,
                        bool DisableVerify, AnalysisID StartBefore,
                        AnalysisID StartAfter, AnalysisID StopAfter,
                        MachineFunctionInitializer *MFInitializer = nullptr) {

  // Add internal analysis passes from the target machine.
  PM.add(createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));

  // Targets may override createPassConfig to provide a target-specific
  // subclass.
  TargetPassConfig *PassConfig = TM->createPassConfig(PM);
  PassConfig->setStartStopPasses(StartBefore, StartAfter, StopAfter);

  // Set PassConfig options provided by TargetMachine.
  PassConfig->setDisableVerify(DisableVerify);

  PM.add(PassConfig);

  PassConfig->addIRPasses();

  PassConfig->addCodeGenPrepare();

  PassConfig->addPassesToHandleExceptions();

  PassConfig->addISelPrepare();

  // Install a MachineModuleInfo class, which is an immutable pass that holds
  // all the per-module stuff we're generating, including MCContext.
  MachineModuleInfo *MMI = new MachineModuleInfo(
      *TM->getMCAsmInfo(), *TM->getMCRegisterInfo(), TM->getObjFileLowering());
  PM.add(MMI);

  // Set up a MachineFunction for the rest of CodeGen to work on.
  PM.add(new MachineFunctionAnalysis(*TM, MFInitializer));

  // Enable FastISel with -fast, but allow that to be overridden.
  if (EnableFastISelOption == cl::BOU_TRUE ||
      (TM->getOptLevel() == CodeGenOpt::None &&
       EnableFastISelOption != cl::BOU_FALSE))
    TM->setFastISel(true);

  // Ask the target for an isel.
  if (PassConfig->addInstSelector())
    return nullptr;

  PassConfig->addMachinePasses();

  PassConfig->setInitialized();

  return &MMI->getContext();
}

bool LLVMTargetMachine::addPassesToEmitFile(
    PassManagerBase &PM, raw_pwrite_stream &Out, CodeGenFileType FileType,
    bool DisableVerify, AnalysisID StartBefore, AnalysisID StartAfter,
    AnalysisID StopAfter, MachineFunctionInitializer *MFInitializer) {
  // Add common CodeGen passes.
  MCContext *Context =
      addPassesToGenerateCode(this, PM, DisableVerify, StartBefore, StartAfter,
                              StopAfter, MFInitializer);
  if (!Context)
    return true;

  if (StopAfter) {
    PM.add(createPrintMIRPass(outs()));
    return false;
  }

  if (Options.MCOptions.MCSaveTempLabels)
    Context->setAllowTemporaryLabels(false);

  const MCSubtargetInfo &STI = *getMCSubtargetInfo();
  const MCAsmInfo &MAI = *getMCAsmInfo();
  const MCRegisterInfo &MRI = *getMCRegisterInfo();
  const MCInstrInfo &MII = *getMCInstrInfo();

  std::unique_ptr<MCStreamer> AsmStreamer;

  switch (FileType) {
  case CGFT_AssemblyFile: {
    MCInstPrinter *InstPrinter = getTarget().createMCInstPrinter(
        getTargetTriple(), MAI.getAssemblerDialect(), MAI, MII, MRI);

    // Create a code emitter if asked to show the encoding.
    MCCodeEmitter *MCE = nullptr;
    if (Options.MCOptions.ShowMCEncoding)
      MCE = getTarget().createMCCodeEmitter(MII, MRI, *Context);

    MCAsmBackend *MAB =
        getTarget().createMCAsmBackend(MRI, getTargetTriple().str(), TargetCPU);
    auto FOut = llvm::make_unique<formatted_raw_ostream>(Out);
    MCStreamer *S = getTarget().createAsmStreamer(
        *Context, std::move(FOut), Options.MCOptions.AsmVerbose,
        Options.MCOptions.MCUseDwarfDirectory, InstPrinter, MCE, MAB,
        Options.MCOptions.ShowMCInst);
    AsmStreamer.reset(S);
    break;
  }
  case CGFT_ObjectFile: {
    // Create the code emitter for the target if it exists.  If not, .o file
    // emission fails.
    MCCodeEmitter *MCE = getTarget().createMCCodeEmitter(MII, MRI, *Context);
    MCAsmBackend *MAB =
        getTarget().createMCAsmBackend(MRI, getTargetTriple().str(), TargetCPU);
    if (!MCE || !MAB)
      return true;

    // Don't waste memory on names of temp labels.
    Context->setUseNamesOnTempLabels(false);

    Triple T(getTargetTriple().str());
    AsmStreamer.reset(getTarget().createMCObjectStreamer(
        T, *Context, *MAB, Out, MCE, STI, Options.MCOptions.MCRelaxAll,
        /*DWARFMustBeAtTheEnd*/ true));
    break;
  }
  case CGFT_Null:
    // The Null output is intended for use for performance analysis and testing,
    // not real users.
    AsmStreamer.reset(getTarget().createNullStreamer(*Context));
    break;
  }

  // Create the AsmPrinter, which takes ownership of AsmStreamer if successful.
  FunctionPass *Printer =
      getTarget().createAsmPrinter(*this, std::move(AsmStreamer));
  if (!Printer)
    return true;

  PM.add(Printer);

  return false;
}

/// addPassesToEmitMC - Add passes to the specified pass manager to get
/// machine code emitted with the MCJIT. This method returns true if machine
/// code is not supported. It fills the MCContext Ctx pointer which can be
/// used to build custom MCStreamer.
///
bool LLVMTargetMachine::addPassesToEmitMC(PassManagerBase &PM, MCContext *&Ctx,
                                          raw_pwrite_stream &Out,
                                          bool DisableVerify) {
  // Add common CodeGen passes.
  Ctx = addPassesToGenerateCode(this, PM, DisableVerify, nullptr, nullptr,
                                nullptr);
  if (!Ctx)
    return true;

  if (Options.MCOptions.MCSaveTempLabels)
    Ctx->setAllowTemporaryLabels(false);

  // Create the code emitter for the target if it exists.  If not, .o file
  // emission fails.
  const MCRegisterInfo &MRI = *getMCRegisterInfo();
  MCCodeEmitter *MCE =
      getTarget().createMCCodeEmitter(*getMCInstrInfo(), MRI, *Ctx);
  MCAsmBackend *MAB =
      getTarget().createMCAsmBackend(MRI, getTargetTriple().str(), TargetCPU);
  if (!MCE || !MAB)
    return true;

  const Triple &T = getTargetTriple();
  const MCSubtargetInfo &STI = *getMCSubtargetInfo();
  std::unique_ptr<MCStreamer> AsmStreamer(getTarget().createMCObjectStreamer(
      T, *Ctx, *MAB, Out, MCE, STI, Options.MCOptions.MCRelaxAll,
      /*DWARFMustBeAtTheEnd*/ true));

  // Create the AsmPrinter, which takes ownership of AsmStreamer if successful.
  FunctionPass *Printer =
      getTarget().createAsmPrinter(*this, std::move(AsmStreamer));
  if (!Printer)
    return true;

  PM.add(Printer);

  return false; // success!
}
