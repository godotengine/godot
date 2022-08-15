//===-LTOCodeGenerator.cpp - LLVM Link Time Optimizer ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Link Time Optimization library. This library is
// intended to be used by linker to optimize code at link time.
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/LTOCodeGenerator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/CodeGen/RuntimeLibcalls.h"
#include "llvm/Config/config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/LTO/LTOModule.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/ObjCARC.h"
#include <system_error>
using namespace llvm;

const char* LTOCodeGenerator::getVersionString() {
#ifdef LLVM_VERSION_INFO
  return PACKAGE_NAME " version " PACKAGE_VERSION ", " LLVM_VERSION_INFO;
#else
  return PACKAGE_NAME " version " PACKAGE_VERSION;
#endif
}

LTOCodeGenerator::LTOCodeGenerator()
    : Context(getGlobalContext()), IRLinker(new Module("ld-temp.o", Context)) {
  initializeLTOPasses();
}

LTOCodeGenerator::LTOCodeGenerator(std::unique_ptr<LLVMContext> Context)
    : OwnedContext(std::move(Context)), Context(*OwnedContext),
      IRLinker(new Module("ld-temp.o", *OwnedContext)) {
  initializeLTOPasses();
}

void LTOCodeGenerator::destroyMergedModule() {
  if (OwnedModule) {
    assert(IRLinker.getModule() == &OwnedModule->getModule() &&
           "The linker's module should be the same as the owned module");
    delete OwnedModule;
    OwnedModule = nullptr;
  } else if (IRLinker.getModule())
    IRLinker.deleteModule();
}

LTOCodeGenerator::~LTOCodeGenerator() {
  destroyMergedModule();

  delete TargetMach;
  TargetMach = nullptr;

  for (std::vector<char *>::iterator I = CodegenOptions.begin(),
                                     E = CodegenOptions.end();
       I != E; ++I)
    free(*I);
}

// Initialize LTO passes. Please keep this funciton in sync with
// PassManagerBuilder::populateLTOPassManager(), and make sure all LTO
// passes are initialized.
void LTOCodeGenerator::initializeLTOPasses() {
  PassRegistry &R = *PassRegistry::getPassRegistry();

  initializeInternalizePassPass(R);
  initializeIPSCCPPass(R);
  initializeGlobalOptPass(R);
  initializeConstantMergePass(R);
  initializeDAHPass(R);
  initializeInstructionCombiningPassPass(R);
  initializeSimpleInlinerPass(R);
  initializePruneEHPass(R);
  initializeGlobalDCEPass(R);
  initializeArgPromotionPass(R);
  initializeJumpThreadingPass(R);
  initializeSROAPass(R);
  initializeSROA_DTPass(R);
  initializeSROA_SSAUpPass(R);
  initializeFunctionAttrsPass(R);
  initializeGlobalsModRefPass(R);
  initializeLICMPass(R);
  initializeMergedLoadStoreMotionPass(R);
  initializeGVNPass(R);
  initializeMemCpyOptPass(R);
  initializeDCEPass(R);
  initializeCFGSimplifyPassPass(R);
}

bool LTOCodeGenerator::addModule(LTOModule *mod) {
  assert(&mod->getModule().getContext() == &Context &&
         "Expected module in same context");

  bool ret = IRLinker.linkInModule(&mod->getModule());

  const std::vector<const char*> &undefs = mod->getAsmUndefinedRefs();
  for (int i = 0, e = undefs.size(); i != e; ++i)
    AsmUndefinedRefs[undefs[i]] = 1;

  return !ret;
}

void LTOCodeGenerator::setModule(LTOModule *Mod) {
  assert(&Mod->getModule().getContext() == &Context &&
         "Expected module in same context");

  // Delete the old merged module.
  destroyMergedModule();
  AsmUndefinedRefs.clear();

  OwnedModule = Mod;
  IRLinker.setModule(&Mod->getModule());

  const std::vector<const char*> &Undefs = Mod->getAsmUndefinedRefs();
  for (int I = 0, E = Undefs.size(); I != E; ++I)
    AsmUndefinedRefs[Undefs[I]] = 1;
}

void LTOCodeGenerator::setTargetOptions(TargetOptions options) {
  Options = options;
}

void LTOCodeGenerator::setDebugInfo(lto_debug_model debug) {
  switch (debug) {
  case LTO_DEBUG_MODEL_NONE:
    EmitDwarfDebugInfo = false;
    return;

  case LTO_DEBUG_MODEL_DWARF:
    EmitDwarfDebugInfo = true;
    return;
  }
  llvm_unreachable("Unknown debug format!");
}

void LTOCodeGenerator::setCodePICModel(lto_codegen_model model) {
  switch (model) {
  case LTO_CODEGEN_PIC_MODEL_STATIC:
  case LTO_CODEGEN_PIC_MODEL_DYNAMIC:
  case LTO_CODEGEN_PIC_MODEL_DYNAMIC_NO_PIC:
  case LTO_CODEGEN_PIC_MODEL_DEFAULT:
    CodeModel = model;
    return;
  }
  llvm_unreachable("Unknown PIC model!");
}

bool LTOCodeGenerator::writeMergedModules(const char *path,
                                          std::string &errMsg) {
  if (!determineTarget(errMsg))
    return false;

  // mark which symbols can not be internalized
  applyScopeRestrictions();

  // create output file
  std::error_code EC;
  tool_output_file Out(path, EC, sys::fs::F_None);
  if (EC) {
    errMsg = "could not open bitcode file for writing: ";
    errMsg += path;
    return false;
  }

  // write bitcode to it
  WriteBitcodeToFile(IRLinker.getModule(), Out.os(), ShouldEmbedUselists);
  Out.os().close();

  if (Out.os().has_error()) {
    errMsg = "could not write bitcode file: ";
    errMsg += path;
    Out.os().clear_error();
    return false;
  }

  Out.keep();
  return true;
}

bool LTOCodeGenerator::compileOptimizedToFile(const char **name,
                                              std::string &errMsg) {
  // make unique temp .o file to put generated object file
  SmallString<128> Filename;
  int FD;
  std::error_code EC =
      sys::fs::createTemporaryFile("lto-llvm", "o", FD, Filename);
  if (EC) {
    errMsg = EC.message();
    return false;
  }

  // generate object file
  tool_output_file objFile(Filename.c_str(), FD);

  bool genResult = compileOptimized(objFile.os(), errMsg);
  objFile.os().close();
  if (objFile.os().has_error()) {
    objFile.os().clear_error();
    sys::fs::remove(Twine(Filename));
    return false;
  }

  objFile.keep();
  if (!genResult) {
    sys::fs::remove(Twine(Filename));
    return false;
  }

  NativeObjectPath = Filename.c_str();
  *name = NativeObjectPath.c_str();
  return true;
}

std::unique_ptr<MemoryBuffer>
LTOCodeGenerator::compileOptimized(std::string &errMsg) {
  const char *name;
  if (!compileOptimizedToFile(&name, errMsg))
    return nullptr;

  // read .o file into memory buffer
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(name, -1, false);
  if (std::error_code EC = BufferOrErr.getError()) {
    errMsg = EC.message();
    sys::fs::remove(NativeObjectPath);
    return nullptr;
  }

  // remove temp files
  sys::fs::remove(NativeObjectPath);

  return std::move(*BufferOrErr);
}


bool LTOCodeGenerator::compile_to_file(const char **name,
                                       bool disableInline,
                                       bool disableGVNLoadPRE,
                                       bool disableVectorization,
                                       std::string &errMsg) {
  if (!optimize(disableInline, disableGVNLoadPRE,
                disableVectorization, errMsg))
    return false;

  return compileOptimizedToFile(name, errMsg);
}

std::unique_ptr<MemoryBuffer>
LTOCodeGenerator::compile(bool disableInline, bool disableGVNLoadPRE,
                          bool disableVectorization, std::string &errMsg) {
  if (!optimize(disableInline, disableGVNLoadPRE,
                disableVectorization, errMsg))
    return nullptr;

  return compileOptimized(errMsg);
}

bool LTOCodeGenerator::determineTarget(std::string &errMsg) {
  if (TargetMach)
    return true;

  std::string TripleStr = IRLinker.getModule()->getTargetTriple();
  if (TripleStr.empty())
    TripleStr = sys::getDefaultTargetTriple();
  llvm::Triple Triple(TripleStr);

  // create target machine from info for merged modules
  const Target *march = TargetRegistry::lookupTarget(TripleStr, errMsg);
  if (!march)
    return false;

  // The relocation model is actually a static member of TargetMachine and
  // needs to be set before the TargetMachine is instantiated.
  Reloc::Model RelocModel = Reloc::Default;
  switch (CodeModel) {
  case LTO_CODEGEN_PIC_MODEL_STATIC:
    RelocModel = Reloc::Static;
    break;
  case LTO_CODEGEN_PIC_MODEL_DYNAMIC:
    RelocModel = Reloc::PIC_;
    break;
  case LTO_CODEGEN_PIC_MODEL_DYNAMIC_NO_PIC:
    RelocModel = Reloc::DynamicNoPIC;
    break;
  case LTO_CODEGEN_PIC_MODEL_DEFAULT:
    // RelocModel is already the default, so leave it that way.
    break;
  }

  // Construct LTOModule, hand over ownership of module and target. Use MAttr as
  // the default set of features.
  SubtargetFeatures Features(MAttr);
  Features.getDefaultSubtargetFeatures(Triple);
  std::string FeatureStr = Features.getString();
  // Set a default CPU for Darwin triples.
  if (MCpu.empty() && Triple.isOSDarwin()) {
    if (Triple.getArch() == llvm::Triple::x86_64)
      MCpu = "core2";
    else if (Triple.getArch() == llvm::Triple::x86)
      MCpu = "yonah";
    else if (Triple.getArch() == llvm::Triple::aarch64)
      MCpu = "cyclone";
  }

  CodeGenOpt::Level CGOptLevel;
  switch (OptLevel) {
  case 0:
    CGOptLevel = CodeGenOpt::None;
    break;
  case 1:
    CGOptLevel = CodeGenOpt::Less;
    break;
  case 2:
    CGOptLevel = CodeGenOpt::Default;
    break;
  case 3:
    CGOptLevel = CodeGenOpt::Aggressive;
    break;
  }

  TargetMach = march->createTargetMachine(TripleStr, MCpu, FeatureStr, Options,
                                          RelocModel, CodeModel::Default,
                                          CGOptLevel);
  return true;
}

void LTOCodeGenerator::
applyRestriction(GlobalValue &GV,
                 ArrayRef<StringRef> Libcalls,
                 std::vector<const char*> &MustPreserveList,
                 SmallPtrSetImpl<GlobalValue*> &AsmUsed,
                 Mangler &Mangler) {
  // There are no restrictions to apply to declarations.
  if (GV.isDeclaration())
    return;

  // There is nothing more restrictive than private linkage.
  if (GV.hasPrivateLinkage())
    return;

  SmallString<64> Buffer;
  TargetMach->getNameWithPrefix(Buffer, &GV, Mangler);

  if (MustPreserveSymbols.count(Buffer))
    MustPreserveList.push_back(GV.getName().data());
  if (AsmUndefinedRefs.count(Buffer))
    AsmUsed.insert(&GV);

  // Conservatively append user-supplied runtime library functions to
  // llvm.compiler.used.  These could be internalized and deleted by
  // optimizations like -globalopt, causing problems when later optimizations
  // add new library calls (e.g., llvm.memset => memset and printf => puts).
  // Leave it to the linker to remove any dead code (e.g. with -dead_strip).
  if (isa<Function>(GV) &&
      std::binary_search(Libcalls.begin(), Libcalls.end(), GV.getName()))
    AsmUsed.insert(&GV);
}

static void findUsedValues(GlobalVariable *LLVMUsed,
                           SmallPtrSetImpl<GlobalValue*> &UsedValues) {
  if (!LLVMUsed) return;

  ConstantArray *Inits = cast<ConstantArray>(LLVMUsed->getInitializer());
  for (unsigned i = 0, e = Inits->getNumOperands(); i != e; ++i)
    if (GlobalValue *GV =
        dyn_cast<GlobalValue>(Inits->getOperand(i)->stripPointerCasts()))
      UsedValues.insert(GV);
}

// Collect names of runtime library functions. User-defined functions with the
// same names are added to llvm.compiler.used to prevent them from being
// deleted by optimizations.
static void accumulateAndSortLibcalls(std::vector<StringRef> &Libcalls,
                                      const TargetLibraryInfo& TLI,
                                      const Module &Mod,
                                      const TargetMachine &TM) {
  // TargetLibraryInfo has info on C runtime library calls on the current
  // target.
  for (unsigned I = 0, E = static_cast<unsigned>(LibFunc::NumLibFuncs);
       I != E; ++I) {
    LibFunc::Func F = static_cast<LibFunc::Func>(I);
    if (TLI.has(F))
      Libcalls.push_back(TLI.getName(F));
  }

  SmallPtrSet<const TargetLowering *, 1> TLSet;

  for (const Function &F : Mod) {
    const TargetLowering *Lowering =
        TM.getSubtargetImpl(F)->getTargetLowering();

    if (Lowering && TLSet.insert(Lowering).second)
      // TargetLowering has info on library calls that CodeGen expects to be
      // available, both from the C runtime and compiler-rt.
      for (unsigned I = 0, E = static_cast<unsigned>(RTLIB::UNKNOWN_LIBCALL);
           I != E; ++I)
        if (const char *Name =
                Lowering->getLibcallName(static_cast<RTLIB::Libcall>(I)))
          Libcalls.push_back(Name);
  }

  array_pod_sort(Libcalls.begin(), Libcalls.end());
  Libcalls.erase(std::unique(Libcalls.begin(), Libcalls.end()),
                 Libcalls.end());
}

void LTOCodeGenerator::applyScopeRestrictions() {
  if (ScopeRestrictionsDone || !ShouldInternalize)
    return;
  Module *mergedModule = IRLinker.getModule();

  // Start off with a verification pass.
  legacy::PassManager passes;
  passes.add(createVerifierPass());

  // mark which symbols can not be internalized
  Mangler Mangler;
  std::vector<const char*> MustPreserveList;
  SmallPtrSet<GlobalValue*, 8> AsmUsed;
  std::vector<StringRef> Libcalls;
  TargetLibraryInfoImpl TLII(Triple(TargetMach->getTargetTriple()));
  TargetLibraryInfo TLI(TLII);

  accumulateAndSortLibcalls(Libcalls, TLI, *mergedModule, *TargetMach);

  for (Module::iterator f = mergedModule->begin(),
         e = mergedModule->end(); f != e; ++f)
    applyRestriction(*f, Libcalls, MustPreserveList, AsmUsed, Mangler);
  for (Module::global_iterator v = mergedModule->global_begin(),
         e = mergedModule->global_end(); v !=  e; ++v)
    applyRestriction(*v, Libcalls, MustPreserveList, AsmUsed, Mangler);
  for (Module::alias_iterator a = mergedModule->alias_begin(),
         e = mergedModule->alias_end(); a != e; ++a)
    applyRestriction(*a, Libcalls, MustPreserveList, AsmUsed, Mangler);

  GlobalVariable *LLVMCompilerUsed =
    mergedModule->getGlobalVariable("llvm.compiler.used");
  findUsedValues(LLVMCompilerUsed, AsmUsed);
  if (LLVMCompilerUsed)
    LLVMCompilerUsed->eraseFromParent();

  if (!AsmUsed.empty()) {
    llvm::Type *i8PTy = llvm::Type::getInt8PtrTy(Context);
    std::vector<Constant*> asmUsed2;
    for (auto *GV : AsmUsed) {
      Constant *c = ConstantExpr::getBitCast(GV, i8PTy);
      asmUsed2.push_back(c);
    }

    llvm::ArrayType *ATy = llvm::ArrayType::get(i8PTy, asmUsed2.size());
    LLVMCompilerUsed =
      new llvm::GlobalVariable(*mergedModule, ATy, false,
                               llvm::GlobalValue::AppendingLinkage,
                               llvm::ConstantArray::get(ATy, asmUsed2),
                               "llvm.compiler.used");

    LLVMCompilerUsed->setSection("llvm.metadata");
  }

  passes.add(createInternalizePass(MustPreserveList));

  // apply scope restrictions
  passes.run(*mergedModule);

  ScopeRestrictionsDone = true;
}

/// Optimize merged modules using various IPO passes
bool LTOCodeGenerator::optimize(bool DisableInline,
                                bool DisableGVNLoadPRE,
                                bool DisableVectorization,
                                std::string &errMsg) {
  if (!this->determineTarget(errMsg))
    return false;

  Module *mergedModule = IRLinker.getModule();

  // Mark which symbols can not be internalized
  this->applyScopeRestrictions();

  // Instantiate the pass manager to organize the passes.
  legacy::PassManager passes;

  // Add an appropriate DataLayout instance for this module...
  mergedModule->setDataLayout(*TargetMach->getDataLayout());

  passes.add(
      createTargetTransformInfoWrapperPass(TargetMach->getTargetIRAnalysis()));

  Triple TargetTriple(TargetMach->getTargetTriple());
  PassManagerBuilder PMB;
  PMB.DisableGVNLoadPRE = DisableGVNLoadPRE;
  PMB.LoopVectorize = !DisableVectorization;
  PMB.SLPVectorize = !DisableVectorization;
  if (!DisableInline)
    PMB.Inliner = createFunctionInliningPass();
  PMB.LibraryInfo = new TargetLibraryInfoImpl(TargetTriple);
  PMB.OptLevel = OptLevel;
  PMB.VerifyInput = true;
  PMB.VerifyOutput = true;

  PMB.populateLTOPassManager(passes);

  // Run our queue of passes all at once now, efficiently.
  passes.run(*mergedModule);

  return true;
}

bool LTOCodeGenerator::compileOptimized(raw_pwrite_stream &out,
                                        std::string &errMsg) {
  if (!this->determineTarget(errMsg))
    return false;

  Module *mergedModule = IRLinker.getModule();

  legacy::PassManager codeGenPasses;

  // If the bitcode files contain ARC code and were compiled with optimization,
  // the ObjCARCContractPass must be run, so do it unconditionally here.
  codeGenPasses.add(createObjCARCContractPass());

  if (TargetMach->addPassesToEmitFile(codeGenPasses, out,
                                      TargetMachine::CGFT_ObjectFile)) {
    errMsg = "target file type not supported";
    return false;
  }

  // Run the code generator, and write assembly file
  codeGenPasses.run(*mergedModule);

  return true;
}

/// setCodeGenDebugOptions - Set codegen debugging options to aid in debugging
/// LTO problems.
void LTOCodeGenerator::setCodeGenDebugOptions(const char *options) {
  for (std::pair<StringRef, StringRef> o = getToken(options);
       !o.first.empty(); o = getToken(o.second)) {
    // ParseCommandLineOptions() expects argv[0] to be program name. Lazily add
    // that.
    if (CodegenOptions.empty())
      CodegenOptions.push_back(strdup("libLLVMLTO"));
    CodegenOptions.push_back(strdup(o.first.str().c_str()));
  }
}

void LTOCodeGenerator::parseCodeGenDebugOptions() {
  // if options were requested, set them
  if (!CodegenOptions.empty())
    cl::ParseCommandLineOptions(CodegenOptions.size(),
                                const_cast<char **>(&CodegenOptions[0]));
}

void LTOCodeGenerator::DiagnosticHandler(const DiagnosticInfo &DI,
                                         void *Context) {
  ((LTOCodeGenerator *)Context)->DiagnosticHandler2(DI);
}

void LTOCodeGenerator::DiagnosticHandler2(const DiagnosticInfo &DI) {
  // Map the LLVM internal diagnostic severity to the LTO diagnostic severity.
  lto_codegen_diagnostic_severity_t Severity;
  switch (DI.getSeverity()) {
  case DS_Error:
    Severity = LTO_DS_ERROR;
    break;
  case DS_Warning:
    Severity = LTO_DS_WARNING;
    break;
  case DS_Remark:
    Severity = LTO_DS_REMARK;
    break;
  case DS_Note:
    Severity = LTO_DS_NOTE;
    break;
  }
  // Create the string that will be reported to the external diagnostic handler.
  std::string MsgStorage;
  raw_string_ostream Stream(MsgStorage);
  DiagnosticPrinterRawOStream DP(Stream);
  DI.print(DP);
  Stream.flush();

  // If this method has been called it means someone has set up an external
  // diagnostic handler. Assert on that.
  assert(DiagHandler && "Invalid diagnostic handler");
  (*DiagHandler)(Severity, MsgStorage.c_str(), DiagContext);
}

void
LTOCodeGenerator::setDiagnosticHandler(lto_diagnostic_handler_t DiagHandler,
                                       void *Ctxt) {
  this->DiagHandler = DiagHandler;
  this->DiagContext = Ctxt;
  if (!DiagHandler)
    return Context.setDiagnosticHandler(nullptr, nullptr);
  // Register the LTOCodeGenerator stub in the LLVMContext to forward the
  // diagnostic to the external DiagHandler.
  Context.setDiagnosticHandler(LTOCodeGenerator::DiagnosticHandler, this,
                               /* RespectFilters */ true);
}
