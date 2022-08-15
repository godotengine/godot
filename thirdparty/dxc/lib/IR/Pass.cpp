//===- Pass.cpp - LLVM Pass Infrastructure Implementation -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVM Pass infrastructure.  It is primarily
// responsible with ensuring that passes are executed and batched together
// optimally.
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm> // HLSL Change
using namespace llvm;

#define DEBUG_TYPE "ir"

//===----------------------------------------------------------------------===//
// Pass Implementation
//

// Force out-of-line virtual method.
Pass::~Pass() {
  delete Resolver;
}

// HLSL Change Starts

void Pass::dumpConfig(llvm::raw_ostream &OS) {
  OS << '-' << this->getPassArgument();
}

const char *Pass::getPassArgument() const {
  AnalysisID AID = getPassID();
  const PassInfo *PI = PassRegistry::getPassRegistry()->getPassInfo(AID);
  if (PI)
    return PI->getPassArgument();
  return "Unnamed pass: implement Pass::getPassArgument()";
}

// HLSL Change Ends

// Force out-of-line virtual method.
ModulePass::~ModulePass() { }

Pass *ModulePass::createPrinterPass(raw_ostream &O,
                                    const std::string &Banner) const {
  return createPrintModulePass(O, Banner);
}

PassManagerType ModulePass::getPotentialPassManagerType() const {
  return PMT_ModulePassManager;
}

bool Pass::mustPreserveAnalysisID(char &AID) const {
  return Resolver->getAnalysisIfAvailable(&AID, true) != nullptr;
}

// dumpPassStructure - Implement the -debug-pass=Structure option
void Pass::dumpPassStructure(unsigned Offset) {
  dbgs().indent(Offset*2) << getPassName() << "\n";
}

/// getPassName - Return a nice clean name for a pass.  This usually
/// implemented in terms of the name that is registered by one of the
/// Registration templates, but can be overloaded directly.
///
StringRef Pass::getPassName() const {
  AnalysisID AID =  getPassID();
  const PassInfo *PI = PassRegistry::getPassRegistry()->getPassInfo(AID);
  if (PI)
    return PI->getPassName();
  return "Unnamed pass: implement Pass::getPassName()";
}

void Pass::preparePassManager(PMStack &) {
  // By default, don't do anything.
}

PassManagerType Pass::getPotentialPassManagerType() const {
  // Default implementation.
  return PMT_Unknown;
}

void Pass::getAnalysisUsage(AnalysisUsage &) const {
  // By default, no analysis results are used, all are invalidated.
}

void Pass::releaseMemory() {
  // By default, don't do anything.
}

void Pass::verifyAnalysis() const {
  // By default, don't do anything.
}

void *Pass::getAdjustedAnalysisPointer(AnalysisID AID) {
  return this;
}

ImmutablePass *Pass::getAsImmutablePass() {
  return nullptr;
}

PMDataManager *Pass::getAsPMDataManager() {
  return nullptr;
}

void Pass::setResolver(AnalysisResolver *AR) {
  assert(!Resolver && "Resolver is already set");
  Resolver = AR;
}

// print - Print out the internal state of the pass.  This is called by Analyze
// to print out the contents of an analysis.  Otherwise it is not necessary to
// implement this method.
//
void Pass::print(raw_ostream &O,const Module*) const {
  O << "Pass::print not implemented for pass: '" << getPassName() << "'!\n";
}

// dump - call print(cerr);
void Pass::dump() const {
  print(dbgs(), nullptr);
}

//===----------------------------------------------------------------------===//
// ImmutablePass Implementation
//
// Force out-of-line virtual method.
ImmutablePass::~ImmutablePass() { }

void ImmutablePass::initializePass() {
  // By default, don't do anything.
}

//===----------------------------------------------------------------------===//
// FunctionPass Implementation
//

Pass *FunctionPass::createPrinterPass(raw_ostream &O,
                                      const std::string &Banner) const {
  return createPrintFunctionPass(O, Banner);
}

PassManagerType FunctionPass::getPotentialPassManagerType() const {
  return PMT_FunctionPassManager;
}

bool FunctionPass::skipOptnoneFunction(const Function &F) const {
  if (F.hasFnAttribute(Attribute::OptimizeNone)) {
    DEBUG(dbgs() << "Skipping pass '" << getPassName()
          << "' on function " << F.getName() << "\n");
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// BasicBlockPass Implementation
//

Pass *BasicBlockPass::createPrinterPass(raw_ostream &O,
                                        const std::string &Banner) const {
  return createPrintBasicBlockPass(O, Banner);
}

bool BasicBlockPass::doInitialization(Function &) {
  // By default, don't do anything.
  return false;
}

bool BasicBlockPass::doFinalization(Function &) {
  // By default, don't do anything.
  return false;
}

bool BasicBlockPass::skipOptnoneFunction(const BasicBlock &BB) const {
  const Function *F = BB.getParent();
  if (F && F->hasFnAttribute(Attribute::OptimizeNone)) {
    // Report this only once per function.
    if (&BB == &F->getEntryBlock())
      DEBUG(dbgs() << "Skipping pass '" << getPassName()
            << "' on function " << F->getName() << "\n");
    return true;
  }
  return false;
}

PassManagerType BasicBlockPass::getPotentialPassManagerType() const {
  return PMT_BasicBlockPassManager;
}

const PassInfo *Pass::lookupPassInfo(const void *TI) {
  return PassRegistry::getPassRegistry()->getPassInfo(TI);
}

const PassInfo *Pass::lookupPassInfo(StringRef Arg) {
  return PassRegistry::getPassRegistry()->getPassInfo(Arg);
}

Pass *Pass::createPass(AnalysisID ID) {
  const PassInfo *PI = PassRegistry::getPassRegistry()->getPassInfo(ID);
  if (!PI)
    return nullptr;
  return PI->createPass();
}

//===----------------------------------------------------------------------===//
//                  Analysis Group Implementation Code
//===----------------------------------------------------------------------===//

// RegisterAGBase implementation
//
RegisterAGBase::RegisterAGBase(const char *Name, const void *InterfaceID,
                               const void *PassID, bool isDefault)
    : PassInfo(Name, InterfaceID) {
  PassRegistry::getPassRegistry()->registerAnalysisGroup(InterfaceID, PassID,
                                                         *this, isDefault);
}

//===----------------------------------------------------------------------===//
// PassRegistrationListener implementation
//

// enumeratePasses - Iterate over the registered passes, calling the
// passEnumerate callback on each PassInfo object.
//
void PassRegistrationListener::enumeratePasses() {
  PassRegistry::getPassRegistry()->enumerateWith(this);
}

PassNameParser::PassNameParser(cl::Option &O)
    : cl::parser<const PassInfo *>(O) {
  PassRegistry::getPassRegistry()->addRegistrationListener(this);
}

PassNameParser::~PassNameParser() {
  // This only gets called during static destruction, in which case the
  // PassRegistry will have already been destroyed by llvm_shutdown().  So
  // attempting to remove the registration listener is an error.
}

//===----------------------------------------------------------------------===//
//   AnalysisUsage Class Implementation
//

namespace {
  struct GetCFGOnlyPasses : public PassRegistrationListener {
    typedef AnalysisUsage::VectorType VectorType;
    VectorType &CFGOnlyList;
    GetCFGOnlyPasses(VectorType &L) : CFGOnlyList(L) {}

    void passEnumerate(const PassInfo *P) override {
      if (P->isCFGOnlyPass())
        CFGOnlyList.push_back(P->getTypeInfo());
    }
  };
}

// setPreservesCFG - This function should be called to by the pass, iff they do
// not:
//
//  1. Add or remove basic blocks from the function
//  2. Modify terminator instructions in any way.
//
// This function annotates the AnalysisUsage info object to say that analyses
// that only depend on the CFG are preserved by this pass.
//
void AnalysisUsage::setPreservesCFG() {
  // Since this transformation doesn't modify the CFG, it preserves all analyses
  // that only depend on the CFG (like dominators, loop info, etc...)
  GetCFGOnlyPasses(Preserved).enumeratePasses();
}

AnalysisUsage &AnalysisUsage::addPreserved(StringRef Arg) {
  const PassInfo *PI = Pass::lookupPassInfo(Arg);
  // If the pass exists, preserve it. Otherwise silently do nothing.
  if (PI) Preserved.push_back(PI->getTypeInfo());
  return *this;
}

AnalysisUsage &AnalysisUsage::addRequiredID(const void *ID) {
  Required.push_back(ID);
  return *this;
}

AnalysisUsage &AnalysisUsage::addRequiredID(char &ID) {
  Required.push_back(&ID);
  return *this;
}

AnalysisUsage &AnalysisUsage::addRequiredTransitiveID(char &ID) {
  Required.push_back(&ID);
  RequiredTransitive.push_back(&ID);
  return *this;
}

// HLSL Changes Start

bool llvm::GetPassOption(PassOptions &O, StringRef name, StringRef *pValue) {
  PassOption OpVal(name, StringRef());
  const PassOption *Op = std::lower_bound(O.begin(), O.end(), OpVal, PassOptionsCompare());
  if (Op != O.end() && Op->first == name) {
    *pValue = Op->second;
    return true;
  }
  return false;
}
bool llvm::GetPassOptionBool(PassOptions &O, llvm::StringRef name, bool *pValue, bool defaultValue) {
  StringRef val;
  if (GetPassOption(O, name, &val)) {
    *pValue = (val.startswith_lower("t") || val.equals("1"));
    return true;
  }
  *pValue = defaultValue;
  return false;
}
bool llvm::GetPassOptionUnsigned(PassOptions &O, llvm::StringRef name, unsigned *pValue, unsigned defaultValue) {
  StringRef val;
  if (GetPassOption(O, name, &val)) {
    val.getAsInteger<unsigned>(0, *pValue);
    return true;
  }
  *pValue = defaultValue;
  return false;
}
bool llvm::GetPassOptionInt(PassOptions &O, llvm::StringRef name, int *pValue, int defaultValue) {
  StringRef val;
  if (GetPassOption(O, name, &val)) {
    val.getAsInteger<int>(0, *pValue);
    return true;
  }
  *pValue = defaultValue;
  return false;
}
bool llvm::GetPassOptionUInt32(PassOptions &O, llvm::StringRef name, uint32_t *pValue, uint32_t defaultValue) {
  StringRef val;
  if (GetPassOption(O, name, &val)) {
    val.getAsInteger<uint32_t>(0, *pValue);
    return true;
  }
  *pValue = defaultValue;
  return false;
}
bool llvm::GetPassOptionUInt64(PassOptions &O, llvm::StringRef name, uint64_t *pValue, uint64_t defaultValue) {
  StringRef val;
  if (GetPassOption(O, name, &val)) {
    val.getAsInteger<uint64_t>(0, *pValue);
    return true;
  }
  *pValue = defaultValue;
  return false;
}
bool llvm::GetPassOptionFloat(PassOptions &O, llvm::StringRef name, float *pValue, float defaultValue) {
  StringRef val;
  if (GetPassOption(O, name, &val)) {
    std::string str;
    str.assign(val.begin(), val.end());
    *pValue = atof(str.c_str());
    return true;
  }
  *pValue = defaultValue;
  return false;
}

// HLSL Changes End
