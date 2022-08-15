///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// PauseResumePasses.cpp                                                     //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Passes to pause/resume pipeline.                                          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/DxilGenerationPass.h"

#include "llvm/Pass.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace hlsl;

static const char kPauseResumeMDName[] = "pauseresume";
static const char kPauseResumeNumFields = 2;
static const char kPauseResumePassNameToPause = 0;
static const char kPauseResumePassNameToResume = 1;

namespace hlsl {

bool ClearPauseResumePasses(Module &M) {
  NamedMDNode *N = M.getNamedMetadata(kPauseResumeMDName);
  if (N) {
    M.eraseNamedMetadata(N);
    return true;
  }
  return false;
}

void GetPauseResumePasses(Module &M, StringRef &pause, StringRef &resume) {
  NamedMDNode *N = M.getNamedMetadata(kPauseResumeMDName);
  if (N && N->getNumOperands() > 0) {
    MDNode *MD = N->getOperand(0);
    pause = dyn_cast<MDString>(MD->getOperand(kPauseResumePassNameToPause).get())->getString();
    resume = dyn_cast<MDString>(MD->getOperand(kPauseResumePassNameToResume).get())->getString();
  }
}

void SetPauseResumePasses(Module &M, StringRef pause, StringRef resume) {
  LLVMContext &Ctx = M.getContext();
  NamedMDNode *N = M.getOrInsertNamedMetadata(kPauseResumeMDName);
  Metadata *MDs[kPauseResumeNumFields];
  MDs[(int)kPauseResumePassNameToPause] = MDString::get(Ctx, pause);
  MDs[(int)kPauseResumePassNameToResume] = MDString::get(Ctx, resume);
  if (N->getNumOperands() == 0)
    N->addOperand(MDNode::get(Ctx, MDs));
  else
    N->setOperand(kPauseResumePassNameToPause, MDNode::get(Ctx, MDs));
}

}

namespace {

class NoPausePasses : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit NoPausePasses() : ModulePass(ID) {}

  StringRef getPassName() const override { return "NoPausePasses"; }

  bool runOnModule(Module &M) override {
    return ClearPauseResumePasses(M);
  }
};

class PausePasses : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit PausePasses() : ModulePass(ID) {}

  StringRef getPassName() const override { return "PausePasses"; }

  bool runOnModule(Module &M) override {
    StringRef pause, resume;
    hlsl::GetPauseResumePasses(M, pause, resume);
    if (!pause.empty()) {
      const PassInfo *PI = PassRegistry::getPassRegistry()->getPassInfo(pause);
      std::unique_ptr<ModulePass> pass((ModulePass *)PI->createPass());
      pass->setOSOverride(OSOverride);
      return pass->runOnModule(M);
    }
    return false;
  }
};

class ResumePasses : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit ResumePasses() : ModulePass(ID) {}

  StringRef getPassName() const override { return "ResumePasses"; }

  bool runOnModule(Module &M) override {
    StringRef pause, resume;
    hlsl::GetPauseResumePasses(M, pause, resume);
    if (!resume.empty()) {
      const PassInfo *PI = PassRegistry::getPassRegistry()->getPassInfo(resume);
      std::unique_ptr<ModulePass> pass((ModulePass *)PI->createPass());
      pass->setOSOverride(OSOverride);
      return pass->runOnModule(M);
    }
    return false;
  }
};

char NoPausePasses::ID = 0;
char PausePasses::ID = 0;
char ResumePasses::ID = 0;

}

ModulePass *llvm::createNoPausePassesPass() {
  return new NoPausePasses();
}
ModulePass *llvm::createPausePassesPass() {
  return new PausePasses();
}
ModulePass *llvm::createResumePassesPass() {
  return new ResumePasses();
}

INITIALIZE_PASS(NoPausePasses,
                "hlsl-passes-nopause",
                "Clears metadata used for pause and resume", false, false)
INITIALIZE_PASS(PausePasses,
                "hlsl-passes-pause",
                "Prepare to pause passes", false, false)
INITIALIZE_PASS(ResumePasses,
                "hlsl-passes-resume",
                "Prepare to resume passes", false, false)
