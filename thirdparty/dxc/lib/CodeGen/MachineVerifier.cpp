//===-- MachineVerifier.cpp - Machine Code Verifier -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Pass to verify generated machine code. The following is checked:
//
// Operand counts: All explicit operands must be present.
//
// Register classes: All physical and virtual register operands must be
// compatible with the register class required by the instruction descriptor.
//
// Register live intervals: Registers must be defined only once, and must be
// defined before use.
//
// The machine code verifier is enabled from LLVMTargetMachine.cpp with the
// command-line option -verify-machineinstrs, or by defining the environment
// variable LLVM_VERIFY_MACHINEINSTRS to the name of a file that will receive
// the verifier errors.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

namespace {
  struct MachineVerifier {

    MachineVerifier(Pass *pass, const char *b) :
      PASS(pass),
      Banner(b)
      {}

    bool runOnMachineFunction(MachineFunction &MF);

    Pass *const PASS;
    const char *Banner;
    const MachineFunction *MF;
    const TargetMachine *TM;
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    const MachineRegisterInfo *MRI;

    unsigned foundErrors;

    typedef SmallVector<unsigned, 16> RegVector;
    typedef SmallVector<const uint32_t*, 4> RegMaskVector;
    typedef DenseSet<unsigned> RegSet;
    typedef DenseMap<unsigned, const MachineInstr*> RegMap;
    typedef SmallPtrSet<const MachineBasicBlock*, 8> BlockSet;

    const MachineInstr *FirstTerminator;
    BlockSet FunctionBlocks;

    BitVector regsReserved;
    RegSet regsLive;
    RegVector regsDefined, regsDead, regsKilled;
    RegMaskVector regMasks;
    RegSet regsLiveInButUnused;

    SlotIndex lastIndex;

    // Add Reg and any sub-registers to RV
    void addRegWithSubRegs(RegVector &RV, unsigned Reg) {
      RV.push_back(Reg);
      if (TargetRegisterInfo::isPhysicalRegister(Reg))
        for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs)
          RV.push_back(*SubRegs);
    }

    struct BBInfo {
      // Is this MBB reachable from the MF entry point?
      bool reachable;

      // Vregs that must be live in because they are used without being
      // defined. Map value is the user.
      RegMap vregsLiveIn;

      // Regs killed in MBB. They may be defined again, and will then be in both
      // regsKilled and regsLiveOut.
      RegSet regsKilled;

      // Regs defined in MBB and live out. Note that vregs passing through may
      // be live out without being mentioned here.
      RegSet regsLiveOut;

      // Vregs that pass through MBB untouched. This set is disjoint from
      // regsKilled and regsLiveOut.
      RegSet vregsPassed;

      // Vregs that must pass through MBB because they are needed by a successor
      // block. This set is disjoint from regsLiveOut.
      RegSet vregsRequired;

      // Set versions of block's predecessor and successor lists.
      BlockSet Preds, Succs;

      BBInfo() : reachable(false) {}

      // Add register to vregsPassed if it belongs there. Return true if
      // anything changed.
      bool addPassed(unsigned Reg) {
        if (!TargetRegisterInfo::isVirtualRegister(Reg))
          return false;
        if (regsKilled.count(Reg) || regsLiveOut.count(Reg))
          return false;
        return vregsPassed.insert(Reg).second;
      }

      // Same for a full set.
      bool addPassed(const RegSet &RS) {
        bool changed = false;
        for (RegSet::const_iterator I = RS.begin(), E = RS.end(); I != E; ++I)
          if (addPassed(*I))
            changed = true;
        return changed;
      }

      // Add register to vregsRequired if it belongs there. Return true if
      // anything changed.
      bool addRequired(unsigned Reg) {
        if (!TargetRegisterInfo::isVirtualRegister(Reg))
          return false;
        if (regsLiveOut.count(Reg))
          return false;
        return vregsRequired.insert(Reg).second;
      }

      // Same for a full set.
      bool addRequired(const RegSet &RS) {
        bool changed = false;
        for (RegSet::const_iterator I = RS.begin(), E = RS.end(); I != E; ++I)
          if (addRequired(*I))
            changed = true;
        return changed;
      }

      // Same for a full map.
      bool addRequired(const RegMap &RM) {
        bool changed = false;
        for (RegMap::const_iterator I = RM.begin(), E = RM.end(); I != E; ++I)
          if (addRequired(I->first))
            changed = true;
        return changed;
      }

      // Live-out registers are either in regsLiveOut or vregsPassed.
      bool isLiveOut(unsigned Reg) const {
        return regsLiveOut.count(Reg) || vregsPassed.count(Reg);
      }
    };

    // Extra register info per MBB.
    DenseMap<const MachineBasicBlock*, BBInfo> MBBInfoMap;

    bool isReserved(unsigned Reg) {
      return Reg < regsReserved.size() && regsReserved.test(Reg);
    }

    bool isAllocatable(unsigned Reg) {
      return Reg < TRI->getNumRegs() && MRI->isAllocatable(Reg);
    }

    // Analysis information if available
    LiveVariables *LiveVars;
    LiveIntervals *LiveInts;
    LiveStacks *LiveStks;
    SlotIndexes *Indexes;

    void visitMachineFunctionBefore();
    void visitMachineBasicBlockBefore(const MachineBasicBlock *MBB);
    void visitMachineBundleBefore(const MachineInstr *MI);
    void visitMachineInstrBefore(const MachineInstr *MI);
    void visitMachineOperand(const MachineOperand *MO, unsigned MONum);
    void visitMachineInstrAfter(const MachineInstr *MI);
    void visitMachineBundleAfter(const MachineInstr *MI);
    void visitMachineBasicBlockAfter(const MachineBasicBlock *MBB);
    void visitMachineFunctionAfter();

    void report(const char *msg, const MachineFunction *MF);
    void report(const char *msg, const MachineBasicBlock *MBB);
    void report(const char *msg, const MachineInstr *MI);
    void report(const char *msg, const MachineOperand *MO, unsigned MONum);
    void report(const char *msg, const MachineFunction *MF,
                const LiveInterval &LI);
    void report(const char *msg, const MachineBasicBlock *MBB,
                const LiveInterval &LI);
    void report(const char *msg, const MachineFunction *MF,
                const LiveRange &LR, unsigned Reg, unsigned LaneMask);
    void report(const char *msg, const MachineBasicBlock *MBB,
                const LiveRange &LR, unsigned Reg, unsigned LaneMask);

    void verifyInlineAsm(const MachineInstr *MI);

    void checkLiveness(const MachineOperand *MO, unsigned MONum);
    void markReachable(const MachineBasicBlock *MBB);
    void calcRegsPassed();
    void checkPHIOps(const MachineBasicBlock *MBB);

    void calcRegsRequired();
    void verifyLiveVariables();
    void verifyLiveIntervals();
    void verifyLiveInterval(const LiveInterval&);
    void verifyLiveRangeValue(const LiveRange&, const VNInfo*, unsigned,
                              unsigned);
    void verifyLiveRangeSegment(const LiveRange&,
                                const LiveRange::const_iterator I, unsigned,
                                unsigned);
    void verifyLiveRange(const LiveRange&, unsigned, unsigned LaneMask = 0);

    void verifyStackFrame();
  };

  struct MachineVerifierPass : public MachineFunctionPass {
    static char ID; // Pass ID, replacement for typeid
    const std::string Banner;

    MachineVerifierPass(const std::string &banner = nullptr)
      : MachineFunctionPass(ID), Banner(banner) {
        initializeMachineVerifierPassPass(*PassRegistry::getPassRegistry());
      }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    bool runOnMachineFunction(MachineFunction &MF) override {
      MF.verify(this, Banner.c_str());
      return false;
    }
  };

}

char MachineVerifierPass::ID = 0;
INITIALIZE_PASS(MachineVerifierPass, "machineverifier",
                "Verify generated machine code", false, false)

FunctionPass *llvm::createMachineVerifierPass(const std::string &Banner) {
  return new MachineVerifierPass(Banner);
}

void MachineFunction::verify(Pass *p, const char *Banner) const {
  MachineVerifier(p, Banner)
    .runOnMachineFunction(const_cast<MachineFunction&>(*this));
}

bool MachineVerifier::runOnMachineFunction(MachineFunction &MF) {
  foundErrors = 0;

  this->MF = &MF;
  TM = &MF.getTarget();
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();

  LiveVars = nullptr;
  LiveInts = nullptr;
  LiveStks = nullptr;
  Indexes = nullptr;
  if (PASS) {
    LiveInts = PASS->getAnalysisIfAvailable<LiveIntervals>();
    // We don't want to verify LiveVariables if LiveIntervals is available.
    if (!LiveInts)
      LiveVars = PASS->getAnalysisIfAvailable<LiveVariables>();
    LiveStks = PASS->getAnalysisIfAvailable<LiveStacks>();
    Indexes = PASS->getAnalysisIfAvailable<SlotIndexes>();
  }

  visitMachineFunctionBefore();
  for (MachineFunction::const_iterator MFI = MF.begin(), MFE = MF.end();
       MFI!=MFE; ++MFI) {
    visitMachineBasicBlockBefore(MFI);
    // Keep track of the current bundle header.
    const MachineInstr *CurBundle = nullptr;
    // Do we expect the next instruction to be part of the same bundle?
    bool InBundle = false;

    for (MachineBasicBlock::const_instr_iterator MBBI = MFI->instr_begin(),
           MBBE = MFI->instr_end(); MBBI != MBBE; ++MBBI) {
      if (MBBI->getParent() != MFI) {
        report("Bad instruction parent pointer", MFI);
        errs() << "Instruction: " << *MBBI;
        continue;
      }

      // Check for consistent bundle flags.
      if (InBundle && !MBBI->isBundledWithPred())
        report("Missing BundledPred flag, "
               "BundledSucc was set on predecessor", MBBI);
      if (!InBundle && MBBI->isBundledWithPred())
        report("BundledPred flag is set, "
               "but BundledSucc not set on predecessor", MBBI);

      // Is this a bundle header?
      if (!MBBI->isInsideBundle()) {
        if (CurBundle)
          visitMachineBundleAfter(CurBundle);
        CurBundle = MBBI;
        visitMachineBundleBefore(CurBundle);
      } else if (!CurBundle)
        report("No bundle header", MBBI);
      visitMachineInstrBefore(MBBI);
      for (unsigned I = 0, E = MBBI->getNumOperands(); I != E; ++I) {
        const MachineInstr &MI = *MBBI;
        const MachineOperand &Op = MI.getOperand(I);
        if (Op.getParent() != &MI) {
          // Make sure to use correct addOperand / RemoveOperand / ChangeTo
          // functions when replacing operands of a MachineInstr.
          report("Instruction has operand with wrong parent set", &MI);
        }

        visitMachineOperand(&Op, I);
      }

      visitMachineInstrAfter(MBBI);

      // Was this the last bundled instruction?
      InBundle = MBBI->isBundledWithSucc();
    }
    if (CurBundle)
      visitMachineBundleAfter(CurBundle);
    if (InBundle)
      report("BundledSucc flag set on last instruction in block", &MFI->back());
    visitMachineBasicBlockAfter(MFI);
  }
  visitMachineFunctionAfter();

  if (foundErrors)
    report_fatal_error("Found "+Twine(foundErrors)+" machine code errors.");

  // Clean up.
  regsLive.clear();
  regsDefined.clear();
  regsDead.clear();
  regsKilled.clear();
  regMasks.clear();
  regsLiveInButUnused.clear();
  MBBInfoMap.clear();

  return false;                 // no changes
}

void MachineVerifier::report(const char *msg, const MachineFunction *MF) {
  assert(MF);
  errs() << '\n';
  if (!foundErrors++) {
    if (Banner)
      errs() << "# " << Banner << '\n';
    MF->print(errs(), Indexes);
  }
  errs() << "*** Bad machine code: " << msg << " ***\n"
      << "- function:    " << MF->getName() << "\n";
}

void MachineVerifier::report(const char *msg, const MachineBasicBlock *MBB) {
  assert(MBB);
  report(msg, MBB->getParent());
  errs() << "- basic block: BB#" << MBB->getNumber()
      << ' ' << MBB->getName()
      << " (" << (const void*)MBB << ')';
  if (Indexes)
    errs() << " [" << Indexes->getMBBStartIdx(MBB)
        << ';' <<  Indexes->getMBBEndIdx(MBB) << ')';
  errs() << '\n';
}

void MachineVerifier::report(const char *msg, const MachineInstr *MI) {
  assert(MI);
  report(msg, MI->getParent());
  errs() << "- instruction: ";
  if (Indexes && Indexes->hasIndex(MI))
    errs() << Indexes->getInstructionIndex(MI) << '\t';
  MI->print(errs(), TM);
}

void MachineVerifier::report(const char *msg,
                             const MachineOperand *MO, unsigned MONum) {
  assert(MO);
  report(msg, MO->getParent());
  errs() << "- operand " << MONum << ":   ";
  MO->print(errs(), TRI);
  errs() << "\n";
}

void MachineVerifier::report(const char *msg, const MachineFunction *MF,
                             const LiveInterval &LI) {
  report(msg, MF);
  errs() << "- interval:    " << LI << '\n';
}

void MachineVerifier::report(const char *msg, const MachineBasicBlock *MBB,
                             const LiveInterval &LI) {
  report(msg, MBB);
  errs() << "- interval:    " << LI << '\n';
}

void MachineVerifier::report(const char *msg, const MachineBasicBlock *MBB,
                             const LiveRange &LR, unsigned Reg,
                             unsigned LaneMask) {
  report(msg, MBB);
  errs() << "- liverange:   " << LR << '\n';
  errs() << "- register:    " << PrintReg(Reg, TRI) << '\n';
  if (LaneMask != 0)
    errs() << "- lanemask:    " << format("%04X\n", LaneMask);
}

void MachineVerifier::report(const char *msg, const MachineFunction *MF,
                             const LiveRange &LR, unsigned Reg,
                             unsigned LaneMask) {
  report(msg, MF);
  errs() << "- liverange:   " << LR << '\n';
  errs() << "- register:    " << PrintReg(Reg, TRI) << '\n';
  if (LaneMask != 0)
    errs() << "- lanemask:    " << format("%04X\n", LaneMask);
}

void MachineVerifier::markReachable(const MachineBasicBlock *MBB) {
  BBInfo &MInfo = MBBInfoMap[MBB];
  if (!MInfo.reachable) {
    MInfo.reachable = true;
    for (MachineBasicBlock::const_succ_iterator SuI = MBB->succ_begin(),
           SuE = MBB->succ_end(); SuI != SuE; ++SuI)
      markReachable(*SuI);
  }
}

void MachineVerifier::visitMachineFunctionBefore() {
  lastIndex = SlotIndex();
  regsReserved = MRI->getReservedRegs();

  // A sub-register of a reserved register is also reserved
  for (int Reg = regsReserved.find_first(); Reg>=0;
       Reg = regsReserved.find_next(Reg)) {
    for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs) {
      // FIXME: This should probably be:
      // assert(regsReserved.test(*SubRegs) && "Non-reserved sub-register");
      regsReserved.set(*SubRegs);
    }
  }

  markReachable(&MF->front());

  // Build a set of the basic blocks in the function.
  FunctionBlocks.clear();
  for (const auto &MBB : *MF) {
    FunctionBlocks.insert(&MBB);
    BBInfo &MInfo = MBBInfoMap[&MBB];

    MInfo.Preds.insert(MBB.pred_begin(), MBB.pred_end());
    if (MInfo.Preds.size() != MBB.pred_size())
      report("MBB has duplicate entries in its predecessor list.", &MBB);

    MInfo.Succs.insert(MBB.succ_begin(), MBB.succ_end());
    if (MInfo.Succs.size() != MBB.succ_size())
      report("MBB has duplicate entries in its successor list.", &MBB);
  }

  // Check that the register use lists are sane.
  MRI->verifyUseLists();

  verifyStackFrame();
}

// Does iterator point to a and b as the first two elements?
static bool matchPair(MachineBasicBlock::const_succ_iterator i,
                      const MachineBasicBlock *a, const MachineBasicBlock *b) {
  if (*i == a)
    return *++i == b;
  if (*i == b)
    return *++i == a;
  return false;
}

void
MachineVerifier::visitMachineBasicBlockBefore(const MachineBasicBlock *MBB) {
  FirstTerminator = nullptr;

  if (MRI->isSSA()) {
    // If this block has allocatable physical registers live-in, check that
    // it is an entry block or landing pad.
    for (MachineBasicBlock::livein_iterator LI = MBB->livein_begin(),
           LE = MBB->livein_end();
         LI != LE; ++LI) {
      unsigned reg = *LI;
      if (isAllocatable(reg) && !MBB->isLandingPad() &&
          MBB != MBB->getParent()->begin()) {
        report("MBB has allocable live-in, but isn't entry or landing-pad.", MBB);
      }
    }
  }

  // Count the number of landing pad successors.
  SmallPtrSet<MachineBasicBlock*, 4> LandingPadSuccs;
  for (MachineBasicBlock::const_succ_iterator I = MBB->succ_begin(),
       E = MBB->succ_end(); I != E; ++I) {
    if ((*I)->isLandingPad())
      LandingPadSuccs.insert(*I);
    if (!FunctionBlocks.count(*I))
      report("MBB has successor that isn't part of the function.", MBB);
    if (!MBBInfoMap[*I].Preds.count(MBB)) {
      report("Inconsistent CFG", MBB);
      errs() << "MBB is not in the predecessor list of the successor BB#"
          << (*I)->getNumber() << ".\n";
    }
  }

  // Check the predecessor list.
  for (MachineBasicBlock::const_pred_iterator I = MBB->pred_begin(),
       E = MBB->pred_end(); I != E; ++I) {
    if (!FunctionBlocks.count(*I))
      report("MBB has predecessor that isn't part of the function.", MBB);
    if (!MBBInfoMap[*I].Succs.count(MBB)) {
      report("Inconsistent CFG", MBB);
      errs() << "MBB is not in the successor list of the predecessor BB#"
          << (*I)->getNumber() << ".\n";
    }
  }

  const MCAsmInfo *AsmInfo = TM->getMCAsmInfo();
  const BasicBlock *BB = MBB->getBasicBlock();
  if (LandingPadSuccs.size() > 1 &&
      !(AsmInfo &&
        AsmInfo->getExceptionHandlingType() == ExceptionHandling::SjLj &&
        BB && isa<SwitchInst>(BB->getTerminator())))
    report("MBB has more than one landing pad successor", MBB);

  // Call AnalyzeBranch. If it succeeds, there several more conditions to check.
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;
  if (!TII->AnalyzeBranch(*const_cast<MachineBasicBlock *>(MBB),
                          TBB, FBB, Cond)) {
    // Ok, AnalyzeBranch thinks it knows what's going on with this block. Let's
    // check whether its answers match up with reality.
    if (!TBB && !FBB) {
      // Block falls through to its successor.
      MachineFunction::const_iterator MBBI = MBB;
      ++MBBI;
      if (MBBI == MF->end()) {
        // It's possible that the block legitimately ends with a noreturn
        // call or an unreachable, in which case it won't actually fall
        // out the bottom of the function.
      } else if (MBB->succ_size() == LandingPadSuccs.size()) {
        // It's possible that the block legitimately ends with a noreturn
        // call or an unreachable, in which case it won't actuall fall
        // out of the block.
      } else if (MBB->succ_size() != 1+LandingPadSuccs.size()) {
        report("MBB exits via unconditional fall-through but doesn't have "
               "exactly one CFG successor!", MBB);
      } else if (!MBB->isSuccessor(MBBI)) {
        report("MBB exits via unconditional fall-through but its successor "
               "differs from its CFG successor!", MBB);
      }
      if (!MBB->empty() && MBB->back().isBarrier() &&
          !TII->isPredicated(&MBB->back())) {
        report("MBB exits via unconditional fall-through but ends with a "
               "barrier instruction!", MBB);
      }
      if (!Cond.empty()) {
        report("MBB exits via unconditional fall-through but has a condition!",
               MBB);
      }
    } else if (TBB && !FBB && Cond.empty()) {
      // Block unconditionally branches somewhere.
      // If the block has exactly one successor, that happens to be a
      // landingpad, accept it as valid control flow.
      if (MBB->succ_size() != 1+LandingPadSuccs.size() &&
          (MBB->succ_size() != 1 || LandingPadSuccs.size() != 1 ||
           *MBB->succ_begin() != *LandingPadSuccs.begin())) {
        report("MBB exits via unconditional branch but doesn't have "
               "exactly one CFG successor!", MBB);
      } else if (!MBB->isSuccessor(TBB)) {
        report("MBB exits via unconditional branch but the CFG "
               "successor doesn't match the actual successor!", MBB);
      }
      if (MBB->empty()) {
        report("MBB exits via unconditional branch but doesn't contain "
               "any instructions!", MBB);
      } else if (!MBB->back().isBarrier()) {
        report("MBB exits via unconditional branch but doesn't end with a "
               "barrier instruction!", MBB);
      } else if (!MBB->back().isTerminator()) {
        report("MBB exits via unconditional branch but the branch isn't a "
               "terminator instruction!", MBB);
      }
    } else if (TBB && !FBB && !Cond.empty()) {
      // Block conditionally branches somewhere, otherwise falls through.
      MachineFunction::const_iterator MBBI = MBB;
      ++MBBI;
      if (MBBI == MF->end()) {
        report("MBB conditionally falls through out of function!", MBB);
      } else if (MBB->succ_size() == 1) {
        // A conditional branch with only one successor is weird, but allowed.
        if (&*MBBI != TBB)
          report("MBB exits via conditional branch/fall-through but only has "
                 "one CFG successor!", MBB);
        else if (TBB != *MBB->succ_begin())
          report("MBB exits via conditional branch/fall-through but the CFG "
                 "successor don't match the actual successor!", MBB);
      } else if (MBB->succ_size() != 2) {
        report("MBB exits via conditional branch/fall-through but doesn't have "
               "exactly two CFG successors!", MBB);
      } else if (!matchPair(MBB->succ_begin(), TBB, MBBI)) {
        report("MBB exits via conditional branch/fall-through but the CFG "
               "successors don't match the actual successors!", MBB);
      }
      if (MBB->empty()) {
        report("MBB exits via conditional branch/fall-through but doesn't "
               "contain any instructions!", MBB);
      } else if (MBB->back().isBarrier()) {
        report("MBB exits via conditional branch/fall-through but ends with a "
               "barrier instruction!", MBB);
      } else if (!MBB->back().isTerminator()) {
        report("MBB exits via conditional branch/fall-through but the branch "
               "isn't a terminator instruction!", MBB);
      }
    } else if (TBB && FBB) {
      // Block conditionally branches somewhere, otherwise branches
      // somewhere else.
      if (MBB->succ_size() == 1) {
        // A conditional branch with only one successor is weird, but allowed.
        if (FBB != TBB)
          report("MBB exits via conditional branch/branch through but only has "
                 "one CFG successor!", MBB);
        else if (TBB != *MBB->succ_begin())
          report("MBB exits via conditional branch/branch through but the CFG "
                 "successor don't match the actual successor!", MBB);
      } else if (MBB->succ_size() != 2) {
        report("MBB exits via conditional branch/branch but doesn't have "
               "exactly two CFG successors!", MBB);
      } else if (!matchPair(MBB->succ_begin(), TBB, FBB)) {
        report("MBB exits via conditional branch/branch but the CFG "
               "successors don't match the actual successors!", MBB);
      }
      if (MBB->empty()) {
        report("MBB exits via conditional branch/branch but doesn't "
               "contain any instructions!", MBB);
      } else if (!MBB->back().isBarrier()) {
        report("MBB exits via conditional branch/branch but doesn't end with a "
               "barrier instruction!", MBB);
      } else if (!MBB->back().isTerminator()) {
        report("MBB exits via conditional branch/branch but the branch "
               "isn't a terminator instruction!", MBB);
      }
      if (Cond.empty()) {
        report("MBB exits via conditinal branch/branch but there's no "
               "condition!", MBB);
      }
    } else {
      report("AnalyzeBranch returned invalid data!", MBB);
    }
  }

  regsLive.clear();
  for (MachineBasicBlock::livein_iterator I = MBB->livein_begin(),
         E = MBB->livein_end(); I != E; ++I) {
    if (!TargetRegisterInfo::isPhysicalRegister(*I)) {
      report("MBB live-in list contains non-physical register", MBB);
      continue;
    }
    for (MCSubRegIterator SubRegs(*I, TRI, /*IncludeSelf=*/true);
         SubRegs.isValid(); ++SubRegs)
      regsLive.insert(*SubRegs);
  }
  regsLiveInButUnused = regsLive;

  const MachineFrameInfo *MFI = MF->getFrameInfo();
  assert(MFI && "Function has no frame info");
  BitVector PR = MFI->getPristineRegs(*MF);
  for (int I = PR.find_first(); I>0; I = PR.find_next(I)) {
    for (MCSubRegIterator SubRegs(I, TRI, /*IncludeSelf=*/true);
         SubRegs.isValid(); ++SubRegs)
      regsLive.insert(*SubRegs);
  }

  regsKilled.clear();
  regsDefined.clear();

  if (Indexes)
    lastIndex = Indexes->getMBBStartIdx(MBB);
}

// This function gets called for all bundle headers, including normal
// stand-alone unbundled instructions.
void MachineVerifier::visitMachineBundleBefore(const MachineInstr *MI) {
  if (Indexes && Indexes->hasIndex(MI)) {
    SlotIndex idx = Indexes->getInstructionIndex(MI);
    if (!(idx > lastIndex)) {
      report("Instruction index out of order", MI);
      errs() << "Last instruction was at " << lastIndex << '\n';
    }
    lastIndex = idx;
  }

  // Ensure non-terminators don't follow terminators.
  // Ignore predicated terminators formed by if conversion.
  // FIXME: If conversion shouldn't need to violate this rule.
  if (MI->isTerminator() && !TII->isPredicated(MI)) {
    if (!FirstTerminator)
      FirstTerminator = MI;
  } else if (FirstTerminator) {
    report("Non-terminator instruction after the first terminator", MI);
    errs() << "First terminator was:\t" << *FirstTerminator;
  }
}

// The operands on an INLINEASM instruction must follow a template.
// Verify that the flag operands make sense.
void MachineVerifier::verifyInlineAsm(const MachineInstr *MI) {
  // The first two operands on INLINEASM are the asm string and global flags.
  if (MI->getNumOperands() < 2) {
    report("Too few operands on inline asm", MI);
    return;
  }
  if (!MI->getOperand(0).isSymbol())
    report("Asm string must be an external symbol", MI);
  if (!MI->getOperand(1).isImm())
    report("Asm flags must be an immediate", MI);
  // Allowed flags are Extra_HasSideEffects = 1, Extra_IsAlignStack = 2,
  // Extra_AsmDialect = 4, Extra_MayLoad = 8, and Extra_MayStore = 16.
  if (!isUInt<5>(MI->getOperand(1).getImm()))
    report("Unknown asm flags", &MI->getOperand(1), 1);

  static_assert(InlineAsm::MIOp_FirstOperand == 2, "Asm format changed");

  unsigned OpNo = InlineAsm::MIOp_FirstOperand;
  unsigned NumOps;
  for (unsigned e = MI->getNumOperands(); OpNo < e; OpNo += NumOps) {
    const MachineOperand &MO = MI->getOperand(OpNo);
    // There may be implicit ops after the fixed operands.
    if (!MO.isImm())
      break;
    NumOps = 1 + InlineAsm::getNumOperandRegisters(MO.getImm());
  }

  if (OpNo > MI->getNumOperands())
    report("Missing operands in last group", MI);

  // An optional MDNode follows the groups.
  if (OpNo < MI->getNumOperands() && MI->getOperand(OpNo).isMetadata())
    ++OpNo;

  // All trailing operands must be implicit registers.
  for (unsigned e = MI->getNumOperands(); OpNo < e; ++OpNo) {
    const MachineOperand &MO = MI->getOperand(OpNo);
    if (!MO.isReg() || !MO.isImplicit())
      report("Expected implicit register after groups", &MO, OpNo);
  }
}

void MachineVerifier::visitMachineInstrBefore(const MachineInstr *MI) {
  const MCInstrDesc &MCID = MI->getDesc();
  if (MI->getNumOperands() < MCID.getNumOperands()) {
    report("Too few operands", MI);
    errs() << MCID.getNumOperands() << " operands expected, but "
        << MI->getNumOperands() << " given.\n";
  }

  // Check the tied operands.
  if (MI->isInlineAsm())
    verifyInlineAsm(MI);

  // Check the MachineMemOperands for basic consistency.
  for (MachineInstr::mmo_iterator I = MI->memoperands_begin(),
       E = MI->memoperands_end(); I != E; ++I) {
    if ((*I)->isLoad() && !MI->mayLoad())
      report("Missing mayLoad flag", MI);
    if ((*I)->isStore() && !MI->mayStore())
      report("Missing mayStore flag", MI);
  }

  // Debug values must not have a slot index.
  // Other instructions must have one, unless they are inside a bundle.
  if (LiveInts) {
    bool mapped = !LiveInts->isNotInMIMap(MI);
    if (MI->isDebugValue()) {
      if (mapped)
        report("Debug instruction has a slot index", MI);
    } else if (MI->isInsideBundle()) {
      if (mapped)
        report("Instruction inside bundle has a slot index", MI);
    } else {
      if (!mapped)
        report("Missing slot index", MI);
    }
  }

  StringRef ErrorInfo;
  if (!TII->verifyInstruction(MI, ErrorInfo))
    report(ErrorInfo.data(), MI);
}

void
MachineVerifier::visitMachineOperand(const MachineOperand *MO, unsigned MONum) {
  const MachineInstr *MI = MO->getParent();
  const MCInstrDesc &MCID = MI->getDesc();

  // The first MCID.NumDefs operands must be explicit register defines
  if (MONum < MCID.getNumDefs()) {
    const MCOperandInfo &MCOI = MCID.OpInfo[MONum];
    if (!MO->isReg())
      report("Explicit definition must be a register", MO, MONum);
    else if (!MO->isDef() && !MCOI.isOptionalDef())
      report("Explicit definition marked as use", MO, MONum);
    else if (MO->isImplicit())
      report("Explicit definition marked as implicit", MO, MONum);
  } else if (MONum < MCID.getNumOperands()) {
    const MCOperandInfo &MCOI = MCID.OpInfo[MONum];
    // Don't check if it's the last operand in a variadic instruction. See,
    // e.g., LDM_RET in the arm back end.
    if (MO->isReg() &&
        !(MI->isVariadic() && MONum == MCID.getNumOperands()-1)) {
      if (MO->isDef() && !MCOI.isOptionalDef())
        report("Explicit operand marked as def", MO, MONum);
      if (MO->isImplicit())
        report("Explicit operand marked as implicit", MO, MONum);
    }

    int TiedTo = MCID.getOperandConstraint(MONum, MCOI::TIED_TO);
    if (TiedTo != -1) {
      if (!MO->isReg())
        report("Tied use must be a register", MO, MONum);
      else if (!MO->isTied())
        report("Operand should be tied", MO, MONum);
      else if (unsigned(TiedTo) != MI->findTiedOperandIdx(MONum))
        report("Tied def doesn't match MCInstrDesc", MO, MONum);
    } else if (MO->isReg() && MO->isTied())
      report("Explicit operand should not be tied", MO, MONum);
  } else {
    // ARM adds %reg0 operands to indicate predicates. We'll allow that.
    if (MO->isReg() && !MO->isImplicit() && !MI->isVariadic() && MO->getReg())
      report("Extra explicit operand on non-variadic instruction", MO, MONum);
  }

  switch (MO->getType()) {
  case MachineOperand::MO_Register: {
    const unsigned Reg = MO->getReg();
    if (!Reg)
      return;
    if (MRI->tracksLiveness() && !MI->isDebugValue())
      checkLiveness(MO, MONum);

    // Verify the consistency of tied operands.
    if (MO->isTied()) {
      unsigned OtherIdx = MI->findTiedOperandIdx(MONum);
      const MachineOperand &OtherMO = MI->getOperand(OtherIdx);
      if (!OtherMO.isReg())
        report("Must be tied to a register", MO, MONum);
      if (!OtherMO.isTied())
        report("Missing tie flags on tied operand", MO, MONum);
      if (MI->findTiedOperandIdx(OtherIdx) != MONum)
        report("Inconsistent tie links", MO, MONum);
      if (MONum < MCID.getNumDefs()) {
        if (OtherIdx < MCID.getNumOperands()) {
          if (-1 == MCID.getOperandConstraint(OtherIdx, MCOI::TIED_TO))
            report("Explicit def tied to explicit use without tie constraint",
                   MO, MONum);
        } else {
          if (!OtherMO.isImplicit())
            report("Explicit def should be tied to implicit use", MO, MONum);
        }
      }
    }

    // Verify two-address constraints after leaving SSA form.
    unsigned DefIdx;
    if (!MRI->isSSA() && MO->isUse() &&
        MI->isRegTiedToDefOperand(MONum, &DefIdx) &&
        Reg != MI->getOperand(DefIdx).getReg())
      report("Two-address instruction operands must be identical", MO, MONum);

    // Check register classes.
    if (MONum < MCID.getNumOperands() && !MO->isImplicit()) {
      unsigned SubIdx = MO->getSubReg();

      if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
        if (SubIdx) {
          report("Illegal subregister index for physical register", MO, MONum);
          return;
        }
        if (const TargetRegisterClass *DRC =
              TII->getRegClass(MCID, MONum, TRI, *MF)) {
          if (!DRC->contains(Reg)) {
            report("Illegal physical register for instruction", MO, MONum);
            errs() << TRI->getName(Reg) << " is not a "
                << TRI->getRegClassName(DRC) << " register.\n";
          }
        }
      } else {
        // Virtual register.
        const TargetRegisterClass *RC = MRI->getRegClass(Reg);
        if (SubIdx) {
          const TargetRegisterClass *SRC =
            TRI->getSubClassWithSubReg(RC, SubIdx);
          if (!SRC) {
            report("Invalid subregister index for virtual register", MO, MONum);
            errs() << "Register class " << TRI->getRegClassName(RC)
                << " does not support subreg index " << SubIdx << "\n";
            return;
          }
          if (RC != SRC) {
            report("Invalid register class for subregister index", MO, MONum);
            errs() << "Register class " << TRI->getRegClassName(RC)
                << " does not fully support subreg index " << SubIdx << "\n";
            return;
          }
        }
        if (const TargetRegisterClass *DRC =
              TII->getRegClass(MCID, MONum, TRI, *MF)) {
          if (SubIdx) {
            const TargetRegisterClass *SuperRC =
                TRI->getLargestLegalSuperClass(RC, *MF);
            if (!SuperRC) {
              report("No largest legal super class exists.", MO, MONum);
              return;
            }
            DRC = TRI->getMatchingSuperRegClass(SuperRC, DRC, SubIdx);
            if (!DRC) {
              report("No matching super-reg register class.", MO, MONum);
              return;
            }
          }
          if (!RC->hasSuperClassEq(DRC)) {
            report("Illegal virtual register for instruction", MO, MONum);
            errs() << "Expected a " << TRI->getRegClassName(DRC)
                << " register, but got a " << TRI->getRegClassName(RC)
                << " register\n";
          }
        }
      }
    }
    break;
  }

  case MachineOperand::MO_RegisterMask:
    regMasks.push_back(MO->getRegMask());
    break;

  case MachineOperand::MO_MachineBasicBlock:
    if (MI->isPHI() && !MO->getMBB()->isSuccessor(MI->getParent()))
      report("PHI operand is not in the CFG", MO, MONum);
    break;

  case MachineOperand::MO_FrameIndex:
    if (LiveStks && LiveStks->hasInterval(MO->getIndex()) &&
        LiveInts && !LiveInts->isNotInMIMap(MI)) {
      LiveInterval &LI = LiveStks->getInterval(MO->getIndex());
      SlotIndex Idx = LiveInts->getInstructionIndex(MI);
      if (MI->mayLoad() && !LI.liveAt(Idx.getRegSlot(true))) {
        report("Instruction loads from dead spill slot", MO, MONum);
        errs() << "Live stack: " << LI << '\n';
      }
      if (MI->mayStore() && !LI.liveAt(Idx.getRegSlot())) {
        report("Instruction stores to dead spill slot", MO, MONum);
        errs() << "Live stack: " << LI << '\n';
      }
    }
    break;

  default:
    break;
  }
}

void MachineVerifier::checkLiveness(const MachineOperand *MO, unsigned MONum) {
  const MachineInstr *MI = MO->getParent();
  const unsigned Reg = MO->getReg();

  // Both use and def operands can read a register.
  if (MO->readsReg()) {
    regsLiveInButUnused.erase(Reg);

    if (MO->isKill())
      addRegWithSubRegs(regsKilled, Reg);

    // Check that LiveVars knows this kill.
    if (LiveVars && TargetRegisterInfo::isVirtualRegister(Reg) &&
        MO->isKill()) {
      LiveVariables::VarInfo &VI = LiveVars->getVarInfo(Reg);
      if (std::find(VI.Kills.begin(), VI.Kills.end(), MI) == VI.Kills.end())
        report("Kill missing from LiveVariables", MO, MONum);
    }

    // Check LiveInts liveness and kill.
    if (LiveInts && !LiveInts->isNotInMIMap(MI)) {
      SlotIndex UseIdx = LiveInts->getInstructionIndex(MI);
      // Check the cached regunit intervals.
      if (TargetRegisterInfo::isPhysicalRegister(Reg) && !isReserved(Reg)) {
        for (MCRegUnitIterator Units(Reg, TRI); Units.isValid(); ++Units) {
          if (const LiveRange *LR = LiveInts->getCachedRegUnit(*Units)) {
            LiveQueryResult LRQ = LR->Query(UseIdx);
            if (!LRQ.valueIn()) {
              report("No live segment at use", MO, MONum);
              errs() << UseIdx << " is not live in " << PrintRegUnit(*Units, TRI)
                  << ' ' << *LR << '\n';
            }
            if (MO->isKill() && !LRQ.isKill()) {
              report("Live range continues after kill flag", MO, MONum);
              errs() << PrintRegUnit(*Units, TRI) << ' ' << *LR << '\n';
            }
          }
        }
      }

      if (TargetRegisterInfo::isVirtualRegister(Reg)) {
        if (LiveInts->hasInterval(Reg)) {
          // This is a virtual register interval.
          const LiveInterval &LI = LiveInts->getInterval(Reg);
          LiveQueryResult LRQ = LI.Query(UseIdx);
          if (!LRQ.valueIn()) {
            report("No live segment at use", MO, MONum);
            errs() << UseIdx << " is not live in " << LI << '\n';
          }
          // Check for extra kill flags.
          // Note that we allow missing kill flags for now.
          if (MO->isKill() && !LRQ.isKill()) {
            report("Live range continues after kill flag", MO, MONum);
            errs() << "Live range: " << LI << '\n';
          }
        } else {
          report("Virtual register has no live interval", MO, MONum);
        }
      }
    }

    // Use of a dead register.
    if (!regsLive.count(Reg)) {
      if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
        // Reserved registers may be used even when 'dead'.
        bool Bad = !isReserved(Reg);
        // We are fine if just any subregister has a defined value.
        if (Bad) {
          for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid();
               ++SubRegs) {
            if (regsLive.count(*SubRegs)) {
              Bad = false;
              break;
            }
          }
        }
        // If there is an additional implicit-use of a super register we stop
        // here. By definition we are fine if the super register is not
        // (completely) dead, if the complete super register is dead we will
        // get a report for its operand.
        if (Bad) {
          for (const MachineOperand &MOP : MI->uses()) {
            if (!MOP.isReg())
              continue;
            if (!MOP.isImplicit())
              continue;
            for (MCSubRegIterator SubRegs(MOP.getReg(), TRI); SubRegs.isValid();
                 ++SubRegs) {
              if (*SubRegs == Reg) {
                Bad = false;
                break;
              }
            }
          }
        }
        if (Bad)
          report("Using an undefined physical register", MO, MONum);
      } else if (MRI->def_empty(Reg)) {
        report("Reading virtual register without a def", MO, MONum);
      } else {
        BBInfo &MInfo = MBBInfoMap[MI->getParent()];
        // We don't know which virtual registers are live in, so only complain
        // if vreg was killed in this MBB. Otherwise keep track of vregs that
        // must be live in. PHI instructions are handled separately.
        if (MInfo.regsKilled.count(Reg))
          report("Using a killed virtual register", MO, MONum);
        else if (!MI->isPHI())
          MInfo.vregsLiveIn.insert(std::make_pair(Reg, MI));
      }
    }
  }

  if (MO->isDef()) {
    // Register defined.
    // TODO: verify that earlyclobber ops are not used.
    if (MO->isDead())
      addRegWithSubRegs(regsDead, Reg);
    else
      addRegWithSubRegs(regsDefined, Reg);

    // Verify SSA form.
    if (MRI->isSSA() && TargetRegisterInfo::isVirtualRegister(Reg) &&
        std::next(MRI->def_begin(Reg)) != MRI->def_end())
      report("Multiple virtual register defs in SSA form", MO, MONum);

    // Check LiveInts for a live segment, but only for virtual registers.
    if (LiveInts && TargetRegisterInfo::isVirtualRegister(Reg) &&
        !LiveInts->isNotInMIMap(MI)) {
      SlotIndex DefIdx = LiveInts->getInstructionIndex(MI);
      DefIdx = DefIdx.getRegSlot(MO->isEarlyClobber());
      if (LiveInts->hasInterval(Reg)) {
        const LiveInterval &LI = LiveInts->getInterval(Reg);
        if (const VNInfo *VNI = LI.getVNInfoAt(DefIdx)) {
          assert(VNI && "NULL valno is not allowed");
          if (VNI->def != DefIdx) {
            report("Inconsistent valno->def", MO, MONum);
            errs() << "Valno " << VNI->id << " is not defined at "
              << DefIdx << " in " << LI << '\n';
          }
        } else {
          report("No live segment at def", MO, MONum);
          errs() << DefIdx << " is not live in " << LI << '\n';
        }
        // Check that, if the dead def flag is present, LiveInts agree.
        if (MO->isDead()) {
          LiveQueryResult LRQ = LI.Query(DefIdx);
          if (!LRQ.isDeadDef()) {
            report("Live range continues after dead def flag", MO, MONum);
            errs() << "Live range: " << LI << '\n';
          }
        }
      } else {
        report("Virtual register has no Live interval", MO, MONum);
      }
    }
  }
}

void MachineVerifier::visitMachineInstrAfter(const MachineInstr *MI) {
}

// This function gets called after visiting all instructions in a bundle. The
// argument points to the bundle header.
// Normal stand-alone instructions are also considered 'bundles', and this
// function is called for all of them.
void MachineVerifier::visitMachineBundleAfter(const MachineInstr *MI) {
  BBInfo &MInfo = MBBInfoMap[MI->getParent()];
  set_union(MInfo.regsKilled, regsKilled);
  set_subtract(regsLive, regsKilled); regsKilled.clear();
  // Kill any masked registers.
  while (!regMasks.empty()) {
    const uint32_t *Mask = regMasks.pop_back_val();
    for (RegSet::iterator I = regsLive.begin(), E = regsLive.end(); I != E; ++I)
      if (TargetRegisterInfo::isPhysicalRegister(*I) &&
          MachineOperand::clobbersPhysReg(Mask, *I))
        regsDead.push_back(*I);
  }
  set_subtract(regsLive, regsDead);   regsDead.clear();
  set_union(regsLive, regsDefined);   regsDefined.clear();
}

void
MachineVerifier::visitMachineBasicBlockAfter(const MachineBasicBlock *MBB) {
  MBBInfoMap[MBB].regsLiveOut = regsLive;
  regsLive.clear();

  if (Indexes) {
    SlotIndex stop = Indexes->getMBBEndIdx(MBB);
    if (!(stop > lastIndex)) {
      report("Block ends before last instruction index", MBB);
      errs() << "Block ends at " << stop
          << " last instruction was at " << lastIndex << '\n';
    }
    lastIndex = stop;
  }
}

// Calculate the largest possible vregsPassed sets. These are the registers that
// can pass through an MBB live, but may not be live every time. It is assumed
// that all vregsPassed sets are empty before the call.
void MachineVerifier::calcRegsPassed() {
  // First push live-out regs to successors' vregsPassed. Remember the MBBs that
  // have any vregsPassed.
  SmallPtrSet<const MachineBasicBlock*, 8> todo;
  for (const auto &MBB : *MF) {
    BBInfo &MInfo = MBBInfoMap[&MBB];
    if (!MInfo.reachable)
      continue;
    for (MachineBasicBlock::const_succ_iterator SuI = MBB.succ_begin(),
           SuE = MBB.succ_end(); SuI != SuE; ++SuI) {
      BBInfo &SInfo = MBBInfoMap[*SuI];
      if (SInfo.addPassed(MInfo.regsLiveOut))
        todo.insert(*SuI);
    }
  }

  // Iteratively push vregsPassed to successors. This will converge to the same
  // final state regardless of DenseSet iteration order.
  while (!todo.empty()) {
    const MachineBasicBlock *MBB = *todo.begin();
    todo.erase(MBB);
    BBInfo &MInfo = MBBInfoMap[MBB];
    for (MachineBasicBlock::const_succ_iterator SuI = MBB->succ_begin(),
           SuE = MBB->succ_end(); SuI != SuE; ++SuI) {
      if (*SuI == MBB)
        continue;
      BBInfo &SInfo = MBBInfoMap[*SuI];
      if (SInfo.addPassed(MInfo.vregsPassed))
        todo.insert(*SuI);
    }
  }
}

// Calculate the set of virtual registers that must be passed through each basic
// block in order to satisfy the requirements of successor blocks. This is very
// similar to calcRegsPassed, only backwards.
void MachineVerifier::calcRegsRequired() {
  // First push live-in regs to predecessors' vregsRequired.
  SmallPtrSet<const MachineBasicBlock*, 8> todo;
  for (const auto &MBB : *MF) {
    BBInfo &MInfo = MBBInfoMap[&MBB];
    for (MachineBasicBlock::const_pred_iterator PrI = MBB.pred_begin(),
           PrE = MBB.pred_end(); PrI != PrE; ++PrI) {
      BBInfo &PInfo = MBBInfoMap[*PrI];
      if (PInfo.addRequired(MInfo.vregsLiveIn))
        todo.insert(*PrI);
    }
  }

  // Iteratively push vregsRequired to predecessors. This will converge to the
  // same final state regardless of DenseSet iteration order.
  while (!todo.empty()) {
    const MachineBasicBlock *MBB = *todo.begin();
    todo.erase(MBB);
    BBInfo &MInfo = MBBInfoMap[MBB];
    for (MachineBasicBlock::const_pred_iterator PrI = MBB->pred_begin(),
           PrE = MBB->pred_end(); PrI != PrE; ++PrI) {
      if (*PrI == MBB)
        continue;
      BBInfo &SInfo = MBBInfoMap[*PrI];
      if (SInfo.addRequired(MInfo.vregsRequired))
        todo.insert(*PrI);
    }
  }
}

// Check PHI instructions at the beginning of MBB. It is assumed that
// calcRegsPassed has been run so BBInfo::isLiveOut is valid.
void MachineVerifier::checkPHIOps(const MachineBasicBlock *MBB) {
  SmallPtrSet<const MachineBasicBlock*, 8> seen;
  for (const auto &BBI : *MBB) {
    if (!BBI.isPHI())
      break;
    seen.clear();

    for (unsigned i = 1, e = BBI.getNumOperands(); i != e; i += 2) {
      unsigned Reg = BBI.getOperand(i).getReg();
      const MachineBasicBlock *Pre = BBI.getOperand(i + 1).getMBB();
      if (!Pre->isSuccessor(MBB))
        continue;
      seen.insert(Pre);
      BBInfo &PrInfo = MBBInfoMap[Pre];
      if (PrInfo.reachable && !PrInfo.isLiveOut(Reg))
        report("PHI operand is not live-out from predecessor",
               &BBI.getOperand(i), i);
    }

    // Did we see all predecessors?
    for (MachineBasicBlock::const_pred_iterator PrI = MBB->pred_begin(),
           PrE = MBB->pred_end(); PrI != PrE; ++PrI) {
      if (!seen.count(*PrI)) {
        report("Missing PHI operand", &BBI);
        errs() << "BB#" << (*PrI)->getNumber()
            << " is a predecessor according to the CFG.\n";
      }
    }
  }
}

void MachineVerifier::visitMachineFunctionAfter() {
  calcRegsPassed();

  for (const auto &MBB : *MF) {
    BBInfo &MInfo = MBBInfoMap[&MBB];

    // Skip unreachable MBBs.
    if (!MInfo.reachable)
      continue;

    checkPHIOps(&MBB);
  }

  // Now check liveness info if available
  calcRegsRequired();

  // Check for killed virtual registers that should be live out.
  for (const auto &MBB : *MF) {
    BBInfo &MInfo = MBBInfoMap[&MBB];
    for (RegSet::iterator
         I = MInfo.vregsRequired.begin(), E = MInfo.vregsRequired.end(); I != E;
         ++I)
      if (MInfo.regsKilled.count(*I)) {
        report("Virtual register killed in block, but needed live out.", &MBB);
        errs() << "Virtual register " << PrintReg(*I)
            << " is used after the block.\n";
      }
  }

  if (!MF->empty()) {
    BBInfo &MInfo = MBBInfoMap[&MF->front()];
    for (RegSet::iterator
         I = MInfo.vregsRequired.begin(), E = MInfo.vregsRequired.end(); I != E;
         ++I)
      report("Virtual register def doesn't dominate all uses.",
             MRI->getVRegDef(*I));
  }

  if (LiveVars)
    verifyLiveVariables();
  if (LiveInts)
    verifyLiveIntervals();
}

void MachineVerifier::verifyLiveVariables() {
  assert(LiveVars && "Don't call verifyLiveVariables without LiveVars");
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    LiveVariables::VarInfo &VI = LiveVars->getVarInfo(Reg);
    for (const auto &MBB : *MF) {
      BBInfo &MInfo = MBBInfoMap[&MBB];

      // Our vregsRequired should be identical to LiveVariables' AliveBlocks
      if (MInfo.vregsRequired.count(Reg)) {
        if (!VI.AliveBlocks.test(MBB.getNumber())) {
          report("LiveVariables: Block missing from AliveBlocks", &MBB);
          errs() << "Virtual register " << PrintReg(Reg)
              << " must be live through the block.\n";
        }
      } else {
        if (VI.AliveBlocks.test(MBB.getNumber())) {
          report("LiveVariables: Block should not be in AliveBlocks", &MBB);
          errs() << "Virtual register " << PrintReg(Reg)
              << " is not needed live through the block.\n";
        }
      }
    }
  }
}

void MachineVerifier::verifyLiveIntervals() {
  assert(LiveInts && "Don't call verifyLiveIntervals without LiveInts");
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);

    // Spilling and splitting may leave unused registers around. Skip them.
    if (MRI->reg_nodbg_empty(Reg))
      continue;

    if (!LiveInts->hasInterval(Reg)) {
      report("Missing live interval for virtual register", MF);
      errs() << PrintReg(Reg, TRI) << " still has defs or uses\n";
      continue;
    }

    const LiveInterval &LI = LiveInts->getInterval(Reg);
    assert(Reg == LI.reg && "Invalid reg to interval mapping");
    verifyLiveInterval(LI);
  }

  // Verify all the cached regunit intervals.
  for (unsigned i = 0, e = TRI->getNumRegUnits(); i != e; ++i)
    if (const LiveRange *LR = LiveInts->getCachedRegUnit(i))
      verifyLiveRange(*LR, i);
}

void MachineVerifier::verifyLiveRangeValue(const LiveRange &LR,
                                           const VNInfo *VNI, unsigned Reg,
                                           unsigned LaneMask) {
  if (VNI->isUnused())
    return;

  const VNInfo *DefVNI = LR.getVNInfoAt(VNI->def);

  if (!DefVNI) {
    report("Valno not live at def and not marked unused", MF, LR, Reg,
           LaneMask);
    errs() << "Valno #" << VNI->id << '\n';
    return;
  }

  if (DefVNI != VNI) {
    report("Live segment at def has different valno", MF, LR, Reg, LaneMask);
    errs() << "Valno #" << VNI->id << " is defined at " << VNI->def
        << " where valno #" << DefVNI->id << " is live\n";
    return;
  }

  const MachineBasicBlock *MBB = LiveInts->getMBBFromIndex(VNI->def);
  if (!MBB) {
    report("Invalid definition index", MF, LR, Reg, LaneMask);
    errs() << "Valno #" << VNI->id << " is defined at " << VNI->def
        << " in " << LR << '\n';
    return;
  }

  if (VNI->isPHIDef()) {
    if (VNI->def != LiveInts->getMBBStartIdx(MBB)) {
      report("PHIDef value is not defined at MBB start", MBB, LR, Reg,
             LaneMask);
      errs() << "Valno #" << VNI->id << " is defined at " << VNI->def
          << ", not at the beginning of BB#" << MBB->getNumber() << '\n';
    }
    return;
  }

  // Non-PHI def.
  const MachineInstr *MI = LiveInts->getInstructionFromIndex(VNI->def);
  if (!MI) {
    report("No instruction at def index", MBB, LR, Reg, LaneMask);
    errs() << "Valno #" << VNI->id << " is defined at " << VNI->def << '\n';
    return;
  }

  if (Reg != 0) {
    bool hasDef = false;
    bool isEarlyClobber = false;
    for (ConstMIBundleOperands MOI(MI); MOI.isValid(); ++MOI) {
      if (!MOI->isReg() || !MOI->isDef())
        continue;
      if (TargetRegisterInfo::isVirtualRegister(Reg)) {
        if (MOI->getReg() != Reg)
          continue;
      } else {
        if (!TargetRegisterInfo::isPhysicalRegister(MOI->getReg()) ||
            !TRI->hasRegUnit(MOI->getReg(), Reg))
          continue;
      }
      if (LaneMask != 0 &&
          (TRI->getSubRegIndexLaneMask(MOI->getSubReg()) & LaneMask) == 0)
        continue;
      hasDef = true;
      if (MOI->isEarlyClobber())
        isEarlyClobber = true;
    }

    if (!hasDef) {
      report("Defining instruction does not modify register", MI);
      errs() << "Valno #" << VNI->id << " in " << LR << '\n';
    }

    // Early clobber defs begin at USE slots, but other defs must begin at
    // DEF slots.
    if (isEarlyClobber) {
      if (!VNI->def.isEarlyClobber()) {
        report("Early clobber def must be at an early-clobber slot", MBB, LR,
               Reg, LaneMask);
        errs() << "Valno #" << VNI->id << " is defined at " << VNI->def << '\n';
      }
    } else if (!VNI->def.isRegister()) {
      report("Non-PHI, non-early clobber def must be at a register slot",
             MBB, LR, Reg, LaneMask);
      errs() << "Valno #" << VNI->id << " is defined at " << VNI->def << '\n';
    }
  }
}

void MachineVerifier::verifyLiveRangeSegment(const LiveRange &LR,
                                             const LiveRange::const_iterator I,
                                             unsigned Reg, unsigned LaneMask) {
  const LiveRange::Segment &S = *I;
  const VNInfo *VNI = S.valno;
  assert(VNI && "Live segment has no valno");

  if (VNI->id >= LR.getNumValNums() || VNI != LR.getValNumInfo(VNI->id)) {
    report("Foreign valno in live segment", MF, LR, Reg, LaneMask);
    errs() << S << " has a bad valno\n";
  }

  if (VNI->isUnused()) {
    report("Live segment valno is marked unused", MF, LR, Reg, LaneMask);
    errs() << S << '\n';
  }

  const MachineBasicBlock *MBB = LiveInts->getMBBFromIndex(S.start);
  if (!MBB) {
    report("Bad start of live segment, no basic block", MF, LR, Reg, LaneMask);
    errs() << S << '\n';
    return;
  }
  SlotIndex MBBStartIdx = LiveInts->getMBBStartIdx(MBB);
  if (S.start != MBBStartIdx && S.start != VNI->def) {
    report("Live segment must begin at MBB entry or valno def", MBB, LR, Reg,
           LaneMask);
    errs() << S << '\n';
  }

  const MachineBasicBlock *EndMBB =
    LiveInts->getMBBFromIndex(S.end.getPrevSlot());
  if (!EndMBB) {
    report("Bad end of live segment, no basic block", MF, LR, Reg, LaneMask);
    errs() << S << '\n';
    return;
  }

  // No more checks for live-out segments.
  if (S.end == LiveInts->getMBBEndIdx(EndMBB))
    return;

  // RegUnit intervals are allowed dead phis.
  if (!TargetRegisterInfo::isVirtualRegister(Reg) && VNI->isPHIDef() &&
      S.start == VNI->def && S.end == VNI->def.getDeadSlot())
    return;

  // The live segment is ending inside EndMBB
  const MachineInstr *MI =
    LiveInts->getInstructionFromIndex(S.end.getPrevSlot());
  if (!MI) {
    report("Live segment doesn't end at a valid instruction", EndMBB, LR, Reg,
           LaneMask);
    errs() << S << '\n';
    return;
  }

  // The block slot must refer to a basic block boundary.
  if (S.end.isBlock()) {
    report("Live segment ends at B slot of an instruction", EndMBB, LR, Reg,
           LaneMask);
    errs() << S << '\n';
  }

  if (S.end.isDead()) {
    // Segment ends on the dead slot.
    // That means there must be a dead def.
    if (!SlotIndex::isSameInstr(S.start, S.end)) {
      report("Live segment ending at dead slot spans instructions", EndMBB, LR,
             Reg, LaneMask);
      errs() << S << '\n';
    }
  }

  // A live segment can only end at an early-clobber slot if it is being
  // redefined by an early-clobber def.
  if (S.end.isEarlyClobber()) {
    if (I+1 == LR.end() || (I+1)->start != S.end) {
      report("Live segment ending at early clobber slot must be "
             "redefined by an EC def in the same instruction", EndMBB, LR, Reg,
             LaneMask);
      errs() << S << '\n';
    }
  }

  // The following checks only apply to virtual registers. Physreg liveness
  // is too weird to check.
  if (TargetRegisterInfo::isVirtualRegister(Reg)) {
    // A live segment can end with either a redefinition, a kill flag on a
    // use, or a dead flag on a def.
    bool hasRead = false;
    bool hasSubRegDef = false;
    for (ConstMIBundleOperands MOI(MI); MOI.isValid(); ++MOI) {
      if (!MOI->isReg() || MOI->getReg() != Reg)
        continue;
      if (LaneMask != 0 &&
          (LaneMask & TRI->getSubRegIndexLaneMask(MOI->getSubReg())) == 0)
        continue;
      if (MOI->isDef() && MOI->getSubReg() != 0)
        hasSubRegDef = true;
      if (MOI->readsReg())
        hasRead = true;
    }
    if (!S.end.isDead()) {
      if (!hasRead) {
        // When tracking subregister liveness, the main range must start new
        // values on partial register writes, even if there is no read.
        if (!MRI->shouldTrackSubRegLiveness(Reg) || LaneMask != 0 ||
            !hasSubRegDef) {
          report("Instruction ending live segment doesn't read the register",
                 MI);
          errs() << S << " in " << LR << '\n';
        }
      }
    }
  }

  // Now check all the basic blocks in this live segment.
  MachineFunction::const_iterator MFI = MBB;
  // Is this live segment the beginning of a non-PHIDef VN?
  if (S.start == VNI->def && !VNI->isPHIDef()) {
    // Not live-in to any blocks.
    if (MBB == EndMBB)
      return;
    // Skip this block.
    ++MFI;
  }
  for (;;) {
    assert(LiveInts->isLiveInToMBB(LR, MFI));
    // We don't know how to track physregs into a landing pad.
    if (!TargetRegisterInfo::isVirtualRegister(Reg) &&
        MFI->isLandingPad()) {
      if (&*MFI == EndMBB)
        break;
      ++MFI;
      continue;
    }

    // Is VNI a PHI-def in the current block?
    bool IsPHI = VNI->isPHIDef() &&
      VNI->def == LiveInts->getMBBStartIdx(MFI);

    // Check that VNI is live-out of all predecessors.
    for (MachineBasicBlock::const_pred_iterator PI = MFI->pred_begin(),
         PE = MFI->pred_end(); PI != PE; ++PI) {
      SlotIndex PEnd = LiveInts->getMBBEndIdx(*PI);
      const VNInfo *PVNI = LR.getVNInfoBefore(PEnd);

      // All predecessors must have a live-out value.
      if (!PVNI) {
        report("Register not marked live out of predecessor", *PI, LR, Reg,
               LaneMask);
        errs() << "Valno #" << VNI->id << " live into BB#" << MFI->getNumber()
            << '@' << LiveInts->getMBBStartIdx(MFI) << ", not live before "
            << PEnd << '\n';
        continue;
      }

      // Only PHI-defs can take different predecessor values.
      if (!IsPHI && PVNI != VNI) {
        report("Different value live out of predecessor", *PI, LR, Reg,
               LaneMask);
        errs() << "Valno #" << PVNI->id << " live out of BB#"
            << (*PI)->getNumber() << '@' << PEnd
            << "\nValno #" << VNI->id << " live into BB#" << MFI->getNumber()
            << '@' << LiveInts->getMBBStartIdx(MFI) << '\n';
      }
    }
    if (&*MFI == EndMBB)
      break;
    ++MFI;
  }
}

void MachineVerifier::verifyLiveRange(const LiveRange &LR, unsigned Reg,
                                      unsigned LaneMask) {
  for (const VNInfo *VNI : LR.valnos)
    verifyLiveRangeValue(LR, VNI, Reg, LaneMask);

  for (LiveRange::const_iterator I = LR.begin(), E = LR.end(); I != E; ++I)
    verifyLiveRangeSegment(LR, I, Reg, LaneMask);
}

void MachineVerifier::verifyLiveInterval(const LiveInterval &LI) {
  unsigned Reg = LI.reg;
  assert(TargetRegisterInfo::isVirtualRegister(Reg));
  verifyLiveRange(LI, Reg);

  unsigned Mask = 0;
  unsigned MaxMask = MRI->getMaxLaneMaskForVReg(Reg);
  for (const LiveInterval::SubRange &SR : LI.subranges()) {
    if ((Mask & SR.LaneMask) != 0)
      report("Lane masks of sub ranges overlap in live interval", MF, LI);
    if ((SR.LaneMask & ~MaxMask) != 0)
      report("Subrange lanemask is invalid", MF, LI);
    Mask |= SR.LaneMask;
    verifyLiveRange(SR, LI.reg, SR.LaneMask);
    if (!LI.covers(SR))
      report("A Subrange is not covered by the main range", MF, LI);
  }

  // Check the LI only has one connected component.
  ConnectedVNInfoEqClasses ConEQ(*LiveInts);
  unsigned NumComp = ConEQ.Classify(&LI);
  if (NumComp > 1) {
    report("Multiple connected components in live interval", MF, LI);
    for (unsigned comp = 0; comp != NumComp; ++comp) {
      errs() << comp << ": valnos";
      for (LiveInterval::const_vni_iterator I = LI.vni_begin(),
           E = LI.vni_end(); I!=E; ++I)
        if (comp == ConEQ.getEqClass(*I))
          errs() << ' ' << (*I)->id;
      errs() << '\n';
    }
  }
}

namespace {
  // FrameSetup and FrameDestroy can have zero adjustment, so using a single
  // integer, we can't tell whether it is a FrameSetup or FrameDestroy if the
  // value is zero.
  // We use a bool plus an integer to capture the stack state.
  struct StackStateOfBB {
    StackStateOfBB() : EntryValue(0), ExitValue(0), EntryIsSetup(false),
      ExitIsSetup(false) { }
    StackStateOfBB(int EntryVal, int ExitVal, bool EntrySetup, bool ExitSetup) :
      EntryValue(EntryVal), ExitValue(ExitVal), EntryIsSetup(EntrySetup),
      ExitIsSetup(ExitSetup) { }
    // Can be negative, which means we are setting up a frame.
    int EntryValue;
    int ExitValue;
    bool EntryIsSetup;
    bool ExitIsSetup;
  };
}

/// Make sure on every path through the CFG, a FrameSetup <n> is always followed
/// by a FrameDestroy <n>, stack adjustments are identical on all
/// CFG edges to a merge point, and frame is destroyed at end of a return block.
void MachineVerifier::verifyStackFrame() {
  unsigned FrameSetupOpcode   = TII->getCallFrameSetupOpcode();
  unsigned FrameDestroyOpcode = TII->getCallFrameDestroyOpcode();

  SmallVector<StackStateOfBB, 8> SPState;
  SPState.resize(MF->getNumBlockIDs());
  SmallPtrSet<const MachineBasicBlock*, 8> Reachable;

  // Visit the MBBs in DFS order.
  for (df_ext_iterator<const MachineFunction*,
                       SmallPtrSet<const MachineBasicBlock*, 8> >
       DFI = df_ext_begin(MF, Reachable), DFE = df_ext_end(MF, Reachable);
       DFI != DFE; ++DFI) {
    const MachineBasicBlock *MBB = *DFI;

    StackStateOfBB BBState;
    // Check the exit state of the DFS stack predecessor.
    if (DFI.getPathLength() >= 2) {
      const MachineBasicBlock *StackPred = DFI.getPath(DFI.getPathLength() - 2);
      assert(Reachable.count(StackPred) &&
             "DFS stack predecessor is already visited.\n");
      BBState.EntryValue = SPState[StackPred->getNumber()].ExitValue;
      BBState.EntryIsSetup = SPState[StackPred->getNumber()].ExitIsSetup;
      BBState.ExitValue = BBState.EntryValue;
      BBState.ExitIsSetup = BBState.EntryIsSetup;
    }

    // Update stack state by checking contents of MBB.
    for (const auto &I : *MBB) {
      if (I.getOpcode() == FrameSetupOpcode) {
        // The first operand of a FrameOpcode should be i32.
        int Size = I.getOperand(0).getImm();
        assert(Size >= 0 &&
          "Value should be non-negative in FrameSetup and FrameDestroy.\n");

        if (BBState.ExitIsSetup)
          report("FrameSetup is after another FrameSetup", &I);
        BBState.ExitValue -= Size;
        BBState.ExitIsSetup = true;
      }

      if (I.getOpcode() == FrameDestroyOpcode) {
        // The first operand of a FrameOpcode should be i32.
        int Size = I.getOperand(0).getImm();
        assert(Size >= 0 &&
          "Value should be non-negative in FrameSetup and FrameDestroy.\n");

        if (!BBState.ExitIsSetup)
          report("FrameDestroy is not after a FrameSetup", &I);
        int AbsSPAdj = BBState.ExitValue < 0 ? -BBState.ExitValue :
                                               BBState.ExitValue;
        if (BBState.ExitIsSetup && AbsSPAdj != Size) {
          report("FrameDestroy <n> is after FrameSetup <m>", &I);
          errs() << "FrameDestroy <" << Size << "> is after FrameSetup <"
              << AbsSPAdj << ">.\n";
        }
        BBState.ExitValue += Size;
        BBState.ExitIsSetup = false;
      }
    }
    SPState[MBB->getNumber()] = BBState;

    // Make sure the exit state of any predecessor is consistent with the entry
    // state.
    for (MachineBasicBlock::const_pred_iterator I = MBB->pred_begin(),
         E = MBB->pred_end(); I != E; ++I) {
      if (Reachable.count(*I) &&
          (SPState[(*I)->getNumber()].ExitValue != BBState.EntryValue ||
           SPState[(*I)->getNumber()].ExitIsSetup != BBState.EntryIsSetup)) {
        report("The exit stack state of a predecessor is inconsistent.", MBB);
        errs() << "Predecessor BB#" << (*I)->getNumber() << " has exit state ("
            << SPState[(*I)->getNumber()].ExitValue << ", "
            << SPState[(*I)->getNumber()].ExitIsSetup
            << "), while BB#" << MBB->getNumber() << " has entry state ("
            << BBState.EntryValue << ", " << BBState.EntryIsSetup << ").\n";
      }
    }

    // Make sure the entry state of any successor is consistent with the exit
    // state.
    for (MachineBasicBlock::const_succ_iterator I = MBB->succ_begin(),
         E = MBB->succ_end(); I != E; ++I) {
      if (Reachable.count(*I) &&
          (SPState[(*I)->getNumber()].EntryValue != BBState.ExitValue ||
           SPState[(*I)->getNumber()].EntryIsSetup != BBState.ExitIsSetup)) {
        report("The entry stack state of a successor is inconsistent.", MBB);
        errs() << "Successor BB#" << (*I)->getNumber() << " has entry state ("
            << SPState[(*I)->getNumber()].EntryValue << ", "
            << SPState[(*I)->getNumber()].EntryIsSetup
            << "), while BB#" << MBB->getNumber() << " has exit state ("
            << BBState.ExitValue << ", " << BBState.ExitIsSetup << ").\n";
      }
    }

    // Make sure a basic block with return ends with zero stack adjustment.
    if (!MBB->empty() && MBB->back().isReturn()) {
      if (BBState.ExitIsSetup)
        report("A return block ends with a FrameSetup.", MBB);
      if (BBState.ExitValue)
        report("A return block ends with a nonzero stack adjustment.", MBB);
    }
  }
}
