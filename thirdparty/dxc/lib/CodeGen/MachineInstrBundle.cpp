//===-- lib/CodeGen/MachineInstrBundle.cpp --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

namespace {
  class UnpackMachineBundles : public MachineFunctionPass {
  public:
    static char ID; // Pass identification
    UnpackMachineBundles(std::function<bool(const Function &)> Ftor = nullptr)
        : MachineFunctionPass(ID), PredicateFtor(Ftor) {
      initializeUnpackMachineBundlesPass(*PassRegistry::getPassRegistry());
    }

    bool runOnMachineFunction(MachineFunction &MF) override;

  private:
    std::function<bool(const Function &)> PredicateFtor;
  };
} // end anonymous namespace

char UnpackMachineBundles::ID = 0;
char &llvm::UnpackMachineBundlesID = UnpackMachineBundles::ID;
INITIALIZE_PASS(UnpackMachineBundles, "unpack-mi-bundles",
                "Unpack machine instruction bundles", false, false)

bool UnpackMachineBundles::runOnMachineFunction(MachineFunction &MF) {
  if (PredicateFtor && !PredicateFtor(*MF.getFunction()))
    return false;

  bool Changed = false;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = &*I;

    for (MachineBasicBlock::instr_iterator MII = MBB->instr_begin(),
           MIE = MBB->instr_end(); MII != MIE; ) {
      MachineInstr *MI = &*MII;

      // Remove BUNDLE instruction and the InsideBundle flags from bundled
      // instructions.
      if (MI->isBundle()) {
        while (++MII != MIE && MII->isBundledWithPred()) {
          MII->unbundleFromPred();
          for (unsigned i = 0, e = MII->getNumOperands(); i != e; ++i) {
            MachineOperand &MO = MII->getOperand(i);
            if (MO.isReg() && MO.isInternalRead())
              MO.setIsInternalRead(false);
          }
        }
        MI->eraseFromParent();

        Changed = true;
        continue;
      }

      ++MII;
    }
  }

  return Changed;
}

FunctionPass *
llvm::createUnpackMachineBundles(std::function<bool(const Function &)> Ftor) {
  return new UnpackMachineBundles(Ftor);
}

namespace {
  class FinalizeMachineBundles : public MachineFunctionPass {
  public:
    static char ID; // Pass identification
    FinalizeMachineBundles() : MachineFunctionPass(ID) {
      initializeFinalizeMachineBundlesPass(*PassRegistry::getPassRegistry());
    }

    bool runOnMachineFunction(MachineFunction &MF) override;
  };
} // end anonymous namespace

char FinalizeMachineBundles::ID = 0;
char &llvm::FinalizeMachineBundlesID = FinalizeMachineBundles::ID;
INITIALIZE_PASS(FinalizeMachineBundles, "finalize-mi-bundles",
                "Finalize machine instruction bundles", false, false)

bool FinalizeMachineBundles::runOnMachineFunction(MachineFunction &MF) {
  return llvm::finalizeBundles(MF);
}


/// finalizeBundle - Finalize a machine instruction bundle which includes
/// a sequence of instructions starting from FirstMI to LastMI (exclusive).
/// This routine adds a BUNDLE instruction to represent the bundle, it adds
/// IsInternalRead markers to MachineOperands which are defined inside the
/// bundle, and it copies externally visible defs and uses to the BUNDLE
/// instruction.
void llvm::finalizeBundle(MachineBasicBlock &MBB,
                          MachineBasicBlock::instr_iterator FirstMI,
                          MachineBasicBlock::instr_iterator LastMI) {
  assert(FirstMI != LastMI && "Empty bundle?");
  MIBundleBuilder Bundle(MBB, FirstMI, LastMI);

  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  MachineInstrBuilder MIB =
      BuildMI(MF, FirstMI->getDebugLoc(), TII->get(TargetOpcode::BUNDLE));
  Bundle.prepend(MIB);

  SmallVector<unsigned, 32> LocalDefs;
  SmallSet<unsigned, 32> LocalDefSet;
  SmallSet<unsigned, 8> DeadDefSet;
  SmallSet<unsigned, 16> KilledDefSet;
  SmallVector<unsigned, 8> ExternUses;
  SmallSet<unsigned, 8> ExternUseSet;
  SmallSet<unsigned, 8> KilledUseSet;
  SmallSet<unsigned, 8> UndefUseSet;
  SmallVector<MachineOperand*, 4> Defs;
  for (; FirstMI != LastMI; ++FirstMI) {
    for (unsigned i = 0, e = FirstMI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = FirstMI->getOperand(i);
      if (!MO.isReg())
        continue;
      if (MO.isDef()) {
        Defs.push_back(&MO);
        continue;
      }

      unsigned Reg = MO.getReg();
      if (!Reg)
        continue;
      assert(TargetRegisterInfo::isPhysicalRegister(Reg));
      if (LocalDefSet.count(Reg)) {
        MO.setIsInternalRead();
        if (MO.isKill())
          // Internal def is now killed.
          KilledDefSet.insert(Reg);
      } else {
        if (ExternUseSet.insert(Reg).second) {
          ExternUses.push_back(Reg);
          if (MO.isUndef())
            UndefUseSet.insert(Reg);
        }
        if (MO.isKill())
          // External def is now killed.
          KilledUseSet.insert(Reg);
      }
    }

    for (unsigned i = 0, e = Defs.size(); i != e; ++i) {
      MachineOperand &MO = *Defs[i];
      unsigned Reg = MO.getReg();
      if (!Reg)
        continue;

      if (LocalDefSet.insert(Reg).second) {
        LocalDefs.push_back(Reg);
        if (MO.isDead()) {
          DeadDefSet.insert(Reg);
        }
      } else {
        // Re-defined inside the bundle, it's no longer killed.
        KilledDefSet.erase(Reg);
        if (!MO.isDead())
          // Previously defined but dead.
          DeadDefSet.erase(Reg);
      }

      if (!MO.isDead()) {
        for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs) {
          unsigned SubReg = *SubRegs;
          if (LocalDefSet.insert(SubReg).second)
            LocalDefs.push_back(SubReg);
        }
      }
    }

    Defs.clear();
  }

  SmallSet<unsigned, 32> Added;
  for (unsigned i = 0, e = LocalDefs.size(); i != e; ++i) {
    unsigned Reg = LocalDefs[i];
    if (Added.insert(Reg).second) {
      // If it's not live beyond end of the bundle, mark it dead.
      bool isDead = DeadDefSet.count(Reg) || KilledDefSet.count(Reg);
      MIB.addReg(Reg, getDefRegState(true) | getDeadRegState(isDead) |
                 getImplRegState(true));
    }
  }

  for (unsigned i = 0, e = ExternUses.size(); i != e; ++i) {
    unsigned Reg = ExternUses[i];
    bool isKill = KilledUseSet.count(Reg);
    bool isUndef = UndefUseSet.count(Reg);
    MIB.addReg(Reg, getKillRegState(isKill) | getUndefRegState(isUndef) |
               getImplRegState(true));
  }
}

/// finalizeBundle - Same functionality as the previous finalizeBundle except
/// the last instruction in the bundle is not provided as an input. This is
/// used in cases where bundles are pre-determined by marking instructions
/// with 'InsideBundle' marker. It returns the MBB instruction iterator that
/// points to the end of the bundle.
MachineBasicBlock::instr_iterator
llvm::finalizeBundle(MachineBasicBlock &MBB,
                     MachineBasicBlock::instr_iterator FirstMI) {
  MachineBasicBlock::instr_iterator E = MBB.instr_end();
  MachineBasicBlock::instr_iterator LastMI = std::next(FirstMI);
  while (LastMI != E && LastMI->isInsideBundle())
    ++LastMI;
  finalizeBundle(MBB, FirstMI, LastMI);
  return LastMI;
}

/// finalizeBundles - Finalize instruction bundles in the specified
/// MachineFunction. Return true if any bundles are finalized.
bool llvm::finalizeBundles(MachineFunction &MF) {
  bool Changed = false;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock &MBB = *I;
    MachineBasicBlock::instr_iterator MII = MBB.instr_begin();
    MachineBasicBlock::instr_iterator MIE = MBB.instr_end();
    if (MII == MIE)
      continue;
    assert(!MII->isInsideBundle() &&
           "First instr cannot be inside bundle before finalization!");

    for (++MII; MII != MIE; ) {
      if (!MII->isInsideBundle())
        ++MII;
      else {
        MII = finalizeBundle(MBB, std::prev(MII));
        Changed = true;
      }
    }
  }

  return Changed;
}

//===----------------------------------------------------------------------===//
// MachineOperand iterator
//===----------------------------------------------------------------------===//

MachineOperandIteratorBase::VirtRegInfo
MachineOperandIteratorBase::analyzeVirtReg(unsigned Reg,
                    SmallVectorImpl<std::pair<MachineInstr*, unsigned> > *Ops) {
  VirtRegInfo RI = { false, false, false };
  for(; isValid(); ++*this) {
    MachineOperand &MO = deref();
    if (!MO.isReg() || MO.getReg() != Reg)
      continue;

    // Remember each (MI, OpNo) that refers to Reg.
    if (Ops)
      Ops->push_back(std::make_pair(MO.getParent(), getOperandNo()));

    // Both defs and uses can read virtual registers.
    if (MO.readsReg()) {
      RI.Reads = true;
      if (MO.isDef())
        RI.Tied = true;
    }

    // Only defs can write.
    if (MO.isDef())
      RI.Writes = true;
    else if (!RI.Tied && MO.getParent()->isRegTiedToDefOperand(getOperandNo()))
      RI.Tied = true;
  }
  return RI;
}

MachineOperandIteratorBase::PhysRegInfo
MachineOperandIteratorBase::analyzePhysReg(unsigned Reg,
                                           const TargetRegisterInfo *TRI) {
  bool AllDefsDead = true;
  PhysRegInfo PRI = {false, false, false, false, false, false};

  assert(TargetRegisterInfo::isPhysicalRegister(Reg) &&
         "analyzePhysReg not given a physical register!");
  for (; isValid(); ++*this) {
    MachineOperand &MO = deref();

    if (MO.isRegMask() && MO.clobbersPhysReg(Reg))
      PRI.Clobbers = true;    // Regmask clobbers Reg.

    if (!MO.isReg())
      continue;

    unsigned MOReg = MO.getReg();
    if (!MOReg || !TargetRegisterInfo::isPhysicalRegister(MOReg))
      continue;

    bool IsRegOrSuperReg = MOReg == Reg || TRI->isSubRegister(MOReg, Reg);
    bool IsRegOrOverlapping = MOReg == Reg || TRI->regsOverlap(MOReg, Reg);

    if (IsRegOrSuperReg && MO.readsReg()) {
      // Reg or a super-reg is read, and perhaps killed also.
      PRI.Reads = true;
      PRI.Kills = MO.isKill();
    }

    if (IsRegOrOverlapping && MO.readsReg()) {
      PRI.ReadsOverlap = true;// Reg or an overlapping register is read.
    }

    if (!MO.isDef())
      continue;

    if (IsRegOrSuperReg) {
      PRI.Defines = true;     // Reg or a super-register is defined.
      if (!MO.isDead())
        AllDefsDead = false;
    }
    if (IsRegOrOverlapping)
      PRI.Clobbers = true;    // Reg or an overlapping reg is defined.
  }

  if (AllDefsDead && PRI.Defines)
    PRI.DefinesDead = true;   // Reg or super-register was defined and was dead.

  return PRI;
}
