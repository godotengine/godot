//===-- RegisterScavenging.cpp - Machine register scavenging --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the machine register scavenger. It can provide
// information, such as unused registers, at any point in a machine basic block.
// It also provides a mechanism to make registers available by evicting them to
// spill slots.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

#define DEBUG_TYPE "reg-scavenging"

/// setUsed - Set the register units of this register as used.
void RegScavenger::setRegUsed(unsigned Reg) {
  for (MCRegUnitIterator RUI(Reg, TRI); RUI.isValid(); ++RUI)
    RegUnitsAvailable.reset(*RUI);
}

void RegScavenger::initRegState() {
  for (SmallVectorImpl<ScavengedInfo>::iterator I = Scavenged.begin(),
         IE = Scavenged.end(); I != IE; ++I) {
    I->Reg = 0;
    I->Restore = nullptr;
  }

  // All register units start out unused.
  RegUnitsAvailable.set();

  if (!MBB)
    return;

  // Live-in registers are in use.
  for (MachineBasicBlock::livein_iterator I = MBB->livein_begin(),
         E = MBB->livein_end(); I != E; ++I)
    setRegUsed(*I);

  // Pristine CSRs are also unavailable.
  const MachineFunction &MF = *MBB->getParent();
  BitVector PR = MF.getFrameInfo()->getPristineRegs(MF);
  for (int I = PR.find_first(); I>0; I = PR.find_next(I))
    setRegUsed(I);
}

void RegScavenger::enterBasicBlock(MachineBasicBlock *mbb) {
  MachineFunction &MF = *mbb->getParent();
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();

  assert((NumRegUnits == 0 || NumRegUnits == TRI->getNumRegUnits()) &&
         "Target changed?");

  // It is not possible to use the register scavenger after late optimization
  // passes that don't preserve accurate liveness information.
  assert(MRI->tracksLiveness() &&
         "Cannot use register scavenger with inaccurate liveness");

  // Self-initialize.
  if (!MBB) {
    NumRegUnits = TRI->getNumRegUnits();
    RegUnitsAvailable.resize(NumRegUnits);
    KillRegUnits.resize(NumRegUnits);
    DefRegUnits.resize(NumRegUnits);
    TmpRegUnits.resize(NumRegUnits);
  }

  MBB = mbb;
  initRegState();

  Tracking = false;
}

void RegScavenger::addRegUnits(BitVector &BV, unsigned Reg) {
  for (MCRegUnitIterator RUI(Reg, TRI); RUI.isValid(); ++RUI)
    BV.set(*RUI);
}

void RegScavenger::determineKillsAndDefs() {
  assert(Tracking && "Must be tracking to determine kills and defs");

  MachineInstr *MI = MBBI;
  assert(!MI->isDebugValue() && "Debug values have no kills or defs");

  // Find out which registers are early clobbered, killed, defined, and marked
  // def-dead in this instruction.
  KillRegUnits.reset();
  DefRegUnits.reset();
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isRegMask()) {
      
      TmpRegUnits.clear();
      for (unsigned RU = 0, RUEnd = TRI->getNumRegUnits(); RU != RUEnd; ++RU) {
        for (MCRegUnitRootIterator RURI(RU, TRI); RURI.isValid(); ++RURI) {
          if (MO.clobbersPhysReg(*RURI)) {
            TmpRegUnits.set(RU);
            break;
          }
        }
      }
      
      // Apply the mask.
      KillRegUnits |= TmpRegUnits;
    }
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg || TargetRegisterInfo::isVirtualRegister(Reg) || isReserved(Reg))
      continue;

    if (MO.isUse()) {
      // Ignore undef uses.
      if (MO.isUndef())
        continue;
      if (MO.isKill())
        addRegUnits(KillRegUnits, Reg);
    } else {
      assert(MO.isDef());
      if (MO.isDead())
        addRegUnits(KillRegUnits, Reg);
      else
        addRegUnits(DefRegUnits, Reg);
    }
  }
}

void RegScavenger::unprocess() {
  assert(Tracking && "Cannot unprocess because we're not tracking");

  MachineInstr *MI = MBBI;
  if (!MI->isDebugValue()) {
    determineKillsAndDefs();

    // Commit the changes.
    setUsed(KillRegUnits);
    setUnused(DefRegUnits);
  }

  if (MBBI == MBB->begin()) {
    MBBI = MachineBasicBlock::iterator(nullptr);
    Tracking = false;
  } else
    --MBBI;
}

void RegScavenger::forward() {
  // Move ptr forward.
  if (!Tracking) {
    MBBI = MBB->begin();
    Tracking = true;
  } else {
    assert(MBBI != MBB->end() && "Already past the end of the basic block!");
    MBBI = std::next(MBBI);
  }
  assert(MBBI != MBB->end() && "Already at the end of the basic block!");

  MachineInstr *MI = MBBI;

  for (SmallVectorImpl<ScavengedInfo>::iterator I = Scavenged.begin(),
         IE = Scavenged.end(); I != IE; ++I) {
    if (I->Restore != MI)
      continue;

    I->Reg = 0;
    I->Restore = nullptr;
  }

  if (MI->isDebugValue())
    return;

  determineKillsAndDefs();

  // Verify uses and defs.
#ifndef NDEBUG
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg || TargetRegisterInfo::isVirtualRegister(Reg) || isReserved(Reg))
      continue;
    if (MO.isUse()) {
      if (MO.isUndef())
        continue;
      if (!isRegUsed(Reg)) {
        // Check if it's partial live: e.g.
        // D0 = insert_subreg D0<undef>, S0
        // ... D0
        // The problem is the insert_subreg could be eliminated. The use of
        // D0 is using a partially undef value. This is not *incorrect* since
        // S1 is can be freely clobbered.
        // Ideally we would like a way to model this, but leaving the
        // insert_subreg around causes both correctness and performance issues.
        bool SubUsed = false;
        for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs)
          if (isRegUsed(*SubRegs)) {
            SubUsed = true;
            break;
          }
        bool SuperUsed = false;
        for (MCSuperRegIterator SR(Reg, TRI); SR.isValid(); ++SR) {
          if (isRegUsed(*SR)) {
            SuperUsed = true;
            break;
          }
        }
        if (!SubUsed && !SuperUsed) {
          MBB->getParent()->verify(nullptr, "In Register Scavenger");
          llvm_unreachable("Using an undefined register!");
        }
        (void)SubUsed;
        (void)SuperUsed;
      }
    } else {
      assert(MO.isDef());
#if 0
      // FIXME: Enable this once we've figured out how to correctly transfer
      // implicit kills during codegen passes like the coalescer.
      assert((KillRegs.test(Reg) || isUnused(Reg) ||
              isLiveInButUnusedBefore(Reg, MI, MBB, TRI, MRI)) &&
             "Re-defining a live register!");
#endif
    }
  }
#endif // NDEBUG

  // Commit the changes.
  setUnused(KillRegUnits);
  setUsed(DefRegUnits);
}

bool RegScavenger::isRegUsed(unsigned Reg, bool includeReserved) const {
  if (includeReserved && isReserved(Reg))
    return true;
  for (MCRegUnitIterator RUI(Reg, TRI); RUI.isValid(); ++RUI)
    if (!RegUnitsAvailable.test(*RUI))
      return true;
  return false;
}

unsigned RegScavenger::FindUnusedReg(const TargetRegisterClass *RC) const {
  for (TargetRegisterClass::iterator I = RC->begin(), E = RC->end();
       I != E; ++I)
    if (!isRegUsed(*I)) {
      DEBUG(dbgs() << "Scavenger found unused reg: " << TRI->getName(*I) <<
            "\n");
      return *I;
    }
  return 0;
}

/// getRegsAvailable - Return all available registers in the register class
/// in Mask.
BitVector RegScavenger::getRegsAvailable(const TargetRegisterClass *RC) {
  BitVector Mask(TRI->getNumRegs());
  for (TargetRegisterClass::iterator I = RC->begin(), E = RC->end();
       I != E; ++I)
    if (!isRegUsed(*I))
      Mask.set(*I);
  return Mask;
}

/// findSurvivorReg - Return the candidate register that is unused for the
/// longest after StartMII. UseMI is set to the instruction where the search
/// stopped.
///
/// No more than InstrLimit instructions are inspected.
///
unsigned RegScavenger::findSurvivorReg(MachineBasicBlock::iterator StartMI,
                                       BitVector &Candidates,
                                       unsigned InstrLimit,
                                       MachineBasicBlock::iterator &UseMI) {
  int Survivor = Candidates.find_first();
  assert(Survivor > 0 && "No candidates for scavenging");

  MachineBasicBlock::iterator ME = MBB->getFirstTerminator();
  assert(StartMI != ME && "MI already at terminator");
  MachineBasicBlock::iterator RestorePointMI = StartMI;
  MachineBasicBlock::iterator MI = StartMI;

  bool inVirtLiveRange = false;
  for (++MI; InstrLimit > 0 && MI != ME; ++MI, --InstrLimit) {
    if (MI->isDebugValue()) {
      ++InstrLimit; // Don't count debug instructions
      continue;
    }
    bool isVirtKillInsn = false;
    bool isVirtDefInsn = false;
    // Remove any candidates touched by instruction.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = MI->getOperand(i);
      if (MO.isRegMask())
        Candidates.clearBitsNotInMask(MO.getRegMask());
      if (!MO.isReg() || MO.isUndef() || !MO.getReg())
        continue;
      if (TargetRegisterInfo::isVirtualRegister(MO.getReg())) {
        if (MO.isDef())
          isVirtDefInsn = true;
        else if (MO.isKill())
          isVirtKillInsn = true;
        continue;
      }
      for (MCRegAliasIterator AI(MO.getReg(), TRI, true); AI.isValid(); ++AI)
        Candidates.reset(*AI);
    }
    // If we're not in a virtual reg's live range, this is a valid
    // restore point.
    if (!inVirtLiveRange) RestorePointMI = MI;

    // Update whether we're in the live range of a virtual register
    if (isVirtKillInsn) inVirtLiveRange = false;
    if (isVirtDefInsn) inVirtLiveRange = true;

    // Was our survivor untouched by this instruction?
    if (Candidates.test(Survivor))
      continue;

    // All candidates gone?
    if (Candidates.none())
      break;

    Survivor = Candidates.find_first();
  }
  // If we ran off the end, that's where we want to restore.
  if (MI == ME) RestorePointMI = ME;
  assert (RestorePointMI != StartMI &&
          "No available scavenger restore location!");

  // We ran out of candidates, so stop the search.
  UseMI = RestorePointMI;
  return Survivor;
}

static unsigned getFrameIndexOperandNum(MachineInstr *MI) {
  unsigned i = 0;
  while (!MI->getOperand(i).isFI()) {
    ++i;
    assert(i < MI->getNumOperands() &&
           "Instr doesn't have FrameIndex operand!");
  }
  return i;
}

unsigned RegScavenger::scavengeRegister(const TargetRegisterClass *RC,
                                        MachineBasicBlock::iterator I,
                                        int SPAdj) {
  // Consider all allocatable registers in the register class initially
  BitVector Candidates =
    TRI->getAllocatableSet(*I->getParent()->getParent(), RC);

  // Exclude all the registers being used by the instruction.
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = I->getOperand(i);
    if (MO.isReg() && MO.getReg() != 0 && !(MO.isUse() && MO.isUndef()) &&
        !TargetRegisterInfo::isVirtualRegister(MO.getReg()))
      Candidates.reset(MO.getReg());
  }

  // Try to find a register that's unused if there is one, as then we won't
  // have to spill.
  BitVector Available = getRegsAvailable(RC);
  Available &= Candidates;
  if (Available.any())
    Candidates = Available;

  // Find the register whose use is furthest away.
  MachineBasicBlock::iterator UseMI;
  unsigned SReg = findSurvivorReg(I, Candidates, 25, UseMI);

  // If we found an unused register there is no reason to spill it.
  if (!isRegUsed(SReg)) {
    DEBUG(dbgs() << "Scavenged register: " << TRI->getName(SReg) << "\n");
    return SReg;
  }

  // Find an available scavenging slot.
  unsigned SI;
  for (SI = 0; SI < Scavenged.size(); ++SI)
    if (Scavenged[SI].Reg == 0)
      break;

  if (SI == Scavenged.size()) {
    // We need to scavenge a register but have no spill slot, the target
    // must know how to do it (if not, we'll assert below).
    Scavenged.push_back(ScavengedInfo());
  }

  // Avoid infinite regress
  Scavenged[SI].Reg = SReg;

  // If the target knows how to save/restore the register, let it do so;
  // otherwise, use the emergency stack spill slot.
  if (!TRI->saveScavengerRegister(*MBB, I, UseMI, RC, SReg)) {
    // Spill the scavenged register before I.
    assert(Scavenged[SI].FrameIndex >= 0 &&
           "Cannot scavenge register without an emergency spill slot!");
    TII->storeRegToStackSlot(*MBB, I, SReg, true, Scavenged[SI].FrameIndex,
                             RC, TRI);
    MachineBasicBlock::iterator II = std::prev(I);

    unsigned FIOperandNum = getFrameIndexOperandNum(II);
    TRI->eliminateFrameIndex(II, SPAdj, FIOperandNum, this);

    // Restore the scavenged register before its use (or first terminator).
    TII->loadRegFromStackSlot(*MBB, UseMI, SReg, Scavenged[SI].FrameIndex,
                              RC, TRI);
    II = std::prev(UseMI);

    FIOperandNum = getFrameIndexOperandNum(II);
    TRI->eliminateFrameIndex(II, SPAdj, FIOperandNum, this);
  }

  Scavenged[SI].Restore = std::prev(UseMI);

  // Doing this here leads to infinite regress.
  // Scavenged[SI].Reg = SReg;

  DEBUG(dbgs() << "Scavenged register (with spill): " << TRI->getName(SReg) <<
        "\n");

  return SReg;
}
