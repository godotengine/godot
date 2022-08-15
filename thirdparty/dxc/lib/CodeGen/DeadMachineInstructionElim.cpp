//===- DeadMachineInstructionElim.cpp - Remove dead machine instructions --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is an extremely simple MachineInstr-level dead-code-elimination pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

#define DEBUG_TYPE "codegen-dce"

STATISTIC(NumDeletes,          "Number of dead instructions deleted");

namespace {
  class DeadMachineInstructionElim : public MachineFunctionPass {
    bool runOnMachineFunction(MachineFunction &MF) override;

    const TargetRegisterInfo *TRI;
    const MachineRegisterInfo *MRI;
    const TargetInstrInfo *TII;
    BitVector LivePhysRegs;

  public:
    static char ID; // Pass identification, replacement for typeid
    DeadMachineInstructionElim() : MachineFunctionPass(ID) {
     initializeDeadMachineInstructionElimPass(*PassRegistry::getPassRegistry());
    }

  private:
    bool isDead(const MachineInstr *MI) const;
  };
}
char DeadMachineInstructionElim::ID = 0;
char &llvm::DeadMachineInstructionElimID = DeadMachineInstructionElim::ID;

INITIALIZE_PASS(DeadMachineInstructionElim, "dead-mi-elimination",
                "Remove dead machine instructions", false, false)

bool DeadMachineInstructionElim::isDead(const MachineInstr *MI) const {
  // Technically speaking inline asm without side effects and no defs can still
  // be deleted. But there is so much bad inline asm code out there, we should
  // let them be.
  if (MI->isInlineAsm())
    return false;

  // Don't delete frame allocation labels.
  if (MI->getOpcode() == TargetOpcode::LOCAL_ESCAPE)
    return false;

  // Don't delete instructions with side effects.
  bool SawStore = false;
  if (!MI->isSafeToMove(nullptr, SawStore) && !MI->isPHI())
    return false;

  // Examine each operand.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isDef()) {
      unsigned Reg = MO.getReg();
      if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
        // Don't delete live physreg defs, or any reserved register defs.
        if (LivePhysRegs.test(Reg) || MRI->isReserved(Reg))
          return false;
      } else {
        if (!MRI->use_nodbg_empty(Reg))
          // This def has a non-debug use. Don't delete the instruction!
          return false;
      }
    }
  }

  // If there are no defs with uses, the instruction is dead.
  return true;
}

bool DeadMachineInstructionElim::runOnMachineFunction(MachineFunction &MF) {
  if (skipOptnoneFunction(*MF.getFunction()))
    return false;

  bool AnyChanges = false;
  MRI = &MF.getRegInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  TII = MF.getSubtarget().getInstrInfo();

  // Loop over all instructions in all blocks, from bottom to top, so that it's
  // more likely that chains of dependent but ultimately dead instructions will
  // be cleaned up.
  for (MachineFunction::reverse_iterator I = MF.rbegin(), E = MF.rend();
       I != E; ++I) {
    MachineBasicBlock *MBB = &*I;

    // Start out assuming that reserved registers are live out of this block.
    LivePhysRegs = MRI->getReservedRegs();

    // Add live-ins from sucessors to LivePhysRegs. Normally, physregs are not
    // live across blocks, but some targets (x86) can have flags live out of a
    // block.
    for (MachineBasicBlock::succ_iterator S = MBB->succ_begin(),
           E = MBB->succ_end(); S != E; S++)
      for (MachineBasicBlock::livein_iterator LI = (*S)->livein_begin();
           LI != (*S)->livein_end(); LI++)
        LivePhysRegs.set(*LI);

    // Now scan the instructions and delete dead ones, tracking physreg
    // liveness as we go.
    for (MachineBasicBlock::reverse_iterator MII = MBB->rbegin(),
         MIE = MBB->rend(); MII != MIE; ) {
      MachineInstr *MI = &*MII;

      // If the instruction is dead, delete it!
      if (isDead(MI)) {
        DEBUG(dbgs() << "DeadMachineInstructionElim: DELETING: " << *MI);
        // It is possible that some DBG_VALUE instructions refer to this
        // instruction.  They get marked as undef and will be deleted
        // in the live debug variable analysis.
        MI->eraseFromParentAndMarkDBGValuesForRemoval();
        AnyChanges = true;
        ++NumDeletes;
        MIE = MBB->rend();
        // MII is now pointing to the next instruction to process,
        // so don't increment it.
        continue;
      }

      // Record the physreg defs.
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        const MachineOperand &MO = MI->getOperand(i);
        if (MO.isReg() && MO.isDef()) {
          unsigned Reg = MO.getReg();
          if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
            // Check the subreg set, not the alias set, because a def
            // of a super-register may still be partially live after
            // this def.
            for (MCSubRegIterator SR(Reg, TRI,/*IncludeSelf=*/true);
                 SR.isValid(); ++SR)
              LivePhysRegs.reset(*SR);
          }
        } else if (MO.isRegMask()) {
          // Register mask of preserved registers. All clobbers are dead.
          LivePhysRegs.clearBitsNotInMask(MO.getRegMask());
        }
      }
      // Record the physreg uses, after the defs, in case a physreg is
      // both defined and used in the same instruction.
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        const MachineOperand &MO = MI->getOperand(i);
        if (MO.isReg() && MO.isUse()) {
          unsigned Reg = MO.getReg();
          if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
            for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI)
              LivePhysRegs.set(*AI);
          }
        }
      }

      // We didn't delete the current instruction, so increment MII to
      // the next one.
      ++MII;
    }
  }

  LivePhysRegs.clear();
  return AnyChanges;
}
