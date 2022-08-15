//===-- ExpandPostRAPseudos.cpp - Pseudo instruction expansion pass -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pass that expands COPY and SUBREG_TO_REG pseudo
// instructions after register allocation.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

#define DEBUG_TYPE "postrapseudos"

namespace {
struct ExpandPostRA : public MachineFunctionPass {
private:
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;

public:
  static char ID; // Pass identification, replacement for typeid
  ExpandPostRA() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addPreservedID(MachineLoopInfoID);
    AU.addPreservedID(MachineDominatorsID);
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  /// runOnMachineFunction - pass entry point
  bool runOnMachineFunction(MachineFunction&) override;

private:
  bool LowerSubregToReg(MachineInstr *MI);
  bool LowerCopy(MachineInstr *MI);

  void TransferImplicitDefs(MachineInstr *MI);
};
} // end anonymous namespace

char ExpandPostRA::ID = 0;
char &llvm::ExpandPostRAPseudosID = ExpandPostRA::ID;

INITIALIZE_PASS(ExpandPostRA, "postrapseudos",
                "Post-RA pseudo instruction expansion pass", false, false)

/// TransferImplicitDefs - MI is a pseudo-instruction, and the lowered
/// replacement instructions immediately precede it.  Copy any implicit-def
/// operands from MI to the replacement instruction.
void
ExpandPostRA::TransferImplicitDefs(MachineInstr *MI) {
  MachineBasicBlock::iterator CopyMI = MI;
  --CopyMI;

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isImplicit() || MO.isUse())
      continue;
    CopyMI->addOperand(MachineOperand::CreateReg(MO.getReg(), true, true));
  }
}

bool ExpandPostRA::LowerSubregToReg(MachineInstr *MI) {
  MachineBasicBlock *MBB = MI->getParent();
  assert((MI->getOperand(0).isReg() && MI->getOperand(0).isDef()) &&
         MI->getOperand(1).isImm() &&
         (MI->getOperand(2).isReg() && MI->getOperand(2).isUse()) &&
          MI->getOperand(3).isImm() && "Invalid subreg_to_reg");

  unsigned DstReg  = MI->getOperand(0).getReg();
  unsigned InsReg  = MI->getOperand(2).getReg();
  assert(!MI->getOperand(2).getSubReg() && "SubIdx on physreg?");
  unsigned SubIdx  = MI->getOperand(3).getImm();

  assert(SubIdx != 0 && "Invalid index for insert_subreg");
  unsigned DstSubReg = TRI->getSubReg(DstReg, SubIdx);

  assert(TargetRegisterInfo::isPhysicalRegister(DstReg) &&
         "Insert destination must be in a physical register");
  assert(TargetRegisterInfo::isPhysicalRegister(InsReg) &&
         "Inserted value must be in a physical register");

  DEBUG(dbgs() << "subreg: CONVERTING: " << *MI);

  if (MI->allDefsAreDead()) {
    MI->setDesc(TII->get(TargetOpcode::KILL));
    DEBUG(dbgs() << "subreg: replaced by: " << *MI);
    return true;
  }

  if (DstSubReg == InsReg) {
    // No need to insert an identity copy instruction.
    // Watch out for case like this:
    // %RAX<def> = SUBREG_TO_REG 0, %EAX<kill>, 3
    // We must leave %RAX live.
    if (DstReg != InsReg) {
      MI->setDesc(TII->get(TargetOpcode::KILL));
      MI->RemoveOperand(3);     // SubIdx
      MI->RemoveOperand(1);     // Imm
      DEBUG(dbgs() << "subreg: replace by: " << *MI);
      return true;
    }
    DEBUG(dbgs() << "subreg: eliminated!");
  } else {
    TII->copyPhysReg(*MBB, MI, MI->getDebugLoc(), DstSubReg, InsReg,
                     MI->getOperand(2).isKill());

    // Implicitly define DstReg for subsequent uses.
    MachineBasicBlock::iterator CopyMI = MI;
    --CopyMI;
    CopyMI->addRegisterDefined(DstReg);
    DEBUG(dbgs() << "subreg: " << *CopyMI);
  }

  DEBUG(dbgs() << '\n');
  MBB->erase(MI);
  return true;
}

bool ExpandPostRA::LowerCopy(MachineInstr *MI) {

  if (MI->allDefsAreDead()) {
    DEBUG(dbgs() << "dead copy: " << *MI);
    MI->setDesc(TII->get(TargetOpcode::KILL));
    DEBUG(dbgs() << "replaced by: " << *MI);
    return true;
  }

  MachineOperand &DstMO = MI->getOperand(0);
  MachineOperand &SrcMO = MI->getOperand(1);

  if (SrcMO.getReg() == DstMO.getReg()) {
    DEBUG(dbgs() << "identity copy: " << *MI);
    // No need to insert an identity copy instruction, but replace with a KILL
    // if liveness is changed.
    if (SrcMO.isUndef() || MI->getNumOperands() > 2) {
      // We must make sure the super-register gets killed. Replace the
      // instruction with KILL.
      MI->setDesc(TII->get(TargetOpcode::KILL));
      DEBUG(dbgs() << "replaced by:   " << *MI);
      return true;
    }
    // Vanilla identity copy.
    MI->eraseFromParent();
    return true;
  }

  DEBUG(dbgs() << "real copy:   " << *MI);
  TII->copyPhysReg(*MI->getParent(), MI, MI->getDebugLoc(),
                   DstMO.getReg(), SrcMO.getReg(), SrcMO.isKill());

  if (MI->getNumOperands() > 2)
    TransferImplicitDefs(MI);
  DEBUG({
    MachineBasicBlock::iterator dMI = MI;
    dbgs() << "replaced by: " << *(--dMI);
  });
  MI->eraseFromParent();
  return true;
}

/// runOnMachineFunction - Reduce subregister inserts and extracts to register
/// copies.
///
bool ExpandPostRA::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(dbgs() << "Machine Function\n"
               << "********** EXPANDING POST-RA PSEUDO INSTRS **********\n"
               << "********** Function: " << MF.getName() << '\n');
  TRI = MF.getSubtarget().getRegisterInfo();
  TII = MF.getSubtarget().getInstrInfo();

  bool MadeChange = false;

  for (MachineFunction::iterator mbbi = MF.begin(), mbbe = MF.end();
       mbbi != mbbe; ++mbbi) {
    for (MachineBasicBlock::iterator mi = mbbi->begin(), me = mbbi->end();
         mi != me;) {
      MachineInstr *MI = mi;
      // Advance iterator here because MI may be erased.
      ++mi;

      // Only expand pseudos.
      if (!MI->isPseudo())
        continue;

      // Give targets a chance to expand even standard pseudos.
      if (TII->expandPostRAPseudo(MI)) {
        MadeChange = true;
        continue;
      }

      // Expand standard pseudos.
      switch (MI->getOpcode()) {
      case TargetOpcode::SUBREG_TO_REG:
        MadeChange |= LowerSubregToReg(MI);
        break;
      case TargetOpcode::COPY:
        MadeChange |= LowerCopy(MI);
        break;
      case TargetOpcode::DBG_VALUE:
        continue;
      case TargetOpcode::INSERT_SUBREG:
      case TargetOpcode::EXTRACT_SUBREG:
        llvm_unreachable("Sub-register pseudos should have been eliminated.");
      }
    }
  }

  return MadeChange;
}
