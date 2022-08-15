//===--- LivePhysRegs.cpp - Live Physical Register Set --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LivePhysRegs utility for tracking liveness of
// physical registers across machine instructions in forward or backward order.
// A more detailed description can be found in the corresponding header file.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;


/// \brief Remove all registers from the set that get clobbered by the register
/// mask.
/// The clobbers set will be the list of live registers clobbered
/// by the regmask.
void LivePhysRegs::removeRegsInMask(const MachineOperand &MO,
        SmallVectorImpl<std::pair<unsigned, const MachineOperand*>> *Clobbers) {
  SparseSet<unsigned>::iterator LRI = LiveRegs.begin();
  while (LRI != LiveRegs.end()) {
    if (MO.clobbersPhysReg(*LRI)) {
      if (Clobbers)
        Clobbers->push_back(std::make_pair(*LRI, &MO));
      LRI = LiveRegs.erase(LRI);
    } else
      ++LRI;
  }
}

/// Simulates liveness when stepping backwards over an instruction(bundle):
/// Remove Defs, add uses. This is the recommended way of calculating liveness.
void LivePhysRegs::stepBackward(const MachineInstr &MI) {
  // Remove defined registers and regmask kills from the set.
  for (ConstMIBundleOperands O(&MI); O.isValid(); ++O) {
    if (O->isReg()) {
      if (!O->isDef())
        continue;
      unsigned Reg = O->getReg();
      if (Reg == 0)
        continue;
      removeReg(Reg);
    } else if (O->isRegMask())
      removeRegsInMask(*O, nullptr);
  }

  // Add uses to the set.
  for (ConstMIBundleOperands O(&MI); O.isValid(); ++O) {
    if (!O->isReg() || !O->readsReg() || O->isUndef())
      continue;
    unsigned Reg = O->getReg();
    if (Reg == 0)
      continue;
    addReg(Reg);
  }
}

/// Simulates liveness when stepping forward over an instruction(bundle): Remove
/// killed-uses, add defs. This is the not recommended way, because it depends
/// on accurate kill flags. If possible use stepBackwards() instead of this
/// function.
void LivePhysRegs::stepForward(const MachineInstr &MI,
        SmallVectorImpl<std::pair<unsigned, const MachineOperand*>> &Clobbers) {
  // Remove killed registers from the set.
  for (ConstMIBundleOperands O(&MI); O.isValid(); ++O) {
    if (O->isReg()) {
      unsigned Reg = O->getReg();
      if (Reg == 0)
        continue;
      if (O->isDef()) {
        // Note, dead defs are still recorded.  The caller should decide how to
        // handle them.
        Clobbers.push_back(std::make_pair(Reg, &*O));
      } else {
        if (!O->isKill())
          continue;
        assert(O->isUse());
        removeReg(Reg);
      }
    } else if (O->isRegMask())
      removeRegsInMask(*O, &Clobbers);
  }

  // Add defs to the set.
  for (auto Reg : Clobbers) {
    // Skip dead defs.  They shouldn't be added to the set.
    if (Reg.second->isReg() && Reg.second->isDead())
      continue;
    addReg(Reg.first);
  }
}

/// Prin the currently live registers to OS.
void LivePhysRegs::print(raw_ostream &OS) const {
  OS << "Live Registers:";
  if (!TRI) {
    OS << " (uninitialized)\n";
    return;
  }

  if (empty()) {
    OS << " (empty)\n";
    return;
  }

  for (const_iterator I = begin(), E = end(); I != E; ++I)
    OS << " " << PrintReg(*I, TRI);
  OS << "\n";
}

/// Dumps the currently live registers to the debug output.
void LivePhysRegs::dump() const {
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  dbgs() << "  " << *this;
#endif
}

/// Add live-in registers of basic block \p MBB to \p LiveRegs.
static void addLiveIns(LivePhysRegs &LiveRegs, const MachineBasicBlock &MBB) {
  for (unsigned Reg : make_range(MBB.livein_begin(), MBB.livein_end()))
    LiveRegs.addReg(Reg);
}

/// Add pristine registers to the given \p LiveRegs. This function removes
/// actually saved callee save registers when \p InPrologueEpilogue is false.
static void addPristines(LivePhysRegs &LiveRegs, const MachineFunction &MF,
                         const TargetRegisterInfo &TRI) {
  const MachineFrameInfo &MFI = *MF.getFrameInfo();
  if (!MFI.isCalleeSavedInfoValid())
    return;

  for (const MCPhysReg *CSR = TRI.getCalleeSavedRegs(&MF); CSR && *CSR; ++CSR)
    LiveRegs.addReg(*CSR);
  for (const CalleeSavedInfo &Info : MFI.getCalleeSavedInfo())
    LiveRegs.removeReg(Info.getReg());
}

void LivePhysRegs::addLiveOuts(const MachineBasicBlock *MBB,
                               bool AddPristines) {
  if (AddPristines) {
    const MachineFunction &MF = *MBB->getParent();
    addPristines(*this, MF, *TRI);
  }
  for (const MachineBasicBlock *Succ : MBB->successors())
    ::addLiveIns(*this, *Succ);
}

void LivePhysRegs::addLiveIns(const MachineBasicBlock *MBB,
                              bool AddPristines) {
  if (AddPristines) {
    const MachineFunction &MF = *MBB->getParent();
    addPristines(*this, MF, *TRI);
  }
  ::addLiveIns(*this, *MBB);
}
