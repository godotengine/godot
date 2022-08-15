//===----- TargetFrameLoweringImpl.cpp - Implement target frame interface --==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the layout of a stack frame on the target machine.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BitVector.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <cstdlib>
using namespace llvm;

TargetFrameLowering::~TargetFrameLowering() {
}

/// The default implementation just looks at attribute "no-frame-pointer-elim".
bool TargetFrameLowering::noFramePointerElim(const MachineFunction &MF) const {
  auto Attr = MF.getFunction()->getFnAttribute("no-frame-pointer-elim");
  return Attr.getValueAsString() == "true";
}

/// getFrameIndexOffset - Returns the displacement from the frame register to
/// the stack frame of the specified index. This is the default implementation
/// which is overridden for some targets.
int TargetFrameLowering::getFrameIndexOffset(const MachineFunction &MF,
                                             int FI) const {
  const MachineFrameInfo *MFI = MF.getFrameInfo();
  return MFI->getObjectOffset(FI) + MFI->getStackSize() -
    getOffsetOfLocalArea() + MFI->getOffsetAdjustment();
}

int TargetFrameLowering::getFrameIndexReference(const MachineFunction &MF,
                                             int FI, unsigned &FrameReg) const {
  const TargetRegisterInfo *RI = MF.getSubtarget().getRegisterInfo();

  // By default, assume all frame indices are referenced via whatever
  // getFrameRegister() says. The target can override this if it's doing
  // something different.
  FrameReg = RI->getFrameRegister(MF);
  return getFrameIndexOffset(MF, FI);
}

bool TargetFrameLowering::needsFrameIndexResolution(
    const MachineFunction &MF) const {
  return MF.getFrameInfo()->hasStackObjects();
}

void TargetFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                               BitVector &SavedRegs,
                                               RegScavenger *RS) const {
  // Get the callee saved register list...
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  const MCPhysReg *CSRegs = TRI.getCalleeSavedRegs(&MF);

  // Early exit if there are no callee saved registers.
  if (!CSRegs || CSRegs[0] == 0)
    return;

  SavedRegs.resize(TRI.getNumRegs());

  // In Naked functions we aren't going to save any registers.
  if (MF.getFunction()->hasFnAttribute(Attribute::Naked))
    return;

  // Functions which call __builtin_unwind_init get all their registers saved.
  bool CallsUnwindInit = MF.getMMI().callsUnwindInit();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  for (unsigned i = 0; CSRegs[i]; ++i) {
    unsigned Reg = CSRegs[i];
    if (CallsUnwindInit || MRI.isPhysRegModified(Reg))
      SavedRegs.set(Reg);
  }
}
