//===-- TargetInstrInfo.cpp - Target Instruction Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/ScoreboardHazardRecognizer.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include <cctype>
using namespace llvm;

static cl::opt<bool> DisableHazardRecognizer(
  "disable-sched-hazard", cl::Hidden, cl::init(false),
  cl::desc("Disable hazard detection during preRA scheduling"));

TargetInstrInfo::~TargetInstrInfo() {
}

const TargetRegisterClass*
TargetInstrInfo::getRegClass(const MCInstrDesc &MCID, unsigned OpNum,
                             const TargetRegisterInfo *TRI,
                             const MachineFunction &MF) const {
  if (OpNum >= MCID.getNumOperands())
    return nullptr;

  short RegClass = MCID.OpInfo[OpNum].RegClass;
  if (MCID.OpInfo[OpNum].isLookupPtrRegClass())
    return TRI->getPointerRegClass(MF, RegClass);

  // Instructions like INSERT_SUBREG do not have fixed register classes.
  if (RegClass < 0)
    return nullptr;

  // Otherwise just look it up normally.
  return TRI->getRegClass(RegClass);
}

/// insertNoop - Insert a noop into the instruction stream at the specified
/// point.
void TargetInstrInfo::insertNoop(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MI) const {
  llvm_unreachable("Target didn't implement insertNoop!");
}

/// Measure the specified inline asm to determine an approximation of its
/// length.
/// Comments (which run till the next SeparatorString or newline) do not
/// count as an instruction.
/// Any other non-whitespace text is considered an instruction, with
/// multiple instructions separated by SeparatorString or newlines.
/// Variable-length instructions are not handled here; this function
/// may be overloaded in the target code to do that.
unsigned TargetInstrInfo::getInlineAsmLength(const char *Str,
                                             const MCAsmInfo &MAI) const {


  // Count the number of instructions in the asm.
  bool atInsnStart = true;
  unsigned Length = 0;
  for (; *Str; ++Str) {
    if (*Str == '\n' || strncmp(Str, MAI.getSeparatorString(),
                                strlen(MAI.getSeparatorString())) == 0)
      atInsnStart = true;
    if (atInsnStart && !std::isspace(static_cast<unsigned char>(*Str))) {
      Length += MAI.getMaxInstLength();
      atInsnStart = false;
    }
    if (atInsnStart && strncmp(Str, MAI.getCommentString(),
                               strlen(MAI.getCommentString())) == 0)
      atInsnStart = false;
  }

  return Length;
}

/// ReplaceTailWithBranchTo - Delete the instruction OldInst and everything
/// after it, replacing it with an unconditional branch to NewDest.
void
TargetInstrInfo::ReplaceTailWithBranchTo(MachineBasicBlock::iterator Tail,
                                         MachineBasicBlock *NewDest) const {
  MachineBasicBlock *MBB = Tail->getParent();

  // Remove all the old successors of MBB from the CFG.
  while (!MBB->succ_empty())
    MBB->removeSuccessor(MBB->succ_begin());

  // Remove all the dead instructions from the end of MBB.
  MBB->erase(Tail, MBB->end());

  // If MBB isn't immediately before MBB, insert a branch to it.
  if (++MachineFunction::iterator(MBB) != MachineFunction::iterator(NewDest))
    InsertBranch(*MBB, NewDest, nullptr, SmallVector<MachineOperand, 0>(),
                 Tail->getDebugLoc());
  MBB->addSuccessor(NewDest);
}

// commuteInstruction - The default implementation of this method just exchanges
// the two operands returned by findCommutedOpIndices.
MachineInstr *TargetInstrInfo::commuteInstruction(MachineInstr *MI,
                                                  bool NewMI) const {
  const MCInstrDesc &MCID = MI->getDesc();
  bool HasDef = MCID.getNumDefs();
  if (HasDef && !MI->getOperand(0).isReg())
    // No idea how to commute this instruction. Target should implement its own.
    return nullptr;
  unsigned Idx1, Idx2;
  if (!findCommutedOpIndices(MI, Idx1, Idx2)) {
    assert(MI->isCommutable() && "Precondition violation: MI must be commutable.");
    return nullptr;
  }

  assert(MI->getOperand(Idx1).isReg() && MI->getOperand(Idx2).isReg() &&
         "This only knows how to commute register operands so far");
  unsigned Reg0 = HasDef ? MI->getOperand(0).getReg() : 0;
  unsigned Reg1 = MI->getOperand(Idx1).getReg();
  unsigned Reg2 = MI->getOperand(Idx2).getReg();
  unsigned SubReg0 = HasDef ? MI->getOperand(0).getSubReg() : 0;
  unsigned SubReg1 = MI->getOperand(Idx1).getSubReg();
  unsigned SubReg2 = MI->getOperand(Idx2).getSubReg();
  bool Reg1IsKill = MI->getOperand(Idx1).isKill();
  bool Reg2IsKill = MI->getOperand(Idx2).isKill();
  bool Reg1IsUndef = MI->getOperand(Idx1).isUndef();
  bool Reg2IsUndef = MI->getOperand(Idx2).isUndef();
  bool Reg1IsInternal = MI->getOperand(Idx1).isInternalRead();
  bool Reg2IsInternal = MI->getOperand(Idx2).isInternalRead();
  // If destination is tied to either of the commuted source register, then
  // it must be updated.
  if (HasDef && Reg0 == Reg1 &&
      MI->getDesc().getOperandConstraint(Idx1, MCOI::TIED_TO) == 0) {
    Reg2IsKill = false;
    Reg0 = Reg2;
    SubReg0 = SubReg2;
  } else if (HasDef && Reg0 == Reg2 &&
             MI->getDesc().getOperandConstraint(Idx2, MCOI::TIED_TO) == 0) {
    Reg1IsKill = false;
    Reg0 = Reg1;
    SubReg0 = SubReg1;
  }

  if (NewMI) {
    // Create a new instruction.
    MachineFunction &MF = *MI->getParent()->getParent();
    MI = MF.CloneMachineInstr(MI);
  }

  if (HasDef) {
    MI->getOperand(0).setReg(Reg0);
    MI->getOperand(0).setSubReg(SubReg0);
  }
  MI->getOperand(Idx2).setReg(Reg1);
  MI->getOperand(Idx1).setReg(Reg2);
  MI->getOperand(Idx2).setSubReg(SubReg1);
  MI->getOperand(Idx1).setSubReg(SubReg2);
  MI->getOperand(Idx2).setIsKill(Reg1IsKill);
  MI->getOperand(Idx1).setIsKill(Reg2IsKill);
  MI->getOperand(Idx2).setIsUndef(Reg1IsUndef);
  MI->getOperand(Idx1).setIsUndef(Reg2IsUndef);
  MI->getOperand(Idx2).setIsInternalRead(Reg1IsInternal);
  MI->getOperand(Idx1).setIsInternalRead(Reg2IsInternal);
  return MI;
}

/// findCommutedOpIndices - If specified MI is commutable, return the two
/// operand indices that would swap value. Return true if the instruction
/// is not in a form which this routine understands.
bool TargetInstrInfo::findCommutedOpIndices(MachineInstr *MI,
                                            unsigned &SrcOpIdx1,
                                            unsigned &SrcOpIdx2) const {
  assert(!MI->isBundle() &&
         "TargetInstrInfo::findCommutedOpIndices() can't handle bundles");

  const MCInstrDesc &MCID = MI->getDesc();
  if (!MCID.isCommutable())
    return false;
  // This assumes v0 = op v1, v2 and commuting would swap v1 and v2. If this
  // is not true, then the target must implement this.
  SrcOpIdx1 = MCID.getNumDefs();
  SrcOpIdx2 = SrcOpIdx1 + 1;
  if (!MI->getOperand(SrcOpIdx1).isReg() ||
      !MI->getOperand(SrcOpIdx2).isReg())
    // No idea.
    return false;
  return true;
}


bool
TargetInstrInfo::isUnpredicatedTerminator(const MachineInstr *MI) const {
  if (!MI->isTerminator()) return false;

  // Conditional branch is a special case.
  if (MI->isBranch() && !MI->isBarrier())
    return true;
  if (!MI->isPredicable())
    return true;
  return !isPredicated(MI);
}

bool TargetInstrInfo::PredicateInstruction(
    MachineInstr *MI, ArrayRef<MachineOperand> Pred) const {
  bool MadeChange = false;

  assert(!MI->isBundle() &&
         "TargetInstrInfo::PredicateInstruction() can't handle bundles");

  const MCInstrDesc &MCID = MI->getDesc();
  if (!MI->isPredicable())
    return false;

  for (unsigned j = 0, i = 0, e = MI->getNumOperands(); i != e; ++i) {
    if (MCID.OpInfo[i].isPredicate()) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg()) {
        MO.setReg(Pred[j].getReg());
        MadeChange = true;
      } else if (MO.isImm()) {
        MO.setImm(Pred[j].getImm());
        MadeChange = true;
      } else if (MO.isMBB()) {
        MO.setMBB(Pred[j].getMBB());
        MadeChange = true;
      }
      ++j;
    }
  }
  return MadeChange;
}

bool TargetInstrInfo::hasLoadFromStackSlot(const MachineInstr *MI,
                                           const MachineMemOperand *&MMO,
                                           int &FrameIndex) const {
  for (MachineInstr::mmo_iterator o = MI->memoperands_begin(),
         oe = MI->memoperands_end();
       o != oe;
       ++o) {
    if ((*o)->isLoad()) {
      if (const FixedStackPseudoSourceValue *Value =
          dyn_cast_or_null<FixedStackPseudoSourceValue>(
              (*o)->getPseudoValue())) {
        FrameIndex = Value->getFrameIndex();
        MMO = *o;
        return true;
      }
    }
  }
  return false;
}

bool TargetInstrInfo::hasStoreToStackSlot(const MachineInstr *MI,
                                          const MachineMemOperand *&MMO,
                                          int &FrameIndex) const {
  for (MachineInstr::mmo_iterator o = MI->memoperands_begin(),
         oe = MI->memoperands_end();
       o != oe;
       ++o) {
    if ((*o)->isStore()) {
      if (const FixedStackPseudoSourceValue *Value =
          dyn_cast_or_null<FixedStackPseudoSourceValue>(
              (*o)->getPseudoValue())) {
        FrameIndex = Value->getFrameIndex();
        MMO = *o;
        return true;
      }
    }
  }
  return false;
}

bool TargetInstrInfo::getStackSlotRange(const TargetRegisterClass *RC,
                                        unsigned SubIdx, unsigned &Size,
                                        unsigned &Offset,
                                        const MachineFunction &MF) const {
  if (!SubIdx) {
    Size = RC->getSize();
    Offset = 0;
    return true;
  }
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  unsigned BitSize = TRI->getSubRegIdxSize(SubIdx);
  // Convert bit size to byte size to be consistent with
  // MCRegisterClass::getSize().
  if (BitSize % 8)
    return false;

  int BitOffset = TRI->getSubRegIdxOffset(SubIdx);
  if (BitOffset < 0 || BitOffset % 8)
    return false;

  Size = BitSize /= 8;
  Offset = (unsigned)BitOffset / 8;

  assert(RC->getSize() >= (Offset + Size) && "bad subregister range");

  if (!MF.getTarget().getDataLayout()->isLittleEndian()) {
    Offset = RC->getSize() - (Offset + Size);
  }
  return true;
}

void TargetInstrInfo::reMaterialize(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator I,
                                    unsigned DestReg,
                                    unsigned SubIdx,
                                    const MachineInstr *Orig,
                                    const TargetRegisterInfo &TRI) const {
  MachineInstr *MI = MBB.getParent()->CloneMachineInstr(Orig);
  MI->substituteRegister(MI->getOperand(0).getReg(), DestReg, SubIdx, TRI);
  MBB.insert(I, MI);
}

bool
TargetInstrInfo::produceSameValue(const MachineInstr *MI0,
                                  const MachineInstr *MI1,
                                  const MachineRegisterInfo *MRI) const {
  return MI0->isIdenticalTo(MI1, MachineInstr::IgnoreVRegDefs);
}

MachineInstr *TargetInstrInfo::duplicate(MachineInstr *Orig,
                                         MachineFunction &MF) const {
  assert(!Orig->isNotDuplicable() &&
         "Instruction cannot be duplicated");
  return MF.CloneMachineInstr(Orig);
}

// If the COPY instruction in MI can be folded to a stack operation, return
// the register class to use.
static const TargetRegisterClass *canFoldCopy(const MachineInstr *MI,
                                              unsigned FoldIdx) {
  assert(MI->isCopy() && "MI must be a COPY instruction");
  if (MI->getNumOperands() != 2)
    return nullptr;
  assert(FoldIdx<2 && "FoldIdx refers no nonexistent operand");

  const MachineOperand &FoldOp = MI->getOperand(FoldIdx);
  const MachineOperand &LiveOp = MI->getOperand(1-FoldIdx);

  if (FoldOp.getSubReg() || LiveOp.getSubReg())
    return nullptr;

  unsigned FoldReg = FoldOp.getReg();
  unsigned LiveReg = LiveOp.getReg();

  assert(TargetRegisterInfo::isVirtualRegister(FoldReg) &&
         "Cannot fold physregs");

  const MachineRegisterInfo &MRI = MI->getParent()->getParent()->getRegInfo();
  const TargetRegisterClass *RC = MRI.getRegClass(FoldReg);

  if (TargetRegisterInfo::isPhysicalRegister(LiveOp.getReg()))
    return RC->contains(LiveOp.getReg()) ? RC : nullptr;

  if (RC->hasSubClassEq(MRI.getRegClass(LiveReg)))
    return RC;

  // FIXME: Allow folding when register classes are memory compatible.
  return nullptr;
}

void TargetInstrInfo::getNoopForMachoTarget(MCInst &NopInst) const {
  llvm_unreachable("Not a MachO target");
}

bool TargetInstrInfo::canFoldMemoryOperand(const MachineInstr *MI,
                                           ArrayRef<unsigned> Ops) const {
  return MI->isCopy() && Ops.size() == 1 && canFoldCopy(MI, Ops[0]);
}

static MachineInstr *foldPatchpoint(MachineFunction &MF, MachineInstr *MI,
                                    ArrayRef<unsigned> Ops, int FrameIndex,
                                    const TargetInstrInfo &TII) {
  unsigned StartIdx = 0;
  switch (MI->getOpcode()) {
  case TargetOpcode::STACKMAP:
    StartIdx = 2; // Skip ID, nShadowBytes.
    break;
  case TargetOpcode::PATCHPOINT: {
    // For PatchPoint, the call args are not foldable.
    PatchPointOpers opers(MI);
    StartIdx = opers.getVarIdx();
    break;
  }
  default:
    llvm_unreachable("unexpected stackmap opcode");
  }

  // Return false if any operands requested for folding are not foldable (not
  // part of the stackmap's live values).
  for (unsigned Op : Ops) {
    if (Op < StartIdx)
      return nullptr;
  }

  MachineInstr *NewMI =
    MF.CreateMachineInstr(TII.get(MI->getOpcode()), MI->getDebugLoc(), true);
  MachineInstrBuilder MIB(MF, NewMI);

  // No need to fold return, the meta data, and function arguments
  for (unsigned i = 0; i < StartIdx; ++i)
    MIB.addOperand(MI->getOperand(i));

  for (unsigned i = StartIdx; i < MI->getNumOperands(); ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (std::find(Ops.begin(), Ops.end(), i) != Ops.end()) {
      unsigned SpillSize;
      unsigned SpillOffset;
      // Compute the spill slot size and offset.
      const TargetRegisterClass *RC =
        MF.getRegInfo().getRegClass(MO.getReg());
      bool Valid =
          TII.getStackSlotRange(RC, MO.getSubReg(), SpillSize, SpillOffset, MF);
      if (!Valid)
        report_fatal_error("cannot spill patchpoint subregister operand");
      MIB.addImm(StackMaps::IndirectMemRefOp);
      MIB.addImm(SpillSize);
      MIB.addFrameIndex(FrameIndex);
      MIB.addImm(SpillOffset);
    }
    else
      MIB.addOperand(MO);
  }
  return NewMI;
}

/// foldMemoryOperand - Attempt to fold a load or store of the specified stack
/// slot into the specified machine instruction for the specified operand(s).
/// If this is possible, a new instruction is returned with the specified
/// operand folded, otherwise NULL is returned. The client is responsible for
/// removing the old instruction and adding the new one in the instruction
/// stream.
MachineInstr *TargetInstrInfo::foldMemoryOperand(MachineBasicBlock::iterator MI,
                                                 ArrayRef<unsigned> Ops,
                                                 int FI) const {
  unsigned Flags = 0;
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    if (MI->getOperand(Ops[i]).isDef())
      Flags |= MachineMemOperand::MOStore;
    else
      Flags |= MachineMemOperand::MOLoad;

  MachineBasicBlock *MBB = MI->getParent();
  assert(MBB && "foldMemoryOperand needs an inserted instruction");
  MachineFunction &MF = *MBB->getParent();

  MachineInstr *NewMI = nullptr;

  if (MI->getOpcode() == TargetOpcode::STACKMAP ||
      MI->getOpcode() == TargetOpcode::PATCHPOINT) {
    // Fold stackmap/patchpoint.
    NewMI = foldPatchpoint(MF, MI, Ops, FI, *this);
    if (NewMI)
      MBB->insert(MI, NewMI);
  } else {
    // Ask the target to do the actual folding.
    NewMI = foldMemoryOperandImpl(MF, MI, Ops, MI, FI);
  }

  if (NewMI) {
    NewMI->setMemRefs(MI->memoperands_begin(), MI->memoperands_end());
    // Add a memory operand, foldMemoryOperandImpl doesn't do that.
    assert((!(Flags & MachineMemOperand::MOStore) ||
            NewMI->mayStore()) &&
           "Folded a def to a non-store!");
    assert((!(Flags & MachineMemOperand::MOLoad) ||
            NewMI->mayLoad()) &&
           "Folded a use to a non-load!");
    const MachineFrameInfo &MFI = *MF.getFrameInfo();
    assert(MFI.getObjectOffset(FI) != -1);
    MachineMemOperand *MMO =
      MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(FI),
                              Flags, MFI.getObjectSize(FI),
                              MFI.getObjectAlignment(FI));
    NewMI->addMemOperand(MF, MMO);

    return NewMI;
  }

  // Straight COPY may fold as load/store.
  if (!MI->isCopy() || Ops.size() != 1)
    return nullptr;

  const TargetRegisterClass *RC = canFoldCopy(MI, Ops[0]);
  if (!RC)
    return nullptr;

  const MachineOperand &MO = MI->getOperand(1-Ops[0]);
  MachineBasicBlock::iterator Pos = MI;
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  if (Flags == MachineMemOperand::MOStore)
    storeRegToStackSlot(*MBB, Pos, MO.getReg(), MO.isKill(), FI, RC, TRI);
  else
    loadRegFromStackSlot(*MBB, Pos, MO.getReg(), FI, RC, TRI);
  return --Pos;
}

/// foldMemoryOperand - Same as the previous version except it allows folding
/// of any load and store from / to any address, not just from a specific
/// stack slot.
MachineInstr *TargetInstrInfo::foldMemoryOperand(MachineBasicBlock::iterator MI,
                                                 ArrayRef<unsigned> Ops,
                                                 MachineInstr *LoadMI) const {
  assert(LoadMI->canFoldAsLoad() && "LoadMI isn't foldable!");
#ifndef NDEBUG
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    assert(MI->getOperand(Ops[i]).isUse() && "Folding load into def!");
#endif
  MachineBasicBlock &MBB = *MI->getParent();
  MachineFunction &MF = *MBB.getParent();

  // Ask the target to do the actual folding.
  MachineInstr *NewMI = nullptr;
  int FrameIndex = 0;

  if ((MI->getOpcode() == TargetOpcode::STACKMAP ||
       MI->getOpcode() == TargetOpcode::PATCHPOINT) &&
      isLoadFromStackSlot(LoadMI, FrameIndex)) {
    // Fold stackmap/patchpoint.
    NewMI = foldPatchpoint(MF, MI, Ops, FrameIndex, *this);
    if (NewMI)
      NewMI = MBB.insert(MI, NewMI);
  } else {
    // Ask the target to do the actual folding.
    NewMI = foldMemoryOperandImpl(MF, MI, Ops, MI, LoadMI);
  }

  if (!NewMI) return nullptr;

  // Copy the memoperands from the load to the folded instruction.
  if (MI->memoperands_empty()) {
    NewMI->setMemRefs(LoadMI->memoperands_begin(),
                      LoadMI->memoperands_end());
  }
  else {
    // Handle the rare case of folding multiple loads.
    NewMI->setMemRefs(MI->memoperands_begin(),
                      MI->memoperands_end());
    for (MachineInstr::mmo_iterator I = LoadMI->memoperands_begin(),
           E = LoadMI->memoperands_end(); I != E; ++I) {
      NewMI->addMemOperand(MF, *I);
    }
  }
  return NewMI;
}

bool TargetInstrInfo::
isReallyTriviallyReMaterializableGeneric(const MachineInstr *MI,
                                         AliasAnalysis *AA) const {
  const MachineFunction &MF = *MI->getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();

  // Remat clients assume operand 0 is the defined register.
  if (!MI->getNumOperands() || !MI->getOperand(0).isReg())
    return false;
  unsigned DefReg = MI->getOperand(0).getReg();

  // A sub-register definition can only be rematerialized if the instruction
  // doesn't read the other parts of the register.  Otherwise it is really a
  // read-modify-write operation on the full virtual register which cannot be
  // moved safely.
  if (TargetRegisterInfo::isVirtualRegister(DefReg) &&
      MI->getOperand(0).getSubReg() && MI->readsVirtualRegister(DefReg))
    return false;

  // A load from a fixed stack slot can be rematerialized. This may be
  // redundant with subsequent checks, but it's target-independent,
  // simple, and a common case.
  int FrameIdx = 0;
  if (isLoadFromStackSlot(MI, FrameIdx) &&
      MF.getFrameInfo()->isImmutableObjectIndex(FrameIdx))
    return true;

  // Avoid instructions obviously unsafe for remat.
  if (MI->isNotDuplicable() || MI->mayStore() ||
      MI->hasUnmodeledSideEffects())
    return false;

  // Don't remat inline asm. We have no idea how expensive it is
  // even if it's side effect free.
  if (MI->isInlineAsm())
    return false;

  // Avoid instructions which load from potentially varying memory.
  if (MI->mayLoad() && !MI->isInvariantLoad(AA))
    return false;

  // If any of the registers accessed are non-constant, conservatively assume
  // the instruction is not rematerializable.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg()) continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0)
      continue;

    // Check for a well-behaved physical register.
    if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
      if (MO.isUse()) {
        // If the physreg has no defs anywhere, it's just an ambient register
        // and we can freely move its uses. Alternatively, if it's allocatable,
        // it could get allocated to something with a def during allocation.
        if (!MRI.isConstantPhysReg(Reg, MF))
          return false;
      } else {
        // A physreg def. We can't remat it.
        return false;
      }
      continue;
    }

    // Only allow one virtual-register def.  There may be multiple defs of the
    // same virtual register, though.
    if (MO.isDef() && Reg != DefReg)
      return false;

    // Don't allow any virtual-register uses. Rematting an instruction with
    // virtual register uses would length the live ranges of the uses, which
    // is not necessarily a good idea, certainly not "trivial".
    if (MO.isUse())
      return false;
  }

  // Everything checked out.
  return true;
}

int TargetInstrInfo::getSPAdjust(const MachineInstr *MI) const {
  const MachineFunction *MF = MI->getParent()->getParent();
  const TargetFrameLowering *TFI = MF->getSubtarget().getFrameLowering();
  bool StackGrowsDown =
    TFI->getStackGrowthDirection() == TargetFrameLowering::StackGrowsDown;

  unsigned FrameSetupOpcode = getCallFrameSetupOpcode();
  unsigned FrameDestroyOpcode = getCallFrameDestroyOpcode();

  if (MI->getOpcode() != FrameSetupOpcode &&
      MI->getOpcode() != FrameDestroyOpcode)
    return 0;
 
  int SPAdj = MI->getOperand(0).getImm();

  if ((!StackGrowsDown && MI->getOpcode() == FrameSetupOpcode) ||
       (StackGrowsDown && MI->getOpcode() == FrameDestroyOpcode))
    SPAdj = -SPAdj;

  return SPAdj;
}

/// isSchedulingBoundary - Test if the given instruction should be
/// considered a scheduling boundary. This primarily includes labels
/// and terminators.
bool TargetInstrInfo::isSchedulingBoundary(const MachineInstr *MI,
                                           const MachineBasicBlock *MBB,
                                           const MachineFunction &MF) const {
  // Terminators and labels can't be scheduled around.
  if (MI->isTerminator() || MI->isPosition())
    return true;

  // Don't attempt to schedule around any instruction that defines
  // a stack-oriented pointer, as it's unlikely to be profitable. This
  // saves compile time, because it doesn't require every single
  // stack slot reference to depend on the instruction that does the
  // modification.
  const TargetLowering &TLI = *MF.getSubtarget().getTargetLowering();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  if (MI->modifiesRegister(TLI.getStackPointerRegisterToSaveRestore(), TRI))
    return true;

  return false;
}

// Provide a global flag for disabling the PreRA hazard recognizer that targets
// may choose to honor.
bool TargetInstrInfo::usePreRAHazardRecognizer() const {
  return !DisableHazardRecognizer;
}

// Default implementation of CreateTargetRAHazardRecognizer.
ScheduleHazardRecognizer *TargetInstrInfo::
CreateTargetHazardRecognizer(const TargetSubtargetInfo *STI,
                             const ScheduleDAG *DAG) const {
  // Dummy hazard recognizer allows all instructions to issue.
  return new ScheduleHazardRecognizer();
}

// Default implementation of CreateTargetMIHazardRecognizer.
ScheduleHazardRecognizer *TargetInstrInfo::
CreateTargetMIHazardRecognizer(const InstrItineraryData *II,
                               const ScheduleDAG *DAG) const {
  return (ScheduleHazardRecognizer *)
    new ScoreboardHazardRecognizer(II, DAG, "misched");
}

// Default implementation of CreateTargetPostRAHazardRecognizer.
ScheduleHazardRecognizer *TargetInstrInfo::
CreateTargetPostRAHazardRecognizer(const InstrItineraryData *II,
                                   const ScheduleDAG *DAG) const {
  return (ScheduleHazardRecognizer *)
    new ScoreboardHazardRecognizer(II, DAG, "post-RA-sched");
}

//===----------------------------------------------------------------------===//
//  SelectionDAG latency interface.
//===----------------------------------------------------------------------===//

int
TargetInstrInfo::getOperandLatency(const InstrItineraryData *ItinData,
                                   SDNode *DefNode, unsigned DefIdx,
                                   SDNode *UseNode, unsigned UseIdx) const {
  if (!ItinData || ItinData->isEmpty())
    return -1;

  if (!DefNode->isMachineOpcode())
    return -1;

  unsigned DefClass = get(DefNode->getMachineOpcode()).getSchedClass();
  if (!UseNode->isMachineOpcode())
    return ItinData->getOperandCycle(DefClass, DefIdx);
  unsigned UseClass = get(UseNode->getMachineOpcode()).getSchedClass();
  return ItinData->getOperandLatency(DefClass, DefIdx, UseClass, UseIdx);
}

int TargetInstrInfo::getInstrLatency(const InstrItineraryData *ItinData,
                                     SDNode *N) const {
  if (!ItinData || ItinData->isEmpty())
    return 1;

  if (!N->isMachineOpcode())
    return 1;

  return ItinData->getStageLatency(get(N->getMachineOpcode()).getSchedClass());
}

//===----------------------------------------------------------------------===//
//  MachineInstr latency interface.
//===----------------------------------------------------------------------===//

unsigned
TargetInstrInfo::getNumMicroOps(const InstrItineraryData *ItinData,
                                const MachineInstr *MI) const {
  if (!ItinData || ItinData->isEmpty())
    return 1;

  unsigned Class = MI->getDesc().getSchedClass();
  int UOps = ItinData->Itineraries[Class].NumMicroOps;
  if (UOps >= 0)
    return UOps;

  // The # of u-ops is dynamically determined. The specific target should
  // override this function to return the right number.
  return 1;
}

/// Return the default expected latency for a def based on it's opcode.
unsigned TargetInstrInfo::defaultDefLatency(const MCSchedModel &SchedModel,
                                            const MachineInstr *DefMI) const {
  if (DefMI->isTransient())
    return 0;
  if (DefMI->mayLoad())
    return SchedModel.LoadLatency;
  if (isHighLatencyDef(DefMI->getOpcode()))
    return SchedModel.HighLatency;
  return 1;
}

unsigned TargetInstrInfo::getPredicationCost(const MachineInstr *) const {
  return 0;
}

unsigned TargetInstrInfo::
getInstrLatency(const InstrItineraryData *ItinData,
                const MachineInstr *MI,
                unsigned *PredCost) const {
  // Default to one cycle for no itinerary. However, an "empty" itinerary may
  // still have a MinLatency property, which getStageLatency checks.
  if (!ItinData)
    return MI->mayLoad() ? 2 : 1;

  return ItinData->getStageLatency(MI->getDesc().getSchedClass());
}

bool TargetInstrInfo::hasLowDefLatency(const TargetSchedModel &SchedModel,
                                       const MachineInstr *DefMI,
                                       unsigned DefIdx) const {
  const InstrItineraryData *ItinData = SchedModel.getInstrItineraries();
  if (!ItinData || ItinData->isEmpty())
    return false;

  unsigned DefClass = DefMI->getDesc().getSchedClass();
  int DefCycle = ItinData->getOperandCycle(DefClass, DefIdx);
  return (DefCycle != -1 && DefCycle <= 1);
}

/// Both DefMI and UseMI must be valid.  By default, call directly to the
/// itinerary. This may be overriden by the target.
int TargetInstrInfo::
getOperandLatency(const InstrItineraryData *ItinData,
                  const MachineInstr *DefMI, unsigned DefIdx,
                  const MachineInstr *UseMI, unsigned UseIdx) const {
  unsigned DefClass = DefMI->getDesc().getSchedClass();
  unsigned UseClass = UseMI->getDesc().getSchedClass();
  return ItinData->getOperandLatency(DefClass, DefIdx, UseClass, UseIdx);
}

/// If we can determine the operand latency from the def only, without itinerary
/// lookup, do so. Otherwise return -1.
int TargetInstrInfo::computeDefOperandLatency(
  const InstrItineraryData *ItinData,
  const MachineInstr *DefMI) const {

  // Let the target hook getInstrLatency handle missing itineraries.
  if (!ItinData)
    return getInstrLatency(ItinData, DefMI);

  if(ItinData->isEmpty())
    return defaultDefLatency(ItinData->SchedModel, DefMI);

  // ...operand lookup required
  return -1;
}

/// computeOperandLatency - Compute and return the latency of the given data
/// dependent def and use when the operand indices are already known. UseMI may
/// be NULL for an unknown use.
///
/// FindMin may be set to get the minimum vs. expected latency. Minimum
/// latency is used for scheduling groups, while expected latency is for
/// instruction cost and critical path.
///
/// Depending on the subtarget's itinerary properties, this may or may not need
/// to call getOperandLatency(). For most subtargets, we don't need DefIdx or
/// UseIdx to compute min latency.
unsigned TargetInstrInfo::
computeOperandLatency(const InstrItineraryData *ItinData,
                      const MachineInstr *DefMI, unsigned DefIdx,
                      const MachineInstr *UseMI, unsigned UseIdx) const {

  int DefLatency = computeDefOperandLatency(ItinData, DefMI);
  if (DefLatency >= 0)
    return DefLatency;

  assert(ItinData && !ItinData->isEmpty() && "computeDefOperandLatency fail");

  int OperLatency = 0;
  if (UseMI)
    OperLatency = getOperandLatency(ItinData, DefMI, DefIdx, UseMI, UseIdx);
  else {
    unsigned DefClass = DefMI->getDesc().getSchedClass();
    OperLatency = ItinData->getOperandCycle(DefClass, DefIdx);
  }
  if (OperLatency >= 0)
    return OperLatency;

  // No operand latency was found.
  unsigned InstrLatency = getInstrLatency(ItinData, DefMI);

  // Expected latency is the max of the stage latency and itinerary props.
  InstrLatency = std::max(InstrLatency,
                          defaultDefLatency(ItinData->SchedModel, DefMI));
  return InstrLatency;
}

bool TargetInstrInfo::getRegSequenceInputs(
    const MachineInstr &MI, unsigned DefIdx,
    SmallVectorImpl<RegSubRegPairAndIdx> &InputRegs) const {
  assert((MI.isRegSequence() ||
          MI.isRegSequenceLike()) && "Instruction do not have the proper type");

  if (!MI.isRegSequence())
    return getRegSequenceLikeInputs(MI, DefIdx, InputRegs);

  // We are looking at:
  // Def = REG_SEQUENCE v0, sub0, v1, sub1, ...
  assert(DefIdx == 0 && "REG_SEQUENCE only has one def");
  for (unsigned OpIdx = 1, EndOpIdx = MI.getNumOperands(); OpIdx != EndOpIdx;
       OpIdx += 2) {
    const MachineOperand &MOReg = MI.getOperand(OpIdx);
    const MachineOperand &MOSubIdx = MI.getOperand(OpIdx + 1);
    assert(MOSubIdx.isImm() &&
           "One of the subindex of the reg_sequence is not an immediate");
    // Record Reg:SubReg, SubIdx.
    InputRegs.push_back(RegSubRegPairAndIdx(MOReg.getReg(), MOReg.getSubReg(),
                                            (unsigned)MOSubIdx.getImm()));
  }
  return true;
}

bool TargetInstrInfo::getExtractSubregInputs(
    const MachineInstr &MI, unsigned DefIdx,
    RegSubRegPairAndIdx &InputReg) const {
  assert((MI.isExtractSubreg() ||
      MI.isExtractSubregLike()) && "Instruction do not have the proper type");

  if (!MI.isExtractSubreg())
    return getExtractSubregLikeInputs(MI, DefIdx, InputReg);

  // We are looking at:
  // Def = EXTRACT_SUBREG v0.sub1, sub0.
  assert(DefIdx == 0 && "EXTRACT_SUBREG only has one def");
  const MachineOperand &MOReg = MI.getOperand(1);
  const MachineOperand &MOSubIdx = MI.getOperand(2);
  assert(MOSubIdx.isImm() &&
         "The subindex of the extract_subreg is not an immediate");

  InputReg.Reg = MOReg.getReg();
  InputReg.SubReg = MOReg.getSubReg();
  InputReg.SubIdx = (unsigned)MOSubIdx.getImm();
  return true;
}

bool TargetInstrInfo::getInsertSubregInputs(
    const MachineInstr &MI, unsigned DefIdx,
    RegSubRegPair &BaseReg, RegSubRegPairAndIdx &InsertedReg) const {
  assert((MI.isInsertSubreg() ||
      MI.isInsertSubregLike()) && "Instruction do not have the proper type");

  if (!MI.isInsertSubreg())
    return getInsertSubregLikeInputs(MI, DefIdx, BaseReg, InsertedReg);

  // We are looking at:
  // Def = INSERT_SEQUENCE v0, v1, sub0.
  assert(DefIdx == 0 && "INSERT_SUBREG only has one def");
  const MachineOperand &MOBaseReg = MI.getOperand(1);
  const MachineOperand &MOInsertedReg = MI.getOperand(2);
  const MachineOperand &MOSubIdx = MI.getOperand(3);
  assert(MOSubIdx.isImm() &&
         "One of the subindex of the reg_sequence is not an immediate");
  BaseReg.Reg = MOBaseReg.getReg();
  BaseReg.SubReg = MOBaseReg.getSubReg();

  InsertedReg.Reg = MOInsertedReg.getReg();
  InsertedReg.SubReg = MOInsertedReg.getSubReg();
  InsertedReg.SubIdx = (unsigned)MOSubIdx.getImm();
  return true;
}
