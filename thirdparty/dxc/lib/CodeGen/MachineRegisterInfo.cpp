//===-- lib/Codegen/MachineRegisterInfo.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the MachineRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

// Pin the vtable to this file.
void MachineRegisterInfo::Delegate::anchor() {}

MachineRegisterInfo::MachineRegisterInfo(const MachineFunction *MF)
  : MF(MF), TheDelegate(nullptr), IsSSA(true), TracksLiveness(true),
    TracksSubRegLiveness(false) {
  VRegInfo.reserve(256);
  RegAllocHints.reserve(256);
  UsedRegUnits.resize(getTargetRegisterInfo()->getNumRegUnits());
  UsedPhysRegMask.resize(getTargetRegisterInfo()->getNumRegs());

  // Create the physreg use/def lists.
  PhysRegUseDefLists.resize(getTargetRegisterInfo()->getNumRegs(), nullptr);
}

/// setRegClass - Set the register class of the specified virtual register.
///
void
MachineRegisterInfo::setRegClass(unsigned Reg, const TargetRegisterClass *RC) {
  assert(RC && RC->isAllocatable() && "Invalid RC for virtual register");
  VRegInfo[Reg].first = RC;
}

const TargetRegisterClass *
MachineRegisterInfo::constrainRegClass(unsigned Reg,
                                       const TargetRegisterClass *RC,
                                       unsigned MinNumRegs) {
  const TargetRegisterClass *OldRC = getRegClass(Reg);
  if (OldRC == RC)
    return RC;
  const TargetRegisterClass *NewRC =
    getTargetRegisterInfo()->getCommonSubClass(OldRC, RC);
  if (!NewRC || NewRC == OldRC)
    return NewRC;
  if (NewRC->getNumRegs() < MinNumRegs)
    return nullptr;
  setRegClass(Reg, NewRC);
  return NewRC;
}

bool
MachineRegisterInfo::recomputeRegClass(unsigned Reg) {
  const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();
  const TargetRegisterClass *OldRC = getRegClass(Reg);
  const TargetRegisterClass *NewRC =
      getTargetRegisterInfo()->getLargestLegalSuperClass(OldRC, *MF);

  // Stop early if there is no room to grow.
  if (NewRC == OldRC)
    return false;

  // Accumulate constraints from all uses.
  for (MachineOperand &MO : reg_nodbg_operands(Reg)) {
    // Apply the effect of the given operand to NewRC.
    MachineInstr *MI = MO.getParent();
    unsigned OpNo = &MO - &MI->getOperand(0);
    NewRC = MI->getRegClassConstraintEffect(OpNo, NewRC, TII,
                                            getTargetRegisterInfo());
    if (!NewRC || NewRC == OldRC)
      return false;
  }
  setRegClass(Reg, NewRC);
  return true;
}

/// createVirtualRegister - Create and return a new virtual register in the
/// function with the specified register class.
///
unsigned
MachineRegisterInfo::createVirtualRegister(const TargetRegisterClass *RegClass){
  assert(RegClass && "Cannot create register without RegClass!");
  assert(RegClass->isAllocatable() &&
         "Virtual register RegClass must be allocatable.");

  // New virtual register number.
  unsigned Reg = TargetRegisterInfo::index2VirtReg(getNumVirtRegs());
  VRegInfo.grow(Reg);
  VRegInfo[Reg].first = RegClass;
  RegAllocHints.grow(Reg);
  if (TheDelegate)
    TheDelegate->MRI_NoteNewVirtualRegister(Reg);
  return Reg;
}

/// clearVirtRegs - Remove all virtual registers (after physreg assignment).
void MachineRegisterInfo::clearVirtRegs() {
#ifndef NDEBUG
  for (unsigned i = 0, e = getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    if (!VRegInfo[Reg].second)
      continue;
    verifyUseList(Reg);
    llvm_unreachable("Remaining virtual register operands");
  }
#endif
  VRegInfo.clear();
}

void MachineRegisterInfo::verifyUseList(unsigned Reg) const {
#ifndef NDEBUG
  bool Valid = true;
  for (MachineOperand &M : reg_operands(Reg)) {
    MachineOperand *MO = &M;
    MachineInstr *MI = MO->getParent();
    if (!MI) {
      errs() << PrintReg(Reg, getTargetRegisterInfo())
             << " use list MachineOperand " << MO
             << " has no parent instruction.\n";
      Valid = false;
      continue;
    }
    MachineOperand *MO0 = &MI->getOperand(0);
    unsigned NumOps = MI->getNumOperands();
    if (!(MO >= MO0 && MO < MO0+NumOps)) {
      errs() << PrintReg(Reg, getTargetRegisterInfo())
             << " use list MachineOperand " << MO
             << " doesn't belong to parent MI: " << *MI;
      Valid = false;
    }
    if (!MO->isReg()) {
      errs() << PrintReg(Reg, getTargetRegisterInfo())
             << " MachineOperand " << MO << ": " << *MO
             << " is not a register\n";
      Valid = false;
    }
    if (MO->getReg() != Reg) {
      errs() << PrintReg(Reg, getTargetRegisterInfo())
             << " use-list MachineOperand " << MO << ": "
             << *MO << " is the wrong register\n";
      Valid = false;
    }
  }
  assert(Valid && "Invalid use list");
#endif
}

void MachineRegisterInfo::verifyUseLists() const {
#ifndef NDEBUG
  for (unsigned i = 0, e = getNumVirtRegs(); i != e; ++i)
    verifyUseList(TargetRegisterInfo::index2VirtReg(i));
  for (unsigned i = 1, e = getTargetRegisterInfo()->getNumRegs(); i != e; ++i)
    verifyUseList(i);
#endif
}

/// Add MO to the linked list of operands for its register.
void MachineRegisterInfo::addRegOperandToUseList(MachineOperand *MO) {
  assert(!MO->isOnRegUseList() && "Already on list");
  MachineOperand *&HeadRef = getRegUseDefListHead(MO->getReg());
  MachineOperand *const Head = HeadRef;

  // Head points to the first list element.
  // Next is NULL on the last list element.
  // Prev pointers are circular, so Head->Prev == Last.

  // Head is NULL for an empty list.
  if (!Head) {
    MO->Contents.Reg.Prev = MO;
    MO->Contents.Reg.Next = nullptr;
    HeadRef = MO;
    return;
  }
  assert(MO->getReg() == Head->getReg() && "Different regs on the same list!");

  // Insert MO between Last and Head in the circular Prev chain.
  MachineOperand *Last = Head->Contents.Reg.Prev;
  assert(Last && "Inconsistent use list");
  assert(MO->getReg() == Last->getReg() && "Different regs on the same list!");
  Head->Contents.Reg.Prev = MO;
  MO->Contents.Reg.Prev = Last;

  // Def operands always precede uses. This allows def_iterator to stop early.
  // Insert def operands at the front, and use operands at the back.
  if (MO->isDef()) {
    // Insert def at the front.
    MO->Contents.Reg.Next = Head;
    HeadRef = MO;
  } else {
    // Insert use at the end.
    MO->Contents.Reg.Next = nullptr;
    Last->Contents.Reg.Next = MO;
  }
}

/// Remove MO from its use-def list.
void MachineRegisterInfo::removeRegOperandFromUseList(MachineOperand *MO) {
  assert(MO->isOnRegUseList() && "Operand not on use list");
  MachineOperand *&HeadRef = getRegUseDefListHead(MO->getReg());
  MachineOperand *const Head = HeadRef;
  assert(Head && "List already empty");

  // Unlink this from the doubly linked list of operands.
  MachineOperand *Next = MO->Contents.Reg.Next;
  MachineOperand *Prev = MO->Contents.Reg.Prev;

  // Prev links are circular, next link is NULL instead of looping back to Head.
  if (MO == Head)
    HeadRef = Next;
  else
    Prev->Contents.Reg.Next = Next;

  (Next ? Next : Head)->Contents.Reg.Prev = Prev;

  MO->Contents.Reg.Prev = nullptr;
  MO->Contents.Reg.Next = nullptr;
}

/// Move NumOps operands from Src to Dst, updating use-def lists as needed.
///
/// The Dst range is assumed to be uninitialized memory. (Or it may contain
/// operands that won't be destroyed, which is OK because the MO destructor is
/// trivial anyway).
///
/// The Src and Dst ranges may overlap.
void MachineRegisterInfo::moveOperands(MachineOperand *Dst,
                                       MachineOperand *Src,
                                       unsigned NumOps) {
  assert(Src != Dst && NumOps && "Noop moveOperands");

  // Copy backwards if Dst is within the Src range.
  int Stride = 1;
  if (Dst >= Src && Dst < Src + NumOps) {
    Stride = -1;
    Dst += NumOps - 1;
    Src += NumOps - 1;
  }

  // Copy one operand at a time.
  do {
    new (Dst) MachineOperand(*Src);

    // Dst takes Src's place in the use-def chain.
    if (Src->isReg()) {
      MachineOperand *&Head = getRegUseDefListHead(Src->getReg());
      MachineOperand *Prev = Src->Contents.Reg.Prev;
      MachineOperand *Next = Src->Contents.Reg.Next;
      assert(Head && "List empty, but operand is chained");
      assert(Prev && "Operand was not on use-def list");

      // Prev links are circular, next link is NULL instead of looping back to
      // Head.
      if (Src == Head)
        Head = Dst;
      else
        Prev->Contents.Reg.Next = Dst;

      // Update Prev pointer. This also works when Src was pointing to itself
      // in a 1-element list. In that case Head == Dst.
      (Next ? Next : Head)->Contents.Reg.Prev = Dst;
    }

    Dst += Stride;
    Src += Stride;
  } while (--NumOps);
}

/// replaceRegWith - Replace all instances of FromReg with ToReg in the
/// machine function.  This is like llvm-level X->replaceAllUsesWith(Y),
/// except that it also changes any definitions of the register as well.
/// If ToReg is a physical register we apply the sub register to obtain the
/// final/proper physical register.
void MachineRegisterInfo::replaceRegWith(unsigned FromReg, unsigned ToReg) {
  assert(FromReg != ToReg && "Cannot replace a reg with itself");

  const TargetRegisterInfo *TRI = getTargetRegisterInfo();
  
  // TODO: This could be more efficient by bulk changing the operands.
  for (reg_iterator I = reg_begin(FromReg), E = reg_end(); I != E; ) {
    MachineOperand &O = *I;
    ++I;
    if (TargetRegisterInfo::isPhysicalRegister(ToReg)) {
      O.substPhysReg(ToReg, *TRI);
    } else {
      O.setReg(ToReg);
    }
  }
}

/// getVRegDef - Return the machine instr that defines the specified virtual
/// register or null if none is found.  This assumes that the code is in SSA
/// form, so there should only be one definition.
MachineInstr *MachineRegisterInfo::getVRegDef(unsigned Reg) const {
  // Since we are in SSA form, we can use the first definition.
  def_instr_iterator I = def_instr_begin(Reg);
  assert((I.atEnd() || std::next(I) == def_instr_end()) &&
         "getVRegDef assumes a single definition or no definition");
  return !I.atEnd() ? &*I : nullptr;
}

/// getUniqueVRegDef - Return the unique machine instr that defines the
/// specified virtual register or null if none is found.  If there are
/// multiple definitions or no definition, return null.
MachineInstr *MachineRegisterInfo::getUniqueVRegDef(unsigned Reg) const {
  if (def_empty(Reg)) return nullptr;
  def_instr_iterator I = def_instr_begin(Reg);
  if (std::next(I) != def_instr_end())
    return nullptr;
  return &*I;
}

bool MachineRegisterInfo::hasOneNonDBGUse(unsigned RegNo) const {
  use_nodbg_iterator UI = use_nodbg_begin(RegNo);
  if (UI == use_nodbg_end())
    return false;
  return ++UI == use_nodbg_end();
}

/// clearKillFlags - Iterate over all the uses of the given register and
/// clear the kill flag from the MachineOperand. This function is used by
/// optimization passes which extend register lifetimes and need only
/// preserve conservative kill flag information.
void MachineRegisterInfo::clearKillFlags(unsigned Reg) const {
  for (MachineOperand &MO : use_operands(Reg))
    MO.setIsKill(false);
}

bool MachineRegisterInfo::isLiveIn(unsigned Reg) const {
  for (livein_iterator I = livein_begin(), E = livein_end(); I != E; ++I)
    if (I->first == Reg || I->second == Reg)
      return true;
  return false;
}

/// getLiveInPhysReg - If VReg is a live-in virtual register, return the
/// corresponding live-in physical register.
unsigned MachineRegisterInfo::getLiveInPhysReg(unsigned VReg) const {
  for (livein_iterator I = livein_begin(), E = livein_end(); I != E; ++I)
    if (I->second == VReg)
      return I->first;
  return 0;
}

/// getLiveInVirtReg - If PReg is a live-in physical register, return the
/// corresponding live-in physical register.
unsigned MachineRegisterInfo::getLiveInVirtReg(unsigned PReg) const {
  for (livein_iterator I = livein_begin(), E = livein_end(); I != E; ++I)
    if (I->first == PReg)
      return I->second;
  return 0;
}

/// EmitLiveInCopies - Emit copies to initialize livein virtual registers
/// into the given entry block.
void
MachineRegisterInfo::EmitLiveInCopies(MachineBasicBlock *EntryMBB,
                                      const TargetRegisterInfo &TRI,
                                      const TargetInstrInfo &TII) {
  // Emit the copies into the top of the block.
  for (unsigned i = 0, e = LiveIns.size(); i != e; ++i)
    if (LiveIns[i].second) {
      if (use_empty(LiveIns[i].second)) {
        // The livein has no uses. Drop it.
        //
        // It would be preferable to have isel avoid creating live-in
        // records for unused arguments in the first place, but it's
        // complicated by the debug info code for arguments.
        LiveIns.erase(LiveIns.begin() + i);
        --i; --e;
      } else {
        // Emit a copy.
        BuildMI(*EntryMBB, EntryMBB->begin(), DebugLoc(),
                TII.get(TargetOpcode::COPY), LiveIns[i].second)
          .addReg(LiveIns[i].first);

        // Add the register to the entry block live-in set.
        EntryMBB->addLiveIn(LiveIns[i].first);
      }
    } else {
      // Add the register to the entry block live-in set.
      EntryMBB->addLiveIn(LiveIns[i].first);
    }
}

unsigned MachineRegisterInfo::getMaxLaneMaskForVReg(unsigned Reg) const
{
  // Lane masks are only defined for vregs.
  assert(TargetRegisterInfo::isVirtualRegister(Reg));
  const TargetRegisterClass &TRC = *getRegClass(Reg);
  return TRC.getLaneMask();
}

#ifndef NDEBUG
void MachineRegisterInfo::dumpUses(unsigned Reg) const {
  for (MachineInstr &I : use_instructions(Reg))
    I.dump();
}
#endif

void MachineRegisterInfo::freezeReservedRegs(const MachineFunction &MF) {
  ReservedRegs = getTargetRegisterInfo()->getReservedRegs(MF);
  assert(ReservedRegs.size() == getTargetRegisterInfo()->getNumRegs() &&
         "Invalid ReservedRegs vector from target");
}

bool MachineRegisterInfo::isConstantPhysReg(unsigned PhysReg,
                                            const MachineFunction &MF) const {
  assert(TargetRegisterInfo::isPhysicalRegister(PhysReg));

  // Check if any overlapping register is modified, or allocatable so it may be
  // used later.
  for (MCRegAliasIterator AI(PhysReg, getTargetRegisterInfo(), true);
       AI.isValid(); ++AI)
    if (!def_empty(*AI) || isAllocatable(*AI))
      return false;
  return true;
}

/// markUsesInDebugValueAsUndef - Mark every DBG_VALUE referencing the
/// specified register as undefined which causes the DBG_VALUE to be
/// deleted during LiveDebugVariables analysis.
void MachineRegisterInfo::markUsesInDebugValueAsUndef(unsigned Reg) const {
  // Mark any DBG_VALUE that uses Reg as undef (but don't delete it.)
  MachineRegisterInfo::use_instr_iterator nextI;
  for (use_instr_iterator I = use_instr_begin(Reg), E = use_instr_end();
       I != E; I = nextI) {
    nextI = std::next(I);  // I is invalidated by the setReg
    MachineInstr *UseMI = &*I;
    if (UseMI->isDebugValue())
      UseMI->getOperand(0).setReg(0U);
  }
}

static const Function *getCalledFunction(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isGlobal())
      continue;
    const Function *Func = dyn_cast<Function>(MO.getGlobal());
    if (Func != nullptr)
      return Func;
  }
  return nullptr;
}

static bool isNoReturnDef(const MachineOperand &MO) {
  // Anything which is not a noreturn function is a real def.
  const MachineInstr &MI = *MO.getParent();
  if (!MI.isCall())
    return false;
  const MachineBasicBlock &MBB = *MI.getParent();
  if (!MBB.succ_empty())
    return false;
  const MachineFunction &MF = *MBB.getParent();
  // We need to keep correct unwind information even if the function will
  // not return, since the runtime may need it.
  if (MF.getFunction()->hasFnAttribute(Attribute::UWTable))
    return false;
  const Function *Called = getCalledFunction(MI);
  if (Called == nullptr || !Called->hasFnAttribute(Attribute::NoReturn)
      || !Called->hasFnAttribute(Attribute::NoUnwind))
    return false;

  return true;
}

bool MachineRegisterInfo::isPhysRegModified(unsigned PhysReg) const {
  if (UsedPhysRegMask.test(PhysReg))
    return true;
  const TargetRegisterInfo *TRI = getTargetRegisterInfo();
  for (MCRegAliasIterator AI(PhysReg, TRI, true); AI.isValid(); ++AI) {
    for (const MachineOperand &MO : make_range(def_begin(*AI), def_end())) {
      if (isNoReturnDef(MO))
        continue;
      return true;
    }
  }
  return false;
}
