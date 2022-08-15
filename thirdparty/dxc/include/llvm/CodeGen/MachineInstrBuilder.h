//===-- CodeGen/MachineInstBuilder.h - Simplify creation of MIs -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes a function named BuildMI, which is useful for dramatically
// simplifying how MachineInstr's are created.  It allows use of code like this:
//
//   M = BuildMI(X86::ADDrr8, 2).addReg(argVal1).addReg(argVal2);
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEINSTRBUILDER_H
#define LLVM_CODEGEN_MACHINEINSTRBUILDER_H

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

class MCInstrDesc;
class MDNode;

namespace RegState {
  enum {
    Define         = 0x2,
    Implicit       = 0x4,
    Kill           = 0x8,
    Dead           = 0x10,
    Undef          = 0x20,
    EarlyClobber   = 0x40,
    Debug          = 0x80,
    InternalRead   = 0x100,
    DefineNoRead   = Define | Undef,
    ImplicitDefine = Implicit | Define,
    ImplicitKill   = Implicit | Kill
  };
}

class MachineInstrBuilder {
  MachineFunction *MF;
  MachineInstr *MI;
public:
  MachineInstrBuilder() : MF(nullptr), MI(nullptr) {}

  /// Create a MachineInstrBuilder for manipulating an existing instruction.
  /// F must be the machine function  that was used to allocate I.
  MachineInstrBuilder(MachineFunction &F, MachineInstr *I) : MF(&F), MI(I) {}

  /// Allow automatic conversion to the machine instruction we are working on.
  ///
  operator MachineInstr*() const { return MI; }
  MachineInstr *operator->() const { return MI; }
  operator MachineBasicBlock::iterator() const { return MI; }

  /// If conversion operators fail, use this method to get the MachineInstr
  /// explicitly.
  MachineInstr *getInstr() const { return MI; }

  /// addReg - Add a new virtual register operand...
  ///
  const
  MachineInstrBuilder &addReg(unsigned RegNo, unsigned flags = 0,
                              unsigned SubReg = 0) const {
    assert((flags & 0x1) == 0 &&
           "Passing in 'true' to addReg is forbidden! Use enums instead.");
    MI->addOperand(*MF, MachineOperand::CreateReg(RegNo,
                                               flags & RegState::Define,
                                               flags & RegState::Implicit,
                                               flags & RegState::Kill,
                                               flags & RegState::Dead,
                                               flags & RegState::Undef,
                                               flags & RegState::EarlyClobber,
                                               SubReg,
                                               flags & RegState::Debug,
                                               flags & RegState::InternalRead));
    return *this;
  }

  /// addImm - Add a new immediate operand.
  ///
  const MachineInstrBuilder &addImm(int64_t Val) const {
    MI->addOperand(*MF, MachineOperand::CreateImm(Val));
    return *this;
  }

  const MachineInstrBuilder &addCImm(const ConstantInt *Val) const {
    MI->addOperand(*MF, MachineOperand::CreateCImm(Val));
    return *this;
  }

  const MachineInstrBuilder &addFPImm(const ConstantFP *Val) const {
    MI->addOperand(*MF, MachineOperand::CreateFPImm(Val));
    return *this;
  }

  const MachineInstrBuilder &addMBB(MachineBasicBlock *MBB,
                                    unsigned char TargetFlags = 0) const {
    MI->addOperand(*MF, MachineOperand::CreateMBB(MBB, TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &addFrameIndex(int Idx) const {
    MI->addOperand(*MF, MachineOperand::CreateFI(Idx));
    return *this;
  }

  const MachineInstrBuilder &addConstantPoolIndex(unsigned Idx,
                                                  int Offset = 0,
                                          unsigned char TargetFlags = 0) const {
    MI->addOperand(*MF, MachineOperand::CreateCPI(Idx, Offset, TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &addTargetIndex(unsigned Idx, int64_t Offset = 0,
                                          unsigned char TargetFlags = 0) const {
    MI->addOperand(*MF, MachineOperand::CreateTargetIndex(Idx, Offset,
                                                          TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &addJumpTableIndex(unsigned Idx,
                                          unsigned char TargetFlags = 0) const {
    MI->addOperand(*MF, MachineOperand::CreateJTI(Idx, TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &addGlobalAddress(const GlobalValue *GV,
                                              int64_t Offset = 0,
                                          unsigned char TargetFlags = 0) const {
    MI->addOperand(*MF, MachineOperand::CreateGA(GV, Offset, TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &addExternalSymbol(const char *FnName,
                                          unsigned char TargetFlags = 0) const {
    MI->addOperand(*MF, MachineOperand::CreateES(FnName, TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &addBlockAddress(const BlockAddress *BA,
                                             int64_t Offset = 0,
                                          unsigned char TargetFlags = 0) const {
    MI->addOperand(*MF, MachineOperand::CreateBA(BA, Offset, TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &addRegMask(const uint32_t *Mask) const {
    MI->addOperand(*MF, MachineOperand::CreateRegMask(Mask));
    return *this;
  }

  const MachineInstrBuilder &addMemOperand(MachineMemOperand *MMO) const {
    MI->addMemOperand(*MF, MMO);
    return *this;
  }

  const MachineInstrBuilder &setMemRefs(MachineInstr::mmo_iterator b,
                                        MachineInstr::mmo_iterator e) const {
    MI->setMemRefs(b, e);
    return *this;
  }


  const MachineInstrBuilder &addOperand(const MachineOperand &MO) const {
    MI->addOperand(*MF, MO);
    return *this;
  }

  const MachineInstrBuilder &addMetadata(const MDNode *MD) const {
    MI->addOperand(*MF, MachineOperand::CreateMetadata(MD));
    assert((MI->isDebugValue() ? static_cast<bool>(MI->getDebugVariable())
                               : true) &&
           "first MDNode argument of a DBG_VALUE not a variable");
    return *this;
  }

  const MachineInstrBuilder &addCFIIndex(unsigned CFIIndex) const {
    MI->addOperand(*MF, MachineOperand::CreateCFIIndex(CFIIndex));
    return *this;
  }

  const MachineInstrBuilder &addSym(MCSymbol *Sym,
                                    unsigned char TargetFlags = 0) const {
    MI->addOperand(*MF, MachineOperand::CreateMCSymbol(Sym, TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &setMIFlags(unsigned Flags) const {
    MI->setFlags(Flags);
    return *this;
  }

  const MachineInstrBuilder &setMIFlag(MachineInstr::MIFlag Flag) const {
    MI->setFlag(Flag);
    return *this;
  }

  // Add a displacement from an existing MachineOperand with an added offset.
  const MachineInstrBuilder &addDisp(const MachineOperand &Disp, int64_t off,
                                     unsigned char TargetFlags = 0) const {
    switch (Disp.getType()) {
      default:
        llvm_unreachable("Unhandled operand type in addDisp()");
      case MachineOperand::MO_Immediate:
        return addImm(Disp.getImm() + off);
      case MachineOperand::MO_GlobalAddress: {
        // If caller specifies new TargetFlags then use it, otherwise the
        // default behavior is to copy the target flags from the existing
        // MachineOperand. This means if the caller wants to clear the
        // target flags it needs to do so explicitly.
        if (TargetFlags)
          return addGlobalAddress(Disp.getGlobal(), Disp.getOffset() + off,
                                  TargetFlags);
        return addGlobalAddress(Disp.getGlobal(), Disp.getOffset() + off,
                                Disp.getTargetFlags());
      }
    }
  }

  /// Copy all the implicit operands from OtherMI onto this one.
  const MachineInstrBuilder &copyImplicitOps(const MachineInstr *OtherMI) {
    MI->copyImplicitOps(*MF, OtherMI);
    return *this;
  }
};

/// BuildMI - Builder interface.  Specify how to create the initial instruction
/// itself.
///
inline MachineInstrBuilder BuildMI(MachineFunction &MF,
                                   DebugLoc DL,
                                   const MCInstrDesc &MCID) {
  return MachineInstrBuilder(MF, MF.CreateMachineInstr(MCID, DL));
}

/// BuildMI - This version of the builder sets up the first operand as a
/// destination virtual register.
///
inline MachineInstrBuilder BuildMI(MachineFunction &MF,
                                   DebugLoc DL,
                                   const MCInstrDesc &MCID,
                                   unsigned DestReg) {
  return MachineInstrBuilder(MF, MF.CreateMachineInstr(MCID, DL))
           .addReg(DestReg, RegState::Define);
}

/// BuildMI - This version of the builder inserts the newly-built
/// instruction before the given position in the given MachineBasicBlock, and
/// sets up the first operand as a destination virtual register.
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB,
                                   MachineBasicBlock::iterator I,
                                   DebugLoc DL,
                                   const MCInstrDesc &MCID,
                                   unsigned DestReg) {
  MachineFunction &MF = *BB.getParent();
  MachineInstr *MI = MF.CreateMachineInstr(MCID, DL);
  BB.insert(I, MI);
  return MachineInstrBuilder(MF, MI).addReg(DestReg, RegState::Define);
}

inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB,
                                   MachineBasicBlock::instr_iterator I,
                                   DebugLoc DL,
                                   const MCInstrDesc &MCID,
                                   unsigned DestReg) {
  MachineFunction &MF = *BB.getParent();
  MachineInstr *MI = MF.CreateMachineInstr(MCID, DL);
  BB.insert(I, MI);
  return MachineInstrBuilder(MF, MI).addReg(DestReg, RegState::Define);
}

inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB,
                                   MachineInstr *I,
                                   DebugLoc DL,
                                   const MCInstrDesc &MCID,
                                   unsigned DestReg) {
  if (I->isInsideBundle()) {
    MachineBasicBlock::instr_iterator MII = I;
    return BuildMI(BB, MII, DL, MCID, DestReg);
  }

  MachineBasicBlock::iterator MII = I;
  return BuildMI(BB, MII, DL, MCID, DestReg);
}

/// BuildMI - This version of the builder inserts the newly-built
/// instruction before the given position in the given MachineBasicBlock, and
/// does NOT take a destination register.
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB,
                                   MachineBasicBlock::iterator I,
                                   DebugLoc DL,
                                   const MCInstrDesc &MCID) {
  MachineFunction &MF = *BB.getParent();
  MachineInstr *MI = MF.CreateMachineInstr(MCID, DL);
  BB.insert(I, MI);
  return MachineInstrBuilder(MF, MI);
}

inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB,
                                   MachineBasicBlock::instr_iterator I,
                                   DebugLoc DL,
                                   const MCInstrDesc &MCID) {
  MachineFunction &MF = *BB.getParent();
  MachineInstr *MI = MF.CreateMachineInstr(MCID, DL);
  BB.insert(I, MI);
  return MachineInstrBuilder(MF, MI);
}

inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB,
                                   MachineInstr *I,
                                   DebugLoc DL,
                                   const MCInstrDesc &MCID) {
  if (I->isInsideBundle()) {
    MachineBasicBlock::instr_iterator MII = I;
    return BuildMI(BB, MII, DL, MCID);
  }

  MachineBasicBlock::iterator MII = I;
  return BuildMI(BB, MII, DL, MCID);
}

/// BuildMI - This version of the builder inserts the newly-built
/// instruction at the end of the given MachineBasicBlock, and does NOT take a
/// destination register.
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock *BB,
                                   DebugLoc DL,
                                   const MCInstrDesc &MCID) {
  return BuildMI(*BB, BB->end(), DL, MCID);
}

/// BuildMI - This version of the builder inserts the newly-built
/// instruction at the end of the given MachineBasicBlock, and sets up the first
/// operand as a destination virtual register.
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock *BB,
                                   DebugLoc DL,
                                   const MCInstrDesc &MCID,
                                   unsigned DestReg) {
  return BuildMI(*BB, BB->end(), DL, MCID, DestReg);
}

/// BuildMI - This version of the builder builds a DBG_VALUE intrinsic
/// for either a value in a register or a register-indirect+offset
/// address.  The convention is that a DBG_VALUE is indirect iff the
/// second operand is an immediate.
///
inline MachineInstrBuilder BuildMI(MachineFunction &MF, DebugLoc DL,
                                   const MCInstrDesc &MCID, bool IsIndirect,
                                   unsigned Reg, unsigned Offset,
                                   const MDNode *Variable, const MDNode *Expr) {
  assert(isa<DILocalVariable>(Variable) && "not a variable");
  assert(cast<DIExpression>(Expr)->isValid() && "not an expression");
  assert(cast<DILocalVariable>(Variable)->isValidLocationForIntrinsic(DL) &&
         "Expected inlined-at fields to agree");
  if (IsIndirect)
    return BuildMI(MF, DL, MCID)
        .addReg(Reg, RegState::Debug)
        .addImm(Offset)
        .addMetadata(Variable)
        .addMetadata(Expr);
  else {
    assert(Offset == 0 && "A direct address cannot have an offset.");
    return BuildMI(MF, DL, MCID)
        .addReg(Reg, RegState::Debug)
        .addReg(0U, RegState::Debug)
        .addMetadata(Variable)
        .addMetadata(Expr);
  }
}

/// BuildMI - This version of the builder builds a DBG_VALUE intrinsic
/// for either a value in a register or a register-indirect+offset
/// address and inserts it at position I.
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB,
                                   MachineBasicBlock::iterator I, DebugLoc DL,
                                   const MCInstrDesc &MCID, bool IsIndirect,
                                   unsigned Reg, unsigned Offset,
                                   const MDNode *Variable, const MDNode *Expr) {
  assert(isa<DILocalVariable>(Variable) && "not a variable");
  assert(cast<DIExpression>(Expr)->isValid() && "not an expression");
  MachineFunction &MF = *BB.getParent();
  MachineInstr *MI =
      BuildMI(MF, DL, MCID, IsIndirect, Reg, Offset, Variable, Expr);
  BB.insert(I, MI);
  return MachineInstrBuilder(MF, MI);
}


inline unsigned getDefRegState(bool B) {
  return B ? RegState::Define : 0;
}
inline unsigned getImplRegState(bool B) {
  return B ? RegState::Implicit : 0;
}
inline unsigned getKillRegState(bool B) {
  return B ? RegState::Kill : 0;
}
inline unsigned getDeadRegState(bool B) {
  return B ? RegState::Dead : 0;
}
inline unsigned getUndefRegState(bool B) {
  return B ? RegState::Undef : 0;
}
inline unsigned getInternalReadRegState(bool B) {
  return B ? RegState::InternalRead : 0;
}
inline unsigned getDebugRegState(bool B) {
  return B ? RegState::Debug : 0;
}


/// Helper class for constructing bundles of MachineInstrs.
///
/// MIBundleBuilder can create a bundle from scratch by inserting new
/// MachineInstrs one at a time, or it can create a bundle from a sequence of
/// existing MachineInstrs in a basic block.
class MIBundleBuilder {
  MachineBasicBlock &MBB;
  MachineBasicBlock::instr_iterator Begin;
  MachineBasicBlock::instr_iterator End;

public:
  /// Create an MIBundleBuilder that inserts instructions into a new bundle in
  /// BB above the bundle or instruction at Pos.
  MIBundleBuilder(MachineBasicBlock &BB,
                  MachineBasicBlock::iterator Pos)
    : MBB(BB), Begin(Pos.getInstrIterator()), End(Begin) {}

  /// Create a bundle from the sequence of instructions between B and E.
  MIBundleBuilder(MachineBasicBlock &BB,
                  MachineBasicBlock::iterator B,
                  MachineBasicBlock::iterator E)
    : MBB(BB), Begin(B.getInstrIterator()), End(E.getInstrIterator()) {
    assert(B != E && "No instructions to bundle");
    ++B;
    while (B != E) {
      MachineInstr *MI = B;
      ++B;
      MI->bundleWithPred();
    }
  }

  /// Create an MIBundleBuilder representing an existing instruction or bundle
  /// that has MI as its head.
  explicit MIBundleBuilder(MachineInstr *MI)
    : MBB(*MI->getParent()), Begin(MI), End(getBundleEnd(MI)) {}

  /// Return a reference to the basic block containing this bundle.
  MachineBasicBlock &getMBB() const { return MBB; }

  /// Return true if no instructions have been inserted in this bundle yet.
  /// Empty bundles aren't representable in a MachineBasicBlock.
  bool empty() const { return Begin == End; }

  /// Return an iterator to the first bundled instruction.
  MachineBasicBlock::instr_iterator begin() const { return Begin; }

  /// Return an iterator beyond the last bundled instruction.
  MachineBasicBlock::instr_iterator end() const { return End; }

  /// Insert MI into this bundle before I which must point to an instruction in
  /// the bundle, or end().
  MIBundleBuilder &insert(MachineBasicBlock::instr_iterator I,
                          MachineInstr *MI) {
    MBB.insert(I, MI);
    if (I == Begin) {
      if (!empty())
        MI->bundleWithSucc();
      Begin = MI;
      return *this;
    }
    if (I == End) {
      MI->bundleWithPred();
      return *this;
    }
    // MI was inserted in the middle of the bundle, so its neighbors' flags are
    // already fine. Update MI's bundle flags manually.
    MI->setFlag(MachineInstr::BundledPred);
    MI->setFlag(MachineInstr::BundledSucc);
    return *this;
  }

  /// Insert MI into MBB by prepending it to the instructions in the bundle.
  /// MI will become the first instruction in the bundle.
  MIBundleBuilder &prepend(MachineInstr *MI) {
    return insert(begin(), MI);
  }

  /// Insert MI into MBB by appending it to the instructions in the bundle.
  /// MI will become the last instruction in the bundle.
  MIBundleBuilder &append(MachineInstr *MI) {
    return insert(end(), MI);
  }
};

} // End llvm namespace

#endif
