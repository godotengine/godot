//===-- llvm/MC/MCInst.h - MCInst class -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MCInst and MCOperand classes, which
// is the basic representation used to represent low-level machine code
// instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCINST_H
#define LLVM_MC_MCINST_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/SMLoc.h"

namespace llvm {
class raw_ostream;
class MCAsmInfo;
class MCInstPrinter;
class MCExpr;
class MCInst;

/// \brief Instances of this class represent operands of the MCInst class.
/// This is a simple discriminated union.
class MCOperand {
  enum MachineOperandType : unsigned char {
    kInvalid,     ///< Uninitialized.
    kRegister,    ///< Register operand.
    kImmediate,   ///< Immediate operand.
    kFPImmediate, ///< Floating-point immediate operand.
    kExpr,        ///< Relocatable immediate operand.
    kInst         ///< Sub-instruction operand.
  };
  MachineOperandType Kind;

  union {
    unsigned RegVal;
    int64_t ImmVal;
    double FPImmVal;
    const MCExpr *ExprVal;
    const MCInst *InstVal;
  };

public:
  MCOperand() : Kind(kInvalid), FPImmVal(0.0) {}

  bool isValid() const { return Kind != kInvalid; }
  bool isReg() const { return Kind == kRegister; }
  bool isImm() const { return Kind == kImmediate; }
  bool isFPImm() const { return Kind == kFPImmediate; }
  bool isExpr() const { return Kind == kExpr; }
  bool isInst() const { return Kind == kInst; }

  /// \brief Returns the register number.
  unsigned getReg() const {
    assert(isReg() && "This is not a register operand!");
    return RegVal;
  }

  /// \brief Set the register number.
  void setReg(unsigned Reg) {
    assert(isReg() && "This is not a register operand!");
    RegVal = Reg;
  }

  int64_t getImm() const {
    assert(isImm() && "This is not an immediate");
    return ImmVal;
  }
  void setImm(int64_t Val) {
    assert(isImm() && "This is not an immediate");
    ImmVal = Val;
  }

  double getFPImm() const {
    assert(isFPImm() && "This is not an FP immediate");
    return FPImmVal;
  }

  void setFPImm(double Val) {
    assert(isFPImm() && "This is not an FP immediate");
    FPImmVal = Val;
  }

  const MCExpr *getExpr() const {
    assert(isExpr() && "This is not an expression");
    return ExprVal;
  }
  void setExpr(const MCExpr *Val) {
    assert(isExpr() && "This is not an expression");
    ExprVal = Val;
  }

  const MCInst *getInst() const {
    assert(isInst() && "This is not a sub-instruction");
    return InstVal;
  }
  void setInst(const MCInst *Val) {
    assert(isInst() && "This is not a sub-instruction");
    InstVal = Val;
  }

  static MCOperand createReg(unsigned Reg) {
    MCOperand Op;
    Op.Kind = kRegister;
    Op.RegVal = Reg;
    return Op;
  }
  static MCOperand createImm(int64_t Val) {
    MCOperand Op;
    Op.Kind = kImmediate;
    Op.ImmVal = Val;
    return Op;
  }
  static MCOperand createFPImm(double Val) {
    MCOperand Op;
    Op.Kind = kFPImmediate;
    Op.FPImmVal = Val;
    return Op;
  }
  static MCOperand createExpr(const MCExpr *Val) {
    MCOperand Op;
    Op.Kind = kExpr;
    Op.ExprVal = Val;
    return Op;
  }
  static MCOperand createInst(const MCInst *Val) {
    MCOperand Op;
    Op.Kind = kInst;
    Op.InstVal = Val;
    return Op;
  }

  void print(raw_ostream &OS) const;
  void dump() const;
};

template <> struct isPodLike<MCOperand> { static const bool value = true; };

/// \brief Instances of this class represent a single low-level machine
/// instruction.
class MCInst {
  unsigned Opcode;
  SMLoc Loc;
  SmallVector<MCOperand, 8> Operands;

public:
  MCInst() : Opcode(0) {}

  void setOpcode(unsigned Op) { Opcode = Op; }
  unsigned getOpcode() const { return Opcode; }

  void setLoc(SMLoc loc) { Loc = loc; }
  SMLoc getLoc() const { return Loc; }

  const MCOperand &getOperand(unsigned i) const { return Operands[i]; }
  MCOperand &getOperand(unsigned i) { return Operands[i]; }
  unsigned getNumOperands() const { return Operands.size(); }

  void addOperand(const MCOperand &Op) { Operands.push_back(Op); }

  typedef SmallVectorImpl<MCOperand>::iterator iterator;
  typedef SmallVectorImpl<MCOperand>::const_iterator const_iterator;
  void clear() { Operands.clear(); }
  void erase(iterator I) { Operands.erase(I); }
  size_t size() const { return Operands.size(); }
  iterator begin() { return Operands.begin(); }
  const_iterator begin() const { return Operands.begin(); }
  iterator end() { return Operands.end(); }
  const_iterator end() const { return Operands.end(); }
  iterator insert(iterator I, const MCOperand &Op) {
    return Operands.insert(I, Op);
  }

  void print(raw_ostream &OS) const;
  void dump() const;

  /// \brief Dump the MCInst as prettily as possible using the additional MC
  /// structures, if given. Operators are separated by the \p Separator
  /// string.
  void dump_pretty(raw_ostream &OS, const MCInstPrinter *Printer = nullptr,
                   StringRef Separator = " ") const;
};

inline raw_ostream& operator<<(raw_ostream &OS, const MCOperand &MO) {
  MO.print(OS);
  return OS;
}

inline raw_ostream& operator<<(raw_ostream &OS, const MCInst &MI) {
  MI.print(OS);
  return OS;
}

} // end namespace llvm

#endif
