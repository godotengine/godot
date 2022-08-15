//===-- llvm/CodeGen/DwarfExpression.h - Dwarf Compile Unit ---*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf compile unit.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DWARFEXPRESSION_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DWARFEXPRESSION_H

#include "llvm/IR/DebugInfo.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class AsmPrinter;
class ByteStreamer;
class TargetRegisterInfo;
class DwarfUnit;
class DIELoc;

/// Base class containing the logic for constructing DWARF expressions
/// independently of whether they are emitted into a DIE or into a .debug_loc
/// entry.
class DwarfExpression {
protected:
  // Various convenience accessors that extract things out of AsmPrinter.
  const TargetRegisterInfo &TRI;
  unsigned DwarfVersion;

public:
  DwarfExpression(const TargetRegisterInfo &TRI,
                  unsigned DwarfVersion)
    : TRI(TRI), DwarfVersion(DwarfVersion) {}
  virtual ~DwarfExpression() {}

  /// Output a dwarf operand and an optional assembler comment.
  virtual void EmitOp(uint8_t Op, const char *Comment = nullptr) = 0;
  /// Emit a raw signed value.
  virtual void EmitSigned(int64_t Value) = 0;
  /// Emit a raw unsigned value.
  virtual void EmitUnsigned(uint64_t Value) = 0;
  /// Return whether the given machine register is the frame register in the
  /// current function.
  virtual bool isFrameRegister(unsigned MachineReg) = 0;

  /// Emit a dwarf register operation.
  void AddReg(int DwarfReg, const char *Comment = nullptr);
  /// Emit an (double-)indirect dwarf register operation.
  void AddRegIndirect(int DwarfReg, int Offset, bool Deref = false);

  /// Emit a dwarf register operation for describing
  /// - a small value occupying only part of a register or
  /// - a register representing only part of a value.
  void AddOpPiece(unsigned SizeInBits, unsigned OffsetInBits = 0);
  /// Emit a shift-right dwarf expression.
  void AddShr(unsigned ShiftBy);

  /// Emit an indirect dwarf register operation for the given machine register.
  /// \return false if no DWARF register exists for MachineReg.
  bool AddMachineRegIndirect(unsigned MachineReg, int Offset = 0);

  /// \brief Emit a partial DWARF register operation.
  /// \param MachineReg        the register
  /// \param PieceSizeInBits   size and
  /// \param PieceOffsetInBits offset of the piece in bits, if this is one
  ///                          piece of an aggregate value.
  ///
  /// If size and offset is zero an operation for the entire
  /// register is emitted: Some targets do not provide a DWARF
  /// register number for every register.  If this is the case, this
  /// function will attempt to emit a DWARF register by emitting a
  /// piece of a super-register or by piecing together multiple
  /// subregisters that alias the register.
  ///
  /// \return false if no DWARF register exists for MachineReg.
  bool AddMachineRegPiece(unsigned MachineReg, unsigned PieceSizeInBits = 0,
                          unsigned PieceOffsetInBits = 0);

  /// Emit a signed constant.
  void AddSignedConstant(int Value);
  /// Emit an unsigned constant.
  void AddUnsignedConstant(unsigned Value);

  /// \brief Emit an entire expression on top of a machine register location.
  ///
  /// \param PieceOffsetInBits If this is one piece out of a fragmented
  /// location, this is the offset of the piece inside the entire variable.
  /// \return false if no DWARF register exists for MachineReg.
  bool AddMachineRegExpression(const DIExpression *Expr, unsigned MachineReg,
                               unsigned PieceOffsetInBits = 0);
  /// Emit a the operations remaining the DIExpressionIterator I.
  /// \param PieceOffsetInBits If this is one piece out of a fragmented
  /// location, this is the offset of the piece inside the entire variable.
  void AddExpression(DIExpression::expr_op_iterator I,
                     DIExpression::expr_op_iterator E,
                     unsigned PieceOffsetInBits = 0);
};

/// DwarfExpression implementation for .debug_loc entries.
class DebugLocDwarfExpression : public DwarfExpression {
  ByteStreamer &BS;

public:
  DebugLocDwarfExpression(const TargetRegisterInfo &TRI,
                          unsigned DwarfVersion, ByteStreamer &BS)
    : DwarfExpression(TRI, DwarfVersion), BS(BS) {}

  void EmitOp(uint8_t Op, const char *Comment = nullptr) override;
  void EmitSigned(int64_t Value) override;
  void EmitUnsigned(uint64_t Value) override;
  bool isFrameRegister(unsigned MachineReg) override;
};

/// DwarfExpression implementation for singular DW_AT_location.
class DIEDwarfExpression : public DwarfExpression {
const AsmPrinter &AP;
  DwarfUnit &DU;
  DIELoc &DIE;

public:
  DIEDwarfExpression(const AsmPrinter &AP, DwarfUnit &DU, DIELoc &DIE);
  void EmitOp(uint8_t Op, const char *Comment = nullptr) override;
  void EmitSigned(int64_t Value) override;
  void EmitUnsigned(uint64_t Value) override;
  bool isFrameRegister(unsigned MachineReg) override;
};
}

#endif
