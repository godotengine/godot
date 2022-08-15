//===-- llvm/MC/MCFixup.h - Instruction Relocation and Patching -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCFIXUP_H
#define LLVM_MC_MCFIXUP_H

#include "llvm/MC/MCExpr.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SMLoc.h"
#include <cassert>

namespace llvm {
class MCExpr;

/// \brief Extensible enumeration to represent the type of a fixup.
enum MCFixupKind {
  FK_Data_1 = 0, ///< A one-byte fixup.
  FK_Data_2,     ///< A two-byte fixup.
  FK_Data_4,     ///< A four-byte fixup.
  FK_Data_8,     ///< A eight-byte fixup.
  FK_PCRel_1,    ///< A one-byte pc relative fixup.
  FK_PCRel_2,    ///< A two-byte pc relative fixup.
  FK_PCRel_4,    ///< A four-byte pc relative fixup.
  FK_PCRel_8,    ///< A eight-byte pc relative fixup.
  FK_GPRel_1,    ///< A one-byte gp relative fixup.
  FK_GPRel_2,    ///< A two-byte gp relative fixup.
  FK_GPRel_4,    ///< A four-byte gp relative fixup.
  FK_GPRel_8,    ///< A eight-byte gp relative fixup.
  FK_SecRel_1,   ///< A one-byte section relative fixup.
  FK_SecRel_2,   ///< A two-byte section relative fixup.
  FK_SecRel_4,   ///< A four-byte section relative fixup.
  FK_SecRel_8,   ///< A eight-byte section relative fixup.

  FirstTargetFixupKind = 128,

  // Limit range of target fixups, in case we want to pack more efficiently
  // later.
  MaxTargetFixupKind = (1 << 8)
};

/// \brief Encode information on a single operation to perform on a byte
/// sequence (e.g., an encoded instruction) which requires assemble- or run-
/// time patching.
///
/// Fixups are used any time the target instruction encoder needs to represent
/// some value in an instruction which is not yet concrete. The encoder will
/// encode the instruction assuming the value is 0, and emit a fixup which
/// communicates to the assembler backend how it should rewrite the encoded
/// value.
///
/// During the process of relaxation, the assembler will apply fixups as
/// symbolic values become concrete. When relaxation is complete, any remaining
/// fixups become relocations in the object file (or errors, if the fixup cannot
/// be encoded on the target).
class MCFixup {
  /// The value to put into the fixup location. The exact interpretation of the
  /// expression is target dependent, usually it will be one of the operands to
  /// an instruction or an assembler directive.
  const MCExpr *Value;

  /// The byte index of start of the relocation inside the encoded instruction.
  uint32_t Offset;

  /// The target dependent kind of fixup item this is. The kind is used to
  /// determine how the operand value should be encoded into the instruction.
  unsigned Kind;

  /// The source location which gave rise to the fixup, if any.
  SMLoc Loc;
public:
  static MCFixup create(uint32_t Offset, const MCExpr *Value,
                        MCFixupKind Kind, SMLoc Loc = SMLoc()) {
    assert(unsigned(Kind) < MaxTargetFixupKind && "Kind out of range!");
    MCFixup FI;
    FI.Value = Value;
    FI.Offset = Offset;
    FI.Kind = unsigned(Kind);
    FI.Loc = Loc;
    return FI;
  }

  MCFixupKind getKind() const { return MCFixupKind(Kind); }

  uint32_t getOffset() const { return Offset; }
  void setOffset(uint32_t Value) { Offset = Value; }

  const MCExpr *getValue() const { return Value; }

  /// \brief Return the generic fixup kind for a value with the given size. It
  /// is an error to pass an unsupported size.
  static MCFixupKind getKindForSize(unsigned Size, bool isPCRel) {
    switch (Size) {
    default: llvm_unreachable("Invalid generic fixup size!");
    case 1: return isPCRel ? FK_PCRel_1 : FK_Data_1;
    case 2: return isPCRel ? FK_PCRel_2 : FK_Data_2;
    case 4: return isPCRel ? FK_PCRel_4 : FK_Data_4;
    case 8: return isPCRel ? FK_PCRel_8 : FK_Data_8;
    }
  }

  SMLoc getLoc() const { return Loc; }
};

} // End llvm namespace

#endif
