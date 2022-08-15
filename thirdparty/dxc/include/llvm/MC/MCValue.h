//===-- llvm/MC/MCValue.h - MCValue class -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MCValue class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCVALUE_H
#define LLVM_MC_MCVALUE_H

#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/DataTypes.h"
#include <cassert>

namespace llvm {
class MCAsmInfo;
class raw_ostream;

/// \brief This represents an "assembler immediate".
///
///  In its most general form, this can hold ":Kind:(SymbolA - SymbolB +
///  imm64)".  Not all targets supports relocations of this general form, but we
///  need to represent this anyway.
///
/// In general both SymbolA and SymbolB will also have a modifier
/// analogous to the top-level Kind. Current targets are not expected
/// to make use of both though. The choice comes down to whether
/// relocation modifiers apply to the closest symbol or the whole
/// expression.
///
/// In the general form, SymbolB can only be defined if SymbolA is, and both
/// must be in the same (non-external) section. The latter constraint is not
/// enforced, since a symbol's section may not be known at construction.
///
/// Note that this class must remain a simple POD value class, because we need
/// it to live in unions etc.
class MCValue {
  const MCSymbolRefExpr *SymA, *SymB;
  int64_t Cst;
  uint32_t RefKind;
public:

  int64_t getConstant() const { return Cst; }
  const MCSymbolRefExpr *getSymA() const { return SymA; }
  const MCSymbolRefExpr *getSymB() const { return SymB; }
  uint32_t getRefKind() const { return RefKind; }

  /// \brief Is this an absolute (as opposed to relocatable) value.
  bool isAbsolute() const { return !SymA && !SymB; }

  /// \brief Print the value to the stream \p OS.
  void print(raw_ostream &OS) const;

  /// \brief Print the value to stderr.
  void dump() const;

  MCSymbolRefExpr::VariantKind getAccessVariant() const;

  static MCValue get(const MCSymbolRefExpr *SymA,
                     const MCSymbolRefExpr *SymB = nullptr,
                     int64_t Val = 0, uint32_t RefKind = 0) {
    MCValue R;
    assert((!SymB || SymA) && "Invalid relocatable MCValue!");
    R.Cst = Val;
    R.SymA = SymA;
    R.SymB = SymB;
    R.RefKind = RefKind;
    return R;
  }

  static MCValue get(int64_t Val) {
    MCValue R;
    R.Cst = Val;
    R.SymA = nullptr;
    R.SymB = nullptr;
    R.RefKind = 0;
    return R;
  }

};

} // end namespace llvm

#endif
