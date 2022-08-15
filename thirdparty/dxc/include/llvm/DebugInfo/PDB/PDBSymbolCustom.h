//===- PDBSymbolCustom.h - compiler-specific types --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBSYMBOLCUSTOM_H
#define LLVM_DEBUGINFO_PDB_PDBSYMBOLCUSTOM_H

#include "PDBSymbol.h"
#include "PDBTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

class raw_ostream;

/// PDBSymbolCustom represents symbols that are compiler-specific and do not
/// fit anywhere else in the lexical hierarchy.
/// https://msdn.microsoft.com/en-us/library/d88sf09h.aspx
class PDBSymbolCustom : public PDBSymbol {
public:
  PDBSymbolCustom(const IPDBSession &PDBSession,
                  std::unique_ptr<IPDBRawSymbol> CustomSymbol);

  DECLARE_PDB_SYMBOL_CONCRETE_TYPE(PDB_SymType::Custom)

  void dump(PDBSymDumper &Dumper) const override;

  void getDataBytes(llvm::SmallVector<uint8_t, 32> &bytes);
  FORWARD_SYMBOL_METHOD(getSymIndexId)
};

} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_PDBSYMBOLCUSTOM_H
