//==- DIAEnumSourceFiles.h - DIA Source File Enumerator impl -----*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_DIA_DIAENUMSOURCEFILES_H
#define LLVM_DEBUGINFO_PDB_DIA_DIAENUMSOURCEFILES_H

#include "DIASupport.h"
#include "llvm/DebugInfo/PDB/IPDBEnumChildren.h"

namespace llvm {

class DIASession;

class DIAEnumSourceFiles : public IPDBEnumChildren<IPDBSourceFile> {
public:
  explicit DIAEnumSourceFiles(const DIASession &PDBSession,
                              CComPtr<IDiaEnumSourceFiles> DiaEnumerator);

  uint32_t getChildCount() const override;
  ChildTypePtr getChildAtIndex(uint32_t Index) const override;
  ChildTypePtr getNext() override;
  void reset() override;
  DIAEnumSourceFiles *clone() const override;

private:
  const DIASession &Session;
  CComPtr<IDiaEnumSourceFiles> Enumerator;
};
}

#endif
