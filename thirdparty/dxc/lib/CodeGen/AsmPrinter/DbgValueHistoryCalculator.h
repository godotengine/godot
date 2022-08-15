//===-- llvm/CodeGen/AsmPrinter/DbgValueHistoryCalculator.h ----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DBGVALUEHISTORYCALCULATOR_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DBGVALUEHISTORYCALCULATOR_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

class MachineFunction;
class MachineInstr;
class DILocalVariable;
class DILocation;
class TargetRegisterInfo;

// For each user variable, keep a list of instruction ranges where this variable
// is accessible. The variables are listed in order of appearance.
class DbgValueHistoryMap {
  // Each instruction range starts with a DBG_VALUE instruction, specifying the
  // location of a variable, which is assumed to be valid until the end of the
  // range. If end is not specified, location is valid until the start
  // instruction of the next instruction range, or until the end of the
  // function.
public:
  typedef std::pair<const MachineInstr *, const MachineInstr *> InstrRange;
  typedef SmallVector<InstrRange, 4> InstrRanges;
  typedef std::pair<const DILocalVariable *, const DILocation *>
      InlinedVariable;
  typedef MapVector<InlinedVariable, InstrRanges> InstrRangesMap;

private:
  InstrRangesMap VarInstrRanges;

public:
  void startInstrRange(InlinedVariable Var, const MachineInstr &MI);
  void endInstrRange(InlinedVariable Var, const MachineInstr &MI);
  // Returns register currently describing @Var. If @Var is currently
  // unaccessible or is not described by a register, returns 0.
  unsigned getRegisterForVar(InlinedVariable Var) const;

  bool empty() const { return VarInstrRanges.empty(); }
  void clear() { VarInstrRanges.clear(); }
  InstrRangesMap::const_iterator begin() const { return VarInstrRanges.begin(); }
  InstrRangesMap::const_iterator end() const { return VarInstrRanges.end(); }
};

void calculateDbgValueHistory(const MachineFunction *MF,
                              const TargetRegisterInfo *TRI,
                              DbgValueHistoryMap &Result);
}

#endif
