//===-- MCInstrAnalysis.cpp - InstrDesc target hooks ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCInstrAnalysis.h"
using namespace llvm;

bool MCInstrAnalysis::evaluateBranch(const MCInst &Inst, uint64_t Addr,
                                     uint64_t Size, uint64_t &Target) const {
  if (Inst.getNumOperands() == 0 ||
      Info->get(Inst.getOpcode()).OpInfo[0].OperandType != MCOI::OPERAND_PCREL)
    return false;

  int64_t Imm = Inst.getOperand(0).getImm();
  Target = Addr+Size+Imm;
  return true;
}
