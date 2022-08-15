//===-- MCELFObjectTargetWriter.cpp - ELF Target Writer Subclass ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"

using namespace llvm;

MCELFObjectTargetWriter::MCELFObjectTargetWriter(bool Is64Bit_,
                                                 uint8_t OSABI_,
                                                 uint16_t EMachine_,
                                                 bool HasRelocationAddend_,
                                                 bool IsN64_)
  : OSABI(OSABI_), EMachine(EMachine_),
    HasRelocationAddend(HasRelocationAddend_), Is64Bit(Is64Bit_),
    IsN64(IsN64_){
}

bool MCELFObjectTargetWriter::needsRelocateWithSymbol(const MCSymbol &Sym,
                                                      unsigned Type) const {
  return false;
}

// ELF doesn't require relocations to be in any order. We sort by the Offset,
// just to match gnu as for easier comparison. The use type is an arbitrary way
// of making the sort deterministic.
// HLSL Change: changed calling convention to __cdecl
static int __cdecl cmpRel(const ELFRelocationEntry *AP, const ELFRelocationEntry *BP) {
  const ELFRelocationEntry &A = *AP;
  const ELFRelocationEntry &B = *BP;
  if (A.Offset != B.Offset)
    return B.Offset - A.Offset;
  if (B.Type != A.Type)
    return A.Type - B.Type;
  //llvm_unreachable("ELFRelocs might be unstable!");
  return 0;
}


void
MCELFObjectTargetWriter::sortRelocs(const MCAssembler &Asm,
                                    std::vector<ELFRelocationEntry> &Relocs) {
  array_pod_sort(Relocs.begin(), Relocs.end(), cmpRel);
}
