//===-- RecordStreamer.cpp - Record asm definde and used symbols ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RecordStreamer.h"
#include "llvm/MC/MCSymbol.h"
using namespace llvm;

void RecordStreamer::markDefined(const MCSymbol &Symbol) {
  State &S = Symbols[Symbol.getName()];
  switch (S) {
  case DefinedGlobal:
  case Global:
    S = DefinedGlobal;
    break;
  case NeverSeen:
  case Defined:
  case Used:
    S = Defined;
    break;
  }
}

void RecordStreamer::markGlobal(const MCSymbol &Symbol) {
  State &S = Symbols[Symbol.getName()];
  switch (S) {
  case DefinedGlobal:
  case Defined:
    S = DefinedGlobal;
    break;

  case NeverSeen:
  case Global:
  case Used:
    S = Global;
    break;
  }
}

void RecordStreamer::markUsed(const MCSymbol &Symbol) {
  State &S = Symbols[Symbol.getName()];
  switch (S) {
  case DefinedGlobal:
  case Defined:
  case Global:
    break;

  case NeverSeen:
  case Used:
    S = Used;
    break;
  }
}

void RecordStreamer::visitUsedSymbol(const MCSymbol &Sym) { markUsed(Sym); }

RecordStreamer::const_iterator RecordStreamer::begin() {
  return Symbols.begin();
}

RecordStreamer::const_iterator RecordStreamer::end() { return Symbols.end(); }

RecordStreamer::RecordStreamer(MCContext &Context) : MCStreamer(Context) {}

void RecordStreamer::EmitInstruction(const MCInst &Inst,
                                     const MCSubtargetInfo &STI) {
  MCStreamer::EmitInstruction(Inst, STI);
}

void RecordStreamer::EmitLabel(MCSymbol *Symbol) {
  MCStreamer::EmitLabel(Symbol);
  markDefined(*Symbol);
}

void RecordStreamer::EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) {
  markDefined(*Symbol);
  MCStreamer::EmitAssignment(Symbol, Value);
}

bool RecordStreamer::EmitSymbolAttribute(MCSymbol *Symbol,
                                         MCSymbolAttr Attribute) {
  if (Attribute == MCSA_Global)
    markGlobal(*Symbol);
  return true;
}

void RecordStreamer::EmitZerofill(MCSection *Section, MCSymbol *Symbol,
                                  uint64_t Size, unsigned ByteAlignment) {
  markDefined(*Symbol);
}

void RecordStreamer::EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                      unsigned ByteAlignment) {
  markDefined(*Symbol);
}
