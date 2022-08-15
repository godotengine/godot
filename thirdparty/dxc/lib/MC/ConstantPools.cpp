//===- ConstantPools.cpp - ConstantPool class --*- C++ -*---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ConstantPool and  AssemblerConstantPools classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/MapVector.h"
#include "llvm/MC/ConstantPools.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
using namespace llvm;
//
// ConstantPool implementation
//
// Emit the contents of the constant pool using the provided streamer.
void ConstantPool::emitEntries(MCStreamer &Streamer) {
  if (Entries.empty())
    return;
  Streamer.EmitDataRegion(MCDR_DataRegion);
  for (EntryVecTy::const_iterator I = Entries.begin(), E = Entries.end();
       I != E; ++I) {
    Streamer.EmitCodeAlignment(I->Size); // align naturally
    Streamer.EmitLabel(I->Label);
    Streamer.EmitValue(I->Value, I->Size);
  }
  Streamer.EmitDataRegion(MCDR_DataRegionEnd);
  Entries.clear();
}

const MCExpr *ConstantPool::addEntry(const MCExpr *Value, MCContext &Context,
                                     unsigned Size) {
  MCSymbol *CPEntryLabel = Context.createTempSymbol();

  Entries.push_back(ConstantPoolEntry(CPEntryLabel, Value, Size));
  return MCSymbolRefExpr::create(CPEntryLabel, Context);
}

bool ConstantPool::empty() { return Entries.empty(); }

//
// AssemblerConstantPools implementation
//
ConstantPool *AssemblerConstantPools::getConstantPool(MCSection *Section) {
  ConstantPoolMapTy::iterator CP = ConstantPools.find(Section);
  if (CP == ConstantPools.end())
    return nullptr;

  return &CP->second;
}

ConstantPool &
AssemblerConstantPools::getOrCreateConstantPool(MCSection *Section) {
  return ConstantPools[Section];
}

static void emitConstantPool(MCStreamer &Streamer, MCSection *Section,
                             ConstantPool &CP) {
  if (!CP.empty()) {
    Streamer.SwitchSection(Section);
    CP.emitEntries(Streamer);
  }
}

void AssemblerConstantPools::emitAll(MCStreamer &Streamer) {
  // Dump contents of assembler constant pools.
  for (ConstantPoolMapTy::iterator CPI = ConstantPools.begin(),
                                   CPE = ConstantPools.end();
       CPI != CPE; ++CPI) {
    MCSection *Section = CPI->first;
    ConstantPool &CP = CPI->second;

    emitConstantPool(Streamer, Section, CP);
  }
}

void AssemblerConstantPools::emitForCurrentSection(MCStreamer &Streamer) {
  MCSection *Section = Streamer.getCurrentSection().first;
  if (ConstantPool *CP = getConstantPool(Section)) {
    emitConstantPool(Streamer, Section, *CP);
  }
}

const MCExpr *AssemblerConstantPools::addEntry(MCStreamer &Streamer,
                                               const MCExpr *Expr,
                                               unsigned Size) {
  MCSection *Section = Streamer.getCurrentSection().first;
  return getOrCreateConstantPool(Section).addEntry(Expr, Streamer.getContext(),
                                                   Size);
}
