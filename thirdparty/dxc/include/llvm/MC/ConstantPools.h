//===- ConstantPool.h - Keep track of assembler-generated  ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ConstantPool and AssemblerConstantPools classes.
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_MC_CONSTANTPOOLS_H
#define LLVM_MC_CONSTANTPOOLS_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
class MCContext;
class MCExpr;
class MCSection;
class MCStreamer;
class MCSymbol;

struct ConstantPoolEntry {
  ConstantPoolEntry(MCSymbol *L, const MCExpr *Val, unsigned Sz)
    : Label(L), Value(Val), Size(Sz) {}
  MCSymbol *Label;
  const MCExpr *Value;
  unsigned Size;
};

// A class to keep track of assembler-generated constant pools that are use to
// implement the ldr-pseudo.
class ConstantPool {
  typedef SmallVector<ConstantPoolEntry, 4> EntryVecTy;
  EntryVecTy Entries;

public:
  // Initialize a new empty constant pool
  ConstantPool() {}

  // Add a new entry to the constant pool in the next slot.
  // \param Value is the new entry to put in the constant pool.
  // \param Size is the size in bytes of the entry
  //
  // \returns a MCExpr that references the newly inserted value
  const MCExpr *addEntry(const MCExpr *Value, MCContext &Context,
                         unsigned Size);

  // Emit the contents of the constant pool using the provided streamer.
  void emitEntries(MCStreamer &Streamer);

  // Return true if the constant pool is empty
  bool empty();
};

class AssemblerConstantPools {
  // Map type used to keep track of per-Section constant pools used by the
  // ldr-pseudo opcode. The map associates a section to its constant pool. The
  // constant pool is a vector of (label, value) pairs. When the ldr
  // pseudo is parsed we insert a new (label, value) pair into the constant pool
  // for the current section and add MCSymbolRefExpr to the new label as
  // an opcode to the ldr. After we have parsed all the user input we
  // output the (label, value) pairs in each constant pool at the end of the
  // section.
  //
  // We use the MapVector for the map type to ensure stable iteration of
  // the sections at the end of the parse. We need to iterate over the
  // sections in a stable order to ensure that we have print the
  // constant pools in a deterministic order when printing an assembly
  // file.
  typedef MapVector<MCSection *, ConstantPool> ConstantPoolMapTy;
  ConstantPoolMapTy ConstantPools;

public:
  void emitAll(MCStreamer &Streamer);
  void emitForCurrentSection(MCStreamer &Streamer);
  const MCExpr *addEntry(MCStreamer &Streamer, const MCExpr *Expr,
                         unsigned Size);

private:
  ConstantPool *getConstantPool(MCSection *Section);
  ConstantPool &getOrCreateConstantPool(MCSection *Section);
};
} // end namespace llvm

#endif
