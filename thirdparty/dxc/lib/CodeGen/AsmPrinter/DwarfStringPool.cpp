//===-- llvm/CodeGen/DwarfStringPool.cpp - Dwarf Debug Framework ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DwarfStringPool.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;

DwarfStringPool::DwarfStringPool(BumpPtrAllocator &A, AsmPrinter &Asm,
                                 StringRef Prefix)
    : Pool(A), Prefix(Prefix),
      ShouldCreateSymbols(Asm.MAI->doesDwarfUseRelocationsAcrossSections()) {}

DwarfStringPool::EntryRef DwarfStringPool::getEntry(AsmPrinter &Asm,
                                                    StringRef Str) {
  auto I = Pool.insert(std::make_pair(Str, EntryTy()));
  if (I.second) {
    auto &Entry = I.first->second;
    Entry.Index = Pool.size() - 1;
    Entry.Offset = NumBytes;
    Entry.Symbol = ShouldCreateSymbols ? Asm.createTempSymbol(Prefix) : nullptr;

    NumBytes += Str.size() + 1;
    assert(NumBytes > Entry.Offset && "Unexpected overflow");
  }
  return EntryRef(*I.first);
}

void DwarfStringPool::emit(AsmPrinter &Asm, MCSection *StrSection,
                           MCSection *OffsetSection) {
  if (Pool.empty())
    return;

  // Start the dwarf str section.
  Asm.OutStreamer->SwitchSection(StrSection);

  // Get all of the string pool entries and put them in an array by their ID so
  // we can sort them.
  SmallVector<const StringMapEntry<EntryTy> *, 64> Entries(Pool.size());

  for (const auto &E : Pool)
    Entries[E.getValue().Index] = &E;

  for (const auto &Entry : Entries) {
    assert(ShouldCreateSymbols == static_cast<bool>(Entry->getValue().Symbol) &&
           "Mismatch between setting and entry");

    // Emit a label for reference from debug information entries.
    if (ShouldCreateSymbols)
      Asm.OutStreamer->EmitLabel(Entry->getValue().Symbol);

    // Emit the string itself with a terminating null byte.
    Asm.OutStreamer->AddComment("string offset=" +
                                Twine(Entry->getValue().Offset));
    Asm.OutStreamer->EmitBytes(
        StringRef(Entry->getKeyData(), Entry->getKeyLength() + 1));
  }

  // If we've got an offset section go ahead and emit that now as well.
  if (OffsetSection) {
    Asm.OutStreamer->SwitchSection(OffsetSection);
    unsigned size = 4; // FIXME: DWARF64 is 8.
    for (const auto &Entry : Entries)
      Asm.OutStreamer->EmitIntValue(Entry->getValue().Offset, size);
  }
}
