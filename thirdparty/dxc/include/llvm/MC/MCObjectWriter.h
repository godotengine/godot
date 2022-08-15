//===-- llvm/MC/MCObjectWriter.h - Object File Writer Interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCOBJECTWRITER_H
#define LLVM_MC_MCOBJECTWRITER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

namespace llvm {
class MCAsmLayout;
class MCAssembler;
class MCFixup;
class MCFragment;
class MCSymbolRefExpr;
class MCValue;

/// Defines the object file and target independent interfaces used by the
/// assembler backend to write native file format object files.
///
/// The object writer contains a few callbacks used by the assembler to allow
/// the object writer to modify the assembler data structures at appropriate
/// points. Once assembly is complete, the object writer is given the
/// MCAssembler instance, which contains all the symbol and section data which
/// should be emitted as part of writeObject().
///
/// The object writer also contains a number of helper methods for writing
/// binary data to the output stream.
class MCObjectWriter {
  MCObjectWriter(const MCObjectWriter &) = delete;
  void operator=(const MCObjectWriter &) = delete;

protected:
  raw_pwrite_stream &OS;

  unsigned IsLittleEndian : 1;

protected: // Can only create subclasses.
  MCObjectWriter(raw_pwrite_stream &OS, bool IsLittleEndian)
      : OS(OS), IsLittleEndian(IsLittleEndian) {}

public:
  virtual ~MCObjectWriter();

  /// lifetime management
  virtual void reset() {}

  bool isLittleEndian() const { return IsLittleEndian; }

  raw_ostream &getStream() { return OS; }

  /// \name High-Level API
  /// @{

  /// Perform any late binding of symbols (for example, to assign symbol
  /// indices for use when generating relocations).
  ///
  /// This routine is called by the assembler after layout and relaxation is
  /// complete.
  virtual void executePostLayoutBinding(MCAssembler &Asm,
                                        const MCAsmLayout &Layout) = 0;

  /// Record a relocation entry.
  ///
  /// This routine is called by the assembler after layout and relaxation, and
  /// post layout binding. The implementation is responsible for storing
  /// information about the relocation so that it can be emitted during
  /// writeObject().
  virtual void recordRelocation(MCAssembler &Asm, const MCAsmLayout &Layout,
                                const MCFragment *Fragment,
                                const MCFixup &Fixup, MCValue Target,
                                bool &IsPCRel, uint64_t &FixedValue) = 0;

  /// Check whether the difference (A - B) between two symbol references is
  /// fully resolved.
  ///
  /// Clients are not required to answer precisely and may conservatively return
  /// false, even when a difference is fully resolved.
  bool isSymbolRefDifferenceFullyResolved(const MCAssembler &Asm,
                                          const MCSymbolRefExpr *A,
                                          const MCSymbolRefExpr *B,
                                          bool InSet) const;

  virtual bool isSymbolRefDifferenceFullyResolvedImpl(const MCAssembler &Asm,
                                                      const MCSymbol &SymA,
                                                      const MCFragment &FB,
                                                      bool InSet,
                                                      bool IsPCRel) const;

  /// True if this symbol (which is a variable) is weak. This is not
  /// just STB_WEAK, but more generally whether or not we can evaluate
  /// past it.
  virtual bool isWeak(const MCSymbol &Sym) const;

  /// Write the object file.
  ///
  /// This routine is called by the assembler after layout and relaxation is
  /// complete, fixups have been evaluated and applied, and relocations
  /// generated.
  virtual void writeObject(MCAssembler &Asm, const MCAsmLayout &Layout) = 0;

  /// @}
  /// \name Binary Output
  /// @{

  void write8(uint8_t Value) { OS << char(Value); }

  void writeLE16(uint16_t Value) {
    support::endian::Writer<support::little>(OS).write(Value);
  }

  void writeLE32(uint32_t Value) {
    support::endian::Writer<support::little>(OS).write(Value);
  }

  void writeLE64(uint64_t Value) {
    support::endian::Writer<support::little>(OS).write(Value);
  }

  void writeBE16(uint16_t Value) {
    support::endian::Writer<support::big>(OS).write(Value);
  }

  void writeBE32(uint32_t Value) {
    support::endian::Writer<support::big>(OS).write(Value);
  }

  void writeBE64(uint64_t Value) {
    support::endian::Writer<support::big>(OS).write(Value);
  }

  void write16(uint16_t Value) {
    if (IsLittleEndian)
      writeLE16(Value);
    else
      writeBE16(Value);
  }

  void write32(uint32_t Value) {
    if (IsLittleEndian)
      writeLE32(Value);
    else
      writeBE32(Value);
  }

  void write64(uint64_t Value) {
    if (IsLittleEndian)
      writeLE64(Value);
    else
      writeBE64(Value);
  }

  void WriteZeros(unsigned N) {
    const char Zeros[16] = {0};

    for (unsigned i = 0, e = N / 16; i != e; ++i)
      OS << StringRef(Zeros, 16);

    OS << StringRef(Zeros, N % 16);
  }

  void writeBytes(const SmallVectorImpl<char> &ByteVec,
                  unsigned ZeroFillSize = 0) {
    writeBytes(StringRef(ByteVec.data(), ByteVec.size()), ZeroFillSize);
  }

  void writeBytes(StringRef Str, unsigned ZeroFillSize = 0) {
    // TODO: this version may need to go away once all fragment contents are
    // converted to SmallVector<char, N>
    assert(
        (ZeroFillSize == 0 || Str.size() <= ZeroFillSize) &&
        "data size greater than fill size, unexpected large write will occur");
    OS << Str;
    if (ZeroFillSize)
      WriteZeros(ZeroFillSize - Str.size());
  }

  /// @}
};

} // End llvm namespace

#endif
