//===-- llvm/MC/MCELFObjectWriter.h - ELF Object Writer ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCELFOBJECTWRITER_H
#define LLVM_MC_MCELFOBJECTWRITER_H

#include "llvm/ADT/Triple.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ELF.h"
#include <vector>

namespace llvm {
class MCAssembler;
class MCFixup;
class MCFragment;
class MCObjectWriter;
class MCSymbol;
class MCSymbolELF;
class MCValue;
class raw_pwrite_stream;

struct ELFRelocationEntry {
  uint64_t Offset; // Where is the relocation.
  const MCSymbolELF *Symbol; // The symbol to relocate with.
  unsigned Type;   // The type of the relocation.
  uint64_t Addend; // The addend to use.

  ELFRelocationEntry(uint64_t Offset, const MCSymbolELF *Symbol, unsigned Type,
                     uint64_t Addend)
      : Offset(Offset), Symbol(Symbol), Type(Type), Addend(Addend) {}
};

class MCELFObjectTargetWriter {
  const uint8_t OSABI;
  const uint16_t EMachine;
  const unsigned HasRelocationAddend : 1;
  const unsigned Is64Bit : 1;
  const unsigned IsN64 : 1;

protected:

  MCELFObjectTargetWriter(bool Is64Bit_, uint8_t OSABI_,
                          uint16_t EMachine_,  bool HasRelocationAddend,
                          bool IsN64=false);

public:
  static uint8_t getOSABI(Triple::OSType OSType) {
    switch (OSType) {
      case Triple::CloudABI:
        return ELF::ELFOSABI_CLOUDABI;
      case Triple::PS4:
      case Triple::FreeBSD:
        return ELF::ELFOSABI_FREEBSD;
      case Triple::Linux:
        return ELF::ELFOSABI_LINUX;
      default:
        return ELF::ELFOSABI_NONE;
    }
  }

  virtual ~MCELFObjectTargetWriter() {}

  virtual unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                                bool IsPCRel) const = 0;

  virtual bool needsRelocateWithSymbol(const MCSymbol &Sym,
                                       unsigned Type) const;

  virtual void sortRelocs(const MCAssembler &Asm,
                          std::vector<ELFRelocationEntry> &Relocs);

  /// \name Accessors
  /// @{
  uint8_t getOSABI() const { return OSABI; }
  uint16_t getEMachine() const { return EMachine; }
  bool hasRelocationAddend() const { return HasRelocationAddend; }
  bool is64Bit() const { return Is64Bit; }
  bool isN64() const { return IsN64; }
  /// @}

  // Instead of changing everyone's API we pack the N64 Type fields
  // into the existing 32 bit data unsigned.
#define R_TYPE_SHIFT 0
#define R_TYPE_MASK 0xffffff00
#define R_TYPE2_SHIFT 8
#define R_TYPE2_MASK 0xffff00ff
#define R_TYPE3_SHIFT 16
#define R_TYPE3_MASK 0xff00ffff
#define R_SSYM_SHIFT 24
#define R_SSYM_MASK 0x00ffffff

  // N64 relocation type accessors
  uint8_t getRType(uint32_t Type) const {
    return (unsigned)((Type >> R_TYPE_SHIFT) & 0xff);
  }
  uint8_t getRType2(uint32_t Type) const {
    return (unsigned)((Type >> R_TYPE2_SHIFT) & 0xff);
  }
  uint8_t getRType3(uint32_t Type) const {
    return (unsigned)((Type >> R_TYPE3_SHIFT) & 0xff);
  }
  uint8_t getRSsym(uint32_t Type) const {
    return (unsigned)((Type >> R_SSYM_SHIFT) & 0xff);
  }

  // N64 relocation type setting
  unsigned setRType(unsigned Value, unsigned Type) const {
    return ((Type & R_TYPE_MASK) | ((Value & 0xff) << R_TYPE_SHIFT));
  }
  unsigned setRType2(unsigned Value, unsigned Type) const {
    return (Type & R_TYPE2_MASK) | ((Value & 0xff) << R_TYPE2_SHIFT);
  }
  unsigned setRType3(unsigned Value, unsigned Type) const {
    return (Type & R_TYPE3_MASK) | ((Value & 0xff) << R_TYPE3_SHIFT);
  }
  unsigned setRSsym(unsigned Value, unsigned Type) const {
    return (Type & R_SSYM_MASK) | ((Value & 0xff) << R_SSYM_SHIFT);
  }
};

/// \brief Construct a new ELF writer instance.
///
/// \param MOTW - The target specific ELF writer subclass.
/// \param OS - The stream to write to.
/// \returns The constructed object writer.
MCObjectWriter *createELFObjectWriter(MCELFObjectTargetWriter *MOTW,
                                      raw_pwrite_stream &OS,
                                      bool IsLittleEndian);
} // End llvm namespace

#endif
