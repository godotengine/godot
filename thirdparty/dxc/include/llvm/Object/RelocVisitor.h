//===-- RelocVisitor.h - Visitor for object file relocations -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides a wrapper around all the different types of relocations
// in different file formats, such that a client can handle them in a unified
// manner by only implementing a minimal number of functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_RELOCVISITOR_H
#define LLVM_OBJECT_RELOCVISITOR_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace object {

struct RelocToApply {
  // The computed value after applying the relevant relocations.
  int64_t Value;

  // The width of the value; how many bytes to touch when applying the
  // relocation.
  char Width;
  RelocToApply(int64_t Value, char Width) : Value(Value), Width(Width) {}
  RelocToApply() : Value(0), Width(0) {}
};

/// @brief Base class for object file relocation visitors.
class RelocVisitor {
public:
  explicit RelocVisitor(const ObjectFile &Obj)
    : ObjToVisit(Obj), HasError(false) {}

  // TODO: Should handle multiple applied relocations via either passing in the
  // previously computed value or just count paired relocations as a single
  // visit.
  RelocToApply visit(uint32_t RelocType, RelocationRef R, uint64_t Value = 0) {
    if (isa<ELFObjectFileBase>(ObjToVisit))
      return visitELF(RelocType, R, Value);
    if (isa<COFFObjectFile>(ObjToVisit))
      return visitCOFF(RelocType, R, Value);
    if (isa<MachOObjectFile>(ObjToVisit))
      return visitMachO(RelocType, R, Value);

    HasError = true;
    return RelocToApply();
  }

  bool error() { return HasError; }

private:
  const ObjectFile &ObjToVisit;
  bool HasError;

  RelocToApply visitELF(uint32_t RelocType, RelocationRef R, uint64_t Value) {
    if (ObjToVisit.getBytesInAddress() == 8) { // 64-bit object file
      switch (ObjToVisit.getArch()) {
      case Triple::x86_64:
        switch (RelocType) {
        case llvm::ELF::R_X86_64_NONE:
          return visitELF_X86_64_NONE(R);
        case llvm::ELF::R_X86_64_64:
          return visitELF_X86_64_64(R, Value);
        case llvm::ELF::R_X86_64_PC32:
          return visitELF_X86_64_PC32(R, Value);
        case llvm::ELF::R_X86_64_32:
          return visitELF_X86_64_32(R, Value);
        case llvm::ELF::R_X86_64_32S:
          return visitELF_X86_64_32S(R, Value);
        default:
          HasError = true;
          return RelocToApply();
        }
      case Triple::aarch64:
        switch (RelocType) {
        case llvm::ELF::R_AARCH64_ABS32:
          return visitELF_AARCH64_ABS32(R, Value);
        case llvm::ELF::R_AARCH64_ABS64:
          return visitELF_AARCH64_ABS64(R, Value);
        default:
          HasError = true;
          return RelocToApply();
        }
      case Triple::mips64el:
      case Triple::mips64:
        switch (RelocType) {
        case llvm::ELF::R_MIPS_32:
          return visitELF_MIPS64_32(R, Value);
        case llvm::ELF::R_MIPS_64:
          return visitELF_MIPS64_64(R, Value);
        default:
          HasError = true;
          return RelocToApply();
        }
      case Triple::ppc64le:
      case Triple::ppc64:
        switch (RelocType) {
        case llvm::ELF::R_PPC64_ADDR32:
          return visitELF_PPC64_ADDR32(R, Value);
        case llvm::ELF::R_PPC64_ADDR64:
          return visitELF_PPC64_ADDR64(R, Value);
        default:
          HasError = true;
          return RelocToApply();
        }
      case Triple::systemz:
        switch (RelocType) {
        case llvm::ELF::R_390_32:
          return visitELF_390_32(R, Value);
        case llvm::ELF::R_390_64:
          return visitELF_390_64(R, Value);
        default:
          HasError = true;
          return RelocToApply();
        }
      case Triple::sparcv9:
        switch (RelocType) {
        case llvm::ELF::R_SPARC_32:
        case llvm::ELF::R_SPARC_UA32:
          return visitELF_SPARCV9_32(R, Value);
        case llvm::ELF::R_SPARC_64:
        case llvm::ELF::R_SPARC_UA64:
          return visitELF_SPARCV9_64(R, Value);
        default:
          HasError = true;
          return RelocToApply();
        }
      default:
        HasError = true;
        return RelocToApply();
      }
    } else if (ObjToVisit.getBytesInAddress() == 4) { // 32-bit object file
      switch (ObjToVisit.getArch()) {
      case Triple::x86:
        switch (RelocType) {
        case llvm::ELF::R_386_NONE:
          return visitELF_386_NONE(R);
        case llvm::ELF::R_386_32:
          return visitELF_386_32(R, Value);
        case llvm::ELF::R_386_PC32:
          return visitELF_386_PC32(R, Value);
        default:
          HasError = true;
          return RelocToApply();
        }
      case Triple::ppc:
        switch (RelocType) {
        case llvm::ELF::R_PPC_ADDR32:
          return visitELF_PPC_ADDR32(R, Value);
        default:
          HasError = true;
          return RelocToApply();
        }
      case Triple::arm:
      case Triple::armeb:
        switch (RelocType) {
        default:
          HasError = true;
          return RelocToApply();
        case llvm::ELF::R_ARM_ABS32:
          return visitELF_ARM_ABS32(R, Value);
        }
      case Triple::mipsel:
      case Triple::mips:
        switch (RelocType) {
        case llvm::ELF::R_MIPS_32:
          return visitELF_MIPS_32(R, Value);
        default:
          HasError = true;
          return RelocToApply();
        }
      case Triple::sparc:
        switch (RelocType) {
        case llvm::ELF::R_SPARC_32:
        case llvm::ELF::R_SPARC_UA32:
          return visitELF_SPARC_32(R, Value);
        default:
          HasError = true;
          return RelocToApply();
        }
      default:
        HasError = true;
        return RelocToApply();
      }
    } else {
      report_fatal_error("Invalid word size in object file");
    }
  }

  RelocToApply visitCOFF(uint32_t RelocType, RelocationRef R, uint64_t Value) {
    switch (ObjToVisit.getArch()) {
    case Triple::x86:
      switch (RelocType) {
      case COFF::IMAGE_REL_I386_SECREL:
        return visitCOFF_I386_SECREL(R, Value);
      case COFF::IMAGE_REL_I386_DIR32:
        return visitCOFF_I386_DIR32(R, Value);
      }
      break;
    case Triple::x86_64:
      switch (RelocType) {
      case COFF::IMAGE_REL_AMD64_SECREL:
        return visitCOFF_AMD64_SECREL(R, Value);
      case COFF::IMAGE_REL_AMD64_ADDR64:
        return visitCOFF_AMD64_ADDR64(R, Value);
      }
      break;
    }
    HasError = true;
    return RelocToApply();
  }

  RelocToApply visitMachO(uint32_t RelocType, RelocationRef R, uint64_t Value) {
    switch (ObjToVisit.getArch()) {
    default: break;
    case Triple::x86_64:
      switch (RelocType) {
        default: break;
        case MachO::X86_64_RELOC_UNSIGNED:
          return visitMACHO_X86_64_UNSIGNED(R, Value);
      }
    }
    HasError = true;
    return RelocToApply();
  }

  int64_t getELFAddend(RelocationRef R) {
    ErrorOr<int64_t> AddendOrErr = ELFRelocationRef(R).getAddend();
    if (std::error_code EC = AddendOrErr.getError())
      report_fatal_error(EC.message());
    return *AddendOrErr;
  }

  uint8_t getLengthMachO64(RelocationRef R) {
    const MachOObjectFile *Obj = cast<MachOObjectFile>(R.getObject());
    return Obj->getRelocationLength(R.getRawDataRefImpl());
  }

  /// Operations

  /// 386-ELF
  RelocToApply visitELF_386_NONE(RelocationRef R) {
    return RelocToApply(0, 0);
  }

  // Ideally the Addend here will be the addend in the data for
  // the relocation. It's not actually the case for Rel relocations.
  RelocToApply visitELF_386_32(RelocationRef R, uint64_t Value) {
    return RelocToApply(Value, 4);
  }

  RelocToApply visitELF_386_PC32(RelocationRef R, uint64_t Value) {
    uint64_t Address = R.getOffset();
    return RelocToApply(Value - Address, 4);
  }

  /// X86-64 ELF
  RelocToApply visitELF_X86_64_NONE(RelocationRef R) {
    return RelocToApply(0, 0);
  }
  RelocToApply visitELF_X86_64_64(RelocationRef R, uint64_t Value) {
    int64_t Addend = getELFAddend(R);
    return RelocToApply(Value + Addend, 8);
  }
  RelocToApply visitELF_X86_64_PC32(RelocationRef R, uint64_t Value) {
    int64_t Addend = getELFAddend(R);
    uint64_t Address = R.getOffset();
    return RelocToApply(Value + Addend - Address, 4);
  }
  RelocToApply visitELF_X86_64_32(RelocationRef R, uint64_t Value) {
    int64_t Addend = getELFAddend(R);
    uint32_t Res = (Value + Addend) & 0xFFFFFFFF;
    return RelocToApply(Res, 4);
  }
  RelocToApply visitELF_X86_64_32S(RelocationRef R, uint64_t Value) {
    int64_t Addend = getELFAddend(R);
    int32_t Res = (Value + Addend) & 0xFFFFFFFF;
    return RelocToApply(Res, 4);
  }

  /// PPC64 ELF
  RelocToApply visitELF_PPC64_ADDR32(RelocationRef R, uint64_t Value) {
    int64_t Addend = getELFAddend(R);
    uint32_t Res = (Value + Addend) & 0xFFFFFFFF;
    return RelocToApply(Res, 4);
  }
  RelocToApply visitELF_PPC64_ADDR64(RelocationRef R, uint64_t Value) {
    int64_t Addend = getELFAddend(R);
    return RelocToApply(Value + Addend, 8);
  }

  /// PPC32 ELF
  RelocToApply visitELF_PPC_ADDR32(RelocationRef R, uint64_t Value) {
    int64_t Addend = getELFAddend(R);
    uint32_t Res = (Value + Addend) & 0xFFFFFFFF;
    return RelocToApply(Res, 4);
  }

  /// MIPS ELF
  RelocToApply visitELF_MIPS_32(RelocationRef R, uint64_t Value) {
    uint32_t Res = Value & 0xFFFFFFFF;
    return RelocToApply(Res, 4);
  }

  /// MIPS64 ELF
  RelocToApply visitELF_MIPS64_32(RelocationRef R, uint64_t Value) {
    int64_t Addend = getELFAddend(R);
    uint32_t Res = (Value + Addend) & 0xFFFFFFFF;
    return RelocToApply(Res, 4);
  }

  RelocToApply visitELF_MIPS64_64(RelocationRef R, uint64_t Value) {
    int64_t Addend = getELFAddend(R);
    uint64_t Res = (Value + Addend);
    return RelocToApply(Res, 8);
  }

  // AArch64 ELF
  RelocToApply visitELF_AARCH64_ABS32(RelocationRef R, uint64_t Value) {
    int64_t Addend = getELFAddend(R);
    int64_t Res =  Value + Addend;

    // Overflow check allows for both signed and unsigned interpretation.
    if (Res < INT32_MIN || Res > UINT32_MAX)
      HasError = true;

    return RelocToApply(static_cast<uint32_t>(Res), 4);
  }

  RelocToApply visitELF_AARCH64_ABS64(RelocationRef R, uint64_t Value) {
    int64_t Addend = getELFAddend(R);
    return RelocToApply(Value + Addend, 8);
  }

  // SystemZ ELF
  RelocToApply visitELF_390_32(RelocationRef R, uint64_t Value) {
    int64_t Addend = getELFAddend(R);
    int64_t Res = Value + Addend;

    // Overflow check allows for both signed and unsigned interpretation.
    if (Res < INT32_MIN || Res > UINT32_MAX)
      HasError = true;

    return RelocToApply(static_cast<uint32_t>(Res), 4);
  }

  RelocToApply visitELF_390_64(RelocationRef R, uint64_t Value) {
    int64_t Addend = getELFAddend(R);
    return RelocToApply(Value + Addend, 8);
  }

  RelocToApply visitELF_SPARC_32(RelocationRef R, uint32_t Value) {
    int32_t Addend = getELFAddend(R);
    return RelocToApply(Value + Addend, 4);
  }

  RelocToApply visitELF_SPARCV9_32(RelocationRef R, uint64_t Value) {
    int32_t Addend = getELFAddend(R);
    return RelocToApply(Value + Addend, 4);
  }

  RelocToApply visitELF_SPARCV9_64(RelocationRef R, uint64_t Value) {
    int64_t Addend = getELFAddend(R);
    return RelocToApply(Value + Addend, 8);
  }

  RelocToApply visitELF_ARM_ABS32(RelocationRef R, uint64_t Value) {
    int64_t Res = Value;

    // Overflow check allows for both signed and unsigned interpretation.
    if (Res < INT32_MIN || Res > UINT32_MAX)
      HasError = true;

    return RelocToApply(static_cast<uint32_t>(Res), 4);
  }

  /// I386 COFF
  RelocToApply visitCOFF_I386_SECREL(RelocationRef R, uint64_t Value) {
    return RelocToApply(static_cast<uint32_t>(Value), /*Width=*/4);
  }

  RelocToApply visitCOFF_I386_DIR32(RelocationRef R, uint64_t Value) {
    return RelocToApply(static_cast<uint32_t>(Value), /*Width=*/4);
  }

  /// AMD64 COFF
  RelocToApply visitCOFF_AMD64_SECREL(RelocationRef R, uint64_t Value) {
    return RelocToApply(static_cast<uint32_t>(Value), /*Width=*/4);
  }

  RelocToApply visitCOFF_AMD64_ADDR64(RelocationRef R, uint64_t Value) {
    return RelocToApply(Value, /*Width=*/8);
  }

  // X86_64 MachO
  RelocToApply visitMACHO_X86_64_UNSIGNED(RelocationRef R, uint64_t Value) {
    uint8_t Length = getLengthMachO64(R);
    Length = 1<<Length;
    return RelocToApply(Value, Length);
  }
};

}
}
#endif
