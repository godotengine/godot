//===-- RuntimeDyldELF.cpp - Run-time dynamic linker for MC-JIT -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of ELF support for the MC-JIT runtime dynamic linker.
//
//===----------------------------------------------------------------------===//

#include "RuntimeDyldELF.h"
#include "RuntimeDyldCheckerImpl.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;
using namespace llvm::object;

#define DEBUG_TYPE "dyld"

static inline std::error_code check(std::error_code Err) {
  if (Err) {
    report_fatal_error(Err.message());
  }
  return Err;
}

namespace {

template <class ELFT> class DyldELFObject : public ELFObjectFile<ELFT> {
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)

  typedef Elf_Shdr_Impl<ELFT> Elf_Shdr;
  typedef Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef Elf_Rel_Impl<ELFT, false> Elf_Rel;
  typedef Elf_Rel_Impl<ELFT, true> Elf_Rela;

  typedef Elf_Ehdr_Impl<ELFT> Elf_Ehdr;

  typedef typename ELFDataTypeTypedefHelper<ELFT>::value_type addr_type;

public:
  DyldELFObject(MemoryBufferRef Wrapper, std::error_code &ec);

  void updateSectionAddress(const SectionRef &Sec, uint64_t Addr);

  void updateSymbolAddress(const SymbolRef &SymRef, uint64_t Addr);

  // Methods for type inquiry through isa, cast and dyn_cast
  static inline bool classof(const Binary *v) {
    return (isa<ELFObjectFile<ELFT>>(v) &&
            classof(cast<ELFObjectFile<ELFT>>(v)));
  }
  static inline bool classof(const ELFObjectFile<ELFT> *v) {
    return v->isDyldType();
  }

};



// The MemoryBuffer passed into this constructor is just a wrapper around the
// actual memory.  Ultimately, the Binary parent class will take ownership of
// this MemoryBuffer object but not the underlying memory.
template <class ELFT>
DyldELFObject<ELFT>::DyldELFObject(MemoryBufferRef Wrapper, std::error_code &EC)
    : ELFObjectFile<ELFT>(Wrapper, EC) {
  this->isDyldELFObject = true;
}

template <class ELFT>
void DyldELFObject<ELFT>::updateSectionAddress(const SectionRef &Sec,
                                               uint64_t Addr) {
  DataRefImpl ShdrRef = Sec.getRawDataRefImpl();
  Elf_Shdr *shdr =
      const_cast<Elf_Shdr *>(reinterpret_cast<const Elf_Shdr *>(ShdrRef.p));

  // This assumes the address passed in matches the target address bitness
  // The template-based type cast handles everything else.
  shdr->sh_addr = static_cast<addr_type>(Addr);
}

template <class ELFT>
void DyldELFObject<ELFT>::updateSymbolAddress(const SymbolRef &SymRef,
                                              uint64_t Addr) {

  Elf_Sym *sym = const_cast<Elf_Sym *>(
      ELFObjectFile<ELFT>::getSymbol(SymRef.getRawDataRefImpl()));

  // This assumes the address passed in matches the target address bitness
  // The template-based type cast handles everything else.
  sym->st_value = static_cast<addr_type>(Addr);
}

class LoadedELFObjectInfo
    : public RuntimeDyld::LoadedObjectInfoHelper<LoadedELFObjectInfo> {
public:
  LoadedELFObjectInfo(RuntimeDyldImpl &RTDyld, unsigned BeginIdx,
                      unsigned EndIdx)
      : LoadedObjectInfoHelper(RTDyld, BeginIdx, EndIdx) {}

  OwningBinary<ObjectFile>
  getObjectForDebug(const ObjectFile &Obj) const override;
};

template <typename ELFT>
std::unique_ptr<DyldELFObject<ELFT>>
createRTDyldELFObject(MemoryBufferRef Buffer,
                      const LoadedELFObjectInfo &L,
                      std::error_code &ec) {
  typedef typename ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename ELFDataTypeTypedefHelper<ELFT>::value_type addr_type;

  std::unique_ptr<DyldELFObject<ELFT>> Obj =
    llvm::make_unique<DyldELFObject<ELFT>>(Buffer, ec);

  // Iterate over all sections in the object.
  for (const auto &Sec : Obj->sections()) {
    StringRef SectionName;
    Sec.getName(SectionName);
    if (SectionName != "") {
      DataRefImpl ShdrRef = Sec.getRawDataRefImpl();
      Elf_Shdr *shdr = const_cast<Elf_Shdr *>(
          reinterpret_cast<const Elf_Shdr *>(ShdrRef.p));

      if (uint64_t SecLoadAddr = L.getSectionLoadAddress(SectionName)) {
        // This assumes that the address passed in matches the target address
        // bitness. The template-based type cast handles everything else.
        shdr->sh_addr = static_cast<addr_type>(SecLoadAddr);
      }
    }
  }

  return Obj;
}

OwningBinary<ObjectFile> createELFDebugObject(const ObjectFile &Obj,
                                              const LoadedELFObjectInfo &L) {
  assert(Obj.isELF() && "Not an ELF object file.");

  std::unique_ptr<MemoryBuffer> Buffer =
    MemoryBuffer::getMemBufferCopy(Obj.getData(), Obj.getFileName());

  std::error_code ec;

  std::unique_ptr<ObjectFile> DebugObj;
  if (Obj.getBytesInAddress() == 4 && Obj.isLittleEndian()) {
    typedef ELFType<support::little, false> ELF32LE;
    DebugObj = createRTDyldELFObject<ELF32LE>(Buffer->getMemBufferRef(), L, ec);
  } else if (Obj.getBytesInAddress() == 4 && !Obj.isLittleEndian()) {
    typedef ELFType<support::big, false> ELF32BE;
    DebugObj = createRTDyldELFObject<ELF32BE>(Buffer->getMemBufferRef(), L, ec);
  } else if (Obj.getBytesInAddress() == 8 && !Obj.isLittleEndian()) {
    typedef ELFType<support::big, true> ELF64BE;
    DebugObj = createRTDyldELFObject<ELF64BE>(Buffer->getMemBufferRef(), L, ec);
  } else if (Obj.getBytesInAddress() == 8 && Obj.isLittleEndian()) {
    typedef ELFType<support::little, true> ELF64LE;
    DebugObj = createRTDyldELFObject<ELF64LE>(Buffer->getMemBufferRef(), L, ec);
  } else
    llvm_unreachable("Unexpected ELF format");

  assert(!ec && "Could not construct copy ELF object file");

  return OwningBinary<ObjectFile>(std::move(DebugObj), std::move(Buffer));
}

OwningBinary<ObjectFile>
LoadedELFObjectInfo::getObjectForDebug(const ObjectFile &Obj) const {
  return createELFDebugObject(Obj, *this);
}

} // namespace

namespace llvm {

RuntimeDyldELF::RuntimeDyldELF(RuntimeDyld::MemoryManager &MemMgr,
                               RuntimeDyld::SymbolResolver &Resolver)
    : RuntimeDyldImpl(MemMgr, Resolver), GOTSectionID(0), CurrentGOTIndex(0) {}
RuntimeDyldELF::~RuntimeDyldELF() {}

void RuntimeDyldELF::registerEHFrames() {
  for (int i = 0, e = UnregisteredEHFrameSections.size(); i != e; ++i) {
    SID EHFrameSID = UnregisteredEHFrameSections[i];
    uint8_t *EHFrameAddr = Sections[EHFrameSID].Address;
    uint64_t EHFrameLoadAddr = Sections[EHFrameSID].LoadAddress;
    size_t EHFrameSize = Sections[EHFrameSID].Size;
    MemMgr.registerEHFrames(EHFrameAddr, EHFrameLoadAddr, EHFrameSize);
    RegisteredEHFrameSections.push_back(EHFrameSID);
  }
  UnregisteredEHFrameSections.clear();
}

void RuntimeDyldELF::deregisterEHFrames() {
  for (int i = 0, e = RegisteredEHFrameSections.size(); i != e; ++i) {
    SID EHFrameSID = RegisteredEHFrameSections[i];
    uint8_t *EHFrameAddr = Sections[EHFrameSID].Address;
    uint64_t EHFrameLoadAddr = Sections[EHFrameSID].LoadAddress;
    size_t EHFrameSize = Sections[EHFrameSID].Size;
    MemMgr.deregisterEHFrames(EHFrameAddr, EHFrameLoadAddr, EHFrameSize);
  }
  RegisteredEHFrameSections.clear();
}

std::unique_ptr<RuntimeDyld::LoadedObjectInfo>
RuntimeDyldELF::loadObject(const object::ObjectFile &O) {
  unsigned SectionStartIdx, SectionEndIdx;
  std::tie(SectionStartIdx, SectionEndIdx) = loadObjectImpl(O);
  return llvm::make_unique<LoadedELFObjectInfo>(*this, SectionStartIdx,
                                                SectionEndIdx);
}

void RuntimeDyldELF::resolveX86_64Relocation(const SectionEntry &Section,
                                             uint64_t Offset, uint64_t Value,
                                             uint32_t Type, int64_t Addend,
                                             uint64_t SymOffset) {
  switch (Type) {
  default:
    llvm_unreachable("Relocation type not implemented yet!");
    break;
  case ELF::R_X86_64_64: {
    support::ulittle64_t::ref(Section.Address + Offset) = Value + Addend;
    DEBUG(dbgs() << "Writing " << format("%p", (Value + Addend)) << " at "
                 << format("%p\n", Section.Address + Offset));
    break;
  }
  case ELF::R_X86_64_32:
  case ELF::R_X86_64_32S: {
    Value += Addend;
    assert((Type == ELF::R_X86_64_32 && (Value <= UINT32_MAX)) ||
           (Type == ELF::R_X86_64_32S &&
            ((int64_t)Value <= INT32_MAX && (int64_t)Value >= INT32_MIN)));
    uint32_t TruncatedAddr = (Value & 0xFFFFFFFF);
    support::ulittle32_t::ref(Section.Address + Offset) = TruncatedAddr;
    DEBUG(dbgs() << "Writing " << format("%p", TruncatedAddr) << " at "
                 << format("%p\n", Section.Address + Offset));
    break;
  }
  case ELF::R_X86_64_PC32: {
    uint64_t FinalAddress = Section.LoadAddress + Offset;
    int64_t RealOffset = Value + Addend - FinalAddress;
    assert(isInt<32>(RealOffset));
    int32_t TruncOffset = (RealOffset & 0xFFFFFFFF);
    support::ulittle32_t::ref(Section.Address + Offset) = TruncOffset;
    break;
  }
  case ELF::R_X86_64_PC64: {
    uint64_t FinalAddress = Section.LoadAddress + Offset;
    int64_t RealOffset = Value + Addend - FinalAddress;
    support::ulittle64_t::ref(Section.Address + Offset) = RealOffset;
    break;
  }
  }
}

void RuntimeDyldELF::resolveX86Relocation(const SectionEntry &Section,
                                          uint64_t Offset, uint32_t Value,
                                          uint32_t Type, int32_t Addend) {
  switch (Type) {
  case ELF::R_386_32: {
    support::ulittle32_t::ref(Section.Address + Offset) = Value + Addend;
    break;
  }
  case ELF::R_386_PC32: {
    uint32_t FinalAddress = ((Section.LoadAddress + Offset) & 0xFFFFFFFF);
    uint32_t RealOffset = Value + Addend - FinalAddress;
    support::ulittle32_t::ref(Section.Address + Offset) = RealOffset;
    break;
  }
  default:
    // There are other relocation types, but it appears these are the
    // only ones currently used by the LLVM ELF object writer
    llvm_unreachable("Relocation type not implemented yet!");
    break;
  }
}

void RuntimeDyldELF::resolveAArch64Relocation(const SectionEntry &Section,
                                              uint64_t Offset, uint64_t Value,
                                              uint32_t Type, int64_t Addend) {
  uint32_t *TargetPtr = reinterpret_cast<uint32_t *>(Section.Address + Offset);
  uint64_t FinalAddress = Section.LoadAddress + Offset;

  DEBUG(dbgs() << "resolveAArch64Relocation, LocalAddress: 0x"
               << format("%llx", Section.Address + Offset)
               << " FinalAddress: 0x" << format("%llx", FinalAddress)
               << " Value: 0x" << format("%llx", Value) << " Type: 0x"
               << format("%x", Type) << " Addend: 0x" << format("%llx", Addend)
               << "\n");

  switch (Type) {
  default:
    llvm_unreachable("Relocation type not implemented yet!");
    break;
  case ELF::R_AARCH64_ABS64: {
    uint64_t *TargetPtr =
        reinterpret_cast<uint64_t *>(Section.Address + Offset);
    *TargetPtr = Value + Addend;
    break;
  }
  case ELF::R_AARCH64_PREL32: {
    uint64_t Result = Value + Addend - FinalAddress;
    assert(static_cast<int64_t>(Result) >= INT32_MIN &&
           static_cast<int64_t>(Result) <= UINT32_MAX);
    *TargetPtr = static_cast<uint32_t>(Result & 0xffffffffU);
    break;
  }
  case ELF::R_AARCH64_CALL26: // fallthrough
  case ELF::R_AARCH64_JUMP26: {
    // Operation: S+A-P. Set Call or B immediate value to bits fff_fffc of the
    // calculation.
    uint64_t BranchImm = Value + Addend - FinalAddress;

    // "Check that -2^27 <= result < 2^27".
    assert(isInt<28>(BranchImm));

    // AArch64 code is emitted with .rela relocations. The data already in any
    // bits affected by the relocation on entry is garbage.
    *TargetPtr &= 0xfc000000U;
    // Immediate goes in bits 25:0 of B and BL.
    *TargetPtr |= static_cast<uint32_t>(BranchImm & 0xffffffcU) >> 2;
    break;
  }
  case ELF::R_AARCH64_MOVW_UABS_G3: {
    uint64_t Result = Value + Addend;

    // AArch64 code is emitted with .rela relocations. The data already in any
    // bits affected by the relocation on entry is garbage.
    *TargetPtr &= 0xffe0001fU;
    // Immediate goes in bits 20:5 of MOVZ/MOVK instruction
    *TargetPtr |= Result >> (48 - 5);
    // Shift must be "lsl #48", in bits 22:21
    assert((*TargetPtr >> 21 & 0x3) == 3 && "invalid shift for relocation");
    break;
  }
  case ELF::R_AARCH64_MOVW_UABS_G2_NC: {
    uint64_t Result = Value + Addend;

    // AArch64 code is emitted with .rela relocations. The data already in any
    // bits affected by the relocation on entry is garbage.
    *TargetPtr &= 0xffe0001fU;
    // Immediate goes in bits 20:5 of MOVZ/MOVK instruction
    *TargetPtr |= ((Result & 0xffff00000000ULL) >> (32 - 5));
    // Shift must be "lsl #32", in bits 22:21
    assert((*TargetPtr >> 21 & 0x3) == 2 && "invalid shift for relocation");
    break;
  }
  case ELF::R_AARCH64_MOVW_UABS_G1_NC: {
    uint64_t Result = Value + Addend;

    // AArch64 code is emitted with .rela relocations. The data already in any
    // bits affected by the relocation on entry is garbage.
    *TargetPtr &= 0xffe0001fU;
    // Immediate goes in bits 20:5 of MOVZ/MOVK instruction
    *TargetPtr |= ((Result & 0xffff0000U) >> (16 - 5));
    // Shift must be "lsl #16", in bits 22:2
    assert((*TargetPtr >> 21 & 0x3) == 1 && "invalid shift for relocation");
    break;
  }
  case ELF::R_AARCH64_MOVW_UABS_G0_NC: {
    uint64_t Result = Value + Addend;

    // AArch64 code is emitted with .rela relocations. The data already in any
    // bits affected by the relocation on entry is garbage.
    *TargetPtr &= 0xffe0001fU;
    // Immediate goes in bits 20:5 of MOVZ/MOVK instruction
    *TargetPtr |= ((Result & 0xffffU) << 5);
    // Shift must be "lsl #0", in bits 22:21.
    assert((*TargetPtr >> 21 & 0x3) == 0 && "invalid shift for relocation");
    break;
  }
  case ELF::R_AARCH64_ADR_PREL_PG_HI21: {
    // Operation: Page(S+A) - Page(P)
    uint64_t Result =
        ((Value + Addend) & ~0xfffULL) - (FinalAddress & ~0xfffULL);

    // Check that -2^32 <= X < 2^32
    assert(isInt<33>(Result) && "overflow check failed for relocation");

    // AArch64 code is emitted with .rela relocations. The data already in any
    // bits affected by the relocation on entry is garbage.
    *TargetPtr &= 0x9f00001fU;
    // Immediate goes in bits 30:29 + 5:23 of ADRP instruction, taken
    // from bits 32:12 of X.
    *TargetPtr |= ((Result & 0x3000U) << (29 - 12));
    *TargetPtr |= ((Result & 0x1ffffc000ULL) >> (14 - 5));
    break;
  }
  case ELF::R_AARCH64_LDST32_ABS_LO12_NC: {
    // Operation: S + A
    uint64_t Result = Value + Addend;

    // AArch64 code is emitted with .rela relocations. The data already in any
    // bits affected by the relocation on entry is garbage.
    *TargetPtr &= 0xffc003ffU;
    // Immediate goes in bits 21:10 of LD/ST instruction, taken
    // from bits 11:2 of X
    *TargetPtr |= ((Result & 0xffc) << (10 - 2));
    break;
  }
  case ELF::R_AARCH64_LDST64_ABS_LO12_NC: {
    // Operation: S + A
    uint64_t Result = Value + Addend;

    // AArch64 code is emitted with .rela relocations. The data already in any
    // bits affected by the relocation on entry is garbage.
    *TargetPtr &= 0xffc003ffU;
    // Immediate goes in bits 21:10 of LD/ST instruction, taken
    // from bits 11:3 of X
    *TargetPtr |= ((Result & 0xff8) << (10 - 3));
    break;
  }
  }
}

void RuntimeDyldELF::resolveARMRelocation(const SectionEntry &Section,
                                          uint64_t Offset, uint32_t Value,
                                          uint32_t Type, int32_t Addend) {
  // TODO: Add Thumb relocations.
  uint32_t *TargetPtr = (uint32_t *)(Section.Address + Offset);
  uint32_t FinalAddress = ((Section.LoadAddress + Offset) & 0xFFFFFFFF);
  Value += Addend;

  DEBUG(dbgs() << "resolveARMRelocation, LocalAddress: "
               << Section.Address + Offset
               << " FinalAddress: " << format("%p", FinalAddress) << " Value: "
               << format("%x", Value) << " Type: " << format("%x", Type)
               << " Addend: " << format("%x", Addend) << "\n");

  switch (Type) {
  default:
    llvm_unreachable("Not implemented relocation type!");

  case ELF::R_ARM_NONE:
    break;
  case ELF::R_ARM_PREL31:
  case ELF::R_ARM_TARGET1:
  case ELF::R_ARM_ABS32:
    *TargetPtr = Value;
    break;
    // Write first 16 bit of 32 bit value to the mov instruction.
    // Last 4 bit should be shifted.
  case ELF::R_ARM_MOVW_ABS_NC:
  case ELF::R_ARM_MOVT_ABS:
    if (Type == ELF::R_ARM_MOVW_ABS_NC)
      Value = Value & 0xFFFF;
    else if (Type == ELF::R_ARM_MOVT_ABS)
      Value = (Value >> 16) & 0xFFFF;
    *TargetPtr &= ~0x000F0FFF;
    *TargetPtr |= Value & 0xFFF;
    *TargetPtr |= ((Value >> 12) & 0xF) << 16;
    break;
    // Write 24 bit relative value to the branch instruction.
  case ELF::R_ARM_PC24: // Fall through.
  case ELF::R_ARM_CALL: // Fall through.
  case ELF::R_ARM_JUMP24:
    int32_t RelValue = static_cast<int32_t>(Value - FinalAddress - 8);
    RelValue = (RelValue & 0x03FFFFFC) >> 2;
    assert((*TargetPtr & 0xFFFFFF) == 0xFFFFFE);
    *TargetPtr &= 0xFF000000;
    *TargetPtr |= RelValue;
    break;
  }
}

void RuntimeDyldELF::resolveMIPSRelocation(const SectionEntry &Section,
                                           uint64_t Offset, uint32_t Value,
                                           uint32_t Type, int32_t Addend) {
  uint8_t *TargetPtr = Section.Address + Offset;
  Value += Addend;

  DEBUG(dbgs() << "resolveMIPSRelocation, LocalAddress: "
               << Section.Address + Offset << " FinalAddress: "
               << format("%p", Section.LoadAddress + Offset) << " Value: "
               << format("%x", Value) << " Type: " << format("%x", Type)
               << " Addend: " << format("%x", Addend) << "\n");

  uint32_t Insn = readBytesUnaligned(TargetPtr, 4);

  switch (Type) {
  default:
    llvm_unreachable("Not implemented relocation type!");
    break;
  case ELF::R_MIPS_32:
    writeBytesUnaligned(Value, TargetPtr, 4);
    break;
  case ELF::R_MIPS_26:
    Insn &= 0xfc000000;
    Insn |= (Value & 0x0fffffff) >> 2;
    writeBytesUnaligned(Insn, TargetPtr, 4);
    break;
  case ELF::R_MIPS_HI16:
    // Get the higher 16-bits. Also add 1 if bit 15 is 1.
    Insn &= 0xffff0000;
    Insn |= ((Value + 0x8000) >> 16) & 0xffff;
    writeBytesUnaligned(Insn, TargetPtr, 4);
    break;
  case ELF::R_MIPS_LO16:
    Insn &= 0xffff0000;
    Insn |= Value & 0xffff;
    writeBytesUnaligned(Insn, TargetPtr, 4);
    break;
  case ELF::R_MIPS_PC32: {
    uint32_t FinalAddress = (Section.LoadAddress + Offset);
    writeBytesUnaligned(Value - FinalAddress, (uint8_t *)TargetPtr, 4);
    break;
  }
  case ELF::R_MIPS_PC16: {
    uint32_t FinalAddress = (Section.LoadAddress + Offset);
    Insn &= 0xffff0000;
    Insn |= ((Value - FinalAddress) >> 2) & 0xffff;
    writeBytesUnaligned(Insn, TargetPtr, 4);
    break;
  }
  case ELF::R_MIPS_PC19_S2: {
    uint32_t FinalAddress = (Section.LoadAddress + Offset);
    Insn &= 0xfff80000;
    Insn |= ((Value - (FinalAddress & ~0x3)) >> 2) & 0x7ffff;
    writeBytesUnaligned(Insn, TargetPtr, 4);
    break;
  }
  case ELF::R_MIPS_PC21_S2: {
    uint32_t FinalAddress = (Section.LoadAddress + Offset);
    Insn &= 0xffe00000;
    Insn |= ((Value - FinalAddress) >> 2) & 0x1fffff;
    writeBytesUnaligned(Insn, TargetPtr, 4);
    break;
  }
  case ELF::R_MIPS_PC26_S2: {
    uint32_t FinalAddress = (Section.LoadAddress + Offset);
    Insn &= 0xfc000000;
    Insn |= ((Value - FinalAddress) >> 2) & 0x3ffffff;
    writeBytesUnaligned(Insn, TargetPtr, 4);
    break;
  }
  case ELF::R_MIPS_PCHI16: {
    uint32_t FinalAddress = (Section.LoadAddress + Offset);
    Insn &= 0xffff0000;
    Insn |= ((Value - FinalAddress + 0x8000) >> 16) & 0xffff;
    writeBytesUnaligned(Insn, TargetPtr, 4);
    break;
  }
  case ELF::R_MIPS_PCLO16: {
    uint32_t FinalAddress = (Section.LoadAddress + Offset);
    Insn &= 0xffff0000;
    Insn |= (Value - FinalAddress) & 0xffff;
    writeBytesUnaligned(Insn, TargetPtr, 4);
    break;
  }
  }
}

void RuntimeDyldELF::setMipsABI(const ObjectFile &Obj) {
  if (Arch == Triple::UnknownArch ||
      !StringRef(Triple::getArchTypePrefix(Arch)).equals("mips")) {
    IsMipsO32ABI = false;
    IsMipsN64ABI = false;
    return;
  }
  unsigned AbiVariant;
  Obj.getPlatformFlags(AbiVariant);
  IsMipsO32ABI = AbiVariant & ELF::EF_MIPS_ABI_O32;
  IsMipsN64ABI = Obj.getFileFormatName().equals("ELF64-mips");
  if (AbiVariant & ELF::EF_MIPS_ABI2)
    llvm_unreachable("Mips N32 ABI is not supported yet");
}

void RuntimeDyldELF::resolveMIPS64Relocation(const SectionEntry &Section,
                                             uint64_t Offset, uint64_t Value,
                                             uint32_t Type, int64_t Addend,
                                             uint64_t SymOffset,
                                             SID SectionID) {
  uint32_t r_type = Type & 0xff;
  uint32_t r_type2 = (Type >> 8) & 0xff;
  uint32_t r_type3 = (Type >> 16) & 0xff;

  // RelType is used to keep information for which relocation type we are
  // applying relocation.
  uint32_t RelType = r_type;
  int64_t CalculatedValue = evaluateMIPS64Relocation(Section, Offset, Value,
                                                     RelType, Addend,
                                                     SymOffset, SectionID);
  if (r_type2 != ELF::R_MIPS_NONE) {
    RelType = r_type2;
    CalculatedValue = evaluateMIPS64Relocation(Section, Offset, 0, RelType,
                                               CalculatedValue, SymOffset,
                                               SectionID);
  }
  if (r_type3 != ELF::R_MIPS_NONE) {
    RelType = r_type3;
    CalculatedValue = evaluateMIPS64Relocation(Section, Offset, 0, RelType,
                                               CalculatedValue, SymOffset,
                                               SectionID);
  }
  applyMIPS64Relocation(Section.Address + Offset, CalculatedValue, RelType);
}

int64_t
RuntimeDyldELF::evaluateMIPS64Relocation(const SectionEntry &Section,
                                         uint64_t Offset, uint64_t Value,
                                         uint32_t Type, int64_t Addend,
                                         uint64_t SymOffset, SID SectionID) {

  DEBUG(dbgs() << "evaluateMIPS64Relocation, LocalAddress: 0x"
               << format("%llx", Section.Address + Offset)
               << " FinalAddress: 0x"
               << format("%llx", Section.LoadAddress + Offset)
               << " Value: 0x" << format("%llx", Value) << " Type: 0x"
               << format("%x", Type) << " Addend: 0x" << format("%llx", Addend)
               << " SymOffset: " << format("%x", SymOffset)
               << "\n");

  switch (Type) {
  default:
    llvm_unreachable("Not implemented relocation type!");
    break;
  case ELF::R_MIPS_JALR:
  case ELF::R_MIPS_NONE:
    break;
  case ELF::R_MIPS_32:
  case ELF::R_MIPS_64:
    return Value + Addend;
  case ELF::R_MIPS_26:
    return ((Value + Addend) >> 2) & 0x3ffffff;
  case ELF::R_MIPS_GPREL16: {
    uint64_t GOTAddr = getSectionLoadAddress(SectionToGOTMap[SectionID]);
    return Value + Addend - (GOTAddr + 0x7ff0);
  }
  case ELF::R_MIPS_SUB:
    return Value - Addend;
  case ELF::R_MIPS_HI16:
    // Get the higher 16-bits. Also add 1 if bit 15 is 1.
    return ((Value + Addend + 0x8000) >> 16) & 0xffff;
  case ELF::R_MIPS_LO16:
    return (Value + Addend) & 0xffff;
  case ELF::R_MIPS_CALL16:
  case ELF::R_MIPS_GOT_DISP:
  case ELF::R_MIPS_GOT_PAGE: {
    uint8_t *LocalGOTAddr =
        getSectionAddress(SectionToGOTMap[SectionID]) + SymOffset;
    uint64_t GOTEntry = readBytesUnaligned(LocalGOTAddr, 8);

    Value += Addend;
    if (Type == ELF::R_MIPS_GOT_PAGE)
      Value = (Value + 0x8000) & ~0xffff;

    if (GOTEntry)
      assert(GOTEntry == Value &&
                   "GOT entry has two different addresses.");
    else
      writeBytesUnaligned(Value, LocalGOTAddr, 8);

    return (SymOffset - 0x7ff0) & 0xffff;
  }
  case ELF::R_MIPS_GOT_OFST: {
    int64_t page = (Value + Addend + 0x8000) & ~0xffff;
    return (Value + Addend - page) & 0xffff;
  }
  case ELF::R_MIPS_GPREL32: {
    uint64_t GOTAddr = getSectionLoadAddress(SectionToGOTMap[SectionID]);
    return Value + Addend - (GOTAddr + 0x7ff0);
  }
  case ELF::R_MIPS_PC16: {
    uint64_t FinalAddress = (Section.LoadAddress + Offset);
    return ((Value + Addend - FinalAddress) >> 2) & 0xffff;
  }
  case ELF::R_MIPS_PC32: {
    uint64_t FinalAddress = (Section.LoadAddress + Offset);
    return Value + Addend - FinalAddress;
  }
  case ELF::R_MIPS_PC18_S3: {
    uint64_t FinalAddress = (Section.LoadAddress + Offset);
    return ((Value + Addend - ((FinalAddress | 7) ^ 7)) >> 3) & 0x3ffff;
  }
  case ELF::R_MIPS_PC19_S2: {
    uint64_t FinalAddress = (Section.LoadAddress + Offset);
    return ((Value + Addend - FinalAddress) >> 2) & 0x7ffff;
  }
  case ELF::R_MIPS_PC21_S2: {
    uint64_t FinalAddress = (Section.LoadAddress + Offset);
    return ((Value + Addend - FinalAddress) >> 2) & 0x1fffff;
  }
  case ELF::R_MIPS_PC26_S2: {
    uint64_t FinalAddress = (Section.LoadAddress + Offset);
    return ((Value + Addend - FinalAddress) >> 2) & 0x3ffffff;
  }
  case ELF::R_MIPS_PCHI16: {
    uint64_t FinalAddress = (Section.LoadAddress + Offset);
    return ((Value + Addend - FinalAddress + 0x8000) >> 16) & 0xffff;
  }
  case ELF::R_MIPS_PCLO16: {
    uint64_t FinalAddress = (Section.LoadAddress + Offset);
    return (Value + Addend - FinalAddress) & 0xffff;
  }
  }
  return 0;
}

void RuntimeDyldELF::applyMIPS64Relocation(uint8_t *TargetPtr,
                                           int64_t CalculatedValue,
                                           uint32_t Type) {
  uint32_t Insn = readBytesUnaligned(TargetPtr, 4);

  switch (Type) {
    default:
      break;
    case ELF::R_MIPS_32:
    case ELF::R_MIPS_GPREL32:
    case ELF::R_MIPS_PC32:
      writeBytesUnaligned(CalculatedValue & 0xffffffff, TargetPtr, 4);
      break;
    case ELF::R_MIPS_64:
    case ELF::R_MIPS_SUB:
      writeBytesUnaligned(CalculatedValue, TargetPtr, 8);
      break;
    case ELF::R_MIPS_26:
    case ELF::R_MIPS_PC26_S2:
      Insn = (Insn & 0xfc000000) | CalculatedValue;
      writeBytesUnaligned(Insn, TargetPtr, 4);
      break;
    case ELF::R_MIPS_GPREL16:
      Insn = (Insn & 0xffff0000) | (CalculatedValue & 0xffff);
      writeBytesUnaligned(Insn, TargetPtr, 4);
      break;
    case ELF::R_MIPS_HI16:
    case ELF::R_MIPS_LO16:
    case ELF::R_MIPS_PCHI16:
    case ELF::R_MIPS_PCLO16:
    case ELF::R_MIPS_PC16:
    case ELF::R_MIPS_CALL16:
    case ELF::R_MIPS_GOT_DISP:
    case ELF::R_MIPS_GOT_PAGE:
    case ELF::R_MIPS_GOT_OFST:
      Insn = (Insn & 0xffff0000) | CalculatedValue;
      writeBytesUnaligned(Insn, TargetPtr, 4);
      break;
    case ELF::R_MIPS_PC18_S3:
      Insn = (Insn & 0xfffc0000) | CalculatedValue;
      writeBytesUnaligned(Insn, TargetPtr, 4);
      break;
    case ELF::R_MIPS_PC19_S2:
      Insn = (Insn & 0xfff80000) | CalculatedValue;
      writeBytesUnaligned(Insn, TargetPtr, 4);
      break;
    case ELF::R_MIPS_PC21_S2:
      Insn = (Insn & 0xffe00000) | CalculatedValue;
      writeBytesUnaligned(Insn, TargetPtr, 4);
      break;
    }
}

// Return the .TOC. section and offset.
void RuntimeDyldELF::findPPC64TOCSection(const ELFObjectFileBase &Obj,
                                         ObjSectionToIDMap &LocalSections,
                                         RelocationValueRef &Rel) {
  // Set a default SectionID in case we do not find a TOC section below.
  // This may happen for references to TOC base base (sym@toc, .odp
  // relocation) without a .toc directive.  In this case just use the
  // first section (which is usually the .odp) since the code won't
  // reference the .toc base directly.
  Rel.SymbolName = NULL;
  Rel.SectionID = 0;

  // The TOC consists of sections .got, .toc, .tocbss, .plt in that
  // order. The TOC starts where the first of these sections starts.
  for (auto &Section: Obj.sections()) {
    StringRef SectionName;
    check(Section.getName(SectionName));

    if (SectionName == ".got"
        || SectionName == ".toc"
        || SectionName == ".tocbss"
        || SectionName == ".plt") {
      Rel.SectionID = findOrEmitSection(Obj, Section, false, LocalSections);
      break;
    }
  }

  // Per the ppc64-elf-linux ABI, The TOC base is TOC value plus 0x8000
  // thus permitting a full 64 Kbytes segment.
  Rel.Addend = 0x8000;
}

// Returns the sections and offset associated with the ODP entry referenced
// by Symbol.
void RuntimeDyldELF::findOPDEntrySection(const ELFObjectFileBase &Obj,
                                         ObjSectionToIDMap &LocalSections,
                                         RelocationValueRef &Rel) {
  // Get the ELF symbol value (st_value) to compare with Relocation offset in
  // .opd entries
  for (section_iterator si = Obj.section_begin(), se = Obj.section_end();
       si != se; ++si) {
    section_iterator RelSecI = si->getRelocatedSection();
    if (RelSecI == Obj.section_end())
      continue;

    StringRef RelSectionName;
    check(RelSecI->getName(RelSectionName));
    if (RelSectionName != ".opd")
      continue;

    for (elf_relocation_iterator i = si->relocation_begin(),
                                 e = si->relocation_end();
         i != e;) {
      // The R_PPC64_ADDR64 relocation indicates the first field
      // of a .opd entry
      uint64_t TypeFunc = i->getType();
      if (TypeFunc != ELF::R_PPC64_ADDR64) {
        ++i;
        continue;
      }

      uint64_t TargetSymbolOffset = i->getOffset();
      symbol_iterator TargetSymbol = i->getSymbol();
      ErrorOr<int64_t> AddendOrErr = i->getAddend();
      Check(AddendOrErr.getError());
      int64_t Addend = *AddendOrErr;

      ++i;
      if (i == e)
        break;

      // Just check if following relocation is a R_PPC64_TOC
      uint64_t TypeTOC = i->getType();
      if (TypeTOC != ELF::R_PPC64_TOC)
        continue;

      // Finally compares the Symbol value and the target symbol offset
      // to check if this .opd entry refers to the symbol the relocation
      // points to.
      if (Rel.Addend != (int64_t)TargetSymbolOffset)
        continue;

      section_iterator tsi(Obj.section_end());
      check(TargetSymbol->getSection(tsi));
      bool IsCode = tsi->isText();
      Rel.SectionID = findOrEmitSection(Obj, (*tsi), IsCode, LocalSections);
      Rel.Addend = (intptr_t)Addend;
      return;
    }
  }
  llvm_unreachable("Attempting to get address of ODP entry!");
}

// Relocation masks following the #lo(value), #hi(value), #ha(value),
// #higher(value), #highera(value), #highest(value), and #highesta(value)
// macros defined in section 4.5.1. Relocation Types of the PPC-elf64abi
// document.

static inline uint16_t applyPPClo(uint64_t value) { return value & 0xffff; }

static inline uint16_t applyPPChi(uint64_t value) {
  return (value >> 16) & 0xffff;
}

static inline uint16_t applyPPCha (uint64_t value) {
  return ((value + 0x8000) >> 16) & 0xffff;
}

static inline uint16_t applyPPChigher(uint64_t value) {
  return (value >> 32) & 0xffff;
}

static inline uint16_t applyPPChighera (uint64_t value) {
  return ((value + 0x8000) >> 32) & 0xffff;
}

static inline uint16_t applyPPChighest(uint64_t value) {
  return (value >> 48) & 0xffff;
}

static inline uint16_t applyPPChighesta (uint64_t value) {
  return ((value + 0x8000) >> 48) & 0xffff;
}

void RuntimeDyldELF::resolvePPC64Relocation(const SectionEntry &Section,
                                            uint64_t Offset, uint64_t Value,
                                            uint32_t Type, int64_t Addend) {
  uint8_t *LocalAddress = Section.Address + Offset;
  switch (Type) {
  default:
    llvm_unreachable("Relocation type not implemented yet!");
    break;
  case ELF::R_PPC64_ADDR16:
    writeInt16BE(LocalAddress, applyPPClo(Value + Addend));
    break;
  case ELF::R_PPC64_ADDR16_DS:
    writeInt16BE(LocalAddress, applyPPClo(Value + Addend) & ~3);
    break;
  case ELF::R_PPC64_ADDR16_LO:
    writeInt16BE(LocalAddress, applyPPClo(Value + Addend));
    break;
  case ELF::R_PPC64_ADDR16_LO_DS:
    writeInt16BE(LocalAddress, applyPPClo(Value + Addend) & ~3);
    break;
  case ELF::R_PPC64_ADDR16_HI:
    writeInt16BE(LocalAddress, applyPPChi(Value + Addend));
    break;
  case ELF::R_PPC64_ADDR16_HA:
    writeInt16BE(LocalAddress, applyPPCha(Value + Addend));
    break;
  case ELF::R_PPC64_ADDR16_HIGHER:
    writeInt16BE(LocalAddress, applyPPChigher(Value + Addend));
    break;
  case ELF::R_PPC64_ADDR16_HIGHERA:
    writeInt16BE(LocalAddress, applyPPChighera(Value + Addend));
    break;
  case ELF::R_PPC64_ADDR16_HIGHEST:
    writeInt16BE(LocalAddress, applyPPChighest(Value + Addend));
    break;
  case ELF::R_PPC64_ADDR16_HIGHESTA:
    writeInt16BE(LocalAddress, applyPPChighesta(Value + Addend));
    break;
  case ELF::R_PPC64_ADDR14: {
    assert(((Value + Addend) & 3) == 0);
    // Preserve the AA/LK bits in the branch instruction
    uint8_t aalk = *(LocalAddress + 3);
    writeInt16BE(LocalAddress + 2, (aalk & 3) | ((Value + Addend) & 0xfffc));
  } break;
  case ELF::R_PPC64_REL16_LO: {
    uint64_t FinalAddress = (Section.LoadAddress + Offset);
    uint64_t Delta = Value - FinalAddress + Addend;
    writeInt16BE(LocalAddress, applyPPClo(Delta));
  } break;
  case ELF::R_PPC64_REL16_HI: {
    uint64_t FinalAddress = (Section.LoadAddress + Offset);
    uint64_t Delta = Value - FinalAddress + Addend;
    writeInt16BE(LocalAddress, applyPPChi(Delta));
  } break;
  case ELF::R_PPC64_REL16_HA: {
    uint64_t FinalAddress = (Section.LoadAddress + Offset);
    uint64_t Delta = Value - FinalAddress + Addend;
    writeInt16BE(LocalAddress, applyPPCha(Delta));
  } break;
  case ELF::R_PPC64_ADDR32: {
    int32_t Result = static_cast<int32_t>(Value + Addend);
    if (SignExtend32<32>(Result) != Result)
      llvm_unreachable("Relocation R_PPC64_ADDR32 overflow");
    writeInt32BE(LocalAddress, Result);
  } break;
  case ELF::R_PPC64_REL24: {
    uint64_t FinalAddress = (Section.LoadAddress + Offset);
    int32_t delta = static_cast<int32_t>(Value - FinalAddress + Addend);
    if (SignExtend32<24>(delta) != delta)
      llvm_unreachable("Relocation R_PPC64_REL24 overflow");
    // Generates a 'bl <address>' instruction
    writeInt32BE(LocalAddress, 0x48000001 | (delta & 0x03FFFFFC));
  } break;
  case ELF::R_PPC64_REL32: {
    uint64_t FinalAddress = (Section.LoadAddress + Offset);
    int32_t delta = static_cast<int32_t>(Value - FinalAddress + Addend);
    if (SignExtend32<32>(delta) != delta)
      llvm_unreachable("Relocation R_PPC64_REL32 overflow");
    writeInt32BE(LocalAddress, delta);
  } break;
  case ELF::R_PPC64_REL64: {
    uint64_t FinalAddress = (Section.LoadAddress + Offset);
    uint64_t Delta = Value - FinalAddress + Addend;
    writeInt64BE(LocalAddress, Delta);
  } break;
  case ELF::R_PPC64_ADDR64:
    writeInt64BE(LocalAddress, Value + Addend);
    break;
  }
}

void RuntimeDyldELF::resolveSystemZRelocation(const SectionEntry &Section,
                                              uint64_t Offset, uint64_t Value,
                                              uint32_t Type, int64_t Addend) {
  uint8_t *LocalAddress = Section.Address + Offset;
  switch (Type) {
  default:
    llvm_unreachable("Relocation type not implemented yet!");
    break;
  case ELF::R_390_PC16DBL:
  case ELF::R_390_PLT16DBL: {
    int64_t Delta = (Value + Addend) - (Section.LoadAddress + Offset);
    assert(int16_t(Delta / 2) * 2 == Delta && "R_390_PC16DBL overflow");
    writeInt16BE(LocalAddress, Delta / 2);
    break;
  }
  case ELF::R_390_PC32DBL:
  case ELF::R_390_PLT32DBL: {
    int64_t Delta = (Value + Addend) - (Section.LoadAddress + Offset);
    assert(int32_t(Delta / 2) * 2 == Delta && "R_390_PC32DBL overflow");
    writeInt32BE(LocalAddress, Delta / 2);
    break;
  }
  case ELF::R_390_PC32: {
    int64_t Delta = (Value + Addend) - (Section.LoadAddress + Offset);
    assert(int32_t(Delta) == Delta && "R_390_PC32 overflow");
    writeInt32BE(LocalAddress, Delta);
    break;
  }
  case ELF::R_390_64:
    writeInt64BE(LocalAddress, Value + Addend);
    break;
  }
}

// The target location for the relocation is described by RE.SectionID and
// RE.Offset.  RE.SectionID can be used to find the SectionEntry.  Each
// SectionEntry has three members describing its location.
// SectionEntry::Address is the address at which the section has been loaded
// into memory in the current (host) process.  SectionEntry::LoadAddress is the
// address that the section will have in the target process.
// SectionEntry::ObjAddress is the address of the bits for this section in the
// original emitted object image (also in the current address space).
//
// Relocations will be applied as if the section were loaded at
// SectionEntry::LoadAddress, but they will be applied at an address based
// on SectionEntry::Address.  SectionEntry::ObjAddress will be used to refer to
// Target memory contents if they are required for value calculations.
//
// The Value parameter here is the load address of the symbol for the
// relocation to be applied.  For relocations which refer to symbols in the
// current object Value will be the LoadAddress of the section in which
// the symbol resides (RE.Addend provides additional information about the
// symbol location).  For external symbols, Value will be the address of the
// symbol in the target address space.
void RuntimeDyldELF::resolveRelocation(const RelocationEntry &RE,
                                       uint64_t Value) {
  const SectionEntry &Section = Sections[RE.SectionID];
  return resolveRelocation(Section, RE.Offset, Value, RE.RelType, RE.Addend,
                           RE.SymOffset, RE.SectionID);
}

void RuntimeDyldELF::resolveRelocation(const SectionEntry &Section,
                                       uint64_t Offset, uint64_t Value,
                                       uint32_t Type, int64_t Addend,
                                       uint64_t SymOffset, SID SectionID) {
  switch (Arch) {
  case Triple::x86_64:
    resolveX86_64Relocation(Section, Offset, Value, Type, Addend, SymOffset);
    break;
  case Triple::x86:
    resolveX86Relocation(Section, Offset, (uint32_t)(Value & 0xffffffffL), Type,
                         (uint32_t)(Addend & 0xffffffffL));
    break;
  case Triple::aarch64:
  case Triple::aarch64_be:
    resolveAArch64Relocation(Section, Offset, Value, Type, Addend);
    break;
  case Triple::arm: // Fall through.
  case Triple::armeb:
  case Triple::thumb:
  case Triple::thumbeb:
    resolveARMRelocation(Section, Offset, (uint32_t)(Value & 0xffffffffL), Type,
                         (uint32_t)(Addend & 0xffffffffL));
    break;
  case Triple::mips: // Fall through.
  case Triple::mipsel:
  case Triple::mips64:
  case Triple::mips64el:
    if (IsMipsO32ABI)
      resolveMIPSRelocation(Section, Offset, (uint32_t)(Value & 0xffffffffL),
                            Type, (uint32_t)(Addend & 0xffffffffL));
    else if (IsMipsN64ABI)
      resolveMIPS64Relocation(Section, Offset, Value, Type, Addend, SymOffset,
                              SectionID);
    else
      llvm_unreachable("Mips ABI not handled");
    break;
  case Triple::ppc64: // Fall through.
  case Triple::ppc64le:
    resolvePPC64Relocation(Section, Offset, Value, Type, Addend);
    break;
  case Triple::systemz:
    resolveSystemZRelocation(Section, Offset, Value, Type, Addend);
    break;
  default:
    llvm_unreachable("Unsupported CPU type!");
  }
}

void *RuntimeDyldELF::computePlaceholderAddress(unsigned SectionID, uint64_t Offset) const {
  return (void*)(Sections[SectionID].ObjAddress + Offset);
}

void RuntimeDyldELF::processSimpleRelocation(unsigned SectionID, uint64_t Offset, unsigned RelType, RelocationValueRef Value) {
  RelocationEntry RE(SectionID, Offset, RelType, Value.Addend, Value.Offset);
  if (Value.SymbolName)
    addRelocationForSymbol(RE, Value.SymbolName);
  else
    addRelocationForSection(RE, Value.SectionID);
}

relocation_iterator RuntimeDyldELF::processRelocationRef(
    unsigned SectionID, relocation_iterator RelI, const ObjectFile &O,
    ObjSectionToIDMap &ObjSectionToID, StubMap &Stubs) {
  const auto &Obj = cast<ELFObjectFileBase>(O);
  uint64_t RelType = RelI->getType();
  ErrorOr<int64_t> AddendOrErr = ELFRelocationRef(*RelI).getAddend();
  int64_t Addend = AddendOrErr ? *AddendOrErr : 0;
  elf_symbol_iterator Symbol = RelI->getSymbol();

  // Obtain the symbol name which is referenced in the relocation
  StringRef TargetName;
  if (Symbol != Obj.symbol_end()) {
    ErrorOr<StringRef> TargetNameOrErr = Symbol->getName();
    if (std::error_code EC = TargetNameOrErr.getError())
      report_fatal_error(EC.message());
    TargetName = *TargetNameOrErr;
  }
  DEBUG(dbgs() << "\t\tRelType: " << RelType << " Addend: " << Addend
               << " TargetName: " << TargetName << "\n");
  RelocationValueRef Value;
  // First search for the symbol in the local symbol table
  SymbolRef::Type SymType = SymbolRef::ST_Unknown;

  // Search for the symbol in the global symbol table
  RTDyldSymbolTable::const_iterator gsi = GlobalSymbolTable.end();
  if (Symbol != Obj.symbol_end()) {
    gsi = GlobalSymbolTable.find(TargetName.data());
    SymType = Symbol->getType();
  }
  if (gsi != GlobalSymbolTable.end()) {
    const auto &SymInfo = gsi->second;
    Value.SectionID = SymInfo.getSectionID();
    Value.Offset = SymInfo.getOffset();
    Value.Addend = SymInfo.getOffset() + Addend;
  } else {
    switch (SymType) {
    case SymbolRef::ST_Debug: {
      // TODO: Now ELF SymbolRef::ST_Debug = STT_SECTION, it's not obviously
      // and can be changed by another developers. Maybe best way is add
      // a new symbol type ST_Section to SymbolRef and use it.
      section_iterator si(Obj.section_end());
      Symbol->getSection(si);
      if (si == Obj.section_end())
        llvm_unreachable("Symbol section not found, bad object file format!");
      DEBUG(dbgs() << "\t\tThis is section symbol\n");
      bool isCode = si->isText();
      Value.SectionID = findOrEmitSection(Obj, (*si), isCode, ObjSectionToID);
      Value.Addend = Addend;
      break;
    }
    case SymbolRef::ST_Data:
    case SymbolRef::ST_Unknown: {
      Value.SymbolName = TargetName.data();
      Value.Addend = Addend;

      // Absolute relocations will have a zero symbol ID (STN_UNDEF), which
      // will manifest here as a NULL symbol name.
      // We can set this as a valid (but empty) symbol name, and rely
      // on addRelocationForSymbol to handle this.
      if (!Value.SymbolName)
        Value.SymbolName = "";
      break;
    }
    default:
      llvm_unreachable("Unresolved symbol type!");
      break;
    }
  }

  uint64_t Offset = RelI->getOffset();

  DEBUG(dbgs() << "\t\tSectionID: " << SectionID << " Offset: " << Offset
               << "\n");
  if ((Arch == Triple::aarch64 || Arch == Triple::aarch64_be) &&
      (RelType == ELF::R_AARCH64_CALL26 || RelType == ELF::R_AARCH64_JUMP26)) {
    // This is an AArch64 branch relocation, need to use a stub function.
    DEBUG(dbgs() << "\t\tThis is an AArch64 branch relocation.");
    SectionEntry &Section = Sections[SectionID];

    // Look for an existing stub.
    StubMap::const_iterator i = Stubs.find(Value);
    if (i != Stubs.end()) {
      resolveRelocation(Section, Offset, (uint64_t)Section.Address + i->second,
                        RelType, 0);
      DEBUG(dbgs() << " Stub function found\n");
    } else {
      // Create a new stub function.
      DEBUG(dbgs() << " Create a new stub function\n");
      Stubs[Value] = Section.StubOffset;
      uint8_t *StubTargetAddr =
          createStubFunction(Section.Address + Section.StubOffset);

      RelocationEntry REmovz_g3(SectionID, StubTargetAddr - Section.Address,
                                ELF::R_AARCH64_MOVW_UABS_G3, Value.Addend);
      RelocationEntry REmovk_g2(SectionID, StubTargetAddr - Section.Address + 4,
                                ELF::R_AARCH64_MOVW_UABS_G2_NC, Value.Addend);
      RelocationEntry REmovk_g1(SectionID, StubTargetAddr - Section.Address + 8,
                                ELF::R_AARCH64_MOVW_UABS_G1_NC, Value.Addend);
      RelocationEntry REmovk_g0(SectionID,
                                StubTargetAddr - Section.Address + 12,
                                ELF::R_AARCH64_MOVW_UABS_G0_NC, Value.Addend);

      if (Value.SymbolName) {
        addRelocationForSymbol(REmovz_g3, Value.SymbolName);
        addRelocationForSymbol(REmovk_g2, Value.SymbolName);
        addRelocationForSymbol(REmovk_g1, Value.SymbolName);
        addRelocationForSymbol(REmovk_g0, Value.SymbolName);
      } else {
        addRelocationForSection(REmovz_g3, Value.SectionID);
        addRelocationForSection(REmovk_g2, Value.SectionID);
        addRelocationForSection(REmovk_g1, Value.SectionID);
        addRelocationForSection(REmovk_g0, Value.SectionID);
      }
      resolveRelocation(Section, Offset,
                        (uint64_t)Section.Address + Section.StubOffset, RelType,
                        0);
      Section.StubOffset += getMaxStubSize();
    }
  } else if (Arch == Triple::arm) {
    if (RelType == ELF::R_ARM_PC24 || RelType == ELF::R_ARM_CALL ||
      RelType == ELF::R_ARM_JUMP24) {
      // This is an ARM branch relocation, need to use a stub function.
      DEBUG(dbgs() << "\t\tThis is an ARM branch relocation.");
      SectionEntry &Section = Sections[SectionID];

      // Look for an existing stub.
      StubMap::const_iterator i = Stubs.find(Value);
      if (i != Stubs.end()) {
        resolveRelocation(Section, Offset, (uint64_t)Section.Address + i->second,
          RelType, 0);
        DEBUG(dbgs() << " Stub function found\n");
      } else {
        // Create a new stub function.
        DEBUG(dbgs() << " Create a new stub function\n");
        Stubs[Value] = Section.StubOffset;
        uint8_t *StubTargetAddr =
          createStubFunction(Section.Address + Section.StubOffset);
        RelocationEntry RE(SectionID, StubTargetAddr - Section.Address,
          ELF::R_ARM_ABS32, Value.Addend);
        if (Value.SymbolName)
          addRelocationForSymbol(RE, Value.SymbolName);
        else
          addRelocationForSection(RE, Value.SectionID);

        resolveRelocation(Section, Offset,
          (uint64_t)Section.Address + Section.StubOffset, RelType,
          0);
        Section.StubOffset += getMaxStubSize();
      }
    } else {
      uint32_t *Placeholder =
        reinterpret_cast<uint32_t*>(computePlaceholderAddress(SectionID, Offset));
      if (RelType == ELF::R_ARM_PREL31 || RelType == ELF::R_ARM_TARGET1 ||
          RelType == ELF::R_ARM_ABS32) {
        Value.Addend += *Placeholder;
      } else if (RelType == ELF::R_ARM_MOVW_ABS_NC || RelType == ELF::R_ARM_MOVT_ABS) {
        // See ELF for ARM documentation
        Value.Addend += (int16_t)((*Placeholder & 0xFFF) | (((*Placeholder >> 16) & 0xF) << 12));
      }
      processSimpleRelocation(SectionID, Offset, RelType, Value);
    }
  } else if (IsMipsO32ABI) {
    uint8_t *Placeholder = reinterpret_cast<uint8_t *>(
        computePlaceholderAddress(SectionID, Offset));
    uint32_t Opcode = readBytesUnaligned(Placeholder, 4);
    if (RelType == ELF::R_MIPS_26) {
      // This is an Mips branch relocation, need to use a stub function.
      DEBUG(dbgs() << "\t\tThis is a Mips branch relocation.");
      SectionEntry &Section = Sections[SectionID];

      // Extract the addend from the instruction.
      // We shift up by two since the Value will be down shifted again
      // when applying the relocation.
      uint32_t Addend = (Opcode & 0x03ffffff) << 2;

      Value.Addend += Addend;

      //  Look up for existing stub.
      StubMap::const_iterator i = Stubs.find(Value);
      if (i != Stubs.end()) {
        RelocationEntry RE(SectionID, Offset, RelType, i->second);
        addRelocationForSection(RE, SectionID);
        DEBUG(dbgs() << " Stub function found\n");
      } else {
        // Create a new stub function.
        DEBUG(dbgs() << " Create a new stub function\n");
        Stubs[Value] = Section.StubOffset;
        uint8_t *StubTargetAddr =
          createStubFunction(Section.Address + Section.StubOffset);

        // Creating Hi and Lo relocations for the filled stub instructions.
        RelocationEntry REHi(SectionID, StubTargetAddr - Section.Address,
          ELF::R_MIPS_HI16, Value.Addend);
        RelocationEntry RELo(SectionID, StubTargetAddr - Section.Address + 4,
          ELF::R_MIPS_LO16, Value.Addend);

        if (Value.SymbolName) {
          addRelocationForSymbol(REHi, Value.SymbolName);
          addRelocationForSymbol(RELo, Value.SymbolName);
        }
        else {
          addRelocationForSection(REHi, Value.SectionID);
          addRelocationForSection(RELo, Value.SectionID);
        }

        RelocationEntry RE(SectionID, Offset, RelType, Section.StubOffset);
        addRelocationForSection(RE, SectionID);
        Section.StubOffset += getMaxStubSize();
      }
    } else {
      // FIXME: Calculate correct addends for R_MIPS_HI16, R_MIPS_LO16,
      // R_MIPS_PCHI16 and R_MIPS_PCLO16 relocations.
      if (RelType == ELF::R_MIPS_HI16 || RelType == ELF::R_MIPS_PCHI16)
        Value.Addend += (Opcode & 0x0000ffff) << 16;
      else if (RelType == ELF::R_MIPS_LO16)
        Value.Addend += (Opcode & 0x0000ffff);
      else if (RelType == ELF::R_MIPS_32)
        Value.Addend += Opcode;
      else if (RelType == ELF::R_MIPS_PCLO16)
        Value.Addend += SignExtend32<16>((Opcode & 0x0000ffff));
      else if (RelType == ELF::R_MIPS_PC16)
        Value.Addend += SignExtend32<18>((Opcode & 0x0000ffff) << 2);
      else if (RelType == ELF::R_MIPS_PC19_S2)
        Value.Addend += SignExtend32<21>((Opcode & 0x0007ffff) << 2);
      else if (RelType == ELF::R_MIPS_PC21_S2)
        Value.Addend += SignExtend32<23>((Opcode & 0x001fffff) << 2);
      else if (RelType == ELF::R_MIPS_PC26_S2)
        Value.Addend += SignExtend32<28>((Opcode & 0x03ffffff) << 2);
      processSimpleRelocation(SectionID, Offset, RelType, Value);
    }
  } else if (IsMipsN64ABI) {
    uint32_t r_type = RelType & 0xff;
    RelocationEntry RE(SectionID, Offset, RelType, Value.Addend);
    if (r_type == ELF::R_MIPS_CALL16 || r_type == ELF::R_MIPS_GOT_PAGE
        || r_type == ELF::R_MIPS_GOT_DISP) {
      StringMap<uint64_t>::iterator i = GOTSymbolOffsets.find(TargetName);
      if (i != GOTSymbolOffsets.end())
        RE.SymOffset = i->second;
      else {
        RE.SymOffset = allocateGOTEntries(SectionID, 1);
        GOTSymbolOffsets[TargetName] = RE.SymOffset;
      }
    }
    if (Value.SymbolName)
      addRelocationForSymbol(RE, Value.SymbolName);
    else
      addRelocationForSection(RE, Value.SectionID);
  } else if (Arch == Triple::ppc64 || Arch == Triple::ppc64le) {
    if (RelType == ELF::R_PPC64_REL24) {
      // Determine ABI variant in use for this object.
      unsigned AbiVariant;
      Obj.getPlatformFlags(AbiVariant);
      AbiVariant &= ELF::EF_PPC64_ABI;
      // A PPC branch relocation will need a stub function if the target is
      // an external symbol (Symbol::ST_Unknown) or if the target address
      // is not within the signed 24-bits branch address.
      SectionEntry &Section = Sections[SectionID];
      uint8_t *Target = Section.Address + Offset;
      bool RangeOverflow = false;
      if (SymType != SymbolRef::ST_Unknown) {
        if (AbiVariant != 2) {
          // In the ELFv1 ABI, a function call may point to the .opd entry,
          // so the final symbol value is calculated based on the relocation
          // values in the .opd section.
          findOPDEntrySection(Obj, ObjSectionToID, Value);
        } else {
          // In the ELFv2 ABI, a function symbol may provide a local entry
          // point, which must be used for direct calls.
          uint8_t SymOther = Symbol->getOther();
          Value.Addend += ELF::decodePPC64LocalEntryOffset(SymOther);
        }
        uint8_t *RelocTarget = Sections[Value.SectionID].Address + Value.Addend;
        int32_t delta = static_cast<int32_t>(Target - RelocTarget);
        // If it is within 24-bits branch range, just set the branch target
        if (SignExtend32<24>(delta) == delta) {
          RelocationEntry RE(SectionID, Offset, RelType, Value.Addend);
          if (Value.SymbolName)
            addRelocationForSymbol(RE, Value.SymbolName);
          else
            addRelocationForSection(RE, Value.SectionID);
        } else {
          RangeOverflow = true;
        }
      }
      if (SymType == SymbolRef::ST_Unknown || RangeOverflow) {
        // It is an external symbol (SymbolRef::ST_Unknown) or within a range
        // larger than 24-bits.
        StubMap::const_iterator i = Stubs.find(Value);
        if (i != Stubs.end()) {
          // Symbol function stub already created, just relocate to it
          resolveRelocation(Section, Offset,
                            (uint64_t)Section.Address + i->second, RelType, 0);
          DEBUG(dbgs() << " Stub function found\n");
        } else {
          // Create a new stub function.
          DEBUG(dbgs() << " Create a new stub function\n");
          Stubs[Value] = Section.StubOffset;
          uint8_t *StubTargetAddr =
              createStubFunction(Section.Address + Section.StubOffset,
                                 AbiVariant);
          RelocationEntry RE(SectionID, StubTargetAddr - Section.Address,
                             ELF::R_PPC64_ADDR64, Value.Addend);

          // Generates the 64-bits address loads as exemplified in section
          // 4.5.1 in PPC64 ELF ABI.  Note that the relocations need to
          // apply to the low part of the instructions, so we have to update
          // the offset according to the target endianness.
          uint64_t StubRelocOffset = StubTargetAddr - Section.Address;
          if (!IsTargetLittleEndian)
            StubRelocOffset += 2;

          RelocationEntry REhst(SectionID, StubRelocOffset + 0,
                                ELF::R_PPC64_ADDR16_HIGHEST, Value.Addend);
          RelocationEntry REhr(SectionID, StubRelocOffset + 4,
                               ELF::R_PPC64_ADDR16_HIGHER, Value.Addend);
          RelocationEntry REh(SectionID, StubRelocOffset + 12,
                              ELF::R_PPC64_ADDR16_HI, Value.Addend);
          RelocationEntry REl(SectionID, StubRelocOffset + 16,
                              ELF::R_PPC64_ADDR16_LO, Value.Addend);

          if (Value.SymbolName) {
            addRelocationForSymbol(REhst, Value.SymbolName);
            addRelocationForSymbol(REhr, Value.SymbolName);
            addRelocationForSymbol(REh, Value.SymbolName);
            addRelocationForSymbol(REl, Value.SymbolName);
          } else {
            addRelocationForSection(REhst, Value.SectionID);
            addRelocationForSection(REhr, Value.SectionID);
            addRelocationForSection(REh, Value.SectionID);
            addRelocationForSection(REl, Value.SectionID);
          }

          resolveRelocation(Section, Offset,
                            (uint64_t)Section.Address + Section.StubOffset,
                            RelType, 0);
          Section.StubOffset += getMaxStubSize();
        }
        if (SymType == SymbolRef::ST_Unknown) {
          // Restore the TOC for external calls
          if (AbiVariant == 2)
            writeInt32BE(Target + 4, 0xE8410018); // ld r2,28(r1)
          else
            writeInt32BE(Target + 4, 0xE8410028); // ld r2,40(r1)
        }
      }
    } else if (RelType == ELF::R_PPC64_TOC16 ||
               RelType == ELF::R_PPC64_TOC16_DS ||
               RelType == ELF::R_PPC64_TOC16_LO ||
               RelType == ELF::R_PPC64_TOC16_LO_DS ||
               RelType == ELF::R_PPC64_TOC16_HI ||
               RelType == ELF::R_PPC64_TOC16_HA) {
      // These relocations are supposed to subtract the TOC address from
      // the final value.  This does not fit cleanly into the RuntimeDyld
      // scheme, since there may be *two* sections involved in determining
      // the relocation value (the section of the symbol refered to by the
      // relocation, and the TOC section associated with the current module).
      //
      // Fortunately, these relocations are currently only ever generated
      // refering to symbols that themselves reside in the TOC, which means
      // that the two sections are actually the same.  Thus they cancel out
      // and we can immediately resolve the relocation right now.
      switch (RelType) {
      case ELF::R_PPC64_TOC16: RelType = ELF::R_PPC64_ADDR16; break;
      case ELF::R_PPC64_TOC16_DS: RelType = ELF::R_PPC64_ADDR16_DS; break;
      case ELF::R_PPC64_TOC16_LO: RelType = ELF::R_PPC64_ADDR16_LO; break;
      case ELF::R_PPC64_TOC16_LO_DS: RelType = ELF::R_PPC64_ADDR16_LO_DS; break;
      case ELF::R_PPC64_TOC16_HI: RelType = ELF::R_PPC64_ADDR16_HI; break;
      case ELF::R_PPC64_TOC16_HA: RelType = ELF::R_PPC64_ADDR16_HA; break;
      default: llvm_unreachable("Wrong relocation type.");
      }

      RelocationValueRef TOCValue;
      findPPC64TOCSection(Obj, ObjSectionToID, TOCValue);
      if (Value.SymbolName || Value.SectionID != TOCValue.SectionID)
        llvm_unreachable("Unsupported TOC relocation.");
      Value.Addend -= TOCValue.Addend;
      resolveRelocation(Sections[SectionID], Offset, Value.Addend, RelType, 0);
    } else {
      // There are two ways to refer to the TOC address directly: either
      // via a ELF::R_PPC64_TOC relocation (where both symbol and addend are
      // ignored), or via any relocation that refers to the magic ".TOC."
      // symbols (in which case the addend is respected).
      if (RelType == ELF::R_PPC64_TOC) {
        RelType = ELF::R_PPC64_ADDR64;
        findPPC64TOCSection(Obj, ObjSectionToID, Value);
      } else if (TargetName == ".TOC.") {
        findPPC64TOCSection(Obj, ObjSectionToID, Value);
        Value.Addend += Addend;
      }

      RelocationEntry RE(SectionID, Offset, RelType, Value.Addend);

      if (Value.SymbolName)
        addRelocationForSymbol(RE, Value.SymbolName);
      else
        addRelocationForSection(RE, Value.SectionID);
    }
  } else if (Arch == Triple::systemz &&
             (RelType == ELF::R_390_PLT32DBL || RelType == ELF::R_390_GOTENT)) {
    // Create function stubs for both PLT and GOT references, regardless of
    // whether the GOT reference is to data or code.  The stub contains the
    // full address of the symbol, as needed by GOT references, and the
    // executable part only adds an overhead of 8 bytes.
    //
    // We could try to conserve space by allocating the code and data
    // parts of the stub separately.  However, as things stand, we allocate
    // a stub for every relocation, so using a GOT in JIT code should be
    // no less space efficient than using an explicit constant pool.
    DEBUG(dbgs() << "\t\tThis is a SystemZ indirect relocation.");
    SectionEntry &Section = Sections[SectionID];

    // Look for an existing stub.
    StubMap::const_iterator i = Stubs.find(Value);
    uintptr_t StubAddress;
    if (i != Stubs.end()) {
      StubAddress = uintptr_t(Section.Address) + i->second;
      DEBUG(dbgs() << " Stub function found\n");
    } else {
      // Create a new stub function.
      DEBUG(dbgs() << " Create a new stub function\n");

      uintptr_t BaseAddress = uintptr_t(Section.Address);
      uintptr_t StubAlignment = getStubAlignment();
      StubAddress = (BaseAddress + Section.StubOffset + StubAlignment - 1) &
                    -StubAlignment;
      unsigned StubOffset = StubAddress - BaseAddress;

      Stubs[Value] = StubOffset;
      createStubFunction((uint8_t *)StubAddress);
      RelocationEntry RE(SectionID, StubOffset + 8, ELF::R_390_64,
                         Value.Offset);
      if (Value.SymbolName)
        addRelocationForSymbol(RE, Value.SymbolName);
      else
        addRelocationForSection(RE, Value.SectionID);
      Section.StubOffset = StubOffset + getMaxStubSize();
    }

    if (RelType == ELF::R_390_GOTENT)
      resolveRelocation(Section, Offset, StubAddress + 8, ELF::R_390_PC32DBL,
                        Addend);
    else
      resolveRelocation(Section, Offset, StubAddress, RelType, Addend);
  } else if (Arch == Triple::x86_64) {
    if (RelType == ELF::R_X86_64_PLT32) {
      // The way the PLT relocations normally work is that the linker allocates
      // the
      // PLT and this relocation makes a PC-relative call into the PLT.  The PLT
      // entry will then jump to an address provided by the GOT.  On first call,
      // the
      // GOT address will point back into PLT code that resolves the symbol. After
      // the first call, the GOT entry points to the actual function.
      //
      // For local functions we're ignoring all of that here and just replacing
      // the PLT32 relocation type with PC32, which will translate the relocation
      // into a PC-relative call directly to the function. For external symbols we
      // can't be sure the function will be within 2^32 bytes of the call site, so
      // we need to create a stub, which calls into the GOT.  This case is
      // equivalent to the usual PLT implementation except that we use the stub
      // mechanism in RuntimeDyld (which puts stubs at the end of the section)
      // rather than allocating a PLT section.
      if (Value.SymbolName) {
        // This is a call to an external function.
        // Look for an existing stub.
        SectionEntry &Section = Sections[SectionID];
        StubMap::const_iterator i = Stubs.find(Value);
        uintptr_t StubAddress;
        if (i != Stubs.end()) {
        StubAddress = uintptr_t(Section.Address) + i->second;
        DEBUG(dbgs() << " Stub function found\n");
        } else {
        // Create a new stub function (equivalent to a PLT entry).
        DEBUG(dbgs() << " Create a new stub function\n");

        uintptr_t BaseAddress = uintptr_t(Section.Address);
        uintptr_t StubAlignment = getStubAlignment();
        StubAddress = (BaseAddress + Section.StubOffset + StubAlignment - 1) &
                -StubAlignment;
        unsigned StubOffset = StubAddress - BaseAddress;
        Stubs[Value] = StubOffset;
        createStubFunction((uint8_t *)StubAddress);

        // Bump our stub offset counter
        Section.StubOffset = StubOffset + getMaxStubSize();

        // Allocate a GOT Entry
        uint64_t GOTOffset = allocateGOTEntries(SectionID, 1);

        // The load of the GOT address has an addend of -4
        resolveGOTOffsetRelocation(SectionID, StubOffset + 2, GOTOffset - 4);

        // Fill in the value of the symbol we're targeting into the GOT
        addRelocationForSymbol(computeGOTOffsetRE(SectionID,GOTOffset,0,ELF::R_X86_64_64),
          Value.SymbolName);
        }

        // Make the target call a call into the stub table.
        resolveRelocation(Section, Offset, StubAddress, ELF::R_X86_64_PC32,
                Addend);
      } else {
        RelocationEntry RE(SectionID, Offset, ELF::R_X86_64_PC32, Value.Addend,
                  Value.Offset);
        addRelocationForSection(RE, Value.SectionID);
      }
    } else if (RelType == ELF::R_X86_64_GOTPCREL) {
      uint64_t GOTOffset = allocateGOTEntries(SectionID, 1);
      resolveGOTOffsetRelocation(SectionID, Offset, GOTOffset + Addend);

      // Fill in the value of the symbol we're targeting into the GOT
      RelocationEntry RE = computeGOTOffsetRE(SectionID, GOTOffset, Value.Offset, ELF::R_X86_64_64);
      if (Value.SymbolName)
        addRelocationForSymbol(RE, Value.SymbolName);
      else
        addRelocationForSection(RE, Value.SectionID);
    } else if (RelType == ELF::R_X86_64_PC32) {
      Value.Addend += support::ulittle32_t::ref(computePlaceholderAddress(SectionID, Offset));
      processSimpleRelocation(SectionID, Offset, RelType, Value);
    } else if (RelType == ELF::R_X86_64_PC64) {
      Value.Addend += support::ulittle64_t::ref(computePlaceholderAddress(SectionID, Offset));
      processSimpleRelocation(SectionID, Offset, RelType, Value);
    } else {
      processSimpleRelocation(SectionID, Offset, RelType, Value);
    }
  } else {
    if (Arch == Triple::x86) {
      Value.Addend += support::ulittle32_t::ref(computePlaceholderAddress(SectionID, Offset));
    }
    processSimpleRelocation(SectionID, Offset, RelType, Value);
  }
  return ++RelI;
}

size_t RuntimeDyldELF::getGOTEntrySize() {
  // We don't use the GOT in all of these cases, but it's essentially free
  // to put them all here.
  size_t Result = 0;
  switch (Arch) {
  case Triple::x86_64:
  case Triple::aarch64:
  case Triple::aarch64_be:
  case Triple::ppc64:
  case Triple::ppc64le:
  case Triple::systemz:
    Result = sizeof(uint64_t);
    break;
  case Triple::x86:
  case Triple::arm:
  case Triple::thumb:
    Result = sizeof(uint32_t);
    break;
  case Triple::mips:
  case Triple::mipsel:
  case Triple::mips64:
  case Triple::mips64el:
    if (IsMipsO32ABI)
      Result = sizeof(uint32_t);
    else if (IsMipsN64ABI)
      Result = sizeof(uint64_t);
    else
      llvm_unreachable("Mips ABI not handled");
    break;
  default:
    llvm_unreachable("Unsupported CPU type!");
  }
  return Result;
}

uint64_t RuntimeDyldELF::allocateGOTEntries(unsigned SectionID, unsigned no)
{
  (void)SectionID; // The GOT Section is the same for all section in the object file
  if (GOTSectionID == 0) {
    GOTSectionID = Sections.size();
    // Reserve a section id. We'll allocate the section later
    // once we know the total size
    Sections.push_back(SectionEntry(".got", 0, 0, 0));
  }
  uint64_t StartOffset = CurrentGOTIndex * getGOTEntrySize();
  CurrentGOTIndex += no;
  return StartOffset;
}

void RuntimeDyldELF::resolveGOTOffsetRelocation(unsigned SectionID, uint64_t Offset, uint64_t GOTOffset)
{
  // Fill in the relative address of the GOT Entry into the stub
  RelocationEntry GOTRE(SectionID, Offset, ELF::R_X86_64_PC32, GOTOffset);
  addRelocationForSection(GOTRE, GOTSectionID);
}

RelocationEntry RuntimeDyldELF::computeGOTOffsetRE(unsigned SectionID, uint64_t GOTOffset, uint64_t SymbolOffset,
                                                   uint32_t Type)
{
  (void)SectionID; // The GOT Section is the same for all section in the object file
  return RelocationEntry(GOTSectionID, GOTOffset, Type, SymbolOffset);
}

void RuntimeDyldELF::finalizeLoad(const ObjectFile &Obj,
                                  ObjSectionToIDMap &SectionMap) {
  // If necessary, allocate the global offset table
  if (GOTSectionID != 0) {
    // Allocate memory for the section
    size_t TotalSize = CurrentGOTIndex * getGOTEntrySize();
    uint8_t *Addr = MemMgr.allocateDataSection(TotalSize, getGOTEntrySize(),
                                                GOTSectionID, ".got", false);
    if (!Addr)
      report_fatal_error("Unable to allocate memory for GOT!");

    Sections[GOTSectionID] = SectionEntry(".got", Addr, TotalSize, 0);

    if (Checker)
      Checker->registerSection(Obj.getFileName(), GOTSectionID);

    // For now, initialize all GOT entries to zero.  We'll fill them in as
    // needed when GOT-based relocations are applied.
    memset(Addr, 0, TotalSize);
    if (IsMipsN64ABI) {
      // To correctly resolve Mips GOT relocations, we need a mapping from
      // object's sections to GOTs.
      for (section_iterator SI = Obj.section_begin(), SE = Obj.section_end();
           SI != SE; ++SI) {
        if (SI->relocation_begin() != SI->relocation_end()) {
          section_iterator RelocatedSection = SI->getRelocatedSection();
          ObjSectionToIDMap::iterator i = SectionMap.find(*RelocatedSection);
          assert (i != SectionMap.end());
          SectionToGOTMap[i->second] = GOTSectionID;
        }
      }
      GOTSymbolOffsets.clear();
    }
  }

  // Look for and record the EH frame section.
  ObjSectionToIDMap::iterator i, e;
  for (i = SectionMap.begin(), e = SectionMap.end(); i != e; ++i) {
    const SectionRef &Section = i->first;
    StringRef Name;
    Section.getName(Name);
    if (Name == ".eh_frame") {
      UnregisteredEHFrameSections.push_back(i->second);
      break;
    }
  }

  GOTSectionID = 0;
  CurrentGOTIndex = 0;
}

bool RuntimeDyldELF::isCompatibleFile(const object::ObjectFile &Obj) const {
  return Obj.isELF();
}

} // namespace llvm
