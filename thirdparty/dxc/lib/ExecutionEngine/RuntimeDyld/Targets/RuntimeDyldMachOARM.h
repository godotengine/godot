//===----- RuntimeDyldMachOARM.h ---- MachO/ARM specific code. ----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_EXECUTIONENGINE_RUNTIMEDYLD_TARGETS_RUNTIMEDYLDMACHOARM_H
#define LLVM_LIB_EXECUTIONENGINE_RUNTIMEDYLD_TARGETS_RUNTIMEDYLDMACHOARM_H

#include "../RuntimeDyldMachO.h"

#define DEBUG_TYPE "dyld"

namespace llvm {

class RuntimeDyldMachOARM
    : public RuntimeDyldMachOCRTPBase<RuntimeDyldMachOARM> {
private:
  typedef RuntimeDyldMachOCRTPBase<RuntimeDyldMachOARM> ParentT;

public:

  typedef uint32_t TargetPtrT;

  RuntimeDyldMachOARM(RuntimeDyld::MemoryManager &MM,
                      RuntimeDyld::SymbolResolver &Resolver)
    : RuntimeDyldMachOCRTPBase(MM, Resolver) {}

  unsigned getMaxStubSize() override { return 8; }

  unsigned getStubAlignment() override { return 4; }

  int64_t decodeAddend(const RelocationEntry &RE) const {
    const SectionEntry &Section = Sections[RE.SectionID];
    uint8_t *LocalAddress = Section.Address + RE.Offset;

    switch (RE.RelType) {
      default:
        return memcpyAddend(RE);
      case MachO::ARM_RELOC_BR24: {
        uint32_t Temp = readBytesUnaligned(LocalAddress, 4);
        Temp &= 0x00ffffff; // Mask out the opcode.
        // Now we've got the shifted immediate, shift by 2, sign extend and ret.
        return SignExtend32<26>(Temp << 2);
      }
    }
  }

  relocation_iterator
  processRelocationRef(unsigned SectionID, relocation_iterator RelI,
                       const ObjectFile &BaseObjT,
                       ObjSectionToIDMap &ObjSectionToID,
                       StubMap &Stubs) override {
    const MachOObjectFile &Obj =
        static_cast<const MachOObjectFile &>(BaseObjT);
    MachO::any_relocation_info RelInfo =
        Obj.getRelocation(RelI->getRawDataRefImpl());
    uint32_t RelType = Obj.getAnyRelocationType(RelInfo);

    if (Obj.isRelocationScattered(RelInfo)) {
      if (RelType == MachO::ARM_RELOC_HALF_SECTDIFF)
        return processHALFSECTDIFFRelocation(SectionID, RelI, Obj,
                                             ObjSectionToID);
      else
        return ++++RelI;
    }

    RelocationEntry RE(getRelocationEntry(SectionID, Obj, RelI));
    RE.Addend = decodeAddend(RE);
    RelocationValueRef Value(
        getRelocationValueRef(Obj, RelI, RE, ObjSectionToID));

    if (RE.IsPCRel)
      makeValueAddendPCRel(Value, RelI, 8);

    if ((RE.RelType & 0xf) == MachO::ARM_RELOC_BR24)
      processBranchRelocation(RE, Value, Stubs);
    else {
      RE.Addend = Value.Offset;
      if (Value.SymbolName)
        addRelocationForSymbol(RE, Value.SymbolName);
      else
        addRelocationForSection(RE, Value.SectionID);
    }

    return ++RelI;
  }

  void resolveRelocation(const RelocationEntry &RE, uint64_t Value) override {
    DEBUG(dumpRelocationToResolve(RE, Value));
    const SectionEntry &Section = Sections[RE.SectionID];
    uint8_t *LocalAddress = Section.Address + RE.Offset;

    // If the relocation is PC-relative, the value to be encoded is the
    // pointer difference.
    if (RE.IsPCRel) {
      uint64_t FinalAddress = Section.LoadAddress + RE.Offset;
      Value -= FinalAddress;
      // ARM PCRel relocations have an effective-PC offset of two instructions
      // (four bytes in Thumb mode, 8 bytes in ARM mode).
      // FIXME: For now, assume ARM mode.
      Value -= 8;
    }

    switch (RE.RelType) {
    default:
      llvm_unreachable("Invalid relocation type!");
    case MachO::ARM_RELOC_VANILLA:
      writeBytesUnaligned(Value + RE.Addend, LocalAddress, 1 << RE.Size);
      break;
    case MachO::ARM_RELOC_BR24: {
      // Mask the value into the target address. We know instructions are
      // 32-bit aligned, so we can do it all at once.
      Value += RE.Addend;
      // The low two bits of the value are not encoded.
      Value >>= 2;
      // Mask the value to 24 bits.
      uint64_t FinalValue = Value & 0xffffff;
      // FIXME: If the destination is a Thumb function (and the instruction
      // is a non-predicated BL instruction), we need to change it to a BLX
      // instruction instead.

      // Insert the value into the instruction.
      uint32_t Temp = readBytesUnaligned(LocalAddress, 4);
      writeBytesUnaligned((Temp & ~0xffffff) | FinalValue, LocalAddress, 4);

      break;
    }
    case MachO::ARM_RELOC_HALF_SECTDIFF: {
      uint64_t SectionABase = Sections[RE.Sections.SectionA].LoadAddress;
      uint64_t SectionBBase = Sections[RE.Sections.SectionB].LoadAddress;
      assert((Value == SectionABase || Value == SectionBBase) &&
             "Unexpected HALFSECTDIFF relocation value.");
      Value = SectionABase - SectionBBase + RE.Addend;
      if (RE.Size & 0x1) // :upper16:
        Value = (Value >> 16);
      Value &= 0xffff;

      uint32_t Insn = readBytesUnaligned(LocalAddress, 4);
      Insn = (Insn & 0xfff0f000) | ((Value & 0xf000) << 4) | (Value & 0x0fff);
      writeBytesUnaligned(Insn, LocalAddress, 4);
      break;
    }

    case MachO::ARM_THUMB_RELOC_BR22:
    case MachO::ARM_THUMB_32BIT_BRANCH:
    case MachO::ARM_RELOC_HALF:
    case MachO::ARM_RELOC_PAIR:
    case MachO::ARM_RELOC_SECTDIFF:
    case MachO::ARM_RELOC_LOCAL_SECTDIFF:
    case MachO::ARM_RELOC_PB_LA_PTR:
      Error("Relocation type not implemented yet!");
      return;
    }
  }

  void finalizeSection(const ObjectFile &Obj, unsigned SectionID,
                       const SectionRef &Section) {
    StringRef Name;
    Section.getName(Name);

    if (Name == "__nl_symbol_ptr")
      populateIndirectSymbolPointersSection(cast<MachOObjectFile>(Obj),
                                            Section, SectionID);
  }

private:

  void processBranchRelocation(const RelocationEntry &RE,
                               const RelocationValueRef &Value,
                               StubMap &Stubs) {
    // This is an ARM branch relocation, need to use a stub function.
    // Look up for existing stub.
    SectionEntry &Section = Sections[RE.SectionID];
    RuntimeDyldMachO::StubMap::const_iterator i = Stubs.find(Value);
    uint8_t *Addr;
    if (i != Stubs.end()) {
      Addr = Section.Address + i->second;
    } else {
      // Create a new stub function.
      Stubs[Value] = Section.StubOffset;
      uint8_t *StubTargetAddr =
          createStubFunction(Section.Address + Section.StubOffset);
      RelocationEntry StubRE(RE.SectionID, StubTargetAddr - Section.Address,
                             MachO::GENERIC_RELOC_VANILLA, Value.Offset, false,
                             2);
      if (Value.SymbolName)
        addRelocationForSymbol(StubRE, Value.SymbolName);
      else
        addRelocationForSection(StubRE, Value.SectionID);
      Addr = Section.Address + Section.StubOffset;
      Section.StubOffset += getMaxStubSize();
    }
    RelocationEntry TargetRE(RE.SectionID, RE.Offset, RE.RelType, 0,
                             RE.IsPCRel, RE.Size);
    resolveRelocation(TargetRE, (uint64_t)Addr);
  }

  relocation_iterator
  processHALFSECTDIFFRelocation(unsigned SectionID, relocation_iterator RelI,
                                const ObjectFile &BaseTObj,
                                ObjSectionToIDMap &ObjSectionToID) {
    const MachOObjectFile &MachO =
        static_cast<const MachOObjectFile&>(BaseTObj);
    MachO::any_relocation_info RE =
        MachO.getRelocation(RelI->getRawDataRefImpl());


    // For a half-diff relocation the length bits actually record whether this
    // is a movw/movt, and whether this is arm or thumb.
    // Bit 0 indicates movw (b0 == 0) or movt (b0 == 1).
    // Bit 1 indicates arm (b1 == 0) or thumb (b1 == 1).
    unsigned HalfDiffKindBits = MachO.getAnyRelocationLength(RE);
    if (HalfDiffKindBits & 0x2)
      llvm_unreachable("Thumb not yet supported.");

    SectionEntry &Section = Sections[SectionID];
    uint32_t RelocType = MachO.getAnyRelocationType(RE);
    bool IsPCRel = MachO.getAnyRelocationPCRel(RE);
    uint64_t Offset = RelI->getOffset();
    uint8_t *LocalAddress = Section.Address + Offset;
    int64_t Immediate = readBytesUnaligned(LocalAddress, 4); // Copy the whole instruction out.
    Immediate = ((Immediate >> 4) & 0xf000) | (Immediate & 0xfff);

    ++RelI;
    MachO::any_relocation_info RE2 =
      MachO.getRelocation(RelI->getRawDataRefImpl());
    uint32_t AddrA = MachO.getScatteredRelocationValue(RE);
    section_iterator SAI = getSectionByAddress(MachO, AddrA);
    assert(SAI != MachO.section_end() && "Can't find section for address A");
    uint64_t SectionABase = SAI->getAddress();
    uint64_t SectionAOffset = AddrA - SectionABase;
    SectionRef SectionA = *SAI;
    bool IsCode = SectionA.isText();
    uint32_t SectionAID =
        findOrEmitSection(MachO, SectionA, IsCode, ObjSectionToID);

    uint32_t AddrB = MachO.getScatteredRelocationValue(RE2);
    section_iterator SBI = getSectionByAddress(MachO, AddrB);
    assert(SBI != MachO.section_end() && "Can't find section for address B");
    uint64_t SectionBBase = SBI->getAddress();
    uint64_t SectionBOffset = AddrB - SectionBBase;
    SectionRef SectionB = *SBI;
    uint32_t SectionBID =
        findOrEmitSection(MachO, SectionB, IsCode, ObjSectionToID);

    uint32_t OtherHalf = MachO.getAnyRelocationAddress(RE2) & 0xffff;
    unsigned Shift = (HalfDiffKindBits & 0x1) ? 16 : 0;
    uint32_t FullImmVal = (Immediate << Shift) | (OtherHalf << (16 - Shift));
    int64_t Addend = FullImmVal - (AddrA - AddrB);

    // addend = Encoded - Expected
    //        = Encoded - (AddrA - AddrB)

    DEBUG(dbgs() << "Found SECTDIFF: AddrA: " << AddrA << ", AddrB: " << AddrB
                 << ", Addend: " << Addend << ", SectionA ID: " << SectionAID
                 << ", SectionAOffset: " << SectionAOffset
                 << ", SectionB ID: " << SectionBID
                 << ", SectionBOffset: " << SectionBOffset << "\n");
    RelocationEntry R(SectionID, Offset, RelocType, Addend, SectionAID,
                      SectionAOffset, SectionBID, SectionBOffset, IsPCRel,
                      HalfDiffKindBits);

    addRelocationForSection(R, SectionAID);
    addRelocationForSection(R, SectionBID);

    return ++RelI;
  }

};
}

#undef DEBUG_TYPE

#endif
