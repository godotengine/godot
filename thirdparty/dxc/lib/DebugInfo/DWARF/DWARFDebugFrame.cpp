//===-- DWARFDebugFrame.h - Parsing of .debug_frame -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

using namespace llvm;
using namespace dwarf;


/// \brief Abstract frame entry defining the common interface concrete
/// entries implement.
class llvm::FrameEntry {
public:
  enum FrameKind {FK_CIE, FK_FDE};
  FrameEntry(FrameKind K, uint64_t Offset, uint64_t Length)
      : Kind(K), Offset(Offset), Length(Length) {}

  virtual ~FrameEntry() {
  }

  FrameKind getKind() const { return Kind; }
  virtual uint64_t getOffset() const { return Offset; }

  /// \brief Parse and store a sequence of CFI instructions from Data,
  /// starting at *Offset and ending at EndOffset. If everything
  /// goes well, *Offset should be equal to EndOffset when this method
  /// returns. Otherwise, an error occurred.
  virtual void parseInstructions(DataExtractor Data, uint32_t *Offset,
                                 uint32_t EndOffset);

  /// \brief Dump the entry header to the given output stream.
  virtual void dumpHeader(raw_ostream &OS) const = 0;

  /// \brief Dump the entry's instructions to the given output stream.
  virtual void dumpInstructions(raw_ostream &OS) const;

protected:
  const FrameKind Kind;

  /// \brief Offset of this entry in the section.
  uint64_t Offset;

  /// \brief Entry length as specified in DWARF.
  uint64_t Length;

  /// An entry may contain CFI instructions. An instruction consists of an
  /// opcode and an optional sequence of operands.
  typedef std::vector<uint64_t> Operands;
  struct Instruction {
    Instruction(uint8_t Opcode)
      : Opcode(Opcode)
    {}

    uint8_t Opcode;
    Operands Ops;
  };

  std::vector<Instruction> Instructions;

  /// Convenience methods to add a new instruction with the given opcode and
  /// operands to the Instructions vector.
  void addInstruction(uint8_t Opcode) {
    Instructions.push_back(Instruction(Opcode));
  }

  void addInstruction(uint8_t Opcode, uint64_t Operand1) {
    Instructions.push_back(Instruction(Opcode));
    Instructions.back().Ops.push_back(Operand1);
  }

  void addInstruction(uint8_t Opcode, uint64_t Operand1, uint64_t Operand2) {
    Instructions.push_back(Instruction(Opcode));
    Instructions.back().Ops.push_back(Operand1);
    Instructions.back().Ops.push_back(Operand2);
  }
};


// See DWARF standard v3, section 7.23
const uint8_t DWARF_CFI_PRIMARY_OPCODE_MASK = 0xc0;
const uint8_t DWARF_CFI_PRIMARY_OPERAND_MASK = 0x3f;

void FrameEntry::parseInstructions(DataExtractor Data, uint32_t *Offset,
                                   uint32_t EndOffset) {
  while (*Offset < EndOffset) {
    uint8_t Opcode = Data.getU8(Offset);
    // Some instructions have a primary opcode encoded in the top bits.
    uint8_t Primary = Opcode & DWARF_CFI_PRIMARY_OPCODE_MASK;

    if (Primary) {
      // If it's a primary opcode, the first operand is encoded in the bottom
      // bits of the opcode itself.
      uint64_t Op1 = Opcode & DWARF_CFI_PRIMARY_OPERAND_MASK;
      switch (Primary) {
        default: llvm_unreachable("Impossible primary CFI opcode");
        case DW_CFA_advance_loc:
        case DW_CFA_restore:
          addInstruction(Primary, Op1);
          break;
        case DW_CFA_offset:
          addInstruction(Primary, Op1, Data.getULEB128(Offset));
          break;
      }
    } else {
      // Extended opcode - its value is Opcode itself.
      switch (Opcode) {
        default: llvm_unreachable("Invalid extended CFI opcode");
        case DW_CFA_nop:
        case DW_CFA_remember_state:
        case DW_CFA_restore_state:
        case DW_CFA_GNU_window_save:
          // No operands
          addInstruction(Opcode);
          break;
        case DW_CFA_set_loc:
          // Operands: Address
          addInstruction(Opcode, Data.getAddress(Offset));
          break;
        case DW_CFA_advance_loc1:
          // Operands: 1-byte delta
          addInstruction(Opcode, Data.getU8(Offset));
          break;
        case DW_CFA_advance_loc2:
          // Operands: 2-byte delta
          addInstruction(Opcode, Data.getU16(Offset));
          break;
        case DW_CFA_advance_loc4:
          // Operands: 4-byte delta
          addInstruction(Opcode, Data.getU32(Offset));
          break;
        case DW_CFA_restore_extended:
        case DW_CFA_undefined:
        case DW_CFA_same_value:
        case DW_CFA_def_cfa_register:
        case DW_CFA_def_cfa_offset:
          // Operands: ULEB128
          addInstruction(Opcode, Data.getULEB128(Offset));
          break;
        case DW_CFA_def_cfa_offset_sf:
          // Operands: SLEB128
          addInstruction(Opcode, Data.getSLEB128(Offset));
          break;
        case DW_CFA_offset_extended:
        case DW_CFA_register:
        case DW_CFA_def_cfa:
        case DW_CFA_val_offset:
          // Operands: ULEB128, ULEB128
          addInstruction(Opcode, Data.getULEB128(Offset),
                                 Data.getULEB128(Offset));
          break;
        case DW_CFA_offset_extended_sf:
        case DW_CFA_def_cfa_sf:
        case DW_CFA_val_offset_sf:
          // Operands: ULEB128, SLEB128
          addInstruction(Opcode, Data.getULEB128(Offset),
                                 Data.getSLEB128(Offset));
          break;
        case DW_CFA_def_cfa_expression:
        case DW_CFA_expression:
        case DW_CFA_val_expression:
          // TODO: implement this
          report_fatal_error("Values with expressions not implemented yet!");
      }
    }
  }
}

namespace {
/// \brief DWARF Common Information Entry (CIE)
class CIE : public FrameEntry {
public:
  // CIEs (and FDEs) are simply container classes, so the only sensible way to
  // create them is by providing the full parsed contents in the constructor.
  CIE(uint64_t Offset, uint64_t Length, uint8_t Version,
      SmallString<8> Augmentation, uint8_t AddressSize,
      uint8_t SegmentDescriptorSize, uint64_t CodeAlignmentFactor,
      int64_t DataAlignmentFactor, uint64_t ReturnAddressRegister)
      : FrameEntry(FK_CIE, Offset, Length), Version(Version),
        Augmentation(std::move(Augmentation)),
        AddressSize(AddressSize),
        SegmentDescriptorSize(SegmentDescriptorSize),
        CodeAlignmentFactor(CodeAlignmentFactor),
        DataAlignmentFactor(DataAlignmentFactor),
        ReturnAddressRegister(ReturnAddressRegister) {}

  ~CIE() override {}

  uint64_t getCodeAlignmentFactor() const { return CodeAlignmentFactor; }
  int64_t getDataAlignmentFactor() const { return DataAlignmentFactor; }

  void dumpHeader(raw_ostream &OS) const override {
    OS << format("%08x %08x %08x CIE",
                 (uint32_t)Offset, (uint32_t)Length, DW_CIE_ID)
       << "\n";
    OS << format("  Version:               %d\n", Version);
    OS << "  Augmentation:          \"" << Augmentation << "\"\n";
    if (Version >= 4) {
      OS << format("  Address size:          %u\n",
                   (uint32_t)AddressSize);
      OS << format("  Segment desc size:     %u\n",
                   (uint32_t)SegmentDescriptorSize);
    }
    OS << format("  Code alignment factor: %u\n",
                 (uint32_t)CodeAlignmentFactor);
    OS << format("  Data alignment factor: %d\n",
                 (int32_t)DataAlignmentFactor);
    OS << format("  Return address column: %d\n",
                 (int32_t)ReturnAddressRegister);
    OS << "\n";
  }

  static bool classof(const FrameEntry *FE) {
    return FE->getKind() == FK_CIE;
  }

private:
  /// The following fields are defined in section 6.4.1 of the DWARF standard v4
  uint8_t Version;
  SmallString<8> Augmentation;
  uint8_t AddressSize;
  uint8_t SegmentDescriptorSize;
  uint64_t CodeAlignmentFactor;
  int64_t DataAlignmentFactor;
  uint64_t ReturnAddressRegister;
};


/// \brief DWARF Frame Description Entry (FDE)
class FDE : public FrameEntry {
public:
  // Each FDE has a CIE it's "linked to". Our FDE contains is constructed with
  // an offset to the CIE (provided by parsing the FDE header). The CIE itself
  // is obtained lazily once it's actually required.
  FDE(uint64_t Offset, uint64_t Length, int64_t LinkedCIEOffset,
      uint64_t InitialLocation, uint64_t AddressRange,
      CIE *Cie)
      : FrameEntry(FK_FDE, Offset, Length), LinkedCIEOffset(LinkedCIEOffset),
        InitialLocation(InitialLocation), AddressRange(AddressRange),
        LinkedCIE(Cie) {}

  ~FDE() override {}

  CIE *getLinkedCIE() const { return LinkedCIE; }

  void dumpHeader(raw_ostream &OS) const override {
    OS << format("%08x %08x %08x FDE ",
                 (uint32_t)Offset, (uint32_t)Length, (int32_t)LinkedCIEOffset);
    OS << format("cie=%08x pc=%08x...%08x\n",
                 (int32_t)LinkedCIEOffset,
                 (uint32_t)InitialLocation,
                 (uint32_t)InitialLocation + (uint32_t)AddressRange);
  }

  static bool classof(const FrameEntry *FE) {
    return FE->getKind() == FK_FDE;
  }

private:
  /// The following fields are defined in section 6.4.1 of the DWARF standard v3
  uint64_t LinkedCIEOffset;
  uint64_t InitialLocation;
  uint64_t AddressRange;
  CIE *LinkedCIE;
};

/// \brief Types of operands to CF instructions.
enum OperandType {
  OT_Unset,
  OT_None,
  OT_Address,
  OT_Offset,
  OT_FactoredCodeOffset,
  OT_SignedFactDataOffset,
  OT_UnsignedFactDataOffset,
  OT_Register,
  OT_Expression
};

} // end anonymous namespace

/// \brief Initialize the array describing the types of operands.
static ArrayRef<OperandType[2]> getOperandTypes() {
  static OperandType OpTypes[DW_CFA_restore+1][2];

#define DECLARE_OP2(OP, OPTYPE0, OPTYPE1)       \
  do {                                          \
    OpTypes[OP][0] = OPTYPE0;                   \
    OpTypes[OP][1] = OPTYPE1;                   \
  } while (0)
#define DECLARE_OP1(OP, OPTYPE0) DECLARE_OP2(OP, OPTYPE0, OT_None)
#define DECLARE_OP0(OP) DECLARE_OP1(OP, OT_None)

  DECLARE_OP1(DW_CFA_set_loc, OT_Address);
  DECLARE_OP1(DW_CFA_advance_loc, OT_FactoredCodeOffset);
  DECLARE_OP1(DW_CFA_advance_loc1, OT_FactoredCodeOffset);
  DECLARE_OP1(DW_CFA_advance_loc2, OT_FactoredCodeOffset);
  DECLARE_OP1(DW_CFA_advance_loc4, OT_FactoredCodeOffset);
  DECLARE_OP1(DW_CFA_MIPS_advance_loc8, OT_FactoredCodeOffset);
  DECLARE_OP2(DW_CFA_def_cfa, OT_Register, OT_Offset);
  DECLARE_OP2(DW_CFA_def_cfa_sf, OT_Register, OT_SignedFactDataOffset);
  DECLARE_OP1(DW_CFA_def_cfa_register, OT_Register);
  DECLARE_OP1(DW_CFA_def_cfa_offset, OT_Offset);
  DECLARE_OP1(DW_CFA_def_cfa_offset_sf, OT_SignedFactDataOffset);
  DECLARE_OP1(DW_CFA_def_cfa_expression, OT_Expression);
  DECLARE_OP1(DW_CFA_undefined, OT_Register);
  DECLARE_OP1(DW_CFA_same_value, OT_Register);
  DECLARE_OP2(DW_CFA_offset, OT_Register, OT_UnsignedFactDataOffset);
  DECLARE_OP2(DW_CFA_offset_extended, OT_Register, OT_UnsignedFactDataOffset);
  DECLARE_OP2(DW_CFA_offset_extended_sf, OT_Register, OT_SignedFactDataOffset);
  DECLARE_OP2(DW_CFA_val_offset, OT_Register, OT_UnsignedFactDataOffset);
  DECLARE_OP2(DW_CFA_val_offset_sf, OT_Register, OT_SignedFactDataOffset);
  DECLARE_OP2(DW_CFA_register, OT_Register, OT_Register);
  DECLARE_OP2(DW_CFA_expression, OT_Register, OT_Expression);
  DECLARE_OP2(DW_CFA_val_expression, OT_Register, OT_Expression);
  DECLARE_OP1(DW_CFA_restore, OT_Register);
  DECLARE_OP1(DW_CFA_restore_extended, OT_Register);
  DECLARE_OP0(DW_CFA_remember_state);
  DECLARE_OP0(DW_CFA_restore_state);
  DECLARE_OP0(DW_CFA_GNU_window_save);
  DECLARE_OP1(DW_CFA_GNU_args_size, OT_Offset);
  DECLARE_OP0(DW_CFA_nop);

#undef DECLARE_OP0
#undef DECLARE_OP1
#undef DECLARE_OP2
  return ArrayRef<OperandType[2]>(&OpTypes[0], DW_CFA_restore+1);
}

static ArrayRef<OperandType[2]> OpTypes = getOperandTypes();

/// \brief Print \p Opcode's operand number \p OperandIdx which has
/// value \p Operand.
static void printOperand(raw_ostream &OS, uint8_t Opcode, unsigned OperandIdx,
                         uint64_t Operand, uint64_t CodeAlignmentFactor,
                         int64_t DataAlignmentFactor) {
  assert(OperandIdx < 2);
  OperandType Type = OpTypes[Opcode][OperandIdx];

  switch (Type) {
  case OT_Unset:
    OS << " Unsupported " << (OperandIdx ? "second" : "first") << " operand to";
    if (const char *OpcodeName = CallFrameString(Opcode))
      OS << " " << OpcodeName;
    else
      OS << format(" Opcode %x",  Opcode);
    break;
  case OT_None:
    break;
  case OT_Address:
    OS << format(" %" PRIx64, Operand);
    break;
  case OT_Offset:
    // The offsets are all encoded in a unsigned form, but in practice
    // consumers use them signed. It's most certainly legacy due to
    // the lack of signed variants in the first Dwarf standards.
    OS << format(" %+" PRId64, int64_t(Operand));
    break;
  case OT_FactoredCodeOffset: // Always Unsigned
    if (CodeAlignmentFactor)
      OS << format(" %" PRId64, Operand * CodeAlignmentFactor);
    else
      OS << format(" %" PRId64 "*code_alignment_factor" , Operand);
    break;
  case OT_SignedFactDataOffset:
    if (DataAlignmentFactor)
      OS << format(" %" PRId64, int64_t(Operand) * DataAlignmentFactor);
    else
      OS << format(" %" PRId64 "*data_alignment_factor" , int64_t(Operand));
    break;
  case OT_UnsignedFactDataOffset:
    if (DataAlignmentFactor)
      OS << format(" %" PRId64, Operand * DataAlignmentFactor);
    else
      OS << format(" %" PRId64 "*data_alignment_factor" , Operand);
    break;
  case OT_Register:
    OS << format(" reg%" PRId64, Operand);
    break;
  case OT_Expression:
    OS << " expression";
    break;
  }
}

void FrameEntry::dumpInstructions(raw_ostream &OS) const {
  uint64_t CodeAlignmentFactor = 0;
  int64_t DataAlignmentFactor = 0;
  const CIE *Cie = dyn_cast<CIE>(this);

  if (!Cie)
    Cie = cast<FDE>(this)->getLinkedCIE();
  if (Cie) {
    CodeAlignmentFactor = Cie->getCodeAlignmentFactor();
    DataAlignmentFactor = Cie->getDataAlignmentFactor();
  }

  for (const auto &Instr : Instructions) {
    uint8_t Opcode = Instr.Opcode;
    if (Opcode & DWARF_CFI_PRIMARY_OPCODE_MASK)
      Opcode &= DWARF_CFI_PRIMARY_OPCODE_MASK;
    OS << "  " << CallFrameString(Opcode) << ":";
    for (unsigned i = 0; i < Instr.Ops.size(); ++i)
      printOperand(OS, Opcode, i, Instr.Ops[i], CodeAlignmentFactor,
                   DataAlignmentFactor);
    OS << '\n';
  }
}

DWARFDebugFrame::DWARFDebugFrame() {
}

DWARFDebugFrame::~DWARFDebugFrame() {
}

static void LLVM_ATTRIBUTE_UNUSED dumpDataAux(DataExtractor Data,
                                              uint32_t Offset, int Length) {
  errs() << "DUMP: ";
  for (int i = 0; i < Length; ++i) {
    uint8_t c = Data.getU8(&Offset);
    errs().write_hex(c); errs() << " ";
  }
  errs() << "\n";
}


void DWARFDebugFrame::parse(DataExtractor Data) {
  uint32_t Offset = 0;
  DenseMap<uint32_t, CIE *> CIEs;

  while (Data.isValidOffset(Offset)) {
    uint32_t StartOffset = Offset;

    bool IsDWARF64 = false;
    uint64_t Length = Data.getU32(&Offset);
    uint64_t Id;

    if (Length == UINT32_MAX) {
      // DWARF-64 is distinguished by the first 32 bits of the initial length
      // field being 0xffffffff. Then, the next 64 bits are the actual entry
      // length.
      IsDWARF64 = true;
      Length = Data.getU64(&Offset);
    }

    // At this point, Offset points to the next field after Length.
    // Length is the structure size excluding itself. Compute an offset one
    // past the end of the structure (needed to know how many instructions to
    // read).
    // TODO: For honest DWARF64 support, DataExtractor will have to treat
    //       offset_ptr as uint64_t*
    uint32_t EndStructureOffset = Offset + static_cast<uint32_t>(Length);

    // The Id field's size depends on the DWARF format
    Id = Data.getUnsigned(&Offset, IsDWARF64 ? 8 : 4);
    bool IsCIE = ((IsDWARF64 && Id == DW64_CIE_ID) || Id == DW_CIE_ID);

    if (IsCIE) {
      uint8_t Version = Data.getU8(&Offset);
      const char *Augmentation = Data.getCStr(&Offset);
      uint8_t AddressSize = Version < 4 ? Data.getAddressSize() : Data.getU8(&Offset);
      Data.setAddressSize(AddressSize);
      uint8_t SegmentDescriptorSize = Version < 4 ? 0 : Data.getU8(&Offset);
      uint64_t CodeAlignmentFactor = Data.getULEB128(&Offset);
      int64_t DataAlignmentFactor = Data.getSLEB128(&Offset);
      uint64_t ReturnAddressRegister = Data.getULEB128(&Offset);

      auto Cie = make_unique<CIE>(StartOffset, Length, Version,
                                  StringRef(Augmentation), AddressSize,
                                  SegmentDescriptorSize, CodeAlignmentFactor,
                                  DataAlignmentFactor, ReturnAddressRegister);
      CIEs[StartOffset] = Cie.get();
      Entries.emplace_back(std::move(Cie));
    } else {
      // FDE
      uint64_t CIEPointer = Id;
      uint64_t InitialLocation = Data.getAddress(&Offset);
      uint64_t AddressRange = Data.getAddress(&Offset);

      Entries.emplace_back(new FDE(StartOffset, Length, CIEPointer,
                                   InitialLocation, AddressRange,
                                   CIEs[CIEPointer]));
    }

    Entries.back()->parseInstructions(Data, &Offset, EndStructureOffset);

    if (Offset != EndStructureOffset) {
      std::string Str;
      raw_string_ostream OS(Str);
      OS << format("Parsing entry instructions at %lx failed", StartOffset);
      report_fatal_error(Str);
    }
  }
}


void DWARFDebugFrame::dump(raw_ostream &OS) const {
  OS << "\n";
  for (const auto &Entry : Entries) {
    Entry->dumpHeader(OS);
    Entry->dumpInstructions(OS);
    OS << "\n";
  }
}

