//===- lib/MC/MCDwarf.cpp - MCDwarf implementation ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCDwarf.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/config.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

// Given a special op, return the address skip amount (in units of
// DWARF2_LINE_MIN_INSN_LENGTH.
#define SPECIAL_ADDR(op) (((op) - DWARF2_LINE_OPCODE_BASE)/DWARF2_LINE_RANGE)

// The maximum address skip amount that can be encoded with a special op.
#define MAX_SPECIAL_ADDR_DELTA         SPECIAL_ADDR(255)

// First special line opcode - leave room for the standard opcodes.
// Note: If you want to change this, you'll have to update the
// "standard_opcode_lengths" table that is emitted in DwarfFileTable::Emit().
#define DWARF2_LINE_OPCODE_BASE         13

// Minimum line offset in a special line info. opcode.  This value
// was chosen to give a reasonable range of values.
#define DWARF2_LINE_BASE                -5

// Range of line offsets in a special line info. opcode.
#define DWARF2_LINE_RANGE               14

static inline uint64_t ScaleAddrDelta(MCContext &Context, uint64_t AddrDelta) {
  unsigned MinInsnLength = Context.getAsmInfo()->getMinInstAlignment();
  if (MinInsnLength == 1)
    return AddrDelta;
  if (AddrDelta % MinInsnLength != 0) {
    // TODO: report this error, but really only once.
    ;
  }
  return AddrDelta / MinInsnLength;
}

//
// This is called when an instruction is assembled into the specified section
// and if there is information from the last .loc directive that has yet to have
// a line entry made for it is made.
//
void MCLineEntry::Make(MCObjectStreamer *MCOS, MCSection *Section) {
  if (!MCOS->getContext().getDwarfLocSeen())
    return;

  // Create a symbol at in the current section for use in the line entry.
  MCSymbol *LineSym = MCOS->getContext().createTempSymbol();
  // Set the value of the symbol to use for the MCLineEntry.
  MCOS->EmitLabel(LineSym);

  // Get the current .loc info saved in the context.
  const MCDwarfLoc &DwarfLoc = MCOS->getContext().getCurrentDwarfLoc();

  // Create a (local) line entry with the symbol and the current .loc info.
  MCLineEntry LineEntry(LineSym, DwarfLoc);

  // clear DwarfLocSeen saying the current .loc info is now used.
  MCOS->getContext().clearDwarfLocSeen();

  // Add the line entry to this section's entries.
  MCOS->getContext()
      .getMCDwarfLineTable(MCOS->getContext().getDwarfCompileUnitID())
      .getMCLineSections()
      .addLineEntry(LineEntry, Section);
}

//
// This helper routine returns an expression of End - Start + IntVal .
//
static inline const MCExpr *MakeStartMinusEndExpr(const MCStreamer &MCOS,
                                                  const MCSymbol &Start,
                                                  const MCSymbol &End,
                                                  int IntVal) {
  MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;
  const MCExpr *Res =
    MCSymbolRefExpr::create(&End, Variant, MCOS.getContext());
  const MCExpr *RHS =
    MCSymbolRefExpr::create(&Start, Variant, MCOS.getContext());
  const MCExpr *Res1 =
    MCBinaryExpr::create(MCBinaryExpr::Sub, Res, RHS, MCOS.getContext());
  const MCExpr *Res2 =
    MCConstantExpr::create(IntVal, MCOS.getContext());
  const MCExpr *Res3 =
    MCBinaryExpr::create(MCBinaryExpr::Sub, Res1, Res2, MCOS.getContext());
  return Res3;
}

//
// This emits the Dwarf line table for the specified section from the entries
// in the LineSection.
//
static inline void
EmitDwarfLineTable(MCObjectStreamer *MCOS, MCSection *Section,
                   const MCLineSection::MCLineEntryCollection &LineEntries) {
  unsigned FileNum = 1;
  unsigned LastLine = 1;
  unsigned Column = 0;
  unsigned Flags = DWARF2_LINE_DEFAULT_IS_STMT ? DWARF2_FLAG_IS_STMT : 0;
  unsigned Isa = 0;
  unsigned Discriminator = 0;
  MCSymbol *LastLabel = nullptr;

  // Loop through each MCLineEntry and encode the dwarf line number table.
  for (auto it = LineEntries.begin(),
            ie = LineEntries.end();
       it != ie; ++it) {

    if (FileNum != it->getFileNum()) {
      FileNum = it->getFileNum();
      MCOS->EmitIntValue(dwarf::DW_LNS_set_file, 1);
      MCOS->EmitULEB128IntValue(FileNum);
    }
    if (Column != it->getColumn()) {
      Column = it->getColumn();
      MCOS->EmitIntValue(dwarf::DW_LNS_set_column, 1);
      MCOS->EmitULEB128IntValue(Column);
    }
    if (Discriminator != it->getDiscriminator()) {
      Discriminator = it->getDiscriminator();
      unsigned Size = getULEB128Size(Discriminator);
      MCOS->EmitIntValue(dwarf::DW_LNS_extended_op, 1);
      MCOS->EmitULEB128IntValue(Size + 1);
      MCOS->EmitIntValue(dwarf::DW_LNE_set_discriminator, 1);
      MCOS->EmitULEB128IntValue(Discriminator);
    }
    if (Isa != it->getIsa()) {
      Isa = it->getIsa();
      MCOS->EmitIntValue(dwarf::DW_LNS_set_isa, 1);
      MCOS->EmitULEB128IntValue(Isa);
    }
    if ((it->getFlags() ^ Flags) & DWARF2_FLAG_IS_STMT) {
      Flags = it->getFlags();
      MCOS->EmitIntValue(dwarf::DW_LNS_negate_stmt, 1);
    }
    if (it->getFlags() & DWARF2_FLAG_BASIC_BLOCK)
      MCOS->EmitIntValue(dwarf::DW_LNS_set_basic_block, 1);
    if (it->getFlags() & DWARF2_FLAG_PROLOGUE_END)
      MCOS->EmitIntValue(dwarf::DW_LNS_set_prologue_end, 1);
    if (it->getFlags() & DWARF2_FLAG_EPILOGUE_BEGIN)
      MCOS->EmitIntValue(dwarf::DW_LNS_set_epilogue_begin, 1);

    int64_t LineDelta = static_cast<int64_t>(it->getLine()) - LastLine;
    MCSymbol *Label = it->getLabel();

    // At this point we want to emit/create the sequence to encode the delta in
    // line numbers and the increment of the address from the previous Label
    // and the current Label.
    const MCAsmInfo *asmInfo = MCOS->getContext().getAsmInfo();
    MCOS->EmitDwarfAdvanceLineAddr(LineDelta, LastLabel, Label,
                                   asmInfo->getPointerSize());

    LastLine = it->getLine();
    LastLabel = Label;
  }

  // Emit a DW_LNE_end_sequence for the end of the section.
  // Use the section end label to compute the address delta and use INT64_MAX
  // as the line delta which is the signal that this is actually a
  // DW_LNE_end_sequence.
  MCSymbol *SectionEnd = MCOS->endSection(Section);

  // Switch back the dwarf line section, in case endSection had to switch the
  // section.
  MCContext &Ctx = MCOS->getContext();
  MCOS->SwitchSection(Ctx.getObjectFileInfo()->getDwarfLineSection());

  const MCAsmInfo *AsmInfo = Ctx.getAsmInfo();
  MCOS->EmitDwarfAdvanceLineAddr(INT64_MAX, LastLabel, SectionEnd,
                                 AsmInfo->getPointerSize());
}

//
// This emits the Dwarf file and the line tables.
//
void MCDwarfLineTable::Emit(MCObjectStreamer *MCOS) {
  MCContext &context = MCOS->getContext();

  auto &LineTables = context.getMCDwarfLineTables();

  // Bail out early so we don't switch to the debug_line section needlessly and
  // in doing so create an unnecessary (if empty) section.
  if (LineTables.empty())
    return;

  // Switch to the section where the table will be emitted into.
  MCOS->SwitchSection(context.getObjectFileInfo()->getDwarfLineSection());

  // Handle the rest of the Compile Units.
  for (const auto &CUIDTablePair : LineTables)
    CUIDTablePair.second.EmitCU(MCOS);
}

void MCDwarfDwoLineTable::Emit(MCStreamer &MCOS) const {
  MCOS.EmitLabel(Header.Emit(&MCOS, None).second);
}

std::pair<MCSymbol *, MCSymbol *> MCDwarfLineTableHeader::Emit(MCStreamer *MCOS) const {
  static const char StandardOpcodeLengths[] = {
      0, // length of DW_LNS_copy
      1, // length of DW_LNS_advance_pc
      1, // length of DW_LNS_advance_line
      1, // length of DW_LNS_set_file
      1, // length of DW_LNS_set_column
      0, // length of DW_LNS_negate_stmt
      0, // length of DW_LNS_set_basic_block
      0, // length of DW_LNS_const_add_pc
      1, // length of DW_LNS_fixed_advance_pc
      0, // length of DW_LNS_set_prologue_end
      0, // length of DW_LNS_set_epilogue_begin
      1  // DW_LNS_set_isa
  };
  assert(array_lengthof(StandardOpcodeLengths) ==
         (DWARF2_LINE_OPCODE_BASE - 1));
  return Emit(MCOS, StandardOpcodeLengths);
}

static const MCExpr *forceExpAbs(MCStreamer &OS, const MCExpr* Expr) {
  MCContext &Context = OS.getContext();
  assert(!isa<MCSymbolRefExpr>(Expr));
  if (Context.getAsmInfo()->hasAggressiveSymbolFolding())
    return Expr;

  MCSymbol *ABS = Context.createTempSymbol();
  OS.EmitAssignment(ABS, Expr);
  return MCSymbolRefExpr::create(ABS, Context);
}

static void emitAbsValue(MCStreamer &OS, const MCExpr *Value, unsigned Size) {
  const MCExpr *ABS = forceExpAbs(OS, Value);
  OS.EmitValue(ABS, Size);
}

std::pair<MCSymbol *, MCSymbol *>
MCDwarfLineTableHeader::Emit(MCStreamer *MCOS,
                             ArrayRef<char> StandardOpcodeLengths) const {

  MCContext &context = MCOS->getContext();

  // Create a symbol at the beginning of the line table.
  MCSymbol *LineStartSym = Label;
  if (!LineStartSym)
    LineStartSym = context.createTempSymbol();
  // Set the value of the symbol, as we are at the start of the line table.
  MCOS->EmitLabel(LineStartSym);

  // Create a symbol for the end of the section (to be set when we get there).
  MCSymbol *LineEndSym = context.createTempSymbol();

  // The first 4 bytes is the total length of the information for this
  // compilation unit (not including these 4 bytes for the length).
  emitAbsValue(*MCOS,
               MakeStartMinusEndExpr(*MCOS, *LineStartSym, *LineEndSym, 4), 4);

  // Next 2 bytes is the Version, which is Dwarf 2.
  MCOS->EmitIntValue(2, 2);

  // Create a symbol for the end of the prologue (to be set when we get there).
  MCSymbol *ProEndSym = context.createTempSymbol(); // Lprologue_end

  // Length of the prologue, is the next 4 bytes.  Which is the start of the
  // section to the end of the prologue.  Not including the 4 bytes for the
  // total length, the 2 bytes for the version, and these 4 bytes for the
  // length of the prologue.
  emitAbsValue(
      *MCOS,
      MakeStartMinusEndExpr(*MCOS, *LineStartSym, *ProEndSym, (4 + 2 + 4)), 4);

  // Parameters of the state machine, are next.
  MCOS->EmitIntValue(context.getAsmInfo()->getMinInstAlignment(), 1);
  MCOS->EmitIntValue(DWARF2_LINE_DEFAULT_IS_STMT, 1);
  MCOS->EmitIntValue(DWARF2_LINE_BASE, 1);
  MCOS->EmitIntValue(DWARF2_LINE_RANGE, 1);
  MCOS->EmitIntValue(StandardOpcodeLengths.size() + 1, 1);

  // Standard opcode lengths
  for (char Length : StandardOpcodeLengths)
    MCOS->EmitIntValue(Length, 1);

  // Put out the directory and file tables.

  // First the directory table.
  for (unsigned i = 0; i < MCDwarfDirs.size(); i++) {
    MCOS->EmitBytes(MCDwarfDirs[i]); // the DirectoryName
    MCOS->EmitBytes(StringRef("\0", 1)); // the null term. of the string
  }
  MCOS->EmitIntValue(0, 1); // Terminate the directory list

  // Second the file table.
  for (unsigned i = 1; i < MCDwarfFiles.size(); i++) {
    assert(!MCDwarfFiles[i].Name.empty());
    MCOS->EmitBytes(MCDwarfFiles[i].Name); // FileName
    MCOS->EmitBytes(StringRef("\0", 1)); // the null term. of the string
    // the Directory num
    MCOS->EmitULEB128IntValue(MCDwarfFiles[i].DirIndex);
    MCOS->EmitIntValue(0, 1); // last modification timestamp (always 0)
    MCOS->EmitIntValue(0, 1); // filesize (always 0)
  }
  MCOS->EmitIntValue(0, 1); // Terminate the file list

  // This is the end of the prologue, so set the value of the symbol at the
  // end of the prologue (that was used in a previous expression).
  MCOS->EmitLabel(ProEndSym);

  return std::make_pair(LineStartSym, LineEndSym);
}

void MCDwarfLineTable::EmitCU(MCObjectStreamer *MCOS) const {
  MCSymbol *LineEndSym = Header.Emit(MCOS).second;

  // Put out the line tables.
  for (const auto &LineSec : MCLineSections.getMCLineEntries())
    EmitDwarfLineTable(MCOS, LineSec.first, LineSec.second);

  // This is the end of the section, so set the value of the symbol at the end
  // of this section (that was used in a previous expression).
  MCOS->EmitLabel(LineEndSym);
}

unsigned MCDwarfLineTable::getFile(StringRef &Directory, StringRef &FileName,
                                   unsigned FileNumber) {
  return Header.getFile(Directory, FileName, FileNumber);
}

unsigned MCDwarfLineTableHeader::getFile(StringRef &Directory,
                                         StringRef &FileName,
                                         unsigned FileNumber) {
  if (Directory == CompilationDir)
    Directory = "";
  if (FileName.empty()) {
    FileName = "<stdin>";
    Directory = "";
  }
  assert(!FileName.empty());
  if (FileNumber == 0) {
    FileNumber = SourceIdMap.size() + 1;
    assert((MCDwarfFiles.empty() || FileNumber == MCDwarfFiles.size()) &&
           "Don't mix autonumbered and explicit numbered line table usage");
    SmallString<256> Buffer;
    auto IterBool = SourceIdMap.insert(
        std::make_pair((Directory + Twine('\0') + FileName).toStringRef(Buffer),
                       FileNumber));
    if (!IterBool.second)
      return IterBool.first->second;
  }
  // Make space for this FileNumber in the MCDwarfFiles vector if needed.
  MCDwarfFiles.resize(FileNumber + 1);

  // Get the new MCDwarfFile slot for this FileNumber.
  MCDwarfFile &File = MCDwarfFiles[FileNumber];

  // It is an error to use see the same number more than once.
  if (!File.Name.empty())
    return 0;

  if (Directory.empty()) {
    // Separate the directory part from the basename of the FileName.
    StringRef tFileName = sys::path::filename(FileName);
    if (!tFileName.empty()) {
      Directory = sys::path::parent_path(FileName);
      if (!Directory.empty())
        FileName = tFileName;
    }
  }

  // Find or make an entry in the MCDwarfDirs vector for this Directory.
  // Capture directory name.
  unsigned DirIndex;
  if (Directory.empty()) {
    // For FileNames with no directories a DirIndex of 0 is used.
    DirIndex = 0;
  } else {
    DirIndex = 0;
    for (unsigned End = MCDwarfDirs.size(); DirIndex < End; DirIndex++) {
      if (Directory == MCDwarfDirs[DirIndex])
        break;
    }
    if (DirIndex >= MCDwarfDirs.size())
      MCDwarfDirs.push_back(Directory);
    // The DirIndex is one based, as DirIndex of 0 is used for FileNames with
    // no directories.  MCDwarfDirs[] is unlike MCDwarfFiles[] in that the
    // directory names are stored at MCDwarfDirs[DirIndex-1] where FileNames
    // are stored at MCDwarfFiles[FileNumber].Name .
    DirIndex++;
  }

  File.Name = FileName;
  File.DirIndex = DirIndex;

  // return the allocated FileNumber.
  return FileNumber;
}

/// Utility function to emit the encoding to a streamer.
void MCDwarfLineAddr::Emit(MCStreamer *MCOS, int64_t LineDelta,
                           uint64_t AddrDelta) {
  MCContext &Context = MCOS->getContext();
  SmallString<256> Tmp;
  raw_svector_ostream OS(Tmp);
  MCDwarfLineAddr::Encode(Context, LineDelta, AddrDelta, OS);
  MCOS->EmitBytes(OS.str());
}

/// Utility function to encode a Dwarf pair of LineDelta and AddrDeltas.
void MCDwarfLineAddr::Encode(MCContext &Context, int64_t LineDelta,
                             uint64_t AddrDelta, raw_ostream &OS) {
  uint64_t Temp, Opcode;
  bool NeedCopy = false;

  // Scale the address delta by the minimum instruction length.
  AddrDelta = ScaleAddrDelta(Context, AddrDelta);

  // A LineDelta of INT64_MAX is a signal that this is actually a
  // DW_LNE_end_sequence. We cannot use special opcodes here, since we want the
  // end_sequence to emit the matrix entry.
  if (LineDelta == INT64_MAX) {
    if (AddrDelta == MAX_SPECIAL_ADDR_DELTA)
      OS << char(dwarf::DW_LNS_const_add_pc);
    else if (AddrDelta) {
      OS << char(dwarf::DW_LNS_advance_pc);
      encodeULEB128(AddrDelta, OS);
    }
    OS << char(dwarf::DW_LNS_extended_op);
    OS << char(1);
    OS << char(dwarf::DW_LNE_end_sequence);
    return;
  }

  // Bias the line delta by the base.
  Temp = LineDelta - DWARF2_LINE_BASE;

  // If the line increment is out of range of a special opcode, we must encode
  // it with DW_LNS_advance_line.
  if (Temp >= DWARF2_LINE_RANGE) {
    OS << char(dwarf::DW_LNS_advance_line);
    encodeSLEB128(LineDelta, OS);

    LineDelta = 0;
    Temp = 0 - DWARF2_LINE_BASE;
    NeedCopy = true;
  }

  // Use DW_LNS_copy instead of a "line +0, addr +0" special opcode.
  if (LineDelta == 0 && AddrDelta == 0) {
    OS << char(dwarf::DW_LNS_copy);
    return;
  }

  // Bias the opcode by the special opcode base.
  Temp += DWARF2_LINE_OPCODE_BASE;

  // Avoid overflow when addr_delta is large.
  if (AddrDelta < 256 + MAX_SPECIAL_ADDR_DELTA) {
    // Try using a special opcode.
    Opcode = Temp + AddrDelta * DWARF2_LINE_RANGE;
    if (Opcode <= 255) {
      OS << char(Opcode);
      return;
    }

    // Try using DW_LNS_const_add_pc followed by special op.
    Opcode = Temp + (AddrDelta - MAX_SPECIAL_ADDR_DELTA) * DWARF2_LINE_RANGE;
    if (Opcode <= 255) {
      OS << char(dwarf::DW_LNS_const_add_pc);
      OS << char(Opcode);
      return;
    }
  }

  // Otherwise use DW_LNS_advance_pc.
  OS << char(dwarf::DW_LNS_advance_pc);
  encodeULEB128(AddrDelta, OS);

  if (NeedCopy)
    OS << char(dwarf::DW_LNS_copy);
  else
    OS << char(Temp);
}

// Utility function to write a tuple for .debug_abbrev.
static void EmitAbbrev(MCStreamer *MCOS, uint64_t Name, uint64_t Form) {
  MCOS->EmitULEB128IntValue(Name);
  MCOS->EmitULEB128IntValue(Form);
}

// When generating dwarf for assembly source files this emits
// the data for .debug_abbrev section which contains three DIEs.
static void EmitGenDwarfAbbrev(MCStreamer *MCOS) {
  MCContext &context = MCOS->getContext();
  MCOS->SwitchSection(context.getObjectFileInfo()->getDwarfAbbrevSection());

  // DW_TAG_compile_unit DIE abbrev (1).
  MCOS->EmitULEB128IntValue(1);
  MCOS->EmitULEB128IntValue(dwarf::DW_TAG_compile_unit);
  MCOS->EmitIntValue(dwarf::DW_CHILDREN_yes, 1);
  EmitAbbrev(MCOS, dwarf::DW_AT_stmt_list, dwarf::DW_FORM_data4);
  if (MCOS->getContext().getGenDwarfSectionSyms().size() > 1 &&
      MCOS->getContext().getDwarfVersion() >= 3) {
    EmitAbbrev(MCOS, dwarf::DW_AT_ranges, dwarf::DW_FORM_data4);
  } else {
    EmitAbbrev(MCOS, dwarf::DW_AT_low_pc, dwarf::DW_FORM_addr);
    EmitAbbrev(MCOS, dwarf::DW_AT_high_pc, dwarf::DW_FORM_addr);
  }
  EmitAbbrev(MCOS, dwarf::DW_AT_name, dwarf::DW_FORM_string);
  if (!context.getCompilationDir().empty())
    EmitAbbrev(MCOS, dwarf::DW_AT_comp_dir, dwarf::DW_FORM_string);
  StringRef DwarfDebugFlags = context.getDwarfDebugFlags();
  if (!DwarfDebugFlags.empty())
    EmitAbbrev(MCOS, dwarf::DW_AT_APPLE_flags, dwarf::DW_FORM_string);
  EmitAbbrev(MCOS, dwarf::DW_AT_producer, dwarf::DW_FORM_string);
  EmitAbbrev(MCOS, dwarf::DW_AT_language, dwarf::DW_FORM_data2);
  EmitAbbrev(MCOS, 0, 0);

  // DW_TAG_label DIE abbrev (2).
  MCOS->EmitULEB128IntValue(2);
  MCOS->EmitULEB128IntValue(dwarf::DW_TAG_label);
  MCOS->EmitIntValue(dwarf::DW_CHILDREN_yes, 1);
  EmitAbbrev(MCOS, dwarf::DW_AT_name, dwarf::DW_FORM_string);
  EmitAbbrev(MCOS, dwarf::DW_AT_decl_file, dwarf::DW_FORM_data4);
  EmitAbbrev(MCOS, dwarf::DW_AT_decl_line, dwarf::DW_FORM_data4);
  EmitAbbrev(MCOS, dwarf::DW_AT_low_pc, dwarf::DW_FORM_addr);
  EmitAbbrev(MCOS, dwarf::DW_AT_prototyped, dwarf::DW_FORM_flag);
  EmitAbbrev(MCOS, 0, 0);

  // DW_TAG_unspecified_parameters DIE abbrev (3).
  MCOS->EmitULEB128IntValue(3);
  MCOS->EmitULEB128IntValue(dwarf::DW_TAG_unspecified_parameters);
  MCOS->EmitIntValue(dwarf::DW_CHILDREN_no, 1);
  EmitAbbrev(MCOS, 0, 0);

  // Terminate the abbreviations for this compilation unit.
  MCOS->EmitIntValue(0, 1);
}

// When generating dwarf for assembly source files this emits the data for
// .debug_aranges section. This section contains a header and a table of pairs
// of PointerSize'ed values for the address and size of section(s) with line
// table entries.
static void EmitGenDwarfAranges(MCStreamer *MCOS,
                                const MCSymbol *InfoSectionSymbol) {
  MCContext &context = MCOS->getContext();

  auto &Sections = context.getGenDwarfSectionSyms();

  MCOS->SwitchSection(context.getObjectFileInfo()->getDwarfARangesSection());

  // This will be the length of the .debug_aranges section, first account for
  // the size of each item in the header (see below where we emit these items).
  int Length = 4 + 2 + 4 + 1 + 1;

  // Figure the padding after the header before the table of address and size
  // pairs who's values are PointerSize'ed.
  const MCAsmInfo *asmInfo = context.getAsmInfo();
  int AddrSize = asmInfo->getPointerSize();
  int Pad = 2 * AddrSize - (Length & (2 * AddrSize - 1));
  if (Pad == 2 * AddrSize)
    Pad = 0;
  Length += Pad;

  // Add the size of the pair of PointerSize'ed values for the address and size
  // of each section we have in the table.
  Length += 2 * AddrSize * Sections.size();
  // And the pair of terminating zeros.
  Length += 2 * AddrSize;


  // Emit the header for this section.
  // The 4 byte length not including the 4 byte value for the length.
  MCOS->EmitIntValue(Length - 4, 4);
  // The 2 byte version, which is 2.
  MCOS->EmitIntValue(2, 2);
  // The 4 byte offset to the compile unit in the .debug_info from the start
  // of the .debug_info.
  if (InfoSectionSymbol)
    MCOS->EmitSymbolValue(InfoSectionSymbol, 4,
                          asmInfo->needsDwarfSectionOffsetDirective());
  else
    MCOS->EmitIntValue(0, 4);
  // The 1 byte size of an address.
  MCOS->EmitIntValue(AddrSize, 1);
  // The 1 byte size of a segment descriptor, we use a value of zero.
  MCOS->EmitIntValue(0, 1);
  // Align the header with the padding if needed, before we put out the table.
  for(int i = 0; i < Pad; i++)
    MCOS->EmitIntValue(0, 1);

  // Now emit the table of pairs of PointerSize'ed values for the section
  // addresses and sizes.
  for (MCSection *Sec : Sections) {
    const MCSymbol *StartSymbol = Sec->getBeginSymbol();
    MCSymbol *EndSymbol = Sec->getEndSymbol(context);
    assert(StartSymbol && "StartSymbol must not be NULL");
    assert(EndSymbol && "EndSymbol must not be NULL");

    const MCExpr *Addr = MCSymbolRefExpr::create(
      StartSymbol, MCSymbolRefExpr::VK_None, context);
    const MCExpr *Size = MakeStartMinusEndExpr(*MCOS,
      *StartSymbol, *EndSymbol, 0);
    MCOS->EmitValue(Addr, AddrSize);
    emitAbsValue(*MCOS, Size, AddrSize);
  }

  // And finally the pair of terminating zeros.
  MCOS->EmitIntValue(0, AddrSize);
  MCOS->EmitIntValue(0, AddrSize);
}

// When generating dwarf for assembly source files this emits the data for
// .debug_info section which contains three parts.  The header, the compile_unit
// DIE and a list of label DIEs.
static void EmitGenDwarfInfo(MCStreamer *MCOS,
                             const MCSymbol *AbbrevSectionSymbol,
                             const MCSymbol *LineSectionSymbol,
                             const MCSymbol *RangesSectionSymbol) {
  MCContext &context = MCOS->getContext();

  MCOS->SwitchSection(context.getObjectFileInfo()->getDwarfInfoSection());

  // Create a symbol at the start and end of this section used in here for the
  // expression to calculate the length in the header.
  MCSymbol *InfoStart = context.createTempSymbol();
  MCOS->EmitLabel(InfoStart);
  MCSymbol *InfoEnd = context.createTempSymbol();

  // First part: the header.

  // The 4 byte total length of the information for this compilation unit, not
  // including these 4 bytes.
  const MCExpr *Length = MakeStartMinusEndExpr(*MCOS, *InfoStart, *InfoEnd, 4);
  emitAbsValue(*MCOS, Length, 4);

  // The 2 byte DWARF version.
  MCOS->EmitIntValue(context.getDwarfVersion(), 2);

  const MCAsmInfo &AsmInfo = *context.getAsmInfo();
  // The 4 byte offset to the debug abbrevs from the start of the .debug_abbrev,
  // it is at the start of that section so this is zero.
  if (AbbrevSectionSymbol == nullptr)
    MCOS->EmitIntValue(0, 4);
  else
    MCOS->EmitSymbolValue(AbbrevSectionSymbol, 4,
                          AsmInfo.needsDwarfSectionOffsetDirective());

  const MCAsmInfo *asmInfo = context.getAsmInfo();
  int AddrSize = asmInfo->getPointerSize();
  // The 1 byte size of an address.
  MCOS->EmitIntValue(AddrSize, 1);

  // Second part: the compile_unit DIE.

  // The DW_TAG_compile_unit DIE abbrev (1).
  MCOS->EmitULEB128IntValue(1);

  // DW_AT_stmt_list, a 4 byte offset from the start of the .debug_line section,
  // which is at the start of that section so this is zero.
  if (LineSectionSymbol)
    MCOS->EmitSymbolValue(LineSectionSymbol, 4,
                          AsmInfo.needsDwarfSectionOffsetDirective());
  else
    MCOS->EmitIntValue(0, 4);

  if (RangesSectionSymbol) {
    // There are multiple sections containing code, so we must use the
    // .debug_ranges sections.

    // AT_ranges, the 4 byte offset from the start of the .debug_ranges section
    // to the address range list for this compilation unit.
    MCOS->EmitSymbolValue(RangesSectionSymbol, 4);
  } else {
    // If we only have one non-empty code section, we can use the simpler
    // AT_low_pc and AT_high_pc attributes.

    // Find the first (and only) non-empty text section
    auto &Sections = context.getGenDwarfSectionSyms();
    const auto TextSection = Sections.begin();
    assert(TextSection != Sections.end() && "No text section found");

    MCSymbol *StartSymbol = (*TextSection)->getBeginSymbol();
    MCSymbol *EndSymbol = (*TextSection)->getEndSymbol(context);
    assert(StartSymbol && "StartSymbol must not be NULL");
    assert(EndSymbol && "EndSymbol must not be NULL");

    // AT_low_pc, the first address of the default .text section.
    const MCExpr *Start = MCSymbolRefExpr::create(
        StartSymbol, MCSymbolRefExpr::VK_None, context);
    MCOS->EmitValue(Start, AddrSize);

    // AT_high_pc, the last address of the default .text section.
    const MCExpr *End = MCSymbolRefExpr::create(
      EndSymbol, MCSymbolRefExpr::VK_None, context);
    MCOS->EmitValue(End, AddrSize);
  }

  // AT_name, the name of the source file.  Reconstruct from the first directory
  // and file table entries.
  const SmallVectorImpl<std::string> &MCDwarfDirs = context.getMCDwarfDirs();
  if (MCDwarfDirs.size() > 0) {
    MCOS->EmitBytes(MCDwarfDirs[0]);
    MCOS->EmitBytes(sys::path::get_separator());
  }
  const SmallVectorImpl<MCDwarfFile> &MCDwarfFiles =
    MCOS->getContext().getMCDwarfFiles();
  MCOS->EmitBytes(MCDwarfFiles[1].Name);
  MCOS->EmitIntValue(0, 1); // NULL byte to terminate the string.

  // AT_comp_dir, the working directory the assembly was done in.
  if (!context.getCompilationDir().empty()) {
    MCOS->EmitBytes(context.getCompilationDir());
    MCOS->EmitIntValue(0, 1); // NULL byte to terminate the string.
  }

  // AT_APPLE_flags, the command line arguments of the assembler tool.
  StringRef DwarfDebugFlags = context.getDwarfDebugFlags();
  if (!DwarfDebugFlags.empty()){
    MCOS->EmitBytes(DwarfDebugFlags);
    MCOS->EmitIntValue(0, 1); // NULL byte to terminate the string.
  }

  // AT_producer, the version of the assembler tool.
  StringRef DwarfDebugProducer = context.getDwarfDebugProducer();
  if (!DwarfDebugProducer.empty())
    MCOS->EmitBytes(DwarfDebugProducer);
  else
    MCOS->EmitBytes(StringRef("llvm-mc (based on LLVM " PACKAGE_VERSION ")"));
  MCOS->EmitIntValue(0, 1); // NULL byte to terminate the string.

  // AT_language, a 4 byte value.  We use DW_LANG_Mips_Assembler as the dwarf2
  // draft has no standard code for assembler.
  MCOS->EmitIntValue(dwarf::DW_LANG_Mips_Assembler, 2);

  // Third part: the list of label DIEs.

  // Loop on saved info for dwarf labels and create the DIEs for them.
  const std::vector<MCGenDwarfLabelEntry> &Entries =
      MCOS->getContext().getMCGenDwarfLabelEntries();
  for (const auto &Entry : Entries) {
    // The DW_TAG_label DIE abbrev (2).
    MCOS->EmitULEB128IntValue(2);

    // AT_name, of the label without any leading underbar.
    MCOS->EmitBytes(Entry.getName());
    MCOS->EmitIntValue(0, 1); // NULL byte to terminate the string.

    // AT_decl_file, index into the file table.
    MCOS->EmitIntValue(Entry.getFileNumber(), 4);

    // AT_decl_line, source line number.
    MCOS->EmitIntValue(Entry.getLineNumber(), 4);

    // AT_low_pc, start address of the label.
    const MCExpr *AT_low_pc = MCSymbolRefExpr::create(Entry.getLabel(),
                                             MCSymbolRefExpr::VK_None, context);
    MCOS->EmitValue(AT_low_pc, AddrSize);

    // DW_AT_prototyped, a one byte flag value of 0 saying we have no prototype.
    MCOS->EmitIntValue(0, 1);

    // The DW_TAG_unspecified_parameters DIE abbrev (3).
    MCOS->EmitULEB128IntValue(3);

    // Add the NULL DIE terminating the DW_TAG_unspecified_parameters DIE's.
    MCOS->EmitIntValue(0, 1);
  }

  // Add the NULL DIE terminating the Compile Unit DIE's.
  MCOS->EmitIntValue(0, 1);

  // Now set the value of the symbol at the end of the info section.
  MCOS->EmitLabel(InfoEnd);
}

// When generating dwarf for assembly source files this emits the data for
// .debug_ranges section. We only emit one range list, which spans all of the
// executable sections of this file.
static void EmitGenDwarfRanges(MCStreamer *MCOS) {
  MCContext &context = MCOS->getContext();
  auto &Sections = context.getGenDwarfSectionSyms();

  const MCAsmInfo *AsmInfo = context.getAsmInfo();
  int AddrSize = AsmInfo->getPointerSize();

  MCOS->SwitchSection(context.getObjectFileInfo()->getDwarfRangesSection());

  for (MCSection *Sec : Sections) {
    const MCSymbol *StartSymbol = Sec->getBeginSymbol();
    MCSymbol *EndSymbol = Sec->getEndSymbol(context);
    assert(StartSymbol && "StartSymbol must not be NULL");
    assert(EndSymbol && "EndSymbol must not be NULL");

    // Emit a base address selection entry for the start of this section
    const MCExpr *SectionStartAddr = MCSymbolRefExpr::create(
      StartSymbol, MCSymbolRefExpr::VK_None, context);
    MCOS->EmitFill(AddrSize, 0xFF);
    MCOS->EmitValue(SectionStartAddr, AddrSize);

    // Emit a range list entry spanning this section
    const MCExpr *SectionSize = MakeStartMinusEndExpr(*MCOS,
      *StartSymbol, *EndSymbol, 0);
    MCOS->EmitIntValue(0, AddrSize);
    emitAbsValue(*MCOS, SectionSize, AddrSize);
  }

  // Emit end of list entry
  MCOS->EmitIntValue(0, AddrSize);
  MCOS->EmitIntValue(0, AddrSize);
}

//
// When generating dwarf for assembly source files this emits the Dwarf
// sections.
//
void MCGenDwarfInfo::Emit(MCStreamer *MCOS) {
  MCContext &context = MCOS->getContext();

  // Create the dwarf sections in this order (.debug_line already created).
  const MCAsmInfo *AsmInfo = context.getAsmInfo();
  bool CreateDwarfSectionSymbols =
      AsmInfo->doesDwarfUseRelocationsAcrossSections();
  MCSymbol *LineSectionSymbol = nullptr;
  if (CreateDwarfSectionSymbols)
    LineSectionSymbol = MCOS->getDwarfLineTableSymbol(0);
  MCSymbol *AbbrevSectionSymbol = nullptr;
  MCSymbol *InfoSectionSymbol = nullptr;
  MCSymbol *RangesSectionSymbol = NULL;

  // Create end symbols for each section, and remove empty sections
  MCOS->getContext().finalizeDwarfSections(*MCOS);

  // If there are no sections to generate debug info for, we don't need
  // to do anything
  if (MCOS->getContext().getGenDwarfSectionSyms().empty())
    return;

  // We only use the .debug_ranges section if we have multiple code sections,
  // and we are emitting a DWARF version which supports it.
  const bool UseRangesSection =
      MCOS->getContext().getGenDwarfSectionSyms().size() > 1 &&
      MCOS->getContext().getDwarfVersion() >= 3;
  CreateDwarfSectionSymbols |= UseRangesSection;

  MCOS->SwitchSection(context.getObjectFileInfo()->getDwarfInfoSection());
  if (CreateDwarfSectionSymbols) {
    InfoSectionSymbol = context.createTempSymbol();
    MCOS->EmitLabel(InfoSectionSymbol);
  }
  MCOS->SwitchSection(context.getObjectFileInfo()->getDwarfAbbrevSection());
  if (CreateDwarfSectionSymbols) {
    AbbrevSectionSymbol = context.createTempSymbol();
    MCOS->EmitLabel(AbbrevSectionSymbol);
  }
  if (UseRangesSection) {
    MCOS->SwitchSection(context.getObjectFileInfo()->getDwarfRangesSection());
    if (CreateDwarfSectionSymbols) {
      RangesSectionSymbol = context.createTempSymbol();
      MCOS->EmitLabel(RangesSectionSymbol);
    }
  }

  assert((RangesSectionSymbol != NULL) || !UseRangesSection);

  MCOS->SwitchSection(context.getObjectFileInfo()->getDwarfARangesSection());

  // Output the data for .debug_aranges section.
  EmitGenDwarfAranges(MCOS, InfoSectionSymbol);

  if (UseRangesSection)
    EmitGenDwarfRanges(MCOS);

  // Output the data for .debug_abbrev section.
  EmitGenDwarfAbbrev(MCOS);

  // Output the data for .debug_info section.
  EmitGenDwarfInfo(MCOS, AbbrevSectionSymbol, LineSectionSymbol,
                   RangesSectionSymbol);
}

//
// When generating dwarf for assembly source files this is called when symbol
// for a label is created.  If this symbol is not a temporary and is in the
// section that dwarf is being generated for, save the needed info to create
// a dwarf label.
//
void MCGenDwarfLabelEntry::Make(MCSymbol *Symbol, MCStreamer *MCOS,
                                     SourceMgr &SrcMgr, SMLoc &Loc) {
  // We won't create dwarf labels for temporary symbols.
  if (Symbol->isTemporary())
    return;
  MCContext &context = MCOS->getContext();
  // We won't create dwarf labels for symbols in sections that we are not
  // generating debug info for.
  if (!context.getGenDwarfSectionSyms().count(MCOS->getCurrentSection().first))
    return;

  // The dwarf label's name does not have the symbol name's leading
  // underbar if any.
  StringRef Name = Symbol->getName();
  if (Name.startswith("_"))
    Name = Name.substr(1, Name.size()-1);

  // Get the dwarf file number to be used for the dwarf label.
  unsigned FileNumber = context.getGenDwarfFileNumber();

  // Finding the line number is the expensive part which is why we just don't
  // pass it in as for some symbols we won't create a dwarf label.
  unsigned CurBuffer = SrcMgr.FindBufferContainingLoc(Loc);
  unsigned LineNumber = SrcMgr.FindLineNumber(Loc, CurBuffer);

  // We create a temporary symbol for use for the AT_high_pc and AT_low_pc
  // values so that they don't have things like an ARM thumb bit from the
  // original symbol. So when used they won't get a low bit set after
  // relocation.
  MCSymbol *Label = context.createTempSymbol();
  MCOS->EmitLabel(Label);

  // Create and entry for the info and add it to the other entries.
  MCOS->getContext().addMCGenDwarfLabelEntry(
      MCGenDwarfLabelEntry(Name, FileNumber, LineNumber, Label));
}

static int getDataAlignmentFactor(MCStreamer &streamer) {
  MCContext &context = streamer.getContext();
  const MCAsmInfo *asmInfo = context.getAsmInfo();
  int size = asmInfo->getCalleeSaveStackSlotSize();
  if (asmInfo->isStackGrowthDirectionUp())
    return size;
  else
    return -size;
}

static unsigned getSizeForEncoding(MCStreamer &streamer,
                                   unsigned symbolEncoding) {
  MCContext &context = streamer.getContext();
  unsigned format = symbolEncoding & 0x0f;
  switch (format) {
  default: llvm_unreachable("Unknown Encoding");
  case dwarf::DW_EH_PE_absptr:
  case dwarf::DW_EH_PE_signed:
    return context.getAsmInfo()->getPointerSize();
  case dwarf::DW_EH_PE_udata2:
  case dwarf::DW_EH_PE_sdata2:
    return 2;
  case dwarf::DW_EH_PE_udata4:
  case dwarf::DW_EH_PE_sdata4:
    return 4;
  case dwarf::DW_EH_PE_udata8:
  case dwarf::DW_EH_PE_sdata8:
    return 8;
  }
}

static void emitFDESymbol(MCObjectStreamer &streamer, const MCSymbol &symbol,
                       unsigned symbolEncoding, bool isEH) {
  MCContext &context = streamer.getContext();
  const MCAsmInfo *asmInfo = context.getAsmInfo();
  const MCExpr *v = asmInfo->getExprForFDESymbol(&symbol,
                                                 symbolEncoding,
                                                 streamer);
  unsigned size = getSizeForEncoding(streamer, symbolEncoding);
  if (asmInfo->doDwarfFDESymbolsUseAbsDiff() && isEH)
    emitAbsValue(streamer, v, size);
  else
    streamer.EmitValue(v, size);
}

static void EmitPersonality(MCStreamer &streamer, const MCSymbol &symbol,
                            unsigned symbolEncoding) {
  MCContext &context = streamer.getContext();
  const MCAsmInfo *asmInfo = context.getAsmInfo();
  const MCExpr *v = asmInfo->getExprForPersonalitySymbol(&symbol,
                                                         symbolEncoding,
                                                         streamer);
  unsigned size = getSizeForEncoding(streamer, symbolEncoding);
  streamer.EmitValue(v, size);
}

namespace {
  class FrameEmitterImpl {
    int CFAOffset;
    int InitialCFAOffset;
    bool IsEH;
    const MCSymbol *SectionStart;
  public:
    FrameEmitterImpl(bool isEH)
        : CFAOffset(0), InitialCFAOffset(0), IsEH(isEH), SectionStart(nullptr) {
    }

    void setSectionStart(const MCSymbol *Label) { SectionStart = Label; }

    /// Emit the unwind information in a compact way.
    void EmitCompactUnwind(MCObjectStreamer &streamer,
                           const MCDwarfFrameInfo &frame);

    const MCSymbol &EmitCIE(MCObjectStreamer &streamer,
                            const MCSymbol *personality,
                            unsigned personalityEncoding,
                            const MCSymbol *lsda,
                            bool IsSignalFrame,
                            unsigned lsdaEncoding,
                            bool IsSimple);
    MCSymbol *EmitFDE(MCObjectStreamer &streamer,
                      const MCSymbol &cieStart,
                      const MCDwarfFrameInfo &frame);
    void EmitCFIInstructions(MCObjectStreamer &streamer,
                             ArrayRef<MCCFIInstruction> Instrs,
                             MCSymbol *BaseLabel);
    void EmitCFIInstruction(MCObjectStreamer &Streamer,
                            const MCCFIInstruction &Instr);
  };

} // end anonymous namespace

static void emitEncodingByte(MCObjectStreamer &Streamer, unsigned Encoding) {
  Streamer.EmitIntValue(Encoding, 1);
}

void FrameEmitterImpl::EmitCFIInstruction(MCObjectStreamer &Streamer,
                                          const MCCFIInstruction &Instr) {
  int dataAlignmentFactor = getDataAlignmentFactor(Streamer);
  auto *MRI = Streamer.getContext().getRegisterInfo();

  switch (Instr.getOperation()) {
  case MCCFIInstruction::OpRegister: {
    unsigned Reg1 = Instr.getRegister();
    unsigned Reg2 = Instr.getRegister2();
    if (!IsEH) {
      Reg1 = MRI->getDwarfRegNum(MRI->getLLVMRegNum(Reg1, true), false);
      Reg2 = MRI->getDwarfRegNum(MRI->getLLVMRegNum(Reg2, true), false);
    }
    Streamer.EmitIntValue(dwarf::DW_CFA_register, 1);
    Streamer.EmitULEB128IntValue(Reg1);
    Streamer.EmitULEB128IntValue(Reg2);
    return;
  }
  case MCCFIInstruction::OpWindowSave: {
    Streamer.EmitIntValue(dwarf::DW_CFA_GNU_window_save, 1);
    return;
  }
  case MCCFIInstruction::OpUndefined: {
    unsigned Reg = Instr.getRegister();
    Streamer.EmitIntValue(dwarf::DW_CFA_undefined, 1);
    Streamer.EmitULEB128IntValue(Reg);
    return;
  }
  case MCCFIInstruction::OpAdjustCfaOffset:
  case MCCFIInstruction::OpDefCfaOffset: {
    const bool IsRelative =
      Instr.getOperation() == MCCFIInstruction::OpAdjustCfaOffset;

    Streamer.EmitIntValue(dwarf::DW_CFA_def_cfa_offset, 1);

    if (IsRelative)
      CFAOffset += Instr.getOffset();
    else
      CFAOffset = -Instr.getOffset();

    Streamer.EmitULEB128IntValue(CFAOffset);

    return;
  }
  case MCCFIInstruction::OpDefCfa: {
    unsigned Reg = Instr.getRegister();
    if (!IsEH)
      Reg = MRI->getDwarfRegNum(MRI->getLLVMRegNum(Reg, true), false);
    Streamer.EmitIntValue(dwarf::DW_CFA_def_cfa, 1);
    Streamer.EmitULEB128IntValue(Reg);
    CFAOffset = -Instr.getOffset();
    Streamer.EmitULEB128IntValue(CFAOffset);

    return;
  }

  case MCCFIInstruction::OpDefCfaRegister: {
    unsigned Reg = Instr.getRegister();
    if (!IsEH)
      Reg = MRI->getDwarfRegNum(MRI->getLLVMRegNum(Reg, true), false);
    Streamer.EmitIntValue(dwarf::DW_CFA_def_cfa_register, 1);
    Streamer.EmitULEB128IntValue(Reg);

    return;
  }

  case MCCFIInstruction::OpOffset:
  case MCCFIInstruction::OpRelOffset: {
    const bool IsRelative =
      Instr.getOperation() == MCCFIInstruction::OpRelOffset;

    unsigned Reg = Instr.getRegister();
    if (!IsEH)
      Reg = MRI->getDwarfRegNum(MRI->getLLVMRegNum(Reg, true), false);

    int Offset = Instr.getOffset();
    if (IsRelative)
      Offset -= CFAOffset;
    Offset = Offset / dataAlignmentFactor;

    if (Offset < 0) {
      Streamer.EmitIntValue(dwarf::DW_CFA_offset_extended_sf, 1);
      Streamer.EmitULEB128IntValue(Reg);
      Streamer.EmitSLEB128IntValue(Offset);
    } else if (Reg < 64) {
      Streamer.EmitIntValue(dwarf::DW_CFA_offset + Reg, 1);
      Streamer.EmitULEB128IntValue(Offset);
    } else {
      Streamer.EmitIntValue(dwarf::DW_CFA_offset_extended, 1);
      Streamer.EmitULEB128IntValue(Reg);
      Streamer.EmitULEB128IntValue(Offset);
    }
    return;
  }
  case MCCFIInstruction::OpRememberState:
    Streamer.EmitIntValue(dwarf::DW_CFA_remember_state, 1);
    return;
  case MCCFIInstruction::OpRestoreState:
    Streamer.EmitIntValue(dwarf::DW_CFA_restore_state, 1);
    return;
  case MCCFIInstruction::OpSameValue: {
    unsigned Reg = Instr.getRegister();
    Streamer.EmitIntValue(dwarf::DW_CFA_same_value, 1);
    Streamer.EmitULEB128IntValue(Reg);
    return;
  }
  case MCCFIInstruction::OpRestore: {
    unsigned Reg = Instr.getRegister();
    if (!IsEH)
      Reg = MRI->getDwarfRegNum(MRI->getLLVMRegNum(Reg, true), false);
    Streamer.EmitIntValue(dwarf::DW_CFA_restore | Reg, 1);
    return;
  }
  case MCCFIInstruction::OpEscape:
    Streamer.EmitBytes(Instr.getValues());
    return;
  }
  llvm_unreachable("Unhandled case in switch");
}

/// Emit frame instructions to describe the layout of the frame.
void FrameEmitterImpl::EmitCFIInstructions(MCObjectStreamer &streamer,
                                           ArrayRef<MCCFIInstruction> Instrs,
                                           MCSymbol *BaseLabel) {
  for (unsigned i = 0, N = Instrs.size(); i < N; ++i) {
    const MCCFIInstruction &Instr = Instrs[i];
    MCSymbol *Label = Instr.getLabel();
    // Throw out move if the label is invalid.
    if (Label && !Label->isDefined()) continue; // Not emitted, in dead code.

    // Advance row if new location.
    if (BaseLabel && Label) {
      MCSymbol *ThisSym = Label;
      if (ThisSym != BaseLabel) {
        streamer.EmitDwarfAdvanceFrameAddr(BaseLabel, ThisSym);
        BaseLabel = ThisSym;
      }
    }

    EmitCFIInstruction(streamer, Instr);
  }
}

/// Emit the unwind information in a compact way.
void FrameEmitterImpl::EmitCompactUnwind(MCObjectStreamer &Streamer,
                                         const MCDwarfFrameInfo &Frame) {
  MCContext &Context = Streamer.getContext();
  const MCObjectFileInfo *MOFI = Context.getObjectFileInfo();

  // range-start range-length  compact-unwind-enc personality-func   lsda
  //  _foo       LfooEnd-_foo  0x00000023          0                 0
  //  _bar       LbarEnd-_bar  0x00000025         __gxx_personality  except_tab1
  //
  //   .section __LD,__compact_unwind,regular,debug
  //
  //   # compact unwind for _foo
  //   .quad _foo
  //   .set L1,LfooEnd-_foo
  //   .long L1
  //   .long 0x01010001
  //   .quad 0
  //   .quad 0
  //
  //   # compact unwind for _bar
  //   .quad _bar
  //   .set L2,LbarEnd-_bar
  //   .long L2
  //   .long 0x01020011
  //   .quad __gxx_personality
  //   .quad except_tab1

  uint32_t Encoding = Frame.CompactUnwindEncoding;
  if (!Encoding) return;
  bool DwarfEHFrameOnly = (Encoding == MOFI->getCompactUnwindDwarfEHFrameOnly());

  // The encoding needs to know we have an LSDA.
  if (!DwarfEHFrameOnly && Frame.Lsda)
    Encoding |= 0x40000000;

  // Range Start
  unsigned FDEEncoding = MOFI->getFDEEncoding();
  unsigned Size = getSizeForEncoding(Streamer, FDEEncoding);
  Streamer.EmitSymbolValue(Frame.Begin, Size);

  // Range Length
  const MCExpr *Range = MakeStartMinusEndExpr(Streamer, *Frame.Begin,
                                              *Frame.End, 0);
  emitAbsValue(Streamer, Range, 4);

  // Compact Encoding
  Size = getSizeForEncoding(Streamer, dwarf::DW_EH_PE_udata4);
  Streamer.EmitIntValue(Encoding, Size);

  // Personality Function
  Size = getSizeForEncoding(Streamer, dwarf::DW_EH_PE_absptr);
  if (!DwarfEHFrameOnly && Frame.Personality)
    Streamer.EmitSymbolValue(Frame.Personality, Size);
  else
    Streamer.EmitIntValue(0, Size); // No personality fn

  // LSDA
  Size = getSizeForEncoding(Streamer, Frame.LsdaEncoding);
  if (!DwarfEHFrameOnly && Frame.Lsda)
    Streamer.EmitSymbolValue(Frame.Lsda, Size);
  else
    Streamer.EmitIntValue(0, Size); // No LSDA
}

static unsigned getCIEVersion(bool IsEH, unsigned DwarfVersion) {
  if (IsEH)
    return 1;
  switch (DwarfVersion) {
  case 2:
    return 1;
  case 3:
    return 3;
  case 4:
    return 4;
  }
  llvm_unreachable("Unknown version");
}

const MCSymbol &FrameEmitterImpl::EmitCIE(MCObjectStreamer &streamer,
                                          const MCSymbol *personality,
                                          unsigned personalityEncoding,
                                          const MCSymbol *lsda,
                                          bool IsSignalFrame,
                                          unsigned lsdaEncoding,
                                          bool IsSimple) {
  MCContext &context = streamer.getContext();
  const MCRegisterInfo *MRI = context.getRegisterInfo();
  const MCObjectFileInfo *MOFI = context.getObjectFileInfo();

  MCSymbol *sectionStart = context.createTempSymbol();
  streamer.EmitLabel(sectionStart);

  MCSymbol *sectionEnd = context.createTempSymbol();

  // Length
  const MCExpr *Length = MakeStartMinusEndExpr(streamer, *sectionStart,
                                               *sectionEnd, 4);
  emitAbsValue(streamer, Length, 4);

  // CIE ID
  unsigned CIE_ID = IsEH ? 0 : -1;
  streamer.EmitIntValue(CIE_ID, 4);

  // Version
  uint8_t CIEVersion = getCIEVersion(IsEH, context.getDwarfVersion());
  streamer.EmitIntValue(CIEVersion, 1);

  // Augmentation String
  SmallString<8> Augmentation;
  if (IsEH) {
    Augmentation += "z";
    if (personality)
      Augmentation += "P";
    if (lsda)
      Augmentation += "L";
    Augmentation += "R";
    if (IsSignalFrame)
      Augmentation += "S";
    streamer.EmitBytes(Augmentation);
  }
  streamer.EmitIntValue(0, 1);

  if (CIEVersion >= 4) {
    // Address Size
    streamer.EmitIntValue(context.getAsmInfo()->getPointerSize(), 1);

    // Segment Descriptor Size
    streamer.EmitIntValue(0, 1);
  }

  // Code Alignment Factor
  streamer.EmitULEB128IntValue(context.getAsmInfo()->getMinInstAlignment());

  // Data Alignment Factor
  streamer.EmitSLEB128IntValue(getDataAlignmentFactor(streamer));

  // Return Address Register
  if (CIEVersion == 1) {
    assert(MRI->getRARegister() <= 255 &&
           "DWARF 2 encodes return_address_register in one byte");
    streamer.EmitIntValue(MRI->getDwarfRegNum(MRI->getRARegister(), IsEH), 1);
  } else {
    streamer.EmitULEB128IntValue(
        MRI->getDwarfRegNum(MRI->getRARegister(), IsEH));
  }

  // Augmentation Data Length (optional)

  unsigned augmentationLength = 0;
  if (IsEH) {
    if (personality) {
      // Personality Encoding
      augmentationLength += 1;
      // Personality
      augmentationLength += getSizeForEncoding(streamer, personalityEncoding);
    }
    if (lsda)
      augmentationLength += 1;
    // Encoding of the FDE pointers
    augmentationLength += 1;

    streamer.EmitULEB128IntValue(augmentationLength);

    // Augmentation Data (optional)
    if (personality) {
      // Personality Encoding
      emitEncodingByte(streamer, personalityEncoding);
      // Personality
      EmitPersonality(streamer, *personality, personalityEncoding);
    }

    if (lsda)
      emitEncodingByte(streamer, lsdaEncoding);

    // Encoding of the FDE pointers
    emitEncodingByte(streamer, MOFI->getFDEEncoding());
  }

  // Initial Instructions

  const MCAsmInfo *MAI = context.getAsmInfo();
  if (!IsSimple) {
    const std::vector<MCCFIInstruction> &Instructions =
        MAI->getInitialFrameState();
    EmitCFIInstructions(streamer, Instructions, nullptr);
  }

  InitialCFAOffset = CFAOffset;

  // Padding
  streamer.EmitValueToAlignment(IsEH ? 4 : MAI->getPointerSize());

  streamer.EmitLabel(sectionEnd);
  return *sectionStart;
}

MCSymbol *FrameEmitterImpl::EmitFDE(MCObjectStreamer &streamer,
                                    const MCSymbol &cieStart,
                                    const MCDwarfFrameInfo &frame) {
  MCContext &context = streamer.getContext();
  MCSymbol *fdeStart = context.createTempSymbol();
  MCSymbol *fdeEnd = context.createTempSymbol();
  const MCObjectFileInfo *MOFI = context.getObjectFileInfo();

  CFAOffset = InitialCFAOffset;

  // Length
  const MCExpr *Length = MakeStartMinusEndExpr(streamer, *fdeStart, *fdeEnd, 0);
  emitAbsValue(streamer, Length, 4);

  streamer.EmitLabel(fdeStart);

  // CIE Pointer
  const MCAsmInfo *asmInfo = context.getAsmInfo();
  if (IsEH) {
    const MCExpr *offset = MakeStartMinusEndExpr(streamer, cieStart, *fdeStart,
                                                 0);
    emitAbsValue(streamer, offset, 4);
  } else if (!asmInfo->doesDwarfUseRelocationsAcrossSections()) {
    const MCExpr *offset = MakeStartMinusEndExpr(streamer, *SectionStart,
                                                 cieStart, 0);
    emitAbsValue(streamer, offset, 4);
  } else {
    streamer.EmitSymbolValue(&cieStart, 4);
  }

  // PC Begin
  unsigned PCEncoding =
      IsEH ? MOFI->getFDEEncoding() : (unsigned)dwarf::DW_EH_PE_absptr;
  unsigned PCSize = getSizeForEncoding(streamer, PCEncoding);
  emitFDESymbol(streamer, *frame.Begin, PCEncoding, IsEH);

  // PC Range
  const MCExpr *Range = MakeStartMinusEndExpr(streamer, *frame.Begin,
                                              *frame.End, 0);
  emitAbsValue(streamer, Range, PCSize);

  if (IsEH) {
    // Augmentation Data Length
    unsigned augmentationLength = 0;

    if (frame.Lsda)
      augmentationLength += getSizeForEncoding(streamer, frame.LsdaEncoding);

    streamer.EmitULEB128IntValue(augmentationLength);

    // Augmentation Data
    if (frame.Lsda)
      emitFDESymbol(streamer, *frame.Lsda, frame.LsdaEncoding, true);
  }

  // Call Frame Instructions
  EmitCFIInstructions(streamer, frame.Instructions, frame.Begin);

  // Padding
  streamer.EmitValueToAlignment(PCSize);

  return fdeEnd;
}

namespace {
  struct CIEKey {
    static const CIEKey getEmptyKey() {
      return CIEKey(nullptr, 0, -1, false, false);
    }
    static const CIEKey getTombstoneKey() {
      return CIEKey(nullptr, -1, 0, false, false);
    }

    CIEKey(const MCSymbol *Personality_, unsigned PersonalityEncoding_,
           unsigned LsdaEncoding_, bool IsSignalFrame_, bool IsSimple_)
        : Personality(Personality_), PersonalityEncoding(PersonalityEncoding_),
          LsdaEncoding(LsdaEncoding_), IsSignalFrame(IsSignalFrame_),
          IsSimple(IsSimple_) {}
    const MCSymbol *Personality;
    unsigned PersonalityEncoding;
    unsigned LsdaEncoding;
    bool IsSignalFrame;
    bool IsSimple;
  };
}

namespace llvm {
  template <>
  struct DenseMapInfo<CIEKey> {
    static CIEKey getEmptyKey() {
      return CIEKey::getEmptyKey();
    }
    static CIEKey getTombstoneKey() {
      return CIEKey::getTombstoneKey();
    }
    static unsigned getHashValue(const CIEKey &Key) {
      return static_cast<unsigned>(hash_combine(Key.Personality,
                                                Key.PersonalityEncoding,
                                                Key.LsdaEncoding,
                                                Key.IsSignalFrame,
                                                Key.IsSimple));
    }
    static bool isEqual(const CIEKey &LHS,
                        const CIEKey &RHS) {
      return LHS.Personality == RHS.Personality &&
        LHS.PersonalityEncoding == RHS.PersonalityEncoding &&
        LHS.LsdaEncoding == RHS.LsdaEncoding &&
        LHS.IsSignalFrame == RHS.IsSignalFrame &&
        LHS.IsSimple == RHS.IsSimple;
    }
  };
}

void MCDwarfFrameEmitter::Emit(MCObjectStreamer &Streamer, MCAsmBackend *MAB,
                               bool IsEH) {
  Streamer.generateCompactUnwindEncodings(MAB);

  MCContext &Context = Streamer.getContext();
  const MCObjectFileInfo *MOFI = Context.getObjectFileInfo();
  FrameEmitterImpl Emitter(IsEH);
  ArrayRef<MCDwarfFrameInfo> FrameArray = Streamer.getDwarfFrameInfos();

  // Emit the compact unwind info if available.
  bool NeedsEHFrameSection = !MOFI->getSupportsCompactUnwindWithoutEHFrame();
  if (IsEH && MOFI->getCompactUnwindSection()) {
    bool SectionEmitted = false;
    for (unsigned i = 0, n = FrameArray.size(); i < n; ++i) {
      const MCDwarfFrameInfo &Frame = FrameArray[i];
      if (Frame.CompactUnwindEncoding == 0) continue;
      if (!SectionEmitted) {
        Streamer.SwitchSection(MOFI->getCompactUnwindSection());
        Streamer.EmitValueToAlignment(Context.getAsmInfo()->getPointerSize());
        SectionEmitted = true;
      }
      NeedsEHFrameSection |=
        Frame.CompactUnwindEncoding ==
          MOFI->getCompactUnwindDwarfEHFrameOnly();
      Emitter.EmitCompactUnwind(Streamer, Frame);
    }
  }

  if (!NeedsEHFrameSection) return;

  MCSection &Section =
      IsEH ? *const_cast<MCObjectFileInfo *>(MOFI)->getEHFrameSection()
           : *MOFI->getDwarfFrameSection();

  Streamer.SwitchSection(&Section);
  MCSymbol *SectionStart = Context.createTempSymbol();
  Streamer.EmitLabel(SectionStart);
  Emitter.setSectionStart(SectionStart);

  MCSymbol *FDEEnd = nullptr;
  DenseMap<CIEKey, const MCSymbol *> CIEStarts;

  const MCSymbol *DummyDebugKey = nullptr;
  NeedsEHFrameSection = !MOFI->getSupportsCompactUnwindWithoutEHFrame();
  for (unsigned i = 0, n = FrameArray.size(); i < n; ++i) {
    const MCDwarfFrameInfo &Frame = FrameArray[i];

    // Emit the label from the previous iteration
    if (FDEEnd) {
      Streamer.EmitLabel(FDEEnd);
      FDEEnd = nullptr;
    }

    if (!NeedsEHFrameSection && Frame.CompactUnwindEncoding !=
          MOFI->getCompactUnwindDwarfEHFrameOnly())
      // Don't generate an EH frame if we don't need one. I.e., it's taken care
      // of by the compact unwind encoding.
      continue;

    CIEKey Key(Frame.Personality, Frame.PersonalityEncoding,
               Frame.LsdaEncoding, Frame.IsSignalFrame, Frame.IsSimple);
    const MCSymbol *&CIEStart = IsEH ? CIEStarts[Key] : DummyDebugKey;
    if (!CIEStart)
      CIEStart = &Emitter.EmitCIE(Streamer, Frame.Personality,
                                  Frame.PersonalityEncoding, Frame.Lsda,
                                  Frame.IsSignalFrame,
                                  Frame.LsdaEncoding,
                                  Frame.IsSimple);

    FDEEnd = Emitter.EmitFDE(Streamer, *CIEStart, Frame);
  }

  Streamer.EmitValueToAlignment(Context.getAsmInfo()->getPointerSize());
  if (FDEEnd)
    Streamer.EmitLabel(FDEEnd);
}

void MCDwarfFrameEmitter::EmitAdvanceLoc(MCObjectStreamer &Streamer,
                                         uint64_t AddrDelta) {
  MCContext &Context = Streamer.getContext();
  SmallString<256> Tmp;
  raw_svector_ostream OS(Tmp);
  MCDwarfFrameEmitter::EncodeAdvanceLoc(Context, AddrDelta, OS);
  Streamer.EmitBytes(OS.str());
}

void MCDwarfFrameEmitter::EncodeAdvanceLoc(MCContext &Context,
                                           uint64_t AddrDelta,
                                           raw_ostream &OS) {
  // Scale the address delta by the minimum instruction length.
  AddrDelta = ScaleAddrDelta(Context, AddrDelta);

  if (AddrDelta == 0) {
  } else if (isUIntN(6, AddrDelta)) {
    uint8_t Opcode = dwarf::DW_CFA_advance_loc | AddrDelta;
    OS << Opcode;
  } else if (isUInt<8>(AddrDelta)) {
    OS << uint8_t(dwarf::DW_CFA_advance_loc1);
    OS << uint8_t(AddrDelta);
  } else if (isUInt<16>(AddrDelta)) {
    OS << uint8_t(dwarf::DW_CFA_advance_loc2);
    if (Context.getAsmInfo()->isLittleEndian())
      support::endian::Writer<support::little>(OS).write<uint16_t>(AddrDelta);
    else
      support::endian::Writer<support::big>(OS).write<uint16_t>(AddrDelta);
  } else {
    assert(isUInt<32>(AddrDelta));
    OS << uint8_t(dwarf::DW_CFA_advance_loc4);
    if (Context.getAsmInfo()->isLittleEndian())
      support::endian::Writer<support::little>(OS).write<uint32_t>(AddrDelta);
    else
      support::endian::Writer<support::big>(OS).write<uint32_t>(AddrDelta);
  }
}
