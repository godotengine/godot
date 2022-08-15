//===-- AsmPrinterDwarf.cpp - AsmPrinter Dwarf Support --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Dwarf emissions parts of AsmPrinter.
//
//===----------------------------------------------------------------------===//

#include "ByteStreamer.h"
#include "DwarfDebug.h"
#include "DwarfExpression.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

#define DEBUG_TYPE "asm-printer"

//===----------------------------------------------------------------------===//
// Dwarf Emission Helper Routines
//===----------------------------------------------------------------------===//

/// EmitSLEB128 - emit the specified signed leb128 value.
void AsmPrinter::EmitSLEB128(int64_t Value, const char *Desc) const {
  if (isVerbose() && Desc)
    OutStreamer->AddComment(Desc);

  OutStreamer->EmitSLEB128IntValue(Value);
}

/// EmitULEB128 - emit the specified signed leb128 value.
void AsmPrinter::EmitULEB128(uint64_t Value, const char *Desc,
                             unsigned PadTo) const {
  if (isVerbose() && Desc)
    OutStreamer->AddComment(Desc);

  OutStreamer->EmitULEB128IntValue(Value, PadTo);
}

/// EmitCFAByte - Emit a .byte 42 directive for a DW_CFA_xxx value.
void AsmPrinter::EmitCFAByte(unsigned Val) const {
  if (isVerbose()) {
    if (Val >= dwarf::DW_CFA_offset && Val < dwarf::DW_CFA_offset + 64)
      OutStreamer->AddComment("DW_CFA_offset + Reg (" +
                              Twine(Val - dwarf::DW_CFA_offset) + ")");
    else
      OutStreamer->AddComment(dwarf::CallFrameString(Val));
  }
  OutStreamer->EmitIntValue(Val, 1);
}

static const char *DecodeDWARFEncoding(unsigned Encoding) {
  switch (Encoding) {
  case dwarf::DW_EH_PE_absptr:
    return "absptr";
  case dwarf::DW_EH_PE_omit:
    return "omit";
  case dwarf::DW_EH_PE_pcrel:
    return "pcrel";
  case dwarf::DW_EH_PE_udata4:
    return "udata4";
  case dwarf::DW_EH_PE_udata8:
    return "udata8";
  case dwarf::DW_EH_PE_sdata4:
    return "sdata4";
  case dwarf::DW_EH_PE_sdata8:
    return "sdata8";
  case dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_udata4:
    return "pcrel udata4";
  case dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4:
    return "pcrel sdata4";
  case dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_udata8:
    return "pcrel udata8";
  case dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata8:
    return "pcrel sdata8";
  case dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_udata4
      :
    return "indirect pcrel udata4";
  case dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4
      :
    return "indirect pcrel sdata4";
  case dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_udata8
      :
    return "indirect pcrel udata8";
  case dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata8
      :
    return "indirect pcrel sdata8";
  }

  return "<unknown encoding>";
}

/// EmitEncodingByte - Emit a .byte 42 directive that corresponds to an
/// encoding.  If verbose assembly output is enabled, we output comments
/// describing the encoding.  Desc is an optional string saying what the
/// encoding is specifying (e.g. "LSDA").
void AsmPrinter::EmitEncodingByte(unsigned Val, const char *Desc) const {
  if (isVerbose()) {
    if (Desc)
      OutStreamer->AddComment(Twine(Desc) + " Encoding = " +
                              Twine(DecodeDWARFEncoding(Val)));
    else
      OutStreamer->AddComment(Twine("Encoding = ") + DecodeDWARFEncoding(Val));
  }

  OutStreamer->EmitIntValue(Val, 1);
}

/// GetSizeOfEncodedValue - Return the size of the encoding in bytes.
unsigned AsmPrinter::GetSizeOfEncodedValue(unsigned Encoding) const {
  if (Encoding == dwarf::DW_EH_PE_omit)
    return 0;

  switch (Encoding & 0x07) {
  default:
    llvm_unreachable("Invalid encoded value.");
  case dwarf::DW_EH_PE_absptr:
    return TM.getDataLayout()->getPointerSize();
  case dwarf::DW_EH_PE_udata2:
    return 2;
  case dwarf::DW_EH_PE_udata4:
    return 4;
  case dwarf::DW_EH_PE_udata8:
    return 8;
  }
}

void AsmPrinter::EmitTTypeReference(const GlobalValue *GV,
                                    unsigned Encoding) const {
  if (GV) {
    const TargetLoweringObjectFile &TLOF = getObjFileLowering();

    const MCExpr *Exp =
        TLOF.getTTypeGlobalReference(GV, Encoding, *Mang, TM, MMI,
                                     *OutStreamer);
    OutStreamer->EmitValue(Exp, GetSizeOfEncodedValue(Encoding));
  } else
    OutStreamer->EmitIntValue(0, GetSizeOfEncodedValue(Encoding));
}

void AsmPrinter::emitDwarfSymbolReference(const MCSymbol *Label,
                                          bool ForceOffset) const {
  if (!ForceOffset) {
    // On COFF targets, we have to emit the special .secrel32 directive.
    if (MAI->needsDwarfSectionOffsetDirective()) {
      OutStreamer->EmitCOFFSecRel32(Label);
      return;
    }

    // If the format uses relocations with dwarf, refer to the symbol directly.
    if (MAI->doesDwarfUseRelocationsAcrossSections()) {
      OutStreamer->EmitSymbolValue(Label, 4);
      return;
    }
  }

  // Otherwise, emit it as a label difference from the start of the section.
  EmitLabelDifference(Label, Label->getSection().getBeginSymbol(), 4);
}

void AsmPrinter::emitDwarfStringOffset(DwarfStringPoolEntryRef S) const {
  if (MAI->doesDwarfUseRelocationsAcrossSections()) {
    emitDwarfSymbolReference(S.getSymbol());
    return;
  }

  // Just emit the offset directly; no need for symbol math.
  EmitInt32(S.getOffset());
}

/// EmitDwarfRegOp - Emit dwarf register operation.
void AsmPrinter::EmitDwarfRegOp(ByteStreamer &Streamer,
                                const MachineLocation &MLoc) const {
  DebugLocDwarfExpression Expr(*MF->getSubtarget().getRegisterInfo(),
                               getDwarfDebug()->getDwarfVersion(), Streamer);
  const MCRegisterInfo *MRI = MMI->getContext().getRegisterInfo();
  int Reg = MRI->getDwarfRegNum(MLoc.getReg(), false);
  if (Reg < 0) {
    // We assume that pointers are always in an addressable register.
    if (MLoc.isIndirect())
      // FIXME: We have no reasonable way of handling errors in here. The
      // caller might be in the middle of a dwarf expression. We should
      // probably assert that Reg >= 0 once debug info generation is more
      // mature.
      return Expr.EmitOp(dwarf::DW_OP_nop,
                         "nop (could not find a dwarf register number)");

    // Attempt to find a valid super- or sub-register.
    if (!Expr.AddMachineRegPiece(MLoc.getReg()))
      Expr.EmitOp(dwarf::DW_OP_nop,
                  "nop (could not find a dwarf register number)");
    return;
  }

  if (MLoc.isIndirect())
    Expr.AddRegIndirect(Reg, MLoc.getOffset());
  else
    Expr.AddReg(Reg);
}

//===----------------------------------------------------------------------===//
// Dwarf Lowering Routines
//===----------------------------------------------------------------------===//

void AsmPrinter::emitCFIInstruction(const MCCFIInstruction &Inst) const {
  switch (Inst.getOperation()) {
  default:
    llvm_unreachable("Unexpected instruction");
  case MCCFIInstruction::OpDefCfaOffset:
    OutStreamer->EmitCFIDefCfaOffset(Inst.getOffset());
    break;
  case MCCFIInstruction::OpDefCfa:
    OutStreamer->EmitCFIDefCfa(Inst.getRegister(), Inst.getOffset());
    break;
  case MCCFIInstruction::OpDefCfaRegister:
    OutStreamer->EmitCFIDefCfaRegister(Inst.getRegister());
    break;
  case MCCFIInstruction::OpOffset:
    OutStreamer->EmitCFIOffset(Inst.getRegister(), Inst.getOffset());
    break;
  case MCCFIInstruction::OpRegister:
    OutStreamer->EmitCFIRegister(Inst.getRegister(), Inst.getRegister2());
    break;
  case MCCFIInstruction::OpWindowSave:
    OutStreamer->EmitCFIWindowSave();
    break;
  case MCCFIInstruction::OpSameValue:
    OutStreamer->EmitCFISameValue(Inst.getRegister());
    break;
  }
}

void AsmPrinter::emitDwarfDIE(const DIE &Die) const {
  // Emit the code (index) for the abbreviation.
  if (isVerbose())
    OutStreamer->AddComment("Abbrev [" + Twine(Die.getAbbrevNumber()) + "] 0x" +
                            Twine::utohexstr(Die.getOffset()) + ":0x" +
                            Twine::utohexstr(Die.getSize()) + " " +
                            dwarf::TagString(Die.getTag()));
  EmitULEB128(Die.getAbbrevNumber());

  // Emit the DIE attribute values.
  for (const auto &V : Die.values()) {
    dwarf::Attribute Attr = V.getAttribute();
    assert(V.getForm() && "Too many attributes for DIE (check abbreviation)");

    if (isVerbose()) {
      OutStreamer->AddComment(dwarf::AttributeString(Attr));
      if (Attr == dwarf::DW_AT_accessibility)
        OutStreamer->AddComment(
            dwarf::AccessibilityString(V.getDIEInteger().getValue()));
    }

    // Emit an attribute using the defined form.
    V.EmitValue(this);
  }

  // Emit the DIE children if any.
  if (Die.hasChildren()) {
    for (auto &Child : Die.children())
      emitDwarfDIE(Child);

    OutStreamer->AddComment("End Of Children Mark");
    EmitInt8(0);
  }
}

void
AsmPrinter::emitDwarfAbbrevs(const std::vector<DIEAbbrev *>& Abbrevs) const {
  // For each abbrevation.
  for (const DIEAbbrev *Abbrev : Abbrevs) {
    // Emit the abbrevations code (base 1 index.)
    EmitULEB128(Abbrev->getNumber(), "Abbreviation Code");

    // Emit the abbreviations data.
    Abbrev->Emit(this);
  }

  // Mark end of abbreviations.
  EmitULEB128(0, "EOM(3)");
}
