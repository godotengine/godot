//===-- MCAsmInfo.cpp - Asm Info -------------------------------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Dwarf.h"
#include <cctype>
#include <cstring>
using namespace llvm;

MCAsmInfo::MCAsmInfo() {
  PointerSize = 4;
  CalleeSaveStackSlotSize = 4;

  IsLittleEndian = true;
  StackGrowsUp = false;
  HasSubsectionsViaSymbols = false;
  HasMachoZeroFillDirective = false;
  HasMachoTBSSDirective = false;
  HasStaticCtorDtorReferenceInStaticMode = false;
  MaxInstLength = 4;
  MinInstAlignment = 1;
  DollarIsPC = false;
  SeparatorString = ";";
  CommentString = "#";
  LabelSuffix = ":";
  UseAssignmentForEHBegin = false;
  NeedsLocalForSize = false;
  PrivateGlobalPrefix = "L";
  PrivateLabelPrefix = PrivateGlobalPrefix;
  LinkerPrivateGlobalPrefix = "";
  InlineAsmStart = "APP";
  InlineAsmEnd = "NO_APP";
  Code16Directive = ".code16";
  Code32Directive = ".code32";
  Code64Directive = ".code64";
  AssemblerDialect = 0;
  AllowAtInName = false;
  SupportsQuotedNames = true;
  UseDataRegionDirectives = false;
  ZeroDirective = "\t.zero\t";
  AsciiDirective = "\t.ascii\t";
  AscizDirective = "\t.asciz\t";
  Data8bitsDirective = "\t.byte\t";
  Data16bitsDirective = "\t.short\t";
  Data32bitsDirective = "\t.long\t";
  Data64bitsDirective = "\t.quad\t";
  SunStyleELFSectionSwitchSyntax = false;
  UsesELFSectionDirectiveForBSS = false;
  AlignmentIsInBytes = true;
  TextAlignFillValue = 0;
  GPRel64Directive = nullptr;
  GPRel32Directive = nullptr;
  GlobalDirective = "\t.globl\t";
  SetDirectiveSuppressesReloc = false;
  HasAggressiveSymbolFolding = true;
  COMMDirectiveAlignmentIsInBytes = true;
  LCOMMDirectiveAlignmentType = LCOMM::NoAlignment;
  HasFunctionAlignment = true;
  HasDotTypeDotSizeDirective = true;
  HasSingleParameterDotFile = true;
  HasIdentDirective = false;
  HasNoDeadStrip = false;
  WeakDirective = "\t.weak\t";
  WeakRefDirective = nullptr;
  HasWeakDefDirective = false;
  HasWeakDefCanBeHiddenDirective = false;
  HasLinkOnceDirective = false;
  HiddenVisibilityAttr = MCSA_Hidden;
  HiddenDeclarationVisibilityAttr = MCSA_Hidden;
  ProtectedVisibilityAttr = MCSA_Protected;
  SupportsDebugInformation = false;
  ExceptionsType = ExceptionHandling::None;
  WinEHEncodingType = WinEH::EncodingType::Invalid;
  DwarfUsesRelocationsAcrossSections = true;
  DwarfFDESymbolsUseAbsDiff = false;
  DwarfRegNumForCFI = false;
  NeedsDwarfSectionOffsetDirective = false;
  UseParensForSymbolVariant = false;
  UseLogicalShr = true;

  // FIXME: Clang's logic should be synced with the logic used to initialize
  //        this member and the two implementations should be merged.
  // For reference:
  // - Solaris always enables the integrated assembler by default
  //   - SparcELFMCAsmInfo and X86ELFMCAsmInfo are handling this case
  // - Windows always enables the integrated assembler by default
  //   - MCAsmInfoCOFF is handling this case, should it be MCAsmInfoMicrosoft?
  // - MachO targets always enables the integrated assembler by default
  //   - MCAsmInfoDarwin is handling this case
  // - Generic_GCC toolchains enable the integrated assembler on a per
  //   architecture basis.
  //   - The target subclasses for AArch64, ARM, and X86 handle these cases
  UseIntegratedAssembler = false;

  CompressDebugSections = false;
}

MCAsmInfo::~MCAsmInfo() {
}

bool MCAsmInfo::isSectionAtomizableBySymbols(const MCSection &Section) const {
  return false;
}

const MCExpr *
MCAsmInfo::getExprForPersonalitySymbol(const MCSymbol *Sym,
                                       unsigned Encoding,
                                       MCStreamer &Streamer) const {
  return getExprForFDESymbol(Sym, Encoding, Streamer);
}

const MCExpr *
MCAsmInfo::getExprForFDESymbol(const MCSymbol *Sym,
                               unsigned Encoding,
                               MCStreamer &Streamer) const {
  if (!(Encoding & dwarf::DW_EH_PE_pcrel))
    return MCSymbolRefExpr::create(Sym, Streamer.getContext());

  MCContext &Context = Streamer.getContext();
  const MCExpr *Res = MCSymbolRefExpr::create(Sym, Context);
  MCSymbol *PCSym = Context.createTempSymbol();
  Streamer.EmitLabel(PCSym);
  const MCExpr *PC = MCSymbolRefExpr::create(PCSym, Context);
  return MCBinaryExpr::createSub(Res, PC, Context);
}

static bool isAcceptableChar(char C) {
  return (C >= 'a' && C <= 'z') || (C >= 'A' && C <= 'Z') ||
         (C >= '0' && C <= '9') || C == '_' || C == '$' || C == '.' || C == '@';
}

bool MCAsmInfo::isValidUnquotedName(StringRef Name) const {
  if (Name.empty())
    return false;

  // If any of the characters in the string is an unacceptable character, force
  // quotes.
  for (char C : Name) {
    if (!isAcceptableChar(C))
      return false;
  }

  return true;
}
