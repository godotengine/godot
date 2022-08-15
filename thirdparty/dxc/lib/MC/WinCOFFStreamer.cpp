//===-- llvm/MC/WinCOFFStreamer.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains an implementation of a Windows COFF object file streamer.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbolCOFF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCWinCOFFStreamer.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "WinCOFFStreamer"

namespace llvm {
MCWinCOFFStreamer::MCWinCOFFStreamer(MCContext &Context, MCAsmBackend &MAB,
                                     MCCodeEmitter &CE, raw_pwrite_stream &OS)
    : MCObjectStreamer(Context, MAB, OS, &CE), CurSymbol(nullptr) {}

void MCWinCOFFStreamer::EmitInstToData(const MCInst &Inst,
                                       const MCSubtargetInfo &STI) {
  MCDataFragment *DF = getOrCreateDataFragment();

  SmallVector<MCFixup, 4> Fixups;
  SmallString<256> Code;
  raw_svector_ostream VecOS(Code);
  getAssembler().getEmitter().encodeInstruction(Inst, VecOS, Fixups, STI);
  VecOS.flush();

  // Add the fixups and data.
  for (unsigned i = 0, e = Fixups.size(); i != e; ++i) {
    Fixups[i].setOffset(Fixups[i].getOffset() + DF->getContents().size());
    DF->getFixups().push_back(Fixups[i]);
  }

  DF->getContents().append(Code.begin(), Code.end());
}

void MCWinCOFFStreamer::InitSections(bool NoExecStack) {
  // FIXME: this is identical to the ELF one.
  // This emulates the same behavior of GNU as. This makes it easier
  // to compare the output as the major sections are in the same order.
  SwitchSection(getContext().getObjectFileInfo()->getTextSection());
  EmitCodeAlignment(4);

  SwitchSection(getContext().getObjectFileInfo()->getDataSection());
  EmitCodeAlignment(4);

  SwitchSection(getContext().getObjectFileInfo()->getBSSSection());
  EmitCodeAlignment(4);

  SwitchSection(getContext().getObjectFileInfo()->getTextSection());
}

void MCWinCOFFStreamer::EmitLabel(MCSymbol *Symbol) {
  assert(Symbol->isUndefined() && "Cannot define a symbol twice!");
  MCObjectStreamer::EmitLabel(Symbol);
}

void MCWinCOFFStreamer::EmitAssemblerFlag(MCAssemblerFlag Flag) {
  llvm_unreachable("not implemented");
}

void MCWinCOFFStreamer::EmitThumbFunc(MCSymbol *Func) {
  llvm_unreachable("not implemented");
}

bool MCWinCOFFStreamer::EmitSymbolAttribute(MCSymbol *Symbol,
                                            MCSymbolAttr Attribute) {
  assert(Symbol && "Symbol must be non-null!");
  assert((!Symbol->isInSection() ||
          Symbol->getSection().getVariant() == MCSection::SV_COFF) &&
         "Got non-COFF section in the COFF backend!");

  getAssembler().registerSymbol(*Symbol);

  switch (Attribute) {
  default: return false;
  case MCSA_WeakReference:
  case MCSA_Weak:
    cast<MCSymbolCOFF>(Symbol)->setIsWeakExternal();
    Symbol->setExternal(true);
    break;
  case MCSA_Global:
    Symbol->setExternal(true);
    break;
  }

  return true;
}

void MCWinCOFFStreamer::EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {
  llvm_unreachable("not implemented");
}

void MCWinCOFFStreamer::BeginCOFFSymbolDef(MCSymbol const *Symbol) {
  assert((!Symbol->isInSection() ||
          Symbol->getSection().getVariant() == MCSection::SV_COFF) &&
         "Got non-COFF section in the COFF backend!");

  if (CurSymbol)
    FatalError("starting a new symbol definition without completing the "
               "previous one");
  CurSymbol = Symbol;
}

void MCWinCOFFStreamer::EmitCOFFSymbolStorageClass(int StorageClass) {
  if (!CurSymbol)
    FatalError("storage class specified outside of symbol definition");

  if (StorageClass & ~COFF::SSC_Invalid)
    FatalError("storage class value '" + Twine(StorageClass) +
               "' out of range");

  getAssembler().registerSymbol(*CurSymbol);
  cast<MCSymbolCOFF>(CurSymbol)->setClass((uint16_t)StorageClass);
}

void MCWinCOFFStreamer::EmitCOFFSymbolType(int Type) {
  if (!CurSymbol)
    FatalError("symbol type specified outside of a symbol definition");

  if (Type & ~0xffff)
    FatalError("type value '" + Twine(Type) + "' out of range");

  getAssembler().registerSymbol(*CurSymbol);
  cast<MCSymbolCOFF>(CurSymbol)->setType((uint16_t)Type);
}

void MCWinCOFFStreamer::EndCOFFSymbolDef() {
  if (!CurSymbol)
    FatalError("ending symbol definition without starting one");
  CurSymbol = nullptr;
}

void MCWinCOFFStreamer::EmitCOFFSafeSEH(MCSymbol const *Symbol) {
  // SafeSEH is a feature specific to 32-bit x86.  It does not exist (and is
  // unnecessary) on all platforms which use table-based exception dispatch.
  if (getContext().getObjectFileInfo()->getTargetTriple().getArch() !=
      Triple::x86)
    return;

  const MCSymbolCOFF *CSymbol = cast<MCSymbolCOFF>(Symbol);
  if (CSymbol->isSafeSEH())
    return;

  MCSection *SXData = getContext().getObjectFileInfo()->getSXDataSection();
  getAssembler().registerSection(*SXData);
  if (SXData->getAlignment() < 4)
    SXData->setAlignment(4);

  new MCSafeSEHFragment(Symbol, SXData);

  getAssembler().registerSymbol(*Symbol);
  CSymbol->setIsSafeSEH();

  // The Microsoft linker requires that the symbol type of a handler be
  // function. Go ahead and oblige it here.
  CSymbol->setType(COFF::IMAGE_SYM_DTYPE_FUNCTION
                   << COFF::SCT_COMPLEX_TYPE_SHIFT);
}

void MCWinCOFFStreamer::EmitCOFFSectionIndex(MCSymbol const *Symbol) {
  MCDataFragment *DF = getOrCreateDataFragment();
  const MCSymbolRefExpr *SRE = MCSymbolRefExpr::create(Symbol, getContext());
  MCFixup Fixup = MCFixup::create(DF->getContents().size(), SRE, FK_SecRel_2);
  DF->getFixups().push_back(Fixup);
  DF->getContents().resize(DF->getContents().size() + 2, 0);
}

void MCWinCOFFStreamer::EmitCOFFSecRel32(MCSymbol const *Symbol) {
  MCDataFragment *DF = getOrCreateDataFragment();
  const MCSymbolRefExpr *SRE = MCSymbolRefExpr::create(Symbol, getContext());
  MCFixup Fixup = MCFixup::create(DF->getContents().size(), SRE, FK_SecRel_4);
  DF->getFixups().push_back(Fixup);
  DF->getContents().resize(DF->getContents().size() + 4, 0);
}

void MCWinCOFFStreamer::EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                         unsigned ByteAlignment) {
  assert((!Symbol->isInSection() ||
          Symbol->getSection().getVariant() == MCSection::SV_COFF) &&
         "Got non-COFF section in the COFF backend!");

  const Triple &T = getContext().getObjectFileInfo()->getTargetTriple();
  if (T.isKnownWindowsMSVCEnvironment()) {
    if (ByteAlignment > 32)
      report_fatal_error("alignment is limited to 32-bytes");

    // Round size up to alignment so that we will honor the alignment request.
    Size = std::max(Size, static_cast<uint64_t>(ByteAlignment));
  }

  AssignSection(Symbol, nullptr);

  getAssembler().registerSymbol(*Symbol);
  Symbol->setExternal(true);
  Symbol->setCommon(Size, ByteAlignment);

  if (!T.isKnownWindowsMSVCEnvironment() && ByteAlignment > 1) {
    SmallString<128> Directive;
    raw_svector_ostream OS(Directive);
    const MCObjectFileInfo *MFI = getContext().getObjectFileInfo();

    OS << " -aligncomm:\"" << Symbol->getName() << "\","
       << Log2_32_Ceil(ByteAlignment);
    OS.flush();

    PushSection();
    SwitchSection(MFI->getDrectveSection());
    EmitBytes(Directive);
    PopSection();
  }
}

void MCWinCOFFStreamer::EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                              unsigned ByteAlignment) {
  assert(!Symbol->isInSection() && "Symbol must not already have a section!");

  MCSection *Section = getContext().getObjectFileInfo()->getBSSSection();
  getAssembler().registerSection(*Section);
  if (Section->getAlignment() < ByteAlignment)
    Section->setAlignment(ByteAlignment);

  getAssembler().registerSymbol(*Symbol);
  Symbol->setExternal(false);

  AssignSection(Symbol, Section);

  if (ByteAlignment != 1)
    new MCAlignFragment(ByteAlignment, /*Value=*/0, /*ValueSize=*/0,
                        ByteAlignment, Section);

  MCFillFragment *Fragment = new MCFillFragment(
      /*Value=*/0, /*ValueSize=*/0, Size, Section);
  Symbol->setFragment(Fragment);
}

void MCWinCOFFStreamer::EmitZerofill(MCSection *Section, MCSymbol *Symbol,
                                     uint64_t Size, unsigned ByteAlignment) {
  llvm_unreachable("not implemented");
}

void MCWinCOFFStreamer::EmitTBSSSymbol(MCSection *Section, MCSymbol *Symbol,
                                       uint64_t Size, unsigned ByteAlignment) {
  llvm_unreachable("not implemented");
}

void MCWinCOFFStreamer::EmitFileDirective(StringRef Filename) {
  getAssembler().addFileName(Filename);
}

// TODO: Implement this if you want to emit .comment section in COFF obj files.
void MCWinCOFFStreamer::EmitIdent(StringRef IdentString) {
  llvm_unreachable("not implemented");
}

void MCWinCOFFStreamer::EmitWinEHHandlerData() {
  llvm_unreachable("not implemented");
}

void MCWinCOFFStreamer::FinishImpl() {
  MCObjectStreamer::FinishImpl();
}

LLVM_ATTRIBUTE_NORETURN
void MCWinCOFFStreamer::FatalError(const Twine &Msg) const {
  getContext().reportFatalError(SMLoc(), Msg);
}
}

