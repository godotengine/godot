//===-- llvm/Target/TargetLoweringObjectFile.cpp - Object File Info -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements classes used to handle lowerings specific to common
// object file formats.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//                              Generic Code
//===----------------------------------------------------------------------===//

/// Initialize - this method must be called before any actual lowering is
/// done.  This specifies the current context for codegen, and gives the
/// lowering implementations a chance to set up their default sections.
void TargetLoweringObjectFile::Initialize(MCContext &ctx,
                                          const TargetMachine &TM) {
  Ctx = &ctx;
  DL = TM.getDataLayout();
  InitMCObjectFileInfo(TM.getTargetTriple(), TM.getRelocationModel(),
                       TM.getCodeModel(), *Ctx);
}

TargetLoweringObjectFile::~TargetLoweringObjectFile() {
}

static bool isSuitableForBSS(const GlobalVariable *GV, bool NoZerosInBSS) {
  const Constant *C = GV->getInitializer();

  // Must have zero initializer.
  if (!C->isNullValue())
    return false;

  // Leave constant zeros in readonly constant sections, so they can be shared.
  if (GV->isConstant())
    return false;

  // If the global has an explicit section specified, don't put it in BSS.
  if (GV->hasSection())
    return false;

  // If -nozero-initialized-in-bss is specified, don't ever use BSS.
  if (NoZerosInBSS)
    return false;

  // Otherwise, put it in BSS!
  return true;
}

/// IsNullTerminatedString - Return true if the specified constant (which is
/// known to have a type that is an array of 1/2/4 byte elements) ends with a
/// nul value and contains no other nuls in it.  Note that this is more general
/// than ConstantDataSequential::isString because we allow 2 & 4 byte strings.
static bool IsNullTerminatedString(const Constant *C) {
  // First check: is we have constant array terminated with zero
  if (const ConstantDataSequential *CDS = dyn_cast<ConstantDataSequential>(C)) {
    unsigned NumElts = CDS->getNumElements();
    assert(NumElts != 0 && "Can't have an empty CDS");
    
    if (CDS->getElementAsInteger(NumElts-1) != 0)
      return false; // Not null terminated.
    
    // Verify that the null doesn't occur anywhere else in the string.
    for (unsigned i = 0; i != NumElts-1; ++i)
      if (CDS->getElementAsInteger(i) == 0)
        return false;
    return true;
  }

  // Another possibility: [1 x i8] zeroinitializer
  if (isa<ConstantAggregateZero>(C))
    return cast<ArrayType>(C->getType())->getNumElements() == 1;

  return false;
}

MCSymbol *TargetLoweringObjectFile::getSymbolWithGlobalValueBase(
    const GlobalValue *GV, StringRef Suffix, Mangler &Mang,
    const TargetMachine &TM) const {
  assert(!Suffix.empty());

  SmallString<60> NameStr;
  NameStr += DL->getPrivateGlobalPrefix();
  TM.getNameWithPrefix(NameStr, GV, Mang);
  NameStr.append(Suffix.begin(), Suffix.end());
  return Ctx->getOrCreateSymbol(NameStr);
}

MCSymbol *TargetLoweringObjectFile::getCFIPersonalitySymbol(
    const GlobalValue *GV, Mangler &Mang, const TargetMachine &TM,
    MachineModuleInfo *MMI) const {
  return TM.getSymbol(GV, Mang);
}

void TargetLoweringObjectFile::emitPersonalityValue(MCStreamer &Streamer,
                                                    const TargetMachine &TM,
                                                    const MCSymbol *Sym) const {
}


/// getKindForGlobal - This is a top-level target-independent classifier for
/// a global variable.  Given an global variable and information from TM, it
/// classifies the global in a variety of ways that make various target
/// implementations simpler.  The target implementation is free to ignore this
/// extra info of course.
SectionKind TargetLoweringObjectFile::getKindForGlobal(const GlobalValue *GV,
                                                       const TargetMachine &TM){
  assert(!GV->isDeclaration() && !GV->hasAvailableExternallyLinkage() &&
         "Can only be used for global definitions");

  Reloc::Model ReloModel = TM.getRelocationModel();

  // Early exit - functions should be always in text sections.
  const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV);
  if (!GVar)
    return SectionKind::getText();

  // Handle thread-local data first.
  if (GVar->isThreadLocal()) {
    if (isSuitableForBSS(GVar, TM.Options.NoZerosInBSS))
      return SectionKind::getThreadBSS();
    return SectionKind::getThreadData();
  }

  // Variables with common linkage always get classified as common.
  if (GVar->hasCommonLinkage())
    return SectionKind::getCommon();

  // Variable can be easily put to BSS section.
  if (isSuitableForBSS(GVar, TM.Options.NoZerosInBSS)) {
    if (GVar->hasLocalLinkage())
      return SectionKind::getBSSLocal();
    else if (GVar->hasExternalLinkage())
      return SectionKind::getBSSExtern();
    return SectionKind::getBSS();
  }

  const Constant *C = GVar->getInitializer();

  // If the global is marked constant, we can put it into a mergable section,
  // a mergable string section, or general .data if it contains relocations.
  if (GVar->isConstant()) {
    // If the initializer for the global contains something that requires a
    // relocation, then we may have to drop this into a writable data section
    // even though it is marked const.
    switch (C->getRelocationInfo()) {
    case Constant::NoRelocation:
      // If the global is required to have a unique address, it can't be put
      // into a mergable section: just drop it into the general read-only
      // section instead.
      if (!GVar->hasUnnamedAddr())
        return SectionKind::getReadOnly();
        
      // If initializer is a null-terminated string, put it in a "cstring"
      // section of the right width.
      if (ArrayType *ATy = dyn_cast<ArrayType>(C->getType())) {
        if (IntegerType *ITy =
              dyn_cast<IntegerType>(ATy->getElementType())) {
          if ((ITy->getBitWidth() == 8 || ITy->getBitWidth() == 16 ||
               ITy->getBitWidth() == 32) &&
              IsNullTerminatedString(C)) {
            if (ITy->getBitWidth() == 8)
              return SectionKind::getMergeable1ByteCString();
            if (ITy->getBitWidth() == 16)
              return SectionKind::getMergeable2ByteCString();

            assert(ITy->getBitWidth() == 32 && "Unknown width");
            return SectionKind::getMergeable4ByteCString();
          }
        }
      }

      // Otherwise, just drop it into a mergable constant section.  If we have
      // a section for this size, use it, otherwise use the arbitrary sized
      // mergable section.
      switch (TM.getDataLayout()->getTypeAllocSize(C->getType())) {
      case 4:  return SectionKind::getMergeableConst4();
      case 8:  return SectionKind::getMergeableConst8();
      case 16: return SectionKind::getMergeableConst16();
      default:
        return SectionKind::getReadOnly();
      }

    case Constant::LocalRelocation:
      // In static relocation model, the linker will resolve all addresses, so
      // the relocation entries will actually be constants by the time the app
      // starts up.  However, we can't put this into a mergable section, because
      // the linker doesn't take relocations into consideration when it tries to
      // merge entries in the section.
      if (ReloModel == Reloc::Static)
        return SectionKind::getReadOnly();

      // Otherwise, the dynamic linker needs to fix it up, put it in the
      // writable data.rel.local section.
      return SectionKind::getReadOnlyWithRelLocal();

    case Constant::GlobalRelocations:
      // In static relocation model, the linker will resolve all addresses, so
      // the relocation entries will actually be constants by the time the app
      // starts up.  However, we can't put this into a mergable section, because
      // the linker doesn't take relocations into consideration when it tries to
      // merge entries in the section.
      if (ReloModel == Reloc::Static)
        return SectionKind::getReadOnly();

      // Otherwise, the dynamic linker needs to fix it up, put it in the
      // writable data.rel section.
      return SectionKind::getReadOnlyWithRel();
    }
  }

  // Okay, this isn't a constant.  If the initializer for the global is going
  // to require a runtime relocation by the dynamic linker, put it into a more
  // specific section to improve startup time of the app.  This coalesces these
  // globals together onto fewer pages, improving the locality of the dynamic
  // linker.
  if (ReloModel == Reloc::Static)
    return SectionKind::getDataNoRel();

  switch (C->getRelocationInfo()) {
  case Constant::NoRelocation:
    return SectionKind::getDataNoRel();
  case Constant::LocalRelocation:
    return SectionKind::getDataRelLocal();
  case Constant::GlobalRelocations:
    return SectionKind::getDataRel();
  }
  llvm_unreachable("Invalid relocation");
}

/// This method computes the appropriate section to emit the specified global
/// variable or function definition.  This should not be passed external (or
/// available externally) globals.
MCSection *
TargetLoweringObjectFile::SectionForGlobal(const GlobalValue *GV,
                                           SectionKind Kind, Mangler &Mang,
                                           const TargetMachine &TM) const {
  // Select section name.
  if (GV->hasSection())
    return getExplicitSectionGlobal(GV, Kind, Mang, TM);


  // Use default section depending on the 'type' of global
  return SelectSectionForGlobal(GV, Kind, Mang, TM);
}

MCSection *TargetLoweringObjectFile::getSectionForJumpTable(
    const Function &F, Mangler &Mang, const TargetMachine &TM) const {
  return getSectionForConstant(SectionKind::getReadOnly(), /*C=*/nullptr);
}

bool TargetLoweringObjectFile::shouldPutJumpTableInFunctionSection(
    bool UsesLabelDifference, const Function &F) const {
  // In PIC mode, we need to emit the jump table to the same section as the
  // function body itself, otherwise the label differences won't make sense.
  // FIXME: Need a better predicate for this: what about custom entries?
  if (UsesLabelDifference)
    return true;

  // We should also do if the section name is NULL or function is declared
  // in discardable section
  // FIXME: this isn't the right predicate, should be based on the MCSection
  // for the function.
  if (F.isWeakForLinker())
    return true;

  return false;
}

/// Given a mergable constant with the specified size and relocation
/// information, return a section that it should be placed in.
MCSection *
TargetLoweringObjectFile::getSectionForConstant(SectionKind Kind,
                                                const Constant *C) const {
  if (Kind.isReadOnly() && ReadOnlySection != nullptr)
    return ReadOnlySection;

  return DataSection;
}

/// getTTypeGlobalReference - Return an MCExpr to use for a
/// reference to the specified global variable from exception
/// handling information.
const MCExpr *TargetLoweringObjectFile::getTTypeGlobalReference(
    const GlobalValue *GV, unsigned Encoding, Mangler &Mang,
    const TargetMachine &TM, MachineModuleInfo *MMI,
    MCStreamer &Streamer) const {
  const MCSymbolRefExpr *Ref =
      MCSymbolRefExpr::create(TM.getSymbol(GV, Mang), getContext());

  return getTTypeReference(Ref, Encoding, Streamer);
}

const MCExpr *TargetLoweringObjectFile::
getTTypeReference(const MCSymbolRefExpr *Sym, unsigned Encoding,
                  MCStreamer &Streamer) const {
  switch (Encoding & 0x70) {
  default:
    report_fatal_error("We do not support this DWARF encoding yet!");
  case dwarf::DW_EH_PE_absptr:
    // Do nothing special
    return Sym;
  case dwarf::DW_EH_PE_pcrel: {
    // Emit a label to the streamer for the current position.  This gives us
    // .-foo addressing.
    MCSymbol *PCSym = getContext().createTempSymbol();
    Streamer.EmitLabel(PCSym);
    const MCExpr *PC = MCSymbolRefExpr::create(PCSym, getContext());
    return MCBinaryExpr::createSub(Sym, PC, getContext());
  }
  }
}

const MCExpr *TargetLoweringObjectFile::getDebugThreadLocalSymbol(const MCSymbol *Sym) const {
  // FIXME: It's not clear what, if any, default this should have - perhaps a
  // null return could mean 'no location' & we should just do that here.
  return MCSymbolRefExpr::create(Sym, *Ctx);
}

void TargetLoweringObjectFile::getNameWithPrefix(
    SmallVectorImpl<char> &OutName, const GlobalValue *GV,
    bool CannotUsePrivateLabel, Mangler &Mang, const TargetMachine &TM) const {
  Mang.getNameWithPrefix(OutName, GV, CannotUsePrivateLabel);
}
