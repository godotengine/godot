//===-- llvm/CodeGen/TargetLoweringObjectFileImpl.cpp - Object File Info --===//
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

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;
using namespace dwarf;

//===----------------------------------------------------------------------===//
//                                  ELF
//===----------------------------------------------------------------------===//

MCSymbol *TargetLoweringObjectFileELF::getCFIPersonalitySymbol(
    const GlobalValue *GV, Mangler &Mang, const TargetMachine &TM,
    MachineModuleInfo *MMI) const {
  unsigned Encoding = getPersonalityEncoding();
  if ((Encoding & 0x80) == dwarf::DW_EH_PE_indirect)
    return getContext().getOrCreateSymbol(StringRef("DW.ref.") +
                                          TM.getSymbol(GV, Mang)->getName());
  if ((Encoding & 0x70) == dwarf::DW_EH_PE_absptr)
    return TM.getSymbol(GV, Mang);
  report_fatal_error("We do not support this DWARF encoding yet!");
}

void TargetLoweringObjectFileELF::emitPersonalityValue(MCStreamer &Streamer,
                                                       const TargetMachine &TM,
                                                       const MCSymbol *Sym) const {
  SmallString<64> NameData("DW.ref.");
  NameData += Sym->getName();
  MCSymbolELF *Label =
      cast<MCSymbolELF>(getContext().getOrCreateSymbol(NameData));
  Streamer.EmitSymbolAttribute(Label, MCSA_Hidden);
  Streamer.EmitSymbolAttribute(Label, MCSA_Weak);
  StringRef Prefix = ".data.";
  NameData.insert(NameData.begin(), Prefix.begin(), Prefix.end());
  unsigned Flags = ELF::SHF_ALLOC | ELF::SHF_WRITE | ELF::SHF_GROUP;
  MCSection *Sec = getContext().getELFSection(NameData, ELF::SHT_PROGBITS,
                                              Flags, 0, Label->getName());
  unsigned Size = TM.getDataLayout()->getPointerSize();
  Streamer.SwitchSection(Sec);
  Streamer.EmitValueToAlignment(TM.getDataLayout()->getPointerABIAlignment());
  Streamer.EmitSymbolAttribute(Label, MCSA_ELF_TypeObject);
  const MCExpr *E = MCConstantExpr::create(Size, getContext());
  Streamer.emitELFSize(Label, E);
  Streamer.EmitLabel(Label);

  Streamer.EmitSymbolValue(Sym, Size);
}

const MCExpr *TargetLoweringObjectFileELF::getTTypeGlobalReference(
    const GlobalValue *GV, unsigned Encoding, Mangler &Mang,
    const TargetMachine &TM, MachineModuleInfo *MMI,
    MCStreamer &Streamer) const {

  if (Encoding & dwarf::DW_EH_PE_indirect) {
    MachineModuleInfoELF &ELFMMI = MMI->getObjFileInfo<MachineModuleInfoELF>();

    MCSymbol *SSym = getSymbolWithGlobalValueBase(GV, ".DW.stub", Mang, TM);

    // Add information about the stub reference to ELFMMI so that the stub
    // gets emitted by the asmprinter.
    MachineModuleInfoImpl::StubValueTy &StubSym = ELFMMI.getGVStubEntry(SSym);
    if (!StubSym.getPointer()) {
      MCSymbol *Sym = TM.getSymbol(GV, Mang);
      StubSym = MachineModuleInfoImpl::StubValueTy(Sym, !GV->hasLocalLinkage());
    }

    return TargetLoweringObjectFile::
      getTTypeReference(MCSymbolRefExpr::create(SSym, getContext()),
                        Encoding & ~dwarf::DW_EH_PE_indirect, Streamer);
  }

  return TargetLoweringObjectFile::
    getTTypeGlobalReference(GV, Encoding, Mang, TM, MMI, Streamer);
}

static SectionKind
getELFKindForNamedSection(StringRef Name, SectionKind K) {
  // N.B.: The defaults used in here are no the same ones used in MC.
  // We follow gcc, MC follows gas. For example, given ".section .eh_frame",
  // both gas and MC will produce a section with no flags. Given
  // section(".eh_frame") gcc will produce:
  //
  //   .section   .eh_frame,"a",@progbits
  if (Name.empty() || Name[0] != '.') return K;

  // Some lame default implementation based on some magic section names.
  if (Name == ".bss" ||
      Name.startswith(".bss.") ||
      Name.startswith(".gnu.linkonce.b.") ||
      Name.startswith(".llvm.linkonce.b.") ||
      Name == ".sbss" ||
      Name.startswith(".sbss.") ||
      Name.startswith(".gnu.linkonce.sb.") ||
      Name.startswith(".llvm.linkonce.sb."))
    return SectionKind::getBSS();

  if (Name == ".tdata" ||
      Name.startswith(".tdata.") ||
      Name.startswith(".gnu.linkonce.td.") ||
      Name.startswith(".llvm.linkonce.td."))
    return SectionKind::getThreadData();

  if (Name == ".tbss" ||
      Name.startswith(".tbss.") ||
      Name.startswith(".gnu.linkonce.tb.") ||
      Name.startswith(".llvm.linkonce.tb."))
    return SectionKind::getThreadBSS();

  return K;
}


static unsigned getELFSectionType(StringRef Name, SectionKind K) {

  if (Name == ".init_array")
    return ELF::SHT_INIT_ARRAY;

  if (Name == ".fini_array")
    return ELF::SHT_FINI_ARRAY;

  if (Name == ".preinit_array")
    return ELF::SHT_PREINIT_ARRAY;

  if (K.isBSS() || K.isThreadBSS())
    return ELF::SHT_NOBITS;

  return ELF::SHT_PROGBITS;
}

static unsigned getELFSectionFlags(SectionKind K) {
  unsigned Flags = 0;

  if (!K.isMetadata())
    Flags |= ELF::SHF_ALLOC;

  if (K.isText())
    Flags |= ELF::SHF_EXECINSTR;

  if (K.isWriteable())
    Flags |= ELF::SHF_WRITE;

  if (K.isThreadLocal())
    Flags |= ELF::SHF_TLS;

  if (K.isMergeableCString() || K.isMergeableConst())
    Flags |= ELF::SHF_MERGE;

  if (K.isMergeableCString())
    Flags |= ELF::SHF_STRINGS;

  return Flags;
}

static const Comdat *getELFComdat(const GlobalValue *GV) {
  const Comdat *C = GV->getComdat();
  if (!C)
    return nullptr;

  if (C->getSelectionKind() != Comdat::Any)
    report_fatal_error("ELF COMDATs only support SelectionKind::Any, '" +
                       C->getName() + "' cannot be lowered.");

  return C;
}

MCSection *TargetLoweringObjectFileELF::getExplicitSectionGlobal(
    const GlobalValue *GV, SectionKind Kind, Mangler &Mang,
    const TargetMachine &TM) const {
  StringRef SectionName = GV->getSection();

  // Infer section flags from the section name if we can.
  Kind = getELFKindForNamedSection(SectionName, Kind);

  StringRef Group = "";
  unsigned Flags = getELFSectionFlags(Kind);
  if (const Comdat *C = getELFComdat(GV)) {
    Group = C->getName();
    Flags |= ELF::SHF_GROUP;
  }
  return getContext().getELFSection(SectionName,
                                    getELFSectionType(SectionName, Kind), Flags,
                                    /*EntrySize=*/0, Group);
}

/// Return the section prefix name used by options FunctionsSections and
/// DataSections.
static StringRef getSectionPrefixForGlobal(SectionKind Kind) {
  if (Kind.isText())
    return ".text";
  if (Kind.isReadOnly())
    return ".rodata";
  if (Kind.isBSS())
    return ".bss";
  if (Kind.isThreadData())
    return ".tdata";
  if (Kind.isThreadBSS())
    return ".tbss";
  if (Kind.isDataNoRel())
    return ".data";
  if (Kind.isDataRelLocal())
    return ".data.rel.local";
  if (Kind.isDataRel())
    return ".data.rel";
  if (Kind.isReadOnlyWithRelLocal())
    return ".data.rel.ro.local";
  assert(Kind.isReadOnlyWithRel() && "Unknown section kind");
  return ".data.rel.ro";
}

static MCSectionELF *
selectELFSectionForGlobal(MCContext &Ctx, const GlobalValue *GV,
                          SectionKind Kind, Mangler &Mang,
                          const TargetMachine &TM, bool EmitUniqueSection,
                          unsigned Flags, unsigned *NextUniqueID) {
  unsigned EntrySize = 0;
  if (Kind.isMergeableCString()) {
    if (Kind.isMergeable2ByteCString()) {
      EntrySize = 2;
    } else if (Kind.isMergeable4ByteCString()) {
      EntrySize = 4;
    } else {
      EntrySize = 1;
      assert(Kind.isMergeable1ByteCString() && "unknown string width");
    }
  } else if (Kind.isMergeableConst()) {
    if (Kind.isMergeableConst4()) {
      EntrySize = 4;
    } else if (Kind.isMergeableConst8()) {
      EntrySize = 8;
    } else {
      assert(Kind.isMergeableConst16() && "unknown data width");
      EntrySize = 16;
    }
  }

  StringRef Group = "";
  if (const Comdat *C = getELFComdat(GV)) {
    Flags |= ELF::SHF_GROUP;
    Group = C->getName();
  }

  bool UniqueSectionNames = TM.getUniqueSectionNames();
  SmallString<128> Name;
  if (Kind.isMergeableCString()) {
    // We also need alignment here.
    // FIXME: this is getting the alignment of the character, not the
    // alignment of the global!
    unsigned Align =
        TM.getDataLayout()->getPreferredAlignment(cast<GlobalVariable>(GV));

    std::string SizeSpec = ".rodata.str" + utostr(EntrySize) + ".";
    Name = SizeSpec + utostr(Align);
  } else if (Kind.isMergeableConst()) {
    Name = ".rodata.cst";
    Name += utostr(EntrySize);
  } else {
    Name = getSectionPrefixForGlobal(Kind);
  }

  if (EmitUniqueSection && UniqueSectionNames) {
    Name.push_back('.');
    TM.getNameWithPrefix(Name, GV, Mang, true);
  }
  unsigned UniqueID = ~0;
  if (EmitUniqueSection && !UniqueSectionNames) {
    UniqueID = *NextUniqueID;
    (*NextUniqueID)++;
  }
  return Ctx.getELFSection(Name, getELFSectionType(Name, Kind), Flags,
                           EntrySize, Group, UniqueID);
}

MCSection *TargetLoweringObjectFileELF::SelectSectionForGlobal(
    const GlobalValue *GV, SectionKind Kind, Mangler &Mang,
    const TargetMachine &TM) const {
  unsigned Flags = getELFSectionFlags(Kind);

  // If we have -ffunction-section or -fdata-section then we should emit the
  // global value to a uniqued section specifically for it.
  bool EmitUniqueSection = false;
  if (!(Flags & ELF::SHF_MERGE) && !Kind.isCommon()) {
    if (Kind.isText())
      EmitUniqueSection = TM.getFunctionSections();
    else
      EmitUniqueSection = TM.getDataSections();
  }
  EmitUniqueSection |= GV->hasComdat();

  return selectELFSectionForGlobal(getContext(), GV, Kind, Mang, TM,
                                   EmitUniqueSection, Flags, &NextUniqueID);
}

MCSection *TargetLoweringObjectFileELF::getSectionForJumpTable(
    const Function &F, Mangler &Mang, const TargetMachine &TM) const {
  // If the function can be removed, produce a unique section so that
  // the table doesn't prevent the removal.
  const Comdat *C = F.getComdat();
  bool EmitUniqueSection = TM.getFunctionSections() || C;
  if (!EmitUniqueSection)
    return ReadOnlySection;

  return selectELFSectionForGlobal(getContext(), &F, SectionKind::getReadOnly(),
                                   Mang, TM, EmitUniqueSection, ELF::SHF_ALLOC,
                                   &NextUniqueID);
}

bool TargetLoweringObjectFileELF::shouldPutJumpTableInFunctionSection(
    bool UsesLabelDifference, const Function &F) const {
  // We can always create relative relocations, so use another section
  // that can be marked non-executable.
  return false;
}

/// Given a mergeable constant with the specified size and relocation
/// information, return a section that it should be placed in.
MCSection *
TargetLoweringObjectFileELF::getSectionForConstant(SectionKind Kind,
                                                   const Constant *C) const {
  if (Kind.isMergeableConst4() && MergeableConst4Section)
    return MergeableConst4Section;
  if (Kind.isMergeableConst8() && MergeableConst8Section)
    return MergeableConst8Section;
  if (Kind.isMergeableConst16() && MergeableConst16Section)
    return MergeableConst16Section;
  if (Kind.isReadOnly())
    return ReadOnlySection;

  if (Kind.isReadOnlyWithRelLocal()) return DataRelROLocalSection;
  assert(Kind.isReadOnlyWithRel() && "Unknown section kind");
  return DataRelROSection;
}

static MCSectionELF *getStaticStructorSection(MCContext &Ctx, bool UseInitArray,
                                              bool IsCtor, unsigned Priority,
                                              const MCSymbol *KeySym) {
  std::string Name;
  unsigned Type;
  unsigned Flags = ELF::SHF_ALLOC | ELF::SHF_WRITE;
  StringRef COMDAT = KeySym ? KeySym->getName() : "";

  if (KeySym)
    Flags |= ELF::SHF_GROUP;

  if (UseInitArray) {
    if (IsCtor) {
      Type = ELF::SHT_INIT_ARRAY;
      Name = ".init_array";
    } else {
      Type = ELF::SHT_FINI_ARRAY;
      Name = ".fini_array";
    }
    if (Priority != 65535) {
      Name += '.';
      Name += utostr(Priority);
    }
  } else {
    // The default scheme is .ctor / .dtor, so we have to invert the priority
    // numbering.
    if (IsCtor)
      Name = ".ctors";
    else
      Name = ".dtors";
    if (Priority != 65535) {
      Name += '.';
      Name += utostr(65535 - Priority);
    }
    Type = ELF::SHT_PROGBITS;
  }

  return Ctx.getELFSection(Name, Type, Flags, 0, COMDAT);
}

MCSection *TargetLoweringObjectFileELF::getStaticCtorSection(
    unsigned Priority, const MCSymbol *KeySym) const {
  return getStaticStructorSection(getContext(), UseInitArray, true, Priority,
                                  KeySym);
}

MCSection *TargetLoweringObjectFileELF::getStaticDtorSection(
    unsigned Priority, const MCSymbol *KeySym) const {
  return getStaticStructorSection(getContext(), UseInitArray, false, Priority,
                                  KeySym);
}

void
TargetLoweringObjectFileELF::InitializeELF(bool UseInitArray_) {
  UseInitArray = UseInitArray_;
  if (!UseInitArray)
    return;

  StaticCtorSection = getContext().getELFSection(
      ".init_array", ELF::SHT_INIT_ARRAY, ELF::SHF_WRITE | ELF::SHF_ALLOC);
  StaticDtorSection = getContext().getELFSection(
      ".fini_array", ELF::SHT_FINI_ARRAY, ELF::SHF_WRITE | ELF::SHF_ALLOC);
}

//===----------------------------------------------------------------------===//
//                                 MachO
//===----------------------------------------------------------------------===//

TargetLoweringObjectFileMachO::TargetLoweringObjectFileMachO()
  : TargetLoweringObjectFile() {
  SupportIndirectSymViaGOTPCRel = true;
}

/// emitModuleFlags - Perform code emission for module flags.
void TargetLoweringObjectFileMachO::
emitModuleFlags(MCStreamer &Streamer,
                ArrayRef<Module::ModuleFlagEntry> ModuleFlags,
                Mangler &Mang, const TargetMachine &TM) const {
  unsigned VersionVal = 0;
  unsigned ImageInfoFlags = 0;
  MDNode *LinkerOptions = nullptr;
  StringRef SectionVal;

  for (ArrayRef<Module::ModuleFlagEntry>::iterator
         i = ModuleFlags.begin(), e = ModuleFlags.end(); i != e; ++i) {
    const Module::ModuleFlagEntry &MFE = *i;

    // Ignore flags with 'Require' behavior.
    if (MFE.Behavior == Module::Require)
      continue;

    StringRef Key = MFE.Key->getString();
    Metadata *Val = MFE.Val;

    if (Key == "Objective-C Image Info Version") {
      VersionVal = mdconst::extract<ConstantInt>(Val)->getZExtValue();
    } else if (Key == "Objective-C Garbage Collection" ||
               Key == "Objective-C GC Only" ||
               Key == "Objective-C Is Simulated" ||
               Key == "Objective-C Image Swift Version") {
      ImageInfoFlags |= mdconst::extract<ConstantInt>(Val)->getZExtValue();
    } else if (Key == "Objective-C Image Info Section") {
      SectionVal = cast<MDString>(Val)->getString();
    } else if (Key == "Linker Options") {
      LinkerOptions = cast<MDNode>(Val);
    }
  }

  // Emit the linker options if present.
  if (LinkerOptions) {
    for (unsigned i = 0, e = LinkerOptions->getNumOperands(); i != e; ++i) {
      MDNode *MDOptions = cast<MDNode>(LinkerOptions->getOperand(i));
      SmallVector<std::string, 4> StrOptions;

      // Convert to strings.
      for (unsigned ii = 0, ie = MDOptions->getNumOperands(); ii != ie; ++ii) {
        MDString *MDOption = cast<MDString>(MDOptions->getOperand(ii));
        StrOptions.push_back(MDOption->getString());
      }

      Streamer.EmitLinkerOptions(StrOptions);
    }
  }

  // The section is mandatory. If we don't have it, then we don't have GC info.
  if (SectionVal.empty()) return;

  StringRef Segment, Section;
  unsigned TAA = 0, StubSize = 0;
  bool TAAParsed;
  std::string ErrorCode =
    MCSectionMachO::ParseSectionSpecifier(SectionVal, Segment, Section,
                                          TAA, TAAParsed, StubSize);
  if (!ErrorCode.empty())
    // If invalid, report the error with report_fatal_error.
    report_fatal_error("Invalid section specifier '" + Section + "': " +
                       ErrorCode + ".");

  // Get the section.
  MCSectionMachO *S = getContext().getMachOSection(
      Segment, Section, TAA, StubSize, SectionKind::getDataNoRel());
  Streamer.SwitchSection(S);
  Streamer.EmitLabel(getContext().
                     getOrCreateSymbol(StringRef("L_OBJC_IMAGE_INFO")));
  Streamer.EmitIntValue(VersionVal, 4);
  Streamer.EmitIntValue(ImageInfoFlags, 4);
  Streamer.AddBlankLine();
}

static void checkMachOComdat(const GlobalValue *GV) {
  const Comdat *C = GV->getComdat();
  if (!C)
    return;

  report_fatal_error("MachO doesn't support COMDATs, '" + C->getName() +
                     "' cannot be lowered.");
}

MCSection *TargetLoweringObjectFileMachO::getExplicitSectionGlobal(
    const GlobalValue *GV, SectionKind Kind, Mangler &Mang,
    const TargetMachine &TM) const {
  // Parse the section specifier and create it if valid.
  StringRef Segment, Section;
  unsigned TAA = 0, StubSize = 0;
  bool TAAParsed;

  checkMachOComdat(GV);

  std::string ErrorCode =
    MCSectionMachO::ParseSectionSpecifier(GV->getSection(), Segment, Section,
                                          TAA, TAAParsed, StubSize);
  if (!ErrorCode.empty()) {
    // If invalid, report the error with report_fatal_error.
    report_fatal_error("Global variable '" + GV->getName() +
                       "' has an invalid section specifier '" +
                       GV->getSection() + "': " + ErrorCode + ".");
  }

  // Get the section.
  MCSectionMachO *S =
      getContext().getMachOSection(Segment, Section, TAA, StubSize, Kind);

  // If TAA wasn't set by ParseSectionSpecifier() above,
  // use the value returned by getMachOSection() as a default.
  if (!TAAParsed)
    TAA = S->getTypeAndAttributes();

  // Okay, now that we got the section, verify that the TAA & StubSize agree.
  // If the user declared multiple globals with different section flags, we need
  // to reject it here.
  if (S->getTypeAndAttributes() != TAA || S->getStubSize() != StubSize) {
    // If invalid, report the error with report_fatal_error.
    report_fatal_error("Global variable '" + GV->getName() +
                       "' section type or attributes does not match previous"
                       " section specifier");
  }

  return S;
}

MCSection *TargetLoweringObjectFileMachO::SelectSectionForGlobal(
    const GlobalValue *GV, SectionKind Kind, Mangler &Mang,
    const TargetMachine &TM) const {
  checkMachOComdat(GV);

  // Handle thread local data.
  if (Kind.isThreadBSS()) return TLSBSSSection;
  if (Kind.isThreadData()) return TLSDataSection;

  if (Kind.isText())
    return GV->isWeakForLinker() ? TextCoalSection : TextSection;

  // If this is weak/linkonce, put this in a coalescable section, either in text
  // or data depending on if it is writable.
  if (GV->isWeakForLinker()) {
    if (Kind.isReadOnly())
      return ConstTextCoalSection;
    return DataCoalSection;
  }

  // FIXME: Alignment check should be handled by section classifier.
  if (Kind.isMergeable1ByteCString() &&
      TM.getDataLayout()->getPreferredAlignment(cast<GlobalVariable>(GV)) < 32)
    return CStringSection;

  // Do not put 16-bit arrays in the UString section if they have an
  // externally visible label, this runs into issues with certain linker
  // versions.
  if (Kind.isMergeable2ByteCString() && !GV->hasExternalLinkage() &&
      TM.getDataLayout()->getPreferredAlignment(cast<GlobalVariable>(GV)) < 32)
    return UStringSection;

  // With MachO only variables whose corresponding symbol starts with 'l' or
  // 'L' can be merged, so we only try merging GVs with private linkage.
  if (GV->hasPrivateLinkage() && Kind.isMergeableConst()) {
    if (Kind.isMergeableConst4())
      return FourByteConstantSection;
    if (Kind.isMergeableConst8())
      return EightByteConstantSection;
    if (Kind.isMergeableConst16())
      return SixteenByteConstantSection;
  }

  // Otherwise, if it is readonly, but not something we can specially optimize,
  // just drop it in .const.
  if (Kind.isReadOnly())
    return ReadOnlySection;

  // If this is marked const, put it into a const section.  But if the dynamic
  // linker needs to write to it, put it in the data segment.
  if (Kind.isReadOnlyWithRel())
    return ConstDataSection;

  // Put zero initialized globals with strong external linkage in the
  // DATA, __common section with the .zerofill directive.
  if (Kind.isBSSExtern())
    return DataCommonSection;

  // Put zero initialized globals with local linkage in __DATA,__bss directive
  // with the .zerofill directive (aka .lcomm).
  if (Kind.isBSSLocal())
    return DataBSSSection;

  // Otherwise, just drop the variable in the normal data section.
  return DataSection;
}

MCSection *
TargetLoweringObjectFileMachO::getSectionForConstant(SectionKind Kind,
                                                     const Constant *C) const {
  // If this constant requires a relocation, we have to put it in the data
  // segment, not in the text segment.
  if (Kind.isDataRel() || Kind.isReadOnlyWithRel())
    return ConstDataSection;

  if (Kind.isMergeableConst4())
    return FourByteConstantSection;
  if (Kind.isMergeableConst8())
    return EightByteConstantSection;
  if (Kind.isMergeableConst16())
    return SixteenByteConstantSection;
  return ReadOnlySection;  // .const
}

const MCExpr *TargetLoweringObjectFileMachO::getTTypeGlobalReference(
    const GlobalValue *GV, unsigned Encoding, Mangler &Mang,
    const TargetMachine &TM, MachineModuleInfo *MMI,
    MCStreamer &Streamer) const {
  // The mach-o version of this method defaults to returning a stub reference.

  if (Encoding & DW_EH_PE_indirect) {
    MachineModuleInfoMachO &MachOMMI =
      MMI->getObjFileInfo<MachineModuleInfoMachO>();

    MCSymbol *SSym =
        getSymbolWithGlobalValueBase(GV, "$non_lazy_ptr", Mang, TM);

    // Add information about the stub reference to MachOMMI so that the stub
    // gets emitted by the asmprinter.
    MachineModuleInfoImpl::StubValueTy &StubSym =
      GV->hasHiddenVisibility() ? MachOMMI.getHiddenGVStubEntry(SSym) :
                                  MachOMMI.getGVStubEntry(SSym);
    if (!StubSym.getPointer()) {
      MCSymbol *Sym = TM.getSymbol(GV, Mang);
      StubSym = MachineModuleInfoImpl::StubValueTy(Sym, !GV->hasLocalLinkage());
    }

    return TargetLoweringObjectFile::
      getTTypeReference(MCSymbolRefExpr::create(SSym, getContext()),
                        Encoding & ~dwarf::DW_EH_PE_indirect, Streamer);
  }

  return TargetLoweringObjectFile::getTTypeGlobalReference(GV, Encoding, Mang,
                                                           TM, MMI, Streamer);
}

MCSymbol *TargetLoweringObjectFileMachO::getCFIPersonalitySymbol(
    const GlobalValue *GV, Mangler &Mang, const TargetMachine &TM,
    MachineModuleInfo *MMI) const {
  // The mach-o version of this method defaults to returning a stub reference.
  MachineModuleInfoMachO &MachOMMI =
    MMI->getObjFileInfo<MachineModuleInfoMachO>();

  MCSymbol *SSym = getSymbolWithGlobalValueBase(GV, "$non_lazy_ptr", Mang, TM);

  // Add information about the stub reference to MachOMMI so that the stub
  // gets emitted by the asmprinter.
  MachineModuleInfoImpl::StubValueTy &StubSym = MachOMMI.getGVStubEntry(SSym);
  if (!StubSym.getPointer()) {
    MCSymbol *Sym = TM.getSymbol(GV, Mang);
    StubSym = MachineModuleInfoImpl::StubValueTy(Sym, !GV->hasLocalLinkage());
  }

  return SSym;
}

const MCExpr *TargetLoweringObjectFileMachO::getIndirectSymViaGOTPCRel(
    const MCSymbol *Sym, const MCValue &MV, int64_t Offset,
    MachineModuleInfo *MMI, MCStreamer &Streamer) const {
  // Although MachO 32-bit targets do not explictly have a GOTPCREL relocation
  // as 64-bit do, we replace the GOT equivalent by accessing the final symbol
  // through a non_lazy_ptr stub instead. One advantage is that it allows the
  // computation of deltas to final external symbols. Example:
  //
  //    _extgotequiv:
  //       .long   _extfoo
  //
  //    _delta:
  //       .long   _extgotequiv-_delta
  //
  // is transformed to:
  //
  //    _delta:
  //       .long   L_extfoo$non_lazy_ptr-(_delta+0)
  //
  //       .section        __IMPORT,__pointers,non_lazy_symbol_pointers
  //    L_extfoo$non_lazy_ptr:
  //       .indirect_symbol        _extfoo
  //       .long   0
  //
  MachineModuleInfoMachO &MachOMMI =
    MMI->getObjFileInfo<MachineModuleInfoMachO>();
  MCContext &Ctx = getContext();

  // The offset must consider the original displacement from the base symbol
  // since 32-bit targets don't have a GOTPCREL to fold the PC displacement.
  Offset = -MV.getConstant();
  const MCSymbol *BaseSym = &MV.getSymB()->getSymbol();

  // Access the final symbol via sym$non_lazy_ptr and generate the appropriated
  // non_lazy_ptr stubs.
  SmallString<128> Name;
  StringRef Suffix = "$non_lazy_ptr";
  Name += DL->getPrivateGlobalPrefix();
  Name += Sym->getName();
  Name += Suffix;
  MCSymbol *Stub = Ctx.getOrCreateSymbol(Name);

  MachineModuleInfoImpl::StubValueTy &StubSym = MachOMMI.getGVStubEntry(Stub);
  if (!StubSym.getPointer())
    StubSym = MachineModuleInfoImpl::
      StubValueTy(const_cast<MCSymbol *>(Sym), true /* access indirectly */);

  const MCExpr *BSymExpr =
    MCSymbolRefExpr::create(BaseSym, MCSymbolRefExpr::VK_None, Ctx);
  const MCExpr *LHS =
    MCSymbolRefExpr::create(Stub, MCSymbolRefExpr::VK_None, Ctx);

  if (!Offset)
    return MCBinaryExpr::createSub(LHS, BSymExpr, Ctx);

  const MCExpr *RHS =
    MCBinaryExpr::createAdd(BSymExpr, MCConstantExpr::create(Offset, Ctx), Ctx);
  return MCBinaryExpr::createSub(LHS, RHS, Ctx);
}

//===----------------------------------------------------------------------===//
//                                  COFF
//===----------------------------------------------------------------------===//

static unsigned
getCOFFSectionFlags(SectionKind K) {
  unsigned Flags = 0;

  if (K.isMetadata())
    Flags |=
      COFF::IMAGE_SCN_MEM_DISCARDABLE;
  else if (K.isText())
    Flags |=
      COFF::IMAGE_SCN_MEM_EXECUTE |
      COFF::IMAGE_SCN_MEM_READ |
      COFF::IMAGE_SCN_CNT_CODE;
  else if (K.isBSS())
    Flags |=
      COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA |
      COFF::IMAGE_SCN_MEM_READ |
      COFF::IMAGE_SCN_MEM_WRITE;
  else if (K.isThreadLocal())
    Flags |=
      COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
      COFF::IMAGE_SCN_MEM_READ |
      COFF::IMAGE_SCN_MEM_WRITE;
  else if (K.isReadOnly() || K.isReadOnlyWithRel())
    Flags |=
      COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
      COFF::IMAGE_SCN_MEM_READ;
  else if (K.isWriteable())
    Flags |=
      COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
      COFF::IMAGE_SCN_MEM_READ |
      COFF::IMAGE_SCN_MEM_WRITE;

  return Flags;
}

static const GlobalValue *getComdatGVForCOFF(const GlobalValue *GV) {
  const Comdat *C = GV->getComdat();
  assert(C && "expected GV to have a Comdat!");

  StringRef ComdatGVName = C->getName();
  const GlobalValue *ComdatGV = GV->getParent()->getNamedValue(ComdatGVName);
  if (!ComdatGV)
    report_fatal_error("Associative COMDAT symbol '" + ComdatGVName +
                       "' does not exist.");

  if (ComdatGV->getComdat() != C)
    report_fatal_error("Associative COMDAT symbol '" + ComdatGVName +
                       "' is not a key for its COMDAT.");

  return ComdatGV;
}

static int getSelectionForCOFF(const GlobalValue *GV) {
  if (const Comdat *C = GV->getComdat()) {
    const GlobalValue *ComdatKey = getComdatGVForCOFF(GV);
    if (const auto *GA = dyn_cast<GlobalAlias>(ComdatKey))
      ComdatKey = GA->getBaseObject();
    if (ComdatKey == GV) {
      switch (C->getSelectionKind()) {
      case Comdat::Any:
        return COFF::IMAGE_COMDAT_SELECT_ANY;
      case Comdat::ExactMatch:
        return COFF::IMAGE_COMDAT_SELECT_EXACT_MATCH;
      case Comdat::Largest:
        return COFF::IMAGE_COMDAT_SELECT_LARGEST;
      case Comdat::NoDuplicates:
        return COFF::IMAGE_COMDAT_SELECT_NODUPLICATES;
      case Comdat::SameSize:
        return COFF::IMAGE_COMDAT_SELECT_SAME_SIZE;
      }
    } else {
      return COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE;
    }
  }
  return 0;
}

MCSection *TargetLoweringObjectFileCOFF::getExplicitSectionGlobal(
    const GlobalValue *GV, SectionKind Kind, Mangler &Mang,
    const TargetMachine &TM) const {
  int Selection = 0;
  unsigned Characteristics = getCOFFSectionFlags(Kind);
  StringRef Name = GV->getSection();
  StringRef COMDATSymName = "";
  if (GV->hasComdat()) {
    Selection = getSelectionForCOFF(GV);
    const GlobalValue *ComdatGV;
    if (Selection == COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE)
      ComdatGV = getComdatGVForCOFF(GV);
    else
      ComdatGV = GV;

    if (!ComdatGV->hasPrivateLinkage()) {
      MCSymbol *Sym = TM.getSymbol(ComdatGV, Mang);
      COMDATSymName = Sym->getName();
      Characteristics |= COFF::IMAGE_SCN_LNK_COMDAT;
    } else {
      Selection = 0;
    }
  }
  return getContext().getCOFFSection(Name,
                                     Characteristics,
                                     Kind,
                                     COMDATSymName,
                                     Selection);
}

static const char *getCOFFSectionNameForUniqueGlobal(SectionKind Kind) {
  if (Kind.isText())
    return ".text";
  if (Kind.isBSS())
    return ".bss";
  if (Kind.isThreadLocal())
    return ".tls$";
  if (Kind.isReadOnly() || Kind.isReadOnlyWithRel())
    return ".rdata";
  return ".data";
}

MCSection *TargetLoweringObjectFileCOFF::SelectSectionForGlobal(
    const GlobalValue *GV, SectionKind Kind, Mangler &Mang,
    const TargetMachine &TM) const {
  // If we have -ffunction-sections then we should emit the global value to a
  // uniqued section specifically for it.
  bool EmitUniquedSection;
  if (Kind.isText())
    EmitUniquedSection = TM.getFunctionSections();
  else
    EmitUniquedSection = TM.getDataSections();

  if ((EmitUniquedSection && !Kind.isCommon()) || GV->hasComdat()) {
    const char *Name = getCOFFSectionNameForUniqueGlobal(Kind);
    unsigned Characteristics = getCOFFSectionFlags(Kind);

    Characteristics |= COFF::IMAGE_SCN_LNK_COMDAT;
    int Selection = getSelectionForCOFF(GV);
    if (!Selection)
      Selection = COFF::IMAGE_COMDAT_SELECT_NODUPLICATES;
    const GlobalValue *ComdatGV;
    if (GV->hasComdat())
      ComdatGV = getComdatGVForCOFF(GV);
    else
      ComdatGV = GV;

    if (!ComdatGV->hasPrivateLinkage()) {
      MCSymbol *Sym = TM.getSymbol(ComdatGV, Mang);
      StringRef COMDATSymName = Sym->getName();
      return getContext().getCOFFSection(Name, Characteristics, Kind,
                                         COMDATSymName, Selection);
    } else {
      SmallString<256> TmpData;
      getNameWithPrefix(TmpData, GV, /*CannotUsePrivateLabel=*/true, Mang, TM);
      return getContext().getCOFFSection(Name, Characteristics, Kind, TmpData,
                                         Selection);
    }
  }

  if (Kind.isText())
    return TextSection;

  if (Kind.isThreadLocal())
    return TLSDataSection;

  if (Kind.isReadOnly() || Kind.isReadOnlyWithRel())
    return ReadOnlySection;

  // Note: we claim that common symbols are put in BSSSection, but they are
  // really emitted with the magic .comm directive, which creates a symbol table
  // entry but not a section.
  if (Kind.isBSS() || Kind.isCommon())
    return BSSSection;

  return DataSection;
}

void TargetLoweringObjectFileCOFF::getNameWithPrefix(
    SmallVectorImpl<char> &OutName, const GlobalValue *GV,
    bool CannotUsePrivateLabel, Mangler &Mang, const TargetMachine &TM) const {
  if (GV->hasPrivateLinkage() &&
      ((isa<Function>(GV) && TM.getFunctionSections()) ||
       (isa<GlobalVariable>(GV) && TM.getDataSections())))
    CannotUsePrivateLabel = true;

  Mang.getNameWithPrefix(OutName, GV, CannotUsePrivateLabel);
}

MCSection *TargetLoweringObjectFileCOFF::getSectionForJumpTable(
    const Function &F, Mangler &Mang, const TargetMachine &TM) const {
  // If the function can be removed, produce a unique section so that
  // the table doesn't prevent the removal.
  const Comdat *C = F.getComdat();
  bool EmitUniqueSection = TM.getFunctionSections() || C;
  if (!EmitUniqueSection)
    return ReadOnlySection;

  // FIXME: we should produce a symbol for F instead.
  if (F.hasPrivateLinkage())
    return ReadOnlySection;

  MCSymbol *Sym = TM.getSymbol(&F, Mang);
  StringRef COMDATSymName = Sym->getName();

  SectionKind Kind = SectionKind::getReadOnly();
  const char *Name = getCOFFSectionNameForUniqueGlobal(Kind);
  unsigned Characteristics = getCOFFSectionFlags(Kind);
  Characteristics |= COFF::IMAGE_SCN_LNK_COMDAT;

  return getContext().getCOFFSection(Name, Characteristics, Kind, COMDATSymName,
                                     COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE);
}

void TargetLoweringObjectFileCOFF::
emitModuleFlags(MCStreamer &Streamer,
                ArrayRef<Module::ModuleFlagEntry> ModuleFlags,
                Mangler &Mang, const TargetMachine &TM) const {
  MDNode *LinkerOptions = nullptr;

  // Look for the "Linker Options" flag, since it's the only one we support.
  for (ArrayRef<Module::ModuleFlagEntry>::iterator
       i = ModuleFlags.begin(), e = ModuleFlags.end(); i != e; ++i) {
    const Module::ModuleFlagEntry &MFE = *i;
    StringRef Key = MFE.Key->getString();
    Metadata *Val = MFE.Val;
    if (Key == "Linker Options") {
      LinkerOptions = cast<MDNode>(Val);
      break;
    }
  }
  if (!LinkerOptions)
    return;

  // Emit the linker options to the linker .drectve section.  According to the
  // spec, this section is a space-separated string containing flags for linker.
  MCSection *Sec = getDrectveSection();
  Streamer.SwitchSection(Sec);
  for (unsigned i = 0, e = LinkerOptions->getNumOperands(); i != e; ++i) {
    MDNode *MDOptions = cast<MDNode>(LinkerOptions->getOperand(i));
    for (unsigned ii = 0, ie = MDOptions->getNumOperands(); ii != ie; ++ii) {
      MDString *MDOption = cast<MDString>(MDOptions->getOperand(ii));
      // Lead with a space for consistency with our dllexport implementation.
      std::string Directive(" ");
      Directive.append(MDOption->getString());
      Streamer.EmitBytes(Directive);
    }
  }
}

MCSection *TargetLoweringObjectFileCOFF::getStaticCtorSection(
    unsigned Priority, const MCSymbol *KeySym) const {
  return getContext().getAssociativeCOFFSection(
      cast<MCSectionCOFF>(StaticCtorSection), KeySym);
}

MCSection *TargetLoweringObjectFileCOFF::getStaticDtorSection(
    unsigned Priority, const MCSymbol *KeySym) const {
  return getContext().getAssociativeCOFFSection(
      cast<MCSectionCOFF>(StaticDtorSection), KeySym);
}

void TargetLoweringObjectFileCOFF::emitLinkerFlagsForGlobal(
    raw_ostream &OS, const GlobalValue *GV, const Mangler &Mang) const {
  if (!GV->hasDLLExportStorageClass() || GV->isDeclaration())
    return;

  const Triple &TT = getTargetTriple();

  if (TT.isKnownWindowsMSVCEnvironment())
    OS << " /EXPORT:";
  else
    OS << " -export:";

  if (TT.isWindowsGNUEnvironment() || TT.isWindowsCygwinEnvironment()) {
    std::string Flag;
    raw_string_ostream FlagOS(Flag);
    Mang.getNameWithPrefix(FlagOS, GV, false);
    FlagOS.flush();
    if (Flag[0] == DL->getGlobalPrefix())
      OS << Flag.substr(1);
    else
      OS << Flag;
  } else {
    Mang.getNameWithPrefix(OS, GV, false);
  }

  if (!GV->getValueType()->isFunctionTy()) {
    if (TT.isKnownWindowsMSVCEnvironment())
      OS << ",DATA";
    else
      OS << ",data";
  }
}
