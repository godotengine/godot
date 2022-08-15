//==-- llvm/CodeGen/TargetLoweringObjectFileImpl.h - Object Info -*- C++ -*-==//
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

#ifndef LLVM_CODEGEN_TARGETLOWERINGOBJECTFILEIMPL_H
#define LLVM_CODEGEN_TARGETLOWERINGOBJECTFILEIMPL_H

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/SectionKind.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

namespace llvm {
  class MachineModuleInfo;
  class Mangler;
  class MCAsmInfo;
  class MCExpr;
  class MCSection;
  class MCSectionMachO;
  class MCSymbol;
  class MCContext;
  class GlobalValue;
  class TargetMachine;


class TargetLoweringObjectFileELF : public TargetLoweringObjectFile {
  bool UseInitArray;
  mutable unsigned NextUniqueID = 0;

public:
  TargetLoweringObjectFileELF() : UseInitArray(false) {}

  ~TargetLoweringObjectFileELF() override {}

  void emitPersonalityValue(MCStreamer &Streamer, const TargetMachine &TM,
                            const MCSymbol *Sym) const override;

  /// Given a constant with the SectionKind, return a section that it should be
  /// placed in.
  MCSection *getSectionForConstant(SectionKind Kind,
                                   const Constant *C) const override;

  MCSection *getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind,
                                      Mangler &Mang,
                                      const TargetMachine &TM) const override;

  MCSection *SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                                    Mangler &Mang,
                                    const TargetMachine &TM) const override;

  MCSection *getSectionForJumpTable(const Function &F, Mangler &Mang,
                                    const TargetMachine &TM) const override;

  bool shouldPutJumpTableInFunctionSection(bool UsesLabelDifference,
                                           const Function &F) const override;

  /// Return an MCExpr to use for a reference to the specified type info global
  /// variable from exception handling information.
  const MCExpr *
  getTTypeGlobalReference(const GlobalValue *GV, unsigned Encoding,
                          Mangler &Mang, const TargetMachine &TM,
                          MachineModuleInfo *MMI,
                          MCStreamer &Streamer) const override;

  // The symbol that gets passed to .cfi_personality.
  MCSymbol *getCFIPersonalitySymbol(const GlobalValue *GV, Mangler &Mang,
                                    const TargetMachine &TM,
                                    MachineModuleInfo *MMI) const override;

  void InitializeELF(bool UseInitArray_);
  MCSection *getStaticCtorSection(unsigned Priority,
                                  const MCSymbol *KeySym) const override;
  MCSection *getStaticDtorSection(unsigned Priority,
                                  const MCSymbol *KeySym) const override;
};



class TargetLoweringObjectFileMachO : public TargetLoweringObjectFile {
public:
  ~TargetLoweringObjectFileMachO() override {}
  TargetLoweringObjectFileMachO();

  /// Emit the module flags that specify the garbage collection information.
  void emitModuleFlags(MCStreamer &Streamer,
                       ArrayRef<Module::ModuleFlagEntry> ModuleFlags,
                       Mangler &Mang, const TargetMachine &TM) const override;

  MCSection *SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                                    Mangler &Mang,
                                    const TargetMachine &TM) const override;

  MCSection *getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind,
                                      Mangler &Mang,
                                      const TargetMachine &TM) const override;

  MCSection *getSectionForConstant(SectionKind Kind,
                                   const Constant *C) const override;

  /// The mach-o version of this method defaults to returning a stub reference.
  const MCExpr *
  getTTypeGlobalReference(const GlobalValue *GV, unsigned Encoding,
                          Mangler &Mang, const TargetMachine &TM,
                          MachineModuleInfo *MMI,
                          MCStreamer &Streamer) const override;

  // The symbol that gets passed to .cfi_personality.
  MCSymbol *getCFIPersonalitySymbol(const GlobalValue *GV, Mangler &Mang,
                                    const TargetMachine &TM,
                                    MachineModuleInfo *MMI) const override;

  /// Get MachO PC relative GOT entry relocation
  const MCExpr *getIndirectSymViaGOTPCRel(const MCSymbol *Sym,
                                          const MCValue &MV, int64_t Offset,
                                          MachineModuleInfo *MMI,
                                          MCStreamer &Streamer) const override;
};



class TargetLoweringObjectFileCOFF : public TargetLoweringObjectFile {
public:
  ~TargetLoweringObjectFileCOFF() override {}

  MCSection *getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind,
                                      Mangler &Mang,
                                      const TargetMachine &TM) const override;

  MCSection *SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind,
                                    Mangler &Mang,
                                    const TargetMachine &TM) const override;

  void getNameWithPrefix(SmallVectorImpl<char> &OutName, const GlobalValue *GV,
                         bool CannotUsePrivateLabel, Mangler &Mang,
                         const TargetMachine &TM) const override;

  MCSection *getSectionForJumpTable(const Function &F, Mangler &Mang,
                                    const TargetMachine &TM) const override;

  /// Emit Obj-C garbage collection and linker options. Only linker option
  /// emission is implemented for COFF.
  void emitModuleFlags(MCStreamer &Streamer,
                       ArrayRef<Module::ModuleFlagEntry> ModuleFlags,
                       Mangler &Mang, const TargetMachine &TM) const override;

  MCSection *getStaticCtorSection(unsigned Priority,
                                  const MCSymbol *KeySym) const override;
  MCSection *getStaticDtorSection(unsigned Priority,
                                  const MCSymbol *KeySym) const override;

  void emitLinkerFlagsForGlobal(raw_ostream &OS, const GlobalValue *GV,
                                const Mangler &Mang) const override;
};

} // end namespace llvm

#endif
