//===- MCSectionELF.h - ELF Machine Code Sections ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCSectionELF class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSECTIONELF_H
#define LLVM_MC_MCSECTIONELF_H

#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class MCSymbol;

/// MCSectionELF - This represents a section on linux, lots of unix variants
/// and some bare metal systems.
class MCSectionELF : public MCSection {
  /// SectionName - This is the name of the section.  The referenced memory is
  /// owned by TargetLoweringObjectFileELF's ELFUniqueMap.
  StringRef SectionName;

  /// Type - This is the sh_type field of a section, drawn from the enums below.
  unsigned Type;

  /// Flags - This is the sh_flags field of a section, drawn from the enums.
  /// below.
  unsigned Flags;

  unsigned UniqueID;

  /// EntrySize - The size of each entry in this section. This size only
  /// makes sense for sections that contain fixed-sized entries. If a
  /// section does not contain fixed-sized entries 'EntrySize' will be 0.
  unsigned EntrySize;

  const MCSymbolELF *Group;

  /// Depending on the type of the section this is sh_link or sh_info.
  const MCSectionELF *Associated;

private:
  friend class MCContext;
  MCSectionELF(StringRef Section, unsigned type, unsigned flags, SectionKind K,
               unsigned entrySize, const MCSymbolELF *group, unsigned UniqueID,
               MCSymbol *Begin, const MCSectionELF *Associated)
      : MCSection(SV_ELF, K, Begin), SectionName(Section), Type(type),
        Flags(flags), UniqueID(UniqueID), EntrySize(entrySize), Group(group),
        Associated(Associated) {
    if (Group)
      Group->setIsSignature();
  }
  ~MCSectionELF() override;

  void setSectionName(StringRef Name) { SectionName = Name; }

public:

  /// ShouldOmitSectionDirective - Decides whether a '.section' directive
  /// should be printed before the section name
  bool ShouldOmitSectionDirective(StringRef Name, const MCAsmInfo &MAI) const;

  StringRef getSectionName() const { return SectionName; }
  unsigned getType() const { return Type; }
  unsigned getFlags() const { return Flags; }
  unsigned getEntrySize() const { return EntrySize; }
  const MCSymbolELF *getGroup() const { return Group; }

  void PrintSwitchToSection(const MCAsmInfo &MAI, raw_ostream &OS,
                            const MCExpr *Subsection) const override;
  bool UseCodeAlign() const override;
  bool isVirtualSection() const override;

  bool isUnique() const { return UniqueID != ~0U; }
  unsigned getUniqueID() const { return UniqueID; }

  const MCSectionELF *getAssociatedSection() const { return Associated; }

  static bool classof(const MCSection *S) {
    return S->getVariant() == SV_ELF;
  }
};

} // end namespace llvm

#endif
