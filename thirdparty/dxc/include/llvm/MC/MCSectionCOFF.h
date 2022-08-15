//===- MCSectionCOFF.h - COFF Machine Code Sections -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCSectionCOFF class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSECTIONCOFF_H
#define LLVM_MC_MCSECTIONCOFF_H

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCSection.h"
#include "llvm/Support/COFF.h"

namespace llvm {
class MCSymbol;

/// MCSectionCOFF - This represents a section on Windows
  class MCSectionCOFF : public MCSection {
    // The memory for this string is stored in the same MCContext as *this.
    StringRef SectionName;

    // FIXME: The following fields should not be mutable, but are for now so
    // the asm parser can honor the .linkonce directive.

    /// Characteristics - This is the Characteristics field of a section,
    /// drawn from the enums below.
    mutable unsigned Characteristics;

    /// The COMDAT symbol of this section. Only valid if this is a COMDAT
    /// section. Two COMDAT sections are merged if they have the same
    /// COMDAT symbol.
    MCSymbol *COMDATSymbol;

    /// Selection - This is the Selection field for the section symbol, if
    /// it is a COMDAT section (Characteristics & IMAGE_SCN_LNK_COMDAT) != 0
    mutable int Selection;

  private:
    friend class MCContext;
    MCSectionCOFF(StringRef Section, unsigned Characteristics,
                  MCSymbol *COMDATSymbol, int Selection, SectionKind K,
                  MCSymbol *Begin)
        : MCSection(SV_COFF, K, Begin), SectionName(Section),
          Characteristics(Characteristics), COMDATSymbol(COMDATSymbol),
          Selection(Selection) {
      assert ((Characteristics & 0x00F00000) == 0 &&
        "alignment must not be set upon section creation");
    }
    ~MCSectionCOFF() override;

  public:
    /// ShouldOmitSectionDirective - Decides whether a '.section' directive
    /// should be printed before the section name
    bool ShouldOmitSectionDirective(StringRef Name, const MCAsmInfo &MAI) const;

    StringRef getSectionName() const { return SectionName; }
    unsigned getCharacteristics() const { return Characteristics; }
    MCSymbol *getCOMDATSymbol() const { return COMDATSymbol; }
    int getSelection() const { return Selection; }

    void setSelection(int Selection) const;

    void PrintSwitchToSection(const MCAsmInfo &MAI, raw_ostream &OS,
                              const MCExpr *Subsection) const override;
    bool UseCodeAlign() const override;
    bool isVirtualSection() const override;

    static bool classof(const MCSection *S) {
      return S->getVariant() == SV_COFF;
    }
  };

} // end namespace llvm

#endif
