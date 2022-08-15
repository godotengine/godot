//===- lib/MC/MCSectionELF.cpp - ELF Code Section Representation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

MCSectionELF::~MCSectionELF() {} // anchor.

// Decides whether a '.section' directive
// should be printed before the section name.
bool MCSectionELF::ShouldOmitSectionDirective(StringRef Name,
                                              const MCAsmInfo &MAI) const {

  if (isUnique())
    return false;

  // FIXME: Does .section .bss/.data/.text work everywhere??
  if (Name == ".text" || Name == ".data" ||
      (Name == ".bss" && !MAI.usesELFSectionDirectiveForBSS()))
    return true;

  return false;
}

static void printName(raw_ostream &OS, StringRef Name) {
  if (Name.find_first_not_of("0123456789_."
                             "abcdefghijklmnopqrstuvwxyz"
                             "ABCDEFGHIJKLMNOPQRSTUVWXYZ") == Name.npos) {
    OS << Name;
    return;
  }
  OS << '"';
  for (const char *B = Name.begin(), *E = Name.end(); B < E; ++B) {
    if (*B == '"') // Unquoted "
      OS << "\\\"";
    else if (*B != '\\') // Neither " or backslash
      OS << *B;
    else if (B + 1 == E) // Trailing backslash
      OS << "\\\\";
    else {
      OS << B[0] << B[1]; // Quoted character
      ++B;
    }
  }
  OS << '"';
}

void MCSectionELF::PrintSwitchToSection(const MCAsmInfo &MAI,
                                        raw_ostream &OS,
                                        const MCExpr *Subsection) const {

  if (ShouldOmitSectionDirective(SectionName, MAI)) {
    OS << '\t' << getSectionName();
    if (Subsection) {
      OS << '\t';
      Subsection->print(OS, &MAI);
    }
    OS << '\n';
    return;
  }

  OS << "\t.section\t";
  printName(OS, getSectionName());

  // Handle the weird solaris syntax if desired.
  if (MAI.usesSunStyleELFSectionSwitchSyntax() &&
      !(Flags & ELF::SHF_MERGE)) {
    if (Flags & ELF::SHF_ALLOC)
      OS << ",#alloc";
    if (Flags & ELF::SHF_EXECINSTR)
      OS << ",#execinstr";
    if (Flags & ELF::SHF_WRITE)
      OS << ",#write";
    if (Flags & ELF::SHF_EXCLUDE)
      OS << ",#exclude";
    if (Flags & ELF::SHF_TLS)
      OS << ",#tls";
    OS << '\n';
    return;
  }

  OS << ",\"";
  if (Flags & ELF::SHF_ALLOC)
    OS << 'a';
  if (Flags & ELF::SHF_EXCLUDE)
    OS << 'e';
  if (Flags & ELF::SHF_EXECINSTR)
    OS << 'x';
  if (Flags & ELF::SHF_GROUP)
    OS << 'G';
  if (Flags & ELF::SHF_WRITE)
    OS << 'w';
  if (Flags & ELF::SHF_MERGE)
    OS << 'M';
  if (Flags & ELF::SHF_STRINGS)
    OS << 'S';
  if (Flags & ELF::SHF_TLS)
    OS << 'T';

  // If there are target-specific flags, print them.
  if (Flags & ELF::XCORE_SHF_CP_SECTION)
    OS << 'c';
  if (Flags & ELF::XCORE_SHF_DP_SECTION)
    OS << 'd';

  OS << '"';

  OS << ',';

  // If comment string is '@', e.g. as on ARM - use '%' instead
  if (MAI.getCommentString()[0] == '@')
    OS << '%';
  else
    OS << '@';

  if (Type == ELF::SHT_INIT_ARRAY)
    OS << "init_array";
  else if (Type == ELF::SHT_FINI_ARRAY)
    OS << "fini_array";
  else if (Type == ELF::SHT_PREINIT_ARRAY)
    OS << "preinit_array";
  else if (Type == ELF::SHT_NOBITS)
    OS << "nobits";
  else if (Type == ELF::SHT_NOTE)
    OS << "note";
  else if (Type == ELF::SHT_PROGBITS)
    OS << "progbits";

  if (EntrySize) {
    assert(Flags & ELF::SHF_MERGE);
    OS << "," << EntrySize;
  }

  if (Flags & ELF::SHF_GROUP) {
    OS << ",";
    printName(OS, Group->getName());
    OS << ",comdat";
  }

  if (isUnique())
    OS << ",unique," << UniqueID;

  OS << '\n';

  if (Subsection) {
    OS << "\t.subsection\t";
    Subsection->print(OS, &MAI);
    OS << '\n';
  }
}

bool MCSectionELF::UseCodeAlign() const {
  return getFlags() & ELF::SHF_EXECINSTR;
}

bool MCSectionELF::isVirtualSection() const {
  return getType() == ELF::SHT_NOBITS;
}
