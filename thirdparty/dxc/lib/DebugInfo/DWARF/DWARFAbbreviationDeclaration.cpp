//===-- DWARFAbbreviationDeclaration.cpp ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace dwarf;

void DWARFAbbreviationDeclaration::clear() {
  Code = 0;
  Tag = 0;
  HasChildren = false;
  AttributeSpecs.clear();
}

DWARFAbbreviationDeclaration::DWARFAbbreviationDeclaration() {
  clear();
}

bool
DWARFAbbreviationDeclaration::extract(DataExtractor Data, uint32_t* OffsetPtr) {
  clear();
  Code = Data.getULEB128(OffsetPtr);
  if (Code == 0) {
    return false;
  }
  Tag = Data.getULEB128(OffsetPtr);
  uint8_t ChildrenByte = Data.getU8(OffsetPtr);
  HasChildren = (ChildrenByte == DW_CHILDREN_yes);

  while (true) {
    uint32_t CurOffset = *OffsetPtr;
    uint16_t Attr = Data.getULEB128(OffsetPtr);
    if (CurOffset == *OffsetPtr) {
      clear();
      return false;
    }
    CurOffset = *OffsetPtr;
    uint16_t Form = Data.getULEB128(OffsetPtr);
    if (CurOffset == *OffsetPtr) {
      clear();
      return false;
    }
    if (Attr == 0 && Form == 0)
      break;
    AttributeSpecs.push_back(AttributeSpec(Attr, Form));
  }

  if (Tag == 0) {
    clear();
    return false;
  }
  return true;
}

void DWARFAbbreviationDeclaration::dump(raw_ostream &OS) const {
  const char *tagString = TagString(getTag());
  OS << '[' << getCode() << "] ";
  if (tagString)
    OS << tagString;
  else
    OS << format("DW_TAG_Unknown_%x", getTag());
  OS << "\tDW_CHILDREN_" << (hasChildren() ? "yes" : "no") << '\n';
  for (const AttributeSpec &Spec : AttributeSpecs) {
    OS << '\t';
    const char *attrString = AttributeString(Spec.Attr);
    if (attrString)
      OS << attrString;
    else
      OS << format("DW_AT_Unknown_%x", Spec.Attr);
    OS << '\t';
    const char *formString = FormEncodingString(Spec.Form);
    if (formString)
      OS << formString;
    else
      OS << format("DW_FORM_Unknown_%x", Spec.Form);
    OS << '\n';
  }
  OS << '\n';
}

uint32_t
DWARFAbbreviationDeclaration::findAttributeIndex(uint16_t attr) const {
  for (uint32_t i = 0, e = AttributeSpecs.size(); i != e; ++i) {
    if (AttributeSpecs[i].Attr == attr)
      return i;
  }
  return -1U;
}
