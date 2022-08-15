//===-- DWARFDebugInfoEntry.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_DWARFDEBUGINFOENTRY_H
#define LLVM_LIB_DEBUGINFO_DWARFDEBUGINFOENTRY_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugRangeList.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class DWARFDebugAranges;
class DWARFCompileUnit;
class DWARFUnit;
class DWARFContext;
class DWARFFormValue;
struct DWARFDebugInfoEntryInlinedChain;

/// DWARFDebugInfoEntryMinimal - A DIE with only the minimum required data.
class DWARFDebugInfoEntryMinimal {
  /// Offset within the .debug_info of the start of this entry.
  uint32_t Offset;

  /// How many to add to "this" to get the sibling.
  uint32_t SiblingIdx;

  const DWARFAbbreviationDeclaration *AbbrevDecl;
public:
  DWARFDebugInfoEntryMinimal()
    : Offset(0), SiblingIdx(0), AbbrevDecl(nullptr) {}

  void dump(raw_ostream &OS, DWARFUnit *u, unsigned recurseDepth,
            unsigned indent = 0) const;
  void dumpAttribute(raw_ostream &OS, DWARFUnit *u, uint32_t *offset_ptr,
                     uint16_t attr, uint16_t form, unsigned indent = 0) const;

  /// Extracts a debug info entry, which is a child of a given unit,
  /// starting at a given offset. If DIE can't be extracted, returns false and
  /// doesn't change OffsetPtr.
  bool extractFast(const DWARFUnit *U, uint32_t *OffsetPtr);

  uint32_t getTag() const { return AbbrevDecl ? AbbrevDecl->getTag() : 0; }
  bool isNULL() const { return AbbrevDecl == nullptr; }

  /// Returns true if DIE represents a subprogram (not inlined).
  bool isSubprogramDIE() const;
  /// Returns true if DIE represents a subprogram or an inlined
  /// subroutine.
  bool isSubroutineDIE() const;

  uint32_t getOffset() const { return Offset; }
  bool hasChildren() const { return !isNULL() && AbbrevDecl->hasChildren(); }

  // We know we are kept in a vector of contiguous entries, so we know
  // our sibling will be some index after "this".
  const DWARFDebugInfoEntryMinimal *getSibling() const {
    return SiblingIdx > 0 ? this + SiblingIdx : nullptr;
  }

  // We know we are kept in a vector of contiguous entries, so we know
  // we don't need to store our child pointer, if we have a child it will
  // be the next entry in the list...
  const DWARFDebugInfoEntryMinimal *getFirstChild() const {
    return hasChildren() ? this + 1 : nullptr;
  }

  void setSibling(const DWARFDebugInfoEntryMinimal *Sibling) {
    if (Sibling) {
      // We know we are kept in a vector of contiguous entries, so we know
      // our sibling will be some index after "this".
      SiblingIdx = Sibling - this;
    } else
      SiblingIdx = 0;
  }

  const DWARFAbbreviationDeclaration *getAbbreviationDeclarationPtr() const {
    return AbbrevDecl;
  }

  bool getAttributeValue(const DWARFUnit *U, const uint16_t Attr,
                         DWARFFormValue &FormValue) const;

  const char *getAttributeValueAsString(const DWARFUnit *U, const uint16_t Attr,
                                        const char *FailValue) const;

  uint64_t getAttributeValueAsAddress(const DWARFUnit *U, const uint16_t Attr,
                                      uint64_t FailValue) const;

  uint64_t getAttributeValueAsUnsignedConstant(const DWARFUnit *U,
                                               const uint16_t Attr,
                                               uint64_t FailValue) const;

  uint64_t getAttributeValueAsReference(const DWARFUnit *U, const uint16_t Attr,
                                        uint64_t FailValue) const;

  uint64_t getAttributeValueAsSectionOffset(const DWARFUnit *U,
                                            const uint16_t Attr,
                                            uint64_t FailValue) const;

  uint64_t getRangesBaseAttribute(const DWARFUnit *U, uint64_t FailValue) const;

  /// Retrieves DW_AT_low_pc and DW_AT_high_pc from CU.
  /// Returns true if both attributes are present.
  bool getLowAndHighPC(const DWARFUnit *U, uint64_t &LowPC,
                       uint64_t &HighPC) const;

  DWARFAddressRangesVector getAddressRanges(const DWARFUnit *U) const;

  void collectChildrenAddressRanges(const DWARFUnit *U,
                                    DWARFAddressRangesVector &Ranges) const;

  bool addressRangeContainsAddress(const DWARFUnit *U,
                                   const uint64_t Address) const;

  /// If a DIE represents a subprogram (or inlined subroutine),
  /// returns its mangled name (or short name, if mangled is missing).
  /// This name may be fetched from specification or abstract origin
  /// for this subprogram. Returns null if no name is found.
  const char *getSubroutineName(const DWARFUnit *U, DINameKind Kind) const;

  /// Return the DIE name resolving DW_AT_sepcification or
  /// DW_AT_abstract_origin references if necessary.
  /// Returns null if no name is found.
  const char *getName(const DWARFUnit *U, DINameKind Kind) const;

  /// Retrieves values of DW_AT_call_file, DW_AT_call_line and
  /// DW_AT_call_column from DIE (or zeroes if they are missing).
  void getCallerFrame(const DWARFUnit *U, uint32_t &CallFile,
                      uint32_t &CallLine, uint32_t &CallColumn) const;

  /// Get inlined chain for a given address, rooted at the current DIE.
  /// Returns empty chain if address is not contained in address range
  /// of current DIE.
  DWARFDebugInfoEntryInlinedChain
  getInlinedChainForAddress(const DWARFUnit *U, const uint64_t Address) const;
};

/// DWARFDebugInfoEntryInlinedChain - represents a chain of inlined_subroutine
/// DIEs, (possibly ending with subprogram DIE), all of which are contained
/// in some concrete inlined instance tree. Address range for each DIE
/// (except the last DIE) in this chain is contained in address
/// range for next DIE in the chain.
struct DWARFDebugInfoEntryInlinedChain {
  DWARFDebugInfoEntryInlinedChain() : U(nullptr) {}
  SmallVector<DWARFDebugInfoEntryMinimal, 4> DIEs;
  const DWARFUnit *U;
};

}

#endif
