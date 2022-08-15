//===-- llvm/CodeGen/DwarfCompileUnit.h - Dwarf Compile Unit ---*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf compile unit.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DWARFCOMPILEUNIT_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DWARFCOMPILEUNIT_H

#include "DwarfUnit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/Support/Dwarf.h"

namespace llvm {

class AsmPrinter;
class DIE;
class DwarfDebug;
class DwarfFile;
class MCSymbol;
class LexicalScope;

class DwarfCompileUnit : public DwarfUnit {
  /// The attribute index of DW_AT_stmt_list in the compile unit DIE, avoiding
  /// the need to search for it in applyStmtList.
  DIE::value_iterator StmtListValue;

  /// Skeleton unit associated with this unit.
  DwarfCompileUnit *Skeleton;

  /// The start of the unit within its section.
  MCSymbol *LabelBegin;

  /// GlobalNames - A map of globally visible named entities for this unit.
  StringMap<const DIE *> GlobalNames;

  /// GlobalTypes - A map of globally visible types for this unit.
  StringMap<const DIE *> GlobalTypes;

  // List of range lists for a given compile unit, separate from the ranges for
  // the CU itself.
  SmallVector<RangeSpanList, 1> CURangeLists;

  // List of ranges for a given compile unit.
  SmallVector<RangeSpan, 2> CURanges;

  // The base address of this unit, if any. Used for relative references in
  // ranges/locs.
  const MCSymbol *BaseAddress;

  /// \brief Construct a DIE for the given DbgVariable without initializing the
  /// DbgVariable's DIE reference.
  DIE *constructVariableDIEImpl(const DbgVariable &DV, bool Abstract);

  bool isDwoUnit() const override;

  bool includeMinimalInlineScopes() const;

public:
  DwarfCompileUnit(unsigned UID, const DICompileUnit *Node, AsmPrinter *A,
                   DwarfDebug *DW, DwarfFile *DWU);

  DwarfCompileUnit *getSkeleton() const {
    return Skeleton;
  }

  void initStmtList();

  /// Apply the DW_AT_stmt_list from this compile unit to the specified DIE.
  void applyStmtList(DIE &D);

  /// getOrCreateGlobalVariableDIE - get or create global variable DIE.
  DIE *getOrCreateGlobalVariableDIE(const DIGlobalVariable *GV);

  /// addLabelAddress - Add a dwarf label attribute data and value using
  /// either DW_FORM_addr or DW_FORM_GNU_addr_index.
  void addLabelAddress(DIE &Die, dwarf::Attribute Attribute,
                       const MCSymbol *Label);

  /// addLocalLabelAddress - Add a dwarf label attribute data and value using
  /// DW_FORM_addr only.
  void addLocalLabelAddress(DIE &Die, dwarf::Attribute Attribute,
                            const MCSymbol *Label);

  /// addSectionDelta - Add a label delta attribute data and value.
  DIE::value_iterator addSectionDelta(DIE &Die, dwarf::Attribute Attribute,
                                      const MCSymbol *Hi, const MCSymbol *Lo);

  DwarfCompileUnit &getCU() override { return *this; }

  unsigned getOrCreateSourceID(StringRef FileName, StringRef DirName) override;

  /// addRange - Add an address range to the list of ranges for this unit.
  void addRange(RangeSpan Range);

  void attachLowHighPC(DIE &D, const MCSymbol *Begin, const MCSymbol *End);

  /// addSectionLabel - Add a Dwarf section label attribute data and value.
  ///
  DIE::value_iterator addSectionLabel(DIE &Die, dwarf::Attribute Attribute,
                                      const MCSymbol *Label,
                                      const MCSymbol *Sec);

  /// \brief Find DIE for the given subprogram and attach appropriate
  /// DW_AT_low_pc and DW_AT_high_pc attributes. If there are global
  /// variables in this scope then create and insert DIEs for these
  /// variables.
  DIE &updateSubprogramScopeDIE(const DISubprogram *SP);

  void constructScopeDIE(LexicalScope *Scope,
                         SmallVectorImpl<DIE *> &FinalChildren);

  /// \brief A helper function to construct a RangeSpanList for a given
  /// lexical scope.
  void addScopeRangeList(DIE &ScopeDIE, SmallVector<RangeSpan, 2> Range);

  void attachRangesOrLowHighPC(DIE &D, SmallVector<RangeSpan, 2> Ranges);

  void attachRangesOrLowHighPC(DIE &D,
                               const SmallVectorImpl<InsnRange> &Ranges);
  /// \brief This scope represents inlined body of a function. Construct
  /// DIE to represent this concrete inlined copy of the function.
  DIE *constructInlinedScopeDIE(LexicalScope *Scope);

  /// \brief Construct new DW_TAG_lexical_block for this scope and
  /// attach DW_AT_low_pc/DW_AT_high_pc labels.
  DIE *constructLexicalScopeDIE(LexicalScope *Scope);

  /// constructVariableDIE - Construct a DIE for the given DbgVariable.
  DIE *constructVariableDIE(DbgVariable &DV, bool Abstract = false);

  DIE *constructVariableDIE(DbgVariable &DV, const LexicalScope &Scope,
                            DIE *&ObjectPointer);

  /// A helper function to create children of a Scope DIE.
  DIE *createScopeChildrenDIE(LexicalScope *Scope,
                              SmallVectorImpl<DIE *> &Children,
                              unsigned *ChildScopeCount = nullptr);

  /// \brief Construct a DIE for this subprogram scope.
  void constructSubprogramScopeDIE(LexicalScope *Scope);

  DIE *createAndAddScopeChildren(LexicalScope *Scope, DIE &ScopeDIE);

  void constructAbstractSubprogramScopeDIE(LexicalScope *Scope);

  /// \brief Construct import_module DIE.
  DIE *constructImportedEntityDIE(const DIImportedEntity *Module);

  void finishSubprogramDefinition(const DISubprogram *SP);

  void collectDeadVariables(const DISubprogram *SP);

  /// Set the skeleton unit associated with this unit.
  void setSkeleton(DwarfCompileUnit &Skel) { Skeleton = &Skel; }

  const MCSymbol *getSectionSym() const {
    assert(Section);
    return Section->getBeginSymbol();
  }

  unsigned getLength() {
    return sizeof(uint32_t) + // Length field
        getHeaderSize() + UnitDie.getSize();
  }

  void emitHeader(bool UseOffsets) override;

  MCSymbol *getLabelBegin() const {
    assert(Section);
    return LabelBegin;
  }

  /// Add a new global name to the compile unit.
  void addGlobalName(StringRef Name, DIE &Die, const DIScope *Context) override;

  /// Add a new global type to the compile unit.
  void addGlobalType(const DIType *Ty, const DIE &Die,
                     const DIScope *Context) override;

  const StringMap<const DIE *> &getGlobalNames() const { return GlobalNames; }
  const StringMap<const DIE *> &getGlobalTypes() const { return GlobalTypes; }

  /// Add DW_AT_location attribute for a DbgVariable based on provided
  /// MachineLocation.
  void addVariableAddress(const DbgVariable &DV, DIE &Die,
                          MachineLocation Location);
  /// Add an address attribute to a die based on the location provided.
  void addAddress(DIE &Die, dwarf::Attribute Attribute,
                  const MachineLocation &Location);

  /// Start with the address based on the location provided, and generate the
  /// DWARF information necessary to find the actual variable (navigating the
  /// extra location information encoded in the type) based on the starting
  /// location.  Add the DWARF information to the die.
  void addComplexAddress(const DbgVariable &DV, DIE &Die,
                         dwarf::Attribute Attribute,
                         const MachineLocation &Location);

  /// Add a Dwarf loclistptr attribute data and value.
  void addLocationList(DIE &Die, dwarf::Attribute Attribute, unsigned Index);
  void applyVariableAttributes(const DbgVariable &Var, DIE &VariableDie);

  /// Add a Dwarf expression attribute data and value.
  void addExpr(DIELoc &Die, dwarf::Form Form, const MCExpr *Expr);

  void applySubprogramAttributesToDefinition(const DISubprogram *SP,
                                             DIE &SPDie);

  /// getRangeLists - Get the vector of range lists.
  const SmallVectorImpl<RangeSpanList> &getRangeLists() const {
    return (Skeleton ? Skeleton : this)->CURangeLists;
  }

  /// getRanges - Get the list of ranges for this unit.
  const SmallVectorImpl<RangeSpan> &getRanges() const { return CURanges; }
  SmallVector<RangeSpan, 2> takeRanges() { return std::move(CURanges); }

  void setBaseAddress(const MCSymbol *Base) { BaseAddress = Base; }
  const MCSymbol *getBaseAddress() const { return BaseAddress; }
};

} // end llvm namespace

#endif
