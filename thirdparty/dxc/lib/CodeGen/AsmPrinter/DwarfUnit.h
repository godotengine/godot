//===-- llvm/CodeGen/DwarfUnit.h - Dwarf Compile Unit ---*- C++ -*--===//
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

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DWARFUNIT_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DWARFUNIT_H

#include "DwarfDebug.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSection.h"

namespace llvm {

class MachineLocation;
class MachineOperand;
class ConstantInt;
class ConstantFP;
class DbgVariable;
class DwarfCompileUnit;

// Data structure to hold a range for range lists.
class RangeSpan {
public:
  RangeSpan(MCSymbol *S, MCSymbol *E) : Start(S), End(E) {}
  const MCSymbol *getStart() const { return Start; }
  const MCSymbol *getEnd() const { return End; }
  void setEnd(const MCSymbol *E) { End = E; }

private:
  const MCSymbol *Start, *End;
};

class RangeSpanList {
private:
  // Index for locating within the debug_range section this particular span.
  MCSymbol *RangeSym;
  // List of ranges.
  SmallVector<RangeSpan, 2> Ranges;

public:
  RangeSpanList(MCSymbol *Sym, SmallVector<RangeSpan, 2> Ranges)
      : RangeSym(Sym), Ranges(std::move(Ranges)) {}
  MCSymbol *getSym() const { return RangeSym; }
  const SmallVectorImpl<RangeSpan> &getRanges() const { return Ranges; }
  void addRange(RangeSpan Range) { Ranges.push_back(Range); }
};
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
/// This dwarf writer support class manages information associated with a
/// source file.
class DwarfUnit {
protected:
  /// A numeric ID unique among all CUs in the module
  unsigned UniqueID;

  /// MDNode for the compile unit.
  const DICompileUnit *CUNode;

  // All DIEValues are allocated through this allocator.
  BumpPtrAllocator DIEValueAllocator;

  /// Unit debug information entry.
  DIE &UnitDie;

  /// Offset of the UnitDie from beginning of debug info section.
  unsigned DebugInfoOffset;

  /// Target of Dwarf emission.
  AsmPrinter *Asm;

  // Holders for some common dwarf information.
  DwarfDebug *DD;
  DwarfFile *DU;

  /// An anonymous type for index type.  Owned by UnitDie.
  DIE *IndexTyDie;

  /// Tracks the mapping of unit level debug information variables to debug
  /// information entries.
  DenseMap<const MDNode *, DIE *> MDNodeToDieMap;

  /// A list of all the DIEBlocks in use.
  std::vector<DIEBlock *> DIEBlocks;

  /// A list of all the DIELocs in use.
  std::vector<DIELoc *> DIELocs;

  /// This map is used to keep track of subprogram DIEs that need
  /// DW_AT_containing_type attribute. This attribute points to a DIE that
  /// corresponds to the MDNode mapped with the subprogram DIE.
  DenseMap<DIE *, const DINode *> ContainingTypeMap;

  /// The section this unit will be emitted in.
  MCSection *Section;

  DwarfUnit(unsigned UID, dwarf::Tag, const DICompileUnit *CU, AsmPrinter *A,
            DwarfDebug *DW, DwarfFile *DWU);

  /// Add a string attribute data and value.
  ///
  /// This is guaranteed to be in the local string pool instead of indirected.
  void addLocalString(DIE &Die, dwarf::Attribute Attribute, StringRef Str);

  void addIndexedString(DIE &Die, dwarf::Attribute Attribute, StringRef Str);

  bool applySubprogramDefinitionAttributes(const DISubprogram *SP, DIE &SPDie);

public:
  virtual ~DwarfUnit();

  void initSection(MCSection *Section);

  MCSection *getSection() const {
    assert(Section);
    return Section;
  }

  // Accessors.
  AsmPrinter* getAsmPrinter() const { return Asm; }
  unsigned getUniqueID() const { return UniqueID; }
  uint16_t getLanguage() const { return CUNode->getSourceLanguage(); }
  const DICompileUnit *getCUNode() const { return CUNode; }
  DIE &getUnitDie() { return UnitDie; }

  unsigned getDebugInfoOffset() const { return DebugInfoOffset; }
  void setDebugInfoOffset(unsigned DbgInfoOff) { DebugInfoOffset = DbgInfoOff; }

  /// Return true if this compile unit has something to write out.
  bool hasContent() const { return UnitDie.hasChildren(); }

  /// Get string containing language specific context for a global name.
  ///
  /// Walks the metadata parent chain in a language specific manner (using the
  /// compile unit language) and returns it as a string. This is done at the
  /// metadata level because DIEs may not currently have been added to the
  /// parent context and walking the DIEs looking for names is more expensive
  /// than walking the metadata.
  std::string getParentContextString(const DIScope *Context) const;

  /// Add a new global name to the compile unit.
  virtual void addGlobalName(StringRef Name, DIE &Die, const DIScope *Context) {
  }

  /// Add a new global type to the compile unit.
  virtual void addGlobalType(const DIType *Ty, const DIE &Die,
                             const DIScope *Context) {}

  /// Add a new name to the namespace accelerator table.
  void addAccelNamespace(StringRef Name, const DIE &Die);

  /// Returns the DIE map slot for the specified debug variable.
  ///
  /// We delegate the request to DwarfDebug when the MDNode can be part of the
  /// type system, since DIEs for the type system can be shared across CUs and
  /// the mappings are kept in DwarfDebug.
  DIE *getDIE(const DINode *D) const;

  /// Returns a fresh newly allocated DIELoc.
  DIELoc *getDIELoc() { return new (DIEValueAllocator) DIELoc; }

  /// Insert DIE into the map.
  ///
  /// We delegate the request to DwarfDebug when the MDNode can be part of the
  /// type system, since DIEs for the type system can be shared across CUs and
  /// the mappings are kept in DwarfDebug.
  void insertDIE(const DINode *Desc, DIE *D);

  /// Add a flag that is true to the DIE.
  void addFlag(DIE &Die, dwarf::Attribute Attribute);

  /// Add an unsigned integer attribute data and value.
  void addUInt(DIE &Die, dwarf::Attribute Attribute, Optional<dwarf::Form> Form,
               uint64_t Integer);

  void addUInt(DIE &Block, dwarf::Form Form, uint64_t Integer);

  /// Add an signed integer attribute data and value.
  void addSInt(DIE &Die, dwarf::Attribute Attribute, Optional<dwarf::Form> Form,
               int64_t Integer);

  void addSInt(DIELoc &Die, Optional<dwarf::Form> Form, int64_t Integer);

  /// Add a string attribute data and value.
  ///
  /// We always emit a reference to the string pool instead of immediate
  /// strings so that DIEs have more predictable sizes. In the case of split
  /// dwarf we emit an index into another table which gets us the static offset
  /// into the string table.
  void addString(DIE &Die, dwarf::Attribute Attribute, StringRef Str);

  /// Add a Dwarf label attribute data and value.
  DIE::value_iterator addLabel(DIE &Die, dwarf::Attribute Attribute,
                               dwarf::Form Form, const MCSymbol *Label);

  void addLabel(DIELoc &Die, dwarf::Form Form, const MCSymbol *Label);

  /// Add an offset into a section attribute data and value.
  void addSectionOffset(DIE &Die, dwarf::Attribute Attribute, uint64_t Integer);

  /// Add a dwarf op address data and value using the form given and an
  /// op of either DW_FORM_addr or DW_FORM_GNU_addr_index.
  void addOpAddress(DIELoc &Die, const MCSymbol *Label);

  /// Add a label delta attribute data and value.
  void addLabelDelta(DIE &Die, dwarf::Attribute Attribute, const MCSymbol *Hi,
                     const MCSymbol *Lo);

  /// Add a DIE attribute data and value.
  void addDIEEntry(DIE &Die, dwarf::Attribute Attribute, DIE &Entry);

  /// Add a DIE attribute data and value.
  void addDIEEntry(DIE &Die, dwarf::Attribute Attribute, DIEEntry Entry);

  void addDIETypeSignature(DIE &Die, const DwarfTypeUnit &Type);

  /// Add block data.
  void addBlock(DIE &Die, dwarf::Attribute Attribute, DIELoc *Block);

  /// Add block data.
  void addBlock(DIE &Die, dwarf::Attribute Attribute, DIEBlock *Block);

  /// Add location information to specified debug information entry.
  void addSourceLine(DIE &Die, unsigned Line, StringRef File,
                     StringRef Directory);
  void addSourceLine(DIE &Die, const DILocalVariable *V);
  void addSourceLine(DIE &Die, const DIGlobalVariable *G);
  void addSourceLine(DIE &Die, const DISubprogram *SP);
  void addSourceLine(DIE &Die, const DIType *Ty);
  void addSourceLine(DIE &Die, const DINamespace *NS);
  void addSourceLine(DIE &Die, const DIObjCProperty *Ty);

  /// Add constant value entry in variable DIE.
  void addConstantValue(DIE &Die, const MachineOperand &MO, const DIType *Ty);
  void addConstantValue(DIE &Die, const ConstantInt *CI, const DIType *Ty);
  void addConstantValue(DIE &Die, const APInt &Val, const DIType *Ty);
  void addConstantValue(DIE &Die, const APInt &Val, bool Unsigned);
  void addConstantValue(DIE &Die, bool Unsigned, uint64_t Val);

  /// Add constant value entry in variable DIE.
  void addConstantFPValue(DIE &Die, const MachineOperand &MO);
  void addConstantFPValue(DIE &Die, const ConstantFP *CFP);

  /// Add a linkage name, if it isn't empty.
  void addLinkageName(DIE &Die, StringRef LinkageName);

  /// Add template parameters in buffer.
  void addTemplateParams(DIE &Buffer, DINodeArray TParams);

  /// Add register operand.
  /// \returns false if the register does not exist, e.g., because it was never
  /// materialized.
  bool addRegisterOpPiece(DIELoc &TheDie, unsigned Reg,
                          unsigned SizeInBits = 0, unsigned OffsetInBits = 0);

  /// Add register offset.
  /// \returns false if the register does not exist, e.g., because it was never
  /// materialized.
  bool addRegisterOffset(DIELoc &TheDie, unsigned Reg, int64_t Offset);

  // FIXME: Should be reformulated in terms of addComplexAddress.
  /// Start with the address based on the location provided, and generate the
  /// DWARF information necessary to find the actual Block variable (navigating
  /// the Block struct) based on the starting location.  Add the DWARF
  /// information to the die.  Obsolete, please use addComplexAddress instead.
  void addBlockByrefAddress(const DbgVariable &DV, DIE &Die,
                            dwarf::Attribute Attribute,
                            const MachineLocation &Location);

  /// Add a new type attribute to the specified entity.
  ///
  /// This takes and attribute parameter because DW_AT_friend attributes are
  /// also type references.
  void addType(DIE &Entity, const DIType *Ty,
               dwarf::Attribute Attribute = dwarf::DW_AT_type);

  DIE *getOrCreateNameSpace(const DINamespace *NS);
  DIE *getOrCreateModule(const DIModule *M);
  DIE *getOrCreateSubprogramDIE(const DISubprogram *SP, bool Minimal = false);

  void applySubprogramAttributes(const DISubprogram *SP, DIE &SPDie,
                                 bool Minimal = false);

  /// Find existing DIE or create new DIE for the given type.
  DIE *getOrCreateTypeDIE(const MDNode *N);

  /// Get context owner's DIE.
  DIE *createTypeDIE(const DICompositeType *Ty);

  /// Get context owner's DIE.
  DIE *getOrCreateContextDIE(const DIScope *Context);

  /// Construct DIEs for types that contain vtables.
  void constructContainingTypeDIEs();

  /// Construct function argument DIEs.
  void constructSubprogramArguments(DIE &Buffer, DITypeRefArray Args);

  /// Create a DIE with the given Tag, add the DIE to its parent, and
  /// call insertDIE if MD is not null.
  DIE &createAndAddDIE(unsigned Tag, DIE &Parent, const DINode *N = nullptr);

  /// Compute the size of a header for this unit, not including the initial
  /// length field.
  virtual unsigned getHeaderSize() const {
    return sizeof(int16_t) + // DWARF version number
           sizeof(int32_t) + // Offset Into Abbrev. Section
           sizeof(int8_t);   // Pointer Size (in bytes)
  }

  /// Emit the header for this unit, not including the initial length field.
  virtual void emitHeader(bool UseOffsets);

  virtual DwarfCompileUnit &getCU() = 0;

  void constructTypeDIE(DIE &Buffer, const DICompositeType *CTy);

protected:
  /// Create new static data member DIE.
  DIE *getOrCreateStaticMemberDIE(const DIDerivedType *DT);

  /// Look up the source ID with the given directory and source file names. If
  /// none currently exists, create a new ID and insert it in the line table.
  virtual unsigned getOrCreateSourceID(StringRef File, StringRef Directory) = 0;

  /// Look in the DwarfDebug map for the MDNode that corresponds to the
  /// reference.
  template <typename T> T *resolve(TypedDINodeRef<T> Ref) const {
    return DD->resolve(Ref);
  }

private:
  void constructTypeDIE(DIE &Buffer, const DIBasicType *BTy);
  void constructTypeDIE(DIE &Buffer, const DIDerivedType *DTy);
  void constructTypeDIE(DIE &Buffer, const DISubroutineType *DTy);
  void constructSubrangeDIE(DIE &Buffer, const DISubrange *SR, DIE *IndexTy);
  void constructArrayTypeDIE(DIE &Buffer, const DICompositeType *CTy);
  void constructEnumTypeDIE(DIE &Buffer, const DICompositeType *CTy);
  void constructMemberDIE(DIE &Buffer, const DIDerivedType *DT);
  void constructTemplateTypeParameterDIE(DIE &Buffer,
                                         const DITemplateTypeParameter *TP);
  void constructTemplateValueParameterDIE(DIE &Buffer,
                                          const DITemplateValueParameter *TVP);

  /// Return the default lower bound for an array.
  ///
  /// If the DWARF version doesn't handle the language, return -1.
  int64_t getDefaultLowerBound() const;

  /// Get an anonymous type for index type.
  DIE *getIndexTyDie();

  /// Set D as anonymous type for index which can be reused later.
  void setIndexTyDie(DIE *D) { IndexTyDie = D; }

  /// If this is a named finished type then include it in the list of types for
  /// the accelerator tables.
  void updateAcceleratorTables(const DIScope *Context, const DIType *Ty,
                               const DIE &TyDIE);

  virtual bool isDwoUnit() const = 0;
};

class DwarfTypeUnit : public DwarfUnit {
  uint64_t TypeSignature;
  const DIE *Ty;
  DwarfCompileUnit &CU;
  MCDwarfDwoLineTable *SplitLineTable;

  unsigned getOrCreateSourceID(StringRef File, StringRef Directory) override;
  bool isDwoUnit() const override;

public:
  DwarfTypeUnit(unsigned UID, DwarfCompileUnit &CU, AsmPrinter *A,
                DwarfDebug *DW, DwarfFile *DWU,
                MCDwarfDwoLineTable *SplitLineTable = nullptr);

  void setTypeSignature(uint64_t Signature) { TypeSignature = Signature; }
  uint64_t getTypeSignature() const { return TypeSignature; }
  void setType(const DIE *Ty) { this->Ty = Ty; }

  /// Emit the header for this unit, not including the initial length field.
  void emitHeader(bool UseOffsets) override;
  unsigned getHeaderSize() const override {
    return DwarfUnit::getHeaderSize() + sizeof(uint64_t) + // Type Signature
           sizeof(uint32_t);                               // Type DIE Offset
  }
  DwarfCompileUnit &getCU() override { return CU; }
};
} // end llvm namespace
#endif
