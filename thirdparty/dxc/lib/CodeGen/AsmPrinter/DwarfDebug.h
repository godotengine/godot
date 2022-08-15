//===-- llvm/CodeGen/DwarfDebug.h - Dwarf Debug Framework ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf debug info into asm files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DWARFDEBUG_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DWARFDEBUG_H

#include "AsmPrinterHandler.h"
#include "DbgValueHistoryCalculator.h"
#include "DebugLocStream.h"
#include "DwarfAccelTable.h"
#include "DwarfFile.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/CodeGen/LexicalScopes.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Support/Allocator.h"
#include <memory>

namespace llvm {

class AsmPrinter;
class ByteStreamer;
class ConstantInt;
class ConstantFP;
class DebugLocEntry;
class DwarfCompileUnit;
class DwarfDebug;
class DwarfTypeUnit;
class DwarfUnit;
class MachineModuleInfo;

//===----------------------------------------------------------------------===//
/// This class is used to record source line correspondence.
class SrcLineInfo {
  unsigned Line;     // Source line number.
  unsigned Column;   // Source column.
  unsigned SourceID; // Source ID number.
  MCSymbol *Label;   // Label in code ID number.
public:
  SrcLineInfo(unsigned L, unsigned C, unsigned S, MCSymbol *label)
      : Line(L), Column(C), SourceID(S), Label(label) {}

  // Accessors
  unsigned getLine() const { return Line; }
  unsigned getColumn() const { return Column; }
  unsigned getSourceID() const { return SourceID; }
  MCSymbol *getLabel() const { return Label; }
};
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
/// This class is used to track local variable information.
///
/// Variables can be created from allocas, in which case they're generated from
/// the MMI table.  Such variables can have multiple expressions and frame
/// indices.  The \a Expr and \a FrameIndices array must match.
///
/// Variables can be created from \c DBG_VALUE instructions.  Those whose
/// location changes over time use \a DebugLocListIndex, while those with a
/// single instruction use \a MInsn and (optionally) a single entry of \a Expr.
///
/// Variables that have been optimized out use none of these fields.
class DbgVariable {
  const DILocalVariable *Var;                /// Variable Descriptor.
  const DILocation *IA;                      /// Inlined at location.
  SmallVector<const DIExpression *, 1> Expr; /// Complex address.
  DIE *TheDIE = nullptr;                     /// Variable DIE.
  unsigned DebugLocListIndex = ~0u;          /// Offset in DebugLocs.
  const MachineInstr *MInsn = nullptr;       /// DBG_VALUE instruction.
  SmallVector<int, 1> FrameIndex;            /// Frame index.
  DwarfDebug *DD;

public:
  /// Construct a DbgVariable.
  ///
  /// Creates a variable without any DW_AT_location.  Call \a initializeMMI()
  /// for MMI entries, or \a initializeDbgValue() for DBG_VALUE instructions.
  DbgVariable(const DILocalVariable *V, const DILocation *IA, DwarfDebug *DD)
      : Var(V), IA(IA), DD(DD) {}

  /// Initialize from the MMI table.
  void initializeMMI(const DIExpression *E, int FI) {
    assert(Expr.empty() && "Already initialized?");
    assert(FrameIndex.empty() && "Already initialized?");
    assert(!MInsn && "Already initialized?");

    assert((!E || E->isValid()) && "Expected valid expression");
    assert(~FI && "Expected valid index");

    Expr.push_back(E);
    FrameIndex.push_back(FI);
  }

  /// Initialize from a DBG_VALUE instruction.
  void initializeDbgValue(const MachineInstr *DbgValue) {
    assert(Expr.empty() && "Already initialized?");
    assert(FrameIndex.empty() && "Already initialized?");
    assert(!MInsn && "Already initialized?");

    assert(Var == DbgValue->getDebugVariable() && "Wrong variable");
    assert(IA == DbgValue->getDebugLoc()->getInlinedAt() && "Wrong inlined-at");

    MInsn = DbgValue;
    if (auto *E = DbgValue->getDebugExpression())
      if (E->getNumElements())
        Expr.push_back(E);
  }

  // Accessors.
  const DILocalVariable *getVariable() const { return Var; }
  const DILocation *getInlinedAt() const { return IA; }
  const ArrayRef<const DIExpression *> getExpression() const { return Expr; }
  void setDIE(DIE &D) { TheDIE = &D; }
  DIE *getDIE() const { return TheDIE; }
  void setDebugLocListIndex(unsigned O) { DebugLocListIndex = O; }
  unsigned getDebugLocListIndex() const { return DebugLocListIndex; }
  StringRef getName() const { return Var->getName(); }
  const MachineInstr *getMInsn() const { return MInsn; }
  const ArrayRef<int> getFrameIndex() const { return FrameIndex; }

  void addMMIEntry(const DbgVariable &V) {
    assert(DebugLocListIndex == ~0U && !MInsn && "not an MMI entry");
    assert(V.DebugLocListIndex == ~0U && !V.MInsn && "not an MMI entry");
    assert(V.Var == Var && "conflicting variable");
    assert(V.IA == IA && "conflicting inlined-at location");

    assert(!FrameIndex.empty() && "Expected an MMI entry");
    assert(!V.FrameIndex.empty() && "Expected an MMI entry");
    assert(Expr.size() == FrameIndex.size() && "Mismatched expressions");
    assert(V.Expr.size() == V.FrameIndex.size() && "Mismatched expressions");

    Expr.append(V.Expr.begin(), V.Expr.end());
    FrameIndex.append(V.FrameIndex.begin(), V.FrameIndex.end());
    assert(std::all_of(Expr.begin(), Expr.end(), [](const DIExpression *E) {
             return E && E->isBitPiece();
           }) && "conflicting locations for variable");
  }

  // Translate tag to proper Dwarf tag.
  dwarf::Tag getTag() const {
    if (Var->getTag() == dwarf::DW_TAG_arg_variable)
      return dwarf::DW_TAG_formal_parameter;

    return dwarf::DW_TAG_variable;
  }
  /// Return true if DbgVariable is artificial.
  bool isArtificial() const {
    if (Var->isArtificial())
      return true;
    if (getType()->isArtificial())
      return true;
    return false;
  }

  bool isObjectPointer() const {
    if (Var->isObjectPointer())
      return true;
    if (getType()->isObjectPointer())
      return true;
    return false;
  }

  bool hasComplexAddress() const {
    assert(MInsn && "Expected DBG_VALUE, not MMI variable");
    assert(FrameIndex.empty() && "Expected DBG_VALUE, not MMI variable");
    assert(
        (Expr.empty() || (Expr.size() == 1 && Expr.back()->getNumElements())) &&
        "Invalid Expr for DBG_VALUE");
    return !Expr.empty();
  }
  bool isBlockByrefVariable() const;
  const DIType *getType() const;

private:
  /// Look in the DwarfDebug map for the MDNode that
  /// corresponds to the reference.
  template <typename T> T *resolve(TypedDINodeRef<T> Ref) const;
};


/// Helper used to pair up a symbol and its DWARF compile unit.
struct SymbolCU {
  SymbolCU(DwarfCompileUnit *CU, const MCSymbol *Sym) : Sym(Sym), CU(CU) {}
  const MCSymbol *Sym;
  DwarfCompileUnit *CU;
};

/// Collects and handles dwarf debug information.
class DwarfDebug : public AsmPrinterHandler {
  /// Target of Dwarf emission.
  AsmPrinter *Asm;

  /// Collected machine module information.
  MachineModuleInfo *MMI;

  /// All DIEValues are allocated through this allocator.
  BumpPtrAllocator DIEValueAllocator;

  /// Maps MDNode with its corresponding DwarfCompileUnit.
  MapVector<const MDNode *, DwarfCompileUnit *> CUMap;

  /// Maps subprogram MDNode with its corresponding DwarfCompileUnit.
  MapVector<const MDNode *, DwarfCompileUnit *> SPMap;

  /// Maps a CU DIE with its corresponding DwarfCompileUnit.
  DenseMap<const DIE *, DwarfCompileUnit *> CUDieMap;

  /// List of all labels used in aranges generation.
  std::vector<SymbolCU> ArangeLabels;

  /// Size of each symbol emitted (for those symbols that have a specific size).
  DenseMap<const MCSymbol *, uint64_t> SymSize;

  LexicalScopes LScopes;

  /// Collection of abstract variables.
  DenseMap<const MDNode *, std::unique_ptr<DbgVariable>> AbstractVariables;
  SmallVector<std::unique_ptr<DbgVariable>, 64> ConcreteVariables;

  /// Collection of DebugLocEntry. Stored in a linked list so that DIELocLists
  /// can refer to them in spite of insertions into this list.
  DebugLocStream DebugLocs;

  /// This is a collection of subprogram MDNodes that are processed to
  /// create DIEs.
  SmallPtrSet<const MDNode *, 16> ProcessedSPNodes;

  /// Maps instruction with label emitted before instruction.
  DenseMap<const MachineInstr *, MCSymbol *> LabelsBeforeInsn;

  /// Maps instruction with label emitted after instruction.
  DenseMap<const MachineInstr *, MCSymbol *> LabelsAfterInsn;

  /// History of DBG_VALUE and clobber instructions for each user
  /// variable.  Variables are listed in order of appearance.
  DbgValueHistoryMap DbgValues;

  /// Previous instruction's location information. This is used to
  /// determine label location to indicate scope boundries in dwarf
  /// debug info.
  DebugLoc PrevInstLoc;
  MCSymbol *PrevLabel;

  /// This location indicates end of function prologue and beginning of
  /// function body.
  DebugLoc PrologEndLoc;

  /// If nonnull, stores the current machine function we're processing.
  const MachineFunction *CurFn;

  /// If nonnull, stores the current machine instruction we're processing.
  const MachineInstr *CurMI;

  /// If nonnull, stores the CU in which the previous subprogram was contained.
  const DwarfCompileUnit *PrevCU;

  /// As an optimization, there is no need to emit an entry in the directory
  /// table for the same directory as DW_AT_comp_dir.
  StringRef CompilationDir;

  /// Holder for the file specific debug information.
  DwarfFile InfoHolder;

  /// Holders for the various debug information flags that we might need to
  /// have exposed. See accessor functions below for description.

  /// Holder for imported entities.
  typedef SmallVector<std::pair<const MDNode *, const MDNode *>, 32>
  ImportedEntityMap;
  ImportedEntityMap ScopesWithImportedEntities;

  /// Map from MDNodes for user-defined types to the type units that
  /// describe them.
  DenseMap<const MDNode *, const DwarfTypeUnit *> DwarfTypeUnits;

  SmallVector<
      std::pair<std::unique_ptr<DwarfTypeUnit>, const DICompositeType *>, 1>
      TypeUnitsUnderConstruction;

  /// Whether to emit the pubnames/pubtypes sections.
  bool HasDwarfPubSections;

  /// Whether or not to use AT_ranges for compilation units.
  bool HasCURanges;

  /// Whether we emitted a function into a section other than the
  /// default text.
  bool UsedNonDefaultText;

  /// Whether to use the GNU TLS opcode (instead of the standard opcode).
  bool UseGNUTLSOpcode;

  /// Version of dwarf we're emitting.
  unsigned DwarfVersion;

  /// Maps from a type identifier to the actual MDNode.
  DITypeIdentifierMap TypeIdentifierMap;

  /// DWARF5 Experimental Options
  /// @{
  bool HasDwarfAccelTables;
  bool HasSplitDwarf;

  /// Separated Dwarf Variables
  /// In general these will all be for bits that are left in the
  /// original object file, rather than things that are meant
  /// to be in the .dwo sections.

  /// Holder for the skeleton information.
  DwarfFile SkeletonHolder;

  /// Store file names for type units under fission in a line table
  /// header that will be emitted into debug_line.dwo.
  // FIXME: replace this with a map from comp_dir to table so that we
  // can emit multiple tables during LTO each of which uses directory
  // 0, referencing the comp_dir of all the type units that use it.
  MCDwarfDwoLineTable SplitTypeUnitFileTable;
  /// @}
  
  /// True iff there are multiple CUs in this module.
  bool SingleCU;
  bool IsDarwin;
  bool IsPS4;

  AddressPool AddrPool;

  DwarfAccelTable AccelNames;
  DwarfAccelTable AccelObjC;
  DwarfAccelTable AccelNamespace;
  DwarfAccelTable AccelTypes;

  DenseMap<const Function *, DISubprogram *> FunctionDIs;

  MCDwarfDwoLineTable *getDwoLineTable(const DwarfCompileUnit &);

  const SmallVectorImpl<std::unique_ptr<DwarfUnit>> &getUnits() {
    return InfoHolder.getUnits();
  }

  typedef DbgValueHistoryMap::InlinedVariable InlinedVariable;

  /// Find abstract variable associated with Var.
  DbgVariable *getExistingAbstractVariable(InlinedVariable IV,
                                           const DILocalVariable *&Cleansed);
  DbgVariable *getExistingAbstractVariable(InlinedVariable IV);
  void createAbstractVariable(const DILocalVariable *DV, LexicalScope *Scope);
  void ensureAbstractVariableIsCreated(InlinedVariable Var,
                                       const MDNode *Scope);
  void ensureAbstractVariableIsCreatedIfScoped(InlinedVariable Var,
                                               const MDNode *Scope);

  DbgVariable *createConcreteVariable(LexicalScope &Scope, InlinedVariable IV);

  /// Construct a DIE for this abstract scope.
  void constructAbstractSubprogramScopeDIE(LexicalScope *Scope);

  /// Compute the size and offset of a DIE given an incoming Offset.
  unsigned computeSizeAndOffset(DIE *Die, unsigned Offset);

  /// Compute the size and offset of all the DIEs.
  void computeSizeAndOffsets();

  /// Collect info for variables that were optimized out.
  void collectDeadVariables();

  void finishVariableDefinitions();

  void finishSubprogramDefinitions();

  /// Finish off debug information after all functions have been
  /// processed.
  void finalizeModuleInfo();

  /// Emit the debug info section.
  void emitDebugInfo();

  /// Emit the abbreviation section.
  void emitAbbreviations();

  /// Emit a specified accelerator table.
  void emitAccel(DwarfAccelTable &Accel, MCSection *Section,
                 StringRef TableName);

  /// Emit visible names into a hashed accelerator table section.
  void emitAccelNames();

  /// Emit objective C classes and categories into a hashed
  /// accelerator table section.
  void emitAccelObjC();

  /// Emit namespace dies into a hashed accelerator table.
  void emitAccelNamespaces();

  /// Emit type dies into a hashed accelerator table.
  void emitAccelTypes();

  /// Emit visible names into a debug pubnames section.
  /// \param GnuStyle determines whether or not we want to emit
  /// additional information into the table ala newer gcc for gdb
  /// index.
  void emitDebugPubNames(bool GnuStyle = false);

  /// Emit visible types into a debug pubtypes section.
  /// \param GnuStyle determines whether or not we want to emit
  /// additional information into the table ala newer gcc for gdb
  /// index.
  void emitDebugPubTypes(bool GnuStyle = false);

  void emitDebugPubSection(
      bool GnuStyle, MCSection *PSec, StringRef Name,
      const StringMap<const DIE *> &(DwarfCompileUnit::*Accessor)() const);

  /// Emit visible names into a debug str section.
  void emitDebugStr();

  /// Emit visible names into a debug loc section.
  void emitDebugLoc();

  /// Emit visible names into a debug loc dwo section.
  void emitDebugLocDWO();

  /// Emit visible names into a debug aranges section.
  void emitDebugARanges();

  /// Emit visible names into a debug ranges section.
  void emitDebugRanges();

  /// Emit inline info using custom format.
  void emitDebugInlineInfo();

  /// DWARF 5 Experimental Split Dwarf Emitters

  /// Initialize common features of skeleton units.
  void initSkeletonUnit(const DwarfUnit &U, DIE &Die,
                        std::unique_ptr<DwarfUnit> NewU);

  /// Construct the split debug info compile unit for the debug info
  /// section.
  DwarfCompileUnit &constructSkeletonCU(const DwarfCompileUnit &CU);

  /// Construct the split debug info compile unit for the debug info
  /// section.
  DwarfTypeUnit &constructSkeletonTU(DwarfTypeUnit &TU);

  /// Emit the debug info dwo section.
  void emitDebugInfoDWO();

  /// Emit the debug abbrev dwo section.
  void emitDebugAbbrevDWO();

  /// Emit the debug line dwo section.
  void emitDebugLineDWO();

  /// Emit the debug str dwo section.
  void emitDebugStrDWO();

  /// Flags to let the linker know we have emitted new style pubnames. Only
  /// emit it here if we don't have a skeleton CU for split dwarf.
  void addGnuPubAttributes(DwarfUnit &U, DIE &D) const;

  /// Create new DwarfCompileUnit for the given metadata node with tag
  /// DW_TAG_compile_unit.
  DwarfCompileUnit &constructDwarfCompileUnit(const DICompileUnit *DIUnit);

  /// Construct imported_module or imported_declaration DIE.
  void constructAndAddImportedEntityDIE(DwarfCompileUnit &TheCU,
                                        const DIImportedEntity *N);

  /// Register a source line with debug info. Returns the unique
  /// label that was emitted and which provides correspondence to the
  /// source line list.
  void recordSourceLine(unsigned Line, unsigned Col, const MDNode *Scope,
                        unsigned Flags);

  /// Indentify instructions that are marking the beginning of or
  /// ending of a scope.
  void identifyScopeMarkers();

  /// Populate LexicalScope entries with variables' info.
  void collectVariableInfo(DwarfCompileUnit &TheCU, const DISubprogram *SP,
                           DenseSet<InlinedVariable> &ProcessedVars);

  /// Build the location list for all DBG_VALUEs in the
  /// function that describe the same variable.
  void buildLocationList(SmallVectorImpl<DebugLocEntry> &DebugLoc,
                         const DbgValueHistoryMap::InstrRanges &Ranges);

  /// Collect variable information from the side table maintained
  /// by MMI.
  void collectVariableInfoFromMMITable(DenseSet<InlinedVariable> &P);

  /// Ensure that a label will be emitted before MI.
  void requestLabelBeforeInsn(const MachineInstr *MI) {
    LabelsBeforeInsn.insert(std::make_pair(MI, nullptr));
  }

  /// Ensure that a label will be emitted after MI.
  void requestLabelAfterInsn(const MachineInstr *MI) {
    LabelsAfterInsn.insert(std::make_pair(MI, nullptr));
  }

public:
  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  DwarfDebug(AsmPrinter *A, Module *M);

  ~DwarfDebug() override;

  /// Emit all Dwarf sections that should come prior to the
  /// content.
  void beginModule();

  /// Emit all Dwarf sections that should come after the content.
  void endModule() override;

  /// Gather pre-function debug information.
  void beginFunction(const MachineFunction *MF) override;

  /// Gather and emit post-function debug information.
  void endFunction(const MachineFunction *MF) override;

  /// Process beginning of an instruction.
  void beginInstruction(const MachineInstr *MI) override;

  /// Process end of an instruction.
  void endInstruction() override;

  /// Add a DIE to the set of types that we're going to pull into
  /// type units.
  void addDwarfTypeUnitType(DwarfCompileUnit &CU, StringRef Identifier,
                            DIE &Die, const DICompositeType *CTy);

  /// Add a label so that arange data can be generated for it.
  void addArangeLabel(SymbolCU SCU) { ArangeLabels.push_back(SCU); }

  /// For symbols that have a size designated (e.g. common symbols),
  /// this tracks that size.
  void setSymbolSize(const MCSymbol *Sym, uint64_t Size) override {
    SymSize[Sym] = Size;
  }

  /// Returns whether to use DW_OP_GNU_push_tls_address, instead of the
  /// standard DW_OP_form_tls_address opcode
  bool useGNUTLSOpcode() const { return UseGNUTLSOpcode; }

  // Experimental DWARF5 features.

  /// Returns whether or not to emit tables that dwarf consumers can
  /// use to accelerate lookup.
  bool useDwarfAccelTables() const { return HasDwarfAccelTables; }

  /// Returns whether or not to change the current debug info for the
  /// split dwarf proposal support.
  bool useSplitDwarf() const { return HasSplitDwarf; }

  /// Returns the Dwarf Version.
  unsigned getDwarfVersion() const { return DwarfVersion; }

  /// Returns the previous CU that was being updated
  const DwarfCompileUnit *getPrevCU() const { return PrevCU; }
  void setPrevCU(const DwarfCompileUnit *PrevCU) { this->PrevCU = PrevCU; }

  /// Returns the entries for the .debug_loc section.
  const DebugLocStream &getDebugLocs() const { return DebugLocs; }

  /// Emit an entry for the debug loc section. This can be used to
  /// handle an entry that's going to be emitted into the debug loc section.
  void emitDebugLocEntry(ByteStreamer &Streamer,
                         const DebugLocStream::Entry &Entry);

  /// Emit the location for a debug loc entry, including the size header.
  void emitDebugLocEntryLocation(const DebugLocStream::Entry &Entry);

  /// Find the MDNode for the given reference.
  template <typename T> T *resolve(TypedDINodeRef<T> Ref) const {
    return Ref.resolve(TypeIdentifierMap);
  }

  /// Return the TypeIdentifierMap.
  const DITypeIdentifierMap &getTypeIdentifierMap() const {
    return TypeIdentifierMap;
  }

  /// Find the DwarfCompileUnit for the given CU Die.
  DwarfCompileUnit *lookupUnit(const DIE *CU) const {
    return CUDieMap.lookup(CU);
  }
  /// isSubprogramContext - Return true if Context is either a subprogram
  /// or another context nested inside a subprogram.
  bool isSubprogramContext(const MDNode *Context);

  void addSubprogramNames(const DISubprogram *SP, DIE &Die);

  AddressPool &getAddressPool() { return AddrPool; }

  void addAccelName(StringRef Name, const DIE &Die);

  void addAccelObjC(StringRef Name, const DIE &Die);

  void addAccelNamespace(StringRef Name, const DIE &Die);

  void addAccelType(StringRef Name, const DIE &Die, char Flags);

  const MachineFunction *getCurrentFunction() const { return CurFn; }

  iterator_range<ImportedEntityMap::const_iterator>
  findImportedEntitiesForScope(const MDNode *Scope) const {
    return make_range(std::equal_range(
        ScopesWithImportedEntities.begin(), ScopesWithImportedEntities.end(),
        std::pair<const MDNode *, const MDNode *>(Scope, nullptr),
        less_first()));
  }

  /// A helper function to check whether the DIE for a given Scope is
  /// going to be null.
  bool isLexicalScopeDIENull(LexicalScope *Scope);

  /// Return Label preceding the instruction.
  MCSymbol *getLabelBeforeInsn(const MachineInstr *MI);

  /// Return Label immediately following the instruction.
  MCSymbol *getLabelAfterInsn(const MachineInstr *MI);

  // FIXME: Sink these functions down into DwarfFile/Dwarf*Unit.

  SmallPtrSet<const MDNode *, 16> &getProcessedSPNodes() {
    return ProcessedSPNodes;
  }
};
} // End of namespace llvm

#endif
