//===-- llvm/CodeGen/DwarfDebug.cpp - Dwarf Debug Framework ---------------===//
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

#include "DwarfDebug.h"
#include "ByteStreamer.h"
#include "DIEHash.h"
#include "DebugLocEntry.h"
#include "DwarfCompileUnit.h"
#include "DwarfExpression.h"
#include "DwarfUnit.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

#define DEBUG_TYPE "dwarfdebug"

static cl::opt<bool>
DisableDebugInfoPrinting("disable-debug-info-print", cl::Hidden,
                         cl::desc("Disable debug info printing"));

static cl::opt<bool> UnknownLocations(
    "use-unknown-locations", cl::Hidden,
    cl::desc("Make an absence of debug location information explicit."),
    cl::init(false));

static cl::opt<bool>
GenerateGnuPubSections("generate-gnu-dwarf-pub-sections", cl::Hidden,
                       cl::desc("Generate GNU-style pubnames and pubtypes"),
                       cl::init(false));

static cl::opt<bool> GenerateARangeSection("generate-arange-section",
                                           cl::Hidden,
                                           cl::desc("Generate dwarf aranges"),
                                           cl::init(false));

namespace {
enum DefaultOnOff { Default, Enable, Disable };
}

static cl::opt<DefaultOnOff>
DwarfAccelTables("dwarf-accel-tables", cl::Hidden,
                 cl::desc("Output prototype dwarf accelerator tables."),
                 cl::values(clEnumVal(Default, "Default for platform"),
                            clEnumVal(Enable, "Enabled"),
                            clEnumVal(Disable, "Disabled"), clEnumValEnd),
                 cl::init(Default));

static cl::opt<DefaultOnOff>
SplitDwarf("split-dwarf", cl::Hidden,
           cl::desc("Output DWARF5 split debug info."),
           cl::values(clEnumVal(Default, "Default for platform"),
                      clEnumVal(Enable, "Enabled"),
                      clEnumVal(Disable, "Disabled"), clEnumValEnd),
           cl::init(Default));

static cl::opt<DefaultOnOff>
DwarfPubSections("generate-dwarf-pub-sections", cl::Hidden,
                 cl::desc("Generate DWARF pubnames and pubtypes sections"),
                 cl::values(clEnumVal(Default, "Default for platform"),
                            clEnumVal(Enable, "Enabled"),
                            clEnumVal(Disable, "Disabled"), clEnumValEnd),
                 cl::init(Default));

static const char *const DWARFGroupName = "DWARF Emission";
static const char *const DbgTimerName = "DWARF Debug Writer";

void DebugLocDwarfExpression::EmitOp(uint8_t Op, const char *Comment) {
  BS.EmitInt8(
      Op, Comment ? Twine(Comment) + " " + dwarf::OperationEncodingString(Op)
                  : dwarf::OperationEncodingString(Op));
}

void DebugLocDwarfExpression::EmitSigned(int64_t Value) {
  BS.EmitSLEB128(Value, Twine(Value));
}

void DebugLocDwarfExpression::EmitUnsigned(uint64_t Value) {
  BS.EmitULEB128(Value, Twine(Value));
}

bool DebugLocDwarfExpression::isFrameRegister(unsigned MachineReg) {
  // This information is not available while emitting .debug_loc entries.
  return false;
}

//===----------------------------------------------------------------------===//

/// resolve - Look in the DwarfDebug map for the MDNode that
/// corresponds to the reference.
template <typename T> T *DbgVariable::resolve(TypedDINodeRef<T> Ref) const {
  return DD->resolve(Ref);
}

bool DbgVariable::isBlockByrefVariable() const {
  assert(Var && "Invalid complex DbgVariable!");
  return Var->getType()
      .resolve(DD->getTypeIdentifierMap())
      ->isBlockByrefStruct();
}

const DIType *DbgVariable::getType() const {
  DIType *Ty = Var->getType().resolve(DD->getTypeIdentifierMap());
  // FIXME: isBlockByrefVariable should be reformulated in terms of complex
  // addresses instead.
  if (Ty->isBlockByrefStruct()) {
    /* Byref variables, in Blocks, are declared by the programmer as
       "SomeType VarName;", but the compiler creates a
       __Block_byref_x_VarName struct, and gives the variable VarName
       either the struct, or a pointer to the struct, as its type.  This
       is necessary for various behind-the-scenes things the compiler
       needs to do with by-reference variables in blocks.

       However, as far as the original *programmer* is concerned, the
       variable should still have type 'SomeType', as originally declared.

       The following function dives into the __Block_byref_x_VarName
       struct to find the original type of the variable.  This will be
       passed back to the code generating the type for the Debug
       Information Entry for the variable 'VarName'.  'VarName' will then
       have the original type 'SomeType' in its debug information.

       The original type 'SomeType' will be the type of the field named
       'VarName' inside the __Block_byref_x_VarName struct.

       NOTE: In order for this to not completely fail on the debugger
       side, the Debug Information Entry for the variable VarName needs to
       have a DW_AT_location that tells the debugger how to unwind through
       the pointers and __Block_byref_x_VarName struct to find the actual
       value of the variable.  The function addBlockByrefType does this.  */
    DIType *subType = Ty;
    uint16_t tag = Ty->getTag();

    if (tag == dwarf::DW_TAG_pointer_type)
      subType = resolve(cast<DIDerivedType>(Ty)->getBaseType());

    auto Elements = cast<DICompositeTypeBase>(subType)->getElements();
    for (unsigned i = 0, N = Elements.size(); i < N; ++i) {
      auto *DT = cast<DIDerivedTypeBase>(Elements[i]);
      if (getName() == DT->getName())
        return resolve(DT->getBaseType());
    }
  }
  return Ty;
}

static LLVM_CONSTEXPR DwarfAccelTable::Atom TypeAtoms[] = {
    DwarfAccelTable::Atom(dwarf::DW_ATOM_die_offset, dwarf::DW_FORM_data4),
    DwarfAccelTable::Atom(dwarf::DW_ATOM_die_tag, dwarf::DW_FORM_data2),
    DwarfAccelTable::Atom(dwarf::DW_ATOM_type_flags, dwarf::DW_FORM_data1)};

DwarfDebug::DwarfDebug(AsmPrinter *A, Module *M)
    : Asm(A), MMI(Asm->MMI), DebugLocs(A->OutStreamer->isVerboseAsm()),
      PrevLabel(nullptr), InfoHolder(A, "info_string", DIEValueAllocator),
      UsedNonDefaultText(false),
      SkeletonHolder(A, "skel_string", DIEValueAllocator),
      IsDarwin(Triple(A->getTargetTriple()).isOSDarwin()),
      IsPS4(Triple(A->getTargetTriple()).isPS4()),
      AccelNames(DwarfAccelTable::Atom(dwarf::DW_ATOM_die_offset,
                                       dwarf::DW_FORM_data4)),
      AccelObjC(DwarfAccelTable::Atom(dwarf::DW_ATOM_die_offset,
                                      dwarf::DW_FORM_data4)),
      AccelNamespace(DwarfAccelTable::Atom(dwarf::DW_ATOM_die_offset,
                                           dwarf::DW_FORM_data4)),
      AccelTypes(TypeAtoms) {

  CurFn = nullptr;
  CurMI = nullptr;

  // Turn on accelerator tables for Darwin by default, pubnames by
  // default for non-Darwin/PS4, and handle split dwarf.
  if (DwarfAccelTables == Default)
    HasDwarfAccelTables = IsDarwin;
  else
    HasDwarfAccelTables = DwarfAccelTables == Enable;

  if (SplitDwarf == Default)
    HasSplitDwarf = false;
  else
    HasSplitDwarf = SplitDwarf == Enable;

  if (DwarfPubSections == Default)
    HasDwarfPubSections = !IsDarwin && !IsPS4;
  else
    HasDwarfPubSections = DwarfPubSections == Enable;

  unsigned DwarfVersionNumber = Asm->TM.Options.MCOptions.DwarfVersion;
  DwarfVersion = DwarfVersionNumber ? DwarfVersionNumber
                                    : MMI->getModule()->getDwarfVersion();

  // Darwin and PS4 use the standard TLS opcode (defined in DWARF 3).
  // Everybody else uses GNU's.
  UseGNUTLSOpcode = !(IsDarwin || IsPS4) || DwarfVersion < 3;

  Asm->OutStreamer->getContext().setDwarfVersion(DwarfVersion);

  {
    NamedRegionTimer T(DbgTimerName, DWARFGroupName, TimePassesIsEnabled);
    beginModule();
  }
}

// Define out of line so we don't have to include DwarfUnit.h in DwarfDebug.h.
DwarfDebug::~DwarfDebug() { }

static bool isObjCClass(StringRef Name) {
  return Name.startswith("+") || Name.startswith("-");
}

static bool hasObjCCategory(StringRef Name) {
  if (!isObjCClass(Name))
    return false;

  return Name.find(") ") != StringRef::npos;
}

static void getObjCClassCategory(StringRef In, StringRef &Class,
                                 StringRef &Category) {
  if (!hasObjCCategory(In)) {
    Class = In.slice(In.find('[') + 1, In.find(' '));
    Category = "";
    return;
  }

  Class = In.slice(In.find('[') + 1, In.find('('));
  Category = In.slice(In.find('[') + 1, In.find(' '));
  return;
}

static StringRef getObjCMethodName(StringRef In) {
  return In.slice(In.find(' ') + 1, In.find(']'));
}

// Add the various names to the Dwarf accelerator table names.
// TODO: Determine whether or not we should add names for programs
// that do not have a DW_AT_name or DW_AT_linkage_name field - this
// is only slightly different than the lookup of non-standard ObjC names.
void DwarfDebug::addSubprogramNames(const DISubprogram *SP, DIE &Die) {
  if (!SP->isDefinition())
    return;
  addAccelName(SP->getName(), Die);

  // If the linkage name is different than the name, go ahead and output
  // that as well into the name table.
  if (SP->getLinkageName() != "" && SP->getName() != SP->getLinkageName())
    addAccelName(SP->getLinkageName(), Die);

  // If this is an Objective-C selector name add it to the ObjC accelerator
  // too.
  if (isObjCClass(SP->getName())) {
    StringRef Class, Category;
    getObjCClassCategory(SP->getName(), Class, Category);
    addAccelObjC(Class, Die);
    if (Category != "")
      addAccelObjC(Category, Die);
    // Also add the base method name to the name table.
    addAccelName(getObjCMethodName(SP->getName()), Die);
  }
}

/// isSubprogramContext - Return true if Context is either a subprogram
/// or another context nested inside a subprogram.
bool DwarfDebug::isSubprogramContext(const MDNode *Context) {
  if (!Context)
    return false;
  if (isa<DISubprogram>(Context))
    return true;
  if (auto *T = dyn_cast<DIType>(Context))
    return isSubprogramContext(resolve(T->getScope()));
  return false;
}

/// Check whether we should create a DIE for the given Scope, return true
/// if we don't create a DIE (the corresponding DIE is null).
bool DwarfDebug::isLexicalScopeDIENull(LexicalScope *Scope) {
  if (Scope->isAbstractScope())
    return false;

  // We don't create a DIE if there is no Range.
  const SmallVectorImpl<InsnRange> &Ranges = Scope->getRanges();
  if (Ranges.empty())
    return true;

  if (Ranges.size() > 1)
    return false;

  // We don't create a DIE if we have a single Range and the end label
  // is null.
  return !getLabelAfterInsn(Ranges.front().second);
}

template <typename Func> void forBothCUs(DwarfCompileUnit &CU, Func F) {
  F(CU);
  if (auto *SkelCU = CU.getSkeleton())
    F(*SkelCU);
}

void DwarfDebug::constructAbstractSubprogramScopeDIE(LexicalScope *Scope) {
  assert(Scope && Scope->getScopeNode());
  assert(Scope->isAbstractScope());
  assert(!Scope->getInlinedAt());

  const MDNode *SP = Scope->getScopeNode();

  ProcessedSPNodes.insert(SP);

  // Find the subprogram's DwarfCompileUnit in the SPMap in case the subprogram
  // was inlined from another compile unit.
  auto &CU = SPMap[SP];
  forBothCUs(*CU, [&](DwarfCompileUnit &CU) {
    CU.constructAbstractSubprogramScopeDIE(Scope);
  });
}

void DwarfDebug::addGnuPubAttributes(DwarfUnit &U, DIE &D) const {
  if (!GenerateGnuPubSections)
    return;

  U.addFlag(D, dwarf::DW_AT_GNU_pubnames);
}

// Create new DwarfCompileUnit for the given metadata node with tag
// DW_TAG_compile_unit.
DwarfCompileUnit &
DwarfDebug::constructDwarfCompileUnit(const DICompileUnit *DIUnit) {
  StringRef FN = DIUnit->getFilename();
  CompilationDir = DIUnit->getDirectory();

  auto OwnedUnit = make_unique<DwarfCompileUnit>(
      InfoHolder.getUnits().size(), DIUnit, Asm, this, &InfoHolder);
  DwarfCompileUnit &NewCU = *OwnedUnit;
  DIE &Die = NewCU.getUnitDie();
  InfoHolder.addUnit(std::move(OwnedUnit));
  if (useSplitDwarf())
    NewCU.setSkeleton(constructSkeletonCU(NewCU));

  // LTO with assembly output shares a single line table amongst multiple CUs.
  // To avoid the compilation directory being ambiguous, let the line table
  // explicitly describe the directory of all files, never relying on the
  // compilation directory.
  if (!Asm->OutStreamer->hasRawTextSupport() || SingleCU)
    Asm->OutStreamer->getContext().setMCLineTableCompilationDir(
        NewCU.getUniqueID(), CompilationDir);

  NewCU.addString(Die, dwarf::DW_AT_producer, DIUnit->getProducer());
  NewCU.addUInt(Die, dwarf::DW_AT_language, dwarf::DW_FORM_data2,
                DIUnit->getSourceLanguage());
  NewCU.addString(Die, dwarf::DW_AT_name, FN);

  if (!useSplitDwarf()) {
    NewCU.initStmtList();

    // If we're using split dwarf the compilation dir is going to be in the
    // skeleton CU and so we don't need to duplicate it here.
    if (!CompilationDir.empty())
      NewCU.addString(Die, dwarf::DW_AT_comp_dir, CompilationDir);

    addGnuPubAttributes(NewCU, Die);
  }

  if (DIUnit->isOptimized())
    NewCU.addFlag(Die, dwarf::DW_AT_APPLE_optimized);

  StringRef Flags = DIUnit->getFlags();
  if (!Flags.empty())
    NewCU.addString(Die, dwarf::DW_AT_APPLE_flags, Flags);

  if (unsigned RVer = DIUnit->getRuntimeVersion())
    NewCU.addUInt(Die, dwarf::DW_AT_APPLE_major_runtime_vers,
                  dwarf::DW_FORM_data1, RVer);

  if (useSplitDwarf())
    NewCU.initSection(Asm->getObjFileLowering().getDwarfInfoDWOSection());
  else
    NewCU.initSection(Asm->getObjFileLowering().getDwarfInfoSection());

  CUMap.insert(std::make_pair(DIUnit, &NewCU));
  CUDieMap.insert(std::make_pair(&Die, &NewCU));
  return NewCU;
}

void DwarfDebug::constructAndAddImportedEntityDIE(DwarfCompileUnit &TheCU,
                                                  const DIImportedEntity *N) {
  if (DIE *D = TheCU.getOrCreateContextDIE(N->getScope()))
    D->addChild(TheCU.constructImportedEntityDIE(N));
}

// Emit all Dwarf sections that should come prior to the content. Create
// global DIEs and emit initial debug info sections. This is invoked by
// the target AsmPrinter.
void DwarfDebug::beginModule() {
  if (DisableDebugInfoPrinting)
    return;

  const Module *M = MMI->getModule();

  FunctionDIs = makeSubprogramMap(*M);

  NamedMDNode *CU_Nodes = M->getNamedMetadata("llvm.dbg.cu");
  if (!CU_Nodes)
    return;
  TypeIdentifierMap = generateDITypeIdentifierMap(CU_Nodes);

  SingleCU = CU_Nodes->getNumOperands() == 1;

  for (MDNode *N : CU_Nodes->operands()) {
    auto *CUNode = cast<DICompileUnit>(N);
    DwarfCompileUnit &CU = constructDwarfCompileUnit(CUNode);
    for (auto *IE : CUNode->getImportedEntities())
      ScopesWithImportedEntities.push_back(std::make_pair(IE->getScope(), IE));
    // Stable sort to preserve the order of appearance of imported entities.
    // This is to avoid out-of-order processing of interdependent declarations
    // within the same scope, e.g. { namespace A = base; namespace B = A; }
    std::stable_sort(ScopesWithImportedEntities.begin(),
                     ScopesWithImportedEntities.end(), less_first());
    for (auto *GV : CUNode->getGlobalVariables())
      CU.getOrCreateGlobalVariableDIE(GV);
    for (auto *SP : CUNode->getSubprograms())
      SPMap.insert(std::make_pair(SP, &CU));
    for (auto *Ty : CUNode->getEnumTypes()) {
      // The enum types array by design contains pointers to
      // MDNodes rather than DIRefs. Unique them here.
      CU.getOrCreateTypeDIE(cast<DIType>(resolve(Ty->getRef())));
    }
    for (auto *Ty : CUNode->getRetainedTypes()) {
      // The retained types array by design contains pointers to
      // MDNodes rather than DIRefs. Unique them here.
      CU.getOrCreateTypeDIE(cast<DIType>(resolve(Ty->getRef())));
    }
    // Emit imported_modules last so that the relevant context is already
    // available.
    for (auto *IE : CUNode->getImportedEntities())
      constructAndAddImportedEntityDIE(CU, IE);
  }

  // Tell MMI that we have debug info.
  MMI->setDebugInfoAvailability(true);
}

void DwarfDebug::finishVariableDefinitions() {
  for (const auto &Var : ConcreteVariables) {
    DIE *VariableDie = Var->getDIE();
    assert(VariableDie);
    // FIXME: Consider the time-space tradeoff of just storing the unit pointer
    // in the ConcreteVariables list, rather than looking it up again here.
    // DIE::getUnit isn't simple - it walks parent pointers, etc.
    DwarfCompileUnit *Unit = lookupUnit(VariableDie->getUnit());
    assert(Unit);
    DbgVariable *AbsVar = getExistingAbstractVariable(
        InlinedVariable(Var->getVariable(), Var->getInlinedAt()));
    if (AbsVar && AbsVar->getDIE()) {
      Unit->addDIEEntry(*VariableDie, dwarf::DW_AT_abstract_origin,
                        *AbsVar->getDIE());
    } else
      Unit->applyVariableAttributes(*Var, *VariableDie);
  }
}

void DwarfDebug::finishSubprogramDefinitions() {
  for (const auto &P : SPMap)
    forBothCUs(*P.second, [&](DwarfCompileUnit &CU) {
      CU.finishSubprogramDefinition(cast<DISubprogram>(P.first));
    });
}


// Collect info for variables that were optimized out.
void DwarfDebug::collectDeadVariables() {
  const Module *M = MMI->getModule();

  if (NamedMDNode *CU_Nodes = M->getNamedMetadata("llvm.dbg.cu")) {
    for (MDNode *N : CU_Nodes->operands()) {
      auto *TheCU = cast<DICompileUnit>(N);
      // Construct subprogram DIE and add variables DIEs.
      DwarfCompileUnit *SPCU =
          static_cast<DwarfCompileUnit *>(CUMap.lookup(TheCU));
      assert(SPCU && "Unable to find Compile Unit!");
      for (auto *SP : TheCU->getSubprograms()) {
        if (ProcessedSPNodes.count(SP) != 0)
          continue;
        SPCU->collectDeadVariables(SP);
      }
    }
  }
}

void DwarfDebug::finalizeModuleInfo() {
  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();

  finishSubprogramDefinitions();

  finishVariableDefinitions();

  // Collect info for variables that were optimized out.
  collectDeadVariables();

  // Handle anything that needs to be done on a per-unit basis after
  // all other generation.
  for (const auto &P : CUMap) {
    auto &TheCU = *P.second;
    // Emit DW_AT_containing_type attribute to connect types with their
    // vtable holding type.
    TheCU.constructContainingTypeDIEs();

    // Add CU specific attributes if we need to add any.
    // If we're splitting the dwarf out now that we've got the entire
    // CU then add the dwo id to it.
    auto *SkCU = TheCU.getSkeleton();
    if (useSplitDwarf()) {
      // Emit a unique identifier for this CU.
      uint64_t ID = DIEHash(Asm).computeCUSignature(TheCU.getUnitDie());
      TheCU.addUInt(TheCU.getUnitDie(), dwarf::DW_AT_GNU_dwo_id,
                    dwarf::DW_FORM_data8, ID);
      SkCU->addUInt(SkCU->getUnitDie(), dwarf::DW_AT_GNU_dwo_id,
                    dwarf::DW_FORM_data8, ID);

      // We don't keep track of which addresses are used in which CU so this
      // is a bit pessimistic under LTO.
      if (!AddrPool.isEmpty()) {
        const MCSymbol *Sym = TLOF.getDwarfAddrSection()->getBeginSymbol();
        SkCU->addSectionLabel(SkCU->getUnitDie(), dwarf::DW_AT_GNU_addr_base,
                              Sym, Sym);
      }
      if (!SkCU->getRangeLists().empty()) {
        const MCSymbol *Sym = TLOF.getDwarfRangesSection()->getBeginSymbol();
        SkCU->addSectionLabel(SkCU->getUnitDie(), dwarf::DW_AT_GNU_ranges_base,
                              Sym, Sym);
      }
    }

    // If we have code split among multiple sections or non-contiguous
    // ranges of code then emit a DW_AT_ranges attribute on the unit that will
    // remain in the .o file, otherwise add a DW_AT_low_pc.
    // FIXME: We should use ranges allow reordering of code ala
    // .subsections_via_symbols in mach-o. This would mean turning on
    // ranges for all subprogram DIEs for mach-o.
    DwarfCompileUnit &U = SkCU ? *SkCU : TheCU;
    if (unsigned NumRanges = TheCU.getRanges().size()) {
      if (NumRanges > 1)
        // A DW_AT_low_pc attribute may also be specified in combination with
        // DW_AT_ranges to specify the default base address for use in
        // location lists (see Section 2.6.2) and range lists (see Section
        // 2.17.3).
        U.addUInt(U.getUnitDie(), dwarf::DW_AT_low_pc, dwarf::DW_FORM_addr, 0);
      else
        U.setBaseAddress(TheCU.getRanges().front().getStart());
      U.attachRangesOrLowHighPC(U.getUnitDie(), TheCU.takeRanges());
    }
  }

  // Compute DIE offsets and sizes.
  InfoHolder.computeSizeAndOffsets();
  if (useSplitDwarf())
    SkeletonHolder.computeSizeAndOffsets();
}

// Emit all Dwarf sections that should come after the content.
void DwarfDebug::endModule() {
  assert(CurFn == nullptr);
  assert(CurMI == nullptr);

  // If we aren't actually generating debug info (check beginModule -
  // conditionalized on !DisableDebugInfoPrinting and the presence of the
  // llvm.dbg.cu metadata node)
  if (!MMI->hasDebugInfo())
    return;

  // Finalize the debug info for the module.
  finalizeModuleInfo();

  emitDebugStr();

  if (useSplitDwarf())
    emitDebugLocDWO();
  else
    // Emit info into a debug loc section.
    emitDebugLoc();

  // Corresponding abbreviations into a abbrev section.
  emitAbbreviations();

  // Emit all the DIEs into a debug info section.
  emitDebugInfo();

  // Emit info into a debug aranges section.
  if (GenerateARangeSection)
    emitDebugARanges();

  // Emit info into a debug ranges section.
  emitDebugRanges();

  if (useSplitDwarf()) {
    emitDebugStrDWO();
    emitDebugInfoDWO();
    emitDebugAbbrevDWO();
    emitDebugLineDWO();
    // Emit DWO addresses.
    AddrPool.emit(*Asm, Asm->getObjFileLowering().getDwarfAddrSection());
  }

  // Emit info into the dwarf accelerator table sections.
  if (useDwarfAccelTables()) {
    emitAccelNames();
    emitAccelObjC();
    emitAccelNamespaces();
    emitAccelTypes();
  }

  // Emit the pubnames and pubtypes sections if requested.
  if (HasDwarfPubSections) {
    emitDebugPubNames(GenerateGnuPubSections);
    emitDebugPubTypes(GenerateGnuPubSections);
  }

  // clean up.
  SPMap.clear();
  AbstractVariables.clear();
}

// Find abstract variable, if any, associated with Var.
DbgVariable *
DwarfDebug::getExistingAbstractVariable(InlinedVariable IV,
                                        const DILocalVariable *&Cleansed) {
  // More then one inlined variable corresponds to one abstract variable.
  Cleansed = IV.first;
  auto I = AbstractVariables.find(Cleansed);
  if (I != AbstractVariables.end())
    return I->second.get();
  return nullptr;
}

DbgVariable *DwarfDebug::getExistingAbstractVariable(InlinedVariable IV) {
  const DILocalVariable *Cleansed;
  return getExistingAbstractVariable(IV, Cleansed);
}

void DwarfDebug::createAbstractVariable(const DILocalVariable *Var,
                                        LexicalScope *Scope) {
  auto AbsDbgVariable = make_unique<DbgVariable>(Var, /* IA */ nullptr, this);
  InfoHolder.addScopeVariable(Scope, AbsDbgVariable.get());
  AbstractVariables[Var] = std::move(AbsDbgVariable);
}

void DwarfDebug::ensureAbstractVariableIsCreated(InlinedVariable IV,
                                                 const MDNode *ScopeNode) {
  const DILocalVariable *Cleansed = nullptr;
  if (getExistingAbstractVariable(IV, Cleansed))
    return;

  createAbstractVariable(Cleansed, LScopes.getOrCreateAbstractScope(
                                       cast<DILocalScope>(ScopeNode)));
}

void DwarfDebug::ensureAbstractVariableIsCreatedIfScoped(
    InlinedVariable IV, const MDNode *ScopeNode) {
  const DILocalVariable *Cleansed = nullptr;
  if (getExistingAbstractVariable(IV, Cleansed))
    return;

  if (LexicalScope *Scope =
          LScopes.findAbstractScope(cast_or_null<DILocalScope>(ScopeNode)))
    createAbstractVariable(Cleansed, Scope);
}

// Collect variable information from side table maintained by MMI.
void DwarfDebug::collectVariableInfoFromMMITable(
    DenseSet<InlinedVariable> &Processed) {
  for (const auto &VI : MMI->getVariableDbgInfo()) {
    if (!VI.Var)
      continue;
    assert(VI.Var->isValidLocationForIntrinsic(VI.Loc) &&
           "Expected inlined-at fields to agree");

    InlinedVariable Var(VI.Var, VI.Loc->getInlinedAt());
    Processed.insert(Var);
    LexicalScope *Scope = LScopes.findLexicalScope(VI.Loc);

    // If variable scope is not found then skip this variable.
    if (!Scope)
      continue;

    ensureAbstractVariableIsCreatedIfScoped(Var, Scope->getScopeNode());
    auto RegVar = make_unique<DbgVariable>(Var.first, Var.second, this);
    RegVar->initializeMMI(VI.Expr, VI.Slot);
    if (InfoHolder.addScopeVariable(Scope, RegVar.get()))
      ConcreteVariables.push_back(std::move(RegVar));
  }
}

// Get .debug_loc entry for the instruction range starting at MI.
static DebugLocEntry::Value getDebugLocValue(const MachineInstr *MI) {
  const DIExpression *Expr = MI->getDebugExpression();

  assert(MI->getNumOperands() == 4);
  if (MI->getOperand(0).isReg()) {
    MachineLocation MLoc;
    // If the second operand is an immediate, this is a
    // register-indirect address.
    if (!MI->getOperand(1).isImm())
      MLoc.set(MI->getOperand(0).getReg());
    else
      MLoc.set(MI->getOperand(0).getReg(), MI->getOperand(1).getImm());
    return DebugLocEntry::Value(Expr, MLoc);
  }
  if (MI->getOperand(0).isImm())
    return DebugLocEntry::Value(Expr, MI->getOperand(0).getImm());
  if (MI->getOperand(0).isFPImm())
    return DebugLocEntry::Value(Expr, MI->getOperand(0).getFPImm());
  if (MI->getOperand(0).isCImm())
    return DebugLocEntry::Value(Expr, MI->getOperand(0).getCImm());

  llvm_unreachable("Unexpected 4-operand DBG_VALUE instruction!");
}

/// Determine whether two variable pieces overlap.
static bool piecesOverlap(const DIExpression *P1, const DIExpression *P2) {
  if (!P1->isBitPiece() || !P2->isBitPiece())
    return true;
  unsigned l1 = P1->getBitPieceOffset();
  unsigned l2 = P2->getBitPieceOffset();
  unsigned r1 = l1 + P1->getBitPieceSize();
  unsigned r2 = l2 + P2->getBitPieceSize();
  // True where [l1,r1[ and [r1,r2[ overlap.
  return (l1 < r2) && (l2 < r1);
}

/// Build the location list for all DBG_VALUEs in the function that
/// describe the same variable.  If the ranges of several independent
/// pieces of the same variable overlap partially, split them up and
/// combine the ranges. The resulting DebugLocEntries are will have
/// strict monotonically increasing begin addresses and will never
/// overlap.
//
// Input:
//
//   Ranges History [var, loc, piece ofs size]
// 0 |      [x, (reg0, piece 0, 32)]
// 1 | |    [x, (reg1, piece 32, 32)] <- IsPieceOfPrevEntry
// 2 | |    ...
// 3   |    [clobber reg0]
// 4        [x, (mem, piece 0, 64)] <- overlapping with both previous pieces of
//                                     x.
//
// Output:
//
// [0-1]    [x, (reg0, piece  0, 32)]
// [1-3]    [x, (reg0, piece  0, 32), (reg1, piece 32, 32)]
// [3-4]    [x, (reg1, piece 32, 32)]
// [4- ]    [x, (mem,  piece  0, 64)]
void
DwarfDebug::buildLocationList(SmallVectorImpl<DebugLocEntry> &DebugLoc,
                              const DbgValueHistoryMap::InstrRanges &Ranges) {
  SmallVector<DebugLocEntry::Value, 4> OpenRanges;

  for (auto I = Ranges.begin(), E = Ranges.end(); I != E; ++I) {
    const MachineInstr *Begin = I->first;
    const MachineInstr *End = I->second;
    assert(Begin->isDebugValue() && "Invalid History entry");

    // Check if a variable is inaccessible in this range.
    if (Begin->getNumOperands() > 1 &&
        Begin->getOperand(0).isReg() && !Begin->getOperand(0).getReg()) {
      OpenRanges.clear();
      continue;
    }

    // If this piece overlaps with any open ranges, truncate them.
    const DIExpression *DIExpr = Begin->getDebugExpression();
    auto Last = std::remove_if(OpenRanges.begin(), OpenRanges.end(),
                               [&](DebugLocEntry::Value R) {
      return piecesOverlap(DIExpr, R.getExpression());
    });
    OpenRanges.erase(Last, OpenRanges.end());

    const MCSymbol *StartLabel = getLabelBeforeInsn(Begin);
    assert(StartLabel && "Forgot label before DBG_VALUE starting a range!");

    const MCSymbol *EndLabel;
    if (End != nullptr)
      EndLabel = getLabelAfterInsn(End);
    else if (std::next(I) == Ranges.end())
      EndLabel = Asm->getFunctionEnd();
    else
      EndLabel = getLabelBeforeInsn(std::next(I)->first);
    assert(EndLabel && "Forgot label after instruction ending a range!");

    DEBUG(dbgs() << "DotDebugLoc: " << *Begin << "\n");

    auto Value = getDebugLocValue(Begin);
    DebugLocEntry Loc(StartLabel, EndLabel, Value);
    bool couldMerge = false;

    // If this is a piece, it may belong to the current DebugLocEntry.
    if (DIExpr->isBitPiece()) {
      // Add this value to the list of open ranges.
      OpenRanges.push_back(Value);

      // Attempt to add the piece to the last entry.
      if (!DebugLoc.empty())
        if (DebugLoc.back().MergeValues(Loc))
          couldMerge = true;
    }

    if (!couldMerge) {
      // Need to add a new DebugLocEntry. Add all values from still
      // valid non-overlapping pieces.
      if (OpenRanges.size())
        Loc.addValues(OpenRanges);

      DebugLoc.push_back(std::move(Loc));
    }

    // Attempt to coalesce the ranges of two otherwise identical
    // DebugLocEntries.
    auto CurEntry = DebugLoc.rbegin();
    DEBUG({
      dbgs() << CurEntry->getValues().size() << " Values:\n";
      for (auto &Value : CurEntry->getValues())
        Value.getExpression()->dump();
      dbgs() << "-----\n";
    });

    auto PrevEntry = std::next(CurEntry);
    if (PrevEntry != DebugLoc.rend() && PrevEntry->MergeRanges(*CurEntry))
      DebugLoc.pop_back();
  }
}

DbgVariable *DwarfDebug::createConcreteVariable(LexicalScope &Scope,
                                                InlinedVariable IV) {
  ensureAbstractVariableIsCreatedIfScoped(IV, Scope.getScopeNode());
  ConcreteVariables.push_back(
      make_unique<DbgVariable>(IV.first, IV.second, this));
  InfoHolder.addScopeVariable(&Scope, ConcreteVariables.back().get());
  return ConcreteVariables.back().get();
}

// Find variables for each lexical scope.
void DwarfDebug::collectVariableInfo(DwarfCompileUnit &TheCU,
                                     const DISubprogram *SP,
                                     DenseSet<InlinedVariable> &Processed) {
  // Grab the variable info that was squirreled away in the MMI side-table.
  collectVariableInfoFromMMITable(Processed);

  for (const auto &I : DbgValues) {
    InlinedVariable IV = I.first;
    if (Processed.count(IV))
      continue;

    // Instruction ranges, specifying where IV is accessible.
    const auto &Ranges = I.second;
    if (Ranges.empty())
      continue;

    LexicalScope *Scope = nullptr;
    if (const DILocation *IA = IV.second)
      Scope = LScopes.findInlinedScope(IV.first->getScope(), IA);
    else
      Scope = LScopes.findLexicalScope(IV.first->getScope());
    // If variable scope is not found then skip this variable.
    if (!Scope)
      continue;

    Processed.insert(IV);
    DbgVariable *RegVar = createConcreteVariable(*Scope, IV);

    const MachineInstr *MInsn = Ranges.front().first;
    assert(MInsn->isDebugValue() && "History must begin with debug value");

    // Check if the first DBG_VALUE is valid for the rest of the function.
    if (Ranges.size() == 1 && Ranges.front().second == nullptr) {
      RegVar->initializeDbgValue(MInsn);
      continue;
    }

    // Handle multiple DBG_VALUE instructions describing one variable.
    DebugLocStream::ListBuilder List(DebugLocs, TheCU, *Asm, *RegVar, *MInsn);

    // Build the location list for this variable.
    SmallVector<DebugLocEntry, 8> Entries;
    buildLocationList(Entries, Ranges);

    // If the variable has an DIBasicType, extract it.  Basic types cannot have
    // unique identifiers, so don't bother resolving the type with the
    // identifier map.
    const DIBasicType *BT = dyn_cast<DIBasicType>(
        static_cast<const Metadata *>(IV.first->getType()));

    // Finalize the entry by lowering it into a DWARF bytestream.
    for (auto &Entry : Entries)
      Entry.finalize(*Asm, List, BT);
  }

  // Collect info for variables that were optimized out.
  for (const DILocalVariable *DV : SP->getVariables()) {
    if (Processed.insert(InlinedVariable(DV, nullptr)).second)
      if (LexicalScope *Scope = LScopes.findLexicalScope(DV->getScope()))
        createConcreteVariable(*Scope, InlinedVariable(DV, nullptr));
  }
}

// Return Label preceding the instruction.
MCSymbol *DwarfDebug::getLabelBeforeInsn(const MachineInstr *MI) {
  MCSymbol *Label = LabelsBeforeInsn.lookup(MI);
  assert(Label && "Didn't insert label before instruction");
  return Label;
}

// Return Label immediately following the instruction.
MCSymbol *DwarfDebug::getLabelAfterInsn(const MachineInstr *MI) {
  return LabelsAfterInsn.lookup(MI);
}

// Process beginning of an instruction.
void DwarfDebug::beginInstruction(const MachineInstr *MI) {
  assert(CurMI == nullptr);
  CurMI = MI;
  // Check if source location changes, but ignore DBG_VALUE locations.
  if (!MI->isDebugValue()) {
    DebugLoc DL = MI->getDebugLoc();
    if (DL != PrevInstLoc) {
      if (DL) {
        unsigned Flags = 0;
        PrevInstLoc = DL;
        if (DL == PrologEndLoc) {
          Flags |= DWARF2_FLAG_PROLOGUE_END;
          PrologEndLoc = DebugLoc();
          Flags |= DWARF2_FLAG_IS_STMT;
        }
        if (DL.getLine() !=
            Asm->OutStreamer->getContext().getCurrentDwarfLoc().getLine())
          Flags |= DWARF2_FLAG_IS_STMT;

        const MDNode *Scope = DL.getScope();
        recordSourceLine(DL.getLine(), DL.getCol(), Scope, Flags);
      } else if (UnknownLocations) {
        PrevInstLoc = DL;
        recordSourceLine(0, 0, nullptr, 0);
      }
    }
  }

  // Insert labels where requested.
  DenseMap<const MachineInstr *, MCSymbol *>::iterator I =
      LabelsBeforeInsn.find(MI);

  // No label needed.
  if (I == LabelsBeforeInsn.end())
    return;

  // Label already assigned.
  if (I->second)
    return;

  if (!PrevLabel) {
    PrevLabel = MMI->getContext().createTempSymbol();
    Asm->OutStreamer->EmitLabel(PrevLabel);
  }
  I->second = PrevLabel;
}

// Process end of an instruction.
void DwarfDebug::endInstruction() {
  assert(CurMI != nullptr);
  // Don't create a new label after DBG_VALUE instructions.
  // They don't generate code.
  if (!CurMI->isDebugValue())
    PrevLabel = nullptr;

  DenseMap<const MachineInstr *, MCSymbol *>::iterator I =
      LabelsAfterInsn.find(CurMI);
  CurMI = nullptr;

  // No label needed.
  if (I == LabelsAfterInsn.end())
    return;

  // Label already assigned.
  if (I->second)
    return;

  // We need a label after this instruction.
  if (!PrevLabel) {
    PrevLabel = MMI->getContext().createTempSymbol();
    Asm->OutStreamer->EmitLabel(PrevLabel);
  }
  I->second = PrevLabel;
}

// Each LexicalScope has first instruction and last instruction to mark
// beginning and end of a scope respectively. Create an inverse map that list
// scopes starts (and ends) with an instruction. One instruction may start (or
// end) multiple scopes. Ignore scopes that are not reachable.
void DwarfDebug::identifyScopeMarkers() {
  SmallVector<LexicalScope *, 4> WorkList;
  WorkList.push_back(LScopes.getCurrentFunctionScope());
  while (!WorkList.empty()) {
    LexicalScope *S = WorkList.pop_back_val();

    const SmallVectorImpl<LexicalScope *> &Children = S->getChildren();
    if (!Children.empty())
      WorkList.append(Children.begin(), Children.end());

    if (S->isAbstractScope())
      continue;

    for (const InsnRange &R : S->getRanges()) {
      assert(R.first && "InsnRange does not have first instruction!");
      assert(R.second && "InsnRange does not have second instruction!");
      requestLabelBeforeInsn(R.first);
      requestLabelAfterInsn(R.second);
    }
  }
}

static DebugLoc findPrologueEndLoc(const MachineFunction *MF) {
  // First known non-DBG_VALUE and non-frame setup location marks
  // the beginning of the function body.
  for (const auto &MBB : *MF)
    for (const auto &MI : MBB)
      if (!MI.isDebugValue() && !MI.getFlag(MachineInstr::FrameSetup) &&
          MI.getDebugLoc()) {
        // Did the target forget to set the FrameSetup flag for CFI insns?
        assert(!MI.isCFIInstruction() &&
               "First non-frame-setup instruction is a CFI instruction.");
        return MI.getDebugLoc();
      }
  return DebugLoc();
}

// Gather pre-function debug information.  Assumes being called immediately
// after the function entry point has been emitted.
void DwarfDebug::beginFunction(const MachineFunction *MF) {
  CurFn = MF;

  // If there's no debug info for the function we're not going to do anything.
  if (!MMI->hasDebugInfo())
    return;

  auto DI = FunctionDIs.find(MF->getFunction());
  if (DI == FunctionDIs.end())
    return;

  // Grab the lexical scopes for the function, if we don't have any of those
  // then we're not going to be able to do anything.
  LScopes.initialize(*MF);
  if (LScopes.empty())
    return;

  assert(DbgValues.empty() && "DbgValues map wasn't cleaned!");

  // Make sure that each lexical scope will have a begin/end label.
  identifyScopeMarkers();

  // Set DwarfDwarfCompileUnitID in MCContext to the Compile Unit this function
  // belongs to so that we add to the correct per-cu line table in the
  // non-asm case.
  LexicalScope *FnScope = LScopes.getCurrentFunctionScope();
  // FnScope->getScopeNode() and DI->second should represent the same function,
  // though they may not be the same MDNode due to inline functions merged in
  // LTO where the debug info metadata still differs (either due to distinct
  // written differences - two versions of a linkonce_odr function
  // written/copied into two separate files, or some sub-optimal metadata that
  // isn't structurally identical (see: file path/name info from clang, which
  // includes the directory of the cpp file being built, even when the file name
  // is absolute (such as an <> lookup header)))
  DwarfCompileUnit *TheCU = SPMap.lookup(FnScope->getScopeNode());
  assert(TheCU && "Unable to find compile unit!");
  if (Asm->OutStreamer->hasRawTextSupport())
    // Use a single line table if we are generating assembly.
    Asm->OutStreamer->getContext().setDwarfCompileUnitID(0);
  else
    Asm->OutStreamer->getContext().setDwarfCompileUnitID(TheCU->getUniqueID());

  // Calculate history for local variables.
  calculateDbgValueHistory(MF, Asm->MF->getSubtarget().getRegisterInfo(),
                           DbgValues);

  // Request labels for the full history.
  for (const auto &I : DbgValues) {
    const auto &Ranges = I.second;
    if (Ranges.empty())
      continue;

    // The first mention of a function argument gets the CurrentFnBegin
    // label, so arguments are visible when breaking at function entry.
    const DILocalVariable *DIVar = Ranges.front().first->getDebugVariable();
    if (DIVar->getTag() == dwarf::DW_TAG_arg_variable &&
        getDISubprogram(DIVar->getScope())->describes(MF->getFunction())) {
      LabelsBeforeInsn[Ranges.front().first] = Asm->getFunctionBegin();
      if (Ranges.front().first->getDebugExpression()->isBitPiece()) {
        // Mark all non-overlapping initial pieces.
        for (auto I = Ranges.begin(); I != Ranges.end(); ++I) {
          const DIExpression *Piece = I->first->getDebugExpression();
          if (std::all_of(Ranges.begin(), I,
                          [&](DbgValueHistoryMap::InstrRange Pred) {
                return !piecesOverlap(Piece, Pred.first->getDebugExpression());
              }))
            LabelsBeforeInsn[I->first] = Asm->getFunctionBegin();
          else
            break;
        }
      }
    }

    for (const auto &Range : Ranges) {
      requestLabelBeforeInsn(Range.first);
      if (Range.second)
        requestLabelAfterInsn(Range.second);
    }
  }

  PrevInstLoc = DebugLoc();
  PrevLabel = Asm->getFunctionBegin();

  // Record beginning of function.
  PrologEndLoc = findPrologueEndLoc(MF);
  if (DILocation *L = PrologEndLoc) {
    // We'd like to list the prologue as "not statements" but GDB behaves
    // poorly if we do that. Revisit this with caution/GDB (7.5+) testing.
    auto *SP = L->getInlinedAtScope()->getSubprogram();
    recordSourceLine(SP->getScopeLine(), 0, SP, DWARF2_FLAG_IS_STMT);
  }
}

// Gather and emit post-function debug information.
void DwarfDebug::endFunction(const MachineFunction *MF) {
  assert(CurFn == MF &&
      "endFunction should be called with the same function as beginFunction");

  if (!MMI->hasDebugInfo() || LScopes.empty() ||
      !FunctionDIs.count(MF->getFunction())) {
    // If we don't have a lexical scope for this function then there will
    // be a hole in the range information. Keep note of this by setting the
    // previously used section to nullptr.
    PrevCU = nullptr;
    CurFn = nullptr;
    return;
  }

  // Set DwarfDwarfCompileUnitID in MCContext to default value.
  Asm->OutStreamer->getContext().setDwarfCompileUnitID(0);

  LexicalScope *FnScope = LScopes.getCurrentFunctionScope();
  auto *SP = cast<DISubprogram>(FnScope->getScopeNode());
  DwarfCompileUnit &TheCU = *SPMap.lookup(SP);

  DenseSet<InlinedVariable> ProcessedVars;
  collectVariableInfo(TheCU, SP, ProcessedVars);

  // Add the range of this function to the list of ranges for the CU.
  TheCU.addRange(RangeSpan(Asm->getFunctionBegin(), Asm->getFunctionEnd()));

  // Under -gmlt, skip building the subprogram if there are no inlined
  // subroutines inside it.
  if (TheCU.getCUNode()->getEmissionKind() == DIBuilder::LineTablesOnly &&
      LScopes.getAbstractScopesList().empty() && !IsDarwin) {
    assert(InfoHolder.getScopeVariables().empty());
    assert(DbgValues.empty());
    // FIXME: This wouldn't be true in LTO with a -g (with inlining) CU followed
    // by a -gmlt CU. Add a test and remove this assertion.
    assert(AbstractVariables.empty());
    LabelsBeforeInsn.clear();
    LabelsAfterInsn.clear();
    PrevLabel = nullptr;
    CurFn = nullptr;
    return;
  }

#ifndef NDEBUG
  size_t NumAbstractScopes = LScopes.getAbstractScopesList().size();
#endif
  // Construct abstract scopes.
  for (LexicalScope *AScope : LScopes.getAbstractScopesList()) {
    auto *SP = cast<DISubprogram>(AScope->getScopeNode());
    // Collect info for variables that were optimized out.
    for (const DILocalVariable *DV : SP->getVariables()) {
      if (!ProcessedVars.insert(InlinedVariable(DV, nullptr)).second)
        continue;
      ensureAbstractVariableIsCreated(InlinedVariable(DV, nullptr),
                                      DV->getScope());
      assert(LScopes.getAbstractScopesList().size() == NumAbstractScopes
             && "ensureAbstractVariableIsCreated inserted abstract scopes");
    }
    constructAbstractSubprogramScopeDIE(AScope);
  }

  TheCU.constructSubprogramScopeDIE(FnScope);
  if (auto *SkelCU = TheCU.getSkeleton())
    if (!LScopes.getAbstractScopesList().empty())
      SkelCU->constructSubprogramScopeDIE(FnScope);

  // Clear debug info
  // Ownership of DbgVariables is a bit subtle - ScopeVariables owns all the
  // DbgVariables except those that are also in AbstractVariables (since they
  // can be used cross-function)
  InfoHolder.getScopeVariables().clear();
  DbgValues.clear();
  LabelsBeforeInsn.clear();
  LabelsAfterInsn.clear();
  PrevLabel = nullptr;
  CurFn = nullptr;
}

// Register a source line with debug info. Returns the  unique label that was
// emitted and which provides correspondence to the source line list.
void DwarfDebug::recordSourceLine(unsigned Line, unsigned Col, const MDNode *S,
                                  unsigned Flags) {
  StringRef Fn;
  StringRef Dir;
  unsigned Src = 1;
  unsigned Discriminator = 0;
  if (auto *Scope = cast_or_null<DIScope>(S)) {
    Fn = Scope->getFilename();
    Dir = Scope->getDirectory();
    if (auto *LBF = dyn_cast<DILexicalBlockFile>(Scope))
      Discriminator = LBF->getDiscriminator();

    unsigned CUID = Asm->OutStreamer->getContext().getDwarfCompileUnitID();
    Src = static_cast<DwarfCompileUnit &>(*InfoHolder.getUnits()[CUID])
              .getOrCreateSourceID(Fn, Dir);
  }
  Asm->OutStreamer->EmitDwarfLocDirective(Src, Line, Col, Flags, 0,
                                          Discriminator, Fn);
}

//===----------------------------------------------------------------------===//
// Emit Methods
//===----------------------------------------------------------------------===//

// Emit the debug info section.
void DwarfDebug::emitDebugInfo() {
  DwarfFile &Holder = useSplitDwarf() ? SkeletonHolder : InfoHolder;
  Holder.emitUnits(/* UseOffsets */ false);
}

// Emit the abbreviation section.
void DwarfDebug::emitAbbreviations() {
  DwarfFile &Holder = useSplitDwarf() ? SkeletonHolder : InfoHolder;

  Holder.emitAbbrevs(Asm->getObjFileLowering().getDwarfAbbrevSection());
}

void DwarfDebug::emitAccel(DwarfAccelTable &Accel, MCSection *Section,
                           StringRef TableName) {
  Accel.FinalizeTable(Asm, TableName);
  Asm->OutStreamer->SwitchSection(Section);

  // Emit the full data.
  Accel.emit(Asm, Section->getBeginSymbol(), this);
}

// Emit visible names into a hashed accelerator table section.
void DwarfDebug::emitAccelNames() {
  emitAccel(AccelNames, Asm->getObjFileLowering().getDwarfAccelNamesSection(),
            "Names");
}

// Emit objective C classes and categories into a hashed accelerator table
// section.
void DwarfDebug::emitAccelObjC() {
  emitAccel(AccelObjC, Asm->getObjFileLowering().getDwarfAccelObjCSection(),
            "ObjC");
}

// Emit namespace dies into a hashed accelerator table.
void DwarfDebug::emitAccelNamespaces() {
  emitAccel(AccelNamespace,
            Asm->getObjFileLowering().getDwarfAccelNamespaceSection(),
            "namespac");
}

// Emit type dies into a hashed accelerator table.
void DwarfDebug::emitAccelTypes() {
  emitAccel(AccelTypes, Asm->getObjFileLowering().getDwarfAccelTypesSection(),
            "types");
}

// Public name handling.
// The format for the various pubnames:
//
// dwarf pubnames - offset/name pairs where the offset is the offset into the CU
// for the DIE that is named.
//
// gnu pubnames - offset/index value/name tuples where the offset is the offset
// into the CU and the index value is computed according to the type of value
// for the DIE that is named.
//
// For type units the offset is the offset of the skeleton DIE. For split dwarf
// it's the offset within the debug_info/debug_types dwo section, however, the
// reference in the pubname header doesn't change.

/// computeIndexValue - Compute the gdb index value for the DIE and CU.
static dwarf::PubIndexEntryDescriptor computeIndexValue(DwarfUnit *CU,
                                                        const DIE *Die) {
  dwarf::GDBIndexEntryLinkage Linkage = dwarf::GIEL_STATIC;

  // We could have a specification DIE that has our most of our knowledge,
  // look for that now.
  if (DIEValue SpecVal = Die->findAttribute(dwarf::DW_AT_specification)) {
    DIE &SpecDIE = SpecVal.getDIEEntry().getEntry();
    if (SpecDIE.findAttribute(dwarf::DW_AT_external))
      Linkage = dwarf::GIEL_EXTERNAL;
  } else if (Die->findAttribute(dwarf::DW_AT_external))
    Linkage = dwarf::GIEL_EXTERNAL;

  switch (Die->getTag()) {
  case dwarf::DW_TAG_class_type:
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_union_type:
  case dwarf::DW_TAG_enumeration_type:
    return dwarf::PubIndexEntryDescriptor(
        dwarf::GIEK_TYPE, CU->getLanguage() != dwarf::DW_LANG_C_plus_plus
                              ? dwarf::GIEL_STATIC
                              : dwarf::GIEL_EXTERNAL);
  case dwarf::DW_TAG_typedef:
  case dwarf::DW_TAG_base_type:
  case dwarf::DW_TAG_subrange_type:
    return dwarf::PubIndexEntryDescriptor(dwarf::GIEK_TYPE, dwarf::GIEL_STATIC);
  case dwarf::DW_TAG_namespace:
    return dwarf::GIEK_TYPE;
  case dwarf::DW_TAG_subprogram:
    return dwarf::PubIndexEntryDescriptor(dwarf::GIEK_FUNCTION, Linkage);
  case dwarf::DW_TAG_variable:
    return dwarf::PubIndexEntryDescriptor(dwarf::GIEK_VARIABLE, Linkage);
  case dwarf::DW_TAG_enumerator:
    return dwarf::PubIndexEntryDescriptor(dwarf::GIEK_VARIABLE,
                                          dwarf::GIEL_STATIC);
  default:
    return dwarf::GIEK_NONE;
  }
}

/// emitDebugPubNames - Emit visible names into a debug pubnames section.
///
void DwarfDebug::emitDebugPubNames(bool GnuStyle) {
  MCSection *PSec = GnuStyle
                        ? Asm->getObjFileLowering().getDwarfGnuPubNamesSection()
                        : Asm->getObjFileLowering().getDwarfPubNamesSection();

  emitDebugPubSection(GnuStyle, PSec, "Names",
                      &DwarfCompileUnit::getGlobalNames);
}

void DwarfDebug::emitDebugPubSection(
    bool GnuStyle, MCSection *PSec, StringRef Name,
    const StringMap<const DIE *> &(DwarfCompileUnit::*Accessor)() const) {
  for (const auto &NU : CUMap) {
    DwarfCompileUnit *TheU = NU.second;

    const auto &Globals = (TheU->*Accessor)();

    if (Globals.empty())
      continue;

    if (auto *Skeleton = TheU->getSkeleton())
      TheU = Skeleton;

    // Start the dwarf pubnames section.
    Asm->OutStreamer->SwitchSection(PSec);

    // Emit the header.
    Asm->OutStreamer->AddComment("Length of Public " + Name + " Info");
    MCSymbol *BeginLabel = Asm->createTempSymbol("pub" + Name + "_begin");
    MCSymbol *EndLabel = Asm->createTempSymbol("pub" + Name + "_end");
    Asm->EmitLabelDifference(EndLabel, BeginLabel, 4);

    Asm->OutStreamer->EmitLabel(BeginLabel);

    Asm->OutStreamer->AddComment("DWARF Version");
    Asm->EmitInt16(dwarf::DW_PUBNAMES_VERSION);

    Asm->OutStreamer->AddComment("Offset of Compilation Unit Info");
    Asm->emitDwarfSymbolReference(TheU->getLabelBegin());

    Asm->OutStreamer->AddComment("Compilation Unit Length");
    Asm->EmitInt32(TheU->getLength());

    // Emit the pubnames for this compilation unit.
    for (const auto &GI : Globals) {
      const char *Name = GI.getKeyData();
      const DIE *Entity = GI.second;

      Asm->OutStreamer->AddComment("DIE offset");
      Asm->EmitInt32(Entity->getOffset());

      if (GnuStyle) {
        dwarf::PubIndexEntryDescriptor Desc = computeIndexValue(TheU, Entity);
        Asm->OutStreamer->AddComment(
            Twine("Kind: ") + dwarf::GDBIndexEntryKindString(Desc.Kind) + ", " +
            dwarf::GDBIndexEntryLinkageString(Desc.Linkage));
        Asm->EmitInt8(Desc.toBits());
      }

      Asm->OutStreamer->AddComment("External Name");
      Asm->OutStreamer->EmitBytes(StringRef(Name, GI.getKeyLength() + 1));
    }

    Asm->OutStreamer->AddComment("End Mark");
    Asm->EmitInt32(0);
    Asm->OutStreamer->EmitLabel(EndLabel);
  }
}

void DwarfDebug::emitDebugPubTypes(bool GnuStyle) {
  MCSection *PSec = GnuStyle
                        ? Asm->getObjFileLowering().getDwarfGnuPubTypesSection()
                        : Asm->getObjFileLowering().getDwarfPubTypesSection();

  emitDebugPubSection(GnuStyle, PSec, "Types",
                      &DwarfCompileUnit::getGlobalTypes);
}

// Emit visible names into a debug str section.
void DwarfDebug::emitDebugStr() {
  DwarfFile &Holder = useSplitDwarf() ? SkeletonHolder : InfoHolder;
  Holder.emitStrings(Asm->getObjFileLowering().getDwarfStrSection());
}

void DwarfDebug::emitDebugLocEntry(ByteStreamer &Streamer,
                                   const DebugLocStream::Entry &Entry) {
  auto &&Comments = DebugLocs.getComments(Entry);
  auto Comment = Comments.begin();
  auto End = Comments.end();
  for (uint8_t Byte : DebugLocs.getBytes(Entry))
    Streamer.EmitInt8(Byte, Comment != End ? *(Comment++) : "");
}

static void emitDebugLocValue(const AsmPrinter &AP, const DIBasicType *BT,
                              ByteStreamer &Streamer,
                              const DebugLocEntry::Value &Value,
                              unsigned PieceOffsetInBits) {
  DebugLocDwarfExpression DwarfExpr(*AP.MF->getSubtarget().getRegisterInfo(),
                                    AP.getDwarfDebug()->getDwarfVersion(),
                                    Streamer);
  // Regular entry.
  if (Value.isInt()) {
    if (BT && (BT->getEncoding() == dwarf::DW_ATE_signed ||
               BT->getEncoding() == dwarf::DW_ATE_signed_char))
      DwarfExpr.AddSignedConstant(Value.getInt());
    else
      DwarfExpr.AddUnsignedConstant(Value.getInt());
  } else if (Value.isLocation()) {
    MachineLocation Loc = Value.getLoc();
    const DIExpression *Expr = Value.getExpression();
    if (!Expr || !Expr->getNumElements())
      // Regular entry.
      AP.EmitDwarfRegOp(Streamer, Loc);
    else {
      // Complex address entry.
      if (Loc.getOffset()) {
        DwarfExpr.AddMachineRegIndirect(Loc.getReg(), Loc.getOffset());
        DwarfExpr.AddExpression(Expr->expr_op_begin(), Expr->expr_op_end(),
                                PieceOffsetInBits);
      } else
        DwarfExpr.AddMachineRegExpression(Expr, Loc.getReg(),
                                          PieceOffsetInBits);
    }
  }
  // else ... ignore constant fp. There is not any good way to
  // to represent them here in dwarf.
  // FIXME: ^
}

void DebugLocEntry::finalize(const AsmPrinter &AP,
                             DebugLocStream::ListBuilder &List,
                             const DIBasicType *BT) {
  DebugLocStream::EntryBuilder Entry(List, Begin, End);
  BufferByteStreamer Streamer = Entry.getStreamer();
  const DebugLocEntry::Value &Value = Values[0];
  if (Value.isBitPiece()) {
    // Emit all pieces that belong to the same variable and range.
    assert(std::all_of(Values.begin(), Values.end(), [](DebugLocEntry::Value P) {
          return P.isBitPiece();
        }) && "all values are expected to be pieces");
    assert(std::is_sorted(Values.begin(), Values.end()) &&
           "pieces are expected to be sorted");
   
    unsigned Offset = 0;
    for (auto Piece : Values) {
      const DIExpression *Expr = Piece.getExpression();
      unsigned PieceOffset = Expr->getBitPieceOffset();
      unsigned PieceSize = Expr->getBitPieceSize();
      assert(Offset <= PieceOffset && "overlapping or duplicate pieces");
      if (Offset < PieceOffset) {
        // The DWARF spec seriously mandates pieces with no locations for gaps.
        DebugLocDwarfExpression Expr(*AP.MF->getSubtarget().getRegisterInfo(),
                                     AP.getDwarfDebug()->getDwarfVersion(),
                                     Streamer);
        Expr.AddOpPiece(PieceOffset-Offset, 0);
        Offset += PieceOffset-Offset;
      }
      Offset += PieceSize;

      emitDebugLocValue(AP, BT, Streamer, Piece, PieceOffset);
    }
  } else {
    assert(Values.size() == 1 && "only pieces may have >1 value");
    emitDebugLocValue(AP, BT, Streamer, Value, 0);
  }
}

void DwarfDebug::emitDebugLocEntryLocation(const DebugLocStream::Entry &Entry) {
  // Emit the size.
  Asm->OutStreamer->AddComment("Loc expr size");
  Asm->EmitInt16(DebugLocs.getBytes(Entry).size());

  // Emit the entry.
  APByteStreamer Streamer(*Asm);
  emitDebugLocEntry(Streamer, Entry);
}

// Emit locations into the debug loc section.
void DwarfDebug::emitDebugLoc() {
  // Start the dwarf loc section.
  Asm->OutStreamer->SwitchSection(
      Asm->getObjFileLowering().getDwarfLocSection());
  unsigned char Size = Asm->getDataLayout().getPointerSize();
  for (const auto &List : DebugLocs.getLists()) {
    Asm->OutStreamer->EmitLabel(List.Label);
    const DwarfCompileUnit *CU = List.CU;
    for (const auto &Entry : DebugLocs.getEntries(List)) {
      // Set up the range. This range is relative to the entry point of the
      // compile unit. This is a hard coded 0 for low_pc when we're emitting
      // ranges, or the DW_AT_low_pc on the compile unit otherwise.
      if (auto *Base = CU->getBaseAddress()) {
        Asm->EmitLabelDifference(Entry.BeginSym, Base, Size);
        Asm->EmitLabelDifference(Entry.EndSym, Base, Size);
      } else {
        Asm->OutStreamer->EmitSymbolValue(Entry.BeginSym, Size);
        Asm->OutStreamer->EmitSymbolValue(Entry.EndSym, Size);
      }

      emitDebugLocEntryLocation(Entry);
    }
    Asm->OutStreamer->EmitIntValue(0, Size);
    Asm->OutStreamer->EmitIntValue(0, Size);
  }
}

void DwarfDebug::emitDebugLocDWO() {
  Asm->OutStreamer->SwitchSection(
      Asm->getObjFileLowering().getDwarfLocDWOSection());
  for (const auto &List : DebugLocs.getLists()) {
    Asm->OutStreamer->EmitLabel(List.Label);
    for (const auto &Entry : DebugLocs.getEntries(List)) {
      // Just always use start_length for now - at least that's one address
      // rather than two. We could get fancier and try to, say, reuse an
      // address we know we've emitted elsewhere (the start of the function?
      // The start of the CU or CU subrange that encloses this range?)
      Asm->EmitInt8(dwarf::DW_LLE_start_length_entry);
      unsigned idx = AddrPool.getIndex(Entry.BeginSym);
      Asm->EmitULEB128(idx);
      Asm->EmitLabelDifference(Entry.EndSym, Entry.BeginSym, 4);

      emitDebugLocEntryLocation(Entry);
    }
    Asm->EmitInt8(dwarf::DW_LLE_end_of_list_entry);
  }
}

struct ArangeSpan {
  const MCSymbol *Start, *End;
};

// Emit a debug aranges section, containing a CU lookup for any
// address we can tie back to a CU.
void DwarfDebug::emitDebugARanges() {
  // Provides a unique id per text section.
  MapVector<MCSection *, SmallVector<SymbolCU, 8>> SectionMap;

  // Filter labels by section.
  for (const SymbolCU &SCU : ArangeLabels) {
    if (SCU.Sym->isInSection()) {
      // Make a note of this symbol and it's section.
      MCSection *Section = &SCU.Sym->getSection();
      if (!Section->getKind().isMetadata())
        SectionMap[Section].push_back(SCU);
    } else {
      // Some symbols (e.g. common/bss on mach-o) can have no section but still
      // appear in the output. This sucks as we rely on sections to build
      // arange spans. We can do it without, but it's icky.
      SectionMap[nullptr].push_back(SCU);
    }
  }

  // Add terminating symbols for each section.
  for (const auto &I : SectionMap) {
    MCSection *Section = I.first;
    MCSymbol *Sym = nullptr;

    if (Section)
      Sym = Asm->OutStreamer->endSection(Section);

    // Insert a final terminator.
    SectionMap[Section].push_back(SymbolCU(nullptr, Sym));
  }

  DenseMap<DwarfCompileUnit *, std::vector<ArangeSpan>> Spans;

  for (auto &I : SectionMap) {
    const MCSection *Section = I.first;
    SmallVector<SymbolCU, 8> &List = I.second;
    if (List.size() < 2)
      continue;

    // If we have no section (e.g. common), just write out
    // individual spans for each symbol.
    if (!Section) {
      for (const SymbolCU &Cur : List) {
        ArangeSpan Span;
        Span.Start = Cur.Sym;
        Span.End = nullptr;
        if (Cur.CU)
          Spans[Cur.CU].push_back(Span);
      }
      continue;
    }

    // Sort the symbols by offset within the section.
    std::sort(List.begin(), List.end(),
              [&](const SymbolCU &A, const SymbolCU &B) {
      unsigned IA = A.Sym ? Asm->OutStreamer->GetSymbolOrder(A.Sym) : 0;
      unsigned IB = B.Sym ? Asm->OutStreamer->GetSymbolOrder(B.Sym) : 0;

      // Symbols with no order assigned should be placed at the end.
      // (e.g. section end labels)
      if (IA == 0)
        return false;
      if (IB == 0)
        return true;
      return IA < IB;
    });

    // Build spans between each label.
    const MCSymbol *StartSym = List[0].Sym;
    for (size_t n = 1, e = List.size(); n < e; n++) {
      const SymbolCU &Prev = List[n - 1];
      const SymbolCU &Cur = List[n];

      // Try and build the longest span we can within the same CU.
      if (Cur.CU != Prev.CU) {
        ArangeSpan Span;
        Span.Start = StartSym;
        Span.End = Cur.Sym;
        Spans[Prev.CU].push_back(Span);
        StartSym = Cur.Sym;
      }
    }
  }

  // Start the dwarf aranges section.
  Asm->OutStreamer->SwitchSection(
      Asm->getObjFileLowering().getDwarfARangesSection());

  unsigned PtrSize = Asm->getDataLayout().getPointerSize();

  // Build a list of CUs used.
  std::vector<DwarfCompileUnit *> CUs;
  for (const auto &it : Spans) {
    DwarfCompileUnit *CU = it.first;
    CUs.push_back(CU);
  }

  // Sort the CU list (again, to ensure consistent output order).
  std::sort(CUs.begin(), CUs.end(), [](const DwarfUnit *A, const DwarfUnit *B) {
    return A->getUniqueID() < B->getUniqueID();
  });

  // Emit an arange table for each CU we used.
  for (DwarfCompileUnit *CU : CUs) {
    std::vector<ArangeSpan> &List = Spans[CU];

    // Describe the skeleton CU's offset and length, not the dwo file's.
    if (auto *Skel = CU->getSkeleton())
      CU = Skel;

    // Emit size of content not including length itself.
    unsigned ContentSize =
        sizeof(int16_t) + // DWARF ARange version number
        sizeof(int32_t) + // Offset of CU in the .debug_info section
        sizeof(int8_t) +  // Pointer Size (in bytes)
        sizeof(int8_t);   // Segment Size (in bytes)

    unsigned TupleSize = PtrSize * 2;

    // 7.20 in the Dwarf specs requires the table to be aligned to a tuple.
    unsigned Padding =
        OffsetToAlignment(sizeof(int32_t) + ContentSize, TupleSize);

    ContentSize += Padding;
    ContentSize += (List.size() + 1) * TupleSize;

    // For each compile unit, write the list of spans it covers.
    Asm->OutStreamer->AddComment("Length of ARange Set");
    Asm->EmitInt32(ContentSize);
    Asm->OutStreamer->AddComment("DWARF Arange version number");
    Asm->EmitInt16(dwarf::DW_ARANGES_VERSION);
    Asm->OutStreamer->AddComment("Offset Into Debug Info Section");
    Asm->emitDwarfSymbolReference(CU->getLabelBegin());
    Asm->OutStreamer->AddComment("Address Size (in bytes)");
    Asm->EmitInt8(PtrSize);
    Asm->OutStreamer->AddComment("Segment Size (in bytes)");
    Asm->EmitInt8(0);

    Asm->OutStreamer->EmitFill(Padding, 0xff);

    for (const ArangeSpan &Span : List) {
      Asm->EmitLabelReference(Span.Start, PtrSize);

      // Calculate the size as being from the span start to it's end.
      if (Span.End) {
        Asm->EmitLabelDifference(Span.End, Span.Start, PtrSize);
      } else {
        // For symbols without an end marker (e.g. common), we
        // write a single arange entry containing just that one symbol.
        uint64_t Size = SymSize[Span.Start];
        if (Size == 0)
          Size = 1;

        Asm->OutStreamer->EmitIntValue(Size, PtrSize);
      }
    }

    Asm->OutStreamer->AddComment("ARange terminator");
    Asm->OutStreamer->EmitIntValue(0, PtrSize);
    Asm->OutStreamer->EmitIntValue(0, PtrSize);
  }
}

// Emit visible names into a debug ranges section.
void DwarfDebug::emitDebugRanges() {
  // Start the dwarf ranges section.
  Asm->OutStreamer->SwitchSection(
      Asm->getObjFileLowering().getDwarfRangesSection());

  // Size for our labels.
  unsigned char Size = Asm->getDataLayout().getPointerSize();

  // Grab the specific ranges for the compile units in the module.
  for (const auto &I : CUMap) {
    DwarfCompileUnit *TheCU = I.second;

    if (auto *Skel = TheCU->getSkeleton())
      TheCU = Skel;

    // Iterate over the misc ranges for the compile units in the module.
    for (const RangeSpanList &List : TheCU->getRangeLists()) {
      // Emit our symbol so we can find the beginning of the range.
      Asm->OutStreamer->EmitLabel(List.getSym());

      for (const RangeSpan &Range : List.getRanges()) {
        const MCSymbol *Begin = Range.getStart();
        const MCSymbol *End = Range.getEnd();
        assert(Begin && "Range without a begin symbol?");
        assert(End && "Range without an end symbol?");
        if (auto *Base = TheCU->getBaseAddress()) {
          Asm->EmitLabelDifference(Begin, Base, Size);
          Asm->EmitLabelDifference(End, Base, Size);
        } else {
          Asm->OutStreamer->EmitSymbolValue(Begin, Size);
          Asm->OutStreamer->EmitSymbolValue(End, Size);
        }
      }

      // And terminate the list with two 0 values.
      Asm->OutStreamer->EmitIntValue(0, Size);
      Asm->OutStreamer->EmitIntValue(0, Size);
    }
  }
}

// DWARF5 Experimental Separate Dwarf emitters.

void DwarfDebug::initSkeletonUnit(const DwarfUnit &U, DIE &Die,
                                  std::unique_ptr<DwarfUnit> NewU) {
  NewU->addString(Die, dwarf::DW_AT_GNU_dwo_name,
                  U.getCUNode()->getSplitDebugFilename());

  if (!CompilationDir.empty())
    NewU->addString(Die, dwarf::DW_AT_comp_dir, CompilationDir);

  addGnuPubAttributes(*NewU, Die);

  SkeletonHolder.addUnit(std::move(NewU));
}

// This DIE has the following attributes: DW_AT_comp_dir, DW_AT_stmt_list,
// DW_AT_low_pc, DW_AT_high_pc, DW_AT_ranges, DW_AT_dwo_name, DW_AT_dwo_id,
// DW_AT_addr_base, DW_AT_ranges_base.
DwarfCompileUnit &DwarfDebug::constructSkeletonCU(const DwarfCompileUnit &CU) {

  auto OwnedUnit = make_unique<DwarfCompileUnit>(
      CU.getUniqueID(), CU.getCUNode(), Asm, this, &SkeletonHolder);
  DwarfCompileUnit &NewCU = *OwnedUnit;
  NewCU.initSection(Asm->getObjFileLowering().getDwarfInfoSection());

  NewCU.initStmtList();

  initSkeletonUnit(CU, NewCU.getUnitDie(), std::move(OwnedUnit));

  return NewCU;
}

// Emit the .debug_info.dwo section for separated dwarf. This contains the
// compile units that would normally be in debug_info.
void DwarfDebug::emitDebugInfoDWO() {
  assert(useSplitDwarf() && "No split dwarf debug info?");
  // Don't emit relocations into the dwo file.
  InfoHolder.emitUnits(/* UseOffsets */ true);
}

// Emit the .debug_abbrev.dwo section for separated dwarf. This contains the
// abbreviations for the .debug_info.dwo section.
void DwarfDebug::emitDebugAbbrevDWO() {
  assert(useSplitDwarf() && "No split dwarf?");
  InfoHolder.emitAbbrevs(Asm->getObjFileLowering().getDwarfAbbrevDWOSection());
}

void DwarfDebug::emitDebugLineDWO() {
  assert(useSplitDwarf() && "No split dwarf?");
  Asm->OutStreamer->SwitchSection(
      Asm->getObjFileLowering().getDwarfLineDWOSection());
  SplitTypeUnitFileTable.Emit(*Asm->OutStreamer);
}

// Emit the .debug_str.dwo section for separated dwarf. This contains the
// string section and is identical in format to traditional .debug_str
// sections.
void DwarfDebug::emitDebugStrDWO() {
  assert(useSplitDwarf() && "No split dwarf?");
  MCSection *OffSec = Asm->getObjFileLowering().getDwarfStrOffDWOSection();
  InfoHolder.emitStrings(Asm->getObjFileLowering().getDwarfStrDWOSection(),
                         OffSec);
}

MCDwarfDwoLineTable *DwarfDebug::getDwoLineTable(const DwarfCompileUnit &CU) {
  if (!useSplitDwarf())
    return nullptr;
  if (SingleCU)
    SplitTypeUnitFileTable.setCompilationDir(CU.getCUNode()->getDirectory());
  return &SplitTypeUnitFileTable;
}

static uint64_t makeTypeSignature(StringRef Identifier) {
  MD5 Hash;
  Hash.update(Identifier);
  // ... take the least significant 8 bytes and return those. Our MD5
  // implementation always returns its results in little endian, swap bytes
  // appropriately.
  MD5::MD5Result Result;
  Hash.final(Result);
  return support::endian::read64le(Result + 8);
}

void DwarfDebug::addDwarfTypeUnitType(DwarfCompileUnit &CU,
                                      StringRef Identifier, DIE &RefDie,
                                      const DICompositeType *CTy) {
  // Fast path if we're building some type units and one has already used the
  // address pool we know we're going to throw away all this work anyway, so
  // don't bother building dependent types.
  if (!TypeUnitsUnderConstruction.empty() && AddrPool.hasBeenUsed())
    return;

  const DwarfTypeUnit *&TU = DwarfTypeUnits[CTy];
  if (TU) {
    CU.addDIETypeSignature(RefDie, *TU);
    return;
  }

  bool TopLevelType = TypeUnitsUnderConstruction.empty();
  AddrPool.resetUsedFlag();

  auto OwnedUnit = make_unique<DwarfTypeUnit>(
      InfoHolder.getUnits().size() + TypeUnitsUnderConstruction.size(), CU, Asm,
      this, &InfoHolder, getDwoLineTable(CU));
  DwarfTypeUnit &NewTU = *OwnedUnit;
  DIE &UnitDie = NewTU.getUnitDie();
  TU = &NewTU;
  TypeUnitsUnderConstruction.push_back(
      std::make_pair(std::move(OwnedUnit), CTy));

  NewTU.addUInt(UnitDie, dwarf::DW_AT_language, dwarf::DW_FORM_data2,
                CU.getLanguage());

  uint64_t Signature = makeTypeSignature(Identifier);
  NewTU.setTypeSignature(Signature);

  if (useSplitDwarf())
    NewTU.initSection(Asm->getObjFileLowering().getDwarfTypesDWOSection());
  else {
    CU.applyStmtList(UnitDie);
    NewTU.initSection(
        Asm->getObjFileLowering().getDwarfTypesSection(Signature));
  }

  NewTU.setType(NewTU.createTypeDIE(CTy));

  if (TopLevelType) {
    auto TypeUnitsToAdd = std::move(TypeUnitsUnderConstruction);
    TypeUnitsUnderConstruction.clear();

    // Types referencing entries in the address table cannot be placed in type
    // units.
    if (AddrPool.hasBeenUsed()) {

      // Remove all the types built while building this type.
      // This is pessimistic as some of these types might not be dependent on
      // the type that used an address.
      for (const auto &TU : TypeUnitsToAdd)
        DwarfTypeUnits.erase(TU.second);

      // Construct this type in the CU directly.
      // This is inefficient because all the dependent types will be rebuilt
      // from scratch, including building them in type units, discovering that
      // they depend on addresses, throwing them out and rebuilding them.
      CU.constructTypeDIE(RefDie, cast<DICompositeType>(CTy));
      return;
    }

    // If the type wasn't dependent on fission addresses, finish adding the type
    // and all its dependent types.
    for (auto &TU : TypeUnitsToAdd)
      InfoHolder.addUnit(std::move(TU.first));
  }
  CU.addDIETypeSignature(RefDie, NewTU);
}

// Accelerator table mutators - add each name along with its companion
// DIE to the proper table while ensuring that the name that we're going
// to reference is in the string table. We do this since the names we
// add may not only be identical to the names in the DIE.
void DwarfDebug::addAccelName(StringRef Name, const DIE &Die) {
  if (!useDwarfAccelTables())
    return;
  AccelNames.AddName(InfoHolder.getStringPool().getEntry(*Asm, Name), &Die);
}

void DwarfDebug::addAccelObjC(StringRef Name, const DIE &Die) {
  if (!useDwarfAccelTables())
    return;
  AccelObjC.AddName(InfoHolder.getStringPool().getEntry(*Asm, Name), &Die);
}

void DwarfDebug::addAccelNamespace(StringRef Name, const DIE &Die) {
  if (!useDwarfAccelTables())
    return;
  AccelNamespace.AddName(InfoHolder.getStringPool().getEntry(*Asm, Name), &Die);
}

void DwarfDebug::addAccelType(StringRef Name, const DIE &Die, char Flags) {
  if (!useDwarfAccelTables())
    return;
  AccelTypes.AddName(InfoHolder.getStringPool().getEntry(*Asm, Name), &Die);
}
