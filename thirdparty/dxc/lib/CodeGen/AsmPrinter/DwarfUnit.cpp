//===-- llvm/CodeGen/DwarfUnit.cpp - Dwarf Type and Compile Units ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for constructing a dwarf compile unit.
//
//===----------------------------------------------------------------------===//

#include "DwarfUnit.h"
#include "DwarfAccelTable.h"
#include "DwarfCompileUnit.h"
#include "DwarfDebug.h"
#include "DwarfExpression.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

#define DEBUG_TYPE "dwarfdebug"

static cl::opt<bool>
GenerateDwarfTypeUnits("generate-type-units", cl::Hidden,
                       cl::desc("Generate DWARF4 type units."),
                       cl::init(false));

DIEDwarfExpression::DIEDwarfExpression(const AsmPrinter &AP, DwarfUnit &DU,
                                       DIELoc &DIE)
    : DwarfExpression(*AP.MF->getSubtarget().getRegisterInfo(),
                      AP.getDwarfDebug()->getDwarfVersion()),
      AP(AP), DU(DU), DIE(DIE) {}

void DIEDwarfExpression::EmitOp(uint8_t Op, const char* Comment) {
  DU.addUInt(DIE, dwarf::DW_FORM_data1, Op);
}
void DIEDwarfExpression::EmitSigned(int64_t Value) {
  DU.addSInt(DIE, dwarf::DW_FORM_sdata, Value);
}
void DIEDwarfExpression::EmitUnsigned(uint64_t Value) {
  DU.addUInt(DIE, dwarf::DW_FORM_udata, Value);
}
bool DIEDwarfExpression::isFrameRegister(unsigned MachineReg) {
  return MachineReg == TRI.getFrameRegister(*AP.MF);
}

DwarfUnit::DwarfUnit(unsigned UID, dwarf::Tag UnitTag,
                     const DICompileUnit *Node, AsmPrinter *A, DwarfDebug *DW,
                     DwarfFile *DWU)
    : UniqueID(UID), CUNode(Node),
      UnitDie(*DIE::get(DIEValueAllocator, UnitTag)), DebugInfoOffset(0),
      Asm(A), DD(DW), DU(DWU), IndexTyDie(nullptr), Section(nullptr) {
  assert(UnitTag == dwarf::DW_TAG_compile_unit ||
         UnitTag == dwarf::DW_TAG_type_unit);
}

DwarfTypeUnit::DwarfTypeUnit(unsigned UID, DwarfCompileUnit &CU, AsmPrinter *A,
                             DwarfDebug *DW, DwarfFile *DWU,
                             MCDwarfDwoLineTable *SplitLineTable)
    : DwarfUnit(UID, dwarf::DW_TAG_type_unit, CU.getCUNode(), A, DW, DWU),
      CU(CU), SplitLineTable(SplitLineTable) {
  if (SplitLineTable)
    addSectionOffset(UnitDie, dwarf::DW_AT_stmt_list, 0);
}

DwarfUnit::~DwarfUnit() {
  for (unsigned j = 0, M = DIEBlocks.size(); j < M; ++j)
    DIEBlocks[j]->~DIEBlock();
  for (unsigned j = 0, M = DIELocs.size(); j < M; ++j)
    DIELocs[j]->~DIELoc();
}

int64_t DwarfUnit::getDefaultLowerBound() const {
  switch (getLanguage()) {
  default:
    break;

  case dwarf::DW_LANG_C89:
  case dwarf::DW_LANG_C99:
  case dwarf::DW_LANG_C:
  case dwarf::DW_LANG_C_plus_plus:
  case dwarf::DW_LANG_ObjC:
  case dwarf::DW_LANG_ObjC_plus_plus:
    return 0;

  case dwarf::DW_LANG_Fortran77:
  case dwarf::DW_LANG_Fortran90:
  case dwarf::DW_LANG_Fortran95:
    return 1;

  // The languages below have valid values only if the DWARF version >= 4.
  case dwarf::DW_LANG_Java:
  case dwarf::DW_LANG_Python:
  case dwarf::DW_LANG_UPC:
  case dwarf::DW_LANG_D:
    if (dwarf::DWARF_VERSION >= 4)
      return 0;
    break;

  case dwarf::DW_LANG_Ada83:
  case dwarf::DW_LANG_Ada95:
  case dwarf::DW_LANG_Cobol74:
  case dwarf::DW_LANG_Cobol85:
  case dwarf::DW_LANG_Modula2:
  case dwarf::DW_LANG_Pascal83:
  case dwarf::DW_LANG_PLI:
    if (dwarf::DWARF_VERSION >= 4)
      return 1;
    break;

  // The languages below have valid values only if the DWARF version >= 5.
  case dwarf::DW_LANG_OpenCL:
  case dwarf::DW_LANG_Go:
  case dwarf::DW_LANG_Haskell:
  case dwarf::DW_LANG_C_plus_plus_03:
  case dwarf::DW_LANG_C_plus_plus_11:
  case dwarf::DW_LANG_OCaml:
  case dwarf::DW_LANG_Rust:
  case dwarf::DW_LANG_C11:
  case dwarf::DW_LANG_Swift:
  case dwarf::DW_LANG_Dylan:
  case dwarf::DW_LANG_C_plus_plus_14:
    if (dwarf::DWARF_VERSION >= 5)
      return 0;
    break;

  case dwarf::DW_LANG_Modula3:
  case dwarf::DW_LANG_Julia:
  case dwarf::DW_LANG_Fortran03:
  case dwarf::DW_LANG_Fortran08:
    if (dwarf::DWARF_VERSION >= 5)
      return 1;
    break;
  }

  return -1;
}

/// Check whether the DIE for this MDNode can be shared across CUs.
static bool isShareableAcrossCUs(const DINode *D) {
  // When the MDNode can be part of the type system, the DIE can be shared
  // across CUs.
  // Combining type units and cross-CU DIE sharing is lower value (since
  // cross-CU DIE sharing is used in LTO and removes type redundancy at that
  // level already) but may be implementable for some value in projects
  // building multiple independent libraries with LTO and then linking those
  // together.
  return (isa<DIType>(D) ||
          (isa<DISubprogram>(D) && !cast<DISubprogram>(D)->isDefinition())) &&
         !GenerateDwarfTypeUnits;
}

DIE *DwarfUnit::getDIE(const DINode *D) const {
  if (isShareableAcrossCUs(D))
    return DU->getDIE(D);
  return MDNodeToDieMap.lookup(D);
}

void DwarfUnit::insertDIE(const DINode *Desc, DIE *D) {
  if (isShareableAcrossCUs(Desc)) {
    DU->insertDIE(Desc, D);
    return;
  }
  MDNodeToDieMap.insert(std::make_pair(Desc, D));
}

void DwarfUnit::addFlag(DIE &Die, dwarf::Attribute Attribute) {
  if (DD->getDwarfVersion() >= 4)
    Die.addValue(DIEValueAllocator, Attribute, dwarf::DW_FORM_flag_present,
                 DIEInteger(1));
  else
    Die.addValue(DIEValueAllocator, Attribute, dwarf::DW_FORM_flag,
                 DIEInteger(1));
}

void DwarfUnit::addUInt(DIE &Die, dwarf::Attribute Attribute,
                        Optional<dwarf::Form> Form, uint64_t Integer) {
  if (!Form)
    Form = DIEInteger::BestForm(false, Integer);
  Die.addValue(DIEValueAllocator, Attribute, *Form, DIEInteger(Integer));
}

void DwarfUnit::addUInt(DIE &Block, dwarf::Form Form, uint64_t Integer) {
  addUInt(Block, (dwarf::Attribute)0, Form, Integer);
}

void DwarfUnit::addSInt(DIE &Die, dwarf::Attribute Attribute,
                        Optional<dwarf::Form> Form, int64_t Integer) {
  if (!Form)
    Form = DIEInteger::BestForm(true, Integer);
  Die.addValue(DIEValueAllocator, Attribute, *Form, DIEInteger(Integer));
}

void DwarfUnit::addSInt(DIELoc &Die, Optional<dwarf::Form> Form,
                        int64_t Integer) {
  addSInt(Die, (dwarf::Attribute)0, Form, Integer);
}

void DwarfUnit::addString(DIE &Die, dwarf::Attribute Attribute,
                          StringRef String) {
  Die.addValue(DIEValueAllocator, Attribute,
               isDwoUnit() ? dwarf::DW_FORM_GNU_str_index : dwarf::DW_FORM_strp,
               DIEString(DU->getStringPool().getEntry(*Asm, String)));
}

DIE::value_iterator DwarfUnit::addLabel(DIE &Die, dwarf::Attribute Attribute,
                                        dwarf::Form Form,
                                        const MCSymbol *Label) {
  return Die.addValue(DIEValueAllocator, Attribute, Form, DIELabel(Label));
}

void DwarfUnit::addLabel(DIELoc &Die, dwarf::Form Form, const MCSymbol *Label) {
  addLabel(Die, (dwarf::Attribute)0, Form, Label);
}

void DwarfUnit::addSectionOffset(DIE &Die, dwarf::Attribute Attribute,
                                 uint64_t Integer) {
  if (DD->getDwarfVersion() >= 4)
    addUInt(Die, Attribute, dwarf::DW_FORM_sec_offset, Integer);
  else
    addUInt(Die, Attribute, dwarf::DW_FORM_data4, Integer);
}

unsigned DwarfTypeUnit::getOrCreateSourceID(StringRef FileName, StringRef DirName) {
  return SplitLineTable ? SplitLineTable->getFile(DirName, FileName)
                        : getCU().getOrCreateSourceID(FileName, DirName);
}

void DwarfUnit::addOpAddress(DIELoc &Die, const MCSymbol *Sym) {
  if (!DD->useSplitDwarf()) {
    addUInt(Die, dwarf::DW_FORM_data1, dwarf::DW_OP_addr);
    addLabel(Die, dwarf::DW_FORM_udata, Sym);
  } else {
    addUInt(Die, dwarf::DW_FORM_data1, dwarf::DW_OP_GNU_addr_index);
    addUInt(Die, dwarf::DW_FORM_GNU_addr_index,
            DD->getAddressPool().getIndex(Sym));
  }
}

void DwarfUnit::addLabelDelta(DIE &Die, dwarf::Attribute Attribute,
                              const MCSymbol *Hi, const MCSymbol *Lo) {
  Die.addValue(DIEValueAllocator, Attribute, dwarf::DW_FORM_data4,
               new (DIEValueAllocator) DIEDelta(Hi, Lo));
}

void DwarfUnit::addDIEEntry(DIE &Die, dwarf::Attribute Attribute, DIE &Entry) {
  addDIEEntry(Die, Attribute, DIEEntry(Entry));
}

void DwarfUnit::addDIETypeSignature(DIE &Die, const DwarfTypeUnit &Type) {
  // Flag the type unit reference as a declaration so that if it contains
  // members (implicit special members, static data member definitions, member
  // declarations for definitions in this CU, etc) consumers don't get confused
  // and think this is a full definition.
  addFlag(Die, dwarf::DW_AT_declaration);

  Die.addValue(DIEValueAllocator, dwarf::DW_AT_signature,
               dwarf::DW_FORM_ref_sig8, DIETypeSignature(Type));
}

void DwarfUnit::addDIEEntry(DIE &Die, dwarf::Attribute Attribute,
                            DIEEntry Entry) {
  const DIE *DieCU = Die.getUnitOrNull();
  const DIE *EntryCU = Entry.getEntry().getUnitOrNull();
  if (!DieCU)
    // We assume that Die belongs to this CU, if it is not linked to any CU yet.
    DieCU = &getUnitDie();
  if (!EntryCU)
    EntryCU = &getUnitDie();
  Die.addValue(DIEValueAllocator, Attribute,
               EntryCU == DieCU ? dwarf::DW_FORM_ref4 : dwarf::DW_FORM_ref_addr,
               Entry);
}

DIE &DwarfUnit::createAndAddDIE(unsigned Tag, DIE &Parent, const DINode *N) {
  assert(Tag != dwarf::DW_TAG_auto_variable &&
         Tag != dwarf::DW_TAG_arg_variable);
  DIE &Die = Parent.addChild(DIE::get(DIEValueAllocator, (dwarf::Tag)Tag));
  if (N)
    insertDIE(N, &Die);
  return Die;
}

void DwarfUnit::addBlock(DIE &Die, dwarf::Attribute Attribute, DIELoc *Loc) {
  Loc->ComputeSize(Asm);
  DIELocs.push_back(Loc); // Memoize so we can call the destructor later on.
  Die.addValue(DIEValueAllocator, Attribute,
               Loc->BestForm(DD->getDwarfVersion()), Loc);
}

void DwarfUnit::addBlock(DIE &Die, dwarf::Attribute Attribute,
                         DIEBlock *Block) {
  Block->ComputeSize(Asm);
  DIEBlocks.push_back(Block); // Memoize so we can call the destructor later on.
  Die.addValue(DIEValueAllocator, Attribute, Block->BestForm(), Block);
}

void DwarfUnit::addSourceLine(DIE &Die, unsigned Line, StringRef File,
                              StringRef Directory) {
  if (Line == 0)
    return;

  unsigned FileID = getOrCreateSourceID(File, Directory);
  assert(FileID && "Invalid file id");
  addUInt(Die, dwarf::DW_AT_decl_file, None, FileID);
  addUInt(Die, dwarf::DW_AT_decl_line, None, Line);
}

void DwarfUnit::addSourceLine(DIE &Die, const DILocalVariable *V) {
  assert(V);

  addSourceLine(Die, V->getLine(), V->getScope()->getFilename(),
                V->getScope()->getDirectory());
}

void DwarfUnit::addSourceLine(DIE &Die, const DIGlobalVariable *G) {
  assert(G);

  addSourceLine(Die, G->getLine(), G->getFilename(), G->getDirectory());
}

void DwarfUnit::addSourceLine(DIE &Die, const DISubprogram *SP) {
  assert(SP);

  addSourceLine(Die, SP->getLine(), SP->getFilename(), SP->getDirectory());
}

void DwarfUnit::addSourceLine(DIE &Die, const DIType *Ty) {
  assert(Ty);

  addSourceLine(Die, Ty->getLine(), Ty->getFilename(), Ty->getDirectory());
}

void DwarfUnit::addSourceLine(DIE &Die, const DIObjCProperty *Ty) {
  assert(Ty);

  addSourceLine(Die, Ty->getLine(), Ty->getFilename(), Ty->getDirectory());
}

void DwarfUnit::addSourceLine(DIE &Die, const DINamespace *NS) {
  addSourceLine(Die, NS->getLine(), NS->getFilename(), NS->getDirectory());
}

bool DwarfUnit::addRegisterOpPiece(DIELoc &TheDie, unsigned Reg,
                                   unsigned SizeInBits, unsigned OffsetInBits) {
  DIEDwarfExpression Expr(*Asm, *this, TheDie);
  Expr.AddMachineRegPiece(Reg, SizeInBits, OffsetInBits);
  return true;
}

bool DwarfUnit::addRegisterOffset(DIELoc &TheDie, unsigned Reg,
                                  int64_t Offset) {
  DIEDwarfExpression Expr(*Asm, *this, TheDie);
  return Expr.AddMachineRegIndirect(Reg, Offset);
}

/* Byref variables, in Blocks, are declared by the programmer as "SomeType
   VarName;", but the compiler creates a __Block_byref_x_VarName struct, and
   gives the variable VarName either the struct, or a pointer to the struct, as
   its type.  This is necessary for various behind-the-scenes things the
   compiler needs to do with by-reference variables in Blocks.

   However, as far as the original *programmer* is concerned, the variable
   should still have type 'SomeType', as originally declared.

   The function getBlockByrefType dives into the __Block_byref_x_VarName
   struct to find the original type of the variable, which is then assigned to
   the variable's Debug Information Entry as its real type.  So far, so good.
   However now the debugger will expect the variable VarName to have the type
   SomeType.  So we need the location attribute for the variable to be an
   expression that explains to the debugger how to navigate through the
   pointers and struct to find the actual variable of type SomeType.

   The following function does just that.  We start by getting
   the "normal" location for the variable. This will be the location
   of either the struct __Block_byref_x_VarName or the pointer to the
   struct __Block_byref_x_VarName.

   The struct will look something like:

   struct __Block_byref_x_VarName {
     ... <various fields>
     struct __Block_byref_x_VarName *forwarding;
     ... <various other fields>
     SomeType VarName;
     ... <maybe more fields>
   };

   If we are given the struct directly (as our starting point) we
   need to tell the debugger to:

   1).  Add the offset of the forwarding field.

   2).  Follow that pointer to get the real __Block_byref_x_VarName
   struct to use (the real one may have been copied onto the heap).

   3).  Add the offset for the field VarName, to find the actual variable.

   If we started with a pointer to the struct, then we need to
   dereference that pointer first, before the other steps.
   Translating this into DWARF ops, we will need to append the following
   to the current location description for the variable:

   DW_OP_deref                    -- optional, if we start with a pointer
   DW_OP_plus_uconst <forward_fld_offset>
   DW_OP_deref
   DW_OP_plus_uconst <varName_fld_offset>

   That is what this function does.  */

void DwarfUnit::addBlockByrefAddress(const DbgVariable &DV, DIE &Die,
                                     dwarf::Attribute Attribute,
                                     const MachineLocation &Location) {
  const DIType *Ty = DV.getType();
  const DIType *TmpTy = Ty;
  uint16_t Tag = Ty->getTag();
  bool isPointer = false;

  StringRef varName = DV.getName();

  if (Tag == dwarf::DW_TAG_pointer_type) {
    auto *DTy = cast<DIDerivedType>(Ty);
    TmpTy = resolve(DTy->getBaseType());
    isPointer = true;
  }

  // Find the __forwarding field and the variable field in the __Block_byref
  // struct.
  DINodeArray Fields = cast<DICompositeTypeBase>(TmpTy)->getElements();
  const DIDerivedType *varField = nullptr;
  const DIDerivedType *forwardingField = nullptr;

  for (unsigned i = 0, N = Fields.size(); i < N; ++i) {
    auto *DT = cast<DIDerivedType>(Fields[i]);
    StringRef fieldName = DT->getName();
    if (fieldName == "__forwarding")
      forwardingField = DT;
    else if (fieldName == varName)
      varField = DT;
  }

  // Get the offsets for the forwarding field and the variable field.
  unsigned forwardingFieldOffset = forwardingField->getOffsetInBits() >> 3;
  unsigned varFieldOffset = varField->getOffsetInBits() >> 2;

  // Decode the original location, and use that as the start of the byref
  // variable's location.
  DIELoc *Loc = new (DIEValueAllocator) DIELoc;

  bool validReg;
  if (Location.isReg())
    validReg = addRegisterOpPiece(*Loc, Location.getReg());
  else
    validReg = addRegisterOffset(*Loc, Location.getReg(), Location.getOffset());

  if (!validReg)
    return;

  // If we started with a pointer to the __Block_byref... struct, then
  // the first thing we need to do is dereference the pointer (DW_OP_deref).
  if (isPointer)
    addUInt(*Loc, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);

  // Next add the offset for the '__forwarding' field:
  // DW_OP_plus_uconst ForwardingFieldOffset.  Note there's no point in
  // adding the offset if it's 0.
  if (forwardingFieldOffset > 0) {
    addUInt(*Loc, dwarf::DW_FORM_data1, dwarf::DW_OP_plus_uconst);
    addUInt(*Loc, dwarf::DW_FORM_udata, forwardingFieldOffset);
  }

  // Now dereference the __forwarding field to get to the real __Block_byref
  // struct:  DW_OP_deref.
  addUInt(*Loc, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);

  // Now that we've got the real __Block_byref... struct, add the offset
  // for the variable's field to get to the location of the actual variable:
  // DW_OP_plus_uconst varFieldOffset.  Again, don't add if it's 0.
  if (varFieldOffset > 0) {
    addUInt(*Loc, dwarf::DW_FORM_data1, dwarf::DW_OP_plus_uconst);
    addUInt(*Loc, dwarf::DW_FORM_udata, varFieldOffset);
  }

  // Now attach the location information to the DIE.
  addBlock(Die, Attribute, Loc);
}

/// Return true if type encoding is unsigned.
static bool isUnsignedDIType(DwarfDebug *DD, const DIType *Ty) {
  if (auto *DTy = dyn_cast<DIDerivedTypeBase>(Ty)) {
    dwarf::Tag T = (dwarf::Tag)Ty->getTag();
    // Encode pointer constants as unsigned bytes. This is used at least for
    // null pointer constant emission.
    // (Pieces of) aggregate types that get hacked apart by SROA may also be
    // represented by a constant. Encode them as unsigned bytes.
    // FIXME: reference and rvalue_reference /probably/ shouldn't be allowed
    // here, but accept them for now due to a bug in SROA producing bogus
    // dbg.values.
    if (T == dwarf::DW_TAG_array_type ||
        T == dwarf::DW_TAG_class_type ||
        T == dwarf::DW_TAG_pointer_type ||
        T == dwarf::DW_TAG_ptr_to_member_type ||
        T == dwarf::DW_TAG_reference_type ||
        T == dwarf::DW_TAG_rvalue_reference_type ||
        T == dwarf::DW_TAG_structure_type ||
        T == dwarf::DW_TAG_union_type)
      return true;
    assert(T == dwarf::DW_TAG_typedef || T == dwarf::DW_TAG_const_type ||
           T == dwarf::DW_TAG_volatile_type ||
           T == dwarf::DW_TAG_restrict_type ||
           T == dwarf::DW_TAG_enumeration_type);
    if (DITypeRef Deriv = DTy->getBaseType())
      return isUnsignedDIType(DD, DD->resolve(Deriv));
    // FIXME: Enums without a fixed underlying type have unknown signedness
    // here, leading to incorrectly emitted constants.
    assert(DTy->getTag() == dwarf::DW_TAG_enumeration_type);
    return false;
  }

  auto *BTy = cast<DIBasicType>(Ty);
  unsigned Encoding = BTy->getEncoding();
  assert((Encoding == dwarf::DW_ATE_unsigned ||
          Encoding == dwarf::DW_ATE_unsigned_char ||
          Encoding == dwarf::DW_ATE_signed ||
          Encoding == dwarf::DW_ATE_signed_char ||
          Encoding == dwarf::DW_ATE_float || Encoding == dwarf::DW_ATE_UTF ||
          Encoding == dwarf::DW_ATE_boolean ||
          (Ty->getTag() == dwarf::DW_TAG_unspecified_type &&
           Ty->getName() == "decltype(nullptr)")) &&
         "Unsupported encoding");
  return Encoding == dwarf::DW_ATE_unsigned ||
         Encoding == dwarf::DW_ATE_unsigned_char ||
         Encoding == dwarf::DW_ATE_UTF || Encoding == dwarf::DW_ATE_boolean ||
         Ty->getTag() == dwarf::DW_TAG_unspecified_type;
}

/// If this type is derived from a base type then return base type size.
static uint64_t getBaseTypeSize(DwarfDebug *DD, const DIDerivedType *Ty) {
  unsigned Tag = Ty->getTag();

  if (Tag != dwarf::DW_TAG_member && Tag != dwarf::DW_TAG_typedef &&
      Tag != dwarf::DW_TAG_const_type && Tag != dwarf::DW_TAG_volatile_type &&
      Tag != dwarf::DW_TAG_restrict_type)
    return Ty->getSizeInBits();

  auto *BaseType = DD->resolve(Ty->getBaseType());

  assert(BaseType && "Unexpected invalid base type");

  // If this is a derived type, go ahead and get the base type, unless it's a
  // reference then it's just the size of the field. Pointer types have no need
  // of this since they're a different type of qualification on the type.
  if (BaseType->getTag() == dwarf::DW_TAG_reference_type ||
      BaseType->getTag() == dwarf::DW_TAG_rvalue_reference_type)
    return Ty->getSizeInBits();

  if (auto *DT = dyn_cast<DIDerivedType>(BaseType))
    return getBaseTypeSize(DD, DT);

  return BaseType->getSizeInBits();
}

void DwarfUnit::addConstantFPValue(DIE &Die, const MachineOperand &MO) {
  assert(MO.isFPImm() && "Invalid machine operand!");
  DIEBlock *Block = new (DIEValueAllocator) DIEBlock;
  APFloat FPImm = MO.getFPImm()->getValueAPF();

  // Get the raw data form of the floating point.
  const APInt FltVal = FPImm.bitcastToAPInt();
  const char *FltPtr = (const char *)FltVal.getRawData();

  int NumBytes = FltVal.getBitWidth() / 8; // 8 bits per byte.
  bool LittleEndian = Asm->getDataLayout().isLittleEndian();
  int Incr = (LittleEndian ? 1 : -1);
  int Start = (LittleEndian ? 0 : NumBytes - 1);
  int Stop = (LittleEndian ? NumBytes : -1);

  // Output the constant to DWARF one byte at a time.
  for (; Start != Stop; Start += Incr)
    addUInt(*Block, dwarf::DW_FORM_data1, (unsigned char)0xFF & FltPtr[Start]);

  addBlock(Die, dwarf::DW_AT_const_value, Block);
}

void DwarfUnit::addConstantFPValue(DIE &Die, const ConstantFP *CFP) {
  // Pass this down to addConstantValue as an unsigned bag of bits.
  addConstantValue(Die, CFP->getValueAPF().bitcastToAPInt(), true);
}

void DwarfUnit::addConstantValue(DIE &Die, const ConstantInt *CI,
                                 const DIType *Ty) {
  addConstantValue(Die, CI->getValue(), Ty);
}

void DwarfUnit::addConstantValue(DIE &Die, const MachineOperand &MO,
                                 const DIType *Ty) {
  assert(MO.isImm() && "Invalid machine operand!");

  addConstantValue(Die, isUnsignedDIType(DD, Ty), MO.getImm());
}

void DwarfUnit::addConstantValue(DIE &Die, bool Unsigned, uint64_t Val) {
  // FIXME: This is a bit conservative/simple - it emits negative values always
  // sign extended to 64 bits rather than minimizing the number of bytes.
  addUInt(Die, dwarf::DW_AT_const_value,
          Unsigned ? dwarf::DW_FORM_udata : dwarf::DW_FORM_sdata, Val);
}

void DwarfUnit::addConstantValue(DIE &Die, const APInt &Val, const DIType *Ty) {
  addConstantValue(Die, Val, isUnsignedDIType(DD, Ty));
}

void DwarfUnit::addConstantValue(DIE &Die, const APInt &Val, bool Unsigned) {
  unsigned CIBitWidth = Val.getBitWidth();
  if (CIBitWidth <= 64) {
    addConstantValue(Die, Unsigned,
                     Unsigned ? Val.getZExtValue() : Val.getSExtValue());
    return;
  }

  DIEBlock *Block = new (DIEValueAllocator) DIEBlock;

  // Get the raw data form of the large APInt.
  const uint64_t *Ptr64 = Val.getRawData();

  int NumBytes = Val.getBitWidth() / 8; // 8 bits per byte.
  bool LittleEndian = Asm->getDataLayout().isLittleEndian();

  // Output the constant to DWARF one byte at a time.
  for (int i = 0; i < NumBytes; i++) {
    uint8_t c;
    if (LittleEndian)
      c = Ptr64[i / 8] >> (8 * (i & 7));
    else
      c = Ptr64[(NumBytes - 1 - i) / 8] >> (8 * ((NumBytes - 1 - i) & 7));
    addUInt(*Block, dwarf::DW_FORM_data1, c);
  }

  addBlock(Die, dwarf::DW_AT_const_value, Block);
}

void DwarfUnit::addLinkageName(DIE &Die, StringRef LinkageName) {
  if (!LinkageName.empty())
    addString(Die,
              DD->getDwarfVersion() >= 4 ? dwarf::DW_AT_linkage_name
                                         : dwarf::DW_AT_MIPS_linkage_name,
              GlobalValue::getRealLinkageName(LinkageName));
}

void DwarfUnit::addTemplateParams(DIE &Buffer, DINodeArray TParams) {
  // Add template parameters.
  for (const auto *Element : TParams) {
    if (auto *TTP = dyn_cast<DITemplateTypeParameter>(Element))
      constructTemplateTypeParameterDIE(Buffer, TTP);
    else if (auto *TVP = dyn_cast<DITemplateValueParameter>(Element))
      constructTemplateValueParameterDIE(Buffer, TVP);
  }
}

DIE *DwarfUnit::getOrCreateContextDIE(const DIScope *Context) {
  if (!Context || isa<DIFile>(Context))
    return &getUnitDie();
  if (auto *T = dyn_cast<DIType>(Context))
    return getOrCreateTypeDIE(T);
  if (auto *NS = dyn_cast<DINamespace>(Context))
    return getOrCreateNameSpace(NS);
  if (auto *SP = dyn_cast<DISubprogram>(Context))
    return getOrCreateSubprogramDIE(SP);
  return getDIE(Context);
}

DIE *DwarfUnit::createTypeDIE(const DICompositeType *Ty) {
  auto *Context = resolve(Ty->getScope());
  DIE *ContextDIE = getOrCreateContextDIE(Context);

  if (DIE *TyDIE = getDIE(Ty))
    return TyDIE;

  // Create new type.
  DIE &TyDIE = createAndAddDIE(Ty->getTag(), *ContextDIE, Ty);

  constructTypeDIE(TyDIE, cast<DICompositeType>(Ty));

  updateAcceleratorTables(Context, Ty, TyDIE);
  return &TyDIE;
}

DIE *DwarfUnit::getOrCreateTypeDIE(const MDNode *TyNode) {
  if (!TyNode)
    return nullptr;

  auto *Ty = cast<DIType>(TyNode);
  assert(Ty == resolve(Ty->getRef()) &&
         "type was not uniqued, possible ODR violation.");

  // DW_TAG_restrict_type is not supported in DWARF2
  if (Ty->getTag() == dwarf::DW_TAG_restrict_type && DD->getDwarfVersion() <= 2)
    return getOrCreateTypeDIE(resolve(cast<DIDerivedType>(Ty)->getBaseType()));

  // Construct the context before querying for the existence of the DIE in case
  // such construction creates the DIE.
  auto *Context = resolve(Ty->getScope());
  DIE *ContextDIE = getOrCreateContextDIE(Context);
  assert(ContextDIE);

  if (DIE *TyDIE = getDIE(Ty))
    return TyDIE;

  // Create new type.
  DIE &TyDIE = createAndAddDIE(Ty->getTag(), *ContextDIE, Ty);

  updateAcceleratorTables(Context, Ty, TyDIE);

  if (auto *BT = dyn_cast<DIBasicType>(Ty))
    constructTypeDIE(TyDIE, BT);
  else if (auto *STy = dyn_cast<DISubroutineType>(Ty))
    constructTypeDIE(TyDIE, STy);
  else if (auto *CTy = dyn_cast<DICompositeType>(Ty)) {
    if (GenerateDwarfTypeUnits && !Ty->isForwardDecl())
      if (MDString *TypeId = CTy->getRawIdentifier()) {
        DD->addDwarfTypeUnitType(getCU(), TypeId->getString(), TyDIE, CTy);
        // Skip updating the accelerator tables since this is not the full type.
        return &TyDIE;
      }
    constructTypeDIE(TyDIE, CTy);
  } else {
    constructTypeDIE(TyDIE, cast<DIDerivedType>(Ty));
  }

  return &TyDIE;
}

void DwarfUnit::updateAcceleratorTables(const DIScope *Context,
                                        const DIType *Ty, const DIE &TyDIE) {
  if (!Ty->getName().empty() && !Ty->isForwardDecl()) {
    bool IsImplementation = 0;
    if (auto *CT = dyn_cast<DICompositeTypeBase>(Ty)) {
      // A runtime language of 0 actually means C/C++ and that any
      // non-negative value is some version of Objective-C/C++.
      IsImplementation = CT->getRuntimeLang() == 0 || CT->isObjcClassComplete();
    }
    unsigned Flags = IsImplementation ? dwarf::DW_FLAG_type_implementation : 0;
    DD->addAccelType(Ty->getName(), TyDIE, Flags);

    if (!Context || isa<DICompileUnit>(Context) || isa<DIFile>(Context) ||
        isa<DINamespace>(Context))
      addGlobalType(Ty, TyDIE, Context);
  }
}

void DwarfUnit::addType(DIE &Entity, const DIType *Ty,
                        dwarf::Attribute Attribute) {
  assert(Ty && "Trying to add a type that doesn't exist?");
  addDIEEntry(Entity, Attribute, DIEEntry(*getOrCreateTypeDIE(Ty)));
}

std::string DwarfUnit::getParentContextString(const DIScope *Context) const {
  if (!Context)
    return "";

  // FIXME: Decide whether to implement this for non-C++ languages.
  if (getLanguage() != dwarf::DW_LANG_C_plus_plus)
    return "";

  std::string CS;
  SmallVector<const DIScope *, 1> Parents;
  while (!isa<DICompileUnit>(Context)) {
    Parents.push_back(Context);
    if (Context->getScope())
      Context = resolve(Context->getScope());
    else
      // Structure, etc types will have a NULL context if they're at the top
      // level.
      break;
  }

  // Reverse iterate over our list to go from the outermost construct to the
  // innermost.
  for (auto I = Parents.rbegin(), E = Parents.rend(); I != E; ++I) {
    const DIScope *Ctx = *I;
    StringRef Name = Ctx->getName();
    if (Name.empty() && isa<DINamespace>(Ctx))
      Name = "(anonymous namespace)";
    if (!Name.empty()) {
      CS += Name;
      CS += "::";
    }
  }
  return CS;
}

void DwarfUnit::constructTypeDIE(DIE &Buffer, const DIBasicType *BTy) {
  // Get core information.
  StringRef Name = BTy->getName();
  // Add name if not anonymous or intermediate type.
  if (!Name.empty())
    addString(Buffer, dwarf::DW_AT_name, Name);

  // An unspecified type only has a name attribute.
  if (BTy->getTag() == dwarf::DW_TAG_unspecified_type)
    return;

  addUInt(Buffer, dwarf::DW_AT_encoding, dwarf::DW_FORM_data1,
          BTy->getEncoding());

  uint64_t Size = BTy->getSizeInBits() >> 3;
  addUInt(Buffer, dwarf::DW_AT_byte_size, None, Size);
}

void DwarfUnit::constructTypeDIE(DIE &Buffer, const DIDerivedType *DTy) {
  // Get core information.
  StringRef Name = DTy->getName();
  uint64_t Size = DTy->getSizeInBits() >> 3;
  uint16_t Tag = Buffer.getTag();

  // Map to main type, void will not have a type.
  const DIType *FromTy = resolve(DTy->getBaseType());
  if (FromTy)
    addType(Buffer, FromTy);

  // Add name if not anonymous or intermediate type.
  if (!Name.empty())
    addString(Buffer, dwarf::DW_AT_name, Name);

  // Add size if non-zero (derived types might be zero-sized.)
  if (Size && Tag != dwarf::DW_TAG_pointer_type
           && Tag != dwarf::DW_TAG_ptr_to_member_type)
    addUInt(Buffer, dwarf::DW_AT_byte_size, None, Size);

  if (Tag == dwarf::DW_TAG_ptr_to_member_type)
    addDIEEntry(
        Buffer, dwarf::DW_AT_containing_type,
        *getOrCreateTypeDIE(resolve(cast<DIDerivedType>(DTy)->getClassType())));
  // Add source line info if available and TyDesc is not a forward declaration.
  if (!DTy->isForwardDecl())
    addSourceLine(Buffer, DTy);
}

void DwarfUnit::constructSubprogramArguments(DIE &Buffer, DITypeRefArray Args) {
  for (unsigned i = 1, N = Args.size(); i < N; ++i) {
    const DIType *Ty = resolve(Args[i]);
    if (!Ty) {
      assert(i == N-1 && "Unspecified parameter must be the last argument");
      createAndAddDIE(dwarf::DW_TAG_unspecified_parameters, Buffer);
    } else {
      DIE &Arg = createAndAddDIE(dwarf::DW_TAG_formal_parameter, Buffer);
      addType(Arg, Ty);
      if (Ty->isArtificial())
        addFlag(Arg, dwarf::DW_AT_artificial);
    }
  }
}

void DwarfUnit::constructTypeDIE(DIE &Buffer, const DISubroutineType *CTy) {
  // Add return type.  A void return won't have a type.
  auto Elements = cast<DISubroutineType>(CTy)->getTypeArray();
  if (Elements.size())
    if (auto RTy = resolve(Elements[0]))
      addType(Buffer, RTy);

  bool isPrototyped = true;
  if (Elements.size() == 2 && !Elements[1])
    isPrototyped = false;

  constructSubprogramArguments(Buffer, Elements);

  // Add prototype flag if we're dealing with a C language and the function has
  // been prototyped.
  uint16_t Language = getLanguage();
  if (isPrototyped &&
      (Language == dwarf::DW_LANG_C89 || Language == dwarf::DW_LANG_C99 ||
       Language == dwarf::DW_LANG_ObjC))
    addFlag(Buffer, dwarf::DW_AT_prototyped);

  if (CTy->isLValueReference())
    addFlag(Buffer, dwarf::DW_AT_reference);

  if (CTy->isRValueReference())
    addFlag(Buffer, dwarf::DW_AT_rvalue_reference);
}

void DwarfUnit::constructTypeDIE(DIE &Buffer, const DICompositeType *CTy) {
  // Add name if not anonymous or intermediate type.
  StringRef Name = CTy->getName();

  uint64_t Size = CTy->getSizeInBits() >> 3;
  uint16_t Tag = Buffer.getTag();

  switch (Tag) {
  case dwarf::DW_TAG_array_type:
    constructArrayTypeDIE(Buffer, CTy);
    break;
  case dwarf::DW_TAG_enumeration_type:
    constructEnumTypeDIE(Buffer, CTy);
    break;
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_union_type:
  case dwarf::DW_TAG_class_type: {
    // Add elements to structure type.
    DINodeArray Elements = CTy->getElements();
    for (const auto *Element : Elements) {
      if (!Element)
        continue;
      if (auto *SP = dyn_cast<DISubprogram>(Element))
        getOrCreateSubprogramDIE(SP);
      else if (auto *DDTy = dyn_cast<DIDerivedType>(Element)) {
        if (DDTy->getTag() == dwarf::DW_TAG_friend) {
          DIE &ElemDie = createAndAddDIE(dwarf::DW_TAG_friend, Buffer);
          addType(ElemDie, resolve(DDTy->getBaseType()), dwarf::DW_AT_friend);
        } else if (DDTy->isStaticMember()) {
          getOrCreateStaticMemberDIE(DDTy);
        } else {
          constructMemberDIE(Buffer, DDTy);
        }
      } else if (auto *Property = dyn_cast<DIObjCProperty>(Element)) {
        DIE &ElemDie = createAndAddDIE(Property->getTag(), Buffer);
        StringRef PropertyName = Property->getName();
        addString(ElemDie, dwarf::DW_AT_APPLE_property_name, PropertyName);
        if (Property->getType())
          addType(ElemDie, resolve(Property->getType()));
        addSourceLine(ElemDie, Property);
        StringRef GetterName = Property->getGetterName();
        if (!GetterName.empty())
          addString(ElemDie, dwarf::DW_AT_APPLE_property_getter, GetterName);
        StringRef SetterName = Property->getSetterName();
        if (!SetterName.empty())
          addString(ElemDie, dwarf::DW_AT_APPLE_property_setter, SetterName);
        if (unsigned PropertyAttributes = Property->getAttributes())
          addUInt(ElemDie, dwarf::DW_AT_APPLE_property_attribute, None,
                  PropertyAttributes);
      }
    }

    if (CTy->isAppleBlockExtension())
      addFlag(Buffer, dwarf::DW_AT_APPLE_block);

    // This is outside the DWARF spec, but GDB expects a DW_AT_containing_type
    // inside C++ composite types to point to the base class with the vtable.
    if (auto *ContainingType =
            dyn_cast_or_null<DICompositeType>(resolve(CTy->getVTableHolder())))
      addDIEEntry(Buffer, dwarf::DW_AT_containing_type,
                  *getOrCreateTypeDIE(ContainingType));

    if (CTy->isObjcClassComplete())
      addFlag(Buffer, dwarf::DW_AT_APPLE_objc_complete_type);

    // Add template parameters to a class, structure or union types.
    // FIXME: The support isn't in the metadata for this yet.
    if (Tag == dwarf::DW_TAG_class_type ||
        Tag == dwarf::DW_TAG_structure_type || Tag == dwarf::DW_TAG_union_type)
      addTemplateParams(Buffer, CTy->getTemplateParams());

    break;
  }
  default:
    break;
  }

  // Add name if not anonymous or intermediate type.
  if (!Name.empty())
    addString(Buffer, dwarf::DW_AT_name, Name);

  if (Tag == dwarf::DW_TAG_enumeration_type ||
      Tag == dwarf::DW_TAG_class_type || Tag == dwarf::DW_TAG_structure_type ||
      Tag == dwarf::DW_TAG_union_type) {
    // Add size if non-zero (derived types might be zero-sized.)
    // TODO: Do we care about size for enum forward declarations?
    if (Size)
      addUInt(Buffer, dwarf::DW_AT_byte_size, None, Size);
    else if (!CTy->isForwardDecl())
      // Add zero size if it is not a forward declaration.
      addUInt(Buffer, dwarf::DW_AT_byte_size, None, 0);

    // If we're a forward decl, say so.
    if (CTy->isForwardDecl())
      addFlag(Buffer, dwarf::DW_AT_declaration);

    // Add source line info if available.
    if (!CTy->isForwardDecl())
      addSourceLine(Buffer, CTy);

    // No harm in adding the runtime language to the declaration.
    unsigned RLang = CTy->getRuntimeLang();
    if (RLang)
      addUInt(Buffer, dwarf::DW_AT_APPLE_runtime_class, dwarf::DW_FORM_data1,
              RLang);
  }
}

void DwarfUnit::constructTemplateTypeParameterDIE(
    DIE &Buffer, const DITemplateTypeParameter *TP) {
  DIE &ParamDIE =
      createAndAddDIE(dwarf::DW_TAG_template_type_parameter, Buffer);
  // Add the type if it exists, it could be void and therefore no type.
  if (TP->getType())
    addType(ParamDIE, resolve(TP->getType()));
  if (!TP->getName().empty())
    addString(ParamDIE, dwarf::DW_AT_name, TP->getName());
}

void DwarfUnit::constructTemplateValueParameterDIE(
    DIE &Buffer, const DITemplateValueParameter *VP) {
  DIE &ParamDIE = createAndAddDIE(VP->getTag(), Buffer);

  // Add the type if there is one, template template and template parameter
  // packs will not have a type.
  if (VP->getTag() == dwarf::DW_TAG_template_value_parameter)
    addType(ParamDIE, resolve(VP->getType()));
  if (!VP->getName().empty())
    addString(ParamDIE, dwarf::DW_AT_name, VP->getName());
  if (Metadata *Val = VP->getValue()) {
    if (ConstantInt *CI = mdconst::dyn_extract<ConstantInt>(Val))
      addConstantValue(ParamDIE, CI, resolve(VP->getType()));
    else if (GlobalValue *GV = mdconst::dyn_extract<GlobalValue>(Val)) {
      // For declaration non-type template parameters (such as global values and
      // functions)
      DIELoc *Loc = new (DIEValueAllocator) DIELoc;
      addOpAddress(*Loc, Asm->getSymbol(GV));
      // Emit DW_OP_stack_value to use the address as the immediate value of the
      // parameter, rather than a pointer to it.
      addUInt(*Loc, dwarf::DW_FORM_data1, dwarf::DW_OP_stack_value);
      addBlock(ParamDIE, dwarf::DW_AT_location, Loc);
    } else if (VP->getTag() == dwarf::DW_TAG_GNU_template_template_param) {
      assert(isa<MDString>(Val));
      addString(ParamDIE, dwarf::DW_AT_GNU_template_name,
                cast<MDString>(Val)->getString());
    } else if (VP->getTag() == dwarf::DW_TAG_GNU_template_parameter_pack) {
      addTemplateParams(ParamDIE, cast<MDTuple>(Val));
    }
  }
}

DIE *DwarfUnit::getOrCreateNameSpace(const DINamespace *NS) {
  // Construct the context before querying for the existence of the DIE in case
  // such construction creates the DIE.
  DIE *ContextDIE = getOrCreateContextDIE(NS->getScope());

  if (DIE *NDie = getDIE(NS))
    return NDie;
  DIE &NDie = createAndAddDIE(dwarf::DW_TAG_namespace, *ContextDIE, NS);

  StringRef Name = NS->getName();
  if (!Name.empty())
    addString(NDie, dwarf::DW_AT_name, NS->getName());
  else
    Name = "(anonymous namespace)";
  DD->addAccelNamespace(Name, NDie);
  addGlobalName(Name, NDie, NS->getScope());
  addSourceLine(NDie, NS);
  return &NDie;
}

DIE *DwarfUnit::getOrCreateModule(const DIModule *M) {
  // Construct the context before querying for the existence of the DIE in case
  // such construction creates the DIE.
  DIE *ContextDIE = getOrCreateContextDIE(M->getScope());

  if (DIE *MDie = getDIE(M))
    return MDie;
  DIE &MDie = createAndAddDIE(dwarf::DW_TAG_module, *ContextDIE, M);

  if (!M->getName().empty()) {
    addString(MDie, dwarf::DW_AT_name, M->getName());
    addGlobalName(M->getName(), MDie, M->getScope());
  }
  if (!M->getConfigurationMacros().empty())
    addString(MDie, dwarf::DW_AT_LLVM_config_macros,
              M->getConfigurationMacros());
  if (!M->getIncludePath().empty())
    addString(MDie, dwarf::DW_AT_LLVM_include_path, M->getIncludePath());
  if (!M->getISysRoot().empty())
    addString(MDie, dwarf::DW_AT_LLVM_isysroot, M->getISysRoot());
  
  return &MDie;
}

DIE *DwarfUnit::getOrCreateSubprogramDIE(const DISubprogram *SP, bool Minimal) {
  // Construct the context before querying for the existence of the DIE in case
  // such construction creates the DIE (as is the case for member function
  // declarations).
  DIE *ContextDIE =
      Minimal ? &getUnitDie() : getOrCreateContextDIE(resolve(SP->getScope()));

  if (DIE *SPDie = getDIE(SP))
    return SPDie;

  if (auto *SPDecl = SP->getDeclaration()) {
    if (!Minimal) {
      // Add subprogram definitions to the CU die directly.
      ContextDIE = &getUnitDie();
      // Build the decl now to ensure it precedes the definition.
      getOrCreateSubprogramDIE(SPDecl);
    }
  }

  // DW_TAG_inlined_subroutine may refer to this DIE.
  DIE &SPDie = createAndAddDIE(dwarf::DW_TAG_subprogram, *ContextDIE, SP);

  // Stop here and fill this in later, depending on whether or not this
  // subprogram turns out to have inlined instances or not.
  if (SP->isDefinition())
    return &SPDie;

  applySubprogramAttributes(SP, SPDie);
  return &SPDie;
}

bool DwarfUnit::applySubprogramDefinitionAttributes(const DISubprogram *SP,
                                                    DIE &SPDie) {
  DIE *DeclDie = nullptr;
  StringRef DeclLinkageName;
  if (auto *SPDecl = SP->getDeclaration()) {
    DeclDie = getDIE(SPDecl);
    assert(DeclDie && "This DIE should've already been constructed when the "
                      "definition DIE was created in "
                      "getOrCreateSubprogramDIE");
    DeclLinkageName = SPDecl->getLinkageName();
  }

  // Add function template parameters.
  addTemplateParams(SPDie, SP->getTemplateParams());

  // Add the linkage name if we have one and it isn't in the Decl.
  StringRef LinkageName = SP->getLinkageName();
  assert(((LinkageName.empty() || DeclLinkageName.empty()) ||
          LinkageName == DeclLinkageName) &&
         "decl has a linkage name and it is different");
  if (DeclLinkageName.empty())
    addLinkageName(SPDie, LinkageName);

  if (!DeclDie)
    return false;

  // Refer to the function declaration where all the other attributes will be
  // found.
  addDIEEntry(SPDie, dwarf::DW_AT_specification, *DeclDie);
  return true;
}

void DwarfUnit::applySubprogramAttributes(const DISubprogram *SP, DIE &SPDie,
                                          bool Minimal) {
  if (!Minimal)
    if (applySubprogramDefinitionAttributes(SP, SPDie))
      return;

  // Constructors and operators for anonymous aggregates do not have names.
  if (!SP->getName().empty())
    addString(SPDie, dwarf::DW_AT_name, SP->getName());

  // Skip the rest of the attributes under -gmlt to save space.
  if (Minimal)
    return;

  addSourceLine(SPDie, SP);

  // Add the prototype if we have a prototype and we have a C like
  // language.
  uint16_t Language = getLanguage();
  if (SP->isPrototyped() &&
      (Language == dwarf::DW_LANG_C89 || Language == dwarf::DW_LANG_C99 ||
       Language == dwarf::DW_LANG_ObjC))
    addFlag(SPDie, dwarf::DW_AT_prototyped);

  const DISubroutineType *SPTy = SP->getType();
  assert(SPTy->getTag() == dwarf::DW_TAG_subroutine_type &&
         "the type of a subprogram should be a subroutine");

  auto Args = SPTy->getTypeArray();
  // Add a return type. If this is a type like a C/C++ void type we don't add a
  // return type.
  if (Args.size())
    if (auto Ty = resolve(Args[0]))
      addType(SPDie, Ty);

  unsigned VK = SP->getVirtuality();
  if (VK) {
    addUInt(SPDie, dwarf::DW_AT_virtuality, dwarf::DW_FORM_data1, VK);
    DIELoc *Block = getDIELoc();
    addUInt(*Block, dwarf::DW_FORM_data1, dwarf::DW_OP_constu);
    addUInt(*Block, dwarf::DW_FORM_udata, SP->getVirtualIndex());
    addBlock(SPDie, dwarf::DW_AT_vtable_elem_location, Block);
    ContainingTypeMap.insert(
        std::make_pair(&SPDie, resolve(SP->getContainingType())));
  }

  if (!SP->isDefinition()) {
    addFlag(SPDie, dwarf::DW_AT_declaration);

    // Add arguments. Do not add arguments for subprogram definition. They will
    // be handled while processing variables.
    constructSubprogramArguments(SPDie, Args);
  }

  if (SP->isArtificial())
    addFlag(SPDie, dwarf::DW_AT_artificial);

  if (!SP->isLocalToUnit())
    addFlag(SPDie, dwarf::DW_AT_external);

  if (SP->isOptimized())
    addFlag(SPDie, dwarf::DW_AT_APPLE_optimized);

  if (unsigned isa = Asm->getISAEncoding())
    addUInt(SPDie, dwarf::DW_AT_APPLE_isa, dwarf::DW_FORM_flag, isa);

  if (SP->isLValueReference())
    addFlag(SPDie, dwarf::DW_AT_reference);

  if (SP->isRValueReference())
    addFlag(SPDie, dwarf::DW_AT_rvalue_reference);

  if (SP->isProtected())
    addUInt(SPDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_data1,
            dwarf::DW_ACCESS_protected);
  else if (SP->isPrivate())
    addUInt(SPDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_data1,
            dwarf::DW_ACCESS_private);
  else if (SP->isPublic())
    addUInt(SPDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_data1,
            dwarf::DW_ACCESS_public);

  if (SP->isExplicit())
    addFlag(SPDie, dwarf::DW_AT_explicit);
}

void DwarfUnit::constructSubrangeDIE(DIE &Buffer, const DISubrange *SR,
                                     DIE *IndexTy) {
  DIE &DW_Subrange = createAndAddDIE(dwarf::DW_TAG_subrange_type, Buffer);
  addDIEEntry(DW_Subrange, dwarf::DW_AT_type, *IndexTy);

  // The LowerBound value defines the lower bounds which is typically zero for
  // C/C++. The Count value is the number of elements.  Values are 64 bit. If
  // Count == -1 then the array is unbounded and we do not emit
  // DW_AT_lower_bound and DW_AT_count attributes.
  int64_t LowerBound = SR->getLowerBound();
  int64_t DefaultLowerBound = getDefaultLowerBound();
  int64_t Count = SR->getCount();

  if (DefaultLowerBound == -1 || LowerBound != DefaultLowerBound)
    addUInt(DW_Subrange, dwarf::DW_AT_lower_bound, None, LowerBound);

  if (Count != -1)
    // FIXME: An unbounded array should reference the expression that defines
    // the array.
    addUInt(DW_Subrange, dwarf::DW_AT_count, None, Count);
}

DIE *DwarfUnit::getIndexTyDie() {
  if (IndexTyDie)
    return IndexTyDie;
  // Construct an integer type to use for indexes.
  IndexTyDie = &createAndAddDIE(dwarf::DW_TAG_base_type, UnitDie);
  addString(*IndexTyDie, dwarf::DW_AT_name, "sizetype");
  addUInt(*IndexTyDie, dwarf::DW_AT_byte_size, None, sizeof(int64_t));
  addUInt(*IndexTyDie, dwarf::DW_AT_encoding, dwarf::DW_FORM_data1,
          dwarf::DW_ATE_unsigned);
  return IndexTyDie;
}

void DwarfUnit::constructArrayTypeDIE(DIE &Buffer, const DICompositeType *CTy) {
  if (CTy->isVector())
    addFlag(Buffer, dwarf::DW_AT_GNU_vector);

  // Emit the element type.
  addType(Buffer, resolve(CTy->getBaseType()));

  // Get an anonymous type for index type.
  // FIXME: This type should be passed down from the front end
  // as different languages may have different sizes for indexes.
  DIE *IdxTy = getIndexTyDie();

  // Add subranges to array type.
  DINodeArray Elements = CTy->getElements();
  for (unsigned i = 0, N = Elements.size(); i < N; ++i) {
    // FIXME: Should this really be such a loose cast?
    if (auto *Element = dyn_cast_or_null<DINode>(Elements[i]))
      if (Element->getTag() == dwarf::DW_TAG_subrange_type)
        constructSubrangeDIE(Buffer, cast<DISubrange>(Element), IdxTy);
  }
}

void DwarfUnit::constructEnumTypeDIE(DIE &Buffer, const DICompositeType *CTy) {
  DINodeArray Elements = CTy->getElements();

  // Add enumerators to enumeration type.
  for (unsigned i = 0, N = Elements.size(); i < N; ++i) {
    auto *Enum = dyn_cast_or_null<DIEnumerator>(Elements[i]);
    if (Enum) {
      DIE &Enumerator = createAndAddDIE(dwarf::DW_TAG_enumerator, Buffer);
      StringRef Name = Enum->getName();
      addString(Enumerator, dwarf::DW_AT_name, Name);
      int64_t Value = Enum->getValue();
      addSInt(Enumerator, dwarf::DW_AT_const_value, dwarf::DW_FORM_sdata,
              Value);
    }
  }
  const DIType *DTy = resolve(CTy->getBaseType());
  if (DTy) {
    addType(Buffer, DTy);
    addFlag(Buffer, dwarf::DW_AT_enum_class);
  }
}

void DwarfUnit::constructContainingTypeDIEs() {
  for (auto CI = ContainingTypeMap.begin(), CE = ContainingTypeMap.end();
       CI != CE; ++CI) {
    DIE &SPDie = *CI->first;
    const DINode *D = CI->second;
    if (!D)
      continue;
    DIE *NDie = getDIE(D);
    if (!NDie)
      continue;
    addDIEEntry(SPDie, dwarf::DW_AT_containing_type, *NDie);
  }
}

void DwarfUnit::constructMemberDIE(DIE &Buffer, const DIDerivedType *DT) {
  DIE &MemberDie = createAndAddDIE(DT->getTag(), Buffer);
  StringRef Name = DT->getName();
  if (!Name.empty())
    addString(MemberDie, dwarf::DW_AT_name, Name);

  addType(MemberDie, resolve(DT->getBaseType()));

  addSourceLine(MemberDie, DT);

  if (DT->getTag() == dwarf::DW_TAG_inheritance && DT->isVirtual()) {

    // For C++, virtual base classes are not at fixed offset. Use following
    // expression to extract appropriate offset from vtable.
    // BaseAddr = ObAddr + *((*ObAddr) - Offset)

    DIELoc *VBaseLocationDie = new (DIEValueAllocator) DIELoc;
    addUInt(*VBaseLocationDie, dwarf::DW_FORM_data1, dwarf::DW_OP_dup);
    addUInt(*VBaseLocationDie, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);
    addUInt(*VBaseLocationDie, dwarf::DW_FORM_data1, dwarf::DW_OP_constu);
    addUInt(*VBaseLocationDie, dwarf::DW_FORM_udata, DT->getOffsetInBits());
    addUInt(*VBaseLocationDie, dwarf::DW_FORM_data1, dwarf::DW_OP_minus);
    addUInt(*VBaseLocationDie, dwarf::DW_FORM_data1, dwarf::DW_OP_deref);
    addUInt(*VBaseLocationDie, dwarf::DW_FORM_data1, dwarf::DW_OP_plus);

    addBlock(MemberDie, dwarf::DW_AT_data_member_location, VBaseLocationDie);
  } else {
    uint64_t Size = DT->getSizeInBits();
    uint64_t FieldSize = getBaseTypeSize(DD, DT);
    uint64_t OffsetInBytes;

    if (FieldSize && Size != FieldSize) {
      // Handle bitfield, assume bytes are 8 bits.
      addUInt(MemberDie, dwarf::DW_AT_byte_size, None, FieldSize/8);
      addUInt(MemberDie, dwarf::DW_AT_bit_size, None, Size);
      //
      // The DWARF 2 DW_AT_bit_offset is counting the bits between the most
      // significant bit of the aligned storage unit containing the bit field to
      // the most significan bit of the bit field.
      //
      // FIXME: DWARF 4 states that DW_AT_data_bit_offset (which
      // counts from the beginning, regardless of endianness) should
      // be used instead.
      //
      //
      // Struct      Align       Align       Align
      // v           v           v           v
      // +-----------+-----*-----+-----*-----+--
      // | ...             |b1|b2|b3|b4|
      // +-----------+-----*-----+-----*-----+--
      // |           |     |<-- Size ->|     |
      // |<---- Offset --->|           |<--->|
      // |           |     |              \_ DW_AT_bit_offset (little endian)
      // |           |<--->|
      // |<--------->|  \_ StartBitOffset = DW_AT_bit_offset (big endian)
      //     \                            = DW_AT_data_bit_offset (biendian)
      //      \_ OffsetInBytes
      uint64_t Offset = DT->getOffsetInBits();
      uint64_t Align = DT->getAlignInBits() ? DT->getAlignInBits() : FieldSize;
      uint64_t AlignMask = ~(Align - 1);
      // The bits from the start of the storage unit to the start of the field.
      uint64_t StartBitOffset = Offset - (Offset & AlignMask);
      // The endian-dependent DWARF 2 offset.
      uint64_t DwarfBitOffset = Asm->getDataLayout().isLittleEndian()
        ? OffsetToAlignment(Offset + Size, Align)
        : StartBitOffset;

      // The byte offset of the field's aligned storage unit inside the struct.
      OffsetInBytes = (Offset - StartBitOffset) / 8;
      addUInt(MemberDie, dwarf::DW_AT_bit_offset, None, DwarfBitOffset);
    } else
      // This is not a bitfield.
      OffsetInBytes = DT->getOffsetInBits() / 8;

    if (DD->getDwarfVersion() <= 2) {
      DIELoc *MemLocationDie = new (DIEValueAllocator) DIELoc;
      addUInt(*MemLocationDie, dwarf::DW_FORM_data1, dwarf::DW_OP_plus_uconst);
      addUInt(*MemLocationDie, dwarf::DW_FORM_udata, OffsetInBytes);
      addBlock(MemberDie, dwarf::DW_AT_data_member_location, MemLocationDie);
    } else
      addUInt(MemberDie, dwarf::DW_AT_data_member_location, None,
              OffsetInBytes);
  }

  if (DT->isProtected())
    addUInt(MemberDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_data1,
            dwarf::DW_ACCESS_protected);
  else if (DT->isPrivate())
    addUInt(MemberDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_data1,
            dwarf::DW_ACCESS_private);
  // Otherwise C++ member and base classes are considered public.
  else if (DT->isPublic())
    addUInt(MemberDie, dwarf::DW_AT_accessibility, dwarf::DW_FORM_data1,
            dwarf::DW_ACCESS_public);
  if (DT->isVirtual())
    addUInt(MemberDie, dwarf::DW_AT_virtuality, dwarf::DW_FORM_data1,
            dwarf::DW_VIRTUALITY_virtual);

  // Objective-C properties.
  if (DINode *PNode = DT->getObjCProperty())
    if (DIE *PDie = getDIE(PNode))
      MemberDie.addValue(DIEValueAllocator, dwarf::DW_AT_APPLE_property,
                         dwarf::DW_FORM_ref4, DIEEntry(*PDie));

  if (DT->isArtificial())
    addFlag(MemberDie, dwarf::DW_AT_artificial);
}

DIE *DwarfUnit::getOrCreateStaticMemberDIE(const DIDerivedType *DT) {
  if (!DT)
    return nullptr;

  // Construct the context before querying for the existence of the DIE in case
  // such construction creates the DIE.
  DIE *ContextDIE = getOrCreateContextDIE(resolve(DT->getScope()));
  assert(dwarf::isType(ContextDIE->getTag()) &&
         "Static member should belong to a type.");

  if (DIE *StaticMemberDIE = getDIE(DT))
    return StaticMemberDIE;

  DIE &StaticMemberDIE = createAndAddDIE(DT->getTag(), *ContextDIE, DT);

  const DIType *Ty = resolve(DT->getBaseType());

  addString(StaticMemberDIE, dwarf::DW_AT_name, DT->getName());
  addType(StaticMemberDIE, Ty);
  addSourceLine(StaticMemberDIE, DT);
  addFlag(StaticMemberDIE, dwarf::DW_AT_external);
  addFlag(StaticMemberDIE, dwarf::DW_AT_declaration);

  // FIXME: We could omit private if the parent is a class_type, and
  // public if the parent is something else.
  if (DT->isProtected())
    addUInt(StaticMemberDIE, dwarf::DW_AT_accessibility, dwarf::DW_FORM_data1,
            dwarf::DW_ACCESS_protected);
  else if (DT->isPrivate())
    addUInt(StaticMemberDIE, dwarf::DW_AT_accessibility, dwarf::DW_FORM_data1,
            dwarf::DW_ACCESS_private);
  else if (DT->isPublic())
    addUInt(StaticMemberDIE, dwarf::DW_AT_accessibility, dwarf::DW_FORM_data1,
            dwarf::DW_ACCESS_public);

  if (const ConstantInt *CI = dyn_cast_or_null<ConstantInt>(DT->getConstant()))
    addConstantValue(StaticMemberDIE, CI, Ty);
  if (const ConstantFP *CFP = dyn_cast_or_null<ConstantFP>(DT->getConstant()))
    addConstantFPValue(StaticMemberDIE, CFP);

  return &StaticMemberDIE;
}

void DwarfUnit::emitHeader(bool UseOffsets) {
  // Emit size of content not including length itself
  Asm->OutStreamer->AddComment("Length of Unit");
  Asm->EmitInt32(getHeaderSize() + UnitDie.getSize());

  Asm->OutStreamer->AddComment("DWARF version number");
  Asm->EmitInt16(DD->getDwarfVersion());
  Asm->OutStreamer->AddComment("Offset Into Abbrev. Section");

  // We share one abbreviations table across all units so it's always at the
  // start of the section. Use a relocatable offset where needed to ensure
  // linking doesn't invalidate that offset.
  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();
  Asm->emitDwarfSymbolReference(TLOF.getDwarfAbbrevSection()->getBeginSymbol(),
                                UseOffsets);

  Asm->OutStreamer->AddComment("Address Size (in bytes)");
  Asm->EmitInt8(Asm->getDataLayout().getPointerSize());
}

void DwarfUnit::initSection(MCSection *Section) {
  assert(!this->Section);
  this->Section = Section;
}

void DwarfTypeUnit::emitHeader(bool UseOffsets) {
  DwarfUnit::emitHeader(UseOffsets);
  Asm->OutStreamer->AddComment("Type Signature");
  Asm->OutStreamer->EmitIntValue(TypeSignature, sizeof(TypeSignature));
  Asm->OutStreamer->AddComment("Type DIE Offset");
  // In a skeleton type unit there is no type DIE so emit a zero offset.
  Asm->OutStreamer->EmitIntValue(Ty ? Ty->getOffset() : 0,
                                 sizeof(Ty->getOffset()));
}

bool DwarfTypeUnit::isDwoUnit() const {
  // Since there are no skeleton type units, all type units are dwo type units
  // when split DWARF is being used.
  return DD->useSplitDwarf();
}
