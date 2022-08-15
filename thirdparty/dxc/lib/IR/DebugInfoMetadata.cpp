//===- DebugInfoMetadata.cpp - Implement debug info metadata --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the debug info Metadata classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfoMetadata.h"
#include "LLVMContextImpl.h"
#include "MetadataImpl.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Function.h"

using namespace llvm;

DILocation::DILocation(LLVMContext &C, StorageType Storage, unsigned Line,
                       unsigned Column, ArrayRef<Metadata *> MDs)
    : MDNode(C, DILocationKind, Storage, MDs) {
  assert((MDs.size() == 1 || MDs.size() == 2) &&
         "Expected a scope and optional inlined-at");

  // Set line and column.
  assert(Column < (1u << 16) && "Expected 16-bit column");

  SubclassData32 = Line;
  SubclassData16 = Column;
}

static void adjustColumn(unsigned &Column) {
  // Set to unknown on overflow.  We only have 16 bits to play with here.
  if (Column >= (1u << 16))
    Column = 0;
}

DILocation *DILocation::getImpl(LLVMContext &Context, unsigned Line,
                                unsigned Column, Metadata *Scope,
                                Metadata *InlinedAt, StorageType Storage,
                                bool ShouldCreate) {
  // Fixup column.
  adjustColumn(Column);

  assert(Scope && "Expected scope");
  if (Storage == Uniqued) {
    if (auto *N =
            getUniqued(Context.pImpl->DILocations,
                       DILocationInfo::KeyTy(Line, Column, Scope, InlinedAt)))
      return N;
    if (!ShouldCreate)
      return nullptr;
  } else {
    assert(ShouldCreate && "Expected non-uniqued nodes to always be created");
  }

  SmallVector<Metadata *, 2> Ops;
  Ops.push_back(Scope);
  if (InlinedAt)
    Ops.push_back(InlinedAt);
  return storeImpl(new (Ops.size())
                       DILocation(Context, Storage, Line, Column, Ops),
                   Storage, Context.pImpl->DILocations);
}

unsigned DILocation::computeNewDiscriminator() const {
  // FIXME: This seems completely wrong.
  //
  //  1. If two modules are generated in the same context, then the second
  //     Module will get different discriminators than it would have if it were
  //     generated in its own context.
  //  2. If this function is called after round-tripping to bitcode instead of
  //     before, it will give a different (and potentially incorrect!) return.
  //
  // The discriminator should instead be calculated from local information
  // where it's actually needed.  This logic should be moved to
  // AddDiscriminators::runOnFunction(), where it doesn't pollute the
  // LLVMContext.
  std::pair<const char *, unsigned> Key(getFilename().data(), getLine());
  return ++getContext().pImpl->DiscriminatorTable[Key];
}

unsigned DINode::getFlag(StringRef Flag) {
  return StringSwitch<unsigned>(Flag)
#define HANDLE_DI_FLAG(ID, NAME) .Case("DIFlag" #NAME, Flag##NAME)
#include "llvm/IR/DebugInfoFlags.def"
      .Default(0);
}

const char *DINode::getFlagString(unsigned Flag) {
  switch (Flag) {
  default:
    return "";
#define HANDLE_DI_FLAG(ID, NAME)                                               \
  case Flag##NAME:                                                             \
    return "DIFlag" #NAME;
#include "llvm/IR/DebugInfoFlags.def"
  }
}

unsigned DINode::splitFlags(unsigned Flags,
                            SmallVectorImpl<unsigned> &SplitFlags) {
  // Accessibility flags need to be specially handled, since they're packed
  // together.
  if (unsigned A = Flags & FlagAccessibility) {
    if (A == FlagPrivate)
      SplitFlags.push_back(FlagPrivate);
    else if (A == FlagProtected)
      SplitFlags.push_back(FlagProtected);
    else
      SplitFlags.push_back(FlagPublic);
    Flags &= ~A;
  }

#define HANDLE_DI_FLAG(ID, NAME)                                               \
  if (unsigned Bit = Flags & ID) {                                             \
    SplitFlags.push_back(Bit);                                                 \
    Flags &= ~Bit;                                                             \
  }
#include "llvm/IR/DebugInfoFlags.def"

  return Flags;
}

DIScopeRef DIScope::getScope() const {
  if (auto *T = dyn_cast<DIType>(this))
    return T->getScope();

  if (auto *SP = dyn_cast<DISubprogram>(this))
    return SP->getScope();

  if (auto *LB = dyn_cast<DILexicalBlockBase>(this))
    return DIScopeRef(LB->getScope());

  if (auto *NS = dyn_cast<DINamespace>(this))
    return DIScopeRef(NS->getScope());

  if (auto *M = dyn_cast<DIModule>(this))
    return DIScopeRef(M->getScope());

  assert((isa<DIFile>(this) || isa<DICompileUnit>(this)) &&
         "Unhandled type of scope.");
  return nullptr;
}

StringRef DIScope::getName() const {
  if (auto *T = dyn_cast<DIType>(this))
    return T->getName();
  if (auto *SP = dyn_cast<DISubprogram>(this))
    return SP->getName();
  if (auto *NS = dyn_cast<DINamespace>(this))
    return NS->getName();
  if (auto *M = dyn_cast<DIModule>(this))
    return M->getName();
  assert((isa<DILexicalBlockBase>(this) || isa<DIFile>(this) ||
          isa<DICompileUnit>(this)) &&
         "Unhandled type of scope.");
  return "";
}

static StringRef getString(const MDString *S) {
  if (S)
    return S->getString();
  return StringRef();
}

#ifndef NDEBUG
static bool isCanonical(const MDString *S) {
  return !S || !S->getString().empty();
}
#endif

GenericDINode *GenericDINode::getImpl(LLVMContext &Context, unsigned Tag,
                                      MDString *Header,
                                      ArrayRef<Metadata *> DwarfOps,
                                      StorageType Storage, bool ShouldCreate) {
  unsigned Hash = 0;
  if (Storage == Uniqued) {
    GenericDINodeInfo::KeyTy Key(Tag, getString(Header), DwarfOps);
    if (auto *N = getUniqued(Context.pImpl->GenericDINodes, Key))
      return N;
    if (!ShouldCreate)
      return nullptr;
    Hash = Key.getHash();
  } else {
    assert(ShouldCreate && "Expected non-uniqued nodes to always be created");
  }

  // Use a nullptr for empty headers.
  assert(isCanonical(Header) && "Expected canonical MDString");
  Metadata *PreOps[] = {Header};
  return storeImpl(new (DwarfOps.size() + 1) GenericDINode(
                       Context, Storage, Hash, Tag, PreOps, DwarfOps),
                   Storage, Context.pImpl->GenericDINodes);
}

void GenericDINode::recalculateHash() {
  setHash(GenericDINodeInfo::KeyTy::calculateHash(this));
}

#define UNWRAP_ARGS_IMPL(...) __VA_ARGS__
#define UNWRAP_ARGS(ARGS) UNWRAP_ARGS_IMPL ARGS
#define DEFINE_GETIMPL_LOOKUP(CLASS, ARGS)                                     \
  do {                                                                         \
    if (Storage == Uniqued) {                                                  \
      if (auto *N = getUniqued(Context.pImpl->CLASS##s,                        \
                               CLASS##Info::KeyTy(UNWRAP_ARGS(ARGS))))         \
        return N;                                                              \
      if (!ShouldCreate)                                                       \
        return nullptr;                                                        \
    } else {                                                                   \
      assert(ShouldCreate &&                                                   \
             "Expected non-uniqued nodes to always be created");               \
    }                                                                          \
  } while (false)
#define DEFINE_GETIMPL_STORE(CLASS, ARGS, OPS)                                 \
  return storeImpl(new (ArrayRef<Metadata *>(OPS).size())                      \
                       CLASS(Context, Storage, UNWRAP_ARGS(ARGS), OPS),        \
                   Storage, Context.pImpl->CLASS##s)
#define DEFINE_GETIMPL_STORE_NO_OPS(CLASS, ARGS)                               \
  return storeImpl(new (0u) CLASS(Context, Storage, UNWRAP_ARGS(ARGS)),        \
                   Storage, Context.pImpl->CLASS##s)
#define DEFINE_GETIMPL_STORE_NO_CONSTRUCTOR_ARGS(CLASS, OPS)                   \
  return storeImpl(new (ArrayRef<Metadata *>(OPS).size())                      \
                       CLASS(Context, Storage, OPS),                           \
                   Storage, Context.pImpl->CLASS##s)

DISubrange *DISubrange::getImpl(LLVMContext &Context, int64_t Count, int64_t Lo,
                                StorageType Storage, bool ShouldCreate) {
  DEFINE_GETIMPL_LOOKUP(DISubrange, (Count, Lo));
  DEFINE_GETIMPL_STORE_NO_OPS(DISubrange, (Count, Lo));
}

DIEnumerator *DIEnumerator::getImpl(LLVMContext &Context, int64_t Value,
                                    MDString *Name, StorageType Storage,
                                    bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DIEnumerator, (Value, getString(Name)));
  Metadata *Ops[] = {Name};
  DEFINE_GETIMPL_STORE(DIEnumerator, (Value), Ops);
}

DIBasicType *DIBasicType::getImpl(LLVMContext &Context, unsigned Tag,
                                  MDString *Name, uint64_t SizeInBits,
                                  uint64_t AlignInBits, unsigned Encoding,
                                  StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(
      DIBasicType, (Tag, getString(Name), SizeInBits, AlignInBits, Encoding));
  Metadata *Ops[] = {nullptr, nullptr, Name};
  DEFINE_GETIMPL_STORE(DIBasicType, (Tag, SizeInBits, AlignInBits, Encoding),
                       Ops);
}

DIDerivedType *DIDerivedType::getImpl(
    LLVMContext &Context, unsigned Tag, MDString *Name, Metadata *File,
    unsigned Line, Metadata *Scope, Metadata *BaseType, uint64_t SizeInBits,
    uint64_t AlignInBits, uint64_t OffsetInBits, unsigned Flags,
    Metadata *ExtraData, StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DIDerivedType, (Tag, getString(Name), File, Line, Scope,
                                        BaseType, SizeInBits, AlignInBits,
                                        OffsetInBits, Flags, ExtraData));
  Metadata *Ops[] = {File, Scope, Name, BaseType, ExtraData};
  DEFINE_GETIMPL_STORE(
      DIDerivedType, (Tag, Line, SizeInBits, AlignInBits, OffsetInBits, Flags),
      Ops);
}

DICompositeType *DICompositeType::getImpl(
    LLVMContext &Context, unsigned Tag, MDString *Name, Metadata *File,
    unsigned Line, Metadata *Scope, Metadata *BaseType, uint64_t SizeInBits,
    uint64_t AlignInBits, uint64_t OffsetInBits, unsigned Flags,
    Metadata *Elements, unsigned RuntimeLang, Metadata *VTableHolder,
    Metadata *TemplateParams, MDString *Identifier, StorageType Storage,
    bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DICompositeType,
                        (Tag, getString(Name), File, Line, Scope, BaseType,
                         SizeInBits, AlignInBits, OffsetInBits, Flags, Elements,
                         RuntimeLang, VTableHolder, TemplateParams,
                         getString(Identifier)));
  Metadata *Ops[] = {File,     Scope,        Name,           BaseType,
                     Elements, VTableHolder, TemplateParams, Identifier};
  DEFINE_GETIMPL_STORE(DICompositeType, (Tag, Line, RuntimeLang, SizeInBits,
                                         AlignInBits, OffsetInBits, Flags),
                       Ops);
}

DISubroutineType *DISubroutineType::getImpl(LLVMContext &Context,
                                            unsigned Flags, Metadata *TypeArray,
                                            StorageType Storage,
                                            bool ShouldCreate) {
  DEFINE_GETIMPL_LOOKUP(DISubroutineType, (Flags, TypeArray));
  Metadata *Ops[] = {nullptr,   nullptr, nullptr, nullptr,
                     TypeArray, nullptr, nullptr, nullptr};
  DEFINE_GETIMPL_STORE(DISubroutineType, (Flags), Ops);
}

DIFile *DIFile::getImpl(LLVMContext &Context, MDString *Filename,
                        MDString *Directory, StorageType Storage,
                        bool ShouldCreate) {
  assert(isCanonical(Filename) && "Expected canonical MDString");
  assert(isCanonical(Directory) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DIFile, (getString(Filename), getString(Directory)));
  Metadata *Ops[] = {Filename, Directory};
  DEFINE_GETIMPL_STORE_NO_CONSTRUCTOR_ARGS(DIFile, Ops);
}

DICompileUnit *DICompileUnit::getImpl(
    LLVMContext &Context, unsigned SourceLanguage, Metadata *File,
    MDString *Producer, bool IsOptimized, MDString *Flags,
    unsigned RuntimeVersion, MDString *SplitDebugFilename,
    unsigned EmissionKind, Metadata *EnumTypes, Metadata *RetainedTypes,
    Metadata *Subprograms, Metadata *GlobalVariables,
    Metadata *ImportedEntities, uint64_t DWOId,
    StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Producer) && "Expected canonical MDString");
  assert(isCanonical(Flags) && "Expected canonical MDString");
  assert(isCanonical(SplitDebugFilename) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(
      DICompileUnit,
      (SourceLanguage, File, getString(Producer), IsOptimized, getString(Flags),
       RuntimeVersion, getString(SplitDebugFilename), EmissionKind, EnumTypes,
       RetainedTypes, Subprograms, GlobalVariables, ImportedEntities, DWOId));
  Metadata *Ops[] = {File, Producer, Flags, SplitDebugFilename, EnumTypes,
                     RetainedTypes, Subprograms, GlobalVariables,
                     ImportedEntities};
  DEFINE_GETIMPL_STORE(
      DICompileUnit,
      (SourceLanguage, IsOptimized, RuntimeVersion, EmissionKind, DWOId), Ops);
}

DISubprogram *DILocalScope::getSubprogram() const {
  if (auto *Block = dyn_cast<DILexicalBlockBase>(this))
    return Block->getScope()->getSubprogram();
  return const_cast<DISubprogram *>(cast<DISubprogram>(this));
}

DISubprogram *DISubprogram::getImpl(
    LLVMContext &Context, Metadata *Scope, MDString *Name,
    MDString *LinkageName, Metadata *File, unsigned Line, Metadata *Type,
    bool IsLocalToUnit, bool IsDefinition, unsigned ScopeLine,
    Metadata *ContainingType, unsigned Virtuality, unsigned VirtualIndex,
    unsigned Flags, bool IsOptimized, Metadata *Function,
    Metadata *TemplateParams, Metadata *Declaration, Metadata *Variables,
    StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  assert(isCanonical(LinkageName) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DISubprogram,
                        (Scope, getString(Name), getString(LinkageName), File,
                         Line, Type, IsLocalToUnit, IsDefinition, ScopeLine,
                         ContainingType, Virtuality, VirtualIndex, Flags,
                         IsOptimized, Function, TemplateParams, Declaration,
                         Variables));
  Metadata *Ops[] = {File,           Scope,       Name,           Name,
                     LinkageName,    Type,        ContainingType, Function,
                     TemplateParams, Declaration, Variables};
  DEFINE_GETIMPL_STORE(DISubprogram,
                       (Line, ScopeLine, Virtuality, VirtualIndex, Flags,
                        IsLocalToUnit, IsDefinition, IsOptimized),
                       Ops);
}

Function *DISubprogram::getFunction() const {
  // FIXME: Should this be looking through bitcasts?
  return dyn_cast_or_null<Function>(getFunctionConstant());
}

bool DISubprogram::describes(const Function *F) const {
  assert(F && "Invalid function");
  if (F == getFunction())
    return true;
  StringRef Name = getLinkageName();
  if (Name.empty())
    Name = getName();
  return F->getName() == Name;
}

void DISubprogram::replaceFunction(Function *F) {
  replaceFunction(F ? ConstantAsMetadata::get(F)
                    : static_cast<ConstantAsMetadata *>(nullptr));
}

DILexicalBlock *DILexicalBlock::getImpl(LLVMContext &Context, Metadata *Scope,
                                        Metadata *File, unsigned Line,
                                        unsigned Column, StorageType Storage,
                                        bool ShouldCreate) {
  assert(Scope && "Expected scope");
  DEFINE_GETIMPL_LOOKUP(DILexicalBlock, (Scope, File, Line, Column));
  Metadata *Ops[] = {File, Scope};
  DEFINE_GETIMPL_STORE(DILexicalBlock, (Line, Column), Ops);
}

DILexicalBlockFile *DILexicalBlockFile::getImpl(LLVMContext &Context,
                                                Metadata *Scope, Metadata *File,
                                                unsigned Discriminator,
                                                StorageType Storage,
                                                bool ShouldCreate) {
  assert(Scope && "Expected scope");
  DEFINE_GETIMPL_LOOKUP(DILexicalBlockFile, (Scope, File, Discriminator));
  Metadata *Ops[] = {File, Scope};
  DEFINE_GETIMPL_STORE(DILexicalBlockFile, (Discriminator), Ops);
}

DINamespace *DINamespace::getImpl(LLVMContext &Context, Metadata *Scope,
                                  Metadata *File, MDString *Name, unsigned Line,
                                  StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DINamespace, (Scope, File, getString(Name), Line));
  Metadata *Ops[] = {File, Scope, Name};
  DEFINE_GETIMPL_STORE(DINamespace, (Line), Ops);
}

DIModule *DIModule::getImpl(LLVMContext &Context, Metadata *Scope,
                            MDString *Name, MDString *ConfigurationMacros,
                            MDString *IncludePath, MDString *ISysRoot,
                            StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DIModule,
    (Scope, getString(Name), getString(ConfigurationMacros),
     getString(IncludePath), getString(ISysRoot)));
  Metadata *Ops[] = {Scope, Name, ConfigurationMacros, IncludePath, ISysRoot};
  DEFINE_GETIMPL_STORE_NO_CONSTRUCTOR_ARGS(DIModule, Ops);
}

DITemplateTypeParameter *DITemplateTypeParameter::getImpl(LLVMContext &Context,
                                                          MDString *Name,
                                                          Metadata *Type,
                                                          StorageType Storage,
                                                          bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DITemplateTypeParameter, (getString(Name), Type));
  Metadata *Ops[] = {Name, Type};
  DEFINE_GETIMPL_STORE_NO_CONSTRUCTOR_ARGS(DITemplateTypeParameter, Ops);
}

DITemplateValueParameter *DITemplateValueParameter::getImpl(
    LLVMContext &Context, unsigned Tag, MDString *Name, Metadata *Type,
    Metadata *Value, StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DITemplateValueParameter,
                        (Tag, getString(Name), Type, Value));
  Metadata *Ops[] = {Name, Type, Value};
  DEFINE_GETIMPL_STORE(DITemplateValueParameter, (Tag), Ops);
}

DIGlobalVariable *
DIGlobalVariable::getImpl(LLVMContext &Context, Metadata *Scope, MDString *Name,
                          MDString *LinkageName, Metadata *File, unsigned Line,
                          Metadata *Type, bool IsLocalToUnit, bool IsDefinition,
                          Metadata *Variable,
                          Metadata *StaticDataMemberDeclaration,
                          StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  assert(isCanonical(LinkageName) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DIGlobalVariable,
                        (Scope, getString(Name), getString(LinkageName), File,
                         Line, Type, IsLocalToUnit, IsDefinition, Variable,
                         StaticDataMemberDeclaration));
  Metadata *Ops[] = {Scope, Name,        File,     Type,
                     Name,  LinkageName, Variable, StaticDataMemberDeclaration};
  DEFINE_GETIMPL_STORE(DIGlobalVariable, (Line, IsLocalToUnit, IsDefinition),
                       Ops);
}

DILocalVariable *DILocalVariable::getImpl(LLVMContext &Context, unsigned Tag,
                                          Metadata *Scope, MDString *Name,
                                          Metadata *File, unsigned Line,
                                          Metadata *Type, unsigned Arg,
                                          unsigned Flags, StorageType Storage,
                                          bool ShouldCreate) {
  // 64K ought to be enough for any frontend.
  assert(Arg <= UINT16_MAX && "Expected argument number to fit in 16-bits");

  assert(Scope && "Expected scope");
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DILocalVariable, (Tag, Scope, getString(Name), File,
                                          Line, Type, Arg, Flags));
  Metadata *Ops[] = {Scope, Name, File, Type};
  DEFINE_GETIMPL_STORE(DILocalVariable, (Tag, Line, Arg, Flags), Ops);
}

DIExpression *DIExpression::getImpl(LLVMContext &Context,
                                    ArrayRef<uint64_t> Elements,
                                    StorageType Storage, bool ShouldCreate) {
  DEFINE_GETIMPL_LOOKUP(DIExpression, (Elements));
  DEFINE_GETIMPL_STORE_NO_OPS(DIExpression, (Elements));
}

unsigned DIExpression::ExprOperand::getSize() const {
  switch (getOp()) {
  case dwarf::DW_OP_bit_piece:
    return 3;
  case dwarf::DW_OP_plus:
    return 2;
  default:
    return 1;
  }
}

bool DIExpression::isValid() const {
  for (auto I = expr_op_begin(), E = expr_op_end(); I != E; ++I) {
    // Check that there's space for the operand.
    if (I->get() + I->getSize() > E->get())
      return false;

    // Check that the operand is valid.
    switch (I->getOp()) {
    default:
      return false;
    case dwarf::DW_OP_bit_piece:
      // Piece expressions must be at the end.
      return I->get() + I->getSize() == E->get();
    case dwarf::DW_OP_plus:
    case dwarf::DW_OP_deref:
      break;
    }
  }
  return true;
}

bool DIExpression::isBitPiece() const {
  assert(isValid() && "Expected valid expression");
  if (unsigned N = getNumElements())
    if (N >= 3)
      return getElement(N - 3) == dwarf::DW_OP_bit_piece;
  return false;
}

uint64_t DIExpression::getBitPieceOffset() const {
  assert(isBitPiece() && "Expected bit piece");
  return getElement(getNumElements() - 2);
}

uint64_t DIExpression::getBitPieceSize() const {
  assert(isBitPiece() && "Expected bit piece");
  return getElement(getNumElements() - 1);
}

DIObjCProperty *DIObjCProperty::getImpl(
    LLVMContext &Context, MDString *Name, Metadata *File, unsigned Line,
    MDString *GetterName, MDString *SetterName, unsigned Attributes,
    Metadata *Type, StorageType Storage, bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  assert(isCanonical(GetterName) && "Expected canonical MDString");
  assert(isCanonical(SetterName) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DIObjCProperty,
                        (getString(Name), File, Line, getString(GetterName),
                         getString(SetterName), Attributes, Type));
  Metadata *Ops[] = {Name, File, GetterName, SetterName, Type};
  DEFINE_GETIMPL_STORE(DIObjCProperty, (Line, Attributes), Ops);
}

DIImportedEntity *DIImportedEntity::getImpl(LLVMContext &Context, unsigned Tag,
                                            Metadata *Scope, Metadata *Entity,
                                            unsigned Line, MDString *Name,
                                            StorageType Storage,
                                            bool ShouldCreate) {
  assert(isCanonical(Name) && "Expected canonical MDString");
  DEFINE_GETIMPL_LOOKUP(DIImportedEntity,
                        (Tag, Scope, Entity, Line, getString(Name)));
  Metadata *Ops[] = {Scope, Entity, Name};
  DEFINE_GETIMPL_STORE(DIImportedEntity, (Tag, Line), Ops);
}
