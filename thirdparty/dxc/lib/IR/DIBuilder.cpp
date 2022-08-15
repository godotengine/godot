//===--- DIBuilder.cpp - Debug Information Builder ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the DIBuilder.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DIBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"

using namespace llvm;
using namespace llvm::dwarf;

namespace {
class HeaderBuilder {
  /// \brief Whether there are any fields yet.
  ///
  /// Note that this is not equivalent to \c Chars.empty(), since \a concat()
  /// may have been called already with an empty string.
  bool IsEmpty;
  SmallVector<char, 256> Chars;

public:
  HeaderBuilder() : IsEmpty(true) {}
  HeaderBuilder(const HeaderBuilder &X) : IsEmpty(X.IsEmpty), Chars(X.Chars) {}
  HeaderBuilder(HeaderBuilder &&X)
      : IsEmpty(X.IsEmpty), Chars(std::move(X.Chars)) {}

  template <class Twineable> HeaderBuilder &concat(Twineable &&X) {
    if (IsEmpty)
      IsEmpty = false;
    else
      Chars.push_back(0);
    Twine(X).toVector(Chars);
    return *this;
  }

  MDString *get(LLVMContext &Context) const {
    return MDString::get(Context, StringRef(Chars.begin(), Chars.size()));
  }

  static HeaderBuilder get(unsigned Tag) {
    return HeaderBuilder().concat("0x" + Twine::utohexstr(Tag));
  }
};
}

DIBuilder::DIBuilder(Module &m, bool AllowUnresolvedNodes)
  : M(m), VMContext(M.getContext()), CUNode(nullptr),
      DeclareFn(nullptr), ValueFn(nullptr),
      AllowUnresolvedNodes(AllowUnresolvedNodes) {}

void DIBuilder::trackIfUnresolved(MDNode *N) {
  if (!N)
    return;
  if (N->isResolved())
    return;

  assert(AllowUnresolvedNodes && "Cannot handle unresolved nodes");
  UnresolvedNodes.emplace_back(N);
}

void DIBuilder::finalize() {
  if (!CUNode) {
    assert(!AllowUnresolvedNodes &&
           "creating type nodes without a CU is not supported");
    return;
  }

  CUNode->replaceEnumTypes(MDTuple::get(VMContext, AllEnumTypes));

  SmallVector<Metadata *, 16> RetainValues;
  // Declarations and definitions of the same type may be retained. Some
  // clients RAUW these pairs, leaving duplicates in the retained types
  // list. Use a set to remove the duplicates while we transform the
  // TrackingVHs back into Values.
  SmallPtrSet<Metadata *, 16> RetainSet;
  for (unsigned I = 0, E = AllRetainTypes.size(); I < E; I++)
    if (RetainSet.insert(AllRetainTypes[I]).second)
      RetainValues.push_back(AllRetainTypes[I]);

  if (!RetainValues.empty())
    CUNode->replaceRetainedTypes(MDTuple::get(VMContext, RetainValues));

  DISubprogramArray SPs = MDTuple::get(VMContext, AllSubprograms);
  if (!AllSubprograms.empty())
    CUNode->replaceSubprograms(SPs.get());

  for (auto *SP : SPs) {
    if (MDTuple *Temp = SP->getVariables().get()) {
      const auto &PV = PreservedVariables.lookup(SP);
      SmallVector<Metadata *, 4> Variables(PV.begin(), PV.end());
      DINodeArray AV = getOrCreateArray(Variables);
      TempMDTuple(Temp)->replaceAllUsesWith(AV.get());
    }
  }

  if (!AllGVs.empty())
    CUNode->replaceGlobalVariables(MDTuple::get(VMContext, AllGVs));

  if (!AllImportedModules.empty())
    CUNode->replaceImportedEntities(MDTuple::get(
        VMContext, SmallVector<Metadata *, 16>(AllImportedModules.begin(),
                                               AllImportedModules.end())));

  // Now that all temp nodes have been replaced or deleted, resolve remaining
  // cycles.
  for (const auto &N : UnresolvedNodes)
    if (N && !N->isResolved())
      N->resolveCycles();
  UnresolvedNodes.clear();

  // Can't handle unresolved nodes anymore.
  AllowUnresolvedNodes = false;
}

/// If N is compile unit return NULL otherwise return N.
static DIScope *getNonCompileUnitScope(DIScope *N) {
  if (!N || isa<DICompileUnit>(N))
    return nullptr;
  return cast<DIScope>(N);
}

DICompileUnit *DIBuilder::createCompileUnit(
    unsigned Lang, StringRef Filename, StringRef Directory, StringRef Producer,
    bool isOptimized, StringRef Flags, unsigned RunTimeVer, StringRef SplitName,
    DebugEmissionKind Kind, uint64_t DWOId, bool EmitDebugInfo) {

  assert(((Lang <= dwarf::DW_LANG_Fortran08 && Lang >= dwarf::DW_LANG_C89) ||
          (Lang <= dwarf::DW_LANG_hi_user && Lang >= dwarf::DW_LANG_lo_user)) &&
         "Invalid Language tag");
  assert(!Filename.empty() &&
         "Unable to create compile unit without filename");

  assert(!CUNode && "Can only make one compile unit per DIBuilder instance");
  CUNode = DICompileUnit::getDistinct(
      VMContext, Lang, DIFile::get(VMContext, Filename, Directory), Producer,
      isOptimized, Flags, RunTimeVer, SplitName, Kind, nullptr,
      nullptr, nullptr, nullptr, nullptr, DWOId);

  // Create a named metadata so that it is easier to find cu in a module.
  // Note that we only generate this when the caller wants to actually
  // emit debug information. When we are only interested in tracking
  // source line locations throughout the backend, we prevent codegen from
  // emitting debug info in the final output by not generating llvm.dbg.cu.
  if (EmitDebugInfo) {
    NamedMDNode *NMD = M.getOrInsertNamedMetadata("llvm.dbg.cu");
    NMD->addOperand(CUNode);
  }

  trackIfUnresolved(CUNode);
  return CUNode;
}

static DIImportedEntity *
createImportedModule(LLVMContext &C, dwarf::Tag Tag, DIScope *Context,
                     Metadata *NS, unsigned Line, StringRef Name,
                     SmallVectorImpl<TrackingMDNodeRef> &AllImportedModules) {
  auto *M = DIImportedEntity::get(C, Tag, Context, DINodeRef(NS), Line, Name);
  AllImportedModules.emplace_back(M);
  return M;
}

DIImportedEntity *DIBuilder::createImportedModule(DIScope *Context,
                                                  DINamespace *NS,
                                                  unsigned Line) {
  return ::createImportedModule(VMContext, dwarf::DW_TAG_imported_module,
                                Context, NS, Line, StringRef(), AllImportedModules);
}

DIImportedEntity *DIBuilder::createImportedModule(DIScope *Context,
                                                  DIImportedEntity *NS,
                                                  unsigned Line) {
  return ::createImportedModule(VMContext, dwarf::DW_TAG_imported_module,
                                Context, NS, Line, StringRef(), AllImportedModules);
}

DIImportedEntity *DIBuilder::createImportedModule(DIScope *Context, DIModule *M,
                                                  unsigned Line) {
  return ::createImportedModule(VMContext, dwarf::DW_TAG_imported_module,
                                Context, M, Line, StringRef(), AllImportedModules);
}

DIImportedEntity *DIBuilder::createImportedDeclaration(DIScope *Context,
                                                       DINode *Decl,
                                                       unsigned Line,
                                                       StringRef Name) {
  // Make sure to use the unique identifier based metadata reference for
  // types that have one.
  return ::createImportedModule(VMContext, dwarf::DW_TAG_imported_declaration,
                                Context, DINodeRef::get(Decl), Line, Name,
                                AllImportedModules);
}

DIFile *DIBuilder::createFile(StringRef Filename, StringRef Directory) {
  return DIFile::get(VMContext, Filename, Directory);
}

DIEnumerator *DIBuilder::createEnumerator(StringRef Name, int64_t Val) {
  assert(!Name.empty() && "Unable to create enumerator without name");
  return DIEnumerator::get(VMContext, Val, Name);
}

DIBasicType *DIBuilder::createUnspecifiedType(StringRef Name) {
  assert(!Name.empty() && "Unable to create type without name");
  return DIBasicType::get(VMContext, dwarf::DW_TAG_unspecified_type, Name);
}

DIBasicType *DIBuilder::createNullPtrType() {
  return createUnspecifiedType("decltype(nullptr)");
}

DIBasicType *DIBuilder::createBasicType(StringRef Name, uint64_t SizeInBits,
                                        uint64_t AlignInBits,
                                        unsigned Encoding) {
  assert(!Name.empty() && "Unable to create type without name");
  return DIBasicType::get(VMContext, dwarf::DW_TAG_base_type, Name, SizeInBits,
                          AlignInBits, Encoding);
}

DIDerivedType *DIBuilder::createQualifiedType(unsigned Tag, DIType *FromTy) {
  return DIDerivedType::get(VMContext, Tag, "", nullptr, 0, nullptr,
                            DITypeRef::get(FromTy), 0, 0, 0, 0);
}

DIDerivedType *DIBuilder::createPointerType(DIType *PointeeTy,
                                            uint64_t SizeInBits,
                                            uint64_t AlignInBits,
                                            StringRef Name) {
  // FIXME: Why is there a name here?
  return DIDerivedType::get(VMContext, dwarf::DW_TAG_pointer_type, Name,
                            nullptr, 0, nullptr, DITypeRef::get(PointeeTy),
                            SizeInBits, AlignInBits, 0, 0);
}

DIDerivedType *DIBuilder::createMemberPointerType(DIType *PointeeTy,
                                                  DIType *Base,
                                                  uint64_t SizeInBits,
                                                  uint64_t AlignInBits) {
  return DIDerivedType::get(VMContext, dwarf::DW_TAG_ptr_to_member_type, "",
                            nullptr, 0, nullptr, DITypeRef::get(PointeeTy),
                            SizeInBits, AlignInBits, 0, 0,
                            DITypeRef::get(Base));
}

DIDerivedType *DIBuilder::createReferenceType(unsigned Tag, DIType *RTy) {
  assert(RTy && "Unable to create reference type");
  return DIDerivedType::get(VMContext, Tag, "", nullptr, 0, nullptr,
                            DITypeRef::get(RTy), 0, 0, 0, 0);
}

DIDerivedType *DIBuilder::createTypedef(DIType *Ty, StringRef Name,
                                        DIFile *File, unsigned LineNo,
                                        DIScope *Context) {
  return DIDerivedType::get(VMContext, dwarf::DW_TAG_typedef, Name, File,
                            LineNo,
                            DIScopeRef::get(getNonCompileUnitScope(Context)),
                            DITypeRef::get(Ty), 0, 0, 0, 0);
}

DIDerivedType *DIBuilder::createFriend(DIType *Ty, DIType *FriendTy) {
  assert(Ty && "Invalid type!");
  assert(FriendTy && "Invalid friend type!");
  return DIDerivedType::get(VMContext, dwarf::DW_TAG_friend, "", nullptr, 0,
                            DITypeRef::get(Ty), DITypeRef::get(FriendTy), 0, 0,
                            0, 0);
}

DIDerivedType *DIBuilder::createInheritance(DIType *Ty, DIType *BaseTy,
                                            uint64_t BaseOffset,
                                            unsigned Flags) {
  assert(Ty && "Unable to create inheritance");
  return DIDerivedType::get(VMContext, dwarf::DW_TAG_inheritance, "", nullptr,
                            0, DITypeRef::get(Ty), DITypeRef::get(BaseTy), 0, 0,
                            BaseOffset, Flags);
}

DIDerivedType *DIBuilder::createMemberType(DIScope *Scope, StringRef Name,
                                           DIFile *File, unsigned LineNumber,
                                           uint64_t SizeInBits,
                                           uint64_t AlignInBits,
                                           uint64_t OffsetInBits,
                                           unsigned Flags, DIType *Ty) {
  return DIDerivedType::get(
      VMContext, dwarf::DW_TAG_member, Name, File, LineNumber,
      DIScopeRef::get(getNonCompileUnitScope(Scope)), DITypeRef::get(Ty),
      SizeInBits, AlignInBits, OffsetInBits, Flags);
}

static ConstantAsMetadata *getConstantOrNull(Constant *C) {
  if (C)
    return ConstantAsMetadata::get(C);
  return nullptr;
}

DIDerivedType *DIBuilder::createStaticMemberType(DIScope *Scope, StringRef Name,
                                                 DIFile *File,
                                                 unsigned LineNumber,
                                                 DIType *Ty, unsigned Flags,
                                                 llvm::Constant *Val) {
  Flags |= DINode::FlagStaticMember;
  return DIDerivedType::get(
      VMContext, dwarf::DW_TAG_member, Name, File, LineNumber,
      DIScopeRef::get(getNonCompileUnitScope(Scope)), DITypeRef::get(Ty), 0, 0,
      0, Flags, getConstantOrNull(Val));
}

DIDerivedType *DIBuilder::createObjCIVar(StringRef Name, DIFile *File,
                                         unsigned LineNumber,
                                         uint64_t SizeInBits,
                                         uint64_t AlignInBits,
                                         uint64_t OffsetInBits, unsigned Flags,
                                         DIType *Ty, MDNode *PropertyNode) {
  return DIDerivedType::get(
      VMContext, dwarf::DW_TAG_member, Name, File, LineNumber,
      DIScopeRef::get(getNonCompileUnitScope(File)), DITypeRef::get(Ty),
      SizeInBits, AlignInBits, OffsetInBits, Flags, PropertyNode);
}

DIObjCProperty *
DIBuilder::createObjCProperty(StringRef Name, DIFile *File, unsigned LineNumber,
                              StringRef GetterName, StringRef SetterName,
                              unsigned PropertyAttributes, DIType *Ty) {
  return DIObjCProperty::get(VMContext, Name, File, LineNumber, GetterName,
                             SetterName, PropertyAttributes,
                             DITypeRef::get(Ty));
}

DITemplateTypeParameter *
DIBuilder::createTemplateTypeParameter(DIScope *Context, StringRef Name,
                                       DIType *Ty) {
  assert((!Context || isa<DICompileUnit>(Context)) && "Expected compile unit");
  return DITemplateTypeParameter::get(VMContext, Name, DITypeRef::get(Ty));
}

static DITemplateValueParameter *
createTemplateValueParameterHelper(LLVMContext &VMContext, unsigned Tag,
                                   DIScope *Context, StringRef Name, DIType *Ty,
                                   Metadata *MD) {
  assert((!Context || isa<DICompileUnit>(Context)) && "Expected compile unit");
  return DITemplateValueParameter::get(VMContext, Tag, Name, DITypeRef::get(Ty),
                                       MD);
}

DITemplateValueParameter *
DIBuilder::createTemplateValueParameter(DIScope *Context, StringRef Name,
                                        DIType *Ty, Constant *Val) {
  return createTemplateValueParameterHelper(
      VMContext, dwarf::DW_TAG_template_value_parameter, Context, Name, Ty,
      getConstantOrNull(Val));
}

DITemplateValueParameter *
DIBuilder::createTemplateTemplateParameter(DIScope *Context, StringRef Name,
                                           DIType *Ty, StringRef Val) {
  return createTemplateValueParameterHelper(
      VMContext, dwarf::DW_TAG_GNU_template_template_param, Context, Name, Ty,
      MDString::get(VMContext, Val));
}

DITemplateValueParameter *
DIBuilder::createTemplateParameterPack(DIScope *Context, StringRef Name,
                                       DIType *Ty, DINodeArray Val) {
  return createTemplateValueParameterHelper(
      VMContext, dwarf::DW_TAG_GNU_template_parameter_pack, Context, Name, Ty,
      Val.get());
}

DICompositeType *DIBuilder::createClassType(
    DIScope *Context, StringRef Name, DIFile *File, unsigned LineNumber,
    uint64_t SizeInBits, uint64_t AlignInBits, uint64_t OffsetInBits,
    unsigned Flags, DIType *DerivedFrom, DINodeArray Elements,
    DIType *VTableHolder, MDNode *TemplateParams, StringRef UniqueIdentifier) {
  assert((!Context || isa<DIScope>(Context)) &&
         "createClassType should be called with a valid Context");

  auto *R = DICompositeType::get(
      VMContext, dwarf::DW_TAG_structure_type, Name, File, LineNumber,
      DIScopeRef::get(getNonCompileUnitScope(Context)),
      DITypeRef::get(DerivedFrom), SizeInBits, AlignInBits, OffsetInBits, Flags,
      Elements, 0, DITypeRef::get(VTableHolder),
      cast_or_null<MDTuple>(TemplateParams), UniqueIdentifier);
  if (!UniqueIdentifier.empty())
    retainType(R);
  trackIfUnresolved(R);
  return R;
}

DICompositeType *DIBuilder::createStructType(
    DIScope *Context, StringRef Name, DIFile *File, unsigned LineNumber,
    uint64_t SizeInBits, uint64_t AlignInBits, unsigned Flags,
    DIType *DerivedFrom, DINodeArray Elements, unsigned RunTimeLang,
    DIType *VTableHolder, StringRef UniqueIdentifier) {
  auto *R = DICompositeType::get(
      VMContext, dwarf::DW_TAG_structure_type, Name, File, LineNumber,
      DIScopeRef::get(getNonCompileUnitScope(Context)),
      DITypeRef::get(DerivedFrom), SizeInBits, AlignInBits, 0, Flags, Elements,
      RunTimeLang, DITypeRef::get(VTableHolder), nullptr, UniqueIdentifier);
  if (!UniqueIdentifier.empty())
    retainType(R);
  trackIfUnresolved(R);
  return R;
}

DICompositeType *DIBuilder::createUnionType(
    DIScope *Scope, StringRef Name, DIFile *File, unsigned LineNumber,
    uint64_t SizeInBits, uint64_t AlignInBits, unsigned Flags,
    DINodeArray Elements, unsigned RunTimeLang, StringRef UniqueIdentifier) {
  auto *R = DICompositeType::get(
      VMContext, dwarf::DW_TAG_union_type, Name, File, LineNumber,
      DIScopeRef::get(getNonCompileUnitScope(Scope)), nullptr, SizeInBits,
      AlignInBits, 0, Flags, Elements, RunTimeLang, nullptr, nullptr,
      UniqueIdentifier);
  if (!UniqueIdentifier.empty())
    retainType(R);
  trackIfUnresolved(R);
  return R;
}

DISubroutineType *DIBuilder::createSubroutineType(DIFile *File,
                                                  DITypeRefArray ParameterTypes,
                                                  unsigned Flags) {
  return DISubroutineType::get(VMContext, Flags, ParameterTypes);
}

DICompositeType *DIBuilder::createEnumerationType(
    DIScope *Scope, StringRef Name, DIFile *File, unsigned LineNumber,
    uint64_t SizeInBits, uint64_t AlignInBits, DINodeArray Elements,
    DIType *UnderlyingType, StringRef UniqueIdentifier) {
  auto *CTy = DICompositeType::get(
      VMContext, dwarf::DW_TAG_enumeration_type, Name, File, LineNumber,
      DIScopeRef::get(getNonCompileUnitScope(Scope)),
      DITypeRef::get(UnderlyingType), SizeInBits, AlignInBits, 0, 0, Elements,
      0, nullptr, nullptr, UniqueIdentifier);
  AllEnumTypes.push_back(CTy);
  if (!UniqueIdentifier.empty())
    retainType(CTy);
  trackIfUnresolved(CTy);
  return CTy;
}

DICompositeType *DIBuilder::createArrayType(uint64_t Size, uint64_t AlignInBits,
                                            DIType *Ty,
                                            DINodeArray Subscripts) {
  auto *R = DICompositeType::get(VMContext, dwarf::DW_TAG_array_type, "",
                                 nullptr, 0, nullptr, DITypeRef::get(Ty), Size,
                                 AlignInBits, 0, 0, Subscripts, 0, nullptr);
  trackIfUnresolved(R);
  return R;
}

DICompositeType *DIBuilder::createVectorType(uint64_t Size,
                                             uint64_t AlignInBits, DIType *Ty,
                                             DINodeArray Subscripts) {
  auto *R =
      DICompositeType::get(VMContext, dwarf::DW_TAG_array_type, "", nullptr, 0,
                           nullptr, DITypeRef::get(Ty), Size, AlignInBits, 0,
                           DINode::FlagVector, Subscripts, 0, nullptr);
  trackIfUnresolved(R);
  return R;
}

static DIType *createTypeWithFlags(LLVMContext &Context, DIType *Ty,
                                   unsigned FlagsToSet) {
  auto NewTy = Ty->clone();
  NewTy->setFlags(NewTy->getFlags() | FlagsToSet);
  return MDNode::replaceWithUniqued(std::move(NewTy));
}

DIType *DIBuilder::createArtificialType(DIType *Ty) {
  // FIXME: Restrict this to the nodes where it's valid.
  if (Ty->isArtificial())
    return Ty;
  return createTypeWithFlags(VMContext, Ty, DINode::FlagArtificial);
}

DIType *DIBuilder::createObjectPointerType(DIType *Ty) {
  // FIXME: Restrict this to the nodes where it's valid.
  if (Ty->isObjectPointer())
    return Ty;
  unsigned Flags = DINode::FlagObjectPointer | DINode::FlagArtificial;
  return createTypeWithFlags(VMContext, Ty, Flags);
}

void DIBuilder::retainType(DIType *T) {
  assert(T && "Expected non-null type");
  AllRetainTypes.emplace_back(T);
}

DIBasicType *DIBuilder::createUnspecifiedParameter() { return nullptr; }

DICompositeType *
DIBuilder::createForwardDecl(unsigned Tag, StringRef Name, DIScope *Scope,
                             DIFile *F, unsigned Line, unsigned RuntimeLang,
                             uint64_t SizeInBits, uint64_t AlignInBits,
                             StringRef UniqueIdentifier) {
  // FIXME: Define in terms of createReplaceableForwardDecl() by calling
  // replaceWithUniqued().
  auto *RetTy = DICompositeType::get(
      VMContext, Tag, Name, F, Line,
      DIScopeRef::get(getNonCompileUnitScope(Scope)), nullptr, SizeInBits,
      AlignInBits, 0, DINode::FlagFwdDecl, nullptr, RuntimeLang, nullptr,
      nullptr, UniqueIdentifier);
  if (!UniqueIdentifier.empty())
    retainType(RetTy);
  trackIfUnresolved(RetTy);
  return RetTy;
}

DICompositeType *DIBuilder::createReplaceableCompositeType(
    unsigned Tag, StringRef Name, DIScope *Scope, DIFile *F, unsigned Line,
    unsigned RuntimeLang, uint64_t SizeInBits, uint64_t AlignInBits,
    unsigned Flags, StringRef UniqueIdentifier) {
  auto *RetTy = DICompositeType::getTemporary(
                    VMContext, Tag, Name, F, Line,
                    DIScopeRef::get(getNonCompileUnitScope(Scope)), nullptr,
                    SizeInBits, AlignInBits, 0, Flags, nullptr, RuntimeLang,
                    nullptr, nullptr, UniqueIdentifier)
                    .release();
  if (!UniqueIdentifier.empty())
    retainType(RetTy);
  trackIfUnresolved(RetTy);
  return RetTy;
}

DINodeArray DIBuilder::getOrCreateArray(ArrayRef<Metadata *> Elements) {
  return MDTuple::get(VMContext, Elements);
}

DITypeRefArray DIBuilder::getOrCreateTypeArray(ArrayRef<Metadata *> Elements) {
  SmallVector<llvm::Metadata *, 16> Elts;
  for (unsigned i = 0, e = Elements.size(); i != e; ++i) {
    if (Elements[i] && isa<MDNode>(Elements[i]))
      Elts.push_back(DITypeRef::get(cast<DIType>(Elements[i])));
    else
      Elts.push_back(Elements[i]);
  }
  return DITypeRefArray(MDNode::get(VMContext, Elts));
}

DISubrange *DIBuilder::getOrCreateSubrange(int64_t Lo, int64_t Count) {
  return DISubrange::get(VMContext, Count, Lo);
}

static void checkGlobalVariableScope(DIScope *Context) {
#ifndef NDEBUG
  if (auto *CT =
          dyn_cast_or_null<DICompositeType>(getNonCompileUnitScope(Context)))
    assert(CT->getIdentifier().empty() &&
           "Context of a global variable should not be a type with identifier");
#endif
}

DIGlobalVariable *DIBuilder::createGlobalVariable(
    DIScope *Context, StringRef Name, StringRef LinkageName, DIFile *F,
    unsigned LineNumber, DIType *Ty, bool isLocalToUnit, Constant *Val,
    MDNode *Decl) {
  checkGlobalVariableScope(Context);

  auto *N = DIGlobalVariable::get(VMContext, cast_or_null<DIScope>(Context),
                                  Name, LinkageName, F, LineNumber,
                                  DITypeRef::get(Ty), isLocalToUnit, true, Val,
                                  cast_or_null<DIDerivedType>(Decl));
  AllGVs.push_back(N);
  return N;
}

DIGlobalVariable *DIBuilder::createTempGlobalVariableFwdDecl(
    DIScope *Context, StringRef Name, StringRef LinkageName, DIFile *F,
    unsigned LineNumber, DIType *Ty, bool isLocalToUnit, Constant *Val,
    MDNode *Decl) {
  checkGlobalVariableScope(Context);

  return DIGlobalVariable::getTemporary(
             VMContext, cast_or_null<DIScope>(Context), Name, LinkageName, F,
             LineNumber, DITypeRef::get(Ty), isLocalToUnit, false, Val,
             cast_or_null<DIDerivedType>(Decl))
      .release();
}

DILocalVariable *DIBuilder::createLocalVariable(
    unsigned Tag, DIScope *Scope, StringRef Name, DIFile *File, unsigned LineNo,
    DIType *Ty, bool AlwaysPreserve, unsigned Flags, unsigned ArgNo) {
  // FIXME: Why getNonCompileUnitScope()?
  // FIXME: Why is "!Context" okay here?
  // FIXME: Why doesn't this check for a subprogram or lexical block (AFAICT
  // the only valid scopes)?
  DIScope *Context = getNonCompileUnitScope(Scope);

  auto *Node = DILocalVariable::get(
      VMContext, Tag, cast_or_null<DILocalScope>(Context), Name, File, LineNo,
      DITypeRef::get(Ty), ArgNo, Flags);
  if (AlwaysPreserve) {
    // The optimizer may remove local variables. If there is an interest
    // to preserve variable info in such situation then stash it in a
    // named mdnode.
    DISubprogram *Fn = getDISubprogram(Scope);
    assert(Fn && "Missing subprogram for local variable");
    PreservedVariables[Fn].emplace_back(Node);
  }
  return Node;
}

DIExpression *DIBuilder::createExpression(ArrayRef<uint64_t> Addr) {
  return DIExpression::get(VMContext, Addr);
}

DIExpression *DIBuilder::createExpression(ArrayRef<int64_t> Signed) {
  // TODO: Remove the callers of this signed version and delete.
  SmallVector<uint64_t, 8> Addr(Signed.begin(), Signed.end());
  return createExpression(Addr);
}

// HLSL Begin Change: Match -InBits suffixes from header
DIExpression *DIBuilder::createBitPieceExpression(unsigned OffsetInBits, 
                                                  unsigned SizeInBits) {
  uint64_t Addr[] = {dwarf::DW_OP_bit_piece, OffsetInBits, SizeInBits};
// HLSL End Change
  return DIExpression::get(VMContext, Addr);
}

DISubprogram *DIBuilder::createFunction(DIScopeRef Context, StringRef Name,
                                        StringRef LinkageName, DIFile *File,
                                        unsigned LineNo, DISubroutineType *Ty,
                                        bool isLocalToUnit, bool isDefinition,
                                        unsigned ScopeLine, unsigned Flags,
                                        bool isOptimized, Function *Fn,
                                        MDNode *TParams, MDNode *Decl) {
  // dragonegg does not generate identifier for types, so using an empty map
  // to resolve the context should be fine.
  DITypeIdentifierMap EmptyMap;
  return createFunction(Context.resolve(EmptyMap), Name, LinkageName, File,
                        LineNo, Ty, isLocalToUnit, isDefinition, ScopeLine,
                        Flags, isOptimized, Fn, TParams, Decl);
}

DISubprogram *DIBuilder::createFunction(DIScope *Context, StringRef Name,
                                        StringRef LinkageName, DIFile *File,
                                        unsigned LineNo, DISubroutineType *Ty,
                                        bool isLocalToUnit, bool isDefinition,
                                        unsigned ScopeLine, unsigned Flags,
                                        bool isOptimized, Function *Fn,
                                        MDNode *TParams, MDNode *Decl) {
  assert(Ty->getTag() == dwarf::DW_TAG_subroutine_type &&
         "function types should be subroutines");
  auto *Node = DISubprogram::get(
      VMContext, DIScopeRef::get(getNonCompileUnitScope(Context)), Name,
      LinkageName, File, LineNo, Ty, isLocalToUnit, isDefinition, ScopeLine,
      nullptr, 0, 0, Flags, isOptimized, Fn, cast_or_null<MDTuple>(TParams),
      cast_or_null<DISubprogram>(Decl),
      nullptr); // HLSL Change - this leaked, better nullptr than empty anyway - MDTuple::getTemporary(VMContext, None).release());

  if (isDefinition)
    AllSubprograms.push_back(Node);
  trackIfUnresolved(Node);
  return Node;
}

DISubprogram *DIBuilder::createTempFunctionFwdDecl(
    DIScope *Context, StringRef Name, StringRef LinkageName, DIFile *File,
    unsigned LineNo, DISubroutineType *Ty, bool isLocalToUnit,
    bool isDefinition, unsigned ScopeLine, unsigned Flags, bool isOptimized,
    Function *Fn, MDNode *TParams, MDNode *Decl) {
  return DISubprogram::getTemporary(
             VMContext, DIScopeRef::get(getNonCompileUnitScope(Context)), Name,
             LinkageName, File, LineNo, Ty, isLocalToUnit, isDefinition,
             ScopeLine, nullptr, 0, 0, Flags, isOptimized, Fn,
             cast_or_null<MDTuple>(TParams), cast_or_null<DISubprogram>(Decl),
             nullptr)
      .release();
}

DISubprogram *
DIBuilder::createMethod(DIScope *Context, StringRef Name, StringRef LinkageName,
                        DIFile *F, unsigned LineNo, DISubroutineType *Ty,
                        bool isLocalToUnit, bool isDefinition, unsigned VK,
                        unsigned VIndex, DIType *VTableHolder, unsigned Flags,
                        bool isOptimized, Function *Fn, MDNode *TParam) {
  assert(Ty->getTag() == dwarf::DW_TAG_subroutine_type &&
         "function types should be subroutines");
  assert(getNonCompileUnitScope(Context) &&
         "Methods should have both a Context and a context that isn't "
         "the compile unit.");
  // FIXME: Do we want to use different scope/lines?
  auto *SP = DISubprogram::get(
      VMContext, DIScopeRef::get(cast<DIScope>(Context)), Name, LinkageName, F,
      LineNo, Ty, isLocalToUnit, isDefinition, LineNo,
      DITypeRef::get(VTableHolder), VK, VIndex, Flags, isOptimized, Fn,
      cast_or_null<MDTuple>(TParam), nullptr, nullptr);

  if (isDefinition)
    AllSubprograms.push_back(SP);
  trackIfUnresolved(SP);
  return SP;
}

DINamespace *DIBuilder::createNameSpace(DIScope *Scope, StringRef Name,
                                        DIFile *File, unsigned LineNo) {
  return DINamespace::get(VMContext, getNonCompileUnitScope(Scope), File, Name,
                          LineNo);
}

DIModule *DIBuilder::createModule(DIScope *Scope, StringRef Name,
                                  StringRef ConfigurationMacros,
                                  StringRef IncludePath,
                                  StringRef ISysRoot) {
 return DIModule::get(VMContext, getNonCompileUnitScope(Scope), Name,
                      ConfigurationMacros, IncludePath, ISysRoot);
}

DILexicalBlockFile *DIBuilder::createLexicalBlockFile(DIScope *Scope,
                                                      DIFile *File,
                                                      unsigned Discriminator) {
  return DILexicalBlockFile::get(VMContext, Scope, File, Discriminator);
}

DILexicalBlock *DIBuilder::createLexicalBlock(DIScope *Scope, DIFile *File,
                                              unsigned Line, unsigned Col) {
  // Make these distinct, to avoid merging two lexical blocks on the same
  // file/line/column.
  return DILexicalBlock::getDistinct(VMContext, getNonCompileUnitScope(Scope),
                                     File, Line, Col);
}

static Value *getDbgIntrinsicValueImpl(LLVMContext &VMContext, Value *V) {
  assert(V && "no value passed to dbg intrinsic");
  return MetadataAsValue::get(VMContext, ValueAsMetadata::get(V));
}

static Instruction *withDebugLoc(Instruction *I, const DILocation *DL) {
  I->setDebugLoc(const_cast<DILocation *>(DL));
  return I;
}

Instruction *DIBuilder::insertDeclare(Value *Storage, DILocalVariable *VarInfo,
                                      DIExpression *Expr, const DILocation *DL,
                                      Instruction *InsertBefore) {
  assert(VarInfo && "empty or invalid DILocalVariable* passed to dbg.declare");
  assert(DL && "Expected debug loc");
  assert(DL->getScope()->getSubprogram() ==
             VarInfo->getScope()->getSubprogram() &&
         "Expected matching subprograms");
  if (!DeclareFn)
    DeclareFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_declare);

  trackIfUnresolved(VarInfo);
  trackIfUnresolved(Expr);
  Value *Args[] = {getDbgIntrinsicValueImpl(VMContext, Storage),
                   MetadataAsValue::get(VMContext, VarInfo),
                   MetadataAsValue::get(VMContext, Expr)};
  return withDebugLoc(CallInst::Create(DeclareFn, Args, "", InsertBefore), DL);
}

Instruction *DIBuilder::insertDeclare(Value *Storage, DILocalVariable *VarInfo,
                                      DIExpression *Expr, const DILocation *DL,
                                      BasicBlock *InsertAtEnd) {
  assert(VarInfo && "empty or invalid DILocalVariable* passed to dbg.declare");
  assert(DL && "Expected debug loc");
  assert(DL->getScope()->getSubprogram() ==
             VarInfo->getScope()->getSubprogram() &&
         "Expected matching subprograms");
  if (!DeclareFn)
    DeclareFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_declare);

  trackIfUnresolved(VarInfo);
  trackIfUnresolved(Expr);
  Value *Args[] = {getDbgIntrinsicValueImpl(VMContext, Storage),
                   MetadataAsValue::get(VMContext, VarInfo),
                   MetadataAsValue::get(VMContext, Expr)};

  // If this block already has a terminator then insert this intrinsic
  // before the terminator.
  if (TerminatorInst *T = InsertAtEnd->getTerminator())
    return withDebugLoc(CallInst::Create(DeclareFn, Args, "", T), DL);
  return withDebugLoc(CallInst::Create(DeclareFn, Args, "", InsertAtEnd), DL);
}

Instruction *DIBuilder::insertDbgValueIntrinsic(Value *V, uint64_t Offset,
                                                DILocalVariable *VarInfo,
                                                DIExpression *Expr,
                                                const DILocation *DL,
                                                Instruction *InsertBefore) {
  assert(V && "no value passed to dbg.value");
  assert(VarInfo && "empty or invalid DILocalVariable* passed to dbg.value");
  assert(DL && "Expected debug loc");
  assert(DL->getScope()->getSubprogram() ==
             VarInfo->getScope()->getSubprogram() &&
         "Expected matching subprograms");
  if (!ValueFn)
    ValueFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_value);

  trackIfUnresolved(VarInfo);
  trackIfUnresolved(Expr);
  Value *Args[] = {getDbgIntrinsicValueImpl(VMContext, V),
                   ConstantInt::get(Type::getInt64Ty(VMContext), Offset),
                   MetadataAsValue::get(VMContext, VarInfo),
                   MetadataAsValue::get(VMContext, Expr)};
  return withDebugLoc(CallInst::Create(ValueFn, Args, "", InsertBefore), DL);
}

Instruction *DIBuilder::insertDbgValueIntrinsic(Value *V, uint64_t Offset,
                                                DILocalVariable *VarInfo,
                                                DIExpression *Expr,
                                                const DILocation *DL,
                                                BasicBlock *InsertAtEnd) {
  assert(V && "no value passed to dbg.value");
  assert(VarInfo && "empty or invalid DILocalVariable* passed to dbg.value");
  assert(DL && "Expected debug loc");
  assert(DL->getScope()->getSubprogram() ==
             VarInfo->getScope()->getSubprogram() &&
         "Expected matching subprograms");
  if (!ValueFn)
    ValueFn = Intrinsic::getDeclaration(&M, Intrinsic::dbg_value);

  trackIfUnresolved(VarInfo);
  trackIfUnresolved(Expr);
  Value *Args[] = {getDbgIntrinsicValueImpl(VMContext, V),
                   ConstantInt::get(Type::getInt64Ty(VMContext), Offset),
                   MetadataAsValue::get(VMContext, VarInfo),
                   MetadataAsValue::get(VMContext, Expr)};

  return withDebugLoc(CallInst::Create(ValueFn, Args, "", InsertAtEnd), DL);
}

void DIBuilder::replaceVTableHolder(DICompositeType *&T,
                                    DICompositeType *VTableHolder) {
  {
    TypedTrackingMDRef<DICompositeType> N(T);
    N->replaceVTableHolder(DITypeRef::get(VTableHolder));
    T = N.get();
  }

  // If this didn't create a self-reference, just return.
  if (T != VTableHolder)
    return;

  // Look for unresolved operands.  T will drop RAUW support, orphaning any
  // cycles underneath it.
  if (T->isResolved())
    for (const MDOperand &O : T->operands())
      if (auto *N = dyn_cast_or_null<MDNode>(O))
        trackIfUnresolved(N);
}

void DIBuilder::replaceArrays(DICompositeType *&T, DINodeArray Elements,
                              DINodeArray TParams) {
  {
    TypedTrackingMDRef<DICompositeType> N(T);
    if (Elements)
      N->replaceElements(Elements);
    if (TParams)
      N->replaceTemplateParams(DITemplateParameterArray(TParams));
    T = N.get();
  }

  // If T isn't resolved, there's no problem.
  if (!T->isResolved())
    return;

  // If T is resolved, it may be due to a self-reference cycle.  Track the
  // arrays explicitly if they're unresolved, or else the cycles will be
  // orphaned.
  if (Elements)
    trackIfUnresolved(Elements.get());
  if (TParams)
    trackIfUnresolved(TParams.get());
}
