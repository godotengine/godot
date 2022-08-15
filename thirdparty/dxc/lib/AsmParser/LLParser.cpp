//===-- LLParser.cpp - Parser Class ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the parser class for .ll files.
//
//===----------------------------------------------------------------------===//

#include "LLParser.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/AsmParser/SlotMapping.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

static std::string getTypeString(Type *T) {
  std::string Result;
  raw_string_ostream Tmp(Result);
  Tmp << *T;
  return Tmp.str();
}

/// Run: module ::= toplevelentity*
bool LLParser::Run() {
  // Prime the lexer.
  Lex.Lex();

  return ParseTopLevelEntities() ||
         ValidateEndOfModule();
}

/// ValidateEndOfModule - Do final validity and sanity checks at the end of the
/// module.
bool LLParser::ValidateEndOfModule() {
  for (unsigned I = 0, E = InstsWithTBAATag.size(); I < E; I++)
    UpgradeInstWithTBAATag(InstsWithTBAATag[I]);

  // Handle any function attribute group forward references.
  for (std::map<Value*, std::vector<unsigned> >::iterator
         I = ForwardRefAttrGroups.begin(), E = ForwardRefAttrGroups.end();
         I != E; ++I) {
    Value *V = I->first;
    std::vector<unsigned> &Vec = I->second;
    AttrBuilder B;

    for (std::vector<unsigned>::iterator VI = Vec.begin(), VE = Vec.end();
         VI != VE; ++VI)
      B.merge(NumberedAttrBuilders[*VI]);

    if (Function *Fn = dyn_cast<Function>(V)) {
      AttributeSet AS = Fn->getAttributes();
      AttrBuilder FnAttrs(AS.getFnAttributes(), AttributeSet::FunctionIndex);
      AS = AS.removeAttributes(Context, AttributeSet::FunctionIndex,
                               AS.getFnAttributes());

      FnAttrs.merge(B);

      // If the alignment was parsed as an attribute, move to the alignment
      // field.
      if (FnAttrs.hasAlignmentAttr()) {
        Fn->setAlignment(FnAttrs.getAlignment());
        FnAttrs.removeAttribute(Attribute::Alignment);
      }

      AS = AS.addAttributes(Context, AttributeSet::FunctionIndex,
                            AttributeSet::get(Context,
                                              AttributeSet::FunctionIndex,
                                              FnAttrs));
      Fn->setAttributes(AS);
    } else if (CallInst *CI = dyn_cast<CallInst>(V)) {
      AttributeSet AS = CI->getAttributes();
      AttrBuilder FnAttrs(AS.getFnAttributes(), AttributeSet::FunctionIndex);
      AS = AS.removeAttributes(Context, AttributeSet::FunctionIndex,
                               AS.getFnAttributes());
      FnAttrs.merge(B);
      AS = AS.addAttributes(Context, AttributeSet::FunctionIndex,
                            AttributeSet::get(Context,
                                              AttributeSet::FunctionIndex,
                                              FnAttrs));
      CI->setAttributes(AS);
    } else if (InvokeInst *II = dyn_cast<InvokeInst>(V)) {
      AttributeSet AS = II->getAttributes();
      AttrBuilder FnAttrs(AS.getFnAttributes(), AttributeSet::FunctionIndex);
      AS = AS.removeAttributes(Context, AttributeSet::FunctionIndex,
                               AS.getFnAttributes());
      FnAttrs.merge(B);
      AS = AS.addAttributes(Context, AttributeSet::FunctionIndex,
                            AttributeSet::get(Context,
                                              AttributeSet::FunctionIndex,
                                              FnAttrs));
      II->setAttributes(AS);
    } else {
      llvm_unreachable("invalid object with forward attribute group reference");
    }
  }

  // If there are entries in ForwardRefBlockAddresses at this point, the
  // function was never defined.
  if (!ForwardRefBlockAddresses.empty())
    return Error(ForwardRefBlockAddresses.begin()->first.Loc,
                 "expected function name in blockaddress");

  for (const auto &NT : NumberedTypes)
    if (NT.second.second.isValid())
      return Error(NT.second.second,
                   "use of undefined type '%" + Twine(NT.first) + "'");

  for (StringMap<std::pair<Type*, LocTy> >::iterator I =
       NamedTypes.begin(), E = NamedTypes.end(); I != E; ++I)
    if (I->second.second.isValid())
      return Error(I->second.second,
                   "use of undefined type named '" + I->getKey() + "'");

  if (!ForwardRefComdats.empty())
    return Error(ForwardRefComdats.begin()->second,
                 "use of undefined comdat '$" +
                     ForwardRefComdats.begin()->first + "'");

  if (!ForwardRefVals.empty())
    return Error(ForwardRefVals.begin()->second.second,
                 "use of undefined value '@" + ForwardRefVals.begin()->first +
                 "'");

  if (!ForwardRefValIDs.empty())
    return Error(ForwardRefValIDs.begin()->second.second,
                 "use of undefined value '@" +
                 Twine(ForwardRefValIDs.begin()->first) + "'");

  if (!ForwardRefMDNodes.empty())
    return Error(ForwardRefMDNodes.begin()->second.second,
                 "use of undefined metadata '!" +
                 Twine(ForwardRefMDNodes.begin()->first) + "'");

  // Resolve metadata cycles.
  for (auto &N : NumberedMetadata) {
    if (N.second && !N.second->isResolved())
      N.second->resolveCycles();
  }

  // Look for intrinsic functions and CallInst that need to be upgraded
  for (Module::iterator FI = M->begin(), FE = M->end(); FI != FE; )
    UpgradeCallsToIntrinsic(FI++); // must be post-increment, as we remove

  UpgradeDebugInfo(*M);

  if (!Slots)
    return false;
  // Initialize the slot mapping.
  // Because by this point we've parsed and validated everything, we can "steal"
  // the mapping from LLParser as it doesn't need it anymore.
  Slots->GlobalValues = std::move(NumberedVals);
  Slots->MetadataNodes = std::move(NumberedMetadata);

  return false;
}

//===----------------------------------------------------------------------===//
// Top-Level Entities
//===----------------------------------------------------------------------===//

bool LLParser::ParseTopLevelEntities() {
  while (1) {
    switch (Lex.getKind()) {
    default:         return TokError("expected top-level entity");
    case lltok::Eof: return false;
    case lltok::kw_declare: if (ParseDeclare()) return true; break;
    case lltok::kw_define:  if (ParseDefine()) return true; break;
    case lltok::kw_module:  if (ParseModuleAsm()) return true; break;
    case lltok::kw_target:  if (ParseTargetDefinition()) return true; break;
    case lltok::kw_deplibs: if (ParseDepLibs()) return true; break;
    case lltok::LocalVarID: if (ParseUnnamedType()) return true; break;
    case lltok::LocalVar:   if (ParseNamedType()) return true; break;
    case lltok::GlobalID:   if (ParseUnnamedGlobal()) return true; break;
    case lltok::GlobalVar:  if (ParseNamedGlobal()) return true; break;
    case lltok::ComdatVar:  if (parseComdat()) return true; break;
    case lltok::exclaim:    if (ParseStandaloneMetadata()) return true; break;
    case lltok::MetadataVar:if (ParseNamedMetadata()) return true; break;

    // The Global variable production with no name can have many different
    // optional leading prefixes, the production is:
    // GlobalVar ::= OptionalLinkage OptionalVisibility OptionalDLLStorageClass
    //               OptionalThreadLocal OptionalAddrSpace OptionalUnnamedAddr
    //               ('constant'|'global') ...
    case lltok::kw_private:             // OptionalLinkage
    case lltok::kw_internal:            // OptionalLinkage
    case lltok::kw_weak:                // OptionalLinkage
    case lltok::kw_weak_odr:            // OptionalLinkage
    case lltok::kw_linkonce:            // OptionalLinkage
    case lltok::kw_linkonce_odr:        // OptionalLinkage
    case lltok::kw_appending:           // OptionalLinkage
    case lltok::kw_common:              // OptionalLinkage
    case lltok::kw_extern_weak:         // OptionalLinkage
    case lltok::kw_external:            // OptionalLinkage
    case lltok::kw_default:             // OptionalVisibility
    case lltok::kw_hidden:              // OptionalVisibility
    case lltok::kw_protected:           // OptionalVisibility
    case lltok::kw_dllimport:           // OptionalDLLStorageClass
    case lltok::kw_dllexport:           // OptionalDLLStorageClass
    case lltok::kw_thread_local:        // OptionalThreadLocal
    case lltok::kw_addrspace:           // OptionalAddrSpace
    case lltok::kw_constant:            // GlobalType
    case lltok::kw_global: {            // GlobalType
      unsigned Linkage, Visibility, DLLStorageClass;
      bool UnnamedAddr;
      GlobalVariable::ThreadLocalMode TLM;
      bool HasLinkage;
      if (ParseOptionalLinkage(Linkage, HasLinkage) ||
          ParseOptionalVisibility(Visibility) ||
          ParseOptionalDLLStorageClass(DLLStorageClass) ||
          ParseOptionalThreadLocal(TLM) ||
          parseOptionalUnnamedAddr(UnnamedAddr) ||
          ParseGlobal("", SMLoc(), Linkage, HasLinkage, Visibility,
                      DLLStorageClass, TLM, UnnamedAddr))
        return true;
      break;
    }

    case lltok::kw_attributes: if (ParseUnnamedAttrGrp()) return true; break;
    case lltok::kw_uselistorder: if (ParseUseListOrder()) return true; break;
    case lltok::kw_uselistorder_bb:
                                 if (ParseUseListOrderBB()) return true; break;
    }
  }
}


/// toplevelentity
///   ::= 'module' 'asm' STRINGCONSTANT
bool LLParser::ParseModuleAsm() {
  assert(Lex.getKind() == lltok::kw_module);
  Lex.Lex();

  std::string AsmStr;
  if (ParseToken(lltok::kw_asm, "expected 'module asm'") ||
      ParseStringConstant(AsmStr)) return true;

  M->appendModuleInlineAsm(AsmStr);
  return false;
}

/// toplevelentity
///   ::= 'target' 'triple' '=' STRINGCONSTANT
///   ::= 'target' 'datalayout' '=' STRINGCONSTANT
bool LLParser::ParseTargetDefinition() {
  assert(Lex.getKind() == lltok::kw_target);
  std::string Str;
  switch (Lex.Lex()) {
  default: return TokError("unknown target property");
  case lltok::kw_triple:
    Lex.Lex();
    if (ParseToken(lltok::equal, "expected '=' after target triple") ||
        ParseStringConstant(Str))
      return true;
    M->setTargetTriple(Str);
    return false;
  case lltok::kw_datalayout:
    Lex.Lex();
    if (ParseToken(lltok::equal, "expected '=' after target datalayout") ||
        ParseStringConstant(Str))
      return true;
    M->setDataLayout(Str);
    return false;
  }
}

/// toplevelentity
///   ::= 'deplibs' '=' '[' ']'
///   ::= 'deplibs' '=' '[' STRINGCONSTANT (',' STRINGCONSTANT)* ']'
/// FIXME: Remove in 4.0. Currently parse, but ignore.
bool LLParser::ParseDepLibs() {
  assert(Lex.getKind() == lltok::kw_deplibs);
  Lex.Lex();
  if (ParseToken(lltok::equal, "expected '=' after deplibs") ||
      ParseToken(lltok::lsquare, "expected '=' after deplibs"))
    return true;

  if (EatIfPresent(lltok::rsquare))
    return false;

  do {
    std::string Str;
    if (ParseStringConstant(Str)) return true;
  } while (EatIfPresent(lltok::comma));

  return ParseToken(lltok::rsquare, "expected ']' at end of list");
}

/// ParseUnnamedType:
///   ::= LocalVarID '=' 'type' type
bool LLParser::ParseUnnamedType() {
  LocTy TypeLoc = Lex.getLoc();
  unsigned TypeID = Lex.getUIntVal();
  Lex.Lex(); // eat LocalVarID;

  if (ParseToken(lltok::equal, "expected '=' after name") ||
      ParseToken(lltok::kw_type, "expected 'type' after '='"))
    return true;

  Type *Result = nullptr;
  if (ParseStructDefinition(TypeLoc, "",
                            NumberedTypes[TypeID], Result)) return true;

  if (!isa<StructType>(Result)) {
    std::pair<Type*, LocTy> &Entry = NumberedTypes[TypeID];
    if (Entry.first)
      return Error(TypeLoc, "non-struct types may not be recursive");
    Entry.first = Result;
    Entry.second = SMLoc();
  }

  return false;
}


/// toplevelentity
///   ::= LocalVar '=' 'type' type
bool LLParser::ParseNamedType() {
  std::string Name = Lex.getStrVal();
  LocTy NameLoc = Lex.getLoc();
  Lex.Lex();  // eat LocalVar.

  if (ParseToken(lltok::equal, "expected '=' after name") ||
      ParseToken(lltok::kw_type, "expected 'type' after name"))
    return true;

  Type *Result = nullptr;
  if (ParseStructDefinition(NameLoc, Name,
                            NamedTypes[Name], Result)) return true;

  if (!isa<StructType>(Result)) {
    std::pair<Type*, LocTy> &Entry = NamedTypes[Name];
    if (Entry.first)
      return Error(NameLoc, "non-struct types may not be recursive");
    Entry.first = Result;
    Entry.second = SMLoc();
  }

  return false;
}


/// toplevelentity
///   ::= 'declare' FunctionHeader
bool LLParser::ParseDeclare() {
  assert(Lex.getKind() == lltok::kw_declare);
  Lex.Lex();

  Function *F;
  return ParseFunctionHeader(F, false);
}

/// toplevelentity
///   ::= 'define' FunctionHeader (!dbg !56)* '{' ...
bool LLParser::ParseDefine() {
  assert(Lex.getKind() == lltok::kw_define);
  Lex.Lex();

  Function *F;
  return ParseFunctionHeader(F, true) ||
         ParseOptionalFunctionMetadata(*F) ||
         ParseFunctionBody(*F);
}

/// ParseGlobalType
///   ::= 'constant'
///   ::= 'global'
bool LLParser::ParseGlobalType(bool &IsConstant) {
  if (Lex.getKind() == lltok::kw_constant)
    IsConstant = true;
  else if (Lex.getKind() == lltok::kw_global)
    IsConstant = false;
  else {
    IsConstant = false;
    return TokError("expected 'global' or 'constant'");
  }
  Lex.Lex();
  return false;
}

/// ParseUnnamedGlobal:
///   OptionalVisibility ALIAS ...
///   OptionalLinkage OptionalVisibility OptionalDLLStorageClass
///                                                     ...   -> global variable
///   GlobalID '=' OptionalVisibility ALIAS ...
///   GlobalID '=' OptionalLinkage OptionalVisibility OptionalDLLStorageClass
///                                                     ...   -> global variable
bool LLParser::ParseUnnamedGlobal() {
  unsigned VarID = NumberedVals.size();
  std::string Name;
  LocTy NameLoc = Lex.getLoc();

  // Handle the GlobalID form.
  if (Lex.getKind() == lltok::GlobalID) {
    if (Lex.getUIntVal() != VarID)
      return Error(Lex.getLoc(), "variable expected to be numbered '%" +
                   Twine(VarID) + "'");
    Lex.Lex(); // eat GlobalID;

    if (ParseToken(lltok::equal, "expected '=' after name"))
      return true;
  }

  bool HasLinkage;
  unsigned Linkage, Visibility, DLLStorageClass;
  GlobalVariable::ThreadLocalMode TLM;
  bool UnnamedAddr;
  if (ParseOptionalLinkage(Linkage, HasLinkage) ||
      ParseOptionalVisibility(Visibility) ||
      ParseOptionalDLLStorageClass(DLLStorageClass) ||
      ParseOptionalThreadLocal(TLM) ||
      parseOptionalUnnamedAddr(UnnamedAddr))
    return true;

  if (Lex.getKind() != lltok::kw_alias)
    return ParseGlobal(Name, NameLoc, Linkage, HasLinkage, Visibility,
                       DLLStorageClass, TLM, UnnamedAddr);
  return ParseAlias(Name, NameLoc, Linkage, Visibility, DLLStorageClass, TLM,
                    UnnamedAddr);
}

/// ParseNamedGlobal:
///   GlobalVar '=' OptionalVisibility ALIAS ...
///   GlobalVar '=' OptionalLinkage OptionalVisibility OptionalDLLStorageClass
///                                                     ...   -> global variable
bool LLParser::ParseNamedGlobal() {
  assert(Lex.getKind() == lltok::GlobalVar);
  LocTy NameLoc = Lex.getLoc();
  std::string Name = Lex.getStrVal();
  Lex.Lex();

  bool HasLinkage;
  unsigned Linkage, Visibility, DLLStorageClass;
  GlobalVariable::ThreadLocalMode TLM;
  bool UnnamedAddr;
  if (ParseToken(lltok::equal, "expected '=' in global variable") ||
      ParseOptionalLinkage(Linkage, HasLinkage) ||
      ParseOptionalVisibility(Visibility) ||
      ParseOptionalDLLStorageClass(DLLStorageClass) ||
      ParseOptionalThreadLocal(TLM) ||
      parseOptionalUnnamedAddr(UnnamedAddr))
    return true;

  if (Lex.getKind() != lltok::kw_alias)
    return ParseGlobal(Name, NameLoc, Linkage, HasLinkage, Visibility,
                       DLLStorageClass, TLM, UnnamedAddr);

  return ParseAlias(Name, NameLoc, Linkage, Visibility, DLLStorageClass, TLM,
                    UnnamedAddr);
}

bool LLParser::parseComdat() {
  assert(Lex.getKind() == lltok::ComdatVar);
  std::string Name = Lex.getStrVal();
  LocTy NameLoc = Lex.getLoc();
  Lex.Lex();

  if (ParseToken(lltok::equal, "expected '=' here"))
    return true;

  if (ParseToken(lltok::kw_comdat, "expected comdat keyword"))
    return TokError("expected comdat type");

  Comdat::SelectionKind SK;
  switch (Lex.getKind()) {
  default:
    return TokError("unknown selection kind");
  case lltok::kw_any:
    SK = Comdat::Any;
    break;
  case lltok::kw_exactmatch:
    SK = Comdat::ExactMatch;
    break;
  case lltok::kw_largest:
    SK = Comdat::Largest;
    break;
  case lltok::kw_noduplicates:
    SK = Comdat::NoDuplicates;
    break;
  case lltok::kw_samesize:
    SK = Comdat::SameSize;
    break;
  }
  Lex.Lex();

  // See if the comdat was forward referenced, if so, use the comdat.
  Module::ComdatSymTabType &ComdatSymTab = M->getComdatSymbolTable();
  Module::ComdatSymTabType::iterator I = ComdatSymTab.find(Name);
  if (I != ComdatSymTab.end() && !ForwardRefComdats.erase(Name))
    return Error(NameLoc, "redefinition of comdat '$" + Name + "'");

  Comdat *C;
  if (I != ComdatSymTab.end())
    C = &I->second;
  else
    C = M->getOrInsertComdat(Name);
  C->setSelectionKind(SK);

  return false;
}

// MDString:
//   ::= '!' STRINGCONSTANT
bool LLParser::ParseMDString(MDString *&Result) {
  std::string Str;
  if (ParseStringConstant(Str)) return true;
  llvm::UpgradeMDStringConstant(Str);
  Result = MDString::get(Context, Str);
  return false;
}

// MDNode:
//   ::= '!' MDNodeNumber
bool LLParser::ParseMDNodeID(MDNode *&Result) {
  // !{ ..., !42, ... }
  unsigned MID = 0;
  if (ParseUInt32(MID))
    return true;

  // If not a forward reference, just return it now.
  if (NumberedMetadata.count(MID)) {
    Result = NumberedMetadata[MID];
    return false;
  }

  // Otherwise, create MDNode forward reference.
  auto &FwdRef = ForwardRefMDNodes[MID];
  FwdRef = std::make_pair(MDTuple::getTemporary(Context, None), Lex.getLoc());

  Result = FwdRef.first.get();
  NumberedMetadata[MID].reset(Result);
  return false;
}

/// ParseNamedMetadata:
///   !foo = !{ !1, !2 }
bool LLParser::ParseNamedMetadata() {
  assert(Lex.getKind() == lltok::MetadataVar);
  std::string Name = Lex.getStrVal();
  Lex.Lex();

  if (ParseToken(lltok::equal, "expected '=' here") ||
      ParseToken(lltok::exclaim, "Expected '!' here") ||
      ParseToken(lltok::lbrace, "Expected '{' here"))
    return true;

  NamedMDNode *NMD = M->getOrInsertNamedMetadata(Name);
  if (Lex.getKind() != lltok::rbrace)
    do {
      if (ParseToken(lltok::exclaim, "Expected '!' here"))
        return true;

      MDNode *N = nullptr;
      if (ParseMDNodeID(N)) return true;
      NMD->addOperand(N);
    } while (EatIfPresent(lltok::comma));

  return ParseToken(lltok::rbrace, "expected end of metadata node");
}

/// ParseStandaloneMetadata:
///   !42 = !{...}
bool LLParser::ParseStandaloneMetadata() {
  assert(Lex.getKind() == lltok::exclaim);
  Lex.Lex();
  unsigned MetadataID = 0;

  MDNode *Init;
  if (ParseUInt32(MetadataID) ||
      ParseToken(lltok::equal, "expected '=' here"))
    return true;

  // Detect common error, from old metadata syntax.
  if (Lex.getKind() == lltok::Type)
    return TokError("unexpected type in metadata definition");

  bool IsDistinct = EatIfPresent(lltok::kw_distinct);
  if (Lex.getKind() == lltok::MetadataVar) {
    if (ParseSpecializedMDNode(Init, IsDistinct))
      return true;
  } else if (ParseToken(lltok::exclaim, "Expected '!' here") ||
             ParseMDTuple(Init, IsDistinct))
    return true;

  // See if this was forward referenced, if so, handle it.
  auto FI = ForwardRefMDNodes.find(MetadataID);
  if (FI != ForwardRefMDNodes.end()) {
    FI->second.first->replaceAllUsesWith(Init);
    ForwardRefMDNodes.erase(FI);

    assert(NumberedMetadata[MetadataID] == Init && "Tracking VH didn't work");
  } else {
    if (NumberedMetadata.count(MetadataID))
      return TokError("Metadata id is already used");
    NumberedMetadata[MetadataID].reset(Init);
  }

  return false;
}

static bool isValidVisibilityForLinkage(unsigned V, unsigned L) {
  return !GlobalValue::isLocalLinkage((GlobalValue::LinkageTypes)L) ||
         (GlobalValue::VisibilityTypes)V == GlobalValue::DefaultVisibility;
}

/// ParseAlias:
///   ::= GlobalVar '=' OptionalLinkage OptionalVisibility
///                     OptionalDLLStorageClass OptionalThreadLocal
///                     OptionalUnnamedAddr 'alias' Aliasee
///
/// Aliasee
///   ::= TypeAndValue
///
/// Everything through OptionalUnnamedAddr has already been parsed.
///
bool LLParser::ParseAlias(const std::string &Name, LocTy NameLoc, unsigned L,
                          unsigned Visibility, unsigned DLLStorageClass,
                          GlobalVariable::ThreadLocalMode TLM,
                          bool UnnamedAddr) {
  assert(Lex.getKind() == lltok::kw_alias);
  Lex.Lex();

  GlobalValue::LinkageTypes Linkage = (GlobalValue::LinkageTypes) L;

  if(!GlobalAlias::isValidLinkage(Linkage))
    return Error(NameLoc, "invalid linkage type for alias");

  if (!isValidVisibilityForLinkage(Visibility, L))
    return Error(NameLoc,
                 "symbol with local linkage must have default visibility");

  Constant *Aliasee;
  LocTy AliaseeLoc = Lex.getLoc();
  if (Lex.getKind() != lltok::kw_bitcast &&
      Lex.getKind() != lltok::kw_getelementptr &&
      Lex.getKind() != lltok::kw_addrspacecast &&
      Lex.getKind() != lltok::kw_inttoptr) {
    if (ParseGlobalTypeAndValue(Aliasee))
      return true;
  } else {
    // The bitcast dest type is not present, it is implied by the dest type.
    ValID ID;
    if (ParseValID(ID))
      return true;
    if (ID.Kind != ValID::t_Constant)
      return Error(AliaseeLoc, "invalid aliasee");
    Aliasee = ID.ConstantVal;
  }

  Type *AliaseeType = Aliasee->getType();
  auto *PTy = dyn_cast<PointerType>(AliaseeType);
  if (!PTy)
    return Error(AliaseeLoc, "An alias must have pointer type");

  // Okay, create the alias but do not insert it into the module yet.
  std::unique_ptr<GlobalAlias> GA(
      GlobalAlias::create(PTy, (GlobalValue::LinkageTypes)Linkage, Name,
                          Aliasee, /*Parent*/ nullptr));
  GA->setThreadLocalMode(TLM);
  GA->setVisibility((GlobalValue::VisibilityTypes)Visibility);
  GA->setDLLStorageClass((GlobalValue::DLLStorageClassTypes)DLLStorageClass);
  GA->setUnnamedAddr(UnnamedAddr);

  if (Name.empty())
    NumberedVals.push_back(GA.get());

  // See if this value already exists in the symbol table.  If so, it is either
  // a redefinition or a definition of a forward reference.
  if (GlobalValue *Val = M->getNamedValue(Name)) {
    // See if this was a redefinition.  If so, there is no entry in
    // ForwardRefVals.
    std::map<std::string, std::pair<GlobalValue*, LocTy> >::iterator
      I = ForwardRefVals.find(Name);
    if (I == ForwardRefVals.end())
      return Error(NameLoc, "redefinition of global named '@" + Name + "'");

    // Otherwise, this was a definition of forward ref.  Verify that types
    // agree.
    if (Val->getType() != GA->getType())
      return Error(NameLoc,
              "forward reference and definition of alias have different types");

    // If they agree, just RAUW the old value with the alias and remove the
    // forward ref info.
    Val->replaceAllUsesWith(GA.get());
    Val->eraseFromParent();
    ForwardRefVals.erase(I);
  }

  // Insert into the module, we know its name won't collide now.
  M->getAliasList().push_back(GA.get());
  assert(GA->getName() == Name && "Should not be a name conflict!");

  // The module owns this now
  GA.release();

  return false;
}

/// ParseGlobal
///   ::= GlobalVar '=' OptionalLinkage OptionalVisibility OptionalDLLStorageClass
///       OptionalThreadLocal OptionalUnnamedAddr OptionalAddrSpace
///       OptionalExternallyInitialized GlobalType Type Const
///   ::= OptionalLinkage OptionalVisibility OptionalDLLStorageClass
///       OptionalThreadLocal OptionalUnnamedAddr OptionalAddrSpace
///       OptionalExternallyInitialized GlobalType Type Const
///
/// Everything up to and including OptionalUnnamedAddr has been parsed
/// already.
///
bool LLParser::ParseGlobal(const std::string &Name, LocTy NameLoc,
                           unsigned Linkage, bool HasLinkage,
                           unsigned Visibility, unsigned DLLStorageClass,
                           GlobalVariable::ThreadLocalMode TLM,
                           bool UnnamedAddr) {
  if (!isValidVisibilityForLinkage(Visibility, Linkage))
    return Error(NameLoc,
                 "symbol with local linkage must have default visibility");

  unsigned AddrSpace;
  bool IsConstant, IsExternallyInitialized;
  LocTy IsExternallyInitializedLoc;
  LocTy TyLoc;

  Type *Ty = nullptr;
  if (ParseOptionalAddrSpace(AddrSpace) ||
      ParseOptionalToken(lltok::kw_externally_initialized,
                         IsExternallyInitialized,
                         &IsExternallyInitializedLoc) ||
      ParseGlobalType(IsConstant) ||
      ParseType(Ty, TyLoc))
    return true;

  // If the linkage is specified and is external, then no initializer is
  // present.
  Constant *Init = nullptr;
  if (!HasLinkage || (Linkage != GlobalValue::ExternalWeakLinkage &&
                      Linkage != GlobalValue::ExternalLinkage)) {
    if (ParseGlobalValue(Ty, Init))
      return true;
  }

  if (Ty->isFunctionTy() || !PointerType::isValidElementType(Ty))
    return Error(TyLoc, "invalid type for global variable");

  GlobalValue *GVal = nullptr;

  // See if the global was forward referenced, if so, use the global.
  if (!Name.empty()) {
    GVal = M->getNamedValue(Name);
    if (GVal) {
      if (!ForwardRefVals.erase(Name) || !isa<GlobalValue>(GVal))
        return Error(NameLoc, "redefinition of global '@" + Name + "'");
    }
  } else {
    std::map<unsigned, std::pair<GlobalValue*, LocTy> >::iterator
      I = ForwardRefValIDs.find(NumberedVals.size());
    if (I != ForwardRefValIDs.end()) {
      GVal = I->second.first;
      ForwardRefValIDs.erase(I);
    }
  }

  GlobalVariable *GV;
  if (!GVal) {
    GV = new GlobalVariable(*M, Ty, false, GlobalValue::ExternalLinkage, nullptr,
                            Name, nullptr, GlobalVariable::NotThreadLocal,
                            AddrSpace);
  } else {
    if (GVal->getValueType() != Ty)
      return Error(TyLoc,
            "forward reference and definition of global have different types");

    GV = cast<GlobalVariable>(GVal);

    // Move the forward-reference to the correct spot in the module.
    M->getGlobalList().splice(M->global_end(), M->getGlobalList(), GV);
  }

  if (Name.empty())
    NumberedVals.push_back(GV);

  // Set the parsed properties on the global.
  if (Init)
    GV->setInitializer(Init);
  GV->setConstant(IsConstant);
  GV->setLinkage((GlobalValue::LinkageTypes)Linkage);
  GV->setVisibility((GlobalValue::VisibilityTypes)Visibility);
  GV->setDLLStorageClass((GlobalValue::DLLStorageClassTypes)DLLStorageClass);
  GV->setExternallyInitialized(IsExternallyInitialized);
  GV->setThreadLocalMode(TLM);
  GV->setUnnamedAddr(UnnamedAddr);

  // Parse attributes on the global.
  while (Lex.getKind() == lltok::comma) {
    Lex.Lex();

    if (Lex.getKind() == lltok::kw_section) {
      Lex.Lex();
      GV->setSection(Lex.getStrVal());
      if (ParseToken(lltok::StringConstant, "expected global section string"))
        return true;
    } else if (Lex.getKind() == lltok::kw_align) {
      unsigned Alignment;
      if (ParseOptionalAlignment(Alignment)) return true;
      GV->setAlignment(Alignment);
    } else {
      Comdat *C;
      if (parseOptionalComdat(Name, C))
        return true;
      if (C)
        GV->setComdat(C);
      else
        return TokError("unknown global variable property!");
    }
  }

  return false;
}

/// ParseUnnamedAttrGrp
///   ::= 'attributes' AttrGrpID '=' '{' AttrValPair+ '}'
bool LLParser::ParseUnnamedAttrGrp() {
  assert(Lex.getKind() == lltok::kw_attributes);
  LocTy AttrGrpLoc = Lex.getLoc();
  Lex.Lex();

  if (Lex.getKind() != lltok::AttrGrpID)
    return TokError("expected attribute group id");

  unsigned VarID = Lex.getUIntVal();
  std::vector<unsigned> unused;
  LocTy BuiltinLoc;
  Lex.Lex();

  if (ParseToken(lltok::equal, "expected '=' here") ||
      ParseToken(lltok::lbrace, "expected '{' here") ||
      ParseFnAttributeValuePairs(NumberedAttrBuilders[VarID], unused, true,
                                 BuiltinLoc) ||
      ParseToken(lltok::rbrace, "expected end of attribute group"))
    return true;

  if (!NumberedAttrBuilders[VarID].hasAttributes())
    return Error(AttrGrpLoc, "attribute group has no attributes");

  return false;
}

/// ParseFnAttributeValuePairs
///   ::= <attr> | <attr> '=' <value>
bool LLParser::ParseFnAttributeValuePairs(AttrBuilder &B,
                                          std::vector<unsigned> &FwdRefAttrGrps,
                                          bool inAttrGrp, LocTy &BuiltinLoc) {
  bool HaveError = false;

  B.clear();

  while (true) {
    lltok::Kind Token = Lex.getKind();
    if (Token == lltok::kw_builtin)
      BuiltinLoc = Lex.getLoc();
    switch (Token) {
    default:
      if (!inAttrGrp) return HaveError;
      return Error(Lex.getLoc(), "unterminated attribute group");
    case lltok::rbrace:
      // Finished.
      return false;

    case lltok::AttrGrpID: {
      // Allow a function to reference an attribute group:
      //
      //   define void @foo() #1 { ... }
      if (inAttrGrp)
        HaveError |=
          Error(Lex.getLoc(),
              "cannot have an attribute group reference in an attribute group");

      unsigned AttrGrpNum = Lex.getUIntVal();
      if (inAttrGrp) break;

      // Save the reference to the attribute group. We'll fill it in later.
      FwdRefAttrGrps.push_back(AttrGrpNum);
      break;
    }
    // Target-dependent attributes:
    case lltok::StringConstant: {
      std::string Attr = Lex.getStrVal();
      Lex.Lex();
      std::string Val;
      if (EatIfPresent(lltok::equal) &&
          ParseStringConstant(Val))
        return true;

      B.addAttribute(Attr, Val);
      continue;
    }

    // Target-independent attributes:
    case lltok::kw_align: {
      // As a hack, we allow function alignment to be initially parsed as an
      // attribute on a function declaration/definition or added to an attribute
      // group and later moved to the alignment field.
      unsigned Alignment;
      if (inAttrGrp) {
        Lex.Lex();
        if (ParseToken(lltok::equal, "expected '=' here") ||
            ParseUInt32(Alignment))
          return true;
      } else {
        if (ParseOptionalAlignment(Alignment))
          return true;
      }
      B.addAlignmentAttr(Alignment);
      continue;
    }
    case lltok::kw_alignstack: {
      unsigned Alignment;
      if (inAttrGrp) {
        Lex.Lex();
        if (ParseToken(lltok::equal, "expected '=' here") ||
            ParseUInt32(Alignment))
          return true;
      } else {
        if (ParseOptionalStackAlignment(Alignment))
          return true;
      }
      B.addStackAlignmentAttr(Alignment);
      continue;
    }
    case lltok::kw_alwaysinline: B.addAttribute(Attribute::AlwaysInline); break;
    case lltok::kw_argmemonly: B.addAttribute(Attribute::ArgMemOnly); break;
    case lltok::kw_builtin: B.addAttribute(Attribute::Builtin); break;
    case lltok::kw_cold: B.addAttribute(Attribute::Cold); break;
    case lltok::kw_convergent: B.addAttribute(Attribute::Convergent); break;
    case lltok::kw_inlinehint: B.addAttribute(Attribute::InlineHint); break;
    case lltok::kw_jumptable: B.addAttribute(Attribute::JumpTable); break;
    case lltok::kw_minsize: B.addAttribute(Attribute::MinSize); break;
    case lltok::kw_naked: B.addAttribute(Attribute::Naked); break;
    case lltok::kw_nobuiltin: B.addAttribute(Attribute::NoBuiltin); break;
    case lltok::kw_noduplicate: B.addAttribute(Attribute::NoDuplicate); break;
    case lltok::kw_noimplicitfloat:
      B.addAttribute(Attribute::NoImplicitFloat); break;
    case lltok::kw_noinline: B.addAttribute(Attribute::NoInline); break;
    case lltok::kw_nonlazybind: B.addAttribute(Attribute::NonLazyBind); break;
    case lltok::kw_noredzone: B.addAttribute(Attribute::NoRedZone); break;
    case lltok::kw_noreturn: B.addAttribute(Attribute::NoReturn); break;
    case lltok::kw_nounwind: B.addAttribute(Attribute::NoUnwind); break;
    case lltok::kw_optnone: B.addAttribute(Attribute::OptimizeNone); break;
    case lltok::kw_optsize: B.addAttribute(Attribute::OptimizeForSize); break;
    case lltok::kw_readnone: B.addAttribute(Attribute::ReadNone); break;
    case lltok::kw_readonly: B.addAttribute(Attribute::ReadOnly); break;
    case lltok::kw_returns_twice:
      B.addAttribute(Attribute::ReturnsTwice); break;
    case lltok::kw_ssp: B.addAttribute(Attribute::StackProtect); break;
    case lltok::kw_sspreq: B.addAttribute(Attribute::StackProtectReq); break;
    case lltok::kw_sspstrong:
      B.addAttribute(Attribute::StackProtectStrong); break;
    case lltok::kw_safestack: B.addAttribute(Attribute::SafeStack); break;
    case lltok::kw_sanitize_address:
      B.addAttribute(Attribute::SanitizeAddress); break;
    case lltok::kw_sanitize_thread:
      B.addAttribute(Attribute::SanitizeThread); break;
    case lltok::kw_sanitize_memory:
      B.addAttribute(Attribute::SanitizeMemory); break;
    case lltok::kw_uwtable: B.addAttribute(Attribute::UWTable); break;

    // Error handling.
    case lltok::kw_inreg:
    case lltok::kw_signext:
    case lltok::kw_zeroext:
      HaveError |=
        Error(Lex.getLoc(),
              "invalid use of attribute on a function");
      break;
    case lltok::kw_byval:
    case lltok::kw_dereferenceable:
    case lltok::kw_dereferenceable_or_null:
    case lltok::kw_inalloca:
    case lltok::kw_nest:
    case lltok::kw_noalias:
    case lltok::kw_nocapture:
    case lltok::kw_nonnull:
    case lltok::kw_returned:
    case lltok::kw_sret:
      HaveError |=
        Error(Lex.getLoc(),
              "invalid use of parameter-only attribute on a function");
      break;
    }

    Lex.Lex();
  }
}

//===----------------------------------------------------------------------===//
// GlobalValue Reference/Resolution Routines.
//===----------------------------------------------------------------------===//

/// GetGlobalVal - Get a value with the specified name or ID, creating a
/// forward reference record if needed.  This can return null if the value
/// exists but does not have the right type.
GlobalValue *LLParser::GetGlobalVal(const std::string &Name, Type *Ty,
                                    LocTy Loc) {
  PointerType *PTy = dyn_cast<PointerType>(Ty);
  if (!PTy) {
    Error(Loc, "global variable reference must have pointer type");
    return nullptr;
  }

  // Look this name up in the normal function symbol table.
  GlobalValue *Val =
    cast_or_null<GlobalValue>(M->getValueSymbolTable().lookup(Name));

  // If this is a forward reference for the value, see if we already created a
  // forward ref record.
  if (!Val) {
    std::map<std::string, std::pair<GlobalValue*, LocTy> >::iterator
      I = ForwardRefVals.find(Name);
    if (I != ForwardRefVals.end())
      Val = I->second.first;
  }

  // If we have the value in the symbol table or fwd-ref table, return it.
  if (Val) {
    if (Val->getType() == Ty) return Val;
    Error(Loc, "'@" + Name + "' defined with type '" +
          getTypeString(Val->getType()) + "'");
    return nullptr;
  }

  // Otherwise, create a new forward reference for this value and remember it.
  GlobalValue *FwdVal;
  if (FunctionType *FT = dyn_cast<FunctionType>(PTy->getElementType()))
    FwdVal = Function::Create(FT, GlobalValue::ExternalWeakLinkage, Name, M);
  else
    FwdVal = new GlobalVariable(*M, PTy->getElementType(), false,
                                GlobalValue::ExternalWeakLinkage, nullptr, Name,
                                nullptr, GlobalVariable::NotThreadLocal,
                                PTy->getAddressSpace());

  ForwardRefVals[Name] = std::make_pair(FwdVal, Loc);
  return FwdVal;
}

GlobalValue *LLParser::GetGlobalVal(unsigned ID, Type *Ty, LocTy Loc) {
  PointerType *PTy = dyn_cast<PointerType>(Ty);
  if (!PTy) {
    Error(Loc, "global variable reference must have pointer type");
    return nullptr;
  }

  GlobalValue *Val = ID < NumberedVals.size() ? NumberedVals[ID] : nullptr;

  // If this is a forward reference for the value, see if we already created a
  // forward ref record.
  if (!Val) {
    std::map<unsigned, std::pair<GlobalValue*, LocTy> >::iterator
      I = ForwardRefValIDs.find(ID);
    if (I != ForwardRefValIDs.end())
      Val = I->second.first;
  }

  // If we have the value in the symbol table or fwd-ref table, return it.
  if (Val) {
    if (Val->getType() == Ty) return Val;
    Error(Loc, "'@" + Twine(ID) + "' defined with type '" +
          getTypeString(Val->getType()) + "'");
    return nullptr;
  }

  // Otherwise, create a new forward reference for this value and remember it.
  GlobalValue *FwdVal;
  if (FunctionType *FT = dyn_cast<FunctionType>(PTy->getElementType()))
    FwdVal = Function::Create(FT, GlobalValue::ExternalWeakLinkage, "", M);
  else
    FwdVal = new GlobalVariable(*M, PTy->getElementType(), false,
                                GlobalValue::ExternalWeakLinkage, nullptr, "");

  ForwardRefValIDs[ID] = std::make_pair(FwdVal, Loc);
  return FwdVal;
}


//===----------------------------------------------------------------------===//
// Comdat Reference/Resolution Routines.
//===----------------------------------------------------------------------===//

Comdat *LLParser::getComdat(const std::string &Name, LocTy Loc) {
  // Look this name up in the comdat symbol table.
  Module::ComdatSymTabType &ComdatSymTab = M->getComdatSymbolTable();
  Module::ComdatSymTabType::iterator I = ComdatSymTab.find(Name);
  if (I != ComdatSymTab.end())
    return &I->second;

  // Otherwise, create a new forward reference for this value and remember it.
  Comdat *C = M->getOrInsertComdat(Name);
  ForwardRefComdats[Name] = Loc;
  return C;
}


//===----------------------------------------------------------------------===//
// Helper Routines.
//===----------------------------------------------------------------------===//

/// ParseToken - If the current token has the specified kind, eat it and return
/// success.  Otherwise, emit the specified error and return failure.
bool LLParser::ParseToken(lltok::Kind T, const char *ErrMsg) {
  if (Lex.getKind() != T)
    return TokError(ErrMsg);
  Lex.Lex();
  return false;
}

/// ParseStringConstant
///   ::= StringConstant
bool LLParser::ParseStringConstant(std::string &Result) {
  if (Lex.getKind() != lltok::StringConstant)
    return TokError("expected string constant");
  Result = Lex.getStrVal();
  Lex.Lex();
  return false;
}

/// ParseUInt32
///   ::= uint32
bool LLParser::ParseUInt32(unsigned &Val) {
  if (Lex.getKind() != lltok::APSInt || Lex.getAPSIntVal().isSigned())
    return TokError("expected integer");
  uint64_t Val64 = Lex.getAPSIntVal().getLimitedValue(0xFFFFFFFFULL+1);
  if (Val64 != unsigned(Val64))
    return TokError("expected 32-bit integer (too large)");
  Val = Val64;
  Lex.Lex();
  return false;
}

/// ParseUInt64
///   ::= uint64
bool LLParser::ParseUInt64(uint64_t &Val) {
  if (Lex.getKind() != lltok::APSInt || Lex.getAPSIntVal().isSigned())
    return TokError("expected integer");
  Val = Lex.getAPSIntVal().getLimitedValue();
  Lex.Lex();
  return false;
}

/// ParseTLSModel
///   := 'localdynamic'
///   := 'initialexec'
///   := 'localexec'
bool LLParser::ParseTLSModel(GlobalVariable::ThreadLocalMode &TLM) {
  switch (Lex.getKind()) {
    default:
      return TokError("expected localdynamic, initialexec or localexec");
    case lltok::kw_localdynamic:
      TLM = GlobalVariable::LocalDynamicTLSModel;
      break;
    case lltok::kw_initialexec:
      TLM = GlobalVariable::InitialExecTLSModel;
      break;
    case lltok::kw_localexec:
      TLM = GlobalVariable::LocalExecTLSModel;
      break;
  }

  Lex.Lex();
  return false;
}

/// ParseOptionalThreadLocal
///   := /*empty*/
///   := 'thread_local'
///   := 'thread_local' '(' tlsmodel ')'
bool LLParser::ParseOptionalThreadLocal(GlobalVariable::ThreadLocalMode &TLM) {
  TLM = GlobalVariable::NotThreadLocal;
  if (!EatIfPresent(lltok::kw_thread_local))
    return false;

  TLM = GlobalVariable::GeneralDynamicTLSModel;
  if (Lex.getKind() == lltok::lparen) {
    Lex.Lex();
    return ParseTLSModel(TLM) ||
      ParseToken(lltok::rparen, "expected ')' after thread local model");
  }
  return false;
}

/// ParseOptionalAddrSpace
///   := /*empty*/
///   := 'addrspace' '(' uint32 ')'
bool LLParser::ParseOptionalAddrSpace(unsigned &AddrSpace) {
  AddrSpace = 0;
  if (!EatIfPresent(lltok::kw_addrspace))
    return false;
  return ParseToken(lltok::lparen, "expected '(' in address space") ||
         ParseUInt32(AddrSpace) ||
         ParseToken(lltok::rparen, "expected ')' in address space");
}

/// ParseOptionalParamAttrs - Parse a potentially empty list of parameter attributes.
bool LLParser::ParseOptionalParamAttrs(AttrBuilder &B) {
  bool HaveError = false;

  B.clear();

  while (1) {
    lltok::Kind Token = Lex.getKind();
    switch (Token) {
    default:  // End of attributes.
      return HaveError;
    case lltok::kw_align: {
      unsigned Alignment;
      if (ParseOptionalAlignment(Alignment))
        return true;
      B.addAlignmentAttr(Alignment);
      continue;
    }
    case lltok::kw_byval:           B.addAttribute(Attribute::ByVal); break;
    case lltok::kw_dereferenceable: {
      uint64_t Bytes;
      if (ParseOptionalDerefAttrBytes(lltok::kw_dereferenceable, Bytes))
        return true;
      B.addDereferenceableAttr(Bytes);
      continue;
    }
    case lltok::kw_dereferenceable_or_null: {
      uint64_t Bytes;
      if (ParseOptionalDerefAttrBytes(lltok::kw_dereferenceable_or_null, Bytes))
        return true;
      B.addDereferenceableOrNullAttr(Bytes);
      continue;
    }
    case lltok::kw_inalloca:        B.addAttribute(Attribute::InAlloca); break;
    case lltok::kw_inreg:           B.addAttribute(Attribute::InReg); break;
    case lltok::kw_nest:            B.addAttribute(Attribute::Nest); break;
    case lltok::kw_noalias:         B.addAttribute(Attribute::NoAlias); break;
    case lltok::kw_nocapture:       B.addAttribute(Attribute::NoCapture); break;
    case lltok::kw_nonnull:         B.addAttribute(Attribute::NonNull); break;
    case lltok::kw_readnone:        B.addAttribute(Attribute::ReadNone); break;
    case lltok::kw_readonly:        B.addAttribute(Attribute::ReadOnly); break;
    case lltok::kw_returned:        B.addAttribute(Attribute::Returned); break;
    case lltok::kw_signext:         B.addAttribute(Attribute::SExt); break;
    case lltok::kw_sret:            B.addAttribute(Attribute::StructRet); break;
    case lltok::kw_zeroext:         B.addAttribute(Attribute::ZExt); break;

    case lltok::kw_alignstack:
    case lltok::kw_alwaysinline:
    case lltok::kw_argmemonly:
    case lltok::kw_builtin:
    case lltok::kw_inlinehint:
    case lltok::kw_jumptable:
    case lltok::kw_minsize:
    case lltok::kw_naked:
    case lltok::kw_nobuiltin:
    case lltok::kw_noduplicate:
    case lltok::kw_noimplicitfloat:
    case lltok::kw_noinline:
    case lltok::kw_nonlazybind:
    case lltok::kw_noredzone:
    case lltok::kw_noreturn:
    case lltok::kw_nounwind:
    case lltok::kw_optnone:
    case lltok::kw_optsize:
    case lltok::kw_returns_twice:
    case lltok::kw_sanitize_address:
    case lltok::kw_sanitize_memory:
    case lltok::kw_sanitize_thread:
    case lltok::kw_ssp:
    case lltok::kw_sspreq:
    case lltok::kw_sspstrong:
    case lltok::kw_safestack:
    case lltok::kw_uwtable:
      HaveError |= Error(Lex.getLoc(), "invalid use of function-only attribute");
      break;
    }

    Lex.Lex();
  }
}

/// ParseOptionalReturnAttrs - Parse a potentially empty list of return attributes.
bool LLParser::ParseOptionalReturnAttrs(AttrBuilder &B) {
  bool HaveError = false;

  B.clear();

  while (1) {
    lltok::Kind Token = Lex.getKind();
    switch (Token) {
    default:  // End of attributes.
      return HaveError;
    case lltok::kw_dereferenceable: {
      uint64_t Bytes;
      if (ParseOptionalDerefAttrBytes(lltok::kw_dereferenceable, Bytes))
        return true;
      B.addDereferenceableAttr(Bytes);
      continue;
    }
    case lltok::kw_dereferenceable_or_null: {
      uint64_t Bytes;
      if (ParseOptionalDerefAttrBytes(lltok::kw_dereferenceable_or_null, Bytes))
        return true;
      B.addDereferenceableOrNullAttr(Bytes);
      continue;
    }
    case lltok::kw_inreg:           B.addAttribute(Attribute::InReg); break;
    case lltok::kw_noalias:         B.addAttribute(Attribute::NoAlias); break;
    case lltok::kw_nonnull:         B.addAttribute(Attribute::NonNull); break;
    case lltok::kw_signext:         B.addAttribute(Attribute::SExt); break;
    case lltok::kw_zeroext:         B.addAttribute(Attribute::ZExt); break;

    // Error handling.
    case lltok::kw_align:
    case lltok::kw_byval:
    case lltok::kw_inalloca:
    case lltok::kw_nest:
    case lltok::kw_nocapture:
    case lltok::kw_returned:
    case lltok::kw_sret:
      HaveError |= Error(Lex.getLoc(), "invalid use of parameter-only attribute");
      break;

    case lltok::kw_alignstack:
    case lltok::kw_alwaysinline:
    case lltok::kw_argmemonly:
    case lltok::kw_builtin:
    case lltok::kw_cold:
    case lltok::kw_inlinehint:
    case lltok::kw_jumptable:
    case lltok::kw_minsize:
    case lltok::kw_naked:
    case lltok::kw_nobuiltin:
    case lltok::kw_noduplicate:
    case lltok::kw_noimplicitfloat:
    case lltok::kw_noinline:
    case lltok::kw_nonlazybind:
    case lltok::kw_noredzone:
    case lltok::kw_noreturn:
    case lltok::kw_nounwind:
    case lltok::kw_optnone:
    case lltok::kw_optsize:
    case lltok::kw_returns_twice:
    case lltok::kw_sanitize_address:
    case lltok::kw_sanitize_memory:
    case lltok::kw_sanitize_thread:
    case lltok::kw_ssp:
    case lltok::kw_sspreq:
    case lltok::kw_sspstrong:
    case lltok::kw_safestack:
    case lltok::kw_uwtable:
      HaveError |= Error(Lex.getLoc(), "invalid use of function-only attribute");
      break;

    case lltok::kw_readnone:
    case lltok::kw_readonly:
      HaveError |= Error(Lex.getLoc(), "invalid use of attribute on return type");
    }

    Lex.Lex();
  }
}

/// ParseOptionalLinkage
///   ::= /*empty*/
///   ::= 'private'
///   ::= 'internal'
///   ::= 'weak'
///   ::= 'weak_odr'
///   ::= 'linkonce'
///   ::= 'linkonce_odr'
///   ::= 'available_externally'
///   ::= 'appending'
///   ::= 'common'
///   ::= 'extern_weak'
///   ::= 'external'
bool LLParser::ParseOptionalLinkage(unsigned &Res, bool &HasLinkage) {
  HasLinkage = false;
  switch (Lex.getKind()) {
  default:                       Res=GlobalValue::ExternalLinkage; return false;
  case lltok::kw_private:        Res = GlobalValue::PrivateLinkage;       break;
  case lltok::kw_internal:       Res = GlobalValue::InternalLinkage;      break;
  case lltok::kw_weak:           Res = GlobalValue::WeakAnyLinkage;       break;
  case lltok::kw_weak_odr:       Res = GlobalValue::WeakODRLinkage;       break;
  case lltok::kw_linkonce:       Res = GlobalValue::LinkOnceAnyLinkage;   break;
  case lltok::kw_linkonce_odr:   Res = GlobalValue::LinkOnceODRLinkage;   break;
  case lltok::kw_available_externally:
    Res = GlobalValue::AvailableExternallyLinkage;
    break;
  case lltok::kw_appending:      Res = GlobalValue::AppendingLinkage;     break;
  case lltok::kw_common:         Res = GlobalValue::CommonLinkage;        break;
  case lltok::kw_extern_weak:    Res = GlobalValue::ExternalWeakLinkage;  break;
  case lltok::kw_external:       Res = GlobalValue::ExternalLinkage;      break;
  }
  Lex.Lex();
  HasLinkage = true;
  return false;
}

/// ParseOptionalVisibility
///   ::= /*empty*/
///   ::= 'default'
///   ::= 'hidden'
///   ::= 'protected'
///
bool LLParser::ParseOptionalVisibility(unsigned &Res) {
  switch (Lex.getKind()) {
  default:                  Res = GlobalValue::DefaultVisibility; return false;
  case lltok::kw_default:   Res = GlobalValue::DefaultVisibility; break;
  case lltok::kw_hidden:    Res = GlobalValue::HiddenVisibility; break;
  case lltok::kw_protected: Res = GlobalValue::ProtectedVisibility; break;
  }
  Lex.Lex();
  return false;
}

/// ParseOptionalDLLStorageClass
///   ::= /*empty*/
///   ::= 'dllimport'
///   ::= 'dllexport'
///
bool LLParser::ParseOptionalDLLStorageClass(unsigned &Res) {
  switch (Lex.getKind()) {
  default:                  Res = GlobalValue::DefaultStorageClass; return false;
  case lltok::kw_dllimport: Res = GlobalValue::DLLImportStorageClass; break;
  case lltok::kw_dllexport: Res = GlobalValue::DLLExportStorageClass; break;
  }
  Lex.Lex();
  return false;
}

/// ParseOptionalCallingConv
///   ::= /*empty*/
///   ::= 'ccc'
///   ::= 'fastcc'
///   ::= 'intel_ocl_bicc'
///   ::= 'coldcc'
///   ::= 'x86_stdcallcc'
///   ::= 'x86_fastcallcc'
///   ::= 'x86_thiscallcc'
///   ::= 'x86_vectorcallcc'
///   ::= 'arm_apcscc'
///   ::= 'arm_aapcscc'
///   ::= 'arm_aapcs_vfpcc'
///   ::= 'msp430_intrcc'
///   ::= 'ptx_kernel'
///   ::= 'ptx_device'
///   ::= 'spir_func'
///   ::= 'spir_kernel'
///   ::= 'x86_64_sysvcc'
///   ::= 'x86_64_win64cc'
///   ::= 'webkit_jscc'
///   ::= 'anyregcc'
///   ::= 'preserve_mostcc'
///   ::= 'preserve_allcc'
///   ::= 'ghccc'
///   ::= 'cc' UINT
///
bool LLParser::ParseOptionalCallingConv(unsigned &CC) {
  switch (Lex.getKind()) {
  default:                       CC = CallingConv::C; return false;
  case lltok::kw_ccc:            CC = CallingConv::C; break;
  case lltok::kw_fastcc:         CC = CallingConv::Fast; break;
  case lltok::kw_coldcc:         CC = CallingConv::Cold; break;
  case lltok::kw_x86_stdcallcc:  CC = CallingConv::X86_StdCall; break;
  case lltok::kw_x86_fastcallcc: CC = CallingConv::X86_FastCall; break;
  case lltok::kw_x86_thiscallcc: CC = CallingConv::X86_ThisCall; break;
  case lltok::kw_x86_vectorcallcc:CC = CallingConv::X86_VectorCall; break;
  case lltok::kw_arm_apcscc:     CC = CallingConv::ARM_APCS; break;
  case lltok::kw_arm_aapcscc:    CC = CallingConv::ARM_AAPCS; break;
  case lltok::kw_arm_aapcs_vfpcc:CC = CallingConv::ARM_AAPCS_VFP; break;
  case lltok::kw_msp430_intrcc:  CC = CallingConv::MSP430_INTR; break;
  case lltok::kw_ptx_kernel:     CC = CallingConv::PTX_Kernel; break;
  case lltok::kw_ptx_device:     CC = CallingConv::PTX_Device; break;
  case lltok::kw_spir_kernel:    CC = CallingConv::SPIR_KERNEL; break;
  case lltok::kw_spir_func:      CC = CallingConv::SPIR_FUNC; break;
  case lltok::kw_intel_ocl_bicc: CC = CallingConv::Intel_OCL_BI; break;
  case lltok::kw_x86_64_sysvcc:  CC = CallingConv::X86_64_SysV; break;
  case lltok::kw_x86_64_win64cc: CC = CallingConv::X86_64_Win64; break;
  case lltok::kw_webkit_jscc:    CC = CallingConv::WebKit_JS; break;
  case lltok::kw_anyregcc:       CC = CallingConv::AnyReg; break;
  case lltok::kw_preserve_mostcc:CC = CallingConv::PreserveMost; break;
  case lltok::kw_preserve_allcc: CC = CallingConv::PreserveAll; break;
  case lltok::kw_ghccc:          CC = CallingConv::GHC; break;
  case lltok::kw_cc: {
      Lex.Lex();
      return ParseUInt32(CC);
    }
  }

  Lex.Lex();
  return false;
}

/// ParseMetadataAttachment
///   ::= !dbg !42
bool LLParser::ParseMetadataAttachment(unsigned &Kind, MDNode *&MD) {
  assert(Lex.getKind() == lltok::MetadataVar && "Expected metadata attachment");

  std::string Name = Lex.getStrVal();
  Kind = M->getMDKindID(Name);
  Lex.Lex();

  return ParseMDNode(MD);
}

/// ParseInstructionMetadata
///   ::= !dbg !42 (',' !dbg !57)*
bool LLParser::ParseInstructionMetadata(Instruction &Inst) {
  do {
    if (Lex.getKind() != lltok::MetadataVar)
      return TokError("expected metadata after comma");

    unsigned MDK;
    MDNode *N;
    if (ParseMetadataAttachment(MDK, N))
      return true;

    Inst.setMetadata(MDK, N);
    if (MDK == LLVMContext::MD_tbaa)
      InstsWithTBAATag.push_back(&Inst);

    // If this is the end of the list, we're done.
  } while (EatIfPresent(lltok::comma));
  return false;
}

/// ParseOptionalFunctionMetadata
///   ::= (!dbg !57)*
bool LLParser::ParseOptionalFunctionMetadata(Function &F) {
  while (Lex.getKind() == lltok::MetadataVar) {
    unsigned MDK;
    MDNode *N;
    if (ParseMetadataAttachment(MDK, N))
      return true;

    F.setMetadata(MDK, N);
  }
  return false;
}

/// ParseOptionalAlignment
///   ::= /* empty */
///   ::= 'align' 4
bool LLParser::ParseOptionalAlignment(unsigned &Alignment) {
  Alignment = 0;
  if (!EatIfPresent(lltok::kw_align))
    return false;
  LocTy AlignLoc = Lex.getLoc();
  if (ParseUInt32(Alignment)) return true;
  if (!isPowerOf2_32(Alignment))
    return Error(AlignLoc, "alignment is not a power of two");
  if (Alignment > Value::MaximumAlignment)
    return Error(AlignLoc, "huge alignments are not supported yet");
  return false;
}

/// ParseOptionalDerefAttrBytes
///   ::= /* empty */
///   ::= AttrKind '(' 4 ')'
///
/// where AttrKind is either 'dereferenceable' or 'dereferenceable_or_null'.
bool LLParser::ParseOptionalDerefAttrBytes(lltok::Kind AttrKind,
                                           uint64_t &Bytes) {
  assert((AttrKind == lltok::kw_dereferenceable ||
          AttrKind == lltok::kw_dereferenceable_or_null) &&
         "contract!");

  Bytes = 0;
  if (!EatIfPresent(AttrKind))
    return false;
  LocTy ParenLoc = Lex.getLoc();
  if (!EatIfPresent(lltok::lparen))
    return Error(ParenLoc, "expected '('");
  LocTy DerefLoc = Lex.getLoc();
  if (ParseUInt64(Bytes)) return true;
  ParenLoc = Lex.getLoc();
  if (!EatIfPresent(lltok::rparen))
    return Error(ParenLoc, "expected ')'");
  if (!Bytes)
    return Error(DerefLoc, "dereferenceable bytes must be non-zero");
  return false;
}

/// ParseOptionalCommaAlign
///   ::=
///   ::= ',' align 4
///
/// This returns with AteExtraComma set to true if it ate an excess comma at the
/// end.
bool LLParser::ParseOptionalCommaAlign(unsigned &Alignment,
                                       bool &AteExtraComma) {
  AteExtraComma = false;
  while (EatIfPresent(lltok::comma)) {
    // Metadata at the end is an early exit.
    if (Lex.getKind() == lltok::MetadataVar) {
      AteExtraComma = true;
      return false;
    }

    if (Lex.getKind() != lltok::kw_align)
      return Error(Lex.getLoc(), "expected metadata or 'align'");

    if (ParseOptionalAlignment(Alignment)) return true;
  }

  return false;
}

/// ParseScopeAndOrdering
///   if isAtomic: ::= 'singlethread'? AtomicOrdering
///   else: ::=
///
/// This sets Scope and Ordering to the parsed values.
bool LLParser::ParseScopeAndOrdering(bool isAtomic, SynchronizationScope &Scope,
                                     AtomicOrdering &Ordering) {
  if (!isAtomic)
    return false;

  Scope = CrossThread;
  if (EatIfPresent(lltok::kw_singlethread))
    Scope = SingleThread;

  return ParseOrdering(Ordering);
}

/// ParseOrdering
///   ::= AtomicOrdering
///
/// This sets Ordering to the parsed value.
bool LLParser::ParseOrdering(AtomicOrdering &Ordering) {
  switch (Lex.getKind()) {
  default: return TokError("Expected ordering on atomic instruction");
  case lltok::kw_unordered: Ordering = Unordered; break;
  case lltok::kw_monotonic: Ordering = Monotonic; break;
  case lltok::kw_acquire: Ordering = Acquire; break;
  case lltok::kw_release: Ordering = Release; break;
  case lltok::kw_acq_rel: Ordering = AcquireRelease; break;
  case lltok::kw_seq_cst: Ordering = SequentiallyConsistent; break;
  }
  Lex.Lex();
  return false;
}

/// ParseOptionalStackAlignment
///   ::= /* empty */
///   ::= 'alignstack' '(' 4 ')'
bool LLParser::ParseOptionalStackAlignment(unsigned &Alignment) {
  Alignment = 0;
  if (!EatIfPresent(lltok::kw_alignstack))
    return false;
  LocTy ParenLoc = Lex.getLoc();
  if (!EatIfPresent(lltok::lparen))
    return Error(ParenLoc, "expected '('");
  LocTy AlignLoc = Lex.getLoc();
  if (ParseUInt32(Alignment)) return true;
  ParenLoc = Lex.getLoc();
  if (!EatIfPresent(lltok::rparen))
    return Error(ParenLoc, "expected ')'");
  if (!isPowerOf2_32(Alignment))
    return Error(AlignLoc, "stack alignment is not a power of two");
  return false;
}

/// ParseIndexList - This parses the index list for an insert/extractvalue
/// instruction.  This sets AteExtraComma in the case where we eat an extra
/// comma at the end of the line and find that it is followed by metadata.
/// Clients that don't allow metadata can call the version of this function that
/// only takes one argument.
///
/// ParseIndexList
///    ::=  (',' uint32)+
///
bool LLParser::ParseIndexList(SmallVectorImpl<unsigned> &Indices,
                              bool &AteExtraComma) {
  AteExtraComma = false;

  if (Lex.getKind() != lltok::comma)
    return TokError("expected ',' as start of index list");

  while (EatIfPresent(lltok::comma)) {
    if (Lex.getKind() == lltok::MetadataVar) {
      if (Indices.empty()) return TokError("expected index");
      AteExtraComma = true;
      return false;
    }
    unsigned Idx = 0;
    if (ParseUInt32(Idx)) return true;
    Indices.push_back(Idx);
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Type Parsing.
//===----------------------------------------------------------------------===//

/// ParseType - Parse a type.
bool LLParser::ParseType(Type *&Result, const Twine &Msg, bool AllowVoid) {
  SMLoc TypeLoc = Lex.getLoc();
  switch (Lex.getKind()) {
  default:
    return TokError(Msg);
  case lltok::Type:
    // Type ::= 'float' | 'void' (etc)
    Result = Lex.getTyVal();
    Lex.Lex();
    break;
  case lltok::lbrace:
    // Type ::= StructType
    if (ParseAnonStructType(Result, false))
      return true;
    break;
  case lltok::lsquare:
    // Type ::= '[' ... ']'
    Lex.Lex(); // eat the lsquare.
    if (ParseArrayVectorType(Result, false))
      return true;
    break;
  case lltok::less: // Either vector or packed struct.
    // Type ::= '<' ... '>'
    Lex.Lex();
    if (Lex.getKind() == lltok::lbrace) {
      if (ParseAnonStructType(Result, true) ||
          ParseToken(lltok::greater, "expected '>' at end of packed struct"))
        return true;
    } else if (ParseArrayVectorType(Result, true))
      return true;
    break;
  case lltok::LocalVar: {
    // Type ::= %foo
    std::pair<Type*, LocTy> &Entry = NamedTypes[Lex.getStrVal()];

    // If the type hasn't been defined yet, create a forward definition and
    // remember where that forward def'n was seen (in case it never is defined).
    if (!Entry.first) {
      Entry.first = StructType::create(Context, Lex.getStrVal());
      Entry.second = Lex.getLoc();
    }
    Result = Entry.first;
    Lex.Lex();
    break;
  }

  case lltok::LocalVarID: {
    // Type ::= %4
    std::pair<Type*, LocTy> &Entry = NumberedTypes[Lex.getUIntVal()];

    // If the type hasn't been defined yet, create a forward definition and
    // remember where that forward def'n was seen (in case it never is defined).
    if (!Entry.first) {
      Entry.first = StructType::create(Context);
      Entry.second = Lex.getLoc();
    }
    Result = Entry.first;
    Lex.Lex();
    break;
  }
  }

  // Parse the type suffixes.
  while (1) {
    switch (Lex.getKind()) {
    // End of type.
    default:
      if (!AllowVoid && Result->isVoidTy())
        return Error(TypeLoc, "void type only allowed for function results");
      return false;

    // Type ::= Type '*'
    case lltok::star:
      if (Result->isLabelTy())
        return TokError("basic block pointers are invalid");
      if (Result->isVoidTy())
        return TokError("pointers to void are invalid - use i8* instead");
      if (!PointerType::isValidElementType(Result))
        return TokError("pointer to this type is invalid");
      Result = PointerType::getUnqual(Result);
      Lex.Lex();
      break;

    // Type ::= Type 'addrspace' '(' uint32 ')' '*'
    case lltok::kw_addrspace: {
      if (Result->isLabelTy())
        return TokError("basic block pointers are invalid");
      if (Result->isVoidTy())
        return TokError("pointers to void are invalid; use i8* instead");
      if (!PointerType::isValidElementType(Result))
        return TokError("pointer to this type is invalid");
      unsigned AddrSpace;
      if (ParseOptionalAddrSpace(AddrSpace) ||
          ParseToken(lltok::star, "expected '*' in address space"))
        return true;

      Result = PointerType::get(Result, AddrSpace);
      break;
    }

    /// Types '(' ArgTypeListI ')' OptFuncAttrs
    case lltok::lparen:
      if (ParseFunctionType(Result))
        return true;
      break;
    }
  }
}

/// ParseParameterList
///    ::= '(' ')'
///    ::= '(' Arg (',' Arg)* ')'
///  Arg
///    ::= Type OptionalAttributes Value OptionalAttributes
bool LLParser::ParseParameterList(SmallVectorImpl<ParamInfo> &ArgList,
                                  PerFunctionState &PFS, bool IsMustTailCall,
                                  bool InVarArgsFunc) {
  if (ParseToken(lltok::lparen, "expected '(' in call"))
    return true;

  unsigned AttrIndex = 1;
  while (Lex.getKind() != lltok::rparen) {
    // If this isn't the first argument, we need a comma.
    if (!ArgList.empty() &&
        ParseToken(lltok::comma, "expected ',' in argument list"))
      return true;

    // Parse an ellipsis if this is a musttail call in a variadic function.
    if (Lex.getKind() == lltok::dotdotdot) {
      const char *Msg = "unexpected ellipsis in argument list for ";
      if (!IsMustTailCall)
        return TokError(Twine(Msg) + "non-musttail call");
      if (!InVarArgsFunc)
        return TokError(Twine(Msg) + "musttail call in non-varargs function");
      Lex.Lex();  // Lex the '...', it is purely for readability.
      return ParseToken(lltok::rparen, "expected ')' at end of argument list");
    }

    // Parse the argument.
    LocTy ArgLoc;
    Type *ArgTy = nullptr;
    AttrBuilder ArgAttrs;
    Value *V;
    if (ParseType(ArgTy, ArgLoc))
      return true;

    if (ArgTy->isMetadataTy()) {
      if (ParseMetadataAsValue(V, PFS))
        return true;
    } else {
      // Otherwise, handle normal operands.
      if (ParseOptionalParamAttrs(ArgAttrs) || ParseValue(ArgTy, V, PFS))
        return true;
    }
    ArgList.push_back(ParamInfo(ArgLoc, V, AttributeSet::get(V->getContext(),
                                                             AttrIndex++,
                                                             ArgAttrs)));
  }

  if (IsMustTailCall && InVarArgsFunc)
    return TokError("expected '...' at end of argument list for musttail call "
                    "in varargs function");

  Lex.Lex();  // Lex the ')'.
  return false;
}



/// ParseArgumentList - Parse the argument list for a function type or function
/// prototype.
///   ::= '(' ArgTypeListI ')'
/// ArgTypeListI
///   ::= /*empty*/
///   ::= '...'
///   ::= ArgTypeList ',' '...'
///   ::= ArgType (',' ArgType)*
///
bool LLParser::ParseArgumentList(SmallVectorImpl<ArgInfo> &ArgList,
                                 bool &isVarArg){
  isVarArg = false;
  assert(Lex.getKind() == lltok::lparen);
  Lex.Lex(); // eat the (.

  if (Lex.getKind() == lltok::rparen) {
    // empty
  } else if (Lex.getKind() == lltok::dotdotdot) {
    isVarArg = true;
    Lex.Lex();
  } else {
    LocTy TypeLoc = Lex.getLoc();
    Type *ArgTy = nullptr;
    AttrBuilder Attrs;
    std::string Name;

    if (ParseType(ArgTy) ||
        ParseOptionalParamAttrs(Attrs)) return true;

    if (ArgTy->isVoidTy())
      return Error(TypeLoc, "argument can not have void type");

    if (Lex.getKind() == lltok::LocalVar) {
      Name = Lex.getStrVal();
      Lex.Lex();
    }

    if (!FunctionType::isValidArgumentType(ArgTy))
      return Error(TypeLoc, "invalid type for function argument");

    unsigned AttrIndex = 1;
    ArgList.emplace_back(TypeLoc, ArgTy, AttributeSet::get(ArgTy->getContext(),
                                                           AttrIndex++, Attrs),
                         std::move(Name));

    while (EatIfPresent(lltok::comma)) {
      // Handle ... at end of arg list.
      if (EatIfPresent(lltok::dotdotdot)) {
        isVarArg = true;
        break;
      }

      // Otherwise must be an argument type.
      TypeLoc = Lex.getLoc();
      if (ParseType(ArgTy) || ParseOptionalParamAttrs(Attrs)) return true;

      if (ArgTy->isVoidTy())
        return Error(TypeLoc, "argument can not have void type");

      if (Lex.getKind() == lltok::LocalVar) {
        Name = Lex.getStrVal();
        Lex.Lex();
      } else {
        Name = "";
      }

      if (!ArgTy->isFirstClassType())
        return Error(TypeLoc, "invalid type for function argument");

      ArgList.emplace_back(
          TypeLoc, ArgTy,
          AttributeSet::get(ArgTy->getContext(), AttrIndex++, Attrs),
          std::move(Name));
    }
  }

  return ParseToken(lltok::rparen, "expected ')' at end of argument list");
}

/// ParseFunctionType
///  ::= Type ArgumentList OptionalAttrs
bool LLParser::ParseFunctionType(Type *&Result) {
  assert(Lex.getKind() == lltok::lparen);

  if (!FunctionType::isValidReturnType(Result))
    return TokError("invalid function return type");

  SmallVector<ArgInfo, 8> ArgList;
  bool isVarArg;
  if (ParseArgumentList(ArgList, isVarArg))
    return true;

  // Reject names on the arguments lists.
  for (unsigned i = 0, e = ArgList.size(); i != e; ++i) {
    if (!ArgList[i].Name.empty())
      return Error(ArgList[i].Loc, "argument name invalid in function type");
    if (ArgList[i].Attrs.hasAttributes(i + 1))
      return Error(ArgList[i].Loc,
                   "argument attributes invalid in function type");
  }

  SmallVector<Type*, 16> ArgListTy;
  for (unsigned i = 0, e = ArgList.size(); i != e; ++i)
    ArgListTy.push_back(ArgList[i].Ty);

  Result = FunctionType::get(Result, ArgListTy, isVarArg);
  return false;
}

/// ParseAnonStructType - Parse an anonymous struct type, which is inlined into
/// other structs.
bool LLParser::ParseAnonStructType(Type *&Result, bool Packed) {
  SmallVector<Type*, 8> Elts;
  if (ParseStructBody(Elts)) return true;

  Result = StructType::get(Context, Elts, Packed);
  return false;
}

/// ParseStructDefinition - Parse a struct in a 'type' definition.
bool LLParser::ParseStructDefinition(SMLoc TypeLoc, StringRef Name,
                                     std::pair<Type*, LocTy> &Entry,
                                     Type *&ResultTy) {
  // If the type was already defined, diagnose the redefinition.
  if (Entry.first && !Entry.second.isValid())
    return Error(TypeLoc, "redefinition of type");

  // If we have opaque, just return without filling in the definition for the
  // struct.  This counts as a definition as far as the .ll file goes.
  if (EatIfPresent(lltok::kw_opaque)) {
    // This type is being defined, so clear the location to indicate this.
    Entry.second = SMLoc();

    // If this type number has never been uttered, create it.
    if (!Entry.first)
      Entry.first = StructType::create(Context, Name);
    ResultTy = Entry.first;
    return false;
  }

  // If the type starts with '<', then it is either a packed struct or a vector.
  bool isPacked = EatIfPresent(lltok::less);

  // If we don't have a struct, then we have a random type alias, which we
  // accept for compatibility with old files.  These types are not allowed to be
  // forward referenced and not allowed to be recursive.
  if (Lex.getKind() != lltok::lbrace) {
    if (Entry.first)
      return Error(TypeLoc, "forward references to non-struct type");

    ResultTy = nullptr;
    if (isPacked)
      return ParseArrayVectorType(ResultTy, true);
    return ParseType(ResultTy);
  }

  // This type is being defined, so clear the location to indicate this.
  Entry.second = SMLoc();

  // If this type number has never been uttered, create it.
  if (!Entry.first)
    Entry.first = StructType::create(Context, Name);

  StructType *STy = cast<StructType>(Entry.first);

  SmallVector<Type*, 8> Body;
  if (ParseStructBody(Body) ||
      (isPacked && ParseToken(lltok::greater, "expected '>' in packed struct")))
    return true;

  STy->setBody(Body, isPacked);
  ResultTy = STy;
  return false;
}


/// ParseStructType: Handles packed and unpacked types.  </> parsed elsewhere.
///   StructType
///     ::= '{' '}'
///     ::= '{' Type (',' Type)* '}'
///     ::= '<' '{' '}' '>'
///     ::= '<' '{' Type (',' Type)* '}' '>'
bool LLParser::ParseStructBody(SmallVectorImpl<Type*> &Body) {
  assert(Lex.getKind() == lltok::lbrace);
  Lex.Lex(); // Consume the '{'

  // Handle the empty struct.
  if (EatIfPresent(lltok::rbrace))
    return false;

  LocTy EltTyLoc = Lex.getLoc();
  Type *Ty = nullptr;
  if (ParseType(Ty)) return true;
  Body.push_back(Ty);

  if (!StructType::isValidElementType(Ty))
    return Error(EltTyLoc, "invalid element type for struct");

  while (EatIfPresent(lltok::comma)) {
    EltTyLoc = Lex.getLoc();
    if (ParseType(Ty)) return true;

    if (!StructType::isValidElementType(Ty))
      return Error(EltTyLoc, "invalid element type for struct");

    Body.push_back(Ty);
  }

  return ParseToken(lltok::rbrace, "expected '}' at end of struct");
}

/// ParseArrayVectorType - Parse an array or vector type, assuming the first
/// token has already been consumed.
///   Type
///     ::= '[' APSINTVAL 'x' Types ']'
///     ::= '<' APSINTVAL 'x' Types '>'
bool LLParser::ParseArrayVectorType(Type *&Result, bool isVector) {
  if (Lex.getKind() != lltok::APSInt || Lex.getAPSIntVal().isSigned() ||
      Lex.getAPSIntVal().getBitWidth() > 64)
    return TokError("expected number in address space");

  LocTy SizeLoc = Lex.getLoc();
  uint64_t Size = Lex.getAPSIntVal().getZExtValue();
  Lex.Lex();

  if (ParseToken(lltok::kw_x, "expected 'x' after element count"))
      return true;

  LocTy TypeLoc = Lex.getLoc();
  Type *EltTy = nullptr;
  if (ParseType(EltTy)) return true;

  if (ParseToken(isVector ? lltok::greater : lltok::rsquare,
                 "expected end of sequential type"))
    return true;

  if (isVector) {
    if (Size == 0)
      return Error(SizeLoc, "zero element vector is illegal");
    if ((unsigned)Size != Size)
      return Error(SizeLoc, "size too large for vector");
    if (!VectorType::isValidElementType(EltTy))
      return Error(TypeLoc, "invalid vector element type");
    Result = VectorType::get(EltTy, unsigned(Size));
  } else {
    if (!ArrayType::isValidElementType(EltTy))
      return Error(TypeLoc, "invalid array element type");
    Result = ArrayType::get(EltTy, Size);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Function Semantic Analysis.
//===----------------------------------------------------------------------===//

LLParser::PerFunctionState::PerFunctionState(LLParser &p, Function &f,
                                             int functionNumber)
  : P(p), F(f), FunctionNumber(functionNumber) {

  // Insert unnamed arguments into the NumberedVals list.
  for (Function::arg_iterator AI = F.arg_begin(), E = F.arg_end();
       AI != E; ++AI)
    if (!AI->hasName())
      NumberedVals.push_back(AI);
}

LLParser::PerFunctionState::~PerFunctionState() {
  // If there were any forward referenced non-basicblock values, delete them.
  for (std::map<std::string, std::pair<Value*, LocTy> >::iterator
       I = ForwardRefVals.begin(), E = ForwardRefVals.end(); I != E; ++I)
    if (!isa<BasicBlock>(I->second.first)) {
      I->second.first->replaceAllUsesWith(
                           UndefValue::get(I->second.first->getType()));
      delete I->second.first;
      I->second.first = nullptr;
    }

  for (std::map<unsigned, std::pair<Value*, LocTy> >::iterator
       I = ForwardRefValIDs.begin(), E = ForwardRefValIDs.end(); I != E; ++I)
    if (!isa<BasicBlock>(I->second.first)) {
      I->second.first->replaceAllUsesWith(
                           UndefValue::get(I->second.first->getType()));
      delete I->second.first;
      I->second.first = nullptr;
    }
}

bool LLParser::PerFunctionState::FinishFunction() {
  if (!ForwardRefVals.empty())
    return P.Error(ForwardRefVals.begin()->second.second,
                   "use of undefined value '%" + ForwardRefVals.begin()->first +
                   "'");
  if (!ForwardRefValIDs.empty())
    return P.Error(ForwardRefValIDs.begin()->second.second,
                   "use of undefined value '%" +
                   Twine(ForwardRefValIDs.begin()->first) + "'");
  return false;
}


/// GetVal - Get a value with the specified name or ID, creating a
/// forward reference record if needed.  This can return null if the value
/// exists but does not have the right type.
Value *LLParser::PerFunctionState::GetVal(const std::string &Name,
                                          Type *Ty, LocTy Loc) {
  // Look this name up in the normal function symbol table.
  Value *Val = F.getValueSymbolTable().lookup(Name);

  // If this is a forward reference for the value, see if we already created a
  // forward ref record.
  if (!Val) {
    std::map<std::string, std::pair<Value*, LocTy> >::iterator
      I = ForwardRefVals.find(Name);
    if (I != ForwardRefVals.end())
      Val = I->second.first;
  }

  // If we have the value in the symbol table or fwd-ref table, return it.
  if (Val) {
    if (Val->getType() == Ty) return Val;
    if (Ty->isLabelTy())
      P.Error(Loc, "'%" + Name + "' is not a basic block");
    else
      P.Error(Loc, "'%" + Name + "' defined with type '" +
              getTypeString(Val->getType()) + "'");
    return nullptr;
  }

  // Don't make placeholders with invalid type.
  if (!Ty->isFirstClassType()) {
    P.Error(Loc, "invalid use of a non-first-class type");
    return nullptr;
  }

  // Otherwise, create a new forward reference for this value and remember it.
  Value *FwdVal;
  if (Ty->isLabelTy())
    FwdVal = BasicBlock::Create(F.getContext(), Name, &F);
  else
    FwdVal = new Argument(Ty, Name);

  ForwardRefVals[Name] = std::make_pair(FwdVal, Loc);
  return FwdVal;
}

Value *LLParser::PerFunctionState::GetVal(unsigned ID, Type *Ty,
                                          LocTy Loc) {
  // Look this name up in the normal function symbol table.
  Value *Val = ID < NumberedVals.size() ? NumberedVals[ID] : nullptr;

  // If this is a forward reference for the value, see if we already created a
  // forward ref record.
  if (!Val) {
    std::map<unsigned, std::pair<Value*, LocTy> >::iterator
      I = ForwardRefValIDs.find(ID);
    if (I != ForwardRefValIDs.end())
      Val = I->second.first;
  }

  // If we have the value in the symbol table or fwd-ref table, return it.
  if (Val) {
    if (Val->getType() == Ty) return Val;
    if (Ty->isLabelTy())
      P.Error(Loc, "'%" + Twine(ID) + "' is not a basic block");
    else
      P.Error(Loc, "'%" + Twine(ID) + "' defined with type '" +
              getTypeString(Val->getType()) + "'");
    return nullptr;
  }

  if (!Ty->isFirstClassType()) {
    P.Error(Loc, "invalid use of a non-first-class type");
    return nullptr;
  }

  // Otherwise, create a new forward reference for this value and remember it.
  Value *FwdVal;
  if (Ty->isLabelTy())
    FwdVal = BasicBlock::Create(F.getContext(), "", &F);
  else
    FwdVal = new Argument(Ty);

  ForwardRefValIDs[ID] = std::make_pair(FwdVal, Loc);
  return FwdVal;
}

/// SetInstName - After an instruction is parsed and inserted into its
/// basic block, this installs its name.
bool LLParser::PerFunctionState::SetInstName(int NameID,
                                             const std::string &NameStr,
                                             LocTy NameLoc, Instruction *Inst) {
  // If this instruction has void type, it cannot have a name or ID specified.
  if (Inst->getType()->isVoidTy()) {
    if (NameID != -1 || !NameStr.empty())
      return P.Error(NameLoc, "instructions returning void cannot have a name");
    return false;
  }

  // If this was a numbered instruction, verify that the instruction is the
  // expected value and resolve any forward references.
  if (NameStr.empty()) {
    // If neither a name nor an ID was specified, just use the next ID.
    if (NameID == -1)
      NameID = NumberedVals.size();

    if (unsigned(NameID) != NumberedVals.size())
      return P.Error(NameLoc, "instruction expected to be numbered '%" +
                     Twine(NumberedVals.size()) + "'");

    std::map<unsigned, std::pair<Value*, LocTy> >::iterator FI =
      ForwardRefValIDs.find(NameID);
    if (FI != ForwardRefValIDs.end()) {
      if (FI->second.first->getType() != Inst->getType())
        return P.Error(NameLoc, "instruction forward referenced with type '" +
                       getTypeString(FI->second.first->getType()) + "'");
      FI->second.first->replaceAllUsesWith(Inst);
      delete FI->second.first;
      ForwardRefValIDs.erase(FI);
    }

    NumberedVals.push_back(Inst);
    return false;
  }

  // Otherwise, the instruction had a name.  Resolve forward refs and set it.
  std::map<std::string, std::pair<Value*, LocTy> >::iterator
    FI = ForwardRefVals.find(NameStr);
  if (FI != ForwardRefVals.end()) {
    if (FI->second.first->getType() != Inst->getType())
      return P.Error(NameLoc, "instruction forward referenced with type '" +
                     getTypeString(FI->second.first->getType()) + "'");
    FI->second.first->replaceAllUsesWith(Inst);
    delete FI->second.first;
    ForwardRefVals.erase(FI);
  }

  // Set the name on the instruction.
  Inst->setName(NameStr);

  if (Inst->getName() != NameStr)
    return P.Error(NameLoc, "multiple definition of local value named '" +
                   NameStr + "'");
  return false;
}

/// GetBB - Get a basic block with the specified name or ID, creating a
/// forward reference record if needed.
BasicBlock *LLParser::PerFunctionState::GetBB(const std::string &Name,
                                              LocTy Loc) {
  return dyn_cast_or_null<BasicBlock>(GetVal(Name,
                                      Type::getLabelTy(F.getContext()), Loc));
}

BasicBlock *LLParser::PerFunctionState::GetBB(unsigned ID, LocTy Loc) {
  return dyn_cast_or_null<BasicBlock>(GetVal(ID,
                                      Type::getLabelTy(F.getContext()), Loc));
}

/// DefineBB - Define the specified basic block, which is either named or
/// unnamed.  If there is an error, this returns null otherwise it returns
/// the block being defined.
BasicBlock *LLParser::PerFunctionState::DefineBB(const std::string &Name,
                                                 LocTy Loc) {
  BasicBlock *BB;
  if (Name.empty())
    BB = GetBB(NumberedVals.size(), Loc);
  else
    BB = GetBB(Name, Loc);
  if (!BB) return nullptr; // Already diagnosed error.

  // Move the block to the end of the function.  Forward ref'd blocks are
  // inserted wherever they happen to be referenced.
  F.getBasicBlockList().splice(F.end(), F.getBasicBlockList(), BB);

  // Remove the block from forward ref sets.
  if (Name.empty()) {
    ForwardRefValIDs.erase(NumberedVals.size());
    NumberedVals.push_back(BB);
  } else {
    // BB forward references are already in the function symbol table.
    ForwardRefVals.erase(Name);
  }

  return BB;
}

//===----------------------------------------------------------------------===//
// Constants.
//===----------------------------------------------------------------------===//

/// ParseValID - Parse an abstract value that doesn't necessarily have a
/// type implied.  For example, if we parse "4" we don't know what integer type
/// it has.  The value will later be combined with its type and checked for
/// sanity.  PFS is used to convert function-local operands of metadata (since
/// metadata operands are not just parsed here but also converted to values).
/// PFS can be null when we are not parsing metadata values inside a function.
bool LLParser::ParseValID(ValID &ID, PerFunctionState *PFS) {
  ID.Loc = Lex.getLoc();
  switch (Lex.getKind()) {
  default: return TokError("expected value token");
  case lltok::GlobalID:  // @42
    ID.UIntVal = Lex.getUIntVal();
    ID.Kind = ValID::t_GlobalID;
    break;
  case lltok::GlobalVar:  // @foo
    ID.StrVal = Lex.getStrVal();
    ID.Kind = ValID::t_GlobalName;
    break;
  case lltok::LocalVarID:  // %42
    ID.UIntVal = Lex.getUIntVal();
    ID.Kind = ValID::t_LocalID;
    break;
  case lltok::LocalVar:  // %foo
    ID.StrVal = Lex.getStrVal();
    ID.Kind = ValID::t_LocalName;
    break;
  case lltok::APSInt:
    ID.APSIntVal = Lex.getAPSIntVal();
    ID.Kind = ValID::t_APSInt;
    break;
  case lltok::APFloat:
    ID.APFloatVal = Lex.getAPFloatVal();
    ID.Kind = ValID::t_APFloat;
    break;
  case lltok::kw_true:
    ID.ConstantVal = ConstantInt::getTrue(Context);
    ID.Kind = ValID::t_Constant;
    break;
  case lltok::kw_false:
    ID.ConstantVal = ConstantInt::getFalse(Context);
    ID.Kind = ValID::t_Constant;
    break;
  case lltok::kw_null: ID.Kind = ValID::t_Null; break;
  case lltok::kw_undef: ID.Kind = ValID::t_Undef; break;
  case lltok::kw_zeroinitializer: ID.Kind = ValID::t_Zero; break;

  case lltok::lbrace: {
    // ValID ::= '{' ConstVector '}'
    Lex.Lex();
    SmallVector<Constant*, 16> Elts;
    if (ParseGlobalValueVector(Elts) ||
        ParseToken(lltok::rbrace, "expected end of struct constant"))
      return true;

    ID.ConstantStructElts = new Constant*[Elts.size()];
    ID.UIntVal = Elts.size();
    memcpy(ID.ConstantStructElts, Elts.data(), Elts.size()*sizeof(Elts[0]));
    ID.Kind = ValID::t_ConstantStruct;
    return false;
  }
  case lltok::less: {
    // ValID ::= '<' ConstVector '>'         --> Vector.
    // ValID ::= '<' '{' ConstVector '}' '>' --> Packed Struct.
    Lex.Lex();
    bool isPackedStruct = EatIfPresent(lltok::lbrace);

    SmallVector<Constant*, 16> Elts;
    LocTy FirstEltLoc = Lex.getLoc();
    if (ParseGlobalValueVector(Elts) ||
        (isPackedStruct &&
         ParseToken(lltok::rbrace, "expected end of packed struct")) ||
        ParseToken(lltok::greater, "expected end of constant"))
      return true;

    if (isPackedStruct) {
      ID.ConstantStructElts = new Constant*[Elts.size()];
      memcpy(ID.ConstantStructElts, Elts.data(), Elts.size()*sizeof(Elts[0]));
      ID.UIntVal = Elts.size();
      ID.Kind = ValID::t_PackedConstantStruct;
      return false;
    }

    if (Elts.empty())
      return Error(ID.Loc, "constant vector must not be empty");

    if (!Elts[0]->getType()->isIntegerTy() &&
        !Elts[0]->getType()->isFloatingPointTy() &&
        !Elts[0]->getType()->isPointerTy())
      return Error(FirstEltLoc,
            "vector elements must have integer, pointer or floating point type");

    // Verify that all the vector elements have the same type.
    for (unsigned i = 1, e = Elts.size(); i != e; ++i)
      if (Elts[i]->getType() != Elts[0]->getType())
        return Error(FirstEltLoc,
                     "vector element #" + Twine(i) +
                    " is not of type '" + getTypeString(Elts[0]->getType()));

    ID.ConstantVal = ConstantVector::get(Elts);
    ID.Kind = ValID::t_Constant;
    return false;
  }
  case lltok::lsquare: {   // Array Constant
    Lex.Lex();
    SmallVector<Constant*, 16> Elts;
    LocTy FirstEltLoc = Lex.getLoc();
    if (ParseGlobalValueVector(Elts) ||
        ParseToken(lltok::rsquare, "expected end of array constant"))
      return true;

    // Handle empty element.
    if (Elts.empty()) {
      // Use undef instead of an array because it's inconvenient to determine
      // the element type at this point, there being no elements to examine.
      ID.Kind = ValID::t_EmptyArray;
      return false;
    }

    if (!Elts[0]->getType()->isFirstClassType())
      return Error(FirstEltLoc, "invalid array element type: " +
                   getTypeString(Elts[0]->getType()));

    ArrayType *ATy = ArrayType::get(Elts[0]->getType(), Elts.size());

    // Verify all elements are correct type!
    for (unsigned i = 0, e = Elts.size(); i != e; ++i) {
      if (Elts[i]->getType() != Elts[0]->getType())
        return Error(FirstEltLoc,
                     "array element #" + Twine(i) +
                     " is not of type '" + getTypeString(Elts[0]->getType()));
    }

    ID.ConstantVal = ConstantArray::get(ATy, Elts);
    ID.Kind = ValID::t_Constant;
    return false;
  }
  case lltok::kw_c:  // c "foo"
    Lex.Lex();
    ID.ConstantVal = ConstantDataArray::getString(Context, Lex.getStrVal(),
                                                  false);
    if (ParseToken(lltok::StringConstant, "expected string")) return true;
    ID.Kind = ValID::t_Constant;
    return false;

  case lltok::kw_asm: {
    // ValID ::= 'asm' SideEffect? AlignStack? IntelDialect? STRINGCONSTANT ','
    //             STRINGCONSTANT
    bool HasSideEffect, AlignStack, AsmDialect;
    Lex.Lex();
    if (ParseOptionalToken(lltok::kw_sideeffect, HasSideEffect) ||
        ParseOptionalToken(lltok::kw_alignstack, AlignStack) ||
        ParseOptionalToken(lltok::kw_inteldialect, AsmDialect) ||
        ParseStringConstant(ID.StrVal) ||
        ParseToken(lltok::comma, "expected comma in inline asm expression") ||
        ParseToken(lltok::StringConstant, "expected constraint string"))
      return true;
    ID.StrVal2 = Lex.getStrVal();
    ID.UIntVal = unsigned(HasSideEffect) | (unsigned(AlignStack)<<1) |
      (unsigned(AsmDialect)<<2);
    ID.Kind = ValID::t_InlineAsm;
    return false;
  }

  case lltok::kw_blockaddress: {
    // ValID ::= 'blockaddress' '(' @foo ',' %bar ')'
    Lex.Lex();

    ValID Fn, Label;

    if (ParseToken(lltok::lparen, "expected '(' in block address expression") ||
        ParseValID(Fn) ||
        ParseToken(lltok::comma, "expected comma in block address expression")||
        ParseValID(Label) ||
        ParseToken(lltok::rparen, "expected ')' in block address expression"))
      return true;

    if (Fn.Kind != ValID::t_GlobalID && Fn.Kind != ValID::t_GlobalName)
      return Error(Fn.Loc, "expected function name in blockaddress");
    if (Label.Kind != ValID::t_LocalID && Label.Kind != ValID::t_LocalName)
      return Error(Label.Loc, "expected basic block name in blockaddress");

    // Try to find the function (but skip it if it's forward-referenced).
    GlobalValue *GV = nullptr;
    if (Fn.Kind == ValID::t_GlobalID) {
      if (Fn.UIntVal < NumberedVals.size())
        GV = NumberedVals[Fn.UIntVal];
    } else if (!ForwardRefVals.count(Fn.StrVal)) {
      GV = M->getNamedValue(Fn.StrVal);
    }
    Function *F = nullptr;
    if (GV) {
      // Confirm that it's actually a function with a definition.
      if (!isa<Function>(GV))
        return Error(Fn.Loc, "expected function name in blockaddress");
      F = cast<Function>(GV);
      if (F->isDeclaration())
        return Error(Fn.Loc, "cannot take blockaddress inside a declaration");
    }

    if (!F) {
      // Make a global variable as a placeholder for this reference.
      GlobalValue *&FwdRef =
          ForwardRefBlockAddresses.insert(std::make_pair(
                                              std::move(Fn),
                                              std::map<ValID, GlobalValue *>()))
              .first->second.insert(std::make_pair(std::move(Label), nullptr))
              .first->second;
      if (!FwdRef)
        FwdRef = new GlobalVariable(*M, Type::getInt8Ty(Context), false,
                                    GlobalValue::InternalLinkage, nullptr, "");
      ID.ConstantVal = FwdRef;
      ID.Kind = ValID::t_Constant;
      return false;
    }

    // We found the function; now find the basic block.  Don't use PFS, since we
    // might be inside a constant expression.
    BasicBlock *BB;
    if (BlockAddressPFS && F == &BlockAddressPFS->getFunction()) {
      if (Label.Kind == ValID::t_LocalID)
        BB = BlockAddressPFS->GetBB(Label.UIntVal, Label.Loc);
      else
        BB = BlockAddressPFS->GetBB(Label.StrVal, Label.Loc);
      if (!BB)
        return Error(Label.Loc, "referenced value is not a basic block");
    } else {
      if (Label.Kind == ValID::t_LocalID)
        return Error(Label.Loc, "cannot take address of numeric label after "
                                "the function is defined");
      BB = dyn_cast_or_null<BasicBlock>(
          F->getValueSymbolTable().lookup(Label.StrVal));
      if (!BB)
        return Error(Label.Loc, "referenced value is not a basic block");
    }

    ID.ConstantVal = BlockAddress::get(F, BB);
    ID.Kind = ValID::t_Constant;
    return false;
  }

  case lltok::kw_trunc:
  case lltok::kw_zext:
  case lltok::kw_sext:
  case lltok::kw_fptrunc:
  case lltok::kw_fpext:
  case lltok::kw_bitcast:
  case lltok::kw_addrspacecast:
  case lltok::kw_uitofp:
  case lltok::kw_sitofp:
  case lltok::kw_fptoui:
  case lltok::kw_fptosi:
  case lltok::kw_inttoptr:
  case lltok::kw_ptrtoint: {
    unsigned Opc = Lex.getUIntVal();
    Type *DestTy = nullptr;
    Constant *SrcVal;
    Lex.Lex();
    if (ParseToken(lltok::lparen, "expected '(' after constantexpr cast") ||
        ParseGlobalTypeAndValue(SrcVal) ||
        ParseToken(lltok::kw_to, "expected 'to' in constantexpr cast") ||
        ParseType(DestTy) ||
        ParseToken(lltok::rparen, "expected ')' at end of constantexpr cast"))
      return true;
    if (!CastInst::castIsValid((Instruction::CastOps)Opc, SrcVal, DestTy))
      return Error(ID.Loc, "invalid cast opcode for cast from '" +
                   getTypeString(SrcVal->getType()) + "' to '" +
                   getTypeString(DestTy) + "'");
    ID.ConstantVal = ConstantExpr::getCast((Instruction::CastOps)Opc,
                                                 SrcVal, DestTy);
    ID.Kind = ValID::t_Constant;
    return false;
  }
  case lltok::kw_extractvalue: {
    Lex.Lex();
    Constant *Val;
    SmallVector<unsigned, 4> Indices;
    if (ParseToken(lltok::lparen, "expected '(' in extractvalue constantexpr")||
        ParseGlobalTypeAndValue(Val) ||
        ParseIndexList(Indices) ||
        ParseToken(lltok::rparen, "expected ')' in extractvalue constantexpr"))
      return true;

    if (!Val->getType()->isAggregateType())
      return Error(ID.Loc, "extractvalue operand must be aggregate type");
    if (!ExtractValueInst::getIndexedType(Val->getType(), Indices))
      return Error(ID.Loc, "invalid indices for extractvalue");
    ID.ConstantVal = ConstantExpr::getExtractValue(Val, Indices);
    ID.Kind = ValID::t_Constant;
    return false;
  }
  case lltok::kw_insertvalue: {
    Lex.Lex();
    Constant *Val0, *Val1;
    SmallVector<unsigned, 4> Indices;
    if (ParseToken(lltok::lparen, "expected '(' in insertvalue constantexpr")||
        ParseGlobalTypeAndValue(Val0) ||
        ParseToken(lltok::comma, "expected comma in insertvalue constantexpr")||
        ParseGlobalTypeAndValue(Val1) ||
        ParseIndexList(Indices) ||
        ParseToken(lltok::rparen, "expected ')' in insertvalue constantexpr"))
      return true;
    if (!Val0->getType()->isAggregateType())
      return Error(ID.Loc, "insertvalue operand must be aggregate type");
    Type *IndexedType =
        ExtractValueInst::getIndexedType(Val0->getType(), Indices);
    if (!IndexedType)
      return Error(ID.Loc, "invalid indices for insertvalue");
    if (IndexedType != Val1->getType())
      return Error(ID.Loc, "insertvalue operand and field disagree in type: '" +
                               getTypeString(Val1->getType()) +
                               "' instead of '" + getTypeString(IndexedType) +
                               "'");
    ID.ConstantVal = ConstantExpr::getInsertValue(Val0, Val1, Indices);
    ID.Kind = ValID::t_Constant;
    return false;
  }
  case lltok::kw_icmp:
  case lltok::kw_fcmp: {
    unsigned PredVal, Opc = Lex.getUIntVal();
    Constant *Val0, *Val1;
    Lex.Lex();
    if (ParseCmpPredicate(PredVal, Opc) ||
        ParseToken(lltok::lparen, "expected '(' in compare constantexpr") ||
        ParseGlobalTypeAndValue(Val0) ||
        ParseToken(lltok::comma, "expected comma in compare constantexpr") ||
        ParseGlobalTypeAndValue(Val1) ||
        ParseToken(lltok::rparen, "expected ')' in compare constantexpr"))
      return true;

    if (Val0->getType() != Val1->getType())
      return Error(ID.Loc, "compare operands must have the same type");

    CmpInst::Predicate Pred = (CmpInst::Predicate)PredVal;

    if (Opc == Instruction::FCmp) {
      if (!Val0->getType()->isFPOrFPVectorTy())
        return Error(ID.Loc, "fcmp requires floating point operands");
      ID.ConstantVal = ConstantExpr::getFCmp(Pred, Val0, Val1);
    } else {
      assert(Opc == Instruction::ICmp && "Unexpected opcode for CmpInst!");
      if (!Val0->getType()->isIntOrIntVectorTy() &&
          !Val0->getType()->getScalarType()->isPointerTy())
        return Error(ID.Loc, "icmp requires pointer or integer operands");
      ID.ConstantVal = ConstantExpr::getICmp(Pred, Val0, Val1);
    }
    ID.Kind = ValID::t_Constant;
    return false;
  }

  // Binary Operators.
  case lltok::kw_add:
  case lltok::kw_fadd:
  case lltok::kw_sub:
  case lltok::kw_fsub:
  case lltok::kw_mul:
  case lltok::kw_fmul:
  case lltok::kw_udiv:
  case lltok::kw_sdiv:
  case lltok::kw_fdiv:
  case lltok::kw_urem:
  case lltok::kw_srem:
  case lltok::kw_frem:
  case lltok::kw_shl:
  case lltok::kw_lshr:
  case lltok::kw_ashr: {
    bool NUW = false;
    bool NSW = false;
    bool Exact = false;
    unsigned Opc = Lex.getUIntVal();
    Constant *Val0, *Val1;
    Lex.Lex();
    LocTy ModifierLoc = Lex.getLoc();
    if (Opc == Instruction::Add || Opc == Instruction::Sub ||
        Opc == Instruction::Mul || Opc == Instruction::Shl) {
      if (EatIfPresent(lltok::kw_nuw))
        NUW = true;
      if (EatIfPresent(lltok::kw_nsw)) {
        NSW = true;
        if (EatIfPresent(lltok::kw_nuw))
          NUW = true;
      }
    } else if (Opc == Instruction::SDiv || Opc == Instruction::UDiv ||
               Opc == Instruction::LShr || Opc == Instruction::AShr) {
      if (EatIfPresent(lltok::kw_exact))
        Exact = true;
    }
    if (ParseToken(lltok::lparen, "expected '(' in binary constantexpr") ||
        ParseGlobalTypeAndValue(Val0) ||
        ParseToken(lltok::comma, "expected comma in binary constantexpr") ||
        ParseGlobalTypeAndValue(Val1) ||
        ParseToken(lltok::rparen, "expected ')' in binary constantexpr"))
      return true;
    if (Val0->getType() != Val1->getType())
      return Error(ID.Loc, "operands of constexpr must have same type");
    if (!Val0->getType()->isIntOrIntVectorTy()) {
      if (NUW)
        return Error(ModifierLoc, "nuw only applies to integer operations");
      if (NSW)
        return Error(ModifierLoc, "nsw only applies to integer operations");
    }
    // Check that the type is valid for the operator.
    switch (Opc) {
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::Shl:
    case Instruction::AShr:
    case Instruction::LShr:
      if (!Val0->getType()->isIntOrIntVectorTy())
        return Error(ID.Loc, "constexpr requires integer operands");
      break;
    case Instruction::FAdd:
    case Instruction::FSub:
    case Instruction::FMul:
    case Instruction::FDiv:
    case Instruction::FRem:
      if (!Val0->getType()->isFPOrFPVectorTy())
        return Error(ID.Loc, "constexpr requires fp operands");
      break;
    default: llvm_unreachable("Unknown binary operator!");
    }
    unsigned Flags = 0;
    if (NUW)   Flags |= OverflowingBinaryOperator::NoUnsignedWrap;
    if (NSW)   Flags |= OverflowingBinaryOperator::NoSignedWrap;
    if (Exact) Flags |= PossiblyExactOperator::IsExact;
    Constant *C = ConstantExpr::get(Opc, Val0, Val1, Flags);
    ID.ConstantVal = C;
    ID.Kind = ValID::t_Constant;
    return false;
  }

  // Logical Operations
  case lltok::kw_and:
  case lltok::kw_or:
  case lltok::kw_xor: {
    unsigned Opc = Lex.getUIntVal();
    Constant *Val0, *Val1;
    Lex.Lex();
    if (ParseToken(lltok::lparen, "expected '(' in logical constantexpr") ||
        ParseGlobalTypeAndValue(Val0) ||
        ParseToken(lltok::comma, "expected comma in logical constantexpr") ||
        ParseGlobalTypeAndValue(Val1) ||
        ParseToken(lltok::rparen, "expected ')' in logical constantexpr"))
      return true;
    if (Val0->getType() != Val1->getType())
      return Error(ID.Loc, "operands of constexpr must have same type");
    if (!Val0->getType()->isIntOrIntVectorTy())
      return Error(ID.Loc,
                   "constexpr requires integer or integer vector operands");
    ID.ConstantVal = ConstantExpr::get(Opc, Val0, Val1);
    ID.Kind = ValID::t_Constant;
    return false;
  }

  case lltok::kw_getelementptr:
  case lltok::kw_shufflevector:
  case lltok::kw_insertelement:
  case lltok::kw_extractelement:
  case lltok::kw_select: {
    unsigned Opc = Lex.getUIntVal();
    SmallVector<Constant*, 16> Elts;
    bool InBounds = false;
    Type *Ty;
    Lex.Lex();

    if (Opc == Instruction::GetElementPtr)
      InBounds = EatIfPresent(lltok::kw_inbounds);

    if (ParseToken(lltok::lparen, "expected '(' in constantexpr"))
      return true;

    LocTy ExplicitTypeLoc = Lex.getLoc();
    if (Opc == Instruction::GetElementPtr) {
      if (ParseType(Ty) ||
          ParseToken(lltok::comma, "expected comma after getelementptr's type"))
        return true;
    }

    if (ParseGlobalValueVector(Elts) ||
        ParseToken(lltok::rparen, "expected ')' in constantexpr"))
      return true;

    if (Opc == Instruction::GetElementPtr) {
      if (Elts.size() == 0 ||
          !Elts[0]->getType()->getScalarType()->isPointerTy())
        return Error(ID.Loc, "base of getelementptr must be a pointer");

      Type *BaseType = Elts[0]->getType();
      auto *BasePointerType = cast<PointerType>(BaseType->getScalarType());
      if (Ty != BasePointerType->getElementType())
        return Error(
            ExplicitTypeLoc,
            "explicit pointee type doesn't match operand's pointee type");

      ArrayRef<Constant *> Indices(Elts.begin() + 1, Elts.end());
      for (Constant *Val : Indices) {
        Type *ValTy = Val->getType();
        if (!ValTy->getScalarType()->isIntegerTy())
          return Error(ID.Loc, "getelementptr index must be an integer");
        if (ValTy->isVectorTy() != BaseType->isVectorTy())
          return Error(ID.Loc, "getelementptr index type missmatch");
        if (ValTy->isVectorTy()) {
          unsigned ValNumEl = ValTy->getVectorNumElements();
          unsigned PtrNumEl = BaseType->getVectorNumElements();
          if (ValNumEl != PtrNumEl)
            return Error(
                ID.Loc,
                "getelementptr vector index has a wrong number of elements");
        }
      }

      SmallPtrSet<const Type*, 4> Visited;
      if (!Indices.empty() && !Ty->isSized(&Visited))
        return Error(ID.Loc, "base element of getelementptr must be sized");

      if (!GetElementPtrInst::getIndexedType(Ty, Indices))
        return Error(ID.Loc, "invalid getelementptr indices");
      ID.ConstantVal =
          ConstantExpr::getGetElementPtr(Ty, Elts[0], Indices, InBounds);
    } else if (Opc == Instruction::Select) {
      if (Elts.size() != 3)
        return Error(ID.Loc, "expected three operands to select");
      if (const char *Reason = SelectInst::areInvalidOperands(Elts[0], Elts[1],
                                                              Elts[2]))
        return Error(ID.Loc, Reason);
      ID.ConstantVal = ConstantExpr::getSelect(Elts[0], Elts[1], Elts[2]);
    } else if (Opc == Instruction::ShuffleVector) {
      if (Elts.size() != 3)
        return Error(ID.Loc, "expected three operands to shufflevector");
      if (!ShuffleVectorInst::isValidOperands(Elts[0], Elts[1], Elts[2]))
        return Error(ID.Loc, "invalid operands to shufflevector");
      ID.ConstantVal =
                 ConstantExpr::getShuffleVector(Elts[0], Elts[1],Elts[2]);
    } else if (Opc == Instruction::ExtractElement) {
      if (Elts.size() != 2)
        return Error(ID.Loc, "expected two operands to extractelement");
      if (!ExtractElementInst::isValidOperands(Elts[0], Elts[1]))
        return Error(ID.Loc, "invalid extractelement operands");
      ID.ConstantVal = ConstantExpr::getExtractElement(Elts[0], Elts[1]);
    } else {
      assert(Opc == Instruction::InsertElement && "Unknown opcode");
      if (Elts.size() != 3)
      return Error(ID.Loc, "expected three operands to insertelement");
      if (!InsertElementInst::isValidOperands(Elts[0], Elts[1], Elts[2]))
        return Error(ID.Loc, "invalid insertelement operands");
      ID.ConstantVal =
                 ConstantExpr::getInsertElement(Elts[0], Elts[1],Elts[2]);
    }

    ID.Kind = ValID::t_Constant;
    return false;
  }
  }

  Lex.Lex();
  return false;
}

/// ParseGlobalValue - Parse a global value with the specified type.
bool LLParser::ParseGlobalValue(Type *Ty, Constant *&C) {
  C = nullptr;
  ValID ID;
  Value *V = nullptr;
  bool Parsed = ParseValID(ID) ||
                ConvertValIDToValue(Ty, ID, V, nullptr);
  if (V && !(C = dyn_cast<Constant>(V)))
    return Error(ID.Loc, "global values must be constants");
  return Parsed;
}

bool LLParser::ParseGlobalTypeAndValue(Constant *&V) {
  Type *Ty = nullptr;
  return ParseType(Ty) ||
         ParseGlobalValue(Ty, V);
}

bool LLParser::parseOptionalComdat(StringRef GlobalName, Comdat *&C) {
  C = nullptr;

  LocTy KwLoc = Lex.getLoc();
  if (!EatIfPresent(lltok::kw_comdat))
    return false;

  if (EatIfPresent(lltok::lparen)) {
    if (Lex.getKind() != lltok::ComdatVar)
      return TokError("expected comdat variable");
    C = getComdat(Lex.getStrVal(), Lex.getLoc());
    Lex.Lex();
    if (ParseToken(lltok::rparen, "expected ')' after comdat var"))
      return true;
  } else {
    if (GlobalName.empty())
      return TokError("comdat cannot be unnamed");
    C = getComdat(GlobalName, KwLoc);
  }

  return false;
}

/// ParseGlobalValueVector
///   ::= /*empty*/
///   ::= TypeAndValue (',' TypeAndValue)*
bool LLParser::ParseGlobalValueVector(SmallVectorImpl<Constant *> &Elts) {
  // Empty list.
  if (Lex.getKind() == lltok::rbrace ||
      Lex.getKind() == lltok::rsquare ||
      Lex.getKind() == lltok::greater ||
      Lex.getKind() == lltok::rparen)
    return false;

  Constant *C;
  if (ParseGlobalTypeAndValue(C)) return true;
  Elts.push_back(C);

  while (EatIfPresent(lltok::comma)) {
    if (ParseGlobalTypeAndValue(C)) return true;
    Elts.push_back(C);
  }

  return false;
}

bool LLParser::ParseMDTuple(MDNode *&MD, bool IsDistinct) {
  SmallVector<Metadata *, 16> Elts;
  if (ParseMDNodeVector(Elts))
    return true;

  MD = (IsDistinct ? MDTuple::getDistinct : MDTuple::get)(Context, Elts);
  return false;
}

/// MDNode:
///  ::= !{ ... }
///  ::= !7
///  ::= !DILocation(...)
bool LLParser::ParseMDNode(MDNode *&N) {
  if (Lex.getKind() == lltok::MetadataVar)
    return ParseSpecializedMDNode(N);

  return ParseToken(lltok::exclaim, "expected '!' here") ||
         ParseMDNodeTail(N);
}

bool LLParser::ParseMDNodeTail(MDNode *&N) {
  // !{ ... }
  if (Lex.getKind() == lltok::lbrace)
    return ParseMDTuple(N);

  // !42
  return ParseMDNodeID(N);
}

namespace {

/// Structure to represent an optional metadata field.
template <class FieldTy> struct MDFieldImpl {
  typedef MDFieldImpl ImplTy;
  FieldTy Val;
  bool Seen;

  void assign(FieldTy Val) {
    Seen = true;
    this->Val = std::move(Val);
  }

  explicit MDFieldImpl(FieldTy Default)
      : Val(std::move(Default)), Seen(false) {}
};

struct MDUnsignedField : public MDFieldImpl<uint64_t> {
  uint64_t Max;

  MDUnsignedField(uint64_t Default = 0, uint64_t Max = UINT64_MAX)
      : ImplTy(Default), Max(Max) {}
};
struct LineField : public MDUnsignedField {
  LineField() : MDUnsignedField(0, UINT32_MAX) {}
};
struct ColumnField : public MDUnsignedField {
  ColumnField() : MDUnsignedField(0, UINT16_MAX) {}
};
struct DwarfTagField : public MDUnsignedField {
  DwarfTagField() : MDUnsignedField(0, dwarf::DW_TAG_hi_user) {}
  DwarfTagField(dwarf::Tag DefaultTag)
      : MDUnsignedField(DefaultTag, dwarf::DW_TAG_hi_user) {}
};
struct DwarfAttEncodingField : public MDUnsignedField {
  DwarfAttEncodingField() : MDUnsignedField(0, dwarf::DW_ATE_hi_user) {}
};
struct DwarfVirtualityField : public MDUnsignedField {
  DwarfVirtualityField() : MDUnsignedField(0, dwarf::DW_VIRTUALITY_max) {}
};
struct DwarfLangField : public MDUnsignedField {
  DwarfLangField() : MDUnsignedField(0, dwarf::DW_LANG_hi_user) {}
};

struct DIFlagField : public MDUnsignedField {
  DIFlagField() : MDUnsignedField(0, UINT32_MAX) {}
};

struct MDSignedField : public MDFieldImpl<int64_t> {
  int64_t Min;
  int64_t Max;

  MDSignedField(int64_t Default = 0)
      : ImplTy(Default), Min(INT64_MIN), Max(INT64_MAX) {}
  MDSignedField(int64_t Default, int64_t Min, int64_t Max)
      : ImplTy(Default), Min(Min), Max(Max) {}
};

struct MDBoolField : public MDFieldImpl<bool> {
  MDBoolField(bool Default = false) : ImplTy(Default) {}
};
struct MDField : public MDFieldImpl<Metadata *> {
  bool AllowNull;

  MDField(bool AllowNull = true) : ImplTy(nullptr), AllowNull(AllowNull) {}
};
struct MDConstant : public MDFieldImpl<ConstantAsMetadata *> {
  MDConstant() : ImplTy(nullptr) {}
};
struct MDStringField : public MDFieldImpl<MDString *> {
  bool AllowEmpty;
  MDStringField(bool AllowEmpty = true)
      : ImplTy(nullptr), AllowEmpty(AllowEmpty) {}
};
struct MDFieldList : public MDFieldImpl<SmallVector<Metadata *, 4>> {
  MDFieldList() : ImplTy(SmallVector<Metadata *, 4>()) {}
};

} // end namespace

namespace llvm {

template <>
bool LLParser::ParseMDField(LocTy Loc, StringRef Name,
                            MDUnsignedField &Result) {
  if (Lex.getKind() != lltok::APSInt || Lex.getAPSIntVal().isSigned())
    return TokError("expected unsigned integer");

  auto &U = Lex.getAPSIntVal();
  if (U.ugt(Result.Max))
    return TokError("value for '" + Name + "' too large, limit is " +
                    Twine(Result.Max));
  Result.assign(U.getZExtValue());
  assert(Result.Val <= Result.Max && "Expected value in range");
  Lex.Lex();
  return false;
}

template <>
bool LLParser::ParseMDField(LocTy Loc, StringRef Name, LineField &Result) {
  return ParseMDField(Loc, Name, static_cast<MDUnsignedField &>(Result));
}
template <>
bool LLParser::ParseMDField(LocTy Loc, StringRef Name, ColumnField &Result) {
  return ParseMDField(Loc, Name, static_cast<MDUnsignedField &>(Result));
}

template <>
bool LLParser::ParseMDField(LocTy Loc, StringRef Name, DwarfTagField &Result) {
  if (Lex.getKind() == lltok::APSInt)
    return ParseMDField(Loc, Name, static_cast<MDUnsignedField &>(Result));

  if (Lex.getKind() != lltok::DwarfTag)
    return TokError("expected DWARF tag");

  unsigned Tag = dwarf::getTag(Lex.getStrVal());
  if (Tag == dwarf::DW_TAG_invalid)
    return TokError("invalid DWARF tag" + Twine(" '") + Lex.getStrVal() + "'");
  assert(Tag <= Result.Max && "Expected valid DWARF tag");

  Result.assign(Tag);
  Lex.Lex();
  return false;
}

template <>
bool LLParser::ParseMDField(LocTy Loc, StringRef Name,
                            DwarfVirtualityField &Result) {
  if (Lex.getKind() == lltok::APSInt)
    return ParseMDField(Loc, Name, static_cast<MDUnsignedField &>(Result));

  if (Lex.getKind() != lltok::DwarfVirtuality)
    return TokError("expected DWARF virtuality code");

  unsigned Virtuality = dwarf::getVirtuality(Lex.getStrVal());
  if (!Virtuality)
    return TokError("invalid DWARF virtuality code" + Twine(" '") +
                    Lex.getStrVal() + "'");
  assert(Virtuality <= Result.Max && "Expected valid DWARF virtuality code");
  Result.assign(Virtuality);
  Lex.Lex();
  return false;
}

template <>
bool LLParser::ParseMDField(LocTy Loc, StringRef Name, DwarfLangField &Result) {
  if (Lex.getKind() == lltok::APSInt)
    return ParseMDField(Loc, Name, static_cast<MDUnsignedField &>(Result));

  if (Lex.getKind() != lltok::DwarfLang)
    return TokError("expected DWARF language");

  unsigned Lang = dwarf::getLanguage(Lex.getStrVal());
  if (!Lang)
    return TokError("invalid DWARF language" + Twine(" '") + Lex.getStrVal() +
                    "'");
  assert(Lang <= Result.Max && "Expected valid DWARF language");
  Result.assign(Lang);
  Lex.Lex();
  return false;
}

template <>
bool LLParser::ParseMDField(LocTy Loc, StringRef Name,
                            DwarfAttEncodingField &Result) {
  if (Lex.getKind() == lltok::APSInt)
    return ParseMDField(Loc, Name, static_cast<MDUnsignedField &>(Result));

  if (Lex.getKind() != lltok::DwarfAttEncoding)
    return TokError("expected DWARF type attribute encoding");

  unsigned Encoding = dwarf::getAttributeEncoding(Lex.getStrVal());
  if (!Encoding)
    return TokError("invalid DWARF type attribute encoding" + Twine(" '") +
                    Lex.getStrVal() + "'");
  assert(Encoding <= Result.Max && "Expected valid DWARF language");
  Result.assign(Encoding);
  Lex.Lex();
  return false;
}

/// DIFlagField
///  ::= uint32
///  ::= DIFlagVector
///  ::= DIFlagVector '|' DIFlagFwdDecl '|' uint32 '|' DIFlagPublic
template <>
bool LLParser::ParseMDField(LocTy Loc, StringRef Name, DIFlagField &Result) {
  assert(Result.Max == UINT32_MAX && "Expected only 32-bits");

  // Parser for a single flag.
  auto parseFlag = [&](unsigned &Val) {
    if (Lex.getKind() == lltok::APSInt && !Lex.getAPSIntVal().isSigned())
      return ParseUInt32(Val);

    if (Lex.getKind() != lltok::DIFlag)
      return TokError("expected debug info flag");

    Val = DINode::getFlag(Lex.getStrVal());
    if (!Val)
      return TokError(Twine("invalid debug info flag flag '") +
                      Lex.getStrVal() + "'");
    Lex.Lex();
    return false;
  };

  // Parse the flags and combine them together.
  unsigned Combined = 0;
  do {
    unsigned Val;
    if (parseFlag(Val))
      return true;
    Combined |= Val;
  } while (EatIfPresent(lltok::bar));

  Result.assign(Combined);
  return false;
}

template <>
bool LLParser::ParseMDField(LocTy Loc, StringRef Name,
                            MDSignedField &Result) {
  if (Lex.getKind() != lltok::APSInt)
    return TokError("expected signed integer");

  auto &S = Lex.getAPSIntVal();
  if (S < Result.Min)
    return TokError("value for '" + Name + "' too small, limit is " +
                    Twine(Result.Min));
  if (S > Result.Max)
    return TokError("value for '" + Name + "' too large, limit is " +
                    Twine(Result.Max));
  Result.assign(S.getExtValue());
  assert(Result.Val >= Result.Min && "Expected value in range");
  assert(Result.Val <= Result.Max && "Expected value in range");
  Lex.Lex();
  return false;
}

template <>
bool LLParser::ParseMDField(LocTy Loc, StringRef Name, MDBoolField &Result) {
  switch (Lex.getKind()) {
  default:
    return TokError("expected 'true' or 'false'");
  case lltok::kw_true:
    Result.assign(true);
    break;
  case lltok::kw_false:
    Result.assign(false);
    break;
  }
  Lex.Lex();
  return false;
}

template <>
bool LLParser::ParseMDField(LocTy Loc, StringRef Name, MDField &Result) {
  if (Lex.getKind() == lltok::kw_null) {
    if (!Result.AllowNull)
      return TokError("'" + Name + "' cannot be null");
    Lex.Lex();
    Result.assign(nullptr);
    return false;
  }

  Metadata *MD;
  if (ParseMetadata(MD, nullptr))
    return true;

  Result.assign(MD);
  return false;
}

template <>
bool LLParser::ParseMDField(LocTy Loc, StringRef Name, MDConstant &Result) {
  Metadata *MD;
  if (ParseValueAsMetadata(MD, "expected constant", nullptr))
    return true;

  Result.assign(cast<ConstantAsMetadata>(MD));
  return false;
}

template <>
bool LLParser::ParseMDField(LocTy Loc, StringRef Name, MDStringField &Result) {
  LocTy ValueLoc = Lex.getLoc();
  std::string S;
  if (ParseStringConstant(S))
    return true;

  if (!Result.AllowEmpty && S.empty())
    return Error(ValueLoc, "'" + Name + "' cannot be empty");

  Result.assign(S.empty() ? nullptr : MDString::get(Context, S));
  return false;
}

template <>
bool LLParser::ParseMDField(LocTy Loc, StringRef Name, MDFieldList &Result) {
  SmallVector<Metadata *, 4> MDs;
  if (ParseMDNodeVector(MDs))
    return true;

  Result.assign(std::move(MDs));
  return false;
}

} // end namespace llvm

template <class ParserTy>
bool LLParser::ParseMDFieldsImplBody(ParserTy parseField) {
  do {
    if (Lex.getKind() != lltok::LabelStr)
      return TokError("expected field label here");

    if (parseField())
      return true;
  } while (EatIfPresent(lltok::comma));

  return false;
}

template <class ParserTy>
bool LLParser::ParseMDFieldsImpl(ParserTy parseField, LocTy &ClosingLoc) {
  assert(Lex.getKind() == lltok::MetadataVar && "Expected metadata type name");
  Lex.Lex();

  if (ParseToken(lltok::lparen, "expected '(' here"))
    return true;
  if (Lex.getKind() != lltok::rparen)
    if (ParseMDFieldsImplBody(parseField))
      return true;

  ClosingLoc = Lex.getLoc();
  return ParseToken(lltok::rparen, "expected ')' here");
}

template <class FieldTy>
bool LLParser::ParseMDField(StringRef Name, FieldTy &Result) {
  if (Result.Seen)
    return TokError("field '" + Name + "' cannot be specified more than once");

  LocTy Loc = Lex.getLoc();
  Lex.Lex();
  return ParseMDField(Loc, Name, Result);
}

bool LLParser::ParseSpecializedMDNode(MDNode *&N, bool IsDistinct) {
  assert(Lex.getKind() == lltok::MetadataVar && "Expected metadata type name");

#define HANDLE_SPECIALIZED_MDNODE_LEAF(CLASS)                                  \
  if (Lex.getStrVal() == #CLASS)                                               \
    return Parse##CLASS(N, IsDistinct);
#include "llvm/IR/Metadata.def"

  return TokError("expected metadata type");
}

#define DECLARE_FIELD(NAME, TYPE, INIT) TYPE NAME INIT
#define NOP_FIELD(NAME, TYPE, INIT)
#define REQUIRE_FIELD(NAME, TYPE, INIT)                                        \
  if (!NAME.Seen)                                                              \
    return Error(ClosingLoc, "missing required field '" #NAME "'");
#define PARSE_MD_FIELD(NAME, TYPE, DEFAULT)                                    \
  if (Lex.getStrVal() == #NAME)                                                \
    return ParseMDField(#NAME, NAME);
#define PARSE_MD_FIELDS()                                                      \
  VISIT_MD_FIELDS(DECLARE_FIELD, DECLARE_FIELD)                                \
  do {                                                                         \
    LocTy ClosingLoc;                                                          \
    if (ParseMDFieldsImpl([&]() -> bool {                                      \
      VISIT_MD_FIELDS(PARSE_MD_FIELD, PARSE_MD_FIELD)                          \
      return TokError(Twine("invalid field '") + Lex.getStrVal() + "'");       \
    }, ClosingLoc))                                                            \
      return true;                                                             \
    VISIT_MD_FIELDS(NOP_FIELD, REQUIRE_FIELD)                                  \
  } while (false)
#define GET_OR_DISTINCT(CLASS, ARGS)                                           \
  (IsDistinct ? CLASS::getDistinct ARGS : CLASS::get ARGS)

/// ParseDILocationFields:
///   ::= !DILocation(line: 43, column: 8, scope: !5, inlinedAt: !6)
bool LLParser::ParseDILocation(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  OPTIONAL(line, LineField, );                                                 \
  OPTIONAL(column, ColumnField, );                                             \
  REQUIRED(scope, MDField, (/* AllowNull */ false));                           \
  OPTIONAL(inlinedAt, MDField, );
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result = GET_OR_DISTINCT(
      DILocation, (Context, line.Val, column.Val, scope.Val, inlinedAt.Val));
  return false;
}

/// ParseGenericDINode:
///   ::= !GenericDINode(tag: 15, header: "...", operands: {...})
bool LLParser::ParseGenericDINode(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  REQUIRED(tag, DwarfTagField, );                                              \
  OPTIONAL(header, MDStringField, );                                           \
  OPTIONAL(operands, MDFieldList, );
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result = GET_OR_DISTINCT(GenericDINode,
                           (Context, tag.Val, header.Val, operands.Val));
  return false;
}

/// ParseDISubrange:
///   ::= !DISubrange(count: 30, lowerBound: 2)
bool LLParser::ParseDISubrange(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  REQUIRED(count, MDSignedField, (-1, -1, INT64_MAX));                         \
  OPTIONAL(lowerBound, MDSignedField, );
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result = GET_OR_DISTINCT(DISubrange, (Context, count.Val, lowerBound.Val));
  return false;
}

/// ParseDIEnumerator:
///   ::= !DIEnumerator(value: 30, name: "SomeKind")
bool LLParser::ParseDIEnumerator(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  REQUIRED(name, MDStringField, );                                             \
  REQUIRED(value, MDSignedField, );
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result = GET_OR_DISTINCT(DIEnumerator, (Context, value.Val, name.Val));
  return false;
}

/// ParseDIBasicType:
///   ::= !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32)
bool LLParser::ParseDIBasicType(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  OPTIONAL(tag, DwarfTagField, (dwarf::DW_TAG_base_type));                     \
  OPTIONAL(name, MDStringField, );                                             \
  OPTIONAL(size, MDUnsignedField, (0, UINT64_MAX));                            \
  OPTIONAL(align, MDUnsignedField, (0, UINT64_MAX));                           \
  OPTIONAL(encoding, DwarfAttEncodingField, );
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result = GET_OR_DISTINCT(DIBasicType, (Context, tag.Val, name.Val, size.Val,
                                         align.Val, encoding.Val));
  return false;
}

/// ParseDIDerivedType:
///   ::= !DIDerivedType(tag: DW_TAG_pointer_type, name: "int", file: !0,
///                      line: 7, scope: !1, baseType: !2, size: 32,
///                      align: 32, offset: 0, flags: 0, extraData: !3)
bool LLParser::ParseDIDerivedType(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  REQUIRED(tag, DwarfTagField, );                                              \
  OPTIONAL(name, MDStringField, );                                             \
  OPTIONAL(file, MDField, );                                                   \
  OPTIONAL(line, LineField, );                                                 \
  OPTIONAL(scope, MDField, );                                                  \
  REQUIRED(baseType, MDField, );                                               \
  OPTIONAL(size, MDUnsignedField, (0, UINT64_MAX));                            \
  OPTIONAL(align, MDUnsignedField, (0, UINT64_MAX));                           \
  OPTIONAL(offset, MDUnsignedField, (0, UINT64_MAX));                          \
  OPTIONAL(flags, DIFlagField, );                                              \
  OPTIONAL(extraData, MDField, );
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result = GET_OR_DISTINCT(DIDerivedType,
                           (Context, tag.Val, name.Val, file.Val, line.Val,
                            scope.Val, baseType.Val, size.Val, align.Val,
                            offset.Val, flags.Val, extraData.Val));
  return false;
}

bool LLParser::ParseDICompositeType(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  REQUIRED(tag, DwarfTagField, );                                              \
  OPTIONAL(name, MDStringField, );                                             \
  OPTIONAL(file, MDField, );                                                   \
  OPTIONAL(line, LineField, );                                                 \
  OPTIONAL(scope, MDField, );                                                  \
  OPTIONAL(baseType, MDField, );                                               \
  OPTIONAL(size, MDUnsignedField, (0, UINT64_MAX));                            \
  OPTIONAL(align, MDUnsignedField, (0, UINT64_MAX));                           \
  OPTIONAL(offset, MDUnsignedField, (0, UINT64_MAX));                          \
  OPTIONAL(flags, DIFlagField, );                                              \
  OPTIONAL(elements, MDField, );                                               \
  OPTIONAL(runtimeLang, DwarfLangField, );                                     \
  OPTIONAL(vtableHolder, MDField, );                                           \
  OPTIONAL(templateParams, MDField, );                                         \
  OPTIONAL(identifier, MDStringField, );
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result = GET_OR_DISTINCT(
      DICompositeType,
      (Context, tag.Val, name.Val, file.Val, line.Val, scope.Val, baseType.Val,
       size.Val, align.Val, offset.Val, flags.Val, elements.Val,
       runtimeLang.Val, vtableHolder.Val, templateParams.Val, identifier.Val));
  return false;
}

bool LLParser::ParseDISubroutineType(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  OPTIONAL(flags, DIFlagField, );                                              \
  REQUIRED(types, MDField, );
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result = GET_OR_DISTINCT(DISubroutineType, (Context, flags.Val, types.Val));
  return false;
}

/// ParseDIFileType:
///   ::= !DIFileType(filename: "path/to/file", directory: "/path/to/dir")
bool LLParser::ParseDIFile(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  REQUIRED(filename, MDStringField, );                                         \
  REQUIRED(directory, MDStringField, );
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result = GET_OR_DISTINCT(DIFile, (Context, filename.Val, directory.Val));
  return false;
}

/// ParseDICompileUnit:
///   ::= !DICompileUnit(language: DW_LANG_C99, file: !0, producer: "clang",
///                      isOptimized: true, flags: "-O2", runtimeVersion: 1,
///                      splitDebugFilename: "abc.debug", emissionKind: 1,
///                      enums: !1, retainedTypes: !2, subprograms: !3,
///                      globals: !4, imports: !5, dwoId: 0x0abcd)
bool LLParser::ParseDICompileUnit(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  REQUIRED(language, DwarfLangField, );                                        \
  REQUIRED(file, MDField, (/* AllowNull */ false));                            \
  OPTIONAL(producer, MDStringField, );                                         \
  OPTIONAL(isOptimized, MDBoolField, );                                        \
  OPTIONAL(flags, MDStringField, );                                            \
  OPTIONAL(runtimeVersion, MDUnsignedField, (0, UINT32_MAX));                  \
  OPTIONAL(splitDebugFilename, MDStringField, );                               \
  OPTIONAL(emissionKind, MDUnsignedField, (0, UINT32_MAX));                    \
  OPTIONAL(enums, MDField, );                                                  \
  OPTIONAL(retainedTypes, MDField, );                                          \
  OPTIONAL(subprograms, MDField, );                                            \
  OPTIONAL(globals, MDField, );                                                \
  OPTIONAL(imports, MDField, );                                                \
  OPTIONAL(dwoId, MDUnsignedField, );
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result = GET_OR_DISTINCT(DICompileUnit,
                           (Context, language.Val, file.Val, producer.Val,
                            isOptimized.Val, flags.Val, runtimeVersion.Val,
                            splitDebugFilename.Val, emissionKind.Val, enums.Val,
                            retainedTypes.Val, subprograms.Val, globals.Val,
                            imports.Val, dwoId.Val));
  return false;
}

/// ParseDISubprogram:
///   ::= !DISubprogram(scope: !0, name: "foo", linkageName: "_Zfoo",
///                     file: !1, line: 7, type: !2, isLocal: false,
///                     isDefinition: true, scopeLine: 8, containingType: !3,
///                     virtuality: DW_VIRTUALTIY_pure_virtual,
///                     virtualIndex: 10, flags: 11,
///                     isOptimized: false, function: void ()* @_Z3foov,
///                     templateParams: !4, declaration: !5, variables: !6)
bool LLParser::ParseDISubprogram(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  OPTIONAL(scope, MDField, );                                                  \
  OPTIONAL(name, MDStringField, );                                             \
  OPTIONAL(linkageName, MDStringField, );                                      \
  OPTIONAL(file, MDField, );                                                   \
  OPTIONAL(line, LineField, );                                                 \
  OPTIONAL(type, MDField, );                                                   \
  OPTIONAL(isLocal, MDBoolField, );                                            \
  OPTIONAL(isDefinition, MDBoolField, (true));                                 \
  OPTIONAL(scopeLine, LineField, );                                            \
  OPTIONAL(containingType, MDField, );                                         \
  OPTIONAL(virtuality, DwarfVirtualityField, );                                \
  OPTIONAL(virtualIndex, MDUnsignedField, (0, UINT32_MAX));                    \
  OPTIONAL(flags, DIFlagField, );                                              \
  OPTIONAL(isOptimized, MDBoolField, );                                        \
  OPTIONAL(function, MDConstant, );                                            \
  OPTIONAL(templateParams, MDField, );                                         \
  OPTIONAL(declaration, MDField, );                                            \
  OPTIONAL(variables, MDField, );
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result = GET_OR_DISTINCT(
      DISubprogram, (Context, scope.Val, name.Val, linkageName.Val, file.Val,
                     line.Val, type.Val, isLocal.Val, isDefinition.Val,
                     scopeLine.Val, containingType.Val, virtuality.Val,
                     virtualIndex.Val, flags.Val, isOptimized.Val, function.Val,
                     templateParams.Val, declaration.Val, variables.Val));
  return false;
}

/// ParseDILexicalBlock:
///   ::= !DILexicalBlock(scope: !0, file: !2, line: 7, column: 9)
bool LLParser::ParseDILexicalBlock(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  REQUIRED(scope, MDField, (/* AllowNull */ false));                           \
  OPTIONAL(file, MDField, );                                                   \
  OPTIONAL(line, LineField, );                                                 \
  OPTIONAL(column, ColumnField, );
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result = GET_OR_DISTINCT(
      DILexicalBlock, (Context, scope.Val, file.Val, line.Val, column.Val));
  return false;
}

/// ParseDILexicalBlockFile:
///   ::= !DILexicalBlockFile(scope: !0, file: !2, discriminator: 9)
bool LLParser::ParseDILexicalBlockFile(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  REQUIRED(scope, MDField, (/* AllowNull */ false));                           \
  OPTIONAL(file, MDField, );                                                   \
  REQUIRED(discriminator, MDUnsignedField, (0, UINT32_MAX));
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result = GET_OR_DISTINCT(DILexicalBlockFile,
                           (Context, scope.Val, file.Val, discriminator.Val));
  return false;
}

/// ParseDINamespace:
///   ::= !DINamespace(scope: !0, file: !2, name: "SomeNamespace", line: 9)
bool LLParser::ParseDINamespace(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  REQUIRED(scope, MDField, );                                                  \
  OPTIONAL(file, MDField, );                                                   \
  OPTIONAL(name, MDStringField, );                                             \
  OPTIONAL(line, LineField, );
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result = GET_OR_DISTINCT(DINamespace,
                           (Context, scope.Val, file.Val, name.Val, line.Val));
  return false;
}

/// ParseDIModule:
///   ::= !DIModule(scope: !0, name: "SomeModule", configMacros: "-DNDEBUG",
///                 includePath: "/usr/include", isysroot: "/")
bool LLParser::ParseDIModule(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  REQUIRED(scope, MDField, );                                                  \
  REQUIRED(name, MDStringField, );                                             \
  OPTIONAL(configMacros, MDStringField, );                                     \
  OPTIONAL(includePath, MDStringField, );                                      \
  OPTIONAL(isysroot, MDStringField, );
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result = GET_OR_DISTINCT(DIModule, (Context, scope.Val, name.Val,
                           configMacros.Val, includePath.Val, isysroot.Val));
  return false;
}

/// ParseDITemplateTypeParameter:
///   ::= !DITemplateTypeParameter(name: "Ty", type: !1)
bool LLParser::ParseDITemplateTypeParameter(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  OPTIONAL(name, MDStringField, );                                             \
  REQUIRED(type, MDField, );
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result =
      GET_OR_DISTINCT(DITemplateTypeParameter, (Context, name.Val, type.Val));
  return false;
}

/// ParseDITemplateValueParameter:
///   ::= !DITemplateValueParameter(tag: DW_TAG_template_value_parameter,
///                                 name: "V", type: !1, value: i32 7)
bool LLParser::ParseDITemplateValueParameter(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  OPTIONAL(tag, DwarfTagField, (dwarf::DW_TAG_template_value_parameter));      \
  OPTIONAL(name, MDStringField, );                                             \
  OPTIONAL(type, MDField, );                                                   \
  REQUIRED(value, MDField, );
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result = GET_OR_DISTINCT(DITemplateValueParameter,
                           (Context, tag.Val, name.Val, type.Val, value.Val));
  return false;
}

/// ParseDIGlobalVariable:
///   ::= !DIGlobalVariable(scope: !0, name: "foo", linkageName: "foo",
///                         file: !1, line: 7, type: !2, isLocal: false,
///                         isDefinition: true, variable: i32* @foo,
///                         declaration: !3)
bool LLParser::ParseDIGlobalVariable(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  REQUIRED(name, MDStringField, (/* AllowEmpty */ false));                     \
  OPTIONAL(scope, MDField, );                                                  \
  OPTIONAL(linkageName, MDStringField, );                                      \
  OPTIONAL(file, MDField, );                                                   \
  OPTIONAL(line, LineField, );                                                 \
  OPTIONAL(type, MDField, );                                                   \
  OPTIONAL(isLocal, MDBoolField, );                                            \
  OPTIONAL(isDefinition, MDBoolField, (true));                                 \
  OPTIONAL(variable, MDConstant, );                                            \
  OPTIONAL(declaration, MDField, );
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result = GET_OR_DISTINCT(DIGlobalVariable,
                           (Context, scope.Val, name.Val, linkageName.Val,
                            file.Val, line.Val, type.Val, isLocal.Val,
                            isDefinition.Val, variable.Val, declaration.Val));
  return false;
}

/// ParseDILocalVariable:
///   ::= !DILocalVariable(tag: DW_TAG_arg_variable, scope: !0, name: "foo",
///                        file: !1, line: 7, type: !2, arg: 2, flags: 7)
bool LLParser::ParseDILocalVariable(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  REQUIRED(tag, DwarfTagField, );                                              \
  REQUIRED(scope, MDField, (/* AllowNull */ false));                           \
  OPTIONAL(name, MDStringField, );                                             \
  OPTIONAL(file, MDField, );                                                   \
  OPTIONAL(line, LineField, );                                                 \
  OPTIONAL(type, MDField, );                                                   \
  OPTIONAL(arg, MDUnsignedField, (0, UINT16_MAX));                             \
  OPTIONAL(flags, DIFlagField, );
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result = GET_OR_DISTINCT(DILocalVariable,
                           (Context, tag.Val, scope.Val, name.Val, file.Val,
                            line.Val, type.Val, arg.Val, flags.Val));
  return false;
}

/// ParseDIExpression:
///   ::= !DIExpression(0, 7, -1)
bool LLParser::ParseDIExpression(MDNode *&Result, bool IsDistinct) {
  assert(Lex.getKind() == lltok::MetadataVar && "Expected metadata type name");
  Lex.Lex();

  if (ParseToken(lltok::lparen, "expected '(' here"))
    return true;

  SmallVector<uint64_t, 8> Elements;
  if (Lex.getKind() != lltok::rparen)
    do {
      if (Lex.getKind() == lltok::DwarfOp) {
        if (unsigned Op = dwarf::getOperationEncoding(Lex.getStrVal())) {
          Lex.Lex();
          Elements.push_back(Op);
          continue;
        }
        return TokError(Twine("invalid DWARF op '") + Lex.getStrVal() + "'");
      }

      if (Lex.getKind() != lltok::APSInt || Lex.getAPSIntVal().isSigned())
        return TokError("expected unsigned integer");

      auto &U = Lex.getAPSIntVal();
      if (U.ugt(UINT64_MAX))
        return TokError("element too large, limit is " + Twine(UINT64_MAX));
      Elements.push_back(U.getZExtValue());
      Lex.Lex();
    } while (EatIfPresent(lltok::comma));

  if (ParseToken(lltok::rparen, "expected ')' here"))
    return true;

  Result = GET_OR_DISTINCT(DIExpression, (Context, Elements));
  return false;
}

/// ParseDIObjCProperty:
///   ::= !DIObjCProperty(name: "foo", file: !1, line: 7, setter: "setFoo",
///                       getter: "getFoo", attributes: 7, type: !2)
bool LLParser::ParseDIObjCProperty(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  OPTIONAL(name, MDStringField, );                                             \
  OPTIONAL(file, MDField, );                                                   \
  OPTIONAL(line, LineField, );                                                 \
  OPTIONAL(setter, MDStringField, );                                           \
  OPTIONAL(getter, MDStringField, );                                           \
  OPTIONAL(attributes, MDUnsignedField, (0, UINT32_MAX));                      \
  OPTIONAL(type, MDField, );
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result = GET_OR_DISTINCT(DIObjCProperty,
                           (Context, name.Val, file.Val, line.Val, setter.Val,
                            getter.Val, attributes.Val, type.Val));
  return false;
}

/// ParseDIImportedEntity:
///   ::= !DIImportedEntity(tag: DW_TAG_imported_module, scope: !0, entity: !1,
///                         line: 7, name: "foo")
bool LLParser::ParseDIImportedEntity(MDNode *&Result, bool IsDistinct) {
#define VISIT_MD_FIELDS(OPTIONAL, REQUIRED)                                    \
  REQUIRED(tag, DwarfTagField, );                                              \
  REQUIRED(scope, MDField, );                                                  \
  OPTIONAL(entity, MDField, );                                                 \
  OPTIONAL(line, LineField, );                                                 \
  OPTIONAL(name, MDStringField, );
  PARSE_MD_FIELDS();
#undef VISIT_MD_FIELDS

  Result = GET_OR_DISTINCT(DIImportedEntity, (Context, tag.Val, scope.Val,
                                              entity.Val, line.Val, name.Val));
  return false;
}

#undef PARSE_MD_FIELD
#undef NOP_FIELD
#undef REQUIRE_FIELD
#undef DECLARE_FIELD

/// ParseMetadataAsValue
///  ::= metadata i32 %local
///  ::= metadata i32 @global
///  ::= metadata i32 7
///  ::= metadata !0
///  ::= metadata !{...}
///  ::= metadata !"string"
bool LLParser::ParseMetadataAsValue(Value *&V, PerFunctionState &PFS) {
  // Note: the type 'metadata' has already been parsed.
  Metadata *MD;
  if (ParseMetadata(MD, &PFS))
    return true;

  V = MetadataAsValue::get(Context, MD);
  return false;
}

/// ParseValueAsMetadata
///  ::= i32 %local
///  ::= i32 @global
///  ::= i32 7
bool LLParser::ParseValueAsMetadata(Metadata *&MD, const Twine &TypeMsg,
                                    PerFunctionState *PFS) {
  Type *Ty;
  LocTy Loc;
  if (ParseType(Ty, TypeMsg, Loc))
    return true;
  if (Ty->isMetadataTy())
    return Error(Loc, "invalid metadata-value-metadata roundtrip");

  Value *V;
  if (ParseValue(Ty, V, PFS))
    return true;

  MD = ValueAsMetadata::get(V);
  return false;
}

/// ParseMetadata
///  ::= i32 %local
///  ::= i32 @global
///  ::= i32 7
///  ::= !42
///  ::= !{...}
///  ::= !"string"
///  ::= !DILocation(...)
bool LLParser::ParseMetadata(Metadata *&MD, PerFunctionState *PFS) {
  if (Lex.getKind() == lltok::MetadataVar) {
    MDNode *N;
    if (ParseSpecializedMDNode(N))
      return true;
    MD = N;
    return false;
  }

  // ValueAsMetadata:
  // <type> <value>
  if (Lex.getKind() != lltok::exclaim)
    return ParseValueAsMetadata(MD, "expected metadata operand", PFS);

  // '!'.
  assert(Lex.getKind() == lltok::exclaim && "Expected '!' here");
  Lex.Lex();

  // MDString:
  //   ::= '!' STRINGCONSTANT
  if (Lex.getKind() == lltok::StringConstant) {
    MDString *S;
    if (ParseMDString(S))
      return true;
    MD = S;
    return false;
  }

  // MDNode:
  // !{ ... }
  // !7
  MDNode *N;
  if (ParseMDNodeTail(N))
    return true;
  MD = N;
  return false;
}


//===----------------------------------------------------------------------===//
// Function Parsing.
//===----------------------------------------------------------------------===//

bool LLParser::ConvertValIDToValue(Type *Ty, ValID &ID, Value *&V,
                                   PerFunctionState *PFS) {
  if (Ty->isFunctionTy())
    return Error(ID.Loc, "functions are not values, refer to them as pointers");

  switch (ID.Kind) {
  case ValID::t_LocalID:
    if (!PFS) return Error(ID.Loc, "invalid use of function-local name");
    V = PFS->GetVal(ID.UIntVal, Ty, ID.Loc);
    return V == nullptr;
  case ValID::t_LocalName:
    if (!PFS) return Error(ID.Loc, "invalid use of function-local name");
    V = PFS->GetVal(ID.StrVal, Ty, ID.Loc);
    return V == nullptr;
  case ValID::t_InlineAsm: {
    PointerType *PTy = dyn_cast<PointerType>(Ty);
    FunctionType *FTy =
      PTy ? dyn_cast<FunctionType>(PTy->getElementType()) : nullptr;
    if (!FTy || !InlineAsm::Verify(FTy, ID.StrVal2))
      return Error(ID.Loc, "invalid type for inline asm constraint string");
    V = InlineAsm::get(FTy, ID.StrVal, ID.StrVal2, ID.UIntVal&1,
                       (ID.UIntVal>>1)&1, (InlineAsm::AsmDialect(ID.UIntVal>>2)));
    return false;
  }
  case ValID::t_GlobalName:
    V = GetGlobalVal(ID.StrVal, Ty, ID.Loc);
    return V == nullptr;
  case ValID::t_GlobalID:
    V = GetGlobalVal(ID.UIntVal, Ty, ID.Loc);
    return V == nullptr;
  case ValID::t_APSInt:
    if (!Ty->isIntegerTy())
      return Error(ID.Loc, "integer constant must have integer type");
    ID.APSIntVal = ID.APSIntVal.extOrTrunc(Ty->getPrimitiveSizeInBits());
    V = ConstantInt::get(Context, ID.APSIntVal);
    return false;
  case ValID::t_APFloat:
    if (!Ty->isFloatingPointTy() ||
        !ConstantFP::isValueValidForType(Ty, ID.APFloatVal))
      return Error(ID.Loc, "floating point constant invalid for type");

    // The lexer has no type info, so builds all half, float, and double FP
    // constants as double.  Fix this here.  Long double does not need this.
    if (&ID.APFloatVal.getSemantics() == &APFloat::IEEEdouble) {
      bool Ignored;
      if (Ty->isHalfTy())
        ID.APFloatVal.convert(APFloat::IEEEhalf, APFloat::rmNearestTiesToEven,
                              &Ignored);
      else if (Ty->isFloatTy())
        ID.APFloatVal.convert(APFloat::IEEEsingle, APFloat::rmNearestTiesToEven,
                              &Ignored);
    }
    V = ConstantFP::get(Context, ID.APFloatVal);

    if (V->getType() != Ty)
      return Error(ID.Loc, "floating point constant does not have type '" +
                   getTypeString(Ty) + "'");

    return false;
  case ValID::t_Null:
    if (!Ty->isPointerTy())
      return Error(ID.Loc, "null must be a pointer type");
    V = ConstantPointerNull::get(cast<PointerType>(Ty));
    return false;
  case ValID::t_Undef:
    // FIXME: LabelTy should not be a first-class type.
    if (!Ty->isFirstClassType() || Ty->isLabelTy())
      return Error(ID.Loc, "invalid type for undef constant");
    V = UndefValue::get(Ty);
    return false;
  case ValID::t_EmptyArray:
    if (!Ty->isArrayTy() || cast<ArrayType>(Ty)->getNumElements() != 0)
      return Error(ID.Loc, "invalid empty array initializer");
    V = UndefValue::get(Ty);
    return false;
  case ValID::t_Zero:
    // FIXME: LabelTy should not be a first-class type.
    if (!Ty->isFirstClassType() || Ty->isLabelTy())
      return Error(ID.Loc, "invalid type for null constant");
    V = Constant::getNullValue(Ty);
    return false;
  case ValID::t_Constant:
    if (ID.ConstantVal->getType() != Ty)
      return Error(ID.Loc, "constant expression type mismatch");

    V = ID.ConstantVal;
    return false;
  case ValID::t_ConstantStruct:
  case ValID::t_PackedConstantStruct:
    if (StructType *ST = dyn_cast<StructType>(Ty)) {
      if (ST->getNumElements() != ID.UIntVal)
        return Error(ID.Loc,
                     "initializer with struct type has wrong # elements");
      if (ST->isPacked() != (ID.Kind == ValID::t_PackedConstantStruct))
        return Error(ID.Loc, "packed'ness of initializer and type don't match");

      // Verify that the elements are compatible with the structtype.
      for (unsigned i = 0, e = ID.UIntVal; i != e; ++i)
        if (ID.ConstantStructElts[i]->getType() != ST->getElementType(i))
          return Error(ID.Loc, "element " + Twine(i) +
                    " of struct initializer doesn't match struct element type");

      V = ConstantStruct::get(ST, makeArrayRef(ID.ConstantStructElts,
                                               ID.UIntVal));
    } else
      return Error(ID.Loc, "constant expression type mismatch");
    return false;
  }
  llvm_unreachable("Invalid ValID");
}

bool LLParser::ParseValue(Type *Ty, Value *&V, PerFunctionState *PFS) {
  V = nullptr;
  ValID ID;
  return ParseValID(ID, PFS) ||
         ConvertValIDToValue(Ty, ID, V, PFS);
}

bool LLParser::ParseTypeAndValue(Value *&V, PerFunctionState *PFS) {
  Type *Ty = nullptr;
  return ParseType(Ty) ||
         ParseValue(Ty, V, PFS);
}

bool LLParser::ParseTypeAndBasicBlock(BasicBlock *&BB, LocTy &Loc,
                                      PerFunctionState &PFS) {
  Value *V;
  Loc = Lex.getLoc();
  if (ParseTypeAndValue(V, PFS)) return true;
  if (!isa<BasicBlock>(V))
    return Error(Loc, "expected a basic block");
  BB = cast<BasicBlock>(V);
  return false;
}


/// FunctionHeader
///   ::= OptionalLinkage OptionalVisibility OptionalCallingConv OptRetAttrs
///       OptUnnamedAddr Type GlobalName '(' ArgList ')' OptFuncAttrs OptSection
///       OptionalAlign OptGC OptionalPrefix OptionalPrologue OptPersonalityFn
bool LLParser::ParseFunctionHeader(Function *&Fn, bool isDefine) {
  // Parse the linkage.
  LocTy LinkageLoc = Lex.getLoc();
  unsigned Linkage;

  unsigned Visibility;
  unsigned DLLStorageClass;
  AttrBuilder RetAttrs;
  unsigned CC;
  Type *RetType = nullptr;
  LocTy RetTypeLoc = Lex.getLoc();
  if (ParseOptionalLinkage(Linkage) ||
      ParseOptionalVisibility(Visibility) ||
      ParseOptionalDLLStorageClass(DLLStorageClass) ||
      ParseOptionalCallingConv(CC) ||
      ParseOptionalReturnAttrs(RetAttrs) ||
      ParseType(RetType, RetTypeLoc, true /*void allowed*/))
    return true;

  // Verify that the linkage is ok.
  switch ((GlobalValue::LinkageTypes)Linkage) {
  case GlobalValue::ExternalLinkage:
    break; // always ok.
  case GlobalValue::ExternalWeakLinkage:
    if (isDefine)
      return Error(LinkageLoc, "invalid linkage for function definition");
    break;
  case GlobalValue::PrivateLinkage:
  case GlobalValue::InternalLinkage:
  case GlobalValue::AvailableExternallyLinkage:
  case GlobalValue::LinkOnceAnyLinkage:
  case GlobalValue::LinkOnceODRLinkage:
  case GlobalValue::WeakAnyLinkage:
  case GlobalValue::WeakODRLinkage:
    if (!isDefine)
      return Error(LinkageLoc, "invalid linkage for function declaration");
    break;
  case GlobalValue::AppendingLinkage:
  case GlobalValue::CommonLinkage:
    return Error(LinkageLoc, "invalid function linkage type");
  }

  if (!isValidVisibilityForLinkage(Visibility, Linkage))
    return Error(LinkageLoc,
                 "symbol with local linkage must have default visibility");

  if (!FunctionType::isValidReturnType(RetType))
    return Error(RetTypeLoc, "invalid function return type");

  LocTy NameLoc = Lex.getLoc();

  std::string FunctionName;
  if (Lex.getKind() == lltok::GlobalVar) {
    FunctionName = Lex.getStrVal();
  } else if (Lex.getKind() == lltok::GlobalID) {     // @42 is ok.
    unsigned NameID = Lex.getUIntVal();

    if (NameID != NumberedVals.size())
      return TokError("function expected to be numbered '%" +
                      Twine(NumberedVals.size()) + "'");
  } else {
    return TokError("expected function name");
  }

  Lex.Lex();

  if (Lex.getKind() != lltok::lparen)
    return TokError("expected '(' in function argument list");

  SmallVector<ArgInfo, 8> ArgList;
  bool isVarArg;
  AttrBuilder FuncAttrs;
  std::vector<unsigned> FwdRefAttrGrps;
  LocTy BuiltinLoc;
  std::string Section;
  unsigned Alignment;
  std::string GC;
  bool UnnamedAddr;
  LocTy UnnamedAddrLoc;
  Constant *Prefix = nullptr;
  Constant *Prologue = nullptr;
  Constant *PersonalityFn = nullptr;
  Comdat *C;

  if (ParseArgumentList(ArgList, isVarArg) ||
      ParseOptionalToken(lltok::kw_unnamed_addr, UnnamedAddr,
                         &UnnamedAddrLoc) ||
      ParseFnAttributeValuePairs(FuncAttrs, FwdRefAttrGrps, false,
                                 BuiltinLoc) ||
      (EatIfPresent(lltok::kw_section) &&
       ParseStringConstant(Section)) ||
      parseOptionalComdat(FunctionName, C) ||
      ParseOptionalAlignment(Alignment) ||
      (EatIfPresent(lltok::kw_gc) &&
       ParseStringConstant(GC)) ||
      (EatIfPresent(lltok::kw_prefix) &&
       ParseGlobalTypeAndValue(Prefix)) ||
      (EatIfPresent(lltok::kw_prologue) &&
       ParseGlobalTypeAndValue(Prologue)) ||
      (EatIfPresent(lltok::kw_personality) &&
       ParseGlobalTypeAndValue(PersonalityFn)))
    return true;

  if (FuncAttrs.contains(Attribute::Builtin))
    return Error(BuiltinLoc, "'builtin' attribute not valid on function");

  // If the alignment was parsed as an attribute, move to the alignment field.
  if (FuncAttrs.hasAlignmentAttr()) {
    Alignment = FuncAttrs.getAlignment();
    FuncAttrs.removeAttribute(Attribute::Alignment);
  }

  // Okay, if we got here, the function is syntactically valid.  Convert types
  // and do semantic checks.
  std::vector<Type*> ParamTypeList;
  SmallVector<AttributeSet, 8> Attrs;

  if (RetAttrs.hasAttributes())
    Attrs.push_back(AttributeSet::get(RetType->getContext(),
                                      AttributeSet::ReturnIndex,
                                      RetAttrs));

  for (unsigned i = 0, e = ArgList.size(); i != e; ++i) {
    ParamTypeList.push_back(ArgList[i].Ty);
    if (ArgList[i].Attrs.hasAttributes(i + 1)) {
      AttrBuilder B(ArgList[i].Attrs, i + 1);
      Attrs.push_back(AttributeSet::get(RetType->getContext(), i + 1, B));
    }
  }

  if (FuncAttrs.hasAttributes())
    Attrs.push_back(AttributeSet::get(RetType->getContext(),
                                      AttributeSet::FunctionIndex,
                                      FuncAttrs));

  AttributeSet PAL = AttributeSet::get(Context, Attrs);

  if (PAL.hasAttribute(1, Attribute::StructRet) && !RetType->isVoidTy())
    return Error(RetTypeLoc, "functions with 'sret' argument must return void");

  FunctionType *FT =
    FunctionType::get(RetType, ParamTypeList, isVarArg);
  PointerType *PFT = PointerType::getUnqual(FT);

  Fn = nullptr;
  if (!FunctionName.empty()) {
    // If this was a definition of a forward reference, remove the definition
    // from the forward reference table and fill in the forward ref.
    std::map<std::string, std::pair<GlobalValue*, LocTy> >::iterator FRVI =
      ForwardRefVals.find(FunctionName);
    if (FRVI != ForwardRefVals.end()) {
      Fn = M->getFunction(FunctionName);
      if (!Fn)
        return Error(FRVI->second.second, "invalid forward reference to "
                     "function as global value!");
      if (Fn->getType() != PFT)
        return Error(FRVI->second.second, "invalid forward reference to "
                     "function '" + FunctionName + "' with wrong type!");

      ForwardRefVals.erase(FRVI);
    } else if ((Fn = M->getFunction(FunctionName))) {
      // Reject redefinitions.
      return Error(NameLoc, "invalid redefinition of function '" +
                   FunctionName + "'");
    } else if (M->getNamedValue(FunctionName)) {
      return Error(NameLoc, "redefinition of function '@" + FunctionName + "'");
    }

  } else {
    // If this is a definition of a forward referenced function, make sure the
    // types agree.
    std::map<unsigned, std::pair<GlobalValue*, LocTy> >::iterator I
      = ForwardRefValIDs.find(NumberedVals.size());
    if (I != ForwardRefValIDs.end()) {
      Fn = cast<Function>(I->second.first);
      if (Fn->getType() != PFT)
        return Error(NameLoc, "type of definition and forward reference of '@" +
                     Twine(NumberedVals.size()) + "' disagree");
      ForwardRefValIDs.erase(I);
    }
  }

  if (!Fn)
    Fn = Function::Create(FT, GlobalValue::ExternalLinkage, FunctionName, M);
  else // Move the forward-reference to the correct spot in the module.
    M->getFunctionList().splice(M->end(), M->getFunctionList(), Fn);

  if (FunctionName.empty())
    NumberedVals.push_back(Fn);

  Fn->setLinkage((GlobalValue::LinkageTypes)Linkage);
  Fn->setVisibility((GlobalValue::VisibilityTypes)Visibility);
  Fn->setDLLStorageClass((GlobalValue::DLLStorageClassTypes)DLLStorageClass);
  Fn->setCallingConv(CC);
  Fn->setAttributes(PAL);
  Fn->setUnnamedAddr(UnnamedAddr);
  Fn->setAlignment(Alignment);
  Fn->setSection(Section);
  Fn->setComdat(C);
  Fn->setPersonalityFn(PersonalityFn);
  if (!GC.empty()) Fn->setGC(GC.c_str());
  Fn->setPrefixData(Prefix);
  Fn->setPrologueData(Prologue);
  ForwardRefAttrGroups[Fn] = FwdRefAttrGrps;

  // Add all of the arguments we parsed to the function.
  Function::arg_iterator ArgIt = Fn->arg_begin();
  for (unsigned i = 0, e = ArgList.size(); i != e; ++i, ++ArgIt) {
    // If the argument has a name, insert it into the argument symbol table.
    if (ArgList[i].Name.empty()) continue;

    // Set the name, if it conflicted, it will be auto-renamed.
    ArgIt->setName(ArgList[i].Name);

    if (ArgIt->getName() != ArgList[i].Name)
      return Error(ArgList[i].Loc, "redefinition of argument '%" +
                   ArgList[i].Name + "'");
  }

  if (isDefine)
    return false;

  // Check the declaration has no block address forward references.
  ValID ID;
  if (FunctionName.empty()) {
    ID.Kind = ValID::t_GlobalID;
    ID.UIntVal = NumberedVals.size() - 1;
  } else {
    ID.Kind = ValID::t_GlobalName;
    ID.StrVal = FunctionName;
  }
  auto Blocks = ForwardRefBlockAddresses.find(ID);
  if (Blocks != ForwardRefBlockAddresses.end())
    return Error(Blocks->first.Loc,
                 "cannot take blockaddress inside a declaration");
  return false;
}

bool LLParser::PerFunctionState::resolveForwardRefBlockAddresses() {
  ValID ID;
  if (FunctionNumber == -1) {
    ID.Kind = ValID::t_GlobalName;
    ID.StrVal = F.getName();
  } else {
    ID.Kind = ValID::t_GlobalID;
    ID.UIntVal = FunctionNumber;
  }

  auto Blocks = P.ForwardRefBlockAddresses.find(ID);
  if (Blocks == P.ForwardRefBlockAddresses.end())
    return false;

  for (const auto &I : Blocks->second) {
    const ValID &BBID = I.first;
    GlobalValue *GV = I.second;

    assert((BBID.Kind == ValID::t_LocalID || BBID.Kind == ValID::t_LocalName) &&
           "Expected local id or name");
    BasicBlock *BB;
    if (BBID.Kind == ValID::t_LocalName)
      BB = GetBB(BBID.StrVal, BBID.Loc);
    else
      BB = GetBB(BBID.UIntVal, BBID.Loc);
    if (!BB)
      return P.Error(BBID.Loc, "referenced value is not a basic block");

    GV->replaceAllUsesWith(BlockAddress::get(&F, BB));
    GV->eraseFromParent();
  }

  P.ForwardRefBlockAddresses.erase(Blocks);
  return false;
}

/// ParseFunctionBody
///   ::= '{' BasicBlock+ UseListOrderDirective* '}'
bool LLParser::ParseFunctionBody(Function &Fn) {
  if (Lex.getKind() != lltok::lbrace)
    return TokError("expected '{' in function body");
  Lex.Lex();  // eat the {.

  int FunctionNumber = -1;
  if (!Fn.hasName()) FunctionNumber = NumberedVals.size()-1;

  PerFunctionState PFS(*this, Fn, FunctionNumber);

  // Resolve block addresses and allow basic blocks to be forward-declared
  // within this function.
  if (PFS.resolveForwardRefBlockAddresses())
    return true;
  SaveAndRestore<PerFunctionState *> ScopeExit(BlockAddressPFS, &PFS);

  // We need at least one basic block.
  if (Lex.getKind() == lltok::rbrace || Lex.getKind() == lltok::kw_uselistorder)
    return TokError("function body requires at least one basic block");

  while (Lex.getKind() != lltok::rbrace &&
         Lex.getKind() != lltok::kw_uselistorder)
    if (ParseBasicBlock(PFS)) return true;

  while (Lex.getKind() != lltok::rbrace)
    if (ParseUseListOrder(&PFS))
      return true;

  // Eat the }.
  Lex.Lex();

  // Verify function is ok.
  return PFS.FinishFunction();
}

/// ParseBasicBlock
///   ::= LabelStr? Instruction*
bool LLParser::ParseBasicBlock(PerFunctionState &PFS) {
  // If this basic block starts out with a name, remember it.
  std::string Name;
  LocTy NameLoc = Lex.getLoc();
  if (Lex.getKind() == lltok::LabelStr) {
    Name = Lex.getStrVal();
    Lex.Lex();
  }

  BasicBlock *BB = PFS.DefineBB(Name, NameLoc);
  if (!BB)
    return Error(NameLoc,
                 "unable to create block named '" + Name + "'");

  std::string NameStr;

  // Parse the instructions in this block until we get a terminator.
  Instruction *Inst;
  do {
    // This instruction may have three possibilities for a name: a) none
    // specified, b) name specified "%foo =", c) number specified: "%4 =".
    LocTy NameLoc = Lex.getLoc();
    int NameID = -1;
    NameStr = "";

    if (Lex.getKind() == lltok::LocalVarID) {
      NameID = Lex.getUIntVal();
      Lex.Lex();
      if (ParseToken(lltok::equal, "expected '=' after instruction id"))
        return true;
    } else if (Lex.getKind() == lltok::LocalVar) {
      NameStr = Lex.getStrVal();
      Lex.Lex();
      if (ParseToken(lltok::equal, "expected '=' after instruction name"))
        return true;
    }

    switch (ParseInstruction(Inst, BB, PFS)) {
    default: llvm_unreachable("Unknown ParseInstruction result!");
    case InstError: return true;
    case InstNormal:
      BB->getInstList().push_back(Inst);

      // With a normal result, we check to see if the instruction is followed by
      // a comma and metadata.
      if (EatIfPresent(lltok::comma))
        if (ParseInstructionMetadata(*Inst))
          return true;
      break;
    case InstExtraComma:
      BB->getInstList().push_back(Inst);

      // If the instruction parser ate an extra comma at the end of it, it
      // *must* be followed by metadata.
      if (ParseInstructionMetadata(*Inst))
        return true;
      break;
    }

    // Set the name on the instruction.
    if (PFS.SetInstName(NameID, NameStr, NameLoc, Inst)) return true;
  } while (!isa<TerminatorInst>(Inst));

  return false;
}

//===----------------------------------------------------------------------===//
// Instruction Parsing.
//===----------------------------------------------------------------------===//

/// ParseInstruction - Parse one of the many different instructions.
///
int LLParser::ParseInstruction(Instruction *&Inst, BasicBlock *BB,
                               PerFunctionState &PFS) {
  lltok::Kind Token = Lex.getKind();
  if (Token == lltok::Eof)
    return TokError("found end of file when expecting more instructions");
  LocTy Loc = Lex.getLoc();
  unsigned KeywordVal = Lex.getUIntVal();
  Lex.Lex();  // Eat the keyword.

  switch (Token) {
  default:                    return Error(Loc, "expected instruction opcode");
  // Terminator Instructions.
  case lltok::kw_unreachable: Inst = new UnreachableInst(Context); return false;
  case lltok::kw_ret:         return ParseRet(Inst, BB, PFS);
  case lltok::kw_br:          return ParseBr(Inst, PFS);
  case lltok::kw_switch:      return ParseSwitch(Inst, PFS);
  case lltok::kw_indirectbr:  return ParseIndirectBr(Inst, PFS);
  case lltok::kw_invoke:      return ParseInvoke(Inst, PFS);
  case lltok::kw_resume:      return ParseResume(Inst, PFS);
  // Binary Operators.
  case lltok::kw_add:
  case lltok::kw_sub:
  case lltok::kw_mul:
  case lltok::kw_shl: {
    bool NUW = EatIfPresent(lltok::kw_nuw);
    bool NSW = EatIfPresent(lltok::kw_nsw);
    if (!NUW) NUW = EatIfPresent(lltok::kw_nuw);

    if (ParseArithmetic(Inst, PFS, KeywordVal, 1)) return true;

    if (NUW) cast<BinaryOperator>(Inst)->setHasNoUnsignedWrap(true);
    if (NSW) cast<BinaryOperator>(Inst)->setHasNoSignedWrap(true);
    return false;
  }
  case lltok::kw_fadd:
  case lltok::kw_fsub:
  case lltok::kw_fmul:
  case lltok::kw_fdiv:
  case lltok::kw_frem: {
    FastMathFlags FMF = EatFastMathFlagsIfPresent();
    int Res = ParseArithmetic(Inst, PFS, KeywordVal, 2);
    if (Res != 0)
      return Res;
    if (FMF.any())
      Inst->setFastMathFlags(FMF);
    return 0;
  }

  case lltok::kw_sdiv:
  case lltok::kw_udiv:
  case lltok::kw_lshr:
  case lltok::kw_ashr: {
    bool Exact = EatIfPresent(lltok::kw_exact);

    if (ParseArithmetic(Inst, PFS, KeywordVal, 1)) return true;
    if (Exact) cast<BinaryOperator>(Inst)->setIsExact(true);
    return false;
  }

  case lltok::kw_urem:
  case lltok::kw_srem:   return ParseArithmetic(Inst, PFS, KeywordVal, 1);
  case lltok::kw_and:
  case lltok::kw_or:
  case lltok::kw_xor:    return ParseLogical(Inst, PFS, KeywordVal);
  case lltok::kw_icmp:   return ParseCompare(Inst, PFS, KeywordVal);
  case lltok::kw_fcmp: {
    FastMathFlags FMF = EatFastMathFlagsIfPresent();
    int Res = ParseCompare(Inst, PFS, KeywordVal);
    if (Res != 0)
      return Res;
    if (FMF.any())
      Inst->setFastMathFlags(FMF);
    return 0;
  }

  // Casts.
  case lltok::kw_trunc:
  case lltok::kw_zext:
  case lltok::kw_sext:
  case lltok::kw_fptrunc:
  case lltok::kw_fpext:
  case lltok::kw_bitcast:
  case lltok::kw_addrspacecast:
  case lltok::kw_uitofp:
  case lltok::kw_sitofp:
  case lltok::kw_fptoui:
  case lltok::kw_fptosi:
  case lltok::kw_inttoptr:
  case lltok::kw_ptrtoint:       return ParseCast(Inst, PFS, KeywordVal);
  // Other.
  case lltok::kw_select:         return ParseSelect(Inst, PFS);
  case lltok::kw_va_arg:         return ParseVA_Arg(Inst, PFS);
  case lltok::kw_extractelement: return ParseExtractElement(Inst, PFS);
  case lltok::kw_insertelement:  return ParseInsertElement(Inst, PFS);
  case lltok::kw_shufflevector:  return ParseShuffleVector(Inst, PFS);
  case lltok::kw_phi:            return ParsePHI(Inst, PFS);
  case lltok::kw_landingpad:     return ParseLandingPad(Inst, PFS);
  // Call.
  case lltok::kw_call:     return ParseCall(Inst, PFS, CallInst::TCK_None);
  case lltok::kw_tail:     return ParseCall(Inst, PFS, CallInst::TCK_Tail);
  case lltok::kw_musttail: return ParseCall(Inst, PFS, CallInst::TCK_MustTail);
  // Memory.
  case lltok::kw_alloca:         return ParseAlloc(Inst, PFS);
  case lltok::kw_load:           return ParseLoad(Inst, PFS);
  case lltok::kw_store:          return ParseStore(Inst, PFS);
  case lltok::kw_cmpxchg:        return ParseCmpXchg(Inst, PFS);
  case lltok::kw_atomicrmw:      return ParseAtomicRMW(Inst, PFS);
  case lltok::kw_fence:          return ParseFence(Inst, PFS);
  case lltok::kw_getelementptr: return ParseGetElementPtr(Inst, PFS);
  case lltok::kw_extractvalue:  return ParseExtractValue(Inst, PFS);
  case lltok::kw_insertvalue:   return ParseInsertValue(Inst, PFS);
  }
}

/// ParseCmpPredicate - Parse an integer or fp predicate, based on Kind.
bool LLParser::ParseCmpPredicate(unsigned &P, unsigned Opc) {
  if (Opc == Instruction::FCmp) {
    switch (Lex.getKind()) {
    default: return TokError("expected fcmp predicate (e.g. 'oeq')");
    case lltok::kw_oeq: P = CmpInst::FCMP_OEQ; break;
    case lltok::kw_one: P = CmpInst::FCMP_ONE; break;
    case lltok::kw_olt: P = CmpInst::FCMP_OLT; break;
    case lltok::kw_ogt: P = CmpInst::FCMP_OGT; break;
    case lltok::kw_ole: P = CmpInst::FCMP_OLE; break;
    case lltok::kw_oge: P = CmpInst::FCMP_OGE; break;
    case lltok::kw_ord: P = CmpInst::FCMP_ORD; break;
    case lltok::kw_uno: P = CmpInst::FCMP_UNO; break;
    case lltok::kw_ueq: P = CmpInst::FCMP_UEQ; break;
    case lltok::kw_une: P = CmpInst::FCMP_UNE; break;
    case lltok::kw_ult: P = CmpInst::FCMP_ULT; break;
    case lltok::kw_ugt: P = CmpInst::FCMP_UGT; break;
    case lltok::kw_ule: P = CmpInst::FCMP_ULE; break;
    case lltok::kw_uge: P = CmpInst::FCMP_UGE; break;
    case lltok::kw_true: P = CmpInst::FCMP_TRUE; break;
    case lltok::kw_false: P = CmpInst::FCMP_FALSE; break;
    }
  } else {
    switch (Lex.getKind()) {
    default: return TokError("expected icmp predicate (e.g. 'eq')");
    case lltok::kw_eq:  P = CmpInst::ICMP_EQ; break;
    case lltok::kw_ne:  P = CmpInst::ICMP_NE; break;
    case lltok::kw_slt: P = CmpInst::ICMP_SLT; break;
    case lltok::kw_sgt: P = CmpInst::ICMP_SGT; break;
    case lltok::kw_sle: P = CmpInst::ICMP_SLE; break;
    case lltok::kw_sge: P = CmpInst::ICMP_SGE; break;
    case lltok::kw_ult: P = CmpInst::ICMP_ULT; break;
    case lltok::kw_ugt: P = CmpInst::ICMP_UGT; break;
    case lltok::kw_ule: P = CmpInst::ICMP_ULE; break;
    case lltok::kw_uge: P = CmpInst::ICMP_UGE; break;
    }
  }
  Lex.Lex();
  return false;
}

//===----------------------------------------------------------------------===//
// Terminator Instructions.
//===----------------------------------------------------------------------===//

/// ParseRet - Parse a return instruction.
///   ::= 'ret' void (',' !dbg, !1)*
///   ::= 'ret' TypeAndValue (',' !dbg, !1)*
bool LLParser::ParseRet(Instruction *&Inst, BasicBlock *BB,
                        PerFunctionState &PFS) {
  SMLoc TypeLoc = Lex.getLoc();
  Type *Ty = nullptr;
  if (ParseType(Ty, true /*void allowed*/)) return true;

  Type *ResType = PFS.getFunction().getReturnType();

  if (Ty->isVoidTy()) {
    if (!ResType->isVoidTy())
      return Error(TypeLoc, "value doesn't match function result type '" +
                   getTypeString(ResType) + "'");

    Inst = ReturnInst::Create(Context);
    return false;
  }

  Value *RV;
  if (ParseValue(Ty, RV, PFS)) return true;

  if (ResType != RV->getType())
    return Error(TypeLoc, "value doesn't match function result type '" +
                 getTypeString(ResType) + "'");

  Inst = ReturnInst::Create(Context, RV);
  return false;
}


/// ParseBr
///   ::= 'br' TypeAndValue
///   ::= 'br' TypeAndValue ',' TypeAndValue ',' TypeAndValue
bool LLParser::ParseBr(Instruction *&Inst, PerFunctionState &PFS) {
  LocTy Loc, Loc2;
  Value *Op0;
  BasicBlock *Op1, *Op2;
  if (ParseTypeAndValue(Op0, Loc, PFS)) return true;

  if (BasicBlock *BB = dyn_cast<BasicBlock>(Op0)) {
    Inst = BranchInst::Create(BB);
    return false;
  }

  if (Op0->getType() != Type::getInt1Ty(Context))
    return Error(Loc, "branch condition must have 'i1' type");

  if (ParseToken(lltok::comma, "expected ',' after branch condition") ||
      ParseTypeAndBasicBlock(Op1, Loc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after true destination") ||
      ParseTypeAndBasicBlock(Op2, Loc2, PFS))
    return true;

  Inst = BranchInst::Create(Op1, Op2, Op0);
  return false;
}

/// ParseSwitch
///  Instruction
///    ::= 'switch' TypeAndValue ',' TypeAndValue '[' JumpTable ']'
///  JumpTable
///    ::= (TypeAndValue ',' TypeAndValue)*
bool LLParser::ParseSwitch(Instruction *&Inst, PerFunctionState &PFS) {
  LocTy CondLoc, BBLoc;
  Value *Cond;
  BasicBlock *DefaultBB;
  if (ParseTypeAndValue(Cond, CondLoc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after switch condition") ||
      ParseTypeAndBasicBlock(DefaultBB, BBLoc, PFS) ||
      ParseToken(lltok::lsquare, "expected '[' with switch table"))
    return true;

  if (!Cond->getType()->isIntegerTy())
    return Error(CondLoc, "switch condition must have integer type");

  // Parse the jump table pairs.
  SmallPtrSet<Value*, 32> SeenCases;
  SmallVector<std::pair<ConstantInt*, BasicBlock*>, 32> Table;
  while (Lex.getKind() != lltok::rsquare) {
    Value *Constant;
    BasicBlock *DestBB;

    if (ParseTypeAndValue(Constant, CondLoc, PFS) ||
        ParseToken(lltok::comma, "expected ',' after case value") ||
        ParseTypeAndBasicBlock(DestBB, PFS))
      return true;

    if (!SeenCases.insert(Constant).second)
      return Error(CondLoc, "duplicate case value in switch");
    if (!isa<ConstantInt>(Constant))
      return Error(CondLoc, "case value is not a constant integer");

    Table.push_back(std::make_pair(cast<ConstantInt>(Constant), DestBB));
  }

  Lex.Lex();  // Eat the ']'.

  SwitchInst *SI = SwitchInst::Create(Cond, DefaultBB, Table.size());
  for (unsigned i = 0, e = Table.size(); i != e; ++i)
    SI->addCase(Table[i].first, Table[i].second);
  Inst = SI;
  return false;
}

/// ParseIndirectBr
///  Instruction
///    ::= 'indirectbr' TypeAndValue ',' '[' LabelList ']'
bool LLParser::ParseIndirectBr(Instruction *&Inst, PerFunctionState &PFS) {
  LocTy AddrLoc;
  Value *Address;
  if (ParseTypeAndValue(Address, AddrLoc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after indirectbr address") ||
      ParseToken(lltok::lsquare, "expected '[' with indirectbr"))
    return true;

  if (!Address->getType()->isPointerTy())
    return Error(AddrLoc, "indirectbr address must have pointer type");

  // Parse the destination list.
  SmallVector<BasicBlock*, 16> DestList;

  if (Lex.getKind() != lltok::rsquare) {
    BasicBlock *DestBB;
    if (ParseTypeAndBasicBlock(DestBB, PFS))
      return true;
    DestList.push_back(DestBB);

    while (EatIfPresent(lltok::comma)) {
      if (ParseTypeAndBasicBlock(DestBB, PFS))
        return true;
      DestList.push_back(DestBB);
    }
  }

  if (ParseToken(lltok::rsquare, "expected ']' at end of block list"))
    return true;

  IndirectBrInst *IBI = IndirectBrInst::Create(Address, DestList.size());
  for (unsigned i = 0, e = DestList.size(); i != e; ++i)
    IBI->addDestination(DestList[i]);
  Inst = IBI;
  return false;
}


/// ParseInvoke
///   ::= 'invoke' OptionalCallingConv OptionalAttrs Type Value ParamList
///       OptionalAttrs 'to' TypeAndValue 'unwind' TypeAndValue
bool LLParser::ParseInvoke(Instruction *&Inst, PerFunctionState &PFS) {
  LocTy CallLoc = Lex.getLoc();
  AttrBuilder RetAttrs, FnAttrs;
  std::vector<unsigned> FwdRefAttrGrps;
  LocTy NoBuiltinLoc;
  unsigned CC;
  Type *RetType = nullptr;
  LocTy RetTypeLoc;
  ValID CalleeID;
  SmallVector<ParamInfo, 16> ArgList;

  BasicBlock *NormalBB, *UnwindBB;
  if (ParseOptionalCallingConv(CC) ||
      ParseOptionalReturnAttrs(RetAttrs) ||
      ParseType(RetType, RetTypeLoc, true /*void allowed*/) ||
      ParseValID(CalleeID) ||
      ParseParameterList(ArgList, PFS) ||
      ParseFnAttributeValuePairs(FnAttrs, FwdRefAttrGrps, false,
                                 NoBuiltinLoc) ||
      ParseToken(lltok::kw_to, "expected 'to' in invoke") ||
      ParseTypeAndBasicBlock(NormalBB, PFS) ||
      ParseToken(lltok::kw_unwind, "expected 'unwind' in invoke") ||
      ParseTypeAndBasicBlock(UnwindBB, PFS))
    return true;

  // If RetType is a non-function pointer type, then this is the short syntax
  // for the call, which means that RetType is just the return type.  Infer the
  // rest of the function argument types from the arguments that are present.
  FunctionType *Ty = dyn_cast<FunctionType>(RetType);
  if (!Ty) {
    // Pull out the types of all of the arguments...
    std::vector<Type*> ParamTypes;
    for (unsigned i = 0, e = ArgList.size(); i != e; ++i)
      ParamTypes.push_back(ArgList[i].V->getType());

    if (!FunctionType::isValidReturnType(RetType))
      return Error(RetTypeLoc, "Invalid result type for LLVM function");

    Ty = FunctionType::get(RetType, ParamTypes, false);
  }

  // Look up the callee.
  Value *Callee;
  if (ConvertValIDToValue(PointerType::getUnqual(Ty), CalleeID, Callee, &PFS))
    return true;

  // Set up the Attribute for the function.
  SmallVector<AttributeSet, 8> Attrs;
  if (RetAttrs.hasAttributes())
    Attrs.push_back(AttributeSet::get(RetType->getContext(),
                                      AttributeSet::ReturnIndex,
                                      RetAttrs));

  SmallVector<Value*, 8> Args;

  // Loop through FunctionType's arguments and ensure they are specified
  // correctly.  Also, gather any parameter attributes.
  FunctionType::param_iterator I = Ty->param_begin();
  FunctionType::param_iterator E = Ty->param_end();
  for (unsigned i = 0, e = ArgList.size(); i != e; ++i) {
    Type *ExpectedTy = nullptr;
    if (I != E) {
      ExpectedTy = *I++;
    } else if (!Ty->isVarArg()) {
      return Error(ArgList[i].Loc, "too many arguments specified");
    }

    if (ExpectedTy && ExpectedTy != ArgList[i].V->getType())
      return Error(ArgList[i].Loc, "argument is not of expected type '" +
                   getTypeString(ExpectedTy) + "'");
    Args.push_back(ArgList[i].V);
    if (ArgList[i].Attrs.hasAttributes(i + 1)) {
      AttrBuilder B(ArgList[i].Attrs, i + 1);
      Attrs.push_back(AttributeSet::get(RetType->getContext(), i + 1, B));
    }
  }

  if (I != E)
    return Error(CallLoc, "not enough parameters specified for call");

  if (FnAttrs.hasAttributes()) {
    if (FnAttrs.hasAlignmentAttr())
      return Error(CallLoc, "invoke instructions may not have an alignment");

    Attrs.push_back(AttributeSet::get(RetType->getContext(),
                                      AttributeSet::FunctionIndex,
                                      FnAttrs));
  }

  // Finish off the Attribute and check them
  AttributeSet PAL = AttributeSet::get(Context, Attrs);

  InvokeInst *II = InvokeInst::Create(Ty, Callee, NormalBB, UnwindBB, Args);
  II->setCallingConv(CC);
  II->setAttributes(PAL);
  ForwardRefAttrGroups[II] = FwdRefAttrGrps;
  Inst = II;
  return false;
}

/// ParseResume
///   ::= 'resume' TypeAndValue
bool LLParser::ParseResume(Instruction *&Inst, PerFunctionState &PFS) {
  Value *Exn; LocTy ExnLoc;
  if (ParseTypeAndValue(Exn, ExnLoc, PFS))
    return true;

  ResumeInst *RI = ResumeInst::Create(Exn);
  Inst = RI;
  return false;
}

//===----------------------------------------------------------------------===//
// Binary Operators.
//===----------------------------------------------------------------------===//

/// ParseArithmetic
///  ::= ArithmeticOps TypeAndValue ',' Value
///
/// If OperandType is 0, then any FP or integer operand is allowed.  If it is 1,
/// then any integer operand is allowed, if it is 2, any fp operand is allowed.
bool LLParser::ParseArithmetic(Instruction *&Inst, PerFunctionState &PFS,
                               unsigned Opc, unsigned OperandType) {
  LocTy Loc; Value *LHS, *RHS;
  if (ParseTypeAndValue(LHS, Loc, PFS) ||
      ParseToken(lltok::comma, "expected ',' in arithmetic operation") ||
      ParseValue(LHS->getType(), RHS, PFS))
    return true;

  bool Valid;
  switch (OperandType) {
  default: llvm_unreachable("Unknown operand type!");
  case 0: // int or FP.
    Valid = LHS->getType()->isIntOrIntVectorTy() ||
            LHS->getType()->isFPOrFPVectorTy();
    break;
  case 1: Valid = LHS->getType()->isIntOrIntVectorTy(); break;
  case 2: Valid = LHS->getType()->isFPOrFPVectorTy(); break;
  }

  if (!Valid)
    return Error(Loc, "invalid operand type for instruction");

  Inst = BinaryOperator::Create((Instruction::BinaryOps)Opc, LHS, RHS);
  return false;
}

/// ParseLogical
///  ::= ArithmeticOps TypeAndValue ',' Value {
bool LLParser::ParseLogical(Instruction *&Inst, PerFunctionState &PFS,
                            unsigned Opc) {
  LocTy Loc; Value *LHS, *RHS;
  if (ParseTypeAndValue(LHS, Loc, PFS) ||
      ParseToken(lltok::comma, "expected ',' in logical operation") ||
      ParseValue(LHS->getType(), RHS, PFS))
    return true;

  if (!LHS->getType()->isIntOrIntVectorTy())
    return Error(Loc,"instruction requires integer or integer vector operands");

  Inst = BinaryOperator::Create((Instruction::BinaryOps)Opc, LHS, RHS);
  return false;
}


/// ParseCompare
///  ::= 'icmp' IPredicates TypeAndValue ',' Value
///  ::= 'fcmp' FPredicates TypeAndValue ',' Value
bool LLParser::ParseCompare(Instruction *&Inst, PerFunctionState &PFS,
                            unsigned Opc) {
  // Parse the integer/fp comparison predicate.
  LocTy Loc;
  unsigned Pred;
  Value *LHS, *RHS;
  if (ParseCmpPredicate(Pred, Opc) ||
      ParseTypeAndValue(LHS, Loc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after compare value") ||
      ParseValue(LHS->getType(), RHS, PFS))
    return true;

  if (Opc == Instruction::FCmp) {
    if (!LHS->getType()->isFPOrFPVectorTy())
      return Error(Loc, "fcmp requires floating point operands");
    Inst = new FCmpInst(CmpInst::Predicate(Pred), LHS, RHS);
  } else {
    assert(Opc == Instruction::ICmp && "Unknown opcode for CmpInst!");
    if (!LHS->getType()->isIntOrIntVectorTy() &&
        !LHS->getType()->getScalarType()->isPointerTy())
      return Error(Loc, "icmp requires integer operands");
    Inst = new ICmpInst(CmpInst::Predicate(Pred), LHS, RHS);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Other Instructions.
//===----------------------------------------------------------------------===//


/// ParseCast
///   ::= CastOpc TypeAndValue 'to' Type
bool LLParser::ParseCast(Instruction *&Inst, PerFunctionState &PFS,
                         unsigned Opc) {
  LocTy Loc;
  Value *Op;
  Type *DestTy = nullptr;
  if (ParseTypeAndValue(Op, Loc, PFS) ||
      ParseToken(lltok::kw_to, "expected 'to' after cast value") ||
      ParseType(DestTy))
    return true;

  if (!CastInst::castIsValid((Instruction::CastOps)Opc, Op, DestTy)) {
    CastInst::castIsValid((Instruction::CastOps)Opc, Op, DestTy);
    return Error(Loc, "invalid cast opcode for cast from '" +
                 getTypeString(Op->getType()) + "' to '" +
                 getTypeString(DestTy) + "'");
  }
  Inst = CastInst::Create((Instruction::CastOps)Opc, Op, DestTy);
  return false;
}

/// ParseSelect
///   ::= 'select' TypeAndValue ',' TypeAndValue ',' TypeAndValue
bool LLParser::ParseSelect(Instruction *&Inst, PerFunctionState &PFS) {
  LocTy Loc;
  Value *Op0, *Op1, *Op2;
  if (ParseTypeAndValue(Op0, Loc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after select condition") ||
      ParseTypeAndValue(Op1, PFS) ||
      ParseToken(lltok::comma, "expected ',' after select value") ||
      ParseTypeAndValue(Op2, PFS))
    return true;

  if (const char *Reason = SelectInst::areInvalidOperands(Op0, Op1, Op2))
    return Error(Loc, Reason);

  Inst = SelectInst::Create(Op0, Op1, Op2);
  return false;
}

/// ParseVA_Arg
///   ::= 'va_arg' TypeAndValue ',' Type
bool LLParser::ParseVA_Arg(Instruction *&Inst, PerFunctionState &PFS) {
  Value *Op;
  Type *EltTy = nullptr;
  LocTy TypeLoc;
  if (ParseTypeAndValue(Op, PFS) ||
      ParseToken(lltok::comma, "expected ',' after vaarg operand") ||
      ParseType(EltTy, TypeLoc))
    return true;

  if (!EltTy->isFirstClassType())
    return Error(TypeLoc, "va_arg requires operand with first class type");

  Inst = new VAArgInst(Op, EltTy);
  return false;
}

/// ParseExtractElement
///   ::= 'extractelement' TypeAndValue ',' TypeAndValue
bool LLParser::ParseExtractElement(Instruction *&Inst, PerFunctionState &PFS) {
  LocTy Loc;
  Value *Op0, *Op1;
  if (ParseTypeAndValue(Op0, Loc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after extract value") ||
      ParseTypeAndValue(Op1, PFS))
    return true;

  if (!ExtractElementInst::isValidOperands(Op0, Op1))
    return Error(Loc, "invalid extractelement operands");

  Inst = ExtractElementInst::Create(Op0, Op1);
  return false;
}

/// ParseInsertElement
///   ::= 'insertelement' TypeAndValue ',' TypeAndValue ',' TypeAndValue
bool LLParser::ParseInsertElement(Instruction *&Inst, PerFunctionState &PFS) {
  LocTy Loc;
  Value *Op0, *Op1, *Op2;
  if (ParseTypeAndValue(Op0, Loc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after insertelement value") ||
      ParseTypeAndValue(Op1, PFS) ||
      ParseToken(lltok::comma, "expected ',' after insertelement value") ||
      ParseTypeAndValue(Op2, PFS))
    return true;

  if (!InsertElementInst::isValidOperands(Op0, Op1, Op2))
    return Error(Loc, "invalid insertelement operands");

  Inst = InsertElementInst::Create(Op0, Op1, Op2);
  return false;
}

/// ParseShuffleVector
///   ::= 'shufflevector' TypeAndValue ',' TypeAndValue ',' TypeAndValue
bool LLParser::ParseShuffleVector(Instruction *&Inst, PerFunctionState &PFS) {
  LocTy Loc;
  Value *Op0, *Op1, *Op2;
  if (ParseTypeAndValue(Op0, Loc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after shuffle mask") ||
      ParseTypeAndValue(Op1, PFS) ||
      ParseToken(lltok::comma, "expected ',' after shuffle value") ||
      ParseTypeAndValue(Op2, PFS))
    return true;

  if (!ShuffleVectorInst::isValidOperands(Op0, Op1, Op2))
    return Error(Loc, "invalid shufflevector operands");

  Inst = new ShuffleVectorInst(Op0, Op1, Op2);
  return false;
}

/// ParsePHI
///   ::= 'phi' Type '[' Value ',' Value ']' (',' '[' Value ',' Value ']')*
int LLParser::ParsePHI(Instruction *&Inst, PerFunctionState &PFS) {
  Type *Ty = nullptr;  LocTy TypeLoc;
  Value *Op0, *Op1;

  if (ParseType(Ty, TypeLoc) ||
      ParseToken(lltok::lsquare, "expected '[' in phi value list") ||
      ParseValue(Ty, Op0, PFS) ||
      ParseToken(lltok::comma, "expected ',' after insertelement value") ||
      ParseValue(Type::getLabelTy(Context), Op1, PFS) ||
      ParseToken(lltok::rsquare, "expected ']' in phi value list"))
    return true;

  bool AteExtraComma = false;
  SmallVector<std::pair<Value*, BasicBlock*>, 16> PHIVals;
  while (1) {
    PHIVals.push_back(std::make_pair(Op0, cast<BasicBlock>(Op1)));

    if (!EatIfPresent(lltok::comma))
      break;

    if (Lex.getKind() == lltok::MetadataVar) {
      AteExtraComma = true;
      break;
    }

    if (ParseToken(lltok::lsquare, "expected '[' in phi value list") ||
        ParseValue(Ty, Op0, PFS) ||
        ParseToken(lltok::comma, "expected ',' after insertelement value") ||
        ParseValue(Type::getLabelTy(Context), Op1, PFS) ||
        ParseToken(lltok::rsquare, "expected ']' in phi value list"))
      return true;
  }

  if (!Ty->isFirstClassType())
    return Error(TypeLoc, "phi node must have first class type");

  PHINode *PN = PHINode::Create(Ty, PHIVals.size());
  for (unsigned i = 0, e = PHIVals.size(); i != e; ++i)
    PN->addIncoming(PHIVals[i].first, PHIVals[i].second);
  Inst = PN;
  return AteExtraComma ? InstExtraComma : InstNormal;
}

/// ParseLandingPad
///   ::= 'landingpad' Type 'personality' TypeAndValue 'cleanup'? Clause+
/// Clause
///   ::= 'catch' TypeAndValue
///   ::= 'filter'
///   ::= 'filter' TypeAndValue ( ',' TypeAndValue )*
bool LLParser::ParseLandingPad(Instruction *&Inst, PerFunctionState &PFS) {
  Type *Ty = nullptr; LocTy TyLoc;

  if (ParseType(Ty, TyLoc))
    return true;

  std::unique_ptr<LandingPadInst> LP(LandingPadInst::Create(Ty, 0));
  LP->setCleanup(EatIfPresent(lltok::kw_cleanup));

  while (Lex.getKind() == lltok::kw_catch || Lex.getKind() == lltok::kw_filter){
    LandingPadInst::ClauseType CT;
    if (EatIfPresent(lltok::kw_catch))
      CT = LandingPadInst::Catch;
    else if (EatIfPresent(lltok::kw_filter))
      CT = LandingPadInst::Filter;
    else
      return TokError("expected 'catch' or 'filter' clause type");

    Value *V;
    LocTy VLoc;
    if (ParseTypeAndValue(V, VLoc, PFS))
      return true;

    // A 'catch' type expects a non-array constant. A filter clause expects an
    // array constant.
    if (CT == LandingPadInst::Catch) {
      if (isa<ArrayType>(V->getType()))
        Error(VLoc, "'catch' clause has an invalid type");
    } else {
      if (!isa<ArrayType>(V->getType()))
        Error(VLoc, "'filter' clause has an invalid type");
    }

    Constant *CV = dyn_cast<Constant>(V);
    if (!CV)
      return Error(VLoc, "clause argument must be a constant");
    LP->addClause(CV);
  }

  Inst = LP.release();
  return false;
}

/// ParseCall
///   ::= 'call' OptionalCallingConv OptionalAttrs Type Value
///       ParameterList OptionalAttrs
///   ::= 'tail' 'call' OptionalCallingConv OptionalAttrs Type Value
///       ParameterList OptionalAttrs
///   ::= 'musttail' 'call' OptionalCallingConv OptionalAttrs Type Value
///       ParameterList OptionalAttrs
bool LLParser::ParseCall(Instruction *&Inst, PerFunctionState &PFS,
                         CallInst::TailCallKind TCK) {
  AttrBuilder RetAttrs, FnAttrs;
  std::vector<unsigned> FwdRefAttrGrps;
  LocTy BuiltinLoc;
  unsigned CC;
  Type *RetType = nullptr;
  LocTy RetTypeLoc;
  ValID CalleeID;
  SmallVector<ParamInfo, 16> ArgList;
  LocTy CallLoc = Lex.getLoc();

  if ((TCK != CallInst::TCK_None &&
       ParseToken(lltok::kw_call, "expected 'tail call'")) ||
      ParseOptionalCallingConv(CC) ||
      ParseOptionalReturnAttrs(RetAttrs) ||
      ParseType(RetType, RetTypeLoc, true /*void allowed*/) ||
      ParseValID(CalleeID) ||
      ParseParameterList(ArgList, PFS, TCK == CallInst::TCK_MustTail,
                         PFS.getFunction().isVarArg()) ||
      ParseFnAttributeValuePairs(FnAttrs, FwdRefAttrGrps, false,
                                 BuiltinLoc))
    return true;

  // If RetType is a non-function pointer type, then this is the short syntax
  // for the call, which means that RetType is just the return type.  Infer the
  // rest of the function argument types from the arguments that are present.
  FunctionType *Ty = dyn_cast<FunctionType>(RetType);
  if (!Ty) {
    // Pull out the types of all of the arguments...
    std::vector<Type*> ParamTypes;
    for (unsigned i = 0, e = ArgList.size(); i != e; ++i)
      ParamTypes.push_back(ArgList[i].V->getType());

    if (!FunctionType::isValidReturnType(RetType))
      return Error(RetTypeLoc, "Invalid result type for LLVM function");

    Ty = FunctionType::get(RetType, ParamTypes, false);
  }

  // Look up the callee.
  Value *Callee;
  if (ConvertValIDToValue(PointerType::getUnqual(Ty), CalleeID, Callee, &PFS))
    return true;

  // Set up the Attribute for the function.
  SmallVector<AttributeSet, 8> Attrs;
  if (RetAttrs.hasAttributes())
    Attrs.push_back(AttributeSet::get(RetType->getContext(),
                                      AttributeSet::ReturnIndex,
                                      RetAttrs));

  SmallVector<Value*, 8> Args;

  // Loop through FunctionType's arguments and ensure they are specified
  // correctly.  Also, gather any parameter attributes.
  FunctionType::param_iterator I = Ty->param_begin();
  FunctionType::param_iterator E = Ty->param_end();
  for (unsigned i = 0, e = ArgList.size(); i != e; ++i) {
    Type *ExpectedTy = nullptr;
    if (I != E) {
      ExpectedTy = *I++;
    } else if (!Ty->isVarArg()) {
      return Error(ArgList[i].Loc, "too many arguments specified");
    }

    if (ExpectedTy && ExpectedTy != ArgList[i].V->getType())
      return Error(ArgList[i].Loc, "argument is not of expected type '" +
                   getTypeString(ExpectedTy) + "'");
    Args.push_back(ArgList[i].V);
    if (ArgList[i].Attrs.hasAttributes(i + 1)) {
      AttrBuilder B(ArgList[i].Attrs, i + 1);
      Attrs.push_back(AttributeSet::get(RetType->getContext(), i + 1, B));
    }
  }

  if (I != E)
    return Error(CallLoc, "not enough parameters specified for call");

  if (FnAttrs.hasAttributes()) {
    if (FnAttrs.hasAlignmentAttr())
      return Error(CallLoc, "call instructions may not have an alignment");

    Attrs.push_back(AttributeSet::get(RetType->getContext(),
                                      AttributeSet::FunctionIndex,
                                      FnAttrs));
  }

  // Finish off the Attribute and check them
  AttributeSet PAL = AttributeSet::get(Context, Attrs);

  CallInst *CI = CallInst::Create(Ty, Callee, Args);
  CI->setTailCallKind(TCK);
  CI->setCallingConv(CC);
  CI->setAttributes(PAL);
  ForwardRefAttrGroups[CI] = FwdRefAttrGrps;
  Inst = CI;
  return false;
}

//===----------------------------------------------------------------------===//
// Memory Instructions.
//===----------------------------------------------------------------------===//

/// ParseAlloc
///   ::= 'alloca' 'inalloca'? Type (',' TypeAndValue)? (',' 'align' i32)?
int LLParser::ParseAlloc(Instruction *&Inst, PerFunctionState &PFS) {
  Value *Size = nullptr;
  LocTy SizeLoc, TyLoc;
  unsigned Alignment = 0;
  Type *Ty = nullptr;

  bool IsInAlloca = EatIfPresent(lltok::kw_inalloca);

  if (ParseType(Ty, TyLoc)) return true;

  if (Ty->isFunctionTy() || !PointerType::isValidElementType(Ty))
    return Error(TyLoc, "invalid type for alloca");

  bool AteExtraComma = false;
  if (EatIfPresent(lltok::comma)) {
    if (Lex.getKind() == lltok::kw_align) {
      if (ParseOptionalAlignment(Alignment)) return true;
    } else if (Lex.getKind() == lltok::MetadataVar) {
      AteExtraComma = true;
    } else {
      if (ParseTypeAndValue(Size, SizeLoc, PFS) ||
          ParseOptionalCommaAlign(Alignment, AteExtraComma))
        return true;
    }
  }

  if (Size && !Size->getType()->isIntegerTy())
    return Error(SizeLoc, "element count must have integer type");

  AllocaInst *AI = new AllocaInst(Ty, Size, Alignment);
  AI->setUsedWithInAlloca(IsInAlloca);
  Inst = AI;
  return AteExtraComma ? InstExtraComma : InstNormal;
}

/// ParseLoad
///   ::= 'load' 'volatile'? TypeAndValue (',' 'align' i32)?
///   ::= 'load' 'atomic' 'volatile'? TypeAndValue
///       'singlethread'? AtomicOrdering (',' 'align' i32)?
int LLParser::ParseLoad(Instruction *&Inst, PerFunctionState &PFS) {
  Value *Val; LocTy Loc;
  unsigned Alignment = 0;
  bool AteExtraComma = false;
  bool isAtomic = false;
  AtomicOrdering Ordering = NotAtomic;
  SynchronizationScope Scope = CrossThread;

  if (Lex.getKind() == lltok::kw_atomic) {
    isAtomic = true;
    Lex.Lex();
  }

  bool isVolatile = false;
  if (Lex.getKind() == lltok::kw_volatile) {
    isVolatile = true;
    Lex.Lex();
  }

  Type *Ty;
  LocTy ExplicitTypeLoc = Lex.getLoc();
  if (ParseType(Ty) ||
      ParseToken(lltok::comma, "expected comma after load's type") ||
      ParseTypeAndValue(Val, Loc, PFS) ||
      ParseScopeAndOrdering(isAtomic, Scope, Ordering) ||
      ParseOptionalCommaAlign(Alignment, AteExtraComma))
    return true;

  if (!Val->getType()->isPointerTy() || !Ty->isFirstClassType())
    return Error(Loc, "load operand must be a pointer to a first class type");
  if (isAtomic && !Alignment)
    return Error(Loc, "atomic load must have explicit non-zero alignment");
  if (Ordering == Release || Ordering == AcquireRelease)
    return Error(Loc, "atomic load cannot use Release ordering");

  if (Ty != cast<PointerType>(Val->getType())->getElementType())
    return Error(ExplicitTypeLoc,
                 "explicit pointee type doesn't match operand's pointee type");

  Inst = new LoadInst(Ty, Val, "", isVolatile, Alignment, Ordering, Scope);
  return AteExtraComma ? InstExtraComma : InstNormal;
}

/// ParseStore

///   ::= 'store' 'volatile'? TypeAndValue ',' TypeAndValue (',' 'align' i32)?
///   ::= 'store' 'atomic' 'volatile'? TypeAndValue ',' TypeAndValue
///       'singlethread'? AtomicOrdering (',' 'align' i32)?
int LLParser::ParseStore(Instruction *&Inst, PerFunctionState &PFS) {
  Value *Val, *Ptr; LocTy Loc, PtrLoc;
  unsigned Alignment = 0;
  bool AteExtraComma = false;
  bool isAtomic = false;
  AtomicOrdering Ordering = NotAtomic;
  SynchronizationScope Scope = CrossThread;

  if (Lex.getKind() == lltok::kw_atomic) {
    isAtomic = true;
    Lex.Lex();
  }

  bool isVolatile = false;
  if (Lex.getKind() == lltok::kw_volatile) {
    isVolatile = true;
    Lex.Lex();
  }

  if (ParseTypeAndValue(Val, Loc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after store operand") ||
      ParseTypeAndValue(Ptr, PtrLoc, PFS) ||
      ParseScopeAndOrdering(isAtomic, Scope, Ordering) ||
      ParseOptionalCommaAlign(Alignment, AteExtraComma))
    return true;

  if (!Ptr->getType()->isPointerTy())
    return Error(PtrLoc, "store operand must be a pointer");
  if (!Val->getType()->isFirstClassType())
    return Error(Loc, "store operand must be a first class value");
  if (cast<PointerType>(Ptr->getType())->getElementType() != Val->getType())
    return Error(Loc, "stored value and pointer type do not match");
  if (isAtomic && !Alignment)
    return Error(Loc, "atomic store must have explicit non-zero alignment");
  if (Ordering == Acquire || Ordering == AcquireRelease)
    return Error(Loc, "atomic store cannot use Acquire ordering");

  Inst = new StoreInst(Val, Ptr, isVolatile, Alignment, Ordering, Scope);
  return AteExtraComma ? InstExtraComma : InstNormal;
}

/// ParseCmpXchg
///   ::= 'cmpxchg' 'weak'? 'volatile'? TypeAndValue ',' TypeAndValue ','
///       TypeAndValue 'singlethread'? AtomicOrdering AtomicOrdering
int LLParser::ParseCmpXchg(Instruction *&Inst, PerFunctionState &PFS) {
  Value *Ptr, *Cmp, *New; LocTy PtrLoc, CmpLoc, NewLoc;
  bool AteExtraComma = false;
  AtomicOrdering SuccessOrdering = NotAtomic;
  AtomicOrdering FailureOrdering = NotAtomic;
  SynchronizationScope Scope = CrossThread;
  bool isVolatile = false;
  bool isWeak = false;

  if (EatIfPresent(lltok::kw_weak))
    isWeak = true;

  if (EatIfPresent(lltok::kw_volatile))
    isVolatile = true;

  if (ParseTypeAndValue(Ptr, PtrLoc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after cmpxchg address") ||
      ParseTypeAndValue(Cmp, CmpLoc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after cmpxchg cmp operand") ||
      ParseTypeAndValue(New, NewLoc, PFS) ||
      ParseScopeAndOrdering(true /*Always atomic*/, Scope, SuccessOrdering) ||
      ParseOrdering(FailureOrdering))
    return true;

  if (SuccessOrdering == Unordered || FailureOrdering == Unordered)
    return TokError("cmpxchg cannot be unordered");
  if (SuccessOrdering < FailureOrdering)
    return TokError("cmpxchg must be at least as ordered on success as failure");
  if (FailureOrdering == Release || FailureOrdering == AcquireRelease)
    return TokError("cmpxchg failure ordering cannot include release semantics");
  if (!Ptr->getType()->isPointerTy())
    return Error(PtrLoc, "cmpxchg operand must be a pointer");
  if (cast<PointerType>(Ptr->getType())->getElementType() != Cmp->getType())
    return Error(CmpLoc, "compare value and pointer type do not match");
  if (cast<PointerType>(Ptr->getType())->getElementType() != New->getType())
    return Error(NewLoc, "new value and pointer type do not match");
  if (!New->getType()->isIntegerTy())
    return Error(NewLoc, "cmpxchg operand must be an integer");
  unsigned Size = New->getType()->getPrimitiveSizeInBits();
  if (Size < 8 || (Size & (Size - 1)))
    return Error(NewLoc, "cmpxchg operand must be power-of-two byte-sized"
                         " integer");

  AtomicCmpXchgInst *CXI = new AtomicCmpXchgInst(
      Ptr, Cmp, New, SuccessOrdering, FailureOrdering, Scope);
  CXI->setVolatile(isVolatile);
  CXI->setWeak(isWeak);
  Inst = CXI;
  return AteExtraComma ? InstExtraComma : InstNormal;
}

/// ParseAtomicRMW
///   ::= 'atomicrmw' 'volatile'? BinOp TypeAndValue ',' TypeAndValue
///       'singlethread'? AtomicOrdering
int LLParser::ParseAtomicRMW(Instruction *&Inst, PerFunctionState &PFS) {
  Value *Ptr, *Val; LocTy PtrLoc, ValLoc;
  bool AteExtraComma = false;
  AtomicOrdering Ordering = NotAtomic;
  SynchronizationScope Scope = CrossThread;
  bool isVolatile = false;
  AtomicRMWInst::BinOp Operation;

  if (EatIfPresent(lltok::kw_volatile))
    isVolatile = true;

  switch (Lex.getKind()) {
  default: return TokError("expected binary operation in atomicrmw");
  case lltok::kw_xchg: Operation = AtomicRMWInst::Xchg; break;
  case lltok::kw_add: Operation = AtomicRMWInst::Add; break;
  case lltok::kw_sub: Operation = AtomicRMWInst::Sub; break;
  case lltok::kw_and: Operation = AtomicRMWInst::And; break;
  case lltok::kw_nand: Operation = AtomicRMWInst::Nand; break;
  case lltok::kw_or: Operation = AtomicRMWInst::Or; break;
  case lltok::kw_xor: Operation = AtomicRMWInst::Xor; break;
  case lltok::kw_max: Operation = AtomicRMWInst::Max; break;
  case lltok::kw_min: Operation = AtomicRMWInst::Min; break;
  case lltok::kw_umax: Operation = AtomicRMWInst::UMax; break;
  case lltok::kw_umin: Operation = AtomicRMWInst::UMin; break;
  }
  Lex.Lex();  // Eat the operation.

  if (ParseTypeAndValue(Ptr, PtrLoc, PFS) ||
      ParseToken(lltok::comma, "expected ',' after atomicrmw address") ||
      ParseTypeAndValue(Val, ValLoc, PFS) ||
      ParseScopeAndOrdering(true /*Always atomic*/, Scope, Ordering))
    return true;

  if (Ordering == Unordered)
    return TokError("atomicrmw cannot be unordered");
  if (!Ptr->getType()->isPointerTy())
    return Error(PtrLoc, "atomicrmw operand must be a pointer");
  if (cast<PointerType>(Ptr->getType())->getElementType() != Val->getType())
    return Error(ValLoc, "atomicrmw value and pointer type do not match");
  if (!Val->getType()->isIntegerTy())
    return Error(ValLoc, "atomicrmw operand must be an integer");
  unsigned Size = Val->getType()->getPrimitiveSizeInBits();
  if (Size < 8 || (Size & (Size - 1)))
    return Error(ValLoc, "atomicrmw operand must be power-of-two byte-sized"
                         " integer");

  AtomicRMWInst *RMWI =
    new AtomicRMWInst(Operation, Ptr, Val, Ordering, Scope);
  RMWI->setVolatile(isVolatile);
  Inst = RMWI;
  return AteExtraComma ? InstExtraComma : InstNormal;
}

/// ParseFence
///   ::= 'fence' 'singlethread'? AtomicOrdering
int LLParser::ParseFence(Instruction *&Inst, PerFunctionState &PFS) {
  AtomicOrdering Ordering = NotAtomic;
  SynchronizationScope Scope = CrossThread;
  if (ParseScopeAndOrdering(true /*Always atomic*/, Scope, Ordering))
    return true;

  if (Ordering == Unordered)
    return TokError("fence cannot be unordered");
  if (Ordering == Monotonic)
    return TokError("fence cannot be monotonic");

  Inst = new FenceInst(Context, Ordering, Scope);
  return InstNormal;
}

/// ParseGetElementPtr
///   ::= 'getelementptr' 'inbounds'? TypeAndValue (',' TypeAndValue)*
int LLParser::ParseGetElementPtr(Instruction *&Inst, PerFunctionState &PFS) {
  Value *Ptr = nullptr;
  Value *Val = nullptr;
  LocTy Loc, EltLoc;

  bool InBounds = EatIfPresent(lltok::kw_inbounds);

  Type *Ty = nullptr;
  LocTy ExplicitTypeLoc = Lex.getLoc();
  if (ParseType(Ty) ||
      ParseToken(lltok::comma, "expected comma after getelementptr's type") ||
      ParseTypeAndValue(Ptr, Loc, PFS))
    return true;

  Type *BaseType = Ptr->getType();
  PointerType *BasePointerType = dyn_cast<PointerType>(BaseType->getScalarType());
  if (!BasePointerType)
    return Error(Loc, "base of getelementptr must be a pointer");

  if (Ty != BasePointerType->getElementType())
    return Error(ExplicitTypeLoc,
                 "explicit pointee type doesn't match operand's pointee type");

  SmallVector<Value*, 16> Indices;
  bool AteExtraComma = false;
  // GEP returns a vector of pointers if at least one of parameters is a vector.
  // All vector parameters should have the same vector width.
  unsigned GEPWidth = BaseType->isVectorTy() ?
    BaseType->getVectorNumElements() : 0;

  while (EatIfPresent(lltok::comma)) {
    if (Lex.getKind() == lltok::MetadataVar) {
      AteExtraComma = true;
      break;
    }
    if (ParseTypeAndValue(Val, EltLoc, PFS)) return true;
    if (!Val->getType()->getScalarType()->isIntegerTy())
      return Error(EltLoc, "getelementptr index must be an integer");

    if (Val->getType()->isVectorTy()) {
      unsigned ValNumEl = Val->getType()->getVectorNumElements();
      if (GEPWidth && GEPWidth != ValNumEl)
        return Error(EltLoc,
          "getelementptr vector index has a wrong number of elements");
      GEPWidth = ValNumEl;
    }
    Indices.push_back(Val);
  }

  SmallPtrSet<const Type*, 4> Visited;
  if (!Indices.empty() && !Ty->isSized(&Visited))
    return Error(Loc, "base element of getelementptr must be sized");

  if (!GetElementPtrInst::getIndexedType(Ty, Indices))
    return Error(Loc, "invalid getelementptr indices");
  Inst = GetElementPtrInst::Create(Ty, Ptr, Indices);
  if (InBounds)
    cast<GetElementPtrInst>(Inst)->setIsInBounds(true);
  return AteExtraComma ? InstExtraComma : InstNormal;
}

/// ParseExtractValue
///   ::= 'extractvalue' TypeAndValue (',' uint32)+
int LLParser::ParseExtractValue(Instruction *&Inst, PerFunctionState &PFS) {
  Value *Val; LocTy Loc;
  SmallVector<unsigned, 4> Indices;
  bool AteExtraComma;
  if (ParseTypeAndValue(Val, Loc, PFS) ||
      ParseIndexList(Indices, AteExtraComma))
    return true;

  if (!Val->getType()->isAggregateType())
    return Error(Loc, "extractvalue operand must be aggregate type");

  if (!ExtractValueInst::getIndexedType(Val->getType(), Indices))
    return Error(Loc, "invalid indices for extractvalue");
  Inst = ExtractValueInst::Create(Val, Indices);
  return AteExtraComma ? InstExtraComma : InstNormal;
}

/// ParseInsertValue
///   ::= 'insertvalue' TypeAndValue ',' TypeAndValue (',' uint32)+
int LLParser::ParseInsertValue(Instruction *&Inst, PerFunctionState &PFS) {
  Value *Val0, *Val1; LocTy Loc0, Loc1;
  SmallVector<unsigned, 4> Indices;
  bool AteExtraComma;
  if (ParseTypeAndValue(Val0, Loc0, PFS) ||
      ParseToken(lltok::comma, "expected comma after insertvalue operand") ||
      ParseTypeAndValue(Val1, Loc1, PFS) ||
      ParseIndexList(Indices, AteExtraComma))
    return true;

  if (!Val0->getType()->isAggregateType())
    return Error(Loc0, "insertvalue operand must be aggregate type");

  Type *IndexedType = ExtractValueInst::getIndexedType(Val0->getType(), Indices);
  if (!IndexedType)
    return Error(Loc0, "invalid indices for insertvalue");
  if (IndexedType != Val1->getType())
    return Error(Loc1, "insertvalue operand and field disagree in type: '" +
                           getTypeString(Val1->getType()) + "' instead of '" +
                           getTypeString(IndexedType) + "'");
  Inst = InsertValueInst::Create(Val0, Val1, Indices);
  return AteExtraComma ? InstExtraComma : InstNormal;
}

//===----------------------------------------------------------------------===//
// Embedded metadata.
//===----------------------------------------------------------------------===//

/// ParseMDNodeVector
///   ::= { Element (',' Element)* }
/// Element
///   ::= 'null' | TypeAndValue
bool LLParser::ParseMDNodeVector(SmallVectorImpl<Metadata *> &Elts) {
  if (ParseToken(lltok::lbrace, "expected '{' here"))
    return true;

  // Check for an empty list.
  if (EatIfPresent(lltok::rbrace))
    return false;

  do {
    // Null is a special case since it is typeless.
    if (EatIfPresent(lltok::kw_null)) {
      Elts.push_back(nullptr);
      continue;
    }

    Metadata *MD;
    if (ParseMetadata(MD, nullptr))
      return true;
    Elts.push_back(MD);
  } while (EatIfPresent(lltok::comma));

  return ParseToken(lltok::rbrace, "expected end of metadata node");
}

//===----------------------------------------------------------------------===//
// Use-list order directives.
//===----------------------------------------------------------------------===//
bool LLParser::sortUseListOrder(Value *V, ArrayRef<unsigned> Indexes,
                                SMLoc Loc) {
  if (V->use_empty())
    return Error(Loc, "value has no uses");

  unsigned NumUses = 0;
  SmallDenseMap<const Use *, unsigned, 16> Order;
  for (const Use &U : V->uses()) {
    if (++NumUses > Indexes.size())
      break;
    Order[&U] = Indexes[NumUses - 1];
  }
  if (NumUses < 2)
    return Error(Loc, "value only has one use");
  if (Order.size() != Indexes.size() || NumUses > Indexes.size())
    return Error(Loc, "wrong number of indexes, expected " +
                          Twine(std::distance(V->use_begin(), V->use_end())));

  V->sortUseList([&](const Use &L, const Use &R) {
    return Order.lookup(&L) < Order.lookup(&R);
  });
  return false;
}

/// ParseUseListOrderIndexes
///   ::= '{' uint32 (',' uint32)+ '}'
bool LLParser::ParseUseListOrderIndexes(SmallVectorImpl<unsigned> &Indexes) {
  SMLoc Loc = Lex.getLoc();
  if (ParseToken(lltok::lbrace, "expected '{' here"))
    return true;
  if (Lex.getKind() == lltok::rbrace)
    return Lex.Error("expected non-empty list of uselistorder indexes");

  // Use Offset, Max, and IsOrdered to check consistency of indexes.  The
  // indexes should be distinct numbers in the range [0, size-1], and should
  // not be in order.
  unsigned Offset = 0;
  unsigned Max = 0;
  bool IsOrdered = true;
  assert(Indexes.empty() && "Expected empty order vector");
  do {
    unsigned Index;
    if (ParseUInt32(Index))
      return true;

    // Update consistency checks.
    Offset += Index - Indexes.size();
    Max = std::max(Max, Index);
    IsOrdered &= Index == Indexes.size();

    Indexes.push_back(Index);
  } while (EatIfPresent(lltok::comma));

  if (ParseToken(lltok::rbrace, "expected '}' here"))
    return true;

  if (Indexes.size() < 2)
    return Error(Loc, "expected >= 2 uselistorder indexes");
  if (Offset != 0 || Max >= Indexes.size())
    return Error(Loc, "expected distinct uselistorder indexes in range [0, size)");
  if (IsOrdered)
    return Error(Loc, "expected uselistorder indexes to change the order");

  return false;
}

/// ParseUseListOrder
///   ::= 'uselistorder' Type Value ',' UseListOrderIndexes
bool LLParser::ParseUseListOrder(PerFunctionState *PFS) {
  SMLoc Loc = Lex.getLoc();
  if (ParseToken(lltok::kw_uselistorder, "expected uselistorder directive"))
    return true;

  Value *V;
  SmallVector<unsigned, 16> Indexes;
  if (ParseTypeAndValue(V, PFS) ||
      ParseToken(lltok::comma, "expected comma in uselistorder directive") ||
      ParseUseListOrderIndexes(Indexes))
    return true;

  return sortUseListOrder(V, Indexes, Loc);
}

/// ParseUseListOrderBB
///   ::= 'uselistorder_bb' @foo ',' %bar ',' UseListOrderIndexes
bool LLParser::ParseUseListOrderBB() {
  assert(Lex.getKind() == lltok::kw_uselistorder_bb);
  SMLoc Loc = Lex.getLoc();
  Lex.Lex();

  ValID Fn, Label;
  SmallVector<unsigned, 16> Indexes;
  if (ParseValID(Fn) ||
      ParseToken(lltok::comma, "expected comma in uselistorder_bb directive") ||
      ParseValID(Label) ||
      ParseToken(lltok::comma, "expected comma in uselistorder_bb directive") ||
      ParseUseListOrderIndexes(Indexes))
    return true;

  // Check the function.
  GlobalValue *GV;
  if (Fn.Kind == ValID::t_GlobalName)
    GV = M->getNamedValue(Fn.StrVal);
  else if (Fn.Kind == ValID::t_GlobalID)
    GV = Fn.UIntVal < NumberedVals.size() ? NumberedVals[Fn.UIntVal] : nullptr;
  else
    return Error(Fn.Loc, "expected function name in uselistorder_bb");
  if (!GV)
    return Error(Fn.Loc, "invalid function forward reference in uselistorder_bb");
  auto *F = dyn_cast<Function>(GV);
  if (!F)
    return Error(Fn.Loc, "expected function name in uselistorder_bb");
  if (F->isDeclaration())
    return Error(Fn.Loc, "invalid declaration in uselistorder_bb");

  // Check the basic block.
  if (Label.Kind == ValID::t_LocalID)
    return Error(Label.Loc, "invalid numeric label in uselistorder_bb");
  if (Label.Kind != ValID::t_LocalName)
    return Error(Label.Loc, "expected basic block name in uselistorder_bb");
  Value *V = F->getValueSymbolTable().lookup(Label.StrVal);
  if (!V)
    return Error(Label.Loc, "invalid basic block in uselistorder_bb");
  if (!isa<BasicBlock>(V))
    return Error(Label.Loc, "expected basic block in uselistorder_bb");

  return sortUseListOrder(V, Indexes, Loc);
}
