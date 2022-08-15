//===- lib/Linker/LinkModules.cpp - Module Linker Implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVM module linker.
//
//===----------------------------------------------------------------------===//

#include "llvm/Linker/Linker.h"
#include "llvm-c/Linker.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/TypeFinder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <cctype>
#include <tuple>
using namespace llvm;


//===----------------------------------------------------------------------===//
// TypeMap implementation.
//===----------------------------------------------------------------------===//

namespace {
class TypeMapTy : public ValueMapTypeRemapper {
  /// This is a mapping from a source type to a destination type to use.
  DenseMap<Type*, Type*> MappedTypes;

  /// When checking to see if two subgraphs are isomorphic, we speculatively
  /// add types to MappedTypes, but keep track of them here in case we need to
  /// roll back.
  SmallVector<Type*, 16> SpeculativeTypes;

  SmallVector<StructType*, 16> SpeculativeDstOpaqueTypes;

  /// This is a list of non-opaque structs in the source module that are mapped
  /// to an opaque struct in the destination module.
  SmallVector<StructType*, 16> SrcDefinitionsToResolve;

  /// This is the set of opaque types in the destination modules who are
  /// getting a body from the source module.
  SmallPtrSet<StructType*, 16> DstResolvedOpaqueTypes;

public:
  TypeMapTy(Linker::IdentifiedStructTypeSet &DstStructTypesSet)
      : DstStructTypesSet(DstStructTypesSet) {}

  Linker::IdentifiedStructTypeSet &DstStructTypesSet;
  /// Indicate that the specified type in the destination module is conceptually
  /// equivalent to the specified type in the source module.
  void addTypeMapping(Type *DstTy, Type *SrcTy);

  /// Produce a body for an opaque type in the dest module from a type
  /// definition in the source module.
  void linkDefinedTypeBodies();

  /// Return the mapped type to use for the specified input type from the
  /// source module.
  Type *get(Type *SrcTy);
  Type *get(Type *SrcTy, SmallPtrSet<StructType *, 8> &Visited);

  void finishType(StructType *DTy, StructType *STy, ArrayRef<Type *> ETypes);

  FunctionType *get(FunctionType *T) {
    return cast<FunctionType>(get((Type *)T));
  }

  /// Dump out the type map for debugging purposes.
  void dump() const {
    for (auto &Pair : MappedTypes) {
      dbgs() << "TypeMap: ";
      Pair.first->print(dbgs());
      dbgs() << " => ";
      Pair.second->print(dbgs());
      dbgs() << '\n';
    }
  }

private:
  Type *remapType(Type *SrcTy) override { return get(SrcTy); }

  bool areTypesIsomorphic(Type *DstTy, Type *SrcTy);
};
}

void TypeMapTy::addTypeMapping(Type *DstTy, Type *SrcTy) {
  assert(SpeculativeTypes.empty());
  assert(SpeculativeDstOpaqueTypes.empty());

  // Check to see if these types are recursively isomorphic and establish a
  // mapping between them if so.
  if (!areTypesIsomorphic(DstTy, SrcTy)) {
    // Oops, they aren't isomorphic.  Just discard this request by rolling out
    // any speculative mappings we've established.
    for (Type *Ty : SpeculativeTypes)
      MappedTypes.erase(Ty);

    SrcDefinitionsToResolve.resize(SrcDefinitionsToResolve.size() -
                                   SpeculativeDstOpaqueTypes.size());
    for (StructType *Ty : SpeculativeDstOpaqueTypes)
      DstResolvedOpaqueTypes.erase(Ty);
  } else {
    for (Type *Ty : SpeculativeTypes)
      if (auto *STy = dyn_cast<StructType>(Ty))
        if (STy->hasName())
          STy->setName("");
  }
  SpeculativeTypes.clear();
  SpeculativeDstOpaqueTypes.clear();
}

/// Recursively walk this pair of types, returning true if they are isomorphic,
/// false if they are not.
bool TypeMapTy::areTypesIsomorphic(Type *DstTy, Type *SrcTy) {
  // Two types with differing kinds are clearly not isomorphic.
  if (DstTy->getTypeID() != SrcTy->getTypeID())
    return false;

  // If we have an entry in the MappedTypes table, then we have our answer.
  Type *&Entry = MappedTypes[SrcTy];
  if (Entry)
    return Entry == DstTy;

  // Two identical types are clearly isomorphic.  Remember this
  // non-speculatively.
  if (DstTy == SrcTy) {
    Entry = DstTy;
    return true;
  }

  // Okay, we have two types with identical kinds that we haven't seen before.

  // If this is an opaque struct type, special case it.
  if (StructType *SSTy = dyn_cast<StructType>(SrcTy)) {
    // Mapping an opaque type to any struct, just keep the dest struct.
    if (SSTy->isOpaque()) {
      Entry = DstTy;
      SpeculativeTypes.push_back(SrcTy);
      return true;
    }

    // Mapping a non-opaque source type to an opaque dest.  If this is the first
    // type that we're mapping onto this destination type then we succeed.  Keep
    // the dest, but fill it in later. If this is the second (different) type
    // that we're trying to map onto the same opaque type then we fail.
    if (cast<StructType>(DstTy)->isOpaque()) {
      // We can only map one source type onto the opaque destination type.
      if (!DstResolvedOpaqueTypes.insert(cast<StructType>(DstTy)).second)
        return false;
      SrcDefinitionsToResolve.push_back(SSTy);
      SpeculativeTypes.push_back(SrcTy);
      SpeculativeDstOpaqueTypes.push_back(cast<StructType>(DstTy));
      Entry = DstTy;
      return true;
    }
  }

  // If the number of subtypes disagree between the two types, then we fail.
  if (SrcTy->getNumContainedTypes() != DstTy->getNumContainedTypes())
    return false;

  // Fail if any of the extra properties (e.g. array size) of the type disagree.
  if (isa<IntegerType>(DstTy))
    return false;  // bitwidth disagrees.
  if (PointerType *PT = dyn_cast<PointerType>(DstTy)) {
    if (PT->getAddressSpace() != cast<PointerType>(SrcTy)->getAddressSpace())
      return false;

  } else if (FunctionType *FT = dyn_cast<FunctionType>(DstTy)) {
    if (FT->isVarArg() != cast<FunctionType>(SrcTy)->isVarArg())
      return false;
  } else if (StructType *DSTy = dyn_cast<StructType>(DstTy)) {
    StructType *SSTy = cast<StructType>(SrcTy);
    if (DSTy->isLiteral() != SSTy->isLiteral() ||
        DSTy->isPacked() != SSTy->isPacked())
      return false;
  } else if (ArrayType *DATy = dyn_cast<ArrayType>(DstTy)) {
    if (DATy->getNumElements() != cast<ArrayType>(SrcTy)->getNumElements())
      return false;
  } else if (VectorType *DVTy = dyn_cast<VectorType>(DstTy)) {
    if (DVTy->getNumElements() != cast<VectorType>(SrcTy)->getNumElements())
      return false;
  }

  // Otherwise, we speculate that these two types will line up and recursively
  // check the subelements.
  Entry = DstTy;
  SpeculativeTypes.push_back(SrcTy);

  for (unsigned I = 0, E = SrcTy->getNumContainedTypes(); I != E; ++I)
    if (!areTypesIsomorphic(DstTy->getContainedType(I),
                            SrcTy->getContainedType(I)))
      return false;

  // If everything seems to have lined up, then everything is great.
  return true;
}

void TypeMapTy::linkDefinedTypeBodies() {
  SmallVector<Type*, 16> Elements;
  for (StructType *SrcSTy : SrcDefinitionsToResolve) {
    StructType *DstSTy = cast<StructType>(MappedTypes[SrcSTy]);
    assert(DstSTy->isOpaque());

    // Map the body of the source type over to a new body for the dest type.
    Elements.resize(SrcSTy->getNumElements());
    for (unsigned I = 0, E = Elements.size(); I != E; ++I)
      Elements[I] = get(SrcSTy->getElementType(I));

    DstSTy->setBody(Elements, SrcSTy->isPacked());
    DstStructTypesSet.switchToNonOpaque(DstSTy);
  }
  SrcDefinitionsToResolve.clear();
  DstResolvedOpaqueTypes.clear();
}

void TypeMapTy::finishType(StructType *DTy, StructType *STy,
                           ArrayRef<Type *> ETypes) {
  DTy->setBody(ETypes, STy->isPacked());

  // Steal STy's name.
  if (STy->hasName()) {
    SmallString<16> TmpName = STy->getName();
    STy->setName("");
    DTy->setName(TmpName);
  }

  DstStructTypesSet.addNonOpaque(DTy);
}

Type *TypeMapTy::get(Type *Ty) {
  SmallPtrSet<StructType *, 8> Visited;
  return get(Ty, Visited);
}

Type *TypeMapTy::get(Type *Ty, SmallPtrSet<StructType *, 8> &Visited) {
  // If we already have an entry for this type, return it.
  Type **Entry = &MappedTypes[Ty];
  if (*Entry)
    return *Entry;

  // These are types that LLVM itself will unique.
  bool IsUniqued = !isa<StructType>(Ty) || cast<StructType>(Ty)->isLiteral();

#ifndef NDEBUG
  if (!IsUniqued) {
    for (auto &Pair : MappedTypes) {
      assert(!(Pair.first != Ty && Pair.second == Ty) &&
             "mapping to a source type");
    }
  }
#endif

  if (!IsUniqued && !Visited.insert(cast<StructType>(Ty)).second) {
    StructType *DTy = StructType::create(Ty->getContext());
    return *Entry = DTy;
  }

  // If this is not a recursive type, then just map all of the elements and
  // then rebuild the type from inside out.
  SmallVector<Type *, 4> ElementTypes;

  // If there are no element types to map, then the type is itself.  This is
  // true for the anonymous {} struct, things like 'float', integers, etc.
  if (Ty->getNumContainedTypes() == 0 && IsUniqued)
    return *Entry = Ty;

  // Remap all of the elements, keeping track of whether any of them change.
  bool AnyChange = false;
  ElementTypes.resize(Ty->getNumContainedTypes());
  for (unsigned I = 0, E = Ty->getNumContainedTypes(); I != E; ++I) {
    ElementTypes[I] = get(Ty->getContainedType(I), Visited);
    AnyChange |= ElementTypes[I] != Ty->getContainedType(I);
  }

  // If we found our type while recursively processing stuff, just use it.
  Entry = &MappedTypes[Ty];
  if (*Entry) {
    if (auto *DTy = dyn_cast<StructType>(*Entry)) {
      if (DTy->isOpaque()) {
        auto *STy = cast<StructType>(Ty);
        finishType(DTy, STy, ElementTypes);
      }
    }
    return *Entry;
  }

  // If all of the element types mapped directly over and the type is not
  // a nomed struct, then the type is usable as-is.
  if (!AnyChange && IsUniqued)
    return *Entry = Ty;

  // Otherwise, rebuild a modified type.
  switch (Ty->getTypeID()) {
  default:
    llvm_unreachable("unknown derived type to remap");
  case Type::ArrayTyID:
    return *Entry = ArrayType::get(ElementTypes[0],
                                   cast<ArrayType>(Ty)->getNumElements());
  case Type::VectorTyID:
    return *Entry = VectorType::get(ElementTypes[0],
                                    cast<VectorType>(Ty)->getNumElements());
  case Type::PointerTyID:
    return *Entry = PointerType::get(ElementTypes[0],
                                     cast<PointerType>(Ty)->getAddressSpace());
  case Type::FunctionTyID:
    return *Entry = FunctionType::get(ElementTypes[0],
                                      makeArrayRef(ElementTypes).slice(1),
                                      cast<FunctionType>(Ty)->isVarArg());
  case Type::StructTyID: {
    auto *STy = cast<StructType>(Ty);
    bool IsPacked = STy->isPacked();
    if (IsUniqued)
      return *Entry = StructType::get(Ty->getContext(), ElementTypes, IsPacked);

    // If the type is opaque, we can just use it directly.
    if (STy->isOpaque()) {
      DstStructTypesSet.addOpaque(STy);
      return *Entry = Ty;
    }

    if (StructType *OldT =
            DstStructTypesSet.findNonOpaque(ElementTypes, IsPacked)) {
      STy->setName("");
      return *Entry = OldT;
    }

    if (!AnyChange) {
      DstStructTypesSet.addNonOpaque(STy);
      return *Entry = Ty;
    }

    StructType *DTy = StructType::create(Ty->getContext());
    finishType(DTy, STy, ElementTypes);
    return *Entry = DTy;
  }
  }
}

//===----------------------------------------------------------------------===//
// ModuleLinker implementation.
//===----------------------------------------------------------------------===//

namespace {
class ModuleLinker;

/// Creates prototypes for functions that are lazily linked on the fly. This
/// speeds up linking for modules with many/ lazily linked functions of which
/// few get used.
class ValueMaterializerTy : public ValueMaterializer {
  TypeMapTy &TypeMap;
  Module *DstM;
  std::vector<GlobalValue *> &LazilyLinkGlobalValues;

public:
  ValueMaterializerTy(TypeMapTy &TypeMap, Module *DstM,
                      std::vector<GlobalValue *> &LazilyLinkGlobalValues)
      : ValueMaterializer(), TypeMap(TypeMap), DstM(DstM),
        LazilyLinkGlobalValues(LazilyLinkGlobalValues) {}

  Value *materializeValueFor(Value *V) override;
};

class LinkDiagnosticInfo : public DiagnosticInfo {
  const Twine &Msg;

public:
  LinkDiagnosticInfo(DiagnosticSeverity Severity, const Twine &Msg);
  void print(DiagnosticPrinter &DP) const override;
};
LinkDiagnosticInfo::LinkDiagnosticInfo(DiagnosticSeverity Severity,
                                       const Twine &Msg)
    : DiagnosticInfo(DK_Linker, Severity), Msg(Msg) {}
void LinkDiagnosticInfo::print(DiagnosticPrinter &DP) const { DP << Msg; }

/// This is an implementation class for the LinkModules function, which is the
/// entrypoint for this file.
class ModuleLinker {
  Module *DstM, *SrcM;

  TypeMapTy TypeMap;
  ValueMaterializerTy ValMaterializer;

  /// Mapping of values from what they used to be in Src, to what they are now
  /// in DstM.  ValueToValueMapTy is a ValueMap, which involves some overhead
  /// due to the use of Value handles which the Linker doesn't actually need,
  /// but this allows us to reuse the ValueMapper code.
  ValueToValueMapTy ValueMap;

  struct AppendingVarInfo {
    GlobalVariable *NewGV;   // New aggregate global in dest module.
    const Constant *DstInit; // Old initializer from dest module.
    const Constant *SrcInit; // Old initializer from src module.
  };

  std::vector<AppendingVarInfo> AppendingVars;

  // Set of items not to link in from source.
  SmallPtrSet<const Value *, 16> DoNotLinkFromSource;

  // Vector of GlobalValues to lazily link in.
  std::vector<GlobalValue *> LazilyLinkGlobalValues;

  /// Functions that have replaced other functions.
  SmallPtrSet<const Function *, 16> OverridingFunctions;

  DiagnosticHandlerFunction DiagnosticHandler;

  /// For symbol clashes, prefer those from Src.
  bool OverrideFromSrc;

public:
  ModuleLinker(Module *dstM, Linker::IdentifiedStructTypeSet &Set, Module *srcM,
               DiagnosticHandlerFunction DiagnosticHandler,
               bool OverrideFromSrc)
      : DstM(dstM), SrcM(srcM), TypeMap(Set),
        ValMaterializer(TypeMap, DstM, LazilyLinkGlobalValues),
        DiagnosticHandler(DiagnosticHandler), OverrideFromSrc(OverrideFromSrc) {
  }

  bool run();

private:
  bool shouldLinkFromSource(bool &LinkFromSrc, const GlobalValue &Dest,
                            const GlobalValue &Src);

  /// Helper method for setting a message and returning an error code.
  bool emitError(const Twine &Message) {
    DiagnosticHandler(LinkDiagnosticInfo(DS_Error, Message));
    return true;
  }

  void emitWarning(const Twine &Message) {
    DiagnosticHandler(LinkDiagnosticInfo(DS_Warning, Message));
  }

  bool getComdatLeader(Module *M, StringRef ComdatName,
                       const GlobalVariable *&GVar);
  bool computeResultingSelectionKind(StringRef ComdatName,
                                     Comdat::SelectionKind Src,
                                     Comdat::SelectionKind Dst,
                                     Comdat::SelectionKind &Result,
                                     bool &LinkFromSrc);
  std::map<const Comdat *, std::pair<Comdat::SelectionKind, bool>>
      ComdatsChosen;
  bool getComdatResult(const Comdat *SrcC, Comdat::SelectionKind &SK,
                       bool &LinkFromSrc);

  /// Given a global in the source module, return the global in the
  /// destination module that is being linked to, if any.
  GlobalValue *getLinkedToGlobal(const GlobalValue *SrcGV) {
    // If the source has no name it can't link.  If it has local linkage,
    // there is no name match-up going on.
    if (!SrcGV->hasName() || SrcGV->hasLocalLinkage())
      return nullptr;

    // Otherwise see if we have a match in the destination module's symtab.
    GlobalValue *DGV = DstM->getNamedValue(SrcGV->getName());
    if (!DGV)
      return nullptr;

    // If we found a global with the same name in the dest module, but it has
    // internal linkage, we are really not doing any linkage here.
    if (DGV->hasLocalLinkage())
      return nullptr;

    // Otherwise, we do in fact link to the destination global.
    return DGV;
  }

  void computeTypeMapping();

  void upgradeMismatchedGlobalArray(StringRef Name);
  void upgradeMismatchedGlobals();

  bool linkAppendingVarProto(GlobalVariable *DstGV,
                             const GlobalVariable *SrcGV);

  bool linkGlobalValueProto(GlobalValue *GV);
  bool linkModuleFlagsMetadata();

  void linkAppendingVarInit(const AppendingVarInfo &AVI);

  void linkGlobalInit(GlobalVariable &Dst, GlobalVariable &Src);
  bool linkFunctionBody(Function &Dst, Function &Src);
  void linkAliasBody(GlobalAlias &Dst, GlobalAlias &Src);
  bool linkGlobalValueBody(GlobalValue &Src);

  void linkNamedMDNodes();
  void stripReplacedSubprograms();
};
}

/// The LLVM SymbolTable class autorenames globals that conflict in the symbol
/// table. This is good for all clients except for us. Go through the trouble
/// to force this back.
static void forceRenaming(GlobalValue *GV, StringRef Name) {
  // If the global doesn't force its name or if it already has the right name,
  // there is nothing for us to do.
  if (GV->hasLocalLinkage() || GV->getName() == Name)
    return;

  Module *M = GV->getParent();

  // If there is a conflict, rename the conflict.
  if (GlobalValue *ConflictGV = M->getNamedValue(Name)) {
    GV->takeName(ConflictGV);
    ConflictGV->setName(Name);    // This will cause ConflictGV to get renamed
    assert(ConflictGV->getName() != Name && "forceRenaming didn't work");
  } else {
    GV->setName(Name);              // Force the name back
  }
}

/// copy additional attributes (those not needed to construct a GlobalValue)
/// from the SrcGV to the DestGV.
static void copyGVAttributes(GlobalValue *DestGV, const GlobalValue *SrcGV) {
  DestGV->copyAttributesFrom(SrcGV);
  forceRenaming(DestGV, SrcGV->getName());
}

static bool isLessConstraining(GlobalValue::VisibilityTypes a,
                               GlobalValue::VisibilityTypes b) {
  if (a == GlobalValue::HiddenVisibility)
    return false;
  if (b == GlobalValue::HiddenVisibility)
    return true;
  if (a == GlobalValue::ProtectedVisibility)
    return false;
  if (b == GlobalValue::ProtectedVisibility)
    return true;
  return false;
}

/// Loop through the global variables in the src module and merge them into the
/// dest module.
static GlobalVariable *copyGlobalVariableProto(TypeMapTy &TypeMap, Module &DstM,
                                               const GlobalVariable *SGVar) {
  // No linking to be performed or linking from the source: simply create an
  // identical version of the symbol over in the dest module... the
  // initializer will be filled in later by LinkGlobalInits.
  GlobalVariable *NewDGV = new GlobalVariable(
      DstM, TypeMap.get(SGVar->getType()->getElementType()),
      SGVar->isConstant(), SGVar->getLinkage(), /*init*/ nullptr,
      SGVar->getName(), /*insertbefore*/ nullptr, SGVar->getThreadLocalMode(),
      SGVar->getType()->getAddressSpace());

  return NewDGV;
}

/// Link the function in the source module into the destination module if
/// needed, setting up mapping information.
static Function *copyFunctionProto(TypeMapTy &TypeMap, Module &DstM,
                                   const Function *SF) {
  // If there is no linkage to be performed or we are linking from the source,
  // bring SF over.
  return Function::Create(TypeMap.get(SF->getFunctionType()), SF->getLinkage(),
                          SF->getName(), &DstM);
}

/// Set up prototypes for any aliases that come over from the source module.
static GlobalAlias *copyGlobalAliasProto(TypeMapTy &TypeMap, Module &DstM,
                                         const GlobalAlias *SGA) {
  // If there is no linkage to be performed or we're linking from the source,
  // bring over SGA.
  auto *PTy = cast<PointerType>(TypeMap.get(SGA->getType()));
  return GlobalAlias::create(PTy, SGA->getLinkage(), SGA->getName(), &DstM);
}

static GlobalValue *copyGlobalValueProto(TypeMapTy &TypeMap, Module &DstM,
                                         const GlobalValue *SGV) {
  GlobalValue *NewGV;
  if (auto *SGVar = dyn_cast<GlobalVariable>(SGV))
    NewGV = copyGlobalVariableProto(TypeMap, DstM, SGVar);
  else if (auto *SF = dyn_cast<Function>(SGV))
    NewGV = copyFunctionProto(TypeMap, DstM, SF);
  else
    NewGV = copyGlobalAliasProto(TypeMap, DstM, cast<GlobalAlias>(SGV));
  copyGVAttributes(NewGV, SGV);
  return NewGV;
}

Value *ValueMaterializerTy::materializeValueFor(Value *V) {
  auto *SGV = dyn_cast<GlobalValue>(V);
  if (!SGV)
    return nullptr;

  GlobalValue *DGV = copyGlobalValueProto(TypeMap, *DstM, SGV);

  if (Comdat *SC = SGV->getComdat()) {
    if (auto *DGO = dyn_cast<GlobalObject>(DGV)) {
      Comdat *DC = DstM->getOrInsertComdat(SC->getName());
      DGO->setComdat(DC);
    }
  }

  LazilyLinkGlobalValues.push_back(SGV);
  return DGV;
}

bool ModuleLinker::getComdatLeader(Module *M, StringRef ComdatName,
                                   const GlobalVariable *&GVar) {
  const GlobalValue *GVal = M->getNamedValue(ComdatName);
  if (const auto *GA = dyn_cast_or_null<GlobalAlias>(GVal)) {
    GVal = GA->getBaseObject();
    if (!GVal)
      // We cannot resolve the size of the aliasee yet.
      return emitError("Linking COMDATs named '" + ComdatName +
                       "': COMDAT key involves incomputable alias size.");
  }

  GVar = dyn_cast_or_null<GlobalVariable>(GVal);
  if (!GVar)
    return emitError(
        "Linking COMDATs named '" + ComdatName +
        "': GlobalVariable required for data dependent selection!");

  return false;
}

bool ModuleLinker::computeResultingSelectionKind(StringRef ComdatName,
                                                 Comdat::SelectionKind Src,
                                                 Comdat::SelectionKind Dst,
                                                 Comdat::SelectionKind &Result,
                                                 bool &LinkFromSrc) {
  // The ability to mix Comdat::SelectionKind::Any with
  // Comdat::SelectionKind::Largest is a behavior that comes from COFF.
  bool DstAnyOrLargest = Dst == Comdat::SelectionKind::Any ||
                         Dst == Comdat::SelectionKind::Largest;
  bool SrcAnyOrLargest = Src == Comdat::SelectionKind::Any ||
                         Src == Comdat::SelectionKind::Largest;
  if (DstAnyOrLargest && SrcAnyOrLargest) {
    if (Dst == Comdat::SelectionKind::Largest ||
        Src == Comdat::SelectionKind::Largest)
      Result = Comdat::SelectionKind::Largest;
    else
      Result = Comdat::SelectionKind::Any;
  } else if (Src == Dst) {
    Result = Dst;
  } else {
    return emitError("Linking COMDATs named '" + ComdatName +
                     "': invalid selection kinds!");
  }

  switch (Result) {
  case Comdat::SelectionKind::Any:
    // Go with Dst.
    LinkFromSrc = false;
    break;
  case Comdat::SelectionKind::NoDuplicates:
    return emitError("Linking COMDATs named '" + ComdatName +
                     "': noduplicates has been violated!");
  case Comdat::SelectionKind::ExactMatch:
  case Comdat::SelectionKind::Largest:
  case Comdat::SelectionKind::SameSize: {
    const GlobalVariable *DstGV;
    const GlobalVariable *SrcGV;
    if (getComdatLeader(DstM, ComdatName, DstGV) ||
        getComdatLeader(SrcM, ComdatName, SrcGV))
      return true;

    const DataLayout &DstDL = DstM->getDataLayout();
    const DataLayout &SrcDL = SrcM->getDataLayout();
    uint64_t DstSize =
        DstDL.getTypeAllocSize(DstGV->getType()->getPointerElementType());
    uint64_t SrcSize =
        SrcDL.getTypeAllocSize(SrcGV->getType()->getPointerElementType());
    if (Result == Comdat::SelectionKind::ExactMatch) {
      if (SrcGV->getInitializer() != DstGV->getInitializer())
        return emitError("Linking COMDATs named '" + ComdatName +
                         "': ExactMatch violated!");
      LinkFromSrc = false;
    } else if (Result == Comdat::SelectionKind::Largest) {
      LinkFromSrc = SrcSize > DstSize;
    } else if (Result == Comdat::SelectionKind::SameSize) {
      if (SrcSize != DstSize)
        return emitError("Linking COMDATs named '" + ComdatName +
                         "': SameSize violated!");
      LinkFromSrc = false;
    } else {
      llvm_unreachable("unknown selection kind");
    }
    break;
  }
  }

  return false;
}

bool ModuleLinker::getComdatResult(const Comdat *SrcC,
                                   Comdat::SelectionKind &Result,
                                   bool &LinkFromSrc) {
  Comdat::SelectionKind SSK = SrcC->getSelectionKind();
  StringRef ComdatName = SrcC->getName();
  Module::ComdatSymTabType &ComdatSymTab = DstM->getComdatSymbolTable();
  Module::ComdatSymTabType::iterator DstCI = ComdatSymTab.find(ComdatName);

  if (DstCI == ComdatSymTab.end()) {
    // Use the comdat if it is only available in one of the modules.
    LinkFromSrc = true;
    Result = SSK;
    return false;
  }

  const Comdat *DstC = &DstCI->second;
  Comdat::SelectionKind DSK = DstC->getSelectionKind();
  return computeResultingSelectionKind(ComdatName, SSK, DSK, Result,
                                       LinkFromSrc);
}

bool ModuleLinker::shouldLinkFromSource(bool &LinkFromSrc,
                                        const GlobalValue &Dest,
                                        const GlobalValue &Src) {
  // Should we unconditionally use the Src?
  if (OverrideFromSrc) {
    LinkFromSrc = true;
    return false;
  }

  // We always have to add Src if it has appending linkage.
  if (Src.hasAppendingLinkage()) {
    LinkFromSrc = true;
    return false;
  }

  bool SrcIsDeclaration = Src.isDeclarationForLinker();
  bool DestIsDeclaration = Dest.isDeclarationForLinker();

  if (SrcIsDeclaration) {
    // If Src is external or if both Src & Dest are external..  Just link the
    // external globals, we aren't adding anything.
    if (Src.hasDLLImportStorageClass()) {
      // If one of GVs is marked as DLLImport, result should be dllimport'ed.
      LinkFromSrc = DestIsDeclaration;
      return false;
    }
    // If the Dest is weak, use the source linkage.
    LinkFromSrc = Dest.hasExternalWeakLinkage();
    return false;
  }

  if (DestIsDeclaration) {
    // If Dest is external but Src is not:
    LinkFromSrc = true;
    return false;
  }

  if (Src.hasCommonLinkage()) {
    if (Dest.hasLinkOnceLinkage() || Dest.hasWeakLinkage()) {
      LinkFromSrc = true;
      return false;
    }

    if (!Dest.hasCommonLinkage()) {
      LinkFromSrc = false;
      return false;
    }

    const DataLayout &DL = Dest.getParent()->getDataLayout();
    uint64_t DestSize = DL.getTypeAllocSize(Dest.getType()->getElementType());
    uint64_t SrcSize = DL.getTypeAllocSize(Src.getType()->getElementType());
    LinkFromSrc = SrcSize > DestSize;
    return false;
  }

  if (Src.isWeakForLinker()) {
    assert(!Dest.hasExternalWeakLinkage());
    assert(!Dest.hasAvailableExternallyLinkage());

    if (Dest.hasLinkOnceLinkage() && Src.hasWeakLinkage()) {
      LinkFromSrc = true;
      return false;
    }

    LinkFromSrc = false;
    return false;
  }

  if (Dest.isWeakForLinker()) {
    assert(Src.hasExternalLinkage());
    LinkFromSrc = true;
    return false;
  }

  assert(!Src.hasExternalWeakLinkage());
  assert(!Dest.hasExternalWeakLinkage());
  assert(Dest.hasExternalLinkage() && Src.hasExternalLinkage() &&
         "Unexpected linkage type!");
  return emitError("Linking globals named '" + Src.getName() +
                   "': symbol multiply defined!");
}

/// Loop over all of the linked values to compute type mappings.  For example,
/// if we link "extern Foo *x" and "Foo *x = NULL", then we have two struct
/// types 'Foo' but one got renamed when the module was loaded into the same
/// LLVMContext.
void ModuleLinker::computeTypeMapping() {
  for (GlobalValue &SGV : SrcM->globals()) {
    GlobalValue *DGV = getLinkedToGlobal(&SGV);
    if (!DGV)
      continue;

    if (!DGV->hasAppendingLinkage() || !SGV.hasAppendingLinkage()) {
      TypeMap.addTypeMapping(DGV->getType(), SGV.getType());
      continue;
    }

    // Unify the element type of appending arrays.
    ArrayType *DAT = cast<ArrayType>(DGV->getType()->getElementType());
    ArrayType *SAT = cast<ArrayType>(SGV.getType()->getElementType());
    TypeMap.addTypeMapping(DAT->getElementType(), SAT->getElementType());
  }

  for (GlobalValue &SGV : *SrcM) {
    if (GlobalValue *DGV = getLinkedToGlobal(&SGV))
      TypeMap.addTypeMapping(DGV->getType(), SGV.getType());
  }

  for (GlobalValue &SGV : SrcM->aliases()) {
    if (GlobalValue *DGV = getLinkedToGlobal(&SGV))
      TypeMap.addTypeMapping(DGV->getType(), SGV.getType());
  }

  // Incorporate types by name, scanning all the types in the source module.
  // At this point, the destination module may have a type "%foo = { i32 }" for
  // example.  When the source module got loaded into the same LLVMContext, if
  // it had the same type, it would have been renamed to "%foo.42 = { i32 }".
  std::vector<StructType *> Types = SrcM->getIdentifiedStructTypes();
  for (StructType *ST : Types) {
    if (!ST->hasName())
      continue;

    // Check to see if there is a dot in the name followed by a digit.
    size_t DotPos = ST->getName().rfind('.');
    if (DotPos == 0 || DotPos == StringRef::npos ||
        ST->getName().back() == '.' ||
        !isdigit(static_cast<unsigned char>(ST->getName()[DotPos + 1])))
      continue;

    // Check to see if the destination module has a struct with the prefix name.
    StructType *DST = DstM->getTypeByName(ST->getName().substr(0, DotPos));
    if (!DST)
      continue;

    // Don't use it if this actually came from the source module. They're in
    // the same LLVMContext after all. Also don't use it unless the type is
    // actually used in the destination module. This can happen in situations
    // like this:
    //
    //      Module A                         Module B
    //      --------                         --------
    //   %Z = type { %A }                %B = type { %C.1 }
    //   %A = type { %B.1, [7 x i8] }    %C.1 = type { i8* }
    //   %B.1 = type { %C }              %A.2 = type { %B.3, [5 x i8] }
    //   %C = type { i8* }               %B.3 = type { %C.1 }
    //
    // When we link Module B with Module A, the '%B' in Module B is
    // used. However, that would then use '%C.1'. But when we process '%C.1',
    // we prefer to take the '%C' version. So we are then left with both
    // '%C.1' and '%C' being used for the same types. This leads to some
    // variables using one type and some using the other.
    if (TypeMap.DstStructTypesSet.hasType(DST))
      TypeMap.addTypeMapping(DST, ST);
  }

  // Now that we have discovered all of the type equivalences, get a body for
  // any 'opaque' types in the dest module that are now resolved.
  TypeMap.linkDefinedTypeBodies();
}

static void upgradeGlobalArray(GlobalVariable *GV) {
  ArrayType *ATy = cast<ArrayType>(GV->getType()->getElementType());
  StructType *OldTy = cast<StructType>(ATy->getElementType());
  assert(OldTy->getNumElements() == 2 && "Expected to upgrade from 2 elements");

  // Get the upgraded 3 element type.
  PointerType *VoidPtrTy = Type::getInt8Ty(GV->getContext())->getPointerTo();
  Type *Tys[3] = {OldTy->getElementType(0), OldTy->getElementType(1),
                  VoidPtrTy};
  StructType *NewTy = StructType::get(GV->getContext(), Tys, false);

  // Build new constants with a null third field filled in.
  Constant *OldInitC = GV->getInitializer();
  ConstantArray *OldInit = dyn_cast<ConstantArray>(OldInitC);
  if (!OldInit && !isa<ConstantAggregateZero>(OldInitC))
    // Invalid initializer; give up.
    return;
  std::vector<Constant *> Initializers;
  if (OldInit && OldInit->getNumOperands()) {
    Value *Null = Constant::getNullValue(VoidPtrTy);
    for (Use &U : OldInit->operands()) {
      ConstantStruct *Init = cast<ConstantStruct>(U.get());
      Initializers.push_back(ConstantStruct::get(
          NewTy, Init->getOperand(0), Init->getOperand(1), Null, nullptr));
    }
  }
  assert(Initializers.size() == ATy->getNumElements() &&
         "Failed to copy all array elements");

  // Replace the old GV with a new one.
  ATy = ArrayType::get(NewTy, Initializers.size());
  Constant *NewInit = ConstantArray::get(ATy, Initializers);
  GlobalVariable *NewGV = new GlobalVariable(
      *GV->getParent(), ATy, GV->isConstant(), GV->getLinkage(), NewInit, "",
      GV, GV->getThreadLocalMode(), GV->getType()->getAddressSpace(),
      GV->isExternallyInitialized());
  NewGV->copyAttributesFrom(GV);
  NewGV->takeName(GV);
  assert(GV->use_empty() && "program cannot use initializer list");
  GV->eraseFromParent();
}

void ModuleLinker::upgradeMismatchedGlobalArray(StringRef Name) {
  // Look for the global arrays.
  auto *DstGV = dyn_cast_or_null<GlobalVariable>(DstM->getNamedValue(Name));
  if (!DstGV)
    return;
  auto *SrcGV = dyn_cast_or_null<GlobalVariable>(SrcM->getNamedValue(Name));
  if (!SrcGV)
    return;

  // Check if the types already match.
  auto *DstTy = cast<ArrayType>(DstGV->getType()->getElementType());
  auto *SrcTy =
      cast<ArrayType>(TypeMap.get(SrcGV->getType()->getElementType()));
  if (DstTy == SrcTy)
    return;

  // Grab the element types.  We can only upgrade an array of a two-field
  // struct.  Only bother if the other one has three-fields.
  auto *DstEltTy = cast<StructType>(DstTy->getElementType());
  auto *SrcEltTy = cast<StructType>(SrcTy->getElementType());
  if (DstEltTy->getNumElements() == 2 && SrcEltTy->getNumElements() == 3) {
    upgradeGlobalArray(DstGV);
    return;
  }
  if (DstEltTy->getNumElements() == 3 && SrcEltTy->getNumElements() == 2)
    upgradeGlobalArray(SrcGV);

  // We can't upgrade any other differences.
}

void ModuleLinker::upgradeMismatchedGlobals() {
  upgradeMismatchedGlobalArray("llvm.global_ctors");
  upgradeMismatchedGlobalArray("llvm.global_dtors");
}

/// If there were any appending global variables, link them together now.
/// Return true on error.
bool ModuleLinker::linkAppendingVarProto(GlobalVariable *DstGV,
                                         const GlobalVariable *SrcGV) {

  if (!SrcGV->hasAppendingLinkage() || !DstGV->hasAppendingLinkage())
    return emitError("Linking globals named '" + SrcGV->getName() +
           "': can only link appending global with another appending global!");

  ArrayType *DstTy = cast<ArrayType>(DstGV->getType()->getElementType());
  ArrayType *SrcTy =
    cast<ArrayType>(TypeMap.get(SrcGV->getType()->getElementType()));
  Type *EltTy = DstTy->getElementType();

  // Check to see that they two arrays agree on type.
  if (EltTy != SrcTy->getElementType())
    return emitError("Appending variables with different element types!");
  if (DstGV->isConstant() != SrcGV->isConstant())
    return emitError("Appending variables linked with different const'ness!");

  if (DstGV->getAlignment() != SrcGV->getAlignment())
    return emitError(
             "Appending variables with different alignment need to be linked!");

  if (DstGV->getVisibility() != SrcGV->getVisibility())
    return emitError(
            "Appending variables with different visibility need to be linked!");

  if (DstGV->hasUnnamedAddr() != SrcGV->hasUnnamedAddr())
    return emitError(
        "Appending variables with different unnamed_addr need to be linked!");

  if (StringRef(DstGV->getSection()) != SrcGV->getSection())
    return emitError(
          "Appending variables with different section name need to be linked!");

  uint64_t NewSize = DstTy->getNumElements() + SrcTy->getNumElements();
  ArrayType *NewType = ArrayType::get(EltTy, NewSize);

  // Create the new global variable.
  GlobalVariable *NG =
    new GlobalVariable(*DstGV->getParent(), NewType, SrcGV->isConstant(),
                       DstGV->getLinkage(), /*init*/nullptr, /*name*/"", DstGV,
                       DstGV->getThreadLocalMode(),
                       DstGV->getType()->getAddressSpace());

  // Propagate alignment, visibility and section info.
  copyGVAttributes(NG, DstGV);

  AppendingVarInfo AVI;
  AVI.NewGV = NG;
  AVI.DstInit = DstGV->getInitializer();
  AVI.SrcInit = SrcGV->getInitializer();
  AppendingVars.push_back(AVI);

  // Replace any uses of the two global variables with uses of the new
  // global.
  ValueMap[SrcGV] = ConstantExpr::getBitCast(NG, TypeMap.get(SrcGV->getType()));

  DstGV->replaceAllUsesWith(ConstantExpr::getBitCast(NG, DstGV->getType()));
  DstGV->eraseFromParent();

  // Track the source variable so we don't try to link it.
  DoNotLinkFromSource.insert(SrcGV);

  return false;
}

bool ModuleLinker::linkGlobalValueProto(GlobalValue *SGV) {
  GlobalValue *DGV = getLinkedToGlobal(SGV);

  // Handle the ultra special appending linkage case first.
  if (DGV && DGV->hasAppendingLinkage())
    return linkAppendingVarProto(cast<GlobalVariable>(DGV),
                                 cast<GlobalVariable>(SGV));

  bool LinkFromSrc = true;
  Comdat *C = nullptr;
  GlobalValue::VisibilityTypes Visibility = SGV->getVisibility();
  bool HasUnnamedAddr = SGV->hasUnnamedAddr();

  if (const Comdat *SC = SGV->getComdat()) {
    Comdat::SelectionKind SK;
    std::tie(SK, LinkFromSrc) = ComdatsChosen[SC];
    C = DstM->getOrInsertComdat(SC->getName());
    C->setSelectionKind(SK);
  } else if (DGV) {
    if (shouldLinkFromSource(LinkFromSrc, *DGV, *SGV))
      return true;
  }

  if (!LinkFromSrc) {
    // Track the source global so that we don't attempt to copy it over when
    // processing global initializers.
    DoNotLinkFromSource.insert(SGV);

    if (DGV)
      // Make sure to remember this mapping.
      ValueMap[SGV] =
          ConstantExpr::getBitCast(DGV, TypeMap.get(SGV->getType()));
  }

  if (DGV) {
    Visibility = isLessConstraining(Visibility, DGV->getVisibility())
                     ? DGV->getVisibility()
                     : Visibility;
    HasUnnamedAddr = HasUnnamedAddr && DGV->hasUnnamedAddr();
  }

  if (!LinkFromSrc && !DGV)
    return false;

  GlobalValue *NewGV;
  if (!LinkFromSrc) {
    NewGV = DGV;
  } else {
    // If the GV is to be lazily linked, don't create it just yet.
    // The ValueMaterializerTy will deal with creating it if it's used.
    if (!DGV && !OverrideFromSrc &&
        (SGV->hasLocalLinkage() || SGV->hasLinkOnceLinkage() ||
         SGV->hasAvailableExternallyLinkage())) {
      DoNotLinkFromSource.insert(SGV);
      return false;
    }

    NewGV = copyGlobalValueProto(TypeMap, *DstM, SGV);

    if (DGV && isa<Function>(DGV))
      if (auto *NewF = dyn_cast<Function>(NewGV))
        OverridingFunctions.insert(NewF);
  }

  NewGV->setUnnamedAddr(HasUnnamedAddr);
  NewGV->setVisibility(Visibility);

  if (auto *NewGO = dyn_cast<GlobalObject>(NewGV)) {
    if (C)
      NewGO->setComdat(C);

    if (DGV && DGV->hasCommonLinkage() && SGV->hasCommonLinkage())
      NewGO->setAlignment(std::max(DGV->getAlignment(), SGV->getAlignment()));
  }

  if (auto *NewGVar = dyn_cast<GlobalVariable>(NewGV)) {
    auto *DGVar = dyn_cast_or_null<GlobalVariable>(DGV);
    auto *SGVar = dyn_cast<GlobalVariable>(SGV);
    if (DGVar && SGVar && DGVar->isDeclaration() && SGVar->isDeclaration() &&
        (!DGVar->isConstant() || !SGVar->isConstant()))
      NewGVar->setConstant(false);
  }

  // Make sure to remember this mapping.
  if (NewGV != DGV) {
    if (DGV) {
      DGV->replaceAllUsesWith(ConstantExpr::getBitCast(NewGV, DGV->getType()));
      DGV->eraseFromParent();
    }
    ValueMap[SGV] = NewGV;
  }

  return false;
}

static void getArrayElements(const Constant *C,
                             SmallVectorImpl<Constant *> &Dest) {
  unsigned NumElements = cast<ArrayType>(C->getType())->getNumElements();

  for (unsigned i = 0; i != NumElements; ++i)
    Dest.push_back(C->getAggregateElement(i));
}

void ModuleLinker::linkAppendingVarInit(const AppendingVarInfo &AVI) {
  // Merge the initializer.
  SmallVector<Constant *, 16> DstElements;
  getArrayElements(AVI.DstInit, DstElements);

  SmallVector<Constant *, 16> SrcElements;
  getArrayElements(AVI.SrcInit, SrcElements);

  ArrayType *NewType = cast<ArrayType>(AVI.NewGV->getType()->getElementType());

  StringRef Name = AVI.NewGV->getName();
  bool IsNewStructor =
      (Name == "llvm.global_ctors" || Name == "llvm.global_dtors") &&
      cast<StructType>(NewType->getElementType())->getNumElements() == 3;

  for (auto *V : SrcElements) {
    if (IsNewStructor) {
      Constant *Key = V->getAggregateElement(2);
      if (DoNotLinkFromSource.count(Key))
        continue;
    }
    DstElements.push_back(
        MapValue(V, ValueMap, RF_None, &TypeMap, &ValMaterializer));
  }
  if (IsNewStructor) {
    NewType = ArrayType::get(NewType->getElementType(), DstElements.size());
    AVI.NewGV->mutateType(PointerType::get(NewType, 0));
  }

  AVI.NewGV->setInitializer(ConstantArray::get(NewType, DstElements));
}

/// Update the initializers in the Dest module now that all globals that may be
/// referenced are in Dest.
void ModuleLinker::linkGlobalInit(GlobalVariable &Dst, GlobalVariable &Src) {
  // Figure out what the initializer looks like in the dest module.
  Dst.setInitializer(MapValue(Src.getInitializer(), ValueMap, RF_None, &TypeMap,
                              &ValMaterializer));
}

/// Copy the source function over into the dest function and fix up references
/// to values. At this point we know that Dest is an external function, and
/// that Src is not.
bool ModuleLinker::linkFunctionBody(Function &Dst, Function &Src) {
  assert(Dst.isDeclaration() && !Src.isDeclaration());

  // Materialize if needed.
  if (std::error_code EC = Src.materialize())
    return emitError(EC.message());

  // Link in the prefix data.
  if (Src.hasPrefixData())
    Dst.setPrefixData(MapValue(Src.getPrefixData(), ValueMap, RF_None, &TypeMap,
                               &ValMaterializer));

  // Link in the prologue data.
  if (Src.hasPrologueData())
    Dst.setPrologueData(MapValue(Src.getPrologueData(), ValueMap, RF_None,
                                 &TypeMap, &ValMaterializer));

  // Link in the personality function.
  if (Src.hasPersonalityFn())
    Dst.setPersonalityFn(MapValue(Src.getPersonalityFn(), ValueMap, RF_None,
                                  &TypeMap, &ValMaterializer));

  // Go through and convert function arguments over, remembering the mapping.
  Function::arg_iterator DI = Dst.arg_begin();
  for (Argument &Arg : Src.args()) {
    DI->setName(Arg.getName());  // Copy the name over.

    // Add a mapping to our mapping.
    ValueMap[&Arg] = DI;
    ++DI;
  }

  // Copy over the metadata attachments.
  SmallVector<std::pair<unsigned, MDNode *>, 8> MDs;
  Src.getAllMetadata(MDs);
  for (const auto &I : MDs)
    Dst.setMetadata(I.first, MapMetadata(I.second, ValueMap, RF_None, &TypeMap,
                                         &ValMaterializer));

  // Splice the body of the source function into the dest function.
  Dst.getBasicBlockList().splice(Dst.end(), Src.getBasicBlockList());

  // At this point, all of the instructions and values of the function are now
  // copied over.  The only problem is that they are still referencing values in
  // the Source function as operands.  Loop through all of the operands of the
  // functions and patch them up to point to the local versions.
  for (BasicBlock &BB : Dst)
    for (Instruction &I : BB)
      RemapInstruction(&I, ValueMap, RF_IgnoreMissingEntries, &TypeMap,
                       &ValMaterializer);

  // There is no need to map the arguments anymore.
  for (Argument &Arg : Src.args())
    ValueMap.erase(&Arg);

  Src.dematerialize();
  return false;
}

void ModuleLinker::linkAliasBody(GlobalAlias &Dst, GlobalAlias &Src) {
  Constant *Aliasee = Src.getAliasee();
  Constant *Val =
      MapValue(Aliasee, ValueMap, RF_None, &TypeMap, &ValMaterializer);
  Dst.setAliasee(Val);
}

bool ModuleLinker::linkGlobalValueBody(GlobalValue &Src) {
  Value *Dst = ValueMap[&Src];
  assert(Dst);
  if (auto *F = dyn_cast<Function>(&Src))
    return linkFunctionBody(cast<Function>(*Dst), *F);
  if (auto *GVar = dyn_cast<GlobalVariable>(&Src)) {
    linkGlobalInit(cast<GlobalVariable>(*Dst), *GVar);
    return false;
  }
  linkAliasBody(cast<GlobalAlias>(*Dst), cast<GlobalAlias>(Src));
  return false;
}

/// Insert all of the named MDNodes in Src into the Dest module.
void ModuleLinker::linkNamedMDNodes() {
  const NamedMDNode *SrcModFlags = SrcM->getModuleFlagsMetadata();
  for (const NamedMDNode &NMD : SrcM->named_metadata()) {
    // Don't link module flags here. Do them separately.
    if (&NMD == SrcModFlags)
      continue;
    NamedMDNode *DestNMD = DstM->getOrInsertNamedMetadata(NMD.getName());
    // Add Src elements into Dest node.
    for (const MDNode *op : NMD.operands())
      DestNMD->addOperand(
          MapMetadata(op, ValueMap, RF_None, &TypeMap, &ValMaterializer));
  }
}

/// Drop DISubprograms that have been superseded.
///
/// FIXME: this creates an asymmetric result: we strip functions from losing
/// subprograms in DstM, but leave losing subprograms in SrcM.
/// TODO: Remove this logic once the backend can correctly determine canonical
/// subprograms.
void ModuleLinker::stripReplacedSubprograms() {
  // Avoid quadratic runtime by returning early when there's nothing to do.
  if (OverridingFunctions.empty())
    return;

  // Move the functions now, so the set gets cleared even on early returns.
  auto Functions = std::move(OverridingFunctions);
  OverridingFunctions.clear();

  // Drop functions from subprograms if they've been overridden by the new
  // compile unit.
  NamedMDNode *CompileUnits = DstM->getNamedMetadata("llvm.dbg.cu");
  if (!CompileUnits)
    return;
  for (unsigned I = 0, E = CompileUnits->getNumOperands(); I != E; ++I) {
    auto *CU = cast<DICompileUnit>(CompileUnits->getOperand(I));
    assert(CU && "Expected valid compile unit");

    for (DISubprogram *SP : CU->getSubprograms()) {
      if (!SP || !SP->getFunction() || !Functions.count(SP->getFunction()))
        continue;

      // Prevent DebugInfoFinder from tagging this as the canonical subprogram,
      // since the canonical one is in the incoming module.
      SP->replaceFunction(nullptr);
    }
  }
}

/// Merge the linker flags in Src into the Dest module.
bool ModuleLinker::linkModuleFlagsMetadata() {
  // If the source module has no module flags, we are done.
  const NamedMDNode *SrcModFlags = SrcM->getModuleFlagsMetadata();
  if (!SrcModFlags) return false;

  // If the destination module doesn't have module flags yet, then just copy
  // over the source module's flags.
  NamedMDNode *DstModFlags = DstM->getOrInsertModuleFlagsMetadata();
  if (DstModFlags->getNumOperands() == 0) {
    for (unsigned I = 0, E = SrcModFlags->getNumOperands(); I != E; ++I)
      DstModFlags->addOperand(SrcModFlags->getOperand(I));

    return false;
  }

  // First build a map of the existing module flags and requirements.
  DenseMap<MDString *, std::pair<MDNode *, unsigned>> Flags;
  SmallSetVector<MDNode*, 16> Requirements;
  for (unsigned I = 0, E = DstModFlags->getNumOperands(); I != E; ++I) {
    MDNode *Op = DstModFlags->getOperand(I);
    ConstantInt *Behavior = mdconst::extract<ConstantInt>(Op->getOperand(0));
    MDString *ID = cast<MDString>(Op->getOperand(1));

    if (Behavior->getZExtValue() == Module::Require) {
      Requirements.insert(cast<MDNode>(Op->getOperand(2)));
    } else {
      Flags[ID] = std::make_pair(Op, I);
    }
  }

  // Merge in the flags from the source module, and also collect its set of
  // requirements.
  bool HasErr = false;
  for (unsigned I = 0, E = SrcModFlags->getNumOperands(); I != E; ++I) {
    MDNode *SrcOp = SrcModFlags->getOperand(I);
    ConstantInt *SrcBehavior =
        mdconst::extract<ConstantInt>(SrcOp->getOperand(0));
    MDString *ID = cast<MDString>(SrcOp->getOperand(1));
    MDNode *DstOp;
    unsigned DstIndex;
    std::tie(DstOp, DstIndex) = Flags.lookup(ID);
    unsigned SrcBehaviorValue = SrcBehavior->getZExtValue();

    // If this is a requirement, add it and continue.
    if (SrcBehaviorValue == Module::Require) {
      // If the destination module does not already have this requirement, add
      // it.
      if (Requirements.insert(cast<MDNode>(SrcOp->getOperand(2)))) {
        DstModFlags->addOperand(SrcOp);
      }
      continue;
    }

    // If there is no existing flag with this ID, just add it.
    if (!DstOp) {
      Flags[ID] = std::make_pair(SrcOp, DstModFlags->getNumOperands());
      DstModFlags->addOperand(SrcOp);
      continue;
    }

    // Otherwise, perform a merge.
    ConstantInt *DstBehavior =
        mdconst::extract<ConstantInt>(DstOp->getOperand(0));
    unsigned DstBehaviorValue = DstBehavior->getZExtValue();

    // If either flag has override behavior, handle it first.
    if (DstBehaviorValue == Module::Override) {
      // Diagnose inconsistent flags which both have override behavior.
      if (SrcBehaviorValue == Module::Override &&
          SrcOp->getOperand(2) != DstOp->getOperand(2)) {
        HasErr |= emitError("linking module flags '" + ID->getString() +
                            "': IDs have conflicting override values");
      }
      continue;
    } else if (SrcBehaviorValue == Module::Override) {
      // Update the destination flag to that of the source.
      DstModFlags->setOperand(DstIndex, SrcOp);
      Flags[ID].first = SrcOp;
      continue;
    }

    // Diagnose inconsistent merge behavior types.
    if (SrcBehaviorValue != DstBehaviorValue) {
      HasErr |= emitError("linking module flags '" + ID->getString() +
                          "': IDs have conflicting behaviors");
      continue;
    }

    auto replaceDstValue = [&](MDNode *New) {
      Metadata *FlagOps[] = {DstOp->getOperand(0), ID, New};
      MDNode *Flag = MDNode::get(DstM->getContext(), FlagOps);
      DstModFlags->setOperand(DstIndex, Flag);
      Flags[ID].first = Flag;
    };

    // Perform the merge for standard behavior types.
    switch (SrcBehaviorValue) {
    case Module::Require:
    case Module::Override: llvm_unreachable("not possible");
    case Module::Error: {
      // Emit an error if the values differ.
      if (SrcOp->getOperand(2) != DstOp->getOperand(2)) {
        HasErr |= emitError("linking module flags '" + ID->getString() +
                            "': IDs have conflicting values");
      }
      continue;
    }
    case Module::Warning: {
      // Emit a warning if the values differ.
      if (SrcOp->getOperand(2) != DstOp->getOperand(2)) {
        emitWarning("linking module flags '" + ID->getString() +
                    "': IDs have conflicting values");
      }
      continue;
    }
    case Module::Append: {
      MDNode *DstValue = cast<MDNode>(DstOp->getOperand(2));
      MDNode *SrcValue = cast<MDNode>(SrcOp->getOperand(2));
      SmallVector<Metadata *, 8> MDs;
      MDs.reserve(DstValue->getNumOperands() + SrcValue->getNumOperands());
      MDs.append(DstValue->op_begin(), DstValue->op_end());
      MDs.append(SrcValue->op_begin(), SrcValue->op_end());

      replaceDstValue(MDNode::get(DstM->getContext(), MDs));
      break;
    }
    case Module::AppendUnique: {
      SmallSetVector<Metadata *, 16> Elts;
      MDNode *DstValue = cast<MDNode>(DstOp->getOperand(2));
      MDNode *SrcValue = cast<MDNode>(SrcOp->getOperand(2));
      Elts.insert(DstValue->op_begin(), DstValue->op_end());
      Elts.insert(SrcValue->op_begin(), SrcValue->op_end());

      replaceDstValue(MDNode::get(DstM->getContext(),
                                  makeArrayRef(Elts.begin(), Elts.end())));
      break;
    }
    }
  }

  // Check all of the requirements.
  for (unsigned I = 0, E = Requirements.size(); I != E; ++I) {
    MDNode *Requirement = Requirements[I];
    MDString *Flag = cast<MDString>(Requirement->getOperand(0));
    Metadata *ReqValue = Requirement->getOperand(1);

    MDNode *Op = Flags[Flag].first;
    if (!Op || Op->getOperand(2) != ReqValue) {
      HasErr |= emitError("linking module flags '" + Flag->getString() +
                          "': does not have the required value");
      continue;
    }
  }

  return HasErr;
}

// This function returns true if the triples match.
static bool triplesMatch(const Triple &T0, const Triple &T1) {
  // If vendor is apple, ignore the version number.
  if (T0.getVendor() == Triple::Apple)
    return T0.getArch() == T1.getArch() &&
           T0.getSubArch() == T1.getSubArch() &&
           T0.getVendor() == T1.getVendor() &&
           T0.getOS() == T1.getOS();

  return T0 == T1;
}

// This function returns the merged triple.
static std::string mergeTriples(const Triple &SrcTriple, const Triple &DstTriple) {
  // If vendor is apple, pick the triple with the larger version number.
  if (SrcTriple.getVendor() == Triple::Apple)
    if (DstTriple.isOSVersionLT(SrcTriple))
      return SrcTriple.str();

  return DstTriple.str();
}

bool ModuleLinker::run() {
  assert(DstM && "Null destination module");
  assert(SrcM && "Null source module");

  // Inherit the target data from the source module if the destination module
  // doesn't have one already.
  if (DstM->getDataLayout().isDefault())
    DstM->setDataLayout(SrcM->getDataLayout());

  if (SrcM->getDataLayout() != DstM->getDataLayout()) {
    emitWarning("Linking two modules of different data layouts: '" +
                SrcM->getModuleIdentifier() + "' is '" +
                SrcM->getDataLayoutStr() + "' whereas '" +
                DstM->getModuleIdentifier() + "' is '" +
                DstM->getDataLayoutStr() + "'\n");
  }

  // Copy the target triple from the source to dest if the dest's is empty.
  if (DstM->getTargetTriple().empty() && !SrcM->getTargetTriple().empty())
    DstM->setTargetTriple(SrcM->getTargetTriple());

  Triple SrcTriple(SrcM->getTargetTriple()), DstTriple(DstM->getTargetTriple());

  if (!SrcM->getTargetTriple().empty() && !triplesMatch(SrcTriple, DstTriple))
    emitWarning("Linking two modules of different target triples: " +
                SrcM->getModuleIdentifier() + "' is '" +
                SrcM->getTargetTriple() + "' whereas '" +
                DstM->getModuleIdentifier() + "' is '" +
                DstM->getTargetTriple() + "'\n");

  DstM->setTargetTriple(mergeTriples(SrcTriple, DstTriple));

  // Append the module inline asm string.
  if (!SrcM->getModuleInlineAsm().empty()) {
    if (DstM->getModuleInlineAsm().empty())
      DstM->setModuleInlineAsm(SrcM->getModuleInlineAsm());
    else
      DstM->setModuleInlineAsm(DstM->getModuleInlineAsm()+"\n"+
                               SrcM->getModuleInlineAsm());
  }

  // Loop over all of the linked values to compute type mappings.
  computeTypeMapping();

  ComdatsChosen.clear();
  for (const auto &SMEC : SrcM->getComdatSymbolTable()) {
    const Comdat &C = SMEC.getValue();
    if (ComdatsChosen.count(&C))
      continue;
    Comdat::SelectionKind SK;
    bool LinkFromSrc;
    if (getComdatResult(&C, SK, LinkFromSrc))
      return true;
    ComdatsChosen[&C] = std::make_pair(SK, LinkFromSrc);
  }

  // Upgrade mismatched global arrays.
  upgradeMismatchedGlobals();

  // Insert all of the globals in src into the DstM module... without linking
  // initializers (which could refer to functions not yet mapped over).
  for (GlobalVariable &GV : SrcM->globals())
    if (linkGlobalValueProto(&GV))
      return true;

  // Link the functions together between the two modules, without doing function
  // bodies... this just adds external function prototypes to the DstM
  // function...  We do this so that when we begin processing function bodies,
  // all of the global values that may be referenced are available in our
  // ValueMap.
  for (Function &F :*SrcM)
    if (linkGlobalValueProto(&F))
      return true;

  // If there were any aliases, link them now.
  for (GlobalAlias &GA : SrcM->aliases())
    if (linkGlobalValueProto(&GA))
      return true;

  for (const AppendingVarInfo &AppendingVar : AppendingVars)
    linkAppendingVarInit(AppendingVar);

  for (const auto &Entry : DstM->getComdatSymbolTable()) {
    const Comdat &C = Entry.getValue();
    if (C.getSelectionKind() == Comdat::Any)
      continue;
    const GlobalValue *GV = SrcM->getNamedValue(C.getName());
    if (GV)
      MapValue(GV, ValueMap, RF_None, &TypeMap, &ValMaterializer);
  }

  // Strip replaced subprograms before mapping any metadata -- so that we're
  // not changing metadata from the source module (note that
  // linkGlobalValueBody() eventually calls RemapInstruction() and therefore
  // MapMetadata()) -- but after linking global value protocols -- so that
  // OverridingFunctions has been built.
  stripReplacedSubprograms();

  // Link in the function bodies that are defined in the source module into
  // DstM.
  for (Function &SF : *SrcM) {
    // Skip if no body (function is external).
    if (SF.isDeclaration())
      continue;

    // Skip if not linking from source.
    if (DoNotLinkFromSource.count(&SF))
      continue;

    if (linkGlobalValueBody(SF))
      return true;
  }

  // Resolve all uses of aliases with aliasees.
  for (GlobalAlias &Src : SrcM->aliases()) {
    if (DoNotLinkFromSource.count(&Src))
      continue;
    linkGlobalValueBody(Src);
  }

  // Remap all of the named MDNodes in Src into the DstM module. We do this
  // after linking GlobalValues so that MDNodes that reference GlobalValues
  // are properly remapped.
  linkNamedMDNodes();

  // Merge the module flags into the DstM module.
  if (linkModuleFlagsMetadata())
    return true;

  // Update the initializers in the DstM module now that all globals that may
  // be referenced are in DstM.
  for (GlobalVariable &Src : SrcM->globals()) {
    // Only process initialized GV's or ones not already in dest.
    if (!Src.hasInitializer() || DoNotLinkFromSource.count(&Src))
      continue;
    linkGlobalValueBody(Src);
  }

  // Process vector of lazily linked in functions.
  while (!LazilyLinkGlobalValues.empty()) {
    GlobalValue *SGV = LazilyLinkGlobalValues.back();
    LazilyLinkGlobalValues.pop_back();

    assert(!SGV->isDeclaration() && "users should not pass down decls");
    if (linkGlobalValueBody(*SGV))
      return true;
  }

  return false;
}

Linker::StructTypeKeyInfo::KeyTy::KeyTy(ArrayRef<Type *> E, bool P)
    : ETypes(E), IsPacked(P) {}

Linker::StructTypeKeyInfo::KeyTy::KeyTy(const StructType *ST)
    : ETypes(ST->elements()), IsPacked(ST->isPacked()) {}

bool Linker::StructTypeKeyInfo::KeyTy::operator==(const KeyTy &That) const {
  if (IsPacked != That.IsPacked)
    return false;
  if (ETypes != That.ETypes)
    return false;
  return true;
}

bool Linker::StructTypeKeyInfo::KeyTy::operator!=(const KeyTy &That) const {
  return !this->operator==(That);
}

StructType *Linker::StructTypeKeyInfo::getEmptyKey() {
  return DenseMapInfo<StructType *>::getEmptyKey();
}

StructType *Linker::StructTypeKeyInfo::getTombstoneKey() {
  return DenseMapInfo<StructType *>::getTombstoneKey();
}

unsigned Linker::StructTypeKeyInfo::getHashValue(const KeyTy &Key) {
  return hash_combine(hash_combine_range(Key.ETypes.begin(), Key.ETypes.end()),
                      Key.IsPacked);
}

unsigned Linker::StructTypeKeyInfo::getHashValue(const StructType *ST) {
  return getHashValue(KeyTy(ST));
}

bool Linker::StructTypeKeyInfo::isEqual(const KeyTy &LHS,
                                        const StructType *RHS) {
  if (RHS == getEmptyKey() || RHS == getTombstoneKey())
    return false;
  return LHS == KeyTy(RHS);
}

bool Linker::StructTypeKeyInfo::isEqual(const StructType *LHS,
                                        const StructType *RHS) {
  if (RHS == getEmptyKey())
    return LHS == getEmptyKey();

  if (RHS == getTombstoneKey())
    return LHS == getTombstoneKey();

  return KeyTy(LHS) == KeyTy(RHS);
}

void Linker::IdentifiedStructTypeSet::addNonOpaque(StructType *Ty) {
  assert(!Ty->isOpaque());
  NonOpaqueStructTypes.insert(Ty);
}

void Linker::IdentifiedStructTypeSet::switchToNonOpaque(StructType *Ty) {
  assert(!Ty->isOpaque());
  NonOpaqueStructTypes.insert(Ty);
  bool Removed = OpaqueStructTypes.erase(Ty);
  (void)Removed;
  assert(Removed);
}

void Linker::IdentifiedStructTypeSet::addOpaque(StructType *Ty) {
  assert(Ty->isOpaque());
  OpaqueStructTypes.insert(Ty);
}

StructType *
Linker::IdentifiedStructTypeSet::findNonOpaque(ArrayRef<Type *> ETypes,
                                               bool IsPacked) {
  Linker::StructTypeKeyInfo::KeyTy Key(ETypes, IsPacked);
  auto I = NonOpaqueStructTypes.find_as(Key);
  if (I == NonOpaqueStructTypes.end())
    return nullptr;
  return *I;
}

bool Linker::IdentifiedStructTypeSet::hasType(StructType *Ty) {
  if (Ty->isOpaque())
    return OpaqueStructTypes.count(Ty);
  auto I = NonOpaqueStructTypes.find(Ty);
  if (I == NonOpaqueStructTypes.end())
    return false;
  return *I == Ty;
}

void Linker::init(Module *M, DiagnosticHandlerFunction DiagnosticHandler) {
  this->Composite = M;
  this->DiagnosticHandler = DiagnosticHandler;

  TypeFinder StructTypes;
  StructTypes.run(*M, true);
  for (StructType *Ty : StructTypes) {
    if (Ty->isOpaque())
      IdentifiedStructTypes.addOpaque(Ty);
    else
      IdentifiedStructTypes.addNonOpaque(Ty);
  }
}

Linker::Linker(Module *M, DiagnosticHandlerFunction DiagnosticHandler) {
  init(M, DiagnosticHandler);
}

Linker::Linker(Module *M) {
  init(M, [this](const DiagnosticInfo &DI) {
    Composite->getContext().diagnose(DI);
  });
}

Linker::~Linker() {
}

void Linker::deleteModule() {
  delete Composite;
  Composite = nullptr;
}

bool Linker::linkInModule(Module *Src, bool OverrideSymbols) {
  ModuleLinker TheLinker(Composite, IdentifiedStructTypes, Src,
                         DiagnosticHandler, OverrideSymbols);
  bool RetCode = TheLinker.run();
  Composite->dropTriviallyDeadConstantArrays();
  return RetCode;
}

void Linker::setModule(Module *Dst) {
  init(Dst, DiagnosticHandler);
}

//===----------------------------------------------------------------------===//
// LinkModules entrypoint.
//===----------------------------------------------------------------------===//

/// This function links two modules together, with the resulting Dest module
/// modified to be the composite of the two input modules. If an error occurs,
/// true is returned and ErrorMsg (if not null) is set to indicate the problem.
/// Upon failure, the Dest module could be in a modified state, and shouldn't be
/// relied on to be consistent.
bool Linker::LinkModules(Module *Dest, Module *Src,
                         DiagnosticHandlerFunction DiagnosticHandler) {
  Linker L(Dest, DiagnosticHandler);
  return L.linkInModule(Src);
}

bool Linker::LinkModules(Module *Dest, Module *Src) {
  Linker L(Dest);
  return L.linkInModule(Src);
}

//===----------------------------------------------------------------------===//
// C API.
//===----------------------------------------------------------------------===//

_Use_decl_annotations_
LLVMBool LLVMLinkModules(LLVMModuleRef Dest, LLVMModuleRef Src,
                         LLVMLinkerMode Unused, char **OutMessages) {
  Module *D = unwrap(Dest);
  std::string Message;
  raw_string_ostream Stream(Message);
  DiagnosticPrinterRawOStream DP(Stream);

  LLVMBool Result = Linker::LinkModules(
      D, unwrap(Src), [&](const DiagnosticInfo &DI) { DI.print(DP); });

  if (OutMessages && Result) {
    Stream.flush();
#ifdef _WIN32
    *OutMessages = _strdup(Message.c_str()); // HLSL Change for strdup
#else
    *OutMessages = strdup(Message.c_str());
#endif
  }
  return Result;
}
