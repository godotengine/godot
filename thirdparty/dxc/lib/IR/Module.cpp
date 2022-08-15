//===-- Module.cpp - Implement the Module class ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Module class for the IR library.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Module.h"
#include "SymbolTableListTraitsImpl.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GVMaterializer.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/TypeFinder.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/RandomNumberGenerator.h"
#include <algorithm>
#include <cstdarg>
#include <cstdlib>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Methods to implement the globals and functions lists.
//

// Explicit instantiations of SymbolTableListTraits since some of the methods
// are not in the public header file.
template class llvm::SymbolTableListTraits<Function, Module>;
template class llvm::SymbolTableListTraits<GlobalVariable, Module>;
template class llvm::SymbolTableListTraits<GlobalAlias, Module>;

//===----------------------------------------------------------------------===//
// Primitive Module methods.
//

Module::Module(StringRef MID, LLVMContext &C)
    : Context(C), Materializer(), ModuleID(MID), DL("") {
  // HLSL Change - use unique_ptr to avoid leaks
  std::unique_ptr<ValueSymbolTable> ValSymTabPtr(new ValueSymbolTable());
  std::unique_ptr<StringMap<NamedMDNode *> > NamedMDSymTabPtr(new StringMap<NamedMDNode *>());
  Context.addModule(this);
  ValSymTab = ValSymTabPtr.release();
  NamedMDSymTab = NamedMDSymTabPtr.release();
}

Module::~Module() {
  // HLSL Change Starts
  ResetHLModule();
  ResetDxilModule();
  // HLSL Change Ends
  Context.removeModule(this);
  dropAllReferences();
  GlobalList.clear();
  FunctionList.clear();
  AliasList.clear();
  NamedMDList.clear();
  delete ValSymTab;
  delete static_cast<StringMap<NamedMDNode *> *>(NamedMDSymTab);
}

RandomNumberGenerator *Module::createRNG(const Pass* P) const {
  SmallString<32> Salt(P->getPassName());

  // This RNG is guaranteed to produce the same random stream only
  // when the Module ID and thus the input filename is the same. This
  // might be problematic if the input filename extension changes
  // (e.g. from .c to .bc or .ll).
  //
  // We could store this salt in NamedMetadata, but this would make
  // the parameter non-const. This would unfortunately make this
  // interface unusable by any Machine passes, since they only have a
  // const reference to their IR Module. Alternatively we can always
  // store salt metadata from the Module constructor.
  Salt += sys::path::filename(getModuleIdentifier());

  return new RandomNumberGenerator(Salt);
}


/// getNamedValue - Return the first global value in the module with
/// the specified name, of arbitrary type.  This method returns null
/// if a global with the specified name is not found.
GlobalValue *Module::getNamedValue(StringRef Name) const {
  return cast_or_null<GlobalValue>(getValueSymbolTable().lookup(Name));
}

/// getMDKindID - Return a unique non-zero ID for the specified metadata kind.
/// This ID is uniqued across modules in the current LLVMContext.
unsigned Module::getMDKindID(StringRef Name) const {
  return Context.getMDKindID(Name);
}

/// getMDKindNames - Populate client supplied SmallVector with the name for
/// custom metadata IDs registered in this LLVMContext.   ID #0 is not used,
/// so it is filled in as an empty string.
void Module::getMDKindNames(SmallVectorImpl<StringRef> &Result) const {
  return Context.getMDKindNames(Result);
}


//===----------------------------------------------------------------------===//
// Methods for easy access to the functions in the module.
//

// getOrInsertFunction - Look up the specified function in the module symbol
// table.  If it does not exist, add a prototype for the function and return
// it.  This is nice because it allows most passes to get away with not handling
// the symbol table directly for this common task.
//
Constant *Module::getOrInsertFunction(StringRef Name,
                                      FunctionType *Ty,
                                      AttributeSet AttributeList) {
  // See if we have a definition for the specified function already.
  GlobalValue *F = getNamedValue(Name);
  if (!F) {
    // Nope, add it
    Function *New = Function::Create(Ty, GlobalVariable::ExternalLinkage, Name);
    if (!New->isIntrinsic())       // Intrinsics get attrs set on construction
      New->setAttributes(AttributeList);
    FunctionList.push_back(New);
    return New;                    // Return the new prototype.
  }

  // If the function exists but has the wrong type, return a bitcast to the
  // right type.
  if (F->getType() != PointerType::getUnqual(Ty))
    return ConstantExpr::getBitCast(F, PointerType::getUnqual(Ty));

  // Otherwise, we just found the existing function or a prototype.
  return F;
}

Constant *Module::getOrInsertFunction(StringRef Name,
                                      FunctionType *Ty) {
  return getOrInsertFunction(Name, Ty, AttributeSet());
}

// getOrInsertFunction - Look up the specified function in the module symbol
// table.  If it does not exist, add a prototype for the function and return it.
// This version of the method takes a null terminated list of function
// arguments, which makes it easier for clients to use.
//
Constant *Module::getOrInsertFunction(StringRef Name,
                                      AttributeSet AttributeList,
                                      Type *RetTy, ...) {
  va_list Args;
  va_start(Args, RetTy);

  // Build the list of argument types...
  std::vector<Type*> ArgTys;
  while (Type *ArgTy = va_arg(Args, Type*))
    ArgTys.push_back(ArgTy);

  va_end(Args);

  // Build the function type and chain to the other getOrInsertFunction...
  return getOrInsertFunction(Name,
                             FunctionType::get(RetTy, ArgTys, false),
                             AttributeList);
}

Constant *Module::getOrInsertFunction(StringRef Name,
                                      Type *RetTy, ...) {
  va_list Args;
  va_start(Args, RetTy);

  // Build the list of argument types...
  std::vector<Type*> ArgTys;
  while (Type *ArgTy = va_arg(Args, Type*))
    ArgTys.push_back(ArgTy);

  va_end(Args);

  // Build the function type and chain to the other getOrInsertFunction...
  return getOrInsertFunction(Name,
                             FunctionType::get(RetTy, ArgTys, false),
                             AttributeSet());
}

// getFunction - Look up the specified function in the module symbol table.
// If it does not exist, return null.
//
Function *Module::getFunction(StringRef Name) const {
  return dyn_cast_or_null<Function>(getNamedValue(Name));
}

//===----------------------------------------------------------------------===//
// Methods for easy access to the global variables in the module.
//

/// getGlobalVariable - Look up the specified global variable in the module
/// symbol table.  If it does not exist, return null.  The type argument
/// should be the underlying type of the global, i.e., it should not have
/// the top-level PointerType, which represents the address of the global.
/// If AllowLocal is set to true, this function will return types that
/// have an local. By default, these types are not returned.
///
GlobalVariable *Module::getGlobalVariable(StringRef Name, bool AllowLocal) {
  if (GlobalVariable *Result =
      dyn_cast_or_null<GlobalVariable>(getNamedValue(Name)))
    if (AllowLocal || !Result->hasLocalLinkage())
      return Result;
  return nullptr;
}

/// getOrInsertGlobal - Look up the specified global in the module symbol table.
///   1. If it does not exist, add a declaration of the global and return it.
///   2. Else, the global exists but has the wrong type: return the function
///      with a constantexpr cast to the right type.
///   3. Finally, if the existing global is the correct declaration, return the
///      existing global.
Constant *Module::getOrInsertGlobal(StringRef Name, Type *Ty) {
  // See if we have a definition for the specified global already.
  GlobalVariable *GV = dyn_cast_or_null<GlobalVariable>(getNamedValue(Name));
  if (!GV) {
    // Nope, add it
    GlobalVariable *New =
      new GlobalVariable(*this, Ty, false, GlobalVariable::ExternalLinkage,
                         nullptr, Name);
     return New;                    // Return the new declaration.
  }

  // If the variable exists but has the wrong type, return a bitcast to the
  // right type.
  Type *GVTy = GV->getType();
  PointerType *PTy = PointerType::get(Ty, GVTy->getPointerAddressSpace());
  if (GVTy != PTy)
    return ConstantExpr::getBitCast(GV, PTy);

  // Otherwise, we just found the existing function or a prototype.
  return GV;
}

//===----------------------------------------------------------------------===//
// Methods for easy access to the global variables in the module.
//

// getNamedAlias - Look up the specified global in the module symbol table.
// If it does not exist, return null.
//
GlobalAlias *Module::getNamedAlias(StringRef Name) const {
  return dyn_cast_or_null<GlobalAlias>(getNamedValue(Name));
}

/// getNamedMetadata - Return the first NamedMDNode in the module with the
/// specified name. This method returns null if a NamedMDNode with the
/// specified name is not found.
NamedMDNode *Module::getNamedMetadata(const Twine &Name) const {
  SmallString<256> NameData;
  StringRef NameRef = Name.toStringRef(NameData);
  return static_cast<StringMap<NamedMDNode*> *>(NamedMDSymTab)->lookup(NameRef);
}

/// getOrInsertNamedMetadata - Return the first named MDNode in the module
/// with the specified name. This method returns a new NamedMDNode if a
/// NamedMDNode with the specified name is not found.
NamedMDNode *Module::getOrInsertNamedMetadata(StringRef Name) {
  NamedMDNode *&NMD =
    (*static_cast<StringMap<NamedMDNode *> *>(NamedMDSymTab))[Name];
  if (!NMD) {
    NMD = new NamedMDNode(Name);
    NMD->setParent(this);
    NamedMDList.push_back(NMD);
  }
  return NMD;
}

/// eraseNamedMetadata - Remove the given NamedMDNode from this module and
/// delete it.
void Module::eraseNamedMetadata(NamedMDNode *NMD) {
  static_cast<StringMap<NamedMDNode *> *>(NamedMDSymTab)->erase(NMD->getName());
  NamedMDList.erase(NMD);
}

bool Module::isValidModFlagBehavior(Metadata *MD, ModFlagBehavior &MFB) {
  if (ConstantInt *Behavior = mdconst::dyn_extract_or_null<ConstantInt>(MD)) {
    uint64_t Val = Behavior->getLimitedValue();
    if (Val >= ModFlagBehaviorFirstVal && Val <= ModFlagBehaviorLastVal) {
      MFB = static_cast<ModFlagBehavior>(Val);
      return true;
    }
  }
  return false;
}

/// getModuleFlagsMetadata - Returns the module flags in the provided vector.
void Module::
getModuleFlagsMetadata(SmallVectorImpl<ModuleFlagEntry> &Flags) const {
  const NamedMDNode *ModFlags = getModuleFlagsMetadata();
  if (!ModFlags) return;

  for (const MDNode *Flag : ModFlags->operands()) {
    ModFlagBehavior MFB;
    if (Flag->getNumOperands() >= 3 &&
        isValidModFlagBehavior(Flag->getOperand(0), MFB) &&
        dyn_cast_or_null<MDString>(Flag->getOperand(1))) {
      // Check the operands of the MDNode before accessing the operands.
      // The verifier will actually catch these failures.
      MDString *Key = cast<MDString>(Flag->getOperand(1));
      Metadata *Val = Flag->getOperand(2);
      Flags.push_back(ModuleFlagEntry(MFB, Key, Val));
    }
  }
}

/// Return the corresponding value if Key appears in module flags, otherwise
/// return null.
Metadata *Module::getModuleFlag(StringRef Key) const {
  SmallVector<Module::ModuleFlagEntry, 8> ModuleFlags;
  getModuleFlagsMetadata(ModuleFlags);
  for (const ModuleFlagEntry &MFE : ModuleFlags) {
    if (Key == MFE.Key->getString())
      return MFE.Val;
  }
  return nullptr;
}

/// getModuleFlagsMetadata - Returns the NamedMDNode in the module that
/// represents module-level flags. This method returns null if there are no
/// module-level flags.
NamedMDNode *Module::getModuleFlagsMetadata() const {
  return getNamedMetadata("llvm.module.flags");
}

/// getOrInsertModuleFlagsMetadata - Returns the NamedMDNode in the module that
/// represents module-level flags. If module-level flags aren't found, it
/// creates the named metadata that contains them.
NamedMDNode *Module::getOrInsertModuleFlagsMetadata() {
  return getOrInsertNamedMetadata("llvm.module.flags");
}

/// addModuleFlag - Add a module-level flag to the module-level flags
/// metadata. It will create the module-level flags named metadata if it doesn't
/// already exist.
void Module::addModuleFlag(ModFlagBehavior Behavior, StringRef Key,
                           Metadata *Val) {
  Type *Int32Ty = Type::getInt32Ty(Context);
  Metadata *Ops[3] = {
      ConstantAsMetadata::get(ConstantInt::get(Int32Ty, Behavior)),
      MDString::get(Context, Key), Val};
  getOrInsertModuleFlagsMetadata()->addOperand(MDNode::get(Context, Ops));
}
void Module::addModuleFlag(ModFlagBehavior Behavior, StringRef Key,
                           Constant *Val) {
  addModuleFlag(Behavior, Key, ConstantAsMetadata::get(Val));
}
void Module::addModuleFlag(ModFlagBehavior Behavior, StringRef Key,
                           uint32_t Val) {
  Type *Int32Ty = Type::getInt32Ty(Context);
  addModuleFlag(Behavior, Key, ConstantInt::get(Int32Ty, Val));
}
void Module::addModuleFlag(MDNode *Node) {
  assert(Node->getNumOperands() == 3 &&
         "Invalid number of operands for module flag!");
  assert(mdconst::hasa<ConstantInt>(Node->getOperand(0)) &&
         isa<MDString>(Node->getOperand(1)) &&
         "Invalid operand types for module flag!");
  getOrInsertModuleFlagsMetadata()->addOperand(Node);
}

void Module::setDataLayout(StringRef Desc) {
  DL.reset(Desc);
}

void Module::setDataLayout(const DataLayout &Other) { DL = Other; }

const DataLayout &Module::getDataLayout() const { return DL; }

//===----------------------------------------------------------------------===//
// Methods to control the materialization of GlobalValues in the Module.
//
void Module::setMaterializer(GVMaterializer *GVM) {
  assert(!Materializer &&
         "Module already has a GVMaterializer.  Call MaterializeAllPermanently"
         " to clear it out before setting another one.");
  Materializer.reset(GVM);
}

bool Module::isDematerializable(const GlobalValue *GV) const {
  if (Materializer)
    return Materializer->isDematerializable(GV);
  return false;
}

std::error_code Module::materialize(GlobalValue *GV) {
  if (!Materializer)
    return std::error_code();

  return Materializer->materialize(GV);
}

void Module::dematerialize(GlobalValue *GV) {
  if (Materializer)
    return Materializer->dematerialize(GV);
}

std::error_code Module::materializeAll() {
  if (!Materializer)
    return std::error_code();
  return Materializer->materializeModule(this);
}

std::error_code Module::materializeAllPermanently() {
  if (std::error_code EC = materializeAll())
    return EC;

  Materializer.reset();
  return std::error_code();
}

std::error_code Module::materializeMetadata() {
  if (!Materializer)
    return std::error_code();
  return Materializer->materializeMetadata();
}

std::error_code Module::materializeSelectNamedMetadata(ArrayRef<StringRef> NamedMetadata) {
  if (!Materializer)
    return std::error_code();
  return Materializer->materializeSelectNamedMetadata(NamedMetadata);
}

//===----------------------------------------------------------------------===//
// Other module related stuff.
//

std::vector<StructType *> Module::getIdentifiedStructTypes() const {
  // If we have a materializer, it is possible that some unread function
  // uses a type that is currently not visible to a TypeFinder, so ask
  // the materializer which types it created.
  if (Materializer)
    return Materializer->getIdentifiedStructTypes();

  std::vector<StructType *> Ret;
  TypeFinder SrcStructTypes;
  SrcStructTypes.run(*this, true);
  Ret.assign(SrcStructTypes.begin(), SrcStructTypes.end());
  return Ret;
}

// dropAllReferences() - This function causes all the subelements to "let go"
// of all references that they are maintaining.  This allows one to 'delete' a
// whole module at a time, even though there may be circular references... first
// all references are dropped, and all use counts go to zero.  Then everything
// is deleted for real.  Note that no operations are valid on an object that
// has "dropped all references", except operator delete.
//
void Module::dropAllReferences() {
  for (Function &F : *this)
    F.dropAllReferences();

  for (GlobalVariable &GV : globals())
    GV.dropAllReferences();

  for (GlobalAlias &GA : aliases())
    GA.dropAllReferences();
}

unsigned Module::getDwarfVersion() const {
  auto *Val = cast_or_null<ConstantAsMetadata>(getModuleFlag("Dwarf Version"));
  if (!Val)
    return dwarf::DWARF_VERSION;
  return cast<ConstantInt>(Val->getValue())->getZExtValue();
}

Comdat *Module::getOrInsertComdat(StringRef Name) {
  auto &Entry = *ComdatSymTab.insert(std::make_pair(Name, Comdat())).first;
  Entry.second.Name = &Entry;
  return &Entry.second;
}

PICLevel::Level Module::getPICLevel() const {
  auto *Val = cast_or_null<ConstantAsMetadata>(getModuleFlag("PIC Level"));

  if (Val == NULL)
    return PICLevel::Default;

  return static_cast<PICLevel::Level>(
      cast<ConstantInt>(Val->getValue())->getZExtValue());
}

void Module::setPICLevel(PICLevel::Level PL) {
  addModuleFlag(ModFlagBehavior::Error, "PIC Level", PL);
}
