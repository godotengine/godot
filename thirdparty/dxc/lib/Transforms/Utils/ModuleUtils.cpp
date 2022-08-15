//===-- ModuleUtils.cpp - Functions to manipulate Modules -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions perform manipulations on Modules.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static void appendToGlobalArray(const char *Array, 
                                Module &M, Function *F, int Priority) {
  IRBuilder<> IRB(M.getContext());
  FunctionType *FnTy = FunctionType::get(IRB.getVoidTy(), false);

  // Get the current set of static global constructors and add the new ctor
  // to the list.
  SmallVector<Constant *, 16> CurrentCtors;
  StructType *EltTy;
  if (GlobalVariable *GVCtor = M.getNamedGlobal(Array)) {
    // If there is a global_ctors array, use the existing struct type, which can
    // have 2 or 3 fields.
    ArrayType *ATy = cast<ArrayType>(GVCtor->getType()->getElementType());
    EltTy = cast<StructType>(ATy->getElementType());
    if (Constant *Init = GVCtor->getInitializer()) {
      unsigned n = Init->getNumOperands();
      CurrentCtors.reserve(n + 1);
      for (unsigned i = 0; i != n; ++i)
        CurrentCtors.push_back(cast<Constant>(Init->getOperand(i)));
    }
    GVCtor->eraseFromParent();
  } else {
    // Use a simple two-field struct if there isn't one already.
    EltTy = StructType::get(IRB.getInt32Ty(), PointerType::getUnqual(FnTy),
                            nullptr);
  }

  // Build a 2 or 3 field global_ctor entry.  We don't take a comdat key.
  Constant *CSVals[3];
  CSVals[0] = IRB.getInt32(Priority);
  CSVals[1] = F;
  // FIXME: Drop support for the two element form in LLVM 4.0.
  if (EltTy->getNumElements() >= 3)
    CSVals[2] = llvm::Constant::getNullValue(IRB.getInt8PtrTy());
  Constant *RuntimeCtorInit =
      ConstantStruct::get(EltTy, makeArrayRef(CSVals, EltTy->getNumElements()));

  CurrentCtors.push_back(RuntimeCtorInit);

  // Create a new initializer.
  ArrayType *AT = ArrayType::get(EltTy, CurrentCtors.size());
  Constant *NewInit = ConstantArray::get(AT, CurrentCtors);

  // Create the new global variable and replace all uses of
  // the old global variable with the new one.
  (void)new GlobalVariable(M, NewInit->getType(), false,
                           GlobalValue::AppendingLinkage, NewInit, Array);
}

void llvm::appendToGlobalCtors(Module &M, Function *F, int Priority) {
  appendToGlobalArray("llvm.global_ctors", M, F, Priority);
}

void llvm::appendToGlobalDtors(Module &M, Function *F, int Priority) {
  appendToGlobalArray("llvm.global_dtors", M, F, Priority);
}

GlobalVariable *
llvm::collectUsedGlobalVariables(Module &M, SmallPtrSetImpl<GlobalValue *> &Set,
                                 bool CompilerUsed) {
  const char *Name = CompilerUsed ? "llvm.compiler.used" : "llvm.used";
  GlobalVariable *GV = M.getGlobalVariable(Name);
  if (!GV || !GV->hasInitializer())
    return GV;

  const ConstantArray *Init = cast<ConstantArray>(GV->getInitializer());
  for (unsigned I = 0, E = Init->getNumOperands(); I != E; ++I) {
    Value *Op = Init->getOperand(I);
    GlobalValue *G = cast<GlobalValue>(Op->stripPointerCastsNoFollowAliases());
    Set.insert(G);
  }
  return GV;
}

Function *llvm::checkSanitizerInterfaceFunction(Constant *FuncOrBitcast) {
  if (isa<Function>(FuncOrBitcast))
    return cast<Function>(FuncOrBitcast);
  FuncOrBitcast->dump();
  std::string Err;
  raw_string_ostream Stream(Err);
  Stream << "Sanitizer interface function redefined: " << *FuncOrBitcast;
  report_fatal_error(Err);
}

std::pair<Function *, Function *> llvm::createSanitizerCtorAndInitFunctions(
    Module &M, StringRef CtorName, StringRef InitName,
    ArrayRef<Type *> InitArgTypes, ArrayRef<Value *> InitArgs) {
  assert(!InitName.empty() && "Expected init function name");
  assert(InitArgTypes.size() == InitArgTypes.size() &&
         "Sanitizer's init function expects different number of arguments");
  Function *Ctor = Function::Create(
      FunctionType::get(Type::getVoidTy(M.getContext()), false),
      GlobalValue::InternalLinkage, CtorName, &M);
  BasicBlock *CtorBB = BasicBlock::Create(M.getContext(), "", Ctor);
  IRBuilder<> IRB(ReturnInst::Create(M.getContext(), CtorBB));
  Function *InitFunction =
      checkSanitizerInterfaceFunction(M.getOrInsertFunction(
          InitName, FunctionType::get(IRB.getVoidTy(), InitArgTypes, false),
          AttributeSet()));
  InitFunction->setLinkage(Function::ExternalLinkage);
  IRB.CreateCall(InitFunction, InitArgs);
  return std::make_pair(Ctor, InitFunction);
}

