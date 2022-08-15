//===---- IndirectionUtils.cpp - Utilities for call indirection in Orc ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <set>
#include <sstream>

namespace llvm {
namespace orc {

Constant* createIRTypedAddress(FunctionType &FT, TargetAddress Addr) {
  Constant *AddrIntVal =
    ConstantInt::get(Type::getInt64Ty(FT.getContext()), Addr);
  Constant *AddrPtrVal =
    ConstantExpr::getCast(Instruction::IntToPtr, AddrIntVal,
                          PointerType::get(&FT, 0));
  return AddrPtrVal;
}

GlobalVariable* createImplPointer(PointerType &PT, Module &M,
                                  const Twine &Name, Constant *Initializer) {
  auto IP = new GlobalVariable(M, &PT, false, GlobalValue::ExternalLinkage,
                               Initializer, Name, nullptr,
                               GlobalValue::NotThreadLocal, 0, true);
  IP->setVisibility(GlobalValue::HiddenVisibility);
  return IP;
}

void makeStub(Function &F, GlobalVariable &ImplPointer) {
  assert(F.isDeclaration() && "Can't turn a definition into a stub.");
  assert(F.getParent() && "Function isn't in a module.");
  Module &M = *F.getParent();
  BasicBlock *EntryBlock = BasicBlock::Create(M.getContext(), "entry", &F);
  IRBuilder<> Builder(EntryBlock);
  LoadInst *ImplAddr = Builder.CreateLoad(&ImplPointer);
  std::vector<Value*> CallArgs;
  for (auto &A : F.args())
    CallArgs.push_back(&A);
  CallInst *Call = Builder.CreateCall(ImplAddr, CallArgs);
  Call->setTailCall();
  Call->setAttributes(F.getAttributes());
  if (F.getReturnType()->isVoidTy())
    Builder.CreateRetVoid();
  else
    Builder.CreateRet(Call);
}

// Utility class for renaming global values and functions during partitioning.
class GlobalRenamer {
public:

  static bool needsRenaming(const Value &New) {
    if (!New.hasName() || New.getName().startswith("\01L"))
      return true;
    return false;
  }

  const std::string& getRename(const Value &Orig) {
    // See if we have a name for this global.
    {
      auto I = Names.find(&Orig);
      if (I != Names.end())
        return I->second;
    }

    // Nope. Create a new one.
    // FIXME: Use a more robust uniquing scheme. (This may blow up if the user
    //        writes a "__orc_anon[[:digit:]]* method).
    unsigned ID = Names.size();
    std::ostringstream NameStream;
    NameStream << "__orc_anon" << ID++;
    auto I = Names.insert(std::make_pair(&Orig, NameStream.str()));
    return I.first->second;
  }
private:
  DenseMap<const Value*, std::string> Names;
};

static void raiseVisibilityOnValue(GlobalValue &V, GlobalRenamer &R) {
  if (V.hasLocalLinkage()) {
    if (R.needsRenaming(V))
      V.setName(R.getRename(V));
    V.setLinkage(GlobalValue::ExternalLinkage);
    V.setVisibility(GlobalValue::HiddenVisibility);
  }
  V.setUnnamedAddr(false);
  assert(!R.needsRenaming(V) && "Invalid global name.");
}

void makeAllSymbolsExternallyAccessible(Module &M) {
  GlobalRenamer Renamer;

  for (auto &F : M)
    raiseVisibilityOnValue(F, Renamer);

  for (auto &GV : M.globals())
    raiseVisibilityOnValue(GV, Renamer);
}

Function* cloneFunctionDecl(Module &Dst, const Function &F,
                            ValueToValueMapTy *VMap) {
  assert(F.getParent() != &Dst && "Can't copy decl over existing function.");
  Function *NewF =
    Function::Create(cast<FunctionType>(F.getType()->getElementType()),
                     F.getLinkage(), F.getName(), &Dst);
  NewF->copyAttributesFrom(&F);

  if (VMap) {
    (*VMap)[&F] = NewF;
    auto NewArgI = NewF->arg_begin();
    for (auto ArgI = F.arg_begin(), ArgE = F.arg_end(); ArgI != ArgE;
         ++ArgI, ++NewArgI)
      (*VMap)[ArgI] = NewArgI;
  }

  return NewF;
}

void moveFunctionBody(Function &OrigF, ValueToValueMapTy &VMap,
                      ValueMaterializer *Materializer,
                      Function *NewF) {
  assert(!OrigF.isDeclaration() && "Nothing to move");
  if (!NewF)
    NewF = cast<Function>(VMap[&OrigF]);
  else
    assert(VMap[&OrigF] == NewF && "Incorrect function mapping in VMap.");
  assert(NewF && "Function mapping missing from VMap.");
  assert(NewF->getParent() != OrigF.getParent() &&
         "moveFunctionBody should only be used to move bodies between "
         "modules.");

  SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
  CloneFunctionInto(NewF, &OrigF, VMap, /*ModuleLevelChanges=*/true, Returns,
                    "", nullptr, nullptr, Materializer);
  OrigF.deleteBody();
}

GlobalVariable* cloneGlobalVariableDecl(Module &Dst, const GlobalVariable &GV,
                                        ValueToValueMapTy *VMap) {
  assert(GV.getParent() != &Dst && "Can't copy decl over existing global var.");
  GlobalVariable *NewGV = new GlobalVariable(
      Dst, GV.getType()->getElementType(), GV.isConstant(),
      GV.getLinkage(), nullptr, GV.getName(), nullptr,
      GV.getThreadLocalMode(), GV.getType()->getAddressSpace());
  NewGV->copyAttributesFrom(&GV);
  if (VMap)
    (*VMap)[&GV] = NewGV;
  return NewGV;
}

void moveGlobalVariableInitializer(GlobalVariable &OrigGV,
                                   ValueToValueMapTy &VMap,
                                   ValueMaterializer *Materializer,
                                   GlobalVariable *NewGV) {
  assert(OrigGV.hasInitializer() && "Nothing to move");
  if (!NewGV)
    NewGV = cast<GlobalVariable>(VMap[&OrigGV]);
  else
    assert(VMap[&OrigGV] == NewGV &&
           "Incorrect global variable mapping in VMap.");
  assert(NewGV->getParent() != OrigGV.getParent() &&
         "moveGlobalVariable should only be used to move initializers between "
         "modules");

  NewGV->setInitializer(MapValue(OrigGV.getInitializer(), VMap, RF_None,
                                 nullptr, Materializer));
}

} // End namespace orc.
} // End namespace llvm.
