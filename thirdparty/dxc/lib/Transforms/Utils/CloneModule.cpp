//===- CloneModule.cpp - Clone an entire module ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CloneModule interface which makes a copy of an
// entire module.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm-c/Core.h"
using namespace llvm;

/// CloneModule - Return an exact copy of the specified module.  This is not as
/// easy as it might seem because we have to worry about making copies of global
/// variables and functions, and making their (initializers and references,
/// respectively) refer to the right globals.
///
Module *llvm::CloneModule(const Module *M) {
  // Create the value map that maps things from the old module over to the new
  // module.
  ValueToValueMapTy VMap;
  return CloneModule(M, VMap);
}

Module *llvm::CloneModule(const Module *M, ValueToValueMapTy &VMap) {
  // First off, we need to create the new module.
  Module *New = new Module(M->getModuleIdentifier(), M->getContext());
  New->setDataLayout(M->getDataLayout());
  New->setTargetTriple(M->getTargetTriple());
  New->setModuleInlineAsm(M->getModuleInlineAsm());
   
  // Loop over all of the global variables, making corresponding globals in the
  // new module.  Here we add them to the VMap and to the new Module.  We
  // don't worry about attributes or initializers, they will come later.
  //
  for (Module::const_global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I) {
    GlobalVariable *GV = new GlobalVariable(*New, 
                                            I->getType()->getElementType(),
                                            I->isConstant(), I->getLinkage(),
                                            (Constant*) nullptr, I->getName(),
                                            (GlobalVariable*) nullptr,
                                            I->getThreadLocalMode(),
                                            I->getType()->getAddressSpace());
    GV->copyAttributesFrom(I);
    VMap[I] = GV;
  }

  // Loop over the functions in the module, making external functions as before
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I) {
    Function *NF =
      Function::Create(cast<FunctionType>(I->getType()->getElementType()),
                       I->getLinkage(), I->getName(), New);
    NF->copyAttributesFrom(I);
    VMap[I] = NF;
  }

  // Loop over the aliases in the module
  for (Module::const_alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E; ++I) {
    auto *PTy = cast<PointerType>(I->getType());
    auto *GA = GlobalAlias::create(PTy, I->getLinkage(), I->getName(), New);
    GA->copyAttributesFrom(I);
    VMap[I] = GA;
  }
  
  // Now that all of the things that global variable initializer can refer to
  // have been created, loop through and copy the global variable referrers
  // over...  We also set the attributes on the global now.
  //
  for (Module::const_global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I) {
    GlobalVariable *GV = cast<GlobalVariable>(VMap[I]);
    if (I->hasInitializer())
      GV->setInitializer(MapValue(I->getInitializer(), VMap));
  }

  // Similarly, copy over function bodies now...
  //
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I) {
    Function *F = cast<Function>(VMap[I]);
    if (!I->isDeclaration()) {
      Function::arg_iterator DestI = F->arg_begin();
      for (Function::const_arg_iterator J = I->arg_begin(); J != I->arg_end();
           ++J) {
        DestI->setName(J->getName());
        VMap[J] = DestI++;
      }

      SmallVector<ReturnInst*, 8> Returns;  // Ignore returns cloned.
      CloneFunctionInto(F, I, VMap, /*ModuleLevelChanges=*/true, Returns);

    }

    if (I->hasPersonalityFn())
      F->setPersonalityFn(MapValue(I->getPersonalityFn(), VMap));
  }

  // And aliases
  for (Module::const_alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E; ++I) {
    GlobalAlias *GA = cast<GlobalAlias>(VMap[I]);
    if (const Constant *C = I->getAliasee())
      GA->setAliasee(MapValue(C, VMap));
  }

  // And named metadata....
  for (Module::const_named_metadata_iterator I = M->named_metadata_begin(),
         E = M->named_metadata_end(); I != E; ++I) {
    const NamedMDNode &NMD = *I;
    NamedMDNode *NewNMD = New->getOrInsertNamedMetadata(NMD.getName());
    for (unsigned i = 0, e = NMD.getNumOperands(); i != e; ++i)
      NewNMD->addOperand(MapMetadata(NMD.getOperand(i), VMap));
  }

  return New;
}

// extern "C" {  // HLSL Change -Don't use c linkage.

LLVMModuleRef LLVMCloneModule(LLVMModuleRef M) {
  return wrap(CloneModule(unwrap(M)));
}

// }             // HLSL Change -Don't use c linkage.
