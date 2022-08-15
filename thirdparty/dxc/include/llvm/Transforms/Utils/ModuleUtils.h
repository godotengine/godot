//===-- ModuleUtils.h - Functions to manipulate Modules ---------*- C++ -*-===//
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

#ifndef LLVM_TRANSFORMS_UTILS_MODULEUTILS_H
#define LLVM_TRANSFORMS_UTILS_MODULEUTILS_H

#include "llvm/ADT/ArrayRef.h"
#include <utility> // for std::pair

namespace llvm {

class Module;
class Function;
class GlobalValue;
class GlobalVariable;
class Constant;
class StringRef;
class Value;
class Type;
template <class PtrType> class SmallPtrSetImpl;

/// Append F to the list of global ctors of module M with the given Priority.
/// This wraps the function in the appropriate structure and stores it along
/// side other global constructors. For details see
/// http://llvm.org/docs/LangRef.html#intg_global_ctors
void appendToGlobalCtors(Module &M, Function *F, int Priority);

/// Same as appendToGlobalCtors(), but for global dtors.
void appendToGlobalDtors(Module &M, Function *F, int Priority);

/// \brief Given "llvm.used" or "llvm.compiler.used" as a global name, collect
/// the initializer elements of that global in Set and return the global itself.
GlobalVariable *collectUsedGlobalVariables(Module &M,
                                           SmallPtrSetImpl<GlobalValue *> &Set,
                                           bool CompilerUsed);

// Validate the result of Module::getOrInsertFunction called for an interface
// function of given sanitizer. If the instrumented module defines a function
// with the same name, their prototypes must match, otherwise
// getOrInsertFunction returns a bitcast.
Function *checkSanitizerInterfaceFunction(Constant *FuncOrBitcast);

/// \brief Creates sanitizer constructor function, and calls sanitizer's init
/// function from it.
/// \return Returns pair of pointers to constructor, and init functions
/// respectively.
std::pair<Function *, Function *> createSanitizerCtorAndInitFunctions(
    Module &M, StringRef CtorName, StringRef InitName,
    ArrayRef<Type *> InitArgTypes, ArrayRef<Value *> InitArgs);
} // End llvm namespace

#endif //  LLVM_TRANSFORMS_UTILS_MODULEUTILS_H
