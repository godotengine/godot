//===- ARCRuntimeEntryPoints.h - ObjC ARC Optimization --*- C++ -*---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains a class ARCRuntimeEntryPoints for use in
/// creating/managing references to entry points to the arc objective c runtime.
///
/// WARNING: This file knows about certain library functions. It recognizes them
/// by name, and hardwires knowledge of their semantics.
///
/// WARNING: This file knows about how certain Objective-C library functions are
/// used. Naive LLVM IR transformations which would otherwise be
/// behavior-preserving may break these assumptions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_OBJCARC_ARCRUNTIMEENTRYPOINTS_H
#define LLVM_LIB_TRANSFORMS_OBJCARC_ARCRUNTIMEENTRYPOINTS_H

#include "ObjCARC.h"

namespace llvm {
namespace objcarc {

enum class ARCRuntimeEntryPointKind {
  AutoreleaseRV,
  Release,
  Retain,
  RetainBlock,
  Autorelease,
  StoreStrong,
  RetainRV,
  RetainAutorelease,
  RetainAutoreleaseRV,
};

/// Declarations for ObjC runtime functions and constants. These are initialized
/// lazily to avoid cluttering up the Module with unused declarations.
class ARCRuntimeEntryPoints {
public:
  ARCRuntimeEntryPoints() : TheModule(nullptr),
                            AutoreleaseRV(nullptr),
                            Release(nullptr),
                            Retain(nullptr),
                            RetainBlock(nullptr),
                            Autorelease(nullptr),
                            StoreStrong(nullptr),
                            RetainRV(nullptr),
                            RetainAutorelease(nullptr),
                            RetainAutoreleaseRV(nullptr) { }

  void init(Module *M) {
    TheModule = M;
    AutoreleaseRV = nullptr;
    Release = nullptr;
    Retain = nullptr;
    RetainBlock = nullptr;
    Autorelease = nullptr;
    StoreStrong = nullptr;
    RetainRV = nullptr;
    RetainAutorelease = nullptr;
    RetainAutoreleaseRV = nullptr;
  }

  Constant *get(ARCRuntimeEntryPointKind kind) {
    assert(TheModule != nullptr && "Not initialized.");

    switch (kind) {
    case ARCRuntimeEntryPointKind::AutoreleaseRV:
      return getI8XRetI8XEntryPoint(AutoreleaseRV,
                                    "objc_autoreleaseReturnValue", true);
    case ARCRuntimeEntryPointKind::Release:
      return getVoidRetI8XEntryPoint(Release, "objc_release");
    case ARCRuntimeEntryPointKind::Retain:
      return getI8XRetI8XEntryPoint(Retain, "objc_retain", true);
    case ARCRuntimeEntryPointKind::RetainBlock:
      return getI8XRetI8XEntryPoint(RetainBlock, "objc_retainBlock", false);
    case ARCRuntimeEntryPointKind::Autorelease:
      return getI8XRetI8XEntryPoint(Autorelease, "objc_autorelease", true);
    case ARCRuntimeEntryPointKind::StoreStrong:
      return getI8XRetI8XXI8XEntryPoint(StoreStrong, "objc_storeStrong");
    case ARCRuntimeEntryPointKind::RetainRV:
      return getI8XRetI8XEntryPoint(RetainRV,
                                    "objc_retainAutoreleasedReturnValue", true);
    case ARCRuntimeEntryPointKind::RetainAutorelease:
      return getI8XRetI8XEntryPoint(RetainAutorelease, "objc_retainAutorelease",
                                    true);
    case ARCRuntimeEntryPointKind::RetainAutoreleaseRV:
      return getI8XRetI8XEntryPoint(RetainAutoreleaseRV,
                                    "objc_retainAutoreleaseReturnValue", true);
    }

    llvm_unreachable("Switch should be a covered switch.");
  }

private:
  /// Cached reference to the module which we will insert declarations into.
  Module *TheModule;

  /// Declaration for ObjC runtime function objc_autoreleaseReturnValue.
  Constant *AutoreleaseRV;
  /// Declaration for ObjC runtime function objc_release.
  Constant *Release;
  /// Declaration for ObjC runtime function objc_retain.
  Constant *Retain;
  /// Declaration for ObjC runtime function objc_retainBlock.
  Constant *RetainBlock;
  /// Declaration for ObjC runtime function objc_autorelease.
  Constant *Autorelease;
  /// Declaration for objc_storeStrong().
  Constant *StoreStrong;
  /// Declaration for objc_retainAutoreleasedReturnValue().
  Constant *RetainRV;
  /// Declaration for objc_retainAutorelease().
  Constant *RetainAutorelease;
  /// Declaration for objc_retainAutoreleaseReturnValue().
  Constant *RetainAutoreleaseRV;

  Constant *getVoidRetI8XEntryPoint(Constant *&Decl,
                                    const char *Name) {
    if (Decl)
      return Decl;

    LLVMContext &C = TheModule->getContext();
    Type *Params[] = { PointerType::getUnqual(Type::getInt8Ty(C)) };
    AttributeSet Attr =
      AttributeSet().addAttribute(C, AttributeSet::FunctionIndex,
                                  Attribute::NoUnwind);
    FunctionType *Fty = FunctionType::get(Type::getVoidTy(C), Params,
                                          /*isVarArg=*/false);
    return Decl = TheModule->getOrInsertFunction(Name, Fty, Attr);
  }

  Constant *getI8XRetI8XEntryPoint(Constant *& Decl,
                                   const char *Name,
                                   bool NoUnwind = false) {
    if (Decl)
      return Decl;

    LLVMContext &C = TheModule->getContext();
    Type *I8X = PointerType::getUnqual(Type::getInt8Ty(C));
    Type *Params[] = { I8X };
    FunctionType *Fty = FunctionType::get(I8X, Params, /*isVarArg=*/false);
    AttributeSet Attr = AttributeSet();

    if (NoUnwind)
      Attr = Attr.addAttribute(C, AttributeSet::FunctionIndex,
                               Attribute::NoUnwind);

    return Decl = TheModule->getOrInsertFunction(Name, Fty, Attr);
  }

  Constant *getI8XRetI8XXI8XEntryPoint(Constant *&Decl,
                                       const char *Name) {
    if (Decl)
      return Decl;

    LLVMContext &C = TheModule->getContext();
    Type *I8X = PointerType::getUnqual(Type::getInt8Ty(C));
    Type *I8XX = PointerType::getUnqual(I8X);
    Type *Params[] = { I8XX, I8X };

    AttributeSet Attr =
      AttributeSet().addAttribute(C, AttributeSet::FunctionIndex,
                                  Attribute::NoUnwind);
    Attr = Attr.addAttribute(C, 1, Attribute::NoCapture);

    FunctionType *Fty = FunctionType::get(Type::getVoidTy(C), Params,
                                          /*isVarArg=*/false);

    return Decl = TheModule->getOrInsertFunction(Name, Fty, Attr);
  }

}; // class ARCRuntimeEntryPoints

} // namespace objcarc
} // namespace llvm

#endif
