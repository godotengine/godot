//===-- llvm/Support/CodeGen.h - CodeGen Concepts ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file define some types which define code generation concepts. For
// example, relocation model.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CODEGEN_H
#define LLVM_SUPPORT_CODEGEN_H

#include "llvm-c/TargetMachine.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

  // Relocation model types.
  namespace Reloc {
    enum Model { Default, Static, PIC_, DynamicNoPIC };
  }

  // Code model types.
  namespace CodeModel {
    enum Model { Default, JITDefault, Small, Kernel, Medium, Large };
  }

  namespace PICLevel {
    enum Level { Default=0, Small=1, Large=2 };
  }

  // TLS models.
  namespace TLSModel {
    enum Model {
      GeneralDynamic,
      LocalDynamic,
      InitialExec,
      LocalExec
    };
  }

  // Code generation optimization level.
  namespace CodeGenOpt {
    enum Level {
      None,        // -O0
      Less,        // -O1
      Default,     // -O2, -Os
      Aggressive   // -O3
    };
  }

  // Create wrappers for C Binding types (see CBindingWrapping.h).
  inline CodeModel::Model unwrap(LLVMCodeModel Model) {
    switch (Model) {
      case LLVMCodeModelDefault:
        return CodeModel::Default;
      case LLVMCodeModelJITDefault:
        return CodeModel::JITDefault;
      case LLVMCodeModelSmall:
        return CodeModel::Small;
      case LLVMCodeModelKernel:
        return CodeModel::Kernel;
      case LLVMCodeModelMedium:
        return CodeModel::Medium;
      case LLVMCodeModelLarge:
        return CodeModel::Large;
    }
    return CodeModel::Default;
  }

  inline LLVMCodeModel wrap(CodeModel::Model Model) {
    switch (Model) {
      case CodeModel::Default:
        return LLVMCodeModelDefault;
      case CodeModel::JITDefault:
        return LLVMCodeModelJITDefault;
      case CodeModel::Small:
        return LLVMCodeModelSmall;
      case CodeModel::Kernel:
        return LLVMCodeModelKernel;
      case CodeModel::Medium:
        return LLVMCodeModelMedium;
      case CodeModel::Large:
        return LLVMCodeModelLarge;
    }
    llvm_unreachable("Bad CodeModel!");
  }
}  // end llvm namespace

#endif
