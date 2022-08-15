//===---------- NullResolver.cpp - Reject symbol lookup requests ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/NullResolver.h"

#include "llvm/Support/ErrorHandling.h"

namespace llvm {
namespace orc {

RuntimeDyld::SymbolInfo NullResolver::findSymbol(const std::string &Name) {
  llvm_unreachable("Unexpected cross-object symbol reference");
}

RuntimeDyld::SymbolInfo
NullResolver::findSymbolInLogicalDylib(const std::string &Name) {
  llvm_unreachable("Unexpected cross-object symbol reference");
}

} // End namespace orc.
} // End namespace llvm.
