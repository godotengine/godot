//===-- LambdaResolverMM - Redirect symbol lookup via a functor -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//   Defines a RuntimeDyld::SymbolResolver subclass that uses a user-supplied
// functor for symbol resolution.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_LAMBDARESOLVER_H
#define LLVM_EXECUTIONENGINE_ORC_LAMBDARESOLVER_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include <memory>
#include <vector>

namespace llvm {
namespace orc {

template <typename ExternalLookupFtorT, typename DylibLookupFtorT>
class LambdaResolver : public RuntimeDyld::SymbolResolver {
public:

  LambdaResolver(ExternalLookupFtorT ExternalLookupFtor,
                 DylibLookupFtorT DylibLookupFtor)
      : ExternalLookupFtor(ExternalLookupFtor),
        DylibLookupFtor(DylibLookupFtor) {}

  RuntimeDyld::SymbolInfo findSymbol(const std::string &Name) final {
    return ExternalLookupFtor(Name);
  }

  RuntimeDyld::SymbolInfo
  findSymbolInLogicalDylib(const std::string &Name) final {
    return DylibLookupFtor(Name);
  }

private:
  ExternalLookupFtorT ExternalLookupFtor;
  DylibLookupFtorT DylibLookupFtor;
};

template <typename ExternalLookupFtorT,
          typename DylibLookupFtorT>
std::unique_ptr<LambdaResolver<ExternalLookupFtorT, DylibLookupFtorT>>
createLambdaResolver(ExternalLookupFtorT ExternalLookupFtor,
                     DylibLookupFtorT DylibLookupFtor) {
  typedef LambdaResolver<ExternalLookupFtorT, DylibLookupFtorT> LR;
  return make_unique<LR>(std::move(ExternalLookupFtor),
                         std::move(DylibLookupFtor));
}

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_LAMBDARESOLVER_H
