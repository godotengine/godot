//===----- IRTransformLayer.h - Run all IR through a functor ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Run all IR passed in through a user supplied functor.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_IRTRANSFORMLAYER_H
#define LLVM_EXECUTIONENGINE_ORC_IRTRANSFORMLAYER_H

#include "JITSymbol.h"

namespace llvm {
namespace orc {

/// @brief IR mutating layer.
///
///   This layer accepts sets of LLVM IR Modules (via addModuleSet). It
/// immediately applies the user supplied functor to each module, then adds
/// the set of transformed modules to the layer below.
template <typename BaseLayerT, typename TransformFtor>
class IRTransformLayer {
public:
  /// @brief Handle to a set of added modules.
  typedef typename BaseLayerT::ModuleSetHandleT ModuleSetHandleT;

  /// @brief Construct an IRTransformLayer with the given BaseLayer
  IRTransformLayer(BaseLayerT &BaseLayer,
                   TransformFtor Transform = TransformFtor())
    : BaseLayer(BaseLayer), Transform(std::move(Transform)) {}

  /// @brief Apply the transform functor to each module in the module set, then
  ///        add the resulting set of modules to the base layer, along with the
  ///        memory manager and symbol resolver.
  ///
  /// @return A handle for the added modules.
  template <typename ModuleSetT, typename MemoryManagerPtrT,
            typename SymbolResolverPtrT>
  ModuleSetHandleT addModuleSet(ModuleSetT Ms,
                                MemoryManagerPtrT MemMgr,
                                SymbolResolverPtrT Resolver) {

    for (auto I = Ms.begin(), E = Ms.end(); I != E; ++I)
      *I = Transform(std::move(*I));

    return BaseLayer.addModuleSet(std::move(Ms), std::move(MemMgr),
                                  std::move(Resolver));
  }

  /// @brief Remove the module set associated with the handle H.
  void removeModuleSet(ModuleSetHandleT H) { BaseLayer.removeModuleSet(H); }

  /// @brief Search for the given named symbol.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it exists.
  JITSymbol findSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
    return BaseLayer.findSymbol(Name, ExportedSymbolsOnly);
  }

  /// @brief Get the address of the given symbol in the context of the set of
  ///        modules represented by the handle H. This call is forwarded to the
  ///        base layer's implementation.
  /// @param H The handle for the module set to search in.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it is found in the
  ///         given module set.
  JITSymbol findSymbolIn(ModuleSetHandleT H, const std::string &Name,
                         bool ExportedSymbolsOnly) {
    return BaseLayer.findSymbolIn(H, Name, ExportedSymbolsOnly);
  }

  /// @brief Immediately emit and finalize the module set represented by the
  ///        given handle.
  /// @param H Handle for module set to emit/finalize.
  void emitAndFinalize(ModuleSetHandleT H) {
    BaseLayer.emitAndFinalize(H);
  }

  /// @brief Access the transform functor directly.
  TransformFtor& getTransform() { return Transform; }

  /// @brief Access the mumate functor directly.
  const TransformFtor& getTransform() const { return Transform; }

private:
  BaseLayerT &BaseLayer;
  TransformFtor Transform;
};

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_IRTRANSFORMLAYER_H
