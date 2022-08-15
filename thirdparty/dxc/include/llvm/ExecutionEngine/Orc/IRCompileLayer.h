//===------ IRCompileLayer.h -- Eagerly compile IR for JIT ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains the definition for a basic, eagerly compiling layer of the JIT.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_IRCOMPILELAYER_H
#define LLVM_EXECUTIONENGINE_ORC_IRCOMPILELAYER_H

#include "JITSymbol.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/Object/ObjectFile.h"
#include <memory>

namespace llvm {
namespace orc {

/// @brief Eager IR compiling layer.
///
///   This layer accepts sets of LLVM IR Modules (via addModuleSet). It
/// immediately compiles each IR module to an object file (each IR Module is
/// compiled separately). The resulting set of object files is then added to
/// the layer below, which must implement the object layer concept.
template <typename BaseLayerT> class IRCompileLayer {
public:
  typedef std::function<object::OwningBinary<object::ObjectFile>(Module &)>
      CompileFtor;

private:
  typedef typename BaseLayerT::ObjSetHandleT ObjSetHandleT;

  typedef std::vector<std::unique_ptr<object::ObjectFile>> OwningObjectVec;
  typedef std::vector<std::unique_ptr<MemoryBuffer>> OwningBufferVec;

public:
  /// @brief Handle to a set of compiled modules.
  typedef ObjSetHandleT ModuleSetHandleT;

  /// @brief Construct an IRCompileLayer with the given BaseLayer, which must
  ///        implement the ObjectLayer concept.
  IRCompileLayer(BaseLayerT &BaseLayer, CompileFtor Compile)
      : BaseLayer(BaseLayer), Compile(std::move(Compile)), ObjCache(nullptr) {}

  /// @brief Set an ObjectCache to query before compiling.
  void setObjectCache(ObjectCache *NewCache) { ObjCache = NewCache; }

  /// @brief Compile each module in the given module set, then add the resulting
  ///        set of objects to the base layer along with the memory manager and
  ///        symbol resolver.
  ///
  /// @return A handle for the added modules.
  template <typename ModuleSetT, typename MemoryManagerPtrT,
            typename SymbolResolverPtrT>
  ModuleSetHandleT addModuleSet(ModuleSetT Ms,
                                MemoryManagerPtrT MemMgr,
                                SymbolResolverPtrT Resolver) {
    OwningObjectVec Objects;
    OwningBufferVec Buffers;

    for (const auto &M : Ms) {
      std::unique_ptr<object::ObjectFile> Object;
      std::unique_ptr<MemoryBuffer> Buffer;

      if (ObjCache)
        std::tie(Object, Buffer) = tryToLoadFromObjectCache(*M).takeBinary();

      if (!Object) {
        std::tie(Object, Buffer) = Compile(*M).takeBinary();
        if (ObjCache)
          ObjCache->notifyObjectCompiled(&*M, Buffer->getMemBufferRef());
      }

      Objects.push_back(std::move(Object));
      Buffers.push_back(std::move(Buffer));
    }

    ModuleSetHandleT H =
      BaseLayer.addObjectSet(Objects, std::move(MemMgr), std::move(Resolver));

    BaseLayer.takeOwnershipOfBuffers(H, std::move(Buffers));

    return H;
  }

  /// @brief Remove the module set associated with the handle H.
  void removeModuleSet(ModuleSetHandleT H) { BaseLayer.removeObjectSet(H); }

  /// @brief Search for the given named symbol.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it exists.
  JITSymbol findSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
    return BaseLayer.findSymbol(Name, ExportedSymbolsOnly);
  }

  /// @brief Get the address of the given symbol in the context of the set of
  ///        compiled modules represented by the handle H. This call is
  ///        forwarded to the base layer's implementation.
  /// @param H The handle for the module set to search in.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it is found in the
  ///         given module set.
  JITSymbol findSymbolIn(ModuleSetHandleT H, const std::string &Name,
                         bool ExportedSymbolsOnly) {
    return BaseLayer.findSymbolIn(H, Name, ExportedSymbolsOnly);
  }

  /// @brief Immediately emit and finalize the moduleOB set represented by the
  ///        given handle.
  /// @param H Handle for module set to emit/finalize.
  void emitAndFinalize(ModuleSetHandleT H) {
    BaseLayer.emitAndFinalize(H);
  }

private:
  object::OwningBinary<object::ObjectFile>
  tryToLoadFromObjectCache(const Module &M) {
    std::unique_ptr<MemoryBuffer> ObjBuffer = ObjCache->getObject(&M);
    if (!ObjBuffer)
      return object::OwningBinary<object::ObjectFile>();

    ErrorOr<std::unique_ptr<object::ObjectFile>> Obj =
        object::ObjectFile::createObjectFile(ObjBuffer->getMemBufferRef());
    if (!Obj)
      return object::OwningBinary<object::ObjectFile>();

    return object::OwningBinary<object::ObjectFile>(std::move(*Obj),
                                                    std::move(ObjBuffer));
  }

  BaseLayerT &BaseLayer;
  CompileFtor Compile;
  ObjectCache *ObjCache;
};

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_IRCOMPILINGLAYER_H
