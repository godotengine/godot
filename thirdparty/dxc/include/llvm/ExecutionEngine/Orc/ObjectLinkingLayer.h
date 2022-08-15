//===- ObjectLinkingLayer.h - Add object files to a JIT process -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains the definition for the object layer of the JIT.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_OBJECTLINKINGLAYER_H
#define LLVM_EXECUTIONENGINE_ORC_OBJECTLINKINGLAYER_H

#include "JITSymbol.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include <list>
#include <memory>

namespace llvm {
namespace orc {

class ObjectLinkingLayerBase {
protected:

  /// @brief Holds a set of objects to be allocated/linked as a unit in the JIT.
  ///
  /// An instance of this class will be created for each set of objects added
  /// via JITObjectLayer::addObjectSet. Deleting the instance (via
  /// removeObjectSet) frees its memory, removing all symbol definitions that
  /// had been provided by this instance. Higher level layers are responsible
  /// for taking any action required to handle the missing symbols.
  class LinkedObjectSet {
    LinkedObjectSet(const LinkedObjectSet&) = delete;
    void operator=(const LinkedObjectSet&) = delete;
  public:
    LinkedObjectSet(RuntimeDyld::MemoryManager &MemMgr,
                    RuntimeDyld::SymbolResolver &Resolver)
        : RTDyld(llvm::make_unique<RuntimeDyld>(MemMgr, Resolver)),
          State(Raw) {}

    virtual ~LinkedObjectSet() {}

    std::unique_ptr<RuntimeDyld::LoadedObjectInfo>
    addObject(const object::ObjectFile &Obj) {
      return RTDyld->loadObject(Obj);
    }

    RuntimeDyld::SymbolInfo getSymbol(StringRef Name) const {
      return RTDyld->getSymbol(Name);
    }

    bool NeedsFinalization() const { return (State == Raw); }

    virtual void Finalize() = 0;

    void mapSectionAddress(const void *LocalAddress, TargetAddress TargetAddr) {
      assert((State != Finalized) &&
             "Attempting to remap sections for finalized objects.");
      RTDyld->mapSectionAddress(LocalAddress, TargetAddr);
    }

    void takeOwnershipOfBuffer(std::unique_ptr<MemoryBuffer> B) {
      OwnedBuffers.push_back(std::move(B));
    }

  protected:
    std::unique_ptr<RuntimeDyld> RTDyld;
    enum { Raw, Finalizing, Finalized } State;

    // FIXME: This ownership hack only exists because RuntimeDyldELF still
    //        wants to be able to inspect the original object when resolving
    //        relocations. As soon as that can be fixed this should be removed.
    std::vector<std::unique_ptr<MemoryBuffer>> OwnedBuffers;
  };

  typedef std::list<std::unique_ptr<LinkedObjectSet>> LinkedObjectSetListT;

public:
  /// @brief Handle to a set of loaded objects.
  typedef LinkedObjectSetListT::iterator ObjSetHandleT;

  // Ownership hack.
  // FIXME: Remove this as soon as RuntimeDyldELF can apply relocations without
  //        referencing the original object.
  template <typename OwningMBSet>
  void takeOwnershipOfBuffers(ObjSetHandleT H, OwningMBSet MBs) {
    for (auto &MB : MBs)
      (*H)->takeOwnershipOfBuffer(std::move(MB));
  }

};

/// @brief Default (no-op) action to perform when loading objects.
class DoNothingOnNotifyLoaded {
public:
  template <typename ObjSetT, typename LoadResult>
  void operator()(ObjectLinkingLayerBase::ObjSetHandleT, const ObjSetT &,
                  const LoadResult &) {}
};

/// @brief Bare bones object linking layer.
///
///   This class is intended to be used as the base layer for a JIT. It allows
/// object files to be loaded into memory, linked, and the addresses of their
/// symbols queried. All objects added to this layer can see each other's
/// symbols.
template <typename NotifyLoadedFtor = DoNothingOnNotifyLoaded>
class ObjectLinkingLayer : public ObjectLinkingLayerBase {
private:

  template <typename MemoryManagerPtrT, typename SymbolResolverPtrT>
  class ConcreteLinkedObjectSet : public LinkedObjectSet {
  public:
    ConcreteLinkedObjectSet(MemoryManagerPtrT MemMgr,
                            SymbolResolverPtrT Resolver)
      : LinkedObjectSet(*MemMgr, *Resolver), MemMgr(std::move(MemMgr)),
        Resolver(std::move(Resolver)) { }

    void Finalize() override {
      State = Finalizing;
      RTDyld->resolveRelocations();
      RTDyld->registerEHFrames();
      MemMgr->finalizeMemory();
      OwnedBuffers.clear();
      State = Finalized;
    }

  private:
    MemoryManagerPtrT MemMgr;
    SymbolResolverPtrT Resolver;
  };

  template <typename MemoryManagerPtrT, typename SymbolResolverPtrT>
  std::unique_ptr<LinkedObjectSet>
  createLinkedObjectSet(MemoryManagerPtrT MemMgr, SymbolResolverPtrT Resolver) {
    typedef ConcreteLinkedObjectSet<MemoryManagerPtrT, SymbolResolverPtrT> LOS;
    return llvm::make_unique<LOS>(std::move(MemMgr), std::move(Resolver));
  }

public:

  /// @brief LoadedObjectInfo list. Contains a list of owning pointers to
  ///        RuntimeDyld::LoadedObjectInfo instances.
  typedef std::vector<std::unique_ptr<RuntimeDyld::LoadedObjectInfo>>
      LoadedObjInfoList;

  /// @brief Functor for receiving finalization notifications.
  typedef std::function<void(ObjSetHandleT)> NotifyFinalizedFtor;

  /// @brief Construct an ObjectLinkingLayer with the given NotifyLoaded,
  ///        and NotifyFinalized functors.
  ObjectLinkingLayer(
      NotifyLoadedFtor NotifyLoaded = NotifyLoadedFtor(),
      NotifyFinalizedFtor NotifyFinalized = NotifyFinalizedFtor())
      : NotifyLoaded(std::move(NotifyLoaded)),
        NotifyFinalized(std::move(NotifyFinalized)) {}

  /// @brief Add a set of objects (or archives) that will be treated as a unit
  ///        for the purposes of symbol lookup and memory management.
  ///
  /// @return A pair containing (1) A handle that can be used to free the memory
  ///         allocated for the objects, and (2) a LoadedObjInfoList containing
  ///         one LoadedObjInfo instance for each object at the corresponding
  ///         index in the Objects list.
  ///
  ///   This version of this method allows the client to pass in an
  /// RTDyldMemoryManager instance that will be used to allocate memory and look
  /// up external symbol addresses for the given objects.
  template <typename ObjSetT,
            typename MemoryManagerPtrT,
            typename SymbolResolverPtrT>
  ObjSetHandleT addObjectSet(const ObjSetT &Objects,
                             MemoryManagerPtrT MemMgr,
                             SymbolResolverPtrT Resolver) {
    ObjSetHandleT Handle =
      LinkedObjSetList.insert(
        LinkedObjSetList.end(),
        createLinkedObjectSet(std::move(MemMgr), std::move(Resolver)));

    LinkedObjectSet &LOS = **Handle;
    LoadedObjInfoList LoadedObjInfos;

    for (auto &Obj : Objects)
      LoadedObjInfos.push_back(LOS.addObject(*Obj));

    NotifyLoaded(Handle, Objects, LoadedObjInfos);

    return Handle;
  }

  /// @brief Remove the set of objects associated with handle H.
  ///
  ///   All memory allocated for the objects will be freed, and the sections and
  /// symbols they provided will no longer be available. No attempt is made to
  /// re-emit the missing symbols, and any use of these symbols (directly or
  /// indirectly) will result in undefined behavior. If dependence tracking is
  /// required to detect or resolve such issues it should be added at a higher
  /// layer.
  void removeObjectSet(ObjSetHandleT H) {
    // How do we invalidate the symbols in H?
    LinkedObjSetList.erase(H);
  }

  /// @brief Search for the given named symbol.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it exists.
  JITSymbol findSymbol(StringRef Name, bool ExportedSymbolsOnly) {
    for (auto I = LinkedObjSetList.begin(), E = LinkedObjSetList.end(); I != E;
         ++I)
      if (auto Symbol = findSymbolIn(I, Name, ExportedSymbolsOnly))
        return Symbol;

    return nullptr;
  }

  /// @brief Search for the given named symbol in the context of the set of
  ///        loaded objects represented by the handle H.
  /// @param H The handle for the object set to search in.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it is found in the
  ///         given object set.
  JITSymbol findSymbolIn(ObjSetHandleT H, StringRef Name,
                         bool ExportedSymbolsOnly) {
    if (auto Sym = (*H)->getSymbol(Name)) {
      if (Sym.isExported() || !ExportedSymbolsOnly) {
        auto Addr = Sym.getAddress();
        auto Flags = Sym.getFlags();
        if (!(*H)->NeedsFinalization()) {
          // If this instance has already been finalized then we can just return
          // the address.
          return JITSymbol(Addr, Flags);
        } else {
          // If this instance needs finalization return a functor that will do
          // it. The functor still needs to double-check whether finalization is
          // required, in case someone else finalizes this set before the
          // functor is called.
          auto GetAddress =
            [this, Addr, H]() {
              if ((*H)->NeedsFinalization()) {
                (*H)->Finalize();
                if (NotifyFinalized)
                  NotifyFinalized(H);
              }
              return Addr;
            };
          return JITSymbol(std::move(GetAddress), Flags);
        }
      }
    }
    return nullptr;
  }

  /// @brief Map section addresses for the objects associated with the handle H.
  void mapSectionAddress(ObjSetHandleT H, const void *LocalAddress,
                         TargetAddress TargetAddr) {
    (*H)->mapSectionAddress(LocalAddress, TargetAddr);
  }

  /// @brief Immediately emit and finalize the object set represented by the
  ///        given handle.
  /// @param H Handle for object set to emit/finalize.
  void emitAndFinalize(ObjSetHandleT H) {
    (*H)->Finalize();
    if (NotifyFinalized)
      NotifyFinalized(H);
  }

private:
  LinkedObjectSetListT LinkedObjSetList;
  NotifyLoadedFtor NotifyLoaded;
  NotifyFinalizedFtor NotifyFinalized;
};

} // End namespace orc.
} // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_OBJECTLINKINGLAYER_H
