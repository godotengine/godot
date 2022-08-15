//===---- OrcMCJITReplacement.h - Orc based MCJIT replacement ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Orc based MCJIT replacement.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_EXECUTIONENGINE_ORC_ORCMCJITREPLACEMENT_H
#define LLVM_LIB_EXECUTIONENGINE_ORC_ORCMCJITREPLACEMENT_H

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LazyEmittingLayer.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/Object/Archive.h"

namespace llvm {
namespace orc {

class OrcMCJITReplacement : public ExecutionEngine {

  // OrcMCJITReplacement needs to do a little extra book-keeping to ensure that
  // Orc's automatic finalization doesn't kick in earlier than MCJIT clients are
  // expecting - see finalizeMemory.
  class MCJITReplacementMemMgr : public MCJITMemoryManager {
  public:
    MCJITReplacementMemMgr(OrcMCJITReplacement &M,
                           std::shared_ptr<MCJITMemoryManager> ClientMM)
      : M(M), ClientMM(std::move(ClientMM)) {}

    uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                                 unsigned SectionID,
                                 StringRef SectionName) override {
      uint8_t *Addr =
          ClientMM->allocateCodeSection(Size, Alignment, SectionID,
                                        SectionName);
      M.SectionsAllocatedSinceLastLoad.insert(Addr);
      return Addr;
    }

    uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                                 unsigned SectionID, StringRef SectionName,
                                 bool IsReadOnly) override {
      uint8_t *Addr = ClientMM->allocateDataSection(Size, Alignment, SectionID,
                                                    SectionName, IsReadOnly);
      M.SectionsAllocatedSinceLastLoad.insert(Addr);
      return Addr;
    }

    void reserveAllocationSpace(uintptr_t CodeSize, uintptr_t DataSizeRO,
                                uintptr_t DataSizeRW) override {
      return ClientMM->reserveAllocationSpace(CodeSize, DataSizeRO,
                                                DataSizeRW);
    }

    bool needsToReserveAllocationSpace() override {
      return ClientMM->needsToReserveAllocationSpace();
    }

    void registerEHFrames(uint8_t *Addr, uint64_t LoadAddr,
                          size_t Size) override {
      return ClientMM->registerEHFrames(Addr, LoadAddr, Size);
    }

    void deregisterEHFrames(uint8_t *Addr, uint64_t LoadAddr,
                            size_t Size) override {
      return ClientMM->deregisterEHFrames(Addr, LoadAddr, Size);
    }

    void notifyObjectLoaded(ExecutionEngine *EE,
                            const object::ObjectFile &O) override {
      return ClientMM->notifyObjectLoaded(EE, O);
    }

    bool finalizeMemory(std::string *ErrMsg = nullptr) override {
      // Each set of objects loaded will be finalized exactly once, but since
      // symbol lookup during relocation may recursively trigger the
      // loading/relocation of other modules, and since we're forwarding all
      // finalizeMemory calls to a single underlying memory manager, we need to
      // defer forwarding the call on until all necessary objects have been
      // loaded. Otherwise, during the relocation of a leaf object, we will end
      // up finalizing memory, causing a crash further up the stack when we
      // attempt to apply relocations to finalized memory.
      // To avoid finalizing too early, look at how many objects have been
      // loaded but not yet finalized. This is a bit of a hack that relies on
      // the fact that we're lazily emitting object files: The only way you can
      // get more than one set of objects loaded but not yet finalized is if
      // they were loaded during relocation of another set.
      if (M.UnfinalizedSections.size() == 1)
        return ClientMM->finalizeMemory(ErrMsg);
      return false;
    }

  private:
    OrcMCJITReplacement &M;
    std::shared_ptr<MCJITMemoryManager> ClientMM;
  };

  class LinkingResolver : public RuntimeDyld::SymbolResolver {
  public:
    LinkingResolver(OrcMCJITReplacement &M) : M(M) {}

    RuntimeDyld::SymbolInfo findSymbol(const std::string &Name) override {
      return M.findMangledSymbol(Name);
    }

    RuntimeDyld::SymbolInfo
    findSymbolInLogicalDylib(const std::string &Name) override {
      return M.ClientResolver->findSymbolInLogicalDylib(Name);
    }

  private:
    OrcMCJITReplacement &M;
  };

private:

  static ExecutionEngine *
  createOrcMCJITReplacement(std::string *ErrorMsg,
                            std::shared_ptr<MCJITMemoryManager> MemMgr,
                            std::shared_ptr<RuntimeDyld::SymbolResolver> Resolver,
                            std::unique_ptr<TargetMachine> TM) {
    return new OrcMCJITReplacement(std::move(MemMgr), std::move(Resolver),
                                   std::move(TM));
  }

public:
  static void Register() {
    OrcMCJITReplacementCtor = createOrcMCJITReplacement;
  }

  OrcMCJITReplacement(
                    std::shared_ptr<MCJITMemoryManager> MemMgr,
                    std::shared_ptr<RuntimeDyld::SymbolResolver> ClientResolver,
                    std::unique_ptr<TargetMachine> TM)
      : TM(std::move(TM)), MemMgr(*this, std::move(MemMgr)),
        Resolver(*this), ClientResolver(std::move(ClientResolver)),
        NotifyObjectLoaded(*this), NotifyFinalized(*this),
        ObjectLayer(NotifyObjectLoaded, NotifyFinalized),
        CompileLayer(ObjectLayer, SimpleCompiler(*this->TM)),
        LazyEmitLayer(CompileLayer) {
    setDataLayout(this->TM->getDataLayout());
  }

  void addModule(std::unique_ptr<Module> M) override {

    // If this module doesn't have a DataLayout attached then attach the
    // default.
    if (M->getDataLayout().isDefault())
      M->setDataLayout(*getDataLayout());

    Modules.push_back(std::move(M));
    std::vector<Module *> Ms;
    Ms.push_back(&*Modules.back());
    LazyEmitLayer.addModuleSet(std::move(Ms), &MemMgr, &Resolver);
  }

  void addObjectFile(std::unique_ptr<object::ObjectFile> O) override {
    std::vector<std::unique_ptr<object::ObjectFile>> Objs;
    Objs.push_back(std::move(O));
    ObjectLayer.addObjectSet(std::move(Objs), &MemMgr, &Resolver);
  }

  void addObjectFile(object::OwningBinary<object::ObjectFile> O) override {
    std::unique_ptr<object::ObjectFile> Obj;
    std::unique_ptr<MemoryBuffer> Buf;
    std::tie(Obj, Buf) = O.takeBinary();
    std::vector<std::unique_ptr<object::ObjectFile>> Objs;
    Objs.push_back(std::move(Obj));
    auto H =
      ObjectLayer.addObjectSet(std::move(Objs), &MemMgr, &Resolver);

    std::vector<std::unique_ptr<MemoryBuffer>> Bufs;
    Bufs.push_back(std::move(Buf));
    ObjectLayer.takeOwnershipOfBuffers(H, std::move(Bufs));
  }

  void addArchive(object::OwningBinary<object::Archive> A) override {
    Archives.push_back(std::move(A));
  }

  uint64_t getSymbolAddress(StringRef Name) {
    return findSymbol(Name).getAddress();
  }

  RuntimeDyld::SymbolInfo findSymbol(StringRef Name) {
    return findMangledSymbol(Mangle(Name));
  }

  void finalizeObject() override {
    // This is deprecated - Aim to remove in ExecutionEngine.
    // REMOVE IF POSSIBLE - Doesn't make sense for New JIT.
  }

  void mapSectionAddress(const void *LocalAddress,
                         uint64_t TargetAddress) override {
    for (auto &P : UnfinalizedSections)
      if (P.second.count(LocalAddress))
        ObjectLayer.mapSectionAddress(P.first, LocalAddress, TargetAddress);
  }

  uint64_t getGlobalValueAddress(const std::string &Name) override {
    return getSymbolAddress(Name);
  }

  uint64_t getFunctionAddress(const std::string &Name) override {
    return getSymbolAddress(Name);
  }

  void *getPointerToFunction(Function *F) override {
    uint64_t FAddr = getSymbolAddress(F->getName());
    return reinterpret_cast<void *>(static_cast<uintptr_t>(FAddr));
  }

  void *getPointerToNamedFunction(StringRef Name,
                                  bool AbortOnFailure = true) override {
    uint64_t Addr = getSymbolAddress(Name);
    if (!Addr && AbortOnFailure)
      llvm_unreachable("Missing symbol!");
    return reinterpret_cast<void *>(static_cast<uintptr_t>(Addr));
  }

  GenericValue runFunction(Function *F,
                           ArrayRef<GenericValue> ArgValues) override;

  void setObjectCache(ObjectCache *NewCache) override {
    CompileLayer.setObjectCache(NewCache);
  }

private:

  RuntimeDyld::SymbolInfo findMangledSymbol(StringRef Name) {
    if (auto Sym = LazyEmitLayer.findSymbol(Name, false))
      return RuntimeDyld::SymbolInfo(Sym.getAddress(), Sym.getFlags());
    if (auto Sym = ClientResolver->findSymbol(Name))
      return RuntimeDyld::SymbolInfo(Sym.getAddress(), Sym.getFlags());
    if (auto Sym = scanArchives(Name))
      return RuntimeDyld::SymbolInfo(Sym.getAddress(), Sym.getFlags());

    return nullptr;
  }

  JITSymbol scanArchives(StringRef Name) {
    for (object::OwningBinary<object::Archive> &OB : Archives) {
      object::Archive *A = OB.getBinary();
      // Look for our symbols in each Archive
      object::Archive::child_iterator ChildIt = A->findSym(Name);
      if (ChildIt != A->child_end()) {
        // FIXME: Support nested archives?
        ErrorOr<std::unique_ptr<object::Binary>> ChildBinOrErr =
            ChildIt->getAsBinary();
        if (ChildBinOrErr.getError())
          continue;
        std::unique_ptr<object::Binary> &ChildBin = ChildBinOrErr.get();
        if (ChildBin->isObject()) {
          std::vector<std::unique_ptr<object::ObjectFile>> ObjSet;
          ObjSet.push_back(std::unique_ptr<object::ObjectFile>(
              static_cast<object::ObjectFile *>(ChildBin.release())));
          ObjectLayer.addObjectSet(std::move(ObjSet), &MemMgr, &Resolver);
          if (auto Sym = ObjectLayer.findSymbol(Name, true))
            return Sym;
        }
      }
    }
    return nullptr;
  }

  class NotifyObjectLoadedT {
  public:
    typedef std::vector<std::unique_ptr<object::ObjectFile>> ObjListT;
    typedef std::vector<std::unique_ptr<RuntimeDyld::LoadedObjectInfo>>
        LoadedObjInfoListT;

    NotifyObjectLoadedT(OrcMCJITReplacement &M) : M(M) {}

    void operator()(ObjectLinkingLayerBase::ObjSetHandleT H,
                    const ObjListT &Objects,
                    const LoadedObjInfoListT &Infos) const {
      M.UnfinalizedSections[H] = std::move(M.SectionsAllocatedSinceLastLoad);
      M.SectionsAllocatedSinceLastLoad = SectionAddrSet();
      assert(Objects.size() == Infos.size() &&
             "Incorrect number of Infos for Objects.");
      for (unsigned I = 0; I < Objects.size(); ++I)
        M.MemMgr.notifyObjectLoaded(&M, *Objects[I]);
    };

  private:
    OrcMCJITReplacement &M;
  };

  class NotifyFinalizedT {
  public:
    NotifyFinalizedT(OrcMCJITReplacement &M) : M(M) {}
    void operator()(ObjectLinkingLayerBase::ObjSetHandleT H) {
      M.UnfinalizedSections.erase(H);
    }

  private:
    OrcMCJITReplacement &M;
  };

  std::string Mangle(StringRef Name) {
    std::string MangledName;
    {
      raw_string_ostream MangledNameStream(MangledName);
      Mang.getNameWithPrefix(MangledNameStream, Name, *TM->getDataLayout());
    }
    return MangledName;
  }

  typedef ObjectLinkingLayer<NotifyObjectLoadedT> ObjectLayerT;
  typedef IRCompileLayer<ObjectLayerT> CompileLayerT;
  typedef LazyEmittingLayer<CompileLayerT> LazyEmitLayerT;

  std::unique_ptr<TargetMachine> TM;
  MCJITReplacementMemMgr MemMgr;
  LinkingResolver Resolver;
  std::shared_ptr<RuntimeDyld::SymbolResolver> ClientResolver;
  Mangler Mang;

  NotifyObjectLoadedT NotifyObjectLoaded;
  NotifyFinalizedT NotifyFinalized;

  ObjectLayerT ObjectLayer;
  CompileLayerT CompileLayer;
  LazyEmitLayerT LazyEmitLayer;

  // We need to store ObjLayerT::ObjSetHandles for each of the object sets
  // that have been emitted but not yet finalized so that we can forward the
  // mapSectionAddress calls appropriately.
  typedef std::set<const void *> SectionAddrSet;
  struct ObjSetHandleCompare {
    bool operator()(ObjectLayerT::ObjSetHandleT H1,
                    ObjectLayerT::ObjSetHandleT H2) const {
      return &*H1 < &*H2;
    }
  };
  SectionAddrSet SectionsAllocatedSinceLastLoad;
  std::map<ObjectLayerT::ObjSetHandleT, SectionAddrSet, ObjSetHandleCompare>
      UnfinalizedSections;

  std::vector<object::OwningBinary<object::Archive>> Archives;
};

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_LIB_EXECUTIONENGINE_ORC_MCJITREPLACEMENT_H
