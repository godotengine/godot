//===- CompileOnDemandLayer.h - Compile each function on demand -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// JIT layer for breaking up modules and inserting callbacks to allow
// individual functions to be compiled on demand.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_COMPILEONDEMANDLAYER_H
#define LLVM_EXECUTIONENGINE_ORC_COMPILEONDEMANDLAYER_H

#include "IndirectionUtils.h"
#include "LambdaResolver.h"
#include "LogicalDylib.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <list>
#include <set>

#include "llvm/Support/Debug.h"

namespace llvm {
namespace orc {

/// @brief Compile-on-demand layer.
///
///   When a module is added to this layer a stub is created for each of its
/// function definitions. The stubs and other global values are immediately
/// added to the layer below. When a stub is called it triggers the extraction
/// of the function body from the original module. The extracted body is then
/// compiled and executed.
template <typename BaseLayerT, typename CompileCallbackMgrT,
          typename PartitioningFtor =
            std::function<std::set<Function*>(Function&)>>
class CompileOnDemandLayer {
private:

  // Utility class for MapValue. Only materializes declarations for global
  // variables.
  class GlobalDeclMaterializer : public ValueMaterializer {
  public:
    typedef std::set<const Function*> StubSet;

    GlobalDeclMaterializer(Module &Dst, const StubSet *StubsToClone = nullptr)
        : Dst(Dst), StubsToClone(StubsToClone) {}

    Value* materializeValueFor(Value *V) final {
      if (auto *GV = dyn_cast<GlobalVariable>(V))
        return cloneGlobalVariableDecl(Dst, *GV);
      else if (auto *F = dyn_cast<Function>(V)) {
        auto *ClonedF = cloneFunctionDecl(Dst, *F);
        if (StubsToClone && StubsToClone->count(F)) {
          GlobalVariable *FnBodyPtr =
            createImplPointer(*ClonedF->getType(), *ClonedF->getParent(),
                              ClonedF->getName() + "$orc_addr", nullptr);
          makeStub(*ClonedF, *FnBodyPtr);
          ClonedF->setLinkage(GlobalValue::AvailableExternallyLinkage);
          ClonedF->addFnAttr(Attribute::AlwaysInline);
        }
        return ClonedF;
      }
      // Else.
      return nullptr;
    }
  private:
    Module &Dst;
    const StubSet *StubsToClone;
  };

  typedef typename BaseLayerT::ModuleSetHandleT BaseLayerModuleSetHandleT;

  struct LogicalModuleResources {
    std::shared_ptr<Module> SourceModule;
    std::set<const Function*> StubsToClone;
  };

  struct LogicalDylibResources {
    typedef std::function<RuntimeDyld::SymbolInfo(const std::string&)>
      SymbolResolverFtor;
    SymbolResolverFtor ExternalSymbolResolver;
    PartitioningFtor Partitioner;
  };

  typedef LogicalDylib<BaseLayerT, LogicalModuleResources,
                       LogicalDylibResources> CODLogicalDylib;

  typedef typename CODLogicalDylib::LogicalModuleHandle LogicalModuleHandle;
  typedef std::list<CODLogicalDylib> LogicalDylibList;

public:
  /// @brief Handle to a set of loaded modules.
  typedef typename LogicalDylibList::iterator ModuleSetHandleT;

  /// @brief Construct a compile-on-demand layer instance.
  CompileOnDemandLayer(BaseLayerT &BaseLayer, CompileCallbackMgrT &CallbackMgr,
                       bool CloneStubsIntoPartitions)
      : BaseLayer(BaseLayer), CompileCallbackMgr(CallbackMgr),
        CloneStubsIntoPartitions(CloneStubsIntoPartitions) {}

  /// @brief Add a module to the compile-on-demand layer.
  template <typename ModuleSetT, typename MemoryManagerPtrT,
            typename SymbolResolverPtrT>
  ModuleSetHandleT addModuleSet(ModuleSetT Ms,
                                MemoryManagerPtrT MemMgr,
                                SymbolResolverPtrT Resolver) {

    assert(MemMgr == nullptr &&
           "User supplied memory managers not supported with COD yet.");

    LogicalDylibs.push_back(CODLogicalDylib(BaseLayer));
    auto &LDResources = LogicalDylibs.back().getDylibResources();

    LDResources.ExternalSymbolResolver =
      [Resolver](const std::string &Name) {
        return Resolver->findSymbol(Name);
      };

    LDResources.Partitioner =
      [](Function &F) {
        std::set<Function*> Partition;
        Partition.insert(&F);
        return Partition;
      };

    // Process each of the modules in this module set.
    for (auto &M : Ms)
      addLogicalModule(LogicalDylibs.back(),
                       std::shared_ptr<Module>(std::move(M)));

    return std::prev(LogicalDylibs.end());
  }

  /// @brief Remove the module represented by the given handle.
  ///
  ///   This will remove all modules in the layers below that were derived from
  /// the module represented by H.
  void removeModuleSet(ModuleSetHandleT H) {
    LogicalDylibs.erase(H);
  }

  /// @brief Search for the given named symbol.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it exists.
  JITSymbol findSymbol(StringRef Name, bool ExportedSymbolsOnly) {
    return BaseLayer.findSymbol(Name, ExportedSymbolsOnly);
  }

  /// @brief Get the address of a symbol provided by this layer, or some layer
  ///        below this one.
  JITSymbol findSymbolIn(ModuleSetHandleT H, const std::string &Name,
                         bool ExportedSymbolsOnly) {
    return H->findSymbol(Name, ExportedSymbolsOnly);
  }

private:

  void addLogicalModule(CODLogicalDylib &LD, std::shared_ptr<Module> SrcM) {

    // Bump the linkage and rename any anonymous/privote members in SrcM to
    // ensure that everything will resolve properly after we partition SrcM.
    makeAllSymbolsExternallyAccessible(*SrcM);

    // Create a logical module handle for SrcM within the logical dylib.
    auto LMH = LD.createLogicalModule();
    auto &LMResources =  LD.getLogicalModuleResources(LMH);
    LMResources.SourceModule = SrcM;

    // Create the GVs-and-stubs module.
    auto GVsAndStubsM = llvm::make_unique<Module>(
                          (SrcM->getName() + ".globals_and_stubs").str(),
                          SrcM->getContext());
    GVsAndStubsM->setDataLayout(SrcM->getDataLayout());
    ValueToValueMapTy VMap;

    // Process module and create stubs.
    // We create the stubs before copying the global variables as we know the
    // stubs won't refer to any globals (they only refer to their implementation
    // pointer) so there's no ordering/value-mapping issues.
    for (auto &F : *SrcM) {

      // Skip declarations.
      if (F.isDeclaration())
        continue;

      // Record all functions defined by this module.
      if (CloneStubsIntoPartitions)
        LMResources.StubsToClone.insert(&F);

      // For each definition: create a callback, a stub, and a function body
      // pointer. Initialize the function body pointer to point at the callback,
      // and set the callback to compile the function body.
      auto CCInfo = CompileCallbackMgr.getCompileCallback(SrcM->getContext());
      Function *StubF = cloneFunctionDecl(*GVsAndStubsM, F, &VMap);
      GlobalVariable *FnBodyPtr =
        createImplPointer(*StubF->getType(), *StubF->getParent(),
                          StubF->getName() + "$orc_addr",
                          createIRTypedAddress(*StubF->getFunctionType(),
                                               CCInfo.getAddress()));
      makeStub(*StubF, *FnBodyPtr);
      CCInfo.setCompileAction(
        [this, &LD, LMH, &F]() {
          return this->extractAndCompile(LD, LMH, F);
        });
    }

    // Now clone the global variable declarations.
    GlobalDeclMaterializer GDMat(*GVsAndStubsM);
    for (auto &GV : SrcM->globals())
      if (!GV.isDeclaration())
        cloneGlobalVariableDecl(*GVsAndStubsM, GV, &VMap);

    // Then clone the initializers.
    for (auto &GV : SrcM->globals())
      if (!GV.isDeclaration())
        moveGlobalVariableInitializer(GV, VMap, &GDMat);

    // Build a resolver for the stubs module and add it to the base layer.
    auto GVsAndStubsResolver = createLambdaResolver(
        [&LD](const std::string &Name) {
          return LD.getDylibResources().ExternalSymbolResolver(Name);
        },
        [](const std::string &Name) {
          return RuntimeDyld::SymbolInfo(nullptr);
        });

    std::vector<std::unique_ptr<Module>> GVsAndStubsMSet;
    GVsAndStubsMSet.push_back(std::move(GVsAndStubsM));
    auto GVsAndStubsH =
      BaseLayer.addModuleSet(std::move(GVsAndStubsMSet),
                             llvm::make_unique<SectionMemoryManager>(),
                             std::move(GVsAndStubsResolver));
    LD.addToLogicalModule(LMH, GVsAndStubsH);
  }

  static std::string Mangle(StringRef Name, const DataLayout &DL) {
    std::string MangledName;
    {
      raw_string_ostream MangledNameStream(MangledName);
      Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    }
    return MangledName;
  }

  TargetAddress extractAndCompile(CODLogicalDylib &LD,
                                  LogicalModuleHandle LMH,
                                  Function &F) {
    Module &SrcM = *LD.getLogicalModuleResources(LMH).SourceModule;

    // If F is a declaration we must already have compiled it.
    if (F.isDeclaration())
      return 0;

    // Grab the name of the function being called here.
    std::string CalledFnName = Mangle(F.getName(), SrcM.getDataLayout());

    auto Partition = LD.getDylibResources().Partitioner(F);
    auto PartitionH = emitPartition(LD, LMH, Partition);

    TargetAddress CalledAddr = 0;
    for (auto *SubF : Partition) {
      std::string FName = SubF->getName();
      auto FnBodySym =
        BaseLayer.findSymbolIn(PartitionH, Mangle(FName, SrcM.getDataLayout()),
                               false);
      auto FnPtrSym =
        BaseLayer.findSymbolIn(*LD.moduleHandlesBegin(LMH),
                               Mangle(FName + "$orc_addr",
                                      SrcM.getDataLayout()),
                               false);
      assert(FnBodySym && "Couldn't find function body.");
      assert(FnPtrSym && "Couldn't find function body pointer.");

      TargetAddress FnBodyAddr = FnBodySym.getAddress();
      void *FnPtrAddr = reinterpret_cast<void*>(
          static_cast<uintptr_t>(FnPtrSym.getAddress()));

      // If this is the function we're calling record the address so we can
      // return it from this function.
      if (SubF == &F)
        CalledAddr = FnBodyAddr;

      memcpy(FnPtrAddr, &FnBodyAddr, sizeof(uintptr_t));
    }

    return CalledAddr;
  }

  template <typename PartitionT>
  BaseLayerModuleSetHandleT emitPartition(CODLogicalDylib &LD,
                                          LogicalModuleHandle LMH,
                                          const PartitionT &Partition) {
    auto &LMResources = LD.getLogicalModuleResources(LMH);
    Module &SrcM = *LMResources.SourceModule;

    // Create the module.
    std::string NewName = SrcM.getName();
    for (auto *F : Partition) {
      NewName += ".";
      NewName += F->getName();
    }

    auto M = llvm::make_unique<Module>(NewName, SrcM.getContext());
    M->setDataLayout(SrcM.getDataLayout());
    ValueToValueMapTy VMap;
    GlobalDeclMaterializer GDM(*M, &LMResources.StubsToClone);

    // Create decls in the new module.
    for (auto *F : Partition)
      cloneFunctionDecl(*M, *F, &VMap);

    // Move the function bodies.
    for (auto *F : Partition)
      moveFunctionBody(*F, VMap, &GDM);

    // Create memory manager and symbol resolver.
    auto MemMgr = llvm::make_unique<SectionMemoryManager>();
    auto Resolver = createLambdaResolver(
        [this, &LD, LMH](const std::string &Name) {
          if (auto Symbol = LD.findSymbolInternally(LMH, Name))
            return RuntimeDyld::SymbolInfo(Symbol.getAddress(),
                                           Symbol.getFlags());
          return LD.getDylibResources().ExternalSymbolResolver(Name);
        },
        [this, &LD, LMH](const std::string &Name) {
          if (auto Symbol = LD.findSymbolInternally(LMH, Name))
            return RuntimeDyld::SymbolInfo(Symbol.getAddress(),
                                           Symbol.getFlags());
          return RuntimeDyld::SymbolInfo(nullptr);
        });
    std::vector<std::unique_ptr<Module>> PartMSet;
    PartMSet.push_back(std::move(M));
    return BaseLayer.addModuleSet(std::move(PartMSet), std::move(MemMgr),
                                  std::move(Resolver));
  }

  BaseLayerT &BaseLayer;
  CompileCallbackMgrT &CompileCallbackMgr;
  LogicalDylibList LogicalDylibs;
  bool CloneStubsIntoPartitions;
};

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_COMPILEONDEMANDLAYER_H
