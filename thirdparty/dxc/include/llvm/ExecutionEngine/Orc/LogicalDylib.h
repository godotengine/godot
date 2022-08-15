//===--- LogicalDylib.h - Simulates dylib-style symbol lookup ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Simulates symbol resolution inside a dylib.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_LOGICALDYLIB_H
#define LLVM_EXECUTIONENGINE_ORC_LOGICALDYLIB_H

namespace llvm {
namespace orc {

template <typename BaseLayerT,
          typename LogicalModuleResources,
          typename LogicalDylibResources>
class LogicalDylib {
public:
  typedef typename BaseLayerT::ModuleSetHandleT BaseLayerModuleSetHandleT;
private:

  typedef std::vector<BaseLayerModuleSetHandleT> BaseLayerHandleList;

  struct LogicalModule {
    LogicalModuleResources Resources;
    BaseLayerHandleList BaseLayerHandles;
  };
  typedef std::vector<LogicalModule> LogicalModuleList;

public:

  typedef typename BaseLayerHandleList::iterator BaseLayerHandleIterator;
  typedef typename LogicalModuleList::iterator LogicalModuleHandle;

  LogicalDylib(BaseLayerT &BaseLayer) : BaseLayer(BaseLayer) {}

  ~LogicalDylib() {
    for (auto &LM : LogicalModules)
      for (auto BLH : LM.BaseLayerHandles)
        BaseLayer.removeModuleSet(BLH);
  }

  LogicalModuleHandle createLogicalModule() {
    LogicalModules.push_back(LogicalModule());
    return std::prev(LogicalModules.end());
  }

  void addToLogicalModule(LogicalModuleHandle LMH,
                          BaseLayerModuleSetHandleT BaseLayerHandle) {
    LMH->BaseLayerHandles.push_back(BaseLayerHandle);
  }

  LogicalModuleResources& getLogicalModuleResources(LogicalModuleHandle LMH) {
    return LMH->Resources;
  }

  BaseLayerHandleIterator moduleHandlesBegin(LogicalModuleHandle LMH) {
    return LMH->BaseLayerHandles.begin();
  }

  BaseLayerHandleIterator moduleHandlesEnd(LogicalModuleHandle LMH) {
    return LMH->BaseLayerHandles.end();
  }

  JITSymbol findSymbolInLogicalModule(LogicalModuleHandle LMH,
                                      const std::string &Name) {
    for (auto BLH : LMH->BaseLayerHandles)
      if (auto Symbol = BaseLayer.findSymbolIn(BLH, Name, false))
        return Symbol;
    return nullptr;
  }

  JITSymbol findSymbolInternally(LogicalModuleHandle LMH,
                                 const std::string &Name) {
    if (auto Symbol = findSymbolInLogicalModule(LMH, Name))
      return Symbol;

    for (auto LMI = LogicalModules.begin(), LME = LogicalModules.end();
           LMI != LME; ++LMI) {
      if (LMI != LMH)
        if (auto Symbol = findSymbolInLogicalModule(LMI, Name))
          return Symbol;
    }

    return nullptr;
  }

  JITSymbol findSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
    for (auto &LM : LogicalModules)
      for (auto BLH : LM.BaseLayerHandles)
        if (auto Symbol =
            BaseLayer.findSymbolIn(BLH, Name, ExportedSymbolsOnly))
          return Symbol;
    return nullptr;
  }

  LogicalDylibResources& getDylibResources() { return DylibResources; }

protected:
  BaseLayerT BaseLayer;
  LogicalModuleList LogicalModules;
  LogicalDylibResources DylibResources;

};

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_LOGICALDYLIB_H
