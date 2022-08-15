//===-- RTDyldMemoryManager.cpp - Memory manager for MC-JIT -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface of the runtime dynamic memory manager base class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_RTDYLDMEMORYMANAGER_H
#define LLVM_EXECUTIONENGINE_RTDYLDMEMORYMANAGER_H

#include "RuntimeDyld.h"
#include "llvm-c/ExecutionEngine.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/Memory.h"

namespace llvm {

class ExecutionEngine;

  namespace object {
    class ObjectFile;
  }

class MCJITMemoryManager : public RuntimeDyld::MemoryManager {
public:
  /// This method is called after an object has been loaded into memory but
  /// before relocations are applied to the loaded sections.  The object load
  /// may have been initiated by MCJIT to resolve an external symbol for another
  /// object that is being finalized.  In that case, the object about which
  /// the memory manager is being notified will be finalized immediately after
  /// the memory manager returns from this call.
  ///
  /// Memory managers which are preparing code for execution in an external
  /// address space can use this call to remap the section addresses for the
  /// newly loaded object.
  virtual void notifyObjectLoaded(ExecutionEngine *EE,
                                  const object::ObjectFile &) {}
};

// RuntimeDyld clients often want to handle the memory management of
// what gets placed where. For JIT clients, this is the subset of
// JITMemoryManager required for dynamic loading of binaries.
//
// FIXME: As the RuntimeDyld fills out, additional routines will be needed
//        for the varying types of objects to be allocated.
class RTDyldMemoryManager : public MCJITMemoryManager,
                            public RuntimeDyld::SymbolResolver {
  RTDyldMemoryManager(const RTDyldMemoryManager&) = delete;
  void operator=(const RTDyldMemoryManager&) = delete;
public:
  RTDyldMemoryManager() {}
  ~RTDyldMemoryManager() override;

  void registerEHFrames(uint8_t *Addr, uint64_t LoadAddr, size_t Size) override;
  void deregisterEHFrames(uint8_t *Addr, uint64_t LoadAddr, size_t Size) override;

  /// This method returns the address of the specified function or variable in
  /// the current process.
  static uint64_t getSymbolAddressInProcess(const std::string &Name);

  /// Legacy symbol lookup - DEPRECATED! Please override findSymbol instead.
  ///
  /// This method returns the address of the specified function or variable.
  /// It is used to resolve symbols during module linking.
  virtual uint64_t getSymbolAddress(const std::string &Name) {
    return getSymbolAddressInProcess(Name);
  }

  /// This method returns a RuntimeDyld::SymbolInfo for the specified function
  /// or variable. It is used to resolve symbols during module linking.
  ///
  /// By default this falls back on the legacy lookup method:
  /// 'getSymbolAddress'. The address returned by getSymbolAddress is treated as
  /// a strong, exported symbol, consistent with historical treatment by
  /// RuntimeDyld.
  ///
  /// Clients writing custom RTDyldMemoryManagers are encouraged to override
  /// this method and return a SymbolInfo with the flags set correctly. This is
  /// necessary for RuntimeDyld to correctly handle weak and non-exported symbols.
  RuntimeDyld::SymbolInfo findSymbol(const std::string &Name) override {
    return RuntimeDyld::SymbolInfo(getSymbolAddress(Name),
                                   JITSymbolFlags::Exported);
  }

  /// Legacy symbol lookup -- DEPRECATED! Please override
  /// findSymbolInLogicalDylib instead.
  ///
  /// Default to treating all modules as separate.
  virtual uint64_t getSymbolAddressInLogicalDylib(const std::string &Name) {
    return 0;
  }

  /// Default to treating all modules as separate.
  ///
  /// By default this falls back on the legacy lookup method:
  /// 'getSymbolAddressInLogicalDylib'. The address returned by
  /// getSymbolAddressInLogicalDylib is treated as a strong, exported symbol,
  /// consistent with historical treatment by RuntimeDyld.
  ///
  /// Clients writing custom RTDyldMemoryManagers are encouraged to override
  /// this method and return a SymbolInfo with the flags set correctly. This is
  /// necessary for RuntimeDyld to correctly handle weak and non-exported symbols.
  RuntimeDyld::SymbolInfo
  findSymbolInLogicalDylib(const std::string &Name) override {
    return RuntimeDyld::SymbolInfo(getSymbolAddressInLogicalDylib(Name),
                                   JITSymbolFlags::Exported);
  }

  /// This method returns the address of the specified function. As such it is
  /// only useful for resolving library symbols, not code generated symbols.
  ///
  /// If \p AbortOnFailure is false and no function with the given name is
  /// found, this function returns a null pointer. Otherwise, it prints a
  /// message to stderr and aborts.
  ///
  /// This function is deprecated for memory managers to be used with
  /// MCJIT or RuntimeDyld.  Use getSymbolAddress instead.
  virtual void *getPointerToNamedFunction(const std::string &Name,
                                          bool AbortOnFailure = true);
};

// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(
    RTDyldMemoryManager, LLVMMCJITMemoryManagerRef)

} // namespace llvm


#endif
