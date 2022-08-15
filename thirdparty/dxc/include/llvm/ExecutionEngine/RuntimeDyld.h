//===-- RuntimeDyld.h - Run-time dynamic linker for MC-JIT ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface for the runtime dynamic linker facilities of the MC-JIT.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_RUNTIMEDYLD_H
#define LLVM_EXECUTIONENGINE_RUNTIMEDYLD_H

#include "JITSymbolFlags.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Memory.h"
#include "llvm/DebugInfo/DIContext.h"
#include <memory>

namespace llvm {

namespace object {
  class ObjectFile;
  template <typename T> class OwningBinary;
}

class RuntimeDyldImpl;
class RuntimeDyldCheckerImpl;

class RuntimeDyld {
  friend class RuntimeDyldCheckerImpl;

  RuntimeDyld(const RuntimeDyld &) = delete;
  void operator=(const RuntimeDyld &) = delete;

protected:
  // Change the address associated with a section when resolving relocations.
  // Any relocations already associated with the symbol will be re-resolved.
  void reassignSectionAddress(unsigned SectionID, uint64_t Addr);
public:

  /// \brief Information about a named symbol.
  class SymbolInfo : public JITSymbolBase {
  public:
    SymbolInfo(std::nullptr_t) : JITSymbolBase(JITSymbolFlags::None), Address(0) {}
    SymbolInfo(uint64_t Address, JITSymbolFlags Flags)
      : JITSymbolBase(Flags), Address(Address) {}
    explicit operator bool() const { return Address != 0; }
    uint64_t getAddress() const { return Address; }
  private:
    uint64_t Address;
  };

  /// \brief Information about the loaded object.
  class LoadedObjectInfo : public llvm::LoadedObjectInfo {
    friend class RuntimeDyldImpl;
  public:
    LoadedObjectInfo(RuntimeDyldImpl &RTDyld, unsigned BeginIdx,
                     unsigned EndIdx)
      : RTDyld(RTDyld), BeginIdx(BeginIdx), EndIdx(EndIdx) { }

    virtual object::OwningBinary<object::ObjectFile>
    getObjectForDebug(const object::ObjectFile &Obj) const = 0;

    uint64_t getSectionLoadAddress(StringRef Name) const;

  protected:
    virtual void anchor();

    RuntimeDyldImpl &RTDyld;
    unsigned BeginIdx, EndIdx;
  };

  template <typename Derived> struct LoadedObjectInfoHelper : LoadedObjectInfo {
    LoadedObjectInfoHelper(RuntimeDyldImpl &RTDyld, unsigned BeginIdx,
                           unsigned EndIdx)
        : LoadedObjectInfo(RTDyld, BeginIdx, EndIdx) {}
    std::unique_ptr<llvm::LoadedObjectInfo> clone() const override {
      return llvm::make_unique<Derived>(static_cast<const Derived &>(*this));
    }
  };

  /// \brief Memory Management.
  class MemoryManager {
  public:
    virtual ~MemoryManager() {};

    /// Allocate a memory block of (at least) the given size suitable for
    /// executable code. The SectionID is a unique identifier assigned by the
    /// RuntimeDyld instance, and optionally recorded by the memory manager to
    /// access a loaded section.
    virtual uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                                         unsigned SectionID,
                                         StringRef SectionName) = 0;

    /// Allocate a memory block of (at least) the given size suitable for data.
    /// The SectionID is a unique identifier assigned by the JIT engine, and
    /// optionally recorded by the memory manager to access a loaded section.
    virtual uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                                         unsigned SectionID,
                                         StringRef SectionName,
                                         bool IsReadOnly) = 0;

    /// Inform the memory manager about the total amount of memory required to
    /// allocate all sections to be loaded:
    /// \p CodeSize - the total size of all code sections
    /// \p DataSizeRO - the total size of all read-only data sections
    /// \p DataSizeRW - the total size of all read-write data sections
    ///
    /// Note that by default the callback is disabled. To enable it
    /// redefine the method needsToReserveAllocationSpace to return true.
    virtual void reserveAllocationSpace(uintptr_t CodeSize,
                                        uintptr_t DataSizeRO,
                                        uintptr_t DataSizeRW) {}

    /// Override to return true to enable the reserveAllocationSpace callback.
    virtual bool needsToReserveAllocationSpace() { return false; }

    /// Register the EH frames with the runtime so that c++ exceptions work.
    ///
    /// \p Addr parameter provides the local address of the EH frame section
    /// data, while \p LoadAddr provides the address of the data in the target
    /// address space.  If the section has not been remapped (which will usually
    /// be the case for local execution) these two values will be the same.
    virtual void registerEHFrames(uint8_t *Addr, uint64_t LoadAddr,
                                  size_t Size) = 0;
    virtual void deregisterEHFrames(uint8_t *addr, uint64_t LoadAddr,
                                    size_t Size) = 0;

    /// This method is called when object loading is complete and section page
    /// permissions can be applied.  It is up to the memory manager implementation
    /// to decide whether or not to act on this method.  The memory manager will
    /// typically allocate all sections as read-write and then apply specific
    /// permissions when this method is called.  Code sections cannot be executed
    /// until this function has been called.  In addition, any cache coherency
    /// operations needed to reliably use the memory are also performed.
    ///
    /// Returns true if an error occurred, false otherwise.
    virtual bool finalizeMemory(std::string *ErrMsg = nullptr) = 0;

  private:
    virtual void anchor();
  };

  /// \brief Symbol resolution.
  class SymbolResolver {
  public:
    virtual ~SymbolResolver() {};

    /// This method returns the address of the specified function or variable.
    /// It is used to resolve symbols during module linking.
    ///
    /// If the returned symbol's address is equal to ~0ULL then RuntimeDyld will
    /// skip all relocations for that symbol, and the client will be responsible
    /// for handling them manually.
    virtual SymbolInfo findSymbol(const std::string &Name) = 0;

    /// This method returns the address of the specified symbol if it exists
    /// within the logical dynamic library represented by this
    /// RTDyldMemoryManager. Unlike getSymbolAddress, queries through this
    /// interface should return addresses for hidden symbols.
    ///
    /// This is of particular importance for the Orc JIT APIs, which support lazy
    /// compilation by breaking up modules: Each of those broken out modules
    /// must be able to resolve hidden symbols provided by the others. Clients
    /// writing memory managers for MCJIT can usually ignore this method.
    ///
    /// This method will be queried by RuntimeDyld when checking for previous
    /// definitions of common symbols. It will *not* be queried by default when
    /// resolving external symbols (this minimises the link-time overhead for
    /// MCJIT clients who don't care about Orc features). If you are writing a
    /// RTDyldMemoryManager for Orc and want "external" symbol resolution to
    /// search the logical dylib, you should override your getSymbolAddress
    /// method call this method directly.
    virtual SymbolInfo findSymbolInLogicalDylib(const std::string &Name) = 0;
  private:
    virtual void anchor();
  };

  /// \brief Construct a RuntimeDyld instance.
  RuntimeDyld(MemoryManager &MemMgr, SymbolResolver &Resolver);
  ~RuntimeDyld();

  /// Add the referenced object file to the list of objects to be loaded and
  /// relocated.
  std::unique_ptr<LoadedObjectInfo> loadObject(const object::ObjectFile &O);

  /// Get the address of our local copy of the symbol. This may or may not
  /// be the address used for relocation (clients can copy the data around
  /// and resolve relocatons based on where they put it).
  void *getSymbolLocalAddress(StringRef Name) const;

  /// Get the target address and flags for the named symbol.
  /// This address is the one used for relocation.
  SymbolInfo getSymbol(StringRef Name) const;

  /// Resolve the relocations for all symbols we currently know about.
  void resolveRelocations();

  /// Map a section to its target address space value.
  /// Map the address of a JIT section as returned from the memory manager
  /// to the address in the target process as the running code will see it.
  /// This is the address which will be used for relocation resolution.
  void mapSectionAddress(const void *LocalAddress, uint64_t TargetAddress);

  /// Register any EH frame sections that have been loaded but not previously
  /// registered with the memory manager.  Note, RuntimeDyld is responsible
  /// for identifying the EH frame and calling the memory manager with the
  /// EH frame section data.  However, the memory manager itself will handle
  /// the actual target-specific EH frame registration.
  void registerEHFrames();

  void deregisterEHFrames();

  bool hasError();
  StringRef getErrorString();

  /// By default, only sections that are "required for execution" are passed to
  /// the RTDyldMemoryManager, and other sections are discarded. Passing 'true'
  /// to this method will cause RuntimeDyld to pass all sections to its
  /// memory manager regardless of whether they are "required to execute" in the
  /// usual sense. This is useful for inspecting metadata sections that may not
  /// contain relocations, E.g. Debug info, stackmaps.
  ///
  /// Must be called before the first object file is loaded.
  void setProcessAllSections(bool ProcessAllSections) {
    assert(!Dyld && "setProcessAllSections must be called before loadObject.");
    this->ProcessAllSections = ProcessAllSections;
  }

private:
  // RuntimeDyldImpl is the actual class. RuntimeDyld is just the public
  // interface.
  std::unique_ptr<RuntimeDyldImpl> Dyld;
  MemoryManager &MemMgr;
  SymbolResolver &Resolver;
  bool ProcessAllSections;
  RuntimeDyldCheckerImpl *Checker;
};

} // end namespace llvm

#endif
