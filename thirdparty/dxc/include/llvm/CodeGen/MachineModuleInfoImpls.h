//===-- llvm/CodeGen/MachineModuleInfoImpls.h -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines object-file format specific implementations of
// MachineModuleInfoImpl.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEMODULEINFOIMPLS_H
#define LLVM_CODEGEN_MACHINEMODULEINFOIMPLS_H

#include "llvm/CodeGen/MachineModuleInfo.h"

namespace llvm {
  class MCSymbol;

  /// MachineModuleInfoMachO - This is a MachineModuleInfoImpl implementation
  /// for MachO targets.
  class MachineModuleInfoMachO : public MachineModuleInfoImpl {
    /// FnStubs - Darwin '$stub' stubs.  The key is something like "Lfoo$stub",
    /// the value is something like "_foo".
    DenseMap<MCSymbol*, StubValueTy> FnStubs;
    
    /// GVStubs - Darwin '$non_lazy_ptr' stubs.  The key is something like
    /// "Lfoo$non_lazy_ptr", the value is something like "_foo". The extra bit
    /// is true if this GV is external.
    DenseMap<MCSymbol*, StubValueTy> GVStubs;
    
    /// HiddenGVStubs - Darwin '$non_lazy_ptr' stubs.  The key is something like
    /// "Lfoo$non_lazy_ptr", the value is something like "_foo".  Unlike GVStubs
    /// these are for things with hidden visibility. The extra bit is true if
    /// this GV is external.
    DenseMap<MCSymbol*, StubValueTy> HiddenGVStubs;
    
    virtual void anchor();  // Out of line virtual method.
  public:
    MachineModuleInfoMachO(const MachineModuleInfo &) {}
    
    StubValueTy &getFnStubEntry(MCSymbol *Sym) {
      assert(Sym && "Key cannot be null");
      return FnStubs[Sym];
    }

    StubValueTy &getGVStubEntry(MCSymbol *Sym) {
      assert(Sym && "Key cannot be null");
      return GVStubs[Sym];
    }

    StubValueTy &getHiddenGVStubEntry(MCSymbol *Sym) {
      assert(Sym && "Key cannot be null");
      return HiddenGVStubs[Sym];
    }

    /// Accessor methods to return the set of stubs in sorted order.
    SymbolListTy GetFnStubList() {
      return getSortedStubs(FnStubs);
    }
    SymbolListTy GetGVStubList() {
      return getSortedStubs(GVStubs);
    }
    SymbolListTy GetHiddenGVStubList() {
      return getSortedStubs(HiddenGVStubs);
    }
  };

  /// MachineModuleInfoELF - This is a MachineModuleInfoImpl implementation
  /// for ELF targets.
  class MachineModuleInfoELF : public MachineModuleInfoImpl {
    /// GVStubs - These stubs are used to materialize global addresses in PIC
    /// mode.
    DenseMap<MCSymbol*, StubValueTy> GVStubs;

    virtual void anchor();  // Out of line virtual method.
  public:
    MachineModuleInfoELF(const MachineModuleInfo &) {}

    StubValueTy &getGVStubEntry(MCSymbol *Sym) {
      assert(Sym && "Key cannot be null");
      return GVStubs[Sym];
    }

    /// Accessor methods to return the set of stubs in sorted order.

    SymbolListTy GetGVStubList() {
      return getSortedStubs(GVStubs);
    }
  };

} // end namespace llvm

#endif
