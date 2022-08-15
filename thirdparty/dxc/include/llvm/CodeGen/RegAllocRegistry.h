//===-- llvm/CodeGen/RegAllocRegistry.h -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation for register allocator function
// pass registry (RegisterRegAlloc).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGALLOCREGISTRY_H
#define LLVM_CODEGEN_REGALLOCREGISTRY_H

#include "llvm/CodeGen/MachinePassRegistry.h"

namespace llvm {

//===----------------------------------------------------------------------===//
///
/// RegisterRegAlloc class - Track the registration of register allocators.
///
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
class RegisterRegAlloc : public MachinePassRegistryNode {

public:

  typedef FunctionPass *(*FunctionPassCtor)();

  static MachinePassRegistry Registry;

  RegisterRegAlloc(const char *N, const char *D, FunctionPassCtor C)
  : MachinePassRegistryNode(N, D, (MachinePassCtor)C)
  { 
     Registry.Add(this); 
  }
  ~RegisterRegAlloc() { Registry.Remove(this); }
  

  // Accessors.
  //
  RegisterRegAlloc *getNext() const {
    return (RegisterRegAlloc *)MachinePassRegistryNode::getNext();
  }
  static RegisterRegAlloc *getList() {
    return (RegisterRegAlloc *)Registry.getList();
  }
  static FunctionPassCtor getDefault() {
    return (FunctionPassCtor)Registry.getDefault();
  }
  static void setDefault(FunctionPassCtor C) {
    Registry.setDefault((MachinePassCtor)C);
  }
  static void setListener(MachinePassRegistryListener *L) {
    Registry.setListener(L);
  }
  
};

} // end namespace llvm


#endif
