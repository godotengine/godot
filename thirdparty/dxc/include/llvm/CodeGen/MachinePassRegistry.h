//===-- llvm/CodeGen/MachinePassRegistry.h ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the mechanics for machine function pass registries.  A
// function pass registry (MachinePassRegistry) is auto filled by the static
// constructors of MachinePassRegistryNode.  Further there is a command line
// parser (RegisterPassParser) which listens to each registry for additions
// and deletions, so that the appropriate command option is updated.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEPASSREGISTRY_H
#define LLVM_CODEGEN_MACHINEPASSREGISTRY_H

#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {

typedef void *(*MachinePassCtor)();


//===----------------------------------------------------------------------===//
///
/// MachinePassRegistryListener - Listener to adds and removals of nodes in
/// registration list.
///
//===----------------------------------------------------------------------===//
class MachinePassRegistryListener {
  virtual void anchor();
public:
  MachinePassRegistryListener() {}
  virtual ~MachinePassRegistryListener() {}
  virtual void NotifyAdd(const char *N, MachinePassCtor C, const char *D) = 0;
  virtual void NotifyRemove(const char *N) = 0;
};


//===----------------------------------------------------------------------===//
///
/// MachinePassRegistryNode - Machine pass node stored in registration list.
///
//===----------------------------------------------------------------------===//
class MachinePassRegistryNode {

private:

  MachinePassRegistryNode *Next;        // Next function pass in list.
  const char *Name;                     // Name of function pass.
  const char *Description;              // Description string.
  MachinePassCtor Ctor;                 // Function pass creator.

public:

  MachinePassRegistryNode(const char *N, const char *D, MachinePassCtor C)
  : Next(nullptr)
  , Name(N)
  , Description(D)
  , Ctor(C)
  {}

  // Accessors
  MachinePassRegistryNode *getNext()      const { return Next; }
  MachinePassRegistryNode **getNextAddress()    { return &Next; }
  const char *getName()                   const { return Name; }
  const char *getDescription()            const { return Description; }
  MachinePassCtor getCtor()               const { return Ctor; }
  void setNext(MachinePassRegistryNode *N)      { Next = N; }

};


//===----------------------------------------------------------------------===//
///
/// MachinePassRegistry - Track the registration of machine passes.
///
//===----------------------------------------------------------------------===//
class MachinePassRegistry {

private:

  MachinePassRegistryNode *List;        // List of registry nodes.
  MachinePassCtor Default;              // Default function pass creator.
  MachinePassRegistryListener* Listener;// Listener for list adds are removes.

public:

  // NO CONSTRUCTOR - we don't want static constructor ordering to mess
  // with the registry.

  // Accessors.
  //
  MachinePassRegistryNode *getList()                    { return List; }
  MachinePassCtor getDefault()                          { return Default; }
  void setDefault(MachinePassCtor C)                    { Default = C; }
  void setDefault(StringRef Name);
  void setListener(MachinePassRegistryListener *L)      { Listener = L; }

  /// Add - Adds a function pass to the registration list.
  ///
  void Add(MachinePassRegistryNode *Node);

  /// Remove - Removes a function pass from the registration list.
  ///
  void Remove(MachinePassRegistryNode *Node);

};


//===----------------------------------------------------------------------===//
///
/// RegisterPassParser class - Handle the addition of new machine passes.
///
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
template<class RegistryClass>
class RegisterPassParser : public MachinePassRegistryListener,
                   public cl::parser<typename RegistryClass::FunctionPassCtor> {
public:
  RegisterPassParser(cl::Option &O)
      : cl::parser<typename RegistryClass::FunctionPassCtor>(O) {}
  ~RegisterPassParser() override { RegistryClass::setListener(nullptr); }

  void initialize() {
    cl::parser<typename RegistryClass::FunctionPassCtor>::initialize();

    // Add existing passes to option.
    for (RegistryClass *Node = RegistryClass::getList();
         Node; Node = Node->getNext()) {
      this->addLiteralOption(Node->getName(),
                      (typename RegistryClass::FunctionPassCtor)Node->getCtor(),
                             Node->getDescription());
    }

    // Make sure we listen for list changes.
    RegistryClass::setListener(this);
  }

  // Implement the MachinePassRegistryListener callbacks.
  //
  void NotifyAdd(const char *N, MachinePassCtor C, const char *D) override {
    this->addLiteralOption(N, (typename RegistryClass::FunctionPassCtor)C, D);
  }
  void NotifyRemove(const char *N) override {
    this->removeLiteralOption(N);
  }
};


} // end namespace llvm

#endif
