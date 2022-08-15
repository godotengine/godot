//===-- CodeGen/MachineInstr.cpp ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the machine function pass registry for register allocators
// and instruction schedulers.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachinePassRegistry.h"

using namespace llvm;

void MachinePassRegistryListener::anchor() { }

/// setDefault - Set the default constructor by name.
void MachinePassRegistry::setDefault(StringRef Name) {
  MachinePassCtor Ctor = nullptr;
  for(MachinePassRegistryNode *R = getList(); R; R = R->getNext()) {
    if (R->getName() == Name) {
      Ctor = R->getCtor();
      break;
    }
  }
  assert(Ctor && "Unregistered pass name");
  setDefault(Ctor);
}

/// Add - Adds a function pass to the registration list.
///
void MachinePassRegistry::Add(MachinePassRegistryNode *Node) {
  Node->setNext(List);
  List = Node;
  if (Listener) Listener->NotifyAdd(Node->getName(),
                                    Node->getCtor(),
                                    Node->getDescription());
}


/// Remove - Removes a function pass from the registration list.
///
void MachinePassRegistry::Remove(MachinePassRegistryNode *Node) {
  for (MachinePassRegistryNode **I = &List; *I; I = (*I)->getNextAddress()) {
    if (*I == Node) {
      if (Listener) Listener->NotifyRemove(Node->getName());
      *I = (*I)->getNext();
      break;
    }
  }
}
