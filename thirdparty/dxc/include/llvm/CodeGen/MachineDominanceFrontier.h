//===- llvm/CodeGen/MachineDominanceFrontier.h ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEDOMINANCEFRONTIER_H
#define LLVM_CODEGEN_MACHINEDOMINANCEFRONTIER_H

#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"


namespace llvm {

class MachineDominanceFrontier : public MachineFunctionPass {
  ForwardDominanceFrontierBase<MachineBasicBlock> Base;
public:
  typedef DominatorTreeBase<MachineBasicBlock> DomTreeT;
  typedef DomTreeNodeBase<MachineBasicBlock> DomTreeNodeT;
  typedef DominanceFrontierBase<MachineBasicBlock>::DomSetType DomSetType;
  typedef DominanceFrontierBase<MachineBasicBlock>::iterator iterator;
  typedef DominanceFrontierBase<MachineBasicBlock>::const_iterator const_iterator;

  void operator=(const MachineDominanceFrontier &) = delete;
  MachineDominanceFrontier(const MachineDominanceFrontier &) = delete;

  static char ID;

  MachineDominanceFrontier();

  DominanceFrontierBase<MachineBasicBlock> &getBase() {
    return Base;
  }

  inline const std::vector<MachineBasicBlock*> &getRoots() const {
    return Base.getRoots();
  }

  MachineBasicBlock *getRoot() const {
    return Base.getRoot();
  }

  bool isPostDominator() const {
    return Base.isPostDominator();
  }

  iterator begin() {
    return Base.begin();
  }

  const_iterator begin() const {
    return Base.begin();
  }

  iterator end() {
    return Base.end();
  }

  const_iterator end() const {
    return Base.end();
  }

  iterator find(MachineBasicBlock *B) {
    return Base.find(B);
  }

  const_iterator find(MachineBasicBlock *B) const {
    return Base.find(B);
  }

  iterator addBasicBlock(MachineBasicBlock *BB, const DomSetType &frontier) {
    return Base.addBasicBlock(BB, frontier);
  }

  void removeBlock(MachineBasicBlock *BB) {
    return Base.removeBlock(BB);
  }

  void addToFrontier(iterator I, MachineBasicBlock *Node) {
    return Base.addToFrontier(I, Node);
  }

  void removeFromFrontier(iterator I, MachineBasicBlock *Node) {
    return Base.removeFromFrontier(I, Node);
  }

  bool compareDomSet(DomSetType &DS1, const DomSetType &DS2) const {
    return Base.compareDomSet(DS1, DS2);
  }

  bool compare(DominanceFrontierBase<MachineBasicBlock> &Other) const {
    return Base.compare(Other);
  }

  bool runOnMachineFunction(MachineFunction &F) override;

  void releaseMemory() override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

}

#endif
