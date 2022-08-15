//=- llvm/CodeGen/DFAPacketizer.h - DFA Packetizer for VLIW ---*- C++ -*-=====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This class implements a deterministic finite automaton (DFA) based
// packetizing mechanism for VLIW architectures. It provides APIs to
// determine whether there exists a legal mapping of instructions to
// functional unit assignments in a packet. The DFA is auto-generated from
// the target's Schedule.td file.
//
// A DFA consists of 3 major elements: states, inputs, and transitions. For
// the packetizing mechanism, the input is the set of instruction classes for
// a target. The state models all possible combinations of functional unit
// consumption for a given set of instructions in a packet. A transition
// models the addition of an instruction to a packet. In the DFA constructed
// by this class, if an instruction can be added to a packet, then a valid
// transition exists from the corresponding state. Invalid transitions
// indicate that the instruction cannot be added to the current packet.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_DFAPACKETIZER_H
#define LLVM_CODEGEN_DFAPACKETIZER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include <map>

namespace llvm {

class MCInstrDesc;
class MachineInstr;
class MachineLoopInfo;
class MachineDominatorTree;
class InstrItineraryData;
class DefaultVLIWScheduler;
class SUnit;

class DFAPacketizer {
private:
  typedef std::pair<unsigned, unsigned> UnsignPair;
  const InstrItineraryData *InstrItins;
  int CurrentState;
  const int (*DFAStateInputTable)[2];
  const unsigned *DFAStateEntryTable;

  // CachedTable is a map from <FromState, Input> to ToState.
  DenseMap<UnsignPair, unsigned> CachedTable;

  // ReadTable - Read the DFA transition table and update CachedTable.
  void ReadTable(unsigned int state);

public:
  DFAPacketizer(const InstrItineraryData *I, const int (*SIT)[2],
                const unsigned *SET);

  // Reset the current state to make all resources available.
  void clearResources() {
    CurrentState = 0;
  }

  // canReserveResources - Check if the resources occupied by a MCInstrDesc
  // are available in the current state.
  bool canReserveResources(const llvm::MCInstrDesc *MID);

  // reserveResources - Reserve the resources occupied by a MCInstrDesc and
  // change the current state to reflect that change.
  void reserveResources(const llvm::MCInstrDesc *MID);

  // canReserveResources - Check if the resources occupied by a machine
  // instruction are available in the current state.
  bool canReserveResources(llvm::MachineInstr *MI);

  // reserveResources - Reserve the resources occupied by a machine
  // instruction and change the current state to reflect that change.
  void reserveResources(llvm::MachineInstr *MI);

  const InstrItineraryData *getInstrItins() const { return InstrItins; }
};

// VLIWPacketizerList - Implements a simple VLIW packetizer using DFA. The
// packetizer works on machine basic blocks. For each instruction I in BB, the
// packetizer consults the DFA to see if machine resources are available to
// execute I. If so, the packetizer checks if I depends on any instruction J in
// the current packet. If no dependency is found, I is added to current packet
// and machine resource is marked as taken. If any dependency is found, a target
// API call is made to prune the dependence.
class VLIWPacketizerList {
protected:
  MachineFunction &MF;
  const TargetInstrInfo *TII;

  // The VLIW Scheduler.
  DefaultVLIWScheduler *VLIWScheduler;

  // Vector of instructions assigned to the current packet.
  std::vector<MachineInstr*> CurrentPacketMIs;
  // DFA resource tracker.
  DFAPacketizer *ResourceTracker;

  // Generate MI -> SU map.
  std::map<MachineInstr*, SUnit*> MIToSUnit;

public:
  VLIWPacketizerList(MachineFunction &MF, MachineLoopInfo &MLI, bool IsPostRA);

  virtual ~VLIWPacketizerList();

  // PacketizeMIs - Implement this API in the backend to bundle instructions.
  void PacketizeMIs(MachineBasicBlock *MBB,
                    MachineBasicBlock::iterator BeginItr,
                    MachineBasicBlock::iterator EndItr);

  // getResourceTracker - return ResourceTracker
  DFAPacketizer *getResourceTracker() {return ResourceTracker;}

  // addToPacket - Add MI to the current packet.
  virtual MachineBasicBlock::iterator addToPacket(MachineInstr *MI) {
    MachineBasicBlock::iterator MII = MI;
    CurrentPacketMIs.push_back(MI);
    ResourceTracker->reserveResources(MI);
    return MII;
  }

  // endPacket - End the current packet.
  void endPacket(MachineBasicBlock *MBB, MachineInstr *MI);

  // initPacketizerState - perform initialization before packetizing
  // an instruction. This function is supposed to be overrided by
  // the target dependent packetizer.
  virtual void initPacketizerState() { return; }

  // ignorePseudoInstruction - Ignore bundling of pseudo instructions.
  virtual bool ignorePseudoInstruction(MachineInstr *I,
                                       MachineBasicBlock *MBB) {
    return false;
  }

  // isSoloInstruction - return true if instruction MI can not be packetized
  // with any other instruction, which means that MI itself is a packet.
  virtual bool isSoloInstruction(MachineInstr *MI) {
    return true;
  }

  // isLegalToPacketizeTogether - Is it legal to packetize SUI and SUJ
  // together.
  virtual bool isLegalToPacketizeTogether(SUnit *SUI, SUnit *SUJ) {
    return false;
  }

  // isLegalToPruneDependencies - Is it legal to prune dependece between SUI
  // and SUJ.
  virtual bool isLegalToPruneDependencies(SUnit *SUI, SUnit *SUJ) {
    return false;
  }

};
}

#endif
