//===-- llvm/MC/MCSchedule.h - Scheduling -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the classes used to describe a subtarget's machine model
// for scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSCHEDULE_H
#define LLVM_MC_MCSCHEDULE_H

#include "llvm/Support/DataTypes.h"
#include <cassert>

namespace llvm {

struct InstrItinerary;

/// Define a kind of processor resource that will be modeled by the scheduler.
struct MCProcResourceDesc {
#ifndef NDEBUG
  const char *Name;
#endif
  unsigned NumUnits; // Number of resource of this kind
  unsigned SuperIdx; // Index of the resources kind that contains this kind.

  // Number of resources that may be buffered.
  //
  // Buffered resources (BufferSize != 0) may be consumed at some indeterminate
  // cycle after dispatch. This should be used for out-of-order cpus when
  // instructions that use this resource can be buffered in a reservaton
  // station.
  //
  // Unbuffered resources (BufferSize == 0) always consume their resource some
  // fixed number of cycles after dispatch. If a resource is unbuffered, then
  // the scheduler will avoid scheduling instructions with conflicting resources
  // in the same cycle. This is for in-order cpus, or the in-order portion of
  // an out-of-order cpus.
  int BufferSize;

  bool operator==(const MCProcResourceDesc &Other) const {
    return NumUnits == Other.NumUnits && SuperIdx == Other.SuperIdx
      && BufferSize == Other.BufferSize;
  }
};

/// Identify one of the processor resource kinds consumed by a particular
/// scheduling class for the specified number of cycles.
struct MCWriteProcResEntry {
  unsigned ProcResourceIdx;
  unsigned Cycles;

  bool operator==(const MCWriteProcResEntry &Other) const {
    return ProcResourceIdx == Other.ProcResourceIdx && Cycles == Other.Cycles;
  }
};

/// Specify the latency in cpu cycles for a particular scheduling class and def
/// index. -1 indicates an invalid latency. Heuristics would typically consider
/// an instruction with invalid latency to have infinite latency.  Also identify
/// the WriteResources of this def. When the operand expands to a sequence of
/// writes, this ID is the last write in the sequence.
struct MCWriteLatencyEntry {
  int Cycles;
  unsigned WriteResourceID;

  bool operator==(const MCWriteLatencyEntry &Other) const {
    return Cycles == Other.Cycles && WriteResourceID == Other.WriteResourceID;
  }
};

/// Specify the number of cycles allowed after instruction issue before a
/// particular use operand reads its registers. This effectively reduces the
/// write's latency. Here we allow negative cycles for corner cases where
/// latency increases. This rule only applies when the entry's WriteResource
/// matches the write's WriteResource.
///
/// MCReadAdvanceEntries are sorted first by operand index (UseIdx), then by
/// WriteResourceIdx.
struct MCReadAdvanceEntry {
  unsigned UseIdx;
  unsigned WriteResourceID;
  int Cycles;

  bool operator==(const MCReadAdvanceEntry &Other) const {
    return UseIdx == Other.UseIdx && WriteResourceID == Other.WriteResourceID
      && Cycles == Other.Cycles;
  }
};

/// Summarize the scheduling resources required for an instruction of a
/// particular scheduling class.
///
/// Defined as an aggregate struct for creating tables with initializer lists.
struct MCSchedClassDesc {
  static const unsigned short InvalidNumMicroOps = UINT16_MAX;
  static const unsigned short VariantNumMicroOps = UINT16_MAX - 1;

#ifndef NDEBUG
  const char* Name;
#endif
  unsigned short NumMicroOps;
  bool     BeginGroup;
  bool     EndGroup;
  unsigned WriteProcResIdx; // First index into WriteProcResTable.
  unsigned NumWriteProcResEntries;
  unsigned WriteLatencyIdx; // First index into WriteLatencyTable.
  unsigned NumWriteLatencyEntries;
  unsigned ReadAdvanceIdx; // First index into ReadAdvanceTable.
  unsigned NumReadAdvanceEntries;

  bool isValid() const {
    return NumMicroOps != InvalidNumMicroOps;
  }
  bool isVariant() const {
    return NumMicroOps == VariantNumMicroOps;
  }
};

/// Machine model for scheduling, bundling, and heuristics.
///
/// The machine model directly provides basic information about the
/// microarchitecture to the scheduler in the form of properties. It also
/// optionally refers to scheduler resource tables and itinerary
/// tables. Scheduler resource tables model the latency and cost for each
/// instruction type. Itinerary tables are an independent mechanism that
/// provides a detailed reservation table describing each cycle of instruction
/// execution. Subtargets may define any or all of the above categories of data
/// depending on the type of CPU and selected scheduler.
struct MCSchedModel {
  // IssueWidth is the maximum number of instructions that may be scheduled in
  // the same per-cycle group.
  unsigned IssueWidth;
  static const unsigned DefaultIssueWidth = 1;

  // MicroOpBufferSize is the number of micro-ops that the processor may buffer
  // for out-of-order execution.
  //
  // "0" means operations that are not ready in this cycle are not considered
  // for scheduling (they go in the pending queue). Latency is paramount. This
  // may be more efficient if many instructions are pending in a schedule.
  //
  // "1" means all instructions are considered for scheduling regardless of
  // whether they are ready in this cycle. Latency still causes issue stalls,
  // but we balance those stalls against other heuristics.
  //
  // "> 1" means the processor is out-of-order. This is a machine independent
  // estimate of highly machine specific characteristics such as the register
  // renaming pool and reorder buffer.
  unsigned MicroOpBufferSize;
  static const unsigned DefaultMicroOpBufferSize = 0;

  // LoopMicroOpBufferSize is the number of micro-ops that the processor may
  // buffer for optimized loop execution. More generally, this represents the
  // optimal number of micro-ops in a loop body. A loop may be partially
  // unrolled to bring the count of micro-ops in the loop body closer to this
  // number.
  unsigned LoopMicroOpBufferSize;
  static const unsigned DefaultLoopMicroOpBufferSize = 0;

  // LoadLatency is the expected latency of load instructions.
  //
  // If MinLatency >= 0, this may be overriden for individual load opcodes by
  // InstrItinerary OperandCycles.
  unsigned LoadLatency;
  static const unsigned DefaultLoadLatency = 4;

  // HighLatency is the expected latency of "very high latency" operations.
  // See TargetInstrInfo::isHighLatencyDef().
  // By default, this is set to an arbitrarily high number of cycles
  // likely to have some impact on scheduling heuristics.
  // If MinLatency >= 0, this may be overriden by InstrItinData OperandCycles.
  unsigned HighLatency;
  static const unsigned DefaultHighLatency = 10;

  // MispredictPenalty is the typical number of extra cycles the processor
  // takes to recover from a branch misprediction.
  unsigned MispredictPenalty;
  static const unsigned DefaultMispredictPenalty = 10;
  
  bool PostRAScheduler; // default value is false

  bool CompleteModel;

  unsigned ProcID;
  const MCProcResourceDesc *ProcResourceTable;
  const MCSchedClassDesc *SchedClassTable;
  unsigned NumProcResourceKinds;
  unsigned NumSchedClasses;
  // Instruction itinerary tables used by InstrItineraryData.
  friend class InstrItineraryData;
  const InstrItinerary *InstrItineraries;

  unsigned getProcessorID() const { return ProcID; }

  /// Does this machine model include instruction-level scheduling.
  bool hasInstrSchedModel() const { return SchedClassTable; }

  /// Return true if this machine model data for all instructions with a
  /// scheduling class (itinerary class or SchedRW list).
  bool isComplete() const { return CompleteModel; }

  unsigned getNumProcResourceKinds() const {
    return NumProcResourceKinds;
  }

  const MCProcResourceDesc *getProcResource(unsigned ProcResourceIdx) const {
    assert(hasInstrSchedModel() && "No scheduling machine model");

    assert(ProcResourceIdx < NumProcResourceKinds && "bad proc resource idx");
    return &ProcResourceTable[ProcResourceIdx];
  }

  const MCSchedClassDesc *getSchedClassDesc(unsigned SchedClassIdx) const {
    assert(hasInstrSchedModel() && "No scheduling machine model");

    assert(SchedClassIdx < NumSchedClasses && "bad scheduling class idx");
    return &SchedClassTable[SchedClassIdx];
  }

  /// Returns the default initialized model.
  static const MCSchedModel &GetDefaultSchedModel() { return Default; }
  static const MCSchedModel Default;
};

} // End llvm namespace

#endif
