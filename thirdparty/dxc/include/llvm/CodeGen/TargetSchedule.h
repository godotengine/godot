//===-- llvm/CodeGen/TargetSchedule.h - Sched Machine Model -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a wrapper around MCSchedModel that allows the interface to
// benefit from information currently only available in TargetInstrInfo.
// Ideally, the scheduling interface would be fully defined in the MC layer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_TARGETSCHEDULE_H
#define LLVM_CODEGEN_TARGETSCHEDULE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/Target/TargetSubtargetInfo.h"

namespace llvm {

class TargetRegisterInfo;
class TargetSubtargetInfo;
class TargetInstrInfo;
class MachineInstr;

/// Provide an instruction scheduling machine model to CodeGen passes.
class TargetSchedModel {
  // For efficiency, hold a copy of the statically defined MCSchedModel for this
  // processor.
  MCSchedModel SchedModel;
  InstrItineraryData InstrItins;
  const TargetSubtargetInfo *STI;
  const TargetInstrInfo *TII;

  SmallVector<unsigned, 16> ResourceFactors;
  unsigned MicroOpFactor; // Multiply to normalize microops to resource units.
  unsigned ResourceLCM;   // Resource units per cycle. Latency normalization factor.

  unsigned computeInstrLatency(const MCSchedClassDesc &SCDesc) const;

public:
  TargetSchedModel(): SchedModel(MCSchedModel::GetDefaultSchedModel()), STI(nullptr), TII(nullptr) {}

  /// \brief Initialize the machine model for instruction scheduling.
  ///
  /// The machine model API keeps a copy of the top-level MCSchedModel table
  /// indices and may query TargetSubtargetInfo and TargetInstrInfo to resolve
  /// dynamic properties.
  void init(const MCSchedModel &sm, const TargetSubtargetInfo *sti,
            const TargetInstrInfo *tii);

  /// Return the MCSchedClassDesc for this instruction.
  const MCSchedClassDesc *resolveSchedClass(const MachineInstr *MI) const;

  /// \brief TargetInstrInfo getter.
  const TargetInstrInfo *getInstrInfo() const { return TII; }

  /// \brief Return true if this machine model includes an instruction-level
  /// scheduling model.
  ///
  /// This is more detailed than the course grain IssueWidth and default
  /// latency properties, but separate from the per-cycle itinerary data.
  bool hasInstrSchedModel() const;

  const MCSchedModel *getMCSchedModel() const { return &SchedModel; }

  /// \brief Return true if this machine model includes cycle-to-cycle itinerary
  /// data.
  ///
  /// This models scheduling at each stage in the processor pipeline.
  bool hasInstrItineraries() const;

  const InstrItineraryData *getInstrItineraries() const {
    if (hasInstrItineraries())
      return &InstrItins;
    return nullptr;
  }

  /// \brief Identify the processor corresponding to the current subtarget.
  unsigned getProcessorID() const { return SchedModel.getProcessorID(); }

  /// \brief Maximum number of micro-ops that may be scheduled per cycle.
  unsigned getIssueWidth() const { return SchedModel.IssueWidth; }

  /// \brief Return the number of issue slots required for this MI.
  unsigned getNumMicroOps(const MachineInstr *MI,
                          const MCSchedClassDesc *SC = nullptr) const;

  /// \brief Get the number of kinds of resources for this target.
  unsigned getNumProcResourceKinds() const {
    return SchedModel.getNumProcResourceKinds();
  }

  /// \brief Get a processor resource by ID for convenience.
  const MCProcResourceDesc *getProcResource(unsigned PIdx) const {
    return SchedModel.getProcResource(PIdx);
  }

#ifndef NDEBUG
  const char *getResourceName(unsigned PIdx) const {
    if (!PIdx)
      return "MOps";
    return SchedModel.getProcResource(PIdx)->Name;
  }
#endif

  typedef const MCWriteProcResEntry *ProcResIter;

  // \brief Get an iterator into the processor resources consumed by this
  // scheduling class.
  ProcResIter getWriteProcResBegin(const MCSchedClassDesc *SC) const {
    // The subtarget holds a single resource table for all processors.
    return STI->getWriteProcResBegin(SC);
  }
  ProcResIter getWriteProcResEnd(const MCSchedClassDesc *SC) const {
    return STI->getWriteProcResEnd(SC);
  }

  /// \brief Multiply the number of units consumed for a resource by this factor
  /// to normalize it relative to other resources.
  unsigned getResourceFactor(unsigned ResIdx) const {
    return ResourceFactors[ResIdx];
  }

  /// \brief Multiply number of micro-ops by this factor to normalize it
  /// relative to other resources.
  unsigned getMicroOpFactor() const {
    return MicroOpFactor;
  }

  /// \brief Multiply cycle count by this factor to normalize it relative to
  /// other resources. This is the number of resource units per cycle.
  unsigned getLatencyFactor() const {
    return ResourceLCM;
  }

  /// \brief Number of micro-ops that may be buffered for OOO execution.
  unsigned getMicroOpBufferSize() const { return SchedModel.MicroOpBufferSize; }

  /// \brief Number of resource units that may be buffered for OOO execution.
  /// \return The buffer size in resource units or -1 for unlimited.
  int getResourceBufferSize(unsigned PIdx) const {
    return SchedModel.getProcResource(PIdx)->BufferSize;
  }

  /// \brief Compute operand latency based on the available machine model.
  ///
  /// Compute and return the latency of the given data dependent def and use
  /// when the operand indices are already known. UseMI may be NULL for an
  /// unknown user.
  unsigned computeOperandLatency(const MachineInstr *DefMI, unsigned DefOperIdx,
                                 const MachineInstr *UseMI, unsigned UseOperIdx)
    const;

  /// \brief Compute the instruction latency based on the available machine
  /// model.
  ///
  /// Compute and return the expected latency of this instruction independent of
  /// a particular use. computeOperandLatency is the preferred API, but this is
  /// occasionally useful to help estimate instruction cost.
  ///
  /// If UseDefaultDefLatency is false and no new machine sched model is
  /// present this method falls back to TII->getInstrLatency with an empty
  /// instruction itinerary (this is so we preserve the previous behavior of the
  /// if converter after moving it to TargetSchedModel).
  unsigned computeInstrLatency(const MachineInstr *MI,
                               bool UseDefaultDefLatency = true) const;
  unsigned computeInstrLatency(unsigned Opcode) const;

  /// \brief Output dependency latency of a pair of defs of the same register.
  ///
  /// This is typically one cycle.
  unsigned computeOutputLatency(const MachineInstr *DefMI, unsigned DefIdx,
                                const MachineInstr *DepMI) const;
};

} // namespace llvm

#endif
