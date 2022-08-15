//===-- llvm/Target/TargetSchedule.cpp - Sched Machine Model ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a wrapper around MCSchedModel that allows the interface
// to benefit from information currently only available in TargetInstrInfo.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

static cl::opt<bool> EnableSchedModel("schedmodel", cl::Hidden, cl::init(true),
  cl::desc("Use TargetSchedModel for latency lookup"));

static cl::opt<bool> EnableSchedItins("scheditins", cl::Hidden, cl::init(true),
  cl::desc("Use InstrItineraryData for latency lookup"));

bool TargetSchedModel::hasInstrSchedModel() const {
  return EnableSchedModel && SchedModel.hasInstrSchedModel();
}

bool TargetSchedModel::hasInstrItineraries() const {
  return EnableSchedItins && !InstrItins.isEmpty();
}

static unsigned gcd(unsigned Dividend, unsigned Divisor) {
  // Dividend and Divisor will be naturally swapped as needed.
  while(Divisor) {
    unsigned Rem = Dividend % Divisor;
    Dividend = Divisor;
    Divisor = Rem;
  };
  return Dividend;
}
static unsigned lcm(unsigned A, unsigned B) {
  unsigned LCM = (uint64_t(A) * B) / gcd(A, B);
  assert((LCM >= A && LCM >= B) && "LCM overflow");
  return LCM;
}

void TargetSchedModel::init(const MCSchedModel &sm,
                            const TargetSubtargetInfo *sti,
                            const TargetInstrInfo *tii) {
  SchedModel = sm;
  STI = sti;
  TII = tii;
  STI->initInstrItins(InstrItins);

  unsigned NumRes = SchedModel.getNumProcResourceKinds();
  ResourceFactors.resize(NumRes);
  ResourceLCM = SchedModel.IssueWidth;
  for (unsigned Idx = 0; Idx < NumRes; ++Idx) {
    unsigned NumUnits = SchedModel.getProcResource(Idx)->NumUnits;
    if (NumUnits > 0)
      ResourceLCM = lcm(ResourceLCM, NumUnits);
  }
  MicroOpFactor = ResourceLCM / SchedModel.IssueWidth;
  for (unsigned Idx = 0; Idx < NumRes; ++Idx) {
    unsigned NumUnits = SchedModel.getProcResource(Idx)->NumUnits;
    ResourceFactors[Idx] = NumUnits ? (ResourceLCM / NumUnits) : 0;
  }
}

unsigned TargetSchedModel::getNumMicroOps(const MachineInstr *MI,
                                          const MCSchedClassDesc *SC) const {
  if (hasInstrItineraries()) {
    int UOps = InstrItins.getNumMicroOps(MI->getDesc().getSchedClass());
    return (UOps >= 0) ? UOps : TII->getNumMicroOps(&InstrItins, MI);
  }
  if (hasInstrSchedModel()) {
    if (!SC)
      SC = resolveSchedClass(MI);
    if (SC->isValid())
      return SC->NumMicroOps;
  }
  return MI->isTransient() ? 0 : 1;
}

// The machine model may explicitly specify an invalid latency, which
// effectively means infinite latency. Since users of the TargetSchedule API
// don't know how to handle this, we convert it to a very large latency that is
// easy to distinguish when debugging the DAG but won't induce overflow.
static unsigned capLatency(int Cycles) {
  return Cycles >= 0 ? Cycles : 1000;
}

/// Return the MCSchedClassDesc for this instruction. Some SchedClasses require
/// evaluation of predicates that depend on instruction operands or flags.
const MCSchedClassDesc *TargetSchedModel::
resolveSchedClass(const MachineInstr *MI) const {

  // Get the definition's scheduling class descriptor from this machine model.
  unsigned SchedClass = MI->getDesc().getSchedClass();
  const MCSchedClassDesc *SCDesc = SchedModel.getSchedClassDesc(SchedClass);
  if (!SCDesc->isValid())
    return SCDesc;

#ifndef NDEBUG
  unsigned NIter = 0;
#endif
  while (SCDesc->isVariant()) {
    assert(++NIter < 6 && "Variants are nested deeper than the magic number");

    SchedClass = STI->resolveSchedClass(SchedClass, MI, this);
    SCDesc = SchedModel.getSchedClassDesc(SchedClass);
  }
  return SCDesc;
}

/// Find the def index of this operand. This index maps to the machine model and
/// is independent of use operands. Def operands may be reordered with uses or
/// merged with uses without affecting the def index (e.g. before/after
/// regalloc). However, an instruction's def operands must never be reordered
/// with respect to each other.
static unsigned findDefIdx(const MachineInstr *MI, unsigned DefOperIdx) {
  unsigned DefIdx = 0;
  for (unsigned i = 0; i != DefOperIdx; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isDef())
      ++DefIdx;
  }
  return DefIdx;
}

/// Find the use index of this operand. This is independent of the instruction's
/// def operands.
///
/// Note that uses are not determined by the operand's isUse property, which
/// is simply the inverse of isDef. Here we consider any readsReg operand to be
/// a "use". The machine model allows an operand to be both a Def and Use.
static unsigned findUseIdx(const MachineInstr *MI, unsigned UseOperIdx) {
  unsigned UseIdx = 0;
  for (unsigned i = 0; i != UseOperIdx; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.readsReg())
      ++UseIdx;
  }
  return UseIdx;
}

// Top-level API for clients that know the operand indices.
unsigned TargetSchedModel::computeOperandLatency(
  const MachineInstr *DefMI, unsigned DefOperIdx,
  const MachineInstr *UseMI, unsigned UseOperIdx) const {

  if (!hasInstrSchedModel() && !hasInstrItineraries())
    return TII->defaultDefLatency(SchedModel, DefMI);

  if (hasInstrItineraries()) {
    int OperLatency = 0;
    if (UseMI) {
      OperLatency = TII->getOperandLatency(&InstrItins, DefMI, DefOperIdx,
                                           UseMI, UseOperIdx);
    }
    else {
      unsigned DefClass = DefMI->getDesc().getSchedClass();
      OperLatency = InstrItins.getOperandCycle(DefClass, DefOperIdx);
    }
    if (OperLatency >= 0)
      return OperLatency;

    // No operand latency was found.
    unsigned InstrLatency = TII->getInstrLatency(&InstrItins, DefMI);

    // Expected latency is the max of the stage latency and itinerary props.
    // Rather than directly querying InstrItins stage latency, we call a TII
    // hook to allow subtargets to specialize latency. This hook is only
    // applicable to the InstrItins model. InstrSchedModel should model all
    // special cases without TII hooks.
    InstrLatency = std::max(InstrLatency,
                            TII->defaultDefLatency(SchedModel, DefMI));
    return InstrLatency;
  }
  // hasInstrSchedModel()
  const MCSchedClassDesc *SCDesc = resolveSchedClass(DefMI);
  unsigned DefIdx = findDefIdx(DefMI, DefOperIdx);
  if (DefIdx < SCDesc->NumWriteLatencyEntries) {
    // Lookup the definition's write latency in SubtargetInfo.
    const MCWriteLatencyEntry *WLEntry =
      STI->getWriteLatencyEntry(SCDesc, DefIdx);
    unsigned WriteID = WLEntry->WriteResourceID;
    unsigned Latency = capLatency(WLEntry->Cycles);
    if (!UseMI)
      return Latency;

    // Lookup the use's latency adjustment in SubtargetInfo.
    const MCSchedClassDesc *UseDesc = resolveSchedClass(UseMI);
    if (UseDesc->NumReadAdvanceEntries == 0)
      return Latency;
    unsigned UseIdx = findUseIdx(UseMI, UseOperIdx);
    int Advance = STI->getReadAdvanceCycles(UseDesc, UseIdx, WriteID);
    if (Advance > 0 && (unsigned)Advance > Latency) // unsigned wrap
      return 0;
    return Latency - Advance;
  }
  // If DefIdx does not exist in the model (e.g. implicit defs), then return
  // unit latency (defaultDefLatency may be too conservative).
#ifndef NDEBUG
  if (SCDesc->isValid() && !DefMI->getOperand(DefOperIdx).isImplicit()
      && !DefMI->getDesc().OpInfo[DefOperIdx].isOptionalDef()
      && SchedModel.isComplete()) {
    std::string Err;
    raw_string_ostream ss(Err);
    ss << "DefIdx " << DefIdx << " exceeds machine model writes for "
       << *DefMI;
    report_fatal_error(ss.str());
  }
#endif
  // FIXME: Automatically giving all implicit defs defaultDefLatency is
  // undesirable. We should only do it for defs that are known to the MC
  // desc like flags. Truly implicit defs should get 1 cycle latency.
  return DefMI->isTransient() ? 0 : TII->defaultDefLatency(SchedModel, DefMI);
}

unsigned
TargetSchedModel::computeInstrLatency(const MCSchedClassDesc &SCDesc) const {
  unsigned Latency = 0;
  for (unsigned DefIdx = 0, DefEnd = SCDesc.NumWriteLatencyEntries;
       DefIdx != DefEnd; ++DefIdx) {
    // Lookup the definition's write latency in SubtargetInfo.
    const MCWriteLatencyEntry *WLEntry =
      STI->getWriteLatencyEntry(&SCDesc, DefIdx);
    Latency = std::max(Latency, capLatency(WLEntry->Cycles));
  }
  return Latency;
}

unsigned TargetSchedModel::computeInstrLatency(unsigned Opcode) const {
  assert(hasInstrSchedModel() && "Only call this function with a SchedModel");

  unsigned SCIdx = TII->get(Opcode).getSchedClass();
  const MCSchedClassDesc *SCDesc = SchedModel.getSchedClassDesc(SCIdx);

  if (SCDesc->isValid() && !SCDesc->isVariant())
    return computeInstrLatency(*SCDesc);

  llvm_unreachable("No MI sched latency");
}

unsigned
TargetSchedModel::computeInstrLatency(const MachineInstr *MI,
                                      bool UseDefaultDefLatency) const {
  // For the itinerary model, fall back to the old subtarget hook.
  // Allow subtargets to compute Bundle latencies outside the machine model.
  if (hasInstrItineraries() || MI->isBundle() ||
      (!hasInstrSchedModel() && !UseDefaultDefLatency))
    return TII->getInstrLatency(&InstrItins, MI);

  if (hasInstrSchedModel()) {
    const MCSchedClassDesc *SCDesc = resolveSchedClass(MI);
    if (SCDesc->isValid())
      return computeInstrLatency(*SCDesc);
  }
  return TII->defaultDefLatency(SchedModel, MI);
}

unsigned TargetSchedModel::
computeOutputLatency(const MachineInstr *DefMI, unsigned DefOperIdx,
                     const MachineInstr *DepMI) const {
  if (SchedModel.MicroOpBufferSize <= 1)
    return 1;

  // MicroOpBufferSize > 1 indicates an out-of-order processor that can dispatch
  // WAW dependencies in the same cycle.

  // Treat predication as a data dependency for out-of-order cpus. In-order
  // cpus do not need to treat predicated writes specially.
  //
  // TODO: The following hack exists because predication passes do not
  // correctly append imp-use operands, and readsReg() strangely returns false
  // for predicated defs.
  unsigned Reg = DefMI->getOperand(DefOperIdx).getReg();
  const MachineFunction &MF = *DefMI->getParent()->getParent();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  if (!DepMI->readsRegister(Reg, TRI) && TII->isPredicated(DepMI))
    return computeInstrLatency(DefMI);

  // If we have a per operand scheduling model, check if this def is writing
  // an unbuffered resource. If so, it treated like an in-order cpu.
  if (hasInstrSchedModel()) {
    const MCSchedClassDesc *SCDesc = resolveSchedClass(DefMI);
    if (SCDesc->isValid()) {
      for (const MCWriteProcResEntry *PRI = STI->getWriteProcResBegin(SCDesc),
             *PRE = STI->getWriteProcResEnd(SCDesc); PRI != PRE; ++PRI) {
        if (!SchedModel.getProcResource(PRI->ProcResourceIdx)->BufferSize)
          return 1;
      }
    }
  }
  return 0;
}
