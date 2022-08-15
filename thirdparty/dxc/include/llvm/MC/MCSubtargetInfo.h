//==-- llvm/MC/MCSubtargetInfo.h - Subtarget Information ---------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the subtarget options of a Target machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSUBTARGETINFO_H
#define LLVM_MC_MCSUBTARGETINFO_H

#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/MC/SubtargetFeature.h"
#include <string>

namespace llvm {

class StringRef;
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
///
/// MCSubtargetInfo - Generic base class for all target subtargets.
///
class MCSubtargetInfo {
  Triple TargetTriple;                        // Target triple
  std::string CPU; // CPU being targeted.
  ArrayRef<SubtargetFeatureKV> ProcFeatures;  // Processor feature list
  ArrayRef<SubtargetFeatureKV> ProcDesc;  // Processor descriptions

  // Scheduler machine model
  const SubtargetInfoKV *ProcSchedModels;
  const MCWriteProcResEntry *WriteProcResTable;
  const MCWriteLatencyEntry *WriteLatencyTable;
  const MCReadAdvanceEntry *ReadAdvanceTable;
  const MCSchedModel *CPUSchedModel;

  const InstrStage *Stages;            // Instruction itinerary stages
  const unsigned *OperandCycles;       // Itinerary operand cycles
  const unsigned *ForwardingPaths;     // Forwarding paths
  FeatureBitset FeatureBits;           // Feature bits for current CPU + FS

  MCSubtargetInfo() = delete;
  MCSubtargetInfo &operator=(MCSubtargetInfo &&) = delete;
  MCSubtargetInfo &operator=(const MCSubtargetInfo &) = delete;

public:
  MCSubtargetInfo(const MCSubtargetInfo &) = default;
  MCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS,
                  ArrayRef<SubtargetFeatureKV> PF,
                  ArrayRef<SubtargetFeatureKV> PD,
                  const SubtargetInfoKV *ProcSched,
                  const MCWriteProcResEntry *WPR, const MCWriteLatencyEntry *WL,
                  const MCReadAdvanceEntry *RA, const InstrStage *IS,
                  const unsigned *OC, const unsigned *FP);

  /// getTargetTriple - Return the target triple string.
  const Triple &getTargetTriple() const { return TargetTriple; }

  /// getCPU - Return the CPU string.
  StringRef getCPU() const {
    return CPU;
  }

  /// getFeatureBits - Return the feature bits.
  ///
  const FeatureBitset& getFeatureBits() const {
    return FeatureBits;
  }

  /// setFeatureBits - Set the feature bits.
  ///
  void setFeatureBits(const FeatureBitset &FeatureBits_) {
    FeatureBits = FeatureBits_;
  }

protected:
  /// Initialize the scheduling model and feature bits.
  ///
  /// FIXME: Find a way to stick this in the constructor, since it should only
  /// be called during initialization.
  void InitMCProcessorInfo(StringRef CPU, StringRef FS);

public:
  /// Set the features to the default for the given CPU.
  void setDefaultFeatures(StringRef CPU);

  /// ToggleFeature - Toggle a feature and returns the re-computed feature
  /// bits. This version does not change the implied bits.
  FeatureBitset ToggleFeature(uint64_t FB);

  /// ToggleFeature - Toggle a feature and returns the re-computed feature
  /// bits. This version does not change the implied bits.
  FeatureBitset ToggleFeature(const FeatureBitset& FB);

  /// ToggleFeature - Toggle a set of features and returns the re-computed
  /// feature bits. This version will also change all implied bits.
  FeatureBitset ToggleFeature(StringRef FS);

  /// Apply a feature flag and return the re-computed feature bits, including
  /// all feature bits implied by the flag.
  FeatureBitset ApplyFeatureFlag(StringRef FS);

  /// getSchedModelForCPU - Get the machine model of a CPU.
  ///
  const MCSchedModel &getSchedModelForCPU(StringRef CPU) const;

  /// Get the machine model for this subtarget's CPU.
  const MCSchedModel &getSchedModel() const { return *CPUSchedModel; }

  /// Return an iterator at the first process resource consumed by the given
  /// scheduling class.
  const MCWriteProcResEntry *getWriteProcResBegin(
    const MCSchedClassDesc *SC) const {
    return &WriteProcResTable[SC->WriteProcResIdx];
  }
  const MCWriteProcResEntry *getWriteProcResEnd(
    const MCSchedClassDesc *SC) const {
    return getWriteProcResBegin(SC) + SC->NumWriteProcResEntries;
  }

  const MCWriteLatencyEntry *getWriteLatencyEntry(const MCSchedClassDesc *SC,
                                                  unsigned DefIdx) const {
    assert(DefIdx < SC->NumWriteLatencyEntries &&
           "MachineModel does not specify a WriteResource for DefIdx");

    return &WriteLatencyTable[SC->WriteLatencyIdx + DefIdx];
  }

  int getReadAdvanceCycles(const MCSchedClassDesc *SC, unsigned UseIdx,
                           unsigned WriteResID) const {
    // TODO: The number of read advance entries in a class can be significant
    // (~50). Consider compressing the WriteID into a dense ID of those that are
    // used by ReadAdvance and representing them as a bitset.
    for (const MCReadAdvanceEntry *I = &ReadAdvanceTable[SC->ReadAdvanceIdx],
           *E = I + SC->NumReadAdvanceEntries; I != E; ++I) {
      if (I->UseIdx < UseIdx)
        continue;
      if (I->UseIdx > UseIdx)
        break;
      // Find the first WriteResIdx match, which has the highest cycle count.
      if (!I->WriteResourceID || I->WriteResourceID == WriteResID) {
        return I->Cycles;
      }
    }
    return 0;
  }

  /// getInstrItineraryForCPU - Get scheduling itinerary of a CPU.
  ///
  InstrItineraryData getInstrItineraryForCPU(StringRef CPU) const;

  /// Initialize an InstrItineraryData instance.
  void initInstrItins(InstrItineraryData &InstrItins) const;

  /// Check whether the CPU string is valid.
  bool isCPUStringValid(StringRef CPU) const {
    auto Found = std::find_if(ProcDesc.begin(), ProcDesc.end(),
                              [=](const SubtargetFeatureKV &KV) {
                                return CPU == KV.Key; 
                              });
    return Found != ProcDesc.end();
  }
};

} // End llvm namespace

#endif
