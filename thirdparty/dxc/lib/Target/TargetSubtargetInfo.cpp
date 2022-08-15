//===-- TargetSubtargetInfo.cpp - General Target Information ---------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the general parts of a Subtarget.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

//---------------------------------------------------------------------------
// TargetSubtargetInfo Class
//
TargetSubtargetInfo::TargetSubtargetInfo(
    const Triple &TT, StringRef CPU, StringRef FS,
    ArrayRef<SubtargetFeatureKV> PF, ArrayRef<SubtargetFeatureKV> PD,
    const SubtargetInfoKV *ProcSched, const MCWriteProcResEntry *WPR,
    const MCWriteLatencyEntry *WL, const MCReadAdvanceEntry *RA,
    const InstrStage *IS, const unsigned *OC, const unsigned *FP)
    : MCSubtargetInfo(TT, CPU, FS, PF, PD, ProcSched, WPR, WL, RA, IS, OC, FP) {
}

TargetSubtargetInfo::~TargetSubtargetInfo() {}

bool TargetSubtargetInfo::enableAtomicExpand() const {
  return true;
}

bool TargetSubtargetInfo::enableMachineScheduler() const {
  return false;
}

bool TargetSubtargetInfo::enableJoinGlobalCopies() const {
  return enableMachineScheduler();
}

bool TargetSubtargetInfo::enableRALocalReassignment(
    CodeGenOpt::Level OptLevel) const {
  return true;
}

bool TargetSubtargetInfo::enablePostRAScheduler() const {
  return getSchedModel().PostRAScheduler;
}

bool TargetSubtargetInfo::useAA() const {
  return false;
}
