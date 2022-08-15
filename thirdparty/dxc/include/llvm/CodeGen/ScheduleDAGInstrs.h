//==- ScheduleDAGInstrs.h - MachineInstr Scheduling --------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ScheduleDAGInstrs class, which implements
// scheduling for a MachineInstr-based dependency graph.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SCHEDULEDAGINSTRS_H
#define LLVM_CODEGEN_SCHEDULEDAGINSTRS_H

#include "llvm/ADT/SparseMultiSet.h"
#include "llvm/ADT/SparseSet.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {
  class MachineFrameInfo;
  class MachineLoopInfo;
  class MachineDominatorTree;
  class LiveIntervals;
  class RegPressureTracker;
  class PressureDiffs;

  /// An individual mapping from virtual register number to SUnit.
  struct VReg2SUnit {
    unsigned VirtReg;
    SUnit *SU;

    VReg2SUnit(unsigned reg, SUnit *su): VirtReg(reg), SU(su) {}

    unsigned getSparseSetIndex() const {
      return TargetRegisterInfo::virtReg2Index(VirtReg);
    }
  };

  /// Record a physical register access.
  /// For non-data-dependent uses, OpIdx == -1.
  struct PhysRegSUOper {
    SUnit *SU;
    int OpIdx;
    unsigned Reg;

    PhysRegSUOper(SUnit *su, int op, unsigned R): SU(su), OpIdx(op), Reg(R) {}

    unsigned getSparseSetIndex() const { return Reg; }
  };

  /// Use a SparseMultiSet to track physical registers. Storage is only
  /// allocated once for the pass. It can be cleared in constant time and reused
  /// without any frees.
  typedef SparseMultiSet<PhysRegSUOper, llvm::identity<unsigned>, uint16_t>
  Reg2SUnitsMap;

  /// Use SparseSet as a SparseMap by relying on the fact that it never
  /// compares ValueT's, only unsigned keys. This allows the set to be cleared
  /// between scheduling regions in constant time as long as ValueT does not
  /// require a destructor.
  typedef SparseSet<VReg2SUnit, VirtReg2IndexFunctor> VReg2SUnitMap;

  /// Track local uses of virtual registers. These uses are gathered by the DAG
  /// builder and may be consulted by the scheduler to avoid iterating an entire
  /// vreg use list.
  typedef SparseMultiSet<VReg2SUnit, VirtReg2IndexFunctor> VReg2UseMap;

  /// ScheduleDAGInstrs - A ScheduleDAG subclass for scheduling lists of
  /// MachineInstrs.
  class ScheduleDAGInstrs : public ScheduleDAG {
  protected:
    const MachineLoopInfo *MLI;
    const MachineFrameInfo *MFI;

    /// Live Intervals provides reaching defs in preRA scheduling.
    LiveIntervals *LIS;

    /// TargetSchedModel provides an interface to the machine model.
    TargetSchedModel SchedModel;

    /// isPostRA flag indicates vregs cannot be present.
    bool IsPostRA;

    /// True if the DAG builder should remove kill flags (in preparation for
    /// rescheduling).
    bool RemoveKillFlags;

    /// The standard DAG builder does not normally include terminators as DAG
    /// nodes because it does not create the necessary dependencies to prevent
    /// reordering. A specialized scheduler can override
    /// TargetInstrInfo::isSchedulingBoundary then enable this flag to indicate
    /// it has taken responsibility for scheduling the terminator correctly.
    bool CanHandleTerminators;

    /// State specific to the current scheduling region.
    /// ------------------------------------------------

    /// The block in which to insert instructions
    MachineBasicBlock *BB;

    /// The beginning of the range to be scheduled.
    MachineBasicBlock::iterator RegionBegin;

    /// The end of the range to be scheduled.
    MachineBasicBlock::iterator RegionEnd;

    /// Instructions in this region (distance(RegionBegin, RegionEnd)).
    unsigned NumRegionInstrs;

    /// After calling BuildSchedGraph, each machine instruction in the current
    /// scheduling region is mapped to an SUnit.
    DenseMap<MachineInstr*, SUnit*> MISUnitMap;

    /// After calling BuildSchedGraph, each vreg used in the scheduling region
    /// is mapped to a set of SUnits. These include all local vreg uses, not
    /// just the uses for a singly defined vreg.
    VReg2UseMap VRegUses;

    /// State internal to DAG building.
    /// -------------------------------

    /// Defs, Uses - Remember where defs and uses of each register are as we
    /// iterate upward through the instructions. This is allocated here instead
    /// of inside BuildSchedGraph to avoid the need for it to be initialized and
    /// destructed for each block.
    Reg2SUnitsMap Defs;
    Reg2SUnitsMap Uses;

    /// Track the last instruction in this region defining each virtual register.
    VReg2SUnitMap VRegDefs;

    /// PendingLoads - Remember where unknown loads are after the most recent
    /// unknown store, as we iterate. As with Defs and Uses, this is here
    /// to minimize construction/destruction.
    std::vector<SUnit *> PendingLoads;

    /// DbgValues - Remember instruction that precedes DBG_VALUE.
    /// These are generated by buildSchedGraph but persist so they can be
    /// referenced when emitting the final schedule.
    typedef std::vector<std::pair<MachineInstr *, MachineInstr *> >
      DbgValueVector;
    DbgValueVector DbgValues;
    MachineInstr *FirstDbgValue;

    /// Set of live physical registers for updating kill flags.
    BitVector LiveRegs;

  public:
    explicit ScheduleDAGInstrs(MachineFunction &mf,
                               const MachineLoopInfo *mli,
                               bool IsPostRAFlag,
                               bool RemoveKillFlags = false,
                               LiveIntervals *LIS = nullptr);

    ~ScheduleDAGInstrs() override {}

    bool isPostRA() const { return IsPostRA; }

    /// \brief Expose LiveIntervals for use in DAG mutators and such.
    LiveIntervals *getLIS() const { return LIS; }

    /// \brief Get the machine model for instruction scheduling.
    const TargetSchedModel *getSchedModel() const { return &SchedModel; }

    /// \brief Resolve and cache a resolved scheduling class for an SUnit.
    const MCSchedClassDesc *getSchedClass(SUnit *SU) const {
      if (!SU->SchedClass && SchedModel.hasInstrSchedModel())
        SU->SchedClass = SchedModel.resolveSchedClass(SU->getInstr());
      return SU->SchedClass;
    }

    /// begin - Return an iterator to the top of the current scheduling region.
    MachineBasicBlock::iterator begin() const { return RegionBegin; }

    /// end - Return an iterator to the bottom of the current scheduling region.
    MachineBasicBlock::iterator end() const { return RegionEnd; }

    /// newSUnit - Creates a new SUnit and return a ptr to it.
    SUnit *newSUnit(MachineInstr *MI);

    /// getSUnit - Return an existing SUnit for this MI, or NULL.
    SUnit *getSUnit(MachineInstr *MI) const;

    /// startBlock - Prepare to perform scheduling in the given block.
    virtual void startBlock(MachineBasicBlock *BB);

    /// finishBlock - Clean up after scheduling in the given block.
    virtual void finishBlock();

    /// Initialize the scheduler state for the next scheduling region.
    virtual void enterRegion(MachineBasicBlock *bb,
                             MachineBasicBlock::iterator begin,
                             MachineBasicBlock::iterator end,
                             unsigned regioninstrs);

    /// Notify that the scheduler has finished scheduling the current region.
    virtual void exitRegion();

    /// buildSchedGraph - Build SUnits from the MachineBasicBlock that we are
    /// input.
    void buildSchedGraph(AliasAnalysis *AA,
                         RegPressureTracker *RPTracker = nullptr,
                         PressureDiffs *PDiffs = nullptr);

    /// addSchedBarrierDeps - Add dependencies from instructions in the current
    /// list of instructions being scheduled to scheduling barrier. We want to
    /// make sure instructions which define registers that are either used by
    /// the terminator or are live-out are properly scheduled. This is
    /// especially important when the definition latency of the return value(s)
    /// are too high to be hidden by the branch or when the liveout registers
    /// used by instructions in the fallthrough block.
    void addSchedBarrierDeps();

    /// schedule - Order nodes according to selected style, filling
    /// in the Sequence member.
    ///
    /// Typically, a scheduling algorithm will implement schedule() without
    /// overriding enterRegion() or exitRegion().
    virtual void schedule() = 0;

    /// finalizeSchedule - Allow targets to perform final scheduling actions at
    /// the level of the whole MachineFunction. By default does nothing.
    virtual void finalizeSchedule() {}

    void dumpNode(const SUnit *SU) const override;

    /// Return a label for a DAG node that points to an instruction.
    std::string getGraphNodeLabel(const SUnit *SU) const override;

    /// Return a label for the region of code covered by the DAG.
    std::string getDAGName() const override;

    /// \brief Fix register kill flags that scheduling has made invalid.
    void fixupKills(MachineBasicBlock *MBB);
  protected:
    void initSUnits();
    void addPhysRegDataDeps(SUnit *SU, unsigned OperIdx);
    void addPhysRegDeps(SUnit *SU, unsigned OperIdx);
    void addVRegDefDeps(SUnit *SU, unsigned OperIdx);
    void addVRegUseDeps(SUnit *SU, unsigned OperIdx);

    /// \brief PostRA helper for rewriting kill flags.
    void startBlockForKills(MachineBasicBlock *BB);

    /// \brief Toggle a register operand kill flag.
    ///
    /// Other adjustments may be made to the instruction if necessary. Return
    /// true if the operand has been deleted, false if not.
    bool toggleKillFlag(MachineInstr *MI, MachineOperand &MO);
  };

  /// newSUnit - Creates a new SUnit and return a ptr to it.
  inline SUnit *ScheduleDAGInstrs::newSUnit(MachineInstr *MI) {
#ifndef NDEBUG
    const SUnit *Addr = SUnits.empty() ? nullptr : &SUnits[0];
#endif
    SUnits.emplace_back(MI, (unsigned)SUnits.size());
    assert((Addr == nullptr || Addr == &SUnits[0]) &&
           "SUnits std::vector reallocated on the fly!");
    SUnits.back().OrigNode = &SUnits.back();
    return &SUnits.back();
  }

  /// getSUnit - Return an existing SUnit for this MI, or NULL.
  inline SUnit *ScheduleDAGInstrs::getSUnit(MachineInstr *MI) const {
    DenseMap<MachineInstr*, SUnit*>::const_iterator I = MISUnitMap.find(MI);
    if (I == MISUnitMap.end())
      return nullptr;
    return I->second;
  }
} // namespace llvm

#endif
