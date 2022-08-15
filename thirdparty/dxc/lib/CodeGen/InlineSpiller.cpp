//===-------- InlineSpiller.cpp - Insert spills and restores inline -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The inline spiller modifies the machine function directly instead of
// inserting spills and restores in VirtRegMap.
//
//===----------------------------------------------------------------------===//

#include "Spiller.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"

using namespace llvm;

#define DEBUG_TYPE "regalloc"

STATISTIC(NumSpilledRanges,   "Number of spilled live ranges");
STATISTIC(NumSnippets,        "Number of spilled snippets");
STATISTIC(NumSpills,          "Number of spills inserted");
STATISTIC(NumSpillsRemoved,   "Number of spills removed");
STATISTIC(NumReloads,         "Number of reloads inserted");
STATISTIC(NumReloadsRemoved,  "Number of reloads removed");
STATISTIC(NumFolded,          "Number of folded stack accesses");
STATISTIC(NumFoldedLoads,     "Number of folded loads");
STATISTIC(NumRemats,          "Number of rematerialized defs for spilling");
STATISTIC(NumOmitReloadSpill, "Number of omitted spills of reloads");
STATISTIC(NumHoists,          "Number of hoisted spills");

static cl::opt<bool> DisableHoisting("disable-spill-hoist", cl::Hidden,
                                     cl::desc("Disable inline spill hoisting"));

namespace {
class InlineSpiller : public Spiller {
  MachineFunction &MF;
  LiveIntervals &LIS;
  LiveStacks &LSS;
  AliasAnalysis *AA;
  MachineDominatorTree &MDT;
  MachineLoopInfo &Loops;
  VirtRegMap &VRM;
  MachineFrameInfo &MFI;
  MachineRegisterInfo &MRI;
  const TargetInstrInfo &TII;
  const TargetRegisterInfo &TRI;
  const MachineBlockFrequencyInfo &MBFI;

  // Variables that are valid during spill(), but used by multiple methods.
  LiveRangeEdit *Edit;
  LiveInterval *StackInt;
  int StackSlot;
  unsigned Original;

  // All registers to spill to StackSlot, including the main register.
  SmallVector<unsigned, 8> RegsToSpill;

  // All COPY instructions to/from snippets.
  // They are ignored since both operands refer to the same stack slot.
  SmallPtrSet<MachineInstr*, 8> SnippetCopies;

  // Values that failed to remat at some point.
  SmallPtrSet<VNInfo*, 8> UsedValues;

public:
  // Information about a value that was defined by a copy from a sibling
  // register.
  struct SibValueInfo {
    // True when all reaching defs were reloads: No spill is necessary.
    bool AllDefsAreReloads;

    // True when value is defined by an original PHI not from splitting.
    bool DefByOrigPHI;

    // True when the COPY defining this value killed its source.
    bool KillsSource;

    // The preferred register to spill.
    unsigned SpillReg;

    // The value of SpillReg that should be spilled.
    VNInfo *SpillVNI;

    // The block where SpillVNI should be spilled. Currently, this must be the
    // block containing SpillVNI->def.
    MachineBasicBlock *SpillMBB;

    // A defining instruction that is not a sibling copy or a reload, or NULL.
    // This can be used as a template for rematerialization.
    MachineInstr *DefMI;

    // List of values that depend on this one.  These values are actually the
    // same, but live range splitting has placed them in different registers,
    // or SSA update needed to insert PHI-defs to preserve SSA form.  This is
    // copies of the current value and phi-kills.  Usually only phi-kills cause
    // more than one dependent value.
    TinyPtrVector<VNInfo*> Deps;

    SibValueInfo(unsigned Reg, VNInfo *VNI)
      : AllDefsAreReloads(true), DefByOrigPHI(false), KillsSource(false),
        SpillReg(Reg), SpillVNI(VNI), SpillMBB(nullptr), DefMI(nullptr) {}

    // Returns true when a def has been found.
    bool hasDef() const { return DefByOrigPHI || DefMI; }
  };

private:
  // Values in RegsToSpill defined by sibling copies.
  typedef DenseMap<VNInfo*, SibValueInfo> SibValueMap;
  SibValueMap SibValues;

  // Dead defs generated during spilling.
  SmallVector<MachineInstr*, 8> DeadDefs;

  ~InlineSpiller() override {}

public:
  InlineSpiller(MachineFunctionPass &pass, MachineFunction &mf, VirtRegMap &vrm)
      : MF(mf), LIS(pass.getAnalysis<LiveIntervals>()),
        LSS(pass.getAnalysis<LiveStacks>()),
        AA(&pass.getAnalysis<AliasAnalysis>()),
        MDT(pass.getAnalysis<MachineDominatorTree>()),
        Loops(pass.getAnalysis<MachineLoopInfo>()), VRM(vrm),
        MFI(*mf.getFrameInfo()), MRI(mf.getRegInfo()),
        TII(*mf.getSubtarget().getInstrInfo()),
        TRI(*mf.getSubtarget().getRegisterInfo()),
        MBFI(pass.getAnalysis<MachineBlockFrequencyInfo>()) {}

  void spill(LiveRangeEdit &) override;

private:
  bool isSnippet(const LiveInterval &SnipLI);
  void collectRegsToSpill();

  bool isRegToSpill(unsigned Reg) {
    return std::find(RegsToSpill.begin(),
                     RegsToSpill.end(), Reg) != RegsToSpill.end();
  }

  bool isSibling(unsigned Reg);
  MachineInstr *traceSiblingValue(unsigned, VNInfo*, VNInfo*);
  void propagateSiblingValue(SibValueMap::iterator, VNInfo *VNI = nullptr);
  void analyzeSiblingValues();

  bool hoistSpill(LiveInterval &SpillLI, MachineInstr *CopyMI);
  void eliminateRedundantSpills(LiveInterval &LI, VNInfo *VNI);

  void markValueUsed(LiveInterval*, VNInfo*);
  bool reMaterializeFor(LiveInterval&, MachineBasicBlock::iterator MI);
  void reMaterializeAll();

  bool coalesceStackAccess(MachineInstr *MI, unsigned Reg);
  bool foldMemoryOperand(ArrayRef<std::pair<MachineInstr*, unsigned> >,
                         MachineInstr *LoadMI = nullptr);
  void insertReload(unsigned VReg, SlotIndex, MachineBasicBlock::iterator MI);
  void insertSpill(unsigned VReg, bool isKill, MachineBasicBlock::iterator MI);

  void spillAroundUses(unsigned Reg);
  void spillAll();
};
}

namespace llvm {

Spiller::~Spiller() { }
void Spiller::anchor() { }

Spiller *createInlineSpiller(MachineFunctionPass &pass,
                             MachineFunction &mf,
                             VirtRegMap &vrm) {
  return new InlineSpiller(pass, mf, vrm);
}

}

//===----------------------------------------------------------------------===//
//                                Snippets
//===----------------------------------------------------------------------===//

// When spilling a virtual register, we also spill any snippets it is connected
// to. The snippets are small live ranges that only have a single real use,
// leftovers from live range splitting. Spilling them enables memory operand
// folding or tightens the live range around the single use.
//
// This minimizes register pressure and maximizes the store-to-load distance for
// spill slots which can be important in tight loops.

/// isFullCopyOf - If MI is a COPY to or from Reg, return the other register,
/// otherwise return 0.
static unsigned isFullCopyOf(const MachineInstr *MI, unsigned Reg) {
  if (!MI->isFullCopy())
    return 0;
  if (MI->getOperand(0).getReg() == Reg)
      return MI->getOperand(1).getReg();
  if (MI->getOperand(1).getReg() == Reg)
      return MI->getOperand(0).getReg();
  return 0;
}

/// isSnippet - Identify if a live interval is a snippet that should be spilled.
/// It is assumed that SnipLI is a virtual register with the same original as
/// Edit->getReg().
bool InlineSpiller::isSnippet(const LiveInterval &SnipLI) {
  unsigned Reg = Edit->getReg();

  // A snippet is a tiny live range with only a single instruction using it
  // besides copies to/from Reg or spills/fills. We accept:
  //
  //   %snip = COPY %Reg / FILL fi#
  //   %snip = USE %snip
  //   %Reg = COPY %snip / SPILL %snip, fi#
  //
  if (SnipLI.getNumValNums() > 2 || !LIS.intervalIsInOneMBB(SnipLI))
    return false;

  MachineInstr *UseMI = nullptr;

  // Check that all uses satisfy our criteria.
  for (MachineRegisterInfo::reg_instr_nodbg_iterator
       RI = MRI.reg_instr_nodbg_begin(SnipLI.reg),
       E = MRI.reg_instr_nodbg_end(); RI != E; ) {
    MachineInstr *MI = &*(RI++);

    // Allow copies to/from Reg.
    if (isFullCopyOf(MI, Reg))
      continue;

    // Allow stack slot loads.
    int FI;
    if (SnipLI.reg == TII.isLoadFromStackSlot(MI, FI) && FI == StackSlot)
      continue;

    // Allow stack slot stores.
    if (SnipLI.reg == TII.isStoreToStackSlot(MI, FI) && FI == StackSlot)
      continue;

    // Allow a single additional instruction.
    if (UseMI && MI != UseMI)
      return false;
    UseMI = MI;
  }
  return true;
}

/// collectRegsToSpill - Collect live range snippets that only have a single
/// real use.
void InlineSpiller::collectRegsToSpill() {
  unsigned Reg = Edit->getReg();

  // Main register always spills.
  RegsToSpill.assign(1, Reg);
  SnippetCopies.clear();

  // Snippets all have the same original, so there can't be any for an original
  // register.
  if (Original == Reg)
    return;

  for (MachineRegisterInfo::reg_instr_iterator
       RI = MRI.reg_instr_begin(Reg), E = MRI.reg_instr_end(); RI != E; ) {
    MachineInstr *MI = &*(RI++);
    unsigned SnipReg = isFullCopyOf(MI, Reg);
    if (!isSibling(SnipReg))
      continue;
    LiveInterval &SnipLI = LIS.getInterval(SnipReg);
    if (!isSnippet(SnipLI))
      continue;
    SnippetCopies.insert(MI);
    if (isRegToSpill(SnipReg))
      continue;
    RegsToSpill.push_back(SnipReg);
    DEBUG(dbgs() << "\talso spill snippet " << SnipLI << '\n');
    ++NumSnippets;
  }
}


//===----------------------------------------------------------------------===//
//                            Sibling Values
//===----------------------------------------------------------------------===//

// After live range splitting, some values to be spilled may be defined by
// copies from sibling registers. We trace the sibling copies back to the
// original value if it still exists. We need it for rematerialization.
//
// Even when the value can't be rematerialized, we still want to determine if
// the value has already been spilled, or we may want to hoist the spill from a
// loop.

bool InlineSpiller::isSibling(unsigned Reg) {
  return TargetRegisterInfo::isVirtualRegister(Reg) &&
           VRM.getOriginal(Reg) == Original;
}

#ifndef NDEBUG
static raw_ostream &operator<<(raw_ostream &OS,
                               const InlineSpiller::SibValueInfo &SVI) {
  OS << "spill " << PrintReg(SVI.SpillReg) << ':'
     << SVI.SpillVNI->id << '@' << SVI.SpillVNI->def;
  if (SVI.SpillMBB)
    OS << " in BB#" << SVI.SpillMBB->getNumber();
  if (SVI.AllDefsAreReloads)
    OS << " all-reloads";
  if (SVI.DefByOrigPHI)
    OS << " orig-phi";
  if (SVI.KillsSource)
    OS << " kill";
  OS << " deps[";
  for (unsigned i = 0, e = SVI.Deps.size(); i != e; ++i)
    OS << ' ' << SVI.Deps[i]->id << '@' << SVI.Deps[i]->def;
  OS << " ]";
  if (SVI.DefMI)
    OS << " def: " << *SVI.DefMI;
  else
    OS << '\n';
  return OS;
}
#endif

/// propagateSiblingValue - Propagate the value in SVI to dependents if it is
/// known.  Otherwise remember the dependency for later.
///
/// @param SVIIter SibValues entry to propagate.
/// @param VNI Dependent value, or NULL to propagate to all saved dependents.
void InlineSpiller::propagateSiblingValue(SibValueMap::iterator SVIIter,
                                          VNInfo *VNI) {
  SibValueMap::value_type *SVI = &*SVIIter;

  // When VNI is non-NULL, add it to SVI's deps, and only propagate to that.
  TinyPtrVector<VNInfo*> FirstDeps;
  if (VNI) {
    FirstDeps.push_back(VNI);
    SVI->second.Deps.push_back(VNI);
  }

  // Has the value been completely determined yet?  If not, defer propagation.
  if (!SVI->second.hasDef())
    return;

  // Work list of values to propagate.
  SmallSetVector<SibValueMap::value_type *, 8> WorkList;
  WorkList.insert(SVI);

  do {
    SVI = WorkList.pop_back_val();
    TinyPtrVector<VNInfo*> *Deps = VNI ? &FirstDeps : &SVI->second.Deps;
    VNI = nullptr;

    SibValueInfo &SV = SVI->second;
    if (!SV.SpillMBB)
      SV.SpillMBB = LIS.getMBBFromIndex(SV.SpillVNI->def);

    DEBUG(dbgs() << "  prop to " << Deps->size() << ": "
                 << SVI->first->id << '@' << SVI->first->def << ":\t" << SV);

    assert(SV.hasDef() && "Propagating undefined value");

    // Should this value be propagated as a preferred spill candidate?  We don't
    // propagate values of registers that are about to spill.
    bool PropSpill = !DisableHoisting && !isRegToSpill(SV.SpillReg);
    unsigned SpillDepth = ~0u;

    for (TinyPtrVector<VNInfo*>::iterator DepI = Deps->begin(),
         DepE = Deps->end(); DepI != DepE; ++DepI) {
      SibValueMap::iterator DepSVI = SibValues.find(*DepI);
      assert(DepSVI != SibValues.end() && "Dependent value not in SibValues");
      SibValueInfo &DepSV = DepSVI->second;
      if (!DepSV.SpillMBB)
        DepSV.SpillMBB = LIS.getMBBFromIndex(DepSV.SpillVNI->def);

      bool Changed = false;

      // Propagate defining instruction.
      if (!DepSV.hasDef()) {
        Changed = true;
        DepSV.DefMI = SV.DefMI;
        DepSV.DefByOrigPHI = SV.DefByOrigPHI;
      }

      // Propagate AllDefsAreReloads.  For PHI values, this computes an AND of
      // all predecessors.
      if (!SV.AllDefsAreReloads && DepSV.AllDefsAreReloads) {
        Changed = true;
        DepSV.AllDefsAreReloads = false;
      }

      // Propagate best spill value.
      if (PropSpill && SV.SpillVNI != DepSV.SpillVNI) {
        if (SV.SpillMBB == DepSV.SpillMBB) {
          // DepSV is in the same block.  Hoist when dominated.
          if (DepSV.KillsSource && SV.SpillVNI->def < DepSV.SpillVNI->def) {
            // This is an alternative def earlier in the same MBB.
            // Hoist the spill as far as possible in SpillMBB. This can ease
            // register pressure:
            //
            //   x = def
            //   y = use x
            //   s = copy x
            //
            // Hoisting the spill of s to immediately after the def removes the
            // interference between x and y:
            //
            //   x = def
            //   spill x
            //   y = use x<kill>
            //
            // This hoist only helps when the DepSV copy kills its source.
            Changed = true;
            DepSV.SpillReg = SV.SpillReg;
            DepSV.SpillVNI = SV.SpillVNI;
            DepSV.SpillMBB = SV.SpillMBB;
          }
        } else {
          // DepSV is in a different block.
          if (SpillDepth == ~0u)
            SpillDepth = Loops.getLoopDepth(SV.SpillMBB);

          // Also hoist spills to blocks with smaller loop depth, but make sure
          // that the new value dominates.  Non-phi dependents are always
          // dominated, phis need checking.

          const BranchProbability MarginProb(4, 5); // 80%
          // Hoist a spill to outer loop if there are multiple dependents (it
          // can be beneficial if more than one dependents are hoisted) or
          // if DepSV (the hoisting source) is hotter than SV (the hoisting
          // destination) (we add a 80% margin to bias a little towards
          // loop depth).
          bool HoistCondition =
            (MBFI.getBlockFreq(DepSV.SpillMBB) >=
             (MBFI.getBlockFreq(SV.SpillMBB) * MarginProb)) ||
            Deps->size() > 1;

          if ((Loops.getLoopDepth(DepSV.SpillMBB) > SpillDepth) &&
              HoistCondition &&
              (!DepSVI->first->isPHIDef() ||
               MDT.dominates(SV.SpillMBB, DepSV.SpillMBB))) {
            Changed = true;
            DepSV.SpillReg = SV.SpillReg;
            DepSV.SpillVNI = SV.SpillVNI;
            DepSV.SpillMBB = SV.SpillMBB;
          }
        }
      }

      if (!Changed)
        continue;

      // Something changed in DepSVI. Propagate to dependents.
      WorkList.insert(&*DepSVI);

      DEBUG(dbgs() << "  update " << DepSVI->first->id << '@'
            << DepSVI->first->def << " to:\t" << DepSV);
    }
  } while (!WorkList.empty());
}

/// traceSiblingValue - Trace a value that is about to be spilled back to the
/// real defining instructions by looking through sibling copies. Always stay
/// within the range of OrigVNI so the registers are known to carry the same
/// value.
///
/// Determine if the value is defined by all reloads, so spilling isn't
/// necessary - the value is already in the stack slot.
///
/// Return a defining instruction that may be a candidate for rematerialization.
///
MachineInstr *InlineSpiller::traceSiblingValue(unsigned UseReg, VNInfo *UseVNI,
                                               VNInfo *OrigVNI) {
  // Check if a cached value already exists.
  SibValueMap::iterator SVI;
  bool Inserted;
  std::tie(SVI, Inserted) =
    SibValues.insert(std::make_pair(UseVNI, SibValueInfo(UseReg, UseVNI)));
  if (!Inserted) {
    DEBUG(dbgs() << "Cached value " << PrintReg(UseReg) << ':'
                 << UseVNI->id << '@' << UseVNI->def << ' ' << SVI->second);
    return SVI->second.DefMI;
  }

  DEBUG(dbgs() << "Tracing value " << PrintReg(UseReg) << ':'
               << UseVNI->id << '@' << UseVNI->def << '\n');

  // List of (Reg, VNI) that have been inserted into SibValues, but need to be
  // processed.
  SmallVector<std::pair<unsigned, VNInfo*>, 8> WorkList;
  WorkList.push_back(std::make_pair(UseReg, UseVNI));

  LiveInterval &OrigLI = LIS.getInterval(Original);
  do {
    unsigned Reg;
    VNInfo *VNI;
    std::tie(Reg, VNI) = WorkList.pop_back_val();
    DEBUG(dbgs() << "  " << PrintReg(Reg) << ':' << VNI->id << '@' << VNI->def
                 << ":\t");

    // First check if this value has already been computed.
    SVI = SibValues.find(VNI);
    assert(SVI != SibValues.end() && "Missing SibValues entry");

    // Trace through PHI-defs created by live range splitting.
    if (VNI->isPHIDef()) {
      // Stop at original PHIs.  We don't know the value at the
      // predecessors. Look up the VNInfo for the current definition
      // in OrigLI, to properly determine whether or not this phi was
      // added by splitting.
      if (VNI->def == OrigLI.getVNInfoAt(VNI->def)->def) {
        DEBUG(dbgs() << "orig phi value\n");
        SVI->second.DefByOrigPHI = true;
        SVI->second.AllDefsAreReloads = false;
        propagateSiblingValue(SVI);
        continue;
      }

      // This is a PHI inserted by live range splitting.  We could trace the
      // live-out value from predecessor blocks, but that search can be very
      // expensive if there are many predecessors and many more PHIs as
      // generated by tail-dup when it sees an indirectbr.  Instead, look at
      // all the non-PHI defs that have the same value as OrigVNI.  They must
      // jointly dominate VNI->def.  This is not optimal since VNI may actually
      // be jointly dominated by a smaller subset of defs, so there is a change
      // we will miss a AllDefsAreReloads optimization.

      // Separate all values dominated by OrigVNI into PHIs and non-PHIs.
      SmallVector<VNInfo*, 8> PHIs, NonPHIs;
      LiveInterval &LI = LIS.getInterval(Reg);

      for (LiveInterval::vni_iterator VI = LI.vni_begin(), VE = LI.vni_end();
           VI != VE; ++VI) {
        VNInfo *VNI2 = *VI;
        if (VNI2->isUnused())
          continue;
        if (!OrigLI.containsOneValue() &&
            OrigLI.getVNInfoAt(VNI2->def) != OrigVNI)
          continue;
        if (VNI2->isPHIDef() && VNI2->def != OrigVNI->def)
          PHIs.push_back(VNI2);
        else
          NonPHIs.push_back(VNI2);
      }
      DEBUG(dbgs() << "split phi value, checking " << PHIs.size()
                   << " phi-defs, and " << NonPHIs.size()
                   << " non-phi/orig defs\n");

      // Create entries for all the PHIs.  Don't add them to the worklist, we
      // are processing all of them in one go here.
      for (unsigned i = 0, e = PHIs.size(); i != e; ++i)
        SibValues.insert(std::make_pair(PHIs[i], SibValueInfo(Reg, PHIs[i])));

      // Add every PHI as a dependent of all the non-PHIs.
      for (unsigned i = 0, e = NonPHIs.size(); i != e; ++i) {
        VNInfo *NonPHI = NonPHIs[i];
        // Known value? Try an insertion.
        std::tie(SVI, Inserted) =
          SibValues.insert(std::make_pair(NonPHI, SibValueInfo(Reg, NonPHI)));
        // Add all the PHIs as dependents of NonPHI.
        SVI->second.Deps.insert(SVI->second.Deps.end(), PHIs.begin(),
                                PHIs.end());
        // This is the first time we see NonPHI, add it to the worklist.
        if (Inserted)
          WorkList.push_back(std::make_pair(Reg, NonPHI));
        else
          // Propagate to all inserted PHIs, not just VNI.
          propagateSiblingValue(SVI);
      }

      // Next work list item.
      continue;
    }

    MachineInstr *MI = LIS.getInstructionFromIndex(VNI->def);
    assert(MI && "Missing def");

    // Trace through sibling copies.
    if (unsigned SrcReg = isFullCopyOf(MI, Reg)) {
      if (isSibling(SrcReg)) {
        LiveInterval &SrcLI = LIS.getInterval(SrcReg);
        LiveQueryResult SrcQ = SrcLI.Query(VNI->def);
        assert(SrcQ.valueIn() && "Copy from non-existing value");
        // Check if this COPY kills its source.
        SVI->second.KillsSource = SrcQ.isKill();
        VNInfo *SrcVNI = SrcQ.valueIn();
        DEBUG(dbgs() << "copy of " << PrintReg(SrcReg) << ':'
                     << SrcVNI->id << '@' << SrcVNI->def
                     << " kill=" << unsigned(SVI->second.KillsSource) << '\n');
        // Known sibling source value? Try an insertion.
        std::tie(SVI, Inserted) = SibValues.insert(
            std::make_pair(SrcVNI, SibValueInfo(SrcReg, SrcVNI)));
        // This is the first time we see Src, add it to the worklist.
        if (Inserted)
          WorkList.push_back(std::make_pair(SrcReg, SrcVNI));
        propagateSiblingValue(SVI, VNI);
        // Next work list item.
        continue;
      }
    }

    // Track reachable reloads.
    SVI->second.DefMI = MI;
    SVI->second.SpillMBB = MI->getParent();
    int FI;
    if (Reg == TII.isLoadFromStackSlot(MI, FI) && FI == StackSlot) {
      DEBUG(dbgs() << "reload\n");
      propagateSiblingValue(SVI);
      // Next work list item.
      continue;
    }

    // Potential remat candidate.
    DEBUG(dbgs() << "def " << *MI);
    SVI->second.AllDefsAreReloads = false;
    propagateSiblingValue(SVI);
  } while (!WorkList.empty());

  // Look up the value we were looking for.  We already did this lookup at the
  // top of the function, but SibValues may have been invalidated.
  SVI = SibValues.find(UseVNI);
  assert(SVI != SibValues.end() && "Didn't compute requested info");
  DEBUG(dbgs() << "  traced to:\t" << SVI->second);
  return SVI->second.DefMI;
}

/// analyzeSiblingValues - Trace values defined by sibling copies back to
/// something that isn't a sibling copy.
///
/// Keep track of values that may be rematerializable.
void InlineSpiller::analyzeSiblingValues() {
  SibValues.clear();

  // No siblings at all?
  if (Edit->getReg() == Original)
    return;

  LiveInterval &OrigLI = LIS.getInterval(Original);
  for (unsigned i = 0, e = RegsToSpill.size(); i != e; ++i) {
    unsigned Reg = RegsToSpill[i];
    LiveInterval &LI = LIS.getInterval(Reg);
    for (LiveInterval::const_vni_iterator VI = LI.vni_begin(),
         VE = LI.vni_end(); VI != VE; ++VI) {
      VNInfo *VNI = *VI;
      if (VNI->isUnused())
        continue;
      MachineInstr *DefMI = nullptr;
      if (!VNI->isPHIDef()) {
       DefMI = LIS.getInstructionFromIndex(VNI->def);
       assert(DefMI && "No defining instruction");
      }
      // Check possible sibling copies.
      if (VNI->isPHIDef() || DefMI->isCopy()) {
        VNInfo *OrigVNI = OrigLI.getVNInfoAt(VNI->def);
        assert(OrigVNI && "Def outside original live range");
        if (OrigVNI->def != VNI->def)
          DefMI = traceSiblingValue(Reg, VNI, OrigVNI);
      }
      if (DefMI && Edit->checkRematerializable(VNI, DefMI, AA)) {
        DEBUG(dbgs() << "Value " << PrintReg(Reg) << ':' << VNI->id << '@'
                     << VNI->def << " may remat from " << *DefMI);
      }
    }
  }
}

/// hoistSpill - Given a sibling copy that defines a value to be spilled, insert
/// a spill at a better location.
bool InlineSpiller::hoistSpill(LiveInterval &SpillLI, MachineInstr *CopyMI) {
  SlotIndex Idx = LIS.getInstructionIndex(CopyMI);
  VNInfo *VNI = SpillLI.getVNInfoAt(Idx.getRegSlot());
  assert(VNI && VNI->def == Idx.getRegSlot() && "Not defined by copy");
  SibValueMap::iterator I = SibValues.find(VNI);
  if (I == SibValues.end())
    return false;

  const SibValueInfo &SVI = I->second;

  // Let the normal folding code deal with the boring case.
  if (!SVI.AllDefsAreReloads && SVI.SpillVNI == VNI)
    return false;

  // SpillReg may have been deleted by remat and DCE.
  if (!LIS.hasInterval(SVI.SpillReg)) {
    DEBUG(dbgs() << "Stale interval: " << PrintReg(SVI.SpillReg) << '\n');
    SibValues.erase(I);
    return false;
  }

  LiveInterval &SibLI = LIS.getInterval(SVI.SpillReg);
  if (!SibLI.containsValue(SVI.SpillVNI)) {
    DEBUG(dbgs() << "Stale value: " << PrintReg(SVI.SpillReg) << '\n');
    SibValues.erase(I);
    return false;
  }

  // Conservatively extend the stack slot range to the range of the original
  // value. We may be able to do better with stack slot coloring by being more
  // careful here.
  assert(StackInt && "No stack slot assigned yet.");
  LiveInterval &OrigLI = LIS.getInterval(Original);
  VNInfo *OrigVNI = OrigLI.getVNInfoAt(Idx);
  StackInt->MergeValueInAsValue(OrigLI, OrigVNI, StackInt->getValNumInfo(0));
  DEBUG(dbgs() << "\tmerged orig valno " << OrigVNI->id << ": "
               << *StackInt << '\n');

  // Already spilled everywhere.
  if (SVI.AllDefsAreReloads) {
    DEBUG(dbgs() << "\tno spill needed: " << SVI);
    ++NumOmitReloadSpill;
    return true;
  }
  // We are going to spill SVI.SpillVNI immediately after its def, so clear out
  // any later spills of the same value.
  eliminateRedundantSpills(SibLI, SVI.SpillVNI);

  MachineBasicBlock *MBB = LIS.getMBBFromIndex(SVI.SpillVNI->def);
  MachineBasicBlock::iterator MII;
  if (SVI.SpillVNI->isPHIDef())
    MII = MBB->SkipPHIsAndLabels(MBB->begin());
  else {
    MachineInstr *DefMI = LIS.getInstructionFromIndex(SVI.SpillVNI->def);
    assert(DefMI && "Defining instruction disappeared");
    MII = DefMI;
    ++MII;
  }
  // Insert spill without kill flag immediately after def.
  TII.storeRegToStackSlot(*MBB, MII, SVI.SpillReg, false, StackSlot,
                          MRI.getRegClass(SVI.SpillReg), &TRI);
  --MII; // Point to store instruction.
  LIS.InsertMachineInstrInMaps(MII);
  DEBUG(dbgs() << "\thoisted: " << SVI.SpillVNI->def << '\t' << *MII);

  ++NumSpills;
  ++NumHoists;
  return true;
}

/// eliminateRedundantSpills - SLI:VNI is known to be on the stack. Remove any
/// redundant spills of this value in SLI.reg and sibling copies.
void InlineSpiller::eliminateRedundantSpills(LiveInterval &SLI, VNInfo *VNI) {
  assert(VNI && "Missing value");
  SmallVector<std::pair<LiveInterval*, VNInfo*>, 8> WorkList;
  WorkList.push_back(std::make_pair(&SLI, VNI));
  assert(StackInt && "No stack slot assigned yet.");

  do {
    LiveInterval *LI;
    std::tie(LI, VNI) = WorkList.pop_back_val();
    unsigned Reg = LI->reg;
    DEBUG(dbgs() << "Checking redundant spills for "
                 << VNI->id << '@' << VNI->def << " in " << *LI << '\n');

    // Regs to spill are taken care of.
    if (isRegToSpill(Reg))
      continue;

    // Add all of VNI's live range to StackInt.
    StackInt->MergeValueInAsValue(*LI, VNI, StackInt->getValNumInfo(0));
    DEBUG(dbgs() << "Merged to stack int: " << *StackInt << '\n');

    // Find all spills and copies of VNI.
    for (MachineRegisterInfo::use_instr_nodbg_iterator
         UI = MRI.use_instr_nodbg_begin(Reg), E = MRI.use_instr_nodbg_end();
         UI != E; ) {
      MachineInstr *MI = &*(UI++);
      if (!MI->isCopy() && !MI->mayStore())
        continue;
      SlotIndex Idx = LIS.getInstructionIndex(MI);
      if (LI->getVNInfoAt(Idx) != VNI)
        continue;

      // Follow sibling copies down the dominator tree.
      if (unsigned DstReg = isFullCopyOf(MI, Reg)) {
        if (isSibling(DstReg)) {
           LiveInterval &DstLI = LIS.getInterval(DstReg);
           VNInfo *DstVNI = DstLI.getVNInfoAt(Idx.getRegSlot());
           assert(DstVNI && "Missing defined value");
           assert(DstVNI->def == Idx.getRegSlot() && "Wrong copy def slot");
           WorkList.push_back(std::make_pair(&DstLI, DstVNI));
        }
        continue;
      }

      // Erase spills.
      int FI;
      if (Reg == TII.isStoreToStackSlot(MI, FI) && FI == StackSlot) {
        DEBUG(dbgs() << "Redundant spill " << Idx << '\t' << *MI);
        // eliminateDeadDefs won't normally remove stores, so switch opcode.
        MI->setDesc(TII.get(TargetOpcode::KILL));
        DeadDefs.push_back(MI);
        ++NumSpillsRemoved;
        --NumSpills;
      }
    }
  } while (!WorkList.empty());
}


//===----------------------------------------------------------------------===//
//                            Rematerialization
//===----------------------------------------------------------------------===//

/// markValueUsed - Remember that VNI failed to rematerialize, so its defining
/// instruction cannot be eliminated. See through snippet copies
void InlineSpiller::markValueUsed(LiveInterval *LI, VNInfo *VNI) {
  SmallVector<std::pair<LiveInterval*, VNInfo*>, 8> WorkList;
  WorkList.push_back(std::make_pair(LI, VNI));
  do {
    std::tie(LI, VNI) = WorkList.pop_back_val();
    if (!UsedValues.insert(VNI).second)
      continue;

    if (VNI->isPHIDef()) {
      MachineBasicBlock *MBB = LIS.getMBBFromIndex(VNI->def);
      for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
             PE = MBB->pred_end(); PI != PE; ++PI) {
        VNInfo *PVNI = LI->getVNInfoBefore(LIS.getMBBEndIdx(*PI));
        if (PVNI)
          WorkList.push_back(std::make_pair(LI, PVNI));
      }
      continue;
    }

    // Follow snippet copies.
    MachineInstr *MI = LIS.getInstructionFromIndex(VNI->def);
    if (!SnippetCopies.count(MI))
      continue;
    LiveInterval &SnipLI = LIS.getInterval(MI->getOperand(1).getReg());
    assert(isRegToSpill(SnipLI.reg) && "Unexpected register in copy");
    VNInfo *SnipVNI = SnipLI.getVNInfoAt(VNI->def.getRegSlot(true));
    assert(SnipVNI && "Snippet undefined before copy");
    WorkList.push_back(std::make_pair(&SnipLI, SnipVNI));
  } while (!WorkList.empty());
}

/// reMaterializeFor - Attempt to rematerialize before MI instead of reloading.
bool InlineSpiller::reMaterializeFor(LiveInterval &VirtReg,
                                     MachineBasicBlock::iterator MI) {

  // Analyze instruction
  SmallVector<std::pair<MachineInstr *, unsigned>, 8> Ops;
  MIBundleOperands::VirtRegInfo RI =
    MIBundleOperands(MI).analyzeVirtReg(VirtReg.reg, &Ops);

  if (!RI.Reads)
    return false;

  SlotIndex UseIdx = LIS.getInstructionIndex(MI).getRegSlot(true);
  VNInfo *ParentVNI = VirtReg.getVNInfoAt(UseIdx.getBaseIndex());

  if (!ParentVNI) {
    DEBUG(dbgs() << "\tadding <undef> flags: ");
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg() && MO.isUse() && MO.getReg() == VirtReg.reg)
        MO.setIsUndef();
    }
    DEBUG(dbgs() << UseIdx << '\t' << *MI);
    return true;
  }

  if (SnippetCopies.count(MI))
    return false;

  // Use an OrigVNI from traceSiblingValue when ParentVNI is a sibling copy.
  LiveRangeEdit::Remat RM(ParentVNI);
  SibValueMap::const_iterator SibI = SibValues.find(ParentVNI);
  if (SibI != SibValues.end())
    RM.OrigMI = SibI->second.DefMI;
  if (!Edit->canRematerializeAt(RM, UseIdx, false)) {
    markValueUsed(&VirtReg, ParentVNI);
    DEBUG(dbgs() << "\tcannot remat for " << UseIdx << '\t' << *MI);
    return false;
  }

  // If the instruction also writes VirtReg.reg, it had better not require the
  // same register for uses and defs.
  if (RI.Tied) {
    markValueUsed(&VirtReg, ParentVNI);
    DEBUG(dbgs() << "\tcannot remat tied reg: " << UseIdx << '\t' << *MI);
    return false;
  }

  // Before rematerializing into a register for a single instruction, try to
  // fold a load into the instruction. That avoids allocating a new register.
  if (RM.OrigMI->canFoldAsLoad() &&
      foldMemoryOperand(Ops, RM.OrigMI)) {
    Edit->markRematerialized(RM.ParentVNI);
    ++NumFoldedLoads;
    return true;
  }

  // Alocate a new register for the remat.
  unsigned NewVReg = Edit->createFrom(Original);

  // Finally we can rematerialize OrigMI before MI.
  SlotIndex DefIdx = Edit->rematerializeAt(*MI->getParent(), MI, NewVReg, RM,
                                           TRI);
  (void)DefIdx;
  DEBUG(dbgs() << "\tremat:  " << DefIdx << '\t'
               << *LIS.getInstructionFromIndex(DefIdx));

  // Replace operands
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    MachineOperand &MO = Ops[i].first->getOperand(Ops[i].second);
    if (MO.isReg() && MO.isUse() && MO.getReg() == VirtReg.reg) {
      MO.setReg(NewVReg);
      MO.setIsKill();
    }
  }
  DEBUG(dbgs() << "\t        " << UseIdx << '\t' << *MI << '\n');

  ++NumRemats;
  return true;
}

/// reMaterializeAll - Try to rematerialize as many uses as possible,
/// and trim the live ranges after.
void InlineSpiller::reMaterializeAll() {
  // analyzeSiblingValues has already tested all relevant defining instructions.
  if (!Edit->anyRematerializable(AA))
    return;

  UsedValues.clear();

  // Try to remat before all uses of snippets.
  bool anyRemat = false;
  for (unsigned i = 0, e = RegsToSpill.size(); i != e; ++i) {
    unsigned Reg = RegsToSpill[i];
    LiveInterval &LI = LIS.getInterval(Reg);
    for (MachineRegisterInfo::reg_bundle_iterator
           RegI = MRI.reg_bundle_begin(Reg), E = MRI.reg_bundle_end();
         RegI != E; ) {
      MachineInstr *MI = &*(RegI++);

      // Debug values are not allowed to affect codegen.
      if (MI->isDebugValue())
        continue;

      anyRemat |= reMaterializeFor(LI, MI);
    }
  }
  if (!anyRemat)
    return;

  // Remove any values that were completely rematted.
  for (unsigned i = 0, e = RegsToSpill.size(); i != e; ++i) {
    unsigned Reg = RegsToSpill[i];
    LiveInterval &LI = LIS.getInterval(Reg);
    for (LiveInterval::vni_iterator I = LI.vni_begin(), E = LI.vni_end();
         I != E; ++I) {
      VNInfo *VNI = *I;
      if (VNI->isUnused() || VNI->isPHIDef() || UsedValues.count(VNI))
        continue;
      MachineInstr *MI = LIS.getInstructionFromIndex(VNI->def);
      MI->addRegisterDead(Reg, &TRI);
      if (!MI->allDefsAreDead())
        continue;
      DEBUG(dbgs() << "All defs dead: " << *MI);
      DeadDefs.push_back(MI);
    }
  }

  // Eliminate dead code after remat. Note that some snippet copies may be
  // deleted here.
  if (DeadDefs.empty())
    return;
  DEBUG(dbgs() << "Remat created " << DeadDefs.size() << " dead defs.\n");
  Edit->eliminateDeadDefs(DeadDefs, RegsToSpill);

  // Get rid of deleted and empty intervals.
  unsigned ResultPos = 0;
  for (unsigned i = 0, e = RegsToSpill.size(); i != e; ++i) {
    unsigned Reg = RegsToSpill[i];
    if (!LIS.hasInterval(Reg))
      continue;

    LiveInterval &LI = LIS.getInterval(Reg);
    if (LI.empty()) {
      Edit->eraseVirtReg(Reg);
      continue;
    }

    RegsToSpill[ResultPos++] = Reg;
  }
  RegsToSpill.erase(RegsToSpill.begin() + ResultPos, RegsToSpill.end());
  DEBUG(dbgs() << RegsToSpill.size() << " registers to spill after remat.\n");
}


//===----------------------------------------------------------------------===//
//                                 Spilling
//===----------------------------------------------------------------------===//

/// If MI is a load or store of StackSlot, it can be removed.
bool InlineSpiller::coalesceStackAccess(MachineInstr *MI, unsigned Reg) {
  int FI = 0;
  unsigned InstrReg = TII.isLoadFromStackSlot(MI, FI);
  bool IsLoad = InstrReg;
  if (!IsLoad)
    InstrReg = TII.isStoreToStackSlot(MI, FI);

  // We have a stack access. Is it the right register and slot?
  if (InstrReg != Reg || FI != StackSlot)
    return false;

  DEBUG(dbgs() << "Coalescing stack access: " << *MI);
  LIS.RemoveMachineInstrFromMaps(MI);
  MI->eraseFromParent();

  if (IsLoad) {
    ++NumReloadsRemoved;
    --NumReloads;
  } else {
    ++NumSpillsRemoved;
    --NumSpills;
  }

  return true;
}

#if !defined(NDEBUG)
// Dump the range of instructions from B to E with their slot indexes.
static void dumpMachineInstrRangeWithSlotIndex(MachineBasicBlock::iterator B,
                                               MachineBasicBlock::iterator E,
                                               LiveIntervals const &LIS,
                                               const char *const header,
                                               unsigned VReg =0) {
  char NextLine = '\n';
  char SlotIndent = '\t';

  if (std::next(B) == E) {
    NextLine = ' ';
    SlotIndent = ' ';
  }

  dbgs() << '\t' << header << ": " << NextLine;

  for (MachineBasicBlock::iterator I = B; I != E; ++I) {
    SlotIndex Idx = LIS.getInstructionIndex(I).getRegSlot();

    // If a register was passed in and this instruction has it as a
    // destination that is marked as an early clobber, print the
    // early-clobber slot index.
    if (VReg) {
      MachineOperand *MO = I->findRegisterDefOperand(VReg);
      if (MO && MO->isEarlyClobber())
        Idx = Idx.getRegSlot(true);
    }

    dbgs() << SlotIndent << Idx << '\t' << *I;
  }
}
#endif

/// foldMemoryOperand - Try folding stack slot references in Ops into their
/// instructions.
///
/// @param Ops    Operand indices from analyzeVirtReg().
/// @param LoadMI Load instruction to use instead of stack slot when non-null.
/// @return       True on success.
bool InlineSpiller::
foldMemoryOperand(ArrayRef<std::pair<MachineInstr*, unsigned> > Ops,
                  MachineInstr *LoadMI) {
  if (Ops.empty())
    return false;
  // Don't attempt folding in bundles.
  MachineInstr *MI = Ops.front().first;
  if (Ops.back().first != MI || MI->isBundled())
    return false;

  bool WasCopy = MI->isCopy();
  unsigned ImpReg = 0;

  bool SpillSubRegs = (MI->getOpcode() == TargetOpcode::STATEPOINT ||
                       MI->getOpcode() == TargetOpcode::PATCHPOINT ||
                       MI->getOpcode() == TargetOpcode::STACKMAP);

  // TargetInstrInfo::foldMemoryOperand only expects explicit, non-tied
  // operands.
  SmallVector<unsigned, 8> FoldOps;
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    unsigned Idx = Ops[i].second;
    assert(MI == Ops[i].first && "Instruction conflict during operand folding");
    MachineOperand &MO = MI->getOperand(Idx);
    if (MO.isImplicit()) {
      ImpReg = MO.getReg();
      continue;
    }
    // FIXME: Teach targets to deal with subregs.
    if (!SpillSubRegs && MO.getSubReg())
      return false;
    // We cannot fold a load instruction into a def.
    if (LoadMI && MO.isDef())
      return false;
    // Tied use operands should not be passed to foldMemoryOperand.
    if (!MI->isRegTiedToDefOperand(Idx))
      FoldOps.push_back(Idx);
  }

  MachineInstrSpan MIS(MI);

  MachineInstr *FoldMI =
                LoadMI ? TII.foldMemoryOperand(MI, FoldOps, LoadMI)
                       : TII.foldMemoryOperand(MI, FoldOps, StackSlot);
  if (!FoldMI)
    return false;

  // Remove LIS for any dead defs in the original MI not in FoldMI.
  for (MIBundleOperands MO(MI); MO.isValid(); ++MO) {
    if (!MO->isReg())
      continue;
    unsigned Reg = MO->getReg();
    if (!Reg || TargetRegisterInfo::isVirtualRegister(Reg) ||
        MRI.isReserved(Reg)) {
      continue;
    }
    // Skip non-Defs, including undef uses and internal reads.
    if (MO->isUse())
      continue;
    MIBundleOperands::PhysRegInfo RI =
      MIBundleOperands(FoldMI).analyzePhysReg(Reg, &TRI);
    if (RI.Defines)
      continue;
    // FoldMI does not define this physreg. Remove the LI segment.
    assert(MO->isDead() && "Cannot fold physreg def");
    SlotIndex Idx = LIS.getInstructionIndex(MI).getRegSlot();
    LIS.removePhysRegDefAt(Reg, Idx);
  }

  LIS.ReplaceMachineInstrInMaps(MI, FoldMI);
  MI->eraseFromParent();

  // Insert any new instructions other than FoldMI into the LIS maps.
  assert(!MIS.empty() && "Unexpected empty span of instructions!");
  for (MachineBasicBlock::iterator MII = MIS.begin(), End = MIS.end();
       MII != End; ++MII)
    if (&*MII != FoldMI)
      LIS.InsertMachineInstrInMaps(&*MII);

  // TII.foldMemoryOperand may have left some implicit operands on the
  // instruction.  Strip them.
  if (ImpReg)
    for (unsigned i = FoldMI->getNumOperands(); i; --i) {
      MachineOperand &MO = FoldMI->getOperand(i - 1);
      if (!MO.isReg() || !MO.isImplicit())
        break;
      if (MO.getReg() == ImpReg)
        FoldMI->RemoveOperand(i - 1);
    }

  DEBUG(dumpMachineInstrRangeWithSlotIndex(MIS.begin(), MIS.end(), LIS,
                                           "folded"));

  if (!WasCopy)
    ++NumFolded;
  else if (Ops.front().second == 0)
    ++NumSpills;
  else
    ++NumReloads;
  return true;
}

void InlineSpiller::insertReload(unsigned NewVReg,
                                 SlotIndex Idx,
                                 MachineBasicBlock::iterator MI) {
  MachineBasicBlock &MBB = *MI->getParent();

  MachineInstrSpan MIS(MI);
  TII.loadRegFromStackSlot(MBB, MI, NewVReg, StackSlot,
                           MRI.getRegClass(NewVReg), &TRI);

  LIS.InsertMachineInstrRangeInMaps(MIS.begin(), MI);

  DEBUG(dumpMachineInstrRangeWithSlotIndex(MIS.begin(), MI, LIS, "reload",
                                           NewVReg));
  ++NumReloads;
}

/// insertSpill - Insert a spill of NewVReg after MI.
void InlineSpiller::insertSpill(unsigned NewVReg, bool isKill,
                                 MachineBasicBlock::iterator MI) {
  MachineBasicBlock &MBB = *MI->getParent();

  MachineInstrSpan MIS(MI);
  TII.storeRegToStackSlot(MBB, std::next(MI), NewVReg, isKill, StackSlot,
                          MRI.getRegClass(NewVReg), &TRI);

  LIS.InsertMachineInstrRangeInMaps(std::next(MI), MIS.end());

  DEBUG(dumpMachineInstrRangeWithSlotIndex(std::next(MI), MIS.end(), LIS,
                                           "spill"));
  ++NumSpills;
}

/// spillAroundUses - insert spill code around each use of Reg.
void InlineSpiller::spillAroundUses(unsigned Reg) {
  DEBUG(dbgs() << "spillAroundUses " << PrintReg(Reg) << '\n');
  LiveInterval &OldLI = LIS.getInterval(Reg);

  // Iterate over instructions using Reg.
  for (MachineRegisterInfo::reg_bundle_iterator
       RegI = MRI.reg_bundle_begin(Reg), E = MRI.reg_bundle_end();
       RegI != E; ) {
    MachineInstr *MI = &*(RegI++);

    // Debug values are not allowed to affect codegen.
    if (MI->isDebugValue()) {
      // Modify DBG_VALUE now that the value is in a spill slot.
      bool IsIndirect = MI->isIndirectDebugValue();
      uint64_t Offset = IsIndirect ? MI->getOperand(1).getImm() : 0;
      const MDNode *Var = MI->getDebugVariable();
      const MDNode *Expr = MI->getDebugExpression();
      DebugLoc DL = MI->getDebugLoc();
      DEBUG(dbgs() << "Modifying debug info due to spill:" << "\t" << *MI);
      MachineBasicBlock *MBB = MI->getParent();
      assert(cast<DILocalVariable>(Var)->isValidLocationForIntrinsic(DL) &&
             "Expected inlined-at fields to agree");
      BuildMI(*MBB, MBB->erase(MI), DL, TII.get(TargetOpcode::DBG_VALUE))
          .addFrameIndex(StackSlot)
          .addImm(Offset)
          .addMetadata(Var)
          .addMetadata(Expr);
      continue;
    }

    // Ignore copies to/from snippets. We'll delete them.
    if (SnippetCopies.count(MI))
      continue;

    // Stack slot accesses may coalesce away.
    if (coalesceStackAccess(MI, Reg))
      continue;

    // Analyze instruction.
    SmallVector<std::pair<MachineInstr*, unsigned>, 8> Ops;
    MIBundleOperands::VirtRegInfo RI =
      MIBundleOperands(MI).analyzeVirtReg(Reg, &Ops);

    // Find the slot index where this instruction reads and writes OldLI.
    // This is usually the def slot, except for tied early clobbers.
    SlotIndex Idx = LIS.getInstructionIndex(MI).getRegSlot();
    if (VNInfo *VNI = OldLI.getVNInfoAt(Idx.getRegSlot(true)))
      if (SlotIndex::isSameInstr(Idx, VNI->def))
        Idx = VNI->def;

    // Check for a sibling copy.
    unsigned SibReg = isFullCopyOf(MI, Reg);
    if (SibReg && isSibling(SibReg)) {
      // This may actually be a copy between snippets.
      if (isRegToSpill(SibReg)) {
        DEBUG(dbgs() << "Found new snippet copy: " << *MI);
        SnippetCopies.insert(MI);
        continue;
      }
      if (RI.Writes) {
        // Hoist the spill of a sib-reg copy.
        if (hoistSpill(OldLI, MI)) {
          // This COPY is now dead, the value is already in the stack slot.
          MI->getOperand(0).setIsDead();
          DeadDefs.push_back(MI);
          continue;
        }
      } else {
        // This is a reload for a sib-reg copy. Drop spills downstream.
        LiveInterval &SibLI = LIS.getInterval(SibReg);
        eliminateRedundantSpills(SibLI, SibLI.getVNInfoAt(Idx));
        // The COPY will fold to a reload below.
      }
    }

    // Attempt to fold memory ops.
    if (foldMemoryOperand(Ops))
      continue;

    // Create a new virtual register for spill/fill.
    // FIXME: Infer regclass from instruction alone.
    unsigned NewVReg = Edit->createFrom(Reg);

    if (RI.Reads)
      insertReload(NewVReg, Idx, MI);

    // Rewrite instruction operands.
    bool hasLiveDef = false;
    for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
      MachineOperand &MO = Ops[i].first->getOperand(Ops[i].second);
      MO.setReg(NewVReg);
      if (MO.isUse()) {
        if (!Ops[i].first->isRegTiedToDefOperand(Ops[i].second))
          MO.setIsKill();
      } else {
        if (!MO.isDead())
          hasLiveDef = true;
      }
    }
    DEBUG(dbgs() << "\trewrite: " << Idx << '\t' << *MI << '\n');

    // FIXME: Use a second vreg if instruction has no tied ops.
    if (RI.Writes)
      if (hasLiveDef)
        insertSpill(NewVReg, true, MI);
  }
}

/// spillAll - Spill all registers remaining after rematerialization.
void InlineSpiller::spillAll() {
  // Update LiveStacks now that we are committed to spilling.
  if (StackSlot == VirtRegMap::NO_STACK_SLOT) {
    StackSlot = VRM.assignVirt2StackSlot(Original);
    StackInt = &LSS.getOrCreateInterval(StackSlot, MRI.getRegClass(Original));
    StackInt->getNextValue(SlotIndex(), LSS.getVNInfoAllocator());
  } else
    StackInt = &LSS.getInterval(StackSlot);

  if (Original != Edit->getReg())
    VRM.assignVirt2StackSlot(Edit->getReg(), StackSlot);

  assert(StackInt->getNumValNums() == 1 && "Bad stack interval values");
  for (unsigned i = 0, e = RegsToSpill.size(); i != e; ++i)
    StackInt->MergeSegmentsInAsValue(LIS.getInterval(RegsToSpill[i]),
                                     StackInt->getValNumInfo(0));
  DEBUG(dbgs() << "Merged spilled regs: " << *StackInt << '\n');

  // Spill around uses of all RegsToSpill.
  for (unsigned i = 0, e = RegsToSpill.size(); i != e; ++i)
    spillAroundUses(RegsToSpill[i]);

  // Hoisted spills may cause dead code.
  if (!DeadDefs.empty()) {
    DEBUG(dbgs() << "Eliminating " << DeadDefs.size() << " dead defs\n");
    Edit->eliminateDeadDefs(DeadDefs, RegsToSpill);
  }

  // Finally delete the SnippetCopies.
  for (unsigned i = 0, e = RegsToSpill.size(); i != e; ++i) {
    for (MachineRegisterInfo::reg_instr_iterator
         RI = MRI.reg_instr_begin(RegsToSpill[i]), E = MRI.reg_instr_end();
         RI != E; ) {
      MachineInstr *MI = &*(RI++);
      assert(SnippetCopies.count(MI) && "Remaining use wasn't a snippet copy");
      // FIXME: Do this with a LiveRangeEdit callback.
      LIS.RemoveMachineInstrFromMaps(MI);
      MI->eraseFromParent();
    }
  }

  // Delete all spilled registers.
  for (unsigned i = 0, e = RegsToSpill.size(); i != e; ++i)
    Edit->eraseVirtReg(RegsToSpill[i]);
}

void InlineSpiller::spill(LiveRangeEdit &edit) {
  ++NumSpilledRanges;
  Edit = &edit;
  assert(!TargetRegisterInfo::isStackSlot(edit.getReg())
         && "Trying to spill a stack slot.");
  // Share a stack slot among all descendants of Original.
  Original = VRM.getOriginal(edit.getReg());
  StackSlot = VRM.getStackSlot(Original);
  StackInt = nullptr;

  DEBUG(dbgs() << "Inline spilling "
               << TRI.getRegClassName(MRI.getRegClass(edit.getReg()))
               << ':' << edit.getParent()
               << "\nFrom original " << PrintReg(Original) << '\n');
  assert(edit.getParent().isSpillable() &&
         "Attempting to spill already spilled value.");
  assert(DeadDefs.empty() && "Previous spill didn't remove dead defs");

  collectRegsToSpill();
  analyzeSiblingValues();
  reMaterializeAll();

  // Remat may handle everything.
  if (!RegsToSpill.empty())
    spillAll();

  Edit->calculateRegClassAndHint(MF, Loops, MBFI);
}
