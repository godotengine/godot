//===---- LiveRangeCalc.cpp - Calculate live ranges -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the LiveRangeCalc class.
//
//===----------------------------------------------------------------------===//

#include "LiveRangeCalc.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "regalloc"

void LiveRangeCalc::resetLiveOutMap() {
  unsigned NumBlocks = MF->getNumBlockIDs();
  Seen.clear();
  Seen.resize(NumBlocks);
  Map.resize(NumBlocks);
}

void LiveRangeCalc::reset(const MachineFunction *mf,
                          SlotIndexes *SI,
                          MachineDominatorTree *MDT,
                          VNInfo::Allocator *VNIA) {
  MF = mf;
  MRI = &MF->getRegInfo();
  Indexes = SI;
  DomTree = MDT;
  Alloc = VNIA;
  resetLiveOutMap();
  LiveIn.clear();
}


static void createDeadDef(SlotIndexes &Indexes, VNInfo::Allocator &Alloc,
                          LiveRange &LR, const MachineOperand &MO) {
    const MachineInstr *MI = MO.getParent();
    SlotIndex DefIdx =
        Indexes.getInstructionIndex(MI).getRegSlot(MO.isEarlyClobber());

    // Create the def in LR. This may find an existing def.
    LR.createDeadDef(DefIdx, Alloc);
}

void LiveRangeCalc::calculate(LiveInterval &LI, bool TrackSubRegs) {
  assert(MRI && Indexes && "call reset() first");

  // Step 1: Create minimal live segments for every definition of Reg.
  // Visit all def operands. If the same instruction has multiple defs of Reg,
  // createDeadDef() will deduplicate.
  const TargetRegisterInfo &TRI = *MRI->getTargetRegisterInfo();
  unsigned Reg = LI.reg;
  for (const MachineOperand &MO : MRI->reg_nodbg_operands(Reg)) {
    if (!MO.isDef() && !MO.readsReg())
      continue;

    unsigned SubReg = MO.getSubReg();
    if (LI.hasSubRanges() || (SubReg != 0 && TrackSubRegs)) {
      unsigned Mask = SubReg != 0 ? TRI.getSubRegIndexLaneMask(SubReg)
                                  : MRI->getMaxLaneMaskForVReg(Reg);

      // If this is the first time we see a subregister def, initialize
      // subranges by creating a copy of the main range.
      if (!LI.hasSubRanges() && !LI.empty()) {
        unsigned ClassMask = MRI->getMaxLaneMaskForVReg(Reg);
        LI.createSubRangeFrom(*Alloc, ClassMask, LI);
      }

      for (LiveInterval::SubRange &S : LI.subranges()) {
        // A Mask for subregs common to the existing subrange and current def.
        unsigned Common = S.LaneMask & Mask;
        if (Common == 0)
          continue;
        // A Mask for subregs covered by the subrange but not the current def.
        unsigned LRest = S.LaneMask & ~Mask;
        LiveInterval::SubRange *CommonRange;
        if (LRest != 0) {
          // Split current subrange into Common and LRest ranges.
          S.LaneMask = LRest;
          CommonRange = LI.createSubRangeFrom(*Alloc, Common, S);
        } else {
          assert(Common == S.LaneMask);
          CommonRange = &S;
        }
        if (MO.isDef())
          createDeadDef(*Indexes, *Alloc, *CommonRange, MO);
        Mask &= ~Common;
      }
      // Create a new SubRange for subregs we did not cover yet.
      if (Mask != 0) {
        LiveInterval::SubRange *NewRange = LI.createSubRange(*Alloc, Mask);
        if (MO.isDef())
          createDeadDef(*Indexes, *Alloc, *NewRange, MO);
      }
    }

    // Create the def in the main liverange. We do not have to do this if
    // subranges are tracked as we recreate the main range later in this case.
    if (MO.isDef() && !LI.hasSubRanges())
      createDeadDef(*Indexes, *Alloc, LI, MO);
  }

  // We may have created empty live ranges for partially undefined uses, we
  // can't keep them because we won't find defs in them later.
  LI.removeEmptySubRanges();

  // Step 2: Extend live segments to all uses, constructing SSA form as
  // necessary.
  if (LI.hasSubRanges()) {
    for (LiveInterval::SubRange &S : LI.subranges()) {
      resetLiveOutMap();
      extendToUses(S, Reg, S.LaneMask);
    }
    LI.clear();
    LI.constructMainRangeFromSubranges(*Indexes, *Alloc);
  } else {
    resetLiveOutMap();
    extendToUses(LI, Reg, ~0u);
  }
}


void LiveRangeCalc::createDeadDefs(LiveRange &LR, unsigned Reg) {
  assert(MRI && Indexes && "call reset() first");

  // Visit all def operands. If the same instruction has multiple defs of Reg,
  // LR.createDeadDef() will deduplicate.
  for (MachineOperand &MO : MRI->def_operands(Reg))
    createDeadDef(*Indexes, *Alloc, LR, MO);
}


void LiveRangeCalc::extendToUses(LiveRange &LR, unsigned Reg, unsigned Mask) {
  // Visit all operands that read Reg. This may include partial defs.
  const TargetRegisterInfo &TRI = *MRI->getTargetRegisterInfo();
  for (MachineOperand &MO : MRI->reg_nodbg_operands(Reg)) {
    // Clear all kill flags. They will be reinserted after register allocation
    // by LiveIntervalAnalysis::addKillFlags().
    if (MO.isUse())
      MO.setIsKill(false);
    else {
      // We only care about uses, but on the main range (mask ~0u) this includes
      // the "virtual" reads happening for subregister defs.
      if (Mask != ~0u)
        continue;
    }

    if (!MO.readsReg())
      continue;
    unsigned SubReg = MO.getSubReg();
    if (SubReg != 0) {
      unsigned SubRegMask = TRI.getSubRegIndexLaneMask(SubReg);
      // Ignore uses not covering the current subrange.
      if ((SubRegMask & Mask) == 0)
        continue;
    }

    // Determine the actual place of the use.
    const MachineInstr *MI = MO.getParent();
    unsigned OpNo = (&MO - &MI->getOperand(0));
    SlotIndex UseIdx;
    if (MI->isPHI()) {
      assert(!MO.isDef() && "Cannot handle PHI def of partial register.");
      // The actual place where a phi operand is used is the end of the pred
      // MBB. PHI operands are paired: (Reg, PredMBB).
      UseIdx = Indexes->getMBBEndIdx(MI->getOperand(OpNo+1).getMBB());
    } else {
      // Check for early-clobber redefs.
      bool isEarlyClobber = false;
      unsigned DefIdx;
      if (MO.isDef())
        isEarlyClobber = MO.isEarlyClobber();
      else if (MI->isRegTiedToDefOperand(OpNo, &DefIdx)) {
        // FIXME: This would be a lot easier if tied early-clobber uses also
        // had an early-clobber flag.
        isEarlyClobber = MI->getOperand(DefIdx).isEarlyClobber();
      }
      UseIdx = Indexes->getInstructionIndex(MI).getRegSlot(isEarlyClobber);
    }

    // MI is reading Reg. We may have visited MI before if it happens to be
    // reading Reg multiple times. That is OK, extend() is idempotent.
    extend(LR, UseIdx, Reg);
  }
}


void LiveRangeCalc::updateFromLiveIns() {
  LiveRangeUpdater Updater;
  for (const LiveInBlock &I : LiveIn) {
    if (!I.DomNode)
      continue;
    MachineBasicBlock *MBB = I.DomNode->getBlock();
    assert(I.Value && "No live-in value found");
    SlotIndex Start, End;
    std::tie(Start, End) = Indexes->getMBBRange(MBB);

    if (I.Kill.isValid())
      // Value is killed inside this block.
      End = I.Kill;
    else {
      // The value is live-through, update LiveOut as well.
      // Defer the Domtree lookup until it is needed.
      assert(Seen.test(MBB->getNumber()));
      Map[MBB] = LiveOutPair(I.Value, nullptr);
    }
    Updater.setDest(&I.LR);
    Updater.add(Start, End, I.Value);
  }
  LiveIn.clear();
}


void LiveRangeCalc::extend(LiveRange &LR, SlotIndex Use, unsigned PhysReg) {
  assert(Use.isValid() && "Invalid SlotIndex");
  assert(Indexes && "Missing SlotIndexes");
  assert(DomTree && "Missing dominator tree");

  MachineBasicBlock *UseMBB = Indexes->getMBBFromIndex(Use.getPrevSlot());
  assert(UseMBB && "No MBB at Use");

  // Is there a def in the same MBB we can extend?
  if (LR.extendInBlock(Indexes->getMBBStartIdx(UseMBB), Use))
    return;

  // Find the single reaching def, or determine if Use is jointly dominated by
  // multiple values, and we may need to create even more phi-defs to preserve
  // VNInfo SSA form.  Perform a search for all predecessor blocks where we
  // know the dominating VNInfo.
  if (findReachingDefs(LR, *UseMBB, Use, PhysReg))
    return;

  // When there were multiple different values, we may need new PHIs.
  calculateValues();
}


// This function is called by a client after using the low-level API to add
// live-out and live-in blocks.  The unique value optimization is not
// available, SplitEditor::transferValues handles that case directly anyway.
void LiveRangeCalc::calculateValues() {
  assert(Indexes && "Missing SlotIndexes");
  assert(DomTree && "Missing dominator tree");
  updateSSA();
  updateFromLiveIns();
}


bool LiveRangeCalc::findReachingDefs(LiveRange &LR, MachineBasicBlock &UseMBB,
                                     SlotIndex Use, unsigned PhysReg) {
  unsigned UseMBBNum = UseMBB.getNumber();

  // Block numbers where LR should be live-in.
  SmallVector<unsigned, 16> WorkList(1, UseMBBNum);

  // Remember if we have seen more than one value.
  bool UniqueVNI = true;
  VNInfo *TheVNI = nullptr;

  // Using Seen as a visited set, perform a BFS for all reaching defs.
  for (unsigned i = 0; i != WorkList.size(); ++i) {
    MachineBasicBlock *MBB = MF->getBlockNumbered(WorkList[i]);

#ifndef NDEBUG
    if (MBB->pred_empty()) {
      MBB->getParent()->verify();
      errs() << "Use of " << PrintReg(PhysReg)
             << " does not have a corresponding definition on every path:\n";
      const MachineInstr *MI = Indexes->getInstructionFromIndex(Use);
      if (MI != nullptr)
        errs() << Use << " " << *MI;
      llvm_unreachable("Use not jointly dominated by defs.");
    }

    if (TargetRegisterInfo::isPhysicalRegister(PhysReg) &&
        !MBB->isLiveIn(PhysReg)) {
      MBB->getParent()->verify();
      errs() << "The register " << PrintReg(PhysReg)
             << " needs to be live in to BB#" << MBB->getNumber()
             << ", but is missing from the live-in list.\n";
      llvm_unreachable("Invalid global physical register");
    }
#endif

    for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
         PE = MBB->pred_end(); PI != PE; ++PI) {
       MachineBasicBlock *Pred = *PI;

       // Is this a known live-out block?
       if (Seen.test(Pred->getNumber())) {
         if (VNInfo *VNI = Map[Pred].first) {
           if (TheVNI && TheVNI != VNI)
             UniqueVNI = false;
           TheVNI = VNI;
         }
         continue;
       }

       SlotIndex Start, End;
       std::tie(Start, End) = Indexes->getMBBRange(Pred);

       // First time we see Pred.  Try to determine the live-out value, but set
       // it as null if Pred is live-through with an unknown value.
       VNInfo *VNI = LR.extendInBlock(Start, End);
       setLiveOutValue(Pred, VNI);
       if (VNI) {
         if (TheVNI && TheVNI != VNI)
           UniqueVNI = false;
         TheVNI = VNI;
         continue;
       }

       // No, we need a live-in value for Pred as well
       if (Pred != &UseMBB)
          WorkList.push_back(Pred->getNumber());
       else
          // Loopback to UseMBB, so value is really live through.
         Use = SlotIndex();
    }
  }

  LiveIn.clear();

  // Both updateSSA() and LiveRangeUpdater benefit from ordered blocks, but
  // neither require it. Skip the sorting overhead for small updates.
  if (WorkList.size() > 4)
    array_pod_sort(WorkList.begin(), WorkList.end());

  // If a unique reaching def was found, blit in the live ranges immediately.
  if (UniqueVNI) {
    LiveRangeUpdater Updater(&LR);
    for (SmallVectorImpl<unsigned>::const_iterator I = WorkList.begin(),
         E = WorkList.end(); I != E; ++I) {
       SlotIndex Start, End;
       std::tie(Start, End) = Indexes->getMBBRange(*I);
       // Trim the live range in UseMBB.
       if (*I == UseMBBNum && Use.isValid())
         End = Use;
       else
         Map[MF->getBlockNumbered(*I)] = LiveOutPair(TheVNI, nullptr);
       Updater.add(Start, End, TheVNI);
    }
    return true;
  }

  // Multiple values were found, so transfer the work list to the LiveIn array
  // where UpdateSSA will use it as a work list.
  LiveIn.reserve(WorkList.size());
  for (SmallVectorImpl<unsigned>::const_iterator
       I = WorkList.begin(), E = WorkList.end(); I != E; ++I) {
    MachineBasicBlock *MBB = MF->getBlockNumbered(*I);
    addLiveInBlock(LR, DomTree->getNode(MBB));
    if (MBB == &UseMBB)
      LiveIn.back().Kill = Use;
  }

  return false;
}


// This is essentially the same iterative algorithm that SSAUpdater uses,
// except we already have a dominator tree, so we don't have to recompute it.
void LiveRangeCalc::updateSSA() {
  assert(Indexes && "Missing SlotIndexes");
  assert(DomTree && "Missing dominator tree");

  // Interate until convergence.
  unsigned Changes;
  do {
    Changes = 0;
    // Propagate live-out values down the dominator tree, inserting phi-defs
    // when necessary.
    for (LiveInBlock &I : LiveIn) {
      MachineDomTreeNode *Node = I.DomNode;
      // Skip block if the live-in value has already been determined.
      if (!Node)
        continue;
      MachineBasicBlock *MBB = Node->getBlock();
      MachineDomTreeNode *IDom = Node->getIDom();
      LiveOutPair IDomValue;

      // We need a live-in value to a block with no immediate dominator?
      // This is probably an unreachable block that has survived somehow.
      bool needPHI = !IDom || !Seen.test(IDom->getBlock()->getNumber());

      // IDom dominates all of our predecessors, but it may not be their
      // immediate dominator. Check if any of them have live-out values that are
      // properly dominated by IDom. If so, we need a phi-def here.
      if (!needPHI) {
        IDomValue = Map[IDom->getBlock()];

        // Cache the DomTree node that defined the value.
        if (IDomValue.first && !IDomValue.second)
          Map[IDom->getBlock()].second = IDomValue.second =
            DomTree->getNode(Indexes->getMBBFromIndex(IDomValue.first->def));

        for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
               PE = MBB->pred_end(); PI != PE; ++PI) {
          LiveOutPair &Value = Map[*PI];
          if (!Value.first || Value.first == IDomValue.first)
            continue;

          // Cache the DomTree node that defined the value.
          if (!Value.second)
            Value.second =
              DomTree->getNode(Indexes->getMBBFromIndex(Value.first->def));

          // This predecessor is carrying something other than IDomValue.
          // It could be because IDomValue hasn't propagated yet, or it could be
          // because MBB is in the dominance frontier of that value.
          if (DomTree->dominates(IDom, Value.second)) {
            needPHI = true;
            break;
          }
        }
      }

      // The value may be live-through even if Kill is set, as can happen when
      // we are called from extendRange. In that case LiveOutSeen is true, and
      // LiveOut indicates a foreign or missing value.
      LiveOutPair &LOP = Map[MBB];

      // Create a phi-def if required.
      if (needPHI) {
        ++Changes;
        assert(Alloc && "Need VNInfo allocator to create PHI-defs");
        SlotIndex Start, End;
        std::tie(Start, End) = Indexes->getMBBRange(MBB);
        LiveRange &LR = I.LR;
        VNInfo *VNI = LR.getNextValue(Start, *Alloc);
        I.Value = VNI;
        // This block is done, we know the final value.
        I.DomNode = nullptr;

        // Add liveness since updateFromLiveIns now skips this node.
        if (I.Kill.isValid())
          LR.addSegment(LiveInterval::Segment(Start, I.Kill, VNI));
        else {
          LR.addSegment(LiveInterval::Segment(Start, End, VNI));
          LOP = LiveOutPair(VNI, Node);
        }
      } else if (IDomValue.first) {
        // No phi-def here. Remember incoming value.
        I.Value = IDomValue.first;

        // If the IDomValue is killed in the block, don't propagate through.
        if (I.Kill.isValid())
          continue;

        // Propagate IDomValue if it isn't killed:
        // MBB is live-out and doesn't define its own value.
        if (LOP.first == IDomValue.first)
          continue;
        ++Changes;
        LOP = IDomValue;
      }
    }
  } while (Changes);
}
