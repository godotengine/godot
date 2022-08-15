//===-- PrologEpilogInserter.cpp - Insert Prolog/Epilog code in function --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is responsible for finalizing the functions frame layout, saving
// callee saved registers, and for emitting prolog & epilog code for the
// function.
//
// This pass must be run after register allocation.  After this pass is
// executed, it is illegal to construct MO_FrameIndex operands.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/CodeGen/WinEHFuncInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <climits>

using namespace llvm;

#define DEBUG_TYPE "pei"

namespace {
class PEI : public MachineFunctionPass {
public:
  static char ID;
  PEI() : MachineFunctionPass(ID) {
    initializePEIPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// runOnMachineFunction - Insert prolog/epilog code and replace abstract
  /// frame indexes with appropriate references.
  ///
  bool runOnMachineFunction(MachineFunction &Fn) override;

private:
  RegScavenger *RS;

  // MinCSFrameIndex, MaxCSFrameIndex - Keeps the range of callee saved
  // stack frame indexes.
  unsigned MinCSFrameIndex, MaxCSFrameIndex;

  // Save and Restore blocks of the current function.
  MachineBasicBlock *SaveBlock;
  SmallVector<MachineBasicBlock *, 4> RestoreBlocks;

  // Flag to control whether to use the register scavenger to resolve
  // frame index materialization registers. Set according to
  // TRI->requiresFrameIndexScavenging() for the current function.
  bool FrameIndexVirtualScavenging;

  void calculateSets(MachineFunction &Fn);
  void calculateCallsInformation(MachineFunction &Fn);
  void assignCalleeSavedSpillSlots(MachineFunction &Fn,
                                   const BitVector &SavedRegs);
  void insertCSRSpillsAndRestores(MachineFunction &Fn);
  void calculateFrameObjectOffsets(MachineFunction &Fn);
  void replaceFrameIndices(MachineFunction &Fn);
  void replaceFrameIndices(MachineBasicBlock *BB, MachineFunction &Fn,
                           int &SPAdj);
  void scavengeFrameVirtualRegs(MachineFunction &Fn);
  void insertPrologEpilogCode(MachineFunction &Fn);

  // Convenience for recognizing return blocks.
  bool isReturnBlock(const MachineBasicBlock *MBB) const;
};
} // namespace

char PEI::ID = 0;
char &llvm::PrologEpilogCodeInserterID = PEI::ID;

static cl::opt<unsigned>
WarnStackSize("warn-stack-size", cl::Hidden, cl::init((unsigned)-1),
              cl::desc("Warn for stack size bigger than the given"
                       " number"));

INITIALIZE_PASS_BEGIN(PEI, "prologepilog",
                "Prologue/Epilogue Insertion", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(StackProtector)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(PEI, "prologepilog",
                    "Prologue/Epilogue Insertion & Frame Finalization",
                    false, false)

STATISTIC(NumScavengedRegs, "Number of frame index regs scavenged");
STATISTIC(NumBytesStackSpace,
          "Number of bytes used for stack in all functions");

void PEI::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addPreserved<MachineLoopInfo>();
  AU.addPreserved<MachineDominatorTree>();
  AU.addRequired<StackProtector>();
  AU.addRequired<TargetPassConfig>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool PEI::isReturnBlock(const MachineBasicBlock* MBB) const {
  return (MBB && !MBB->empty() && MBB->back().isReturn());
}

/// Compute the set of return blocks
void PEI::calculateSets(MachineFunction &Fn) {
  const MachineFrameInfo *MFI = Fn.getFrameInfo();

  // Even when we do not change any CSR, we still want to insert the
  // prologue and epilogue of the function.
  // So set the save points for those.

  // Use the points found by shrink-wrapping, if any.
  if (MFI->getSavePoint()) {
    SaveBlock = MFI->getSavePoint();
    assert(MFI->getRestorePoint() && "Both restore and save must be set");
    MachineBasicBlock *RestoreBlock = MFI->getRestorePoint();
    // If RestoreBlock does not have any successor and is not a return block
    // then the end point is unreachable and we do not need to insert any
    // epilogue.
    if (!RestoreBlock->succ_empty() || isReturnBlock(RestoreBlock))
      RestoreBlocks.push_back(RestoreBlock);
    return;
  }

  // Save refs to entry and return blocks.
  SaveBlock = Fn.begin();
  for (MachineFunction::iterator MBB = Fn.begin(), E = Fn.end();
       MBB != E; ++MBB)
    if (isReturnBlock(MBB))
      RestoreBlocks.push_back(MBB);

  return;
}

/// StackObjSet - A set of stack object indexes
typedef SmallSetVector<int, 8> StackObjSet;

/// runOnMachineFunction - Insert prolog/epilog code and replace abstract
/// frame indexes with appropriate references.
///
bool PEI::runOnMachineFunction(MachineFunction &Fn) {
  const Function* F = Fn.getFunction();
  const TargetRegisterInfo *TRI = Fn.getSubtarget().getRegisterInfo();
  const TargetFrameLowering *TFI = Fn.getSubtarget().getFrameLowering();

  assert(!Fn.getRegInfo().getNumVirtRegs() && "Regalloc must assign all vregs");

  RS = TRI->requiresRegisterScavenging(Fn) ? new RegScavenger() : nullptr;
  FrameIndexVirtualScavenging = TRI->requiresFrameIndexScavenging(Fn);

  // Calculate the MaxCallFrameSize and AdjustsStack variables for the
  // function's frame information. Also eliminates call frame pseudo
  // instructions.
  calculateCallsInformation(Fn);

  // Determine which of the registers in the callee save list should be saved.
  BitVector SavedRegs;
  TFI->determineCalleeSaves(Fn, SavedRegs, RS);

  // Insert spill code for any callee saved registers that are modified.
  assignCalleeSavedSpillSlots(Fn, SavedRegs);

  // Determine placement of CSR spill/restore code:
  // place all spills in the entry block, all restores in return blocks.
  calculateSets(Fn);

  // Add the code to save and restore the callee saved registers
  if (!F->hasFnAttribute(Attribute::Naked))
    insertCSRSpillsAndRestores(Fn);

  // Allow the target machine to make final modifications to the function
  // before the frame layout is finalized.
  TFI->processFunctionBeforeFrameFinalized(Fn, RS);

  // Calculate actual frame offsets for all abstract stack objects...
  calculateFrameObjectOffsets(Fn);

  // Add prolog and epilog code to the function.  This function is required
  // to align the stack frame as necessary for any stack variables or
  // called functions.  Because of this, calculateCalleeSavedRegisters()
  // must be called before this function in order to set the AdjustsStack
  // and MaxCallFrameSize variables.
  if (!F->hasFnAttribute(Attribute::Naked))
    insertPrologEpilogCode(Fn);

  // Replace all MO_FrameIndex operands with physical register references
  // and actual offsets.
  //
  replaceFrameIndices(Fn);

  // If register scavenging is needed, as we've enabled doing it as a
  // post-pass, scavenge the virtual registers that frame index elimination
  // inserted.
  if (TRI->requiresRegisterScavenging(Fn) && FrameIndexVirtualScavenging)
    scavengeFrameVirtualRegs(Fn);

  // Clear any vregs created by virtual scavenging.
  Fn.getRegInfo().clearVirtRegs();

  // Warn on stack size when we exceeds the given limit.
  MachineFrameInfo *MFI = Fn.getFrameInfo();
  uint64_t StackSize = MFI->getStackSize();
  if (WarnStackSize.getNumOccurrences() > 0 && WarnStackSize < StackSize) {
    DiagnosticInfoStackSize DiagStackSize(*F, StackSize);
    F->getContext().diagnose(DiagStackSize);
  }

  delete RS;
  RestoreBlocks.clear();
  return true;
}

/// calculateCallsInformation - Calculate the MaxCallFrameSize and AdjustsStack
/// variables for the function's frame information and eliminate call frame
/// pseudo instructions.
void PEI::calculateCallsInformation(MachineFunction &Fn) {
  const TargetInstrInfo &TII = *Fn.getSubtarget().getInstrInfo();
  const TargetFrameLowering *TFI = Fn.getSubtarget().getFrameLowering();
  MachineFrameInfo *MFI = Fn.getFrameInfo();

  unsigned MaxCallFrameSize = 0;
  bool AdjustsStack = MFI->adjustsStack();

  // Get the function call frame set-up and tear-down instruction opcode
  unsigned FrameSetupOpcode = TII.getCallFrameSetupOpcode();
  unsigned FrameDestroyOpcode = TII.getCallFrameDestroyOpcode();

  // Early exit for targets which have no call frame setup/destroy pseudo
  // instructions.
  if (FrameSetupOpcode == ~0u && FrameDestroyOpcode == ~0u)
    return;

  std::vector<MachineBasicBlock::iterator> FrameSDOps;
  for (MachineFunction::iterator BB = Fn.begin(), E = Fn.end(); BB != E; ++BB)
    for (MachineBasicBlock::iterator I = BB->begin(); I != BB->end(); ++I)
      if (I->getOpcode() == FrameSetupOpcode ||
          I->getOpcode() == FrameDestroyOpcode) {
        assert(I->getNumOperands() >= 1 && "Call Frame Setup/Destroy Pseudo"
               " instructions should have a single immediate argument!");
        unsigned Size = I->getOperand(0).getImm();
        if (Size > MaxCallFrameSize) MaxCallFrameSize = Size;
        AdjustsStack = true;
        FrameSDOps.push_back(I);
      } else if (I->isInlineAsm()) {
        // Some inline asm's need a stack frame, as indicated by operand 1.
        unsigned ExtraInfo = I->getOperand(InlineAsm::MIOp_ExtraInfo).getImm();
        if (ExtraInfo & InlineAsm::Extra_IsAlignStack)
          AdjustsStack = true;
      }

  MFI->setAdjustsStack(AdjustsStack);
  MFI->setMaxCallFrameSize(MaxCallFrameSize);

  for (std::vector<MachineBasicBlock::iterator>::iterator
         i = FrameSDOps.begin(), e = FrameSDOps.end(); i != e; ++i) {
    MachineBasicBlock::iterator I = *i;

    // If call frames are not being included as part of the stack frame, and
    // the target doesn't indicate otherwise, remove the call frame pseudos
    // here. The sub/add sp instruction pairs are still inserted, but we don't
    // need to track the SP adjustment for frame index elimination.
    if (TFI->canSimplifyCallFramePseudos(Fn))
      TFI->eliminateCallFramePseudoInstr(Fn, *I->getParent(), I);
  }
}

void PEI::assignCalleeSavedSpillSlots(MachineFunction &F,
                                      const BitVector &SavedRegs) {
  // These are used to keep track the callee-save area. Initialize them.
  MinCSFrameIndex = INT_MAX;
  MaxCSFrameIndex = 0;

  if (SavedRegs.empty())
    return;

  const TargetRegisterInfo *RegInfo = F.getSubtarget().getRegisterInfo();
  const MCPhysReg *CSRegs = RegInfo->getCalleeSavedRegs(&F);

  std::vector<CalleeSavedInfo> CSI;
  for (unsigned i = 0; CSRegs[i]; ++i) {
    unsigned Reg = CSRegs[i];
    if (SavedRegs.test(Reg))
      CSI.push_back(CalleeSavedInfo(Reg));
  }

  const TargetFrameLowering *TFI = F.getSubtarget().getFrameLowering();
  MachineFrameInfo *MFI = F.getFrameInfo();
  if (!TFI->assignCalleeSavedSpillSlots(F, RegInfo, CSI)) {
    // If target doesn't implement this, use generic code.

    if (CSI.empty())
      return; // Early exit if no callee saved registers are modified!

    unsigned NumFixedSpillSlots;
    const TargetFrameLowering::SpillSlot *FixedSpillSlots =
        TFI->getCalleeSavedSpillSlots(NumFixedSpillSlots);

    // Now that we know which registers need to be saved and restored, allocate
    // stack slots for them.
    for (std::vector<CalleeSavedInfo>::iterator I = CSI.begin(), E = CSI.end();
         I != E; ++I) {
      unsigned Reg = I->getReg();
      const TargetRegisterClass *RC = RegInfo->getMinimalPhysRegClass(Reg);

      int FrameIdx;
      if (RegInfo->hasReservedSpillSlot(F, Reg, FrameIdx)) {
        I->setFrameIdx(FrameIdx);
        continue;
      }

      // Check to see if this physreg must be spilled to a particular stack slot
      // on this target.
      const TargetFrameLowering::SpillSlot *FixedSlot = FixedSpillSlots;
      while (FixedSlot != FixedSpillSlots + NumFixedSpillSlots &&
             FixedSlot->Reg != Reg)
        ++FixedSlot;

      if (FixedSlot == FixedSpillSlots + NumFixedSpillSlots) {
        // Nope, just spill it anywhere convenient.
        unsigned Align = RC->getAlignment();
        unsigned StackAlign = TFI->getStackAlignment();

        // We may not be able to satisfy the desired alignment specification of
        // the TargetRegisterClass if the stack alignment is smaller. Use the
        // min.
        Align = std::min(Align, StackAlign);
        FrameIdx = MFI->CreateStackObject(RC->getSize(), Align, true);
        if ((unsigned)FrameIdx < MinCSFrameIndex) MinCSFrameIndex = FrameIdx;
        if ((unsigned)FrameIdx > MaxCSFrameIndex) MaxCSFrameIndex = FrameIdx;
      } else {
        // Spill it to the stack where we must.
        FrameIdx =
            MFI->CreateFixedSpillStackObject(RC->getSize(), FixedSlot->Offset);
      }

      I->setFrameIdx(FrameIdx);
    }
  }

  MFI->setCalleeSavedInfo(CSI);
}

/// Helper function to update the liveness information for the callee-saved
/// registers.
static void updateLiveness(MachineFunction &MF) {
  MachineFrameInfo *MFI = MF.getFrameInfo();
  // Visited will contain all the basic blocks that are in the region
  // where the callee saved registers are alive:
  // - Anything that is not Save or Restore -> LiveThrough.
  // - Save -> LiveIn.
  // - Restore -> LiveOut.
  // The live-out is not attached to the block, so no need to keep
  // Restore in this set.
  SmallPtrSet<MachineBasicBlock *, 8> Visited;
  SmallVector<MachineBasicBlock *, 8> WorkList;
  MachineBasicBlock *Entry = &MF.front();
  MachineBasicBlock *Save = MFI->getSavePoint();

  if (!Save)
    Save = Entry;

  if (Entry != Save) {
    WorkList.push_back(Entry);
    Visited.insert(Entry);
  }
  Visited.insert(Save);

  MachineBasicBlock *Restore = MFI->getRestorePoint();
  if (Restore)
    // By construction Restore cannot be visited, otherwise it
    // means there exists a path to Restore that does not go
    // through Save.
    WorkList.push_back(Restore);

  while (!WorkList.empty()) {
    const MachineBasicBlock *CurBB = WorkList.pop_back_val();
    // By construction, the region that is after the save point is
    // dominated by the Save and post-dominated by the Restore.
    if (CurBB == Save)
      continue;
    // Enqueue all the successors not already visited.
    // Those are by construction either before Save or after Restore.
    for (MachineBasicBlock *SuccBB : CurBB->successors())
      if (Visited.insert(SuccBB).second)
        WorkList.push_back(SuccBB);
  }

  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();

  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    for (MachineBasicBlock *MBB : Visited)
      // Add the callee-saved register as live-in.
      // It's killed at the spill.
      MBB->addLiveIn(CSI[i].getReg());
  }
}

/// insertCSRSpillsAndRestores - Insert spill and restore code for
/// callee saved registers used in the function.
///
void PEI::insertCSRSpillsAndRestores(MachineFunction &Fn) {
  // Get callee saved register information.
  MachineFrameInfo *MFI = Fn.getFrameInfo();
  const std::vector<CalleeSavedInfo> &CSI = MFI->getCalleeSavedInfo();

  MFI->setCalleeSavedInfoValid(true);

  // Early exit if no callee saved registers are modified!
  if (CSI.empty())
    return;

  const TargetInstrInfo &TII = *Fn.getSubtarget().getInstrInfo();
  const TargetFrameLowering *TFI = Fn.getSubtarget().getFrameLowering();
  const TargetRegisterInfo *TRI = Fn.getSubtarget().getRegisterInfo();
  MachineBasicBlock::iterator I;

  // Spill using target interface.
  I = SaveBlock->begin();
  if (!TFI->spillCalleeSavedRegisters(*SaveBlock, I, CSI, TRI)) {
    for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
      // Insert the spill to the stack frame.
      unsigned Reg = CSI[i].getReg();
      const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
      TII.storeRegToStackSlot(*SaveBlock, I, Reg, true, CSI[i].getFrameIdx(),
                              RC, TRI);
    }
  }
  // Update the live-in information of all the blocks up to the save point.
  updateLiveness(Fn);

  // Restore using target interface.
  for (MachineBasicBlock *MBB : RestoreBlocks) {
    I = MBB->end();

    // Skip over all terminator instructions, which are part of the return
    // sequence.
    MachineBasicBlock::iterator I2 = I;
    while (I2 != MBB->begin() && (--I2)->isTerminator())
      I = I2;

    bool AtStart = I == MBB->begin();
    MachineBasicBlock::iterator BeforeI = I;
    if (!AtStart)
      --BeforeI;

    // Restore all registers immediately before the return and any
    // terminators that precede it.
    if (!TFI->restoreCalleeSavedRegisters(*MBB, I, CSI, TRI)) {
      for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
        unsigned Reg = CSI[i].getReg();
        const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
        TII.loadRegFromStackSlot(*MBB, I, Reg, CSI[i].getFrameIdx(), RC, TRI);
        assert(I != MBB->begin() &&
               "loadRegFromStackSlot didn't insert any code!");
        // Insert in reverse order.  loadRegFromStackSlot can insert
        // multiple instructions.
        if (AtStart)
          I = MBB->begin();
        else {
          I = BeforeI;
          ++I;
        }
      }
    }
  }
}

/// AdjustStackOffset - Helper function used to adjust the stack frame offset.
static inline void
AdjustStackOffset(MachineFrameInfo *MFI, int FrameIdx,
                  bool StackGrowsDown, int64_t &Offset,
                  unsigned &MaxAlign) {
  // If the stack grows down, add the object size to find the lowest address.
  if (StackGrowsDown)
    Offset += MFI->getObjectSize(FrameIdx);

  unsigned Align = MFI->getObjectAlignment(FrameIdx);

  // If the alignment of this object is greater than that of the stack, then
  // increase the stack alignment to match.
  MaxAlign = std::max(MaxAlign, Align);

  // Adjust to alignment boundary.
  Offset = (Offset + Align - 1) / Align * Align;

  if (StackGrowsDown) {
    DEBUG(dbgs() << "alloc FI(" << FrameIdx << ") at SP[" << -Offset << "]\n");
    MFI->setObjectOffset(FrameIdx, -Offset); // Set the computed offset
  } else {
    DEBUG(dbgs() << "alloc FI(" << FrameIdx << ") at SP[" << Offset << "]\n");
    MFI->setObjectOffset(FrameIdx, Offset);
    Offset += MFI->getObjectSize(FrameIdx);
  }
}

/// AssignProtectedObjSet - Helper function to assign large stack objects (i.e.,
/// those required to be close to the Stack Protector) to stack offsets.
static void
AssignProtectedObjSet(const StackObjSet &UnassignedObjs,
                      SmallSet<int, 16> &ProtectedObjs,
                      MachineFrameInfo *MFI, bool StackGrowsDown,
                      int64_t &Offset, unsigned &MaxAlign) {

  for (StackObjSet::const_iterator I = UnassignedObjs.begin(),
        E = UnassignedObjs.end(); I != E; ++I) {
    int i = *I;
    AdjustStackOffset(MFI, i, StackGrowsDown, Offset, MaxAlign);
    ProtectedObjs.insert(i);
  }
}

/// calculateFrameObjectOffsets - Calculate actual frame offsets for all of the
/// abstract stack objects.
///
void PEI::calculateFrameObjectOffsets(MachineFunction &Fn) {
  const TargetFrameLowering &TFI = *Fn.getSubtarget().getFrameLowering();
  StackProtector *SP = &getAnalysis<StackProtector>();

  bool StackGrowsDown =
    TFI.getStackGrowthDirection() == TargetFrameLowering::StackGrowsDown;

  // Loop over all of the stack objects, assigning sequential addresses...
  MachineFrameInfo *MFI = Fn.getFrameInfo();

  // Start at the beginning of the local area.
  // The Offset is the distance from the stack top in the direction
  // of stack growth -- so it's always nonnegative.
  int LocalAreaOffset = TFI.getOffsetOfLocalArea();
  if (StackGrowsDown)
    LocalAreaOffset = -LocalAreaOffset;
  assert(LocalAreaOffset >= 0
         && "Local area offset should be in direction of stack growth");
  int64_t Offset = LocalAreaOffset;

  // If there are fixed sized objects that are preallocated in the local area,
  // non-fixed objects can't be allocated right at the start of local area.
  // We currently don't support filling in holes in between fixed sized
  // objects, so we adjust 'Offset' to point to the end of last fixed sized
  // preallocated object.
  for (int i = MFI->getObjectIndexBegin(); i != 0; ++i) {
    int64_t FixedOff;
    if (StackGrowsDown) {
      // The maximum distance from the stack pointer is at lower address of
      // the object -- which is given by offset. For down growing stack
      // the offset is negative, so we negate the offset to get the distance.
      FixedOff = -MFI->getObjectOffset(i);
    } else {
      // The maximum distance from the start pointer is at the upper
      // address of the object.
      FixedOff = MFI->getObjectOffset(i) + MFI->getObjectSize(i);
    }
    if (FixedOff > Offset) Offset = FixedOff;
  }

  // First assign frame offsets to stack objects that are used to spill
  // callee saved registers.
  if (StackGrowsDown) {
    for (unsigned i = MinCSFrameIndex; i <= MaxCSFrameIndex; ++i) {
      // If the stack grows down, we need to add the size to find the lowest
      // address of the object.
      Offset += MFI->getObjectSize(i);

      unsigned Align = MFI->getObjectAlignment(i);
      // Adjust to alignment boundary
      Offset = RoundUpToAlignment(Offset, Align);

      MFI->setObjectOffset(i, -Offset);        // Set the computed offset
    }
  } else {
    int MaxCSFI = MaxCSFrameIndex, MinCSFI = MinCSFrameIndex;
    for (int i = MaxCSFI; i >= MinCSFI ; --i) {
      unsigned Align = MFI->getObjectAlignment(i);
      // Adjust to alignment boundary
      Offset = RoundUpToAlignment(Offset, Align);

      MFI->setObjectOffset(i, Offset);
      Offset += MFI->getObjectSize(i);
    }
  }

  unsigned MaxAlign = MFI->getMaxAlignment();

  // Make sure the special register scavenging spill slot is closest to the
  // incoming stack pointer if a frame pointer is required and is closer
  // to the incoming rather than the final stack pointer.
  const TargetRegisterInfo *RegInfo = Fn.getSubtarget().getRegisterInfo();
  bool EarlyScavengingSlots = (TFI.hasFP(Fn) &&
                               TFI.isFPCloseToIncomingSP() &&
                               RegInfo->useFPForScavengingIndex(Fn) &&
                               !RegInfo->needsStackRealignment(Fn));
  if (RS && EarlyScavengingSlots) {
    SmallVector<int, 2> SFIs;
    RS->getScavengingFrameIndices(SFIs);
    for (SmallVectorImpl<int>::iterator I = SFIs.begin(),
           IE = SFIs.end(); I != IE; ++I)
      AdjustStackOffset(MFI, *I, StackGrowsDown, Offset, MaxAlign);
  }

  // FIXME: Once this is working, then enable flag will change to a target
  // check for whether the frame is large enough to want to use virtual
  // frame index registers. Functions which don't want/need this optimization
  // will continue to use the existing code path.
  if (MFI->getUseLocalStackAllocationBlock()) {
    unsigned Align = MFI->getLocalFrameMaxAlign();

    // Adjust to alignment boundary.
    Offset = RoundUpToAlignment(Offset, Align);

    DEBUG(dbgs() << "Local frame base offset: " << Offset << "\n");

    // Resolve offsets for objects in the local block.
    for (unsigned i = 0, e = MFI->getLocalFrameObjectCount(); i != e; ++i) {
      std::pair<int, int64_t> Entry = MFI->getLocalFrameObjectMap(i);
      int64_t FIOffset = (StackGrowsDown ? -Offset : Offset) + Entry.second;
      DEBUG(dbgs() << "alloc FI(" << Entry.first << ") at SP[" <<
            FIOffset << "]\n");
      MFI->setObjectOffset(Entry.first, FIOffset);
    }
    // Allocate the local block
    Offset += MFI->getLocalFrameSize();

    MaxAlign = std::max(Align, MaxAlign);
  }

  // Make sure that the stack protector comes before the local variables on the
  // stack.
  SmallSet<int, 16> ProtectedObjs;
  if (MFI->getStackProtectorIndex() >= 0) {
    StackObjSet LargeArrayObjs;
    StackObjSet SmallArrayObjs;
    StackObjSet AddrOfObjs;

    AdjustStackOffset(MFI, MFI->getStackProtectorIndex(), StackGrowsDown,
                      Offset, MaxAlign);

    // Assign large stack objects first.
    for (unsigned i = 0, e = MFI->getObjectIndexEnd(); i != e; ++i) {
      if (MFI->isObjectPreAllocated(i) &&
          MFI->getUseLocalStackAllocationBlock())
        continue;
      if (i >= MinCSFrameIndex && i <= MaxCSFrameIndex)
        continue;
      if (RS && RS->isScavengingFrameIndex((int)i))
        continue;
      if (MFI->isDeadObjectIndex(i))
        continue;
      if (MFI->getStackProtectorIndex() == (int)i)
        continue;

      switch (SP->getSSPLayout(MFI->getObjectAllocation(i))) {
      case StackProtector::SSPLK_None:
        continue;
      case StackProtector::SSPLK_SmallArray:
        SmallArrayObjs.insert(i);
        continue;
      case StackProtector::SSPLK_AddrOf:
        AddrOfObjs.insert(i);
        continue;
      case StackProtector::SSPLK_LargeArray:
        LargeArrayObjs.insert(i);
        continue;
      }
      llvm_unreachable("Unexpected SSPLayoutKind.");
    }

    AssignProtectedObjSet(LargeArrayObjs, ProtectedObjs, MFI, StackGrowsDown,
                          Offset, MaxAlign);
    AssignProtectedObjSet(SmallArrayObjs, ProtectedObjs, MFI, StackGrowsDown,
                          Offset, MaxAlign);
    AssignProtectedObjSet(AddrOfObjs, ProtectedObjs, MFI, StackGrowsDown,
                          Offset, MaxAlign);
  }

  // Then assign frame offsets to stack objects that are not used to spill
  // callee saved registers.
  for (unsigned i = 0, e = MFI->getObjectIndexEnd(); i != e; ++i) {
    if (MFI->isObjectPreAllocated(i) &&
        MFI->getUseLocalStackAllocationBlock())
      continue;
    if (i >= MinCSFrameIndex && i <= MaxCSFrameIndex)
      continue;
    if (RS && RS->isScavengingFrameIndex((int)i))
      continue;
    if (MFI->isDeadObjectIndex(i))
      continue;
    if (MFI->getStackProtectorIndex() == (int)i)
      continue;
    if (ProtectedObjs.count(i))
      continue;

    AdjustStackOffset(MFI, i, StackGrowsDown, Offset, MaxAlign);
  }

  // Make sure the special register scavenging spill slot is closest to the
  // stack pointer.
  if (RS && !EarlyScavengingSlots) {
    SmallVector<int, 2> SFIs;
    RS->getScavengingFrameIndices(SFIs);
    for (SmallVectorImpl<int>::iterator I = SFIs.begin(),
           IE = SFIs.end(); I != IE; ++I)
      AdjustStackOffset(MFI, *I, StackGrowsDown, Offset, MaxAlign);
  }

  if (!TFI.targetHandlesStackFrameRounding()) {
    // If we have reserved argument space for call sites in the function
    // immediately on entry to the current function, count it as part of the
    // overall stack size.
    if (MFI->adjustsStack() && TFI.hasReservedCallFrame(Fn))
      Offset += MFI->getMaxCallFrameSize();

    // Round up the size to a multiple of the alignment.  If the function has
    // any calls or alloca's, align to the target's StackAlignment value to
    // ensure that the callee's frame or the alloca data is suitably aligned;
    // otherwise, for leaf functions, align to the TransientStackAlignment
    // value.
    unsigned StackAlign;
    if (MFI->adjustsStack() || MFI->hasVarSizedObjects() ||
        (RegInfo->needsStackRealignment(Fn) && MFI->getObjectIndexEnd() != 0))
      StackAlign = TFI.getStackAlignment();
    else
      StackAlign = TFI.getTransientStackAlignment();

    // If the frame pointer is eliminated, all frame offsets will be relative to
    // SP not FP. Align to MaxAlign so this works.
    StackAlign = std::max(StackAlign, MaxAlign);
    Offset = RoundUpToAlignment(Offset, StackAlign);
  }

  // Update frame info to pretend that this is part of the stack...
  int64_t StackSize = Offset - LocalAreaOffset;
  MFI->setStackSize(StackSize);
  NumBytesStackSpace += StackSize;
}

/// insertPrologEpilogCode - Scan the function for modified callee saved
/// registers, insert spill code for these callee saved registers, then add
/// prolog and epilog code to the function.
///
void PEI::insertPrologEpilogCode(MachineFunction &Fn) {
  const TargetFrameLowering &TFI = *Fn.getSubtarget().getFrameLowering();

  // Add prologue to the function...
  TFI.emitPrologue(Fn, *SaveBlock);

  // Add epilogue to restore the callee-save registers in each exiting block.
  for (MachineBasicBlock *RestoreBlock : RestoreBlocks)
    TFI.emitEpilogue(Fn, *RestoreBlock);

  // Emit additional code that is required to support segmented stacks, if
  // we've been asked for it.  This, when linked with a runtime with support
  // for segmented stacks (libgcc is one), will result in allocating stack
  // space in small chunks instead of one large contiguous block.
  if (Fn.shouldSplitStack())
    TFI.adjustForSegmentedStacks(Fn, *SaveBlock);

  // Emit additional code that is required to explicitly handle the stack in
  // HiPE native code (if needed) when loaded in the Erlang/OTP runtime. The
  // approach is rather similar to that of Segmented Stacks, but it uses a
  // different conditional check and another BIF for allocating more stack
  // space.
  if (Fn.getFunction()->getCallingConv() == CallingConv::HiPE)
    TFI.adjustForHiPEPrologue(Fn, *SaveBlock);
}

/// replaceFrameIndices - Replace all MO_FrameIndex operands with physical
/// register references and actual offsets.
///
void PEI::replaceFrameIndices(MachineFunction &Fn) {
  const TargetFrameLowering &TFI = *Fn.getSubtarget().getFrameLowering();
  if (!TFI.needsFrameIndexResolution(Fn)) return;

  MachineModuleInfo &MMI = Fn.getMMI();
  const Function *F = Fn.getFunction();
  const Function *ParentF = MMI.getWinEHParent(F);
  unsigned FrameReg;
  if (F == ParentF) {
    WinEHFuncInfo &FuncInfo = MMI.getWinEHFuncInfo(Fn.getFunction());
    // FIXME: This should be unconditional but we have bugs in the preparation
    // pass.
    if (FuncInfo.UnwindHelpFrameIdx != INT_MAX)
      FuncInfo.UnwindHelpFrameOffset = TFI.getFrameIndexReferenceFromSP(
          Fn, FuncInfo.UnwindHelpFrameIdx, FrameReg);
  } else if (MMI.hasWinEHFuncInfo(F)) {
    WinEHFuncInfo &FuncInfo = MMI.getWinEHFuncInfo(Fn.getFunction());
    auto I = FuncInfo.CatchHandlerParentFrameObjIdx.find(F);
    if (I != FuncInfo.CatchHandlerParentFrameObjIdx.end())
      FuncInfo.CatchHandlerParentFrameObjOffset[F] =
          TFI.getFrameIndexReferenceFromSP(Fn, I->second, FrameReg);
  }

  // Store SPAdj at exit of a basic block.
  SmallVector<int, 8> SPState;
  SPState.resize(Fn.getNumBlockIDs());
  SmallPtrSet<MachineBasicBlock*, 8> Reachable;

  // Iterate over the reachable blocks in DFS order.
  for (auto DFI = df_ext_begin(&Fn, Reachable), DFE = df_ext_end(&Fn, Reachable);
       DFI != DFE; ++DFI) {
    int SPAdj = 0;
    // Check the exit state of the DFS stack predecessor.
    if (DFI.getPathLength() >= 2) {
      MachineBasicBlock *StackPred = DFI.getPath(DFI.getPathLength() - 2);
      assert(Reachable.count(StackPred) &&
             "DFS stack predecessor is already visited.\n");
      SPAdj = SPState[StackPred->getNumber()];
    }
    MachineBasicBlock *BB = *DFI;
    replaceFrameIndices(BB, Fn, SPAdj);
    SPState[BB->getNumber()] = SPAdj;
  }

  // Handle the unreachable blocks.
  for (MachineFunction::iterator BB = Fn.begin(), E = Fn.end(); BB != E; ++BB) {
    if (Reachable.count(BB))
      // Already handled in DFS traversal.
      continue;
    int SPAdj = 0;
    replaceFrameIndices(BB, Fn, SPAdj);
  }
}

void PEI::replaceFrameIndices(MachineBasicBlock *BB, MachineFunction &Fn,
                              int &SPAdj) {
  assert(Fn.getSubtarget().getRegisterInfo() &&
         "getRegisterInfo() must be implemented!");
  const TargetInstrInfo &TII = *Fn.getSubtarget().getInstrInfo();
  const TargetRegisterInfo &TRI = *Fn.getSubtarget().getRegisterInfo();
  const TargetFrameLowering *TFI = Fn.getSubtarget().getFrameLowering();
  unsigned FrameSetupOpcode = TII.getCallFrameSetupOpcode();
  unsigned FrameDestroyOpcode = TII.getCallFrameDestroyOpcode();

  if (RS && !FrameIndexVirtualScavenging) RS->enterBasicBlock(BB);

  bool InsideCallSequence = false;

  for (MachineBasicBlock::iterator I = BB->begin(); I != BB->end(); ) {

    if (I->getOpcode() == FrameSetupOpcode ||
        I->getOpcode() == FrameDestroyOpcode) {
      InsideCallSequence = (I->getOpcode() == FrameSetupOpcode);
      SPAdj += TII.getSPAdjust(I);

      MachineBasicBlock::iterator PrevI = BB->end();
      if (I != BB->begin()) PrevI = std::prev(I);
      TFI->eliminateCallFramePseudoInstr(Fn, *BB, I);

      // Visit the instructions created by eliminateCallFramePseudoInstr().
      if (PrevI == BB->end())
        I = BB->begin();     // The replaced instr was the first in the block.
      else
        I = std::next(PrevI);
      continue;
    }

    MachineInstr *MI = I;
    bool DoIncr = true;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      if (!MI->getOperand(i).isFI())
        continue;

      // Frame indicies in debug values are encoded in a target independent
      // way with simply the frame index and offset rather than any
      // target-specific addressing mode.
      if (MI->isDebugValue()) {
        assert(i == 0 && "Frame indicies can only appear as the first "
                         "operand of a DBG_VALUE machine instruction");
        unsigned Reg;
        MachineOperand &Offset = MI->getOperand(1);
        Offset.setImm(Offset.getImm() +
                      TFI->getFrameIndexReference(
                          Fn, MI->getOperand(0).getIndex(), Reg));
        MI->getOperand(0).ChangeToRegister(Reg, false /*isDef*/);
        continue;
      }

      // TODO: This code should be commoned with the code for
      // PATCHPOINT. There's no good reason for the difference in
      // implementation other than historical accident.  The only
      // remaining difference is the unconditional use of the stack
      // pointer as the base register.
      if (MI->getOpcode() == TargetOpcode::STATEPOINT) {
        assert((!MI->isDebugValue() || i == 0) &&
               "Frame indicies can only appear as the first operand of a "
               "DBG_VALUE machine instruction");
        unsigned Reg;
        MachineOperand &Offset = MI->getOperand(i + 1);
        const unsigned refOffset =
          TFI->getFrameIndexReferenceFromSP(Fn, MI->getOperand(i).getIndex(),
                                            Reg);

        Offset.setImm(Offset.getImm() + refOffset);
        MI->getOperand(i).ChangeToRegister(Reg, false /*isDef*/);
        continue;
      }

      // Some instructions (e.g. inline asm instructions) can have
      // multiple frame indices and/or cause eliminateFrameIndex
      // to insert more than one instruction. We need the register
      // scavenger to go through all of these instructions so that
      // it can update its register information. We keep the
      // iterator at the point before insertion so that we can
      // revisit them in full.
      bool AtBeginning = (I == BB->begin());
      if (!AtBeginning) --I;

      // If this instruction has a FrameIndex operand, we need to
      // use that target machine register info object to eliminate
      // it.
      TRI.eliminateFrameIndex(MI, SPAdj, i,
                              FrameIndexVirtualScavenging ?  nullptr : RS);

      // Reset the iterator if we were at the beginning of the BB.
      if (AtBeginning) {
        I = BB->begin();
        DoIncr = false;
      }

      MI = nullptr;
      break;
    }

    // If we are looking at a call sequence, we need to keep track of
    // the SP adjustment made by each instruction in the sequence.
    // This includes both the frame setup/destroy pseudos (handled above),
    // as well as other instructions that have side effects w.r.t the SP.
    // Note that this must come after eliminateFrameIndex, because 
    // if I itself referred to a frame index, we shouldn't count its own
    // adjustment.
    if (MI && InsideCallSequence)
      SPAdj += TII.getSPAdjust(MI);

    if (DoIncr && I != BB->end()) ++I;

    // Update register states.
    if (RS && !FrameIndexVirtualScavenging && MI) RS->forward(MI);
  }
}

/// scavengeFrameVirtualRegs - Replace all frame index virtual registers
/// with physical registers. Use the register scavenger to find an
/// appropriate register to use.
///
/// FIXME: Iterating over the instruction stream is unnecessary. We can simply
/// iterate over the vreg use list, which at this point only contains machine
/// operands for which eliminateFrameIndex need a new scratch reg.
void
PEI::scavengeFrameVirtualRegs(MachineFunction &Fn) {
  // Run through the instructions and find any virtual registers.
  for (MachineFunction::iterator BB = Fn.begin(),
       E = Fn.end(); BB != E; ++BB) {
    RS->enterBasicBlock(BB);

    int SPAdj = 0;

    // The instruction stream may change in the loop, so check BB->end()
    // directly.
    for (MachineBasicBlock::iterator I = BB->begin(); I != BB->end(); ) {
      // We might end up here again with a NULL iterator if we scavenged a
      // register for which we inserted spill code for definition by what was
      // originally the first instruction in BB.
      if (I == MachineBasicBlock::iterator(nullptr))
        I = BB->begin();

      MachineInstr *MI = I;
      MachineBasicBlock::iterator J = std::next(I);
      MachineBasicBlock::iterator P =
                         I == BB->begin() ? MachineBasicBlock::iterator(nullptr)
                                          : std::prev(I);

      // RS should process this instruction before we might scavenge at this
      // location. This is because we might be replacing a virtual register
      // defined by this instruction, and if so, registers killed by this
      // instruction are available, and defined registers are not.
      RS->forward(I);

      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        if (MI->getOperand(i).isReg()) {
          MachineOperand &MO = MI->getOperand(i);
          unsigned Reg = MO.getReg();
          if (Reg == 0)
            continue;
          if (!TargetRegisterInfo::isVirtualRegister(Reg))
            continue;

          // When we first encounter a new virtual register, it
          // must be a definition.
          assert(MI->getOperand(i).isDef() &&
                 "frame index virtual missing def!");
          // Scavenge a new scratch register
          const TargetRegisterClass *RC = Fn.getRegInfo().getRegClass(Reg);
          unsigned ScratchReg = RS->scavengeRegister(RC, J, SPAdj);

          ++NumScavengedRegs;

          // Replace this reference to the virtual register with the
          // scratch register.
          assert (ScratchReg && "Missing scratch register!");
          MachineRegisterInfo &MRI = Fn.getRegInfo();
          Fn.getRegInfo().replaceRegWith(Reg, ScratchReg);
          
          // Make sure MRI now accounts this register as used.
          MRI.setPhysRegUsed(ScratchReg);

          // Because this instruction was processed by the RS before this
          // register was allocated, make sure that the RS now records the
          // register as being used.
          RS->setRegUsed(ScratchReg);
        }
      }

      // If the scavenger needed to use one of its spill slots, the
      // spill code will have been inserted in between I and J. This is a
      // problem because we need the spill code before I: Move I to just
      // prior to J.
      if (I != std::prev(J)) {
        BB->splice(J, BB, I);

        // Before we move I, we need to prepare the RS to visit I again.
        // Specifically, RS will assert if it sees uses of registers that
        // it believes are undefined. Because we have already processed
        // register kills in I, when it visits I again, it will believe that
        // those registers are undefined. To avoid this situation, unprocess
        // the instruction I.
        assert(RS->getCurrentPosition() == I &&
          "The register scavenger has an unexpected position");
        I = P;
        RS->unprocess(P);
      } else
        ++I;
    }
  }
}
