//===-- llvm/CodeGen/VirtRegMap.cpp - Virtual Register Map ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the VirtRegMap class.
//
// It also contains implementations of the Spiller interface, which, given a
// virtual register map and a machine function, eliminates all virtual
// references by replacing them with physical register references - adding spill
// code as necessary.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/VirtRegMap.h"
#include "LiveDebugVariables.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SparseSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <algorithm>
using namespace llvm;

#define DEBUG_TYPE "regalloc"

STATISTIC(NumSpillSlots, "Number of spill slots allocated");
STATISTIC(NumIdCopies,   "Number of identity moves eliminated after rewriting");

//===----------------------------------------------------------------------===//
//  VirtRegMap implementation
//===----------------------------------------------------------------------===//

char VirtRegMap::ID = 0;

INITIALIZE_PASS(VirtRegMap, "virtregmap", "Virtual Register Map", false, false)

bool VirtRegMap::runOnMachineFunction(MachineFunction &mf) {
  MRI = &mf.getRegInfo();
  TII = mf.getSubtarget().getInstrInfo();
  TRI = mf.getSubtarget().getRegisterInfo();
  MF = &mf;

  Virt2PhysMap.clear();
  Virt2StackSlotMap.clear();
  Virt2SplitMap.clear();

  grow();
  return false;
}

void VirtRegMap::grow() {
  unsigned NumRegs = MF->getRegInfo().getNumVirtRegs();
  Virt2PhysMap.resize(NumRegs);
  Virt2StackSlotMap.resize(NumRegs);
  Virt2SplitMap.resize(NumRegs);
}

unsigned VirtRegMap::createSpillSlot(const TargetRegisterClass *RC) {
  int SS = MF->getFrameInfo()->CreateSpillStackObject(RC->getSize(),
                                                      RC->getAlignment());
  ++NumSpillSlots;
  return SS;
}

bool VirtRegMap::hasPreferredPhys(unsigned VirtReg) {
  unsigned Hint = MRI->getSimpleHint(VirtReg);
  if (!Hint)
    return 0;
  if (TargetRegisterInfo::isVirtualRegister(Hint))
    Hint = getPhys(Hint);
  return getPhys(VirtReg) == Hint;
}

bool VirtRegMap::hasKnownPreference(unsigned VirtReg) {
  std::pair<unsigned, unsigned> Hint = MRI->getRegAllocationHint(VirtReg);
  if (TargetRegisterInfo::isPhysicalRegister(Hint.second))
    return true;
  if (TargetRegisterInfo::isVirtualRegister(Hint.second))
    return hasPhys(Hint.second);
  return false;
}

int VirtRegMap::assignVirt2StackSlot(unsigned virtReg) {
  assert(TargetRegisterInfo::isVirtualRegister(virtReg));
  assert(Virt2StackSlotMap[virtReg] == NO_STACK_SLOT &&
         "attempt to assign stack slot to already spilled register");
  const TargetRegisterClass* RC = MF->getRegInfo().getRegClass(virtReg);
  return Virt2StackSlotMap[virtReg] = createSpillSlot(RC);
}

void VirtRegMap::assignVirt2StackSlot(unsigned virtReg, int SS) {
  assert(TargetRegisterInfo::isVirtualRegister(virtReg));
  assert(Virt2StackSlotMap[virtReg] == NO_STACK_SLOT &&
         "attempt to assign stack slot to already spilled register");
  assert((SS >= 0 ||
          (SS >= MF->getFrameInfo()->getObjectIndexBegin())) &&
         "illegal fixed frame index");
  Virt2StackSlotMap[virtReg] = SS;
}

void VirtRegMap::print(raw_ostream &OS, const Module*) const {
  OS << "********** REGISTER MAP **********\n";
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    if (Virt2PhysMap[Reg] != (unsigned)VirtRegMap::NO_PHYS_REG) {
      OS << '[' << PrintReg(Reg, TRI) << " -> "
         << PrintReg(Virt2PhysMap[Reg], TRI) << "] "
         << TRI->getRegClassName(MRI->getRegClass(Reg)) << "\n";
    }
  }

  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    if (Virt2StackSlotMap[Reg] != VirtRegMap::NO_STACK_SLOT) {
      OS << '[' << PrintReg(Reg, TRI) << " -> fi#" << Virt2StackSlotMap[Reg]
         << "] " << TRI->getRegClassName(MRI->getRegClass(Reg)) << "\n";
    }
  }
  OS << '\n';
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void VirtRegMap::dump() const {
  print(dbgs());
}
#endif

//===----------------------------------------------------------------------===//
//                              VirtRegRewriter
//===----------------------------------------------------------------------===//
//
// The VirtRegRewriter is the last of the register allocator passes.
// It rewrites virtual registers to physical registers as specified in the
// VirtRegMap analysis. It also updates live-in information on basic blocks
// according to LiveIntervals.
//
namespace {
class VirtRegRewriter : public MachineFunctionPass {
  MachineFunction *MF;
  const TargetMachine *TM;
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;
  MachineRegisterInfo *MRI;
  SlotIndexes *Indexes;
  LiveIntervals *LIS;
  VirtRegMap *VRM;
  SparseSet<unsigned> PhysRegs;

  void rewrite();
  void addMBBLiveIns();
  bool readsUndefSubreg(const MachineOperand &MO) const;
public:
  static char ID;
  VirtRegRewriter() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction&) override;
};
} // end anonymous namespace

char &llvm::VirtRegRewriterID = VirtRegRewriter::ID;

INITIALIZE_PASS_BEGIN(VirtRegRewriter, "virtregrewriter",
                      "Virtual Register Rewriter", false, false)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(LiveDebugVariables)
INITIALIZE_PASS_DEPENDENCY(LiveStacks)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_END(VirtRegRewriter, "virtregrewriter",
                    "Virtual Register Rewriter", false, false)

char VirtRegRewriter::ID = 0;

void VirtRegRewriter::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<LiveIntervals>();
  AU.addRequired<SlotIndexes>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequired<LiveDebugVariables>();
  AU.addRequired<LiveStacks>();
  AU.addPreserved<LiveStacks>();
  AU.addRequired<VirtRegMap>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool VirtRegRewriter::runOnMachineFunction(MachineFunction &fn) {
  MF = &fn;
  TM = &MF->getTarget();
  TRI = MF->getSubtarget().getRegisterInfo();
  TII = MF->getSubtarget().getInstrInfo();
  MRI = &MF->getRegInfo();
  Indexes = &getAnalysis<SlotIndexes>();
  LIS = &getAnalysis<LiveIntervals>();
  VRM = &getAnalysis<VirtRegMap>();
  DEBUG(dbgs() << "********** REWRITE VIRTUAL REGISTERS **********\n"
               << "********** Function: "
               << MF->getName() << '\n');
  DEBUG(VRM->dump());

  // Add kill flags while we still have virtual registers.
  LIS->addKillFlags(VRM);

  // Live-in lists on basic blocks are required for physregs.
  addMBBLiveIns();

  // Rewrite virtual registers.
  rewrite();

  // Write out new DBG_VALUE instructions.
  getAnalysis<LiveDebugVariables>().emitDebugValues(VRM);

  // All machine operands and other references to virtual registers have been
  // replaced. Remove the virtual registers and release all the transient data.
  VRM->clearAllVirt();
  MRI->clearVirtRegs();
  return true;
}

// Compute MBB live-in lists from virtual register live ranges and their
// assignments.
void VirtRegRewriter::addMBBLiveIns() {
  SmallVector<MachineBasicBlock*, 16> LiveIn;
  for (unsigned Idx = 0, IdxE = MRI->getNumVirtRegs(); Idx != IdxE; ++Idx) {
    unsigned VirtReg = TargetRegisterInfo::index2VirtReg(Idx);
    if (MRI->reg_nodbg_empty(VirtReg))
      continue;
    LiveInterval &LI = LIS->getInterval(VirtReg);
    if (LI.empty() || LIS->intervalIsInOneMBB(LI))
      continue;
    // This is a virtual register that is live across basic blocks. Its
    // assigned PhysReg must be marked as live-in to those blocks.
    unsigned PhysReg = VRM->getPhys(VirtReg);
    assert(PhysReg != VirtRegMap::NO_PHYS_REG && "Unmapped virtual register.");

    if (LI.hasSubRanges()) {
      for (LiveInterval::SubRange &S : LI.subranges()) {
        for (const auto &Seg : S.segments) {
          if (!Indexes->findLiveInMBBs(Seg.start, Seg.end, LiveIn))
            continue;
          for (MCSubRegIndexIterator SR(PhysReg, TRI); SR.isValid(); ++SR) {
            unsigned SubReg = SR.getSubReg();
            unsigned SubRegIndex = SR.getSubRegIndex();
            unsigned SubRegLaneMask = TRI->getSubRegIndexLaneMask(SubRegIndex);
            if ((SubRegLaneMask & S.LaneMask) == 0)
              continue;
            for (unsigned i = 0, e = LiveIn.size(); i != e; ++i) {
              LiveIn[i]->addLiveIn(SubReg);
            }
          }
          LiveIn.clear();
        }
      }
    } else {
      // Scan the segments of LI.
      for (const auto &Seg : LI.segments) {
        if (!Indexes->findLiveInMBBs(Seg.start, Seg.end, LiveIn))
          continue;
        for (unsigned i = 0, e = LiveIn.size(); i != e; ++i)
          LiveIn[i]->addLiveIn(PhysReg);
        LiveIn.clear();
      }
    }
  }

  // Sort and unique MBB LiveIns as we've not checked if SubReg/PhysReg were in
  // each MBB's LiveIns set before calling addLiveIn on them.
  for (MachineBasicBlock &MBB : *MF)
    MBB.sortUniqueLiveIns();
}

/// Returns true if the given machine operand \p MO only reads undefined lanes.
/// The function only works for use operands with a subregister set.
bool VirtRegRewriter::readsUndefSubreg(const MachineOperand &MO) const {
  // Shortcut if the operand is already marked undef.
  if (MO.isUndef())
    return true;

  unsigned Reg = MO.getReg();
  const LiveInterval &LI = LIS->getInterval(Reg);
  const MachineInstr &MI = *MO.getParent();
  SlotIndex BaseIndex = LIS->getInstructionIndex(&MI);
  // This code is only meant to handle reading undefined subregisters which
  // we couldn't properly detect before.
  assert(LI.liveAt(BaseIndex) &&
         "Reads of completely dead register should be marked undef already");
  unsigned SubRegIdx = MO.getSubReg();
  unsigned UseMask = TRI->getSubRegIndexLaneMask(SubRegIdx);
  // See if any of the relevant subregister liveranges is defined at this point.
  for (const LiveInterval::SubRange &SR : LI.subranges()) {
    if ((SR.LaneMask & UseMask) != 0 && SR.liveAt(BaseIndex))
      return false;
  }
  return true;
}

void VirtRegRewriter::rewrite() {
  bool NoSubRegLiveness = !MRI->subRegLivenessEnabled();
  SmallVector<unsigned, 8> SuperDeads;
  SmallVector<unsigned, 8> SuperDefs;
  SmallVector<unsigned, 8> SuperKills;
  SmallPtrSet<const MachineInstr *, 4> NoReturnInsts;

  // Here we have a SparseSet to hold which PhysRegs are actually encountered
  // in the MF we are about to iterate over so that later when we call
  // setPhysRegUsed, we are only doing it for physRegs that were actually found
  // in the program and not for all of the possible physRegs for the given
  // target architecture. If the target has a lot of physRegs, then for a small
  // program there will be a significant compile time reduction here.
  PhysRegs.clear();
  PhysRegs.setUniverse(TRI->getNumRegs());

  // The function with uwtable should guarantee that the stack unwinder
  // can unwind the stack to the previous frame.  Thus, we can't apply the
  // noreturn optimization if the caller function has uwtable attribute.
  bool HasUWTable = MF->getFunction()->hasFnAttribute(Attribute::UWTable);

  for (MachineFunction::iterator MBBI = MF->begin(), MBBE = MF->end();
       MBBI != MBBE; ++MBBI) {
    DEBUG(MBBI->print(dbgs(), Indexes));
    bool IsExitBB = MBBI->succ_empty();
    for (MachineBasicBlock::instr_iterator
           MII = MBBI->instr_begin(), MIE = MBBI->instr_end(); MII != MIE;) {
      MachineInstr *MI = MII;
      ++MII;

      // Check if this instruction is a call to a noreturn function.  If this
      // is a call to noreturn function and we don't need the stack unwinding
      // functionality (i.e. this function does not have uwtable attribute and
      // the callee function has the nounwind attribute), then we can ignore
      // the definitions set by this instruction.
      if (!HasUWTable && IsExitBB && MI->isCall()) {
        for (MachineInstr::mop_iterator MOI = MI->operands_begin(),
               MOE = MI->operands_end(); MOI != MOE; ++MOI) {
          MachineOperand &MO = *MOI;
          if (!MO.isGlobal())
            continue;
          const Function *Func = dyn_cast<Function>(MO.getGlobal());
          if (!Func || !Func->hasFnAttribute(Attribute::NoReturn) ||
              // We need to keep correct unwind information
              // even if the function will not return, since the
              // runtime may need it.
              !Func->hasFnAttribute(Attribute::NoUnwind))
            continue;
          NoReturnInsts.insert(MI);
          break;
        }
      }

      for (MachineInstr::mop_iterator MOI = MI->operands_begin(),
           MOE = MI->operands_end(); MOI != MOE; ++MOI) {
        MachineOperand &MO = *MOI;

        // Make sure MRI knows about registers clobbered by regmasks.
        if (MO.isRegMask())
          MRI->addPhysRegsUsedFromRegMask(MO.getRegMask());

        // If we encounter a VirtReg or PhysReg then get at the PhysReg and add
        // it to the physreg bitset.  Later we use only the PhysRegs that were
        // actually encountered in the MF to populate the MRI's used physregs.
        if (MO.isReg() && MO.getReg())
          PhysRegs.insert(
              TargetRegisterInfo::isVirtualRegister(MO.getReg()) ?
              VRM->getPhys(MO.getReg()) :
              MO.getReg());

        if (!MO.isReg() || !TargetRegisterInfo::isVirtualRegister(MO.getReg()))
          continue;
        unsigned VirtReg = MO.getReg();
        unsigned PhysReg = VRM->getPhys(VirtReg);
        assert(PhysReg != VirtRegMap::NO_PHYS_REG &&
               "Instruction uses unmapped VirtReg");
        assert(!MRI->isReserved(PhysReg) && "Reserved register assignment");

        // Preserve semantics of sub-register operands.
        unsigned SubReg = MO.getSubReg();
        if (SubReg != 0) {
          if (NoSubRegLiveness) {
            // A virtual register kill refers to the whole register, so we may
            // have to add <imp-use,kill> operands for the super-register.  A
            // partial redef always kills and redefines the super-register.
            if (MO.readsReg() && (MO.isDef() || MO.isKill()))
              SuperKills.push_back(PhysReg);

            if (MO.isDef()) {
              // Also add implicit defs for the super-register.
              if (MO.isDead())
                SuperDeads.push_back(PhysReg);
              else
                SuperDefs.push_back(PhysReg);
            }
          } else {
            if (MO.isUse()) {
              if (readsUndefSubreg(MO))
                // We need to add an <undef> flag if the subregister is
                // completely undefined (and we are not adding super-register
                // defs).
                MO.setIsUndef(true);
            } else if (!MO.isDead()) {
              assert(MO.isDef());
              // Things get tricky when we ran out of lane mask bits and
              // merged multiple lanes into the overflow bit: In this case
              // our subregister liveness tracking isn't precise and we can't
              // know what subregister parts are undefined, fall back to the
              // implicit super-register def then.
              unsigned LaneMask = TRI->getSubRegIndexLaneMask(SubReg);
              if (TargetRegisterInfo::isImpreciseLaneMask(LaneMask))
                SuperDefs.push_back(PhysReg);
            }
          }

          // The <def,undef> flag only makes sense for sub-register defs, and
          // we are substituting a full physreg.  An <imp-use,kill> operand
          // from the SuperKills list will represent the partial read of the
          // super-register.
          if (MO.isDef())
            MO.setIsUndef(false);

          // PhysReg operands cannot have subregister indexes.
          PhysReg = TRI->getSubReg(PhysReg, SubReg);
          assert(PhysReg && "Invalid SubReg for physical register");
          MO.setSubReg(0);
        }
        // Rewrite. Note we could have used MachineOperand::substPhysReg(), but
        // we need the inlining here.
        MO.setReg(PhysReg);
      }

      // Add any missing super-register kills after rewriting the whole
      // instruction.
      while (!SuperKills.empty())
        MI->addRegisterKilled(SuperKills.pop_back_val(), TRI, true);

      while (!SuperDeads.empty())
        MI->addRegisterDead(SuperDeads.pop_back_val(), TRI, true);

      while (!SuperDefs.empty())
        MI->addRegisterDefined(SuperDefs.pop_back_val(), TRI);

      DEBUG(dbgs() << "> " << *MI);

      // Finally, remove any identity copies.
      if (MI->isIdentityCopy()) {
        ++NumIdCopies;
        DEBUG(dbgs() << "Deleting identity copy.\n");
        if (Indexes)
          Indexes->removeMachineInstrFromMaps(MI);
        // It's safe to erase MI because MII has already been incremented.
        MI->eraseFromParent();
      }
    }
  }

  // Tell MRI about physical registers in use.
  if (NoReturnInsts.empty()) {
    for (SparseSet<unsigned>::iterator
        RegI = PhysRegs.begin(), E = PhysRegs.end(); RegI != E; ++RegI)
      if (!MRI->reg_nodbg_empty(*RegI))
        MRI->setPhysRegUsed(*RegI);
  } else {
    for (SparseSet<unsigned>::iterator
        I = PhysRegs.begin(), E = PhysRegs.end(); I != E; ++I) {
      unsigned Reg = *I;
      if (MRI->reg_nodbg_empty(Reg))
        continue;
      // Check if this register has a use that will impact the rest of the
      // code. Uses in debug and noreturn instructions do not impact the
      // generated code.
      for (MachineInstr &It : MRI->reg_nodbg_instructions(Reg)) {
        if (!NoReturnInsts.count(&It)) {
          MRI->setPhysRegUsed(Reg);
          break;
        }
      }
    }
  }
}

