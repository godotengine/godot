//===-- llvm/CodeGen/MachineBasicBlock.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Collect the sequence of machine instructions for a basic block.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <algorithm>
using namespace llvm;

#define DEBUG_TYPE "codegen"

MachineBasicBlock::MachineBasicBlock(MachineFunction &mf, const BasicBlock *bb)
  : BB(bb), Number(-1), xParent(&mf), Alignment(0), IsLandingPad(false),
    AddressTaken(false), CachedMCSymbol(nullptr) {
  Insts.Parent = this;
}

MachineBasicBlock::~MachineBasicBlock() {
}

/// getSymbol - Return the MCSymbol for this basic block.
///
MCSymbol *MachineBasicBlock::getSymbol() const {
  if (!CachedMCSymbol) {
    const MachineFunction *MF = getParent();
    MCContext &Ctx = MF->getContext();
    const char *Prefix = Ctx.getAsmInfo()->getPrivateLabelPrefix();
    CachedMCSymbol = Ctx.getOrCreateSymbol(Twine(Prefix) + "BB" +
                                           Twine(MF->getFunctionNumber()) +
                                           "_" + Twine(getNumber()));
  }

  return CachedMCSymbol;
}


raw_ostream &llvm::operator<<(raw_ostream &OS, const MachineBasicBlock &MBB) {
  MBB.print(OS);
  return OS;
}

/// addNodeToList (MBB) - When an MBB is added to an MF, we need to update the
/// parent pointer of the MBB, the MBB numbering, and any instructions in the
/// MBB to be on the right operand list for registers.
///
/// MBBs start out as #-1. When a MBB is added to a MachineFunction, it
/// gets the next available unique MBB number. If it is removed from a
/// MachineFunction, it goes back to being #-1.
void ilist_traits<MachineBasicBlock>::addNodeToList(MachineBasicBlock *N) {
  MachineFunction &MF = *N->getParent();
  N->Number = MF.addToMBBNumbering(N);

  // Make sure the instructions have their operands in the reginfo lists.
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  for (MachineBasicBlock::instr_iterator
         I = N->instr_begin(), E = N->instr_end(); I != E; ++I)
    I->AddRegOperandsToUseLists(RegInfo);
}

void ilist_traits<MachineBasicBlock>::removeNodeFromList(MachineBasicBlock *N) {
  N->getParent()->removeFromMBBNumbering(N->Number);
  N->Number = -1;
}


/// addNodeToList (MI) - When we add an instruction to a basic block
/// list, we update its parent pointer and add its operands from reg use/def
/// lists if appropriate.
void ilist_traits<MachineInstr>::addNodeToList(MachineInstr *N) {
  assert(!N->getParent() && "machine instruction already in a basic block");
  N->setParent(Parent);

  // Add the instruction's register operands to their corresponding
  // use/def lists.
  MachineFunction *MF = Parent->getParent();
  N->AddRegOperandsToUseLists(MF->getRegInfo());
}

/// removeNodeFromList (MI) - When we remove an instruction from a basic block
/// list, we update its parent pointer and remove its operands from reg use/def
/// lists if appropriate.
void ilist_traits<MachineInstr>::removeNodeFromList(MachineInstr *N) {
  assert(N->getParent() && "machine instruction not in a basic block");

  // Remove from the use/def lists.
  if (MachineFunction *MF = N->getParent()->getParent())
    N->RemoveRegOperandsFromUseLists(MF->getRegInfo());

  N->setParent(nullptr);
}

/// transferNodesFromList (MI) - When moving a range of instructions from one
/// MBB list to another, we need to update the parent pointers and the use/def
/// lists.
void ilist_traits<MachineInstr>::
transferNodesFromList(ilist_traits<MachineInstr> &fromList,
                      ilist_iterator<MachineInstr> first,
                      ilist_iterator<MachineInstr> last) {
  assert(Parent->getParent() == fromList.Parent->getParent() &&
        "MachineInstr parent mismatch!");

  // Splice within the same MBB -> no change.
  if (Parent == fromList.Parent) return;

  // If splicing between two blocks within the same function, just update the
  // parent pointers.
  for (; first != last; ++first)
    first->setParent(Parent);
}

void ilist_traits<MachineInstr>::deleteNode(MachineInstr* MI) {
  assert(!MI->getParent() && "MI is still in a block!");
  Parent->getParent()->DeleteMachineInstr(MI);
}

MachineBasicBlock::iterator MachineBasicBlock::getFirstNonPHI() {
  instr_iterator I = instr_begin(), E = instr_end();
  while (I != E && I->isPHI())
    ++I;
  assert((I == E || !I->isInsideBundle()) &&
         "First non-phi MI cannot be inside a bundle!");
  return I;
}

MachineBasicBlock::iterator
MachineBasicBlock::SkipPHIsAndLabels(MachineBasicBlock::iterator I) {
  iterator E = end();
  while (I != E && (I->isPHI() || I->isPosition() || I->isDebugValue()))
    ++I;
  // FIXME: This needs to change if we wish to bundle labels / dbg_values
  // inside the bundle.
  assert((I == E || !I->isInsideBundle()) &&
         "First non-phi / non-label instruction is inside a bundle!");
  return I;
}

MachineBasicBlock::iterator MachineBasicBlock::getFirstTerminator() {
  iterator B = begin(), E = end(), I = E;
  while (I != B && ((--I)->isTerminator() || I->isDebugValue()))
    ; /*noop */
  while (I != E && !I->isTerminator())
    ++I;
  return I;
}

MachineBasicBlock::instr_iterator MachineBasicBlock::getFirstInstrTerminator() {
  instr_iterator B = instr_begin(), E = instr_end(), I = E;
  while (I != B && ((--I)->isTerminator() || I->isDebugValue()))
    ; /*noop */
  while (I != E && !I->isTerminator())
    ++I;
  return I;
}

MachineBasicBlock::iterator MachineBasicBlock::getFirstNonDebugInstr() {
  // Skip over begin-of-block dbg_value instructions.
  iterator I = begin(), E = end();
  while (I != E && I->isDebugValue())
    ++I;
  return I;
}

MachineBasicBlock::iterator MachineBasicBlock::getLastNonDebugInstr() {
  // Skip over end-of-block dbg_value instructions.
  instr_iterator B = instr_begin(), I = instr_end();
  while (I != B) {
    --I;
    // Return instruction that starts a bundle.
    if (I->isDebugValue() || I->isInsideBundle())
      continue;
    return I;
  }
  // The block is all debug values.
  return end();
}

const MachineBasicBlock *MachineBasicBlock::getLandingPadSuccessor() const {
  // A block with a landing pad successor only has one other successor.
  if (succ_size() > 2)
    return nullptr;
  for (const_succ_iterator I = succ_begin(), E = succ_end(); I != E; ++I)
    if ((*I)->isLandingPad())
      return *I;
  return nullptr;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void MachineBasicBlock::dump() const {
  print(dbgs());
}
#endif

StringRef MachineBasicBlock::getName() const {
  if (const BasicBlock *LBB = getBasicBlock())
    return LBB->getName();
  else
    return "(null)";
}

/// Return a hopefully unique identifier for this block.
std::string MachineBasicBlock::getFullName() const {
  std::string Name;
  if (getParent())
    Name = (getParent()->getName() + ":").str();
  if (getBasicBlock())
    Name += getBasicBlock()->getName();
  else
    Name += ("BB" + Twine(getNumber())).str();
  return Name;
}

void MachineBasicBlock::print(raw_ostream &OS, SlotIndexes *Indexes) const {
  const MachineFunction *MF = getParent();
  if (!MF) {
    OS << "Can't print out MachineBasicBlock because parent MachineFunction"
       << " is null\n";
    return;
  }
  const Function *F = MF->getFunction();
  const Module *M = F ? F->getParent() : nullptr;
  ModuleSlotTracker MST(M);
  print(OS, MST, Indexes);
}

void MachineBasicBlock::print(raw_ostream &OS, ModuleSlotTracker &MST,
                              SlotIndexes *Indexes) const {
  const MachineFunction *MF = getParent();
  if (!MF) {
    OS << "Can't print out MachineBasicBlock because parent MachineFunction"
       << " is null\n";
    return;
  }

  if (Indexes)
    OS << Indexes->getMBBStartIdx(this) << '\t';

  OS << "BB#" << getNumber() << ": ";

  const char *Comma = "";
  if (const BasicBlock *LBB = getBasicBlock()) {
    OS << Comma << "derived from LLVM BB ";
    LBB->printAsOperand(OS, /*PrintType=*/false, MST);
    Comma = ", ";
  }
  if (isLandingPad()) { OS << Comma << "EH LANDING PAD"; Comma = ", "; }
  if (hasAddressTaken()) { OS << Comma << "ADDRESS TAKEN"; Comma = ", "; }
  if (Alignment)
    OS << Comma << "Align " << Alignment << " (" << (1u << Alignment)
       << " bytes)";

  OS << '\n';

  const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
  if (!livein_empty()) {
    if (Indexes) OS << '\t';
    OS << "    Live Ins:";
    for (livein_iterator I = livein_begin(),E = livein_end(); I != E; ++I)
      OS << ' ' << PrintReg(*I, TRI);
    OS << '\n';
  }
  // Print the preds of this block according to the CFG.
  if (!pred_empty()) {
    if (Indexes) OS << '\t';
    OS << "    Predecessors according to CFG:";
    for (const_pred_iterator PI = pred_begin(), E = pred_end(); PI != E; ++PI)
      OS << " BB#" << (*PI)->getNumber();
    OS << '\n';
  }

  for (const_instr_iterator I = instr_begin(); I != instr_end(); ++I) {
    if (Indexes) {
      if (Indexes->hasIndex(I))
        OS << Indexes->getInstructionIndex(I);
      OS << '\t';
    }
    OS << '\t';
    if (I->isInsideBundle())
      OS << "  * ";
    I->print(OS, MST);
  }

  // Print the successors of this block according to the CFG.
  if (!succ_empty()) {
    if (Indexes) OS << '\t';
    OS << "    Successors according to CFG:";
    for (const_succ_iterator SI = succ_begin(), E = succ_end(); SI != E; ++SI) {
      OS << " BB#" << (*SI)->getNumber();
      if (!Weights.empty())
        OS << '(' << *getWeightIterator(SI) << ')';
    }
    OS << '\n';
  }
}

void MachineBasicBlock::printAsOperand(raw_ostream &OS, bool /*PrintType*/) const {
  OS << "BB#" << getNumber();
}

void MachineBasicBlock::removeLiveIn(unsigned Reg) {
  std::vector<unsigned>::iterator I =
    std::find(LiveIns.begin(), LiveIns.end(), Reg);
  if (I != LiveIns.end())
    LiveIns.erase(I);
}

bool MachineBasicBlock::isLiveIn(unsigned Reg) const {
  livein_iterator I = std::find(livein_begin(), livein_end(), Reg);
  return I != livein_end();
}

unsigned
MachineBasicBlock::addLiveIn(unsigned PhysReg, const TargetRegisterClass *RC) {
  assert(getParent() && "MBB must be inserted in function");
  assert(TargetRegisterInfo::isPhysicalRegister(PhysReg) && "Expected physreg");
  assert(RC && "Register class is required");
  assert((isLandingPad() || this == &getParent()->front()) &&
         "Only the entry block and landing pads can have physreg live ins");

  bool LiveIn = isLiveIn(PhysReg);
  iterator I = SkipPHIsAndLabels(begin()), E = end();
  MachineRegisterInfo &MRI = getParent()->getRegInfo();
  const TargetInstrInfo &TII = *getParent()->getSubtarget().getInstrInfo();

  // Look for an existing copy.
  if (LiveIn)
    for (;I != E && I->isCopy(); ++I)
      if (I->getOperand(1).getReg() == PhysReg) {
        unsigned VirtReg = I->getOperand(0).getReg();
        if (!MRI.constrainRegClass(VirtReg, RC))
          llvm_unreachable("Incompatible live-in register class.");
        return VirtReg;
      }

  // No luck, create a virtual register.
  unsigned VirtReg = MRI.createVirtualRegister(RC);
  BuildMI(*this, I, DebugLoc(), TII.get(TargetOpcode::COPY), VirtReg)
    .addReg(PhysReg, RegState::Kill);
  if (!LiveIn)
    addLiveIn(PhysReg);
  return VirtReg;
}

void MachineBasicBlock::moveBefore(MachineBasicBlock *NewAfter) {
  getParent()->splice(NewAfter, this);
}

void MachineBasicBlock::moveAfter(MachineBasicBlock *NewBefore) {
  MachineFunction::iterator BBI = NewBefore;
  getParent()->splice(++BBI, this);
}

void MachineBasicBlock::updateTerminator() {
  const TargetInstrInfo *TII = getParent()->getSubtarget().getInstrInfo();
  // A block with no successors has no concerns with fall-through edges.
  if (this->succ_empty()) return;

  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;
  DebugLoc dl;  // FIXME: this is nowhere
  bool B = TII->AnalyzeBranch(*this, TBB, FBB, Cond);
  (void) B;
  assert(!B && "UpdateTerminators requires analyzable predecessors!");
  if (Cond.empty()) {
    if (TBB) {
      // The block has an unconditional branch. If its successor is now
      // its layout successor, delete the branch.
      if (isLayoutSuccessor(TBB))
        TII->RemoveBranch(*this);
    } else {
      // The block has an unconditional fallthrough. If its successor is not
      // its layout successor, insert a branch. First we have to locate the
      // only non-landing-pad successor, as that is the fallthrough block.
      for (succ_iterator SI = succ_begin(), SE = succ_end(); SI != SE; ++SI) {
        if ((*SI)->isLandingPad())
          continue;
        assert(!TBB && "Found more than one non-landing-pad successor!");
        TBB = *SI;
      }

      // If there is no non-landing-pad successor, the block has no
      // fall-through edges to be concerned with.
      if (!TBB)
        return;

      // Finally update the unconditional successor to be reached via a branch
      // if it would not be reached by fallthrough.
      if (!isLayoutSuccessor(TBB))
        TII->InsertBranch(*this, TBB, nullptr, Cond, dl);
    }
  } else {
    if (FBB) {
      // The block has a non-fallthrough conditional branch. If one of its
      // successors is its layout successor, rewrite it to a fallthrough
      // conditional branch.
      if (isLayoutSuccessor(TBB)) {
        if (TII->ReverseBranchCondition(Cond))
          return;
        TII->RemoveBranch(*this);
        TII->InsertBranch(*this, FBB, nullptr, Cond, dl);
      } else if (isLayoutSuccessor(FBB)) {
        TII->RemoveBranch(*this);
        TII->InsertBranch(*this, TBB, nullptr, Cond, dl);
      }
    } else {
      // Walk through the successors and find the successor which is not
      // a landing pad and is not the conditional branch destination (in TBB)
      // as the fallthrough successor.
      MachineBasicBlock *FallthroughBB = nullptr;
      for (succ_iterator SI = succ_begin(), SE = succ_end(); SI != SE; ++SI) {
        if ((*SI)->isLandingPad() || *SI == TBB)
          continue;
        assert(!FallthroughBB && "Found more than one fallthrough successor.");
        FallthroughBB = *SI;
      }
      if (!FallthroughBB && canFallThrough()) {
        // We fallthrough to the same basic block as the conditional jump
        // targets. Remove the conditional jump, leaving unconditional
        // fallthrough.
        // FIXME: This does not seem like a reasonable pattern to support, but it
        // has been seen in the wild coming out of degenerate ARM test cases.
        TII->RemoveBranch(*this);

        // Finally update the unconditional successor to be reached via a branch
        // if it would not be reached by fallthrough.
        if (!isLayoutSuccessor(TBB))
          TII->InsertBranch(*this, TBB, nullptr, Cond, dl);
        return;
      }

      // The block has a fallthrough conditional branch.
      if (isLayoutSuccessor(TBB)) {
        if (TII->ReverseBranchCondition(Cond)) {
          // We can't reverse the condition, add an unconditional branch.
          Cond.clear();
          TII->InsertBranch(*this, FallthroughBB, nullptr, Cond, dl);
          return;
        }
        TII->RemoveBranch(*this);
        TII->InsertBranch(*this, FallthroughBB, nullptr, Cond, dl);
      } else if (!isLayoutSuccessor(FallthroughBB)) {
        TII->RemoveBranch(*this);
        TII->InsertBranch(*this, TBB, FallthroughBB, Cond, dl);
      }
    }
  }
}

void MachineBasicBlock::addSuccessor(MachineBasicBlock *succ, uint32_t weight) {

  // If we see non-zero value for the first time it means we actually use Weight
  // list, so we fill all Weights with 0's.
  if (weight != 0 && Weights.empty())
    Weights.resize(Successors.size());

  if (weight != 0 || !Weights.empty())
    Weights.push_back(weight);

   Successors.push_back(succ);
   succ->addPredecessor(this);
 }

void MachineBasicBlock::removeSuccessor(MachineBasicBlock *succ) {
  succ->removePredecessor(this);
  succ_iterator I = std::find(Successors.begin(), Successors.end(), succ);
  assert(I != Successors.end() && "Not a current successor!");

  // If Weight list is empty it means we don't use it (disabled optimization).
  if (!Weights.empty()) {
    weight_iterator WI = getWeightIterator(I);
    Weights.erase(WI);
  }

  Successors.erase(I);
}

MachineBasicBlock::succ_iterator
MachineBasicBlock::removeSuccessor(succ_iterator I) {
  assert(I != Successors.end() && "Not a current successor!");

  // If Weight list is empty it means we don't use it (disabled optimization).
  if (!Weights.empty()) {
    weight_iterator WI = getWeightIterator(I);
    Weights.erase(WI);
  }

  (*I)->removePredecessor(this);
  return Successors.erase(I);
}

void MachineBasicBlock::replaceSuccessor(MachineBasicBlock *Old,
                                         MachineBasicBlock *New) {
  if (Old == New)
    return;

  succ_iterator E = succ_end();
  succ_iterator NewI = E;
  succ_iterator OldI = E;
  for (succ_iterator I = succ_begin(); I != E; ++I) {
    if (*I == Old) {
      OldI = I;
      if (NewI != E)
        break;
    }
    if (*I == New) {
      NewI = I;
      if (OldI != E)
        break;
    }
  }
  assert(OldI != E && "Old is not a successor of this block");
  Old->removePredecessor(this);

  // If New isn't already a successor, let it take Old's place.
  if (NewI == E) {
    New->addPredecessor(this);
    *OldI = New;
    return;
  }

  // New is already a successor.
  // Update its weight instead of adding a duplicate edge.
  if (!Weights.empty()) {
    weight_iterator OldWI = getWeightIterator(OldI);
    *getWeightIterator(NewI) += *OldWI;
    Weights.erase(OldWI);
  }
  Successors.erase(OldI);
}

void MachineBasicBlock::addPredecessor(MachineBasicBlock *pred) {
  Predecessors.push_back(pred);
}

void MachineBasicBlock::removePredecessor(MachineBasicBlock *pred) {
  pred_iterator I = std::find(Predecessors.begin(), Predecessors.end(), pred);
  assert(I != Predecessors.end() && "Pred is not a predecessor of this block!");
  Predecessors.erase(I);
}

void MachineBasicBlock::transferSuccessors(MachineBasicBlock *fromMBB) {
  if (this == fromMBB)
    return;

  while (!fromMBB->succ_empty()) {
    MachineBasicBlock *Succ = *fromMBB->succ_begin();
    uint32_t Weight = 0;

    // If Weight list is empty it means we don't use it (disabled optimization).
    if (!fromMBB->Weights.empty())
      Weight = *fromMBB->Weights.begin();

    addSuccessor(Succ, Weight);
    fromMBB->removeSuccessor(Succ);
  }
}

void
MachineBasicBlock::transferSuccessorsAndUpdatePHIs(MachineBasicBlock *fromMBB) {
  if (this == fromMBB)
    return;

  while (!fromMBB->succ_empty()) {
    MachineBasicBlock *Succ = *fromMBB->succ_begin();
    uint32_t Weight = 0;
    if (!fromMBB->Weights.empty())
      Weight = *fromMBB->Weights.begin();
    addSuccessor(Succ, Weight);
    fromMBB->removeSuccessor(Succ);

    // Fix up any PHI nodes in the successor.
    for (MachineBasicBlock::instr_iterator MI = Succ->instr_begin(),
           ME = Succ->instr_end(); MI != ME && MI->isPHI(); ++MI)
      for (unsigned i = 2, e = MI->getNumOperands()+1; i != e; i += 2) {
        MachineOperand &MO = MI->getOperand(i);
        if (MO.getMBB() == fromMBB)
          MO.setMBB(this);
      }
  }
}

bool MachineBasicBlock::isPredecessor(const MachineBasicBlock *MBB) const {
  return std::find(pred_begin(), pred_end(), MBB) != pred_end();
}

bool MachineBasicBlock::isSuccessor(const MachineBasicBlock *MBB) const {
  return std::find(succ_begin(), succ_end(), MBB) != succ_end();
}

bool MachineBasicBlock::isLayoutSuccessor(const MachineBasicBlock *MBB) const {
  MachineFunction::const_iterator I(this);
  return std::next(I) == MachineFunction::const_iterator(MBB);
}

bool MachineBasicBlock::canFallThrough() {
  MachineFunction::iterator Fallthrough = this;
  ++Fallthrough;
  // If FallthroughBlock is off the end of the function, it can't fall through.
  if (Fallthrough == getParent()->end())
    return false;

  // If FallthroughBlock isn't a successor, no fallthrough is possible.
  if (!isSuccessor(Fallthrough))
    return false;

  // Analyze the branches, if any, at the end of the block.
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;
  const TargetInstrInfo *TII = getParent()->getSubtarget().getInstrInfo();
  if (TII->AnalyzeBranch(*this, TBB, FBB, Cond)) {
    // If we couldn't analyze the branch, examine the last instruction.
    // If the block doesn't end in a known control barrier, assume fallthrough
    // is possible. The isPredicated check is needed because this code can be
    // called during IfConversion, where an instruction which is normally a
    // Barrier is predicated and thus no longer an actual control barrier.
    return empty() || !back().isBarrier() || TII->isPredicated(&back());
  }

  // If there is no branch, control always falls through.
  if (!TBB) return true;

  // If there is some explicit branch to the fallthrough block, it can obviously
  // reach, even though the branch should get folded to fall through implicitly.
  if (MachineFunction::iterator(TBB) == Fallthrough ||
      MachineFunction::iterator(FBB) == Fallthrough)
    return true;

  // If it's an unconditional branch to some block not the fall through, it
  // doesn't fall through.
  if (Cond.empty()) return false;

  // Otherwise, if it is conditional and has no explicit false block, it falls
  // through.
  return FBB == nullptr;
}

MachineBasicBlock *
MachineBasicBlock::SplitCriticalEdge(MachineBasicBlock *Succ, Pass *P) {
  // Splitting the critical edge to a landing pad block is non-trivial. Don't do
  // it in this generic function.
  if (Succ->isLandingPad())
    return nullptr;

  MachineFunction *MF = getParent();
  DebugLoc dl;  // FIXME: this is nowhere

  // Performance might be harmed on HW that implements branching using exec mask
  // where both sides of the branches are always executed.
  if (MF->getTarget().requiresStructuredCFG())
    return nullptr;

  // We may need to update this's terminator, but we can't do that if
  // AnalyzeBranch fails. If this uses a jump table, we won't touch it.
  const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;
  if (TII->AnalyzeBranch(*this, TBB, FBB, Cond))
    return nullptr;

  // Avoid bugpoint weirdness: A block may end with a conditional branch but
  // jumps to the same MBB is either case. We have duplicate CFG edges in that
  // case that we can't handle. Since this never happens in properly optimized
  // code, just skip those edges.
  if (TBB && TBB == FBB) {
    DEBUG(dbgs() << "Won't split critical edge after degenerate BB#"
                 << getNumber() << '\n');
    return nullptr;
  }

  MachineBasicBlock *NMBB = MF->CreateMachineBasicBlock();
  MF->insert(std::next(MachineFunction::iterator(this)), NMBB);
  DEBUG(dbgs() << "Splitting critical edge:"
        " BB#" << getNumber()
        << " -- BB#" << NMBB->getNumber()
        << " -- BB#" << Succ->getNumber() << '\n');

  LiveIntervals *LIS = P->getAnalysisIfAvailable<LiveIntervals>();
  SlotIndexes *Indexes = P->getAnalysisIfAvailable<SlotIndexes>();
  if (LIS)
    LIS->insertMBBInMaps(NMBB);
  else if (Indexes)
    Indexes->insertMBBInMaps(NMBB);

  // On some targets like Mips, branches may kill virtual registers. Make sure
  // that LiveVariables is properly updated after updateTerminator replaces the
  // terminators.
  LiveVariables *LV = P->getAnalysisIfAvailable<LiveVariables>();

  // Collect a list of virtual registers killed by the terminators.
  SmallVector<unsigned, 4> KilledRegs;
  if (LV)
    for (instr_iterator I = getFirstInstrTerminator(), E = instr_end();
         I != E; ++I) {
      MachineInstr *MI = I;
      for (MachineInstr::mop_iterator OI = MI->operands_begin(),
           OE = MI->operands_end(); OI != OE; ++OI) {
        if (!OI->isReg() || OI->getReg() == 0 ||
            !OI->isUse() || !OI->isKill() || OI->isUndef())
          continue;
        unsigned Reg = OI->getReg();
        if (TargetRegisterInfo::isPhysicalRegister(Reg) ||
            LV->getVarInfo(Reg).removeKill(MI)) {
          KilledRegs.push_back(Reg);
          DEBUG(dbgs() << "Removing terminator kill: " << *MI);
          OI->setIsKill(false);
        }
      }
    }

  SmallVector<unsigned, 4> UsedRegs;
  if (LIS) {
    for (instr_iterator I = getFirstInstrTerminator(), E = instr_end();
         I != E; ++I) {
      MachineInstr *MI = I;

      for (MachineInstr::mop_iterator OI = MI->operands_begin(),
           OE = MI->operands_end(); OI != OE; ++OI) {
        if (!OI->isReg() || OI->getReg() == 0)
          continue;

        unsigned Reg = OI->getReg();
        if (std::find(UsedRegs.begin(), UsedRegs.end(), Reg) == UsedRegs.end())
          UsedRegs.push_back(Reg);
      }
    }
  }

  ReplaceUsesOfBlockWith(Succ, NMBB);

  // If updateTerminator() removes instructions, we need to remove them from
  // SlotIndexes.
  SmallVector<MachineInstr*, 4> Terminators;
  if (Indexes) {
    for (instr_iterator I = getFirstInstrTerminator(), E = instr_end();
         I != E; ++I)
      Terminators.push_back(I);
  }

  updateTerminator();

  if (Indexes) {
    SmallVector<MachineInstr*, 4> NewTerminators;
    for (instr_iterator I = getFirstInstrTerminator(), E = instr_end();
         I != E; ++I)
      NewTerminators.push_back(I);

    for (SmallVectorImpl<MachineInstr*>::iterator I = Terminators.begin(),
        E = Terminators.end(); I != E; ++I) {
      if (std::find(NewTerminators.begin(), NewTerminators.end(), *I) ==
          NewTerminators.end())
       Indexes->removeMachineInstrFromMaps(*I);
    }
  }

  // Insert unconditional "jump Succ" instruction in NMBB if necessary.
  NMBB->addSuccessor(Succ);
  if (!NMBB->isLayoutSuccessor(Succ)) {
    Cond.clear();
    MF->getSubtarget().getInstrInfo()->InsertBranch(*NMBB, Succ, nullptr, Cond,
                                                    dl);

    if (Indexes) {
      for (instr_iterator I = NMBB->instr_begin(), E = NMBB->instr_end();
           I != E; ++I) {
        // Some instructions may have been moved to NMBB by updateTerminator(),
        // so we first remove any instruction that already has an index.
        if (Indexes->hasIndex(I))
          Indexes->removeMachineInstrFromMaps(I);
        Indexes->insertMachineInstrInMaps(I);
      }
    }
  }

  // Fix PHI nodes in Succ so they refer to NMBB instead of this
  for (MachineBasicBlock::instr_iterator
         i = Succ->instr_begin(),e = Succ->instr_end();
       i != e && i->isPHI(); ++i)
    for (unsigned ni = 1, ne = i->getNumOperands(); ni != ne; ni += 2)
      if (i->getOperand(ni+1).getMBB() == this)
        i->getOperand(ni+1).setMBB(NMBB);

  // Inherit live-ins from the successor
  for (MachineBasicBlock::livein_iterator I = Succ->livein_begin(),
         E = Succ->livein_end(); I != E; ++I)
    NMBB->addLiveIn(*I);

  // Update LiveVariables.
  const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
  if (LV) {
    // Restore kills of virtual registers that were killed by the terminators.
    while (!KilledRegs.empty()) {
      unsigned Reg = KilledRegs.pop_back_val();
      for (instr_iterator I = instr_end(), E = instr_begin(); I != E;) {
        if (!(--I)->addRegisterKilled(Reg, TRI, /* addIfNotFound= */ false))
          continue;
        if (TargetRegisterInfo::isVirtualRegister(Reg))
          LV->getVarInfo(Reg).Kills.push_back(I);
        DEBUG(dbgs() << "Restored terminator kill: " << *I);
        break;
      }
    }
    // Update relevant live-through information.
    LV->addNewBlock(NMBB, this, Succ);
  }

  if (LIS) {
    // After splitting the edge and updating SlotIndexes, live intervals may be
    // in one of two situations, depending on whether this block was the last in
    // the function. If the original block was the last in the function, all live
    // intervals will end prior to the beginning of the new split block. If the
    // original block was not at the end of the function, all live intervals will
    // extend to the end of the new split block.

    bool isLastMBB =
      std::next(MachineFunction::iterator(NMBB)) == getParent()->end();

    SlotIndex StartIndex = Indexes->getMBBEndIdx(this);
    SlotIndex PrevIndex = StartIndex.getPrevSlot();
    SlotIndex EndIndex = Indexes->getMBBEndIdx(NMBB);

    // Find the registers used from NMBB in PHIs in Succ.
    SmallSet<unsigned, 8> PHISrcRegs;
    for (MachineBasicBlock::instr_iterator
         I = Succ->instr_begin(), E = Succ->instr_end();
         I != E && I->isPHI(); ++I) {
      for (unsigned ni = 1, ne = I->getNumOperands(); ni != ne; ni += 2) {
        if (I->getOperand(ni+1).getMBB() == NMBB) {
          MachineOperand &MO = I->getOperand(ni);
          unsigned Reg = MO.getReg();
          PHISrcRegs.insert(Reg);
          if (MO.isUndef())
            continue;

          LiveInterval &LI = LIS->getInterval(Reg);
          VNInfo *VNI = LI.getVNInfoAt(PrevIndex);
          assert(VNI && "PHI sources should be live out of their predecessors.");
          LI.addSegment(LiveInterval::Segment(StartIndex, EndIndex, VNI));
        }
      }
    }

    MachineRegisterInfo *MRI = &getParent()->getRegInfo();
    for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
      unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
      if (PHISrcRegs.count(Reg) || !LIS->hasInterval(Reg))
        continue;

      LiveInterval &LI = LIS->getInterval(Reg);
      if (!LI.liveAt(PrevIndex))
        continue;

      bool isLiveOut = LI.liveAt(LIS->getMBBStartIdx(Succ));
      if (isLiveOut && isLastMBB) {
        VNInfo *VNI = LI.getVNInfoAt(PrevIndex);
        assert(VNI && "LiveInterval should have VNInfo where it is live.");
        LI.addSegment(LiveInterval::Segment(StartIndex, EndIndex, VNI));
      } else if (!isLiveOut && !isLastMBB) {
        LI.removeSegment(StartIndex, EndIndex);
      }
    }

    // Update all intervals for registers whose uses may have been modified by
    // updateTerminator().
    LIS->repairIntervalsInRange(this, getFirstTerminator(), end(), UsedRegs);
  }

  if (MachineDominatorTree *MDT =
      P->getAnalysisIfAvailable<MachineDominatorTree>())
    MDT->recordSplitCriticalEdge(this, Succ, NMBB);

  if (MachineLoopInfo *MLI = P->getAnalysisIfAvailable<MachineLoopInfo>())
    if (MachineLoop *TIL = MLI->getLoopFor(this)) {
      // If one or the other blocks were not in a loop, the new block is not
      // either, and thus LI doesn't need to be updated.
      if (MachineLoop *DestLoop = MLI->getLoopFor(Succ)) {
        if (TIL == DestLoop) {
          // Both in the same loop, the NMBB joins loop.
          DestLoop->addBasicBlockToLoop(NMBB, MLI->getBase());
        } else if (TIL->contains(DestLoop)) {
          // Edge from an outer loop to an inner loop.  Add to the outer loop.
          TIL->addBasicBlockToLoop(NMBB, MLI->getBase());
        } else if (DestLoop->contains(TIL)) {
          // Edge from an inner loop to an outer loop.  Add to the outer loop.
          DestLoop->addBasicBlockToLoop(NMBB, MLI->getBase());
        } else {
          // Edge from two loops with no containment relation.  Because these
          // are natural loops, we know that the destination block must be the
          // header of its loop (adding a branch into a loop elsewhere would
          // create an irreducible loop).
          assert(DestLoop->getHeader() == Succ &&
                 "Should not create irreducible loops!");
          if (MachineLoop *P = DestLoop->getParentLoop())
            P->addBasicBlockToLoop(NMBB, MLI->getBase());
        }
      }
    }

  return NMBB;
}

/// Prepare MI to be removed from its bundle. This fixes bundle flags on MI's
/// neighboring instructions so the bundle won't be broken by removing MI.
static void unbundleSingleMI(MachineInstr *MI) {
  // Removing the first instruction in a bundle.
  if (MI->isBundledWithSucc() && !MI->isBundledWithPred())
    MI->unbundleFromSucc();
  // Removing the last instruction in a bundle.
  if (MI->isBundledWithPred() && !MI->isBundledWithSucc())
    MI->unbundleFromPred();
  // If MI is not bundled, or if it is internal to a bundle, the neighbor flags
  // are already fine.
}

MachineBasicBlock::instr_iterator
MachineBasicBlock::erase(MachineBasicBlock::instr_iterator I) {
  unbundleSingleMI(I);
  return Insts.erase(I);
}

MachineInstr *MachineBasicBlock::remove_instr(MachineInstr *MI) {
  unbundleSingleMI(MI);
  MI->clearFlag(MachineInstr::BundledPred);
  MI->clearFlag(MachineInstr::BundledSucc);
  return Insts.remove(MI);
}

MachineBasicBlock::instr_iterator
MachineBasicBlock::insert(instr_iterator I, MachineInstr *MI) {
  assert(!MI->isBundledWithPred() && !MI->isBundledWithSucc() &&
         "Cannot insert instruction with bundle flags");
  // Set the bundle flags when inserting inside a bundle.
  if (I != instr_end() && I->isBundledWithPred()) {
    MI->setFlag(MachineInstr::BundledPred);
    MI->setFlag(MachineInstr::BundledSucc);
  }
  return Insts.insert(I, MI);
}

/// removeFromParent - This method unlinks 'this' from the containing function,
/// and returns it, but does not delete it.
MachineBasicBlock *MachineBasicBlock::removeFromParent() {
  assert(getParent() && "Not embedded in a function!");
  getParent()->remove(this);
  return this;
}


/// eraseFromParent - This method unlinks 'this' from the containing function,
/// and deletes it.
void MachineBasicBlock::eraseFromParent() {
  assert(getParent() && "Not embedded in a function!");
  getParent()->erase(this);
}


/// ReplaceUsesOfBlockWith - Given a machine basic block that branched to
/// 'Old', change the code and CFG so that it branches to 'New' instead.
void MachineBasicBlock::ReplaceUsesOfBlockWith(MachineBasicBlock *Old,
                                               MachineBasicBlock *New) {
  assert(Old != New && "Cannot replace self with self!");

  MachineBasicBlock::instr_iterator I = instr_end();
  while (I != instr_begin()) {
    --I;
    if (!I->isTerminator()) break;

    // Scan the operands of this machine instruction, replacing any uses of Old
    // with New.
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
      if (I->getOperand(i).isMBB() &&
          I->getOperand(i).getMBB() == Old)
        I->getOperand(i).setMBB(New);
  }

  // Update the successor information.
  replaceSuccessor(Old, New);
}

/// CorrectExtraCFGEdges - Various pieces of code can cause excess edges in the
/// CFG to be inserted.  If we have proven that MBB can only branch to DestA and
/// DestB, remove any other MBB successors from the CFG.  DestA and DestB can be
/// null.
///
/// Besides DestA and DestB, retain other edges leading to LandingPads
/// (currently there can be only one; we don't check or require that here).
/// Note it is possible that DestA and/or DestB are LandingPads.
bool MachineBasicBlock::CorrectExtraCFGEdges(MachineBasicBlock *DestA,
                                             MachineBasicBlock *DestB,
                                             bool isCond) {
  // The values of DestA and DestB frequently come from a call to the
  // 'TargetInstrInfo::AnalyzeBranch' method. We take our meaning of the initial
  // values from there.
  //
  // 1. If both DestA and DestB are null, then the block ends with no branches
  //    (it falls through to its successor).
  // 2. If DestA is set, DestB is null, and isCond is false, then the block ends
  //    with only an unconditional branch.
  // 3. If DestA is set, DestB is null, and isCond is true, then the block ends
  //    with a conditional branch that falls through to a successor (DestB).
  // 4. If DestA and DestB is set and isCond is true, then the block ends with a
  //    conditional branch followed by an unconditional branch. DestA is the
  //    'true' destination and DestB is the 'false' destination.

  bool Changed = false;

  MachineFunction::iterator FallThru =
    std::next(MachineFunction::iterator(this));

  if (!DestA && !DestB) {
    // Block falls through to successor.
    DestA = FallThru;
    DestB = FallThru;
  } else if (DestA && !DestB) {
    if (isCond)
      // Block ends in conditional jump that falls through to successor.
      DestB = FallThru;
  } else {
    assert(DestA && DestB && isCond &&
           "CFG in a bad state. Cannot correct CFG edges");
  }

  // Remove superfluous edges. I.e., those which aren't destinations of this
  // basic block, duplicate edges, or landing pads.
  SmallPtrSet<const MachineBasicBlock*, 8> SeenMBBs;
  MachineBasicBlock::succ_iterator SI = succ_begin();
  while (SI != succ_end()) {
    const MachineBasicBlock *MBB = *SI;
    if (!SeenMBBs.insert(MBB).second ||
        (MBB != DestA && MBB != DestB && !MBB->isLandingPad())) {
      // This is a superfluous edge, remove it.
      SI = removeSuccessor(SI);
      Changed = true;
    } else {
      ++SI;
    }
  }

  return Changed;
}

/// findDebugLoc - find the next valid DebugLoc starting at MBBI, skipping
/// any DBG_VALUE instructions.  Return UnknownLoc if there is none.
DebugLoc
MachineBasicBlock::findDebugLoc(instr_iterator MBBI) {
  DebugLoc DL;
  instr_iterator E = instr_end();
  if (MBBI == E)
    return DL;

  // Skip debug declarations, we don't want a DebugLoc from them.
  while (MBBI != E && MBBI->isDebugValue())
    MBBI++;
  if (MBBI != E)
    DL = MBBI->getDebugLoc();
  return DL;
}

/// getSuccWeight - Return weight of the edge from this block to MBB.
///
uint32_t MachineBasicBlock::getSuccWeight(const_succ_iterator Succ) const {
  if (Weights.empty())
    return 0;

  return *getWeightIterator(Succ);
}

/// Set successor weight of a given iterator.
void MachineBasicBlock::setSuccWeight(succ_iterator I, uint32_t weight) {
  if (Weights.empty())
    return;
  *getWeightIterator(I) = weight;
}

/// getWeightIterator - Return wight iterator corresonding to the I successor
/// iterator
MachineBasicBlock::weight_iterator MachineBasicBlock::
getWeightIterator(MachineBasicBlock::succ_iterator I) {
  assert(Weights.size() == Successors.size() && "Async weight list!");
  size_t index = std::distance(Successors.begin(), I);
  assert(index < Weights.size() && "Not a current successor!");
  return Weights.begin() + index;
}

/// getWeightIterator - Return wight iterator corresonding to the I successor
/// iterator
MachineBasicBlock::const_weight_iterator MachineBasicBlock::
getWeightIterator(MachineBasicBlock::const_succ_iterator I) const {
  assert(Weights.size() == Successors.size() && "Async weight list!");
  const size_t index = std::distance(Successors.begin(), I);
  assert(index < Weights.size() && "Not a current successor!");
  return Weights.begin() + index;
}

/// Return whether (physical) register "Reg" has been <def>ined and not <kill>ed
/// as of just before "MI".
/// 
/// Search is localised to a neighborhood of
/// Neighborhood instructions before (searching for defs or kills) and N
/// instructions after (searching just for defs) MI.
MachineBasicBlock::LivenessQueryResult
MachineBasicBlock::computeRegisterLiveness(const TargetRegisterInfo *TRI,
                                           unsigned Reg, const_iterator Before,
                                           unsigned Neighborhood) const {
  unsigned N = Neighborhood;

  // Start by searching backwards from Before, looking for kills, reads or defs.
  const_iterator I(Before);
  // If this is the first insn in the block, don't search backwards.
  if (I != begin()) {
    do {
      --I;

      MachineOperandIteratorBase::PhysRegInfo Analysis =
        ConstMIOperands(I).analyzePhysReg(Reg, TRI);

      if (Analysis.Defines)
        // Outputs happen after inputs so they take precedence if both are
        // present.
        return Analysis.DefinesDead ? LQR_Dead : LQR_Live;

      if (Analysis.Kills || Analysis.Clobbers)
        // Register killed, so isn't live.
        return LQR_Dead;

      else if (Analysis.ReadsOverlap)
        // Defined or read without a previous kill - live.
        return Analysis.Reads ? LQR_Live : LQR_OverlappingLive;

    } while (I != begin() && --N > 0);
  }

  // Did we get to the start of the block?
  if (I == begin()) {
    // If so, the register's state is definitely defined by the live-in state.
    for (MCRegAliasIterator RAI(Reg, TRI, /*IncludeSelf=*/true);
         RAI.isValid(); ++RAI) {
      if (isLiveIn(*RAI))
        return (*RAI == Reg) ? LQR_Live : LQR_OverlappingLive;
    }

    return LQR_Dead;
  }

  N = Neighborhood;

  // Try searching forwards from Before, looking for reads or defs.
  I = const_iterator(Before);
  // If this is the last insn in the block, don't search forwards.
  if (I != end()) {
    for (++I; I != end() && N > 0; ++I, --N) {
      MachineOperandIteratorBase::PhysRegInfo Analysis =
        ConstMIOperands(I).analyzePhysReg(Reg, TRI);

      if (Analysis.ReadsOverlap)
        // Used, therefore must have been live.
        return (Analysis.Reads) ?
          LQR_Live : LQR_OverlappingLive;

      else if (Analysis.Clobbers || Analysis.Defines)
        // Defined (but not read) therefore cannot have been live.
        return LQR_Dead;
    }
  }

  // At this point we have no idea of the liveness of the register.
  return LQR_Unknown;
}
