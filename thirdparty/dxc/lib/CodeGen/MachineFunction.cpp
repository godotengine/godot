//===-- MachineFunction.cpp -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Collect native machine code information for a function.  This allows
// target-specific information about the generated code to be stored with each
// function.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionInitializer.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

#define DEBUG_TYPE "codegen"

void MachineFunctionInitializer::anchor() {}

//===----------------------------------------------------------------------===//
// MachineFunction implementation
//===----------------------------------------------------------------------===//

// Out-of-line virtual method.
MachineFunctionInfo::~MachineFunctionInfo() {}

void ilist_traits<MachineBasicBlock>::deleteNode(MachineBasicBlock *MBB) {
  MBB->getParent()->DeleteMachineBasicBlock(MBB);
}

MachineFunction::MachineFunction(const Function *F, const TargetMachine &TM,
                                 unsigned FunctionNum, MachineModuleInfo &mmi)
    : Fn(F), Target(TM), STI(TM.getSubtargetImpl(*F)), Ctx(mmi.getContext()),
      MMI(mmi) {
  if (STI->getRegisterInfo())
    RegInfo = new (Allocator) MachineRegisterInfo(this);
  else
    RegInfo = nullptr;

  MFInfo = nullptr;
  FrameInfo = new (Allocator)
      MachineFrameInfo(STI->getFrameLowering()->getStackAlignment(),
                       STI->getFrameLowering()->isStackRealignable(),
                       !F->hasFnAttribute("no-realign-stack"));

  if (Fn->hasFnAttribute(Attribute::StackAlignment))
    FrameInfo->ensureMaxAlignment(Fn->getFnStackAlignment());

  ConstantPool = new (Allocator) MachineConstantPool(getDataLayout());
  Alignment = STI->getTargetLowering()->getMinFunctionAlignment();

  // FIXME: Shouldn't use pref alignment if explicit alignment is set on Fn.
  if (!Fn->hasFnAttribute(Attribute::OptimizeForSize))
    Alignment = std::max(Alignment,
                         STI->getTargetLowering()->getPrefFunctionAlignment());

  FunctionNumber = FunctionNum;
  JumpTableInfo = nullptr;
}

MachineFunction::~MachineFunction() {
  // Don't call destructors on MachineInstr and MachineOperand. All of their
  // memory comes from the BumpPtrAllocator which is about to be purged.
  //
  // Do call MachineBasicBlock destructors, it contains std::vectors.
  for (iterator I = begin(), E = end(); I != E; I = BasicBlocks.erase(I))
    I->Insts.clearAndLeakNodesUnsafely();

  InstructionRecycler.clear(Allocator);
  OperandRecycler.clear(Allocator);
  BasicBlockRecycler.clear(Allocator);
  if (RegInfo) {
    RegInfo->~MachineRegisterInfo();
    Allocator.Deallocate(RegInfo);
  }
  if (MFInfo) {
    MFInfo->~MachineFunctionInfo();
    Allocator.Deallocate(MFInfo);
  }

  FrameInfo->~MachineFrameInfo();
  Allocator.Deallocate(FrameInfo);

  ConstantPool->~MachineConstantPool();
  Allocator.Deallocate(ConstantPool);

  if (JumpTableInfo) {
    JumpTableInfo->~MachineJumpTableInfo();
    Allocator.Deallocate(JumpTableInfo);
  }
}

const DataLayout &MachineFunction::getDataLayout() const {
  return Fn->getParent()->getDataLayout();
}

/// Get the JumpTableInfo for this function.
/// If it does not already exist, allocate one.
MachineJumpTableInfo *MachineFunction::
getOrCreateJumpTableInfo(unsigned EntryKind) {
  if (JumpTableInfo) return JumpTableInfo;

  JumpTableInfo = new (Allocator)
    MachineJumpTableInfo((MachineJumpTableInfo::JTEntryKind)EntryKind);
  return JumpTableInfo;
}

/// Should we be emitting segmented stack stuff for the function
bool MachineFunction::shouldSplitStack() {
  return getFunction()->hasFnAttribute("split-stack");
}

/// This discards all of the MachineBasicBlock numbers and recomputes them.
/// This guarantees that the MBB numbers are sequential, dense, and match the
/// ordering of the blocks within the function.  If a specific MachineBasicBlock
/// is specified, only that block and those after it are renumbered.
void MachineFunction::RenumberBlocks(MachineBasicBlock *MBB) {
  if (empty()) { MBBNumbering.clear(); return; }
  MachineFunction::iterator MBBI, E = end();
  if (MBB == nullptr)
    MBBI = begin();
  else
    MBBI = MBB;

  // Figure out the block number this should have.
  unsigned BlockNo = 0;
  if (MBBI != begin())
    BlockNo = std::prev(MBBI)->getNumber() + 1;

  for (; MBBI != E; ++MBBI, ++BlockNo) {
    if (MBBI->getNumber() != (int)BlockNo) {
      // Remove use of the old number.
      if (MBBI->getNumber() != -1) {
        assert(MBBNumbering[MBBI->getNumber()] == &*MBBI &&
               "MBB number mismatch!");
        MBBNumbering[MBBI->getNumber()] = nullptr;
      }

      // If BlockNo is already taken, set that block's number to -1.
      if (MBBNumbering[BlockNo])
        MBBNumbering[BlockNo]->setNumber(-1);

      MBBNumbering[BlockNo] = MBBI;
      MBBI->setNumber(BlockNo);
    }
  }

  // Okay, all the blocks are renumbered.  If we have compactified the block
  // numbering, shrink MBBNumbering now.
  assert(BlockNo <= MBBNumbering.size() && "Mismatch!");
  MBBNumbering.resize(BlockNo);
}

/// Allocate a new MachineInstr. Use this instead of `new MachineInstr'.
MachineInstr *
MachineFunction::CreateMachineInstr(const MCInstrDesc &MCID,
                                    DebugLoc DL, bool NoImp) {
  return new (InstructionRecycler.Allocate<MachineInstr>(Allocator))
    MachineInstr(*this, MCID, DL, NoImp);
}

/// Create a new MachineInstr which is a copy of the 'Orig' instruction,
/// identical in all ways except the instruction has no parent, prev, or next.
MachineInstr *
MachineFunction::CloneMachineInstr(const MachineInstr *Orig) {
  return new (InstructionRecycler.Allocate<MachineInstr>(Allocator))
             MachineInstr(*this, *Orig);
}

/// Delete the given MachineInstr.
///
/// This function also serves as the MachineInstr destructor - the real
/// ~MachineInstr() destructor must be empty.
void
MachineFunction::DeleteMachineInstr(MachineInstr *MI) {
  // Strip it for parts. The operand array and the MI object itself are
  // independently recyclable.
  if (MI->Operands)
    deallocateOperandArray(MI->CapOperands, MI->Operands);
  // Don't call ~MachineInstr() which must be trivial anyway because
  // ~MachineFunction drops whole lists of MachineInstrs wihout calling their
  // destructors.
  InstructionRecycler.Deallocate(Allocator, MI);
}

/// Allocate a new MachineBasicBlock. Use this instead of
/// `new MachineBasicBlock'.
MachineBasicBlock *
MachineFunction::CreateMachineBasicBlock(const BasicBlock *bb) {
  return new (BasicBlockRecycler.Allocate<MachineBasicBlock>(Allocator))
             MachineBasicBlock(*this, bb);
}

/// Delete the given MachineBasicBlock.
void
MachineFunction::DeleteMachineBasicBlock(MachineBasicBlock *MBB) {
  assert(MBB->getParent() == this && "MBB parent mismatch!");
  MBB->~MachineBasicBlock();
  BasicBlockRecycler.Deallocate(Allocator, MBB);
}

MachineMemOperand *
MachineFunction::getMachineMemOperand(MachinePointerInfo PtrInfo, unsigned f,
                                      uint64_t s, unsigned base_alignment,
                                      const AAMDNodes &AAInfo,
                                      const MDNode *Ranges) {
  return new (Allocator) MachineMemOperand(PtrInfo, f, s, base_alignment,
                                           AAInfo, Ranges);
}

MachineMemOperand *
MachineFunction::getMachineMemOperand(const MachineMemOperand *MMO,
                                      int64_t Offset, uint64_t Size) {
  if (MMO->getValue())
    return new (Allocator)
               MachineMemOperand(MachinePointerInfo(MMO->getValue(),
                                                    MMO->getOffset()+Offset),
                                 MMO->getFlags(), Size,
                                 MMO->getBaseAlignment());
  return new (Allocator)
             MachineMemOperand(MachinePointerInfo(MMO->getPseudoValue(),
                                                  MMO->getOffset()+Offset),
                               MMO->getFlags(), Size,
                               MMO->getBaseAlignment());
}

MachineInstr::mmo_iterator
MachineFunction::allocateMemRefsArray(unsigned long Num) {
  return Allocator.Allocate<MachineMemOperand *>(Num);
}

std::pair<MachineInstr::mmo_iterator, MachineInstr::mmo_iterator>
MachineFunction::extractLoadMemRefs(MachineInstr::mmo_iterator Begin,
                                    MachineInstr::mmo_iterator End) {
  // Count the number of load mem refs.
  unsigned Num = 0;
  for (MachineInstr::mmo_iterator I = Begin; I != End; ++I)
    if ((*I)->isLoad())
      ++Num;

  // Allocate a new array and populate it with the load information.
  MachineInstr::mmo_iterator Result = allocateMemRefsArray(Num);
  unsigned Index = 0;
  for (MachineInstr::mmo_iterator I = Begin; I != End; ++I) {
    if ((*I)->isLoad()) {
      if (!(*I)->isStore())
        // Reuse the MMO.
        Result[Index] = *I;
      else {
        // Clone the MMO and unset the store flag.
        MachineMemOperand *JustLoad =
          getMachineMemOperand((*I)->getPointerInfo(),
                               (*I)->getFlags() & ~MachineMemOperand::MOStore,
                               (*I)->getSize(), (*I)->getBaseAlignment(),
                               (*I)->getAAInfo());
        Result[Index] = JustLoad;
      }
      ++Index;
    }
  }
  return std::make_pair(Result, Result + Num);
}

std::pair<MachineInstr::mmo_iterator, MachineInstr::mmo_iterator>
MachineFunction::extractStoreMemRefs(MachineInstr::mmo_iterator Begin,
                                     MachineInstr::mmo_iterator End) {
  // Count the number of load mem refs.
  unsigned Num = 0;
  for (MachineInstr::mmo_iterator I = Begin; I != End; ++I)
    if ((*I)->isStore())
      ++Num;

  // Allocate a new array and populate it with the store information.
  MachineInstr::mmo_iterator Result = allocateMemRefsArray(Num);
  unsigned Index = 0;
  for (MachineInstr::mmo_iterator I = Begin; I != End; ++I) {
    if ((*I)->isStore()) {
      if (!(*I)->isLoad())
        // Reuse the MMO.
        Result[Index] = *I;
      else {
        // Clone the MMO and unset the load flag.
        MachineMemOperand *JustStore =
          getMachineMemOperand((*I)->getPointerInfo(),
                               (*I)->getFlags() & ~MachineMemOperand::MOLoad,
                               (*I)->getSize(), (*I)->getBaseAlignment(),
                               (*I)->getAAInfo());
        Result[Index] = JustStore;
      }
      ++Index;
    }
  }
  return std::make_pair(Result, Result + Num);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void MachineFunction::dump() const {
  print(dbgs());
}
#endif

StringRef MachineFunction::getName() const {
  assert(getFunction() && "No function!");
  return getFunction()->getName();
}

void MachineFunction::print(raw_ostream &OS, SlotIndexes *Indexes) const {
  OS << "# Machine code for function " << getName() << ": ";
  if (RegInfo) {
    OS << (RegInfo->isSSA() ? "SSA" : "Post SSA");
    if (!RegInfo->tracksLiveness())
      OS << ", not tracking liveness";
  }
  OS << '\n';

  // Print Frame Information
  FrameInfo->print(*this, OS);

  // Print JumpTable Information
  if (JumpTableInfo)
    JumpTableInfo->print(OS);

  // Print Constant Pool
  ConstantPool->print(OS);

  const TargetRegisterInfo *TRI = getSubtarget().getRegisterInfo();

  if (RegInfo && !RegInfo->livein_empty()) {
    OS << "Function Live Ins: ";
    for (MachineRegisterInfo::livein_iterator
         I = RegInfo->livein_begin(), E = RegInfo->livein_end(); I != E; ++I) {
      OS << PrintReg(I->first, TRI);
      if (I->second)
        OS << " in " << PrintReg(I->second, TRI);
      if (std::next(I) != E)
        OS << ", ";
    }
    OS << '\n';
  }

  ModuleSlotTracker MST(getFunction()->getParent());
  MST.incorporateFunction(*getFunction());
  for (const auto &BB : *this) {
    OS << '\n';
    BB.print(OS, MST, Indexes);
  }

  OS << "\n# End machine code for function " << getName() << ".\n\n";
}

namespace llvm {
  template<>
  struct DOTGraphTraits<const MachineFunction*> : public DefaultDOTGraphTraits {

  DOTGraphTraits (bool isSimple=false) : DefaultDOTGraphTraits(isSimple) {}

    static std::string getGraphName(const MachineFunction *F) {
      return ("CFG for '" + F->getName() + "' function").str();
    }

    std::string getNodeLabel(const MachineBasicBlock *Node,
                             const MachineFunction *Graph) {
      std::string OutStr;
      {
        raw_string_ostream OSS(OutStr);

        if (isSimple()) {
          OSS << "BB#" << Node->getNumber();
          if (const BasicBlock *BB = Node->getBasicBlock())
            OSS << ": " << BB->getName();
        } else
          Node->print(OSS);
      }

      if (OutStr[0] == '\n') OutStr.erase(OutStr.begin());

      // Process string output to make it nicer...
      for (unsigned i = 0; i != OutStr.length(); ++i)
        if (OutStr[i] == '\n') {                            // Left justify
          OutStr[i] = '\\';
          OutStr.insert(OutStr.begin()+i+1, 'l');
        }
      return OutStr;
    }
  };
}

void MachineFunction::viewCFG() const
{
#ifndef NDEBUG
  ViewGraph(this, "mf" + getName());
#else
  errs() << "MachineFunction::viewCFG is only available in debug builds on "
         << "systems with Graphviz or gv!\n";
#endif // NDEBUG
}

void MachineFunction::viewCFGOnly() const
{
#ifndef NDEBUG
  ViewGraph(this, "mf" + getName(), true);
#else
  errs() << "MachineFunction::viewCFGOnly is only available in debug builds on "
         << "systems with Graphviz or gv!\n";
#endif // NDEBUG
}

/// Add the specified physical register as a live-in value and
/// create a corresponding virtual register for it.
unsigned MachineFunction::addLiveIn(unsigned PReg,
                                    const TargetRegisterClass *RC) {
  MachineRegisterInfo &MRI = getRegInfo();
  unsigned VReg = MRI.getLiveInVirtReg(PReg);
  if (VReg) {
    const TargetRegisterClass *VRegRC = MRI.getRegClass(VReg);
    (void)VRegRC;
    // A physical register can be added several times.
    // Between two calls, the register class of the related virtual register
    // may have been constrained to match some operation constraints.
    // In that case, check that the current register class includes the
    // physical register and is a sub class of the specified RC.
    assert((VRegRC == RC || (VRegRC->contains(PReg) &&
                             RC->hasSubClassEq(VRegRC))) &&
            "Register class mismatch!");
    return VReg;
  }
  VReg = MRI.createVirtualRegister(RC);
  MRI.addLiveIn(PReg, VReg);
  return VReg;
}

/// Return the MCSymbol for the specified non-empty jump table.
/// If isLinkerPrivate is specified, an 'l' label is returned, otherwise a
/// normal 'L' label is returned.
MCSymbol *MachineFunction::getJTISymbol(unsigned JTI, MCContext &Ctx,
                                        bool isLinkerPrivate) const {
  const DataLayout &DL = getDataLayout();
  assert(JumpTableInfo && "No jump tables");
  assert(JTI < JumpTableInfo->getJumpTables().size() && "Invalid JTI!");

  const char *Prefix = isLinkerPrivate ? DL.getLinkerPrivateGlobalPrefix()
                                       : DL.getPrivateGlobalPrefix();
  SmallString<60> Name;
  raw_svector_ostream(Name)
    << Prefix << "JTI" << getFunctionNumber() << '_' << JTI;
  return Ctx.getOrCreateSymbol(Name);
}

/// Return a function-local symbol to represent the PIC base.
MCSymbol *MachineFunction::getPICBaseSymbol() const {
  const DataLayout &DL = getDataLayout();
  return Ctx.getOrCreateSymbol(Twine(DL.getPrivateGlobalPrefix()) +
                               Twine(getFunctionNumber()) + "$pb");
}

//===----------------------------------------------------------------------===//
//  MachineFrameInfo implementation
//===----------------------------------------------------------------------===//

/// Make sure the function is at least Align bytes aligned.
void MachineFrameInfo::ensureMaxAlignment(unsigned Align) {
  if (!StackRealignable || !RealignOption)
    assert(Align <= StackAlignment &&
           "For targets without stack realignment, Align is out of limit!");
  if (MaxAlignment < Align) MaxAlignment = Align;
}

/// Clamp the alignment if requested and emit a warning.
static inline unsigned clampStackAlignment(bool ShouldClamp, unsigned Align,
                                           unsigned StackAlign) {
  if (!ShouldClamp || Align <= StackAlign)
    return Align;
  DEBUG(dbgs() << "Warning: requested alignment " << Align
               << " exceeds the stack alignment " << StackAlign
               << " when stack realignment is off" << '\n');
  return StackAlign;
}

/// Create a new statically sized stack object, returning a nonnegative
/// identifier to represent it.
int MachineFrameInfo::CreateStackObject(uint64_t Size, unsigned Alignment,
                      bool isSS, const AllocaInst *Alloca) {
  assert(Size != 0 && "Cannot allocate zero size stack objects!");
  Alignment = clampStackAlignment(!StackRealignable || !RealignOption,
                                  Alignment, StackAlignment);
  Objects.push_back(StackObject(Size, Alignment, 0, false, isSS, Alloca,
                                !isSS));
  int Index = (int)Objects.size() - NumFixedObjects - 1;
  assert(Index >= 0 && "Bad frame index!");
  ensureMaxAlignment(Alignment);
  return Index;
}

/// Create a new statically sized stack object that represents a spill slot,
/// returning a nonnegative identifier to represent it.
int MachineFrameInfo::CreateSpillStackObject(uint64_t Size,
                                             unsigned Alignment) {
  Alignment = clampStackAlignment(!StackRealignable || !RealignOption,
                                  Alignment, StackAlignment);
  CreateStackObject(Size, Alignment, true);
  int Index = (int)Objects.size() - NumFixedObjects - 1;
  ensureMaxAlignment(Alignment);
  return Index;
}

/// Notify the MachineFrameInfo object that a variable sized object has been
/// created. This must be created whenever a variable sized object is created,
/// whether or not the index returned is actually used.
int MachineFrameInfo::CreateVariableSizedObject(unsigned Alignment,
                                                const AllocaInst *Alloca) {
  HasVarSizedObjects = true;
  Alignment = clampStackAlignment(!StackRealignable || !RealignOption,
                                  Alignment, StackAlignment);
  Objects.push_back(StackObject(0, Alignment, 0, false, false, Alloca, true));
  ensureMaxAlignment(Alignment);
  return (int)Objects.size()-NumFixedObjects-1;
}

/// Create a new object at a fixed location on the stack.
/// All fixed objects should be created before other objects are created for
/// efficiency. By default, fixed objects are immutable. This returns an
/// index with a negative value.
int MachineFrameInfo::CreateFixedObject(uint64_t Size, int64_t SPOffset,
                                        bool Immutable, bool isAliased) {
  assert(Size != 0 && "Cannot allocate zero size fixed stack objects!");
  // The alignment of the frame index can be determined from its offset from
  // the incoming frame position.  If the frame object is at offset 32 and
  // the stack is guaranteed to be 16-byte aligned, then we know that the
  // object is 16-byte aligned.
  unsigned Align = MinAlign(SPOffset, StackAlignment);
  Align = clampStackAlignment(!StackRealignable || !RealignOption, Align,
                              StackAlignment);
  Objects.insert(Objects.begin(), StackObject(Size, Align, SPOffset, Immutable,
                                              /*isSS*/   false,
                                              /*Alloca*/ nullptr, isAliased));
  return -++NumFixedObjects;
}

/// Create a spill slot at a fixed location on the stack.
/// Returns an index with a negative value.
int MachineFrameInfo::CreateFixedSpillStackObject(uint64_t Size,
                                                  int64_t SPOffset) {
  unsigned Align = MinAlign(SPOffset, StackAlignment);
  Align = clampStackAlignment(!StackRealignable || !RealignOption, Align,
                              StackAlignment);
  Objects.insert(Objects.begin(), StackObject(Size, Align, SPOffset,
                                              /*Immutable*/ true,
                                              /*isSS*/ true,
                                              /*Alloca*/ nullptr,
                                              /*isAliased*/ false));
  return -++NumFixedObjects;
}

BitVector MachineFrameInfo::getPristineRegs(const MachineFunction &MF) const {
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  BitVector BV(TRI->getNumRegs());

  // Before CSI is calculated, no registers are considered pristine. They can be
  // freely used and PEI will make sure they are saved.
  if (!isCalleeSavedInfoValid())
    return BV;

  for (const MCPhysReg *CSR = TRI->getCalleeSavedRegs(&MF); CSR && *CSR; ++CSR)
    BV.set(*CSR);

  // Saved CSRs are not pristine.
  const std::vector<CalleeSavedInfo> &CSI = getCalleeSavedInfo();
  for (std::vector<CalleeSavedInfo>::const_iterator I = CSI.begin(),
         E = CSI.end(); I != E; ++I)
    BV.reset(I->getReg());

  return BV;
}

unsigned MachineFrameInfo::estimateStackSize(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();
  unsigned MaxAlign = getMaxAlignment();
  int Offset = 0;

  // This code is very, very similar to PEI::calculateFrameObjectOffsets().
  // It really should be refactored to share code. Until then, changes
  // should keep in mind that there's tight coupling between the two.

  for (int i = getObjectIndexBegin(); i != 0; ++i) {
    int FixedOff = -getObjectOffset(i);
    if (FixedOff > Offset) Offset = FixedOff;
  }
  for (unsigned i = 0, e = getObjectIndexEnd(); i != e; ++i) {
    if (isDeadObjectIndex(i))
      continue;
    Offset += getObjectSize(i);
    unsigned Align = getObjectAlignment(i);
    // Adjust to alignment boundary
    Offset = (Offset+Align-1)/Align*Align;

    MaxAlign = std::max(Align, MaxAlign);
  }

  if (adjustsStack() && TFI->hasReservedCallFrame(MF))
    Offset += getMaxCallFrameSize();

  // Round up the size to a multiple of the alignment.  If the function has
  // any calls or alloca's, align to the target's StackAlignment value to
  // ensure that the callee's frame or the alloca data is suitably aligned;
  // otherwise, for leaf functions, align to the TransientStackAlignment
  // value.
  unsigned StackAlign;
  if (adjustsStack() || hasVarSizedObjects() ||
      (RegInfo->needsStackRealignment(MF) && getObjectIndexEnd() != 0))
    StackAlign = TFI->getStackAlignment();
  else
    StackAlign = TFI->getTransientStackAlignment();

  // If the frame pointer is eliminated, all frame offsets will be relative to
  // SP not FP. Align to MaxAlign so this works.
  StackAlign = std::max(StackAlign, MaxAlign);
  unsigned AlignMask = StackAlign - 1;
  Offset = (Offset + AlignMask) & ~uint64_t(AlignMask);

  return (unsigned)Offset;
}

void MachineFrameInfo::print(const MachineFunction &MF, raw_ostream &OS) const{
  if (Objects.empty()) return;

  const TargetFrameLowering *FI = MF.getSubtarget().getFrameLowering();
  int ValOffset = (FI ? FI->getOffsetOfLocalArea() : 0);

  OS << "Frame Objects:\n";

  for (unsigned i = 0, e = Objects.size(); i != e; ++i) {
    const StackObject &SO = Objects[i];
    OS << "  fi#" << (int)(i-NumFixedObjects) << ": ";
    if (SO.Size == ~0ULL) {
      OS << "dead\n";
      continue;
    }
    if (SO.Size == 0)
      OS << "variable sized";
    else
      OS << "size=" << SO.Size;
    OS << ", align=" << SO.Alignment;

    if (i < NumFixedObjects)
      OS << ", fixed";
    if (i < NumFixedObjects || SO.SPOffset != -1) {
      int64_t Off = SO.SPOffset - ValOffset;
      OS << ", at location [SP";
      if (Off > 0)
        OS << "+" << Off;
      else if (Off < 0)
        OS << Off;
      OS << "]";
    }
    OS << "\n";
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void MachineFrameInfo::dump(const MachineFunction &MF) const {
  print(MF, dbgs());
}
#endif

//===----------------------------------------------------------------------===//
//  MachineJumpTableInfo implementation
//===----------------------------------------------------------------------===//

/// Return the size of each entry in the jump table.
unsigned MachineJumpTableInfo::getEntrySize(const DataLayout &TD) const {
  // The size of a jump table entry is 4 bytes unless the entry is just the
  // address of a block, in which case it is the pointer size.
  switch (getEntryKind()) {
  case MachineJumpTableInfo::EK_BlockAddress:
    return TD.getPointerSize();
  case MachineJumpTableInfo::EK_GPRel64BlockAddress:
    return 8;
  case MachineJumpTableInfo::EK_GPRel32BlockAddress:
  case MachineJumpTableInfo::EK_LabelDifference32:
  case MachineJumpTableInfo::EK_Custom32:
    return 4;
  case MachineJumpTableInfo::EK_Inline:
    return 0;
  }
  llvm_unreachable("Unknown jump table encoding!");
}

/// Return the alignment of each entry in the jump table.
unsigned MachineJumpTableInfo::getEntryAlignment(const DataLayout &TD) const {
  // The alignment of a jump table entry is the alignment of int32 unless the
  // entry is just the address of a block, in which case it is the pointer
  // alignment.
  switch (getEntryKind()) {
  case MachineJumpTableInfo::EK_BlockAddress:
    return TD.getPointerABIAlignment();
  case MachineJumpTableInfo::EK_GPRel64BlockAddress:
    return TD.getABIIntegerTypeAlignment(64);
  case MachineJumpTableInfo::EK_GPRel32BlockAddress:
  case MachineJumpTableInfo::EK_LabelDifference32:
  case MachineJumpTableInfo::EK_Custom32:
    return TD.getABIIntegerTypeAlignment(32);
  case MachineJumpTableInfo::EK_Inline:
    return 1;
  }
  llvm_unreachable("Unknown jump table encoding!");
}

/// Create a new jump table entry in the jump table info.
unsigned MachineJumpTableInfo::createJumpTableIndex(
                               const std::vector<MachineBasicBlock*> &DestBBs) {
  assert(!DestBBs.empty() && "Cannot create an empty jump table!");
  JumpTables.push_back(MachineJumpTableEntry(DestBBs));
  return JumpTables.size()-1;
}

/// If Old is the target of any jump tables, update the jump tables to branch
/// to New instead.
bool MachineJumpTableInfo::ReplaceMBBInJumpTables(MachineBasicBlock *Old,
                                                  MachineBasicBlock *New) {
  assert(Old != New && "Not making a change?");
  bool MadeChange = false;
  for (size_t i = 0, e = JumpTables.size(); i != e; ++i)
    ReplaceMBBInJumpTable(i, Old, New);
  return MadeChange;
}

/// If Old is a target of the jump tables, update the jump table to branch to
/// New instead.
bool MachineJumpTableInfo::ReplaceMBBInJumpTable(unsigned Idx,
                                                 MachineBasicBlock *Old,
                                                 MachineBasicBlock *New) {
  assert(Old != New && "Not making a change?");
  bool MadeChange = false;
  MachineJumpTableEntry &JTE = JumpTables[Idx];
  for (size_t j = 0, e = JTE.MBBs.size(); j != e; ++j)
    if (JTE.MBBs[j] == Old) {
      JTE.MBBs[j] = New;
      MadeChange = true;
    }
  return MadeChange;
}

void MachineJumpTableInfo::print(raw_ostream &OS) const {
  if (JumpTables.empty()) return;

  OS << "Jump Tables:\n";

  for (unsigned i = 0, e = JumpTables.size(); i != e; ++i) {
    OS << "  jt#" << i << ": ";
    for (unsigned j = 0, f = JumpTables[i].MBBs.size(); j != f; ++j)
      OS << " BB#" << JumpTables[i].MBBs[j]->getNumber();
  }

  OS << '\n';
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void MachineJumpTableInfo::dump() const { print(dbgs()); }
#endif


//===----------------------------------------------------------------------===//
//  MachineConstantPool implementation
//===----------------------------------------------------------------------===//

void MachineConstantPoolValue::anchor() { }

Type *MachineConstantPoolEntry::getType() const {
  if (isMachineConstantPoolEntry())
    return Val.MachineCPVal->getType();
  return Val.ConstVal->getType();
}


unsigned MachineConstantPoolEntry::getRelocationInfo() const {
  if (isMachineConstantPoolEntry())
    return Val.MachineCPVal->getRelocationInfo();
  return Val.ConstVal->getRelocationInfo();
}

SectionKind
MachineConstantPoolEntry::getSectionKind(const DataLayout *DL) const {
  SectionKind Kind;
  switch (getRelocationInfo()) {
  default:
    llvm_unreachable("Unknown section kind");
  case Constant::GlobalRelocations:
    Kind = SectionKind::getReadOnlyWithRel();
    break;
  case Constant::LocalRelocation:
    Kind = SectionKind::getReadOnlyWithRelLocal();
    break;
  case Constant::NoRelocation:
    switch (DL->getTypeAllocSize(getType())) {
    case 4:
      Kind = SectionKind::getMergeableConst4();
      break;
    case 8:
      Kind = SectionKind::getMergeableConst8();
      break;
    case 16:
      Kind = SectionKind::getMergeableConst16();
      break;
    default:
      Kind = SectionKind::getReadOnly();
      break;
    }
  }
  return Kind;
}

MachineConstantPool::~MachineConstantPool() {
  for (unsigned i = 0, e = Constants.size(); i != e; ++i)
    if (Constants[i].isMachineConstantPoolEntry())
      delete Constants[i].Val.MachineCPVal;
  for (DenseSet<MachineConstantPoolValue*>::iterator I =
       MachineCPVsSharingEntries.begin(), E = MachineCPVsSharingEntries.end();
       I != E; ++I)
    delete *I;
}

/// Test whether the given two constants can be allocated the same constant pool
/// entry.
static bool CanShareConstantPoolEntry(const Constant *A, const Constant *B,
                                      const DataLayout &DL) {
  // Handle the trivial case quickly.
  if (A == B) return true;

  // If they have the same type but weren't the same constant, quickly
  // reject them.
  if (A->getType() == B->getType()) return false;

  // We can't handle structs or arrays.
  if (isa<StructType>(A->getType()) || isa<ArrayType>(A->getType()) ||
      isa<StructType>(B->getType()) || isa<ArrayType>(B->getType()))
    return false;

  // For now, only support constants with the same size.
  uint64_t StoreSize = DL.getTypeStoreSize(A->getType());
  if (StoreSize != DL.getTypeStoreSize(B->getType()) || StoreSize > 128)
    return false;

  Type *IntTy = IntegerType::get(A->getContext(), StoreSize*8);

  // Try constant folding a bitcast of both instructions to an integer.  If we
  // get two identical ConstantInt's, then we are good to share them.  We use
  // the constant folding APIs to do this so that we get the benefit of
  // DataLayout.
  if (isa<PointerType>(A->getType()))
    A = ConstantFoldInstOperands(Instruction::PtrToInt, IntTy,
                                 const_cast<Constant *>(A), DL);
  else if (A->getType() != IntTy)
    A = ConstantFoldInstOperands(Instruction::BitCast, IntTy,
                                 const_cast<Constant *>(A), DL);
  if (isa<PointerType>(B->getType()))
    B = ConstantFoldInstOperands(Instruction::PtrToInt, IntTy,
                                 const_cast<Constant *>(B), DL);
  else if (B->getType() != IntTy)
    B = ConstantFoldInstOperands(Instruction::BitCast, IntTy,
                                 const_cast<Constant *>(B), DL);

  return A == B;
}

/// Create a new entry in the constant pool or return an existing one.
/// User must specify the log2 of the minimum required alignment for the object.
unsigned MachineConstantPool::getConstantPoolIndex(const Constant *C,
                                                   unsigned Alignment) {
  assert(Alignment && "Alignment must be specified!");
  if (Alignment > PoolAlignment) PoolAlignment = Alignment;

  // Check to see if we already have this constant.
  //
  // FIXME, this could be made much more efficient for large constant pools.
  for (unsigned i = 0, e = Constants.size(); i != e; ++i)
    if (!Constants[i].isMachineConstantPoolEntry() &&
        CanShareConstantPoolEntry(Constants[i].Val.ConstVal, C, DL)) {
      if ((unsigned)Constants[i].getAlignment() < Alignment)
        Constants[i].Alignment = Alignment;
      return i;
    }

  Constants.push_back(MachineConstantPoolEntry(C, Alignment));
  return Constants.size()-1;
}

unsigned MachineConstantPool::getConstantPoolIndex(MachineConstantPoolValue *V,
                                                   unsigned Alignment) {
  assert(Alignment && "Alignment must be specified!");
  if (Alignment > PoolAlignment) PoolAlignment = Alignment;

  // Check to see if we already have this constant.
  //
  // FIXME, this could be made much more efficient for large constant pools.
  int Idx = V->getExistingMachineCPValue(this, Alignment);
  if (Idx != -1) {
    MachineCPVsSharingEntries.insert(V);
    return (unsigned)Idx;
  }

  Constants.push_back(MachineConstantPoolEntry(V, Alignment));
  return Constants.size()-1;
}

void MachineConstantPool::print(raw_ostream &OS) const {
  if (Constants.empty()) return;

  OS << "Constant Pool:\n";
  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    OS << "  cp#" << i << ": ";
    if (Constants[i].isMachineConstantPoolEntry())
      Constants[i].Val.MachineCPVal->print(OS);
    else
      Constants[i].Val.ConstVal->printAsOperand(OS, /*PrintType=*/false);
    OS << ", align=" << Constants[i].getAlignment();
    OS << "\n";
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void MachineConstantPool::dump() const { print(dbgs()); }
#endif
