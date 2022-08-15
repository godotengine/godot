//===- TargetRegisterInfo.cpp - Target Register Information Implementation ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TargetRegisterInfo interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

TargetRegisterInfo::TargetRegisterInfo(const TargetRegisterInfoDesc *ID,
                             regclass_iterator RCB, regclass_iterator RCE,
                             const char *const *SRINames,
                             const unsigned *SRILaneMasks,
                             unsigned SRICoveringLanes)
  : InfoDesc(ID), SubRegIndexNames(SRINames),
    SubRegIndexLaneMasks(SRILaneMasks),
    RegClassBegin(RCB), RegClassEnd(RCE),
    CoveringLanes(SRICoveringLanes) {
}

TargetRegisterInfo::~TargetRegisterInfo() {}

void PrintReg::print(raw_ostream &OS) const {
  if (!Reg)
    OS << "%noreg";
  else if (TargetRegisterInfo::isStackSlot(Reg))
    OS << "SS#" << TargetRegisterInfo::stackSlot2Index(Reg);
  else if (TargetRegisterInfo::isVirtualRegister(Reg))
    OS << "%vreg" << TargetRegisterInfo::virtReg2Index(Reg);
  else if (TRI && Reg < TRI->getNumRegs())
    OS << '%' << TRI->getName(Reg);
  else
    OS << "%physreg" << Reg;
  if (SubIdx) {
    if (TRI)
      OS << ':' << TRI->getSubRegIndexName(SubIdx);
    else
      OS << ":sub(" << SubIdx << ')';
  }
}

void PrintRegUnit::print(raw_ostream &OS) const {
  // Generic printout when TRI is missing.
  if (!TRI) {
    OS << "Unit~" << Unit;
    return;
  }

  // Check for invalid register units.
  if (Unit >= TRI->getNumRegUnits()) {
    OS << "BadUnit~" << Unit;
    return;
  }

  // Normal units have at least one root.
  MCRegUnitRootIterator Roots(Unit, TRI);
  assert(Roots.isValid() && "Unit has no roots.");
  OS << TRI->getName(*Roots);
  for (++Roots; Roots.isValid(); ++Roots)
    OS << '~' << TRI->getName(*Roots);
}

void PrintVRegOrUnit::print(raw_ostream &OS) const {
  if (TRI && TRI->isVirtualRegister(Unit)) {
    OS << "%vreg" << TargetRegisterInfo::virtReg2Index(Unit);
    return;
  }
  PrintRegUnit::print(OS);
}

/// getAllocatableClass - Return the maximal subclass of the given register
/// class that is alloctable, or NULL.
const TargetRegisterClass *
TargetRegisterInfo::getAllocatableClass(const TargetRegisterClass *RC) const {
  if (!RC || RC->isAllocatable())
    return RC;

  const unsigned *SubClass = RC->getSubClassMask();
  for (unsigned Base = 0, BaseE = getNumRegClasses();
       Base < BaseE; Base += 32) {
    unsigned Idx = Base;
    for (unsigned Mask = *SubClass++; Mask; Mask >>= 1) {
      unsigned Offset = countTrailingZeros(Mask);
      const TargetRegisterClass *SubRC = getRegClass(Idx + Offset);
      if (SubRC->isAllocatable())
        return SubRC;
      Mask >>= Offset;
      Idx += Offset + 1;
    }
  }
  return nullptr;
}

/// getMinimalPhysRegClass - Returns the Register Class of a physical
/// register of the given type, picking the most sub register class of
/// the right type that contains this physreg.
const TargetRegisterClass *
TargetRegisterInfo::getMinimalPhysRegClass(unsigned reg, MVT VT) const {
  assert(isPhysicalRegister(reg) && "reg must be a physical register");

  // Pick the most sub register class of the right type that contains
  // this physreg.
  const TargetRegisterClass* BestRC = nullptr;
  for (regclass_iterator I = regclass_begin(), E = regclass_end(); I != E; ++I){
    const TargetRegisterClass* RC = *I;
    if ((VT == MVT::Other || RC->hasType(VT)) && RC->contains(reg) &&
        (!BestRC || BestRC->hasSubClass(RC)))
      BestRC = RC;
  }

  assert(BestRC && "Couldn't find the register class");
  return BestRC;
}

/// getAllocatableSetForRC - Toggle the bits that represent allocatable
/// registers for the specific register class.
static void getAllocatableSetForRC(const MachineFunction &MF,
                                   const TargetRegisterClass *RC, BitVector &R){
  assert(RC->isAllocatable() && "invalid for nonallocatable sets");
  ArrayRef<MCPhysReg> Order = RC->getRawAllocationOrder(MF);
  for (unsigned i = 0; i != Order.size(); ++i)
    R.set(Order[i]);
}

BitVector TargetRegisterInfo::getAllocatableSet(const MachineFunction &MF,
                                          const TargetRegisterClass *RC) const {
  BitVector Allocatable(getNumRegs());
  if (RC) {
    // A register class with no allocatable subclass returns an empty set.
    const TargetRegisterClass *SubClass = getAllocatableClass(RC);
    if (SubClass)
      getAllocatableSetForRC(MF, SubClass, Allocatable);
  } else {
    for (TargetRegisterInfo::regclass_iterator I = regclass_begin(),
         E = regclass_end(); I != E; ++I)
      if ((*I)->isAllocatable())
        getAllocatableSetForRC(MF, *I, Allocatable);
  }

  // Mask out the reserved registers
  BitVector Reserved = getReservedRegs(MF);
  Allocatable &= Reserved.flip();

  return Allocatable;
}

static inline
const TargetRegisterClass *firstCommonClass(const uint32_t *A,
                                            const uint32_t *B,
                                            const TargetRegisterInfo *TRI) {
  for (unsigned I = 0, E = TRI->getNumRegClasses(); I < E; I += 32)
    if (unsigned Common = *A++ & *B++)
      return TRI->getRegClass(I + countTrailingZeros(Common));
  return nullptr;
}

const TargetRegisterClass *
TargetRegisterInfo::getCommonSubClass(const TargetRegisterClass *A,
                                      const TargetRegisterClass *B) const {
  // First take care of the trivial cases.
  if (A == B)
    return A;
  if (!A || !B)
    return nullptr;

  // Register classes are ordered topologically, so the largest common
  // sub-class it the common sub-class with the smallest ID.
  return firstCommonClass(A->getSubClassMask(), B->getSubClassMask(), this);
}

const TargetRegisterClass *
TargetRegisterInfo::getMatchingSuperRegClass(const TargetRegisterClass *A,
                                             const TargetRegisterClass *B,
                                             unsigned Idx) const {
  assert(A && B && "Missing register class");
  assert(Idx && "Bad sub-register index");

  // Find Idx in the list of super-register indices.
  for (SuperRegClassIterator RCI(B, this); RCI.isValid(); ++RCI)
    if (RCI.getSubReg() == Idx)
      // The bit mask contains all register classes that are projected into B
      // by Idx. Find a class that is also a sub-class of A.
      return firstCommonClass(RCI.getMask(), A->getSubClassMask(), this);
  return nullptr;
}

const TargetRegisterClass *TargetRegisterInfo::
getCommonSuperRegClass(const TargetRegisterClass *RCA, unsigned SubA,
                       const TargetRegisterClass *RCB, unsigned SubB,
                       unsigned &PreA, unsigned &PreB) const {
  assert(RCA && SubA && RCB && SubB && "Invalid arguments");

  // Search all pairs of sub-register indices that project into RCA and RCB
  // respectively. This is quadratic, but usually the sets are very small. On
  // most targets like X86, there will only be a single sub-register index
  // (e.g., sub_16bit projecting into GR16).
  //
  // The worst case is a register class like DPR on ARM.
  // We have indices dsub_0..dsub_7 projecting into that class.
  //
  // It is very common that one register class is a sub-register of the other.
  // Arrange for RCA to be the larger register so the answer will be found in
  // the first iteration. This makes the search linear for the most common
  // case.
  const TargetRegisterClass *BestRC = nullptr;
  unsigned *BestPreA = &PreA;
  unsigned *BestPreB = &PreB;
  if (RCA->getSize() < RCB->getSize()) {
    std::swap(RCA, RCB);
    std::swap(SubA, SubB);
    std::swap(BestPreA, BestPreB);
  }

  // Also terminate the search one we have found a register class as small as
  // RCA.
  unsigned MinSize = RCA->getSize();

  for (SuperRegClassIterator IA(RCA, this, true); IA.isValid(); ++IA) {
    unsigned FinalA = composeSubRegIndices(IA.getSubReg(), SubA);
    for (SuperRegClassIterator IB(RCB, this, true); IB.isValid(); ++IB) {
      // Check if a common super-register class exists for this index pair.
      const TargetRegisterClass *RC =
        firstCommonClass(IA.getMask(), IB.getMask(), this);
      if (!RC || RC->getSize() < MinSize)
        continue;

      // The indexes must compose identically: PreA+SubA == PreB+SubB.
      unsigned FinalB = composeSubRegIndices(IB.getSubReg(), SubB);
      if (FinalA != FinalB)
        continue;

      // Is RC a better candidate than BestRC?
      if (BestRC && RC->getSize() >= BestRC->getSize())
        continue;

      // Yes, RC is the smallest super-register seen so far.
      BestRC = RC;
      *BestPreA = IA.getSubReg();
      *BestPreB = IB.getSubReg();

      // Bail early if we reached MinSize. We won't find a better candidate.
      if (BestRC->getSize() == MinSize)
        return BestRC;
    }
  }
  return BestRC;
}

// Compute target-independent register allocator hints to help eliminate copies.
void
TargetRegisterInfo::getRegAllocationHints(unsigned VirtReg,
                                          ArrayRef<MCPhysReg> Order,
                                          SmallVectorImpl<MCPhysReg> &Hints,
                                          const MachineFunction &MF,
                                          const VirtRegMap *VRM) const {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  std::pair<unsigned, unsigned> Hint = MRI.getRegAllocationHint(VirtReg);

  // Hints with HintType != 0 were set by target-dependent code.
  // Such targets must provide their own implementation of
  // TRI::getRegAllocationHints to interpret those hint types.
  assert(Hint.first == 0 && "Target must implement TRI::getRegAllocationHints");

  // Target-independent hints are either a physical or a virtual register.
  unsigned Phys = Hint.second;
  if (VRM && isVirtualRegister(Phys))
    Phys = VRM->getPhys(Phys);

  // Check that Phys is a valid hint in VirtReg's register class.
  if (!isPhysicalRegister(Phys))
    return;
  if (MRI.isReserved(Phys))
    return;
  // Check that Phys is in the allocation order. We shouldn't heed hints
  // from VirtReg's register class if they aren't in the allocation order. The
  // target probably has a reason for removing the register.
  if (std::find(Order.begin(), Order.end(), Phys) == Order.end())
    return;

  // All clear, tell the register allocator to prefer this register.
  Hints.push_back(Phys);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void
TargetRegisterInfo::dumpReg(unsigned Reg, unsigned SubRegIndex,
                            const TargetRegisterInfo *TRI) {
  dbgs() << PrintReg(Reg, TRI, SubRegIndex) << "\n";
}
#endif
