//=== Target/TargetRegisterInfo.h - Target Register Information -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes an abstract interface used to get information about a
// target machines register file.  This information is used for a variety of
// purposed, especially register allocation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETREGISTERINFO_H
#define LLVM_TARGET_TARGETREGISTERINFO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineValueType.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/MC/MCRegisterInfo.h"
#include <cassert>
#include <functional>

namespace llvm {

class BitVector;
class MachineFunction;
class RegScavenger;
template<class T> class SmallVectorImpl;
class VirtRegMap;
class raw_ostream;

class TargetRegisterClass {
public:
  typedef const MCPhysReg* iterator;
  typedef const MCPhysReg* const_iterator;
  typedef const MVT::SimpleValueType* vt_iterator;
  typedef const TargetRegisterClass* const * sc_iterator;

  // Instance variables filled by tablegen, do not use!
  const MCRegisterClass *MC;
  const vt_iterator VTs;
  const uint32_t *SubClassMask;
  const uint16_t *SuperRegIndices;
  const unsigned LaneMask;
  /// Classes with a higher priority value are assigned first by register
  /// allocators using a greedy heuristic. The value is in the range [0,63].
  const uint8_t AllocationPriority;
  /// Whether the class supports two (or more) disjunct subregister indices.
  const bool HasDisjunctSubRegs;
  const sc_iterator SuperClasses;
  ArrayRef<MCPhysReg> (*OrderFunc)(const MachineFunction&);

  /// getID() - Return the register class ID number.
  ///
  unsigned getID() const { return MC->getID(); }

  /// begin/end - Return all of the registers in this class.
  ///
  iterator       begin() const { return MC->begin(); }
  iterator         end() const { return MC->end(); }

  /// getNumRegs - Return the number of registers in this class.
  ///
  unsigned getNumRegs() const { return MC->getNumRegs(); }

  /// getRegister - Return the specified register in the class.
  ///
  unsigned getRegister(unsigned i) const {
    return MC->getRegister(i);
  }

  /// contains - Return true if the specified register is included in this
  /// register class.  This does not include virtual registers.
  bool contains(unsigned Reg) const {
    return MC->contains(Reg);
  }

  /// contains - Return true if both registers are in this class.
  bool contains(unsigned Reg1, unsigned Reg2) const {
    return MC->contains(Reg1, Reg2);
  }

  /// getSize - Return the size of the register in bytes, which is also the size
  /// of a stack slot allocated to hold a spilled copy of this register.
  unsigned getSize() const { return MC->getSize(); }

  /// getAlignment - Return the minimum required alignment for a register of
  /// this class.
  unsigned getAlignment() const { return MC->getAlignment(); }

  /// getCopyCost - Return the cost of copying a value between two registers in
  /// this class. A negative number means the register class is very expensive
  /// to copy e.g. status flag register classes.
  int getCopyCost() const { return MC->getCopyCost(); }

  /// isAllocatable - Return true if this register class may be used to create
  /// virtual registers.
  bool isAllocatable() const { return MC->isAllocatable(); }

  /// hasType - return true if this TargetRegisterClass has the ValueType vt.
  ///
  bool hasType(MVT vt) const {
    for(int i = 0; VTs[i] != MVT::Other; ++i)
      if (MVT(VTs[i]) == vt)
        return true;
    return false;
  }

  /// vt_begin / vt_end - Loop over all of the value types that can be
  /// represented by values in this register class.
  vt_iterator vt_begin() const {
    return VTs;
  }

  vt_iterator vt_end() const {
    vt_iterator I = VTs;
    while (*I != MVT::Other) ++I;
    return I;
  }

  /// hasSubClass - return true if the specified TargetRegisterClass
  /// is a proper sub-class of this TargetRegisterClass.
  bool hasSubClass(const TargetRegisterClass *RC) const {
    return RC != this && hasSubClassEq(RC);
  }

  /// hasSubClassEq - Returns true if RC is a sub-class of or equal to this
  /// class.
  bool hasSubClassEq(const TargetRegisterClass *RC) const {
    unsigned ID = RC->getID();
    return (SubClassMask[ID / 32] >> (ID % 32)) & 1;
  }

  /// hasSuperClass - return true if the specified TargetRegisterClass is a
  /// proper super-class of this TargetRegisterClass.
  bool hasSuperClass(const TargetRegisterClass *RC) const {
    return RC->hasSubClass(this);
  }

  /// hasSuperClassEq - Returns true if RC is a super-class of or equal to this
  /// class.
  bool hasSuperClassEq(const TargetRegisterClass *RC) const {
    return RC->hasSubClassEq(this);
  }

  /// getSubClassMask - Returns a bit vector of subclasses, including this one.
  /// The vector is indexed by class IDs, see hasSubClassEq() above for how to
  /// use it.
  const uint32_t *getSubClassMask() const {
    return SubClassMask;
  }

  /// getSuperRegIndices - Returns a 0-terminated list of sub-register indices
  /// that project some super-register class into this register class. The list
  /// has an entry for each Idx such that:
  ///
  ///   There exists SuperRC where:
  ///     For all Reg in SuperRC:
  ///       this->contains(Reg:Idx)
  ///
  const uint16_t *getSuperRegIndices() const {
    return SuperRegIndices;
  }

  /// getSuperClasses - Returns a NULL terminated list of super-classes.  The
  /// classes are ordered by ID which is also a topological ordering from large
  /// to small classes.  The list does NOT include the current class.
  sc_iterator getSuperClasses() const {
    return SuperClasses;
  }

  /// isASubClass - return true if this TargetRegisterClass is a subset
  /// class of at least one other TargetRegisterClass.
  bool isASubClass() const {
    return SuperClasses[0] != nullptr;
  }

  /// getRawAllocationOrder - Returns the preferred order for allocating
  /// registers from this register class in MF. The raw order comes directly
  /// from the .td file and may include reserved registers that are not
  /// allocatable. Register allocators should also make sure to allocate
  /// callee-saved registers only after all the volatiles are used. The
  /// RegisterClassInfo class provides filtered allocation orders with
  /// callee-saved registers moved to the end.
  ///
  /// The MachineFunction argument can be used to tune the allocatable
  /// registers based on the characteristics of the function, subtarget, or
  /// other criteria.
  ///
  /// By default, this method returns all registers in the class.
  ///
  ArrayRef<MCPhysReg> getRawAllocationOrder(const MachineFunction &MF) const {
    return OrderFunc ? OrderFunc(MF) : makeArrayRef(begin(), getNumRegs());
  }

  /// Returns the combination of all lane masks of register in this class.
  /// The lane masks of the registers are the combination of all lane masks
  /// of their subregisters.
  unsigned getLaneMask() const {
    return LaneMask;
  }
};

/// TargetRegisterInfoDesc - Extra information, not in MCRegisterDesc, about
/// registers. These are used by codegen, not by MC.
struct TargetRegisterInfoDesc {
  unsigned CostPerUse;          // Extra cost of instructions using register.
  bool inAllocatableClass;      // Register belongs to an allocatable regclass.
};

/// Each TargetRegisterClass has a per register weight, and weight
/// limit which must be less than the limits of its pressure sets.
struct RegClassWeight {
  unsigned RegWeight;
  unsigned WeightLimit;
};

/// TargetRegisterInfo base class - We assume that the target defines a static
/// array of TargetRegisterDesc objects that represent all of the machine
/// registers that the target has.  As such, we simply have to track a pointer
/// to this array so that we can turn register number into a register
/// descriptor.
///
class TargetRegisterInfo : public MCRegisterInfo {
public:
  typedef const TargetRegisterClass * const * regclass_iterator;
private:
  const TargetRegisterInfoDesc *InfoDesc;     // Extra desc array for codegen
  const char *const *SubRegIndexNames;        // Names of subreg indexes.
  // Pointer to array of lane masks, one per sub-reg index.
  const unsigned *SubRegIndexLaneMasks;

  regclass_iterator RegClassBegin, RegClassEnd;   // List of regclasses
  unsigned CoveringLanes;

protected:
  TargetRegisterInfo(const TargetRegisterInfoDesc *ID,
                     regclass_iterator RegClassBegin,
                     regclass_iterator RegClassEnd,
                     const char *const *SRINames,
                     const unsigned *SRILaneMasks,
                     unsigned CoveringLanes);
  virtual ~TargetRegisterInfo();
public:

  // Register numbers can represent physical registers, virtual registers, and
  // sometimes stack slots. The unsigned values are divided into these ranges:
  //
  //   0           Not a register, can be used as a sentinel.
  //   [1;2^30)    Physical registers assigned by TableGen.
  //   [2^30;2^31) Stack slots. (Rarely used.)
  //   [2^31;2^32) Virtual registers assigned by MachineRegisterInfo.
  //
  // Further sentinels can be allocated from the small negative integers.
  // DenseMapInfo<unsigned> uses -1u and -2u.

  /// isStackSlot - Sometimes it is useful the be able to store a non-negative
  /// frame index in a variable that normally holds a register. isStackSlot()
  /// returns true if Reg is in the range used for stack slots.
  ///
  /// Note that isVirtualRegister() and isPhysicalRegister() cannot handle stack
  /// slots, so if a variable may contains a stack slot, always check
  /// isStackSlot() first.
  ///
  static bool isStackSlot(unsigned Reg) {
    return int(Reg) >= (1 << 30);
  }

  /// stackSlot2Index - Compute the frame index from a register value
  /// representing a stack slot.
  static int stackSlot2Index(unsigned Reg) {
    assert(isStackSlot(Reg) && "Not a stack slot");
    return int(Reg - (1u << 30));
  }

  /// index2StackSlot - Convert a non-negative frame index to a stack slot
  /// register value.
  static unsigned index2StackSlot(int FI) {
    assert(FI >= 0 && "Cannot hold a negative frame index.");
    return FI + (1u << 30);
  }

  /// isPhysicalRegister - Return true if the specified register number is in
  /// the physical register namespace.
  static bool isPhysicalRegister(unsigned Reg) {
    assert(!isStackSlot(Reg) && "Not a register! Check isStackSlot() first.");
    return int(Reg) > 0;
  }

  /// isVirtualRegister - Return true if the specified register number is in
  /// the virtual register namespace.
  static bool isVirtualRegister(unsigned Reg) {
    assert(!isStackSlot(Reg) && "Not a register! Check isStackSlot() first.");
    return int(Reg) < 0;
  }

  /// virtReg2Index - Convert a virtual register number to a 0-based index.
  /// The first virtual register in a function will get the index 0.
  static unsigned virtReg2Index(unsigned Reg) {
    assert(isVirtualRegister(Reg) && "Not a virtual register");
    return Reg & ~(1u << 31);
  }

  /// index2VirtReg - Convert a 0-based index to a virtual register number.
  /// This is the inverse operation of VirtReg2IndexFunctor below.
  static unsigned index2VirtReg(unsigned Index) {
    return Index | (1u << 31);
  }

  /// getMinimalPhysRegClass - Returns the Register Class of a physical
  /// register of the given type, picking the most sub register class of
  /// the right type that contains this physreg.
  const TargetRegisterClass *
    getMinimalPhysRegClass(unsigned Reg, MVT VT = MVT::Other) const;

  /// getAllocatableClass - Return the maximal subclass of the given register
  /// class that is alloctable, or NULL.
  const TargetRegisterClass *
    getAllocatableClass(const TargetRegisterClass *RC) const;

  /// getAllocatableSet - Returns a bitset indexed by register number
  /// indicating if a register is allocatable or not. If a register class is
  /// specified, returns the subset for the class.
  BitVector getAllocatableSet(const MachineFunction &MF,
                              const TargetRegisterClass *RC = nullptr) const;

  /// getCostPerUse - Return the additional cost of using this register instead
  /// of other registers in its class.
  unsigned getCostPerUse(unsigned RegNo) const {
    return InfoDesc[RegNo].CostPerUse;
  }

  /// isInAllocatableClass - Return true if the register is in the allocation
  /// of any register class.
  bool isInAllocatableClass(unsigned RegNo) const {
    return InfoDesc[RegNo].inAllocatableClass;
  }

  /// getSubRegIndexName - Return the human-readable symbolic target-specific
  /// name for the specified SubRegIndex.
  const char *getSubRegIndexName(unsigned SubIdx) const {
    assert(SubIdx && SubIdx < getNumSubRegIndices() &&
           "This is not a subregister index");
    return SubRegIndexNames[SubIdx-1];
  }

  /// getSubRegIndexLaneMask - Return a bitmask representing the parts of a
  /// register that are covered by SubIdx.
  ///
  /// Lane masks for sub-register indices are similar to register units for
  /// physical registers. The individual bits in a lane mask can't be assigned
  /// any specific meaning. They can be used to check if two sub-register
  /// indices overlap.
  ///
  /// If the target has a register such that:
  ///
  ///   getSubReg(Reg, A) overlaps getSubReg(Reg, B)
  ///
  /// then:
  ///
  ///   (getSubRegIndexLaneMask(A) & getSubRegIndexLaneMask(B)) != 0
  ///
  /// The converse is not necessarily true. If two lane masks have a common
  /// bit, the corresponding sub-registers may not overlap, but it can be
  /// assumed that they usually will.
  /// SubIdx == 0 is allowed, it has the lane mask ~0u.
  unsigned getSubRegIndexLaneMask(unsigned SubIdx) const {
    assert(SubIdx < getNumSubRegIndices() && "This is not a subregister index");
    return SubRegIndexLaneMasks[SubIdx];
  }

  /// Returns true if the given lane mask is imprecise.
  ///
  /// LaneMasks as given by getSubRegIndexLaneMask() have a limited number of
  /// bits, so for targets with more than 31 disjunct subregister indices there
  /// may be cases where:
  ///    getSubReg(Reg,A) does not overlap getSubReg(Reg,B)
  /// but we still have
  ///    (getSubRegIndexLaneMask(A) & getSubRegIndexLaneMask(B)) != 0.
  /// This function returns true in those cases.
  static bool isImpreciseLaneMask(unsigned LaneMask) {
    return LaneMask & 0x80000000u;
  }

  /// The lane masks returned by getSubRegIndexLaneMask() above can only be
  /// used to determine if sub-registers overlap - they can't be used to
  /// determine if a set of sub-registers completely cover another
  /// sub-register.
  ///
  /// The X86 general purpose registers have two lanes corresponding to the
  /// sub_8bit and sub_8bit_hi sub-registers. Both sub_32bit and sub_16bit have
  /// lane masks '3', but the sub_16bit sub-register doesn't fully cover the
  /// sub_32bit sub-register.
  ///
  /// On the other hand, the ARM NEON lanes fully cover their registers: The
  /// dsub_0 sub-register is completely covered by the ssub_0 and ssub_1 lanes.
  /// This is related to the CoveredBySubRegs property on register definitions.
  ///
  /// This function returns a bit mask of lanes that completely cover their
  /// sub-registers. More precisely, given:
  ///
  ///   Covering = getCoveringLanes();
  ///   MaskA = getSubRegIndexLaneMask(SubA);
  ///   MaskB = getSubRegIndexLaneMask(SubB);
  ///
  /// If (MaskA & ~(MaskB & Covering)) == 0, then SubA is completely covered by
  /// SubB.
  unsigned getCoveringLanes() const { return CoveringLanes; }

  /// regsOverlap - Returns true if the two registers are equal or alias each
  /// other. The registers may be virtual register.
  bool regsOverlap(unsigned regA, unsigned regB) const {
    if (regA == regB) return true;
    if (isVirtualRegister(regA) || isVirtualRegister(regB))
      return false;

    // Regunits are numerically ordered. Find a common unit.
    MCRegUnitIterator RUA(regA, this);
    MCRegUnitIterator RUB(regB, this);
    do {
      if (*RUA == *RUB) return true;
      if (*RUA < *RUB) ++RUA;
      else             ++RUB;
    } while (RUA.isValid() && RUB.isValid());
    return false;
  }

  /// hasRegUnit - Returns true if Reg contains RegUnit.
  bool hasRegUnit(unsigned Reg, unsigned RegUnit) const {
    for (MCRegUnitIterator Units(Reg, this); Units.isValid(); ++Units)
      if (*Units == RegUnit)
        return true;
    return false;
  }

  /// getCalleeSavedRegs - Return a null-terminated list of all of the
  /// callee saved registers on this target. The register should be in the
  /// order of desired callee-save stack frame offset. The first register is
  /// closest to the incoming stack pointer if stack grows down, and vice versa.
  ///
  virtual const MCPhysReg*
  getCalleeSavedRegs(const MachineFunction *MF) const = 0;

  /// getCallPreservedMask - Return a mask of call-preserved registers for the
  /// given calling convention on the current function.  The mask should
  /// include all call-preserved aliases.  This is used by the register
  /// allocator to determine which registers can be live across a call.
  ///
  /// The mask is an array containing (TRI::getNumRegs()+31)/32 entries.
  /// A set bit indicates that all bits of the corresponding register are
  /// preserved across the function call.  The bit mask is expected to be
  /// sub-register complete, i.e. if A is preserved, so are all its
  /// sub-registers.
  ///
  /// Bits are numbered from the LSB, so the bit for physical register Reg can
  /// be found as (Mask[Reg / 32] >> Reg % 32) & 1.
  ///
  /// A NULL pointer means that no register mask will be used, and call
  /// instructions should use implicit-def operands to indicate call clobbered
  /// registers.
  ///
  virtual const uint32_t *getCallPreservedMask(const MachineFunction &MF,
                                               CallingConv::ID) const {
    // The default mask clobbers everything.  All targets should override.
    return nullptr;
  }

  /// Return all the call-preserved register masks defined for this target.
  virtual ArrayRef<const uint32_t *> getRegMasks() const = 0;
  virtual ArrayRef<const char *> getRegMaskNames() const = 0;

  /// getReservedRegs - Returns a bitset indexed by physical register number
  /// indicating if a register is a special register that has particular uses
  /// and should be considered unavailable at all times, e.g. SP, RA. This is
  /// used by register scavenger to determine what registers are free.
  virtual BitVector getReservedRegs(const MachineFunction &MF) const = 0;

  /// Prior to adding the live-out mask to a stackmap or patchpoint
  /// instruction, provide the target the opportunity to adjust it (mainly to
  /// remove pseudo-registers that should be ignored).
  virtual void adjustStackMapLiveOutMask(uint32_t *Mask) const { }

  /// getMatchingSuperReg - Return a super-register of the specified register
  /// Reg so its sub-register of index SubIdx is Reg.
  unsigned getMatchingSuperReg(unsigned Reg, unsigned SubIdx,
                               const TargetRegisterClass *RC) const {
    return MCRegisterInfo::getMatchingSuperReg(Reg, SubIdx, RC->MC);
  }

  /// getMatchingSuperRegClass - Return a subclass of the specified register
  /// class A so that each register in it has a sub-register of the
  /// specified sub-register index which is in the specified register class B.
  ///
  /// TableGen will synthesize missing A sub-classes.
  virtual const TargetRegisterClass *
  getMatchingSuperRegClass(const TargetRegisterClass *A,
                           const TargetRegisterClass *B, unsigned Idx) const;

  /// getSubClassWithSubReg - Returns the largest legal sub-class of RC that
  /// supports the sub-register index Idx.
  /// If no such sub-class exists, return NULL.
  /// If all registers in RC already have an Idx sub-register, return RC.
  ///
  /// TableGen generates a version of this function that is good enough in most
  /// cases.  Targets can override if they have constraints that TableGen
  /// doesn't understand.  For example, the x86 sub_8bit sub-register index is
  /// supported by the full GR32 register class in 64-bit mode, but only by the
  /// GR32_ABCD regiister class in 32-bit mode.
  ///
  /// TableGen will synthesize missing RC sub-classes.
  virtual const TargetRegisterClass *
  getSubClassWithSubReg(const TargetRegisterClass *RC, unsigned Idx) const {
    assert(Idx == 0 && "Target has no sub-registers");
    return RC;
  }

  /// composeSubRegIndices - Return the subregister index you get from composing
  /// two subregister indices.
  ///
  /// The special null sub-register index composes as the identity.
  ///
  /// If R:a:b is the same register as R:c, then composeSubRegIndices(a, b)
  /// returns c. Note that composeSubRegIndices does not tell you about illegal
  /// compositions. If R does not have a subreg a, or R:a does not have a subreg
  /// b, composeSubRegIndices doesn't tell you.
  ///
  /// The ARM register Q0 has two D subregs dsub_0:D0 and dsub_1:D1. It also has
  /// ssub_0:S0 - ssub_3:S3 subregs.
  /// If you compose subreg indices dsub_1, ssub_0 you get ssub_2.
  ///
  unsigned composeSubRegIndices(unsigned a, unsigned b) const {
    if (!a) return b;
    if (!b) return a;
    return composeSubRegIndicesImpl(a, b);
  }

  /// Transforms a LaneMask computed for one subregister to the lanemask that
  /// would have been computed when composing the subsubregisters with IdxA
  /// first. @sa composeSubRegIndices()
  unsigned composeSubRegIndexLaneMask(unsigned IdxA, unsigned LaneMask) const {
    if (!IdxA)
      return LaneMask;
    return composeSubRegIndexLaneMaskImpl(IdxA, LaneMask);
  }

  /// Debugging helper: dump register in human readable form to dbgs() stream.
  static void dumpReg(unsigned Reg, unsigned SubRegIndex = 0,
                      const TargetRegisterInfo* TRI = nullptr);

protected:
  /// Overridden by TableGen in targets that have sub-registers.
  virtual unsigned composeSubRegIndicesImpl(unsigned, unsigned) const {
    llvm_unreachable("Target has no sub-registers");
  }

  /// Overridden by TableGen in targets that have sub-registers.
  virtual unsigned
  composeSubRegIndexLaneMaskImpl(unsigned, unsigned) const {
    llvm_unreachable("Target has no sub-registers");
  }

public:
  /// getCommonSuperRegClass - Find a common super-register class if it exists.
  ///
  /// Find a register class, SuperRC and two sub-register indices, PreA and
  /// PreB, such that:
  ///
  ///   1. PreA + SubA == PreB + SubB  (using composeSubRegIndices()), and
  ///
  ///   2. For all Reg in SuperRC: Reg:PreA in RCA and Reg:PreB in RCB, and
  ///
  ///   3. SuperRC->getSize() >= max(RCA->getSize(), RCB->getSize()).
  ///
  /// SuperRC will be chosen such that no super-class of SuperRC satisfies the
  /// requirements, and there is no register class with a smaller spill size
  /// that satisfies the requirements.
  ///
  /// SubA and SubB must not be 0. Use getMatchingSuperRegClass() instead.
  ///
  /// Either of the PreA and PreB sub-register indices may be returned as 0. In
  /// that case, the returned register class will be a sub-class of the
  /// corresponding argument register class.
  ///
  /// The function returns NULL if no register class can be found.
  ///
  const TargetRegisterClass*
  getCommonSuperRegClass(const TargetRegisterClass *RCA, unsigned SubA,
                         const TargetRegisterClass *RCB, unsigned SubB,
                         unsigned &PreA, unsigned &PreB) const;

  //===--------------------------------------------------------------------===//
  // Register Class Information
  //

  /// Register class iterators
  ///
  regclass_iterator regclass_begin() const { return RegClassBegin; }
  regclass_iterator regclass_end() const { return RegClassEnd; }

  unsigned getNumRegClasses() const {
    return (unsigned)(regclass_end()-regclass_begin());
  }

  /// getRegClass - Returns the register class associated with the enumeration
  /// value.  See class MCOperandInfo.
  const TargetRegisterClass *getRegClass(unsigned i) const {
    assert(i < getNumRegClasses() && "Register Class ID out of range");
    return RegClassBegin[i];
  }

  /// getRegClassName - Returns the name of the register class.
  const char *getRegClassName(const TargetRegisterClass *Class) const {
    return MCRegisterInfo::getRegClassName(Class->MC);
  }

  /// getCommonSubClass - find the largest common subclass of A and B. Return
  /// NULL if there is no common subclass.
  const TargetRegisterClass *
  getCommonSubClass(const TargetRegisterClass *A,
                    const TargetRegisterClass *B) const;

  /// getPointerRegClass - Returns a TargetRegisterClass used for pointer
  /// values.  If a target supports multiple different pointer register classes,
  /// kind specifies which one is indicated.
  virtual const TargetRegisterClass *
  getPointerRegClass(const MachineFunction &MF, unsigned Kind=0) const {
    llvm_unreachable("Target didn't implement getPointerRegClass!");
  }

  /// getCrossCopyRegClass - Returns a legal register class to copy a register
  /// in the specified class to or from. If it is possible to copy the register
  /// directly without using a cross register class copy, return the specified
  /// RC. Returns NULL if it is not possible to copy between a two registers of
  /// the specified class.
  virtual const TargetRegisterClass *
  getCrossCopyRegClass(const TargetRegisterClass *RC) const {
    return RC;
  }

  /// getLargestLegalSuperClass - Returns the largest super class of RC that is
  /// legal to use in the current sub-target and has the same spill size.
  /// The returned register class can be used to create virtual registers which
  /// means that all its registers can be copied and spilled.
  virtual const TargetRegisterClass *
  getLargestLegalSuperClass(const TargetRegisterClass *RC,
                            const MachineFunction &) const {
    /// The default implementation is very conservative and doesn't allow the
    /// register allocator to inflate register classes.
    return RC;
  }

  /// getRegPressureLimit - Return the register pressure "high water mark" for
  /// the specific register class. The scheduler is in high register pressure
  /// mode (for the specific register class) if it goes over the limit.
  ///
  /// Note: this is the old register pressure model that relies on a manually
  /// specified representative register class per value type.
  virtual unsigned getRegPressureLimit(const TargetRegisterClass *RC,
                                       MachineFunction &MF) const {
    return 0;
  }

  /// Get the weight in units of pressure for this register class.
  virtual const RegClassWeight &getRegClassWeight(
    const TargetRegisterClass *RC) const = 0;

  /// Get the weight in units of pressure for this register unit.
  virtual unsigned getRegUnitWeight(unsigned RegUnit) const = 0;

  /// Get the number of dimensions of register pressure.
  virtual unsigned getNumRegPressureSets() const = 0;

  /// Get the name of this register unit pressure set.
  virtual const char *getRegPressureSetName(unsigned Idx) const = 0;

  /// Get the register unit pressure limit for this dimension.
  /// This limit must be adjusted dynamically for reserved registers.
  virtual unsigned getRegPressureSetLimit(const MachineFunction &MF,
                                          unsigned Idx) const = 0;

  /// Get the dimensions of register pressure impacted by this register class.
  /// Returns a -1 terminated array of pressure set IDs.
  virtual const int *getRegClassPressureSets(
    const TargetRegisterClass *RC) const = 0;

  /// Get the dimensions of register pressure impacted by this register unit.
  /// Returns a -1 terminated array of pressure set IDs.
  virtual const int *getRegUnitPressureSets(unsigned RegUnit) const = 0;

  /// Get a list of 'hint' registers that the register allocator should try
  /// first when allocating a physical register for the virtual register
  /// VirtReg. These registers are effectively moved to the front of the
  /// allocation order.
  ///
  /// The Order argument is the allocation order for VirtReg's register class
  /// as returned from RegisterClassInfo::getOrder(). The hint registers must
  /// come from Order, and they must not be reserved.
  ///
  /// The default implementation of this function can resolve
  /// target-independent hints provided to MRI::setRegAllocationHint with
  /// HintType == 0. Targets that override this function should defer to the
  /// default implementation if they have no reason to change the allocation
  /// order for VirtReg. There may be target-independent hints.
  virtual void getRegAllocationHints(unsigned VirtReg,
                                     ArrayRef<MCPhysReg> Order,
                                     SmallVectorImpl<MCPhysReg> &Hints,
                                     const MachineFunction &MF,
                                     const VirtRegMap *VRM = nullptr) const;

  /// updateRegAllocHint - A callback to allow target a chance to update
  /// register allocation hints when a register is "changed" (e.g. coalesced)
  /// to another register. e.g. On ARM, some virtual registers should target
  /// register pairs, if one of pair is coalesced to another register, the
  /// allocation hint of the other half of the pair should be changed to point
  /// to the new register.
  virtual void updateRegAllocHint(unsigned Reg, unsigned NewReg,
                                  MachineFunction &MF) const {
    // Do nothing.
  }

  /// Allow the target to reverse allocation order of local live ranges. This
  /// will generally allocate shorter local live ranges first. For targets with
  /// many registers, this could reduce regalloc compile time by a large
  /// factor. It is disabled by default for three reasons:
  /// (1) Top-down allocation is simpler and easier to debug for targets that
  /// don't benefit from reversing the order.
  /// (2) Bottom-up allocation could result in poor evicition decisions on some
  /// targets affecting the performance of compiled code.
  /// (3) Bottom-up allocation is no longer guaranteed to optimally color.
  virtual bool reverseLocalAssignment() const { return false; }

  /// Allow the target to override the cost of using a callee-saved register for
  /// the first time. Default value of 0 means we will use a callee-saved
  /// register if it is available.
  virtual unsigned getCSRFirstUseCost() const { return 0; }

  /// requiresRegisterScavenging - returns true if the target requires (and can
  /// make use of) the register scavenger.
  virtual bool requiresRegisterScavenging(const MachineFunction &MF) const {
    return false;
  }

  /// useFPForScavengingIndex - returns true if the target wants to use
  /// frame pointer based accesses to spill to the scavenger emergency spill
  /// slot.
  virtual bool useFPForScavengingIndex(const MachineFunction &MF) const {
    return true;
  }

  /// requiresFrameIndexScavenging - returns true if the target requires post
  /// PEI scavenging of registers for materializing frame index constants.
  virtual bool requiresFrameIndexScavenging(const MachineFunction &MF) const {
    return false;
  }

  /// requiresVirtualBaseRegisters - Returns true if the target wants the
  /// LocalStackAllocation pass to be run and virtual base registers
  /// used for more efficient stack access.
  virtual bool requiresVirtualBaseRegisters(const MachineFunction &MF) const {
    return false;
  }

  /// hasReservedSpillSlot - Return true if target has reserved a spill slot in
  /// the stack frame of the given function for the specified register. e.g. On
  /// x86, if the frame register is required, the first fixed stack object is
  /// reserved as its spill slot. This tells PEI not to create a new stack frame
  /// object for the given register. It should be called only after
  /// processFunctionBeforeCalleeSavedScan().
  virtual bool hasReservedSpillSlot(const MachineFunction &MF, unsigned Reg,
                                    int &FrameIdx) const {
    return false;
  }

  /// trackLivenessAfterRegAlloc - returns true if the live-ins should be tracked
  /// after register allocation.
  virtual bool trackLivenessAfterRegAlloc(const MachineFunction &MF) const {
    return false;
  }

  /// needsStackRealignment - true if storage within the function requires the
  /// stack pointer to be aligned more than the normal calling convention calls
  /// for.
  virtual bool needsStackRealignment(const MachineFunction &MF) const {
    return false;
  }

  /// getFrameIndexInstrOffset - Get the offset from the referenced frame
  /// index in the instruction, if there is one.
  virtual int64_t getFrameIndexInstrOffset(const MachineInstr *MI,
                                           int Idx) const {
    return 0;
  }

  /// needsFrameBaseReg - Returns true if the instruction's frame index
  /// reference would be better served by a base register other than FP
  /// or SP. Used by LocalStackFrameAllocation to determine which frame index
  /// references it should create new base registers for.
  virtual bool needsFrameBaseReg(MachineInstr *MI, int64_t Offset) const {
    return false;
  }

  /// materializeFrameBaseRegister - Insert defining instruction(s) for
  /// BaseReg to be a pointer to FrameIdx before insertion point I.
  virtual void materializeFrameBaseRegister(MachineBasicBlock *MBB,
                                            unsigned BaseReg, int FrameIdx,
                                            int64_t Offset) const {
    llvm_unreachable("materializeFrameBaseRegister does not exist on this "
                     "target");
  }

  /// resolveFrameIndex - Resolve a frame index operand of an instruction
  /// to reference the indicated base register plus offset instead.
  virtual void resolveFrameIndex(MachineInstr &MI, unsigned BaseReg,
                                 int64_t Offset) const {
    llvm_unreachable("resolveFrameIndex does not exist on this target");
  }

  /// isFrameOffsetLegal - Determine whether a given base register plus offset
  /// immediate is encodable to resolve a frame index.
  virtual bool isFrameOffsetLegal(const MachineInstr *MI, unsigned BaseReg,
                                  int64_t Offset) const {
    llvm_unreachable("isFrameOffsetLegal does not exist on this target");
  }


  /// saveScavengerRegister - Spill the register so it can be used by the
  /// register scavenger. Return true if the register was spilled, false
  /// otherwise. If this function does not spill the register, the scavenger
  /// will instead spill it to the emergency spill slot.
  ///
  virtual bool saveScavengerRegister(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I,
                                     MachineBasicBlock::iterator &UseMI,
                                     const TargetRegisterClass *RC,
                                     unsigned Reg) const {
    return false;
  }

  /// eliminateFrameIndex - This method must be overriden to eliminate abstract
  /// frame indices from instructions which may use them.  The instruction
  /// referenced by the iterator contains an MO_FrameIndex operand which must be
  /// eliminated by this method.  This method may modify or replace the
  /// specified instruction, as long as it keeps the iterator pointing at the
  /// finished product.  SPAdj is the SP adjustment due to call frame setup
  /// instruction.  FIOperandNum is the FI operand number.
  virtual void eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                   int SPAdj, unsigned FIOperandNum,
                                   RegScavenger *RS = nullptr) const = 0;

  //===--------------------------------------------------------------------===//
  /// Subtarget Hooks

  /// \brief SrcRC and DstRC will be morphed into NewRC if this returns true.
  virtual bool shouldCoalesce(MachineInstr *MI,
                              const TargetRegisterClass *SrcRC,
                              unsigned SubReg,
                              const TargetRegisterClass *DstRC,
                              unsigned DstSubReg,
                              const TargetRegisterClass *NewRC) const
  { return true; }

  //===--------------------------------------------------------------------===//
  /// Debug information queries.

  /// getFrameRegister - This method should return the register used as a base
  /// for values allocated in the current stack frame.
  virtual unsigned getFrameRegister(const MachineFunction &MF) const = 0;
};


//===----------------------------------------------------------------------===//
//                           SuperRegClassIterator
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
//
// Iterate over the possible super-registers for a given register class. The
// iterator will visit a list of pairs (Idx, Mask) corresponding to the
// possible classes of super-registers.
//
// Each bit mask will have at least one set bit, and each set bit in Mask
// corresponds to a SuperRC such that:
//
//   For all Reg in SuperRC: Reg:Idx is in RC.
//
// The iterator can include (O, RC->getSubClassMask()) as the first entry which
// also satisfies the above requirement, assuming Reg:0 == Reg.
//
class SuperRegClassIterator {
  const unsigned RCMaskWords;
  unsigned SubReg;
  const uint16_t *Idx;
  const uint32_t *Mask;

public:
  /// Create a SuperRegClassIterator that visits all the super-register classes
  /// of RC. When IncludeSelf is set, also include the (0, sub-classes) entry.
  SuperRegClassIterator(const TargetRegisterClass *RC,
                        const TargetRegisterInfo *TRI,
                        bool IncludeSelf = false)
    : RCMaskWords((TRI->getNumRegClasses() + 31) / 32),
      SubReg(0),
      Idx(RC->getSuperRegIndices()),
      Mask(RC->getSubClassMask()) {
    if (!IncludeSelf)
      ++*this;
  }

  /// Returns true if this iterator is still pointing at a valid entry.
  bool isValid() const { return Idx; }

  /// Returns the current sub-register index.
  unsigned getSubReg() const { return SubReg; }

  /// Returns the bit mask if register classes that getSubReg() projects into
  /// RC.
  const uint32_t *getMask() const { return Mask; }

  /// Advance iterator to the next entry.
  void operator++() {
    assert(isValid() && "Cannot move iterator past end.");
    Mask += RCMaskWords;
    SubReg = *Idx++;
    if (!SubReg)
      Idx = nullptr;
  }
};

// This is useful when building IndexedMaps keyed on virtual registers
struct VirtReg2IndexFunctor {
  unsigned operator()(unsigned Reg) const {
    return TargetRegisterInfo::virtReg2Index(Reg);
  }
};

/// PrintReg - Helper class for printing registers on a raw_ostream.
/// Prints virtual and physical registers with or without a TRI instance.
///
/// The format is:
///   %noreg          - NoRegister
///   %vreg5          - a virtual register.
///   %vreg5:sub_8bit - a virtual register with sub-register index (with TRI).
///   %EAX            - a physical register
///   %physreg17      - a physical register when no TRI instance given.
///
/// Usage: OS << PrintReg(Reg, TRI) << '\n';
///
class PrintReg {
  const TargetRegisterInfo *TRI;
  unsigned Reg;
  unsigned SubIdx;
public:
  explicit PrintReg(unsigned reg, const TargetRegisterInfo *tri = nullptr,
                    unsigned subidx = 0)
    : TRI(tri), Reg(reg), SubIdx(subidx) {}
  void print(raw_ostream&) const;
};

static inline raw_ostream &operator<<(raw_ostream &OS, const PrintReg &PR) {
  PR.print(OS);
  return OS;
}

/// PrintRegUnit - Helper class for printing register units on a raw_ostream.
///
/// Register units are named after their root registers:
///
///   AL      - Single root.
///   FP0~ST7 - Dual roots.
///
/// Usage: OS << PrintRegUnit(Unit, TRI) << '\n';
///
class PrintRegUnit {
protected:
  const TargetRegisterInfo *TRI;
  unsigned Unit;
public:
  PrintRegUnit(unsigned unit, const TargetRegisterInfo *tri)
    : TRI(tri), Unit(unit) {}
  void print(raw_ostream&) const;
};

static inline raw_ostream &operator<<(raw_ostream &OS, const PrintRegUnit &PR) {
  PR.print(OS);
  return OS;
}

/// PrintVRegOrUnit - It is often convenient to track virtual registers and
/// physical register units in the same list.
class PrintVRegOrUnit : protected PrintRegUnit {
public:
  PrintVRegOrUnit(unsigned VRegOrUnit, const TargetRegisterInfo *tri)
    : PrintRegUnit(VRegOrUnit, tri) {}
  void print(raw_ostream&) const;
};

static inline raw_ostream &operator<<(raw_ostream &OS,
                                      const PrintVRegOrUnit &PR) {
  PR.print(OS);
  return OS;
}

} // End llvm namespace

#endif
