//===-- llvm/CodeGen/MachineRegisterInfo.h ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MachineRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEREGISTERINFO_H
#define LLVM_CODEGEN_MACHINEREGISTERINFO_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <vector>

namespace llvm {
class PSetIterator;

/// MachineRegisterInfo - Keep track of information for virtual and physical
/// registers, including vreg register classes, use/def chains for registers,
/// etc.
class MachineRegisterInfo {
public:
  class Delegate {
    virtual void anchor();
  public:
    virtual void MRI_NoteNewVirtualRegister(unsigned Reg) = 0;

    virtual ~Delegate() {}
  };

private:
  const MachineFunction *MF;
  Delegate *TheDelegate;

  /// IsSSA - True when the machine function is in SSA form and virtual
  /// registers have a single def.
  bool IsSSA;

  /// TracksLiveness - True while register liveness is being tracked accurately.
  /// Basic block live-in lists, kill flags, and implicit defs may not be
  /// accurate when after this flag is cleared.
  bool TracksLiveness;

  /// True if subregister liveness is tracked.
  bool TracksSubRegLiveness;

  /// VRegInfo - Information we keep for each virtual register.
  ///
  /// Each element in this list contains the register class of the vreg and the
  /// start of the use/def list for the register.
  IndexedMap<std::pair<const TargetRegisterClass*, MachineOperand*>,
             VirtReg2IndexFunctor> VRegInfo;

  /// RegAllocHints - This vector records register allocation hints for virtual
  /// registers. For each virtual register, it keeps a register and hint type
  /// pair making up the allocation hint. Hint type is target specific except
  /// for the value 0 which means the second value of the pair is the preferred
  /// register for allocation. For example, if the hint is <0, 1024>, it means
  /// the allocator should prefer the physical register allocated to the virtual
  /// register of the hint.
  IndexedMap<std::pair<unsigned, unsigned>, VirtReg2IndexFunctor> RegAllocHints;

  /// PhysRegUseDefLists - This is an array of the head of the use/def list for
  /// physical registers.
  std::vector<MachineOperand *> PhysRegUseDefLists;

  /// getRegUseDefListHead - Return the head pointer for the register use/def
  /// list for the specified virtual or physical register.
  MachineOperand *&getRegUseDefListHead(unsigned RegNo) {
    if (TargetRegisterInfo::isVirtualRegister(RegNo))
      return VRegInfo[RegNo].second;
    return PhysRegUseDefLists[RegNo];
  }

  MachineOperand *getRegUseDefListHead(unsigned RegNo) const {
    if (TargetRegisterInfo::isVirtualRegister(RegNo))
      return VRegInfo[RegNo].second;
    return PhysRegUseDefLists[RegNo];
  }

  /// Get the next element in the use-def chain.
  static MachineOperand *getNextOperandForReg(const MachineOperand *MO) {
    assert(MO && MO->isReg() && "This is not a register operand!");
    return MO->Contents.Reg.Next;
  }

  /// UsedRegUnits - This is a bit vector that is computed and set by the
  /// register allocator, and must be kept up to date by passes that run after
  /// register allocation (though most don't modify this).  This is used
  /// so that the code generator knows which callee save registers to save and
  /// for other target specific uses.
  /// This vector has bits set for register units that are modified in the
  /// current function. It doesn't include registers clobbered by function
  /// calls with register mask operands.
  BitVector UsedRegUnits;

  /// UsedPhysRegMask - Additional used physregs including aliases.
  /// This bit vector represents all the registers clobbered by function calls.
  /// It can model things that UsedRegUnits can't, such as function calls that
  /// clobber ymm7 but preserve the low half in xmm7.
  BitVector UsedPhysRegMask;

  /// ReservedRegs - This is a bit vector of reserved registers.  The target
  /// may change its mind about which registers should be reserved.  This
  /// vector is the frozen set of reserved registers when register allocation
  /// started.
  BitVector ReservedRegs;

  /// Keep track of the physical registers that are live in to the function.
  /// Live in values are typically arguments in registers.  LiveIn values are
  /// allowed to have virtual registers associated with them, stored in the
  /// second element.
  std::vector<std::pair<unsigned, unsigned> > LiveIns;

  MachineRegisterInfo(const MachineRegisterInfo&) = delete;
  void operator=(const MachineRegisterInfo&) = delete;
public:
  explicit MachineRegisterInfo(const MachineFunction *MF);

  const TargetRegisterInfo *getTargetRegisterInfo() const {
    return MF->getSubtarget().getRegisterInfo();
  }

  void resetDelegate(Delegate *delegate) {
    // Ensure another delegate does not take over unless the current
    // delegate first unattaches itself. If we ever need to multicast
    // notifications, we will need to change to using a list.
    assert(TheDelegate == delegate &&
           "Only the current delegate can perform reset!");
    TheDelegate = nullptr;
  }

  void setDelegate(Delegate *delegate) {
    assert(delegate && !TheDelegate &&
           "Attempted to set delegate to null, or to change it without "
           "first resetting it!");

    TheDelegate = delegate;
  }

  //===--------------------------------------------------------------------===//
  // Function State
  //===--------------------------------------------------------------------===//

  // isSSA - Returns true when the machine function is in SSA form. Early
  // passes require the machine function to be in SSA form where every virtual
  // register has a single defining instruction.
  //
  // The TwoAddressInstructionPass and PHIElimination passes take the machine
  // function out of SSA form when they introduce multiple defs per virtual
  // register.
  bool isSSA() const { return IsSSA; }

  // leaveSSA - Indicates that the machine function is no longer in SSA form.
  void leaveSSA() { IsSSA = false; }

  /// tracksLiveness - Returns true when tracking register liveness accurately.
  ///
  /// While this flag is true, register liveness information in basic block
  /// live-in lists and machine instruction operands is accurate. This means it
  /// can be used to change the code in ways that affect the values in
  /// registers, for example by the register scavenger.
  ///
  /// When this flag is false, liveness is no longer reliable.
  bool tracksLiveness() const { return TracksLiveness; }

  /// invalidateLiveness - Indicates that register liveness is no longer being
  /// tracked accurately.
  ///
  /// This should be called by late passes that invalidate the liveness
  /// information.
  void invalidateLiveness() { TracksLiveness = false; }

  /// Returns true if liveness for register class @p RC should be tracked at
  /// the subregister level.
  bool shouldTrackSubRegLiveness(const TargetRegisterClass &RC) const {
    return subRegLivenessEnabled() && RC.HasDisjunctSubRegs;
  }
  bool shouldTrackSubRegLiveness(unsigned VReg) const {
    assert(TargetRegisterInfo::isVirtualRegister(VReg) && "Must pass a VReg");
    return shouldTrackSubRegLiveness(*getRegClass(VReg));
  }
  bool subRegLivenessEnabled() const {
    return TracksSubRegLiveness;
  }

  void enableSubRegLiveness(bool Enable = true) {
    TracksSubRegLiveness = Enable;
  }

  //===--------------------------------------------------------------------===//
  // Register Info
  //===--------------------------------------------------------------------===//

  // Strictly for use by MachineInstr.cpp.
  void addRegOperandToUseList(MachineOperand *MO);

  // Strictly for use by MachineInstr.cpp.
  void removeRegOperandFromUseList(MachineOperand *MO);

  // Strictly for use by MachineInstr.cpp.
  void moveOperands(MachineOperand *Dst, MachineOperand *Src, unsigned NumOps);

  /// Verify the sanity of the use list for Reg.
  void verifyUseList(unsigned Reg) const;

  /// Verify the use list of all registers.
  void verifyUseLists() const;

  /// reg_begin/reg_end - Provide iteration support to walk over all definitions
  /// and uses of a register within the MachineFunction that corresponds to this
  /// MachineRegisterInfo object.
  template<bool Uses, bool Defs, bool SkipDebug,
           bool ByOperand, bool ByInstr, bool ByBundle>
  class defusechain_iterator;
  template<bool Uses, bool Defs, bool SkipDebug,
           bool ByOperand, bool ByInstr, bool ByBundle>
  class defusechain_instr_iterator;

  // Make it a friend so it can access getNextOperandForReg().
  template<bool, bool, bool, bool, bool, bool>
    friend class defusechain_iterator;
  template<bool, bool, bool, bool, bool, bool>
    friend class defusechain_instr_iterator;



  /// reg_iterator/reg_begin/reg_end - Walk all defs and uses of the specified
  /// register.
  typedef defusechain_iterator<true,true,false,true,false,false>
          reg_iterator;
  reg_iterator reg_begin(unsigned RegNo) const {
    return reg_iterator(getRegUseDefListHead(RegNo));
  }
  static reg_iterator reg_end() { return reg_iterator(nullptr); }

  inline iterator_range<reg_iterator>  reg_operands(unsigned Reg) const {
    return iterator_range<reg_iterator>(reg_begin(Reg), reg_end());
  }

  /// reg_instr_iterator/reg_instr_begin/reg_instr_end - Walk all defs and uses
  /// of the specified register, stepping by MachineInstr.
  typedef defusechain_instr_iterator<true,true,false,false,true,false>
          reg_instr_iterator;
  reg_instr_iterator reg_instr_begin(unsigned RegNo) const {
    return reg_instr_iterator(getRegUseDefListHead(RegNo));
  }
  static reg_instr_iterator reg_instr_end() {
    return reg_instr_iterator(nullptr);
  }

  inline iterator_range<reg_instr_iterator>
  reg_instructions(unsigned Reg) const {
    return iterator_range<reg_instr_iterator>(reg_instr_begin(Reg),
                                              reg_instr_end());
  }

  /// reg_bundle_iterator/reg_bundle_begin/reg_bundle_end - Walk all defs and uses
  /// of the specified register, stepping by bundle.
  typedef defusechain_instr_iterator<true,true,false,false,false,true>
          reg_bundle_iterator;
  reg_bundle_iterator reg_bundle_begin(unsigned RegNo) const {
    return reg_bundle_iterator(getRegUseDefListHead(RegNo));
  }
  static reg_bundle_iterator reg_bundle_end() {
    return reg_bundle_iterator(nullptr);
  }

  inline iterator_range<reg_bundle_iterator> reg_bundles(unsigned Reg) const {
    return iterator_range<reg_bundle_iterator>(reg_bundle_begin(Reg),
                                               reg_bundle_end());
  }

  /// reg_empty - Return true if there are no instructions using or defining the
  /// specified register (it may be live-in).
  bool reg_empty(unsigned RegNo) const { return reg_begin(RegNo) == reg_end(); }

  /// reg_nodbg_iterator/reg_nodbg_begin/reg_nodbg_end - Walk all defs and uses
  /// of the specified register, skipping those marked as Debug.
  typedef defusechain_iterator<true,true,true,true,false,false>
          reg_nodbg_iterator;
  reg_nodbg_iterator reg_nodbg_begin(unsigned RegNo) const {
    return reg_nodbg_iterator(getRegUseDefListHead(RegNo));
  }
  static reg_nodbg_iterator reg_nodbg_end() {
    return reg_nodbg_iterator(nullptr);
  }

  inline iterator_range<reg_nodbg_iterator>
  reg_nodbg_operands(unsigned Reg) const {
    return iterator_range<reg_nodbg_iterator>(reg_nodbg_begin(Reg),
                                              reg_nodbg_end());
  }

  /// reg_instr_nodbg_iterator/reg_instr_nodbg_begin/reg_instr_nodbg_end - Walk
  /// all defs and uses of the specified register, stepping by MachineInstr,
  /// skipping those marked as Debug.
  typedef defusechain_instr_iterator<true,true,true,false,true,false>
          reg_instr_nodbg_iterator;
  reg_instr_nodbg_iterator reg_instr_nodbg_begin(unsigned RegNo) const {
    return reg_instr_nodbg_iterator(getRegUseDefListHead(RegNo));
  }
  static reg_instr_nodbg_iterator reg_instr_nodbg_end() {
    return reg_instr_nodbg_iterator(nullptr);
  }

  inline iterator_range<reg_instr_nodbg_iterator>
  reg_nodbg_instructions(unsigned Reg) const {
    return iterator_range<reg_instr_nodbg_iterator>(reg_instr_nodbg_begin(Reg),
                                                    reg_instr_nodbg_end());
  }

  /// reg_bundle_nodbg_iterator/reg_bundle_nodbg_begin/reg_bundle_nodbg_end - Walk
  /// all defs and uses of the specified register, stepping by bundle,
  /// skipping those marked as Debug.
  typedef defusechain_instr_iterator<true,true,true,false,false,true>
          reg_bundle_nodbg_iterator;
  reg_bundle_nodbg_iterator reg_bundle_nodbg_begin(unsigned RegNo) const {
    return reg_bundle_nodbg_iterator(getRegUseDefListHead(RegNo));
  }
  static reg_bundle_nodbg_iterator reg_bundle_nodbg_end() {
    return reg_bundle_nodbg_iterator(nullptr);
  }

  inline iterator_range<reg_bundle_nodbg_iterator> 
  reg_nodbg_bundles(unsigned Reg) const {
    return iterator_range<reg_bundle_nodbg_iterator>(reg_bundle_nodbg_begin(Reg),
                                                     reg_bundle_nodbg_end());
  }

  /// reg_nodbg_empty - Return true if the only instructions using or defining
  /// Reg are Debug instructions.
  bool reg_nodbg_empty(unsigned RegNo) const {
    return reg_nodbg_begin(RegNo) == reg_nodbg_end();
  }

  /// def_iterator/def_begin/def_end - Walk all defs of the specified register.
  typedef defusechain_iterator<false,true,false,true,false,false>
          def_iterator;
  def_iterator def_begin(unsigned RegNo) const {
    return def_iterator(getRegUseDefListHead(RegNo));
  }
  static def_iterator def_end() { return def_iterator(nullptr); }

  inline iterator_range<def_iterator> def_operands(unsigned Reg) const {
    return iterator_range<def_iterator>(def_begin(Reg), def_end());
  }

  /// def_instr_iterator/def_instr_begin/def_instr_end - Walk all defs of the
  /// specified register, stepping by MachineInst.
  typedef defusechain_instr_iterator<false,true,false,false,true,false>
          def_instr_iterator;
  def_instr_iterator def_instr_begin(unsigned RegNo) const {
    return def_instr_iterator(getRegUseDefListHead(RegNo));
  }
  static def_instr_iterator def_instr_end() {
    return def_instr_iterator(nullptr);
  }

  inline iterator_range<def_instr_iterator>
  def_instructions(unsigned Reg) const {
    return iterator_range<def_instr_iterator>(def_instr_begin(Reg),
                                              def_instr_end());
  }

  /// def_bundle_iterator/def_bundle_begin/def_bundle_end - Walk all defs of the
  /// specified register, stepping by bundle.
  typedef defusechain_instr_iterator<false,true,false,false,false,true>
          def_bundle_iterator;
  def_bundle_iterator def_bundle_begin(unsigned RegNo) const {
    return def_bundle_iterator(getRegUseDefListHead(RegNo));
  }
  static def_bundle_iterator def_bundle_end() {
    return def_bundle_iterator(nullptr);
  }

  inline iterator_range<def_bundle_iterator> def_bundles(unsigned Reg) const {
    return iterator_range<def_bundle_iterator>(def_bundle_begin(Reg),
                                               def_bundle_end());
  }

  /// def_empty - Return true if there are no instructions defining the
  /// specified register (it may be live-in).
  bool def_empty(unsigned RegNo) const { return def_begin(RegNo) == def_end(); }

  /// hasOneDef - Return true if there is exactly one instruction defining the
  /// specified register.
  bool hasOneDef(unsigned RegNo) const {
    def_iterator DI = def_begin(RegNo);
    if (DI == def_end())
      return false;
    return ++DI == def_end();
  }

  /// use_iterator/use_begin/use_end - Walk all uses of the specified register.
  typedef defusechain_iterator<true,false,false,true,false,false>
          use_iterator;
  use_iterator use_begin(unsigned RegNo) const {
    return use_iterator(getRegUseDefListHead(RegNo));
  }
  static use_iterator use_end() { return use_iterator(nullptr); }

  inline iterator_range<use_iterator> use_operands(unsigned Reg) const {
    return iterator_range<use_iterator>(use_begin(Reg), use_end());
  }

  /// use_instr_iterator/use_instr_begin/use_instr_end - Walk all uses of the
  /// specified register, stepping by MachineInstr.
  typedef defusechain_instr_iterator<true,false,false,false,true,false>
          use_instr_iterator;
  use_instr_iterator use_instr_begin(unsigned RegNo) const {
    return use_instr_iterator(getRegUseDefListHead(RegNo));
  }
  static use_instr_iterator use_instr_end() {
    return use_instr_iterator(nullptr);
  }

  inline iterator_range<use_instr_iterator>
  use_instructions(unsigned Reg) const {
    return iterator_range<use_instr_iterator>(use_instr_begin(Reg),
                                              use_instr_end());
  }

  /// use_bundle_iterator/use_bundle_begin/use_bundle_end - Walk all uses of the
  /// specified register, stepping by bundle.
  typedef defusechain_instr_iterator<true,false,false,false,false,true>
          use_bundle_iterator;
  use_bundle_iterator use_bundle_begin(unsigned RegNo) const {
    return use_bundle_iterator(getRegUseDefListHead(RegNo));
  }
  static use_bundle_iterator use_bundle_end() {
    return use_bundle_iterator(nullptr);
  }

  inline iterator_range<use_bundle_iterator> use_bundles(unsigned Reg) const {
    return iterator_range<use_bundle_iterator>(use_bundle_begin(Reg),
                                               use_bundle_end());
  }

  /// use_empty - Return true if there are no instructions using the specified
  /// register.
  bool use_empty(unsigned RegNo) const { return use_begin(RegNo) == use_end(); }

  /// hasOneUse - Return true if there is exactly one instruction using the
  /// specified register.
  bool hasOneUse(unsigned RegNo) const {
    use_iterator UI = use_begin(RegNo);
    if (UI == use_end())
      return false;
    return ++UI == use_end();
  }

  /// use_nodbg_iterator/use_nodbg_begin/use_nodbg_end - Walk all uses of the
  /// specified register, skipping those marked as Debug.
  typedef defusechain_iterator<true,false,true,true,false,false>
          use_nodbg_iterator;
  use_nodbg_iterator use_nodbg_begin(unsigned RegNo) const {
    return use_nodbg_iterator(getRegUseDefListHead(RegNo));
  }
  static use_nodbg_iterator use_nodbg_end() {
    return use_nodbg_iterator(nullptr);
  }

  inline iterator_range<use_nodbg_iterator>
  use_nodbg_operands(unsigned Reg) const {
    return iterator_range<use_nodbg_iterator>(use_nodbg_begin(Reg),
                                              use_nodbg_end());
  }

  /// use_instr_nodbg_iterator/use_instr_nodbg_begin/use_instr_nodbg_end - Walk
  /// all uses of the specified register, stepping by MachineInstr, skipping
  /// those marked as Debug.
  typedef defusechain_instr_iterator<true,false,true,false,true,false>
          use_instr_nodbg_iterator;
  use_instr_nodbg_iterator use_instr_nodbg_begin(unsigned RegNo) const {
    return use_instr_nodbg_iterator(getRegUseDefListHead(RegNo));
  }
  static use_instr_nodbg_iterator use_instr_nodbg_end() {
    return use_instr_nodbg_iterator(nullptr);
  }

  inline iterator_range<use_instr_nodbg_iterator>
  use_nodbg_instructions(unsigned Reg) const {
    return iterator_range<use_instr_nodbg_iterator>(use_instr_nodbg_begin(Reg),
                                                    use_instr_nodbg_end());
  }

  /// use_bundle_nodbg_iterator/use_bundle_nodbg_begin/use_bundle_nodbg_end - Walk
  /// all uses of the specified register, stepping by bundle, skipping
  /// those marked as Debug.
  typedef defusechain_instr_iterator<true,false,true,false,false,true>
          use_bundle_nodbg_iterator;
  use_bundle_nodbg_iterator use_bundle_nodbg_begin(unsigned RegNo) const {
    return use_bundle_nodbg_iterator(getRegUseDefListHead(RegNo));
  }
  static use_bundle_nodbg_iterator use_bundle_nodbg_end() {
    return use_bundle_nodbg_iterator(nullptr);
  }

  inline iterator_range<use_bundle_nodbg_iterator>
  use_nodbg_bundles(unsigned Reg) const {
    return iterator_range<use_bundle_nodbg_iterator>(use_bundle_nodbg_begin(Reg),
                                                     use_bundle_nodbg_end());
  }

  /// use_nodbg_empty - Return true if there are no non-Debug instructions
  /// using the specified register.
  bool use_nodbg_empty(unsigned RegNo) const {
    return use_nodbg_begin(RegNo) == use_nodbg_end();
  }

  /// hasOneNonDBGUse - Return true if there is exactly one non-Debug
  /// instruction using the specified register.
  bool hasOneNonDBGUse(unsigned RegNo) const;

  /// replaceRegWith - Replace all instances of FromReg with ToReg in the
  /// machine function.  This is like llvm-level X->replaceAllUsesWith(Y),
  /// except that it also changes any definitions of the register as well.
  ///
  /// Note that it is usually necessary to first constrain ToReg's register
  /// class to match the FromReg constraints using:
  ///
  ///   constrainRegClass(ToReg, getRegClass(FromReg))
  ///
  /// That function will return NULL if the virtual registers have incompatible
  /// constraints.
  ///
  /// Note that if ToReg is a physical register the function will replace and
  /// apply sub registers to ToReg in order to obtain a final/proper physical
  /// register.
  void replaceRegWith(unsigned FromReg, unsigned ToReg);
  
  /// getVRegDef - Return the machine instr that defines the specified virtual
  /// register or null if none is found.  This assumes that the code is in SSA
  /// form, so there should only be one definition.
  MachineInstr *getVRegDef(unsigned Reg) const;

  /// getUniqueVRegDef - Return the unique machine instr that defines the
  /// specified virtual register or null if none is found.  If there are
  /// multiple definitions or no definition, return null.
  MachineInstr *getUniqueVRegDef(unsigned Reg) const;

  /// clearKillFlags - Iterate over all the uses of the given register and
  /// clear the kill flag from the MachineOperand. This function is used by
  /// optimization passes which extend register lifetimes and need only
  /// preserve conservative kill flag information.
  void clearKillFlags(unsigned Reg) const;

#ifndef NDEBUG
  void dumpUses(unsigned RegNo) const;
#endif

  /// isConstantPhysReg - Returns true if PhysReg is unallocatable and constant
  /// throughout the function.  It is safe to move instructions that read such
  /// a physreg.
  bool isConstantPhysReg(unsigned PhysReg, const MachineFunction &MF) const;

  /// Get an iterator over the pressure sets affected by the given physical or
  /// virtual register. If RegUnit is physical, it must be a register unit (from
  /// MCRegUnitIterator).
  PSetIterator getPressureSets(unsigned RegUnit) const;

  //===--------------------------------------------------------------------===//
  // Virtual Register Info
  //===--------------------------------------------------------------------===//

  /// getRegClass - Return the register class of the specified virtual register.
  ///
  const TargetRegisterClass *getRegClass(unsigned Reg) const {
    return VRegInfo[Reg].first;
  }

  /// setRegClass - Set the register class of the specified virtual register.
  ///
  void setRegClass(unsigned Reg, const TargetRegisterClass *RC);

  /// constrainRegClass - Constrain the register class of the specified virtual
  /// register to be a common subclass of RC and the current register class,
  /// but only if the new class has at least MinNumRegs registers.  Return the
  /// new register class, or NULL if no such class exists.
  /// This should only be used when the constraint is known to be trivial, like
  /// GR32 -> GR32_NOSP. Beware of increasing register pressure.
  ///
  const TargetRegisterClass *constrainRegClass(unsigned Reg,
                                               const TargetRegisterClass *RC,
                                               unsigned MinNumRegs = 0);

  /// recomputeRegClass - Try to find a legal super-class of Reg's register
  /// class that still satisfies the constraints from the instructions using
  /// Reg.  Returns true if Reg was upgraded.
  ///
  /// This method can be used after constraints have been removed from a
  /// virtual register, for example after removing instructions or splitting
  /// the live range.
  ///
  bool recomputeRegClass(unsigned Reg);

  /// createVirtualRegister - Create and return a new virtual register in the
  /// function with the specified register class.
  ///
  unsigned createVirtualRegister(const TargetRegisterClass *RegClass);

  /// getNumVirtRegs - Return the number of virtual registers created.
  ///
  unsigned getNumVirtRegs() const { return VRegInfo.size(); }

  /// clearVirtRegs - Remove all virtual registers (after physreg assignment).
  void clearVirtRegs();

  /// setRegAllocationHint - Specify a register allocation hint for the
  /// specified virtual register.
  void setRegAllocationHint(unsigned VReg, unsigned Type, unsigned PrefReg) {
    assert(TargetRegisterInfo::isVirtualRegister(VReg));
    RegAllocHints[VReg].first  = Type;
    RegAllocHints[VReg].second = PrefReg;
  }

  /// getRegAllocationHint - Return the register allocation hint for the
  /// specified virtual register.
  std::pair<unsigned, unsigned>
  getRegAllocationHint(unsigned VReg) const {
    assert(TargetRegisterInfo::isVirtualRegister(VReg));
    return RegAllocHints[VReg];
  }

  /// getSimpleHint - Return the preferred register allocation hint, or 0 if a
  /// standard simple hint (Type == 0) is not set.
  unsigned getSimpleHint(unsigned VReg) const {
    assert(TargetRegisterInfo::isVirtualRegister(VReg));
    std::pair<unsigned, unsigned> Hint = getRegAllocationHint(VReg);
    return Hint.first ? 0 : Hint.second;
  }

  /// markUsesInDebugValueAsUndef - Mark every DBG_VALUE referencing the
  /// specified register as undefined which causes the DBG_VALUE to be
  /// deleted during LiveDebugVariables analysis.
  void markUsesInDebugValueAsUndef(unsigned Reg) const;

  /// Return true if the specified register is modified in this function.
  /// This checks that no defining machine operands exist for the register or
  /// any of its aliases. Definitions found on functions marked noreturn are
  /// ignored.
  bool isPhysRegModified(unsigned PhysReg) const;

  //===--------------------------------------------------------------------===//
  // Physical Register Use Info
  //===--------------------------------------------------------------------===//

  /// isPhysRegUsed - Return true if the specified register is used in this
  /// function. Also check for clobbered aliases and registers clobbered by
  /// function calls with register mask operands.
  ///
  /// This only works after register allocation.
  bool isPhysRegUsed(unsigned Reg) const {
    if (UsedPhysRegMask.test(Reg))
      return true;
    for (MCRegUnitIterator Units(Reg, getTargetRegisterInfo());
         Units.isValid(); ++Units)
      if (UsedRegUnits.test(*Units))
        return true;
    return false;
  }

  /// Mark the specified register unit as used in this function.
  /// This should only be called during and after register allocation.
  void setRegUnitUsed(unsigned RegUnit) {
    UsedRegUnits.set(RegUnit);
  }

  /// setPhysRegUsed - Mark the specified register used in this function.
  /// This should only be called during and after register allocation.
  void setPhysRegUsed(unsigned Reg) {
    for (MCRegUnitIterator Units(Reg, getTargetRegisterInfo());
         Units.isValid(); ++Units)
      UsedRegUnits.set(*Units);
  }

  /// addPhysRegsUsedFromRegMask - Mark any registers not in RegMask as used.
  /// This corresponds to the bit mask attached to register mask operands.
  void addPhysRegsUsedFromRegMask(const uint32_t *RegMask) {
    UsedPhysRegMask.setBitsNotInMask(RegMask);
  }

  /// setPhysRegUnused - Mark the specified register unused in this function.
  /// This should only be called during and after register allocation.
  void setPhysRegUnused(unsigned Reg) {
    UsedPhysRegMask.reset(Reg);
    for (MCRegUnitIterator Units(Reg, getTargetRegisterInfo());
         Units.isValid(); ++Units)
      UsedRegUnits.reset(*Units);
  }


  //===--------------------------------------------------------------------===//
  // Reserved Register Info
  //===--------------------------------------------------------------------===//
  //
  // The set of reserved registers must be invariant during register
  // allocation.  For example, the target cannot suddenly decide it needs a
  // frame pointer when the register allocator has already used the frame
  // pointer register for something else.
  //
  // These methods can be used by target hooks like hasFP() to avoid changing
  // the reserved register set during register allocation.

  /// freezeReservedRegs - Called by the register allocator to freeze the set
  /// of reserved registers before allocation begins.
  void freezeReservedRegs(const MachineFunction&);

  /// reservedRegsFrozen - Returns true after freezeReservedRegs() was called
  /// to ensure the set of reserved registers stays constant.
  bool reservedRegsFrozen() const {
    return !ReservedRegs.empty();
  }

  /// canReserveReg - Returns true if PhysReg can be used as a reserved
  /// register.  Any register can be reserved before freezeReservedRegs() is
  /// called.
  bool canReserveReg(unsigned PhysReg) const {
    return !reservedRegsFrozen() || ReservedRegs.test(PhysReg);
  }

  /// getReservedRegs - Returns a reference to the frozen set of reserved
  /// registers. This method should always be preferred to calling
  /// TRI::getReservedRegs() when possible.
  const BitVector &getReservedRegs() const {
    assert(reservedRegsFrozen() &&
           "Reserved registers haven't been frozen yet. "
           "Use TRI::getReservedRegs().");
    return ReservedRegs;
  }

  /// isReserved - Returns true when PhysReg is a reserved register.
  ///
  /// Reserved registers may belong to an allocatable register class, but the
  /// target has explicitly requested that they are not used.
  ///
  bool isReserved(unsigned PhysReg) const {
    return getReservedRegs().test(PhysReg);
  }

  /// isAllocatable - Returns true when PhysReg belongs to an allocatable
  /// register class and it hasn't been reserved.
  ///
  /// Allocatable registers may show up in the allocation order of some virtual
  /// register, so a register allocator needs to track its liveness and
  /// availability.
  bool isAllocatable(unsigned PhysReg) const {
    return getTargetRegisterInfo()->isInAllocatableClass(PhysReg) &&
      !isReserved(PhysReg);
  }

  //===--------------------------------------------------------------------===//
  // LiveIn Management
  //===--------------------------------------------------------------------===//

  /// addLiveIn - Add the specified register as a live-in.  Note that it
  /// is an error to add the same register to the same set more than once.
  void addLiveIn(unsigned Reg, unsigned vreg = 0) {
    LiveIns.push_back(std::make_pair(Reg, vreg));
  }

  // Iteration support for the live-ins set.  It's kept in sorted order
  // by register number.
  typedef std::vector<std::pair<unsigned,unsigned> >::const_iterator
  livein_iterator;
  livein_iterator livein_begin() const { return LiveIns.begin(); }
  livein_iterator livein_end()   const { return LiveIns.end(); }
  bool            livein_empty() const { return LiveIns.empty(); }

  bool isLiveIn(unsigned Reg) const;

  /// getLiveInPhysReg - If VReg is a live-in virtual register, return the
  /// corresponding live-in physical register.
  unsigned getLiveInPhysReg(unsigned VReg) const;

  /// getLiveInVirtReg - If PReg is a live-in physical register, return the
  /// corresponding live-in physical register.
  unsigned getLiveInVirtReg(unsigned PReg) const;

  /// EmitLiveInCopies - Emit copies to initialize livein virtual registers
  /// into the given entry block.
  void EmitLiveInCopies(MachineBasicBlock *EntryMBB,
                        const TargetRegisterInfo &TRI,
                        const TargetInstrInfo &TII);

  /// Returns a mask covering all bits that can appear in lane masks of
  /// subregisters of the virtual register @p Reg.
  unsigned getMaxLaneMaskForVReg(unsigned Reg) const;

  /// defusechain_iterator - This class provides iterator support for machine
  /// operands in the function that use or define a specific register.  If
  /// ReturnUses is true it returns uses of registers, if ReturnDefs is true it
  /// returns defs.  If neither are true then you are silly and it always
  /// returns end().  If SkipDebug is true it skips uses marked Debug
  /// when incrementing.
  template<bool ReturnUses, bool ReturnDefs, bool SkipDebug,
           bool ByOperand, bool ByInstr, bool ByBundle>
  class defusechain_iterator
    : public std::iterator<std::forward_iterator_tag, MachineInstr, ptrdiff_t> {
    MachineOperand *Op;
    explicit defusechain_iterator(MachineOperand *op) : Op(op) {
      // If the first node isn't one we're interested in, advance to one that
      // we are interested in.
      if (op) {
        if ((!ReturnUses && op->isUse()) ||
            (!ReturnDefs && op->isDef()) ||
            (SkipDebug && op->isDebug()))
          advance();
      }
    }
    friend class MachineRegisterInfo;

    void advance() {
      assert(Op && "Cannot increment end iterator!");
      Op = getNextOperandForReg(Op);

      // All defs come before the uses, so stop def_iterator early.
      if (!ReturnUses) {
        if (Op) {
          if (Op->isUse())
            Op = nullptr;
          else
            assert(!Op->isDebug() && "Can't have debug defs");
        }
      } else {
        // If this is an operand we don't care about, skip it.
        while (Op && ((!ReturnDefs && Op->isDef()) ||
                      (SkipDebug && Op->isDebug())))
          Op = getNextOperandForReg(Op);
      }
    }
  public:
    typedef std::iterator<std::forward_iterator_tag,
                          MachineInstr, ptrdiff_t>::reference reference;
    typedef std::iterator<std::forward_iterator_tag,
                          MachineInstr, ptrdiff_t>::pointer pointer;

    defusechain_iterator() : Op(nullptr) {}

    bool operator==(const defusechain_iterator &x) const {
      return Op == x.Op;
    }
    bool operator!=(const defusechain_iterator &x) const {
      return !operator==(x);
    }

    /// atEnd - return true if this iterator is equal to reg_end() on the value.
    bool atEnd() const { return Op == nullptr; }

    // Iterator traversal: forward iteration only
    defusechain_iterator &operator++() {          // Preincrement
      assert(Op && "Cannot increment end iterator!");
      if (ByOperand)
        advance();
      else if (ByInstr) {
        MachineInstr *P = Op->getParent();
        do {
          advance();
        } while (Op && Op->getParent() == P);
      } else if (ByBundle) {
        MachineInstr *P = getBundleStart(Op->getParent());
        do {
          advance();
        } while (Op && getBundleStart(Op->getParent()) == P);
      }

      return *this;
    }
    defusechain_iterator operator++(int) {        // Postincrement
      defusechain_iterator tmp = *this; ++*this; return tmp;
    }

    /// getOperandNo - Return the operand # of this MachineOperand in its
    /// MachineInstr.
    unsigned getOperandNo() const {
      assert(Op && "Cannot dereference end iterator!");
      return Op - &Op->getParent()->getOperand(0);
    }

    // Retrieve a reference to the current operand.
    MachineOperand &operator*() const {
      assert(Op && "Cannot dereference end iterator!");
      return *Op;
    }

    MachineOperand *operator->() const {
      assert(Op && "Cannot dereference end iterator!");
      return Op;
    }
  };

  /// defusechain_iterator - This class provides iterator support for machine
  /// operands in the function that use or define a specific register.  If
  /// ReturnUses is true it returns uses of registers, if ReturnDefs is true it
  /// returns defs.  If neither are true then you are silly and it always
  /// returns end().  If SkipDebug is true it skips uses marked Debug
  /// when incrementing.
  template<bool ReturnUses, bool ReturnDefs, bool SkipDebug,
           bool ByOperand, bool ByInstr, bool ByBundle>
  class defusechain_instr_iterator
    : public std::iterator<std::forward_iterator_tag, MachineInstr, ptrdiff_t> {
    MachineOperand *Op;
    explicit defusechain_instr_iterator(MachineOperand *op) : Op(op) {
      // If the first node isn't one we're interested in, advance to one that
      // we are interested in.
      if (op) {
        if ((!ReturnUses && op->isUse()) ||
            (!ReturnDefs && op->isDef()) ||
            (SkipDebug && op->isDebug()))
          advance();
      }
    }
    friend class MachineRegisterInfo;

    void advance() {
      assert(Op && "Cannot increment end iterator!");
      Op = getNextOperandForReg(Op);

      // All defs come before the uses, so stop def_iterator early.
      if (!ReturnUses) {
        if (Op) {
          if (Op->isUse())
            Op = nullptr;
          else
            assert(!Op->isDebug() && "Can't have debug defs");
        }
      } else {
        // If this is an operand we don't care about, skip it.
        while (Op && ((!ReturnDefs && Op->isDef()) ||
                      (SkipDebug && Op->isDebug())))
          Op = getNextOperandForReg(Op);
      }
    }
  public:
    typedef std::iterator<std::forward_iterator_tag,
                          MachineInstr, ptrdiff_t>::reference reference;
    typedef std::iterator<std::forward_iterator_tag,
                          MachineInstr, ptrdiff_t>::pointer pointer;

    defusechain_instr_iterator() : Op(nullptr) {}

    bool operator==(const defusechain_instr_iterator &x) const {
      return Op == x.Op;
    }
    bool operator!=(const defusechain_instr_iterator &x) const {
      return !operator==(x);
    }

    /// atEnd - return true if this iterator is equal to reg_end() on the value.
    bool atEnd() const { return Op == nullptr; }

    // Iterator traversal: forward iteration only
    defusechain_instr_iterator &operator++() {          // Preincrement
      assert(Op && "Cannot increment end iterator!");
      if (ByOperand)
        advance();
      else if (ByInstr) {
        MachineInstr *P = Op->getParent();
        do {
          advance();
        } while (Op && Op->getParent() == P);
      } else if (ByBundle) {
        MachineInstr *P = getBundleStart(Op->getParent());
        do {
          advance();
        } while (Op && getBundleStart(Op->getParent()) == P);
      }

      return *this;
    }
    defusechain_instr_iterator operator++(int) {        // Postincrement
      defusechain_instr_iterator tmp = *this; ++*this; return tmp;
    }

    // Retrieve a reference to the current operand.
    MachineInstr &operator*() const {
      assert(Op && "Cannot dereference end iterator!");
      if (ByBundle) return *(getBundleStart(Op->getParent()));
      return *Op->getParent();
    }

    MachineInstr *operator->() const {
      assert(Op && "Cannot dereference end iterator!");
      if (ByBundle) return getBundleStart(Op->getParent());
      return Op->getParent();
    }
  };
};

/// Iterate over the pressure sets affected by the given physical or virtual
/// register. If Reg is physical, it must be a register unit (from
/// MCRegUnitIterator).
class PSetIterator {
  const int *PSet;
  unsigned Weight;
public:
  PSetIterator(): PSet(nullptr), Weight(0) {}
  PSetIterator(unsigned RegUnit, const MachineRegisterInfo *MRI) {
    const TargetRegisterInfo *TRI = MRI->getTargetRegisterInfo();
    if (TargetRegisterInfo::isVirtualRegister(RegUnit)) {
      const TargetRegisterClass *RC = MRI->getRegClass(RegUnit);
      PSet = TRI->getRegClassPressureSets(RC);
      Weight = TRI->getRegClassWeight(RC).RegWeight;
    }
    else {
      PSet = TRI->getRegUnitPressureSets(RegUnit);
      Weight = TRI->getRegUnitWeight(RegUnit);
    }
    if (*PSet == -1)
      PSet = nullptr;
  }
  bool isValid() const { return PSet; }

  unsigned getWeight() const { return Weight; }

  unsigned operator*() const { return *PSet; }

  void operator++() {
    assert(isValid() && "Invalid PSetIterator.");
    ++PSet;
    if (*PSet == -1)
      PSet = nullptr;
  }
};

inline PSetIterator MachineRegisterInfo::
getPressureSets(unsigned RegUnit) const {
  return PSetIterator(RegUnit, this);
}

} // End llvm namespace

#endif
