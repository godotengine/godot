//===-- llvm/Target/TargetFrameLowering.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface to describe the layout of a stack frame on the target machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETFRAMELOWERING_H
#define LLVM_TARGET_TARGETFRAMELOWERING_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include <utility>
#include <vector>

namespace llvm {
  class BitVector;
  class CalleeSavedInfo;
  class MachineFunction;
  class RegScavenger;

/// Information about stack frame layout on the target.  It holds the direction
/// of stack growth, the known stack alignment on entry to each function, and
/// the offset to the locals area.
///
/// The offset to the local area is the offset from the stack pointer on
/// function entry to the first location where function data (local variables,
/// spill locations) can be stored.
class TargetFrameLowering {
public:
  enum StackDirection {
    StackGrowsUp,        // Adding to the stack increases the stack address
    StackGrowsDown       // Adding to the stack decreases the stack address
  };

  // Maps a callee saved register to a stack slot with a fixed offset.
  struct SpillSlot {
    unsigned Reg;
    int Offset; // Offset relative to stack pointer on function entry.
  };
private:
  StackDirection StackDir;
  unsigned StackAlignment;
  unsigned TransientStackAlignment;
  int LocalAreaOffset;
  bool StackRealignable;
public:
  TargetFrameLowering(StackDirection D, unsigned StackAl, int LAO,
                      unsigned TransAl = 1, bool StackReal = true)
    : StackDir(D), StackAlignment(StackAl), TransientStackAlignment(TransAl),
      LocalAreaOffset(LAO), StackRealignable(StackReal) {}

  virtual ~TargetFrameLowering();

  // These methods return information that describes the abstract stack layout
  // of the target machine.

  /// getStackGrowthDirection - Return the direction the stack grows
  ///
  StackDirection getStackGrowthDirection() const { return StackDir; }

  /// getStackAlignment - This method returns the number of bytes to which the
  /// stack pointer must be aligned on entry to a function.  Typically, this
  /// is the largest alignment for any data object in the target.
  ///
  unsigned getStackAlignment() const { return StackAlignment; }

  /// getTransientStackAlignment - This method returns the number of bytes to
  /// which the stack pointer must be aligned at all times, even between
  /// calls.
  ///
  unsigned getTransientStackAlignment() const {
    return TransientStackAlignment;
  }

  /// isStackRealignable - This method returns whether the stack can be
  /// realigned.
  bool isStackRealignable() const {
    return StackRealignable;
  }

  /// getOffsetOfLocalArea - This method returns the offset of the local area
  /// from the stack pointer on entrance to a function.
  ///
  int getOffsetOfLocalArea() const { return LocalAreaOffset; }

  /// isFPCloseToIncomingSP - Return true if the frame pointer is close to
  /// the incoming stack pointer, false if it is close to the post-prologue
  /// stack pointer.
  virtual bool isFPCloseToIncomingSP() const { return true; }

  /// assignCalleeSavedSpillSlots - Allows target to override spill slot
  /// assignment logic.  If implemented, assignCalleeSavedSpillSlots() should
  /// assign frame slots to all CSI entries and return true.  If this method
  /// returns false, spill slots will be assigned using generic implementation.
  /// assignCalleeSavedSpillSlots() may add, delete or rearrange elements of
  /// CSI.
  virtual bool
  assignCalleeSavedSpillSlots(MachineFunction &MF,
                              const TargetRegisterInfo *TRI,
                              std::vector<CalleeSavedInfo> &CSI) const {
    return false;
  }

  /// getCalleeSavedSpillSlots - This method returns a pointer to an array of
  /// pairs, that contains an entry for each callee saved register that must be
  /// spilled to a particular stack location if it is spilled.
  ///
  /// Each entry in this array contains a <register,offset> pair, indicating the
  /// fixed offset from the incoming stack pointer that each register should be
  /// spilled at. If a register is not listed here, the code generator is
  /// allowed to spill it anywhere it chooses.
  ///
  virtual const SpillSlot *
  getCalleeSavedSpillSlots(unsigned &NumEntries) const {
    NumEntries = 0;
    return nullptr;
  }

  /// targetHandlesStackFrameRounding - Returns true if the target is
  /// responsible for rounding up the stack frame (probably at emitPrologue
  /// time).
  virtual bool targetHandlesStackFrameRounding() const {
    return false;
  }

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  virtual void emitPrologue(MachineFunction &MF,
                            MachineBasicBlock &MBB) const = 0;
  virtual void emitEpilogue(MachineFunction &MF,
                            MachineBasicBlock &MBB) const = 0;

  /// Adjust the prologue to have the function use segmented stacks. This works
  /// by adding a check even before the "normal" function prologue.
  virtual void adjustForSegmentedStacks(MachineFunction &MF,
                                        MachineBasicBlock &PrologueMBB) const {}

  /// Adjust the prologue to add Erlang Run-Time System (ERTS) specific code in
  /// the assembly prologue to explicitly handle the stack.
  virtual void adjustForHiPEPrologue(MachineFunction &MF,
                                     MachineBasicBlock &PrologueMBB) const {}

  /// Adjust the prologue to add an allocation at a fixed offset from the frame
  /// pointer.
  virtual void
  adjustForFrameAllocatePrologue(MachineFunction &MF,
                                 MachineBasicBlock &PrologueMBB) const {}

  /// spillCalleeSavedRegisters - Issues instruction(s) to spill all callee
  /// saved registers and returns true if it isn't possible / profitable to do
  /// so by issuing a series of store instructions via
  /// storeRegToStackSlot(). Returns false otherwise.
  virtual bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MI,
                                        const std::vector<CalleeSavedInfo> &CSI,
                                         const TargetRegisterInfo *TRI) const {
    return false;
  }

  /// restoreCalleeSavedRegisters - Issues instruction(s) to restore all callee
  /// saved registers and returns true if it isn't possible / profitable to do
  /// so by issuing a series of load instructions via loadRegToStackSlot().
  /// Returns false otherwise.
  virtual bool restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator MI,
                                        const std::vector<CalleeSavedInfo> &CSI,
                                        const TargetRegisterInfo *TRI) const {
    return false;
  }

  /// Return true if the target needs to disable frame pointer elimination.
  virtual bool noFramePointerElim(const MachineFunction &MF) const;

  /// hasFP - Return true if the specified function should have a dedicated
  /// frame pointer register. For most targets this is true only if the function
  /// has variable sized allocas or if frame pointer elimination is disabled.
  virtual bool hasFP(const MachineFunction &MF) const = 0;

  /// hasReservedCallFrame - Under normal circumstances, when a frame pointer is
  /// not required, we reserve argument space for call sites in the function
  /// immediately on entry to the current function. This eliminates the need for
  /// add/sub sp brackets around call sites. Returns true if the call frame is
  /// included as part of the stack frame.
  virtual bool hasReservedCallFrame(const MachineFunction &MF) const {
    return !hasFP(MF);
  }

  /// canSimplifyCallFramePseudos - When possible, it's best to simplify the
  /// call frame pseudo ops before doing frame index elimination. This is
  /// possible only when frame index references between the pseudos won't
  /// need adjusting for the call frame adjustments. Normally, that's true
  /// if the function has a reserved call frame or a frame pointer. Some
  /// targets (Thumb2, for example) may have more complicated criteria,
  /// however, and can override this behavior.
  virtual bool canSimplifyCallFramePseudos(const MachineFunction &MF) const {
    return hasReservedCallFrame(MF) || hasFP(MF);
  }

  // needsFrameIndexResolution - Do we need to perform FI resolution for
  // this function. Normally, this is required only when the function
  // has any stack objects. However, targets may want to override this.
  virtual bool needsFrameIndexResolution(const MachineFunction &MF) const;

  /// getFrameIndexOffset - Returns the displacement from the frame register to
  /// the stack frame of the specified index.
  virtual int getFrameIndexOffset(const MachineFunction &MF, int FI) const;

  /// getFrameIndexReference - This method should return the base register
  /// and offset used to reference a frame index location. The offset is
  /// returned directly, and the base register is returned via FrameReg.
  virtual int getFrameIndexReference(const MachineFunction &MF, int FI,
                                     unsigned &FrameReg) const;

  /// Same as above, except that the 'base register' will always be RSP, not
  /// RBP on x86.  This is used exclusively for lowering STATEPOINT nodes.
  /// TODO: This should really be a parameterizable choice.
  virtual int getFrameIndexReferenceFromSP(const MachineFunction &MF, int FI,
                                          unsigned &FrameReg) const {
    // default to calling normal version, we override this on x86 only
    llvm_unreachable("unimplemented for non-x86");
    return 0;
  }

  /// This method determines which of the registers reported by
  /// TargetRegisterInfo::getCalleeSavedRegs() should actually get saved.
  /// The default implementation checks populates the \p SavedRegs bitset with
  /// all registers which are modified in the function, targets may override
  /// this function to save additional registers.
  /// This method also sets up the register scavenger ensuring there is a free
  /// register or a frameindex available.
  virtual void determineCalleeSaves(MachineFunction &MF, BitVector &SavedRegs,
                                    RegScavenger *RS = nullptr) const;

  /// processFunctionBeforeFrameFinalized - This method is called immediately
  /// before the specified function's frame layout (MF.getFrameInfo()) is
  /// finalized.  Once the frame is finalized, MO_FrameIndex operands are
  /// replaced with direct constants.  This method is optional.
  ///
  virtual void processFunctionBeforeFrameFinalized(MachineFunction &MF,
                                             RegScavenger *RS = nullptr) const {
  }

  /// eliminateCallFramePseudoInstr - This method is called during prolog/epilog
  /// code insertion to eliminate call frame setup and destroy pseudo
  /// instructions (but only if the Target is using them).  It is responsible
  /// for eliminating these instructions, replacing them with concrete
  /// instructions.  This method need only be implemented if using call frame
  /// setup/destroy pseudo instructions.
  ///
  virtual void
  eliminateCallFramePseudoInstr(MachineFunction &MF,
                                MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MI) const {
    llvm_unreachable("Call Frame Pseudo Instructions do not exist on this "
                     "target!");
  }

  /// Check whether or not the given \p MBB can be used as a prologue
  /// for the target.
  /// The prologue will be inserted first in this basic block.
  /// This method is used by the shrink-wrapping pass to decide if
  /// \p MBB will be correctly handled by the target.
  /// As soon as the target enable shrink-wrapping without overriding
  /// this method, we assume that each basic block is a valid
  /// prologue.
  virtual bool canUseAsPrologue(const MachineBasicBlock &MBB) const {
    return true;
  }

  /// Check whether or not the given \p MBB can be used as a epilogue
  /// for the target.
  /// The epilogue will be inserted before the first terminator of that block.
  /// This method is used by the shrink-wrapping pass to decide if
  /// \p MBB will be correctly handled by the target.
  /// As soon as the target enable shrink-wrapping without overriding
  /// this method, we assume that each basic block is a valid
  /// epilogue.
  virtual bool canUseAsEpilogue(const MachineBasicBlock &MBB) const {
    return true;
  }
};

} // End llvm namespace

#endif
