//===-- llvm/Target/TargetOpcodes.h - Target Indep Opcodes ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the target independent instruction opcodes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETOPCODES_H
#define LLVM_TARGET_TARGETOPCODES_H

namespace llvm {

/// Invariant opcodes: All instruction sets have these as their low opcodes.
///
/// Every instruction defined here must also appear in Target.td and the order
/// must be the same as in CodeGenTarget.cpp.
///
namespace TargetOpcode {
enum {
  PHI = 0,
  INLINEASM = 1,
  CFI_INSTRUCTION = 2,
  EH_LABEL = 3,
  GC_LABEL = 4,

  /// KILL - This instruction is a noop that is used only to adjust the
  /// liveness of registers. This can be useful when dealing with
  /// sub-registers.
  KILL = 5,

  /// EXTRACT_SUBREG - This instruction takes two operands: a register
  /// that has subregisters, and a subregister index. It returns the
  /// extracted subregister value. This is commonly used to implement
  /// truncation operations on target architectures which support it.
  EXTRACT_SUBREG = 6,

  /// INSERT_SUBREG - This instruction takes three operands: a register that
  /// has subregisters, a register providing an insert value, and a
  /// subregister index. It returns the value of the first register with the
  /// value of the second register inserted. The first register is often
  /// defined by an IMPLICIT_DEF, because it is commonly used to implement
  /// anyext operations on target architectures which support it.
  INSERT_SUBREG = 7,

  /// IMPLICIT_DEF - This is the MachineInstr-level equivalent of undef.
  IMPLICIT_DEF = 8,

  /// SUBREG_TO_REG - This instruction is similar to INSERT_SUBREG except that
  /// the first operand is an immediate integer constant. This constant is
  /// often zero, because it is commonly used to assert that the instruction
  /// defining the register implicitly clears the high bits.
  SUBREG_TO_REG = 9,

  /// COPY_TO_REGCLASS - This instruction is a placeholder for a plain
  /// register-to-register copy into a specific register class. This is only
  /// used between instruction selection and MachineInstr creation, before
  /// virtual registers have been created for all the instructions, and it's
  /// only needed in cases where the register classes implied by the
  /// instructions are insufficient. It is emitted as a COPY MachineInstr.
  COPY_TO_REGCLASS = 10,

  /// DBG_VALUE - a mapping of the llvm.dbg.value intrinsic
  DBG_VALUE = 11,

  /// REG_SEQUENCE - This variadic instruction is used to form a register that
  /// represents a consecutive sequence of sub-registers. It's used as a
  /// register coalescing / allocation aid and must be eliminated before code
  /// emission.
  // In SDNode form, the first operand encodes the register class created by
  // the REG_SEQUENCE, while each subsequent pair names a vreg + subreg index
  // pair.  Once it has been lowered to a MachineInstr, the regclass operand
  // is no longer present.
  /// e.g. v1027 = REG_SEQUENCE v1024, 3, v1025, 4, v1026, 5
  /// After register coalescing references of v1024 should be replace with
  /// v1027:3, v1025 with v1027:4, etc.
  REG_SEQUENCE = 12,

  /// COPY - Target-independent register copy. This instruction can also be
  /// used to copy between subregisters of virtual registers.
  COPY = 13,

  /// BUNDLE - This instruction represents an instruction bundle. Instructions
  /// which immediately follow a BUNDLE instruction which are marked with
  /// 'InsideBundle' flag are inside the bundle.
  BUNDLE = 14,

  /// Lifetime markers.
  LIFETIME_START = 15,
  LIFETIME_END = 16,

  /// A Stackmap instruction captures the location of live variables at its
  /// position in the instruction stream. It is followed by a shadow of bytes
  /// that must lie within the function and not contain another stackmap.
  STACKMAP = 17,

  /// Patchable call instruction - this instruction represents a call to a
  /// constant address, followed by a series of NOPs. It is intended to
  /// support optimizations for dynamic languages (such as javascript) that
  /// rewrite calls to runtimes with more efficient code sequences.
  /// This also implies a stack map.
  PATCHPOINT = 18,

  /// This pseudo-instruction loads the stack guard value. Targets which need
  /// to prevent the stack guard value or address from being spilled to the
  /// stack should override TargetLowering::emitLoadStackGuardNode and
  /// additionally expand this pseudo after register allocation.
  LOAD_STACK_GUARD = 19,

  /// Call instruction with associated vm state for deoptimization and list
  /// of live pointers for relocation by the garbage collector.  It is
  /// intended to support garbage collection with fully precise relocating
  /// collectors and deoptimizations in either the callee or caller.
  STATEPOINT = 20,

  /// Instruction that records the offset of a local stack allocation passed to
  /// llvm.localescape. It has two arguments: the symbol for the label and the
  /// frame index of the local stack allocation.
  LOCAL_ESCAPE = 21,

  /// Loading instruction that may page fault, bundled with associated
  /// information on how to handle such a page fault.  It is intended to support
  /// "zero cost" null checks in managed languages by allowing LLVM to fold
  /// comparisions into existing memory operations.
  FAULTING_LOAD_OP = 22,
};
} // end namespace TargetOpcode
} // end namespace llvm

#endif
