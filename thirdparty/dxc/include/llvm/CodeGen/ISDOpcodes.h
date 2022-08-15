//===-- llvm/CodeGen/ISDOpcodes.h - CodeGen opcodes -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares codegen opcodes and related utilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_ISDOPCODES_H
#define LLVM_CODEGEN_ISDOPCODES_H

namespace llvm {

/// ISD namespace - This namespace contains an enum which represents all of the
/// SelectionDAG node types and value types.
///
namespace ISD {

  //===--------------------------------------------------------------------===//
  /// ISD::NodeType enum - This enum defines the target-independent operators
  /// for a SelectionDAG.
  ///
  /// Targets may also define target-dependent operator codes for SDNodes. For
  /// example, on x86, these are the enum values in the X86ISD namespace.
  /// Targets should aim to use target-independent operators to model their
  /// instruction sets as much as possible, and only use target-dependent
  /// operators when they have special requirements.
  ///
  /// Finally, during and after selection proper, SNodes may use special
  /// operator codes that correspond directly with MachineInstr opcodes. These
  /// are used to represent selected instructions. See the isMachineOpcode()
  /// and getMachineOpcode() member functions of SDNode.
  ///
  enum NodeType {
    /// DELETED_NODE - This is an illegal value that is used to catch
    /// errors.  This opcode is not a legal opcode for any node.
    DELETED_NODE,

    /// EntryToken - This is the marker used to indicate the start of a region.
    EntryToken,

    /// TokenFactor - This node takes multiple tokens as input and produces a
    /// single token result. This is used to represent the fact that the operand
    /// operators are independent of each other.
    TokenFactor,

    /// AssertSext, AssertZext - These nodes record if a register contains a
    /// value that has already been zero or sign extended from a narrower type.
    /// These nodes take two operands.  The first is the node that has already
    /// been extended, and the second is a value type node indicating the width
    /// of the extension
    AssertSext, AssertZext,

    /// Various leaf nodes.
    BasicBlock, VALUETYPE, CONDCODE, Register, RegisterMask,
    Constant, ConstantFP,
    GlobalAddress, GlobalTLSAddress, FrameIndex,
    JumpTable, ConstantPool, ExternalSymbol, BlockAddress,

    /// The address of the GOT
    GLOBAL_OFFSET_TABLE,

    /// FRAMEADDR, RETURNADDR - These nodes represent llvm.frameaddress and
    /// llvm.returnaddress on the DAG.  These nodes take one operand, the index
    /// of the frame or return address to return.  An index of zero corresponds
    /// to the current function's frame or return address, an index of one to
    /// the parent's frame or return address, and so on.
    FRAMEADDR, RETURNADDR,

    /// LOCAL_RECOVER - Represents the llvm.localrecover intrinsic.
    /// Materializes the offset from the local object pointer of another
    /// function to a particular local object passed to llvm.localescape. The
    /// operand is the MCSymbol label used to represent this offset, since
    /// typically the offset is not known until after code generation of the
    /// parent.
    LOCAL_RECOVER,

    /// READ_REGISTER, WRITE_REGISTER - This node represents llvm.register on
    /// the DAG, which implements the named register global variables extension.
    READ_REGISTER,
    WRITE_REGISTER,

    /// FRAME_TO_ARGS_OFFSET - This node represents offset from frame pointer to
    /// first (possible) on-stack argument. This is needed for correct stack
    /// adjustment during unwind.
    FRAME_TO_ARGS_OFFSET,

    /// OUTCHAIN = EH_RETURN(INCHAIN, OFFSET, HANDLER) - This node represents
    /// 'eh_return' gcc dwarf builtin, which is used to return from
    /// exception. The general meaning is: adjust stack by OFFSET and pass
    /// execution to HANDLER. Many platform-related details also :)
    EH_RETURN,

    /// RESULT, OUTCHAIN = EH_SJLJ_SETJMP(INCHAIN, buffer)
    /// This corresponds to the eh.sjlj.setjmp intrinsic.
    /// It takes an input chain and a pointer to the jump buffer as inputs
    /// and returns an outchain.
    EH_SJLJ_SETJMP,

    /// OUTCHAIN = EH_SJLJ_LONGJMP(INCHAIN, buffer)
    /// This corresponds to the eh.sjlj.longjmp intrinsic.
    /// It takes an input chain and a pointer to the jump buffer as inputs
    /// and returns an outchain.
    EH_SJLJ_LONGJMP,

    /// TargetConstant* - Like Constant*, but the DAG does not do any folding,
    /// simplification, or lowering of the constant. They are used for constants
    /// which are known to fit in the immediate fields of their users, or for
    /// carrying magic numbers which are not values which need to be
    /// materialized in registers.
    TargetConstant,
    TargetConstantFP,

    /// TargetGlobalAddress - Like GlobalAddress, but the DAG does no folding or
    /// anything else with this node, and this is valid in the target-specific
    /// dag, turning into a GlobalAddress operand.
    TargetGlobalAddress,
    TargetGlobalTLSAddress,
    TargetFrameIndex,
    TargetJumpTable,
    TargetConstantPool,
    TargetExternalSymbol,
    TargetBlockAddress,

    MCSymbol,

    /// TargetIndex - Like a constant pool entry, but with completely
    /// target-dependent semantics. Holds target flags, a 32-bit index, and a
    /// 64-bit index. Targets can use this however they like.
    TargetIndex,

    /// RESULT = INTRINSIC_WO_CHAIN(INTRINSICID, arg1, arg2, ...)
    /// This node represents a target intrinsic function with no side effects.
    /// The first operand is the ID number of the intrinsic from the
    /// llvm::Intrinsic namespace.  The operands to the intrinsic follow.  The
    /// node returns the result of the intrinsic.
    INTRINSIC_WO_CHAIN,

    /// RESULT,OUTCHAIN = INTRINSIC_W_CHAIN(INCHAIN, INTRINSICID, arg1, ...)
    /// This node represents a target intrinsic function with side effects that
    /// returns a result.  The first operand is a chain pointer.  The second is
    /// the ID number of the intrinsic from the llvm::Intrinsic namespace.  The
    /// operands to the intrinsic follow.  The node has two results, the result
    /// of the intrinsic and an output chain.
    INTRINSIC_W_CHAIN,

    /// OUTCHAIN = INTRINSIC_VOID(INCHAIN, INTRINSICID, arg1, arg2, ...)
    /// This node represents a target intrinsic function with side effects that
    /// does not return a result.  The first operand is a chain pointer.  The
    /// second is the ID number of the intrinsic from the llvm::Intrinsic
    /// namespace.  The operands to the intrinsic follow.
    INTRINSIC_VOID,

    /// CopyToReg - This node has three operands: a chain, a register number to
    /// set to this value, and a value.
    CopyToReg,

    /// CopyFromReg - This node indicates that the input value is a virtual or
    /// physical register that is defined outside of the scope of this
    /// SelectionDAG.  The register is available from the RegisterSDNode object.
    CopyFromReg,

    /// UNDEF - An undefined node.
    UNDEF,

    /// EXTRACT_ELEMENT - This is used to get the lower or upper (determined by
    /// a Constant, which is required to be operand #1) half of the integer or
    /// float value specified as operand #0.  This is only for use before
    /// legalization, for values that will be broken into multiple registers.
    EXTRACT_ELEMENT,

    /// BUILD_PAIR - This is the opposite of EXTRACT_ELEMENT in some ways.
    /// Given two values of the same integer value type, this produces a value
    /// twice as big.  Like EXTRACT_ELEMENT, this can only be used before
    /// legalization.
    BUILD_PAIR,

    /// MERGE_VALUES - This node takes multiple discrete operands and returns
    /// them all as its individual results.  This nodes has exactly the same
    /// number of inputs and outputs. This node is useful for some pieces of the
    /// code generator that want to think about a single node with multiple
    /// results, not multiple nodes.
    MERGE_VALUES,

    /// Simple integer binary arithmetic operators.
    ADD, SUB, MUL, SDIV, UDIV, SREM, UREM,

    /// SMUL_LOHI/UMUL_LOHI - Multiply two integers of type iN, producing
    /// a signed/unsigned value of type i[2*N], and return the full value as
    /// two results, each of type iN.
    SMUL_LOHI, UMUL_LOHI,

    /// SDIVREM/UDIVREM - Divide two integers and produce both a quotient and
    /// remainder result.
    SDIVREM, UDIVREM,

    /// CARRY_FALSE - This node is used when folding other nodes,
    /// like ADDC/SUBC, which indicate the carry result is always false.
    CARRY_FALSE,

    /// Carry-setting nodes for multiple precision addition and subtraction.
    /// These nodes take two operands of the same value type, and produce two
    /// results.  The first result is the normal add or sub result, the second
    /// result is the carry flag result.
    ADDC, SUBC,

    /// Carry-using nodes for multiple precision addition and subtraction. These
    /// nodes take three operands: The first two are the normal lhs and rhs to
    /// the add or sub, and the third is the input carry flag.  These nodes
    /// produce two results; the normal result of the add or sub, and the output
    /// carry flag.  These nodes both read and write a carry flag to allow them
    /// to them to be chained together for add and sub of arbitrarily large
    /// values.
    ADDE, SUBE,

    /// RESULT, BOOL = [SU]ADDO(LHS, RHS) - Overflow-aware nodes for addition.
    /// These nodes take two operands: the normal LHS and RHS to the add. They
    /// produce two results: the normal result of the add, and a boolean that
    /// indicates if an overflow occurred (*not* a flag, because it may be store
    /// to memory, etc.).  If the type of the boolean is not i1 then the high
    /// bits conform to getBooleanContents.
    /// These nodes are generated from llvm.[su]add.with.overflow intrinsics.
    SADDO, UADDO,

    /// Same for subtraction.
    SSUBO, USUBO,

    /// Same for multiplication.
    SMULO, UMULO,

    /// Simple binary floating point operators.
    FADD, FSUB, FMUL, FDIV, FREM,

    /// FMA - Perform a * b + c with no intermediate rounding step.
    FMA,

    /// FMAD - Perform a * b + c, while getting the same result as the
    /// separately rounded operations.
    FMAD,

    /// FCOPYSIGN(X, Y) - Return the value of X with the sign of Y.  NOTE: This
    /// DAG node does not require that X and Y have the same type, just that
    /// they are both floating point.  X and the result must have the same type.
    /// FCOPYSIGN(f32, f64) is allowed.
    FCOPYSIGN,

    /// INT = FGETSIGN(FP) - Return the sign bit of the specified floating point
    /// value as an integer 0/1 value.
    FGETSIGN,

    /// BUILD_VECTOR(ELT0, ELT1, ELT2, ELT3,...) - Return a vector with the
    /// specified, possibly variable, elements.  The number of elements is
    /// required to be a power of two.  The types of the operands must all be
    /// the same and must match the vector element type, except that integer
    /// types are allowed to be larger than the element type, in which case
    /// the operands are implicitly truncated.
    BUILD_VECTOR,

    /// INSERT_VECTOR_ELT(VECTOR, VAL, IDX) - Returns VECTOR with the element
    /// at IDX replaced with VAL.  If the type of VAL is larger than the vector
    /// element type then VAL is truncated before replacement.
    INSERT_VECTOR_ELT,

    /// EXTRACT_VECTOR_ELT(VECTOR, IDX) - Returns a single element from VECTOR
    /// identified by the (potentially variable) element number IDX.  If the
    /// return type is an integer type larger than the element type of the
    /// vector, the result is extended to the width of the return type.
    EXTRACT_VECTOR_ELT,

    /// CONCAT_VECTORS(VECTOR0, VECTOR1, ...) - Given a number of values of
    /// vector type with the same length and element type, this produces a
    /// concatenated vector result value, with length equal to the sum of the
    /// lengths of the input vectors.
    CONCAT_VECTORS,

    /// INSERT_SUBVECTOR(VECTOR1, VECTOR2, IDX) - Returns a vector
    /// with VECTOR2 inserted into VECTOR1 at the (potentially
    /// variable) element number IDX, which must be a multiple of the
    /// VECTOR2 vector length.  The elements of VECTOR1 starting at
    /// IDX are overwritten with VECTOR2.  Elements IDX through
    /// vector_length(VECTOR2) must be valid VECTOR1 indices.
    INSERT_SUBVECTOR,

    /// EXTRACT_SUBVECTOR(VECTOR, IDX) - Returns a subvector from VECTOR (an
    /// vector value) starting with the element number IDX, which must be a
    /// constant multiple of the result vector length.
    EXTRACT_SUBVECTOR,

    /// VECTOR_SHUFFLE(VEC1, VEC2) - Returns a vector, of the same type as
    /// VEC1/VEC2.  A VECTOR_SHUFFLE node also contains an array of constant int
    /// values that indicate which value (or undef) each result element will
    /// get.  These constant ints are accessible through the
    /// ShuffleVectorSDNode class.  This is quite similar to the Altivec
    /// 'vperm' instruction, except that the indices must be constants and are
    /// in terms of the element size of VEC1/VEC2, not in terms of bytes.
    VECTOR_SHUFFLE,

    /// SCALAR_TO_VECTOR(VAL) - This represents the operation of loading a
    /// scalar value into element 0 of the resultant vector type.  The top
    /// elements 1 to N-1 of the N-element vector are undefined.  The type
    /// of the operand must match the vector element type, except when they
    /// are integer types.  In this case the operand is allowed to be wider
    /// than the vector element type, and is implicitly truncated to it.
    SCALAR_TO_VECTOR,

    /// MULHU/MULHS - Multiply high - Multiply two integers of type iN,
    /// producing an unsigned/signed value of type i[2*N], then return the top
    /// part.
    MULHU, MULHS,

    /// [US]{MIN/MAX} - Binary minimum or maximum or signed or unsigned
    /// integers.
    SMIN, SMAX, UMIN, UMAX,

    /// Bitwise operators - logical and, logical or, logical xor.
    AND, OR, XOR,

    /// Shift and rotation operations.  After legalization, the type of the
    /// shift amount is known to be TLI.getShiftAmountTy().  Before legalization
    /// the shift amount can be any type, but care must be taken to ensure it is
    /// large enough.  TLI.getShiftAmountTy() is i8 on some targets, but before
    /// legalization, types like i1024 can occur and i8 doesn't have enough bits
    /// to represent the shift amount.
    /// When the 1st operand is a vector, the shift amount must be in the same
    /// type. (TLI.getShiftAmountTy() will return the same type when the input
    /// type is a vector.)
    SHL, SRA, SRL, ROTL, ROTR,

    /// Byte Swap and Counting operators.
    BSWAP, CTTZ, CTLZ, CTPOP,

    /// Bit counting operators with an undefined result for zero inputs.
    CTTZ_ZERO_UNDEF, CTLZ_ZERO_UNDEF,

    /// Select(COND, TRUEVAL, FALSEVAL).  If the type of the boolean COND is not
    /// i1 then the high bits must conform to getBooleanContents.
    SELECT,

    /// Select with a vector condition (op #0) and two vector operands (ops #1
    /// and #2), returning a vector result.  All vectors have the same length.
    /// Much like the scalar select and setcc, each bit in the condition selects
    /// whether the corresponding result element is taken from op #1 or op #2.
    /// At first, the VSELECT condition is of vXi1 type. Later, targets may
    /// change the condition type in order to match the VSELECT node using a
    /// pattern. The condition follows the BooleanContent format of the target.
    VSELECT,

    /// Select with condition operator - This selects between a true value and
    /// a false value (ops #2 and #3) based on the boolean result of comparing
    /// the lhs and rhs (ops #0 and #1) of a conditional expression with the
    /// condition code in op #4, a CondCodeSDNode.
    SELECT_CC,

    /// SetCC operator - This evaluates to a true value iff the condition is
    /// true.  If the result value type is not i1 then the high bits conform
    /// to getBooleanContents.  The operands to this are the left and right
    /// operands to compare (ops #0, and #1) and the condition code to compare
    /// them with (op #2) as a CondCodeSDNode. If the operands are vector types
    /// then the result type must also be a vector type.
    SETCC,

    /// SHL_PARTS/SRA_PARTS/SRL_PARTS - These operators are used for expanded
    /// integer shift operations, just like ADD/SUB_PARTS.  The operation
    /// ordering is:
    ///       [Lo,Hi] = op [LoLHS,HiLHS], Amt
    SHL_PARTS, SRA_PARTS, SRL_PARTS,

    /// Conversion operators.  These are all single input single output
    /// operations.  For all of these, the result type must be strictly
    /// wider or narrower (depending on the operation) than the source
    /// type.

    /// SIGN_EXTEND - Used for integer types, replicating the sign bit
    /// into new bits.
    SIGN_EXTEND,

    /// ZERO_EXTEND - Used for integer types, zeroing the new bits.
    ZERO_EXTEND,

    /// ANY_EXTEND - Used for integer types.  The high bits are undefined.
    ANY_EXTEND,

    /// TRUNCATE - Completely drop the high bits.
    TRUNCATE,

    /// [SU]INT_TO_FP - These operators convert integers (whose interpreted sign
    /// depends on the first letter) to floating point.
    SINT_TO_FP,
    UINT_TO_FP,

    /// SIGN_EXTEND_INREG - This operator atomically performs a SHL/SRA pair to
    /// sign extend a small value in a large integer register (e.g. sign
    /// extending the low 8 bits of a 32-bit register to fill the top 24 bits
    /// with the 7th bit).  The size of the smaller type is indicated by the 1th
    /// operand, a ValueType node.
    SIGN_EXTEND_INREG,

    /// ANY_EXTEND_VECTOR_INREG(Vector) - This operator represents an
    /// in-register any-extension of the low lanes of an integer vector. The
    /// result type must have fewer elements than the operand type, and those
    /// elements must be larger integer types such that the total size of the
    /// operand type and the result type match. Each of the low operand
    /// elements is any-extended into the corresponding, wider result
    /// elements with the high bits becoming undef.
    ANY_EXTEND_VECTOR_INREG,

    /// SIGN_EXTEND_VECTOR_INREG(Vector) - This operator represents an
    /// in-register sign-extension of the low lanes of an integer vector. The
    /// result type must have fewer elements than the operand type, and those
    /// elements must be larger integer types such that the total size of the
    /// operand type and the result type match. Each of the low operand
    /// elements is sign-extended into the corresponding, wider result
    /// elements.
    // FIXME: The SIGN_EXTEND_INREG node isn't specifically limited to
    // scalars, but it also doesn't handle vectors well. Either it should be
    // restricted to scalars or this node (and its handling) should be merged
    // into it.
    SIGN_EXTEND_VECTOR_INREG,

    /// ZERO_EXTEND_VECTOR_INREG(Vector) - This operator represents an
    /// in-register zero-extension of the low lanes of an integer vector. The
    /// result type must have fewer elements than the operand type, and those
    /// elements must be larger integer types such that the total size of the
    /// operand type and the result type match. Each of the low operand
    /// elements is zero-extended into the corresponding, wider result
    /// elements.
    ZERO_EXTEND_VECTOR_INREG,

    /// FP_TO_[US]INT - Convert a floating point value to a signed or unsigned
    /// integer.
    FP_TO_SINT,
    FP_TO_UINT,

    /// X = FP_ROUND(Y, TRUNC) - Rounding 'Y' from a larger floating point type
    /// down to the precision of the destination VT.  TRUNC is a flag, which is
    /// always an integer that is zero or one.  If TRUNC is 0, this is a
    /// normal rounding, if it is 1, this FP_ROUND is known to not change the
    /// value of Y.
    ///
    /// The TRUNC = 1 case is used in cases where we know that the value will
    /// not be modified by the node, because Y is not using any of the extra
    /// precision of source type.  This allows certain transformations like
    /// FP_EXTEND(FP_ROUND(X,1)) -> X which are not safe for
    /// FP_EXTEND(FP_ROUND(X,0)) because the extra bits aren't removed.
    FP_ROUND,

    /// FLT_ROUNDS_ - Returns current rounding mode:
    /// -1 Undefined
    ///  0 Round to 0
    ///  1 Round to nearest
    ///  2 Round to +inf
    ///  3 Round to -inf
    FLT_ROUNDS_,

    /// X = FP_ROUND_INREG(Y, VT) - This operator takes an FP register, and
    /// rounds it to a floating point value.  It then promotes it and returns it
    /// in a register of the same size.  This operation effectively just
    /// discards excess precision.  The type to round down to is specified by
    /// the VT operand, a VTSDNode.
    FP_ROUND_INREG,

    /// X = FP_EXTEND(Y) - Extend a smaller FP type into a larger FP type.
    FP_EXTEND,

    /// BITCAST - This operator converts between integer, vector and FP
    /// values, as if the value was stored to memory with one type and loaded
    /// from the same address with the other type (or equivalently for vector
    /// format conversions, etc).  The source and result are required to have
    /// the same bit size (e.g.  f32 <-> i32).  This can also be used for
    /// int-to-int or fp-to-fp conversions, but that is a noop, deleted by
    /// getNode().
    BITCAST,

    /// ADDRSPACECAST - This operator converts between pointers of different
    /// address spaces.
    ADDRSPACECAST,

    /// CONVERT_RNDSAT - This operator is used to support various conversions
    /// between various types (float, signed, unsigned and vectors of those
    /// types) with rounding and saturation. NOTE: Avoid using this operator as
    /// most target don't support it and the operator might be removed in the
    /// future. It takes the following arguments:
    ///   0) value
    ///   1) dest type (type to convert to)
    ///   2) src type (type to convert from)
    ///   3) rounding imm
    ///   4) saturation imm
    ///   5) ISD::CvtCode indicating the type of conversion to do
    CONVERT_RNDSAT,

    /// FP16_TO_FP, FP_TO_FP16 - These operators are used to perform promotions
    /// and truncation for half-precision (16 bit) floating numbers. These nodes
    /// form a semi-softened interface for dealing with f16 (as an i16), which
    /// is often a storage-only type but has native conversions.
    FP16_TO_FP, FP_TO_FP16,

    /// FNEG, FABS, FSQRT, FSIN, FCOS, FPOWI, FPOW,
    /// FLOG, FLOG2, FLOG10, FEXP, FEXP2,
    /// FCEIL, FTRUNC, FRINT, FNEARBYINT, FROUND, FFLOOR - Perform various unary
    /// floating point operations. These are inspired by libm.
    FNEG, FABS, FSQRT, FSIN, FCOS, FPOWI, FPOW,
    FLOG, FLOG2, FLOG10, FEXP, FEXP2,
    FCEIL, FTRUNC, FRINT, FNEARBYINT, FROUND, FFLOOR,
    FMINNUM, FMAXNUM,

    /// FSINCOS - Compute both fsin and fcos as a single operation.
    FSINCOS,

    /// LOAD and STORE have token chains as their first operand, then the same
    /// operands as an LLVM load/store instruction, then an offset node that
    /// is added / subtracted from the base pointer to form the address (for
    /// indexed memory ops).
    LOAD, STORE,

    /// DYNAMIC_STACKALLOC - Allocate some number of bytes on the stack aligned
    /// to a specified boundary.  This node always has two return values: a new
    /// stack pointer value and a chain. The first operand is the token chain,
    /// the second is the number of bytes to allocate, and the third is the
    /// alignment boundary.  The size is guaranteed to be a multiple of the
    /// stack alignment, and the alignment is guaranteed to be bigger than the
    /// stack alignment (if required) or 0 to get standard stack alignment.
    DYNAMIC_STACKALLOC,

    /// Control flow instructions.  These all have token chains.

    /// BR - Unconditional branch.  The first operand is the chain
    /// operand, the second is the MBB to branch to.
    BR,

    /// BRIND - Indirect branch.  The first operand is the chain, the second
    /// is the value to branch to, which must be of the same type as the
    /// target's pointer type.
    BRIND,

    /// BR_JT - Jumptable branch. The first operand is the chain, the second
    /// is the jumptable index, the last one is the jumptable entry index.
    BR_JT,

    /// BRCOND - Conditional branch.  The first operand is the chain, the
    /// second is the condition, the third is the block to branch to if the
    /// condition is true.  If the type of the condition is not i1, then the
    /// high bits must conform to getBooleanContents.
    BRCOND,

    /// BR_CC - Conditional branch.  The behavior is like that of SELECT_CC, in
    /// that the condition is represented as condition code, and two nodes to
    /// compare, rather than as a combined SetCC node.  The operands in order
    /// are chain, cc, lhs, rhs, block to branch to if condition is true.
    BR_CC,

    /// INLINEASM - Represents an inline asm block.  This node always has two
    /// return values: a chain and a flag result.  The inputs are as follows:
    ///   Operand #0  : Input chain.
    ///   Operand #1  : a ExternalSymbolSDNode with a pointer to the asm string.
    ///   Operand #2  : a MDNodeSDNode with the !srcloc metadata.
    ///   Operand #3  : HasSideEffect, IsAlignStack bits.
    ///   After this, it is followed by a list of operands with this format:
    ///     ConstantSDNode: Flags that encode whether it is a mem or not, the
    ///                     of operands that follow, etc.  See InlineAsm.h.
    ///     ... however many operands ...
    ///   Operand #last: Optional, an incoming flag.
    ///
    /// The variable width operands are required to represent target addressing
    /// modes as a single "operand", even though they may have multiple
    /// SDOperands.
    INLINEASM,

    /// EH_LABEL - Represents a label in mid basic block used to track
    /// locations needed for debug and exception handling tables.  These nodes
    /// take a chain as input and return a chain.
    EH_LABEL,

    /// STACKSAVE - STACKSAVE has one operand, an input chain.  It produces a
    /// value, the same type as the pointer type for the system, and an output
    /// chain.
    STACKSAVE,

    /// STACKRESTORE has two operands, an input chain and a pointer to restore
    /// to it returns an output chain.
    STACKRESTORE,

    /// CALLSEQ_START/CALLSEQ_END - These operators mark the beginning and end
    /// of a call sequence, and carry arbitrary information that target might
    /// want to know.  The first operand is a chain, the rest are specified by
    /// the target and not touched by the DAG optimizers.
    /// CALLSEQ_START..CALLSEQ_END pairs may not be nested.
    CALLSEQ_START,  // Beginning of a call sequence
    CALLSEQ_END,    // End of a call sequence

    /// VAARG - VAARG has four operands: an input chain, a pointer, a SRCVALUE,
    /// and the alignment. It returns a pair of values: the vaarg value and a
    /// new chain.
    VAARG,

    /// VACOPY - VACOPY has 5 operands: an input chain, a destination pointer,
    /// a source pointer, a SRCVALUE for the destination, and a SRCVALUE for the
    /// source.
    VACOPY,

    /// VAEND, VASTART - VAEND and VASTART have three operands: an input chain,
    /// pointer, and a SRCVALUE.
    VAEND, VASTART,

    /// SRCVALUE - This is a node type that holds a Value* that is used to
    /// make reference to a value in the LLVM IR.
    SRCVALUE,

    /// MDNODE_SDNODE - This is a node that holdes an MDNode*, which is used to
    /// reference metadata in the IR.
    MDNODE_SDNODE,

    /// PCMARKER - This corresponds to the pcmarker intrinsic.
    PCMARKER,

    /// READCYCLECOUNTER - This corresponds to the readcyclecounter intrinsic.
    /// The only operand is a chain and a value and a chain are produced.  The
    /// value is the contents of the architecture specific cycle counter like
    /// register (or other high accuracy low latency clock source)
    READCYCLECOUNTER,

    /// HANDLENODE node - Used as a handle for various purposes.
    HANDLENODE,

    /// INIT_TRAMPOLINE - This corresponds to the init_trampoline intrinsic.  It
    /// takes as input a token chain, the pointer to the trampoline, the pointer
    /// to the nested function, the pointer to pass for the 'nest' parameter, a
    /// SRCVALUE for the trampoline and another for the nested function
    /// (allowing targets to access the original Function*).
    /// It produces a token chain as output.
    INIT_TRAMPOLINE,

    /// ADJUST_TRAMPOLINE - This corresponds to the adjust_trampoline intrinsic.
    /// It takes a pointer to the trampoline and produces a (possibly) new
    /// pointer to the same trampoline with platform-specific adjustments
    /// applied.  The pointer it returns points to an executable block of code.
    ADJUST_TRAMPOLINE,

    /// TRAP - Trapping instruction
    TRAP,

    /// DEBUGTRAP - Trap intended to get the attention of a debugger.
    DEBUGTRAP,

    /// PREFETCH - This corresponds to a prefetch intrinsic. The first operand
    /// is the chain.  The other operands are the address to prefetch,
    /// read / write specifier, locality specifier and instruction / data cache
    /// specifier.
    PREFETCH,

    /// OUTCHAIN = ATOMIC_FENCE(INCHAIN, ordering, scope)
    /// This corresponds to the fence instruction. It takes an input chain, and
    /// two integer constants: an AtomicOrdering and a SynchronizationScope.
    ATOMIC_FENCE,

    /// Val, OUTCHAIN = ATOMIC_LOAD(INCHAIN, ptr)
    /// This corresponds to "load atomic" instruction.
    ATOMIC_LOAD,

    /// OUTCHAIN = ATOMIC_STORE(INCHAIN, ptr, val)
    /// This corresponds to "store atomic" instruction.
    ATOMIC_STORE,

    /// Val, OUTCHAIN = ATOMIC_CMP_SWAP(INCHAIN, ptr, cmp, swap)
    /// For double-word atomic operations:
    /// ValLo, ValHi, OUTCHAIN = ATOMIC_CMP_SWAP(INCHAIN, ptr, cmpLo, cmpHi,
    ///                                          swapLo, swapHi)
    /// This corresponds to the cmpxchg instruction.
    ATOMIC_CMP_SWAP,

    /// Val, Success, OUTCHAIN
    ///     = ATOMIC_CMP_SWAP_WITH_SUCCESS(INCHAIN, ptr, cmp, swap)
    /// N.b. this is still a strong cmpxchg operation, so
    /// Success == "Val == cmp".
    ATOMIC_CMP_SWAP_WITH_SUCCESS,

    /// Val, OUTCHAIN = ATOMIC_SWAP(INCHAIN, ptr, amt)
    /// Val, OUTCHAIN = ATOMIC_LOAD_[OpName](INCHAIN, ptr, amt)
    /// For double-word atomic operations:
    /// ValLo, ValHi, OUTCHAIN = ATOMIC_SWAP(INCHAIN, ptr, amtLo, amtHi)
    /// ValLo, ValHi, OUTCHAIN = ATOMIC_LOAD_[OpName](INCHAIN, ptr, amtLo, amtHi)
    /// These correspond to the atomicrmw instruction.
    ATOMIC_SWAP,
    ATOMIC_LOAD_ADD,
    ATOMIC_LOAD_SUB,
    ATOMIC_LOAD_AND,
    ATOMIC_LOAD_OR,
    ATOMIC_LOAD_XOR,
    ATOMIC_LOAD_NAND,
    ATOMIC_LOAD_MIN,
    ATOMIC_LOAD_MAX,
    ATOMIC_LOAD_UMIN,
    ATOMIC_LOAD_UMAX,

    // Masked load and store - consecutive vector load and store operations
    // with additional mask operand that prevents memory accesses to the
    // masked-off lanes.
    MLOAD, MSTORE,

    // Masked gather and scatter - load and store operations for a vector of
    // random addresses with additional mask operand that prevents memory
    // accesses to the masked-off lanes.
    MGATHER, MSCATTER,

    /// This corresponds to the llvm.lifetime.* intrinsics. The first operand
    /// is the chain and the second operand is the alloca pointer.
    LIFETIME_START, LIFETIME_END,

    /// GC_TRANSITION_START/GC_TRANSITION_END - These operators mark the
    /// beginning and end of GC transition  sequence, and carry arbitrary
    /// information that target might need for lowering.  The first operand is
    /// a chain, the rest are specified by the target and not touched by the DAG
    /// optimizers. GC_TRANSITION_START..GC_TRANSITION_END pairs may not be
    /// nested.
    GC_TRANSITION_START,
    GC_TRANSITION_END,

    /// BUILTIN_OP_END - This must be the last enum value in this list.
    /// The target-specific pre-isel opcode values start here.
    BUILTIN_OP_END
  };

  /// FIRST_TARGET_MEMORY_OPCODE - Target-specific pre-isel operations
  /// which do not reference a specific memory location should be less than
  /// this value. Those that do must not be less than this value, and can
  /// be used with SelectionDAG::getMemIntrinsicNode.
  static const int FIRST_TARGET_MEMORY_OPCODE = BUILTIN_OP_END+300;

  //===--------------------------------------------------------------------===//
  /// MemIndexedMode enum - This enum defines the load / store indexed
  /// addressing modes.
  ///
  /// UNINDEXED    "Normal" load / store. The effective address is already
  ///              computed and is available in the base pointer. The offset
  ///              operand is always undefined. In addition to producing a
  ///              chain, an unindexed load produces one value (result of the
  ///              load); an unindexed store does not produce a value.
  ///
  /// PRE_INC      Similar to the unindexed mode where the effective address is
  /// PRE_DEC      the value of the base pointer add / subtract the offset.
  ///              It considers the computation as being folded into the load /
  ///              store operation (i.e. the load / store does the address
  ///              computation as well as performing the memory transaction).
  ///              The base operand is always undefined. In addition to
  ///              producing a chain, pre-indexed load produces two values
  ///              (result of the load and the result of the address
  ///              computation); a pre-indexed store produces one value (result
  ///              of the address computation).
  ///
  /// POST_INC     The effective address is the value of the base pointer. The
  /// POST_DEC     value of the offset operand is then added to / subtracted
  ///              from the base after memory transaction. In addition to
  ///              producing a chain, post-indexed load produces two values
  ///              (the result of the load and the result of the base +/- offset
  ///              computation); a post-indexed store produces one value (the
  ///              the result of the base +/- offset computation).
  enum MemIndexedMode {
    UNINDEXED = 0,
    PRE_INC,
    PRE_DEC,
    POST_INC,
    POST_DEC,
    LAST_INDEXED_MODE
  };

  //===--------------------------------------------------------------------===//
  /// LoadExtType enum - This enum defines the three variants of LOADEXT
  /// (load with extension).
  ///
  /// SEXTLOAD loads the integer operand and sign extends it to a larger
  ///          integer result type.
  /// ZEXTLOAD loads the integer operand and zero extends it to a larger
  ///          integer result type.
  /// EXTLOAD  is used for two things: floating point extending loads and
  ///          integer extending loads [the top bits are undefined].
  enum LoadExtType {
    NON_EXTLOAD = 0,
    EXTLOAD,
    SEXTLOAD,
    ZEXTLOAD,
    LAST_LOADEXT_TYPE
  };

  NodeType getExtForLoadExtType(bool IsFP, LoadExtType);

  //===--------------------------------------------------------------------===//
  /// ISD::CondCode enum - These are ordered carefully to make the bitfields
  /// below work out, when considering SETFALSE (something that never exists
  /// dynamically) as 0.  "U" -> Unsigned (for integer operands) or Unordered
  /// (for floating point), "L" -> Less than, "G" -> Greater than, "E" -> Equal
  /// to.  If the "N" column is 1, the result of the comparison is undefined if
  /// the input is a NAN.
  ///
  /// All of these (except for the 'always folded ops') should be handled for
  /// floating point.  For integer, only the SETEQ,SETNE,SETLT,SETLE,SETGT,
  /// SETGE,SETULT,SETULE,SETUGT, and SETUGE opcodes are used.
  ///
  /// Note that these are laid out in a specific order to allow bit-twiddling
  /// to transform conditions.
  enum CondCode {
    // Opcode          N U L G E       Intuitive operation
    SETFALSE,      //    0 0 0 0       Always false (always folded)
    SETOEQ,        //    0 0 0 1       True if ordered and equal
    SETOGT,        //    0 0 1 0       True if ordered and greater than
    SETOGE,        //    0 0 1 1       True if ordered and greater than or equal
    SETOLT,        //    0 1 0 0       True if ordered and less than
    SETOLE,        //    0 1 0 1       True if ordered and less than or equal
    SETONE,        //    0 1 1 0       True if ordered and operands are unequal
    SETO,          //    0 1 1 1       True if ordered (no nans)
    SETUO,         //    1 0 0 0       True if unordered: isnan(X) | isnan(Y)
    SETUEQ,        //    1 0 0 1       True if unordered or equal
    SETUGT,        //    1 0 1 0       True if unordered or greater than
    SETUGE,        //    1 0 1 1       True if unordered, greater than, or equal
    SETULT,        //    1 1 0 0       True if unordered or less than
    SETULE,        //    1 1 0 1       True if unordered, less than, or equal
    SETUNE,        //    1 1 1 0       True if unordered or not equal
    SETTRUE,       //    1 1 1 1       Always true (always folded)
    // Don't care operations: undefined if the input is a nan.
    SETFALSE2,     //  1 X 0 0 0       Always false (always folded)
    SETEQ,         //  1 X 0 0 1       True if equal
    SETGT,         //  1 X 0 1 0       True if greater than
    SETGE,         //  1 X 0 1 1       True if greater than or equal
    SETLT,         //  1 X 1 0 0       True if less than
    SETLE,         //  1 X 1 0 1       True if less than or equal
    SETNE,         //  1 X 1 1 0       True if not equal
    SETTRUE2,      //  1 X 1 1 1       Always true (always folded)

    SETCC_INVALID       // Marker value.
  };

  /// isSignedIntSetCC - Return true if this is a setcc instruction that
  /// performs a signed comparison when used with integer operands.
  inline bool isSignedIntSetCC(CondCode Code) {
    return Code == SETGT || Code == SETGE || Code == SETLT || Code == SETLE;
  }

  /// isUnsignedIntSetCC - Return true if this is a setcc instruction that
  /// performs an unsigned comparison when used with integer operands.
  inline bool isUnsignedIntSetCC(CondCode Code) {
    return Code == SETUGT || Code == SETUGE || Code == SETULT || Code == SETULE;
  }

  /// isTrueWhenEqual - Return true if the specified condition returns true if
  /// the two operands to the condition are equal.  Note that if one of the two
  /// operands is a NaN, this value is meaningless.
  inline bool isTrueWhenEqual(CondCode Cond) {
    return ((int)Cond & 1) != 0;
  }

  /// getUnorderedFlavor - This function returns 0 if the condition is always
  /// false if an operand is a NaN, 1 if the condition is always true if the
  /// operand is a NaN, and 2 if the condition is undefined if the operand is a
  /// NaN.
  inline unsigned getUnorderedFlavor(CondCode Cond) {
    return ((int)Cond >> 3) & 3;
  }

  /// getSetCCInverse - Return the operation corresponding to !(X op Y), where
  /// 'op' is a valid SetCC operation.
  CondCode getSetCCInverse(CondCode Operation, bool isInteger);

  /// getSetCCSwappedOperands - Return the operation corresponding to (Y op X)
  /// when given the operation for (X op Y).
  CondCode getSetCCSwappedOperands(CondCode Operation);

  /// getSetCCOrOperation - Return the result of a logical OR between different
  /// comparisons of identical values: ((X op1 Y) | (X op2 Y)).  This
  /// function returns SETCC_INVALID if it is not possible to represent the
  /// resultant comparison.
  CondCode getSetCCOrOperation(CondCode Op1, CondCode Op2, bool isInteger);

  /// getSetCCAndOperation - Return the result of a logical AND between
  /// different comparisons of identical values: ((X op1 Y) & (X op2 Y)).  This
  /// function returns SETCC_INVALID if it is not possible to represent the
  /// resultant comparison.
  CondCode getSetCCAndOperation(CondCode Op1, CondCode Op2, bool isInteger);

  //===--------------------------------------------------------------------===//
  /// CvtCode enum - This enum defines the various converts CONVERT_RNDSAT
  /// supports.
  enum CvtCode {
    CVT_FF,     /// Float from Float
    CVT_FS,     /// Float from Signed
    CVT_FU,     /// Float from Unsigned
    CVT_SF,     /// Signed from Float
    CVT_UF,     /// Unsigned from Float
    CVT_SS,     /// Signed from Signed
    CVT_SU,     /// Signed from Unsigned
    CVT_US,     /// Unsigned from Signed
    CVT_UU,     /// Unsigned from Unsigned
    CVT_INVALID /// Marker - Invalid opcode
  };

} // end llvm::ISD namespace

} // end llvm namespace

#endif
