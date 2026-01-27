/*
 *    Stack-less Just-In-Time compiler
 *
 *    Copyright Zoltan Herczeg (hzmester@freemail.hu). All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are
 * permitted provided that the following conditions are met:
 *
 *   1. Redistributions of source code must retain the above copyright notice, this list of
 *      conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright notice, this list
 *      of conditions and the following disclaimer in the documentation and/or other materials
 *      provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER(S) AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDER(S) OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SLJIT_LIR_H_
#define SLJIT_LIR_H_

/*
   ------------------------------------------------------------------------
    Stack-Less JIT compiler for multiple architectures (x86, ARM, PowerPC)
   ------------------------------------------------------------------------

   Short description
    Advantages:
      - The execution can be continued from any LIR instruction. In other
        words, it is possible to jump to any label from anywhere, even from
        a code fragment, which is compiled later, as long as the compiling
        context is the same. See sljit_emit_enter for more details.
      - Supports self modifying code: target of any jump and call
        instructions and some constant values can be dynamically modified
        during runtime. See SLJIT_REWRITABLE_JUMP.
        - although it is not suggested to do it frequently
        - can be used for inline caching: save an important value once
          in the instruction stream
      - A fixed stack space can be allocated for local variables
      - The compiler is thread-safe
      - The compiler is highly configurable through preprocessor macros.
        You can disable unneeded features (multithreading in single
        threaded applications), and you can use your own system functions
        (including memory allocators). See sljitConfig.h.
    Disadvantages:
      - The compiler is more like a platform independent assembler, so
        there is no built-in variable management. Registers and stack must
        be managed manually (the name of the compiler refers to this).
    In practice:
      - This approach is very effective for interpreters
        - One of the saved registers typically points to a stack interface
        - It can jump to any exception handler anytime (even if it belongs
          to another function)
        - Hot paths can be modified during runtime reflecting the changes
          of the fastest execution path of the dynamic language
        - SLJIT supports complex memory addressing modes
        - mainly position and context independent code (except some cases)

    For valgrind users:
      - pass --smc-check=all argument to valgrind, since JIT is a "self-modifying code"
*/

#if (defined SLJIT_HAVE_CONFIG_PRE && SLJIT_HAVE_CONFIG_PRE)
#include "sljitConfigPre.h"
#endif /* SLJIT_HAVE_CONFIG_PRE */

#include "sljitConfigCPU.h"
#include "sljitConfig.h"

/* The following header file defines useful macros for fine tuning
SLJIT based code generators. They are listed in the beginning
of sljitConfigInternal.h */

#include "sljitConfigInternal.h"

#if (defined SLJIT_HAVE_CONFIG_POST && SLJIT_HAVE_CONFIG_POST)
#include "sljitConfigPost.h"
#endif /* SLJIT_HAVE_CONFIG_POST */

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* Version numbers. */
#define SLJIT_MAJOR_VERSION	0
#define SLJIT_MINOR_VERSION	95

/* --------------------------------------------------------------------- */
/*  Error codes                                                          */
/* --------------------------------------------------------------------- */

/* Indicates no error. */
#define SLJIT_SUCCESS			0
/* After the call of sljit_generate_code(), the error code of the compiler
   is set to this value to avoid further code generation.
   The complier should be freed after sljit_generate_code(). */
#define SLJIT_ERR_COMPILED		1
/* Cannot allocate non-executable memory. */
#define SLJIT_ERR_ALLOC_FAILED		2
/* Cannot allocate executable memory.
   Only sljit_generate_code() returns with this error code. */
#define SLJIT_ERR_EX_ALLOC_FAILED	3
/* Unsupported instruction form. */
#define SLJIT_ERR_UNSUPPORTED		4
/* An invalid argument is passed to any SLJIT function. */
#define SLJIT_ERR_BAD_ARGUMENT		5

/* --------------------------------------------------------------------- */
/*  Registers                                                            */
/* --------------------------------------------------------------------- */

/*
  Scratch (R) registers: registers which may not preserve their values
  across function calls.

  Saved (S) registers: registers which preserve their values across
  function calls.

  The scratch and saved register sets overlap. The last scratch register
  is the first saved register, the one before the last is the second saved
  register, and so on.

  For example, in an architecture with only five registers (A-E), if two
  are scratch and three saved registers, they will be defined as follows:

    A |   R0   |      |  R0 always represent scratch register A
    B |   R1   |      |  R1 always represent scratch register B
    C |  [R2]  |  S2  |  R2 and S2 represent the same physical register C
    D |  [R3]  |  S1  |  R3 and S1 represent the same physical register D
    E |  [R4]  |  S0  |  R4 and S0 represent the same physical register E

  Note: SLJIT_NUMBER_OF_SCRATCH_REGISTERS will be 2 and
        SLJIT_NUMBER_OF_SAVED_REGISTERS will be 3.

  Note: For all supported architectures SLJIT_NUMBER_OF_REGISTERS >= 12
        and SLJIT_NUMBER_OF_SAVED_REGISTERS >= 6. However, 6 registers
        are virtual on x86-32. See below.

  The purpose of this definition is convenience: saved registers can
  be used as extra scratch registers. For example, building in the
  previous example, four registers can be specified as scratch registers
  and the fifth one as saved register, allowing any user code which requires
  four scratch registers to run unmodified. The SLJIT compiler automatically
  saves the content of the two extra scratch register on the stack. Scratch
  registers can also be preserved by saving their value on the stack but
  that needs to be done manually.

  Note: To emphasize that registers assigned to R2-R4 are saved
        registers, they are enclosed by square brackets.

  Note: sljit_emit_enter and sljit_set_context define whether a register
        is S or R register. E.g: if in the previous example 3 scratches and
        1 saved are mapped by sljit_emit_enter, the allowed register set
        will be: R0-R2 and S0. Although S2 is mapped to the same register
        than R2, it is not available in that configuration. Furthermore
        the S1 register cannot be used at all.
*/

/* Scratch registers. */
#define SLJIT_R0	1
#define SLJIT_R1	2
#define SLJIT_R2	3
/* Note: on x86-32, R3 - R6 (same as S3 - S6) are emulated (they
   are allocated on the stack). These registers are called virtual
   and cannot be used for memory addressing (cannot be part of
   any SLJIT_MEM1, SLJIT_MEM2 construct). There is no such
   limitation on other CPUs. See sljit_get_register_index(). */
#define SLJIT_R3	4
#define SLJIT_R4	5
#define SLJIT_R5	6
#define SLJIT_R6	7
#define SLJIT_R7	8
#define SLJIT_R8	9
#define SLJIT_R9	10
/* All R registers provided by the architecture can be accessed by SLJIT_R(i)
   The i parameter must be >= 0 and < SLJIT_NUMBER_OF_REGISTERS. */
#define SLJIT_R(i)	(1 + (i))

/* Saved registers. */
#define SLJIT_S0	(SLJIT_NUMBER_OF_REGISTERS)
#define SLJIT_S1	(SLJIT_NUMBER_OF_REGISTERS - 1)
#define SLJIT_S2	(SLJIT_NUMBER_OF_REGISTERS - 2)
/* Note: on x86-32, S3 - S6 (same as R3 - R6) are emulated (they
   are allocated on the stack). These registers are called virtual
   and cannot be used for memory addressing (cannot be part of
   any SLJIT_MEM1, SLJIT_MEM2 construct). There is no such
   limitation on other CPUs. See sljit_get_register_index(). */
#define SLJIT_S3	(SLJIT_NUMBER_OF_REGISTERS - 3)
#define SLJIT_S4	(SLJIT_NUMBER_OF_REGISTERS - 4)
#define SLJIT_S5	(SLJIT_NUMBER_OF_REGISTERS - 5)
#define SLJIT_S6	(SLJIT_NUMBER_OF_REGISTERS - 6)
#define SLJIT_S7	(SLJIT_NUMBER_OF_REGISTERS - 7)
#define SLJIT_S8	(SLJIT_NUMBER_OF_REGISTERS - 8)
#define SLJIT_S9	(SLJIT_NUMBER_OF_REGISTERS - 9)
/* All S registers provided by the architecture can be accessed by SLJIT_S(i)
   The i parameter must be >= 0 and < SLJIT_NUMBER_OF_SAVED_REGISTERS. */
#define SLJIT_S(i)	(SLJIT_NUMBER_OF_REGISTERS - (i))

/* Registers >= SLJIT_FIRST_SAVED_REG are saved registers. */
#define SLJIT_FIRST_SAVED_REG (SLJIT_S0 - SLJIT_NUMBER_OF_SAVED_REGISTERS + 1)

/* The SLJIT_SP provides direct access to the linear stack space allocated by
   sljit_emit_enter. It can only be used in the following form: SLJIT_MEM1(SLJIT_SP).
   The immediate offset is extended by the relative stack offset automatically.
   sljit_get_local_base can be used to obtain the real address of a value. */
#define SLJIT_SP	(SLJIT_NUMBER_OF_REGISTERS + 1)

/* Return with machine word. */

#define SLJIT_RETURN_REG	SLJIT_R0

/* --------------------------------------------------------------------- */
/*  Floating point registers                                             */
/* --------------------------------------------------------------------- */

/* Each floating point register can store a 32 or a 64 bit precision
   value. The FR and FS register sets overlap in the same way as R
   and S register sets. See above. */

/* Floating point scratch registers. */
#define SLJIT_FR0	1
#define SLJIT_FR1	2
#define SLJIT_FR2	3
#define SLJIT_FR3	4
#define SLJIT_FR4	5
#define SLJIT_FR5	6
#define SLJIT_FR6	7
#define SLJIT_FR7	8
#define SLJIT_FR8	9
#define SLJIT_FR9	10
/* All FR registers provided by the architecture can be accessed by SLJIT_FR(i)
   The i parameter must be >= 0 and < SLJIT_NUMBER_OF_FLOAT_REGISTERS. */
#define SLJIT_FR(i)	(1 + (i))

/* Floating point saved registers. */
#define SLJIT_FS0	(SLJIT_NUMBER_OF_FLOAT_REGISTERS)
#define SLJIT_FS1	(SLJIT_NUMBER_OF_FLOAT_REGISTERS - 1)
#define SLJIT_FS2	(SLJIT_NUMBER_OF_FLOAT_REGISTERS - 2)
#define SLJIT_FS3	(SLJIT_NUMBER_OF_FLOAT_REGISTERS - 3)
#define SLJIT_FS4	(SLJIT_NUMBER_OF_FLOAT_REGISTERS - 4)
#define SLJIT_FS5	(SLJIT_NUMBER_OF_FLOAT_REGISTERS - 5)
#define SLJIT_FS6	(SLJIT_NUMBER_OF_FLOAT_REGISTERS - 6)
#define SLJIT_FS7	(SLJIT_NUMBER_OF_FLOAT_REGISTERS - 7)
#define SLJIT_FS8	(SLJIT_NUMBER_OF_FLOAT_REGISTERS - 8)
#define SLJIT_FS9	(SLJIT_NUMBER_OF_FLOAT_REGISTERS - 9)
/* All FS registers provided by the architecture can be accessed by SLJIT_FS(i)
   The i parameter must be >= 0 and < SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS. */
#define SLJIT_FS(i)	(SLJIT_NUMBER_OF_FLOAT_REGISTERS - (i))

/* Float registers >= SLJIT_FIRST_SAVED_FLOAT_REG are saved registers. */
#define SLJIT_FIRST_SAVED_FLOAT_REG (SLJIT_FS0 - SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS + 1)

/* Return with floating point arg. */

#define SLJIT_RETURN_FREG	SLJIT_FR0

/* --------------------------------------------------------------------- */
/*  Vector registers                                                     */
/* --------------------------------------------------------------------- */

/* Vector registers are storage areas, which are used for Single Instruction
   Multiple Data (SIMD) computations. The VR and VS register sets overlap
   in the same way as R and S register sets. See above.

   The storage space of vector registers often overlap with floating point
   registers. In this case setting the value of SLJIT_VR(i) destroys the
   value of SLJIT_FR(i) and vice versa. See SLJIT_SEPARATE_VECTOR_REGISTERS
   macro. */

/* Vector scratch registers. */
#define SLJIT_VR0	1
#define SLJIT_VR1	2
#define SLJIT_VR2	3
#define SLJIT_VR3	4
#define SLJIT_VR4	5
#define SLJIT_VR5	6
#define SLJIT_VR6	7
#define SLJIT_VR7	8
#define SLJIT_VR8	9
#define SLJIT_VR9	10
/* All VR registers provided by the architecture can be accessed by SLJIT_VR(i)
   The i parameter must be >= 0 and < SLJIT_NUMBER_OF_VECTOR_REGISTERS. */
#define SLJIT_VR(i)	(1 + (i))

/* Vector saved registers. */
#define SLJIT_VS0	(SLJIT_NUMBER_OF_VECTOR_REGISTERS)
#define SLJIT_VS1	(SLJIT_NUMBER_OF_VECTOR_REGISTERS - 1)
#define SLJIT_VS2	(SLJIT_NUMBER_OF_VECTOR_REGISTERS - 2)
#define SLJIT_VS3	(SLJIT_NUMBER_OF_VECTOR_REGISTERS - 3)
#define SLJIT_VS4	(SLJIT_NUMBER_OF_VECTOR_REGISTERS - 4)
#define SLJIT_VS5	(SLJIT_NUMBER_OF_VECTOR_REGISTERS - 5)
#define SLJIT_VS6	(SLJIT_NUMBER_OF_VECTOR_REGISTERS - 6)
#define SLJIT_VS7	(SLJIT_NUMBER_OF_VECTOR_REGISTERS - 7)
#define SLJIT_VS8	(SLJIT_NUMBER_OF_VECTOR_REGISTERS - 8)
#define SLJIT_VS9	(SLJIT_NUMBER_OF_VECTOR_REGISTERS - 9)
/* All VS registers provided by the architecture can be accessed by SLJIT_VS(i)
   The i parameter must be >= 0 and < SLJIT_NUMBER_OF_SAVED_VECTOR_REGISTERS. */
#define SLJIT_VS(i)	(SLJIT_NUMBER_OF_VECTOR_REGISTERS - (i))

/* Vector registers >= SLJIT_FIRST_SAVED_VECTOR_REG are saved registers. */
#define SLJIT_FIRST_SAVED_VECTOR_REG (SLJIT_VS0 - SLJIT_NUMBER_OF_SAVED_VECTOR_REGISTERS + 1)

/* --------------------------------------------------------------------- */
/*  Argument type definitions                                            */
/* --------------------------------------------------------------------- */

/* The following argument type definitions are used by sljit_emit_enter,
   sljit_set_context, sljit_emit_call, and sljit_emit_icall functions.

   For sljit_emit_call and sljit_emit_icall, the first integer argument
   must be placed into SLJIT_R0, the second one into SLJIT_R1, and so on.
   Similarly the first floating point argument must be placed into SLJIT_FR0,
   the second one into SLJIT_FR1, and so on.

   For sljit_emit_enter, the integer arguments can be stored in scratch
   or saved registers. Scratch registers are identified by a _R suffix.

   If only saved registers are used, then the allocation mirrors what is
   done for the "call" functions but using saved registers, meaning that
   the first integer argument goes to SLJIT_S0, the second one goes into
   SLJIT_S1, and so on.

   If scratch registers are used, then the way the integer registers are
   allocated changes so that SLJIT_S0, SLJIT_S1, etc; will be assigned
   only for the arguments not using scratch registers, while SLJIT_R<n>
   will be used for the ones using scratch registers.

   Furthermore, the index (shown as "n" above) that will be used for the
   scratch register depends on how many previous integer registers
   (scratch or saved) were used already, starting with SLJIT_R0.
   Eventhough some indexes will be likely skipped, they still need to be
   accounted for in the scratches parameter of sljit_emit_enter. See below
   for some examples.

   The floating point arguments always use scratch registers (but not the
   _R suffix like the integer arguments) and must use SLJIT_FR0, SLJIT_FR1,
   just like in the "call" functions.

   Note: the mapping for scratch registers is part of the compiler context
         and therefore a new context after sljit_emit_call/sljit_emit_icall
         could remove access to some scratch registers that were used as
         arguments.

   Example function definition:
     sljit_f32 SLJIT_FUNC example_c_callback(void *arg_a,
         sljit_f64 arg_b, sljit_u32 arg_c, sljit_f32 arg_d);

   Argument type definition:
     SLJIT_ARG_RETURN(SLJIT_ARG_TYPE_F32)
        | SLJIT_ARG_VALUE(SLJIT_ARG_TYPE_P, 1) | SLJIT_ARG_VALUE(SLJIT_ARG_TYPE_F64, 2)
        | SLJIT_ARG_VALUE(SLJIT_ARG_TYPE_32, 3) | SLJIT_ARG_VALUE(SLJIT_ARG_TYPE_F32, 4)

   Short form of argument type definition:
     SLJIT_ARGS4(F32, P, F64, 32, F32)

   Argument passing:
     arg_a must be placed in SLJIT_R0
     arg_b must be placed in SLJIT_FR0
     arg_c must be placed in SLJIT_R1
     arg_d must be placed in SLJIT_FR1

   Examples for argument processing by sljit_emit_enter:
     SLJIT_ARGS4V(P, 32_R, F32, W)
     Arguments are placed into: SLJIT_S0, SLJIT_R1, SLJIT_FR0, SLJIT_S1
     The type of the result is void.

     SLJIT_ARGS4(F32, W, W_R, W, W_R)
     Arguments are placed into: SLJIT_S0, SLJIT_R1, SLJIT_S1, SLJIT_R3
     The type of the result is sljit_f32.

     SLJIT_ARGS4(P, W, F32, P_R)
     Arguments are placed into: SLJIT_FR0, SLJIT_S0, SLJIT_FR1, SLJIT_R1
     The type of the result is pointer.

     Note: it is recommended to pass the scratch arguments first
     followed by the saved arguments:

       SLJIT_ARGS4(W, W_R, W_R, W, W)
       Arguments are placed into: SLJIT_R0, SLJIT_R1, SLJIT_S0, SLJIT_S1
       The type of the result is sljit_sw / sljit_uw.
*/

/* The following flag is only allowed for the integer arguments of
   sljit_emit_enter. When the flag is set, the integer argument is
   stored in a scratch register instead of a saved register. */
#define SLJIT_ARG_TYPE_SCRATCH_REG 0x8

/* No return value, only supported by SLJIT_ARG_RETURN. */
#define SLJIT_ARG_TYPE_RET_VOID		0
/* Machine word sized integer argument or result. */
#define SLJIT_ARG_TYPE_W		1
#define SLJIT_ARG_TYPE_W_R	(SLJIT_ARG_TYPE_W | SLJIT_ARG_TYPE_SCRATCH_REG)
/* 32 bit integer argument or result. */
#define SLJIT_ARG_TYPE_32		2
#define SLJIT_ARG_TYPE_32_R	(SLJIT_ARG_TYPE_32 | SLJIT_ARG_TYPE_SCRATCH_REG)
/* Pointer sized integer argument or result. */
#define SLJIT_ARG_TYPE_P		3
#define SLJIT_ARG_TYPE_P_R	(SLJIT_ARG_TYPE_P | SLJIT_ARG_TYPE_SCRATCH_REG)
/* 64 bit floating point argument or result. */
#define SLJIT_ARG_TYPE_F64		4
/* 32 bit floating point argument or result. */
#define SLJIT_ARG_TYPE_F32		5

#define SLJIT_ARG_SHIFT 4
#define SLJIT_ARG_RETURN(type) (type)
#define SLJIT_ARG_VALUE(type, idx) ((type) << ((idx) * SLJIT_ARG_SHIFT))

/* Simplified argument list definitions.

   The following definition:
       SLJIT_ARG_RETURN(SLJIT_ARG_TYPE_W) | SLJIT_ARG_VALUE(SLJIT_ARG_TYPE_F32, 1)

   can be shortened to:
       SLJIT_ARGS1(W, F32)

   Another example where no value is returned:
       SLJIT_ARG_RETURN(SLJIT_ARG_TYPE_RET_VOID) | SLJIT_ARG_VALUE(SLJIT_ARG_TYPE_W_R, 1)

   can be shortened to:
       SLJIT_ARGS1V(W_R)
*/

#define SLJIT_ARG_TO_TYPE(type) SLJIT_ARG_TYPE_ ## type

#define SLJIT_ARGS0(ret) \
	SLJIT_ARG_RETURN(SLJIT_ARG_TO_TYPE(ret))
#define SLJIT_ARGS0V() \
	SLJIT_ARG_RETURN(SLJIT_ARG_TYPE_RET_VOID)

#define SLJIT_ARGS1(ret, arg1) \
	(SLJIT_ARGS0(ret) | SLJIT_ARG_VALUE(SLJIT_ARG_TO_TYPE(arg1), 1))
#define SLJIT_ARGS1V(arg1) \
	(SLJIT_ARGS0V() | SLJIT_ARG_VALUE(SLJIT_ARG_TO_TYPE(arg1), 1))

#define SLJIT_ARGS2(ret, arg1, arg2) \
	(SLJIT_ARGS1(ret, arg1) | SLJIT_ARG_VALUE(SLJIT_ARG_TO_TYPE(arg2), 2))
#define SLJIT_ARGS2V(arg1, arg2) \
	(SLJIT_ARGS1V(arg1) | SLJIT_ARG_VALUE(SLJIT_ARG_TO_TYPE(arg2), 2))

#define SLJIT_ARGS3(ret, arg1, arg2, arg3) \
	(SLJIT_ARGS2(ret, arg1, arg2) | SLJIT_ARG_VALUE(SLJIT_ARG_TO_TYPE(arg3), 3))
#define SLJIT_ARGS3V(arg1, arg2, arg3) \
	(SLJIT_ARGS2V(arg1, arg2) | SLJIT_ARG_VALUE(SLJIT_ARG_TO_TYPE(arg3), 3))

#define SLJIT_ARGS4(ret, arg1, arg2, arg3, arg4) \
	(SLJIT_ARGS3(ret, arg1, arg2, arg3) | SLJIT_ARG_VALUE(SLJIT_ARG_TO_TYPE(arg4), 4))
#define SLJIT_ARGS4V(arg1, arg2, arg3, arg4) \
	(SLJIT_ARGS3V(arg1, arg2, arg3) | SLJIT_ARG_VALUE(SLJIT_ARG_TO_TYPE(arg4), 4))

/* --------------------------------------------------------------------- */
/*  Main structures and functions                                        */
/* --------------------------------------------------------------------- */

/*
	The following structures are private, and can be changed in the
	future. Keeping them here allows code inlining.
*/

struct sljit_memory_fragment {
	struct sljit_memory_fragment *next;
	sljit_uw used_size;
	/* Must be aligned to sljit_sw. */
	sljit_u8 memory[1];
};

struct sljit_label {
	struct sljit_label *next;
	union {
		sljit_uw index;
		sljit_uw addr;
	} u;
	/* The maximum size difference. */
	sljit_uw size;
};

struct sljit_jump {
	struct sljit_jump *next;
	sljit_uw addr;
	/* Architecture dependent flags. */
	sljit_uw flags;
	union {
		sljit_uw target;
		struct sljit_label *label;
	} u;
};

struct sljit_const {
	struct sljit_const *next;
	sljit_uw addr;
};

struct sljit_generate_code_buffer {
	void *buffer;
	sljit_uw size;
	sljit_sw executable_offset;
};

struct sljit_read_only_buffer {
	struct sljit_read_only_buffer *next;
	sljit_uw size;
	/* Label can be replaced by address after sljit_generate_code. */
	union {
		struct sljit_label *label;
		sljit_uw addr;
	} u;
};

struct sljit_compiler {
	sljit_s32 error;
	sljit_s32 options;

	struct sljit_label *labels;
	struct sljit_jump *jumps;
	struct sljit_const *consts;
	struct sljit_label *last_label;
	struct sljit_jump *last_jump;
	struct sljit_const *last_const;

	void *allocator_data;
	void *user_data;
	struct sljit_memory_fragment *buf;
	struct sljit_memory_fragment *abuf;

	/* Number of labels created by the compiler. */
	sljit_uw label_count;
	/* Available scratch registers. */
	sljit_s32 scratches;
	/* Available saved registers. */
	sljit_s32 saveds;
	/* Available float scratch registers. */
	sljit_s32 fscratches;
	/* Available float saved registers. */
	sljit_s32 fsaveds;
#if (defined SLJIT_SEPARATE_VECTOR_REGISTERS && SLJIT_SEPARATE_VECTOR_REGISTERS) \
		|| (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS) \
		|| (defined SLJIT_DEBUG && SLJIT_DEBUG) \
		|| (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	/* Available vector scratch registers. */
	sljit_s32 vscratches;
	/* Available vector saved registers. */
	sljit_s32 vsaveds;
#endif /* SLJIT_SEPARATE_VECTOR_REGISTERS || SLJIT_ARGUMENT_CHECKS || SLJIT_DEBUG || SLJIT_VERBOSE */
	/* Local stack size. */
	sljit_s32 local_size;
	/* Maximum code size. */
	sljit_uw size;
	/* Relative offset of the executable mapping from the writable mapping. */
	sljit_sw executable_offset;
	/* Executable size for statistical purposes. */
	sljit_uw executable_size;

#if (defined SLJIT_HAS_STATUS_FLAGS_STATE && SLJIT_HAS_STATUS_FLAGS_STATE)
	sljit_s32 status_flags_state;
#endif /* SLJIT_HAS_STATUS_FLAGS_STATE */

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	sljit_s32 args_size;
#endif /* SLJIT_CONFIG_X86_32 */

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	/* Temporary fields. */
	sljit_s32 mode32;
#endif /* SLJIT_CONFIG_X86_64 */

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	/* Constant pool handling. */
	sljit_uw *cpool;
	sljit_u8 *cpool_unique;
	sljit_uw cpool_diff;
	sljit_uw cpool_fill;
	/* Other members. */
	/* Contains pointer, "ldr pc, [...]" pairs. */
	sljit_uw patches;
#endif /* SLJIT_CONFIG_ARM_V6 */

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6) || (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7)
	/* Temporary fields. */
	sljit_uw shift_imm;
#endif /* SLJIT_CONFIG_ARM_V6 || SLJIT_CONFIG_ARM_V6 */

#if (defined SLJIT_CONFIG_ARM_32 && SLJIT_CONFIG_ARM_32) && (defined __SOFTFP__)
	sljit_uw args_size;
#endif /* SLJIT_CONFIG_ARM_32 && __SOFTFP__ */

#if (defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC)
	/* Temporary fields. */
	sljit_u32 imm;
#endif /* SLJIT_CONFIG_PPC */

#if (defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS)
	sljit_s32 delay_slot;
	/* Temporary fields. */
	sljit_s32 cache_arg;
	sljit_sw cache_argw;
#endif /* SLJIT_CONFIG_MIPS */

#if (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	sljit_uw args_size;
#endif /* SLJIT_CONFIG_MIPS_32 */

#if (defined SLJIT_CONFIG_RISCV && SLJIT_CONFIG_RISCV)
	/* Temporary fields. */
	sljit_s32 cache_arg;
	sljit_sw cache_argw;
#endif /* SLJIT_CONFIG_RISCV */

#if (defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X)
	/* Need to allocate register save area to make calls. */
	/* Temporary fields. */
	sljit_s32 mode;
#endif /* SLJIT_CONFIG_S390X */

#if (defined SLJIT_CONFIG_LOONGARCH && SLJIT_CONFIG_LOONGARCH)
	/* Temporary fields. */
	sljit_s32 cache_arg;
	sljit_sw cache_argw;
#endif /* SLJIT_CONFIG_LOONGARCH */

#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	FILE* verbose;
#endif /* SLJIT_VERBOSE */

	/* Note: SLJIT_DEBUG enables SLJIT_ARGUMENT_CHECKS. */
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS) \
		|| (defined SLJIT_DEBUG && SLJIT_DEBUG)
	/* Flags specified by the last arithmetic instruction.
	   It contains the type of the variable flag. */
	sljit_s32 last_flags;
	/* Return value type set by entry functions. */
	sljit_s32 last_return;
	/* Local size passed to entry functions. */
	sljit_s32 logical_local_size;
#endif /* SLJIT_ARGUMENT_CHECKS || SLJIT_DEBUG */

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS) \
		|| (defined SLJIT_DEBUG && SLJIT_DEBUG) \
		|| (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
#if !(defined SLJIT_SEPARATE_VECTOR_REGISTERS && SLJIT_SEPARATE_VECTOR_REGISTERS)
	/* Available float scratch registers. */
	sljit_s32 real_fscratches;
	/* Available float saved registers. */
	sljit_s32 real_fsaveds;
#endif /* !SLJIT_SEPARATE_VECTOR_REGISTERS */

	/* Trust arguments when an API function is called.
	   Used internally for calling API functions. */
	sljit_s32 skip_checks;
#endif /* SLJIT_ARGUMENT_CHECKS || SLJIT_DEBUG || SLJIT_VERBOSE */
};

/* --------------------------------------------------------------------- */
/*  Main functions                                                       */
/* --------------------------------------------------------------------- */

/* Creates an SLJIT compiler. The allocator_data is required by some
   custom memory managers. This pointer is passed to SLJIT_MALLOC
   and SLJIT_FREE macros. Most allocators (including the default
   one) ignores this value, and it is recommended to pass NULL
   as a dummy value for allocator_data.

   Returns NULL if failed. */
SLJIT_API_FUNC_ATTRIBUTE struct sljit_compiler* sljit_create_compiler(void *allocator_data);

/* Frees everything except the compiled machine code. */
SLJIT_API_FUNC_ATTRIBUTE void sljit_free_compiler(struct sljit_compiler *compiler);

/* Returns the current error code. If an error occurres, future calls
   which uses the same compiler argument returns early with the same
   error code. Thus there is no need for checking the error after every
   call, it is enough to do it after the code is compiled. Removing
   these checks increases the performance of the compiling process. */
static SLJIT_INLINE sljit_s32 sljit_get_compiler_error(struct sljit_compiler *compiler) { return compiler->error; }

/* Sets the compiler error code to SLJIT_ERR_ALLOC_FAILED except
   if an error was detected before. After the error code is set
   the compiler behaves as if the allocation failure happened
   during an SLJIT function call. This can greatly simplify error
   checking, since it is enough to check the compiler status
   after the code is compiled. */
SLJIT_API_FUNC_ATTRIBUTE void sljit_set_compiler_memory_error(struct sljit_compiler *compiler);

/* Allocate a small amount of memory. The size must be <= 64 bytes on 32 bit,
   and <= 128 bytes on 64 bit architectures. The memory area is owned by the
   compiler, and freed by sljit_free_compiler. The returned pointer is
   sizeof(sljit_sw) aligned. Excellent for allocating small blocks during
   compiling, and no need to worry about freeing them. The size is enough
   to contain at most 16 pointers. If the size is outside of the range,
   the function will return with NULL. However, this return value does not
   indicate that there is no more memory (does not set the current error code
   of the compiler to out-of-memory status). */
SLJIT_API_FUNC_ATTRIBUTE void* sljit_alloc_memory(struct sljit_compiler *compiler, sljit_s32 size);

/* Returns the allocator data passed to sljit_create_compiler. */
static SLJIT_INLINE void* sljit_compiler_get_allocator_data(struct sljit_compiler *compiler) { return compiler->allocator_data; }
/* Sets/get the user data for a compiler. */
static SLJIT_INLINE void sljit_compiler_set_user_data(struct sljit_compiler *compiler, void *user_data) { compiler->user_data = user_data; }
static SLJIT_INLINE void* sljit_compiler_get_user_data(struct sljit_compiler *compiler) { return compiler->user_data; }

#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
/* Passing NULL disables verbose. */
SLJIT_API_FUNC_ATTRIBUTE void sljit_compiler_verbose(struct sljit_compiler *compiler, FILE* verbose);
#endif /* SLJIT_VERBOSE */

/* Option bits for sljit_generate_code. */

/* The exec_allocator_data points to a pre-allocated
   buffer which type is sljit_generate_code_buffer. */
#define SLJIT_GENERATE_CODE_BUFFER		0x1

/* When SLJIT_INDIRECT_CALL is defined, no function context is
created for the generated code (see sljit_set_function_context),
so the returned pointer cannot be directly called from C code.
The flag is ignored when SLJIT_INDIRECT_CALL is not defined. */
#define SLJIT_GENERATE_CODE_NO_CONTEXT		0x2

/* Create executable code from the instruction stream. This is the final step
   of the code generation, and no more instructions can be emitted after this call.

   options is the combination of SLJIT_GENERATE_CODE_* bits
   exec_allocator_data is passed to SLJIT_MALLOC_EXEC and
                       SLJIT_MALLOC_FREE functions */

SLJIT_API_FUNC_ATTRIBUTE void* sljit_generate_code(struct sljit_compiler *compiler, sljit_s32 options, void *exec_allocator_data);

/* Free executable code. */

SLJIT_API_FUNC_ATTRIBUTE void sljit_free_code(void* code, void *exec_allocator_data);

/* When the protected executable allocator is used the JIT code is mapped
   twice. The first mapping has read/write and the second mapping has read/exec
   permissions. This function returns with the relative offset of the executable
   mapping using the writable mapping as the base after the machine code is
   successfully generated. The returned value is always 0 for the normal executable
   allocator, since it uses only one mapping with read/write/exec permissions.
   Dynamic code modifications requires this value.

   Before a successful code generation, this function returns with 0. */
static SLJIT_INLINE sljit_sw sljit_get_executable_offset(struct sljit_compiler *compiler) { return compiler->executable_offset; }

/* The executable memory consumption of the generated code can be retrieved by
   this function. The returned value can be used for statistical purposes.

   Before a successful code generation, this function returns with 0. */
static SLJIT_INLINE sljit_uw sljit_get_generated_code_size(struct sljit_compiler *compiler) { return compiler->executable_size; }

/* Returns with non-zero if the feature or limitation type passed as its
   argument is present on the current CPU. The return value is one, if a
   feature is fully supported, and it is two, if partially supported.

   Some features (e.g. floating point operations) require hardware (CPU)
   support while others (e.g. move with update) are emulated if not available.
   However, even when a feature is emulated, specialized code paths may be
   faster than the emulation. Some limitations are emulated as well so their
   general case is supported but it has extra performance costs.

   Note: sljitConfigInternal.h also provides several feature detection macros. */

/* [Not emulated] Floating-point support is available. */
#define SLJIT_HAS_FPU			0
/* [Limitation] Some registers are virtual registers. */
#define SLJIT_HAS_VIRTUAL_REGISTERS	1
/* [Emulated] Has zero register (setting a memory location to zero is efficient). */
#define SLJIT_HAS_ZERO_REGISTER		2
/* [Emulated] Count leading zero is supported. */
#define SLJIT_HAS_CLZ			3
/* [Emulated] Count trailing zero is supported. */
#define SLJIT_HAS_CTZ			4
/* [Emulated] Reverse the order of bytes is supported. */
#define SLJIT_HAS_REV			5
/* [Emulated] Rotate left/right is supported. */
#define SLJIT_HAS_ROT			6
/* [Emulated] Conditional move is supported. */
#define SLJIT_HAS_CMOV			7
/* [Emulated] Prefetch instruction is available (emulated as a nop). */
#define SLJIT_HAS_PREFETCH		8
/* [Emulated] Copy from/to f32 operation is available (see sljit_emit_fcopy). */
#define SLJIT_HAS_COPY_F32		9
/* [Emulated] Copy from/to f64 operation is available (see sljit_emit_fcopy). */
#define SLJIT_HAS_COPY_F64		10
/* [Not emulated] The 64 bit floating point registers can be used as
   two separate 32 bit floating point registers (e.g. ARM32). The
   second 32 bit part can be accessed by SLJIT_F64_SECOND. */
#define SLJIT_HAS_F64_AS_F32_PAIR	11
/* [Not emulated] Some SIMD operations are supported by the compiler. */
#define SLJIT_HAS_SIMD			12
/* [Not emulated] SIMD registers are mapped to a pair of double precision
   floating point registers. E.g. passing either SLJIT_FR0 or SLJIT_FR1 to
   a simd operation represents the same 128 bit register, and both SLJIT_FR0
   and SLJIT_FR1 are overwritten. */
#define SLJIT_SIMD_REGS_ARE_PAIRS	13
/* [Not emulated] Atomic support is available. */
#define SLJIT_HAS_ATOMIC		14
/* [Not emulated] Memory barrier support is available. */
#define SLJIT_HAS_MEMORY_BARRIER		15

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)
/* [Not emulated] AVX support is available on x86. */
#define SLJIT_HAS_AVX			100
/* [Not emulated] AVX2 support is available on x86. */
#define SLJIT_HAS_AVX2			101
#endif /* SLJIT_CONFIG_X86 */

#if (defined SLJIT_CONFIG_LOONGARCH)
/* [Not emulated] LASX support is available on LoongArch */
#define SLJIT_HAS_LASX        201
#endif /* SLJIT_CONFIG_LOONGARCH */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_has_cpu_feature(sljit_s32 feature_type);

/* If type is between SLJIT_ORDERED_EQUAL and SLJIT_ORDERED_LESS_EQUAL,
   sljit_cmp_info returns with:
     zero - if the cpu supports the floating point comparison type
     one - if the comparison requires two machine instructions
     two - if the comparison requires more than two machine instructions

   When the result is non-zero, it is recommended to avoid
   using the specified comparison type if it is easy to do so.

   Otherwise it returns zero. */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_cmp_info(sljit_s32 type);

/* The following functions generate machine code. If there is no
   error, they return with SLJIT_SUCCESS, otherwise they return
   with an error code. */

/*
   The executable code is a callable function from the viewpoint
   of the C language. Function calls must conform with the ABI
   (Application Binary Interface) of the target platform, which
   specify the purpose of machine registers and stack handling
   among other things. The sljit_emit_enter function emits the
   necessary instructions for setting up an entry point for the
   executable code. This is often called as function prologue.

   The "options" argument can be used to pass configuration options
   to the sljit compiler which affects the generated code, until
   another sljit_emit_enter or sljit_set_context is called. The
   available options are listed before sljit_emit_enter.

   The function argument list is specified by the SLJIT_ARGSx
   (SLJIT_ARGS0 .. SLJIT_ARGS4) macros. Currently maximum four
   arguments are supported. See the description of SLJIT_ARGSx
   macros about argument passing.

   The register set used by the function must be declared as well.
   The number of scratch and saved registers available to the
   function must be passed to sljit_emit_enter. Only R registers
   between R0 and "scratches" argument can be used later. E.g.
   if "scratches" is set to two, the scratch register set will
   be limited to SLJIT_R0 and SLJIT_R1. The S registers are
   declared in a similar manner, but their count is specified
   by "saveds" argument. The floating point scratch and saved
   registers can be set by using "scratches" and "saveds" argument
   as well, but their value must be passed to the SLJIT_ENTER_FLOAT
   macro, see below.

   The sljit_emit_enter is also capable of allocating a stack
   space for local data. The "local_size" argument contains the
   size in bytes of this local area, and it can be accessed using
   SLJIT_MEM1(SLJIT_SP). The memory area between SLJIT_SP (inclusive)
   and SLJIT_SP + local_size (exclusive) can be modified freely
   until the function returns. The alocated stack space is an
   uninitialized memory area.

   Floating point scratch and saved registers must be specified
   by the SLJIT_ENTER_FLOAT macro, which result value should be
   combined with scratches / saveds argument.

   Examples:
       To use three scratch and four floating point scratch
       registers, the "scratches" argument must be set to:
            3 | SLJIT_ENTER_FLOAT(4)

       To use six saved and five floating point saved
       registers, the "saveds" argument must be set to:
            6 | SLJIT_ENTER_FLOAT(5)

   Note: the following conditions must met:
         0 <= scratches <= SLJIT_NUMBER_OF_REGISTERS
         0 <= saveds <= SLJIT_NUMBER_OF_SAVED_REGISTERS
         scratches + saveds <= SLJIT_NUMBER_OF_REGISTERS

         0 <= float scratches <= SLJIT_NUMBER_OF_FLOAT_REGISTERS
         0 <= float saveds <= SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS
         float scratches + float saveds <= SLJIT_NUMBER_OF_FLOAT_REGISTERS

   Note: the compiler can use saved registers as scratch registers,
         but the opposite is not supported

   Note: every call of sljit_emit_enter and sljit_set_context
         overwrites the previous context.
*/

/* The following options are available for sljit_emit_enter. */

/* Saved registers between SLJIT_S0 and SLJIT_S(n - 1) (inclusive)
   are not saved / restored on function enter / return. Instead,
   these registers can be used to pass / return data (such as
   global / local context pointers) across function calls. The
   value of n must be between 1 and 3. This option is only
   supported by SLJIT_ENTER_REG_ARG calling convention. */
#define SLJIT_ENTER_KEEP(n)		(n)

/* The compiled function uses an SLJIT specific register argument
   calling convention. This is a lightweight function call type where
   both the caller and the called functions must be compiled by
   SLJIT. The type argument of the call must be SLJIT_CALL_REG_ARG
   and all arguments must be stored in scratch registers. */
#define SLJIT_ENTER_REG_ARG		0x00000004

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)
/* Use VEX prefix for all SIMD operations on x86. */
#define SLJIT_ENTER_USE_VEX		0x00010000
#endif /* !SLJIT_CONFIG_X86 */

/* Macros for other sljit_emit_enter arguments. */

/* Floating point scratch and saved registers can be
   specified by SLJIT_ENTER_FLOAT. */
#define SLJIT_ENTER_FLOAT(regs)		((regs) << 8)

/* Vector scratch and saved registers can be specified
   by SLJIT_ENTER_VECTOR. */
#define SLJIT_ENTER_VECTOR(regs)	((regs) << 16)

/* The local_size must be >= 0 and <= SLJIT_MAX_LOCAL_SIZE. */
#define SLJIT_MAX_LOCAL_SIZE		1048576

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types,
	sljit_s32 scratches, sljit_s32 saveds, sljit_s32 local_size);

/* The SLJIT compiler has a current context (which contains the local
   stack space size, number of used registers, etc.) which is initialized
   by sljit_emit_enter. Several functions (such as sljit_emit_return)
   requires this context to be able to generate the appropriate code.
   However, some code fragments (compiled separately) may have no
   normal entry point so their context is unknown to the compiler.

   sljit_set_context and sljit_emit_enter have the same arguments,
   but sljit_set_context does not generate any machine code.

   Note: every call of sljit_emit_enter and sljit_set_context overwrites
         the previous context. */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types,
	sljit_s32 scratches, sljit_s32 saveds, sljit_s32 local_size);

/* Return to the caller function. The sljit_emit_return_void function
   does not return with any value. The sljit_emit_return function returns
   with a single value loaded from its source operand. The load operation
   can be between SLJIT_MOV and SLJIT_MOV_P (see sljit_emit_op1) and
   SLJIT_MOV_F32/SLJIT_MOV_F64 (see sljit_emit_fop1) depending on the
   return value specified by sljit_emit_enter/sljit_set_context. */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_void(struct sljit_compiler *compiler);

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src, sljit_sw srcw);

/* Restores the saved registers and free the stack area, then the execution
   continues from the address specified by the source operand. This
   operation is similar to sljit_emit_return, but it ignores the return
   address. The code where the exection continues should use the same context
   as the caller function (see sljit_set_context). A word (pointer) value
   can be passed in the SLJIT_RETURN_REG register. This function can be used
   to jump to exception handlers. */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_to(struct sljit_compiler *compiler,
	sljit_s32 src, sljit_sw srcw);

/*
   Source and destination operands for arithmetical instructions
    imm              - a simple immediate value (cannot be used as a destination)
    reg              - any of the available registers (immediate argument must be 0)
    [imm]            - absolute memory address
    [reg+imm]        - indirect memory address
    [reg+(reg<<imm)] - indirect indexed memory address (shift must be between 0 and 3)
                       useful for accessing arrays (fully supported by both x86 and
                       ARM architectures, and cheap operation on others)
*/

/*
   IMPORTANT NOTE: memory accesses MUST be naturally aligned unless
                   SLJIT_UNALIGNED macro is defined and its value is 1.

     length | alignment
   ---------+-----------
     byte   | 1 byte (any physical_address is accepted)
     half   | 2 byte (physical_address & 0x1 == 0)
     int    | 4 byte (physical_address & 0x3 == 0)
     word   | 4 byte if SLJIT_32BIT_ARCHITECTURE is defined and its value is 1
            | 8 byte if SLJIT_64BIT_ARCHITECTURE is defined and its value is 1
    pointer | size of sljit_up type (4 byte on 32 bit machines, 4 or 8 byte
            | on 64 bit machines)

   Note:   Different architectures have different addressing limitations.
           A single instruction is enough for the following addressing
           modes. Other addressing modes are emulated by instruction
           sequences. This information could help to improve those code
           generators which focuses only a few architectures.

   x86:    [reg+imm], -2^32+1 <= imm <= 2^32-1 (full address space on x86-32)
           [reg+(reg<<imm)] is supported
           [imm], -2^32+1 <= imm <= 2^32-1 is supported
           Write-back is not supported
   arm:    [reg+imm], -4095 <= imm <= 4095 or -255 <= imm <= 255 for signed
                bytes, any halfs or floating point values)
           [reg+(reg<<imm)] is supported
           Write-back is supported
   arm-t2: [reg+imm], -255 <= imm <= 4095
           [reg+(reg<<imm)] is supported
           Write back is supported only for [reg+imm], where -255 <= imm <= 255
   arm64:  [reg+imm], -256 <= imm <= 255, 0 <= aligned imm <= 4095 * alignment
           [reg+(reg<<imm)] is supported
           Write back is supported only for [reg+imm], where -256 <= imm <= 255
   ppc:    [reg+imm], -65536 <= imm <= 65535. 64 bit loads/stores and 32 bit
                signed load on 64 bit requires immediates divisible by 4.
                [reg+imm] is not supported for signed 8 bit values.
           [reg+reg] is supported
           Write-back is supported except for one instruction: 32 bit signed
                load with [reg+imm] addressing mode on 64 bit.
   mips:   [reg+imm], -65536 <= imm <= 65535
           Write-back is not supported
   riscv:  [reg+imm], -2048 <= imm <= 2047
           Write-back is not supported
   s390x:  [reg+imm], -2^19 <= imm < 2^19
           [reg+reg] is supported
           Write-back is not supported
   loongarch:  [reg+imm], -2048 <= imm <= 2047
           [reg+reg] is supported
           Write-back is not supported
*/

/* Macros for specifying operand types. */
#define SLJIT_MEM		0x80
#define SLJIT_MEM0()		(SLJIT_MEM)
#define SLJIT_MEM1(r1)		(SLJIT_MEM | (r1))
#define SLJIT_MEM2(r1, r2)	(SLJIT_MEM | (r1) | ((r2) << 8))
#define SLJIT_IMM		0x7f
#define SLJIT_REG_PAIR(r1, r2)	((r1) | ((r2) << 8))

/* Macros for checking operand types (only for valid arguments). */
#define SLJIT_IS_REG(arg)	((arg) > 0 && (arg) < SLJIT_IMM)
#define SLJIT_IS_MEM(arg)	((arg) & SLJIT_MEM)
#define SLJIT_IS_MEM0(arg)	((arg) == SLJIT_MEM)
#define SLJIT_IS_MEM1(arg)	((arg) > SLJIT_MEM && (arg) < (SLJIT_MEM << 1))
#define SLJIT_IS_MEM2(arg)	(((arg) & SLJIT_MEM) && (arg) >= (SLJIT_MEM << 1))
#define SLJIT_IS_IMM(arg)	((arg) == SLJIT_IMM)
#define SLJIT_IS_REG_PAIR(arg)	(!((arg) & SLJIT_MEM) && (arg) >= (SLJIT_MEM << 1))

/* Macros for extracting registers from operands. */
/* Support operands which contains a single register or
   constructed using SLJIT_MEM1, SLJIT_MEM2, or SLJIT_REG_PAIR. */
#define SLJIT_EXTRACT_REG(arg)		((arg) & 0x7f)
/* Support operands which constructed using SLJIT_MEM2, or SLJIT_REG_PAIR. */
#define SLJIT_EXTRACT_SECOND_REG(arg)	((arg) >> 8)

/* Sets 32 bit operation mode on 64 bit CPUs. This option is ignored on
   32 bit CPUs. When this option is set for an arithmetic operation, only
   the lower 32 bits of the input registers are used, and the CPU status
   flags are set according to the 32 bit result. Although the higher 32 bit
   of the input and the result registers are not defined by SLJIT, it might
   be defined by the CPU architecture (e.g. MIPS). To satisfy these CPU
   requirements all source registers must be the result of those operations
   where this option was also set. Memory loads read 32 bit values rather
   than 64 bit ones. In other words 32 bit and 64 bit operations cannot be
   mixed. The only exception is SLJIT_MOV32 which source register can hold
   any 32 or 64 bit value, and it is converted to a 32 bit compatible format
   first. When the source and destination registers are the same, this
   conversion is free (no instructions are emitted) on most CPUs. A 32 bit
   value can also be converted to a 64 bit value by SLJIT_MOV_S32
   (sign extension) or SLJIT_MOV_U32 (zero extension).

   As for floating-point operations, this option sets 32 bit single
   precision mode. Similar to the integer operations, all register arguments
   must be the result of those operations where this option was also set.

   Note: memory addressing always uses 64 bit values on 64 bit systems so
         the result of a 32 bit operation must not be used with SLJIT_MEMx
         macros.

   This option is part of the instruction name, so there is no need to
   manually set it. E.g:

     SLJIT_ADD32 == (SLJIT_ADD | SLJIT_32) */
#define SLJIT_32		0x100

/* Many CPUs (x86, ARM, PPC) have status flag bits which can be set according
   to the result of an operation. Other CPUs (MIPS) do not have status
   flag bits, and results must be stored in registers. To cover both
   architecture types efficiently only two flags are defined by SLJIT:

    * Zero (equal) flag: it is set if the result is zero
    * Variable flag: its value is defined by the arithmetic operation

   SLJIT instructions can set any or both of these flags. The value of
   these flags is undefined if the instruction does not specify their
   value. The description of each instruction contains the list of
   allowed flag types.

   Note: the logical or operation can be used to set flags.

   Example: SLJIT_ADD can set the Z, OVERFLOW, CARRY flags hence

     sljit_op2(..., SLJIT_ADD, ...)
       Both the zero and variable flags are undefined so they can
       have any value after the operation is completed.

     sljit_op2(..., SLJIT_ADD | SLJIT_SET_Z, ...)
       Sets the zero flag if the result is zero, clears it otherwise.
       The variable flag is undefined.

     sljit_op2(..., SLJIT_ADD | SLJIT_SET_OVERFLOW, ...)
       Sets the variable flag if an integer overflow occurs, clears
       it otherwise. The zero flag is undefined.

     sljit_op2(..., SLJIT_ADD | SLJIT_SET_Z | SLJIT_SET_CARRY, ...)
       Sets the zero flag if the result is zero, clears it otherwise.
       Sets the variable flag if unsigned overflow (carry) occurs,
       clears it otherwise.

   Certain instructions (e.g. SLJIT_MOV) does not modify flags, so
   status flags are unchanged.

   Example:

     sljit_op2(..., SLJIT_ADD | SLJIT_SET_Z, ...)
     sljit_op1(..., SLJIT_MOV, ...)
       Zero flag is set according to the result of SLJIT_ADD.

     sljit_op2(..., SLJIT_ADD | SLJIT_SET_Z, ...)
     sljit_op2(..., SLJIT_ADD, ...)
       Zero flag has unknown value.

   These flags can be used for code optimization. E.g. a fast loop can be
   implemented by decreasing a counter register and set the zero flag
   using a single instruction. The zero register can be used by a
   conditional jump to restart the loop. A single comparison can set a
   zero and less flags to check if a value is less, equal, or greater
   than another value.

   Motivation: although some CPUs can set a large number of flag bits,
   usually their values are ignored or only a few of them are used. Emulating
   a large number of flags on systems without a flag register is complicated
   so SLJIT instructions must specify the flag they want to use and only
   that flag is computed. The last arithmetic instruction can be repeated if
   multiple flags need to be checked.
*/

/* Set Zero status flag. */
#define SLJIT_SET_Z			0x0200
/* Set the variable status flag if condition is true.
   See comparison types (e.g. SLJIT_SET_LESS, SLJIT_SET_F_EQUAL). */
#define SLJIT_SET(condition)			((condition) << 10)

/* Starting index of opcodes for sljit_emit_op0. */
#define SLJIT_OP0_BASE			0

/* Flags: - (does not modify flags)
   Note: breakpoint instruction is not supported by all architectures (e.g. ppc)
         It falls back to SLJIT_NOP in those cases. */
#define SLJIT_BREAKPOINT		(SLJIT_OP0_BASE + 0)
/* Flags: - (does not modify flags)
   Note: may or may not cause an extra cycle wait
         it can even decrease the runtime in a few cases. */
#define SLJIT_NOP			(SLJIT_OP0_BASE + 1)
/* Flags: - (may destroy flags)
   Unsigned multiplication of SLJIT_R0 and SLJIT_R1.
   Result is placed into SLJIT_R1:SLJIT_R0 (high:low) word */
#define SLJIT_LMUL_UW			(SLJIT_OP0_BASE + 2)
/* Flags: - (may destroy flags)
   Signed multiplication of SLJIT_R0 and SLJIT_R1.
   Result is placed into SLJIT_R1:SLJIT_R0 (high:low) word */
#define SLJIT_LMUL_SW			(SLJIT_OP0_BASE + 3)
/* Flags: - (may destroy flags)
   Unsigned divide of the value in SLJIT_R0 by the value in SLJIT_R1.
   The result is placed into SLJIT_R0 and the remainder into SLJIT_R1.
   Note: if SLJIT_R1 is 0, the behaviour is undefined. */
#define SLJIT_DIVMOD_UW			(SLJIT_OP0_BASE + 4)
#define SLJIT_DIVMOD_U32		(SLJIT_DIVMOD_UW | SLJIT_32)
/* Flags: - (may destroy flags)
   Signed divide of the value in SLJIT_R0 by the value in SLJIT_R1.
   The result is placed into SLJIT_R0 and the remainder into SLJIT_R1.
   Note: if SLJIT_R1 is 0, the behaviour is undefined.
   Note: if SLJIT_R1 is -1 and SLJIT_R0 is integer min (0x800..00),
         the behaviour is undefined. */
#define SLJIT_DIVMOD_SW			(SLJIT_OP0_BASE + 5)
#define SLJIT_DIVMOD_S32		(SLJIT_DIVMOD_SW | SLJIT_32)
/* Flags: - (may destroy flags)
   Unsigned divide of the value in SLJIT_R0 by the value in SLJIT_R1.
   The result is placed into SLJIT_R0. SLJIT_R1 preserves its value.
   Note: if SLJIT_R1 is 0, the behaviour is undefined. */
#define SLJIT_DIV_UW			(SLJIT_OP0_BASE + 6)
#define SLJIT_DIV_U32			(SLJIT_DIV_UW | SLJIT_32)
/* Flags: - (may destroy flags)
   Signed divide of the value in SLJIT_R0 by the value in SLJIT_R1.
   The result is placed into SLJIT_R0. SLJIT_R1 preserves its value.
   Note: if SLJIT_R1 is 0, the behaviour is undefined.
   Note: if SLJIT_R1 is -1 and SLJIT_R0 is integer min (0x800..00),
         the behaviour is undefined. */
#define SLJIT_DIV_SW			(SLJIT_OP0_BASE + 7)
#define SLJIT_DIV_S32			(SLJIT_DIV_SW | SLJIT_32)
/* Flags: - (does not modify flags)
   May return with SLJIT_ERR_UNSUPPORTED if SLJIT_HAS_MEMORY_BARRIER
   feature is not supported (calling sljit_has_cpu_feature() with
   this feature option returns with 0). */
#define SLJIT_MEMORY_BARRIER		(SLJIT_OP0_BASE + 8)
/* Flags: - (does not modify flags)
   ENDBR32 instruction for x86-32 and ENDBR64 instruction for x86-64
   when Intel Control-flow Enforcement Technology (CET) is enabled.
   No instructions are emitted for other architectures. */
#define SLJIT_ENDBR			(SLJIT_OP0_BASE + 9)
/* Flags: - (may destroy flags)
   Skip stack frames before return when Intel Control-flow
   Enforcement Technology (CET) is enabled. No instructions
   are emitted for other architectures. */
#define SLJIT_SKIP_FRAMES_BEFORE_RETURN	(SLJIT_OP0_BASE + 10)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op0(struct sljit_compiler *compiler, sljit_s32 op);

/* Starting index of opcodes for sljit_emit_op1. */
#define SLJIT_OP1_BASE			32

/* The MOV instruction transfers data from source to destination.

   MOV instruction suffixes:

   U8  - unsigned 8 bit data transfer
   S8  - signed 8 bit data transfer
   U16 - unsigned 16 bit data transfer
   S16 - signed 16 bit data transfer
   U32 - unsigned int (32 bit) data transfer
   S32 - signed int (32 bit) data transfer
   P   - pointer (sljit_up) data transfer
*/

/* Flags: - (does not modify flags) */
#define SLJIT_MOV			(SLJIT_OP1_BASE + 0)
/* Flags: - (does not modify flags) */
#define SLJIT_MOV_U8			(SLJIT_OP1_BASE + 1)
#define SLJIT_MOV32_U8			(SLJIT_MOV_U8 | SLJIT_32)
/* Flags: - (does not modify flags) */
#define SLJIT_MOV_S8			(SLJIT_OP1_BASE + 2)
#define SLJIT_MOV32_S8			(SLJIT_MOV_S8 | SLJIT_32)
/* Flags: - (does not modify flags) */
#define SLJIT_MOV_U16			(SLJIT_OP1_BASE + 3)
#define SLJIT_MOV32_U16			(SLJIT_MOV_U16 | SLJIT_32)
/* Flags: - (does not modify flags) */
#define SLJIT_MOV_S16			(SLJIT_OP1_BASE + 4)
#define SLJIT_MOV32_S16			(SLJIT_MOV_S16 | SLJIT_32)
/* Flags: - (does not modify flags)
   Note: no SLJIT_MOV32_U32 form, since it is the same as SLJIT_MOV32 */
#define SLJIT_MOV_U32			(SLJIT_OP1_BASE + 5)
/* Flags: - (does not modify flags)
   Note: no SLJIT_MOV32_S32 form, since it is the same as SLJIT_MOV32 */
#define SLJIT_MOV_S32			(SLJIT_OP1_BASE + 6)
/* Flags: - (does not modify flags) */
#define SLJIT_MOV32			(SLJIT_OP1_BASE + 7)
/* Flags: - (does not modify flags)
   Note: loads a pointer sized data, useful on x32 mode (a 64 bit mode
         on x86-64 which uses 32 bit pointers) or similar compiling modes */
#define SLJIT_MOV_P			(SLJIT_OP1_BASE + 8)
/* Count leading zeroes
   Flags: - (may destroy flags)
   Note: immediate source argument is not supported */
#define SLJIT_CLZ			(SLJIT_OP1_BASE + 9)
#define SLJIT_CLZ32			(SLJIT_CLZ | SLJIT_32)
/* Count trailing zeroes
   Flags: - (may destroy flags)
   Note: immediate source argument is not supported */
#define SLJIT_CTZ			(SLJIT_OP1_BASE + 10)
#define SLJIT_CTZ32			(SLJIT_CTZ | SLJIT_32)
/* Reverse the order of bytes
   Flags: - (may destroy flags)
   Note: converts between little and big endian formats
   Note: immediate source argument is not supported */
#define SLJIT_REV			(SLJIT_OP1_BASE + 11)
#define SLJIT_REV32			(SLJIT_REV | SLJIT_32)
/* Reverse the order of bytes in the lower 16 bit and extend as unsigned
   Flags: - (may destroy flags)
   Note: converts between little and big endian formats
   Note: immediate source argument is not supported */
#define SLJIT_REV_U16			(SLJIT_OP1_BASE + 12)
#define SLJIT_REV32_U16			(SLJIT_REV_U16 | SLJIT_32)
/* Reverse the order of bytes in the lower 16 bit and extend as signed
   Flags: - (may destroy flags)
   Note: converts between little and big endian formats
   Note: immediate source argument is not supported */
#define SLJIT_REV_S16			(SLJIT_OP1_BASE + 13)
#define SLJIT_REV32_S16			(SLJIT_REV_S16 | SLJIT_32)
/* Reverse the order of bytes in the lower 32 bit and extend as unsigned
   Flags: - (may destroy flags)
   Note: converts between little and big endian formats
   Note: immediate source argument is not supported */
#define SLJIT_REV_U32			(SLJIT_OP1_BASE + 14)
/* Reverse the order of bytes in the lower 32 bit and extend as signed
   Flags: - (may destroy flags)
   Note: converts between little and big endian formats
   Note: immediate source argument is not supported */
#define SLJIT_REV_S32			(SLJIT_OP1_BASE + 15)

/* The following unary operations are supported by using sljit_emit_op2:
     - binary not: SLJIT_XOR with immedate -1 as src1 or src2
     - negate: SLJIT_SUB with immedate 0 as src1
   Note: these operations are optimized by the compiler if the
     target CPU has specialized instruction forms for them. */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw);

/* Starting index of opcodes for sljit_emit_op2. */
#define SLJIT_OP2_BASE			64

/* Flags: Z | OVERFLOW | CARRY */
#define SLJIT_ADD			(SLJIT_OP2_BASE + 0)
#define SLJIT_ADD32			(SLJIT_ADD | SLJIT_32)
/* Flags: CARRY */
#define SLJIT_ADDC			(SLJIT_OP2_BASE + 1)
#define SLJIT_ADDC32			(SLJIT_ADDC | SLJIT_32)
/* Flags: Z | LESS | GREATER_EQUAL | GREATER | LESS_EQUAL
          SIG_LESS | SIG_GREATER_EQUAL | SIG_GREATER
          SIG_LESS_EQUAL | OVERFLOW | CARRY */
#define SLJIT_SUB			(SLJIT_OP2_BASE + 2)
#define SLJIT_SUB32			(SLJIT_SUB | SLJIT_32)
/* Flags: CARRY */
#define SLJIT_SUBC			(SLJIT_OP2_BASE + 3)
#define SLJIT_SUBC32			(SLJIT_SUBC | SLJIT_32)
/* Note: integer mul
   Flags: OVERFLOW */
#define SLJIT_MUL			(SLJIT_OP2_BASE + 4)
#define SLJIT_MUL32			(SLJIT_MUL | SLJIT_32)
/* Flags: Z */
#define SLJIT_AND			(SLJIT_OP2_BASE + 5)
#define SLJIT_AND32			(SLJIT_AND | SLJIT_32)
/* Flags: Z */
#define SLJIT_OR			(SLJIT_OP2_BASE + 6)
#define SLJIT_OR32			(SLJIT_OR | SLJIT_32)
/* Flags: Z */
#define SLJIT_XOR			(SLJIT_OP2_BASE + 7)
#define SLJIT_XOR32			(SLJIT_XOR | SLJIT_32)
/* Flags: Z
   Let bit_length be the length of the shift operation: 32 or 64.
   If src2 is immediate, src2w is masked by (bit_length - 1).
   Otherwise, if the content of src2 is outside the range from 0
   to bit_length - 1, the result is undefined. */
#define SLJIT_SHL			(SLJIT_OP2_BASE + 8)
#define SLJIT_SHL32			(SLJIT_SHL | SLJIT_32)
/* Flags: Z
   Same as SLJIT_SHL, except the the second operand is
   always masked by the length of the shift operation. */
#define SLJIT_MSHL			(SLJIT_OP2_BASE + 9)
#define SLJIT_MSHL32			(SLJIT_MSHL | SLJIT_32)
/* Flags: Z
   Let bit_length be the length of the shift operation: 32 or 64.
   If src2 is immediate, src2w is masked by (bit_length - 1).
   Otherwise, if the content of src2 is outside the range from 0
   to bit_length - 1, the result is undefined. */
#define SLJIT_LSHR			(SLJIT_OP2_BASE + 10)
#define SLJIT_LSHR32			(SLJIT_LSHR | SLJIT_32)
/* Flags: Z
   Same as SLJIT_LSHR, except the the second operand is
   always masked by the length of the shift operation. */
#define SLJIT_MLSHR			(SLJIT_OP2_BASE + 11)
#define SLJIT_MLSHR32			(SLJIT_MLSHR | SLJIT_32)
/* Flags: Z
   Let bit_length be the length of the shift operation: 32 or 64.
   If src2 is immediate, src2w is masked by (bit_length - 1).
   Otherwise, if the content of src2 is outside the range from 0
   to bit_length - 1, the result is undefined. */
#define SLJIT_ASHR			(SLJIT_OP2_BASE + 12)
#define SLJIT_ASHR32			(SLJIT_ASHR | SLJIT_32)
/* Flags: Z
   Same as SLJIT_ASHR, except the the second operand is
   always masked by the length of the shift operation. */
#define SLJIT_MASHR			(SLJIT_OP2_BASE + 13)
#define SLJIT_MASHR32			(SLJIT_MASHR | SLJIT_32)
/* Flags: - (may destroy flags)
   Let bit_length be the length of the rotate operation: 32 or 64.
   The second operand is always masked by (bit_length - 1). */
#define SLJIT_ROTL			(SLJIT_OP2_BASE + 14)
#define SLJIT_ROTL32			(SLJIT_ROTL | SLJIT_32)
/* Flags: - (may destroy flags)
   Let bit_length be the length of the rotate operation: 32 or 64.
   The second operand is always masked by (bit_length - 1). */
#define SLJIT_ROTR			(SLJIT_OP2_BASE + 15)
#define SLJIT_ROTR32			(SLJIT_ROTR | SLJIT_32)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w);

/* The sljit_emit_op2u function is the same as sljit_emit_op2
   except the result is discarded. */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2u(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w);

/* Starting index of opcodes for sljit_emit_op2r. */
#define SLJIT_OP2R_BASE			96

/* Flags: - (may destroy flags) */
#define SLJIT_MULADD			(SLJIT_OP2R_BASE + 0)
#define SLJIT_MULADD32			(SLJIT_MULADD | SLJIT_32)

/* Similar to sljit_emit_fop2, except the destination is always a register. */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2r(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w);

/* Emit a left or right shift operation, where the bits shifted
   in comes from a separate source operand. All operands are
   interpreted as unsigned integers.

   In the followings the value_mask variable is 31 for 32 bit
     operations and word_size - 1 otherwise.

   op must be one of the following operations:
     SLJIT_SHL or SLJIT_SHL32:
       dst_reg = src1_reg << src3_reg
       dst_reg |= ((src2_reg >> 1) >> (src3 ^ value_mask))
     SLJIT_MSHL or SLJIT_MSHL32:
       src3 &= value_mask
       perform the SLJIT_SHL or SLJIT_SHL32 operation
     SLJIT_LSHR or SLJIT_LSHR32:
       dst_reg = src1_reg >> src3_reg
       dst_reg |= ((src2_reg << 1) << (src3 ^ value_mask))
     SLJIT_MLSHR or SLJIT_MLSHR32:
       src3 &= value_mask
       perform the SLJIT_LSHR or SLJIT_LSHR32 operation

   op can be combined (or'ed) with SLJIT_SHIFT_INTO_NON_ZERO

   dst_reg specifies the destination register, where dst_reg
     and src2_reg cannot be the same registers
   src1_reg specifies the source register
   src2_reg specifies the register which is shifted into src1_reg
   src3 / src3w contains the shift amount

   Note: a rotate operation is performed if src1_reg and
         src2_reg are the same registers

   Flags: - (may destroy flags) */

/* The src3 operand contains a non-zero value. Improves
   the generated code on certain architectures, which
   provides a small performance improvement. */
#define SLJIT_SHIFT_INTO_NON_ZERO	0x200

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_shift_into(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 src1_reg,
	sljit_s32 src2_reg,
	sljit_s32 src3, sljit_sw src3w);

/* The following options are used by sljit_emit_op2_shift. */

/* The src2 argument is shifted left by an immedate value. */
#define SLJIT_SHL_IMM			(1 << 9)
/* When src2 argument is a register, its value is undefined after the operation. */
#define SLJIT_SRC2_UNDEFINED		(1 << 10)

/* Emits an addition operation, where the second argument is shifted by a value.

   op must be SLJIT_ADD | SLJIT_SHL_IMM, where the immedate value is stored in shift_arg

   Flags: - (may destroy flags) */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2_shift(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w,
	sljit_sw shift_arg);

/* Starting index of opcodes for sljit_emit_op_src
   and sljit_emit_op_dst. */
#define SLJIT_OP_SRC_DST_BASE		112

/* Fast return, see SLJIT_FAST_CALL for more details.
   Note: src cannot be an immedate value
   Flags: - (does not modify flags) */
#define SLJIT_FAST_RETURN		(SLJIT_OP_SRC_DST_BASE + 0)
/* Skip stack frames before fast return.
   Note: src cannot be an immedate value
   Flags: may destroy flags. */
#define SLJIT_SKIP_FRAMES_BEFORE_FAST_RETURN	(SLJIT_OP_SRC_DST_BASE + 1)
/* Prefetch value into the level 1 data cache
   Note: if the target CPU does not support data prefetch,
         no instructions are emitted.
   Note: this instruction never fails, even if the memory address is invalid.
   Flags: - (does not modify flags) */
#define SLJIT_PREFETCH_L1		(SLJIT_OP_SRC_DST_BASE + 2)
/* Prefetch value into the level 2 data cache
   Note: same as SLJIT_PREFETCH_L1 if the target CPU
         does not support this instruction form.
   Note: this instruction never fails, even if the memory address is invalid.
   Flags: - (does not modify flags) */
#define SLJIT_PREFETCH_L2		(SLJIT_OP_SRC_DST_BASE + 3)
/* Prefetch value into the level 3 data cache
   Note: same as SLJIT_PREFETCH_L2 if the target CPU
         does not support this instruction form.
   Note: this instruction never fails, even if the memory address is invalid.
   Flags: - (does not modify flags) */
#define SLJIT_PREFETCH_L3		(SLJIT_OP_SRC_DST_BASE + 4)
/* Prefetch a value which is only used once (and can be discarded afterwards)
   Note: same as SLJIT_PREFETCH_L1 if the target CPU
         does not support this instruction form.
   Note: this instruction never fails, even if the memory address is invalid.
   Flags: - (does not modify flags) */
#define SLJIT_PREFETCH_ONCE		(SLJIT_OP_SRC_DST_BASE + 5)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_src(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src, sljit_sw srcw);

/* Fast enter, see SLJIT_FAST_CALL for more details.
   Flags: - (does not modify flags) */
#define SLJIT_FAST_ENTER		(SLJIT_OP_SRC_DST_BASE + 6)

/* Copies the return address into dst. The return address is the
   address where the execution continues after the called function
   returns (see: sljit_emit_return / sljit_emit_return_void).
   Flags: - (does not modify flags) */
#define SLJIT_GET_RETURN_ADDRESS	(SLJIT_OP_SRC_DST_BASE + 7)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_dst(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw);

/* Starting index of opcodes for sljit_emit_fop1. */
#define SLJIT_FOP1_BASE			144

/* Flags: - (does not modify flags) */
#define SLJIT_MOV_F64			(SLJIT_FOP1_BASE + 0)
#define SLJIT_MOV_F32			(SLJIT_MOV_F64 | SLJIT_32)
/* Convert opcodes: CONV[DST_TYPE].FROM[SRC_TYPE]
   SRC/DST TYPE can be: F64, F32, S32, SW
   Rounding mode when the destination is SW or S32: round towards zero. */
/* Flags: - (may destroy flags) */
#define SLJIT_CONV_F64_FROM_F32		(SLJIT_FOP1_BASE + 1)
#define SLJIT_CONV_F32_FROM_F64		(SLJIT_CONV_F64_FROM_F32 | SLJIT_32)
/* Flags: - (may destroy flags) */
#define SLJIT_CONV_SW_FROM_F64		(SLJIT_FOP1_BASE + 2)
#define SLJIT_CONV_SW_FROM_F32		(SLJIT_CONV_SW_FROM_F64 | SLJIT_32)
/* Flags: - (may destroy flags) */
#define SLJIT_CONV_S32_FROM_F64		(SLJIT_FOP1_BASE + 3)
#define SLJIT_CONV_S32_FROM_F32		(SLJIT_CONV_S32_FROM_F64 | SLJIT_32)
/* Flags: - (may destroy flags) */
#define SLJIT_CONV_F64_FROM_SW		(SLJIT_FOP1_BASE + 4)
#define SLJIT_CONV_F32_FROM_SW		(SLJIT_CONV_F64_FROM_SW | SLJIT_32)
/* Flags: - (may destroy flags) */
#define SLJIT_CONV_F64_FROM_S32		(SLJIT_FOP1_BASE + 5)
#define SLJIT_CONV_F32_FROM_S32		(SLJIT_CONV_F64_FROM_S32 | SLJIT_32)
/* Flags: - (may destroy flags) */
#define SLJIT_CONV_F64_FROM_UW		(SLJIT_FOP1_BASE + 6)
#define SLJIT_CONV_F32_FROM_UW		(SLJIT_CONV_F64_FROM_UW | SLJIT_32)
/* Flags: - (may destroy flags) */
#define SLJIT_CONV_F64_FROM_U32		(SLJIT_FOP1_BASE + 7)
#define SLJIT_CONV_F32_FROM_U32		(SLJIT_CONV_F64_FROM_U32 | SLJIT_32)
/* Note: dst is the left and src is the right operand for SLJIT_CMP_F32/64.
   Flags: EQUAL_F | LESS_F | GREATER_EQUAL_F | GREATER_F | LESS_EQUAL_F */
#define SLJIT_CMP_F64			(SLJIT_FOP1_BASE + 8)
#define SLJIT_CMP_F32			(SLJIT_CMP_F64 | SLJIT_32)
/* Flags: - (may destroy flags) */
#define SLJIT_NEG_F64			(SLJIT_FOP1_BASE + 9)
#define SLJIT_NEG_F32			(SLJIT_NEG_F64 | SLJIT_32)
/* Flags: - (may destroy flags) */
#define SLJIT_ABS_F64			(SLJIT_FOP1_BASE + 10)
#define SLJIT_ABS_F32			(SLJIT_ABS_F64 | SLJIT_32)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw);

/* Starting index of opcodes for sljit_emit_fop2. */
#define SLJIT_FOP2_BASE			176

/* Flags: - (may destroy flags) */
#define SLJIT_ADD_F64			(SLJIT_FOP2_BASE + 0)
#define SLJIT_ADD_F32			(SLJIT_ADD_F64 | SLJIT_32)
/* Flags: - (may destroy flags) */
#define SLJIT_SUB_F64			(SLJIT_FOP2_BASE + 1)
#define SLJIT_SUB_F32			(SLJIT_SUB_F64 | SLJIT_32)
/* Flags: - (may destroy flags) */
#define SLJIT_MUL_F64			(SLJIT_FOP2_BASE + 2)
#define SLJIT_MUL_F32			(SLJIT_MUL_F64 | SLJIT_32)
/* Flags: - (may destroy flags) */
#define SLJIT_DIV_F64			(SLJIT_FOP2_BASE + 3)
#define SLJIT_DIV_F32			(SLJIT_DIV_F64 | SLJIT_32)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w);

/* Starting index of opcodes for sljit_emit_fop2r. */
#define SLJIT_FOP2R_BASE		192

/* Flags: - (may destroy flags) */
#define SLJIT_COPYSIGN_F64		(SLJIT_FOP2R_BASE + 0)
#define SLJIT_COPYSIGN_F32		(SLJIT_COPYSIGN_F64 | SLJIT_32)

/* Similar to sljit_emit_fop2, except the destination is always a register. */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop2r(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_freg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w);

/* Sets a floating point register to an immediate value. */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fset32(struct sljit_compiler *compiler,
	sljit_s32 freg, sljit_f32 value);
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fset64(struct sljit_compiler *compiler,
	sljit_s32 freg, sljit_f64 value);

/* The following opcodes are used by sljit_emit_fcopy(). */

/* 64 bit: copy a 64 bit value from an integer register into a
           64 bit floating point register without any modifications.
   32 bit: copy a 32 bit register or register pair into a 64 bit
           floating point register without any modifications. The
           register, or the first register of the register pair
           replaces the high order 32 bit of the floating point
           register. If a register pair is passed, the low
           order 32 bit is replaced by the second register.
           Otherwise, the low order 32 bit is unchanged. */
#define SLJIT_COPY_TO_F64		1
/* Copy a 32 bit value from an integer register into a 32 bit
   floating point register without any modifications. */
#define SLJIT_COPY32_TO_F32		(SLJIT_COPY_TO_F64 | SLJIT_32)
/* 64 bit: copy the value of a 64 bit floating point register into
           an integer register without any modifications.
   32 bit: copy a 64 bit floating point register into a 32 bit register
           or a 32 bit register pair without any modifications. The
           high order 32 bit of the floating point register is copied
           into the register, or the first register of the register
           pair. If a register pair is passed, the low order 32 bit
           is copied into the second register. */
#define SLJIT_COPY_FROM_F64		2
/* Copy the value of a 32 bit floating point register into an integer
   register without any modifications. The register should be processed
   with 32 bit operations later. */
#define SLJIT_COPY32_FROM_F32		(SLJIT_COPY_FROM_F64 | SLJIT_32)

/* Special data copy which involves floating point registers.

  op must be between SLJIT_COPY_TO_F64 and SLJIT_COPY32_FROM_F32
  freg must be a floating point register
  reg must be a register or register pair */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fcopy(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 freg, sljit_s32 reg);

/* Label and jump instructions. */

/* Emits a label which can be the target of jump / mov_addr instructions. */

SLJIT_API_FUNC_ATTRIBUTE struct sljit_label* sljit_emit_label(struct sljit_compiler *compiler);

/* Alignment values for sljit_emit_aligned_label. */

#define SLJIT_LABEL_ALIGN_1	0
#define SLJIT_LABEL_ALIGN_2	1
#define SLJIT_LABEL_ALIGN_4	2
#define SLJIT_LABEL_ALIGN_8	3
#define SLJIT_LABEL_ALIGN_16	4
#define SLJIT_LABEL_ALIGN_W	SLJIT_WORD_SHIFT
#define SLJIT_LABEL_ALIGN_P	SLJIT_POINTER_SHIFT

/* Emits a label which address is aligned to a power of 2 value. When some
   extra space needs to be added to align the label, that space is filled
   with SLJIT_NOP instructions. These labels usually represent the end of a
   compilation block, and a new function or some read-only data (e.g. a
   jump table) follows it. In these typical cases the SLJIT_NOPs are never
   executed.

   Optionally, buffers for storing read-only data or code can be allocated
   by this operation. The buffers are passed as a chain list, and a separate
   memory area is allocated for each item in the list. All buffers are aligned
   to SLJIT_NOP instruction size, and their starting address is returned as
   as a label. The sljit_get_label_abs_addr function or the SLJIT_MOV_ABS_ADDR
   operation can be used to get the real address. The label of the first buffer
   is always the same as the returned label. The buffers are initially
   initialized with SLJIT_NOP instructions. The alignment of the buffers can
   be controlled by their starting address and sizes. If the starting address
   is aligned to N, and size is also divisible by N, the next buffer is aligned
   to N. I.e. if a buffer is 16 byte aligned, and its size is divisible by 4,
   the next buffer is 4 byte aligned. Note: if a buffer is N (>=2) byte aligned,
   it is also N/2 byte aligned.

   align represents the alignment, and its value can
         be specified by SLJIT_LABEL_* constants

   buffers is a list of read-only buffers stored in a chain list.
           After calling sljit_generate_code, these buffers can be
           modified by sljit_read_only_buffer_start_writing() /
           sljit_read_only_buffer_end_writing() functions

   Note: the constant pool (if present) may be stored before the label. */
SLJIT_API_FUNC_ATTRIBUTE struct sljit_label* sljit_emit_aligned_label(struct sljit_compiler *compiler,
	sljit_s32 alignment, struct sljit_read_only_buffer *buffers);

/* The SLJIT_FAST_CALL is a calling method for creating lightweight function
   calls. This type of calls preserve the values of all registers and stack
   frame. Unlike normal function calls, the enter and return operations must
   be performed by the SLJIT_FAST_ENTER and SLJIT_FAST_RETURN operations
   respectively. The return address is stored in the dst argument of the
   SLJIT_FAST_ENTER operation, and this return address should be passed as
   the src argument for the SLJIT_FAST_RETURN operation to return from the
   called function.

   Fast calls are cheap operations (usually only a single call instruction is
   emitted) but they do not preserve any registers. However the callee function
   can freely use / update any registers and the locals area which can be
   efficiently exploited by various optimizations. Registers can be saved
   and restored manually if needed.

   Although returning to different address by SLJIT_FAST_RETURN is possible,
   this address usually cannot be predicted by the return address predictor of
   modern CPUs which may reduce performance. Furthermore certain security
   enhancement technologies such as Intel Control-flow Enforcement Technology
   (CET) may disallow returning to a different address (indirect jumps
   can be used instead, see SLJIT_SKIP_FRAMES_BEFORE_FAST_RETURN). */

/* Invert (negate) conditional type: xor (^) with 0x1 */

/* Integer comparison types. */
#define SLJIT_EQUAL			0
#define SLJIT_ZERO			SLJIT_EQUAL
#define SLJIT_NOT_EQUAL			1
#define SLJIT_NOT_ZERO			SLJIT_NOT_EQUAL

#define SLJIT_LESS			2
#define SLJIT_SET_LESS			SLJIT_SET(SLJIT_LESS)
#define SLJIT_GREATER_EQUAL		3
#define SLJIT_SET_GREATER_EQUAL		SLJIT_SET(SLJIT_LESS)
#define SLJIT_GREATER			4
#define SLJIT_SET_GREATER		SLJIT_SET(SLJIT_GREATER)
#define SLJIT_LESS_EQUAL		5
#define SLJIT_SET_LESS_EQUAL		SLJIT_SET(SLJIT_GREATER)
#define SLJIT_SIG_LESS			6
#define SLJIT_SET_SIG_LESS		SLJIT_SET(SLJIT_SIG_LESS)
#define SLJIT_SIG_GREATER_EQUAL		7
#define SLJIT_SET_SIG_GREATER_EQUAL	SLJIT_SET(SLJIT_SIG_LESS)
#define SLJIT_SIG_GREATER		8
#define SLJIT_SET_SIG_GREATER		SLJIT_SET(SLJIT_SIG_GREATER)
#define SLJIT_SIG_LESS_EQUAL		9
#define SLJIT_SET_SIG_LESS_EQUAL	SLJIT_SET(SLJIT_SIG_GREATER)

#define SLJIT_OVERFLOW			10
#define SLJIT_SET_OVERFLOW		SLJIT_SET(SLJIT_OVERFLOW)
#define SLJIT_NOT_OVERFLOW		11

/* Unlike other flags, sljit_emit_jump may destroy the carry flag. */
#define SLJIT_CARRY			12
#define SLJIT_SET_CARRY			SLJIT_SET(SLJIT_CARRY)
#define SLJIT_NOT_CARRY			13

#define SLJIT_ATOMIC_STORED		14
#define SLJIT_SET_ATOMIC_STORED		SLJIT_SET(SLJIT_ATOMIC_STORED)
#define SLJIT_ATOMIC_NOT_STORED		15

/* Basic floating point comparison types.

   Note: when the comparison result is unordered, their behaviour is unspecified. */

#define SLJIT_F_EQUAL				16
#define SLJIT_SET_F_EQUAL			SLJIT_SET(SLJIT_F_EQUAL)
#define SLJIT_F_NOT_EQUAL			17
#define SLJIT_SET_F_NOT_EQUAL			SLJIT_SET(SLJIT_F_EQUAL)
#define SLJIT_F_LESS				18
#define SLJIT_SET_F_LESS			SLJIT_SET(SLJIT_F_LESS)
#define SLJIT_F_GREATER_EQUAL			19
#define SLJIT_SET_F_GREATER_EQUAL		SLJIT_SET(SLJIT_F_LESS)
#define SLJIT_F_GREATER				20
#define SLJIT_SET_F_GREATER			SLJIT_SET(SLJIT_F_GREATER)
#define SLJIT_F_LESS_EQUAL			21
#define SLJIT_SET_F_LESS_EQUAL			SLJIT_SET(SLJIT_F_GREATER)

/* Jumps when either argument contains a NaN value. */
#define SLJIT_UNORDERED				22
#define SLJIT_SET_UNORDERED			SLJIT_SET(SLJIT_UNORDERED)
/* Jumps when neither argument contains a NaN value. */
#define SLJIT_ORDERED				23
#define SLJIT_SET_ORDERED			SLJIT_SET(SLJIT_UNORDERED)

/* Ordered / unordered floating point comparison types.

   Note: each comparison type has an ordered and unordered form. Some
         architectures supports only either of them (see: sljit_cmp_info). */

#define SLJIT_ORDERED_EQUAL			24
#define SLJIT_SET_ORDERED_EQUAL			SLJIT_SET(SLJIT_ORDERED_EQUAL)
#define SLJIT_UNORDERED_OR_NOT_EQUAL		25
#define SLJIT_SET_UNORDERED_OR_NOT_EQUAL	SLJIT_SET(SLJIT_ORDERED_EQUAL)
#define SLJIT_ORDERED_LESS			26
#define SLJIT_SET_ORDERED_LESS			SLJIT_SET(SLJIT_ORDERED_LESS)
#define SLJIT_UNORDERED_OR_GREATER_EQUAL	27
#define SLJIT_SET_UNORDERED_OR_GREATER_EQUAL	SLJIT_SET(SLJIT_ORDERED_LESS)
#define SLJIT_ORDERED_GREATER			28
#define SLJIT_SET_ORDERED_GREATER		SLJIT_SET(SLJIT_ORDERED_GREATER)
#define SLJIT_UNORDERED_OR_LESS_EQUAL		29
#define SLJIT_SET_UNORDERED_OR_LESS_EQUAL	SLJIT_SET(SLJIT_ORDERED_GREATER)

#define SLJIT_UNORDERED_OR_EQUAL		30
#define SLJIT_SET_UNORDERED_OR_EQUAL		SLJIT_SET(SLJIT_UNORDERED_OR_EQUAL)
#define SLJIT_ORDERED_NOT_EQUAL			31
#define SLJIT_SET_ORDERED_NOT_EQUAL		SLJIT_SET(SLJIT_UNORDERED_OR_EQUAL)
#define SLJIT_UNORDERED_OR_LESS			32
#define SLJIT_SET_UNORDERED_OR_LESS		SLJIT_SET(SLJIT_UNORDERED_OR_LESS)
#define SLJIT_ORDERED_GREATER_EQUAL		33
#define SLJIT_SET_ORDERED_GREATER_EQUAL		SLJIT_SET(SLJIT_UNORDERED_OR_LESS)
#define SLJIT_UNORDERED_OR_GREATER		34
#define SLJIT_SET_UNORDERED_OR_GREATER		SLJIT_SET(SLJIT_UNORDERED_OR_GREATER)
#define SLJIT_ORDERED_LESS_EQUAL		35
#define SLJIT_SET_ORDERED_LESS_EQUAL		SLJIT_SET(SLJIT_UNORDERED_OR_GREATER)

/* Unconditional jump types. */
#define SLJIT_JUMP			36
/* Fast calling method. See the description above. */
#define SLJIT_FAST_CALL			37
/* Default C calling convention. */
#define SLJIT_CALL			38
/* Called function must be compiled by SLJIT.
   See SLJIT_ENTER_REG_ARG option. */
#define SLJIT_CALL_REG_ARG		39

/* The target can be changed during runtime (see: sljit_set_jump_addr). */
#define SLJIT_REWRITABLE_JUMP		0x10000
/* When this flag is passed, the execution of the current function ends and
   the called function returns to the caller of the current function. The
   stack usage is reduced before the call, but it is not necessarily reduced
   to zero. In the latter case the compiler needs to allocate space for some
   arguments and the return address must be stored on the stack as well. */
#define SLJIT_CALL_RETURN		0x20000

/* Emit a jump instruction. The destination is not set, only the type of the jump.
    type must be between SLJIT_JUMP and SLJIT_FAST_CALL
    type can be combined (or'ed) with SLJIT_REWRITABLE_JUMP

   Flags: does not modify flags. */
SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_jump(struct sljit_compiler *compiler, sljit_s32 type);

/* Emit a C compiler (ABI) compatible function call.
    type must be SLJIT_CALL or SLJIT_CALL_REG_ARG
    type can be combined (or'ed) with SLJIT_REWRITABLE_JUMP and/or SLJIT_CALL_RETURN
    arg_types can be specified by SLJIT_ARGSx (SLJIT_ARG_RETURN / SLJIT_ARG_VALUE) macros

   Flags: destroy all flags. */
SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_call(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 arg_types);

/* Integer comparison operation. In most architectures it is implemented
   as a compare (sljit_emit_op2u with SLJIT_SUB) operation followed by
   an sljit_emit_jump. However, some architectures (e.g: ARM64 or RISCV)
   may optimize the generated code further. It is suggested to use this
   comparison form when appropriate.
    type must be between SLJIT_EQUAL and SLJIT_SIG_LESS_EQUAL
    type can be combined (or'ed) with SLJIT_32 or SLJIT_REWRITABLE_JUMP

   Flags: may destroy flags. */
SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_cmp(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w);

/* Floating point comparison operation. In most architectures it is
   implemented as a SLJIT_CMP_F32/64 operation (setting appropriate
   flags) followed by a sljit_emit_jump. However, some architectures
   (e.g: MIPS) may optimize the generated code further. It is suggested
   to use this comparison form when appropriate.
    type must be between SLJIT_F_EQUAL and SLJIT_ORDERED_LESS_EQUAL
    type can be combined (or'ed) with SLJIT_32 or SLJIT_REWRITABLE_JUMP

   Flags: destroy flags.
   Note: when any operand is NaN the behaviour depends on the comparison type. */
SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_fcmp(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w);

/* The following flags are used by sljit_emit_op2cmpz(). */
#define SLJIT_JUMP_IF_NON_ZERO		0
#define SLJIT_JUMP_IF_ZERO		SLJIT_SET_Z

/* Perform an integer arithmetic operation, then its result is compared to
   zero. In most architectures it is implemented as an sljit_emit_op2
   followed by an sljit_emit_jump. However, some architectures (e.g: RISCV)
   may optimize the generated code further. It is suggested to use this
   operation form when appropriate (e.g. for loops with counters).

   op must be an sljit_emit_op2 operation where zero flag can be set,
   op can be combined with SLJIT_SET_* status flag setters except
     SLJIT_SET_Z, SLJIT_REWRITABLE_JUMP or SLJIT_JUMP_IF_* option bits.

   Note: SLJIT_JUMP_IF_NON_ZERO is the default operation if neither
      SLJIT_JUMP_IF_ZERO or SLJIT_JUMP_IF_NON_ZERO is specified.
   Flags: sets the variable flag depending on op argument, the
      zero flag is undefined. */
SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_op2cmpz(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w);

/* Set the destination of the jump to this label. */
SLJIT_API_FUNC_ATTRIBUTE void sljit_set_label(struct sljit_jump *jump, struct sljit_label* label);
/* Set the destination address of the jump to this label. */
SLJIT_API_FUNC_ATTRIBUTE void sljit_set_target(struct sljit_jump *jump, sljit_uw target);

/* Emit an indirect jump or fast call.
   Direct form: set src to SLJIT_IMM() and srcw to the address
   Indirect form: any other valid addressing mode
    type must be between SLJIT_JUMP and SLJIT_FAST_CALL

   Flags: does not modify flags. */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_ijump(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 src, sljit_sw srcw);

/* Emit a C compiler (ABI) compatible function call.
   Direct form: set src to SLJIT_IMM() and srcw to the address
   Indirect form: any other valid addressing mode
    type must be SLJIT_CALL or SLJIT_CALL_REG_ARG
    type can be combined (or'ed) with SLJIT_CALL_RETURN
    arg_types can be specified by SLJIT_ARGSx (SLJIT_ARG_RETURN / SLJIT_ARG_VALUE) macros

   Flags: destroy all flags. */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_icall(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 arg_types, sljit_s32 src, sljit_sw srcw);

/* Perform an operation using the conditional flags as the second argument.
   Type must always be between SLJIT_EQUAL and SLJIT_ORDERED_LESS_EQUAL.
   The value represented by the type is 1, if the condition represented
   by type is fulfilled, and 0 otherwise.

   When op is SLJIT_MOV or SLJIT_MOV32:
     Set dst to the value represented by the type (0 or 1).
     Flags: - (does not modify flags)
   When op is SLJIT_AND, SLJIT_AND32, SLJIT_OR, SLJIT_OR32, SLJIT_XOR, or SLJIT_XOR32
     Performs the binary operation using dst as the first, and the value
     represented by type as the second argument. Result is written into dst.
     Flags: Z (may destroy flags) */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_flags(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 type);

/* The following flags are used by sljit_emit_select(). */

/* Compare src1 and src2_reg operands before executing select
   (i.e. converts the select operation to a min/max operation). */
#define SLJIT_COMPARE_SELECT	SLJIT_SET_Z

/* Emit a conditional select instruction which moves src1 to dst_reg,
   if the conditional flag is set, or src2_reg to dst_reg otherwise.
   The conditional flag should be set before executing the select
   instruction unless SLJIT_COMPARE_SELECT is specified.

   type must be between SLJIT_EQUAL and SLJIT_ORDERED_LESS_EQUAL
       when SLJIT_COMPARE_SELECT option is NOT specified
   type must be between SLJIT_LESS and SLJIT_SET_SIG_LESS_EQUAL
       when SLJIT_COMPARE_SELECT option is specified
   type can be combined (or'ed) with SLJIT_32 to move 32 bit
       register values instead of word sized ones
   type can be combined (or'ed) with SLJIT_COMPARE_SELECT
       which compares src1 and src2_reg before executing the select
   dst_reg and src2_reg must be valid registers
   src1 must be valid operand

   Note: if src1 is a memory operand, its value
         might be loaded even if the condition is false

   Note: when SLJIT_COMPARE_SELECT is specified, the status flag
         bits might not represent the result of a normal compare
         operation, hence flags are not specified after the operation

   Note: if sljit_has_cpu_feature(SLJIT_HAS_CMOV) returns with a non-zero value:
         (a) conditional register move (dst_reg==src2_reg, src1 is register)
             can be performed using a single instruction, except on RISCV,
             where three instructions are needed
         (b) conditional clearing (dst_reg==src2_reg, src1==SLJIT_IMM,
             src1w==0) can be performed using a single instruction,
             except on x86, where two instructions are needed

   Flags:
     When SLJIT_COMPARE_SELECT is NOT specified: - (does not modify flags)
     When SLJIT_COMPARE_SELECT is specified: - (may destroy flags) */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_select(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_reg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2_reg);

/* Emit a conditional floating point select instruction which moves
   src1 to dst_reg, if the conditional flag is set, or src2_reg to
   dst_reg otherwise. The conditional flag should be set before
   executing the select instruction.

   type must be between SLJIT_EQUAL and SLJIT_ORDERED_LESS_EQUAL
   type can be combined (or'ed) with SLJIT_32 to move 32 bit
       floating point values instead of 64 bit ones
   dst_freg and src2_freg must be valid floating point registers
   src1 must be valid operand

   Note: if src1 is a memory operand, its value
         might be loaded even if the condition is false.

   Flags: - (does not modify flags) */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fselect(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_freg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2_freg);

/* The following flags are used by sljit_emit_mem(), sljit_emit_mem_update(),
   sljit_emit_fmem(), and sljit_emit_fmem_update(). */

/* Memory load operation. This is the default. */
#define SLJIT_MEM_LOAD		0x000000
/* Memory store operation. */
#define SLJIT_MEM_STORE		0x000200

/* The following flags are used by sljit_emit_mem() and sljit_emit_fmem(). */

/* Load or stora data from an unaligned (byte aligned) address. */
#define SLJIT_MEM_UNALIGNED	0x000400
/* Load or stora data from a 16 bit aligned address. */
#define SLJIT_MEM_ALIGNED_16	0x000800
/* Load or stora data from a 32 bit aligned address. */
#define SLJIT_MEM_ALIGNED_32	0x001000

/* The following flags are used by sljit_emit_mem_update(),
   and sljit_emit_fmem_update(). */

/* Base register is updated before the memory access (default). */
#define SLJIT_MEM_PRE		0x000000
/* Base register is updated after the memory access. */
#define SLJIT_MEM_POST		0x000400

/* When SLJIT_MEM_SUPP is passed, no instructions are emitted.
   Instead the function returns with SLJIT_SUCCESS if the instruction
   form is supported and SLJIT_ERR_UNSUPPORTED otherwise. This flag
   allows runtime checking of available instruction forms. */
#define SLJIT_MEM_SUPP		0x000800

/* The sljit_emit_mem emits instructions for various memory operations:

   When SLJIT_MEM_UNALIGNED / SLJIT_MEM_ALIGNED_16 /
        SLJIT_MEM_ALIGNED_32 is set in type argument:
     Emit instructions for unaligned memory loads or stores. When
     SLJIT_UNALIGNED is not defined, the only way to access unaligned
     memory data is using sljit_emit_mem. Otherwise all operations (e.g.
     sljit_emit_op1/2, or sljit_emit_fop1/2) supports unaligned access.
     In general, the performance of unaligned memory accesses are often
     lower than aligned and should be avoided.

   When a pair of registers is passed in reg argument:
     Emit instructions for moving data between a register pair and
     memory. The register pair can be specified by the SLJIT_REG_PAIR
     macro. The first register is loaded from or stored into the
     location specified by the mem/memw arguments, and the end address
     of this operation is the starting address of the data transfer
     between the second register and memory. The type argument must
     be SLJIT_MOV. The SLJIT_MEM_UNALIGNED / SLJIT_MEM_ALIGNED_*
     options are allowed for this operation.

   type must be between SLJIT_MOV and SLJIT_MOV_P and can be
     combined (or'ed) with SLJIT_MEM_* flags
   reg is a register or register pair, which is the source or
     destination of the operation
   mem must be a memory operand

   Flags: - (does not modify flags) */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_mem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw);

/* Emit a single memory load or store with update instruction.
   When the requested instruction form is not supported by the CPU,
   it returns with SLJIT_ERR_UNSUPPORTED instead of emulating the
   instruction. This allows specializing tight loops based on
   the supported instruction forms (see SLJIT_MEM_SUPP flag).
   Absolute address (SLJIT_MEM0) forms are never supported
   and the base (first) register specified by the mem argument
   must not be SLJIT_SP and must also be different from the
   register specified by the reg argument.

   type must be between SLJIT_MOV and SLJIT_MOV_P and can be
     combined (or'ed) with SLJIT_MEM_* flags
   reg is the source or destination register of the operation
   mem must be a memory operand

   Flags: - (does not modify flags) */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_mem_update(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw);

/* Same as sljit_emit_mem except the followings:

   Loading or storing a pair of registers is not supported.

   type must be SLJIT_MOV_F64 or SLJIT_MOV_F32 and can be
     combined (or'ed) with SLJIT_MEM_* flags.
   freg is the source or destination floating point register
     of the operation
   mem must be a memory operand

   Flags: - (does not modify flags) */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fmem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 mem, sljit_sw memw);

/* Same as sljit_emit_mem_update except the followings:

   type must be SLJIT_MOV_F64 or SLJIT_MOV_F32 and can be
     combined (or'ed) with SLJIT_MEM_* flags
   freg is the source or destination floating point register
     of the operation
   mem must be a memory operand

   Flags: - (does not modify flags) */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fmem_update(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 mem, sljit_sw memw);

/* The following options are used by several simd operations. */

/* Load data into a vector register, this is the default */
#define SLJIT_SIMD_LOAD			0x000000
/* Store data from a vector register */
#define SLJIT_SIMD_STORE		0x000001
/* The vector register contains floating point values */
#define SLJIT_SIMD_FLOAT		0x000400
/* Tests whether the operation is available */
#define SLJIT_SIMD_TEST			0x000800
/* Move data to/from a 64 bit (8 byte) long vector register */
#define SLJIT_SIMD_REG_64		(3 << 12)
/* Move data to/from a 128 bit (16 byte) long vector register */
#define SLJIT_SIMD_REG_128		(4 << 12)
/* Move data to/from a 256 bit (32 byte) long vector register */
#define SLJIT_SIMD_REG_256		(5 << 12)
/* Move data to/from a 512 bit (64 byte) long vector register */
#define SLJIT_SIMD_REG_512		(6 << 12)
/* Element size is 8 bit long (this is the default), usually cannot be combined with SLJIT_SIMD_FLOAT */
#define SLJIT_SIMD_ELEM_8		(0 << 18)
/* Element size is 16 bit long, usually cannot be combined with SLJIT_SIMD_FLOAT */
#define SLJIT_SIMD_ELEM_16		(1 << 18)
/* Element size is 32 bit long */
#define SLJIT_SIMD_ELEM_32		(2 << 18)
/* Element size is 64 bit long */
#define SLJIT_SIMD_ELEM_64		(3 << 18)
/* Element size is 128 bit long */
#define SLJIT_SIMD_ELEM_128		(4 << 18)
/* Element size is 256 bit long */
#define SLJIT_SIMD_ELEM_256		(5 << 18)

/* The following options are used by sljit_emit_simd_mov()
   and sljit_emit_simd_op2(). */

/* Memory address is unaligned (this is the default) */
#define SLJIT_SIMD_MEM_UNALIGNED	(0 << 24)
/* Memory address is 16 bit aligned */
#define SLJIT_SIMD_MEM_ALIGNED_16	(1 << 24)
/* Memory address is 32 bit aligned */
#define SLJIT_SIMD_MEM_ALIGNED_32	(2 << 24)
/* Memory address is 64 bit aligned */
#define SLJIT_SIMD_MEM_ALIGNED_64	(3 << 24)
/* Memory address is 128 bit aligned */
#define SLJIT_SIMD_MEM_ALIGNED_128	(4 << 24)
/* Memory address is 256 bit aligned */
#define SLJIT_SIMD_MEM_ALIGNED_256	(5 << 24)
/* Memory address is 512 bit aligned */
#define SLJIT_SIMD_MEM_ALIGNED_512	(6 << 24)

/* Moves data between a vector register and memory.

   If the operation is not supported, it returns with
   SLJIT_ERR_UNSUPPORTED. If SLJIT_SIMD_TEST is passed,
   it does not emit any instructions.

   type must be a combination of SLJIT_SIMD_* and
     SLJIT_SIMD_MEM_* options
   vreg is the source or destination vector register
     of the operation
   srcdst must be a memory operand or a vector register

   Note:
       The alignment and element size must be
       less or equal than vector register size.

   Flags: - (does not modify flags) */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_mov(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg,
	sljit_s32 srcdst, sljit_sw srcdstw);

/* Replicates a scalar value to all lanes of a vector
   register.

   If the operation is not supported, it returns with
   SLJIT_ERR_UNSUPPORTED. If SLJIT_SIMD_TEST is passed,
   it does not emit any instructions.

   type must be a combination of SLJIT_SIMD_* options
     except SLJIT_SIMD_STORE.
   vreg is the destination vector register of the operation
   src is the value which is replicated

   Note:
       The src == SLJIT_IMM and srcw == 0 can be used to
       clear a register even when SLJIT_SIMD_FLOAT is set.

   Flags: - (does not modify flags) */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_replicate(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg,
	sljit_s32 src, sljit_sw srcw);

/* The following options are used by sljit_emit_simd_lane_mov(). */

/* Clear all bits of the simd register before loading the lane. */
#define SLJIT_SIMD_LANE_ZERO		0x000002
/* Sign extend the integer value stored from the lane. */
#define SLJIT_SIMD_LANE_SIGNED		0x000004

/* Moves data between a vector register lane and a register or
   memory. If the srcdst argument is a register, it must be
   a floating point register when SLJIT_SIMD_FLOAT is specified,
   or a general purpose register otherwise.

   If the operation is not supported, it returns with
   SLJIT_ERR_UNSUPPORTED. If SLJIT_SIMD_TEST is passed,
   it does not emit any instructions.

   type must be a combination of SLJIT_SIMD_* options
     Further options:
       SLJIT_32 - when SLJIT_SIMD_FLOAT is not set
       SLJIT_SIMD_LANE_SIGNED - when SLJIT_SIMD_STORE
           is set and SLJIT_SIMD_FLOAT is not set
       SLJIT_SIMD_LANE_ZERO - when SLJIT_SIMD_LOAD
           is specified
   vreg is the source or destination vector register
     of the operation
   lane_index is the index of the lane
   srcdst is the destination operand for loads, and
     source operand for stores

   Note:
       The elem size must be lower than register size.

   Flags: - (does not modify flags) */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_lane_mov(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg, sljit_s32 lane_index,
	sljit_s32 srcdst, sljit_sw srcdstw);

/* Replicates a scalar value from a lane to all lanes
   of a vector register.

   If the operation is not supported, it returns with
   SLJIT_ERR_UNSUPPORTED. If SLJIT_SIMD_TEST is passed,
   it does not emit any instructions.

   type must be a combination of SLJIT_SIMD_* options
     except SLJIT_SIMD_STORE.
   vreg is the destination vector register of the operation
   src is the vector register which lane is replicated
   src_lane_index is the lane index of the src register

   Flags: - (does not modify flags) */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_lane_replicate(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg,
	sljit_s32 src, sljit_s32 src_lane_index);

/* The following options are used by sljit_emit_simd_load_extend(). */

/* Sign extend the integer elements */
#define SLJIT_SIMD_EXTEND_SIGNED	0x000002
/* Extend data to 16 bit */
#define SLJIT_SIMD_EXTEND_16		(1 << 24)
/* Extend data to 32 bit */
#define SLJIT_SIMD_EXTEND_32		(2 << 24)
/* Extend data to 64 bit */
#define SLJIT_SIMD_EXTEND_64		(3 << 24)

/* Extend elements and stores them in a vector register.
   The extension operation increases the size of the
   elements (e.g. from 16 bit to 64 bit). For integer
   values, the extension can be signed or unsigned.

   If the operation is not supported, it returns with
   SLJIT_ERR_UNSUPPORTED. If SLJIT_SIMD_TEST is passed,
   it does not emit any instructions.

   type must be a combination of SLJIT_SIMD_*, and
     SLJIT_SIMD_EXTEND_* options except SLJIT_SIMD_STORE
   vreg is the destination vector register of the operation
   src must be a memory operand or a vector register.
     In the latter case, the source elements are stored
     in the lower half of the register.

   Flags: - (does not modify flags) */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_extend(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg,
	sljit_s32 src, sljit_sw srcw);

/* Extract the highest bit (usually the sign bit) from
   each elements of a vector.

   If the operation is not supported, it returns with
   SLJIT_ERR_UNSUPPORTED. If SLJIT_SIMD_TEST is passed,
   it does not emit any instructions.

   type must be a combination of SLJIT_SIMD_* and SLJIT_32
     options except SLJIT_SIMD_LOAD
   vreg is the source vector register of the operation
   dst is the destination operand

   Flags: - (does not modify flags) */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_sign(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg,
	sljit_s32 dst, sljit_sw dstw);

/* The following operations are used by sljit_emit_simd_op2(). */

/* Binary 'and' operation */
#define SLJIT_SIMD_OP2_AND		0x000001
/* Binary 'or' operation */
#define SLJIT_SIMD_OP2_OR		0x000002
/* Binary 'xor' operation */
#define SLJIT_SIMD_OP2_XOR		0x000003
/* Shuffle bytes of src1 using the indicies in src2 */
#define SLJIT_SIMD_OP2_SHUFFLE		0x000004

/* Perform simd operations using vector registers.

   If the operation is not supported, it returns with
   SLJIT_ERR_UNSUPPORTED. If SLJIT_SIMD_TEST is passed,
   it does not emit any instructions.

   type must be a combination of SLJIT_SIMD_*, SLJIT_SIMD_MEM_*
     and SLJIT_SIMD_OP2_* options except SLJIT_SIMD_LOAD
     and SLJIT_SIMD_STORE
   dst_vreg is the destination register of the operation
   src1_vreg is the first source register of the operation
   src2 is the second source operand of the operation

   Flags: - (does not modify flags) */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_op2(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_vreg, sljit_s32 src1_vreg, sljit_s32 src2, sljit_sw src2w);

/* The following operations are used by sljit_emit_atomic_load() and
   sljit_emit_atomic_store() operations. */

/* Tests whether the atomic operation is available (does not generate
   any instructions). When a load from is allowed, its corresponding
   store form is allowed and vice versa. */
#define SLJIT_ATOMIC_TEST 0x10000
/* The compiler must generate compare and swap instruction.
   When this bit is set, calling sljit_emit_atomic_load() is optional. */
#define SLJIT_ATOMIC_USE_CAS 0x20000
/* The compiler must generate load-acquire and store-release instructions.
   When this bit is set, the temp_reg for sljit_emit_atomic_store is not used. */
#define SLJIT_ATOMIC_USE_LS 0x40000

/* The sljit_emit_atomic_load and sljit_emit_atomic_store operation pair
   can perform an atomic read-modify-write operation. First, an unsigned
   value must be loaded from memory using sljit_emit_atomic_load. Then,
   the updated value must be written back to the same memory location by
   sljit_emit_atomic_store. A thread can only perform a single atomic
   operation at a time.

   The following conditions must be satisfied, or the operation
   is undefined:
     - the address provided in mem_reg must be divisible by the size of
       the value (only naturally aligned updates are supported)
     - no memory operations are allowed between the load and store operations
     - the memory operation (op) and the base address (stored in mem_reg)
       passed to the load/store operations must be the same (the mem_reg
       can be a different register, only its value must be the same)
     - a store must always follow a load for the same transaction.

   op must be between SLJIT_MOV and SLJIT_MOV_P
   dst_reg is the register where the data will be loaded into
   mem_reg is the base address of the memory load (it cannot be
     SLJIT_SP or a virtual register on x86-32)

   Flags: - (does not modify flags) */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_atomic_load(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 mem_reg);

/* The sljit_emit_atomic_load and sljit_emit_atomic_store operations
   allows performing an atomic read-modify-write operation. See the
   description of sljit_emit_atomic_load.

   op must be between SLJIT_MOV and SLJIT_MOV_P
   src_reg is the register which value is stored into the memory
   mem_reg is the base address of the memory store (it cannot be
     SLJIT_SP or a virtual register on x86-32)
   temp_reg is a scratch register, which must be initialized with
     the value loaded into the dst_reg during the corresponding
     sljit_emit_atomic_load operation, or the operation is undefined.
     The temp_reg register preserves its value, if the memory store
     is successful. Otherwise, its value is undefined.

   Flags: ATOMIC_STORED
     if ATOMIC_STORED flag is set, it represents that the memory
     is updated with a new value. Otherwise the memory is unchanged. */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_atomic_store(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src_reg,
	sljit_s32 mem_reg,
	sljit_s32 temp_reg);

/* Copies the base address of SLJIT_SP + offset to dst. The offset can
   represent the starting address of a value in the local data (stack).
   The offset is not limited by the local data limits, it can be any value.
   For example if an array of bytes are stored on the stack from
   offset 0x40, and R0 contains the offset of an array item plus 0x120,
   this item can be changed by two SLJIT instructions:

   sljit_get_local_base(compiler, SLJIT_R1, 0, 0x40 - 0x120);
   sljit_emit_op1(compiler, SLJIT_MOV_U8, SLJIT_MEM2(SLJIT_R1, SLJIT_R0), 0, SLJIT_IMM, 0x5);

   Flags: - (may destroy flags) */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_local_base(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw offset);

/* Store a value that can be changed at runtime. The constant
   can be managed by sljit_get_const_addr and sljit_set_const.

   op must be SLJIT_MOV, SLJIT_MOV32, SLJIT_MOV_S32,
     SLJIT_MOV_U8, SLJIT_MOV32_U8

   Note: when SLJIT_MOV_U8 is used, and dst is a register,
         init_value supports a 9 bit signed value between [-256..255]

   Flags: - (does not modify flags) */
SLJIT_API_FUNC_ATTRIBUTE struct sljit_const* sljit_emit_const(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_sw init_value);

/* Opcodes for sljit_emit_mov_addr. */

/* The address is suitable for jump/call target. */
#define SLJIT_MOV_ADDR 0
/* The address is suitable for reading memory. */
#define SLJIT_MOV_ABS_ADDR 1
/* Add absolute address. */
#define SLJIT_ADD_ABS_ADDR 2

/* Store the value of a label (see: sljit_set_label / sljit_set_target)
   Flags: - (does not modify flags) */
SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_op_addr(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw);

/* Returns the address of a label after sljit_generate_code is called, and
   before the compiler is freed by sljit_free_compiler. It is recommended
   to save these addresses elsewhere before sljit_free_compiler is called.

   The address returned by sljit_get_label_addr is suitable for a jump/call
   target, and the address returned by sljit_get_label_abs_addr is suitable
   for reading memory. */

static SLJIT_INLINE sljit_uw sljit_get_label_addr(struct sljit_label *label) { return label->u.addr; }
#if (defined SLJIT_CONFIG_ARM_THUMB2 && SLJIT_CONFIG_ARM_THUMB2)
static SLJIT_INLINE sljit_uw sljit_get_label_abs_addr(struct sljit_label *label) { return label->u.addr & ~(sljit_uw)1; }
#else /* !SLJIT_CONFIG_ARM_THUMB2 */
static SLJIT_INLINE sljit_uw sljit_get_label_abs_addr(struct sljit_label *label) { return label->u.addr; }
#endif /* SLJIT_CONFIG_ARM_THUMB2 */

/* Returns the address of jump and const instructions after sljit_generate_code
   is called, and before the compiler is freed by sljit_free_compiler. It is
   recommended to save these addresses elsewhere before sljit_free_compiler is called. */

static SLJIT_INLINE sljit_uw sljit_get_jump_addr(struct sljit_jump *jump) { return jump->addr; }
static SLJIT_INLINE sljit_uw sljit_get_const_addr(struct sljit_const *const_) { return const_->addr; }

/* Only the address and executable offset are required to perform dynamic
   code modifications. See sljit_get_executable_offset function. */
SLJIT_API_FUNC_ATTRIBUTE void sljit_set_jump_addr(sljit_uw addr, sljit_uw new_target, sljit_sw executable_offset);
/* The op opcode must be set to the same value that was passed to sljit_emit_const. */
SLJIT_API_FUNC_ATTRIBUTE void sljit_set_const(sljit_uw addr, sljit_s32 op, sljit_sw new_constant, sljit_sw executable_offset);

/* Only a single buffer is writable at a time, so sljit_read_only_buffer_end_writing()
   must be called before sljit_read_only_buffer_start_writing() is called again. */
SLJIT_API_FUNC_ATTRIBUTE void* sljit_read_only_buffer_start_writing(sljit_uw addr, sljit_uw size, sljit_sw executable_offset);
SLJIT_API_FUNC_ATTRIBUTE void sljit_read_only_buffer_end_writing(sljit_uw addr, sljit_uw size, sljit_sw executable_offset);

/* --------------------------------------------------------------------- */
/*  CPU specific functions                                               */
/* --------------------------------------------------------------------- */

/* Types for sljit_get_register_index */

/* General purpose (integer) registers. */
#define SLJIT_GP_REGISTER 0
/* Floating point registers. */
#define SLJIT_FLOAT_REGISTER 1

/* The following function is a helper function for sljit_emit_op_custom.
   It returns with the real machine register index ( >=0 ) of any registers.

   When type is SLJIT_GP_REGISTER:
      reg must be an SLJIT_R(i), SLJIT_S(i), or SLJIT_SP register

   When type is SLJIT_FLOAT_REGISTER:
      reg must be an SLJIT_FR(i) or SLJIT_FS(i) register

   When type is SLJIT_SIMD_REG_64 / 128 / 256 / 512 :
      reg must be an SLJIT_FR(i) or SLJIT_FS(i) register

   Note: it returns with -1 for unknown registers, such as virtual
         registers on x86-32 or unsupported simd registers. */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_register_index(sljit_s32 type, sljit_s32 reg);

/* Any instruction can be inserted into the instruction stream by
   sljit_emit_op_custom. It has a similar purpose as inline assembly.
   The size parameter must match to the instruction size of the target
   architecture:

         x86: 0 < size <= 15, the instruction argument can be byte aligned.
      Thumb2: if size == 2, the instruction argument must be 2 byte aligned.
              if size == 4, the instruction argument must be 4 byte aligned.
       s390x: size can be 2, 4, or 6, the instruction argument can be byte aligned.
   Otherwise: size must be 4 and instruction argument must be 4 byte aligned. */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_custom(struct sljit_compiler *compiler,
	void *instruction, sljit_u32 size);

/* Flags were set by a 32 bit operation. */
#define SLJIT_CURRENT_FLAGS_32			SLJIT_32

/* Flags were set by an ADD or ADDC operations. */
#define SLJIT_CURRENT_FLAGS_ADD			0x01
/* Flags were set by a SUB or SUBC operation. */
#define SLJIT_CURRENT_FLAGS_SUB			0x02

/* Flags were set by sljit_emit_op2u with SLJIT_SUB opcode.
   Must be combined with SLJIT_CURRENT_FLAGS_SUB. */
#define SLJIT_CURRENT_FLAGS_COMPARE		0x04

/* Flags were set by sljit_emit_op2cmpz operation. */
#define SLJIT_CURRENT_FLAGS_OP2CMPZ		0x08

/* Define the currently available CPU status flags. It is usually used after
   an sljit_emit_label or sljit_emit_op_custom operations to define which CPU
   status flags are available.

   The current_flags must be a valid combination of SLJIT_SET_* and
   SLJIT_CURRENT_FLAGS_* constants. */

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_current_flags(struct sljit_compiler *compiler,
	sljit_s32 current_flags);

/* --------------------------------------------------------------------- */
/*  Serialization functions                                              */
/* --------------------------------------------------------------------- */

/* Label/jump/const enumeration functions. The items in each group
   are enumerated in creation order. Serialization / deserialization
   preserves this order for each group. For example the fifth label
   after deserialization refers to the same machine code location as
   the fifth label before the serialization. */
static SLJIT_INLINE struct sljit_label *sljit_get_first_label(struct sljit_compiler *compiler) { return compiler->labels; }
static SLJIT_INLINE struct sljit_jump *sljit_get_first_jump(struct sljit_compiler *compiler) { return compiler->jumps; }
static SLJIT_INLINE struct sljit_const *sljit_get_first_const(struct sljit_compiler *compiler) { return compiler->consts; }

static SLJIT_INLINE struct sljit_label *sljit_get_next_label(struct sljit_label *label) { return label->next; }
static SLJIT_INLINE struct sljit_jump *sljit_get_next_jump(struct sljit_jump *jump) { return jump->next; }
static SLJIT_INLINE struct sljit_const *sljit_get_next_const(struct sljit_const *const_) { return const_->next; }

/* A number starting from 0 is assigned to each label, which
represents its creation index. The first label created by the
compiler has index 0, the second one has index 1, the third one
has index 2, and so on. The returned value is unspecified after
sljit_generate_code() is called.

It is recommended to use this function to get the creation index
of a label, since sljit_emit_label() may return with the last label,
if no code is generated since the last sljit_emit_label() call. */
SLJIT_API_FUNC_ATTRIBUTE sljit_uw sljit_get_label_index(struct sljit_label *label);

/* The sljit_jump_has_label() and sljit_jump_has_target() functions
returns non-zero value if a label or target is set for the jump
respectively. Both may return with a zero value. The other two
functions return the value assigned to the jump. */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_jump_has_label(struct sljit_jump *jump);
static SLJIT_INLINE struct sljit_label *sljit_jump_get_label(struct sljit_jump *jump) { return jump->u.label; }
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_jump_has_target(struct sljit_jump *jump);
static SLJIT_INLINE sljit_uw sljit_jump_get_target(struct sljit_jump *jump) { return jump->u.target; }
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_jump_is_mov_addr(struct sljit_jump *jump);

/* Option bits for sljit_serialize_compiler. */

/* When debugging is enabled, the serialized buffer contains
debugging information unless this option is specified. */
#define SLJIT_SERIALIZE_IGNORE_DEBUG		0x1

/* Serialize the internal structure of the compiler into a buffer.
If the serialization is successful, the returned value is a newly
allocated buffer which is allocated by the memory allocator assigned
to the compiler. Otherwise the returned value is NULL. Unlike
sljit_generate_code(), serialization does not modify the internal
state of the compiler, so the code generation can be continued.

  options must be the combination of SLJIT_SERIALIZE_* option bits
  size is an output argument, which is set to the byte size of
    the result buffer if the operation is successful

Notes:
  - This function is useful for ahead-of-time compilation (AOT).
  - The returned buffer must be freed later by the caller.
    The SLJIT_FREE() macro is suitable for this purpose:
    SLJIT_FREE(returned_buffer, sljit_get_allocator_data(compiler))
  - Memory allocated by sljit_alloc_memory() is not serialized.
  - The type of the returned buffer is sljit_uw* to emphasize that
    the buffer is word aligned. However, the 'size' output argument
    contains the byte size, so this value is always divisible by
    sizeof(sljit_uw).
*/
SLJIT_API_FUNC_ATTRIBUTE sljit_uw* sljit_serialize_compiler(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_uw *size);

/* Construct a new compiler instance from a buffer produced by
sljit_serialize_compiler(). If the operation is successful, the new
compiler instance is returned. Otherwise the returned value is NULL.

  buffer points to a word aligned memory data which was
    created by sljit_serialize_compiler()
  size is the byte size of the buffer
  options must be 0
  allocator_data specify an allocator specific data, see
                 sljit_create_compiler() for further details

Notes:
  - Labels assigned to jumps are restored with their
    corresponding label in the label set created by
    the deserializer. Target addresses assigned to
    jumps are also restored. Uninitialized jumps
    remain uninitialized.
  - After the deserialization, sljit_generate_code() does
    not need to be the next operation on the returned
    compiler, the code generation can be continued.
    Even sljit_serialize_compiler() can be called again.
  - When debugging is enabled, a buffers without debug
    information cannot be deserialized.
*/
SLJIT_API_FUNC_ATTRIBUTE struct sljit_compiler *sljit_deserialize_compiler(sljit_uw* buffer, sljit_uw size,
	sljit_s32 options, void *allocator_data);

/* --------------------------------------------------------------------- */
/*  Miscellaneous utility functions                                      */
/* --------------------------------------------------------------------- */

/* Get the human readable name of the platform. Can be useful on platforms
   like ARM, where ARM and Thumb2 functions can be mixed, and it is useful
   to know the type of the code generator. */
SLJIT_API_FUNC_ATTRIBUTE const char* sljit_get_platform_name(void);

/* Portable helper function to get an offset of a member.
   Same as offsetof() macro defined in stddef.h */
#define SLJIT_OFFSETOF(base, member) ((sljit_sw)(&((base*)0x10)->member) - 0x10)

#if (defined SLJIT_UTIL_STACK && SLJIT_UTIL_STACK)

/* The sljit_stack structure and its manipulation functions provides
   an implementation for a top-down stack. The stack top is stored
   in the end field of the sljit_stack structure and the stack goes
   down to the min_start field, so the memory region reserved for
   this stack is between min_start (inclusive) and end (exclusive)
   fields. However the application can only use the region between
   start (inclusive) and end (exclusive) fields. The sljit_stack_resize
   function can be used to extend this region up to min_start.

   This feature uses the "address space reserve" feature of modern
   operating systems. Instead of allocating a large memory block
   applications can allocate a small memory region and extend it
   later without moving the content of the memory area. Therefore
   after a successful resize by sljit_stack_resize all pointers into
   this region are still valid.

   Note:
     this structure may not be supported by all operating systems.
     end and max_limit fields are aligned to PAGE_SIZE bytes (usually
         4 Kbyte or more).
     stack should grow in larger steps, e.g. 4Kbyte, 16Kbyte or more. */

struct sljit_stack {
	/* User data, anything can be stored here.
	   Initialized to the same value as the end field. */
	sljit_u8 *top;
/* These members are read only. */
	/* End address of the stack */
	sljit_u8 *end;
	/* Current start address of the stack. */
	sljit_u8 *start;
	/* Lowest start address of the stack. */
	sljit_u8 *min_start;
};

/* Allocates a new stack. Returns NULL if unsuccessful.
   Note: see sljit_create_compiler for the explanation of allocator_data. */
SLJIT_API_FUNC_ATTRIBUTE struct sljit_stack* SLJIT_FUNC sljit_allocate_stack(sljit_uw start_size, sljit_uw max_size, void *allocator_data);
SLJIT_API_FUNC_ATTRIBUTE void SLJIT_FUNC sljit_free_stack(struct sljit_stack *stack, void *allocator_data);

/* Can be used to increase (extend) or decrease (shrink) the stack
   memory area. Returns with new_start if successful and NULL otherwise.
   It always fails if new_start is less than min_start or greater or equal
   than end fields. The fields of the stack are not changed if the returned
   value is NULL (the current memory content is never lost). */
SLJIT_API_FUNC_ATTRIBUTE sljit_u8 *SLJIT_FUNC sljit_stack_resize(struct sljit_stack *stack, sljit_u8 *new_start);

#endif /* (defined SLJIT_UTIL_STACK && SLJIT_UTIL_STACK) */

#if !(defined SLJIT_INDIRECT_CALL && SLJIT_INDIRECT_CALL)

/* Get the entry address of a given function (signed, unsigned result). */
#define SLJIT_FUNC_ADDR(func_name)	((sljit_sw)func_name)
#define SLJIT_FUNC_UADDR(func_name)	((sljit_uw)func_name)

#else /* !(defined SLJIT_INDIRECT_CALL && SLJIT_INDIRECT_CALL) */

/* All JIT related code should be placed in the same context (library, binary, etc.). */

/* Get the entry address of a given function (signed, unsigned result). */
#define SLJIT_FUNC_ADDR(func_name)	(*(sljit_sw*)(void*)func_name)
#define SLJIT_FUNC_UADDR(func_name)	(*(sljit_uw*)(void*)func_name)

/* For powerpc64, the function pointers point to a context descriptor. */
struct sljit_function_context {
	sljit_uw addr;
	sljit_uw r2;
	sljit_uw r11;
};

/* Fill the context arguments using the addr and the function.
   If func_ptr is NULL, it will not be set to the address of context
   If addr is NULL, the function address also comes from the func pointer. */
SLJIT_API_FUNC_ATTRIBUTE void sljit_set_function_context(void** func_ptr, struct sljit_function_context* context, sljit_uw addr, void* func);

#endif /* !(defined SLJIT_INDIRECT_CALL && SLJIT_INDIRECT_CALL) */

#if (defined SLJIT_EXECUTABLE_ALLOCATOR && SLJIT_EXECUTABLE_ALLOCATOR)
/* Free unused executable memory. The allocator keeps some free memory
   around to reduce the number of OS executable memory allocations.
   This improves performance since these calls are costly. However
   it is sometimes desired to free all unused memory regions, e.g.
   before the application terminates. */
SLJIT_API_FUNC_ATTRIBUTE void sljit_free_unused_memory_exec(void);
#endif /* SLJIT_EXECUTABLE_ALLOCATOR */

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* SLJIT_LIR_H_ */
