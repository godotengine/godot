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

#ifndef _SLJIT_LIR_H_
#define _SLJIT_LIR_H_

/*
   ------------------------------------------------------------------------
    Stack-Less JIT compiler for multiple architectures (x86, ARM, PowerPC)
   ------------------------------------------------------------------------

   Short description
    Advantages:
      - The execution can be continued from any LIR instruction. In other
        words, it is possible to jump to any label from anywhere, even from
        a code fragment, which is compiled later, if both compiled code
        shares the same context. See sljit_emit_enter for more details
      - Supports self modifying code: target of (conditional) jump and call
        instructions and some constant values can be dynamically modified
        during runtime
        - although it is not suggested to do it frequently
        - can be used for inline caching: save an important value once
          in the instruction stream
        - since this feature limits the optimization possibilities, a
          special flag must be passed at compile time when these
          instructions are emitted
      - A fixed stack space can be allocated for local variables
      - The compiler is thread-safe
      - The compiler is highly configurable through preprocessor macros.
        You can disable unneeded features (multithreading in single
        threaded applications), and you can use your own system functions
        (including memory allocators). See sljitConfig.h
    Disadvantages:
      - No automatic register allocation, and temporary results are
        not stored on the stack. (hence the name comes)
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

#if !(defined SLJIT_NO_DEFAULT_CONFIG && SLJIT_NO_DEFAULT_CONFIG)
#include "sljitConfig.h"
#endif

/* The following header file defines useful macros for fine tuning
sljit based code generators. They are listed in the beginning
of sljitConfigInternal.h */

#include "sljitConfigInternal.h"

/* --------------------------------------------------------------------- */
/*  Error codes                                                          */
/* --------------------------------------------------------------------- */

/* Indicates no error. */
#define SLJIT_SUCCESS			0
/* After the call of sljit_generate_code(), the error code of the compiler
   is set to this value to avoid future sljit calls (in debug mode at least).
   The complier should be freed after sljit_generate_code(). */
#define SLJIT_ERR_COMPILED		1
/* Cannot allocate non executable memory. */
#define SLJIT_ERR_ALLOC_FAILED		2
/* Cannot allocate executable memory.
   Only for sljit_generate_code() */
#define SLJIT_ERR_EX_ALLOC_FAILED	3
/* Return value for SLJIT_CONFIG_UNSUPPORTED placeholder architecture. */
#define SLJIT_ERR_UNSUPPORTED		4
/* An ivalid argument is passed to any SLJIT function. */
#define SLJIT_ERR_BAD_ARGUMENT		5
/* Dynamic code modification is not enabled. */
#define SLJIT_ERR_DYN_CODE_MOD		6

/* --------------------------------------------------------------------- */
/*  Registers                                                            */
/* --------------------------------------------------------------------- */

/*
  Scratch (R) registers: registers whose may not preserve their values
  across function calls.

  Saved (S) registers: registers whose preserve their values across
  function calls.

  The scratch and saved register sets are overlap. The last scratch register
  is the first saved register, the one before the last is the second saved
  register, and so on.

  If an architecture provides two scratch and three saved registers,
  its scratch and saved register sets are the following:

     R0   |        |   R0 is always a scratch register
     R1   |        |   R1 is always a scratch register
    [R2]  |   S2   |   R2 and S2 represent the same physical register
    [R3]  |   S1   |   R3 and S1 represent the same physical register
    [R4]  |   S0   |   R4 and S0 represent the same physical register

  Note: SLJIT_NUMBER_OF_SCRATCH_REGISTERS would be 2 and
        SLJIT_NUMBER_OF_SAVED_REGISTERS would be 3 for this architecture.

  Note: On all supported architectures SLJIT_NUMBER_OF_REGISTERS >= 12
        and SLJIT_NUMBER_OF_SAVED_REGISTERS >= 6. However, 6 registers
        are virtual on x86-32. See below.

  The purpose of this definition is convenience: saved registers can
  be used as extra scratch registers. For example four registers can
  be specified as scratch registers and the fifth one as saved register
  on the CPU above and any user code which requires four scratch
  registers can run unmodified. The SLJIT compiler automatically saves
  the content of the two extra scratch register on the stack. Scratch
  registers can also be preserved by saving their value on the stack
  but this needs to be done manually.

  Note: To emphasize that registers assigned to R2-R4 are saved
        registers, they are enclosed by square brackets.

  Note: sljit_emit_enter and sljit_set_context defines whether a register
        is S or R register. E.g: when 3 scratches and 1 saved is mapped
        by sljit_emit_enter, the allowed register set will be: R0-R2 and
        S0. Although S2 is mapped to the same position as R2, it does not
        available in the current configuration. Furthermore the S1 register
        is not available at all.
*/

/* When SLJIT_UNUSED is specified as the destination of sljit_emit_op1
   or sljit_emit_op2 operations the result is discarded. If no status
   flags are set, no instructions are emitted for these operations. Data
   prefetch is a special exception, see SLJIT_MOV operation. Other SLJIT
   operations do not support SLJIT_UNUSED as a destination operand. */
#define SLJIT_UNUSED		0

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
   The sljit_get_local_base can be used to obtain the absolute offset. */
#define SLJIT_SP	(SLJIT_NUMBER_OF_REGISTERS + 1)

/* Return with machine word. */

#define SLJIT_RETURN_REG	SLJIT_R0

/* --------------------------------------------------------------------- */
/*  Floating point registers                                             */
/* --------------------------------------------------------------------- */

/* Each floating point register can store a 32 or a 64 bit precision
   value. The FR and FS register sets are overlap in the same way as R
   and S register sets. See above. */

/* Note: SLJIT_UNUSED as destination is not valid for floating point
   operations, since they cannot be used for setting flags. */

/* Floating point scratch registers. */
#define SLJIT_FR0	1
#define SLJIT_FR1	2
#define SLJIT_FR2	3
#define SLJIT_FR3	4
#define SLJIT_FR4	5
#define SLJIT_FR5	6
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
/* All S registers provided by the architecture can be accessed by SLJIT_FS(i)
   The i parameter must be >= 0 and < SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS. */
#define SLJIT_FS(i)	(SLJIT_NUMBER_OF_FLOAT_REGISTERS - (i))

/* Float registers >= SLJIT_FIRST_SAVED_FLOAT_REG are saved registers. */
#define SLJIT_FIRST_SAVED_FLOAT_REG (SLJIT_FS0 - SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS + 1)

/* --------------------------------------------------------------------- */
/*  Argument type definitions                                            */
/* --------------------------------------------------------------------- */

/* Argument type definitions.
   Used by SLJIT_[DEF_]ARGx and SLJIT_[DEF]_RET macros. */

#define SLJIT_ARG_TYPE_VOID 0
#define SLJIT_ARG_TYPE_SW 1
#define SLJIT_ARG_TYPE_UW 2
#define SLJIT_ARG_TYPE_S32 3
#define SLJIT_ARG_TYPE_U32 4
#define SLJIT_ARG_TYPE_F32 5
#define SLJIT_ARG_TYPE_F64 6

/* The following argument type definitions are used by sljit_emit_enter,
   sljit_set_context, sljit_emit_call, and sljit_emit_icall functions.
   The following return type definitions are used by sljit_emit_call
   and sljit_emit_icall functions.

   When a function is called, the first integer argument must be placed
   in SLJIT_R0, the second in SLJIT_R1, and so on. Similarly the first
   floating point argument must be placed in SLJIT_FR0, the second in
   SLJIT_FR1, and so on.

   Example function definition:
     sljit_f32 SLJIT_FUNC example_c_callback(sljit_sw arg_a,
         sljit_f64 arg_b, sljit_u32 arg_c, sljit_f32 arg_d);

   Argument type definition:
     SLJIT_DEF_RET(SLJIT_ARG_TYPE_F32)
        | SLJIT_DEF_ARG1(SLJIT_ARG_TYPE_SW) | SLJIT_DEF_ARG2(SLJIT_ARG_TYPE_F64)
        | SLJIT_DEF_ARG3(SLJIT_ARG_TYPE_U32) | SLJIT_DEF_ARG2(SLJIT_ARG_TYPE_F32)

   Short form of argument type definition:
     SLJIT_RET(F32) | SLJIT_ARG1(SW) | SLJIT_ARG2(F64)
        | SLJIT_ARG3(S32) | SLJIT_ARG4(F32)

   Argument passing:
     arg_a must be placed in SLJIT_R0
     arg_c must be placed in SLJIT_R1
     arg_b must be placed in SLJIT_FR0
     arg_d must be placed in SLJIT_FR1

Note:
   The SLJIT_ARG_TYPE_VOID type is only supported by
   SLJIT_DEF_RET, and SLJIT_ARG_TYPE_VOID is also the
   default value when SLJIT_DEF_RET is not specified. */
#define SLJIT_DEF_SHIFT 4
#define SLJIT_DEF_RET(type) (type)
#define SLJIT_DEF_ARG1(type) ((type) << SLJIT_DEF_SHIFT)
#define SLJIT_DEF_ARG2(type) ((type) << (2 * SLJIT_DEF_SHIFT))
#define SLJIT_DEF_ARG3(type) ((type) << (3 * SLJIT_DEF_SHIFT))
#define SLJIT_DEF_ARG4(type) ((type) << (4 * SLJIT_DEF_SHIFT))

/* Short form of the macros above.

   For example the following definition:
   SLJIT_DEF_RET(SLJIT_ARG_TYPE_SW) | SLJIT_DEF_ARG1(SLJIT_ARG_TYPE_F32)

   can be shortened to:
   SLJIT_RET(SW) | SLJIT_ARG1(F32)

Note:
   The VOID type is only supported by SLJIT_RET, and
   VOID is also the default value when SLJIT_RET is
   not specified. */
#define SLJIT_RET(type) SLJIT_DEF_RET(SLJIT_ARG_TYPE_ ## type)
#define SLJIT_ARG1(type) SLJIT_DEF_ARG1(SLJIT_ARG_TYPE_ ## type)
#define SLJIT_ARG2(type) SLJIT_DEF_ARG2(SLJIT_ARG_TYPE_ ## type)
#define SLJIT_ARG3(type) SLJIT_DEF_ARG3(SLJIT_ARG_TYPE_ ## type)
#define SLJIT_ARG4(type) SLJIT_DEF_ARG4(SLJIT_ARG_TYPE_ ## type)

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
	sljit_uw addr;
	/* The maximum size difference. */
	sljit_uw size;
};

struct sljit_jump {
	struct sljit_jump *next;
	sljit_uw addr;
	sljit_uw flags;
	union {
		sljit_uw target;
		struct sljit_label *label;
	} u;
};

struct sljit_put_label {
	struct sljit_put_label *next;
	struct sljit_label *label;
	sljit_uw addr;
	sljit_uw flags;
};

struct sljit_const {
	struct sljit_const *next;
	sljit_uw addr;
};

struct sljit_compiler {
	sljit_s32 error;
	sljit_s32 options;

	struct sljit_label *labels;
	struct sljit_jump *jumps;
	struct sljit_put_label *put_labels;
	struct sljit_const *consts;
	struct sljit_label *last_label;
	struct sljit_jump *last_jump;
	struct sljit_const *last_const;
	struct sljit_put_label *last_put_label;

	void *allocator_data;
	struct sljit_memory_fragment *buf;
	struct sljit_memory_fragment *abuf;

	/* Used scratch registers. */
	sljit_s32 scratches;
	/* Used saved registers. */
	sljit_s32 saveds;
	/* Used float scratch registers. */
	sljit_s32 fscratches;
	/* Used float saved registers. */
	sljit_s32 fsaveds;
	/* Local stack size. */
	sljit_s32 local_size;
	/* Code size. */
	sljit_uw size;
	/* Relative offset of the executable mapping from the writable mapping. */
	sljit_uw executable_offset;
	/* Executable size for statistical purposes. */
	sljit_uw executable_size;

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	sljit_s32 args;
	sljit_s32 locals_offset;
	sljit_s32 saveds_offset;
	sljit_s32 stack_tmp_size;
#endif

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	sljit_s32 mode32;
#ifdef _WIN64
	sljit_s32 locals_offset;
#endif
#endif

#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5)
	/* Constant pool handling. */
	sljit_uw *cpool;
	sljit_u8 *cpool_unique;
	sljit_uw cpool_diff;
	sljit_uw cpool_fill;
	/* Other members. */
	/* Contains pointer, "ldr pc, [...]" pairs. */
	sljit_uw patches;
#endif

#if (defined SLJIT_CONFIG_ARM_V5 && SLJIT_CONFIG_ARM_V5) || (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7)
	/* Temporary fields. */
	sljit_uw shift_imm;
#endif

#if (defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC)
	sljit_sw imm;
#endif

#if (defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS)
	sljit_s32 delay_slot;
	sljit_s32 cache_arg;
	sljit_sw cache_argw;
#endif

#if (defined SLJIT_CONFIG_SPARC_32 && SLJIT_CONFIG_SPARC_32)
	sljit_s32 delay_slot;
	sljit_s32 cache_arg;
	sljit_sw cache_argw;
#endif

#if (defined SLJIT_CONFIG_TILEGX && SLJIT_CONFIG_TILEGX)
	sljit_s32 cache_arg;
	sljit_sw cache_argw;
#endif

#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	FILE* verbose;
#endif

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS) \
		|| (defined SLJIT_DEBUG && SLJIT_DEBUG)
	/* Flags specified by the last arithmetic instruction.
	   It contains the type of the variable flag. */
	sljit_s32 last_flags;
	/* Local size passed to the functions. */
	sljit_s32 logical_local_size;
#endif

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS) \
		|| (defined SLJIT_DEBUG && SLJIT_DEBUG) \
		|| (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	/* Trust arguments when the API function is called. */
	sljit_s32 skip_checks;
#endif
};

/* --------------------------------------------------------------------- */
/*  Main functions                                                       */
/* --------------------------------------------------------------------- */

/* Creates an sljit compiler. The allocator_data is required by some
   custom memory managers. This pointer is passed to SLJIT_MALLOC
   and SLJIT_FREE macros. Most allocators (including the default
   one) ignores this value, and it is recommended to pass NULL
   as a dummy value for allocator_data.

   Returns NULL if failed. */
SLJIT_API_FUNC_ATTRIBUTE struct sljit_compiler* sljit_create_compiler(void *allocator_data);

/* Frees everything except the compiled machine code. */
SLJIT_API_FUNC_ATTRIBUTE void sljit_free_compiler(struct sljit_compiler *compiler);

/* Returns the current error code. If an error is occurred, future sljit
   calls which uses the same compiler argument returns early with the same
   error code. Thus there is no need for checking the error after every
   call, it is enough to do it before the code is compiled. Removing
   these checks increases the performance of the compiling process. */
static SLJIT_INLINE sljit_s32 sljit_get_compiler_error(struct sljit_compiler *compiler) { return compiler->error; }

/* Sets the compiler error code to SLJIT_ERR_ALLOC_FAILED except
   if an error was detected before. After the error code is set
   the compiler behaves as if the allocation failure happened
   during an sljit function call. This can greatly simplify error
   checking, since only the compiler status needs to be checked
   after the compilation. */
SLJIT_API_FUNC_ATTRIBUTE void sljit_set_compiler_memory_error(struct sljit_compiler *compiler);

/*
   Allocate a small amount of memory. The size must be <= 64 bytes on 32 bit,
   and <= 128 bytes on 64 bit architectures. The memory area is owned by the
   compiler, and freed by sljit_free_compiler. The returned pointer is
   sizeof(sljit_sw) aligned. Excellent for allocating small blocks during
   the compiling, and no need to worry about freeing them. The size is
   enough to contain at most 16 pointers. If the size is outside of the range,
   the function will return with NULL. However, this return value does not
   indicate that there is no more memory (does not set the current error code
   of the compiler to out-of-memory status).
*/
SLJIT_API_FUNC_ATTRIBUTE void* sljit_alloc_memory(struct sljit_compiler *compiler, sljit_s32 size);

#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
/* Passing NULL disables verbose. */
SLJIT_API_FUNC_ATTRIBUTE void sljit_compiler_verbose(struct sljit_compiler *compiler, FILE* verbose);
#endif

/*
   Create executable code from the sljit instruction stream. This is the final step
   of the code generation so no more instructions can be added after this call.
*/

SLJIT_API_FUNC_ATTRIBUTE void* sljit_generate_code(struct sljit_compiler *compiler);

/* Free executable code. */

SLJIT_API_FUNC_ATTRIBUTE void sljit_free_code(void* code);

/*
   When the protected executable allocator is used the JIT code is mapped
   twice. The first mapping has read/write and the second mapping has read/exec
   permissions. This function returns with the relative offset of the executable
   mapping using the writable mapping as the base after the machine code is
   successfully generated. The returned value is always 0 for the normal executable
   allocator, since it uses only one mapping with read/write/exec permissions.
   Dynamic code modifications requires this value.

   Before a successful code generation, this function returns with 0.
*/
static SLJIT_INLINE sljit_sw sljit_get_executable_offset(struct sljit_compiler *compiler) { return compiler->executable_offset; }

/*
   The executable memory consumption of the generated code can be retrieved by
   this function. The returned value can be used for statistical purposes.

   Before a successful code generation, this function returns with 0.
*/
static SLJIT_INLINE sljit_uw sljit_get_generated_code_size(struct sljit_compiler *compiler) { return compiler->executable_size; }

/* Returns with non-zero if the feature or limitation type passed as its
   argument is present on the current CPU.

   Some features (e.g. floating point operations) require hardware (CPU)
   support while others (e.g. move with update) are emulated if not available.
   However even if a feature is emulated, specialized code paths can be faster
   than the emulation. Some limitations are emulated as well so their general
   case is supported but it has extra performance costs. */

/* [Not emulated] Floating-point support is available. */
#define SLJIT_HAS_FPU			0
/* [Limitation] Some registers are virtual registers. */
#define SLJIT_HAS_VIRTUAL_REGISTERS	1
/* [Emulated] Count leading zero is supported. */
#define SLJIT_HAS_CLZ			2
/* [Emulated] Conditional move is supported. */
#define SLJIT_HAS_CMOV			3

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)
/* [Not emulated] SSE2 support is available on x86. */
#define SLJIT_HAS_SSE2			100
#endif

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_has_cpu_feature(sljit_s32 feature_type);

/* Instruction generation. Returns with any error code. If there is no
   error, they return with SLJIT_SUCCESS. */

/*
   The executable code is a function from the viewpoint of the C
   language. The function calls must obey to the ABI (Application
   Binary Interface) of the platform, which specify the purpose of
   machine registers and stack handling among other things. The
   sljit_emit_enter function emits the necessary instructions for
   setting up a new context for the executable code and moves function
   arguments to the saved registers. Furthermore the options argument
   can be used to pass configuration options to the compiler. The
   available options are listed before sljit_emit_enter.

   The function argument list is the combination of SLJIT_ARGx
   (SLJIT_DEF_ARG1) macros. Currently maximum 3 SW / UW
   (SLJIT_ARG_TYPE_SW / LJIT_ARG_TYPE_UW) arguments are supported.
   The first argument goes to SLJIT_S0, the second goes to SLJIT_S1
   and so on. The register set used by the function must be declared
   as well. The number of scratch and saved registers used by the
   function must be passed to sljit_emit_enter. Only R registers
   between R0 and "scratches" argument can be used later. E.g. if
   "scratches" is set to 2, the scratch register set will be limited
   to SLJIT_R0 and SLJIT_R1. The S registers and the floating point
   registers ("fscratches" and "fsaveds") are specified in a similar
   manner. The sljit_emit_enter is also capable of allocating a stack
   space for local variables. The "local_size" argument contains the
   size in bytes of this local area and its staring address is stored
   in SLJIT_SP. The memory area between SLJIT_SP (inclusive) and
   SLJIT_SP + local_size (exclusive) can be modified freely until
   the function returns. The stack space is not initialized.

   Note: the following conditions must met:
         0 <= scratches <= SLJIT_NUMBER_OF_REGISTERS
         0 <= saveds <= SLJIT_NUMBER_OF_REGISTERS
         scratches + saveds <= SLJIT_NUMBER_OF_REGISTERS
         0 <= fscratches <= SLJIT_NUMBER_OF_FLOAT_REGISTERS
         0 <= fsaveds <= SLJIT_NUMBER_OF_FLOAT_REGISTERS
         fscratches + fsaveds <= SLJIT_NUMBER_OF_FLOAT_REGISTERS

   Note: every call of sljit_emit_enter and sljit_set_context
         overwrites the previous context.
*/

/* The absolute address returned by sljit_get_local_base with
offset 0 is aligned to sljit_f64. Otherwise it is aligned to sljit_sw. */
#define SLJIT_F64_ALIGNMENT 0x00000001

/* The local_size must be >= 0 and <= SLJIT_MAX_LOCAL_SIZE. */
#define SLJIT_MAX_LOCAL_SIZE	65536

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size);

/* The machine code has a context (which contains the local stack space size,
   number of used registers, etc.) which initialized by sljit_emit_enter. Several
   functions (like sljit_emit_return) requres this context to be able to generate
   the appropriate code. However, some code fragments (like inline cache) may have
   no normal entry point so their context is unknown for the compiler. Their context
   can be provided to the compiler by the sljit_set_context function.

   Note: every call of sljit_emit_enter and sljit_set_context overwrites
         the previous context. */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size);

/* Return from machine code.  The op argument can be SLJIT_UNUSED which means the
   function does not return with anything or any opcode between SLJIT_MOV and
   SLJIT_MOV_P (see sljit_emit_op1). As for src and srcw they must be 0 if op
   is SLJIT_UNUSED, otherwise see below the description about source and
   destination arguments. */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src, sljit_sw srcw);

/* Generating entry and exit points for fast call functions (see SLJIT_FAST_CALL).
   Both sljit_emit_fast_enter and sljit_emit_fast_return functions preserve the
   values of all registers and stack frame. The return address is stored in the
   dst argument of sljit_emit_fast_enter, and this return address can be passed
   to sljit_emit_fast_return to continue the execution after the fast call.

   Fast calls are cheap operations (usually only a single call instruction is
   emitted) but they do not preserve any registers. However the callee function
   can freely use / update any registers and stack values which can be
   efficiently exploited by various optimizations. Registers can be saved
   manually by the callee function if needed.

   Although returning to different address by sljit_emit_fast_return is possible,
   this address usually cannot be predicted by the return address predictor of
   modern CPUs which may reduce performance. Furthermore using sljit_emit_ijump
   to return is also inefficient since return address prediction is usually
   triggered by a specific form of ijump.

   Flags: - (does not modify flags). */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fast_enter(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw);
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fast_return(struct sljit_compiler *compiler, sljit_s32 src, sljit_sw srcw);

/*
   Source and destination operands for arithmetical instructions
    imm              - a simple immediate value (cannot be used as a destination)
    reg              - any of the registers (immediate argument must be 0)
    [imm]            - absolute immediate memory address
    [reg+imm]        - indirect memory address
    [reg+(reg<<imm)] - indirect indexed memory address (shift must be between 0 and 3)
                       useful for (byte, half, int, sljit_sw) array access
                       (fully supported by both x86 and ARM architectures, and cheap operation on others)
*/

/*
   IMPORATNT NOTE: memory access MUST be naturally aligned except
                   SLJIT_UNALIGNED macro is defined and its value is 1.

     length | alignment
   ---------+-----------
     byte   | 1 byte (any physical_address is accepted)
     half   | 2 byte (physical_address & 0x1 == 0)
     int    | 4 byte (physical_address & 0x3 == 0)
     word   | 4 byte if SLJIT_32BIT_ARCHITECTURE is defined and its value is 1
            | 8 byte if SLJIT_64BIT_ARCHITECTURE is defined and its value is 1
    pointer | size of sljit_p type (4 byte on 32 bit machines, 4 or 8 byte
            | on 64 bit machines)

   Note:   Different architectures have different addressing limitations.
           A single instruction is enough for the following addressing
           modes. Other adrressing modes are emulated by instruction
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
   sparc:  [reg+imm], -4096 <= imm <= 4095
           [reg+reg] is supported
*/

/* Macros for specifying operand types. */
#define SLJIT_MEM		0x80
#define SLJIT_MEM0()		(SLJIT_MEM)
#define SLJIT_MEM1(r1)		(SLJIT_MEM | (r1))
#define SLJIT_MEM2(r1, r2)	(SLJIT_MEM | (r1) | ((r2) << 8))
#define SLJIT_IMM		0x40

/* Set 32 bit operation mode (I) on 64 bit CPUs. This option is ignored on
   32 bit CPUs. When this option is set for an arithmetic operation, only
   the lower 32 bit of the input registers are used, and the CPU status
   flags are set according to the 32 bit result. Although the higher 32 bit
   of the input and the result registers are not defined by SLJIT, it might
   be defined by the CPU architecture (e.g. MIPS). To satisfy these CPU
   requirements all source registers must be the result of those operations
   where this option was also set. Memory loads read 32 bit values rather
   than 64 bit ones. In other words 32 bit and 64 bit operations cannot
   be mixed. The only exception is SLJIT_MOV32 and SLJIT_MOVU32 whose source
   register can hold any 32 or 64 bit value, and it is converted to a 32 bit
   compatible format first. This conversion is free (no instructions are
   emitted) on most CPUs. A 32 bit value can also be converted to a 64 bit
   value by SLJIT_MOV_S32 (sign extension) or SLJIT_MOV_U32 (zero extension).

   Note: memory addressing always uses 64 bit values on 64 bit systems so
         the result of a 32 bit operation must not be used with SLJIT_MEMx
         macros.

   This option is part of the instruction name, so there is no need to
   manually set it. E.g:

     SLJIT_ADD32 == (SLJIT_ADD | SLJIT_I32_OP) */
#define SLJIT_I32_OP		0x100

/* Set F32 (single) precision mode for floating-point computation. This
   option is similar to SLJIT_I32_OP, it just applies to floating point
   registers. When this option is passed, the CPU performs 32 bit floating
   point operations, rather than 64 bit one. Similar to SLJIT_I32_OP, all
   register arguments must be the result of those operations where this
   option was also set.

   This option is part of the instruction name, so there is no need to
   manually set it. E.g:

     SLJIT_MOV_F32 = (SLJIT_MOV_F64 | SLJIT_F32_OP)
 */
#define SLJIT_F32_OP		SLJIT_I32_OP

/* Many CPUs (x86, ARM, PPC) have status flags which can be set according
   to the result of an operation. Other CPUs (MIPS) do not have status
   flags, and results must be stored in registers. To cover both architecture
   types efficiently only two flags are defined by SLJIT:

    * Zero (equal) flag: it is set if the result is zero
    * Variable flag: its value is defined by the last arithmetic operation

   SLJIT instructions can set any or both of these flags. The value of
   these flags is undefined if the instruction does not specify their value.
   The description of each instruction contains the list of allowed flag
   types.

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

   If an instruction (e.g. SLJIT_MOV) does not modify flags the flags are
   unchanged.

   Using these flags can reduce the number of emitted instructions. E.g. a
   fast loop can be implemented by decreasing a counter register and set the
   zero flag to jump back if the counter register has not reached zero.

   Motivation: although CPUs can set a large number of flags, usually their
   values are ignored or only one of them is used. Emulating a large number
   of flags on systems without flag register is complicated so SLJIT
   instructions must specify the flag they want to use and only that flag
   will be emulated. The last arithmetic instruction can be repeated if
   multiple flags need to be checked.
*/

/* Set Zero status flag. */
#define SLJIT_SET_Z			0x0200
/* Set the variable status flag if condition is true.
   See comparison types. */
#define SLJIT_SET(condition)			((condition) << 10)

/* Notes:
     - you cannot postpone conditional jump instructions except if noted that
       the instruction does not set flags (See: SLJIT_KEEP_FLAGS).
     - flag combinations: '|' means 'logical or'. */

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
#define SLJIT_DIVMOD_U32		(SLJIT_DIVMOD_UW | SLJIT_I32_OP)
/* Flags: - (may destroy flags)
   Signed divide of the value in SLJIT_R0 by the value in SLJIT_R1.
   The result is placed into SLJIT_R0 and the remainder into SLJIT_R1.
   Note: if SLJIT_R1 is 0, the behaviour is undefined.
   Note: if SLJIT_R1 is -1 and SLJIT_R0 is integer min (0x800..00),
         the behaviour is undefined. */
#define SLJIT_DIVMOD_SW			(SLJIT_OP0_BASE + 5)
#define SLJIT_DIVMOD_S32		(SLJIT_DIVMOD_SW | SLJIT_I32_OP)
/* Flags: - (may destroy flags)
   Unsigned divide of the value in SLJIT_R0 by the value in SLJIT_R1.
   The result is placed into SLJIT_R0. SLJIT_R1 preserves its value.
   Note: if SLJIT_R1 is 0, the behaviour is undefined. */
#define SLJIT_DIV_UW			(SLJIT_OP0_BASE + 6)
#define SLJIT_DIV_U32			(SLJIT_DIV_UW | SLJIT_I32_OP)
/* Flags: - (may destroy flags)
   Signed divide of the value in SLJIT_R0 by the value in SLJIT_R1.
   The result is placed into SLJIT_R0. SLJIT_R1 preserves its value.
   Note: if SLJIT_R1 is 0, the behaviour is undefined.
   Note: if SLJIT_R1 is -1 and SLJIT_R0 is integer min (0x800..00),
         the behaviour is undefined. */
#define SLJIT_DIV_SW			(SLJIT_OP0_BASE + 7)
#define SLJIT_DIV_S32			(SLJIT_DIV_SW | SLJIT_I32_OP)

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
   P   - pointer (sljit_p) data transfer

   If the destination of a MOV instruction is SLJIT_UNUSED and the source
   operand is a memory address the compiler emits a prefetch instruction
   if this instruction is supported by the current CPU. Higher data sizes
   bring the data closer to the core: a MOV with word size loads the data
   into a higher level cache than a byte size. Otherwise the type does not
   affect the prefetch instruction. Furthermore a prefetch instruction
   never fails, so it can be used to prefetch a data from an address and
   check whether that address is NULL afterwards.
*/

/* Flags: - (does not modify flags) */
#define SLJIT_MOV			(SLJIT_OP1_BASE + 0)
/* Flags: - (does not modify flags) */
#define SLJIT_MOV_U8			(SLJIT_OP1_BASE + 1)
#define SLJIT_MOV32_U8			(SLJIT_MOV_U8 | SLJIT_I32_OP)
/* Flags: - (does not modify flags) */
#define SLJIT_MOV_S8			(SLJIT_OP1_BASE + 2)
#define SLJIT_MOV32_S8			(SLJIT_MOV_S8 | SLJIT_I32_OP)
/* Flags: - (does not modify flags) */
#define SLJIT_MOV_U16			(SLJIT_OP1_BASE + 3)
#define SLJIT_MOV32_U16			(SLJIT_MOV_U16 | SLJIT_I32_OP)
/* Flags: - (does not modify flags) */
#define SLJIT_MOV_S16			(SLJIT_OP1_BASE + 4)
#define SLJIT_MOV32_S16			(SLJIT_MOV_S16 | SLJIT_I32_OP)
/* Flags: - (does not modify flags)
   Note: no SLJIT_MOV32_U32 form, since it is the same as SLJIT_MOV32 */
#define SLJIT_MOV_U32			(SLJIT_OP1_BASE + 5)
/* Flags: - (does not modify flags)
   Note: no SLJIT_MOV32_S32 form, since it is the same as SLJIT_MOV32 */
#define SLJIT_MOV_S32			(SLJIT_OP1_BASE + 6)
/* Flags: - (does not modify flags) */
#define SLJIT_MOV32			(SLJIT_MOV_S32 | SLJIT_I32_OP)
/* Flags: - (does not modify flags)
   Note: load a pointer sized data, useful on x32 (a 32 bit mode on x86-64
         where all x64 features are available, e.g. 16 register) or similar
         compiling modes */
#define SLJIT_MOV_P			(SLJIT_OP1_BASE + 7)
/* Flags: Z
   Note: immediate source argument is not supported */
#define SLJIT_NOT			(SLJIT_OP1_BASE + 8)
#define SLJIT_NOT32			(SLJIT_NOT | SLJIT_I32_OP)
/* Flags: Z | OVERFLOW
   Note: immediate source argument is not supported */
#define SLJIT_NEG			(SLJIT_OP1_BASE + 9)
#define SLJIT_NEG32			(SLJIT_NEG | SLJIT_I32_OP)
/* Count leading zeroes
   Flags: - (may destroy flags)
   Note: immediate source argument is not supported */
#define SLJIT_CLZ			(SLJIT_OP1_BASE + 10)
#define SLJIT_CLZ32			(SLJIT_CLZ | SLJIT_I32_OP)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw);

/* Starting index of opcodes for sljit_emit_op2. */
#define SLJIT_OP2_BASE			96

/* Flags: Z | OVERFLOW | CARRY */
#define SLJIT_ADD			(SLJIT_OP2_BASE + 0)
#define SLJIT_ADD32			(SLJIT_ADD | SLJIT_I32_OP)
/* Flags: CARRY */
#define SLJIT_ADDC			(SLJIT_OP2_BASE + 1)
#define SLJIT_ADDC32			(SLJIT_ADDC | SLJIT_I32_OP)
/* Flags: Z | LESS | GREATER_EQUAL | GREATER | LESS_EQUAL
          SIG_LESS | SIG_GREATER_EQUAL | SIG_GREATER
          SIG_LESS_EQUAL | CARRY */
#define SLJIT_SUB			(SLJIT_OP2_BASE + 2)
#define SLJIT_SUB32			(SLJIT_SUB | SLJIT_I32_OP)
/* Flags: CARRY */
#define SLJIT_SUBC			(SLJIT_OP2_BASE + 3)
#define SLJIT_SUBC32			(SLJIT_SUBC | SLJIT_I32_OP)
/* Note: integer mul
   Flags: MUL_OVERFLOW */
#define SLJIT_MUL			(SLJIT_OP2_BASE + 4)
#define SLJIT_MUL32			(SLJIT_MUL | SLJIT_I32_OP)
/* Flags: Z */
#define SLJIT_AND			(SLJIT_OP2_BASE + 5)
#define SLJIT_AND32			(SLJIT_AND | SLJIT_I32_OP)
/* Flags: Z */
#define SLJIT_OR			(SLJIT_OP2_BASE + 6)
#define SLJIT_OR32			(SLJIT_OR | SLJIT_I32_OP)
/* Flags: Z */
#define SLJIT_XOR			(SLJIT_OP2_BASE + 7)
#define SLJIT_XOR32			(SLJIT_XOR | SLJIT_I32_OP)
/* Flags: Z
   Let bit_length be the length of the shift operation: 32 or 64.
   If src2 is immediate, src2w is masked by (bit_length - 1).
   Otherwise, if the content of src2 is outside the range from 0
   to bit_length - 1, the result is undefined. */
#define SLJIT_SHL			(SLJIT_OP2_BASE + 8)
#define SLJIT_SHL32			(SLJIT_SHL | SLJIT_I32_OP)
/* Flags: Z
   Let bit_length be the length of the shift operation: 32 or 64.
   If src2 is immediate, src2w is masked by (bit_length - 1).
   Otherwise, if the content of src2 is outside the range from 0
   to bit_length - 1, the result is undefined. */
#define SLJIT_LSHR			(SLJIT_OP2_BASE + 9)
#define SLJIT_LSHR32			(SLJIT_LSHR | SLJIT_I32_OP)
/* Flags: Z
   Let bit_length be the length of the shift operation: 32 or 64.
   If src2 is immediate, src2w is masked by (bit_length - 1).
   Otherwise, if the content of src2 is outside the range from 0
   to bit_length - 1, the result is undefined. */
#define SLJIT_ASHR			(SLJIT_OP2_BASE + 10)
#define SLJIT_ASHR32			(SLJIT_ASHR | SLJIT_I32_OP)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w);

/* Starting index of opcodes for sljit_emit_fop1. */
#define SLJIT_FOP1_BASE			128

/* Flags: - (does not modify flags) */
#define SLJIT_MOV_F64			(SLJIT_FOP1_BASE + 0)
#define SLJIT_MOV_F32			(SLJIT_MOV_F64 | SLJIT_F32_OP)
/* Convert opcodes: CONV[DST_TYPE].FROM[SRC_TYPE]
   SRC/DST TYPE can be: D - double, S - single, W - signed word, I - signed int
   Rounding mode when the destination is W or I: round towards zero. */
/* Flags: - (does not modify flags) */
#define SLJIT_CONV_F64_FROM_F32		(SLJIT_FOP1_BASE + 1)
#define SLJIT_CONV_F32_FROM_F64		(SLJIT_CONV_F64_FROM_F32 | SLJIT_F32_OP)
/* Flags: - (does not modify flags) */
#define SLJIT_CONV_SW_FROM_F64		(SLJIT_FOP1_BASE + 2)
#define SLJIT_CONV_SW_FROM_F32		(SLJIT_CONV_SW_FROM_F64 | SLJIT_F32_OP)
/* Flags: - (does not modify flags) */
#define SLJIT_CONV_S32_FROM_F64		(SLJIT_FOP1_BASE + 3)
#define SLJIT_CONV_S32_FROM_F32		(SLJIT_CONV_S32_FROM_F64 | SLJIT_F32_OP)
/* Flags: - (does not modify flags) */
#define SLJIT_CONV_F64_FROM_SW		(SLJIT_FOP1_BASE + 4)
#define SLJIT_CONV_F32_FROM_SW		(SLJIT_CONV_F64_FROM_SW | SLJIT_F32_OP)
/* Flags: - (does not modify flags) */
#define SLJIT_CONV_F64_FROM_S32		(SLJIT_FOP1_BASE + 5)
#define SLJIT_CONV_F32_FROM_S32		(SLJIT_CONV_F64_FROM_S32 | SLJIT_F32_OP)
/* Note: dst is the left and src is the right operand for SLJIT_CMPD.
   Flags: EQUAL_F | LESS_F | GREATER_EQUAL_F | GREATER_F | LESS_EQUAL_F */
#define SLJIT_CMP_F64			(SLJIT_FOP1_BASE + 6)
#define SLJIT_CMP_F32			(SLJIT_CMP_F64 | SLJIT_F32_OP)
/* Flags: - (does not modify flags) */
#define SLJIT_NEG_F64			(SLJIT_FOP1_BASE + 7)
#define SLJIT_NEG_F32			(SLJIT_NEG_F64 | SLJIT_F32_OP)
/* Flags: - (does not modify flags) */
#define SLJIT_ABS_F64			(SLJIT_FOP1_BASE + 8)
#define SLJIT_ABS_F32			(SLJIT_ABS_F64 | SLJIT_F32_OP)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw);

/* Starting index of opcodes for sljit_emit_fop2. */
#define SLJIT_FOP2_BASE			160

/* Flags: - (does not modify flags) */
#define SLJIT_ADD_F64			(SLJIT_FOP2_BASE + 0)
#define SLJIT_ADD_F32			(SLJIT_ADD_F64 | SLJIT_F32_OP)
/* Flags: - (does not modify flags) */
#define SLJIT_SUB_F64			(SLJIT_FOP2_BASE + 1)
#define SLJIT_SUB_F32			(SLJIT_SUB_F64 | SLJIT_F32_OP)
/* Flags: - (does not modify flags) */
#define SLJIT_MUL_F64			(SLJIT_FOP2_BASE + 2)
#define SLJIT_MUL_F32			(SLJIT_MUL_F64 | SLJIT_F32_OP)
/* Flags: - (does not modify flags) */
#define SLJIT_DIV_F64			(SLJIT_FOP2_BASE + 3)
#define SLJIT_DIV_F32			(SLJIT_DIV_F64 | SLJIT_F32_OP)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w);

/* Label and jump instructions. */

SLJIT_API_FUNC_ATTRIBUTE struct sljit_label* sljit_emit_label(struct sljit_compiler *compiler);

/* Invert (negate) conditional type: xor (^) with 0x1 */

/* Integer comparison types. */
#define SLJIT_EQUAL			0
#define SLJIT_EQUAL32			(SLJIT_EQUAL | SLJIT_I32_OP)
#define SLJIT_ZERO			0
#define SLJIT_ZERO32			(SLJIT_ZERO | SLJIT_I32_OP)
#define SLJIT_NOT_EQUAL			1
#define SLJIT_NOT_EQUAL32		(SLJIT_NOT_EQUAL | SLJIT_I32_OP)
#define SLJIT_NOT_ZERO			1
#define SLJIT_NOT_ZERO32		(SLJIT_NOT_ZERO | SLJIT_I32_OP)

#define SLJIT_LESS			2
#define SLJIT_LESS32			(SLJIT_LESS | SLJIT_I32_OP)
#define SLJIT_SET_LESS			SLJIT_SET(SLJIT_LESS)
#define SLJIT_GREATER_EQUAL		3
#define SLJIT_GREATER_EQUAL32		(SLJIT_GREATER_EQUAL | SLJIT_I32_OP)
#define SLJIT_SET_GREATER_EQUAL		SLJIT_SET(SLJIT_GREATER_EQUAL)
#define SLJIT_GREATER			4
#define SLJIT_GREATER32			(SLJIT_GREATER | SLJIT_I32_OP)
#define SLJIT_SET_GREATER		SLJIT_SET(SLJIT_GREATER)
#define SLJIT_LESS_EQUAL		5
#define SLJIT_LESS_EQUAL32		(SLJIT_LESS_EQUAL | SLJIT_I32_OP)
#define SLJIT_SET_LESS_EQUAL		SLJIT_SET(SLJIT_LESS_EQUAL)
#define SLJIT_SIG_LESS			6
#define SLJIT_SIG_LESS32		(SLJIT_SIG_LESS | SLJIT_I32_OP)
#define SLJIT_SET_SIG_LESS		SLJIT_SET(SLJIT_SIG_LESS)
#define SLJIT_SIG_GREATER_EQUAL		7
#define SLJIT_SIG_GREATER_EQUAL32	(SLJIT_SIG_GREATER_EQUAL | SLJIT_I32_OP)
#define SLJIT_SET_SIG_GREATER_EQUAL	SLJIT_SET(SLJIT_SIG_GREATER_EQUAL)
#define SLJIT_SIG_GREATER		8
#define SLJIT_SIG_GREATER32		(SLJIT_SIG_GREATER | SLJIT_I32_OP)
#define SLJIT_SET_SIG_GREATER		SLJIT_SET(SLJIT_SIG_GREATER)
#define SLJIT_SIG_LESS_EQUAL		9
#define SLJIT_SIG_LESS_EQUAL32		(SLJIT_SIG_LESS_EQUAL | SLJIT_I32_OP)
#define SLJIT_SET_SIG_LESS_EQUAL	SLJIT_SET(SLJIT_SIG_LESS_EQUAL)

#define SLJIT_OVERFLOW			10
#define SLJIT_OVERFLOW32		(SLJIT_OVERFLOW | SLJIT_I32_OP)
#define SLJIT_SET_OVERFLOW		SLJIT_SET(SLJIT_OVERFLOW)
#define SLJIT_NOT_OVERFLOW		11
#define SLJIT_NOT_OVERFLOW32		(SLJIT_NOT_OVERFLOW | SLJIT_I32_OP)

#define SLJIT_MUL_OVERFLOW		12
#define SLJIT_MUL_OVERFLOW32		(SLJIT_MUL_OVERFLOW | SLJIT_I32_OP)
#define SLJIT_SET_MUL_OVERFLOW		SLJIT_SET(SLJIT_MUL_OVERFLOW)
#define SLJIT_MUL_NOT_OVERFLOW		13
#define SLJIT_MUL_NOT_OVERFLOW32	(SLJIT_MUL_NOT_OVERFLOW | SLJIT_I32_OP)

/* There is no SLJIT_CARRY or SLJIT_NOT_CARRY. */
#define SLJIT_SET_CARRY			SLJIT_SET(14)

/* Floating point comparison types. */
#define SLJIT_EQUAL_F64			16
#define SLJIT_EQUAL_F32			(SLJIT_EQUAL_F64 | SLJIT_F32_OP)
#define SLJIT_SET_EQUAL_F		SLJIT_SET(SLJIT_EQUAL_F64)
#define SLJIT_NOT_EQUAL_F64		17
#define SLJIT_NOT_EQUAL_F32		(SLJIT_NOT_EQUAL_F64 | SLJIT_F32_OP)
#define SLJIT_SET_NOT_EQUAL_F		SLJIT_SET(SLJIT_NOT_EQUAL_F64)
#define SLJIT_LESS_F64			18
#define SLJIT_LESS_F32			(SLJIT_LESS_F64 | SLJIT_F32_OP)
#define SLJIT_SET_LESS_F		SLJIT_SET(SLJIT_LESS_F64)
#define SLJIT_GREATER_EQUAL_F64		19
#define SLJIT_GREATER_EQUAL_F32		(SLJIT_GREATER_EQUAL_F64 | SLJIT_F32_OP)
#define SLJIT_SET_GREATER_EQUAL_F	SLJIT_SET(SLJIT_GREATER_EQUAL_F64)
#define SLJIT_GREATER_F64		20
#define SLJIT_GREATER_F32		(SLJIT_GREATER_F64 | SLJIT_F32_OP)
#define SLJIT_SET_GREATER_F		SLJIT_SET(SLJIT_GREATER_F64)
#define SLJIT_LESS_EQUAL_F64		21
#define SLJIT_LESS_EQUAL_F32		(SLJIT_LESS_EQUAL_F64 | SLJIT_F32_OP)
#define SLJIT_SET_LESS_EQUAL_F		SLJIT_SET(SLJIT_LESS_EQUAL_F64)
#define SLJIT_UNORDERED_F64		22
#define SLJIT_UNORDERED_F32		(SLJIT_UNORDERED_F64 | SLJIT_F32_OP)
#define SLJIT_SET_UNORDERED_F		SLJIT_SET(SLJIT_UNORDERED_F64)
#define SLJIT_ORDERED_F64		23
#define SLJIT_ORDERED_F32		(SLJIT_ORDERED_F64 | SLJIT_F32_OP)
#define SLJIT_SET_ORDERED_F		SLJIT_SET(SLJIT_ORDERED_F64)

/* Unconditional jump types. */
#define SLJIT_JUMP			24
	/* Fast calling method. See sljit_emit_fast_enter / sljit_emit_fast_return. */
#define SLJIT_FAST_CALL			25
	/* Called function must be declared with the SLJIT_FUNC attribute. */
#define SLJIT_CALL			26
	/* Called function must be declared with cdecl attribute.
	   This is the default attribute for C functions. */
#define SLJIT_CALL_CDECL		27

/* The target can be changed during runtime (see: sljit_set_jump_addr). */
#define SLJIT_REWRITABLE_JUMP		0x1000

/* Emit a jump instruction. The destination is not set, only the type of the jump.
    type must be between SLJIT_EQUAL and SLJIT_FAST_CALL
    type can be combined (or'ed) with SLJIT_REWRITABLE_JUMP

   Flags: does not modify flags. */
SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_jump(struct sljit_compiler *compiler, sljit_s32 type);

/* Emit a C compiler (ABI) compatible function call.
    type must be SLJIT_CALL or SLJIT_CALL_CDECL
    type can be combined (or'ed) with SLJIT_REWRITABLE_JUMP
    arg_types is the combination of SLJIT_RET / SLJIT_ARGx (SLJIT_DEF_RET / SLJIT_DEF_ARGx) macros

   Flags: destroy all flags. */
SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_call(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 arg_types);

/* Basic arithmetic comparison. In most architectures it is implemented as
   an SLJIT_SUB operation (with SLJIT_UNUSED destination and setting
   appropriate flags) followed by a sljit_emit_jump. However some
   architectures (i.e: ARM64 or MIPS) may employ special optimizations here.
   It is suggested to use this comparison form when appropriate.
    type must be between SLJIT_EQUAL and SLJIT_I_SIG_LESS_EQUAL
    type can be combined (or'ed) with SLJIT_REWRITABLE_JUMP

   Flags: may destroy flags. */
SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_cmp(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w);

/* Basic floating point comparison. In most architectures it is implemented as
   an SLJIT_FCMP operation (setting appropriate flags) followed by a
   sljit_emit_jump. However some architectures (i.e: MIPS) may employ
   special optimizations here. It is suggested to use this comparison form
   when appropriate.
    type must be between SLJIT_EQUAL_F64 and SLJIT_ORDERED_F32
    type can be combined (or'ed) with SLJIT_REWRITABLE_JUMP
   Flags: destroy flags.
   Note: if either operand is NaN, the behaviour is undefined for
         types up to SLJIT_S_LESS_EQUAL. */
SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_fcmp(struct sljit_compiler *compiler, sljit_s32 type,
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
    type must be SLJIT_CALL or SLJIT_CALL_CDECL
    arg_types is the combination of SLJIT_RET / SLJIT_ARGx (SLJIT_DEF_RET / SLJIT_DEF_ARGx) macros

   Flags: destroy all flags. */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_icall(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 arg_types, sljit_s32 src, sljit_sw srcw);

/* Perform the operation using the conditional flags as the second argument.
   Type must always be between SLJIT_EQUAL and SLJIT_ORDERED_F64. The value
   represented by the type is 1, if the condition represented by the type
   is fulfilled, and 0 otherwise.

   If op == SLJIT_MOV, SLJIT_MOV32:
     Set dst to the value represented by the type (0 or 1).
     Flags: - (does not modify flags)
   If op == SLJIT_OR, op == SLJIT_AND, op == SLJIT_XOR
     Performs the binary operation using dst as the first, and the value
     represented by type as the second argument. Result is written into dst.
     Flags: Z (may destroy flags) */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_flags(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 type);

/* Emit a conditional mov instruction which moves source to destination,
   if the condition is satisfied. Unlike other arithmetic operations this
   instruction does not support memory access.

   type must be between SLJIT_EQUAL and SLJIT_ORDERED_F64
   dst_reg must be a valid register and it can be combined
      with SLJIT_I32_OP to perform a 32 bit arithmetic operation
   src must be register or immediate (SLJIT_IMM)

   Flags: - (does not modify flags) */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_cmov(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_reg,
	sljit_s32 src, sljit_sw srcw);

/* The following flags are used by sljit_emit_mem() and sljit_emit_fmem(). */

/* When SLJIT_MEM_SUPP is passed, no instructions are emitted.
   Instead the function returns with SLJIT_SUCCESS if the instruction
   form is supported and SLJIT_ERR_UNSUPPORTED otherwise. This flag
   allows runtime checking of available instruction forms. */
#define SLJIT_MEM_SUPP		0x0200
/* Memory load operation. This is the default. */
#define SLJIT_MEM_LOAD		0x0000
/* Memory store operation. */
#define SLJIT_MEM_STORE		0x0400
/* Base register is updated before the memory access. */
#define SLJIT_MEM_PRE		0x0800
/* Base register is updated after the memory access. */
#define SLJIT_MEM_POST		0x1000

/* Emit a single memory load or store with update instruction. When the
   requested instruction form is not supported by the CPU, it returns
   with SLJIT_ERR_UNSUPPORTED instead of emulating the instruction. This
   allows specializing tight loops based on the supported instruction
   forms (see SLJIT_MEM_SUPP flag).

   type must be between SLJIT_MOV and SLJIT_MOV_P and can be
     combined with SLJIT_MEM_* flags. Either SLJIT_MEM_PRE
     or SLJIT_MEM_POST must be specified.
   reg is the source or destination register, and must be
     different from the base register of the mem operand
   mem must be a SLJIT_MEM1() or SLJIT_MEM2() operand

   Flags: - (does not modify flags) */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_mem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw);

/* Same as sljit_emit_mem except the followings:

   type must be SLJIT_MOV_F64 or SLJIT_MOV_F32 and can be
     combined with SLJIT_MEM_* flags. Either SLJIT_MEM_PRE
     or SLJIT_MEM_POST must be specified.
   freg is the source or destination floating point register */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fmem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 mem, sljit_sw memw);

/* Copies the base address of SLJIT_SP + offset to dst. The offset can be
   anything to negate the effect of relative addressing. For example if an
   array of sljit_sw values is stored on the stack from offset 0x40, and R0
   contains the offset of an array item plus 0x120, this item can be
   overwritten by two SLJIT instructions:

   sljit_get_local_base(compiler, SLJIT_R1, 0, 0x40 - 0x120);
   sljit_emit_op1(compiler, SLJIT_MOV, SLJIT_MEM2(SLJIT_R1, SLJIT_R0), 0, SLJIT_IMM, 0x5);

   Flags: - (may destroy flags) */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_local_base(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw offset);

/* Store a value that can be changed runtime (see: sljit_get_const_addr / sljit_set_const)
   Flags: - (does not modify flags) */
SLJIT_API_FUNC_ATTRIBUTE struct sljit_const* sljit_emit_const(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw init_value);

/* Store the value of a label (see: sljit_set_put_label)
   Flags: - (does not modify flags) */
SLJIT_API_FUNC_ATTRIBUTE struct sljit_put_label* sljit_emit_put_label(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw);

/* Set the value stored by put_label to this label. */
SLJIT_API_FUNC_ATTRIBUTE void sljit_set_put_label(struct sljit_put_label *put_label, struct sljit_label *label);

/* After the code generation the address for label, jump and const instructions
   are computed. Since these structures are freed by sljit_free_compiler, the
   addresses must be preserved by the user program elsewere. */
static SLJIT_INLINE sljit_uw sljit_get_label_addr(struct sljit_label *label) { return label->addr; }
static SLJIT_INLINE sljit_uw sljit_get_jump_addr(struct sljit_jump *jump) { return jump->addr; }
static SLJIT_INLINE sljit_uw sljit_get_const_addr(struct sljit_const *const_) { return const_->addr; }

/* Only the address and executable offset are required to perform dynamic
   code modifications. See sljit_get_executable_offset function. */
SLJIT_API_FUNC_ATTRIBUTE void sljit_set_jump_addr(sljit_uw addr, sljit_uw new_target, sljit_sw executable_offset);
SLJIT_API_FUNC_ATTRIBUTE void sljit_set_const(sljit_uw addr, sljit_sw new_constant, sljit_sw executable_offset);

/* --------------------------------------------------------------------- */
/*  Miscellaneous utility functions                                      */
/* --------------------------------------------------------------------- */

#define SLJIT_MAJOR_VERSION	0
#define SLJIT_MINOR_VERSION	94

/* Get the human readable name of the platform. Can be useful on platforms
   like ARM, where ARM and Thumb2 functions can be mixed, and
   it is useful to know the type of the code generator. */
SLJIT_API_FUNC_ATTRIBUTE const char* sljit_get_platform_name(void);

/* Portable helper function to get an offset of a member. */
#define SLJIT_OFFSETOF(base, member) ((sljit_sw)(&((base*)0x10)->member) - 0x10)

#if (defined SLJIT_UTIL_GLOBAL_LOCK && SLJIT_UTIL_GLOBAL_LOCK)
/* This global lock is useful to compile common functions. */
SLJIT_API_FUNC_ATTRIBUTE void SLJIT_FUNC sljit_grab_lock(void);
SLJIT_API_FUNC_ATTRIBUTE void SLJIT_FUNC sljit_release_lock(void);
#endif

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

/* Get the entry address of a given function. */
#define SLJIT_FUNC_OFFSET(func_name)	((sljit_sw)func_name)

#else /* !(defined SLJIT_INDIRECT_CALL && SLJIT_INDIRECT_CALL) */

/* All JIT related code should be placed in the same context (library, binary, etc.). */

#define SLJIT_FUNC_OFFSET(func_name)	(*(sljit_sw*)(void*)func_name)

/* For powerpc64, the function pointers point to a context descriptor. */
struct sljit_function_context {
	sljit_sw addr;
	sljit_sw r2;
	sljit_sw r11;
};

/* Fill the context arguments using the addr and the function.
   If func_ptr is NULL, it will not be set to the address of context
   If addr is NULL, the function address also comes from the func pointer. */
SLJIT_API_FUNC_ATTRIBUTE void sljit_set_function_context(void** func_ptr, struct sljit_function_context* context, sljit_sw addr, void* func);

#endif /* !(defined SLJIT_INDIRECT_CALL && SLJIT_INDIRECT_CALL) */

#if (defined SLJIT_EXECUTABLE_ALLOCATOR && SLJIT_EXECUTABLE_ALLOCATOR)
/* Free unused executable memory. The allocator keeps some free memory
   around to reduce the number of OS executable memory allocations.
   This improves performance since these calls are costly. However
   it is sometimes desired to free all unused memory regions, e.g.
   before the application terminates. */
SLJIT_API_FUNC_ATTRIBUTE void sljit_free_unused_memory_exec(void);
#endif

/* --------------------------------------------------------------------- */
/*  CPU specific functions                                               */
/* --------------------------------------------------------------------- */

/* The following function is a helper function for sljit_emit_op_custom.
   It returns with the real machine register index ( >=0 ) of any SLJIT_R,
   SLJIT_S and SLJIT_SP registers.

   Note: it returns with -1 for virtual registers (only on x86-32). */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_register_index(sljit_s32 reg);

/* The following function is a helper function for sljit_emit_op_custom.
   It returns with the real machine register index of any SLJIT_FLOAT register.

   Note: the index is always an even number on ARM (except ARM-64), MIPS, and SPARC. */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_float_register_index(sljit_s32 reg);

/* Any instruction can be inserted into the instruction stream by
   sljit_emit_op_custom. It has a similar purpose as inline assembly.
   The size parameter must match to the instruction size of the target
   architecture:

         x86: 0 < size <= 15. The instruction argument can be byte aligned.
      Thumb2: if size == 2, the instruction argument must be 2 byte aligned.
              if size == 4, the instruction argument must be 4 byte aligned.
   Otherwise: size must be 4 and instruction argument must be 4 byte aligned. */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_custom(struct sljit_compiler *compiler,
	void *instruction, sljit_s32 size);

/* Define the currently available CPU status flags. It is usually used after an
   sljit_emit_op_custom call to define which flags are set. */

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_current_flags(struct sljit_compiler *compiler,
	sljit_s32 current_flags);

#endif /* _SLJIT_LIR_H_ */
