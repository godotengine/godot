/*
 *    Stack-less Just-In-Time compiler
 *
 *    Copyright 2009-2012 Zoltan Herczeg (hzmester@freemail.hu). All rights reserved.
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

     R0   |  [S4]  |   R0 and S4 represent the same physical register
     R1   |  [S3]  |   R1 and S3 represent the same physical register
    [R2]  |   S2   |   R2 and S2 represent the same physical register
    [R3]  |   S1   |   R3 and S1 represent the same physical register
    [R4]  |   S0   |   R4 and S0 represent the same physical register

  Note: SLJIT_NUMBER_OF_SCRATCH_REGISTERS would be 2 and
        SLJIT_NUMBER_OF_SAVED_REGISTERS would be 3 for this architecture.

  Note: On all supported architectures SLJIT_NUMBER_OF_REGISTERS >= 10
        and SLJIT_NUMBER_OF_SAVED_REGISTERS >= 5. However, 4 registers
        are virtual on x86-32. See below.

  The purpose of this definition is convenience. Although a register
  is either scratch register or saved register, SLJIT allows accessing
  them from the other set. For example, four registers can be used as
  scratch registers and the fifth one as saved register on the architecture
  above. Of course the last two scratch registers (R2 and R3) from this
  four will be saved on the stack, because they are defined as saved
  registers in the application binary interface. Still R2 and R3 can be
  used for referencing to these registers instead of S2 and S1, which
  makes easier to write platform independent code. Scratch registers
  can be saved registers in a similar way, but these extra saved
  registers will not be preserved across function calls! Hence the
  application must save them on those platforms, where the number of
  saved registers is too low. This can be done by copy them onto
  the stack and restore them after a function call.

  Note: To emphasize that registers assigned to R2-R4 are saved
        registers, they are enclosed by square brackets. S3-S4
        are marked in a similar way.

  Note: sljit_emit_enter and sljit_set_context defines whether a register
        is S or R register. E.g: when 3 scratches and 1 saved is mapped
        by sljit_emit_enter, the allowed register set will be: R0-R2 and
        S0. Although S2 is mapped to the same position as R2, it does not
        available in the current configuration. Furthermore the R3 (S1)
        register does not available as well.
*/

/* When SLJIT_UNUSED is specified as destination, the result is discarded. */
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

/* x86 prefers specific registers for special purposes. In case of shift
   by register it supports only SLJIT_R2 for shift argument
   (which is the src2 argument of sljit_emit_op2). If another register is
   used, sljit must exchange data between registers which cause a minor
   slowdown. Other architectures has no such limitation. */

#define SLJIT_PREF_SHIFT_REG	SLJIT_R2

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
	sljit_sw flags;
	union {
		sljit_uw target;
		struct sljit_label* label;
	} u;
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
	struct sljit_const *consts;
	struct sljit_label *last_label;
	struct sljit_jump *last_jump;
	struct sljit_const *last_const;

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
#endif

#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
	sljit_s32 mode32;
#endif

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)
	sljit_s32 flags_saved;
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
	sljit_s32 cache_arg;
	sljit_sw cache_argw;
#endif

#if (defined SLJIT_CONFIG_ARM_THUMB2 && SLJIT_CONFIG_ARM_THUMB2)
	sljit_s32 cache_arg;
	sljit_sw cache_argw;
#endif

#if (defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64)
	sljit_s32 cache_arg;
	sljit_sw cache_argw;
#endif

#if (defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC)
	sljit_sw imm;
	sljit_s32 cache_arg;
	sljit_sw cache_argw;
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
	/* Local size passed to the functions. */
	sljit_s32 logical_local_size;
#endif

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS) \
		|| (defined SLJIT_DEBUG && SLJIT_DEBUG) \
		|| (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
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

/* Instruction generation. Returns with any error code. If there is no
   error, they return with SLJIT_SUCCESS. */

/*
   The executable code is a function call from the viewpoint of the C
   language. The function calls must obey to the ABI (Application
   Binary Interface) of the platform, which specify the purpose of
   all machine registers and stack handling among other things. The
   sljit_emit_enter function emits the necessary instructions for
   setting up a new context for the executable code and moves function
   arguments to the saved registers. Furthermore the options argument
   can be used to pass configuration options to the compiler. The
   available options are listed before sljit_emit_enter.

   The number of sljit_sw arguments passed to the generated function
   are specified in the "args" parameter. The number of arguments must
   be less than or equal to 3. The first argument goes to SLJIT_S0,
   the second goes to SLJIT_S1 and so on. The register set used by
   the function must be declared as well. The number of scratch and
   saved registers used by the function must be passed to sljit_emit_enter.
   Only R registers between R0 and "scratches" argument can be used
   later. E.g. if "scratches" is set to 2, the register set will be
   limited to R0 and R1. The S registers and the floating point
   registers ("fscratches" and "fsaveds") are specified in a similar
   way. The sljit_emit_enter is also capable of allocating a stack
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
offset 0 is aligned to sljit_d. Otherwise it is aligned to sljit_uw. */
#define SLJIT_DOUBLE_ALIGNMENT 0x00000001

/* The local_size must be >= 0 and <= SLJIT_MAX_LOCAL_SIZE. */
#define SLJIT_MAX_LOCAL_SIZE	65536

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 args, sljit_s32 scratches, sljit_s32 saveds,
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
	sljit_s32 options, sljit_s32 args, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size);

/* Return from machine code.  The op argument can be SLJIT_UNUSED which means the
   function does not return with anything or any opcode between SLJIT_MOV and
   SLJIT_MOV_P (see sljit_emit_op1). As for src and srcw they must be 0 if op
   is SLJIT_UNUSED, otherwise see below the description about source and
   destination arguments. */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src, sljit_sw srcw);

/* Fast calling mechanism for utility functions (see SLJIT_FAST_CALL). All registers and
   even the stack frame is passed to the callee. The return address is preserved in
   dst/dstw by sljit_emit_fast_enter (the type of the value stored by this function
   is sljit_p), and sljit_emit_fast_return can use this as a return value later. */

/* Note: only for sljit specific, non ABI compilant calls. Fast, since only a few machine
   instructions are needed. Excellent for small uility functions, where saving registers
   and setting up a new stack frame would cost too much performance. However, it is still
   possible to return to the address of the caller (or anywhere else). */

/* Note: flags are not changed (unlike sljit_emit_enter / sljit_emit_return). */

/* Note: although sljit_emit_fast_return could be replaced by an ijump, it is not suggested,
   since many architectures do clever branch prediction on call / return instruction pairs. */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fast_enter(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw);
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fast_return(struct sljit_compiler *compiler, sljit_s32 src, sljit_sw srcw);

/*
   Source and destination values for arithmetical instructions
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

/* Register output: simply the name of the register.
   For destination, you can use SLJIT_UNUSED as well. */
#define SLJIT_MEM		0x80
#define SLJIT_MEM0()		(SLJIT_MEM)
#define SLJIT_MEM1(r1)		(SLJIT_MEM | (r1))
#define SLJIT_MEM2(r1, r2)	(SLJIT_MEM | (r1) | ((r2) << 8))
#define SLJIT_IMM		0x40

/* Set 32 bit operation mode (I) on 64 bit CPUs. This flag is ignored on 32
   bit CPUs. When this flag is set for an arithmetic operation, only the
   lower 32 bit of the input register(s) are used, and the CPU status flags
   are set according to the 32 bit result. Although the higher 32 bit of
   the input and the result registers are not defined by SLJIT, it might be
   defined by the CPU architecture (e.g. MIPS). To satisfy these requirements
   all source registers must be computed by operations where this flag is
   also set. In other words 32 and 64 bit arithmetic operations cannot be
   mixed. The only exception is SLJIT_IMOV and SLJIT_IMOVU whose source
   register can hold any 32 or 64 bit value. This source register is
   converted to a 32 bit compatible format. SLJIT does not generate any
   instructions on certain CPUs (e.g. on x86 and ARM) if the source and
   destination operands are the same registers. Affects sljit_emit_op0,
   sljit_emit_op1 and sljit_emit_op2. */
#define SLJIT_I32_OP		0x100

/* F32 precision mode (SP). This flag is similar to SLJIT_I32_OP, just
   it applies to floating point registers (it is even the same bit). When
   this flag is passed, the CPU performs 32 bit floating point operations.
   Similar to SLJIT_I32_OP, all register arguments must be computed by
   floating point operations where this flag is also set. Affects
   sljit_emit_fop1, sljit_emit_fop2 and sljit_emit_fcmp. */
#define SLJIT_F32_OP		0x100

/* Common CPU status flags for all architectures (x86, ARM, PPC)
    - carry flag
    - overflow flag
    - zero flag
    - negative/positive flag (depends on arc)
   On mips, these flags are emulated by software. */

/* By default, the instructions may, or may not set the CPU status flags.
   Forcing to set or keep status flags can be done with the following flags: */

/* Note: sljit tries to emit the minimum number of instructions. Using these
   flags can increase them, so use them wisely to avoid unnecessary code generation. */

/* Set Equal (Zero) status flag (E). */
#define SLJIT_SET_E			0x0200
/* Set unsigned status flag (U). */
#define SLJIT_SET_U			0x0400
/* Set signed status flag (S). */
#define SLJIT_SET_S			0x0800
/* Set signed overflow flag (O). */
#define SLJIT_SET_O			0x1000
/* Set carry flag (C).
   Note: Kinda unsigned overflow, but behaves differently on various cpus. */
#define SLJIT_SET_C			0x2000
/* Do not modify the flags (K).
   Note: This flag cannot be combined with any other SLJIT_SET_* flag. */
#define SLJIT_KEEP_FLAGS		0x4000

/* Notes:
     - you cannot postpone conditional jump instructions except if noted that
       the instruction does not set flags (See: SLJIT_KEEP_FLAGS).
     - flag combinations: '|' means 'logical or'. */

/* Starting index of opcodes for sljit_emit_op0. */
#define SLJIT_OP0_BASE			0

/* Flags: - (never set any flags)
   Note: breakpoint instruction is not supported by all architectures (e.g. ppc)
         It falls back to SLJIT_NOP in those cases. */
#define SLJIT_BREAKPOINT		(SLJIT_OP0_BASE + 0)
/* Flags: - (never set any flags)
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
/* Flags: I - (may destroy flags)
   Unsigned divide of the value in SLJIT_R0 by the value in SLJIT_R1.
   The result is placed into SLJIT_R0 and the remainder into SLJIT_R1.
   Note: if SLJIT_R1 is 0, the behaviour is undefined. */
#define SLJIT_DIVMOD_UW			(SLJIT_OP0_BASE + 4)
#define SLJIT_DIVMOD_U32		(SLJIT_DIVMOD_UW | SLJIT_I32_OP)
/* Flags: I - (may destroy flags)
   Signed divide of the value in SLJIT_R0 by the value in SLJIT_R1.
   The result is placed into SLJIT_R0 and the remainder into SLJIT_R1.
   Note: if SLJIT_R1 is 0, the behaviour is undefined.
   Note: if SLJIT_R1 is -1 and SLJIT_R0 is integer min (0x800..00),
         the behaviour is undefined. */
#define SLJIT_DIVMOD_SW			(SLJIT_OP0_BASE + 5)
#define SLJIT_DIVMOD_S32		(SLJIT_DIVMOD_SW | SLJIT_I32_OP)
/* Flags: I - (may destroy flags)
   Unsigned divide of the value in SLJIT_R0 by the value in SLJIT_R1.
   The result is placed into SLJIT_R0. SLJIT_R1 preserves its value.
   Note: if SLJIT_R1 is 0, the behaviour is undefined. */
#define SLJIT_DIV_UW			(SLJIT_OP0_BASE + 6)
#define SLJIT_DIV_U32			(SLJIT_DIV_UW | SLJIT_I32_OP)
/* Flags: I - (may destroy flags)
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

/* Notes for MOV instructions:
   U = Mov with update (pre form). If source or destination defined as SLJIT_MEM1(r1)
       or SLJIT_MEM2(r1, r2), r1 is increased by the sum of r2 and the constant argument
   UB = unsigned byte (8 bit)
   SB = signed byte (8 bit)
   UH = unsigned half (16 bit)
   SH = signed half (16 bit)
   UI = unsigned int (32 bit)
   SI = signed int (32 bit)
   P  = pointer (sljit_p) size */

/* Flags: - (never set any flags) */
#define SLJIT_MOV			(SLJIT_OP1_BASE + 0)
/* Flags: I - (never set any flags) */
#define SLJIT_MOV_U8			(SLJIT_OP1_BASE + 1)
#define SLJIT_MOV32_U8			(SLJIT_MOV_U8 | SLJIT_I32_OP)
/* Flags: I - (never set any flags) */
#define SLJIT_MOV_S8			(SLJIT_OP1_BASE + 2)
#define SLJIT_MOV32_S8			(SLJIT_MOV_S8 | SLJIT_I32_OP)
/* Flags: I - (never set any flags) */
#define SLJIT_MOV_U16			(SLJIT_OP1_BASE + 3)
#define SLJIT_MOV32_U16			(SLJIT_MOV_U16 | SLJIT_I32_OP)
/* Flags: I - (never set any flags) */
#define SLJIT_MOV_S16			(SLJIT_OP1_BASE + 4)
#define SLJIT_MOV32_S16			(SLJIT_MOV_S16 | SLJIT_I32_OP)
/* Flags: I - (never set any flags)
   Note: no SLJIT_MOV32_U32 form, since it is the same as SLJIT_MOV32 */
#define SLJIT_MOV_U32			(SLJIT_OP1_BASE + 5)
/* Flags: I - (never set any flags)
   Note: no SLJIT_MOV32_S32 form, since it is the same as SLJIT_MOV32 */
#define SLJIT_MOV_S32			(SLJIT_OP1_BASE + 6)
/* Flags: I - (never set any flags) */
#define SLJIT_MOV32			(SLJIT_MOV_S32 | SLJIT_I32_OP)
/* Flags: - (never set any flags) */
#define SLJIT_MOV_P			(SLJIT_OP1_BASE + 7)
/* Flags: - (never set any flags) */
#define SLJIT_MOVU			(SLJIT_OP1_BASE + 8)
/* Flags: I - (never set any flags) */
#define SLJIT_MOVU_U8			(SLJIT_OP1_BASE + 9)
#define SLJIT_MOVU32_U8			(SLJIT_MOVU_U8 | SLJIT_I32_OP)
/* Flags: I - (never set any flags) */
#define SLJIT_MOVU_S8			(SLJIT_OP1_BASE + 10)
#define SLJIT_MOVU32_S8			(SLJIT_MOVU_S8 | SLJIT_I32_OP)
/* Flags: I - (never set any flags) */
#define SLJIT_MOVU_U16			(SLJIT_OP1_BASE + 11)
#define SLJIT_MOVU32_U16			(SLJIT_MOVU_U16 | SLJIT_I32_OP)
/* Flags: I - (never set any flags) */
#define SLJIT_MOVU_S16			(SLJIT_OP1_BASE + 12)
#define SLJIT_MOVU32_S16		(SLJIT_MOVU_S16 | SLJIT_I32_OP)
/* Flags: I - (never set any flags)
   Note: no SLJIT_MOVU32_U32 form, since it is the same as SLJIT_MOVU32 */
#define SLJIT_MOVU_U32			(SLJIT_OP1_BASE + 13)
/* Flags: I - (never set any flags)
   Note: no SLJIT_MOVU32_S32 form, since it is the same as SLJIT_MOVU32 */
#define SLJIT_MOVU_S32			(SLJIT_OP1_BASE + 14)
/* Flags: I - (never set any flags) */
#define SLJIT_MOVU32			(SLJIT_MOVU_S32 | SLJIT_I32_OP)
/* Flags: - (never set any flags) */
#define SLJIT_MOVU_P			(SLJIT_OP1_BASE + 15)
/* Flags: I | E | K */
#define SLJIT_NOT			(SLJIT_OP1_BASE + 16)
#define SLJIT_NOT32			(SLJIT_NOT | SLJIT_I32_OP)
/* Flags: I | E | O | K */
#define SLJIT_NEG			(SLJIT_OP1_BASE + 17)
#define SLJIT_NEG32			(SLJIT_NEG | SLJIT_I32_OP)
/* Count leading zeroes
   Flags: I | E | K
   Important note! Sparc 32 does not support K flag, since
   the required popc instruction is introduced only in sparc 64. */
#define SLJIT_CLZ			(SLJIT_OP1_BASE + 18)
#define SLJIT_CLZ32			(SLJIT_CLZ | SLJIT_I32_OP)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw);

/* Starting index of opcodes for sljit_emit_op2. */
#define SLJIT_OP2_BASE			96

/* Flags: I | E | O | C | K */
#define SLJIT_ADD			(SLJIT_OP2_BASE + 0)
#define SLJIT_ADD32			(SLJIT_ADD | SLJIT_I32_OP)
/* Flags: I | C | K */
#define SLJIT_ADDC			(SLJIT_OP2_BASE + 1)
#define SLJIT_ADDC32			(SLJIT_ADDC | SLJIT_I32_OP)
/* Flags: I | E | U | S | O | C | K */
#define SLJIT_SUB			(SLJIT_OP2_BASE + 2)
#define SLJIT_SUB32			(SLJIT_SUB | SLJIT_I32_OP)
/* Flags: I | C | K */
#define SLJIT_SUBC			(SLJIT_OP2_BASE + 3)
#define SLJIT_SUBC32			(SLJIT_SUBC | SLJIT_I32_OP)
/* Note: integer mul
   Flags: I | O (see SLJIT_C_MUL_*) | K */
#define SLJIT_MUL			(SLJIT_OP2_BASE + 4)
#define SLJIT_MUL32			(SLJIT_MUL | SLJIT_I32_OP)
/* Flags: I | E | K */
#define SLJIT_AND			(SLJIT_OP2_BASE + 5)
#define SLJIT_AND32			(SLJIT_AND | SLJIT_I32_OP)
/* Flags: I | E | K */
#define SLJIT_OR			(SLJIT_OP2_BASE + 6)
#define SLJIT_OR32			(SLJIT_OR | SLJIT_I32_OP)
/* Flags: I | E | K */
#define SLJIT_XOR			(SLJIT_OP2_BASE + 7)
#define SLJIT_XOR32			(SLJIT_XOR | SLJIT_I32_OP)
/* Flags: I | E | K
   Let bit_length be the length of the shift operation: 32 or 64.
   If src2 is immediate, src2w is masked by (bit_length - 1).
   Otherwise, if the content of src2 is outside the range from 0
   to bit_length - 1, the result is undefined. */
#define SLJIT_SHL			(SLJIT_OP2_BASE + 8)
#define SLJIT_SHL32			(SLJIT_SHL | SLJIT_I32_OP)
/* Flags: I | E | K
   Let bit_length be the length of the shift operation: 32 or 64.
   If src2 is immediate, src2w is masked by (bit_length - 1).
   Otherwise, if the content of src2 is outside the range from 0
   to bit_length - 1, the result is undefined. */
#define SLJIT_LSHR			(SLJIT_OP2_BASE + 9)
#define SLJIT_LSHR32			(SLJIT_LSHR | SLJIT_I32_OP)
/* Flags: I | E | K
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

/* Returns with non-zero if fpu is available. */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_is_fpu_available(void);

/* Starting index of opcodes for sljit_emit_fop1. */
#define SLJIT_FOP1_BASE			128

/* Flags: SP - (never set any flags) */
#define SLJIT_MOV_F64			(SLJIT_FOP1_BASE + 0)
#define SLJIT_MOV_F32			(SLJIT_MOV_F64 | SLJIT_F32_OP)
/* Convert opcodes: CONV[DST_TYPE].FROM[SRC_TYPE]
   SRC/DST TYPE can be: D - double, S - single, W - signed word, I - signed int
   Rounding mode when the destination is W or I: round towards zero. */
/* Flags: SP - (never set any flags) */
#define SLJIT_CONV_F64_FROM_F32		(SLJIT_FOP1_BASE + 1)
#define SLJIT_CONV_F32_FROM_F64		(SLJIT_CONV_F64_FROM_F32 | SLJIT_F32_OP)
/* Flags: SP - (never set any flags) */
#define SLJIT_CONV_SW_FROM_F64		(SLJIT_FOP1_BASE + 2)
#define SLJIT_CONV_SW_FROM_F32		(SLJIT_CONV_SW_FROM_F64 | SLJIT_F32_OP)
/* Flags: SP - (never set any flags) */
#define SLJIT_CONV_S32_FROM_F64		(SLJIT_FOP1_BASE + 3)
#define SLJIT_CONV_S32_FROM_F32		(SLJIT_CONV_S32_FROM_F64 | SLJIT_F32_OP)
/* Flags: SP - (never set any flags) */
#define SLJIT_CONV_F64_FROM_SW		(SLJIT_FOP1_BASE + 4)
#define SLJIT_CONV_F32_FROM_SW		(SLJIT_CONV_F64_FROM_SW | SLJIT_F32_OP)
/* Flags: SP - (never set any flags) */
#define SLJIT_CONV_F64_FROM_S32		(SLJIT_FOP1_BASE + 5)
#define SLJIT_CONV_F32_FROM_S32		(SLJIT_CONV_F64_FROM_S32 | SLJIT_F32_OP)
/* Note: dst is the left and src is the right operand for SLJIT_CMPD.
   Note: NaN check is always performed. If SLJIT_C_FLOAT_UNORDERED flag
         is set, the comparison result is unpredictable.
   Flags: SP | E | S (see SLJIT_C_FLOAT_*) */
#define SLJIT_CMP_F64			(SLJIT_FOP1_BASE + 6)
#define SLJIT_CMP_F32			(SLJIT_CMP_F64 | SLJIT_F32_OP)
/* Flags: SP - (never set any flags) */
#define SLJIT_NEG_F64			(SLJIT_FOP1_BASE + 7)
#define SLJIT_NEG_F32			(SLJIT_NEG_F64 | SLJIT_F32_OP)
/* Flags: SP - (never set any flags) */
#define SLJIT_ABS_F64			(SLJIT_FOP1_BASE + 8)
#define SLJIT_ABS_F32			(SLJIT_ABS_F64 | SLJIT_F32_OP)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw);

/* Starting index of opcodes for sljit_emit_fop2. */
#define SLJIT_FOP2_BASE			160

/* Flags: SP - (never set any flags) */
#define SLJIT_ADD_F64			(SLJIT_FOP2_BASE + 0)
#define SLJIT_ADD_F32			(SLJIT_ADD_F64 | SLJIT_F32_OP)
/* Flags: SP - (never set any flags) */
#define SLJIT_SUB_F64			(SLJIT_FOP2_BASE + 1)
#define SLJIT_SUB_F32			(SLJIT_SUB_F64 | SLJIT_F32_OP)
/* Flags: SP - (never set any flags) */
#define SLJIT_MUL_F64			(SLJIT_FOP2_BASE + 2)
#define SLJIT_MUL_F32			(SLJIT_MUL_F64 | SLJIT_F32_OP)
/* Flags: SP - (never set any flags) */
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
#define SLJIT_GREATER_EQUAL		3
#define SLJIT_GREATER_EQUAL32		(SLJIT_GREATER_EQUAL | SLJIT_I32_OP)
#define SLJIT_GREATER			4
#define SLJIT_GREATER32			(SLJIT_GREATER | SLJIT_I32_OP)
#define SLJIT_LESS_EQUAL		5
#define SLJIT_LESS_EQUAL32		(SLJIT_LESS_EQUAL | SLJIT_I32_OP)
#define SLJIT_SIG_LESS			6
#define SLJIT_SIG_LESS32		(SLJIT_SIG_LESS | SLJIT_I32_OP)
#define SLJIT_SIG_GREATER_EQUAL		7
#define SLJIT_SIG_GREATER_EQUAL32	(SLJIT_SIG_GREATER_EQUAL | SLJIT_I32_OP)
#define SLJIT_SIG_GREATER		8
#define SLJIT_SIG_GREATER32		(SLJIT_SIG_GREATER | SLJIT_I32_OP)
#define SLJIT_SIG_LESS_EQUAL		9
#define SLJIT_SIG_LESS_EQUAL32		(SLJIT_SIG_LESS_EQUAL | SLJIT_I32_OP)

#define SLJIT_OVERFLOW			10
#define SLJIT_OVERFLOW32		(SLJIT_OVERFLOW | SLJIT_I32_OP)
#define SLJIT_NOT_OVERFLOW		11
#define SLJIT_NOT_OVERFLOW32		(SLJIT_NOT_OVERFLOW | SLJIT_I32_OP)

#define SLJIT_MUL_OVERFLOW		12
#define SLJIT_MUL_OVERFLOW32		(SLJIT_MUL_OVERFLOW | SLJIT_I32_OP)
#define SLJIT_MUL_NOT_OVERFLOW		13
#define SLJIT_MUL_NOT_OVERFLOW32	(SLJIT_MUL_NOT_OVERFLOW | SLJIT_I32_OP)

/* Floating point comparison types. */
#define SLJIT_EQUAL_F64			14
#define SLJIT_EQUAL_F32			(SLJIT_EQUAL_F64 | SLJIT_F32_OP)
#define SLJIT_NOT_EQUAL_F64		15
#define SLJIT_NOT_EQUAL_F32		(SLJIT_NOT_EQUAL_F64 | SLJIT_F32_OP)
#define SLJIT_LESS_F64			16
#define SLJIT_LESS_F32			(SLJIT_LESS_F64 | SLJIT_F32_OP)
#define SLJIT_GREATER_EQUAL_F64		17
#define SLJIT_GREATER_EQUAL_F32		(SLJIT_GREATER_EQUAL_F64 | SLJIT_F32_OP)
#define SLJIT_GREATER_F64		18
#define SLJIT_GREATER_F32		(SLJIT_GREATER_F64 | SLJIT_F32_OP)
#define SLJIT_LESS_EQUAL_F64		19
#define SLJIT_LESS_EQUAL_F32		(SLJIT_LESS_EQUAL_F64 | SLJIT_F32_OP)
#define SLJIT_UNORDERED_F64		20
#define SLJIT_UNORDERED_F32		(SLJIT_UNORDERED_F64 | SLJIT_F32_OP)
#define SLJIT_ORDERED_F64		21
#define SLJIT_ORDERED_F32		(SLJIT_ORDERED_F64 | SLJIT_F32_OP)

/* Unconditional jump types. */
#define SLJIT_JUMP			22
#define SLJIT_FAST_CALL			23
#define SLJIT_CALL0			24
#define SLJIT_CALL1			25
#define SLJIT_CALL2			26
#define SLJIT_CALL3			27

/* Fast calling method. See sljit_emit_fast_enter / sljit_emit_fast_return. */

/* The target can be changed during runtime (see: sljit_set_jump_addr). */
#define SLJIT_REWRITABLE_JUMP		0x1000

/* Emit a jump instruction. The destination is not set, only the type of the jump.
    type must be between SLJIT_EQUAL and SLJIT_CALL3
    type can be combined (or'ed) with SLJIT_REWRITABLE_JUMP
   Flags: - (never set any flags) for both conditional and unconditional jumps.
   Flags: destroy all flags for calls. */
SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_jump(struct sljit_compiler *compiler, sljit_s32 type);

/* Basic arithmetic comparison. In most architectures it is implemented as
   an SLJIT_SUB operation (with SLJIT_UNUSED destination and setting
   appropriate flags) followed by a sljit_emit_jump. However some
   architectures (i.e: ARM64 or MIPS) may employ special optimizations here.
   It is suggested to use this comparison form when appropriate.
    type must be between SLJIT_EQUAL and SLJIT_I_SIG_LESS_EQUAL
    type can be combined (or'ed) with SLJIT_REWRITABLE_JUMP
   Flags: destroy flags. */
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

/* Call function or jump anywhere. Both direct and indirect form
    type must be between SLJIT_JUMP and SLJIT_CALL3
    Direct form: set src to SLJIT_IMM() and srcw to the address
    Indirect form: any other valid addressing mode
   Flags: - (never set any flags) for unconditional jumps.
   Flags: destroy all flags for calls. */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_ijump(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 src, sljit_sw srcw);

/* Perform the operation using the conditional flags as the second argument.
   Type must always be between SLJIT_EQUAL and SLJIT_S_ORDERED. The value
   represented by the type is 1, if the condition represented by the type
   is fulfilled, and 0 otherwise.

   If op == SLJIT_MOV, SLJIT_MOV_S32, SLJIT_MOV_U32:
     Set dst to the value represented by the type (0 or 1).
     Src must be SLJIT_UNUSED, and srcw must be 0
     Flags: - (never set any flags)
   If op == SLJIT_OR, op == SLJIT_AND, op == SLJIT_XOR
     Performs the binary operation using src as the first, and the value
     represented by type as the second argument.
     Important note: only dst=src and dstw=srcw is supported at the moment!
     Flags: I | E | K
   Note: sljit_emit_op_flags does nothing, if dst is SLJIT_UNUSED (regardless of op). */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_flags(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw,
	sljit_s32 type);

/* Copies the base address of SLJIT_SP + offset to dst.
   Flags: - (never set any flags) */
SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_local_base(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw offset);

/* The constant can be changed runtime (see: sljit_set_const)
   Flags: - (never set any flags) */
SLJIT_API_FUNC_ATTRIBUTE struct sljit_const* sljit_emit_const(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw init_value);

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
#define SLJIT_MINOR_VERSION	93

/* Get the human readable name of the platform. Can be useful on platforms
   like ARM, where ARM and Thumb2 functions can be mixed, and
   it is useful to know the type of the code generator. */
SLJIT_API_FUNC_ATTRIBUTE const char* sljit_get_platform_name(void);

/* Portable helper function to get an offset of a member. */
#define SLJIT_OFFSETOF(base, member) ((sljit_sw)(&((base*)0x10)->member) - 0x10)

#if (defined SLJIT_UTIL_GLOBAL_LOCK && SLJIT_UTIL_GLOBAL_LOCK)
/* This global lock is useful to compile common functions. */
SLJIT_API_FUNC_ATTRIBUTE void SLJIT_CALL sljit_grab_lock(void);
SLJIT_API_FUNC_ATTRIBUTE void SLJIT_CALL sljit_release_lock(void);
#endif

#if (defined SLJIT_UTIL_STACK && SLJIT_UTIL_STACK)

/* The sljit_stack is a utiliy feature of sljit, which allocates a
   writable memory region between base (inclusive) and limit (exclusive).
   Both base and limit is a pointer, and base is always <= than limit.
   This feature uses the "address space reserve" feature
   of modern operating systems. Basically we don't need to allocate a
   huge memory block in one step for the worst case, we can start with
   a smaller chunk and extend it later. Since the address space is
   reserved, the data never copied to other regions, thus it is safe
   to store pointers here. */

/* Note: The base field is aligned to PAGE_SIZE bytes (usually 4k or more).
   Note: stack growing should not happen in small steps: 4k, 16k or even
     bigger growth is better.
   Note: this structure may not be supported by all operating systems.
     Some kind of fallback mechanism is suggested when SLJIT_UTIL_STACK
     is not defined. */

struct sljit_stack {
	/* User data, anything can be stored here.
	   Starting with the same value as base. */
	sljit_uw top;
	/* These members are read only. */
	sljit_uw base;
	sljit_uw limit;
	sljit_uw max_limit;
};

/* Returns NULL if unsuccessful.
   Note: limit and max_limit contains the size for stack allocation.
   Note: the top field is initialized to base.
   Note: see sljit_create_compiler for the explanation of allocator_data. */
SLJIT_API_FUNC_ATTRIBUTE struct sljit_stack* SLJIT_CALL sljit_allocate_stack(sljit_uw limit, sljit_uw max_limit, void *allocator_data);
SLJIT_API_FUNC_ATTRIBUTE void SLJIT_CALL sljit_free_stack(struct sljit_stack *stack, void *allocator_data);

/* Can be used to increase (allocate) or decrease (free) the memory area.
   Returns with a non-zero value if unsuccessful. If new_limit is greater than
   max_limit, it will fail. It is very easy to implement a stack data structure,
   since the growth ratio can be added to the current limit, and sljit_stack_resize
   will do all the necessary checks. The fields of the stack are not changed if
   sljit_stack_resize fails. */
SLJIT_API_FUNC_ATTRIBUTE sljit_sw SLJIT_CALL sljit_stack_resize(struct sljit_stack *stack, sljit_uw new_limit);

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

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)

/* Returns with non-zero if sse2 is available. */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_x86_is_sse2_available(void);

/* Returns with non-zero if cmov instruction is available. */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_x86_is_cmov_available(void);

/* Emit a conditional mov instruction on x86 CPUs. This instruction
   moves src to destination, if the condition is satisfied. Unlike
   other arithmetic instructions, destination must be a register.
   Before such instructions are emitted, cmov support should be
   checked by sljit_x86_is_cmov_available function.
    type must be between SLJIT_EQUAL and SLJIT_S_ORDERED
    dst_reg must be a valid register and it can be combined
      with SLJIT_I32_OP to perform 32 bit arithmetic
   Flags: I - (never set any flags)
 */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_x86_emit_cmov(struct sljit_compiler *compiler,
	sljit_s32 type,
	sljit_s32 dst_reg,
	sljit_s32 src, sljit_sw srcw);

#endif

#endif /* _SLJIT_LIR_H_ */
