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

#include "sljitLir.h"

#ifdef _WIN32

#include <windows.h>

#endif /* _WIN32 */

#if !(defined SLJIT_STD_MACROS_DEFINED && SLJIT_STD_MACROS_DEFINED)

/* These libraries are needed for the macros below. */
#include <stdlib.h>
#include <string.h>

#endif /* SLJIT_STD_MACROS_DEFINED */

#define CHECK_ERROR() \
	do { \
		if (SLJIT_UNLIKELY(compiler->error)) \
			return compiler->error; \
	} while (0)

#define CHECK_ERROR_PTR() \
	do { \
		if (SLJIT_UNLIKELY(compiler->error)) \
			return NULL; \
	} while (0)

#define FAIL_IF(expr) \
	do { \
		if (SLJIT_UNLIKELY(expr)) \
			return compiler->error; \
	} while (0)

#define PTR_FAIL_IF(expr) \
	do { \
		if (SLJIT_UNLIKELY(expr)) \
			return NULL; \
	} while (0)

#define FAIL_IF_NULL(ptr) \
	do { \
		if (SLJIT_UNLIKELY(!(ptr))) { \
			compiler->error = SLJIT_ERR_ALLOC_FAILED; \
			return SLJIT_ERR_ALLOC_FAILED; \
		} \
	} while (0)

#define PTR_FAIL_IF_NULL(ptr) \
	do { \
		if (SLJIT_UNLIKELY(!(ptr))) { \
			compiler->error = SLJIT_ERR_ALLOC_FAILED; \
			return NULL; \
		} \
	} while (0)

#define PTR_FAIL_WITH_EXEC_IF(ptr) \
	do { \
		if (SLJIT_UNLIKELY(!(ptr))) { \
			compiler->error = SLJIT_ERR_EX_ALLOC_FAILED; \
			return NULL; \
		} \
	} while (0)

#if !(defined SLJIT_CONFIG_UNSUPPORTED && SLJIT_CONFIG_UNSUPPORTED)

#define SSIZE_OF(type) ((sljit_s32)sizeof(sljit_ ## type))

#define VARIABLE_FLAG_SHIFT (10)
/* All variable flags are even. */
#define VARIABLE_FLAG_MASK (0x3e << VARIABLE_FLAG_SHIFT)
#define GET_FLAG_TYPE(op) ((op) >> VARIABLE_FLAG_SHIFT)
#define GET_FLAG_TYPE_MASK(op) (((op) >> VARIABLE_FLAG_SHIFT) & 0x3e)

#define GET_OPCODE(op) \
	((op) & 0xff)

#define HAS_FLAGS(op) \
	((op) & (SLJIT_SET_Z | VARIABLE_FLAG_MASK))

#define GET_ALL_FLAGS(op) \
	((op) & (SLJIT_32 | SLJIT_SET_Z | VARIABLE_FLAG_MASK))

#if (defined SLJIT_64BIT_ARCHITECTURE && SLJIT_64BIT_ARCHITECTURE)
#define TYPE_CAST_NEEDED(op) \
	((op) >= SLJIT_MOV_U8 && (op) <= SLJIT_MOV_S32)
#else /* !SLJIT_64BIT_ARCHITECTURE */
#define TYPE_CAST_NEEDED(op) \
	((op) >= SLJIT_MOV_U8 && (op) <= SLJIT_MOV_S16)
#endif /* SLJIT_64BIT_ARCHITECTURE */

#define BUF_SIZE	4096

#if (defined SLJIT_32BIT_ARCHITECTURE && SLJIT_32BIT_ARCHITECTURE)
#define ABUF_SIZE	2048
#else /* !SLJIT_32BIT_ARCHITECTURE */
#define ABUF_SIZE	4096
#endif /* SLJIT_32BIT_ARCHITECTURE */

/* Parameter parsing. */
#define REG_MASK		0x7f
#define OFFS_REG(reg)		(((reg) >> 8) & REG_MASK)
#define OFFS_REG_MASK		(REG_MASK << 8)
#define TO_OFFS_REG(reg)	((reg) << 8)
#define FAST_IS_REG(reg)	((reg) < REG_MASK)

/* Mask for argument types. */
#define SLJIT_ARG_MASK		0x7
#define SLJIT_ARG_FULL_MASK	(SLJIT_ARG_MASK | SLJIT_ARG_TYPE_SCRATCH_REG)

/* Mask for register pairs. */
#define REG_PAIR_MASK		0x7f00
#define REG_PAIR_FIRST(reg)	((reg) & 0x7f)
#define REG_PAIR_SECOND(reg)	((reg) >> 8)

/* Mask for sljit_emit_enter. */
#define ENTER_GET_REGS(regs)			((regs) & 0xff)
#define ENTER_GET_FLOAT_REGS(regs)		(((regs) >> 8) & 0xff)
#define ENTER_GET_VECTOR_REGS(regs)		(((regs) >> 16) & 0xff)
#define SLJIT_KEPT_SAVEDS_COUNT(options)	((options) & 0x3)

/* Getters for simd operations, which returns with log2(size). */
#define SLJIT_SIMD_GET_OPCODE(type)		((type) & 0xff)
#define SLJIT_SIMD_GET_REG_SIZE(type)		(((type) >> 12) & 0x3f)
#define SLJIT_SIMD_GET_ELEM_SIZE(type)		(((type) >> 18) & 0x3f)
#define SLJIT_SIMD_GET_ELEM2_SIZE(type)		(((type) >> 24) & 0x3f)

#define SLJIT_SIMD_CHECK_REG(type) (((type) & 0x3f000) >= SLJIT_SIMD_REG_64 && ((type) & 0x3f000) <= SLJIT_SIMD_REG_512)
#define SLJIT_SIMD_TYPE_MASK(m) ((sljit_s32)0xff000fff & ~(SLJIT_SIMD_FLOAT | SLJIT_SIMD_TEST | (m)))
#define SLJIT_SIMD_TYPE_MASK2(m) ((sljit_s32)0xc0000fff & ~(SLJIT_SIMD_FLOAT | SLJIT_SIMD_TEST | (m)))

/* Jump flags. */
#define JUMP_ADDR	0x1
#define JUMP_MOV_ADDR	0x2
/* SLJIT_REWRITABLE_JUMP is 0x1000. */

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)
#	define PATCH_MB		0x04
#	define PATCH_MW		0x08
#if (defined SLJIT_CONFIG_X86_64 && SLJIT_CONFIG_X86_64)
#	define PATCH_MD		0x10
#	define MOV_ADDR_HI	0x20
#	define JUMP_MAX_SIZE	((sljit_uw)(10 + 3))
#	define CJUMP_MAX_SIZE	((sljit_uw)(2 + 10 + 3))
#else /* !SLJIT_CONFIG_X86_64 */
#	define JUMP_MAX_SIZE	((sljit_uw)5)
#	define CJUMP_MAX_SIZE	((sljit_uw)6)
#endif /* SLJIT_CONFIG_X86_64 */
#	define TYPE_SHIFT	13
#if (defined SLJIT_DEBUG && SLJIT_DEBUG)
/* Bits 7..12 is for debug jump size, SLJIT_REWRITABLE_JUMP is 0x1000 */
#	define JUMP_SIZE_SHIFT	7
#endif /* SLJIT_DEBUG */
#endif /* SLJIT_CONFIG_X86 */

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6) || (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7)
#	define IS_BL		0x04
#	define PATCH_B		0x08
#endif /* SLJIT_CONFIG_ARM_V6 || SLJIT_CONFIG_ARM_V7 */

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
#	define CPOOL_SIZE	512
#endif /* SLJIT_CONFIG_ARM_V6 */

#if (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7)
#	define JUMP_SIZE_SHIFT	26
#	define JUMP_MAX_SIZE	((sljit_uw)3)
#endif /* SLJIT_CONFIG_ARM_V7 */

#if (defined SLJIT_CONFIG_ARM_THUMB2 && SLJIT_CONFIG_ARM_THUMB2)
#	define IS_COND		0x04
#	define IS_BL		0x08
	/* conditional + imm8 */
#	define PATCH_TYPE1	0x10
	/* conditional + imm20 */
#	define PATCH_TYPE2	0x20
	/* imm11 */
#	define PATCH_TYPE3	0x30
	/* imm24 */
#	define PATCH_TYPE4	0x40
	/* BL + imm24 */
#	define PATCH_TYPE5	0x50
	/* addwi/subwi */
#	define PATCH_TYPE6	0x60
	/* 0xf00 cc code for branches */
#	define JUMP_SIZE_SHIFT	26
#	define JUMP_MAX_SIZE	((sljit_uw)5)
#endif /* SLJIT_CONFIG_ARM_THUMB2 */

#if (defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64)
#	define IS_COND		0x004
#	define IS_CBZ		0x008
#	define IS_BL		0x010
#	define PATCH_COND	0x020
#	define PATCH_B		0x040
#	define PATCH_B32	0x080
#	define PATCH_ABS48	0x100
#	define PATCH_ABS64	0x200
#	define JUMP_SIZE_SHIFT	58
#	define JUMP_MAX_SIZE	((sljit_uw)5)
#endif /* SLJIT_CONFIG_ARM_64 */

#if (defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC)
#	define IS_COND		0x004
#	define IS_CALL		0x008
#	define PATCH_B		0x010
#	define PATCH_ABS_B	0x020
#if (defined SLJIT_CONFIG_PPC_64 && SLJIT_CONFIG_PPC_64)
#	define PATCH_ABS32	0x040
#	define PATCH_ABS48	0x080
#	define JUMP_SIZE_SHIFT	58
#	define JUMP_MAX_SIZE	((sljit_uw)7)
#else /* !SLJIT_CONFIG_PPC_64 */
#	define JUMP_SIZE_SHIFT	26
#	define JUMP_MAX_SIZE	((sljit_uw)4)
#endif /* SLJIT_CONFIG_PPC_64 */
#endif /* SLJIT_CONFIG_PPC */

#if (defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS)
#	define IS_MOVABLE	0x004
#	define IS_JAL		0x008
#	define IS_CALL		0x010
#	define IS_BIT26_COND	0x020
#	define IS_BIT16_COND	0x040
#	define IS_BIT23_COND	0x080

#	define IS_COND		(IS_BIT26_COND | IS_BIT16_COND | IS_BIT23_COND)

#	define PATCH_B		0x100
#	define PATCH_J		0x200

#if (defined SLJIT_CONFIG_MIPS_64 && SLJIT_CONFIG_MIPS_64)
#	define PATCH_ABS32	0x400
#	define PATCH_ABS48	0x800
#endif /* SLJIT_CONFIG_MIPS_64 */

	/* instruction types */
#	define MOVABLE_INS	0
	/* 1 - 31 last destination register */
	/* no destination (i.e: store) */
#	define UNMOVABLE_INS	32
	/* FPU status register */
#	define FCSR_FCC		33
#endif /* SLJIT_CONFIG_MIPS */

#if (defined SLJIT_CONFIG_RISCV && SLJIT_CONFIG_RISCV)
#	define IS_COND		0x004
#	define IS_CALL		0x008

#	define PATCH_B		0x010
#	define PATCH_J		0x020

#if (defined SLJIT_CONFIG_RISCV_64 && SLJIT_CONFIG_RISCV_64)
#	define PATCH_REL32	0x040
#	define PATCH_ABS32	0x080
#	define PATCH_ABS44	0x100
#	define PATCH_ABS52	0x200
#	define JUMP_SIZE_SHIFT	58
#	define JUMP_MAX_SIZE	((sljit_uw)6)
#else /* !SLJIT_CONFIG_RISCV_64 */
#	define JUMP_SIZE_SHIFT	26
#	define JUMP_MAX_SIZE	((sljit_uw)2)
#endif /* SLJIT_CONFIG_RISCV_64 */
#endif /* SLJIT_CONFIG_RISCV */

#if (defined SLJIT_CONFIG_LOONGARCH && SLJIT_CONFIG_LOONGARCH)
#	define IS_COND		0x004
#	define IS_CALL		0x008

#	define PATCH_B		0x010
#	define PATCH_J		0x020

#	define PATCH_REL32	0x040
#	define PATCH_ABS32	0x080
#	define PATCH_ABS52	0x100
#	define JUMP_SIZE_SHIFT	58
#	define JUMP_MAX_SIZE	((sljit_uw)4)

#endif /* SLJIT_CONFIG_LOONGARCH */
/* Stack management. */

#define GET_SAVED_REGISTERS_SIZE(scratches, saveds, extra) \
	(((scratches < SLJIT_NUMBER_OF_SCRATCH_REGISTERS ? 0 : (scratches - SLJIT_NUMBER_OF_SCRATCH_REGISTERS)) + \
		(saveds) + (sljit_s32)(extra)) * (sljit_s32)sizeof(sljit_sw))

#define GET_SAVED_FLOAT_REGISTERS_SIZE(fscratches, fsaveds, type) \
	(((fscratches < SLJIT_NUMBER_OF_SCRATCH_FLOAT_REGISTERS ? 0 : (fscratches - SLJIT_NUMBER_OF_SCRATCH_FLOAT_REGISTERS)) + \
		(fsaveds)) * SSIZE_OF(type))

#define ADJUST_LOCAL_OFFSET(p, i) \
	if ((p) == (SLJIT_MEM1(SLJIT_SP))) \
		(i) += SLJIT_LOCALS_OFFSET;

#endif /* !(defined SLJIT_CONFIG_UNSUPPORTED && SLJIT_CONFIG_UNSUPPORTED) */

/* Utils can still be used even if SLJIT_CONFIG_UNSUPPORTED is set. */
#include "sljitUtils.c"

#if (defined SLJIT_CONFIG_ARM_THUMB2 && SLJIT_CONFIG_ARM_THUMB2)
#define SLJIT_CODE_TO_PTR(code) ((void*)((sljit_up)(code) & ~(sljit_up)0x1))
#elif (defined SLJIT_INDIRECT_CALL && SLJIT_INDIRECT_CALL)
#define SLJIT_CODE_TO_PTR(code) ((void*)(*(sljit_up*)code))
#else /* !SLJIT_CONFIG_ARM_THUMB2 && !SLJIT_INDIRECT_CALL */
#define SLJIT_CODE_TO_PTR(code) ((void*)(code))
#endif /* SLJIT_CONFIG_ARM_THUMB2 || SLJIT_INDIRECT_CALL */

#if !(defined SLJIT_CONFIG_UNSUPPORTED && SLJIT_CONFIG_UNSUPPORTED)

#if (defined SLJIT_EXECUTABLE_ALLOCATOR && SLJIT_EXECUTABLE_ALLOCATOR)

#if (defined SLJIT_PROT_EXECUTABLE_ALLOCATOR && SLJIT_PROT_EXECUTABLE_ALLOCATOR)

#if defined(__NetBSD__)
#include "allocator_src/sljitProtExecAllocatorNetBSD.c"
#else /* !__NetBSD__ */
#include "allocator_src/sljitProtExecAllocatorPosix.c"
#endif /* __NetBSD__ */

#elif (defined SLJIT_WX_EXECUTABLE_ALLOCATOR && SLJIT_WX_EXECUTABLE_ALLOCATOR)

#if defined(_WIN32)
#include "allocator_src/sljitWXExecAllocatorWindows.c"
#else /* !_WIN32 */
#include "allocator_src/sljitWXExecAllocatorPosix.c"
#endif /* _WIN32 */

#else /* !SLJIT_PROT_EXECUTABLAE_ALLOCATOR && !SLJIT_WX_EXECUTABLE_ALLOCATOR */

#if defined(_WIN32)
#include "allocator_src/sljitExecAllocatorWindows.c"
#elif defined(__APPLE__)
#include "allocator_src/sljitExecAllocatorApple.c"
#elif defined(__FreeBSD__)
#include "allocator_src/sljitExecAllocatorFreeBSD.c"
#else /* !_WIN32 && !__APPLE__ && !__FreeBSD__ */
#include "allocator_src/sljitExecAllocatorPosix.c"
#endif /* _WIN32 */

#endif /* SLJIT_PROT_EXECUTABLE_ALLOCATOR */

#else /* !SLJIT_EXECUTABLE_ALLOCATOR */

#ifndef SLJIT_UPDATE_WX_FLAGS
#define SLJIT_UPDATE_WX_FLAGS(from, to, enable_exec)
#endif /* SLJIT_UPDATE_WX_FLAGS */

#endif /* SLJIT_EXECUTABLE_ALLOCATOR */

#if (defined SLJIT_PROT_EXECUTABLE_ALLOCATOR && SLJIT_PROT_EXECUTABLE_ALLOCATOR)
#define SLJIT_ADD_EXEC_OFFSET(ptr, exec_offset) ((sljit_u8 *)(ptr) + (exec_offset))
#else /* !SLJIT_PROT_EXECUTABLE_ALLOCATOR */
#define SLJIT_ADD_EXEC_OFFSET(ptr, exec_offset) ((sljit_u8 *)(ptr))
#endif /* SLJIT_PROT_EXECUTABLE_ALLOCATOR */

/* Argument checking features. */

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)

/* Returns with error when an invalid argument is passed. */

#define CHECK_ARGUMENT(x) \
	do { \
		if (SLJIT_UNLIKELY(!(x))) \
			return 1; \
	} while (0)

#define CHECK_RETURN_TYPE sljit_s32
#define CHECK_RETURN_OK return 0

#define CHECK(x) \
	do { \
		if (SLJIT_UNLIKELY(x)) { \
			compiler->error = SLJIT_ERR_BAD_ARGUMENT; \
			return SLJIT_ERR_BAD_ARGUMENT; \
		} \
	} while (0)

#define CHECK_PTR(x) \
	do { \
		if (SLJIT_UNLIKELY(x)) { \
			compiler->error = SLJIT_ERR_BAD_ARGUMENT; \
			return NULL; \
		} \
	} while (0)

#define CHECK_REG_INDEX(x) \
	do { \
		if (SLJIT_UNLIKELY(x)) { \
			return -2; \
		} \
	} while (0)

#elif (defined SLJIT_DEBUG && SLJIT_DEBUG)

/* Assertion failure occures if an invalid argument is passed. */
#undef SLJIT_ARGUMENT_CHECKS
#define SLJIT_ARGUMENT_CHECKS 1

#define CHECK_ARGUMENT(x) SLJIT_ASSERT(x)
#define CHECK_RETURN_TYPE void
#define CHECK_RETURN_OK return
#define CHECK(x) x
#define CHECK_PTR(x) x
#define CHECK_REG_INDEX(x) x

#elif (defined SLJIT_VERBOSE && SLJIT_VERBOSE)

/* Arguments are not checked. */
#define CHECK_RETURN_TYPE void
#define CHECK_RETURN_OK return
#define CHECK(x) x
#define CHECK_PTR(x) x
#define CHECK_REG_INDEX(x) x

#else /* !SLJIT_ARGUMENT_CHECKS && !SLJIT_DEBUG && !SLJIT_VERBOSE */

/* Arguments are not checked. */
#define CHECK(x)
#define CHECK_PTR(x)
#define CHECK_REG_INDEX(x)

#endif /* SLJIT_ARGUMENT_CHECKS */

/* --------------------------------------------------------------------- */
/*  Public functions                                                     */
/* --------------------------------------------------------------------- */

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)
#define SLJIT_NEEDS_COMPILER_INIT 1
static sljit_s32 compiler_initialized = 0;
/* A thread safe initialization. */
static void init_compiler(void);
#endif /* SLJIT_CONFIG_X86 */

SLJIT_API_FUNC_ATTRIBUTE struct sljit_compiler* sljit_create_compiler(void *allocator_data)
{
	struct sljit_compiler *compiler = (struct sljit_compiler*)SLJIT_MALLOC(sizeof(struct sljit_compiler), allocator_data);
	if (!compiler)
		return NULL;
	SLJIT_ZEROMEM(compiler, sizeof(struct sljit_compiler));

	SLJIT_COMPILE_ASSERT(
		sizeof(sljit_s8) == 1 && sizeof(sljit_u8) == 1
		&& sizeof(sljit_s16) == 2 && sizeof(sljit_u16) == 2
		&& sizeof(sljit_s32) == 4 && sizeof(sljit_u32) == 4
		&& (sizeof(sljit_up) == 4 || sizeof(sljit_up) == 8)
		&& sizeof(sljit_up) <= sizeof(sljit_sw)
		&& sizeof(sljit_up) == sizeof(sljit_sp)
		&& (sizeof(sljit_sw) == 4 || sizeof(sljit_sw) == 8)
		&& (sizeof(sljit_uw) == sizeof(sljit_sw)),
		invalid_integer_types);
	SLJIT_COMPILE_ASSERT(SLJIT_REWRITABLE_JUMP != SLJIT_32,
		rewritable_jump_and_single_op_must_not_be_the_same);
	SLJIT_COMPILE_ASSERT(!(SLJIT_EQUAL & 0x1) && !(SLJIT_LESS & 0x1) && !(SLJIT_F_EQUAL & 0x1) && !(SLJIT_JUMP & 0x1),
		conditional_flags_must_be_even_numbers);

	/* Only the non-zero members must be set. */
	compiler->error = SLJIT_SUCCESS;

	compiler->allocator_data = allocator_data;
	compiler->buf = (struct sljit_memory_fragment*)SLJIT_MALLOC(BUF_SIZE, allocator_data);
	compiler->abuf = (struct sljit_memory_fragment*)SLJIT_MALLOC(ABUF_SIZE, allocator_data);

	if (!compiler->buf || !compiler->abuf) {
		if (compiler->buf)
			SLJIT_FREE(compiler->buf, allocator_data);
		if (compiler->abuf)
			SLJIT_FREE(compiler->abuf, allocator_data);
		SLJIT_FREE(compiler, allocator_data);
		return NULL;
	}

	compiler->buf->next = NULL;
	compiler->buf->used_size = 0;
	compiler->abuf->next = NULL;
	compiler->abuf->used_size = 0;

	compiler->scratches = -1;
	compiler->saveds = -1;
	compiler->fscratches = -1;
	compiler->fsaveds = -1;
#if (defined SLJIT_SEPARATE_VECTOR_REGISTERS && SLJIT_SEPARATE_VECTOR_REGISTERS) \
		|| (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS) \
		|| (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	compiler->vscratches = -1;
	compiler->vsaveds = -1;
#endif /* SLJIT_SEPARATE_VECTOR_REGISTERS || SLJIT_ARGUMENT_CHECKS || SLJIT_VERBOSE */
	compiler->local_size = -1;

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
	compiler->args_size = -1;
#endif /* SLJIT_CONFIG_X86_32 */

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	compiler->cpool = (sljit_uw*)SLJIT_MALLOC(CPOOL_SIZE * sizeof(sljit_uw)
		+ CPOOL_SIZE * sizeof(sljit_u8), allocator_data);
	if (!compiler->cpool) {
		SLJIT_FREE(compiler->buf, allocator_data);
		SLJIT_FREE(compiler->abuf, allocator_data);
		SLJIT_FREE(compiler, allocator_data);
		return NULL;
	}
	compiler->cpool_unique = (sljit_u8*)(compiler->cpool + CPOOL_SIZE);
	compiler->cpool_diff = 0xffffffff;
#endif /* SLJIT_CONFIG_ARM_V6 */

#if (defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS)
	compiler->delay_slot = UNMOVABLE_INS;
#endif /* SLJIT_CONFIG_MIPS */

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS) \
		|| (defined SLJIT_DEBUG && SLJIT_DEBUG)
	SLJIT_ASSERT(compiler->last_flags == 0 && compiler->logical_local_size == 0);
	compiler->last_return = -1;
#endif /* SLJIT_ARGUMENT_CHECKS || SLJIT_DEBUG */

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS) \
		|| (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
#if !(defined SLJIT_SEPARATE_VECTOR_REGISTERS && SLJIT_SEPARATE_VECTOR_REGISTERS)
	compiler->real_fscratches = -1;
	compiler->real_fsaveds = -1;
#endif /* !SLJIT_SEPARATE_VECTOR_REGISTERS */
	SLJIT_ASSERT(compiler->skip_checks == 0);
#endif /* SLJIT_ARGUMENT_CHECKS || SLJIT_VERBOSE */

#if (defined SLJIT_NEEDS_COMPILER_INIT && SLJIT_NEEDS_COMPILER_INIT)
	if (!compiler_initialized) {
		init_compiler();
		compiler_initialized = 1;
	}
#endif /* SLJIT_NEEDS_COMPILER_INIT */

	return compiler;
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_free_compiler(struct sljit_compiler *compiler)
{
	struct sljit_memory_fragment *buf;
	struct sljit_memory_fragment *curr;
	void *allocator_data = compiler->allocator_data;
	SLJIT_UNUSED_ARG(allocator_data);

	buf = compiler->buf;
	while (buf) {
		curr = buf;
		buf = buf->next;
		SLJIT_FREE(curr, allocator_data);
	}

	buf = compiler->abuf;
	while (buf) {
		curr = buf;
		buf = buf->next;
		SLJIT_FREE(curr, allocator_data);
	}

#if (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
	SLJIT_FREE(compiler->cpool, allocator_data);
#endif /* SLJIT_CONFIG_ARM_V6 */
	SLJIT_FREE(compiler, allocator_data);
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_compiler_memory_error(struct sljit_compiler *compiler)
{
	if (compiler->error == SLJIT_SUCCESS)
		compiler->error = SLJIT_ERR_ALLOC_FAILED;
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_free_code(void* code, void *exec_allocator_data)
{
	SLJIT_UNUSED_ARG(exec_allocator_data);

	SLJIT_FREE_EXEC(SLJIT_CODE_TO_PTR(code), exec_allocator_data);
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_label(struct sljit_jump *jump, struct sljit_label* label)
{
	if (SLJIT_LIKELY(!!jump) && SLJIT_LIKELY(!!label)) {
		jump->flags &= (sljit_uw)~JUMP_ADDR;
		jump->u.label = label;
	}
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_target(struct sljit_jump *jump, sljit_uw target)
{
	if (SLJIT_LIKELY(!!jump)) {
		jump->flags |= JUMP_ADDR;
		jump->u.target = target;
	}
}

#define SLJIT_CURRENT_FLAGS_ALL \
	(SLJIT_CURRENT_FLAGS_32 | SLJIT_CURRENT_FLAGS_ADD | SLJIT_CURRENT_FLAGS_SUB | SLJIT_CURRENT_FLAGS_COMPARE)

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_current_flags(struct sljit_compiler *compiler, sljit_s32 current_flags)
{
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(current_flags);

#if (defined SLJIT_HAS_STATUS_FLAGS_STATE && SLJIT_HAS_STATUS_FLAGS_STATE)
	compiler->status_flags_state = current_flags;
#endif /* SLJIT_HAS_STATUS_FLAGS_STATE */

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	compiler->last_flags = 0;
	if ((current_flags & ~(VARIABLE_FLAG_MASK | SLJIT_SET_Z | SLJIT_CURRENT_FLAGS_ALL)) == 0) {
		compiler->last_flags = GET_FLAG_TYPE(current_flags) | (current_flags & (SLJIT_32 | SLJIT_SET_Z));
	}
#endif /* SLJIT_ARGUMENT_CHECKS */
}

/* --------------------------------------------------------------------- */
/*  Private functions                                                    */
/* --------------------------------------------------------------------- */

static void* ensure_buf(struct sljit_compiler *compiler, sljit_uw size)
{
	sljit_u8 *ret;
	struct sljit_memory_fragment *new_frag;

	SLJIT_ASSERT(size <= 256);
	if (compiler->buf->used_size + size <= (BUF_SIZE - (sljit_uw)SLJIT_OFFSETOF(struct sljit_memory_fragment, memory))) {
		ret = compiler->buf->memory + compiler->buf->used_size;
		compiler->buf->used_size += size;
		return ret;
	}
	new_frag = (struct sljit_memory_fragment*)SLJIT_MALLOC(BUF_SIZE, compiler->allocator_data);
	PTR_FAIL_IF_NULL(new_frag);
	new_frag->next = compiler->buf;
	compiler->buf = new_frag;
	new_frag->used_size = size;
	return new_frag->memory;
}

static void* ensure_abuf(struct sljit_compiler *compiler, sljit_uw size)
{
	sljit_u8 *ret;
	struct sljit_memory_fragment *new_frag;

	SLJIT_ASSERT(size <= 256);
	if (compiler->abuf->used_size + size <= (ABUF_SIZE - (sljit_uw)SLJIT_OFFSETOF(struct sljit_memory_fragment, memory))) {
		ret = compiler->abuf->memory + compiler->abuf->used_size;
		compiler->abuf->used_size += size;
		return ret;
	}
	new_frag = (struct sljit_memory_fragment*)SLJIT_MALLOC(ABUF_SIZE, compiler->allocator_data);
	PTR_FAIL_IF_NULL(new_frag);
	new_frag->next = compiler->abuf;
	compiler->abuf = new_frag;
	new_frag->used_size = size;
	return new_frag->memory;
}

SLJIT_API_FUNC_ATTRIBUTE void* sljit_alloc_memory(struct sljit_compiler *compiler, sljit_s32 size)
{
	CHECK_ERROR_PTR();

#if (defined SLJIT_64BIT_ARCHITECTURE && SLJIT_64BIT_ARCHITECTURE)
	if (size <= 0 || size > 128)
		return NULL;
	size = (size + 7) & ~7;
#else /* !SLJIT_64BIT_ARCHITECTURE */
	if (size <= 0 || size > 64)
		return NULL;
	size = (size + 3) & ~3;
#endif /* SLJIT_64BIT_ARCHITECTURE */
	return ensure_abuf(compiler, (sljit_uw)size);
}

static SLJIT_INLINE void reverse_buf(struct sljit_compiler *compiler)
{
	struct sljit_memory_fragment *buf = compiler->buf;
	struct sljit_memory_fragment *prev = NULL;
	struct sljit_memory_fragment *tmp;

	do {
		tmp = buf->next;
		buf->next = prev;
		prev = buf;
		buf = tmp;
	} while (buf != NULL);

	compiler->buf = prev;
}

static SLJIT_INLINE void* allocate_executable_memory(sljit_uw size, sljit_s32 options,
	void *exec_allocator_data, sljit_sw *executable_offset)
{
	void *code;
	struct sljit_generate_code_buffer *buffer;

	if (SLJIT_LIKELY(!(options & SLJIT_GENERATE_CODE_BUFFER))) {
		code = SLJIT_MALLOC_EXEC(size, exec_allocator_data);
		*executable_offset = SLJIT_EXEC_OFFSET(code);
		return code;
	}

	buffer = (struct sljit_generate_code_buffer*)exec_allocator_data;

	if (size <= buffer->size) {
		*executable_offset = buffer->executable_offset;
		return buffer->buffer;
	}

	return NULL;
}

#define SLJIT_MAX_ADDRESS ~(sljit_uw)0

#define SLJIT_GET_NEXT_SIZE(ptr) (ptr != NULL) ? ((ptr)->size) : SLJIT_MAX_ADDRESS
#define SLJIT_GET_NEXT_ADDRESS(ptr) (ptr != NULL) ? ((ptr)->addr) : SLJIT_MAX_ADDRESS

#if !(defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)

#define SLJIT_NEXT_DEFINE_TYPES \
	sljit_uw next_label_size; \
	sljit_uw next_jump_addr; \
	sljit_uw next_const_addr; \
	sljit_uw next_min_addr

#define SLJIT_NEXT_INIT_TYPES() \
	next_label_size = SLJIT_GET_NEXT_SIZE(label); \
	next_jump_addr = SLJIT_GET_NEXT_ADDRESS(jump); \
	next_const_addr = SLJIT_GET_NEXT_ADDRESS(const_);

#define SLJIT_GET_NEXT_MIN() \
	next_min_addr = sljit_get_next_min(next_label_size, next_jump_addr, next_const_addr);

static SLJIT_INLINE sljit_uw sljit_get_next_min(sljit_uw next_label_size,
	sljit_uw next_jump_addr, sljit_uw next_const_addr)
{
	sljit_uw result = next_jump_addr;

	SLJIT_ASSERT(result == SLJIT_MAX_ADDRESS || result != next_const_addr);

	if (next_const_addr < result)
		result = next_const_addr;

	if (next_label_size < result)
		result = next_label_size;

	return result;
}

#endif /* !SLJIT_CONFIG_X86 */

#if !(defined SLJIT_SEPARATE_VECTOR_REGISTERS && SLJIT_SEPARATE_VECTOR_REGISTERS)

static void update_float_register_count(struct sljit_compiler *compiler, sljit_s32 scratches, sljit_s32 saveds)
{
	sljit_s32 vscratches = ENTER_GET_VECTOR_REGS(scratches);
	sljit_s32 vsaveds = ENTER_GET_VECTOR_REGS(saveds);

	if (compiler->fscratches < vscratches)
		compiler->fscratches = vscratches;

	if (compiler->fsaveds < vsaveds)
		compiler->fsaveds = vsaveds;

	if (compiler->fsaveds + compiler->fscratches > SLJIT_NUMBER_OF_FLOAT_REGISTERS)
		compiler->fscratches = SLJIT_NUMBER_OF_FLOAT_REGISTERS - compiler->fsaveds;
}

#endif /* !SLJIT_SEPARATE_VECTOR_REGISTERS */

static SLJIT_INLINE void set_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 args,
	sljit_s32 scratches, sljit_s32 saveds, sljit_s32 local_size)
{
	SLJIT_UNUSED_ARG(args);
	SLJIT_UNUSED_ARG(local_size);

	compiler->options = options;
	compiler->scratches = ENTER_GET_REGS(scratches);
	compiler->saveds = ENTER_GET_REGS(saveds);
	/* These members may be copied to real_* members below. */
	compiler->fscratches = ENTER_GET_FLOAT_REGS(scratches);
	compiler->fsaveds = ENTER_GET_FLOAT_REGS(saveds);
#if (defined SLJIT_SEPARATE_VECTOR_REGISTERS && SLJIT_SEPARATE_VECTOR_REGISTERS)
	compiler->vscratches = ENTER_GET_VECTOR_REGS(scratches);
	compiler->vsaveds = ENTER_GET_VECTOR_REGS(saveds);
#else /* !SLJIT_SEPARATE_VECTOR_REGISTERS */
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS) \
		|| (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	compiler->real_fscratches = compiler->fscratches;
	compiler->real_fsaveds = compiler->fsaveds;
	compiler->vscratches = ENTER_GET_VECTOR_REGS(scratches);
	compiler->vsaveds = ENTER_GET_VECTOR_REGS(saveds);
#endif /* SLJIT_ARGUMENT_CHECKS || SLJIT_VERBOSE */
	update_float_register_count(compiler, scratches, saveds);
#endif /* SLJIT_SEPARATE_VECTOR_REGISTERS */
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	compiler->last_return = args & SLJIT_ARG_MASK;
	compiler->logical_local_size = local_size;
#endif /* SLJIT_ARGUMENT_CHECKS */
}

static SLJIT_INLINE void set_label(struct sljit_label *label, struct sljit_compiler *compiler)
{
	label->next = NULL;
	label->u.index = compiler->label_count++;
	label->size = compiler->size;
	if (compiler->last_label != NULL)
		compiler->last_label->next = label;
	else
		compiler->labels = label;
	compiler->last_label = label;
}

static SLJIT_INLINE void set_jump(struct sljit_jump *jump, struct sljit_compiler *compiler, sljit_u32 flags)
{
	jump->next = NULL;
	jump->flags = flags;
	jump->u.label = NULL;
	if (compiler->last_jump != NULL)
		compiler->last_jump->next = jump;
	else
		compiler->jumps = jump;
	compiler->last_jump = jump;
}

static SLJIT_INLINE void set_mov_addr(struct sljit_jump *jump, struct sljit_compiler *compiler, sljit_uw offset)
{
	jump->next = NULL;
	jump->addr = compiler->size - offset;
	jump->flags = JUMP_MOV_ADDR;
	jump->u.label = NULL;
	if (compiler->last_jump != NULL)
		compiler->last_jump->next = jump;
	else
		compiler->jumps = jump;
	compiler->last_jump = jump;
}

static SLJIT_INLINE void set_const(struct sljit_const *const_, struct sljit_compiler *compiler)
{
	const_->next = NULL;
	const_->addr = compiler->size;
	if (compiler->last_const != NULL)
		compiler->last_const->next = const_;
	else
		compiler->consts = const_;
	compiler->last_const = const_;
}

#define ADDRESSING_DEPENDS_ON(exp, reg) \
	(((exp) & SLJIT_MEM) && (((exp) & REG_MASK) == reg || OFFS_REG(exp) == reg))

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)

static sljit_s32 function_check_arguments(sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds, sljit_s32 fscratches)
{
	sljit_s32 word_arg_count, scratch_arg_end, saved_arg_count, float_arg_count, curr_type;

	curr_type = (arg_types & SLJIT_ARG_FULL_MASK);

	if (curr_type >= SLJIT_ARG_TYPE_F64) {
		if (curr_type > SLJIT_ARG_TYPE_F32 || fscratches == 0)
			return 0;
	} else if (curr_type >= SLJIT_ARG_TYPE_W) {
		if (scratches == 0)
			return 0;
	}

	arg_types >>= SLJIT_ARG_SHIFT;

	word_arg_count = 0;
	scratch_arg_end = 0;
	saved_arg_count = 0;
	float_arg_count = 0;
	while (arg_types != 0) {
		if (word_arg_count + float_arg_count >= 4)
			return 0;

		curr_type = (arg_types & SLJIT_ARG_MASK);

		if (arg_types & SLJIT_ARG_TYPE_SCRATCH_REG) {
			if (saveds == -1 || curr_type < SLJIT_ARG_TYPE_W || curr_type > SLJIT_ARG_TYPE_P)
				return 0;

			word_arg_count++;
			scratch_arg_end = word_arg_count;
		} else {
			if (curr_type < SLJIT_ARG_TYPE_W || curr_type > SLJIT_ARG_TYPE_F32)
				return 0;

			if (curr_type < SLJIT_ARG_TYPE_F64) {
				word_arg_count++;
				saved_arg_count++;
			} else
				float_arg_count++;
		}

		arg_types >>= SLJIT_ARG_SHIFT;
	}

	if (saveds == -1)
		return (word_arg_count <= scratches && float_arg_count <= fscratches);

	return (saved_arg_count <= saveds && scratch_arg_end <= scratches && float_arg_count <= fscratches);
}

#define FUNCTION_CHECK_IS_REG(r) \
	(((r) >= SLJIT_R0 && (r) < (SLJIT_R0 + compiler->scratches)) \
	|| ((r) > (SLJIT_S0 - compiler->saveds) && (r) <= SLJIT_S0) \
	|| ((r) >= SLJIT_TMP_REGISTER_BASE && (r) < (SLJIT_TMP_REGISTER_BASE + SLJIT_NUMBER_OF_TEMPORARY_REGISTERS)))

#if (defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32)
#define CHECK_IF_VIRTUAL_REGISTER(p) ((p) <= SLJIT_S3 && (p) >= SLJIT_S8)
#else /* !SLJIT_CONFIG_X86_32 */
#define CHECK_IF_VIRTUAL_REGISTER(p) 0
#endif /* SLJIT_CONFIG_X86_32 */

static sljit_s32 function_check_src_mem(struct sljit_compiler *compiler, sljit_s32 p, sljit_sw i)
{
	if (compiler->scratches == -1)
		return 0;

	if (!(p & SLJIT_MEM))
		return 0;

	if (p == SLJIT_MEM1(SLJIT_SP))
		return (i >= 0 && i < compiler->logical_local_size);

	if (!(!(p & REG_MASK) || FUNCTION_CHECK_IS_REG(p & REG_MASK)))
		return 0;

	if (CHECK_IF_VIRTUAL_REGISTER(p & REG_MASK))
		return 0;

	if (p & OFFS_REG_MASK) {
		if (!(p & REG_MASK))
			return 0;

		if (!(FUNCTION_CHECK_IS_REG(OFFS_REG(p))))
			return 0;

		if (CHECK_IF_VIRTUAL_REGISTER(OFFS_REG(p)))
			return 0;

		if ((i & ~0x3) != 0)
			return 0;
	}

	return (p & ~(SLJIT_MEM | REG_MASK | OFFS_REG_MASK)) == 0;
}

#define FUNCTION_CHECK_SRC_MEM(p, i) \
	CHECK_ARGUMENT(function_check_src_mem(compiler, p, i));

static sljit_s32 function_check_src(struct sljit_compiler *compiler, sljit_s32 p, sljit_sw i)
{
	if (compiler->scratches == -1)
		return 0;

	if (FUNCTION_CHECK_IS_REG(p))
		return (i == 0);

	if (p == SLJIT_IMM)
		return 1;

	return function_check_src_mem(compiler, p, i);
}

#define FUNCTION_CHECK_SRC(p, i) \
	CHECK_ARGUMENT(function_check_src(compiler, p, i));

static sljit_s32 function_check_dst(struct sljit_compiler *compiler, sljit_s32 p, sljit_sw i)
{
	if (compiler->scratches == -1)
		return 0;

	if (FUNCTION_CHECK_IS_REG(p))
		return (i == 0);

	return function_check_src_mem(compiler, p, i);
}

#define FUNCTION_CHECK_DST(p, i) \
	CHECK_ARGUMENT(function_check_dst(compiler, p, i));

#if (defined SLJIT_CONFIG_ARM_32 && SLJIT_CONFIG_ARM_32) \
	|| (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)

#define FUNCTION_CHECK_IS_FREG(fr, is_32) \
	function_check_is_freg(compiler, (fr), (is_32))

static sljit_s32 function_check_is_freg(struct sljit_compiler *compiler, sljit_s32 fr, sljit_s32 is_32);

#define FUNCTION_FCHECK(p, i, is_32) \
	CHECK_ARGUMENT(function_fcheck(compiler, (p), (i), (is_32)));

static sljit_s32 function_fcheck(struct sljit_compiler *compiler, sljit_s32 p, sljit_sw i, sljit_s32 is_32)
{
	if (compiler->scratches == -1)
		return 0;

	if (FUNCTION_CHECK_IS_FREG(p, is_32))
		return (i == 0);

	return function_check_src_mem(compiler, p, i);
}

static sljit_s32 function_check_is_vreg(struct sljit_compiler *compiler, sljit_s32 vr, sljit_s32 type);

#define FUNCTION_CHECK_IS_VREG(vr, type) \
	function_check_is_vreg(compiler, (vr), (type))

#define FUNCTION_VCHECK(p, i, type) \
	CHECK_ARGUMENT(function_vcheck(compiler, (p), (i), (type)))

static sljit_s32 function_vcheck(struct sljit_compiler *compiler, sljit_s32 p, sljit_sw i, sljit_s32 type)
{
	if (compiler->scratches == -1)
		return 0;

	if (FUNCTION_CHECK_IS_VREG(p, type))
		return (i == 0);

	return function_check_src_mem(compiler, p, i);
}

#else /* !SLJIT_CONFIG_ARM_32 && !SLJIT_CONFIG_MIPS_32 */

#define FUNCTION_CHECK_IS_FREG(fr, is_32) \
	function_check_is_freg(compiler, (fr))

static sljit_s32 function_check_is_freg(struct sljit_compiler *compiler, sljit_s32 fr)
{
	sljit_s32 fscratches, fsaveds;

	if (compiler->scratches == -1)
		return 0;

#if (defined SLJIT_SEPARATE_VECTOR_REGISTERS && SLJIT_SEPARATE_VECTOR_REGISTERS)
	fscratches = compiler->fscratches;
	fsaveds = compiler->fsaveds;
#else /* SLJIT_SEPARATE_VECTOR_REGISTERS */
	fscratches = compiler->real_fscratches;
	fsaveds = compiler->real_fsaveds;
#endif /* !SLJIT_SEPARATE_VECTOR_REGISTERS */

	return (fr >= SLJIT_FR0 && fr < (SLJIT_FR0 + fscratches))
		|| (fr > (SLJIT_FS0 - fsaveds) && fr <= SLJIT_FS0)
		|| (fr >= SLJIT_TMP_FREGISTER_BASE && fr < (SLJIT_TMP_FREGISTER_BASE + SLJIT_NUMBER_OF_TEMPORARY_FLOAT_REGISTERS));
}

#define FUNCTION_FCHECK(p, i, is_32) \
	CHECK_ARGUMENT(function_fcheck(compiler, (p), (i)));

static sljit_s32 function_fcheck(struct sljit_compiler *compiler, sljit_s32 p, sljit_sw i)
{
	if (compiler->scratches == -1)
		return 0;

	if (function_check_is_freg(compiler, p))
		return (i == 0);

	return function_check_src_mem(compiler, p, i);
}

#define FUNCTION_CHECK_IS_VREG(vr, type) \
	function_check_is_vreg(compiler, (vr))

static sljit_s32 function_check_is_vreg(struct sljit_compiler *compiler, sljit_s32 vr)
{
	if (compiler->scratches == -1)
		return 0;

	return (vr >= SLJIT_VR0 && vr < (SLJIT_VR0 + compiler->vscratches))
		|| (vr > (SLJIT_VS0 - compiler->vsaveds) && vr <= SLJIT_VS0)
		|| (vr >= SLJIT_TMP_VREGISTER_BASE && vr < (SLJIT_TMP_VREGISTER_BASE + SLJIT_NUMBER_OF_TEMPORARY_VECTOR_REGISTERS));
}

#define FUNCTION_VCHECK(p, i, type) \
	CHECK_ARGUMENT(function_vcheck(compiler, (p), (i)))

static sljit_s32 function_vcheck(struct sljit_compiler *compiler, sljit_s32 p, sljit_sw i)
{
	if (compiler->scratches == -1)
		return 0;

	if (function_check_is_vreg(compiler, p))
		return (i == 0);

	return function_check_src_mem(compiler, p, i);
}

#endif /* SLJIT_CONFIG_ARM_32 || SLJIT_CONFIG_MIPS_32 */

#endif /* SLJIT_ARGUMENT_CHECKS */

#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)

SLJIT_API_FUNC_ATTRIBUTE void sljit_compiler_verbose(struct sljit_compiler *compiler, FILE* verbose)
{
	compiler->verbose = verbose;
}

#if (defined SLJIT_64BIT_ARCHITECTURE && SLJIT_64BIT_ARCHITECTURE)
#ifdef _WIN64
#ifdef __GNUC__
#	define SLJIT_PRINT_D	"ll"
#else /* !__GNUC__ */
#	define SLJIT_PRINT_D	"I64"
#endif  /* __GNUC__ */
#else /* !_WIN64 */
#	define SLJIT_PRINT_D	"l"
#endif /* _WIN64 */
#else /* !SLJIT_64BIT_ARCHITECTURE */
#	define SLJIT_PRINT_D	""
#endif /* SLJIT_64BIT_ARCHITECTURE */

static void sljit_verbose_reg(struct sljit_compiler *compiler, sljit_s32 r)
{
	if (r < (SLJIT_R0 + compiler->scratches))
		fprintf(compiler->verbose, "r%d", r - SLJIT_R0);
	else if (r < SLJIT_SP)
		fprintf(compiler->verbose, "s%d", SLJIT_NUMBER_OF_REGISTERS - r);
	else if (r == SLJIT_SP)
		fprintf(compiler->verbose, "sp");
	else
		fprintf(compiler->verbose, "t%d", r - SLJIT_TMP_REGISTER_BASE);
}

static void sljit_verbose_freg(struct sljit_compiler *compiler, sljit_s32 r)
{
#if (defined SLJIT_CONFIG_ARM_32 && SLJIT_CONFIG_ARM_32) \
		|| (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	if (r >= SLJIT_F64_SECOND(SLJIT_FR0)) {
		fprintf(compiler->verbose, "^");
		r -= SLJIT_F64_SECOND(0);
	}
#endif /* SLJIT_CONFIG_ARM_32 || SLJIT_CONFIG_MIPS_32 */

	if (r < (SLJIT_FR0 + compiler->fscratches))
		fprintf(compiler->verbose, "fr%d", r - SLJIT_FR0);
	else if (r < SLJIT_TMP_FREGISTER_BASE)
		fprintf(compiler->verbose, "fs%d", SLJIT_NUMBER_OF_FLOAT_REGISTERS - r);
	else
		fprintf(compiler->verbose, "ft%d", r - SLJIT_TMP_FREGISTER_BASE);
}

static void sljit_verbose_vreg(struct sljit_compiler *compiler, sljit_s32 r)
{
#if (defined SLJIT_CONFIG_ARM_32 && SLJIT_CONFIG_ARM_32) \
		|| (defined SLJIT_CONFIG_MIPS_32 && SLJIT_CONFIG_MIPS_32)
	if (r >= SLJIT_F64_SECOND(SLJIT_VR0)) {
		fprintf(compiler->verbose, "^");
		r -= SLJIT_F64_SECOND(0);
	}
#endif /* SLJIT_CONFIG_ARM_32 || SLJIT_CONFIG_MIPS_32 */

	if (r < (SLJIT_VR0 + compiler->vscratches))
		fprintf(compiler->verbose, "vr%d", r - SLJIT_VR0);
	else if (r < SLJIT_TMP_VREGISTER_BASE)
		fprintf(compiler->verbose, "vs%d", SLJIT_NUMBER_OF_VECTOR_REGISTERS - r);
	else
		fprintf(compiler->verbose, "vt%d", r - SLJIT_TMP_VREGISTER_BASE);
}

static void sljit_verbose_mem(struct sljit_compiler *compiler, sljit_s32 p, sljit_sw i)
{
	if (!(p & REG_MASK)) {
		fprintf(compiler->verbose, "[%" SLJIT_PRINT_D "d]", i);
		return;
	}

	fputc('[', compiler->verbose);
	sljit_verbose_reg(compiler, (p) & REG_MASK);
	if (p & OFFS_REG_MASK) {
		fprintf(compiler->verbose, " + ");
		sljit_verbose_reg(compiler, OFFS_REG(p));
		if (i)
			fprintf(compiler->verbose, " * %d", 1 << (i));
	} else if (i)
		fprintf(compiler->verbose, " + %" SLJIT_PRINT_D "d", (i));
	fputc(']', compiler->verbose);
}

static void sljit_verbose_param(struct sljit_compiler *compiler, sljit_s32 p, sljit_sw i)
{
	if (p == SLJIT_IMM)
		fprintf(compiler->verbose, "#%" SLJIT_PRINT_D "d", i);
	else if (p & SLJIT_MEM)
		sljit_verbose_mem(compiler, p, i);
	else
		sljit_verbose_reg(compiler, p);
}

static void sljit_verbose_fparam(struct sljit_compiler *compiler, sljit_s32 p, sljit_sw i)
{
	if (p & SLJIT_MEM)
		sljit_verbose_mem(compiler, p, i);
	else
		sljit_verbose_freg(compiler, p);
}

static void sljit_verbose_vparam(struct sljit_compiler *compiler, sljit_s32 p, sljit_sw i)
{
	if (p & SLJIT_MEM)
		sljit_verbose_mem(compiler, p, i);
	else
		sljit_verbose_vreg(compiler, p);
}

static const char* op0_names[] = {
	"breakpoint", "nop", "lmul.uw", "lmul.sw",
	"divmod.u", "divmod.s", "div.u", "div.s",
	"memory_barrier", "endbr", "skip_frames_before_return"
};

static const char* op1_names[] = {
	"mov", "mov", "mov", "mov",
	"mov", "mov", "mov", "mov",
	"mov", "clz", "ctz", "rev",
	"rev", "rev", "rev", "rev"
};

static const char* op1_types[] = {
	"", ".u8", ".s8", ".u16",
	".s16", ".u32", ".s32", "32",
	".p", "", "", "",
	".u16", ".s16", ".u32", ".s32"
};

static const char* op2_names[] = {
	"add", "addc", "sub", "subc",
	"mul", "and", "or", "xor",
	"shl", "mshl", "lshr", "mlshr",
	"ashr", "mashr", "rotl", "rotr"
};

static const char* op2r_names[] = {
	"muladd"
};

static const char* op_src_dst_names[] = {
	"fast_return", "skip_frames_before_fast_return",
	"prefetch_l1", "prefetch_l2",
	"prefetch_l3", "prefetch_once",
	"fast_enter", "get_return_address"
};

static const char* fop1_names[] = {
	"mov", "conv", "conv", "conv",
	"conv", "conv", "conv", "conv",
	"cmp", "neg", "abs",
};

static const char* fop1_conv_types[] = {
	"sw", "s32", "sw", "s32",
	"uw", "u32"
};

static const char* fop2_names[] = {
	"add", "sub", "mul", "div"
};

static const char* fop2r_names[] = {
	"copysign"
};

static const char* simd_op2_names[] = {
	"and", "or", "xor", "shuffle"
};

static const char* jump_names[] = {
	"equal", "not_equal",
	"less", "greater_equal",
	"greater", "less_equal",
	"sig_less", "sig_greater_equal",
	"sig_greater", "sig_less_equal",
	"overflow", "not_overflow",
	"carry", "not_carry",
	"atomic_stored", "atomic_not_stored",
	"f_equal", "f_not_equal",
	"f_less", "f_greater_equal",
	"f_greater", "f_less_equal",
	"unordered", "ordered",
	"ordered_equal", "unordered_or_not_equal",
	"ordered_less", "unordered_or_greater_equal",
	"ordered_greater", "unordered_or_less_equal",
	"unordered_or_equal", "ordered_not_equal",
	"unordered_or_less", "ordered_greater_equal",
	"unordered_or_greater", "ordered_less_equal",
	"jump", "fast_call",
	"call", "call_reg_arg"
};

static const char* call_arg_names[] = {
	"void", "w", "32", "p", "f64", "f32"
};

#endif /* SLJIT_VERBOSE */

/* --------------------------------------------------------------------- */
/*  Arch dependent                                                       */
/* --------------------------------------------------------------------- */

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS) \
	|| (defined SLJIT_VERBOSE && SLJIT_VERBOSE)

#define SLJIT_SKIP_CHECKS(compiler) (compiler)->skip_checks = 1
#define SLJIT_CHECK_OPCODE(op, flags) ((op) & ~(SLJIT_32 | SLJIT_SET_Z | VARIABLE_FLAG_MASK | (flags)))

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_generate_code(struct sljit_compiler *compiler)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	struct sljit_jump *jump;
#endif /* SLJIT_ARGUMENT_CHECKS */

	SLJIT_UNUSED_ARG(compiler);

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(compiler->size > 0);
	jump = compiler->jumps;
	while (jump) {
		/* All jumps have target. */
		CHECK_ARGUMENT((jump->flags & JUMP_ADDR) || jump->u.label != NULL);
		jump = jump->next;
	}
#endif /* SLJIT_ARGUMENT_CHECKS */
	CHECK_RETURN_OK;
}

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)
#define SLJIT_ENTER_CPU_SPECIFIC_OPTIONS (SLJIT_ENTER_USE_VEX)
#else /* !SLJIT_CONFIG_X86 */
#define SLJIT_ENTER_CPU_SPECIFIC_OPTIONS (0)
#endif /* !SLJIT_CONFIG_X86 */

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types,
	sljit_s32 scratches, sljit_s32 saveds, sljit_s32 local_size)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	sljit_s32 real_scratches = ENTER_GET_REGS(scratches);
	sljit_s32 real_saveds = ENTER_GET_REGS(saveds);
	sljit_s32 real_fscratches = ENTER_GET_FLOAT_REGS(scratches);
	sljit_s32 real_fsaveds = ENTER_GET_FLOAT_REGS(saveds);
	sljit_s32 real_vscratches = ENTER_GET_VECTOR_REGS(scratches);
	sljit_s32 real_vsaveds = ENTER_GET_VECTOR_REGS(saveds);
#endif /* SLJIT_ARGUMENT_CHECKS */
	SLJIT_UNUSED_ARG(compiler);

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	if (options & SLJIT_ENTER_REG_ARG) {
		CHECK_ARGUMENT(!(options & ~(0x3 | SLJIT_ENTER_REG_ARG | SLJIT_ENTER_CPU_SPECIFIC_OPTIONS)));
	} else {
		CHECK_ARGUMENT((options & ~SLJIT_ENTER_CPU_SPECIFIC_OPTIONS) == 0);
	}
	CHECK_ARGUMENT(SLJIT_KEPT_SAVEDS_COUNT(options) <= 3 && SLJIT_KEPT_SAVEDS_COUNT(options) <= saveds);
	CHECK_ARGUMENT((scratches & ~0xffffff) == 0 && (saveds & ~0xffffff) == 0);
	CHECK_ARGUMENT(real_scratches >= 0 && real_scratches <= SLJIT_NUMBER_OF_REGISTERS);
	CHECK_ARGUMENT(real_saveds >= 0 && real_saveds <= SLJIT_NUMBER_OF_SAVED_REGISTERS);
	CHECK_ARGUMENT(real_scratches + real_saveds <= SLJIT_NUMBER_OF_REGISTERS);
	CHECK_ARGUMENT(real_fscratches >= 0 && real_fscratches <= SLJIT_NUMBER_OF_FLOAT_REGISTERS);
	CHECK_ARGUMENT(real_fsaveds >= 0 && real_fsaveds <= SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS);
	CHECK_ARGUMENT(real_fscratches + real_fsaveds <= SLJIT_NUMBER_OF_FLOAT_REGISTERS);
	CHECK_ARGUMENT(real_vscratches >= 0 && real_vscratches <= SLJIT_NUMBER_OF_VECTOR_REGISTERS);
	CHECK_ARGUMENT(real_vsaveds >= 0 && real_vsaveds <= SLJIT_NUMBER_OF_SAVED_VECTOR_REGISTERS);
	CHECK_ARGUMENT(real_vscratches + real_vsaveds <= SLJIT_NUMBER_OF_VECTOR_REGISTERS);
	CHECK_ARGUMENT(local_size >= 0 && local_size <= SLJIT_MAX_LOCAL_SIZE);
	CHECK_ARGUMENT((arg_types & SLJIT_ARG_FULL_MASK) <= SLJIT_ARG_TYPE_F32);
	CHECK_ARGUMENT(function_check_arguments(arg_types, real_scratches,
		(options & SLJIT_ENTER_REG_ARG) ? 0 : real_saveds, real_fscratches));

	compiler->last_flags = 0;
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  enter ret[%s", call_arg_names[arg_types & SLJIT_ARG_MASK]);

		arg_types >>= SLJIT_ARG_SHIFT;
		if (arg_types) {
			fprintf(compiler->verbose, "], args[");
			do {
				fprintf(compiler->verbose, "%s%s", call_arg_names[arg_types & SLJIT_ARG_MASK],
					(arg_types & SLJIT_ARG_TYPE_SCRATCH_REG) ? "_r" : "");
				arg_types >>= SLJIT_ARG_SHIFT;
				if (arg_types)
					fprintf(compiler->verbose, ",");
			} while (arg_types);
		}

		fprintf(compiler->verbose, "],");

		if (options & SLJIT_ENTER_REG_ARG) {
			if (SLJIT_KEPT_SAVEDS_COUNT(options) > 0)
				fprintf(compiler->verbose, " opt:reg_arg(%d),", SLJIT_KEPT_SAVEDS_COUNT(options));
			else
				fprintf(compiler->verbose, " opt:reg_arg,");
		}

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)
		if (options & SLJIT_ENTER_USE_VEX) {
				fprintf(compiler->verbose, " opt:use_vex,");
		}
#endif /* !SLJIT_CONFIG_X86 */

		fprintf(compiler->verbose, " scratches:%d, saveds:%d, fscratches:%d, fsaveds:%d, vscratches:%d, vsaveds:%d, local_size:%d\n",
			ENTER_GET_REGS(scratches), ENTER_GET_REGS(saveds), ENTER_GET_FLOAT_REGS(scratches), ENTER_GET_FLOAT_REGS(saveds),
			ENTER_GET_VECTOR_REGS(scratches), ENTER_GET_VECTOR_REGS(saveds), local_size);
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types,
	sljit_s32 scratches, sljit_s32 saveds, sljit_s32 local_size)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	sljit_s32 real_scratches = ENTER_GET_REGS(scratches);
	sljit_s32 real_saveds = ENTER_GET_REGS(saveds);
	sljit_s32 real_fscratches = ENTER_GET_FLOAT_REGS(scratches);
	sljit_s32 real_fsaveds = ENTER_GET_FLOAT_REGS(saveds);
	sljit_s32 real_vscratches = ENTER_GET_VECTOR_REGS(scratches);
	sljit_s32 real_vsaveds = ENTER_GET_VECTOR_REGS(saveds);
#endif /* SLJIT_ARGUMENT_CHECKS */
	SLJIT_UNUSED_ARG(compiler);

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	if (options & SLJIT_ENTER_REG_ARG) {
		CHECK_ARGUMENT(!(options & ~(0x3 | SLJIT_ENTER_REG_ARG | SLJIT_ENTER_CPU_SPECIFIC_OPTIONS)));
	} else {
		CHECK_ARGUMENT((options & ~SLJIT_ENTER_CPU_SPECIFIC_OPTIONS) == 0);
	}
	CHECK_ARGUMENT(SLJIT_KEPT_SAVEDS_COUNT(options) <= 3 && SLJIT_KEPT_SAVEDS_COUNT(options) <= saveds);
	CHECK_ARGUMENT((scratches & ~0xffffff) == 0 && (saveds & ~0xffffff) == 0);
	CHECK_ARGUMENT(real_scratches >= 0 && real_scratches <= SLJIT_NUMBER_OF_REGISTERS);
	CHECK_ARGUMENT(real_saveds >= 0 && real_saveds <= SLJIT_NUMBER_OF_SAVED_REGISTERS);
	CHECK_ARGUMENT(real_scratches + real_saveds <= SLJIT_NUMBER_OF_REGISTERS);
	CHECK_ARGUMENT(real_fscratches >= 0 && real_fscratches <= SLJIT_NUMBER_OF_FLOAT_REGISTERS);
	CHECK_ARGUMENT(real_fsaveds >= 0 && real_fsaveds <= SLJIT_NUMBER_OF_SAVED_FLOAT_REGISTERS);
	CHECK_ARGUMENT(real_fscratches + real_fsaveds <= SLJIT_NUMBER_OF_FLOAT_REGISTERS);
	CHECK_ARGUMENT(real_vscratches >= 0 && real_vscratches <= SLJIT_NUMBER_OF_VECTOR_REGISTERS);
	CHECK_ARGUMENT(real_vsaveds >= 0 && real_vsaveds <= SLJIT_NUMBER_OF_SAVED_VECTOR_REGISTERS);
	CHECK_ARGUMENT(real_vscratches + real_vsaveds <= SLJIT_NUMBER_OF_VECTOR_REGISTERS);
	CHECK_ARGUMENT(local_size >= 0 && local_size <= SLJIT_MAX_LOCAL_SIZE);
	CHECK_ARGUMENT((arg_types & SLJIT_ARG_FULL_MASK) < SLJIT_ARG_TYPE_F64);
	CHECK_ARGUMENT(function_check_arguments(arg_types, real_scratches,
		(options & SLJIT_ENTER_REG_ARG) ? 0 : real_saveds, real_fscratches));

	compiler->last_flags = 0;
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  set_context ret[%s", call_arg_names[arg_types & SLJIT_ARG_MASK]);

		arg_types >>= SLJIT_ARG_SHIFT;
		if (arg_types) {
			fprintf(compiler->verbose, "], args[");
			do {
				fprintf(compiler->verbose, "%s%s", call_arg_names[arg_types & SLJIT_ARG_MASK],
					(arg_types & SLJIT_ARG_TYPE_SCRATCH_REG) ? "_r" : "");
				arg_types >>= SLJIT_ARG_SHIFT;
				if (arg_types)
					fprintf(compiler->verbose, ",");
			} while (arg_types);
		}

		fprintf(compiler->verbose, "],");

		if (options & SLJIT_ENTER_REG_ARG) {
			if (SLJIT_KEPT_SAVEDS_COUNT(options) > 0)
				fprintf(compiler->verbose, " opt:reg_arg(%d),", SLJIT_KEPT_SAVEDS_COUNT(options));
			else
				fprintf(compiler->verbose, " opt:reg_arg,");
		}

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)
		if (options & SLJIT_ENTER_USE_VEX) {
				fprintf(compiler->verbose, " opt:use_vex,");
		}
#endif /* !SLJIT_CONFIG_X86 */

		fprintf(compiler->verbose, " scratches:%d, saveds:%d, fscratches:%d, fsaveds:%d, vscratches:%d, vsaveds:%d, local_size:%d\n",
			ENTER_GET_REGS(scratches), ENTER_GET_REGS(saveds), ENTER_GET_FLOAT_REGS(scratches), ENTER_GET_FLOAT_REGS(saveds),
			ENTER_GET_VECTOR_REGS(scratches), ENTER_GET_VECTOR_REGS(saveds), local_size);
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

#undef SLJIT_ENTER_CPU_SPECIFIC_OPTIONS

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_return_void(struct sljit_compiler *compiler)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(compiler->last_return == SLJIT_ARG_TYPE_RET_VOID);
#endif /* SLJIT_ARGUMENT_CHECKS */

#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  return_void\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_return(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(compiler->scratches >= 0);

	switch (compiler->last_return) {
	case SLJIT_ARG_TYPE_W:
		CHECK_ARGUMENT(op >= SLJIT_MOV && op <= SLJIT_MOV_S32);
		break;
	case SLJIT_ARG_TYPE_32:
		CHECK_ARGUMENT(op == SLJIT_MOV32 || (op >= SLJIT_MOV32_U8 && op <= SLJIT_MOV32_S16));
		break;
	case SLJIT_ARG_TYPE_P:
		CHECK_ARGUMENT(op == SLJIT_MOV_P);
		break;
	case SLJIT_ARG_TYPE_F64:
		CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
		CHECK_ARGUMENT(op == SLJIT_MOV_F64);
		break;
	case SLJIT_ARG_TYPE_F32:
		CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
		CHECK_ARGUMENT(op == SLJIT_MOV_F32);
		break;
	default:
		/* Context not initialized, void, etc. */
		CHECK_ARGUMENT(0);
		break;
	}

	if (SLJIT_CHECK_OPCODE(op, 0) < SLJIT_MOV_F64) {
		FUNCTION_CHECK_SRC(src, srcw);
	} else {
		FUNCTION_FCHECK(src, srcw, op & SLJIT_32);
	}
	compiler->last_flags = 0;
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		if (GET_OPCODE(op) < SLJIT_MOV_F64) {
			fprintf(compiler->verbose, "  return%s%s ", !(op & SLJIT_32) ? "" : "32",
				op1_types[GET_OPCODE(op) - SLJIT_OP1_BASE]);
			sljit_verbose_param(compiler, src, srcw);
		} else {
			fprintf(compiler->verbose, "  return%s ", !(op & SLJIT_32) ? ".f64" : ".f32");
			sljit_verbose_fparam(compiler, src, srcw);
		}
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_return_to(struct sljit_compiler *compiler,
	sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	FUNCTION_CHECK_SRC(src, srcw);
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  return_to ");
		sljit_verbose_param(compiler, src, srcw);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_op0(struct sljit_compiler *compiler, sljit_s32 op)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT((op >= SLJIT_BREAKPOINT && op <= SLJIT_LMUL_SW)
		|| ((op & ~SLJIT_32) >= SLJIT_DIVMOD_UW && (op & ~SLJIT_32) <= SLJIT_DIV_SW)
		|| (op >= SLJIT_MEMORY_BARRIER && op <= SLJIT_SKIP_FRAMES_BEFORE_RETURN));
	CHECK_ARGUMENT(SLJIT_CHECK_OPCODE(op, 0) < SLJIT_LMUL_UW || SLJIT_CHECK_OPCODE(op, 0) >= SLJIT_MEMORY_BARRIER || compiler->scratches >= 2);
	if ((SLJIT_CHECK_OPCODE(op, 0) >= SLJIT_LMUL_UW && SLJIT_CHECK_OPCODE(op, 0) <= SLJIT_DIV_SW) || op == SLJIT_SKIP_FRAMES_BEFORE_RETURN)
		compiler->last_flags = 0;
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose))
	{
		fprintf(compiler->verbose, "  %s", op0_names[GET_OPCODE(op) - SLJIT_OP0_BASE]);
		if (GET_OPCODE(op) >= SLJIT_DIVMOD_UW && GET_OPCODE(op) <= SLJIT_DIV_SW) {
			fprintf(compiler->verbose, (op & SLJIT_32) ? "32" : "w");
		}
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_op1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(SLJIT_CHECK_OPCODE(op, 0) >= SLJIT_MOV && SLJIT_CHECK_OPCODE(op, 0) <= SLJIT_REV_S32);

	switch (GET_OPCODE(op)) {
	case SLJIT_MOV:
	case SLJIT_MOV_U32:
	case SLJIT_MOV_S32:
	case SLJIT_MOV32:
	case SLJIT_MOV_P:
	case SLJIT_REV_U32:
	case SLJIT_REV_S32:
		/* Nothing allowed */
		CHECK_ARGUMENT(!(op & (SLJIT_32 | SLJIT_SET_Z | VARIABLE_FLAG_MASK)));
		break;
	default:
		/* Only SLJIT_32 is allowed. */
		CHECK_ARGUMENT(!(op & (SLJIT_SET_Z | VARIABLE_FLAG_MASK)));
		break;
	}

	FUNCTION_CHECK_DST(dst, dstw);
	FUNCTION_CHECK_SRC(src, srcw);
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s%s%s ", op1_names[GET_OPCODE(op) - SLJIT_OP1_BASE],
			!(op & SLJIT_32) ? "" : "32", op1_types[GET_OPCODE(op) - SLJIT_OP1_BASE]);

		sljit_verbose_param(compiler, dst, dstw);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, src, srcw);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_atomic_load(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 mem_reg)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_ATOMIC));
	CHECK_ARGUMENT(SLJIT_CHECK_OPCODE(op, SLJIT_ATOMIC_TEST | SLJIT_ATOMIC_USE_CAS | SLJIT_ATOMIC_USE_LS | SLJIT_SET_Z | VARIABLE_FLAG_MASK) >= SLJIT_MOV
		&& SLJIT_CHECK_OPCODE(op, SLJIT_ATOMIC_TEST | SLJIT_ATOMIC_USE_CAS | SLJIT_ATOMIC_USE_LS | SLJIT_SET_Z | VARIABLE_FLAG_MASK) <= SLJIT_MOV_P);
	CHECK_ARGUMENT((op & (SLJIT_ATOMIC_USE_CAS | SLJIT_ATOMIC_USE_LS)) != (SLJIT_ATOMIC_USE_CAS | SLJIT_ATOMIC_USE_LS));

	/* All arguments must be valid registers. */
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(dst_reg));
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(mem_reg) && !CHECK_IF_VIRTUAL_REGISTER(mem_reg));

	if (GET_OPCODE(op) < SLJIT_MOV_U8 || GET_OPCODE(op) > SLJIT_MOV_S16) {
		/* Nothing allowed. */
		CHECK_ARGUMENT(!(op & SLJIT_32));
	}

	compiler->last_flags = 0;
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		if (op & SLJIT_ATOMIC_TEST)
			CHECK_RETURN_OK;
		if (sljit_emit_atomic_load(compiler, op | SLJIT_ATOMIC_TEST, dst_reg, mem_reg)) {
			fprintf(compiler->verbose, "    # atomic_load: unsupported form, no instructions are emitted\n");
			CHECK_RETURN_OK;
		}

		fprintf(compiler->verbose, "  atomic_load");
		if (op & SLJIT_ATOMIC_USE_CAS)
			fprintf(compiler->verbose, "_cas");
		if (op & SLJIT_ATOMIC_USE_LS)
			fprintf(compiler->verbose, "_ls");

		fprintf(compiler->verbose, "%s%s ", !(op & SLJIT_32) ? "" : "32",
				op1_types[GET_OPCODE(op) - SLJIT_OP1_BASE]);
		sljit_verbose_reg(compiler, dst_reg);
		fprintf(compiler->verbose, ", [");
		sljit_verbose_reg(compiler, mem_reg);
		fprintf(compiler->verbose, "]\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_atomic_store(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src_reg,
	sljit_s32 mem_reg,
	sljit_s32 temp_reg)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_ATOMIC));
	CHECK_ARGUMENT(SLJIT_CHECK_OPCODE(op, SLJIT_ATOMIC_TEST | SLJIT_ATOMIC_USE_CAS | SLJIT_ATOMIC_USE_LS | SLJIT_SET_Z) >= SLJIT_MOV
		&& SLJIT_CHECK_OPCODE(op, SLJIT_ATOMIC_TEST | SLJIT_ATOMIC_USE_CAS | SLJIT_ATOMIC_USE_LS | SLJIT_SET_Z) <= SLJIT_MOV_P);
	CHECK_ARGUMENT((op & (SLJIT_ATOMIC_USE_CAS | SLJIT_ATOMIC_USE_LS)) != (SLJIT_ATOMIC_USE_CAS | SLJIT_ATOMIC_USE_LS));

	/* All arguments must be valid registers. */
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(src_reg));
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(mem_reg) && !CHECK_IF_VIRTUAL_REGISTER(mem_reg));
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(temp_reg) && (src_reg != temp_reg || (op & SLJIT_ATOMIC_USE_LS)));

	CHECK_ARGUMENT(!(op & VARIABLE_FLAG_MASK) || GET_FLAG_TYPE_MASK(op) == SLJIT_ATOMIC_STORED);

	if (GET_OPCODE(op) < SLJIT_MOV_U8 || GET_OPCODE(op) > SLJIT_MOV_S16) {
		/* Nothing allowed. */
		CHECK_ARGUMENT(!(op & SLJIT_32));
	}

	compiler->last_flags = GET_FLAG_TYPE_MASK(op) | (op & SLJIT_32);
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		if (op & SLJIT_ATOMIC_TEST)
			CHECK_RETURN_OK;
		if (sljit_emit_atomic_store(compiler, op | SLJIT_ATOMIC_TEST, src_reg, mem_reg, temp_reg)) {
			fprintf(compiler->verbose, "    # atomic_store: unsupported form, no instructions are emitted\n");
			CHECK_RETURN_OK;
		}

		fprintf(compiler->verbose, "  atomic_store");
		if (op & SLJIT_ATOMIC_USE_CAS)
			fprintf(compiler->verbose, "_cas");
		if (op & SLJIT_ATOMIC_USE_LS)
			fprintf(compiler->verbose, "_ls");

		fprintf(compiler->verbose, "%s%s%s ", !(op & SLJIT_32) ? "" : "32",
				op1_types[GET_OPCODE(op) - SLJIT_OP1_BASE], !(op & VARIABLE_FLAG_MASK) ? "" : ".stored");
		sljit_verbose_reg(compiler, src_reg);
		fprintf(compiler->verbose, ", [");
		sljit_verbose_reg(compiler, mem_reg);
		fprintf(compiler->verbose, "], ");
		sljit_verbose_reg(compiler, temp_reg);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_op2(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 unset,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(SLJIT_CHECK_OPCODE(op, 0) >= SLJIT_ADD && SLJIT_CHECK_OPCODE(op, 0) <= SLJIT_ROTR);

	switch (GET_OPCODE(op)) {
	case SLJIT_AND:
	case SLJIT_OR:
	case SLJIT_XOR:
	case SLJIT_SHL:
	case SLJIT_MSHL:
	case SLJIT_LSHR:
	case SLJIT_MLSHR:
	case SLJIT_ASHR:
	case SLJIT_MASHR:
		CHECK_ARGUMENT(!(op & VARIABLE_FLAG_MASK));
		break;
	case SLJIT_MUL:
		CHECK_ARGUMENT(!(op & SLJIT_SET_Z));
		CHECK_ARGUMENT(!(op & VARIABLE_FLAG_MASK)
			|| GET_FLAG_TYPE(op) == SLJIT_OVERFLOW);
		break;
	case SLJIT_ADD:
		CHECK_ARGUMENT(!(op & VARIABLE_FLAG_MASK)
			|| GET_FLAG_TYPE(op) == GET_FLAG_TYPE(SLJIT_SET_CARRY)
			|| GET_FLAG_TYPE(op) == SLJIT_OVERFLOW);
		break;
	case SLJIT_SUB:
		CHECK_ARGUMENT(!(op & VARIABLE_FLAG_MASK)
			|| (GET_FLAG_TYPE(op) >= SLJIT_LESS && GET_FLAG_TYPE(op) <= SLJIT_OVERFLOW)
			|| GET_FLAG_TYPE(op) == GET_FLAG_TYPE(SLJIT_SET_CARRY));
		break;
	case SLJIT_ADDC:
	case SLJIT_SUBC:
		CHECK_ARGUMENT(!(op & VARIABLE_FLAG_MASK)
			|| GET_FLAG_TYPE(op) == GET_FLAG_TYPE(SLJIT_SET_CARRY));
		CHECK_ARGUMENT((compiler->last_flags & 0xff) == GET_FLAG_TYPE(SLJIT_SET_CARRY));
		CHECK_ARGUMENT((op & SLJIT_32) == (compiler->last_flags & SLJIT_32));
		break;
	case SLJIT_ROTL:
	case SLJIT_ROTR:
		CHECK_ARGUMENT(!(op & (SLJIT_SET_Z | VARIABLE_FLAG_MASK)));
		break;
	default:
		SLJIT_UNREACHABLE();
		break;
	}

	if (unset) {
		CHECK_ARGUMENT(HAS_FLAGS(op));
	} else {
		FUNCTION_CHECK_DST(dst, dstw);
	}
	FUNCTION_CHECK_SRC(src1, src1w);
	FUNCTION_CHECK_SRC(src2, src2w);
	compiler->last_flags = GET_FLAG_TYPE(op) | (op & (SLJIT_32 | SLJIT_SET_Z));
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s%s%s%s%s ", op2_names[GET_OPCODE(op) - SLJIT_OP2_BASE], !(op & SLJIT_32) ? "" : "32",
			!(op & SLJIT_SET_Z) ? "" : ".z", !(op & VARIABLE_FLAG_MASK) ? "" : ".",
			!(op & VARIABLE_FLAG_MASK) ? "" : jump_names[GET_FLAG_TYPE(op)]);
		if (unset)
			fprintf(compiler->verbose, "unset");
		else
			sljit_verbose_param(compiler, dst, dstw);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, src1, src1w);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, src2, src2w);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_op2r(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT((op | SLJIT_32) == SLJIT_MULADD32);
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(dst_reg));
	FUNCTION_CHECK_SRC(src1, src1w);
	FUNCTION_CHECK_SRC(src2, src2w);
	compiler->last_flags = 0;
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s%s ", op2r_names[GET_OPCODE(op) - SLJIT_OP2R_BASE], !(op & SLJIT_32) ? "" : "32");

		sljit_verbose_reg(compiler, dst_reg);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, src1, src1w);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, src2, src2w);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_shift_into(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_reg,
	sljit_s32 src1_reg,
	sljit_s32 src2_reg,
	sljit_s32 src3, sljit_sw src3w)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(SLJIT_CHECK_OPCODE(op, 0) == SLJIT_SHL || SLJIT_CHECK_OPCODE(op, 0) == SLJIT_LSHR
		|| SLJIT_CHECK_OPCODE(op, 0) == SLJIT_MSHL || SLJIT_CHECK_OPCODE(op, 0) == SLJIT_MLSHR);
	CHECK_ARGUMENT((op & ~(0xff | SLJIT_32 | SLJIT_SHIFT_INTO_NON_ZERO)) == 0);
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(dst_reg));
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(src1_reg));
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(src2_reg));
	FUNCTION_CHECK_SRC(src3, src3w);
	CHECK_ARGUMENT(dst_reg != src2_reg);
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s%s.into%s ", op2_names[GET_OPCODE(op) - SLJIT_OP2_BASE], !(op & SLJIT_32) ? "" : "32",
			(op & SLJIT_SHIFT_INTO_NON_ZERO) ? ".nz" : "");

		sljit_verbose_reg(compiler, dst_reg);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_reg(compiler, src1_reg);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_reg(compiler, src2_reg);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, src3, src3w);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_op_src(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(op >= SLJIT_FAST_RETURN && op <= SLJIT_PREFETCH_ONCE);
	FUNCTION_CHECK_SRC(src, srcw);

	if (op == SLJIT_FAST_RETURN || op == SLJIT_SKIP_FRAMES_BEFORE_FAST_RETURN) {
		CHECK_ARGUMENT(src != SLJIT_IMM);
		compiler->last_flags = 0;
	} else if (op >= SLJIT_PREFETCH_L1 && op <= SLJIT_PREFETCH_ONCE) {
		CHECK_ARGUMENT(src & SLJIT_MEM);
	}
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s ", op_src_dst_names[op - SLJIT_OP_SRC_DST_BASE]);
		sljit_verbose_param(compiler, src, srcw);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_op_dst(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(op >= SLJIT_FAST_ENTER && op <= SLJIT_GET_RETURN_ADDRESS);
	FUNCTION_CHECK_DST(dst, dstw);

	if (op == SLJIT_FAST_ENTER)
		compiler->last_flags = 0;
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s ", op_src_dst_names[op - SLJIT_OP_SRC_DST_BASE]);
		sljit_verbose_param(compiler, dst, dstw);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_get_register_index(sljit_s32 type, sljit_s32 reg)
{
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(reg);
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	if (type == SLJIT_GP_REGISTER) {
		CHECK_ARGUMENT((reg > 0 && reg <= SLJIT_NUMBER_OF_REGISTERS)
			|| (reg >= SLJIT_TMP_REGISTER_BASE && reg < (SLJIT_TMP_REGISTER_BASE + SLJIT_NUMBER_OF_TEMPORARY_REGISTERS)));
	}
#if (defined SLJIT_SEPARATE_VECTOR_REGISTERS && SLJIT_SEPARATE_VECTOR_REGISTERS)
	else if (((type >> 12) == 0 || ((type >> 12) >= 3 && (type >> 12) <= 6))) {
		CHECK_ARGUMENT((reg > 0 && reg <= SLJIT_NUMBER_OF_VECTOR_REGISTERS)
			|| (reg >= SLJIT_TMP_VREGISTER_BASE && reg < (SLJIT_TMP_VREGISTER_BASE + SLJIT_NUMBER_OF_TEMPORARY_VECTOR_REGISTERS)));
	}
#endif /* SLJIT_SEPARATE_VECTOR_REGISTERS */
	else {
		CHECK_ARGUMENT(type == SLJIT_FLOAT_REGISTER || ((type >> 12) == 0 || ((type >> 12) >= 3 && (type >> 12) <= 6) || (type & (3 << 12)) || (type & (4 << 12)) || (type & (5 << 12)) || (type & (6 << 12))));
		CHECK_ARGUMENT((reg > 0 && reg <= SLJIT_NUMBER_OF_FLOAT_REGISTERS)
			|| (reg >= SLJIT_TMP_FREGISTER_BASE && reg < (SLJIT_TMP_FREGISTER_BASE + SLJIT_NUMBER_OF_TEMPORARY_FLOAT_REGISTERS)));
	}
#endif /* SLJIT_ARGUMENT_CHECKS */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_op_custom(struct sljit_compiler *compiler,
	void *instruction, sljit_u32 size)
{
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	sljit_u32 i;
#endif /* SLJIT_VERBOSE */

	SLJIT_UNUSED_ARG(compiler);

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(instruction);

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)
	CHECK_ARGUMENT(size > 0 && size < 16);
#elif (defined SLJIT_CONFIG_ARM_THUMB2 && SLJIT_CONFIG_ARM_THUMB2)
	CHECK_ARGUMENT((size == 2 && (((sljit_sw)instruction) & 0x1) == 0)
		|| (size == 4 && (((sljit_sw)instruction) & 0x3) == 0));
#elif (defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X)
	CHECK_ARGUMENT(size == 2 || size == 4 || size == 6);
#else /* !SLJIT_CONFIG_X86 && !SLJIT_CONFIG_ARM_THUMB2 && !SLJIT_CONFIG_S390X */
	CHECK_ARGUMENT(size == 4 && (((sljit_sw)instruction) & 0x3) == 0);
#endif /* SLJIT_CONFIG_X86 */

	compiler->last_flags = 0;
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  op_custom");
		for (i = 0; i < size; i++)
			fprintf(compiler->verbose, " 0x%x", ((sljit_u8*)instruction)[i]);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fop1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
	CHECK_ARGUMENT(SLJIT_CHECK_OPCODE(op, 0) >= SLJIT_MOV_F64 && SLJIT_CHECK_OPCODE(op, 0) <= SLJIT_ABS_F64);
	CHECK_ARGUMENT(!(op & (SLJIT_SET_Z | VARIABLE_FLAG_MASK)));
	FUNCTION_FCHECK(src, srcw, op & SLJIT_32);
	FUNCTION_FCHECK(dst, dstw, op & SLJIT_32);
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_F32)
			fprintf(compiler->verbose, "  %s%s ", fop1_names[SLJIT_CONV_F64_FROM_F32 - SLJIT_FOP1_BASE],
				(op & SLJIT_32) ? ".f32.from.f64" : ".f64.from.f32");
		else
			fprintf(compiler->verbose, "  %s%s ", fop1_names[GET_OPCODE(op) - SLJIT_FOP1_BASE],
				(op & SLJIT_32) ? ".f32" : ".f64");

		sljit_verbose_fparam(compiler, dst, dstw);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_fparam(compiler, src, srcw);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fop1_cmp(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	compiler->last_flags = GET_FLAG_TYPE(op) | (op & SLJIT_32);
#endif /* SLJIT_ARGUMENT_CHECKS */

	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
	CHECK_ARGUMENT(SLJIT_CHECK_OPCODE(op, 0) == SLJIT_CMP_F64);
	CHECK_ARGUMENT(!(op & SLJIT_SET_Z));
	CHECK_ARGUMENT((op & VARIABLE_FLAG_MASK)
		|| (GET_FLAG_TYPE(op) >= SLJIT_F_EQUAL && GET_FLAG_TYPE(op) <= SLJIT_ORDERED_LESS_EQUAL));
	FUNCTION_FCHECK(src1, src1w, op & SLJIT_32);
	FUNCTION_FCHECK(src2, src2w, op & SLJIT_32);
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s%s", fop1_names[SLJIT_CMP_F64 - SLJIT_FOP1_BASE], (op & SLJIT_32) ? ".f32" : ".f64");
		if (op & VARIABLE_FLAG_MASK) {
			fprintf(compiler->verbose, ".%s", jump_names[GET_FLAG_TYPE(op)]);
		}
		fprintf(compiler->verbose, " ");
		sljit_verbose_fparam(compiler, src1, src1w);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_fparam(compiler, src2, src2w);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fop1_conv_sw_from_f64(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
	CHECK_ARGUMENT(!(op & (SLJIT_SET_Z | VARIABLE_FLAG_MASK)));
	FUNCTION_FCHECK(src, srcw, op & SLJIT_32);
	FUNCTION_CHECK_DST(dst, dstw);
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s%s.from%s ", fop1_names[GET_OPCODE(op) - SLJIT_FOP1_BASE],
			fop1_conv_types[GET_OPCODE(op) - SLJIT_CONV_SW_FROM_F64],
			(op & SLJIT_32) ? ".f32" : ".f64");
		sljit_verbose_param(compiler, dst, dstw);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_fparam(compiler, src, srcw);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fop1_conv_f64_from_w(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
	CHECK_ARGUMENT(!(op & (SLJIT_SET_Z | VARIABLE_FLAG_MASK)));
	FUNCTION_CHECK_SRC(src, srcw);
	FUNCTION_FCHECK(dst, dstw, op & SLJIT_32);
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s%s.from.%s ", fop1_names[GET_OPCODE(op) - SLJIT_FOP1_BASE],
			(op & SLJIT_32) ? ".f32" : ".f64",
			fop1_conv_types[GET_OPCODE(op) - SLJIT_CONV_SW_FROM_F64]);
		sljit_verbose_fparam(compiler, dst, dstw);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, src, srcw);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fop2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
	CHECK_ARGUMENT(SLJIT_CHECK_OPCODE(op, 0) >= SLJIT_ADD_F64 && SLJIT_CHECK_OPCODE(op, 0) <= SLJIT_DIV_F64);
	CHECK_ARGUMENT(!(op & (SLJIT_SET_Z | VARIABLE_FLAG_MASK)));
	FUNCTION_FCHECK(src1, src1w, op & SLJIT_32);
	FUNCTION_FCHECK(src2, src2w, op & SLJIT_32);
	FUNCTION_FCHECK(dst, dstw, op & SLJIT_32);
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s%s ", fop2_names[GET_OPCODE(op) - SLJIT_FOP2_BASE], (op & SLJIT_32) ? ".f32" : ".f64");
		sljit_verbose_fparam(compiler, dst, dstw);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_fparam(compiler, src1, src1w);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_fparam(compiler, src2, src2w);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fop2r(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_freg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
	CHECK_ARGUMENT(SLJIT_CHECK_OPCODE(op, 0) == SLJIT_COPYSIGN_F64);
	FUNCTION_FCHECK(src1, src1w, op & SLJIT_32);
	FUNCTION_FCHECK(src2, src2w, op & SLJIT_32);
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_FREG(dst_freg, op & SLJIT_32));
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s%s ", fop2r_names[GET_OPCODE(op) - SLJIT_FOP2R_BASE], (op & SLJIT_32) ? ".f32" : ".f64");
		sljit_verbose_freg(compiler, dst_freg);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_fparam(compiler, src1, src1w);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_fparam(compiler, src2, src2w);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fset32(struct sljit_compiler *compiler,
	sljit_s32 freg, sljit_f32 value)
{
	SLJIT_UNUSED_ARG(value);

	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_FREG(freg, 1));
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  fset32 ");
		sljit_verbose_freg(compiler, freg);
		fprintf(compiler->verbose, ", %f\n", value);
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fset64(struct sljit_compiler *compiler,
	sljit_s32 freg, sljit_f64 value)
{
	SLJIT_UNUSED_ARG(value);

	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_FREG(freg, 0));
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  fset64 ");
		sljit_verbose_freg(compiler, freg);
		fprintf(compiler->verbose, ", %f\n", value);
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fcopy(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 freg, sljit_s32 reg)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
	CHECK_ARGUMENT(SLJIT_CHECK_OPCODE(op, 0) >= SLJIT_COPY_TO_F64 && SLJIT_CHECK_OPCODE(op, 0) <= SLJIT_COPY_FROM_F64);
	CHECK_ARGUMENT(!(op & (SLJIT_SET_Z | VARIABLE_FLAG_MASK)));
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_FREG(freg, op & SLJIT_32));

#if (defined SLJIT_64BIT_ARCHITECTURE && SLJIT_64BIT_ARCHITECTURE)
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(reg));
#else /* !SLJIT_64BIT_ARCHITECTURE */
	switch (op) {
	case SLJIT_COPY32_TO_F32:
	case SLJIT_COPY32_FROM_F32:
		CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(reg));
		break;
	case SLJIT_COPY_TO_F64:
	case SLJIT_COPY_FROM_F64:
		if (reg & REG_PAIR_MASK) {
			CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(REG_PAIR_FIRST(reg)));
			CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(REG_PAIR_SECOND(reg)));

			if (op == SLJIT_COPY_TO_F64)
				break;

			CHECK_ARGUMENT(REG_PAIR_FIRST(reg) != REG_PAIR_SECOND(reg));
			break;
		}

		CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(reg));
		break;
	}
#endif /* SLJIT_64BIT_ARCHITECTURE */
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  copy%s_%s_f%s ", (op & SLJIT_32) ? "32" : "",
			GET_OPCODE(op) == SLJIT_COPY_TO_F64 ? "to" : "from", (op & SLJIT_32) ? "32" : "64");

		sljit_verbose_freg(compiler, freg);

		if (reg & REG_PAIR_MASK) {
			fprintf(compiler->verbose, ", {");
			sljit_verbose_reg(compiler, REG_PAIR_FIRST(reg));
			fprintf(compiler->verbose, ", ");
			sljit_verbose_reg(compiler, REG_PAIR_SECOND(reg));
			fprintf(compiler->verbose, "}\n");
		} else {
			fprintf(compiler->verbose, ", ");
			sljit_verbose_reg(compiler, reg);
			fprintf(compiler->verbose, "\n");
		}
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_label(struct sljit_compiler *compiler)
{
	SLJIT_UNUSED_ARG(compiler);

	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	compiler->last_flags = 0;
#endif /* SLJIT_ARGUMENT_CHECKS */

#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose))
		fprintf(compiler->verbose, "label:\n");
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86) \
	|| (defined SLJIT_CONFIG_ARM && SLJIT_CONFIG_ARM)
#define CHECK_UNORDERED(type, last_flags) \
	((((type) & 0xfe) == SLJIT_ORDERED) && \
		((last_flags) & 0xff) >= SLJIT_UNORDERED && ((last_flags) & 0xff) <= SLJIT_ORDERED_LESS_EQUAL)
#else /* !SLJIT_CONFIG_X86 || SLJIT_CONFIG_ARM */
#define CHECK_UNORDERED(type, last_flags) 0
#endif /* SLJIT_CONFIG_X86 || SLJIT_CONFIG_ARM */
#endif /* SLJIT_ARGUMENT_CHECKS */

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_jump(struct sljit_compiler *compiler, sljit_s32 type)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(!(type & ~(0xff | SLJIT_REWRITABLE_JUMP)));
	CHECK_ARGUMENT((type & 0xff) >= SLJIT_EQUAL && (type & 0xff) <= SLJIT_FAST_CALL);

	if ((type & 0xff) < SLJIT_JUMP) {
		if ((type & 0xff) <= SLJIT_NOT_ZERO)
			CHECK_ARGUMENT(compiler->last_flags & SLJIT_SET_Z);
		else if ((compiler->last_flags & 0xff) == SLJIT_CARRY) {
			CHECK_ARGUMENT((type & 0xfe) == SLJIT_CARRY);
			compiler->last_flags = 0;
		} else
			CHECK_ARGUMENT((type & 0xfe) == (compiler->last_flags & 0xff)
				|| CHECK_UNORDERED(type, compiler->last_flags));
	}
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose))
		fprintf(compiler->verbose, "  jump%s %s\n", !(type & SLJIT_REWRITABLE_JUMP) ? "" : ".r",
			jump_names[type & 0xff]);
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_call(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(!(type & ~(0xff | SLJIT_REWRITABLE_JUMP | SLJIT_CALL_RETURN)));
	CHECK_ARGUMENT((type & 0xff) >= SLJIT_CALL && (type & 0xff) <= SLJIT_CALL_REG_ARG);
	CHECK_ARGUMENT(function_check_arguments(arg_types, compiler->scratches, -1, compiler->fscratches));

	if (type & SLJIT_CALL_RETURN) {
		CHECK_ARGUMENT((arg_types & SLJIT_ARG_MASK) == compiler->last_return);

		if (compiler->options & SLJIT_ENTER_REG_ARG) {
			CHECK_ARGUMENT((type & 0xff) == SLJIT_CALL_REG_ARG);
		} else {
			CHECK_ARGUMENT((type & 0xff) != SLJIT_CALL_REG_ARG);
		}
	}
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s%s%s ret[%s", jump_names[type & 0xff],
			!(type & SLJIT_REWRITABLE_JUMP) ? "" : ".r",
			!(type & SLJIT_CALL_RETURN) ? "" : ".ret",
			call_arg_names[arg_types & SLJIT_ARG_MASK]);

		arg_types >>= SLJIT_ARG_SHIFT;
		if (arg_types) {
			fprintf(compiler->verbose, "], args[");
			do {
				fprintf(compiler->verbose, "%s", call_arg_names[arg_types & SLJIT_ARG_MASK]);
				arg_types >>= SLJIT_ARG_SHIFT;
				if (arg_types)
					fprintf(compiler->verbose, ",");
			} while (arg_types);
		}
		fprintf(compiler->verbose, "]\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_cmp(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(!(type & ~(0xff | SLJIT_REWRITABLE_JUMP | SLJIT_32)));
	CHECK_ARGUMENT((type & 0xff) >= SLJIT_EQUAL && (type & 0xff) <= SLJIT_SIG_LESS_EQUAL);
	FUNCTION_CHECK_SRC(src1, src1w);
	FUNCTION_CHECK_SRC(src2, src2w);
	compiler->last_flags = 0;
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  cmp%s%s %s, ", (type & SLJIT_32) ? "32" : "",
			!(type & SLJIT_REWRITABLE_JUMP) ? "" : ".r", jump_names[type & 0xff]);
		sljit_verbose_param(compiler, src1, src1w);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, src2, src2w);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fcmp(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
	CHECK_ARGUMENT(!(type & ~(0xff | SLJIT_REWRITABLE_JUMP | SLJIT_32)));
	CHECK_ARGUMENT((type & 0xff) >= SLJIT_F_EQUAL && (type & 0xff) <= SLJIT_ORDERED_LESS_EQUAL);
	FUNCTION_FCHECK(src1, src1w, type & SLJIT_32);
	FUNCTION_FCHECK(src2, src2w, type & SLJIT_32);
	compiler->last_flags = 0;
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  fcmp%s%s %s, ", (type & SLJIT_32) ? ".f32" : ".f64",
			!(type & SLJIT_REWRITABLE_JUMP) ? "" : ".r", jump_names[type & 0xff]);
		sljit_verbose_fparam(compiler, src1, src1w);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_fparam(compiler, src2, src2w);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_ijump(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 src, sljit_sw srcw)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(type >= SLJIT_JUMP && type <= SLJIT_FAST_CALL);
	FUNCTION_CHECK_SRC(src, srcw);
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  ijump.%s ", jump_names[type]);
		sljit_verbose_param(compiler, src, srcw);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_icall(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types,
	sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(!(type & ~(0xff | SLJIT_CALL_RETURN)));
	CHECK_ARGUMENT((type & 0xff) >= SLJIT_CALL && (type & 0xff) <= SLJIT_CALL_REG_ARG);
	CHECK_ARGUMENT(function_check_arguments(arg_types, compiler->scratches, -1, compiler->fscratches));
	FUNCTION_CHECK_SRC(src, srcw);

	if (type & SLJIT_CALL_RETURN) {
		CHECK_ARGUMENT((arg_types & SLJIT_ARG_MASK) == compiler->last_return);

		if (compiler->options & SLJIT_ENTER_REG_ARG) {
			CHECK_ARGUMENT((type & 0xff) == SLJIT_CALL_REG_ARG);
		} else {
			CHECK_ARGUMENT((type & 0xff) != SLJIT_CALL_REG_ARG);
		}
	}
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  i%s%s ret[%s", jump_names[type & 0xff],
			!(type & SLJIT_CALL_RETURN) ? "" : ".ret",
			call_arg_names[arg_types & SLJIT_ARG_MASK]);

		arg_types >>= SLJIT_ARG_SHIFT;
		if (arg_types) {
			fprintf(compiler->verbose, "], args[");
			do {
				fprintf(compiler->verbose, "%s", call_arg_names[arg_types & SLJIT_ARG_MASK]);
				arg_types >>= SLJIT_ARG_SHIFT;
				if (arg_types)
					fprintf(compiler->verbose, ",");
			} while (arg_types);
		}
		fprintf(compiler->verbose, "], ");
		sljit_verbose_param(compiler, src, srcw);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_op_flags(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 type)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(type >= SLJIT_EQUAL && type <= SLJIT_ORDERED_LESS_EQUAL);
	CHECK_ARGUMENT(op == SLJIT_MOV || op == SLJIT_MOV32
		|| (SLJIT_CHECK_OPCODE(op, 0) >= SLJIT_AND && SLJIT_CHECK_OPCODE(op, 0) <= SLJIT_XOR));
	CHECK_ARGUMENT(!(op & VARIABLE_FLAG_MASK));

	if (type <= SLJIT_NOT_ZERO)
		CHECK_ARGUMENT(compiler->last_flags & SLJIT_SET_Z);
	else
		CHECK_ARGUMENT((type & 0xfe) == (compiler->last_flags & 0xff)
			|| CHECK_UNORDERED(type, compiler->last_flags));

	FUNCTION_CHECK_DST(dst, dstw);

	if (GET_OPCODE(op) >= SLJIT_ADD)
		compiler->last_flags = GET_FLAG_TYPE(op) | (op & (SLJIT_32 | SLJIT_SET_Z));
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  flags.%s%s%s ",
			GET_OPCODE(op) < SLJIT_OP2_BASE ? "mov" : op2_names[GET_OPCODE(op) - SLJIT_OP2_BASE],
			GET_OPCODE(op) < SLJIT_OP2_BASE ? op1_types[GET_OPCODE(op) - SLJIT_OP1_BASE] : ((op & SLJIT_32) ? "32" : ""),
			!(op & SLJIT_SET_Z) ? "" : ".z");
		sljit_verbose_param(compiler, dst, dstw);
		fprintf(compiler->verbose, ", %s\n", jump_names[type]);
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_select(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_reg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2_reg)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	sljit_s32 cond = type & ~SLJIT_32;

	CHECK_ARGUMENT(cond >= SLJIT_EQUAL && cond <= SLJIT_ORDERED_LESS_EQUAL);

	CHECK_ARGUMENT(compiler->scratches != -1 && compiler->saveds != -1);
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(dst_reg));
	FUNCTION_CHECK_SRC(src1, src1w);
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(src2_reg));

	if (cond <= SLJIT_NOT_ZERO)
		CHECK_ARGUMENT(compiler->last_flags & SLJIT_SET_Z);
	else if ((compiler->last_flags & 0xff) == SLJIT_CARRY) {
		CHECK_ARGUMENT((type & 0xfe) == SLJIT_CARRY);
		compiler->last_flags = 0;
	} else
		CHECK_ARGUMENT((cond & 0xfe) == (compiler->last_flags & 0xff)
			|| CHECK_UNORDERED(cond, compiler->last_flags));
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  select%s %s, ",
			!(type & SLJIT_32) ? "" : "32",
			jump_names[type & ~SLJIT_32]);
		sljit_verbose_reg(compiler, dst_reg);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, src1, src1w);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_reg(compiler, src2_reg);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fselect(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_freg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2_freg)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	sljit_s32 cond = type & ~SLJIT_32;

	CHECK_ARGUMENT(cond >= SLJIT_EQUAL && cond <= SLJIT_ORDERED_LESS_EQUAL);

	CHECK_ARGUMENT(compiler->fscratches != -1 && compiler->fsaveds != -1);
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_FREG(dst_freg, type & SLJIT_32));
	FUNCTION_FCHECK(src1, src1w, type & SLJIT_32);
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_FREG(src2_freg, type & SLJIT_32));

	if (cond <= SLJIT_NOT_ZERO)
		CHECK_ARGUMENT(compiler->last_flags & SLJIT_SET_Z);
	else if ((compiler->last_flags & 0xff) == SLJIT_CARRY) {
		CHECK_ARGUMENT((type & 0xfe) == SLJIT_CARRY);
		compiler->last_flags = 0;
	} else
		CHECK_ARGUMENT((cond & 0xfe) == (compiler->last_flags & 0xff)
			|| CHECK_UNORDERED(cond, compiler->last_flags));
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  fselect%s %s, ",
			!(type & SLJIT_32) ? "" : "32",
			jump_names[type & ~SLJIT_32]);
		sljit_verbose_freg(compiler, dst_freg);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_fparam(compiler, src1, src1w);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_freg(compiler, src2_freg);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_mem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	sljit_s32 allowed_flags;
#endif /* SLJIT_ARGUMENT_CHECKS */

	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	if (type & SLJIT_MEM_UNALIGNED) {
		CHECK_ARGUMENT(!(type & (SLJIT_MEM_ALIGNED_16 | SLJIT_MEM_ALIGNED_32)));
	} else if (type & SLJIT_MEM_ALIGNED_16) {
		CHECK_ARGUMENT(!(type & SLJIT_MEM_ALIGNED_32));
	} else {
		CHECK_ARGUMENT((reg & REG_PAIR_MASK) || (type & SLJIT_MEM_ALIGNED_32));
	}

	allowed_flags = SLJIT_MEM_UNALIGNED;

	switch (type & 0xff) {
	case SLJIT_MOV_P:
	case SLJIT_MOV:
		allowed_flags |= SLJIT_MEM_ALIGNED_32;
		/* fallthrough */
	case SLJIT_MOV_U32:
	case SLJIT_MOV_S32:
	case SLJIT_MOV32:
		allowed_flags |= SLJIT_MEM_ALIGNED_16;
		break;
	}

	CHECK_ARGUMENT((type & ~(0xff | SLJIT_32 | SLJIT_MEM_STORE | allowed_flags)) == 0);

	if (reg & REG_PAIR_MASK) {
		CHECK_ARGUMENT((type & 0xff) == SLJIT_MOV);
		CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(REG_PAIR_FIRST(reg)));
		CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(REG_PAIR_SECOND(reg)));
		CHECK_ARGUMENT(REG_PAIR_FIRST(reg) != REG_PAIR_SECOND(reg));
	} else {
		CHECK_ARGUMENT((type & 0xff) >= SLJIT_MOV && (type & 0xff) <= SLJIT_MOV_P);
		CHECK_ARGUMENT(!(type & SLJIT_32) || ((type & 0xff) >= SLJIT_MOV_U8 && (type & 0xff) <= SLJIT_MOV_S16));
		CHECK_ARGUMENT(FUNCTION_CHECK_IS_REG(reg));
	}

	FUNCTION_CHECK_SRC_MEM(mem, memw);
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		if ((type & 0xff) == SLJIT_MOV32)
			fprintf(compiler->verbose, "  %s32",
				(type & SLJIT_MEM_STORE) ? "store" : "load");
		else
			fprintf(compiler->verbose, "  %s%s%s",
				(type & SLJIT_MEM_STORE) ? "store" : "load",
				!(type & SLJIT_32) ? "" : "32", op1_types[(type & 0xff) - SLJIT_OP1_BASE]);

		if (type & SLJIT_MEM_UNALIGNED)
			printf(".unal");
		else if (type & SLJIT_MEM_ALIGNED_16)
			printf(".al16");
		else if (type & SLJIT_MEM_ALIGNED_32)
			printf(".al32");

		if (reg & REG_PAIR_MASK) {
			fprintf(compiler->verbose, " {");
			sljit_verbose_reg(compiler, REG_PAIR_FIRST(reg));
			fprintf(compiler->verbose, ", ");
			sljit_verbose_reg(compiler, REG_PAIR_SECOND(reg));
			fprintf(compiler->verbose, "}, ");
		} else {
			fprintf(compiler->verbose, " ");
			sljit_verbose_reg(compiler, reg);
			fprintf(compiler->verbose, ", ");
		}
		sljit_verbose_param(compiler, mem, memw);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_mem_update(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw)
{
	if (SLJIT_UNLIKELY(compiler->skip_checks)) {
		compiler->skip_checks = 0;
		CHECK_RETURN_OK;
	}

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT((type & 0xff) >= SLJIT_MOV && (type & 0xff) <= SLJIT_MOV_P);
	CHECK_ARGUMENT((type & ~(0xff | SLJIT_32 | SLJIT_MEM_STORE | SLJIT_MEM_SUPP | SLJIT_MEM_POST)) == 0);
	CHECK_ARGUMENT((mem & REG_MASK) != 0 && (mem & REG_MASK) != reg);

	FUNCTION_CHECK_SRC_MEM(mem, memw);
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		if (type & SLJIT_MEM_SUPP)
			CHECK_RETURN_OK;
		if (sljit_emit_mem_update(compiler, type | SLJIT_MEM_SUPP, reg, mem, memw) == SLJIT_ERR_UNSUPPORTED) {
			fprintf(compiler->verbose, "    # mem: unsupported form, no instructions are emitted\n");
			CHECK_RETURN_OK;
		}

		if ((type & 0xff) == SLJIT_MOV32)
			fprintf(compiler->verbose, "  %s32.%s ",
				(type & SLJIT_MEM_STORE) ? "store" : "load",
				(type & SLJIT_MEM_POST) ? "post" : "pre");
		else
			fprintf(compiler->verbose, "  %s%s%s.%s ",
				(type & SLJIT_MEM_STORE) ? "store" : "load",
				!(type & SLJIT_32) ? "" : "32",
				op1_types[(type & 0xff) - SLJIT_OP1_BASE],
				(type & SLJIT_MEM_POST) ? "post" : "pre");

		sljit_verbose_reg(compiler, reg);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, mem, memw);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fmem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 mem, sljit_sw memw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
	CHECK_ARGUMENT((type & 0xff) == SLJIT_MOV_F64);

	if (type & SLJIT_MEM_UNALIGNED) {
		CHECK_ARGUMENT(!(type & (SLJIT_MEM_ALIGNED_16 | SLJIT_MEM_ALIGNED_32)));
	} else if (type & SLJIT_MEM_ALIGNED_16) {
		CHECK_ARGUMENT(!(type & SLJIT_MEM_ALIGNED_32));
	} else {
		CHECK_ARGUMENT(type & SLJIT_MEM_ALIGNED_32);
		CHECK_ARGUMENT(!(type & SLJIT_32));
	}

	CHECK_ARGUMENT(!(type & ~(0xff | SLJIT_32 | SLJIT_MEM_STORE | SLJIT_MEM_UNALIGNED | SLJIT_MEM_ALIGNED_16 | SLJIT_MEM_ALIGNED_32)));
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_FREG(freg, type & SLJIT_32));
	FUNCTION_CHECK_SRC_MEM(mem, memw);
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  %s.%s",
			(type & SLJIT_MEM_STORE) ? "store" : "load",
			!(type & SLJIT_32) ? "f64" : "f32");

		if (type & SLJIT_MEM_UNALIGNED)
			printf(".unal");
		else if (type & SLJIT_MEM_ALIGNED_16)
			printf(".al16");
		else if (type & SLJIT_MEM_ALIGNED_32)
			printf(".al32");

		fprintf(compiler->verbose, " ");
		sljit_verbose_freg(compiler, freg);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, mem, memw);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_fmem_update(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 mem, sljit_sw memw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_FPU));
	CHECK_ARGUMENT((type & 0xff) == SLJIT_MOV_F64);
	CHECK_ARGUMENT((type & ~(0xff | SLJIT_32 | SLJIT_MEM_STORE | SLJIT_MEM_SUPP | SLJIT_MEM_POST)) == 0);
	FUNCTION_CHECK_SRC_MEM(mem, memw);
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_FREG(freg, type & SLJIT_32));
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		if (type & SLJIT_MEM_SUPP)
			CHECK_RETURN_OK;
		if (sljit_emit_fmem_update(compiler, type | SLJIT_MEM_SUPP, freg, mem, memw) == SLJIT_ERR_UNSUPPORTED) {
			fprintf(compiler->verbose, "    # fmem: unsupported form, no instructions are emitted\n");
			CHECK_RETURN_OK;
		}

		fprintf(compiler->verbose, "  %s.%s.%s ",
			(type & SLJIT_MEM_STORE) ? "store" : "load",
			!(type & SLJIT_32) ? "f64" : "f32",
			(type & SLJIT_MEM_POST) ? "post" : "pre");

		sljit_verbose_freg(compiler, freg);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, mem, memw);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_simd_mov(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg,
	sljit_s32 srcdst, sljit_sw srcdstw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_SIMD));
	CHECK_ARGUMENT((type & SLJIT_SIMD_TYPE_MASK2(SLJIT_SIMD_STORE)) == 0);
	CHECK_ARGUMENT(SLJIT_SIMD_CHECK_REG(type));
	CHECK_ARGUMENT(SLJIT_SIMD_GET_ELEM_SIZE(type) <= SLJIT_SIMD_GET_REG_SIZE(type));
	CHECK_ARGUMENT(SLJIT_SIMD_GET_ELEM2_SIZE(type) <= (srcdst & SLJIT_MEM) ? SLJIT_SIMD_GET_REG_SIZE(type) : 0);
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_VREG(vreg, type));
	FUNCTION_VCHECK(srcdst, srcdstw, type);
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		if (type & SLJIT_SIMD_TEST)
			CHECK_RETURN_OK;
		if (sljit_emit_simd_mov(compiler, type | SLJIT_SIMD_TEST, vreg, srcdst, srcdstw) == SLJIT_ERR_UNSUPPORTED) {
			fprintf(compiler->verbose, "    # simd_mem: unsupported form, no instructions are emitted\n");
			CHECK_RETURN_OK;
		}

		fprintf(compiler->verbose, "  simd_%s.%d.%s%d",
			(type & SLJIT_SIMD_STORE) ? "store" : "load",
			(8 << SLJIT_SIMD_GET_REG_SIZE(type)),
			(type & SLJIT_SIMD_FLOAT) ? "f" : "",
			(8 << SLJIT_SIMD_GET_ELEM_SIZE(type)));

		if ((type & 0x3f000000) == SLJIT_SIMD_MEM_UNALIGNED)
			fprintf(compiler->verbose, ".unal ");
		else
			fprintf(compiler->verbose, ".al%d ", (8 << SLJIT_SIMD_GET_ELEM2_SIZE(type)));

		sljit_verbose_vreg(compiler, vreg);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_vparam(compiler, srcdst, srcdstw);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_simd_replicate(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg,
	sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_SIMD));
	CHECK_ARGUMENT((type & SLJIT_SIMD_TYPE_MASK(0)) == 0);
	CHECK_ARGUMENT(SLJIT_SIMD_CHECK_REG(type));
	CHECK_ARGUMENT(SLJIT_SIMD_GET_ELEM_SIZE(type) < SLJIT_SIMD_GET_REG_SIZE(type));
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_VREG(vreg, type));

	if (type & SLJIT_SIMD_FLOAT) {
		if (src == SLJIT_IMM) {
			CHECK_ARGUMENT(srcw == 0);
		} else {
			FUNCTION_FCHECK(src, srcw, SLJIT_SIMD_GET_ELEM_SIZE(type) == 2);
		}
	} else if (src != SLJIT_IMM) {
		FUNCTION_CHECK_DST(src, srcw);
	}
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		if (type & SLJIT_SIMD_TEST)
			CHECK_RETURN_OK;
		if (sljit_emit_simd_replicate(compiler, type | SLJIT_SIMD_TEST, vreg, src, srcw) == SLJIT_ERR_UNSUPPORTED) {
			fprintf(compiler->verbose, "    # simd_dup: unsupported form, no instructions are emitted\n");
			CHECK_RETURN_OK;
		}

		fprintf(compiler->verbose, "  simd_replicate.%d.%s%d ",
			(8 << SLJIT_SIMD_GET_REG_SIZE(type)),
			(type & SLJIT_SIMD_FLOAT) ? "f" : "",
			(8 << SLJIT_SIMD_GET_ELEM_SIZE(type)));

		sljit_verbose_vreg(compiler, vreg);
		fprintf(compiler->verbose, ", ");
		if (type & SLJIT_SIMD_FLOAT)
			sljit_verbose_fparam(compiler, src, srcw);
		else
			sljit_verbose_param(compiler, src, srcw);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_simd_lane_mov(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg, sljit_s32 lane_index,
	sljit_s32 srcdst, sljit_sw srcdstw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_SIMD));
	CHECK_ARGUMENT((type & SLJIT_SIMD_TYPE_MASK(SLJIT_SIMD_STORE | SLJIT_SIMD_LANE_ZERO | SLJIT_SIMD_LANE_SIGNED | SLJIT_32)) == 0);
	CHECK_ARGUMENT((type & (SLJIT_SIMD_STORE | SLJIT_SIMD_LANE_ZERO)) != (SLJIT_SIMD_STORE | SLJIT_SIMD_LANE_ZERO));
	CHECK_ARGUMENT((type & (SLJIT_SIMD_STORE | SLJIT_SIMD_LANE_SIGNED)) != SLJIT_SIMD_LANE_SIGNED);
	CHECK_ARGUMENT(!(type & SLJIT_SIMD_FLOAT) || !(type & (SLJIT_SIMD_LANE_SIGNED | SLJIT_32)));
	CHECK_ARGUMENT(SLJIT_SIMD_CHECK_REG(type));
	CHECK_ARGUMENT(SLJIT_SIMD_GET_ELEM_SIZE(type) < SLJIT_SIMD_GET_REG_SIZE(type));
	CHECK_ARGUMENT(!(type & SLJIT_32) || SLJIT_SIMD_GET_ELEM_SIZE(type) <= 2);
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_VREG(vreg, type));
	CHECK_ARGUMENT(lane_index >= 0 && lane_index < (1 << (SLJIT_SIMD_GET_REG_SIZE(type) - SLJIT_SIMD_GET_ELEM_SIZE(type))));

	if (type & SLJIT_SIMD_FLOAT) {
		FUNCTION_FCHECK(srcdst, srcdstw, SLJIT_SIMD_GET_ELEM_SIZE(type) == 2);
	} else if ((type & SLJIT_SIMD_STORE) || srcdst != SLJIT_IMM) {
		FUNCTION_CHECK_DST(srcdst, srcdstw);
	}
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		if (type & SLJIT_SIMD_TEST)
			CHECK_RETURN_OK;
		if (sljit_emit_simd_lane_mov(compiler, type | SLJIT_SIMD_TEST, vreg, lane_index, srcdst, srcdstw) == SLJIT_ERR_UNSUPPORTED) {
			fprintf(compiler->verbose, "    # simd_move_lane: unsupported form, no instructions are emitted\n");
			CHECK_RETURN_OK;
		}

		fprintf(compiler->verbose, "  simd_%s_lane%s%s%s.%d.%s%d ",
			(type & SLJIT_SIMD_STORE) ? "store" : "load",
			(type & SLJIT_32) ? "32" : "",
			(type & SLJIT_SIMD_LANE_ZERO) ? "_z" : "",
			(type & SLJIT_SIMD_LANE_SIGNED) ? "_s" : "",
			(8 << SLJIT_SIMD_GET_REG_SIZE(type)),
			(type & SLJIT_SIMD_FLOAT) ? "f" : "",
			(8 << SLJIT_SIMD_GET_ELEM_SIZE(type)));

		sljit_verbose_vreg(compiler, vreg);
		fprintf(compiler->verbose, "[%d], ", lane_index);
		if (type & SLJIT_SIMD_FLOAT)
			sljit_verbose_fparam(compiler, srcdst, srcdstw);
		else
			sljit_verbose_param(compiler, srcdst, srcdstw);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_simd_lane_replicate(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg,
	sljit_s32 src, sljit_s32 src_lane_index)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_SIMD));
	CHECK_ARGUMENT((type & SLJIT_SIMD_TYPE_MASK(0)) == 0);
	CHECK_ARGUMENT(SLJIT_SIMD_CHECK_REG(type));
	CHECK_ARGUMENT(SLJIT_SIMD_GET_ELEM_SIZE(type) < SLJIT_SIMD_GET_REG_SIZE(type));
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_VREG(vreg, type));
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_VREG(src, type));
	CHECK_ARGUMENT(src_lane_index >= 0 && src_lane_index < (1 << (SLJIT_SIMD_GET_REG_SIZE(type) - SLJIT_SIMD_GET_ELEM_SIZE(type))));
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		if (type & SLJIT_SIMD_TEST)
			CHECK_RETURN_OK;
		if (sljit_emit_simd_lane_replicate(compiler, type | SLJIT_SIMD_TEST, vreg, src, src_lane_index) == SLJIT_ERR_UNSUPPORTED) {
			fprintf(compiler->verbose, "    # simd_lane_replicate: unsupported form, no instructions are emitted\n");
			CHECK_RETURN_OK;
		}

		fprintf(compiler->verbose, "  simd_lane_replicate.%d.%s%d ",
			(8 << SLJIT_SIMD_GET_REG_SIZE(type)),
			(type & SLJIT_SIMD_FLOAT) ? "f" : "",
			(8 << SLJIT_SIMD_GET_ELEM_SIZE(type)));

		sljit_verbose_vreg(compiler, vreg);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_vreg(compiler, src);
		fprintf(compiler->verbose, "[%d]\n", src_lane_index);
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_simd_extend(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg,
	sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_SIMD));
	CHECK_ARGUMENT((type & SLJIT_SIMD_TYPE_MASK2(SLJIT_SIMD_EXTEND_SIGNED)) == 0);
	CHECK_ARGUMENT((type & (SLJIT_SIMD_EXTEND_SIGNED | SLJIT_SIMD_FLOAT)) != (SLJIT_SIMD_EXTEND_SIGNED | SLJIT_SIMD_FLOAT));
	CHECK_ARGUMENT(SLJIT_SIMD_CHECK_REG(type));
	CHECK_ARGUMENT(SLJIT_SIMD_GET_ELEM2_SIZE(type) < SLJIT_SIMD_GET_REG_SIZE(type));
	CHECK_ARGUMENT(SLJIT_SIMD_GET_ELEM_SIZE(type) < SLJIT_SIMD_GET_ELEM2_SIZE(type));
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_VREG(vreg, type));
	FUNCTION_VCHECK(src, srcw, type);
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		if (type & SLJIT_SIMD_TEST)
			CHECK_RETURN_OK;
		if (sljit_emit_simd_extend(compiler, type | SLJIT_SIMD_TEST, vreg, src, srcw) == SLJIT_ERR_UNSUPPORTED) {
			fprintf(compiler->verbose, "    # simd_extend: unsupported form, no instructions are emitted\n");
			CHECK_RETURN_OK;
		}

		fprintf(compiler->verbose, "  simd_load_extend%s.%d.%s%d.%s%d ",
			(type & SLJIT_SIMD_EXTEND_SIGNED) ? "_s" : "",
			(8 << SLJIT_SIMD_GET_REG_SIZE(type)),
			(type & SLJIT_SIMD_FLOAT) ? "f" : "",
			(8 << SLJIT_SIMD_GET_ELEM2_SIZE(type)),
			(type & SLJIT_SIMD_FLOAT) ? "f" : "",
			(8 << SLJIT_SIMD_GET_ELEM_SIZE(type)));

		sljit_verbose_vreg(compiler, vreg);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_vparam(compiler, src, srcw);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_simd_sign(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg,
	sljit_s32 dst, sljit_sw dstw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_SIMD));
	CHECK_ARGUMENT((type & SLJIT_SIMD_TYPE_MASK(SLJIT_32)) == SLJIT_SIMD_STORE);
	CHECK_ARGUMENT(SLJIT_SIMD_CHECK_REG(type));
	CHECK_ARGUMENT(SLJIT_SIMD_GET_ELEM_SIZE(type) < SLJIT_SIMD_GET_REG_SIZE(type));
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_VREG(vreg, type));
	FUNCTION_CHECK_DST(dst, dstw);
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		if (type & SLJIT_SIMD_TEST)
			CHECK_RETURN_OK;
		if (sljit_emit_simd_sign(compiler, type | SLJIT_SIMD_TEST, vreg, dst, dstw) == SLJIT_ERR_UNSUPPORTED) {
			fprintf(compiler->verbose, "    # simd_sign: unsupported form, no instructions are emitted\n");
			CHECK_RETURN_OK;
		}

		fprintf(compiler->verbose, "  simd_store_sign%s.%d.%s%d ",
			(type & SLJIT_32) ? "32" : "",
			(8 << SLJIT_SIMD_GET_REG_SIZE(type)),
			(type & SLJIT_SIMD_FLOAT) ? "f" : "",
			(8 << SLJIT_SIMD_GET_ELEM_SIZE(type)));

		sljit_verbose_vreg(compiler, vreg);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_param(compiler, dst, dstw);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_simd_op2(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_vreg, sljit_s32 src1_vreg, sljit_s32 src2, sljit_sw src2w)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	CHECK_ARGUMENT(sljit_has_cpu_feature(SLJIT_HAS_SIMD));
	CHECK_ARGUMENT((type & SLJIT_SIMD_TYPE_MASK2(0)) >= SLJIT_SIMD_OP2_AND && (type & SLJIT_SIMD_TYPE_MASK2(0)) <= SLJIT_SIMD_OP2_SHUFFLE);
	CHECK_ARGUMENT(SLJIT_SIMD_CHECK_REG(type));
	CHECK_ARGUMENT(SLJIT_SIMD_GET_ELEM_SIZE(type) <= SLJIT_SIMD_GET_REG_SIZE(type));
	CHECK_ARGUMENT(SLJIT_SIMD_GET_OPCODE(type) != SLJIT_SIMD_OP2_SHUFFLE || (SLJIT_SIMD_GET_ELEM_SIZE(type) == 0 && !(type & SLJIT_SIMD_FLOAT)));
	CHECK_ARGUMENT(SLJIT_SIMD_GET_ELEM2_SIZE(type) <= (src2 & SLJIT_MEM) ? SLJIT_SIMD_GET_REG_SIZE(type) : 0);
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_VREG(dst_vreg, type));
	CHECK_ARGUMENT(FUNCTION_CHECK_IS_VREG(src1_vreg, type));
	FUNCTION_VCHECK(src2, src2w, type);
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		if (type & SLJIT_SIMD_TEST)
			CHECK_RETURN_OK;
		if (sljit_emit_simd_op2(compiler, type | SLJIT_SIMD_TEST, dst_vreg, src1_vreg, src2, src2w) == SLJIT_ERR_UNSUPPORTED) {
			fprintf(compiler->verbose, "    # simd_op2: unsupported form, no instructions are emitted\n");
			CHECK_RETURN_OK;
		}

		fprintf(compiler->verbose, "  simd_%s.%d.%s%d",
			simd_op2_names[SLJIT_SIMD_GET_OPCODE(type) - 1],
			(8 << SLJIT_SIMD_GET_REG_SIZE(type)),
			(type & SLJIT_SIMD_FLOAT) ? "f" : "",
			(8 << SLJIT_SIMD_GET_ELEM_SIZE(type)));

		if ((type & 0x3f000000) != SLJIT_SIMD_MEM_UNALIGNED)
			fprintf(compiler->verbose, ".al%d", (8 << SLJIT_SIMD_GET_ELEM2_SIZE(type)));

		fprintf(compiler->verbose, " ");
		sljit_verbose_vreg(compiler, dst_vreg);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_vreg(compiler, src1_vreg);
		fprintf(compiler->verbose, ", ");
		sljit_verbose_vparam(compiler, src2, src2w);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_get_local_base(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw offset)
{
	/* Any offset is allowed. */
	SLJIT_UNUSED_ARG(offset);

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	FUNCTION_CHECK_DST(dst, dstw);
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  local_base ");
		sljit_verbose_param(compiler, dst, dstw);
		fprintf(compiler->verbose, ", #%" SLJIT_PRINT_D "d\n", offset);
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_const(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw init_value)
{
	SLJIT_UNUSED_ARG(init_value);

#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	FUNCTION_CHECK_DST(dst, dstw);
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  const ");
		sljit_verbose_param(compiler, dst, dstw);
		fprintf(compiler->verbose, ", #%" SLJIT_PRINT_D "d\n", init_value);
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

static SLJIT_INLINE CHECK_RETURN_TYPE check_sljit_emit_mov_addr(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw)
{
#if (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	FUNCTION_CHECK_DST(dst, dstw);
#endif /* SLJIT_ARGUMENT_CHECKS */
#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE)
	if (SLJIT_UNLIKELY(!!compiler->verbose)) {
		fprintf(compiler->verbose, "  mov_addr ");
		sljit_verbose_param(compiler, dst, dstw);
		fprintf(compiler->verbose, "\n");
	}
#endif /* SLJIT_VERBOSE */
	CHECK_RETURN_OK;
}

#else /* !SLJIT_ARGUMENT_CHECKS && !SLJIT_VERBOSE */

#define SLJIT_SKIP_CHECKS(compiler)

#endif /* SLJIT_ARGUMENT_CHECKS || SLJIT_VERBOSE */

#define SELECT_FOP1_OPERATION_WITH_CHECKS(compiler, op, dst, dstw, src, srcw) \
	SLJIT_COMPILE_ASSERT(!(SLJIT_CONV_SW_FROM_F64 & 0x1) && !(SLJIT_CONV_F64_FROM_SW & 0x1) && !(SLJIT_CONV_F64_FROM_UW & 0x1), \
		invalid_float_opcodes); \
	if (GET_OPCODE(op) >= SLJIT_CONV_SW_FROM_F64 && GET_OPCODE(op) <= SLJIT_CMP_F64) { \
		if (GET_OPCODE(op) == SLJIT_CMP_F64) { \
			CHECK(check_sljit_emit_fop1_cmp(compiler, op, dst, dstw, src, srcw)); \
			ADJUST_LOCAL_OFFSET(dst, dstw); \
			ADJUST_LOCAL_OFFSET(src, srcw); \
			return sljit_emit_fop1_cmp(compiler, op, dst, dstw, src, srcw); \
		} \
		if ((GET_OPCODE(op) | 0x1) == SLJIT_CONV_S32_FROM_F64) { \
			CHECK(check_sljit_emit_fop1_conv_sw_from_f64(compiler, op, dst, dstw, src, srcw)); \
			ADJUST_LOCAL_OFFSET(dst, dstw); \
			ADJUST_LOCAL_OFFSET(src, srcw); \
			return sljit_emit_fop1_conv_sw_from_f64(compiler, op, dst, dstw, src, srcw); \
		} \
		if ((GET_OPCODE(op) | 0x1) == SLJIT_CONV_F64_FROM_S32) { \
			CHECK(check_sljit_emit_fop1_conv_f64_from_w(compiler, op, dst, dstw, src, srcw)); \
			ADJUST_LOCAL_OFFSET(dst, dstw); \
			ADJUST_LOCAL_OFFSET(src, srcw); \
			return sljit_emit_fop1_conv_f64_from_sw(compiler, op, dst, dstw, src, srcw); \
		} \
		CHECK(check_sljit_emit_fop1_conv_f64_from_w(compiler, op, dst, dstw, src, srcw)); \
		ADJUST_LOCAL_OFFSET(dst, dstw); \
		ADJUST_LOCAL_OFFSET(src, srcw); \
		return sljit_emit_fop1_conv_f64_from_uw(compiler, op, dst, dstw, src, srcw); \
	} \
	CHECK(check_sljit_emit_fop1(compiler, op, dst, dstw, src, srcw)); \
	ADJUST_LOCAL_OFFSET(dst, dstw); \
	ADJUST_LOCAL_OFFSET(src, srcw);

#if (!(defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS) || (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6))

static sljit_s32 sljit_emit_mem_unaligned(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw)
{
	SLJIT_SKIP_CHECKS(compiler);

	if (type & SLJIT_MEM_STORE)
		return sljit_emit_op1(compiler, type & (0xff | SLJIT_32), mem, memw, reg, 0);
	return sljit_emit_op1(compiler, type & (0xff | SLJIT_32), reg, 0, mem, memw);
}

#endif /* (!SLJIT_CONFIG_MIPS || SLJIT_MIPS_REV >= 6) */

#if (!(defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS) || (defined SLJIT_MIPS_REV && SLJIT_MIPS_REV >= 6)) \
	&& !(defined SLJIT_CONFIG_ARM_32 && SLJIT_CONFIG_ARM_32)

static sljit_s32 sljit_emit_fmem_unaligned(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 mem, sljit_sw memw)
{
	SLJIT_SKIP_CHECKS(compiler);

	if (type & SLJIT_MEM_STORE)
		return sljit_emit_fop1(compiler, type & (0xff | SLJIT_32), mem, memw, freg, 0);
	return sljit_emit_fop1(compiler, type & (0xff | SLJIT_32), freg, 0, mem, memw);
}

#endif /* (!SLJIT_CONFIG_MIPS || SLJIT_MIPS_REV >= 6) && !SLJIT_CONFIG_ARM */

/* CPU description section */

#if (defined SLJIT_32BIT_ARCHITECTURE && SLJIT_32BIT_ARCHITECTURE)
#define SLJIT_CPUINFO_PART1 " 32bit ("
#elif (defined SLJIT_64BIT_ARCHITECTURE && SLJIT_64BIT_ARCHITECTURE)
#define SLJIT_CPUINFO_PART1 " 64bit ("
#else /* !SLJIT_32BIT_ARCHITECTURE && !SLJIT_64BIT_ARCHITECTURE */
#error "Internal error: CPU type info missing"
#endif /* SLJIT_32BIT_ARCHITECTURE */

#if (defined SLJIT_LITTLE_ENDIAN && SLJIT_LITTLE_ENDIAN)
#define SLJIT_CPUINFO_PART2 "little endian + "
#elif (defined SLJIT_BIG_ENDIAN && SLJIT_BIG_ENDIAN)
#define SLJIT_CPUINFO_PART2 "big endian + "
#else /* !SLJIT_LITTLE_ENDIAN && !SLJIT_BIG_ENDIAN */
#error "Internal error: CPU type info missing"
#endif /* SLJIT_LITTLE_ENDIAN */

#if (defined SLJIT_UNALIGNED && SLJIT_UNALIGNED)
#define SLJIT_CPUINFO_PART3 "unaligned)"
#else /* !SLJIT_UNALIGNED */
#define SLJIT_CPUINFO_PART3 "aligned)"
#endif /* SLJIT_UNALIGNED */

#define SLJIT_CPUINFO SLJIT_CPUINFO_PART1 SLJIT_CPUINFO_PART2 SLJIT_CPUINFO_PART3

#if (defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86)
#	include "sljitNativeX86_common.c"
#elif (defined SLJIT_CONFIG_ARM_V6 && SLJIT_CONFIG_ARM_V6)
#	include "sljitNativeARM_32.c"
#elif (defined SLJIT_CONFIG_ARM_V7 && SLJIT_CONFIG_ARM_V7)
#	include "sljitNativeARM_32.c"
#elif (defined SLJIT_CONFIG_ARM_THUMB2 && SLJIT_CONFIG_ARM_THUMB2)
#	include "sljitNativeARM_T2_32.c"
#elif (defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64)
#	include "sljitNativeARM_64.c"
#elif (defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC)
#	include "sljitNativePPC_common.c"
#elif (defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS)
#	include "sljitNativeMIPS_common.c"
#elif (defined SLJIT_CONFIG_RISCV && SLJIT_CONFIG_RISCV)
#	include "sljitNativeRISCV_common.c"
#elif (defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X)
#	include "sljitNativeS390X.c"
#elif (defined SLJIT_CONFIG_LOONGARCH && SLJIT_CONFIG_LOONGARCH)
#	include "sljitNativeLOONGARCH_64.c"
#endif /* SLJIT_CONFIG_X86 */

#include "sljitSerialize.c"

static SLJIT_INLINE sljit_s32 emit_mov_before_return(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 src, sljit_sw srcw)
{
#if (defined SLJIT_64BIT_ARCHITECTURE && SLJIT_64BIT_ARCHITECTURE)
	/* At the moment the pointer size is always equal to sljit_sw. May be changed in the future. */
	if (src == SLJIT_RETURN_REG && (op == SLJIT_MOV || op == SLJIT_MOV_P))
		return SLJIT_SUCCESS;
#else /* !SLJIT_64BIT_ARCHITECTURE */
	if (src == SLJIT_RETURN_REG && (op == SLJIT_MOV || op == SLJIT_MOV_U32 || op == SLJIT_MOV_S32 || op == SLJIT_MOV_P))
		return SLJIT_SUCCESS;
#endif /* SLJIT_64BIT_ARCHITECTURE */

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_op1(compiler, op, SLJIT_RETURN_REG, 0, src, srcw);
}

#if !(defined SLJIT_CONFIG_X86_32 && SLJIT_CONFIG_X86_32) \
	&& !((defined SLJIT_CONFIG_ARM_32 && SLJIT_CONFIG_ARM_32) && defined __SOFTFP__)

static SLJIT_INLINE sljit_s32 emit_fmov_before_return(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 src, sljit_sw srcw)
{
	if (src == SLJIT_FR0)
		return SLJIT_SUCCESS;

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_fop1(compiler, op, SLJIT_RETURN_FREG, 0, src, srcw);
}

#endif /* !SLJIT_CONFIG_X86_32 && !(SLJIT_CONFIG_ARM_32 && __SOFTFP__) */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return(struct sljit_compiler *compiler, sljit_s32 op, sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_return(compiler, op, src, srcw));

	if (GET_OPCODE(op) < SLJIT_MOV_F64) {
		FAIL_IF(emit_mov_before_return(compiler, op, src, srcw));
	} else {
		FAIL_IF(emit_fmov_before_return(compiler, op, src, srcw));
	}

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_return_void(compiler);
}

#if !(defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86) \
	&& !(defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X) \
	&& !(defined(SLJIT_CONFIG_LOONGARCH_64) && SLJIT_CONFIG_LOONGARCH_64)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop2r(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst_freg,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_fop2r(compiler, op, dst_freg, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_fop2(compiler, op, dst_freg, 0, src1, src1w, src2, src2w);
}

#endif /* !SLJIT_CONFIG_X86 && !SLJIT_CONFIG_S390X && !SLJIT_CONFIG_LOONGARCH_64 */

#if !(defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS) \
	&& !(defined SLJIT_CONFIG_RISCV && SLJIT_CONFIG_RISCV) \
	&& !(defined SLJIT_CONFIG_LOONGARCH && SLJIT_CONFIG_LOONGARCH)

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_cmp(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	/* Default compare for most architectures. */
	sljit_s32 flags, tmp_src, condition;
	sljit_sw tmp_srcw;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_cmp(compiler, type, src1, src1w, src2, src2w));

	condition = type & 0xff;
#if (defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64)
	if ((condition == SLJIT_EQUAL || condition == SLJIT_NOT_EQUAL)) {
		if (src1 == SLJIT_IMM && !src1w) {
			src1 = src2;
			src1w = src2w;
			src2 = SLJIT_IMM;
			src2w = 0;
		}
		if (src2 == SLJIT_IMM && !src2w)
			return emit_cmp_to0(compiler, type, src1, src1w);
	}
#endif /* SLJIT_CONFIG_ARM_64 */

	if (SLJIT_UNLIKELY(src1 == SLJIT_IMM && src2 != SLJIT_IMM)) {
		/* Immediate is preferred as second argument by most architectures. */
		switch (condition) {
		case SLJIT_LESS:
			condition = SLJIT_GREATER;
			break;
		case SLJIT_GREATER_EQUAL:
			condition = SLJIT_LESS_EQUAL;
			break;
		case SLJIT_GREATER:
			condition = SLJIT_LESS;
			break;
		case SLJIT_LESS_EQUAL:
			condition = SLJIT_GREATER_EQUAL;
			break;
		case SLJIT_SIG_LESS:
			condition = SLJIT_SIG_GREATER;
			break;
		case SLJIT_SIG_GREATER_EQUAL:
			condition = SLJIT_SIG_LESS_EQUAL;
			break;
		case SLJIT_SIG_GREATER:
			condition = SLJIT_SIG_LESS;
			break;
		case SLJIT_SIG_LESS_EQUAL:
			condition = SLJIT_SIG_GREATER_EQUAL;
			break;
		}

		type = condition | (type & (SLJIT_32 | SLJIT_REWRITABLE_JUMP));
		tmp_src = src1;
		src1 = src2;
		src2 = tmp_src;
		tmp_srcw = src1w;
		src1w = src2w;
		src2w = tmp_srcw;
	}

	if (condition <= SLJIT_NOT_ZERO)
		flags = SLJIT_SET_Z;
	else
		flags = (condition & 0xfe) << VARIABLE_FLAG_SHIFT;

	SLJIT_SKIP_CHECKS(compiler);
	PTR_FAIL_IF(sljit_emit_op2u(compiler,
		SLJIT_SUB | flags | (type & SLJIT_32), src1, src1w, src2, src2w));

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_jump(compiler, condition | (type & (SLJIT_REWRITABLE_JUMP | SLJIT_32)));
}

#endif /* !SLJIT_CONFIG_MIPS */

#if (defined SLJIT_CONFIG_ARM_32 && SLJIT_CONFIG_ARM_32)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_cmp_info(sljit_s32 type)
{
	switch (type) {
	case SLJIT_UNORDERED_OR_EQUAL:
	case SLJIT_ORDERED_NOT_EQUAL:
		return 1;
	}

	return 0;
}

#endif /* SLJIT_CONFIG_ARM */

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_fcmp(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_fcmp(compiler, type, src1, src1w, src2, src2w));

	SLJIT_SKIP_CHECKS(compiler);
	sljit_emit_fop1(compiler, SLJIT_CMP_F64 | ((type & 0xfe) << VARIABLE_FLAG_SHIFT) | (type & SLJIT_32), src1, src1w, src2, src2w);

	SLJIT_SKIP_CHECKS(compiler);
	return sljit_emit_jump(compiler, type);
}

#if !(defined SLJIT_CONFIG_ARM && SLJIT_CONFIG_ARM) \
	&& !(defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_mem_update(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_mem_update(compiler, type, reg, mem, memw));
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(reg);
	SLJIT_UNUSED_ARG(mem);
	SLJIT_UNUSED_ARG(memw);

	return SLJIT_ERR_UNSUPPORTED;
}

#endif /* !SLJIT_CONFIG_ARM && !SLJIT_CONFIG_PPC */

#if !(defined SLJIT_CONFIG_ARM_32 && SLJIT_CONFIG_ARM_32) \
	&& !(defined SLJIT_CONFIG_MIPS && SLJIT_CONFIG_MIPS)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fmem(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 mem, sljit_sw memw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_fmem(compiler, type, freg, mem, memw));

	return sljit_emit_fmem_unaligned(compiler, type, freg, mem, memw);
}

#endif /* !SLJIT_CONFIG_ARM_32 && !SLJIT_CONFIG_MIPS */

#if !(defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64) \
	&& !(defined SLJIT_CONFIG_PPC && SLJIT_CONFIG_PPC)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fmem_update(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 freg,
	sljit_s32 mem, sljit_sw memw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_fmem_update(compiler, type, freg, mem, memw));
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(freg);
	SLJIT_UNUSED_ARG(mem);
	SLJIT_UNUSED_ARG(memw);

	return SLJIT_ERR_UNSUPPORTED;
}

#endif /* !SLJIT_CONFIG_ARM_64 && !SLJIT_CONFIG_PPC */

#if !(defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86) \
	&& !(defined SLJIT_CONFIG_ARM && SLJIT_CONFIG_ARM) \
	&& !(defined SLJIT_CONFIG_S390X && SLJIT_CONFIG_S390X) \
	&& !(defined SLJIT_CONFIG_RISCV && SLJIT_CONFIG_RISCV) \
	&& !(defined SLJIT_CONFIG_LOONGARCH && SLJIT_CONFIG_LOONGARCH)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_mov(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg,
	sljit_s32 srcdst, sljit_sw srcdstw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_mov(compiler, type, vreg, srcdst, srcdstw));
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(vreg);
	SLJIT_UNUSED_ARG(srcdst);
	SLJIT_UNUSED_ARG(srcdstw);

	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_replicate(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg,
	sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_replicate(compiler, type, vreg, src, srcw));
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(vreg);
	SLJIT_UNUSED_ARG(src);
	SLJIT_UNUSED_ARG(srcw);

	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_lane_mov(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg, sljit_s32 lane_index,
	sljit_s32 srcdst, sljit_sw srcdstw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_lane_mov(compiler, type, vreg, lane_index, srcdst, srcdstw));
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(vreg);
	SLJIT_UNUSED_ARG(lane_index);
	SLJIT_UNUSED_ARG(srcdst);
	SLJIT_UNUSED_ARG(srcdstw);

	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_lane_replicate(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg,
	sljit_s32 src, sljit_s32 src_lane_index)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_lane_replicate(compiler, type, vreg, src, src_lane_index));
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(vreg);
	SLJIT_UNUSED_ARG(src);
	SLJIT_UNUSED_ARG(src_lane_index);

	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_extend(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg,
	sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_extend(compiler, type, vreg, src, srcw));
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(vreg);
	SLJIT_UNUSED_ARG(src);
	SLJIT_UNUSED_ARG(srcw);

	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_sign(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 vreg,
	sljit_s32 dst, sljit_sw dstw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_sign(compiler, type, vreg, dst, dstw));
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(vreg);
	SLJIT_UNUSED_ARG(dst);
	SLJIT_UNUSED_ARG(dstw);

	return SLJIT_ERR_UNSUPPORTED;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_simd_op2(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_vreg, sljit_s32 src1_vreg, sljit_s32 src2, sljit_sw src2w)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_simd_op2(compiler, type, dst_vreg, src1_vreg, src2, src2w));
	SLJIT_UNUSED_ARG(compiler);
	SLJIT_UNUSED_ARG(type);
	SLJIT_UNUSED_ARG(dst_vreg);
	SLJIT_UNUSED_ARG(src1_vreg);
	SLJIT_UNUSED_ARG(src2);
	SLJIT_UNUSED_ARG(src2w);

	return SLJIT_ERR_UNSUPPORTED;
}

#endif /* !SLJIT_CONFIG_X86 && !SLJIT_CONFIG_ARM && !SLJIT_CONFIG_S390X && !SLJIT_CONFIG_LOONGARCH */

#if !(defined SLJIT_CONFIG_X86 && SLJIT_CONFIG_X86) \
	&& !(defined SLJIT_CONFIG_ARM_64 && SLJIT_CONFIG_ARM_64)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_local_base(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw offset)
{
	CHECK_ERROR();
	CHECK(check_sljit_get_local_base(compiler, dst, dstw, offset));

	ADJUST_LOCAL_OFFSET(SLJIT_MEM1(SLJIT_SP), offset);

	SLJIT_SKIP_CHECKS(compiler);

	if (offset != 0)
		return sljit_emit_op2(compiler, SLJIT_ADD, dst, dstw, SLJIT_SP, 0, SLJIT_IMM, offset);
	return sljit_emit_op1(compiler, SLJIT_MOV, dst, dstw, SLJIT_SP, 0);
}

#endif /* !SLJIT_CONFIG_X86 && !SLJIT_CONFIG_ARM_64 */

#endif /* !SLJIT_CONFIG_UNSUPPORTED */
